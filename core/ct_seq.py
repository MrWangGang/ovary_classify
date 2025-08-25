import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import cv2
from sklearn.metrics import f1_score, recall_score, jaccard_score
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- 配置 ---
class Config:
    NAME = "ct" # 数据集/模态前缀
    TRAIN_ROOT_DIR = f'./datasets/{NAME}_train'
    TEST_ROOT_DIR = f'./datasets/{NAME}_test' # 验证集根目录

    # 报告相关的路径配置，调整为 report/seq/{DATA_PREFIX}/
    REPORT_BASE_DIR = f'./report/seq/{NAME}'
    MODEL_SAVE_DIR = os.path.join(REPORT_BASE_DIR, 'models')
    MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, f'best_{NAME}_model.pth')

    METRICS_LOG_DIR = os.path.join(REPORT_BASE_DIR, 'logs')
    METRICS_JSON_FILENAME = 'training_metrics_log.json'
    METRICS_JSON_PATH = os.path.join(METRICS_LOG_DIR, METRICS_JSON_FILENAME)

    PLOTS_SAVE_DIR = os.path.join(REPORT_BASE_DIR, 'plots')

    # --- 预测图片相关的配置 ---
    # 不再需要手动指定 PREDICT_IMAGE_PATH，程序将自动从测试集获取第一张图片
    PREDICT_SAVE_DIR = os.path.join(REPORT_BASE_DIR, 'predictions')
    # --------------------------------

    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    ENCODER = 'resnet34'
    ENCODER_WEIGHTS = 'imagenet'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()

# --- 数据集类 ---
class CTDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx] # 修正：这里应该是 self.mask_paths

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # OpenCV 默认是 BGR，转换为 RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 将掩码二值化 (肿瘤为1，背景为0)
        mask = (mask > 0).astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # 为掩码添加通道维度，适应模型输入 (例如，UNet通常期望 CxHxW)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        return image, mask

# --- 数据加载函数 ---
def load_data_paths(root_dir):
    """加载指定目录下所有 BT 和 MT 图像和掩码的路径，并合并。"""
    image_paths_bt = sorted(glob.glob(os.path.join(root_dir, 'bt', 'images', '*.jpg')))
    mask_paths_bt = sorted(glob.glob(os.path.join(root_dir, 'bt', 'masks', '*.png')))
    image_paths_mt = sorted(glob.glob(os.path.join(root_dir, 'mt', 'images', '*.jpg')))
    mask_paths_mt = sorted(glob.glob(os.path.join(root_dir, 'mt', 'masks', '*.png')))

    all_image_paths = image_paths_bt + image_paths_mt
    all_mask_paths = mask_paths_bt + mask_paths_mt

    print(f"在 {root_dir} 中找到 {len(all_image_paths)} 张图像。")
    return all_image_paths, all_mask_paths

# --- 定义数据增强 ---
train_transform = A.Compose([
    A.Resize(config.IMG_HEIGHT, config.IMG_WIDTH),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=45, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # ImageNet 统计值
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(config.IMG_HEIGHT, config.IMG_WIDTH),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# --- 评估函数 ---
def evaluate_model(model, dataloader, device, loss_fn):
    model.eval()
    total_val_loss = 0.0
    all_preds_tensors = []
    all_masks_tensors = []

    if len(dataloader) == 0:
        return 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)
            total_val_loss += loss.item()

            preds = (outputs > 0.5).float()

            all_preds_tensors.append(preds.cpu())
            all_masks_tensors.append(masks.cpu())

    avg_val_loss = total_val_loss / len(dataloader)

    if not all_preds_tensors:
        return 0.0, 0.0, 0.0, avg_val_loss

    all_preds = torch.cat(all_preds_tensors).numpy().flatten()
    all_masks = torch.cat(all_masks_tensors).numpy().flatten()

    if np.sum(all_masks) == 0 and np.sum(all_preds) == 0:
        f1, recall, iou = 1.0, 1.0, 1.0
    elif np.sum(all_masks) == 0 or np.sum(all_preds) == 0:
        f1, recall, iou = 0.0, 0.0, 0.0
    else:
        f1 = f1_score(all_masks, all_preds)
        recall = recall_score(all_masks, all_preds)
        iou = jaccard_score(all_masks, all_preds)

    return f1, recall, iou, avg_val_loss

# --- 绘制指标曲线图函数 ---
def plot_metrics(metrics_data, plots_save_dir):
    epochs = [entry['epoch'] for entry in metrics_data]
    train_losses = [entry['train_loss'] for entry in metrics_data]
    val_losses = [entry['val_loss'] for entry in metrics_data]
    val_ious = [entry['val_iou'] for entry in metrics_data]
    val_f1s = [entry['val_f1'] for entry in metrics_data]

    os.makedirs(plots_save_dir, exist_ok=True)

    # 图1: 训练损失和验证损失曲线图
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='red')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_save_dir, 'loss_curves.png'))
    plt.close()

    # 图2: 验证 IoU 曲线图
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_ious, label='Validation IoU', color='green')
    plt.title('Validation IoU over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_save_dir, 'val_iou_curve.png'))
    plt.close()

    # 图3: 验证 F1 曲线图
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_f1s, label='Validation F1-Score', color='purple')
    plt.title('Validation F1-Score over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_save_dir, 'val_f1_curve.png'))
    plt.close()

    print(f"指标曲线图已保存到: {plots_save_dir}")

# --- 预测和可视化函数 ---
def predict_and_visualize(model, image_path, save_dir, img_height, img_width, device, transform):
    os.makedirs(save_dir, exist_ok=True)
    img_name = os.path.basename(image_path)
    base_name, _ = os.path.splitext(img_name)

    # 1. 加载原图
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # 检查图片是否成功加载
    if original_image is None:
        print(f"错误: 无法加载图片或图片不存在: {image_path}. 请检查路径和文件完整性。")
        return

    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # 尝试找到对应的真实掩膜
    mask_path = image_path.replace('images', 'masks').replace('.jpg', '.png')
    true_mask_display = None
    if os.path.exists(mask_path):
        true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if true_mask is not None:
            # 确保真实掩膜也是原图尺寸，并二值化为0/255用于显示
            true_mask_display = cv2.resize((true_mask > 0).astype(np.uint8) * 255,
                                           (original_image.shape[1], original_image.shape[0]),
                                           interpolation=cv2.INTER_NEAREST)
        else:
            print(f"警告: 找到真实掩膜文件但无法读取: {mask_path}")
    else:
        print(f"未找到对应真实掩膜: {mask_path}")
        # 如果没有真实掩膜，可以创建一个全黑的图像来占位
        true_mask_display = np.zeros_like(original_image_rgb[:,:,0])

    # 2. 对原图进行预处理并用模型进行预测
    input_image = original_image.copy()
    augmented_image = transform(image=input_image)['image']
    input_tensor = augmented_image.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    predicted_mask_tensor = (output > 0.5).float()
    predicted_mask_np = predicted_mask_tensor.squeeze(0).squeeze(0).cpu().numpy()

    # 将预测掩膜 resize 回原图尺寸
    predicted_mask_resized = cv2.resize(predicted_mask_np,
                                        (original_image.shape[1], original_image.shape[0]),
                                        interpolation=cv2.INTER_LINEAR)
    predicted_mask_display = (predicted_mask_resized * 255).astype(np.uint8)

    # 3. 将预测掩膜用红色叠加回原图
    overlay_image = original_image_rgb.copy()
    red_color = np.array([255, 0, 0], dtype=np.uint8)

    colored_mask = np.zeros_like(overlay_image)
    colored_mask[predicted_mask_resized > 0.5] = red_color

    alpha = 0.5 # 透明度
    blended_image = cv2.addWeighted(overlay_image, 1 - alpha, colored_mask, alpha, 0)

    # --- 创建一行四列的图像 ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 6)) # 调整 figsize 以适应一行四列

    # 宫格1: 原图
    axes[0].imshow(original_image_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # 宫格2: 真实掩膜
    axes[1].imshow(true_mask_display, cmap='gray') # 掩膜用灰度图显示
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')

    # 宫格3: 模型预测的掩膜
    axes[2].imshow(predicted_mask_display, cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')

    # 宫格4: 叠加预测的图像
    axes[3].imshow(blended_image)
    axes[3].set_title('Overlayed Prediction')
    axes[3].axis('off')

    plt.tight_layout() # 自动调整子图参数，使之填充整个图像区域

    output_path = os.path.join(save_dir, f'{base_name}_single_row_prediction.jpg') # 修改保存文件名
    plt.savefig(output_path, dpi=300) # 提高 dpi 增加图片清晰度
    plt.close(fig) # 关闭图形，释放内存

    print(f"单行四列预测图已保存到: {output_path}")


# --- 主训练流程 ---
if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(config.METRICS_LOG_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_SAVE_DIR, exist_ok=True)
    os.makedirs(config.PREDICT_SAVE_DIR, exist_ok=True) # 确保预测结果保存目录也存在

    # 加载训练数据 (所有肿瘤类型合并)
    train_image_paths, train_mask_paths = load_data_paths(config.TRAIN_ROOT_DIR)

    # 加载验证数据 (所有肿瘤类型合并，只为整体评估)
    val_image_paths, val_mask_paths = load_data_paths(config.TEST_ROOT_DIR)

    print("数据泄露检查：训练集和验证集来自不同的目录。")

    # 创建数据集和数据加载器
    train_dataset = CTDataset(train_image_paths, train_mask_paths, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = CTDataset(val_image_paths, val_mask_paths, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 模型、损失函数和优化器
    model = smp.Unet(
        encoder_name=config.ENCODER,
        encoder_weights=config.ENCODER_WEIGHTS,
        in_channels=3,
        classes=1, # 二值分割，肿瘤 vs. 背景
        activation='sigmoid'
    ).to(config.DEVICE)

    loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_iou = -1.0 # 记录最佳 IoU
    metrics_history = [] # 用于保存每个 epoch 的指标数据

    print(f"开始在 {config.DEVICE} 上训练...")
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} 训练")
        for images, masks in train_bar:
            images = images.to(config.DEVICE)
            masks = masks.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)

        # --- 验证评估 ---
        model.eval()
        # 从 evaluate_model 函数中获取所有指标，包括平均验证损失
        val_f1, val_recall, val_iou, avg_val_loss = evaluate_model(model, val_loader, config.DEVICE, loss_fn)

        print(f"\n--- Epoch {epoch+1}/{config.NUM_EPOCHS} ---")
        print(f"平均训练损失: {avg_train_loss:.4f}")
        print(f"平均验证损失: {avg_val_loss:.4f}") # 打印平均验证损失
        print(f"验证指标: F1: {val_f1:.4f} | Recall: {val_recall:.4f} | IoU: {val_iou:.4f}")

        # 记录当前 epoch 的指标
        metrics_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss, # 使用计算出的平均验证损失
            'val_f1': val_f1,
            'val_recall': val_recall,
            'val_iou': val_iou
        })

        # 保存最佳模型 (基于整体 IoU)
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"  --> 已保存最佳模型，验证 IoU: {best_iou:.4f}")

    print("\n训练完成!")
    print(f"最佳模型保存到: {config.MODEL_SAVE_PATH}")

    # 保存指标历史到 JSON 文件
    with open(config.METRICS_JSON_PATH, 'w') as f:
        json.dump(metrics_history, f, indent=4)
    print(f"训练指标已保存到: {config.METRICS_JSON_PATH}")

    # 绘制并保存曲线图
    plot_metrics(metrics_history, config.PLOTS_SAVE_DIR)

    # --- 训练结束后，使用最佳模型进行预测和可视化 ---
    print("\n--- 开始使用最佳模型进行预测和可视化 ---")
    # 重新加载最佳模型
    best_model = smp.Unet(
        encoder_name=config.ENCODER,
        encoder_weights=None, # 不加载预训练权重，因为我们加载的是自己的训练权重
        in_channels=3,
        classes=1,
        activation='sigmoid'
    ).to(config.DEVICE)

    # 确保模型权重文件存在
    if os.path.exists(config.MODEL_SAVE_PATH):
        best_model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
        print(f"已加载最佳模型：{config.MODEL_SAVE_PATH}")
    else:
        print(f"错误: 未找到最佳模型文件: {config.MODEL_SAVE_PATH}. 无法进行预测可视化。")
        exit()

    # 定义用于预测的图片转换 (只包含 Resize 和 Normalize)
    predict_transform = A.Compose([
        A.Resize(config.IMG_HEIGHT, config.IMG_WIDTH),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # --- 自动获取第一张测试图片路径 ---
    first_test_image_path = None
    all_test_image_paths, _ = load_data_paths(config.TEST_ROOT_DIR)

    if all_test_image_paths:
        first_test_image_path = all_test_image_paths[0]
        print(f"将使用测试集中的第一张图片进行预测: {first_test_image_path}")
        predict_and_visualize(
            model=best_model,
            image_path=first_test_image_path,
            save_dir=config.PREDICT_SAVE_DIR,
            img_height=config.IMG_HEIGHT,
            img_width=config.IMG_WIDTH,
            device=config.DEVICE,
            transform=predict_transform
        )
    else:
        print("错误: 未能在测试集中找到任何图片进行预测。请确保测试集路径正确且包含图片。")

    print("预测和可视化完成!")