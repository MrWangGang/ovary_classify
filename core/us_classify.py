import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import seaborn as sns
from PIL import Image
import random
import cv2
from torch.utils.data import Dataset # 导入 Dataset 基类
import matplotlib.font_manager as fm # 导入 font_manager 用于加载字体
import os
import platform # 导入 platform 模块来判断操作系统

# --- Matplotlib 跨平台中文字体配置 ---
def set_matplotlib_chinese_font():
    """
    配置 Matplotlib 使用当前操作系统的中文字体。
    尝试根据操作系统检测并设置常见的中文无衬线字体。
    """
    system_platform = platform.system()
    found_font = None

    # 列出常见的中文无衬线字体名称
    # 按照优先级排列，通常优先使用系统自带的，其次是常用且广泛支持的
    common_chinese_fonts = {
        'Windows': ['Microsoft YaHei', 'SimHei', 'FangSong', 'KaiTi'],
        'Darwin': ['PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'STHeiti'], # macOS
        'Linux': ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'Source Han Sans SC', 'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'SimHei', 'Arial Unicode MS']
    }

    # 获取所有可用的字体
    font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    available_font_names = {fm.FontProperties(fname=f).get_name() for f in font_paths}

    print(f"当前操作系统: {system_platform}")
    print("尝试查找并设置中文字体...")

    # 根据操作系统优先级查找字体
    fonts_to_try = common_chinese_fonts.get(system_platform, [])
    for font_name_candidate in fonts_to_try:
        if font_name_candidate in available_font_names:
            found_font = font_name_candidate
            break

    # 如果特定系统没有找到，尝试通用列表
    if not found_font:
        print("未找到系统推荐的字体，尝试通用中文字体列表。")
        generic_chinese_fonts = ['Microsoft YaHei', 'PingFang SC', 'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'Arial Unicode MS']
        for font_name_candidate in generic_chinese_fonts:
            if font_name_candidate in available_font_names:
                found_font = font_name_candidate
                break

    if found_font:
        plt.rcParams['font.sans-serif'] = [found_font, 'Arial Unicode MS', 'DejaVu Sans'] # 将找到的字体放在首位，并加入备用字体
        plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
        print(f"Matplotlib 已成功配置使用字体: '{found_font}'")
    else:
        print("警告: 未能在系统中找到合适的中文显示字体。中文字符可能无法正常显示。")
        print("当前可用的字体数量:", len(available_font_names))
        # 可以选择退回默认或打印更多信息
        # plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] # 退回默认英文
        # plt.rcParams['axes.unicode_minus'] = True # 退回默认英文负号

    # 强制重建 Matplotlib 字体缓存（可选，但在遇到问题时强烈推荐）
    # 这个方法在某些Matplotlib版本中是私有方法，但通常有效
    # 如果您频繁遇到字体问题，可以取消注释下面一行，或者手动删除缓存目录
    # try:
    #     fm._rebuild()
    #     print("Matplotlib 字体缓存已重建。")
    # except AttributeError:
    #     print("警告: 无法直接调用 fm._rebuild()。若字体问题持续，请手动清理缓存。")
    #     print(f"字体缓存目录通常在: {fm.get_cachedir()}")


# 调用字体配置函数
set_matplotlib_chinese_font()
# --- 全局配置 ---
DATA_PREFIX = 'us' # 数据集/模态前缀，用于定义文件路径
# --- 全局配置结束 ---


## 路径与超参数配置

DATA_DIR = './datasets' # 数据集根目录
TRAIN_DIR = os.path.join(DATA_DIR, f'{DATA_PREFIX}_train') # 训练集目录
TEST_DIR = os.path.join(DATA_DIR, f'{DATA_PREFIX}_test') # 测试集（验证集）目录

MODEL_SAVE_DIR = f'./report/classify/{DATA_PREFIX}/models' # 模型保存目录
BEST_MODEL_FILENAME = 'best_resnet18_classification_model.pth' # 最佳模型文件名
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, BEST_MODEL_FILENAME) # 最佳模型完整保存路径

METRICS_LOG_DIR = f'./report/classify/{DATA_PREFIX}/logs' # 指标日志保存目录
METRICS_JSON_FILENAME = 'training_metrics_log.json' # 指标日志文件名
METRICS_JSON_PATH = os.path.join(METRICS_LOG_DIR, METRICS_JSON_FILENAME) # 指标日志完整保存路径

PLOTS_SAVE_DIR = f'./report/classify/{DATA_PREFIX}/plots' # 绘图保存目录
CONFUSION_MATRIX_PATH = os.path.join(PLOTS_SAVE_DIR, 'confusion_matrix.png') # 混淆矩阵图路径
HEATMAP_VISUALIZATION_PATH = os.path.join(PLOTS_SAVE_DIR, 'heatmap_visualizations.png') # 热力图可视化路径
LOSS_TREND_PATH = os.path.join(PLOTS_SAVE_DIR, 'loss_trend.png') # 新增损失趋势图路径
AUGMENTATION_VISUALIZATION_PATH = os.path.join(PLOTS_SAVE_DIR, 'augmentation_visualizations.png') # 数据增强可视化路径

TREND_PLOT_PATHS = { # 各项指标趋势图的保存路径
    'roc_auc': os.path.join(PLOTS_SAVE_DIR, 'roc_auc_trend.png'),
    'pr_auc': os.path.join(PLOTS_SAVE_DIR, 'pr_auc_trend.png'),
    'acc': os.path.join(PLOTS_SAVE_DIR, 'accuracy_trend.png'),
    'specificity': os.path.join(PLOTS_SAVE_DIR, 'specificity_trend.png'),
    'sensitivity': os.path.join(PLOTS_SAVE_DIR, 'sensitivity_trend.png'),
    'ppv': os.path.join(PLOTS_SAVE_DIR, 'ppv_trend.png'),
    'npv': os.path.join(PLOTS_SAVE_DIR, 'npv_trend.png'),
    'f1': os.path.join(PLOTS_SAVE_DIR, 'f1_trend.png'),
}

NUM_CLASSES = 2 # 分类类别数量 (例如：二分类)
BATCH_SIZE = 32 # 批处理大小
LEARNING_RATE = 0.0001 # 学习率
NUM_EPOCHS = 100 # 训练的总轮数
IMAGE_SIZE = 224 # 图像大小 (H x W)

# 创建必要的目录（如果不存在）
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(METRICS_LOG_DIR, exist_ok=True)
os.makedirs(PLOTS_SAVE_DIR, exist_ok=True)


## 图像变换

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE), # 随机裁剪并缩放
        transforms.RandomRotation(15), # 随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # 随机颜色抖动
        transforms.RandomGrayscale(p=0.1), # 随机灰度化
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)), # 随机高斯模糊
        transforms.ToTensor(), # 将PIL图像或numpy数组转换为PyTorch Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 归一化 (ImageNet 统计值)
    ]),
    'test': transforms.Compose([
        transforms.Resize(IMAGE_SIZE + 32), # 缩放图像，使其较短边达到指定大小
        transforms.CenterCrop(IMAGE_SIZE), # 从中心裁剪图像
        transforms.ToTensor(), # 将PIL图像或numpy数组转换为PyTorch Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 归一化
    ]),
    'raw': transforms.Compose([ # 用于可视化原始图像，不进行归一化，但进行缩放和中心裁剪以保持一致
        transforms.Resize(IMAGE_SIZE + 32),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor()
    ])
}


## 自定义数据集类

class CustomImageDataset(Dataset):
    """
    自定义数据集类，用于加载特定子文件夹 (例如 'images') 中的图片。
    期望的目录结构:
    root_dir/
    ├── class_A/
    │   ├── images/
    │   │   ├── img1.png
    │   │   └── ...
    │   └── masks/ (会被忽略)
    ├── class_B/
    │   ├── images/
    │   │   ├── img_x.png
    │   │   └── ...
    │   └── masks/ (会被忽略)
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [] # 存储所有图片的完整路径
        self.labels = []      # 存储对应图片的数字标签
        self.classes = []     # 存储类别名称 (例如 ['bt', 'mt'])
        self.class_to_idx = {} # 类别名称到数字索引的映射

        # 遍历 root_dir 下的所有一级子文件夹 (例如 'bt', 'mt')
        class_folders = sorted([d.name for d in os.scandir(root_dir) if d.is_dir()])
        self.classes = class_folders
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # 收集所有 'images' 子文件夹下的图片路径和标签
        for class_name in class_folders:
            class_path = os.path.join(root_dir, class_name)
            images_folder_path = os.path.join(class_path, 'images') # 明确指向 'images' 子文件夹

            if not os.path.isdir(images_folder_path):
                print(f"警告: 类别 '{class_name}' 下没有 'images' 文件夹或路径错误: {images_folder_path}")
                continue # 跳过没有 'images' 文件夹的类别

            # 获取所有图片名称并进行排序，以确保选择的图片具有一致性
            img_names = sorted([img_name for img_name in os.listdir(images_folder_path)
                                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])

            for img_name in img_names:
                self.image_paths.append(os.path.join(images_folder_path, img_name))
                self.labels.append(self.class_to_idx[class_name])

        # 将标签转换为torch.LongTensor，方便后续使用
        self.labels = torch.tensor(self.labels, dtype=torch.long)


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB') # 打开图片并转换为RGB格式
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image) # 应用图像变换

        # 返回图像、标签和图像路径 (用于打印文件名)
        return image, label, img_path

# --- 使用自定义数据集类加载数据 ---
# 注意: CustomImageDataset.__getitem__ 现在返回 (image, label, img_path)
# DataLoader 不直接处理 img_path，但我们的可视化函数会直接调用 dataset[idx]
image_datasets = {
    'train': CustomImageDataset(TRAIN_DIR, data_transforms['train']),
    'test': CustomImageDataset(TEST_DIR, data_transforms['test'])
}

# 创建数据加载器
# 对于训练和测试阶段，我们只需要图片和标签，所以 DataLoader 的 collate_fn 默认会处理
# 如果需要从 DataLoader 中直接获取路径，需要自定义 collate_fn
dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
    'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
}

# 获取类别名称 (现在从自定义数据集获取)
class_names = image_datasets['train'].classes
print(f"检测到的类别: {class_names}")
print(f"类别到索引的映射: {image_datasets['train'].class_to_idx}")
print(f"训练集大小: {len(image_datasets['train'])}")
print(f"验证集大小: {len(image_datasets['test'])}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 设置训练设备 (GPU 或 CPU)
print(f"使用设备: {device}")


## 模型、损失函数和优化器

# 加载预训练的 ResNet-18 模型
model_ft = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model_ft.fc.in_features # 获取全连接层的输入特征数
model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES) # 修改全连接层以适应新的类别数量
model_ft = model_ft.to(device) # 将模型移动到指定设备

criterion = nn.CrossEntropyLoss() # 定义交叉熵损失函数
optimizer_ft = optim.Adam(model_ft.parameters(), lr=LEARNING_RATE) # 定义 Adam 优化器


## 训练模型

def train_model(model, criterion, optimizer, num_epochs=NUM_EPOCHS):
    best_acc = 0.0 # 记录最佳验证准确率
    metrics_log = [] # 存储每个epoch的指标

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        epoch_metrics = { # 初始化当前epoch的指标字典
            'epoch': epoch,
            'loss': {'train': 0.0, 'val': 0.0},
            'train_acc': 0.0, # 训练集准确率（仅用于日志，不用于趋势图）
            'test_acc': 0.0, # 测试集准确率（用于最佳模型保存）
            'roc_auc': {'total': 0.0},
            'pr_auc': {'total': 0.0},
            'acc': {'total': 0.0}, # 验证集准确率（用于趋势图）
            'specificity': {'total': 0.0},
            'sensitivity': {'total': 0.0},
            'ppv': {'total': 0.0},
            'npv': {'total': 0.0},
            'f1': {'total': 0.0}
        }
        # 初始化每个类别的具体指标（用于日志，不用于整体趋势图）
        for class_name in class_names:
            epoch_metrics['specificity'][class_name] = 0.0
            epoch_metrics['sensitivity'][class_name] = 0.0
            epoch_metrics['ppv'][class_name] = 0.0
            epoch_metrics['npv'][class_name] = 0.0
            epoch_metrics['f1'][class_name] = 0.0
            epoch_metrics['roc_auc'][class_name] = 0.0
            epoch_metrics['pr_auc'][class_name] = 0.0
            epoch_metrics['acc'][class_name] = 0.0


        for phase in ['train', 'test']: # 训练和测试（验证）阶段
            if phase == 'train':
                model.train() # 设置模型为训练模式
            else:
                model.eval() # 设置模型为评估模式

            running_loss = 0.0
            all_labels = [] # 存储所有真实标签
            all_preds = [] # 存储所有预测标签
            all_probs = []

            # 注意: DataLoader 默认会处理 CustomImageDataset 的 __getitem__ 返回的多个值
            # 如果是 (image, label, path)，DataLoader 会将它们分别打包成批次
            # 所以这里接收 inputs, labels, _ 来忽略 path
            for inputs, labels, _ in dataloaders[phase]: # 遍历数据加载器
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad() # 清零梯度

                with torch.set_grad_enabled(phase == 'train'): # 只在训练阶段计算梯度
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) # 获取预测类别
                    loss = criterion(outputs, labels) # 计算损失
                    probs = torch.softmax(outputs, dim=1) # 计算 Softmax 概率

                    if phase == 'train':
                        loss.backward() # 反向传播
                        optimizer.step() # 更新模型参数

                running_loss += loss.item() * inputs.size(0) # 累加损失
                all_labels.extend(labels.cpu().numpy()) # 收集真实标签
                all_preds.extend(preds.cpu().numpy()) # 收集预测标签
                all_probs.extend(probs.cpu().detach().numpy()) # 收集预测概率

            epoch_loss = running_loss / len(image_datasets[phase]) # 计算平均损失
            epoch_acc = accuracy_score(all_labels, all_preds) # 计算准确率

            if phase == 'train':
                epoch_metrics['loss']['train'] = epoch_loss
                epoch_metrics['train_acc'] = epoch_acc
            else: # phase == 'test' (验证阶段)
                epoch_metrics['loss']['val'] = epoch_loss
                epoch_metrics['test_acc'] = epoch_acc # 存储测试集准确率，用于判断是否保存最佳模型
                epoch_metrics['acc']['total'] = epoch_acc # 这个值将用于绘制准确率趋势图

                # 计算并记录整体指标
                # 确保 all_labels 包含至少两个不同的类别，否则 roc_auc_score 会报错
                if len(np.unique(all_labels)) == 2:
                    epoch_metrics['roc_auc']['total'] = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
                    precision, recall, _ = precision_recall_curve(all_labels, np.array(all_probs)[:, 1])
                    epoch_metrics['pr_auc']['total'] = auc(recall, precision)
                else: # 如果只有一个类别，ROC AUC 和 PR AUC 无意义
                    epoch_metrics['roc_auc']['total'] = np.nan
                    epoch_metrics['pr_auc']['total'] = np.nan

                epoch_metrics['f1']['total'] = f1_score(all_labels, all_preds, average='binary') # F1分数

                # 对于二分类 (NUM_CLASSES=2)，灵敏度、特异性、PPV、NPV 通常针对其中一个类别（例如，阳性类）计算
                # 假设类别1是阳性类
                if NUM_CLASSES == 2:
                    # 检查混淆矩阵是否可以计算 (即，真实标签中是否包含两个类别)
                    if len(np.unique(all_labels)) == 2:
                        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
                    else: # 如果标签中只有一个类别，假设计数为0
                        tn, fp, fn, tp = (0, 0, 0, 0)

                    epoch_metrics['sensitivity']['total'] = tp / (tp + fn) if (tp + fn) > 0 else 0 # 灵敏度 (真阳性率)
                    epoch_metrics['specificity']['total'] = tn / (tn + fp) if (tn + fp) > 0 else 0 # 特异性 (真阴性率)
                    epoch_metrics['ppv']['total'] = tp / (tp + fp) if (tp + fp) > 0 else 0 # 阳性预测值
                    epoch_metrics['npv']['total'] = tn / (tn + fn) if (tn + fn) > 0 else 0 # 阴性预测值
                else: # 对于多分类，这些指标通常是平均值或按类别计算
                    epoch_metrics['sensitivity']['total'] = recall_score(all_labels, all_preds, average='weighted')
                    epoch_metrics['precision']['total'] = precision_score(all_labels, all_preds, average='weighted')
                    epoch_metrics['specificity']['total'] = 0.0 # 多分类中没有直接的特异性
                    epoch_metrics['npv']['total'] = 0.0 # 多分类中没有直接的 NPV

                # 仍然计算每个类别的具体指标，尽管它们不用于整体趋势图
                for i, class_name in enumerate(class_names):
                    binary_true = (np.array(all_labels) == i).astype(int)
                    binary_pred = (np.array(all_preds) == i).astype(int)

                    # 确保 binary_true 中包含 0 和 1，以便计算准确的混淆矩阵和相关指标
                    if len(np.unique(binary_true)) == 2:
                        cm = confusion_matrix(binary_true, binary_pred, labels=[0, 1])
                        tn, fp, fn, tp = cm.ravel()
                    else: # 如果批次中只包含一个类别
                        if i in all_labels: # 如果当前类别 'i' 存在于真实标签中
                            tp = np.sum((np.array(all_labels) == i) & (np.array(all_preds) == i))
                            fn = np.sum((np.array(all_labels) == i) & (np.array(all_preds) != i))
                            fp = np.sum((np.array(all_labels) != i) & (np.array(all_preds) == i))
                            tn = np.sum((np.array(all_labels) != i) & (np.array(all_preds) != i))
                        else: # 如果当前类别 'i' 不存在于真实标签中
                            tn, fp, fn, tp = (0, 0, 0, 0)

                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                    f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0

                    epoch_metrics['specificity'][class_name] = specificity
                    epoch_metrics['sensitivity'][class_name] = sensitivity
                    epoch_metrics['ppv'][class_name] = ppv
                    epoch_metrics['npv'][class_name] = npv
                    epoch_metrics['f1'][class_name] = f1
                    # 确保 ROC AUC 和 PR AUC 只有在真实标签中包含两种类别时才计算
                    if len(np.unique(binary_true)) == 2:
                        epoch_metrics['roc_auc'][class_name] = roc_auc_score(binary_true, np.array(all_probs)[:, i])
                        prec, rec, _ = precision_recall_curve(binary_true, np.array(all_probs)[:, i])
                        epoch_metrics['pr_auc'][class_name] = auc(rec, prec)
                    else: # 如果批次中只有单一类别，AUC 无意义
                        epoch_metrics['roc_auc'][class_name] = np.nan
                        epoch_metrics['pr_auc'][class_name] = np.nan
                    epoch_metrics['acc'][class_name] = accuracy_score(binary_true, binary_pred)


            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 如果当前验证准确率优于历史最佳，则保存模型
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"在 Epoch {epoch} 保存了更好的模型，测试准确率: {best_acc:.4f}")

        metrics_log.append(epoch_metrics) # 将当前epoch的指标添加到日志中

        # 每个epoch结束后保存指标日志
        with open(METRICS_JSON_PATH, 'w') as f:
            json.dump(metrics_log, f, indent=4)

    print('训练完成')
    return model, metrics_log

def evaluate_model(model, dataloader, criterion):
    """
    在测试集上评估模型性能。
    """
    model.eval() # 设置模型为评估模式
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad(): # 在评估模式下，不计算梯度
        for inputs, labels, _ in dataloader: # 同样忽略路径
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    test_loss = running_loss / len(dataloader.dataset) # 计算最终测试损失
    test_acc = accuracy_score(all_labels, all_preds) # 计算最终测试准确率

    print(f'最终测试损失: {test_loss:.4f} 准确率: {test_acc:.4f}')

    return all_labels, all_preds, all_probs

def plot_metrics(metrics_log, trend_plot_paths):
    """
    绘制训练和验证损失趋势图以及其他评估指标的趋势图。
    """
    metric_names_english = { # 指标的英文名称映射
        'roc_auc': 'ROC AUC',
        'pr_auc': 'PR AUC',
        'acc': 'Accuracy',
        'specificity': 'Specificity',
        'sensitivity': 'Sensitivity',
        'ppv': 'Positive Predictive Value (PPV)',
        'npv': 'Negative Predictive Value (NPV)',
        'f1': 'F1 Score'
    }

    metrics_to_plot = ['roc_auc', 'pr_auc', 'acc', 'specificity', 'sensitivity', 'ppv', 'npv', 'f1'] # 需要绘制的指标
    epochs = [entry['epoch'] for entry in metrics_log] # epoch 列表

    # 绘制训练和验证损失趋势图
    plt.figure(figsize=(10, 6))
    train_losses = [entry['loss']['train'] for entry in metrics_log]
    val_losses = [entry['loss']['val'] for entry in metrics_log]
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='x')
    plt.title('Training and Validation Loss Trend over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOSS_TREND_PATH)
    plt.close()
    print(f"训练和验证损失图已保存到 {LOSS_TREND_PATH}")

    # 绘制其他指标的趋势图 (仅整体验证集指标)
    for metric_key in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        # 准确率趋势图只使用验证集数据
        if metric_key == 'acc':
            total_values = [entry['acc']['total'] for entry in metrics_log]
            plt.plot(epochs, total_values, label='Validation Accuracy', marker='o')
            plt.title('Model Validation Accuracy Trend over Epochs')
            plt.ylabel('Accuracy')
        else:
            total_values = [entry[metric_key]['total'] for entry in metrics_log]
            plt.plot(epochs, total_values, label=f'Overall Validation {metric_names_english[metric_key]}', marker='o')
            plt.title(f'Model Overall Validation {metric_names_english[metric_key]} Trend over Epochs')
            plt.ylabel(metric_names_english[metric_key])

        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        save_path = trend_plot_paths[metric_key]
        plt.savefig(save_path)
        plt.close()
    print(f"其他总体指标图已保存到 {PLOTS_SAVE_DIR}")


def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """
    绘制混淆矩阵。
    """
    cm = confusion_matrix(y_true, y_pred) # 计算混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes) # 使用 seaborn 绘制热力图
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()
    print(f"混淆矩阵已保存到 {save_path}")


# --- Grad-CAM 实现 ---
class GradCAM:
    """
    Grad-CAM 类，用于生成特征图热力图。
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer # 指定目标层
        self.gradients = None # 存储梯度
        self.activations = None # 存储激活

        # 注册钩子，在正向传播时保存激活，在反向传播时保存梯度
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """正向传播钩子，保存目标层的激活值。"""
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        """反向传播钩子，保存目标层的梯度。"""
        self.gradients = grad_output[0]

    def __call__(self, input_tensor, target_class=None):
        """
        生成 Grad-CAM 热力图。
        :param input_tensor: 输入图像张量。
        :param target_class: 目标类别索引。如果为 None，则使用模型预测的最高概率类别。
        :return: Grad-CAM 热力图 (numpy 数组)。
        """
        self.model.eval() # 设置模型为评估模式

        # 确保输入张量可计算梯度
        if not input_tensor.requires_grad:
            input_tensor.requires_grad_(True)

        output = self.model(input_tensor) # 正向传播

        if target_class is None:
            target_class = output.argmax(dim=1).item() # 获取预测的最高概率类别

        self.model.zero_grad() # 清零模型梯度

        # 为目标类别创建 One-Hot 编码，并进行反向传播
        one_hot_output = torch.zeros_like(output)
        one_hot_output[0][target_class] = 1
        output.backward(gradient=one_hot_output, retain_graph=True) # 反向传播，保留图以便后续操作

        # 计算权重 (梯度的平均值)
        weights = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.squeeze(0) # 移除批次维度

        # 计算 CAM (Class Activation Map)
        cam = torch.zeros(activations.shape[1], activations.shape[2], dtype=activations.dtype, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam) # 应用 ReLU

        # 归一化 CAM 到 [0, 1] 范围
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        else: # 避免除以零
            cam = torch.zeros_like(cam)

        return cam.detach().cpu().numpy() # 返回 numpy 格式的热力图


def denormalize_image(img_tensor):
    """
    对图像张量进行反归一化，以便正确显示。
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.cpu().numpy().transpose((1, 2, 0)) # 将 (C, H, W) 转换为 (H, W, C)
    img = img * std + mean # 反归一化
    img = np.clip(img, 0, 1) # 将像素值裁剪到 [0, 1] 范围
    return img

def visualize_gradcam_and_original(original_img_tensor, heatmap, ax_original, ax_heatmap, class_name, pred_label_name, filename):
    """
    在一个子图中显示原始图像，在另一个子图中显示叠加了 Grad-CAM 热力图的图像。
    """
    # 原始图像
    img_original = denormalize_image(original_img_tensor)
    ax_original.imshow(img_original)
    ax_original.set_title(f'Original\nTrue: {class_name}\nFile: {os.path.basename(filename)}', fontsize=7) # 显示文件名
    ax_original.axis('off')

    # 热力图叠加图像
    img_heatmap_overlay = denormalize_image(original_img_tensor) # 使用反归一化后的原始图像作为背景
    h, w, _ = img_heatmap_overlay.shape
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR) # 调整热力图大小

    ax_heatmap.imshow(img_heatmap_overlay)
    ax_heatmap.imshow(heatmap_resized, cmap='jet', alpha=0.5) # 叠加热力图
    ax_heatmap.set_title(f'Heatmap\nPredicted: {pred_label_name}', fontsize=7)
    ax_heatmap.axis('off')


# 新增：统一的图片选择函数
def select_fixed_images(dataset, indices_to_plot=[0, 4, 9, 14, 19]):
    """
    选择固定索引的图片，确保数据增强图和热力图使用相同的原图
    """
    # 获取 'bt' 和 'mt' 类别对应的索引
    bt_indices_all = [i for i, label in enumerate(dataset.labels) if dataset.classes[label] == 'bt']
    mt_indices_all = [i for i, label in enumerate(dataset.labels) if dataset.classes[label] == 'mt']

    # 按要求排序（这里假设数据集已经在初始化时排序过）
    # 从每个类别中选择指定索引的图片
    selected_bt_indices = [bt_indices_all[i] for i in indices_to_plot if i < len(bt_indices_all)]
    selected_mt_indices = [mt_indices_all[i] for i in indices_to_plot if i < len(mt_indices_all)]

    # 收集所有选中的图片信息
    selected_images = []
    for idx in selected_bt_indices:
        img_tensor, label, img_path = dataset[idx]
        selected_images.append({
            'idx': idx,
            'img_tensor': img_tensor,
            'label': label,
            'class_name': dataset.classes[label],
            'img_path': img_path
        })

    for idx in selected_mt_indices:
        img_tensor, label, img_path = dataset[idx]
        selected_images.append({
            'idx': idx,
            'img_tensor': img_tensor,
            'label': label,
            'class_name': dataset.classes[label],
            'img_path': img_path
        })

    return selected_images


def visualize_predictions(model, test_dataset, class_names, indices_to_plot=[0, 4, 9, 14, 19], save_path=HEATMAP_VISUALIZATION_PATH):
    """
    可视化模型预测结果，并显示 Grad-CAM 热力图及其对应的原始图像。
    此函数会利用自定义数据集类 (CustomImageDataset) 已经加载的图片路径。
    每类显示 indices_to_plot 中指定的索引对应的图片。
    """
    model.eval() # 设置模型为评估模式

    # 选择 Grad-CAM 的目标层，通常是最后一个卷积层
    target_layer = model.layer4[-1].conv2
    grad_cam = GradCAM(model, target_layer)

    # 使用统一的图片选择函数
    images_to_process = select_fixed_images(test_dataset, indices_to_plot)

    num_samples_to_plot = len(images_to_process)
    # 每张图片需要原始图和热力图两列
    num_cols = 2
    # 每行显示一对（原始+热力图），总行数等于要显示的样本数
    num_rows = num_samples_to_plot

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 3))
    # 如果只有一行，axes 可能不是二维数组，需要展平
    if num_rows == 1:
        axes = axes.reshape(1, -1)

    axes = axes.flatten()

    for i, img_info in enumerate(images_to_process):
        img_tensor = img_info['img_tensor'].unsqueeze(0).to(device)
        true_class_name = img_info['class_name']
        true_label_idx = img_info['label']
        img_filename = img_info['img_path'] # 获取图片路径

        img_tensor_for_gradcam = img_tensor.clone().detach().requires_grad_(True)

        with torch.no_grad():
            outputs = model(img_tensor_for_gradcam)
            _, predicted = torch.max(outputs.data, 1)
            predicted_class_idx = predicted.item()
            predicted_class_name = class_names[predicted_class_idx]

        heatmap = grad_cam(img_tensor_for_gradcam, target_class=predicted_class_idx)

        # 原始图的子图索引
        ax_original_idx = i * 2
        # 热力图的子图索引
        ax_heatmap_idx = i * 2 + 1

        visualize_gradcam_and_original(
            img_info['img_tensor'], heatmap,
            axes[ax_original_idx], axes[ax_heatmap_idx],
            true_class_name, predicted_class_name, img_filename # 传递文件名
        )

    plt.suptitle("Grad-CAM Heatmap Visualization (Original vs. Heatmap)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"热图可视化已保存到 {save_path}")


def visualize_augmentations(dataset, class_names, indices_to_plot=[0, 4, 9, 14, 19], save_path=AUGMENTATION_VISUALIZATION_PATH):
    """
    可视化原始图像及其增强后的版本。
    为 'bt' 和 'mt' 类别各取 indices_to_plot 中指定的索引对应的图片，并进行对比展示。
    """
    # 创建一个用于原始图像的数据集，不进行增强，但进行缩放和中心裁剪以保持一致
    original_transform = data_transforms['raw']
    original_dataset = CustomImageDataset(dataset.root_dir, original_transform)

    # 使用统一的图片选择函数
    selected_images = select_fixed_images(dataset, indices_to_plot)

    # 按类别分组
    bt_images = [img for img in selected_images if img['class_name'] == 'bt']
    mt_images = [img for img in selected_images if img['class_name'] == 'mt']

    # 处理 'bt' 类别
    if bt_images:
        num_images_to_show = len(bt_images)
        fig_bt, axes_bt = plt.subplots(2, num_images_to_show, figsize=(num_images_to_show * 3, 6))
        # 如果只有一张图片，axes 可能不是二维数组，需要展平
        if num_images_to_show == 1:
            axes_bt = axes_bt.reshape(2, -1)
        axes_bt = axes_bt.flatten()

        for i, img_info in enumerate(bt_images):
            original_img_tensor = original_dataset[img_info['idx']][0]  # 获取原始图像
            augmented_img_tensor = img_info['img_tensor']  # 获取增强图像
            img_path = img_info['img_path']  # 获取图像路径

            original_img_display = original_img_tensor.permute(1, 2, 0).cpu().numpy()
            augmented_img_display = denormalize_image(augmented_img_tensor)

            # 绘制原始图 (第一行)
            axes_bt[i].imshow(original_img_display)
            axes_bt[i].set_title(f'Original BT\nFile: {os.path.basename(img_path)}', fontsize=7)
            axes_bt[i].axis('off')

            # 绘制增强图 (第二行)
            axes_bt[i + num_images_to_show].imshow(augmented_img_display)
            axes_bt[i + num_images_to_show].set_title(f'Augmented BT', fontsize=7)
            axes_bt[i + num_images_to_show].axis('off')

        plt.suptitle("Data Augmentation Visualization - BT Class", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path.replace('.png', '_bt.png'))
        plt.close(fig_bt)
        print(f"BT 类数据增强可视化已保存到 {save_path.replace('.png', '_bt.png')}")

    # 处理 'mt' 类别
    if mt_images:
        num_images_to_show = len(mt_images)
        fig_mt, axes_mt = plt.subplots(2, num_images_to_show, figsize=(num_images_to_show * 3, 6))
        if num_images_to_show == 1:
            axes_mt = axes_mt.reshape(2, -1)
        axes_mt = axes_mt.flatten()

        for i, img_info in enumerate(mt_images):
            original_img_tensor = original_dataset[img_info['idx']][0]  # 获取原始图像
            augmented_img_tensor = img_info['img_tensor']  # 获取增强图像
            img_path = img_info['img_path']  # 获取图像路径

            original_img_display = original_img_tensor.permute(1, 2, 0).cpu().numpy()
            augmented_img_display = denormalize_image(augmented_img_tensor)

            # 绘制原始图 (第一行)
            axes_mt[i].imshow(original_img_display)
            axes_mt[i].set_title(f'Original MT\nFile: {os.path.basename(img_path)}', fontsize=7)
            axes_mt[i].axis('off')

            # 绘制增强图 (第二行)
            axes_mt[i + num_images_to_show].imshow(augmented_img_display)
            axes_mt[i + num_images_to_show].set_title(f'Augmented MT', fontsize=7)
            axes_mt[i + num_images_to_show].axis('off')

        plt.suptitle("Data Augmentation Visualization - MT Class", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path.replace('.png', '_mt.png'))
        plt.close(fig_mt)
        print(f"MT 类数据增强可视化已保存到 {save_path.replace('.png', '_mt.png')}")

    print("\n所有数据增强可视化已完成。")


# --- 主程序流 ---
print("--- 开始训练模型 ---")
model_trained, metrics_log = train_model(model_ft, criterion, optimizer_ft, num_epochs=NUM_EPOCHS)

print("\n--- 开始生成指标趋势图 ---")
plot_metrics(metrics_log, TREND_PLOT_PATHS)

print("\n--- 加载最佳模型进行最终评估 ---")
best_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = best_model.fc.in_features
best_model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
best_model.load_state_dict(torch.load(MODEL_SAVE_PATH)) # 加载保存的最佳模型
best_model = best_model.to(device)

final_true_labels, final_predictions, final_probabilities = evaluate_model(best_model, dataloaders['test'], criterion)

print("\n--- 生成混淆矩阵 ---")
plot_confusion_matrix(final_true_labels, final_predictions, class_names, CONFUSION_MATRIX_PATH)

print("\n--- 生成热力图可视化 ---")
# 使用固定的索引：每个类别的前1、5、10、15、20张图片（索引0、4、9、14、19）
visualize_predictions(best_model, image_datasets['train'], class_names, indices_to_plot=[0, 4, 9, 14, 19])

print("\n--- 生成数据增强可视化 ---")
# 使用相同的固定索引
visualize_augmentations(image_datasets['train'], class_names, indices_to_plot=[0, 4, 9, 14, 19])

print("\n所有任务完成!")