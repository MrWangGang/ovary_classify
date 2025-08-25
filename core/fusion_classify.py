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
import platform # 导入 platform 模块来判断操作系统
import matplotlib.font_manager as fm # 导入 font_manager 用于加载字体

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
# 数据集/模态前缀，用于标识两个模态 (超声波 US, 计算机断层扫描 CT)
DATA_PREFIXES = ['us', 'ct']
# 融合模型的前缀，用于报告和保存目录
FUSION_MODEL_PREFIX = 'fusion'

# --- 全局配置结束 ---


## 路径与超参数配置

DATA_DIR = './datasets' # 数据集根目录

# 为每个模态定义单独的训练和测试目录
# 这些路径应该指向包含 'bt' 和 'mt' 等类别文件夹的父目录。
# 例如，如果数据集结构是 us_train/bt/images/...，那么 US_TRAIN_DIR 就是 us_train
US_TRAIN_DIR = os.path.join(DATA_DIR, f'{DATA_PREFIXES[0]}_train')
US_TEST_DIR = os.path.join(DATA_DIR, f'{DATA_PREFIXES[0]}_test')

CT_TRAIN_DIR = os.path.join(DATA_DIR, f'{DATA_PREFIXES[1]}_train')
CT_TEST_DIR = os.path.join(DATA_DIR, f'{DATA_PREFIXES[1]}_test')

# 融合模型的保存目录
MODEL_SAVE_DIR = f'./report/classify/{FUSION_MODEL_PREFIX}/models'
BEST_MODEL_FILENAME = 'best_fusion_classification_model.pth'
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, BEST_MODEL_FILENAME) # 最佳模型保存路径

METRICS_LOG_DIR = f'./report/classify/{FUSION_MODEL_PREFIX}/logs'
METRICS_JSON_FILENAME = 'training_metrics_log.json'
METRICS_JSON_PATH = os.path.join(METRICS_LOG_DIR, METRICS_JSON_FILENAME) # 训练指标日志保存路径

PLOTS_SAVE_DIR = f'./report/classify/{FUSION_MODEL_PREFIX}/plots'
CONFUSION_MATRIX_PATH = os.path.join(PLOTS_SAVE_DIR, 'confusion_matrix.png') # 混淆矩阵保存路径

LOSS_TREND_PATH = os.path.join(PLOTS_SAVE_DIR, 'loss_trend.png') # 损失趋势图路径

# 各项指标趋势图的保存路径
TREND_PLOT_PATHS = {
    'roc_auc': os.path.join(PLOTS_SAVE_DIR, 'roc_auc_trend.png'),
    'pr_auc': os.path.join(PLOTS_SAVE_DIR, 'pr_auc_trend.png'),
    'accuracy': os.path.join(PLOTS_SAVE_DIR, 'accuracy_trend.png'),
    'sensitivity': os.path.join(PLOTS_SAVE_DIR, 'sensitivity_trend.png'),
    'specificity': os.path.join(PLOTS_SAVE_DIR, 'specificity_trend.png'),
    'precision': os.path.join(PLOTS_SAVE_DIR, 'precision_trend.png'),
    'npv': os.path.join(PLOTS_SAVE_DIR, 'npv_trend.png'),
    'f1_score': os.path.join(PLOTS_SAVE_DIR, 'f1_score_trend.png')
}

NUM_CLASSES = 2 # 分类类别数量 (例如：良性/恶性)
BATCH_SIZE = 32 # 批处理大小
LEARNING_RATE = 0.0001 # 学习率
NUM_EPOCHS = 100 # 训练轮数
IMAGE_SIZE = 224 # 图像大小 (所有输入图像将被调整为这个尺寸)

# 创建必要的目录，如果它们不存在的话
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(METRICS_LOG_DIR, exist_ok=True)
os.makedirs(PLOTS_SAVE_DIR, exist_ok=True)

# 确定运行设备 (优先使用 GPU，如果可用)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


## 图像变换和数据加载器

# 定义不同阶段的图像预处理和数据增强
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE), # 随机裁剪并缩放为 IMAGE_SIZE
        transforms.RandomRotation(15), # 随机旋转 ±15 度
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # 随机调整亮度、对比度、饱和度和色调
        transforms.RandomGrayscale(p=0.1), # 10% 的概率转换为灰度图
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)), # 随机高斯模糊
        transforms.ToTensor(), # 将 PIL Image 或 NumPy array 转换为 PyTorch Tensor (HWC -> CHW, 0-255 -> 0.0-1.0)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 使用 ImageNet 的均值和标准差进行归一化
    ]),
    'test': transforms.Compose([
        transforms.Resize(IMAGE_SIZE + 32), # 放大后裁剪，确保中心区域
        transforms.CenterCrop(IMAGE_SIZE), # 中心裁剪
        transforms.ToTensor(), # 转换为Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 归一化
    ])
}

class FusionImageFolder(torch.utils.data.Dataset):
    """
    一个自定义数据集类，用于加载US和CT图像对。
    它会查找两个根目录中具有相同相对路径的图像。
    关键改进：**明确过滤掉路径中包含 'masks' 文件夹的文件**。
    """
    def __init__(self, us_root_dir, ct_root_dir, transform=None):
        self.us_root_dir = us_root_dir
        self.ct_root_dir = ct_root_dir
        self.transform = transform

        # 使用原始 ImageFolder 来获取所有图像路径及其对应的类别信息。
        # 注意：ImageFolder 默认会将根目录下的直接子目录视为类别。
        # 我们的目标是识别 'bt' 和 'mt' 作为类别。
        self.us_dataset_raw = datasets.ImageFolder(us_root_dir)
        self.ct_dataset_raw = datasets.ImageFolder(ct_root_dir)

        # 定义需要排除的子文件夹名称列表，例如 'masks' 文件夹不应包含分类图像
        self.excluded_subfolders = ['masks']

        # 初始化真实的分类类别信息。
        # 从 US 数据集获取类别名称和索引映射。
        self.classes = self.us_dataset_raw.classes
        self.class_to_idx = self.us_dataset_raw.class_to_idx
        # 为方便访问，创建索引到类别名称的反向映射
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # 验证类别数量是否正确
        if len(self.classes) != NUM_CLASSES:
            print(f"Warning: Detected {len(self.classes)} classes, but expected {NUM_CLASSES}.")
            print(f"Detected classes: {self.classes}")
            print("Please check your dataset structure to ensure only relevant top-level class folders exist.")

        # 存储 (us_image_path, ct_image_path, label) 元组的列表
        # 并且存储了每个样本的真实标签，方便后续按类别索引
        self.samples, self.labels = self._find_common_samples_and_filter()

    def _find_common_samples_and_filter(self):
        """
        查找US和CT目录中具有相同相对路径的图像样本，并**明确过滤掉来自 'masks' 文件夹的图片**。
        同时，确保标签映射到正确的分类类别。
        返回 (common_samples, labels_list)
        """
        common_samples = []
        labels_list = []
        # 用于存储US图片信息，键是相对于 us_root_dir 的路径，值是 (完整US路径, 标签)
        # 例如: {'bt/images/image_001.png': ('/path/to/us_train/bt/images/image_001.png', 0)}
        us_samples_map = {}

        # 遍历US数据集的原始样本 (ImageFolder 找到的所有图片)
        for path, original_label_idx in self.us_dataset_raw.samples:
            normalized_path = os.path.normpath(path) # 标准化路径，处理不同操作系统的路径分隔符
            is_mask_path = False
            for subfolder_name in self.excluded_subfolders: # 检查路径中是否包含排除的子文件夹名称 (如 'masks')
                # 检查路径中是否包含 '/subfolder_name/' 这样的段，例如 '/masks/'，以确保匹配的是子文件夹而非文件名的一部分
                if f'{os.sep}{subfolder_name}{os.sep}' in normalized_path.replace('\\', os.sep):
                    is_mask_path = True
                    break

            if not is_mask_path: # 如果图片路径中不包含 'masks' 文件夹
                relative_path_from_root = os.path.relpath(path, self.us_root_dir) # 获取相对于根目录的相对路径
                class_name_from_path = relative_path_from_root.split(os.sep)[0] # 从相对路径中提取顶级类别名称 (如 'bt' 或 'mt')
                if class_name_from_path in self.class_to_idx:
                    current_label = self.class_to_idx[class_name_from_path] # 获取对应的数值标签
                    us_samples_map[relative_path_from_root] = (path, current_label) # 存储 US 图像信息

        # 遍历CT数据集的原始样本，并过滤掉来自 'masks' 文件夹的图片
        for ct_path, original_label_idx in self.ct_dataset_raw.samples:
            normalized_ct_path = os.path.normpath(ct_path)
            is_mask_path = False
            for subfolder_name in self.excluded_subfolders:
                if f'{os.sep}{subfolder_name}{os.sep}' in normalized_ct_path.replace('\\', os.sep):
                    is_mask_path = True
                    break

            if not is_mask_path: # 如果图片路径中不包含 'masks' 文件夹
                relative_path_from_root_ct = os.path.relpath(ct_path, self.ct_root_dir)
                class_name_from_path_ct = relative_path_from_root_ct.split(os.sep)[0]
                if class_name_from_path_ct in self.class_to_idx:
                    # 如果 CT 图片的相对路径在 US 样本映射中存在，则找到了一个匹配的样本对
                    if relative_path_from_root_ct in us_samples_map:
                        us_path, label = us_samples_map[relative_path_from_root_ct]
                        common_samples.append((us_path, ct_path, label)) # 添加到共同样本列表
                        labels_list.append(label) # 同时记录标签

        if not common_samples:
            raise RuntimeError(f"No common non-mask image samples found in {self.us_root_dir} and {self.ct_root_dir}. Please check dataset paths and structure.")

        print(f"Successfully found {len(common_samples)} common US-CT image pairs.")
        return common_samples, labels_list

    def __len__(self):
        """返回数据集中的样本对数量"""
        return len(self.samples)

    def __getitem__(self, idx):
        """根据索引加载并返回一个 US-CT 图像对及其标签"""
        us_path, ct_path, label = self.samples[idx]

        # 打开图像并转换为 RGB 格式 (确保3通道)
        us_image = Image.open(us_path).convert('RGB')
        ct_image = Image.open(ct_path).convert('RGB')

        # 应用图像变换
        if self.transform:
            us_image = self.transform(us_image)
            ct_image = self.transform(ct_image)

        return us_image, ct_image, label

# 创建数据集实例
fusion_image_datasets = {
    'train': FusionImageFolder(US_TRAIN_DIR, CT_TRAIN_DIR, transform=data_transforms['train']),
    'test': FusionImageFolder(US_TEST_DIR, CT_TEST_DIR, transform=data_transforms['test'])
}

# 打印训练集和验证集的大小
print(f"Fusion model training set sample pairs: {len(fusion_image_datasets['train'])}")
print(f"  (US channel image count: {len(fusion_image_datasets['train'])}, CT channel image count: {len(fusion_image_datasets['train'])})")
print(f"Fusion model validation set sample pairs: {len(fusion_image_datasets['test'])}")
print(f"  (US channel image count: {len(fusion_image_datasets['test'])}, CT channel image count: {len(fusion_image_datasets['test'])})")
# --- 添加的代码开始 ---
print("\n--- Details of the first 5 sample pairs in the fusion model training set ---")
for i in range(min(5, len(fusion_image_datasets['train'].samples))): # 确保不超过样本总数
    us_path, ct_path, label = fusion_image_datasets['train'].samples[i]
    # 将数值标签转换为可读的类别名称
    class_name = fusion_image_datasets['train'].idx_to_class[label]

    print(f"Sample pair {i+1}:")
    print(f"  US image path: {us_path}")
    print(f"  CT image path: {ct_path}")
    print(f"  Class label: {label} ({class_name})")
    print("-" * 30)

print("--- Details printed ---")
# --- 添加的代码结束 ---

# 创建数据加载器
dataloaders = {
    'train': torch.utils.data.DataLoader(fusion_image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
    'test': torch.utils.data.DataLoader(fusion_image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
}

# 获取类别名称和索引映射
class_names = fusion_image_datasets['train'].classes
class_to_idx = fusion_image_datasets['train'].class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()} # 索引到类别名称的反向映射
print(f"Detected classes: {class_names}")
print(f"Class index mapping: {class_to_idx}")


## 模型定义 (FusionResNet)

class FusionResNet(nn.Module):
    """
    基于两个预训练 ResNet 模型的特征融合分类器。
    US 和 CT 图像分别通过独立的 ResNet 骨干网络提取特征，
    然后将特征拼接起来，通过一个自定义的分类头进行分类。
    """
    def __init__(self, num_classes=2):
        super(FusionResNet, self).__init__()
        # 加载预训练的 ResNet50 模型作为 US 图像的特征提取器
        self.resnet_us = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # 加载预训练的 ResNet50 模型作为 CT 图像的特征提取器
        self.resnet_ct = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # 移除原 ResNet 的最后分类层 (avgpool 和 fc 层)
        # 这样我们得到的是平均池化之前的特征图 (2048维特征向量)
        self.resnet_us = nn.Sequential(*list(self.resnet_us.children())[:-1])
        self.resnet_ct = nn.Sequential(*list(self.resnet_ct.children())[:-1])

        # 定义融合后的分类器
        # ResNet50 的特征输出维度是 2048。两个模态融合后，特征维度是 2048 * 2 = 4096。
        # 这是一个简单的全连接网络，包含一个 ReLU 激活和一个 Dropout 层。
        self.classifier = nn.Sequential(
            nn.Linear(2048 * 2, 512), # 从 4096 降维到 512
            nn.ReLU(True), # 激活函数
            nn.Dropout(0.5), # 随机失活，防止过拟合
            nn.Linear(512, num_classes) # 输出到类别数量的线性层
        )

    def forward(self, us_input, ct_input):
        """
        前向传播函数。
        us_input: US 图像的输入张量
        ct_input: CT 图像的输入张量
        """
        # 提取 US 图像特征
        us_features = self.resnet_us(us_input)
        us_features = torch.flatten(us_features, 1) # 将特征展平为一维向量 (Batch_Size, 2048)

        # 提取 CT 图像特征
        ct_features = self.resnet_ct(ct_input)
        ct_features = torch.flatten(ct_features, 1) # 将特征展平为一维向量 (Batch_Size, 2048)

        # 融合特征 (这里简单地进行特征拼接，dim=1 表示在特征维度上拼接)
        fused_features = torch.cat((us_features, ct_features), dim=1) # (Batch_Size, 4096)

        # 经过分类器
        outputs = self.classifier(fused_features)
        return outputs

# 实例化模型，损失函数和优化器
model = FusionResNet(num_classes=NUM_CLASSES)
model = model.to(device) # 将模型移动到指定设备 (CPU/GPU)

criterion = nn.CrossEntropyLoss() # 交叉熵损失函数，适用于多分类问题
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # Adam 优化器
# 学习率调度器：当验证损失停止改善时，降低学习率 (mode='min' 监控最小值，factor=0.1 降低10倍，patience=10 忍受10个epoch不改善)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)


## 辅助函数

def calculate_metrics(y_true, y_pred_probs, y_pred_labels):
    """
    计算并返回分类指标。
    y_true: 真实标签列表 (NumPy 数组)
    y_pred_probs: 预测正类概率列表 (NumPy 数组)
    y_pred_labels: 预测的类别标签列表 (NumPy 数组)
    """

    # 确保输入是 NumPy 数组
    y_true = np.array(y_true)
    y_pred_labels = np.array(y_pred_labels)
    y_pred_probs = np.array(y_pred_probs)

    metrics = {}

    # ROC AUC (需要至少两个不同的真实类别才能计算)
    if len(np.unique(y_true)) > 1:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_probs)
    else:
        metrics['roc_auc'] = np.nan # 如果只有一个类别，则AUC无意义

    # PR AUC (精确度-召回率曲线下面积)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    metrics['pr_auc'] = auc(recall, precision)

    # 混淆矩阵的四个基本值 (真阴性 TN, 假阳性 FP, 假阴性 FN, 真阳性 TP)
    # 假设类别 0 是负类，1 是正类
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_labels, labels=[0, 1]).ravel()

    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0 # 准确率
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # 灵敏度 (召回率, Recall)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # 特异度
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0 # 精确度 (阳性预测值, PPV)
    # F1 Score 是精确度和召回率的调和平均值
    metrics['f1_score'] = 2 * (metrics['precision'] * metrics['sensitivity']) / (metrics['precision'] + metrics['sensitivity']) if (metrics['precision'] + metrics['sensitivity']) > 0 else 0.0

    # 阴性预测值 (Negative Predictive Value, NPV)
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return metrics

def plot_confusion_matrix(y_true, y_pred_labels, class_names, save_path):
    """绘制并保存混淆矩阵图"""
    cm = confusion_matrix(y_true, y_pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_training_metrics(metrics_log, save_dir):
    """绘制并保存训练过程中各项指标的趋势图"""
    epochs = range(1, len(metrics_log) + 1)

    # 绘制损失曲线
    train_losses = [m['train_loss'] for m in metrics_log]
    test_losses = [m['test_loss'] for m in metrics_log]
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, test_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Trend')
    plt.legend()
    plt.grid(True)
    plt.savefig(LOSS_TREND_PATH)
    plt.close()

    # 绘制其他指标曲线 (例如 ROC AUC, Accuracy 等)
    metrics_to_plot = {
        'roc_auc': 'ROC AUC',
        'pr_auc': 'PR AUC',
        'accuracy': 'Accuracy',
        'sensitivity': 'Sensitivity',
        'specificity': 'Specificity',
        'precision': 'Precision',
        'npv': 'NPV',
        'f1_score': 'F1 Score'
    }

    for metric_key, metric_name in metrics_to_plot.items():
        if f'test_{metric_key}' in metrics_log[0]: # 检查指标是否存在
            values = [m[f'test_{metric_key}'] for m in metrics_log]
            plt.figure(figsize=(10, 5))
            plt.plot(epochs, values, label=f'Validation {metric_name}', marker='o', markersize=4)
            plt.xlabel('Epoch')
            plt.ylabel(metric_name)
            plt.title(f'{metric_name} Trend')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, f'{metric_key}_trend.png'))
            plt.close()

## 训练和评估循环

def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=NUM_EPOCHS):
    """
    训练和验证模型的函数。
    model: PyTorch 模型
    criterion: 损失函数
    optimizer: 优化器
    scheduler: 学习率调度器
    dataloaders: 包含 'train' 和 'test' DataLoader 的字典
    num_epochs: 训练的总轮数
    """
    best_acc = 0.0 # 记录最佳验证准确率
    metrics_log = [] # 记录每个 epoch 的所有指标

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 每个 epoch 都包含训练和验证阶段
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train() # 设置模型为训练模式
            else:
                model.eval() # 设置模型为评估模式

            running_loss = 0.0
            all_labels = [] # 存储所有真实标签
            all_preds = [] # 存储所有预测标签
            all_probs = [] # 存储所有预测概率 (正类概率)

            # 遍历数据加载器中的每个批次
            for us_inputs, ct_inputs, labels in dataloaders[phase]:
                # 将数据移动到指定设备 (GPU/CPU)
                us_inputs = us_inputs.to(device)
                ct_inputs = ct_inputs.to(device)
                labels = labels.to(device)

                # 梯度清零
                optimizer.zero_grad()

                # 前向传播
                # 只有在训练阶段才计算梯度 (用于反向传播)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(us_inputs, ct_inputs) # 模型前向传播
                    loss = criterion(outputs, labels) # 计算损失

                    _, preds = torch.max(outputs, 1) # 获取预测类别 (得分最高的类别索引)
                    probs = torch.softmax(outputs, dim=1)[:, 1] # 获取正类 (索引为1) 的概率

                    # 后向传播 + 优化 (仅在训练阶段执行)
                    if phase == 'train':
                        loss.backward() # 反向传播计算梯度
                        optimizer.step() # 更新模型参数

                # 统计当前批次的损失和预测结果
                running_loss += loss.item() * us_inputs.size(0) # 累加损失 (乘以批次大小)
                all_labels.extend(labels.cpu().numpy()) # 收集真实标签 (移动到 CPU 并转换为 NumPy)
                all_preds.extend(preds.cpu().numpy()) # 收集预测标签 (移动到 CPU 并转换为 NumPy)
                all_probs.extend(probs.cpu().detach().numpy()) # 收集预测概率 (移动到 CPU, 分离梯度，转换为 NumPy)

            # 计算当前 epoch 的平均损失和各项指标
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_metrics = calculate_metrics(all_labels, all_probs, all_preds)

            # 只输出 Loss 和 Acc
            print(f'{phase} Loss: {epoch_loss:.4f} Accuracy: {epoch_metrics["accuracy"]:.4f}')

            # 记录所有指标到日志，以便后续绘制曲线
            if phase == 'train':
                current_epoch_metrics = {'epoch': epoch + 1, 'train_loss': epoch_loss}
                for k, v in epoch_metrics.items():
                    current_epoch_metrics[f'train_{k}'] = v
            else: # test phase
                current_epoch_metrics['test_loss'] = epoch_loss
                for k, v in epoch_metrics.items():
                    current_epoch_metrics[f'test_{k}'] = v

                # 更新学习率调度器 (基于验证损失)
                scheduler.step(epoch_loss)

                # 保存最佳模型 (基于验证准确率)
                if epoch_metrics['accuracy'] > best_acc:
                    best_acc = epoch_metrics['accuracy']
                    torch.save(model.state_dict(), MODEL_SAVE_PATH) # 保存模型的状态字典
                    print(f"Current best model accuracy: {best_acc:.4f}")

        metrics_log.append(current_epoch_metrics) # 每个 epoch 结束后，将该 epoch 的所有指标添加到日志中
        print() # 打印空行分隔不同 epoch 的输出

    print(f'Training completed. Best validation accuracy: {best_acc:.4f}')

    # 将训练指标日志保存到 JSON 文件
    with open(METRICS_JSON_PATH, 'w') as f:
        json.dump(metrics_log, f, indent=4) # indent=4 使 JSON 文件更易读
    print(f"Training metrics log saved to {METRICS_JSON_PATH}")

    return model, metrics_log


## 主执行流程

if __name__ == '__main__':
    # 训练模型，并获取训练后的模型和指标历史记录
    model_ft, metrics_history = train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=NUM_EPOCHS)

    # 绘制并保存训练过程中各项指标的趋势图
    plot_training_metrics(metrics_history, PLOTS_SAVE_DIR)
    print(f"Training loss and metrics trend plots saved to {PLOTS_SAVE_DIR}")

    # 加载在训练过程中保存的最佳模型进行最终评估和可视化
    model_ft.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model_ft.eval() # 确保模型处于评估模式 (禁用 Dropout 等)

    # 在验证集上收集所有真实标签、预测标签和预测概率
    all_test_labels = []
    all_test_preds = []
    all_test_probs = []

    with torch.no_grad(): # 在最终评估阶段不计算梯度
        for us_inputs, ct_inputs, labels in dataloaders['test']:
            us_inputs = us_inputs.to(device)
            ct_inputs = ct_inputs.to(device)
            labels = labels.to(device)

            outputs = model_ft(us_inputs, ct_inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1] # 获取正类概率
            _, preds = torch.max(outputs, 1) # 获取预测类别

            all_test_labels.extend(labels.cpu().numpy())
            all_test_preds.extend(preds.cpu().numpy())
            all_test_probs.extend(probs.cpu().numpy())

    # 计算最终的验证集指标
    final_metrics = calculate_metrics(all_test_labels, all_test_probs, all_test_preds)
    print("\n--- Final Validation Metrics ---")
    for key, value in final_metrics.items():
        # 将指标名称中的下划线替换为空格，并首字母大写，用于更友好的输出
        print(f"{key.replace('_', ' ').capitalize()}: {value:.4f}")

    # 绘制最终的混淆矩阵图
    plot_confusion_matrix(all_test_labels, all_test_preds, class_names, CONFUSION_MATRIX_PATH)
    print(f"Confusion matrix plot saved to {CONFUSION_MATRIX_PATH}")

    print("\nAll tasks completed.")