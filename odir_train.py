"""
=====================================================================================
                    ODIR-5K 眼底图像八分类深度学习微调系统
=====================================================================================
设计概述：
  本脚本基于 PyTorch 框架，实现针对 ODIR-5K 医疗图像数据集的 8 种眼科疾病自动分类。
  系统选用现代卷积神经网络 ConvNeXt-Tiny 作为骨干特征提取器，并对其分类头进行定制化微调。

核心技术架构：
  1. 图像预处理：引入 CLAHE（自适应直方图均衡化）提升医疗图像中微小血管和出血点的对比度。
  2. 防过拟合策略：融合高级数据增强（包含随机平移与缩放仿射变换）与增强型双层 Dropout 机制。
  3. 样本均衡控制：组合应用重采样器（WeightedRandomSampler）与损失权重微调（平滑开根号 alpha）。
  4. 训练机制：采用混合精度加速（AMP），并结合余弦退火学习率策略（CosineAnnealing）与早停机制。
"""

import os
import ast
import time
from collections import Counter
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from PIL import Image
from tqdm import tqdm

# =====================================================================================
# 1. 全局超参数及路径配置
# =====================================================================================
# 自动检测当前计算环境是否成功配置了 CUDA GPU 硬件。若显示为 CPU 请检查当前环境的 PyTorch 版本。
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGE_SIZE = 384    # 图像缩放目标分辨率。医疗眼底图使用大分辨率可防止微小病灶在下采样中丢失。
BATCH_SIZE = 16     # 单次投喂给显卡的样本数。配合大图采用 16 可以防止 8G/12G 显存的显卡溢出(OOM)。
EPOCHS = 50         # 最大微调总轮次。配合早停机制与高效数据增强，50 轮足以为模型提供充分的收敛空间。
LR = 1e-4           # 基础学习率，后续会对特征层和分类层实施分层差异化控制。
NUM_CLASSES = 8     # 固定的 8 分类任务（N, D, G, C, A, H, M, O）
SEED = 42           # 固定随机种子，确保数据集切割和参数初始化结果在每次运行时完全可复现。

# 严格根据真值 CSV 文件中 target 向量索引 [0-7] 建立的疾病名称映射表
CLASS_CHINESE = [
    "正常眼底 (N)",
    "糖尿病视网膜病变 (D)",
    "青光眼 (G)",
    "白内障 (C)",
    "年龄相关性黄斑变性 (A)",
    "高血压眼底病变 (H)",
    "近视黄斑病变 (M)",
    "其他眼病 (O)"
]

# 本地数据与输出路径（请确保文件和文件夹实际存在于该路径下）
CSV_PATH = "D:/python/KG/代码/archive_odir/full_df.csv"
IMAGE_ROOT = "D:/python/KG/代码/archive_odir/preprocessed_images"
SAVE_DIR = "./models/convnext_odir" # 存放整场演练中表现最好的权重文件(.pth)
os.makedirs(SAVE_DIR, exist_ok=True)

# 锁定所有随机数发生器的种子，保证实验结果的严谨与稳定
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =====================================================================================
# 2. 医疗图像增强算法（CLAHE 直方图局部均衡化）
# =====================================================================================
def apply_clahe(img_pil: Image.Image) -> Image.Image:
    """
    专门解决眼底图光照不均、病灶与背景对比度低的问题。
    将图像转换至 LAB 色彩空间，仅对亮度通道(L)进行限制对比度直方图均衡化，随后转回 RGB 空间。
    """
    img = np.array(img_pil)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    # clipLimit=2.0 能够抑制噪点的过度放大；tileGridSize=(8, 8) 表示将图像分成 8x8 的网格局部处理
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


# =====================================================================================
# 3. PyTorch 数据集封装封装类 (Dataset)
# =====================================================================================
class ODIRDataset(Dataset):
    """
    用于对接 PyTorch DataLoader 的数据集类，负责在程序运行时单张读取图像并应用预处理。
    """
    def __init__(self, samples, transform=None, use_clahe=True):
        self.samples = samples        # 格式为: [(图片绝对路径, 类别整数标签), ...]
        self.transform = transform    # torchvision 数据增强管线
        self.use_clahe = use_clahe    # 是否开启 CLAHE 医疗图像增强开关

    def __len__(self):
        return len(self.samples)      # 返回数据集的总样本数

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            # 安全防线：万一本地有极个别图像文件损坏，自动生成一张纯黑背景图，防止程序运行崩溃
            img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color=0)

        # 优先执行图像对比度增强，突出血管和出血点细节
        if self.use_clahe:
            img = apply_clahe(img)

        # 触发后续的随机裁剪、翻转、缩放及归一化
        if self.transform:
            img = self.transform(img)
        return img, label


# =====================================================================================
# 4. 数据增强策略配置（构建防过拟合的阻击阵地）
# =====================================================================================
train_transform = transforms.Compose([
    # 随机裁剪图像的 80%~100% 区域，并统一下采样至 384x384 像素规模
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),  # 50% 概率水平镜面翻转
    transforms.RandomVerticalFlip(),    # 50% 概率垂直镜面翻转
    transforms.RandomRotation(15),      # 随机旋转正负 15 度

    # 仿射变换：允许图像产生最大 5% 的位移以及 0.95~1.05 之间的随机缩放。
    # 作用：打破网络对图像固定像素位置的依赖，迫使其寻找真正的几何几何病灶特征。
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),

    # 随机在合理范围内扰动图像的亮度、对比度和饱和度
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    transforms.ToTensor(),              # 转为张量并把像素值映射至 [0.0, 1.0] 空间
    # 使用 ImageNet 官方统计的标准均值与标准差进行通道标准化，以迎合预训练模型的权重初始化
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 验证集与测试集严禁进行任何随机丢弃或几何变形，仅执行标准的像素缩放与通道归一化
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# =====================================================================================
# 5. ConvNeXt-Tiny 网络拓扑及分类头重组
# =====================================================================================
class ODIRModel(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        # 载入官方在 ImageNet-1K 百万级通用数据集上训练得出的顶级权重
        self.model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

        # 捕捉 ConvNeXt 原生最后一层线性全连接层的输入特征维度（Tiny版本原生为 768）
        in_features = self.model.classifier[2].in_features

        # 双层密集 Dropout 防御：
        # 将原有的随机失活概率分别强化至 0.5 与 0.4。
        # 作用：在每次迭代训练时，强制抹除近一半的特征连接，斩断网络对特定图像细节的
        # 记忆链条，强力抑制“训练集准确率飙升至 97%，测试集停留在 72%”的过拟合顽疾。
        self.model.classifier[2] = nn.Sequential(
            nn.Dropout(0.5),                    # 第一级特征空间高比例失活
            nn.Linear(in_features, 512),        # 中间过渡隐藏层（降维）
            nn.GELU(),                          # 现代高级激活函数
            nn.Dropout(0.4),                    # 第二级分类决策层失活
            nn.Linear(512, num_classes)         # 最终的 8 维 logits 分数输出
        )

    def forward(self, x):
        return self.model(x)


# =====================================================================================
# 6. 独立格式化报告打印工具
# =====================================================================================
def print_class_report(y_true, y_pred, title):
    """
    调用 sklearn 内置算子，精确计算每一个类别的 精确率、召回率、F1-score 以及样本容量。
    """
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, average=None, labels=range(NUM_CLASSES), zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"\n【{title}】总体准确率: {acc:.4f} ({acc*100:.2f}%)  Macro-F1: {macro_f1:.4f}")
    print(f"类别                    精确率     召回率        F1    样本数")
    print("-" * 65)

    for i in range(NUM_CLASSES):
        name = CLASS_CHINESE[i]
        # 针对中文字符在终端排版中占双英文字符位的问题进行尾部空格对齐补偿处理
        if len(name) < 22:
            name += " " * (22 - len(name))
        print(f"{name} {p[i]:.4f}    {r[i]:.4f}    {f1[i]:.4f}    {s[i]}")


# =====================================================================================
# 7. 主训练与验证流水线
# =====================================================================================
def train():
    # 实例化 PyTorch 官方的半精度梯度缩放器（FP16），可将显存占用砍掉近半并加速显卡并行计算
    scaler = GradScaler("cuda")
    print("=" * 85)
    print("  ODIR-5K | 骨干架构: ConvNeXt-Tiny ")
    print("=" * 85)

    # 1. 解析本地 CSV 数据源
    df = pd.read_csv(CSV_PATH, encoding="utf-8", low_memory=False)

    samples, missing = [], 0
    for _, row in df.iterrows():
        try:
            # 安全反序列化：将字符串类型的独热向量（如 "[1,0,0,0,0,0,0,0]"）解析为 Python 数组
            target_list = ast.literal_eval(row["target"])
            if sum(target_list) != 1:
                continue  # 剔除存在多标签标注冲突的数据，确保单标签微调的纯净性
            label_idx = target_list.index(1)  # 抓取数值 1 所在的索引，作为目标病种的整数编码

            img_path = row["filepath"]
            # 兼容性路径校验：若表格中登记的相对路径失效，自动切入预处理根目录进行文件名二次匹配
            if not os.path.exists(img_path):
                img_path = os.path.join(IMAGE_ROOT, os.path.basename(row["filepath"]))
                if not os.path.exists(img_path):
                    missing += 1
                    continue
            samples.append((img_path, label_idx))
        except Exception:
            continue

    print(f"ODIR-5K: 共找到 {len(samples)} 张图像，本地路径未寻获文件 {missing} 张。")

    # 统计汇总当前真实参与本次演练的数据集初始病种分布
    dist = Counter(s[1] for s in samples)
    for i in range(NUM_CLASSES):
        print(f"  [{i}] {CLASS_CHINESE[i]}: {dist.get(i, 0)} 张")

    # 2. 严格按病种分布实施层级切分（20% 划分至测试集，80% 划分至训练集），杜绝不均衡扩散
    by_class = {}
    for s in samples:
        by_class.setdefault(s[1], []).append(s)

    train_samples, test_samples = [], []
    for cls_samples in by_class.values():
        np.random.shuffle(cls_samples)
        n = max(1, int(len(cls_samples) * 0.2))  # 计算当前类别 20% 的截断基准
        test_samples += cls_samples[:n]          # 切入独立测试池
        train_samples += cls_samples[n:]         # 留存训练池

    np.random.shuffle(train_samples)
    np.random.shuffle(test_samples)
    print(f"\n分层切割完毕 -> 训练集体量: {len(train_samples)}    测试集体量: {len(test_samples)}")

    # 3. 分别绑定对应的数据集实例
    train_ds = ODIRDataset(train_samples, train_transform, use_clahe=True)      # 训练专用（包含强数据扰动）
    test_ds = ODIRDataset(test_samples, val_transform, use_clahe=True)          # 测试验证专用（严格静默）
    train_eval_ds = ODIRDataset(train_samples, val_transform, use_clahe=True)   # 训练指标扫描专用（剔除扰动干扰）

    # 4. 解决长尾极度不均衡的数据利器：加权随机采样器 (WeightedRandomSampler)
    # 原理：算出训练集各类别频次，为极度稀缺的病种（如高血压眼底病变）赋予极高的抽样概率系数
    cnt_train = Counter(s[1] for s in train_samples)
    sample_weights = [1.0 / cnt_train[s[1]] for s in train_samples]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # 激活有放回的重复采样机制
    )

    # 5. 实例化高效 DataLoader 管道（自动适配修改后的 BATCH_SIZE=16）
    train_loader = DataLoader(train_ds, BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    train_eval_loader = DataLoader(train_eval_ds, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 6. 损失权重平滑缩放控制（开根号 alpha 法）
    # 在数据端进行重采样的同时，在损失计算端对各病种的 Loss 实施开根号平滑缩放，引导网络关注冷门稀有类别
    n_train = len(train_samples)
    alpha = torch.tensor(
        [np.sqrt(n_train / cnt_train.get(i, 1)) for i in range(NUM_CLASSES)],
        dtype=torch.float32
    ).to(DEVICE)
    # label_smoothing=0.1 引入标签平滑正则化，将真值概率由严格的 1.0 软化为 0.9，防止系统陷入盲目自信
    criterion = nn.CrossEntropyLoss(weight=alpha, label_smoothing=0.1)

    # 7. 分层差异化调优优化器
    model = ODIRModel(NUM_CLASSES).to(DEVICE)

    # 骨干特征提取层（features）富含珍贵的 ImageNet 通用视觉常识，采用 0.5 倍超低学习率进行保迹温和调优；
    # 分类模块（classifier）完全随机初始化，必须全力冲刺，采用全额的 1e-4 基准学习率快速寻优。
    optimizer = optim.AdamW([
        {'params': model.model.features.parameters(), 'lr': LR * 0.5},
        {'params': model.model.avgpool.parameters(), 'lr': LR},
        {'params': model.model.classifier.parameters(), 'lr': LR}
    ], weight_decay=1e-2)

    # 余弦退火策略周期绑定：设置 T_0=25，即在 50 轮的全生命周期中，学习率会经历两次完美的波峰到波谷的蜕变
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, eta_min=1e-6)

    best_f1 = 0.0
    patience = 0
    best_epoch = 1

    print(f"\n运行配置就绪 -> 运算核心设备: {DEVICE} | 预备执行 {EPOCHS} 个完整 Epoch 微调训练...")

    # 8. 主训练 Epoch 迭代核心区
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        model.train()  # 激活训练状态模式（使能 Dropout 的随机掐断特征功能）
        total_loss = 0
        preds_all, labels_all = [], []

        # tqdm 组件将在终端直接绘制图形化进度条
        for img, label in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS}"):
            img, label = img.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad() # 抹除上一批次计算残存的梯度张量

            # 混算作用域：在此范围内的前向计算将全面自动降维至半精度运行
            with autocast("cuda", dtype=torch.float16):
                logits = model(img)
                loss = criterion(logits, label)

            # 缩放器接管反向传播：防止半精度状态下发生梯度下溢
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * img.size(0)
            pred = logits.argmax(1).detach().cpu().numpy()
            preds_all.extend(pred)
            labels_all.extend(label.cpu().numpy())

        train_loss = total_loss / len(train_loader.dataset)
        train_acc = accuracy_score(labels_all, preds_all)

        # 验证阶段：评估当前 Epoch 模型在未见过的测试集上的实际表现
        model.eval()  # 切入评估状态模式（冻结 Dropout 连接，实现确定性输出）
        test_loss, y_pred, y_true = 0, [], []
        with torch.no_grad(), autocast("cuda", dtype=torch.float16):
            for img, label in test_loader:
                img, label = img.to(DEVICE), label.to(DEVICE)
                logits = model(img)
                loss = criterion(logits, label)
                test_loss += loss.item() * img.size(0)
                y_pred.extend(logits.argmax(1).cpu().numpy())
                y_true.extend(label.cpu().numpy())

        test_loss /= len(test_loader.dataset)
        test_acc = accuracy_score(y_true, y_pred)
        test_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        cost = int(time.time() - start_time)

        # 终端实时播报本轮次多项指标
        print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss:{train_loss:.3f} Acc:{train_acc:.3f} | Val Loss:{test_loss:.3f} Acc:{test_acc:.3f} | Macro-F1:{test_f1:.4f} | {cost}s")

        # 检验是否刷新了历史最佳综合性能成绩（设定的有效特征跃升阈值为 0.002）
        if test_f1 > best_f1 + 0.002:
            best_f1 = test_f1
            best_epoch = epoch
            torch.save(model.state_dict(), f"{SAVE_DIR}/best.pth") # 在本地完美覆盖存储最优权重
            print(f"  [BEST] 成功捕获到真正具备高泛化能力的特征跃升，最优模型已同步至本地 (val_f1={test_f1:.4f})")
            patience = 0 # 重置早停计数器
        else:
            patience += 1
            # 止损防御=
            # 若连续 10 个周期内综合指标 Macro-F1 陷入停滞不前的状态，判定系统已达饱和饱和态，提前终止。
            if patience >= 10:
                print("【早停机制触发】：连续 10 个轮次多类性能均未见明显跃升，提前终止训练。")
                break
        scheduler.step()  # 依循余弦退火轨迹更新下一轮的学习率大小

    # =====================================================================================
    # 9. 最终评估
    # =====================================================================================
    print("\n" + "=" * 85)
    print("  最终评估报告 ")
    print("=" * 85)
    print(f"本次训练的最佳阶段记录产生于第 {best_epoch} 个 Epoch")

    # 强行自本地拉回最优权重参数
    model.load_state_dict(torch.load(f"{SAVE_DIR}/best.pth", map_location=DEVICE))
    model.eval()

    # 1. 扫描测试集数据流，输出标准清晰的测试集最终报告
    y_true_test, y_pred_test = [], []
    with torch.no_grad(), autocast("cuda", dtype=torch.float16):
        for img, label in test_loader:
            img = img.to(DEVICE)
            logits = model(img)
            y_pred_test.extend(logits.argmax(1).cpu().numpy())
            y_true_test.extend(label.numpy())
    print_class_report(y_true_test, y_pred_test, "测试集最终报告")

    # 2. 扫描纯净的训练集数据流（关闭重采样机制），核验其抗过拟合策略的最终拦截成效
    y_true_train, y_pred_train = [], []
    with torch.no_grad(), autocast("cuda", dtype=torch.float16):
        for img, label in train_eval_loader:
            img = img.to(DEVICE)
            logits = model(img)
            y_pred_train.extend(logits.argmax(1).cpu().numpy())
            y_true_train.extend(label.numpy())
    print_class_report(y_true_train, y_pred_train, "训练集最终报告")


if __name__ == "__main__":
    train()
