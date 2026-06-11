"""
外眼病多模态数据集模块。

类别体系（固定）：
    0: 白内障 (Cataract)       — 高风险
    1: 结膜炎 (Conjunctivitis) — 高风险
    2: 眼睑疾病 (Eyelid)        — 高风险
    3: 葡萄膜炎 (Uveitis)       — 高风险
    4: 正常 (Normal)

数据组织：
    data/archive_external/<类别名>/<image.jpg|png|...>
    data/annotations/symptoms.json    # 可选：{image_file_name: "症状文本"}

功能：
    - ExternalEyeDataset: (image, text, disease_label)
    - 训练 / 验证 / 测试集三层划分（7:1.5:1.5，可配置）
    - 训练集增广：随机翻转、仿射、颜色抖动、高斯模糊
    - 尺寸统一 300×300，ImageNet 均值/方差归一化

用法：
    from models.dataset import build_dataloaders, EXTERNAL_EYE_CLASSES, DISEASE_CLASS_NAMES
    loaders, meta = build_dataloaders()
"""
from __future__ import annotations

# 标准库
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# 第三方库
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms

# 项目内部：类别体系、路径、训练参数、高风险名单
from config import (
    CLASS_MAP,
    DATA_SPLIT,
    EXTERNAL_EYE_DATA_DIR,
    EXTERNAL_EYE_SUBCLASS,
    HIGH_RISK_EXTERNAL,
    IMAGE_EXTS,
    RANDOM_SEED,
    SYMPTOM_JSON,
    TRAIN_CONFIG,
)

# ── 固定的细分类目顺序 ─────────────────────────────────
# 该顺序即模型输出维度 0..N-1，训练 / 推理 / 评估均需保持一致
EXTERNAL_EYE_CLASSES: List[str] = [
    *list(EXTERNAL_EYE_SUBCLASS.keys()),  # 4 类外眼病
    "Normal",                              # 正常
]

# 中文名称（与 EXTERNAL_EYE_CLASSES 一一对应）
DISEASE_CLASS_NAMES: List[str] = [CLASS_MAP[k] for k in EXTERNAL_EYE_CLASSES]


@dataclass
class DatasetMeta:
    num_disease_classes: int
    disease_class_names: List[str]       # 中文名称，与索引对应
    disease_class_keys: List[str]        # 英文 key，与索引对应
    high_risk_indices: List[int]         # 高风险类别索引
    class_counts: List[int]               # 每类样本数
    train_size: int
    val_size: int
    test_size: int

    @property
    def total_samples(self) -> int:
        return self.train_size + self.val_size + self.test_size


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ExternalEyeDataset(Dataset):
    """
    外眼病多模态数据集：返回 (image_tensor, text, disease_label)。
    目录结构：<root>/<class_key>/<image.ext>；
    可选症状 JSON：{image_file_name: "症状文本"}。
    对缺失目录的类别自动跳过，不报错。
    """

    def __init__(
        self,
        root: Path = EXTERNAL_EYE_DATA_DIR,
        symptom_json: Optional[Path] = SYMPTOM_JSON,
        transform=None,
        class_keys: Sequence[str] = EXTERNAL_EYE_CLASSES,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.class_keys = list(class_keys)
        # 类别键 -> 模型输出 logit 索引
        self.class_to_idx = {name: i for i, name in enumerate(self.class_keys)}

        # 加载症状文本映射
        self.symptoms: Dict[str, str] = {}
        if symptom_json and Path(symptom_json).exists():
            with open(symptom_json, "r", encoding="utf-8") as f:
                self.symptoms = json.load(f)

        # 收集每个类别的样本 (path, disease_key, text)
        self.samples: List[Tuple[Path, str, str]] = []
        for class_key in self.class_keys:
            class_dir = self.root / class_key
            if not class_dir.exists():
                print(f"[dataset] WARNING: 目录不存在，跳过: {class_dir}")
                continue
            n_in_class = 0
            for img_path in sorted(class_dir.iterdir()):
                if not img_path.is_file():
                    continue
                if img_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                text = self.symptoms.get(img_path.name, "")
                self.samples.append((img_path, class_key, text))
                n_in_class += 1
            if n_in_class == 0:
                print(f"[dataset] WARNING: {class_dir} 为空")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, disease_key, text = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        # 标签：把类别键 -> 整型索引，用于 CrossEntropyLoss / FocalLoss
        disease_label = torch.tensor(self.class_to_idx[disease_key], dtype=torch.long)
        return image, text, disease_label

    def disease_counts(self) -> List[int]:
        """每类样本数，可用于类别不均衡采样 / 打印统计。"""
        counts = [0] * len(self.class_keys)
        for _, key, _ in self.samples:
            counts[self.class_to_idx[key]] += 1
        return counts


def build_transforms(image_size: int = TRAIN_CONFIG["image_size"]):
    """
    构造 train / eval 两组 transform：
        - 训练集：随机水平/垂直翻转 + 仿射 + 颜色抖动 + 高斯模糊，模拟真实采集偏差
        - 验证/测试集：仅 resize + 归一化，保证可复现性
    归一化遵循 ImageNet 均值方差，以便与 EfficientNet-B3 预训练权重对齐。
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomAffine(
            degrees=(-15, 15), translate=(0.05, 0.05), scale=(0.9, 1.1)
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, eval_transform


def _split_indices(n: int, split: Dict[str, float], seed: int = RANDOM_SEED):
    """
    按比例 (train / val / test) 对索引做随机切分。
    - 样本过少时，保证每个 split 至少 1 条；
    - 使用全局 RANDOM_SEED，保证可复现。
    """
    g = torch.Generator().manual_seed(seed)
    n_train = max(int(n * split["train"]), 1)
    n_val = max(int(n * split["val"]), 1)
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        n_train = max(n - n_val - n_test, 1)
    return random_split(range(n), [n_train, n_val, n_test], generator=g)


class _TrainSubset(Dataset):
    """
    训练子集包装器：为训练样本应用特定的 transform（如数据增强）。
    移到模块级别以支持 Windows 上的多进程 (pickle 序列化)。
    """
    def __init__(self, base: ExternalEyeDataset, indices, transform):
        self.base = base
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        path, disease_key, text = self.base.samples[self.indices[idx]]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(self.base.class_to_idx[disease_key], dtype=torch.long)
        return img, text, label


def build_dataloaders(
    root: Path = EXTERNAL_EYE_DATA_DIR,
    batch_size: int = TRAIN_CONFIG["batch_size"],
    num_workers: int = TRAIN_CONFIG["num_workers"],
    image_size: int = TRAIN_CONFIG["image_size"],
    split: Optional[Dict[str, float]] = None,
    seed: int = RANDOM_SEED,
) -> Tuple[Dict[str, DataLoader], DatasetMeta]:
    """
    构造 (train / val / test) 三个 DataLoader，以及数据集元信息 DatasetMeta。
    - 划分：按 DATA_SPLIT 比例随机切分，使用固定 seed 保证可复现
    - 训练集：单独套一层随机增强（水平翻转 / 仿射 / 颜色抖动 / 高斯模糊）
    - 验证 / 测试：仅 resize + 归一化，避免数据泄露
    - 高风险类别：通过 HIGH_RISK_EXTERNAL 反查 class_to_idx，返回给训练/评估脚本
    """
    split = split or DATA_SPLIT
    train_tfm, eval_tfm = build_transforms(image_size)

    # 全集用 eval transform（不影响随机划分的正确性，只对 train 后面替换 transform）
    full_ds = ExternalEyeDataset(root=root, transform=eval_tfm, class_keys=EXTERNAL_EYE_CLASSES)
    if len(full_ds) == 0:
        raise RuntimeError(
            f"数据集为空：{root}；请在 {root}/<类别名>/ 下放入图像，"
            f"可选类别：{EXTERNAL_EYE_CLASSES}"
        )

    train_idx, val_idx, test_idx = _split_indices(len(full_ds), split, seed=seed)

    # 训练子集：使用模块级别的 _TrainSubset 类
    train_ds = _TrainSubset(full_ds, train_idx.indices, train_tfm)
    val_ds = Subset(full_ds, val_idx.indices)
    test_ds = Subset(full_ds, test_idx.indices)

    # DataLoader：train shuffle=True，val/test 保持原始顺序避免偏差
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, worker_init_fn=_seed_worker, generator=g, drop_last=False,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 统计信息：类别计数 + 高风险类别索引，供训练 / 评估脚本使用
    counts = full_ds.disease_counts()
    high_risk_indices = [
        full_ds.class_to_idx[k] for k in HIGH_RISK_EXTERNAL if k in full_ds.class_to_idx
    ]
    meta = DatasetMeta(
        num_disease_classes=len(EXTERNAL_EYE_CLASSES),
        disease_class_names=list(DISEASE_CLASS_NAMES),
        disease_class_keys=list(EXTERNAL_EYE_CLASSES),
        high_risk_indices=high_risk_indices,
        class_counts=counts,
        train_size=len(train_ds),
        val_size=len(val_ds),
        test_size=len(test_ds),
    )
    print("[dataset] 外眼数据集统计：")
    for k, cn, c in zip(EXTERNAL_EYE_CLASSES, DISEASE_CLASS_NAMES, counts):
        tag = "（高风险）" if k in HIGH_RISK_EXTERNAL else ""
        print(f"  - {k:<16}({cn})  {c} 张 {tag}")
    print(f"[dataset] train={meta.train_size}, val={meta.val_size}, test={meta.test_size}")
    print(f"[dataset] 高风险类别索引: {meta.high_risk_indices}")
    return {"train": train_loader, "val": val_loader, "test": test_loader}, meta
