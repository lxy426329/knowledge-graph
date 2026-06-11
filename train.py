r"""
外眼病多模态模型训练：
    图像 EfficientNet-B3 + 文本 BERT → 融合 → 5 类外眼病细分类。

运行：
    .venv\Scripts\python.exe train.py

产物：
    checkpoints/best.pt            # 最佳权重（含类别映射，便于推理使用）
    checkpoints/class_map.json     # 类别映射（供推理 / 知识图谱使用）
    runs/train_<timestamp>.log     # 训练日志
"""
from __future__ import annotations

# 标准库
import json
import logging
import time
from pathlib import Path
from typing import Dict

# PyTorch
import torch
import torch.nn.functional as F
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss

# 项目内部：配置 + 数据集 + 模型
from config import (
    CHECKPOINT_DIR,
    CLASS_MAP,
    HIGH_RISK_EXTERNAL,
    RANDOM_SEED,
    TRAIN_CONFIG,
    TRAIN_OUTPUT_DIR,
)
from models.dataset import build_dataloaders
from models.multimodal import MultiModalEyeModel


# ── 日志配置：同时输出到控制台与文件（文件编码 utf-8，避免中文乱码） ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            TRAIN_OUTPUT_DIR / f"train_{time.strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8",
        ),
    ],
)
log = logging.getLogger("train")


def set_seed(seed: int = RANDOM_SEED) -> None:
    """固定随机种子，保证实验可复现。"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FocalLoss(torch.nn.Module):
    """
    带 label smoothing 的 Focal Loss：
        - focal term：(1 - pt)^gamma 用于抑制样本量过多样本的影响
        - label smoothing：把硬 one-hot 软化，防止模型过自信
    与 CrossEntropyLoss 对比：对类别不均衡更鲁棒。
    """

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.05):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        # 把 target 转为平滑的 one-hot
        smooth = self.label_smoothing / max(n_classes - 1, 1)
        one_hot = torch.full_like(log_probs, smooth).scatter_(
            -1, target.unsqueeze(-1), 1 - self.label_smoothing
        )
        pt = (one_hot * probs).sum(-1)
        log_pt = (one_hot * log_probs).sum(-1)
        loss = -(1 - pt).pow(self.gamma) * log_pt
        return loss.mean()


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """训练用：准确率，快速判断收敛。"""
    return (logits.argmax(-1) == labels).float().mean().item()


def _recall_at_classes(
    logits: torch.Tensor, labels: torch.Tensor, target_indices: list
) -> float:
    """
    在高风险类别子集中计算召回率（医疗场景最关键指标）：
        recall = 样本真实标签属于 target_indices 且被正确预测的比例。
    """
    if not target_indices:
        return 0.0
    pred = logits.argmax(-1)
    target_set = torch.tensor(target_indices, dtype=labels.dtype, device=labels.device)
    mask = torch.isin(labels, target_set)
    if mask.sum() == 0:
        return 0.0
    tp = ((pred == labels) & mask).sum().item()
    return tp / mask.sum().item()


def _macro_f1(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    """
    宏平均 F1：对每类单独计算 P / R / F1，再平均。
    适合类别不均衡场景，不会被大类别主导。
    """
    pred = logits.argmax(-1)
    f1_sum = 0.0
    valid_classes = 0
    for c in range(num_classes):
        tp = int(((pred == c) & (labels == c)).sum())
        fp = int(((pred == c) & (labels != c)).sum())
        fn = int(((pred != c) & (labels == c)).sum())
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        if precision + recall == 0:
            continue
        f1_sum += 2 * precision * recall / (precision + recall)
        valid_classes += 1
    return f1_sum / valid_classes if valid_classes else 0.0


def _run_epoch(model, loader, optimizer, scaler, device, cfg, meta, train: bool):
    """
    训练 / 验证一个 epoch：
        - train=True：启用反向传播 + 梯度裁剪 + AMP（仅在 CUDA 上）
        - train=False：仅前向传播，不更新权重
    返回：
        {"loss", "accuracy", "macro_f1", "high_risk_recall"} 各项在当前 epoch 的均值。
    """
    total_loss = 0.0
    total_acc = 0.0
    total_macro_f1 = 0.0
    total_high_risk_recall = 0.0
    n_batches = 0
    # 选择损失函数：FocalLoss 处理类别不均衡，否则用 CrossEntropyLoss（带 label smoothing）
    criterion = (
        FocalLoss(gamma=cfg["focal_gamma"], label_smoothing=cfg["label_smoothing"])
        if cfg["use_focal_loss"]
        else CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    )

    # train / eval 切换 dropout 与 BN
    model.train() if train else model.eval()

    for batch in loader:
        images, texts, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device)

        # 文本 tokenize（模型内部持有 tokenizer，可直接复用）
        input_ids, attention_mask = model.text_features_from_texts(texts)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        optimizer.zero_grad(set_to_none=True) if train else None
        # AMP（仅在 CUDA 上启用，提升训练速度与显存利用率）
        with autocast(enabled=device.type == "cuda"):
            out = model(images, input_ids, attention_mask)
            loss = criterion(out.disease_logits, labels)

        if train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()

        # 累计指标（batch 维度取平均）
        total_loss += loss.item()
        total_acc += _accuracy(out.disease_logits, labels)
        total_macro_f1 += _macro_f1(out.disease_logits, labels, meta.num_disease_classes)
        total_high_risk_recall += _recall_at_classes(
            out.disease_logits, labels, meta.high_risk_indices
        )
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": total_acc / max(n_batches, 1),
        "macro_f1": total_macro_f1 / max(n_batches, 1),
        "high_risk_recall": total_high_risk_recall / max(n_batches, 1),
    }


def save_metadata(checkpoint_dir: Path, meta) -> Path:
    """
    保存类别元数据（类别数量 / 名称 / 高风险索引 / 样本分布等），
    供推理 / 知识图谱模块使用。
    """
    payload = {
        "num_disease_classes": meta.num_disease_classes,
        "disease_class_keys": meta.disease_class_keys,
        "disease_class_names": meta.disease_class_names,
        "class_map": dict(zip(meta.disease_class_keys, meta.disease_class_names)),
        "high_risk_indices": meta.high_risk_indices,
        "high_risk_keys": [k for k in HIGH_RISK_EXTERNAL if k in set(meta.disease_class_keys)],
        "class_counts": meta.class_counts,
        "train_size": meta.train_size,
        "val_size": meta.val_size,
        "test_size": meta.test_size,
    }
    path = checkpoint_dir / "class_map.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def train_main() -> Path:
    """训练入口：构建数据 -> 初始化模型 -> 训练 + 早停 -> 在 test 上评估 -> 返回 best 权重路径。"""
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"设备: {device}")
    cfg = TRAIN_CONFIG
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 构建 train / val / test DataLoader
    log.info("加载数据集...")
    loaders, meta = build_dataloaders()
    meta_path = save_metadata(CHECKPOINT_DIR, meta)
    log.info(f"保存类别映射: {meta_path}")

    # 2) 初始化多模态模型（主干权重预训练）
    model = MultiModalEyeModel(
        num_disease_classes=meta.num_disease_classes,
        feature_dim=cfg["feature_dim"],
        dropout=cfg["dropout"],
        image_pretrained=True,
        text_pretrained=cfg["text_encoder"],
        text_max_length=cfg["text_max_length"],
    ).to(device)

    # 3) 优化器 / 学习率调度 / AMP 缩放器
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    scaler = GradScaler(enabled=device.type == "cuda")

    # 4) 训练 + 早停：在 val loss 上监测最佳
    best_val_loss = float("inf")
    best_path = CHECKPOINT_DIR / "best.pt"
    patience_counter = 0

    log.info("开始训练 %d 轮...", cfg["epochs"])
    for epoch in range(1, cfg["epochs"] + 1):
        train_metrics = _run_epoch(model, loaders["train"], optimizer, scaler, device, cfg, meta, train=True)
        scheduler.step()
        with torch.no_grad():
            val_metrics = _run_epoch(model, loaders["val"], optimizer, scaler, device, cfg, meta, train=False)

        log.info(
            f"Epoch {epoch:02d} | "
            f"train loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.3f} "
            f"f1={train_metrics['macro_f1']:.3f} hr_recall={train_metrics['high_risk_recall']:.3f} | "
            f"val loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.3f} "
            f"f1={val_metrics['macro_f1']:.3f} hr_recall={val_metrics['high_risk_recall']:.3f}"
        )

        # 保存 val loss 最佳权重（带类别映射，便于推理直接加载）
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_high_risk_recall": val_metrics["high_risk_recall"],
                    "num_disease_classes": meta.num_disease_classes,
                    "disease_class_keys": meta.disease_class_keys,
                    "disease_class_names": meta.disease_class_names,
                    "high_risk_indices": meta.high_risk_indices,
                    "train_config": cfg,
                },
                best_path,
            )
            log.info(f"  → 保存最佳权重: {best_path}")
        else:
            patience_counter += 1
            # 早停：连续 early_stop_patience 轮 val loss 未下降时停止，避免过拟合
            if patience_counter >= cfg["early_stop_patience"]:
                log.info(f"  → 早停（val loss 已 {cfg['early_stop_patience']} 轮未下降）")
                break

    # 5) 在 test 集上评估最佳模型并报告
    log.info("在 test 集上评估最佳模型...")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    with torch.no_grad():
        test_metrics = _run_epoch(model, loaders["test"], optimizer, scaler, device, cfg, meta, train=False)
    log.info(
        f"[TEST] loss={test_metrics['loss']:.4f} acc={test_metrics['accuracy']:.3f} "
        f"macro_f1={test_metrics['macro_f1']:.3f} high_risk_recall={test_metrics['high_risk_recall']:.3f}"
    )
    log.info("训练完成。")
    return best_path


if __name__ == "__main__":
    train_main()
