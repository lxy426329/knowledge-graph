r"""
外眼病多模态模型评估脚本。

运行：
    # 默认评估 checkpoints/best.pt（无权重则走 DEMO 模式）
    .venv\Scripts\python.exe -m evaluation.multimodal_eval
    # 指定权重
    .venv\Scripts\python.exe evaluation/multimodal_eval.py --checkpoint checkpoints/best.pt

核心指标：
    - accuracy:         外眼病细分类准确率
    - macro_f1:         外眼病分类宏 F1
    - high_risk_recall: 高风险外眼病召回率（医疗场景核心指标）
    - per_class:        每类 P / R / F1
    - confusion_matrix: 混淆矩阵（行=真实, 列=预测）

另附 "对比消融实验"：
    - 仅图像 (image_only): 把症状文本固定为空
    - 随机猜测 (random_baseline)
"""
from __future__ import annotations

# 标准库
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

# 第三方库
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# 把项目根目录加入 sys.path，便于脚本直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 项目内部（noqa: E402 禁用 import 顺序告警，因为 sys.path.insert 需要放在前面）
from config import CHECKPOINT_DIR, TRAIN_CONFIG  # noqa: E402
from models.dataset import (  # noqa: E402
    build_dataloaders,
    build_transforms,
)
from predict_module import ExternalEyePredictor  # noqa: E402


# ── 日志配置 ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("eval")


def _per_class_metrics(preds: np.ndarray, labels: np.ndarray, num_classes: int):
    """
    计算每类 precision / recall / f1 与混淆矩阵。
    - 混淆矩阵 cm[i, j]：真实为 i，模型预测为 j 的样本数
    - 某类无样本 / 无预测时 precision / recall / f1 记为 0，避免 0/0 除零
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(preds, labels):
        cm[int(t), int(p)] += 1

    per = []
    for c in range(num_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per.append({"precision": prec, "recall": rec, "f1": f1, "support": int(cm[c, :].sum())})
    return per, cm


def _eval_on_loader(predictor: ExternalEyePredictor, loader: DataLoader) -> Dict:
    """
    对一个 DataLoader 做批量推理，返回指标字典。
    推理流程：原始 tensor -> PIL -> predictor.predict_batch(images, texts)
             再把预测键名 -> 类别索引 -> 计算准确率 / F1 / 高风险召回率
    """
    preds_list: List[int] = []
    labels_list: List[int] = []

    for batch in loader:
        images, texts, labels = batch
        # PIL 化（模型内部的 mobile_preprocess 要求 PIL 输入）
        pil_images = []
        for x in images:
            arr = x.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            from PIL import Image as _Image
            pil_images.append(_Image.fromarray(arr))

        # 批量推理，再把预测键名转为类别索引
        results = predictor.predict_batch(pil_images, list(texts))
        preds_list += [predictor.class_keys.index(r.disease_key) for r in results]
        labels_list += labels.tolist()

    preds = np.asarray(preds_list)
    labels = np.asarray(labels_list)
    num_classes = predictor.num_classes

    accuracy = float((preds == labels).mean())
    per_class, cm = _per_class_metrics(preds, labels, num_classes)
    macro_f1 = float(np.mean([m["f1"] for m in per_class]))

    # 高风险召回率：真实标签为高风险类别样本中被正确识别的比例
    hr_mask = np.isin(labels, predictor.high_risk_indices)
    if hr_mask.sum() > 0:
        hr_tp = int(((preds == labels) & hr_mask).sum())
        high_risk_recall = hr_tp / int(hr_mask.sum())
    else:
        high_risk_recall = 0.0

    return {
        "num_samples": int(len(labels)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "high_risk_recall": float(high_risk_recall),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def _run_image_only_ablation(predictor: ExternalEyePredictor, loader: DataLoader) -> Dict:
    """消融：把所有症状文本置空（仅靠图像），观察图像模型本身的上限。"""
    preds_list: List[int] = []
    labels_list: List[int] = []

    for batch in loader:
        images, texts, labels = batch
        pil_images = []
        for x in images:
            arr = x.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            from PIL import Image as _Image
            pil_images.append(_Image.fromarray(arr))
        results = predictor.predict_batch(pil_images, ["" for _ in texts])
        preds_list += [predictor.class_keys.index(r.disease_key) for r in results]
        labels_list += labels.tolist()

    preds = np.asarray(preds_list)
    labels = np.asarray(labels_list)
    num_classes = predictor.num_classes

    accuracy = float((preds == labels).mean())
    per_class, _ = _per_class_metrics(preds, labels, num_classes)
    macro_f1 = float(np.mean([m["f1"] for m in per_class]))

    # 高风险召回率
    hr_mask = np.isin(labels, predictor.high_risk_indices)
    high_risk_recall = (
        float(((preds == labels) & hr_mask).sum() / hr_mask.sum()) if hr_mask.sum() else 0.0
    )
    return {
        "num_samples": int(len(labels)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "high_risk_recall": high_risk_recall,
    }


def _run_random_baseline(num_classes: int, num_samples: int, high_risk_indices: List[int]) -> Dict:
    """
    随机猜测 baseline（固定 seed=0 以便复现）：
    preds / labels 都从均匀分布 [0, num_classes) 采样，得到 "最差可达" 的上限。
    """
    rng = np.random.default_rng(0)
    preds = rng.integers(0, num_classes, size=num_samples)
    labels = rng.integers(0, num_classes, size=num_samples)
    per_class, _ = _per_class_metrics(preds, labels, num_classes)
    macro_f1 = float(np.mean([m["f1"] for m in per_class]))
    accuracy = float((preds == labels).mean())
    hr_mask = np.isin(labels, high_risk_indices)
    high_risk_recall = (
        float(((preds == labels) & hr_mask).sum() / hr_mask.sum()) if hr_mask.sum() else 0.0
    )
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "high_risk_recall": high_risk_recall,
    }


def main() -> None:
    """评估脚本入口：主评估 + 消融实验，最终保存 evaluation_report.json。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=int(TRAIN_CONFIG["image_size"]))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ablation", action="store_true", default=True, help="是否跑消融对比")
    args = parser.parse_args()

    # 1) 加载推理器（若 checkpoint 不存在，自动降级到 DEMO 模式）
    ckpt_path = args.checkpoint or str(Path(CHECKPOINT_DIR) / "best.pt")
    predictor = ExternalEyePredictor(weight_path=ckpt_path)

    # 2) 构建评估数据集（图像尺寸与训练保持一致）
    log.info("准备评估数据集 (data/archive_external) ...")
    loaders, meta = build_dataloaders(batch_size=args.batch_size, image_size=args.image_size)

    result: Dict = {
        "class_keys": predictor.class_keys,
        "class_names": predictor.class_names,
        "high_risk_indices": predictor.high_risk_indices,
        "high_risk_classes": [
            k for i, k in enumerate(predictor.class_keys) if i in predictor.high_risk_indices
        ],
        "checkpoint": ckpt_path,
        "has_trained_weights": Path(ckpt_path).exists(),
    }

    # 3) 主评估：train / val / test 三个 split 均测一次，便于观察过拟合
    for split in ("train", "val", "test"):
        log.info(f"评估 {split} ...")
        result[split] = _eval_on_loader(predictor, loaders[split])
        log.info(
            f"  {split} | acc={result[split]['accuracy']:.3f} "
            f"f1={result[split]['macro_f1']:.3f} "
            f"hr_recall={result[split]['high_risk_recall']:.3f}"
        )

    # 4) 消融实验（默认开启）：仅图像 / 随机猜测
    if args.ablation:
        log.info("消融实验：仅图像 (image_only) ...")
        result["ablation_image_only"] = _run_image_only_ablation(predictor, loaders["test"])
        log.info(
            f"  image_only | acc={result['ablation_image_only']['accuracy']:.3f} "
            f"f1={result['ablation_image_only']['macro_f1']:.3f} "
            f"hr_recall={result['ablation_image_only']['high_risk_recall']:.3f}"
        )

        log.info("消融实验：随机猜测 baseline ...")
        n_test = result["test"]["num_samples"]
        result["ablation_random_baseline"] = _run_random_baseline(
            num_classes=predictor.num_classes,
            num_samples=n_test,
            high_risk_indices=predictor.high_risk_indices,
        )

    # 5) 保存 JSON 报告（中文字段，便于团队协作查看）
    save_path = Path(CHECKPOINT_DIR) / "evaluation_report.json"
    save_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"评估报告已保存: {save_path}")


if __name__ == "__main__":
    main()
