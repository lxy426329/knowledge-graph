"""
项目全局配置 — 外眼多模态模型（EfficientNet-B3 + BERT）。

新链路：
    数据 → 训练 → 推理 → API → 评估
其他模块应从这里导入，避免各自维护重复常量。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List


# ── 提前设置 HuggingFace 镜像源（必须在首次 import transformers 之前） ──
HF_MIRROR = "https://hf-mirror.com"
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = HF_MIRROR


# ── 根目录与数据目录 ──────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
ANNOTATIONS_DIR: Path = DATA_DIR / "annotations"
EXTERNAL_EYE_DATA_DIR: Path = DATA_DIR / "archive_external"


# ── 统一的图像扩展名（大小写不敏感，匹配时使用 lower()） ─────
IMAGE_EXTS: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")


# ── 训练配置（EfficientNet-B3 + BERT 多模态联合训练） ───────
TRAIN_CONFIG: Dict = {
    "image_size": 300,
    "num_workers": 2,
    "epochs": 20,
    "batch_size": 16,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "label_smoothing": 0.05,
    "lr_scheduler": "cosine",
    "grad_clip_norm": 5.0,
    "early_stop_patience": 5,
    "dropout": 0.3,
    "feature_dim": 512,
    "use_focal_loss": True,
    "focal_gamma": 2.0,
    "text_encoder": "bert-base-chinese",
    "text_max_length": 128,
}

# 训练产物输出目录
TRAIN_OUTPUT_DIR: Path = BASE_DIR / "runs"
CHECKPOINT_DIR: Path = BASE_DIR / "checkpoints"
MODEL_OUTPUT_DIR: Path = BASE_DIR / "models"

# 数据集划分比例
DATA_SPLIT: Dict[str, float] = {"train": 0.7, "val": 0.15, "test": 0.15}
RANDOM_SEED: int = 42


# ── 外眼病类别体系 ───────────────────────────────────────────
# 外眼病细分类目：白内障 / 结膜炎 / 眼睑疾病 / 葡萄膜炎 + 正常
# 参考数据规模（合计 2298 张）：
#   Cataract      544 张   （白内障）
#   Conjunctivitis 357 张  （结膜炎）
#   Eyelid        525 张   （眼睑疾病）
#   Normal        649 张   （正常）
#   Uveitis       223 张   （葡萄膜炎）
EXTERNAL_EYE_SUBCLASS: Dict[str, str] = {
    "Cataract": "白内障",
    "Conjunctivitis": "结膜炎",
    "Eyelid": "眼睑疾病",
    "Uveitis": "葡萄膜炎",
}

CLASS_MAP: Dict[str, str] = {
    **EXTERNAL_EYE_SUBCLASS,
    "Normal": "正常",
}

# 疾病中文描述，用于报告 / API 返回结果
DISEASE_DESC: Dict[str, str] = {
    "Cataract":       "白内障 — 晶状体混浊，视力进行性下降（需手术干预）",
    "Conjunctivitis": "结膜炎 — 眼结膜发炎，眼睛红肿分泌物增多",
    "Eyelid":         "眼睑疾病 — 含麦粒肿/霰粒肿/睑缘炎等眼睑病变",
    "Normal":         "正常 — 眼睛健康，无异常",
    "Uveitis":        "葡萄膜炎 — 眼球葡萄膜层炎症，高风险（需及时就医）",
}

# 高风险外眼病（召回率需要特别关注）：白内障 / 结膜炎 / 眼睑疾病 / 葡萄膜炎
HIGH_RISK_EXTERNAL: List[str] = ["Cataract", "Conjunctivitis", "Eyelid", "Uveitis"]

# 症状文本 JSON：{"image_file_name": "患者自述的症状文本"}
SYMPTOM_JSON: Path = ANNOTATIONS_DIR / "symptoms.json"


# ── API 配置 ───────────────────────────────────────────────────
API_CONFIG: Dict = {
    "host": "0.0.0.0",
    "port": 8000,
    "title": "外眼识别 API",
    "description": "基于 EfficientNet-B3 + BERT 的外眼疾病多模态识别 API（白内障/结膜炎/眼睑疾病/葡萄膜炎/正常）",
    "version": "1.0.0",
}


# ── 创建必要的目录 ────────────────────────────────────────────
for directory in (ANNOTATIONS_DIR, EXTERNAL_EYE_DATA_DIR,
                  TRAIN_OUTPUT_DIR, CHECKPOINT_DIR, MODEL_OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)
