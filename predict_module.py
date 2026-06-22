"""
外眼病多模态推理接口。

主要类：
    ExternalEyePredictor: 对 (图像, 可选症状文本) 进行推理，
        输出标准化结果 PredictResult，含：
            - disease_key / disease_name（中文）/ confidence
            - risk_level / risk_reason (low / medium / high)
            - image_features / text_features / fused_features（供下游知识图谱/RAG 用）
            - all_probabilities（每类置信度）

        若无训练权重 checkpoints/best.pt，自动降级为 DEMO 模式
        （随机权重 + 随机但稳定的输出，方便联调）。

模块功能：
    - mobile_preprocess: 对手机实拍图做适配优化（模糊抑制、光线补偿、角度校正）
"""
from __future__ import annotations

# 标准库
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Union

# 第三方库
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms

# 项目内部
from config import (
    CLASS_MAP,
    CHECKPOINT_DIR,
    HIGH_RISK_EXTERNAL,
    TRAIN_CONFIG,
)

from models.multimodal import MultiModalEyeModel


# ── 常量区 ─────────────────────────────────────────────────
# 类别数 & 类别键（当 checkpoint 缺失时使用默认值，保证 DEMO 模式一致）
DEFAULT_NUM_CLASSES = 5
DEFAULT_CLASS_KEYS: List[str] = [
    "Cataract",        # 白内障
    "Conjunctivitis",  # 结膜炎
    "Eyelid",          # 眼睑疾病
    "Uveitis",         # 葡萄膜炎
    "Normal",          # 正常
]


@dataclass
class PredictResult:
    disease_key: str
    disease_name: str
    confidence: float
    all_probabilities: List[float]
    all_class_keys: List[str]
    all_class_names: List[str]
    risk_level: str
    risk_reason: str
    image_feature: List[float] = field(default_factory=list)
    text_feature: List[float] = field(default_factory=list)
    fused_feature: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_string(self, include_probabilities: bool = True) -> str:
        """
        把推理结果格式化为一段可读中文报告。

        示例：
            外眼病识别结果：结膜炎 (Conjunctivitis)
            置信度：82.3%
            风险等级：high
            风险说明：高风险外眼病（结膜炎），置信度较高，建议尽快就诊
            各类别概率：
              白内障   82.3%
              结膜炎    3.1%
              ...
        """
        lines: List[str] = []
        lines.append(f"外眼病识别结果：{self.disease_name} ({self.disease_key})")
        lines.append(f"置信度：{self.confidence * 100:.2f}%")
        lines.append(f"风险等级：{self.risk_level}")
        lines.append(f"风险说明：{self.risk_reason}")
        if include_probabilities:
            lines.append("各类别概率：")
            for name, p in zip(self.all_class_names, self.all_probabilities):
                lines.append(f"  {name:<6}  {p * 100:.2f}%")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_string(include_probabilities=True)


# ── 便捷入口：图片路径 -> 字符串 ─────────────────────────
_default_predictor = None


def _get_default_predictor():
    """延迟创建一个全局单例 predictor，避免重复加载模型。"""
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = ExternalEyePredictor()
    return _default_predictor


def predict_text(image_path: Union[str, Path], symptom_text: str = "") -> str:
    """
    输入一张图片路径字符串，输出一段外眼病识别报告字符串。

    用法：
        text = predict_text("data/archive_external/Conjunctivitis/xx.jpg")
        text = predict_text("phone_photo.jpg", symptom_text="左眼红肿3天")
    """
    predictor = _get_default_predictor()
    result = predictor.predict(image_path, symptom_text=symptom_text)
    return result.to_string()


def predict_json(image_path: Union[str, Path], symptom_text: str = "") -> str:
    """
    输入一张图片路径字符串，输出一段 JSON 格式字符串（含特征与概率）。

    用法：
        j = predict_json("data/archive_external/Conjunctivitis/xx.jpg", symptom_text="畏光流泪")
    """
    predictor = _get_default_predictor()
    result = predictor.predict(image_path, symptom_text=symptom_text)
    import json  # 延迟导入，避免未使用 warning
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)


# ── 手机实拍图自适应预处理 ─────────────────────────
def mobile_preprocess(image: Image.Image, image_size: int = 300) -> Image.Image:
    """
    对手机实拍图做适配优化：
        1. 轻微锐化，抑制模糊
        2. 自适应对比度/亮度，补偿光线不足
        3. 中心裁剪 + resize 到固定尺寸，减弱角度偏差
    """
    img = image.convert("RGB")
    # 锐化
    img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))
    # 对比度
    enhancer_c = ImageEnhance.Contrast(img)
    img = enhancer_c.enhance(1.15)
    # 亮度
    enhancer_b = ImageEnhance.Brightness(img)
    img = enhancer_b.enhance(1.05)
    # 色彩
    enhancer_col = ImageEnhance.Color(img)
    img = enhancer_col.enhance(1.05)
    # 中心裁剪成正方形
    w, h = img.size
    side = min(w, h)
    img = img.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2))
    try:
        resample = Image.Resampling.BILINEAR
    except AttributeError:
        resample = Image.BILINEAR  # Pillow < 10
    img = img.resize((image_size, image_size), resample)
    return img


def _build_eval_transform(image_size: int = TRAIN_CONFIG["image_size"]) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class ExternalEyePredictor:
    """外眼病多模态推理器。

    用法：
        predictor = ExternalEyePredictor()
        r = predictor.predict("xxx.jpg", symptom_text="眼睛红肿2天")
        print(r.disease_name, r.risk_level, r.confidence)
    """

    def __init__(self, weight_path: Optional[Union[str, Path]] = None, device=None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = int(TRAIN_CONFIG["image_size"])
        self.transform = _build_eval_transform(self.image_size)

        # 加载权重
        checkpoint_path = Path(weight_path) if weight_path else (Path(CHECKPOINT_DIR) / "best.pt")
        checkpoint = None
        if checkpoint_path.exists():
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                print(f"[predict] 加载权重: {checkpoint_path}")
            except Exception as exc:
                print(f"[predict] 加载权重失败: {exc}")

        if checkpoint is not None:
            num_classes = int(checkpoint.get("num_disease_classes", DEFAULT_NUM_CLASSES))
            class_keys = list(checkpoint.get("disease_class_keys", DEFAULT_CLASS_KEYS))
            high_risk_indices = list(checkpoint.get("high_risk_indices", []))
        else:
            print("[predict] 未找到训练权重，启用 DEMO 模式（便于联调）。")
            num_classes = DEFAULT_NUM_CLASSES
            class_keys = list(DEFAULT_CLASS_KEYS)
            high_risk_indices = [
                i for i, k in enumerate(class_keys) if k in HIGH_RISK_EXTERNAL
            ]

        self.num_classes = num_classes
        self.class_keys = class_keys
        self.class_names = [CLASS_MAP.get(k, k) for k in class_keys]
        self.high_risk_indices = high_risk_indices

        self.model = MultiModalEyeModel(
            num_disease_classes=num_classes,
            feature_dim=int(TRAIN_CONFIG["feature_dim"]),
            dropout=0.0,
            image_pretrained=(checkpoint is None),
            text_pretrained=TRAIN_CONFIG["text_encoder"],
            text_max_length=int(TRAIN_CONFIG["text_max_length"]),
        ).to(self.device).eval()

        if checkpoint is not None:
            try:
                self.model.load_state_dict(checkpoint["state_dict"])
            except Exception as exc:
                print(f"[predict] 权重不完全匹配，尝试部分加载: {exc}")
                self.model.load_state_dict(checkpoint["state_dict"], strict=False)

    # ── 风险等级 ─────────────────────────
    def _risk_for(self, idx: int, confidence: float, probs: np.ndarray) -> (str, str):
        is_high = idx in self.high_risk_indices
        top2_diff = float(np.sort(probs)[-1] - np.sort(probs)[-2]) if len(probs) >= 2 else 1.0
        if is_high and confidence > 0.5:
            return (
                "high",
                f"高风险外眼病（{self.class_names[idx]}），置信度较高，建议尽快就诊",
            )
        if is_high and confidence > 0.35:
            return (
                "medium",
                f"疑似高风险外眼病（{self.class_names[idx]}），建议线下复查",
            )
        if confidence < 0.35 or top2_diff < 0.08:
            return "medium", "模型判断存在不确定性，建议线下复查"
        return "low", f"{self.class_names[idx]}，注意观察即可"

    # ── 主推理 ───────────────────────────
    @torch.no_grad()
    def predict(self, image: Union[Image.Image, str, Path], symptom_text: str = "") -> PredictResult:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")

        # 手机实拍图适配预处理
        image = mobile_preprocess(image, image_size=self.image_size)

        x = self.transform(image).unsqueeze(0).to(self.device)
        text = (symptom_text or "").strip() or "无明显症状"
        input_ids, mask = self.model.text_features_from_texts([text])
        out = self.model(x, input_ids.to(self.device), mask.to(self.device))

        probs = F.softmax(out.disease_logits, dim=-1)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        risk_level, risk_reason = self._risk_for(idx, confidence, probs)

        return PredictResult(
            disease_key=self.class_keys[idx],
            disease_name=self.class_names[idx],
            confidence=confidence,
            all_probabilities=[float(v) for v in probs],
            all_class_keys=list(self.class_keys),
            all_class_names=list(self.class_names),
            risk_level=risk_level,
            risk_reason=risk_reason,
            image_feature=out.image_features[0].cpu().numpy().tolist(),
            text_feature=out.text_features[0].cpu().numpy().tolist(),
            fused_feature=out.fused_features[0].cpu().numpy().tolist(),
        )

    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Image.Image],
        texts: Optional[List[str]] = None,
    ) -> List[PredictResult]:
        """
        批量推理：
            images: PIL 图像列表（与 texts 一一对应）
            texts:  症状文本列表，None 时视为全部为空
        返回：每条图像一个 PredictResult
        """
        texts = texts or [""] * len(images)
        # 对症状文本做标准化：None -> ""；空字符串 -> "无明显症状"（便于 BERT 得到稳定表示）
        texts = [(t or "").strip() or "无明显症状" for t in texts]

        # 批量预处理：手机实拍图适配 -> 统一 tensor batch
        processed = [mobile_preprocess(im.convert("RGB"), image_size=self.image_size) for im in images]
        batch = torch.stack([self.transform(im) for im in processed]).to(self.device)

        # 文本编码（复用模型内部 tokenizer）
        input_ids, mask = self.model.text_features_from_texts(texts)
        input_ids = input_ids.to(self.device)
        mask = mask.to(self.device)

        out = self.model(batch, input_ids, mask)
        probs = F.softmax(out.disease_logits, dim=-1).cpu().numpy()
        img_feat = out.image_features.cpu().numpy()
        txt_feat = out.text_features.cpu().numpy()
        fused_feat = out.fused_features.cpu().numpy()

        results: List[PredictResult] = []
        for i in range(len(images)):
            idx = int(np.argmax(probs[i]))
            confidence = float(probs[i, idx])
            risk_level, risk_reason = self._risk_for(idx, confidence, probs[i])
            results.append(PredictResult(
                disease_key=self.class_keys[idx],
                disease_name=self.class_names[idx],
                confidence=confidence,
                all_probabilities=[float(v) for v in probs[i]],
                all_class_keys=list(self.class_keys),
                all_class_names=list(self.class_names),
                risk_level=risk_level,
                risk_reason=risk_reason,
                image_feature=img_feat[i].tolist(),
                text_feature=txt_feat[i].tolist(),
                fused_feature=fused_feat[i].tolist(),
            ))
        return results
