"""
外眼病多模态分类模型：
    图像 (EfficientNet-B3) + 文本 (BERT) -> 门控融合 -> 细分类头
对外统一输出 ModelOutput：
    - disease_logits：[B, num_disease_classes]，细分类 logits
    - image_features / text_features / fused_features：[B, feature_dim]
                        （L2 归一化，便于下游知识图谱 / 向量检索使用）

注意：内外眼二分类已移除，当前仅输出外眼病细分。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bert_encoder import BertTextEncoder
from models.efficientnet import EfficientNetImageEncoder


@dataclass
class ModelOutput:
    """多模态模型统一输出，便于训练 / 推理 / 评估统一使用。"""
    disease_logits: torch.Tensor       # [B, num_disease_classes]
    image_features: torch.Tensor       # [B, feature_dim] 图像特征（L2 归一化）
    text_features: torch.Tensor        # [B, feature_dim] 文本特征（L2 归一化）
    fused_features: torch.Tensor       # [B, feature_dim] 融合特征（L2 归一化）


class MultiModalEyeModel(nn.Module):
    """外眼病多模态模型：图像 + 文本 -> 门控融合 -> 细分类头。"""

    def __init__(
        self,
        num_disease_classes: int,
        feature_dim: int = 512,
        dropout: float = 0.3,
        image_pretrained: bool = True,
        text_pretrained: str = "bert-base-chinese",
        text_max_length: int = 128,
    ) -> None:
        super().__init__()
        # 图像编码器（EfficientNet-B3，ImageNet 预训练）
        self.image_encoder = EfficientNetImageEncoder(
            num_classes=num_disease_classes,
            feature_dim=feature_dim,
            dropout=dropout,
            pretrained=image_pretrained,
        )
        # 文本编码器（bert-base-chinese，HuggingFace 预训练）
        self.text_encoder = BertTextEncoder(
            num_classes=num_disease_classes,
            pretrained=text_pretrained,
            feature_dim=feature_dim,
            max_length=text_max_length,
        )

        # ── 融合头：门控权重 + 投影 ─────────────────────────────────
        # 门控权重：sigmoid(Linear([img_feat; txt_feat]))，值域 [0,1]
        self.fusion_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid(),
        )
        # 投影：把 [gate*img_feat; (1-gate)*txt_feat] 投影到 feature_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # 外眼病细分分类头（直接用融合特征预测）
        self.disease_head = nn.Linear(feature_dim, num_disease_classes)

    @property
    def tokenizer(self):
        """暴露 BERT tokenizer，供训练 / 推理外部直接 tokenize。"""
        return self.text_encoder.tokenizer

    def text_features_from_texts(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """便捷方法：List[str] -> (input_ids, attention_mask)。"""
        return self.text_encoder.tokenize(texts)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> ModelOutput:
        # 提取图像 / 文本特征（各自的 logits 用于双任务训练，但主损失仍走融合头）
        image_feat, _image_logits = self.image_encoder(images)
        text_feat, _text_logits = self.text_encoder(input_ids, attention_mask)

        # 计算门控权重，融合图像 / 文本特征
        gate = self.fusion_gate(torch.cat([image_feat, text_feat], dim=-1))
        fused = self.fusion_proj(torch.cat([image_feat * gate, text_feat * (1 - gate)], dim=-1))

        # L2 归一化，便于下游知识图谱 / 向量检索 / 余弦相似度
        image_feat = F.normalize(image_feat, p=2, dim=-1)
        text_feat = F.normalize(text_feat, p=2, dim=-1)
        fused_feat = F.normalize(fused, p=2, dim=-1)

        return ModelOutput(
            disease_logits=self.disease_head(fused),
            image_features=image_feat,
            text_features=text_feat,
            fused_features=fused_feat,
        )
