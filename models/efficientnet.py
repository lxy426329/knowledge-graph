"""
图像特征提取器：EfficientNet-B3（torchvision 预训练权重）。
输出一个固定维度的特征向量（默认 512），供多模态融合模块使用；
同时暴露分类头，便于单独训练图像基线模型。
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models


class EfficientNetImageEncoder(nn.Module):
    """
    EfficientNet-B3 图像编码器：
        - 主干：ImageNet 预训练的 EfficientNet-B3 特征网络（features + avgpool）
        - 投影头：Linear(feature_dim, feature_dim) + BN + ReLU，降维到 feature_dim
        - 分类头：Linear(feature_dim, num_classes)，用于图像分类损失
    forward(x): 返回 (feature_vector, logits)
    """

    def __init__(self, num_classes: int, feature_dim: int = 512, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        # 加载 torchvision EfficientNet-B3（默认使用 ImageNet 预训练权重）
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b3(weights=weights)

        # B3 分类头前的通道数为 1536，用一个线性层压缩到 feature_dim
        in_features = backbone.classifier[1].in_features  # 1536

        # 特征提取部分：主干特征 -> 全局均值池化 -> 展平 -> dropout -> 线性投影 -> BN -> ReLU
        self.features = nn.Sequential(
            backbone.features,
            backbone.avgpool,
            nn.Flatten(1),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )
        # 细分类头：从 feature_dim -> num_classes
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(x)          # [B, feature_dim]，用于多模态融合
        logits = self.classifier(feat)   # [B, num_classes]，用于图像分类损失
        return feat, logits
