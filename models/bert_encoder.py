"""
文本特征提取器：基于 BERT（默认 bert-base-chinese）。
对外暴露 (feature, logits)：
    - feature：投影到 feature_dim 的文本特征，供多模态融合模块使用
    - logits：文本分类基线的分类头，可用于纯文本任务 / 对比训练
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class BertTextEncoder(nn.Module):
    """
    BERT 文本编码器：
        - tokenizer：调用 AutoTokenizer.from_pretrained(pretrained)
        - bert：调用 AutoModel.from_pretrained(pretrained)
        - 投影：Linear(768, feature_dim) + LayerNorm + ReLU + Dropout
        - 分类头：Linear(feature_dim, num_classes)
    forward(input_ids, attention_mask) -> (feature, logits)
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: str = "bert-base-chinese",
        feature_dim: int = 512,
        max_length: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        # 分词器与 BERT 主干（默认在初始化时下载预训练权重，需要网络）
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.bert = AutoModel.from_pretrained(pretrained)
        hidden_size = self.bert.config.hidden_size  # bert-base: 768

        # 投影：把 BERT 的 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> 向量投影到 feature_dim
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        # 文本分类头（用于纯文本基线 / 多任务训练）
        self.classifier = nn.Linear(feature_dim, num_classes)

    def tokenize(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对 List[str] 做 tokenize，返回 (input_ids, attention_mask)，
        便于训练 / 推理时直接 forward。
        """
        enc = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 取 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> 向量作为句表示（BERT 文本分类的标准做法）
        pooled = out.last_hidden_state[:, 0, :]
        feat = self.proj(pooled)        # [B, feature_dim]
        logits = self.classifier(feat)  # [B, num_classes]
        return feat, logits
