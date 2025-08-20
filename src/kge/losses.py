# -*- coding: utf-8 -*-
"""Loss functions for KGE training."""
from __future__ import annotations

import torch
from torch import nn


def margin_ranking_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor, margin: float) -> torch.Tensor:
    """合页损失：max(0, m - (pos - neg))。

    Args:
        pos_scores: 正样本打分 (B,)
        neg_scores: 负样本打分 (B,)
        margin: 边界 m
    Returns:
        标量损失。
    """
    zeros = torch.zeros_like(pos_scores)
    loss = torch.mean(torch.maximum(zeros, margin - (pos_scores - neg_scores)))
    return loss