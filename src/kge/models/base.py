# -*- coding: utf-8 -*-
"""Base classes for KGE models (TransE-family)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch
from torch import nn


class BaseKGEModel(nn.Module, ABC):
    """KGE 模型基类：统一 score 接口与约束投影。"""

    def __init__(self, num_entities: int, num_relations: int, ent_dim: int, rel_dim: Optional[int] = None) -> None:
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.ent_dim = ent_dim
        self.rel_dim = rel_dim if rel_dim is not None else ent_dim

        # 实体/关系嵌入
        self.ent_emb = nn.Embedding(num_entities, ent_dim)
        self.rel_emb = nn.Embedding(num_relations, self.rel_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 使用均匀初始化，保证尺度合适；随后可在训练中进行单位化约束
        nn.init.uniform_(self.ent_emb.weight, -6 / self.ent_dim ** 0.5, 6 / self.ent_dim ** 0.5)
        nn.init.uniform_(self.rel_emb.weight, -6 / self.rel_dim ** 0.5, 6 / self.rel_dim ** 0.5)

    @abstractmethod
    def score_triples(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """计算给定三元组的打分（越大越好）。"""

    def entity_constraint(self) -> None:
        """默认实体投影到单位球，避免范数无限增大。"""
        with torch.no_grad():
            self.ent_emb.weight.data = torch.clamp(self.ent_emb.weight.data, min=-2.0, max=2.0)
            norms = torch.norm(self.ent_emb.weight.data, p=2, dim=1, keepdim=True) + 1e-9
            self.ent_emb.weight.data = self.ent_emb.weight.data / torch.clamp(norms, min=1.0)

    def relation_constraint(self) -> None:
        """某些模型需要关系约束；基类默认不操作。"""
        return

    def regularization(self) -> torch.Tensor:
        """L2 正则（可加权到损失），防止过拟合。"""
        return (
            torch.mean(self.ent_emb.weight.pow(2)) + torch.mean(self.rel_emb.weight.pow(2))
        )