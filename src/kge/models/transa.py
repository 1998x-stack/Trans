# -*- coding: utf-8 -*-
"""TransA implementation (adaptive metric)."""
from __future__ import annotations

import torch
from torch import nn
from .base import BaseKGEModel


class TransA(BaseKGEModel):
    """TransA：引入关系特定的对角加权矩阵（简化版 A_r）。

    原论文中的自适应度量可能为半正定矩阵，这里采用对角矩阵近似，稳定且计算高效：
        score = - || diag(w_r) * (h + r - t) ||_1/2
    """

    def __init__(self, num_entities: int, num_relations: int, ent_dim: int, p_norm: int = 1) -> None:
        super().__init__(num_entities, num_relations, ent_dim)
        assert p_norm in (1, 2)
        self.p_norm = p_norm
        # 对角权重向量（要求非负），通过 softplus 保证
        self.diag_weight = nn.Embedding(num_relations, ent_dim)
        nn.init.zeros_(self.diag_weight.weight)
        self.softplus = nn.Softplus()

    def score_triples(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h_e = self.ent_emb(h)
        r_e = self.rel_emb(r)
        t_e = self.ent_emb(t)
        w = self.softplus(self.diag_weight(r))  # (B, D) >= 0
        diff = h_e + r_e - t_e
        weighted = w * diff
        dist = torch.norm(weighted, p=self.p_norm, dim=1)
        return -dist