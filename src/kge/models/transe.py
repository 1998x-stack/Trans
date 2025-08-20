# -*- coding: utf-8 -*-
"""TransE implementation (L1/L2)."""
from __future__ import annotations

import torch
from torch import nn
from .base import BaseKGEModel


class TransE(BaseKGEModel):
    """TransE: score = -||h + r - t||_p (we return negative distance as score)."""

    def __init__(self, num_entities: int, num_relations: int, ent_dim: int, p_norm: int = 1) -> None:
        super().__init__(num_entities, num_relations, ent_dim)
        assert p_norm in (1, 2), "p_norm must be 1 or 2"
        self.p_norm = p_norm

    def score_triples(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h_e = self.ent_emb(h)
        r_e = self.rel_emb(r)
        t_e = self.ent_emb(t)
        diff = h_e + r_e - t_e
        # 负距离作为分数（越大越好）
        if self.p_norm == 1:
            dist = torch.norm(diff, p=1, dim=1)
        else:
            dist = torch.norm(diff, p=2, dim=1)
        return -dist

    def relation_constraint(self) -> None:
        # 对关系也做单位约束有助于稳定（可选）
        with torch.no_grad():
            norms = torch.norm(self.rel_emb.weight.data, p=2, dim=1, keepdim=True) + 1e-9
            self.rel_emb.weight.data = self.rel_emb.weight.data / torch.clamp(norms, min=1.0)