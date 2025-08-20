# -*- coding: utf-8 -*-
"""TransH implementation."""
from __future__ import annotations

import torch
from torch import nn
from .base import BaseKGEModel


class TransH(BaseKGEModel):
    """TransH：每个关系有法向量 w_r 与平移向量 d_r。实体需先投影到超平面。"""

    def __init__(self, num_entities: int, num_relations: int, ent_dim: int, p_norm: int = 1) -> None:
        super().__init__(num_entities, num_relations, ent_dim)
        assert p_norm in (1, 2)
        self.p_norm = p_norm
        # 法向量（需要单位化）
        self.norm_vec = nn.Embedding(num_relations, ent_dim)
        nn.init.xavier_uniform_(self.norm_vec.weight)

    def project(self, e: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # 将实体向量投影到关系超平面：e_perp = e - <e, w> * w
        # 先单位化 w
        w = torch.nn.functional.normalize(w, p=2, dim=1)
        dot = torch.sum(e * w, dim=1, keepdim=True)
        return e - dot * w

    def score_triples(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h_e = self.ent_emb(h)
        t_e = self.ent_emb(t)
        d_r = self.rel_emb(r)  # 平移向量
        w_r = self.norm_vec(r)  # 法向量
        h_p = self.project(h_e, w_r)
        t_p = self.project(t_e, w_r)
        diff = h_p + d_r - t_p
        if self.p_norm == 1:
            dist = torch.norm(diff, p=1, dim=1)
        else:
            dist = torch.norm(diff, p=2, dim=1)
        return -dist

    def relation_constraint(self) -> None:
        with torch.no_grad():
            self.norm_vec.weight.data = torch.nn.functional.normalize(
                self.norm_vec.weight.data, p=2, dim=1
            )