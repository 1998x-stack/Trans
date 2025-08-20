# -*- coding: utf-8 -*-
"""TransR implementation."""
from __future__ import annotations

import torch
from torch import nn
from .base import BaseKGEModel


class TransR(BaseKGEModel):
    """TransR：实体与关系在不同空间；每个关系有投影矩阵 M_r。"""

    def __init__(self, num_entities: int, num_relations: int, ent_dim: int, rel_dim: int, p_norm: int = 1) -> None:
        super().__init__(num_entities, num_relations, ent_dim, rel_dim)
        assert p_norm in (1, 2)
        self.p_norm = p_norm
        # 每个关系一个投影矩阵（rel_dim x ent_dim）
        self.proj = nn.Embedding(num_relations, rel_dim * ent_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        self.rel_dim = rel_dim
        self.ent_dim = ent_dim

    def _project(self, e: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        # e: (B, ent_dim); r: (B,)  -> (B, rel_dim)
        Mr = self.proj(r).view(-1, self.rel_dim, self.ent_dim)
        e = e.unsqueeze(2)  # (B, ent_dim, 1)
        proj_e = torch.bmm(Mr, e).squeeze(2)  # (B, rel_dim)
        return proj_e

    def score_triples(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h_e = self.ent_emb(h)
        t_e = self.ent_emb(t)
        r_e = self.rel_emb(r)
        h_p = self._project(h_e, r)
        t_p = self._project(t_e, r)
        diff = h_p + r_e - t_p
        dist = torch.norm(diff, p=self.p_norm, dim=1)
        return -dist