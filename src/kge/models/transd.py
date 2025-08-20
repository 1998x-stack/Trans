# -*- coding: utf-8 -*-
"""TransD implementation."""
from __future__ import annotations

import torch
from torch import nn
from .base import BaseKGEModel


class TransD(BaseKGEModel):
    """TransD：动态映射，实体/关系均有投影向量，构造 M_{r,e}。"""

    def __init__(self, num_entities: int, num_relations: int, ent_dim: int, rel_dim: int, p_norm: int = 1) -> None:
        super().__init__(num_entities, num_relations, ent_dim, rel_dim)
        assert p_norm in (1, 2)
        self.p_norm = p_norm
        # 实体与关系的投影向量
        self.ent_proj = nn.Embedding(num_entities, ent_dim)
        self.rel_proj = nn.Embedding(num_relations, rel_dim)
        nn.init.xavier_uniform_(self.ent_proj.weight)
        nn.init.xavier_uniform_(self.rel_proj.weight)

    def _project(self, e: torch.Tensor, e_p: torch.Tensor, r: torch.Tensor, r_p: torch.Tensor) -> torch.Tensor:
        # 论文公式：M_{r,e} = r_p * e_p^T + I；e' = M_{r,e} * e
        # 这里 I 的形状为 (rel_dim, ent_dim)；当 ent_dim != rel_dim 时仍可成立
        # 计算 r_p * e_p^T -> (B, rel_dim, ent_dim)
        B = e.shape[0]
        rp = r_p.unsqueeze(2)  # (B, rel_dim, 1)
        ep = e_p.unsqueeze(1)  # (B, 1, ent_dim)
        outer = torch.bmm(rp, ep)  # (B, rel_dim, ent_dim)
        # 构建 I，由于批量不同关系实体，使用广播法构造 batch-identity
        I = torch.zeros((B, self.rel_dim, self.ent_dim), device=e.device, dtype=e.dtype)
        for b in range(B):
            d = min(self.rel_dim, self.ent_dim)
            I[b, torch.arange(d), torch.arange(d)] = 1.0
        M = outer + I
        e = e.unsqueeze(2)  # (B, ent_dim, 1)
        proj = torch.bmm(M, e).squeeze(2)  # (B, rel_dim)
        return proj

    def score_triples(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h_e = self.ent_emb(h)
        t_e = self.ent_emb(t)
        r_e = self.rel_emb(r)
        h_pv = self.ent_proj(h)
        t_pv = self.ent_proj(t)
        r_pv = self.rel_proj(r)
        h_proj = self._project(h_e, h_pv, r, r_pv)
        t_proj = self._project(t_e, t_pv, r, r_pv)
        diff = h_proj + r_e - t_proj
        dist = torch.norm(diff, p=self.p_norm, dim=1)
        return -dist