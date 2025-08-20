# -*- coding: utf-8 -*-
"""Negative sampling utilities with Bernoulli trick.

实现负采样（head/tail 替换），并依据关系的多重性采用 Bernoulli 策略：
- 统计每个关系的 tph/hpt（tail-per-head / head-per-tail），
- 动态决定替换头或尾的概率，缓解一对多/多对一的训练偏差。
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import numpy as np

from .data import Triple


class BernoulliSampler:
    """基于 Bernoulli 策略的负采样器。"""

    def __init__(self, num_entities: int, train_triples: Sequence[Triple]) -> None:
        self.num_entities = num_entities
        self.train_triples = list(train_triples)
        self._build_stats()

    def _build_stats(self) -> None:
        """计算每个关系的 tph/hpt 并生成替换头部的概率 prob_replace_head。"""
        from collections import defaultdict

        tails_per_head = defaultdict(set)
        heads_per_tail = defaultdict(set)
        for h, r, t in self.train_triples:
            tails_per_head[(h, r)].add(t)
            heads_per_tail[(t, r)].add(h)
        tph = {}
        hpt = {}
        for _, r, _ in self.train_triples:
            # 计算 tph/hpt 的平均（注意防止除零）
            r_heads = [k for k in tails_per_head.keys() if k[1] == r]
            r_tails = [k for k in heads_per_tail.keys() if k[1] == r]
            tph[r] = (
                np.mean([len(tails_per_head[k]) for k in r_heads]) if r_heads else 0.0
            )
            hpt[r] = (
                np.mean([len(heads_per_tail[k]) for k in r_tails]) if r_tails else 0.0
            )
        self.prob_replace_head: Dict[int, float] = {}
        for r in set([tr[1] for tr in self.train_triples]):
            denom = tph[r] + hpt[r]
            p_h = tph[r] / denom if denom > 0 else 0.5
            self.prob_replace_head[r] = float(p_h)

    def corrupt_batch(
        self, batch: Sequence[Triple], neg_ratio: int, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """对一个 batch 做负采样，返回负样本的 (heads, rels, tails)。

        Args:
            batch: 正样本三元组序列。
            neg_ratio: 每个正样本生成的负样本数。
            rng: NumPy 随机数发生器（保证可重复）。
        Returns:
            neg_triples_h, neg_triples_t: 两组负样本（替换头 or 替换尾）。
        """
        bsz = len(batch)
        # 需要生成 bsz * neg_ratio 个负样本
        num_negs = bsz * neg_ratio

        # 展开正样本
        h = np.array([x[0] for x in batch], dtype=np.int64)
        r = np.array([x[1] for x in batch], dtype=np.int64)
        t = np.array([x[2] for x in batch], dtype=np.int64)

        # 按概率决定替换头或尾
        prob = np.array([self.prob_replace_head[int(ri)] for ri in r])
        # 重复到 neg 数量
        prob = np.repeat(prob, neg_ratio)
        r_rep = np.repeat(r, neg_ratio)
        choose_head = rng.random(num_negs) < prob

        # 随机采样实体
        corrupt_ents = rng.integers(low=0, high=self.num_entities, size=num_negs, dtype=np.int64)

        # 生成两组负样本数组
        neg_h = np.repeat(h, neg_ratio)
        neg_t = np.repeat(t, neg_ratio)
        # 替换对应的一侧
        neg_h[choose_head] = corrupt_ents[choose_head]
        neg_t[~choose_head] = corrupt_ents[~choose_head]

        return (
            np.stack([neg_h, r_rep, neg_t], axis=1).astype(np.int64),
            choose_head.astype(np.bool_),
        )