# -*- coding: utf-8 -*-
"""Utilities: seeding, batching, metrics, checkpoint, logging helpers."""
from __future__ import annotations

import os
import random
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from loguru import logger

from .data import Triple


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batchify(data: Sequence[Triple], batch_size: int, shuffle: bool = True, seed: int = 0):
    idx = np.arange(len(data))
    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(data), batch_size):
        sel = idx[i : i + batch_size]
        yield [data[j] for j in sel]


def save_checkpoint(path: str, state: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    logger.info(f"✅ Checkpoint saved to {path}")


def metrics_ranking(
    ranks: List[int], hits_ks: List[int]
) -> Dict[str, float]:
    """根据 rank 列表统计 MR / MRR / Hits@K。"""
    mr = float(np.mean(ranks))
    mrr = float(np.mean([1.0 / r for r in ranks]))
    metrics = {"MR": mr, "MRR": mrr}
    for k in hits_ks:
        metrics[f"Hits@{k}"] = float(np.mean([1.0 if r <= k else 0.0 for r in ranks]))
    return metrics