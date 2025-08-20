# -*- coding: utf-8 -*-
"""Trainer and evaluator for KGE models (filtered setting)."""
from __future__ import annotations

from typing import Dict, List, Tuple, Type
import os
import time

import numpy as np
import torch
from torch import nn, optim
from loguru import logger

from .config import Config
from .data import KGDataset, Triple
from .sampling import BernoulliSampler
from .losses import margin_ranking_loss
from .utils import batchify, metrics_ranking, save_checkpoint, set_seed
from .models.transe import TransE
from .models.transh import TransH
from .models.transr import TransR
from .models.transd import TransD
from .models.transa import TransA


MODEL_REGISTRY = {
    "transe": TransE,
    "transh": TransH,
    "transr": TransR,
    "transd": TransD,
    "transa": TransA,
}


def build_model(cfg: Config, ds: KGDataset) -> nn.Module:
    """根据配置实例化模型。"""
    if cfg.model_name == "transe":
        return TransE(ds.num_entities, ds.num_relations, cfg.ent_dim, p_norm=cfg.p_norm)
    if cfg.model_name == "transh":
        return TransH(ds.num_entities, ds.num_relations, cfg.ent_dim, p_norm=cfg.p_norm)
    if cfg.model_name == "transr":
        return TransR(ds.num_entities, ds.num_relations, cfg.ent_dim, cfg.rel_dim, p_norm=cfg.p_norm)
    if cfg.model_name == "transd":
        return TransD(ds.num_entities, ds.num_relations, cfg.ent_dim, cfg.rel_dim, p_norm=cfg.p_norm)
    if cfg.model_name == "transa":
        return TransA(ds.num_entities, ds.num_relations, cfg.ent_dim, p_norm=cfg.p_norm)
    raise ValueError(f"Unknown model: {cfg.model_name}")


def train_one_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    ds: KGDataset,
    sampler: BernoulliSampler,
    cfg: Config,
    epoch: int,
    rng: np.random.Generator,
) -> float:
    """单轮训练，返回平均损失。"""
    model.train()
    total_loss = 0.0
    n_steps = 0

    for step, batch in enumerate(batchify(ds.train, cfg.batch_size, shuffle=True, seed=cfg.seed + epoch)):
        # 正样本
        h = torch.tensor([x[0] for x in batch], dtype=torch.long, device=cfg.device)
        r = torch.tensor([x[1] for x in batch], dtype=torch.long, device=cfg.device)
        t = torch.tensor([x[2] for x in batch], dtype=torch.long, device=cfg.device)
        pos_scores = model.score_triples(h, r, t)

        # 负样本（Bernoulli 选择替换头/尾）
        neg_triples, choose_head = sampler.corrupt_batch(batch, cfg.neg_ratio, rng)
        neg_triples = torch.tensor(neg_triples, dtype=torch.long, device=cfg.device)
        hn, rn, tn = neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]
        neg_scores = model.score_triples(hn, rn, tn)

        # 合页损失 + L2 正则
        loss = margin_ranking_loss(
            pos_scores.repeat_interleave(cfg.neg_ratio), neg_scores, cfg.margin
        )
        if cfg.reg_weight > 0:
            loss = loss + cfg.reg_weight * model.regularization()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 约束投影（单位化）
        if hasattr(model, "entity_constraint"):
            model.entity_constraint()
        if hasattr(model, "relation_constraint"):
            model.relation_constraint()

        total_loss += float(loss.item())
        n_steps += 1
        if (step + 1) % cfg.log_every == 0:
            logger.info(f"epoch={epoch} step={step+1} loss={loss.item():.4f}")

    return total_loss / max(1, n_steps)


@torch.no_grad()
def evaluate_filtered(model: nn.Module, ds: KGDataset, cfg: Config, split: str = "valid") -> Dict[str, float]:
    """Filtered 评估（link prediction）：MR/MRR/Hits@K。

    对每个三元组 (h, r, t)，对 head 与 tail 分别做替换；对所有实体打分并排序，
    特别地，将已有真三元组过滤（除了当前查询 triple）。
    """
    model.eval()
    triples = getattr(ds, split)
    all_true = ds.all_true

    ranks: List[int] = []
    n_ents = ds.num_entities

    for i in range(0, len(triples), cfg.eval_batch_size):
        batch = triples[i : i + cfg.eval_batch_size]
        h = torch.tensor([x[0] for x in batch], dtype=torch.long, device=cfg.device)
        r = torch.tensor([x[1] for x in batch], dtype=torch.long, device=cfg.device)
        t = torch.tensor([x[2] for x in batch], dtype=torch.long, device=cfg.device)

        # 替换尾实体
        all_entities = torch.arange(n_ents, device=cfg.device)
        h_rep = h.unsqueeze(1).repeat(1, n_ents)
        r_rep = r.unsqueeze(1).repeat(1, n_ents)
        t_cand = all_entities.unsqueeze(0).repeat(h.shape[0], 1)
        scores = model.score_triples(h_rep.reshape(-1), r_rep.reshape(-1), t_cand.reshape(-1))
        scores = scores.view(h.shape[0], n_ents)
        # 过滤已知真三元组
        for bi, (hi, ri, ti) in enumerate(batch):
            for cand in range(n_ents):
                if cand == ti:
                    continue
                if (hi, ri, cand) in all_true:
                    scores[bi, cand] = -1e9  # 极小化，避免影响排名
        # 计算 rank
        true_scores = model.score_triples(h, r, t)
        # 排名（1-based）
        ranks_t = 1 + torch.sum((scores >= true_scores.unsqueeze(1)).int(), dim=1)
        ranks.extend([int(x) for x in ranks_t.tolist()])

        # 替换头实体
        t_rep = t.unsqueeze(1).repeat(1, n_ents)
        h_cand = all_entities.unsqueeze(0).repeat(t.shape[0], 1)
        scores = model.score_triples(h_cand.reshape(-1), r_rep.reshape(-1), t_rep.reshape(-1))
        scores = scores.view(t.shape[0], n_ents)
        for bi, (hi, ri, ti) in enumerate(batch):
            for cand in range(n_ents):
                if cand == hi:
                    continue
                if (cand, ri, ti) in all_true:
                    scores[bi, cand] = -1e9
        true_scores = model.score_triples(h, r, t)
        ranks_h = 1 + torch.sum((scores >= true_scores.unsqueeze(1)).int(), dim=1)
        ranks.extend([int(x) for x in ranks_h.tolist()])

    metrics = metrics_ranking(ranks, cfg.hits_ks)
    return metrics


def fit(cfg: Config) -> Dict[str, float]:
    """训练 + 验证评估；返回验证集指标。"""
    set_seed(cfg.seed)
    os.makedirs(cfg.work_dir, exist_ok=True)

    logger.add(os.path.join(cfg.work_dir, f"{cfg.model_name}.log"), enqueue=True)
    logger.info(f"🚀 Start training: model={cfg.model_name} dataset={cfg.dataset_name}")

    ds = KGDataset(os.path.join(cfg.dataset_root, cfg.dataset_name))
    model = build_model(cfg, ds).to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    sampler = BernoulliSampler(ds.num_entities, ds.train)
    rng = np.random.default_rng(cfg.seed)

    best_mrr = -1.0
    best_metrics: Dict[str, float] = {}

    for epoch in range(1, cfg.epochs + 1):
        avg_loss = train_one_epoch(model, optimizer, ds, sampler, cfg, epoch, rng)
        metrics = evaluate_filtered(model, ds, cfg, split="valid")
        logger.info(
            f"epoch={epoch} loss={avg_loss:.4f} | val MRR={metrics['MRR']:.4f} "
            + " ".join([f"H@{k}={metrics[f'Hits@{k}']:.4f}" for k in cfg.hits_ks])
        )
        if metrics["MRR"] > best_mrr:
            best_mrr = metrics["MRR"]
            best_metrics = metrics
            if cfg.save_ckpt:
                ckpt_path = os.path.join(cfg.work_dir, f"{cfg.model_name}_best.pt")
                save_checkpoint(ckpt_path, {"model": model.state_dict(), "cfg": cfg.__dict__})

    logger.info(f"✅ Best Val: {best_metrics}")
    return best_metrics