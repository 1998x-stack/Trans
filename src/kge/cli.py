# -*- coding: utf-8 -*-
"""Command-line interface for KGE experiments."""
from __future__ import annotations

import argparse
import os
from loguru import logger

from .config import Config
from .ablation import run_ablation
from .train import fit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TransE-series Ablation on FB15k / FB15k-237",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_root", type=str, required=True, help="数据集根路径，包含 FB15k 和/或 FB15k-237 目录")
    parser.add_argument("--dataset_name", type=str, default="FB15k-237", choices=["FB15k", "FB15k-237"], help="选择数据集")
    parser.add_argument("--work_dir", type=str, default="./runs", help="输出目录")
    parser.add_argument("--device", type=str, default="cuda", help="设备，例如 cuda 或 cpu")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model", type=str, default="transe", choices=["transe", "transh", "transr", "transd", "transa"], help="单模型训练/评估")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--p_norm", type=int, default=1, choices=[1, 2])
    parser.add_argument("--neg_ratio", type=int, default=1)
    parser.add_argument("--reg_weight", type=float, default=0.0)

    parser.add_argument("--ent_dim", type=int, default=200)
    parser.add_argument("--rel_dim", type=int, default=200)

    parser.add_argument("--hits", type=str, default="1,3,10", help="评估 Hits@K 列表，逗号分隔")
    parser.add_argument("--save_ckpt", action="store_true")
    parser.add_argument("--log_every", type=int, default=200)

    parser.add_argument("--ablate", action="store_true", help="是否运行全量消融（全部模型×两个数据集）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hits_ks = [int(x) for x in args.hits.split(",") if x]
    cfg = Config(
        dataset_root=args.dataset_root,
        dataset_name=args.dataset_name,
        work_dir=args.work_dir,
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        margin=args.margin,
        p_norm=args.p_norm,
        neg_ratio=args.neg_ratio,
        reg_weight=args.reg_weight,
        ent_dim=args.ent_dim,
        rel_dim=args.rel_dim,
        hits_ks=hits_ks,
        model_name=args.model,
        save_ckpt=args.save_ckpt,
        log_every=args.log_every,
    )

    logger.add(os.path.join(cfg.work_dir, "master.log"), enqueue=True)

    if args.ablate:
        results = run_ablation(
            base_cfg=cfg,
            models=["transe", "transh", "transr", "transd", "transa"],
            datasets=["FB15k", "FB15k-237"],
        )
        logger.info(f"Ablation done: {results}")
    else:
        metrics = fit(cfg)
        logger.info(f"Training done. Best val metrics: {metrics}")


if __name__ == "__main__":
    main()