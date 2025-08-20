# -*- coding: utf-8 -*-
"""Ablation harness: compare TransE/H/R/D/A on FB15k & FB15k-237."""
from __future__ import annotations

from typing import Dict, List
import os
import json
from copy import deepcopy

from loguru import logger

from .config import Config
from .train import fit


def run_ablation(base_cfg: Config, models: List[str], datasets: List[str]) -> Dict[str, Dict[str, float]]:
    """在多个模型/数据集上运行实验，并返回结果字典。

    Args:
        base_cfg: 基础配置（将被复制，每次修改 model/dataset）。
        models: 模型名列表。
        datasets: 数据集名称列表（"FB15k" / "FB15k-237"）。
    Returns:
        结果字典：{"dataset/model": metrics_dict}
    """
    results: Dict[str, Dict[str, float]] = {}
    for ds_name in datasets:
        for m in models:
            cfg = deepcopy(base_cfg)
            cfg.dataset_name = ds_name
            cfg.model_name = m
            cfg.work_dir = os.path.join(base_cfg.work_dir, ds_name, m)
            os.makedirs(cfg.work_dir, exist_ok=True)
            logger.info(f"=== Ablation: dataset={ds_name} model={m} ===")
            metrics = fit(cfg)
            key = f"{ds_name}/{m}"
            results[key] = metrics
            # 保存单项结果
            with open(os.path.join(cfg.work_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
    # 保存总表
    summary_path = os.path.join(base_cfg.work_dir, "ablation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"📊 Ablation summary saved to {summary_path}")
    return results