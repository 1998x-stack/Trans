# -*- coding: utf-8 -*-
"""Configuration utilities for KGE experiments.

本模块提供：
1) Config 数据类：集中管理超参数。
2) 从 YAML/字典 构建 Config。

遵循 PEP 8/257 与类型标注；包含中文注释以便团队协作与维护。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml


@dataclass
class Config:
    """全局配置数据类（可序列化、便于记录实验）。

    Attributes:
        dataset_root: 数据集根目录，包含 FB15k 或 FB15k-237 的 train/valid/test 文本文件。
        dataset_name: 数据集名称，"FB15k" 或 "FB15k-237"。
        work_dir: 输出目录（日志、ckpt、结果）。
        seed: 随机种子。
        device: 训练设备，例如 "cuda" 或 "cpu"。

        # 训练相关
        epochs: 训练轮数。
        batch_size: 批大小。
        lr: 学习率。
        margin: 合页损失（margin ranking loss）的边界 m。
        p_norm: TransE/TransH 等的范数类型（1 或 2）。
        neg_ratio: 负采样比例（负样本/正样本）。
        reg_weight: L2 正则权重。

        # 嵌入维度
        ent_dim: 实体向量维度。
        rel_dim: 关系向量维度（TransR/TransD 可不同于 ent_dim）。

        # 评估
        eval_batch_size: 评估时的批大小。
        hits_ks: 计算 Hits@K 的列表。

        # 运行
        model_name: 要训练/评估的模型名（transe/transh/transr/transd/transa）。
        save_ckpt: 是否保存模型参数。
        log_every: 日志打印频率（以 step 计）。
    """

    dataset_root: str
    dataset_name: str = "FB15k-237"
    work_dir: str = "./runs"
    seed: int = 42
    device: str = "cuda"

    epochs: int = 200
    batch_size: int = 1024
    lr: float = 0.001
    margin: float = 1.0
    p_norm: int = 1
    neg_ratio: int = 1
    reg_weight: float = 0.0

    ent_dim: int = 200
    rel_dim: int = 200

    eval_batch_size: int = 256
    hits_ks: List[int] = field(default_factory=lambda: [1, 3, 10])

    model_name: str = "transe"
    save_ckpt: bool = True
    log_every: int = 200

    extra: Dict[str, float] = field(default_factory=dict)  # 预留扩展字段

    @staticmethod
    def from_yaml(path: str) -> "Config":
        """从 YAML 文件读取配置。

        Args:
            path: YAML 文件路径。
        Returns:
            Config 实例。
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return Config(**data)

    @staticmethod
    def from_dict(data: Dict) -> "Config":
        """从字典构建配置。"""
        return Config(**data)