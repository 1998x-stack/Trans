# -*- coding: utf-8 -*-
"""轻量级冒烟测试，确保主要模块能正常导入与简单跑通一个 batch。"""
from __future__ import annotations

import os
import torch

from src.kge.config import Config
from src.kge.data import KGDataset
from src.kge.train import build_model


def test_import_and_forward(tmp_path):
    # 构造一个极小的伪数据集
    root = tmp_path / "FB15k-237"
    root.mkdir(parents=True, exist_ok=True)
    for split in ["train", "valid", "test"]:
        with open(root / f"{split}.txt", "w", encoding="utf-8") as f:
            f.write("e1\tr1\te2\n")
            f.write("e2\tr1\te3\n")
    ds = KGDataset(str(tmp_path))  # 注意：构造函数期待的是父目录
    cfg = Config(dataset_root=str(tmp_path), model_name="transe")

    model = build_model(cfg, ds)
    h, r, t = torch.tensor([0, 1]), torch.tensor([0, 0]), torch.tensor([1, 2])
    out = model.score_triples(h, r, t)
    assert out.shape == (2,)