# -*- coding: utf-8 -*-
"""Data loading and utilities for KGE datasets (FB15k / FB15k-237).

提供：
- KGDataset：读取三元组文件，建立实体/关系 id 映射；
- 辅助函数：构造过滤集合（filtered evaluation 需要）。

数据格式要求：train.txt / valid.txt / test.txt，每行: head <tab> relation <tab> tail
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple
import os


Triple = Tuple[int, int, int]  # (h_id, r_id, t_id)


@dataclass
class Vocab:
    """实体与关系词表。

    Attributes:
        ent2id: 实体到 id 的映射。
        rel2id: 关系到 id 的映射。
        id2ent: 反向实体映射。
        id2rel: 反向关系映射。
    """

    ent2id: Dict[str, int]
    rel2id: Dict[str, int]
    id2ent: List[str]
    id2rel: List[str]


class KGDataset:
    """知识图谱数据集加载器。

    - 自动扫描 root 下的 train/valid/test 文本文件；
    - 建立实体/关系映射；
    - 提供三元组 id 序列；
    - 构建 filtered 集（用于评估）。
    """

    def __init__(self, root: str) -> None:
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Dataset root not found: {root}")
        self.root = root
        self.train_path = os.path.join(root, "train.txt")
        self.valid_path = os.path.join(root, "valid.txt")
        self.test_path = os.path.join(root, "test.txt")
        for p in [self.train_path, self.valid_path, self.test_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing file: {p}")

        # 读取所有三元组，构建词表
        raw_train = self._read_raw(self.train_path)
        raw_valid = self._read_raw(self.valid_path)
        raw_test = self._read_raw(self.test_path)

        ents = sorted({h for h, _, _ in raw_train + raw_valid + raw_test} |
                      {t for _, _, t in raw_train + raw_valid + raw_test})
        rels = sorted({r for _, r, _ in raw_train + raw_valid + raw_test})
        ent2id = {e: i for i, e in enumerate(ents)}
        rel2id = {r: i for i, r in enumerate(rels)}
        id2ent = ents
        id2rel = rels
        self.vocab = Vocab(ent2id, rel2id, id2ent, id2rel)

        self.train: List[Triple] = [
            (ent2id[h], rel2id[r], ent2id[t]) for (h, r, t) in raw_train
        ]
        self.valid: List[Triple] = [
            (ent2id[h], rel2id[r], ent2id[t]) for (h, r, t) in raw_valid
        ]
        self.test: List[Triple] = [
            (ent2id[h], rel2id[r], ent2id[t]) for (h, r, t) in raw_test
        ]

        # filtered 集合：包含所有已知真三元组
        self.all_true: Set[Triple] = set(self.train) | set(self.valid) | set(self.test)

    @staticmethod
    def _read_raw(path: str) -> List[Tuple[str, str, str]]:
        """读取一列三元组文本。

        Args:
            path: 文本文件路径。
        Returns:
            (h, r, t) 的字符串三元组列表。
        """
        triples: List[Tuple[str, str, str]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 3:
                    raise ValueError(
                        f"Invalid triple format in {path}, line: {line}"
                    )
                triples.append((parts[0], parts[1], parts[2]))
        return triples

    # 便捷属性
    @property
    def num_entities(self) -> int:
        return len(self.vocab.id2ent)

    @property
    def num_relations(self) -> int:
        return len(self.vocab.id2rel)