# -*- coding: utf-8 -*-
"""One-click downloader for FB15k and FB15k-237.

注意：不同来源的压缩包结构可能略有差异，请根据需要调整解压与路径。
"""
from __future__ import annotations

import os
import zipfile
import io
from typing import Dict
import requests
from loguru import logger

# 公开镜像（示例）；请根据实际可用性更新
URLS: Dict[str, str] = {
    "FB15k": "https://data.deepai.org/FB15k.zip",
    "FB15k-237": "https://data.deepai.org/FB15k-237.zip",
}


def fetch_and_extract(name: str, root: str) -> None:
    if name not in URLS:
        raise ValueError(f"Unknown dataset: {name}")
    url = URLS[name]
    os.makedirs(root, exist_ok=True)
    logger.info(f"Downloading {name} from {url}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(resp.content))
    z.extractall(os.path.join(root, name))
    logger.info(f"Extracted to {os.path.join(root, name)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./datasets")
    parser.add_argument("--name", type=str, choices=["FB15k", "FB15k-237"], required=True)
    args = parser.parse_args()

    fetch_and_extract(args.name, args.root)