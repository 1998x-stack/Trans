# Trans
FB15k / FB15k-237 上对 TransE / TransH / TransR / TransD / TransA 的系统对比与消融实验代码（工业级结构、PEP8/257、类型标注、loguru 日志、可扩展）。

# TransE-Series-Ablation-KGE

- 数据集：FB15k / FB15k-237（`scripts/download_datasets.py` 一键下载）
- 模型：TransE / TransH / TransR / TransD / TransA
- 训练：margin ranking + Bernoulli 负采样 + 过滤评估（MR/MRR/Hits@K）

## 快速开始
```bash
# 1) 安装依赖（建议 Conda / venv）
pip install -r requirements.txt

# 2) 下载数据
python scripts/download_datasets.py --root ./datasets --name FB15k
python scripts/download_datasets.py --root ./datasets --name FB15k-237

# 3) 单模型训练（示例）
PYTHONPATH=. python -m src.kge.cli \
  --dataset_root ./datasets \
  --dataset_name FB15k-237 \
  --model transe \
  --epochs 100 --batch_size 1024 --lr 1e-3 --margin 1.0 --p_norm 1 --neg_ratio 1 \
  --ent_dim 200 --rel_dim 200 --save_ckpt

# 4) 全量消融
PYTHONPATH=. python -m src.kge.cli \
  --dataset_root ./datasets \
  --work_dir ./runs \
  --ablate
````

## 结果文件

* `runs/<dataset>/<model>/metrics.json`：最优验证指标
* `runs/ablation_summary.json`：总汇表

## 设计说明

* 代码遵循 PEP 8/257；中文注释覆盖关键步骤；loguru 追踪训练进度；
* 模型实现保持一致接口，便于横向对比与后续扩展（如 RotatE、ComplEx）。