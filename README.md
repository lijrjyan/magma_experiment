# MAGMA Execution Workspace

GitHub 仓库（所有阶段都 push 到这里）：`git@github.com:lijrjyan/magma_experiment.git`

工作流（最重要）：`EXECUTION_PLAN.md` 里每完成一个阶段 → 立刻 `git commit` → `git push origin main`。没有 push 的步骤视为未完成。

## 这是什么

MAGMA（Manifold-Aware Geometric Aggregation）与 Dual Defense (DDFed) 的复现与对比实验工程。主线是将所有联邦学习实验（MNIST/FMNIST/CIFAR10/TinyImageNet × 各类攻击 × MAGMA vs 明文 baselines vs DDFed）放进同一个、可审计的流水线上，确保每阶段都能回滚。

核心特点：

- 统一入口 `scripts/run_exp.py` + YAML 配置（数据切分、模型、聚合器、攻击、FHE）；
- `magma/` 负责距离矩阵、层次聚类、jump ratio 与 largest component 聚合；
- `baselines/` 提供 FedAvg / Krum / Median / TrimmedMean / ClipMedian / CosDefense / DDFed；
- `attacks/` 覆盖 IPM / ALIE / Scaling / Label Flip；
- `results/<run_id>/` 记录 `logs/`、`tensorboard/`、`artifacts/{metrics.jsonl,summary.json,resolved_config.yaml}`。

## 文档入口

- `EXECUTION_PLAN.md`：唯一执行清单；Step/Stage 的交付、验收和 commit 规范。
- `docs/dual_defense_comparison.md`：MAGMA vs DDFed 观测信号 / 选择机制 / 误检策略 / 交互开销的对照笔记（逐阶段补充数据与图表）。

## 快速开始

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_exp.py --config configs/mnist.yaml --run-name mnist_sanity
```

TinyImageNet 需先在 `./data/tinyimagenet/` 准备数据后再运行：

```bash
python scripts/run_exp.py --config configs/tinyimagenet.yaml --run-name tinyimagenet_sanity
```

## 结果速览（WIP）

- 默认 sanity：`python scripts/run_exp.py --config configs/mnist.yaml --run-name mnist_sanity`（记录路径 `results/mnist_sanity/`）。
- 攻击验证：见 `configs/attacks.yaml`；示例证据 `results/stage3_mnist_fedavg_ipm/`（FedAvg + IPM，单轮攻击导致精度明显下降）。
- 明文 baselines：见 `configs/aggregators.yaml`；示例证据 `results/stage4_mnist_noattack_fedavg/`、`results/stage4_mnist_scaling_clipmedian/`。
- MAGMA vs DDFed：Stage 6-9 会在 `results/summary_table.{csv,json}` 和 `docs/dual_defense_comparison.md` 中发布最终精度/FP/FN/time；当前仅有结构说明。

## 数据说明

本仓库不提交任何数据文件（`data/` 下的原始数据、缓存、切分统计都在 `.gitignore` 中）。在本地准备好 MNIST/FMNIST/CIFAR10/TinyImageNet 后，再运行 `scripts/run_exp.py`。默认 non-IID 设置为 Dirichlet α=0.5，可在 `configs/*.yaml` 中覆盖；切分统计会保存成 `results/<run_id>/artifacts/client_stats.json`。

## 目录速记

| 目录 | 作用 |
| --- | --- |
| `configs/` | 数据集 / 模型 / 攻击 / 聚合器 / FHE 参数模板 |
| `scripts/` | `run_exp.py` + 未来的 sweep / summarize / plot 工具脚本 |
| `magma/` | 几何聚合组件（距离 → 层次聚类 → jump ratio → aggregator） |
| `baselines/` | FedAvg、Krum、Median、TrimmedMean、ClipMedian、CosDefense、DDFed |
| `attacks/` | IPM、ALIE、Scaling、Label Flip |
| `results/` | 每次实验自带日志、metrics、图表、配置快照 |
| `docs/` | 比较文档与未来的 threat-model / reproduce / results 报告 |

如需新增功能或文档，先更新 `EXECUTION_PLAN.md`，再以 Stage 形式提交，确保可审计。
