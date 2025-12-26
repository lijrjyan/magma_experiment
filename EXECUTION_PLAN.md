# EXECUTION_PLAN.md

> 目标：把 **MAGMA（Manifold-Aware Geometric Aggregation）** 的实验与论文对比（重点对比 **DDFed / Dual Defense**）做成**可复现、可审计、每一步都可回滚**的工程执行计划。  
> 强制要求：**每个阶段/步骤完成后必须 commit + push（同一步骤内可多次小提交，但结束时至少 1 次 push）。**

---

## 0) Repo / Branch / 强制 Push 规则

- GitHub 仓库（唯一远端）： `git@github.com:lijrjyan/magma_experiment.git`
- 分支：`main`

阶段完成后必须执行：

```bash
git status
git add -A
git commit -m "stageX: <本阶段一句话总结>"
git push origin main
```

> 备注：如需临时分支可自建，但每阶段结束前必须 merge 回 main 并 push。

---

## 1) 总体交付物清单

### 1.1 代码侧

- [ ] `configs/`：所有实验设置参数化（数据划分/模型/攻击/聚合器/FHE）
- [ ] `magma/`：距离矩阵 → 层次聚类 → jump ratio → largest component → 聚合
- [ ] `baselines/`：FedAvg / Krum / Median / TrimmedMean / ClipMedian / CosDefense / DDFed
- [ ] `attacks/`：IPM / ALIE / Scaling / LabelFlip
- [ ] `scripts/`：`run_exp.py` / `sweep.sh` / `summarize.py` / `plot_curves.py`
- [ ] `results/`：固定结构保存 raw logs + summary tables + figures
- [ ] `REPRODUCE.md`：环境、数据、命令一键复现

### 1.2 论文侧

- [ ] `docs/dual_defense_comparison.md`：逐段对比 DDFed 与 MAGMA
- [ ] `docs/threat_model_alignment.md` & `paper/`：Related Work / Method / Experiment / Discussion 描述
- [ ] 图表：流程图 + accuracy 曲线 + accuracy/FP/FN/time 表格

---

## 2) 目录骨架建议

```
.
├── EXECUTION_PLAN.md
├── README.md
├── magma/
├── baselines/
├── attacks/
├── data/
├── models/
├── configs/
├── scripts/
├── results/
├── docs/
└── REPRODUCE.md (Stage 11)
```

遵循“最小破坏”原则：已有模块直接复用，新增文件夹承接计划需求。

---

## 3) 统一实验口径

### 3.1 FL 默认配置

| 参数 | 默认值 |
| --- | --- |
| 客户端总数 `N` | 100 |
| 每轮采样 `m` | 10 |
| 本地训练 | epochs=3, batch=64, SGD lr=0.01, momentum=0.9 |
| 训练轮数 `T` | 100 |
| 攻击开始轮 `t_attack` | 50（另做 cold-start） |
| attacker ratio | 默认 0.3，sweep 0.1/0.2/0.3/0.4 |
| Non-IID | Dirichlet alpha=0.5（外加 extreme shard 实验） |

### 3.2 模型

- MNIST/FMNIST：CNN9 (~200k 参数)
- CIFAR10：CNN9 或 ResNet-18（至少 1 个深模型支撑“可扩展性”）

### 3.3 MAGMA 口径

- 表征：`Δ_i = W_i^last - W_G^last`
- 距离：`d_ij = ||Δ_i - Δ_j||_2`
- 聚类：Ward linkage
- jump ratio：`r_k = h_{k+1} / h_k`
- 阈值：`λ = 1.5`
- 选择：cut at `h_{k*}` 后取 largest component；若无 jump → 全保留

### 3.4 评估指标

- Accuracy 曲线（每轮 test acc）
- 关键点：`acc_pre_attack (t=49)`、`min_acc_after_attack (t>=50)`、`final_acc`、`recovery_rounds`（恢复到 `0.95*acc_pre_attack` 的轮数）
- 过滤质量：`FP_rate`、`FN_rate`、`avg_kept_clients`
- 开销：`time_per_round_avg ± std`，并讨论通信成本（MAGMA 距离矩阵 vs DDFed 交互）

---

## 4) 分阶段执行计划

每阶段完成后必须 push。必要时在 `results/` 中保留 sanity 日志作为验收证据。

### Stage 1：工程骨架 + 统一入口 ✅（已完成）
- 目标：`scripts/run_exp.py` 统一入口、配置驱动、日志/metrics/summaries 固化
- 产物：`scripts/run_exp.py`、`configs/*`、`utils/logging.py`、`results/<run_id>/...`
- 验收：`python scripts/run_exp.py --config configs/mnist.yaml --run-name sanity` 能跑并生成日志
- 提交信息：`stage1: add unified experiment entrypoint + logging + config skeleton`

### Stage 2：数据加载与 Non-IID 切分
- 目标：MNIST/FMNIST/CIFAR10 数据下载 + IID + Dirichlet 切分可复现
- 产物：`data/loaders.py`、`data/partition.py`、`configs/{mnist,fmnist,cifar10}.yaml`
- 验收：同 dataset/seed/alpha 结果一致；输出 `results/.../client_stats.json`
- 提交信息：`stage2: add dataset loaders + reproducible iid/dirichlet partitioning with client stats`

### Stage 3：攻击实现（IPM / ALIE / Scaling / LabelFlip）
- 目标：攻击模块化，支持 start_round/ratio/强度
- 产物：`attacks/*.py` + `configs/attacks.yaml`
- 验收：单轮 sanity + FedAvg accuracy 明显下降
- 提交信息：`stage3: implement poisoning attacks (ipm/alie/scaling/label-flip) with unified interfaces`

### Stage 4：明文 baselines（FedAvg/Krum/Median/TrimmedMean/ClipMedian/CosDefense）
- 目标：无加密下所有聚合器跑通
- 产物：`baselines/*.py`、`configs/aggregators.yaml`
- 验收：无攻击时精度接近；Scaling attack 下 ClipMedian 稳定
- 提交信息：`stage4: add plaintext baselines (krum/median/trimmed/clipmedian/cosdefense) and validate curves`

### Stage 5：MAGMA 明文模式
- 目标：实现 Ward + jump ratio + largest component；记录 k*、kept_clients、FP/FN
- 产物：`magma/distance.py`、`magma/clustering.py`、`magma/aggregator.py`、`baselines/magma.py`
- 验收：Non-IID 下 FP 低、accuracy 稳定；无 jump 时全保留
- 提交信息：`stage5: implement MAGMA plaintext (ward + jump ratio) with per-round FP/FN logging`

### Stage 6：DDFed 基线复现
- 目标：两阶段相似度 + majority vote + 可选 clipping；记录交互次数
- 产物：`baselines/ddfed.py`、`docs/dual_defense_comparison.md`（初稿）
- 验收：复现 DDFed 趋势；日志包含交互统计
- 提交信息：`stage6: add DDFed baseline (two-phase selection + optional clipping) and write initial MAGMA vs DDFed comparison`

### Stage 7：FHE 模式打通
- 目标：`--fhe on/off`；MAGMA 距离/FHE mock；DDFed 相似度/FHE mock；后端封装
- 产物：`magma/fhe_backend.py`、`configs/fhe.yaml`
- 验收：`--fhe mock` 结果与明文一致；若接入 TenSEAL，至少小规模跑通 1 个 round
- 提交信息：`stage7: add FHE abstraction (mock + optional CKKS backend) for MAGMA distances and DDFed similarities`

### Stage 8：主实验矩阵
- 目标：MNIST/FMNIST/CIFAR10 × attacks × iid/dirichlet × ratios × seeds
- 产物：`scripts/sweep.sh`、`results/summary_table.{csv,json}`
- 验收：组合可跑完；summary 含 final/min/recovery/FP/FN/time
- 提交信息：`stage8: run main experiment grid and generate summary tables (accuracy/FPFN/runtime)`

### Stage 9：作图与论文落地
- 目标：生成曲线/表格；完善对比文本
- 产物：`scripts/plot_curves.py`、`results/figures/`、`docs/dual_defense_comparison.md`（完整版）
- 验收：`python scripts/plot_curves.py --input results/summary_table.csv --out results/figures` 成功；图可直接入论文
- 提交信息：`stage9: add plotting + finalize MAGMA vs DDFed comparison text and paper-ready figures/tables`

### Stage 10：消融实验
- 目标：λ、层级、heterogeneity 强度 三组消融
- 产物：`results/ablation/*.csv`、`results/figures/ablation_*.png`、`docs/threat_model_alignment.md`
- 验收：支撑论文的“why/when/conservative”叙述
- 提交信息：`stage10: add MAGMA ablations (lambda/layer/heterogeneity) with analysis-ready plots`

### Stage 11：最终复现包
- 目标：REPRODUCE 文档 + env 描述 + 一键脚本
- 产物：`REPRODUCE.md`、`environment.yml`（或 requirements） 、`scripts/reproduce_main.sh`、`CITATION.bib`
- 验收：在全新环境跑 MNIST/FMNIST 主结果
- 提交信息：`stage11: add full reproduction package (env + one-click scripts + docs)`

---

## 5) MAGMA vs DDFed 对比写作提纲

1. **观测信号**：
   - DDFed：global ↔ client cosine（reference-based）
   - MAGMA：client ↔ client distances（intrinsic geometry）
2. **选择机制**：
   - DDFed：mean-threshold + majority vote
   - MAGMA：Ward dendrogram + jump ratio → largest component
3. **误检行为**：
   - DDFed：异质性强 → 稀有 benign 易被误杀
   - MAGMA：无 gap → 保守接受；但协同攻击可能 FN↑
4. **交互与开销**：
   - DDFed：更多 server↔client 交互，较轻计算
   - MAGMA：少交互，计算 O(m²d)；需要 runtime & 通信分析

这些要点需落地在 `docs/dual_defense_comparison.md` 与论文相关章节。

---

## 6) 风险提示

- 攻击参数需与论文对齐，否则比较失真
- Dirichlet 切分需固定 seed 并输出 label stats
- 指标必须包含 `min_after_attack` 与 `recovery_rounds`
- FHE 先 mock 再真 backend，避免一次吃胖
- 论文写作要强调“信号/假设/交互”差异，而非单纯数值

---

## 7) Quick Commands

```bash
# 1) 无攻击 sanity
python scripts/run_exp.py --config configs/mnist.yaml --run-name mnist_no_attack \
    --override attacks.strategy=none

# 2) IPM 对比（MAGMA vs DDFed）
python scripts/run_exp.py --config configs/fmnist.yaml --run-name fmnist_magma_ipm \
    --override aggregation.method=magma --override attacks.strategy=model_poisoning_ipm
python scripts/run_exp.py --config configs/fmnist.yaml --run-name fmnist_ddfed_ipm \
    --override aggregation.method=dual_defense --override attacks.strategy=model_poisoning_ipm

# 3) 批量 sweep
bash scripts/sweep.sh

# 4) 汇总 + 作图
python scripts/summarize.py --results_dir results --out results/summary_table.csv
python scripts/plot_curves.py --input results/summary_table.csv --out results/figures
```

> 如需更新计划，请同步修改本文件并在 README 中注明。
