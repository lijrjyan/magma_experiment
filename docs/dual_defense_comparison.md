# Dual Defense (DDFed) vs. MAGMA（对照速记）

> 文档定位：把 MAGMA 与 Dual Defense (DDFed) 的差异“结构化”下来，方便 Stage 6-9 在跑实验时逐项填充证据。结果数据将在 `results/summary_table.*` 生成后补充；此版本仅记录判定逻辑与未来需要的指标。

## 观测信号

- **DDFed**：服务端-客户端余弦相似度（reference-based），需要两阶段交互收集客户端反馈。
- **MAGMA**：客户端间流形距离（intrinsic geometry），以 `Δ_i = W_i^last - W_G^last` 为基础构造 pairwise matrix，可扩展到更丰富的 embedding。

## 选择机制

- **DDFed**：均值阈值 + 客户端 majority vote；得分低于阈值即剔除，可附带 clipping。
- **MAGMA**：Ward linkage → dendrogram → jump ratio `r_k = h_{k+1}/h_k` → 在 `k*` 处切分并取 largest component；无显著 jump 时默认全保留。

## 误检/漏检策略

- **DDFed**：依赖 reference；异质性增大时稀有正常客户端容易被判定为 outlier（FP↑）。
- **MAGMA**：借助结构 gap 降低 FP；但协同攻击若能伪造一个大组件，则可能出现 FN↑。Stage 8/9 需要报告 `FP_rate`、`FN_rate`、`avg_kept_clients`。

## 交互与开销

- **DDFed**：server-client 多轮交互，通信开销高但局部计算较轻，需要记录交互次数、反馈消息尺寸。
- **MAGMA**：一次收集所有客户端更新，在 server 侧执行 `O(m^2 d)` 的距离与聚类；通信较少但需要统计 `time_per_round_avg ± std` 与距离矩阵的额外内存占用。

## 实验填空（Stage 对应）

- Stage 6：复现 DDFed，输出交互统计与 clipping 选项验证。
- Stage 7：MAGMA / DDFed 的 FHE mock（验证明文 vs mock 的一致性）。
- Stage 8：主实验矩阵填入 `accuracy_pre_attack / min_after_attack / final / recovery_rounds`。
- Stage 9：生成对比图、表，并在此文档中贴关键图（accuracy、FP/FN、runtime）。

> 注：提交新实验前，请把结果文件路径与命令写入 `docs/repro_runs/<date>_<scenario>.md`，以便引用到本页面。
