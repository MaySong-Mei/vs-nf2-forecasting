# P1 回归：用全部可得历史与协变量预测下一次体积变化

- **阶段 / 状态 / 日期**：P1 简单基线 / complete（公共散发性队列部分）/ 2026-09-11
- **科学问题和可证伪假设**：在 [体积基线](2026-09-11-p1-volume-baselines.md) 的基础上，放宽三个限制：使用截至当前的全部历史（而非固定两点）、加入治疗状态和采集条件协变量、允许非线性模型（梯度提升树）。假设：若这些信息中任一项携带可用信号，则回归对 log 体积变化的预测误差应低于 persistence（预测变化为 0）。可证伪：患者级五折下，所有回归相对 persistence 的配对差异置信区间跨零或为正。
- **预测截点、可用历史、目标和时间范围**：截点 t_k；历史为自上次跨治疗以来到 t_k 的全部扫描（1 到 3 次）；目标 y = log(V_{k+1}/V_k)；预测间隔中位数 365 天。目标扫描的采集参数不进入特征。
- **数据版本及有效患者/肿瘤/时点数，纳排规则**：UCSD-VS-Longitudinal v1。所有相邻区间中，目标扫描不是治疗后首次扫描、两端 mask 非空、非无残留者：**317 个区间 / 165 患者**，其中治疗前 203、治疗后 114；历史长度 1/2/3 次为 166/98/53。
- **疾病、来源、治疗及分割 provenance**：同体积基线记录。治疗状态取自官方 Pre/Post 字段与治疗类型。
- **患者级 train/validation/test 划分和随机种子**：冻结 `ucsd_splits.csv` 的 98 名患者保留原折；其余 67 名患者按 seed 100 打乱后补齐到最少的折，各折 33 名患者、61 到 65 个区间。扩展划分表（含患者 ID）存 Git 忽略目录，SHA-256 记录在 summary.json。内层调参为训练折内患者分组 5 折。
- **基线、信息增量与控制变量**：
  - persistence（ŷ=0）；train_median_change（训练折中位数变化）
  - shrunk_trend：ŷ = α · 最近斜率 · 间隔，α ∈ {0, 0.1, …, 1} 内层 CV
  - ridge 与 HistGradientBoosting（绝对误差损失，min_samples_leaf 10，深度/学习率内层 CV），各自跑 5 组嵌套特征：history（log V_k、历史次数、最近斜率、整段 OLS 斜率、历史跨度、间隔及交互）→ +年龄性别 → +治疗状态与类型 → +采集条件（层厚、面内像素、IAC 序列、场强、体素体积）→ 全部
- **主指标、次指标、患者级区间与分层**：主指标 |ŷ−y| 均值（|log 变化| 误差）及与 persistence 的配对差异（患者聚类 bootstrap 95% CI，1000 次）；次指标中位数、MAE mm³、相对零预测的 R²、Spearman。分层：治疗前/后、历史 1 次 vs ≥2 次、间隔 ≤/>1 年、肿瘤 <100 vs ≥100 mm³。
- **代码 commit、配置、环境及命令**：`tumor` commit 1e9f0cc（工作区未提交）；scikit-learn 1.7.2 本次装入 `.venv`。脚本 [scripts/p1_regression.py](../scripts/p1_regression.py)，仓库根目录运行：

  ```bash
  .venv/Scripts/python.exe projects/frank_vs_forecasting/scripts/p1_regression.py \
    --manifest reproductions/deepgrowth_ucsd/data/processed/ucsd_manifest.csv \
    --clinical reproductions/deepgrowth_ucsd/data/raw/metadata/UCSD-VS-Longitudinal_clinical_information.tsv \
    --volumes projects/frank_vs_forecasting/data/p1_volume_baselines/timepoint_volumes.csv \
    --spacing projects/frank_vs_forecasting/data/p1_volume_baselines/timepoint_spacing.csv \
    --splits reproductions/deepgrowth_ucsd/data/processed/ucsd_splits.csv \
    --private-dir projects/frank_vs_forecasting/data/p1_regression \
    --results-dir projects/frank_vs_forecasting/results/2026-09-11-p1-regression
  ```

- **结果存储标识和聚合报告链接**：[results/2026-09-11-p1-regression/summary.json](../results/2026-09-11-p1-regression/summary.json) 与 `table.txt`（聚合，无患者 ID）；逐区间输出在忽略目录 `data/p1_regression/`。

## 结果

全部 317 个区间（165 患者）。"更优比例"是逐区间 |误差| 低于 persistence 的比例；差异 CI 为负且不跨零才算超过 persistence。

| 模型 \| 特征 | \|logΔ\| 误差均值 | 中位数 | MAE mm³ | R² 对零 | 更优比例 | 差异 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| persistence | 0.343 | 0.200 | 204 | 0 | — | — |
| train_median_change | 0.344 | 0.200 | 210 | −0.001 | 51% | [−0.002, +0.003] |
| shrunk_trend（α=0.1 每折） | 0.342 | 0.201 | 202 | −0.023 | 25% | [−0.007, +0.004] |
| ridge \| history | 0.372 | 0.235 | 265 | −0.064 | 41% | [+0.013, +0.043] |
| ridge \| all | 0.374 | 0.255 | 263 | −0.040 | 44% | [+0.012, +0.050] |
| gbm \| history | 0.345 | 0.209 | 201 | +0.007 | 47% | [−0.005, +0.008] |
| gbm \| +clinical | 0.347 | 0.207 | 198 | +0.002 | 49% | [−0.004, +0.012] |
| gbm \| +treatment | **0.339** | **0.198** | 198 | +0.019 | 53% | [−0.012, +0.003] |
| gbm \| +acquisition | 0.346 | 0.207 | 210 | +0.008 | 48% | [−0.004, +0.010] |
| gbm \| all | 0.345 | 0.207 | 206 | +0.023 | 46% | [−0.007, +0.010] |

分层（gbm \| +treatment，最好的一组）：

| 子群 | n | persistence 误差 | gbm 误差 | R² 对零 | 更优比例 | 差异 CI |
|---|---:|---:|---:|---:|---:|---:|
| 治疗前 | 203 | 0.377 | 0.378 | +0.011 | 50% | [−0.007, +0.008] |
| 治疗后 | 114 | 0.283 | 0.270 | +0.078 | 58% | [−0.028, +0.001] |
| 历史 1 次 | 166 | 0.371 | 0.366 | +0.024 | 52% | [−0.014, +0.005] |
| 历史 ≥2 次 | 151 | 0.313 | 0.310 | +0.012 | 54% | [−0.016, +0.009] |
| 肿瘤 <100 mm³ | 107 | 0.433 | 0.432 | +0.021 | 48% | [−0.012, +0.011] |
| 肿瘤 ≥100 mm³ | 210 | 0.298 | 0.292 | +0.017 | 56% | [−0.016, +0.004] |

## 实现、时间泄漏、数据重叠、测量误差检查

- 训练/测试患者逐折断言无交集；目标扫描的层厚、序列、扫描仪均未进入特征；治疗后首次扫描不作目标。
- **内层 CV 自己收敛到 persistence**：ridge 在 5 折 × 5 组特征全部选到网格上限 λ=100；shrunk_trend 在 5 折全部选 α=0.1。这是训练数据在说"最好的线性预测几乎是零变化"，不是网格设计的意外。λ 网格未再向上扩展，扩展只会让 ridge 更接近 persistence，不改变结论。
- 目标 y 均值 0.014、SD 0.67：治疗前后混合的队列平均变化接近零，而单区间波动很大，与数据审计的噪声结论一致。
- GBM 在治疗后子群 R² 对零 0.08、更优 58%，CI 上界 +0.001，是唯一接近显著的信号；对应放疗/术后残留肿瘤的缓慢一致缩小。它没有通过预登记的"CI 不跨零"标准。

## 结果与解释

- **观察**：使用全部历史、治疗状态、采集条件和非线性模型之后，仍无任何回归在患者级独立评估中超过 persistence。线性模型显著更差，树模型与 persistence 统计上不可区分。
- **假设**：历史体积轨迹在本队列中的可预测成分小于单次测量噪声；治疗后残留肿瘤有微弱可预测趋势，但样本（114 区间 / 63 患者）不足以确认。
- **局限**：单中心散发性队列；历史最多 3 次；未做重复分割噪声估计；GBM 调参网格小；无 NF2。

## 阶段退出标准是否满足及证据

对公共散发性队列，P1 的退出标准已满足：persistence、趋势、线性/LMM/正则化/树模型基线齐全，患者级评估、区间、分层、可复跑配置、泄漏检查均有记录，负结果明确。**Frank 主队列（Yale/NF2）的 P1 仍待 P0 决策。**

## 下一步与改变方向的触发条件

1. 向 Frank 汇报：在 UCSD 上，"体积历史 + 临床 + 采集协变量"没有超过 persistence 的可预测增量，噪声下界（短间隔 |相对变化| 中位数 0.13）与年增长同量级。P2 的影像增量应以降低测量噪声（分割一致性、序列一致性）为首要目标。
2. 若要继续在体积层面推进，先做重复分割或跨序列体积一致性实验，得到噪声 SD；只有模型误差低于噪声 SD 才有意义。
3. 治疗后残留子群若 Frank 认为有临床意义，可单独扩大样本（MC-RC2 等公共队列）再验证 GBM 的边缘信号；否则从主线剔除。
4. 触发改变方向：Yale/NF2 队列的 persistence 误差若明显低于 UCSD（更一致的分割协议），才值得在该队列重跑本实验。
