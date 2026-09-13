# P1 体积基线：UCSD 公共散发性队列（第一轮）

- **阶段 / 状态 / 日期**：P1 简单基线 / active / 2026-09-11
- **科学问题和可证伪假设**：仅用既往两次 mask 体积和时间间隔，能否比"下次体积 = 当前体积"（persistence）更准确地预测下一次随访体积？可证伪：若线性/对数线性外推、队列增长率、ridge、LMM 在患者级独立评估中均不优于 persistence，则两点历史趋势在本队列没有可用的预测增量。
- **预测截点、可用历史、目标和时间范围**：截点为 t2；输入 t1、t2 的原生空间 mask 体积、两段间隔、请求的预测间隔 Δt2，可选年龄和性别；目标是 t3 的 mask 体积（mm³）。Δt2 中位数 373 天，范围 37–4102 天。
- **数据版本及有效患者/肿瘤/时点数，纳排规则**：UCSD-VS-Longitudinal v1（DOI 10.7937/WEFA-CP23），570 时点 / 191 患者。滚动三元组 188（117 患者）；排除 t2 或 t3 区间跨治疗 33、含无残留时点 26（与空 mask 26 完全重合）；合格 151 三元组 / 98 患者。与 DeepGrowth 复现的临床合格队列完全一致；本实验不需要配准，因此不受 103 例配准 QC 子集限制。**数据审计发现**（见 [数据审计报告](../results/2026-09-11-ucsd-data-audit/report.md)）：滚动三元组规则只禁止区间跨治疗，151 例中 91 例（60 患者）三个时点全部治疗前，60 例（38 患者）全部治疗后（放疗后 40、术后 20，残留肿瘤随访）。本记录下文按此分层报告。
- **疾病、来源、治疗及分割 provenance**：散发性 VS，单中心公共数据，分割为数据集自带 seg_t1post/seg_t1postIAC，标签 0/1。体积 = 前景体素数 × |det(affine)|，未重采样。治疗规则复用 `reproductions/deepgrowth_ucsd` 的官方临床表适配（治疗相对天数符号翻转 ∪ Pre→Post）。
- **患者级 train/validation/test 划分和随机种子**：复用冻结的 `ucsd_splits.csv`（98 患者，5 折，seed 100）；各折测试 30/28/31/32/30 例、20/20/20/19/19 患者。需要拟合的模型只用训练折；ridge 的 λ 在训练折内做患者分组 5 折内层 CV。bootstrap 与内层划分 seed 100。
- **基线、信息增量与控制变量**：
  1. persistence：V̂3 = V2
  2. linear_volume：V2 + (V2−V1)/Δt1·Δt2，截断到 ≥0
  3. log_linear_volume：V2·exp(r·Δt2)，r 为两点对数增长率
  4. pooled_rate：V2·exp(r̄·Δt2)，r̄ 为训练折患者全部未治疗区间对数增长率的均值
  5. ridge_history：log V3 ~ log V2、log(V2/V1)、r、Δt2、r·Δt2、log Δt1
  6. ridge_history_clinical：5 + 年龄、性别
  7. lmm_random_slope：log V ~ 时间，患者随机截距+斜率（statsmodels MixedLM，REML），对测试患者用其 t1、t2 做 BLUP 后外推
- **主指标、次指标、患者级区间与分层**：主指标 |log(V̂/V3)| 均值与 |RVD| 中位数（对小体积稳健）；次指标 MAE (mm³)、signed RVD、mean |RVD|（附患者聚类 bootstrap 95% CI）、预测变化与真实变化的 Spearman。分层：|真实相对变化| 前 20%（截点 0.40，n=31）、Δt2 ≤ / > 365 天、配准 QC 通过的 103 例子集。
- **代码 commit、配置、环境及命令**：`tumor` 仓库 commit 1e9f0cc（工作区含本次未提交改动）；Python 3.10.15，numpy 2.2.6，pandas 2.3.3，statsmodels 0.15.0（本次装入 `.venv`）。脚本 [scripts/p1_volume_baselines.py](../scripts/p1_volume_baselines.py)，在仓库根目录运行：

  ```bash
  .venv/Scripts/python.exe projects/frank_vs_forecasting/scripts/p1_volume_baselines.py \
    --manifest reproductions/deepgrowth_ucsd/data/processed/ucsd_manifest.csv \
    --data-root reproductions/deepgrowth_ucsd/data/raw/ucsd_images/UCSD-VS-Longitudinal \
    --clinical reproductions/deepgrowth_ucsd/data/raw/metadata/UCSD-VS-Longitudinal_clinical_information.tsv \
    --splits reproductions/deepgrowth_ucsd/data/processed/ucsd_splits.csv \
    --qc-metadata reproductions/deepgrowth_ucsd/data/ucsd_prepared/metadata.csv \
    --private-dir projects/frank_vs_forecasting/data/p1_volume_baselines \
    --results-dir projects/frank_vs_forecasting/results/2026-09-11-p1-volume-baselines
  ```

- **结果存储标识和聚合报告链接**：聚合结果 [results/2026-09-11-p1-volume-baselines/summary.json](../results/2026-09-11-p1-volume-baselines/summary.json)（已核查不含患者 ID）；逐病例结果与时点体积缓存在 Git 忽略的 `data/p1_volume_baselines/`。

## 结果

全部 151 例（98 患者）：

| 方法 | \|log 比\| 均值 ↓ | \|RVD\| 中位数 ↓ | MAE mm³ ↓ | mean \|RVD\| [95% CI] | 优于 persistence 的病例比例 |
|---|---:|---:|---:|---:|---:|
| persistence | **0.313** | **0.178** | 183.1 | 2.70 [0.25, 7.89] | — |
| pooled_rate | 0.313 | 0.180 | **178.8** | 2.74 | 0.47 |
| ridge_history | 0.357 | 0.221 | 211.9 | 2.12 | 0.48 |
| ridge_history_clinical | 0.369 | 0.224 | 221.2 | 2.21 | 0.42 |
| lmm_random_slope | 0.354 | 0.250 | 280.3 | **0.48** [0.31, 0.78] | 0.43 |
| linear_volume | 0.851 | 0.326 | 229.1 | 4.48 | 0.30 |
| log_linear_volume | 0.559 | 0.341 | 1078.1 | 173.0 | 0.29 |

前 20% 变化子集（n=31）：persistence \|RVD\| 中位数 0.714、MAE 254；ridge_history MAE 167；LMM MAE 158、\|log 比\| 0.79。趋势模型在此子集 MAE 更低，但在稳健指标上仍未超过 persistence。

配准 QC 通过的 103 例子集：persistence MAE 142.8 mm³，与 DeepGrowth 复现中 stable-mask 基线在重采样裁剪空间的 147.2 mm³ 一致，说明两条流水线的体积口径可互相印证。

预测变化与真实变化的 Spearman：linear 0.06、log-linear 0.06、pooled 0.05、ridge 0.20、ridge+clinical 0.14、LMM 0.15。

### 按治疗状态分层

| 子群 | 方法 | \|log 比\| 均值 | \|RVD\| 中位数 | MAE mm³ | Spearman |
|---|---|---:|---:|---:|---:|
| 全部治疗前 (n=91, 60 患者) | persistence | **0.361** | **0.147** | **97** | — |
| | pooled_rate | 0.361 | 0.163 | 100 | 0.12 |
| | ridge_history | 0.417 | 0.199 | 127 | 0.11 |
| | lmm_random_slope | 0.400 | 0.261 | 160 | 0.01 |
| | linear_volume | 0.658 | 0.300 | 123 | −0.07 |
| 全部治疗后 (n=60, 38 患者) | persistence | **0.240** | **0.200** | 313 | — |
| | pooled_rate | 0.239 | 0.204 | **298** | −0.01 |
| | ridge_history | 0.267 | 0.228 | 340 | 0.28 |
| | lmm_random_slope | 0.284 | 0.245 | 462 | 0.22 |
| | linear_volume | 1.142 | 0.388 | 390 | 0.22 |

两个子群里 persistence 都没有被超过。治疗前子群肿瘤更小（MAE 97 mm³），相对噪声更大，趋势模型的 Spearman 接近零；治疗后残留肿瘤更大，相对噪声较小，趋势模型开始有微弱的方向信息（Spearman 0.2 到 0.3），与放疗后逐渐缩小的一致性动态相符，但仍未转化为更低的误差。

## 实现、时间泄漏、数据重叠、测量误差检查

- 训练/测试患者集合逐折断言无交集；t3 只作为目标；Δt2 视为已知请求日期。
- 队列计数（188 → 151 / 98，各折例数）与 REPRODUCTION.md 中冻结的投影完全一致。
- 队列平均对数增长率各折在 −0.03 至 +0.02 /年之间，即整体近似不增长，所以 pooled_rate ≈ persistence。
- LMM 5 折中仅 1 折报告收敛；随机斜率 SD 0.09–0.40 /年，**残差 SD 0.40–0.52（log 体积）**，即扫描间约 40–50% 的体积波动无法被患者线性轨迹解释。
- 放射科报告分类（相对上一次）与 mask 体积变化 ±20% 的交叉表：报告"Unchanged"的 124 例中 56 例（45%）体积变化超过 20%（29 减 / 27 增）；报告"Increased"17 例中 11 例体积增 >20%；"Decreased"10 例中 7 例体积减 >20%。
- 数据 QC 发现：1 例无记录治疗但体积从约 1761 mm³ 降到 4.8 mm³（可疑未记录切除或分割失败），单独贡献 mean \|RVD\| 的绝大部分；30/151 目标体积 < 50 mm³，相对误差在这些病例极不稳定。剔除 <50 mm³ 后（n=121）：persistence mean/median \|RVD\| 0.239/0.164，LMM 0.272/0.218，ridge 0.259/0.192，结论不变。

## 结果与解释

- **观察**：没有任何简单模型在稳健指标上超过 persistence；两点外推（线性/对数线性）显著更差，且预测间隔越长越差（linear \|log 比\| ≤1 年 0.64，>1 年 1.02）。ridge 只在极端变化子集的 MAE 上略好，Spearman 0.20 说明历史斜率携带的方向信息很弱。
- **假设**：单次分割的体积噪声（残差 SD ≈ 0.4–0.5 log）与两次随访的真实增长量级相当，两点斜率因此基本是噪声；队列平均增长接近零也削弱了任何"生长先验"。
- **局限**：151 例混合了治疗前自然史和治疗后残留随访，主表数字是两者的混合，分层表才是各自的结论；单中心公共散发性队列，不是 NF2；输入被限制为两次历史以与 DeepGrowth 三元组可比，未用更长历史；LMM 多数折未收敛，其结果只能作为收缩基线参考；未做重复分割评估测量噪声；未处理同一患者多个三元组的相关性（bootstrap 按患者聚类已部分处理）。

## 阶段退出标准是否满足及证据

部分满足：persistence、趋势、线性/LMM 基线均完成，患者级独立评估、效应区间、分层、可复跑配置齐全，泄漏与队列一致性检查通过。**未满足**：P0 中 Yale/NF2 的目标、协变量、时间范围仍未确认；本轮只覆盖公共散发性数据，不能作为 Frank 项目主队列的 P1 完成证据。

## 下一步与改变方向的触发条件

1. 把"persistence 难以超越、两点斜率≈噪声"作为向 Frank 汇报的第一条基线事实，并附上放射科分类 vs 体积变化交叉表。
2. 在 P2 之前先量化测量噪声：至少对一部分病例做重复分割或用不同序列（t1post vs t1postIAC）的体积差估计噪声下界；若噪声 SD 与年增长同量级，P2 的影像增量必须以"降低噪声"而不是"预测趋势"为目标。
3. 允许使用全部历史（≥3 次）时再测一次 LMM 和 ridge；若仍不超过 persistence，则把"体积轨迹外推"从主线降级。
4. 触发改变方向：若 Yale/NF2 队列有效 N < 40 患者或无未治疗区间，P3 迁移实验先降为 NF2-only 的 persistence 对照。
