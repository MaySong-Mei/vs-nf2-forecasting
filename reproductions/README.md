# 其他代码复现

独立记录外部方法及其复现任务。每个方法使用独立文件夹，保留上游来源、复现协议、环境、数据角色、完成状态与结果边界。

| 实验 | 问题与数据 | 已有证据 | 后续 |
|---|---|---|---|
| [DeepGrowth × UCSD](deepgrowth_ucsd/README.md) | 两次历史增强 T1 与 mask 预测第三次肿瘤形状；公共 UCSD 队列 | 103 triples / 73 patients；五折训练完成，模型未超过稳定 mask 和 linear-SDF 基线 | 单折采样／损失消融以检验塌缩解释；尚未证实修复 |

UCSD 实验是公共队列方法重实现和外部数据验证，不是原论文私有队列数值的精确复现。它为 Frank 项目提供方法与工程参考；其指标不能直接作为 Yale 或 NF2 性能证据。

原始实现、维护的 `ucsd_repro/`、脚本、测试、配置、实验结果和数据均在该子目录内，运行命令以子目录为工作目录。详细历史协议见 [REPRODUCTION.md](deepgrowth_ucsd/REPRODUCTION.md)。
