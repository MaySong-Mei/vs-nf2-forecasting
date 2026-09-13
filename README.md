# 肿瘤纵向预测研究

本仓库分为两个独立部分。每个项目维护自己的任务定义、阶段状态、实验协议和结果；复现实验的完成不代表 Frank 项目的模型已经验证。

| 部分 | 内容 | 入口 |
|---|---|---|
| 其他代码复现 | 已发表方法的实现、公共数据实验、偏差与失败分析 | [reproductions](reproductions/README.md) |
| Frank 的研究项目 | VS 纵向预测、影像信息增量和散发性到 NF2 迁移 | [Frank proposal](projects/frank_vs_forecasting/README.md) |

## 目录

```text
reproductions/
  deepgrowth_ucsd/       # 原有 DeepGrowth 代码及 UCSD 实验完整迁入
projects/
  frank_vs_forecasting/  # proposal、阶段计划、实验管理
.venv/                  # 原有本地环境，未移动
```

## 运行已有 UCSD 实验

从仓库根目录进入实验目录后运行原有命令：

```powershell
.\.venv\Scripts\Activate.ps1
cd reproductions/deepgrowth_ucsd
python -m unittest discover -s tests -v
```

新建环境及完整实验步骤见子项目 README。已有环境若曾 editable 安装旧根目录，请在子项目目录执行 `python -m pip install --no-deps -e .` 更新安装位置。

## 项目与阶段管理

每个项目 README 必须写明科学问题、输入、预测时点、输出、数据范围、当前阶段、下一步与完成标准。阶段只在有可审查产物和验证证据后推进；历史记录、计划和已完成实验分开说明。每次实验记录所属阶段、配置、数据版本、患者级划分、主要指标、结果位置和解释。详细规则见 [Frank 阶段计划](projects/frank_vs_forecasting/PHASES.md)。

## 迁移说明

2026-09-11：原根目录的代码、配置、测试、实验文档，以及本地 data/manifests/outputs 一并迁至 `reproductions/deepgrowth_ucsd/`。原 README 的未提交内容保留在子项目 README 中。Git 历史及远程仓库不变。

历史冻结清单、运行元数据可能记录迁移前绝对路径；它们作为原始证据保留，不静默重写哈希或声称可直接续训。重跑时从新目录执行，检查输入路径，生成新的运行记录；不得覆盖已冻结结果。
