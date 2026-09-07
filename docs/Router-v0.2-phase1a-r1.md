# Router v0.2 Phase 1A-R1：Pilot Surface Quality Revision

> 状态：READY FOR SPOT REVIEW（自动检查通过；未执行或宣称 Human Surface Audit PASS）
>
> Generator：`2.1.0`
>
> Pilot：`router-v0.2-pilot-v2`

## 修订范围

R1 保留 Benchmark Spec v0.3、5 个 task、12 项 capability、6 个匿名模型、质量阈值、task-conditioned language policy、namespace RNG 和 observable/sidecar 边界。本轮只修订 Pilot 的 programmatic composition、content payload、surface realization 和审计工具。

`router-v0.2-pilot-v1` 的 config、Query、sidecar、manifest、audit 和 review packet 均保留。v2 继续使用 master seed `20260907`，将 `content`、`surface` namespace 提升到 v2 salt，使新的表面语义不会冒用 v1 随机流；capability 与 difficulty namespace 保持不变，便于对照 latent assignment。正式 `dataset_version` 仍为 `router-benchmark-v0.2.0`。

## Task-specific Content Payload

`payloads_v2.py` 在 Layer 2 中加入 `ContentPayload`：

- code 提供代码、SQL/schema、日志、失败输入/预期输出、API response、模块关系、CSV 样例或依赖图；
- math 提供可实际计算的方程、条件概率、递推、样本表、几何尺寸、整数优化约束、票价方程或季度数据；
- QA 提供 2–5 段带定义、例外、时间、修订或证据冲突的材料；
- summary 提供 3–6 段会议、研究、事故、指标、冲突报告、指南或双语正文；
- translation 提供与场景说明分开的技术说明、合同、通知、领域报告、UI、事故或歧义源文。

明显数字必须进入方程、样例、时间线、口径、限制或待保留事实。Capability evidence 现在指向这些结构片段；SC 由 JSON/CSV/schema/函数签名等输出契约体现，FT 由具体数字、例外、行为和不确定性体现，AM/MH/LC 等由实际材料关系体现。

Surface 不再把独立 capability 要求句追加到正文，也不把任务目标重复渲染成“背景”。四种 realization 只调整任务、材料和交付形式的顺序。这样仍保留可见、可学习的任务结构，同时避免跨 task 的固定 cue block 和重复前言。

## Answerability 与 Surface Lint

生成阶段验证 payload 的结构 marker 和 capability evidence。自动审计再对最终 Query 执行 task-specific 检查：debug 必须有代码、日志和失败输入/预期输出；math 必须有数值/符号关系和明确目标；QA 必须有证据段；summary 必须有正文；translation 必须有独立源文。

Surface lint 检查英文冠词重复、英文重复词、中文连续重复短语、异常标点串、中英文标点碰撞，以及 v1 式完整英文 capability instruction 硬插入 mixed Query。Lint 只用于发现基础问题，不能代替人工自然度判断。

## Shortcut 与重复结构审计

v2 audit 同时报告 exact/normalized sentence frequency、重复英文四词短语/中文十字片段、每项 capability 的词汇集中度和 realization diversity。Token–capability candidate 增加 task/frame 分布、surface style 数、evidence variant 数，并标记为 likely legitimate、likely generator artifact 或 uncertain。

Cue masking diagnostic 删除候选 token 后，检查对应 evidence span 是否仍保留数据、关系或结构，以及完整 Query 是否仍含充分上下文。本诊断不训练 Predictor，也不证明 capability 已可学习。

## 生成与复核

```powershell
.\.venv\Scripts\python.exe experiments\make_v0_2_pilot_v2.py
.\.venv\Scripts\python.exe -m pytest
```

主要产物：

- `data/router_v0_2/pilot_v2/pilot_queries.jsonl`
- `data/router_v0_2/pilot_v2/pilot_sidecar.jsonl`
- `data/router_v0_2/pilot_v2/pilot_manifest.json`
- `outputs/router_v0_2/pilot_v2/pilot_audit.json`
- `outputs/router_v0_2/pilot_v2/pilot_v1_vs_v2_comparison.json`
- `outputs/router_v0_2/pilot_v2/human_surface_review.csv`
- `outputs/router_v0_2/pilot_v2/spot_review_50.csv`

Spot Review 按 task 固定抽取 10 条；每个 task 覆盖全部 8 个 semantic frame，并优先覆盖 capability、difficulty、language、capability 数量和高关联 lexical candidate。即使自动分类没有 `likely_generator_artifact`，仍将 `uncertain` candidate 纳入抽样，留给人工判断。所有人工评分字段保持空白，文件不包含模型、quality、cost、latency、winner、outcome 或 Router result。

## 当前自动结果与边界

v2 共 250 条，每个 task 50 条；task-specific answerability 为 250/250，基础 lint failure 为 0，exact/normalized duplicate group 均为 0。平均 Query 长度由 347.552 降为 233.692 个字符，原因是实际 payload 取代了重复的抽象描述和 capability 要求句，而不是减少完成任务所需的输入。沿用 v1 审计定义的最大固定句占比由 10.4% 降至 2.8%；R1 normalized sentence share 由 16.0% 降至 4.0%，最大重复短语占比为 4.0%。

新 lexical 分类中，top-100 association candidate 内 `likely_generator_artifact` 从 99 降至 0；当前 top-100 均为 `uncertain`，不能据此宣称 shortcut 风险已消失。Cue masking 对 719 个候选诊断的最低结构保留率为 100%，说明移除单个关联 token 后仍保留可见 payload 结构；该诊断不等价于训练结果或人工判断。

自动结果只支持进入 50 条 Spot Review。只有 Spot Review 明显改善并由团队确认后，才值得进行完整 250 条 Human Surface Audit；本轮不进入 Minimum Benchmark、outcome generator、Predictor、Router 或 baseline。
