# Router v0.2 Phase 1A：Benchmark Infrastructure 与 Pilot Surface Set

> 状态：READY FOR HUMAN REVIEW（仅表示自动检查通过，人工 Surface Audit 尚未完成）
>
> 对应规格：`docs/Router-v0.2-benchmark-spec.md` v0.3
>
> Pilot：`router-v0.2-pilot-v1`

## 目标与边界

Phase 1A 落地 v0.2 的 schema、canonical config、namespace RNG、三层 Surface 工作流和 250 条 Pilot Query。Pilot 只用于检查 Query 表面质量与生成机制，不进入任何 Train、Validation、Development Test 或 Frozen Final split，也不生成 quality、cost、latency、winner，不训练 Predictor，不运行 Router 或 baseline。

## Schema 与信息边界

`src/zhiheng_router/benchmark_v02/schema.py` 将数据物理分为两份：

- observable record 只保存 `query_id`、`query_text`、`language`、显式输出约束、可见结构、`source_kind` 和 `dataset_version`；
- diagnostic sidecar 保存 task、capability 组合/权重/要求、difficulty、四层生成标识中的 Pilot family 标识、semantic frame、surface realization 和 RNG provenance。

两类记录按 `query_id` 一一对应。schema 使用严格字段集合，拒绝缺失或额外字段、未知枚举、空 Query、重复 ID、越界或非有限数值及格式错误的 family ID。未来 deployable Predictor 只能加载 observable 数据，不能加载 sidecar。

## Canonical config

`configs/router_v0_2_benchmark.json` 固定规格 v0.3 批准的版本号、5 个 coarse task、12 项 capability、6 个匿名 development model profile、task residual、synergy、synthetic runtime profile、task-conditioned language policy、质量阈值 `0.8` 及 `0.7/0.9` sensitivity。Pilot 尚不使用的 quality 和 synthetic workload 参数只作冻结记录，Phase 1A 没有实现 outcome 计算。

## Namespace RNG

`NamespaceRNG` 用 SHA256 对以下材料派生 128-bit seed：

`master_seed + namespace salt + stable object ID + stream`

每次操作都创建独立的 `random.Random`，没有跨 Query 的全局可变 RNG。当前 Pilot 使用 `content`、`surface`、`capability`、`difficulty`；配置同时预留 `quality_noise`、`cost_noise`、`latency_noise` 和 `split`。稳定 ID 使新增 Query、改变输入顺序或改变无关 namespace 不会扰动已有对象的对应属性。

## 三层 Surface 工作流

Layer 1 在 `semantic_frames.py` 中提供 40 个人工语义框架，每个 task 8 个。每个框架包含双语 scenario/goal、所需输入、可选约束、允许的 capability 集合、额外禁止组合、输出格式与 difficulty knobs。

Layer 2 为每条 Query 组合 2–4 项 capability（1 primary、1–3 secondary），生成归一化权重、`[0.35, 0.95]` 内的 requirement level、`[0, 1]` 内的 difficulty、上下文变量、可见约束、输出要求和 task-conditioned 语言/翻译方向。组合必须属于 semantic frame 的允许集合，并通过禁止组合校验。

Layer 3 提供直接请求、场景化请求、约束前置和约束后置四类布局，并从自然任务要求中选择不同句式和中英文 cue。生成器拒绝显式 capability label、model ID、winner 或 outcome shortcut。

## Pilot 生成与产物

运行：

```powershell
.\.venv\Scripts\python.exe experiments\make_v0_2_pilot.py
```

生成：

- `data/router_v0_2/pilot/pilot_queries.jsonl`
- `data/router_v0_2/pilot/pilot_sidecar.jsonl`
- `data/router_v0_2/pilot/pilot_manifest.json`
- `outputs/router_v0_2/pilot/pilot_audit.json`
- `outputs/router_v0_2/pilot/human_surface_review.csv`

JSON 使用 UTF-8、LF、稳定 key/order 并拒绝 NaN/Inf；CSV 使用 UTF-8 BOM 和 LF。所有内容先在内存中完成序列化，再通过同目录 temporary file 与 atomic replace 写入。Pilot core manifest 记录 canonical config 与其余四个产物的 SHA256；它没有正式 benchmark 的 split、outcome、Final seed 或 access ledger 身份。

## 自动审计

`pilot_audit.json` 检查数量与 task 平衡、primary/secondary capability 与 pair 分布、语言/翻译方向、difficulty、长度、40 个 frame 覆盖、exact/normalized duplicate、ID/family 格式、surface evidence、显式标签/模型结果文本、高频固定句以及 token–capability 强关联候选。

token–capability 关联只是需要人工检查的风险清单。自动检查只能决定是否可以进入人工审查，不能把 Human Surface Audit 标为 PASS。

## Human Surface Review

`human_surface_review.csv` 向 Reviewer 提供 Query、task、最小 capability/difficulty rubric、语言/翻译方向和 semantic scenario，以及八项空白评分字段：Naturalness、Task completeness、Constraint consistency、Capability cue naturalness、Template artifact、Shortcut risk、Difficulty plausibility、Ambiguity/unanswerable risk。另有 `reviewer_note` 和 `reason_code`。

Review packet 不包含 model profile、quality、cost、latency、quality-best model、efficient winner、sampled outcome 或 Router result。团队成员必须实际完成评分；本阶段不预填或推断人工结论。

## 当前限制与 Phase 1B 入口

Pilot 只有 internally authored synthetic Query，有限的 frame、词句池和 surface 布局仍可能形成模板感或词汇捷径。自动关联候选和人工评分用于发现这类问题。当前数据没有 Query × Model outcome，不能评估 quality/routing heterogeneity、noise dominance、baseline difficulty 或 Router 效果，也不能支持任何真实模型、成本、时延或 Kunpeng 结论。

只有真实团队成员完成 Pilot Human Surface Review，并按批准规格判定其通过后，才进入 Phase 1B：生成 2,400 条 Minimum Benchmark、实现 quality/cost/latency generator 和独立 Benchmark Validator。若人工审查发现问题，应先修订 surface 机制并提升相关版本，不能直接进入 Phase 1B。
