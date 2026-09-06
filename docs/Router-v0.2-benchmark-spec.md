# Router v0.2 Synthetic Benchmark 设计规格

> 状态：Draft for review
>
> 文档版本：0.2
>
> 日期：2026-09-07
>
> 基线版本：`router-mvp-v0.1`
>
> 本阶段范围：只定义数据集与实验设计，不实现生成器、Predictor 或 Router

## 1. Motivation

Router v0.1 已经验证了从历史数据、特征提取、每模型质量预测到质量约束路由与离线评价的完整工程链路。v0.2 当前最重要的问题不是提高某个 Router 的测试分数，而是建立一个能够公平检验“Query 中的细粒度信息是否有路由价值”的 Synthetic Benchmark。

本规格把 benchmark 看作一个受控实验环境。它需要同时满足两点：

1. 在无噪声期望值中，模型选择差异由可解释的 Query capability requirement 与 Model capability profile 交互产生；
2. 在只允许读取真实 Query 的条件下，学习方法有机会恢复其中一部分结构，但任何特定 Query-aware 方法都不被预设为赢家。

评价目标继续遵循项目原有优先级：先满足质量阈值，再在达标模型中最小化成本，成本可比时再最小化时延。除明确标为 diagnostic oracle 的分析外，任何可部署策略都不能在决策前读取当前 Query × Model 的 outcome。

### 1.1 Benchmark Construction 与 Router Evaluation 的阶段边界

**Benchmark Construction** 只允许根据 benchmark 自身的结构审计修改 generator、model profile、taxonomy 与 surface realization。允许依据的证据限于 Distribution、Human Surface Quality、Quality/Route Heterogeneity、Learnability Diagnostic、Noise Dominance 和 Baseline Difficulty Audit。任何数据参数都不得根据 Query-aware Router 相对 Baseline 的胜负、排名或指标差值进行调节。

**Router Evaluation** 从 benchmark 的 dataset/generator/config/profile/split 版本冻结时开始。此后 Router 可以按预注册协议在 Train/Validation 上训练和选择，并在 Development Test 上作一次阶段性检查，但不得反向修改 benchmark 结构。任何影响生成语义、分布、表面表达或随机流的结构变更，都必须提升对应 dataset/generator 版本并重新运行全部结构审计；原版本结果与新版本结果不得拼接。

## 2. v0.1 Benchmark Limitations

v0.1 的核心结构是 coarse task、标量 difficulty、模型任务加成、共同 Query/Group shock 和独立噪声。其工程实现是可复现的，但作为算法比较 benchmark 存在以下限制：

- 预设 Expert Rule-based 的 task hint → model 映射与生成器中的模型任务专长高度一致，规则获得了较强的专家先验；
- 同一 coarse task 内虽然有难度差异，但模型间的系统性交叉优势不足，最优模型变化较多来自粗任务和随机扰动；
- 对所有模型相同的 Query shock 可以改变总体达标难度，却不能独立创造模型间可学习的排序差异；
- 少量共享要求句和模板变体可能让模型学习表面捷径，不能充分代表 unseen wording 或能力组合泛化；
- 单一 seed、单次 split 和 600 个 Query 不足以量化结果稳定性；
- v0.1 结果只能说明该固定模拟世界中的表现，不能推出真实模型、真实 API 或企业场景结论。

这些限制不构成 test label leakage，也不否定 v0.1 的工程价值。v0.2 的任务是扩大可检验的问题范围，而不是修改或美化 v0.1 结果。

## 3. v0.2 Research Questions

- **RQ1：**如何让同一粗粒度任务内部出现稳定、可观察、可学习的模型选择差异？
- **RQ2：**如何避免 benchmark 被单一 task → model 规则轻易解决？
- **RQ3：**如何证明主要差异来自 Query capability × Model capability 的结构性交互，而不是随机噪声？
- **RQ4：**如何证明 Train、Validation、Development Test 和 Frozen Final Test 之间没有明显模板、改写族或 outcome 信息泄漏？
- **RQ5：**如何证明结论对生成 seed、split seed 和小规模数据扰动具有足够稳定性？
- **RQ6：**哪些证据能够支持 benchmark 具有合理的内部有效性，同时不把 synthetic 结果误述为真实环境外部有效性？

## 4. Definition of Benchmark Quality

| 属性 | 本项目中的含义 | 可检查要求 |
|---|---|---|
| Representativeness | Query 包含真实任务意图、上下文、约束和输出要求，不是标签词与数字的简单拼接 | 每个 capability 至少覆盖 3 个语义场景族和 4 种表面表达；任何固定样板句不得覆盖该 capability 超过 20% 的 Query |
| Human Surface Quality | Query 表面自然、任务完整、约束自洽，且不会暴露人工标签或明显模板捷径 | 分层人工审计覆盖 Naturalness、完整性、一致性、cue 自然度、模板痕迹、shortcut、难度可信度和歧义/不可答风险 |
| Within-task Heterogeneity | 同一 task 的不同能力需求产生不同质量优势与有效路由区域 | Expected quality-best model 与 Expected efficient winner 分别达到第 12、18 节的多样性门槛 |
| Learnable Structure | 产生差异的 cue 出现在可见 Query 中 | out-of-family Dev Test 上，Text diagnostic 相对 task-only 有明确增益，移除 capability cue 后显著下降 |
| Model Diversity | 模型有交叉优势，而不是单一强弱序列 | 任意两个模型都不得在所有 capability 上被另一个模型以 0.05 以上幅度支配 |
| Non-trivial Routing | 固定模型和 task-only 规则不能接近完成全部路由 | task-only 对 Expected quality-best model 的 modal coverage 每个 task 不高于 0.65；Expert Task Rule 与 Expected efficient winner、Sampled Outcome Oracle 均保留可测差距 |
| Controlled Stochasticity | 噪声模拟重复调用波动，但不决定主要 winner 结构 | quality-best 与 efficient 两类无噪声/采样 agreement 总体均至少 0.75；交互信号与独立噪声方差比至少 2.0 |
| Reproducibility | 数据、配置、seed、版本和 hash 可追踪 | 同 config、版本和 seed 的数据及 split 文件字节一致；非确定时间字段不进入稳定内容 hash |
| Split Integrity | 同源模板、改写和派生 Query 不跨 split | 所有 family ID 交集为零；exact/normalized duplicate 为零；高相似跨 split 对必须人工审计 |
| Diagnostic Verifiability | benchmark 的设计目标可由独立 audit 验证 | Distribution、human surface、quality/routing heterogeneity、learnability、noise、baseline 和 multi-seed 报告齐全 |

上述门槛在看到任何候选 Router 的 Frozen Final Test 结果前确定。后续只能基于 benchmark 自身的结构审计修改生成器；不能因为某种 Router 输赢而反向调整数据。

## 5. Query Taxonomy

v0.2 保留五个 coarse task，便于与 v0.1 对照，但 coarse task 只描述请求的主要形式，不决定模型 winner：

| Coarse task | 典型场景 | 必须覆盖的任务内变化 |
|---|---|---|
| code | 代码生成、理解、修复、数据处理 | 算法、调试、SQL、API、重构、长上下文、结构化输出 |
| math | 计算、符号操作、证明、统计 | 算术、方程、多步推理、概率统计、几何、格式与精度约束 |
| qa | 事实问答、技术解释、上下文问答 | factual、多跳、领域推理、长上下文、歧义消解、结构化回答 |
| summary | 会议、报告、研究材料压缩 | 事实一致性、覆盖率、长上下文、术语、结构和冲突信息处理 |
| translation | 通知、合同、技术材料翻译 | 忠实度、术语、地道表达、长上下文、格式保留和领域适配 |

每个 Query 由以下组成部分生成：

\[
q = (t, A_q, r_q, d_q, s_q, o_q)
\]

- \(t\)：coarse task；
- \(A_q\)：2–4 个活跃 capability，至少包含一个主能力和一个辅助能力；
- \(r_q\)：各活跃能力的需求强度；
- \(d_q\)：整体难度与推理负载；
- \(s_q\)：语义场景、实体、输入材料及变量约束；
- \(o_q\)：表面表达、语言、格式和措辞。

Query 生成采用四层 family：

1. `source_family`：同一个语义场景或原始请求种子；
2. `template_family`：相同意图与约束结构；
3. `paraphrase_family`：同一语义内容的改写集合；
4. `query_variant`：实体、长度、难度和可选约束的具体实例。

同一上游 family 的所有派生项必须进入同一 split。表面生成至少混合人工编写的语义框架、句式级改写、约束顺序变化、同义表达、中英文及混合语言表达。变量必须改变任务内容或约束，不能只是替换对质量无影响的数字。

语言分布按 task 的真实表达需要配置，不使用全局固定比例。`benchmark-config-v1` 的初始目标如下；后续只能依据语言真实性与结构审计修订并提升版本：

| Coarse task | 中文 | 中英混合 | 英文 | 说明 |
|---|---:|---:|---:|---|
| code | 55% | 35% | 10% | 仅代码、API 名和英文标识符不使中文请求自动归为 mixed |
| math | 80% | 10% | 10% | 覆盖中文题面、双语符号说明与英文题面 |
| qa | 70% | 15% | 15% | 领域术语可以保留英文，但分类由主要自然语言决定 |
| summary | 65% | 10% | 25% | 按主要输入材料语言统计；mixed 包括双语材料或跨语言说明 |
| translation | — | — | — | 单独按方向分层：中译英 40%、英译中 40%、双语修订/术语统一 20% |

该设计使总体语料以中文为主，同时保留比赛和企业技术场景中合理的英语与混合语言请求。Audit 必须报告 `task × language/direction` 分布；在适用的主要 capability subgroup 内至少覆盖两种语言形态，并检查语言本身是否成为 capability 或 winner 的单一捷径。

未来可以引入公开 Benchmark Query 作为 source/template seed，但实施前必须记录来源、许可、清洗规则和 family 归属。本阶段不下载任何公开数据。

## 6. Capability Taxonomy

v0.2 采用 `capability-taxonomy-v1` 的 12 个跨任务能力维度：

| ID | Capability | Query 中的自然线索 | 主要适用 task |
|---|---|---|---|
| AR | algorithmic_reasoning | 复杂度要求、状态转移、算法比较、边界证明 | code、math、qa |
| SQ | symbolic_quantitative | 公式、变量、精度、符号变换、数值推导 | math、code、qa |
| DE | debugging_error_localization | 错误日志、失败样例、定位并修复、回归约束 | code |
| DS | data_sql | 表结构、聚合、连接、数据校验、SQL 约束 | code、qa |
| AU | api_library_usage | 指定接口、库版本、调用约束、错误处理 | code、qa |
| FK | factual_domain_knowledge | 专业概念、事实核查、领域术语 | qa、summary、translation |
| MH | multi_hop_reasoning | 多份证据、依赖链、条件组合、反事实 | qa、math、code |
| LC | long_context_integration | 多段材料、跨段引用、长输入一致性 | qa、summary、translation、code |
| FT | faithful_transformation | 不新增事实、保留含义、覆盖关键点 | summary、translation、code refactor |
| MT | multilingual_terminology | 双语术语表、专业译名、语言风格 | translation、summary、qa |
| AM | ambiguity_resolution | 指代冲突、缺失条件、多种解释、澄清要求 | qa、summary、translation |
| SC | structured_constraint_following | JSON/schema、固定段落、长度、字段与格式约束 | 全部 task |

每个 `(task, primary capability)` 是基本分析 subgroup。高频 capability pair 是组合 subgroup。推荐数据中每个主 subgroup 至少 120 个 Query，每个预声明高频 pair 至少 60 个 Query；各 split 的最低计数见第 10 节。

Capability label、需求强度、difficulty、family ID 和无噪声期望质量属于 generator/diagnostic metadata。它们必须保存在与 learner 输入隔离的 sidecar 中。训练和推理阶段默认只能读取原始 Query；`task_type_hint` 或其他 hint 必须由 Query 本身通过同一确定性函数或只在 Train 上训练的模型获得。

Query 文本必须包含能够表达 capability 的真实语义线索，但不得嵌入 `[capability=DS]`、隐藏 ID、固定标签前缀或与 outcome 直接对应的人工 token。可见语义线索是研究对象，不属于泄漏；generator hidden label 直接进入 Predictor 才属于泄漏。

## 7. Model Capability Profiles

v0.2 正式采用 6 个匿名开发模型。相较 5 个模型，它提供更充分的交叉优势空间，更容易形成 capability-specific routing；同时仍处于赛题要求的 5–10 个候选模型范围内，Recommended 档的全覆盖计算规模也保持可管理。以下 `model-profile-v1` 数值范围为 `[0,1]`，表示相对能力，不代表任何真实商业模型：

| Model | AR | SQ | DE | DS | AU | FK | MH | LC | FT | MT | AM | SC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev-model-a | .91 | .83 | .78 | .66 | .61 | .73 | .92 | .90 | .70 | .58 | .75 | .76 |
| dev-model-b | .87 | .62 | .93 | .95 | .91 | .60 | .72 | .64 | .57 | .52 | .66 | .82 |
| dev-model-c | .76 | .96 | .62 | .70 | .55 | .72 | .88 | .57 | .65 | .63 | .69 | .92 |
| dev-model-d | .58 | .54 | .50 | .61 | .64 | .74 | .67 | .81 | .96 | .95 | .76 | .86 |
| dev-model-e | .63 | .60 | .68 | .72 | .70 | .91 | .71 | .52 | .72 | .69 | .93 | .65 |
| dev-model-f | .72 | .69 | .73 | .76 | .79 | .68 | .74 | .94 | .85 | .77 | .70 | .96 |

设计中的交叉优势为：A 偏复杂推理与长上下文，B 偏调试/SQL/API，C 偏数学与严格结构，D 偏忠实变换与多语言，E 偏事实知识与歧义消解，F 偏长上下文与复杂约束。任何模型都不在全部维度支配另一个模型。

Coarse task interaction 仅允许作为小幅残差项，绝对值不超过 `0.04`：

| Model | code | math | qa | summary | translation |
|---|---:|---:|---:|---:|---:|
| dev-model-a | .01 | .02 | .03 | .00 | -.02 |
| dev-model-b | .04 | -.02 | -.01 | -.02 | -.03 |
| dev-model-c | -.01 | .04 | .01 | .00 | -.01 |
| dev-model-d | -.03 | -.03 | .00 | .03 | .04 |
| dev-model-e | .00 | -.01 | .04 | .01 | .00 |
| dev-model-f | .01 | .00 | -.01 | .03 | .02 |

每个模型还可以声明少量 capability synergy，绝对值不超过 `0.08`。初始正向组合为 A:`AR×MH`、`MH×LC`；B:`DE×AU`、`DS×SC`；C:`SQ×SC`、`SQ×MH`；D:`FT×MT`、`FT×LC`；E:`FK×AM`、`DS×FK`；F:`LC×SC`、`AU×SC`。未声明组合为零。任何后续修改都必须更新 profile 版本并重新运行完整 benchmark audit。

模型成本、速度和能力不得共用同一个排序。建议初始运行参数如下，单位仅为 synthetic benchmark 单位：

| Model | input price | output price | startup ms | prefill units/s | decode units/s | difficulty resilience |
|---|---:|---:|---:|---:|---:|---:|
| dev-model-a | .040 | .060 | 800 | 7000 | 45 | .90 |
| dev-model-b | .022 | .035 | 500 | 9000 | 65 | .78 |
| dev-model-c | .028 | .042 | 650 | 8000 | 55 | .84 |
| dev-model-d | .015 | .026 | 450 | 10000 | 70 | .72 |
| dev-model-e | .007 | .014 | 300 | 6500 | 80 | .68 |
| dev-model-f | .020 | .030 | 380 | 12000 | 75 | .82 |

这些数值是待审查的 synthetic 参数，不是实测性能。正式实现只能根据第 12、13 节的 benchmark 结构门槛调整它们，不能根据 Query-aware Router 是否获胜来调节。

## 8. Quality Generation Mechanism

### 8.1 无噪声结构

对于 Query \(q\)，活跃 capability 集为 \(A_q\)。每个活跃能力具有权重 \(w_{qk}\) 与需求水平 \(r_{qk}\)，其中 \(\sum w_{qk}=1\)，\(r_{qk}\in[0.35,0.95]\)。模型能力为 \(c_{mk}\)。定义 margin：

\[
g_{qmk}=c_{mk}-r_{qk}
\]

能力匹配同时使用平均覆盖与瓶颈覆盖：

\[
M_{qm}=\sum_{k\in A_q}w_{qk}g_{qmk}
\]

\[
B_{qm}=-\tau\log\left(\sum_{k\in A_q}w_{qk}\exp(-g_{qmk}/\tau)\right),\quad \tau=0.12
\]

`M` 表示整体匹配，soft-min `B` 对某一必需能力明显不足的模型施加额外惩罚。组合交互为：

\[
S_{qm}=\sum_{k<l,\,k,l\in A_q}2\sqrt{w_{qk}w_{ql}}\,s_{mkl}
\]

其中 \(s_{mkl}\) 是第 7 节的稀疏 synergy。模型–Query 的 latent score 为：

\[
z_{qm}=1.35+b_m+\delta_{m,t(q)}+1.35M_{qm}+0.65B_{qm}+S_{qm}
-1.10d_q(1-\rho_m)+u_{family(q)}+u_q
\]

- \(b_m\)：模型小幅全局偏置，限制在 `[-0.08,0.08]`；
- \(\delta_{m,t}\)：小幅 model-task residual，限制在 `[-0.04,0.04]`；
- \(d_q\in[0,1]\)：由步骤数、上下文长度、约束数量和冲突程度组成的可解释难度；
- \(\rho_m\)：difficulty resilience；
- \(u_{family}\)、\(u_q\)：均值为零的小幅共同效应，只改变总体难度，不作为模型排序的主要来源。

无噪声期望质量为：

\[
\mu_{qm}=0.02+0.96\,\operatorname{sigmoid}(z_{qm})
\]

主评价固定使用 `quality_score_threshold = 0.8`。`0.7` 与 `0.9` 只作为预注册 sensitivity thresholds，在主结果完成后报告稳健性；它们不得用于 feature selection、Router threshold tuning、benchmark construction 或主排名。正式报告必须将 `0.8` 标为 **Primary Result**，将 `0.7/0.9` 标为 **Sensitivity Result**。全局截距只能为了满足预先声明的 pass/fail 分布门槛而在 Benchmark Construction 阶段调整；不得针对任何 Router 的相对结果调整。

### 8.2 受控随机性

一次模拟观测为：

\[
y_{qm}=\operatorname{clip}(\mu_{qm}+\epsilon_{qm},0,1)
\]

其中独立噪声 \(\epsilon_{qm}\) 来自均值为零、按模型声明 `σ_m∈[0.025,0.040]` 的截断正态分布，截断在 `±2.5σ_m`。随机流按 `quality-noise/query_id/model_id/repeat_id` 派生，新增 Query 不应改变已有 Query 的噪声。

这一机制产生任务内异质性的原因是：同一 task 的 Query 具有不同的 capability 集合、权重、需求水平与组合效应；模型的交叉能力画像使 winner 随需求系统性变化。task-only rule 只能利用很小的 \(\delta_{m,t}\)，无法恢复 \(M\)、\(B\) 和 \(S\)。Query-aware 方法可以从 Query 中自然出现的能力、长度和约束线索学习这些交互。独立噪声只在结构期望值附近模拟波动，第 12 节要求用重复采样证明它没有主导 winner。

### 8.3 三个不同的质量与路由概念

Benchmark audit 与实验报告必须严格区分以下三个概念：

- **Expected quality-best model**：`argmax_m μ_qm`，只表示无噪声期望质量最高的模型，完全忽略 cost 和 latency。精确并列时按 model ID 确定性打破，并同时报告并列数与第一、第二名 margin。
- **Expected efficient winner**：先用 \(\mu_{qm}\) 和主质量阈值判断达标模型，再按 expected cost、expected latency、model ID 选择；若无模型达标，按最高 \(\mu\) fallback。它表示 benchmark 无噪声世界中质量→成本→时延→fallback 目标的有效选择。
- **Sampled Outcome Oracle**：读取当前 Query 全部模型的 sampled quality/cost/latency 后事后执行相同决策目标，是不可部署的 hindsight upper bound，只用于测量采样结果下的理论空间。

Quality structure 使用 Expected quality-best model 审计；Routing structure 使用 Expected efficient winner 审计；Sampled Outcome Oracle 只用于噪声与上界分析。报告不得使用未限定的 `expected winner`、`sampled winner` 或 `Oracle` 混称三者，也不得把 Sampled Outcome Oracle 当成可部署 Router。

## 9. Cost / Latency Generation

成本不再只是模型常数。先通过版本固定的 tokenizer/长度估计规则，从最终可见 Query 和输出约束得到 `input_units` 与 `requested_output_units`：

\[
E[cost_{qm}]=p^{in}_m\,input\_units_q+p^{out}_m\,requested\_output\_units_q
\]

采样成本使用保持均值不变的小幅 log-normal 波动，`σ_cost` 建议为 `0.05–0.10`。价格参数与 capability profile 分开配置；质量噪声不得进入成本公式。

期望时延由启动、prefill、decode 和可见推理工作量组成：

\[
E[latency_{qm}]=startup_m+\frac{input\_units_q}{prefill_m}
+\frac{requested\_output\_units_q}{decode_m}+\frac{reasoning\_work_q}{reasoning\_rate_m}
\]

采样时延使用 `σ_latency=0.08–0.15` 的 log-normal 波动。`reasoning_work` 只能由 Query 中可观察的步骤、材料长度和约束数量决定，不能读取 sampled quality。Audit 必须报告 capability 均值与成本/时延的 Spearman 相关；绝对相关系数达到 `0.80` 进入 NEEDS REVISION，达到 `0.95` 判定 FAIL。

## 10. Dataset Scale

默认按 5 个 task、6 个模型、全模型覆盖设计：

| 档位 | Template families / task | Query / family | Query 总数 | Observation 总数 | 用途与预计成本 |
|---|---:|---:|---:|---:|---|
| Minimum | 20 | 24 | 2,400 | 14,400 | 验证 schema、公式和 audit；关键 subgroup 统计功效有限 |
| Recommended | 40 | 30 | 6,000 | 36,000 | v0.2 默认；每模型约 3,000 个 Train Query，稀疏 LR 与统计 audit 仍属轻量 |
| Extended | 80 | 30 | 12,000 | 72,000 | 稳定性与复杂组合研究；存储、重复采样和多 seed 时间约为 Recommended 的 2 倍 |

推荐档每个 task 1,200 Query。按 50/15/15/20 划分时，每个 task 分别为 600/180/180/240 Query。每个 `(task, primary capability)` 推荐总计至少 120，Train 至少 60，Validation 与 Development Test 各至少 15，Frozen Final Test 至少 20；预声明高频 capability pair 总计至少 60，Train 至少 30，其余三个 split 各至少 8。

如果 Minimum 档无法达到上述 subgroup 下限，它只能用于 generator smoke test，不能作为正式 Router 比较数据。正式实现从 Recommended 档起步；只有 audit 表明统计不稳定时才升级 Extended，不能为获得更好结果盲目扩大数据。

按完整 8-seed 协议计算，Recommended 档共处理 288,000 条主 observation；即使再加入 20% Query 的 20 次噪声重复采样，仍属于可在普通 CPU 上分批完成的离线统计规模。Minimum、Recommended、Extended 的主数据计算量约为 `1× / 2.5× / 5×`，多 seed 与重复采样按相同比例增长。实施时必须记录实际生成耗时、峰值内存和文件大小，本规格不把估算值当成 measured performance。

## 11. Dataset Split Protocol

推荐比例为：

- Train：50%，用于所有可学习参数、统计量和预处理 fit；
- Validation：15%，用于 feature、阈值、超参数和候选方案选择；
- Development Test：15%，用于研究过程中的阶段性分析，不回写训练；
- Frozen Final Test：20%，方案与分析计划冻结后只进行正式评估。

划分单位是最上游可追踪 family，而不是 observation、Query 或 model row。`source_family_id`、`template_family_id`、`paraphrase_family_id` 任一相同的 Query 都必须在同一 split。每个 Query × Model 必须完整覆盖并随 Query 一起划分。

Recommended 档每个 task 的 40 个 template family 固定分为 20/6/6/8，对应 Train/Validation/Development Test/Frozen Final Test。分配使用独立 `split_seed`，在生成 outcome 前完成，并按 task、primary capability 和 difficulty bin 分层；不得反复换 seed 直到指标好看。

Frozen Final Test 设计为：

- 75% 是 Train 中出现过 capability 组合的新 template/source family，记为 **Familiar**；
- 25% 是预先声明的新 capability 组合，但每个单独 capability 必须已在 Train 出现，记为 **Novel**；
- 完全未见 capability 只能进入单独的 out-of-taxonomy stress set，不能混入主 Final 指标。

Frozen Final Test 必须分别报告 **Familiar、Novel、Overall** 三组结果；主结果 Overall 不能掩盖 Novel 组合的退化。完全未见 capability 的 stress set 单独版本化、单独报告，不参与主排名。

所有文本预处理、词表、scaler、统计量和模型只能 fit Train。Validation 可用于选择；Development Test 被查看后不得用于继续调参。Frozen Final Test 的首个正式 outcome 一旦被读取，如果之后继续修改算法、规则、feature、阈值、baseline 或分析计划，该集合立即失去严格 final-test 地位，报告必须改称 `previously observed test`；再次正式评价必须预提交新的 Final seed commitment 和数据版本。

### 11.1 Frozen Final Test 治理与 seed commitment

Frozen Final 的治理顺序固定如下：

1. 先冻结 generator、schema、canonical config、model profile、split definition、validator 和分析计划，并记录对应 Git commit；
2. 由不参与日常 Router 调参与方案选择的 custodian 离线生成高熵 `final_master_seed`，普通开发仓库只提交 `SHA256(final_master_seed)` commitment，不保存明文 seed；
3. 冻结候选算法、feature、所有训练/选择阈值、baseline、主指标与 sensitivity 分析；
4. custodian 在登记时间公开 seed，评价者先验证其 SHA256 与 commitment 一致，再按冻结 commit 生成 Frozen Final Query 与 diagnostic sidecar；
5. 在生成或读取任何 Final outcome 前完成 Final Query 的 split/family/duplicate 与 Human Surface Quality 检查；如果 Query surface 未通过，当前 Final 版本作废并记录原因，不得挑选、重抽或修补个别 Query；新版本必须使用新的预提交 commitment 重走完整流程；
6. Query audit 通过后生成 Final outcome，只执行一次正式评价，并记录 seed reveal 时间、代码 commit、evaluator、文件 hash、每次 access event 和结果发布时间；
7. 首次 outcome 读取后的任何算法或分析变更，只能把该集合称为 `previously observed test`，或改用新的预提交 Final 版本。

Final release checklist：

- [ ] generator/schema/config/profile/split/analysis commit 已冻结；
- [ ] 明文 seed 由独立 custodian 保管，普通开发仓库中只有 SHA256 commitment；
- [ ] 算法、feature、阈值、baseline 和分析计划已冻结；
- [ ] reveal 时间、custodian、evaluator 和 access ledger 已建立；
- [ ] reveal 后 commitment 验证成功；
- [ ] Final Query 在 outcome 读取前通过完整性与 Human Surface Quality 审计；
- [ ] Familiar/Novel/Overall 和 stress set 的报告边界已锁定；
- [ ] 正式 outcome 只读取一次，所有文件 hash 与访问事件已归档。

该流程通过职责分离、承诺验证和访问留痕降低意外污染与事后挑选的风险，但它不是能够阻止所有恶意行为的安全或密码学沙箱。

## 12. Benchmark Validation Protocol

Benchmark validator 必须独立于 Router 训练入口运行，生成机器可读 JSON 和面向审查的 Markdown 报告。所有标准在首次运行候选 Router 前固定。

### 12.1 A. Distribution Audit

必须报告：

- 每个 split 的 task、primary capability、secondary capability、capability pair、difficulty bin、语言和长度分布；
- 每模型 observation coverage、expected/sample quality、pass/fail、cost 和 latency 分布；
- 每个 Query 的 expected-qualified model 数量；
- family 数、重复文本、近重复文本和跨 split 相似对；
- Train 与其他 split 的 Jensen–Shannon divergence，以及 difficulty standardized mean difference。

Recommended 档验收范围：

- 全覆盖严格等于 `num_queries × num_models`，每对恰好一条主观测；
- task share 相对目标偏差不超过 3 个百分点；
- 非刻意 novel-combination 部分的 capability 分布 JS divergence 不超过 `0.05`；
- difficulty standardized mean difference 绝对值不超过 `0.15`；
- 每模型总体 pass rate 在 `[0.15,0.90]`；每个样本充足的 model-task cell 在 `[0.05,0.95]`；
- 至少 25% Query 有两个及以上 expected-qualified 模型，至少 10% 只有一个，no-qualified 比例建议在 `[0.05,0.25]`。

### 12.2 B. Human Surface Quality Audit

Recommended 档一个完整生命周期默认人工审查 **288 个 Query**：Benchmark Construction 审查 240 个，Final reveal 后再审查 48 个真实 Final Query。该总量位于 150–300 的可管理区间内；最终发布数据的验收统计仍使用 240 个不重复 Query（192 个非 Final + 48 个真实 Final），construction 时的 48 个 final-like shadow 只用于提前发现机制问题，不进入最终分母。这样既为五个 task 提供足够覆盖，也不会把审查变成第二个标注项目。样本采用多维配额加稀有 subgroup 过采样，不要求对所有维度做不可实现的全笛卡尔积：

- 按 split 分配 Train 120、Validation 36、Development Test 36、Frozen Final Test 48；
- 同时跨 task、primary capability、高频 capability pair、difficulty bin、language/direction 分层；
- 至少一半样本来自多 capability、较高 difficulty、mixed language、Novel 组合或其他预声明高风险单元；
- Benchmark Construction 阶段先审查 192 个非 Final 样本，并从公开 audit seed 生成 48 个 final-like shadow 样本；隐藏 seed reveal 后，再在读取 outcome 前用真实 Final Query 替换 shadow 样本完成最终 240 项审计。

每个 Query 按以下八个维度评分：

1. `Naturalness`：是否像真实用户请求；
2. `Task completeness`：输入、目标和必要上下文是否足够；
3. `Constraint consistency`：约束是否相互兼容；
4. `Capability cue naturalness`：能力线索是否由任务自然表达，而非标签化提示；
5. `Template artifact`：是否存在重复骨架、占位符或机械拼接痕迹；
6. `Shortcut risk`：词汇、格式或语言是否可直接映射 hidden capability/model winner；
7. `Difficulty plausibility`：标称难度是否与可见请求相称；
8. `Ambiguity/unanswerable risk`：是否存在非设计目的的歧义、矛盾或不可答问题。

所有维度使用简单三级量表：`0 = Fail`、`1 = Acceptable`、`2 = Good`。对风险维度，`2` 表示未发现实质风险，`1` 表示轻微但可接受，`0` 表示不可接受。至少两名 reviewer 独立审查；默认 80 个样本由两人重叠评分，其余均由一人评分。重叠样本出现 2 分分歧、任一 reviewer 判定整体无效，或双方对 `Template artifact`/`Shortcut risk` 是否为 0 不一致时，由第三人裁决。报告每维 exact agreement、within-one agreement 和简单 unweighted Cohen's κ；κ 只用于解释一致性，不单独决定通过。

以下是预注册的项目内部门槛：

| 判定 | Human Surface Quality 要求 |
|---|---|
| PASS | 八维均值均≥1.50；`Naturalness`、`Task completeness`、`Constraint consistency`、`Ambiguity/unanswerable risk` 得分≥1 的比例均≥95%；`Template artifact=0`≤3%，`Shortcut risk=0`≤2%；重叠样本 exact agreement≥75%、within-one≥95%；任一有至少 20 个审计样本的预声明 subgroup 的整体无效率≤5% |
| NEEDS REVISION | 无 FAIL 条件，但任一维均值在 `[1.25,1.50)`、任一核心有效比例在 `[90%,95%)`、template Fail 在 `(3%,7%]`、shortcut Fail 在 `(2%,7%]`、exact agreement 在 `[60%,75%)`、within-one 在 `[90%,95%)`，或任一样本充足 subgroup 的整体无效率在 `(5%,10%]` |
| FAIL | 任一维均值<1.25，任一核心有效比例<90%，template/shortcut Fail>7%，exact agreement<60%或within-one<90%，存在系统性标签/模型捷径，或任一样本充足 subgroup 的整体无效率>10% |

Reviewer 必须记录简短原因代码与可复核样例。任一核心有效性维度为 0，或 `Template artifact`/`Shortcut risk` 为 0，即把该 Query 计为“整体无效”。发现 FAIL 时只能按第 1.1 节提升版本并重做全部审计；Final reveal 后不得通过重抽或定点修补来保留原 Final 身份。

### 12.3 C. Quality and Routing Heterogeneity Test

**Quality structure** 只使用 Expected quality-best model，计算：

1. 每个 task 的 quality-best 分布、normalized entropy `H(W_quality|T=t) / log(num_models)` 和 task-only modal coverage；
2. 每个 `(task, primary capability)` 与高频 capability pair 的 modal quality-best、winner share、margin 和 bootstrap 95% CI；
3. `I(W_quality; capability | task) / H(W_quality | task)`；
4. capability requirement 做 ±0.03 小扰动时 quality-best 的稳定率；
5. 已知生成公式的 component ablation/方差分解，分别报告 capability match (`M/B/S`)、coarse task residual、model bias、difficulty interaction 和共同效应对结构化 model×query 差异的贡献。

Quality PASS 要求每个 task 至少 3 个模型各占 10%，normalized entropy 至少 `0.45`，task-only modal coverage 每个 task 不超过 `0.65`、宏平均不超过 `0.60`；每个 task 至少有 3 个样本充足的 capability subgroup，且其中至少出现 2 个不同 modal quality-best。Conditional MI 应不低于 `0.15`；低于 `0.08` 表示 capability 结构过弱。Capability match 与 synergy 应解释至少 60% 的结构化 model×query 方差，coarse task residual 不得超过 20%。结构归因必须使用 generator component 的反事实消融，不能用某个 Predictor 的拟合优度替代。

**Routing structure** 只使用同一冻结 RoutingPolicy 下的 Expected efficient winner，计算：

1. 每个 task 的 efficient-winner 分布、normalized entropy 与 modal coverage；
2. 每个 capability subgroup 的 efficient-winner 变化和 expected-qualified model 数；
3. Expected efficient winner 与 Expected quality-best model 的 agreement；
4. 因成本优先而选择“已达标但不是 quality-best”的 Query 比例及其质量 margin、cost saving；
5. latency tie-break 的触发比例，以及它是否来自预声明的正成本容差或真正等成本；
6. fallback 比例及 fallback 与 Expected quality-best model 的一致性。

Routing PASS 要求每个 task 至少 2 个 efficient winner 各占 10%，normalized entropy 至少 `0.35`，modal coverage 不超过 `0.75`；每个 task 的样本充足 capability subgroup 至少出现 2 个不同 modal efficient winner。成本导致选择达标的非 quality-best 模型的比例建议在 `[0.10,0.70]`，超出时进入成因审查。Latency tie-break 必须完整报告；当冻结 policy 的 `cost_tolerance_abs=0` 且没有真正等成本时，该比例可以为零，不能为了产生时延作用而人为放宽容差。Quality 与 Routing 两组结果必须分别成表，不能用 routing 多样性替代 quality 多样性。

### 12.4 D. Learnability Test

使用固定超参数的轻量 diagnostic learner，在 out-of-family Development Test 上比较：

- `Hidden capability oracle`：读取真实 capability 向量，只用于诊断上界；
- `True task-only diagnostic`：只读取 hidden coarse task，用于测量 task→model 的结构上限，不属于可部署方法；
- `Query-derived hint`：只从文本得到 task hint；
- `Text / Structural`：只读取部署时可见的 Query；
- `Cue-removed text`：删除或改写与 capability 相关的语义片段，保留 task 与长度近似；
- `Within-task shuffled text/features`：在同 task 内打乱 capability 与 Query 的对应关系。

主要结构诊断指标为 Expected quality-best model 的 macro-F1/top-1 accuracy；另行报告每模型 pass prediction log loss/Brier score，以及相对 Expected efficient winner 的 expected feasibility regret、cost regret 和 routing agreement。定义：

\[
recovery=\frac{score_{text}-score_{task}}{score_{hidden\ oracle}-score_{task}}
\]

当 oracle-task 差距至少 `0.15` 时，PASS 要求 Text 相对 task-only 的 macro-F1 提升至少 `0.08`、`recovery ≥ 0.50`、相对 within-task shuffled 提升至少 `0.10`；移除 capability cue 后 macro-F1 至少下降 `0.07`，或 expected routing regret 恶化至少 15%。如果 Hidden oracle 明显有效而 Text 不可学习，说明 surface realization 没有可靠承载 capability；如果 shuffled 不下降，说明观察到的增益不是来自所设计的结构。

这些门槛用于验证“信息存在且可学”，不要求当前 Router 或某一算法击败所有 Baseline。

### 12.5 E. Noise Dominance Test

在每个 seed 中抽取分层的 20% Query，对每个 Query × Model 重复采样 `R=20` 次，不改变 latent Query：

- 比较 Expected quality-best model 与每次 sampled quality-best model；
- 比较 Expected efficient winner 与每次 Sampled Outcome Oracle；
- 分别计算两种比较的 overall/per-task agreement；
- 对去除 task main effect 和 model main effect后的 \(\mu_{qm}\) 交互项做方差分解；
- 定义 `interaction_SNR = Var(structured query×model interaction) / Var(independent noise)`；
- 统计 capability subgroup 的 modal winner 在重复采样和 seed 间是否稳定；
- 报告共同 family/query shock 的方差，它不能计入模型排序信号。

PASS 要求 quality-best/sample quality-best 与 efficient/Sampled Outcome Oracle 两种 overall agreement 均至少 `0.75`，每个 task 均至少 `0.65`，`interaction_SNR ≥ 2.0`，至少 80% 样本充足的 capability subgroup 保持与对应无噪声分析相同的 modal winner。任何主要 winner 多样性如果只在加入噪声后出现，直接判定 FAIL。

### 12.6 F. Baseline Difficulty Audit

在训练正式 Router 前运行：Random、Always Strongest、Always Lowest Cost、Global Historical、Expert Task Rule、Learned Task Rule、Hidden capability oracle 和 Sampled Outcome Oracle。后两者必须明确标为 diagnostic/hindsight，不能作为可部署算法。

至少检查：

- Expected/Sampled quality pass rate、average quality、cost、latency、fallback；
- 对 Expected efficient winner 的 agreement，并与 Expected quality-best model 的 agreement 分列；
- 相对 Always Strongest 的 cost saving，但只在质量差异可接受时解释；
- Expert Rule 与 Learned Task Rule 的差异，用于量化专家先验；
- Sampled Outcome Oracle 与最佳可部署 baseline 的间距，用于量化剩余区分空间。

Benchmark 不要求 Query-aware 必须获胜。推荐接受区间为：Always Strongest pass rate 低于 `0.97`；Always Lowest Cost pass rate 在 `[0.15,0.75]`；Expert Task Rule 对 Expected efficient winner 的 agreement 不高于 `0.65`；Sampled Outcome Oracle 相对 Expert Rule 至少存在 `0.08` 的 pass-rate 差距，或在相近 pass rate 下存在至少 15% 成本差距。若任何 task-only/fixed 方法对 Expected efficient winner 的 agreement 超过 `0.80`，判定 benchmark 结构过于简单。

## 13. Baseline and Comparison Matrix

正式 v0.2 实验计划包含以下十项，并固定信息边界：

| 方法 | 决策前允许的信息 | 目的 |
|---|---|---|
| Random | 候选集合 | 随机下界 |
| Always Strongest | Train 全局质量统计 | 固定高质量基线 |
| Always Lowest Cost | Train 全局成本统计 | 固定低成本基线 |
| Global Historical | Train 全局 pass/cost/latency | 无 Query 信息的历史基线 |
| Expert Rule-based | 预声明 Query-only hint 与专家映射 | 测量人工先验价值 |
| Learned Task Rule | Train 中 query-derived hint × model 统计 | 测量从历史学习粗任务规则的能力 |
| Hint-only Predictor | Query-derived hint | 测量粗任务特征可解释度 |
| Text / Structural Predictor | 原始 Query 及可见结构 | 测量细粒度 Query 信息价值 |
| Text / Structural + Hint | 原始 Query、可见结构、query-derived hint | 测量显式 hint 的增量价值 |
| Sampled Outcome Oracle | 当前 Query 的全模型 sampled outcome | 仅作为 hindsight 上界 |

核心比较应按预注册问题解释，而不是按单一排行榜解释：Expert vs Learned 衡量专家先验；Hint-only vs Text 衡量粗任务之外的信息；Learned Task Rule vs Query-aware 衡量任务内特征；Text vs Text+Hint 衡量显式 hint；Query-aware vs Sampled Outcome Oracle 衡量剩余空间。

## 14. Multi-seed Stability Protocol

Recommended 档的 Benchmark Construction 与 Development Stability Audit 使用 **8 个预先声明的公开 development seeds**。每个 development seed 通过 SHA256 加 namespace 派生 `content`、`surface`、`quality_noise`、`cost_noise`、`latency_noise` 和 `split` 随机流，禁止共享一个可变 RNG 流。禁止删掉不利 seed 或只报告最好 seed。这八个公开 seed 与第 11.1 节由 custodian 保管的隐藏 `final_master_seed` 相互独立，不能用任一 development seed 代替正式 Final seed。

主报告对八个 seed 给出 mean ± std，并保留每个 seed 的原始结果，至少覆盖：

- Quality Pass Rate、Average Quality；
- Average Cost、相对预声明 baseline 的 Cost Saving；
- Average/P95 Latency；
- Fallback Rate；
- Predictor Log Loss、Brier Score；
- Query-aware 与关键 baseline 的逐 seed 差值和相对排名。

另外进行两项敏感性检查：

1. **Split sensitivity**：固定三个最先声明的 generation seed，各使用四个预声明 split seed，共 12 次；
2. **Local perturbation**：对前三个 seed 分别移除 5% family、重采样 5% surface realization、将连续参数做不改变排序意图的 ±3% 扰动。

PASS 要求第 18 节中适用于公开 development seed 的自动化 Hard Gates 至少在 7/8 个 seed 中通过；关键 baseline 差值方向至少在 6/8 个 seed 中一致，否则报告为不稳定而不是选择有利 seed。Human Surface Quality 在主候选 construction 与实际 Final 上分别执行，不要求对八个 development seed 各审 240 条。建议 Quality Pass Rate 的跨 seed std 不超过 `0.03`、Fallback Rate 不超过 `0.04`、Average Cost 的相对 CV 不超过 `0.10`、Log Loss std 不超过 `0.05`、Brier std 不超过 `0.03`。超过建议值进入 NEEDS REVISION，并优先增加样本或检查结构，而不是更换 seed。Router 与 baseline 的跨 seed 差值属于 benchmark 冻结后的 Evaluation Stability 结果；它不得作为回调 generator/profile/surface 的依据。

## 15. Leakage Prevention

必须执行以下控制：

1. **先 split、后 fit**：任何词表、scaler、统计量、规则学习、阈值和超参数都不得在 split 前 fit；
2. **family isolation**：source/template/paraphrase family 的跨 split 交集必须为零；
3. **文本去重**：检查原文、Unicode/空白/标点归一化文本，以及字符 5-gram Jaccard；跨 split Jaccard `>0.85` 的 pair 全部进入审计清单；
4. **隐藏字段隔离**：`coarse_task`、capability、difficulty、profile、\(\mu\)、noise 和 outcome 不进入 deployable Predictor；
5. **Query-derived hint**：Train 与推理使用同一函数或 Train-only classifier；不得用 generator true task 补齐；
6. **Outcome boundary**：选择模型后才查询当前 split 中所选 Query × Model 的 sampled outcome；
7. **Baseline boundary**：Strongest、Lowest Cost、Global 和 Learned Task Rule 只使用 Train；Expert Rule 必须在看结果前登记；
8. **Final governance**：先冻结 generator/schema/config/profile/分析，再提交隐藏 Final seed 的 SHA256 commitment；算法、feature、阈值和 baseline 冻结后才能 reveal、验证并生成 Final；
9. **无 test 回调**：读取 Dev Test 后的改动必须记录；首次读取 Frozen Final outcome 后的改动必须把它标为 `previously observed test`，或预提交新 seed/version；
10. **审计追踪**：记录 Final seed reveal 时间、commit、custodian、evaluator、文件 hash 与每次 access event。

Capability cue 自然出现在 Query 中是预期信号。泄漏是把 generator metadata、family ID、真实 task label、expected/sample outcome 或与它们一一对应的人工 token 交给算法。

## 16. Dataset Versioning

建议独立版本：

- `dataset_version`: `router-benchmark-v0.2.0`；
- `generator_version`: `2.0.0`；
- `config_version`: `benchmark-config-v1`；
- `capability_taxonomy_version`: `capability-taxonomy-v1`；
- `model_profile_version`: `model-profile-v1`；
- `split_definition_version`: `family-split-v1`。

Manifest 至少记录：

- `source_kind`、全部版本、generator Git commit、creation timestamp；
- Benchmark Construction 使用的公开 development seed 及 namespace 派生 seed；
- Final manifest 在 reveal 前只记录 `final_master_seed_sha256` commitment；reveal 后的明文 seed 进入受控的不可变评价记录，不进入普通开发配置；
- Query/model/observation 数和 full-coverage 声明；
- task、capability、combination、difficulty、language 与 split 计数；
- noise、quality、cost、latency 参数；
- 完整 model capability profile 与 task/synergy residual；
- family split 定义、Final Familiar/Novel 清单及各自与 Overall 的 hash；
- canonical config、各 split 数据、diagnostic sidecar 和 audit report 的 SHA256；
- synthetic 限制声明与允许输入字段。

同一版本、config 和 seed 必须生成字节完全相同的 canonical data、sidecar 与 split assignment。JSON 使用 UTF-8、LF、稳定 key/order、固定浮点精度并拒绝 NaN/Inf。`created_at_utc` 属于运行 provenance，可以变化，但不进入 deterministic core manifest hash；其余语义字段必须一致。任何影响数据语义、分布、surface realization 或随机流的代码修改都必须更新 generator 或 dataset 版本，并重跑第 12、14、18 节全部 audit。Final commitment 验证失败、Final surface audit 失败或首次 outcome 读取后发生变更时，不得覆盖原文件；必须登记旧版本状态并创建新的 dataset version 与 seed commitment。

## 17. Experimental Boundaries

高质量 Synthetic Benchmark 可以支持：

- Router 与质量约束决策逻辑验证；
- Query-aware 信息相对 task-only 信息的增量价值验证；
- 在同一受控世界中的 Baseline 公平比较；
- capability/hint/feature/noise ablation；
- Predictor 概率诊断与后续 calibration 研究；
- 可控分布变化和动态场景实验。

它不能单独支持：

- 真实企业成本节约百分比；
- 真实 API 的质量保证或 SLA；
- 真实商业模型能力、成本或速度排名；
- 生产日志中的反事实无偏估计；
- 真实流量分布与长期漂移结论；
- 鲲鹏、openEuler、多核或 NEON 的真实性能收益。

整体验证路线保持为：

`High-quality Synthetic Benchmark → Controlled Real-model Benchmark → Real API / Model Pool → Dynamic Runtime Experiment → Kunpeng / openEuler Deployment`

所有报告必须将 simulated、real-model measured、API measured 和 system measured 分开标识。

只有在第 18 节门槛、family isolation、observable learnability、noise dominance、baseline difficulty 与 multi-seed audit 全部留下可复核证据后，项目才能声称该 benchmark 对所定义的 synthetic world 具有合理的 **internal validity**。这一措辞只表示因果结构、信息边界和比较协议在受控环境内自洽；无论内部审计结果多好，都不能据此声称对真实企业请求、真实模型池或部署平台具有 **external validity**。

## 18. Acceptance Criteria

本节数值统一定义为 **Pre-registered Internal Benchmark Acceptance Criteria / 预注册的项目内部 Benchmark 验收标准**。它们是针对本项目规模、生成机制和研究问题作出的启发式工程选择，用于在看到正式 Router 结果前约束团队判断；它们不是通用学术标准、统计学定理或生产 SLA。首次候选 benchmark audit 前冻结这些门槛；若证据表明门槛本身不适用，只能记录理由、提升规格与数据版本并完整重跑，不能追溯性改变旧版本判定。

### 18.1 判定规则

- **PASS**：所有 Hard Gate 通过，Human Surface Quality Gate 为 PASS，并且 Quantitative Gate 至少 9/10 通过；唯一未通过项必须处于 NEEDS REVISION 区间，且不能涉及 leakage、reproducibility、noise dominance 或 Final governance；
- **NEEDS REVISION**：Hard Gate 全部通过、Human Surface Quality 不为 FAIL，但 Quantitative Gate 只有 7–8 项通过，或关键指标落在灰区；允许仅依据结构审计修改 generator/profile/surface 后重新版本化和完整重跑；
- **FAIL**：任一 Hard Gate 失败、Human Surface Quality 为 FAIL、Quantitative Gate 通过不超过 6 项，或 quality/routing winner 多样性主要只在加入噪声后出现。

### 18.2 Hard Gate checklist

- [ ] 同版本 + config + seed 的 canonical 数据、sidecar、split assignment 字节级复现；
- [ ] Query × Model full coverage 完整，每对恰好一条主观测；
- [ ] source/template/paraphrase family 跨 split 交集为零；
- [ ] 无 exact/normalized duplicate，所有高相似跨 split pair 已审计；
- [ ] deployable 算法无法读取 hidden task/capability/difficulty/expected quality/outcome；
- [ ] Expected quality-best model 与 Expected efficient winner 的任务内多样性都已在无噪声期望值上存在；
- [ ] `interaction_SNR ≥ 2.0`；Expected quality-best model 对 sampled quality-best、Expected efficient winner 对 Sampled Outcome Oracle 的总体 agreement 均至少 0.75；
- [ ] Train-only preprocessing/statistics 与 decision-after-outcome 边界有自动测试；
- [ ] Frozen Final 的隐藏 seed commitment 已在 reveal 前提交；reveal 后 hash 验证成功，冻结 commit、custodian、evaluator 与 access ledger 可核验；
- [ ] 所有 synthetic 结果均有明确 simulated 标识。

### 18.3 Quantitative Gate checklist

| Gate | PASS | NEEDS REVISION | FAIL |
|---|---|---|---|
| Human Surface Quality | 第 12.2 节八维均值、有效比例、template/shortcut 与 reviewer agreement 全部达 PASS | 无系统捷径，且仅落入第 12.2 节灰区 | 任一第 12.2 节 FAIL 条件成立 |
| Subgroup support | 每主 subgroup 总计≥120、Train≥60、各 eval split 达下限 | 任一为目标的 75–99% | 任一低于目标 75% |
| Quality-best diversity | 每 task ≥3 个 Expected quality-best model 各≥10%，entropy≥.45，task-only coverage 每 task≤.65且 macro≤.60 | 仅一个 task 略低但 entropy≥.35，或 macro coverage .60–.70 | 任一 task 只有 1 个稳定 quality-best，macro entropy<.30，或固定 task rule agreement>.80 |
| Efficient-winner diversity | 每 task ≥2 个 Expected efficient winner 各≥10%，entropy≥.35，modal coverage≤.75；cost/latency/fallback 成因报告齐全 | 仅一个 task 略低，或 cost-induced non-quality-best 比例超出建议区间但可解释 | 任一 task 只有 1 个稳定 efficient winner，或缺少成本、时延、fallback 成因数据 |
| Capability structure | conditional MI≥.15、capability 项解释≥60%结构方差、task residual≤20% | MI .08–.15 或 capability 方差 40–60% | MI<.08、capability 方差<40%或 subgroup winner 不变化 |
| Observable learnability | Text-task≥.08，recovery≥.50，Text-shuffle≥.10 | 三项中两项通过 | Hidden oracle 有效但 Text 与 shuffled 无差异 |
| Cue ablation | macro-F1 下降≥.07 或 regret 恶化≥15% | 下降 .03–.07 或 regret 5–15% | 无下降或反向且无法解释 |
| Distribution balance | task ±3pp、JS≤.05、difficulty SMD≤.15、pass rate 合规 | task ±5pp、JS≤.10、SMD≤.25 | 极端失衡或模型总体 pass<.05/>.95 |
| Baseline separation | Expert 对 Expected efficient winner agreement≤.65，且与 Sampled Outcome Oracle 有≥.08 pass 或≥15% cost 空间 | Expert .65–.75 或与 Sampled Outcome Oracle 间距较小 | fixed/task-only agreement>.80 或 Sampled Outcome Oracle 无区分空间 |
| Multi-seed stability | 适用的自动化 Hard Gates 7/8 development seeds 通过，关键差值方向≥6/8 | 6/8 通过或方差超建议值 | ≤5/8 通过或排名由少数 seed 决定 |

Acceptance report 必须列出每一项的分子、分母、置信区间、seed 明细和原始统计，不得使用“看起来合理”作为通过理由。

## 19. Risks and Open Questions

主要剩余风险：

- 人工 capability taxonomy 仍可能遗漏真实请求中的能力，或把相关能力拆分/合并得不合理；
- surface realization 可能产生新的词汇捷径，即使没有显式 label token；
- 数学公式及 profile 系数是人为设定，内部有效性不能消除外部有效性限制；
- full-coverage 数据回避了真实生产路由的反事实选择偏差；
- 同一团队同时设计 benchmark 和 Router，仍存在无意识地让方法适配 benchmark 的风险；
- Final Test 的 custodian、commitment 和 access ledger 能降低意外污染，但不能构成绝对防作弊机制；
- 公开 Query 引入后会增加许可、污染、答案质量和 family 去重问题。

### 19.1 已确认的负责人决定

1. **模型数**：固定为 6 个匿名开发模型，在赛题 5–10 个范围内增加交叉优势空间并控制计算量；
2. **语言分布**：采用第 5 节 task-conditioned 分布，总体中文占主导，不使用全局统一比例；
3. **Final 组合**：固定 75% Familiar 与 25% Novel capability combinations，并分别报告 Familiar、Novel、Overall；完全 unseen capability 进入独立 stress set；
4. **Final 治理**：使用不参与日常 Router 调参的 custodian、隐藏 `final_master_seed`、普通仓库中的 SHA256 commitment 与完整 access ledger；
5. **质量阈值**：主评价固定为 `0.8`；`0.7` 和 `0.9` 只做预注册 sensitivity，不参与 feature、threshold、benchmark 或主排名选择。

### 19.2 仍需在实现前解决的问题

1. **Surface authoring workflow**：人工语义框架、程序化约束组合和改写各占多少，以及 reviewer 的具体排期和责任人；
2. **Diagnostic learner**：固定使用哪一种轻量文本表示、tokenizer 与超参数，使 learnability audit 可复现且不演变为算法竞赛；
3. **Cost/latency unit**：`input_units`、`requested_output_units`、`reasoning_work` 的版本化估计规则和 synthetic 单位命名；
4. **Public source policy**：若未来引入公开 query seed，允许的许可、污染检查、答案质量与 family 归属标准。

这些问题可以在不改变已确认决策的前提下落地。具体生成系数只能在不查看候选 Router 胜负的 Benchmark Construction audit 中调整，并必须遵守第 1.1、16、18 节的版本化和预注册门槛。

## 20. Implementation Plan

本节只规定后续顺序，本轮不执行：

1. 解决第 19.2 节的实现问题，并冻结已确认的 taxonomy、模型数、语言分布、Final 组合与治理规则；
2. 定义 v0.2 schema、canonical config、manifest、diagnostic sidecar、版本提升规则和 Final access ledger；
3. 实现分 namespace RNG、四层 family、task-conditioned language 和 Query surface realization；
4. 实现无噪声 quality/cost/latency、受控采样，以及三个质量/路由概念的独立计算；
5. 实现独立 benchmark validator、Human Surface Quality 工具和 acceptance report；
6. 使用公开 development seeds 只在 Train/Validation/Development 构建候选 benchmark，运行 8-seed、perturbation 与全部结构 audit；
7. 仅根据结构 audit 修订 generator/profile/taxonomy/surface；每次语义变更提升版本并完整重跑，禁止读取 Query-aware 与 baseline 的胜负来调数据；
8. 规格再次通过 Code Review 后，才开始 baseline、ablation 和 Predictor 的 Development 实验；
9. 冻结 generator/schema/config/profile/split/validator/analysis commit，并由 custodian 预提交隐藏 Final seed 的 SHA256 commitment；
10. 冻结算法、feature、训练与路由阈值、baseline 和主/敏感性分析；custodian reveal seed，评价者验证 commitment；
11. 生成 Final Query 后、读取 outcome 前完成 family/duplicate/Human Surface audit；通过后生成 outcome 并只执行一次正式评价；
12. 归档 reveal 时间、commit、evaluator、文件 hash 和 access event；首次读取后将后续使用标为 `previously observed test`，需要新结论时创建新 seed commitment/version。

本规格通过审查之前，不开始 v0.2 generator、Predictor、Router、Calibration、Online Estimator、真实 API 或鲲鹏实现。
