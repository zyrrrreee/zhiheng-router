# Router MVP 使用与实现约定

本文件记录已确认并实现的第一阶段范围。原始命题 PDF、AGENTS.md 和《项目介绍与初步安排》保持不变。

## 范围与入口

- 安装：仓库根目录执行 `python -m pip install -e ".[dev]"`，推荐 Python 3.12 虚拟环境。
- 数据生成：`python experiments/make_development_data.py`。
- 实验：`python experiments/run.py`。
- 测试：`python -m pytest -q`。
- 交互演示：`python demo.py`；也支持 `python demo.py --query "请解释缓存"`。
- 生成器和实验支持 `--config`；配置中的数据与输出路径相对于仓库根目录。
- Demo 支持 `--artifact`；默认读取 `outputs/mvp/router.joblib`，不读取当前配置重新拟合。

数据是 synthetic，所有质量、调用费用、模型时延的评价值均标识为 simulated。仅本地 Router 耗时为 measured，不是模型服务或鲲鹏性能。

## 数据和训练契约

默认生成 600 个 Query、60 个独立模板族、5 个中性开发模型，共 3,000 条观测。每条 Query 对每个模型恰有一次模拟观测。

字段为 `call_id`、`query_id`、`group_id`、`query`、`model_id`、`quality_score`、`cost`、`latency_ms`、`task_type`。数据 manifest 保存来源、版本、seed、单位、评分定义、生成配置和 SHA256。读取器验证文件摘要与全模型覆盖；原始行保留连续质量分。

模板族整体划分，不按模型观测行随机拆分。默认 36 / 12 / 12 个组，即 360 / 120 / 120 个 Query，分别用于 train / validation / test。分组不保证任务比例严格相同，报告保存各分区任务计数及 Query/Group ID。

TF-IDF 仅在唯一训练 Query 上拟合。长度与近似片段数先做 log1p，结构特征的尺度只由训练 Query 计算。字符 TF-IDF 与结构特征保持稀疏；同 Query 的特征供所有模型共用。空白输入报错，一字输入允许没有文本 n-gram。

原始 `task_type` 仅参与分层评价与数据一致性检查，不输入 Predictor。`task_type_hint(query)` 是 Rule-based Baseline 使用的独立确定性函数；Predictor 不使用该 hint。

每个模型分别训练二分类 LR；不做 model selection 多分类，不要求各模型的 pass probability 加总为一。训练至少满足配置中的样本数且同时有正负类；单类、无数据和不收敛明确失败，不自动换 seed 或伪造概率。

成本、时延、平均质量、全局达标率及 Baseline 选择仅来自训练数据。超参数和阈值在读取测试结果前已固定；validation 用于诊断，本版没有自动调参或额外校准模型。

## 两个阈值和 Router

- `quality_score_threshold`：`quality_score >= threshold` 定义 pass label，属于训练配置并随产物保存；修改它需要重训。
- `router_probability_threshold`：筛选预测达标概率，属于 RoutingPolicy。
- `cost_tolerance_abs`：相对全部 qualified 模型的最低估计成本计算差值；差值不超过容差者进入时延比较。默认 0.0。零容差使用严格的浮点成本相等，不引入隐式 epsilon；正容差将已验证浮点值的十进制文本表示转换为 `Decimal` 后比较，避免 `1.1 - 1.0` 的二进制舍入误差错误排除边界值。正容差是项目扩展，不是命题的固定要求。

`route(candidates, estimates, policy)` 是纯函数，不训练、不调用模型、不读取历史或测试标签。

正常路径：enabled → 验证估计 → probability 达标 → 最低成本及绝对容差带 → latency → cost → model_id。

Fallback：无 qualified 模型时，按 `-pass_probability, cost, latency, model_id` 选择，返回 `FALLBACK_MAX_PASS_PROBABILITY`，明确表示预测质量门槛未满足。

无候选、全 disabled、未知/重复模型 ID、缺少估计、非有限值或非法概率均报错。`enabled` 只是静态配置，不表示服务健康。各模型成本/时延在第一版均是固定训练均值。

## Baseline 与评价

| 策略 | 只使用训练阶段允许的信息 |
|---|---|
| Always Strongest | 训练集平均质量最高的固定模型，平局按 ID |
| Always Lowest Cost | 训练集平均成本最低的固定模型，平局按 ID |
| Expert Rule-based | 配置中预先声明的 Query-only task hint → model 映射 |
| Global Historical Router | 用训练集模型达标率替代 Query-aware 概率，复用同一 Router Policy，固定全局选择 |

Expert Rule-based Baseline 使用较强的人工先验：当前 synthetic generator 的模型任务专长与预设 hint → model 映射具有明显结构一致性。这不是 test label leakage，因为规则只读取 Query 并且未查询测试 outcome；但当前结果既不能证明规则方法一般优于 Query-aware routing，也没有证明 Query-aware Router 优于规则方法。v0.2 将通过更丰富的 within-task heterogeneity、learned task-rule baseline、hint ablation 和新的冻结测试集进一步评价，本版不据此调整规则、数据或结果。

前三种策略没有概率门槛，fallback 相关指标为 `null`（不适用），不是 0%。Global Historical Router 与 Query-aware Router 报告真实的 fallback 标记。某个 normal/fallback 子集为空时，其达标率也为 `null`，避免把无样本写成零。

评价函数只将 Query 文本传给策略；选择完成后才查找 Query × selected model 的模拟观测。缺少观测时报错，不拿预测值补充。所有请求（包括 fallback）进入总体质量、成本、时延指标；另报告正常/fallback 子集、任务分层及所选模型的概率诊断。

log loss、Brier score 和概率分箱同时报告 validation / test、各模型和汇总。分箱包含每箱样本数，空箱为空值。汇总的 Query × Model 观测不是相互独立的 Query 样本，不能据此夸大统计确定性。

本版不计算 Oracle。未来若加入，只能作为 full-coverage evaluation 的 hindsight 参考，不进入可部署策略。

## 产物和复现

- `data/development.jsonl` 与 `.meta.json`：提交的小型可重建开发数据；相同生成配置和 seed 产生相同字节。
- `.gitattributes` 固定数据文件为 LF，避免 Windows Git 自动换行破坏 manifest 的 SHA256 校验。
- `outputs/mvp/router.joblib`：共享提取器、每模型 LR、训练统计、候选池、Policy、manifest、配置和环境版本。只能加载可信的本地文件。
- `outputs/mvp/report.json`：完整划分、训练正负样本、预测指标、分箱、五种策略、逐 Query 决策、任务分层及环境信息。
- `outputs/` 被现有 `.gitignore` 忽略；从干净源代码运行生成和训练入口即可重建，不依赖预存模型。

数值库线程数通过实验配置固定，目的是控制运行条件，不是多核优化。Router 计时包含特征提取、预测、估计组装和决策，排除启动、训练、产物加载、结果查表与实际模型调用；有预热，报告均值、P50、P95 和串行决策时间对应的 QPS。耗时随机器和负载变化，不能要求字节级复现。

## 与早期文档的差异及限制

早期文档的 `Predicted Quality` 示例是质量分含义。本版输出的是固定质量分阈值下的达标概率，不能混用。第一阶段现已包含轻量质量 Predictor，完整在线性能感知仍未实现。

模拟世界包含模型任务专长、同任务难度变化、非线性难度影响、模板/Query 共同扰动和每模型噪声；费用独立于采样得到的质量，时延存在随机波动。它仍是人为构造的有限世界，不能证明真实业务增益，也没有解决生产日志中的反事实选择偏差。

本次固定配置的模拟实验中，Query-aware Router 未优于 Rule-based Baseline，且部分高概率样本存在过度自信；保持这些结果，不使用测试集回调参数。

后续 Online Performance Estimator 在决策前调整候选和近期时延估计；调用与 Telemetry 在决策后产生实际观测；Quality Evaluator 与 Critic 服务于后续反馈。当前只保留数据和函数边界，不实现这些系统。

原始命题中的鲲鹏 CPU、NEON、多核加速仍是最终交付要求。本地 Python/稀疏矩阵实现只为后续迁移提供基础，不能替代鲲鹏部署与优化验收。
