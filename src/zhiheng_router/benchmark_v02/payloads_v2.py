"""Task-specific, answerable content payloads for Pilot Surface v2."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .rng import NamespaceRNG
from .surface import SemanticComposition


@dataclass(frozen=True)
class ContentPayload:
    payload_kind: str
    scenario: str
    material: str
    task_instruction: str
    output_contract: str
    structural_markers: tuple[str, ...]
    passage_count: int
    capability_evidence: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for value in (self.payload_kind, self.scenario, self.material,
                      self.task_instruction, self.output_contract):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("content payload text must be nonblank")
        if not self.structural_markers or self.passage_count < 1:
            raise ValueError("content payload needs structural markers and content")


def _numbers(stable_id: str, rng: NamespaceRNG) -> dict[str, int]:
    randomizer = rng.random("content", stable_id, stream="payload-values")
    x = randomizer.randint(8, 40)
    y = randomizer.randint(2, 7)
    return {
        "x": x,
        "y": y,
        "total": x + y,
        "weighted": 2 * x + y,
        "price_a": randomizer.randint(12, 28),
        "price_b": randomizer.randint(5, 11),
        "rate": randomizer.choice((5, 8, 10, 12)),
        "limit": randomizer.randint(3, 6),
    }


def _capability_evidence(composition: SemanticComposition, material: str,
                         instruction: str, rng: NamespaceRNG) -> dict[str, str]:
    """Select evidence from the task payload itself; never append cross-task cue text."""
    segments = [line.strip() for line in material.splitlines()
                if len(line.strip()) >= 10 and not line.strip().startswith("```")]
    segments.append(instruction.strip())
    patterns = {
        "AR": r"依赖|边界|比较|优化|方程|证明|算法|dependency|boundary|compare|optim|equation|proof|algorithm",
        "SQ": r"\d|比例|总量|面积|概率|均值|收入|equation|total|area|probability|mean|revenue",
        "DE": r"失败|错误|日志|回滚|实际|预期|fail|error|log|rollback|actual|expected",
        "DS": r"schema|表|数据|行|连接|汇总|source|row|join|aggregate",
        "AU": r"api|接口|http|命令|版本|cursor|retry|endpoint|command|version",
        "FK": r"材料|制度|手册|文档|证据|确认|policy|manual|document|evidence|confirmed",
        "MH": r"依赖|条件|之后|先|证据|步骤|链|depends|condition|after|before|evidence|step|chain",
        "LC": r"附录|末尾|开篇|章节|记录|版本|appendix|final|opening|chapter|record|version",
        "FT": r"保留|不得|事实|数字|例外|行为|不确定|retain|preserv|fact|figure|exception|behavior|uncertain",
        "MT": r"术语|译|中英文|glossary|translat|bilingual|term",
        "AM": r"未|可能|歧义|无法|不同|待确认|unclear|may|could|unresolved|confirm|different",
        "SC": r"格式|字段|列|签名|结构|小节|schema|json|csv|format|field|column|signature|section",
    }
    evidence: dict[str, str] = {}
    for capability in composition.active_capabilities:
        candidates = [segment for segment in segments
                      if re.search(patterns[capability], segment, re.IGNORECASE)]
        if not candidates:
            candidates = segments
        evidence[capability] = rng.choice(
            "content", composition.stable_id, candidates,
            stream=f"payload-evidence-{capability}")
    return evidence

def _code_payload(composition: SemanticComposition, values: dict[str, int]) -> tuple[str, str, str, set[str], int]:
    if composition.language == "en":
        return _code_payload_en(composition, values)
    slug = composition.frame.frame_id.removeprefix("frame-code-")
    x, y, total = values["x"], values["y"], values["total"]
    if slug == "scheduler":
        material = (f"任务耗时：A={x}ms、B={y}ms、C={y + 1}ms、D={x + y}ms、E={x - y}ms。A 无依赖；B、C 依赖 A；D 依赖 B 和 C；E 依赖 C。\n"
                    f"并发上限：{values['limit']}。目标是在满足依赖的前提下缩短总完成时间；任务失败后，不再启动依赖它的任务。\n"
                    "输入格式：`[{\"id\": \"A\", \"depends_on\": []}, ...]`。")
        instruction = "实现 `schedule(tasks, limit)`，返回合法的分批执行顺序，并说明环依赖和未知依赖如何报告。"
        return material, instruction, "Python function + two boundary examples", {"dependency_graph", "sample_input"}, 3
    if slug == "service-debug":
        material = f"""代码：
```python
def append_once(store, event):
    key = event.get("request_id", "")
    if key in store:
        return store[key]
    value = {{"request_id": key, "amount": event["amount"]}}
    store[key] = value
    if event.get("retry"):
        store[key + "-retry"] = value
    return value
```
失败输入：`{{"request_id":"r{x}","amount":{total},"retry":true}}`
日志：`09:31 cache miss r{x}`；`09:31 write r{x}`；`09:31 write r{x}-retry`。
预期：同一 `request_id` 只保留一条记录。"""
        instruction = "定位导致重复记录的最早状态变化，给出最小修复，并写出覆盖普通请求和 retry 请求的测试。"
        return material, instruction, "patch + regression tests", {"code_block", "log", "sample_io"}, 4
    if slug == "sql-report":
        material = f"""Schema：
`orders(order_id, customer_id, paid_amount, paid_at)`
`refunds(refund_id, order_id, refund_amount, approved_at)`
样例：orders=`(o1,c1,{total},2026-08-02)`, `(o2,c1,{x},2026-08-03)`；
refunds=`(r1,o1,{y},2026-08-05)`, `(r2,o1,1,2026-08-06)`。
口径：统计 8 月每位客户的净支付额；没有退款的订单必须保留。"""
        instruction = "编写不会因一对多退款放大订单金额的 SQL，并说明聚合与连接顺序。"
        return material, instruction, "SQL + result columns", {"schema", "sample_rows"}, 4
    if slug == "api-client":
        material = f"""接口：`GET /v2/items?cursor=<token>`
成功响应：`{{"items":[{{"id":"i{x}","state":"ready"}}],"next_cursor":"p2"}}`
末页响应：`{{"items":[]}}`
限流响应：HTTP 429，`{{"retry_after":{values['limit']}}}`
异常样例：HTTP 200 但某个 item 缺少 `id`。"""
        instruction = "实现分页客户端，区分末页、限流和 schema 错误；给出超时与最大重试策略。"
        return material, instruction, "typed Python client + error cases", {"api_contract", "json_sample"}, 4
    if slug == "refactor":
        material = f"""现有代码：
```python
def transform(rows):
    out = {{}}
    for row in rows:
        if not row.get("id"):
            continue
        out[row["id"]] = int(row.get("value", 0))
    return list(out.items())
```
行为样例：空 `id` 被忽略；重复 `id` 保留最后值；输入 `[{{"id":"a","value":"{x}"}},{{"id":"b","value":"{y}"}}]` 返回 `[('a',{x}),('b',{y})]`。"""
        instruction = "把解析、校验和聚合职责拆开，同时保持列出的外部行为，并给出特征测试。"
        return material, instruction, "refactoring patch + characterization tests", {"code_block", "sample_io"}, 3
    if slug == "repository-review":
        material = f"""模块关系：`cli.py → config.py → cache.py → service.py`。
`config.py` 在 import 时读取环境变量；`cache.py` 在 import 时创建单例；测试先 import cache 再设置环境变量。
失败顺序：`import cache` → `set ROUTER_TTL={x}` → `import service`，实际 TTL 仍为 {y * 10}。
成功顺序：先设置环境变量，再 import 任一模块。"""
        instruction = "说明配置值沿模块传播的路径，指出首次固化错误值的位置，并提出最小修改。"
        return material, instruction, "review memo + call-flow", {"module_graph", "failing_sequence"}, 4
    if slug == "data-pipeline":
        material = f"""来源 A：`id,date,value`，样例 `a{y},2026-08-03,{total}`。
来源 B：`record_id,event_date,value`，样例 `a{y},03/08/2026,NA`。
规则：日期统一为 ISO；`NA` 表示缺失而不是 0；同一 id/date 重复时保留来源 A，并记录冲突。"""
        instruction = "设计解析和标准化流程，给出统一输出 schema、错误记录以及上述两行的结果。"
        return material, instruction, "pipeline pseudocode + JSON schema", {"schema", "sample_rows", "config"}, 3
    material = f"""当前实现：
```python
def contains_pair(values, target):
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if values[i] + values[j] == target:
                return True
    return False
```
数据最多 {total * 1000} 项；值可重复且可为负数；空列表返回 False。"""
    instruction = "比较排序双指针与哈希集合两种优化，选择一种实现，并分析重复值和负数边界。"
    return material, instruction, "analysis + runnable Python + benchmark cases", {"code_block", "boundary_input"}, 3


def _code_payload_en(composition: SemanticComposition, values: dict[str, int]) -> tuple[str, str, str, set[str], int]:
    slug = composition.frame.frame_id.removeprefix("frame-code-")
    x, y, total = values["x"], values["y"], values["total"]
    payloads = {
        "scheduler": (
            f"Durations: A={x}ms, B={y}ms, C={y + 1}ms, D={x + y}ms, E={x - y}ms. A has no dependency; B and C depend on A; D depends on B and C; E depends on C.\nConcurrency limit: {values['limit']}. Minimize completion time while respecting dependencies; a failed job blocks its dependents.\nInput shape: `[{{\"id\":\"A\",\"depends_on\":[]}}, ...]`.",
            "Implement `schedule(tasks, limit)`, return valid execution batches, and define cycle and unknown-dependency errors.",
            "Python function + two boundary examples", {"dependency_graph", "sample_input"}, 3),
        "service-debug": (
            f"""Code:
```python
def append_once(store, event):
    key = event.get("request_id", "")
    if key in store:
        return store[key]
    value = {{"request_id": key, "amount": event["amount"]}}
    store[key] = value
    if event.get("retry"):
        store[key + "-retry"] = value
    return value
```
Failing input: `{{"request_id":"r{x}","amount":{total},"retry":true}}`
Log: `09:31 cache miss r{x}`; `09:31 write r{x}`; `09:31 write r{x}-retry`.
Expected: retain one row for each `request_id`.""",
            "Locate the earliest state change that creates the duplicate, give a minimal patch, and cover normal and retry requests.",
            "patch + regression tests", {"code_block", "log", "sample_io"}, 4),
        "sql-report": (
            f"Schema: `orders(order_id, customer_id, paid_amount, paid_at)` and `refunds(refund_id, order_id, refund_amount, approved_at)`.\nRows: orders `(o1,c1,{total},2026-08-02)`, `(o2,c1,{x},2026-08-03)`; refunds `(r1,o1,{y},2026-08-05)`, `(r2,o1,1,2026-08-06)`.\nDefinition: August net paid amount per customer; orders without refunds remain.",
            "Write SQL that avoids multiplying order amounts through the one-to-many refund join, and explain aggregation order.",
            "SQL + result columns", {"schema", "sample_rows"}, 4),
        "api-client": (
            f"Endpoint: `GET /v2/items?cursor=<token>`.\nSuccess: `{{\"items\":[{{\"id\":\"i{x}\",\"state\":\"ready\"}}],\"next_cursor\":\"p2\"}}`.\nLast page: `{{\"items\":[]}}`. HTTP 429: `{{\"retry_after\":{values['limit']}}}`. A malformed 200 response may omit item `id`.",
            "Implement pagination and distinguish completion, rate limiting, timeouts, and schema errors.",
            "typed Python client + error cases", {"api_contract", "json_sample"}, 4),
        "refactor": (
            f"""Current code:
```python
def transform(rows):
    out = {{}}
    for row in rows:
        if not row.get("id"):
            continue
        out[row["id"]] = int(row.get("value", 0))
    return list(out.items())
```
Observed behavior: blank IDs are ignored, duplicate IDs keep the last value, and `[{{"id":"a","value":"{x}"}},{{"id":"b","value":"{y}"}}]` returns `[('a',{x}),('b',{y})]`.""",
            "Separate parsing, validation, and aggregation while preserving every listed behavior; include characterization tests.",
            "refactoring patch + characterization tests", {"code_block", "sample_io"}, 3),
        "repository-review": (
            f"Module path: `cli.py → config.py → cache.py → service.py`. `config.py` reads the environment at import time; `cache.py` creates a singleton at import. The test imports cache, sets `ROUTER_TTL={x}`, then imports service; the TTL remains {y * 10}. Setting the variable first succeeds.",
            "Trace the value through the modules, identify where the wrong value first becomes fixed, and propose the smallest change.",
            "review memo + call-flow", {"module_graph", "failing_sequence"}, 4),
        "data-pipeline": (
            f"Source A: `id,date,value`, row `a{y},2026-08-03,{total}`.\nSource B: `record_id,event_date,value`, row `a{y},03/08/2026,NA`.\nRules: dates become ISO; `NA` is missing, not zero; for duplicate id/date, retain source A and log the conflict.",
            "Design parsing and normalization, give the output schema and error record, and show the result for both rows.",
            "pipeline pseudocode + JSON schema", {"schema", "sample_rows", "config"}, 3),
        "performance": (
            f"""Current code:
```python
def contains_pair(values, target):
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if values[i] + values[j] == target:
                return True
    return False
```
The input may contain {total * 1000} integers, duplicates, and negative values; an empty list returns False.""",
            "Compare sorting with two pointers against a hash set, implement one, and analyze duplicates and negative values.",
            "analysis + runnable Python + benchmark cases", {"code_block", "boundary_input"}, 3),
    }
    return payloads[slug]


def _math_payload(composition: SemanticComposition, values: dict[str, int]) -> tuple[str, str, str, set[str], int]:
    if composition.language == "en":
        return _math_payload_en(composition, values)
    slug = composition.frame.frame_id.removeprefix("frame-math-")
    x, y, total = values["x"], values["y"], values["total"]
    if slug == "equation-system":
        return (f"某项目采购 x 台标准设备和 y 台增强设备。设备总数为 {total}，按标准设备 2 个积分、增强设备 1 个积分计算，总积分为 {values['weighted']}。",
                "建立方程并求 x、y，代回两条条件核验。", "equations, solution, substitution check", {"numeric_conditions", "equations"}, 2)
    if slug == "probability":
        return (f"袋中有 {x} 个红球和 {y} 个蓝球，不放回连续抽取 2 个。已知第一次抽到红球。",
                "计算第二次仍为红球的条件概率，并写出分子与分母。", "exact fraction + decimal", {"numeric_conditions", "conditional_event"}, 2)
    if slug == "proof":
        return (f"数列满足 `a₁={x}`，`a_(n+1)=a_n+{y}`。命题：对所有正整数 n，`a_n={x}+(n-1)×{y}`。",
                "用数学归纳法证明命题，分别写出基础步和归纳步。", "structured proof", {"recurrence", "claim"}, 2)
    if slug == "statistics":
        sample_a = [x, x + 1, x - 1, x, x + 2]
        sample_b = [y, y + 1, y, y + 2, total * 5]
        return (f"A 组观测为 `{sample_a}`；B 组为 `{sample_b}`；最后一个 B 值来自传感器告警，是否有效尚未确认。",
                "分别计算两组中位数与均值，并说明待确认离群值如何影响比较。", "calculation table + conditional conclusion", {"numeric_conditions", "data_table"}, 2)
    if slug == "geometry":
        return (f"一个 L 形区域可看作 {total + 2}m × {y + 3}m 的大矩形，右上角挖去 {y}m × {y + 1}m 的小矩形。各边均与坐标轴平行。",
                "求剩余面积，列出大矩形和挖空部分的计算，并保留单位。", "formula + area in m²", {"numeric_conditions", "geometry_conditions"}, 2)
    if slug == "optimization":
        return (f"产品 A 每件利润 {values['price_a']}，耗时 2 小时；产品 B 每件利润 {values['price_b']}，耗时 1 小时。可用 {total} 小时，A 至少生产 1 件，A、B 均为非负整数。",
                "写出整数优化模型并找出利润最大的生产组合。", "constraints + optimum + verification", {"numeric_conditions", "optimization_constraints"}, 2)
    if slug == "word-problem":
        return (f"影院售出成人票和学生票共 {total} 张。成人票每张 {values['price_a']} 元，学生票每张 {values['price_b']} 元，总收入为 {x * values['price_a'] + y * values['price_b']} 元。",
                "求两类票各售出多少张，并验证票数与收入。", "equations + checked answer", {"numeric_conditions", "equations"}, 2)
    return (f"季度数据如下：Q1={x * 10}，Q2={total * 10}，Q3={values['weighted'] * 10}，Q4={y * 10}。附注称 Q3 中有 {x} 个单位应回溯计入 Q2。",
            "先按原表计算 Q2→Q3 增长率，再按附注调整后重新计算，并比较差异。", "two calculations + interpretation", {"numeric_conditions", "data_table", "revision_rule"}, 2)


def _math_payload_en(composition: SemanticComposition, values: dict[str, int]) -> tuple[str, str, str, set[str], int]:
    slug = composition.frame.frame_id.removeprefix("frame-math-")
    x, y, total = values["x"], values["y"], values["total"]
    payloads = {
        "equation-system": (f"A project buys x standard devices and y enhanced devices. The total is {total}. A standard device counts for 2 points and an enhanced device for 1, giving {values['weighted']} points.", "Form and solve the equations, then substitute the answer into both conditions.", "equations, solution, substitution check", {"numeric_conditions", "equations"}, 2),
        "probability": (f"A bag contains {x} red balls and {y} blue balls. 2 balls are drawn without replacement, and the first is known to be red.", "Find the conditional probability that the second ball is red, showing numerator and denominator.", "exact fraction + decimal", {"numeric_conditions", "conditional_event"}, 2),
        "proof": (f"A sequence satisfies `a₁={x}` and `a_(n+1)=a_n+{y}`. Claim: for every positive integer n, `a_n={x}+(n-1)×{y}`.", "Prove the claim by induction, separating the base and induction steps.", "structured proof", {"recurrence", "claim"}, 2),
        "statistics": (f"Sample A is `{[x, x + 1, x - 1, x, x + 2]}`. Sample B is `{[y, y + 1, y, y + 2, total * 5]}`. The final B value came from a sensor alert and is not yet validated.", "Calculate the median and mean of both samples and explain how the uncertain outlier changes the comparison.", "calculation table + conditional conclusion", {"numeric_conditions", "data_table"}, 2),
        "geometry": (f"An L-shaped region is a {total + 2} m by {y + 3} m rectangle with a {y} m by {y + 1} m rectangle removed from its upper-right corner. All edges are axis-aligned.", "Find the remaining area and show the whole and removed areas with units.", "formula + area in m²", {"numeric_conditions", "geometry_conditions"}, 2),
        "optimization": (f"Product A earns {values['price_a']} per unit and uses 2 hours; product B earns {values['price_b']} and uses 1 hour. There are {total} hours. At least one A is required, and quantities are nonnegative integers.", "Write the integer optimization model and find the profit-maximizing quantities.", "constraints + optimum + verification", {"numeric_conditions", "optimization_constraints"}, 2),
        "word-problem": (f"A cinema sold {total} adult and student tickets. Adult tickets cost {values['price_a']}; student tickets cost {values['price_b']}. Total revenue was {x * values['price_a'] + y * values['price_b']}.", "Find the count of each ticket type and verify both ticket count and revenue.", "equations + checked answer", {"numeric_conditions", "equations"}, 2),
        "chart-analysis": (f"Quarterly values are Q1={x * 10}, Q2={total * 10}, Q3={values['weighted'] * 10}, Q4={y * 10}. A note says {x} units in Q3 should be reassigned to Q2.", "Compute Q2-to-Q3 growth before and after the revision, then explain the difference.", "two calculations + interpretation", {"numeric_conditions", "data_table", "revision_rule"}, 2),
    }
    return payloads[slug]


def _qa_payload(composition: SemanticComposition, values: dict[str, int]) -> tuple[str, str, str, set[str], int]:
    if composition.language == "en":
        return _qa_payload_en(composition, values)
    slug = composition.frame.frame_id.removeprefix("frame-qa-")
    x, y = values["x"], values["y"]
    materials = {
        "technical-docs": (
            f"[材料A] v2.{y} 文档规定，worker 收到 `retry_after` 后等待 {x} 秒再重试。\n"
            f"[材料B] v2.{y + 1} 变更说明只取消了登录接口的自动重试，批处理接口保持原规则。\n"
            "[材料C] 本次日志来自批处理接口，返回 HTTP 429 和 `retry_after`。",
            "本次 worker 是否应自动重试？说明每段材料在结论中的作用。"),
        "evidence-chain": (
            "[材料A] 只有通过安全审批的版本才能进入部署队列。\n"
            f"[材料B] 记录显示版本 R{x} 已通过安全审批并于周二进入队列。\n"
            f"[材料C] 队列条目进入后，需要窗口负责人确认；R{x} 尚无确认记录。",
            f"R{x} 当前可以正式部署吗？按证据链说明已经满足和仍未满足的条件。"),
        "ambiguous-policy": (
            "[材料A] 制度规定：最近发生两次严重故障的服务需要复审。\n"
            f"[材料B] 服务 S{x} 在过去 7 天有 {y - 1} 次、过去 30 天有 {y + 1} 次严重故障。\n"
            "[材料C] 本页没有定义“最近”的时间范围。",
            f"能否确定服务 S{x} 必须复审？分别说明确定事实、可能解释和所需澄清。"),
        "source-comparison": (
            f"[材料A] 运营表记录 {x} 次成功和 {y} 次失败，统计窗口为自然日。\n"
            f"[材料B] 技术表记录 {x + 1} 次成功和 {y} 次失败，统计窗口为最近 24 小时。\n"
            "[材料C] 其中一条成功记录发生在昨日 00:05。",
            "两份成功数为何可能不同？哪些结论可以直接比较，哪些不能？"),
        "troubleshooting": (
            f"[材料A] 09:{x:02d} 配置把 `mode` 从 safe 改为 fast。\n"
            f"[材料B] 09:{x + 2:02d} 第一次失败日志为 `schema mismatch: result.items`。\n"
            f"[材料C] 09:{x + 4:02d} 回滚 mode 后错误仍存在；09:{x + 6:02d} 恢复旧 schema 后请求成功。",
            "最早可证实的故障点是什么？配置变化是否为根因？给出下一项验证。"),
        "data-explanation": (
            f"[材料A] 报表总量为 {x + y}，其中 active={x}、suspended={y}。\n"
            "[材料B] 仪表盘只显示 active，标题却写“全部账户”。\n"
            "[材料C] 数据字典把 suspended 定义为仍保留但不可登录的账户。",
            "解释仪表盘数字与报表总量的差异，并指出标题是否准确。"),
        "long-policy": (
            "[材料A·定义] 有效申请必须在截止日前提交。\n"
            f"[材料B·例外] 系统故障超过两小时，可在恢复后一个工作日内补交。\n"
            f"[材料C·记录] 截止日当天系统中断 {y + 2} 小时，申请 P{x} 在次日补交。\n"
            f"[材料D·附录] 该例外不适用于主动撤回后重新提交的申请；P{x} 没有撤回记录。",
            f"申请 P{x} 是否满足时间规则？综合定义、例外、记录和附录。"),
        "bilingual-terms": (
            "[材料A] 中文制度把 `service window` 译为“服务窗口”，指允许停机的时段。\n"
            "[Evidence B] English log says the job ran outside the service window.\n"
            f"[材料C] v1.{x}.{y} 说明曾把同一词误译为“客服窗口”，本次事件使用新版术语。",
            "用中文解释日志结论，并说明应采用哪个术语及理由。"),
    }
    material, instruction = materials[slug]
    return material, instruction, "answer + cited material labels", {"evidence_passages"}, material.count("[")


def _qa_payload_en(composition: SemanticComposition, values: dict[str, int]) -> tuple[str, str, str, set[str], int]:
    slug = composition.frame.frame_id.removeprefix("frame-qa-")
    x, y = values["x"], values["y"]
    materials = {
        "technical-docs": (f"[Evidence A] In v2.{y} a worker waits {x} seconds after `retry_after`.\n[Evidence B] The v2.{y + 1} change removes automatic retry only for login; batch endpoints retain it.\n[Evidence C] This batch request returned HTTP 429 with `retry_after`.", "Should this worker retry automatically? Explain how each passage supports the answer."),
        "evidence-chain": (f"[Evidence A] Only a security-approved release may enter the deployment queue.\n[Evidence B] R{x} passed review and entered the queue Tuesday.\n[Evidence C] A queued release also needs window-owner confirmation; R{x} has none.", f"Can R{x} be deployed now? Identify satisfied and unsatisfied conditions."),
        "ambiguous-policy": (f"[Evidence A] A service with two serious incidents in the recent period requires review.\n[Evidence B] Service S{x} had {y - 1} incidents in 7 days and {y + 1} in 30 days.\n[Evidence C] This policy page does not define “recent.”", "Can the review requirement be determined? Separate facts, interpretations, and the required clarification."),
        "source-comparison": (f"[Evidence A] Operations lists {x} successes and {y} failures by calendar day.\n[Evidence B] Engineering lists {x + 1} successes and {y} failures over the latest 24 hours.\n[Evidence C] One success occurred yesterday at 00:05.", "Why may the success totals differ, and which claims are directly comparable?"),
        "troubleshooting": (f"[Evidence A] At 09:{x:02d} `mode` changed from safe to fast.\n[Evidence B] At 09:{x + 2:02d} the first failure was `schema mismatch: result.items`.\n[Evidence C] Rolling back mode at 09:{x + 4:02d} did not help; restoring the old schema at 09:{x + 6:02d} did.", "What is the earliest confirmed failure point? Is mode the root cause, and what should be checked next?"),
        "data-explanation": (f"[Evidence A] The report totals {x + y}: active={x}, suspended={y}.\n[Evidence B] The dashboard displays only active accounts but is titled “All accounts.”\n[Evidence C] The dictionary says suspended accounts still exist but cannot sign in.", "Explain the difference and assess whether the dashboard title is accurate."),
        "long-policy": (f"[Definition A] A valid application is submitted before the deadline.\n[Exception B] After an outage over two hours, submission may occur within one business day of recovery.\n[Record C] The system failed for {y + 2} hours on deadline day; P{x} was submitted the next day.\n[Appendix D] The exception excludes withdrawn resubmissions; P{x} was never withdrawn.", f"Does P{x} meet the timing rule? Integrate the definition, exception, record, and appendix."),
        "bilingual-terms": (f"[Evidence A] The Chinese policy translates `service window` as 服务窗口, meaning an allowed downtime period.\n[Evidence B] The log says the job ran outside the service window.\n[Evidence C] Edition v1.{x}.{y} used 客服窗口 incorrectly; this incident uses the new glossary.", "Explain the log conclusion and identify the correct Chinese term with evidence."),
    }
    material, instruction = materials[slug]
    return material, instruction, "answer + cited material labels", {"evidence_passages"}, material.count("[")


def _summary_payload(composition: SemanticComposition, values: dict[str, int]) -> tuple[str, str, str, set[str], int]:
    if composition.language == "en":
        return _summary_payload_en(composition, values)
    slug = composition.frame.frame_id.removeprefix("frame-summary-")
    x, y = values["x"], values["y"]
    common = {
        "meeting": [
            f"[记录1] 团队决定周三先发布 R{x} 的只读查询，写入功能延后 {y} 天。",
            "[记录2] 运维认为缓存指标仍缺失，发布前需补上命中率告警。",
            "[记录3] 产品负责人同意只读范围，但移动端兼容性由李明在周二前确认。",
            "[记录4] 未决事项：若兼容性测试失败，是否取消周三发布尚无决定。",
        ],
        "research": [
            f"[段落1] 试验组包含 {x + y} 个样本，对照组包含 {x + y + 2} 个样本。",
            f"[段落2] 主要指标提高 {y}.{x}%，但 95% 置信区间跨过零。",
            "[段落3] 作者认为结果值得继续验证，没有声称效果已经确定。",
            "[段落4] 局限包括样本量小、单中心和两例失访。",
        ],
        "incident": [
            f"[事件1] 09:{x:02d} 发布配置，两分钟后首次出现写入超时。",
            f"[事件2] 09:{x + 10:02d} 流量切回旧集群，错误率从 {x + y}% 降至 {y}%。",
            "[事件3] 10:05 修复连接池上限后恢复全部流量，错误率回到基线。",
            "[事件4] 已证实连接池耗尽；配置变更是否直接触发耗尽仍待验证。",
        ],
        "metrics": [
            f"[表1] 8 月：收入 {x * 100} 万元，退款 {y * 10} 万元，周期为 8 月 1–31 日。",
            f"[表2] 活跃用户 {x * 1000}，周期为 7 月 25 日–8 月 24 日。",
            "[说明3] 收入口径不含税，退款表按审批日而不是支付日归属月份。",
            "[说明4] 管理层需要同时看到可比结论和不可直接比较之处。",
        ],
        "conflicting-reports": [
            f"[报告A] 项目 R{x} 完成 {70 + y}%，延期原因是接口变更。",
            f"[报告B] 项目 R{x} 完成 {60 + y}%，延期原因是测试环境晚到。",
            f"[附件C] R{x} 的接口变更影响了 {y} 个模块；测试环境比计划晚 {y + 1} 天开放。",
            "[备注D] 两份报告采用的完成率计算口径没有记录。",
        ],
        "technical-guide": [
            f"[章节1] 构建 R{x} 安装前要求 Python 3.11，并创建独立虚拟环境。",
            "[章节2] 配置文件必须先通过 `router validate`，再启动服务。",
            "[章节3] 若迁移失败，保留数据库并用 `router rollback --config <path>` 回滚。",
            "[附录4] v2.0 不支持从 v0.x 直接升级；必须先迁移到 v1.5。",
        ],
        "bilingual-report": [
            "[中文1] 数据导入已完成，验证工作计划周四结束。",
            f"[English 2] Data import is complete; validation found {y} unresolved duplicate IDs.",
            f"[中文3] 当前版本没有记录这 {y} 个重复 ID，负责人尚未确认处理方式。",
            "[English 4] Deployment remains scheduled for Friday, subject to validation approval.",
        ],
        "executive": [
            f"[结论1] 试点覆盖 {x} 个团队，其中 {x - 1} 个按期完成。",
            f"[风险2] 剩余团队缺少接口权限，最早在 {y} 个工作日后解决。",
            "[决定3] 管理层批准扩大试点，但要求先完成权限整改。",
            "[下一步4] 王宁负责周四前提交整改清单；预算增补尚未批准。",
        ],
    }
    material = "\n".join(common[slug])
    instruction = composition.goal
    return material, instruction, "four-section brief or source-aligned table", {"source_material"}, len(common[slug])


def _summary_payload_en(composition: SemanticComposition, values: dict[str, int]) -> tuple[str, str, str, set[str], int]:
    slug = composition.frame.frame_id.removeprefix("frame-summary-")
    x, y = values["x"], values["y"]
    common = {
        "meeting": [f"[Note 1] The team approved read-only release R{x} Wednesday and delayed writes {y} days.", "[Note 2] Operations requires a cache-hit alert before release.", "[Note 3] Mobile compatibility is assigned to Li Ming for Tuesday.", "[Note 4] Whether a failed compatibility test cancels release is unresolved."],
        "research": [f"[Paragraph 1] The treatment group had {x + y} samples and control had {x + y + 2}.", f"[Paragraph 2] The primary metric rose {y}.{x}%, but its 95% interval crossed zero.", "[Paragraph 3] The authors call for validation and do not claim a confirmed effect.", "[Paragraph 4] Limits include small size, one center, and two losses to follow-up."],
        "incident": [f"[Event 1] Configuration was released at 09:{x:02d}; write timeouts began two minutes later.", f"[Event 2] Traffic returned to the old cluster at 09:{x + 10:02d} and errors fell from {x + y}% to {y}%.", "[Event 3] Full traffic resumed after a pool-limit fix at 10:05.", "[Event 4] Pool exhaustion is confirmed; whether the configuration triggered it is unresolved."],
        "metrics": [f"[Table 1] August revenue was {x * 100}k and refunds {y * 10}k for August 1–31.", f"[Table 2] Active users were {x * 1000} for July 25–August 24.", "[Note 3] Revenue excludes tax; refunds use approval date rather than payment date.", "[Note 4] Management needs comparable findings and explicit non-comparability."],
        "conflicting-reports": [f"[Report A] Project R{x} is {70 + y}% complete; API changes caused delay.", f"[Report B] Project R{x} is {60 + y}% complete; a late test environment caused delay.", f"[Appendix C] R{x} API changes affected {y} modules and the environment opened {y + 1} days late.", "[Note D] Neither report records its completion-rate definition."],
        "technical-guide": [f"[Chapter 1] Build R{x} requires Python 3.11 and an isolated environment.", "[Chapter 2] Run `router validate` before service startup.", "[Chapter 3] On migration failure, retain the database and run `router rollback --config <path>`.", "[Appendix 4] v2.0 cannot upgrade directly from v0.x; migrate to v1.5 first."],
        "bilingual-report": ["[Chinese 1] 数据导入已完成，验证计划周四结束。", f"[English 2] Validation found {y} unresolved duplicate IDs.", f"[Chinese 3] 中文版本没有记录这 {y} 个 ID，负责人尚未确认处理方式。", "[English 4] Friday deployment remains conditional on validation approval."],
        "executive": [f"[Finding 1] The pilot covered {x} teams and {x - 1} finished on time.", f"[Risk 2] The remaining team lacks API permission; resolution takes at least {y} business days.", "[Decision 3] Expansion is approved after permission remediation.", "[Next step 4] Wang Ning owns Thursday's remediation list; extra budget is unapproved."],
    }
    material = "\n".join(common[slug])
    return material, composition.goal, "four-section brief or source-aligned table", {"source_material"}, len(common[slug])


def _translation_payload(composition: SemanticComposition, values: dict[str, int]) -> tuple[str, str, str, set[str], int]:
    slug = composition.frame.frame_id.removeprefix("frame-translation-")
    x, y = values["x"], values["y"]
    september_day = 8 + (x % 21)
    zh_sources = {
        "technical": f"在升级到 v2.{y} 前，请导出 `router.yaml`。如果 health check 连续 {x} 秒失败，执行 `router rollback --keep-data`；不要删除现有索引。",
        "contract": f"批次 B{x} 的供应商应在周五 {12 + values['limit']}:00 前交付测试报告；但是，如果“其验收数据”在周四 {y}:00 后才提供，期限顺延至下周一。“其”所指主体需由双方确认。",
        "notice": f"维护窗口定于 9 月 {september_day} 日 22:00–23:30。期间只读查询可用，写入请求将排队；排队超过 {y} 分钟的写入会失败，且不会自动重放。",
        "domain-report": f"本报告中的“节点”指变电站监测点，附录中的 node 指计算节点。告警阈值暂定为 {x}.{y} kV，该数值仍待现场校准。",
        "ui-localization": f"`Retry` 按钮重新发送当前请求；`Discard` 放弃未保存更改。错误消息：`Request {{request_id}} timed out after {x} s.`",
        "incident": f"09:{x:02d} 网关 G{x} 开始返回 502；09:{x + 4:02d} 值班员回滚配置；09:{x + 6:02d} 首次请求成功。根因尚未确认，连接池耗尽只是当前假设。",
        "ambiguous-reference": f"网关 G{x} 通知 worker W{y} 在它重启后刷新令牌。由于“它”可能指网关或 worker，执行顺序目前无法唯一确定。",
        "glossary": f"当前术语表规定：rollback＝回滚，service window＝服务窗口。历史译文中有 {y} 处“客服窗口”均指 service window，需要统一修订。",
    }
    en_sources = {
        "technical": f"Before upgrading to v2.{y}, export `router.yaml`. If the health check fails for {x} consecutive seconds, run `router rollback --keep-data`; do not delete the existing index.",
        "contract": f"For batch B{x}, the supplier shall deliver the test report by {12 + values['limit']}:00 Friday. If its acceptance data arrives after {y}:00 Thursday, the deadline moves to Monday. The referent of “its” requires confirmation by both parties.",
        "notice": f"Maintenance is scheduled for September {september_day}, 22:00–23:30. Read-only queries remain available; writes queued for more than {y} minutes fail and are not replayed automatically.",
        "domain-report": f"In the main report, “node” means a substation monitoring point; in the appendix it means a compute node. The {x}.{y} kV threshold is provisional pending field calibration.",
        "ui-localization": f"`Retry` resends the current request; `Discard` removes unsaved changes. Error: `Request {{request_id}} timed out after {x} s.`",
        "incident": f"At 09:{x:02d} gateway G{x} began returning 502. At 09:{x + 4:02d} the operator rolled back the configuration; the first success arrived at 09:{x + 6:02d}. Root cause remains unconfirmed; pool exhaustion is only the current hypothesis.",
        "ambiguous-reference": f"Gateway G{x} told worker W{y} to refresh the token after it restarted. “It” may refer to either component, so the execution order is not uniquely determined.",
        "glossary": f"The current glossary maps 回滚 to rollback and 服务窗口 to service window. {y} older entries use customer-service window for the latter and require revision.",
    }
    if composition.translation_direction == "zh_to_en":
        material = f"[源文]\n{zh_sources[slug]}"
        instruction = f"{composition.frame.goal_zh} 目标语言为英文。"
    elif composition.translation_direction == "en_to_zh":
        material = f"[Source text]\n{en_sources[slug]}"
        instruction = f"{composition.frame.goal_zh} 目标语言为中文。"
    else:
        material = f"[中文版本]\n{zh_sources[slug]}\n[English version]\n{en_sources[slug]}"
        instruction = f"{composition.frame.goal_zh} 同时交付修订后的中英文。"
    return material, instruction, "translated text + concise translator notes", {"source_text"}, 2 if composition.translation_direction == "bilingual_revision" else 1


def generate_content_payload(composition: SemanticComposition,
                             rng: NamespaceRNG) -> ContentPayload:
    values = _numbers(composition.stable_id, rng)
    if composition.frame.task == "code":
        material, instruction, output, markers, passages = _code_payload(composition, values)
    elif composition.frame.task == "math":
        material, instruction, output, markers, passages = _math_payload(composition, values)
        markers.add("numeric_conditions")
    elif composition.frame.task == "qa":
        material, instruction, output, markers, passages = _qa_payload(composition, values)
    elif composition.frame.task == "summary":
        material, instruction, output, markers, passages = _summary_payload(composition, values)
    else:
        material, instruction, output, markers, passages = _translation_payload(composition, values)
    output = composition.output_format
    if (composition.frame.task == "translation"
            and composition.translation_direction != "bilingual_revision"
            and "bilingual" in output):
        output = (
            "clause-aligned translation"
            if "contract" in composition.frame.frame_id
            else "formatted notice"
        )
    evidence = _capability_evidence(composition, material, instruction, rng)
    markers.update(f"capability_structure:{capability}" for capability in evidence)
    scenario = (composition.frame.goal_en if composition.language == "en"
                else composition.frame.goal_zh)
    payload = ContentPayload(
        payload_kind=composition.frame.frame_id.removeprefix("frame-"),
        scenario=scenario,
        material=material,
        task_instruction=instruction,
        output_contract=output,
        structural_markers=tuple(sorted(markers)),
        passage_count=passages,
        capability_evidence=tuple(sorted(evidence.items())),
    )
    validate_content_payload(composition, payload)
    return payload


def validate_content_payload(composition: SemanticComposition, payload: ContentPayload) -> None:
    markers = set(payload.structural_markers)
    task = composition.frame.task
    if set(dict(payload.capability_evidence)) != set(composition.active_capabilities):
        raise ValueError("payload evidence must cover every active capability")
    visible_text = "\n".join((payload.material, payload.task_instruction, payload.output_contract))
    if any(value not in visible_text for value in dict(payload.capability_evidence).values()):
        raise ValueError("capability evidence must be physically present in the payload")
    if task == "code":
        if not markers.intersection({
            "code_block", "log", "schema", "api_contract", "module_graph",
            "dependency_graph", "sample_rows", "config",
        }):
            raise ValueError("code payload lacks analyzable code/log/schema/input material")
        if composition.frame.frame_id.endswith("service-debug") and not {"code_block", "log", "sample_io"} <= markers:
            raise ValueError("debug payload requires code, log, and failing input/output")
    elif task == "math":
        if not {"numeric_conditions"} <= markers or len([c for c in payload.material if c.isdigit()]) < 3:
            raise ValueError("math payload lacks usable numerical or symbolic conditions")
    elif task == "qa":
        if "evidence_passages" not in markers or not 2 <= payload.passage_count <= 5:
            raise ValueError("QA payload requires 2-5 evidence passages")
    elif task == "summary":
        if "source_material" not in markers or not 3 <= payload.passage_count <= 6:
            raise ValueError("summary payload requires 3-6 source passages")
    elif task == "translation":
        if "source_text" not in markers or payload.passage_count not in (1, 2):
            raise ValueError("translation payload requires distinct source text")
        if payload.material.strip() == payload.scenario.strip() or len(payload.material) < 50:
            raise ValueError("translation source text is missing or only repeats the scenario")
