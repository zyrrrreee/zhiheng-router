"""Human-authored semantic frames for the Phase 1A Pilot Surface Set."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import CAPABILITY_IDS, TASK_IDS


@dataclass(frozen=True)
class SemanticFrame:
    frame_id: str
    task: str
    scenario_zh: str
    scenario_en: str
    goal_zh: str
    goal_en: str
    required_inputs: tuple[str, ...]
    possible_constraints: tuple[str, ...]
    compatible_capabilities: tuple[str, ...]
    incompatible_pairs: tuple[tuple[str, str], ...]
    output_formats: tuple[str, ...]
    difficulty_knobs: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.task not in TASK_IDS or not self.frame_id.startswith(f"frame-{self.task}-"):
            raise ValueError(f"invalid semantic frame identity: {self.frame_id}")
        if any(not value.strip() for value in (
            self.scenario_zh, self.scenario_en, self.goal_zh, self.goal_en,
        )):
            raise ValueError(f"incomplete semantic frame text: {self.frame_id}")
        if (len(self.compatible_capabilities) < 2
                or len(set(self.compatible_capabilities)) != len(self.compatible_capabilities)
                or not set(self.compatible_capabilities) <= set(CAPABILITY_IDS)):
            raise ValueError(f"invalid capabilities for {self.frame_id}")
        if any(set(pair) - set(self.compatible_capabilities) or len(set(pair)) != 2
               for pair in self.incompatible_pairs):
            raise ValueError(f"invalid incompatible pair for {self.frame_id}")
        for values in (self.required_inputs, self.possible_constraints, self.output_formats,
                       self.difficulty_knobs):
            if not values or any(not value.strip() for value in values):
                raise ValueError(f"incomplete semantic frame: {self.frame_id}")


def _frame(task: str, slug: str, scenario_zh: str, scenario_en: str, goal_zh: str,
           goal_en: str, capabilities: tuple[str, ...], formats: tuple[str, ...],
           *, inputs: tuple[str, ...] = ("scenario material", "user objective"),
           constraints: tuple[str, ...] = ("scope", "verification", "output shape"),
           knobs: tuple[str, ...] = ("material length", "dependency count", "constraint count"),
           incompatible: tuple[tuple[str, str], ...] = ()) -> SemanticFrame:
    return SemanticFrame(
        frame_id=f"frame-{task}-{slug}", task=task, scenario_zh=scenario_zh,
        scenario_en=scenario_en, goal_zh=goal_zh, goal_en=goal_en,
        required_inputs=inputs, possible_constraints=constraints,
        compatible_capabilities=capabilities, incompatible_pairs=incompatible,
        output_formats=formats, difficulty_knobs=knobs,
    )


SEMANTIC_FRAMES = (
    # code
    _frame("code", "scheduler", "{org}需要安排{n}个存在依赖关系的任务，并限制同时运行数量。",
           "{org} must schedule {n} dependent jobs with a concurrency limit.",
           "设计可执行的调度函数，解释依赖处理和异常输入。",
           "Design an executable scheduling function and explain dependency handling and invalid input.",
           ("AR", "MH", "SC", "AM"), ("Python function", "JSON contract")),
    _frame("code", "service-debug", "{system}在处理第{k}批请求时出现重复写入，日志给出两个不同的错误位置。",
           "{system} duplicates writes in request batch {k}, while two log sections point to different locations.",
           "定位最早的偏差来源并给出最小修复和回归检查。",
           "Locate the earliest divergence and provide a minimal fix with a regression check.",
           ("DE", "AU", "LC", "SC"), ("patch", "diagnostic checklist")),
    _frame("code", "sql-report", "{org}有订单、退款和客户三张表，需要核对{n}条记录的月度汇总。",
           "{org} needs a monthly reconciliation across orders, refunds, and customers for {n} records.",
           "编写查询并说明连接、去重和空值如何处理。",
           "Write the query and explain joins, deduplication, and null handling.",
           ("DS", "SQ", "SC", "AM"), ("SQL", "result schema")),
    _frame("code", "api-client", "{system}要接入一个分页接口；接口可能限流，并在部分页面返回缺失字段。",
           "{system} integrates a paginated API that may rate-limit requests and omit fields on some pages.",
           "实现稳健的客户端调用流程，覆盖重试、校验和失败报告。",
           "Implement a robust client flow covering retries, validation, and failure reporting.",
           ("AU", "DE", "SC", "AR"), ("Python module", "typed interface")),
    _frame("code", "refactor", "一个{n}行的数据转换模块混合了解析、校验和输出逻辑，现有行为必须保持。",
           "A {n}-line transformation module mixes parsing, validation, and output logic, and existing behavior must remain.",
           "提出重构方案并给出关键代码，说明如何证明外部行为未改变。",
           "Propose a refactor with key code and explain how unchanged external behavior is verified.",
           ("FT", "LC", "SC", "DE"), ("refactoring plan", "code diff outline")),
    _frame("code", "repository-review", "{org}的仓库包含入口、配置、缓存和测试四部分，问题只在特定调用顺序出现。",
           "The {org} repository has entry, configuration, cache, and test modules; the defect appears only in one call order.",
           "综合各模块信息解释执行路径，并指出最小修改位置。",
           "Integrate evidence across modules, explain the execution path, and identify the smallest change location.",
           ("LC", "MH", "DE", "AM"), ("review memo", "call-flow diagram")),
    _frame("code", "data-pipeline", "{org}每天接收{n}个批次的CSV文件，不同来源对日期和缺失值使用不同表示。",
           "{org} receives {n} CSV batches daily, with source-specific date and missing-value conventions.",
           "设计标准化流水线，保留可追踪的错误记录和输出契约。",
           "Design a normalization pipeline with traceable errors and a stable output contract.",
           ("DS", "AU", "FT", "SC"), ("pipeline pseudocode", "JSON schema")),
    _frame("code", "performance", "现有搜索过程需要检查{n}个状态，在边界输入下会明显变慢。",
           "The current search examines {n} states and slows sharply on boundary inputs.",
           "比较两种优化方式，给出复杂度依据和可复现的验证方法。",
           "Compare two optimizations with complexity evidence and a reproducible validation method.",
           ("AR", "SQ", "AM", "SC"), ("analysis with code", "benchmark plan")),

    # math
    _frame("math", "equation-system", "一个资源分配问题形成{n}个变量和{k}条约束，其中两条约束可能相关。",
           "A resource allocation problem yields {n} variables and {k} constraints, two of which may be dependent.",
           "建立方程、求解，并检查解是否满足原始限制。",
           "Form the equations, solve them, and verify the result against the original constraints.",
           ("SQ", "AR", "SC", "AM"), ("stepwise derivation", "solution table")),
    _frame("math", "probability", "抽样过程分为{k}个阶段，每个阶段的选择会改变下一阶段的候选数。",
           "A sampling process has {k} stages, and each choice changes the candidates in the next stage.",
           "计算目标事件概率，并说明各阶段为何不能简单独立相乘。",
           "Compute the target probability and explain why the stages cannot be multiplied as independent events.",
           ("SQ", "MH", "AM", "SC"), ("probability tree", "formula derivation")),
    _frame("math", "proof", "给定一个由{n}项构成的递推序列，需要判断命题对所有正整数是否成立。",
           "A recurrence with {n} initial terms is used in a claim that should hold for all positive integers.",
           "给出证明，并单独检查起始项和递推边界。",
           "Provide a proof and check the initial cases and recurrence boundary separately.",
           ("AR", "MH", "SC", "SQ"), ("formal proof", "proof outline")),
    _frame("math", "statistics", "{org}比较两组观测，其中一组有缺失值，另一组包含一个明显离群点。",
           "{org} compares two samples; one has missing values and the other contains a clear outlier.",
           "选择合适的汇总量并解释结论对处理方式是否敏感。",
           "Choose appropriate summaries and explain whether the conclusion is sensitive to data handling.",
           ("SQ", "DS", "AM", "FT"), ("calculation table", "statistical note")),
    _frame("math", "geometry", "一个复合图形由{k}个矩形和两个重叠区域组成，部分尺寸通过比例给出。",
           "A composite shape contains {k} rectangles and two overlapping regions, with some dimensions given as ratios.",
           "推导总面积，标明重复计算的区域并验证单位。",
           "Derive the total area, identify double-counted regions, and verify units.",
           ("SQ", "AR", "SC", "MH"), ("annotated derivation", "formula list")),
    _frame("math", "optimization", "{org}要在预算和容量限制下分配{n}项资源，并存在最低服务量。",
           "{org} allocates {n} resources under budget, capacity, and minimum-service constraints.",
           "建立目标函数，比较候选方案并说明最优性条件。",
           "Formulate the objective, compare candidate solutions, and explain the optimality conditions.",
           ("AR", "SQ", "MH", "SC"), ("optimization model", "decision table")),
    _frame("math", "word-problem", "一段业务描述同时给出总量、变化率和时间范围，但“本期”可能有两种解释。",
           "A business problem gives totals, a rate of change, and a time range, but 'current period' has two plausible meanings.",
           "列出合理解释，选择可计算的定义并分别给出结果。",
           "List the plausible interpretations, select computable definitions, and give each result.",
           ("AM", "SQ", "MH", "FT"), ("assumption table", "worked solution")),
    _frame("math", "chart-analysis", "一张表记录{k}个月的数量和比例，部分月份的统计口径发生变化。",
           "A table reports counts and ratios over {k} months, with a definition change in some months.",
           "计算可比变化并避免把口径变化误当作真实增长。",
           "Calculate comparable changes without treating a definition change as real growth.",
           ("DS", "SQ", "FT", "AM"), ("comparison table", "calculation memo")),

    # qa
    _frame("qa", "technical-docs", "用户根据{system}的两段版本说明询问一个配置项为何失效。",
           "A user asks why a setting stopped working after reading two release-note sections for {system}.",
           "结合版本条件解释原因，并给出可验证的处理步骤。",
           "Use the version conditions to explain the cause and give verifiable resolution steps.",
           ("FK", "LC", "AU", "SC"), ("technical answer", "checklist")),
    _frame("qa", "evidence-chain", "现有三段记录分别描述事件前、事件中和事件后的状态，但时间戳格式不同。",
           "Three records describe states before, during, and after an event, but their timestamp formats differ.",
           "判断事件顺序，并说明中间记录如何改变最终结论。",
           "Determine the event order and explain how the middle record changes the conclusion.",
           ("MH", "LC", "FK", "AM"), ("evidence chain", "timeline")),
    _frame("qa", "ambiguous-policy", "{org}的一条规则使用“及时”和“重大影响”等未定义词语。",
           "A {org} policy uses undefined terms such as 'promptly' and 'material impact'.",
           "区分能直接回答的部分与需要澄清的部分，并提出具体澄清问题。",
           "Separate what can be answered from what needs clarification and ask concrete follow-up questions.",
           ("AM", "FK", "SC", "FT"), ("qualified answer", "clarification list")),
    _frame("qa", "source-comparison", "两份资料对同一现象给出不同解释，一份强调环境，一份强调操作过程。",
           "Two sources explain the same observation differently: one emphasizes environment, the other procedure.",
           "比较证据强弱，说明哪些结论能同时得到支持。",
           "Compare the evidence and state which conclusions are supported by both sources.",
           ("MH", "FK", "FT", "LC"), ("comparison matrix", "evidence-based answer")),
    _frame("qa", "troubleshooting", "{system}的操作按文档执行后仍失败，用户提供了环境版本和一段错误信息。",
           "A documented procedure still fails in {system}; the user provides an environment version and one error message.",
           "给出按优先级排列的排查步骤，并说明每步如何缩小原因范围。",
           "Give prioritized troubleshooting steps and explain how each narrows the cause.",
           ("AU", "DE", "SC", "FK"), ("troubleshooting tree", "ordered checklist")),
    _frame("qa", "data-explanation", "{org}的指标在{k}个月内上升，但样本量和缺失率也同时变化。",
           "A metric at {org} rises over {k} months while sample size and missingness also change.",
           "解释可能原因，区分数据事实、推断和仍需验证的假设。",
           "Explain possible causes while separating observed facts, inferences, and unverified hypotheses.",
           ("DS", "FK", "SQ", "AM"), ("layered explanation", "evidence table")),
    _frame("qa", "long-policy", "一份长制度在开头定义适用范围，在后文列出例外和过渡安排。",
           "A long policy defines scope at the beginning and lists exceptions and transition rules later.",
           "回答具体情形是否适用，并引用相互关联的条款。",
           "Decide whether a concrete case is covered and cite the related clauses.",
           ("LC", "AM", "FT", "MH"), ("clause-based answer", "decision memo")),
    _frame("qa", "bilingual-terms", "用户同时提供中文规范和英文接口文档，两个版本对一个术语的范围表述不同。",
           "The user provides a Chinese specification and an English API document that scope one term differently.",
           "统一术语后解释差异，并避免把翻译差别当作事实冲突。",
           "Normalize terminology, explain the difference, and avoid treating translation variation as factual conflict.",
           ("MT", "FK", "SC", "LC"), ("bilingual explanation", "terminology table")),

    # summary
    _frame("summary", "meeting", "{org}的会议记录包含产品、研发和运营三方意见，部分决定带有前置条件。",
           "A {org} meeting record contains product, engineering, and operations views, with conditional decisions.",
           "提炼决定、分歧、负责人和未决事项，不补写会议中没有的信息。",
           "Extract decisions, disagreements, owners, and open items without adding unsupported facts.",
           ("LC", "FT", "SC", "AM"), ("action table", "executive summary")),
    _frame("summary", "research", "一份研究材料包含方法、{n}个观察结果、限制和作者的谨慎结论。",
           "A research note includes methods, {n} observations, limitations, and a cautious conclusion.",
           "压缩主要发现，同时保留限制并区分观察与解释。",
           "Condense the findings while preserving limitations and separating observations from interpretations.",
           ("FT", "FK", "LC", "SC"), ("structured abstract", "bullet summary")),
    _frame("summary", "incident", "故障记录由监控、值班人员和修复日志组成，三部分时间范围有重叠。",
           "An incident record combines monitoring, operator notes, and a repair log with overlapping time ranges.",
           "总结时间线、影响、已证实原因和待验证事项。",
           "Summarize the timeline, impact, confirmed cause, and items still requiring verification.",
           ("LC", "DE", "FT", "MH"), ("incident brief", "timeline table")),
    _frame("summary", "metrics", "月报列出收入、退款和活跃用户，但两个表使用不同统计周期。",
           "A monthly report lists revenue, refunds, and active users, but two tables use different periods.",
           "给出可比的管理摘要，并明确不可直接比较的数据。",
           "Produce a comparable management summary and flag figures that should not be directly compared.",
           ("DS", "SQ", "FT", "AM"), ("metric table", "management note")),
    _frame("summary", "conflicting-reports", "两份项目报告对延期原因和完成比例存在冲突，附件提供了部分佐证。",
           "Two project reports conflict on delay causes and completion rate, with partial evidence in an appendix.",
           "并列冲突、评估证据，并给出不超出材料的综合摘要。",
           "Present the conflicts, assess evidence, and write a synthesis that stays within the material.",
           ("AM", "MH", "FT", "LC"), ("conflict matrix", "synthesis")),
    _frame("summary", "technical-guide", "一份{system}指南跨多个章节描述安装、配置、失败恢复和版本限制。",
           "A {system} guide covers installation, configuration, recovery, and version constraints across chapters.",
           "为实施人员整理顺序化摘要，保留关键前置条件。",
           "Create an ordered implementation summary that retains important prerequisites.",
           ("LC", "AU", "FT", "SC"), ("runbook summary", "ordered checklist")),
    _frame("summary", "bilingual-report", "同一项目有中英文两版进展材料，英文版包含一项中文版本没有的更新。",
           "A project has Chinese and English progress notes, with one update appearing only in the English version.",
           "统一术语并形成合并摘要，标注仅来自单一版本的信息。",
           "Normalize terminology and create a merged summary that marks version-specific information.",
           ("MT", "FT", "LC", "AM"), ("bilingual summary", "term-aligned table")),
    _frame("summary", "executive", "管理层需要从{n}页材料中快速了解结论、风险、数字和下一步。",
           "Executives need conclusions, risks, figures, and next steps from a {n}-page document.",
           "在严格字段和长度限制下生成可核查摘要。",
           "Produce a verifiable summary under strict field and length constraints.",
           ("SC", "FT", "FK", "LC"), ("JSON summary", "four-section brief")),

    # translation
    _frame("translation", "technical", "{system}的技术说明包含版本号、命令、参数名和故障恢复步骤。",
           "A technical note for {system} includes versions, commands, parameter names, and recovery steps.",
           "完成翻译，保持技术标识不变并统一关键术语。",
           "Translate the note while preserving technical identifiers and consistent terminology.",
           ("MT", "FT", "AU", "SC"), ("parallel translation", "translated guide")),
    _frame("translation", "contract", "合同片段包含交付批次、验收条件和一个指代不清的例外条款。",
           "A contract excerpt includes delivery batches, acceptance conditions, and an exception with an unclear reference.",
           "忠实翻译并标出无法在不澄清的情况下唯一确定的指代。",
           "Translate faithfully and flag references that cannot be resolved without clarification.",
           ("FT", "MT", "AM", "SC"), ("clause-aligned translation", "bilingual clauses")),
    _frame("translation", "notice", "{org}的通知包含日期、地点、报名条件和临时变更安排。",
           "A {org} notice contains dates, location, eligibility, and contingency changes.",
           "翻译为目标语言，保留项目顺序和所有条件。",
           "Translate into the target language while preserving item order and every condition.",
           ("FT", "SC", "MT", "AM"), ("formatted notice", "bilingual notice")),
    _frame("translation", "domain-report", "一份{domain}报告使用多个缩写，并在后文重新定义其中一个术语。",
           "A {domain} report uses several abbreviations and redefines one term in a later section.",
           "翻译节选并使术语选择与上下文定义一致。",
           "Translate the excerpt and align terminology with the contextual definitions.",
           ("FK", "MT", "FT", "LC"), ("annotated translation", "terminology table")),
    _frame("translation", "ui-localization", "{system}的界面文本含按钮、占位符、错误消息和字符长度限制。",
           "The {system} UI text contains buttons, placeholders, errors, and character limits.",
           "完成本地化，保持占位符和字段键不变。",
           "Localize the text while preserving placeholders and field keys.",
           ("MT", "AU", "SC", "FT"), ("key-value table", "localized resource")),
    _frame("translation", "incident", "双语故障记录分散在工单、聊天和修复说明中，时间线需要保持一致。",
           "A bilingual incident record is split across a ticket, chat, and repair note, and its timeline must remain consistent.",
           "翻译并合并相关段落，不改变事件顺序或确定性程度。",
           "Translate and merge the relevant passages without changing event order or certainty.",
           ("LC", "FT", "MT", "MH"), ("aligned timeline", "merged translation")),
    _frame("translation", "ambiguous-reference", "源文中的“其”和“该方案”可能分别指向前两段的不同对象。",
           "In the source, 'it' and 'the proposal' may refer to different entities in the preceding paragraphs.",
           "给出保留歧义的忠实译文，并列出需要作者确认的指代。",
           "Provide a faithful translation that preserves ambiguity and list references requiring author confirmation.",
           ("AM", "FT", "MT", "LC"), ("translation with notes", "reference table")),
    _frame("translation", "glossary", "{org}提供了双语术语表，但历史材料中有{k}处使用了旧译名。",
           "{org} provides a bilingual glossary, but historical material uses an old translation in {k} places.",
           "按当前术语表修订译文，并记录发生变化的术语。",
           "Revise the translation to the current glossary and record changed terms.",
           ("MT", "FK", "SC", "FT"), ("revised translation", "change log")),
)


FRAMES_BY_TASK = {
    task: tuple(frame for frame in SEMANTIC_FRAMES if frame.task == task)
    for task in TASK_IDS
}

if any(len(frames) != 8 for frames in FRAMES_BY_TASK.values()):
    raise RuntimeError("Phase 1A requires exactly eight semantic frames per task")
if len({frame.frame_id for frame in SEMANTIC_FRAMES}) != len(SEMANTIC_FRAMES):
    raise RuntimeError("semantic frame IDs must be unique")
