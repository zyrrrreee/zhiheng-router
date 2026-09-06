"""Three-layer Pilot Query composition and surface realization."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .rng import NamespaceRNG
from .schema import (
    DiagnosticSidecar,
    GenerationProvenance,
    ObservableQueryRecord,
    TASK_IDS,
    VisibleStructure,
    validate_pilot_records,
)
from .semantic_frames import FRAMES_BY_TASK, SemanticFrame


SURFACE_STYLES = ("direct", "scenario", "constraints-first", "constraints-last")

CAPABILITY_RUBRICS = {
    "AR": "比较可行路径并检查关键步骤与边界条件",
    "SQ": "明确变量关系、计算过程与数值核验",
    "DE": "利用失败现象定位最早偏差并设计回归检查",
    "DS": "正确处理表格、连接、聚合或数据口径",
    "AU": "遵守接口或库约束并处理失败路径",
    "FK": "区分材料事实、领域常识和待验证假设",
    "MH": "连接多段证据或有依赖的条件",
    "LC": "综合相隔较远的材料并保持前后一致",
    "FT": "保留原意和事实，不添加材料外信息",
    "MT": "跨语言保持术语和语域一致",
    "AM": "识别多种合理解释并说明澄清需求",
    "SC": "严格满足字段、格式、长度和结构约束",
}

_CUES = {
    "AR": {
        "zh": ("比较两条可行路径，并说明关键步骤在边界输入下为何成立。", "给出选择方案的依据，并检查最小规模和极端规模的情况。", "先列出候选做法，再解释其中一条在特殊输入下是否仍有效。"),
        "en": ("Compare two viable approaches and justify the key step on boundary inputs.", "Explain the choice of approach and check both minimal and extreme cases.", "List candidate approaches before testing whether the chosen one still works on an edge case."),
    },
    "SQ": {
        "zh": ("写清变量之间的关系，并对最终数值做一次反向核验。", "涉及比例和总量时保留计算过程，并检查单位是否一致。", "给出所用公式，数值结果保留两位小数并说明舍入位置。"),
        "en": ("State the variable relationships and verify the final number in reverse.", "Show calculations for ratios and totals, and check that units remain consistent.", "Give the formula, round numerical results to two decimals, and state where rounding occurs."),
    },
    "DE": {
        "zh": ("结合失败样例找出最早出现偏差的位置，并补一个不会复发的检查。", "不要只处理报错表面；请沿输入到输出定位首次不一致。", "说明如何复现问题、确认根因，并给出覆盖该路径的回归样例。"),
        "en": ("Use the failing example to locate the first divergence and add a regression check.", "Do not patch only the visible error; trace the first inconsistency from input to output.", "Explain reproduction, root-cause confirmation, and a regression case for the failing path."),
    },
    "DS": {
        "zh": ("核对连接键、重复记录和空值处理，避免汇总口径被放大。", "先说明数据粒度，再处理筛选、分组和聚合。", "对表间匹配失败和统计口径变化给出明确处理规则。"),
        "en": ("Check join keys, duplicate rows, and null handling so aggregation is not inflated.", "State the data grain before filtering, grouping, and aggregation.", "Define how unmatched rows and changes in reporting definitions are handled."),
    },
    "AU": {
        "zh": ("考虑接口版本、分页和失败重试，并保留可诊断的错误信息。", "调用过程需要说明版本前提、超时处理和返回字段校验。", "请覆盖限流、部分成功和接口返回缺少字段的情况。"),
        "en": ("Handle API versioning, pagination, retries, and diagnostic error details.", "State version prerequisites, timeout handling, and response-field validation.", "Cover rate limiting, partial success, and responses with missing fields."),
    },
    "FK": {
        "zh": ("把材料中明确给出的事实、常见领域知识和仍待确认的假设分开。", "关键结论要指出依据来自哪段材料，无法核实的内容需明确标注。", "遇到领域术语时给出准确含义，不把推测写成已证实事实。"),
        "en": ("Separate stated facts, established domain knowledge, and assumptions that still need verification.", "Tie each key conclusion to its supporting passage and flag anything unverifiable.", "Define domain terms accurately and do not present an inference as a confirmed fact."),
    },
    "MH": {
        "zh": ("串联前后材料中的依赖关系，并说明中间条件如何改变结论。", "按发生顺序连接至少三条信息，不要跳过决定性中间步骤。", "先处理前置条件，再利用其结果判断后续分支。"),
        "en": ("Connect dependencies across the material and explain how an intermediate condition changes the conclusion.", "Link at least three pieces of information in order without skipping the decisive middle step.", "Resolve prerequisites first, then use their results to evaluate later branches."),
    },
    "LC": {
        "zh": ("同时使用开头的定义和后文的例外，保持跨段引用一致。", "综合相隔较远的材料，遇到冲突时指出各自适用范围。", "结论需要覆盖材料前、中、后三部分，不能只依据最近一段。"),
        "en": ("Use both the opening definitions and later exceptions while keeping cross-references consistent.", "Integrate distant sections and state the scope of each when they conflict.", "Base the conclusion on the beginning, middle, and end rather than only the nearest paragraph."),
    },
    "FT": {
        "zh": ("保留原材料中的条件、数字和不确定性，不补写未提供的事实。", "压缩或改写时不得改变因果方向，也不要遗漏限制条件。", "区分原文结论和编辑说明，所有实体与数值需保持一致。"),
        "en": ("Preserve conditions, figures, and uncertainty without adding facts absent from the source.", "Do not reverse causality or omit limitations while condensing or rewriting.", "Separate source conclusions from editorial notes and keep every entity and figure consistent."),
    },
    "MT": {
        "zh": ("术语首次出现时给出统一译法，后文保持一致并保留必要英文缩写。", "依据给定语境选择译名，不要在同一文档中切换同义术语。", "核对双语术语的范围是否一致，并记录需要保留的原文标识。"),
        "en": ("Choose one translation when a term first appears, reuse it consistently, and preserve required abbreviations.", "Select terminology from context and avoid switching synonyms within the same document.", "Check whether bilingual terms have the same scope and record identifiers that must remain unchanged."),
    },
    "AM": {
        "zh": ("列出可能影响答案的两种解释，并说明需要补充哪项信息才能确定。", "不要默认未说明的口径；先写出采用的解释及其影响。", "发现指代或范围不唯一时保留歧义，并提出具体澄清问题。"),
        "en": ("List two interpretations that could change the answer and state what information would resolve them.", "Do not assume an unstated definition; declare the chosen interpretation and its effect.", "Preserve unresolved reference or scope ambiguity and ask a concrete clarification question."),
    },
    "SC": {
        "zh": ("严格按指定字段输出；缺少信息时使用空值并在备注中解释。", "保持字段顺序和层级，不得增加未声明的顶层字段。", "最终答案需符合给定格式，并逐项核对必填内容。"),
        "en": ("Follow the requested fields exactly; use null for missing information and explain it in notes.", "Preserve field order and hierarchy without adding undeclared top-level fields.", "Match the requested format and verify every required item before finishing."),
    },
}

_ORGANIZATIONS = ("研发团队", "运维中心", "教学项目组", "数据平台主管", "区域服务部门")
_ORGANIZATIONS_EN = ("the engineering team", "the operations center", "a teaching project", "the data platform team", "a regional service unit")
_SYSTEMS = ("订单服务", "日志平台", "资产管理系统", "知识库接口", "报表流水线")
_SYSTEMS_EN = ("the order service", "the logging platform", "the asset system", "the knowledge-base API", "the reporting pipeline")
_DOMAINS = ("设备维护", "供应链", "环境监测", "软件交付", "客户支持")
_DOMAINS_EN = ("equipment maintenance", "supply chain", "environmental monitoring", "software delivery", "customer support")

_TASK_CONSTRAINTS = {
    "code": (
        "给出一个最小可运行示例。", "说明错误输入如何处理。", "不要依赖未声明的第三方服务。",
        "列出两条关键测试。", "接口名称保持前后一致。",
    ),
    "math": (
        "写出中间步骤。", "检查结果是否满足原条件。", "说明使用的单位。",
        "不要省略边界情况。", "区分精确值与近似值。",
    ),
    "qa": (
        "把事实和推断分开。", "给出可执行的下一步。", "无法由材料确定时明确说明。",
        "引用相关材料位置。", "回答控制在四个小节内。",
    ),
    "summary": (
        "不得添加材料外信息。", "保留所有关键数字。", "单独列出未决事项。",
        "冲突信息需并列呈现。", "摘要不超过四个小节。",
    ),
    "translation": (
        "保留数字、日期和专有标识。", "不要擅自消除原文歧义。", "术语译法保持一致。",
        "保留原有段落结构。", "必要说明放在译者注中。",
    ),
}

_TASK_CONSTRAINTS_EN = {
    "code": ("Include a minimal runnable example.", "Explain invalid-input handling.", "Do not rely on undeclared external services.", "List two key tests.", "Keep interface names consistent."),
    "math": ("Show intermediate steps.", "Check the result against the original conditions.", "State the units used.", "Do not omit boundary cases.", "Distinguish exact and approximate values."),
    "qa": ("Separate facts from inferences.", "Give an actionable next step.", "State when the material is insufficient.", "Cite the relevant material section.", "Use no more than four sections."),
    "summary": ("Do not add information absent from the source.", "Preserve every key figure.", "List unresolved items separately.", "Present conflicting information side by side.", "Use no more than four sections."),
    "translation": ("Preserve figures, dates, and identifiers.", "Do not silently remove source ambiguity.", "Use terminology consistently.", "Preserve paragraph structure.", "Put necessary explanations in translator notes."),
}

_LABEL_PATTERN = re.compile(
    r"(?i)(?:capability\s*[=:]|algorithmic_reasoning|symbolic_quantitative|"
    r"debugging_error_localization|data_sql|api_library_usage|factual_domain_knowledge|"
    r"multi_hop_reasoning|long_context_integration|faithful_transformation|"
    r"multilingual_terminology|ambiguity_resolution|structured_constraint_following|"
    r"dev-model-[a-z]|quality-best|efficient winner|sampled outcome oracle)"
)


@dataclass(frozen=True)
class SemanticComposition:
    stable_id: str
    query_id: str
    frame: SemanticFrame
    primary_capability: str
    secondary_capabilities: tuple[str, ...]
    capability_weights: dict[str, float]
    requirement_levels: dict[str, float]
    difficulty: float
    difficulty_bin: str
    language: str
    translation_direction: str | None
    scenario: str
    goal: str
    material: str
    constraints: tuple[str, ...]
    output_format: str
    content_values: dict[str, Any]

    @property
    def active_capabilities(self) -> tuple[str, ...]:
        return (self.primary_capability, *self.secondary_capabilities)


def pilot_stable_ids(config: dict[str, Any]) -> list[str]:
    count = config["pilot"]["queries_per_task"]
    return [f"{task}-{index:03d}" for task in TASK_IDS for index in range(count)]


def _parse_stable_id(stable_id: str) -> tuple[str, int]:
    match = re.fullmatch(r"(code|math|qa|summary|translation)-(\d{3})", stable_id)
    if not match:
        raise ValueError(f"malformed Pilot stable ID: {stable_id}")
    return match.group(1), int(match.group(2))


def _cycle_value(values: tuple[str, ...], task: str, index: int, rng: NamespaceRNG,
                 stream: str) -> str:
    offset = rng.derive_seed("content", f"{stream}-{task}", stream="cycle-offset") % len(values)
    return values[(index + offset) % len(values)]


def _balanced_cycle(counts: dict[str, int], order: tuple[str, ...]) -> tuple[str, ...]:
    total = sum(counts.values())
    assigned = {key: 0 for key in order}
    result: list[str] = []
    for position in range(total):
        available = [key for key in order if assigned[key] < counts[key]]
        selected = max(
            available,
            key=lambda key: (counts[key] * (position + 1) / total - assigned[key],
                             -order.index(key)),
        )
        result.append(selected)
        assigned[selected] += 1
    return tuple(result)


def _language_and_direction(config: dict[str, Any], task: str, index: int,
                            rng: NamespaceRNG) -> tuple[str, str | None]:
    if task == "translation":
        directions = _balanced_cycle(
            {"zh_to_en": 8, "en_to_zh": 8, "bilingual_revision": 4},
            ("zh_to_en", "en_to_zh", "bilingual_revision"),
        )
        direction = _cycle_value(directions, task, index, rng, "direction")
        return ("zh" if direction == "zh_to_en" else "mixed"), direction
    probabilities = config["language_distribution"][task]
    counts = {language: round(probabilities[language] * 20) for language in ("zh", "mixed", "en")}
    cycle = _balanced_cycle(counts, ("zh", "mixed", "en"))
    if len(cycle) != 20:
        raise ValueError(f"language distribution for {task} cannot form a 20-item cycle")
    return _cycle_value(cycle, task, index, rng, "language"), None


def _content_values(stable_id: str, rng: NamespaceRNG) -> dict[str, Any]:
    randomizer = rng.random("content", stable_id)
    position = randomizer.randrange(len(_ORGANIZATIONS))
    system_position = randomizer.randrange(len(_SYSTEMS))
    domain_position = randomizer.randrange(len(_DOMAINS))
    n = randomizer.randint(18, 180)
    k = randomizer.randint(3, min(18, n - 1))
    return {
        "org": _ORGANIZATIONS[position], "org_en": _ORGANIZATIONS_EN[position],
        "system": _SYSTEMS[system_position], "system_en": _SYSTEMS_EN[system_position],
        "domain": _DOMAINS[domain_position], "domain_en": _DOMAINS_EN[domain_position],
        "n": n, "k": k,
    }


def validate_capability_combination(frame: SemanticFrame, primary: str,
                                    secondaries: tuple[str, ...]) -> None:
    active = (primary, *secondaries)
    if not 2 <= len(active) <= 4 or len(set(active)) != len(active):
        raise ValueError("capability combination must contain 2-4 distinct capabilities")
    if not set(active) <= set(frame.compatible_capabilities):
        raise ValueError(f"capability combination is incompatible with {frame.frame_id}")
    active_set = set(active)
    if any(set(pair) <= active_set for pair in frame.incompatible_pairs):
        raise ValueError(f"forbidden capability pair for {frame.frame_id}")


def _weights(active: tuple[str, ...], stable_id: str, rng: NamespaceRNG) -> dict[str, float]:
    randomizer = rng.random("capability", stable_id, stream="weights")
    primary_weight = 0.46 + randomizer.random() * 0.14
    secondary_raw = [0.5 + randomizer.random() for _ in active[1:]]
    remaining = 1.0 - primary_weight
    values = [primary_weight] + [remaining * value / sum(secondary_raw) for value in secondary_raw]
    rounded = [round(value, 12) for value in values]
    rounded[-1] = round(rounded[-1] + (1.0 - sum(rounded)), 12)
    return dict(zip(active, rounded, strict=True))


def _material(task: str, language: str, direction: str | None, scenario_zh: str,
              scenario_en: str, values: dict[str, Any]) -> str:
    n, k = values["n"], values["k"]
    if task == "translation":
        if direction == "zh_to_en":
            return f"待译内容：{scenario_zh} 计划分{k}批处理，并要求在第{n}项记录后再次核对状态。"
        if direction == "en_to_zh":
            return f"Source text: {scenario_en} The work is divided into {k} batches, with a status review after item {n}."
        return (f"现有中英文版本需要统一。中文：{scenario_zh} 英文：{scenario_en} "
                f"The Chinese version records {n} items, while the English appendix mentions {k} review batches.")
    if language == "en":
        materials = {
            "code": f"Input notes: case {n} has {k} dependent operations; the last operation may return an empty field.",
            "math": f"Given values: the total is {n}, the selected subset is {k}, and the stated rate uses the original total.",
            "qa": f"Evidence A reports {n} events, evidence B revises {k} entries, and a later note limits the revision's scope.",
            "summary": f"Source excerpts contain {n} observations, {k} disputed items, two decisions, and one unresolved dependency.",
        }
        return materials[task]
    if language == "mixed":
        materials = {
            "code": f"输入说明包含{n}项操作，其中{k}项有依赖；English log 写明 the final field may be empty.",
            "math": f"题面给出总量{n}和子集{k}；the reported rate is based on the original total，而不是修订值。",
            "qa": f"中文记录列出{n}次事件和{k}项修订；the later note limits the revision to approved cases.",
            "summary": f"材料包含{n}条观察和{k}项争议；English appendix adds one unresolved dependency.",
        }
        return materials[task]
    materials = {
        "code": f"输入摘要：第{n}号样例包含{k}个有依赖的操作，最后一个操作可能返回空字段。",
        "math": f"已知总量为{n}，其中{k}项属于目标子集；题面中的比例以原始总量为基准。",
        "qa": f"记录甲报告{n}次事件，记录乙修订其中{k}项，后续说明把修订范围限定为已批准情形。",
        "summary": f"材料共有{n}条观察、{k}项争议、两项决定和一个尚未解决的依赖条件。",
    }
    return materials[task]


def compose_semantics(config: dict[str, Any], stable_id: str, rng: NamespaceRNG) -> SemanticComposition:
    task, index = _parse_stable_id(stable_id)
    frames = FRAMES_BY_TASK[task]
    frame = frames[index % len(frames)]
    capability_rng = rng.random("capability", stable_id, stream="selection")
    frame_occurrence = index // len(frames)
    primary = frame.compatible_capabilities[
        frame_occurrence % len(frame.compatible_capabilities)
    ]
    difficulty_rng = rng.random("difficulty", stable_id)
    difficulty = round(0.15 + 0.80 * difficulty_rng.random(), 6)
    difficulty_bin = "low" if difficulty < 0.4 else "medium" if difficulty < 0.7 else "high"
    capability_count = 2 if difficulty_bin == "low" else 3 if difficulty_bin == "medium" else 4
    candidates = [capability for capability in frame.compatible_capabilities if capability != primary]
    secondaries = tuple(capability_rng.sample(candidates, capability_count - 1))
    validate_capability_combination(frame, primary, secondaries)
    active = (primary, *secondaries)
    requirement_rng = rng.random("difficulty", stable_id, stream="requirements")
    requirements = {
        capability: round(min(0.95, 0.35 + 0.42 * difficulty + 0.14 * requirement_rng.random()
                              + (0.03 if capability == primary else 0.0)), 6)
        for capability in active
    }
    language, direction = _language_and_direction(config, task, index, rng)
    values = _content_values(stable_id, rng)
    scenario_zh = frame.scenario_zh.format(
        org=values["org"], system=values["system"], domain=values["domain"],
        n=values["n"], k=values["k"])
    scenario_en = frame.scenario_en.format(
        org=values["org_en"], system=values["system_en"], domain=values["domain_en"],
        n=values["n"], k=values["k"])
    goal_zh = frame.goal_zh.format(**values)
    goal_en = frame.goal_en.format(**values)
    scenario = scenario_en if language == "en" else scenario_zh
    goal = goal_en if language == "en" else goal_zh
    if task == "translation":
        if direction == "zh_to_en":
            goal = "将下面的源文译为英文，并按给定输出要求交付。"
            scenario = scenario_zh
        elif direction == "en_to_zh":
            goal = "将下面的 source text 译为中文，并按给定输出要求交付。"
            scenario = scenario_en
        else:
            goal = "Review the bilingual material，统一术语并给出修订后的双语版本。"
            scenario = f"{scenario_zh} / {scenario_en}"
    material = _material(task, language, direction, scenario_zh, scenario_en, values)
    constraint_pool = _TASK_CONSTRAINTS_EN[task] if language == "en" else _TASK_CONSTRAINTS[task]
    constraint_count = 1 if difficulty_bin == "low" else 2 if difficulty_bin == "medium" else 3
    constraints = tuple(rng.sample("content", stable_id, constraint_pool, constraint_count,
                                   stream="constraints"))
    output_format = rng.choice("content", stable_id, frame.output_formats, stream="output-format")
    return SemanticComposition(
        stable_id=stable_id, query_id=f"pilot-q-{stable_id}", frame=frame,
        primary_capability=primary, secondary_capabilities=secondaries,
        capability_weights=_weights(active, stable_id, rng), requirement_levels=requirements,
        difficulty=difficulty, difficulty_bin=difficulty_bin, language=language,
        translation_direction=direction, scenario=scenario, goal=goal, material=material,
        constraints=constraints, output_format=output_format, content_values=values,
    )


def _surface_cues(composition: SemanticComposition, rng: NamespaceRNG) -> dict[str, str]:
    evidence: dict[str, str] = {}
    for position, capability in enumerate(composition.active_capabilities):
        cue_language = "en" if composition.language == "en" else (
            "en" if composition.language == "mixed" and position % 2 else "zh")
        evidence[capability] = rng.choice(
            "surface", composition.stable_id, _CUES[capability][cue_language],
            stream=f"cue-{capability}")
    return evidence


def _render(composition: SemanticComposition, evidence: dict[str, str], style: str) -> str:
    cues = list(evidence.values())
    if composition.language == "en":
        requirements = " ".join((*cues, *composition.constraints))
        output = f"Output format: {composition.output_format}."
        if style == "direct":
            parts = (composition.scenario, composition.goal, composition.material, requirements, output)
        elif style == "scenario":
            parts = (f"Context: {composition.scenario}", f"Material: {composition.material}",
                     f"Task: {composition.goal}", f"Requirements: {requirements}", output)
        elif style == "constraints-first":
            parts = (f"Requirements first: {requirements}", composition.scenario, composition.material,
                     composition.goal, output)
        else:
            parts = (composition.scenario, composition.material, composition.goal,
                     output, f"Before finalizing: {requirements}")
    else:
        requirements = " ".join((*cues, *composition.constraints))
        output = f"输出形式：{composition.output_format}。"
        if style == "direct":
            parts = (composition.scenario, composition.goal, composition.material, requirements, output)
        elif style == "scenario":
            parts = (f"使用场景：{composition.scenario}", f"现有材料：{composition.material}",
                     f"需要完成：{composition.goal}", f"具体要求：{requirements}", output)
        elif style == "constraints-first":
            parts = (f"请先遵守这些要求：{requirements}", composition.scenario, composition.material,
                     composition.goal, output)
        else:
            parts = (composition.scenario, composition.material, composition.goal,
                     output, f"完成前请逐项确认：{requirements}")
    return "\n\n".join(part.strip() for part in parts if part.strip())


def realize_query(config: dict[str, Any], composition: SemanticComposition,
                  rng: NamespaceRNG) -> tuple[ObservableQueryRecord, DiagnosticSidecar]:
    _, index = _parse_stable_id(composition.stable_id)
    style_offset = rng.derive_seed("surface", composition.frame.task, stream="style-offset")
    style = SURFACE_STYLES[(index + style_offset) % len(SURFACE_STYLES)]
    evidence = _surface_cues(composition, rng)
    query_text = _render(composition, evidence, style)
    if _LABEL_PATTERN.search(query_text):
        raise ValueError(f"surface contains label-like or forbidden model text: {composition.query_id}")
    source_slug = composition.frame.frame_id.removeprefix("frame-")
    group = index // 4
    provenance = GenerationProvenance(
        generator_version=config["generator_version"], config_version=config["config_version"],
        master_seed=config["pilot"]["master_seed"], stable_object_id=composition.stable_id,
        content_namespace=config["pilot"]["namespaces"]["content"],
        capability_namespace=config["pilot"]["namespaces"]["capability"],
        difficulty_namespace=config["pilot"]["namespaces"]["difficulty"],
        surface_namespace=config["pilot"]["namespaces"]["surface"],
    )
    observable = ObservableQueryRecord(
        query_id=composition.query_id, query_text=query_text, language=composition.language,
        explicit_output_constraints=(*composition.constraints,
                                     f"output_format={composition.output_format}"),
        visible_structure=VisibleStructure(
            constraint_count=len(composition.constraints) + len(evidence) + 1,
            input_length_bin=("short" if len(query_text) < 280 else
                              "medium" if len(query_text) < 520 else "long"),
            requested_output_format=composition.output_format,
            translation_direction=composition.translation_direction,
        ),
        source_kind=config["source_kind"], dataset_version=config["dataset_version"],
    )
    sidecar = DiagnosticSidecar(
        query_id=composition.query_id, coarse_task=composition.frame.task,
        primary_capability=composition.primary_capability,
        secondary_capabilities=composition.secondary_capabilities,
        capability_weights=tuple(sorted(composition.capability_weights.items())),
        capability_requirement_levels=tuple(sorted(composition.requirement_levels.items())),
        capability_surface_evidence=tuple(sorted(evidence.items())),
        difficulty=composition.difficulty, difficulty_bin=composition.difficulty_bin,
        source_family_id=f"src-{source_slug}", template_family_id=f"tpl-{source_slug}",
        paraphrase_family_id=f"para-{source_slug}-{group:03d}",
        semantic_frame_id=composition.frame.frame_id,
        surface_realization_id=f"surface-{style}-{composition.stable_id}",
        semantic_scenario=composition.scenario, generation_provenance=provenance,
    )
    return observable, sidecar


def generate_pilot(config: dict[str, Any], stable_ids: list[str] | None = None
                   ) -> tuple[list[ObservableQueryRecord], list[DiagnosticSidecar]]:
    rng = NamespaceRNG(config["pilot"]["master_seed"], config["pilot"]["namespaces"])
    selected_ids = sorted(stable_ids if stable_ids is not None else pilot_stable_ids(config))
    pairs = [realize_query(config, compose_semantics(config, stable_id, rng), rng)
             for stable_id in selected_ids]
    observables = [pair[0] for pair in pairs]
    sidecars = [pair[1] for pair in pairs]
    validate_pilot_records(observables, sidecars)
    return observables, sidecars
