"""R3 payload guardrails for semantic siblings and task completeness."""

from __future__ import annotations

from dataclasses import replace

from .payloads_v2 import ContentPayload
from .payloads_v3 import build_alignment_evidence, generate_content_payload_v3
from .rng import NamespaceRNG
from .semantic_frames import FRAMES_BY_TASK
from .semantics_v3 import payload_variant_index
from .surface import SemanticComposition


TASK_SIBLING_MODIFIERS = {
    "code": (
        ("另外说明一个输入边界及必须保持的行为。",
         "Also state one input boundary and the behavior that must be preserved."),
        ("另外给出一个失败用例及其预期可观察行为。",
         "Also give one failure case and its expected observable behavior."),
    ),
    "math": (
        ("再把一个已给数值改写为参数，并说明结果如何随它变化。",
         "Rewrite one given value as a parameter and state how the result changes with it."),
        ("再用一种独立方法或不变量核验结论。",
         "Verify the conclusion with an independent method or invariant."),
    ),
    "qa": (
        ("另外指出支撑结论的最强证据和一个明确限制。",
         "Also identify the strongest supporting evidence and one explicit limitation."),
        ("另外说明哪一项证据变化会使当前结论不再成立。",
         "Also state which evidence change would make the current conclusion no longer hold."),
    ),
    "summary": (
        ("另外把每项判断标为材料事实或谨慎推断。",
         "Also mark each judgment as a source fact or a cautious inference."),
        ("另外给出一个更短版本，同时保留全部例外和不确定性。",
         "Also provide a shorter version that preserves every exception and uncertainty."),
    ),
    "translation": (
        ("另外对约束最多的一句进行回译核对，并说明残余歧义。",
         "Also back-translate the most constrained sentence and note residual ambiguity."),
        ("另外引用一个决定术语或歧义处理的准确源文片段并说明理由。",
         "Also cite one exact source phrase that controls a terminology or ambiguity choice."),
    ),
}


def sibling_variant_index(composition: SemanticComposition) -> int:
    index = int(composition.stable_id.rsplit("-", 1)[1])
    occurrence = index // len(FRAMES_BY_TASK[composition.frame.task])
    return occurrence // 3


def _append_sibling_requirement(payload: ContentPayload,
                                composition: SemanticComposition) -> ContentPayload:
    cycle = sibling_variant_index(composition)
    if cycle == 0:
        return payload
    zh, en = TASK_SIBLING_MODIFIERS[composition.frame.task][cycle - 1]
    sentence = en if composition.language == "en" else zh
    return replace(
        payload,
        task_instruction=f"{payload.task_instruction} {sentence}",
        structural_markers=tuple(sorted({
            *payload.structural_markers, f"sibling_variant:v4-{cycle}",
        })),
    )


def generate_content_payload_v4(composition: SemanticComposition,
                                rng: NamespaceRNG) -> ContentPayload:
    payload = generate_content_payload_v3(composition, rng)
    variant = payload_variant_index(composition.stable_id)

    if composition.frame.frame_id == "frame-code-data-pipeline" and variant == 1:
        timestamp = (
            "Conversion record: Source B timestamp is `2026-08-03 00:30+08:00`."
            if composition.language == "en"
            else "换算记录：来源 B 的时间戳为 `2026-08-03 00:30+08:00`。"
        )
        payload = replace(payload, material=f"{payload.material}\n{timestamp}")

    if composition.frame.frame_id == "frame-qa-source-comparison" and variant == 1:
        timestamp = (
            "[Evidence E] The event timestamp is `2026-08-03 00:05+08:00`; convert "
            "that complete timestamp before assigning its UTC reporting date."
            if composition.language == "en"
            else "[材料E] 该事件时间戳为 `2026-08-03 00:05+08:00`；先换算完整时间戳，"
                 "再确定其 UTC 统计日。"
        )
        payload = replace(
            payload,
            material=f"{payload.material}\n{timestamp}",
            passage_count=payload.passage_count + 1,
        )

    if composition.frame.frame_id == "frame-translation-incident" and variant == 1:
        timestamp = (
            "[Time anchor] The same event is `2026-08-03 23:30+00:00` in chat and "
            "`2026-08-04 07:30+08:00` in the ticket."
            if composition.translation_direction == "en_to_zh"
            else "[时间锚点] 同一事件在聊天中为 `2026-08-03 23:30+00:00`，"
                 "在工单中为 `2026-08-04 07:30+08:00`。"
        )
        payload = replace(payload, material=f"{payload.material}\n{timestamp}")

    if composition.frame.frame_id == "frame-code-api-client":
        payload = replace(
            payload,
            output_contract=(
                f"{payload.output_contract}; include timeout_policy and maximum_retries fields"
            ),
        )

    if (composition.frame.task == "translation"
            and composition.translation_direction == "bilingual_revision"):
        payload = replace(
            payload,
            task_instruction=(
                "核对并修订中英文版本；" + composition.frame.goal_zh
            ),
        )

    payload = _append_sibling_requirement(payload, composition)
    return replace(
        payload,
        capability_evidence=build_alignment_evidence(composition, payload),
    )
