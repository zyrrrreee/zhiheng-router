"""Deterministic capability-to-evidence semantic alignment checks."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import re
from typing import Any

from .payloads_v3 import _PATTERNS
from .schema import DiagnosticSidecar, ObservableQueryRecord
from .semantics_v3 import FRAME_CAPABILITIES


_SC_PATTERN = re.compile(
    r"json|schema|字段|列|表|签名|函数|小节|格式|证明|步骤|清单|时间线|"
    r"table|field|column|function|section|format|proof|checklist|timeline|"
    r"matrix|contract|resource|guide",
    re.IGNORECASE,
)
_MH_PATTERN = re.compile(
    r"依赖|条件|约束|之后|先|链|修订|回滚|比较|综合|验证|收入|冲突|评估|"
    r"只有|需要|例外|定义|记录|不同|原因|窗口|第一次|第二次|"
    r"depends|condition|constraint|after|before|chain|revision|rollback|step|"
    r"compare|integrate|verify|revenue|conflict|assess|requires|needs|only|"
    r"exception|definition|record|differ|different|cause|window|calendar day|24 hours|"
    r"first|second",
    re.IGNORECASE,
)
_LC_PATTERN = re.compile(
    r"多段|跨|附录|章节|模块|版本|来源|记录|材料|报告|timeline|appendix|"
    r"源文|chapter|module|version|source|source text|bilingual|evidence|report",
    re.IGNORECASE,
)


def _evidence_semantically_valid(capability: str, evidence: str, query_text: str,
                                 task: str) -> bool:
    if capability == "SC":
        return bool(_SC_PATTERN.search(evidence))
    if capability == "MH":
        return (
            ("\n" in evidence and bool(_MH_PATTERN.search(evidence)))
            or (task == "translation"
                and len(re.findall(r"\d{2}:\d{2}", evidence)) >= 3)
            or (task == "math" and bool(re.search(
                    r"连续抽取.*第一次|2 balls.*first|第一次.*第二次|first.*second|"
                    r"总数.*收入|共.*总收入|sold.*total revenue|total.*revenue",
                    evidence, re.IGNORECASE,
                )))
        )
    if capability == "LC":
        return evidence.count("\n") >= 2 or bool(_LC_PATTERN.search(evidence))
    if capability == "SQ":
        return bool(re.search(_PATTERNS["SQ"], evidence, re.IGNORECASE)) and len(
            re.findall(r"\d+(?:\.\d+)?", query_text)) >= 3
    if capability == "FT" and task in {"summary", "translation"}:
        return bool(re.search(_PATTERNS["FT"], evidence, re.IGNORECASE)) or (
            bool(re.search(r"\d", evidence))
            and bool(re.search(r"摘要|总结|翻译|译文|summary|translat", query_text, re.IGNORECASE))
        )
    return bool(re.search(_PATTERNS[capability], evidence, re.IGNORECASE))


def validate_capability_alignment(record: ObservableQueryRecord,
                                  sidecar: DiagnosticSidecar) -> dict[str, Any]:
    evidence_by_capability = dict(sidecar.capability_surface_evidence)
    invalid: list[dict[str, str]] = []
    allowed = set(FRAME_CAPABILITIES.get(sidecar.semantic_frame_id, ()))
    for capability in sidecar.active_capabilities:
        evidence = evidence_by_capability.get(capability, "")
        if capability not in allowed:
            reason = "capability_not_allowed_for_frame"
        elif not evidence or evidence not in record.query_text:
            reason = "evidence_missing_from_observable_query"
        elif not _evidence_semantically_valid(
                capability, evidence, record.query_text, sidecar.coarse_task):
            reason = "evidence_does_not_realize_capability"
        else:
            continue
        invalid.append({"capability": capability, "reason": reason})
    return {"query_id": record.query_id, "valid": not invalid, "invalid": invalid}


def validate_capability_alignment_set(
        observables: Sequence[ObservableQueryRecord],
        sidecars: Sequence[DiagnosticSidecar]) -> dict[str, Any]:
    sidecar_by_id = {record.query_id: record for record in sidecars}
    results = [validate_capability_alignment(record, sidecar_by_id[record.query_id])
               for record in observables]
    failures = [result for result in results if not result["valid"]]
    total_assignments = sum(len(record.active_capabilities) for record in sidecars)
    invalid_counts = Counter(
        issue["capability"] for result in failures for issue in result["invalid"])
    invalid_total = sum(invalid_counts.values())
    return {
        "query_count": len(results),
        "valid_query_count": len(results) - len(failures),
        "active_capability_count": total_assignments,
        "valid_capability_count": total_assignments - invalid_total,
        "invalid_evidence_count": invalid_total,
        "alignment_valid_rate": round(
            (total_assignments - invalid_total) / total_assignments, 6),
        "per_capability_invalid_realization_count": {
            capability: invalid_counts.get(capability, 0)
            for capability in sorted({cap for record in sidecars
                                      for cap in record.active_capabilities})
        },
        "failures": failures,
    }
