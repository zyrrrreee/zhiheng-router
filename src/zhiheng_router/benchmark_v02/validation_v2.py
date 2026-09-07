"""Structural answerability checks and lightweight language lint for Pilot v2."""

from __future__ import annotations

from collections.abc import Sequence
import re
from typing import Any

from .schema import DiagnosticSidecar, ObservableQueryRecord


_PASSAGE_RE = re.compile(
    r"\[(?:材料|证据|记录|段落|事件|表|说明|报告|附件|备注|章节|附录|"
    r"中文|结论|风险|决定|下一步|Evidence|Definition|Exception|Record|Appendix|"
    r"Note|Paragraph|Event|Table|Report|Chapter|Chinese|English|Finding|Risk|Decision|Next step)[^\]]*\]",
    re.IGNORECASE,
)
_OLD_INSTRUCTION_CUES = re.compile(
    r"(?i)(?:connect dependencies across the material|compare two viable approaches|"
    r"state the variable relationships|follow the requested fields exactly|"
    r"do not assume an unstated definition|preserve conditions, figures, and uncertainty|"
    r"link at least three pieces of information)"
)


def validate_answerability(record: ObservableQueryRecord,
                           sidecar: DiagnosticSidecar) -> dict[str, Any]:
    text = record.query_text
    task = sidecar.coarse_task
    checks: dict[str, bool]
    if task == "code":
        material_present = bool(re.search(
            r"```(?:python|sql)?|schema|接口|endpoint|模块关系|module path|来源 A|source A|"
            r"任务：A|任务耗时|jobs: A|durations:", text, re.IGNORECASE))
        checks = {"analyzable_payload": material_present}
        if sidecar.semantic_frame_id.endswith("service-debug"):
            checks.update({
                "debug_code": "```python" in text,
                "debug_log": bool(re.search(r"日志|\bLog:", text)),
                "debug_sample_io": bool(re.search(r"失败输入|Failing input", text))
                                   and bool(re.search(r"预期|Expected", text)),
            })
    elif task == "math":
        checks = {
            "numeric_conditions": len(re.findall(r"\d+(?:\.\d+)?", text)) >= 3,
            "relation_or_operation": bool(re.search(
                r"[=+×/%]|概率|面积|均值|利润|总数|总收入|方程|equation|probability|"
                r"area|mean|profit|total|revenue|hours|tickets", text, re.IGNORECASE)),
            "explicit_target": bool(re.search(
                r"求|计算|证明|找出|最大|find|compute|prove|solve|maximi[sz]e|determine|calculate",
                text, re.IGNORECASE)),
        }
    elif task == "qa":
        checks = {
            "evidence_passages": len(_PASSAGE_RE.findall(text)) >= 2,
            "evidence_question": bool(re.search(
                r"\?|？|是否|能否|解释|说明|判断|why|what|should|can |explain|assess|identify",
                text, re.IGNORECASE)),
        }
    elif task == "summary":
        checks = {
            "source_passages": len(_PASSAGE_RE.findall(text)) >= 3,
            "summary_target": bool(re.search(
                r"摘要|总结|整理|提取|提炼|压缩|形成|生成|统一|summary|summarize|create|produce|normalize",
                text, re.IGNORECASE)),
        }
    else:
        source_markers = tuple(marker for marker in (
            "[源文]", "[Source text]", "[中文版本]", "[English version]"
        ) if marker in text)
        checks = {
            "source_text": bool(source_markers),
            "source_separated": "给定材料" in text and len(text) >= 100,
            "translation_target": bool(re.search(
                r"译|翻译|本地化|translate|translation|校订|localize|revise",
                text, re.IGNORECASE)),
        }
    failures = [name for name, passed in checks.items() if not passed]
    return {"query_id": record.query_id, "task": task, "valid": not failures,
            "checks": checks, "failures": failures}


def validate_answerability_set(observables: Sequence[ObservableQueryRecord],
                               sidecars: Sequence[DiagnosticSidecar]) -> dict[str, Any]:
    sidecar_by_id = {record.query_id: record for record in sidecars}
    results = [validate_answerability(record, sidecar_by_id[record.query_id])
               for record in observables]
    failures = [result for result in results if not result["valid"]]
    by_task = {
        task: {
            "valid": sum(result["valid"] for result in results if result["task"] == task),
            "total": sum(result["task"] == task for result in results),
        }
        for task in sorted({result["task"] for result in results})
    }
    return {
        "valid_count": len(results) - len(failures),
        "total": len(results),
        "valid_rate": round((len(results) - len(failures)) / len(results), 6),
        "by_task": by_task,
        "failures": failures,
    }


def lint_surface(text: str, language: str) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    patterns = (
        ("duplicated_determiner", r"(?i)\b(the|a|an)\s+\1\b|\b(?:a|an)\s+the\b"),
        ("repeated_english_word", r"(?i)\b([a-z]{2,})\s+\1\b"),
        ("repeated_chinese_phrase", r"([\u4e00-\u9fff]{2,4})\1"),
        ("punctuation_run", r"[,.;:!?，。；：！？]{4,}"),
        ("mixed_punctuation_collision", r"(?:，,|,，|。\.|\.。|：:|:：|；;|;；)"),
    )
    for code, pattern in patterns:
        match = re.search(pattern, text)
        if match:
            failures.append({"code": code, "match": match.group()})
    if language == "mixed":
        match = _OLD_INSTRUCTION_CUES.search(text)
        if match:
            failures.append({"code": "hard_inserted_english_instruction", "match": match.group()})
    return failures


def lint_surface_set(observables: Sequence[ObservableQueryRecord]) -> dict[str, Any]:
    failures = [
        {"query_id": record.query_id, "language": record.language, "issues": issues}
        for record in observables
        if (issues := lint_surface(record.query_text, record.language))
    ]
    return {"failure_count": len(failures), "failures": failures}
