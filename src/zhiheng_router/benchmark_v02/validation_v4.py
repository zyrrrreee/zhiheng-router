"""Deterministic human-review guardrails for Pilot v4."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import hashlib
import re
import unicodedata
from typing import Any

from .audit_v3 import semantic_skeleton
from .schema import DiagnosticSidecar, ObservableQueryRecord
from .semantics_v4 import (
    FRAME_CAPABILITIES_V4,
    FRAME_REQUIRED_CAPABILITIES,
)


_HEADING_RE = re.compile(
    r"^(给定材料|Provided material|任务|Task|交付形式|Deliverable)\r?$", re.MULTILINE)
_HEADING_KEYS = {
    "给定材料": "material", "Provided material": "material",
    "任务": "task", "Task": "task",
    "交付形式": "deliverable", "Deliverable": "deliverable",
}


def _sections(text: str) -> dict[str, str]:
    matches = list(_HEADING_RE.finditer(text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[_HEADING_KEYS[match.group(1)]] = text[match.end():end].strip()
    return sections


def _normalize_sibling_component(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).lower()
    complete_timestamp = (
        re.search(
            r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}\s*月\s*\d{1,2}\s*日?|"
            r"(?:january|february|march|april|may|june|july|august|september|"
            r"october|november|december)\s+\d{1,2}", value)
        and re.search(r"\b\d{1,2}:\d{2}\b", value)
        and re.search(r"utc(?:[+-]\d{1,2}(?::\d{2})?)?|[+-]\d{2}:\d{2}|时区|time zone", value)
    )
    if not complete_timestamp:
        value = re.sub(r"(?m)^.*(?:utc|时区|time zone).*$", "", value)
    value = re.sub(r"\bv\d+(?:\.\d+)*\b", "<version>", value)
    value = re.sub(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", "<date>", value)
    value = re.sub(
        r"\b(?:january|february|march|april|may|june|july|august|september|"
        r"october|november|december)\s+\d{1,2}\b|\d{1,2}\s*月\s*\d{1,2}\s*日?",
        "<date>", value,
    )
    value = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:[+-]\d{2}:\d{2})?\b", "<time>", value)
    value = re.sub(r"\b[a-z][a-z_-]{0,15}[-_]?\d+\b", "<id>", value)
    value = re.sub(r"(?<=['\"`])\w{1,12}\d+(?=['\"`])", "<id>", value)
    value = re.sub(r"(?<![\w])(?:x|y|n|i|j|a|b|c|d|e)(?![\w])", "<var>", value)
    value = re.sub(r"\d+(?:\.\d+)?", "#", value)
    value = re.sub(r"\[(?:材料|说明|附录|记录|段落|事件|报告|附件|备注|结论|风险|决定|下一步)\s*[a-z#·]*\]", "<label>", value)
    value = re.sub(r"\[(?:evidence|definition|exception|record|appendix|note|table|"
                   r"paragraph|event|report|finding|risk|decision|next step)\s*[a-z#·]*\]",
                   "<label>", value)
    return re.sub(r"[^a-z_一-鿿<>+=/%]+", "", value)


def semantic_sibling_signature(text: str) -> str:
    sections = _sections(text)
    canonical = "\0".join(
        _normalize_sibling_component(sections.get(key, ""))
        for key in ("material", "task")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def semantic_sibling_duplicates(
        observables: Sequence[ObservableQueryRecord],
        sidecars: Sequence[DiagnosticSidecar]) -> dict[str, Any]:
    sidecar_by_id = {record.query_id: record for record in sidecars}
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for record in observables:
        key = (
            sidecar_by_id[record.query_id].semantic_frame_id,
            semantic_sibling_signature(record.query_text),
        )
        groups[key].append(record.query_id)
    duplicates = [
        {"semantic_frame_id": frame_id, "signature": signature,
         "query_ids": sorted(query_ids), "count": len(query_ids)}
        for (frame_id, signature), query_ids in sorted(groups.items())
        if len(query_ids) > 1
    ]
    return {"duplicate_group_count": len(duplicates), "duplicate_groups": duplicates}


def _allowed_for_record(record: ObservableQueryRecord,
                        sidecar: DiagnosticSidecar) -> set[str]:
    allowed = set(FRAME_CAPABILITIES_V4[sidecar.semantic_frame_id])
    if sidecar.semantic_frame_id == "frame-math-probability" and not re.search(
            r"隐藏硬币|硬币结果不可见|hidden coin|coin.*not visible",
            record.query_text, re.IGNORECASE):
        allowed.discard("MH")
    return allowed


def validate_capability_roles(
        observables: Sequence[ObservableQueryRecord],
        sidecars: Sequence[DiagnosticSidecar]) -> dict[str, Any]:
    observable_by_id = {record.query_id: record for record in observables}
    necessity_failures: list[dict[str, Any]] = []
    coverage_failures: list[dict[str, Any]] = []
    invalid_count = 0
    missing_count = 0
    for sidecar in sidecars:
        observable = observable_by_id[sidecar.query_id]
        active = set(sidecar.active_capabilities)
        invalid = sorted(active - _allowed_for_record(observable, sidecar))
        if sidecar.coarse_task == "translation":
            invalid = sorted({*invalid, *(active & {"AU", "DE"})})
        if invalid:
            invalid_count += len(invalid)
            necessity_failures.append({"query_id": sidecar.query_id, "capabilities": invalid})

        required = set(FRAME_REQUIRED_CAPABILITIES[sidecar.semantic_frame_id])
        missing = sorted(required - active)
        if sidecar.coarse_task == "translation" and not active & {"MT", "FT"}:
            missing = sorted({*missing, "MT_or_FT"})
        if missing:
            missing_count += len(missing)
            coverage_failures.append({"query_id": sidecar.query_id, "capabilities": missing})
    total = len(sidecars)
    return {
        "capability_necessity_valid_rate": round(
            (total - len(necessity_failures)) / total, 6),
        "capability_coverage_valid_rate": round(
            (total - len(coverage_failures)) / total, 6),
        "invalid_active_capability_count": invalid_count,
        "missing_required_capability_count": missing_count,
        "necessity_failures": necessity_failures,
        "coverage_failures": coverage_failures,
    }


def validate_frame_completeness(
        observables: Sequence[ObservableQueryRecord],
        sidecars: Sequence[DiagnosticSidecar]) -> dict[str, Any]:
    sidecar_by_id = {record.query_id: record for record in sidecars}
    ambiguous_targets: list[str] = []
    incomplete_timezones: list[str] = []
    deliverable_mismatches: list[str] = []
    for record in observables:
        sidecar = sidecar_by_id[record.query_id]
        sections = _sections(record.query_text)
        task = sections.get("task", "")
        deliverable = sections.get("deliverable", "")
        if sidecar.coarse_task == "translation":
            bilingual = "[中文版本]" in record.query_text and "[English version]" in record.query_text
            if bilingual:
                explicit = bool(re.search(
                    r"核对并修订中英文版本|cross-check and revise both (?:the )?chinese and english",
                    task, re.IGNORECASE,
                ))
            else:
                explicit = bool(re.search(
                    r"目标语言为(?:英文|中文)|target language (?:is|:)\s*(?:english|chinese)",
                    task, re.IGNORECASE,
                ))
            if not explicit:
                ambiguous_targets.append(record.query_id)

        needs_timezone_conversion = bool(re.search(
            r"utc.{0,40}(?:日期|统计日|date|reporting day|前一天|prior day|跨日)|"
            r"(?:日期|统计日|date|reporting day|跨日).{0,40}utc",
            record.query_text, re.IGNORECASE,
        ))
        has_complete_timestamp = bool(re.search(
            r"\d{4}-\d{2}-\d{2}[ t]\d{2}:\d{2}(?:\d{2})?[+-]\d{2}:\d{2}",
            record.query_text, re.IGNORECASE,
        ))
        if needs_timezone_conversion and not has_complete_timestamp:
            incomplete_timezones.append(record.query_id)

        asks_timeout = bool(re.search(r"超时|timeout", task, re.IGNORECASE))
        asks_max_retry = bool(re.search(
            r"最大重试|max(?:imum)?[_ -]?(?:retry|retries)", task, re.IGNORECASE))
        if asks_timeout and asks_max_retry and not (
                re.search(r"超时|timeout", deliverable, re.IGNORECASE)
                and re.search(r"最大重试|max(?:imum)?[_ -]?(?:retry|retries)",
                              deliverable, re.IGNORECASE)):
            deliverable_mismatches.append(record.query_id)
    return {
        "ambiguous_translation_target_count": len(ambiguous_targets),
        "incomplete_timezone_conversion_count": len(incomplete_timezones),
        "task_deliverable_mismatch_count": len(deliverable_mismatches),
        "ambiguous_translation_targets": ambiguous_targets,
        "incomplete_timezone_conversions": incomplete_timezones,
        "task_deliverable_mismatches": deliverable_mismatches,
    }


def sibling_difficulty_conflicts(
        observables: Sequence[ObservableQueryRecord],
        sidecars: Sequence[DiagnosticSidecar]) -> dict[str, Any]:
    observable_by_id = {record.query_id: record for record in observables}
    groups: dict[tuple[str, str], list[DiagnosticSidecar]] = defaultdict(list)
    for sidecar in sidecars:
        groups[(sidecar.semantic_frame_id,
                semantic_skeleton(observable_by_id[sidecar.query_id].query_text))].append(sidecar)
    rank = {"low": 0, "medium": 1, "high": 2}
    conflicts = []
    for (frame_id, skeleton), records in sorted(groups.items()):
        bins = {record.difficulty_bin for record in records}
        if bins and max(map(rank.get, bins)) - min(map(rank.get, bins)) >= 2:
            conflicts.append({
                "semantic_frame_id": frame_id,
                "skeleton": hashlib.sha256(skeleton.encode("utf-8")).hexdigest()[:16],
                "query_ids": sorted(record.query_id for record in records),
                "difficulty_bins": sorted(bins, key=rank.get),
            })
    return {
        "severe_sibling_difficulty_conflict_count": len(conflicts),
        "conflicts": conflicts,
    }
