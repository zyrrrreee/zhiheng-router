"""Expanded lexical, structural, and language audit for Pilot Surface v2."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
import re
import unicodedata
from typing import Any

from .audit import _tokens, audit_pilot
from .schema import DiagnosticSidecar, ObservableQueryRecord
from .validation_v2 import lint_surface_set, validate_answerability_set


_SENTENCE_SPLIT = re.compile(r"[。！？.!?;；\n]+")
_INTRINSIC_TECHNICAL_TERMS = {
    "api", "sql", "json", "schema", "http", "retry_after", "request_id",
    "rollback", "etag", "python", "worker", "cursor",
}
_STRUCTURAL_HEADINGS = {
    "context", "providedmaterial", "task", "deliverable",
    "背景", "给定材料", "任务", "交付形式",
}


def _normalize_sentence(value: str, *, replace_numbers: bool) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    if replace_numbers:
        value = re.sub(r"\d+(?:\.\d+)?", "#", value)
    return re.sub(r"[^a-z0-9_\u4e00-\u9fff]+", "", value)


def _sentence_frequencies(records: Sequence[ObservableQueryRecord], *, replace_numbers: bool
                         ) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    display: dict[str, str] = {}
    for record in records:
        seen: set[str] = set()
        for sentence in _SENTENCE_SPLIT.split(record.query_text):
            normalized = _normalize_sentence(sentence, replace_numbers=replace_numbers)
            if len(normalized) < 12 or normalized in seen or normalized in _STRUCTURAL_HEADINGS:
                continue
            seen.add(normalized)
            counts[normalized] += 1
            display.setdefault(normalized, " ".join(sentence.split()))
    total = len(records)
    return [
        {"sentence": display[key], "count": count, "share": round(count / total, 6)}
        for key, count in counts.most_common(50) if count >= 3
    ]


def _phrases(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    phrases: set[str] = set()
    for words in re.findall(r"[a-z][a-z0-9_'-]*(?:\s+[a-z][a-z0-9_'-]*){3,}", normalized):
        tokens = words.split()
        phrases.update(" ".join(tokens[index:index + 4]) for index in range(len(tokens) - 3))
    for run in re.findall(r"[\u4e00-\u9fff]{10,}", normalized):
        phrases.update(run[index:index + 10] for index in range(0, len(run) - 9, 3))
    return phrases


def _repeated_phrases(records: Sequence[ObservableQueryRecord]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(sorted(_phrases(record.query_text)))
    return [
        {"phrase": phrase, "count": count, "share": round(count / len(records), 6)}
        for phrase, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:50]
        if count >= 4
    ]


def _risk_label(token: str, support: int, frames: set[str], evidence_variants: set[str]) -> str:
    if token in _INTRINSIC_TECHNICAL_TERMS:
        return "likely_legitimate_semantic_cue"
    if support >= 8 and len(evidence_variants) <= 2:
        return "likely_generator_artifact"
    if support >= 12 and len(frames) <= 2:
        return "likely_generator_artifact"
    return "uncertain"


def _shortcut_diagnostics(observables: Sequence[ObservableQueryRecord],
                          sidecars: Sequence[DiagnosticSidecar]
                         ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    sidecar_by_id = {record.query_id: record for record in sidecars}
    observable_by_id = {record.query_id: record for record in observables}
    token_queries: dict[str, set[str]] = defaultdict(set)
    token_cap_queries: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for record in observables:
        active = sidecar_by_id[record.query_id].active_capabilities
        for token in sorted(_tokens(record.query_text)):
            token_queries[token].add(record.query_id)
            for capability in active:
                token_cap_queries[token][capability].add(record.query_id)

    candidates: list[dict[str, Any]] = []
    masking_rows: list[dict[str, Any]] = []
    for token, query_ids in token_queries.items():
        support = len(query_ids)
        if support < 5:
            continue
        capability, matched_ids = max(
            token_cap_queries[token].items(), key=lambda item: (len(item[1]), item[0]))
        precision = len(matched_ids) / support
        if precision < 0.9:
            continue
        matched_sidecars = [sidecar_by_id[query_id] for query_id in matched_ids]
        frames = {record.semantic_frame_id for record in matched_sidecars}
        styles = {
            record.surface_realization_id.removeprefix("surface-v2-").rsplit("-", 2)[0]
            for record in matched_sidecars
        }
        evidence_variants = {
            dict(record.capability_surface_evidence)[capability]
            for record in matched_sidecars
        }
        task_distribution = dict(sorted(Counter(record.coarse_task for record in matched_sidecars).items()))
        frame_distribution = dict(sorted(Counter(record.semantic_frame_id for record in matched_sidecars).items()))
        classification = _risk_label(token, support, frames, evidence_variants)
        candidate = {
            "token": token,
            "support": support,
            "precision": round(precision, 6),
            "capability": capability,
            "task_distribution": task_distribution,
            "semantic_frame_distribution": frame_distribution,
            "surface_realization_count": len(styles),
            "appears_in_multiple_surface_realizations": len(styles) >= 2,
            "evidence_variant_count": len(evidence_variants),
            "classification": classification,
        }
        candidates.append(candidate)

        retained = 0
        for sidecar in matched_sidecars:
            evidence = dict(sidecar.capability_surface_evidence)[capability]
            masked_evidence = re.sub(re.escape(token), "", evidence, flags=re.IGNORECASE)
            masked_query = re.sub(
                re.escape(token), "", observable_by_id[sidecar.query_id].query_text,
                flags=re.IGNORECASE)
            evidence_remains = (
                token.lower() not in evidence.lower()
                or
                len(_normalize_sentence(masked_evidence, replace_numbers=True)) >= 6
                or bool(re.search(r"[=→{}\[\]|]|\d", masked_evidence))
            )
            if len(_normalize_sentence(masked_query, replace_numbers=True)) >= 30 and evidence_remains:
                retained += 1
        masking_rows.append({
            "token": token,
            "capability": capability,
            "affected_queries": len(matched_sidecars),
            "structure_retained_after_mask": retained,
            "retained_rate": round(retained / len(matched_sidecars), 6),
        })

    candidates.sort(key=lambda row: (-row["precision"], -row["support"], row["token"]))
    masking_rows.sort(key=lambda row: (-row["affected_queries"], row["token"]))

    concentration: dict[str, Any] = {}
    for capability in sorted({cap for sidecar in sidecars for cap in sidecar.active_capabilities}):
        relevant = [record for record in observables
                    if capability in sidecar_by_id[record.query_id].active_capabilities]
        counts: Counter[str] = Counter()
        for record in relevant:
            counts.update(sorted(_tokens(record.query_text)))
        token, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        concentration[capability] = {
            "query_count": len(relevant),
            "unique_token_count": len(counts),
            "top_token": token,
            "top_token_support": count,
            "top_token_share": round(count / len(relevant), 6),
        }

    masking_summary = {
        "candidate_count": len(masking_rows),
        "minimum_structure_retained_rate": min(
            (row["retained_rate"] for row in masking_rows), default=1.0),
        "diagnostics": masking_rows[:100],
        "method": "Remove the associated token from its capability evidence span and check whether substantive structure remains; no Predictor is trained.",
    }
    return candidates[:100], concentration, masking_summary


def _realization_diversity(sidecars: Sequence[DiagnosticSidecar]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for capability in sorted({cap for sidecar in sidecars for cap in sidecar.active_capabilities}):
        relevant = [record for record in sidecars if capability in record.active_capabilities]
        evidence = {dict(record.capability_surface_evidence)[capability] for record in relevant}
        frames = {record.semantic_frame_id for record in relevant}
        styles = {
            record.surface_realization_id.removeprefix("surface-v2-").rsplit("-", 2)[0]
            for record in relevant
        }
        result[capability] = {
            "query_count": len(relevant),
            "semantic_frame_count": len(frames),
            "evidence_variant_count": len(evidence),
            "surface_style_count": len(styles),
        }
    return result


def audit_pilot_v2(config: dict[str, Any], observables: Sequence[ObservableQueryRecord],
                   sidecars: Sequence[DiagnosticSidecar]) -> dict[str, Any]:
    report = audit_pilot(config, observables, sidecars)
    answerability = validate_answerability_set(observables, sidecars)
    lint = lint_surface_set(observables)
    exact_sentences = _sentence_frequencies(observables, replace_numbers=False)
    normalized_sentences = _sentence_frequencies(observables, replace_numbers=True)
    repeated_phrases = _repeated_phrases(observables)
    candidates, concentration, masking = _shortcut_diagnostics(observables, sidecars)
    diversity = _realization_diversity(sidecars)
    max_exact = max((row["share"] for row in exact_sentences), default=0.0)
    max_normalized = max((row["share"] for row in normalized_sentences), default=0.0)
    max_phrase = max((row["share"] for row in repeated_phrases), default=0.0)
    added_checks = {
        "content_payload_presence": answerability["valid_count"] == len(observables),
        "task_specific_answerability": answerability["valid_count"] == len(observables),
        "surface_lint": lint["failure_count"] == 0,
        "capability_realization_diversity": all(
            row["evidence_variant_count"] >= 4 and row["semantic_frame_count"] >= 3
            for row in diversity.values()
        ),
        "cue_masking_structure_retained": masking["minimum_structure_retained_rate"] >= 0.9,
        "normalized_fixed_sentence_below_v1": max_normalized < 0.104,
    }
    report["checks"].update(added_checks)
    report.update({
        "stage": "pilot_surface_revision_2",
        "ready_for_human_review": False,
        "ready_for_spot_review": all(report["checks"].values()),
        "answerability": answerability,
        "surface_lint": lint,
        "exact_sentence_frequency": exact_sentences,
        "normalized_sentence_frequency": normalized_sentences,
        "repeated_multiword_phrase_frequency": repeated_phrases,
        "max_exact_sentence_share": max_exact,
        "max_normalized_sentence_share": max_normalized,
        "max_repeated_phrase_share": max_phrase,
        "capability_realization_diversity": diversity,
        "per_capability_lexical_concentration": concentration,
        "token_capability_association_candidates": candidates,
        "token_capability_candidate_count": len(candidates),
        "likely_generator_artifact_count": sum(
            row["classification"] == "likely_generator_artifact" for row in candidates),
        "cue_masking_diagnostic": masking,
        "manual_review_required": {
            "spot_review_scores_are_unfilled": True,
            "full_human_scores_are_unfilled": True,
            "human_surface_audit_passed": False,
        },
        "limitations": [
            "Lexical and structural diagnostics do not establish naturalness or Human Surface Audit PASS.",
            "No quality, cost, latency, outcome, Predictor, Router, or baseline result is generated.",
        ],
    })
    return report
