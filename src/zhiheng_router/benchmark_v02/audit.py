"""Automatic checks that prepare the Pilot Surface Set for human review."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
import itertools
import math
import re
import statistics
import unicodedata
from typing import Any

from .schema import CAPABILITY_IDS, DiagnosticSidecar, ObservableQueryRecord, TASK_IDS, validate_pilot_records
from .semantic_frames import FRAMES_BY_TASK


_FORBIDDEN_PATTERN = re.compile(
    r"(?i)(?:\[\s*(?:AR|SQ|DE|DS|AU|FK|MH|LC|FT|MT|AM|SC)\s*\]|"
    r"capability\s*[=:]|algorithmic_reasoning|symbolic_quantitative|"
    r"debugging_error_localization|data_sql|api_library_usage|factual_domain_knowledge|"
    r"multi_hop_reasoning|long_context_integration|faithful_transformation|"
    r"multilingual_terminology|ambiguity_resolution|structured_constraint_following|"
    r"dev-model-[a-z]|quality-best|efficient winner|sampled outcome oracle)"
)
_WORD_RE = re.compile(r"[a-z][a-z0-9_-]{1,}|[\u4e00-\u9fff]{2,}", re.IGNORECASE)


def normalize_query(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text)


def _tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens: set[str] = set()
    for match in _WORD_RE.finditer(normalized):
        token = match.group()
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            tokens.update(token[index:index + 2] for index in range(len(token) - 1))
        elif len(token) >= 3:
            tokens.add(token)
    return tokens


def _sentences(text: str) -> set[str]:
    parts = re.split(r"[。！？.!?;；\n]+", text)
    results = set()
    for part in parts:
        normalized = re.sub(r"\d+", "#", " ".join(part.strip().lower().split()))
        if len(normalized) >= 18:
            results.add(normalized)
    return results


def _distribution(values: Sequence[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _language_deviation(config: dict[str, Any], task: str, languages: list[str],
                        directions: list[str]) -> float:
    if task == "translation":
        expected = config["language_distribution"][task]["directions"]
        actual = Counter(directions)
        return max(abs(actual[key] / len(directions) - expected[key]) for key in expected)
    expected = config["language_distribution"][task]
    actual = Counter(languages)
    return max(abs(actual[key] / len(languages) - expected[key]) for key in expected)


def audit_pilot(config: dict[str, Any], observables: Sequence[ObservableQueryRecord],
                sidecars: Sequence[DiagnosticSidecar]) -> dict[str, Any]:
    validate_pilot_records(observables, sidecars)
    by_id = {record.query_id: record for record in observables}
    sidecar_by_id = {record.query_id: record for record in sidecars}
    query_count = len(observables)
    expected_count = config["pilot"]["queries_per_task"] * len(TASK_IDS)

    exact_groups: dict[str, list[str]] = defaultdict(list)
    normalized_groups: dict[str, list[str]] = defaultdict(list)
    sentence_counts: Counter[str] = Counter()
    token_counts: Counter[str] = Counter()
    token_capability_counts: dict[str, Counter[str]] = defaultdict(Counter)
    evidence_missing: list[dict[str, str]] = []
    label_like: list[dict[str, str]] = []
    forbidden_model_or_result: list[str] = []
    capability_counts: Counter[str] = Counter()
    secondary_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()

    for record in observables:
        sidecar = sidecar_by_id[record.query_id]
        exact_groups[record.query_text].append(record.query_id)
        normalized_groups[normalize_query(record.query_text)].append(record.query_id)
        sentence_counts.update(_sentences(record.query_text))
        match = _FORBIDDEN_PATTERN.search(record.query_text)
        if match:
            label_like.append({"query_id": record.query_id, "match": match.group()})
        lowered = record.query_text.lower()
        if "dev-model-" in lowered or "router result" in lowered:
            forbidden_model_or_result.append(record.query_id)
        active = set(sidecar.active_capabilities)
        capability_counts[sidecar.primary_capability] += 1
        secondary_counts.update(sidecar.secondary_capabilities)
        pair_counts.update("+".join(sorted(pair)) for pair in itertools.combinations(active, 2))
        for capability, evidence in sidecar.capability_surface_evidence:
            if evidence not in record.query_text:
                evidence_missing.append({"query_id": record.query_id, "capability": capability})
        for token in _tokens(record.query_text):
            token_counts[token] += 1
            for capability in active:
                token_capability_counts[token][capability] += 1

    exact_duplicates = [ids for ids in exact_groups.values() if len(ids) > 1]
    normalized_duplicates = [ids for ids in normalized_groups.values() if len(ids) > 1]
    frequent_sentences = [
        {"sentence": sentence, "count": count, "share": round(count / query_count, 6)}
        for sentence, count in sentence_counts.most_common()
        if count >= 5
    ]
    strong_associations = []
    for token, support in token_counts.items():
        if support < 5:
            continue
        capability, joint = token_capability_counts[token].most_common(1)[0]
        precision = joint / support
        if precision >= 0.9:
            strong_associations.append({
                "token": token, "capability": capability, "support": support,
                "precision": round(precision, 6),
            })
    strong_associations.sort(key=lambda row: (-row["precision"], -row["support"], row["token"]))

    task_counts = _distribution([sidecar.coarse_task for sidecar in sidecars])
    language_counts = _distribution([record.language for record in observables])
    language_by_task: dict[str, dict[str, int]] = {}
    direction_counts: dict[str, int] = {}
    max_language_deviation = 0.0
    for task in TASK_IDS:
        ids = [record.query_id for record in observables if sidecar_by_id[record.query_id].coarse_task == task]
        task_languages = [by_id[query_id].language for query_id in ids]
        task_directions = [by_id[query_id].visible_structure.translation_direction for query_id in ids
                           if by_id[query_id].visible_structure.translation_direction is not None]
        language_by_task[task] = _distribution(task_languages)
        if task == "translation":
            direction_counts = _distribution(task_directions)
        max_language_deviation = max(
            max_language_deviation,
            _language_deviation(config, task, task_languages, task_directions),
        )

    lengths = [len(record.query_text) for record in observables]
    frame_counts = _distribution([sidecar.semantic_frame_id for sidecar in sidecars])
    expected_frames = {frame.frame_id for frames in FRAMES_BY_TASK.values() for frame in frames}
    difficulty_values = [sidecar.difficulty for sidecar in sidecars]
    active_counts = [len(sidecar.active_capabilities) for sidecar in sidecars]
    max_fixed_sentence_share = max((row["share"] for row in frequent_sentences), default=0.0)

    checks = {
        "query_count": query_count == expected_count and 200 <= query_count <= 300,
        "task_balance": all(task_counts.get(task) == config["pilot"]["queries_per_task"] for task in TASK_IDS),
        "capability_count_2_to_4": all(2 <= count <= 4 for count in active_counts),
        "all_capabilities_covered": set(capability_counts) | set(secondary_counts) == set(CAPABILITY_IDS),
        "language_policy": max_language_deviation <= 0.06,
        "difficulty_bounds": all(math.isfinite(value) and 0 <= value <= 1 for value in difficulty_values),
        "query_length": min(lengths) >= 80 and max(lengths) <= 2000,
        "semantic_frame_coverage": set(frame_counts) == expected_frames,
        "no_exact_duplicate": not exact_duplicates,
        "no_normalized_duplicate": not normalized_duplicates,
        "id_and_family_uniqueness": (
            len({record.query_id for record in observables}) == query_count
            and len({record.query_id for record in sidecars}) == query_count
            and all(sidecar.source_family_id and sidecar.template_family_id
                    and sidecar.paraphrase_family_id for sidecar in sidecars)
        ),
        "surface_evidence_present": not evidence_missing,
        "no_label_like_token": not label_like,
        "no_model_or_result_text": not forbidden_model_or_result,
        "fixed_sentence_share": max_fixed_sentence_share <= 0.20,
    }
    return {
        "stage": "pilot_surface_only",
        "pilot_version": config["pilot"]["pilot_version"],
        "ready_for_human_review": all(checks.values()),
        "checks": checks,
        "query_count": query_count,
        "task_distribution": task_counts,
        "primary_capability_distribution": dict(sorted(capability_counts.items())),
        "secondary_capability_distribution": dict(sorted(secondary_counts.items())),
        "capability_pair_distribution": dict(sorted(pair_counts.items())),
        "language_distribution": language_counts,
        "task_language_distribution": language_by_task,
        "translation_direction_distribution": direction_counts,
        "max_language_policy_deviation": round(max_language_deviation, 6),
        "difficulty_distribution": _distribution([sidecar.difficulty_bin for sidecar in sidecars]),
        "difficulty_summary": {
            "min": min(difficulty_values), "max": max(difficulty_values),
            "mean": round(statistics.fmean(difficulty_values), 6),
        },
        "query_length_summary": {
            "min": min(lengths), "max": max(lengths),
            "mean": round(statistics.fmean(lengths), 3),
        },
        "semantic_frame_coverage": frame_counts,
        "exact_duplicate_groups": exact_duplicates,
        "normalized_duplicate_groups": normalized_duplicates,
        "missing_surface_evidence": evidence_missing,
        "label_like_matches": label_like,
        "high_frequency_fixed_sentences": frequent_sentences[:30],
        "max_fixed_sentence_share": max_fixed_sentence_share,
        "token_capability_association_candidates": strong_associations[:100],
        "manual_review_required": {
            "human_scores_are_unfilled": True,
            "association_candidates_require_review": bool(strong_associations),
            "human_surface_audit_passed": False,
        },
        "limitations": [
            "No quality, cost, latency, winner, Router, or baseline outcome is generated.",
            "Automatic checks prepare human review; they do not establish Human Surface Audit PASS.",
        ],
    }
