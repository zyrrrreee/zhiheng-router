"""Deterministic Pilot v2 serialization, comparison, and review packets."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from .audit_v2 import audit_pilot_v2
from .config import VERSION_FIELDS, load_benchmark_config
from .pilot import (
    FORBIDDEN_REVIEW_FIELD_FRAGMENTS,
    _atomic_write_many,
    canonical_json_bytes,
    make_review_packet,
    serialize_jsonl,
    sha256_bytes,
)
from .schema import DiagnosticSidecar, ObservableQueryRecord, TASK_IDS, validate_pilot_records
from .surface import CAPABILITY_RUBRICS
from .surface_v2 import generate_pilot_v2


SPOT_REVIEW_PATH = "outputs/router_v0_2/pilot_v2/spot_review_50.csv"
COMPARISON_PATH = "outputs/router_v0_2/pilot_v2/pilot_v1_vs_v2_comparison.json"
V1_CONFIG_PATH = "configs/router_v0_2_benchmark.json"
V1_QUERY_PATH = "data/router_v0_2/pilot/pilot_queries.jsonl"
V1_SIDECAR_PATH = "data/router_v0_2/pilot/pilot_sidecar.jsonl"
SPOT_REVIEW_FIELDS = (
    "review_id", "query_id", "query_text", "task", "minimal_capability_rubric",
    "difficulty", "language", "semantic_frame", "active_capability_count",
    "high_risk_lexical_association", "naturalness", "task_completeness",
    "constraint_consistency", "capability_cue_naturalness", "template_artifact",
    "shortcut_risk", "difficulty_plausibility", "ambiguity_unanswerable_risk",
    "reviewer_note", "reason_code",
)


def _load_observables(path: Path) -> list[ObservableQueryRecord]:
    return [ObservableQueryRecord.from_dict(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines() if line]


def _load_sidecars(path: Path) -> list[DiagnosticSidecar]:
    return [DiagnosticSidecar.from_dict(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines() if line]


def _comparison_metrics(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_count": audit["query_count"],
        "task_distribution": audit["task_distribution"],
        "language_distribution": audit["language_distribution"],
        "average_query_length": audit["query_length_summary"]["mean"],
        "content_payload_presence_rate": audit["answerability"]["valid_rate"],
        "answerability_valid_count": audit["answerability"]["valid_count"],
        "answerability_total": audit["answerability"]["total"],
        "exact_duplicate_group_count": len(audit["exact_duplicate_groups"]),
        "normalized_duplicate_group_count": len(audit["normalized_duplicate_groups"]),
        "max_fixed_sentence_share": audit["max_fixed_sentence_share"],
        "r1_normalized_sentence_share": audit["max_normalized_sentence_share"],
        "top_repeated_phrases": audit["repeated_multiword_phrase_frequency"][:10],
        "token_capability_association_candidate_count": audit["token_capability_candidate_count"],
        "likely_generator_artifact_count_among_top_candidates": audit["likely_generator_artifact_count"],
        "surface_lint_failure_count": audit["surface_lint"]["failure_count"],
        "semantic_frame_count": len(audit["semantic_frame_coverage"]),
        "primary_capability_distribution": audit["primary_capability_distribution"],
        "secondary_capability_distribution": audit["secondary_capability_distribution"],
    }


def compare_v1_v2(root: Path, v2_config: dict[str, Any],
                  v2_observables: list[ObservableQueryRecord],
                  v2_sidecars: list[DiagnosticSidecar],
                  v2_audit: dict[str, Any]) -> dict[str, Any]:
    v1_config = load_benchmark_config(root / V1_CONFIG_PATH)
    v1_observables = _load_observables(root / V1_QUERY_PATH)
    v1_sidecars = _load_sidecars(root / V1_SIDECAR_PATH)
    validate_pilot_records(v1_observables, v1_sidecars)
    v1_audit = audit_pilot_v2(v1_config, v1_observables, v1_sidecars)
    v1_metrics = _comparison_metrics(v1_audit)
    v2_metrics = _comparison_metrics(v2_audit)
    return {
        "comparison": "router-v0.2-pilot-v1_vs_router-v0.2-pilot-v2",
        "same_master_seed": v1_config["pilot"]["master_seed"] == v2_config["pilot"]["master_seed"],
        "v1_generator_version": v1_config["generator_version"],
        "v2_generator_version": v2_config["generator_version"],
        "v1": v1_metrics,
        "v2": v2_metrics,
        "delta_v2_minus_v1": {
            "average_query_length": round(
                v2_metrics["average_query_length"] - v1_metrics["average_query_length"], 3),
            "content_payload_presence_rate": round(
                v2_metrics["content_payload_presence_rate"]
                - v1_metrics["content_payload_presence_rate"], 6),
            "answerability_valid_count": (
                v2_metrics["answerability_valid_count"] - v1_metrics["answerability_valid_count"]),
            "max_fixed_sentence_share": round(
                v2_metrics["max_fixed_sentence_share"]
                - v1_metrics["max_fixed_sentence_share"], 6),
            "r1_normalized_sentence_share": round(
                v2_metrics["r1_normalized_sentence_share"]
                - v1_metrics["r1_normalized_sentence_share"], 6),
            "likely_generator_artifact_count_among_top_candidates": (
                v2_metrics["likely_generator_artifact_count_among_top_candidates"]
                - v1_metrics["likely_generator_artifact_count_among_top_candidates"]),
            "surface_lint_failure_count": (
                v2_metrics["surface_lint_failure_count"]
                - v1_metrics["surface_lint_failure_count"]),
        },
        "interpretation_boundary": (
            "These automatic diagnostics support Spot Review selection; they do not establish "
            "Human Surface Audit PASS."
        ),
    }


def _high_risk_tokens(audit: dict[str, Any]) -> tuple[str, ...]:
    candidates = audit["token_capability_association_candidates"]
    artifacts = [row["token"] for row in candidates
                 if row["classification"] == "likely_generator_artifact"]
    uncertain = [row["token"] for row in candidates
                 if row["classification"] == "uncertain"]
    return tuple((artifacts + uncertain)[:30])


def select_spot_review_ids(observables: list[ObservableQueryRecord],
                           sidecars: list[DiagnosticSidecar],
                           audit: dict[str, Any]) -> list[str]:
    by_id = {record.query_id: record for record in observables}
    risk_tokens = _high_risk_tokens(audit)
    selected: list[str] = []
    for task in TASK_IDS:
        candidates = [record for record in sidecars if record.coarse_task == task]
        task_selected: list[DiagnosticSidecar] = []
        covered_capabilities: set[str] = set()
        covered_difficulties: set[str] = set()
        covered_languages: set[str] = set()
        covered_counts: set[int] = set()
        covered_frames: set[str] = set()
        def score(record: DiagnosticSidecar) -> tuple[int, int, str]:
            observable = by_id[record.query_id]
            risk = sum(token in observable.query_text.lower() for token in risk_tokens)
            gain = (
                4 * len(set(record.active_capabilities) - covered_capabilities)
                + 3 * (record.difficulty_bin not in covered_difficulties)
                + 3 * (observable.language not in covered_languages)
                + 2 * (len(record.active_capabilities) not in covered_counts)
                + 4 * (record.semantic_frame_id not in covered_frames)
                + min(risk, 4)
            )
            return gain, risk, record.query_id

        def take(pool: list[DiagnosticSidecar]) -> None:
            chosen = max(pool, key=score)
            selected.append(chosen.query_id)
            task_selected.append(chosen)
            covered_capabilities.update(chosen.active_capabilities)
            covered_difficulties.add(chosen.difficulty_bin)
            covered_languages.add(by_id[chosen.query_id].language)
            covered_counts.add(len(chosen.active_capabilities))
            covered_frames.add(chosen.semantic_frame_id)

        for language in ("zh", "mixed", "en"):
            pool = [record for record in candidates
                    if by_id[record.query_id].language == language
                    and record.query_id not in selected]
            if pool:
                take(pool)
        for difficulty in ("low", "medium", "high"):
            if difficulty not in covered_difficulties:
                take([record for record in candidates
                      if record.difficulty_bin == difficulty and record.query_id not in selected])
        for frame_id in sorted({record.semantic_frame_id for record in candidates}):
            if frame_id not in covered_frames:
                take([record for record in candidates
                      if record.semantic_frame_id == frame_id and record.query_id not in selected])
        while len(task_selected) < 10:
            take([record for record in candidates if record.query_id not in selected])
    return selected


def make_spot_review_packet(observables: list[ObservableQueryRecord],
                            sidecars: list[DiagnosticSidecar],
                            audit: dict[str, Any]) -> bytes:
    by_id = {record.query_id: record for record in observables}
    sidecar_by_id = {record.query_id: record for record in sidecars}
    selected = select_spot_review_ids(observables, sidecars, audit)
    risk_tokens = _high_risk_tokens(audit)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=SPOT_REVIEW_FIELDS, lineterminator="\n")
    writer.writeheader()
    for query_id in selected:
        observable = by_id[query_id]
        sidecar = sidecar_by_id[query_id]
        matching = sorted(token for token in risk_tokens if token in observable.query_text.lower())
        row = {
            "review_id": f"spot-{query_id}",
            "query_id": query_id,
            "query_text": observable.query_text,
            "task": sidecar.coarse_task,
            "minimal_capability_rubric": "；".join(
                CAPABILITY_RUBRICS[capability] for capability in sidecar.active_capabilities),
            "difficulty": sidecar.difficulty_bin,
            "language": observable.language,
            "semantic_frame": sidecar.semantic_frame_id,
            "active_capability_count": len(sidecar.active_capabilities),
            "high_risk_lexical_association": " | ".join(matching[:5]),
            "naturalness": "",
            "task_completeness": "",
            "constraint_consistency": "",
            "capability_cue_naturalness": "",
            "template_artifact": "",
            "shortcut_risk": "",
            "difficulty_plausibility": "",
            "ambiguity_unanswerable_risk": "",
            "reviewer_note": "",
            "reason_code": "",
        }
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8-sig")


def build_pilot_v2_artifacts(config_path: Path, root: Path
                            ) -> tuple[dict[str, bytes], dict[str, Any]]:
    config = load_benchmark_config(config_path)
    if config["generator_version"] != "2.1.0" or config["pilot"]["pilot_version"] != "router-v0.2-pilot-v2":
        raise ValueError("Pilot v2 requires generator 2.1.0 and router-v0.2-pilot-v2")
    observables, sidecars = generate_pilot_v2(config)
    audit = audit_pilot_v2(config, observables, sidecars)
    comparison = compare_v1_v2(root, config, observables, sidecars, audit)
    paths = config["pilot"]
    content_by_path = {
        paths["query_data_path"]: serialize_jsonl(observables),
        paths["sidecar_data_path"]: serialize_jsonl(sidecars),
        paths["audit_path"]: canonical_json_bytes(audit),
        paths["review_packet_path"]: make_review_packet(observables, sidecars),
        SPOT_REVIEW_PATH: make_spot_review_packet(observables, sidecars, audit),
        COMPARISON_PATH: canonical_json_bytes(comparison),
    }
    manifest = {
        "stage": "pilot_surface_revision_2",
        "pilot_version": paths["pilot_version"],
        "generator_version": config["generator_version"],
        "dataset_version": config["dataset_version"],
        "source_kind": config["source_kind"],
        "master_seed": paths["master_seed"],
        "namespaces": paths["namespaces"],
        "query_count": len(observables),
        "task_query_counts": audit["task_distribution"],
        "versions": {field: config[field] for field in VERSION_FIELDS},
        "v1_preserved": True,
        "split_assignment": "none_pilot_only",
        "router_training_allowed": False,
        "outcomes_included": False,
        "human_scores_completed": False,
        "automatic_audit_ready_for_spot_review": audit["ready_for_spot_review"],
        "config_sha256": sha256_bytes(config_path.read_bytes()),
        "v1_evidence_sha256": {
            path: sha256_bytes((root / path).read_bytes())
            for path in (V1_CONFIG_PATH, V1_QUERY_PATH, V1_SIDECAR_PATH)
        },
        "files": {
            path: {"sha256": sha256_bytes(content), "bytes": len(content)}
            for path, content in sorted(content_by_path.items())
        },
    }
    content_by_path[paths["manifest_path"]] = canonical_json_bytes(manifest)
    return content_by_path, manifest


def write_pilot_v2_artifacts(config_path: Path, root: Path) -> dict[str, Any]:
    content_by_path, manifest = build_pilot_v2_artifacts(config_path, root)
    _atomic_write_many({root / path: content for path, content in content_by_path.items()})
    return manifest


def review_packet_has_forbidden_fields(fieldnames: tuple[str, ...]) -> bool:
    return any(any(fragment in field.lower() for fragment in FORBIDDEN_REVIEW_FIELD_FRAGMENTS)
               for field in fieldnames)
