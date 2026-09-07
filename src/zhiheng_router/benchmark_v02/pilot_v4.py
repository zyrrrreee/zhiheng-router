"""Pilot v4 serialization, compact comparison, and risk-focused Spot Review."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from .audit_v3 import semantic_skeleton
from .audit_v4 import audit_pilot_v4
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
from .surface_v4 import generate_pilot_v4


SPOT_REVIEW_PATH = "outputs/router_v0_2/pilot_v4/spot_review_50.csv"
COMPARISON_PATH = "outputs/router_v0_2/pilot_v4/pilot_v3_vs_v4_comparison.json"
V3_CONFIG_PATH = "configs/router_v0_2_pilot_v3.json"
V3_QUERY_PATH = "data/router_v0_2/pilot_v3/pilot_queries.jsonl"
V3_SIDECAR_PATH = "data/router_v0_2/pilot_v3/pilot_sidecar.jsonl"
SPOT_REVIEW_FIELDS = (
    "review_id", "query_id", "query_text", "task", "minimal_capability_rubric",
    "difficulty", "difficulty_value", "language", "semantic_frame",
    "active_capability_count", "semantic_sibling_risk_group_size", "guardrail_risk",
    "difficulty_boundary_distance", "naturalness", "task_completeness",
    "constraint_consistency", "capability_cue_naturalness", "template_artifact",
    "shortcut_risk", "difficulty_plausibility", "ambiguity_unanswerable_risk",
    "reviewer_note", "reason_code",
)
HUMAN_SCORE_FIELDS = (
    "naturalness", "task_completeness", "constraint_consistency",
    "capability_cue_naturalness", "template_artifact", "shortcut_risk",
    "difficulty_plausibility", "ambiguity_unanswerable_risk",
    "reviewer_note", "reason_code",
)


def _load_observables(path: Path) -> list[ObservableQueryRecord]:
    return [ObservableQueryRecord.from_dict(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines() if line]


def _load_sidecars(path: Path) -> list[DiagnosticSidecar]:
    return [DiagnosticSidecar.from_dict(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines() if line]


def _comparison_metrics(audit: dict[str, Any]) -> dict[str, Any]:
    roles = audit["capability_role_validation"]
    completeness = audit["frame_semantic_completeness"]
    return {
        "query_count": audit["query_count"],
        "answerability_valid_count": audit["answerability"]["valid_count"],
        "capability_alignment_valid_rate": audit[
            "capability_evidence_alignment"]["alignment_valid_rate"],
        "capability_necessity_valid_rate": roles["capability_necessity_valid_rate"],
        "capability_coverage_valid_rate": roles["capability_coverage_valid_rate"],
        "invalid_active_capability_count": roles["invalid_active_capability_count"],
        "missing_required_capability_count": roles["missing_required_capability_count"],
        "semantic_sibling_duplicate_group_count": audit[
            "semantic_sibling_duplicates"]["duplicate_group_count"],
        "minimum_frame_skeleton_count": audit[
            "semantic_skeleton_diversity"]["minimum_distinct_skeleton_count"],
        "maximum_frame_skeleton_share": audit[
            "semantic_skeleton_diversity"]["maximum_largest_skeleton_share"],
        "pearson_difficulty_capability_count": audit[
            "difficulty_capability_diagnostics"]["pearson_correlation"],
        "spearman_difficulty_capability_count": audit[
            "difficulty_capability_diagnostics"]["spearman_correlation"],
        "ambiguous_translation_target_count": completeness[
            "ambiguous_translation_target_count"],
        "incomplete_timezone_conversion_count": completeness[
            "incomplete_timezone_conversion_count"],
        "task_deliverable_mismatch_count": completeness[
            "task_deliverable_mismatch_count"],
        "severe_sibling_difficulty_conflict_count": audit[
            "sibling_difficulty_consistency"]["severe_sibling_difficulty_conflict_count"],
        "exact_duplicate_group_count": len(audit["exact_duplicate_groups"]),
        "normalized_duplicate_group_count": len(audit["normalized_duplicate_groups"]),
        "surface_lint_failure_count": audit["surface_lint"]["failure_count"],
    }


def compare_v3_v4(root: Path, v4_config: dict[str, Any],
                  v4_observables: list[ObservableQueryRecord],
                  v4_sidecars: list[DiagnosticSidecar],
                  v4_audit: dict[str, Any]) -> dict[str, Any]:
    v3_config = load_benchmark_config(root / V3_CONFIG_PATH)
    v3_observables = _load_observables(root / V3_QUERY_PATH)
    v3_sidecars = _load_sidecars(root / V3_SIDECAR_PATH)
    validate_pilot_records(v3_observables, v3_sidecars)
    v3_audit = audit_pilot_v4(v3_config, v3_observables, v3_sidecars)
    return {
        "comparison": "router-v0.2-pilot-v3_vs_router-v0.2-pilot-v4",
        "same_master_seed": v3_config["pilot"]["master_seed"] == v4_config["pilot"]["master_seed"],
        "v3_generator_version": v3_config["generator_version"],
        "v4_generator_version": v4_config["generator_version"],
        "v3": _comparison_metrics(v3_audit),
        "v4": _comparison_metrics(v4_audit),
        "interpretation_boundary": (
            "Automatic guardrails prepare final Human Spot Review; they do not establish "
            "Human Review PASS."
        ),
    }


def _loose_sibling_sizes(observables: list[ObservableQueryRecord],
                         sidecars: list[DiagnosticSidecar]) -> dict[str, int]:
    observable_by_id = {record.query_id: record for record in observables}
    groups: dict[tuple[str, str], list[str]] = {}
    for sidecar in sidecars:
        key = (sidecar.semantic_frame_id,
               semantic_skeleton(observable_by_id[sidecar.query_id].query_text))
        groups.setdefault(key, []).append(sidecar.query_id)
    return {query_id: len(query_ids) for query_ids in groups.values() for query_id in query_ids}


def select_spot_review_ids(observables: list[ObservableQueryRecord],
                           sidecars: list[DiagnosticSidecar]) -> list[str]:
    observable_by_id = {record.query_id: record for record in observables}
    loose_sizes = _loose_sibling_sizes(observables, sidecars)
    selected: list[str] = []
    completeness_frames = {
        "frame-code-api-client", "frame-code-data-pipeline",
    }
    for task in TASK_IDS:
        candidates = [record for record in sidecars if record.coarse_task == task]
        task_selected: list[DiagnosticSidecar] = []
        frames: set[str] = set()
        languages: set[str] = set()
        difficulties: set[str] = set()
        counts: set[int] = set()

        def score(record: DiagnosticSidecar) -> tuple[float, str]:
            observable = observable_by_id[record.query_id]
            boundary = min(abs(record.difficulty - 0.4), abs(record.difficulty - 0.7))
            guardrail_risk = (
                2 * (record.coarse_task == "translation")
                + 2 * (record.semantic_frame_id in completeness_frames)
                + sum(capability in {"AM", "MH", "AU", "DE"}
                      for capability in record.active_capabilities)
            )
            gain = (
                7 * (record.semantic_frame_id not in frames)
                + 4 * (observable.language not in languages)
                + 4 * (record.difficulty_bin not in difficulties)
                + 4 * (len(record.active_capabilities) not in counts)
                + loose_sizes[record.query_id] + guardrail_risk - boundary
            )
            return gain, record.query_id

        def take(pool: list[DiagnosticSidecar]) -> None:
            chosen = max(pool, key=score)
            selected.append(chosen.query_id)
            task_selected.append(chosen)
            frames.add(chosen.semantic_frame_id)
            languages.add(observable_by_id[chosen.query_id].language)
            difficulties.add(chosen.difficulty_bin)
            counts.add(len(chosen.active_capabilities))

        for language in ("zh", "mixed", "en"):
            pool = [record for record in candidates
                    if observable_by_id[record.query_id].language == language
                    and record.query_id not in selected]
            if pool:
                take(pool)
        for difficulty in ("low", "medium", "high"):
            if difficulty not in difficulties:
                take([record for record in candidates
                      if record.difficulty_bin == difficulty and record.query_id not in selected])
        for count in (2, 3, 4):
            if count not in counts:
                pool = [record for record in candidates
                        if len(record.active_capabilities) == count
                        and record.query_id not in selected]
                if pool:
                    take(pool)
        for frame_id in sorted({record.semantic_frame_id for record in candidates}):
            if frame_id not in frames:
                take([record for record in candidates
                      if record.semantic_frame_id == frame_id and record.query_id not in selected])
        while len(task_selected) < 10:
            take([record for record in candidates if record.query_id not in selected])
        if len(task_selected) != 10:
            raise ValueError(f"spot review selection for {task} exceeded 10 rows")
    return selected


def make_spot_review_packet(observables: list[ObservableQueryRecord],
                            sidecars: list[DiagnosticSidecar]) -> bytes:
    observable_by_id = {record.query_id: record for record in observables}
    sidecar_by_id = {record.query_id: record for record in sidecars}
    loose_sizes = _loose_sibling_sizes(observables, sidecars)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=SPOT_REVIEW_FIELDS, lineterminator="\n")
    writer.writeheader()
    for query_id in select_spot_review_ids(observables, sidecars):
        observable = observable_by_id[query_id]
        sidecar = sidecar_by_id[query_id]
        risk = []
        if sidecar.coarse_task == "translation":
            risk.append("translation_capability")
        if sidecar.semantic_frame_id in {"frame-code-api-client", "frame-code-data-pipeline"}:
            risk.append("task_completeness")
        if loose_sizes[query_id] > 1:
            risk.append("semantic_sibling")
        writer.writerow({
            "review_id": f"spot-{query_id}",
            "query_id": query_id,
            "query_text": observable.query_text,
            "task": sidecar.coarse_task,
            "minimal_capability_rubric": "；".join(
                CAPABILITY_RUBRICS[capability] for capability in sidecar.active_capabilities),
            "difficulty": sidecar.difficulty_bin,
            "difficulty_value": sidecar.difficulty,
            "language": observable.language,
            "semantic_frame": sidecar.semantic_frame_id,
            "active_capability_count": len(sidecar.active_capabilities),
            "semantic_sibling_risk_group_size": loose_sizes[query_id],
            "guardrail_risk": " | ".join(risk),
            "difficulty_boundary_distance": round(
                min(abs(sidecar.difficulty - 0.4), abs(sidecar.difficulty - 0.7)), 6),
            **{field: "" for field in HUMAN_SCORE_FIELDS},
        })
    return buffer.getvalue().encode("utf-8-sig")


def build_pilot_v4_artifacts(config_path: Path, root: Path
                            ) -> tuple[dict[str, bytes], dict[str, Any]]:
    config = load_benchmark_config(config_path)
    if config["generator_version"] != "2.3.0":
        raise ValueError("Pilot v4 requires generator 2.3.0")
    if config["pilot"]["pilot_version"] != "router-v0.2-pilot-v4":
        raise ValueError("Pilot v4 requires router-v0.2-pilot-v4")
    observables, sidecars = generate_pilot_v4(config)
    audit = audit_pilot_v4(config, observables, sidecars)
    if not audit["ready_for_final_human_spot_review"]:
        failed = sorted(key for key, passed in audit["checks"].items() if not passed)
        raise ValueError(f"Pilot v4 audit failed: {failed}")
    comparison = compare_v3_v4(root, config, observables, sidecars, audit)
    paths = config["pilot"]
    content_by_path = {
        paths["query_data_path"]: serialize_jsonl(observables),
        paths["sidecar_data_path"]: serialize_jsonl(sidecars),
        paths["audit_path"]: canonical_json_bytes(audit),
        paths["review_packet_path"]: make_review_packet(observables, sidecars),
        SPOT_REVIEW_PATH: make_spot_review_packet(observables, sidecars),
        COMPARISON_PATH: canonical_json_bytes(comparison),
    }
    protected_v3_paths = (
        V3_CONFIG_PATH, V3_QUERY_PATH, V3_SIDECAR_PATH,
        "data/router_v0_2/pilot_v3/pilot_manifest.json",
        "outputs/router_v0_2/pilot_v3/pilot_audit.json",
        "outputs/router_v0_2/pilot_v3/human_surface_review.csv",
        "outputs/router_v0_2/pilot_v3/spot_review_50.csv",
        "outputs/router_v0_2/pilot_v3/pilot_v2_vs_v3_comparison.json",
    )
    manifest = {
        "stage": "pilot_human_review_guardrail_fix",
        "pilot_version": paths["pilot_version"],
        "generator_version": config["generator_version"],
        "dataset_version": config["dataset_version"],
        "source_kind": config["source_kind"],
        "master_seed": paths["master_seed"],
        "namespaces": paths["namespaces"],
        "query_count": len(observables),
        "task_query_counts": audit["task_distribution"],
        "versions": {field: config[field] for field in VERSION_FIELDS},
        "effective_config_path": str(config_path.relative_to(root)).replace("\\", "/"),
        "protected_prior_pilots_unchanged": True,
        "split_assignment": "none_pilot_only",
        "router_training_allowed": False,
        "outcomes_included": False,
        "human_scores_completed": False,
        "automatic_audit_ready_for_final_human_spot_review": True,
        "config_sha256": sha256_bytes(config_path.read_bytes()),
        "v3_evidence_sha256": {
            path: sha256_bytes((root / path).read_bytes()) for path in protected_v3_paths
        },
        "files": {
            path: {"sha256": sha256_bytes(content), "bytes": len(content)}
            for path, content in sorted(content_by_path.items())
        },
    }
    content_by_path[paths["manifest_path"]] = canonical_json_bytes(manifest)
    return content_by_path, manifest


def write_pilot_v4_artifacts(config_path: Path, root: Path) -> dict[str, Any]:
    content_by_path, manifest = build_pilot_v4_artifacts(config_path, root)
    _atomic_write_many({root / path: content for path, content in content_by_path.items()})
    return manifest


def review_packet_has_forbidden_fields(fieldnames: tuple[str, ...]) -> bool:
    return any(any(fragment in field.lower() for fragment in FORBIDDEN_REVIEW_FIELD_FRAGMENTS)
               for field in fieldnames)
