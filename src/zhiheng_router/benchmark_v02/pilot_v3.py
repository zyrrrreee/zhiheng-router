"""Pilot v3 serialization, v2 comparison, and final spot-review sampling."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from .audit_v3 import audit_pilot_v3, semantic_skeleton
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
from .surface_v3 import generate_pilot_v3


SPOT_REVIEW_PATH = "outputs/router_v0_2/pilot_v3/spot_review_50.csv"
COMPARISON_PATH = "outputs/router_v0_2/pilot_v3/pilot_v2_vs_v3_comparison.json"
V2_CONFIG_PATH = "configs/router_v0_2_pilot_v2.json"
V2_QUERY_PATH = "data/router_v0_2/pilot_v2/pilot_queries.jsonl"
V2_SIDECAR_PATH = "data/router_v0_2/pilot_v2/pilot_sidecar.jsonl"
SPOT_REVIEW_FIELDS = (
    "review_id", "query_id", "query_text", "task", "minimal_capability_rubric",
    "difficulty", "difficulty_value", "language", "semantic_frame",
    "active_capability_count", "alignment_risk_capabilities", "skeleton_group_size",
    "difficulty_boundary_distance", "lexical_association", "naturalness",
    "task_completeness", "constraint_consistency", "capability_cue_naturalness",
    "template_artifact", "shortcut_risk", "difficulty_plausibility",
    "ambiguity_unanswerable_risk", "reviewer_note", "reason_code",
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
    confounding = audit["difficulty_capability_diagnostics"]
    skeletons = audit["semantic_skeleton_diversity"]
    alignment = audit["capability_evidence_alignment"]
    return {
        "query_count": audit["query_count"],
        "task_distribution": audit["task_distribution"],
        "language_distribution": audit["language_distribution"],
        "difficulty_distribution": audit["difficulty_distribution"],
        "capability_count_distribution": confounding["capability_count_distribution"],
        "difficulty_by_capability_count": confounding["difficulty_by_capability_count"],
        "pearson_difficulty_capability_count": confounding["pearson_correlation"],
        "spearman_difficulty_capability_count": confounding["spearman_correlation"],
        "answerability_valid_count": audit["answerability"]["valid_count"],
        "answerability_total": audit["answerability"]["total"],
        "payload_presence_rate": audit["answerability"]["valid_rate"],
        "surface_lint_failure_count": audit["surface_lint"]["failure_count"],
        "exact_duplicate_group_count": len(audit["exact_duplicate_groups"]),
        "normalized_duplicate_group_count": len(audit["normalized_duplicate_groups"]),
        "minimum_frame_skeleton_count": skeletons["minimum_distinct_skeleton_count"],
        "maximum_frame_skeleton_share": skeletons["maximum_largest_skeleton_share"],
        "capability_alignment_valid_rate": alignment["alignment_valid_rate"],
        "invalid_capability_evidence_count": alignment["invalid_evidence_count"],
        "max_fixed_sentence_share": audit["max_fixed_sentence_share"],
        "max_repeated_phrase_share": audit["max_repeated_phrase_share"],
        "lexical_shortcut_candidate_count": audit["token_capability_candidate_count"],
        "likely_generator_artifact_count": audit["likely_generator_artifact_count"],
    }


def compare_v2_v3(root: Path, v3_config: dict[str, Any],
                  v3_observables: list[ObservableQueryRecord],
                  v3_sidecars: list[DiagnosticSidecar],
                  v3_audit: dict[str, Any]) -> dict[str, Any]:
    v2_config = load_benchmark_config(root / V2_CONFIG_PATH)
    v2_observables = _load_observables(root / V2_QUERY_PATH)
    v2_sidecars = _load_sidecars(root / V2_SIDECAR_PATH)
    validate_pilot_records(v2_observables, v2_sidecars)
    v2_audit = audit_pilot_v3(v2_config, v2_observables, v2_sidecars)
    return {
        "comparison": "router-v0.2-pilot-v2_vs_router-v0.2-pilot-v3",
        "same_master_seed": v2_config["pilot"]["master_seed"] == v3_config["pilot"]["master_seed"],
        "v2_generator_version": v2_config["generator_version"],
        "v3_generator_version": v3_config["generator_version"],
        "v2": _comparison_metrics(v2_audit),
        "v3": _comparison_metrics(v3_audit),
        "interpretation_boundary": (
            "Deterministic diagnostics prepare final Spot Review; they do not establish "
            "Human Surface Audit PASS."
        ),
    }


def _lexical_tokens(audit: dict[str, Any]) -> tuple[str, ...]:
    rows = audit["token_capability_association_candidates"]
    return tuple(row["token"] for row in rows if row["classification"] !=
                 "likely_legitimate_semantic_cue")[:30]


def _skeleton_group_sizes(observables: list[ObservableQueryRecord],
                          sidecars: list[DiagnosticSidecar]) -> dict[str, int]:
    observable_by_id = {record.query_id: record for record in observables}
    groups: dict[tuple[str, str], list[str]] = {}
    for sidecar in sidecars:
        key = (sidecar.semantic_frame_id,
               semantic_skeleton(observable_by_id[sidecar.query_id].query_text))
        groups.setdefault(key, []).append(sidecar.query_id)
    return {query_id: len(query_ids) for query_ids in groups.values() for query_id in query_ids}


def select_spot_review_ids(observables: list[ObservableQueryRecord],
                           sidecars: list[DiagnosticSidecar],
                           audit: dict[str, Any]) -> list[str]:
    observable_by_id = {record.query_id: record for record in observables}
    skeleton_sizes = _skeleton_group_sizes(observables, sidecars)
    lexical_tokens = _lexical_tokens(audit)
    selected: list[str] = []
    for task in TASK_IDS:
        candidates = [record for record in sidecars if record.coarse_task == task]
        task_selected: list[DiagnosticSidecar] = []
        covered_capabilities: set[str] = set()
        covered_difficulties: set[str] = set()
        covered_languages: set[str] = set()
        covered_counts: set[int] = set()
        covered_frames: set[str] = set()

        def score(record: DiagnosticSidecar) -> tuple[float, str]:
            observable = observable_by_id[record.query_id]
            alignment_risk = sum(capability in {"AM", "MH", "LC"}
                                 for capability in record.active_capabilities)
            lexical_risk = sum(token in observable.query_text.lower()
                               for token in lexical_tokens)
            boundary = min(abs(record.difficulty - 0.4), abs(record.difficulty - 0.7))
            gain = (
                5 * (record.semantic_frame_id not in covered_frames)
                + 4 * len(set(record.active_capabilities) - covered_capabilities)
                + 3 * (record.difficulty_bin not in covered_difficulties)
                + 3 * (observable.language not in covered_languages)
                + 3 * (len(record.active_capabilities) not in covered_counts)
                + 2 * alignment_risk + min(lexical_risk, 3)
                + skeleton_sizes[record.query_id] - boundary
            )
            return gain, record.query_id

        def take(pool: list[DiagnosticSidecar]) -> None:
            chosen = max(pool, key=score)
            selected.append(chosen.query_id)
            task_selected.append(chosen)
            covered_capabilities.update(chosen.active_capabilities)
            covered_difficulties.add(chosen.difficulty_bin)
            covered_languages.add(observable_by_id[chosen.query_id].language)
            covered_counts.add(len(chosen.active_capabilities))
            covered_frames.add(chosen.semantic_frame_id)

        for language in ("zh", "mixed", "en"):
            pool = [record for record in candidates
                    if observable_by_id[record.query_id].language == language
                    and record.query_id not in selected]
            if pool:
                take(pool)
        for difficulty in ("low", "medium", "high"):
            if difficulty not in covered_difficulties:
                take([record for record in candidates
                      if record.difficulty_bin == difficulty and record.query_id not in selected])
        for count in (2, 3, 4):
            if count not in covered_counts:
                pool = [record for record in candidates
                        if len(record.active_capabilities) == count
                        and record.query_id not in selected]
                if pool:
                    take(pool)
        for frame_id in sorted({record.semantic_frame_id for record in candidates}):
            if frame_id not in covered_frames:
                take([record for record in candidates
                      if record.semantic_frame_id == frame_id and record.query_id not in selected])
        while len(task_selected) < 10:
            take([record for record in candidates if record.query_id not in selected])
        if len(task_selected) != 10:
            raise ValueError(f"spot review selection for {task} exceeded 10 rows")
    return selected


def make_spot_review_packet(observables: list[ObservableQueryRecord],
                            sidecars: list[DiagnosticSidecar],
                            audit: dict[str, Any]) -> bytes:
    observable_by_id = {record.query_id: record for record in observables}
    sidecar_by_id = {record.query_id: record for record in sidecars}
    skeleton_sizes = _skeleton_group_sizes(observables, sidecars)
    lexical_tokens = _lexical_tokens(audit)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=SPOT_REVIEW_FIELDS, lineterminator="\n")
    writer.writeheader()
    for query_id in select_spot_review_ids(observables, sidecars, audit):
        observable = observable_by_id[query_id]
        sidecar = sidecar_by_id[query_id]
        row = {
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
            "alignment_risk_capabilities": " | ".join(
                capability for capability in sidecar.active_capabilities
                if capability in {"AM", "MH", "LC"}),
            "skeleton_group_size": skeleton_sizes[query_id],
            "difficulty_boundary_distance": round(
                min(abs(sidecar.difficulty - 0.4), abs(sidecar.difficulty - 0.7)), 6),
            "lexical_association": " | ".join(
                token for token in lexical_tokens if token in observable.query_text.lower())[:200],
            **{field: "" for field in HUMAN_SCORE_FIELDS},
        }
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8-sig")


def build_pilot_v3_artifacts(config_path: Path, root: Path
                            ) -> tuple[dict[str, bytes], dict[str, Any]]:
    config = load_benchmark_config(config_path)
    if config["generator_version"] != "2.2.0":
        raise ValueError("Pilot v3 requires generator 2.2.0")
    if config["pilot"]["pilot_version"] != "router-v0.2-pilot-v3":
        raise ValueError("Pilot v3 requires router-v0.2-pilot-v3")
    observables, sidecars = generate_pilot_v3(config)
    audit = audit_pilot_v3(config, observables, sidecars)
    comparison = compare_v2_v3(root, config, observables, sidecars, audit)
    paths = config["pilot"]
    content_by_path = {
        paths["query_data_path"]: serialize_jsonl(observables),
        paths["sidecar_data_path"]: serialize_jsonl(sidecars),
        paths["audit_path"]: canonical_json_bytes(audit),
        paths["review_packet_path"]: make_review_packet(observables, sidecars),
        SPOT_REVIEW_PATH: make_spot_review_packet(observables, sidecars, audit),
        COMPARISON_PATH: canonical_json_bytes(comparison),
    }
    protected_v2_paths = (
        V2_CONFIG_PATH, V2_QUERY_PATH, V2_SIDECAR_PATH,
        "data/router_v0_2/pilot_v2/pilot_manifest.json",
        "outputs/router_v0_2/pilot_v2/pilot_audit.json",
        "outputs/router_v0_2/pilot_v2/human_surface_review.csv",
        "outputs/router_v0_2/pilot_v2/spot_review_50.csv",
        "outputs/router_v0_2/pilot_v2/pilot_v1_vs_v2_comparison.json",
    )
    manifest = {
        "stage": "pilot_benchmark_final_calibration",
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
        "v1_v2_preserved": True,
        "split_assignment": "none_pilot_only",
        "router_training_allowed": False,
        "outcomes_included": False,
        "human_scores_completed": False,
        "automatic_audit_ready_for_final_spot_review": audit["ready_for_final_spot_review"],
        "config_sha256": sha256_bytes(config_path.read_bytes()),
        "v2_evidence_sha256": {
            path: sha256_bytes((root / path).read_bytes()) for path in protected_v2_paths
        },
        "files": {
            path: {"sha256": sha256_bytes(content), "bytes": len(content)}
            for path, content in sorted(content_by_path.items())
        },
    }
    content_by_path[paths["manifest_path"]] = canonical_json_bytes(manifest)
    return content_by_path, manifest


def write_pilot_v3_artifacts(config_path: Path, root: Path) -> dict[str, Any]:
    content_by_path, manifest = build_pilot_v3_artifacts(config_path, root)
    _atomic_write_many({root / path: content for path, content in content_by_path.items()})
    return manifest


def review_packet_has_forbidden_fields(fieldnames: tuple[str, ...]) -> bool:
    return any(any(fragment in field.lower() for fragment in FORBIDDEN_REVIEW_FIELD_FRAGMENTS)
               for field in fieldnames)
