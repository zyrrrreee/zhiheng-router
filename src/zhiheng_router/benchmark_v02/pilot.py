"""Deterministic serialization for the Router v0.2 Pilot Surface Set."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .audit import audit_pilot
from .config import VERSION_FIELDS, load_benchmark_config
from .schema import DiagnosticSidecar, ObservableQueryRecord
from .surface import CAPABILITY_RUBRICS, generate_pilot


REVIEW_FIELDS = (
    "review_id", "query_id", "query_text", "task", "minimal_capability_rubric",
    "difficulty_rubric", "language", "translation_direction", "semantic_scenario",
    "naturalness", "task_completeness", "constraint_consistency",
    "capability_cue_naturalness", "template_artifact", "shortcut_risk",
    "difficulty_plausibility", "ambiguity_unanswerable_risk", "reviewer_note", "reason_code",
)
FORBIDDEN_REVIEW_FIELD_FRAGMENTS = (
    "model_profile", "quality", "cost", "latency", "quality_best", "efficient_winner",
    "sampled_outcome", "router_result", "capability_weight", "requirement_level",
)


def _json_line(data: dict[str, Any]) -> bytes:
    return (json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def serialize_jsonl(records: list[ObservableQueryRecord] | list[DiagnosticSidecar]) -> bytes:
    return b"".join(_json_line(record.to_dict()) for record in records)


def canonical_json_bytes(data: dict[str, Any]) -> bytes:
    return (json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            + "\n").encode("utf-8")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def make_review_packet(observables: list[ObservableQueryRecord],
                       sidecars: list[DiagnosticSidecar]) -> bytes:
    by_id = {sidecar.query_id: sidecar for sidecar in sidecars}
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=REVIEW_FIELDS, lineterminator="\n")
    writer.writeheader()
    for observable in observables:
        sidecar = by_id[observable.query_id]
        active = sidecar.active_capabilities
        writer.writerow({
            "review_id": f"review-{observable.query_id}",
            "query_id": observable.query_id,
            "query_text": observable.query_text,
            "task": sidecar.coarse_task,
            "minimal_capability_rubric": "；".join(CAPABILITY_RUBRICS[item] for item in active),
            "difficulty_rubric": (
                f"{sidecar.difficulty_bin}; active_requirements={len(active)}; "
                f"visible_constraints={observable.visible_structure.constraint_count}"
            ),
            "language": observable.language,
            "translation_direction": observable.visible_structure.translation_direction or "",
            "semantic_scenario": sidecar.semantic_scenario,
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
        })
    return buffer.getvalue().encode("utf-8-sig")


def _atomic_write_many(files: dict[Path, bytes]) -> None:
    temporary: dict[Path, Path] = {}
    originals = {path: path.read_bytes() if path.exists() else None for path in files}
    replaced: list[Path] = []
    try:
        for path, content in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
            temporary_path = Path(temporary_name)
            temporary[path] = temporary_path
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        for path in files:
            os.replace(temporary[path], path)
            replaced.append(path)
    except Exception:
        for path in reversed(replaced):
            original = originals[path]
            if original is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(original)
        raise
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def build_pilot_artifacts(config_path: Path, root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    config = load_benchmark_config(config_path)
    observables, sidecars = generate_pilot(config)
    audit = audit_pilot(config, observables, sidecars)
    query_content = serialize_jsonl(observables)
    sidecar_content = serialize_jsonl(sidecars)
    review_content = make_review_packet(observables, sidecars)
    audit_content = canonical_json_bytes(audit)
    paths = config["pilot"]
    content_by_relative_path = {
        paths["query_data_path"]: query_content,
        paths["sidecar_data_path"]: sidecar_content,
        paths["audit_path"]: audit_content,
        paths["review_packet_path"]: review_content,
    }
    manifest = {
        "stage": "pilot_surface_only",
        "pilot_version": paths["pilot_version"],
        "source_kind": config["source_kind"],
        "dataset_version": config["dataset_version"],
        "versions": {field: config[field] for field in VERSION_FIELDS},
        "master_seed": paths["master_seed"],
        "namespaces": paths["namespaces"],
        "query_count": len(observables),
        "task_query_counts": audit["task_distribution"],
        "semantic_frame_count": len(audit["semantic_frame_coverage"]),
        "split_assignment": "none_pilot_only",
        "router_training_allowed": False,
        "outcomes_included": False,
        "human_scores_completed": False,
        "automatic_audit_ready_for_human_review": audit["ready_for_human_review"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "files": {
            relative_path: {"sha256": sha256_bytes(content), "bytes": len(content)}
            for relative_path, content in sorted(content_by_relative_path.items())
        },
    }
    manifest_content = canonical_json_bytes(manifest)
    content_by_relative_path[paths["manifest_path"]] = manifest_content
    return content_by_relative_path, manifest


def write_pilot_artifacts(config_path: Path, root: Path) -> dict[str, Any]:
    content_by_relative_path, manifest = build_pilot_artifacts(config_path, root)
    files = {root / relative_path: content for relative_path, content in content_by_relative_path.items()}
    _atomic_write_many(files)
    return manifest
