from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re

import pytest

from zhiheng_router.benchmark_v02.audit import audit_pilot
from zhiheng_router.benchmark_v02.config import load_benchmark_config, validate_benchmark_config
from zhiheng_router.benchmark_v02.pilot import (
    FORBIDDEN_REVIEW_FIELD_FRAGMENTS,
    REVIEW_FIELDS,
    build_pilot_artifacts,
    write_pilot_artifacts,
)
from zhiheng_router.benchmark_v02.rng import NamespaceRNG
from zhiheng_router.benchmark_v02.schema import (
    CAPABILITY_IDS,
    DiagnosticSidecar,
    ObservableQueryRecord,
    TASK_IDS,
    validate_pilot_records,
)
from zhiheng_router.benchmark_v02.semantic_frames import FRAMES_BY_TASK, SEMANTIC_FRAMES
from zhiheng_router.benchmark_v02.surface import (
    compose_semantics,
    generate_pilot,
    pilot_stable_ids,
    realize_query,
    validate_capability_combination,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/router_v0_2_benchmark.json"
SCORE_FIELDS = (
    "naturalness",
    "task_completeness",
    "constraint_consistency",
    "capability_cue_naturalness",
    "template_artifact",
    "shortcut_risk",
    "difficulty_plausibility",
    "ambiguity_unanswerable_risk",
)


@pytest.fixture(scope="module")
def config() -> dict:
    return load_benchmark_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def pilot(config: dict) -> tuple[list[ObservableQueryRecord], list[DiagnosticSidecar]]:
    return generate_pilot(config)


@pytest.fixture
def sample_dicts(pilot: tuple[list[ObservableQueryRecord], list[DiagnosticSidecar]]) -> tuple[dict, dict]:
    return pilot[0][0].to_dict(), pilot[1][0].to_dict()


def _sidecar_by_id(sidecars: list[DiagnosticSidecar]) -> dict[str, DiagnosticSidecar]:
    return {record.query_id: record for record in sidecars}


def test_canonical_config_loads_with_frozen_identity(config: dict) -> None:
    assert config["dataset_version"] == "router-benchmark-v0.2.0"
    assert config["source_kind"] == "synthetic"
    assert config["quality_thresholds"] == {"primary": 0.8, "sensitivity": [0.7, 0.9]}
    assert [model["model_id"] for model in config["models"]] == [
        f"dev-model-{letter}" for letter in "abcdef"
    ]


def test_config_rejects_nonfinite_number(config: dict) -> None:
    invalid = deepcopy(config)
    invalid["models"][0]["capability_profile"]["AR"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        validate_benchmark_config(invalid)


def test_config_loader_rejects_nonfinite_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        load_benchmark_config(path)


def test_valid_observable_round_trip(sample_dicts: tuple[dict, dict]) -> None:
    data, _ = sample_dicts
    assert ObservableQueryRecord.from_dict(data).to_dict() == data


def test_valid_sidecar_round_trip(sample_dicts: tuple[dict, dict]) -> None:
    _, data = sample_dicts
    assert DiagnosticSidecar.from_dict(data).to_dict() == data


def test_hidden_field_is_rejected_by_observable_schema(sample_dicts: tuple[dict, dict]) -> None:
    data, _ = sample_dicts
    data["coarse_task"] = "code"
    with pytest.raises(ValueError, match="extra"):
        ObservableQueryRecord.from_dict(data)


def test_sidecar_rejects_unknown_capability(sample_dicts: tuple[dict, dict]) -> None:
    _, data = sample_dicts
    data["primary_capability"] = "UNKNOWN"
    with pytest.raises(ValueError, match="capability"):
        DiagnosticSidecar.from_dict(data)


@pytest.mark.parametrize("difficulty", [-0.01, 1.01, float("nan"), float("inf")])
def test_sidecar_rejects_invalid_difficulty(sample_dicts: tuple[dict, dict], difficulty: float) -> None:
    _, data = sample_dicts
    data["difficulty"] = difficulty
    with pytest.raises(ValueError, match="difficulty"):
        DiagnosticSidecar.from_dict(data)


def test_observable_rejects_empty_query(sample_dicts: tuple[dict, dict]) -> None:
    data, _ = sample_dicts
    data["query_text"] = "  "
    with pytest.raises(ValueError, match="query_text"):
        ObservableQueryRecord.from_dict(data)


def test_observable_rejects_invalid_language(sample_dicts: tuple[dict, dict]) -> None:
    data, _ = sample_dicts
    data["language"] = "fr"
    with pytest.raises(ValueError, match="language"):
        ObservableQueryRecord.from_dict(data)


def test_sidecar_rejects_malformed_family_id(sample_dicts: tuple[dict, dict]) -> None:
    _, data = sample_dicts
    data["source_family_id"] = "source family 1"
    with pytest.raises(ValueError, match="source_family_id"):
        DiagnosticSidecar.from_dict(data)


def test_record_sets_reject_duplicate_ids(pilot: tuple[list[ObservableQueryRecord], list[DiagnosticSidecar]]) -> None:
    observable, sidecar = pilot[0][0], pilot[1][0]
    with pytest.raises(ValueError, match="duplicate query_id"):
        validate_pilot_records([observable, observable], [sidecar, sidecar])


def test_record_sets_require_one_to_one_ids(pilot: tuple[list[ObservableQueryRecord], list[DiagnosticSidecar]]) -> None:
    with pytest.raises(ValueError, match="one-to-one"):
        validate_pilot_records(pilot[0][:2], pilot[1][:1])


def test_namespace_rng_is_deterministic(config: dict) -> None:
    rng = NamespaceRNG(config["pilot"]["master_seed"], config["pilot"]["namespaces"])
    assert rng.derive_seed("content", "code-001") == rng.derive_seed("content", "code-001")
    assert rng.random("content", "code-001").random() == rng.random("content", "code-001").random()


def test_rng_namespaces_are_isolated(config: dict) -> None:
    rng = NamespaceRNG(config["pilot"]["master_seed"], config["pilot"]["namespaces"])
    seeds = {rng.derive_seed(namespace, "code-001") for namespace in config["pilot"]["namespaces"]}
    assert len(seeds) == len(config["pilot"]["namespaces"])


def test_surface_namespace_change_preserves_capability_assignment(config: dict) -> None:
    changed = deepcopy(config)
    changed["pilot"]["namespaces"]["surface"] += "/changed"
    original = generate_pilot(config, ["code-017"])[1][0]
    modified = generate_pilot(changed, ["code-017"])[1][0]
    assert original.primary_capability == modified.primary_capability
    assert original.secondary_capabilities == modified.secondary_capabilities
    assert original.capability_weights == modified.capability_weights
    assert original.capability_requirement_levels == modified.capability_requirement_levels
    assert original.difficulty == modified.difficulty


def test_quality_noise_namespace_change_preserves_pilot_text(config: dict) -> None:
    changed = deepcopy(config)
    changed["pilot"]["namespaces"]["quality_noise"] += "/changed"
    original = [record.query_text for record in generate_pilot(config)[0]]
    modified = [record.query_text for record in generate_pilot(changed)[0]]
    assert modified == original


def test_generation_order_does_not_change_stable_id_output(config: dict) -> None:
    stable_ids = ["summary-011", "code-037", "translation-004", "math-022"]
    forward = generate_pilot(config, stable_ids)
    reverse = generate_pilot(config, list(reversed(stable_ids)))
    assert [record.to_dict() for record in forward[0]] == [record.to_dict() for record in reverse[0]]
    assert [record.to_dict() for record in forward[1]] == [record.to_dict() for record in reverse[1]]


@pytest.mark.parametrize("task", TASK_IDS)
def test_each_task_generates_a_query(config: dict, task: str) -> None:
    observable, sidecar = generate_pilot(config, [f"{task}-000"])
    assert observable[0].query_text
    assert sidecar[0].coarse_task == task


def test_every_query_has_two_to_four_capabilities(pilot: tuple[list[ObservableQueryRecord], list[DiagnosticSidecar]]) -> None:
    assert all(2 <= len(record.active_capabilities) <= 4 for record in pilot[1])


def test_primary_capabilities_are_frame_compatible(pilot: tuple[list[ObservableQueryRecord], list[DiagnosticSidecar]]) -> None:
    frames = {frame.frame_id: frame for frame in SEMANTIC_FRAMES}
    assert all(record.primary_capability in frames[record.semantic_frame_id].compatible_capabilities
               for record in pilot[1])


def test_language_and_translation_policy_is_within_pilot_tolerance(
    config: dict, pilot: tuple[list[ObservableQueryRecord], list[DiagnosticSidecar]],
) -> None:
    audit = audit_pilot(config, *pilot)
    assert audit["checks"]["language_policy"] is True
    assert audit["max_language_policy_deviation"] <= 0.06
    assert audit["translation_direction_distribution"] == {
        "bilingual_revision": 10, "en_to_zh": 20, "zh_to_en": 20,
    }


def test_incompatible_capability_combination_is_rejected() -> None:
    frame = FRAMES_BY_TASK["code"][0]
    outside = next(capability for capability in CAPABILITY_IDS
                   if capability not in frame.compatible_capabilities)
    with pytest.raises(ValueError, match="incompatible"):
        validate_capability_combination(frame, frame.compatible_capabilities[0], (outside,))


def test_query_surfaces_exclude_labels_models_and_winners(
    config: dict, pilot: tuple[list[ObservableQueryRecord], list[DiagnosticSidecar]],
) -> None:
    audit = audit_pilot(config, *pilot)
    assert audit["checks"]["no_label_like_token"] is True
    assert audit["checks"]["no_model_or_result_text"] is True


def test_family_ids_are_traceable(pilot: tuple[list[ObservableQueryRecord], list[DiagnosticSidecar]]) -> None:
    assert all(re.fullmatch(r"src-[a-z0-9-]+", record.source_family_id) for record in pilot[1])
    assert all(re.fullmatch(r"tpl-[a-z0-9-]+", record.template_family_id) for record in pilot[1])
    assert all(re.fullmatch(r"para-[a-z0-9-]+", record.paraphrase_family_id) for record in pilot[1])


def test_all_forty_semantic_frames_are_covered(
    config: dict, pilot: tuple[list[ObservableQueryRecord], list[DiagnosticSidecar]],
) -> None:
    assert len(SEMANTIC_FRAMES) == 40
    assert all(len(FRAMES_BY_TASK[task]) == 8 for task in TASK_IDS)
    assert audit_pilot(config, *pilot)["checks"]["semantic_frame_coverage"] is True


def test_observable_output_contains_no_sidecar_fields(
    pilot: tuple[list[ObservableQueryRecord], list[DiagnosticSidecar]],
) -> None:
    hidden = set(DiagnosticSidecar.__dataclass_fields__) - {"query_id"}
    assert all(not hidden.intersection(record.to_dict()) for record in pilot[0])


def test_pilot_has_exactly_250_balanced_queries(
    pilot: tuple[list[ObservableQueryRecord], list[DiagnosticSidecar]],
) -> None:
    assert len(pilot[0]) == len(pilot[1]) == 250
    counts = {task: sum(record.coarse_task == task for record in pilot[1]) for task in TASK_IDS}
    assert counts == {task: 50 for task in TASK_IDS}


def test_pilot_build_is_byte_reproducible() -> None:
    first, first_manifest = build_pilot_artifacts(CONFIG_PATH, ROOT)
    second, second_manifest = build_pilot_artifacts(CONFIG_PATH, ROOT)
    assert first == second
    assert first_manifest == second_manifest


def test_observable_and_sidecar_artifacts_match_one_to_one() -> None:
    artifacts, _ = build_pilot_artifacts(CONFIG_PATH, ROOT)
    observables = [json.loads(line) for line in artifacts["data/router_v0_2/pilot/pilot_queries.jsonl"].splitlines()]
    sidecars = [json.loads(line) for line in artifacts["data/router_v0_2/pilot/pilot_sidecar.jsonl"].splitlines()]
    assert {row["query_id"] for row in observables} == {row["query_id"] for row in sidecars}
    assert len(observables) == len(sidecars) == 250


def test_review_packet_uses_allowlist_and_has_blank_human_scores() -> None:
    artifacts, _ = build_pilot_artifacts(CONFIG_PATH, ROOT)
    text = artifacts["outputs/router_v0_2/pilot/human_surface_review.csv"].decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    assert tuple(rows[0]) == REVIEW_FIELDS
    assert len(rows) == 250
    assert all(not any(fragment in field.lower() for fragment in FORBIDDEN_REVIEW_FIELD_FRAGMENTS)
               for field in rows[0])
    assert all(all(row[field] == "" for field in SCORE_FIELDS) for row in rows)
    assert not any(token in text.lower() for token in (
        "dev-model-", "quality-best", "efficient winner", "sampled outcome",
        "router result", "model profile",
    ))


def test_manifest_hashes_every_non_manifest_artifact() -> None:
    artifacts, manifest = build_pilot_artifacts(CONFIG_PATH, ROOT)
    for relative_path, metadata in manifest["files"].items():
        content = artifacts[relative_path]
        assert metadata == {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    assert manifest["config_sha256"] == hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()


def test_pilot_manifest_declares_no_outcomes_or_training() -> None:
    _, manifest = build_pilot_artifacts(CONFIG_PATH, ROOT)
    assert manifest["stage"] == "pilot_surface_only"
    assert manifest["outcomes_included"] is False
    assert manifest["router_training_allowed"] is False
    assert manifest["split_assignment"] == "none_pilot_only"
    assert manifest["human_scores_completed"] is False


def test_atomic_writer_round_trips_all_artifacts(tmp_path: Path) -> None:
    expected, expected_manifest = build_pilot_artifacts(CONFIG_PATH, tmp_path)
    actual_manifest = write_pilot_artifacts(CONFIG_PATH, tmp_path)
    assert actual_manifest == expected_manifest
    assert all((tmp_path / path).read_bytes() == content for path, content in expected.items())


def test_pilot_automatic_audit_is_ready_but_human_audit_is_unfilled(
    config: dict, pilot: tuple[list[ObservableQueryRecord], list[DiagnosticSidecar]],
) -> None:
    audit = audit_pilot(config, *pilot)
    assert audit["ready_for_human_review"] is True
    assert all(audit["checks"].values())
    assert audit["manual_review_required"]["human_scores_are_unfilled"] is True
    assert audit["manual_review_required"]["human_surface_audit_passed"] is False
