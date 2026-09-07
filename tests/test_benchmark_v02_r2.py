from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from zhiheng_router.benchmark_v02.audit_v3 import audit_pilot_v3, semantic_skeleton
from zhiheng_router.benchmark_v02.config import load_benchmark_config
from zhiheng_router.benchmark_v02.payloads_v3 import generate_content_payload_v3
from zhiheng_router.benchmark_v02.pilot_v3 import (
    COMPARISON_PATH,
    HUMAN_SCORE_FIELDS,
    SPOT_REVIEW_FIELDS,
    SPOT_REVIEW_PATH,
    build_pilot_v3_artifacts,
    review_packet_has_forbidden_fields,
)
from zhiheng_router.benchmark_v02.rng import NamespaceRNG
from zhiheng_router.benchmark_v02.schema import TASK_IDS
from zhiheng_router.benchmark_v02.semantics_v3 import compose_semantics_v3
from zhiheng_router.benchmark_v02.surface_v3 import generate_pilot_v3


ROOT = Path(__file__).resolve().parents[1]
V3_CONFIG_PATH = ROOT / "configs/router_v0_2_pilot_v3.json"
PROTECTED_HASHES = {
    "configs/router_v0_2_benchmark.json": "7706307ea88db1309c998d7e87f84dbfb275b28daffc890875ca581f200fd964",
    "data/router_v0_2/pilot/pilot_manifest.json": "6307eec5690726c94cbacb97c9729df3dcce04cf65eeb6455cb7da6678c8a6f0",
    "data/router_v0_2/pilot/pilot_queries.jsonl": "1e903fa8681a08e637a6e28a7b8903c9276d42332072829cc4d1ff876829ff70",
    "data/router_v0_2/pilot/pilot_sidecar.jsonl": "2acfcaa14320c09ec067913555bd750abadc6197307da6158ec1f766eb84ced0",
    "outputs/router_v0_2/pilot/human_surface_review.csv": "882ea7e82f1db261603ff3a618b7a93532ed905a1850892f691acfcee7a5baae",
    "outputs/router_v0_2/pilot/pilot_audit.json": "4f51153ad1b96965d6138cbca9f2d7da899328f22a6f1f72fe23121dbb8a5999",
    "configs/router_v0_2_pilot_v2.json": "5b1a5f0be71d8b76bd1412a69513a4e8a3435599f1cd673015c0e5eec765ed5b",
    "data/router_v0_2/pilot_v2/pilot_queries.jsonl": "2473eda31fd94278b35c8329bfef029224842d1df0f6ab19fdc1051860f3b1a4",
    "data/router_v0_2/pilot_v2/pilot_sidecar.jsonl": "a326b122e47ce8cea1c896389c9c1ca4e8deb3ef1320dd20bcc0976f89c44562",
    "data/router_v0_2/pilot_v2/pilot_manifest.json": "5c73d6ba4148c5c9aea58b16af01bc8328388c32db6ffa95441fec21e7efec17",
    "outputs/router_v0_2/pilot_v2/pilot_audit.json": "dff507e74f66fa30bf8fb6214a200ccd250e7763db60316a0184b7079e7b8304",
    "outputs/router_v0_2/pilot_v2/human_surface_review.csv": "b8ff352939a462323dd55d94d6bf65606d1b5f8720e34eee75ae42e4d6487325",
    "outputs/router_v0_2/pilot_v2/spot_review_50.csv": "c7808a4716b45d423c541f49830e20d3a6eae776a16046c1098be9537e4873d7",
    "outputs/router_v0_2/pilot_v2/pilot_v1_vs_v2_comparison.json": "e8f70012d48ca4552fec4a3aefbaa10d2302dee5b7a2249a30455812be08b19c",
}


@pytest.fixture(scope="module")
def v3_config() -> dict:
    return load_benchmark_config(V3_CONFIG_PATH)


@pytest.fixture(scope="module")
def v3_pilot(v3_config: dict):
    return generate_pilot_v3(v3_config)


@pytest.fixture(scope="module")
def v3_audit(v3_config: dict, v3_pilot):
    return audit_pilot_v3(v3_config, *v3_pilot)


@pytest.fixture(scope="module")
def v3_artifacts():
    return build_pilot_v3_artifacts(V3_CONFIG_PATH, ROOT)


def test_difficulty_and_capability_count_are_decoupled(v3_audit: dict) -> None:
    diagnostics = v3_audit["difficulty_capability_diagnostics"]
    assert abs(diagnostics["pearson_correlation"]) <= 0.30
    assert abs(diagnostics["spearman_correlation"]) <= 0.30
    assert all(sum(value > 0 for value in row.values()) >= 2
               for row in diagnostics["difficulty_by_capability_count"].values())
    assert all(sum(value > 0 for value in row.values()) >= 2
               for row in diagnostics["capability_count_by_difficulty"].values())


def test_every_active_capability_has_valid_observable_evidence(v3_audit: dict) -> None:
    alignment = v3_audit["capability_evidence_alignment"]
    assert alignment["alignment_valid_rate"] == 1.0
    assert alignment["invalid_evidence_count"] == 0
    assert set(alignment["per_capability_invalid_realization_count"]) == {
        "AR", "SQ", "DE", "DS", "AU", "FK", "MH", "LC", "FT", "MT", "AM", "SC",
    }
    assert not any(alignment["per_capability_invalid_realization_count"].values())


def test_invalid_capability_realization_is_rejected(v3_config: dict) -> None:
    rng = NamespaceRNG(v3_config["pilot"]["master_seed"], v3_config["pilot"]["namespaces"])
    composition = compose_semantics_v3(v3_config, "code-000", rng)
    invalid = replace(
        composition,
        primary_capability="AM",
        secondary_capabilities=("AR",),
        capability_weights={"AM": 0.5, "AR": 0.5},
        requirement_levels={"AM": 0.5, "AR": 0.5},
    )
    with pytest.raises(ValueError, match="invalid capability realization"):
        generate_content_payload_v3(invalid, rng)


def test_every_frame_has_multiple_semantic_payload_skeletons(v3_audit: dict) -> None:
    diversity = v3_audit["semantic_skeleton_diversity"]
    assert diversity["minimum_distinct_skeleton_count"] >= 3
    assert diversity["maximum_largest_skeleton_share"] <= 0.5
    assert all(row["distinct_skeleton_count"] >= 3 for row in diversity["frames"].values())


def test_semantic_skeleton_masks_parameter_only_changes() -> None:
    first = "给定材料\n记录 R12 使用 v1.2，x=3。\n\n任务\n分析。"
    second = "给定材料\n记录 R98 使用 v9.4，y=57。\n\n任务\n分析。"
    assert semantic_skeleton(first) == semantic_skeleton(second)


def test_v3_generation_is_byte_reproducible(v3_artifacts) -> None:
    first, first_manifest = v3_artifacts
    second, second_manifest = build_pilot_v3_artifacts(V3_CONFIG_PATH, ROOT)
    assert first == second
    assert first_manifest == second_manifest


def test_written_v3_artifacts_match_reproducible_build(v3_artifacts) -> None:
    artifacts, _ = v3_artifacts
    assert all((ROOT / relative_path).read_bytes() == content
               for relative_path, content in artifacts.items())


def test_v3_generation_is_reproducible_across_process_hash_seeds() -> None:
    script = """
import hashlib
import json
from pathlib import Path
from zhiheng_router.benchmark_v02.pilot_v3 import build_pilot_v3_artifacts
root = Path.cwd()
artifacts, _ = build_pilot_v3_artifacts(root / 'configs/router_v0_2_pilot_v3.json', root)
print(json.dumps({path: hashlib.sha256(content).hexdigest()
                  for path, content in sorted(artifacts.items())}, sort_keys=True))
"""

    def hashes(seed: str) -> str:
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        return subprocess.run(
            [sys.executable, "-c", script], cwd=ROOT, env=environment,
            check=True, capture_output=True, text=True,
        ).stdout

    assert hashes("1") == hashes("2")


def test_v3_paths_versions_and_provenance_match_effective_config(
        v3_config: dict, v3_pilot, v3_artifacts) -> None:
    artifacts, manifest = v3_artifacts
    assert manifest["pilot_version"] == "router-v0.2-pilot-v3"
    assert manifest["generator_version"] == "2.2.0"
    assert manifest["dataset_version"] == "router-benchmark-v0.2.0"
    assert manifest["effective_config_path"] == "configs/router_v0_2_pilot_v3.json"
    assert manifest["master_seed"] == 20260907
    assert manifest["namespaces"] == v3_config["pilot"]["namespaces"]
    assert set(artifacts) == {
        "data/router_v0_2/pilot_v3/pilot_queries.jsonl",
        "data/router_v0_2/pilot_v3/pilot_sidecar.jsonl",
        "data/router_v0_2/pilot_v3/pilot_manifest.json",
        "outputs/router_v0_2/pilot_v3/pilot_audit.json",
        "outputs/router_v0_2/pilot_v3/human_surface_review.csv",
        SPOT_REVIEW_PATH,
        COMPARISON_PATH,
    }
    for sidecar in v3_pilot[1]:
        provenance = sidecar.generation_provenance
        assert provenance.generator_version == v3_config["generator_version"]
        assert provenance.master_seed == v3_config["pilot"]["master_seed"]
        assert provenance.content_namespace == v3_config["pilot"]["namespaces"]["content"]
        assert provenance.capability_namespace == v3_config["pilot"]["namespaces"]["capability"]
        assert provenance.difficulty_namespace == v3_config["pilot"]["namespaces"]["difficulty"]
        assert provenance.surface_namespace == v3_config["pilot"]["namespaces"]["surface"]


def test_v3_observable_and_sidecar_remain_physically_separate(v3_pilot) -> None:
    observables, sidecars = v3_pilot
    assert len(observables) == len(sidecars) == 250
    assert set(observables[0].to_dict()) == {
        "query_id", "query_text", "language", "explicit_output_constraints",
        "visible_structure", "source_kind", "dataset_version",
    }
    assert all(record.query_id == sidecar.query_id
               for record, sidecar in zip(observables, sidecars, strict=True))
    hidden_tokens = (
        '"primary_capability":', '"secondary_capabilities":', '"difficulty_bin":',
        '"semantic_frame_id":', '"source_family_id":', '"template_family_id":',
        '"quality":', '"cost":', '"latency":', "dev-model-",
    )
    serialized = "\n".join(json.dumps(record.to_dict(), sort_keys=True)
                           for record in observables).lower()
    assert not any(token in serialized for token in hidden_tokens)


def test_v3_contains_no_outcomes_and_spot_review_is_balanced(v3_artifacts) -> None:
    artifacts, manifest = v3_artifacts
    assert manifest["outcomes_included"] is False
    assert manifest["router_training_allowed"] is False
    rows = list(csv.DictReader(io.StringIO(
        artifacts[SPOT_REVIEW_PATH].decode("utf-8-sig"))))
    assert tuple(rows[0]) == SPOT_REVIEW_FIELDS
    assert not review_packet_has_forbidden_fields(SPOT_REVIEW_FIELDS)
    assert len(rows) == 50
    assert {task: sum(row["task"] == task for row in rows) for task in TASK_IDS} == {
        task: 10 for task in TASK_IDS
    }
    assert {"low", "medium", "high"} == {row["difficulty"] for row in rows}
    assert {"2", "3", "4"} == {row["active_capability_count"] for row in rows}
    assert {"zh", "mixed", "en"} <= {row["language"] for row in rows}
    assert all(len({row["semantic_frame"] for row in rows if row["task"] == task}) == 8
               for task in TASK_IDS)
    assert all(all(row[field] == "" for field in HUMAN_SCORE_FIELDS) for row in rows)


def test_v2_v3_comparison_is_reproducible_and_reports_r2_changes(v3_artifacts) -> None:
    artifacts, _ = v3_artifacts
    comparison = json.loads(artifacts[COMPARISON_PATH])
    assert comparison["same_master_seed"] is True
    assert comparison["v2"]["pearson_difficulty_capability_count"] > 0.9
    assert abs(comparison["v3"]["pearson_difficulty_capability_count"]) <= 0.30
    assert comparison["v2"]["minimum_frame_skeleton_count"] == 1
    assert comparison["v3"]["minimum_frame_skeleton_count"] >= 3
    assert comparison["v3"]["capability_alignment_valid_rate"] == 1.0
    assert artifacts[COMPARISON_PATH] == build_pilot_v3_artifacts(
        V3_CONFIG_PATH, ROOT)[0][COMPARISON_PATH]


def test_v3_generation_order_is_stable(v3_config: dict) -> None:
    ids = ["translation-014", "code-001", "summary-023", "math-007", "qa-030"]
    first = generate_pilot_v3(v3_config, ids)
    second = generate_pilot_v3(v3_config, list(reversed(ids)))
    assert [record.to_dict() for record in first[0]] == [record.to_dict() for record in second[0]]
    assert [record.to_dict() for record in first[1]] == [record.to_dict() for record in second[1]]


def test_protected_v1_v2_artifacts_remain_byte_identical() -> None:
    for relative_path, expected_hash in PROTECTED_HASHES.items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == expected_hash
