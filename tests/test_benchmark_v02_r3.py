from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from zhiheng_router.benchmark_v02.audit_v4 import audit_pilot_v4
from zhiheng_router.benchmark_v02.config import load_benchmark_config
from zhiheng_router.benchmark_v02.pilot_v4 import (
    COMPARISON_PATH,
    HUMAN_SCORE_FIELDS,
    SPOT_REVIEW_FIELDS,
    SPOT_REVIEW_PATH,
    build_pilot_v4_artifacts,
    review_packet_has_forbidden_fields,
)
from zhiheng_router.benchmark_v02.schema import ObservableQueryRecord, TASK_IDS
from zhiheng_router.benchmark_v02.surface_v4 import generate_pilot_v4
from zhiheng_router.benchmark_v02.validation_v4 import (
    semantic_sibling_signature,
    sibling_difficulty_conflicts,
    validate_capability_roles,
    validate_frame_completeness,
)


ROOT = Path(__file__).resolve().parents[1]
V4_CONFIG_PATH = ROOT / "configs/router_v0_2_pilot_v4.json"
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
    "configs/router_v0_2_pilot_v3.json": "35844a7d7231e41c4c5fbddffb056868c9c7499a712aebf081210ef40c8b9743",
    "data/router_v0_2/pilot_v3/pilot_queries.jsonl": "044f5038155ac3994416f476b0de361a9765e8087e8da73952230e9315d7dda3",
    "data/router_v0_2/pilot_v3/pilot_sidecar.jsonl": "ce771c618ba8751036b2bae4bba4e10d80ec225244ed105bd43bda35869bb3d7",
    "data/router_v0_2/pilot_v3/pilot_manifest.json": "3af8bb015ec5f7c9cd5f3b6d6e5944e492f29127cc8683d426624515a9519768",
    "outputs/router_v0_2/pilot_v3/pilot_audit.json": "7855b83fa006a16e52ab37a182db4c3249f48eca19a355095c6f046fa13d32ff",
    "outputs/router_v0_2/pilot_v3/human_surface_review.csv": "45da489a01a01cf1256ebc3850e83cfadf3f592192e09bfb3e9702e9243bcf59",
    "outputs/router_v0_2/pilot_v3/spot_review_50.csv": "ced9c1eec74d09056f7718534dd96b21079c5de55377190f64b852cd6d0535ed",
    "outputs/router_v0_2/pilot_v3/pilot_v2_vs_v3_comparison.json": "2c4c54117ff96097428ff3a4f0608ab1e649d21a05cb8cb8be58a724a9e98d4b",
}


@pytest.fixture(scope="module")
def v4_config() -> dict:
    return load_benchmark_config(V4_CONFIG_PATH)


@pytest.fixture(scope="module")
def v4_pilot(v4_config: dict):
    return generate_pilot_v4(v4_config)


@pytest.fixture(scope="module")
def v4_audit(v4_config: dict, v4_pilot):
    return audit_pilot_v4(v4_config, *v4_pilot)


@pytest.fixture(scope="module")
def v4_artifacts():
    return build_pilot_v4_artifacts(V4_CONFIG_PATH, ROOT)


def _v3_observables() -> dict[str, ObservableQueryRecord]:
    path = ROOT / "data/router_v0_2/pilot_v3/pilot_queries.jsonl"
    records = [ObservableQueryRecord.from_dict(json.loads(line))
               for line in path.read_text(encoding="utf-8").splitlines()]
    return {record.query_id: record for record in records}


def test_sibling_signature_detects_value_only_v3_siblings() -> None:
    records = _v3_observables()
    pairs = (
        ("pilot-q-code-005", "pilot-q-code-029"),
        ("pilot-q-math-049", "pilot-q-math-001"),
        ("pilot-q-summary-026", "pilot-q-summary-002"),
        ("pilot-q-translation-037", "pilot-q-translation-029"),
    )
    assert all(semantic_sibling_signature(records[first].query_text)
               == semantic_sibling_signature(records[second].query_text)
               for first, second in pairs)


def test_sibling_signature_allows_real_structural_difference() -> None:
    first = "给定材料\nA 依赖 B。\n\n任务\n给出合法调度。\n\n交付形式\n列表"
    second = "给定材料\nA 与 B 共享锁，失败后必须回滚。\n\n任务\n定位状态错误。\n\n交付形式\n列表"
    assert semantic_sibling_signature(first) != semantic_sibling_signature(second)


def test_translation_always_has_mt_or_ft(v4_pilot) -> None:
    translations = [record for record in v4_pilot[1] if record.coarse_task == "translation"]
    assert len(translations) == 50
    assert all(set(record.active_capabilities) & {"MT", "FT"} for record in translations)


def test_translation_technical_tokens_do_not_trigger_au_or_de(v4_pilot) -> None:
    risky_frames = {"frame-translation-technical", "frame-translation-incident"}
    records = [record for record in v4_pilot[1]
               if record.semantic_frame_id in risky_frames]
    assert records
    assert all(not (set(record.active_capabilities) & {"AU", "DE"}) for record in records)


def test_ambiguous_translation_target_is_rejected(v4_pilot) -> None:
    pair = next((record, sidecar) for record, sidecar in zip(*v4_pilot, strict=True)
                if sidecar.coarse_task == "translation"
                and record.visible_structure.translation_direction == "bilingual_revision")
    record, sidecar = pair
    vague = replace(record, query_text=record.query_text.replace(
        "核对并修订中英文版本", "翻译为目标语言"))
    result = validate_frame_completeness([vague], [sidecar])
    assert result["ambiguous_translation_target_count"] == 1


def test_timezone_conversion_without_time_of_day_is_rejected(v4_pilot) -> None:
    pair = next((record, sidecar) for record, sidecar in zip(*v4_pilot, strict=True)
                if sidecar.semantic_frame_id == "frame-code-data-pipeline"
                and "UTC 日期" in record.query_text)
    record, sidecar = pair
    incomplete = replace(record, query_text=record.query_text.replace(" 00:30", ""))
    result = validate_frame_completeness([incomplete], [sidecar])
    assert result["incomplete_timezone_conversion_count"] == 1


def test_timeout_retry_task_requires_complete_deliverable(v4_pilot) -> None:
    pair = next((record, sidecar) for record, sidecar in zip(*v4_pilot, strict=True)
                if sidecar.semantic_frame_id == "frame-code-api-client")
    record, sidecar = pair
    incomplete_text = re.sub(
        r"; include timeout_policy and maximum_retries fields", "",
        record.query_text,
    )
    result = validate_frame_completeness(
        [replace(record, query_text=incomplete_text)], [sidecar])
    assert result["task_deliverable_mismatch_count"] == 1


def test_data_pipeline_and_simple_math_do_not_gain_unneeded_capabilities(v4_pilot) -> None:
    sidecars = v4_pilot[1]
    pipelines = [record for record in sidecars
                 if record.semantic_frame_id == "frame-code-data-pipeline"]
    simple_math = [record for record in sidecars if record.semantic_frame_id in {
        "frame-math-probability", "frame-math-proof",
    }]
    assert pipelines and simple_math
    assert all("AU" not in record.active_capabilities for record in pipelines)
    assert all("MH" not in record.active_capabilities for record in simple_math)


def test_severe_sibling_low_high_conflict_is_rejected(v4_pilot) -> None:
    observable = v4_pilot[0][0]
    sidecar = v4_pilot[1][0]
    low = replace(sidecar, difficulty=0.2, difficulty_bin="low")
    other_id = "pilot-q-code-998"
    high_observable = replace(observable, query_id=other_id)
    high = replace(sidecar, query_id=other_id, difficulty=0.8, difficulty_bin="high")
    result = sibling_difficulty_conflicts(
        [observable, high_observable], [low, high])
    assert result["severe_sibling_difficulty_conflict_count"] == 1


def test_all_r2_and_r3_automatic_gates_pass(v4_audit: dict) -> None:
    assert v4_audit["ready_for_final_human_spot_review"] is True
    assert all(v4_audit["checks"].values())
    assert v4_audit["semantic_sibling_duplicates"]["duplicate_group_count"] == 0
    roles = v4_audit["capability_role_validation"]
    assert roles["capability_necessity_valid_rate"] == 1.0
    assert roles["capability_coverage_valid_rate"] == 1.0
    assert roles["invalid_active_capability_count"] == 0
    assert roles["missing_required_capability_count"] == 0
    assert not any(v4_audit["frame_semantic_completeness"][key] for key in (
        "ambiguous_translation_target_count", "incomplete_timezone_conversion_count",
        "task_deliverable_mismatch_count",
    ))
    assert v4_audit["sibling_difficulty_consistency"][
        "severe_sibling_difficulty_conflict_count"] == 0


def test_capability_role_validator_accepts_all_generated_records(v4_pilot) -> None:
    result = validate_capability_roles(*v4_pilot)
    assert result["capability_necessity_valid_rate"] == 1.0
    assert result["capability_coverage_valid_rate"] == 1.0


def test_v4_generation_is_byte_reproducible(v4_artifacts) -> None:
    first, first_manifest = v4_artifacts
    second, second_manifest = build_pilot_v4_artifacts(V4_CONFIG_PATH, ROOT)
    assert first == second
    assert first_manifest == second_manifest


def test_v4_generation_is_reproducible_across_process_hash_seeds() -> None:
    script = """
import hashlib
import json
from pathlib import Path
from zhiheng_router.benchmark_v02.pilot_v4 import build_pilot_v4_artifacts
root = Path.cwd()
artifacts, _ = build_pilot_v4_artifacts(root / 'configs/router_v0_2_pilot_v4.json', root)
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


def test_written_v4_artifacts_match_reproducible_build(v4_artifacts) -> None:
    artifacts, _ = v4_artifacts
    assert all((ROOT / path).read_bytes() == content for path, content in artifacts.items())


def test_v4_versions_paths_and_no_outcomes(v4_pilot, v4_artifacts) -> None:
    artifacts, manifest = v4_artifacts
    assert manifest["pilot_version"] == "router-v0.2-pilot-v4"
    assert manifest["generator_version"] == "2.3.0"
    assert manifest["dataset_version"] == "router-benchmark-v0.2.0"
    assert manifest["outcomes_included"] is False
    assert manifest["human_scores_completed"] is False
    assert all(sidecar.generation_provenance.generator_version == "2.3.0"
               for sidecar in v4_pilot[1])
    assert "data/router_v0_2/pilot_v4/pilot_manifest.json" in artifacts
    assert COMPARISON_PATH in artifacts


def test_spot_review_is_balanced_and_unscored(v4_artifacts) -> None:
    rows = list(csv.DictReader(io.StringIO(
        v4_artifacts[0][SPOT_REVIEW_PATH].decode("utf-8-sig"))))
    assert tuple(rows[0]) == SPOT_REVIEW_FIELDS
    assert not review_packet_has_forbidden_fields(SPOT_REVIEW_FIELDS)
    assert len(rows) == 50
    assert {task: sum(row["task"] == task for row in rows) for task in TASK_IDS} == {
        task: 10 for task in TASK_IDS
    }
    assert {"zh", "mixed", "en"} <= {row["language"] for row in rows}
    assert {"low", "medium", "high"} == {row["difficulty"] for row in rows}
    assert all(all(row[field] == "" for field in HUMAN_SCORE_FIELDS) for row in rows)


def test_v3_v4_comparison_is_compact_and_reproducible(v4_artifacts) -> None:
    comparison = json.loads(v4_artifacts[0][COMPARISON_PATH])
    assert comparison["same_master_seed"] is True
    assert comparison["v3"]["semantic_sibling_duplicate_group_count"] > 0
    assert comparison["v4"]["semantic_sibling_duplicate_group_count"] == 0
    assert comparison["v4"]["capability_necessity_valid_rate"] == 1.0
    assert comparison["v4"]["capability_coverage_valid_rate"] == 1.0
    assert v4_artifacts[0][COMPARISON_PATH] == build_pilot_v4_artifacts(
        V4_CONFIG_PATH, ROOT)[0][COMPARISON_PATH]


def test_v1_v2_v3_artifacts_remain_byte_identical() -> None:
    for relative_path, expected_hash in PROTECTED_HASHES.items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == expected_hash
