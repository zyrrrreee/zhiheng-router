from __future__ import annotations

import csv
from copy import deepcopy
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

from zhiheng_router.benchmark_v02.audit_v2 import audit_pilot_v2
from zhiheng_router.benchmark_v02.config import load_benchmark_config
from zhiheng_router.benchmark_v02.payloads_v2 import generate_content_payload
from zhiheng_router.benchmark_v02.pilot_v2 import (
    COMPARISON_PATH,
    SPOT_REVIEW_FIELDS,
    SPOT_REVIEW_PATH,
    build_pilot_v2_artifacts,
    review_packet_has_forbidden_fields,
    write_pilot_v2_artifacts,
)
from zhiheng_router.benchmark_v02.rng import NamespaceRNG
from zhiheng_router.benchmark_v02.schema import TASK_IDS
from zhiheng_router.benchmark_v02.surface import compose_semantics
from zhiheng_router.benchmark_v02.surface_v2 import generate_pilot_v2, realize_query_v2
from zhiheng_router.benchmark_v02.validation_v2 import lint_surface, validate_answerability


ROOT = Path(__file__).resolve().parents[1]
V2_CONFIG_PATH = ROOT / "configs/router_v0_2_pilot_v2.json"
HUMAN_SCORE_FIELDS = (
    "naturalness", "task_completeness", "constraint_consistency",
    "capability_cue_naturalness", "template_artifact", "shortcut_risk",
    "difficulty_plausibility", "ambiguity_unanswerable_risk",
)
V1_HASHES = {
    "configs/router_v0_2_benchmark.json": "7706307ea88db1309c998d7e87f84dbfb275b28daffc890875ca581f200fd964",
    "data/router_v0_2/pilot/pilot_manifest.json": "6307eec5690726c94cbacb97c9729df3dcce04cf65eeb6455cb7da6678c8a6f0",
    "data/router_v0_2/pilot/pilot_queries.jsonl": "1e903fa8681a08e637a6e28a7b8903c9276d42332072829cc4d1ff876829ff70",
    "data/router_v0_2/pilot/pilot_sidecar.jsonl": "2acfcaa14320c09ec067913555bd750abadc6197307da6158ec1f766eb84ced0",
    "outputs/router_v0_2/pilot/human_surface_review.csv": "882ea7e82f1db261603ff3a618b7a93532ed905a1850892f691acfcee7a5baae",
    "outputs/router_v0_2/pilot/pilot_audit.json": "4f51153ad1b96965d6138cbca9f2d7da899328f22a6f1f72fe23121dbb8a5999",
}


@pytest.fixture(scope="module")
def v2_config() -> dict:
    return load_benchmark_config(V2_CONFIG_PATH)


@pytest.fixture(scope="module")
def v2_pilot(v2_config: dict):
    return generate_pilot_v2(v2_config)


@pytest.fixture(scope="module")
def v2_audit(v2_config: dict, v2_pilot):
    return audit_pilot_v2(v2_config, *v2_pilot)


@pytest.fixture(scope="module")
def v2_artifacts():
    return build_pilot_v2_artifacts(V2_CONFIG_PATH, ROOT)


def _payload(v2_config: dict, stable_id: str):
    rng = NamespaceRNG(v2_config["pilot"]["master_seed"], v2_config["pilot"]["namespaces"])
    composition = compose_semantics(v2_config, stable_id, rng)
    return composition, generate_content_payload(composition, rng), rng


def test_code_debug_payload_has_code_log_and_failing_io(v2_config: dict) -> None:
    _, payload, _ = _payload(v2_config, "code-001")
    assert {"code_block", "log", "sample_io"} <= set(payload.structural_markers)
    assert "```python" in payload.material
    assert re.search(r"失败输入|Failing input", payload.material)
    assert re.search(r"预期|Expected", payload.material)


@pytest.mark.parametrize("stable_id", [f"math-{index:03d}" for index in range(8)])
def test_each_math_frame_has_sufficient_solve_conditions(v2_config: dict, stable_id: str) -> None:
    _, payload, _ = _payload(v2_config, stable_id)
    assert "numeric_conditions" in payload.structural_markers
    assert len(re.findall(r"\d+(?:\.\d+)?", payload.material)) >= 3
    assert payload.task_instruction


def test_qa_payload_contains_real_evidence_passages(v2_config: dict) -> None:
    _, payload, _ = _payload(v2_config, "qa-006")
    assert "evidence_passages" in payload.structural_markers
    assert 2 <= payload.passage_count <= 5
    assert payload.material.count("[") >= 2


def test_summary_payload_contains_three_to_six_source_passages(v2_config: dict) -> None:
    _, payload, _ = _payload(v2_config, "summary-002")
    assert "source_material" in payload.structural_markers
    assert 3 <= payload.passage_count <= 6
    assert payload.material.count("[") >= 3


@pytest.mark.parametrize("stable_id", ["translation-000", "translation-001", "translation-002"])
def test_translation_payload_has_distinct_real_source_text(v2_config: dict, stable_id: str) -> None:
    composition, payload, _ = _payload(v2_config, stable_id)
    assert "source_text" in payload.structural_markers
    assert payload.material != payload.scenario
    assert len(payload.material) >= 50
    assert any(marker in payload.material for marker in (
        "[源文]", "[Source text]", "[中文版本]", "[English version]"))
    if composition.translation_direction != "bilingual_revision":
        assert "bilingual" not in payload.output_contract


def test_translation_notice_dates_are_valid(v2_pilot) -> None:
    sidecar_by_id = {record.query_id: record for record in v2_pilot[1]}
    notices = [record for record in v2_pilot[0]
               if sidecar_by_id[record.query_id].semantic_frame_id.endswith("translation-notice")]
    assert notices
    for record in notices:
        match = re.search(r"(?:9 月|September )\s*(\d{1,2})", record.query_text)
        assert match and 1 <= int(match.group(1)) <= 30


def test_every_v2_query_passes_task_specific_answerability(v2_pilot) -> None:
    sidecar_by_id = {record.query_id: record for record in v2_pilot[1]}
    assert all(validate_answerability(record, sidecar_by_id[record.query_id])["valid"]
               for record in v2_pilot[0])


def test_capabilities_have_multiple_structural_realizations(v2_audit: dict) -> None:
    assert all(row["evidence_variant_count"] >= 4
               for row in v2_audit["capability_realization_diversity"].values())
    assert all(row["semantic_frame_count"] >= 3
               for row in v2_audit["capability_realization_diversity"].values())


def test_cue_masking_retains_structural_evidence(v2_audit: dict) -> None:
    assert v2_audit["cue_masking_diagnostic"]["minimum_structure_retained_rate"] >= 0.9


def test_explicit_capability_label_is_still_rejected(v2_config: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    import zhiheng_router.benchmark_v02.surface_v2 as surface_v2

    composition, payload, rng = _payload(v2_config, "code-001")
    monkeypatch.setattr(surface_v2, "generate_content_payload",
                        lambda *_: replace(payload, material=payload.material + "\n[AR]"))
    with pytest.raises(ValueError, match="forbidden label"):
        realize_query_v2(v2_config, composition, rng)


@pytest.mark.parametrize("text", ["the the worker failed", "A the worker failed"])
def test_lint_detects_duplicated_determiner(text: str) -> None:
    assert any(issue["code"] == "duplicated_determiner" for issue in lint_surface(text, "en"))


def test_lint_detects_repeated_word() -> None:
    assert any(issue["code"] == "repeated_english_word"
               for issue in lint_surface("The worker worker failed.", "en"))


def test_lint_detects_hard_mixed_instruction_injection() -> None:
    issues = lint_surface("请处理日志。 Connect dependencies across the material.", "mixed")
    assert any(issue["code"] == "hard_inserted_english_instruction" for issue in issues)


def test_lint_detects_punctuation_collision() -> None:
    assert any(issue["code"] == "mixed_punctuation_collision"
               for issue in lint_surface("结果如下，,请核对。", "mixed"))


def test_generated_v2_surfaces_pass_basic_lint(v2_audit: dict) -> None:
    assert v2_audit["surface_lint"] == {"failure_count": 0, "failures": []}


def test_surface_does_not_render_scenario_as_duplicate_task_preamble(v2_config: dict) -> None:
    records, _ = generate_pilot_v2(v2_config)
    assert all("背景\n" not in record.query_text and "Context\n" not in record.query_text
               for record in records)


def test_task_payloads_do_not_contain_unrelated_generic_cue_blocks(v2_pilot) -> None:
    sidecar_by_id = {record.query_id: record for record in v2_pilot[1]}
    forbidden_by_task = {
        "math": ("ROUTER_TTL", "request_id", "LEFT JOIN", "analyze(records"),
        "qa": ("def analyze(records", "schedule(tasks, limit)"),
        "summary": ("def analyze(records", "schedule(tasks, limit)", "next_cursor"),
        "translation": ("def analyze(records", "schedule(tasks, limit)"),
    }
    for record in v2_pilot[0]:
        task = sidecar_by_id[record.query_id].coarse_task
        assert not any(cue.lower() in record.query_text.lower()
                       for cue in forbidden_by_task.get(task, ()))


def test_english_code_and_math_queries_are_actually_english(v2_pilot) -> None:
    sidecar_by_id = {record.query_id: record for record in v2_pilot[1]}
    records = [record for record in v2_pilot[0]
               if record.language == "en" and sidecar_by_id[record.query_id].coarse_task in {"code", "math"}]
    assert records
    assert all(len(re.findall(r"[\u4e00-\u9fff]", record.query_text)) == 0 for record in records)


def test_v1_artifacts_remain_byte_identical() -> None:
    for relative_path, expected_hash in V1_HASHES.items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == expected_hash


def test_v2_generation_is_byte_reproducible(v2_artifacts) -> None:
    first, first_manifest = v2_artifacts
    second, second_manifest = build_pilot_v2_artifacts(V2_CONFIG_PATH, ROOT)
    assert first == second
    assert first_manifest == second_manifest


def test_v2_generation_is_reproducible_across_process_hash_seeds() -> None:
    script = """
import hashlib
import json
from pathlib import Path
from zhiheng_router.benchmark_v02.pilot_v2 import build_pilot_v2_artifacts
root = Path.cwd()
artifacts, _ = build_pilot_v2_artifacts(root / 'configs/router_v0_2_pilot_v2.json', root)
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


def test_v2_manifest_points_to_new_versions(v2_artifacts) -> None:
    artifacts, manifest = v2_artifacts
    assert manifest["pilot_version"] == "router-v0.2-pilot-v2"
    assert manifest["generator_version"] == "2.1.0"
    assert manifest["dataset_version"] == "router-benchmark-v0.2.0"
    assert manifest["automatic_audit_ready_for_spot_review"] is True
    assert manifest["outcomes_included"] is False
    assert "data/router_v0_2/pilot_v2/pilot_manifest.json" in artifacts


def test_v1_v2_comparison_is_reproducible_and_reports_improvement(v2_artifacts) -> None:
    artifacts, _ = v2_artifacts
    comparison = json.loads(artifacts[COMPARISON_PATH])
    assert comparison["same_master_seed"] is True
    assert comparison["v1"]["answerability_valid_count"] < comparison["v2"]["answerability_valid_count"]
    assert comparison["v2"]["answerability_valid_count"] == 250
    assert comparison["v2"]["max_fixed_sentence_share"] < comparison["v1"]["max_fixed_sentence_share"]
    assert artifacts[COMPARISON_PATH] == build_pilot_v2_artifacts(V2_CONFIG_PATH, ROOT)[0][COMPARISON_PATH]


def test_spot_review_has_50_balanced_rows_and_no_hidden_results(v2_artifacts) -> None:
    artifacts, _ = v2_artifacts
    text = artifacts[SPOT_REVIEW_PATH].decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    assert tuple(rows[0]) == SPOT_REVIEW_FIELDS
    assert not review_packet_has_forbidden_fields(SPOT_REVIEW_FIELDS)
    assert len(rows) == 50
    assert {task: sum(row["task"] == task for row in rows) for task in TASK_IDS} == {
        task: 10 for task in TASK_IDS
    }
    assert all(all(row[field] == "" for field in HUMAN_SCORE_FIELDS) for row in rows)
    assert not any(token in text.lower() for token in (
        "dev-model-", "quality-best", "efficient winner", "sampled outcome", "router result",
    ))


def test_spot_review_covers_language_difficulty_and_capability_count(v2_artifacts) -> None:
    rows = list(csv.DictReader(io.StringIO(v2_artifacts[0][SPOT_REVIEW_PATH].decode("utf-8-sig"))))
    assert {"zh", "mixed", "en"} <= {row["language"] for row in rows}
    assert {"low", "medium", "high"} == {row["difficulty"] for row in rows}
    assert {"2", "3", "4"} == {row["active_capability_count"] for row in rows}
    assert any(row["high_risk_lexical_association"] for row in rows)
    assert all(len({row["semantic_frame"] for row in rows if row["task"] == task}) == 8
               for task in TASK_IDS)


def test_writing_v2_does_not_change_v1() -> None:
    before = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in V1_HASHES}
    write_pilot_v2_artifacts(V2_CONFIG_PATH, ROOT)
    after = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in V1_HASHES}
    assert after == before == V1_HASHES


def test_surface_namespace_change_preserves_v2_capabilities(v2_config: dict) -> None:
    changed = deepcopy(v2_config)
    changed["pilot"]["namespaces"]["surface"] += "/changed"
    original = generate_pilot_v2(v2_config, ["qa-017"])[1][0]
    modified = generate_pilot_v2(changed, ["qa-017"])[1][0]
    assert original.active_capabilities == modified.active_capabilities
    assert original.capability_weights == modified.capability_weights


def test_v2_generation_order_is_stable(v2_config: dict) -> None:
    ids = ["translation-014", "code-001", "summary-023", "math-007", "qa-030"]
    first = generate_pilot_v2(v2_config, ids)
    second = generate_pilot_v2(v2_config, list(reversed(ids)))
    assert [record.to_dict() for record in first[0]] == [record.to_dict() for record in second[0]]
    assert [record.to_dict() for record in first[1]] == [record.to_dict() for record in second[1]]
