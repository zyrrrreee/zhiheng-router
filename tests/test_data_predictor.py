"""Input, group isolation, training and serialization checks."""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

import experiments.make_development_data as development_data
from experiments.make_development_data import generate_records, write_dataset
from experiments.run import run_experiment
from zhiheng_router.data import group_split, load_history, read_config, validate_history
from zhiheng_router.estimates import estimate_candidates, summarize_models
from zhiheng_router.features import FeatureExtractor, structural_features
from zhiheng_router.predictor import QualityPredictor, RoutingArtifact, load_artifact, save_artifact
from zhiheng_router.router import route
from zhiheng_router.schemas import (FeatureConfig, HistoricalRecord, ModelCandidate,
                                    RoutingPolicy, TrainingConfig)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def observations():
    rows = []
    for i in range(80):
        query = (f"Python代码编写列表排序，检查{i}个元素。" if i % 2 == 0
                 else f"数学求解方程 x + {i} = 100，证明过程。")
        for model in ("a", "b"):
            good = (model == "a") == (i % 2 == 0)
            rows.append(HistoricalRecord(f"{i}-{model}", str(i), f"family-{i // 4}", query,
                                         model, .9 if good else .6, 1 if model == "a" else 2,
                                         10 if model == "a" else 20, "annotation-only"))
    return rows


@pytest.fixture(scope="module")
def trained(observations):
    with threadpool_limits(limits=1):
        return QualityPredictor(FeatureConfig(max_features=256),
                                TrainingConfig(min_model_samples=4)).fit(observations, ["a", "b"])


@pytest.mark.parametrize("query", ["请解释缓存", "def f(x): return x + 1", "求解 x² + 3 = 7",
                                  "用 Python explain this", "中", "hi", "?"])
def test_normal_code_math_mixed_and_short_query(trained, query):
    values = trained.predict_pass_probabilities(query)
    assert set(values) == {"a", "b"}
    assert all(np.isfinite(p) and 0 <= p <= 1 for p in values.values())


@pytest.mark.parametrize("query", ["", " ", "\n\t"])
def test_blank_query_rejected(trained, query):
    with pytest.raises(ValueError, match="nonblank"):
        trained.predict_pass_probabilities(query)


def test_structural_flags():
    plain = structural_features("普通中文")
    code = structural_features("def f(x): return x + 2")
    math = structural_features("求解 x = 3")
    assert plain[2:] == [0, 0, 0]
    assert code[2:] == [1, 1, 1]
    assert math[2:] == [0, 1, 1]


def test_all_short_training_corpus_and_empty_ngram_inference():
    extractor = FeatureExtractor().fit(["中", "文"])
    result = extractor.transform(["?"])
    assert issparse(result) and result.shape == (1, 5)
    assert np.isfinite(result.data).all()


def test_group_split_is_disjoint_complete_and_reproducible(observations):
    splits = group_split(observations, seed=73)
    assert splits == group_split(observations, seed=73)
    assert sum(map(len, splits.values())) == len(observations)
    for attribute in ("query_id", "group_id", "query"):
        sets = [{getattr(r, attribute) for r in rows} for rows in splits.values()]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])


def test_same_text_with_different_identity_is_rejected(observations):
    with pytest.raises(ValueError, match="identical Query"):
        validate_history([observations[0], replace(observations[1], query_id="different")])


def test_inconsistent_query_group_is_rejected(observations):
    with pytest.raises(ValueError, match="inconsistent query_id"):
        validate_history([observations[0], replace(observations[1], group_id="other")])


def test_duplicate_call_and_missing_coverage(observations):
    with pytest.raises(ValueError, match="duplicate call_id"):
        validate_history([observations[0], observations[0]])
    with pytest.raises(ValueError, match="full coverage"):
        validate_history(observations[:-1], ["a", "b"], full_coverage=True)


def test_unique_training_queries_only_and_no_test_vocabulary(observations):
    train = group_split(observations)["train"]
    predictor = QualityPredictor(FeatureConfig(max_features=256), TrainingConfig(min_model_samples=4)).fit(train, ["a", "b"])
    assert predictor.features.scaler.n_samples_seen_ == len({r.query for r in train})
    vocabulary_before = dict(predictor.features.vectorizer.vocabulary_)
    scaler_before = predictor.features.scaler.scale_.copy()
    predictor.predict_pass_probabilities("𐐀𐐁𐐂 only-held-out-symbols")
    assert predictor.features.vectorizer.vocabulary_ == vocabulary_before
    np.testing.assert_array_equal(predictor.features.scaler.scale_, scaler_before)
    assert "𐐀𐐁" not in vocabulary_before


def test_independent_models_can_reverse_specialization(trained):
    code = trained.predict_pass_probabilities("Python代码编写列表排序，检查200个元素。")
    math = trained.predict_pass_probabilities("数学求解方程 x + 200 = 100，证明过程。")
    assert code["a"] > code["b"]
    assert math["b"] > math["a"]


def test_feature_transform_called_once_for_all_models(trained, monkeypatch):
    calls = []
    original = trained.features.transform

    def capture(queries):
        calls.append(queries)
        return original(queries)

    monkeypatch.setattr(trained.features, "transform", capture)
    trained.predict_pass_probabilities("普通新 Query")
    assert len(calls) == 1


def test_raw_task_annotation_does_not_affect_predictor(observations, trained):
    changed = [replace(r, task_type="a completely different annotation") for r in observations]
    other = QualityPredictor(FeatureConfig(max_features=256), TrainingConfig(min_model_samples=4)).fit(changed, ["a", "b"])
    assert other.predict_pass_probabilities("数学求解 x = 2") == trained.predict_pass_probabilities("数学求解 x = 2")


def test_label_threshold_is_inclusive(observations):
    rows = [replace(r, quality_score=.8 if r.quality_score > .8 else .7) for r in observations]
    predictor = QualityPredictor(FeatureConfig(max_features=128), TrainingConfig(min_model_samples=4)).fit(rows, ["a"])
    assert predictor.training_counts["a"]["positive_count"] == 40


def test_single_class_and_insufficient_samples_fail(observations):
    with pytest.raises(ValueError, match="single quality label class"):
        QualityPredictor(training_config=TrainingConfig(min_model_samples=4)).fit(
            [replace(r, quality_score=1) for r in observations], ["a"])
    with pytest.raises(ValueError, match="insufficient"):
        QualityPredictor().fit(observations[:2], ["a"])


def test_nonconvergence_fails_instead_of_emitting_artifact(observations):
    with pytest.raises(ConvergenceWarning):
        QualityPredictor(training_config=TrainingConfig(max_iter=1, min_model_samples=4)).fit(observations, ["a"])


def test_unknown_model_rejected(trained, observations):
    with pytest.raises(ValueError, match="unknown model"):
        trained.predict_pass_probabilities("普通 Query", ["new-model"])
    with pytest.raises(ValueError, match="missing training"):
        summarize_models(observations, ["missing"], .8)


def make_artifact(trained, observations):
    statistics = summarize_models(observations, ["a", "b"], .8)
    candidates = (ModelCandidate("a"), ModelCandidate("b"))
    policy = RoutingPolicy()
    config = {
        "models": [asdict(candidate) for candidate in candidates],
        "features": asdict(trained.features.config),
        "training": asdict(trained.config),
        "policy": asdict(policy),
    }
    return RoutingArtifact(deepcopy(trained), statistics, candidates, policy,
                           {"source_kind": "synthetic"}, config, {})


def test_artifact_round_trip_preserves_decisions(trained, observations, tmp_path):
    artifact = make_artifact(trained, observations)
    path = tmp_path / "router.joblib"
    save_artifact(artifact, path)
    restored = load_artifact(path)
    query = "数学求解 x = 2"
    original = route(artifact.candidates, estimate_candidates(
        query, ["a", "b"], artifact.predictor, artifact.statistics), artifact.policy)
    reloaded = route(restored.candidates, estimate_candidates(
        query, ["a", "b"], restored.predictor, restored.statistics), restored.policy)
    assert original == reloaded


@pytest.mark.parametrize("field,value", [
    ("router_probability_threshold", .9),
    ("cost_tolerance_abs", .1),
])
def test_artifact_rejects_policy_drift(trained, observations, tmp_path, field, value):
    artifact = make_artifact(trained, observations)
    artifact.policy = replace(artifact.policy, **{field: value})
    path = tmp_path / f"policy-{field}.joblib"
    save_artifact(artifact, path)
    with pytest.raises(ValueError, match="RoutingPolicy"):
        load_artifact(path)


def test_artifact_rejects_predictor_quality_threshold_drift(trained, observations, tmp_path):
    artifact = make_artifact(trained, observations)
    artifact.predictor.quality_score_threshold = .9
    path = tmp_path / "quality-threshold.joblib"
    save_artifact(artifact, path)
    with pytest.raises(ValueError, match="quality_score_threshold"):
        load_artifact(path)


def test_artifact_rejects_config_quality_threshold_drift(trained, observations, tmp_path):
    artifact = make_artifact(trained, observations)
    artifact.config["training"]["quality_score_threshold"] = .9
    path = tmp_path / "config-quality-threshold.joblib"
    save_artifact(artifact, path)
    with pytest.raises(ValueError, match="quality_score_threshold"):
        load_artifact(path)


def test_artifact_rejects_missing_predictor_model(trained, observations, tmp_path):
    artifact = make_artifact(trained, observations)
    artifact.predictor.models.pop("b")
    path = tmp_path / "missing-predictor-model.joblib"
    save_artifact(artifact, path)
    with pytest.raises(ValueError, match="predictor models"):
        load_artifact(path)


def test_artifact_rejects_missing_model_statistics(trained, observations, tmp_path):
    artifact = make_artifact(trained, observations)
    artifact.statistics.pop("b")
    path = tmp_path / "missing-statistics.joblib"
    save_artifact(artifact, path)
    with pytest.raises(ValueError, match="training statistics"):
        load_artifact(path)


def test_artifact_rejects_extra_unknown_model(trained, observations, tmp_path):
    artifact = make_artifact(trained, observations)
    artifact.predictor.models["unknown"] = artifact.predictor.models["a"]
    path = tmp_path / "extra-predictor-model.joblib"
    save_artifact(artifact, path)
    with pytest.raises(ValueError, match=r"extra=\['unknown'\]"):
        load_artifact(path)


def test_generator_reproducibility_coverage_and_nontrivial_outcomes(tmp_path):
    config = read_config(ROOT / "configs/mvp.json")
    rows, meta = generate_records(config)
    assert len(rows) == 3000 and meta["query_count"] == 600 and meta["group_count"] == 60
    assert (rows, meta) == generate_records(config)
    validate_history(rows, meta["model_ids"], full_coverage=True)
    assert any(r.model_id == "dev-model-e" and "只需简短" in r.query and r.quality_score < .8 for r in rows)
    best = {}
    for row in rows:
        best.setdefault(row.query_id, {})[row.model_id] = row.quality_score
    assert any(max(v.values()) > v["dev-model-a"] for v in best.values())
    for task in {r.task_type for r in rows}:
        texts = {r.query for r in rows if r.task_type == task}
        assert any("只需简短" in q for q in texts) and any("反例" in q for q in texts)
    write_dataset(config, tmp_path)
    original = (tmp_path / config["data_path"]).read_bytes()
    assert b"\r\n" not in original
    assert b"\r\n" not in (tmp_path / config["manifest_path"]).read_bytes()
    write_dataset(config, tmp_path)
    assert original == (tmp_path / config["data_path"]).read_bytes()


@pytest.mark.parametrize("path,value,message", [
    (("seed",), None, "seed"),
    (("profiles", "dev-model-b", "task_bonus", "code"), float("nan"), "task_bonus.code"),
    (("profiles", "dev-model-a", "ability"), float("inf"), "ability"),
    (("profiles", "dev-model-a", "difficulty_penalty"), float("nan"), "difficulty_penalty"),
    (("profiles", "dev-model-c", "base_cost"), 0.0, "base_cost"),
    (("profiles", "dev-model-d", "base_latency_ms"), -1.0, "base_latency_ms"),
    (("quality_noise_std",), float("inf"), "quality_noise_std"),
])
def test_generator_rejects_invalid_configuration_before_generation(path, value, message):
    config = read_config(ROOT / "configs/mvp.json")
    target = config["generator"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        generate_records(config)


def test_invalid_generator_config_leaves_existing_files_unchanged(tmp_path):
    config = read_config(ROOT / "configs/mvp.json")
    write_dataset(config, tmp_path)
    data_path = tmp_path / config["data_path"]
    meta_path = tmp_path / config["manifest_path"]
    before = data_path.read_bytes(), meta_path.read_bytes()
    invalid = deepcopy(config)
    invalid["generator"]["profiles"]["dev-model-b"]["task_bonus"]["code"] = float("nan")
    with pytest.raises(ValueError, match="task_bonus.code"):
        write_dataset(invalid, tmp_path)
    assert (data_path.read_bytes(), meta_path.read_bytes()) == before


def test_pair_replacement_failure_rolls_back_both_files(tmp_path, monkeypatch):
    config = read_config(ROOT / "configs/mvp.json")
    write_dataset(config, tmp_path)
    data_path = tmp_path / config["data_path"]
    meta_path = tmp_path / config["manifest_path"]
    before = data_path.read_bytes(), meta_path.read_bytes()
    real_replace = development_data.os.replace
    calls = 0

    def fail_second_replace(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated manifest replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(development_data.os, "replace", fail_second_replace)
    with pytest.raises(OSError, match="manifest replace"):
        write_dataset(config, tmp_path)
    assert (data_path.read_bytes(), meta_path.read_bytes()) == before
    assert not list(tmp_path.rglob("*.tmp"))


def test_test_outcome_changes_cannot_change_training_or_selection(tmp_path):
    config = read_config(ROOT / "configs/mvp.json")
    write_dataset(config, tmp_path)
    first = run_experiment(config, tmp_path)
    original_artifact = load_artifact(tmp_path / config["output_dir"] / "router.joblib")
    path = tmp_path / config["data_path"]
    test_ids = set(first["split"]["test"]["query_ids"])
    changed = [replace(r, quality_score=1-r.quality_score, cost=r.cost*1000,
                       latency_ms=r.latency_ms*1000, task_type="changed-test-annotation")
               if r.query_id in test_ids else r for r in load_history(path)]
    content = "".join(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in changed).encode("utf-8")
    path.write_bytes(content)
    meta_path = tmp_path / config["manifest_path"]
    manifest = json.loads(meta_path.read_text(encoding="utf-8"))
    manifest["sha256"] = hashlib.sha256(content).hexdigest()
    meta_path.write_text(json.dumps(manifest), encoding="utf-8")
    second = run_experiment(config, tmp_path)
    updated_artifact = load_artifact(tmp_path / config["output_dir"] / "router.joblib")
    assert first["training_counts"] == second["training_counts"]
    assert first["training_statistics"] == second["training_statistics"]
    assert first["baseline_definition"] == second["baseline_definition"]
    assert first["predictor"]["validation"] == second["predictor"]["validation"]
    for name in first["policies"]["test"]:
        before = [r["selected_model"] for r in first["policies"]["test"][name]["decisions"]]
        after = [r["selected_model"] for r in second["policies"]["test"][name]["decisions"]]
        assert before == after
    for model in original_artifact.predictor.models:
        np.testing.assert_array_equal(original_artifact.predictor.models[model].coef_,
                                      updated_artifact.predictor.models[model].coef_)
    assert first["policies"]["test"]["query_aware_router"]["metrics"]["average_cost"] != second["policies"]["test"]["query_aware_router"]["metrics"]["average_cost"]
