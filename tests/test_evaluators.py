"""Independent quality evaluator contracts and deterministic mock results."""

import pytest

from zhiheng_router.evaluators import MockQualityEvaluator, QualityEvaluator, QualityResult


@pytest.mark.parametrize("score", [-0.01, 1.01, float("nan"), float("inf"), True])
def test_quality_result_rejects_invalid_score(score: float) -> None:
    with pytest.raises(ValueError, match="quality_score"):
        QualityResult(score, False, "simulated")


def test_quality_result_validates_schema() -> None:
    result = QualityResult(0.9, True, "simulated", {"source": "test"})
    assert result.quality_score == 0.9
    assert result.metadata == {"source": "test"}
    assert not hasattr(result, "cost")
    assert not hasattr(result, "latency")
    with pytest.raises(ValueError, match="quality_pass"):
        QualityResult(0.9, 1, "simulated")
    with pytest.raises(ValueError, match="measurement_kind"):
        QualityResult(0.9, True, "unknown")


def test_mock_evaluator_is_deterministic() -> None:
    evaluator = MockQualityEvaluator({("query-1", "response-1"): 0.9})
    first = evaluator.evaluate("query-1", "response-1")
    second = evaluator.evaluate("query-1", "response-1")
    assert first == second
    assert isinstance(evaluator, QualityEvaluator)
    assert first.measurement_kind == "simulated"
    assert first.metadata["score_source"] == "fixed_mapping"


def test_quality_pass_uses_inclusive_threshold() -> None:
    evaluator = MockQualityEvaluator(
        {("pass", "answer"): 0.8, ("fail", "answer"): 0.6}, threshold=0.8)
    assert evaluator.evaluate("pass", "answer").quality_pass is True
    assert evaluator.evaluate("fail", "answer").quality_pass is False


def test_unconfigured_pair_uses_default_score() -> None:
    evaluator = MockQualityEvaluator(default_score=0.7, threshold=0.8)
    result = evaluator.evaluate("unconfigured query", "unconfigured response")
    assert result.quality_score == 0.7
    assert result.quality_pass is False
    assert result.metadata["score_source"] == "default"


def test_mock_evaluator_does_not_depend_on_model_id_metadata() -> None:
    evaluator = MockQualityEvaluator({("same query", "same response"): 0.9})
    first = evaluator.evaluate(
        "same query", "same response", metadata={"model_id": "dev-model-a"})
    second = evaluator.evaluate(
        "same query", "same response", metadata={"model_id": "dev-model-f"})
    assert first == second
