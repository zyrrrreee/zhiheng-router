"""LLM Judge evaluation without real provider calls."""

from types import SimpleNamespace

import pytest

from zhiheng_router.evaluators import (
    LLMJudgeError,
    LLMJudgeEvaluator,
    QualityEvaluator,
)


class FakeCompletions:
    def __init__(self, *, response: object | None = None,
                 error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def judge_response(content: str) -> object:
    return SimpleNamespace(
        id="judge-response-1",
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
    )


def test_mock_judge_response_parses_score_without_model_leakage() -> None:
    completions = FakeCompletions(response=judge_response(
        '{"quality_score": 0.8, "reason": "Correct and complete."}'))
    evaluator = LLMJudgeEvaluator(
        "deepseek-v4-pro",
        api_key="test-key",
        threshold=0.8,
        client=FakeClient(completions),
    )

    result = evaluator.evaluate(
        "What is 2 + 2?",
        "The answer is 4.",
        metadata={"model_id": "evaluated-secret-model"},
    )

    assert isinstance(evaluator, QualityEvaluator)
    assert result.quality_score == 0.8
    assert result.quality_pass is True
    assert result.measurement_kind == "model_judge"
    assert result.metadata == {
        "judge_model_id": "deepseek-v4-pro",
        "judge_response_id": "judge-response-1",
        "threshold": 0.8,
        "reason": "Correct and complete.",
    }
    assert completions.calls[0]["model"] == "deepseek-v4-pro"
    assert completions.calls[0]["response_format"] == {"type": "json_object"}
    prompt = str(completions.calls[0]["messages"])
    assert "evaluated-secret-model" not in prompt


@pytest.mark.parametrize("content", [
    "not JSON",
    '{"quality_score": 1.1, "reason": "Out of range."}',
    '{"quality_score": 0.7}',
])
def test_invalid_judge_score_response_raises_structured_error(content: str) -> None:
    evaluator = LLMJudgeEvaluator(
        "deepseek-v4-pro",
        api_key="test-key",
        client=FakeClient(FakeCompletions(response=judge_response(content))),
    )

    with pytest.raises(LLMJudgeError, match="invalid score response") as captured:
        evaluator.evaluate("Query", "Response")

    assert captured.value.judge_model_id == "deepseek-v4-pro"
    assert captured.value.error_kind == "invalid_response"


def test_judge_api_error_is_converted_to_structured_error() -> None:
    evaluator = LLMJudgeEvaluator(
        "deepseek-v4-pro",
        api_key="test-key",
        client=FakeClient(FakeCompletions(error=RuntimeError("provider unavailable"))),
    )

    with pytest.raises(LLMJudgeError, match="invocation failed") as captured:
        evaluator.evaluate("Query", "Response")

    assert captured.value.judge_model_id == "deepseek-v4-pro"
    assert captured.value.error_kind == "error"
    assert isinstance(captured.value.__cause__, RuntimeError)
