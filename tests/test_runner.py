"""Single-query Router-to-Adapter execution behavior."""

from collections.abc import Callable

import pytest

from zhiheng_router.adapters import AdapterError, MockModelAdapter, ModelRequest, ModelResponse
from zhiheng_router.evaluators import MockQualityEvaluator, QualityResult
from zhiheng_router.runner import BenchmarkRunError, BenchmarkRunner
from zhiheng_router.schemas import CandidateEstimate, RoutingDecision


def decision(model_id: str, *, fallback_used: bool = False) -> RoutingDecision:
    estimate = CandidateEstimate(model_id, 0.9, 0.2, 12.0)
    return RoutingDecision(
        selected_model=model_id,
        pass_probability=estimate.pass_probability,
        estimated_cost=estimate.estimated_cost,
        estimated_latency_ms=estimate.estimated_latency_ms,
        fallback_used=fallback_used,
        decision_reason="TEST_SELECTION",
        qualified_models=(model_id,),
        candidate_estimates=(estimate,),
    )


class RecordingAdapter:
    def __init__(self, adapter: MockModelAdapter) -> None:
        self.adapter = adapter
        self.requests: list[ModelRequest] = []

    @property
    def model_id(self) -> str:
        return self.adapter.model_id

    def invoke(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.adapter.invoke(request)


class RecordingQualityEvaluator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, object]] = []

    def evaluate(self, query_text: str, response_text: str,
                 metadata: object = None) -> QualityResult:
        self.calls.append((query_text, response_text, metadata))
        return QualityResult(0.9, True, "simulated")


def make_adapter(model_id: str, response_text: str = "固定回答",
                 *, error_kind: str | None = None) -> MockModelAdapter:
    return MockModelAdapter(
        model_id,
        response_text=response_text,
        input_units=12,
        output_units=4,
        cost=0.25,
        model_latency_ms=18.5,
        error_kind=error_kind,
    )


def test_query_completes_router_adapter_evaluator_record_flow() -> None:
    selector_inputs: list[str] = []

    def selector(query_text: str) -> RoutingDecision:
        selector_inputs.append(query_text)
        return decision("dev-model-a")

    evaluator = MockQualityEvaluator(
        {("请总结这段材料。", "固定回答"): 0.9}, threshold=0.8)
    runner = BenchmarkRunner(
        selector, {"dev-model-a": make_adapter("dev-model-a")}, evaluator)
    record = runner.run_one("query-1", "请总结这段材料。")

    assert selector_inputs == ["请总结这段材料。"]
    assert record.query_id == "query-1"
    assert record.selected_model == "dev-model-a"
    assert record.response_text == "固定回答"
    assert record.cost == 0.25
    assert record.model_latency_ms == 18.5
    assert record.status == "success"
    assert record.measurement_kind == "simulated"
    assert record.quality_score == 0.9
    assert record.quality_pass is True


def test_only_selected_model_is_invoked() -> None:
    selected = RecordingAdapter(make_adapter("dev-model-b", "selected"))
    other = RecordingAdapter(make_adapter("dev-model-a", "not selected"))
    runner = BenchmarkRunner(
        lambda query: decision("dev-model-b"),
        {"dev-model-a": other, "dev-model-b": selected},
    )

    record = runner.run_one("query-1", "普通问题")

    assert record.response_text == "selected"
    assert len(selected.requests) == 1
    assert other.requests == []


def test_evaluator_receives_only_query_and_response_text() -> None:
    evaluator = RecordingQualityEvaluator()
    runner = BenchmarkRunner(
        lambda query: decision("dev-model-b"),
        {"dev-model-b": make_adapter("dev-model-b", "selected response")},
        evaluator,
    )

    record = runner.run_one("query-1", "original query")

    assert evaluator.calls == [("original query", "selected response", None)]
    assert record.quality_score == 0.9
    assert record.quality_pass is True


def test_unknown_model_raises_with_failure_record() -> None:
    configured = RecordingAdapter(make_adapter("dev-model-a"))
    runner = BenchmarkRunner(
        lambda query: decision("dev-model-unknown"), {"dev-model-a": configured})

    with pytest.raises(BenchmarkRunError) as captured:
        runner.run_one("query-1", "普通问题")

    assert captured.value.record.status == "unknown_model"
    assert captured.value.record.error_kind == "unknown_model"
    assert captured.value.record.selected_model == "dev-model-unknown"
    assert configured.requests == []


@pytest.mark.parametrize("error_kind", ["timeout", "error"])
def test_adapter_error_is_recorded_and_propagated(error_kind: str) -> None:
    runner = BenchmarkRunner(
        lambda query: decision("dev-model-a"),
        {"dev-model-a": make_adapter("dev-model-a", error_kind=error_kind)},
    )

    with pytest.raises(BenchmarkRunError) as captured:
        runner.run_one("query-1", "普通问题")

    record = captured.value.record
    assert record.status == "adapter_error"
    assert record.error_kind == error_kind
    assert record.response_text is None
    assert record.measurement_kind == "not_available"
    assert isinstance(captured.value.__cause__, AdapterError)


def test_single_query_result_is_reproducible() -> None:
    selector: Callable[[str], RoutingDecision] = lambda query: decision(
        "dev-model-a", fallback_used=True)
    evaluator = MockQualityEvaluator({("相同问题", "固定回答"): 0.7})
    runner = BenchmarkRunner(
        selector, {"dev-model-a": make_adapter("dev-model-a")}, evaluator)

    first = runner.run_one("query-1", "相同问题")
    second = runner.run_one("query-1", "相同问题")

    assert first == second
    assert first.request_id.startswith("request-")


def test_runner_without_evaluator_remains_compatible() -> None:
    runner = BenchmarkRunner(
        lambda query: decision("dev-model-a"),
        {"dev-model-a": make_adapter("dev-model-a")},
    )

    record = runner.run_one("query-1", "普通问题")

    assert record.status == "success"
    assert record.quality_score is None
    assert record.quality_pass is None
