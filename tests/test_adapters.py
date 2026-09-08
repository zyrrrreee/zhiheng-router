"""Unified adapter contracts and deterministic mock behavior."""

import pytest

from zhiheng_router.adapters import (
    AdapterError,
    MockModelAdapter,
    ModelAdapter,
    ModelRequest,
    ModelResponse,
)


def request() -> ModelRequest:
    return ModelRequest("request-1", "query-1", "请总结这段材料。")


def test_request_and_response_schema() -> None:
    response = ModelResponse(
        request_id="request-1",
        model_id="dev-model-a",
        text="固定回答",
        input_units=12,
        output_units=4,
        cost=0.25,
        model_latency_ms=18.5,
        finish_reason="stop",
        measurement_kind="simulated",
        metadata={"provider": "mock"},
    )
    assert request().query_text == "请总结这段材料。"
    assert response.metadata == {"provider": "mock"}
    assert not hasattr(response, "quality_score")


def test_mock_adapter_is_deterministic() -> None:
    adapter = MockModelAdapter(
        "dev-model-a",
        response_text="固定回答",
        input_units=12,
        output_units=4,
        cost=0.25,
        model_latency_ms=18.5,
        metadata={"provider": "mock"},
    )
    first = adapter.invoke(request())
    second = adapter.invoke(request())
    assert first == second
    assert first is not second
    assert isinstance(adapter, ModelAdapter)


@pytest.mark.parametrize("field,value", [
    ("input_units", -1),
    ("output_units", float("inf")),
    ("cost", -0.01),
    ("cost", float("nan")),
    ("model_latency_ms", -0.01),
    ("model_latency_ms", float("inf")),
])
def test_response_rejects_invalid_numeric_values(field: str, value: float) -> None:
    values = {
        "request_id": "request-1",
        "model_id": "dev-model-a",
        "text": "固定回答",
        "input_units": 12,
        "output_units": 4,
        "cost": 0.25,
        "model_latency_ms": 18.5,
        "finish_reason": "stop",
        "measurement_kind": "simulated",
    }
    values[field] = value
    with pytest.raises(ValueError, match="finite nonnegative"):
        ModelResponse(**values)


@pytest.mark.parametrize("error_kind", ["timeout", "error"])
def test_mock_adapter_raises_structured_error(error_kind: str) -> None:
    adapter = MockModelAdapter(
        "dev-model-a", response_text="unused", error_kind=error_kind)
    with pytest.raises(AdapterError) as captured:
        adapter.invoke(request())
    assert captured.value.model_id == adapter.model_id
    assert captured.value.request_id == "request-1"
    assert captured.value.error_kind == error_kind


def test_mock_response_model_id_matches_adapter() -> None:
    adapter = MockModelAdapter("dev-model-f", response_text="固定回答")
    response = adapter.invoke(request())
    assert response.model_id == adapter.model_id == "dev-model-f"
