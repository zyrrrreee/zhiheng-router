"""Deterministic, network-free adapters for integration and runner tests."""

from collections.abc import Mapping
from typing import Any, Literal

from .base import AdapterError, ModelRequest, ModelResponse


class MockModelAdapter:
    """Return one fixed response and fixed telemetry, or one configured error."""

    def __init__(self, model_id: str, *, response_text: str,
                 input_units: int | float = 0, output_units: int | float = 0,
                 cost: float = 0.0, model_latency_ms: float = 0.0,
                 finish_reason: str = "stop", measurement_kind: str = "simulated",
                 metadata: Mapping[str, Any] | None = None,
                 error_kind: Literal["timeout", "error"] | None = None) -> None:
        if error_kind not in (None, "timeout", "error"):
            raise ValueError("error_kind must be 'timeout', 'error', or None")
        validated = ModelResponse(
            request_id="mock-configuration",
            model_id=model_id,
            text=response_text,
            input_units=input_units,
            output_units=output_units,
            cost=cost,
            model_latency_ms=model_latency_ms,
            finish_reason=finish_reason,
            measurement_kind=measurement_kind,
            metadata={} if metadata is None else metadata,
        )
        self._model_id = validated.model_id
        self._response_text = validated.text
        self._input_units = validated.input_units
        self._output_units = validated.output_units
        self._cost = validated.cost
        self._model_latency_ms = validated.model_latency_ms
        self._finish_reason = validated.finish_reason
        self._measurement_kind = validated.measurement_kind
        self._metadata = dict(validated.metadata)
        self._error_kind = error_kind

    @property
    def model_id(self) -> str:
        return self._model_id

    def invoke(self, request: ModelRequest) -> ModelResponse:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a ModelRequest")
        request.__post_init__()
        if self._error_kind is not None:
            raise AdapterError(
                f"mock model {self._error_kind}",
                model_id=self.model_id,
                request_id=request.request_id,
                error_kind=self._error_kind,
            )
        return ModelResponse(
            request_id=request.request_id,
            model_id=self.model_id,
            text=self._response_text,
            input_units=self._input_units,
            output_units=self._output_units,
            cost=self._cost,
            model_latency_ms=self._model_latency_ms,
            finish_reason=self._finish_reason,
            measurement_kind=self._measurement_kind,
            metadata=self._metadata,
        )
