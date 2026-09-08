"""Provider-independent contracts for one model invocation."""

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Any, Protocol, runtime_checkable


def _nonblank(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value


def _nonnegative_number(value: object, name: str) -> None:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be a finite nonnegative number")


@dataclass(frozen=True)
class ModelRequest:
    request_id: str
    query_id: str
    query_text: str

    def __post_init__(self) -> None:
        _nonblank(self.request_id, "request_id")
        _nonblank(self.query_id, "query_id")
        _nonblank(self.query_text, "query_text")


@dataclass(frozen=True)
class ModelResponse:
    request_id: str
    model_id: str
    text: str
    input_units: int | float
    output_units: int | float
    cost: float
    model_latency_ms: float
    finish_reason: str
    measurement_kind: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonblank(self.request_id, "request_id")
        _nonblank(self.model_id, "model_id")
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        for name in ("input_units", "output_units", "cost", "model_latency_ms"):
            _nonnegative_number(getattr(self, name), name)
        _nonblank(self.finish_reason, "finish_reason")
        _nonblank(self.measurement_kind, "measurement_kind")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "metadata", dict(self.metadata))


class AdapterError(RuntimeError):
    """A model invocation failure with stable fields for a future runner."""

    def __init__(self, message: str, *, model_id: str, request_id: str | None = None,
                 error_kind: str = "error") -> None:
        super().__init__(_nonblank(message, "message"))
        self.model_id = _nonblank(model_id, "model_id")
        self.request_id = (_nonblank(request_id, "request_id")
                           if request_id is not None else None)
        self.error_kind = _nonblank(error_kind, "error_kind")


@runtime_checkable
class ModelAdapter(Protocol):
    @property
    def model_id(self) -> str:
        """Stable candidate model identifier served by this adapter."""
        ...

    def invoke(self, request: ModelRequest) -> ModelResponse:
        """Invoke the configured model once."""
        ...
