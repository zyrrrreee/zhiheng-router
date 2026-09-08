"""Minimal synchronous execution of one Query through a selected adapter."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real

from zhiheng_router.adapters import AdapterError, ModelAdapter, ModelRequest
from zhiheng_router.evaluators import QualityEvaluator, QualityResult
from zhiheng_router.schemas import RoutingDecision


_STATUSES = {"success", "unknown_model", "adapter_error"}


def _nonblank(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value


def _optional_nonnegative_number(value: object, name: str) -> None:
    if value is None:
        return
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be a finite nonnegative number or None")


def _optional_quality_score(value: object) -> None:
    if value is None:
        return
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("quality_score must be a finite number in [0,1] or None")


@dataclass(frozen=True)
class RunRecord:
    query_id: str
    request_id: str
    query_text: str
    selected_model: str
    fallback_used: bool
    response_text: str | None
    cost: float | None
    model_latency_ms: float | None
    status: str
    measurement_kind: str
    quality_score: float | None = None
    quality_pass: bool | None = None
    error_kind: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        for name in ("query_id", "request_id", "query_text", "selected_model",
                     "status", "measurement_kind"):
            _nonblank(getattr(self, name), name)
        if not isinstance(self.fallback_used, bool):
            raise ValueError("fallback_used must be a boolean")
        if self.status not in _STATUSES:
            raise ValueError(f"unknown run status: {self.status}")
        if self.response_text is not None and not isinstance(self.response_text, str):
            raise ValueError("response_text must be a string or None")
        _optional_nonnegative_number(self.cost, "cost")
        _optional_nonnegative_number(self.model_latency_ms, "model_latency_ms")
        _optional_quality_score(self.quality_score)
        if self.quality_pass is not None and not isinstance(self.quality_pass, bool):
            raise ValueError("quality_pass must be a boolean or None")
        if (self.quality_score is None) != (self.quality_pass is None):
            raise ValueError("quality_score and quality_pass must both be present or both be None")
        if self.status == "success":
            if self.response_text is None or self.cost is None or self.model_latency_ms is None:
                raise ValueError("a successful run requires response text, cost, and latency")
            if self.error_kind is not None or self.error_message is not None:
                raise ValueError("a successful run cannot contain error details")
        else:
            if self.response_text is not None or self.cost is not None or self.model_latency_ms is not None:
                raise ValueError("a failed run cannot contain response measurements")
            if self.quality_score is not None or self.quality_pass is not None:
                raise ValueError("a failed run cannot contain quality results")
            _nonblank(self.error_kind, "error_kind")
            _nonblank(self.error_message, "error_message")


class BenchmarkRunError(RuntimeError):
    """A failed single-query run with its failure record attached."""

    def __init__(self, record: RunRecord) -> None:
        super().__init__(record.error_message)
        self.record = record


class BenchmarkRunner:
    def __init__(self, selector: Callable[[str], RoutingDecision],
                 adapters: Mapping[str, ModelAdapter],
                 quality_evaluator: QualityEvaluator | None = None) -> None:
        if not callable(selector):
            raise TypeError("selector must be callable")
        if not isinstance(adapters, Mapping) or not adapters:
            raise ValueError("adapters must be a nonempty mapping")
        copied: dict[str, ModelAdapter] = {}
        for model_id, adapter in adapters.items():
            _nonblank(model_id, "adapter model ID")
            if not isinstance(adapter, ModelAdapter):
                raise TypeError(f"adapter for {model_id} does not implement ModelAdapter")
            if adapter.model_id != model_id:
                raise ValueError(f"adapter key/model_id mismatch: {model_id} != {adapter.model_id}")
            copied[model_id] = adapter
        if quality_evaluator is not None and not isinstance(quality_evaluator, QualityEvaluator):
            raise TypeError("quality_evaluator must implement QualityEvaluator")
        self._selector = selector
        self._adapters = copied
        self._quality_evaluator = quality_evaluator

    @staticmethod
    def _request_id(query_id: str, query_text: str) -> str:
        canonical = json.dumps(
            [query_id, query_text], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return f"request-{hashlib.sha256(canonical).hexdigest()[:16]}"

    def _failure_record(self, *, query_id: str, request_id: str, query_text: str,
                        decision: RoutingDecision, status: str, error_kind: str,
                        error_message: str) -> RunRecord:
        return RunRecord(
            query_id=query_id,
            request_id=request_id,
            query_text=query_text,
            selected_model=decision.selected_model,
            fallback_used=decision.fallback_used,
            response_text=None,
            cost=None,
            model_latency_ms=None,
            status=status,
            measurement_kind="not_available",
            error_kind=error_kind,
            error_message=error_message,
        )

    def run_one(self, query_id: str, query_text: str) -> RunRecord:
        _nonblank(query_id, "query_id")
        _nonblank(query_text, "query_text")
        request_id = self._request_id(query_id, query_text)

        decision = self._selector(query_text)
        if not isinstance(decision, RoutingDecision):
            raise TypeError("selector must return a RoutingDecision")

        adapter = self._adapters.get(decision.selected_model)
        if adapter is None:
            record = self._failure_record(
                query_id=query_id,
                request_id=request_id,
                query_text=query_text,
                decision=decision,
                status="unknown_model",
                error_kind="unknown_model",
                error_message=f"no adapter configured for selected model: {decision.selected_model}",
            )
            raise BenchmarkRunError(record)

        request = ModelRequest(request_id, query_id, query_text)
        try:
            response = adapter.invoke(request)
        except AdapterError as exc:
            record = self._failure_record(
                query_id=query_id,
                request_id=request_id,
                query_text=query_text,
                decision=decision,
                status="adapter_error",
                error_kind=exc.error_kind,
                error_message=str(exc),
            )
            raise BenchmarkRunError(record) from exc

        if response.request_id != request.request_id:
            raise ValueError("adapter response request_id does not match request")
        if response.model_id != decision.selected_model:
            raise ValueError("adapter response model_id does not match selected model")
        quality: QualityResult | None = None
        if self._quality_evaluator is not None:
            quality = self._quality_evaluator.evaluate(query_text, response.text)
            if not isinstance(quality, QualityResult):
                raise TypeError("quality_evaluator must return a QualityResult")
        return RunRecord(
            query_id=query_id,
            request_id=request_id,
            query_text=query_text,
            selected_model=decision.selected_model,
            fallback_used=decision.fallback_used,
            response_text=response.text,
            cost=response.cost,
            model_latency_ms=response.model_latency_ms,
            status="success",
            measurement_kind=response.measurement_kind,
            quality_score=quality.quality_score if quality is not None else None,
            quality_pass=quality.quality_pass if quality is not None else None,
        )
