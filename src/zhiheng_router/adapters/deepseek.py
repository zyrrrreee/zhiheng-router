"""DeepSeek adapter using the provider's OpenAI-compatible chat API."""

from dataclasses import dataclass
import math
from numbers import Real
import os
from time import perf_counter
from typing import Any

from httpx2 import NetworkError, TimeoutException
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from .base import AdapterError, ModelRequest, ModelResponse


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def _nonblank(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value


def _nonnegative_number(value: object, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


def _positive_number(value: object, name: str) -> float:
    number = _nonnegative_number(value, name)
    if number == 0:
        raise ValueError(f"{name} must be positive")
    return number


def _usage_units(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class DeepSeekPricing:
    """USD prices per one million tokens, supplied by experiment config."""

    cache_hit_input_per_million: float
    cache_miss_input_per_million: float
    output_per_million: float

    def __post_init__(self) -> None:
        for name in (
            "cache_hit_input_per_million",
            "cache_miss_input_per_million",
            "output_per_million",
        ):
            object.__setattr__(
                self, name, _nonnegative_number(getattr(self, name), name))

    def calculate(self, *, cache_hit_input_units: int,
                  cache_miss_input_units: int, output_units: int) -> float:
        return (
            cache_hit_input_units * self.cache_hit_input_per_million
            + cache_miss_input_units * self.cache_miss_input_per_million
            + output_units * self.output_per_million
        ) / 1_000_000


def _create_client(*, api_key: str, base_url: str, timeout: float) -> Any:
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=0,
    )


def _error_kind(exc: Exception) -> str:
    if isinstance(exc, (APITimeoutError, TimeoutException, TimeoutError)):
        return "timeout"
    if isinstance(exc, (APIConnectionError, NetworkError, ConnectionError, OSError)):
        return "network_error"
    if isinstance(exc, APIError):
        return "api_error"
    return "error"


class DeepSeekAdapter:
    """Invoke one DeepSeek text model and return provider-independent telemetry."""

    def __init__(self, model_id: str, *, pricing: DeepSeekPricing,
                 api_key: str | None = None,
                 timeout: float = 60.0,
                 base_url: str = DEFAULT_DEEPSEEK_BASE_URL,
                 client: Any | None = None) -> None:
        self._model_id = _nonblank(model_id, "model_id")
        if not isinstance(pricing, DeepSeekPricing):
            raise TypeError("pricing must be a DeepSeekPricing")
        self._pricing = pricing
        self._timeout = _positive_number(timeout, "timeout")
        self._base_url = _nonblank(base_url, "base_url")

        resolved_api_key = api_key if api_key is not None else os.getenv("DEEPSEEK_API_KEY")
        resolved_api_key = _nonblank(resolved_api_key, "DeepSeek API key")
        self._client = client if client is not None else _create_client(
            api_key=resolved_api_key,
            base_url=self._base_url,
            timeout=self._timeout,
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    def invoke(self, request: ModelRequest) -> ModelResponse:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a ModelRequest")
        request.__post_init__()

        started_at = perf_counter()
        try:
            raw_response = self._client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": request.query_text}],
                stream=False,
                timeout=self._timeout,
            )
        except Exception as exc:
            elapsed_ms = (perf_counter() - started_at) * 1_000
            raise AdapterError(
                f"DeepSeek invocation failed after {elapsed_ms:.3f} ms: {exc}",
                model_id=self.model_id,
                request_id=request.request_id,
                error_kind=_error_kind(exc),
            ) from exc

        elapsed_ms = (perf_counter() - started_at) * 1_000
        try:
            return self._to_model_response(raw_response, request, elapsed_ms)
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise AdapterError(
                f"DeepSeek returned an invalid response: {exc}",
                model_id=self.model_id,
                request_id=request.request_id,
                error_kind="invalid_response",
            ) from exc

    def _to_model_response(self, raw_response: Any, request: ModelRequest,
                           elapsed_ms: float) -> ModelResponse:
        if not raw_response.choices:
            raise ValueError("choices must contain at least one item")
        choice = raw_response.choices[0]
        text = choice.message.content
        if not isinstance(text, str):
            raise ValueError("choice message content must be a string")
        finish_reason = _nonblank(choice.finish_reason, "finish_reason")

        usage = raw_response.usage
        input_units = _usage_units(usage.prompt_tokens, "prompt_tokens")
        output_units = _usage_units(usage.completion_tokens, "completion_tokens")
        raw_cache_hit = getattr(usage, "prompt_cache_hit_tokens", None)
        raw_cache_miss = getattr(usage, "prompt_cache_miss_tokens", None)
        cache_hit_units = (0 if raw_cache_hit is None else
                           _usage_units(raw_cache_hit, "prompt_cache_hit_tokens"))
        cache_miss_units = (
            input_units - cache_hit_units if raw_cache_miss is None else
            _usage_units(raw_cache_miss, "prompt_cache_miss_tokens"))
        if cache_hit_units + cache_miss_units != input_units:
            raise ValueError(
                "cache hit and miss input tokens must sum to prompt_tokens")

        cost = self._pricing.calculate(
            cache_hit_input_units=cache_hit_units,
            cache_miss_input_units=cache_miss_units,
            output_units=output_units,
        )
        metadata = {
            "provider": "deepseek",
            "provider_response_id": getattr(raw_response, "id", None),
            "provider_model": getattr(raw_response, "model", None),
            "prompt_cache_hit_tokens": cache_hit_units,
            "prompt_cache_miss_tokens": cache_miss_units,
            "total_tokens": input_units + output_units,
            "cost_currency": "USD",
            "cost_basis": "configured_prices_per_million_tokens",
            "base_url": self._base_url,
        }
        return ModelResponse(
            request_id=request.request_id,
            model_id=self.model_id,
            text=text,
            input_units=input_units,
            output_units=output_units,
            cost=cost,
            model_latency_ms=elapsed_ms,
            finish_reason=finish_reason,
            measurement_kind="measured",
            metadata=metadata,
        )
