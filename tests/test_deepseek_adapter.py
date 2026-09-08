"""DeepSeek adapter behavior without real network access."""

from types import SimpleNamespace

import pytest
from httpx2 import Request, Response
from openai import APIStatusError, APITimeoutError

import zhiheng_router.adapters.deepseek as deepseek_module
from zhiheng_router.adapters import (
    AdapterError,
    DeepSeekAdapter,
    DeepSeekPricing,
    ModelAdapter,
    ModelRequest,
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


def pricing() -> DeepSeekPricing:
    return DeepSeekPricing(
        cache_hit_input_per_million=0.1,
        cache_miss_input_per_million=0.2,
        output_per_million=0.4,
    )


def api_response() -> object:
    return SimpleNamespace(
        id="response-123",
        model="deepseek-test-version",
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="DeepSeek response"),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            prompt_cache_hit_tokens=60,
            prompt_cache_miss_tokens=40,
        ),
    )


def test_mock_api_response_populates_model_response(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = iter([10.0, 10.0125])
    monkeypatch.setattr(deepseek_module, "perf_counter", lambda: next(clock))
    completions = FakeCompletions(response=api_response())
    adapter = DeepSeekAdapter(
        "deepseek-test", pricing=pricing(), api_key="test-key",
        client=FakeClient(completions),
    )
    request = ModelRequest("request-1", "query-1", "Hello")

    response = adapter.invoke(request)

    assert isinstance(adapter, ModelAdapter)
    assert response.request_id == "request-1"
    assert response.model_id == adapter.model_id == "deepseek-test"
    assert response.text == "DeepSeek response"
    assert response.input_units == 100
    assert response.output_units == 20
    assert response.cost == pytest.approx(0.000022)
    assert response.model_latency_ms == pytest.approx(12.5)
    assert response.finish_reason == "stop"
    assert response.measurement_kind == "measured"
    assert response.metadata["provider"] == "deepseek"
    assert response.metadata["prompt_cache_hit_tokens"] == 60
    assert response.metadata["prompt_cache_miss_tokens"] == 40
    assert "api_key" not in response.metadata
    assert completions.calls == [{
        "model": "deepseek-test",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
        "timeout": 60.0,
    }]


def test_environment_key_timeout_and_base_url_configure_client(
        monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    fake_client = FakeClient(FakeCompletions(response=api_response()))

    def create_client(**kwargs: object) -> object:
        captured.update(kwargs)
        return fake_client

    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-key")
    monkeypatch.setattr(deepseek_module, "_create_client", create_client)

    adapter = DeepSeekAdapter(
        "deepseek-test",
        pricing=pricing(),
        timeout=15.0,
        base_url="https://deepseek.example.test",
    )

    assert adapter.model_id == "deepseek-test"
    assert captured == {
        "api_key": "environment-key",
        "base_url": "https://deepseek.example.test",
        "timeout": 15.0,
    }


def test_client_uses_timeout_and_disables_sdk_retries(
        monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    def create_openai(**kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(deepseek_module, "OpenAI", create_openai)

    client = deepseek_module._create_client(
        api_key="test-key",
        base_url="https://api.deepseek.com",
        timeout=7.5,
    )

    assert client is sentinel
    assert captured == {
        "api_key": "test-key",
        "base_url": "https://api.deepseek.com",
        "timeout": 7.5,
        "max_retries": 0,
    }


def test_api_error_is_converted_to_adapter_error() -> None:
    adapter = DeepSeekAdapter(
        "deepseek-test",
        pricing=pricing(),
        api_key="test-key",
        client=FakeClient(FakeCompletions(error=RuntimeError("provider unavailable"))),
    )
    request = ModelRequest("request-1", "query-1", "Hello")

    with pytest.raises(AdapterError, match="DeepSeek invocation failed") as captured:
        adapter.invoke(request)

    assert captured.value.model_id == "deepseek-test"
    assert captured.value.request_id == "request-1"
    assert captured.value.error_kind == "error"
    assert isinstance(captured.value.__cause__, RuntimeError)


def test_timeout_is_converted_to_adapter_error_with_original_message() -> None:
    timeout = APITimeoutError(Request("POST", "https://api.deepseek.com/chat/completions"))
    adapter = DeepSeekAdapter(
        "deepseek-test",
        pricing=pricing(),
        api_key="test-key",
        timeout=3.5,
        client=FakeClient(FakeCompletions(error=timeout)),
    )

    with pytest.raises(AdapterError, match="Request timed out") as captured:
        adapter.invoke(ModelRequest("request-1", "query-1", "Hello"))

    assert captured.value.model_id == "deepseek-test"
    assert captured.value.request_id == "request-1"
    assert captured.value.error_kind == "timeout"
    assert captured.value.__cause__ is timeout


def test_provider_api_error_is_converted_with_original_message() -> None:
    request = Request("POST", "https://api.deepseek.com/chat/completions")
    response = Response(503, request=request)
    api_error = APIStatusError(
        "DeepSeek service unavailable",
        response=response,
        body={"error": "unavailable"},
    )
    adapter = DeepSeekAdapter(
        "deepseek-test",
        pricing=pricing(),
        api_key="test-key",
        client=FakeClient(FakeCompletions(error=api_error)),
    )

    with pytest.raises(AdapterError, match="DeepSeek service unavailable") as captured:
        adapter.invoke(ModelRequest("request-1", "query-1", "Hello"))

    assert captured.value.model_id == "deepseek-test"
    assert captured.value.request_id == "request-1"
    assert captured.value.error_kind == "api_error"
    assert captured.value.__cause__ is api_error
