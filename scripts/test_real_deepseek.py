"""Run one real DeepSeek API request through the unified adapter."""

import os

from zhiheng_router.adapters import (
    DeepSeekAdapter,
    DeepSeekPricing,
    ModelRequest,
)


MODEL_ID = "deepseek-chat"


def main() -> None:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key is None or not api_key.strip():
        raise SystemExit("DEEPSEEK_API_KEY is not set")

    adapter = DeepSeekAdapter(
        MODEL_ID,
        api_key=api_key,
        timeout=60.0,
        pricing=DeepSeekPricing(
            cache_hit_input_per_million=0.07,
            cache_miss_input_per_million=0.27,
            output_per_million=1.10,
        ),
    )
    response = adapter.invoke(ModelRequest(
        request_id="deepseek-smoke-request-1",
        query_id="deepseek-smoke-query-1",
        query_text="请用一句话说明什么是大模型路由。",
    ))

    print(f"model_id: {response.model_id}")
    print(f"response text: {response.text}")
    print(f"latency_ms: {response.model_latency_ms:.3f}")
    print(f"input tokens: {response.input_units}")
    print(f"output tokens: {response.output_units}")
    print(f"cost: {response.cost:.10f} USD")


if __name__ == "__main__":
    main()
