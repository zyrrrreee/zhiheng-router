"""Make one small real request to each selected DeepSeek model."""

import os
import sys

from zhiheng_router.adapters import (
    AdapterError,
    DeepSeekAdapter,
    DeepSeekPricing,
    ModelRequest,
)


MODEL_PRICING = {
    "deepseek-v4-flash": DeepSeekPricing(
        cache_hit_input_per_million=0.0028,
        cache_miss_input_per_million=0.14,
        output_per_million=0.28,
    ),
    "deepseek-v4-pro": DeepSeekPricing(
        cache_hit_input_per_million=0.003625,
        cache_miss_input_per_million=0.435,
        output_per_million=0.87,
    ),
}


def main() -> int:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key is None or not api_key.strip():
        print("DeepSeek smoke test failed: DEEPSEEK_API_KEY is not set", file=sys.stderr)
        return 1

    failed = False
    for index, (model_id, pricing) in enumerate(MODEL_PRICING.items(), start=1):
        adapter = DeepSeekAdapter(
            model_id,
            pricing=pricing,
            api_key=api_key,
            timeout=60.0,
        )
        request = ModelRequest(
            request_id=f"deepseek-real-smoke-{index}",
            query_id="shared-smoke-query",
            query_text="用一句话说明为什么模型路由可以降低推理成本。",
        )

        print(f"\n[{model_id}]")
        try:
            response = adapter.invoke(request)
        except AdapterError as exc:
            failed = True
            print(f"status: failed ({exc.error_kind})")
            print(f"error: {exc}")
            continue

        print("status: success")
        print(f"model_id: {response.model_id}")
        print(f"response text: {response.text}")
        print(f"latency_ms: {response.model_latency_ms:.3f}")
        print(f"input tokens: {response.input_units}")
        print(f"output tokens: {response.output_units}")
        print(f"cost: {response.cost:.10f} USD")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
