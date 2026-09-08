"""List the models available to the configured DeepSeek API key."""

import os
import sys

from openai import OpenAI


def main() -> int:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key is None or not api_key.strip():
        print(
            "DeepSeek model listing failed: DEEPSEEK_API_KEY is not set",
            file=sys.stderr,
        )
        return 1

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
        response = client.models.list()
    except Exception as exc:
        print(
            f"DeepSeek model listing failed ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        return 1

    model_ids = sorted(model.id for model in response.data)
    if not model_ids:
        print("No models were returned by the DeepSeek API.")
        return 0

    for model_id in model_ids:
        print(model_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
