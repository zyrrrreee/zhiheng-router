"""Independent response scoring through an OpenAI-compatible LLM judge."""

from collections.abc import Mapping
import json
import math
from numbers import Real
import os
from typing import Any

from .base import QualityResult


DEFAULT_JUDGE_BASE_URL = "https://api.deepseek.com"


def _nonblank(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value


def _score(value: object, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError(f"{name} must be a finite number in [0,1]")
    return float(value)


def _positive_number(value: object, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def _create_client(*, api_key: str, base_url: str, timeout: float) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - incomplete installation only
        raise RuntimeError(
            "the openai package is required to use LLMJudgeEvaluator") from exc
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


class LLMJudgeError(RuntimeError):
    """A stable error raised for provider or score-parsing failures."""

    def __init__(self, message: str, *, judge_model_id: str,
                 error_kind: str) -> None:
        super().__init__(_nonblank(message, "message"))
        self.judge_model_id = _nonblank(judge_model_id, "judge_model_id")
        self.error_kind = _nonblank(error_kind, "error_kind")


class LLMJudgeEvaluator:
    """Score only observable Query and response text with a separate model."""

    def __init__(self, judge_model_id: str, *, api_key: str | None = None,
                 threshold: float = 0.8, timeout: float = 60.0,
                 base_url: str = DEFAULT_JUDGE_BASE_URL,
                 client: Any | None = None) -> None:
        self._judge_model_id = _nonblank(judge_model_id, "judge_model_id")
        self._threshold = _score(threshold, "threshold")
        self._timeout = _positive_number(timeout, "timeout")
        self._base_url = _nonblank(base_url, "base_url")
        resolved_api_key = (api_key if api_key is not None else
                            os.getenv("LLM_JUDGE_API_KEY") or
                            os.getenv("DEEPSEEK_API_KEY"))
        resolved_api_key = _nonblank(resolved_api_key, "LLM Judge API key")
        self._client = client if client is not None else _create_client(
            api_key=resolved_api_key,
            base_url=self._base_url,
            timeout=self._timeout,
        )

    @property
    def judge_model_id(self) -> str:
        return self._judge_model_id

    def evaluate(self, query_text: str, response_text: str,
                 metadata: Mapping[str, Any] | None = None) -> QualityResult:
        query = _nonblank(query_text, "query_text")
        if not isinstance(response_text, str):
            raise ValueError("response_text must be a string")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError("metadata must be a mapping or None")

        try:
            raw_response = self._client.chat.completions.create(
                model=self.judge_model_id,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Evaluate whether an assistant response correctly and completely "
                            "answers the user query. Return one JSON object with exactly two "
                            "fields: quality_score, a number from 0 to 1, and reason, a short "
                            "explanation. Judge only the supplied query and response."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"User query:\n{query}\n\nAssistant response:\n{response_text}",
                    },
                ],
                response_format={"type": "json_object"},
                stream=False,
            )
        except Exception as exc:
            error_kind = ("timeout" if "timeout" in type(exc).__name__.lower()
                          else "error")
            raise LLMJudgeError(
                f"LLM Judge invocation failed: {exc}",
                judge_model_id=self.judge_model_id,
                error_kind=error_kind,
            ) from exc

        try:
            if not raw_response.choices:
                raise ValueError("choices must contain at least one item")
            content = raw_response.choices[0].message.content
            if not isinstance(content, str):
                raise ValueError("Judge response content must be a string")
            payload = json.loads(content)
            if not isinstance(payload, dict):
                raise ValueError("Judge response must be a JSON object")
            if set(payload) != {"quality_score", "reason"}:
                raise ValueError(
                    "Judge response must contain exactly quality_score and reason")
            score = _score(payload["quality_score"], "quality_score")
            reason = _nonblank(payload["reason"], "reason")
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
            raise LLMJudgeError(
                f"LLM Judge returned an invalid score response: {exc}",
                judge_model_id=self.judge_model_id,
                error_kind="invalid_response",
            ) from exc

        return QualityResult(
            quality_score=score,
            quality_pass=score >= self._threshold,
            measurement_kind="model_judge",
            metadata={
                "judge_model_id": self.judge_model_id,
                "judge_response_id": getattr(raw_response, "id", None),
                "threshold": self._threshold,
                "reason": reason,
            },
        )
