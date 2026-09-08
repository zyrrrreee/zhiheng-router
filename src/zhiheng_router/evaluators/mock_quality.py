"""Deterministic simulated quality evaluation for local tests."""

from collections.abc import Mapping
import math
from numbers import Real
from typing import Any

from .base import QualityResult


def _score(value: object, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError(f"{name} must be a finite number in [0,1]")
    return float(value)


def _query_text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("query_text must be a nonblank string")
    return value


def _response_text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("response_text must be a string")
    return value


class MockQualityEvaluator:
    """Use an exact (Query, response) score mapping with a fixed default."""

    def __init__(self, scores: Mapping[tuple[str, str], float] | None = None, *,
                 default_score: float = 0.0, threshold: float = 0.8) -> None:
        if scores is not None and not isinstance(scores, Mapping):
            raise ValueError("scores must be a mapping")
        validated: dict[tuple[str, str], float] = {}
        for key, value in (scores or {}).items():
            if not isinstance(key, tuple) or len(key) != 2:
                raise ValueError("score keys must be (query_text, response_text) tuples")
            query_text, response_text = key
            pair = (_query_text(query_text), _response_text(response_text))
            validated[pair] = _score(value, "quality score")
        self._scores = validated
        self._default_score = _score(default_score, "default_score")
        self._threshold = _score(threshold, "threshold")

    def evaluate(self, query_text: str, response_text: str,
                 metadata: Mapping[str, Any] | None = None) -> QualityResult:
        pair = (_query_text(query_text), _response_text(response_text))
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError("metadata must be a mapping or None")
        matched = pair in self._scores
        score = self._scores.get(pair, self._default_score)
        return QualityResult(
            quality_score=score,
            quality_pass=score >= self._threshold,
            measurement_kind="simulated",
            metadata={
                "score_source": "fixed_mapping" if matched else "default",
                "threshold": self._threshold,
            },
        )
