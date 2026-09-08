"""Provider- and router-independent quality evaluation contracts."""

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Any, Protocol, runtime_checkable


MEASUREMENT_KINDS = {"simulated", "human", "model_judge"}


@dataclass(frozen=True)
class QualityResult:
    quality_score: float
    quality_pass: bool
    measurement_kind: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        score = self.quality_score
        if (isinstance(score, bool) or not isinstance(score, Real)
                or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("quality_score must be a finite number in [0,1]")
        object.__setattr__(self, "quality_score", float(score))
        if not isinstance(self.quality_pass, bool):
            raise ValueError("quality_pass must be a boolean")
        if self.measurement_kind not in MEASUREMENT_KINDS:
            raise ValueError(
                "measurement_kind must be simulated, human, or model_judge")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "metadata", dict(self.metadata))


@runtime_checkable
class QualityEvaluator(Protocol):
    def evaluate(self, query_text: str, response_text: str,
                 metadata: Mapping[str, Any] | None = None) -> QualityResult:
        """Evaluate response quality without model or routing information."""
        ...
