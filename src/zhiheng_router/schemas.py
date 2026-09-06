"""Small immutable records; units and probability semantics are explicit."""

from dataclasses import dataclass
import math
from numbers import Real


def nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")


def finite_number(value: float, name: str, *, minimum: float = 0.0,
                  maximum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside the allowed range")


@dataclass(frozen=True)
class HistoricalRecord:
    call_id: str
    query_id: str
    group_id: str
    query: str
    model_id: str
    quality_score: float
    cost: float
    latency_ms: float
    task_type: str

    def __post_init__(self) -> None:
        for name in ("call_id", "query_id", "group_id", "query", "model_id", "task_type"):
            nonempty(getattr(self, name), name)
        finite_number(self.quality_score, "quality_score", maximum=1.0)
        finite_number(self.cost, "cost")
        finite_number(self.latency_ms, "latency_ms")


@dataclass(frozen=True)
class ModelCandidate:
    model_id: str
    enabled: bool = True

    def __post_init__(self) -> None:
        nonempty(self.model_id, "model_id")
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean; it is not a health measurement")


@dataclass(frozen=True)
class CandidateEstimate:
    model_id: str
    pass_probability: float
    estimated_cost: float
    estimated_latency_ms: float

    def __post_init__(self) -> None:
        nonempty(self.model_id, "model_id")
        finite_number(self.pass_probability, "pass_probability", maximum=1.0)
        finite_number(self.estimated_cost, "estimated_cost")
        finite_number(self.estimated_latency_ms, "estimated_latency_ms")


@dataclass(frozen=True)
class RoutingPolicy:
    router_probability_threshold: float = 0.8
    cost_tolerance_abs: float = 0.0

    def __post_init__(self) -> None:
        finite_number(self.router_probability_threshold, "router_probability_threshold", maximum=1.0)
        finite_number(self.cost_tolerance_abs, "cost_tolerance_abs")


@dataclass(frozen=True)
class ModelStatistics:
    model_id: str
    mean_cost: float
    mean_latency_ms: float
    observation_count: int
    mean_quality: float
    pass_rate: float


@dataclass(frozen=True)
class RoutingDecision:
    selected_model: str
    pass_probability: float
    estimated_cost: float
    estimated_latency_ms: float
    fallback_used: bool
    decision_reason: str
    qualified_models: tuple[str, ...]
    candidate_estimates: tuple[CandidateEstimate, ...]


@dataclass(frozen=True)
class FeatureConfig:
    ngram_range: tuple[int, int] = (2, 4)
    max_features: int = 3000

    def __post_init__(self) -> None:
        if (len(self.ngram_range) != 2 or any(type(n) is not int for n in self.ngram_range)
                or not 1 <= self.ngram_range[0] <= self.ngram_range[1]):
            raise ValueError("ngram_range must be two increasing positive integers")
        if type(self.max_features) is not int or self.max_features < 1:
            raise ValueError("max_features must be a positive integer")


@dataclass(frozen=True)
class TrainingConfig:
    quality_score_threshold: float = 0.8
    C: float = 1.0
    max_iter: int = 1000
    seed: int = 20260906
    min_model_samples: int = 40

    def __post_init__(self) -> None:
        finite_number(self.quality_score_threshold, "quality_score_threshold", maximum=1.0)
        finite_number(self.C, "C")
        if self.C == 0:
            raise ValueError("C must be positive")
        for name in ("max_iter", "min_model_samples"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
