"""Minimal aggregate metrics for a list of RunRecord values."""

from dataclasses import dataclass
import math
from numbers import Real
from statistics import fmean

from zhiheng_router.runner import RunRecord


def _optional_number(value: object, name: str, *, maximum: float | None = None) -> None:
    if value is None:
        return
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value < 0
            or maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside the allowed range")


@dataclass(frozen=True)
class MetricsResult:
    total_count: int
    success_count: int
    failure_count: int
    success_rate: float | None
    quality_pass_rate: float | None
    average_quality: float | None
    average_cost: float | None
    average_latency_ms: float | None
    fallback_rate: float | None

    def __post_init__(self) -> None:
        for name in ("total_count", "success_count", "failure_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.total_count != self.success_count + self.failure_count:
            raise ValueError("total_count must equal success_count + failure_count")
        for name in ("success_rate", "quality_pass_rate", "average_quality", "fallback_rate"):
            _optional_number(getattr(self, name), name, maximum=1.0)
        _optional_number(self.average_cost, "average_cost")
        _optional_number(self.average_latency_ms, "average_latency_ms")


def _mean(values: list[float]) -> float | None:
    return fmean(values) if values else None


class MetricsCalculator:
    def compute(self, records: list[RunRecord]) -> MetricsResult:
        if not isinstance(records, list):
            raise TypeError("records must be a list of RunRecord values")
        if any(not isinstance(record, RunRecord) for record in records):
            raise TypeError("records must contain only RunRecord values")

        successful = [record for record in records if record.status == "success"]
        quality_records = [record for record in successful if record.quality_score is not None]
        total_count = len(records)
        success_count = len(successful)
        return MetricsResult(
            total_count=total_count,
            success_count=success_count,
            failure_count=total_count - success_count,
            success_rate=success_count / total_count if total_count else None,
            quality_pass_rate=_mean([
                float(record.quality_pass) for record in quality_records
            ]),
            average_quality=_mean([
                record.quality_score for record in quality_records
                if record.quality_score is not None
            ]),
            average_cost=_mean([
                record.cost for record in successful if record.cost is not None
            ]),
            average_latency_ms=_mean([
                record.model_latency_ms for record in successful
                if record.model_latency_ms is not None
            ]),
            fallback_rate=_mean([float(record.fallback_used) for record in successful]),
        )
