"""Basic aggregate metrics over successful and failed RunRecord values."""

import pytest

from zhiheng_router.metrics import MetricsCalculator
from zhiheng_router.runner import RunRecord


def success(query_id: str, *, quality_score: float | None,
            quality_pass: bool | None, cost: float, latency: float,
            fallback: bool = False) -> RunRecord:
    return RunRecord(
        query_id=query_id,
        request_id=f"request-{query_id}",
        query_text=f"text-{query_id}",
        selected_model="dev-model-a",
        fallback_used=fallback,
        response_text="response",
        cost=cost,
        model_latency_ms=latency,
        status="success",
        measurement_kind="simulated",
        quality_score=quality_score,
        quality_pass=quality_pass,
    )


def failure(query_id: str) -> RunRecord:
    return RunRecord(
        query_id=query_id,
        request_id=f"request-{query_id}",
        query_text=f"text-{query_id}",
        selected_model="dev-model-a",
        fallback_used=False,
        response_text=None,
        cost=None,
        model_latency_ms=None,
        status="adapter_error",
        measurement_kind="not_available",
        error_kind="timeout",
        error_message="mock timeout",
    )


def sample_successes() -> list[RunRecord]:
    return [
        success("1", quality_score=0.9, quality_pass=True, cost=1.0, latency=10.0),
        success("2", quality_score=0.7, quality_pass=False, cost=3.0,
                latency=30.0, fallback=True),
    ]


def test_multiple_success_records_compute_expected_metrics() -> None:
    result = MetricsCalculator().compute(sample_successes())

    assert result.total_count == 2
    assert result.success_count == 2
    assert result.failure_count == 0
    assert result.success_rate == 1.0
    assert result.quality_pass_rate == 0.5
    assert result.average_quality == pytest.approx(0.8)
    assert result.average_cost == 2.0
    assert result.average_latency_ms == 20.0
    assert result.fallback_rate == 0.5


def test_failure_record_does_not_affect_success_averages() -> None:
    expected = MetricsCalculator().compute(sample_successes())
    result = MetricsCalculator().compute([*sample_successes(), failure("3")])

    assert result.total_count == 3
    assert result.success_count == 2
    assert result.failure_count == 1
    assert result.success_rate == pytest.approx(2 / 3)
    assert result.quality_pass_rate == expected.quality_pass_rate
    assert result.average_quality == expected.average_quality
    assert result.average_cost == expected.average_cost
    assert result.average_latency_ms == expected.average_latency_ms
    assert result.fallback_rate == expected.fallback_rate


def test_missing_quality_is_excluded_from_quality_metrics() -> None:
    records = [
        success("1", quality_score=None, quality_pass=None, cost=1.0, latency=10.0),
        success("2", quality_score=None, quality_pass=None, cost=3.0, latency=30.0),
    ]

    result = MetricsCalculator().compute(records)

    assert result.quality_pass_rate is None
    assert result.average_quality is None
    assert result.average_cost == 2.0
    assert result.average_latency_ms == 20.0


def test_empty_list_returns_zero_counts_and_undefined_rates() -> None:
    result = MetricsCalculator().compute([])

    assert result.total_count == 0
    assert result.success_count == 0
    assert result.failure_count == 0
    assert result.success_rate is None
    assert result.quality_pass_rate is None
    assert result.average_quality is None
    assert result.average_cost is None
    assert result.average_latency_ms is None
    assert result.fallback_rate is None


def test_repeated_compute_is_reproducible() -> None:
    records = [*sample_successes(), failure("3")]
    calculator = MetricsCalculator()

    assert calculator.compute(records) == calculator.compute(records)
