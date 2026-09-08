"""Minimal four-strategy experiment comparison."""

from collections.abc import Callable

from experiments.run_comparison import SELECTOR_NAMES, run_comparison
from zhiheng_router.adapters import MockModelAdapter
from zhiheng_router.baselines import (
    LowestCostBaseline,
    RuleBasedBaseline,
    StrongestBaseline,
)
from zhiheng_router.evaluators import MockQualityEvaluator
from zhiheng_router.metrics import MetricsResult
from zhiheng_router.router import route
from zhiheng_router.schemas import (
    CandidateEstimate,
    ModelCandidate,
    RoutingDecision,
    RoutingPolicy,
)


QUERIES = [
    ("query-1", "请调试 Python 代码"),
    ("query-2", "求解这个数学方程"),
]
CANDIDATES = [
    ModelCandidate("dev-model-a"),
    ModelCandidate("dev-model-b"),
    ModelCandidate("dev-model-c"),
]
ESTIMATES = [
    CandidateEstimate("dev-model-a", 0.95, 5.0, 30.0),
    CandidateEstimate("dev-model-b", 0.85, 1.0, 20.0),
    CandidateEstimate("dev-model-c", 0.70, 3.0, 10.0),
]


def selectors() -> dict[str, Callable[[str], RoutingDecision]]:
    strongest = StrongestBaseline()
    lowest_cost = LowestCostBaseline()
    rule_based = RuleBasedBaseline({
        "code": "dev-model-b",
        "math": "dev-model-c",
        "general": "dev-model-a",
    })
    return {
        "router": lambda query: route(CANDIDATES, ESTIMATES, RoutingPolicy()),
        "strongest": lambda query: strongest.select(CANDIDATES, ESTIMATES, query),
        "lowest_cost": lambda query: lowest_cost.select(CANDIDATES, ESTIMATES, query),
        "rule_based": lambda query: rule_based.select(CANDIDATES, ESTIMATES, query),
    }


def adapters() -> dict[str, MockModelAdapter]:
    return {
        model_id: MockModelAdapter(
            model_id,
            response_text=f"response from {model_id}",
            cost=cost,
            model_latency_ms=latency,
        )
        for model_id, cost, latency in (
            ("dev-model-a", 5.0, 30.0),
            ("dev-model-b", 1.0, 20.0),
            ("dev-model-c", 3.0, 10.0),
        )
    }


def evaluator() -> MockQualityEvaluator:
    return MockQualityEvaluator(default_score=0.9)


def test_all_four_selectors_execute_for_every_query() -> None:
    calls = {name: [] for name in SELECTOR_NAMES}
    tracked: dict[str, Callable[[str], RoutingDecision]] = {}
    for name, selector in selectors().items():
        def capture(query: str, *, strategy: str = name,
                    select: Callable[[str], RoutingDecision] = selector
                    ) -> RoutingDecision:
            calls[strategy].append(query)
            return select(query)

        tracked[name] = capture

    run_comparison(QUERIES, tracked, adapters(), evaluator())

    assert calls == {name: [query for _, query in QUERIES] for name in SELECTOR_NAMES}


def test_output_contains_four_metrics_results() -> None:
    result = run_comparison(QUERIES, selectors(), adapters(), evaluator())

    assert tuple(result) == SELECTOR_NAMES
    assert all(isinstance(metrics, MetricsResult) for metrics in result.values())
    assert all(metrics.total_count == len(QUERIES) for metrics in result.values())
    assert result["strongest"].average_cost == 5.0
    assert result["lowest_cost"].average_cost == 1.0


def test_same_comparison_input_is_reproducible() -> None:
    first = run_comparison(QUERIES, selectors(), adapters(), evaluator())
    second = run_comparison(QUERIES, selectors(), adapters(), evaluator())
    assert first == second
