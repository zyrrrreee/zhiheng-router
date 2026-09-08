"""Compare four query selectors on one shared in-memory Query batch."""

from collections.abc import Callable, Mapping

from zhiheng_router.adapters import ModelAdapter
from zhiheng_router.evaluators import QualityEvaluator
from zhiheng_router.metrics import MetricsCalculator, MetricsResult
from zhiheng_router.runner import BatchRunner, BenchmarkRunner
from zhiheng_router.schemas import RoutingDecision


SELECTOR_NAMES = ("router", "strongest", "lowest_cost", "rule_based")


def run_comparison(
        queries: list[tuple[str, str]],
        selectors: Mapping[str, Callable[[str], RoutingDecision]],
        adapters: Mapping[str, ModelAdapter],
        evaluator: QualityEvaluator) -> dict[str, MetricsResult]:
    if not isinstance(selectors, Mapping) or set(selectors) != set(SELECTOR_NAMES):
        raise ValueError(
            "selectors must contain router, strongest, lowest_cost, and rule_based")

    calculator = MetricsCalculator()
    results: dict[str, MetricsResult] = {}
    for name in SELECTOR_NAMES:
        single_runner = BenchmarkRunner(selectors[name], adapters, evaluator)
        records = BatchRunner(single_runner).run_many(queries)
        results[name] = calculator.compute(records)
    return results
