"""Evaluation uses selected observations, not the policy's predictions."""

import pytest

from zhiheng_router.baselines import BaselineSelection, Baselines
from zhiheng_router.estimates import summarize_models
from zhiheng_router.evaluation import evaluate_policy, probability_diagnostics
from zhiheng_router.router import route
from zhiheng_router.schemas import CandidateEstimate as E, HistoricalRecord as H, ModelCandidate as M, RoutingPolicy as P


def records():
    return [H("1-a", "1", "g1", "普通问题", "a", .2, 3, 20, "qa"),
            H("1-b", "1", "g1", "普通问题", "b", .9, 7, 40, "qa"),
            H("2-a", "2", "g2", "Python代码问题", "a", 1.0, 5, 30, "code"),
            H("2-b", "2", "g2", "Python代码问题", "b", .7, 9, 60, "code")]


def test_actual_results_and_fallback_are_not_predictions():
    inputs = []

    def select(query):
        inputs.append(query)
        if query == "普通问题":
            return route([M("a")], [E("a", .99, 1000, 1000)], P())
        return route([M("a")], [E("a", .4, 1000, 1000)], P())

    report = evaluate_policy(records(), select, .8)
    metrics = report["metrics"]
    assert inputs == ["普通问题", "Python代码问题"]
    assert metrics["quality_pass_rate"] == .5
    assert metrics["average_quality"] == .6
    assert metrics["average_cost"] == 4
    assert metrics["average_latency_ms"] == 25
    assert metrics["fallback_rate"] == .5
    assert metrics["normal_routing_actual_pass_rate"] == 0
    assert metrics["fallback_actual_pass_rate"] == 1
    assert metrics["measurement_kind"] == "simulated"


def test_missing_counterfactual_is_not_imputed():
    with pytest.raises(ValueError, match="unobserved selected"):
        evaluate_policy(records(), lambda q: BaselineSelection("unobserved"), .8)


def test_non_gate_baselines_report_fallback_as_not_applicable():
    metrics = evaluate_policy(records(), lambda q: BaselineSelection("a"), .8)["metrics"]
    assert metrics["fallback_rate"] is None
    assert metrics["normal_routing_actual_pass_rate"] is None
    assert metrics["fallback_actual_pass_rate"] is None


def test_empty_fallback_subset_is_null_not_zero():
    decision = route([M("a")], [E("a", .9, 1, 10)], P())
    metrics = evaluate_policy(records(), lambda q: decision, .8)["metrics"]
    assert metrics["fallback_rate"] == 0
    assert metrics["fallback_actual_pass_rate"] is None


def test_probability_bin_boundaries_and_known_brier_score():
    result = probability_diagnostics([0, 1, 0, 1], [0.0, .5, .5, 1.0], 2)
    assert result["brier_score"] == pytest.approx(.125)
    assert [b["count"] for b in result["bins"]] == [1, 3]
    assert result["bins"][1]["observed_pass_rate"] == pytest.approx(2/3)
    empty = probability_diagnostics([1], [1.0], 5)["bins"][0]
    assert empty["count"] == 0 and empty["observed_pass_rate"] is None


def test_baselines_are_train_defined_and_global_is_query_independent():
    stats = summarize_models(records(), ["a", "b"], .8)
    rules = {task: "a" for task in ("code", "math", "summary", "translation", "qa")}
    baseline = Baselines([M("a"), M("b")], stats, P(.5), rules)
    assert baseline.strongest("新问题").selected_model == "b"
    assert baseline.lowest_cost("新问题").selected_model == "a"
    assert baseline.rule_based("数学求解 x = 2").selected_model == "a"
    assert baseline.global_historical("新问题") == baseline.global_historical("完全不同的代码 Query")
    assert baseline.global_decision.selected_model == "a"


def test_statistics_quality_label_boundary_is_inclusive():
    row = H("c", "q", "g", "query", "a", .8, 0, 0, "qa")
    stats = summarize_models([row], ["a"], .8)["a"]
    assert stats.pass_rate == 1 and stats.mean_cost == 0


def test_runtime_is_separately_marked_measured():
    report = evaluate_policy(records(), lambda q: BaselineSelection("a"), .8, measure_runtime=True)
    assert report["metrics"]["measurement_kind"] == "simulated"
    assert report["runtime"]["measurement_kind"] == "measured"
    assert report["runtime"]["router_latency_mean_ms"] >= 0
