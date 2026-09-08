"""Uniform selector-style baseline behavior."""

import pytest

from zhiheng_router.baselines import (
    BaselineSelector,
    LowestCostBaseline,
    RuleBasedBaseline,
    StrongestBaseline,
)
from zhiheng_router.schemas import CandidateEstimate, ModelCandidate, RoutingDecision


@pytest.fixture
def candidates() -> list[ModelCandidate]:
    return [
        ModelCandidate("dev-model-a"),
        ModelCandidate("dev-model-b"),
        ModelCandidate("dev-model-c"),
    ]


@pytest.fixture
def estimates() -> list[CandidateEstimate]:
    return [
        CandidateEstimate("dev-model-a", 0.95, 5.0, 30.0),
        CandidateEstimate("dev-model-b", 0.80, 1.0, 20.0),
        CandidateEstimate("dev-model-c", 0.70, 3.0, 10.0),
    ]


def rule_baseline() -> RuleBasedBaseline:
    return RuleBasedBaseline({
        "code": "dev-model-b",
        "math": "dev-model-c",
        "general": "dev-model-a",
    })


@pytest.mark.parametrize("baseline,query_text", [
    (StrongestBaseline(), None),
    (LowestCostBaseline(), None),
    (rule_baseline(), "请编写 Python 代码"),
])
def test_all_baselines_return_routing_decision(
        baseline: BaselineSelector, query_text: str | None,
        candidates: list[ModelCandidate],
        estimates: list[CandidateEstimate]) -> None:
    result = baseline.select(candidates, estimates, query_text)
    assert isinstance(result, RoutingDecision)
    assert isinstance(baseline, BaselineSelector)
    assert result.fallback_used is False


def test_strongest_selects_highest_pass_probability(
        candidates: list[ModelCandidate],
        estimates: list[CandidateEstimate]) -> None:
    result = StrongestBaseline().select(candidates, estimates)
    assert result.selected_model == "dev-model-a"
    assert result.decision_reason == "BASELINE_HIGHEST_PASS_PROBABILITY"


def test_lowest_cost_selects_minimum_estimated_cost(
        candidates: list[ModelCandidate],
        estimates: list[CandidateEstimate]) -> None:
    result = LowestCostBaseline().select(candidates, estimates)
    assert result.selected_model == "dev-model-b"
    assert result.decision_reason == "BASELINE_LOWEST_ESTIMATED_COST"


@pytest.mark.parametrize("query_text,expected,reason", [
    ("请调试这段 Python 代码", "dev-model-b", "BASELINE_RULE_CODE"),
    ("求解这个数学方程", "dev-model-c", "BASELINE_RULE_MATH"),
    ("请介绍这个概念", "dev-model-a", "BASELINE_RULE_GENERAL"),
])
def test_rule_based_selects_keyword_model(
        query_text: str, expected: str, reason: str,
        candidates: list[ModelCandidate],
        estimates: list[CandidateEstimate]) -> None:
    result = rule_baseline().select(candidates, estimates, query_text)
    assert result.selected_model == expected
    assert result.decision_reason == reason


@pytest.mark.parametrize("baseline,query_text", [
    (StrongestBaseline(), "same query"),
    (LowestCostBaseline(), "same query"),
    (rule_baseline(), "请调试 Python 代码"),
])
def test_same_input_is_reproducible(
        baseline: BaselineSelector, query_text: str,
        candidates: list[ModelCandidate],
        estimates: list[CandidateEstimate]) -> None:
    first = baseline.select(candidates, estimates, query_text)
    second = baseline.select(candidates, estimates, query_text)
    assert first == second
