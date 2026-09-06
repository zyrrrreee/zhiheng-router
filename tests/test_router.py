"""Decision contract tests use hand-calculated outcomes, independent of the predictor."""

from itertools import permutations

import pytest

from zhiheng_router.router import route
from zhiheng_router.schemas import CandidateEstimate as E, ModelCandidate as M, RoutingPolicy as P


def decide(rows, policy=P(), candidates=None):
    return route(candidates if candidates is not None else [M(r.model_id) for r in rows], rows, policy)


def test_multiple_qualified_cost_precedes_quality_and_latency():
    result = decide([E("a", .99, 10, 10), E("b", .81, 5, 100), E("c", .7, 1, 1)])
    assert result.selected_model == "b"
    assert result.qualified_models == ("a", "b")
    assert not result.fallback_used


def test_only_one_qualified_and_inclusive_probability_threshold():
    result = decide([E("a", .8, 10, 100), E("b", .79999, 1, 1)])
    assert result.selected_model == "a"
    assert result.decision_reason == "QUALIFIED_ONLY_MODEL"


def test_no_qualified_uses_max_probability_not_actual_quality():
    result = decide([E("a", .79, 10, 100), E("b", .7, 1, 1)])
    assert result.selected_model == "a"
    assert result.fallback_used
    assert result.qualified_models == ()
    assert result.decision_reason == "FALLBACK_MAX_PASS_PROBABILITY"


def test_same_cost_compares_latency():
    assert decide([E("a", .95, 3, 100), E("b", .85, 3, 50)]).selected_model == "b"


def test_tolerance_boundary_is_inclusive_and_anchored_to_global_minimum():
    rows = [E("a", .9, 10, 100), E("b", .9, 10.5, 50), E("c", .9, 11, 1)]
    assert decide(rows, P(.8, .5)).selected_model == "b"
    assert decide(rows, P(.8, 0)).selected_model == "a"


def test_cost_just_outside_tolerance_cannot_win_by_latency():
    rows = [E("a", .9, 10, 100), E("b", .9, 10.5001, 1)]
    assert decide(rows, P(.8, .5)).selected_model == "a"


def test_latency_tie_within_band_uses_cost_before_id():
    rows = [E("a", .9, 10.5, 50), E("z", .85, 10, 50)]
    assert decide(rows, P(.8, 1)).selected_model == "z"


def test_exact_ties_use_id_and_are_permutation_invariant():
    rows = [E("c", .9, 3, 10), E("a", .8, 3, 10), E("b", .95, 3, 10)]
    results = [decide(list(order)) for order in permutations(rows)]
    assert all(result == results[0] for result in results)
    assert results[0].selected_model == "a"


@pytest.mark.parametrize("rows, expected", [
    ([E("a", .7, 2, 10), E("b", .7, 1, 100)], "b"),
    ([E("a", .7, 1, 20), E("b", .7, 1, 10)], "b"),
    ([E("a", .7, 1, 10), E("b", .7, 1, 10)], "a"),
])
def test_fallback_probability_ties(rows, expected):
    result = decide(rows)
    assert result.selected_model == expected
    assert result.fallback_used


def test_disabled_is_a_static_filter():
    result = decide([E("a", .9, 10, 100)], candidates=[M("a"), M("b", False)])
    assert result.selected_model == "a"
    assert len(result.candidate_estimates) == 1


@pytest.mark.parametrize("candidates, estimates, message", [
    ([], [], "NoEligibleModel"),
    ([M("a", False)], [], "NoEligibleModel"),
    ([M("a"), M("a")], [E("a", .9, 1, 1)], "duplicate"),
    ([M("a")], [E("a", .9, 1, 1), E("a", .9, 1, 1)], "duplicate"),
    ([M("a")], [E("unknown", .9, 1, 1)], "unknown"),
    ([M("a"), M("b")], [E("a", .9, 1, 1)], "missing"),
])
def test_invalid_candidate_contract(candidates, estimates, message):
    with pytest.raises(ValueError, match=message):
        route(candidates, estimates, P())


@pytest.mark.parametrize("field", ["pass_probability", "estimated_cost", "estimated_latency_ms"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -0.01, True])
def test_invalid_estimate_values(field, value):
    args = dict(model_id="a", pass_probability=.9, estimated_cost=1, estimated_latency_ms=1)
    args[field] = value
    with pytest.raises(ValueError):
        decide([E(**args)])


def test_probability_above_one():
    with pytest.raises(ValueError):
        E("a", 1.001, 1, 1)


@pytest.mark.parametrize("value", [-.1, 1.1, float("nan"), float("inf")])
def test_invalid_probability_threshold(value):
    with pytest.raises(ValueError):
        P(value, 0)


@pytest.mark.parametrize("value", [-.1, float("nan"), float("inf")])
def test_invalid_tolerance(value):
    with pytest.raises(ValueError):
        P(.8, value)


def test_zero_cost_is_valid_not_missing():
    assert decide([E("a", .9, 0, 100), E("b", .9, 1, 1)]).selected_model == "a"
