"""Pure quality-constraint -> cost -> latency selection, without model calls."""

from collections.abc import Sequence
from decimal import Decimal

from .schemas import CandidateEstimate, ModelCandidate, RoutingDecision, RoutingPolicy


def _within_cost_band(cost: float, minimum_cost: float, tolerance: float) -> bool:
    """Apply exact minimum-cost semantics at zero and decimal boundary semantics above zero."""
    if tolerance == 0.0:
        return cost == minimum_cost
    return Decimal(str(cost)) - Decimal(str(minimum_cost)) <= Decimal(str(tolerance))


def route(candidates: Sequence[ModelCandidate], estimates: Sequence[CandidateEstimate],
          policy: RoutingPolicy) -> RoutingDecision:
    policy.__post_init__()
    ids = [c.model_id for c in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate model id in candidates")
    for candidate in candidates:
        candidate.__post_init__()
    active = {c.model_id for c in candidates if c.enabled}
    if not active:
        raise ValueError("NoEligibleModel: no enabled candidates; this is not fallback")
    by_id: dict[str, CandidateEstimate] = {}
    for estimate in estimates:
        estimate.__post_init__()
        if estimate.model_id not in ids:
            raise ValueError(f"unknown model in estimates: {estimate.model_id}")
        if estimate.model_id in by_id:
            raise ValueError("duplicate model id in estimates")
        by_id[estimate.model_id] = estimate
    missing = active - by_id.keys()
    if missing:
        raise ValueError(f"missing estimate: {sorted(missing)}")
    rows = tuple(by_id[m] for m in sorted(active))
    qualified = tuple(e for e in rows if e.pass_probability >= policy.router_probability_threshold)
    if qualified:
        minimum_cost = min(e.estimated_cost for e in qualified)
        band = [e for e in qualified
                if _within_cost_band(e.estimated_cost, minimum_cost, policy.cost_tolerance_abs)]
        selected = min(band, key=lambda e: (e.estimated_latency_ms, e.estimated_cost, e.model_id))
        reason = ("QUALIFIED_ONLY_MODEL" if len(qualified) == 1 else
                  "QUALIFIED_MINIMUM_COST" if len(band) == 1 else
                  "QUALIFIED_LATENCY_COST_ID_TIEBREAK")
    else:
        selected = min(rows, key=lambda e: (-e.pass_probability, e.estimated_cost,
                                           e.estimated_latency_ms, e.model_id))
        reason = "FALLBACK_MAX_PASS_PROBABILITY"
    return RoutingDecision(
        selected.model_id, selected.pass_probability, selected.estimated_cost,
        selected.estimated_latency_ms, not bool(qualified), reason,
        tuple(e.model_id for e in qualified), rows,
    )
