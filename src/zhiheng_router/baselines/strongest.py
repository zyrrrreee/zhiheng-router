"""Baseline that selects the highest predicted pass probability."""

from collections.abc import Sequence

from ..schemas import CandidateEstimate, ModelCandidate, RoutingDecision


def _active_estimates(candidates: Sequence[ModelCandidate],
                      estimates: Sequence[CandidateEstimate]
                      ) -> tuple[CandidateEstimate, ...]:
    candidate_ids = [candidate.model_id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("duplicate model id in candidates")
    for candidate in candidates:
        candidate.__post_init__()
    active = {candidate.model_id for candidate in candidates if candidate.enabled}
    if not active:
        raise ValueError("no enabled baseline candidates")

    by_id: dict[str, CandidateEstimate] = {}
    for estimate in estimates:
        estimate.__post_init__()
        if estimate.model_id not in candidate_ids:
            raise ValueError(f"unknown model in estimates: {estimate.model_id}")
        if estimate.model_id in by_id:
            raise ValueError("duplicate model id in estimates")
        by_id[estimate.model_id] = estimate
    missing = active - by_id.keys()
    if missing:
        raise ValueError(f"missing estimate: {sorted(missing)}")
    return tuple(by_id[model_id] for model_id in sorted(active))


def _decision(selected: CandidateEstimate, rows: tuple[CandidateEstimate, ...],
              reason: str) -> RoutingDecision:
    return RoutingDecision(
        selected_model=selected.model_id,
        pass_probability=selected.pass_probability,
        estimated_cost=selected.estimated_cost,
        estimated_latency_ms=selected.estimated_latency_ms,
        fallback_used=False,
        decision_reason=reason,
        qualified_models=(),
        candidate_estimates=rows,
    )


class StrongestBaseline:
    def select(self, candidates: Sequence[ModelCandidate],
               estimates: Sequence[CandidateEstimate],
               query_text: str | None = None) -> RoutingDecision:
        rows = _active_estimates(candidates, estimates)
        selected = min(rows, key=lambda estimate: (-estimate.pass_probability,
                                                   estimate.model_id))
        return _decision(selected, rows, "BASELINE_HIGHEST_PASS_PROBABILITY")
