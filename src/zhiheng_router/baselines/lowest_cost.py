"""Baseline that selects the lowest estimated cost."""

from collections.abc import Sequence

from ..schemas import CandidateEstimate, ModelCandidate, RoutingDecision
from .strongest import _active_estimates, _decision


class LowestCostBaseline:
    def select(self, candidates: Sequence[ModelCandidate],
               estimates: Sequence[CandidateEstimate],
               query_text: str | None = None) -> RoutingDecision:
        rows = _active_estimates(candidates, estimates)
        selected = min(rows, key=lambda estimate: (estimate.estimated_cost,
                                                   estimate.model_id))
        return _decision(selected, rows, "BASELINE_LOWEST_ESTIMATED_COST")
