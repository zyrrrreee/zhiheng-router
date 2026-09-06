"""Four deployable baselines using training statistics or predefined Query rules."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .estimates import estimates_from_probabilities
from .features import task_type_hint
from .router import route
from .schemas import ModelCandidate, ModelStatistics, RoutingDecision, RoutingPolicy, nonempty


@dataclass(frozen=True)
class BaselineSelection:
    selected_model: str
    # Fixed/rule baselines do not enforce the probability gate: fallback is N/A.
    fallback_used: bool | None = None


class Baselines:
    def __init__(self, candidates: Sequence[ModelCandidate], statistics: Mapping[str, ModelStatistics],
                 policy: RoutingPolicy, rules: Mapping[str, str]):
        self.candidates = tuple(candidates)
        self.rules = dict(rules)
        active = {c.model_id for c in candidates if c.enabled}
        if not active or not active <= statistics.keys():
            raise ValueError("baseline training statistics missing for enabled candidates")
        if set(rules) != {"code", "math", "summary", "translation", "qa"} or not set(rules.values()) <= active:
            raise ValueError("rules must cover all hints and reference enabled models")
        self.strongest_model = min(active, key=lambda m: (-statistics[m].mean_quality, m))
        self.lowest_cost_model = min(active, key=lambda m: (statistics[m].mean_cost, m))
        estimates = estimates_from_probabilities({m: statistics[m].pass_rate for m in active}, statistics)
        # No current Query is used: one global selection from training pass rates.
        self.global_decision = route(candidates, estimates, policy)

    def strongest(self, query: str) -> BaselineSelection:
        nonempty(query, "query")
        return BaselineSelection(self.strongest_model)

    def lowest_cost(self, query: str) -> BaselineSelection:
        nonempty(query, "query")
        return BaselineSelection(self.lowest_cost_model)

    def rule_based(self, query: str) -> BaselineSelection:
        return BaselineSelection(self.rules[task_type_hint(query)])

    def global_historical(self, query: str) -> RoutingDecision:
        nonempty(query, "query")
        return self.global_decision
