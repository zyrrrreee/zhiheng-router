"""Simple Query-keyword baseline with code, math, and general rules."""

from collections.abc import Mapping, Sequence
import re

from ..schemas import CandidateEstimate, ModelCandidate, RoutingDecision
from .strongest import _active_estimates, _decision


_CODE = re.compile(r"\b(?:code|debug|function|python|sql)\b|代码|编程|调试", re.I)
_MATH = re.compile(r"\b(?:math|equation|calculate|prove)\b|数学|方程|计算|求解|证明", re.I)
_RULE_KEYS = {"code", "math", "general"}


class RuleBasedBaseline:
    def __init__(self, rules: Mapping[str, str]) -> None:
        if not isinstance(rules, Mapping) or set(rules) != _RULE_KEYS:
            raise ValueError("rules must define code, math, and general models")
        if any(not isinstance(model_id, str) or not model_id.strip()
               for model_id in rules.values()):
            raise ValueError("rule model IDs must be nonblank strings")
        self._rules = dict(rules)

    @staticmethod
    def _classify(query_text: str | None) -> str:
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("query_text is required for RuleBasedBaseline")
        if _CODE.search(query_text):
            return "code"
        if _MATH.search(query_text):
            return "math"
        return "general"

    def select(self, candidates: Sequence[ModelCandidate],
               estimates: Sequence[CandidateEstimate],
               query_text: str | None = None) -> RoutingDecision:
        rows = _active_estimates(candidates, estimates)
        category = self._classify(query_text)
        selected_model = self._rules[category]
        by_id = {estimate.model_id: estimate for estimate in rows}
        if selected_model not in by_id:
            raise ValueError(
                f"rule selected unknown or disabled model: {selected_model}")
        return _decision(
            by_id[selected_model], rows, f"BASELINE_RULE_{category.upper()}")
