"""Training-only global means; these do not estimate live service state."""

from collections.abc import Mapping, Sequence
from statistics import fmean

from .schemas import CandidateEstimate, HistoricalRecord, ModelStatistics, finite_number
from .predictor import QualityPredictor


def summarize_models(records: Sequence[HistoricalRecord], model_ids: Sequence[str],
                     quality_score_threshold: float) -> dict[str, ModelStatistics]:
    finite_number(quality_score_threshold, "quality_score_threshold", maximum=1)
    if not model_ids or len(set(model_ids)) != len(model_ids):
        raise ValueError("statistics model IDs must be nonempty and unique")
    for row in records:
        row.__post_init__()
    result = {}
    for model_id in sorted(model_ids):
        rows = [r for r in records if r.model_id == model_id]
        if not rows:
            raise ValueError(f"missing training observations: {model_id}")
        result[model_id] = ModelStatistics(
            model_id, fmean(r.cost for r in rows), fmean(r.latency_ms for r in rows), len(rows),
            fmean(r.quality_score for r in rows),
            fmean(r.quality_score >= quality_score_threshold for r in rows),
        )
    return result


def estimates_from_probabilities(probabilities: Mapping[str, float],
                                statistics: Mapping[str, ModelStatistics]
                                ) -> tuple[CandidateEstimate, ...]:
    rows = []
    for model_id, probability in sorted(probabilities.items()):
        if model_id not in statistics:
            raise ValueError(f"missing training statistics: {model_id}")
        stats = statistics[model_id]
        rows.append(CandidateEstimate(model_id, probability, stats.mean_cost, stats.mean_latency_ms))
    return tuple(rows)


def estimate_candidates(query: str, model_ids: Sequence[str], predictor: QualityPredictor,
                        statistics: Mapping[str, ModelStatistics]) -> tuple[CandidateEstimate, ...]:
    return estimates_from_probabilities(predictor.predict_pass_probabilities(query, model_ids), statistics)
