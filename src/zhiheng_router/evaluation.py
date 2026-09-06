"""Read held-out simulated observations only after a policy has selected a model."""

from collections.abc import Callable, Sequence
from statistics import fmean
from time import perf_counter_ns

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

from .baselines import BaselineSelection
from .data import validate_history
from .predictor import QualityPredictor
from .schemas import HistoricalRecord, RoutingDecision, finite_number

Selection = BaselineSelection | RoutingDecision


def probability_diagnostics(labels: Sequence[int], probabilities: Sequence[float], bins: int) -> dict:
    if type(bins) is not int or bins < 1:
        raise ValueError("probability_bins must be a positive integer")
    if len(labels) == 0 or len(labels) != len(probabilities) or any(y not in (0, 1) for y in labels):
        raise ValueError("probability diagnostics require paired binary labels and predictions")
    for probability in probabilities:
        finite_number(probability, "pass_probability", maximum=1)
    y, p = np.asarray(labels), np.asarray(probabilities)
    bucket = np.minimum((p * bins).astype(int), bins - 1)
    diagnostics = []
    for i in range(bins):
        mask = bucket == i
        count = int(mask.sum())
        diagnostics.append({
            "lower_inclusive": i / bins, "upper": (i + 1) / bins,
            "upper_inclusive": i == bins - 1, "count": count,
            "mean_predicted_probability": float(p[mask].mean()) if count else None,
            "observed_pass_rate": float(y[mask].mean()) if count else None,
        })
    return {"observation_count": len(y), "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier_score": float(brier_score_loss(y, p)), "bins": diagnostics}


def evaluate_predictor(predictor: QualityPredictor, records: Sequence[HistoricalRecord], bins: int) -> dict:
    validate_history(records)
    queries = {r.query_id: r.query for r in records}
    ids = sorted(queries)
    predictions = dict(zip(ids, predictor.predict_many([queries[q] for q in ids])))
    per_model = {}
    all_labels, all_probabilities = [], []
    for model_id in sorted(predictor.models):
        rows = [r for r in records if r.model_id == model_id]
        if not rows:
            raise ValueError(f"no held-out observations for {model_id}")
        labels = [int(r.quality_score >= predictor.quality_score_threshold) for r in rows]
        probabilities = [predictions[r.query_id][model_id] for r in rows]
        per_model[model_id] = probability_diagnostics(labels, probabilities, bins)
        all_labels.extend(labels)
        all_probabilities.extend(probabilities)
    return {"measurement_kind": "simulated", "calibration_model": None,
            "query_count": len(ids), "per_model": per_model,
            "pooled": probability_diagnostics(all_labels, all_probabilities, bins)}


def _outcome_metrics(rows: list[dict]) -> dict:
    def pass_rate(subset: list[dict]) -> float | None:
        return fmean(r["quality_pass"] for r in subset) if subset else None

    flags = [r["fallback_used"] for r in rows]
    applicable = all(flag is not None for flag in flags)
    if any(flag is None for flag in flags) and not all(flag is None for flag in flags):
        raise ValueError("one policy must have consistent fallback semantics")
    normal = [r for r in rows if r["fallback_used"] is False]
    fallback = [r for r in rows if r["fallback_used"] is True]
    return {
        "measurement_kind": "simulated", "query_count": len(rows),
        "quality_pass_rate": pass_rate(rows),
        "average_quality": fmean(r["observed_quality_score"] for r in rows),
        "average_cost": fmean(r["observed_cost"] for r in rows),
        "average_latency_ms": fmean(r["observed_latency_ms"] for r in rows),
        "fallback_rate": len(fallback) / len(rows) if applicable else None,
        "normal_query_count": len(normal) if applicable else None,
        "fallback_query_count": len(fallback) if applicable else None,
        "normal_routing_actual_pass_rate": pass_rate(normal),
        "fallback_actual_pass_rate": pass_rate(fallback),
    }


def evaluate_policy(records: Sequence[HistoricalRecord], select: Callable[[str], Selection],
                    quality_score_threshold: float, *, measure_runtime: bool = False,
                    probability_bins: int = 5) -> dict:
    validate_history(records)
    finite_number(quality_score_threshold, "quality_score_threshold", maximum=1)
    queries = {r.query_id: r.query for r in records}
    observations = {}
    for record in records:
        pair = (record.query_id, record.model_id)
        if pair in observations:
            raise ValueError("evaluation requires one observation per Query x Model")
        observations[pair] = record
    rows, durations = [], []
    for query_id in sorted(queries):
        start = perf_counter_ns()
        choice = select(queries[query_id])  # Only Query text crosses the policy boundary.
        elapsed_ms = (perf_counter_ns() - start) / 1_000_000
        if measure_runtime:
            durations.append(elapsed_ms)
        key = (query_id, choice.selected_model)
        if key not in observations:
            raise ValueError(f"unobserved selected Query x Model: {key}; cannot impute an outcome")
        actual = observations[key]  # Outcome access follows selection, never prediction substitution.
        rows.append({
            "query_id": query_id, "task_type": actual.task_type,
            "selected_model": choice.selected_model, "fallback_used": choice.fallback_used,
            "predicted_pass_probability": choice.pass_probability if isinstance(choice, RoutingDecision) else None,
            "decision_reason": choice.decision_reason if isinstance(choice, RoutingDecision) else None,
            "observed_quality_score": actual.quality_score, "observed_cost": actual.cost,
            "observed_latency_ms": actual.latency_ms,
            "quality_pass": actual.quality_score >= quality_score_threshold,
        })
    result = {"metrics": _outcome_metrics(rows), "decisions": rows,
              "by_task_type": {task: _outcome_metrics([r for r in rows if r["task_type"] == task])
                               for task in sorted({r["task_type"] for r in rows})}}
    if all(r["predicted_pass_probability"] is not None for r in rows):
        result["selected_probability_diagnostics"] = probability_diagnostics(
            [int(r["quality_pass"]) for r in rows], [r["predicted_pass_probability"] for r in rows], probability_bins)
    if durations:
        result["runtime"] = {
            "measurement_kind": "measured", "scope": "serial Query feature extraction + prediction + estimates + router; excludes model invocation, training, load and outcome lookup",
            "query_count": len(durations), "router_latency_mean_ms": fmean(durations),
            "router_latency_p50_ms": float(np.percentile(durations, 50)),
            "router_latency_p95_ms": float(np.percentile(durations, 95)),
            "serial_router_throughput_qps": 1000 * len(durations) / sum(durations),
        }
    return result
