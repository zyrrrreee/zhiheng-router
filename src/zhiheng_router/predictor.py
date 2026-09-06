"""Shared features plus one binary LogisticRegression per enabled model."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .data import validate_history
from .features import FeatureExtractor
from .schemas import (FeatureConfig, HistoricalRecord, ModelCandidate, ModelStatistics,
                      RoutingPolicy, TrainingConfig)


class QualityPredictor:
    def __init__(self, feature_config: FeatureConfig = FeatureConfig(),
                 training_config: TrainingConfig = TrainingConfig()):
        self.features = FeatureExtractor(feature_config)
        self.config = training_config
        self.quality_score_threshold = training_config.quality_score_threshold
        self.models: dict[str, LogisticRegression] = {}
        self.training_counts: dict[str, dict] = {}

    def fit(self, records: Sequence[HistoricalRecord], model_ids: Sequence[str]) -> "QualityPredictor":
        validate_history(records)
        if not model_ids or len(set(model_ids)) != len(model_ids):
            raise ValueError("training model IDs must be nonempty and unique")
        subsets = {m: [r for r in records if r.model_id == m] for m in sorted(model_ids)}
        counts = {}
        for model_id, rows in subsets.items():
            positive = sum(r.quality_score >= self.quality_score_threshold for r in rows)
            counts[model_id] = {"observation_count": len(rows), "positive_count": positive,
                                "negative_count": len(rows) - positive,
                                "positive_rate": positive / len(rows) if rows else None}
            if len(rows) < self.config.min_model_samples:
                raise ValueError(f"insufficient training samples for {model_id}: {counts[model_id]}")
            if positive == 0 or positive == len(rows):
                raise ValueError(f"single quality label class for {model_id}: {counts[model_id]}")
        unique = sorted({r.query for rows in subsets.values() for r in rows})
        # Fit into local objects so a failed refit cannot leave a half-updated predictor.
        features = FeatureExtractor(self.features.config).fit(unique)
        matrix = features.transform(unique)
        query_index = {q: i for i, q in enumerate(unique)}
        fitted = {}
        for model_id, rows in subsets.items():
            labels = np.array([int(r.quality_score >= self.quality_score_threshold) for r in rows])
            classifier = LogisticRegression(C=self.config.C, max_iter=self.config.max_iter,
                                            solver="lbfgs", random_state=self.config.seed,
                                            class_weight=None)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                classifier.fit(matrix[[query_index[r.query] for r in rows]], labels)
            fitted[model_id] = classifier
        self.features, self.models, self.training_counts = features, fitted, counts
        return self

    def predict_many(self, queries: Sequence[str], model_ids: Sequence[str] | None = None
                     ) -> list[dict[str, float]]:
        if not self.models:
            raise ValueError("QualityPredictor is not fitted")
        ids = sorted(self.models) if model_ids is None else list(model_ids)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("prediction model IDs must be nonempty and unique")
        unknown = set(ids) - self.models.keys()
        if unknown:
            raise ValueError(f"unknown model: {sorted(unknown)}")
        matrix = self.features.transform(queries)  # Once for all model predictors.
        columns = {m: self.models[m].predict_proba(matrix)[:, list(self.models[m].classes_).index(1)]
                   for m in ids}
        return [{m: float(columns[m][i]) for m in ids} for i in range(len(queries))]

    def predict_pass_probabilities(self, query: str, model_ids: Sequence[str] | None = None
                                   ) -> dict[str, float]:
        return self.predict_many([query], model_ids)[0]


@dataclass
class RoutingArtifact:
    predictor: QualityPredictor
    statistics: dict[str, ModelStatistics]
    candidates: tuple[ModelCandidate, ...]
    policy: RoutingPolicy
    dataset_manifest: dict
    config: dict
    environment: dict
    artifact_version: int = 1


def save_artifact(artifact: RoutingArtifact, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_artifact(path: Path) -> RoutingArtifact:
    """Read only locally generated, trusted joblib files."""
    result = joblib.load(path)
    if not isinstance(result, RoutingArtifact) or result.artifact_version != 1:
        raise ValueError("unsupported routing artifact")
    if not isinstance(result.predictor, QualityPredictor):
        raise ValueError("artifact predictor has an invalid type")
    try:
        configured_policy = RoutingPolicy(**result.config["policy"])
        configured_training = TrainingConfig(**result.config["training"])
        configured_features = FeatureConfig(**result.config["features"])
        configured_candidates = tuple(ModelCandidate(**row) for row in result.config["models"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"artifact contains invalid configuration: {exc}") from exc
    if result.policy != configured_policy:
        raise ValueError("artifact RoutingPolicy does not match config policy")
    thresholds = (
        result.predictor.quality_score_threshold,
        result.predictor.config.quality_score_threshold,
        configured_training.quality_score_threshold,
    )
    if any(threshold != thresholds[0] for threshold in thresholds[1:]):
        raise ValueError("artifact quality_score_threshold mismatch; retraining is required")
    if result.predictor.config != configured_training:
        raise ValueError("artifact predictor training config does not match artifact config")
    if result.predictor.features.config != configured_features:
        raise ValueError("artifact predictor feature config does not match artifact config")
    if result.candidates != configured_candidates:
        raise ValueError("artifact candidates do not match config models")

    candidate_ids = [candidate.model_id for candidate in result.candidates]
    if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("artifact candidate model IDs must be nonempty and unique")
    expected_ids = set(candidate_ids)
    model_sets = {
        "predictor models": set(result.predictor.models),
        "predictor training counts": set(result.predictor.training_counts),
        "training statistics": set(result.statistics),
    }
    for name, actual_ids in model_sets.items():
        if actual_ids != expected_ids:
            raise ValueError(
                f"artifact {name} do not match candidate model IDs; "
                f"missing={sorted(expected_ids - actual_ids)}, extra={sorted(actual_ids - expected_ids)}"
            )
    for model_id, statistics in result.statistics.items():
        if not isinstance(statistics, ModelStatistics) or statistics.model_id != model_id:
            raise ValueError(f"artifact training statistics are invalid for {model_id}")
    return result
