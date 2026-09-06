"""File validation and group splits, before any learned processing."""

from collections import Counter
from collections.abc import Sequence
import hashlib
import json
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit

from .schemas import (FeatureConfig, HistoricalRecord, ModelCandidate, RoutingPolicy,
                      TrainingConfig, finite_number)


def read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    candidates = [ModelCandidate(**row) for row in config["models"]]
    ids = [c.model_id for c in candidates]
    if len(ids) != len(set(ids)) or not any(c.enabled for c in candidates):
        raise ValueError("model configuration needs unique IDs and an enabled model")
    FeatureConfig(**config["features"])
    TrainingConfig(**config["training"])
    RoutingPolicy(**config["policy"])
    active = {c.model_id for c in candidates if c.enabled}
    if not set(config["rules"].values()) <= active:
        raise ValueError("rules reference unknown or disabled models")
    if set(config["rules"]) != {"code", "math", "summary", "translation", "qa"}:
        raise ValueError("rules must cover code, math, summary, translation, qa")
    return config


def load_history(path: Path) -> list[HistoricalRecord]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            records.append(HistoricalRecord(**json.loads(line)))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid history at line {line_number}: {exc}") from exc
    validate_history(records)
    return records


def validate_history(records: Sequence[HistoricalRecord], model_ids: Sequence[str] | None = None,
                     *, full_coverage: bool = False) -> None:
    if not records:
        raise ValueError("history is empty")
    calls: set[str] = set()
    queries: dict[str, tuple[str, str, str]] = {}
    text_ids: dict[str, str] = {}
    pairs: Counter = Counter()
    expected = set(model_ids) if model_ids is not None else {r.model_id for r in records}
    if not expected:
        raise ValueError("no model IDs")
    for row in records:
        row.__post_init__()
        if row.call_id in calls:
            raise ValueError(f"duplicate call_id: {row.call_id}")
        calls.add(row.call_id)
        identity = (row.query, row.group_id, row.task_type)
        if row.query_id in queries and queries[row.query_id] != identity:
            raise ValueError(f"inconsistent query_id identity: {row.query_id}")
        queries[row.query_id] = identity
        text = row.query.strip()
        if text in text_ids and text_ids[text] != row.query_id:
            raise ValueError("identical Query has different query_id; merge its group before splitting")
        text_ids[text] = row.query_id
        if row.model_id not in expected:
            raise ValueError(f"unknown model in history: {row.model_id}")
        pairs[row.query_id, row.model_id] += 1
    if full_coverage and any(pairs[q, m] != 1 for q in queries for m in expected):
        raise ValueError("full coverage requires exactly one observation per Query x Model")


def load_development_dataset(data_path: Path, manifest_path: Path,
                             model_ids: Sequence[str]) -> tuple[list[HistoricalRecord], dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"source_kind", "data_version", "generator_version", "seed", "quality_score_range",
                "cost_unit", "latency_unit", "coverage_kind", "quality_definition", "sha256"}
    if not required <= manifest.keys():
        raise ValueError(f"manifest is missing {sorted(required - manifest.keys())}")
    if (manifest["source_kind"] != "synthetic" or manifest["coverage_kind"] != "full"
            or manifest["quality_score_range"] != [0.0, 1.0] or manifest["latency_unit"] != "ms"):
        raise ValueError("this MVP runner requires synthetic full-coverage data, quality [0,1], latency ms")
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("dataset SHA256 does not match manifest")
    records = load_history(data_path)
    validate_history(records, model_ids, full_coverage=True)
    return records, manifest


def group_split(records: Sequence[HistoricalRecord], *, train_fraction: float = 0.6,
                validation_fraction: float = 0.2, seed: int = 20260906
                ) -> dict[str, list[HistoricalRecord]]:
    validate_history(records)
    finite_number(train_fraction, "train_fraction", maximum=1)
    finite_number(validation_fraction, "validation_fraction", maximum=1)
    if not (train_fraction > 0 and validation_fraction > 0
            and train_fraction + validation_fraction < 1):
        raise ValueError("train, validation and test fractions must all be positive")
    if type(seed) is not int or seed < 0:
        raise ValueError("split seed must be a nonnegative integer")
    # Split one entry per unique Query, never individual model observations.
    unique = {r.query_id: r for r in records}
    ids = sorted(unique)
    groups = [unique[q].group_id for q in ids]
    if len(set(groups)) < 3:
        raise ValueError("at least three groups are required")
    train_idx, held_idx = next(GroupShuffleSplit(
        n_splits=1, train_size=train_fraction, random_state=seed,
    ).split(ids, groups=groups))
    held_groups = [groups[i] for i in held_idx]
    val_idx, test_idx = next(GroupShuffleSplit(
        n_splits=1, train_size=validation_fraction / (1 - train_fraction), random_state=seed,
    ).split(held_idx, groups=held_groups))
    membership = {
        "train": {ids[i] for i in train_idx},
        "validation": {ids[held_idx[i]] for i in val_idx},
        "test": {ids[held_idx[i]] for i in test_idx},
    }
    return {name: [r for r in records if r.query_id in query_ids]
            for name, query_ids in membership.items()}


def split_summary(splits: dict[str, list[HistoricalRecord]]) -> dict:
    result = {}
    for name, rows in splits.items():
        unique = {r.query_id: r for r in rows}
        result[name] = {
            "observation_count": len(rows), "query_count": len(unique),
            "group_count": len({r.group_id for r in rows}),
            "query_ids": sorted(unique), "group_ids": sorted({r.group_id for r in rows}),
            "task_query_counts": dict(sorted(Counter(r.task_type for r in unique.values()).items())),
        }
    return result
