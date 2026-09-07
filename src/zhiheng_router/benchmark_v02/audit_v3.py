"""R2 audits for difficulty confounding, evidence alignment, and payload skeletons."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
import hashlib
import math
import re
import unicodedata
from typing import Any

from .audit_v2 import audit_pilot_v2
from .schema import DIFFICULTY_BINS, DiagnosticSidecar, ObservableQueryRecord
from .validation_v3 import validate_capability_alignment_set


_MATERIAL_RE = re.compile(
    r"(?:给定材料|Provided material)\n(?P<material>.*?)(?="
    r"\n\n(?:任务|Task|交付形式|Deliverable)\n|\Z)",
    re.DOTALL,
)


def semantic_skeleton(text: str) -> str:
    match = _MATERIAL_RE.search(text)
    material = match.group("material") if match else text
    material = unicodedata.normalize("NFKC", material).lower()
    material = re.sub(r"\bv\d+(?:\.\d+)*\b", "<version>", material)
    material = re.sub(r"\b(?:[a-z]{1,8}[-_])?\d+\b", "<id>", material)
    material = re.sub(r"\d+(?:\.\d+)?", "#", material)
    material = re.sub(r"\b[xy]\b|\{[a-z_][a-z0-9_]*\}", "<var>", material)
    return re.sub(r"[^a-z_一-鿿<>]+", "", material)


def skeleton_diversity(observables: Sequence[ObservableQueryRecord],
                       sidecars: Sequence[DiagnosticSidecar]) -> dict[str, Any]:
    observable_by_id = {record.query_id: record for record in observables}
    by_frame: dict[str, list[DiagnosticSidecar]] = defaultdict(list)
    for sidecar in sidecars:
        by_frame[sidecar.semantic_frame_id].append(sidecar)
    frames: dict[str, Any] = {}
    for frame_id, records in sorted(by_frame.items()):
        groups: dict[str, list[str]] = defaultdict(list)
        for sidecar in records:
            skeleton = semantic_skeleton(observable_by_id[sidecar.query_id].query_text)
            digest = hashlib.sha256(skeleton.encode("utf-8")).hexdigest()[:16]
            groups[digest].append(sidecar.query_id)
        sizes = sorted((len(ids) for ids in groups.values()), reverse=True)
        frames[frame_id] = {
            "query_count": len(records),
            "distinct_skeleton_count": len(groups),
            "largest_skeleton_group_count": sizes[0],
            "largest_skeleton_share": round(sizes[0] / len(records), 6),
            "groups": [
                {"skeleton_id": digest, "query_ids": sorted(ids), "count": len(ids)}
                for digest, ids in sorted(groups.items())
            ],
        }
    return {
        "frame_count": len(frames),
        "minimum_distinct_skeleton_count": min(
            row["distinct_skeleton_count"] for row in frames.values()),
        "maximum_largest_skeleton_share": max(
            row["largest_skeleton_share"] for row in frames.values()),
        "frames": frames,
        "method": (
            "Extract provided material, normalize Unicode, numbers, IDs, versions, variables, "
            "whitespace, and punctuation, then group within each semantic frame."
        ),
    }


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    x_scale = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_scale = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    return 0.0 if not x_scale or not y_scale else numerator / (x_scale * y_scale)


def _ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and ordered[end][1] == ordered[position][1]:
            end += 1
        average_rank = (position + 1 + end) / 2
        for index, _ in ordered[position:end]:
            result[index] = average_rank
        position = end
    return result


def difficulty_capability_diagnostics(sidecars: Sequence[DiagnosticSidecar]) -> dict[str, Any]:
    difficulties = [record.difficulty for record in sidecars]
    counts = [len(record.active_capabilities) for record in sidecars]
    matrix = {
        difficulty: {
            str(count): sum(record.difficulty_bin == difficulty
                            and len(record.active_capabilities) == count for record in sidecars)
            for count in (2, 3, 4)
        }
        for difficulty in DIFFICULTY_BINS
    }
    by_count = {
        str(count): {
            difficulty: matrix[difficulty][str(count)] for difficulty in DIFFICULTY_BINS
        }
        for count in (2, 3, 4)
    }
    return {
        "difficulty_by_capability_count": matrix,
        "capability_count_by_difficulty": by_count,
        "pearson_correlation": round(_pearson(difficulties, counts), 6),
        "spearman_correlation": round(
            _pearson(_ranks(difficulties), _ranks([float(value) for value in counts])), 6),
        "capability_count_distribution": dict(sorted(Counter(counts).items())),
    }


def audit_pilot_v3(config: dict[str, Any], observables: Sequence[ObservableQueryRecord],
                   sidecars: Sequence[DiagnosticSidecar]) -> dict[str, Any]:
    report = audit_pilot_v2(config, observables, sidecars)
    confounding = difficulty_capability_diagnostics(sidecars)
    alignment = validate_capability_alignment_set(observables, sidecars)
    skeletons = skeleton_diversity(observables, sidecars)
    matrix = confounding["difficulty_by_capability_count"]
    by_count = confounding["capability_count_by_difficulty"]
    added_checks = {
        "difficulty_capability_correlation": (
            abs(confounding["pearson_correlation"]) <= 0.30
            and abs(confounding["spearman_correlation"]) <= 0.30
        ),
        "difficulty_bins_have_multiple_capability_counts": all(
            sum(value > 0 for value in matrix[difficulty].values()) >= 2
            for difficulty in DIFFICULTY_BINS
        ),
        "capability_counts_span_multiple_difficulty_bins": all(
            sum(value > 0 for value in by_count[str(count)].values()) >= 2
            for count in (2, 3, 4)
        ),
        "capability_evidence_alignment": alignment["alignment_valid_rate"] == 1.0,
        "semantic_skeleton_diversity": (
            skeletons["minimum_distinct_skeleton_count"] >= 3
            and skeletons["maximum_largest_skeleton_share"] <= 0.5
        ),
    }
    report["checks"].update(added_checks)
    report.update({
        "stage": "pilot_benchmark_final_calibration",
        "ready_for_human_review": False,
        "ready_for_spot_review": all(report["checks"].values()),
        "ready_for_final_spot_review": all(report["checks"].values()),
        "difficulty_capability_diagnostics": confounding,
        "capability_evidence_alignment": alignment,
        "semantic_skeleton_diversity": skeletons,
        "manual_review_required": {
            "spot_review_scores_are_unfilled": True,
            "human_surface_audit_passed": False,
        },
    })
    return report
