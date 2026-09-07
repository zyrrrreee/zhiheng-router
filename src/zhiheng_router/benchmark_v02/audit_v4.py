"""R3 automatic gates for Pilot v4."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .audit_v3 import audit_pilot_v3
from .schema import DiagnosticSidecar, ObservableQueryRecord
from .validation_v4 import (
    semantic_sibling_duplicates,
    sibling_difficulty_conflicts,
    validate_capability_roles,
    validate_frame_completeness,
)


def audit_pilot_v4(config: dict[str, Any], observables: Sequence[ObservableQueryRecord],
                   sidecars: Sequence[DiagnosticSidecar]) -> dict[str, Any]:
    report = audit_pilot_v3(config, observables, sidecars)
    siblings = semantic_sibling_duplicates(observables, sidecars)
    roles = validate_capability_roles(observables, sidecars)
    completeness = validate_frame_completeness(observables, sidecars)
    difficulty = sibling_difficulty_conflicts(observables, sidecars)
    report["checks"].update({
        "semantic_sibling_duplicates": siblings["duplicate_group_count"] == 0,
        "capability_necessity": (
            roles["capability_necessity_valid_rate"] == 1.0
            and roles["invalid_active_capability_count"] == 0
        ),
        "capability_coverage": (
            roles["capability_coverage_valid_rate"] == 1.0
            and roles["missing_required_capability_count"] == 0
        ),
        "translation_target_complete": (
            completeness["ambiguous_translation_target_count"] == 0),
        "timezone_conversion_complete": (
            completeness["incomplete_timezone_conversion_count"] == 0),
        "task_deliverable_complete": (
            completeness["task_deliverable_mismatch_count"] == 0),
        "sibling_difficulty_consistent": (
            difficulty["severe_sibling_difficulty_conflict_count"] == 0),
    })
    ready = all(report["checks"].values())
    report.update({
        "stage": "pilot_human_review_guardrail_fix",
        "ready_for_human_review": False,
        "ready_for_spot_review": ready,
        "ready_for_final_spot_review": ready,
        "ready_for_final_human_spot_review": ready,
        "semantic_sibling_duplicates": siblings,
        "capability_role_validation": roles,
        "frame_semantic_completeness": completeness,
        "sibling_difficulty_consistency": difficulty,
        "manual_review_required": {
            "spot_review_scores_are_unfilled": True,
            "human_surface_audit_passed": False,
        },
    })
    return report
