"""R2 semantic calibration with independent capability count and structural difficulty."""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Any

from .rng import NamespaceRNG
from .semantic_frames import FRAMES_BY_TASK
from .surface import SemanticComposition, _weights, compose_semantics, validate_capability_combination


FRAME_CAPABILITIES: dict[str, tuple[str, ...]] = {
    "frame-code-scheduler": ("AR", "MH", "SC"),
    "frame-code-service-debug": ("DE", "AU", "LC", "SC"),
    "frame-code-sql-report": ("DS", "SQ", "SC"),
    "frame-code-api-client": ("AU", "DE", "SC", "AR"),
    "frame-code-refactor": ("FT", "LC", "SC"),
    "frame-code-repository-review": ("LC", "MH", "DE"),
    "frame-code-data-pipeline": ("DS", "AU", "FT", "SC"),
    "frame-code-performance": ("AR", "SC"),
    "frame-math-equation-system": ("SQ", "AR", "SC"),
    "frame-math-probability": ("SQ", "MH", "SC"),
    "frame-math-proof": ("AR", "MH", "SC", "SQ"),
    "frame-math-statistics": ("SQ", "DS", "AM"),
    "frame-math-geometry": ("SQ", "AR", "SC"),
    "frame-math-optimization": ("AR", "SQ", "SC"),
    "frame-math-word-problem": ("SQ", "MH"),
    "frame-math-chart-analysis": ("DS", "SQ", "FT"),
    "frame-qa-technical-docs": ("FK", "LC", "AU", "SC"),
    "frame-qa-evidence-chain": ("MH", "LC", "FK"),
    "frame-qa-ambiguous-policy": ("AM", "FK", "SC", "FT"),
    "frame-qa-source-comparison": ("MH", "FK", "LC"),
    "frame-qa-troubleshooting": ("AU", "DE", "SC", "FK"),
    "frame-qa-data-explanation": ("DS", "FK", "SQ", "AM"),
    "frame-qa-long-policy": ("LC", "FT", "MH"),
    "frame-qa-bilingual-terms": ("MT", "FK", "SC", "LC"),
    "frame-summary-meeting": ("LC", "FT", "SC", "AM"),
    "frame-summary-research": ("FT", "FK", "LC", "SC"),
    "frame-summary-incident": ("LC", "DE", "FT", "MH"),
    "frame-summary-metrics": ("DS", "SQ", "FT", "AM"),
    "frame-summary-conflicting-reports": ("AM", "MH", "FT", "LC"),
    "frame-summary-technical-guide": ("LC", "AU", "FT", "SC"),
    "frame-summary-bilingual-report": ("MT", "FT", "LC", "AM"),
    "frame-summary-executive": ("SC", "FT", "FK", "LC"),
    "frame-translation-technical": ("MT", "FT", "AU", "SC"),
    "frame-translation-contract": ("FT", "MT", "AM", "SC"),
    "frame-translation-notice": ("FT", "SC", "MT"),
    "frame-translation-domain-report": ("FK", "MT", "FT", "LC"),
    "frame-translation-ui-localization": ("MT", "AU", "SC", "FT"),
    "frame-translation-incident": ("LC", "FT", "MT", "MH"),
    "frame-translation-ambiguous-reference": ("AM", "FT", "MT"),
    "frame-translation-glossary": ("MT", "FK", "SC", "FT"),
}


def payload_variant_index(stable_id: str) -> int:
    match = re.fullmatch(r"(code|math|qa|summary|translation)-(\d{3})", stable_id)
    if not match:
        raise ValueError(f"malformed Pilot stable ID: {stable_id}")
    task, raw_index = match.groups()
    return (int(raw_index) // len(FRAMES_BY_TASK[task])) % 3


def compose_semantics_v3(config: dict[str, Any], stable_id: str,
                         rng: NamespaceRNG) -> SemanticComposition:
    base = compose_semantics(config, stable_id, rng)
    allowed = FRAME_CAPABILITIES[base.frame.frame_id]
    count = rng.choice(
        "capability", stable_id, tuple(range(2, min(4, len(allowed)) + 1)),
        stream="v3-capability-count",
    )
    occurrence = int(stable_id.rsplit("-", 1)[1]) // len(FRAMES_BY_TASK[base.frame.task])
    primary = allowed[occurrence % len(allowed)]
    secondary_pool = tuple(capability for capability in allowed if capability != primary)
    secondaries = tuple(rng.sample(
        "capability", stable_id, secondary_pool, count - 1,
        stream="v3-secondary-selection",
    ))
    validate_capability_combination(base.frame, primary, secondaries)
    active = (primary, *secondaries)

    structural_level = payload_variant_index(stable_id)
    randomizer = rng.random("difficulty", stable_id, stream="v3-structural-score")
    lower, width = ((0.22, 0.14), (0.47, 0.16), (0.74, 0.18))[structural_level]
    difficulty = round(lower + width * randomizer.random(), 6)
    difficulty_bin = ("low", "medium", "high")[structural_level]
    requirement_rng = rng.random("difficulty", stable_id, stream="v3-requirements")
    requirements = {
        capability: round(min(
            0.95,
            0.35 + 0.42 * difficulty + 0.14 * requirement_rng.random()
            + (0.03 if capability == primary else 0.0),
        ), 6)
        for capability in active
    }
    return replace(
        base,
        primary_capability=primary,
        secondary_capabilities=secondaries,
        capability_weights=_weights(active, stable_id, rng),
        requirement_levels=requirements,
        difficulty=difficulty,
        difficulty_bin=difficulty_bin,
    )
