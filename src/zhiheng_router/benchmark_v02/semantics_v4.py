"""R3 frame-level capability necessity and coverage rules."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .rng import NamespaceRNG
from .semantics_v3 import FRAME_CAPABILITIES, compose_semantics_v3, payload_variant_index
from .surface import SemanticComposition, _weights


_REMOVED_CAPABILITIES = {
    "frame-code-service-debug": {"AU"},
    "frame-code-data-pipeline": {"AU"},
    "frame-math-probability": {"MH"},
    "frame-math-proof": {"MH"},
    "frame-summary-research": {"FK"},
    "frame-translation-technical": {"AU", "DE"},
    "frame-translation-ui-localization": {"AU", "DE"},
}
FRAME_CAPABILITIES_V4 = {
    frame_id: tuple(capability for capability in capabilities
                    if capability not in _REMOVED_CAPABILITIES.get(frame_id, set()))
    for frame_id, capabilities in FRAME_CAPABILITIES.items()
}
FRAME_REQUIRED_CAPABILITIES: dict[str, tuple[str, ...]] = {
    "frame-code-scheduler": ("AR",),
    "frame-code-service-debug": ("DE",),
    "frame-code-sql-report": ("DS",),
    "frame-code-api-client": ("AU",),
    "frame-code-refactor": ("FT",),
    "frame-code-repository-review": ("MH",),
    "frame-code-data-pipeline": ("DS",),
    "frame-code-performance": ("AR",),
    "frame-math-equation-system": ("SQ",),
    "frame-math-probability": ("SQ",),
    "frame-math-proof": ("AR",),
    "frame-math-statistics": ("SQ",),
    "frame-math-geometry": ("SQ",),
    "frame-math-optimization": ("AR",),
    "frame-math-word-problem": ("SQ",),
    "frame-math-chart-analysis": ("DS",),
    "frame-qa-technical-docs": ("FK",),
    "frame-qa-evidence-chain": ("MH",),
    "frame-qa-ambiguous-policy": ("AM",),
    "frame-qa-source-comparison": ("MH",),
    "frame-qa-troubleshooting": ("DE",),
    "frame-qa-data-explanation": ("DS",),
    "frame-qa-long-policy": ("LC",),
    "frame-qa-bilingual-terms": ("MT",),
    "frame-summary-meeting": ("FT",),
    "frame-summary-research": ("FT",),
    "frame-summary-incident": ("FT",),
    "frame-summary-metrics": ("FT",),
    "frame-summary-conflicting-reports": ("FT",),
    "frame-summary-technical-guide": ("FT",),
    "frame-summary-bilingual-report": ("FT",),
    "frame-summary-executive": ("FT",),
    "frame-translation-technical": ("FT", "MT"),
    "frame-translation-contract": ("FT",),
    "frame-translation-notice": ("FT",),
    "frame-translation-domain-report": ("FT", "MT"),
    "frame-translation-ui-localization": ("FT", "MT"),
    "frame-translation-incident": ("FT",),
    "frame-translation-ambiguous-reference": ("FT",),
    "frame-translation-glossary": ("FT", "MT"),
}


def allowed_capabilities_v4(frame_id: str, _variant: int) -> tuple[str, ...]:
    return FRAME_CAPABILITIES_V4[frame_id]


def required_capabilities_v4(frame_id: str) -> tuple[str, ...]:
    return FRAME_REQUIRED_CAPABILITIES[frame_id]


def compose_semantics_v4(config: dict[str, Any], stable_id: str,
                         rng: NamespaceRNG) -> SemanticComposition:
    base = compose_semantics_v3(config, stable_id, rng)
    variant = payload_variant_index(stable_id)
    allowed = allowed_capabilities_v4(base.frame.frame_id, variant)
    required = required_capabilities_v4(base.frame.frame_id)
    if not set(required) <= set(allowed):
        raise ValueError(f"required capability is not allowed for {base.frame.frame_id}")

    count = rng.choice(
        "capability", stable_id, tuple(range(max(2, len(required)), min(4, len(allowed)) + 1)),
        stream="v4-capability-count",
    )
    optional = tuple(capability for capability in allowed if capability not in required)
    selected = [*required, *rng.sample(
        "capability", stable_id, optional, count - len(required),
        stream="v4-optional-capabilities",
    )]
    offset = rng.derive_seed("capability", stable_id, stream="v4-primary") % len(selected)
    active = tuple(selected[offset:] + selected[:offset])
    primary, secondaries = active[0], active[1:]
    if len(set(active)) != len(active) or not 2 <= len(active) <= 4:
        raise ValueError(f"invalid v4 capability combination for {base.frame.frame_id}")
    requirement_rng = rng.random("capability", stable_id, stream="v4-requirements")
    requirements = {
        capability: round(min(
            0.95,
            0.35 + 0.42 * base.difficulty + 0.14 * requirement_rng.random()
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
    )
