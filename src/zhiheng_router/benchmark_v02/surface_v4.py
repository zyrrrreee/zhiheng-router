"""Pilot v4 surface generation with R3 human-review guardrails."""

from __future__ import annotations

import re
from typing import Any

from .payloads_v2 import ContentPayload
from .payloads_v4 import generate_content_payload_v4
from .rng import NamespaceRNG
from .schema import (
    DiagnosticSidecar,
    GenerationProvenance,
    ObservableQueryRecord,
    VisibleStructure,
    validate_pilot_records,
)
from .semantics_v4 import compose_semantics_v4
from .surface import SURFACE_STYLES, SemanticComposition, pilot_stable_ids
from .validation_v4 import (
    semantic_sibling_duplicates,
    sibling_difficulty_conflicts,
    validate_capability_roles,
    validate_frame_completeness,
)


_FORBIDDEN_SURFACE = re.compile(
    r"(?i)(?:\[\s*(?:AR|SQ|DE|DS|AU|FK|MH|LC|FT|MT|AM|SC)\s*\]|"
    r"capability\s*[=:]|algorithmic_reasoning|symbolic_quantitative|"
    r"debugging_error_localization|data_sql|api_library_usage|factual_domain_knowledge|"
    r"multi_hop_reasoning|long_context_integration|faithful_transformation|"
    r"multilingual_terminology|ambiguity_resolution|structured_constraint_following|"
    r"dev-model-[a-z]|quality-best|efficient winner|sampled outcome oracle)"
)


def _render_payload(composition: SemanticComposition, payload: ContentPayload,
                    style: str) -> str:
    english = composition.language == "en"
    labelled = {
        "material": f"{'Provided material' if english else '给定材料'}\n{payload.material}",
        "task": f"{'Task' if english else '任务'}\n{payload.task_instruction}",
        "output": f"{'Deliverable' if english else '交付形式'}\n{payload.output_contract}",
    }
    orders = {
        "direct": ("task", "material", "output"),
        "scenario": ("material", "task", "output"),
        "constraints-first": ("output", "material", "task"),
        "constraints-last": ("material", "output", "task"),
    }
    return "\n\n".join(labelled[key] for key in orders[style])


def realize_query_v4(config: dict[str, Any], composition: SemanticComposition,
                     rng: NamespaceRNG) -> tuple[ObservableQueryRecord, DiagnosticSidecar, ContentPayload]:
    payload = generate_content_payload_v4(composition, rng)
    index = int(composition.stable_id.rsplit("-", 1)[1])
    style_offset = rng.derive_seed("surface", composition.frame.task, stream="v4-style-offset")
    style = SURFACE_STYLES[(index + style_offset) % len(SURFACE_STYLES)]
    query_text = _render_payload(composition, payload, style)
    if _FORBIDDEN_SURFACE.search(query_text):
        raise ValueError(f"v4 surface contains a forbidden label or result: {composition.query_id}")
    source_slug = composition.frame.frame_id.removeprefix("frame-")
    group = index // 4
    provenance = GenerationProvenance(
        generator_version=config["generator_version"],
        config_version=config["config_version"],
        master_seed=config["pilot"]["master_seed"],
        stable_object_id=composition.stable_id,
        content_namespace=config["pilot"]["namespaces"]["content"],
        capability_namespace=config["pilot"]["namespaces"]["capability"],
        difficulty_namespace=config["pilot"]["namespaces"]["difficulty"],
        surface_namespace=config["pilot"]["namespaces"]["surface"],
    )
    observable = ObservableQueryRecord(
        query_id=composition.query_id,
        query_text=query_text,
        language=composition.language,
        explicit_output_constraints=(payload.output_contract,),
        visible_structure=VisibleStructure(
            constraint_count=len(composition.active_capabilities) + 1,
            input_length_bin=(
                "short" if len(query_text) < 500 else
                "medium" if len(query_text) < 1000 else "long"
            ),
            requested_output_format=payload.output_contract,
            translation_direction=composition.translation_direction,
        ),
        source_kind=config["source_kind"],
        dataset_version=config["dataset_version"],
    )
    sidecar = DiagnosticSidecar(
        query_id=composition.query_id,
        coarse_task=composition.frame.task,
        primary_capability=composition.primary_capability,
        secondary_capabilities=composition.secondary_capabilities,
        capability_weights=tuple(sorted(composition.capability_weights.items())),
        capability_requirement_levels=tuple(sorted(composition.requirement_levels.items())),
        capability_surface_evidence=payload.capability_evidence,
        difficulty=composition.difficulty,
        difficulty_bin=composition.difficulty_bin,
        source_family_id=f"src-v4-{source_slug}",
        template_family_id=f"tpl-v4-{source_slug}",
        paraphrase_family_id=f"para-v4-{source_slug}-{group:03d}",
        semantic_frame_id=composition.frame.frame_id,
        surface_realization_id=f"surface-v4-{style}-{composition.stable_id}",
        semantic_scenario=payload.scenario,
        generation_provenance=provenance,
    )
    return observable, sidecar, payload


def _enforce_r3_guardrails(observables: list[ObservableQueryRecord],
                           sidecars: list[DiagnosticSidecar]) -> None:
    siblings = semantic_sibling_duplicates(observables, sidecars)
    roles = validate_capability_roles(observables, sidecars)
    completeness = validate_frame_completeness(observables, sidecars)
    difficulty = sibling_difficulty_conflicts(observables, sidecars)
    failed = {
        "semantic_sibling_duplicate_group_count": siblings["duplicate_group_count"],
        "invalid_active_capability_count": roles["invalid_active_capability_count"],
        "missing_required_capability_count": roles["missing_required_capability_count"],
        "ambiguous_translation_target_count": completeness["ambiguous_translation_target_count"],
        "incomplete_timezone_conversion_count": completeness["incomplete_timezone_conversion_count"],
        "task_deliverable_mismatch_count": completeness["task_deliverable_mismatch_count"],
        "severe_sibling_difficulty_conflict_count": difficulty[
            "severe_sibling_difficulty_conflict_count"],
    }
    if any(failed.values()):
        raise ValueError(f"Pilot v4 human-review guardrail failure: {failed}")


def generate_pilot_v4(config: dict[str, Any], stable_ids: list[str] | None = None
                      ) -> tuple[list[ObservableQueryRecord], list[DiagnosticSidecar]]:
    rng = NamespaceRNG(config["pilot"]["master_seed"], config["pilot"]["namespaces"])
    selected = sorted(stable_ids if stable_ids is not None else pilot_stable_ids(config))
    triples = [realize_query_v4(config, compose_semantics_v4(config, stable_id, rng), rng)
               for stable_id in selected]
    observables = [triple[0] for triple in triples]
    sidecars = [triple[1] for triple in triples]
    validate_pilot_records(observables, sidecars)
    _enforce_r3_guardrails(observables, sidecars)
    return observables, sidecars
