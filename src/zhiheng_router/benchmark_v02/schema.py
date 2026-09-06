"""Strict, physically separable schemas for the Router v0.2 Pilot dataset."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import re
from typing import Any


TASK_IDS = ("code", "math", "qa", "summary", "translation")
CAPABILITY_IDS = ("AR", "SQ", "DE", "DS", "AU", "FK", "MH", "LC", "FT", "MT", "AM", "SC")
LANGUAGE_IDS = ("zh", "en", "mixed")
LENGTH_BINS = ("short", "medium", "long")
DIFFICULTY_BINS = ("low", "medium", "high")
TRANSLATION_DIRECTIONS = ("zh_to_en", "en_to_zh", "bilingual_revision")
SOURCE_KIND = "synthetic"

_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)+$")
_FAMILY_PREFIXES = {
    "source_family_id": "src-",
    "template_family_id": "tpl-",
    "paraphrase_family_id": "para-",
}


def _nonblank(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value


def _identifier(value: object, name: str, *, prefix: str | None = None) -> str:
    text = _nonblank(value, name)
    if not _ID_RE.fullmatch(text) or (prefix is not None and not text.startswith(prefix)):
        raise ValueError(f"{name} is malformed")
    return text


def _finite(value: object, name: str, *, minimum: float = 0.0,
            maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    number = float(value)
    if number < minimum or (maximum is not None and number > maximum):
        raise ValueError(f"{name} is outside the allowed range")
    return number


def _exact_keys(data: Mapping[str, Any], required: set[str], name: str) -> None:
    missing = required - set(data)
    extra = set(data) - required
    if missing or extra:
        raise ValueError(f"{name} keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}")


def _string_tuple(values: object, name: str, *, minimum: int = 0) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list")
    result = tuple(_nonblank(value, name) for value in values)
    if len(result) < minimum or len(result) != len(set(result)):
        raise ValueError(f"{name} has an invalid count or duplicates")
    return result


@dataclass(frozen=True)
class VisibleStructure:
    constraint_count: int
    input_length_bin: str
    requested_output_format: str
    translation_direction: str | None = None

    def __post_init__(self) -> None:
        if type(self.constraint_count) is not int or not 0 <= self.constraint_count <= 20:
            raise ValueError("constraint_count must be an integer in [0,20]")
        if self.input_length_bin not in LENGTH_BINS:
            raise ValueError(f"invalid input_length_bin: {self.input_length_bin}")
        _nonblank(self.requested_output_format, "requested_output_format")
        if self.translation_direction is not None and self.translation_direction not in TRANSLATION_DIRECTIONS:
            raise ValueError(f"invalid translation_direction: {self.translation_direction}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_count": self.constraint_count,
            "input_length_bin": self.input_length_bin,
            "requested_output_format": self.requested_output_format,
            "translation_direction": self.translation_direction,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VisibleStructure":
        required = {"constraint_count", "input_length_bin", "requested_output_format", "translation_direction"}
        _exact_keys(data, required, "visible_structure")
        return cls(**dict(data))


@dataclass(frozen=True)
class ObservableQueryRecord:
    """Only fields a future deployable Router may inspect."""

    query_id: str
    query_text: str
    language: str
    explicit_output_constraints: tuple[str, ...]
    visible_structure: VisibleStructure
    source_kind: str
    dataset_version: str

    def __post_init__(self) -> None:
        _identifier(self.query_id, "query_id", prefix="pilot-q-")
        _nonblank(self.query_text, "query_text")
        if self.language not in LANGUAGE_IDS:
            raise ValueError(f"invalid language: {self.language}")
        constraints = _string_tuple(self.explicit_output_constraints, "explicit_output_constraints", minimum=1)
        object.__setattr__(self, "explicit_output_constraints", constraints)
        if not isinstance(self.visible_structure, VisibleStructure):
            raise ValueError("visible_structure must be a VisibleStructure")
        if self.source_kind != SOURCE_KIND:
            raise ValueError("source_kind must be synthetic")
        _nonblank(self.dataset_version, "dataset_version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "language": self.language,
            "explicit_output_constraints": list(self.explicit_output_constraints),
            "visible_structure": self.visible_structure.to_dict(),
            "source_kind": self.source_kind,
            "dataset_version": self.dataset_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObservableQueryRecord":
        required = {
            "query_id", "query_text", "language", "explicit_output_constraints",
            "visible_structure", "source_kind", "dataset_version",
        }
        _exact_keys(data, required, "observable record")
        values = dict(data)
        values["explicit_output_constraints"] = _string_tuple(
            values["explicit_output_constraints"], "explicit_output_constraints", minimum=1)
        if not isinstance(values["visible_structure"], Mapping):
            raise ValueError("visible_structure must be an object")
        values["visible_structure"] = VisibleStructure.from_dict(values["visible_structure"])
        return cls(**values)


@dataclass(frozen=True)
class GenerationProvenance:
    generator_version: str
    config_version: str
    master_seed: int
    stable_object_id: str
    content_namespace: str
    capability_namespace: str
    difficulty_namespace: str
    surface_namespace: str

    def __post_init__(self) -> None:
        for name in ("generator_version", "config_version", "stable_object_id",
                     "content_namespace", "capability_namespace", "difficulty_namespace",
                     "surface_namespace"):
            _nonblank(getattr(self, name), name)
        if type(self.master_seed) is not int or self.master_seed < 0:
            raise ValueError("master_seed must be a nonnegative integer")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GenerationProvenance":
        required = set(cls.__dataclass_fields__)
        _exact_keys(data, required, "generation_provenance")
        return cls(**dict(data))


def _numeric_map(values: object, name: str, active: set[str], *, minimum: float,
                 maximum: float) -> tuple[tuple[str, float], ...]:
    if not isinstance(values, Mapping) or set(values) != active:
        raise ValueError(f"{name} keys must equal active capabilities")
    return tuple((capability, _finite(values[capability], f"{name}.{capability}",
                                      minimum=minimum, maximum=maximum))
                 for capability in sorted(active))


@dataclass(frozen=True)
class DiagnosticSidecar:
    """Generator-only metadata that must never enter the deployable schema."""

    query_id: str
    coarse_task: str
    primary_capability: str
    secondary_capabilities: tuple[str, ...]
    capability_weights: tuple[tuple[str, float], ...]
    capability_requirement_levels: tuple[tuple[str, float], ...]
    capability_surface_evidence: tuple[tuple[str, str], ...]
    difficulty: float
    difficulty_bin: str
    source_family_id: str
    template_family_id: str
    paraphrase_family_id: str
    semantic_frame_id: str
    surface_realization_id: str
    semantic_scenario: str
    generation_provenance: GenerationProvenance

    def __post_init__(self) -> None:
        _identifier(self.query_id, "query_id", prefix="pilot-q-")
        if self.coarse_task not in TASK_IDS:
            raise ValueError(f"invalid coarse_task: {self.coarse_task}")
        if self.primary_capability not in CAPABILITY_IDS:
            raise ValueError(f"unknown capability: {self.primary_capability}")
        secondaries = _string_tuple(self.secondary_capabilities, "secondary_capabilities", minimum=1)
        if len(secondaries) > 3 or self.primary_capability in secondaries:
            raise ValueError("a Query needs 1-3 distinct secondary capabilities")
        unknown = set(secondaries) - set(CAPABILITY_IDS)
        if unknown:
            raise ValueError(f"unknown capability: {sorted(unknown)}")
        object.__setattr__(self, "secondary_capabilities", secondaries)
        active = {self.primary_capability, *secondaries}
        if not 2 <= len(active) <= 4:
            raise ValueError("active capability count must be in [2,4]")
        for name in ("capability_weights", "capability_requirement_levels"):
            entries = getattr(self, name)
            if not isinstance(entries, tuple) or {key for key, _ in entries} != active:
                raise ValueError(f"{name} keys must equal active capabilities")
        weights = dict(self.capability_weights)
        for capability, value in weights.items():
            _finite(value, f"capability_weights.{capability}", maximum=1.0)
        if not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("capability_weights must sum to 1")
        for capability, value in self.capability_requirement_levels:
            _finite(value, f"capability_requirement_levels.{capability}", minimum=0.35, maximum=0.95)
        evidence = dict(self.capability_surface_evidence)
        if set(evidence) != active or any(not isinstance(v, str) or not v.strip() for v in evidence.values()):
            raise ValueError("capability_surface_evidence must cover all active capabilities")
        _finite(self.difficulty, "difficulty", maximum=1.0)
        if self.difficulty_bin not in DIFFICULTY_BINS:
            raise ValueError(f"invalid difficulty_bin: {self.difficulty_bin}")
        for name, prefix in _FAMILY_PREFIXES.items():
            _identifier(getattr(self, name), name, prefix=prefix)
        _identifier(self.semantic_frame_id, "semantic_frame_id", prefix="frame-")
        _identifier(self.surface_realization_id, "surface_realization_id", prefix="surface-")
        _nonblank(self.semantic_scenario, "semantic_scenario")
        if not isinstance(self.generation_provenance, GenerationProvenance):
            raise ValueError("generation_provenance must be GenerationProvenance")

    @property
    def active_capabilities(self) -> tuple[str, ...]:
        return (self.primary_capability, *self.secondary_capabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "coarse_task": self.coarse_task,
            "primary_capability": self.primary_capability,
            "secondary_capabilities": list(self.secondary_capabilities),
            "capability_weights": dict(self.capability_weights),
            "capability_requirement_levels": dict(self.capability_requirement_levels),
            "capability_surface_evidence": dict(self.capability_surface_evidence),
            "difficulty": self.difficulty,
            "difficulty_bin": self.difficulty_bin,
            "source_family_id": self.source_family_id,
            "template_family_id": self.template_family_id,
            "paraphrase_family_id": self.paraphrase_family_id,
            "semantic_frame_id": self.semantic_frame_id,
            "surface_realization_id": self.surface_realization_id,
            "semantic_scenario": self.semantic_scenario,
            "generation_provenance": self.generation_provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DiagnosticSidecar":
        required = set(cls.__dataclass_fields__)
        _exact_keys(data, required, "diagnostic sidecar")
        values = dict(data)
        secondaries = _string_tuple(values["secondary_capabilities"], "secondary_capabilities", minimum=1)
        active = {values["primary_capability"], *secondaries}
        values["secondary_capabilities"] = secondaries
        values["capability_weights"] = _numeric_map(
            values["capability_weights"], "capability_weights", active, minimum=0.0, maximum=1.0)
        values["capability_requirement_levels"] = _numeric_map(
            values["capability_requirement_levels"], "capability_requirement_levels", active,
            minimum=0.35, maximum=0.95)
        evidence = values["capability_surface_evidence"]
        if not isinstance(evidence, Mapping) or set(evidence) != active:
            raise ValueError("capability_surface_evidence keys must equal active capabilities")
        values["capability_surface_evidence"] = tuple(
            (capability, _nonblank(evidence[capability], f"evidence.{capability}"))
            for capability in sorted(active))
        if not isinstance(values["generation_provenance"], Mapping):
            raise ValueError("generation_provenance must be an object")
        values["generation_provenance"] = GenerationProvenance.from_dict(values["generation_provenance"])
        return cls(**values)


def validate_pilot_records(observables: Sequence[ObservableQueryRecord],
                           sidecars: Sequence[DiagnosticSidecar]) -> None:
    if not observables or not sidecars:
        raise ValueError("Pilot observable and sidecar records must be nonempty")
    observable_ids = [record.query_id for record in observables]
    sidecar_ids = [record.query_id for record in sidecars]
    if len(observable_ids) != len(set(observable_ids)):
        raise ValueError("duplicate query_id in observable records")
    if len(sidecar_ids) != len(set(sidecar_ids)):
        raise ValueError("duplicate query_id in diagnostic sidecar")
    if set(observable_ids) != set(sidecar_ids):
        raise ValueError("observable and diagnostic query IDs must match one-to-one")
    for record in observables:
        record.__post_init__()
    for record in sidecars:
        record.__post_init__()
