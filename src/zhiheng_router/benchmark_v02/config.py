"""Canonical Router v0.2 benchmark configuration validation."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path
from typing import Any

from .schema import CAPABILITY_IDS, TASK_IDS, TRANSLATION_DIRECTIONS


VERSION_FIELDS = (
    "dataset_version", "generator_version", "config_version",
    "capability_taxonomy_version", "model_profile_version", "runtime_profile_version",
    "synthetic_workload_version", "latency_model_version", "diagnostic_protocol_version",
    "split_definition_version",
)
RNG_NAMESPACES = (
    "content", "surface", "capability", "difficulty", "quality_noise", "cost_noise",
    "latency_noise", "split",
)
MODEL_IDS = tuple(f"dev-model-{letter}" for letter in "abcdef")
TOP_LEVEL_FIELDS = {
    *VERSION_FIELDS,
    "source_kind", "coarse_tasks", "capabilities", "quality_thresholds",
    "quality_model", "language_distribution", "models", "synthetic_workload",
    "diagnostic_protocol", "split_definition", "pilot",
}


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _finite(value: object, name: str, *, minimum: float | None = None,
            maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    number = float(value)
    if minimum is not None and number < minimum or maximum is not None and number > maximum:
        raise ValueError(f"{name} is outside the allowed range")
    return number


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{name} keys mismatch: missing={sorted(expected-set(value))}, extra={sorted(set(value)-expected)}")


def _probability_distribution(value: object, expected: set[str], name: str) -> None:
    mapping = _mapping(value, name)
    _exact_keys(mapping, expected, name)
    total = sum(_finite(item, f"{name}.{key}", minimum=0.0, maximum=1.0)
                for key, item in mapping.items())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{name} must sum to 1")


def validate_benchmark_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate all frozen Phase 1A configuration without generating data."""
    config = dict(config)
    _exact_keys(config, TOP_LEVEL_FIELDS, "benchmark config")
    for field in VERSION_FIELDS:
        if not isinstance(config.get(field), str) or not config[field].strip():
            raise ValueError(f"{field} must be a nonblank string")
    if config.get("source_kind") != "synthetic":
        raise ValueError("source_kind must be synthetic")
    if tuple(config.get("coarse_tasks", ())) != TASK_IDS:
        raise ValueError("coarse_tasks must be the five canonical tasks in canonical order")
    capabilities = _mapping(config.get("capabilities"), "capabilities")
    _exact_keys(capabilities, set(CAPABILITY_IDS), "capabilities")
    if any(not isinstance(value, str) or not value.strip() for value in capabilities.values()):
        raise ValueError("capability names must be nonblank")

    thresholds = _mapping(config.get("quality_thresholds"), "quality_thresholds")
    _exact_keys(thresholds, {"primary", "sensitivity"}, "quality_thresholds")
    if _finite(thresholds["primary"], "quality_thresholds.primary", minimum=0, maximum=1) != 0.8:
        raise ValueError("primary quality threshold must be 0.8")
    if thresholds["sensitivity"] != [0.7, 0.9]:
        raise ValueError("sensitivity quality thresholds must be [0.7, 0.9]")

    quality = _mapping(config.get("quality_model"), "quality_model")
    _exact_keys(quality, {
        "mu_floor", "mu_scale", "intercept", "match_coefficient",
        "bottleneck_coefficient", "softmin_tau", "difficulty_coefficient",
        "model_bias_bounds", "task_residual_abs_max", "synergy_abs_max",
        "requirement_level_bounds", "quality_noise_std_bounds",
        "quality_noise_truncation_sigma",
    }, "quality_model")
    for key, value in quality.items():
        if isinstance(value, list):
            if len(value) != 2:
                raise ValueError(f"quality_model.{key} bounds need two values")
            bounds = [_finite(item, f"quality_model.{key}[{index}]")
                      for index, item in enumerate(value)]
            if bounds[0] > bounds[1]:
                raise ValueError(f"quality_model.{key} bounds must be ordered")
        else:
            _finite(value, f"quality_model.{key}")

    languages = _mapping(config.get("language_distribution"), "language_distribution")
    _exact_keys(languages, set(TASK_IDS), "language_distribution")
    for task in TASK_IDS[:-1]:
        _probability_distribution(languages[task], {"zh", "mixed", "en"}, f"language_distribution.{task}")
    translation = _mapping(languages["translation"], "language_distribution.translation")
    _exact_keys(translation, {"directions"}, "language_distribution.translation")
    _probability_distribution(translation["directions"], set(TRANSLATION_DIRECTIONS),
                              "language_distribution.translation.directions")

    models = config.get("models")
    if not isinstance(models, list) or [model.get("model_id") for model in models] != list(MODEL_IDS):
        raise ValueError("models must be exactly dev-model-a through dev-model-f in canonical order")
    for model in models:
        model_id = model["model_id"]
        _exact_keys(model, {"model_id", "capability_profile", "task_residuals", "synergies",
                            "runtime_profile"}, model_id)
        profile = _mapping(model["capability_profile"], f"{model_id}.capability_profile")
        _exact_keys(profile, set(CAPABILITY_IDS), f"{model_id}.capability_profile")
        for capability, value in profile.items():
            _finite(value, f"{model_id}.capability_profile.{capability}", minimum=0, maximum=1)
        residuals = _mapping(model["task_residuals"], f"{model_id}.task_residuals")
        _exact_keys(residuals, set(TASK_IDS), f"{model_id}.task_residuals")
        for task, value in residuals.items():
            if abs(_finite(value, f"{model_id}.task_residuals.{task}")) > 0.04:
                raise ValueError(f"{model_id}.task_residuals.{task} exceeds 0.04")
        synergies = _mapping(model["synergies"], f"{model_id}.synergies")
        for pair, value in synergies.items():
            parts = pair.split("+")
            if len(parts) != 2 or len(set(parts)) != 2 or not set(parts) <= set(CAPABILITY_IDS):
                raise ValueError(f"invalid synergy pair: {model_id}.{pair}")
            if abs(_finite(value, f"{model_id}.synergies.{pair}")) > 0.08:
                raise ValueError(f"{model_id}.synergies.{pair} exceeds 0.08")
        runtime = _mapping(model["runtime_profile"], f"{model_id}.runtime_profile")
        runtime_fields = {"input_cost_rate", "output_cost_rate", "startup_synthetic_ms",
                          "prefill_rate", "decode_rate", "reasoning_rate", "difficulty_resilience"}
        _exact_keys(runtime, runtime_fields, f"{model_id}.runtime_profile")
        for field, value in runtime.items():
            _finite(value, f"{model_id}.runtime_profile.{field}", minimum=0.0,
                    maximum=1.0 if field == "difficulty_resilience" else None)
            if field != "difficulty_resilience" and value == 0:
                raise ValueError(f"{model_id}.runtime_profile.{field} must be positive")

    workload = _mapping(config.get("synthetic_workload"), "synthetic_workload")
    _exact_keys(workload, {
        "input_unit_coefficients", "requested_output_defaults", "summary_output_ratio",
        "summary_output_bounds", "translation_output_ratio", "translation_output_bounds",
        "reasoning_work", "cost_noise_sigma_bounds", "latency_noise_sigma_bounds",
        "latency_unit", "cost_unit", "report_label",
    }, "synthetic_workload")
    for value in workload.values():
        if isinstance(value, Mapping):
            for nested_name, nested_value in value.items():
                if isinstance(nested_value, list):
                    for item in nested_value:
                        _finite(item, f"synthetic_workload.{nested_name}")
                else:
                    _finite(nested_value, f"synthetic_workload.{nested_name}")
        elif isinstance(value, list):
            for item in value:
                _finite(item, "synthetic_workload.bounds")
    if workload.get("latency_unit") != "synthetic_ms" or workload.get("cost_unit") != "synthetic_cost_units":
        raise ValueError("synthetic workload units are invalid")
    if workload.get("report_label") != "SIMULATED / SYNTHETIC UNITS":
        raise ValueError("synthetic report label is invalid")

    diagnostic = _mapping(config.get("diagnostic_protocol"), "diagnostic_protocol")
    _exact_keys(diagnostic, {
        "vectorizer", "logistic_regression", "fit_split", "tuning_on_development_test",
    }, "diagnostic_protocol")
    if diagnostic.get("fit_split") != "train" or diagnostic.get("tuning_on_development_test") is not False:
        raise ValueError("diagnostic protocol must fit Train and forbid Development Test tuning")
    split = _mapping(config.get("split_definition"), "split_definition")
    _exact_keys(split, {
        "unit", "train", "validation", "development_test", "frozen_final_test",
        "final_familiar", "final_novel",
    }, "split_definition")
    fractions = [_finite(split[name], f"split_definition.{name}", minimum=0.0, maximum=1.0)
                 for name in ("train", "validation", "development_test", "frozen_final_test")]
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("split fractions must be finite and sum to 1")
    if split.get("final_familiar") != 0.75 or split.get("final_novel") != 0.25:
        raise ValueError("Final Familiar/Novel fractions must be 0.75/0.25")

    pilot = _mapping(config.get("pilot"), "pilot")
    _exact_keys(pilot, {
        "pilot_version", "master_seed", "queries_per_task", "namespaces",
        "query_data_path", "sidecar_data_path", "manifest_path", "audit_path",
        "review_packet_path",
    }, "pilot")
    if type(pilot.get("master_seed")) is not int or pilot["master_seed"] < 0:
        raise ValueError("pilot.master_seed must be a nonnegative integer")
    count = pilot.get("queries_per_task")
    if type(count) is not int or not 40 <= count <= 60 or not 200 <= count * len(TASK_IDS) <= 300:
        raise ValueError("pilot query count must be 40-60 per task and 200-300 total")
    namespaces = _mapping(pilot.get("namespaces"), "pilot.namespaces")
    _exact_keys(namespaces, set(RNG_NAMESPACES), "pilot.namespaces")
    if any(not isinstance(value, str) or not value.strip() for value in namespaces.values()):
        raise ValueError("pilot namespace salts must be nonblank")
    if len(set(namespaces.values())) != len(namespaces):
        raise ValueError("pilot namespace salts must be unique")
    for path_field in ("query_data_path", "sidecar_data_path", "manifest_path", "audit_path",
                       "review_packet_path", "pilot_version"):
        if not isinstance(pilot.get(path_field), str) or not pilot[path_field].strip():
            raise ValueError(f"pilot.{path_field} must be nonblank")
    return config


def load_benchmark_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid benchmark config: {exc}") from exc
    if not isinstance(config, Mapping):
        raise ValueError("benchmark config root must be an object")
    return validate_benchmark_config(config)
