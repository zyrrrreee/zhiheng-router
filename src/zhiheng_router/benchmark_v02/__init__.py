"""Router v0.2 synthetic benchmark infrastructure (Pilot stage only)."""

from .config import load_benchmark_config, validate_benchmark_config
from .rng import NamespaceRNG
from .schema import (
    CAPABILITY_IDS,
    LANGUAGE_IDS,
    TASK_IDS,
    DiagnosticSidecar,
    GenerationProvenance,
    ObservableQueryRecord,
    VisibleStructure,
    validate_pilot_records,
)

__all__ = [
    "CAPABILITY_IDS",
    "LANGUAGE_IDS",
    "TASK_IDS",
    "DiagnosticSidecar",
    "GenerationProvenance",
    "NamespaceRNG",
    "ObservableQueryRecord",
    "VisibleStructure",
    "load_benchmark_config",
    "validate_benchmark_config",
    "validate_pilot_records",
]
