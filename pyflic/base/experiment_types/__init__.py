"""Experiment Types: the composed strategy layer above Chamber Layout (ADR-0007)."""

from .base import LAYOUT_CHAMBER_SIZE, LAYOUTS, ExperimentType
from .custom import CustomExperimentType
from .hedonic import HedonicExperimentType
from .progressive_ratio import ProgressiveRatioExperimentType
from .registry import (
    RETIRED_AS_LAYOUT,
    available_experiment_types,
    get_experiment_type,
)

__all__ = [
    "ExperimentType",
    "CustomExperimentType",
    "HedonicExperimentType",
    "ProgressiveRatioExperimentType",
    "get_experiment_type",
    "available_experiment_types",
    "RETIRED_AS_LAYOUT",
    "LAYOUT_CHAMBER_SIZE",
    "LAYOUTS",
]
