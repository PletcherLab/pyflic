from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__: str = _pkg_version("pyflic")
except PackageNotFoundError:
    __version__ = "unknown"

from .base.dfm import DFM
from .base.experiment import Experiment
from .base.experiment_design import ExperimentDesign
from .base.hedonic_experiment import HedonicFeedingExperiment
from .base.parameters import Parameters
from .base.progressive_ratio_experiment import ProgressiveRatioExperiment
from .base.single_well_experiment import SingleWellExperiment
from .base.treatment import Treatment, TreatmentChamber
from .base.two_well_experiment import TwoWellExperiment
from .base.yaml_config import load_experiment_yaml

# Higher-level analytics (lazy imports happen at call time inside)
from .base.analytics import (
    bootstrap_metric,
    bout_transition_matrix,
    compare_configs,
    compare_treatments,
    light_phase_summary,
    parameter_sensitivity,
    tidy_events,
)
from .base.pdf_report import write_experiment_report
from .base.yaml_lint import lint_flic_config

__all__ = [
    "DFM",
    "Parameters",
    "Treatment",
    "TreatmentChamber",
    "ExperimentDesign",
    "Experiment",
    "SingleWellExperiment",
    "TwoWellExperiment",
    "HedonicFeedingExperiment",
    "ProgressiveRatioExperiment",
    "load_experiment_yaml",
    # Analytics
    "tidy_events",
    "bootstrap_metric",
    "compare_treatments",
    "light_phase_summary",
    "parameter_sensitivity",
    "bout_transition_matrix",
    "compare_configs",
    "write_experiment_report",
    "lint_flic_config",
]
