"""pyflic — FLIC (Fly Liquid-food Interaction Counter) analysis toolkit.

The public names below are imported **lazily** (PEP 562).  ``from pyflic import
load_experiment_yaml`` behaves exactly as before, but merely importing a
subpackage — ``pyflic.help``, say — no longer drags in pandas, statsmodels and
plotnine along with the whole analysis stack.

Two things depend on that:

* ``pyflic.help`` is meant to be replaceable without touching analysis code, and
  vice versa.  Eager imports here coupled them in both directions regardless of
  what the help package itself imports.
* ``pyflic --help``, ``pyflic version`` and ``pyflic help`` no longer pay for
  importing the numerical stack they never use.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import TYPE_CHECKING, Any

try:
    __version__: str = _pkg_version("pyflic")
except PackageNotFoundError:
    __version__ = "unknown"

if TYPE_CHECKING:  # pragma: no cover - for type checkers and IDEs only
    from .base.analytics import (
        bootstrap_metric,
        bout_transition_matrix,
        compare_configs,
        compare_treatments,
        light_phase_summary,
        parameter_sensitivity,
        tidy_events,
    )
    from .base.dfm import DFM
    from .base.experiment import Experiment
    from .base.experiment_design import ExperimentDesign
    from .base.hedonic_experiment import HedonicFeedingExperiment
    from .base.parameters import Parameters
    from .base.pdf_report import write_experiment_report
    from .base.progressive_ratio_experiment import ProgressiveRatioExperiment
    from .base.single_well_experiment import SingleWellExperiment
    from .base.treatment import Treatment, TreatmentChamber
    from .base.two_well_experiment import TwoWellExperiment
    from .base.yaml_config import load_experiment_yaml
    from .base.yaml_lint import lint_flic_config

#: Public name → module it lives in.  ``tests/test_public_api.py`` asserts this
#: stays in step with ``__all__``.
_LAZY: dict[str, str] = {
    "DFM": ".base.dfm",
    "Experiment": ".base.experiment",
    "ExperimentDesign": ".base.experiment_design",
    "HedonicFeedingExperiment": ".base.hedonic_experiment",
    "Parameters": ".base.parameters",
    "ProgressiveRatioExperiment": ".base.progressive_ratio_experiment",
    "SingleWellExperiment": ".base.single_well_experiment",
    "Treatment": ".base.treatment",
    "TreatmentChamber": ".base.treatment",
    "TwoWellExperiment": ".base.two_well_experiment",
    "load_experiment_yaml": ".base.yaml_config",
    # Analytics
    "bootstrap_metric": ".base.analytics",
    "bout_transition_matrix": ".base.analytics",
    "compare_configs": ".base.analytics",
    "compare_treatments": ".base.analytics",
    "light_phase_summary": ".base.analytics",
    "parameter_sensitivity": ".base.analytics",
    "tidy_events": ".base.analytics",
    # Reporting / validation
    "write_experiment_report": ".base.pdf_report",
    "lint_flic_config": ".base.yaml_lint",
}

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


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value          # cache, so this runs once per name
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
