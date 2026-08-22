"""The Custom Experiment Type — the absence of a chosen type.

A config with no ``experiment_type`` key IS a Custom Experiment; it can also be
selected explicitly.  Behaviourally identical to the permissive base class; it
exists so the registry, the menus, and the report cover have something concrete
to name.
"""

from __future__ import annotations

from .base import ExperimentType


class CustomExperimentType(ExperimentType):
    name = "Custom"
    display_name = "Custom Experiment"

    def report_intro(self) -> str:
        return ("A FLIC feeding experiment analysed without a specific "
                "experiment type. No type-level constraints were applied: the "
                "chamber layout, facets and quality cutoffs are whatever the "
                "config states.")
