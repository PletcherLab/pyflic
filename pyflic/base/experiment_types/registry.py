"""Registry mapping an ``experiment_type`` name to its ``ExperimentType``.

Adding a type = adding a subclass and one line here (ADR-0007).

Note what is *not* here: ``two_well`` and ``single_well``.  Those stopped being
experiment types — they are Chamber Layouts, one level down.  A config still
naming one of them is reported by :func:`get_experiment_type` with the
migration in the message, because a silent fallback to Custom would quietly
change which analyses run.
"""

from __future__ import annotations

from .base import ExperimentType
from .custom import CustomExperimentType
from .hedonic import HedonicExperimentType
from .progressive_ratio import ProgressiveRatioExperimentType

# name (lowercase, punctuation-insensitive) -> class
_REGISTRY: dict[str, type[ExperimentType]] = {
    CustomExperimentType.name.lower(): CustomExperimentType,
    HedonicExperimentType.name.lower(): HedonicExperimentType,
    ProgressiveRatioExperimentType.name.lower(): ProgressiveRatioExperimentType,
    # Tolerated spellings of the same names.
    "progressive_ratio": ProgressiveRatioExperimentType,
    "progressiveratio": ProgressiveRatioExperimentType,
}

#: Values that used to be experiment types and are now Chamber Layouts.
RETIRED_AS_LAYOUT = {"two_well", "twowell", "single_well", "singlewell"}


def _key(name) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def get_experiment_type(name) -> ExperimentType:
    """Return an ExperimentType instance for *name*.

    ``None`` or blank resolves to the Custom Experiment.  A retired layout name
    raises with the migration spelled out; any other unknown name raises with
    the known list.
    """
    if name is None or str(name).strip() == "":
        return CustomExperimentType()
    key = _key(name)
    if key in _REGISTRY:
        return _REGISTRY[key]()
    if key in RETIRED_AS_LAYOUT:
        layout = "single_well" if "single" in key else "two_well"
        raise ValueError(
            f"experiment_type '{name}' is no longer an experiment type — it is "
            f"a chamber layout (ADR-0007). Remove the experiment_type key and "
            f"set 'chamber_layout: {layout}' under global: instead. Run "
            f"'pyflic lint' for the full migration."
        )
    known = ", ".join(sorted({cls.name for cls in _REGISTRY.values()}))
    raise ValueError(f"Unknown experiment_type '{name}'. Known types: {known}.")


def available_experiment_types() -> list[ExperimentType]:
    """One instance of each registered type, Custom first."""
    seen: dict[str, type[ExperimentType]] = {}
    for cls in _REGISTRY.values():
        seen[cls.name] = cls
    ordered = sorted(seen.values(),
                     key=lambda c: (c is not CustomExperimentType, c.name))
    return [cls() for cls in ordered]
