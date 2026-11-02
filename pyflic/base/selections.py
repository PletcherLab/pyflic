"""The QC Set, the Analysis Set and the Plot Set, and the Selection (ADR-0018).

What a member's *Run QC*, *Run analysis* and *Generate plots* can produce, one
:class:`Item` per checkbox of the Hub's QC, Analyze and Plots panels, and
which of them a member's runs leave out: ``global.omit:``, with ``qc:``,
``analyses:`` and ``plots:`` lists of item ids.  The Selection stores what is
left *out*, so a config nobody touched runs everything and an item a later
version adds is included until someone unticks it.  For a Member it lives in
the Project's design (``project.yaml`` ``design.global.omit``), the authority
over every Member's ``global:``; for a standalone experiment, in its own
``flic_config.yaml``.  A Project's *Analyze all* and a Batch Run's
``run_all_analyses`` keep Basic analysis and do not read it.

Pure: no Qt and no plotting imports, so the Hub, the script runner and
``pyflic lint`` all read the same tables.  The runs themselves are in
:mod:`pyflic.base.run_selection`.  Plan: ``docs/hub-selection-plan.md``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

#: The ``omit:`` kinds, one per set.
QC, ANALYSES, PLOTS = "qc", "analyses", "plots"
OMIT_KINDS: tuple[str, ...] = (QC, ANALYSES, PLOTS)
#: The ``global:`` key that holds the Selection.
OMIT_KEY = "omit"
#: The Slices a run leaves out (``global.omit.slices``, ADR-0020): not a set
#: of outputs but which of the design's Slices the Hub's runs cover beside
#: the whole recording, which always runs.
SLICES = "slices"

#: What an item can require of a member (``Item.requires``): a two-well
#: layout, an optogenetic experiment, or an Analysis Module its Experiment
#: Type includes (ADR-0019).  The groups the Progressive Ratio and Hedonic
#: items sit in are those modules'.
TWO_WELL, OPTOGENETIC = "two_well", "optogenetic"
RATIO, HEDONIC, YOKED = "ratio", "hedonic", "yoked"
MODULE_REQUIREMENTS: tuple[str, ...] = (RATIO, HEDONIC, YOKED)
REQUIREMENTS: tuple[str, ...] = (TWO_WELL, OPTOGENETIC, *MODULE_REQUIREMENTS)

#: What a run says of an item whose requirement the member does not meet.
_NOT_MET: dict[str, str] = {
    TWO_WELL: "needs a two-well chamber layout",
    OPTOGENETIC: ("needs an optogenetic experiment (optogenetics: yes, or a "
                  "data/Program.txt or light data)"),
    RATIO: "needs an experiment type with the Progressive Ratio module",
    HEDONIC: "needs an experiment type with the Hedonic module",
    YOKED: "needs an experiment type with the Paired and yoked module",
}

#: The run's own name for each kind, as the Hub's buttons say it.
RUN_NAMES: dict[str, str] = {QC: "Run QC", ANALYSES: "Run analysis",
                             PLOTS: "Generate plots"}


@dataclass(frozen=True)
class Item:
    """One checkbox: an output a run can produce or leave out."""

    id: str                    #: what ``global.omit.<kind>`` names
    kind: str                  #: :data:`QC` | :data:`ANALYSES` | :data:`PLOTS`
    group: str                 #: the group box's title
    label: str
    caption: str               #: the tooltip: what it is and what it writes
    #: One of :data:`REQUIREMENTS`, a tuple of them (all needed), or ``None``.
    requires: str | tuple[str, ...] | None
    outputs: tuple[str, ...]   #: globs under the Experiment Directory

    @property
    def requirements(self) -> tuple[str, ...]:
        """Every requirement the item names."""
        if self.requires is None:
            return ()
        return (self.requires,) if isinstance(self.requires, str) else tuple(self.requires)


def _qc(id: str, group: str, label: str, caption: str, requires: str | None,
        *outputs: str) -> Item:
    return Item(id, QC, group, label, caption, requires, tuple(outputs))


def _an(id: str, group: str, label: str, caption: str, requires,
        *outputs: str) -> Item:
    return Item(id, ANALYSES, group, label, caption, requires, tuple(outputs))


def _pl(id: str, group: str, label: str, caption: str, requires: str | None,
        *outputs: str) -> Item:
    return Item(id, PLOTS, group, label, caption, requires, tuple(outputs))


# ---------------------------------------------------------------------------
# The three sets, in panel order.  The outputs are the names the code writes
# (Experiment.write_qc_item, the Progressive Ratio and Hedonic writers, the
# script runner's figure names).  A run never removes them: an item left out
# keeps whatever an earlier run wrote.
# ---------------------------------------------------------------------------

QC_SET: tuple[Item, ...] = (
    _qc("integrity", "QC reports", "Data integrity",
        "Per DFM: rows, time span, lost and repeated records and, on firmware 4.0.0, "
        "the Paradigm sequence and restarts.  Writes qc/integrity/DFM<id>_integrity_report"
        ".csv and .txt.", None,
        "qc/integrity/DFM*_integrity_report.csv", "qc/integrity/DFM*_integrity_report.txt"),
    _qc("data_breaks", "QC reports", "Data breaks",
        "Per DFM: the rows that follow a gap in time longer than four sample periods.  "
        "Writes qc/data_breaks/DFM<id>_data_breaks.csv for a DFM that has any.", None,
        "qc/data_breaks/DFM*_data_breaks.csv"),
    _qc("raw_signal", "QC reports", "Raw signal plots",
        "Per DFM: every well's raw signal.  Writes qc/raw_signal/DFM<id>_raw.png.", None,
        "qc/raw_signal/DFM*_raw.png"),
    _qc("baselined_signal", "QC reports", "Baselined signal plots",
        "Per DFM: every well's baselined signal with its feeding thresholds "
        "(feeding_threshold and feeding_minimum) as dashed lines.  "
        "Writes qc/baselined/DFM<id>_baselined.png.", None,
        "qc/baselined/DFM*_baselined.png"),
    _qc("cumulative_licks", "QC reports", "Cumulative licks plots",
        "Per DFM: every well's cumulative licks.  Writes "
        "qc/cumulative_licks/DFM<id>_cumulative_licks.png.", None,
        "qc/cumulative_licks/DFM*_cumulative_licks.png"),
    _qc("simultaneous_feeding", "QC reports", "Simultaneous feeding",
        "Per DFM and chamber: the samples licked at each well and at both at once, after "
        "and before the dual-feeding correction.  Writes "
        "qc/simultaneous_feeding/DFM<id>_simultaneous_feeding_matrix.csv.", TWO_WELL,
        "qc/simultaneous_feeding/DFM*_simultaneous_feeding_matrix.csv"),
    _qc("bleeding", "QC reports", "Bleeding between wells",
        "Per DFM: how far each well's signal follows the others' (cross-talk).  Writes "
        "qc/bleeding/DFM<id>_bleeding_matrix.csv and DFM<id>_bleeding_alldata.csv.",
        TWO_WELL,
        "qc/bleeding/DFM*_bleeding_matrix.csv", "qc/bleeding/DFM*_bleeding_alldata.csv"),
    _qc("opto_light_qc", "Optogenetics", "Opto light QC tables",
        "Per linkage group: was the light where the licks were?  Writes qc/opto/"
        "opto_light_qc.csv, opto_light_intervals.csv, opto_light_events.csv and "
        "opto_program.csv (and opto_baselines.csv when the data have PyMCU's Baseline "
        "columns), and logs every flagged group.", OPTOGENETIC,
        "qc/opto/opto_light_qc.csv", "qc/opto/opto_light_intervals.csv",
        "qc/opto/opto_light_events.csv", "qc/opto/opto_program.csv"),
    _qc("opto_light_figures", "Optogenetics", "Light explained by licks",
        "Per DFM: each linkage group's lit time, explained by a trigger-well lick or touch "
        "and unexplained.  Writes qc/opto/opto_light_dfm<id>.png.", OPTOGENETIC,
        "qc/opto/opto_light_dfm*.png"),
    _qc("program_timeline", "Optogenetics", "Program timeline",
        "Per DFM: the Program Intervals by Paradigm, and each well's Acclimation, "
        "Activations and light.  Needs firmware 4.0.0 data or data/Program.txt.  Writes "
        "qc/program/program_timeline_dfm<id>.png.", OPTOGENETIC,
        "qc/program/program_timeline_dfm*.png"),
    _qc("pr_light_qc", "Progressive Ratio", "Light QC table",
        "Per chamber group: did the paired fly earn its light?  Writes "
        "analysis/pr_light_qc.csv and analysis/pr_light_events.csv, and logs every "
        "flagged group.", RATIO,
        "analysis/pr_light_qc.csv", "analysis/pr_light_events.csv"),
    _qc("pr_light_event_figures", "Progressive Ratio", "Licks per light event",
        "Per DFM: the sucrose licks credited to each Test light event; a working "
        "progressive ratio climbs.  Writes analysis/pr_light_events_dfm<id>.png.",
        RATIO,
        "analysis/pr_light_events_dfm*.png"),
    _qc("pr_resting_level_figures", "Progressive Ratio", "Sucrose Well resting level",
        "Per DFM: the paired Sucrose Well's per-minute median raw signal against the "
        "DFM's other Sucrose Wells.  Writes analysis/pr_resting_level_dfm<id>.png.",
        RATIO,
        "analysis/pr_resting_level_dfm*.png"),
)

ANALYSIS_SET: tuple[Item, ...] = (
    _an("binned_summary", "Analyses", "Binned summary",
        "Per chamber: the feeding metrics in bins of the Bin size.  Writes "
        "analysis/binned_feeding_summary.csv.", None,
        "analysis/binned_feeding_summary.csv"),
    _an("event_statistics", "Analyses", "Event statistics",
        "One row per feeding event: its time, duration, licks and intensity.  Writes "
        "analysis/tidy_feeding_events.csv.", None,
        "analysis/tidy_feeding_events.csv"),
    _an("light_tables", "Optogenetics", "Light tables",
        "Per chamber and per well: lit time, Activations, Acclimation, Firmware Events "
        "and the Paradigm type's own measures.  Writes analysis/light_summary.csv and "
        "light_wells.csv and, with Facets, their _facet variants.", OPTOGENETIC,
        "analysis/light_summary.csv", "analysis/light_summary_facet.csv",
        "analysis/light_wells.csv", "analysis/light_wells_facet.csv"),
    _an("yoked_comparison", "Optogenetics", "Paired and yoked comparison",
        "Per yoked linkage group and Facet: the Paired chamber's metrics, its Yoked "
        "chambers' mean and the difference.  Writes analysis/yoked_comparison.csv.",
        (OPTOGENETIC, YOKED), "analysis/yoked_comparison.csv"),
    _an("paired_yoked_diff", "Progressive Ratio", "Paired-Yoked Difference",
        "Per chamber group and Facet: paired minus yoked, the primary table for "
        "Progressive Ratio statistics.  Writes analysis/paired_yoked_diff.csv.",
        RATIO, "analysis/paired_yoked_diff.csv"),
    _an("cumulative_diff", "Progressive Ratio", "Cumulative difference table",
        "The paired minus yoked cumulative licks since training end, in one-minute bins, "
        "which the Project pools.  Writes analysis/pr_cumulative_diff.csv.",
        RATIO, "analysis/pr_cumulative_diff.csv"),
    _an("breaking_point", "Progressive Ratio", "Breaking point",
        "Per chamber group: the paired fly's breaking point, and its sensitivity to the "
        "gap.  Writes analysis/pr_breaking_point.csv.", RATIO,
        "analysis/pr_breaking_point.csv"),
    _an("weighted_duration", "Hedonic", "Weighted duration summary",
        "Per treatment: median bout duration at each well, plain and weighted by each "
        "chamber's events.  Writes analysis/treatment_means.csv.", HEDONIC,
        "analysis/treatment_means.csv"),
    _an("experiment_report", "Report", "Experiment report (PDF)",
        "Quality control, the inference figures with their statistics, and the "
        "appendices, written last so it reflects the run.  Writes "
        "analysis/experiment_report.pdf.", None,
        "analysis/experiment_report.pdf"),
)

PLOT_SET: tuple[Item, ...] = (
    _pl("feeding_summary_plot", "Figures", "Feeding summary",
        "Per-chamber feeding metrics by treatment.  Writes analysis/feeding_summary.png.",
        None, "analysis/feeding_summary.png"),
    _pl("well_comparison", "Figures", "Well A vs B",
        "Median bout duration at well A beside well B, by treatment.  Writes "
        "analysis/well_comparison_MedDuration.png.", TWO_WELL,
        "analysis/well_comparison_*.png"),
    _pl("binned_time_course", "Chosen metric", "Binned time course",
        "The chosen metric in bins of the Bin size, by treatment.  Writes "
        "analysis/binned_<metric>.png.", None,
        "analysis/binned_*.png"),
    _pl("dot_plot", "Chosen metric", "Dot plot",
        "The chosen metric, one point per chamber (per Chamber Group under Progressive "
        "Ratio), by treatment.  Writes analysis/dot_<metric>.png.", None,
        "analysis/dot_*.png"),
    _pl("pr_cumulative_diff_plot", "Progressive Ratio", "Cumulative difference curve",
        "Paired minus yoked cumulative licks since training end, by treatment.  Writes "
        "analysis/pr_cumulative_diff.png.", RATIO,
        "analysis/pr_cumulative_diff.png"),
    _pl("pr_cumulative_licks", "Progressive Ratio", "Training-aligned traces",
        "Per DFM: paired and yoked cumulative licks since training end, one panel per "
        "chamber group.  Writes analysis/pr_cumulative_licks_dfm<id>.png.",
        RATIO, "analysis/pr_cumulative_licks_dfm*.png"),
    _pl("pr_still_responding", "Progressive Ratio", "Still-responding curve",
        "Per treatment: the fraction of paired flies whose breaking point reached each "
        "ratio.  Writes analysis/pr_still_responding.png.", RATIO,
        "analysis/pr_still_responding.png"),
    _pl("pr_breaking_point_plots", "Progressive Ratio", "Breaking-point plots",
        "Per DFM: licks per light-on period since training end, the break marked.  "
        "Writes analysis/breaking_point_dfm<id>.png.", RATIO,
        "analysis/breaking_point_dfm*.png"),
    _pl("hedonic_plot", "Hedonic", "Hedonic feeding plot",
        "Median bout duration at each well, one point per chamber, by treatment.  "
        "Writes analysis/hedonic_feeding_plot.png.", HEDONIC,
        "analysis/hedonic_feeding_plot.png"),
)

_SETS: dict[str, tuple[Item, ...]] = {QC: QC_SET, ANALYSES: ANALYSIS_SET, PLOTS: PLOT_SET}


def _check_kind(kind: str) -> str:
    if kind not in _SETS:
        raise ValueError(f"Unknown omit kind {kind!r}; use {', '.join(OMIT_KINDS)}.")
    return kind


def items(kind: str) -> tuple[Item, ...]:
    """The set of *kind*, in panel order."""
    return _SETS[_check_kind(kind)]


def item(kind: str, item_id: str) -> Item:
    """The item *item_id* of the set of *kind*; ``KeyError`` if there is none."""
    for candidate in items(kind):
        if candidate.id == item_id:
            return candidate
    raise KeyError(f"No {kind} item {item_id!r}; known: "
                   f"{', '.join(i.id for i in items(kind))}")


def groups(kind: str) -> list[str]:
    """The group boxes of the set of *kind*, in panel order."""
    return list(dict.fromkeys(i.group for i in items(kind)))


# ---------------------------------------------------------------------------
# What a member can have
# ---------------------------------------------------------------------------

def _meets(experiment: Any, requirement: str) -> bool:
    """Whether *experiment* meets *requirement*: the class for the layout,
    the light QC's coverage for optogenetic, and its Experiment Type's
    modules for a module (ADR-0019)."""
    if requirement == OPTOGENETIC:
        return getattr(experiment, "is_optogenetic", False) is True
    if requirement == TWO_WELL:
        from .two_well_experiment import TwoWellExperiment

        return isinstance(experiment, TwoWellExperiment)
    if requirement in MODULE_REQUIREMENTS:
        etype = getattr(experiment, "experiment_type", None)
        has = getattr(etype, "has_module", None)
        return bool(has(requirement)) if callable(has) else False
    raise ValueError(f"Unknown requirement {requirement!r}; use {', '.join(REQUIREMENTS)}.")


def not_applicable_reason(item: Item, experiment: Any) -> str | None:
    """Why *item* does not apply to *experiment* (its layout, its type's
    modules or its light rule it out), or ``None`` when it applies."""
    for requirement in item.requirements:
        if not _meets(experiment, requirement):
            return _NOT_MET[requirement]
    return None


def applies(item: Item, experiment: Any) -> bool:
    """Whether *experiment* can have *item* at all; the Hub hides it when not."""
    return not_applicable_reason(item, experiment) is None


def _no_timeline(experiment: Any) -> str | None:
    from .program_timeline import timeline_dfms

    if timeline_dfms(experiment):
        return None
    return "No timeline: no DFM has a Program.txt section or firmware 4.0.0 Paradigm columns"


def _no_yoked_group(experiment: Any) -> str | None:
    from .analytics import yoked_groups

    if yoked_groups(experiment):
        return None
    return "No yoked group: no linkage group lights a Yoked chamber"


def _no_yoked_chamber(experiment: Any) -> str | None:
    if experiment.has_yoked_controls():
        return None
    return "No yoked chamber: every chamber group is a solo group"


_UNAVAILABLE = {
    "program_timeline": _no_timeline,
    "yoked_comparison": _no_yoked_group,
    "paired_yoked_diff": _no_yoked_chamber,
}


def unavailable(item: Item, experiment: Any) -> str | None:
    """Why the data cannot answer *item*, which applies (the Hub greys it,
    with this as its tooltip), or ``None``.  ``None`` too for an item that
    does not apply: that is :func:`not_applicable_reason`'s to say."""
    if not applies(item, experiment):
        return None
    check = _UNAVAILABLE.get(item.id)
    return None if check is None else check(experiment)


# ---------------------------------------------------------------------------
# The Selection
# ---------------------------------------------------------------------------

def _global_of(source: Any) -> Mapping:
    """The ``global:`` block of *source*: an Experiment (its resolved
    ``global_config``, the Design's merged under a Member's own), a whole
    config or design mapping (its ``global:``), or a ``global:`` block."""
    if hasattr(source, "global_config"):
        found = getattr(source, "global_config")
        return found if isinstance(found, Mapping) else {}
    if not isinstance(source, Mapping):
        return {}
    if OMIT_KEY not in source and isinstance(source.get("global"), Mapping):
        return source["global"]
    return source


def _ids(value: Any) -> list[str]:
    """An ``omit.<kind>`` value as ids: a list, or one id as a string."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def omitted(experiment_or_config: Any, kind: str) -> frozenset[str]:
    """The ids of the set of *kind* that every run of this experiment leaves
    out: the boxes unticked in the Hub.  *experiment_or_config* is a loaded
    experiment, a config or design mapping, or a ``global:`` block."""
    _check_kind(kind)
    section = _global_of(experiment_or_config).get(OMIT_KEY)
    if not isinstance(section, Mapping):
        return frozenset()
    return frozenset(_ids(section.get(kind)))


def validate_omit(config: Any) -> list[str]:
    """Problems with ``global.omit:`` in *config* (a config or design mapping,
    or a ``global:`` block), for ``pyflic lint``.  A misspelt id leaves
    nothing out: the run then includes what the user meant to drop, and
    nothing on screen says why."""
    g = _global_of(config)
    section = g.get(OMIT_KEY)
    if section is None:
        return []
    if not isinstance(section, Mapping):
        return ["global.omit must be a mapping of qc:, analyses: and plots: to lists of "
                "item ids, e.g. {analyses: [event_statistics]}"]
    problems: list[str] = []
    for kind, value in section.items():
        if kind == SLICES:
            from . import slices as _slices

            if value is not None and not isinstance(value, (list, tuple, str)):
                problems.append(f"global.omit.{SLICES} must be a list of Slice names")
                continue
            declared = _slices.slice_names(g)
            for name in _ids(value):
                if name not in declared:
                    problems.append(f"global.omit.{SLICES} names {name!r}, which the design "
                                    f"does not declare (declared: "
                                    f"{', '.join(declared) or 'none'})")
            continue
        if kind not in _SETS:
            problems.append(f"global.omit.{kind} is not something a run can leave out "
                            f"(use {', '.join(OMIT_KINDS)})")
            continue
        if value is not None and not isinstance(value, (list, tuple, str)):
            problems.append(f"global.omit.{kind} must be a list of item ids")
            continue
        known = [i.id for i in items(kind)]
        for item_id in _ids(value):
            if item_id not in known:
                problems.append(f"global.omit.{kind} names {item_id!r}, which no "
                                f"{RUN_NAMES[kind]} produces (known: {', '.join(known)})")
    return problems


def _project_file(experiment_dir: Path) -> Path | None:
    """``project.yaml`` beside *experiment_dir*, when it is a Member."""
    from .project import PROJECT_FILENAME

    marker = Path(experiment_dir).parent / PROJECT_FILENAME
    return marker if marker.is_file() else None


def selection_file(experiment: Any) -> Path:
    """The file that holds *experiment*'s Selection: its Project's
    ``project.yaml`` for a Member, else its own ``flic_config.yaml``.

    Raises ``ValueError`` for a Member of a Project with no ``design:
    global:`` section: its Members are validated against each other, so one
    of them stating an ``omit:`` the others do not would stop the Project
    loading, and there is no design to hold it.
    """
    from .project import CONFIG_FILENAME, read_design

    directory = getattr(experiment, "experiment_dir", None)
    if directory is None:
        raise ValueError("The experiment has no Experiment Directory to hold its Selection.")
    directory = Path(directory)
    marker = _project_file(directory)
    if marker is None:
        return directory / CONFIG_FILENAME
    if not (read_design(marker.parent).get("global") or {}):
        raise ValueError(
            f"{directory.name} is a Member of a Project whose project.yaml has no design: "
            f"global: section, so its Selection has nowhere to live.  Write the design "
            f"(the Project's Design Editor) first.")
    return marker


def selection_where(experiment: Any, kind: str) -> str:
    """Where the Selection of *kind* is stated, for a log line:
    ``"project.yaml design.global.omit.qc"`` or ``"flic_config.yaml
    global.omit.qc"``."""
    try:
        path = selection_file(experiment)
    except ValueError:
        return f"global.omit.{kind}"
    from .project import PROJECT_FILENAME

    if path.name == PROJECT_FILENAME:
        return f"{PROJECT_FILENAME} design.global.omit.{kind}"
    return f"{path.name} global.omit.{kind}"


def _toggled(section: Any, kind: str, item_id: str, included: bool,
             known: list[str] | None = None) -> dict:
    """The ``omit:`` block *section* with *item_id* of *kind* ticked
    (*included*) or unticked.  Ids are kept in panel order, an id no run
    knows (hand-written, or from a later version) after them; an empty list
    is dropped."""
    block = dict(section) if isinstance(section, Mapping) else {}
    current = _ids(block.get(kind))
    if included:
        current = [i for i in current if i != item_id]
    elif item_id not in current:
        current.append(item_id)
    if known is None:
        known = [i.id for i in items(kind)]
    ordered = [i for i in known if i in current] + [i for i in current if i not in known]
    if ordered:
        block[kind] = ordered
    else:
        block.pop(kind, None)
    order = (*OMIT_KINDS, SLICES)
    return {**{k: block[k] for k in order if k in block},
            **{k: v for k, v in block.items() if k not in order}}


def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a YAML mapping.")
    return payload


def _write_yaml(path: Path, payload: dict) -> None:
    ## As project.create_project_file and adopt_design write it.
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _set_omit(g: dict, block: dict) -> None:
    if block:
        g[OMIT_KEY] = block
    else:
        g.pop(OMIT_KEY, None)


def omitted_slices(experiment_or_config: Any) -> frozenset[str]:
    """The Slices the Hub's runs leave out (``global.omit.slices``)."""
    section = _global_of(experiment_or_config).get(OMIT_KEY)
    if not isinstance(section, Mapping):
        return frozenset()
    return frozenset(_ids(section.get(SLICES)))


def selected_slices(experiment_or_config: Any) -> list[str]:
    """The design's usable Slices a run covers beside the whole recording,
    in the order declared: every one ``omit.slices`` does not leave out."""
    from . import slices as _slices

    g = _global_of(experiment_or_config)
    etype = getattr(experiment_or_config, "experiment_type", None)
    left_out = omitted_slices(experiment_or_config)
    return [n for n in _slices.slice_names(g, etype) if n not in left_out]


def set_slice_included(experiment: Any, name: str, included: bool, *,
                       project: Any = None) -> Path:
    """Tick (*included*) or untick the Slice *name*, written to the
    Selection as :func:`set_included` writes an item."""
    from . import slices as _slices

    declared = _slices.slice_names(_global_of(experiment))
    if name not in declared:
        raise ValueError(f"The design declares no Slice named {name!r} "
                         f"(declared: {', '.join(declared) or 'none'}).")
    return _write_selection(experiment, SLICES, name, included, project=project,
                            known=declared)


def set_included(experiment: Any, kind: str, item_id: str, included: bool, *,
                 project: Any = None) -> Path:
    """Tick (*included*) or untick one item, written to the Selection at once
    so the Hub's buttons and a script's ``run_qc`` / ``run_analysis`` /
    ``run_plots`` steps leave out the same things.  Returns the file written
    (:func:`selection_file`).

    A Member's Selection is the Project's: written to ``project.yaml``
    ``design.global.omit``, and a Member whose own ``global:`` states an
    ``omit:`` has it removed so it inherits the new one (a Member stating a
    different value would stop the Project loading).  A standalone
    experiment's goes to its ``flic_config.yaml`` ``global.omit``.  An empty
    list is removed, and the block with it.

    *experiment*'s resolved ``global_config`` is updated too, so its next
    run honours the change without a reload; pass the loaded *project* (a
    :class:`~pyflic.base.project.Project`) to update its design in memory,
    so the members it loads next see it as well.
    """
    _check_kind(kind)
    item(kind, item_id)                    # an unknown id is a typo, not a choice
    return _write_selection(experiment, kind, item_id, included, project=project)


def _write_selection(experiment: Any, kind: str, item_id: str, included: bool, *,
                     project: Any = None, known: list[str] | None = None) -> Path:
    from .project import PROJECT_FILENAME

    target = selection_file(experiment)
    payload = _read_yaml(target)
    if target.name == PROJECT_FILENAME:
        design = payload.get("design")
        design = dict(design) if isinstance(design, Mapping) else {}
        g = dict(design.get("global") or {})
        _set_omit(g, _toggled(g.get(OMIT_KEY), kind, item_id, included, known))
        design["global"] = g
        payload["design"] = design
    else:
        g = payload.get("global")
        g = dict(g) if isinstance(g, Mapping) else {}
        _set_omit(g, _toggled(g.get(OMIT_KEY), kind, item_id, included, known))
        if g:
            payload["global"] = g
        else:
            payload.pop("global", None)
    _write_yaml(target, payload)

    if target.name == PROJECT_FILENAME:
        inheriting = _members_inherit_omit(target.parent)
        if project is not None:
            project.meta = payload
            project.design = dict(payload.get("design") or {})
            project.design_global = dict(project.design.get("global") or {})
            for name, config in inheriting.items():
                if name in getattr(project, "configs", {}):
                    project.configs[name] = config
    _sync_in_memory(experiment, g.get(OMIT_KEY))
    return target


def _members_inherit_omit(project_dir: Path) -> dict[str, dict]:
    """Remove ``omit:`` from every Member's own ``global:`` under
    *project_dir*, so each inherits the design's; returns the rewritten
    configs by Member name."""
    from .project import CONFIG_FILENAME

    changed: dict[str, dict] = {}
    for entry in sorted(os.listdir(project_dir)):
        path = Path(project_dir) / entry / CONFIG_FILENAME
        if not path.is_file():
            continue
        try:
            config = _read_yaml(path)
        except Exception:  # noqa: BLE001 - an unreadable member is the loader's to report
            continue
        own = config.get("global")
        if not isinstance(own, Mapping) or OMIT_KEY not in own:
            continue
        own = {k: v for k, v in own.items() if k != OMIT_KEY}
        if own:
            config["global"] = own
        else:
            config.pop("global")
        _write_yaml(path, config)
        changed[entry] = config
    return changed


def _sync_in_memory(experiment: Any, block: Any) -> None:
    """Put the new ``omit:`` block into the loaded experiment's resolved
    ``global:`` (``global_config`` and ``config["global"]``, one dict as the
    loader builds them, but set both in case they are not)."""
    targets = [getattr(experiment, "global_config", None)]
    config = getattr(experiment, "config", None)
    if isinstance(config, dict):
        targets.append(config.get("global"))
    for g in targets:
        if isinstance(g, dict):
            if block:
                g[OMIT_KEY] = block
            else:
                g.pop(OMIT_KEY, None)


def all_items() -> tuple[Item, ...]:
    """Every item of the three sets."""
    return QC_SET + ANALYSIS_SET + PLOT_SET
