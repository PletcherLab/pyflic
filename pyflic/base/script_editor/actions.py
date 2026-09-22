"""Action catalogue for the Script Editor.

Every entry here mirrors one branch of the dispatch table in
the dispatch in :mod:`pyflic.base.script_editor.runner`.  This module is the
only place the parameter schema lives — the Palette, Canvas, Inspector, and
YAML preview all consume it.

Adding a new action involves:

1. Registering it in :mod:`pyflic.base.script_editor.runner` (the executor)
2. Appending an entry here (the editor)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ..ui import Category

# ---------------------------------------------------------------------------
# Metric catalogues (imported lazily to avoid cycles).
# ---------------------------------------------------------------------------

def _two_well_binned_metrics() -> list[tuple[str, str, str]]:
    from ..metrics import TWO_WELL_BINNED
    return list(TWO_WELL_BINNED)


def _single_well_binned_metrics() -> list[tuple[str, str, str]]:
    from ..metrics import SINGLE_WELL_BINNED
    return list(SINGLE_WELL_BINNED)


def _well_cmp_metrics() -> list[str]:
    from ..metrics import WELL_CMP_METRICS
    return list(WELL_CMP_METRICS)


def _metric_default_mode() -> dict[str, str]:
    from ..metrics import METRIC_DEFAULT_MODE
    return dict(METRIC_DEFAULT_MODE)


# ---------------------------------------------------------------------------
# Parameter schema — a tiny ad-hoc type system.
# ---------------------------------------------------------------------------

ParamType = Literal[
    "choice",     # fixed list of options
    "metric",     # binned-metric catalogue (two-well vs single-well aware)
    "well_metric",# well-comparison metric catalogue
    "float",      # number with optional unit
    "int",
    "bool",
    "string",
    "list_str",   # comma-separated list of strings
    "list_float", # comma-separated list of floats
]

Requires = Literal["two_well", "hedonic", "progressive_ratio"]

#: Experiment Type name (registry spelling) -> the ``requires`` key above.
_REQUIRES_KEY_BY_TYPE = {"hedonic": "hedonic", "progressiveratio": "progressive_ratio"}


def requires_key_for(experiment_type: str | None) -> str | None:
    """The ``requires`` key an Experiment Type answers to, from any spelling
    the registry accepts (``ProgressiveRatio``, ``progressive_ratio``, …), or
    ``None`` for a Custom Experiment / unknown value."""
    raw = str(experiment_type or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not raw:
        return None
    compact = raw.replace("_", "")
    return _REQUIRES_KEY_BY_TYPE.get(compact)


@dataclass(frozen=True)
class Param:
    """Declarative metadata for one step parameter."""

    key: str
    label: str
    type: ParamType
    note: str = ""
    default: Any = None
    choices: list[str] | None = None       # explicit choice list
    unit: str = ""                          # displayed after numeric inputs
    inheritable: bool = False               # "inherit from UI / script" toggle
    derived_from: str | None = None         # default derived from another key
    required: bool = False                  # required for the step to be valid


@dataclass(frozen=True)
class Action:
    """Declarative metadata for one action (one step kind)."""

    action: str                             # canonical string in YAML
    label: str                              # short human label
    blurb: str                              # one-line description for tiles
    icon: str                               # key from ui.icons._GLYPHS
    category: Category
    produces: Literal["figure", "csv", "pdf", "none"]
    params: list[Param] = field(default_factory=list)
    requires: Requires | None = None        # experiment-type gating
    notes: str = ""                         # longer help for the inspector


# ---------------------------------------------------------------------------
# Reusable param building blocks.
# ---------------------------------------------------------------------------

_START = Param(
    key="start", label="Start minute", type="float",
    note="Time window start (minutes). Leave blank to inherit from the "
         "script-level value, or from the hub's Start Min spinbox.",
    default=None, unit="min", inheritable=True,
)
_END = Param(
    key="end", label="End minute", type="float",
    note="Time window end (minutes). 0 means 'through end of recording'. "
         "Leave blank to inherit.",
    default=None, unit="min", inheritable=True,
)
_BINSIZE = Param(
    key="binsize", label="Bin size", type="float",
    note="Time-bin width in minutes.",
    default=None, unit="min", inheritable=True,
)
_MM_WINDOW = Param(
    key="window", label="Window", type="float",
    note="Sliding-window width in minutes.",
    default=60.0, unit="min", required=True,
)
_MM_STEP = Param(
    key="step", label="Step", type="float",
    note="Distance the window advances each step, in minutes.",
    default=30.0, unit="min", required=True,
)


# ---------------------------------------------------------------------------
# The catalogue.
# ---------------------------------------------------------------------------

ACTIONS: list[Action] = [
    # ── LOAD ─────────────────────────────────────────────────────────────
    Action(
        action="load",
        label="Load experiment",
        blurb="Read DFM CSVs into memory for the rest of the script.",
        icon="load", category=Category.LOAD, produces="none",
        params=[
            Param(key="parallel", label="Load DFMs in parallel", type="bool",
                  note="Loads DFMs on a thread pool. Turn off for easier "
                       "debugging on machines with slow disks.",
                  default=None, inheritable=True),
            _START, _END,
        ],
    ),
    Action(
        action="write_summary",
        label="Write summary",
        blurb="Write summary.txt capturing excluded chambers and experiment metadata.",
        icon="csv", category=Category.LOAD, produces="csv",
        notes="Writes experiment_dir/analysis/summary.txt with experiment metadata, "
              "design table, and any excluded chambers. Safe to call at any point "
              "in the script — e.g. right after remove_chambers.",
    ),
    Action(
        action="remove_chambers",
        label="Remove chambers",
        blurb="Apply chamber exclusions from remove_chambers.csv.",
        icon="remove", category=Category.LOAD, produces="none",
        params=[
            Param(key="group", label="Exclusion group", type="string",
                  note="Named exclusion group in remove_chambers.csv. "
                       "Defaults to the script name when left blank.",
                  default=None),
        ],
        notes="Reads remove_chambers.csv in the project directory and removes "
              "the chambers listed under the named group from the loaded "
              "experiment design.  Leave 'group' blank to use the script's "
              "own name as the group key.",
    ),

    # ── ANALYZE ──────────────────────────────────────────────────────────
    Action(
        action="basic_analysis",
        label="Basic analysis",
        blurb="Run the built-in feeding-summary analysis pipeline.",
        icon="basic", category=Category.ANALYZE, produces="none",
        params=[_START, _END],
    ),
    Action(
        action="run_qc",
        label="QC reports",
        blurb="Write per-DFM QC reports and signal plots into qc/.",
        icon="qc", category=Category.QC, produces="none",
        notes="Integrity report, data breaks, simultaneous feeding and "
              "bleeding matrices (two-well), and the per-DFM Raw Signal, "
              "Baselined, and Cumulative Licks plots the QC Viewer shows.",
    ),
    Action(
        action="feeding_csv",
        label="Write feeding CSV",
        blurb="Dump the per-chamber feeding summary to a CSV file.",
        icon="csv", category=Category.ANALYZE, produces="csv",
        params=[_START, _END],
    ),
    Action(
        action="binned_csv",
        label="Write binned feeding CSV",
        blurb="Per-chamber feeding metrics binned over time.",
        icon="binned", category=Category.ANALYZE, produces="csv",
        params=[_BINSIZE, _START, _END],
    ),
    Action(
        action="weighted_duration",
        label="Write weighted-duration CSV",
        blurb="Weighted-duration summary (hedonic paradigm).",
        icon="weighted", category=Category.ANALYZE, produces="csv",
        requires="hedonic",
        params=[_START, _END],
    ),
    Action(
        ## The action key stays ``tidy_export``: it is written into every
        ## script on disk, and the label is what people read, not what they
        ## type.  "Tidy" named the shape of the file; this names what is in
        ## it — one row per feeding or tasting event, with its own time,
        ## duration, licks and intensity.
        action="tidy_export",
        label="Event statistics",
        blurb="One row per individual event: time, duration, licks, intensity.",
        icon="tidy", category=Category.ANALYZE, produces="csv",
        params=[
            Param(key="kind", label="Event kind", type="choice",
                  choices=["feeding", "tasting"],
                  note="Which event class to export.",
                  default="feeding"),
            _START, _END,
        ],
    ),
    Action(
        action="bootstrap",
        label="Bootstrap CIs",
        blurb="Non-parametric bootstrap confidence intervals for a metric.",
        icon="bootstrap", category=Category.ANALYZE, produces="csv",
        params=[
            Param(key="metric", label="Metric", type="string",
                  note="Metric name (e.g. PI, MedDuration, Events).",
                  default="PI", required=True),
            Param(key="mode", label="Mode", type="choice",
                  choices=["total", "A", "B", "mean_ab"],
                  note="Which well/aggregation.",
                  default="total"),
            Param(key="n_boot", label="Bootstrap iterations", type="int",
                  note="More = tighter CIs but slower.",
                  default=2000),
            Param(key="ci", label="CI level", type="float",
                  note="e.g. 0.95 for 95% confidence intervals.",
                  default=0.95),
            Param(key="seed", label="Random seed", type="int",
                  note="Optional — set for reproducibility.",
                  default=0),
            _START, _END,
        ],
    ),
    Action(
        action="compare",
        label="Compare treatments",
        blurb="ANOVA or linear mixed model across treatment groups.",
        icon="compare", category=Category.ANALYZE, produces="csv",
        params=[
            Param(key="metric", label="Metric", type="string",
                  note="Metric to compare across groups.",
                  default="MedDuration", required=True),
            Param(key="mode", label="Mode", type="choice",
                  choices=["total", "A", "B", "mean_ab"],
                  note="Which well/aggregation.",
                  default="A"),
            Param(key="model", label="Statistical model", type="choice",
                  choices=["aov", "lmm"],
                  note="aov = analysis of variance; lmm = linear mixed model.",
                  default="aov"),
            Param(key="factors", label="Factors", type="list_str",
                  note="Optional comma-separated list of design factors to "
                       "include in the model.",
                  default=None),
            _START, _END,
        ],
    ),
    Action(
        action="light_phase_summary",
        label="Light-phase summary",
        blurb="Feeding summary split by light / dark phase.",
        icon="lightphase", category=Category.ANALYZE, produces="csv",
        params=[],
    ),
    Action(
        action="param_sensitivity",
        label="Parameter sensitivity sweep",
        blurb="Vary a detection parameter and record the effect on metrics.",
        icon="sensitivity", category=Category.ANALYZE, produces="csv",
        params=[
            Param(key="parameter", label="Parameter to sweep", type="choice",
                  choices=["feeding_event_link_gap", "feeding_threshold",
                           "feeding_minimum", "tasting_minimum",
                           "tasting_maximum", "feeding_minevents",
                           "tasting_minevents", "baseline_window_minutes",
                           "samples_per_second"],
                  note="Detection parameter to vary.",
                  default="feeding_event_link_gap", required=True),
            Param(key="values", label="Values", type="list_float",
                  note="Comma-separated numeric values to sweep through.",
                  default=None, required=True),
            _START, _END,
        ],
    ),
    Action(
        action="transition_matrix",
        label="Bout transition matrix",
        blurb="Transition probabilities between bout types.",
        icon="transition", category=Category.ANALYZE, produces="csv",
        requires="two_well",
        params=[_START, _END],
    ),
    Action(
        action="pdf_report",
        label="Write PDF report",
        blurb="Bundle of binned plots + tables into one PDF.",
        icon="pdf", category=Category.ANALYZE, produces="pdf",
        params=[
            Param(key="metrics", label="Metrics to include", type="list_str",
                  note="Comma-separated metric names, e.g. "
                       "Licks, Events, MedDuration.",
                  default="Licks, Events, MedDuration"),
            _BINSIZE, _START, _END,
        ],
    ),

    # ── PLOTS ────────────────────────────────────────────────────────────
    Action(
        action="plot_feeding_summary",
        label="Feeding summary plot",
        blurb="Bar / jitter plot of per-chamber feeding metrics.",
        icon="feeding", category=Category.PLOTS, produces="figure",
        params=[_START, _END],
    ),
    Action(
        action="plot_binned",
        label="Binned time-course plot",
        blurb="Line plot of a metric binned over time, by treatment.",
        icon="binned", category=Category.PLOTS, produces="figure",
        params=[
            Param(key="metric", label="Metric", type="metric",
                  note="Which metric to plot.",
                  default="Licks", required=True),
            Param(key="mode", label="Mode", type="choice",
                  choices=["total", "A", "B", "mean_ab"],
                  note="Which well/aggregation.",
                  default=None, derived_from="metric"),
            _BINSIZE, _START, _END,
        ],
    ),
    Action(
        action="plot_dot",
        label="Dot plot",
        blurb="Jittered dot plot of a (non-binned) metric by treatment.",
        icon="dot", category=Category.PLOTS, produces="figure",
        params=[
            Param(key="metric", label="Metric", type="metric",
                  note="Which metric to plot.",
                  default="Licks", required=True),
            Param(key="mode", label="Mode", type="choice",
                  choices=["total", "A", "B", "mean_ab"],
                  note="Which well/aggregation.",
                  default=None, derived_from="metric"),
            _START, _END,
        ],
        notes="One point per chamber, mean ± SEM over each treatment. On a "
              "Progressive Ratio member it is instead one point per Chamber "
              "Group — the paired-minus-yoked difference in the metric, "
              "panelled by Facet — because pooling a paired fly with its own "
              "yoked control averages an effect with its control.",
    ),
    Action(
        action="plot_moving_window",
        label="Moving window metric (treatments)",
        blurb="Treatment mean ± SEM of any feeding metric over a sliding window.",
        icon="plot", category=Category.PLOTS, produces="figure",
        params=[
            Param(key="metric", label="Metric", type="metric",
                  note="Which metric to plot.",
                  default="MedDuration", required=True),
            Param(key="mode", label="Mode", type="choice",
                  choices=["total", "A", "B", "mean_ab"],
                  note="Which well/aggregation.",
                  default=None, derived_from="metric"),
            _MM_WINDOW, _MM_STEP, _START, _END,
        ],
        notes="The moving-window counterpart to the binned plot: the same "
              "metric set, but values are computed over overlapping windows "
              "(see 'window'/'step') rather than disjoint bins, giving a "
              "smoother time course. Per-treatment mean ± SEM across chambers.",
    ),
    Action(
        action="plot_moving_median_chambers",
        label="Moving median duration (chambers)",
        blurb="Time-dependent median bout duration, one line per chamber, by treatment.",
        icon="binned", category=Category.PLOTS, produces="figure",
        params=[
            _MM_WINDOW, _MM_STEP,
            Param(key="mode", label="Mode", type="choice",
                  choices=["mean_ab", "A", "B"],
                  note="Two-well only: average wells A/B or pick one. "
                       "Ignored for single-well designs.",
                  default="mean_ab"),
            _START, _END,
        ],
        notes="Median feeding-bout duration computed over a sliding window "
              "(see 'window'/'step'), drawn as one trajectory per chamber and "
              "coloured by treatment. Not part of any standard analysis.",
    ),
    Action(
        action="plot_moving_median_treatment",
        label="Moving median duration (treatments)",
        blurb="Treatment mean ± SEM of the time-dependent median bout duration.",
        icon="plot", category=Category.PLOTS, produces="figure",
        params=[
            _MM_WINDOW, _MM_STEP,
            Param(key="mode", label="Mode", type="choice",
                  choices=["mean_ab", "A", "B"],
                  note="Two-well only: average wells A/B or pick one. "
                       "Ignored for single-well designs.",
                  default="mean_ab"),
            _START, _END,
        ],
        notes="Per-treatment mean (± SEM across chambers) of the sliding-window "
              "median feeding-bout duration at each time point. Not part of any "
              "standard analysis.",
    ),
    Action(
        action="plot_well_comparison",
        label="Well A vs B comparison",
        blurb="Side-by-side comparison of a metric for well A vs well B.",
        icon="well", category=Category.PLOTS, produces="figure",
        requires="two_well",
        params=[
            Param(key="metric", label="Metric", type="well_metric",
                  note="Which metric to compare.",
                  default="MedDuration", required=True),
            _START, _END,
        ],
    ),
    Action(
        action="plot_hedonic",
        label="Hedonic feeding plot",
        blurb="Hedonic-specific feeding plot.",
        icon="feeding", category=Category.PLOTS, produces="figure",
        requires="hedonic",
        params=[_START, _END],
    ),
    Action(
        action="paired_yoked_diff",
        label="Paired − yoked difference table",
        blurb="Per chamber group, per Facet: paired minus yoked (progressive ratio).",
        icon="csv", category=Category.ANALYZE, produces="csv",
        requires="progressive_ratio",
        params=[],
        notes="Writes analysis/paired_yoked_diff.csv — one row per chamber "
              "group per Facet (Training, Test), the primary table for "
              "Progressive Ratio statistics.",
    ),
    Action(
        action="plot_pr_cumulative_diff",
        label="Cumulative difference curve",
        blurb="Paired − yoked cumulative licks since training end, by treatment.",
        icon="plot", category=Category.PLOTS, produces="figure",
        requires="progressive_ratio",
        params=[
            Param(key="binsize", label="Bin size", type="float", unit="min",
                  note="Resolution of the cumulative curve.", default=1.0),
        ],
        notes="Mean ± SEM per treatment across chamber groups, the individual "
              "group traces faint behind. Also writes "
              "analysis/pr_cumulative_diff.csv, which the Project pools for the "
              "timecourse_pr_diff figure.",
    ),
    Action(
        action="plot_pr_cumulative_licks",
        label="Training-aligned traces (QC)",
        blurb="Per DFM: paired and yoked cumulative licks since training end.",
        icon="plot", category=Category.PLOTS, produces="figure",
        requires="progressive_ratio",
        params=[
            Param(key="binsize", label="Bin size", type="float", unit="min",
                  note="Resolution of the cumulative curve.", default=1.0),
        ],
        notes="One panel per chamber group; light-on bins are drawn as points. "
              "A QC figure, not a result.",
    ),
    Action(
        action="plot_breaking_point",
        label="Breaking-point plots",
        blurb="Per-DFM ΔLicks per light-on period since training end.",
        icon="plot", category=Category.PLOTS, produces="figure",
        requires="progressive_ratio",
        params=[],
        notes="One panel per chamber. The training end and roles come from the "
              "config's paired_chambers and the data's training flag; there is "
              "no configuration index any more.",
    ),
]

_ACTION_INDEX: dict[str, Action] = {a.action: a for a in ACTIONS}


def get_action(name: str) -> Action | None:
    return _ACTION_INDEX.get(name)


def actions_by_category() -> dict[Category, list[Action]]:
    """Return actions grouped by category, preserving insertion order."""
    out: dict[Category, list[Action]] = {}
    for a in ACTIONS:
        out.setdefault(a.category, []).append(a)
    return out


# ---------------------------------------------------------------------------
# Metric helpers — bridge between a step's parameter schema and the
# layout-aware metric lists in pyflic.base.metrics.
# ---------------------------------------------------------------------------

def metric_choices(kind: str, is_two_well: bool) -> list[tuple[str, str]]:
    """Return [(display_label, metric_key), ...] for a metric-typed param."""
    if kind == "well_metric":
        return [(m, m) for m in _well_cmp_metrics()]
    catalogue = _two_well_binned_metrics() if is_two_well else _single_well_binned_metrics()
    return [(display, metric) for display, metric, _mode in catalogue]


def default_mode_for_metric(metric: str) -> str | None:
    return _metric_default_mode().get(metric)


# ---------------------------------------------------------------------------
# Step helpers.
# ---------------------------------------------------------------------------

def describe_step(step: dict[str, Any]) -> str:
    """Return a one-line chip summary used on canvas step cards.

    Non-None values from ``step`` (excluding ``action``) are rendered as
    ``key: value`` segments joined by ' · '.
    """
    action_name = step.get("action", "")
    act = get_action(action_name)
    if act is None:
        return "(unknown action)"
    parts: list[str] = []
    for p in act.params:
        if p.key not in step:
            continue
        v = step[p.key]
        if v is None or v == "":
            continue
        if isinstance(v, list):
            rendered = ", ".join(str(x) for x in v)
        elif isinstance(v, bool):
            rendered = "on" if v else "off"
        else:
            rendered = str(v)
        if p.unit:
            rendered = f"{rendered} {p.unit}"
        parts.append(f"{p.key}: {rendered}")
    return " · ".join(parts) if parts else "(no parameters)"


def validation_issues(step: dict[str, Any], *, experiment_type: str | None) -> list[str]:
    """Return a list of human-readable warnings for *step*."""
    issues: list[str] = []
    act = get_action(step.get("action", ""))
    if act is None:
        return [f"Unknown action: {step.get('action')!r}"]
    if act.requires and act.requires != experiment_type:
        issues.append(
            f"Requires an experiment of type {act.requires!r}; current is "
            f"{experiment_type or 'unspecified'}."
        )
    for p in act.params:
        if p.required and p.key not in step:
            issues.append(f"Missing required field: {p.label} ({p.key}).")
    return issues
