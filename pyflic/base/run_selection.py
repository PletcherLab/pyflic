"""Run a member's QC Set, Analysis Set or Plot Set, minus its Selection (ADR-0018).

:func:`run_qc`, :func:`run_analysis` and :func:`run_plots` are what the Hub's
*Run QC*, *Run analysis* and *Generate plots* run (through the script actions
of the same names) and what a script's ``run_qc`` / ``run_analysis`` /
``run_plots`` steps run, so a click and a script produce the same files.  A
Project's *Analyze all* and a Batch Run's ``run_all_analyses`` are not these:
they keep Basic analysis and do not read the Selection.  Each run:

* logs what it will run, what it leaves out (a choice: the Selection, or
  what the caller adds) and, separately, what does not apply (a property of
  the data);
* runs each remaining item by the function its single script action calls,
  with the same inputs.

No run removes a file: the earlier outputs of an item left out stay where
they are.

*Run analysis* writes the core first (the auto-removal and
``removed_chambers.csv``, ``summary.txt``, ``feeding_summary.csv`` and, with
Facets, ``feeding_summary_facet.csv``), because every other item reads it,
and the experiment report last, so it reflects the run.  Its Selection lines
go into ``summary.txt``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from . import selections as sel
from .selections import ANALYSES, PLOTS, QC, Item

#: The time window every item reads: all of the loaded experiment, the whole
#: recording or the Slice it was loaded as (ADR-0020).
WHOLE_RECORDING: tuple[float, float] = (0.0, 0.0)
#: :func:`each_slice`'s default: the whole recording and every Slice the
#: Selection does not leave out.
SELECTED = object()


def each_slice(exp, *, only: Any = SELECTED, log=print) -> Iterator[Any]:
    """The experiments a Hub run covers (ADR-0020): the whole recording,
    then every Slice ``global.omit.slices`` does not leave out, each loaded
    the way *exp* was (*exp* itself is reused for the one it is).  *only*
    names one Slice instead (``None``: the whole recording alone).

    A Member that does not cover a Slice has that recorded in
    ``slices/<name>/slice.json`` and is logged, not raised: it is left out of
    that Slice, not failed."""
    from . import slices as _slices

    names = [None] + sel.selected_slices(exp) if only is SELECTED else [only]
    for name in names:
        if name == getattr(exp, "slice_name", None):
            yield exp
            continue
        if name is not None:
            log(f"--- Slice {name} ---")
        try:
            yield exp.load_slice(name)
        except _slices.SliceNotCovered as err:
            if exp.experiment_dir is not None:
                _slices.record_not_covered(exp.experiment_dir, err)
            log(f"Slice {name}: not covered, left out ({err.reason})")
#: The resolution of the Progressive Ratio's cumulative curves, the table's
#: and the figures' — their script actions' default (and Basic analysis's),
#: not the Bin size, which is the binned summary's, the report's and the
#: binned time course's.
PR_CURVE_BINSIZE_MIN = 1.0
#: The experiment report's time courses: the ``pdf_report`` action's default.
REPORT_METRICS: tuple[str, ...] = ("Licks", "Events", "MedDuration")
#: The Well A vs B figure's metric, as its button drew it.
WELL_COMPARISON_METRIC = "MedDuration"
#: The QC reports that read :meth:`~pyflic.base.experiment.Experiment.compute_qc_for_reports`.
_READS_QC_RESULTS = frozenset({"integrity", "data_breaks", "simultaneous_feeding", "bleeding"})


@dataclass
class SelectionRun:
    """What one run did."""

    #: The items run, in order.
    ran: list[Item] = field(default_factory=list)
    #: The items left out by choice.
    left_out: list[Item] = field(default_factory=list)
    #: The items not run because they do not apply, or the data cannot
    #: answer them, each with the reason.
    not_applicable: list[tuple[Item, str]] = field(default_factory=list)
    #: ``(title, figure)`` for the Hub's tabs: :func:`run_plots` with
    #: ``keep_figures``, else empty.
    figures: list[tuple[str, Any]] = field(default_factory=list)
    #: :data:`~pyflic.base.selections.QC`, ``ANALYSES`` or ``PLOTS``.
    kind: str = ""
    #: Every file the run wrote, the core's included.
    written: list[Path] = field(default_factory=list)
    #: The files each item that ran wrote, by id.
    item_outputs: dict[str, list[Path]] = field(default_factory=dict)
    #: Why each left-out item was left out, by id.
    reasons: dict[str, str] = field(default_factory=dict)

    def lines(self) -> list[str]:
        """One line per left-out and per not-applicable item, as the log and
        ``summary.txt`` state them."""
        out = [f"Left out: {i.label} ({i.id}) — {self.reasons.get(i.id, 'left out')}"
               for i in self.left_out]
        out += [f"Not applicable: {i.label} ({i.id}) — {why}"
                for i, why in self.not_applicable]
        return out


class _Context:
    """What the item functions share within one run."""

    def __init__(self, exp, log: Callable[[str], None], *, binsize_min: float = 30.0,
                 metric: str = "Licks", keep_figures: bool = False):
        self.exp = exp
        self.log = log
        self.binsize_min = float(binsize_min)
        self.metric = str(metric)
        self.keep_figures = bool(keep_figures)
        self.figures: list[tuple[str, Any]] = []
        self.qc_dir: Path | None = None
        self._qc: dict | None = None

    def qc_results(self) -> dict:
        """The QC computed once for every QC report of the run."""
        if self._qc is None:
            self._qc = self.exp.compute_qc_for_reports()
        return self._qc

    def figure(self, title: str, fig: Any) -> None:
        """Keep *fig* for a tab, or let it go."""
        if self.keep_figures:
            self.figures.append((title, fig))
        elif type(fig).__module__.startswith("matplotlib"):
            import matplotlib.pyplot as plt

            plt.close(fig)


def _save(fig: Any, path: Path) -> None:
    """Save as the script actions save their figures."""
    from .script_editor.runner import _save_figure

    _save_figure(fig, path, lambda _message: None)


def _relative(path: Path, root: Path | None) -> str:
    try:
        return Path(path).relative_to(root).as_posix() if root is not None else str(path)
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# The QC Set
# ---------------------------------------------------------------------------

def _qc_report(item_id: str):
    def run(exp, ctx: _Context) -> list[Path]:
        qc = ctx.qc_results() if item_id in _READS_QC_RESULTS else None
        return exp.write_qc_item(item_id, ctx.qc_dir, qc)
    return run


def _pr_light_qc(exp, ctx: _Context) -> list[Path]:
    paths = list(exp.write_light_qc().values())
    lines = exp.light_qc_lines()
    ctx.log("\n".join(lines) if lines else "Light QC: every chamber group is ok.")
    return paths


def _per_dfm_figures(stem: str, method: str, title: str, **kwargs):
    """One figure per DFM, ``analysis/<stem>_dfm<id>.png``, as the
    Progressive Ratio's per-DFM plot actions draw them."""
    def run(exp, ctx: _Context) -> list[Path]:
        written: list[Path] = []
        for dfm_id in sorted(exp.dfms):
            fig = getattr(exp, method)(dfm_id, **kwargs)
            path = Path(exp.analysis_dir) / f"{stem}_dfm{dfm_id}.png"
            _save(fig, path)
            ctx.figure(f"{title} — DFM {dfm_id}", fig)
            written.append(path)
        return written
    return run


# ---------------------------------------------------------------------------
# The Analysis Set
# ---------------------------------------------------------------------------

def _binned_summary(exp, ctx: _Context) -> list[Path]:
    df = exp.binned_feeding_summary(binsize_min=ctx.binsize_min,
                                    range_minutes=WHOLE_RECORDING, save=True)
    ctx.log(f"  Binned rows: {len(df)} ({ctx.binsize_min:g}-min bins)")
    return [Path(exp.analysis_dir) / "binned_feeding_summary.csv"]


def _event_statistics(exp, ctx: _Context) -> list[Path]:
    ## The tidy_export action with its default kind, as the button ran it.
    from . import analytics

    df = analytics.tidy_events(exp, kind="feeding")
    out = Path(exp.analysis_dir) / "tidy_feeding_events.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    ctx.log(f"  {len(df)} bouts")
    return [out]


def _light_tables(exp, ctx: _Context) -> list[Path]:
    from .light_summary import write as write_light_tables

    return list(write_light_tables(exp, range_minutes=WHOLE_RECORDING).values())


def _yoked_comparison(exp, ctx: _Context) -> list[Path]:
    from .analytics import write_yoked_comparison

    path = write_yoked_comparison(exp, range_minutes=WHOLE_RECORDING)
    return [] if path is None else [path]


def _breaking_point(exp, ctx: _Context) -> list[Path]:
    path = exp.write_breaking_point()
    ctx.log("\n".join(exp.breaking_point_lines()))
    return [path]


def _weighted_duration(exp, ctx: _Context) -> list[Path]:
    exp.weighted_duration_summary(save=True, range_minutes=WHOLE_RECORDING)
    return [Path(exp.analysis_dir) / "treatment_means.csv"]


def _experiment_report(exp, ctx: _Context) -> list[Path]:
    from .pdf_report import write_experiment_report

    return [write_experiment_report(exp, metrics=REPORT_METRICS,
                                    binsize_min=ctx.binsize_min,
                                    range_minutes=WHOLE_RECORDING)]


# ---------------------------------------------------------------------------
# The Plot Set
# ---------------------------------------------------------------------------

def _one_figure(name: Callable[[_Context], str], title: Callable[[_Context], str],
                draw: Callable[[Any, _Context], Any]):
    """One figure saved as ``analysis/<name>``, as its plot action saves it."""
    def run(exp, ctx: _Context) -> list[Path]:
        fig = draw(exp, ctx)
        path = Path(exp.analysis_dir) / name(ctx)
        _save(fig, path)
        ctx.figure(title(ctx), fig)
        return [path]
    return run


def _metric_mode(ctx: _Context) -> str:
    from .metrics import METRIC_DEFAULT_MODE

    return METRIC_DEFAULT_MODE.get(ctx.metric, "total")


def _safe_metric(ctx: _Context) -> str:
    return ctx.metric.replace("/", "_")


def _hedonic_plot(exp, ctx: _Context) -> list[Path]:
    fig = exp.hedonic_feeding_plot(save=True, range_minutes=WHOLE_RECORDING)
    ctx.figure("Hedonic Feeding Plot", fig)
    return [Path(exp.analysis_dir) / "hedonic_feeding_plot.png"]


#: Each item's function: ``(experiment, context) -> files written``.
_RUNNERS: dict[str, Callable[[Any, _Context], list[Path]]] = {
    **{item_id: _qc_report(item_id) for item_id in (
        "integrity", "data_breaks", "raw_signal", "baselined_signal", "cumulative_licks",
        "simultaneous_feeding", "bleeding", "opto_light_qc", "opto_light_figures",
        "program_timeline")},
    "pr_light_qc": _pr_light_qc,
    "pr_light_event_figures": _per_dfm_figures(
        "pr_light_events", "plot_light_events_dfm", "Licks per light event"),
    "pr_resting_level_figures": _per_dfm_figures(
        "pr_resting_level", "plot_resting_level_dfm", "Sucrose Well resting level"),

    "binned_summary": _binned_summary,
    "event_statistics": _event_statistics,
    "light_tables": _light_tables,
    "yoked_comparison": _yoked_comparison,
    "paired_yoked_diff": lambda exp, ctx: [exp.write_paired_yoked_diff()],
    "cumulative_diff": lambda exp, ctx: [exp.write_cumulative_diff(
        binsize_min=PR_CURVE_BINSIZE_MIN)],
    "breaking_point": _breaking_point,
    "weighted_duration": _weighted_duration,
    "experiment_report": _experiment_report,

    "feeding_summary_plot": _one_figure(
        lambda ctx: "feeding_summary.png", lambda ctx: "Feeding Summary",
        lambda exp, ctx: exp.plot_feeding_summary(range_minutes=WHOLE_RECORDING)),
    "well_comparison": _one_figure(
        lambda ctx: f"well_comparison_{WELL_COMPARISON_METRIC}.png",
        lambda ctx: f"Well A vs B: {WELL_COMPARISON_METRIC}",
        lambda exp, ctx: exp.facet_plot_well_durations(
            metric=WELL_COMPARISON_METRIC, range_minutes=WHOLE_RECORDING)),
    "binned_time_course": _one_figure(
        lambda ctx: f"binned_{_safe_metric(ctx)}.png", lambda ctx: f"Binned: {ctx.metric}",
        lambda exp, ctx: exp.plot_binned_metric_by_treatment(
            metric=ctx.metric, two_well_mode=_metric_mode(ctx),
            binsize_min=ctx.binsize_min, range_minutes=WHOLE_RECORDING)),
    "dot_plot": _one_figure(
        lambda ctx: f"dot_{_safe_metric(ctx)}.png", lambda ctx: f"Dot: {ctx.metric}",
        lambda exp, ctx: exp.plot_dot_metric_by_treatment(
            metric=ctx.metric, two_well_mode=_metric_mode(ctx),
            range_minutes=WHOLE_RECORDING)),
    "pr_cumulative_diff_plot": _one_figure(
        lambda ctx: "pr_cumulative_diff.png", lambda ctx: "Paired − yoked cumulative licks",
        lambda exp, ctx: exp.plot_cumulative_diff(binsize_min=PR_CURVE_BINSIZE_MIN)),
    "pr_cumulative_licks": _per_dfm_figures(
        "pr_cumulative_licks", "plot_cumulative_licks_dfm", "Training-aligned traces",
        binsize_min=PR_CURVE_BINSIZE_MIN),
    "pr_still_responding": _one_figure(
        lambda ctx: "pr_still_responding.png", lambda ctx: "Still responding",
        lambda exp, ctx: exp.plot_still_responding()),
    "pr_breaking_point_plots": _per_dfm_figures(
        "breaking_point", "plot_breaking_point_dfm", "Breaking Point"),
    "hedonic_plot": _hedonic_plot,
}


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def _plan(exp, kind: str, omit: Iterable[str] | None) -> tuple[SelectionRun, list[Item]]:
    """Sort the set of *kind* into what runs, what is left out and what does
    not apply.  *omit* ``None`` is the experiment's Selection; otherwise it
    is used as given, an id outside the Selection said to be left out by
    this run."""
    selection = sel.omitted(exp, kind)
    if omit is None:
        chosen = selection
    else:
        chosen = frozenset([omit] if isinstance(omit, str) else (str(i) for i in omit))
    where = sel.selection_where(exp, kind)
    run = SelectionRun(kind=kind)
    todo: list[Item] = []
    for item in sel.items(kind):
        why = sel.not_applicable_reason(item, exp)
        if why is not None:
            run.not_applicable.append((item, why))
            continue
        if item.id in chosen:
            run.left_out.append(item)
            run.reasons[item.id] = (f"unticked ({where})" if item.id in selection
                                    else "left out by this run")
            continue
        why = sel.unavailable(item, exp)
        if why is not None:
            run.not_applicable.append((item, why))
            continue
        todo.append(item)
    return run, todo


def _announce(run: SelectionRun, todo: list[Item], log) -> None:
    name = sel.RUN_NAMES[run.kind]
    if todo:
        log(f"{name}: {len(todo)} item(s) — {', '.join(i.label for i in todo)}")
    else:
        log(f"{name}: nothing to run")
    for item in run.left_out:
        log(f"  Left out — {item.label}: {run.reasons[item.id]}")
    for item, why in run.not_applicable:
        log(f"  Not applicable — {item.label}: {why}")


def _describe(written: list[Path], root: Path | None) -> str:
    if not written:
        return "  nothing to write"
    if len(written) == 1:
        return f"  Wrote {_relative(written[0], root)}"
    folders = {_relative(p.parent, root) for p in written}
    where = f"{folders.pop()}/" if len(folders) == 1 else f"{len(folders)} folders"
    return f"  Wrote {len(written)} files into {where}"


def _run_items(exp, run: SelectionRun, todo: list[Item], ctx: _Context) -> None:
    root = getattr(exp, "experiment_dir", None)
    for n, item in enumerate(todo, 1):
        ctx.log(f"[{n}/{len(todo)}] {item.label}")
        written = _RUNNERS[item.id](exp, ctx)
        run.written += written
        run.item_outputs[item.id] = list(written)
        run.ran.append(item)
        ctx.log(_describe(written, root))


def run_qc(exp, *, omit: Iterable[str] | None = None, log=print) -> SelectionRun:
    """*Run QC*: the QC Set minus *omit* (default: the experiment's
    Selection), into ``qc/`` (the Progressive Ratio's light QC into
    ``analysis/``, as its action writes it).

    The QC reports are computed once and written item by item, by the
    writers :meth:`~pyflic.base.experiment.Experiment.write_qc_reports`
    uses, so with nothing left out this writes what it writes.
    """
    run, todo = _plan(exp, QC, omit)
    _announce(run, todo, log)
    ctx = _Context(exp, log)
    ctx.qc_dir = exp.qc_output_dir()
    _run_items(exp, run, todo, ctx)
    exp.qc_report_dir = ctx.qc_dir
    ## A Slice's outputs say which definition of it they describe (ADR-0020).
    exp.write_slice_record(qc_only=True)
    log(f"{sel.RUN_NAMES[QC]} complete — {len(run.ran)} item(s) → {ctx.qc_dir}")
    return run


def _write_core(exp, run: SelectionRun, todo: list[Item], ctx: _Context) -> list[Path]:
    """The core, which every other item reads: the auto-removal and
    ``removed_chambers.csv``, ``summary.txt`` (with this run's Selection),
    ``feeding_summary.csv`` and, with Facets, ``feeding_summary_facet.csv`` —
    as Basic analysis writes them."""
    ctx.log("Core (always runs): auto-removal and removed_chambers.csv, summary.txt, "
            "feeding_summary.csv and, with Facets, feeding_summary_facet.csv")
    if exp.filtered_chambers is None:
        removed = exp.apply_auto_removal(range_minutes=WHOLE_RECORDING)
        ctx.log(f"  Auto-removal: {len(removed)} chamber(s) removed.")
    else:
        ctx.log("  Auto-removal already applied to this experiment — kept its record.")
    written: list[Path] = []
    removed_path = exp.write_removed_chambers()
    if removed_path is not None:
        written.append(removed_path)
    summary = exp.write_summary()
    _append_selection(summary, run, todo)
    written.append(summary)
    feeding = exp.write_feeding_summary(range_minutes=WHOLE_RECORDING)
    written.append(feeding)
    ## Beside it, the settings it was computed from, which the stale rule
    ## compares with the config as it stands (ADR-0022).
    from . import analysis_settings

    settings_record = feeding.parent / analysis_settings.RECORD_FILENAME
    if getattr(exp, "analysis_settings", None) is not None and settings_record.is_file():
        written.append(settings_record)
    facet = exp.write_feeding_summary_facet()
    if facet is None:
        ctx.log("  Not faceted: no feeding_summary_facet.csv.")
    else:
        written.append(facet)
    ## A Slice's results say which definition of it they describe, so a
    ## later edit of its bounds, groups or cutoffs shows them stale (ADR-0020).
    record = exp.write_slice_record()
    if record is not None:
        written.append(record)
    ctx.log(_describe(written, getattr(exp, "experiment_dir", None)))
    return written


def _append_selection(summary: Path, run: SelectionRun, todo: list[Item]) -> None:
    """Add the Analysis Set's Selection to ``summary.txt``: what this run
    includes, what it leaves out (a choice) and what does not apply."""
    title = "Analysis Set (Run analysis)"
    lines = ["", title, "-" * len(title),
             "Core (always runs): auto-removal, removed_chambers.csv, summary.txt, "
             "feeding_summary.csv, feeding_summary_facet.csv (with Facets)",
             "Included: " + (", ".join(f"{i.label} ({i.id})" for i in todo) or "(none)")]
    lines += run.lines() or ["Left out: (none)"]
    with open(summary, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def run_analysis(exp, *, binsize_min: float = 30.0, omit: Iterable[str] | None = None,
                 log=print) -> SelectionRun:
    """*Run analysis*: the core, then the Analysis Set minus *omit* (default:
    the experiment's Selection), the experiment report last.  *binsize_min*
    is the binned summary's and the report's bin."""
    run, todo = _plan(exp, ANALYSES, omit)
    _announce(run, todo, log)
    ctx = _Context(exp, log, binsize_min=binsize_min)
    run.written += _write_core(exp, run, todo, ctx)
    ## The report reads what the others wrote, so it goes last whatever the
    ## set's order says.
    ordered = ([i for i in todo if i.id != "experiment_report"]
               + [i for i in todo if i.id == "experiment_report"])
    _run_items(exp, run, ordered, ctx)
    log(f"{sel.RUN_NAMES[ANALYSES]} complete — the core and {len(run.ran)} item(s)")
    return run


#: Between a figure's title and the Slice it was drawn for (ADR-0020).
SLICE_TITLE_MARK = " · Slice "


def slice_title(title: str, slice_name: str | None) -> str:
    """*title* naming the Slice its figure was drawn for —
    ``"Feeding Summary · Slice early"`` — and as it is for the whole
    recording's."""
    return str(title) if slice_name is None else f"{title}{SLICE_TITLE_MARK}{slice_name}"


def title_slice(title: str) -> str | None:
    """The Slice a :func:`slice_title` names; ``None`` for the whole
    recording's figure."""
    head, mark, name = str(title).rpartition(SLICE_TITLE_MARK)
    return name if mark and head and name else None


def run_plots(exp, *, metric: str = "Licks", binsize_min: float = 30.0,
              omit: Iterable[str] | None = None, log=print,
              keep_figures: bool = True) -> SelectionRun:
    """*Generate plots*: the Plot Set minus *omit* (default: the experiment's
    Selection), each saved into ``analysis/``.  *metric* and *binsize_min*
    are the Chosen metric figures' (the binned time course's bin).  With
    *keep_figures* the figures come back in :attr:`SelectionRun.figures`
    for the Hub's tabs, a Slice's titled with its name
    (:func:`slice_title`); pass ``False`` where nothing shows them."""
    run, todo = _plan(exp, PLOTS, omit)
    _announce(run, todo, log)
    ctx = _Context(exp, log, binsize_min=binsize_min, metric=metric,
                   keep_figures=keep_figures)
    _run_items(exp, run, todo, ctx)
    ## A run of every Slice hands back the same figures once per Slice: the
    ## title says which, so no Slice's figure takes the whole recording's tab
    ## and the Hub can show the Slice it shows.
    name = getattr(exp, "slice_name", None)
    run.figures = [(slice_title(title, name), fig) for title, fig in ctx.figures]
    log(f"{sel.RUN_NAMES[PLOTS]} complete — {len(run.ran)} item(s)")
    return run
