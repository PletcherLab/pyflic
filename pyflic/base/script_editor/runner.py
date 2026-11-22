"""Headless executor for **Experiment Scripts**.

This is the one implementation of "run these steps against this experiment".
It used to live inside ``AnalysisHubWindow._build_script_task``, closed over UI
state, which meant the ``run_in_experiments`` bridge (ADR-0005) had no way to
reach it.  Extracted here it serves both: the Hub wraps it in a worker thread
and collects figures, a Project Script calls it once per Member.

Every step's action mirrors one entry in :mod:`pyflic.base.script_editor.actions`.
Adding an action means adding it in both places.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

#: Steps whose action is gated on the Chamber Layout rather than the
#: Experiment Type (ADR-0007).  ``requires_layout`` on the Action carries the
#: same information for the editor; this is the runtime half.
_LAYOUT_GATED = {"plot_well_comparison", "transition_matrix"}
_PR_GATED = {"plot_breaking_point", "paired_yoked_diff", "breaking_point",
             "plot_pr_cumulative_diff", "plot_pr_cumulative_licks",
             "plot_pr_still_responding",
             "pr_light_qc", "plot_pr_light_events", "plot_pr_resting_level"}
#: Steps that need an optogenetic experiment, of any type (optogenetics: yes,
#: or auto with a Program.txt or light data).
_OPTO_GATED = {"opto_light_qc", "plot_opto_light", "light_summary"}


def _as_list(value) -> list:
    """A ``list_str`` / ``list_float`` step parameter as a list.  The Script
    Editor stores a parameter nobody edited as its default text
    (``"Licks, Events, MedDuration"``), and a hand-written script may do the
    same: a string is split on commas, never iterated letter by letter."""
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [v for v in value if not (isinstance(v, str) and not v.strip())]
    return [value]


class ScriptContext:
    """Everything a run needs that is not the script itself.

    Defaults are the headless ones; the Hub overrides them with its spinbox
    values so a script run from the UI behaves exactly as before.
    """

    def __init__(self, *, binsize: float = 30.0, parallel: bool = True,
                 exclusion_group: str | None = None,
                 loader: Callable[..., Any] | None = None,
                 log: Callable[[str], None] | None = None):
        self.binsize = float(binsize)
        self.parallel = bool(parallel)
        self.exclusion_group = exclusion_group
        self.loader = loader
        self._log = log

    def log(self, message: str) -> None:
        if self._log is not None:
            self._log(message)
        else:
            print(message, flush=True)


def _slice_param(value, default: str | None = None) -> str | None:
    """A ``slice:`` parameter as a Slice name, or *default* when blank."""
    if value is None or not str(value).strip():
        return default
    return str(value).strip()


def _save_figure(fig, path: Path, log) -> None:
    """Save a plotnine ggplot or a matplotlib Figure to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(fig, "save"):
        fig.save(str(path), dpi=300, verbose=False)
    else:
        fig.savefig(str(path), dpi=300, bbox_inches="tight")
    log(f"Wrote: {path}")


def run_experiment_script(
    experiment,
    script: dict,
    *,
    context: ScriptContext | None = None,
    log: Callable[[str], None] | None = None,
) -> list[tuple[str, Any]]:
    """Run *script* against a loaded *experiment*.

    Returns ``[(title, figure), ...]`` for whatever the script drew, so the Hub
    can show them; a headless caller simply ignores the return value — the
    figures are on disk either way.

    A step gated on a Chamber Layout or Experiment Type the experiment does not
    have is **skipped with a log line**, never an error: a central Experiment
    Script broadcast across members of mixed shape should do what it can.

    A step works on the Slice its ``slice:`` names, or the script's
    (ADR-0020), loaded once the way *experiment* was; with neither, on
    *experiment* itself.  ``run_qc``, ``run_analysis`` and ``run_plots``
    without a Slice run the whole recording and every Slice the Selection
    keeps, as the Hub's buttons do.  The ``start``/``end`` a step once took
    are retired: a step that still states a window is skipped with a log line
    naming the Slice to declare, never run over a window nothing records.
    """
    ctx = context or ScriptContext(log=log)
    if log is not None:
        ctx._log = log
    steps = script.get("steps") or []
    script_name = str(script.get("name") or "general")
    figures: list[tuple[str, Any]] = []
    script_slice = _slice_param(script.get("slice"))

    def window_for(step: dict) -> tuple[float, float]:
        ## What a pre-Slice script asked for: a step's start/end, else the
        ## script's.  (0, 0) is the whole of whatever is loaded.
        def value(key: str) -> float:
            raw = step.get(key, script.get(key))
            try:
                return float(raw or 0.0)
            except (TypeError, ValueError):
                return 0.0
        return value("start"), value("end")

    slice_cache: dict[str | None, Any] = {}

    def experiment_for(name: str | None):
        """The experiment a step with Slice *name* works on, or ``None``
        when the recording does not cover the Slice."""
        from .. import slices as _slices

        if name is None or name == getattr(experiment, "slice_name", None):
            return experiment
        if name not in slice_cache:
            try:
                slice_cache[name] = experiment.load_slice(name)
            except _slices.SliceNotCovered as err:
                if experiment.experiment_dir is not None:
                    _slices.record_not_covered(experiment.experiment_dir, err)
                ctx.log(f"Slice {name}: not covered, left out ({err.reason})")
                slice_cache[name] = None
        return slice_cache[name]

    from .. import analytics
    from ..metrics import METRIC_DEFAULT_MODE
    from ..hedonic_experiment import HedonicFeedingExperiment
    from ..progressive_ratio_experiment import ProgressiveRatioExperiment
    from ..two_well_experiment import TwoWellExperiment

    for index, step in enumerate(steps, start=1):
        action = str(step.get("action", "")).strip().lower()
        rm = (0.0, 0.0)
        ctx.log(f"\n[Script step {index}/{len(steps)}] {action}")

        window = window_for(step)
        if window != (0.0, 0.0):
            if action == "load":
                ctx.log(f"[Note] start/end {window} on load are retired and change "
                        f"nothing (ADR-0020); declare a Slice and name it with slice:.")
            else:
                end = f"{window[1]:g}" if window[1] > 0 else "end"
                ctx.log(f"[Skip] {action}: start/end {window} are retired (ADR-0020). "
                        f"Declare a Slice with start: {window[0]:g}, end: {end} under "
                        f"global: slices: and name it with slice: on the step or the script.")
                continue

        step_slice = _slice_param(step.get("slice"), script_slice)
        exp = experiment_for(step_slice)
        if exp is None:
            ctx.log(f"[Skip] {action}: the recording does not cover Slice {step_slice}.")
            continue
        if step_slice is not None:
            ctx.log(f"(Slice {step_slice})")

        if action in _LAYOUT_GATED and not isinstance(exp, TwoWellExperiment):
            ctx.log(f"[Skip] {action} requires a two-well chamber layout.")
            continue
        if action in ("weighted_duration", "plot_hedonic") \
                and not isinstance(exp, HedonicFeedingExperiment):
            ctx.log(f"[Skip] {action} requires a Hedonic experiment.")
            continue
        if action in _PR_GATED and not isinstance(exp, ProgressiveRatioExperiment):
            ctx.log(f"[Skip] {action} requires a Progressive Ratio experiment.")
            continue
        if action in _OPTO_GATED and not getattr(exp, "is_optogenetic", False):
            ctx.log(f"[Skip] {action} needs an optogenetic experiment (optogenetics: "
                    f"yes, or a data/Program.txt or light data).")
            continue

        analysis_dir = exp.analysis_dir

        if action == "load":
            ## Already loaded by the caller. Kept as a no-op so existing
            ## scripts (which all start with `load`) run unchanged.
            ctx.log("Experiment already loaded.")

        elif action == "remove_chambers":
            from ..exclusions import read_exclusions

            group = str(step.get("group", script_name))
            groups = read_exclusions(exp.experiment_dir)
            group_excl = groups.get(group, {})
            if not group_excl:
                ctx.log(f"  No chambers listed for group '{group}' in "
                        f"remove_chambers.csv.")
            else:
                remove_set = {(d, c) for d, chs in group_excl.items() for c in chs}
                exp._remove_chambers_from_design(remove_set)
                for dfm_id, chambers in sorted(group_excl.items()):
                    ctx.log(f"  DFM {dfm_id}: removed chamber(s) "
                            f"{sorted(chambers)}")
                ctx.log(f"  Total: {len(remove_set)} chamber(s) removed "
                        f"(group '{group}').")
                exp.excluded_chambers = {k: sorted(v)
                                         for k, v in group_excl.items()}
                exp.exclusion_group = group

        elif action == "write_summary":
            path = exp.write_summary()
            ctx.log(f"Wrote: {path}")

        elif action == "basic_analysis":
            exp.execute_basic_analysis(range_minutes=rm, skip_qc=True)

        elif action == "run_qc":
            ## The QC Set minus the Selection (ADR-0018): with nothing left
            ## out, the files write_qc_reports writes.  Without a Slice, the
            ## whole recording and every Slice the Selection keeps (ADR-0020).
            from ..run_selection import SELECTED, each_slice, run_qc

            for target in each_slice(exp, only=SELECTED if step_slice is None else step_slice,
                                     log=ctx.log):
                run_qc(target, log=ctx.log)
                ctx.log(f"QC reports → {target.qc_report_dir}")

        elif action in ("run_analysis", "run_plots"):
            ## The Hub's Run analysis and Generate plots: the set minus the
            ## Selection, over the same Slices as run_qc.
            from ..run_selection import SELECTED, each_slice, run_analysis, run_plots

            raw = step.get("binsize")
            bs = ctx.binsize if raw is None or str(raw).strip() == "" else float(raw)
            for target in each_slice(exp, only=SELECTED if step_slice is None else step_slice,
                                     log=ctx.log):
                if action == "run_analysis":
                    run_analysis(target, binsize_min=bs, log=ctx.log)
                else:
                    ## A Slice's figures come back titled with its name, so
                    ## the Hub shows the Slice it shows from the same step.
                    run = run_plots(target, metric=str(step.get("metric") or "Licks"),
                                    binsize_min=bs, log=ctx.log)
                    figures.extend(run.figures)

        elif action == "feeding_csv":
            ctx.log(f"Wrote: {exp.write_feeding_summary(range_minutes=rm)}")

        elif action == "facet_csv":
            path = exp.write_feeding_summary_facet()
            ctx.log("Skipped — experiment is not faceted." if path is None
                    else f"Wrote: {path}")

        elif action == "binned_csv":
            bs = float(step.get("binsize", ctx.binsize))
            df = exp.binned_feeding_summary(binsize_min=bs, range_minutes=rm,
                                            save=True)
            ctx.log(f"Binned rows: {len(df)}")

        elif action == "weighted_duration":
            ## It returns the table; the file it writes is this one.
            exp.weighted_duration_summary(save=True, range_minutes=rm)
            ctx.log(f"Wrote: {analysis_dir / 'treatment_means.csv'}")

        elif action == "plot_feeding_summary":
            fig = exp.plot_feeding_summary(range_minutes=rm)
            _save_figure(fig, analysis_dir / "feeding_summary.png", ctx.log)
            figures.append(("Feeding Summary", fig))

        elif action in ("plot_binned", "plot_dot"):
            metric = str(step.get("metric", "Licks"))
            mode = str(step.get("mode",
                                METRIC_DEFAULT_MODE.get(metric, "total")))
            safe = metric.replace("/", "_")
            if action == "plot_binned":
                bs = float(step.get("binsize", ctx.binsize))
                fig = exp.plot_binned_metric_by_treatment(
                    metric=metric, two_well_mode=mode, binsize_min=bs,
                    range_minutes=rm)
                _save_figure(fig, analysis_dir / f"binned_{safe}.png", ctx.log)
                figures.append((f"Binned: {metric}", fig))
            else:
                fig = exp.plot_dot_metric_by_treatment(
                    metric=metric, two_well_mode=mode, range_minutes=rm)
                _save_figure(fig, analysis_dir / f"dot_{safe}.png", ctx.log)
                figures.append((f"Dot: {metric}", fig))

        elif action == "plot_moving_window":
            metric = str(step.get("metric", "MedDuration"))
            mode = str(step.get("mode",
                                METRIC_DEFAULT_MODE.get(metric, "mean_ab")))
            fig = exp.plot_moving_window_metric_by_treatment(
                metric=metric, two_well_mode=mode,
                window_min=float(step.get("window", 60.0)),
                step_min=float(step.get("step", 30.0)), range_minutes=rm)
            safe = metric.replace("/", "_")
            _save_figure(fig, analysis_dir / f"moving_window_{safe}.png", ctx.log)
            figures.append((f"Moving window: {metric}", fig))

        elif action in ("plot_moving_median_chambers",
                        "plot_moving_median_treatment"):
            window = float(step.get("window", 60.0))
            mstep = float(step.get("step", 30.0))
            mode = str(step.get("mode", "mean_ab"))
            if action == "plot_moving_median_chambers":
                fig = exp.plot_moving_median_duration_by_chamber(
                    window_min=window, step_min=mstep, two_well_mode=mode,
                    range_minutes=rm)
                name, title = ("moving_median_duration_chambers.png",
                               "Moving MedDuration: chambers")
            else:
                fig = exp.plot_moving_median_duration_by_treatment(
                    window_min=window, step_min=mstep, two_well_mode=mode,
                    range_minutes=rm)
                name, title = ("moving_median_duration_treatments.png",
                               "Moving MedDuration: treatments")
            _save_figure(fig, analysis_dir / name, ctx.log)
            figures.append((title, fig))

        elif action == "plot_well_comparison":
            metric = str(step.get("metric", "MedDuration"))
            fig = exp.facet_plot_well_durations(metric=metric, range_minutes=rm)
            _save_figure(fig, analysis_dir / f"well_comparison_{metric}.png",
                         ctx.log)
            figures.append((f"Well A vs B: {metric}", fig))

        elif action == "plot_hedonic":
            fig = exp.hedonic_feeding_plot(save=True, range_minutes=rm)
            ctx.log("Wrote hedonic feeding plot.")
            figures.append(("Hedonic Feeding Plot", fig))

        elif action == "plot_breaking_point":
            analysis_dir.mkdir(parents=True, exist_ok=True)
            for dfm_id in sorted(exp.dfms):
                fig = exp.plot_breaking_point_dfm(dfm_id)
                _save_figure(fig, analysis_dir / f"breaking_point_dfm{dfm_id}.png",
                             ctx.log)
                figures.append((f"Breaking Point — DFM {dfm_id}", fig))

        elif action == "paired_yoked_diff":
            ctx.log(f"Wrote: {exp.write_paired_yoked_diff()}")

        elif action == "breaking_point":
            ctx.log(f"Wrote: {exp.write_breaking_point()}")
            if exp.simulated_yoked_enabled():
                ctx.log(f"Wrote: {exp.write_simulated_yoked_lights()}")
            ctx.log("\n".join(exp.breaking_point_lines()))

        elif action == "plot_pr_still_responding":
            fig = exp.plot_still_responding()
            _save_figure(fig, analysis_dir / "pr_still_responding.png", ctx.log)
            figures.append(("Still responding", fig))

        elif action == "plot_pr_cumulative_diff":
            bs = float(step.get("binsize", 1.0))
            ctx.log(f"Wrote: {exp.write_cumulative_diff(binsize_min=bs)}")
            fig = exp.plot_cumulative_diff(binsize_min=bs)
            _save_figure(fig, analysis_dir / "pr_cumulative_diff.png", ctx.log)
            figures.append(("Paired − yoked cumulative licks", fig))

        elif action == "plot_pr_cumulative_licks":
            bs = float(step.get("binsize", 1.0))
            for dfm_id in sorted(exp.dfms):
                fig = exp.plot_cumulative_licks_dfm(dfm_id, binsize_min=bs)
                _save_figure(fig, analysis_dir / f"pr_cumulative_licks_dfm{dfm_id}.png",
                             ctx.log)
                figures.append((f"Training-aligned traces — DFM {dfm_id}", fig))

        elif action == "pr_light_qc":
            for path in exp.write_light_qc().values():
                ctx.log(f"Wrote: {path}")
            lines = exp.light_qc_lines()
            ctx.log("\n".join(lines) if lines else "Light QC: every chamber group is ok.")

        elif action == "opto_light_qc":
            for path in exp.opto.write().values():
                ctx.log(f"Wrote: {path}")
            lines = exp.opto.lines()
            ctx.log("\n".join(lines) if lines
                    else "Opto light QC: every linkage group is ok.")

        elif action == "light_summary":
            from ..light_summary import write as write_light_tables

            for path in write_light_tables(exp, range_minutes=rm).values():
                ctx.log(f"Wrote: {path}")

        elif action == "plot_opto_light":
            bs = float(step.get("binsize", 10.0))
            out_dir = Path(exp.qc_dir) / "opto"
            for dfm_id in exp.opto.dfm_ids():
                fig = exp.opto.plot_dfm(dfm_id, binsize_min=bs)
                _save_figure(fig, out_dir / f"opto_light_dfm{dfm_id}.png", ctx.log)
                figures.append((f"Light explained by licks — DFM {dfm_id}", fig))

        elif action in ("plot_pr_light_events", "plot_pr_resting_level"):
            light_events = action == "plot_pr_light_events"
            stem = "pr_light_events" if light_events else "pr_resting_level"
            label = ("Licks per light event" if light_events
                     else "Sucrose Well resting level")
            for dfm_id in sorted(exp.dfms):
                fig = (exp.plot_light_events_dfm(dfm_id) if light_events
                       else exp.plot_resting_level_dfm(dfm_id))
                _save_figure(fig, analysis_dir / f"{stem}_dfm{dfm_id}.png", ctx.log)
                figures.append((f"{label} — DFM {dfm_id}", fig))

        elif action == "tidy_export":
            kind = str(step.get("kind", "feeding")).strip().lower()
            df = analytics.tidy_events(exp, kind=kind)
            out = analysis_dir / f"tidy_{kind}_events.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            ctx.log(f"Wrote: {out}  ({len(df)} bouts)")

        elif action == "bootstrap":
            ## The catalogue's default (param_default): PI on a two-well
            ## layout; a single-well one has no PI.
            metric = str(step.get("metric")
                         or ("PI" if isinstance(exp, TwoWellExperiment) else "Licks"))
            seed = step.get("seed", 0)
            res = analytics.bootstrap_metric(
                exp, metric=metric, two_well_mode=str(step.get("mode", "total")),
                n_boot=int(step.get("n_boot", 2000)),
                ci=float(step.get("ci", 0.95)), range_minutes=rm,
                seed=int(seed) if seed is not None else None)
            out = analysis_dir / f"bootstrap_{metric}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            res.summary.to_csv(out, index=False)
            ctx.log(f"Wrote: {out}\n{res.summary.to_string(index=False)}")

        elif action == "compare":
            metric = str(step.get("metric", "MedDuration"))
            model = str(step.get("model", "aov"))
            factors = step.get("factors")
            res = analytics.compare_treatments(
                exp, metric=metric, two_well_mode=str(step.get("mode", "A")),
                model=model,
                factors=tuple(_as_list(factors)) or None,
                range_minutes=rm)
            out = analysis_dir / f"compare_{metric}_{model}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            res.table.to_csv(out, index=False)
            if res.posthoc is not None:
                res.posthoc.to_csv(out.with_name(out.stem + "_posthoc.csv"),
                                   index=False)
            ctx.log(f"Wrote: {out}\n{res.table.to_string(index=False)}")

        elif action == "light_phase_summary":
            df = analytics.light_phase_summary(exp)
            out = analysis_dir / "light_phase_summary.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            ctx.log(f"Wrote: {out}  ({len(df)} rows)")

        elif action == "param_sensitivity":
            param = str(step.get("parameter", "feeding_event_link_gap"))
            res = analytics.parameter_sensitivity(
                exp, parameter=param,
                values=[float(v) for v in _as_list(step.get("values"))],
                range_minutes=rm)
            out = analysis_dir / f"param_sensitivity_{param}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            res.grid.to_csv(out, index=False)
            ctx.log(f"Wrote: {out}\n{res.grid.to_string(index=False)}")

        elif action == "transition_matrix":
            df = analytics.bout_transition_matrix(exp)
            out = analysis_dir / "bout_transition_matrix.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            ctx.log(f"Wrote: {out}  ({len(df)} rows)")

        elif action == "plot_program_timeline":
            from ..program_timeline import plot_dfm, timeline_dfms, timeline_path

            ## Gated like the opto light QC, here rather than in _OPTO_GATED
            ## so the action stays one block.
            if not getattr(exp, "is_optogenetic", False):
                ctx.log(f"[Skip] {action} needs an optogenetic experiment (optogenetics: "
                        f"yes, or a data/Program.txt or light data).")
                continue
            ids = timeline_dfms(exp)
            if not ids:
                ctx.log("No DFM has a Program.txt section or firmware 4.0.0 Paradigm "
                        "columns: no Program timeline to draw.")
            bs = float(step.get("binsize", 10.0))
            for dfm_id in ids:
                fig = plot_dfm(exp, dfm_id, binsize_min=bs)
                _save_figure(fig, timeline_path(exp.qc_dir, dfm_id), ctx.log)
                figures.append((f"Program timeline — DFM {dfm_id}", fig))

        elif action == "pdf_report":
            from ..pdf_report import write_experiment_report

            metrics = _as_list(step.get("metrics")) or ("Licks", "Events", "MedDuration")
            path = write_experiment_report(
                exp, metrics=tuple(metrics),
                binsize_min=float(step.get("binsize", ctx.binsize)),
                range_minutes=rm)
            ctx.log(f"Wrote: {path}")

        else:
            ctx.log(f"[Skip] Unknown action: {action!r}")

    return figures
