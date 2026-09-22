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
_PR_GATED = {"plot_breaking_point", "paired_yoked_diff",
             "plot_pr_cumulative_diff", "plot_pr_cumulative_licks"}


class ScriptContext:
    """Everything a run needs that is not the script itself.

    Defaults are the headless ones; the Hub overrides them with its spinbox
    values so a script run from the UI behaves exactly as before.
    """

    def __init__(self, *, binsize: float = 30.0, parallel: bool = True,
                 exclusion_group: str | None = None,
                 loader: Callable[..., Any] | None = None,
                 on_range: Callable[[float, float], None] | None = None,
                 log: Callable[[str], None] | None = None):
        self.binsize = float(binsize)
        self.parallel = bool(parallel)
        self.exclusion_group = exclusion_group
        self.loader = loader
        self.on_range = on_range
        self._log = log

    def log(self, message: str) -> None:
        if self._log is not None:
            self._log(message)
        else:
            print(message, flush=True)


def _save_figure(fig, path: Path, log) -> None:
    """Save a plotnine ggplot or a matplotlib Figure to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(fig, "save"):
        fig.save(str(path), dpi=300)
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
    """
    ctx = context or ScriptContext(log=log)
    if log is not None:
        ctx._log = log
    steps = script.get("steps") or []
    script_name = str(script.get("name") or "general")
    script_start = float(script.get("start", 0.0) or 0.0)
    script_end = float(script.get("end", 0.0) or 0.0)
    figures: list[tuple[str, Any]] = []

    def rm_for(step: dict) -> tuple[float, float]:
        return (float(step.get("start", script_start)),
                float(step.get("end", script_end)))

    from .. import analytics
    from ..metrics import METRIC_DEFAULT_MODE
    from ..hedonic_experiment import HedonicFeedingExperiment
    from ..progressive_ratio_experiment import ProgressiveRatioExperiment
    from ..two_well_experiment import TwoWellExperiment

    exp = experiment
    for index, step in enumerate(steps, start=1):
        action = str(step.get("action", "")).strip().lower()
        rm = rm_for(step)
        ctx.log(f"\n[Script step {index}/{len(steps)}] {action}")

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

        analysis_dir = exp.analysis_dir

        if action == "load":
            ## Already loaded by the caller. Kept as a no-op so existing
            ## scripts (which all start with `load`) run unchanged.
            if ctx.on_range is not None:
                ctx.on_range(rm[0], rm[1])
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
            ctx.log(f"QC reports → {exp.write_qc_reports()}")

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
            ctx.log(f"Wrote: {exp.weighted_duration_summary(save=True, range_minutes=rm)}")

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

        elif action == "tidy_export":
            kind = str(step.get("kind", "feeding")).strip().lower()
            df = analytics.tidy_events(exp, kind=kind)
            out = analysis_dir / f"tidy_{kind}_events.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            ctx.log(f"Wrote: {out}  ({len(df)} bouts)")

        elif action == "bootstrap":
            metric = str(step.get("metric", "PI"))
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
                factors=tuple(factors) if isinstance(factors, list) else None,
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
                values=[float(v) for v in (step.get("values") or [])],
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

        elif action == "pdf_report":
            from ..pdf_report import write_experiment_report

            metrics = step.get("metrics") or ("Licks", "Events", "MedDuration")
            path = write_experiment_report(
                exp, metrics=tuple(metrics),
                binsize_min=float(step.get("binsize", ctx.binsize)),
                range_minutes=rm)
            ctx.log(f"Wrote: {path}")

        else:
            ctx.log(f"[Skip] Unknown action: {action!r}")

    return figures
