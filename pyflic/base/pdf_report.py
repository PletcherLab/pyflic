"""The experiment report: one member, one PDF — ``analysis/experiment_report.pdf``.

Laid out by :mod:`pyflic.base.report_layout` on fixed US Letter pages:

* **Cover** — what was recorded and how it was analysed, the quality-control
  verdicts at a glance, and the contents.
* **1 Quality control** — per-DFM data integrity, two-well cross-talk
  (simultaneous feeding, bleeding), the chambers excluded and why, then the
  Experiment Type's own checks (:meth:`Experiment.report_qc_blocks`;
  Progressive Ratio: training and the light QC).
* **2 Results** — the inference figures with their statistics, chosen by the
  Chamber Layout (two-well: preference, then consumption; single-well:
  consumption) and extended or replaced by the Experiment Type
  (:meth:`Experiment.report_results_blocks`; Progressive Ratio: the
  paired − yoked difference and the breaking point).
* **Appendix A** — every feeding metric by treatment; **Appendix B** — the
  analysis settings.

The report describes the analysis the pipeline runs, so it applies the
design's auto-removal first — once per loaded experiment, exactly as basic
analysis does — and every figure and table after the Quality control section
stands on the chambers that survive it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import report_content as rc
from . import report_layout as rl
from .experiment import Experiment


@dataclass
class ReportOptions:
    """What the caller asked of the report (the Script Editor's `pdf_report`
    step and Analyze all pass these)."""
    metrics: Sequence[str] = ("Licks", "Events", "MedDuration")
    binsize_min: float = 30.0
    range_minutes: Sequence[float] = (0, 0)
    transform_licks: bool = True
    include_comparison: bool = True


def write_experiment_report(
    experiment: Experiment,
    path: str | Path | None = None,
    *,
    metrics: Sequence[str] = ("Licks", "Events", "MedDuration"),
    binsize_min: float = 30.0,
    range_minutes: Sequence[float] = (0, 0),
    transform_licks: bool | None = None,
    include_comparison: bool = True,
) -> Path:
    """Write the experiment report for *experiment* and return its path —
    ``analysis/experiment_report.pdf`` unless *path* says otherwise.

    *metrics* are the time courses the Results section draws beside the
    type's own figures; *binsize_min* is their bin.  *include_comparison*
    adds the treatment statistics under each figure.
    """
    if path is None:
        out_dir = experiment.analysis_dir
        if out_dir is None:
            raise ValueError("path must be given when experiment has no experiment_dir")
        path = out_dir / "experiment_report.pdf"
    doc = build_experiment_report(
        experiment, metrics=metrics, binsize_min=binsize_min,
        range_minutes=range_minutes, transform_licks=transform_licks,
        include_comparison=include_comparison)
    return doc.save(path)


def build_experiment_report(
    experiment: Experiment,
    *,
    metrics: Sequence[str] = ("Licks", "Events", "MedDuration"),
    binsize_min: float = 30.0,
    range_minutes: Sequence[float] = (0, 0),
    transform_licks: bool | None = None,
    include_comparison: bool = True,
) -> rl.ReportDocument:
    """The report as a :class:`~pyflic.base.report_layout.ReportDocument`,
    not yet written — ``save`` it, or ``save_pngs`` to look at it."""
    options = ReportOptions(
        metrics=tuple(metrics), binsize_min=float(binsize_min),
        range_minutes=tuple(range_minutes),
        transform_licks=(experiment.transform_licks if transform_licks is None
                         else bool(transform_licks)),
        include_comparison=bool(include_comparison))
    experiment.apply_auto_removal(range_minutes=options.range_minutes)
    qc = experiment.qc_results
    if qc is None:
        ## The integrity check narrates itself to stdout; the report has
        ## its own table for it.
        from contextlib import redirect_stdout
        from io import StringIO

        with redirect_stdout(StringIO()):
            qc = experiment.compute_qc_results()
    name = _member_name(experiment)

    doc = rl.ReportDocument("Experiment report", name)
    doc.add(_cover(experiment, qc, name))
    doc.add(rl.Heading("Quality control"), _qc_blocks(experiment, qc))
    doc.add(rl.PageBreak(), rl.Heading("Results"))
    doc.add(experiment.report_results_blocks(_generic_results(experiment, options), options))
    doc.add(rl.PageBreak(),
            rl.Heading("Appendix A — All feeding metrics", numbered=False),
            _all_metrics(experiment, options))
    doc.add(rl.Heading("Appendix B — Analysis settings", numbered=False),
            rc.settings_blocks(experiment.global_config, experiment.global_constants))
    return doc


# ---------------------------------------------------------------------------
# Cover
# ---------------------------------------------------------------------------

def _member_name(experiment: Experiment) -> str:
    if experiment.experiment_dir is not None:
        return Path(experiment.experiment_dir).name
    return "experiment"


def _is_two_well(experiment: Experiment) -> bool:
    return (experiment.chamber_layout or "two_well") == "two_well"


def _treatment_label(experiment: Experiment, dfm_id: int, chamber: int) -> str:
    """A chamber's treatment for a QC table — still named once auto-removal
    has taken the chamber out of the design, and marked so."""
    name = experiment.design.treatment_for(dfm_id, chamber)
    if name is not None:
        return name
    filtered = experiment.filtered_chambers
    if filtered is not None and not filtered.empty:
        hit = filtered[(filtered["DFM"].astype(int) == int(dfm_id))
                       & (filtered["Chamber"].astype(int) == int(chamber))]
        if not hit.empty:
            return f"{hit['Treatment'].iloc[0]} (excluded)"
    return "–"


def _recording_line(experiment: Experiment, qc: dict) -> str:
    starts, ends, spans = [], [], []
    for dfm_id in sorted(experiment.dfms):
        integrity = (qc.get(dfm_id) or {}).get("integrity") or {}
        start, end = integrity.get("start_time"), integrity.get("end_time")
        if start is not None and pd.notna(start):
            starts.append(start)
        if end is not None and pd.notna(end):
            ends.append(end)
        span = integrity.get("elapsed_minutes_from_minutes_col")
        if span is not None:
            spans.append(float(span))
    hours = (max(spans) / 60.0) if spans else None
    if starts and ends:
        fmt = "%Y-%m-%d %H:%M"
        line = f"Recorded {min(starts).strftime(fmt)} → {max(ends).strftime(fmt)}"
        return line + (f" · {hours:.1f} h" if hours is not None else "")
    return f"Recording length {hours:.1f} h" if hours is not None else ""


def _treatment_counts(experiment: Experiment) -> str:
    removed: dict[str, int] = {}
    filtered = experiment.filtered_chambers
    if filtered is not None and not filtered.empty and "Treatment" in filtered.columns:
        for t in filtered["Treatment"].astype(str):
            removed[t] = removed.get(t, 0) + 1
    parts = []
    for name, treatment in experiment.design.treatments.items():
        n = len(treatment.chambers)
        extra = f", {removed[name]} excluded" if removed.get(name) else ""
        parts.append(f"{name}: {n} chamber{'s' if n != 1 else ''}{extra}")
    for name, count in removed.items():
        if name not in experiment.design.treatments:
            parts.append(f"{name}: 0 chambers, {count} excluded")
    return " · ".join(parts) if parts else "none assigned"


def _facet_description(experiment: Experiment) -> str:
    labels = list(experiment.facet_labels()) if hasattr(experiment, "facet_labels") else []
    etype = getattr(experiment, "experiment_type", None)
    if getattr(etype, "data_derived_facets", False):
        return (f"{' / '.join(labels)} — split at each chamber group's own training end")
    windows = experiment.facet_windows()
    if not windows:
        return "none (whole recording)"
    from . import windowing

    return " · ".join(f"{label} {windowing.format_range(w)}"
                      for label, w in zip(labels, windows))


def _cover(experiment: Experiment, qc: dict, name: str) -> list[Any]:
    etype = getattr(experiment, "experiment_type", None)
    type_name = getattr(etype, "display_name", "Custom")
    layout = (experiment.chamber_layout or "two_well").replace("_", "-")
    n_dfm = len(experiment.dfms)
    lines = [f"{type_name} · {layout} · {n_dfm} DFM{'s' if n_dfm != 1 else ''}",
             _recording_line(experiment, qc)]
    rows: list[tuple[str, str]] = []
    if experiment.experiment_dir is not None:
        rows.append(("Directory", str(experiment.experiment_dir)))
    rows.append(("DFMs", ", ".join(str(d) for d in sorted(experiment.dfms))))
    rows.append(("Treatments", _treatment_counts(experiment)))
    if experiment.design_factors:
        levels = (experiment.global_config or {}).get("experimental_design_factors") or {}
        rows.append(("Design factors", " · ".join(
            f"{f} ({', '.join(str(v) for v in levels.get(f, []))})" if levels.get(f) else f
            for f in experiment.design_factors)))
    if _is_two_well(experiment) and experiment.well_names:
        rows.append(("Wells", f"A = {experiment.well_names.get('A', 'A')}, "
                              f"B = {experiment.well_names.get('B', 'B')}"))
    rows.append(("Facets", _facet_description(experiment)))
    rows.append(("Lick counts", "fourth-root transformed (Licks^0.25) in tables and "
                                "statistics" if experiment.transform_licks else "raw counts"))

    blocks: list[Any] = [rl.Cover("Experiment report", name, [l for l in lines if l]),
                         rl.KeyValues(rows)]
    intro = etype.report_intro() if etype is not None and hasattr(etype, "report_intro") else ""
    if intro and not getattr(etype, "is_custom", False):
        blocks.append(rl.Paragraph(intro, size=rl.SIZE_SMALL + 0.5, color=rl.MUTED))
    blocks.append(rl.Heading("At a glance", level=2, numbered=False))
    blocks.append(_integrity_callout(experiment, qc))
    blocks.append(_exclusion_callout(experiment))
    blocks.extend(experiment.report_glance_blocks())
    blocks.append(rl.Contents())
    return blocks


# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------

def _error_flags(experiment: Experiment, dfm_id: int) -> int:
    raw = experiment.dfms[dfm_id].raw_df
    total = 0
    for col in raw.columns:
        if "error" in str(col).lower():
            values = pd.to_numeric(raw[col], errors="coerce")
            total += int((values.notna() & (values != 0)).sum())
    return total


def _integrity_frame(experiment: Experiment, qc: dict) -> pd.DataFrame:
    rows = []
    for dfm_id in sorted(experiment.dfms):
        r = qc.get(dfm_id) or {}
        integrity = r.get("integrity") or {}
        start = integrity.get("start_time")
        span = integrity.get("elapsed_minutes_from_minutes_col")
        index_ok = integrity.get("index_increments_by_one")
        breaks = int(r.get("data_breaks_count", 0) or 0)
        errors = _error_flags(experiment, dfm_id)
        verdict = "ok" if (breaks == 0 and errors == 0 and index_ok is not False) else "warning"
        rows.append({
            "DFM": dfm_id,
            "Samples": int(integrity.get("n_rawdata") or len(experiment.dfms[dfm_id].raw_df)),
            "Start": start.strftime("%Y-%m-%d %H:%M") if start is not None and pd.notna(start) else "–",
            "Hours": (float(span) / 60.0) if span is not None else np.nan,
            "Index continuous": ("–" if index_ok is None else ("yes" if index_ok else "NO")),
            "Data breaks": breaks,
            "Error flags": errors,
            "Verdict": verdict,
        })
    return pd.DataFrame(rows)


def _integrity_callout(experiment: Experiment, qc: dict) -> rl.Callout:
    frame = _integrity_frame(experiment, qc)
    bad = frame[frame["Verdict"] != "ok"]
    if bad.empty:
        return rl.Callout("No data breaks, no firmware error flags, and a continuous "
                          "sample index on every DFM.", tone="ok", title="Data integrity")
    detail = "; ".join(
        f"DFM {int(r.DFM)}: {int(r['Data breaks'])} data break(s), "
        f"{int(r['Error flags'])} error flag(s)" for _, r in bad.iterrows())
    return rl.Callout(detail + ".  See Quality control.", tone="warning",
                      title="Data integrity")


def _exclusion_frame(experiment: Experiment) -> pd.DataFrame:
    rows = []
    group = experiment.exclusion_group or "file"
    for dfm_id, chambers in sorted((experiment.excluded_chambers or {}).items()):
        for chamber in chambers:
            rows.append({"DFM": int(dfm_id), "Chamber": int(chamber), "Treatment": "",
                         "Source": "by hand",
                         "Reason": f"remove_chambers.csv, group '{group}'"})
    filtered = experiment.filtered_chambers
    if filtered is not None and not filtered.empty:
        for _, r in filtered.iterrows():
            rows.append({"DFM": int(r["DFM"]), "Chamber": int(r["Chamber"]),
                         "Treatment": str(r.get("Treatment", "")), "Source": "automatic",
                         "Reason": str(r.get("Reason", ""))})
    return pd.DataFrame(rows, columns=["DFM", "Chamber", "Treatment", "Source", "Reason"])


def _exclusion_callout(experiment: Experiment) -> rl.Callout:
    frame = _exclusion_frame(experiment)
    if frame.empty:
        return rl.Callout("No chamber was excluded, by hand or by the design's cutoffs.",
                          tone="ok", title="Exclusions")
    manual = int((frame["Source"] == "by hand").sum())
    auto = int((frame["Source"] == "automatic").sum())
    parts = []
    if manual:
        parts.append(f"{manual} by hand (remove_chambers.csv)")
    if auto:
        parts.append(f"{auto} automatically by the design's cutoffs")
    return rl.Callout(f"{len(frame)} chamber(s) excluded: {' and '.join(parts)}.  The "
                      f"results stand on the chambers that remain.",
                      tone="info", title="Exclusions")


def _crosstalk_blocks(experiment: Experiment, qc: dict) -> list[Any]:
    rows = []
    for dfm_id in sorted(experiment.dfms):
        matrix = (qc.get(dfm_id) or {}).get("simultaneous_feeding_matrix")
        if not isinstance(matrix, pd.DataFrame):
            continue
        for index, (_label, r) in enumerate(matrix.iterrows(), start=1):
            a, b, both = float(r["Licks1"]), float(r["Licks2"]), float(r["Both"])
            union = a + b - both
            rows.append({
                "DFM": dfm_id, "Chamber": index,
                "Treatment": _treatment_label(experiment, dfm_id, index),
                "Lick samples, left well": a, "Lick samples, right well": b,
                "Both at once": both,
                "Both, % of lick samples": (100.0 * both / union) if union > 0 else 0.0,
            })
    blocks: list[Any] = []
    if rows:
        corrected = bool(getattr(next(iter(experiment.dfms.values())).params,
                                 "correct_for_dual_feeding", False))
        blocks += [
            rl.Heading("Simultaneous feeding", level=2),
            rl.Paragraph(
                "A fly cannot lick both wells of a chamber at once, so samples that "
                "register a lick on both are cross-talk between the wells or a fly "
                "bridging them.  "
                + ("The dual-feeding correction was applied (correct_for_dual_feeding), "
                   "which gives each such sample to the stronger well."
                   if corrected else
                   "The dual-feeding correction was not applied "
                   "(correct_for_dual_feeding is off).")),
            rl.Table(pd.DataFrame(rows), caption="Lick samples per well and on both wells "
                                                 "at once, per chamber",
                     formats={"Both, % of lick samples": "{:.2f}"}),
        ]
    bleed_rows = []
    threshold = None
    for dfm_id in sorted(experiment.dfms):
        r = qc.get(dfm_id) or {}
        bleeding = r.get("bleeding")
        if not isinstance(bleeding, dict) or "Matrix" not in bleeding:
            continue
        matrix = bleeding["Matrix"].to_numpy(dtype=float).copy()
        np.fill_diagonal(matrix, np.nan)
        if np.all(np.isnan(matrix)):
            continue
        i, j = np.unravel_index(np.nanargmax(matrix), matrix.shape)
        threshold = float(experiment.dfms[dfm_id].params.feeding_threshold)
        value = float(matrix[i, j])
        bleed_rows.append({
            "DFM": dfm_id, "Cutoff": float(r.get("bleeding_cutoff", 50.0)),
            "Largest response elsewhere": value,
            "Signalling well": f"W{i + 1}", "Responding well": f"W{j + 1}",
            "Verdict": "warning" if value > threshold else "ok",
        })
    if bleed_rows:
        blocks += [
            rl.Heading("Bleeding between wells", level=2),
            rl.Paragraph(
                "While one well's signal is above the cutoff, the mean signal of every "
                "other well should stay near zero.  A response above the feeding "
                f"threshold ({threshold:g}) would register as licks that never happened."),
            rl.Table(pd.DataFrame(bleed_rows), caption="Largest mean response in another "
                                                       "well, per DFM",
                     status={"Verdict": rl.tone_of}),
        ]
    return blocks


def _qc_blocks(experiment: Experiment, qc: dict) -> list[Any]:
    blocks: list[Any] = [
        rl.Heading("Data integrity", level=2),
        rl.Paragraph("Per DFM: samples recorded, gaps in the sample clock longer than "
                     f"{qc.get(next(iter(qc), 0), {}).get('data_breaks_multiplier', 4.0):g}× "
                     "the sampling interval (data breaks), non-zero firmware error codes, "
                     "and whether the sample index runs without a gap."),
        rl.Table(_integrity_frame(experiment, qc), caption="Data integrity by DFM",
                 formats={"Hours": "{:.1f}"}, status={"Verdict": rl.tone_of}),
    ]
    if _is_two_well(experiment):
        blocks += _crosstalk_blocks(experiment, qc)
    exclusions = _exclusion_frame(experiment)
    blocks.append(rl.Heading("Excluded chambers", level=2))
    if exclusions.empty:
        blocks.append(rl.Callout("No chamber was excluded.", tone="ok"))
    else:
        blocks.append(rl.Paragraph(
            "By hand: the member's remove_chambers.csv for the active exclusion group.  "
            "Automatic: the design's constants: cutoffs, applied once before any result "
            "was computed."))
        blocks.append(rl.Table(exclusions, caption="Chambers left out of the results"))
    blocks.extend(experiment.report_qc_blocks())
    return blocks


# ---------------------------------------------------------------------------
# Results, by Chamber Layout
# ---------------------------------------------------------------------------

def _summary_frames(experiment: Experiment, options: ReportOptions
                    ) -> tuple[pd.DataFrame, list[tuple[str, pd.DataFrame]], str | None]:
    """``(rows for the dot plots, [(phase, rows)] for the statistics, the
    panel column)`` — per Facet when the experiment has them."""
    facet = pd.DataFrame()
    if experiment.facet_windows():
        facet = experiment.feeding_summary_facet(transform_licks=options.transform_licks)
    if facet is not None and not facet.empty and "Facet" in facet.columns:
        labels = list(dict.fromkeys(facet["Facet"].astype(str)))
        frames = [(label, facet[facet["Facet"].astype(str) == label]) for label in labels]
        return facet, frames, "Facet"
    summary = experiment.feeding_summary(range_minutes=options.range_minutes,
                                         transform_licks=options.transform_licks)
    return summary, [("Whole recording", summary)], None


def _value(frame: pd.DataFrame, metric: str) -> pd.Series:
    from .analytics import _resolve_metric_col

    return pd.to_numeric(_resolve_metric_col(frame, metric), errors="coerce")


def _dot(experiment: Experiment, frame: pd.DataFrame, metric: str, label: str,
         facet_col: str | None, *, hline: float | None = None):
    def build():
        df = frame.copy()
        df["_Value"] = _value(df, metric)
        return rc.dot_plot(df, "_Value", y_label=label, facet_col=facet_col,
                           hline_at=hline, factors=experiment.design_factors)
    return build


def _timecourse(experiment: Experiment, metric: str, mode: str, options: ReportOptions):
    def build():
        return experiment.plot_binned_metric_by_treatment(
            metric=metric, two_well_mode=mode, binsize_min=options.binsize_min,
            range_minutes=options.range_minutes, transform_licks=options.transform_licks)
    return build


def _generic_results(experiment: Experiment, options: ReportOptions) -> list[Any]:
    """The layout's inference figures: two-well preference and consumption,
    single-well consumption.  A type's :meth:`report_results_blocks` may keep,
    extend or replace them."""
    from .analytics import treatment_comparisons

    two_well = _is_two_well(experiment)
    names = experiment.well_names or {}
    frame, frames, facet_col = _summary_frames(experiment, options)
    if frame is None or frame.empty:
        return [rl.Callout("No chamber has a treatment, so there is nothing to compare.",
                           tone="warning")]
    tx = " (transformed)" if options.transform_licks else ""
    per = "per Facet" if facet_col else "over the whole recording"
    blocks: list[Any] = []

    def label(metric: str) -> str:
        return rc.metric_label(metric, names, two_well=two_well)

    if two_well:
        blocks += [
            rl.Heading("Preference", level=2),
            rl.Paragraph(
                f"The preference index is (A − B) / (A + B) of each chamber's licks — +1 "
                f"only {names.get('A', 'well A')}, −1 only {names.get('B', 'well B')}; "
                f"the event PI is the same on feeding events.  One point per chamber, "
                f"{per}; the diamond and bars are the treatment mean ± SEM."),
            rl.PlotRow([
                rl.Plot(_dot(experiment, frame, "PI", "PI", facet_col, hline=0.0),
                        title="Preference index (PI)"),
                rl.Plot(_dot(experiment, frame, "EventPI", "Event PI", facet_col,
                             hline=0.0), title="Event PI"),
            ], height=3.0),
        ]
        if options.include_comparison:
            blocks += rc.stats_table(treatment_comparisons(frames, ["PI", "EventPI"]),
                                     caption="Treatment comparisons: preference",
                                     well_names=names, show_phase=bool(facet_col))
        blocks.append(rl.Plot(_timecourse(experiment, "PI", "total", options), height=2.8,
                              title="Preference over time",
                              caption=f"Mean ± SEM PI per treatment in "
                                      f"{options.binsize_min:g}-minute bins."))
    blocks += [
        rl.Heading("Consumption", level=2),
        rl.Paragraph(f"Licks and feeding events per chamber{tx}, {per}; median bout "
                     f"duration in seconds.  The diamond and bars are the treatment "
                     f"mean ± SEM."),
        rl.PlotRow([
            rl.Plot(_dot(experiment, frame, "Licks", label("Licks") + tx, facet_col),
                    title=label("Licks") + tx),
            rl.Plot(_dot(experiment, frame, "Events", label("Events"), facet_col),
                    title=label("Events")),
        ], height=3.0),
    ]
    if two_well:
        med_b = "Median duration, " + (names.get("B") or "B") + " (s)"
        blocks.append(rl.PlotRow([
            rl.Plot(_dot(experiment, frame, "MedDurationA", label("MedDuration"), facet_col),
                    title=label("MedDuration")),
            rl.Plot(_dot(experiment, frame, "MedDurationB", med_b, facet_col), title=med_b),
        ], height=3.0))
        stat_metrics = ["Licks", "Events", "MedDuration"]
    else:
        blocks.append(rl.PlotRow([
            rl.Plot(_dot(experiment, frame, "MedDuration", label("MedDuration"), facet_col),
                    title=label("MedDuration")),
            rl.Plot(_dot(experiment, frame, "MeanDuration", label("MeanDuration"),
                         facet_col), title=label("MeanDuration")),
        ], height=3.0))
        stat_metrics = ["Licks", "Events", "MedDuration", "MeanDuration"]
    if options.include_comparison:
        blocks += rc.stats_table(treatment_comparisons(frames, stat_metrics),
                                 caption="Treatment comparisons: consumption",
                                 well_names=names, show_phase=bool(facet_col),
                                 two_well=two_well)
    courses = [m for m in options.metrics if m not in ("PI", "EventPI")]
    if courses:
        blocks.append(rl.Heading("Over time", level=2))
        count = {"Licks", "Events"}
        for metric in courses:
            mode = "total" if metric in count else ("A" if two_well else "total")
            blocks.append(rl.Plot(
                _timecourse(experiment, metric, mode, options), height=2.6,
                title=f"{label(metric) if metric in rc.METRIC_LABELS else metric} over time",
                caption=f"Mean ± SEM per treatment in {options.binsize_min:g}-minute bins."))
    return blocks


def _all_metrics(experiment: Experiment, options: ReportOptions) -> list[Any]:
    return [
        rl.Paragraph("Every feeding-summary metric by treatment, over the whole recording — "
                     "for reference; the Results section holds the ones the assay is read by.",
                     color=rl.MUTED),
        rl.Plot(lambda: experiment.plot_feeding_summary(
            range_minutes=options.range_minutes, transform_licks=options.transform_licks),
            height=7.6),
    ]
