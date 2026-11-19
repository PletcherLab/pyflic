"""The experiment report: one member, one PDF — ``analysis/experiment_report.pdf``.

Laid out by :mod:`pyflic.base.report_layout` on fixed US Letter pages:

* **Cover** — what was recorded and how it was analysed, the quality-control
  verdicts at a glance, and the contents.
* **1 Quality control** — per-DFM data integrity, two-well cross-talk
  (simultaneous feeding, bleeding), the Program timeline of each DFM with a
  program or a Paradigm state, the chambers excluded and why, the
  optogenetic light QC (on a firmware 4.0.0 recording with each Paradigm's
  rule checks, :mod:`pyflic.base.rule_figures`), then the Experiment Type's
  own checks (:meth:`Experiment.report_qc_blocks`; Progressive Ratio:
  training and its light QC).
* **2 Results** — the inference figures with their statistics, chosen by the
  Chamber Layout (two-well: preference, then consumption; single-well:
  consumption) and extended or replaced by the Experiment Type
  (:meth:`Experiment.report_results_blocks`; Progressive Ratio: the
  paired − yoked difference and the breaking point), then for an
  optogenetic member the light table's **Light behaviour**, one subsection
  per Paradigm Family the program runs, and **Paired and yoked** where a
  Linkage Group yokes chambers to a Paired one.
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
    ## The type's Control Comparisons lead the Results (ADR-0019); a paired −
    ## yoked comparison that leads is not repeated after the light results.
    lead, paired_yoked_leads = _control_blocks(experiment, options)
    doc.add(lead)
    doc.add(experiment.report_results_blocks(_generic_results(experiment, options), options))
    doc.add(_light_results(experiment, options, yoked=not paired_yoked_leads))
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
        name = Path(experiment.experiment_dir).name
        return name if experiment.slice is None else f"{name} · Slice {experiment.slice.name}"
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
    from .program_facets import CLOCK_SCHEDULE_NOTE, SOURCE_LABELS, is_program_source

    source = getattr(experiment, "facet_source", "cutoffs")
    if is_program_source(source):
        ## Windows differ by DFM, so the labels (Paradigm names) are the Facets.
        by_clock = any(CLOCK_SCHEDULE_NOTE in line for line in experiment.facet_notes())
        clock = ("; DFMs without recorded Program Intervals laid out by the "
                 "program's clock schedule, approximate" if by_clock else "")
        return (f"{' · '.join(labels) or 'none'} — {SOURCE_LABELS[source]}, each "
                f"DFM's own (facet_source: {source}){clock}")
    windows = experiment.facet_windows()
    if not windows:
        return "none (whole recording)"
    from . import windowing

    return " · ".join(f"{label} {windowing.format_range(w)}"
                      for label, w in zip(labels, windows))


def _per_dfm(values: dict[int, str]) -> str:
    """One text for every DFM, or ``DFM 1, 2: … · DFM 3: …`` when they differ."""
    by_text: dict[str, list[int]] = {}
    for dfm_id, text in sorted(values.items()):
        by_text.setdefault(text, []).append(dfm_id)
    if len(by_text) == 1:
        return next(iter(by_text))
    return " · ".join(f"DFM {', '.join(str(d) for d in ids)}: {text}"
                      for text, ids in by_text.items())


def _recording_rows(experiment: Experiment) -> list[tuple[str, str]]:
    """The cover's Generation line (what shape of data file each DFM
    recorded, ``CONTEXT.md``) and, for an optogenetic member, where its lit
    time comes from (decision 31)."""
    from .paradigm_state import GENERATION_DESCRIPTIONS

    generations = {
        int(d): (f"{dfm.generation} — "
                 f"{GENERATION_DESCRIPTIONS.get(dfm.generation, 'unknown')}")
        for d, dfm in experiment.dfms.items() if getattr(dfm, "generation", None)}
    rows: list[tuple[str, str]] = []
    if generations:
        rows.append(("Generation", _per_dfm(generations)))
    if experiment.is_optogenetic:
        sources = {int(d): str(experiment.dfms[d].light_source)
                   for d in experiment.opto.dfm_ids()}
        if sources:
            rows.append(("Lit time from", _per_dfm(sources)))
    return rows


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
    location = getattr(experiment, "data_location", None)
    if location is not None and location.referenced:
        ## Where the recording was read, both ways (ADR-0021); an experiment
        ## reading its own data/ keeps the cover it always had.
        rows.append(("Data folder", location.describe()))
    if experiment.slice is not None:
        ## A Slice is its own analysis of these minutes (ADR-0020).
        rows.append(("Slice", f"{experiment.slice.name}, minutes "
                              f"{experiment.slice.bounds_label()}, analysed as if the "
                              f"recording were only these minutes"))
    rows.append(("DFMs", ", ".join(str(d) for d in sorted(experiment.dfms))))
    rows.extend(_recording_rows(experiment))
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
    blocks.extend(_program_callout(experiment))
    ## The optogenetic light QC is every type's, not a type's own.
    blocks.extend(experiment.opto.report_glance_blocks())
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


def _ceiling_cell(experiment: Experiment, dfm_id: int) -> str:
    """The DFM's wells at the signal ceiling for longer than
    ``max_ceiling_minutes`` (its default when the removal is off: the time is
    lost from the record either way), as ``W4 538 min, …``; ``""`` for none."""
    from . import ceiling

    limit = ceiling.max_minutes(experiment.global_constants)
    limit = float(ceiling.DEFAULT_MAX_MINUTES) if limit is None else limit
    found = experiment.dfms[dfm_id].ceilings()
    return ", ".join(f"W{w} {found[w].minutes} min" for w in sorted(found)
                     if found[w].minutes > limit)


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
        lost = integrity.get("lost_records")
        ceiling_wells = _ceiling_cell(experiment, dfm_id)
        verdict = "ok" if (breaks == 0 and errors == 0 and index_ok is not False
                           and not lost and not ceiling_wells) else "warning"
        rows.append({
            "DFM": dfm_id,
            "Samples": int(integrity.get("n_rawdata") or len(experiment.dfms[dfm_id].raw_df)),
            "Start": start.strftime("%Y-%m-%d %H:%M") if start is not None and pd.notna(start) else "–",
            "Hours": (float(span) / 60.0) if span is not None else np.nan,
            "Index continuous": ("–" if index_ok is None else ("yes" if index_ok else "NO")),
            "Lost records": np.nan if lost is None else int(lost),
            "Data breaks": breaks,
            "Error flags": errors,
            "At signal ceiling": ceiling_wells or "–",
            "Verdict": verdict,
        })
    return pd.DataFrame(rows)


def _integrity_callout(experiment: Experiment, qc: dict) -> rl.Callout:
    frame = _integrity_frame(experiment, qc)
    bad = frame[frame["Verdict"] != "ok"]
    if bad.empty:
        return rl.Callout("No data breaks, no firmware error flags, a continuous "
                          "sample index and no well at the signal ceiling on every DFM.",
                          tone="ok", title="Data integrity")
    detail = "; ".join(
        f"DFM {int(r.DFM)}: {int(r['Data breaks'])} data break(s), "
        f"{int(r['Error flags'])} error flag(s)"
        + (f", {int(r['Lost records'])} lost record(s)"
           if pd.notna(r["Lost records"]) and r["Lost records"] > 0 else "")
        + (f", at the signal ceiling {r['At signal ceiling']}"
           if r["At signal ceiling"] != "–" else "")
        for _, r in bad.iterrows())
    return rl.Callout(detail + ".  See Quality control.", tone="warning",
                      title="Data integrity")


def _exclusion_frame(experiment: Experiment) -> pd.DataFrame:
    rows = []
    group = experiment.exclusion_groups_label()
    for dfm_id, chambers in sorted((experiment.excluded_chambers or {}).items()):
        for chamber in chambers:
            rows.append({"DFM": int(dfm_id), "Chamber": int(chamber), "Treatment": "",
                         "Source": "by hand",
                         "Reason": f"remove_chambers.csv, group {group}"})
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


def _program_callout(experiment: Experiment) -> list[rl.Callout]:
    """Decision 4's warning, for DFMs recorded with firmware 4.0.0 that have
    no ``Program.txt`` section; nothing when every DFM has one or needs none."""
    from .opto_light import dfms_without_program, no_program_warning

    lacking = dfms_without_program(experiment)
    if not lacking:
        return []
    return [rl.Callout(f"{no_program_warning(lacking)}.  The rows name each Paradigm but "
                       f"not its type, so the type-dependent light checks and columns are "
                       f"not made; copy the Program.txt PyMCU exported for this recording "
                       f"into data/.", tone="warning", title="Program.txt")]


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
                ## Decision 23: the count before the dual-feeding correction
                ## beside the one after it, which the correction drives to 0.
                "Both at once, before correction": float(r.get("Both_pre", np.nan)),
                "Both at once, after": both,
                "Both after, % of lick samples": (100.0 * both / union) if union > 0 else 0.0,
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
                   "which gives each such sample to the stronger well: the count before "
                   "it is the cross-talk found, the count after it what remains."
                   if corrected else
                   "The dual-feeding correction was not applied "
                   "(correct_for_dual_feeding is off), so the counts before and after it "
                   "agree.")),
            rl.Table(pd.DataFrame(rows), caption="Lick samples per well and on both wells "
                                                 "at once, per chamber",
                     formats={"Both after, % of lick samples": "{:.2f}"}),
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


def _timeline_blocks(experiment: Experiment) -> list[Any]:
    """The Program timeline of every DFM that has one
    (:mod:`pyflic.base.program_timeline`)."""
    from . import program_timeline

    from .paradigm_state import GENERATION_PARADIGM

    ids = program_timeline.timeline_dfms(experiment)
    if not ids:
        return []
    firmware4 = any(getattr(experiment.dfms[d], "generation", None) == GENERATION_PARADIGM
                    for d in ids)
    blocks: list[Any] = [
        rl.Heading("Program timeline", level=2),
        rl.Paragraph(
            "What each DFM's program ran and what each well did: the Program Intervals, "
            "coloured by Paradigm type, above one lane per well with its Acclimation "
            "(Phase A) shaded, its Activations as ticks and the share of each bin its LED "
            "was on.  A firmware 4.0.0 recording states all of this row by row; an older "
            "one is laid out from Program.txt's clock times and shows the sampled light "
            "only."
            + ("  On a firmware 4.0.0 recording a lane is also hatched where Program.txt "
               "sets its well On and grey where it sets it Off, and each restart of the "
               "DFM is a red band." if firmware4 else "")),
    ]
    ## The figure titles itself ("DFM n — Program timeline").
    for dfm_id in ids:
        blocks.append(rl.Plot(lambda d=dfm_id: program_timeline.plot_dfm(experiment, d),
                              height=5.4))
    return blocks


def _qc_blocks(experiment: Experiment, qc: dict) -> list[Any]:
    blocks: list[Any] = [
        rl.Heading("Data integrity", level=2),
        rl.Paragraph("Per DFM: samples recorded, gaps in the sample clock longer than "
                     f"{qc.get(next(iter(qc), 0), {}).get('data_breaks_multiplier', 4.0):g}× "
                     "the sampling interval (data breaks), non-zero firmware error codes, "
                     "whether the sample index runs without a gap, with the records "
                     "its gaps lost, and the wells that rested at the signal ceiling "
                     "(the top of the sensor's range, where no lick can show) for longer "
                     "than max_ceiling_minutes."),
        rl.Table(_integrity_frame(experiment, qc), caption="Data integrity by DFM",
                 formats={"Hours": "{:.1f}"}, status={"Verdict": rl.tone_of}),
    ]
    if _is_two_well(experiment):
        blocks += _crosstalk_blocks(experiment, qc)
    blocks += _timeline_blocks(experiment)
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
    blocks.extend(experiment.opto.report_qc_blocks())
    blocks.extend(experiment.report_qc_blocks())
    return blocks


# ---------------------------------------------------------------------------
# Results, by Chamber Layout
# ---------------------------------------------------------------------------

def _summary_frames(experiment: Experiment, options: ReportOptions
                    ) -> tuple[pd.DataFrame, list[tuple[str, pd.DataFrame]], str | None]:
    """``(rows for the dot plots, [(phase, rows)] for the statistics, the
    panel column)`` — per Facet when the experiment has them, grouped by the
    Facet label (Program Facets have no windows shared by every DFM)."""
    facet = pd.DataFrame()
    if experiment.facet_windows() or getattr(experiment, "facet_source", "cutoffs") != "cutoffs":
        facet = experiment.feeding_summary_facet(transform_licks=options.transform_licks)
    if facet is not None and not facet.empty and "Facet" in facet.columns:
        labels = list(dict.fromkeys(facet["Facet"].astype(str)))
        frames = [(label, facet[facet["Facet"].astype(str) == label]) for label in labels]
        return facet, frames, "Facet"
    summary = experiment.feeding_summary(range_minutes=options.range_minutes,
                                         transform_licks=options.transform_licks)
    from .slices import whole_label

    return summary, [(whole_label(experiment.slice), summary)], None


def _value(frame: pd.DataFrame, metric: str, *, transformed: bool = False) -> pd.Series:
    from .analytics import _resolve_metric_col

    return pd.to_numeric(_resolve_metric_col(frame, metric, transformed=transformed),
                         errors="coerce")


#: Metrics measured on feeding events (bout durations and the gaps between
#: them): undefined in a chamber with no feeding event.
_EVENT_METRIC_PARTS = ("Duration", "Int", "TimeBtw", "EventPI")


def _total(frame: pd.DataFrame, *columns: str) -> float | None:
    present = [c for c in columns if c in frame.columns]
    if not present:
        return None
    return float(sum(pd.to_numeric(frame[c], errors="coerce").fillna(0).sum()
                     for c in present))


def _empty_reason(frame: pd.DataFrame, metric: str) -> str | None:
    """Why *metric* has no value in any chamber of *frame*, when the frame
    can say: no feeding event, or no lick at all.  ``None`` when it cannot."""
    if any(part in metric for part in _EVENT_METRIC_PARTS):
        events = _total(frame, "Events", "EventsA", "EventsB")
        if events == 0:
            return ("No feeding event was detected in any chamber, so there is no bout "
                    "to measure.")
    if metric in ("PI", "EventPI"):
        licks = _total(frame, "Licks", "LicksA", "LicksB")
        if licks == 0:
            return "No chamber licked at either well, so there is no preference to measure."
    return None


def _dot(experiment: Experiment, frame: pd.DataFrame, metric: str, label: str,
         facet_col: str | None, *, hline: float | None = None, transformed: bool = False):
    def build():
        df = frame.copy()
        df["_Value"] = _value(df, metric, transformed=transformed)
        return rc.dot_plot(df, "_Value", y_label=label, facet_col=facet_col,
                           hline_at=hline, factors=experiment.design_factors,
                           empty_reason=_empty_reason(frame, metric))
    return build


def _timecourse(experiment: Experiment, metric: str, mode: str, options: ReportOptions):
    def build():
        ## A metric no chamber defines over the whole recording has nothing to
        ## bin either: say why, rather than letting the binned plot fail.
        summary = experiment.feeding_summary(range_minutes=options.range_minutes,
                                             transform_licks=options.transform_licks)
        try:
            values = _value(summary, metric, transformed=options.transform_licks)
        except Exception:  # noqa: BLE001 - an unknown metric fails in the plot below
            values = None
        if values is not None and not pd.to_numeric(values, errors="coerce").notna().any():
            reason = _empty_reason(summary, metric)
            label = (rc.metric_label(metric, experiment.well_names or {},
                                     two_well=_is_two_well(experiment))
                     if metric in rc.METRIC_LABELS else metric)
            raise rl.NothingToDraw(f"No chamber has a value for {label}."
                                   + (f"  {reason}" if reason else ""))
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
    ## One treatment: say once that nothing is compared, instead of a
    ## "no comparison was possible" line under every figure.
    levels = rc.treatment_levels(frame)
    compare = options.include_comparison and len(levels) != 1
    if len(levels) == 1:
        blocks.append(rc.single_treatment_note(levels[0]))

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
        if compare:
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
            rl.Plot(_dot(experiment, frame, "Licks", label("Licks") + tx, facet_col,
                         transformed=options.transform_licks),
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
    if compare:
        blocks += rc.stats_table(treatment_comparisons(frames, stat_metrics,
                                                       transformed=options.transform_licks),
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


def _light_programs(experiment: Experiment) -> tuple[list, dict]:
    """``([(program, DFMs)], {(None, DFM): light keys})`` of the member's
    generation-3 DFMs the light QC covers, for the *Light behaviour*
    subsections (:func:`report_content.light_families`)."""
    from .paradigm_state import GENERATION_PARADIGM

    program = getattr(experiment, "opto_program", None)
    ids = [d for d in experiment.opto.dfm_ids()
           if int(getattr(experiment.dfms[d], "generation", 0) or 0) == GENERATION_PARADIGM]
    if program is None or not ids:
        return [], {}
    return [(program, ids)], {(None, int(d)): rc.program_light_keys(program, [d]) for d in ids}


def _facet_frames(rows: pd.DataFrame, facet_col: str | None) -> list[tuple[str, pd.DataFrame]]:
    if facet_col:
        labels = list(dict.fromkeys(rows[facet_col].astype(str)))
        return [(label, rows[rows[facet_col].astype(str) == label]) for label in labels]
    return [("Whole recording", rows)]


def _light_results(experiment: Experiment, options: ReportOptions, *,
                   yoked: bool = True) -> list[Any]:
    """"Light behaviour", after the feeding results, for an optogenetic
    member: one subsection per Paradigm Family the program runs (decision 9;
    fixed length by its branch, and the Acclimation), each with a dot plot
    per headline metric of the light table by Treatment, per Facet when the
    experiment has them, and the generic treatment comparisons; open loop
    adds *Feeding in light and dark* (decision 11).  A member whose program
    runs no family (no generation-3 DFM, or no program) keeps the four
    metrics of before in one section.  A metric no row defines is left out
    and named.  Then *Paired and yoked* when a Linkage Group yokes chambers
    to a Paired one (decision 12), unless *yoked* is false because it led
    the Results as the type's Role comparison."""
    if not experiment.is_optogenetic:
        return []
    tail = _yoked_blocks(experiment, options) if yoked else []
    from . import light_summary as ls

    facet = ls.light_summary_facet(experiment)
    if facet is not None and not facet.empty:
        frame, facet_col = facet, "Facet"
    else:
        frame = ls.light_summary(experiment, range_minutes=options.range_minutes)
        facet_col = None
    kept = ls.behaviour_rows(frame)
    blocks: list[Any] = [rl.Heading("Light behaviour", level=2)]
    if kept is None or kept.empty:
        return blocks + [rl.Callout("No chamber with a treatment has a light table row.",
                                    tone="warning")] + tail
    programs, keys = _light_programs(experiment)
    families = rc.light_families(programs)
    if not families:
        return (blocks + _light_headline_blocks(experiment, frame, kept, facet_col, options)
                + tail)
    per = "per Facet" if facet_col else "over the whole recording"
    blocks.append(rl.Paragraph(
        f"What the light did, from the light table (analysis/light_summary.csv), one point "
        f"per chamber, {per}, under each Paradigm Family the program runs: each family's "
        f"own measures, for the chambers that ran it.  A chamber whose window ran Paradigms "
        f"of several families shows under each.  The diamond and bars are the treatment "
        f"mean ± SEM."
        + _role_sentence(frame, kept)))
    compare = options.include_comparison and len(rc.treatment_levels(kept)) != 1
    for family in families:
        rows = family.rows(kept, keys)
        blocks += _light_family_blocks(experiment, family, rows, facet_col, compare=compare)
    return blocks + tail


def _role_sentence(frame: pd.DataFrame, kept: pd.DataFrame) -> str:
    """What the Roles do to the statistics (decision 12): Progressive Ratio
    compares its paired chambers only; under any other type a yoked fly
    keeps its lit time, and the Activation statistics leave it out."""
    if len(kept) < len(frame):
        return "  Paired chambers only: a yoked fly's light is its partner's."
    roles = kept["Role"].astype(str) if "Role" in kept.columns else pd.Series(dtype=str)
    if (roles == "Yoked").any():
        return ("  Yoked chambers keep their lit time, beside their partners'; the "
                "statistics of Activations and the other Activation measures compare the "
                "chambers that earn light (Paired, Self, Mutual).")
    return ""


def _light_family_blocks(experiment: Experiment, family, rows: pd.DataFrame | None,
                         facet_col: str | None, *, compare: bool) -> list[Any]:
    """One family's subsection of *Light behaviour*: its dot plots, two to a
    row, the open-loop *Feeding in light and dark*, and the comparisons."""
    from .analytics import treatment_comparisons
    from .paradigm_families import OPEN_LOOP

    blocks: list[Any] = [rl.Heading(f"Light behaviour: {family.title}", level=2)]
    if rows is None or rows.empty:
        return blocks + [rl.Paragraph(f"No chamber with a treatment ran it: {family.about}.",
                                      color=rl.MUTED, italic=True)]
    metrics = [m for m in family.metrics
               if m in rows.columns and pd.to_numeric(rows[m], errors="coerce").notna().any()]
    missing = [m for m in family.metrics if m not in metrics]
    blocks.append(rl.Paragraph(
        f"{family.about[0].upper()}{family.about[1:]}."
        + (f"  Not defined here: {', '.join(rc.metric_label(m) for m in missing)}."
           if missing else "")))
    pending: list[rl.Plot] = []
    for metric in metrics:
        label = rc.metric_label(metric)
        ## Half the width each, a last one alone too, as the Project's.
        pending.append(rl.Plot(_dot(experiment, rows, metric, label, facet_col), title=label,
                               width=(rl.CONTENT_W - 0.25) / 2))
        if len(pending) == 2:
            blocks.append(rl.PlotRow(pending, height=3.0))
            pending = []
    if pending:
        blocks.append(rl.PlotRow(pending, height=3.0))
    if family.key == OPEN_LOOP and _has_module(experiment, "light_dark"):
        blocks.append(rl.Plot(lambda: rc.light_dark_plot(rows, facet_col=facet_col),
                              height=3.0, title="Feeding in light and dark",
                              caption="Each open-loop chamber's licks per minute in its own "
                                      "light and in the dark, joined by a line; the diamond "
                                      "is the treatment mean.  The light feeding index above "
                                      "is (lit − dark) / (lit + dark) of the two."))
    if compare and metrics:
        blocks += rc.light_comparison_blocks(
            metrics, rows, lambda keep: treatment_comparisons(_facet_frames(rows, facet_col), keep),
            caption=f"Treatment comparisons: light behaviour, {family.title}",
            show_phase=bool(facet_col))
    return blocks


def _control_blocks(experiment: Experiment, options: ReportOptions
                    ) -> tuple[list[Any], bool]:
    """``(blocks, paired_yoked_leads)``: the type's Control Comparisons
    (:mod:`control_comparisons`), the Role comparison first, for the front
    of the Results; nothing under a type with no ``control:`` block.  A
    Role comparison of Paired and Yoked is the paired − yoked comparison
    the report already makes: the Paired and yoked module's, moved here
    (*paired_yoked_leads*, so it is not repeated), or the Ratio module's
    Paired-Yoked Difference, which opens the results that follow."""
    from . import control_comparisons as cc

    etype = getattr(experiment, "experiment_type", None)
    roles, factor = cc.role_pair(etype), cc.treatment_control(etype)
    if roles is None and factor is None:
        return [], False
    plot_frame, frames, facet_col = _summary_frames(experiment, options)
    names = experiment.well_names or {}
    two_well = _is_two_well(experiment)
    blocks: list[Any] = []
    leads = False
    if roles is not None and cc.is_paired_yoked(roles):
        if _has_module(experiment, "yoked"):
            leads = True
            blocks += _yoked_blocks(experiment, options) or [
                rl.Heading("Paired and yoked", level=2),
                rl.Paragraph("The type compares Paired with Yoked chambers, and no linkage "
                             "group here lights a Yoked chamber with a Paired one.",
                             color=rl.MUTED, italic=True)]
        elif _has_module(experiment, "ratio"):
            blocks += [rl.Heading("Paired and yoked", level=2),
                       rl.Paragraph("The type compares Paired with Yoked chambers: the "
                                    "Paired-Yoked Difference that opens the results below.")]
    elif roles is not None:
        blocks += cc.role_blocks(frames, roles, plot_frame=plot_frame, facet_col=facet_col,
                                 well_names=names, two_well=two_well)
    if factor is not None:
        blocks += cc.control_blocks(
            frames, factor, etype.control_level(experiment.global_config),
            factors=experiment.design_factors, plot_frame=plot_frame, facet_col=facet_col,
            well_names=names, two_well=two_well)
    return blocks, leads


def _has_module(source: Any, module_id: str) -> bool:
    """Whether *source*'s Experiment Type includes the Analysis Module
    *module_id* (ADR-0019)."""
    has = getattr(getattr(source, "experiment_type", None), "has_module", None)
    return bool(has(module_id)) if callable(has) else False


def _yoked_blocks(experiment: Experiment, options: ReportOptions) -> list[Any]:
    """*Paired and yoked* (decision 12): where a Linkage Group yokes
    chambers to a Paired one, under a type with the Paired and yoked module
    (the Ratio module keeps its own), the paired fly's feeding minus its
    yoked partners', by treatment (:func:`analytics.yoked_comparison`,
    :mod:`yoked_report`)."""
    from . import analytics
    from . import yoked_report as yr

    if not _has_module(experiment, "yoked"):
        return []

    try:
        if not analytics.yoked_groups(experiment):
            return []
        frame = analytics.yoked_comparison(experiment, range_minutes=options.range_minutes,
                                           transform_licks=options.transform_licks)
    except Exception as exc:  # noqa: BLE001 - one subsection, not the report
        return [rl.Heading("Paired and yoked", level=2),
                rl.Callout(f"The paired − yoked comparison could not be made: {exc}",
                           tone="failed")]
    tables = yr.yoked_table_blocks(frame, well_names=experiment.well_names or {},
                                   two_well=_is_two_well(experiment),
                                   include_comparison=options.include_comparison,
                                   transformed=options.transform_licks)
    if not tables:
        return []
    return [rl.Heading("Paired and yoked", level=2),
            rl.Paragraph("A Linkage Group that lights a Paired chamber's fly by its own "
                         "feeding lights its Yoked chambers at the same moments, whatever "
                         "their flies do.  Per group, the paired fly's feeding minus the mean "
                         "of its yoked partners', by treatment: is the paired fly different "
                         "(against zero), and do treatments differ?"),
            *tables,
            *yr.yoked_figure_blocks(frame, well_names=experiment.well_names or {},
                                    factors=experiment.design_factors,
                                    transformed=options.transform_licks)]


def _light_headline_blocks(experiment: Experiment, frame: pd.DataFrame, kept: pd.DataFrame,
                           facet_col: str | None, options: ReportOptions) -> list[Any]:
    """*Light behaviour* as before the Paradigm Families, for a member whose
    program runs none (generations 1 and 2, or no program): the four
    metrics of before, in one section."""
    from . import light_summary as ls
    from .analytics import treatment_comparisons

    blocks: list[Any] = []
    frames = _facet_frames(kept, facet_col)
    headline = ls.experiment_headline_metrics(experiment)
    metrics = [m for m in headline
               if m in kept.columns and pd.to_numeric(kept[m], errors="coerce").notna().any()]
    per = "per Facet" if facet_col else "over the whole recording"
    blocks.append(rl.Paragraph(
        f"What the light did, from the light table (analysis/light_summary.csv), one "
        f"point per chamber, {per}: Activations are the LED on-periods a well's own "
        f"Paradigm started, lit time is the chamber's, Firmware Events count under a "
        f"ratio Paradigm, and Acclimation end is the minute the chamber's wells left "
        f"Acclimation.  The diamond and bars are the treatment mean ± SEM."
        + ("  Paired chambers only: a yoked fly's light is its partner's."
           if len(kept) < len(frame) else "")
        + ("" if len(metrics) == len(headline) else
           f"  Not defined here: "
           f"{', '.join(m for m in headline if m not in metrics)}.")))
    pending: list[rl.Plot] = []
    for metric in metrics:
        label = rc.metric_label(metric)
        pending.append(rl.Plot(_dot(experiment, kept, metric, label, facet_col), title=label))
        if len(pending) == 2:
            blocks.append(rl.PlotRow(pending, height=3.0))
            pending = []
    if pending:
        blocks.append(rl.PlotRow(pending, height=3.0))
    if options.include_comparison and metrics and len(rc.treatment_levels(kept)) != 1:
        blocks += rc.stats_table(treatment_comparisons(frames, metrics),
                                 caption="Treatment comparisons: light behaviour",
                                 show_phase=bool(facet_col))
    return blocks


def _all_metrics(experiment: Experiment, options: ReportOptions) -> list[Any]:
    text = ("Every feeding-summary metric by treatment, over the whole recording — "
            "for reference; the Results section holds the ones the assay is read by.")
    ## The figure leaves out a metric no chamber has a value for; say which.
    missing = experiment.feeding_plot_metrics_without_values(
        range_minutes=options.range_minutes, transform_licks=options.transform_licks)
    if missing:
        summary = experiment.feeding_summary(range_minutes=options.range_minutes,
                                             transform_licks=options.transform_licks)
        events = _total(summary, "Events", "EventsA", "EventsB")
        text += (f"  Not drawn, because no chamber has a value: {', '.join(missing)}"
                 + ("; no feeding event was detected, so there is no bout to measure."
                    if events == 0 else "."))
    return [
        rl.Paragraph(text, color=rl.MUTED),
        rl.Plot(lambda: experiment.plot_feeding_summary(
            range_minutes=options.range_minutes, transform_licks=options.transform_licks),
            height=7.6),
    ]
