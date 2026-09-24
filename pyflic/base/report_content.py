"""Content helpers the reports share: statistics tables, dot plots, verdict
summaries.  The page layout is :mod:`pyflic.base.report_layout`; what goes
on the pages is decided by :mod:`pyflic.base.pdf_report` (a member),
:mod:`pyflic.base.project_report` (a Project) and the Experiment Types'
report hooks, all of which build from these.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import report_layout as rl

#: How a metric reads in a report table or axis ({A}/{B}: the well names).
METRIC_LABELS: dict[str, str] = {
    "PI": "Preference index (PI)",
    "EventPI": "Event PI",
    "Licks": "Licks ({A} + {B})",
    "Events": "Events ({A} + {B})",
    "MedDuration": "Median duration, {A} (s)",
    "MeanDuration": "Mean duration (s)",
    "dLicksA": "Δ {A} licks (paired − yoked)",
    "dLicksB": "Δ {B} licks (paired − yoked)",
    "dEventsA": "Δ {A} events (paired − yoked)",
    "dEventsB": "Δ {B} events (paired − yoked)",
    "dPI": "Δ PI (paired − yoked)",
    "dMedDurationA": "Δ median duration, {A} (s)",
    "BreakingPoint": "Breaking point (Test light events)",
}
#: Single-well layouts have no A/B.
SINGLE_WELL_LABELS: dict[str, str] = {
    "Licks": "Licks", "Events": "Events",
    "MedDuration": "Median duration (s)", "MeanDuration": "Mean duration (s)",
}


def metric_label(metric: str, well_names: dict | None = None, *,
                 two_well: bool = True) -> str:
    if not two_well and metric in SINGLE_WELL_LABELS:
        return SINGLE_WELL_LABELS[metric]
    names = well_names or {}
    template = METRIC_LABELS.get(metric, metric)
    return template.format(A=names.get("A") or "A", B=names.get("B") or "B")


def stats_table(rows: Sequence[dict], *, caption: str | None = None,
                well_names: dict | None = None, show_phase: bool = True,
                two_well: bool = True) -> list[Any]:
    """Treatment comparisons (:func:`pyflic.base.analytics.treatment_comparisons`)
    as a report table, significant p-values highlighted, plus a note on the
    tests; a short paragraph instead when there is nothing to compare."""
    if not rows:
        return [rl.Paragraph("No treatment comparison was possible here: at least two "
                             "treatments with two or more observations each are needed.",
                             color=rl.MUTED, italic=True)]
    records = []
    has_mixed = any(r.get("p_mixed") is not None for r in rows)
    for r in rows:
        rec = {"Measure": metric_label(r["metric"], well_names, two_well=two_well)}
        if show_phase:
            rec["Phase"] = r["phase"]
        rec.update({
            "Comparison": f"{r['b']} vs {r['a']}",
            "n": f"{r['n_b']} / {r['n_a']}",
            "Mean": f"{rl.fmt_value(r['mean_b'])} / {rl.fmt_value(r['mean_a'])}",
            "Difference": r["diff"],
            "Test": r.get("test", ""),
            "p": r["p_pooled"],
        })
        if has_mixed:
            rec["p (mixed)"] = r["p_mixed"] if r["p_mixed"] is not None else np.nan
        records.append(rec)
    frame = pd.DataFrame(records)

    def p_tone(value: Any) -> str | None:
        try:
            return "info" if float(value) < 0.05 else None
        except (TypeError, ValueError):
            return None

    blocks: list[Any] = [rl.Table(
        frame, caption=caption,
        formats={"p": rl.fmt_p, "p (mixed)": rl.fmt_p},
        status={"p": p_tone, "p (mixed)": p_tone})]
    note = ("n and Mean read as the first treatment named / the second.  Difference is "
            "the first minus the second.  Two treatments: Welch's t-test; more: Tukey HSD. "
            "Highlighted: p < 0.05.")
    if has_mixed:
        note += ("  p (mixed): linear mixed model with DFM nested within experiment, "
                 "which accounts for between-member and between-device variation.")
    blocks.append(rl.Paragraph(note, size=rl.SIZE_SMALL, color=rl.MUTED))
    return blocks


def dot_plot(frame: pd.DataFrame, value_col: str, *, y_label: str,
             factors: Sequence[str] | None = None, facet_col: str | None = None,
             hline_at: float | None = None):
    """A jitter + mean ± SEM dot plot of *value_col* by treatment, in the
    report's size: x is the first design factor present (or the Treatment),
    panels are *facet_col* when given (a Facet, usually)."""
    from .experiment import Experiment

    df = frame.copy()
    present = [f for f in (factors or []) if f in df.columns]
    x_col = present[0] if present else "Treatment"
    if facet_col is None or facet_col not in df.columns or df[facet_col].nunique() < 1:
        df["_Facet"] = "All"
        facet = "_Facet"
    else:
        order = list(dict.fromkeys(df[facet_col].astype(str)))
        df["_Panel"] = pd.Categorical(df[facet_col].astype(str), categories=order,
                                      ordered=True)
        facet = "_Panel"
    df = df.dropna(subset=[value_col])
    return Experiment.plot_jitter_summary(
        df, x_col=x_col, y_col=value_col, facet_col=facet, title="",
        x_label="", y_label=y_label, hline_at=hline_at, base_font_size=10.0,
        point_size=2.2)


def verdict_counts(frame: pd.DataFrame, column: str = "Verdict") -> dict[str, int]:
    if frame is None or frame.empty or column not in frame.columns:
        return {}
    return {str(k): int(v) for k, v in frame[column].astype(str).value_counts().items()}


def join_groups(pairs: Sequence[tuple[int, int]]) -> str:
    """``DFM 1 groups 1 and 2; DFM 2 group 3``."""
    by_dfm: dict[int, list[int]] = {}
    for dfm, group in pairs:
        by_dfm.setdefault(int(dfm), []).append(int(group))
    parts = []
    for dfm, groups in sorted(by_dfm.items()):
        groups = sorted(groups)
        if len(groups) == 1:
            parts.append(f"DFM {dfm} group {groups[0]}")
        else:
            parts.append(f"DFM {dfm} groups " + ", ".join(str(g) for g in groups[:-1])
                         + f" and {groups[-1]}")
    return "; ".join(parts)


def settings_blocks(global_cfg: dict | None, constants: dict | None) -> list[Any]:
    """The detection parameters and the ``constants:`` block as two tables."""
    blocks: list[Any] = []
    params = dict((global_cfg or {}).get("params") or {})
    if params:
        frame = pd.DataFrame({"Parameter": list(params), "Value": [str(v) for v in params.values()]})
        blocks.append(rl.Table(frame, caption="Detection parameters (global.params)",
                               stretch=False))
    if constants:
        frame = pd.DataFrame({"Constant": list(constants),
                              "Value": [str(v) for v in constants.values()]})
        blocks.append(rl.Table(frame, caption="Auto-removal and QC constants "
                                              "(type defaults under global.constants)",
                               stretch=False))
    return blocks
