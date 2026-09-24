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
    "dPersistA": "Δ {A} persistence, min (paired − yoked)",
    "PersistA": "{A} persistence (min)",
    "BreakingPoint": "Breaking point (light events earned)",
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


def _p_tone(value: Any) -> str | None:
    try:
        return "info" if float(value) < 0.05 else None
    except (TypeError, ValueError):
        return None


def stats_table(rows: Sequence[dict], *, caption: str | None = None,
                well_names: dict | None = None, show_phase: bool = True,
                two_well: bool = True) -> list[Any]:
    """Treatment comparisons (:func:`pyflic.base.analytics.treatment_comparisons`)
    as a report table, significant p-values highlighted, plus a note on the
    tests; a short paragraph instead when there is nothing to compare.  Rows
    from :func:`~pyflic.base.analytics.breaking_point_comparisons` add a
    log-rank column."""
    if not rows:
        return [rl.Paragraph("No treatment comparison was possible here: at least two "
                             "treatments with two or more observations each are needed.",
                             color=rl.MUTED, italic=True)]
    records = []
    has_mixed = any(r.get("p_mixed") is not None for r in rows)
    has_logrank = any(r.get("p_logrank") is not None for r in rows)
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
        if has_logrank:
            rec["p (log-rank)"] = (r["p_logrank"] if r.get("p_logrank") is not None
                                   else np.nan)
        records.append(rec)
    frame = pd.DataFrame(records)

    p_cols = ("p", "p (mixed)", "p (log-rank)")
    blocks: list[Any] = [rl.Table(
        frame, caption=caption,
        formats={c: rl.fmt_p for c in p_cols},
        status={c: _p_tone for c in p_cols})]
    note = ("n and Mean read as the first treatment named / the second.  Difference is "
            "the first minus the second.  Two treatments: Welch's t-test; more: Tukey HSD. "
            "Highlighted: p < 0.05.")
    if has_mixed:
        note += ("  p (mixed): linear mixed model with DFM nested within experiment, "
                 "which accounts for between-member and between-device variation.")
    if has_logrank:
        note += ("  p (log-rank): pairwise log-rank test on the ratio reached, which "
                 "treats a censored breaking point as the lower bound it is; the other "
                 "tests enter it as observed.")
    blocks.append(rl.Paragraph(note, size=rl.SIZE_SMALL, color=rl.MUTED))
    return blocks


def zero_test_table(rows: Sequence[dict], *, caption: str | None = None,
                    well_names: dict | None = None,
                    show_phase: bool = False) -> list[Any]:
    """Paired − yoked differences tested against zero within each treatment
    (:func:`pyflic.base.analytics.zero_tests`) as a report table, plus a note
    on the tests; a short paragraph instead when nothing could be tested."""
    if not rows:
        return [rl.Paragraph("No test against zero was possible here: a treatment needs "
                             "at least two chamber groups whose differences are not all "
                             "equal.", color=rl.MUTED, italic=True)]
    has_mixed = any(r.get("p_mixed") is not None for r in rows)
    records = []
    for r in rows:
        rec = {"Measure": metric_label(r["metric"], well_names)}
        if show_phase:
            rec["Phase"] = r["phase"]
        rec.update({
            "Treatment": r["treatment"],
            "n": r["n"],
            "Mean ± SEM": f"{rl.fmt_value(r['mean'])} ± {rl.fmt_value(r['sem'])}",
            "p (paired t)": r["p_t"],
            "p (Wilcoxon)": r["p_wilcoxon"] if r.get("p_wilcoxon") is not None else np.nan,
        })
        if has_mixed:
            rec["p (mixed)"] = r["p_mixed"] if r.get("p_mixed") is not None else np.nan
        records.append(rec)
    p_cols = ("p (paired t)", "p (Wilcoxon)", "p (mixed)")
    blocks: list[Any] = [rl.Table(
        pd.DataFrame(records), caption=caption,
        formats={c: rl.fmt_p for c in p_cols},
        status={c: _p_tone for c in p_cols})]
    note = ("Is the paired fly different from its yoked partner?  Each treatment's "
            "chamber-group differences (paired − yoked) against zero: the paired t-test, "
            "which is a one-sample t-test on the differences, with the Wilcoxon "
            "signed-rank test beside it.  Highlighted: p < 0.05.")
    if has_mixed:
        note += ("  p (mixed): the intercept of a linear mixed model with DFM nested "
                 "within experiment.")
    blocks.append(rl.Paragraph(note, size=rl.SIZE_SMALL, color=rl.MUTED))
    return blocks


def censored_dot_plot(frame: pd.DataFrame, value_col: str, *, y_label: str,
                      censored_col: str = "Censored",
                      factors: Sequence[str] | None = None):
    """A jitter + mean ± SEM dot plot of *value_col* by treatment in which a
    censored observation, a lower bound, is an open symbol and an observed one
    is filled.  x is the first design factor present, or the Treatment."""
    import plotnine as p9

    from .analytics import as_bool
    from .experiment import _OKABE_ITO

    df = frame.copy()
    present = [f for f in (factors or []) if f in df.columns]
    x_col = present[0] if present else "Treatment"
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])
    if df.empty:
        return p9.ggplot() + p9.labs(title="No data")
    levels = sorted(df[x_col].astype(str).unique())
    df["_x"] = pd.Categorical(df[x_col].astype(str), categories=levels, ordered=True)
    censored = (as_bool(df[censored_col]) if censored_col in df.columns
                else np.zeros(len(df), dtype=bool))
    palette = {lvl: _OKABE_ITO[i % len(_OKABE_ITO)] for i, lvl in enumerate(levels)}
    g = p9.ggplot(df, p9.aes("_x", value_col))
    if (~censored).any():
        g += p9.geom_jitter(p9.aes(color="_x"), data=df[~censored], width=0.2, height=0,
                            size=2.4, alpha=0.85, random_state=1)
    if censored.any():
        g += p9.geom_jitter(p9.aes(color="_x"), data=df[censored], width=0.2, height=0,
                            size=2.4, fill="white", stroke=0.9, random_state=2)
    g += p9.stat_summary(fun_y=np.mean, geom="point", color="black", shape="D", size=3.5,
                         fill="black")
    g += p9.stat_summary(fun_ymin=lambda x: x.mean() - x.sem(),
                         fun_ymax=lambda x: x.mean() + x.sem(),
                         geom="errorbar", color="black", size=0.7, width=0.15)
    caption = "Open: censored, a lower bound" if censored.any() else None
    return (g
            + p9.scale_color_manual(values=palette, guide=None)
            + p9.labs(x="", y=y_label, caption=caption)
            + p9.theme_classic(base_size=10.0))


def still_responding_plot(frame: pd.DataFrame, *, value_col: str = "BreakingPoint",
                          censored_col: str = "Censored", group_col: str = "Treatment",
                          base_font_size: float = 10.0,
                          figsize: tuple[float, float] | None = None):
    """The still-responding curve (:func:`pyflic.base.analytics.still_responding`):
    per treatment, the Kaplan-Meier fraction of paired flies that reached each
    ratio, a censored fly as a tick on its curve."""
    import plotnine as p9

    from .analytics import still_responding
    from .experiment import _OKABE_ITO

    steps, ticks = still_responding(frame, value_col=value_col,
                                    censored_col=censored_col, group_col=group_col)
    if steps.empty:
        return p9.ggplot() + p9.labs(
            title="Still responding: no chamber group with a breaking point")
    levels = sorted(steps[group_col].astype(str).unique())
    n_of = steps.groupby(group_col)["n"].first()
    label = {g: f"{g} (n = {int(n_of[g])})" for g in levels}
    palette = {label[g]: _OKABE_ITO[i % len(_OKABE_ITO)] for i, g in enumerate(levels)}
    order = [label[g] for g in levels]
    steps = steps.copy()
    steps["_g"] = pd.Categorical(steps[group_col].astype(str).map(label),
                                 categories=order, ordered=True)
    g = (p9.ggplot(steps, p9.aes("Ratio", "Fraction", color="_g"))
         + p9.geom_step(direction="hv", size=0.9))
    if not ticks.empty:
        ticks = ticks.copy()
        ticks["_g"] = pd.Categorical(ticks[group_col].astype(str).map(label),
                                     categories=order, ordered=True)
        g += p9.geom_point(ticks, p9.aes("Ratio", "Fraction", color="_g"), shape="|",
                           size=5, stroke=1.0)
    g = (g
         + p9.scale_color_manual(values=palette)
         + p9.scale_y_continuous(limits=(0.0, 1.02), breaks=[0.0, 0.25, 0.5, 0.75, 1.0])
         + p9.labs(title="Still responding: paired flies reaching each ratio",
                   x="Ratio reached (lick-backed Test light events)",
                   y="Fraction of paired flies", color="",
                   caption="Kaplan-Meier.  Ticks: censored flies.")
         + p9.theme_bw(base_size=base_font_size)
         + p9.theme(legend_position="bottom"))
    if figsize is not None:
        g += p9.theme(figure_size=figsize)
    return g


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
