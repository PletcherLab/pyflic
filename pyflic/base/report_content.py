"""Content helpers the reports share: statistics tables, dot plots, verdict
summaries.  The page layout is :mod:`pyflic.base.report_layout`; what goes
on the pages is decided by :mod:`pyflic.base.pdf_report` (a member),
:mod:`pyflic.base.project_report` (a Project) and the Experiment Types'
report hooks, all of which build from these.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from . import opto_program as op
from . import paradigm_families as pfam
from . import report_layout as rl
from .pubfigures import LIGHT_METRICS

#: How a metric reads in a report table or axis ({A}/{B}: the well names).
#: The light table's metrics read as the Plot Editor names them
#: (:data:`pubfigures.LIGHT_METRICS <pyflic.base.pubfigures.LIGHT_METRICS>`,
#: with their units).
METRIC_LABELS: dict[str, str] = {
    **dict(LIGHT_METRICS),
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
    "SimulatedYokedBreakingPoint": "Simulated yoked breaking point",
    "dBreakingPoint": "Δ breaking point (paired − simulated yoked)",
}
#: Single-well layouts have no A/B.
SINGLE_WELL_LABELS: dict[str, str] = {
    "Licks": "Licks", "Events": "Events",
    "MedDuration": "Median duration (s)", "MeanDuration": "Mean duration (s)",
}

#: How a label says its licks are on the fourth-root scale
#: (``transform_licks``).  A paired − yoked difference of transformed licks
#: is kept on that scale — a variance-stabilised contrast — and says so.
ROOT_SCALE = "¼-root scale"


def is_lick_difference(metric: str) -> bool:
    """Whether *metric* is a difference of Licks (``dLicksA``, ``dLicks``):
    on the fourth-root scale when the licks were transformed."""
    return str(metric).startswith("dLicks")


def on_root_scale(label: str) -> str:
    """*label* marked as on the fourth-root scale, inside its last
    parentheses when it has them: ``Δ Sucrose licks (paired − yoked)`` →
    ``Δ Sucrose licks (¼-root scale, paired − yoked)``."""
    if label.endswith(")") and "(" in label:
        at = label.rindex("(")
        return f"{label[:at + 1]}{ROOT_SCALE}, {label[at + 1:]}"
    return f"{label} ({ROOT_SCALE})"


def metric_label(metric: str, well_names: dict | None = None, *,
                 two_well: bool = True, transformed: bool = False) -> str:
    """How *metric* reads in a table or on an axis.  With *transformed* (the
    licks were fourth-root transformed), a lick difference says it is on the
    ¼-root scale (:func:`on_root_scale`)."""
    if not two_well and metric in SINGLE_WELL_LABELS:
        return SINGLE_WELL_LABELS[metric]
    names = well_names or {}
    template = METRIC_LABELS.get(metric, metric)
    label = template.format(A=names.get("A") or "A", B=names.get("B") or "B")
    if transformed and is_lick_difference(metric):
        label = on_root_scale(label)
    return label


def _p_tone(value: Any) -> str | None:
    try:
        return "info" if float(value) < 0.05 else None
    except (TypeError, ValueError):
        return None


def treatment_levels(frame: pd.DataFrame | None, column: str = "Treatment") -> list[str]:
    """The treatments *frame* holds rows for, in order of appearance."""
    if frame is None or frame.empty or column not in frame.columns:
        return []
    return list(dict.fromkeys(frame[column].dropna().astype(str)))


def single_treatment_note(treatment: str) -> Any:
    """Said once, where a report would otherwise repeat under every figure
    that no comparison was possible: a one-treatment design is a design, not
    a failure."""
    return rl.Paragraph(f"One treatment, {treatment}, so nothing is compared between "
                        f"treatments: the figures describe it alone.",
                        color=rl.MUTED, italic=True)


def light_comparison_blocks(metrics: Sequence[str], rows: pd.DataFrame,
                            compare: Callable[[list[str]], Sequence[dict]],
                            **table_kw: Any) -> list[Any]:
    """The treatment comparisons of a *Light behaviour* subsection, member
    or Project.  A metric with the same value in every chamber has nothing
    to compare (an open-loop chamber's lit time, when its light was on for
    the whole program): it is named once, rather than leaving
    :func:`stats_table` to say that two treatments are needed, which they
    were not.  *compare* takes the metrics that vary and returns their
    comparison rows; *table_kw* goes to :func:`stats_table`."""
    varying = [m for m in metrics
               if pd.to_numeric(rows[m], errors="coerce").dropna().nunique() > 1]
    constant = [m for m in metrics if m not in varying]
    blocks: list[Any] = []
    if constant:
        names = ", ".join(metric_label(m) for m in constant)
        blocks.append(rl.Paragraph(
            f"{names} {'is' if len(constant) == 1 else 'are'} the same in every chamber, "
            f"so not compared between treatments.", color=rl.MUTED, italic=True))
    if varying:
        blocks += stats_table(compare(varying), **table_kw)
    return blocks


def stats_table(rows: Sequence[dict], *, caption: str | None = None,
                well_names: dict | None = None, show_phase: bool = True,
                two_well: bool = True, transformed: bool = False) -> list[Any]:
    """Treatment comparisons (:func:`pyflic.base.analytics.treatment_comparisons`)
    as a report table, significant p-values highlighted, plus a note on the
    tests; a short paragraph instead when there is nothing to compare.  Rows
    from :func:`~pyflic.base.analytics.breaking_point_comparisons` add a
    log-rank column.  *transformed*: the licks were fourth-root transformed,
    so a lick difference is labelled as on that scale."""
    if not rows:
        return [rl.Paragraph("No treatment comparison was possible here: at least two "
                             "treatments with two or more observations each are needed.",
                             color=rl.MUTED, italic=True)]
    records = []
    has_mixed = any(r.get("p_mixed") is not None for r in rows)
    has_logrank = any(r.get("p_logrank") is not None for r in rows)
    for r in rows:
        rec = {"Measure": metric_label(r["metric"], well_names, two_well=two_well,
                                       transformed=transformed)}
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
                    show_phase: bool = False, transformed: bool = False) -> list[Any]:
    """Paired − yoked differences tested against zero within each treatment
    (:func:`pyflic.base.analytics.zero_tests`) as a report table, plus a note
    on the tests; a short paragraph instead when nothing could be tested.
    *transformed*: the licks were fourth-root transformed, so a lick
    difference is labelled as on that scale."""
    if not rows:
        return [rl.Paragraph("No test against zero was possible here: a treatment needs "
                             "at least two chamber groups whose differences are not all "
                             "equal.", color=rl.MUTED, italic=True)]
    has_mixed = any(r.get("p_mixed") is not None for r in rows)
    records = []
    for r in rows:
        rec = {"Measure": metric_label(r["metric"], well_names, transformed=transformed)}
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
             hline_at: float | None = None, empty_reason: str | None = None):
    """A jitter + mean ± SEM dot plot of *value_col* by treatment, in the
    report's size: x is the first design factor present (or the Treatment),
    panels are *facet_col* when given (a Facet, usually).

    When no chamber has a value it raises :class:`report_layout.NothingToDraw`
    with *empty_reason* (why the values are undefined) rather than handing
    plotnine an empty frame, which fails with a message about faceting."""
    from .experiment import Experiment

    values = (pd.to_numeric(frame[value_col], errors="coerce")
              if value_col in frame.columns else pd.Series(dtype=float))
    if not values.notna().any():
        raise rl.NothingToDraw(f"No chamber has a value for {y_label}."
                               + (f"  {empty_reason}" if empty_reason else ""))
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


# ---------------------------------------------------------------------------
# Light behaviour, by Paradigm Family (docs/paradigm-analyses-plan.md, S3)
# ---------------------------------------------------------------------------

#: The key of the Light behaviour subsection of the Acclimation that
#: FixedInterval, ProgressiveRatio and FixedRatio can begin with (decision 9's
#: last row): closed loop, but read from its own columns.
ACCLIMATION = "acclimation"
#: What a light-table row of several Paradigms says in ``ParadigmType``
#: (``light_summary.MIXED``).
_MIXED = "mixed"


@dataclass(frozen=True)
class LightFamily:
    """One subsection of the reports' *Light behaviour*: a Paradigm Family,
    fixed length by its branch (their headline sets differ, decision 9), or
    the Acclimation of the types that have one.  *metrics* are its headline
    metrics; *types* the ``ParadigmType`` (PyMCU's spelling) of the
    light-table rows it shows; *about* what they are, in a sentence."""

    key: str
    title: str
    metrics: tuple[str, ...]
    types: tuple[str, ...]
    about: str

    def rows(self, frame: pd.DataFrame | None,
             keys: dict[tuple[str | None, int], set[str]] | None = None
             ) -> pd.DataFrame | None:
        """The rows of a light table (:func:`light_summary.behaviour_rows
        <pyflic.base.light_summary.behaviour_rows>`'s) this subsection shows:
        those whose window ran one of its *types*.  A row of several
        Paradigms (``mixed``) shows under each family its DFM's program runs
        (*keys*: ``{(Experiment or None, DFM): light_keys}``, from
        :func:`program_light_keys`).  An Open chamber (lit regardless of its
        fly) is open loop's alone; the Acclimation subsection shows the rows
        with Acclimation columns."""
        if frame is None or frame.empty:
            return frame
        n = len(frame)

        def column(name: str) -> pd.Series:
            return frame[name] if name in frame.columns else pd.Series([pd.NA] * n,
                                                                      index=frame.index)

        if self.key == ACCLIMATION:
            return frame[pd.to_numeric(column("AcclimationActivations"),
                                       errors="coerce").notna().to_numpy()]
        ptype = column("ParadigmType").astype("string").fillna("").str.strip()
        mask = np.array(ptype.isin(self.types), dtype=bool)
        mixed = ptype.eq(_MIXED).to_numpy(dtype=bool)
        if mixed.any() and keys:
            experiments = (frame["Experiment"].astype(str).tolist()
                           if "Experiment" in frame.columns else [None] * n)
            dfms = pd.to_numeric(column("DFM"), errors="coerce").tolist()
            for i in np.flatnonzero(mixed).tolist():
                if pd.notna(dfms[i]) and \
                        self.key in keys.get((experiments[i], int(dfms[i])), set()):
                    mask[i] = True
        open_role = column("Role").astype("string").fillna("").str.strip().eq("Open") \
            .to_numpy(dtype=bool)
        if self.key == pfam.OPEN_LOOP:
            rates = pd.to_numeric(column("LicksPerMinLit"), errors="coerce").notna()
            mask |= open_role | rates.to_numpy(dtype=bool)
        else:
            mask &= ~open_role
        return frame[mask]


#: Every *Light behaviour* subsection, by key.
LIGHT_FAMILIES: dict[str, LightFamily] = {
    pfam.OPEN_LOOP: LightFamily(
        pfam.OPEN_LOOP, "open loop", pfam.FAMILIES[pfam.OPEN_LOOP].headline, ("OpenLoop",),
        "chambers lit whatever their fly does (an OpenLoop Paradigm, or a Well On): "
        "their lit time, and the light feeding index, (lit − dark) / (lit + dark) of "
        "their licks per minute in their own light and in the dark"),
    pfam.CLOSED_LOOP: LightFamily(
        pfam.CLOSED_LOOP, "closed loop", pfam.FAMILIES[pfam.CLOSED_LOOP].headline,
        ("ClosedLoop",),
        "chambers under a ClosedLoop Paradigm, lit by their own contact and its Hold: "
        "Activations, lit time and the median Activation's length"),
    pfam.INTERVAL_BRANCH: LightFamily(
        pfam.INTERVAL_BRANCH, "fixed length, interval",
        pfam.BRANCH_HEADLINE[pfam.INTERVAL_BRANCH], ("FixedInterval",),
        "chambers under a FixedInterval Paradigm: Activations, the reward efficiency "
        "(Main Activations over the most the lockout allows) and the median return "
        "after the lockout"),
    pfam.RATIO_BRANCH: LightFamily(
        pfam.RATIO_BRANCH, "fixed length, ratio", pfam.BRANCH_HEADLINE[pfam.RATIO_BRANCH],
        ("ProgressiveRatio", "FixedRatio"),
        "chambers under a ProgressiveRatio or FixedRatio Paradigm: Activations, the last "
        "Requirement met and the Firmware Events"),
    pfam.NON_FEEDING: LightFamily(
        pfam.NON_FEEDING, "non-feeding", pfam.FAMILIES[pfam.NON_FEEDING].headline,
        ("NonFeeding",),
        "chambers under a NonFeeding Paradigm, dark while their fly feeds: the fraction "
        "of the time lit, and the suppressions, light periods their contact ended"),
    ACCLIMATION: LightFamily(
        ACCLIMATION, "Acclimation (closed loop)", pfam.ACCLIMATION_HEADLINE,
        ("FixedInterval", "ProgressiveRatio", "FixedRatio"),
        "chambers whose Paradigm began with Acclimation: the minute their wells left it"),
}
#: The subsections' order: the families' (decision 9's table), fixed length
#: by its branches, the Acclimation last.
LIGHT_FAMILY_ORDER: tuple[str, ...] = (pfam.OPEN_LOOP, pfam.CLOSED_LOOP, pfam.INTERVAL_BRANCH,
                                       pfam.RATIO_BRANCH, pfam.NON_FEEDING, ACCLIMATION)


def program_light_keys(program, dfm_ids=None) -> set[str]:
    """The *Light behaviour* subsections *program* gives for *dfm_ids*
    (:data:`LIGHT_FAMILY_ORDER` keys): open loop and non-feeding as
    :func:`paradigm_families.families_in
    <pyflic.base.paradigm_families.families_in>` finds them, closed loop for
    a ClosedLoop Paradigm with a Trigger Well, the fixed-length branches
    (:func:`~pyflic.base.paradigm_families.branches_in`) and the Acclimation
    (:func:`~pyflic.base.paradigm_families.acclimation_in`)."""
    if program is None or getattr(program, "grammar", None) == op.GRAMMAR_SIMPLE:
        return set()
    ids = None if dfm_ids is None else [int(d) for d in dfm_ids]
    keys = set(pfam.families_in(program, ids)) & {pfam.OPEN_LOOP, pfam.NON_FEEDING}
    keys |= set(pfam.branches_in(program, ids))
    sections = [program.section(d) for d in (sorted(program.dfms) if ids is None else ids)]
    if any(iv.trigger_wells() and iv.paradigm.type == op.CLOSED_LOOP
           for s in sections if s is not None for iv in s.intervals):
        keys.add(pfam.CLOSED_LOOP)
    if pfam.acclimation_in(program, ids):
        keys.add(ACCLIMATION)
    return keys


def light_families(programs: Iterable[tuple[Any, Iterable[int] | None]]) -> list[LightFamily]:
    """The *Light behaviour* subsections of members running *programs*, each
    an ``(OptoProgram, DFM ids or None)`` pair, in :data:`LIGHT_FAMILY_ORDER`
    (decision 9), each with its headline metrics: closed loop's ``CutOffs``
    only where a ClosedLoop runs with ``MaxOnMs``.  Empty when no program
    runs a family (no program, or a recording made without one): such a
    member keeps the four metrics of before, in one section."""
    keys: set[str] = set()
    max_on = False
    for program, dfm_ids in programs:
        keys |= program_light_keys(program, dfm_ids)
        if program is not None and getattr(program, "grammar", None) != op.GRAMMAR_SIMPLE:
            max_on |= pfam.max_on_in(program, None if dfm_ids is None else list(dfm_ids))
    out: list[LightFamily] = []
    for key in LIGHT_FAMILY_ORDER:
        if key not in keys:
            continue
        family = LIGHT_FAMILIES[key]
        if key == pfam.CLOSED_LOOP:
            if max_on:
                family = LightFamily(family.key, family.title, family.metrics, family.types,
                                     family.about + ", with the Activations its MaxOnMs cut "
                                                    "off")
            else:
                family = LightFamily(family.key, family.title,
                                     tuple(m for m in family.metrics
                                           if m != pfam.MAX_ON_HEADLINE),
                                     family.types, family.about)
        out.append(family)
    return out


def light_dark_plot(frame: pd.DataFrame, *, facet_col: str | None = None):
    """*Feeding in light and dark* (decision 11): each open-loop chamber's
    licks per minute in its own light and in the dark (``LicksPerMinLit``,
    ``LicksPerMinDark``) as two points joined by a line, a panel per
    treatment (and per *facet_col* value, a Facet, when given), with each
    phase's treatment mean as a black diamond.  Raises
    :class:`report_layout.NothingToDraw` when no chamber has both rates."""
    import plotnine as p9

    lit = pd.to_numeric(frame["LicksPerMinLit"], errors="coerce") \
        if "LicksPerMinLit" in frame.columns else pd.Series(dtype=float)
    dark = pd.to_numeric(frame["LicksPerMinDark"], errors="coerce") \
        if "LicksPerMinDark" in frame.columns else pd.Series(dtype=float)
    both = (lit.notna() & dark.notna()) if len(lit) else pd.Series(dtype=bool)
    if not len(both) or not both.any():
        raise rl.NothingToDraw("No open-loop chamber was both lit and dark, so its feeding "
                               "in the light and in the dark cannot be compared.")
    df = frame[both.to_numpy()]
    key_parts = [c for c in ("Experiment", "DFM", "Chamber") if c in df.columns]
    key = df[key_parts].astype(str).agg(" ".join, axis=1) if key_parts else \
        pd.Series(range(len(df)), index=df.index).astype(str)
    panel = (df[facet_col].astype(str) if facet_col and facet_col in df.columns
             else pd.Series(["All"] * len(df), index=df.index))
    phases = ["In the light", "In the dark"]
    long = pd.concat([
        pd.DataFrame({"Chamber": key + " " + panel, "Phase": phases[0],
                      "Rate": lit[both].to_numpy(), "Treatment": df["Treatment"].astype(str),
                      "Panel": panel}),
        pd.DataFrame({"Chamber": key + " " + panel, "Phase": phases[1],
                      "Rate": dark[both].to_numpy(), "Treatment": df["Treatment"].astype(str),
                      "Panel": panel})], ignore_index=True)
    long["Phase"] = pd.Categorical(long["Phase"], categories=phases, ordered=True)
    long["Treatment"] = pd.Categorical(long["Treatment"],
                                       categories=list(dict.fromkeys(long["Treatment"])),
                                       ordered=True)
    long["Panel"] = pd.Categorical(long["Panel"], categories=list(dict.fromkeys(long["Panel"])),
                                   ordered=True)
    g = (p9.ggplot(long, p9.aes("Phase", "Rate", group="Chamber"))
         + p9.geom_line(color="#9a9a9a", size=0.5, alpha=0.8)
         + p9.geom_point(p9.aes(color="Phase"), size=2.0, alpha=0.85)
         + p9.stat_summary(p9.aes(group="Phase"), fun_y=np.mean, geom="point", shape="D",
                           size=3.4, color="black", fill="black")
         + p9.scale_color_manual(values={phases[0]: "#E69F00", phases[1]: "#4D4D4D"},
                                 guide=None)
         + p9.labs(x="", y="Licks per minute")
         + p9.theme_classic(base_size=10.0)
         + p9.theme(strip_background=p9.element_rect(fill="#f0f0f0", color="black",
                                                     size=0.5)))
    if long["Panel"].nunique() > 1:
        g += p9.facet_grid("Panel ~ Treatment")
    else:
        g += p9.facet_wrap("~ Treatment", nrow=1)
    return g


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
