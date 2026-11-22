"""
Higher-level analytical helpers built on top of ``Experiment``.

These functions add inference, bootstrap CIs, microstructure, light-phase
splits, parameter sensitivity sweeps and tidy long-format export without
touching the core feeding/tasting pipeline.

All helpers are pure functions that take an ``Experiment`` (or a project
directory) so they can be invoked from the Python API, the YAML script
runner (``script_editor.runner`` actions), or the GUI buttons.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple, Sequence

import numpy as np
import pandas as pd

from .dfm import DFM
from .experiment import Experiment
from .single_well_experiment import SingleWellExperiment
from .slices import WHOLE_RECORDING_LABEL
from .two_well_experiment import TwoWellExperiment
from .utils import lick_total, range_bounds


# ---------------------------------------------------------------------------
# Tidy long-format export
# ---------------------------------------------------------------------------

def tidy_events(
    experiment: Experiment,
    *,
    kind: Literal["feeding", "tasting"] = "feeding",
) -> pd.DataFrame:
    """
    Return a long-format DataFrame with one row per bout.

    Columns: ``DFM, Chamber, Well, WellLabel, Treatment, <factors...>,
    StartMin, Licks, Duration, AvgIntensity, MaxIntensity``.

    *kind* selects ``"feeding"`` (DFM.durations) or ``"tasting"``
    (DFM.tasting_durations) bouts.
    """
    rows: list[dict[str, Any]] = []
    chamber_to_treatment: dict[tuple[int, int], str] = {}
    for trt_name, trt in experiment.design.treatments.items():
        for tc in trt.chambers:
            chamber_to_treatment[(int(tc.dfm_id), int(tc.chamber_index))] = trt_name

    for dfm_id, dfm in experiment.dfms.items():
        well_labels = (dfm.well_names or experiment.well_names or {})
        durations = dfm.tasting_durations if kind == "tasting" else dfm.durations
        if not durations:
            continue
        for chamber in dfm.chambers or []:
            ch_idx = int(chamber.index)
            treatment = chamber_to_treatment.get((int(dfm_id), ch_idx), "")
            for well in chamber.wells:
                cname = f"W{int(well)}"
                df = durations.get(cname)
                if not isinstance(df, pd.DataFrame) or df.empty:
                    continue
                # Map well number → 'A'/'B' label for two-well
                well_label = ""
                if hasattr(chamber, "well_a"):
                    well_label = "A" if well == chamber.well_a else (
                        "B" if well == chamber.well_b else ""
                    )
                friendly = well_labels.get(well_label, well_label)
                for _, r in df.iterrows():
                    rows.append({
                        "DFM": int(dfm_id),
                        "Chamber": ch_idx,
                        "Well": int(well),
                        "WellLabel": well_label,
                        "WellName": friendly,
                        "Treatment": treatment,
                        "StartMin": float(r["Minutes"]),
                        "Licks": int(r["Licks"]),
                        "Duration": float(r["Duration"]),
                        "AvgIntensity": float(r.get("AvgIntensity", float("nan"))),
                        "MaxIntensity": float(r.get("MaxIntensity", float("nan"))),
                    })

    out = pd.DataFrame(rows)
    if out.empty or not experiment.design_factors:
        return out

    factor_lookup = experiment.chamber_factors or {}
    for factor in experiment.design_factors:
        out[factor] = [
            factor_lookup.get((int(d), int(c)), {}).get(factor, "")
            for d, c in zip(out["DFM"], out["Chamber"], strict=True)
        ]
    return out


# ---------------------------------------------------------------------------
# Bootstrap CIs
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BootstrapResult:
    metric: str
    group_col: str
    summary: pd.DataFrame  # mean, sem, ci_low, ci_high per group
    samples: pd.DataFrame  # one row per (group, bootstrap_iter)


def bootstrap_metric(
    experiment: Experiment,
    *,
    metric: str = "PI",
    two_well_mode: str = "total",
    n_boot: int = 2000,
    ci: float = 0.95,
    range_minutes: Sequence[float] = (0, 0),
    transform_licks: bool | None = None,
    seed: int | None = 0,
    group_col: str | None = None,
) -> BootstrapResult:
    """
    Bootstrap confidence intervals for *metric* per treatment / factor group.

    Resampling is at the **chamber** level (chambers are independent
    biological units).  PI / EventPI are bounded and skewed, so the
    nonparametric percentile interval is more honest than parametric SE.

    *group_col* defaults to the experiment's resolved group column
    (``"Treatment"`` or ``"_Group"`` when factors are defined).
    """
    df = experiment.feeding_summary(
        range_minutes=range_minutes, transform_licks=transform_licks,
    )
    if df.empty:
        raise ValueError("feeding_summary returned no rows; nothing to bootstrap")

    df, resolved_group = experiment._resolve_group_col(df)
    grp = group_col or resolved_group
    transformed = bool(experiment.transform_licks if transform_licks is None
                       else transform_licks)

    if metric not in df.columns:
        # Try two-well A/B aggregation modes
        a_col, b_col = f"{metric}A", f"{metric}B"
        if a_col in df.columns and b_col in df.columns:
            if two_well_mode == "A":
                df = df.assign(_metric=df[a_col])
            elif two_well_mode == "B":
                df = df.assign(_metric=df[b_col])
            elif two_well_mode == "total":
                df = df.assign(_metric=lick_total(df[a_col], df[b_col],
                                                  transformed=transformed)
                               if metric == "Licks" else df[a_col] + df[b_col])
            elif two_well_mode == "diff":
                df = df.assign(_metric=df[a_col] - df[b_col])
            else:
                raise ValueError(
                    f"two_well_mode {two_well_mode!r} not supported (use A/B/total/diff)"
                )
            value_col = "_metric"
        else:
            raise ValueError(f"metric {metric!r} not present in feeding summary")
    else:
        value_col = metric

    rng = np.random.default_rng(seed)
    samples_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    alpha = (1.0 - float(ci)) / 2.0

    for group, sub in df.groupby(grp):
        vals = sub[value_col].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        n = vals.size
        if n == 0:
            summary_rows.append({
                grp: group, "n": 0, "mean": np.nan, "sem": np.nan,
                "ci_low": np.nan, "ci_high": np.nan,
            })
            continue
        idx = rng.integers(0, n, size=(n_boot, n))
        boot_means = vals[idx].mean(axis=1)
        for i, m in enumerate(boot_means):
            samples_rows.append({grp: group, "iter": i, "mean": float(m)})
        summary_rows.append({
            grp: group,
            "n": int(n),
            "mean": float(vals.mean()),
            "sem": float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
            "ci_low": float(np.quantile(boot_means, alpha)),
            "ci_high": float(np.quantile(boot_means, 1 - alpha)),
        })

    return BootstrapResult(
        metric=metric,
        group_col=grp,
        summary=pd.DataFrame(summary_rows),
        samples=pd.DataFrame(samples_rows),
    )


# ---------------------------------------------------------------------------
# Inference: ANOVA / linear mixed model
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ComparisonResult:
    metric: str
    model: str
    formula: str
    table: pd.DataFrame      # main result table (ANOVA or coefs)
    posthoc: pd.DataFrame | None
    n_observations: int


def compare_treatments(
    experiment: Experiment,
    *,
    metric: str = "MedDuration",
    two_well_mode: str = "A",
    factors: Sequence[str] | None = None,
    model: Literal["aov", "lmm"] = "aov",
    posthoc: Literal["tukey", "none"] = "tukey",
    range_minutes: Sequence[float] = (0, 0),
    transform_licks: bool | None = None,
) -> ComparisonResult:
    """
    Run an ANOVA (default) or linear mixed model comparing *metric* across treatments.

    For two-well experiments, *two_well_mode* selects ``"A"`` / ``"B"`` /
    ``"total"`` / ``"diff"`` when *metric* refers to a per-well column
    (e.g. ``"MedDuration"`` → ``"MedDurationA"``).

    *factors* defaults to the experiment's design_factors when set, otherwise
    a single ``"Treatment"`` term.

    *model="lmm"* fits a mixed-effects model with DFM as a random intercept
    (useful when chambers are nested within DFMs / cohorts).

    Lazy-imports ``statsmodels`` so the rest of pyflic works without it.
    """
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
    except ImportError as exc:
        raise RuntimeError(
            "compare_treatments requires statsmodels. "
            "Install it with: pip install statsmodels"
        ) from exc

    df = experiment.feeding_summary(
        range_minutes=range_minutes, transform_licks=transform_licks,
    )
    if df.empty:
        raise ValueError("feeding_summary is empty; nothing to compare")

    # Normalise metric column for two-well A/B aggregation modes
    transformed = bool(experiment.transform_licks if transform_licks is None
                       else transform_licks)
    if metric not in df.columns:
        a_col, b_col = f"{metric}A", f"{metric}B"
        if a_col in df.columns and b_col in df.columns:
            if two_well_mode == "A":
                df = df.assign(_metric=df[a_col])
            elif two_well_mode == "B":
                df = df.assign(_metric=df[b_col])
            elif two_well_mode == "total":
                df = df.assign(_metric=lick_total(df[a_col], df[b_col],
                                                  transformed=transformed)
                               if metric == "Licks" else df[a_col] + df[b_col])
            elif two_well_mode == "diff":
                df = df.assign(_metric=df[a_col] - df[b_col])
            else:
                raise ValueError(
                    f"two_well_mode {two_well_mode!r} not supported (use A/B/total/diff)"
                )
            value_col = "_metric"
        else:
            raise ValueError(f"metric {metric!r} not present in feeding summary")
    else:
        value_col = metric

    df = df.dropna(subset=[value_col])
    if df.empty:
        raise ValueError(f"No non-NaN observations of {metric!r} after filtering")

    if factors is None:
        factors = experiment.design_factors or ["Treatment"]
    factors = [f for f in factors if f in df.columns]
    if not factors:
        raise ValueError("No usable factor columns present in feeding summary")
    ## A factor with one level has nothing to compare; say so plainly rather
    ## than letting the model fail on an empty constraint matrix.
    varying = [f for f in factors if df[f].nunique(dropna=True) > 1]
    if not varying:
        levels = ", ".join(sorted({str(v) for f in factors for v in df[f].dropna()}))
        raise ValueError(f"Only one treatment ({levels}): there is nothing to compare "
                         f"between treatments.")
    factors = varying

    # Sanitise the metric column name for patsy (no spaces/dots).
    df = df.rename(columns={value_col: "Y"})
    rhs = " * ".join(f"C({f})" for f in factors)
    formula = f"Y ~ {rhs}"

    if model == "aov":
        fit = smf.ols(formula, data=df).fit()
        table = sm.stats.anova_lm(fit, typ=2).reset_index().rename(columns={"index": "term"})
        model_name = "ANOVA (Type II)"
    elif model == "lmm":
        if "DFM" not in df.columns:
            raise ValueError("LMM requires a DFM column for the random intercept")
        fit = smf.mixedlm(formula, data=df, groups=df["DFM"]).fit()
        table = (
            pd.DataFrame(fit.summary().tables[1])
            .reset_index()
            .rename(columns={"index": "term"})
        )
        model_name = "Linear mixed model (DFM random intercept)"
    else:
        raise ValueError(f"model must be 'aov' or 'lmm', got {model!r}")

    posthoc_df: pd.DataFrame | None = None
    if posthoc == "tukey":
        if len(factors) == 1:
            try:
                tuk = pairwise_tukeyhsd(df["Y"].to_numpy(dtype=float), df[factors[0]])
                posthoc_df = pd.DataFrame(
                    data=tuk.summary().data[1:], columns=tuk.summary().data[0]
                )
            except Exception as exc:  # pragma: no cover
                posthoc_df = pd.DataFrame({"warning": [str(exc)]})
        else:
            import warnings
            warnings.warn(
                f"Tukey HSD posthoc is only supported for single-factor models, "
                f"but {len(factors)} factors were specified ({', '.join(factors)}). "
                f"Skipping posthoc — use single-factor subsets or manual contrasts.",
                stacklevel=2,
            )

    return ComparisonResult(
        metric=metric,
        model=model_name,
        formula=formula,
        table=table,
        posthoc=posthoc_df,
        n_observations=int(len(df)),
    )


# ---------------------------------------------------------------------------
# Light-phase summary
# ---------------------------------------------------------------------------

#: The two phases of :func:`light_phase_summary`, in its row order.
LIGHT_PHASES: tuple[str, ...] = ("light", "dark")


class PhaseCounts(NamedTuple):
    """One chamber's samples in one phase, and each of its wells' lick
    samples and feeding-event starts in them, keyed by well (1-12)."""

    samples: int
    licks: dict[int, int]
    events: dict[int, int]


def dfm_lit(dfm: DFM) -> np.ndarray | None:
    """``(n, 12)`` bool: whether each well's LED was lit in each sample, its
    ``OnMs`` above 0 on a generation-3 recording (light from its Linkage
    Group included), else its ``OptoCol1`` bit; ``None`` when the DFM has no
    light columns."""
    from .paradigm_state import GENERATION_PARADIGM

    if dfm.generation == GENERATION_PARADIGM and dfm.paradigm is not None:
        return np.asarray(dfm.paradigm.on_ms) > 0
    lights = dfm.lights_df
    columns = [f"W{w}" for w in range(1, 13)]
    if lights is None or not set(columns) <= set(lights.columns):
        return None
    return lights[columns].fillna(0).to_numpy().astype(bool)


def chamber_light_phases(dfm: DFM, wells: Sequence[int], *, lit: np.ndarray | None = None,
                         rows: np.ndarray | None = None) -> dict[str, PhaseCounts]:
    """:class:`PhaseCounts` of one chamber, its *wells*, keyed by phase
    (:data:`LIGHT_PHASES`): a sample is "light" when one of the chamber's
    own wells is lit (*lit*, default :func:`dfm_lit`), "dark" otherwise.
    Over the sample indices *rows*, default every sample.

    :func:`light_phase_summary` reports these over the whole recording;
    the light table's open-loop rates (``light_summary``) over its
    windows."""
    lit = dfm_lit(dfm) if lit is None else lit
    is_light = lit[:, [int(w) - 1 for w in wells]].any(axis=1)
    if rows is not None:
        is_light = is_light[rows]
    columns = {}
    for well in wells:
        cname = f"W{int(well)}"
        licks = dfm.lick_df[cname].to_numpy(dtype=bool)
        events = dfm.event_df[cname].to_numpy(dtype=int)
        if rows is not None:
            licks, events = licks[rows], events[rows]
        columns[int(well)] = (licks, events)
    out: dict[str, PhaseCounts] = {}
    for phase_name, mask in zip(LIGHT_PHASES, (is_light, ~is_light)):
        out[phase_name] = PhaseCounts(
            samples=int(np.sum(mask)),
            licks={w: int(np.sum(licks & mask)) for w, (licks, _e) in columns.items()},
            events={w: int(np.flatnonzero((events > 0) & mask).size)
                    for w, (_l, events) in columns.items()})
    return out


def light_phase_summary(
    experiment: Experiment,
    *,
    transform_licks: bool | None = None,
) -> pd.DataFrame:
    """
    Per-chamber feeding metrics split by **light vs dark** phase.

    Phase assignment is **per chamber** (decision 24 of
    ``docs/v4-data-plan.md``): a sample is "light" for a chamber when one of
    its own wells is lit — its ``OptoCol1`` bit, or on a generation-3
    recording its ``OnMs`` above 0, light from its Linkage Group included.
    Until firmware 4.0.0 support the phase was the whole DFM's (any
    ``OptoCol1`` bit set), which put every chamber in "light" whenever any
    one was lit.

    Returns one row per (DFM, Chamber, Phase) with Licks, Events
    (and the A/B variants for two-well).  The counts come from
    :func:`chamber_light_phases`, which the light table's open-loop rates
    share (``light_summary``: LicksPerMinLit and LicksPerMinDark).
    """
    if transform_licks is None:
        transform_licks = experiment.transform_licks
    rows: list[dict[str, Any]] = []
    chamber_to_treatment: dict[tuple[int, int], str] = {}
    for trt_name, trt in experiment.design.treatments.items():
        for tc in trt.chambers:
            chamber_to_treatment[(int(tc.dfm_id), int(tc.chamber_index))] = trt_name

    for dfm_id, dfm in experiment.dfms.items():
        lit = dfm_lit(dfm)
        if lit is None:
            continue
        for chamber in dfm.chambers or []:
            ch_idx = int(chamber.index)
            phases = chamber_light_phases(dfm, chamber.wells, lit=lit)
            treatment = chamber_to_treatment.get((int(dfm_id), ch_idx), "")
            for phase_name in LIGHT_PHASES:
                counts = phases[phase_name]
                row: dict[str, Any] = {
                    "DFM": int(dfm_id),
                    "Chamber": ch_idx,
                    "Treatment": treatment,
                    "Phase": phase_name,
                    "PhaseSeconds": float(counts.samples) / float(dfm.params.samples_per_second),
                }
                for well in chamber.wells:
                    n_licks = counts.licks[int(well)]
                    n_events = counts.events[int(well)]
                    if transform_licks:
                        n_licks_disp: float = float(n_licks) ** 0.25
                    else:
                        n_licks_disp = float(n_licks)
                    suffix = ""
                    if hasattr(chamber, "well_a"):
                        suffix = "A" if well == chamber.well_a else "B"
                    row[f"Licks{suffix}"] = n_licks_disp
                    row[f"Events{suffix}"] = n_events
                rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty or not experiment.design_factors:
        return out

    factor_lookup = experiment.chamber_factors or {}
    for factor in experiment.design_factors:
        out[factor] = [
            factor_lookup.get((int(d), int(c)), {}).get(factor, "")
            for d, c in zip(out["DFM"], out["Chamber"], strict=True)
        ]
    return out


# ---------------------------------------------------------------------------
# Parameter sensitivity sweep
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SensitivityResult:
    parameter: str
    metric: str
    grid: pd.DataFrame   # one row per (param_value, treatment) with mean/sem


_SUPPORTED_SWEEP_PARAMS = {
    "feeding_event_link_gap",
    "feeding_threshold",
    "feeding_minimum",
    "tasting_minimum",
    "tasting_maximum",
    "feeding_minevents",
    "tasting_minevents",
    "baseline_window_minutes",
    "samples_per_second",
}


_SENSITIVITY_METRICS = ("Licks", "Events", "MedDuration")


def treatment_comparisons(
    frames: Sequence[tuple[str, pd.DataFrame]],
    metrics: Sequence[str],
    *,
    mixed_p: Any = None,
    transformed: bool = False,
) -> list[dict]:
    """Treatment comparisons for every metric in every ``(label, frame)``.

    Two treatments: Welch's t-test; three or more: Tukey HSD on every pair.
    One observation per row of the frame, so a per-chamber summary compares
    chambers and a Paired-Yoked Difference table compares Chamber Groups.
    Treatments with fewer than two observations are left out; so is a window
    in which no treatment varies within itself (no spread to test against),
    and a pair whose p-value is not finite.

    *mixed_p* ``(data, a, b) -> p | None`` adds the linear-mixed-model p-value
    when the frames span more than one ``Experiment`` (the Project's pooled
    statistics); otherwise ``p_mixed`` is ``None``.  *transformed* says the
    frames' Licks are on the fourth-root scale, so a two-well ``Licks`` total
    is (A + B)^¼ (:func:`_resolve_metric_col`).

    Each row: ``metric, phase, a, n_a, mean_a, b, n_b, mean_b, diff`` (mean
    of *b* minus mean of *a*), ``p_pooled, significant, p_mixed, test``.
    """
    import itertools

    from scipy import stats as sstats

    n_experiments = max(
        [int(f["Experiment"].nunique()) for _, f in frames if "Experiment" in f.columns]
        or [1])
    rows: list[dict] = []
    for metric in metrics:
        for label, frame in frames:
            if frame is None or frame.empty or "Treatment" not in frame.columns:
                continue
            value = pd.to_numeric(_resolve_metric_col(frame, metric, transformed=transformed),
                                  errors="coerce")
            if value.isna().all():
                continue
            data = pd.DataFrame({
                "Treatment": frame["Treatment"].astype(str).str.strip(),
                "Experiment": frame["Experiment"] if "Experiment" in frame.columns else "one",
                "DFM": frame["DFM"] if "DFM" in frame.columns else 0,
                "Value": value,
            }).dropna(subset=["Value"])
            data = data[data["Treatment"] != ""]
            groups = {t: g["Value"].values
                      for t, g in data.groupby("Treatment", sort=False) if len(g) >= 2}
            if len(groups) < 2:
                continue
            ## No treatment varies within itself (every chamber the same in
            ## this window: a facet in which the light ran alike everywhere, a
            ## tail with no feeding).  Both tests divide by that zero pooled
            ## spread, giving nan for equal means, a spurious p = 0 for
            ## unequal ones and a RuntimeWarning either way: nothing to test.
            if all(np.ptp(v) == 0 for v in groups.values()):
                continue
            try:
                if len(groups) == 2:
                    (name_a, va), (name_b, vb) = groups.items()
                    _stat, p = sstats.ttest_ind(va, vb, equal_var=False)
                    pairs = [(name_a, name_b, float(np.mean(vb) - np.mean(va)), float(p))]
                    test = "Welch t"
                else:
                    from statsmodels.stats.multicomp import pairwise_tukeyhsd

                    endog = np.concatenate(list(groups.values()))
                    labels = np.concatenate([[t] * len(v) for t, v in groups.items()])
                    res = pairwise_tukeyhsd(endog=endog, groups=labels, alpha=0.05)
                    pairs = [(str(a), str(b), float(d), float(pv))
                             for (a, b), d, pv in zip(
                                 itertools.combinations(res.groupsunique, 2),
                                 res.meandiffs, res.pvalues)]
                    test = "Tukey HSD"
            except Exception:  # noqa: BLE001
                continue
            for a, b, diff, p_pooled in pairs:
                ## A non-finite p means the groups carry no usable variance in
                ## this window (commonly: a tail with no feeding).  Printing
                ## "nan" as a result invites reading it as one.
                if not np.isfinite(p_pooled):
                    continue
                rows.append({
                    "metric": metric, "phase": label,
                    "a": a, "n_a": len(groups[a]), "mean_a": float(np.mean(groups[a])),
                    "b": b, "n_b": len(groups[b]), "mean_b": float(np.mean(groups[b])),
                    "diff": diff, "p_pooled": p_pooled,
                    "significant": bool(p_pooled < 0.05),
                    "p_mixed": (mixed_p(data, a, b)
                                if mixed_p is not None and n_experiments > 1 else None),
                    "test": test,
                })
    return rows


def _resolve_metric_col(df: pd.DataFrame, metric: str, *,
                        transformed: bool = False) -> pd.Series:
    """Return per-row values for *metric*.

    For single-well DataFrames the column is used directly.  For two-well
    DataFrames (where ``Licks`` is split into ``LicksA`` / ``LicksB``):
    - Licks and Events are summed across wells (A+B); with *transformed*
      (the frame's Licks are on the fourth-root scale, ``transform_licks``)
      the Licks total is (A + B)^¼ (:func:`pyflic.base.utils.lick_total`).
    - MedDuration uses well A only (summing medians is not meaningful).
    """
    if metric in df.columns:
        return df[metric]
    a_col, b_col = f"{metric}A", f"{metric}B"
    if a_col in df.columns and b_col in df.columns:
        if metric == "MedDuration":
            return df[a_col]
        if metric == "Licks":
            return lick_total(df[a_col], df[b_col], transformed=transformed)
        return df[a_col] + df[b_col]
    return pd.Series(float("nan"), index=df.index)


# ---------------------------------------------------------------------------
# Censored counts and within-group differences (Progressive Ratio, ADR-0014)
# ---------------------------------------------------------------------------

def as_bool(values: Any) -> np.ndarray:
    """Truthiness of a column that may have come back from a CSV as text.

    ``True``, ``"true"``, ``"yes"`` and non-zero numbers are True; ``False``,
    ``"false"``, zero, empty and missing are False.
    """
    series = values.astype(object) if isinstance(values, pd.Series) else \
        pd.Series(list(np.atleast_1d(values)), dtype=object)

    def one(v: Any) -> bool:
        if v is None:
            return False
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes", "y", "t")
        try:
            if pd.isna(v):
                return False
        except (TypeError, ValueError):
            pass
        try:
            return bool(v)
        except (TypeError, ValueError):
            return False

    return np.array([one(v) for v in series], dtype=bool)


def kaplan_meier(times: Any, observed: Any) -> pd.DataFrame:
    """Kaplan-Meier estimate of ``S(t) = P(T > t)`` at every distinct time.

    *observed* is False for a censored time.  A censored observation is at
    risk at its own time, the usual convention: a fly censored after *k*
    completed ratios was still attempting ratio *k* + 1 when the recording
    stopped.  Columns: ``time, at_risk, events, censored, survival``.
    """
    t = np.asarray(times, dtype=float)
    e = np.asarray(observed, dtype=bool)
    keep = np.isfinite(t)
    t, e = t[keep], e[keep]
    rows: list[dict] = []
    s = 1.0
    for u in np.unique(t):
        at_risk = int((t >= u).sum())
        here = t == u
        d = int((here & e).sum())
        if at_risk > 0 and d > 0:
            s *= 1.0 - d / at_risk
        rows.append({"time": float(u), "at_risk": at_risk, "events": d,
                     "censored": int((here & ~e).sum()), "survival": s})
    return pd.DataFrame(rows, columns=["time", "at_risk", "events", "censored",
                                       "survival"])


def still_responding(
    frame: pd.DataFrame,
    *,
    value_col: str = "BreakingPoint",
    censored_col: str = "Censored",
    group_col: str = "Treatment",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The "still responding" curve behind the breaking point.

    Per *group_col*, the Kaplan-Meier fraction of paired flies that reached
    ratio *k* — completed *k* Test light events — which is
    ``P(BreakingPoint >= k)`` with a censored count taken as the lower bound
    it is.  Returns ``(steps, ticks)``: *steps* (``group, Ratio, Fraction,
    n``) are the corners of an ``hv`` step curve starting at ``(0, 1)``;
    *ticks* (``group, Ratio, Fraction``) mark each censored fly on its curve.
    """
    step_cols = [group_col, "Ratio", "Fraction", "n"]
    tick_cols = [group_col, "Ratio", "Fraction"]
    if frame is None or frame.empty or value_col not in frame.columns:
        return pd.DataFrame(columns=step_cols), pd.DataFrame(columns=tick_cols)
    data = pd.DataFrame({
        "g": (frame[group_col].astype(str).str.strip() if group_col in frame.columns
              else "All"),
        "t": pd.to_numeric(frame[value_col], errors="coerce"),
        "c": (as_bool(frame[censored_col]) if censored_col in frame.columns
              else np.zeros(len(frame), dtype=bool)),
    }).dropna(subset=["t"])
    steps: list[dict] = []
    ticks: list[dict] = []
    for g, sub in data.groupby("g", sort=False):
        t = sub["t"].to_numpy(dtype=float)
        censored = sub["c"].to_numpy(dtype=bool)
        km = kaplan_meier(t, ~censored)

        def reached(k: float, _km: pd.DataFrame = km) -> float:
            before = _km.loc[_km["time"] <= k - 1, "survival"]
            return float(before.iloc[-1]) if not before.empty else 1.0

        ## A fly that broke after k ratios did not reach ratio k + 1, so its
        ## drop is drawn at k + 1.
        xs, ys = [0.0], [1.0]
        for row in km.itertuples(index=False):
            if row.events > 0:
                xs.append(float(row.time) + 1.0)
                ys.append(float(row.survival))
        x_end = max(float(t.max()), xs[-1])
        if x_end > xs[-1]:
            xs.append(x_end)
            ys.append(ys[-1])
        steps += [{group_col: g, "Ratio": x, "Fraction": y, "n": int(t.size)}
                  for x, y in zip(xs, ys)]
        ticks += [{group_col: g, "Ratio": float(c), "Fraction": reached(float(c))}
                  for c in t[censored]]
    return pd.DataFrame(steps, columns=step_cols), pd.DataFrame(ticks, columns=tick_cols)


def logrank_p(times_a: Any, observed_a: Any, times_b: Any, observed_b: Any) -> float:
    """Two-sample log-rank test (chi-square, 1 df) on possibly tied, censored
    times; ``nan`` when there is no event or no variance to test."""
    from scipy import stats as sstats

    ta, tb = np.asarray(times_a, dtype=float), np.asarray(times_b, dtype=float)
    t = np.concatenate([ta, tb])
    e = np.concatenate([np.asarray(observed_a, dtype=bool),
                        np.asarray(observed_b, dtype=bool)])
    in_a = np.concatenate([np.ones(ta.size, dtype=bool), np.zeros(tb.size, dtype=bool)])
    keep = np.isfinite(t)
    t, e, in_a = t[keep], e[keep], in_a[keep]
    observed_minus_expected, variance = 0.0, 0.0
    for u in np.unique(t[e]):
        at_risk = t >= u
        n = float(at_risk.sum())
        n_a = float((at_risk & in_a).sum())
        died = (t == u) & e
        d = float(died.sum())
        observed_minus_expected += float((died & in_a).sum()) - d * n_a / n
        if n > 1:
            variance += d * (n_a / n) * (1.0 - n_a / n) * (n - d) / (n - 1.0)
    if not variance > 0:
        return float("nan")
    return float(sstats.chi2.sf(observed_minus_expected ** 2 / variance, df=1))


def breaking_point_comparisons(
    frame: pd.DataFrame,
    *,
    phase: str = "Test",
    value_col: str = "BreakingPoint",
    censored_col: str = "Censored",
    mixed_p: Any = None,
) -> list[dict]:
    """Treatment comparisons of the breaking point, one observation per
    Chamber Group.

    The rows of :func:`treatment_comparisons` (Welch's t or Tukey HSD with a
    censored count entered as observed, and the mixed model when *mixed_p* is
    given and the frame spans more than one member) plus ``p_logrank``, the
    pairwise log-rank test on ratio reached, which treats a censored count as
    the lower bound it is.  Unadjusted when there are more than two
    treatments.
    """
    rows = treatment_comparisons([(phase, frame)], [value_col], mixed_p=mixed_p)
    if not rows:
        return rows
    data = pd.DataFrame({
        "Treatment": frame["Treatment"].astype(str).str.strip(),
        "t": pd.to_numeric(frame[value_col], errors="coerce"),
        "c": (as_bool(frame[censored_col]) if censored_col in frame.columns
              else np.zeros(len(frame), dtype=bool)),
    }).dropna(subset=["t"])
    for r in rows:
        a = data[data["Treatment"] == str(r["a"])]
        b = data[data["Treatment"] == str(r["b"])]
        p = logrank_p(a["t"], ~a["c"].to_numpy(dtype=bool),
                      b["t"], ~b["c"].to_numpy(dtype=bool))
        r["p_logrank"] = p if np.isfinite(p) else None
    return rows


def zero_tests(
    frames: Sequence[tuple[str, pd.DataFrame]],
    metrics: Sequence[str],
    *,
    mixed_p0: Any = None,
) -> list[dict]:
    """Is the Paired-Yoked Difference non-zero?  The paired-versus-yoked
    question itself, where :func:`treatment_comparisons` asks whether
    treatments differ in it.

    For every metric in every ``(label, frame)`` and every Treatment: the
    one-sample t-test of the Chamber Groups' differences against zero (the
    paired t-test of paired against yoked) with the Wilcoxon signed-rank test
    beside it.  *mixed_p0* ``(data) -> p | None`` adds the intercept p-value of
    a linear mixed model when the frames span more than one ``Experiment``;
    otherwise ``p_mixed`` is ``None``.  A treatment needs two differences that
    are not all equal.

    Each row: ``metric, phase, treatment, n, mean, sem, p_t, p_wilcoxon,
    p_mixed, significant`` (``p_t < 0.05``), ``test``.
    """
    import warnings

    from scipy import stats as sstats

    n_experiments = max(
        [int(f["Experiment"].nunique()) for _, f in frames
         if f is not None and "Experiment" in f.columns] or [1])
    rows: list[dict] = []
    for metric in metrics:
        for label, frame in frames:
            if frame is None or frame.empty or "Treatment" not in frame.columns \
                    or metric not in frame.columns:
                continue
            data = pd.DataFrame({
                "Treatment": frame["Treatment"].astype(str).str.strip(),
                "Experiment": frame["Experiment"] if "Experiment" in frame.columns else "one",
                "DFM": frame["DFM"] if "DFM" in frame.columns else 0,
                "Value": pd.to_numeric(frame[metric], errors="coerce"),
            }).dropna(subset=["Value"])
            data = data[data["Treatment"] != ""]
            for treatment, sub in data.groupby("Treatment", sort=False):
                v = sub["Value"].to_numpy(dtype=float)
                if v.size < 2 or np.all(v == v[0]):
                    continue
                ## Nearly equal differences make scipy warn about precision;
                ## the p-value it returns is still the one to report.
                with warnings.catch_warnings(), np.errstate(all="ignore"):
                    warnings.simplefilter("ignore")
                    p_t = float(sstats.ttest_1samp(v, 0.0).pvalue)
                    try:
                        p_w = float(sstats.wilcoxon(v).pvalue)
                    except ValueError:
                        p_w = float("nan")
                if not np.isfinite(p_t):
                    continue
                rows.append({
                    "metric": metric, "phase": label, "treatment": str(treatment),
                    "n": int(v.size), "mean": float(v.mean()),
                    "sem": float(v.std(ddof=1) / np.sqrt(v.size)),
                    "p_t": p_t,
                    "p_wilcoxon": p_w if np.isfinite(p_w) else None,
                    "p_mixed": (mixed_p0(sub) if mixed_p0 is not None
                                and n_experiments > 1 else None),
                    "significant": bool(p_t < 0.05),
                    "test": "paired t (one-sample on the differences)",
                })
    return rows


# ---------------------------------------------------------------------------
# The yoked comparison, under every Experiment Type (decision 12 of
# docs/paradigm-analyses-plan.md)
# ---------------------------------------------------------------------------

#: The feeding-summary metrics :func:`yoked_comparison` takes by default, the
#: report's own; per well (``LicksA``, ``LicksB``, ...) on a two-well layout.
YOKED_METRICS: tuple[str, ...] = ("Licks", "Events", "MedDuration")

#: The ``Facet`` of :func:`yoked_comparison`'s whole-recording rows: the
#: Slices module's label, kept under this name for its importers.
WHOLE_RECORDING = WHOLE_RECORDING_LABEL

#: :func:`yoked_comparison`'s leading columns; the design's factors follow
#: ``Treatment``, then ``Paired<m>``, ``Yoked<m>`` and ``d<m>`` per metric.
YOKED_COLUMNS: tuple[str, ...] = ("Facet", "FacetRange", "DFM", "Group", "PairedChamber",
                                  "YokedChambers", "Treatment")


class YokedGroup(NamedTuple):
    """A Linkage Group whose chambers are one Paired chamber and one or more
    Yoked ones (``CONTEXT.md``: Paired Chamber / Yoked Chamber)."""

    dfm: int
    group: int | str           # the Linkage Group's label: the program's linkage number
    paired: int
    yoked: tuple[int, ...]


def yoked_groups(experiment: Experiment) -> list[YokedGroup]:
    """The yoked groups of every DFM the optogenetic light QC covers.

    A Linkage Group is one when exactly one of its chambers holds its
    Trigger Wells and has the Role Paired, and every other chamber it lights
    has the Role Yoked.  The Roles and groups are the light table's
    (:func:`opto_roles.dfm_linkage_view <pyflic.base.opto_roles.dfm_linkage_view>`:
    the program's Linkage, else on generation 3 the data's).  None under an
    Experiment Type that assigns its own Roles (Progressive Ratio keeps its
    own paired-minus-yoked table), and none without a source for the Roles.
    """
    from .light_summary import type_owns_role
    from .opto_roles import ROLE_PAIRED, ROLE_YOKED, chamber_roles, dfm_linkage_view

    if type_owns_role(experiment) or not getattr(experiment, "is_optogenetic", False):
        return []
    program = getattr(experiment, "opto_program", None)
    out: list[YokedGroup] = []
    for dfm_id in experiment.opto.dfm_ids():
        dfm = experiment.dfms[int(dfm_id)]
        chamber_of = {int(w): int(ch.index) for ch in (dfm.chambers or []) for w in ch.wells}
        view = dfm_linkage_view(dfm, program.section(int(dfm_id)) if program is not None else None)
        roles = chamber_roles(view, chamber_of)
        for label, wells in view.groups.items():
            members = sorted({chamber_of[w] for w in wells if w in chamber_of})
            triggers = {chamber_of[w] for w in wells if w in chamber_of and w in view.trigger_wells}
            if len(triggers) != 1 or len(members) < 2:
                continue
            (paired,) = triggers
            yoked = tuple(c for c in members if c != paired)
            if roles.get(paired) != ROLE_PAIRED or any(roles.get(c) != ROLE_YOKED for c in yoked):
                continue
            out.append(YokedGroup(int(dfm_id), int(label) if str(label).isdigit() else str(label),
                                  paired, yoked))
    return out


def yoked_metric_columns(frame: pd.DataFrame, metrics: Sequence[str] | None = None) -> list[str]:
    """The feeding-summary columns *metrics* name (default
    :data:`YOKED_METRICS`): a column of that name, else its two wells
    (``<m>A``, ``<m>B``) as a two-well summary names them.  A metric given
    and found in neither form is an error."""
    out: list[str] = []
    for metric in (YOKED_METRICS if metrics is None else metrics):
        if metric in frame.columns:
            out.append(metric)
        elif f"{metric}A" in frame.columns and f"{metric}B" in frame.columns:
            out += [f"{metric}A", f"{metric}B"]
        elif metrics is not None:
            raise ValueError(f"metric {metric!r} is not in the feeding summary")
    return out


def yoked_comparison(
    experiment: Experiment,
    metrics: Sequence[str] | None = None,
    *,
    range_minutes: Any = (0, 0),
    transform_licks: bool | None = None,
) -> pd.DataFrame:
    """Paired minus yoked, per yoked group (:func:`yoked_groups`) and window,
    from the feeding summary: the comparison Progressive Ratio has, for a
    yoked design under any Experiment Type (decision 12).

    One row per yoked group and window: ``Facet`` is ``Whole recording``
    (over *range_minutes*) and then each Facet, cutoff or Program Facets,
    when the experiment has them (``FacetRange`` its window).  ``DFM``,
    ``Group`` (the Linkage Group), ``PairedChamber``, ``YokedChambers``,
    ``Treatment`` and the design's factors (the paired chamber's), then per
    metric column ``m`` (:func:`yoked_metric_columns`): ``Paired<m>``,
    ``Yoked<m>``, the mean of the yoked chambers' values, and ``d<m>``,
    their difference.  A chamber the design no longer assigns (excluded,
    auto-removed) is left out: a group without its Paired chamber or any
    Yoked one has no row, and the Yoked mean is over the chambers present
    with a value.  Test the differences against zero with :func:`zero_tests`,
    as Progressive Ratio does.  Empty when there is no yoked group."""
    from . import windowing

    groups = yoked_groups(experiment)
    factors = list(getattr(experiment, "design_factors", None) or [])
    summary = experiment.feeding_summary(range_minutes=range_minutes,
                                         transform_licks=transform_licks) if groups else None
    if not groups or summary is None or summary.empty:
        return pd.DataFrame(columns=[*YOKED_COLUMNS, *factors])
    columns = yoked_metric_columns(summary, metrics)
    ## The window as a Facet's FacetRange reads, an open end as inf: never the
    ## "(0, 0)" sentinel.
    whole = (windowing.format_ranges(range_minutes) if windowing.is_window_list(range_minutes)
             else windowing.format_range((0.0, float("inf"))
                                         if windowing.is_full_range(range_minutes)
                                         else range_bounds(range_minutes)))
    ## A Slice's rows are named for it and carry its minutes (ADR-0020).
    from . import slices

    the_slice = getattr(experiment, "slice", None)
    if the_slice is not None and windowing.is_full_range(range_minutes):
        whole = windowing.format_range(the_slice.bounds)
    windows: list[tuple[str, pd.DataFrame]] = [
        (slices.whole_label(the_slice), summary.assign(FacetRange=whole))]
    facet = experiment.feeding_summary_facet(transform_licks=transform_licks)
    if facet is not None and not facet.empty and "Facet" in facet.columns:
        windows += [(label, facet[facet["Facet"].astype(str) == label])
                    for label in dict.fromkeys(facet["Facet"].astype(str))]
    rows: list[dict[str, Any]] = []
    for label, frame in windows:
        at = {(int(d), int(c)): r for (d, c), r in zip(
            zip(pd.to_numeric(frame["DFM"]), pd.to_numeric(frame["Chamber"])),
            frame.to_dict("records"))}
        for g in groups:
            p = at.get((g.dfm, g.paired))
            present = [c for c in g.yoked if (g.dfm, c) in at]
            if p is None or not present:
                continue
            row: dict[str, Any] = {
                "Facet": label, "FacetRange": p.get("FacetRange"), "DFM": g.dfm,
                "Group": g.group, "PairedChamber": g.paired,
                "YokedChambers": ", ".join(str(c) for c in present),
                "Treatment": p["Treatment"], **{f: p.get(f, "") for f in factors}}
            for m in columns:
                paired = pd.to_numeric(p.get(m), errors="coerce")
                values = [pd.to_numeric(at[(g.dfm, c)].get(m), errors="coerce") for c in present]
                values = [float(v) for v in values if pd.notna(v)]
                yoked = float(np.mean(values)) if values else np.nan
                row[f"Paired{m}"] = float(paired) if pd.notna(paired) else np.nan
                row[f"Yoked{m}"] = yoked
                row[f"d{m}"] = (row[f"Paired{m}"] - yoked
                                if pd.notna(row[f"Paired{m}"]) and pd.notna(yoked) else np.nan)
            rows.append(row)
    metric_cols = [f"{kind}{m}" for m in columns for kind in ("Paired", "Yoked", "d")]
    return pd.DataFrame(rows, columns=[*YOKED_COLUMNS, *factors, *metric_cols])


#: The ``analysis/`` file :func:`write_yoked_comparison` writes.
YOKED_COMPARISON_FILE = "yoked_comparison.csv"


def write_yoked_comparison(experiment: Experiment, out_dir: Any = None, *,
                           range_minutes: Any = (0, 0),
                           transform_licks: bool | None = None) -> Path | None:
    """Write :func:`yoked_comparison` to ``analysis/yoked_comparison.csv``
    and return its path; ``None``, writing nothing, when the experiment has
    no yoked group, so an absent file means "no yoked design" (one left by
    an earlier run is removed)."""
    if out_dir is None:
        if experiment.analysis_dir is None:
            raise ValueError("experiment_dir must be set to write the yoked comparison.")
        out_dir = experiment.analysis_dir
    path = Path(out_dir).expanduser().resolve() / YOKED_COMPARISON_FILE
    if not yoked_groups(experiment):
        path.unlink(missing_ok=True)
        return None
    frame = yoked_comparison(experiment, range_minutes=range_minutes,
                             transform_licks=transform_licks)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, na_rep="NA")
    return path


def parameter_sensitivity(
    experiment: Experiment,
    *,
    parameter: str,
    values: Sequence[float],
    range_minutes: Sequence[float] = (0, 0),
    transform_licks: bool | None = None,
) -> SensitivityResult:
    """
    Sweep *parameter* across *values*, recompute each DFM and report
    Licks, Events, and MedDuration per treatment group.

    Output columns (per metric M in {Licks, Events, MedDuration}):
      - ``mean_M`` — mean of per-chamber M across chambers in the group
      - ``sem_M``  — standard error of that mean

    For two-well experiments, Licks and Events are summed across wells
    (A+B) while MedDuration uses well A only.

    This is **expensive** — each value re-runs the per-well feeding/tasting
    pipeline.  Sweep over a handful of values (e.g. 5-10), not hundreds.
    """
    if parameter not in _SUPPORTED_SWEEP_PARAMS:
        raise ValueError(
            f"parameter {parameter!r} is not sweepable; choose one of {sorted(_SUPPORTED_SWEEP_PARAMS)}"
        )

    rows: list[dict[str, Any]] = []
    base_dfms = dict(experiment.dfms)
    transformed = bool(experiment.transform_licks if transform_licks is None
                       else transform_licks)

    for v in values:
        new_dfms: dict[int, DFM] = {}
        for dfm_id, dfm in base_dfms.items():
            new_params = dfm.params.with_updates(**{parameter: type(getattr(dfm.params, parameter))(v)})
            new_dfms[dfm_id] = dfm.with_params(new_params)

        from copy import copy
        tmp_exp = copy(experiment)
        tmp_exp.dfms = new_dfms
        ## A design of its own over the new DFMs: the copy is shallow, and a
        ## serial summary reads the treatments' chambers, not dfms.
        tmp_exp.design = experiment.design.with_dfms(new_dfms)
        tmp_exp._feeding_summary_cache = {}
        df = tmp_exp.feeding_summary(
            range_minutes=range_minutes, transform_licks=transform_licks,
        )
        if df.empty:
            continue
        df, group_col = tmp_exp._resolve_group_col(df)
        for grp, sub in df.groupby(group_col):
            row: dict[str, Any] = {
                parameter: float(v),
                "Group": grp,
                "n_chambers": int(len(sub)),
            }
            for m in _SENSITIVITY_METRICS:
                vals = _resolve_metric_col(sub, m, transformed=transformed).dropna().to_numpy(
                    dtype=float)
                n = vals.size
                row[f"mean_{m}"] = float(vals.mean()) if n else float("nan")
                row[f"sem_{m}"] = float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
            rows.append(row)

    return SensitivityResult(
        parameter=parameter,
        metric="Licks, Events, MedDuration",
        grid=pd.DataFrame(rows),
    )


# ---------------------------------------------------------------------------
# Bout transition matrix (two-well microstructure)
# ---------------------------------------------------------------------------

def bout_transition_matrix(experiment: Experiment) -> pd.DataFrame:
    """
    For two-well experiments, count consecutive bout transitions per chamber.

    Transitions counted: A->A, A->B, B->A, B->B.  Returns a long
    DataFrame with columns: ``DFM, Chamber, Treatment, FromWell, ToWell, Count``.

    Bout starts from both wells are sorted by sample index.  When two
    bouts start at the same sample (simultaneous feeding), well A is
    ordered before well B by convention.
    """
    if not isinstance(experiment, TwoWellExperiment):
        raise ValueError("bout_transition_matrix requires a TwoWellExperiment")

    chamber_to_treatment: dict[tuple[int, int], str] = {}
    for trt_name, trt in experiment.design.treatments.items():
        for tc in trt.chambers:
            chamber_to_treatment[(int(tc.dfm_id), int(tc.chamber_index))] = trt_name

    rows: list[dict[str, Any]] = []
    for dfm_id, dfm in experiment.dfms.items():
        for chamber in dfm.chambers or []:
            if not hasattr(chamber, "well_a"):
                continue
            ch_idx = int(chamber.index)
            treatment = chamber_to_treatment.get((int(dfm_id), ch_idx), "")
            ev_a = dfm.event_df[f"W{chamber.well_a}"].to_numpy(dtype=int)
            ev_b = dfm.event_df[f"W{chamber.well_b}"].to_numpy(dtype=int)
            starts_a = np.flatnonzero(ev_a > 0)
            starts_b = np.flatnonzero(ev_b > 0)
            tagged = sorted(
                [(int(s), "A") for s in starts_a]
                + [(int(s), "B") for s in starts_b]
            )
            counts = {"AA": 0, "AB": 0, "BA": 0, "BB": 0}
            for (_, w0), (_, w1) in zip(tagged, tagged[1:]):
                counts[f"{w0}{w1}"] += 1
            for k, v in counts.items():
                rows.append({
                    "DFM": int(dfm_id),
                    "Chamber": ch_idx,
                    "Treatment": treatment,
                    "FromWell": k[0],
                    "ToWell": k[1],
                    "Count": int(v),
                })

    out = pd.DataFrame(rows)
    if out.empty or not experiment.design_factors:
        return out

    factor_lookup = experiment.chamber_factors or {}
    for factor in experiment.design_factors:
        out[factor] = [
            factor_lookup.get((int(d), int(c)), {}).get(factor, "")
            for d, c in zip(out["DFM"], out["Chamber"], strict=True)
        ]
    return out


# ---------------------------------------------------------------------------
# Side-by-side config diff
# ---------------------------------------------------------------------------

def compare_configs(
    experiment_dir_a: str | Path,
    experiment_dir_b: str | Path,
    *,
    metrics: Sequence[str] = ("Licks", "Events", "MedDuration"),
    two_well_mode: str = "total",
    range_minutes: Sequence[float] = (0, 0),
    transform_licks: bool = True,
) -> pd.DataFrame:
    """
    Load two project directories and compare per-treatment means of
    *metrics* between them.  Returns a DataFrame with one row per
    (Treatment, Metric) and columns ``mean_a, mean_b, delta, pct_change``.
    """
    from .yaml_config import load_experiment_yaml

    exp_a = load_experiment_yaml(experiment_dir_a, range_minutes=range_minutes, parallel=True)
    exp_b = load_experiment_yaml(experiment_dir_b, range_minutes=range_minutes, parallel=True)
    return _compare_two_experiments(
        exp_a, exp_b,
        metrics=metrics, two_well_mode=two_well_mode,
        transform_licks=transform_licks, range_minutes=range_minutes,
    )


def _compare_two_experiments(
    exp_a: Experiment,
    exp_b: Experiment,
    *,
    metrics: Sequence[str],
    two_well_mode: str,
    transform_licks: bool,
    range_minutes: Sequence[float],
) -> pd.DataFrame:
    # Copy to avoid mutating the cached feeding summary DataFrames.
    df_a = exp_a.feeding_summary(range_minutes=range_minutes, transform_licks=transform_licks).copy()
    df_b = exp_b.feeding_summary(range_minutes=range_minutes, transform_licks=transform_licks).copy()
    if df_a.empty or df_b.empty:
        return pd.DataFrame()

    df_a, gcol_a = exp_a._resolve_group_col(df_a)
    df_b, gcol_b = exp_b._resolve_group_col(df_b)

    rows: list[dict[str, Any]] = []
    for metric in metrics:
        for df in (df_a, df_b):
            df[f"_{metric}"] = _resolve_metric_col(df, metric,
                                                   transformed=bool(transform_licks))

        groups = sorted(set(df_a[gcol_a]) | set(df_b[gcol_b]))
        for grp in groups:
            mean_a = float(df_a.loc[df_a[gcol_a] == grp, f"_{metric}"].mean()) if f"_{metric}" in df_a else float("nan")
            mean_b = float(df_b.loc[df_b[gcol_b] == grp, f"_{metric}"].mean()) if f"_{metric}" in df_b else float("nan")
            delta = mean_b - mean_a
            pct = (delta / mean_a * 100.0) if (mean_a not in (0.0, float("nan")) and not np.isnan(mean_a)) else float("nan")
            rows.append({
                "Group": grp,
                "Metric": metric,
                "mean_a": mean_a,
                "mean_b": mean_b,
                "delta": delta,
                "pct_change": pct,
            })
    return pd.DataFrame(rows)
