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
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from .dfm import DFM
from .experiment import Experiment
from .single_well_experiment import SingleWellExperiment
from .two_well_experiment import TwoWellExperiment


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

    if metric not in df.columns:
        # Try two-well A/B aggregation modes
        a_col, b_col = f"{metric}A", f"{metric}B"
        if a_col in df.columns and b_col in df.columns:
            if two_well_mode == "A":
                df = df.assign(_metric=df[a_col])
            elif two_well_mode == "B":
                df = df.assign(_metric=df[b_col])
            elif two_well_mode == "total":
                df = df.assign(_metric=df[a_col] + df[b_col])
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
    if metric not in df.columns:
        a_col, b_col = f"{metric}A", f"{metric}B"
        if a_col in df.columns and b_col in df.columns:
            if two_well_mode == "A":
                df = df.assign(_metric=df[a_col])
            elif two_well_mode == "B":
                df = df.assign(_metric=df[b_col])
            elif two_well_mode == "total":
                df = df.assign(_metric=df[a_col] + df[b_col])
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

def light_phase_summary(
    experiment: Experiment,
    *,
    transform_licks: bool | None = None,
) -> pd.DataFrame:
    """
    Per-chamber feeding metrics split by **light vs dark** phase.

    Phase assignment is **global per DFM row**: a sample is classified as
    "light" iff ``OptoCol1 != 0`` at that time point.  This treats the
    opto signal as a room-level (or DFM-level) synchronization flag, not
    a per-well bitmask.  If your hardware encodes independent per-well
    light states, decode ``lights_df["W{n}"]`` columns instead.

    Returns one row per (DFM, Chamber, Phase) with Licks, Events
    (and the A/B variants for two-well).
    """
    if transform_licks is None:
        transform_licks = experiment.transform_licks
    rows: list[dict[str, Any]] = []
    chamber_to_treatment: dict[tuple[int, int], str] = {}
    for trt_name, trt in experiment.design.treatments.items():
        for tc in trt.chambers:
            chamber_to_treatment[(int(tc.dfm_id), int(tc.chamber_index))] = trt_name

    for dfm_id, dfm in experiment.dfms.items():
        lights = dfm.lights_df
        opto = lights["OptoCol1"].fillna(0).to_numpy() if "OptoCol1" in lights.columns else None
        if opto is None:
            continue
        is_light = opto.astype(int) != 0
        for chamber in dfm.chambers or []:
            ch_idx = int(chamber.index)
            treatment = chamber_to_treatment.get((int(dfm_id), ch_idx), "")
            for phase_name, mask in (("light", is_light), ("dark", ~is_light)):
                row: dict[str, Any] = {
                    "DFM": int(dfm_id),
                    "Chamber": ch_idx,
                    "Treatment": treatment,
                    "Phase": phase_name,
                    "PhaseSeconds": float(np.sum(mask)) / float(dfm.params.samples_per_second),
                }
                for well in chamber.wells:
                    cname = f"W{int(well)}"
                    licks = dfm.lick_df[cname].to_numpy(dtype=bool)
                    events = dfm.event_df[cname].to_numpy(dtype=int)
                    n_licks = int(np.sum(licks & mask))
                    starts = np.flatnonzero((events > 0) & mask)
                    n_events = int(starts.size)
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
) -> list[dict]:
    """Treatment comparisons for every metric in every ``(label, frame)``.

    Two treatments: Welch's t-test; three or more: Tukey HSD on every pair.
    One observation per row of the frame, so a per-chamber summary compares
    chambers and a Paired-Yoked Difference table compares Chamber Groups.
    Treatments with fewer than two observations are left out, and so is a
    pair whose p-value is not finite (no usable variance in that window).

    *mixed_p* ``(data, a, b) -> p | None`` adds the linear-mixed-model p-value
    when the frames span more than one ``Experiment`` (the Project's pooled
    statistics); otherwise ``p_mixed`` is ``None``.

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
            value = pd.to_numeric(_resolve_metric_col(frame, metric), errors="coerce")
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


def _resolve_metric_col(df: pd.DataFrame, metric: str) -> pd.Series:
    """Return per-row values for *metric*.

    For single-well DataFrames the column is used directly.  For two-well
    DataFrames (where ``Licks`` is split into ``LicksA`` / ``LicksB``):
    - Licks and Events are summed across wells (A+B).
    - MedDuration uses well A only (summing medians is not meaningful).
    """
    if metric in df.columns:
        return df[metric]
    a_col, b_col = f"{metric}A", f"{metric}B"
    if a_col in df.columns and b_col in df.columns:
        if metric == "MedDuration":
            return df[a_col]
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

    for v in values:
        new_dfms: dict[int, DFM] = {}
        for dfm_id, dfm in base_dfms.items():
            new_params = dfm.params.with_updates(**{parameter: type(getattr(dfm.params, parameter))(v)})
            new_dfms[dfm_id] = dfm.with_params(new_params)

        from copy import copy
        tmp_exp = copy(experiment)
        tmp_exp.dfms = new_dfms
        tmp_exp.design.dfms = dict(new_dfms)
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
                vals = _resolve_metric_col(sub, m).dropna().to_numpy(dtype=float)
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
            df[f"_{metric}"] = _resolve_metric_col(df, metric)

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
