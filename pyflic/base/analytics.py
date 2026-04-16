"""
Higher-level analytical helpers built on top of ``Experiment``.

These functions add inference, bootstrap CIs, microstructure, light-phase
splits, parameter sensitivity sweeps and tidy long-format export without
touching the core feeding/tasting pipeline.

All helpers are pure functions that take an ``Experiment`` (or a project
directory) so they can be invoked from the Python API, the YAML script
runner (``analysis_hub`` actions), or the GUI buttons.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

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
    transform_licks: bool = True,
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
    transform_licks: bool = True,
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
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

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
    if posthoc == "tukey" and len(factors) == 1:
        try:
            tuk = pairwise_tukeyhsd(df["Y"].to_numpy(dtype=float), df[factors[0]])
            posthoc_df = pd.DataFrame(
                data=tuk.summary().data[1:], columns=tuk.summary().data[0]
            )
        except Exception as exc:  # pragma: no cover
            posthoc_df = pd.DataFrame({"warning": [str(exc)]})

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
    transform_licks: bool = True,
) -> pd.DataFrame:
    """
    Per-chamber feeding metrics split by **light vs dark** phase.

    A sample is "light" iff ``OptoCol1 != 0`` in the DFM's ``lights_df``.
    Returns one row per (DFM, Chamber, Phase) with Licks, Events,
    MedDuration (and the A/B variants for two-well).
    """
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


def parameter_sensitivity(
    experiment: Experiment,
    *,
    parameter: str,
    values: Sequence[float],
    metric: str = "Events",
    two_well_mode: str = "total",
    range_minutes: Sequence[float] = (0, 0),
    transform_licks: bool = True,
) -> SensitivityResult:
    """
    Sweep *parameter* across *values*, recompute each DFM and aggregate
    *metric* per treatment.

    This is **expensive** — each value re-runs the per-well feeding/tasting
    pipeline.  Sweep over a handful of values (e.g. 5–10), not hundreds.
    """
    if parameter not in _SUPPORTED_SWEEP_PARAMS:
        raise ValueError(
            f"parameter {parameter!r} is not sweepable; choose one of {sorted(_SUPPORTED_SWEEP_PARAMS)}"
        )

    rows: list[dict[str, Any]] = []
    base_dfms = dict(experiment.dfms)

    for v in values:
        # Build per-DFM with overridden parameter
        new_dfms: dict[int, DFM] = {}
        for dfm_id, dfm in base_dfms.items():
            new_params = dfm.params.with_updates(**{parameter: type(getattr(dfm.params, parameter))(v)})
            new_dfms[dfm_id] = dfm.with_params(new_params)

        # Build a temporary experiment with the same design but new DFMs
        # so cached methods aren't reused.
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
                    raise ValueError(f"two_well_mode {two_well_mode!r}")
                value_col = "_metric"
            else:
                raise ValueError(f"metric {metric!r} not present")
        else:
            value_col = metric
        for grp, sub in df.groupby(group_col):
            v_arr = sub[value_col].dropna().to_numpy(dtype=float)
            n = v_arr.size
            rows.append({
                parameter: float(v),
                "Group": grp,
                "n": int(n),
                "mean": float(v_arr.mean()) if n else float("nan"),
                "sem": float(v_arr.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
            })

    return SensitivityResult(
        parameter=parameter,
        metric=metric,
        grid=pd.DataFrame(rows),
    )


# ---------------------------------------------------------------------------
# Bout transition matrix (two-well microstructure)
# ---------------------------------------------------------------------------

def bout_transition_matrix(experiment: Experiment) -> pd.DataFrame:
    """
    For two-well experiments, count **A→A, A→B, B→A, B→B** consecutive bout
    transitions per chamber.  Returns a long DataFrame with columns:
    ``DFM, Chamber, Treatment, FromWell, ToWell, Count``.
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
    project_dir_a: str | Path,
    project_dir_b: str | Path,
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

    exp_a = load_experiment_yaml(project_dir_a, range_minutes=range_minutes, parallel=True)
    exp_b = load_experiment_yaml(project_dir_b, range_minutes=range_minutes, parallel=True)
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
    df_a = exp_a.feeding_summary(range_minutes=range_minutes, transform_licks=transform_licks)
    df_b = exp_b.feeding_summary(range_minutes=range_minutes, transform_licks=transform_licks)
    if df_a.empty or df_b.empty:
        return pd.DataFrame()

    df_a, gcol_a = exp_a._resolve_group_col(df_a)
    df_b, gcol_b = exp_b._resolve_group_col(df_b)

    rows: list[dict[str, Any]] = []
    for metric in metrics:
        for df, label, gcol in ((df_a, "a", gcol_a), (df_b, "b", gcol_b)):
            if metric in df.columns:
                df[f"_{metric}"] = df[metric]
            else:
                a_col, b_col = f"{metric}A", f"{metric}B"
                if a_col in df.columns and b_col in df.columns:
                    if two_well_mode == "total":
                        df[f"_{metric}"] = df[a_col] + df[b_col]
                    elif two_well_mode == "A":
                        df[f"_{metric}"] = df[a_col]
                    elif two_well_mode == "B":
                        df[f"_{metric}"] = df[b_col]
                    elif two_well_mode == "diff":
                        df[f"_{metric}"] = df[a_col] - df[b_col]

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
