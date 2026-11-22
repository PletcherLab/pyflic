"""
Smoke tests for pyflic.base.analytics.

Builds a tiny two-well experiment from synthetic CSV data and exercises
each new analytical helper end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyflic import (
    bootstrap_metric,
    compare_treatments,
    light_phase_summary,
    load_experiment_yaml,
    parameter_sensitivity,
    tidy_events,
)


def _write_yaml(proj: Path) -> None:
    (proj / "data").mkdir(parents=True, exist_ok=True)
    (proj / "flic_config.yaml").write_text(
        """
global:
  params:
    chamber_size: 2
    feeding_threshold: 5
    feeding_minimum: 5
    tasting_minimum: 1
    tasting_maximum: 4
    feeding_event_link_gap: 2
    samples_per_second: 5
    correct_for_dual_feeding: false
  experimental_design_factors:
    Treatment: [Ctrl, Exp]
dfms:
  1:
    params: {pi_direction: left}
    chambers:
      1: Ctrl
      2: Ctrl
      3: Exp
      4: Exp
      5: Ctrl
      6: Exp
""".lstrip()
    )


def _write_synthetic_csv(proj: Path, *, dfm_id: int = 1, n_samples: int = 1500) -> None:
    """Write a CSV with deliberate "feeding events" on every well so that
    feeding_summary returns non-trivial rows."""
    rng = np.random.default_rng(seed=dfm_id)
    cols: dict[str, list[float] | list[int]] = {
        "Sample": list(range(1, n_samples + 1)),
        "Seconds": [i / 5.0 for i in range(n_samples)],
        "OptoCol1": [(1 if (i // 300) % 2 == 0 else 0) for i in range(n_samples)],
    }
    for w in range(1, 13):
        sig = rng.normal(0.0, 0.3, size=n_samples)
        # Insert ~10 "events" per well of 8 high-amplitude samples each
        for k in range(10):
            start = 50 + 100 * k + (w * 7) % 30
            sig[start:start + 8] += 10.0 + rng.normal(0, 0.5, size=8)
        cols[f"W{w}"] = sig.tolist()
    pd.DataFrame(cols).to_csv(proj / "data" / f"DFM{dfm_id}_0.csv", index=False)


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    proj = tmp_path_factory.mktemp("analytics_proj")
    _write_yaml(proj)
    _write_synthetic_csv(proj, dfm_id=1)
    return load_experiment_yaml(proj, parallel=False)


def test_tidy_events_returns_long_dataframe(experiment):
    df = tidy_events(experiment, kind="feeding")
    assert isinstance(df, pd.DataFrame)
    if df.empty:
        pytest.skip("No feeding events in synthetic data — adjust fixture")
    expected = {"DFM", "Chamber", "Well", "Treatment", "StartMin", "Licks", "Duration"}
    assert expected.issubset(df.columns)


def test_bootstrap_metric_runs_and_returns_summary(experiment):
    res = bootstrap_metric(
        experiment, metric="Events", two_well_mode="total",
        n_boot=200, seed=42,
    )
    assert {"n", "mean", "ci_low", "ci_high"}.issubset(res.summary.columns)
    assert (res.summary["ci_high"] >= res.summary["ci_low"]).all()


def test_compare_treatments_aov(experiment):
    res = compare_treatments(
        experiment, metric="MedDuration", two_well_mode="A",
        model="aov", posthoc="tukey",
    )
    assert "term" in res.table.columns
    assert res.n_observations > 0


def test_light_phase_summary_has_two_phases(experiment):
    df = light_phase_summary(experiment)
    assert {"DFM", "Chamber", "Treatment", "Phase"}.issubset(df.columns)
    assert set(df["Phase"]).issubset({"light", "dark"})


def test_light_phase_summary_is_per_chamber(experiment):
    """Decision 24 of docs/v4-data-plan.md: a sample is "light" for a chamber
    when its own wells are lit.  The fixture's OptoCol1 is 1 or 0 (well 1
    only), so chamber 1 alternates and the other five stay dark; before, all
    six were in "light" whenever well 1 was."""
    df = light_phase_summary(experiment)
    light = df[df["Phase"] == "light"].set_index("Chamber")["PhaseSeconds"]
    lit_seconds = experiment.dfms[1].lights_df["W1"].sum() / 5.0
    assert light.loc[1] == pytest.approx(lit_seconds) and lit_seconds > 0
    assert (light.loc[[2, 3, 4, 5, 6]] == 0.0).all()


def test_parameter_sensitivity_grid(experiment):
    res = parameter_sensitivity(
        experiment, parameter="feeding_event_link_gap",
        values=[0, 5, 20],
    )
    assert {"feeding_event_link_gap", "Group", "n_chambers"}.issubset(res.grid.columns)
    # All three metrics should appear as mean/sem columns
    for m in ("Licks", "Events", "MedDuration"):
        assert f"mean_{m}" in res.grid.columns, f"missing mean_{m}"
        assert f"sem_{m}" in res.grid.columns, f"missing sem_{m}"
    # 3 values × at least 1 group → ≥3 rows
    assert len(res.grid) >= 3


def test_yaml_lint_clean_config(tmp_path: Path):
    from pyflic.base.yaml_lint import lint_flic_config
    proj = tmp_path / "p"
    _write_yaml(proj)
    issues = lint_flic_config(proj / "flic_config.yaml")
    errors = [i for i in issues if i.severity == "error"]
    assert errors == []


def test_yaml_lint_catches_bad_pi_direction(tmp_path: Path):
    from pyflic.base.yaml_lint import lint_flic_config
    proj = tmp_path / "bad"
    proj.mkdir()
    (proj / "flic_config.yaml").write_text(
        """
global:
  params: {chamber_size: 2}
dfms:
  1:
    params: {pi_direction: sideways}
    chambers: {1: A}
""".lstrip()
    )
    issues = lint_flic_config(proj / "flic_config.yaml")
    assert any("pi_direction" in i.message for i in issues if i.severity == "error")


def test_disk_cache_round_trip(experiment, tmp_path: Path):
    from pyflic.base import cache as _cache
    df = experiment.feeding_summary()
    p = _cache.save_feeding_summary(
        df, experiment.experiment_dir, range_minutes=(0, 0), transform_licks=True,
    )
    assert p.is_file()
    df2 = _cache.load_feeding_summary(
        experiment.experiment_dir, range_minutes=(0, 0), transform_licks=True,
    )
    assert df2 is not None
    assert len(df2) == len(df)


def test_a_cache_hit_keeps_a_treatment_named_like_a_missing_value(tmp_path: Path):
    """A cached summary is read back from CSV: "None" and "NA" came back as
    missing and "01" as the number 1, so the second load of an unchanged
    experiment gave another summary than the first."""
    proj = tmp_path / "proj"
    _write_yaml(proj)
    cfg = (proj / "flic_config.yaml").read_text()
    cfg = cfg.replace("Treatment: [Ctrl, Exp]", "Drug: [None, NA, '01']")
    for chamber, level in ((1, "None"), (2, "NA"), (3, "'01'"), (4, "None"),
                           (5, "NA"), (6, "'01'")):
        cfg = cfg.replace(f"      {chamber}: Ctrl\n", f"      {chamber}: {level}\n")
        cfg = cfg.replace(f"      {chamber}: Exp\n", f"      {chamber}: {level}\n")
    (proj / "flic_config.yaml").write_text(cfg)
    _write_synthetic_csv(proj, dfm_id=1)
    first = load_experiment_yaml(proj, parallel=False, use_disk_cache=True).feeding_summary()
    second = load_experiment_yaml(proj, parallel=False, use_disk_cache=True).feeding_summary()
    assert sorted(set(first["Treatment"])) == ["01", "NA", "None"]
    pd.testing.assert_frame_equal(first, second, check_dtype=False)
    assert second["Treatment"].tolist() == first["Treatment"].tolist()


def test_pdf_report_writes_file(experiment, tmp_path: Path):
    from pyflic.base.pdf_report import write_experiment_report
    out = tmp_path / "report.pdf"
    p = write_experiment_report(experiment, out, metrics=("Licks",), include_comparison=False)
    assert p.is_file()
    assert p.stat().st_size > 1000


def test_an_open_ended_range_runs_to_the_end_of_the_recording(experiment):
    """``(start, 0)`` is how the tail Facet reaches the loaders (``windowing.
    as_range_minutes``).  It used to be read as a literal end of minute 0 and
    left the last facet of every faceted experiment empty."""
    tail = experiment.feeding_summary(range_minutes=(2, 0), transform_licks=False)
    explicit = experiment.feeding_summary(range_minutes=(2, float("inf")),
                                          transform_licks=False)
    assert tail["LicksA"].sum() > 0
    assert tail["LicksA"].tolist() == explicit["LicksA"].tolist()
    head = experiment.feeding_summary(range_minutes=(0, 2), transform_licks=False)
    whole = experiment.feeding_summary(transform_licks=False)
    assert head["LicksA"].sum() + tail["LicksA"].sum() == pytest.approx(
        whole["LicksA"].sum())


# ---------------------------------------------------------------------------
# The fourth-root transform in sums and differences (review C2, owner decision)
# ---------------------------------------------------------------------------

def test_a_two_well_lick_total_is_the_root_of_the_summed_counts(experiment):
    """Licks summed over the two wells of a transformed summary are
    (A + B)^0.25, the transform of the total, never A^0.25 + B^0.25; Events
    are never transformed and sum as before."""
    from pyflic.base.analytics import _resolve_metric_col, treatment_comparisons
    from pyflic.base.utils import lick_total

    raw = experiment.feeding_summary(transform_licks=False)
    root = experiment.feeding_summary(transform_licks=True)
    expected = (raw["LicksA"] + raw["LicksB"]) ** 0.25
    got = _resolve_metric_col(root, "Licks", transformed=True)
    np.testing.assert_allclose(got.to_numpy(), expected.to_numpy())
    assert not np.allclose(got, root["LicksA"] + root["LicksB"])
    pd.testing.assert_series_equal(_resolve_metric_col(raw, "Licks"),
                                   raw["LicksA"] + raw["LicksB"], check_names=False)
    pd.testing.assert_series_equal(_resolve_metric_col(root, "Events", transformed=True),
                                   root["EventsA"] + root["EventsB"], check_names=False)
    assert lick_total(16.0, 65.0, transformed=False) == 81.0
    assert lick_total(2.0, 0.0, transformed=True) == pytest.approx(2.0)
    rows = treatment_comparisons([("all", root)], ["Licks"], transformed=True)
    means = {r["a"]: r["mean_a"] for r in rows} | {r["b"]: r["mean_b"] for r in rows}
    for treatment, mean in means.items():
        mask = root["Treatment"] == treatment
        assert mean == pytest.approx(float(expected[mask].mean()))


def test_the_binned_and_dot_totals_use_the_root_of_the_sum(experiment):
    binned = experiment._binned_licks_table_by_treatment(binsize_min=2,
                                                         transform_licks=False)
    root = experiment._binned_licks_table_by_treatment(binsize_min=2, transform_licks=True)
    total = experiment._metric_series_from_binned_rows(root, metric="Licks",
                                                       two_well_mode="total",
                                                       transform_licks=True)
    np.testing.assert_allclose(total.to_numpy(),
                               ((binned["LicksA"] + binned["LicksB"]) ** 0.25).to_numpy())
    ## mean_ab and the single wells are unchanged by the flag.
    a = experiment._metric_series_from_binned_rows(root, metric="Licks", two_well_mode="A",
                                                   transform_licks=True)
    pd.testing.assert_series_equal(a, root["LicksA"], check_names=False)
    res = bootstrap_metric(experiment, metric="Licks", two_well_mode="total",
                           transform_licks=True, n_boot=50, seed=1)
    raw = experiment.feeding_summary(transform_licks=False)
    expected = ((raw["LicksA"] + raw["LicksB"]) ** 0.25).groupby(raw["Treatment"]).mean()
    for group, mean in zip(res.summary[res.group_col], res.summary["mean"]):
        assert mean == pytest.approx(float(expected[str(group)]))


def test_a_lick_difference_says_it_is_on_the_root_scale_when_transformed():
    from pyflic.base import report_content as rc

    names = {"A": "Sucrose", "B": "Yeast"}
    assert rc.metric_label("dLicksA", names) == "Δ Sucrose licks (paired − yoked)"
    assert rc.metric_label("dLicksA", names, transformed=True) == \
        "Δ Sucrose licks (¼-root scale, paired − yoked)"
    ## Only licks are transformed: other differences read as before.
    assert rc.metric_label("dEventsA", names, transformed=True) == \
        rc.metric_label("dEventsA", names)
    assert rc.metric_label("dPI", names, transformed=True) == "Δ PI (paired − yoked)"
    rows = [{"metric": "dLicksA", "phase": "Test", "a": "Ctrl", "n_a": 3, "mean_a": 0.1,
             "b": "Exp", "n_b": 3, "mean_b": 0.4, "diff": 0.3, "p_pooled": 0.2,
             "p_mixed": None, "test": "Welch t"}]
    table = rc.stats_table(rows, well_names=names, transformed=True)[0]
    assert "¼-root scale" in table.frame["Measure"].iloc[0]
    zero = [{"metric": "dLicksA", "phase": "Test", "treatment": "Ctrl", "n": 3, "mean": 0.1,
             "sem": 0.05, "p_t": 0.3, "p_wilcoxon": 0.4, "p_mixed": None}]
    assert "¼-root scale" in rc.zero_test_table(zero, well_names=names,
                                               transformed=True)[0].frame["Measure"].iloc[0]
    assert "¼-root" not in rc.zero_test_table(zero, well_names=names)[0].frame["Measure"].iloc[0]


# ---------------------------------------------------------------------------
# No spread within any treatment (Jove Females, light behaviour facets)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("values", [
    {"a": [10.0] * 3, "b": [10.0] * 3},                      # Welch, equal means
    {"a": [0.0] * 3, "b": [10.0] * 3},                       # Welch, unequal means
    {"a": [10.0] * 3, "b": [10.0] * 4, "c": [10.0] * 3},     # Tukey, equal means
    {"a": [0.0] * 3, "b": [10.0] * 4, "c": [5.0] * 3},       # Tukey, unequal means
])
def test_no_spread_within_any_treatment_is_not_tested(values, recwarn):
    """Where every chamber of every treatment has one value (a facet in which
    the light ran the same in every chamber), the tests divide by a zero
    pooled spread: nan for equal means, a spurious p = 0 for unequal ones,
    and a RuntimeWarning in the console.  Nothing is compared and nothing is
    warned."""
    from pyflic.base.analytics import treatment_comparisons

    frame = pd.DataFrame({
        "Treatment": [t for t, v in values.items() for _ in v],
        "Lit": [x for v in values.values() for x in v],
    })
    assert treatment_comparisons([("Test", frame)], ["Lit"]) == []
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_one_treatment_with_spread_is_still_tested():
    """A treatment that varies gives the pooled spread the tests need, even
    when the others are constant."""
    from pyflic.base.analytics import treatment_comparisons

    frame = pd.DataFrame({"Treatment": ["a"] * 3 + ["b"] * 3 + ["c"] * 3,
                          "Lit": [0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 4.0, 5.0, 6.0]})
    rows = treatment_comparisons([("Test", frame)], ["Lit"])
    assert len(rows) == 3 and all(np.isfinite(r["p_pooled"]) for r in rows)
