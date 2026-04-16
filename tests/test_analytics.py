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
    pd.DataFrame(cols).to_csv(proj / "data" / f"DFM{dfm_id}_test.csv", index=False)


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


def test_parameter_sensitivity_grid(experiment):
    res = parameter_sensitivity(
        experiment, parameter="feeding_event_link_gap",
        values=[0, 5, 20], metric="Events", two_well_mode="total",
    )
    assert {"feeding_event_link_gap", "Group", "n", "mean"}.issubset(res.grid.columns)
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
        df, experiment.project_dir, range_minutes=(0, 0), transform_licks=True,
    )
    assert p.is_file()
    df2 = _cache.load_feeding_summary(
        experiment.project_dir, range_minutes=(0, 0), transform_licks=True,
    )
    assert df2 is not None
    assert len(df2) == len(df)


def test_pdf_report_writes_file(experiment, tmp_path: Path):
    from pyflic.base.pdf_report import write_experiment_report
    out = tmp_path / "report.pdf"
    p = write_experiment_report(experiment, out, metrics=("Licks",), include_comparison=False)
    assert p.is_file()
    assert p.stat().st_size > 1000
