"""Progressive Ratio breaking point and Sucrose Persistence (ADR-0014).

The rule under test is the first-gap rule: responses are read in order from
the group's training end and the fly is taken to have stopped at the first
pause longer than ``pr_break_gap_min``.  With no such pause before the Test
window ends the value is censored — a lower bound.  The synthetic fixtures
are an hour long, shorter than the default 120-minute gap, so the tests that
need to see a fly stop set the gap to five minutes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from pyflic.base import analytics
from pyflic.base import pr_breaking_point as pbp
from pyflic.base.progressive_ratio_experiment import ProgressiveRatioExperiment
from pyflic.base.yaml_config import load_experiment_yaml

from pr_fixtures import make_pr_experiment_dir, make_pr_failure_dir


# ---------------------------------------------------------------------------
# The first-gap rule
# ---------------------------------------------------------------------------

def test_the_count_stops_at_the_first_gap_longer_than_the_threshold():
    r = pbp.first_gap_break(np.array([10.0, 50.0, 90.0, 300.0, 310.0]), 500.0, 120.0)
    assert (r.count, r.last_min, r.censored) == (3, 90.0, False)
    assert r.counted.tolist() == [True, True, True, False, False]


def test_a_gap_of_exactly_the_threshold_does_not_break():
    r = pbp.first_gap_break(np.array([120.0, 240.0]), 360.0, 120.0)
    assert (r.count, r.censored) == (2, True)


def test_the_tail_counts_as_a_gap_so_a_fly_that_stopped_is_observed():
    r = pbp.first_gap_break(np.array([10.0, 20.0]), 500.0, 120.0)
    assert (r.count, r.last_min, r.censored) == (2, 20.0, False)


def test_no_gap_before_the_window_ends_is_censored():
    r = pbp.first_gap_break(np.array([10.0, 100.0, 200.0]), 250.0, 120.0)
    assert (r.count, r.last_min, r.censored) == (3, 200.0, True)


def test_the_first_gap_runs_from_training_end():
    r = pbp.first_gap_break(np.array([130.0, 140.0]), 500.0, 120.0)
    assert (r.count, r.last_min, r.censored) == (0, 0.0, False)


def test_no_responses_is_a_break_at_zero_or_censored_when_the_window_is_short():
    assert pbp.first_gap_break(np.array([]), 500.0, 120.0).censored is False
    short = pbp.first_gap_break(np.array([]), 50.0, 120.0)
    assert (short.count, short.last_min, short.censored) == (0, 0.0, True)


def test_responses_past_the_window_are_ignored_and_censoring_is_judged_there():
    r = pbp.first_gap_break(np.array([10.0, 60.0, 400.0]), 100.0, 120.0)
    assert (r.count, r.censored) == (2, True)
    assert r.counted.tolist() == [True, True, False]


def test_unsorted_input_keeps_its_own_order_in_the_mask():
    r = pbp.first_gap_break(np.array([50.0, 10.0, 400.0]), 500.0, 120.0)
    assert r.count == 2 and r.counted.tolist() == [True, True, False]


def test_a_lick_free_light_event_neither_counts_nor_ends_a_gap():
    minutes = np.array([10.0, 70.0, 140.0])
    ## Counted as a response, the lick-free event at 70 would split the
    ## 130-minute pause into two short ones and let the event at 140 count.
    r = pbp.breaking_point(minutes, np.array([True, False, True]), 500.0, 100.0)
    assert (r.count, r.last_min) == (1, 10.0)
    assert r.counted.tolist() == [True, False, False]


def test_the_real_dataset_group_the_rule_was_designed_on():
    """DFM 1 group 3 of the first real recording: two lick-free events at the
    start, a working ratio through the evening, then a 794-minute pause."""
    minutes = np.array([0.0, 10.7, 37.8, 51.0, 63.6, 107.0, 142.6, 174.4, 191.9, 233.5,
                        254.8, 287.0, 373.6, 398.3, 427.6, 452.0, 511.3, 1305.3])
    backed = np.array([False, False] + [True] * 16)
    got = {gap: pbp.breaking_point(minutes, backed, 1335.2, gap) for gap in (60, 120, 240)}
    assert [got[g].count for g in (60, 120, 240)] == [10, 15, 15]
    assert got[120].last_min == pytest.approx(511.3) and not got[120].censored


def test_persistence_is_the_last_response_before_the_break():
    assert pbp.persistence(np.array([5.0, 30.0, 400.0]), 600.0, 120.0) == (30.0, False)
    assert pbp.persistence(np.array([5.0, 30.0]), 100.0, 120.0) == (30.0, True)


def test_settings_come_from_the_constants():
    s = pbp.BreakSettings.from_constants(None)
    assert (s.gap_min, s.test_window_min) == (120.0, None)
    assert s.describe() == "pr_break_gap_min=120, pr_test_window_min=off"
    s = pbp.BreakSettings.from_constants({"pr_break_gap_min": 45, "pr_test_window_min": 300})
    assert (s.gap_min, s.test_window_min) == (45.0, 300.0)
    assert s.test_end(1000.0) == 300.0 and s.test_end(200.0) == 200.0
    ## 0, blank and nonsense switch the cap off or fall back to the default.
    assert pbp.BreakSettings.from_constants({"pr_test_window_min": 0}).test_window_min is None
    assert pbp.BreakSettings.from_constants({"pr_test_window_min": ""}).test_window_min is None
    assert pbp.BreakSettings.from_constants({"pr_break_gap_min": "x"}).gap_min == 120.0
    assert s.with_gap(60).gap_min == 60.0 and s.with_gap(60).test_window_min == 300.0
    assert pbp.format_count(12, True) == "12+" and pbp.format_count(3, False) == "3"


def test_the_type_carries_the_gap_default():
    from pyflic.base import experiment_types as et

    item = et.get_experiment_type("ProgressiveRatio")
    assert item.default_constants["pr_break_gap_min"] == 120
    assert "pr_test_window_min" not in item.default_constants
    assert "pr_breaking_point.csv" in item.output_manifest()


def test_the_type_validates_its_constants():
    """One description of the constants, so the loader, the linter and both
    editors refuse the same values."""
    from pyflic.base import experiment_types as et

    item = et.get_experiment_type("ProgressiveRatio")
    g = {"experiment_type": "ProgressiveRatio",
         "well_names": {"A": "Sucrose", "B": "Yeast"}}
    assert item.validate({**g, "constants": dict(item.default_constants)}) == []
    ## An unset cap and a constant pyflic knows nothing about are fine.
    assert item.validate({**g, "constants": {"pr_test_window_min": None,
                                             "my_own_note": 3}}) == []
    problems = item.validate({**g, "constants": {
        "pr_break_gap_min": 0, "pr_trend_min_rho": 2, "pr_lick_free_run": 2.5,
        "exclude_failed_pr_groups": "maybe", "pr_resting_level_rise": "x"}})
    assert len(problems) == 5
    joined = "\n".join(problems)
    for key in ("pr_break_gap_min", "pr_trend_min_rho", "pr_lick_free_run",
                "exclude_failed_pr_groups", "pr_resting_level_rise"):
        assert key in joined


def test_a_bad_constant_stops_the_load_and_names_itself(tmp_path):
    root = make_pr_experiment_dir(tmp_path / "exp", constants={"pr_break_gap_min": -5})
    with pytest.raises(ValueError, match="pr_break_gap_min"):
        load_experiment_yaml(root, parallel=False, use_disk_cache=False)


# ---------------------------------------------------------------------------
# The statistics that use the censoring
# ---------------------------------------------------------------------------

def test_kaplan_meier_matches_the_textbook():
    km = analytics.kaplan_meier([1, 2, 2, 3, 4], [True, True, True, False, True])
    assert km["survival"].round(3).tolist() == [0.8, 0.4, 0.4, 0.0]
    assert km["at_risk"].tolist() == [5, 4, 2, 1]


def test_the_still_responding_curve_drops_past_each_break_and_ticks_the_censored():
    frame = pd.DataFrame({
        "Treatment": ["A"] * 4,
        "BreakingPoint": [3, 5, 5, 8],
        "Censored": [False, False, True, False],
    })
    steps, ticks = analytics.still_responding(frame)
    assert list(zip(steps["Ratio"], steps["Fraction"].round(3))) == [
        (0.0, 1.0), (4.0, 0.75), (6.0, 0.5), (9.0, 0.0)]
    assert list(zip(ticks["Ratio"], ticks["Fraction"])) == [(5.0, 0.75)]


def test_logrank_matches_statsmodels():
    from statsmodels.duration.survfunc import survdiff

    rng = np.random.default_rng(7)
    for _ in range(4):
        ta, tb = rng.integers(0, 15, 12), rng.integers(0, 15, 9)
        ea, eb = rng.random(12) < 0.7, rng.random(9) < 0.7
        _chi, expected = survdiff(np.r_[ta, tb], np.r_[ea, eb].astype(int),
                                  np.r_[np.zeros(12), np.ones(9)])
        assert analytics.logrank_p(ta, ea, tb, eb) == pytest.approx(expected)
    assert np.isnan(analytics.logrank_p([3, 4], [False, False], [5], [False]))


def test_breaking_point_comparisons_add_the_logrank_p():
    frame = pd.DataFrame({
        "Treatment": ["A"] * 4 + ["B"] * 4,
        "BreakingPoint": [3, 5, 5, 8, 1, 2, 2, 4],
        "Censored": ["False", "False", "True", "False", "False", "False", "False", "True"],
    })
    rows = analytics.breaking_point_comparisons(frame)
    assert len(rows) == 1 and rows[0]["test"] == "Welch t"
    expected = analytics.logrank_p([3, 5, 5, 8], [1, 1, 0, 1], [1, 2, 2, 4], [1, 1, 1, 0])
    assert rows[0]["p_logrank"] == pytest.approx(expected)


def test_zero_tests_are_the_paired_t_and_the_signed_rank():
    from scipy import stats

    d = pd.DataFrame({"Treatment": ["A"] * 5 + ["B"] * 3 + ["C"] * 2,
                      "dLicksA": [5, 7, 9, 4, 6, -1, 1, 0.5, 2, 2]})
    rows = {r["treatment"]: r for r in analytics.zero_tests([("Test", d)], ["dLicksA"])}
    assert set(rows) == {"A", "B"}                 # C: every difference equal
    a = np.array([5, 7, 9, 4, 6], dtype=float)
    assert rows["A"]["p_t"] == pytest.approx(stats.ttest_1samp(a, 0.0).pvalue)
    assert rows["A"]["p_wilcoxon"] == pytest.approx(stats.wilcoxon(a).pvalue)
    assert rows["A"]["mean"] == pytest.approx(6.2) and rows["A"]["significant"]
    assert rows["A"]["p_mixed"] is None            # one member: no mixed model


def test_the_mixed_intercept_runs_only_across_members():
    calls = []
    d = pd.DataFrame({"Treatment": ["A"] * 6, "Experiment": ["r1"] * 3 + ["r2"] * 3,
                      "DFM": [1, 1, 2, 1, 2, 2], "dPI": [0.2, 0.3, 0.1, 0.25, 0.4, 0.15]})
    rows = analytics.zero_tests([("Test", d)], ["dPI"],
                                mixed_p0=lambda sub: calls.append(len(sub)) or 0.01)
    assert calls == [6] and rows[0]["p_mixed"] == 0.01


def test_csv_booleans_are_read_back():
    got = analytics.as_bool(pd.Series(["True", "False", np.nan, True, 0, 1, None]))
    assert got.tolist() == [True, False, False, True, False, True, False]


# ---------------------------------------------------------------------------
# One experiment: the failure fixture with a five-minute gap
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bp_dir(tmp_path_factory) -> Path:
    return make_pr_failure_dir(tmp_path_factory.mktemp("prbp") / "exp",
                               constants={"pr_break_gap_min": 5})


@pytest.fixture(scope="module")
def bexp(bp_dir) -> ProgressiveRatioExperiment:
    return load_experiment_yaml(bp_dir, parallel=False, use_disk_cache=False)


def _row(frame: pd.DataFrame, dfm: int, group: int) -> pd.Series:
    return frame[(frame.DFM == dfm) & (frame.Group == group)].iloc[0]


def test_the_summary_counts_until_the_fly_stops(bexp):
    s = bexp.breaking_point_summary()
    assert list(s.columns) == ["Treatment", "Genotype", "DFM", "Group", "PairedChamber",
                               "BreakingPoint", "BreakMin", "Censored", "TestMinutes",
                               "LargestRequirement", "LickFreeLightEvents", "LightQC"]
    working, stopped = _row(s, 1, 1), _row(s, 2, 1)
    assert working.BreakingPoint == 12 and not working.Censored
    assert working.BreakMin == pytest.approx(11.59, abs=0.02)
    assert stopped.BreakingPoint == 3 and not stopped.Censored
    ## Every Test light event of the self-triggered group is lick-free.
    sensor = _row(s, 1, 2)
    assert sensor.BreakingPoint == 0 and sensor.LickFreeLightEvents > 0
    assert "self-triggered light" in sensor.LightQC


def test_the_ledger_marks_what_the_breaking_point_counts(bexp):
    ledger = bexp.light_events_ledger()
    assert "Counted" in ledger.columns
    counted = ledger.groupby(["DFM", "Group"])["Counted"].sum()
    s = bexp.breaking_point_summary().set_index(["DFM", "Group"])["BreakingPoint"]
    for key, value in s.items():
        assert counted.get(key, 0) == value
    assert not ledger.loc[ledger["LickFree"], "Counted"].any()


def test_the_sensitivity_table_shows_the_count_at_each_gap(bexp):
    sens = bexp.breaking_point_sensitivity()
    assert {"BP_5", "BP_60", "BP_120", "BP_240"} <= set(sens.columns)
    row = _row(sens, 1, 1)
    assert row.BP_5 == 12 and not row.Censored_5 and bool(row.Censored_120)
    lines = "\n".join(bexp.breaking_point_lines())
    assert "gap 5*" in lines and "12+" in lines and "pr_break_gap_min=5" in lines


def test_the_test_window_cap_limits_the_count_and_the_censoring(tmp_path):
    root = make_pr_failure_dir(tmp_path / "exp", constants={"pr_break_gap_min": 120,
                                                            "pr_test_window_min": 5})
    e = load_experiment_yaml(root, parallel=False, use_disk_cache=False)
    row = _row(e.breaking_point_summary(), 1, 1)
    ## Events at 0.7 … 4.47 minutes fall inside the five-minute window.
    assert row.TestMinutes == 5.0 and row.BreakingPoint == 5 and row.Censored


def test_the_default_gap_censors_an_hour_long_recording(tmp_path):
    root = make_pr_experiment_dir(tmp_path / "exp")
    e = load_experiment_yaml(root, parallel=False, use_disk_cache=False)
    s = e.breaking_point_summary()
    assert not s.empty and s["Censored"].all()
    ## DFM 2 group 3 never finished training: no breaking point at all.
    assert not ((s.DFM == 2) & (s.Group == 3)).any()


def test_persistence_is_on_every_row_but_training_facets(bexp):
    fs = bexp.feeding_summary()
    assert {"PersistA", "PersistACensored"} <= set(fs.columns)
    paired = fs[(fs.DFM == 1) & (fs.Chamber == 1)].iloc[0]
    assert paired.PersistA == pytest.approx(11.41, abs=0.02)
    assert paired.PersistACensored is False
    facet = bexp.feeding_summary_facet()
    training = facet[facet.Facet == "Training"]
    assert training["PersistA"].isna().all() and training["PersistACensored"].isna().all()
    test = facet[facet.Facet == "Test"].set_index(["DFM", "Chamber"])["PersistA"]
    whole = fs.set_index(["DFM", "Chamber"])["PersistA"]
    assert np.allclose(test.sort_index(), whole.loc[test.sort_index().index])


def test_persistence_is_the_rule_on_the_sucrose_feeding_events(bexp):
    dfm = bexp.dfms[1]
    gt = bexp.group_training(1, 1)
    column = f"W{gt.sucrose_well}"
    mins = dfm.event_df["Minutes"].to_numpy(dtype=float)
    onsets = mins[dfm.event_df[column].to_numpy(dtype=float) > 0]
    rel = onsets[onsets > gt.training_end] - gt.training_end
    window = float(dfm.raw_df["Minutes"].max()) - gt.training_end
    assert bexp.chamber_persistence(1, gt.paired_chamber) == pbp.persistence(rel, window, 5.0)


def test_the_difference_table_carries_persistence(bexp):
    diff = bexp.paired_yoked_diff()
    assert {"dPersistA", "dPersistCensored"} <= set(diff.columns)
    assert diff.loc[diff.Facet == "Training", "dPersistA"].isna().all()
    fs = bexp.feeding_summary().set_index(["DFM", "Chamber"])["PersistA"]
    row = _row(diff[diff.Facet == "Test"], 1, 1)
    assert row.dPersistA == pytest.approx(fs[(1, 1)] - fs[(1, 2)])
    assert row.dPersistCensored is False


def test_the_figures_build(bexp):
    from pyflic.base import report_content as rc

    for dfm_id in (1, 2):
        bexp.plot_breaking_point_dfm(dfm_id).draw()
    bexp.plot_still_responding().draw()
    rc.censored_dot_plot(bexp.breaking_point_summary(), "BreakingPoint",
                         y_label="Light events earned").draw()


def test_the_pipeline_writes_the_breaking_point_and_the_summary(bp_dir):
    e = load_experiment_yaml(bp_dir, parallel=False, use_disk_cache=False)
    result = e.execute_basic_analysis(skip_qc=True)
    analysis = bp_dir / "analysis"
    table = pd.read_csv(analysis / "pr_breaking_point.csv")
    ## Auto-removal took the two failed groups out before the table was written.
    assert set(zip(table.DFM, table.Group)) == {(1, 1), (2, 1), (2, 2), (2, 3)}
    assert result["pr_breaking_point"] == (analysis / "pr_breaking_point.csv").resolve()
    assert (analysis / "pr_still_responding.png").is_file()
    text = (analysis / "summary.txt").read_text(encoding="utf-8")
    assert "Progressive ratio breaking point" in text
    assert "Breaking point at other gaps" in text
    ledger = pd.read_csv(analysis / "pr_light_events.csv")
    assert "Counted" in ledger.columns


def test_the_script_actions_run(tmp_path):
    from pyflic.base.script_editor.actions import get_action
    from pyflic.base.script_editor.runner import run_experiment_script

    root = make_pr_failure_dir(tmp_path / "exp", constants={"pr_break_gap_min": 5})
    e = load_experiment_yaml(root, parallel=False, use_disk_cache=False)
    logged: list[str] = []
    figures = run_experiment_script(
        e, {"name": "t", "steps": [{"action": "breaking_point"},
                                   {"action": "plot_pr_still_responding"}]},
        log=logged.append)
    assert (root / "analysis" / "pr_breaking_point.csv").is_file()
    assert (root / "analysis" / "pr_still_responding.png").is_file()
    assert any("Breaking point at other gaps" in line for line in logged)
    assert [title for title, _fig in figures] == ["Still responding"]
    for action in ("breaking_point", "plot_pr_still_responding"):
        assert get_action(action).requires == "progressive_ratio"


def test_the_report_tests_against_zero_and_uses_the_logrank(bp_dir, tmp_path):
    from pyflic.base import report_layout as rl
    from pyflic.base.pdf_report import build_experiment_report

    e = load_experiment_yaml(bp_dir, parallel=False, use_disk_cache=False)
    doc = build_experiment_report(e, metrics=("Licks",))
    tables = [b for b in doc.blocks if isinstance(b, rl.Table)]
    captions = [t.caption or "" for t in tables]
    assert any(c.startswith("Paired − yoked difference against zero") for c in captions)
    bp_stats = next(t for t in tables if t.caption == "Treatment comparisons: breaking point")
    assert "p (log-rank)" in bp_stats.frame.columns
    by_group = next(t for t in tables if (t.caption or "").startswith(
        "Breaking point by chamber group"))
    assert "Breaking point" in by_group.frame.columns
    assert doc.save(tmp_path / "report.pdf").is_file()


# ---------------------------------------------------------------------------
# Project level
# ---------------------------------------------------------------------------

def test_the_project_pools_and_tests_the_breaking_point(tmp_path):
    from pyflic.base.project import Project
    from pyflic.base.project_report import build_project_report

    root = tmp_path / "proj"
    root.mkdir()
    design = None
    for name in ("rep1", "rep2"):
        make_pr_failure_dir(root / name, constants={"pr_break_gap_min": 5})
        cfg = yaml.safe_load((root / name / "flic_config.yaml").read_text())
        design = design or cfg["global"]
        cfg.pop("global")
        (root / name / "flic_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (root / "project.yaml").write_text(yaml.safe_dump(
        {"name": "proj", "design": {"global": design}}, sort_keys=False))

    project = Project(root)
    for name in project.member_names:
        project.load_member(name, parallel=False,
                            use_disk_cache=False).execute_basic_analysis(skip_qc=True)
    written = {Path(p).name for p in project.build_combined_analysis()["written"]}
    assert "proj_BreakingPoint.csv" in written
    pooled = pd.read_csv(root / "analysis" / "proj_BreakingPoint.csv")
    assert set(pooled.Experiment) == {"rep1", "rep2"} and len(pooled) == 8

    rows = project.breaking_point_rows(pooled)
    assert rows and all("p_logrank" in r for r in rows)
    diff = project.combined_diff_frame()
    zero = project.diff_zero_rows(diff)
    assert {r["metric"] for r in zero} >= {"dLicksA", "dPersistA"}
    assert {r["phase"] for r in zero} == {"Test"}

    text = (root / "analysis" / "proj_Stats.txt").read_text(encoding="utf-8")
    assert "Paired − yoked difference against zero" in text
    assert "Breaking point (paired fly; one observation per chamber group)" in text
    assert "p_logrank" in text
    assert text.index("Breaking point (paired fly") < text.index("Per-chamber metrics")

    from pyflic.base import report_layout as rl

    doc = build_project_report(project)
    headings = [b.text for b in doc.blocks if isinstance(b, rl.Heading)]
    assert "Breaking point" in headings
    captions = [b.caption for b in doc.blocks if isinstance(b, rl.Table)]
    assert "Treatment comparisons: breaking point" in captions
    assert "Paired − yoked difference against zero, per treatment" in captions
