"""Progressive Ratio light QC: did the paired fly earn its light?

The failure this guards against was seen in real data: a Sucrose Well whose
resting level creeps up looks continuously touched to the firmware, which
then fires the light on its own clock, while the baselined signal — the one
pyflic counts licks from — stays flat.  The fixtures reproduce it
(``pr_fixtures.make_pr_failure_dir``) beside a working progressive ratio.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from pyflic.base import pr_light_qc as lqc
from pyflic.base.progressive_ratio_experiment import ProgressiveRatioExperiment
from pyflic.base.yaml_config import load_experiment_yaml

from pr_fixtures import make_pr_experiment_dir, make_pr_failure_dir


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------

def test_light_events_are_onsets_and_first_dark_samples():
    light = np.array([0, 1, 1, 0, 0, 1, 0, 1], dtype=bool)
    onsets, ends = lqc.light_events(light)
    assert onsets.tolist() == [1, 5, 7]
    assert ends.tolist() == [3, 6, 8]          # still lit at the end -> len


def test_licks_are_credited_from_the_previous_event_end_to_this_one():
    licks = np.zeros(20, dtype=bool)
    licks[[2, 3]] = True        # before event 1
    licks[6] = True             # on event 1's first lit sample: still event 1's
    licks[12] = True            # between events 1 and 2
    ends = np.array([8, 15, 19])
    assert lqc.licks_between_events(licks, ends).tolist() == [3, 1, 0]


def test_runs():
    mask = np.array([1, 0, 1, 1, 1, 0, 1, 1], dtype=bool)
    assert lqc.longest_run(mask) == (3, 2)
    assert lqc.first_run_start(mask, 2) == 2
    assert lqc.first_run_start(mask, 4) is None
    assert lqc.longest_run(np.zeros(4, dtype=bool)) == (0, None)


def test_trend_is_rank_based_and_undefined_for_a_constant_series():
    rho, slope = lqc.lick_trend(np.array([0, 5, 10, 15, 21, 25]))
    assert rho == pytest.approx(1.0) and slope == pytest.approx(5.0, abs=0.2)
    rho, slope = lqc.lick_trend(np.array([8, 8, 8, 8, 8]))
    assert np.isnan(rho) and slope == pytest.approx(0.0)


def test_resting_summary_uses_30_minute_windows():
    minutes = np.arange(120)
    level = pd.Series(np.where(minutes < 60, 10.0, 40.0), index=minutes)
    level.iloc[5] = 500.0                       # one minute on the well is not a level
    s = lqc.resting_summary(level)
    assert s["start"] == pytest.approx(10.0)
    assert s["end"] == pytest.approx(40.0)
    assert s["max"] == pytest.approx(40.0)


def _judge(**overrides):
    kwargs = dict(training_complete=True, training_light_events=9,
                  training_licks=5, test_counts=np.arange(1, 13) * 5,
                  resting={"start": 10.0, "max": 12.0, "end": 11.0, "level": 11.0},
                  reference_level=12.0, any_light=True)
    kwargs.update(overrides)
    return lqc.judge_group(lqc.LightQCSettings(), **kwargs)


def test_a_working_group_is_ok():
    v = _judge()
    assert v.verdict == lqc.VERDICT_OK and v.flags == ()


def test_training_fails_only_on_zero_licks_not_on_fewer_licks_than_pairings():
    ## Brief touches trigger the firmware but fall under pyflic's feeding
    ## threshold; 5 lick samples over 9 pairings is a real, healthy training.
    assert lqc.IMPLAUSIBLE_TRAINING not in _judge(training_licks=5).flags
    v = _judge(training_licks=0)
    assert lqc.IMPLAUSIBLE_TRAINING in v.flags and v.failed


def test_isolated_lick_free_events_are_not_a_failure_but_a_run_is():
    counts = np.arange(1, 13) * 5
    counts[[3, 7]] = 0
    assert lqc.SELF_TRIGGERED not in _judge(test_counts=counts).flags
    counts[4:9] = 0
    v = _judge(test_counts=counts)
    assert lqc.SELF_TRIGGERED in v.flags and v.lick_free_run_start == 3


def test_a_falling_trend_warns_and_too_few_events_is_only_a_note():
    v = _judge(test_counts=np.array([60, 50, 40, 30, 20, 10]))
    assert v.flags == (lqc.NO_TREND,) and v.verdict == lqc.VERDICT_WARNING
    v = _judge(test_counts=np.array([5, 10, 15]))
    assert v.flags == () and "needed for a trend verdict" in v.notes[0]
    v = _judge(test_counts=np.array([], dtype=int))
    assert v.flags == () and "breaking point" in v.notes[0]


def test_resting_level_warnings_need_a_margin_as_well_as_a_ratio():
    v = _judge(resting={"start": 10.0, "max": 40.0, "end": 38.0, "level": 20.0})
    assert lqc.RESTING_RISE in v.flags and not v.failed
    ## 12 counts on a DFM resting at 2 is four times the reference, and noise.
    quiet = _judge(resting={"start": 12.0, "max": 12.0, "end": 12.0, "level": 12.0},
                   reference_level=2.0)
    assert lqc.RESTING_ELEVATED not in quiet.flags
    high = _judge(resting={"start": 90.0, "max": 95.0, "end": 90.0, "level": 90.0},
                  reference_level=12.0)
    assert lqc.RESTING_ELEVATED in high.flags and not high.failed


def test_settings_come_from_the_constants():
    s = lqc.LightQCSettings.from_constants({"exclude_failed_pr_groups": "false",
                                            "pr_lick_free_run": 8})
    assert s.exclude is False and s.lick_free_run == 8
    assert s.trend_min_rho == pytest.approx(0.3)


def test_increment_is_the_median_over_rising_groups():
    rising = np.arange(1, 9) * 5.0
    flat = np.zeros(8)
    short = np.array([5.0, 10.0])
    slope, _intercept, n = lqc.estimate_increment([rising, rising + 1, flat, short])
    assert n == 2 and slope == pytest.approx(5.0)
    assert lqc.estimate_increment([flat, short]) is None


# ---------------------------------------------------------------------------
# One experiment: a working ratio beside the two failures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fail_dir(tmp_path_factory) -> Path:
    return make_pr_failure_dir(tmp_path_factory.mktemp("prqc") / "exp")


@pytest.fixture(scope="module")
def fexp(fail_dir) -> ProgressiveRatioExperiment:
    return load_experiment_yaml(fail_dir, parallel=False, use_disk_cache=False)


def _row(table: pd.DataFrame, dfm: int, group: int) -> pd.Series:
    return table[(table.DFM == dfm) & (table.Group == group)].iloc[0]


def test_the_light_qc_table_finds_each_failure(fexp):
    t = fexp.light_qc_table()
    assert len(t) == 6
    ok = _row(t, 1, 1)
    assert ok.Verdict == "ok" and ok.Flags == "" and ok.TrendRho == pytest.approx(1.0)
    sensor = _row(t, 1, 2)
    assert sensor.Verdict == "failed" and bool(sensor.Excluded)
    assert "self-triggered light" in sensor.Flags
    assert "resting level rise" in sensor.Flags
    assert sensor.LongestLickFreeRun == sensor.TestLightEvents
    assert sensor.LickFreeRunStartMin == pytest.approx(0.34, abs=0.05)
    training = _row(t, 1, 3)
    assert training.TrainingLicks == 0 and training.TrainingLightEvents == 9
    assert "implausible training" in training.Flags
    assert "resting level elevated" in training.Flags
    stopped = _row(t, 2, 1)
    assert stopped.Verdict == "ok" and stopped.TestLightEvents == 3
    assert "needed for a trend verdict" in stopped.Notes


def test_the_ledger_is_the_paired_breaking_point_table(fexp):
    bp = fexp.breaking_point_table(1, 1)
    assert bp["LicksSincePrev"].tolist()[:4] == [8, 12, 16, 20]
    assert not bp["LickFree"].any()
    assert fexp.breaking_point_table(1, 3)["LickFree"].all()
    ledger = fexp.light_events_ledger()
    assert list(ledger.columns[:4]) == ["DFM", "Group", "PairedChamber", "Event"]
    assert len(ledger) == 12 + 69 + 71 + 3 + 12 + 12
    slope, _intercept, n = fexp.estimated_increment()
    assert slope == pytest.approx(4.0) and n == 3


def test_every_summary_row_carries_the_group_verdict(fexp):
    fs = fexp.feeding_summary()
    flagged = fs[(fs.DFM == 1) & (fs.Group == 2)]
    assert len(flagged) == 2 and flagged["LightQC"].str.contains("self-triggered").all()
    assert (fs[(fs.DFM == 1) & (fs.Group == 1)]["LightQC"] == "").all()
    diff = fexp.paired_yoked_diff()
    assert {"LightQC", "LickFreeLightEvents"}.issubset(diff.columns)


def test_a_cached_summary_gets_fresh_light_qc_columns(fexp):
    """The disk cache stores the augmented frame, keyed on the member's
    config only; the light QC thresholds live in the design, so the cached
    columns are never trusted."""
    stale = fexp.feeding_summary().copy()
    stale["LightQC"] = "stale"
    fresh = fexp._augment_rows(stale)
    assert "stale" not in set(fresh["LightQC"])
    assert list(fresh.columns) == list(stale.columns)


def test_failed_groups_leave_through_auto_removal(fail_dir):
    e = load_experiment_yaml(fail_dir, parallel=False, use_disk_cache=False)
    removed = e.auto_remove_chambers()
    keys = set(zip(removed.DFM.astype(int), removed.Chamber.astype(int)))
    assert keys == {(1, 3), (1, 4), (1, 5), (1, 6)}
    ## Chamber 5's sucrose well recorded no licks at all, so the ordinary lick
    ## cutoff takes it first; its partner leaves on the light QC.
    reasons = removed.set_index("Chamber")["Reason"]
    assert "min_untransformed_licks_cutoff" in reasons[5]
    for chamber in (3, 4, 6):
        assert "exclude_failed_pr_groups" in reasons[chamber]
    ## The QC table and figures keep the removed groups in view...
    assert len(e.light_qc_table()) == 6
    traces = e.cumulative_curve_data(qc=True)
    assert set(traces[traces.DFM == 1]["Group"]) == {1, 2, 3}
    e.plot_cumulative_licks_dfm(1).draw()
    ## ...the results do not.
    diff = e.cumulative_diff_data()
    assert set(zip(diff.DFM, diff.Group)) == {(1, 1), (2, 1), (2, 2), (2, 3)}


def test_exclusion_can_be_switched_off(tmp_path):
    root = make_pr_failure_dir(tmp_path / "exp",
                               constants={"exclude_failed_pr_groups": False})
    e = load_experiment_yaml(root, parallel=False, use_disk_cache=False)
    removed = e.auto_remove_chambers()
    ## Only the ordinary lick cutoff acts (chamber 5 recorded no licks).
    assert not removed.Reason.str.contains("exclude_failed_pr_groups").any()
    t = e.light_qc_table()
    assert _row(t, 1, 2).Verdict == "failed" and not bool(_row(t, 1, 2).Excluded)
    assert "retained" in e.summary_text(include_qc=False)


def test_the_thresholds_are_the_designs(tmp_path):
    root = make_pr_failure_dir(tmp_path / "exp",
                               constants={"pr_lick_free_run": 1000})
    e = load_experiment_yaml(root, parallel=False, use_disk_cache=False)
    assert "self-triggered light" not in _row(e.light_qc_table(), 1, 2).Flags


def test_the_pipeline_writes_and_applies_the_light_qc(tmp_path):
    root = make_pr_failure_dir(tmp_path / "exp")
    e = load_experiment_yaml(root, parallel=False, use_disk_cache=False)
    e.execute_basic_analysis(skip_qc=True)
    analysis = root / "analysis"
    for name in ("pr_light_qc.csv", "pr_light_events.csv", "removed_chambers.csv",
                 "pr_light_events_dfm1.png", "pr_resting_level_dfm1.png",
                 "pr_cumulative_licks_dfm1.png"):
        assert (analysis / name).is_file(), name
    qc = pd.read_csv(analysis / "pr_light_qc.csv")
    assert len(qc) == 6 and int(qc["Excluded"].sum()) == 2
    removed = pd.read_csv(analysis / "removed_chambers.csv")
    assert len(removed) == 4
    fs = pd.read_csv(analysis / "feeding_summary.csv")
    assert not ((fs.DFM == 1) & fs.Group.isin([2, 3])).any()
    text = (analysis / "summary.txt").read_text(encoding="utf-8")
    assert "Progressive ratio light QC" in text
    assert "EXCLUDED (exclude_failed_pr_groups)" in text
    assert "Chambers removed by auto_remove_chambers" in text
    ## A second run on the same object keeps the first run's record instead of
    ## finding nothing left to remove and overwriting it with an empty table.
    e.execute_basic_analysis(skip_qc=True)
    assert len(pd.read_csv(analysis / "removed_chambers.csv")) == 4


def test_the_qc_figures_build(fexp):
    for dfm_id in (1, 2):
        fexp.plot_light_events_dfm(dfm_id).draw()
        fexp.plot_resting_level_dfm(dfm_id).draw()
        fexp.plot_cumulative_licks_dfm(dfm_id).draw()


def test_the_report_carries_the_light_qc_and_the_breaking_point(fail_dir, tmp_path):
    from pyflic.base import report_layout as rl
    from pyflic.base.pdf_report import build_experiment_report

    e = load_experiment_yaml(fail_dir, parallel=False, use_disk_cache=False)
    doc = build_experiment_report(e, metrics=("Licks",))
    headings = [b.text for b in doc.blocks if isinstance(b, rl.Heading)]
    for wanted in ("Quality control", "Progressive ratio: training",
                   "Progressive ratio: light QC", "Results",
                   "Cumulative difference curve",
                   "Paired − yoked difference, Test phase", "Breaking point"):
        assert wanted in headings, wanted
    assert headings.index("Progressive ratio: light QC") < headings.index("Results")
    glance = [b for b in doc.blocks if isinstance(b, rl.Callout)
              and (b.title or "").startswith("Light QC")]
    assert glance and glance[0].tone == "failed"
    ## Auto-removal ran for the report, as basic analysis would have.
    assert e.filtered_chambers is not None and len(e.filtered_chambers) == 4
    path = doc.save(tmp_path / "report.pdf")
    assert path.is_file() and path.stat().st_size > 0


def test_the_breaking_point_counts_lick_backed_light_events(fexp):
    summary = fexp.breaking_point_summary()
    ## The working ratio earned 12, the fly that stopped 3.  An hour-long
    ## recording is shorter than the default 120-minute gap, so no group can
    ## be seen to stop: every count is censored, a lower bound (ADR-0014).
    got = {(int(r.DFM), int(r.Group)): int(r.BreakingPoint) for r in summary.itertuples()}
    assert got[(1, 1)] == 12 and got[(2, 1)] == 3
    assert summary["Censored"].all()
    ## The self-triggered group's light events are all lick-free: none counts.
    assert got[(1, 2)] == 0
    row = summary[(summary.DFM == 1) & (summary.Group == 1)].iloc[0]
    assert row.LargestRequirement == 12 * 4 + 4
    assert row.TestMinutes == pytest.approx(54.0, abs=0.1)


def test_the_script_actions_run(tmp_path):
    from pyflic.base.script_editor.actions import get_action
    from pyflic.base.script_editor.runner import run_experiment_script

    root = make_pr_failure_dir(tmp_path / "exp")
    e = load_experiment_yaml(root, parallel=False, use_disk_cache=False)
    logged: list[str] = []
    figures = run_experiment_script(
        e, {"name": "t", "steps": [{"action": "pr_light_qc"},
                                   {"action": "plot_pr_light_events"},
                                   {"action": "plot_pr_resting_level"}]},
        log=logged.append)
    assert (root / "analysis" / "pr_light_qc.csv").is_file()
    assert any("self-triggered light" in line for line in logged)
    assert [title for title, _fig in figures] == [
        "Licks per light event — DFM 1", "Licks per light event — DFM 2",
        "Sucrose Well resting level — DFM 1", "Sucrose Well resting level — DFM 2"]
    for action in ("pr_light_qc", "plot_pr_light_events", "plot_pr_resting_level"):
        assert get_action(action).requires == "progressive_ratio"


def test_a_parameter_recompute_keeps_the_training_flags(fexp):
    dfm = fexp.dfms[1]
    again = dfm.with_params(dfm.params)
    for well in range(1, 13):
        assert again.training_flag_state(well) == dfm.training_flag_state(well)
    assert again.training_end_minutes(1) == pytest.approx(dfm.training_end_minutes(1))


def test_the_standard_fixture_is_not_failed(tmp_path):
    """The default fixture's light follows its paired fly throughout, so the
    light QC must not take anything out of it."""
    root = make_pr_experiment_dir(tmp_path / "exp", incomplete_group_on_dfm2=False)
    e = load_experiment_yaml(root, parallel=False, use_disk_cache=False)
    t = e.light_qc_table()
    assert not (t["Verdict"] == "failed").any()
    assert e.auto_remove_chambers().empty


# ---------------------------------------------------------------------------
# Project level
# ---------------------------------------------------------------------------

def test_the_project_lists_flagged_groups(tmp_path):
    from pyflic.base.project import Project
    from pyflic.base.project_report import write_project_report

    root = tmp_path / "proj"
    root.mkdir()
    design = None
    for name, maker in (("rep1", make_pr_failure_dir), ("rep2", make_pr_experiment_dir)):
        maker(root / name)
        cfg = yaml.safe_load((root / name / "flic_config.yaml").read_text())
        g = cfg.pop("global")
        design = design or g
        (root / name / "flic_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (root / "project.yaml").write_text(yaml.safe_dump(
        {"name": "proj", "design": {"global": design}}, sort_keys=False))

    project = Project(root)
    for name in project.member_names:
        project.load_member(name, parallel=False,
                            use_disk_cache=False).execute_basic_analysis(skip_qc=True)
    written = {Path(p).name for p in project.build_combined_analysis()["written"]}
    assert "proj_LightQC.csv" in written
    flagged = project.flagged_groups()
    failed = flagged[flagged.Verdict == "failed"]
    assert set(zip(failed.Experiment, failed.DFM, failed.Group)) == {
        ("rep1", 1, 2), ("rep1", 1, 3)}
    assert set(failed.Status) == {"excluded"}
    text = (root / "analysis" / "proj_Stats.txt").read_text(encoding="utf-8")
    assert "Flagged Chamber Groups" in text
    assert text.index("Flagged Chamber Groups") < text.index("Paired − yoked difference")
    excluded = pd.read_csv(root / "analysis" / "proj_Excluded.csv")
    ## Four chambers leave rep1; one on the lick cutoff, three on the light QC.
    assert (excluded.Experiment == "rep1").sum() == 4
    assert excluded.Note.str.contains("exclude_failed_pr_groups").sum() == 3
    assert Path(write_project_report(project, log=lambda *_a, **_k: None)).is_file()


def test_a_retained_failure_is_labelled_as_in_the_pooled_numbers():
    from pyflic.base.project import flagged_light_qc_groups

    frame = pd.DataFrame({
        "Experiment": ["a", "a", "b"], "DFM": [1, 1, 2], "Group": [1, 2, 3],
        "Treatment": ["x", "y", "x"], "Verdict": ["failed", "warning", "failed"],
        "Flags": ["self-triggered light", "resting level rise", "implausible training"],
        "Excluded": [True, False, False], "LickFreeRunStartMin": [12.0, np.nan, np.nan],
    })
    status = flagged_light_qc_groups(frame).set_index("Group")["Status"]
    assert status[1] == "excluded"
    assert status[2] == "kept — warning only"
    assert status[3].startswith("retained")
