"""Progressive Ratio on recorded state (docs/v4-data-plan.md, stream 4).

A generation-3 recording from the firmware 4.0.0 simulator (``v4_fixtures``)
analysed as a Progressive Ratio experiment: Training from the Paired Sucrose
Well's Phase (decision 11), Light Events from its Activations with the
Requirement each met and the Firmware Events it took (decision 13), the Test
window ending with the ratio Paradigm's run (decision 12), FixedRatio
(decision 14), and a generation-2 recording, which has no Training at all
(decision 15).  Generation 1 is covered, unchanged, by
``test_progressive_ratio.py``, ``test_pr_light_qc.py`` and
``test_pr_breaking_point.py``.

``choice_pr_spec`` acts out ``CHOICE_PR_CASES`` on both DFMs; the configs
here make DFM 1's paired chambers 1, 3, 5 (healthy, drifting,
never-training) and DFM 2's 2, 4, 6 (healthy, dead LED, healthy).
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from pyflic.base import light_summary as ls
from pyflic.base import opto_program as op
from pyflic.base import pr_light_qc as lqc
from pyflic.base import report_layout as rl
from pyflic.base.experiment_types.progressive_ratio import break_requirement_view
from pyflic.base.paradigm_state import ParadigmState
from pyflic.base.progressive_ratio_experiment import (
    EMPTY_TRAINING_NOTE,
    NO_TRAINING_INFORMATION,
    ProgressiveRatioExperiment,
)
from pyflic.base.yaml_config import load_experiment_yaml

from pr_fixtures import TRAINED_PR, pr_config, write_trained_pr_type
from v4_fixtures import (
    CHOICE_LINKAGE,
    CHOICE_PR_CASES,
    CHOICE_THRESHOLDS,
    OFF,
    PR10,
    IntervalSpec,
    choice_pr_spec,
    family_specs,
    fixed_ratio_spec,
    program_text,
    simulate,
    write_gen2_recording,
    write_v4_recording,
    yoked_controls_spec,
)

RECORD_MIN = 0.2 / 60.0
PAIRED = {1: [1, 3, 5], 2: [2, 4, 6]}
TREATMENTS = {1: "Ctrl", 2: "Ctrl", 3: "Exp", 4: "Exp", 5: "Third", 6: "Third"}


def write_config(root: Path, dfm_ids, *, constants: dict | None = None) -> None:
    """A Progressive Ratio config under :data:`TRAINED_PR`, copied beside it:
    the choice recording's program runs a ClosedLoop Training interval before
    PR10, which the shipped type does not expect (ADR-0019)."""
    cfg = pr_config([{"id": int(d), "paired_chambers": PAIRED.get(int(d), [1, 3, 5]),
                      "chambers": dict(TREATMENTS)} for d in dfm_ids],
                    constants=constants)
    cfg["global"]["experiment_type"] = TRAINED_PR
    (root / "flic_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    write_trained_pr_type(root)


def pr_dir(root: Path, spec, results, *, writer=write_v4_recording, constants=None,
           **kw) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    writer(root / "data", spec, results=results, **kw)
    write_config(root, [d.dfm_id for d in spec.dfms], constants=constants)
    return root


def load(root: Path) -> ProgressiveRatioExperiment:
    e = load_experiment_yaml(root, parallel=False, use_disk_cache=False)
    assert isinstance(e, ProgressiveRatioExperiment)
    return e


@lru_cache(maxsize=None)
def choice():
    spec = choice_pr_spec()
    return spec, simulate(spec)


@lru_cache(maxsize=None)
def fixed():
    spec = fixed_ratio_spec(dfm_ids=(1,))
    return spec, simulate(spec)


@pytest.fixture(scope="module")
def gen3(tmp_path_factory):
    spec, results = choice()
    return load(pr_dir(tmp_path_factory.mktemp("gen3"), spec, results)), results


@pytest.fixture(scope="module")
def gen2(tmp_path_factory):
    spec, results = choice()
    return load(pr_dir(tmp_path_factory.mktemp("gen2"), spec, results,
                       writer=write_gen2_recording)), results


@pytest.fixture(scope="module")
def fr(tmp_path_factory):
    spec, results = fixed()
    return load(pr_dir(tmp_path_factory.mktemp("fr"), spec, results)), results


def case_of(dfm_id: int, group: int) -> str:
    return CHOICE_PR_CASES[next(c for c in PAIRED[dfm_id] if (c + 1) // 2 == group)]


def groups_of(case: str) -> list[tuple[int, int]]:
    return [(d, g) for d in PAIRED for g in (1, 2, 3) if case_of(d, g) == case]


# ---------------------------------------------------------------------------
# Training (decision 11)
# ---------------------------------------------------------------------------

def test_each_groups_training_ends_where_its_acclimation_ends(gen3):
    e, results = gen3
    table = e.training_table().set_index(["DFM", "Group"])
    for (dfm_id, group), row in table.iterrows():
        truth = results[dfm_id].truth
        end_ms = truth.acclimation_end_ms[int(row["SucroseWell"])]
        if case_of(dfm_id, group) in ("never_training", "dead_led"):
            assert end_ms is None
            assert row["TrainingComplete"] == False  # noqa: E712
            assert pd.isna(row["TrainingEndMin"])
            assert "training never completed" in row["Notes"]
            continue
        ## The last Training row is the one before the first Main row under PR10.
        minutes = e.dfms[dfm_id].raw_df["Minutes"].to_numpy()
        assert row["TrainingComplete"] == True  # noqa: E712
        assert row["TrainingEndMin"] == minutes[end_ms // 200 - 1]
    ## Healthy and drifting both left Acclimation; never-training stayed in A.
    assert table.loc[(1, 3), "Notes"] == ("paired Sucrose Well W9 was still in Acclimation "
                                          "(Phase A) when the recording ended — training "
                                          "never completed")


def test_training_notes_speak_of_phase_and_never_of_flags(gen3):
    e, _results = gen3
    ## The three other wells of a group are Off and read M: nothing to note.
    for dfm_id, group in groups_of("healthy"):
        assert e.group_training(dfm_id, group).notes == ()
    notes = e.training_warnings()
    assert notes and not any("flag" in n.lower() for n in notes)
    summary = e.summary_text(include_qc=False)
    assert "Training notes (QC, not errors)" in summary
    assert "Training-flag notes" not in summary


def test_an_empty_training_is_complete_and_said_so(tmp_path):
    spec, _ = fixed()
    spec = replace(spec, paradigms={"FR5": replace(spec.paradigms["FR5"], acclimation=0)})
    e = load(pr_dir(tmp_path, spec, simulate(spec)))
    dfm = e.dfms[1]
    for group in (1, 2, 3):
        gt = e.group_training(1, group)
        assert dfm.training_flag_state(gt.sucrose_well) == "never_flagged"
        assert gt.complete and gt.training_complete is True
        assert gt.notes == (EMPTY_TRAINING_NOTE,)
    assert e.training_table()["TrainingComplete"].all()
    assert not e.auto_remove_chambers()["Reason"].str.contains("training").any()


def test_a_later_ratio_run_restarts_and_ends_the_test_window(tmp_path):
    """Closed, PR10, Closed, PR10: the second PR10 resets every Well (a new
    name came between), so it is a restart in Acclimation, and the Test
    window is the first PR10's run only."""
    spec = choice_pr_spec(dfm_ids=(1,))
    intervals = tuple(IntervalSpec(name, True, CHOICE_THRESHOLDS, minutes)
                      for name, minutes in (("Closed", 5.0), ("PR10", 12.0),
                                            ("Closed", 5.0), ("PR10", 8.0)))
    spec = replace(spec, dfms=(replace(spec.dfms[0], intervals=intervals),))
    results = simulate(spec)
    e = load(pr_dir(tmp_path, spec, results))
    state = e.dfms[1].paradigm
    gt = e.group_training(1, 1)                      # healthy, paired chamber 1
    assert gt.complete
    assert "Paradigm restarted at Program Interval 4 (after the Test window, which is " \
           "the first ratio run only)" in gt.notes
    ## The window ends with the last row of Interval 2, not the recording.
    last = np.flatnonzero(state.program_interval.to_numpy(dtype=np.int64, na_value=-1) == 2)[-1]
    window = e._test_window(1, 1, e.break_settings())
    assert window == pytest.approx(float(state.minutes[last]) - gt.training_end)
    assert window < e._recording_end(e.dfms[1]) - gt.training_end - 10.0
    row = e.breaking_point_summary().set_index(["DFM", "Group"]).loc[(1, 1)]
    assert row["TestMinutes"] == pytest.approx(window)
    ## The cap still applies after.
    capped = e.break_settings().__class__(gap_min=120.0, test_window_min=3.0)
    assert e._test_window(1, 1, capped) == 3.0


# ---------------------------------------------------------------------------
# Light Events: the Paired Sucrose Well's Activations (decision 13)
# ---------------------------------------------------------------------------

def test_light_events_are_the_sucrose_wells_activations(gen3):
    e, results = gen3
    for dfm_id in PAIRED:
        truth = results[dfm_id].truth
        minutes = e.dfms[dfm_id].raw_df["Minutes"].to_numpy()
        for group in (1, 2, 3):
            gt = e.group_training(dfm_id, group)
            events = e.light_events_table(dfm_id, group)
            onsets = truth.activation_onset_ms[gt.sucrose_well]
            assert len(events) == len(onsets)
            assert np.array_equal(events["RecordingMinute"].to_numpy(),
                                  minutes[[t // 200 for t in onsets]])
            ## An Acclimation Activation is a Training Light Event, in Phase A.
            phases = truth.activation_phase[gt.sucrose_well]
            assert events["Phase"].tolist() == phases
            training = events[events["Facet"] == "Training"]
            assert (events.loc[events["Phase"] == "A", "Facet"] == "Training").all()
            if gt.complete:
                assert int((training["Phase"] == "A").sum()) == PR10.acclimation


def test_the_ledger_records_the_requirement_each_light_event_met(gen3):
    e, results = gen3
    ledger = e.light_events_ledger()
    series = [PR10.first_requirement + PR10.step * k for k in range(40)]
    for dfm_id, group in groups_of("healthy"):
        rows = ledger[(ledger.DFM == dfm_id) & (ledger.Group == group)]
        assert len(rows) >= 3
        assert rows["Requirement"].tolist() == series[:len(rows)]       # 1, 11, 21, ...
        assert rows["FirmwareEvents"].tolist() == rows["Requirement"].tolist()
        assert (rows["Phase"] == "M").all()
    ## Against the engine itself, every Main Activation of every group.
    for dfm_id in PAIRED:
        truth = results[dfm_id].truth
        for group in (1, 2, 3):
            gt = e.group_training(dfm_id, group)
            events = e.light_events_table(dfm_id, group)
            got = [None if pd.isna(v) else int(v) for v in events["Requirement"]]
            assert got == truth.requirement_met[gt.sucrose_well]
            main = events[(events["Phase"] == "M") & events["Requirement"].notna()]
            assert (main["FirmwareEvents"] == main["Requirement"]).all()


def test_the_ledger_csv_carries_the_firmware_columns(gen3, tmp_path):
    e, _results = gen3
    out = e.write_light_qc()
    ledger = pd.read_csv(out["pr_light_events"])
    assert list(ledger.columns[-4:]) == ["Requirement", "FirmwareEvents", "Phase", "Counted"]
    assert ledger["Requirement"].dropna().astype(int).min() == 1
    qc = pd.read_csv(out["pr_light_qc"])
    assert "TrendRhoFirmware" in qc.columns
    assert qc["TrendRhoFirmware"].notna().any()


def test_a_requirement_no_row_shows_comes_from_the_program():
    """An Activation on the first row of a new Paradigm met a Requirement no
    row shows (the row before belongs to the old Paradigm); the program's
    series gives it.  An Acclimation one meets none."""
    n = 10
    frame = pd.DataFrame({
        "Interval": [1] * 5 + [2] * 5,
        "Paradigm": ["Closed"] * 5 + ["PRX"] * 5,
        "Phase1": ["M"] * n,
        "Activations1": [0, 0, 0, 0, 0, 1, 1, 1, 2, 2],
        "Requirement1": [np.nan] * 5 + [3, 3, 3, 5, 5],
        "Progress1": [np.nan] * 5 + [0, 1, 2, 0, 0],
        "OnMs1": [0, 0, 0, 0, 0, 200, 0, 0, 200, 0],
    })
    state = ParadigmState.from_frame(frame, minutes=np.arange(n) * RECORD_MIN)
    dfm = SimpleNamespace(generation=3, paradigm=state)
    prx = op.Paradigm(name="PRX", type=op.PROGRESSIVE_RATIO, frequency_hz=40,
                      pulse_width_ms=8, activation_ms=500, series=op.SERIES_ARITHMETIC,
                      first_requirement=1, step=2)
    onsets = state.activation_onsets(1)
    assert onsets.tolist() == [5, 8]

    def columns(program):
        host = SimpleNamespace(opto_program=program,
                               _reads_phase=ProgressiveRatioExperiment._reads_phase)
        return ProgressiveRatioExperiment._firmware_columns(host, dfm, 1, onsets)

    with_program = columns(SimpleNamespace(paradigms={"PRX": prx}))
    assert list(with_program["Requirement"]) == [1, 3]
    assert list(with_program["FirmwareEvents"])[1] == 3
    assert list(with_program["Phase"]) == ["M", "M"]
    without = columns(None)
    assert pd.isna(without["Requirement"][0]) and without["Requirement"][1] == 3


# ---------------------------------------------------------------------------
# Breaking Point (ADR-0014 addendum)
# ---------------------------------------------------------------------------

def test_the_breaking_point_reports_the_requirement_it_reached(gen3):
    e, _results = gen3
    summary = e.breaking_point_summary().set_index(["DFM", "Group"])
    ledger = e.light_events_ledger()
    assert len(summary)
    for (dfm_id, group), row in summary.iterrows():
        counted = ledger[(ledger.DFM == dfm_id) & (ledger.Group == group) & ledger.Counted]
        assert row["BreakingPoint"] == len(counted)
        assert row["BreakRequirement"] == counted["Requirement"].iloc[-1]
        assert row["LargestRequirement"] == row["BreakRequirement"]
        assert row["FirmwareEvents"] == counted["FirmwareEvents"].sum()
    for dfm_id, group in groups_of("healthy"):
        n = int(summary.loc[(dfm_id, group), "BreakingPoint"])
        assert summary.loc[(dfm_id, group), "BreakRequirement"] == 1 + 10 * (n - 1)
    ## The text and the report put it beside the Breaking Point.
    lines = "\n".join(e.breaking_point_lines())
    assert "BreakingPoint BreakReq" in lines
    blocks = e.report_results_blocks([], SimpleNamespace(include_comparison=False))
    tables = [b for b in blocks if isinstance(b, rl.Table) and "Breaking point" in b.frame.columns]
    columns = list(tables[0].frame.columns)
    assert columns[columns.index("Breaking point") + 1] == "Break requirement"
    assert any(isinstance(b, rl.Heading) and b.text == "Breaking point" for b in blocks)


def test_the_report_says_a_transformed_lick_difference_is_on_the_root_scale(tmp_path):
    """Review C2 (owner decision): dLicksA stays paired^0.25 - yoked^0.25
    under transform_licks, and the report's labels and text say so; without
    the transform they say nothing of it."""
    spec, results = choice()
    e = load(pr_dir(tmp_path / "exp", spec, results))

    def lick_measures(transformed: bool) -> tuple[set[str], str]:
        e.transform_licks = transformed
        blocks = e.report_results_blocks([], SimpleNamespace(include_comparison=True))
        measures = {m for b in blocks if isinstance(b, rl.Table) and "Measure" in b.frame
                    for m in b.frame["Measure"] if "licks" in str(m) and "Δ" in str(m)}
        text = " ".join(b.text for b in blocks if isinstance(b, rl.Paragraph))
        return measures, text

    measures, text = lick_measures(True)
    assert measures and all("¼-root scale" in m for m in measures)
    assert "Lick differences are on the ¼-root scale" in text
    diff = e.paired_yoked_diff()
    facet = e.feeding_summary_facet()
    row = diff.dropna(subset=["dLicksA"]).iloc[0]
    rows = facet[(facet["DFM"] == row["DFM"]) & (facet["Facet"] == row["Facet"])
                 ].set_index("Chamber")
    assert row["dLicksA"] == pytest.approx(rows.loc[row["PairedChamber"], "LicksA"]
                                           - rows.loc[row["YokedChamber"], "LicksA"])
    measures, text = lick_measures(False)
    assert measures and not any("¼-root" in m for m in measures)
    assert "¼-root" not in text


def test_the_project_report_shows_the_break_requirement_beside_the_breaking_point(gen3):
    e, _results = gen3
    view = break_requirement_view(e.breaking_point_summary())
    columns = list(view.columns)
    assert columns[columns.index("Breaking point") + 1] == "Break requirement"
    ctrl = view.set_index("Treatment").loc["Ctrl"]
    assert ctrl["Break requirement"] == 31 and ctrl["Groups"] == 2
    ## Nothing new for a project of generation-1 members.
    old = e.breaking_point_summary().assign(BreakRequirement=pd.NA)
    assert break_requirement_view(old) is None
    assert break_requirement_view(old.drop(columns="BreakRequirement")) is None


def test_the_project_pools_the_break_requirement(tmp_path):
    from pyflic.base.project import Project
    from pyflic.base.project_report import build_project_report

    spec, results = choice()
    root = tmp_path / "proj"
    design = None
    for name in ("rep1", "rep2"):
        member = pr_dir(root / name, spec, results)
        cfg = yaml.safe_load((member / "flic_config.yaml").read_text())
        design = design or cfg.pop("global")
        cfg.pop("global", None)
        (member / "flic_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (root / "project.yaml").write_text(yaml.safe_dump(
        {"name": "proj", "design": {"global": design}}, sort_keys=False))
    ## A Project's lab type is copied beside project.yaml, for every Member.
    write_trained_pr_type(root)
    project = Project(root)
    for name in project.member_names:
        project.load_member(name, parallel=False,
                            use_disk_cache=False).execute_basic_analysis(skip_qc=True)
    project.build_combined_analysis()
    pooled = pd.read_csv(root / "analysis" / "proj_BreakingPoint.csv")
    assert {"BreakRequirement", "FirmwareEvents"} <= set(pooled.columns)
    text = (root / "analysis" / "proj_Stats.txt").read_text(encoding="utf-8")
    line = next(t for t in text.splitlines() if t.startswith("Break requirement"))
    assert "Ctrl BreakingPoint 4, BreakRequirement 31" in line
    doc = build_project_report(project)
    tables = [b for b in doc.blocks if isinstance(b, rl.Table)
              and (b.caption or "").startswith("Breaking point and the Requirement")]
    assert tables and "Break requirement" in tables[0].frame.columns


def test_light_on_seconds_are_the_on_ms_of_the_chamber(gen3):
    e, results = gen3
    summary = e.feeding_summary()
    for _, row in summary.iterrows():
        rows = results[int(row["DFM"])].rows
        a, b = 2 * int(row["Chamber"]) - 1, 2 * int(row["Chamber"])
        expected = np.maximum(rows[f"OnMs{a}"].to_numpy(), rows[f"OnMs{b}"].to_numpy()).sum()
        assert row["LightOn_sec"] == pytest.approx(expected / 1000.0)
    ## Over a Facet's window, the rows inside it.
    facet = e.feeding_summary_facet()
    test = facet[(facet.Facet == "Test") & (facet.DFM == 1) & (facet.Chamber == 1)].iloc[0]
    dfm = e.dfms[1]
    inside = dfm.paradigm.minutes > float(test["StartMin"])
    assert test["LightOn_sec"] == pytest.approx(
        dfm.paradigm.on_ms[inside][:, [0, 1]].max(axis=1).sum() / 1000.0)


# ---------------------------------------------------------------------------
# The trend on Firmware Events
# ---------------------------------------------------------------------------

def _judge(**overrides):
    kwargs = dict(training_complete=True, training_light_events=3, training_licks=20,
                  test_counts=np.array([5, 5, 5, 5, 5, 5]), resting={"start": 8, "max": 8,
                                                                     "end": 8, "level": 8},
                  reference_level=8.0, any_light=True)
    kwargs.update(overrides)
    return lqc.judge_group(lqc.LightQCSettings(), **kwargs)


def test_firmware_events_witness_the_trend_but_do_not_set_the_warning():
    rising = np.array([1, 11, 21, 31, 41, 51], dtype=float)
    flat = _judge(firmware_events=rising)
    assert lqc.NO_TREND in flat.flags
    assert lqc.FIRMWARE_TREND_NOTE in flat.notes
    assert flat.trend_rho_firmware == pytest.approx(1.0)
    ## Licks that rise with it: no warning, no note.
    both = _judge(test_counts=np.array([2, 12, 20, 33, 40, 55]), firmware_events=rising)
    assert lqc.NO_TREND not in both.flags and lqc.FIRMWARE_TREND_NOTE not in both.notes
    ## Neither rising: the warning, no note; NA Events are left out.
    neither = _judge(firmware_events=np.array([5, 5, np.nan, 5, 5, 5]))
    assert lqc.NO_TREND in neither.flags and lqc.FIRMWARE_TREND_NOTE not in neither.notes
    ## Generation 1 has none to give.
    assert np.isnan(_judge().trend_rho_firmware)


def test_the_drifting_group_gets_the_detection_note(gen3):
    e, _results = gen3
    table = e.light_qc_table().set_index(["DFM", "Group"])
    for dfm_id, group in groups_of("drifting"):
        row = table.loc[(dfm_id, group)]
        assert lqc.NO_TREND in row["Flags"]
        assert row["TrendRhoFirmware"] == pytest.approx(1.0)
        assert lqc.FIRMWARE_TREND_NOTE in row["Notes"]
    ## The note is one line in the summary, not split at its semicolon.
    lines = e.light_qc_lines()
    assert any(line.strip() == f"· {lqc.FIRMWARE_TREND_NOTE}" for line in lines)


# ---------------------------------------------------------------------------
# FixedRatio (decision 14)
# ---------------------------------------------------------------------------

def test_a_fixed_ratio_breaks_at_its_ratio_without_a_trend_check(fr):
    e, _results = fr
    assert e.test_paradigm(1, 1)[0] == "FR5"
    assert e.test_paradigm(1, 1)[1].type == op.FIXED_RATIO
    summary = e.breaking_point_summary()
    assert len(summary) == 3
    assert (summary["BreakRequirement"] == 5).all()
    assert (summary["LargestRequirement"] == 5).all()
    assert (summary["FirmwareEvents"] == 5 * summary["BreakingPoint"]).all()
    table = e.light_qc_table()
    assert (table["TestLightEvents"] >= lqc.LightQCSettings().trend_min_events).all()
    assert not table["Flags"].str.contains(lqc.NO_TREND).any()
    ledger = e.light_events_ledger()
    assert (ledger["Requirement"] == 5).all()
    blocks = e.report_results_blocks([], SimpleNamespace(include_comparison=False))
    headings = [b.text for b in blocks if isinstance(b, rl.Heading)]
    assert "Fixed ratio: activations earned" in headings
    assert "Breaking point" not in headings


def test_without_a_program_the_type_is_unknown_and_taken_as_progressive(tmp_path):
    spec, results = fixed()
    e = load(pr_dir(tmp_path, spec, results, program=False))
    assert e.test_paradigm(1, 1) == ("FR5", None)
    table = e.light_qc_table()
    assert table["Notes"].str.contains("type is unknown").all()
    ## Judged as progressive: the trend check runs, and the rows still show 5.
    assert not table["Notes"].str.contains("fixed ratio").any()
    assert (e.breaking_point_summary()["BreakRequirement"] == 5).all()
    ## Its type is NA in the breaking point table, not a guess.
    assert e.breaking_point_summary()["ParadigmType"].isna().all()


def test_the_breaking_point_table_names_the_test_paradigms_type(gen3, fr):
    progressive, fixed_ratio = gen3[0].breaking_point_summary(), fr[0].breaking_point_summary()
    assert list(progressive.columns)[-1] == "ParadigmType"
    assert (progressive["ParadigmType"] == "ProgressiveRatio").all()
    assert (fixed_ratio["ParadigmType"] == "FixedRatio").all()


def test_the_project_report_titles_a_fixed_ratio_pool_as_activations_earned(gen3, fr):
    from pyflic.base.experiment_types import get_experiment_type

    etype = get_experiment_type("ProgressiveRatio")

    def heading(bp) -> str:
        project = SimpleNamespace(combined_diff_frame=lambda: None,
                                  combined_breaking_point_frame=lambda: bp,
                                  design_global={}, _mixed_p=None)
        blocks = etype.project_results_blocks(project)
        return next(b.text for b in blocks if isinstance(b, rl.Heading))

    fixed_ratio = fr[0].breaking_point_summary()
    progressive = gen3[0].breaking_point_summary()
    assert heading(fixed_ratio) == "Fixed ratio: activations earned"
    ## A pool with any progressive or unknown row, or written before the
    ## column existed, keeps the breaking point's title.
    assert heading(pd.concat([fixed_ratio, progressive])) == "Breaking point"
    assert heading(fixed_ratio.assign(ParadigmType=None)) == "Breaking point"
    assert heading(fixed_ratio.drop(columns="ParadigmType")) == "Breaking point"


def test_the_faceted_rows_carry_each_windows_light_and_activations(gen3):
    """Training and Test rows each carry LitSec and Activations over their
    own window, where the whole-recording summary puts them."""
    e, _results = gen3
    facet = e.feeding_summary_facet()
    whole = e.feeding_summary()
    assert list(facet.columns[2:]) == list(whole.columns)
    light = ls.light_summary_facet(e).set_index(["Facet", "DFM", "Chamber"])
    for row in facet.itertuples(index=False):
        key = (row.Facet, row.DFM, row.Chamber)
        assert row.LitSec == pytest.approx(light.loc[key, "LitSec"])
        assert row.Activations == light.loc[key, "Activations"]
    ## The Facets partition the recording, so their Activations add up.
    summed = facet.groupby(["DFM", "Chamber"])["Activations"].sum()
    pd.testing.assert_series_equal(
        summed, whole.set_index(["DFM", "Chamber"])["Activations"].sort_index())
    assert set(facet["Role"]) == {"paired", "yoked"}


def test_the_facets_split_at_training_end_so_acclimation_ends_in_training(gen3):
    """The Training window ends on the last Training row and Test begins on
    the first Main row, as the Light Events split, so the paired Sucrose
    Well's Acclimation ends in Training, where the whole recording puts it."""
    e, _results = gen3
    whole = ls.light_summary(e).set_index(["DFM", "Chamber"])
    light = ls.light_summary_facet(e).set_index(["Facet", "DFM", "Chamber"])
    for row in e.training_table().itertuples(index=False):
        if not row.TrainingComplete:
            continue
        minutes = e.dfms[row.DFM].raw_df["Minutes"].to_numpy()
        last = int(np.flatnonzero(minutes == row.TrainingEndMin)[0])
        (_, training), (_, test) = e.group_windows(row.DFM, row.Group)
        assert training[1] == test[0]
        assert minutes[last] <= training[1] < minutes[last + 1]
        key = (row.DFM, row.PairedChamber)
        assert light.loc[("Training", *key), "AcclimationComplete"] == True  # noqa: E712
        assert light.loc[("Training", *key), "AcclimationEndMin"] == \
            whole.loc[key, "AcclimationEndMin"]
        assert pd.isna(light.loc[("Test", *key), "AcclimationEndMin"])


# ---------------------------------------------------------------------------
# Generation 2 (decision 15)
# ---------------------------------------------------------------------------

def test_generation_2_is_one_recording_facet_with_no_removals(gen2, tmp_path):
    e, _results = gen2
    assert e.training_information() == NO_TRAINING_INFORMATION
    assert e.facet_labels() == ["Recording"]
    facet = e.feeding_summary_facet()
    assert set(facet["Facet"]) == {"Recording"} and len(facet) == 12
    assert facet["PersistA"].isna().all()
    table = e.training_table()
    assert table["TrainingComplete"].isna().all()
    assert (table["Notes"] == NO_TRAINING_INFORMATION).all()
    qc = e.light_qc_table()
    assert qc["TrainingComplete"].isna().all()
    assert not qc["Flags"].str.contains(lqc.IMPLAUSIBLE_TRAINING).any()
    ## One wording everywhere: the light QC's note, the Hub's and the Project
    ## Report's (light_summary) are the type's own.
    lit = ~qc["Notes"].str.startswith("no light events")
    assert qc.loc[lit, "Notes"].str.startswith(NO_TRAINING_INFORMATION).all() and lit.any()
    assert ls.NO_TRAINING_INFORMATION == NO_TRAINING_INFORMATION
    assert e.breaking_point_summary().empty
    diff = e.paired_yoked_diff()
    assert set(diff["Facet"]) == {"Recording"} and diff["TrainingComplete"].isna().all()
    ## require_training_complete removes nothing: there is nothing to judge.
    removed = e.auto_remove_chambers()
    assert not removed["Reason"].str.contains("training").any()
    assert e.light_qc_table()["Excluded"].sum() == 0


def test_generation_2_says_so_in_the_summary_and_the_report(gen2):
    e, _results = gen2
    summary = e.summary_text(include_qc=False)
    assert NO_TRAINING_INFORMATION in summary
    assert "no Breaking Point or Sucrose Persistence" in summary
    callouts = [b for b in e.report_glance_blocks() if isinstance(b, rl.Callout)]
    assert any(NO_TRAINING_INFORMATION in b.text for b in callouts)
    ## Not reported as a training that failed.
    assert not any("Training never completed" in b.text for b in callouts)


@pytest.mark.parametrize("writer", [write_v4_recording, write_gen2_recording],
                         ids=["gen3", "gen2"])
def test_the_pipeline_and_the_report_run(tmp_path, writer):
    from pyflic.base.pdf_report import build_experiment_report

    spec, results = choice()
    root = pr_dir(tmp_path / "exp", spec, results, writer=writer)
    e = load(root)
    e.execute_basic_analysis(skip_qc=True)
    analysis = root / "analysis"
    for name in e.experiment_type.output_manifest():
        assert (analysis / name).is_file(), name
    summary = (analysis / "summary.txt").read_text(encoding="utf-8")
    assert (NO_TRAINING_INFORMATION in summary) == (writer is write_gen2_recording)
    doc = build_experiment_report(load(root), metrics=("Licks",))
    assert doc.save(tmp_path / "report.pdf").stat().st_size > 0


def test_a_mixed_experiment_has_the_union_of_the_facets(tmp_path):
    spec, results = choice()
    data = tmp_path / "data"
    write_v4_recording(data, spec, results={1: results[1]}, program=False, messages=False)
    write_gen2_recording(data, spec, results={2: results[2]}, program=False, messages=False)
    (data / "Program.txt").write_text(program_text(spec), encoding="utf-8")
    write_config(tmp_path, [1, 2])
    e = load(tmp_path)
    assert [e.dfms[d].generation for d in (1, 2)] == [3, 2]
    assert e.facet_labels() == ["Training", "Test", "Recording"]
    assert e.training_information() == f"DFM 2: {NO_TRAINING_INFORMATION}"
    facet = e.feeding_summary_facet()
    assert set(facet.loc[facet.DFM == 2, "Facet"]) == {"Recording"}
    assert {"Training", "Test"} <= set(facet.loc[facet.DFM == 1, "Facet"])
    assert set(e.breaking_point_summary()["DFM"]) == {1}


# ===========================================================================
# Paradigm analyses, stream S4 (docs/paradigm-analyses-plan.md): solo Chamber
# Groups (decision 13), the Fixed Ratio wording (decision 14), restart
# censoring (decision 15) and the place of the requirement series check.
# ===========================================================================

#: Every chamber its own treatment pattern: legal for solo groups only.
ALTERNATING = {1: "Ctrl", 2: "Exp", 3: "Ctrl", 4: "Exp", 5: "Ctrl", 6: "Exp"}


def unpaired_dir(root: Path, spec, results, *, chamber_groups: str | None = None,
                 chambers: dict | None = None, program: bool = True, **kw) -> Path:
    """A Progressive Ratio recording whose config states no
    ``paired_chambers``: the Chamber Groups come from Program.txt's Linkage,
    or from ``chamber_groups`` without it."""
    root.mkdir(parents=True, exist_ok=True)
    write_v4_recording(root / "data", spec, results=results, program=program, **kw)
    cfg = pr_config([{"id": int(d.dfm_id), "paired_chambers": [1, 3, 5],
                      "chambers": dict(chambers or ALTERNATING)} for d in spec.dfms])
    for node in cfg["dfms"]:
        node.pop("paired_chambers")
        if chamber_groups is not None:
            node["chamber_groups"] = chamber_groups
    cfg["global"]["experiment_type"] = TRAINED_PR
    (root / "flic_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    write_trained_pr_type(root)
    return root


@pytest.fixture(scope="module")
def solo(tmp_path_factory):
    """The choice recording without ``paired_chambers``: CHOICE_LINKAGE links
    each chamber's two wells alone, and every A well triggers, so Program.txt
    makes twelve solo groups."""
    spec, results = choice()
    return load(unpaired_dir(tmp_path_factory.mktemp("solo"), spec, results)), results


def _text(blocks) -> str:
    out = []
    for b in blocks:
        if isinstance(b, rl.Table):
            out += [str(b.caption), " ".join(map(str, b.frame.columns))]
        elif isinstance(b, rl.Plot):
            out.append(f"{b.title} {b.caption}")
        elif isinstance(b, rl.PlotRow):
            out += [f"{p.title} {p.caption}" for p in b.plots]
        elif isinstance(b, rl.Bullets):
            out += list(b.items)
        else:
            out.append(str(getattr(b, "text", "")))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Solo Chamber Groups (decision 13)
# ---------------------------------------------------------------------------

def test_program_linkage_makes_a_solo_group_of_each_chamber(solo):
    e, _results = solo
    for dfm_id in PAIRED:
        assert [(g.number, g.paired, g.yoked, g.solo) for g in e.chamber_groups(dfm_id)] == \
            [(c, c, None, True) for c in range(1, 7)]
    assert any("chamber groups taken from Program.txt's linkage: 6 groups, each a single "
               "chamber, with no yoked control" in n for n in e.program_notes)
    roles = e.roles_table()
    assert (roles["Role"] == "paired").all() and roles["Partner"].isna().all()
    assert e.paired_chambers_by_dfm() == {1: [1, 2, 3, 4, 5, 6], 2: [1, 2, 3, 4, 5, 6]}
    ## The group columns keep their names; YokedChamber is NA.
    training = e.training_table()
    assert training["YokedChamber"].isna().all() and len(training) == 12
    assert (training["PairedChamber"] == training["Group"]).all()
    assert e.light_qc_table()["YokedChamber"].isna().all()
    ## Each chamber keeps its own treatment: no pair to share one with.
    assert e.light_qc_table().set_index(["DFM", "Group"]).loc[(1, 2), "Treatment"] == "Exp"


def test_solo_groups_have_a_breaking_point_and_no_paired_yoked_difference(solo):
    e, results = solo
    assert e.paired_yoked_diff().empty and e.cumulative_diff_data().empty
    assert not e.has_yoked_controls() and len(e.solo_groups()) == 12
    summary = e.breaking_point_summary().set_index(["DFM", "Group"])
    ## Every group whose fly finished Acclimation has a breaking point.
    trained = {(d, g.number) for d in PAIRED for g in e.chamber_groups(d)
               if e.group_training(d, g.number).complete}
    assert set(summary.index) == trained
    for (dfm_id, group), row in summary.iterrows():
        assert row["PairedChamber"] == group
        assert row["Treatment"] == ALTERNATING[group]
    ## Healthy chambers 1, 2 and 6 earn PR10's series, 1, 11, 21, ...
    for chamber in (1, 2, 6):
        n = int(summary.loc[(1, chamber), "BreakingPoint"])
        assert summary.loc[(1, chamber), "BreakRequirement"] == 1 + 10 * (n - 1)
    ## The facet summary has one paired row per chamber and Facet.
    facet = e.feeding_summary_facet()
    assert set(facet["Role"]) == {"paired"}
    assert not facet.duplicated(["Facet", "DFM", "Chamber"]).any()


def test_the_report_says_once_that_the_design_has_no_yoked_controls(solo):
    from pyflic.base.progressive_ratio_experiment import NO_YOKED_CONTROLS

    e, _results = solo
    results_text = _text(e.report_results_blocks([], SimpleNamespace(include_comparison=True)))
    assert results_text.count(NO_YOKED_CONTROLS.capitalize()) == 1
    assert "Cumulative difference curve" not in results_text
    assert "Paired − yoked difference, Test phase" not in results_text
    assert "The yoked fly has no breaking point" not in results_text
    assert "Breaking point" in results_text
    qc_text = _text(e.report_qc_blocks())
    assert "Yoked chamber" not in qc_text            # a column of NA says nothing
    assert NO_YOKED_CONTROLS.capitalize() not in qc_text
    glance = _text(e.report_glance_blocks())
    assert "the chamber of each was excluded" in glance          # never-training, dead LED
    assert "both chambers" not in glance


def test_solo_groups_in_a_figure_wrap_three_to_a_row(solo):
    from pyflic.base.progressive_ratio_experiment import _panel_figsize

    e, _results = solo
    ## Three groups keep the paired layout's one row; six take two.
    assert _panel_figsize(3) == (12.0, 5.0) and _panel_figsize(6) == (12.0, 10.0)
    for figure in (e.plot_light_events_dfm(1), e.plot_cumulative_licks_dfm(1),
                   e.plot_resting_level_dfm(1), e.plot_breaking_point_dfm(1),
                   e.plot_cumulative_diff()):
        figure.draw()
    assert "no yoked controls" in str(getattr(e.plot_cumulative_diff().labels, "title", ""))


def test_chamber_groups_single_without_a_program(tmp_path):
    from pyflic.base.yaml_lint import lint_flic_config

    spec, results = fixed()
    root = unpaired_dir(tmp_path / "single", spec, results, chamber_groups="single",
                        program=False)
    issues = [i.message for i in lint_flic_config(root / "flic_config.yaml")]
    assert not [m for m in issues if "paired_chambers" in m or "chamber_groups" in m]
    e = load(root)
    assert [(g.paired, g.yoked) for g in e.chamber_groups(1)] == [(c, None) for c in range(1, 7)]
    assert len(e.breaking_point_summary()) == 6
    ## Without either key and without Program.txt, a pair cannot be resolved.
    bare = unpaired_dir(tmp_path / "bare", spec, results, program=False)
    errors = [i.message for i in lint_flic_config(bare / "flic_config.yaml")
              if i.severity == "error"]
    assert any("'paired_chambers' is required" in m and "chamber_groups: single" in m
               for m in errors)
    with pytest.raises(ValueError, match="paired_chambers"):
        load(bare)


def test_paired_chambers_and_single_cannot_both_be_stated():
    from pyflic.base import experiment_types as et

    pr = et.get_experiment_type("ProgressiveRatio")
    problems = pr.validate_dfm(1, {"paired_chambers": [1, 3, 5], "chamber_groups": "single"},
                               {1: "Ctrl", 2: "Exp"})
    assert len(problems) == 1 and "cannot be stated with 'chamber_groups: single'" in problems[0]
    problems = pr.validate_dfm(1, {"chamber_groups": "triples"}, {})
    assert any("'chamber_groups' must be one of paired, single" in p for p in problems)
    ## Single chambers share nothing: split treatments are fine.
    assert pr.validate_dfm(1, {"chamber_groups": "single"}, dict(ALTERNATING)) == []


def test_program_pairs_need_not_be_adjacent(tmp_path):
    """YOKED_LINKAGE links chamber c with chamber c + 3; chambers 1-3 trigger.
    The groups are those pairs, and a pair still shares one treatment."""
    from pyflic.base.experiment_types.progressive_ratio import (
        groups_from_program,
        paired_from_program,
    )

    spec = yoked_controls_spec()
    section = op.parse_program(program_text(spec)).section(1)
    groups, problems, _notes = groups_from_program(section)
    assert problems == []
    assert [(g.number, g.paired, g.yoked) for g in groups] == [(1, 1, 4), (2, 2, 5), (3, 3, 6)]
    ## The paired layout's reading of the same section fails, as before.
    assert paired_from_program(section)[0] is None
    shared = {1: "Ctrl", 2: "Exp", 3: "Third", 4: "Ctrl", 5: "Exp", 6: "Third"}
    e = load(unpaired_dir(tmp_path / "yoked", spec, simulate(spec), chambers=shared))
    assert [(g.paired, g.yoked) for g in e.chamber_groups(1)] == [(1, 4), (2, 5), (3, 6)]
    assert e.role_of(1, 4).role == "yoked" and e.role_of(1, 4).partner == 1
    diff = e.paired_yoked_diff()
    assert sorted(zip(diff["PairedChamber"], diff["YokedChamber"])) == [(1, 4), (2, 5), (3, 6)]
    with pytest.raises(ValueError, match="chambers 1 and 4 form one chamber group"):
        load(unpaired_dir(tmp_path / "split", spec, simulate(spec), chambers=ALTERNATING))


def test_a_linkage_can_mix_pairs_and_solo_groups():
    from pyflic.base.experiment_types.progressive_ratio import groups_from_program

    spec, _results = choice()
    thresholds = (20, OFF, OFF, OFF) + (20, OFF) * 4          # chamber 2 does not trigger
    dfm = replace(spec.dfms[0], linkage=(1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5),
                  intervals=(IntervalSpec("PR10", True, thresholds, 30.0),))
    mixed = replace(spec, dfms=(dfm,))
    groups, problems, _notes = groups_from_program(op.parse_program(program_text(mixed)).section(1))
    assert problems == []
    assert [(g.number, g.paired, g.yoked) for g in groups] == \
        [(1, 1, 2), (2, 3, None), (3, 4, None), (4, 5, None), (5, 6, None)]
    ## Three chambers on one light circuit, or a chamber nothing lights, make none.
    wide = replace(dfm, linkage=(1,) * 6 + (2, 2, 3, 3, 4, 4))
    groups, problems, _notes = groups_from_program(
        op.parse_program(program_text(replace(spec, dfms=(wide,)))).section(1))
    assert groups is None and any("one light circuit" in p for p in problems)
    dark = replace(dfm, linkage=CHOICE_LINKAGE)
    groups, problems, _notes = groups_from_program(
        op.parse_program(program_text(replace(spec, dfms=(dark,)))).section(1))
    assert groups is None and "Program.txt lights chamber 2 from no trigger well" in problems


def test_single_disagreeing_with_program_pairs_is_noted(tmp_path):
    from pyflic.base.yaml_lint import lint_flic_config

    spec = yoked_controls_spec()
    root = unpaired_dir(tmp_path / "exp", spec, simulate(spec), chamber_groups="single")
    e = load(root)
    assert all(g.solo for g in e.chamber_groups(1))
    note = ("DFM 1: chamber_groups single disagrees with Program.txt, whose linkage makes "
            "chamber groups 1 (paired 1, yoked 4), 2 (paired 2, yoked 5), 3 (paired 3, yoked 6); "
            "the config's are used")
    assert note in e.program_notes
    warnings = [i.message for i in lint_flic_config(root / "flic_config.yaml")
                if i.severity == "warning"]
    assert note in warnings


def test_a_solo_group_drops_alone_and_a_pair_keeps_its_breaking_point(solo, tmp_path):
    spec, results = choice()
    e = load(unpaired_dir(tmp_path / "solo", spec, results))
    before = set(e.breaking_point_summary().set_index(["DFM", "Group"]).index)
    e._remove_chambers_from_design({(1, 2)})
    after = set(e.breaking_point_summary().set_index(["DFM", "Group"]).index)
    assert before - after == {(1, 2)}
    ## A pair: the Breaking Point is the Paired fly's alone, so removing the
    ## Yoked chamber keeps the group's row, unchanged (review C1); only the
    ## Paired-Yoked Difference, which needs both flies, loses it.
    paired = load(pr_dir(tmp_path / "paired", spec, results))
    whole = paired.breaking_point_summary().set_index(["DFM", "Group"])
    assert (1, 1) in set(whole.index)
    paired_chamber = int(whole.loc[(1, 1), "PairedChamber"])
    yoked = paired.group_training(1, 1).yoked_chamber
    paired._remove_chambers_from_design({(1, int(yoked))})
    left = paired.breaking_point_summary().set_index(["DFM", "Group"])
    assert (1, 1) in set(left.index)
    assert int(left.loc[(1, 1), "BreakingPoint"]) == int(whole.loc[(1, 1), "BreakingPoint"])
    assert paired.paired_yoked_diff().query("Group == 1 and DFM == 1").empty
    ## Removing the Paired chamber does take the group out.
    paired._remove_chambers_from_design({(1, paired_chamber)})
    gone = set(paired.breaking_point_summary().set_index(["DFM", "Group"]).index)
    assert (1, 1) not in gone


def test_the_project_report_says_there_are_no_yoked_controls(solo):
    from pyflic.base.experiment_types import get_experiment_type
    from pyflic.base.experiment_types.progressive_ratio import pooled_yoked_controls

    e, _results = solo
    etype = get_experiment_type("ProgressiveRatio")
    light_qc = e.light_qc_table().assign(Source="PR")

    def project(qc, diff):
        return SimpleNamespace(combined_diff_frame=lambda: diff,
                               combined_breaking_point_frame=e.breaking_point_summary,
                               combined_light_qc_frame=lambda: qc,
                               design_global={}, _mixed_p=None)

    solo_project = project(light_qc, e.paired_yoked_diff())
    assert not pooled_yoked_controls(solo_project)
    text = _text(etype.project_results_blocks(solo_project))
    assert text.count("The design has no yoked controls") == 1
    ## A pooled design with any pair, or none known, says nothing of it.
    paired = light_qc.assign(YokedChamber=2)
    assert pooled_yoked_controls(project(paired, None))
    assert pooled_yoked_controls(SimpleNamespace())
    assert "no yoked controls" not in _text(etype.project_results_blocks(project(paired, None)))


def test_the_config_editor_writes_the_chamber_groups(tmp_path):
    pytest.importorskip("PyQt6")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from pyflic.base.config_editor import FLICConfigEditor

    _app = QApplication.instance() or QApplication([])
    cfg = pr_config([{"id": 1, "paired_chambers": [1, 3, 5], "chambers": dict(ALTERNATING)},
                     {"id": 2, "paired_chambers": [1, 3, 5], "chambers": dict(ALTERNATING)}])
    del cfg["dfms"][0]["paired_chambers"]
    cfg["dfms"][0]["chamber_groups"] = "single"
    del cfg["dfms"][1]["paired_chambers"]
    (tmp_path / "flic_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    win = FLICConfigEditor(initial_path=tmp_path)
    try:
        single, program = win._dfm_widgets
        assert single.groups_mode() == "single" and program.groups_mode() == "program"
        out = win._collect_yaml()["dfms"]
        assert out[0]["chamber_groups"] == "single" and "paired_chambers" not in out[0]
        ## Program.txt's groups: neither key, so saving keeps the program's.
        assert "chamber_groups" not in out[1] and "paired_chambers" not in out[1]
        ## No pair, so no shared treatment to ask for.
        assert not any("must share a treatment" in p for p in single.chamber_problems())
        single.set_paired_chambers([2, 3, 6])
        assert single.groups_mode() == "paired"
        assert win._collect_yaml()["dfms"][0]["paired_chambers"] == [2, 3, 6]
        assert any("must share a treatment" in p for p in single.chamber_problems())
    finally:
        win.close()


# ---------------------------------------------------------------------------
# The Fixed Ratio wording (decision 14)
# ---------------------------------------------------------------------------

#: Text that belongs to a progressive ratio only.
PROGRESSIVE_ONLY = ("estimated requirement", "requirement increment",
                    "progressive ratio climbs")


def test_a_fixed_ratio_report_has_none_of_the_progressive_only_text(fr, gen3):
    def every_text(e) -> str:
        figure = e.plot_light_events_dfm(1)
        return "\n".join([
            e.summary_text(include_qc=False),
            _text(e.report_qc_blocks()),
            _text(e.report_results_blocks([], SimpleNamespace(include_comparison=False))),
            str(getattr(figure.labels, "caption", "") or ""),
        ]).lower()

    fixed_ratio = every_text(fr[0])
    assert not [t for t in PROGRESSIVE_ONLY if t.lower() in fixed_ratio]
    assert "under a fixed ratio" in fixed_ratio
    assert fr[0].estimated_increment() is None and fr[0].progressive_groups() == []
    ## A progressive one keeps every word.
    progressive = every_text(gen3[0])
    assert all(t.lower() in progressive for t in PROGRESSIVE_ONLY)


# ---------------------------------------------------------------------------
# Restart censoring (decision 15)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def restarted(tmp_path_factory):
    """``family_specs()['restart']``: DFM 1 FR3 (Acclimation 2) restarting
    halfway through, in Main; DFM 2 ClosedLoop.  CHOICE_LINKAGE, so each
    chamber is a solo group."""
    spec = family_specs()["restart"]
    results = simulate(spec)
    chambers = {c: "Ctrl" for c in range(1, 7)}
    return load(unpaired_dir(tmp_path_factory.mktemp("restart"), spec, results,
                             chambers=chambers)), results


def test_the_fixed_ratio_groups_are_censored_at_the_restart(restarted):
    e, results = restarted
    state = e.dfms[1].paradigm
    (span,) = state.restarts()
    down_ms, _resume_ms, _missing = results[1].truth.restarts_ms[0]
    summary = e.breaking_point_summary().set_index(["DFM", "Group"])
    assert sorted(summary.index) == [(1, g) for g in range(1, 7)]   # DFM 2 never ran a ratio
    for group in range(1, 7):
        assert e.test_restart(1, group) == span
        gt = e.group_training(1, group)
        row = summary.loc[(1, group)]
        assert row["ParadigmType"] == "FixedRatio" and row["BreakRequirement"] == 3
        ## The window ends at the last row before the restart, and the count
        ## still running there is censored.
        assert row["TestMinutes"] == pytest.approx(float(state.minutes[span.row - 1])
                                                   - gt.training_end)
        assert row["Censored"]
        ledger = e.light_events_ledger()
        mine = ledger[(ledger["DFM"] == 1) & (ledger["Group"] == group)]
        after = mine["Minutes"] > row["TestMinutes"]
        assert after.any() and not mine.loc[after, "Counted"].any()
        assert int(mine["Counted"].sum()) == row["BreakingPoint"]
        ## The light events counted all started before the DFM went down.
        onsets = truth_onsets_ms(results[1].truth, gt.sucrose_well)
        assert sum(t < down_ms for t in onsets) >= row["BreakingPoint"]


def test_the_restart_is_named_where_the_breaking_point_is_read(restarted):
    e, _results = restarted
    qc = e.light_qc_table().set_index(["DFM", "Group"])
    lines = "\n".join(e.breaking_point_lines())
    for group in range(1, 7):
        assert "the DFM restarted at minute 10.1" in qc.loc[(1, group), "Notes"]
        assert f"DFM 1 group {group}: the DFM restarted at minute 10.1" in lines
        assert any("after the DFM restarted at minute 10.1 (after the Test window, which "
                   "ends at the restart)" in n for n in e.group_training(1, group).notes)
    assert "Test window ends at the restart" in _text(
        e.report_results_blocks([], SimpleNamespace(include_comparison=False)))
    ## The ClosedLoop DFM's restart is outside any Test window.
    assert all(e.test_restart(2, g.number) is None for g in e.chamber_groups(2))


# ---------------------------------------------------------------------------
# Requirement series wrong: the place for stream S1's check
# ---------------------------------------------------------------------------

def test_the_series_check_fails_a_group_and_is_the_light_qcs_own(gen3):
    from dataclasses import replace

    e, _results = gen3
    assert lqc.SERIES_WRONG in lqc.FAILING_FLAGS
    failed = _judge(series_mismatches=2)
    assert lqc.SERIES_WRONG in failed.flags and failed.failed
    for quiet in (None, 0):
        assert lqc.SERIES_WRONG not in _judge(series_mismatches=quiet).flags
    ## The rows the check reads: the Test window's Main Activations of the
    ## Paired Sucrose Well, numbered from 1 and meeting PR10's series.
    for dfm_id, group in groups_of("healthy"):
        rows = e._test_ledger_rows(dfm_id, group)
        assert len(rows) >= 3 and (rows["Phase"] == "M").all()
        assert rows["K"].tolist() == list(range(1, len(rows) + 1))
        paradigm = e.test_paradigm(dfm_id, group)[1]
        assert rows["RequirementMet"].tolist() == [paradigm.requirement_for(k) for k in rows["K"]]
        ## The optogenetic light QC's check, called here: clean on the
        ## program that ran, and every event but the first wrong against a
        ## program whose Step differs (the first Requirement is the same).
        assert lqc.requirement_series_mismatches(rows, paradigm) == 0
        other = replace(paradigm, step=paradigm.step + 5)
        assert lqc.requirement_series_mismatches(rows, other) == len(rows) - 1
    assert not e.light_qc_table()["Flags"].str.contains(lqc.SERIES_WRONG).any()
    ## Off generation 3 there are no rows to read.
    assert e._test_ledger_rows(1, 3).empty                  # never-training: no Test


def truth_onsets_ms(truth, well: int) -> list[int]:
    return list(truth.activation_onset_ms[int(well)])
