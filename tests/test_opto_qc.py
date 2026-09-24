"""The optogenetic light QC: was the light where the licks were?

Three layers: the arithmetic on plain arrays, one DFM judged from hand-built
arrays and a program, and whole experiments (``opto_fixtures``) through the
loader, the pipeline, the report and the script actions.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from pyflic.base import opto_program as op
from pyflic.base import opto_qc as oq
from pyflic.base.yaml_config import load_experiment_yaml

from opto_fixtures import (
    WELL_CASES, interval_line, make_opto_dir, pr_program, program_text,
)

T0 = datetime(2025, 1, 1, 8, 0, 0)
SPS = 5


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------

def test_near_and_explained_light():
    act = np.zeros(20, dtype=bool)
    act[5] = True
    assert np.flatnonzero(oq.near(act, 2, 1)).tolist() == [4, 5, 6, 7]
    ## A lick explains the light from the tolerance before it to the decay
    ## plus the tolerance after it.
    explained = oq.explained_light(np.ones(20, dtype=bool), act, decay_samples=3,
                                   tolerance=1)
    assert np.flatnonzero(explained).tolist() == [4, 5, 6, 7, 8, 9]


def test_the_decay_rounds_up_to_whole_samples():
    assert oq.decay_samples_for(500, 5) == 3
    assert oq.decay_samples_for(1000, 5) == 5
    assert oq.decay_samples_for(0, 5) == 0


def test_the_onset_is_the_first_window_over_the_warning_fraction():
    minutes = np.arange(120, dtype=float)
    lit = np.ones(120, dtype=bool)
    unexplained = minutes >= 65
    assert oq.onset_minute(minutes, lit, unexplained, 0.10) == 60.0
    assert np.isnan(oq.onset_minute(minutes, lit, np.zeros(120, bool), 0.10))


def test_event_spans_read_the_event_column():
    assert oq.event_spans(np.array([0, 3, 0, 0, 0, 2, 0])) == [(1, 4), (5, 7)]


def test_settings_come_from_the_constants():
    s = oq.OptoQCSettings.from_constants({"opto_unexplained_fail_fraction": 0.5,
                                          "exclude_failed_opto_chambers": True,
                                          "opto_decay_tolerance_samples": "x"})
    assert s.fail_fraction == 0.5 and s.exclude is True
    assert s.tolerance_samples == 2 and s.warn_fraction == pytest.approx(0.10)
    assert "opto_unexplained_min_sec=30" in s.describe()


def test_constant_problems_are_named():
    problems = oq.opto_constant_problems({
        "opto_unexplained_warn_fraction": 1.5, "opto_decay_tolerance_samples": 1.5,
        "exclude_failed_opto_chambers": "yes"})
    assert len(problems) == 3
    assert oq.opto_constant_problems({"opto_unexplained_warn_fraction": 0.4,
                                      "opto_unexplained_fail_fraction": 0.2}) == [
        "'constants.opto_unexplained_warn_fraction' must not exceed "
        "'constants.opto_unexplained_fail_fraction'"]


def test_every_type_validates_the_setting_and_the_constants():
    from pyflic.base.experiment_types import get_experiment_type

    for name in (None, "Hedonic", "ProgressiveRatio"):
        problems = get_experiment_type(name).validate(
            {"optogenetics": "maybe",
             "constants": {"opto_unexplained_fail_fraction": 2}})
        assert any("optogenetics" in p for p in problems)
        assert any("opto_unexplained_fail_fraction" in p for p in problems)


def test_verdicts():
    assert oq.verdict_of([]) == "ok"
    assert oq.verdict_of([oq.PARTLY_UNEXPLAINED]) == "warning"
    assert oq.verdict_of([oq.PARTLY_UNEXPLAINED, oq.LIGHT_WHILE_OFF]) == "failed"


# ---------------------------------------------------------------------------
# One DFM from arrays
# ---------------------------------------------------------------------------

def _inputs(n, *, lights=None, activity=None, feeding=None, raw=None, origin=T0,
            **columns) -> oq.DFMInputs:
    L = np.zeros((n, 12), dtype=bool)
    A = np.zeros((n, 12), dtype=bool)
    F = np.zeros((n, 12), dtype=int)
    R = np.full((n, 12), 5.0)
    for w, mask in (lights or {}).items():
        L[:, w - 1] = mask
    for w, mask in (activity or {}).items():
        A[:, w - 1] = mask
    for w, spans in (feeding or {}).items():
        for a, b in spans:
            F[a, w - 1] = b - a
            A[a:b, w - 1] = True
    for w, values in (raw or {}).items():
        R[:, w - 1] = values
    return oq.DFMInputs(dfm_id=1, minutes=np.arange(n) / SPS / 60.0,
                        samples_per_second=SPS, lights=L, activity=A, feeding=F, raw=R,
                        chamber_of_well={w: w for w in range(1, 13)}, origin=origin,
                        data_frequency=columns.get("freq"),
                        data_pulse_width=columns.get("pw"),
                        data_dark=columns.get("dark"))


def _program(thresholds, *, minutes=10.0, decay=1000, delay=0, max_on=0, pw=8,
             linkage=None, intervals=None, start=T0, baseline=True) -> op.OptoProgram:
    lines = intervals or [interval_line(thresholds, start=start, duration_min=minutes,
                                        decay=decay, delay=delay, max_on=max_on, pw=pw)]
    total = sum(float(line.rsplit(",", 1)[1].replace("min.", "")) for line in lines)
    section = {"intervals": lines}
    if linkage is not None:
        section["linkage"] = linkage
    return op.parse_program(program_text({1: section}, start=start, minutes=total,
                                         baseline="Yes" if baseline else "No"))


def _analyze(inputs, program, *, setting="auto", **constants):
    settings = oq.OptoQCSettings.from_constants(constants)
    return oq.analyze_dfm(inputs, program=program, setting=setting, settings=settings)


def _group(result, label):
    return next(g for g in result.groups if g["Group"] == label)


THRESHOLDS = [20] + [-1] * 11
N = 3000                     # ten minutes


def test_light_within_the_decay_of_a_lick_is_explained():
    lit = np.zeros(N, dtype=bool)
    spans = [(100 + 300 * k, 110 + 300 * k) for k in range(8)]
    for a, b in spans:
        lit[a:b + 5] = True                           # the burst, then 1 s decay
    result = _analyze(_inputs(N, lights={1: lit}, feeding={1: spans}),
                      _program(THRESHOLDS))
    g = _group(result, 1)
    assert g["Verdict"] == "ok" and g["UnexplainedSec"] == 0.0
    assert g["LightEvents"] == 8 and g["UnexplainedEvents"] == 0


def test_long_light_without_licks_fails_and_says_when():
    lit = np.zeros(N, dtype=bool)
    lit[600:2400] = True
    result = _analyze(_inputs(N, lights={1: lit}), _program(THRESHOLDS))
    g = _group(result, 1)
    assert g["Flags"] == oq.UNEXPLAINED and g["Verdict"] == "failed"
    assert g["UnexplainedFraction"] == pytest.approx(1.0)
    assert g["UnexplainedOnsetMin"] == 0.0
    assert g["LikelyCause"] == oq.CAUSE_HARDWARE      # the signal never rose
    event = result.events[0]
    assert event["Explained"] is False and event["OverrunSec"] == pytest.approx(360.0)


def test_a_little_unexplained_light_is_a_note_not_a_verdict():
    lit = np.zeros(N, dtype=bool)
    lit[100:130] = True                                # six seconds, no lick
    result = _analyze(_inputs(N, lights={1: lit}), _program(THRESHOLDS))
    g = _group(result, 1)
    assert g["Verdict"] == "ok" and "under opto_unexplained_min_sec" in g["Notes"]
    judged = _analyze(_inputs(N, lights={1: lit}), _program(THRESHOLDS),
                      opto_unexplained_min_sec=0)
    assert _group(judged, 1)["Flags"] == oq.UNEXPLAINED


def test_the_warning_band():
    lit = np.zeros(N, dtype=bool)
    spans = [(100 + 20 * k, 110 + 20 * k) for k in range(100)]    # lit and licked
    for a, b in spans:
        lit[a:b] = True
    lit[2500:2700] = True                                           # 40 s unexplained
    result = _analyze(_inputs(N, lights={1: lit}, feeding={1: spans}),
                      _program(THRESHOLDS))
    g = _group(result, 1)
    assert 0.10 <= g["UnexplainedFraction"] < 0.30
    assert g["Flags"] == oq.PARTLY_UNEXPLAINED and g["Verdict"] == "warning"


def test_light_while_every_threshold_is_minus_one_fails():
    lit = np.zeros(N, dtype=bool)
    lit[1000:1010] = True
    result = _analyze(_inputs(N, lights={2: lit}), _program(THRESHOLDS))
    g = _group(result, 2)
    assert g["Flags"] == oq.LIGHT_WHILE_OFF and g["LitWhileOffSec"] == pytest.approx(2.0)


def test_open_loop_must_be_lit():
    thresholds = [0, 0] + [-1] * 10
    full = np.ones(N, dtype=bool)
    half = np.arange(N) < N // 2
    result = _analyze(_inputs(N, lights={1: full, 2: half}), _program(thresholds))
    assert _group(result, 1)["Verdict"] == "ok"
    assert _group(result, 1)["Paradigm"] == op.OPEN_LOOP
    assert _group(result, 2)["Flags"] == oq.OPEN_LOOP_DARK


def test_feeding_that_never_lights_a_closed_loop_warns():
    spans = [(200 + 300 * k, 210 + 300 * k) for k in range(6)]
    result = _analyze(_inputs(N, feeding={1: spans}), _program(THRESHOLDS))
    g = _group(result, 1)
    assert g["Flags"] == f"{oq.UNLIT_FEEDING}, {oq.NO_LIGHT_EVENTS}"
    assert g["Verdict"] == "warning" and g["UnlitFeedingEvents"] == 6


def test_the_reverse_check_is_closed_loop_only():
    """Under a progressive ratio most feeding is supposed to go unlit."""
    spans = [(200 + 300 * k, 210 + 300 * k) for k in range(6)]
    lit = np.zeros(N, dtype=bool)
    lit[200:215] = True
    result = _analyze(_inputs(N, lights={1: lit}, feeding={1: spans}),
                      _program(THRESHOLDS, decay=1000, delay=100, max_on=10000))
    g = _group(result, 1)
    assert g["Paradigm"] == op.PROGRESSIVE_RATIO and g["Verdict"] == "ok"
    assert pd.isna(g["UnlitFeedingEvents"])


def test_a_linkage_group_answers_to_its_trigger_wells_only():
    thresholds = [20, -1] + [-1] * 10
    linkage = [1, 1] + list(range(3, 13))
    lit = np.zeros(N, dtype=bool)
    lit[500:515] = True
    lit[1500:1700] = True                              # only W2 was touched here
    act2 = np.zeros(N, dtype=bool)
    act2[1500:1700] = True
    result = _analyze(_inputs(N, lights={1: lit, 2: lit}, feeding={1: [(500, 510)]},
                              activity={2: act2}),
                      _program(thresholds, linkage=linkage))
    g = _group(result, 1)
    assert g["Wells"] == "W1, W2" and g["TriggerWells"] == "W1"
    ## W1's bout explains its own light; W2's touching explains nothing.
    assert g["UnexplainedSec"] == pytest.approx(200 / SPS)


def test_the_emulated_trigger_names_a_drifting_sensor():
    raw = np.full(N, 5.0)
    raw[1000:] += np.linspace(0, 80, N - 1000)          # firmware reads contact
    lit = (raw - 5.0) > 20
    result = _analyze(_inputs(N, lights={1: lit}, raw={1: raw}), _program(THRESHOLDS))
    g = _group(result, 1)
    assert g["Flags"] == oq.UNEXPLAINED and g["LikelyCause"] == oq.CAUSE_DRIFT
    assert g["ContactWithoutActivitySec"] == pytest.approx(g["ContactSec"])
    assert g["EmulationApproximate"] is False


def test_baseline_no_compares_the_raw_signal():
    raw = np.full(N, 30.0)                              # above 20 from the start
    result = _analyze(_inputs(N, raw={1: raw}), _program(THRESHOLDS, baseline=False))
    assert _group(result, 1)["ContactSec"] == pytest.approx(N / SPS - 0.2, abs=0.5)
    result = _analyze(_inputs(N, raw={1: raw}), _program(THRESHOLDS))
    assert _group(result, 1)["ContactSec"] == 0.0       # baseline removes it


def test_the_program_and_the_data_are_cross_checked():
    freq = np.full(N, 50.0)
    result = _analyze(_inputs(N, freq=freq, pw=np.full(N, 8.0), dark=np.zeros(N)),
                      _program(THRESHOLDS))
    assert oq.PROGRAM_MISMATCH in result.flags
    assert any("OptoFreq is 50" in note for note in result.notes)
    assert all(oq.PROGRAM_MISMATCH in g["Flags"] for g in result.groups)
    late = _analyze(_inputs(N, origin=T0 + timedelta(minutes=5)),
                    _program(THRESHOLDS, minutes=20))
    assert any("5.0 min after the program's Start Time" in n for n in late.notes)
    assert _group(late, 1)["EmulationApproximate"] is True


def test_light_left_over_from_the_interval_before_is_not_judged():
    lines = [interval_line(THRESHOLDS, start=T0, duration_min=5),
             interval_line([-1] * 12, start=T0 + timedelta(minutes=5), duration_min=5)]
    boundary = 5 * 60 * SPS
    lit = np.zeros(N, dtype=bool)
    lit[boundary - 10:boundary + 4] = True               # the decay runs over
    spans = [(boundary - 10, boundary - 1)]
    result = _analyze(_inputs(N, lights={1: lit}, feeding={1: spans}),
                      _program(THRESHOLDS, intervals=lines))
    g = _group(result, 1)
    assert g["Verdict"] == "ok" and g["Paradigm"] == f"{op.CLOSED_LOOP}, {op.LIGHTS_OFF}"
    lit[boundary + 500:boundary + 510] = True
    result = _analyze(_inputs(N, lights={1: lit}, feeding={1: spans}),
                      _program(THRESHOLDS, intervals=lines))
    assert _group(result, 1)["Flags"] == oq.LIGHT_WHILE_OFF


def test_non_feeding_activation_is_not_judged():
    lit = np.ones(N, dtype=bool)
    result = _analyze(_inputs(N, lights={1: lit}),
                      _program(THRESHOLDS, pw=32776, delay=500, decay=21600000))
    g = _group(result, 1)
    assert g["Verdict"] == "ok" and "non-feeding activation" in g["Notes"]


def test_without_a_program_linkage_is_inferred_and_nothing_fails():
    lit = np.zeros(N, dtype=bool)
    lit[600:2400] = True
    other = np.zeros(N, dtype=bool)
    other[10:20] = True
    result = _analyze(_inputs(N, lights={1: lit, 2: lit, 3: other}), None)
    assert result.program_status == oq.PROGRAM_NO_FILE
    assert [g["Wells"] for g in result.groups] == ["W1, W2", "W3"]
    g = result.groups[0]
    assert g["Flags"] == oq.UNEXPLAINED_NO_PROGRAM and g["Verdict"] == "warning"
    assert "no Program.txt" in result.notes[0]


def test_a_dfm_the_program_does_not_cover():
    program = op.parse_program(program_text({2: {"intervals": [
        interval_line(THRESHOLDS, start=T0, duration_min=10)]}}, start=T0, minutes=10))
    result = _analyze(_inputs(N, lights={1: np.ones(N, bool)}), program)
    assert result.program_status == oq.PROGRAM_NO_SECTION
    assert oq.NO_PROGRAM_SECTION in result.groups[0]["Flags"]


def test_yes_with_no_light_at_all_fails_the_dfm():
    result = _analyze(_inputs(N), None, setting="yes")
    assert [g["Group"] for g in result.groups] == [0]
    assert result.groups[0]["Flags"] == oq.NO_LIGHT
    assert result.groups[0]["Verdict"] == "failed"
    ## Nothing was expected to light: every threshold -1.
    result = _analyze(_inputs(N), _program([-1] * 12), setting="yes")
    assert all(g["Verdict"] == "ok" for g in result.groups)
    ## ...and auto never fails on darkness alone.
    assert _analyze(_inputs(N), None).groups[0]["Verdict"] == "ok"


def test_the_decay_profile_follows_the_schedule():
    lines = [interval_line(THRESHOLDS, start=T0, duration_min=5, decay=500),
             interval_line(THRESHOLDS, start=T0 + timedelta(minutes=5), duration_min=5,
                           decay=2000)]
    program = _program(THRESHOLDS, intervals=lines)
    minutes = np.arange(N) / SPS / 60.0
    profile = oq.decay_profile(1, minutes, SPS, T0, program, oq.OptoQCSettings())
    assert profile[0] == 3 and profile[-1] == 10
    assert oq.decay_profile(1, minutes, SPS, T0, None, oq.OptoQCSettings())[0] == 5


# ---------------------------------------------------------------------------
# Whole experiments
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rig_dir(tmp_path_factory) -> Path:
    return make_opto_dir(tmp_path_factory.mktemp("opto") / "exp")


@pytest.fixture(scope="module")
def rig(rig_dir):
    return load_experiment_yaml(rig_dir, parallel=False, use_disk_cache=False)


EXPECTED = {1: "ok", 2: "failed", 3: "failed", 4: "failed", 5: "ok", 6: "warning",
            7: "warning", 8: "ok"}


def test_each_case_gets_its_verdict(rig):
    assert rig.is_optogenetic and rig.opto.dfm_ids() == [1]
    table = rig.opto.qc_table()
    verdicts = dict(zip(table["Group"], table["Verdict"]))
    for well, verdict in EXPECTED.items():
        assert verdicts[well] == verdict, (well, WELL_CASES[well][0])
    by_group = table.set_index("Group")
    assert by_group.loc[2, "LikelyCause"] == oq.CAUSE_DRIFT
    assert by_group.loc[3, "LikelyCause"] == oq.CAUSE_HARDWARE
    assert by_group.loc[3, "UnexplainedOnsetMin"] == 30.0
    assert by_group.loc[2, "Treatment"] == "Exp" and by_group.loc[3, "Treatment"] == "Ctrl"
    assert not by_group["Excluded"].any()
    assert rig.opto.program_line().startswith("Program.txt: 1 DFM section(s)")


def test_every_chamber_row_carries_its_groups_verdict(rig):
    fs = rig.feeding_summary()
    row = fs.set_index("Chamber")
    assert row.loc[2, "OptoLightQC"] == oq.UNEXPLAINED
    assert row.loc[1, "OptoLightQC"] == "" and row.loc[1, "OptoUnexplainedFraction"] == 0.0
    assert row.loc[2, "OptoUnexplainedFraction"] > 0.9


def test_the_lines_say_what_failed(rig):
    text = "\n".join(rig.opto.lines())
    assert "DFM 1 linkage group 2 (W2; chamber(s) 2; Exp): FAILED, kept and flagged" in text
    assert "drifting baseline or sustained contact" in text
    summary = rig.summary_text(include_qc=False)
    assert "Optogenetic light QC" in summary
    assert "FAILED but kept and flagged" in summary


def test_the_pipeline_writes_qc_opto(tmp_path):
    root = make_opto_dir(tmp_path / "exp")
    e = load_experiment_yaml(root, parallel=False, use_disk_cache=False)
    result = e.execute_basic_analysis(skip_qc=True)
    opto = root / "qc" / "opto"
    for name in ("opto_light_qc.csv", "opto_light_intervals.csv", "opto_light_events.csv",
                 "opto_program.csv", "opto_light_dfm1.png"):
        assert (opto / name).is_file(), name
    assert result["opto_light_qc"] == opto / "opto_light_qc.csv"
    qc = pd.read_csv(opto / "opto_light_qc.csv")
    assert list(qc.columns) == list(oq.OPTO_QC_COLUMNS)
    program = pd.read_csv(opto / "opto_program.csv")
    assert list(program.columns) == list(oq.OPTO_PROGRAM_COLUMNS)
    assert program.loc[0, "Paradigm"] == op.CLOSED_LOOP
    assert program.loc[0, "DecayMs"] == 1000
    ## The values the MCU echoed, beside their decoding.
    assert (program.loc[0, "F"], program.loc[0, "FrequencyHz"]) == (40, 40)
    assert (program.loc[0, "P"], bool(program.loc[0, "NonFeeding"])) == (8, False)
    ## Failures are flagged, not excluded, by default.
    removed = pd.read_csv(root / "analysis" / "removed_chambers.csv")
    assert not removed["Reason"].astype(str).str.contains("opto").any()
    assert "Optogenetic light QC" in (root / "analysis" / "summary.txt").read_text(
        encoding="utf-8")


def test_the_switch_excludes_failed_groups(tmp_path):
    root = make_opto_dir(tmp_path / "exp",
                         constants={"exclude_failed_opto_chambers": True})
    e = load_experiment_yaml(root, parallel=False, use_disk_cache=False)
    removed = e.apply_auto_removal()
    opto = removed[removed["Reason"].str.contains("opto light QC failed")]
    assert sorted(opto["Chamber"].astype(int)) == [2, 3, 4]
    table = e.opto.qc_table().set_index("Group")
    assert table.loc[2, "Excluded"] and not table.loc[6, "Excluded"]
    assert table.loc[2, "Treatment"] == "Exp"       # still named after removal
    assert "exclude_failed_opto_chambers = true" in e.filter_criteria_summary


def test_the_setting_decides_what_is_covered(tmp_path):
    dark = make_opto_dir(tmp_path / "dark", program=None, light_cases=False)
    e = load_experiment_yaml(dark, parallel=False, use_disk_cache=False)
    assert not e.is_optogenetic                 # auto: no program, no light
    assert "OptoLightQC" not in e.feeding_summary().columns
    e = load_experiment_yaml(make_opto_dir(tmp_path / "yes", program=None,
                                           light_cases=False, optogenetics=True),
                             parallel=False, use_disk_cache=False)
    assert e.opto.qc_table()["Flags"].tolist() == [oq.NO_LIGHT]
    e = load_experiment_yaml(make_opto_dir(tmp_path / "no", optogenetics=False),
                             parallel=False, use_disk_cache=False)
    assert not e.is_optogenetic and e.opto_program is None
    e = load_experiment_yaml(make_opto_dir(tmp_path / "dfm", optogenetics=True,
                                           dfm_extra={"optogenetics": False}),
                             parallel=False, use_disk_cache=False)
    assert not e.is_optogenetic                 # the DFM's own setting wins
    e = load_experiment_yaml(make_opto_dir(tmp_path / "light", program=None),
                             parallel=False, use_disk_cache=False)
    assert e.is_optogenetic and e.opto.result(1).program_status == oq.PROGRAM_NO_FILE


def test_bad_settings_stop_the_load(tmp_path):
    with pytest.raises(ValueError, match="optogenetics"):
        load_experiment_yaml(make_opto_dir(tmp_path / "a", optogenetics="sometimes"),
                             parallel=False, use_disk_cache=False)
    with pytest.raises(ValueError, match="opto_unexplained_fail_fraction"):
        load_experiment_yaml(make_opto_dir(
            tmp_path / "b", constants={"opto_unexplained_fail_fraction": 3}),
            parallel=False, use_disk_cache=False)
    with pytest.raises(ValueError, match=r"dfms\[1\]\.optogenetics"):
        load_experiment_yaml(make_opto_dir(tmp_path / "c",
                                           dfm_extra={"optogenetics": "sometimes"}),
                             parallel=False, use_disk_cache=False)


def test_an_unreadable_program_is_set_aside(tmp_path):
    root = make_opto_dir(tmp_path / "exp", program="nothing to see here\n")
    e = load_experiment_yaml(root, parallel=False, use_disk_cache=False)
    assert e.opto_program is None and "no DFM section" in e.opto_program_error
    assert e.opto.program_line().startswith("Program.txt could not be used")
    assert e.is_optogenetic                     # the data are lit all the same


def test_a_mismatched_frequency_is_reported(tmp_path):
    e = load_experiment_yaml(make_opto_dir(tmp_path / "exp", freq=50),
                             parallel=False, use_disk_cache=False)
    assert oq.PROGRAM_MISMATCH in e.opto.result(1).flags
    assert any("OptoFreq is 50" in line for line in e.opto.lines())


def test_the_report_carries_the_opto_light_qc(rig):
    from pyflic.base import report_layout as rl
    from pyflic.base.pdf_report import build_experiment_report

    doc = build_experiment_report(rig)
    blocks = list(doc.blocks) if hasattr(doc, "blocks") else list(doc._blocks)
    headings = [b.text for b in blocks if isinstance(b, rl.Heading)]
    assert "Optogenetic light QC" in headings
    callouts = [b for b in blocks if isinstance(b, rl.Callout)
                and (b.title or "").startswith("Opto light QC")]
    assert callouts and callouts[0].tone == "failed"


def test_the_script_actions(rig, tmp_path):
    from pyflic.base.script_editor.runner import run_experiment_script

    logs: list[str] = []
    figures = run_experiment_script(
        rig, {"name": "t", "steps": [{"action": "opto_light_qc"},
                                     {"action": "plot_opto_light", "binsize": 5}]},
        log=logs.append)
    assert (Path(rig.qc_dir) / "opto" / "opto_light_qc.csv").is_file()
    assert (Path(rig.qc_dir) / "opto" / "opto_light_dfm1.png").is_file()
    assert figures and "DFM 1" in figures[0][0]
    dark = load_experiment_yaml(make_opto_dir(tmp_path / "dark", program=None,
                                              light_cases=False),
                                parallel=False, use_disk_cache=False)
    logs.clear()
    run_experiment_script(dark, {"name": "t", "steps": [{"action": "opto_light_qc"}]},
                          log=logs.append)
    assert any("[Skip] opto_light_qc needs an optogenetic experiment" in m for m in logs)


def test_the_figure_builds(rig):
    fig = rig.opto.plot_dfm(1, binsize_min=5)
    fig.draw()


# ---------------------------------------------------------------------------
# Progressive Ratio
# ---------------------------------------------------------------------------

def _pr_dir(root: Path, *, drop_paired: bool, program_paired=None) -> Path:
    from pr_fixtures import make_pr_experiment_dir

    make_pr_experiment_dir(root)
    cfg = yaml.safe_load((root / "flic_config.yaml").read_text())
    if drop_paired:
        for node in cfg["dfms"]:
            node.pop("paired_chambers")
    (root / "flic_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    paired = program_paired or {1: (1, 4, 5), 2: (2, 3, 6)}
    ## The fixture's DFM 2 has its Sucrose Well on the right.
    text = pr_program({1: paired[1]}, pi_direction="left")
    right = pr_program({2: paired[2]}, pi_direction="right")
    text = text + right[right.index("***DFM 2***"):]
    (root / "data" / "Program.txt").write_text(text, encoding="utf-8")
    return root


def test_paired_chambers_come_from_the_program(tmp_path):
    e = load_experiment_yaml(_pr_dir(tmp_path / "exp", drop_paired=True),
                             parallel=False, use_disk_cache=False)
    assert e.paired_chambers_by_dfm() == {1: [1, 4, 5], 2: [2, 3, 6]}
    assert any("taken from Program.txt" in note for note in e.program_notes)
    assert any("taken from Program.txt" in n for n in e.opto.experiment_notes())


def test_a_disagreeing_program_is_reported_and_the_config_kept(tmp_path):
    root = _pr_dir(tmp_path / "exp", drop_paired=False,
                   program_paired={1: (2, 4, 5), 2: (2, 3, 6)})
    e = load_experiment_yaml(root, parallel=False, use_disk_cache=False)
    assert e.paired_chambers_by_dfm()[1] == [1, 4, 5]
    assert any("disagrees with Program.txt" in note for note in e.program_notes)


def test_pr_light_events_are_decay_aware(tmp_path):
    e = load_experiment_yaml(_pr_dir(tmp_path / "exp", drop_paired=False),
                             parallel=False, use_disk_cache=False)
    events = e.light_events_table(1, 1)
    assert "Explained" in events.columns
    assert (events["LickFree"] == ((events["LicksSincePrev"] == 0)
                                   & ~events["Explained"])).all()
    ## The PR program's decay (500 ms) is what the events are read against.
    assert int(e.opto.decay_samples(1)[0]) == 3
    notes = e.opto.experiment_notes()
    assert not any("runs no progressive-ratio interval" in n for n in notes)


def test_a_program_that_is_not_the_types_paradigm_is_noted(rig):
    ## The rig is a Custom experiment running closed loop: nothing to note.
    assert not any("progressive ratio" in n for n in rig.opto.experiment_notes())


# ---------------------------------------------------------------------------
# The Project roll-up
# ---------------------------------------------------------------------------

def test_the_project_stacks_opto_verdicts_with_a_source(tmp_path):
    from pyflic.base.project import Project

    root = tmp_path / "proj"
    member = make_opto_dir(root / "rig1")
    cfg = yaml.safe_load((member / "flic_config.yaml").read_text())
    design = cfg.pop("global")
    (member / "flic_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (root / "project.yaml").write_text(yaml.safe_dump(
        {"name": "proj", "design": {"global": design}}, sort_keys=False))
    project = Project(root)
    project.load_member("rig1", parallel=False,
                        use_disk_cache=False).execute_basic_analysis(skip_qc=True)
    written = {Path(p).name for p in project.build_combined_analysis()["written"]}
    assert "proj_LightQC.csv" in written
    stacked = pd.read_csv(root / "analysis" / "proj_LightQC.csv")
    assert set(stacked["Source"]) == {"Opto"}
    flagged = project.flagged_groups().set_index("Group")
    assert flagged.loc[2, "Status"] == "retained (exclude_failed_opto_chambers: false)"
    assert flagged.loc[6, "Status"] == "kept — warning only"
    assert flagged.loc[2, "Wells"] == "W2"
    text = (root / "analysis" / "proj_Stats.txt").read_text(encoding="utf-8")
    assert "linkage group 2 [W2]" in text and "[Opto]" in text


def test_a_project_design_carries_the_setting(tmp_path):
    from pyflic.base.project import DESIGN_KEYS, Project

    assert "optogenetics" in DESIGN_KEYS
    root = tmp_path / "proj"
    member = make_opto_dir(root / "rig1")
    cfg = yaml.safe_load((member / "flic_config.yaml").read_text())
    design = cfg.pop("global")
    design["optogenetics"] = False
    cfg["dfms"][0]["optogenetics"] = True        # the DFM overrides, inside a Project
    (member / "flic_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (root / "project.yaml").write_text(yaml.safe_dump(
        {"name": "proj", "design": {"global": design}}, sort_keys=False))
    e = Project(root).load_member("rig1", parallel=False, use_disk_cache=False)
    assert e.optogenetics == "no" and e.is_optogenetic


# ---------------------------------------------------------------------------
# The linter
# ---------------------------------------------------------------------------

def test_the_linter_knows_the_setting(tmp_path):
    from pyflic.base.yaml_lint import lint_flic_config

    root = make_opto_dir(tmp_path / "ok", optogenetics=True,
                         dfm_extra={"optogenetics": False})
    messages = [i.message for i in lint_flic_config(root / "flic_config.yaml")]
    assert not any("optogenetics" in m for m in messages)
    root = make_opto_dir(tmp_path / "bad", optogenetics="sometimes")
    issues = lint_flic_config(root / "flic_config.yaml")
    assert any(i.severity == "error" and "optogenetics" in i.message for i in issues)


def test_the_linter_takes_paired_chambers_from_the_program(tmp_path):
    from pyflic.base.yaml_lint import lint_flic_config

    root = _pr_dir(tmp_path / "exp", drop_paired=True)
    messages = [i.message for i in lint_flic_config(root / "flic_config.yaml")]
    assert not any("'paired_chambers' is required" in m for m in messages)
    (root / "data" / "Program.txt").unlink()
    messages = [i.message for i in lint_flic_config(root / "flic_config.yaml")]
    assert any("'paired_chambers' is required" in m for m in messages)
