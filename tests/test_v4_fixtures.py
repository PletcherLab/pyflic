"""The firmware 4.0.0 simulator (tests/v4_fixtures.py) against its golden values.

The Paradigm rules are pinned to the values the firmware's own unit tests
(DFMV3_V2 ``tests/test_ledcontrol.c``) and ``docs/v4-data-plan.md`` give; the
files to PyMCU's formats, and where PyMCU's checkout is beside this one, to
PyMCU's own exporter and row formatter run on the same values.  The
millisecond-for-millisecond comparison with the compiled firmware is
``test_ledcontrol_crosscheck.py`` (``-m slow``).
"""

from __future__ import annotations

import ast
import csv
import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from v4_fixtures import START as V4_START
from v4_fixtures import (
    CHOICE_LINKAGE,
    CHOICE_PR_CASES,
    CL500,
    COLUMNS,
    DATA_FILE_DTYPES,
    FAMILY_SPEC_KEYS,
    FI10S,
    FI500,
    FR3,
    NF_MAX5S,
    NF1000,
    OFF,
    ON,
    PARADIGM_COLUMNS,
    PR10,
    PR_EXP,
    YOKED_LINKAGE,
    Bout,
    DFMSpec,
    IntervalSpec,
    OnMsFault,
    ParadigmSpec,
    RecordingSpec,
    Restart,
    SimResult,
    choice_pr_spec,
    family_specs,
    firmware_inputs,
    fixed_ratio_spec,
    messages_text,
    one_of_each_spec,
    paradigm_frame,
    program_text,
    read_v4_csv,
    repeating_open_loop_spec,
    requirement_series,
    restart_messages,
    simulate,
    stress_spec,
    write_gen2_recording,
    write_v4_recording,
    yoked_controls_spec,
)

PYMCU_DIR = Path(os.environ.get("PYMCU_DIR", "C:/Users/scott/Desktop/GithubLocal/PyMCU"))
START = datetime(2026, 10, 16, 12, 0, 0)
IDENTITY = tuple(range(1, 13))
TRIGGER_ONLY: tuple[int | str, ...] = (20,) + (OFF,) * 11
RECORD_MS = 200
EXPONENTIAL_DEFAULTS = [1, 2, 4, 6, 9, 12, 15, 20, 25, 32, 40, 50, 62, 77, 95]
ChoiceRun = tuple[RecordingSpec, dict[int, SimResult]]


# ---------------------------------------------------------------------------
# Small recordings
# ---------------------------------------------------------------------------

def _iv(name: str, minutes: float, thresholds: tuple[int | str, ...] = TRIGGER_ONLY, dark: bool = True) -> IntervalSpec:
    return IntervalSpec(name, dark, thresholds, minutes)


def _spec(intervals: list[IntervalSpec], paradigms: list[ParadigmSpec], bouts: dict[int, list[Bout]], *,
          minutes: float, linkage: tuple[int, ...] = IDENTITY, program_type: str = "Linear") -> RecordingSpec:
    dfm = DFMSpec(1, tuple(intervals), bouts, linkage=linkage, program_type=program_type)
    return RecordingSpec(start=START, duration_min=minutes, paradigms={p.name: p for p in paradigms}, dfms=(dfm,))


def _run(spec: RecordingSpec) -> SimResult:
    return simulate(spec)[spec.dfms[0].dfm_id]


def _train(first_s: float, n: int, length_s: float, every_s: float) -> list[Bout]:
    """*n* bouts of *length_s*, starting every *every_s* from *first_s*."""
    return [Bout(round(first_s + k * every_s, 3), length_s) for k in range(n)]


def _ms(bouts: list[Bout]) -> list[int]:
    return [round(b.start_s * 1000) for b in bouts]


def _record(ms: int) -> int:
    """The row (from 0) of the record holding step *ms*."""
    return ms // RECORD_MS


@pytest.fixture(scope="module")
def choice() -> ChoiceRun:
    spec = choice_pr_spec()
    return spec, simulate(spec)


# ---------------------------------------------------------------------------
# The Requirement series and the ratio rules
# ---------------------------------------------------------------------------

def test_the_exponential_series_defaults_are_richardson_roberts() -> None:
    expo = ParadigmSpec("Expo", "ProgressiveRatio", activation_ms=1, series="Exponential")
    assert requirement_series(expo, 15) == EXPONENTIAL_DEFAULTS
    # The fifteen need 450 Events: bouts of 50 ms, 300 ms apart, one Firmware Event each.
    bouts = _train(6.0, 460, 0.05, 0.3)
    result = _run(_spec([_iv("Expo", 3.0)], [expo], {1: bouts}, minutes=3.0))
    starts = _ms(bouts)
    assert result.truth.requirement_met[1][:15] == EXPONENTIAL_DEFAULTS
    assert result.truth.activation_onset_ms[1][:15] == [starts[n - 1] for n in np.cumsum(EXPONENTIAL_DEFAULTS)]


def test_arithmetic_1_2_cap_5_activates_at_events_1_4_9_14() -> None:
    pr = ParadigmSpec("PR", "ProgressiveRatio", activation_ms=10, first_requirement=1, step=2, cap=5)
    bouts = _train(6.0, 16, 0.1, 1.0)
    truth = _run(_spec([_iv("PR", 1.0)], [pr], {1: bouts}, minutes=1.0)).truth
    starts = _ms(bouts)
    assert truth.activation_onset_ms[1] == [starts[e - 1] for e in (1, 4, 9, 14)]
    assert truth.requirement_met[1] == [1, 3, 5, 5]


def test_fixed_ratio_3_activates_at_events_3_6_9() -> None:
    fr = ParadigmSpec("FR3", "FixedRatio", activation_ms=50, ratio=3)
    bouts = _train(6.0, 10, 0.1, 1.0)
    truth = _run(_spec([_iv("FR3", 1.0)], [fr], {1: bouts}, minutes=1.0)).truth
    starts = _ms(bouts)
    assert truth.activation_onset_ms[1] == [starts[e - 1] for e in (3, 6, 9)]
    assert truth.requirement_met[1] == [3, 3, 3]
    assert truth.activation_phase[1] == ["M", "M", "M"]


def test_acclimation_3_gives_three_closed_loop_activations_then_the_first_requirement() -> None:
    # The firmware test's settings: ActivationMs 50, Arithmetic from 2 by 1.
    pr = ParadigmSpec("PRA", "ProgressiveRatio", activation_ms=50, first_requirement=2, step=1, cap=100,
                      acclimation=3)
    bouts = _train(6.0, 6, 0.2, 1.0)
    result = _run(_spec([_iv("PRA", 1.0)], [pr], {1: bouts}, minutes=1.0))
    truth, rows = result.truth, result.rows
    starts = _ms(bouts)
    assert truth.activation_phase[1] == ["A", "A", "A", "M"]
    # Each Acclimation Activation is lit for the contact plus ActivationMs.
    assert truth.lit_spans_ms[1][:3] == [(s, s + 200 + 50) for s in starts[:3]]
    assert truth.acclimation_end_ms[1] == starts[2] + 250
    # The first Main Activation needs Requirement k = 1 (2 Events), not more.
    assert truth.activation_onset_ms[1][3] == starts[4]
    assert truth.requirement_met[1] == [None, None, None, 2]
    acclimating = rows[rows["Phase1"] == "A"]
    assert len(acclimating) > 0
    assert (acclimating["Requirement1"] == 2).all() and (acclimating["Progress1"] == 0).all()
    assert acclimating["Activations1"].max() == 3
    first_main = rows.index[rows["Phase1"].eq("M") & (rows.index > acclimating.index[0])][0]
    assert first_main == _record(truth.acclimation_end_ms[1])
    assert (rows.loc[first_main:, "Phase1"] == "M").all()


def test_on_the_row_where_activations_rise_progress_is_0_and_the_next_requirement_shows(choice: ChoiceRun) -> None:
    _, results = choice
    series = requirement_series(PR10, 40)
    checked = 0
    for result in results.values():
        rows, truth = result.rows, result.truth
        for chamber, case in CHOICE_PR_CASES.items():
            well = 2 * chamber - 1
            main = 0
            for onset, phase, met in zip(truth.activation_onset_ms[well], truth.activation_phase[well],
                                         truth.requirement_met[well]):
                if rows.at[_record(onset), "Paradigm"] != "PR10":
                    continue
                r = _record(onset)
                assert rows.at[r, f"Progress{well}"] == 0
                if phase == "M":
                    main += 1
                    # The Requirement met is on the row before; the next one already shows.
                    assert rows.at[r - 1, f"Requirement{well}"] == met == series[main - 1]
                assert rows.at[r, f"Requirement{well}"] == series[main]
                checked += 1
    assert checked > 100


# ---------------------------------------------------------------------------
# Linkage, On and Off, and the other Paradigm types
# ---------------------------------------------------------------------------

def test_a_linked_off_well_gets_the_trigger_wells_light_and_nothing_else() -> None:
    fr = ParadigmSpec("FR2", "FixedRatio", activation_ms=500, ratio=2)
    bouts = {1: _train(6.0, 20, 0.3, 1.5), 2: _train(6.7, 20, 0.3, 1.5)}
    rows = _run(_spec([_iv("FR2", 1.0)], [fr], bouts, minutes=1.0, linkage=CHOICE_LINKAGE)).rows
    assert rows["OnMs1"].sum() > 0
    assert rows["OnMs2"].equals(rows["OnMs1"])
    assert (rows["Activations2"] == 0).all() and rows["Activations1"].max() == 10
    assert rows["Requirement2"].isna().all() and rows["Progress2"].isna().all()
    assert rows["Requirement1"].iloc[25:].notna().all()


def test_open_loop_and_on_wells_are_lit_every_record_with_no_activations() -> None:
    ol = ParadigmSpec("OL", "OpenLoop")
    cl = ParadigmSpec("CL", "ClosedLoop", hold_ms=200)
    thresholds: tuple[int | str, ...] = (ON, 20, OFF) + (OFF,) * 9
    bouts = {w: _train(6.0, 20, 0.5, 2.0) for w in (1, 2, 3)}
    open_loop = _run(_spec([_iv("OL", 1.0, thresholds)], [ol], bouts, minutes=1.0)).rows.iloc[25:]
    assert (open_loop["OnMs1"] == 200).all() and (open_loop["OnMs2"] == 200).all()
    assert (open_loop["OnMs3"] == 0).all()
    assert (open_loop[[f"Activations{w}" for w in range(1, 13)]] == 0).all().all()
    closed = _run(_spec([_iv("CL", 1.0, thresholds)], [cl], bouts, minutes=1.0)).rows.iloc[25:]
    assert (closed["OnMs1"] == 200).all() and (closed["Activations1"] == 0).all()
    assert closed["Activations2"].max() == 20


def test_a_closed_loop_cut_off_holds_until_the_event_ends() -> None:
    cl = ParadigmSpec("CL", "ClosedLoop", hold_ms=500, max_on_ms=1000)
    # 3 s of contact, a 50 ms dip (shorter than the Event gap), 1 s more, then a new bout.
    bouts = [Bout(10.0, 3.0), Bout(13.05, 1.0), Bout(20.0, 0.3)]
    truth = _run(_spec([_iv("CL", 1.0)], [cl], {1: bouts}, minutes=1.0)).truth
    assert truth.activation_onset_ms[1] == [10_000, 20_000]
    assert truth.lit_spans_ms[1] == [(10_000, 11_000), (20_000, 20_800)]
    # An Event ends EventGapMs after its last contact (14 049 and 20 299).
    assert truth.event_spans_ms[1] == [(10_000, 14_149), (20_000, 20_399)]


def test_fixed_interval_contact_held_through_lockouts_activates_at_each_expiry() -> None:
    fi = ParadigmSpec("FI", "FixedInterval", activation_ms=100, lockout_ms=1000)
    truth = _run(_spec([_iv("FI", 1.0)], [fi], {1: [Bout(10.0, 3.5)]}, minutes=1.0)).truth
    assert truth.activation_onset_ms[1] == [10_000, 11_000, 12_000, 13_000]
    assert truth.lit_spans_ms[1] == [(t, t + 100) for t in (10_000, 11_000, 12_000, 13_000)]


def test_non_feeding_lights_at_the_start_and_after_the_delay() -> None:
    nf = ParadigmSpec("NF", "NonFeeding", on_delay_ms=500, max_on_ms=0)
    truth = _run(_spec([_iv("NF", 1.0)], [nf], {1: [Bout(10.0, 1.0), Bout(20.0, 1.0)]}, minutes=1.0)).truth
    lead_in = 25 * RECORD_MS
    assert truth.lit_spans_ms[1] == [(lead_in, 10_000), (11_500, 20_000), (21_500, 60_000)]
    assert truth.activation_onset_ms[1] == [lead_in, 11_500, 21_500]


# ---------------------------------------------------------------------------
# Keep-state and reset
# ---------------------------------------------------------------------------

def _two_intervals(first: str, second: str, paradigms: list[ParadigmSpec], bouts: list[Bout]) -> SimResult:
    # Interval 2 starts at 60 s, its scheduled time.
    return _run(_spec([_iv(first, 1.0), _iv(second, 1.0, (25,) + (OFF,) * 11)], paradigms, {1: bouts},
                      minutes=2.0))


def test_the_same_paradigm_name_keeps_progress_and_phase_and_a_new_one_resets_them() -> None:
    fr_a = ParadigmSpec("FRa", "FixedRatio", activation_ms=50, ratio=3, acclimation=3)
    fr_b = replace(fr_a, name="FRb")
    # Two Acclimation Events before 60 s, then bouts every 2 s.
    bouts = [Bout(20.0, 0.2), Bout(30.0, 0.2)] + _train(62.0, 12, 0.2, 2.0)
    starts = _ms(bouts)
    kept = _two_intervals("FRa", "FRa", [fr_a], bouts)
    assert kept.truth.schedule == [(5_000, 1, 1, "FRa"), (60_000, 2, 1, "FRa")]
    assert kept.truth.activation_phase[1][:4] == ["A", "A", "A", "M"]
    # The third Acclimation Event is the first after the change; Main needs 3 more.
    assert kept.truth.activation_onset_ms[1][2:4] == [starts[2], starts[5]]
    reset = _two_intervals("FRa", "FRb", [fr_a, fr_b], bouts)
    # Acclimation starts again: three more, then the third Main Event.
    assert reset.truth.activation_phase[1][:6] == ["A", "A", "A", "A", "A", "M"]
    assert reset.truth.activation_onset_ms[1][2:6] == [starts[2], starts[3], starts[4], starts[7]]
    assert reset.rows.at[_record(60_000), "Phase1"] == "A"
    # Activations never reset within a recording.
    assert reset.rows["Activations1"].is_monotonic_increasing


def test_a_reset_counts_a_new_activation_for_a_well_that_stays_lit() -> None:
    cl_a = ParadigmSpec("CLa", "ClosedLoop")
    cl_b = replace(cl_a, name="CLb")
    contact = [Bout(50.0, 20.0)]           # across the change at 60 s
    kept = _two_intervals("CLa", "CLa", [cl_a], contact)
    assert kept.truth.activation_onset_ms[1] == [50_000]
    reset = _two_intervals("CLa", "CLb", [cl_a, cl_b], contact)
    assert reset.truth.activation_onset_ms[1] == [50_000, 60_000]
    # The light itself never went off.
    assert reset.truth.lit_spans_ms[1] == [(50_000, 70_000)]


# ---------------------------------------------------------------------------
# The schedule and the rows
# ---------------------------------------------------------------------------

def test_the_lead_in_rows_are_interval_0_with_no_paradigm() -> None:
    """PyMCU names no instruction before the program: the first bench
    recording's lead-in reads Interval 0, Cycle NA and Paradigm NA."""
    rows = _run(fixed_ratio_spec(minutes=1.0)).rows
    lead, program = rows.iloc[:25], rows.iloc[25:]
    assert (lead["Interval"] == 0).all() and lead["Cycle"].isna().all()
    assert lead["Paradigm"].isna().all()
    assert (lead[[f"Phase{w}" for w in range(1, 13)]] == "M").all().all()
    assert (lead[[f"Activations{w}" for w in range(1, 13)]] == 0).all().all()
    assert lead[[f"{c}{w}" for c in ("Requirement", "Progress") for w in range(1, 13)]].isna().all().all()
    assert (lead[[f"OnMs{w}" for w in range(1, 13)] + ["Dark", "OptoFreq", "OptoPW", "OptoCol1"]] == 0).all().all()
    assert (program["Interval"] == 1).all() and (program["Cycle"] == 1).all()
    assert (program["Paradigm"] == "FR5").all() and (program["OptoFreq"] == 40).all()
    assert (program["OptoPW"] == 8).all() and (program["Dark"] == 1).all()
    # Phase A for the level-threshold Wells only.
    assert (program["Phase1"].iloc[0] == "A") and (program["Phase2"] == "M").all()
    stamps = pd.to_datetime(rows["Date"] + " " + rows["Time"], format="%m/%d/%Y %H:%M:%S") \
        + pd.to_timedelta(rows["MSec"], unit="ms")
    assert (stamps == pd.Timestamp(V4_START) + pd.to_timedelta(rows["Index"] * RECORD_MS, unit="ms")).all()
    assert (rows["VoltsIn"] == 5.01).all() and (rows[["Temp", "Humid", "LUX", "OptoCol2", "Error"]] == 0).all().all()


def test_each_program_type_lays_out_intervals_and_cycles() -> None:
    cl = ParadigmSpec("CL", "ClosedLoop")
    ol = ParadigmSpec("OL", "OpenLoop")
    two = [_iv("CL", 1.0), _iv("OL", 0.5)]
    linear = _run(_spec(two, [cl, ol], {}, minutes=3.0)).truth.schedule
    assert linear == [(5_000, 1, 1, "CL"), (60_000, 2, 1, "OL")]          # holds its last Interval
    repeating = _run(_spec(two, [cl, ol], {}, minutes=3.0, program_type="Repeating")).truth.schedule
    assert repeating == [(5_000, 1, 1, "CL"), (60_000, 2, 1, "OL"), (90_000, 1, 2, "CL"),
                         (150_000, 2, 2, "OL")]
    constant = _run(_spec(two, [cl, ol], {}, minutes=3.0, program_type="Constant")).truth.schedule
    assert constant == [(5_000, 1, 1, "CL")]
    # Circadian follows the clock: 23:56 is in the first Interval, midnight starts Cycle 2.
    circadian = simulate(stress_spec(minutes=6.0))[3].truth.schedule
    assert circadian == [(5_000, 1, 1, "FR1"), (120_000, 2, 1, "FI1"), (240_000, 1, 2, "FR1")]


def test_the_truth_agrees_with_the_rows(choice: ChoiceRun) -> None:
    _, results = choice
    for result in results.values():
        rows, truth = result.rows, result.truth
        record_end = rows["Index"].to_numpy() * RECORD_MS - 1
        for w in range(1, 13):
            onsets = np.asarray(truth.activation_onset_ms[w], dtype=np.int64)
            counted = np.searchsorted(np.sort(onsets), record_end, side="right")
            assert np.array_equal(rows[f"Activations{w}"].to_numpy(np.int64), counted)
            assert truth.on_ms_total[w] == rows[f"OnMs{w}"].sum()
            end = truth.acclimation_end_ms[w]
            if end is not None:
                assert rows.at[_record(end), f"Phase{w}"] == "M" and rows.at[_record(end) - 1, f"Phase{w}"] == "A"
            # Under PR10 every Main Activation met exactly as many Events as its Requirement.
            event_starts = np.asarray([a for a, _ in truth.event_spans_ms[w]], dtype=np.int64)
            previous = None
            for onset, phase, met in zip(truth.activation_onset_ms[w], truth.activation_phase[w],
                                         truth.requirement_met[w]):
                if phase == "M" and met is not None and previous is not None:
                    between = ((event_starts > previous) & (event_starts <= onset)).sum()
                    assert between == met
                if met is not None or phase == "A":
                    previous = onset
        lit = sum(truth.on_ms_total.values())
        assert lit == rows[[f"OnMs{w}" for w in range(1, 13)]].to_numpy(np.int64).sum()


def test_the_event_driven_engine_matches_stepping_every_millisecond() -> None:
    spec = stress_spec(minutes=1.5)
    quick, slow = simulate(spec), simulate(spec, _step_every_ms=True)
    for dfm_id in quick:
        pd.testing.assert_frame_equal(quick[dfm_id].rows, slow[dfm_id].rows)
        assert quick[dfm_id].truth == slow[dfm_id].truth


def test_a_recording_is_determined_by_its_seed() -> None:
    spec = fixed_ratio_spec(minutes=2.0)
    first, again = simulate(spec)[1], simulate(spec)[1]
    pd.testing.assert_frame_equal(first.rows, again.rows)
    assert first.truth == again.truth
    other = simulate(fixed_ratio_spec(minutes=2.0, seed=8))[1]
    assert not other.rows["W1"].equals(first.rows["W1"])


# ---------------------------------------------------------------------------
# The ready-made specs
# ---------------------------------------------------------------------------

def test_the_choice_spec_acts_out_its_cases(choice: ChoiceRun) -> None:
    spec, results = choice
    closed_end = round(spec.dfms[0].intervals[0].minutes * 60_000)
    series = requirement_series(PR10, 40)
    for dfm_id, result in results.items():
        truth, rows = result.truth, result.rows
        assert truth.schedule == [(5_000, 1, 1, "Closed"), (closed_end, 2, 1, "PR10")]
        assert rows.at[_record(closed_end), "Paradigm"] == "PR10"
        assert rows.at[_record(closed_end) - 1, "Paradigm"] == "Closed"
        for chamber, case in CHOICE_PR_CASES.items():
            a, b = 2 * chamber - 1, 2 * chamber
            assert truth.activation_onset_ms[b] == [] and rows[f"OnMs{a}"].equals(rows[f"OnMs{b}"])
            main = [m for m, p in zip(truth.requirement_met[a], truth.activation_phase[a]) if p == "M" and m]
            acclimation = truth.activation_phase[a].count("A")
            if case == "healthy":
                assert truth.acclimation_end_ms[a] > closed_end and acclimation == 10
                assert main == series[:len(main)] and len(main) >= 3
            elif case == "drifting":
                assert not spec.dfms[dfm_id - 1].bouts[a]
                assert len(truth.event_spans_ms[a]) > 500 and len(main) >= 5
                # The LED is lit through much of Closed with no feeding at all.
                assert rows.loc[: _record(closed_end), f"OnMs{a}"].sum() > 30_000
            elif case == "dead_led":
                assert truth.baseline[a] > 40 and len(spec.dfms[dfm_id - 1].bouts[a]) > 50
                assert truth.activation_onset_ms[a] == [] and truth.event_spans_ms[a] == []
                assert truth.on_ms_total[a] == 0
            elif case == "never_training":
                assert truth.acclimation_end_ms[a] is None and acclimation == 6
                assert rows[f"Phase{a}"].iloc[-1] == "A"


def test_the_other_ready_made_specs_have_their_structure() -> None:
    each = one_of_each_spec(minutes=4.0)
    types = {d.dfm_id: [each.paradigms[iv.paradigm].type for iv in d.intervals] for d in each.dfms}
    assert types == {1: ["OpenLoop"], 2: ["ClosedLoop"], 3: ["FixedInterval"], 4: ["NonFeeding"],
                     5: ["ClosedLoop", "ProgressiveRatio"], 6: ["FixedRatio"]}
    results = simulate(each)
    open_rows = results[1].rows.iloc[25:]
    assert (open_rows[[f"OnMs{w}" for w in range(1, 13)]] == 200).all().all()
    assert (open_rows["Dark"] == 0).all()
    for dfm_id in (2, 3, 4, 6):
        assert results[dfm_id].rows["Activations1"].iloc[-1] > 0

    fr = simulate(fixed_ratio_spec(minutes=10.0))[1].truth
    for w in range(1, 13, 2):
        assert fr.activation_phase[w][:4] == ["A", "A", "A", "M"]
        assert set(m for m in fr.requirement_met[w] if m is not None) == {5}

    repeating = simulate(repeating_open_loop_spec())[1]
    assert [s[1:] for s in repeating.truth.schedule] == [(1, 1, "Open"), (2, 1, "Open"), (1, 2, "Open"),
                                                         (2, 2, "Open"), (1, 3, "Open"), (2, 3, "Open")]
    rows = repeating.rows.iloc[25:]
    on = rows["Interval"] == 1
    assert (rows.loc[on, "OnMs3"] == 200).all() and (rows.loc[~on, "OnMs3"] == 0).all()
    assert (rows[[f"Activations{w}" for w in range(1, 13)]] == 0).all().all()

    yoked = simulate(yoked_controls_spec(minutes=5.0))[1]
    assert yoked.truth.schedule == [(5_000, 1, 1, "Closed")] and YOKED_LINKAGE[6:] == (1, 1, 2, 2, 3, 3)
    for chamber in (1, 2, 3):
        a, partner = 2 * chamber - 1, 2 * (chamber + 3) - 1
        assert yoked.truth.activation_onset_ms[a]
        assert yoked.rows[f"OnMs{partner}"].equals(yoked.rows[f"OnMs{a}"])
        assert (yoked.rows[[f"Activations{partner}", f"Activations{partner + 1}"]] == 0).all().all()


def test_the_full_length_choice_spec_is_pymcus_example() -> None:
    # PyMCU/docs/DataFileColumns.md, "Program.txt".
    expected = (
        "Start Time: 10/16/2026 23:34:00\n"
        "End Time: 10/17/2026 05:34:00\n"
        "Duration: 360 min\n"
        "Default Linkage: 1,1,2,2,3,3,4,4,5,5,6,6\n"
        "Default Program Type: Linear\n"
        "Baseline: Yes\n"
        "\n"
        "Paradigms:\n"
        "Closed (ClosedLoop): Frequency 40 Hz, PulseWidth 8 ms, HoldMs 500 ms, MaxOnMs 2500 ms, EventGapMs 100 ms\n"
        "PR10 (ProgressiveRatio): Frequency 40 Hz, PulseWidth 8 ms, ActivationMs 500 ms, Series Arithmetic, "
        "FirstRequirement 1, Step 10, Cap 200, EventGapMs 100 ms, Acclimation 10\n"
        "\n"
        "***DFM 5***\n"
        "Linkage: 1,1,2,2,3,3,4,4,5,5,6,6\n"
        "Program Type: Linear\n"
        "(10/16/2026 23:34:00) Closed, Dark, 20,Off,20,Off,20,Off,20,Off,20,Off,20,Off, 30 min\n"
        "(10/17/2026 00:04:00) PR10, Dark, 20,Off,20,Off,20,Off,20,Off,20,Off,20,Off, 330 min\n")
    assert program_text(choice_pr_spec(minutes=360.0, closed_minutes=30.0, dfm_ids=(5,))) == expected


# ---------------------------------------------------------------------------
# Program.txt, Messages.csv and the instruction
# ---------------------------------------------------------------------------

def test_program_text_matches_the_guides_export_example() -> None:
    # The export example of the retired docs/dfm-firmware-4-data-guide.md (section
    # 7.1; kept as tests/fixtures/Program_v4.txt), without its Env Mon block, and
    # with each interval's Paradigm before Dark or Lit as PyMCU now writes it.
    closed = ParadigmSpec("Closed", "ClosedLoop", hold_ms=500)
    legacy = replace(closed, name="ClosedLoop-40Hz-8ms-Hold500-MaxOn0")
    choice: tuple[int | str, ...] = (20, OFF) * 6
    dfms = (DFMSpec(1, (_iv("Closed", 30, choice), _iv("PR10", 330, choice)), {}),
            DFMSpec(3, (_iv(legacy.name, 360, choice, dark=False),), {}),
            DFMSpec(2, (_iv("Idle", 360, (OFF,) * 12, dark=False),), {}))
    spec = RecordingSpec(start=datetime(2026, 10, 12, 20, 0), duration_min=360,
                         paradigms={p.name: p for p in (closed, PR10, legacy)}, dfms=dfms)
    expected = """Start Time: 10/12/2026 20:00:00
End Time: 10/13/2026 02:00:00
Duration: 360 min
Default Linkage: 1,1,2,2,3,3,4,4,5,5,6,6
Default Program Type: Linear
Baseline: Yes

Paradigms:
Closed (ClosedLoop): Frequency 40 Hz, PulseWidth 8 ms, HoldMs 500 ms, MaxOnMs 0 ms, EventGapMs 100 ms
PR10 (ProgressiveRatio): Frequency 40 Hz, PulseWidth 8 ms, ActivationMs 500 ms, Series Arithmetic, FirstRequirement 1, Step 10, Cap 200, EventGapMs 100 ms, Acclimation 10
ClosedLoop-40Hz-8ms-Hold500-MaxOn0 (ClosedLoop): Frequency 40 Hz, PulseWidth 8 ms, HoldMs 500 ms, MaxOnMs 0 ms, EventGapMs 100 ms
Idle (ClosedLoop): Frequency 40 Hz, PulseWidth 8 ms, HoldMs 0 ms, MaxOnMs 0 ms, EventGapMs 100 ms

***DFM 1***
Linkage: 1,1,2,2,3,3,4,4,5,5,6,6
Program Type: Linear
(10/12/2026 20:00:00) Closed, Dark, 20,Off,20,Off,20,Off,20,Off,20,Off,20,Off, 30 min
(10/12/2026 20:30:00) PR10, Dark, 20,Off,20,Off,20,Off,20,Off,20,Off,20,Off, 330 min

***DFM 2***
Linkage: 1,1,2,2,3,3,4,4,5,5,6,6
Program Type: Linear
(10/12/2026 20:00:00) Idle, Lit, Off,Off,Off,Off,Off,Off,Off,Off,Off,Off,Off,Off, 360 min

***DFM 3***
Linkage: 1,1,2,2,3,3,4,4,5,5,6,6
Program Type: Linear
(10/12/2026 20:00:00) ClosedLoop-40Hz-8ms-Hold500-MaxOn0, Lit, 20,Off,20,Off,20,Off,20,Off,20,Off,20,Off, 360 min
"""
    assert program_text(spec) == expected


def test_program_text_lists_series_settings_and_omits_intervals_past_the_end() -> None:
    expo = ParadigmSpec("Expo", "ProgressiveRatio", activation_ms=500, series="Exponential", scale=2.5,
                        rate=0.25, cap=300, acclimation=2)
    spec = _spec([_iv("Expo", 1.5), _iv("Expo", 0.5), _iv("Expo", 2.0)], [expo], {}, minutes=2.0)
    text = program_text(spec)
    assert ("Expo (ProgressiveRatio): Frequency 40 Hz, PulseWidth 8 ms, ActivationMs 500 ms, Series Exponential, "
            "Scale 2.5, Rate 0.25, Cap 300, EventGapMs 100 ms, Acclimation 2\n") in text
    assert text.endswith("(10/16/2026 12:00:00) Expo, Dark, 20,Off,Off,Off,Off,Off,Off,Off,Off,Off,Off,Off, 1.5 min\n"
                         "(10/16/2026 12:01:30) Expo, Dark, 20,Off,Off,Off,Off,Off,Off,Off,Off,Off,Off,Off, 0.5 min\n")


def test_messages_text_has_pymcus_rows() -> None:
    spec = fixed_ratio_spec(minutes=2.0)
    text = messages_text(spec, index_gaps={1: [(100, 7)]})
    assert text == ("DFM,Date,Time,Millisecond,Sample,Message,MsgType\n"
                    "0,10/16/2026,23:34:00,0.0,0,Recording started,Unknown\n"
                    '1,10/16/2026,23:34:20,200.0,108,"Lost 7 records (1.4 s): Index 101 to 107, from 23:34:20.",Error\n'
                    "0,10/16/2026,23:36:00,0.0,0,Recording ended,Unknown\n")


def test_the_instruction_frame_is_the_shared_golden_one() -> None:
    # GOLDEN_INSTRUCTION of PyMCU tests/test_protocol.py and DFMV3_V2 tests/test_ledcontrol.c.
    golden = ("05f701020401010028000800640003000001f400000000000000010000000a000000c8"
              "0014fffeffff001e03ff03ff03ff03ff03ff03ff03ff03fffffff05f")
    pr = ParadigmSpec("PR10", "ProgressiveRatio", activation_ms=500, first_requirement=1, step=10, cap=200,
                      acclimation=3)
    frame = paradigm_frame(5, 258, pr, keep=True, dark=True, thresholds=(20, 0xFFFE, 0xFFFF, 30) + (1023,) * 8)
    assert frame.hex() == golden


# ---------------------------------------------------------------------------
# The files
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("blank", [False, True])
def test_a_written_recording_reloads_with_the_documented_dtypes(tmp_path: Path, blank: bool) -> None:
    spec = fixed_ratio_spec(minutes=3.0)
    results = simulate(spec)
    files = write_v4_recording(tmp_path, spec, blank_missing=blank, results=results)
    assert sorted(files) == ["DFM1_0.csv", "Messages.csv", "Program.txt"]
    text = files["DFM1_0.csv"].read_text()
    lines = text.splitlines()
    assert lines[0] == ",".join(COLUMNS) and "\r" not in text
    assert all(line.count(",") == 89 for line in lines)
    assert (",NA," in text) != blank
    rows = read_v4_csv(files["DFM1_0.csv"])
    assert len(rows.columns) == 90
    for column, dtype in DATA_FILE_DTYPES.items():
        assert str(rows[column].dtype) == dtype, column
    pd.testing.assert_frame_equal(rows, results[1].rows)
    assert files["Program.txt"].read_text() == program_text(spec)


def test_an_injected_gap_shows_in_index_not_sample_and_files_split(tmp_path: Path) -> None:
    spec = fixed_ratio_spec(minutes=3.0)
    results = simulate(spec)
    files = write_v4_recording(tmp_path, spec, split_hours=1 / 60, index_gaps={1: [(100, 7)]},
                               error_bits={1: [(50, 4)]}, results=results)
    names = sorted(n for n in files if n.startswith("DFM"))
    assert names == ["DFM1_0.csv", "DFM1_1.csv", "DFM1_2.csv"]
    parts = [read_v4_csv(files[n]) for n in names]
    assert [len(p) for p in parts] == [293, 300, 300]
    rows = pd.concat(parts, ignore_index=True)
    assert rows["Sample"].tolist() == list(range(1, 894))
    index = rows["Index"].tolist()
    assert index == list(range(1, 101)) + list(range(108, 901))
    assert rows.loc[rows["Index"] == 50, "Error"].tolist() == [4] and rows["Error"].sum() == 4
    expected = results[1].rows.loc[results[1].rows["Index"].isin(index)].reset_index(drop=True)
    expected["Sample"] = rows["Sample"]
    expected.loc[expected["Index"] == 50, "Error"] = 4
    pd.testing.assert_frame_equal(rows, expected)
    assert ('108,"Lost 7 records (1.4 s): Index 101 to 107, from 23:34:20.",Error'
            in files["Messages.csv"].read_text())


def test_a_generation_2_recording_has_every_paradigm_column_na(tmp_path: Path) -> None:
    spec = fixed_ratio_spec(minutes=2.0)
    results = simulate(spec)
    files = write_gen2_recording(tmp_path, spec, results=results)
    rows = read_v4_csv(files["DFM1_0.csv"])
    assert rows[list(PARADIGM_COLUMNS)].isna().all().all()
    data = [c for c in COLUMNS if c not in PARADIGM_COLUMNS]
    pd.testing.assert_frame_equal(rows[data], results[1].rows[data])
    assert (rows[[f"W{w}" for w in range(1, 13)]] <= 1023).all().all()


# ---------------------------------------------------------------------------
# Restarts (PyMCU 529b926)
# ---------------------------------------------------------------------------

def _wells(prefix: str) -> list[str]:
    return [f"{prefix}{w}" for w in range(1, 13)]


def _on_ms_from_spans(result: SimResult, linkage: tuple[int, ...]) -> np.ndarray:
    """Each row's OnMs from the truth's own lit spans, after Linkage."""
    index = result.rows["Index"].to_numpy(dtype=np.int64)
    n_ms = int(index.max()) * RECORD_MS
    out = np.zeros((len(index), 12), dtype=np.int64)
    for w in range(12):
        edges = np.zeros(n_ms + 1, dtype=np.int64)
        for j in (j for j in range(12) if linkage[j] == linkage[w]):
            for a, b in result.truth.lit_spans_ms[j + 1]:
                edges[a] += 1
                edges[b] -= 1
        lit = np.cumsum(edges)[:n_ms] > 0
        out[:, w] = lit.reshape(-1, RECORD_MS).sum(axis=1)[index - 1]
    return out


@pytest.fixture(scope="module")
def restart_run() -> ChoiceRun:
    spec = family_specs()["restart"]
    return spec, simulate(spec)


def test_a_restart_leaves_a_gap_then_idle_rows_then_the_instruction_afresh(restart_run: ChoiceRun) -> None:
    spec, results = restart_run
    n = int(spec.duration_min * 60 * 5)
    for d in spec.dfms:
        result = results[d.dfm_id]
        rows, truth = result.rows, result.truth
        (r,) = d.restarts
        down, resume = round(r.at_s * 1000), round((r.at_s + r.down_s + r.resend_s) * 1000)
        missing, idle_rows = round(r.down_s * 5), round(r.resend_s * 5)
        assert truth.restarts_ms == [(down, resume, missing)]
        # No records while down: Index and the times jump, Sample counts the rows written.
        after = down // RECORD_MS
        assert rows["Index"].tolist() == list(range(1, after + 1)) + list(range(after + missing + 1, n + 1))
        assert rows["Sample"].tolist() == list(range(1, len(rows) + 1))
        stamps = pd.to_datetime(rows["Date"] + " " + rows["Time"], format="%m/%d/%Y %H:%M:%S") \
            + pd.to_timedelta(rows["MSec"], unit="ms")
        assert (stamps == pd.Timestamp(V4_START) + pd.to_timedelta(rows["Index"] * RECORD_MS, unit="ms")).all()
        # Then the idle rows: no instruction PyMCU can name, every Well Off, the totals carried.
        idle = rows.iloc[after: after + idle_rows]
        assert idle["Index"].tolist() == list(range(after + missing + 1, resume // RECORD_MS + 1))
        assert idle[["Interval", "Cycle", "Paradigm"]].isna().all().all()
        assert (idle[["Dark", "OptoFreq", "OptoPW", "OptoCol1"] + _wells("OnMs")] == 0).all().all()
        assert (idle[_wells("Phase")] == "M").all().all()
        assert idle[_wells("Requirement") + _wells("Progress")].isna().all().all()
        before, resumed = rows.iloc[after - 1], rows.iloc[after + idle_rows]
        totals = before[_wells("Activations")].to_numpy(dtype=np.int64)
        assert (idle[_wells("Activations")].to_numpy(dtype=np.int64) == totals).all()
        # Then the instruction in force, applied again: the same Interval, Cycle and Paradigm.
        for column in ("Interval", "Cycle", "Paradigm", "Dark", "OptoFreq", "OptoPW"):
            assert resumed[column] == before[column], column
        # The rows are the truth's, the restart included.
        record_end = rows["Index"].to_numpy() * RECORD_MS - 1
        for w in range(1, 13):
            onsets = np.sort(np.asarray(truth.activation_onset_ms[w], dtype=np.int64))
            assert np.array_equal(rows[f"Activations{w}"].to_numpy(np.int64),
                                  np.searchsorted(onsets, record_end, side="right"))
            assert truth.on_ms_total[w] == rows[f"OnMs{w}"].sum()
            # Dark, and nothing counted, from the outage to the resend.
            assert all(b <= down or a >= resume for a, b in truth.lit_spans_ms[w])
            assert all(b <= down or a >= resume for a, b in truth.event_spans_ms[w])
            assert not any(down <= t < resume for t in onsets)
        assert np.array_equal(rows[_wells("OnMs")].to_numpy(np.int64), _on_ms_from_spans(result, d.linkage))


def test_a_ratio_well_restarting_in_main_starts_its_acclimation_again(restart_run: ChoiceRun) -> None:
    spec, results = restart_run
    rows, truth = results[1].rows, results[1].truth
    ((down, resume, _missing),) = truth.restarts_ms
    before = rows.iloc[down // RECORD_MS - 1]
    resumed = rows.loc[rows["Index"] == resume // RECORD_MS + 1].iloc[0]
    for w in range(1, 13, 2):
        assert before[f"Phase{w}"] == "M"
        # Every Well as under a new Paradigm: Acclimation, the first Requirement, no Progress.
        assert resumed[f"Phase{w}"] == "A" and resumed[f"Requirement{w}"] == 3 and resumed[f"Progress{w}"] == 0
        after = [(p, m) for t, p, m in zip(truth.activation_onset_ms[w], truth.activation_phase[w],
                                           truth.requirement_met[w]) if t >= resume]
        assert [p for p, _ in after[:2]] == ["A", "A"] and {p for p, _ in after[2:]} == {"M"}
        assert [m for _, m in after] == [None, None] + [3] * (len(after) - 2) and len(after) > 4
        # acclimation_end_ms is the first Acclimation's.
        assert truth.acclimation_end_ms[w] < down


def test_a_closed_loop_well_restarting_mid_bout_relights_when_the_instruction_returns(
        restart_run: ChoiceRun) -> None:
    spec, results = restart_run
    truth = results[2].truth
    (r,) = spec.dfms[1].restarts
    ((down, resume, _missing),) = truth.restarts_ms
    (held,) = [b for b in spec.dfms[1].bouts[1] if b.start_s < r.at_s < b.start_s + b.duration_s]
    start, end = round(held.start_s * 1000), round((held.start_s + held.duration_s) * 1000)
    assert end > resume
    # Lit from the bout's start until the DFM went down, then again, as a new Activation, from
    # the resend to the bout's end and its Hold.
    assert (start, down) in truth.lit_spans_ms[1] and (resume, end + 500) in truth.lit_spans_ms[1]
    assert start in truth.activation_onset_ms[1] and resume in truth.activation_onset_ms[1]
    # The Event in progress ended with the outage; the contact after it is a new one, which ends
    # EventGapMs after its last contact.
    assert (start, down) in truth.event_spans_ms[1] and (resume, end - 1 + 100) in truth.event_spans_ms[1]


def test_restart_messages_follow_the_first_record_after_the_outage(restart_run: ChoiceRun) -> None:
    spec, _results = restart_run
    text = messages_text(spec)
    for d in spec.dfms:
        (r,) = d.restarts
        after, missing = round(r.at_s * 5), round(r.down_s * 5)
        when = V4_START + pd.Timedelta(milliseconds=(after + missing + 1) * RECORD_MS)
        stamp = f"{d.dfm_id},{when:%m/%d/%Y},{when:%H:%M:%S},{when.microsecond / 1000}"
        started, finished = restart_messages(d.dfm_id, after, after, missing)
        assert f"{stamp},1,{started},Error\n{stamp},{after},{finished},Error\n" in text


def test_a_second_restart_counts_from_the_first(tmp_path: Path) -> None:
    spec = family_specs(minutes=6.0)["restart"]
    dfm = replace(spec.dfms[0], restarts=(Restart(60.0, down_s=2.0), Restart(200.0)))
    spec = replace(spec, dfms=(dfm,))
    result = simulate(spec)[1]
    assert result.truth.restarts_ms == [(60_000, 63_000, 10), (200_000, 206_000, 25)]
    index = result.rows["Index"].tolist()
    assert index == list(range(1, 301)) + list(range(311, 1001)) + list(range(1026, 1801))
    # The DFM's own count began again after the first: it had reached 1000 - 310 = 690.
    text = messages_text(spec)
    assert restart_messages(1, 300, 300, 10)[0] in text and restart_messages(1, 690, 1000, 25)[0] in text
    assert "Index 1001 to 1025 are missing." in text


def test_restarts_are_validated() -> None:
    spec = family_specs(minutes=6.0)["restart"]
    dfm = spec.dfms[0]
    bad = [(Restart(3.0),), (Restart(8.0),), (Restart(100.0, down_s=0.0),), (Restart(100.0, resend_s=0.05),),
           (Restart(359.0),), (Restart(100.0), Restart(110.0))]
    for restarts in bad:
        with pytest.raises(ValueError):
            simulate(replace(spec, dfms=(replace(dfm, restarts=restarts),)))
    # A restart is no part of LEDControl.c.
    with pytest.raises(ValueError, match="restart"):
        firmware_inputs(spec, 1)


def test_the_event_driven_engine_matches_stepping_every_millisecond_through_a_restart() -> None:
    spec = family_specs(minutes=1.5)["restart"]
    quick, slow = simulate(spec), simulate(spec, _step_every_ms=True)
    for dfm_id in quick:
        assert quick[dfm_id].truth.restarts_ms
        pd.testing.assert_frame_equal(quick[dfm_id].rows, slow[dfm_id].rows)
        assert quick[dfm_id].truth == slow[dfm_id].truth


def test_a_written_restart_reloads_as_simulated(tmp_path: Path, restart_run: ChoiceRun) -> None:
    spec, results = restart_run
    files = write_v4_recording(tmp_path, spec, results=results)
    for d in spec.dfms:
        pd.testing.assert_frame_equal(read_v4_csv(files[f"DFM{d.dfm_id}_0.csv"]), results[d.dfm_id].rows)


# ---------------------------------------------------------------------------
# Faults: the wrong Program.txt and a misreported LED
# ---------------------------------------------------------------------------

def test_the_fault_arguments_change_nothing_when_absent(tmp_path: Path) -> None:
    spec = family_specs(minutes=3.0)["restart"]
    results = simulate(spec)
    plain = write_v4_recording(tmp_path / "plain", spec, results=results)
    none = write_v4_recording(tmp_path / "none", spec, results=results, program_paradigms=None, on_ms_faults=None)
    empty = write_v4_recording(tmp_path / "empty", spec, results=results, program_paradigms={}, on_ms_faults={})
    assert sorted(plain) == sorted(none) == sorted(empty)
    for name, path in plain.items():
        assert none[name].read_bytes() == path.read_bytes() == empty[name].read_bytes(), name


def test_program_paradigms_describe_a_program_that_did_not_run(tmp_path: Path) -> None:
    spec = family_specs(minutes=3.0)["fixed_interval"]
    results = simulate(spec)
    files = write_v4_recording(tmp_path, spec, results=results,
                               program_paradigms={"FI10s": replace(FI10S, lockout_ms=20000)})
    right = program_text(spec)
    assert right.count("LockoutMs 10000 ms") == 1
    assert files["Program.txt"].read_text() == right.replace("LockoutMs 10000 ms", "LockoutMs 20000 ms")
    # The data are the program that ran.
    for dfm_id in (1, 2):
        pd.testing.assert_frame_equal(read_v4_csv(files[f"DFM{dfm_id}_0.csv"]), results[dfm_id].rows)
    with pytest.raises(ValueError):
        write_v4_recording(tmp_path / "other", spec, results=results,
                           program_paradigms={"Other": replace(FI10S, name="Other")})


def test_an_on_ms_fault_sets_on_ms_and_the_led_bit_and_nothing_else(tmp_path: Path) -> None:
    spec = family_specs(minutes=3.0)["closed_max"]
    results = simulate(spec)
    clean = results[1].rows
    files = write_v4_recording(tmp_path, spec, results=results,
                               on_ms_faults={1: [OnMsFault(1, 60.0, 90.0, 0), OnMsFault(4, 100.0, 101.0, 200)]})
    rows = read_v4_csv(files["DFM1_0.csv"])
    dead = ((clean["Index"] > 300) & (clean["Index"] <= 450)).to_numpy()
    stuck = ((clean["Index"] > 500) & (clean["Index"] <= 505)).to_numpy()
    assert clean.loc[dead, "OnMs1"].sum() > 0 and (clean.loc[stuck, "OnMs4"] < 200).any()
    expected = clean.copy()
    expected.loc[dead, "OnMs1"] = 0
    expected.loc[dead, "OptoCol1"] = expected.loc[dead, "OptoCol1"] & ~1
    expected.loc[stuck, "OnMs4"] = 200
    expected.loc[stuck, "OptoCol1"] = expected.loc[stuck, "OptoCol1"] | (1 << 3)
    # Linkage is not applied: Well 1's partner keeps its light, and so does Well 4's.
    pd.testing.assert_frame_equal(rows, expected)
    with pytest.raises(ValueError):
        write_v4_recording(tmp_path / "bad", spec, results=results, on_ms_faults={1: [OnMsFault(13, 0, 1, 0)]})


# ---------------------------------------------------------------------------
# The Paradigm Family specs
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def families() -> dict[str, ChoiceRun]:
    return {key: (spec, simulate(spec)) for key, spec in family_specs().items()}


def _contact_ms(bout: Bout) -> tuple[int, int]:
    """[start, end) of a bout's contact: every bout is well above the threshold."""
    start = int(round(bout.start_s * 1000))
    return start, start + int(round(bout.duration_s * 1000))


def test_the_family_specs_run_their_paradigms_for_twenty_minutes(families: dict[str, ChoiceRun]) -> None:
    assert tuple(families) == FAMILY_SPEC_KEYS
    types = {key: {d.dfm_id: [spec.paradigms[iv.paradigm].type for iv in d.intervals] for d in spec.dfms}
             for key, (spec, _results) in families.items()}
    assert types == {"closed_max": {1: ["ClosedLoop"]},
                     "fixed_interval": {1: ["FixedInterval"], 2: ["FixedInterval"]},
                     "non_feeding": {1: ["NonFeeding"], 2: ["NonFeeding"]},
                     "exponential": {1: ["ProgressiveRatio"]},
                     "restart": {1: ["FixedRatio"], 2: ["ClosedLoop"]}}
    assert {key: [d.dfm_id for d in spec.dfms if d.restarts] for key, (spec, _r) in families.items()} == \
        {"closed_max": [], "fixed_interval": [], "non_feeding": [], "exponential": [], "restart": [1, 2]}
    for spec, _results in families.values():
        assert spec.duration_min == 20.0 and all(d.linkage == CHOICE_LINKAGE for d in spec.dfms)
    assert families["fixed_interval"][0].paradigms == {"FI10s": FI10S, "FI500": FI500}
    assert families["non_feeding"][0].paradigms == {"NF1000": NF1000, "NFMax5s": NF_MAX5S}
    assert families["restart"][0].paradigms["FR3"] == FR3


def test_closed_max_cuts_off_the_long_bouts(families: dict[str, ChoiceRun]) -> None:
    spec, results = families["closed_max"]
    truth = results[1].truth
    assert spec.paradigms == {"CL500": CL500}
    for w in range(1, 13, 2):
        spans = truth.lit_spans_ms[w]
        lengths = [b - a for a, b in spans]
        # Each on-period is one Activation, and none outlasts MaxOnMs.
        assert [a for a, _b in spans] == truth.activation_onset_ms[w]
        assert max(lengths) == 2000 and lengths.count(2000) >= 10 and sum(x < 2000 for x in lengths) >= 10
        # After a cut-off the light stays off until the bout ends: a cut-off span ends inside contact.
        contact = [_contact_ms(b) for b in spec.dfms[0].bouts[w]]
        cut = [b for a, b in spans if b - a == 2000]
        assert sum(any(c0 < b < c1 for c0, c1 in contact) for b in cut) >= len(cut) // 2


def test_fixed_interval_keeps_its_lockout_and_its_length(families: dict[str, ChoiceRun]) -> None:
    spec, results = families["fixed_interval"]
    long, short = results[1].truth, results[2].truth
    repeats = continuous = 0
    for w in range(1, 13, 2):
        onsets = np.diff(long.activation_onset_ms[w])
        assert (onsets >= 10_000).all()
        repeats += int((onsets == 10_000).sum())           # contact held through a lockout
        assert all(b - a == 1000 for a, b in long.lit_spans_ms[w])
        # Many bouts start in a lockout and earn nothing.
        starts = [_contact_ms(b)[0] for b in spec.dfms[0].bouts[w]]
        locked = [s for s in starts if any(t < s < t + 10_000 for t in long.activation_onset_ms[w])]
        assert len(locked) >= 20
        phases = short.activation_phase[w]
        assert phases[:3] == ["A"] * 3 and set(phases[3:]) == {"M"}
        main = [t for t, p in zip(short.activation_onset_ms[w], phases) if p == "M"]
        # LockoutMs shorter than ActivationMs: held contact relights every 1.5 s, without a gap.
        continuous += int((np.diff(main) == 1500).sum())
        assert max(b - a for a, b in short.lit_spans_ms[w]) >= 3000
    assert repeats >= 6 and continuous >= 30


def test_non_feeding_is_dark_during_contact_and_relights_after_the_delay(families: dict[str, ChoiceRun]) -> None:
    spec, results = families["non_feeding"]
    for d, max_on in zip(spec.dfms, (None, 5000)):
        truth = results[d.dfm_id].truth
        for w in range(1, 13, 2):
            contact = [_contact_ms(b) for b in d.bouts[w]]
            spans = truth.lit_spans_ms[w]
            assert spans[0][0] == 25 * RECORD_MS            # lit at the Interval's start
            assert [a for a, _b in spans] == truth.activation_onset_ms[w]
            for a, b in spans:
                assert all(b <= c0 or a >= c1 for c0, c1 in contact)
            ends = {c1 for _c0, c1 in contact}
            assert all(a - 1000 in ends for a, _b in spans[1:])
            # Some gaps between bouts are shorter than the delay, so it stays dark through them.
            assert sum(n0 - c1 < 1000 for (_c0, c1), (n0, _n1) in zip(contact, contact[1:])) >= 5
            if max_on:
                lengths = [b - a for a, b in spans]
                assert max(lengths) == max_on and lengths.count(max_on) >= 3


def test_the_exponential_spec_climbs_its_series(families: dict[str, ChoiceRun]) -> None:
    _spec, results = families["exponential"]
    truth = results[1].truth
    series = requirement_series(PR_EXP, 30)
    assert series[:6] == [1, 2, 4, 6, 9, 12]
    for w in range(1, 13, 2):
        phases, met = truth.activation_phase[w], truth.requirement_met[w]
        assert phases[:2] == ["A", "A"] and set(phases[2:]) == {"M"}
        main = met[2:]
        assert main == series[:len(main)] and len(main) >= 12


# ---------------------------------------------------------------------------
# Against PyMCU's own code, when its checkout is beside this one
# ---------------------------------------------------------------------------

def _pymcu(code: str, stdin: str) -> str:
    if not (PYMCU_DIR / "Program.py").is_file():
        pytest.skip(f"PyMCU not found at {PYMCU_DIR} (set PYMCU_DIR)")
    # -B: nothing is written into PyMCU's checkout.
    done = subprocess.run([sys.executable, "-B", "-c", code], input=stdin, capture_output=True, text=True,
                          encoding="utf-8", cwd=PYMCU_DIR, timeout=120)
    assert done.returncode == 0, done.stderr + done.stdout
    return done.stdout


def _program_file(spec: RecordingSpec) -> str:
    """A PyMCU Program file (ProgramFormat.md) for *spec*, to load with PyMCU."""
    linkage = {tuple(d.linkage) for d in spec.dfms}
    types = {d.program_type for d in spec.dfms}
    lines = ["[General]", f"StartTime: {spec.start:%m/%d/%Y %H:%M:%S}", f"ExpDurationMin: {spec.duration_min!r}",
             "Linkage: " + ",".join(map(str, linkage.pop() if len(linkage) == 1 else IDENTITY)),
             f"ProgramType: {types.pop() if len(types) == 1 else 'Linear'}",
             f"Baseline: {'Yes' if spec.baseline else 'No'}"]
    used = {iv.paradigm for d in spec.dfms for iv in d.intervals}
    for name in sorted(used):
        p = spec.paradigms.get(name, ParadigmSpec(name, "ClosedLoop"))
        settings = {"ClosedLoop": [("HoldMs", p.hold_ms), ("MaxOnMs", p.max_on_ms), ("EventGapMs", p.event_gap_ms)],
                    "FixedInterval": [("ActivationMs", p.activation_ms), ("LockoutMs", p.lockout_ms),
                                      ("EventGapMs", p.event_gap_ms), ("Acclimation", p.acclimation)],
                    "NonFeeding": [("OnDelayMs", p.on_delay_ms), ("MaxOnMs", p.max_on_ms)],
                    "ProgressiveRatio": [("ActivationMs", p.activation_ms), ("Series", p.series)]
                    + ([("Scale", p.scale), ("Rate", p.rate)] if p.series == "Exponential"
                       else [("FirstRequirement", p.first_requirement), ("Step", p.step)])
                    + [("Cap", p.cap), ("EventGapMs", p.event_gap_ms), ("Acclimation", p.acclimation)],
                    "FixedRatio": [("ActivationMs", p.activation_ms), ("Ratio", p.ratio),
                                   ("EventGapMs", p.event_gap_ms), ("Acclimation", p.acclimation)],
                    "OpenLoop": []}[p.type]
        lines += [f"[Paradigm {name}]", f"Type: {p.type}", f"Frequency: {p.frequency}",
                  f"PulseWidth: {p.pulse_width}"] + [f"{k}: {v}" for k, v in settings]
    for d in spec.dfms:
        lines += ["[DFM]", f"ID: {d.dfm_id}", f"ProgramType: {d.program_type}",
                  "Linkage: " + ",".join(map(str, d.linkage))]
        lines += [f"Interval: {iv.paradigm}, {'Dark' if iv.dark else 'Lit'}, "
                  + ",".join(str(t) for t in iv.thresholds) + f", {iv.minutes!r}" for iv in d.intervals]
    return "\n".join(lines) + "\n"


_EXPORT = """
import sys
sys.path.insert(0, ".")
import Program
program = Program.MCUProgram()
if not program.LoadProgram(sys.stdin.read().splitlines(True), None):
    sys.exit("load error: " + program.loadError)
sys.stdout.write(str(program))
"""


@pytest.mark.parametrize("make", [choice_pr_spec, one_of_each_spec, stress_spec, repeating_open_loop_spec,
                                  yoked_controls_spec])
def test_program_text_is_pymcus_own_export(make) -> None:
    spec = make()
    assert _pymcu(_EXPORT, _program_file(spec)) == program_text(spec)


_ROWS = """
import datetime, json, sys
sys.path.insert(0, ".")
import Enums, Message, MessagesList, StatusPacket
data = json.loads(sys.stdin.read())
start = datetime.datetime.fromisoformat(data["start"])
out = [StatusPacket.DATA_COLUMNS + "," + StatusPacket.PARADIGM_COLUMNS + "\\n"]
for r in data["rows"]:
    sp = StatusPacket.StatusPacket(r["sample"], data["dfm"], Enums.DFMTYPE.PLETCHERV3)
    sp.packetTime = start + datetime.timedelta(seconds=r["index"] * StatusPacket.V3_PACKET_INTERVAL)
    sp.recordIndex = r["index"]
    sp.analogValues = r["w"]
    sp.temp, sp.humidity, sp.lux, sp.voltsIn = 0, 0, 0, 5.01
    sp.darkStatus, sp.optoFrequency, sp.optoPulseWidth = r["dark"], r["freq"], r["pw"]
    sp.optoState1, sp.optoState2, sp.errorFlags = r["led"], 0, r["error"]
    sp.isV4 = True
    sp.paradigmType, sp.phases, sp.onMs = r["type"], r["phases"], r["on_ms"]
    sp.activationTotals, sp.requirement, sp.progress = r["activations"], r["requirement"], r["progress"]
    sp.interval, sp.cycle, sp.paradigmName = r["interval"], r["cycle"], r["paradigm"]
    sp.isBeforeProgram = r["before"]
    line = sp.GetDataBufferPrintPacket()
    # The fixtures are 90-column files, as PyMCU 4.0.0 wrote before its Baseline
    # columns; a PyMCU that writes them adds 12 more, NA for a record with no
    # Baseline set, which are checked and left off (test_adaptive_baseline_sim.py
    # checks the columns themselves against PyMCU's simulator).
    if hasattr(StatusPacket, "BASELINE_COLUMNS"):
        line, extra = line.rstrip("\\n").rsplit(",", 12)[0], line.rstrip("\\n").split(",")[-12:]
        assert extra == ["NA"] * 12, extra
        line += "\\n"
    out.append(line)
messages = MessagesList.MessageList()
messages.AddMessage(Message.Message(0, start, 0, "Recording started", Enums.MESSAGETYPE.ANNOTATION, -99))
for dfm, when, sample, text in data.get("errors", []):
    messages.AddMessage(Message.Message(dfm, datetime.datetime.fromisoformat(when), sample, text,
                                        Enums.MESSAGETYPE.ERROR, -99))
for dfm, index, after in data["gaps"]:
    # The wording and date of DFM._ReportLostRecords (DFM itself needs pyserial).
    count = index - after - 1
    when = start + datetime.timedelta(seconds=(index - count) * StatusPacket.V3_PACKET_INTERVAL)
    span = "Index {}".format(after + 1) if count == 1 else "Index {} to {}".format(after + 1, index - 1)
    text = "Lost {} record{} ({:.1f} s): {}, from {}.".format(
        count, "" if count == 1 else "s", count * StatusPacket.V3_PACKET_INTERVAL, span, when.strftime("%H:%M:%S"))
    messages.AddMessage(Message.Message(dfm, when, index, text, Enums.MESSAGETYPE.ERROR, -99))
end = start + datetime.timedelta(minutes=data["minutes"])
messages.AddMessage(Message.Message(0, end, 0, "Recording ended", Enums.MESSAGETYPE.ANNOTATION, -99))
sys.stdout.write("".join(out) + "@@\\n" + messages.GetMessageStringForFile())
"""


def _pymcu_rows(rows: pd.DataFrame, spec: RecordingSpec) -> list[dict]:
    """The data file's rows as the values PyMCU's StatusPacket holds: an
    Interval 0 row is from before the program, an NA one from an instruction
    PyMCU cannot name (a restart's idle rows)."""
    types = {name: ["OpenLoop", "ClosedLoop", "FixedInterval", "NonFeeding", "ProgressiveRatio",
                    "FixedRatio"].index(p.type) for name, p in spec.paradigms.items()}

    def wells(prefix: str) -> list[list[int]]:
        return rows[[f"{prefix}{w}" for w in range(1, 13)]].fillna(0).astype(int).values.tolist()

    named = rows["Paradigm"].notna().to_numpy()
    return [
        {"sample": int(r.Sample), "index": int(r.Index), "w": w, "dark": int(r.Dark), "freq": int(r.OptoFreq),
         "pw": int(r.OptoPW), "led": int(r.OptoCol1), "error": int(r.Error),
         "type": types.get(r.Paradigm, 1) if known else 1, "phases": [int(p == "A") for p in ph], "on_ms": on,
         "activations": act, "requirement": req, "progress": prog,
         "interval": None if pd.isna(r.Interval) or r.Interval == 0 else int(r.Interval) - 1,
         "before": bool(not pd.isna(r.Interval) and r.Interval == 0),
         "cycle": None if pd.isna(r.Cycle) else int(r.Cycle), "paradigm": r.Paradigm if known else None}
        for r, known, w, ph, on, act, req, prog in zip(
            rows.itertuples(index=False), named, wells("W"), rows[[f"Phase{w}" for w in range(1, 13)]].values.tolist(),
            wells("OnMs"), wells("Activations"), wells("Requirement"), wells("Progress"))]


def test_rows_and_messages_are_pymcus_own_formatting(tmp_path: Path) -> None:
    # DFM 2 of the stress spec: ratio and other types, Cycles, and midnight.
    spec = stress_spec(minutes=6.0)
    spec = replace(spec, dfms=(spec.dfms[1],))
    files = write_v4_recording(tmp_path, spec, index_gaps={2: [(400, 3)]}, error_bits={2: [(30, 2)]})
    rows = read_v4_csv(files["DFM2_0.csv"])
    payload = {"start": spec.start.isoformat(), "dfm": 2, "minutes": spec.duration_min, "gaps": [[2, 404, 400]],
               "rows": _pymcu_rows(rows, spec)}
    data_text, message_text = _pymcu(_ROWS, json.dumps(payload)).split("@@\n")
    assert files["DFM2_0.csv"].read_text() == data_text
    assert files["Messages.csv"].read_text() == message_text
    assert rows["Date"].nunique() == 2


def test_a_restarts_rows_and_messages_are_pymcus_own_formatting(tmp_path: Path) -> None:
    """The lead-in's and the idle rows' NA Interval, Cycle and Paradigm, and
    the restart's two errors, as PyMCU writes them."""
    spec = family_specs(minutes=6.0)["restart"]
    spec = replace(spec, dfms=(spec.dfms[0],))
    files = write_v4_recording(tmp_path, spec)
    rows = read_v4_csv(files["DFM1_0.csv"])
    assert rows["Interval"].isna().sum() == 5 and rows["Paradigm"].isna().sum() == 25 + 5
    with open(files["Messages.csv"], newline="", encoding="utf-8") as f:
        errors = [[int(m["DFM"]), (datetime.strptime(f"{m['Date']} {m['Time']}", "%m/%d/%Y %H:%M:%S")
                                   + pd.Timedelta(milliseconds=float(m["Millisecond"]))).isoformat(),
                   int(m["Sample"]), m["Message"]]
                  for m in csv.DictReader(f) if m["MsgType"] == "Error"]
    payload = {"start": spec.start.isoformat(), "dfm": 1, "minutes": spec.duration_min, "gaps": [],
               "errors": errors, "rows": _pymcu_rows(rows, spec)}
    assert len(payload["errors"]) == 2
    data_text, message_text = _pymcu(_ROWS, json.dumps(payload)).split("@@\n")
    assert files["DFM1_0.csv"].read_text() == data_text
    assert files["Messages.csv"].read_text() == message_text


def _pymcu_message_format(function: str) -> str:
    """The text PyMCU's ``DFM.<function>`` formats its message from, read
    from the checkout's source (the DFM module itself needs pyserial)."""
    path = PYMCU_DIR / "DFM.py"
    if not path.is_file():
        pytest.skip(f"PyMCU not found at {PYMCU_DIR} (set PYMCU_DIR)")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function)
    (text,) = [n.func.value.value for n in ast.walk(body)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "format"
               and isinstance(n.func.value, ast.Constant) and str(n.func.value.value).startswith("DFM {}")]
    return text


def test_restart_messages_are_worded_as_pymcus() -> None:
    started, finished = restart_messages(3, 2400, 5000, 25)
    assert started == _pymcu_message_format("_StartRestart").format(3, 2400, 1)
    # _FinishRestart: lost * V3_PACKET_INTERVAL seconds, then Index restartLastIndex + 1 to + lost.
    assert finished == _pymcu_message_format("_FinishRestart").format(3, 25 * 0.2, 5001, 5025)
    assert finished == "DFM 3 recorded nothing for about 5 s while it restarted: Index 5001 to 5025 are missing."
