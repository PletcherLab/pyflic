"""The Opto Program: reading the MCU's exported Program.txt.

``fixtures/Program.txt`` is a real export — the one behind the first
Progressive Ratio dataset — so the parser is held to the grammar the MCU
actually writes, not to a paraphrase of it.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from pyflic.base import opto_program as op
from pyflic.base.experiment_types.progressive_ratio import paired_from_program

FIXTURE = Path(__file__).parent / "fixtures" / "Program.txt"


@pytest.fixture(scope="module")
def sample() -> op.OptoProgram:
    return op.read_program(FIXTURE)


# ---------------------------------------------------------------------------
# The sample export
# ---------------------------------------------------------------------------

def test_the_header(sample):
    assert sample.start == datetime(2025, 7, 28, 12, 1, 0)
    assert sample.end == datetime(2025, 7, 29, 12, 1, 0)
    assert sample.duration_min == pytest.approx(1440.0)
    assert sample.baseline is True
    assert sorted(sample.dfms) == list(range(1, 9))
    assert sample.rejected == {} and sample.warnings == []


def test_linkage_and_trigger_wells(sample):
    dfm1 = sample.section(1)
    assert dfm1.linkage_groups() == {1: (1, 2, 3, 4), 2: (5, 6, 7, 8), 3: (9, 10, 11, 12)}
    assert dfm1.trigger_wells() == (1, 5, 9)
    assert sample.section(2).trigger_wells() == (3, 7, 11)
    assert op.linkage_label(dfm1) == "1: W1-W4 · 2: W5-W8 · 3: W9-W12"


def test_the_echoed_interval_is_decoded(sample):
    """F:5160 is 40 Hz carrying 10 acclimation events; D, L and M are the
    decay, the ratio step and the cap — the interval line wins over the
    section's own 0 ms defaults, because it is what the MCU parsed."""
    iv = sample.section(1).intervals[0]
    assert iv.start == datetime(2025, 7, 28, 12, 1, 0)
    assert iv.dark is False and iv.duration_min == pytest.approx(1440.0)
    p = iv.params
    assert (p.frequency, p.pulse_width, p.decay, p.delay, p.max_time_on) == \
        (5160, 8, 500, 1000, 1000000000000)
    assert p.frequency_hz == 40 and p.acclimation_events == 10
    assert p.pulse_width_ms == 8 and not p.inverted
    assert p.paradigm == op.PROGRESSIVE_RATIO
    assert iv.group_mode((1, 2, 3, 4)) == op.PROGRESSIVE_RATIO
    assert "10 acclimation events" in p.describe()


def test_the_schedule_covers_the_run(sample):
    sched = sample.section(1).schedule(sample.start, sample.resolved_end())
    assert [(s.start, s.end, s.occurrence) for s in sched] == [
        (sample.start, sample.end, 1)]


# ---------------------------------------------------------------------------
# Encodings and paradigms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("delay,max_on,paradigm", [
    (0, 0, op.CLOSED_LOOP), (0, 2500, op.CLOSED_LOOP_MAX),
    (60000, 0, op.FIXED_INTERVAL), (1000, 200, op.PROGRESSIVE_RATIO),
])
def test_the_paradigm_table(delay, max_on, paradigm):
    assert op.OptoParams(40, 8, 500, delay, max_on).paradigm == paradigm


def test_the_top_bit_of_the_pulse_width_is_non_feeding_activation():
    p = op.OptoParams(40, 32776, 21600000, 500, 0)
    assert p.inverted and p.pulse_width_ms == 8 and p.paradigm == op.NON_FEEDING


def test_a_zero_threshold_is_open_loop_and_all_negative_is_off():
    iv = op.ProgramInterval(1, None, False, (0, -1, 20, -1) + (-1,) * 8,
                            op.OptoParams(40, 8, 0, 0, 0), 10.0)
    assert iv.group_mode((1, 2)) == op.OPEN_LOOP            # any member at 0
    assert iv.group_mode((2, 4)) == op.LIGHTS_OFF
    assert iv.group_mode((3, 4)) == op.CLOSED_LOOP
    assert iv.trigger_wells((3, 4)) == (3,) and iv.open_wells() == (1,)


# ---------------------------------------------------------------------------
# The schedule
# ---------------------------------------------------------------------------

def _section(kind: str, durations: list[float], stamps=None) -> op.DFMProgram:
    intervals = tuple(
        op.ProgramInterval(i + 1, None if stamps is None else stamps[i], False,
                           (20,) + (-1,) * 11, op.OptoParams(40, 8, 0, 0, 0), d)
        for i, d in enumerate(durations))
    return op.DFMProgram(1, op.DEFAULT_LINKAGE, kind, intervals)


T0 = datetime(2025, 1, 1, 8, 0, 0)


def _spans(sched):
    return [((s.start - T0).total_seconds() / 60, (s.end - T0).total_seconds() / 60,
             s.interval.index) for s in sched]


def test_linear_holds_its_last_interval_to_the_end():
    sched = _section("Linear", [10, 20]).schedule(T0, T0 + timedelta(minutes=60))
    assert _spans(sched) == [(0, 10, 1), (10, 60, 2)]


def test_repeating_starts_over_and_is_cut_at_the_end():
    sched = _section("Repeating", [10, 20]).schedule(T0, T0 + timedelta(minutes=45))
    assert _spans(sched) == [(0, 10, 1), (10, 30, 2), (30, 40, 1), (40, 45, 2)]
    assert [s.occurrence for s in sched] == [1, 1, 2, 2]


def test_constant_runs_its_first_interval_throughout():
    sched = _section("Constant", [10, 20]).schedule(T0, T0 + timedelta(minutes=45))
    assert _spans(sched) == [(0, 45, 1)]


def test_the_printed_timestamps_are_kept():
    stamps = [T0, T0 + timedelta(minutes=15)]
    sched = _section("Linear", [10, 20], stamps).schedule(T0, T0 + timedelta(minutes=60))
    assert _spans(sched) == [(0, 10, 1), (15, 60, 2)]


# ---------------------------------------------------------------------------
# What the parser tolerates, and what it refuses
# ---------------------------------------------------------------------------

def _text(*dfm_blocks: str, header: str = "Start Time: 01/01/2025 08:00:00\n") -> str:
    return header + "\n".join(dfm_blocks)


GOOD = ("***DFM 1***\nLinkage: 1,2,3,4,5,6,7,8,9,10,11,12\nProgram Type: Linear\n"
        "(01/01/2025 08:00:00) Dark Off,20,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"
        "F:40,P:8,D:1000,L:0,M:0,60.0min.\n")


def test_unknown_lines_and_other_sections_are_skipped_with_a_warning():
    text = _text("# a comment\nFavourite Colour: blue\n" + GOOD,
                 "***ENVMON***\nRelay: 1\n(01/01/2025 08:00:00) Dark Off,1,0,1,1,0,0,0,0,0,"
                 "0,0,0,F:40,P:8,D:0,L:0,M:0,60min.\n")
    program = op.parse_program(text)
    assert sorted(program.dfms) == [1]
    assert any("Favourite Colour" in w for w in program.warnings)
    assert any("ENVMON" in w and "not a DFM" in w for w in program.warnings)


def test_a_bad_section_is_dropped_and_the_rest_kept():
    bad = GOOD.replace("DFM 1", "DFM 2").replace("20,-1,", "20,x,", 1)
    program = op.parse_program(_text(GOOD, bad))
    assert sorted(program.dfms) == [1]
    assert 2 in program.rejected and "not all integers" in program.rejected[2]
    assert program.mentions(2) and not program.mentions(3)


def test_the_authored_nineteen_value_interval_is_read_too():
    authored = GOOD.replace("F:40,P:8,D:1000,L:0,M:0,60.0min.", "40,8,500,1000,200,30")
    iv = op.parse_program(_text(authored)).section(1).intervals[0]
    assert iv.params.paradigm == op.PROGRESSIVE_RATIO and iv.duration_min == 30.0


def test_a_file_with_no_readable_section_is_refused():
    with pytest.raises(op.ProgramError, match="no DFM section"):
        op.parse_program(_text(GOOD.replace("60.0min.", "")))


def test_an_authored_program_is_named_as_such():
    with pytest.raises(op.ProgramError, match="authored MCU program"):
        op.parse_program("[General]\nExpDurationMin: 60\n[DFM]\nID: 1\n"
                         "Interval: 0,20,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,60\n")


def test_finding_the_file(tmp_path):
    assert op.find_program(tmp_path) is None
    (tmp_path / "program.TXT").write_text(GOOD)
    assert op.find_program(tmp_path).name == "program.TXT"
    assert op.find_program(tmp_path / "missing") is None


def test_two_program_files_are_refused(tmp_path):
    (tmp_path / "Program.txt").write_text(GOOD)
    (tmp_path / "PROGRAM.txt").write_text(GOOD)
    if len(list(tmp_path.iterdir())) < 2:
        pytest.skip("case-insensitive file system: only one name can exist")
    with pytest.raises(op.ProgramError, match="more than one"):
        op.find_program(tmp_path)


# ---------------------------------------------------------------------------
# The setting, and small helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (None, "auto"), ("auto", "auto"), (True, "yes"), (False, "no"),
    ("Yes", "yes"), ("off", "no"), (1, "yes"), (0, "no"),
])
def test_the_setting(value, expected):
    assert op.normalize_setting(value) == expected


def test_a_bad_setting_is_named():
    with pytest.raises(ValueError, match="'global.optogenetics' must be auto, yes or no"):
        op.normalize_setting("sometimes", where="global.optogenetics")
    assert op.setting_problem("sometimes") and op.setting_problem(True) is None
    assert op.setting_to_yaml("auto") is None and op.setting_to_yaml("yes") is True


def test_wells_label():
    assert op.wells_label([1, 2, 3, 4]) == "W1-W4"
    assert op.wells_label([1, 3]) == "W1, W3"
    assert op.wells_label([5, 6]) == "W5, W6"
    assert op.wells_label([]) == "—"


# ---------------------------------------------------------------------------
# Progressive Ratio: paired chambers from the trigger wells
# ---------------------------------------------------------------------------

def test_paired_chambers_come_from_the_trigger_wells(sample):
    assert paired_from_program(sample.section(1)) == ([1, 3, 5], [], [])
    assert paired_from_program(sample.section(2)) == ([2, 4, 6], [], [])
    ## DFM 3 triggers on the right-hand wells: well A under pi_direction right.
    paired, problems, notes = paired_from_program(sample.section(3), None, "right")
    assert paired == [1, 3, 5] and not problems and not notes
    _paired, _problems, notes = paired_from_program(sample.section(3), None, "left")
    assert len(notes) == 3 and "well B" in notes[0]


def test_a_group_without_one_trigger_well_cannot_be_derived():
    section = _section("Linear", [60])
    both = op.DFMProgram(1, op.DEFAULT_LINKAGE, "Linear", (op.ProgramInterval(
        1, None, False, (20, -1, 20) + (-1,) * 9, op.OptoParams(40, 8, 0, 0, 0), 60),))
    paired, problems, _notes = paired_from_program(both)
    assert paired is None and any("both chambers of group 1" in p for p in problems)
    paired, problems, _notes = paired_from_program(section)
    assert paired is None and any("no trigger well in chamber group 2" in p
                                  for p in problems)
