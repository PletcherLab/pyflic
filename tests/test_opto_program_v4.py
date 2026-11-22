"""The Opto Program's second grammar: the PyMCU 4.0.0 ``Program.txt``.

``fixtures/Program_v4.txt`` is the example §7.1 of the retired
``docs/dfm-firmware-4-data-guide.md`` gave, verbatim (the guide is replaced by
``docs/firmware-4-data-in-pyflic.md``): PyMCU ``v4_Dev``'s own export of a Program with two named
Paradigms, a DFM in the old 19-number form (written back with a generated
name), a connected DFM with no section and an Environmental Monitor.
``fixtures/Program_simple.txt`` is what PyMCU writes for a recording started
without a Program.  The legacy grammar's own tests are in
``test_opto_program.py``; here, beside the new grammar, is what the legacy
path must still give through the new API.
"""

from __future__ import annotations

import contextlib
import io
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyflic.base import opto_program as op
from pyflic.base import opto_qc as oq
from pyflic.base.experiment_types.progressive_ratio import paired_from_program

from opto_fixtures import START, WELL_CASES, case_program, make_opto_dir, pr_program

FIXTURES = Path(__file__).parent / "fixtures"
V4 = FIXTURES / "Program_v4.txt"
SIMPLE = FIXTURES / "Program_simple.txt"
LEGACY = FIXTURES / "Program.txt"

ODD_WELLS = (1, 3, 5, 7, 9, 11)
EVEN_WELLS = (2, 4, 6, 8, 10, 12)


@pytest.fixture(scope="module")
def v4() -> op.OptoProgram:
    return op.read_program(V4)


def _v4_text(paradigms: list[str], intervals: list[str], *, header: str = "",
             linkage: str = "1,2,3,4,5,6,7,8,9,10,11,12") -> str:
    """A 4.0.0 export with one DFM section holding *intervals*."""
    return ("Start Time: 10/12/2026 20:00:00\nEnd Time: 10/12/2026 21:00:00\n"
            "Duration: 60 min\nDefault Linkage: 1,2,3,4,5,6,7,8,9,10,11,12\n"
            "Default Program Type: Linear\nBaseline: Yes\n" + header +
            "\nParadigms:\n" + "".join(p + "\n" for p in paradigms) +
            f"\n***DFM 1***\nLinkage: {linkage}\nProgram Type: Linear\n" +
            "".join(i + "\n" for i in intervals))


CLOSED_LINE = ("Closed (ClosedLoop): Frequency 40 Hz, PulseWidth 8 ms, HoldMs 500 ms, "
               "MaxOnMs 0 ms, EventGapMs 100 ms")
STAMP = "(10/12/2026 20:00:00)"


# ---------------------------------------------------------------------------
# The §7.1 example
# ---------------------------------------------------------------------------

def test_the_example_is_read_with_every_dfm_section(v4):
    assert v4.grammar == op.GRAMMAR_V4
    assert sorted(v4.dfms) == [1, 2, 3]
    assert v4.rejected == {} and v4.warnings == []
    assert v4.start == datetime(2026, 10, 12, 20, 0, 0)
    assert v4.end == datetime(2026, 10, 13, 2, 0, 0)
    assert v4.duration_min == pytest.approx(360.0) and v4.baseline is True
    assert list(v4.paradigms) == ["Closed", "PR10", "ClosedLoop-40Hz-8ms-Hold500-MaxOn0",
                                  "Idle"]


def test_the_env_mon_section_is_ignored(v4):
    """Its relay and output states print through the threshold formatter (a
    0 state reads ``On``); it is not a DFM and is not worth a warning."""
    assert not v4.mentions(99) and v4.section(99) is None
    assert not any("Env Mon" in w for w in v4.warnings)


def test_trigger_wells_are_the_level_wells(v4):
    dfm1 = v4.section(1)
    assert dfm1.linkage_groups() == {g: (2 * g - 1, 2 * g) for g in range(1, 7)}
    assert dfm1.program_type == "Linear" and dfm1.trigger_wells() == ODD_WELLS
    first, second = dfm1.intervals
    assert first.start == datetime(2026, 10, 12, 20, 0, 0) and first.dark is True
    assert first.duration_min == pytest.approx(30.0)
    assert first.thresholds == (20, op.OFF) * 6
    assert first.trigger_wells() == ODD_WELLS and first.off_wells() == EVEN_WELLS
    assert first.open_wells() == ()
    assert first.paradigm.name == "Closed" and second.paradigm.name == "PR10"
    assert second.start == datetime(2026, 10, 12, 20, 30, 0)
    assert second.duration_min == pytest.approx(330.0)
    ## The trigger well of each chamber decides; the linked Off well follows.
    assert first.group_mode((1, 2)) == op.CLOSED_LOOP
    assert second.group_mode((1, 2)) == op.PROGRESSIVE_RATIO


def test_a_dfm_without_a_section_gets_idle_with_every_well_off(v4):
    iv = v4.section(2).intervals[0]
    assert iv.paradigm.name == "Idle" and iv.dark is False
    assert iv.thresholds == (op.OFF,) * 12 and iv.trigger_wells() == ()
    assert iv.off_wells() == tuple(range(1, 13))
    assert iv.group_mode((1, 2)) == op.LIGHTS_OFF and iv.hold_ms == 0


def test_the_example_paradigms(v4):
    pr = v4.paradigms["PR10"]
    assert (pr.type, pr.type_name, pr.mode) == (op.PROGRESSIVE_RATIO, "ProgressiveRatio",
                                                op.PROGRESSIVE_RATIO)
    assert (pr.frequency_hz, pr.pulse_width_ms, pr.activation_ms) == (40, 8, 500)
    assert (pr.series, pr.first_requirement, pr.step, pr.cap) == ("Arithmetic", 1, 10, 200)
    assert (pr.event_gap_ms, pr.acclimation, pr.scale, pr.rate) == (100, 10, None, None)
    assert pr.hold == 500 and not pr.generated and pr.legacy is None
    closed = v4.paradigms["Closed"]
    assert (closed.hold_ms, closed.max_on_ms, closed.event_gap_ms) == (500, 0, 100)
    assert closed.activation_ms is None and not closed.generated


def test_describe_prints_the_line_back(v4):
    lines = V4.read_text(encoding="utf-8").splitlines()
    block = lines[lines.index("Paradigms:") + 1:lines.index("Paradigms:") + 5]
    assert [p.describe() for p in v4.paradigms.values()] == block


def test_interval_for_is_the_kth_printed_line(v4):
    assert v4.interval_for(1, 1) is v4.section(1).intervals[0]
    assert v4.interval_for(1, 2).paradigm.name == "PR10"
    ## Past the printed lines (a Repeating program's later pass, an Interval
    ## starting after the duration), on a DFM with no section, or 0.
    assert v4.interval_for(1, 3) is None
    assert v4.interval_for(4, 1) is None and v4.interval_for(1, 0) is None
    legacy = op.read_program(LEGACY)
    assert legacy.interval_for(2, 1) is legacy.section(2).intervals[0]


def _paradigm_first(text: str) -> str:
    """*text* with each interval line's first two values swapped: the
    Paradigm, then Dark or Lit, as PyMCU now writes them, in the order of a
    Program file's Interval line."""
    def swap(match):
        first, second, rest = match.group("body").split(",", 2)
        return f"({match.group('stamp')}) {second.strip()}, {first.strip()},{rest}"
    return "\n".join(swap(m) if (m := op._INTERVAL_RE.match(line)) else line
                     for line in text.splitlines())


def test_an_interval_may_name_its_paradigm_before_dark_or_lit(v4):
    text = _paradigm_first(V4.read_text(encoding="utf-8"))
    assert f"{STAMP} Closed, Dark, " in text
    program = op.parse_program(text)
    assert program.grammar == op.GRAMMAR_V4
    assert program.rejected == {} and program.warnings == []
    for dfm in (1, 2, 3):
        assert program.section(dfm).intervals == v4.section(dfm).intervals


# ---------------------------------------------------------------------------
# Generated names
# ---------------------------------------------------------------------------

def test_a_generated_name_parses_with_its_settings(v4):
    p = v4.paradigms["ClosedLoop-40Hz-8ms-Hold500-MaxOn0"]
    assert p.type == op.CLOSED_LOOP and p.generated and p.legacy is None
    assert (p.frequency_hz, p.pulse_width_ms, p.hold_ms, p.max_on_ms, p.event_gap_ms) == \
        (40, 8, 500, 0, 100)
    assert p.hold == 500 and p.mode == op.CLOSED_LOOP
    iv = v4.section(3).intervals[0]
    assert iv.paradigm is p and iv.trigger_wells() == ODD_WELLS and iv.hold_ms == 500


@pytest.mark.parametrize("params,name", [
    (op.OptoParams(40, 8, 500, 0, 0), "ClosedLoop-40Hz-8ms-Hold500-MaxOn0"),
    (op.OptoParams(40, 8, 500, 0, 2500), "ClosedLoop-40Hz-8ms-Hold500-MaxOn2500"),
    ## -1 meant no limit (PyMCU: "-1 counts as 0").
    (op.OptoParams(40, 8, 500, 0, -1), "ClosedLoop-40Hz-8ms-Hold500-MaxOn0"),
    (op.OptoParams(40, 8, 1000, 60000, 0), "FixedInterval-40Hz-8ms-Activation1000-Lockout60000"),
    (op.OptoParams(40, 32776, 21600000, 500, 0), "NonFeeding-40Hz-8ms-OnDelay500-MaxOn21600000"),
    ## What PyMCU refuses, pyflic still names, extending the pattern.
    (op.OptoParams(5160, 8, 500, 1000, 1000000000000),
     "ProgressiveRatio-40Hz-8ms-Activation500-Delay1000-MaxOn1000000000000-Acclimation10"),
])
def test_legacy_settings_get_pymcus_generated_name(params, name):
    p = op.paradigm_from_legacy(params)
    assert p.name == name and p.generated and p.legacy == params


def test_the_legacy_mapping():
    closed = op.paradigm_from_legacy(op.OptoParams(40, 8, 500, 0, 2500))
    assert (closed.type, closed.hold_ms, closed.max_on_ms) == (op.CLOSED_LOOP, 500, 2500)
    assert closed.mode == op.CLOSED_LOOP_MAX
    fixed = op.paradigm_from_legacy(op.OptoParams(40, 8, 1000, 60000, 0))
    assert (fixed.type, fixed.activation_ms, fixed.lockout_ms) == (op.FIXED_INTERVAL, 1000,
                                                                   60000)
    nonfeeding = op.paradigm_from_legacy(op.OptoParams(40, 32776, 21600000, 500, 0))
    assert (nonfeeding.type, nonfeeding.on_delay_ms, nonfeeding.max_on_ms,
            nonfeeding.pulse_width_ms) == (op.NON_FEEDING, 500, 21600000, 8)
    old_pr = op.paradigm_from_legacy(op.OptoParams(5160, 8, 500, 1000, 1000000000000))
    assert (old_pr.type, old_pr.activation_ms, old_pr.acclimation) == \
        (op.PROGRESSIVE_RATIO, 500, 10)
    assert old_pr.series is None and old_pr.requirement_for(1) is None
    assert old_pr.event_gap_ms is None             # the old firmware has no Event gap


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (0, op.ON), (-1, op.OFF), (-5, op.OFF), (1024, op.OFF), (40000, op.OFF),
    (1, 1), (20, 20), (1023, 1023),
])
def test_threshold_from_legacy(value, expected):
    assert op.threshold_from_legacy(value) == expected


def test_is_trigger():
    assert op.is_trigger(20) and op.is_trigger(1023)
    assert not op.is_trigger(op.ON) and not op.is_trigger(op.OFF)
    assert not op.is_trigger(True)


def test_thresholds_are_case_insensitive_and_bounded():
    text = _v4_text([CLOSED_LINE], [f"{STAMP} lit, closed, 20,off,ON,Off,1,1023,"
                                    f"Off,Off,Off,Off,Off,Off, 60 min"])
    iv = op.parse_program(text).section(1).intervals[0]
    assert iv.thresholds[:6] == (20, op.OFF, op.ON, op.OFF, 1, 1023)
    assert iv.paradigm.name == "Closed" and iv.dark is False
    for bad in ("0", "1024", "-1", "x"):
        line = f"{STAMP} Lit, Closed, {bad},Off,Off,Off,Off,Off,Off,Off,Off,Off,Off,Off, 60 min"
        with pytest.raises(op.ProgramError, match="On, Off or a level from 1 to 1023"):
            op.parse_program(_v4_text([CLOSED_LINE], [line]))


# ---------------------------------------------------------------------------
# Modes and Holds
# ---------------------------------------------------------------------------

def _paradigm(kind: str, **settings) -> op.Paradigm:
    return op.Paradigm("P", kind, 40, 8, **settings)


def _interval(paradigm: op.Paradigm, thresholds) -> op.ProgramInterval:
    return op.ProgramInterval(1, None, False, tuple(thresholds), paradigm, 60.0)


@pytest.mark.parametrize("paradigm,thresholds,mode", [
    (_paradigm(op.CLOSED_LOOP, hold_ms=500, max_on_ms=0), (20, op.OFF), op.CLOSED_LOOP),
    (_paradigm(op.CLOSED_LOOP, hold_ms=500, max_on_ms=2500), (20, op.OFF),
     op.CLOSED_LOOP_MAX),
    ## Any On member lights the group; every member Off keeps it dark.
    (_paradigm(op.CLOSED_LOOP, hold_ms=500, max_on_ms=0), (20, op.ON), op.OPEN_LOOP),
    (_paradigm(op.FIXED_RATIO, activation_ms=500, ratio=5), (op.OFF, op.OFF),
     op.LIGHTS_OFF),
    (_paradigm(op.OPEN_LOOP), (20, op.OFF), op.OPEN_LOOP),
    (_paradigm(op.FIXED_RATIO, activation_ms=500, ratio=5), (20, op.OFF), op.FIXED_RATIO),
    (_paradigm(op.NON_FEEDING, on_delay_ms=500, max_on_ms=0), (20, op.OFF),
     op.NON_FEEDING),
])
def test_group_mode(paradigm, thresholds, mode):
    iv = _interval(paradigm, (*thresholds, *(op.OFF,) * 10))
    assert iv.group_mode((1, 2)) == mode


@pytest.mark.parametrize("paradigm,hold", [
    (_paradigm(op.CLOSED_LOOP, hold_ms=500, max_on_ms=2500), 500),
    (_paradigm(op.FIXED_INTERVAL, activation_ms=1000, lockout_ms=60000), 1000),
    (_paradigm(op.PROGRESSIVE_RATIO, activation_ms=500, series="Arithmetic",
               first_requirement=1, step=10), 500),
    (_paradigm(op.FIXED_RATIO, activation_ms=300, ratio=3), 300),
    (_paradigm(op.NON_FEEDING, on_delay_ms=500, max_on_ms=1000), None),
    (_paradigm(op.OPEN_LOOP), None),
])
def test_the_hold_of_each_type(paradigm, hold):
    assert paradigm.hold == hold
    assert _interval(paradigm, (20,) + (op.OFF,) * 11).hold_ms == (hold or 0)


@pytest.mark.parametrize("params", [
    op.OptoParams(40, 8, 750, 0, 0), op.OptoParams(40, 8, 750, 0, 2500),
    op.OptoParams(40, 8, 750, 60000, 0), op.OptoParams(5160, 8, 750, 1000, 10 ** 12),
    op.OptoParams(40, 32776, 750, 500, 0),
])
def test_a_legacy_interval_keeps_its_decay_as_the_hold(params):
    """Whatever the paradigm — the non-feeding one has no Hold of its own —
    the light QC reads a legacy interval's decay as before."""
    iv = _interval(op.paradigm_from_legacy(params), (20,) + (op.OFF,) * 11)
    assert iv.hold_ms == 750
    assert iv.paradigm.mode == params.paradigm


# ---------------------------------------------------------------------------
# Requirements
# ---------------------------------------------------------------------------

def _requirements(paradigm: op.Paradigm, n: int) -> list[int | None]:
    return [paradigm.requirement_for(k) for k in range(1, n + 1)]


def test_exponential_defaults_give_richardson_and_roberts():
    p = _paradigm(op.PROGRESSIVE_RATIO, activation_ms=500, series="Exponential",
                  scale=5.0, rate=0.2, cap=65535)
    assert _requirements(p, 15) == [1, 2, 4, 6, 9, 12, 15, 20, 25, 32, 40, 50, 62, 77, 95]
    capped = _paradigm(op.PROGRESSIVE_RATIO, activation_ms=500, series="Exponential",
                       scale=5.0, rate=0.2, cap=20)
    assert _requirements(capped, 10) == [1, 2, 4, 6, 9, 12, 15, 20, 20, 20]


def test_an_overflowing_exponential_series_gives_the_cap_as_the_firmware_does():
    """DFMV3_V2 024b692: e^(Rate k) past a float's range gives the Cap, and
    Scale 0 gives 1 (never 0 x infinity).  pyflic raised OverflowError."""
    huge = _paradigm(op.PROGRESSIVE_RATIO, activation_ms=1, series="Exponential",
                     scale=5.0, rate=100.0, cap=50)
    assert huge.requirement_for(1) == 50 and huge.requirement_for(65535) == 50
    flat = _paradigm(op.PROGRESSIVE_RATIO, activation_ms=1, series="Exponential",
                     scale=0.0, rate=100.0, cap=50)
    assert flat.requirement_for(1) == 1 and flat.requirement_for(65535) == 1
    defaults = _paradigm(op.PROGRESSIVE_RATIO, activation_ms=1, series="Exponential",
                         scale=5.0, rate=0.2, cap=65535)
    assert defaults.requirement_for(65535) == 65535


def test_the_exponential_series_rounds_in_single_precision_as_the_firmware_does():
    """The firmware computes in single precision; double precision rounded
    Scale 5, Rate 0.05, k 146 to 7396 where the firmware gives 7397.  The
    simulator's copy of RequirementFor is checked against the compiled
    firmware in test_ledcontrol_crosscheck."""
    from types import SimpleNamespace

    import v4_fixtures as vf

    for scale in (0.5, 5.0, 7.333):
        for rate in (0.05, 0.2, 0.37, 2.5):
            p = _paradigm(op.PROGRESSIVE_RATIO, activation_ms=1, series="Exponential",
                          scale=scale, rate=rate, cap=65535)
            ins = SimpleNamespace(type=vf._PROGRESSIVERATIO,
                                  settings=(1, 1, round(scale * 1000), round(rate * 1000), 65535))
            assert _requirements(p, 300) == [vf._requirement_for(ins, k) for k in range(1, 301)]


def test_arithmetic_series_are_capped():
    small = _paradigm(op.PROGRESSIVE_RATIO, activation_ms=500, series="Arithmetic",
                      first_requirement=1, step=2, cap=5)
    assert _requirements(small, 4) == [1, 3, 5, 5]
    pr10 = _paradigm(op.PROGRESSIVE_RATIO, activation_ms=500, series="Arithmetic",
                     first_requirement=1, step=10, cap=200)
    assert _requirements(pr10, 23) == [1 + 10 * k for k in range(20)] + [200, 200, 200]


def test_fixed_ratio_is_constant_and_other_types_have_none(v4):
    fr = _paradigm(op.FIXED_RATIO, activation_ms=500, ratio=3)
    assert _requirements(fr, 5) == [3] * 5
    assert v4.paradigms["Closed"].requirement_for(1) is None
    assert v4.paradigms["PR10"].requirement_for(21) == 200
    with pytest.raises(ValueError):
        fr.requirement_for(0)


# ---------------------------------------------------------------------------
# What the parser tolerates, and what it refuses
# ---------------------------------------------------------------------------

def test_an_interval_naming_a_missing_paradigm_drops_its_section():
    text = _v4_text([CLOSED_LINE], [f"{STAMP} Lit, PR10, " + ",".join(["20"] * 12)
                                    + ", 60 min"])
    text += ("\n***DFM 2***\nLinkage: 1,2,3,4,5,6,7,8,9,10,11,12\nProgram Type: Linear\n"
             f"{STAMP} Lit, Closed, " + ",".join(["20"] * 12) + ", 60 min\n")
    program = op.parse_program(text)
    assert sorted(program.dfms) == [2] and program.grammar == op.GRAMMAR_V4
    assert "no Paradigm named 'PR10'" in program.rejected[1]


def test_an_unknown_type_drops_the_paradigm_and_an_unknown_setting_is_ignored():
    text = _v4_text(
        [CLOSED_LINE.replace("EventGapMs 100 ms", "EventGapMs 100 ms, Colour 3"),
         "Odd (Sometimes): Frequency 40 Hz, PulseWidth 8 ms",
         "Fr (FixedRatio): Frequency 40 Hz, PulseWidth 8 ms, ActivationMs 500 ms, "
         "Ratio 5, HoldMs 20 ms, EventGapMs 100 ms, Acclimation 3"],
        [f"{STAMP} Lit, Fr, " + ",".join(["20"] * 12) + ", 60 min"])
    program = op.parse_program(text)
    assert list(program.paradigms) == ["Closed", "Fr"]
    assert any("Odd" in w and "dropped" in w for w in program.warnings)
    assert any("unknown setting 'Colour'" in w for w in program.warnings)
    assert any("FixedRatio Paradigms have no HoldMs" in w for w in program.warnings)
    fr = program.paradigms["Fr"]
    assert (fr.type, fr.ratio, fr.acclimation, fr.hold_ms) == (op.FIXED_RATIO, 5, 3, None)
    assert program.section(1).intervals[0].group_mode((1,)) == op.FIXED_RATIO


def test_legacy_default_keys_are_not_read_beside_a_paradigms_block():
    text = _v4_text([CLOSED_LINE], [f"{STAMP} Lit, Closed, " + ",".join(["20"] * 12)
                                    + ", 60 min"],
                    header="Default Opto Decay: 9000ms\n")
    program = op.parse_program(text)
    assert any("Default Opto Decay" in w and "pre-4.0.0" in w for w in program.warnings)
    assert program.section(1).intervals[0].hold_ms == 500


def test_an_exponential_paradigm_round_trips():
    line = ("Exp (ProgressiveRatio): Frequency 20 Hz, PulseWidth 10 ms, ActivationMs 500 ms, "
            "Series Exponential, Scale 5, Rate 0.2, Cap 65535, EventGapMs 100 ms, "
            "Acclimation 0")
    program = op.parse_program(_v4_text([line], [f"{STAMP} Dark, exp, "
                                                 + ",".join(["20"] * 12) + ", 30.5 min"]))
    p = program.paradigms["Exp"]
    assert (p.series, p.scale, p.rate, p.first_requirement, p.step) == \
        ("Exponential", 5.0, 0.2, None, None)
    assert p.describe() == line
    iv = program.section(1).intervals[0]
    assert iv.paradigm is p and iv.dark and iv.duration_min == pytest.approx(30.5)


# ---------------------------------------------------------------------------
# The simple export: a recording without a Program
# ---------------------------------------------------------------------------

def test_a_recording_without_a_program_is_every_well_off():
    program = op.read_program(SIMPLE)
    assert program.grammar == op.GRAMMAR_SIMPLE
    assert program.dfms == {} and program.rejected == {}
    assert program.warnings == [op.SIMPLE_WARNING]
    assert program.start == datetime(2026, 10, 12, 20, 0, 0)
    assert program.end == datetime(2026, 10, 12, 23, 0, 0)
    assert program.duration_min == pytest.approx(180.0) and program.baseline is True
    assert program.section(1) is None and not program.mentions(1)
    assert program.interval_for(1, 1) is None


def test_an_experiment_loads_with_the_simple_export(tmp_path):
    from pyflic.base.yaml_config import load_experiment_yaml

    root = make_opto_dir(tmp_path / "exp", program=SIMPLE.read_text(encoding="utf-8"))
    with contextlib.redirect_stdout(io.StringIO()):
        e = load_experiment_yaml(root, parallel=False, use_disk_cache=False)
    assert e.opto_program is not None and e.opto_program_error is None
    assert e.opto_program.grammar == op.GRAMMAR_SIMPLE
    assert e.opto.program_table().empty
    assert any(op.SIMPLE_WARNING in n for n in e.opto.experiment_notes())


@pytest.mark.parametrize("extra, skipped", [("", 0), ("gibberish\n", 1)])
def test_the_simple_exports_warning_is_not_counted_as_a_skipped_line(tmp_path, extra,
                                                                     skipped):
    """The loader's note said "1 line(s) skipped" of a clean simple export:
    it counted the export's own warning."""
    from pyflic.base.yaml_config import load_experiment_yaml

    text = SIMPLE.read_text(encoding="utf-8") + extra
    assert len(op.parse_program(text).skipped) == skipped
    root = make_opto_dir(tmp_path / "exp", program=text)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        e = load_experiment_yaml(root, parallel=False, use_disk_cache=False)
    printed = out.getvalue()
    assert ("1 line(s) skipped" in printed) == bool(skipped)
    assert f"Program.txt: {op.SIMPLE_WARNING}" in printed
    notes = e.opto.experiment_notes()
    assert f"Program.txt: {op.SIMPLE_WARNING}" in notes
    assert sum(n.startswith("Program.txt: line ") for n in notes) == skipped


# ---------------------------------------------------------------------------
# The program table
# ---------------------------------------------------------------------------

def test_program_rows_for_the_example(v4):
    frame = oq.program_frame(v4)
    assert list(frame.columns) == list(oq.OPTO_PROGRAM_COLUMNS)
    assert list(zip(frame["DFM"], frame["Interval"], frame["Paradigm"])) == [
        (1, 1, "Closed"), (1, 2, "PR10"), (2, 1, "Idle"),
        (3, 1, "ClosedLoop-40Hz-8ms-Hold500-MaxOn0")]
    rows = frame.set_index(["DFM", "Interval"])
    closed, pr, idle, generated = (rows.loc[(1, 1)], rows.loc[(1, 2)], rows.loc[(2, 1)],
                                   rows.loc[(3, 1)])
    assert (closed.ParadigmType, closed.Generated, closed.Dark) == ("ClosedLoop", False, True)
    assert (closed.HoldMs, closed.MaxOnMs, closed.EventGapMs) == (500, 0, 100)
    assert pd.isna(closed.ActivationMs) and pd.isna(closed.Acclimation)
    assert closed.TriggerWells == "W1, W3, W5, W7, W9, W11" and closed.OpenLoopWells == "—"
    assert closed.Linkage == "1: W1, W2 · 2: W3, W4 · 3: W5, W6 · 4: W7, W8 · " \
                             "5: W9, W10 · 6: W11, W12"
    assert (pr.ParadigmType, pr.ActivationMs, pr.Series, pr.FirstRequirement, pr.Step,
            pr.Cap, pr.Acclimation) == ("ProgressiveRatio", 500, "Arithmetic", 1, 10, 200, 10)
    assert pd.isna(pr.HoldMs) and pd.isna(pr.Scale) and pd.isna(pr.Rate)
    assert (pr.DurationMin, pr.Start) == (330.0, "2026-10-12 20:30:00")
    assert idle.TriggerWells == "—" and idle.HoldMs == 0
    assert generated.Generated and generated.HoldMs == 500
    ## The legacy columns are the old export's alone.
    assert frame["LegacyF"].isna().all() and frame["LegacyP"].isna().all()
    assert oq.paradigms_in(v4) == {op.CLOSED_LOOP, op.PROGRESSIVE_RATIO}


# ---------------------------------------------------------------------------
# Legacy exports through the new API
# ---------------------------------------------------------------------------

LEGACY_CASES = {
    ## name: (interval keywords, old mode, ParadigmType, the hold's column)
    "closed loop": ({}, op.CLOSED_LOOP, "ClosedLoop", "HoldMs"),
    "closed loop, max": ({"max_on": 2500}, op.CLOSED_LOOP_MAX, "ClosedLoop", "HoldMs"),
    "fixed interval": ({"delay": 60000}, op.FIXED_INTERVAL, "FixedInterval", "ActivationMs"),
    "non-feeding": ({"pw": 32776, "delay": 500}, op.NON_FEEDING, "NonFeeding", "MaxOnMs"),
    "progressive ratio": ({"freq": 5160, "delay": 1000, "max_on": 10 ** 12},
                          op.PROGRESSIVE_RATIO, "ProgressiveRatio", "ActivationMs"),
}


@pytest.mark.parametrize("case", list(LEGACY_CASES))
def test_a_legacy_export_parses_unchanged(case):
    """The opto fixture's program in each legacy paradigm: the wells, the
    mode and the decay the light QC reads are what they always were, and the
    program table shows the same type and hold under its new column names."""
    keywords, mode, type_name, hold_column = LEGACY_CASES[case]
    program = op.parse_program(case_program(**keywords))
    assert program.grammar == op.GRAMMAR_LEGACY
    assert program.warnings == [] and program.rejected == {}
    iv = program.section(1).intervals[0]
    raw = [WELL_CASES[w][1] for w in range(1, 13)]
    assert iv.trigger_wells() == tuple(w for w in range(1, 13) if raw[w - 1] > 0)
    assert iv.open_wells() == tuple(w for w in range(1, 13) if raw[w - 1] == 0)
    assert iv.off_wells() == tuple(w for w in range(1, 13) if raw[w - 1] < 0)
    assert iv.group_mode((1,)) == mode and iv.group_mode((5,)) == op.OPEN_LOOP
    assert iv.group_mode((4,)) == op.LIGHTS_OFF
    assert iv.hold_ms == 1000                         # the fixture's decay
    assert iv.paradigm.generated and iv.paradigm.legacy is not None
    row = oq.program_frame(program).iloc[0]
    assert (row.ParadigmType, row[hold_column]) == (type_name, 1000)
    assert (row.LegacyF, row.LegacyP) == (iv.paradigm.legacy.frequency,
                                          iv.paradigm.legacy.pulse_width)
    assert program.paradigms == {iv.paradigm.name: iv.paradigm}


def test_the_real_export_through_the_new_api():
    program = op.read_program(LEGACY)
    assert program.grammar == op.GRAMMAR_LEGACY
    frame = oq.program_frame(program)
    assert len(frame) == 8
    assert set(frame["ParadigmType"]) == {"ProgressiveRatio"}
    assert set(frame["ActivationMs"]) == {500} and set(frame["Acclimation"]) == {10}
    assert set(frame["LegacyF"]) == {5160} and set(frame["LegacyP"]) == {8}
    assert oq.paradigms_in(program) == {op.PROGRESSIVE_RATIO}
    assert list(program.paradigms) == [
        "ProgressiveRatio-40Hz-8ms-Activation500-Delay1000-MaxOn1000000000000-Acclimation10"]


def test_paired_chambers_still_come_from_the_pr_program_fixtures():
    left = op.parse_program(pr_program({1: (1, 4, 5)}, pi_direction="left"))
    assert paired_from_program(left.section(1)) == ([1, 4, 5], [], [])
    right = op.parse_program(pr_program({2: (2, 3, 6)}, pi_direction="right"))
    assert paired_from_program(right.section(2), None, "right") == ([2, 3, 6], [], [])


# ---------------------------------------------------------------------------
# One program, either grammar, the same light QC
# ---------------------------------------------------------------------------

def _v4_case_program() -> str:
    """:func:`opto_fixtures.case_program` written as PyMCU 4.0.0 exports it."""
    thresholds = ",".join(op.threshold_text(op.threshold_from_legacy(WELL_CASES[w][1]))
                          for w in range(1, 13))
    stamp = START.strftime("%m/%d/%Y %H:%M:%S")
    end = (START + pd.Timedelta(minutes=60)).strftime("%m/%d/%Y %H:%M:%S")
    return (f"Start Time: {stamp}\nEnd Time: {end}\nDuration: 60 min\n"
            "Default Linkage: 1,2,3,4,5,6,7,8,9,10,11,12\nDefault Program Type: Linear\n"
            "Baseline: Yes\n\nParadigms:\n"
            "Closed (ClosedLoop): Frequency 40 Hz, PulseWidth 8 ms, HoldMs 1000 ms, "
            "MaxOnMs 0 ms, EventGapMs 100 ms\n\n***DFM 1***\n"
            "Linkage: 1,2,3,4,5,6,7,8,9,10,11,12\nProgram Type: Linear\n"
            f"({stamp}) Lit, Closed, {thresholds}, 60 min\n")


def test_the_same_program_in_either_grammar_gives_the_same_light_qc(tmp_path):
    from pyflic.base.yaml_config import load_experiment_yaml

    tables = {}
    for name, text in (("legacy", case_program()), ("v4", _v4_case_program())):
        root = make_opto_dir(tmp_path / name, program=text)
        with contextlib.redirect_stdout(io.StringIO()):
            e = load_experiment_yaml(root, parallel=False, use_disk_cache=False)
        assert e.opto_program.grammar == name
        tables[name] = (e.opto.qc_table(), e.opto.interval_table(), e.opto.events_table(),
                        e.opto.decay_samples(1))
    for legacy, v4 in zip(tables["legacy"][:3], tables["v4"][:3]):
        pd.testing.assert_frame_equal(legacy, v4)
    assert np.array_equal(tables["legacy"][3], tables["v4"][3])
    verdicts = dict(zip(tables["v4"][0]["Group"], tables["v4"][0]["Verdict"]))
    assert verdicts[2] == "failed" and verdicts[5] == "ok"


def test_a_v4_mismatch_is_checked_against_the_paradigm(tmp_path):
    from pyflic.base.yaml_config import load_experiment_yaml

    root = make_opto_dir(tmp_path / "exp", program=_v4_case_program(), freq=50)
    with contextlib.redirect_stdout(io.StringIO()):
        e = load_experiment_yaml(root, parallel=False, use_disk_cache=False)
    assert oq.PROGRAM_MISMATCH in e.opto.result(1).flags
    assert any("OptoFreq is 50, the program says 40 Hz" in n for n in e.opto.result(1).notes)


# ---------------------------------------------------------------------------
# What ran, in a line for a person (ADR-0027)
# ---------------------------------------------------------------------------
#
# Program.txt names no protocol and no version.  The Experiment Editor shows
# what each DFM ran and which PyMCU wrote the file, and copies them into the
# About's protocol only when asked.

def test_the_summary_says_what_each_dfm_ran_in_order(v4):
    ## A Paradigm a person named is named; PyMCU's generated name is not; an
    ## interval that lights no well is "lights off".
    assert op.program_summary(v4) == (
        "DFM 1: closed loop Closed 30 min, then progressive ratio PR10 5.5 h; "
        "DFM 2: lights off 6 h; DFM 3: closed loop 6 h")
    assert op.pymcu_version(v4) == "PyMCU 4.x"


def test_dfms_that_ran_the_same_are_named_together_and_a_repeat_is_said():
    legacy = op.read_program(LEGACY)
    assert op.program_summary(legacy) == "DFM 1–8: progressive ratio 24 h (repeating)"
    assert op.pymcu_version(legacy) == "PyMCU 3.x"


def test_a_recording_without_a_program_says_so_and_claims_no_version():
    simple = op.read_program(SIMPLE)
    assert op.program_summary(simple) == "no program: every light off, 3 h"
    assert op.pymcu_version(simple) is None
