"""The ratio Paradigms' Counter (PyMCU 4.0.1, DFM firmware 4.0.1; PyMCU
``docs/adr/0006-ratio-paradigms-count-contact-by-default.md``).

A ratio Paradigm counts ms of contact (``Counter Contact``, PyMCU's default
from 4.0.1) or Events (``Counter Events``).  ``Program.txt`` says which on the
Paradigm's line; a line without a Counter is from before 4.0.1 and counted
Events.  Under Contact, Requirement and Progress are ms of contact, so their
rises are not Firmware Events and the counts pyflic derives from them are NA.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pyflic.base import opto_program as op
from pyflic.base.paradigm_state import ParadigmState

V4 = Path(__file__).parent / "fixtures" / "Program_v4.txt"

CONTACT_LINE = ("PRC (ProgressiveRatio): Frequency 40 Hz, PulseWidth 8 ms, ActivationMs 500 ms, "
                "Counter Contact, Series Arithmetic, FirstRequirement 1000 ms, Step 1000 ms, "
                "Cap 655350 ms, EventGapMs 100 ms, Acclimation 10")
EVENTS_LINE = ("PRE (ProgressiveRatio): Frequency 40 Hz, PulseWidth 8 ms, ActivationMs 500 ms, "
               "Counter Events, Series Arithmetic, FirstRequirement 1, Step 10, Cap 200, "
               "EventGapMs 100 ms, Acclimation 10")
EXP_LINE = ("PRX (ProgressiveRatio): Frequency 40 Hz, PulseWidth 8 ms, ActivationMs 500 ms, "
            "Counter Contact, Series Exponential, Scale 5000 ms, Rate 0.2, Cap 655350 ms, "
            "EventGapMs 100 ms, Acclimation 0")
FR_LINE = ("FRC (FixedRatio): Frequency 40 Hz, PulseWidth 8 ms, ActivationMs 500 ms, "
           "Counter Contact, Ratio 2000 ms, EventGapMs 100 ms, Acclimation 0")


def _program() -> op.OptoProgram:
    text = V4.read_text(encoding="utf-8").replace(
        "Paradigms:\n", "Paradigms:\n" + "\n".join((CONTACT_LINE, EVENTS_LINE, EXP_LINE, FR_LINE)) + "\n")
    return op.parse_program(text)


def test_the_counter_is_read() -> None:
    program = _program()
    p = program.paradigms
    assert not program.skipped, program.skipped
    assert (p["PRC"].counter, p["PRC"].counts_contact) == ("Contact", True)
    assert (p["PRE"].counter, p["PRE"].counts_contact) == ("Events", False)
    ## The Requirement settings are read in ms under Contact.
    assert (p["PRC"].first_requirement, p["PRC"].step, p["PRC"].cap) == (1000, 1000, 655350)
    assert p["PRX"].scale == 5000.0 and p["FRC"].ratio == 2000
    ## A line without a Counter is from before 4.0.1: Events.
    pr10 = p["PR10"]
    assert pr10.counter is None and not pr10.counts_contact
    assert program.contact_paradigms() == frozenset({"prc", "prx", "frc"})


def test_the_lines_print_as_pymcu_writes_them() -> None:
    p = _program().paradigms
    for name, line in (("PRC", CONTACT_LINE), ("PRE", EVENTS_LINE), ("PRX", EXP_LINE), ("FRC", FR_LINE)):
        assert p[name].describe() == line


def test_contact_requirements_are_ms_as_the_firmware_counts_them() -> None:
    p = _program().paradigms
    ## Arithmetic: 1 s, 2 s, 3 s ...
    assert [p["PRC"].requirement_for(k) for k in (1, 2, 3)] == [1000, 2000, 3000]
    assert [p["PRE"].requirement_for(k) for k in (1, 2, 3, 21, 22)] == [1, 11, 21, 200, 200]
    assert p["FRC"].requirement_for(4) == 2000
    ## Exponential: the firmware computes in 10 ms units from Scale / 10, so
    ## 5000 ms is the Richardson-Roberts series in units of 500 ms ... in
    ## 10 ms steps: round(500 x e^(0.2k) - 500) units.
    expected = [int(np.float32(500.0) * np.exp(np.float32(0.2) * np.float32(k)) - np.float32(500.0)
                    + np.float32(0.5)) * 10 for k in (1, 2, 3)]
    assert [p["PRX"].requirement_for(k) for k in (1, 2, 3)] == expected
    assert all(v % 10 == 0 for v in expected)


def _state(paradigms, progress, requirement) -> ParadigmState:
    n = len(paradigms)
    frame = pd.DataFrame({"Interval": [1] * n, "Cycle": [1] * n, "Paradigm": paradigms,
                          **{f"Phase{w}": ["M"] * n for w in range(1, 13)},
                          **{f"Activations{w}": [0] * n for w in range(1, 13)},
                          **{f"Requirement{w}": requirement for w in range(1, 13)},
                          **{f"Progress{w}": progress for w in range(1, 13)},
                          **{f"OnMs{w}": [0] * n for w in range(1, 13)}})
    return ParadigmState.from_frame(frame, minutes=np.arange(n) / 300.0)


def test_contact_rows_hold_no_firmware_events() -> None:
    ## Three rows under an Events Paradigm, then three under a Contact one.
    state = _state(["PRE"] * 3 + ["PRC"] * 3, [0, 1, 2, 0, 500, 900], [5, 5, 5, 1000, 1000, 1000])
    assert list(state.firmware_event_rises(1)) == [1, 2, 4, 5]
    assert state.firmware_events_between(1, 0, 3) == 2
    state.mark_contact(_program().contact_paradigms())
    assert list(state.contact) == [False] * 3 + [True] * 3
    ## Progress under Contact rises with every 10 ms of contact: not Events.
    assert list(state.firmware_event_rises(1)) == [1, 2]
    assert state.firmware_events_between(1, 0, 3) == 2
    assert np.isnan(state.firmware_events_between(1, 2, 6))
    assert state.counts_contact() and not state.counts_contact(0, 3) and state.counts_contact(3, 6)
    ## A range's subset keeps the marks.
    assert list(state.subset(np.array([False, True, True, True, False, False])).contact) == \
        [False, False, True]
