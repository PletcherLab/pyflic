"""PyMCU's Baseline columns and ``Baseline: Adaptive`` (PyMCU
``docs/adaptive-baseline-plan.md``, stage S5).

PyMCU writes ``Baseline1``-``Baseline12`` after ``OnMs12``: the Baseline in
force behind each record's thresholds.  pyflic takes them off ``raw_df`` at
load, reads ``Baseline: Adaptive`` and its two settings from ``Program.txt``,
uses the columns in the Emulated Trigger, tells a drifting baseline from
sustained contact, and writes how each well's baseline moved.  The end-to-end
check against PyMCU's own simulator is in ``test_adaptive_baseline_sim.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pyflic.base import opto_program as op
from pyflic.base import opto_qc as oq
from pyflic.base.paradigm_state import BASELINE_COLUMNS, split_baseline_columns

V4 = Path(__file__).parent / "fixtures" / "Program_v4.txt"


def _program(baseline_lines: str) -> op.OptoProgram:
    text = V4.read_text(encoding="utf-8").replace("Baseline: Yes\n", baseline_lines + "\n")
    return op.parse_program(text)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_the_baseline_columns_come_off_the_frame() -> None:
    frame = pd.DataFrame({"W1": [5, 6, 7], "Index": [1, 2, 3],
                          **{c: ["NA", 12, 14] for c in BASELINE_COLUMNS}})
    frame = frame.replace("NA", np.nan)
    rest, recorded = split_baseline_columns(frame)
    assert list(rest.columns) == ["W1", "Index"]
    assert recorded.shape == (3, 12)
    assert np.isnan(recorded[0]).all()
    assert recorded[2, 11] == 14


def test_a_file_without_them_has_none() -> None:
    frame = pd.DataFrame({"W1": [5], "OnMs12": [0]})
    rest, recorded = split_baseline_columns(frame)
    assert recorded is None and rest is frame


# ---------------------------------------------------------------------------
# Program.txt
# ---------------------------------------------------------------------------

def test_adaptive_and_its_settings_are_read() -> None:
    program = _program("Baseline: Adaptive\nBaselineRatePerHour: 5\nBaselineCapCounts: 50")
    assert program.baseline and program.adaptive_baseline
    assert (program.baseline_rate_per_hour, program.baseline_cap_counts) == (5.0, 50)
    assert not program.skipped


def test_yes_is_not_adaptive() -> None:
    program = _program("Baseline: Yes")
    assert program.baseline and not program.adaptive_baseline
    assert program.baseline_cap_counts is None


def test_the_settings_without_adaptive_are_ignored_with_a_warning() -> None:
    program = _program("Baseline: No\nBaselineCapCounts: 50")
    assert not program.baseline and program.baseline_cap_counts is None
    assert any("only with Baseline: Adaptive" in w for w in program.skipped)


# ---------------------------------------------------------------------------
# Drift or sustained contact
# ---------------------------------------------------------------------------

def test_the_floor_is_each_minutes_tenth_percentile() -> None:
    minutes = np.repeat([0.0, 1.0], 10) + np.tile(np.arange(10) / 10, 2)
    raw = np.zeros((20, 12))
    raw[:10, 0] = np.arange(10)          # minute 0: 0..9
    raw[10:, 0] = 100 + np.arange(10)    # minute 1: 100..109
    floor = oq.resting_floor(minutes, raw)
    assert floor.shape == (20, 12)
    assert np.allclose(floor[:10, 0], 0.9) and np.allclose(floor[10:, 0], 100.9)


def test_contact_at_a_steady_baseline_is_sustained_contact() -> None:
    n = 100
    recorded = np.full((n, 12), 10.0)
    floor = np.full((n, 12), 12.0)                     # resting 2 above the baseline in force
    evidence = np.zeros(n, dtype=bool)
    evidence[20:60] = True
    drift = oq.baseline_drift_rows(recorded, floor, {1})
    assert not drift.any()
    assert oq.contact_cause(np.ones(n), evidence, drift) == oq.CAUSE_SUSTAINED_CONTACT


def test_contact_where_the_well_rests_above_its_baseline_is_drift() -> None:
    n = 100
    recorded = np.full((n, 12), 10.0)
    floor = np.full((n, 12), 10.0)
    floor[:, 2] = 25.0                                  # well 3 rests 15 above
    evidence = np.zeros(n, dtype=bool)
    evidence[20:60] = True
    assert oq.contact_cause(np.ones(n), evidence,
                            oq.baseline_drift_rows(recorded, floor, {3})) == oq.CAUSE_DRIFTING_BASELINE
    ## Another trigger well's drift doesn't count for this group.
    assert oq.contact_cause(np.ones(n), evidence,
                            oq.baseline_drift_rows(recorded, floor, {1})) == oq.CAUSE_SUSTAINED_CONTACT


def test_without_the_columns_the_cause_stays_merged() -> None:
    assert oq.contact_cause(np.ones(5), np.ones(5, dtype=bool), None) == oq.CAUSE_DRIFT


# ---------------------------------------------------------------------------
# How the baseline moved
# ---------------------------------------------------------------------------

def test_the_baseline_table_and_note() -> None:
    recorded = np.full((6, 12), 10.0)
    recorded[0] = np.nan                                # Interval 0
    recorded[3:, 0] = 20.0                              # well 1 updated once, +10
    recorded[2:, 5] = 60.0                              # well 6 reached a cap of 50
    program = _program("Baseline: Adaptive\nBaselineRatePerHour: 5\nBaselineCapCounts: 50")
    rows = {r["Well"]: r for r in oq.baseline_rows(4, recorded, program)}
    assert len(rows) == 12
    assert (rows[1]["BaselineStart"], rows[1]["BaselineEnd"], rows[1]["Updates"]) == (10.0, 20.0, 1)
    assert rows[1]["BaselineMaxMove"] == 10.0 and not rows[1]["ReachedCap"]
    assert rows[6]["ReachedCap"] and rows[6]["CapCounts"] == 50
    assert rows[2]["Updates"] == 0
    note = oq.baseline_note(list(rows.values()))
    assert "moved on 2 well(s)" in note and "well(s) 6 reached the cap" in note


def test_no_columns_no_rows() -> None:
    assert oq.baseline_rows(1, None, None) == []
