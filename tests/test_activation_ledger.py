"""The Activation ledger (``pyflic/base/activation_ledger.py``) against the simulator's truth.

Contract C3 of ``docs/paradigm-analyses-plan.md``: on every simulator spec,
every Activation whose light period is its own (not ``Overlap``) and timed
to the millisecond (``Timing`` exact) has the truth's onset, end and lit
time, from the Well's own output before Linkage (``SimTruth.lit_spans_ms``);
``K``, ``Phase`` and ``RequirementMet`` are the truth's too.  The rules for
``Overlap``, ``Boundary`` and ``Timing`` are then shown one by one, on small
hand-made states and on the specs built for them (``family_specs``).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from pyflic.base import activation_ledger as al
from pyflic.base import opto_program as op
from pyflic.base import paradigm_state as ps

from v4_fixtures import (
    FAMILY_SPEC_KEYS,
    choice_pr_spec,
    family_specs,
    fixed_ratio_spec,
    one_of_each_spec,
    program_text,
    repeating_open_loop_spec,
    simulate,
    stress_spec,
    yoked_controls_spec,
)

RECORD_MS = 200
EXPERIMENTS = {
    "choice_pr": choice_pr_spec,
    "fixed_ratio": fixed_ratio_spec,
    "one_of_each": one_of_each_spec,
    "yoked_controls": yoked_controls_spec,
    "repeating_open_loop": repeating_open_loop_spec,
}
FAMILIES = [f"family_{key}" for key in FAMILY_SPEC_KEYS]
#: Every spec but the stress one, where the rows say everything the truth does.
EXACT_SPECS = [*EXPERIMENTS, *FAMILIES]


@lru_cache(maxsize=None)
def _simulated(name: str):
    if name.startswith("family_"):
        spec = family_specs()[name.removeprefix("family_")]
    else:
        spec = stress_spec() if name == "stress" else EXPERIMENTS[name]()
    return spec, simulate(spec)


def state_of(rows: pd.DataFrame) -> ps.ParadigmState:
    _base, columns = ps.split_paradigm_columns(rows)
    minutes = rows["Index"].to_numpy(dtype=float) * RECORD_MS / 60000.0
    return ps.ParadigmState.from_frame(columns, minutes=minutes, index=rows["Index"])


def linked_of(linkage) -> dict[int, tuple[int, ...]]:
    groups: dict[int, list[int]] = {}
    for well, label in enumerate(linkage, start=1):
        groups.setdefault(label, []).append(well)
    return {well: tuple(groups[label]) for well, label in enumerate(linkage, start=1)}


@lru_cache(maxsize=None)
def ledgers(name: str):
    """``(dfm id, SimResult, ParadigmState, ledger)`` for every DFM of a spec,
    the ledger given the spec's own Linkage and the state marked with its
    ``Program.txt``'s Acclimation, as the loader marks it."""
    spec, results = _simulated(name)
    program = op.parse_program(program_text(spec))
    out = []
    for dfm in spec.dfms:
        r = results[dfm.dfm_id]
        state = state_of(r.rows)
        state.mark_acclimation(program.acclimation_rows(dfm.dfm_id, state))
        out.append((dfm.dfm_id, r, state, al.ledger(state, linked=linked_of(dfm.linkage))))
    return out


def truth_k(truth, well: int) -> list[int | None]:
    """The truth's number of each Main Activation in its Paradigm-state run:
    the firmware resets a Well when an instruction names another Paradigm
    than the last, and when the DFM restarts."""
    resets, last = [], None
    for ms, _interval, _cycle, name in truth.schedule:
        if name != last:
            resets.append(ms)
        last = name
    for down, resume, _missing in truth.restarts_ms:
        resets += [down, resume]
    resets.sort()
    counts: dict[int, int] = {}
    out: list[int | None] = []
    for ms, phase in zip(truth.activation_onset_ms[well], truth.activation_phase[well]):
        if phase == "A":
            out.append(None)
            continue
        run = int(np.searchsorted(resets, ms, side="right"))
        counts[run] = counts.get(run, 0) + 1
        out.append(counts[run])
    return out


def _value(v):
    return None if pd.isna(v) else int(v)


ALL_BOUNDARIES = (al.BOUNDARY_NONE, al.BOUNDARY_INTERVAL, al.BOUNDARY_GAP, al.BOUNDARY_END)


def compare_with_truth(r, book, boundaries=ALL_BOUNDARIES) -> dict[str, list]:
    """Each ledger row beside the truth, in the truth's order: the
    unoverlapped, exact ones whose ``Boundary`` is in *boundaries* must
    have the truth's times (counted under ``"timed"``).  Returns the rows
    whose K, Phase or RequirementMet differ, by column, as (well, row)."""
    differ: dict[str, list] = {"K": [], "Phase": [], "RequirementMet": []}
    timed = 0
    for well in range(1, 13):
        mine = book[book["Well"] == well]
        onsets = r.truth.activation_onset_ms[well]
        assert len(mine) == len(onsets), well
        spans = dict(r.truth.lit_spans_ms[well])
        for row, t0, phase, met, k in zip(mine.itertuples(), onsets, r.truth.activation_phase[well],
                                          r.truth.requirement_met[well], truth_k(r.truth, well)):
            if not row.Overlap and row.Timing == al.TIMING_EXACT and row.Boundary in boundaries:
                end = spans.get(t0)
                assert (row.OnsetMs, row.EndMs, row.LitMs) == (t0, end, end - t0), \
                    (well, row.Row, row.Boundary, t0, end)
                timed += 1
            if _value(row.K) != k:
                differ["K"].append((well, row.Row))
            if row.Phase != phase:
                differ["Phase"].append((well, row.Row))
            if _value(row.RequirementMet) != met:
                differ["RequirementMet"].append((well, row.Row))
    differ["timed"] = timed
    return differ


# ---------------------------------------------------------------------------
# The truth, on every spec
# ---------------------------------------------------------------------------

#: Activations timed exactly on each spec, at the default length and seed:
#: a change to the simulator or the rules shows here.
TIMED = {"choice_pr": 308, "fixed_ratio": 159, "one_of_each": 1973, "yoked_controls": 363,
         "repeating_open_loop": 0, "family_closed_max": 570, "family_fixed_interval": 784,
         "family_non_feeding": 1043, "family_exponential": 96, "family_restart": 696,
         "stress": 1983}


@pytest.mark.parametrize("name", EXACT_SPECS)
def test_the_ledger_is_the_truths(name: str) -> None:
    """Every Activation, in the truth's order; the unoverlapped exact ones to
    the millisecond; K, Phase and RequirementMet exactly."""
    timed = 0
    for _dfm, r, state, book in ledgers(name):
        assert list(book.columns) == list(al.LEDGER_COLUMNS)
        for well in range(1, 13):
            assert book.loc[book["Well"] == well, "Row"].tolist() == \
                state.activation_onsets(well).tolist()
        differ = compare_with_truth(r, book)
        timed += differ.pop("timed")
        assert differ == {"K": [], "Phase": [], "RequirementMet": []}
    assert timed == TIMED[name]


def test_the_ledger_is_the_truths_on_the_stress_spec() -> None:
    """The stress spec's Intervals start within a record, so the first row
    of a new Paradigm holds the old one's last milliseconds: an Activation
    counted there may be the old run's, which the rows cannot say.  Every
    timing is still the truth's, and K, Phase and RequirementMet differ only
    in a state run whose first row holds such an Activation (K carries the
    one it counted on), or, for RequirementMet, on a Paradigm's first row."""
    timed = checked = 0
    ambiguous_runs = set()
    for dfm_id, r, state, book in ledgers("stress"):
        starts = np.array([a for a, _b in state.state_runs()])
        first_rows = set(starts[1:].tolist())

        def run_of(row: int) -> int:
            return int(np.searchsorted(starts, row, side="right") - 1)

        differ = compare_with_truth(r, book)
        timed += differ.pop("timed")
        checked += len(book)
        held = {(w, run_of(row)) for w, row in zip(book["Well"], book["Row"]) if row in first_rows}
        for column in ("K", "Phase"):
            for well, row in differ[column]:
                assert (well, run_of(row)) in held, (dfm_id, column, well, row)
                ambiguous_runs.add((dfm_id, well, run_of(row)))
        for well, row in differ["RequirementMet"]:
            assert row in first_rows or row - 1 in first_rows, (dfm_id, well, row)
    assert timed == TIMED["stress"] and checked > 5500
    assert 0 < len(ambiguous_runs) <= 10


def test_the_truth_mostly_counts_a_first_row_activation_in_the_new_run() -> None:
    """Why the ledger counts an Activation on a run's first row in the new
    run: a Paradigm change resets every Well, and one that stays lit counts
    a new Activation at once, so most of them are the new run's."""
    new = old = 0
    for _dfm, r, state, book in ledgers("stress"):
        changes = [ms for ms, *_ in r.truth.schedule]
        first_rows = {a for a, _b in state.state_runs()[1:]}
        for well, row in zip(book["Well"], book["Row"]):
            at = [ms for ms in changes if ms // RECORD_MS == row]
            if row not in first_rows or not at:
                continue
            onsets = [ms for ms in r.truth.activation_onset_ms[well] if ms // RECORD_MS == row]
            new += sum(ms >= at[0] for ms in onsets)
            old += sum(ms < at[0] for ms in onsets)
    assert new > 10 * old > 0


def test_the_leaving_row_of_a_short_acclimation_reads_acclimation() -> None:
    """Under PR1 (ActivationMs 60) an Acclimation Activation can start and
    end in the record where the well leaves Acclimation, which reads M: the
    Requirement did not advance there, so the ledger calls it Acclimation."""
    seen = 0
    for _dfm, r, state, book in ledgers("stress"):
        for row in book.itertuples():
            c = row.Well - 1
            if row.Row > 0 and state.phase_a[row.Row - 1, c] and not state.phase_a[row.Row, c] \
                    and not np.isnan(state.requirement[row.Row, c]) \
                    and state.requirement[row.Row, c] == state.requirement[row.Row - 1, c] \
                    and row.Row not in {a for a, _b in state.state_runs()}:
                assert row.Phase == "A" and pd.isna(row.K)
                seen += 1
    assert seen > 0


# ---------------------------------------------------------------------------
# The rules, on hand-made states
# ---------------------------------------------------------------------------

def synthetic(n: int, *, on: dict | None = None, onsets: dict | None = None,
              paradigm: list | None = None, interval: list | None = None,
              index: list | None = None, phase: dict | None = None,
              requirement: dict | None = None) -> ps.ParadigmState:
    """A state of *n* rows: *on* maps a Well to its ``OnMs`` per row, *onsets*
    to the rows its Activations are counted on (a row twice for two)."""
    on, onsets, phase, requirement = on or {}, onsets or {}, phase or {}, requirement or {}
    frame: dict[str, object] = {
        "Interval": pd.array(interval or [1] * n, dtype="Int64"),
        "Cycle": pd.array([1] * n, dtype="Int64"),
        "Paradigm": pd.array(paradigm or ["P"] * n, dtype="string"),
    }
    for w in range(1, 13):
        frame[f"Phase{w}"] = pd.array(list(phase.get(w, "M" * n)), dtype="string")
    for w in range(1, 13):
        counts = np.cumsum(np.bincount(np.asarray(onsets.get(w, []), dtype=int), minlength=n))
        frame[f"Activations{w}"] = pd.array(counts[:n], dtype="Int64")
    for w in range(1, 13):
        frame[f"Requirement{w}"] = pd.array(requirement.get(w, [None] * n), dtype="Int64")
    for w in range(1, 13):
        frame[f"Progress{w}"] = pd.array([None] * n, dtype="Int64")
    for w in range(1, 13):
        frame[f"OnMs{w}"] = pd.array(on.get(w, [0] * n), dtype="Int64")
    rows = np.arange(1, n + 1) if index is None else np.asarray(index)
    return ps.ParadigmState.from_frame(pd.DataFrame(frame), minutes=rows * RECORD_MS / 60000.0,
                                       index=rows)


def only(book: pd.DataFrame, well: int = 1) -> pd.Series:
    (row,) = [r for _i, r in book[book["Well"] == well].iterrows()]
    return row


def test_a_light_period_is_timed_from_its_edges() -> None:
    state = synthetic(9, on={1: [0, 0, 130, 200, 200, 70, 0, 0, 0]}, onsets={1: [2]})
    row = only(al.ledger(state))
    ## Row 2 is Index 3, ms 400-600: lit for its last 130 ms.
    assert (row.Row, row.EndRow, row.Index) == (2, 5, 3)
    assert (row.OnsetMs, row.EndMs, row.LitMs) == (470, 1070, 600)
    assert row.Timing == al.TIMING_EXACT and not row.Overlap and row.Boundary == al.BOUNDARY_NONE
    assert (row.Phase, row.K, row.Paradigm, row.ProgramInterval, row.Cycle) == ("M", 1, "P", 1, 1)
    assert pd.isna(row.SinceLastOnsetMs) and pd.isna(row.RequirementMet)
    assert row.OnsetMin == pytest.approx(3 * RECORD_MS / 60000.0)


@pytest.mark.parametrize("on_ms", [
    [0, 0, 50, 0, 0, 0],           # one row: where in it the light was is unknown
    [0, 0, 130, 150, 200, 70],     # an interior row not whole
    [0, 0, 130, 230, 70, 0],       # a late record
])
def test_timing_is_record_unless_the_edges_fix_the_period(on_ms) -> None:
    row = only(al.ledger(synthetic(len(on_ms), on={1: on_ms}, onsets={1: [2]})))
    assert row.Timing == al.TIMING_RECORD
    assert row.LitMs == sum(on_ms)


def test_a_single_row_period_ends_its_lit_time_after_the_onset() -> None:
    row = only(al.ledger(synthetic(5, on={1: [0, 0, 50, 0, 0]}, onsets={1: [2]})))
    assert (row.OnsetMs, row.EndMs, row.EndRow) == (550, 600, 2)


def test_light_already_on_is_an_overlap() -> None:
    ## W1 lit by its partner W2 (no Activation of its own) before its onset.
    state = synthetic(8, on={1: [0, 90, 130, 200, 70, 0, 0, 0]}, onsets={1: [2]})
    assert only(al.ledger(state)).Overlap
    ## A dark row between: the light came on in row 2.
    state = synthetic(8, on={1: [90, 0, 130, 200, 70, 0, 0, 0]}, onsets={1: [2]})
    assert not only(al.ledger(state)).Overlap


def test_a_break_in_index_before_the_onset_is_an_overlap() -> None:
    """Records lost before the onset row may hold the light's start: the
    Activations they counted show on the row after them."""
    state = synthetic(8, on={1: [0, 0, 130, 200, 70, 0, 0, 0]}, onsets={1: [2]},
                      index=[1, 2, 9, 10, 11, 12, 13, 14])
    row = only(al.ledger(state))
    assert row.Overlap and (row.OnsetMs, row.EndMs) == (9 * RECORD_MS - 130, 11 * RECORD_MS + 70 - 200)


def test_a_partners_onset_in_the_period_is_an_overlap() -> None:
    on = [0, 0, 130, 200, 200, 200, 60, 0, 0, 0]
    state = synthetic(10, on={1: on, 3: on}, onsets={1: [2], 3: [4]})
    book = al.ledger(state, linked={1: (1, 3), 3: (1, 3)})
    assert only(book, 1).Overlap and only(book, 3).Overlap
    ## Unlinked, W1's period is its own (W3's still begins lit).
    book = al.ledger(state, linked={1: (1,), 3: (3,)})
    assert not only(book, 1).Overlap and only(book, 3).Overlap


def test_light_running_on_into_the_next_activation_is_an_overlap() -> None:
    """Held contact under a FixedInterval lockout shorter than ActivationMs:
    the light never goes out between Activations, so neither length can be
    read."""
    on = [0, 0, 130, 200, 200, 200, 200, 200, 90, 0]
    book = al.ledger(synthetic(10, on={1: on}, onsets={1: [2, 6]}))
    first, second = (r for _i, r in book.iterrows())
    assert (first.EndRow, second.EndRow) == (5, 8)
    assert first.Overlap and second.Overlap
    ## Out for part of the row before the next onset: the first one ended there.
    on[5] = 150
    first, second = (r for _i, r in al.ledger(synthetic(10, on={1: on}, onsets={1: [2, 6]})).iterrows())
    assert not first.Overlap and first.EndMs == 5 * RECORD_MS + 150
    assert second.Overlap                       # the row before it was lit


def test_two_onsets_in_one_row_overlap() -> None:
    on = [0, 0, 160, 200, 40, 0, 0]
    book = al.ledger(synthetic(7, on={1: on}, onsets={1: [2, 2]}))
    first, second = (r for _i, r in book.iterrows())
    assert first.Overlap and second.Overlap
    assert (first.Row, first.EndRow, second.Row, second.EndRow) == (2, 2, 2, 4)
    assert (first.K, second.K) == (1, 2)


def test_the_boundary_says_what_stopped_the_period() -> None:
    on = [0, 0, 130, 200, 200, 70, 0, 0, 0, 0]
    ## A Paradigm start inside the period.
    state = synthetic(10, on={1: on}, onsets={1: [2]}, paradigm=["P"] * 4 + ["Q"] * 6)
    assert only(al.ledger(state)).Boundary == al.BOUNDARY_INTERVAL
    ## A Program Interval change in the row after it.
    state = synthetic(10, on={1: on}, onsets={1: [2]}, interval=[1] * 6 + [2] * 4)
    assert only(al.ledger(state)).Boundary == al.BOUNDARY_INTERVAL
    ## Two rows after it is the rule's doing.
    state = synthetic(10, on={1: on}, onsets={1: [2]}, interval=[1] * 7 + [2] * 3)
    assert only(al.ledger(state)).Boundary == al.BOUNDARY_NONE
    ## A break in Index right after it: records lost, or a restart.
    state = synthetic(10, on={1: [0, 0, 130, 200, 200, 200, 0, 0, 0, 0]}, onsets={1: [2]},
                      index=[1, 2, 3, 4, 5, 6, 30, 31, 32, 33])
    row = only(al.ledger(state))
    assert row.Boundary == al.BOUNDARY_GAP and row.EndRow == 5
    ## The last row: lit to the end of the recording, with no Activation after.
    state = synthetic(6, on={1: [0, 0, 130, 200, 200, 200]}, onsets={1: [2]})
    row = only(al.ledger(state))
    assert row.Boundary == al.BOUNDARY_END and not row.Overlap and row.EndMs == 6 * RECORD_MS
    ## A gap is named before a Paradigm start after it (a restart has both).
    state = synthetic(10, on={1: [0, 0, 130, 200, 200, 200, 0, 0, 0, 0]}, onsets={1: [2]},
                      index=[1, 2, 3, 4, 5, 6, 30, 31, 32, 33],
                      paradigm=["P"] * 6 + [None] * 2 + ["P"] * 2)
    assert only(al.ledger(state)).Boundary == al.BOUNDARY_GAP


def test_k_and_since_last_onset_count_within_a_state_run() -> None:
    """P acclimates in rows 1-3; Q, from row 5, acclimates again, then
    counts its Main Activations from 1."""
    on = [0, 50, 0, 50, 0, 50, 0, 50, 0, 50, 0, 0]
    state = synthetic(12, on={1: on}, onsets={1: [1, 3, 5, 7, 9]},
                      paradigm=["P"] * 5 + ["Q"] * 7, phase={1: "MAAAMAAMMMMM"})
    book = al.ledger(state)
    assert book["Phase"].tolist() == ["A", "A", "A", "M", "M"]
    assert [_value(k) for k in book["K"]] == [None, None, None, 1, 2]
    assert [_value(v) for v in book["SinceLastOnsetMs"]] == [None, 400, None, 400, 400]
    assert book["Paradigm"].tolist() == ["P", "P", "Q", "Q", "Q"]


def _well_one_acclimation(n: int, rows: slice, count: int) -> np.ndarray:
    """``ParadigmState.mark_acclimation``'s array: Well 1 runs a Paradigm
    with Acclimation *count* in *rows*."""
    out = np.zeros((n, 12), dtype=np.int64)
    out[rows, 0] = count
    return out


def test_a_well_set_on_starts_its_acclimation_where_it_first_runs() -> None:
    """A Well set On for an Interval runs no Paradigm, and the next Interval
    of the same Paradigm keeps state, so the Acclimation the run's reset left
    it starts where it first runs: two Acclimation Activations shorter than a
    record can start and end in that row, which reads M (stress_spec's DFM 2
    Well 10 under PyMCU 4.0.1's Starting baseline).  The rows cannot say so;
    Program.txt's Acclimation, marked on the state, can."""
    n = 8
    state = synthetic(n, on={1: [0, 0, 0, 0, 120, 0, 50, 0]}, onsets={1: [4, 4, 6]},
                      interval=[1] * 4 + [2] * 4, requirement={1: [None] * 4 + [2, 2, 3, 3]})
    assert al.ledger(state)["Phase"].tolist() == ["M", "M", "M"]
    state.mark_acclimation(_well_one_acclimation(n, slice(4, n), 2))
    book = al.ledger(state)
    assert book["Phase"].tolist() == ["A", "A", "M"]
    assert [_value(k) for k in book["K"]] == [None, None, 1]


def test_only_a_clipped_first_run_hides_where_a_well_first_ran() -> None:
    """A range that starts mid-run cannot show the run's reset, so its first
    run keeps the rows' Phase; the next run, Q from row 2, is whole, and the
    Acclimation marked carries into the range's state."""
    n = 8
    state = synthetic(n, on={1: [0, 0, 0, 0, 120, 0, 50, 0]}, onsets={1: [4, 4, 6]},
                      paradigm=["P"] * 2 + ["Q"] * 6, interval=[1] * 4 + [2] * 4,
                      requirement={1: [None] * 4 + [2, 2, 3, 3]})
    state.mark_acclimation(_well_one_acclimation(n, slice(4, n), 2))
    assert al.ledger(state.subset(np.arange(1, n)))["Phase"].tolist() == ["A", "A", "M"]
    assert al.ledger(state.subset(np.arange(3, n)))["Phase"].tolist() == ["M", "M", "M"]


def test_a_well_back_from_on_leaves_acclimation_against_its_last_running_row() -> None:
    """A Well in Acclimation set On for an Interval keeps its state; back
    under the ratio Paradigm it can leave Acclimation in its first row, which
    reads M.  That row is compared with the Well's last row under the
    Paradigm (row 2), not the On row before it, which shows no Requirement."""
    state = synthetic(8, on={1: [0, 0, 0, 0, 0, 60, 0, 0]}, onsets={1: [5]},
                      phase={1: "MAAMMMMM"}, interval=[1, 1, 1, 2, 2, 3, 3, 3],
                      requirement={1: [None, 2, 2, None, None, 2, 2, 2]})
    assert al.ledger(state)["Phase"].tolist() == ["A"]


def test_a_two_activation_leaving_row_has_its_last_main() -> None:
    """Under a ratio Paradigm the row where the well leaves Acclimation with
    its Requirement advanced holds an Acclimation Activation and then a
    Main one; with the Requirement unchanged, Acclimation ones only."""
    on = [0, 0, 120, 0, 0, 0]
    advanced = synthetic(6, on={1: on}, onsets={1: [2, 2]}, phase={1: "MAMMMM"},
                         requirement={1: [None, 1, 2, 2, 2, 2]})
    assert al.ledger(advanced)["Phase"].tolist() == ["A", "M"]
    assert [_value(k) for k in al.ledger(advanced)["K"]] == [None, 1]
    same = synthetic(6, on={1: on}, onsets={1: [2]}, phase={1: "MAMMMM"},
                     requirement={1: [None, 1, 1, 1, 1, 1]})
    assert al.ledger(same)["Phase"].tolist() == ["A"]
    ## Without a Requirement (FixedInterval) the row's own Phase stands.
    plain = synthetic(6, on={1: on}, onsets={1: [2]}, phase={1: "MAMMMM"})
    assert al.ledger(plain)["Phase"].tolist() == ["M"]


def test_the_data_give_the_linkage_without_one() -> None:
    """Without *linked*, Wells with Activations whose OnMs are identical are
    linked, as the firmware's Linkage makes them."""
    for name in ("stress", "family_fixed_interval", "yoked_controls"):
        for _d, _r, state, book in ledgers(name):
            pd.testing.assert_frame_equal(al.ledger(state), book)


def test_an_empty_state_or_no_wells_give_the_columns() -> None:
    state = synthetic(5)
    for book in (al.ledger(state), al.ledger(state, wells=[]), al.ledger(state.subset(np.zeros(5, bool)))):
        assert list(book.columns) == list(al.LEDGER_COLUMNS) and book.empty
    with pytest.raises(ValueError):
        al.ledger(state, wells=[13])


def test_the_ledger_is_typed() -> None:
    _dfm, _r, _state, book = ledgers("choice_pr")[0]
    for column in ("Well", "Row", "EndRow", "Index", "OnsetMs", "EndMs", "LitMs"):
        assert book[column].dtype == np.int64, column
    for column in ("SinceLastOnsetMs", "K", "ProgramInterval", "Cycle", "RequirementMet"):
        assert str(book[column].dtype) == "Int64", column
    assert book["Overlap"].dtype == bool and str(book["Paradigm"].dtype) == "string"
    assert set(book["Timing"]) <= {al.TIMING_EXACT, al.TIMING_RECORD}
    assert set(book["Boundary"]) <= {al.BOUNDARY_NONE, al.BOUNDARY_INTERVAL, al.BOUNDARY_GAP,
                                     al.BOUNDARY_END}
    assert book.equals(book.sort_values(["Well", "Row"], kind="stable"))


def test_activation_rows_mark_each_period() -> None:
    on = [0, 0, 130, 200, 70, 0, 50, 0, 0]
    book = al.ledger(synthetic(9, on={1: on}, onsets={1: [2, 6]}))
    assert np.flatnonzero(al.activation_rows(book, 1, 9)).tolist() == [2, 3, 4, 6]
    assert np.flatnonzero(al.activation_rows(book, 1, 9, onset_row=False)).tolist() == [3, 4]
    assert not al.activation_rows(book, 2, 9).any()


# ---------------------------------------------------------------------------
# The specs built for the rules
# ---------------------------------------------------------------------------

def test_held_contact_under_a_short_lockout_overlaps() -> None:
    """FI500 (LockoutMs 500, ActivationMs 1500): held contact gives a new
    Activation every 1.5 s in one unbroken light, which the truth's own
    output shows; the ledger marks each such Activation."""
    (_d1, _r1, _s1, _b1), (_d2, r, state, book) = ledgers("family_fixed_interval")
    ran_on = 0
    for well in range(1, 13, 2):
        mine = book[book["Well"] == well].reset_index(drop=True)
        onsets = r.truth.activation_onset_ms[well]
        spans = r.truth.lit_spans_ms[well]
        for i in range(len(mine) - 1):
            end_row = int(mine.loc[i, "EndRow"])
            if mine.loc[i + 1, "Row"] == end_row + 1 and state.on_ms[end_row, well - 1] == RECORD_MS:
                assert mine.loc[i, "Overlap"] and mine.loc[i + 1, "Overlap"]
                ## The truth's light did run on, one span holding both onsets.
                if any(a <= onsets[i] and onsets[i + 1] < b for a, b in spans):
                    ran_on += 1
    assert ran_on > 100


def test_a_restart_cuts_the_light_and_starts_k_again() -> None:
    """The restart spec: DFM 2's ClosedLoop Activation lit when the DFM went
    down ends at the gap; DFM 1's FixedRatio wells acclimate again and count
    their Main Activations from 1 after it, as the firmware's Requirement
    series starts again."""
    (_d1, r1, state1, book1), (_d2, r2, state2, book2) = ledgers("family_restart")
    ## DFM 2: the one Activation the outage cut.
    (down, _resume, _missing), = r2.truth.restarts_ms
    (span,) = state2.restarts()
    cut = book2[book2["Boundary"] == al.BOUNDARY_GAP]
    assert len(cut) == 1
    row = cut.iloc[0]
    assert row.Well == 1 and row.EndRow == span.row - 1 and row.EndMs == down
    assert not row.Overlap and row.Timing == al.TIMING_EXACT
    ## DFM 1: K and the Acclimation start again at the restart.
    (span,) = state1.restarts()
    for well in range(1, 13, 2):
        mine = book1[book1["Well"] == well]
        before = mine[mine["Row"] < span.row]
        after = mine[mine["Row"] >= span.resume_row]
        assert before["K"].max() > 1
        assert after["Phase"].tolist()[:2] == ["A", "A"]
        main = after[after["Phase"] == "M"]
        assert [_value(k) for k in main["K"]] == list(range(1, len(main) + 1)) and len(main)
        assert pd.isna(after["SinceLastOnsetMs"].iloc[0])
        ## FixedRatio 3 meets 3 each time, before and after.
        assert set(main["RequirementMet"].dropna()) == {3}


def test_lost_records_leave_the_other_timings_exact() -> None:
    """Records lost from Index: the periods they cut end at a gap, an
    Activation counted on the row after them is not timed, and every other
    timing is still the truth's."""
    spec, results = _simulated("choice_pr")
    r = results[1]
    lit = r.rows["OnMs1"].to_numpy(dtype=float) > 0
    after = int(r.rows["Index"][np.flatnonzero(lit)[400]])        # a lit record of W1
    keep = ~((r.rows["Index"] > after) & (r.rows["Index"] <= after + 40))
    state = state_of(r.rows[keep].reset_index(drop=True))
    book = al.ledger(state, linked=linked_of(spec.dfms[0].linkage))
    cut = book[book["Boundary"] == al.BOUNDARY_GAP]
    assert 1 in set(cut["Well"])
    ## The light carried on through the lost records, so what the cut
    ## periods measure is not their length: the truth's is longer.
    differ = compare_with_truth(r, book, boundaries=(al.BOUNDARY_NONE, al.BOUNDARY_END))
    assert differ.pop("timed") > 100
    for row in cut.itertuples():
        end = dict(r.truth.lit_spans_ms[row.Well]).get(row.OnsetMs)
        assert end is None or end > row.EndMs
    ## An Activation counted on the row after them is never timed.
    after_gap = int(np.flatnonzero(np.diff(state.index) > 1)[0]) + 1
    assert book.loc[book["Row"] == after_gap, "Overlap"].all()
