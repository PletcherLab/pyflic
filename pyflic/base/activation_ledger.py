"""The Activation ledger: one row per Activation, timed to the millisecond.

Firmware 4.0.0 records, per Well and per 200 ms record, how many milliseconds
the LED was lit (``OnMs``, after Linkage) and how many Activations the Well's
own Paradigm has counted.  An Activation is counted in the record where its
light starts, so that record's ``OnMs`` says how long before the record's
end the light came on; the last record of the light period says, the same
way, how long after its start it went out.  The ledger reads each
Activation's onset, end and lit time from those edges
(``docs/paradigm-analyses-plan.md``, decision 4 and contract C3).

Every family check and summary of the Paradigm analyses aggregates this
table; nothing downstream re-reads the rows.  The columns are
:data:`LEDGER_COLUMNS`:

* ``Well``, ``Row``, ``EndRow``, ``Index``, ``OnsetMin``: the Well (1-12),
  the row its Activation was counted on, the light period's last row
  (inclusive), the record's ``Index`` and pyflic's ``Minutes`` at ``Row``.
* ``OnsetMs``, ``EndMs``, ``LitMs``: the light period in milliseconds from
  the start of the record with ``Index`` 1, the time base of the simulator's
  and PyMCU's truth.  pyflic's ``Minutes`` count from the first row's time
  stamp instead, a sub-second offset away, so ``OnsetMin`` is never derived
  from ``OnsetMs``.
* ``Timing``: :data:`TIMING_EXACT` when the period's edge records fix both
  ends to the millisecond, else :data:`TIMING_RECORD` (good to one record).
* ``Overlap``: the period's length cannot be the Activation's own, because
  other light merges with it: a linked partner's, light already on, or the
  Well's next Activation.  Checks skip these rows.
* ``Boundary``: why the period stopped other than by the rule:
  :data:`BOUNDARY_GAP`, :data:`BOUNDARY_INTERVAL` or :data:`BOUNDARY_END`;
  :data:`BOUNDARY_NONE` (empty) otherwise.
* ``SinceLastOnsetMs``: from the same Well's previous onset in the same
  Paradigm-state run (``ParadigmState.state_runs``), any Phase.
* ``Phase``, ``K``: ``A`` or ``M`` at ``Row``, and the Activation's number
  among the Well's Phase-``M`` Activations of its state run (``NA`` in
  ``A``).  Under a ratio Paradigm the k-th Main Activation meets
  ``requirement_for(k)``.  A restart starts a new run, so ``K`` starts over
  after one, as the firmware's Requirement series does.  Three edges, where a
  record holds two states:

  - the row where a Well leaves Acclimation under a ratio Paradigm reads
    ``M``, but its Activations are Acclimation ones unless the Requirement
    advanced there (only a Main Activation advances it), and then only the
    row's last is Main, as ``ParadigmState.requirement_met`` reads that row.
    It is compared with the Well's last row under the Paradigm in its state
    run, which is the row before unless the Well was set On or Off between;
  - a Well's first row under a Paradigm with Acclimation in a state run
    (the run's first row, or a later Interval's for a Well set On or Off
    until then: the same Paradigm sent again keeps state) can hold the
    whole Acclimation and read ``M``.  The rows cannot say so;
    ``ParadigmState.acclimation``, marked from ``Program.txt`` by the
    loader, makes the row's first N Activations Phase ``A``.  Without it
    they read ``M``;
  - the first row of a state run shows the new Paradigm, but an
    instruction that arrives within a record leaves the record's start to
    the old one, so an Activation counted there may be the old run's.  The
    rows cannot say which (``requirement_met`` shows no Requirement there),
    and the ledger counts it in the new run.
* ``Paradigm``, ``ProgramInterval``, ``Cycle``: at ``Row``.
* ``RequirementMet``: ``ParadigmState.requirement_met``.  The first row of a
  new Paradigm shows none; filling it from the program is the consumer's job.

Pure numpy and pandas over a :class:`~pyflic.base.paradigm_state.ParadigmState`:
no program, no pyflic feeding events and no Qt.  ``EndedBy``, which needs the
program's settings and pyflic's contact, is stream S1's (contract C4).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .paradigm_state import N_WELLS, RECORD_MS, ParadigmState

#: The ledger's columns, in order (contract C3).
LEDGER_COLUMNS: tuple[str, ...] = (
    "Well", "Row", "EndRow", "Index", "OnsetMin", "OnsetMs", "EndMs", "LitMs",
    "Timing", "Overlap", "Boundary", "SinceLastOnsetMs", "Phase", "K",
    "Paradigm", "ProgramInterval", "Cycle", "RequirementMet",
)

#: ``Timing``: both edges exact to the millisecond, or good to one record.
TIMING_EXACT, TIMING_RECORD = "exact", "record"

#: ``Boundary``: the period stopped at a Paradigm start or Program Interval
#: change, at a break in ``Index``, or at the last row; empty when nothing
#: but the Paradigm's rule can have ended it.
BOUNDARY_INTERVAL, BOUNDARY_GAP, BOUNDARY_END = "interval", "gap", "end"
BOUNDARY_NONE = ""

_INTEGER = ("Well", "Row", "EndRow", "Index", "OnsetMs", "EndMs", "LitMs")
_NULLABLE = ("SinceLastOnsetMs", "K", "ProgramInterval", "Cycle", "RequirementMet")


def _typed(frame: pd.DataFrame) -> pd.DataFrame:
    """The ledger's dtypes: whole numbers int64, or nullable ``Int64`` where
    a value can be missing; ``Paradigm`` a string column."""
    for column in _INTEGER:
        frame[column] = frame[column].astype(np.int64)
    for column in _NULLABLE:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        missing = np.isnan(values)
        frame[column] = pd.arrays.IntegerArray(
            np.where(missing, 0, np.round(values)).astype(np.int64), missing)
    frame["OnsetMin"] = frame["OnsetMin"].astype(float)
    frame["Overlap"] = frame["Overlap"].astype(bool)
    for column in ("Timing", "Boundary", "Phase"):
        frame[column] = frame[column].astype(object)
    frame["Paradigm"] = frame["Paradigm"].astype("string")
    return frame


def _empty() -> pd.DataFrame:
    return _typed(pd.DataFrame({c: [] for c in LEDGER_COLUMNS}))


def _data_linkage(state: ParadigmState) -> dict[int, tuple[int, ...]]:
    """The Wells with Activations whose ``OnMs`` match over the whole state:
    the Linkage the data show without a program (``OnMs`` counts light after
    Linkage, so linked Wells match to the millisecond)."""
    linked: dict[int, tuple[int, ...]] = {}
    for group in state.identical_on_ms_groups(state.wells_with_activations()):
        for well in group:
            linked[well] = tuple(w for w in group if w != well)
    return linked


def _cumulative(mask: np.ndarray) -> np.ndarray:
    """``out[i]``: how many of ``mask[:i]`` are true (length n + 1)."""
    return np.concatenate(([0], np.cumsum(mask, dtype=np.int64)))


def _nullable_at(values: pd.arrays.IntegerArray, rows: np.ndarray) -> np.ndarray:
    return values[rows].to_numpy(dtype=float, na_value=np.nan)


def _unseen_at_first_row(state: ParadigmState, c: int, rows: np.ndarray,
                         run_of_row: np.ndarray) -> np.ndarray:
    """Per onset of Well ``c + 1`` (onset rows *rows*, in order): one of the
    first N Activations counted on the Well's first row under a Paradigm with
    Acclimation N in a state run, where that row reads ``M``.

    The run's start reset every Well into Acclimation, and a Well set On or
    Off runs no Paradigm, so it keeps that state until its first row under
    one, which is the run's first row or a later Interval's (the same
    Paradigm sent again keeps state).  An Acclimation shorter than a record
    can start and end in that row, and the row before has no Acclimation to
    compare with.  The rows cannot say where a Well first ran, or N;
    ``ParadigmState.acclimation`` can, from ``Program.txt``, and is 0
    without it.  A range that clipped the first run's start leaves it out:
    its reset is not in the rows."""
    out = np.zeros(rows.size, dtype=bool)
    counts = state.acclimation[:, c]
    runs = np.flatnonzero(counts > 0)
    if not runs.size or not rows.size:
        return out
    _ids, first = np.unique(run_of_row[runs], return_index=True)
    starts = runs[first]
    if state.activations_before is not None:
        starts = starts[run_of_row[starts] > 0]
    starts = starts[~state.phase_a[starts, c]]
    rank = np.arange(rows.size) - np.searchsorted(rows, rows, side="left")
    return np.isin(rows, starts) & (rank < counts[rows])


def ledger(state: ParadigmState, *, wells: Iterable[int] | None = None,
           linked: Mapping[int, Sequence[int]] | None = None,
           record_ms: int = RECORD_MS) -> pd.DataFrame:
    """One row per Activation (``state.activation_onsets(well)``) of each
    Well of *wells* (default ``state.wells_with_activations()``), sorted by
    ``Well``, then ``Row``; two Activations counted in one row keep their
    order.  The columns are :data:`LEDGER_COLUMNS`.

    *linked* maps a Well to the Wells sharing its light, its Linkage Group
    (a Well listed as its own partner is ignored).  Without it the Linkage is
    the data's: Wells with Activations whose ``OnMs`` are identical.

    Row ``r`` spans ``[(index[r] - 1) * record_ms, index[r] * record_ms)``.
    An Activation's light period runs from its row through the following rows
    that continue ``Index`` without a break and have the Well's ``OnMs`` above
    0, cut before the Well's next onset:

    * ``LitMs`` is ``OnMs`` summed over ``[Row, EndRow]``; ``OnsetMs`` the end
      of ``Row`` less ``OnMs[Row]``; ``EndMs`` the start of ``EndRow`` plus
      ``OnMs[EndRow]`` when ``EndRow > Row``, else ``OnsetMs + LitMs``.
    * ``Timing`` is exact when the period spans more than one row, its
      interior rows are whole records and no row exceeds one.
    * ``Overlap`` when a linked partner has an onset in ``[Row, EndRow]``;
      the row before ``Row`` was lit, with no break between (the LED was
      already on), or a break in ``Index`` comes right before ``Row`` (the
      records lost may hold the light's start); the Well's next onset is the
      row after ``EndRow`` and
      ``OnMs[EndRow]`` is the whole record (the light ran on into it, as held
      contact does under a FixedInterval lockout shorter than ActivationMs);
      or two of the Well's onsets fall in ``Row``.
    * ``Boundary`` is ``gap`` for a break in ``Index`` right after ``EndRow``
      (records lost, or a restart), else ``interval`` for a Paradigm start
      or Program Interval change in ``(Row, EndRow + 1]``, else ``end`` when
      ``EndRow`` is the last row, else empty.
    """
    n = len(state)
    chosen = sorted({int(w) for w in (state.wells_with_activations() if wells is None
                                       else wells)})
    for w in chosen:
        if not 1 <= w <= N_WELLS:
            raise ValueError(f"wells are 1-{N_WELLS}, got {w!r}")
    if not n or not chosen:
        return _empty()
    P = int(record_ms)
    partners = _data_linkage(state) if linked is None else \
        {int(w): tuple(int(x) for x in ws if int(x) != int(w)) for w, ws in linked.items()}

    index = np.asarray(state.index, dtype=np.int64)
    ## A row continues the one before when its Index is the next one; any
    ## other step (records lost, a restart, a repeat) breaks the time base.
    joined = np.zeros(n, dtype=bool)
    joined[1:] = np.diff(index) == 1
    break_after = np.ones(n, dtype=bool)
    break_after[:-1] = ~joined[1:]

    ## Paradigm starts and Program Interval changes: a light period crossing
    ## one may have been stopped, or carried on, by the program.
    run_starts = np.array([a for a, _b in state.state_runs()], dtype=np.int64)
    run_of_row = np.searchsorted(run_starts, np.arange(n), side="right") - 1
    codes = state.program_interval.to_numpy(dtype=np.int64, na_value=-1)
    change = np.zeros(n, dtype=bool)
    change[run_starts] = True
    change[1:] |= codes[1:] != codes[:-1]
    change[0] = False
    cum_change = _cumulative(change)

    onsets_of = {w: state.activation_onsets(w).astype(np.int64)
                 for w in set(chosen) | {p for w in chosen for p in partners.get(w, ())}}
    minutes = np.asarray(state.minutes, dtype=float)
    frames: list[pd.DataFrame] = []
    for well in chosen:
        rows = onsets_of[well]
        m = rows.size
        if not m:
            continue
        c = well - 1
        on = np.asarray(state.on_ms[:, c], dtype=np.int64)
        lit = on > 0

        ## The last row of each run of lit rows that continue Index.
        next_lit = np.zeros(n, dtype=bool)
        next_lit[:-1] = lit[1:]
        run_ends = np.flatnonzero(lit & ~(next_lit & ~break_after))
        lit_end = np.arange(n)
        lit_rows = np.flatnonzero(lit)
        lit_end[lit_rows] = run_ends[np.searchsorted(run_ends, lit_rows, side="left")]

        following = np.append(rows[1:], n)
        end_row = np.maximum(np.minimum(lit_end[rows], following - 1), rows)

        cum_on = np.concatenate(([0], np.cumsum(on)))
        lit_ms = cum_on[end_row + 1] - cum_on[rows]
        onset_ms = index[rows] * P - on[rows]
        end_ms = np.where(end_row > rows, (index[end_row] - 1) * P + on[end_row],
                          onset_ms + lit_ms)

        ## Timing: more than one row, the interior rows whole records, and no
        ## row longer than one (a late record's OnMs runs over).
        cum_partial = _cumulative(on != P)
        cum_over = _cumulative(on > P)
        interior = np.where(end_row - rows >= 2,
                            cum_partial[end_row] - cum_partial[np.minimum(rows + 1, n)], 0)
        over = cum_over[end_row + 1] - cum_over[rows]
        exact = (end_row > rows) & (interior == 0) & (over == 0)

        ## Overlap: the period's length cannot be this Activation's own.
        theirs = [onsets_of[p] for p in partners.get(well, ()) if p in onsets_of]
        others = np.sort(np.concatenate(theirs)) if theirs else np.zeros(0, dtype=np.int64)
        partner_inside = (np.searchsorted(others, rows, side="left")
                          < np.searchsorted(others, end_row, side="right"))
        ## The LED already on: the row before was lit, or a break in Index
        ## hides whether it was (an Activation counted on the first row after
        ## lost records may have started in them).
        already_on = (rows > 0) & (~joined[rows] | (on[np.maximum(rows - 1, 0)] > 0))
        has_next = np.arange(m) < m - 1
        ran_on = has_next & (following == end_row + 1) & (on[end_row] >= P)
        shared_row = np.bincount(rows, minlength=n)[rows] > 1
        overlap = partner_inside | already_on | ran_on | shared_row

        ## Boundary, the most telling first.
        upto = np.minimum(end_row + 1, n - 1)
        crossed = (cum_change[upto + 1] - cum_change[rows + 1]) > 0
        boundary = np.full(m, BOUNDARY_NONE, dtype=object)
        boundary[(end_row == n - 1)] = BOUNDARY_END
        boundary[crossed] = BOUNDARY_INTERVAL
        boundary[(end_row < n - 1) & break_after[end_row]] = BOUNDARY_GAP

        ## SinceLastOnsetMs, Phase and K, within each Paradigm-state run.
        run = run_of_row[rows]
        same_run = np.zeros(m, dtype=bool)
        same_run[1:] = run[1:] == run[:-1]
        since = np.full(m, np.nan)
        since[1:] = (onset_ms[1:] - onset_ms[:-1]).astype(float)
        since[~same_run] = np.nan
        in_a = np.asarray(state.phase_a[rows, c], dtype=bool)
        ## The row where the well leaves Acclimation reads M, though an
        ## Acclimation Activation shorter than a record can start and end in
        ## it.  Under a ratio Paradigm only a Main Activation advances the
        ## Requirement, so there, as in requirement_met, the row's
        ## Activations are Acclimation ones unless the Requirement advanced,
        ## and then its last one is Main.  The row it is compared with is
        ## the well's last one under the ratio Paradigm in its state run: a
        ## well set On or Off for an Interval keeps its state, and those rows
        ## show no Requirement.
        req = state.requirement[:, c]
        known = ~np.isnan(req)
        before = np.full(n, -1, dtype=np.int64)
        if n > 1:
            before[1:] = np.maximum.accumulate(np.where(known, np.arange(n), -1))[:-1]
        before[before < run_starts[run_of_row]] = -1
        prior = np.maximum(before, 0)
        left = (before >= 0) & known & state.phase_a[prior, c] & ~state.phase_a[:, c]
        advanced = req != req[prior]
        last_in_row = following != rows
        in_a |= left[rows] & ~(advanced[rows] & last_in_row)
        in_a |= _unseen_at_first_row(state, c, rows, run_of_row)
        k = pd.Series((~in_a).astype(np.int64)).groupby(run).cumsum().to_numpy(dtype=float)
        k[in_a] = np.nan

        met = [state.requirement_met(well, r) for r in rows.tolist()]
        frames.append(pd.DataFrame({
            "Well": np.full(m, well, dtype=np.int64),
            "Row": rows, "EndRow": end_row, "Index": index[rows],
            "OnsetMin": minutes[rows],
            "OnsetMs": onset_ms, "EndMs": end_ms, "LitMs": lit_ms,
            "Timing": np.where(exact, TIMING_EXACT, TIMING_RECORD).astype(object),
            "Overlap": overlap, "Boundary": boundary,
            "SinceLastOnsetMs": since,
            "Phase": np.where(in_a, "A", "M").astype(object),
            "K": k,
            "Paradigm": state.paradigm[rows].to_numpy(dtype=object, na_value=None),
            "ProgramInterval": _nullable_at(state.program_interval, rows),
            "Cycle": _nullable_at(state.cycle, rows),
            "RequirementMet": np.array([np.nan if v is None else float(v) for v in met]),
        }))
    if not frames:
        return _empty()
    return _typed(pd.concat(frames, ignore_index=True)[list(LEDGER_COLUMNS)])


def activation_rows(book: pd.DataFrame, well: int, n: int, *,
                    onset_row: bool = True) -> np.ndarray:
    """(n,) bool: the rows of *well*'s own Activation periods in the ledger
    *book*, ``Row`` to ``EndRow`` inclusive, over a state of *n* rows.

    Without *onset_row*, each period's first row is left out, leaving the
    rows whose record began with the Activation already lit: the onset row
    holds the start of whatever started the Activation, a ratio Paradigm's
    Firmware Event included."""
    mine = book[book["Well"] == int(well)]
    steps = np.zeros(int(n) + 1, dtype=np.int64)
    if len(mine):
        rows = mine["Row"].to_numpy(dtype=np.int64)
        ends = mine["EndRow"].to_numpy(dtype=np.int64)
        if not onset_row:
            rows = rows + 1
            keep = rows <= ends
            rows, ends = rows[keep], ends[keep]
        np.add.at(steps, rows, 1)
        np.add.at(steps, ends + 1, -1)
    return np.cumsum(steps)[: int(n)] > 0
