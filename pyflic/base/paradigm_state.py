"""The Paradigm state of a recording: what DFM firmware 4.0.0 decided, row by row.

PyMCU 4.0.0 writes 63 columns after ``Index`` (``PyMCU/docs/DataFileColumns.md``):
``Interval``, ``Cycle``, ``Paradigm``, then per Well ``Phase``,
``Activations``, ``Requirement``, ``Progress`` and ``OnMs``.  Which of them a
file carries, and whether they hold anything, is its **Generation**
(``CONTEXT.md``), decided once at load by :func:`detect_generation`:

* generation 1 (:data:`GENERATION_FLAG`): 27 columns, PyMCU 3.x; Training is a
  flag inside the ``W`` signal;
* generation 2 (:data:`GENERATION_EMPTY`): 90 columns, every Paradigm column
  ``NA``: PyMCU 4.0.0 recording a DFM whose firmware is older than 4.0.0, so
  there is no Training information at all;
* generation 3 (:data:`GENERATION_PARADIGM`): 90 columns, filled.

PyMCU 4.0.0 releases add 12 more, ``Baseline1``-``Baseline12`` (102 columns):
the Baseline in force behind each record's thresholds, which PyMCU writes for
every firmware.  They don't change the Generation; :func:`split_baseline_columns`
takes them off the frame too, and the DFM keeps them as ``baseline_in_force``.

:func:`split_paradigm_columns` takes the 63 columns off the frame the rest of
pyflic works on, and :class:`ParadigmState` holds them for generation 3 as
numpy arrays, one row per record, aligned with the DFM's ``raw_df``.  Its
methods answer the questions the light QC, the Progressive Ratio type and the
light table ask of the firmware's own decisions, so nothing downstream reads
the columns directly.

Facts from the firmware source (``DFMV3_V2/src/LEDControl.c``) the methods
rely on, all reproduced by ``tests/v4_fixtures.py``:

* ``Activations`` is cumulative for the whole recording and rises in the
  record where the Activation's light starts.
* Under a ratio Paradigm, ``Progress`` returns to 0 in the same millisecond
  an Activation starts and ``Requirement`` moves to the next value then, so
  **the Requirement an Activation met is on the row before the one where
  Activations rises**.
* During Acclimation (Phase ``A``) ``Requirement`` already shows the first
  Main value and ``Progress`` stays 0: Acclimation counts Activations, not
  Events (each time contact lights the LED is one, so a fly back within the
  Event gap gets a new Activation without a new Event), and Acclimation
  Activations meet no Requirement.
* Wells set ``On`` or ``Off`` report Phase ``M`` and ``NA`` Requirement and
  Progress.
* The same Paradigm name in consecutive Intervals keeps every Well's state;
  a different name resets it.
* A DFM that restarts on its own (PyMCU 529b926) writes nothing while it is
  down, so ``Index`` jumps, then rows with ``Interval`` and ``Paradigm``
  ``NA`` until PyMCU's instruction applies again, afresh: every Well's state
  starts over there, while ``Activations`` carries on (:meth:`ParadigmState.restarts`).

Pure numpy and pandas; no Qt and no pyflic imports, so the loader, the light
QC and the tests can all use it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

GENERATION_FLAG, GENERATION_EMPTY, GENERATION_PARADIGM = 1, 2, 3

#: What each Generation is, in the words ``summary.txt`` uses.
GENERATION_DESCRIPTIONS: dict[int, str] = {
    GENERATION_FLAG: "27 columns, training flag in the signal",
    GENERATION_EMPTY: "90 or 102 columns, paradigm columns empty: DFM firmware older than 4.0.0",
    GENERATION_PARADIGM: "90 or 102 columns, firmware 4.0.0 paradigm state",
}

#: What pyflic says of a generation-2 DFM wherever its Training would show
#: (decision 15): ``summary.txt``, the report, the Hub's Loaded line, the
#: Project Report and the light QC.  One wording everywhere, starting "no
#: training information", which is what ``light_summary`` looks for.
NO_TRAINING_INFORMATION = ("no training information: recorded with DFM firmware "
                           "older than 4.0.0")

N_WELLS = 12
_PER_WELL = ("Phase", "Activations", "Requirement", "Progress", "OnMs")

#: PyMCU's ``StatusPacket.PARADIGM_COLUMNS``, in file order.
PARADIGM_COLUMNS: tuple[str, ...] = (
    "Interval", "Cycle", "Paradigm",
    *(f"{name}{w}" for name in _PER_WELL for w in range(1, N_WELLS + 1)))

#: PyMCU's ``StatusPacket.BASELINE_COLUMNS``, after the Paradigm columns: the
#: Baseline in force (``PyMCU/docs/DataFileColumns.md``), ``NA`` in Interval 0,
#: outside the Program and with ``Baseline: No``.
BASELINE_COLUMNS: tuple[str, ...] = tuple(f"Baseline{w}" for w in range(1, N_WELLS + 1))

#: PyMCU's own Paradigm: ClosedLoop with every Well Off, the Paradigm of a
#: program whose Intervals name none (the simple export) and of the DFM after
#: the Program ends.  The lead-in rows before the first instruction name no
#: Paradigm at all (``NA``).
IDLE = "Idle"

#: The milliseconds one record covers; ``OnMs`` above it marks a late record.
RECORD_MS = 200


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def detect_generation(df: pd.DataFrame) -> int:
    """The Generation of a DFM's concatenated data files.

    No ``Phase1`` column is generation 1; ``Phase1`` present with every value
    ``NA`` or blank is generation 2; any ``A`` or ``M`` is generation 3.
    ``Phase`` is the column tested because the firmware fills it on every
    4.0.0 row, where ``Interval``, ``Cycle`` and ``Paradigm`` can be ``NA``.

    Works on a frame concatenated from 12-hour split files whose dtypes
    differ: pandas reads an all-``NA`` file's ``Phase1`` as float and a filled
    one's as text, and the concatenation is an object column of both.
    """
    if "Phase1" not in df.columns:
        return GENERATION_FLAG
    values = df["Phase1"].dropna()
    if values.empty:
        return GENERATION_EMPTY
    text = values.astype(str).str.strip().str.upper()
    return GENERATION_PARADIGM if text.isin(("A", "M")).any() else GENERATION_EMPTY


def split_paradigm_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """``(the frame without the Paradigm columns, the Paradigm columns)``.

    The second item is ``None`` on generation 1, which has none.  The first
    keeps every other column (the 27 original ones, and ``Minutes`` /
    ``Seconds`` if they are already there), so nothing that copies ``raw_df``
    into its derived frames carries 63 columns it never reads.  Both keep the
    index of *df*, row for row.
    """
    if detect_generation(df) == GENERATION_FLAG:
        return df, None
    present = [c for c in PARADIGM_COLUMNS if c in df.columns]
    return df.drop(columns=present), df[present].copy()


def split_baseline_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray | None]:
    """``(the frame without the Baseline columns, the Baseline in force)``.

    The second item is ``(n, 12)`` float, Wells 1-12 as columns, NaN where the
    file has ``NA``; ``None`` when the file has no Baseline columns (PyMCU 3.x
    and 4.0.0 development builds).  The frame keeps its index, row for row.
    """
    present = [c for c in BASELINE_COLUMNS if c in df.columns]
    if not present:
        return df, None
    values = np.column_stack([_numbers(df, c, len(df)) for c in BASELINE_COLUMNS])
    return df.drop(columns=present), values


# ---------------------------------------------------------------------------
# Column normalisation
# ---------------------------------------------------------------------------

def _numbers(df: pd.DataFrame, column: str, n: int) -> np.ndarray:
    """A column as float with NaN for missing, whatever dtype pandas chose
    (int, float, ``Int64``, or text when a blank-written file mixed types)."""
    if column not in df.columns:
        return np.full(n, np.nan)
    ## A copy: under copy-on-write the frame's own buffer comes back read-only.
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float, na_value=np.nan, copy=True)


def _integer_array(values: np.ndarray) -> pd.arrays.IntegerArray:
    missing = np.isnan(values)
    return pd.arrays.IntegerArray(np.where(missing, 0, np.round(values)).astype(np.int64), missing)


def _per_well(df: pd.DataFrame, name: str, n: int) -> np.ndarray:
    return np.column_stack([_numbers(df, f"{name}{w}", n) for w in range(1, N_WELLS + 1)]) \
        if n else np.zeros((0, N_WELLS))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[start, end)`` of each run of True in *mask*."""
    if not mask.size:
        return []
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


# ---------------------------------------------------------------------------
# The state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IntervalRun:
    """A contiguous run of rows recorded under one (Interval, Cycle)."""

    program_interval: int | None      # None = unknown instruction; 0 = baseline
    cycle: int | None
    paradigm: str | None
    i0: int                           # sample index, inclusive
    i1: int                           # exclusive
    kind: str                         # "program" | "baseline" | "unknown"


@dataclass(frozen=True)
class RestartSpan:
    """A DFM restart the rows show: an ``Index`` gap followed by a run of
    unknown-instruction rows (:meth:`ParadigmState.restarts`)."""

    row: int                          # the first unknown-instruction row after the gap
    resume_row: int                   # the first row after that run (n if it runs to the end)
    missing: int                      # records the gap lost
    minute: float                     # minutes[row]


class ParadigmState:
    """One DFM's generation-3 Paradigm columns, one row per ``raw_df`` row.

    Wells are numbered 1-12 in every method; row indices are positions in
    the arrays (and so in ``raw_df``), from 0.  A window ``[i0, i1)`` is
    clipped to the rows that exist.  Treat an instance as read-only: copies
    of a DFM share it.
    """

    __slots__ = ("minutes", "index", "program_interval", "cycle", "paradigm", "phase_a",
                 "activations", "requirement", "progress", "on_ms", "activations_before",
                 "on_ms_before", "_cache")

    def __init__(self, *, minutes: np.ndarray, index: np.ndarray,
                 program_interval: pd.arrays.IntegerArray, cycle: pd.arrays.IntegerArray,
                 paradigm: pd.arrays.StringArray, phase_a: np.ndarray, activations: np.ndarray,
                 requirement: np.ndarray, progress: np.ndarray, on_ms: np.ndarray,
                 activations_before: np.ndarray | None = None,
                 on_ms_before: np.ndarray | None = None) -> None:
        self.minutes = minutes                      # (n,) float
        self.index = index                          # (n,) int64, the DFM's record Index
        self.program_interval = program_interval    # Int64, NA = unknown instruction
        self.cycle = cycle                          # Int64
        self.paradigm = paradigm                    # string, NA = unknown
        self.phase_a = phase_a                      # (n, 12) bool
        self.activations = activations              # (n, 12) int64
        self.requirement = requirement              # (n, 12) float, NaN where NA
        self.progress = progress                    # (n, 12) float, NaN where NA
        self.on_ms = on_ms                          # (n, 12) int64
        #: The ``Activations`` and ``OnMs`` of the row before row 0, (12,),
        #: when a range clipped the recording there (ADR-0020): an Activation
        #: counted on a Slice's first row is then its own, and a light on at
        #: that row is seen to have come on before it.  ``None`` for a state
        #: that starts with the recording, whose row 0 has no row before.
        self.activations_before = activations_before
        self.on_ms_before = on_ms_before
        #: Per-well arrays built on first use (the running total of Firmware
        #: Events, which rows credit a Requirement): a Light Event Ledger asks
        #: one question per Activation, and each answer is a full pass.
        self._cache: dict[tuple, np.ndarray] = {}

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, *, minutes, index=None) -> ParadigmState:
        """The state from the Paradigm columns :func:`split_paradigm_columns`
        returned, with the matching rows' ``Minutes`` and ``Index``.

        Missing values may be ``NA`` or blank (development builds before 16
        Oct 2026 wrote blanks).  A leading run of missing ``Interval`` is the
        baseline, 0, as those builds wrote it.  A missing ``Activations``
        carries the previous total (it is cumulative), a missing ``OnMs`` is
        0 and a missing ``Phase`` is not ``A``.
        """
        n = len(frame)
        minutes = np.asarray(minutes, dtype=float).reshape(-1)
        if minutes.size != n:
            raise ValueError(f"{minutes.size} minutes for {n} Paradigm rows")
        if index is None:
            index_values = np.arange(1, n + 1, dtype=np.int64)
        else:
            raw_index = pd.to_numeric(pd.Series(np.asarray(index).reshape(-1)), errors="coerce")
            index_values = raw_index.ffill().fillna(0).to_numpy(dtype=np.int64)

        interval = _numbers(frame, "Interval", n)
        known = np.flatnonzero(~np.isnan(interval))
        if known.size and known[0] > 0:
            interval[: known[0]] = 0.0
        cycle = _numbers(frame, "Cycle", n)
        if "Paradigm" in frame.columns:
            names = frame["Paradigm"].astype("string").str.strip()
            names = names.mask(names.eq("").fillna(False))
            paradigm = pd.array(names.to_numpy(dtype=object, na_value=pd.NA), dtype="string")
        else:
            paradigm = pd.array([pd.NA] * n, dtype="string")

        phase_a = np.zeros((n, N_WELLS), dtype=bool)
        for w in range(1, N_WELLS + 1):
            column = f"Phase{w}"
            if column in frame.columns:
                text = frame[column].astype("string").str.strip().str.upper()
                phase_a[:, w - 1] = text.eq("A").fillna(False).to_numpy(dtype=bool)
        activations = pd.DataFrame(_per_well(frame, "Activations", n)).ffill().fillna(0.0)
        on_ms = np.nan_to_num(_per_well(frame, "OnMs", n), nan=0.0)
        return cls(
            minutes=minutes, index=index_values,
            program_interval=_integer_array(interval), cycle=_integer_array(cycle),
            paradigm=paradigm, phase_a=phase_a,
            activations=activations.to_numpy(dtype=np.int64).reshape(n, N_WELLS),
            requirement=_per_well(frame, "Requirement", n),
            progress=_per_well(frame, "Progress", n),
            on_ms=on_ms.astype(np.int64))

    def subset(self, rows) -> ParadigmState:
        """The state of the rows a boolean mask or an index array selects, as
        the loader's range filter selects ``raw_df`` rows.  The row before
        the first one selected is kept as :attr:`activations_before` /
        :attr:`on_ms_before`, so a range that starts mid-recording still
        knows what its first row changed."""
        rows = np.asarray(rows)
        picked = np.flatnonzero(rows) if rows.dtype == bool else rows.astype(np.int64)
        before_acts, before_on = self.activations_before, self.on_ms_before
        if picked.size and int(picked[0]) > 0:
            before_acts = self.activations[int(picked[0]) - 1].copy()
            before_on = self.on_ms[int(picked[0]) - 1].copy()
        return ParadigmState(
            minutes=self.minutes[rows], index=self.index[rows],
            program_interval=self.program_interval[rows], cycle=self.cycle[rows],
            paradigm=self.paradigm[rows], phase_a=self.phase_a[rows],
            activations=self.activations[rows], requirement=self.requirement[rows],
            progress=self.progress[rows], on_ms=self.on_ms[rows],
            activations_before=before_acts, on_ms_before=before_on)

    def __len__(self) -> int:
        return int(self.minutes.size)

    def __repr__(self) -> str:
        return f"ParadigmState({len(self)} rows)"

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _column(well: int) -> int:
        w = int(well)
        if not 1 <= w <= N_WELLS:
            raise ValueError(f"wells are 1-{N_WELLS}, got {well!r}")
        return w - 1

    def _window(self, i0: int, i1: int) -> tuple[int, int]:
        n = len(self)
        a = min(max(int(i0), 0), n)
        b = min(max(int(i1), a), n)
        return a, b

    def _interval_codes(self) -> np.ndarray:
        """``program_interval`` as int64 with -1 for NA (Intervals are >= 0)."""
        return self.program_interval.to_numpy(dtype=np.int64, na_value=-1)

    def _activation_steps(self, w: int) -> np.ndarray:
        """Activations counted in each row: on row 0 against
        :attr:`activations_before` when a range clipped the recording there,
        else 0, its previous total not being in the data."""
        a = self.activations[:, w]
        if not a.size:
            return np.zeros(0, dtype=np.int64)
        first = a[:1] if self.activations_before is None else self.activations_before[w:w + 1]
        return np.clip(np.diff(a, prepend=first), 0, None)

    def _paradigm_starts(self) -> np.ndarray:
        """(n,) bool: rows where the Paradigm name differs from the row
        before (``NA`` a name of its own), so the firmware reset every Well;
        row 0 starts one by definition."""
        key = ("starts",)
        if key not in self._cache:
            n = len(self)
            codes, _ = pd.factorize(self.paradigm, use_na_sentinel=True)
            starts = np.ones(n, dtype=bool)
            if n > 1:
                starts[1:] = codes[1:] != codes[:-1]
            self._cache[key] = starts
        return self._cache[key]

    # -- Program Intervals -----------------------------------------------------

    def interval_runs(self) -> list[IntervalRun]:
        """Contiguous runs of rows by (``program_interval``, ``cycle``), in
        order.  A run's Paradigm is the first name recorded in it."""
        n = len(self)
        if not n:
            return []
        interval = self._interval_codes()
        cycle = self.cycle.to_numpy(dtype=np.int64, na_value=-1)
        change = np.flatnonzero((interval[1:] != interval[:-1]) | (cycle[1:] != cycle[:-1])) + 1
        starts = np.concatenate(([0], change)).tolist()
        ends = np.concatenate((change, [n])).tolist()
        names = self.paradigm.to_numpy(dtype=object, na_value=None)
        runs: list[IntervalRun] = []
        for a, b in zip(starts, ends):
            k = int(interval[a])
            kind = "unknown" if k < 0 else "baseline" if k == 0 else "program"
            name = next((str(v) for v in names[a:b] if v is not None), None)
            runs.append(IntervalRun(
                program_interval=None if k < 0 else k,
                cycle=None if cycle[a] < 0 else int(cycle[a]),
                paradigm=name, i0=int(a), i1=int(b), kind=kind))
        return runs

    def lead_in_rows(self) -> int:
        """The baseline rows at the start (``Interval`` 0, Paradigm ``NA``,
        or ``Idle`` in development builds), before the first instruction of
        the Program reached the DFM."""
        interval = self._interval_codes()
        other = np.flatnonzero(interval != 0)
        return int(other[0]) if other.size else len(self)

    def unknown_rows(self) -> np.ndarray:
        """(n,) bool: rows after the lead-in recorded under an instruction
        outside the Program (``Interval`` ``NA``)."""
        out = self._interval_codes() < 0
        out[: self.lead_in_rows()] = False
        return out

    def late_rows(self) -> np.ndarray:
        """(n,) bool: rows where any Well's ``OnMs`` exceeds the record's 200
        ms, so the record came late (``OnMs`` counts 1 ms steps since the
        previous record and is not clamped at 200)."""
        return (self.on_ms > RECORD_MS).any(axis=1)

    def index_gaps(self) -> list[tuple[int, int, float]]:
        """``(last index before, records missing, minute)`` for each gap in
        ``Index``; the minute is that of the last row before the gap.  A step
        back or a repeated Index is not a gap (the integrity report's
        business)."""
        if len(self) < 2:
            return []
        step = np.diff(self.index)
        at = np.flatnonzero(step > 1)
        return [(int(self.index[i]), int(step[i] - 1), float(self.minutes[i])) for i in at]

    def restarts(self) -> list[RestartSpan]:
        """Each DFM restart the rows show, in order: an ``Index`` gap whose
        next row begins a run of unknown-instruction rows (``Interval`` ``NA``
        after the lead-in).  This is the data's own rule: PyMCU 529b926 writes
        no records while the DFM is down and the idle instruction's rows until
        it resends the Program's.  A gap without such rows is lost records,
        and such rows without a gap are an instruction the DFM did not take
        up; neither is a restart.  ``Messages.csv`` names restarts too (the
        integrity report compares the two)."""
        if len(self) < 2:
            return []
        unknown = self.unknown_rows()
        runs = {a: b for a, b in _runs(unknown)}
        step = np.diff(self.index)
        out: list[RestartSpan] = []
        for i in np.flatnonzero(step > 1):
            row = int(i) + 1
            if row in runs:
                out.append(RestartSpan(row=row, resume_row=int(runs[row]), missing=int(step[i] - 1),
                                       minute=float(self.minutes[row])))
        return out

    def state_runs(self) -> list[tuple[int, int]]:
        """``[i0, i1)`` of each Paradigm-state run, in order: the rows between
        consecutive Paradigm starts, where the firmware reset every Well (a
        change of Paradigm name, ``NA`` a name of its own).  A restart's
        ``NA`` rows are a run of their own, so the run after them starts
        afresh, as the firmware did; the lead-in is the first run."""
        starts = np.flatnonzero(self._paradigm_starts()).tolist()
        return [(int(a), int(b)) for a, b in zip(starts, [*starts[1:], len(self)])]

    # -- Activations and light -----------------------------------------------

    def activation_onsets(self, well: int) -> np.ndarray:
        """Row indices where *well*'s Activations rose, one entry per
        Activation (an index repeats when the count rose by 2 in one row).
        Row 0 is an onset only when a range clipped the recording there
        (:attr:`activations_before`); otherwise the total before it is not in
        the data."""
        steps = self._activation_steps(self._column(well))
        return np.repeat(np.arange(steps.size), steps)

    def activations_between(self, well: int, i0: int, i1: int) -> int:
        """Activations of *well* counted in rows ``[i0, i1)``."""
        a, b = self._window(i0, i1)
        return int(self._activation_steps(self._column(well))[a:b].sum())

    def lit(self) -> np.ndarray:
        """(n, 12) bool: the Well's LED was active at some point in the
        record, by any cause (its own Paradigm, ``On``, Linkage)."""
        return self.on_ms > 0

    def lit_seconds(self, well: int, i0: int, i1: int) -> float:
        """Seconds *well*'s LED was active in rows ``[i0, i1)``: the sum of
        its ``OnMs`` / 1000, exact to the millisecond."""
        a, b = self._window(i0, i1)
        return float(self.on_ms[a:b, self._column(well)].sum()) / 1000.0

    # -- Ratio Paradigms, Acclimation and Training ---------------------------

    def ratio_rows(self, well: int) -> np.ndarray:
        """(n,) bool: rows where *well* runs a ratio Paradigm (its
        ``Requirement`` is not ``NA``), in Acclimation or Main."""
        return ~np.isnan(self.requirement[:, self._column(well)])

    def acclimation_runs(self, well: int) -> list[tuple[int, int]]:
        """``[i0, i1)`` of each run of rows *well* spent in Phase ``A``."""
        return _runs(self.phase_a[:, self._column(well)])

    def acclimation_end(self, well: int) -> int | None:
        """The first row in Phase ``M`` after *well*'s first run in ``A``, or
        ``None`` if it was never in ``A`` or never left it."""
        runs = self.acclimation_runs(well)
        if not runs:
            return None
        end = runs[0][1]
        return end if end < len(self) else None

    def training_end(self, well: int) -> int | None:
        """The first row where *well* is under a ratio Paradigm and not in
        Phase ``A``: where Training ends on generation 3 (decision 11 of
        ``docs/v4-data-plan.md``).  A ClosedLoop Interval before the ratio
        Paradigm is part of Training; a ratio Paradigm without Acclimation
        ends it on its first row.  ``None`` if the well never reaches Main
        under a ratio Paradigm.  Computable without the program."""
        w = self._column(well)
        main = ~np.isnan(self.requirement[:, w]) & ~self.phase_a[:, w]
        rows = np.flatnonzero(main)
        return int(rows[0]) if rows.size else None

    def requirement_met(self, well: int, onset: int) -> int | None:
        """The Requirement the Activation counted on row *onset* met: the
        ``Requirement`` on the row before, since the firmware moves to the
        next one in the millisecond the Activation starts.

        ``None`` when the Activation met no Requirement the rows show:

        * *onset* is 0 or outside the rows;
        * that Requirement is ``NA`` (no ratio Paradigm on the row before);
        * row *onset* starts a new Paradigm name: the Activation ran under
          the new Paradigm, whose first Requirement no row shows, or under
          the old one before the change, and the rows cannot say which;
        * the row before is in Phase ``A``: an Acclimation Activation meets
          no Requirement, though the row shows the first Main one.  The
          exception is the row where the well leaves ``A`` with its
          Requirement advanced, which only a Main Activation does: the
          Acclimation ended and a Main Activation started in one record.
          Under FixedRatio, whose Requirement never advances, that case reads
          as Acclimation.

        Two Activations counted in one row both get this value; the second
        really met the next Requirement, which no row shows (exact under
        FixedRatio and once a series reaches its Cap).
        """
        w = self._column(well)
        r = int(onset)
        if r <= 0 or r >= len(self):
            return None
        if not self._meets_requirement(w)[r]:
            return None
        return int(self.requirement[r - 1, w])

    def _meets_requirement(self, w: int) -> np.ndarray:
        """(n,) bool: an Activation counted on this row was a Main-phase
        ratio Activation whose Requirement is on the row before (the rules of
        :meth:`requirement_met`)."""
        key = ("meets", w)
        if key in self._cache:
            return self._cache[key]
        n = len(self)
        out = self._cache[key] = np.zeros(n, dtype=bool)
        if n < 2:
            return out
        req = self.requirement[:, w]
        phase_a = self.phase_a[:, w]
        known = ~np.isnan(req[:-1])
        advanced = known & ~np.isnan(req[1:]) & (req[1:] != req[:-1])
        left_acclimation = phase_a[:-1] & ~phase_a[1:] & advanced
        out[1:] = known & (~phase_a[:-1] | left_acclimation)
        out[self._paradigm_starts()] = False
        return out

    def firmware_event_rises(self, well: int) -> np.ndarray:
        """Row indices where *well*'s ``Progress`` rose (``NA`` read as 0):
        rows in which at least one Firmware Event started under a ratio
        Paradigm without completing a Requirement.  A row may hold more than
        one Event; :meth:`firmware_events_between` counts them."""
        p = np.nan_to_num(self.progress[:, self._column(well)], nan=0.0)
        if not p.size:
            return np.zeros(0, dtype=np.int64)
        return np.flatnonzero(np.diff(p, prepend=p[:1]) > 0)

    def _main_activation_steps(self, w: int) -> np.ndarray:
        """Main-phase ratio Activations counted in each row: every
        Activation of a row :meth:`requirement_met` credits, except in the
        row where the well left Acclimation, which holds at most one."""
        steps = self._activation_steps(w)
        meets = self._meets_requirement(w)
        main = np.where(meets, steps, 0)
        if main.size > 1:
            leaving = np.zeros(main.size, dtype=bool)
            leaving[1:] = self.phase_a[:-1, w]
            main[leaving] = np.minimum(main[leaving], 1)
        return main

    def firmware_events_between(self, well: int, i0: int, i1: int) -> int:
        """Firmware Events *well* produced under a ratio Paradigm in Phase
        ``M`` in rows ``[i0, i1)``: the Requirements met by the Activations
        counted there, plus the change in ``Progress`` across the window.

        Exact on the simulator's truth, with these edges:

        * Only Events that ``Progress`` counts are counted.  Acclimation is
          not (``Progress`` stays 0 in Phase ``A``, and Acclimation counts
          Activations, which :meth:`acclimation_runs` and
          :meth:`activations_between` count), nor Events under any other
          Paradigm type (``Progress`` is ``NA``).
        * A change of Paradigm name resets ``Progress`` without an Event;
          the window's accounting restarts there.  An Event in the same
          record as the change, before it, is lost with the reset.  A drop
          in ``Progress`` without an Activation and without a name change
          is read as a reset too.
        * ``Progress`` is carried across rows where it is ``NA`` within one
          Paradigm (an Interval that sets the well ``On`` or ``Off`` and
          keeps the state).
        * An Activation :meth:`requirement_met` cannot credit (the first row
          of a new Paradigm; under FixedRatio, a Main Activation in the record
          where Acclimation ended) adds no Requirement, so its Events are
          missed.  Two Activations in one row are both credited the
          Requirement on the row before.
        * On row 0 the Progress before is unknown and taken as 0.
        """
        w = self._column(well)
        a, b = self._window(i0, i1)
        if a >= b:
            return 0
        key = ("events", w)
        total = self._cache.get(key)
        if total is None:
            total = self._cache[key] = np.concatenate(([0.0], np.cumsum(self._event_rows(w))))
        return int(round(float(total[b] - total[a])))

    def _event_rows(self, w: int) -> np.ndarray:
        """Firmware Events counted in each row (see
        :meth:`firmware_events_between`): an Activation's Requirement, less
        the Progress already made before the row, plus the Progress at its
        end."""
        starts = self._paradigm_starts()
        segment = np.cumsum(starts)
        progress = pd.Series(self.progress[:, w]).groupby(segment).ffill().fillna(0.0).to_numpy()
        before = np.zeros_like(progress)
        before[1:] = progress[:-1]
        before[starts] = 0.0
        met = np.zeros_like(progress)
        met[1:] = np.nan_to_num(self.requirement[:-1, w], nan=0.0)
        main = self._main_activation_steps(w)
        events = main * met + progress - before
        drop = (main == 0) & (events < 0)
        events[drop] = progress[drop]
        return events

    # -- Light without a program ---------------------------------------------

    def wells_with_activations(self) -> frozenset[int]:
        """Wells whose Activations count is above 0: the Trigger Wells the
        data shows when no program states them (decision 10)."""
        if not len(self):
            return frozenset()
        return frozenset(int(w) + 1 for w in np.flatnonzero(self.activations.max(axis=0) > 0))

    def identical_on_ms_groups(self, wells) -> list[tuple[int, ...]]:
        """*wells* grouped by identical ``OnMs`` columns over the whole state:
        the Linkage the data shows when no program states it (``OnMs``
        counts light after Linkage, so linked wells match millisecond for
        millisecond).  Every well is in exactly one group, a singleton when
        nothing matches it; groups are in order of their first well.  Pass
        only lit wells: unlit ones all match each other."""
        groups: dict[bytes, list[int]] = {}
        for well in sorted({int(w) for w in wells}):
            column = np.ascontiguousarray(self.on_ms[:, self._column(well)])
            groups.setdefault(column.tobytes(), []).append(well)
        return sorted((tuple(g) for g in groups.values()), key=lambda g: g[0])
