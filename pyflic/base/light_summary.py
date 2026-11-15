"""The light table: what each well's light did, in the feeding summary's shape.

Every optogenetic experiment, of any Experiment Type, gets one (decisions 6
and 16 of ``docs/v4-data-plan.md``): ``analysis/light_wells.csv``, one row
per well, and ``analysis/light_summary.csv``, one row per chamber, for every
DFM the optogenetic light QC covers, over the chambers the design assigns a
Treatment (the feeding summary's rows, after auto-removal).  They carry the
feeding summary's Treatment and design-factor columns, so the Combined
Analysis, the reports and the Plot Editor pool and test light behaviour as
they do feeding.  The columns tested between treatments are the headline
metrics of the Paradigm Families the member's program runs
(:func:`experiment_headline_metrics`, decision 9 of
``docs/paradigm-analyses-plan.md``); :data:`HEADLINE_METRICS`, the fixed four
of before, stays for a member without a generation-3 recording.

What a row can say depends on the recording's **Generation** (``CONTEXT.md``):

* **Generation 3** reads the firmware's own decisions from
  :attr:`DFM.paradigm <pyflic.base.dfm.DFM.paradigm>`: lit time exact from
  ``OnMs``, Activations, Acclimation (Phase ``A``), Firmware Events and the
  Requirements met under a ratio Paradigm, and the Paradigm, Program
  Interval and Cycle each row was recorded under.
* **Generations 1 and 2** have only ``OptoCol1``, the LEDs sampled once a
  record: ``LitSec`` counts the lit samples (``LitSource`` says which), and
  every firmware column is ``NA``.  ``LightOnsets``, the rising edges of a
  well's own light, is there on every generation: on generation 1 it is the
  nearest thing to an Activation count.  With a program, the Paradigm, its
  type and the Program Interval come from the program's clock schedule
  (approximate, as the generation-1 light QC's windows are).

The Paradigm **type** comes from ``Program.txt``: without the DFM's section
``ParadigmType`` and every type-specific column are ``NA`` (decision 4), and
a type-specific column is ``NA`` wherever the well ran no Paradigm of a type
that defines it:

* Acclimation columns: FixedInterval and the ratio types, with Acclimation;
* ``FirmwareEvents``, ``LastRequirementMet``, ``ProgressAtEnd``,
  ``EventsPerActivation``: the ratio types (ProgressiveRatio, FixedRatio);
* ``UnlitSec``: NonFeeding.

The Paradigm Families' own columns (stream S2 of
``docs/paradigm-analyses-plan.md``, decisions 8, 10 and 11) are read from
the Activation ledger (:mod:`pyflic.base.activation_ledger`), on generation 3
with a program only, each ``NA`` outside its family.  An Activation belongs
to the window holding its onset row; one is *judgeable* when it is not
``Overlap`` and has no ``Boundary``, and ``tol`` is
``opto_activation_tolerance_ms`` (:func:`activation_tolerance_ms`), one
record wider for an Activation timed to the record:

* closed loop (ClosedLoop, and any Phase-``A`` Activation):
  ``MedianActivationSec``, the median ``LitMs`` of its judgeable
  Activations; ``CutOffs`` (ClosedLoop with ``MaxOnMs``), the judgeable ones
  within ``tol`` of ``MaxOnMs``;
* ``MainActivations`` (any Paradigm with Acclimation): the Phase-``M``
  Activations;
* FixedInterval: ``MaxActivations``, its Main time divided by
  max(``LockoutMs``, ``ActivationMs``), floored; ``RewardEfficiency``, its
  Main Activations divided by that; ``MedianReturnSec``, the median of
  ``SinceLastOnsetMs`` less that spacing between consecutive Main
  Activations; ``LockedOutContactSec`` (redefined, decision 10), pyflic's
  lick time in its Main rows while its own lockout runs (``LockoutMs`` from
  an onset) and its own Activation is not lit (``ActivationMs`` from it),
  so a linked partner's light no longer hides it;
* NonFeeding: ``Suppressions`` (redefined, decision 10), the judgeable
  Activations not ended by ``MaxOnMs``, since contact is the only other way
  the light goes out; ``MedianSuppressionSec``, the median dark time from a
  suppression to the next Activation of its state run;
* open loop (decision 11; the chamber's Role is Open, or in the window its
  wells run open loop and nothing else): ``LicksPerMinLit`` and
  ``LicksPerMinDark``, the licks while the chamber's own light is on and
  off per minute of each, over the rows a Program Interval ran
  (:func:`analytics.chamber_light_phases
  <pyflic.base.analytics.chamber_light_phases>`, which
  ``light_phase_summary`` reports over the whole recording, baseline
  included), and ``LightFeedingIndex``, (lit - dark) / (lit + dark) of the
  two.  A phase shorter than :data:`MIN_LIGHT_PHASE_MIN` has no rate, and
  the index is ``NA`` when either rate is, or both are 0.

A window is the feeding summary's: one ``(a, b]`` range of minutes
(``(0, 0)`` the whole recording), or a list of them, as a Program Facet
pooling a Repeating program's Cycles has.  ``Paradigm``, ``ParadigmType``,
``ProgramInterval`` and ``Cycle`` hold the one value of the window, or
``mixed``; the baseline (Program Interval 0) and the rows of an unknown
instruction are left out of them (decision 30).

``Role`` is the chamber's (:mod:`pyflic.base.opto_roles`), except where the
Experiment Type assigns its own (Progressive Ratio's paired / yoked), which
it keeps everywhere.  :func:`with_light_columns` puts ``Role``, ``LitSec``
and ``Activations`` on the feeding summary's rows.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, NamedTuple, Sequence

import numpy as np
import pandas as pd

from . import activation_ledger as al
from . import opto_program as op
from . import paradigm_families as pfam
from . import windowing
from .opto_roles import ROLE_OPEN
from .paradigm_state import GENERATION_PARADIGM, NO_TRAINING_INFORMATION  # noqa: F401

#: The Paradigm Families' columns (stream S2), grouped by family as
#: :data:`paradigm_families.FAMILIES` declares them.
CLOSED_LOOP_COLUMNS: tuple[str, ...] = ("MedianActivationSec", "CutOffs")
FIXED_INTERVAL_COLUMNS: tuple[str, ...] = ("MaxActivations", "RewardEfficiency",
                                           "MedianReturnSec", "LockedOutContactSec")
NON_FEEDING_COLUMNS: tuple[str, ...] = ("UnlitSec", "Suppressions", "MedianSuppressionSec")
OPEN_LOOP_COLUMNS: tuple[str, ...] = ("LicksPerMinLit", "LicksPerMinDark", "LightFeedingIndex")

#: The per-well table, ``analysis/light_wells.csv``.  The design's factor
#: columns follow ``Treatment``, as in the feeding summary.
LIGHT_WELL_COLUMNS: tuple[str, ...] = (
    "DFM", "Well", "Chamber", "Treatment", "Role", "Generation",
    "Paradigm", "ParadigmType", "ProgramInterval", "Cycle", "WindowMinutes",
    "LitSec", "LitFraction", "LitSource", "Activations", "ActivationsPerHour", "LightOnsets",
    "AcclimationActivations", "AcclimationEndMin", "AcclimationComplete", "MainActivations",
    "FirmwareEvents", "LastRequirementMet", "ProgressAtEnd", "EventsPerActivation",
    *CLOSED_LOOP_COLUMNS, *FIXED_INTERVAL_COLUMNS, *NON_FEEDING_COLUMNS, *OPEN_LOOP_COLUMNS,
    "PyflicEvents", "PyflicLicks")

#: The per-chamber table, ``analysis/light_summary.csv``: counts and times
#: summed over the chamber's wells, except ``LitSec`` and ``LightOnsets``,
#: which describe the chamber's own light (lit while any of its wells is),
#: ``AcclimationEndMin`` (the latest) and ``AcclimationComplete`` (all).
#: The medians are over the Activations of all its wells, the open-loop
#: rates are the chamber's licks per minute of its own light and dark, and
#: ``RewardEfficiency`` is its wells' FixedInterval Main Activations over
#: their ``MaxActivations``.
LIGHT_CHAMBER_COLUMNS: tuple[str, ...] = (
    "DFM", "Chamber", "Treatment", "Role", "Generation",
    "Paradigm", "ParadigmType", "ProgramInterval", "Cycle", "WindowMinutes",
    "LitSec", "LitFraction", "LitSource", "Activations", "ActivationsPerHour", "LightOnsets",
    "AcclimationActivations", "AcclimationEndMin", "AcclimationComplete", "MainActivations",
    "FirmwareEvents", "LastRequirementMet", "ProgressAtEnd", "EventsPerActivation",
    *CLOSED_LOOP_COLUMNS, *FIXED_INTERVAL_COLUMNS, *NON_FEEDING_COLUMNS, *OPEN_LOOP_COLUMNS,
    "EventsA", "EventsB")

#: The light metrics tested between treatments before the Paradigm Families
#: (decision 16 of ``docs/v4-data-plan.md``).  Kept for compatibility, and
#: for a member or Project without a generation-3 recording: the headline
#: metrics now follow the families run (:func:`experiment_headline_metrics`).
HEADLINE_METRICS: tuple[str, ...] = ("Activations", "LitSec", "FirmwareEvents",
                                     "AcclimationEndMin")

#: ``opto_activation_tolerance_ms``'s default (decision 7).
DEFAULT_ACTIVATION_TOLERANCE_MS = 25

#: What :func:`with_light_columns` adds to an optogenetic feeding summary.
FEEDING_COLUMNS: tuple[str, ...] = ("Role", "LitSec", "Activations")

#: ``LitSource`` values.
SOURCE_ON_MS, SOURCE_OPTOCOL1 = "OnMs", "OptoCol1"
MIXED = "mixed"

#: Progressive Ratio's own Role words (``experiment_types.progressive_ratio``).
_PR_ROLES = ("paired", "yoked")

_COUNT_COLUMNS = ("Activations", "AcclimationActivations", "FirmwareEvents", "ProgressAtEnd",
                  "LockedOutContactSec", "UnlitSec", "Suppressions", "MainActivations",
                  "CutOffs", "MaxActivations")
#: The per-well values the chamber's medians pool (hidden keys of a well's
#: values), by the column they give.
_POOLED = {"MedianActivationSec": "_ActivationSec", "MedianReturnSec": "_ReturnSec",
           "MedianSuppressionSec": "_SuppressionSec"}


def activation_tolerance_ms(exp) -> int:
    """``opto_activation_tolerance_ms`` (decision 7), the tolerance the
    family columns judge an Activation's length with: the light QC's own
    setting, read from *exp*'s design ``constants:`` as the light QC reads
    it, so the light table and the rule checks always judge alike.  A value
    that is missing or not a number gives the default,
    :data:`DEFAULT_ACTIVATION_TOLERANCE_MS`."""
    from .opto_qc import OptoQCSettings

    constants = getattr(exp, "global_constants", None) or {}
    return int(OptoQCSettings.from_constants(constants).activation_tolerance_ms)


# ---------------------------------------------------------------------------
# Windows: one (a, b] range of minutes, or several
# ---------------------------------------------------------------------------

def _format_ranges(window: Any) -> str:
    """The ``FacetRange`` cell of one window or several, as the feeding
    summary's Facets write it."""
    if windowing.is_window_list(window):
        return windowing.format_ranges(window)
    return windowing.format_range(window)


class _Window(NamedTuple):
    """The rows of a window: their indices, the contiguous segments they
    form (for the Paradigm state's ``[i0, i1)`` questions), and the row after
    the last."""

    rows: np.ndarray
    segments: list[tuple[int, int]]
    end: int

    @property
    def size(self) -> int:
        return int(self.rows.size)

    def contains(self, at: np.ndarray) -> np.ndarray:
        """Which of the row indices *at* fall inside the window."""
        inside = np.zeros(np.asarray(at).size, dtype=bool)
        for a, b in self.segments:
            inside |= (at >= a) & (at < b)
        return inside


def _window(minutes: np.ndarray, range_minutes: Any) -> _Window:
    ## The rows the feeding summary selects: (a, b] per window, None for all.
    mask = windowing.range_mask(minutes, range_minutes)
    rows = np.arange(minutes.size) if mask is None else np.flatnonzero(mask)
    if not rows.size:
        return _Window(rows, [], 0)
    breaks = np.flatnonzero(np.diff(rows) > 1)
    starts = np.concatenate(([rows[0]], rows[breaks + 1]))
    ends = np.concatenate((rows[breaks] + 1, [rows[-1] + 1]))
    return _Window(rows, [(int(a), int(b)) for a, b in zip(starts, ends)], int(rows[-1]) + 1)


# ---------------------------------------------------------------------------
# One DFM, prepared once
# ---------------------------------------------------------------------------

def type_owns_role(exp) -> bool:
    """Whether *exp*'s Experiment Type assigns each chamber its own Role
    (Progressive Ratio's paired / yoked), which the light layer then keeps
    rather than deriving one."""
    return callable(getattr(exp, "role_of", None))


class _DFMLight:
    """What every window of one DFM reads: its light per row and well (and
    the row before's, for onsets), its chambers and their Roles, and on
    generation 3 the Activations counted per row and, with a program, the
    Paradigm type each row ran and which wells ran it.  Built once per DFM
    object and program, and kept on the experiment."""

    def __init__(self, exp, dfm_id: int) -> None:
        dfm = exp.dfms[int(dfm_id)]
        self.dfm = dfm
        self.program = getattr(exp, "opto_program", None)
        self.dfm_id = int(dfm_id)
        self.generation = int(dfm.generation)
        self.sps = float(dfm.params.samples_per_second)
        self.minutes = pd.to_numeric(dfm.raw_df["Minutes"], errors="coerce").to_numpy(float)
        n = self.minutes.size
        state = dfm.paradigm if self.generation == GENERATION_PARADIGM else None
        self.state = state
        self.source = SOURCE_ON_MS if state is not None else SOURCE_OPTOCOL1
        self.steps = None
        lit_first = np.zeros(op.N_WELLS, dtype=bool)
        if state is not None:
            self.on_ms = np.asarray(state.on_ms)
            self.lit = self.on_ms > 0
            ## Activations counted in each row; row 0's against the row before
            ## it when a range clipped the recording there, else its total
            ## predates the file.
            counts = np.asarray(state.activations)
            first = (counts[:1] if state.activations_before is None
                     else np.asarray(state.activations_before).reshape(1, -1))
            self.steps = np.clip(np.diff(counts, axis=0, prepend=first), 0, None) \
                if n else np.zeros((0, op.N_WELLS), dtype=np.int64)
            if state.on_ms_before is not None:
                lit_first = np.asarray(state.on_ms_before) > 0
        else:
            self.on_ms = None
            self.lit = np.zeros((n, op.N_WELLS), dtype=bool)
            frame = dfm.lights_df
            if frame is not None:
                for w in range(1, op.N_WELLS + 1):
                    if f"W{w}" in frame.columns:
                        self.lit[:, w - 1] = frame[f"W{w}"].to_numpy()[:n].astype(bool)
            before = getattr(dfm, "before_range", None)
            if before is not None:
                lit_first = np.asarray(before.lit, dtype=bool)
        ## Before row 0 is the row a range clipped off (a Slice's light already
        ## on at its first row came on before it), else dark: a light on at
        ## the recording's first record came on.
        self.lit_before = np.zeros_like(self.lit)
        if n:
            self.lit_before[0] = lit_first
        self.lit_before[1:] = self.lit[:-1]
        self.licks = _matrix(dfm.lick_df, n, bool)
        self.events = _matrix(dfm.event_df, n, np.int64) > 0

        self.chambers = {int(ch.index): ch for ch in (dfm.chambers or [])}
        self.chamber_of = {int(w): c for c, ch in self.chambers.items() for w in ch.wells}
        self.section = (self.program.section(self.dfm_id)
                        if self.program is not None else None)
        if type_owns_role(exp):
            self.roles = {c: str(exp.role_of(self.dfm_id, c).role) for c in self.chambers}
        else:
            from .opto_roles import chamber_roles, dfm_linkage_view

            self.roles = chamber_roles(dfm_linkage_view(dfm, self.section), self.chamber_of)

        self.typed = state is not None and self.section is not None
        if self.typed:
            self._read_types()

    # -- generation 3 with a program ------------------------------------------

    def _read_types(self) -> None:
        """Per row: the Paradigm in force (by the name the row records) and
        its type; per row and well: whether the well ran that Paradigm (a
        level threshold in the row's Program Interval), and whether it ran
        open loop (``On``, or a level under OpenLoop) or a rule its fly
        drives (a level under any other type)."""
        state, program = self.state, self.program
        n = len(state)
        names = state.paradigm.to_numpy(dtype=object, na_value=None)
        codes, uniques = pd.factorize(pd.Series(names, dtype=object), use_na_sentinel=True)
        self.names = names
        ## An NA name is code -1, which indexes the trailing None.
        self.codes = np.asarray(codes)
        self.paradigm_by_code = [program.paradigms.get(str(u)) for u in uniques] + [None]
        self.row_type = np.asarray([p.type if p is not None else None
                                    for p in self.paradigm_by_code], dtype=object)[self.codes]
        self._type_masks: dict[tuple, np.ndarray] = {}
        self.trigger = np.zeros((n, op.N_WELLS), dtype=bool)
        always_on = np.zeros((n, op.N_WELLS), dtype=bool)
        fallback = list(self.section.trigger_wells())
        for run in state.interval_runs():
            if run.kind != "program":
                continue
            interval = program.interval_for(self.dfm_id, int(run.program_interval))
            for w in (interval.trigger_wells() if interval is not None else fallback):
                self.trigger[run.i0:run.i1, int(w) - 1] = True
            for w in (interval.open_wells() if interval is not None else ()):
                always_on[run.i0:run.i1, int(w) - 1] = True
        open_type = (self.row_type == op.OPEN_LOOP)[:, None]
        known = np.asarray([t is not None for t in self.row_type], dtype=bool)[:, None]
        self.open_loop = always_on | (self.trigger & open_type)
        self.driven = self.trigger & known & ~open_type
        ## The Activation ledger and what is read from it, per well, on first use.
        self._book: pd.DataFrame | None = None
        self._activations: dict[int, _Activations] = {}
        self._timed: dict[tuple[int, str], np.ndarray] = {}

    def type_rows(self, w: int, types: Iterable[str], win: _Window, *,
                  acclimation: bool = False) -> np.ndarray:
        """Per row of *win*: whether well *w* ran a Paradigm of one of *types*
        (with Acclimation, when *acclimation*)."""
        key = (tuple(types), bool(acclimation))
        return self.paradigm_rows(w, win, key, lambda p: p.type in key[0] and (
            not acclimation or int(p.acclimation or 0) > 0))

    def paradigm_rows(self, w: int, win: _Window, key: Any,
                      keep: Callable[[op.Paradigm], bool]) -> np.ndarray:
        """Per row of *win*: whether well *w* ran a Paradigm *keep* accepts;
        *key* names the test, for the cache."""
        rows = self._type_masks.get(key)
        if rows is None:
            ## Decided once per Paradigm name, then spread over the rows.
            per_code = [p is not None and bool(keep(p)) for p in self.paradigm_by_code]
            rows = self._type_masks[key] = np.asarray(per_code, dtype=bool)[self.codes]
        return rows[win.rows] & self.trigger[win.rows, w - 1]

    def window(self, range_minutes: Any) -> _Window:
        return _window(self.minutes, range_minutes)

    # -- the Activation ledger (generation 3 with a program) -------------------

    @property
    def record_ms(self) -> float:
        return 1000.0 / self.sps

    @property
    def book(self) -> pd.DataFrame:
        """The DFM's Activation ledger, with the program's Linkage Groups,
        as the light QC reads it (``opto_qc._analyze_paradigm_state``)."""
        if self._book is None:
            groups = self.section.linkage_groups()
            self._book = al.ledger(self.state,
                                   linked={w: tuple(g) for g in groups.values() for w in g},
                                   record_ms=int(round(self.record_ms)))
        return self._book

    def activations(self, w: int) -> _Activations:
        """Well *w*'s ledger rows, as arrays."""
        found = self._activations.get(int(w))
        if found is None:
            found = self._activations[int(w)] = _Activations.of(self, int(w))
        return found

    def fixed_rows(self, w: int, setting: str) -> np.ndarray:
        """(n,) bool: the rows during which a timer of well *w*'s
        FixedInterval Main Activations runs, *setting* (``lockout_ms`` or
        ``activation_ms``) from each one's onset (``OnsetMs``), within its
        state run.  The firmware times both from the start of the
        Activation (``LEDControl.c``, ``StepFixedInterval``), so
        ``activation_ms`` gives the rows the well's own Activation lit,
        whatever light a partner adds.  A row is in when any of its record
        is."""
        key = (int(w), setting)
        found = self._timed.get(key)
        if found is not None:
            return found
        state, n = self.state, len(self.state)
        acts = self.activations(w)
        steps = np.zeros(n + 1, dtype=np.int64)
        length = np.nan_to_num(acts.setting(setting))
        chosen = acts.main & acts.is_type(op.FIXED_INTERVAL) & (length > 0)
        if chosen.any():
            ## Records start at (Index - 1) * record; kept rising for the search.
            starts = np.maximum.accumulate((np.asarray(state.index, dtype=float) - 1.0)
                                           * self.record_ms)
            run_starts = np.array([a for a, _b in state.state_runs()], dtype=np.int64)
            run_ends = np.append(run_starts[1:], n)
            r0 = acts.row[chosen]
            r1 = np.searchsorted(starts, acts.onset_ms[chosen] + length[chosen], side="left")
            r1 = np.minimum(r1, run_ends[np.searchsorted(run_starts, r0, side="right") - 1])
            r1 = np.maximum(r1, r0 + 1)
            np.add.at(steps, r0, 1)
            np.add.at(steps, r1, -1)
        found = self._timed[key] = np.cumsum(steps)[:n] > 0
        return found


class _Activations(NamedTuple):
    """One well's rows of the Activation ledger, as arrays in onset order,
    with what the family columns read from them."""

    row: np.ndarray            # the onset row
    onset_ms: np.ndarray       # float
    end_ms: np.ndarray         # float
    lit_ms: np.ndarray         # float
    record: np.ndarray         # Timing record: the times are good to one record
    judgeable: np.ndarray      # not Overlap, no Boundary: the length is the rule's
    main: np.ndarray           # the ledger's Phase M
    since_ms: np.ndarray       # SinceLastOnsetMs, NaN for a state run's first
    after_main: np.ndarray     # the previous Activation, in the same state run, is Main
    next_onset_ms: np.ndarray  # the next Activation's onset in the same state run, or NaN
    paradigm: list             # the Paradigm in force at the onset row, or None
    unseen: np.ndarray         # Phase A on a row that reads M: its Acclimation began and ended in the row

    @classmethod
    def of(cls, ctx: _DFMLight, w: int) -> _Activations:
        book = ctx.book
        mine = book[book["Well"] == int(w)]
        row = mine["Row"].to_numpy(dtype=np.int64)
        since = mine["SinceLastOnsetMs"].to_numpy(dtype=float, na_value=np.nan)
        paradigm = [ctx.paradigm_by_code[c] for c in ctx.codes[row]]
        main = (mine["Phase"] == "M").to_numpy(dtype=bool)
        unseen = ~main & ~ctx.state.phase_a[row, int(w) - 1]
        onset = mine["OnsetMs"].to_numpy(dtype=float)
        same_run_next = np.append(~np.isnan(since[1:]), False)
        after_main = np.insert(main[:-1], 0, False) & ~np.isnan(since) if row.size else main
        return cls(
            row=row, onset_ms=onset, end_ms=mine["EndMs"].to_numpy(dtype=float),
            lit_ms=mine["LitMs"].to_numpy(dtype=float),
            record=(mine["Timing"] == al.TIMING_RECORD).to_numpy(dtype=bool),
            judgeable=(~mine["Overlap"].to_numpy(dtype=bool)
                       & (mine["Boundary"] == al.BOUNDARY_NONE).to_numpy(dtype=bool)),
            main=main, since_ms=since, after_main=after_main,
            next_onset_ms=np.where(same_run_next, np.append(onset[1:], np.nan), np.nan),
            paradigm=paradigm, unseen=unseen)

    def is_type(self, *types: str) -> np.ndarray:
        """Per Activation: whether its Paradigm is of one of *types*."""
        return np.asarray([p is not None and p.type in types for p in self.paradigm], dtype=bool)

    def setting(self, name: str) -> np.ndarray:
        """Per Activation: its Paradigm's *name* setting, NaN where not set."""
        out = [getattr(p, name, None) if p is not None else None for p in self.paradigm]
        return np.asarray([np.nan if v is None else float(v) for v in out], dtype=float)

    def slack_ms(self, tol: float, record_ms: float) -> np.ndarray:
        """``tol`` per Activation, one record wider where it is timed to the
        record (stream S1's rule)."""
        return float(tol) + np.where(self.record, float(record_ms), 0.0)


def _matrix(frame: pd.DataFrame | None, n: int, dtype) -> np.ndarray:
    out = np.zeros((n, op.N_WELLS), dtype=dtype)
    if frame is None:
        return out
    for w in range(1, op.N_WELLS + 1):
        if f"W{w}" in frame.columns:
            values = pd.to_numeric(frame[f"W{w}"], errors="coerce").fillna(0).to_numpy()
            out[:, w - 1] = values[:n].astype(dtype)
    return out


def _context(exp, dfm_id: int) -> _DFMLight:
    """The DFM's :class:`_DFMLight`, rebuilt when its DFM object or the
    program changed (a QC Viewer recompute swaps in re-detected DFMs)."""
    cache = getattr(exp, "_light_contexts", None)
    if cache is None:
        return _DFMLight(exp, dfm_id)
    entry = cache.get(int(dfm_id))
    if entry is not None and entry.dfm is exp.dfms[int(dfm_id)] \
            and entry.program is getattr(exp, "opto_program", None):
        return entry
    entry = cache[int(dfm_id)] = _DFMLight(exp, dfm_id)
    return entry


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _single(values: Iterable[Any]) -> Any:
    """The one distinct value, :data:`MIXED` for several, ``None`` for none."""
    seen = list(dict.fromkeys(v for v in values if v is not None))
    if not seen:
        return None
    return seen[0] if len(seen) == 1 else MIXED


def _sum(values: Iterable[Any]) -> float:
    """Sum of the values that are there; NaN when none is."""
    present = [float(v) for v in values if v is not None and not pd.isna(v)]
    return float(sum(present)) if present else np.nan


def _per_hour(count: float, minutes: float) -> float:
    if pd.isna(count) or not minutes > 0:
        return np.nan
    return float(count) / (float(minutes) / 60.0)


def _fraction(seconds: float, minutes: float) -> float:
    if pd.isna(seconds) or not minutes > 0:
        return np.nan
    return float(seconds) / (float(minutes) * 60.0)


# ---------------------------------------------------------------------------
# What a window was recorded under
# ---------------------------------------------------------------------------

def _window_paradigm(ctx: _DFMLight, win: _Window) -> dict[str, Any]:
    """``Paradigm``, ``ParadigmType``, ``ProgramInterval`` and ``Cycle`` of
    the window: one value, ``mixed`` or ``None``."""
    out: dict[str, Any] = {"Paradigm": None, "ParadigmType": None,
                           "ProgramInterval": None, "Cycle": None}
    state = ctx.state
    if state is None:
        return _scheduled_paradigm(ctx, win, out)
    if not win.size:
        return out
    interval = state.program_interval.to_numpy(dtype=np.int64, na_value=-1)[win.rows]
    program = interval > 0
    if not program.any():
        return out
    rows = win.rows[program]
    names = state.paradigm.to_numpy(dtype=object, na_value=None)[rows]
    cycles = state.cycle.to_numpy(dtype=np.int64, na_value=-1)[rows]
    out["Paradigm"] = _single(str(v) for v in names if v is not None)
    out["ProgramInterval"] = _single(int(v) for v in interval[program])
    out["Cycle"] = _single(int(v) for v in cycles if v >= 0)
    if ctx.typed:
        out["ParadigmType"] = _single(op.TYPE_NAMES.get(t, t) for t in ctx.row_type[rows]
                                      if t is not None)
    return out


def _scheduled_paradigm(ctx: _DFMLight, win: _Window, out: dict) -> dict:
    """Generations 1 and 2: the Program Intervals the program's clock
    schedule puts in the window's minutes (the light QC's own windows on
    these generations), when there is a section and the file has dates."""
    program, section = ctx.program, ctx.section
    if section is None or not win.size or program.start is None:
        return out
    from datetime import timedelta

    from .opto_light import recording_origin

    origin = recording_origin(ctx.dfm.raw_df)
    end = program.resolved_end()
    if origin is None or end is None:
        return out
    schedule = section.schedule(program.start, end)
    hits = []
    for a, b in win.segments:
        t0 = origin + timedelta(minutes=float(ctx.minutes[a]))
        t1 = origin + timedelta(minutes=float(ctx.minutes[b - 1]))
        hits += [s for s in schedule if s.start <= t1 and s.end > t0]
    if not hits:
        return out
    out["Paradigm"] = _single(s.interval.paradigm.name for s in hits)
    out["ParadigmType"] = _single(s.interval.paradigm.type_name for s in hits)
    out["ProgramInterval"] = _single(int(s.interval.index) for s in hits)
    out["Cycle"] = _single(int(s.occurrence) for s in hits)
    return out


# ---------------------------------------------------------------------------
# One well over one window
# ---------------------------------------------------------------------------

#: Every family column's value off its family, and the hidden values the
#: chamber pools.
_NO_FAMILY: dict[str, Any] = {
    **{c: np.nan for c in (*CLOSED_LOOP_COLUMNS, *FIXED_INTERVAL_COLUMNS,
                           *NON_FEEDING_COLUMNS, *OPEN_LOOP_COLUMNS, "MainActivations")},
    **{hidden: np.zeros(0) for hidden in _POOLED.values()}, "_FixedMain": np.nan,
}


def _well_values(ctx: _DFMLight, w: int, win: _Window, *, tol: float = 0.0,
                 phases: dict | None = None) -> dict[str, Any]:
    """Every metric of :data:`LIGHT_WELL_COLUMNS` for well *w* over *win*,
    plus the hidden values the chamber's columns pool: ``_Main`` (its Main
    ratio Activations) for EventsPerActivation, ``_FixedMain`` for
    RewardEfficiency and the ``_POOLED`` lists for the medians.  *tol* is
    ``opto_activation_tolerance_ms``; *phases*, the chamber's
    :func:`analytics.chamber_light_phases <pyflic.base.analytics.chamber_light_phases>`
    over *win* when the chamber is open loop there (decision 11)."""
    col, rows = w - 1, win.rows
    minutes = win.size / ctx.sps / 60.0
    lit = ctx.lit[rows, col]
    if ctx.on_ms is not None:
        lit_sec = float(ctx.on_ms[rows, col].sum()) / 1000.0
    else:
        lit_sec = float(np.count_nonzero(lit)) / ctx.sps
    out: dict[str, Any] = {
        "WindowMinutes": minutes, "LitSec": lit_sec,
        "LitFraction": _fraction(lit_sec, minutes), "LitSource": ctx.source,
        "LightOnsets": float(np.count_nonzero(lit & ~ctx.lit_before[rows, col])),
        "PyflicEvents": float(np.count_nonzero(ctx.events[rows, col])),
        "PyflicLicks": float(np.count_nonzero(ctx.licks[rows, col])),
        "Activations": np.nan, "ActivationsPerHour": np.nan,
        "AcclimationActivations": np.nan, "AcclimationEndMin": np.nan,
        "AcclimationComplete": None, "FirmwareEvents": np.nan,
        "LastRequirementMet": np.nan, "ProgressAtEnd": np.nan,
        "EventsPerActivation": np.nan, "_Main": np.nan, **_NO_FAMILY,
    }
    state = ctx.state
    if state is None or not win.size:
        return out
    activations = float(ctx.steps[rows, col].sum())
    out["Activations"] = activations
    out["ActivationsPerHour"] = _per_hour(activations, minutes)
    if not ctx.typed:
        return out
    acts = ctx.activations(w)
    inside = win.contains(acts.row)
    out.update(_acclimation(ctx, w, win, acts, inside))
    out.update(_ratio(ctx, w, win))
    out.update(_closed_loop(ctx, w, win, acts, inside, tol))
    if ctx.type_rows(w, op.ACCLIMATION_PARADIGMS, win, acclimation=True).any():
        out["MainActivations"] = float(np.count_nonzero(inside & acts.main))
    out.update(_fixed_interval(ctx, w, win, acts, inside))
    nonfeeding = ctx.type_rows(w, (op.NON_FEEDING,), win)
    if nonfeeding.any():
        on_ms = ctx.on_ms[rows, col]
        out["UnlitSec"] = max(0.0, float(np.count_nonzero(nonfeeding)) / ctx.sps
                              - float(on_ms[nonfeeding].sum()) / 1000.0)
        out.update(_suppressions(ctx, acts, inside, tol))
    if phases is not None:
        out.update(_open_loop_rates(ctx, phases, (w,)))
    return out


def _median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    return float(np.median(values)) if values.size else np.nan


def _closed_loop(ctx: _DFMLight, w: int, win: _Window, acts: _Activations,
                 inside: np.ndarray, tol: float) -> dict[str, Any]:
    """MedianActivationSec and CutOffs: where the well ran closed loop in
    the window, ClosedLoop or a Phase-``A`` row; CutOffs only under a
    ClosedLoop with ``MaxOnMs``.  Its closed-loop Activations are its
    ClosedLoop ones and those the ledger puts in Phase ``A``."""
    col = w - 1
    acclimating = ctx.type_rows(w, op.ACCLIMATION_PARADIGMS, win, acclimation=True) \
        & ctx.state.phase_a[win.rows, col]
    if not (ctx.type_rows(w, (op.CLOSED_LOOP,), win) | acclimating).any():
        return {}
    closed = acts.is_type(op.CLOSED_LOOP)
    judged = inside & acts.judgeable
    seconds = acts.lit_ms[judged & (closed | ~acts.main)] / 1000.0
    out: dict[str, Any] = {"MedianActivationSec": _median(seconds), "_ActivationSec": seconds}
    if ctx.paradigm_rows(w, win, "max_on", lambda p: p.mode == op.CLOSED_LOOP_MAX).any():
        max_on = acts.setting("max_on_ms")
        limited = closed & acts.main & (max_on > 0)
        at_max = np.abs(acts.lit_ms - np.nan_to_num(max_on)) <= acts.slack_ms(tol, ctx.record_ms)
        out["CutOffs"] = float(np.count_nonzero(judged & limited & at_max))
    return out


def _spacing_ms(p: op.Paradigm | None) -> float:
    """A FixedInterval's shortest time between Activation starts:
    max(``LockoutMs``, ``ActivationMs``), since a new one needs both the
    last one over and the lockout passed."""
    if p is None:
        return np.nan
    return float(max(int(p.lockout_ms or 0), int(p.activation_ms or 0)))


def _fixed_interval(ctx: _DFMLight, w: int, win: _Window, acts: _Activations,
                    inside: np.ndarray) -> dict[str, Any]:
    """MaxActivations, RewardEfficiency, MedianReturnSec and
    LockedOutContactSec, where the well ran FixedInterval in the window."""
    col, rows = w - 1, win.rows
    fixed = ctx.type_rows(w, (op.FIXED_INTERVAL,), win)
    if not fixed.any():
        return {}
    main_rows = fixed & ~ctx.state.phase_a[rows, col]
    ## The most it could earn: each Paradigm's Main time over its spacing.
    most = 0.0
    codes, counts = np.unique(ctx.codes[rows][main_rows], return_counts=True)
    for code, count in zip(codes.tolist(), counts.tolist()):
        spacing = _spacing_ms(ctx.paradigm_by_code[code])
        if spacing > 0:
            most += math.floor(count * ctx.record_ms / spacing)
    earned_rows = inside & acts.main & acts.is_type(op.FIXED_INTERVAL)
    earned = float(np.count_nonzero(earned_rows))
    spacing = np.asarray([_spacing_ms(p) for p in acts.paradigm], dtype=float)
    after = earned_rows & acts.after_main
    returns = (acts.since_ms[after] - spacing[after]) / 1000.0
    ## Decision 10: its own lockout running and its own Activation dark, so
    ## a partner's light no longer hides the feeding.
    locked = (main_rows & ctx.licks[rows, col] & ctx.fixed_rows(w, "lockout_ms")[rows]
              & ~ctx.fixed_rows(w, "activation_ms")[rows])
    return {
        "MaxActivations": float(most),
        "RewardEfficiency": earned / most if most > 0 else np.nan,
        "_FixedMain": earned,
        "MedianReturnSec": _median(returns), "_ReturnSec": returns,
        "LockedOutContactSec": float(np.count_nonzero(locked)) / ctx.sps,
    }


def _suppressions(ctx: _DFMLight, acts: _Activations, inside: np.ndarray,
                  tol: float) -> dict[str, Any]:
    """Suppressions and MedianSuppressionSec (the well ran NonFeeding in the
    window): its judgeable NonFeeding Activations that ``MaxOnMs`` did not
    end, so contact did, and the dark time from each to the next Activation
    of its state run."""
    max_on = np.nan_to_num(acts.setting("max_on_ms"))
    by_max = (max_on > 0) & (np.abs(acts.lit_ms - max_on) <= acts.slack_ms(tol, ctx.record_ms))
    suppressed = inside & acts.judgeable & acts.is_type(op.NON_FEEDING) & ~by_max
    dark = (acts.next_onset_ms[suppressed] - acts.end_ms[suppressed]) / 1000.0
    dark = dark[~np.isnan(dark)]
    return {"Suppressions": float(np.count_nonzero(suppressed)),
            "MedianSuppressionSec": _median(dark), "_SuppressionSec": dark}


def _light_index(lit: float, dark: float) -> float:
    """(lit - dark) / (lit + dark); ``NA`` when either rate is missing or
    both are 0."""
    if pd.isna(lit) or pd.isna(dark) or lit + dark <= 0:
        return np.nan
    return float((lit - dark) / (lit + dark))


#: The least time, in minutes, a light phase must last before its licks per
#: minute is given.  A few seconds of dark (an Interval boundary, a late
#: record) cannot stand for a fly's feeding in the dark; with less, the rate
#: and the LightFeedingIndex are NA.
MIN_LIGHT_PHASE_MIN = 1.0


def _open_loop_rates(ctx: _DFMLight, phases: dict, wells: Sequence[int]) -> dict[str, Any]:
    """LicksPerMinLit, LicksPerMinDark and LightFeedingIndex of *wells*' licks,
    by the chamber's own light (*phases*); a phase shorter than
    :data:`MIN_LIGHT_PHASE_MIN` has no rate."""
    rates = {}
    for phase, column in (("light", "LicksPerMinLit"), ("dark", "LicksPerMinDark")):
        counts = phases[phase]
        phase_min = counts.samples / ctx.sps / 60.0
        licks = sum(counts.licks[int(w)] for w in wells)
        rates[column] = float(licks) / phase_min if phase_min >= MIN_LIGHT_PHASE_MIN else np.nan
    rates["LightFeedingIndex"] = _light_index(rates["LicksPerMinLit"], rates["LicksPerMinDark"])
    return rates


def _open_loop_phases(ctx: _DFMLight, chamber: int, win: _Window) -> dict | None:
    """The chamber's light and dark over *win*
    (:func:`analytics.chamber_light_phases`) when its light there does not
    depend on its fly (decision 11): its Role is Open, or its wells run open
    loop (``On``, or under OpenLoop) in the window and no rule of theirs
    their fly drives.  ``None`` otherwise, and off generation 3 with a
    program.

    Only the window's rows under a Program Interval count: the baseline
    lead-in (``Interval`` 0) and rows under an unknown instruction are
    dark because no program step ran, not because the program turned the
    light off.  Counting them made a chamber lit for the whole program look
    as if it had a dark phase of a few seconds."""
    if not ctx.typed or not win.size:
        return None
    cols = [int(w) - 1 for w in ctx.chambers[chamber].wells]
    if ctx.roles.get(chamber) != ROLE_OPEN:
        open_loop = ctx.open_loop[win.rows][:, cols].any()
        driven = ctx.driven[win.rows][:, cols].any()
        if not open_loop or driven:
            return None
    from .analytics import chamber_light_phases

    in_program = ctx.state.program_interval.to_numpy(dtype=np.int64, na_value=-1) > 0
    rows = win.rows[in_program[win.rows]]
    return chamber_light_phases(ctx.dfm, ctx.chambers[chamber].wells, lit=ctx.lit, rows=rows)


def _acclimation(ctx: _DFMLight, w: int, win: _Window, acts: _Activations,
                 inside: np.ndarray) -> dict[str, Any]:
    """The Acclimation columns: defined where the well ran a FixedInterval or
    ratio Paradigm with Acclimation in the window.  AcclimationActivations
    counts the ledger's Phase-``A`` Activations with their onset in the
    window, as MainActivations counts its Phase-``M`` ones, so the two sum
    to Activations: the row's Phase misreads a row where an Acclimation
    Activation started and ended inside a record that already reads ``M``,
    the row a well leaves Acclimation or its first under the Paradigm in a
    state run (``activation_ledger``).  The end is the first row in Main
    after the window's first run in Phase ``A``, or the row of an
    Acclimation that ran inside one row (``_Activations.unseen``);
    a well still in ``A`` when the window ends has none and is not
    complete.  A run whose last row is the window's last row ended in the
    window: a Progressive Ratio Training window ends exactly there."""
    if not ctx.type_rows(w, op.ACCLIMATION_PARADIGMS, win, acclimation=True).any():
        return {}
    state = ctx.state
    count = float(np.count_nonzero(inside & ~acts.main))
    runs = [(a, b) for a, b in state.acclimation_runs(w)
            if any(a < sb and b > sa for sa, sb in win.segments)]
    ## An Acclimation no row reads as Phase A began and ended in its row.
    runs = sorted(runs + [(r, r) for r in np.unique(acts.row[inside & acts.unseen]).tolist()])
    if not runs:
        return {"AcclimationActivations": count}

    def ended(end: int) -> bool:
        return end <= win.end and end < len(state)

    first_end = runs[0][1]
    return {
        "AcclimationActivations": count,
        "AcclimationEndMin": float(state.minutes[first_end]) if ended(first_end) else np.nan,
        "AcclimationComplete": ended(runs[-1][1]),
    }


def _ratio(ctx: _DFMLight, w: int, win: _Window) -> dict[str, Any]:
    """The ratio columns: defined where the well ran ProgressiveRatio or
    FixedRatio in the window."""
    if not ctx.type_rows(w, op.RATIO_PARADIGMS, win).any():
        return {}
    state, col, rows = ctx.state, w - 1, win.rows
    events = float(sum(state.firmware_events_between(w, a, b) for a, b in win.segments))
    shown = ~np.isnan(state.requirement[rows, col])
    main = float(ctx.steps[rows, col][shown & ~state.phase_a[rows, col]].sum())
    last = np.flatnonzero(shown)
    progress = (float(np.nan_to_num(state.progress[rows[last[-1]], col], nan=0.0))
                if last.size else np.nan)
    return {
        "FirmwareEvents": events,
        "LastRequirementMet": _last_requirement(ctx, w, win),
        "ProgressAtEnd": progress,
        "EventsPerActivation": events / main if main > 0 else np.nan,
        "_Main": main,
    }


def _last_requirement(ctx: _DFMLight, w: int, win: _Window) -> float:
    """The Requirement the window's last Main ratio Activation met.  An
    Activation on the first row of a Paradigm (no row before shows its
    Requirement) met the Paradigm's first one."""
    state, col = ctx.state, w - 1
    onsets = state.activation_onsets(w)
    for r in onsets[win.contains(onsets)][::-1]:
        met = state.requirement_met(w, int(r))
        if met is not None:
            return float(met)
        starts = r > 0 and ctx.names[r] != ctx.names[r - 1]
        paradigm = ctx.paradigm_by_code[ctx.codes[r]]
        if starts and paradigm is not None and paradigm.type in op.RATIO_PARADIGMS \
                and not state.phase_a[r, col] and ctx.trigger[r, col]:
            first = paradigm.requirement_for(1)
            if first is not None:
                return float(first)
    return np.nan


# ---------------------------------------------------------------------------
# A chamber over one window
# ---------------------------------------------------------------------------

def _chamber_light(ctx: _DFMLight, wells: Sequence[int], win: _Window) -> tuple[float, int]:
    """``(LitSec, LightOnsets)`` of the chamber's own light: lit while any of
    its wells is; on generation 3 the per-record maximum of their ``OnMs``,
    since linked wells share one light and a sum would count it twice."""
    cols = [int(w) - 1 for w in wells]
    if not cols or not win.size:
        return 0.0, 0
    lit = ctx.lit[win.rows][:, cols].any(axis=1)
    before = ctx.lit_before[win.rows][:, cols].any(axis=1)
    if ctx.on_ms is not None:
        seconds = float(ctx.on_ms[win.rows][:, cols].max(axis=1).sum()) / 1000.0
    else:
        seconds = float(np.count_nonzero(lit)) / ctx.sps
    return seconds, int(np.count_nonzero(lit & ~before))


def _chamber_values(ctx: _DFMLight, chamber: int, wells: dict[int, dict],
                    win: _Window, phases: dict | None = None) -> dict[str, Any]:
    ch = ctx.chambers[chamber]
    values = list(wells.values())
    minutes = win.size / ctx.sps / 60.0
    lit_sec, onsets = _chamber_light(ctx, list(wells), win)
    out: dict[str, Any] = {"WindowMinutes": minutes, "LitSec": lit_sec,
                           "LitFraction": _fraction(lit_sec, minutes),
                           "LitSource": ctx.source, "LightOnsets": float(onsets)}
    for column in _COUNT_COLUMNS:
        out[column] = _sum(v[column] for v in values)
    out["ActivationsPerHour"] = _per_hour(out["Activations"], minutes)
    ## The medians pool the wells' Activations; a well off the family has none.
    for column, hidden in _POOLED.items():
        out[column] = _median(np.concatenate([v[hidden] for v in values])) \
            if any(pd.notna(v[column]) for v in values) else np.nan
    earned = _sum(v["_FixedMain"] for v in values)
    out["RewardEfficiency"] = (earned / out["MaxActivations"]
                               if pd.notna(earned) and pd.notna(out["MaxActivations"])
                               and out["MaxActivations"] > 0 else np.nan)
    out.update(_open_loop_rates(ctx, phases, list(wells)) if phases is not None
               else {c: np.nan for c in OPEN_LOOP_COLUMNS})
    acclimated = [v for v in values if v["AcclimationComplete"] is not None]
    if acclimated:
        complete = all(bool(v["AcclimationComplete"]) for v in acclimated)
        ends = [v["AcclimationEndMin"] for v in acclimated]
        out["AcclimationComplete"] = complete
        out["AcclimationEndMin"] = (max(float(e) for e in ends)
                                    if complete and all(pd.notna(e) for e in ends) else np.nan)
    else:
        out["AcclimationComplete"] = None
        out["AcclimationEndMin"] = np.nan
    met = [float(v["LastRequirementMet"]) for v in values if pd.notna(v["LastRequirementMet"])]
    out["LastRequirementMet"] = max(met) if met else np.nan
    main = _sum(v["_Main"] for v in values)
    out["EventsPerActivation"] = (out["FirmwareEvents"] / main
                                  if pd.notna(out["FirmwareEvents"]) and pd.notna(main)
                                  and main > 0 else np.nan)
    well_a = getattr(ch, "well_a", None)
    if well_a is not None:
        out["EventsA"] = wells[int(well_a)]["PyflicEvents"]
        out["EventsB"] = wells[int(ch.well_b)]["PyflicEvents"]
    else:
        out["EventsA"] = values[0]["PyflicEvents"] if values else np.nan
        out["EventsB"] = np.nan
    return out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _rows(exp, dfm_id: int, range_minutes: Any,
          chambers: Iterable[int] | None = None) -> tuple[list[dict], list[dict]]:
    """``(well rows, chamber rows)`` of one DFM over one window, for the
    chambers the design assigns a Treatment (of *chambers*, when given)."""
    ctx = _context(exp, dfm_id)
    win = ctx.window(range_minutes)
    wanted = None if chambers is None else {int(c) for c in chambers}
    head = {"Generation": ctx.generation, **_window_paradigm(ctx, win)}
    tol = activation_tolerance_ms(exp)
    well_rows: list[dict] = []
    chamber_rows: list[dict] = []
    for chamber in sorted(ctx.chambers):
        if wanted is not None and chamber not in wanted:
            continue
        treatment = exp.design.treatment_for(ctx.dfm_id, chamber)
        if treatment is None:
            continue
        ident = {"DFM": ctx.dfm_id, "Chamber": chamber, "Treatment": treatment,
                 "Role": ctx.roles.get(chamber, "")}
        phases = _open_loop_phases(ctx, chamber, win)
        wells = {int(w): _well_values(ctx, int(w), win, tol=tol, phases=phases)
                 for w in sorted(ctx.chambers[chamber].wells)}
        for w, values in wells.items():
            well_rows.append({**ident, "Well": w, **head, **values})
        chamber_rows.append({**ident, **head,
                             **_chamber_values(ctx, chamber, wells, win, phases)})
    return well_rows, chamber_rows


def _frame(exp, rows: list[dict], columns: Sequence[str], lead: Sequence[str] = ()) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=[*lead, *columns]) if rows else \
        pd.DataFrame(columns=[*lead, *columns])
    for column in ("ProgramInterval", "Cycle", "Paradigm", "ParadigmType", "AcclimationComplete"):
        df[column] = df[column].astype(object).where(df[column].notna(), None)
    if df.empty:
        return df
    return exp._append_factor_columns(df) if hasattr(exp, "_append_factor_columns") else df


def _dfm_ids(exp) -> list[int]:
    return list(exp.opto.dfm_ids()) if getattr(exp, "is_optogenetic", False) else []


def light_wells(exp, *, range_minutes: Any = (0, 0)) -> pd.DataFrame:
    """One row per well of every assigned chamber of every DFM the
    optogenetic light QC covers, over *range_minutes* (the feeding summary's
    ``(a, b]``, or a list of them; ``(0, 0)``: the whole recording).
    Columns :data:`LIGHT_WELL_COLUMNS`, the design's factors after
    ``Treatment``."""
    rows = [r for d in _dfm_ids(exp) for r in _rows(exp, d, range_minutes)[0]]
    return _frame(exp, rows, LIGHT_WELL_COLUMNS)


def light_summary(exp, *, range_minutes: Any = (0, 0)) -> pd.DataFrame:
    """One row per assigned chamber of every DFM the optogenetic light QC
    covers (:data:`LIGHT_CHAMBER_COLUMNS`), over *range_minutes*."""
    rows = [r for d in _dfm_ids(exp) for r in _rows(exp, d, range_minutes)[1]]
    return _frame(exp, rows, LIGHT_CHAMBER_COLUMNS)


def facet_windows(exp, dfm_id: int) -> list[tuple[str, Any, list[int] | None]]:
    """``[(label, range_minutes, chambers or None), ...]``: the Facets one DFM
    is split by, each window as the feeding summary takes it.

    * Progressive Ratio: each Chamber Group's own (:meth:`group_windows`);
    * else the experiment's per-DFM Facets
      (:meth:`Experiment.facet_windows_for`, cutoff or Program Facets): each
      a label and one or more windows, the list passed through.
    """
    group_windows = getattr(exp, "group_windows", None)
    if callable(group_windows) and type_owns_role(exp):
        groups: dict[int, list[int]] = {}
        for ch in exp.dfms[int(dfm_id)].chambers or []:
            groups.setdefault(int(exp.role_of(dfm_id, int(ch.index)).group), []).append(
                int(ch.index))
        return [(label, window, chambers) for group, chambers in groups.items()
                for label, window in group_windows(dfm_id, group)]
    return [(label, windows, None) for label, windows in exp.facet_windows_for(dfm_id)]


def _as_range(window: Any) -> Any:
    """A Facet window as ``range_minutes``: one ``(a, b)`` through
    :func:`windowing.as_range_minutes`, a list as it is."""
    return window if windowing.is_window_list(window) else windowing.as_range_minutes(window)


def _facet_tables(exp) -> tuple[pd.DataFrame, pd.DataFrame]:
    wells: list[dict] = []
    chambers: list[dict] = []
    for dfm_id in _dfm_ids(exp):
        for label, window, which in facet_windows(exp, dfm_id):
            lead = {"Facet": label, "FacetRange": _format_ranges(window)}
            w_rows, c_rows = _rows(exp, dfm_id, _as_range(window), which)
            wells += [{**lead, **r} for r in w_rows]
            chambers += [{**lead, **r} for r in c_rows]
    lead = ("Facet", "FacetRange")
    out = []
    for rows, columns, keys in ((wells, LIGHT_WELL_COLUMNS, ["DFM", "Chamber", "Well"]),
                                (chambers, LIGHT_CHAMBER_COLUMNS, ["DFM", "Chamber"])):
        df = _frame(exp, rows, columns, lead)
        if not df.empty:
            labels = list(dict.fromkeys([*map(str, exp.facet_labels()),
                                         *df["Facet"].astype(str)]))
            df["_o"] = df["Facet"].astype(str).map({l: i for i, l in enumerate(labels)})
            df = df.sort_values(["_o", *keys], kind="stable").drop(columns="_o")
            df = df.reset_index(drop=True)
        out.append(df)
    return out[0], out[1]


def light_wells_facet(exp) -> pd.DataFrame:
    """:func:`light_wells` once per Facet, stacked, led by ``Facet`` and
    ``FacetRange`` as :meth:`Experiment.feeding_summary_facet` is; empty when
    the experiment is not faceted."""
    return _facet_tables(exp)[0]


def light_summary_facet(exp) -> pd.DataFrame:
    """:func:`light_summary` once per Facet, stacked (see
    :func:`light_wells_facet`)."""
    return _facet_tables(exp)[1]


#: ``analysis/`` file names of the four tables, keyed as basic analysis
#: returns them.
FILES: dict[str, str] = {
    "light_wells": "light_wells.csv",
    "light_summary": "light_summary.csv",
    "light_wells_facet": "light_wells_facet.csv",
    "light_summary_facet": "light_summary_facet.csv",
}


def write(exp, out_dir=None, *, range_minutes: Any = (0, 0)) -> dict:
    """Write the light tables into ``analysis/`` and return their paths.
    The ``_facet`` files are written only for a faceted experiment, as
    ``feeding_summary_facet.csv`` is: an absent file means "no Facets"."""
    from pathlib import Path

    if out_dir is None:
        if exp.analysis_dir is None:
            raise ValueError("experiment_dir must be set to write the light tables.")
        out_dir = exp.analysis_dir
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    wells_facet, summary_facet = _facet_tables(exp)
    tables = {
        "light_wells": light_wells(exp, range_minutes=range_minutes),
        "light_summary": light_summary(exp, range_minutes=range_minutes),
        "light_wells_facet": wells_facet,
        "light_summary_facet": summary_facet,
    }
    written: dict = {}
    for key, frame in tables.items():
        if key.endswith("_facet") and frame.empty:
            continue
        path = out / FILES[key]
        frame.to_csv(path, index=False, na_rep="NA")
        written[key] = path
    return written


# ---------------------------------------------------------------------------
# The feeding summary's light columns
# ---------------------------------------------------------------------------

def with_light_columns(exp, df: pd.DataFrame, range_minutes: Any = (0, 0)) -> pd.DataFrame:
    """*df* (per-chamber feeding-summary rows over *range_minutes*, one
    ``(a, b]`` window or a list) with ``Role`` after ``Chamber`` and
    ``LitSec`` and ``Activations`` after the ``OptoOn_sec`` columns, on an
    optogenetic experiment.

    Redone on every call, as the light QC's columns are: a summary read back
    from the disk cache carries whatever they said when it was written.  A
    ``Role`` the Experiment Type writes (Progressive Ratio) is never touched;
    rows of a DFM the light QC does not cover get an empty Role and NaN.
    """
    if df is None or df.empty or "DFM" not in df.columns or "Chamber" not in df.columns:
        return df
    owns_role = type_owns_role(exp)
    stale = [c for c in FEEDING_COLUMNS if c in df.columns and not (c == "Role" and owns_role)]
    if stale:
        df = df.drop(columns=stale)
    if not getattr(exp, "is_optogenetic", False):
        return df
    covered = set(exp.opto.dfm_ids())
    keys = list(zip(pd.to_numeric(df["DFM"]).astype(int), pd.to_numeric(df["Chamber"]).astype(int)))
    found: dict[tuple[int, int], tuple[str, float, float]] = {}
    for dfm_id in sorted({d for d, _c in keys if d in covered and d in exp.dfms}):
        ctx = _context(exp, dfm_id)
        win = ctx.window(range_minutes)
        for chamber, ch in ctx.chambers.items():
            cols = [int(w) - 1 for w in ch.wells]
            lit_sec, _onsets = _chamber_light(ctx, ch.wells, win)
            activations = (float(ctx.steps[win.rows][:, cols].sum())
                           if ctx.steps is not None else np.nan)
            found[(dfm_id, chamber)] = (ctx.roles.get(chamber, ""), lit_sec, activations)
    df = df.copy()
    if not owns_role:
        df.insert(list(df.columns).index("Chamber") + 1, "Role",
                  [found[k][0] if k in found else "" for k in keys])
    anchor = [i for i, c in enumerate(df.columns) if str(c).startswith("OptoOn_sec")]
    at = (anchor[-1] + 1) if anchor else len(df.columns)
    df.insert(at, "LitSec", [found[k][1] if k in found else np.nan for k in keys])
    df.insert(at + 1, "Activations", [found[k][2] if k in found else np.nan for k in keys])
    return df


# ---------------------------------------------------------------------------
# For the reports
# ---------------------------------------------------------------------------

def program_headline_metrics(programs: Iterable[tuple[Any, Iterable[int] | None]]) -> list[str]:
    """The headline metrics of the light table (decision 9) for members
    running *programs*, each an ``(OptoProgram, DFM ids or None for every
    section)`` pair: :func:`paradigm_families.headline_metrics
    <pyflic.base.paradigm_families.headline_metrics>` of the families
    :func:`~pyflic.base.paradigm_families.families_in` finds, fixed length
    given by its branches, ``CutOffs`` with a ClosedLoop ``MaxOnMs`` and
    ``AcclimationEndMin`` with Acclimation, pooled per family as a Project
    pools them.  :data:`HEADLINE_METRICS` when no program runs a family (no
    program, or one recorded without a program)."""
    keys: list[str] = []
    max_on = acclimation = False
    for program, dfm_ids in programs:
        if program is None:
            continue
        ids = None if dfm_ids is None else list(dfm_ids)
        keys += [f for f in pfam.families_in(program, ids) if f != pfam.FIXED_LENGTH]
        keys += pfam.branches_in(program, ids)
        max_on |= pfam.max_on_in(program, ids)
        acclimation |= pfam.acclimation_in(program, ids)
    if not keys:
        return list(HEADLINE_METRICS)
    return pfam.headline_metrics(keys, max_on=max_on, acclimation=acclimation)


def experiment_headline_metrics(exp) -> list[str]:
    """The headline metrics of *exp*'s light table: those of the families
    its program runs on its generation-3 DFMs (:func:`program_headline_metrics`).
    A member with no generation-3 DFM the light QC covers keeps
    :data:`HEADLINE_METRICS`, as before the families: its light table has
    lit time only (decision 3)."""
    if not getattr(exp, "is_optogenetic", False):
        return list(HEADLINE_METRICS)
    ids = [d for d in exp.opto.dfm_ids() if int(exp.dfms[d].generation) == GENERATION_PARADIGM]
    if not ids:
        return list(HEADLINE_METRICS)
    return program_headline_metrics([(getattr(exp, "opto_program", None), ids)])


#: The light table's columns that count or time a chamber's own Activations
#: (decision 12 of ``docs/paradigm-analyses-plan.md``): only a chamber whose
#: Role earns light has them to compare.
ACTIVATION_COLUMNS: tuple[str, ...] = (
    "Activations", "ActivationsPerHour",
    "AcclimationActivations", "AcclimationEndMin", "AcclimationComplete", "MainActivations",
    "FirmwareEvents", "LastRequirementMet", "ProgressAtEnd", "EventsPerActivation",
    *CLOSED_LOOP_COLUMNS, *FIXED_INTERVAL_COLUMNS, "Suppressions", "MedianSuppressionSec")


def behaviour_rows(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    """The light table's rows for comparing light behaviour between
    treatments, by Role (decision 12 of ``docs/paradigm-analyses-plan.md``).

    * A chamber whose Role earns light (Paired, Self, Mutual:
      :data:`opto_roles.TRIGGER_ROLES <pyflic.base.opto_roles.TRIGGER_ROLES>`)
      keeps every column, and so does one whose Role is unknown, since
      nothing says its light is not its own.
    * Any other chamber (Yoked, Open, Unlit) keeps its row, its Role and its
      light (``LitSec``, ``LitFraction``), so a yoked fly's light is seen
      beside its partner's, but its :data:`ACTIVATION_COLUMNS` are missing:
      its Activations are 0 by construction, and pooled they would dilute
      the chambers that earn light.
    * A table carrying Progressive Ratio's own ``paired`` / ``yoked`` Roles
      keeps its paired rows, as before: that type compares its paired
      chambers only.  (A Project pooling such a member with others drops
      its ``yoked`` rows and keeps its ``paired`` ones whole.)
    """
    from .opto_roles import ROLE_UNKNOWN, TRIGGER_ROLES

    if frame is None or frame.empty or "Role" not in frame.columns:
        return frame
    roles = frame["Role"].fillna("").astype(str).str.strip()
    known = set(roles[roles != ""])
    if known and known <= set(_PR_ROLES):
        return frame[roles == "paired"]
    kept = (roles != "yoked").to_numpy()
    out = frame if kept.all() else frame[kept]
    blank = ~roles.isin((*TRIGGER_ROLES, ROLE_UNKNOWN, "paired")).to_numpy()[kept]
    columns = [c for c in ACTIVATION_COLUMNS if c in out.columns]
    if not blank.any() or not columns:
        return out
    out = out.copy()
    for column in columns:
        if out[column].dtype == object or out[column].dtype == bool:
            out[column] = out[column].astype(object).where(~blank, None)
        else:
            out[column] = out[column].where(~blank, np.nan)
    return out


def has_generation_2(frame: pd.DataFrame | None) -> bool:
    """Whether a saved light table has any generation-2 row."""
    if frame is None or frame.empty or "Generation" not in frame.columns:
        return False
    return bool((pd.to_numeric(frame["Generation"], errors="coerce") == 2).any())


## What a generation-2 member says wherever its Training would show is
## paradigm_state.NO_TRAINING_INFORMATION, imported above and used from here
## by the Hub and the Project Report.


def no_training_information(exp) -> bool:
    """Whether a loaded experiment has a DFM without Training information.

    The Experiment Type is asked first when it can answer
    (``training_information()``, Progressive Ratio): a text answer of
    ``"none"`` or one starting "no training information" means none.
    Otherwise, any DFM recorded as generation 2 (``training_info ==
    "none"``)."""
    ask = getattr(exp, "training_information", None)
    if callable(ask):
        try:
            answer = ask()
        except Exception:  # noqa: BLE001 - a status line must not fail
            answer = None
        if isinstance(answer, str):
            text = answer.strip().lower()
            return text == "none" or text.startswith("no training information")
    dfms = getattr(exp, "dfms", None) or {}
    return any(getattr(d, "training_info", None) == "none" for d in dfms.values())
