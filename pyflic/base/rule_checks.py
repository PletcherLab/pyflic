"""The rule checks: each Paradigm checked against its own rule, one Activation at a time.

Firmware 4.0.0 records every Activation and its light to the millisecond
(:mod:`pyflic.base.activation_ledger`), and DFMV3_V2's ``LEDControl.c`` keeps
each Paradigm type's rule exactly.  So a light period that breaks the rule
of the Paradigm ``Program.txt`` names means the file is not the program that
ran, or the LED or its report is faulty (``docs/paradigm-analyses-plan.md``,
stream S1 and decisions 5 to 7; ADR-0015's last addendum).

One function per check.  Each reads the ledger rows of a Linkage Group's
Trigger Wells under one Paradigm (:data:`~pyflic.base.activation_ledger.LEDGER_COLUMNS`),
the Paradigm's settings and, where the rule turns on contact, pyflic's
activity at the well, and returns a :class:`CheckResult`: the count that is
its column in ``opto_light_qc.csv``, how many it could judge, whether its flag
fires and the reason in words, naming the setting and the value measured.

**Judgeable.**  A row is judgeable for a length when it is not ``Overlap``
and its ``Boundary`` is empty (:func:`judgeable`); the rest are counted as
not judged.  ``tol`` is ``opto_activation_tolerance_ms``; a row whose
``Timing`` is ``record`` is good to one record only, so its tolerance widens
by one (:func:`tolerances`).

========================  ===========================================  ==========================
Check (flag)              Applies to                                   Verdict
========================  ===========================================  ==========================
short activations         ClosedLoop Main; every Acclimation           fails above a fraction
max on exceeded           ClosedLoop and NonFeeding with ``MaxOnMs``   fails on any one
(cut-offs)                ClosedLoop with ``MaxOnMs``                  reported
wrong activation length   FixedInterval, PR and FR in Main             fails above a fraction
lockout broken            FixedInterval Main                           fails above a fraction
contact without           FixedInterval Main                           warns above a fraction
activation
requirement series wrong  ProgressiveRatio and FixedRatio Main         fails on any one
non-feeding dark          NonFeeding                                   warns above a fraction
(relight delay)           NonFeeding                                   reported
open loop not lit         the lowest open-loop window                  warns below a fraction
interval started late     every Program Interval's start               warns above seconds
DFM restarted             each restart the rows show                   warns
========================  ===========================================  ==========================

:func:`requirement_series_mismatches` is importable on its own: the
Progressive Ratio light QC calls it too.  :func:`ended_by` says, for the light
events, what ended each Activation.

Pure numpy and pandas over the ledger, the Paradigm and plain arrays; no Qt.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from . import opto_program as op
from .activation_ledger import (BOUNDARY_END, BOUNDARY_GAP, BOUNDARY_INTERVAL, BOUNDARY_NONE,
                                TIMING_RECORD)
from .paradigm_state import RECORD_MS, ParadigmState, RestartSpan

# ---------------------------------------------------------------------------
# Flags, columns and words
# ---------------------------------------------------------------------------

SHORT_ACTIVATIONS = "short activations"
MAX_ON_EXCEEDED = "max on exceeded"
WRONG_ACTIVATION_LENGTH = "wrong activation length"
LOCKOUT_BROKEN = "lockout broken"
CONTACT_WITHOUT_ACTIVATION = "contact without activation"
REQUIREMENT_SERIES_WRONG = "requirement series wrong"
NON_FEEDING_DARK = "non-feeding dark"
INTERVAL_STARTED_LATE = "interval started late"
DFM_RESTARTED = "DFM restarted"
#: What the cut-off count reports under (it raises no flag).
CUT_OFFS = "cut-offs"
#: What the relight delay reports under (it raises no flag).
RELIGHT_DELAY = "relight delay"

#: Decision 6: these fail a Linkage Group, because the firmware is
#: deterministic, so a miss means a wrong program or broken hardware ...
FAILING_RULE_FLAGS: tuple[str, ...] = (WRONG_ACTIVATION_LENGTH, LOCKOUT_BROKEN,
                                       REQUIREMENT_SERIES_WRONG, MAX_ON_EXCEEDED,
                                       SHORT_ACTIVATIONS)
#: ... and these warn, because they rest on pyflic's lick detection against
#: the firmware's threshold, or on the DFM's own bookkeeping.
WARNING_RULE_FLAGS: tuple[str, ...] = (CONTACT_WITHOUT_ACTIVATION, NON_FEEDING_DARK,
                                       INTERVAL_STARTED_LATE, DFM_RESTARTED)

#: The checks' columns in ``opto_light_qc.csv`` (``paradigm_families``
#: declares them per family; ``OpenLoopLitFraction`` was there already).
SHORT_COLUMN = "ShortActivations"
OVER_MAX_ON_COLUMN = "OverMaxOnActivations"
CUT_OFFS_COLUMN = "CutOffs"
WRONG_LENGTH_COLUMN = "WrongLengthActivations"
LOCKOUT_COLUMN = "LockoutViolations"
MISSED_CONTACTS_COLUMN = "MissedContacts"
SERIES_COLUMN = "SeriesMismatches"
NON_FEEDING_DARK_COLUMN = "NonFeedingDarkSec"
RELIGHT_DELAY_COLUMN = "RelightDelayMs"
OPEN_LOOP_COLUMN = "OpenLoopLitFraction"
INTERVAL_LAG_COLUMN = "IntervalLagSec"
RESTARTS_COLUMN = "Restarts"
#: The new columns, in the order the QC table lists them.
RULE_QC_COLUMNS: tuple[str, ...] = (
    SHORT_COLUMN, OVER_MAX_ON_COLUMN, CUT_OFFS_COLUMN, WRONG_LENGTH_COLUMN, LOCKOUT_COLUMN,
    MISSED_CONTACTS_COLUMN, SERIES_COLUMN, NON_FEEDING_DARK_COLUMN, RELIGHT_DELAY_COLUMN,
    INTERVAL_LAG_COLUMN, RESTARTS_COLUMN,
)
#: Those that are counts, held as nullable integers.
RULE_QC_INTEGER_COLUMNS: tuple[str, ...] = (
    SHORT_COLUMN, OVER_MAX_ON_COLUMN, CUT_OFFS_COLUMN, WRONG_LENGTH_COLUMN, LOCKOUT_COLUMN,
    MISSED_CONTACTS_COLUMN, SERIES_COLUMN, RESTARTS_COLUMN,
)

#: ``EndedBy`` in ``opto_light_events.csv``: what ended an Activation's light
#: period.  The ledger's ``Boundary`` values (``interval``, ``gap``, ``end``)
#: come first; then ``unknown`` for a period merged with other light; then
#: the family's rule.
ENDED_CONTACT, ENDED_HOLD, ENDED_MAX_ON, ENDED_LENGTH, ENDED_UNKNOWN = (
    "contact", "Hold", "MaxOnMs", "length", "unknown")
ENDED_BY_VALUES: tuple[str, ...] = (ENDED_CONTACT, ENDED_HOLD, ENDED_MAX_ON, ENDED_LENGTH,
                                    BOUNDARY_INTERVAL, BOUNDARY_GAP, BOUNDARY_END,
                                    ENDED_UNKNOWN)

#: Decision 5: what a failing check's miss most likely means.
LIKELY_CAUSE = ("the firmware keeps this rule to the millisecond, so Program.txt is probably "
                "not the program that ran, or the LED or its OnMs report is faulty")

#: Free runs of pyflic activity *contact without activation* needs before
#: it judges (the plan's "at least 5 free runs").
MIN_FREE_RUNS = 5
#: Seconds a NonFeeding rule must say lit before *non-feeding dark* judges:
#: fifty records.  What the rule says lit for certain is shortened by two
#: records at every bout (pyflic's activity is per record), so a short
#: ``MaxOnMs`` leaves a sliver per bout whose edges are all uncertainty.
MIN_EXPECTED_LIT_SEC = 10.0
#: How long after its Start Time a DFM's first Program Interval may begin.
#: PyMCU sends a DFM no instruction until it has taken its baseline
#: (``DFMGroup.UpdateProgramStatus``), so the first Interval follows a
#: lead-in of Interval-0 rows: 20 records, its first row stamped 4.8 s after
#: the Start Time, on the bench (``FLICMCU_10_17_2026_00_17``) and in
#: PyMCU's simulator; 25 records (5.2 s) in pyflic's.  The first Interval's
#: lag is counted past this allowance; every later start's from its time.
LEAD_IN_SEC = 5.0
#: Examples a reason quotes, at most.
_EXAMPLES = 3


@dataclass(frozen=True)
class CheckResult:
    """What one rule check found among one Paradigm's light periods.

    *applies* is false when the Paradigm has no such rule (the column is
    then ``NA``).  *value* is the check's column (a count, seconds or
    milliseconds); *judged* what it could judge (Activations, contact runs,
    or seconds the rule says lit) and *not_judged* the Activations it could
    not.  *fired* says the flag fires, *reason* says why in words, and
    *note* is anything else worth a group note."""

    flag: str
    applies: bool
    value: float | None = None
    judged: float = 0
    not_judged: int = 0
    fired: bool = False
    reason: str = ""
    note: str = ""


def not_applicable(flag: str) -> CheckResult:
    return CheckResult(flag, False)


def combine(results: Iterable[CheckResult]) -> CheckResult | None:
    """One check's results under several Paradigms of a group, summed: the
    value and the counts add up, the flag fires if any fired, and the
    reasons and notes are joined.  ``None`` when none applies."""
    applying = [r for r in results if r.applies]
    if not applying:
        return None
    values = [r.value for r in applying if r.value is not None and not _isnan(r.value)]
    return CheckResult(
        applying[0].flag, True, value=float(sum(values)) if values else None,
        judged=sum(r.judged for r in applying),
        not_judged=sum(r.not_judged for r in applying),
        fired=any(r.fired for r in applying),
        reason="; ".join(r.reason for r in applying if r.fired and r.reason),
        note="; ".join(r.note for r in applying if r.note))


def _isnan(value) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Reading the ledger
# ---------------------------------------------------------------------------

def _numbers(rows: pd.DataFrame, column: str) -> np.ndarray:
    """A ledger column as float, ``NaN`` where missing (``Int64`` or not)."""
    return pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def _overlap(rows: pd.DataFrame) -> np.ndarray:
    return rows["Overlap"].astype("boolean").fillna(True).to_numpy(dtype=bool)


def _phase(rows: pd.DataFrame) -> np.ndarray:
    return rows["Phase"].astype(object).to_numpy()


def judgeable(rows: pd.DataFrame) -> np.ndarray:
    """(len(rows),) bool: rows whose light period is the Activation's own
    and stopped by its rule: not ``Overlap``, ``Boundary`` empty."""
    boundary = rows["Boundary"].astype(object).where(rows["Boundary"].notna(), BOUNDARY_NONE)
    return ~_overlap(rows) & (boundary.to_numpy() == BOUNDARY_NONE)


def tolerances(rows: pd.DataFrame, tol_ms: float, record_ms: int = RECORD_MS) -> np.ndarray:
    """Each row's tolerance in ms: *tol_ms*, widened by one record where its
    ``Timing`` is ``record`` (its times are good to one record only)."""
    record = rows["Timing"].astype(object).to_numpy() == TIMING_RECORD
    return np.where(record, float(tol_ms) + float(record_ms), float(tol_ms))


def _pct(count: float, total: float) -> str:
    return f"{100.0 * count / total:.0f}%" if total else "-"


def _ms(value: float) -> str:
    return f"{value:.0f} ms"


def _span(values: np.ndarray) -> str:
    """``1500 ms`` or ``480-1500 ms, median 1500 ms``."""
    values = np.asarray(values, dtype=float)
    lo, hi = float(values.min()), float(values.max())
    if lo == hi:
        return _ms(lo)
    return f"{lo:.0f}-{hi:.0f} ms, median {float(np.median(values)):.0f} ms"


# ---------------------------------------------------------------------------
# Closed loop
# ---------------------------------------------------------------------------

def _closed_loop_rule(paradigm: op.Paradigm, rows: pd.DataFrame
                      ) -> tuple[np.ndarray, float | None, str]:
    """``(the rows a closed-loop rule lit, the shortest light it allows, its
    words)``: ClosedLoop's Main Activations, at least its ``HoldMs`` (or its
    ``MaxOnMs`` when shorter); the Acclimation Activations of the types that
    have one, at least their Hold, ``ActivationMs``."""
    phase = _phase(rows)
    if paradigm.type == op.CLOSED_LOOP:
        hold = paradigm.hold_ms
        if hold is None:
            return np.zeros(len(rows), dtype=bool), None, ""
        limit, words = float(hold), f"its HoldMs of {int(hold)} ms"
        max_on = int(paradigm.max_on_ms or 0)
        if 0 < max_on < hold:
            limit, words = float(max_on), f"its MaxOnMs of {max_on} ms (shorter than its Hold)"
        return phase == "M", limit, words
    if paradigm.type in op.ACCLIMATION_PARADIGMS and int(paradigm.acclimation or 0) > 0 \
            and paradigm.activation_ms is not None:
        return (phase == "A", float(paradigm.activation_ms),
                f"its Acclimation Hold, ActivationMs {int(paradigm.activation_ms)} ms")
    return np.zeros(len(rows), dtype=bool), None, ""


def short_activations(rows: pd.DataFrame, paradigm: op.Paradigm, *, tol_ms: float,
                      fail_fraction: float, record_ms: int = RECORD_MS) -> CheckResult:
    """*short activations*: closed-loop light periods shorter than the Hold.

    A closed-loop Activation (ClosedLoop in Main, and every Acclimation
    Activation) is lit for the contact and then its Hold, so a judgeable one
    lit for less than min(Hold, ``MaxOnMs`` if above 0) − tol is short.  The
    Hold is ``HoldMs``, or ``ActivationMs`` in Acclimation.  Fails when the
    count is above *fail_fraction* (``opto_activation_length_fail_fraction``)
    of the judgeable ones."""
    applies, limit, words = _closed_loop_rule(paradigm, rows)
    if limit is None:
        return not_applicable(SHORT_ACTIVATIONS)
    ok = judgeable(rows)
    judged = applies & ok
    lit = _numbers(rows, "LitMs")
    short = judged & (lit < limit - tolerances(rows, tol_ms, record_ms))
    count, n = int(short.sum()), int(judged.sum())
    fired = n > 0 and count > fail_fraction * n
    reason = ""
    if fired:
        reason = (f"{SHORT_ACTIVATIONS}: {count} of {n} judged {paradigm.name} closed-loop "
                  f"Activations ({_pct(count, n)}) were lit for less than {words} less "
                  f"{tol_ms:g} ms (opto_activation_tolerance_ms), the shortest for "
                  f"{_ms(lit[short].min())}; opto_activation_length_fail_fraction is "
                  f"{fail_fraction:g}.  A closed-loop Activation is lit for the contact and "
                  f"its Hold, so never less: {LIKELY_CAUSE}")
    return CheckResult(SHORT_ACTIVATIONS, True, value=count, judged=n,
                       not_judged=int((applies & ~ok).sum()), fired=fired, reason=reason)


def _max_on_rows(paradigm: op.Paradigm, rows: pd.DataFrame) -> tuple[np.ndarray, int]:
    """The rows a ``MaxOnMs`` limits, and the limit (0: none): ClosedLoop's
    Main Activations and every NonFeeding one."""
    max_on = int(paradigm.max_on_ms or 0)
    if max_on <= 0:
        return np.zeros(len(rows), dtype=bool), 0
    if paradigm.type == op.CLOSED_LOOP:
        return _phase(rows) == "M", max_on
    if paradigm.type == op.NON_FEEDING:
        return np.ones(len(rows), dtype=bool), max_on
    return np.zeros(len(rows), dtype=bool), 0


def max_on_exceeded(rows: pd.DataFrame, paradigm: op.Paradigm, *, tol_ms: float,
                    record_ms: int = RECORD_MS) -> CheckResult:
    """*max on exceeded*: a judgeable light period of a ClosedLoop or
    NonFeeding Paradigm with ``MaxOnMs`` lit longer than ``MaxOnMs`` + tol.
    The firmware cuts every one off there, so one is enough to fail."""
    applies, max_on = _max_on_rows(paradigm, rows)
    if max_on <= 0:
        return not_applicable(MAX_ON_EXCEEDED)
    ok = judgeable(rows)
    judged = applies & ok
    lit = _numbers(rows, "LitMs")
    over = judged & (lit > max_on + tolerances(rows, tol_ms, record_ms))
    count, n = int(over.sum()), int(judged.sum())
    reason = ""
    if count:
        reason = (f"{MAX_ON_EXCEEDED}: {count} of {n} judged {paradigm.name} Activations were "
                  f"lit for more than its MaxOnMs of {max_on} ms plus {tol_ms:g} ms "
                  f"(opto_activation_tolerance_ms): {_span(lit[over])}.  The firmware cuts "
                  f"every light period off at MaxOnMs: {LIKELY_CAUSE}")
    return CheckResult(MAX_ON_EXCEEDED, True, value=count, judged=n,
                       not_judged=int((applies & ~ok).sum()), fired=count > 0, reason=reason)


def cut_offs(rows: pd.DataFrame, paradigm: op.Paradigm, *, tol_ms: float,
             record_ms: int = RECORD_MS) -> CheckResult:
    """``CutOffs``: judgeable ClosedLoop Main Activations whose light lasted
    ``MaxOnMs`` within tol, the ones its ``MaxOnMs`` cut off.  Reported; no
    flag."""
    if paradigm.type != op.CLOSED_LOOP or int(paradigm.max_on_ms or 0) <= 0:
        return not_applicable(CUT_OFFS)
    applies, max_on = _max_on_rows(paradigm, rows)
    ok = judgeable(rows)
    judged = applies & ok
    lit = _numbers(rows, "LitMs")
    cut = judged & (np.abs(lit - max_on) <= tolerances(rows, tol_ms, record_ms))
    return CheckResult(CUT_OFFS, True, value=int(cut.sum()), judged=int(judged.sum()),
                       not_judged=int((applies & ~ok).sum()))


# ---------------------------------------------------------------------------
# Fixed length
# ---------------------------------------------------------------------------

def wrong_activation_length(rows: pd.DataFrame, paradigm: op.Paradigm, *, tol_ms: float,
                            fail_fraction: float, record_ms: int = RECORD_MS) -> CheckResult:
    """*wrong activation length*: a judgeable Main Activation of
    FixedInterval, ProgressiveRatio or FixedRatio whose light did not last
    ``ActivationMs`` within tol.  The firmware lights one for exactly
    ``ActivationMs``; fails above *fail_fraction*
    (``opto_activation_length_fail_fraction``) of the judgeable ones."""
    if paradigm.type not in (op.FIXED_INTERVAL, *op.RATIO_PARADIGMS) \
            or paradigm.activation_ms is None:
        return not_applicable(WRONG_ACTIVATION_LENGTH)
    length = float(paradigm.activation_ms)
    applies = _phase(rows) == "M"
    ok = judgeable(rows)
    judged = applies & ok
    lit = _numbers(rows, "LitMs")
    wrong = judged & (np.abs(lit - length) > tolerances(rows, tol_ms, record_ms))
    count, n = int(wrong.sum()), int(judged.sum())
    fired = n > 0 and count > fail_fraction * n
    reason = ""
    if fired:
        reason = (f"{WRONG_ACTIVATION_LENGTH}: {count} of {n} judged {paradigm.name} Main "
                  f"Activations ({_pct(count, n)}) did not last its ActivationMs of "
                  f"{int(length)} ms within {tol_ms:g} ms (opto_activation_tolerance_ms): "
                  f"they lasted {_span(lit[wrong])}; opto_activation_length_fail_fraction is "
                  f"{fail_fraction:g}.  The firmware lights a fixed-length Activation for "
                  f"exactly ActivationMs: {LIKELY_CAUSE}")
    return CheckResult(WRONG_ACTIVATION_LENGTH, True, value=count, judged=n,
                       not_judged=int((applies & ~ok).sum()), fired=fired, reason=reason)


def _spacing(paradigm: op.Paradigm) -> float:
    """The shortest time between two FixedInterval onsets: the firmware
    needs both the Activation ended and the lockout passed."""
    return float(max(int(paradigm.lockout_ms or 0), int(paradigm.activation_ms or 0)))


def _previous(rows: pd.DataFrame) -> np.ndarray:
    """Each row's position of the same Well's previous Activation in its
    state run, when that one is in *rows* too, else -1 (the ledger's
    ``SinceLastOnsetMs`` names it; rows are the ledger's, by Well then Row)."""
    wells = rows["Well"].to_numpy(dtype=np.int64)
    onset = _numbers(rows, "OnsetMs")
    since = _numbers(rows, "SinceLastOnsetMs")
    out = np.full(len(rows), -1, dtype=np.int64)
    if len(rows) > 1:
        same = (wells[1:] == wells[:-1]) & np.isfinite(since[1:]) \
            & (np.abs(onset[1:] - onset[:-1] - since[1:]) < 0.5)
        out[1:][same] = np.flatnonzero(same)
    return out


def lockout_violations(rows: pd.DataFrame, paradigm: op.Paradigm, *, tol_ms: float,
                       fail_fraction: float, record_ms: int = RECORD_MS) -> CheckResult:
    """*lockout broken*: consecutive FixedInterval Main Activations of one
    state run whose onsets are closer than max(``LockoutMs``,
    ``ActivationMs``) − tol (``SinceLastOnsetMs``).  Both onsets must be
    readable, so neither may be ``Overlap`` (light already on hides an
    onset); the tolerance widens by a record where either ``Timing`` is
    ``record``.  Fails above *fail_fraction* (``opto_lockout_fail_fraction``)
    of the pairs judged."""
    if paradigm.type != op.FIXED_INTERVAL or paradigm.activation_ms is None:
        return not_applicable(LOCKOUT_BROKEN)
    spacing = _spacing(paradigm)
    phase = _phase(rows)
    overlap = _overlap(rows)
    since = _numbers(rows, "SinceLastOnsetMs")
    tol = tolerances(rows, tol_ms, record_ms)
    prev = _previous(rows)
    has = prev >= 0
    pairs = (phase == "M") & has & (phase[np.maximum(prev, 0)] == "M")
    judged = pairs & ~overlap & ~overlap[np.maximum(prev, 0)]
    tol_pair = np.maximum(tol, tol[np.maximum(prev, 0)])
    broken = judged & (since < spacing - tol_pair)
    count, n = int(broken.sum()), int(judged.sum())
    fired = n > 0 and count > fail_fraction * n
    reason = ""
    if fired:
        reason = (f"{LOCKOUT_BROKEN}: {count} of {n} judged {paradigm.name} Main Activations "
                  f"({_pct(count, n)}) started less than {spacing:.0f} ms (the longer of "
                  f"LockoutMs {int(paradigm.lockout_ms or 0)} and ActivationMs "
                  f"{int(paradigm.activation_ms)}) less {tol_ms:g} ms "
                  f"(opto_activation_tolerance_ms) after the one before, the closest "
                  f"{_ms(since[broken].min())} after; opto_lockout_fail_fraction is "
                  f"{fail_fraction:g}.  The firmware starts none sooner: {LIKELY_CAUSE}")
    return CheckResult(LOCKOUT_BROKEN, True, value=count, judged=n,
                       not_judged=int((pairs & ~judged).sum()), fired=fired, reason=reason)


def _free_runs(rows: pd.DataFrame, paradigm: op.Paradigm, activity: np.ndarray,
               index: np.ndarray, eligible: np.ndarray, *, tol_ms: float,
               record_ms: int) -> tuple[int, int]:
    """``(free runs, missed ones)`` of one FixedInterval Trigger Well
    (:func:`missed_contacts`); *rows* are that well's ledger rows."""
    activity = np.asarray(activity, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    n = activity.size
    P = float(record_ms)
    before = np.zeros(n, dtype=bool)
    before[1:] = activity[:-1]
    run_starts = np.flatnonzero(activity & ~before & eligible)
    if not run_starts.size:
        return 0, 0
    onset_row = rows["Row"].to_numpy(dtype=np.int64)
    order = np.argsort(onset_row, kind="stable")
    onset_row = onset_row[order]
    onset_ms = _numbers(rows, "OnsetMs")[order]
    end_ms = _numbers(rows, "EndMs")[order]
    t_start = (np.asarray(index, dtype=float)[run_starts] - 1.0) * P
    free = np.ones(run_starts.size, dtype=bool)
    hit = np.zeros(run_starts.size, dtype=bool)
    if onset_row.size:
        ## The last Activation counted before each run's first row ...
        after = np.searchsorted(onset_row, run_starts, side="left")
        last = np.maximum(after - 1, 0)
        free = (after == 0) | ((end_ms[last] <= t_start)
                               & (t_start - onset_ms[last] >= _spacing(paradigm)))
        ## ... and one counted in that row, or within tol after it.
        reach = 1 + int(float(tol_ms) // P)
        ok = after < onset_row.size
        hit[ok] = onset_row[after[ok]] <= run_starts[ok] + reach
    return int(free.sum()), int((free & ~hit).sum())


def missed_contacts(rows: pd.DataFrame, paradigm: op.Paradigm,
                    activity: Mapping[int, np.ndarray], index: np.ndarray,
                    eligible: Mapping[int, np.ndarray], *, tol_ms: float,
                    warn_fraction: float, record_ms: int = RECORD_MS,
                    min_runs: int = MIN_FREE_RUNS) -> CheckResult:
    """*contact without activation*: FixedInterval contact that brought no
    Activation though the well was free.

    *rows* are the ledger rows of the group's Trigger Wells (every
    Activation each counted); *activity* maps each Trigger Well to its pyflic
    activity per row (a feeding lick or a tasting sample), *eligible* to the
    rows where it ran *paradigm* in Main as a Trigger Well; *index* is the
    rows' ``Index``, the ledger's time base.

    A run of activity that begins in an eligible row while its well is free
    should bring an Activation: free means the well's last Activation's
    light is over and max(``LockoutMs``, ``ActivationMs``) has passed since
    that onset, at the start of the run's first row (so for the whole row;
    an Acclimation onset counts too, so the lockout is never taken as passed
    sooner than it did).  The run is missed when no Activation of the well
    is counted in its first row or within tol after it (the next row, at
    the default; pyflic's activity is good to a record).  Warns when more
    than *warn_fraction* (``opto_missed_contact_warn_fraction``) of at least
    *min_runs* free runs are missed."""
    if paradigm.type != op.FIXED_INTERVAL or paradigm.activation_ms is None:
        return not_applicable(CONTACT_WITHOUT_ACTIVATION)
    wells = rows["Well"].to_numpy(dtype=np.int64)
    n_free = n_missed = 0
    for well, mask in eligible.items():
        free, missed = _free_runs(rows[wells == int(well)], paradigm, activity[well], index,
                                  mask, tol_ms=tol_ms, record_ms=record_ms)
        n_free += free
        n_missed += missed
    fired = n_free >= int(min_runs) and n_missed > warn_fraction * n_free
    reason = ""
    if fired:
        reason = (f"{CONTACT_WITHOUT_ACTIVATION}: {n_missed} of {n_free} runs of pyflic "
                  f"activity at {op.wells_label(sorted(eligible))} that began with the well "
                  f"free under {paradigm.name} ({_pct(n_missed, n_free)}) brought no "
                  f"Activation; opto_missed_contact_warn_fraction is {warn_fraction:g}.  Free: "
                  f"its Activation over and {_spacing(paradigm):.0f} ms (the longer of "
                  f"LockoutMs and ActivationMs) since its last onset.  pyflic's lick detection "
                  f"is not the firmware's threshold: contact that did not reach the "
                  f"firmware's threshold, or a dead sensor")
    return CheckResult(CONTACT_WITHOUT_ACTIVATION, True, value=n_missed, judged=n_free,
                       fired=fired, reason=reason)


# ---------------------------------------------------------------------------
# Fixed length, ratio branch: the Requirement series
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SeriesMismatch:
    """One ratio Main Activation that met a Requirement the series does not
    give: Well *well*'s *k*-th Main Activation of its state run, counted on
    ledger row *row* at minute *onset_min*, met *met*; ``requirement_for(k)``
    is *expected*."""

    well: int
    row: int
    k: int
    met: int
    expected: int
    onset_min: float


def _state_run_ids(rows: pd.DataFrame) -> np.ndarray:
    """A number per state run of each Well among *rows* (by Well, then
    Row): a run starts where ``SinceLastOnsetMs`` is missing (the first
    Activation of a run) or ``K`` falls back."""
    wells = rows["Well"].to_numpy(dtype=np.int64)
    since = _numbers(rows, "SinceLastOnsetMs")
    k = _numbers(rows, "K")
    new = np.ones(len(rows), dtype=bool)
    if len(rows) > 1:
        prev_k = pd.Series(k).ffill().to_numpy()
        new[1:] = (wells[1:] != wells[:-1]) | np.isnan(since[1:]) | \
            (np.isfinite(k[1:]) & np.isfinite(prev_k[:-1]) & (k[1:] <= prev_k[:-1]))
    return np.cumsum(new)


def _comparable(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(comparable, K, RequirementMet)``: Main rows where both are known
    and the row holds one onset of its Well (two onsets in one row share the
    Requirement on the row before, though the second met the next)."""
    k = _numbers(rows, "K")
    met = _numbers(rows, "RequirementMet")
    two = rows.duplicated(["Well", "Row"], keep=False).to_numpy(dtype=bool)
    main = _phase(rows) == "M"
    return main & np.isfinite(k) & np.isfinite(met) & ~two, k, met


def _series_shifted(paradigm: op.Paradigm, rows: pd.DataFrame, run: np.ndarray,
                    comparable: np.ndarray, k: np.ndarray, met: np.ndarray) -> bool:
    """Whether one state run's mismatches are the ledger's off-by-one: its
    first Main Activation was counted on the run's first row, which shows
    no Requirement (``RequirementMet`` missing), and may be the old
    Paradigm's, the instruction arriving within the record; if so every K
    of the run is one too many, and with the K one less every Requirement
    the run met is the series'."""
    phase = _phase(rows)
    first = np.flatnonzero(run & (phase == "M") & (k == 1))
    if not first.size or np.isfinite(met[first[0]]):
        return False
    check = run & comparable & (k >= 2)
    if not check.any():
        return False
    return all(met[i] == paradigm.requirement_for(int(k[i]) - 1) for i in np.flatnonzero(check))


def requirement_series_mismatches(ledger_rows: pd.DataFrame, paradigm: op.Paradigm
                                  ) -> list[SeriesMismatch]:
    """The Main Activations of a ProgressiveRatio or FixedRatio *paradigm*
    among *ledger_rows* (the Activation ledger's rows, any Wells, by Well
    then Row) whose ``RequirementMet`` is not ``paradigm.requirement_for(K)``,
    where both are known and the row holds one onset of its Well.

    A state run whose first Main Activation was counted on the run's first
    row is resolved before it fails: that row shows no Requirement, and the
    Activation may be the old Paradigm's (an instruction arriving within a
    record leaves the record's start to the old one), which leaves every K of
    the run one too many.  When the run's Requirements all match the series
    with K one less, the run has no mismatch (:func:`series_shifted_runs`
    counts them).  Empty for any other Paradigm type, and for a series the
    program does not state."""
    return _series_scan(ledger_rows, paradigm)[0]


def series_shifted_runs(ledger_rows: pd.DataFrame, paradigm: op.Paradigm) -> int:
    """How many state runs :func:`requirement_series_mismatches` resolved as
    the ledger's off-by-one (their first Main Activation taken as the old
    Paradigm's)."""
    return _series_scan(ledger_rows, paradigm)[1]


def _series_scan(ledger_rows: pd.DataFrame, paradigm: op.Paradigm
                 ) -> tuple[list[SeriesMismatch], int]:
    """``(the mismatches, the state runs resolved as off by one)``."""
    if paradigm.type not in op.RATIO_PARADIGMS or len(ledger_rows) == 0 \
            or paradigm.requirement_for(1) is None:
        return [], 0
    rows = ledger_rows.reset_index(drop=True)
    comparable, k, met = _comparable(rows)
    runs = _state_run_ids(rows)
    minutes = _numbers(rows, "OnsetMin")
    out: list[SeriesMismatch] = []
    shifted = 0
    for r in np.unique(runs[comparable]).tolist():
        run = runs == r
        sel = np.flatnonzero(run & comparable)
        expected = np.array([paradigm.requirement_for(int(k[i])) for i in sel], dtype=float)
        bad = sel[met[sel] != expected]
        if not bad.size:
            continue
        if _series_shifted(paradigm, rows, run, comparable, k, met):
            shifted += 1
            continue
        out.extend(SeriesMismatch(well=int(rows["Well"].iloc[i]), row=int(rows["Row"].iloc[i]),
                                  k=int(k[i]), met=int(met[i]),
                                  expected=int(paradigm.requirement_for(int(k[i]))),
                                  onset_min=float(minutes[i]))
                   for i in bad.tolist())
    return out, shifted


def _series_words(paradigm: op.Paradigm) -> str:
    if paradigm.type == op.FIXED_RATIO:
        return f"Ratio {paradigm.ratio}"
    values = ", ".join(str(paradigm.requirement_for(k)) for k in range(1, 5))
    if paradigm.series == op.SERIES_EXPONENTIAL:
        how = (f"Exponential, Scale {float(paradigm.scale or 0):g}, "
               f"Rate {float(paradigm.rate or 0):g}")
    else:
        how = (f"Arithmetic, FirstRequirement {paradigm.first_requirement}, "
               f"Step {paradigm.step}")
    return f"{how}: {values}, ..."


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def requirement_series(rows: pd.DataFrame, paradigm: op.Paradigm) -> CheckResult:
    """*requirement series wrong*: :func:`requirement_series_mismatches`
    counted.  It cannot happen on a working system with the right program,
    so one fails."""
    if paradigm.type not in op.RATIO_PARADIGMS or paradigm.requirement_for(1) is None:
        return not_applicable(REQUIREMENT_SERIES_WRONG)
    mismatches, shifted = _series_scan(rows, paradigm)
    comparable, _k, _met = _comparable(rows.reset_index(drop=True))
    main = _phase(rows) == "M"
    n = int(comparable.sum())
    reason = ""
    if mismatches:
        shown = "; ".join(f"W{m.well}'s {_ordinal(m.k)} met {m.met}, the series gives "
                          f"{m.expected} (minute {m.onset_min:.1f})"
                          for m in mismatches[:_EXAMPLES])
        more = f"; and {len(mismatches) - _EXAMPLES} more" if len(mismatches) > _EXAMPLES else ""
        reason = (f"{REQUIREMENT_SERIES_WRONG}: {len(mismatches)} of {n} {paradigm.name} Main "
                  f"Activations met a Requirement its series does not give "
                  f"({_series_words(paradigm)}): {shown}{more}.  {LIKELY_CAUSE[0].upper()}"
                  f"{LIKELY_CAUSE[1:]}")
    note = ""
    if shifted:
        note = (f"{paradigm.name}: in {shifted} state run(s) the Activation counted on the run's "
                f"first row is taken as the previous Paradigm's, so the series is matched from "
                f"the next one")
    return CheckResult(REQUIREMENT_SERIES_WRONG, True, value=len(mismatches), judged=n,
                       not_judged=int((main & ~comparable).sum()), fired=bool(mismatches),
                       reason=reason, note=note)


# ---------------------------------------------------------------------------
# Non-feeding
# ---------------------------------------------------------------------------

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[start, end)`` of each run of True in *mask*."""
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def expected_lit_ms(activity: np.ndarray, index: np.ndarray, eligible: np.ndarray,
                    starts: np.ndarray, paradigm: op.Paradigm, *, tol_ms: float,
                    record_ms: int = RECORD_MS) -> np.ndarray:
    """(n,) the milliseconds of each row a NonFeeding *paradigm*'s rule says
    the well was lit, for certain, in its *eligible* rows.

    The rule (``LEDControl.c``, ``StepNonFeeding``): dark during contact,
    lit ``OnDelayMs`` after it ends, for at most ``MaxOnMs`` (0: until the
    next contact), and lit at once when the Paradigm starts (*starts*: the
    rows where its state started afresh).  pyflic's activity is per record,
    and the record where a bout ends may read below pyflic's threshold
    while the firmware still has contact, so contact may have ended
    anywhere from the start of the last activity row to the end of the row
    after it: what is lit for certain runs from that end plus ``OnDelayMs``
    + tol, and, with ``MaxOnMs``, to the row's start plus ``OnDelayMs`` +
    ``MaxOnMs`` − tol.  From a Paradigm start, from the end of its first
    row (the instruction applies within it) + tol to its start plus
    ``MaxOnMs`` − tol.  Elsewhere (an Interval that keeps the state, the
    first row after a gap in ``Index``) nothing is expected until the next
    activity."""
    activity = np.asarray(activity, dtype=bool)
    n = activity.size
    P = float(record_ms)
    t = (np.asarray(index, dtype=float) - 1.0) * P
    delay = float(paradigm.on_delay_ms or 0)
    max_on = float(paradigm.max_on_ms or 0)
    tol = float(tol_ms)
    out = np.zeros(n)
    eligible = np.asarray(eligible, dtype=bool).copy()
    gap = np.zeros(n, dtype=bool)
    gap[1:] = np.diff(np.asarray(index, dtype=np.int64)) != 1
    for a0, b in _runs(eligible):
        ## A break in Index starts a segment whose state is not known.
        cuts = [a0, *(np.flatnonzero(gap[a0 + 1:b]) + a0 + 1).tolist(), b]
        for a, e in zip(cuts[:-1], cuts[1:]):
            seg_t = t[a:e]
            act = activity[a:e]
            last_end = np.maximum.accumulate(np.where(act, seg_t + P, -np.inf))
            known = np.isfinite(last_end)
            lo = last_end + P + delay + tol
            hi = (last_end - P + delay + max_on - tol) if max_on > 0 else np.full(e - a, np.inf)
            if a == a0 and bool(np.asarray(starts, dtype=bool)[a]):
                ## Lit at once when the Paradigm starts, until the first contact.
                lo = np.where(known, lo, seg_t[0] + P + tol)
                hi = np.where(known, hi, (seg_t[0] + max_on - tol) if max_on > 0 else np.inf)
                known = np.ones(e - a, dtype=bool)
            row_lo = np.maximum(seg_t, lo)
            row_hi = np.minimum(seg_t + P, hi)
            out[a:e] = np.where(known, np.clip(row_hi - row_lo, 0.0, P), 0.0)
    return out


def non_feeding_dark(on_ms: Mapping[int, np.ndarray], activity: Mapping[int, np.ndarray],
                     index: np.ndarray, eligible: Mapping[int, np.ndarray], starts: np.ndarray,
                     paradigm: op.Paradigm, *, tol_ms: float, warn_fraction: float,
                     record_ms: int = RECORD_MS) -> CheckResult:
    """*non-feeding dark*: the time a NonFeeding Trigger Well's ``OnMs`` is
    0 although *paradigm*'s rule says lit (:func:`expected_lit_ms`), in
    seconds, over the group's Trigger Wells.

    *on_ms*, *activity* and *eligible* map each Trigger Well to its ``OnMs``,
    its pyflic activity and the rows where it ran *paradigm* as a Trigger
    Well; *starts* marks the rows where the Paradigm state started afresh.
    Warns when the dark time is above *warn_fraction*
    (``opto_non_feeding_dark_warn_fraction``) of the time the rule says
    lit, once that is :data:`MIN_EXPECTED_LIT_SEC` at least."""
    if paradigm.type != op.NON_FEEDING:
        return not_applicable(NON_FEEDING_DARK)
    expected_ms = dark_ms = 0.0
    for well, mask in eligible.items():
        expected = expected_lit_ms(activity[well], index, mask, starts, paradigm,
                                   tol_ms=tol_ms, record_ms=record_ms)
        expected_ms += float(expected.sum())
        dark_ms += float((expected * (np.asarray(on_ms[well]) == 0)).sum())
    dark, expected = dark_ms / 1000.0, expected_ms / 1000.0
    fired = expected >= MIN_EXPECTED_LIT_SEC and dark > warn_fraction * expected
    reason = ""
    if fired:
        max_on = int(paradigm.max_on_ms or 0)
        window = f" and within its MaxOnMs of {max_on} ms after that" if max_on else ""
        reason = (f"{NON_FEEDING_DARK}: dark for {dark:.0f} s of the {expected:.0f} s "
                  f"{paradigm.name}'s rule says lit ({_pct(dark, expected)}): no pyflic "
                  f"activity at {op.wells_label(sorted(eligible))} for longer than its "
                  f"OnDelayMs of {int(paradigm.on_delay_ms or 0)} ms plus {tol_ms:g} ms "
                  f"(opto_activation_tolerance_ms){window}; "
                  f"opto_non_feeding_dark_warn_fraction is {warn_fraction:g}.  Contact the "
                  f"firmware read that pyflic did not (a drifting baseline), or a longer "
                  f"OnDelayMs than Program.txt states")
    return CheckResult(NON_FEEDING_DARK, True, value=dark, judged=expected, fired=fired,
                       reason=reason)


def relight_delays(rows: pd.DataFrame, activity: Mapping[int, np.ndarray], index: np.ndarray,
                   *, record_ms: int = RECORD_MS) -> np.ndarray:
    """(len(rows),) each NonFeeding Activation's relight delay in ms, ``NaN``
    where it has none: its ``OnsetMs`` minus the end of the last pyflic
    activity sample at its Well before it, for the Activations that follow
    activity (some from the Well's previous onset row on) and whose onset is
    readable (not ``Overlap``).  Rule figures draw these; ``RelightDelayMs``
    is their median.

    *rows* are ledger rows, *activity* maps each of their Wells to its
    activity per row and *index* is the rows' ``Index``.  An activity sample
    is good to a record, so a delay can read up to a record short of
    ``OnDelayMs``."""
    P = float(record_ms)
    t = (np.asarray(index, dtype=float) - 1.0) * P
    wells = rows["Well"].to_numpy(dtype=np.int64)
    row = rows["Row"].to_numpy(dtype=np.int64)
    onset = _numbers(rows, "OnsetMs")
    overlap = _overlap(rows)
    out = np.full(len(rows), np.nan)
    for well in np.unique(wells).tolist():
        if well not in activity:
            continue
        acts = np.flatnonzero(np.asarray(activity[well], dtype=bool))
        mine = np.flatnonzero(wells == well)
        previous = -1
        for i in mine.tolist():
            since, previous = previous, int(row[i])
            if overlap[i] or not acts.size:
                continue
            ## Activity samples that start before the onset, up to its row.
            cand = acts[:np.searchsorted(acts, row[i], side="right")]
            cand = cand[t[cand] < onset[i]]
            if not cand.size or cand[-1] < max(since, 0):
                continue
            out[i] = onset[i] - (t[cand[-1]] + P)
    return out


def relight_delay(delays: np.ndarray) -> CheckResult:
    """``RelightDelayMs``: the median of :func:`relight_delays`, reported
    against ``OnDelayMs``; no flag."""
    delays = np.asarray(delays, dtype=float)
    delays = delays[np.isfinite(delays)]
    value = float(np.median(delays)) if delays.size else float("nan")
    return CheckResult(RELIGHT_DELAY, True, value=value, judged=int(delays.size))


# ---------------------------------------------------------------------------
# Open loop, every family: the windows, the Interval starts, restarts
# ---------------------------------------------------------------------------

def open_loop_not_lit(fractions: Sequence[tuple[str, float]], *,
                      min_fraction: float) -> CheckResult | None:
    """*open loop not lit*: the lowest lit fraction over a group's open-loop
    windows (*fractions*: ``(window words, fraction)``), so one dark pass of
    a Repeating program shows.  Warns below *min_fraction*
    (``opto_open_loop_min_lit_fraction``).  ``None`` without a window."""
    known = [(w, f) for w, f in fractions if f is not None and np.isfinite(f)]
    if not known:
        return None
    where, lowest = min(known, key=lambda x: x[1])
    fired = lowest < min_fraction
    reason = ""
    if fired:
        reason = (f"open loop not lit: lit for {100 * lowest:.0f}% of {where}, the lowest of "
                  f"its {len(known)} open-loop window(s); opto_open_loop_min_lit_fraction is "
                  f"{min_fraction:g}")
    return CheckResult("open loop not lit", True, value=lowest, judged=len(known),
                       fired=fired, reason=reason)


@dataclass(frozen=True)
class IntervalLag:
    """One Program Interval's start: Interval *program_interval* (from 1)
    in Cycle *cycle* began on row *row* (minute *minute*, clock *started*),
    *lag_sec* after its ``Program.txt`` time *scheduled* (past
    :data:`LEAD_IN_SEC` for the DFM's first, *first*).  *after_restart*:
    the row is a restart's resume row.  *joined*: the recording began after
    that time, so *scheduled* is the recording's start instead."""

    program_interval: int
    cycle: int
    row: int
    minute: float
    scheduled: datetime
    started: datetime
    lag_sec: float
    first: bool = False
    after_restart: bool = False
    joined: bool = False


def _elapsed_starts(section: op.DFMProgram) -> dict[int, timedelta]:
    """Each printed Interval's start from the program's start, as PyMCU's
    ``InstructionSet`` lays them out: the durations before it, added."""
    out: dict[int, timedelta] = {}
    t = timedelta(0)
    for iv in sorted(section.intervals, key=lambda iv: iv.index):
        out[int(iv.index)] = t
        t += timedelta(minutes=float(iv.duration_min))
    return out


def scheduled_start(section: op.DFMProgram, start: datetime, program_interval: int,
                    cycle: int) -> datetime | None:
    """When ``Program.txt`` has *section*'s Interval *program_interval* begin
    in Cycle *cycle*, as PyMCU's ``InstructionSet.GetPosition`` runs it:
    Linear and Constant from the Start Time; Repeating a cycle's length
    later each pass; Circadian on the time of day, Cycle n being the n-th
    calendar day.  ``None`` for an Interval the file does not print, or
    one a Circadian program was already in when it started."""
    elapsed = _elapsed_starts(section)
    k = int(program_interval)
    if k not in elapsed:
        return None
    kind = section.program_type.strip().lower()
    c = max(int(cycle or 1), 1)
    if kind == "circadian":
        midnight = datetime.combine(start.date(), datetime.min.time(), tzinfo=start.tzinfo)
        when = midnight + timedelta(days=c - 1) + elapsed[k]
        return when if when >= start else None
    if kind == "repeating":
        total = sum((timedelta(minutes=float(iv.duration_min)) for iv in section.intervals),
                    timedelta(0))
        return start + total * (c - 1) + elapsed[k]
    if kind == "constant":
        return start if k == 1 and c == 1 else None
    return start + elapsed[k] if c == 1 else None


def interval_lags(state: ParadigmState, section: op.DFMProgram, start: datetime,
                  origin: datetime, *, lead_in_sec: float = LEAD_IN_SEC) -> list[IntervalLag]:
    """Each Program Interval start the rows show, on every Cycle, with its
    lag behind ``Program.txt``: the first row of each (Interval, Cycle) run
    (a later run of the same pair, after a restart or unknown rows, is not a
    start), its clock time (*origin* plus its ``Minutes``: the row's stamp,
    at its record's end) less :func:`scheduled_start`.  The firmware's own
    lag is one record.  The DFM's first Interval follows PyMCU's lead-in,
    so its lag is counted past *lead_in_sec* and never below 0.  PyMCU runs
    a Program on the clock (``InstructionSet.GetPosition``), so a recording
    started after an Interval's time joins it partway: that first Interval
    is due when the data begin (*origin*), not at its ``Program.txt`` time."""
    seen: set[tuple[int, int]] = set()
    resumes = {s.resume_row for s in state.restarts()}
    minutes = np.asarray(state.minutes, dtype=float)
    out: list[IntervalLag] = []
    first = True
    for run in state.interval_runs():
        if run.kind != "program":
            continue
        key = (int(run.program_interval), int(run.cycle or 1))
        is_first, first = first, False
        if key in seen:
            continue
        seen.add(key)
        if run.i0 == 0:
            ## The rows begin inside it (a range filter): its start is not in them.
            continue
        when = scheduled_start(section, start, *key)
        if when is None:
            continue
        joined = is_first and when < origin
        if joined:
            when = origin
        started = origin + timedelta(minutes=float(minutes[run.i0]))
        lag = (started - when).total_seconds()
        if is_first:
            lag = max(0.0, lag - float(lead_in_sec))
        out.append(IntervalLag(program_interval=key[0], cycle=key[1], row=int(run.i0),
                               minute=float(minutes[run.i0]), scheduled=when, started=started,
                               lag_sec=lag, first=is_first, after_restart=run.i0 in resumes,
                               joined=joined))
    return out


def interval_started_late(lags: Sequence[IntervalLag], *, warn_sec: float) -> CheckResult:
    """*interval started late*: the worst lag (``IntervalLagSec``), warning
    above *warn_sec* (``opto_interval_lag_warn_sec``)."""
    if not lags:
        return not_applicable(INTERVAL_STARTED_LATE)
    worst = max(lag.lag_sec for lag in lags)
    late = sorted((g for g in lags if g.lag_sec > warn_sec), key=lambda g: -g.lag_sec)
    reason = ""
    if late:
        shown = "; ".join(
            f"Program Interval {g.program_interval} (Cycle {g.cycle}) began {g.lag_sec:.1f} s "
            + ("past the lead-in after " if g.first else "after ")
            + ("the recording began" if g.joined else "its Program.txt time")
            + f" {g.scheduled:%H:%M:%S}, at minute {g.minute:.1f}"
            + (", where the DFM came back from a restart" if g.after_restart else "")
            for g in late[:_EXAMPLES])
        more = f"; and {len(late) - _EXAMPLES} more" if len(late) > _EXAMPLES else ""
        reason = (f"{INTERVAL_STARTED_LATE}: {shown}{more}; opto_interval_lag_warn_sec is "
                  f"{warn_sec:g}.  The firmware's own lag is one record: the DFM took the "
                  f"instruction late, PyMCU sending it again")
    return CheckResult(INTERVAL_STARTED_LATE, True, value=worst, judged=len(lags),
                       fired=bool(late), reason=reason)


def dfm_restarted(spans: Sequence[RestartSpan], minutes: np.ndarray, *,
                  origin: datetime | None = None,
                  record_ms: int = RECORD_MS) -> CheckResult:
    """*DFM restarted*: each restart the rows show
    (``ParadigmState.restarts``), with its time, the records missing and
    where the Paradigm state started over.  Warns on any."""
    spans = list(spans)
    reason = ""
    if spans:
        minutes = np.asarray(minutes, dtype=float)
        parts = []
        for s in spans:
            clock = (f" ({(origin + timedelta(minutes=s.minute)):%H:%M:%S})"
                     if origin is not None else "")
            idle = s.resume_row - s.row
            resumed = (f"its Paradigm state started over at minute {minutes[s.resume_row]:.1f}"
                       if s.resume_row < minutes.size else "it took no instruction again")
            parts.append(f"at minute {s.minute:.1f}{clock}, {s.missing} record(s) "
                         f"({s.missing * record_ms / 1000.0:.1f} s) missing from Index, then "
                         f"{idle} row(s) under no instruction; {resumed}")
        reason = (f"{DFM_RESTARTED}: {len(spans)} restart(s): " + "; ".join(parts)
                  + ".  Acclimation, the Main-Activation number K and the Requirement series "
                    "start again after a restart")
    return CheckResult(DFM_RESTARTED, True, value=len(spans), judged=len(spans),
                       fired=bool(spans), reason=reason)


def gaps_outside_restarts(state: ParadigmState, spans: Sequence[RestartSpan] | None = None
                          ) -> tuple[list[tuple[int, int, float]], int]:
    """``(the Index gaps no restart explains, records the restarts lost)``:
    ``state.index_gaps()`` less the gap right before each restart's first
    row, which *DFM restarted* reports, not *lost records*."""
    spans = state.restarts() if spans is None else list(spans)
    restart_after = {int(state.index[s.row - 1]) for s in spans if s.row > 0}
    gaps = state.index_gaps()
    outside = [g for g in gaps if g[0] not in restart_after]
    lost_to_restarts = int(sum(g[1] for g in gaps if g[0] in restart_after))
    return outside, lost_to_restarts


# ---------------------------------------------------------------------------
# EndedBy
# ---------------------------------------------------------------------------

def ended_by(rows: pd.DataFrame, paradigm: op.Paradigm | None, *,
             activity: Mapping[int, np.ndarray], index: np.ndarray, tol_ms: float,
             record_ms: int = RECORD_MS) -> np.ndarray:
    """(len(rows),) object: what ended each Activation (:data:`ENDED_BY_VALUES`),
    for ``opto_light_events.csv``.

    * ``interval``, ``gap`` or ``end``: its ``Boundary``;
    * ``unknown``: ``Overlap``, or no Paradigm known (no program);
    * closed loop (ClosedLoop Main, any Acclimation): ``MaxOnMs`` when its
      light lasted ``MaxOnMs`` within tol, else ``Hold``;
    * fixed length (FixedInterval and the ratio types in Main): ``length``
      when it lasted ``ActivationMs`` within tol, else ``unknown``;
    * non-feeding: ``MaxOnMs`` when it lasted ``MaxOnMs`` within tol,
      ``contact`` when pyflic activity at its Well (*activity* maps each
      Well to its activity per row) is there within tol of ``EndMs``,
      widened by a record each way, as pyflic's activity is per record;
      else ``unknown``.

    tol widens by a record where ``Timing`` is ``record``."""
    m = len(rows)
    out = np.full(m, ENDED_UNKNOWN, dtype=object)
    if not m:
        return out
    boundary = rows["Boundary"].astype(object).where(rows["Boundary"].notna(), BOUNDARY_NONE) \
        .to_numpy()
    overlap = _overlap(rows)
    lit = _numbers(rows, "LitMs")
    end = _numbers(rows, "EndMs")
    phase = _phase(rows)
    tol = tolerances(rows, tol_ms, record_ms)
    rule = (boundary == BOUNDARY_NONE) & ~overlap
    if paradigm is not None:
        max_on = float(paradigm.max_on_ms or 0)
        if paradigm.type == op.CLOSED_LOOP or paradigm.type in op.ACCLIMATION_PARADIGMS:
            closed = rule & ((phase == "A") | (paradigm.type == op.CLOSED_LOOP))
            cut = closed & (phase == "M") & (max_on > 0) & (np.abs(lit - max_on) <= tol)
            out[closed] = ENDED_HOLD
            out[cut] = ENDED_MAX_ON
        if paradigm.type in (op.FIXED_INTERVAL, *op.RATIO_PARADIGMS) \
                and paradigm.activation_ms is not None:
            fixed = rule & (phase == "M")
            out[fixed & (np.abs(lit - float(paradigm.activation_ms)) <= tol)] = ENDED_LENGTH
        if paradigm.type == op.NON_FEEDING:
            P = float(record_ms)
            t = (np.asarray(index, dtype=float) - 1.0) * P
            wells = rows["Well"].to_numpy(dtype=np.int64)
            ## Each Well's activity samples, by their start in ms.
            starts_ms = {int(w): t[np.flatnonzero(np.asarray(a, dtype=bool))]
                         for w, a in activity.items()}
            for i in np.flatnonzero(rule).tolist():
                if max_on > 0 and abs(lit[i] - max_on) <= tol[i]:
                    out[i] = ENDED_MAX_ON
                    continue
                ta = starts_ms.get(int(wells[i]))
                if ta is None:
                    continue
                ## An activity sample [t, t + P) reaching within tol and a
                ## record of EndMs.
                lo, hi = end[i] - tol[i] - P, end[i] + tol[i] + P
                near = ta[np.searchsorted(ta, lo - P, side="left"):
                          np.searchsorted(ta, hi, side="right")]
                if ((near + P >= lo) & (near <= hi)).any():
                    out[i] = ENDED_CONTACT
    out[boundary != BOUNDARY_NONE] = boundary[boundary != BOUNDARY_NONE]
    return out
