"""Optogenetic light QC: was the light where the licks were?

The DFM firmware switches an Optolid LED from its own reading of a Trigger
Well, every millisecond, during the run.  pyflic counts licks afterwards, from
the *baselined* signal.  The two should agree, and where they do not the light
is evidence about the sensor rather than the fly: a well whose signal drifts
up is "touched" continuously for the firmware and flat for pyflic, so the light
runs with no lick behind it.  The firmware's decay keeps an LED lit for a set
time after contact ends, so a short tail of light after the last lick is
normal; long stretches of light with no lick are not.

This module holds the arithmetic on plain arrays, for any Experiment Type; the
experiment assembles it per DFM (:mod:`pyflic.base.opto_light`).  Terms, fixed
in ``CONTEXT.md``:

* **Explained Light** is a lit sample with activity — a feeding lick or a
  tasting sample — in one of its Linkage Group's Trigger Wells, from the
  interval's decay plus a tolerance before it to the tolerance after it;
* the **Emulated Trigger** is the firmware's own test re-run on the recorded
  signal: raw minus the mean of the run's first ten seconds, against the
  interval's threshold.  Contact the Emulated Trigger sees and pyflic does not
  is the drifting-baseline signature.

The unit of judgement is the Linkage Group: its light is one circuit, lit when
any member is triggered.  Verdicts are broadcast to every chamber the group
touches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Mapping

import numpy as np
import pandas as pd

from . import opto_program as op
from .constant_fields import ConstantField
from .constant_fields import constant_problems as _constant_problems
from .pr_light_qc import VERDICT_FAILED, VERDICT_OK, VERDICT_WARNING, light_events

#: Design ``constants:`` read by the optogenetic light QC, with their defaults.
#: Shared by every Experiment Type and never written into a new config: the
#: editors show them as placeholders (ADR-0011) and the QC merges them under
#: whatever the design states.
DEFAULT_CONSTANTS: dict[str, Any] = {
    "exclude_failed_opto_chambers": False,
    "opto_unexplained_warn_fraction": 0.10,
    "opto_unexplained_fail_fraction": 0.30,
    "opto_unexplained_min_sec": 30,
    "opto_default_decay_ms": 1000,
    "opto_decay_tolerance_samples": 2,
    "opto_unlit_feeding_fraction": 0.50,
    "opto_open_loop_min_lit_fraction": 0.95,
}

#: The editors' and the validation's description of those constants.
OPTO_CONSTANT_FIELDS: tuple[ConstantField, ...] = (
    ConstantField(
        "exclude_failed_opto_chambers", "Exclude chambers that fail the opto light QC",
        "exclude_failed_opto_chambers — every chamber of a linkage group whose light "
        "ran without the licks to explain it leaves the analysis through auto-removal.  "
        "Off (the default) keeps them and flags them: a faulty light does not "
        "invalidate the feeding record.  qc/opto/opto_light_qc.csv lists them either "
        "way.",
        "switch", short_label="Opto light QC failed",
        choices=("exclude the chambers", "keep and flag")),
    ConstantField(
        "opto_unexplained_warn_fraction", "Unexplained light, warning (fraction)",
        "opto_unexplained_warn_fraction — a linkage group whose lit time is at least "
        "this fraction unexplained (no trigger-well lick or touch within the decay "
        "window) gets a warning.  Default 0.10.",
        "opto", minimum=0, maximum=1),
    ConstantField(
        "opto_unexplained_fail_fraction", "Unexplained light, failure (fraction)",
        "opto_unexplained_fail_fraction — at least this fraction of a linkage group's "
        "lit time unexplained fails the group.  Default 0.30.",
        "opto", minimum=0, maximum=1),
    ConstantField(
        "opto_unexplained_min_sec", "Unexplained light at least (s)",
        "opto_unexplained_min_sec — neither fraction is judged until a group has at "
        "least this many seconds of unexplained light, so a few stray samples in a "
        "run lit for seconds are not called a fault.  Default 30; 0 judges any.",
        "opto", minimum=0),
    ConstantField(
        "opto_default_decay_ms", "Decay without Program.txt (ms)",
        "opto_default_decay_ms — the light decay assumed when data/ holds no "
        "Program.txt stating it.  Default 1000.",
        "opto", integer=True, minimum=0),
    ConstantField(
        "opto_decay_tolerance_samples", "Decay tolerance (samples)",
        "opto_decay_tolerance_samples — light within the decay plus this many "
        "samples of a lick still counts as explained; it absorbs the difference "
        "between the firmware's 1 ms decisions and the recorded samples.  Default 2.",
        "opto", integer=True, minimum=0),
    ConstantField(
        "opto_unlit_feeding_fraction", "Unlit feeding, warning (fraction)",
        "opto_unlit_feeding_fraction — in a closed-loop interval every feeding bout "
        "at a trigger well should light its group; when at least this fraction do "
        "not, the group gets a warning (a dead LED, a loose lid, or a threshold the "
        "fly never crossed).  Default 0.50.",
        "opto", minimum=0, maximum=1, minimum_exclusive=True),
    ConstantField(
        "opto_open_loop_min_lit_fraction", "Open loop lit at least (fraction)",
        "opto_open_loop_min_lit_fraction — an open-loop interval (threshold 0) should "
        "be lit throughout; lit for less than this fraction of it is a warning.  "
        "Default 0.95.",
        "opto", minimum=0, maximum=1),
)

# ---------------------------------------------------------------------------
# Flags and verdicts
# ---------------------------------------------------------------------------

UNEXPLAINED = "unexplained light"
PARTLY_UNEXPLAINED = "partly unexplained light"
UNEXPLAINED_NO_PROGRAM = "unexplained light (no Program.txt)"
LIGHT_WHILE_OFF = "light while off"
NO_LIGHT = "no light recorded"
NO_LIGHT_EVENTS = "no light events"
UNLIT_FEEDING = "unlit feeding"
OPEN_LOOP_DARK = "open loop not lit"
PROGRAM_MISMATCH = "program mismatch"
NO_PROGRAM_SECTION = "no program section"

#: Flags that fail a Linkage Group.  The rest are warnings: they point at the
#: setup or the program rather than proving the light was wrong.
FAILING_FLAGS: tuple[str, ...] = (UNEXPLAINED, LIGHT_WHILE_OFF, NO_LIGHT)
WARNING_FLAGS: tuple[str, ...] = (PARTLY_UNEXPLAINED, UNEXPLAINED_NO_PROGRAM,
                                  NO_LIGHT_EVENTS, UNLIT_FEEDING, OPEN_LOOP_DARK,
                                  PROGRAM_MISMATCH, NO_PROGRAM_SECTION)

#: Where a DFM's program came from.
PROGRAM_OK = "Program.txt"
PROGRAM_NO_FILE = "no Program.txt"
PROGRAM_NO_SECTION = "no section"
PROGRAM_DROPPED = "section unreadable"

#: Minutes per window of the unexplained-light onset (as the PR Resting Level).
WINDOW_MIN = 30
#: Feeding bouts a closed-loop group needs before its unlit share is judged.
MIN_FEEDING_EVENTS = 5
#: The firmware's baseline: the mean of the run's first ten seconds.
BASELINE_SECONDS = 10.0
#: A data start this far from the program's Start Time is a mismatch.
START_TOLERANCE_SEC = 60.0
#: Data starting later than this after the program leaves the firmware's
#: baseline window unrecorded, so the Emulated Trigger is approximate.
BASELINE_ALIGN_SEC = 2.0

CAUSE_DRIFT = "signal above the firmware threshold with no pyflic activity: a " \
              "drifting baseline or sustained contact"
CAUSE_HARDWARE = "light without firmware-visible contact: hardware, linkage or " \
                 "program mismatch"

#: ``qc/opto/opto_light_qc.csv`` — one row per Linkage Group per DFM.
OPTO_QC_COLUMNS: tuple[str, ...] = (
    "DFM", "Group", "Linkage", "Wells", "TriggerWells", "Chambers", "Treatment",
    "Program", "Paradigm", "LitSec", "JudgedLitSec", "UnexplainedSec",
    "UnexplainedFraction", "UnexplainedOnsetMin", "LightEvents", "UnexplainedEvents",
    "FeedingEvents", "ClosedLoopFeedingEvents", "UnlitFeedingEvents",
    "OpenLoopLitFraction", "LitWhileOffSec",
    "ContactSec", "ContactWithoutActivitySec", "UnexplainedWithContactSec",
    "EmulationApproximate", "LikelyCause", "Flags", "Verdict", "Excluded", "Notes",
)
#: ``qc/opto/opto_light_intervals.csv`` — one row per group per scheduled interval.
OPTO_INTERVAL_COLUMNS: tuple[str, ...] = (
    "DFM", "Group", "Window", "Interval", "Occurrence", "StartMin", "EndMin", "Mode",
    "DecayMs", "TriggerWells", "Samples", "LitSec", "LitFraction", "UnexplainedSec",
    "ContactSec", "ContactWithoutActivitySec", "FeedingEvents", "UnlitFeedingEvents",
)
#: ``qc/opto/opto_light_events.csv`` — one row per light event.
OPTO_EVENT_COLUMNS: tuple[str, ...] = (
    "DFM", "Group", "Event", "OnsetMin", "EndMin", "DurationSec", "Interval", "Mode",
    "UnexplainedSec", "Explained", "OverrunSec",
)
#: ``qc/opto/opto_program.csv`` — one row per DFM interval of the program.
#: ``F`` and ``P`` are the values the MCU echoed; ``FrequencyHz``,
#: ``AcclimationEvents``, ``PulseWidthMs`` and ``NonFeeding`` decode them.
OPTO_PROGRAM_COLUMNS: tuple[str, ...] = (
    "DFM", "Interval", "Start", "DurationMin", "ProgramType", "Paradigm", "Dark",
    "F", "FrequencyHz", "AcclimationEvents", "P", "PulseWidthMs", "NonFeeding",
    "DecayMs", "Delay", "MaxTimeOn", "Linkage", "TriggerWells", "OpenLoopWells",
)


def _truthy(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(value)


@dataclass(frozen=True, slots=True)
class OptoQCSettings:
    """The optogenetic light QC's thresholds, from the design's ``constants:``."""

    exclude: bool = False
    warn_fraction: float = 0.10
    fail_fraction: float = 0.30
    min_unexplained_sec: float = 30.0
    default_decay_ms: int = 1000
    tolerance_samples: int = 2
    unlit_fraction: float = 0.50
    open_loop_min_fraction: float = 0.95

    @classmethod
    def from_constants(cls, constants: Mapping[str, Any] | None) -> OptoQCSettings:
        merged = {**DEFAULT_CONSTANTS, **dict(constants or {})}

        def number(key: str) -> float:
            try:
                value = float(merged[key])
            except (TypeError, ValueError):
                return float(DEFAULT_CONSTANTS[key])
            return value if math.isfinite(value) else float(DEFAULT_CONSTANTS[key])

        return cls(
            exclude=_truthy(merged["exclude_failed_opto_chambers"]),
            warn_fraction=min(max(number("opto_unexplained_warn_fraction"), 0.0), 1.0),
            fail_fraction=min(max(number("opto_unexplained_fail_fraction"), 0.0), 1.0),
            min_unexplained_sec=max(0.0, number("opto_unexplained_min_sec")),
            default_decay_ms=max(0, int(number("opto_default_decay_ms"))),
            tolerance_samples=max(0, int(number("opto_decay_tolerance_samples"))),
            unlit_fraction=min(max(number("opto_unlit_feeding_fraction"), 0.0), 1.0),
            open_loop_min_fraction=min(max(number("opto_open_loop_min_lit_fraction"),
                                           0.0), 1.0),
        )

    def describe(self) -> str:
        return (f"exclude_failed_opto_chambers={'true' if self.exclude else 'false'}, "
                f"opto_unexplained_warn_fraction={self.warn_fraction:g}, "
                f"opto_unexplained_fail_fraction={self.fail_fraction:g}, "
                f"opto_unexplained_min_sec={self.min_unexplained_sec:g}, "
                f"opto_default_decay_ms={self.default_decay_ms}, "
                f"opto_decay_tolerance_samples={self.tolerance_samples}, "
                f"opto_unlit_feeding_fraction={self.unlit_fraction:g}, "
                f"opto_open_loop_min_lit_fraction={self.open_loop_min_fraction:g}")


def opto_constant_problems(constants: Mapping[str, Any] | None) -> list[str]:
    """What is wrong with the optogenetic light QC constants a ``constants:``
    block states (:data:`OPTO_CONSTANT_FIELDS`), plus a warning fraction above
    the failure one.  Never raises."""
    problems = _constant_problems(OPTO_CONSTANT_FIELDS, constants)
    stated = dict(constants or {})
    warn = stated.get("opto_unexplained_warn_fraction")
    fail = stated.get("opto_unexplained_fail_fraction")
    numeric = (int, float)
    if isinstance(warn, numeric) and isinstance(fail, numeric) \
            and not isinstance(warn, bool) and not isinstance(fail, bool) \
            and float(warn) > float(fail):
        problems.append("'constants.opto_unexplained_warn_fraction' must not exceed "
                        "'constants.opto_unexplained_fail_fraction'")
    return problems


def verdict_of(flags) -> str:
    """``ok``, ``warning`` or ``failed`` for a list of flags."""
    flags = [f for f in flags if f]
    if any(f in FAILING_FLAGS for f in flags):
        return VERDICT_FAILED
    return VERDICT_WARNING if flags else VERDICT_OK


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------

def near(mask: np.ndarray, before: int, after: int) -> np.ndarray:
    """``out[t]``: whether *mask* is true anywhere in ``[t - before, t + after]``."""
    m = np.asarray(mask, dtype=bool)
    n = m.size
    if n == 0:
        return m.copy()
    cum = np.concatenate([[0], np.cumsum(m, dtype=np.int64)])
    idx = np.arange(n)
    lo = np.clip(idx - int(before), 0, n)
    hi = np.clip(idx + int(after) + 1, 0, n)
    return (cum[hi] - cum[lo]) > 0


def explained_light(lit: np.ndarray, activity: np.ndarray, decay_samples: int,
                    tolerance: int) -> np.ndarray:
    """Lit samples with *activity* from ``decay + tolerance`` samples before to
    *tolerance* samples after them — Explained Light."""
    lit = np.asarray(lit, dtype=bool)
    return lit & near(activity, int(decay_samples) + int(tolerance), int(tolerance))


def decay_samples_for(decay_ms: float, samples_per_second: float) -> int:
    """The decay in whole samples, rounded up: a 500 ms decay at 5 Hz can keep
    the LED lit into a third sample."""
    return int(math.ceil(max(0.0, float(decay_ms)) * float(samples_per_second) / 1000.0
                         - 1e-9))


def event_spans(column: np.ndarray) -> list[tuple[int, int]]:
    """``[(start, end)]`` sample spans of pyflic's events in one ``event_df``
    column, which holds each event's length at its first sample."""
    col = np.asarray(column)
    starts = np.flatnonzero(col > 0)
    return [(int(s), int(s + col[s])) for s in starts]


def firmware_baseline(raw: np.ndarray, start: int, samples: int) -> float:
    """The mean of *samples* raw values from *start* — the firmware's baseline."""
    raw = np.asarray(raw, dtype=float)
    seg = raw[max(0, int(start)):max(0, int(start)) + max(1, int(samples))]
    seg = seg[np.isfinite(seg)]
    return float(seg.mean()) if seg.size else 0.0


def onset_minute(minutes: np.ndarray, lit: np.ndarray, unexplained: np.ndarray,
                 fraction: float, window_min: float = WINDOW_MIN) -> float:
    """Start minute of the first *window_min*-minute window of the recording in
    which at least *fraction* of the lit samples are unexplained; NaN when none."""
    lit = np.asarray(lit, dtype=bool)
    if not lit.any():
        return float("nan")
    mins = np.asarray(minutes, dtype=float)
    bins = np.floor(mins / float(window_min)).astype(int)
    lit_bins = bins[lit]
    unexplained_bins = bins[np.asarray(unexplained, dtype=bool) & lit]
    lit_n = pd.Series(1, index=lit_bins).groupby(level=0).sum()
    un_n = pd.Series(1, index=unexplained_bins).groupby(level=0).sum() \
        .reindex(lit_n.index, fill_value=0)
    share = un_n / lit_n
    hit = share[share >= float(fraction) - 1e-12]
    if hit.empty:
        return float("nan")
    return float(hit.index.min() * float(window_min))


def mode_value(values: np.ndarray):
    """The most common finite value of *values*, or ``None``."""
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if v.empty:
        return None
    return float(v.mode().iloc[0])


# ---------------------------------------------------------------------------
# One DFM
# ---------------------------------------------------------------------------

@dataclass
class DFMInputs:
    """One DFM's arrays, each ``n`` samples long (wells are columns 0..11)."""

    dfm_id: int
    minutes: np.ndarray
    samples_per_second: float
    lights: np.ndarray                   # bool (n, 12): OptoCol1, the LEDs lit
    activity: np.ndarray                 # bool (n, 12): feeding lick or tasting sample
    feeding: np.ndarray                  # int (n, 12): event_df, lengths at starts
    raw: np.ndarray | None = None        # float (n, 12): signal, training flag removed
    chamber_of_well: dict[int, int] = field(default_factory=dict)
    origin: datetime | None = None       # absolute time of recording minute 0
    data_frequency: np.ndarray | None = None
    data_pulse_width: np.ndarray | None = None
    data_dark: np.ndarray | None = None


@dataclass
class DFMOptoResult:
    """The light QC of one DFM."""

    dfm_id: int
    setting: str
    program_status: str
    groups: list[dict]
    intervals: list[dict]
    events: list[dict]
    flags: list[str]
    notes: list[str]
    decay_samples: np.ndarray             # per sample, for the PR light QC
    lit_samples: int = 0
    #: Per Linkage Group, seconds per recording minute: ``Minute, LitSec,
    #: ExplainedSec, UnexplainedSec, LitWhileOffSec, NotJudgedSec,
    #: ContactWithoutActivitySec`` (the QC figure).  Lit time splits four
    #: ways: judged against licks and explained or not, lit while every
    #: threshold was -1, and lit where licks do not decide (open loop,
    #: non-feeding activation, outside the schedule).
    timelines: dict = field(default_factory=dict)


@dataclass
class _Window:
    index: int
    interval: op.ProgramInterval | None
    occurrence: int
    start_min: float
    end_min: float
    i0: int
    i1: int


def _chambers_label(chambers) -> str:
    return ", ".join(str(c) for c in sorted(set(chambers))) or "—"


def _to_minutes(when: datetime, origin: datetime) -> float:
    return (when - origin).total_seconds() / 60.0


def _program_windows(section: op.DFMProgram, program: op.OptoProgram,
                     inp: DFMInputs, origin: datetime) -> list[_Window]:
    minutes = inp.minutes
    n = minutes.size
    anchor = program.start or origin
    data_end = origin + timedelta(minutes=float(minutes[-1]) if n else 0.0,
                                  seconds=1.0 / float(inp.samples_per_second))
    end = program.resolved_end() or data_end
    windows: list[_Window] = []
    for k, sched in enumerate(section.schedule(anchor, end)):
        a = _to_minutes(sched.start, origin)
        b = _to_minutes(sched.end, origin)
        i0 = int(np.searchsorted(minutes, a, side="left"))
        i1 = int(np.searchsorted(minutes, b, side="left"))
        windows.append(_Window(index=k + 1, interval=sched.interval,
                               occurrence=sched.occurrence, start_min=a, end_min=b,
                               i0=i0, i1=i1))
    return windows


def decay_profile(dfm_id: int, minutes: np.ndarray, samples_per_second: float,
                  origin: datetime | None, program: op.OptoProgram | None,
                  settings: OptoQCSettings) -> np.ndarray:
    """Per sample, the light decay in samples: the program's interval decay
    where one of *dfm_id*'s intervals applies, else ``opto_default_decay_ms``."""
    minutes = np.asarray(minutes, dtype=float)
    sps = float(samples_per_second)
    out = np.full(minutes.size, decay_samples_for(settings.default_decay_ms, sps),
                  dtype=int)
    section = program.section(dfm_id) if program is not None else None
    if section is None or minutes.size == 0:
        return out
    anchor = origin or program.start or datetime(2000, 1, 1)
    probe = DFMInputs(dfm_id=int(dfm_id), minutes=minutes, samples_per_second=sps,
                      lights=np.zeros((0, op.N_WELLS), bool),
                      activity=np.zeros((0, op.N_WELLS), bool),
                      feeding=np.zeros((0, op.N_WELLS), int), origin=origin)
    for w in _program_windows(section, program, probe, anchor):
        if w.i1 > w.i0:
            out[w.i0:w.i1] = decay_samples_for(w.interval.params.decay_ms, sps)
    return out


def _inferred_groups(lights: np.ndarray) -> dict[int, tuple[int, ...]]:
    """Wells whose light is identical over the whole recording, grouped — the
    linkage the data shows when no program states it.  Unlit wells are left
    out: there is no light of theirs to explain."""
    groups: dict[bytes, list[int]] = {}
    for w in range(1, op.N_WELLS + 1):
        col = lights[:, w - 1]
        if not col.any():
            continue
        groups.setdefault(np.packbits(col).tobytes(), []).append(w)
    ordered = sorted(groups.values(), key=lambda ws: ws[0])
    return {k + 1: tuple(ws) for k, ws in enumerate(ordered)}


def analyze_dfm(inp: DFMInputs, *, program: op.OptoProgram | None, setting: str,
                settings: OptoQCSettings) -> DFMOptoResult:
    """The optogenetic light QC of one DFM.

    With a program section, each Linkage Group is judged interval by interval
    under that interval's mode: open loop (lit fraction), off (any light is a
    fault), a feeding paradigm (Explained Light, and for closed loop the
    reverse — feeding that never lit the group) or non-feeding activation (not
    judged).  Without one, groups are inferred from wells lit together, every
    member counts as a Trigger Well, the decay is the design's fallback and
    unexplained light can warn but not fail: an open-loop schedule is
    indistinguishable from a stuck light without the program.
    """
    minutes = np.asarray(inp.minutes, dtype=float)
    n = minutes.size
    sps = float(inp.samples_per_second)
    tol = int(settings.tolerance_samples)
    lights = np.asarray(inp.lights, dtype=bool).reshape(n, op.N_WELLS)
    activity = np.asarray(inp.activity, dtype=bool).reshape(n, op.N_WELLS)
    feeding = np.asarray(inp.feeding).reshape(n, op.N_WELLS)
    raw = None if inp.raw is None else np.asarray(inp.raw, dtype=float).reshape(n, op.N_WELLS)
    fallback_decay = decay_samples_for(settings.default_decay_ms, sps)
    decay_by_sample = np.full(n, fallback_decay, dtype=int)
    dfm_flags: list[str] = []
    dfm_notes: list[str] = []

    section = program.section(inp.dfm_id) if program is not None else None
    if program is None:
        status = PROGRAM_NO_FILE
        dfm_notes.append("no Program.txt in data/, so this QC is limited: linkage is "
                         "inferred from wells lit together, every lit well counts as a "
                         f"trigger well, the decay is taken as {settings.default_decay_ms} "
                         "ms (opto_default_decay_ms) and unexplained light can warn "
                         "but not fail")
    elif section is None:
        if inp.dfm_id in program.rejected:
            status = PROGRAM_DROPPED
            dfm_notes.append(f"Program.txt's section for DFM {inp.dfm_id} could not be "
                             f"read ({program.rejected[inp.dfm_id]}), so this DFM is "
                             f"judged as if there were no Program.txt")
        else:
            status = PROGRAM_NO_SECTION
            dfm_notes.append(f"Program.txt has no section for DFM {inp.dfm_id}, so this "
                             f"DFM is judged as if there were no Program.txt")
        dfm_flags.append(NO_PROGRAM_SECTION)
    else:
        status = PROGRAM_OK

    origin = inp.origin
    if section is not None and origin is None:
        origin = program.start or datetime(2000, 1, 1)
        dfm_notes.append("the data carry no date and time, so the recording is taken "
                         "to start at the program's Start Time")

    ## ---- windows and groups ----
    windows: list[_Window]
    if section is not None:
        windows = _program_windows(section, program, inp, origin)
        groups = section.linkage_groups()
        linkage_source = "program"
    else:
        windows = [_Window(index=1, interval=None, occurrence=1,
                           start_min=float(minutes[0]) if n else 0.0,
                           end_min=float(minutes[-1]) if n else 0.0, i0=0, i1=n)]
        groups = _inferred_groups(lights) if n else {}
        linkage_source = "inferred"
    for w in windows:
        if w.interval is not None and w.i1 > w.i0:
            decay_by_sample[w.i0:w.i1] = decay_samples_for(w.interval.params.decay_ms, sps)

    ## ---- the firmware's baseline, for the Emulated Trigger ----
    emulate = section is not None and raw is not None
    approximate = False
    base = np.zeros(op.N_WELLS)
    if emulate:
        n_base = max(1, int(round(BASELINE_SECONDS * sps)))
        start_idx = 0
        if program.start is not None and inp.origin is not None:
            lag = (inp.origin - program.start).total_seconds()
            if lag > BASELINE_ALIGN_SEC:
                approximate = True
                dfm_notes.append(f"the data start {lag / 60.0:.1f} min after the "
                                 f"program, so the firmware's baseline is not in them; "
                                 f"the emulated trigger uses the first "
                                 f"{BASELINE_SECONDS:g} s of data instead (approximate)")
            else:
                start_idx = int(np.searchsorted(minutes, _to_minutes(program.start, origin),
                                                side="left"))
        elif inp.origin is None:
            approximate = True
        if program.baseline:
            base = np.array([firmware_baseline(raw[:, w], start_idx, n_base)
                             for w in range(op.N_WELLS)])

    ## ---- program versus data ----
    if section is not None:
        if program.start is not None and inp.origin is not None:
            lag = (inp.origin - program.start).total_seconds()
            if abs(lag) > START_TOLERANCE_SEC:
                dfm_flags.append(PROGRAM_MISMATCH)
                dfm_notes.append(f"the data start {abs(lag) / 60.0:.1f} min "
                                 f"{'after' if lag > 0 else 'before'} the program's "
                                 f"Start Time")
        end = program.resolved_end()
        if end is not None and n:
            past = (origin + timedelta(minutes=float(minutes[-1])) - end).total_seconds()
            if past > 60.0:
                dfm_notes.append(f"the data run {past / 60.0:.1f} min past the program's "
                                 f"End Time; light after it is not judged")
        seen: set[str] = set()
        for w in windows:
            if w.interval is None or w.i1 <= w.i0:
                continue
            params = w.interval.params
            checks = (
                ("OptoFreq", inp.data_frequency, (params.frequency_hz, params.frequency),
                 f"{params.frequency_hz} Hz"),
                ("OptoPW", inp.data_pulse_width, (params.pulse_width_ms, params.pulse_width),
                 f"{params.pulse_width_ms} ms"),
                ("Dark", inp.data_dark, (1.0 if w.interval.dark else 0.0,),
                 "on" if w.interval.dark else "off"),
            )
            for column, values, allowed, expected in checks:
                if values is None:
                    continue
                observed = mode_value(np.asarray(values)[w.i0:w.i1])
                if observed is None:
                    continue
                if column == "Dark":
                    observed = 1.0 if observed != 0 else 0.0
                if any(abs(observed - float(a)) < 1e-9 for a in allowed):
                    continue
                key = f"{column}:{w.interval.index}"
                if key in seen:
                    continue
                seen.add(key)
                if PROGRAM_MISMATCH not in dfm_flags:
                    dfm_flags.append(PROGRAM_MISMATCH)
                shown = (("on" if observed else "off") if column == "Dark"
                         else f"{observed:g}")
                dfm_notes.append(f"interval {w.interval.index}: the data's {column} is "
                                 f"{shown}, the program says {expected}")

    ## ---- each Linkage Group ----
    group_rows: list[dict] = []
    interval_rows: list[dict] = []
    event_rows: list[dict] = []
    ## Which window each sample falls in (-1: outside the program's schedule).
    window_of = np.full(n, -1, dtype=int)
    for k, w in enumerate(windows):
        window_of[w.i0:w.i1] = k
    covered = window_of >= 0
    near_cache: dict[tuple, np.ndarray] = {}

    def any_activity(trig: tuple[int, ...]) -> np.ndarray:
        if not trig:
            return np.zeros(n, dtype=bool)
        return activity[:, [t - 1 for t in trig]].any(axis=1)

    def activity_near(trig: tuple[int, ...], before: int, after: int) -> np.ndarray:
        key = (trig, before, after)
        if key not in near_cache:
            near_cache[key] = near(any_activity(trig), before, after)
        return near_cache[key]

    expects_light = False
    timelines: dict = {}
    for label, wells in groups.items():
        idx = [w - 1 for w in wells]
        lit = lights[:, idx].any(axis=1)
        flags: list[str] = []
        notes: list[str] = []
        judged = np.zeros(n, dtype=bool)
        unexplained = np.zeros(n, dtype=bool)
        contact_all = np.zeros(n, dtype=bool)
        contact_near_all = np.zeros(n, dtype=bool)
        contact_alone = np.zeros(n, dtype=bool)
        off_mask = np.zeros(n, dtype=bool)
        feeding_events = unlit_events = closed_events = 0
        open_samples = open_lit = 0
        off_lit = 0
        nonfeeding_samples = 0
        contact_without = 0
        paradigms: list[str] = []
        window_modes: list[str] = []
        triggers_seen: set[int] = set()
        partial = lit & ~lights[:, idx].all(axis=1)
        if len(idx) > 1 and partial.sum() > max(tol, 0.01 * lit.sum()):
            notes.append(f"its linked wells were not lit together for "
                         f"{partial.sum() / sps:.0f} s")

        for k, w in enumerate(windows):
            if w.interval is not None:
                mode = w.interval.group_mode(wells)
                trig = w.interval.trigger_wells(wells)
                decay_ms = w.interval.params.decay_ms
            else:
                mode = op.UNKNOWN_PARADIGM
                trig = tuple(wells)
                decay_ms = settings.default_decay_ms
            window_modes.append(mode)
            if w.i1 <= w.i0:
                continue
            decay = decay_samples_for(decay_ms, sps)
            ## Light left over from the interval before is not this one's to
            ## explain: skip its decay at every boundary.
            grace = decay + tol if k > 0 else 0
            s0 = min(w.i0 + grace, w.i1)
            seg = slice(s0, w.i1)
            full = slice(w.i0, w.i1)
            if mode not in paradigms:
                paradigms.append(mode)
            triggers_seen.update(trig)
            row = {"DFM": inp.dfm_id, "Group": label, "Window": w.index,
                   "Interval": w.interval.index if w.interval is not None else np.nan,
                   "Occurrence": w.occurrence, "StartMin": w.start_min,
                   "EndMin": w.end_min, "Mode": mode, "DecayMs": decay_ms,
                   "TriggerWells": op.wells_label(trig), "Samples": int(w.i1 - w.i0),
                   "LitSec": float(lit[full].sum()) / sps,
                   "LitFraction": float(lit[full].mean()),
                   "UnexplainedSec": np.nan, "ContactSec": np.nan,
                   "ContactWithoutActivitySec": np.nan, "FeedingEvents": np.nan,
                   "UnlitFeedingEvents": np.nan}
            if mode == op.OPEN_LOOP:
                open_samples += int(w.i1 - s0)
                open_lit += int(lit[seg].sum())
                expects_light = expects_light or (w.i1 - s0) > 0
            elif mode == op.LIGHTS_OFF:
                off_lit += int(lit[seg].sum())
                off_mask[seg] |= lit[seg]
            elif mode == op.NON_FEEDING:
                nonfeeding_samples += int(w.i1 - w.i0)
            else:
                explained = activity_near(tuple(trig), decay + tol, tol)
                un = lit[seg] & ~explained[seg]
                judged[seg] |= lit[seg]
                unexplained[seg] |= un
                row["UnexplainedSec"] = float(un.sum()) / sps
                closed = mode in op.CLOSED_LOOP_PARADIGMS
                n_feed = n_unlit = 0
                for t in trig:
                    for a, b in event_spans(feeding[full, t - 1]):
                        a, b = a + w.i0, b + w.i0
                        n_feed += 1
                        if closed and not lit[max(0, a - tol):min(n, b + decay + tol)].any():
                            n_unlit += 1
                feeding_events += n_feed
                row["FeedingEvents"] = n_feed
                if closed:
                    closed_events += n_feed
                    unlit_events += n_unlit
                    row["UnlitFeedingEvents"] = n_unlit
                if emulate and trig:
                    contact = np.zeros(w.i1 - w.i0, dtype=bool)
                    for t in trig:
                        thr = float(w.interval.threshold(t))
                        contact |= (raw[full, t - 1] - base[t - 1]) > thr
                    contact_all[full] |= contact
                    act_close = activity_near(tuple(trig), tol, tol)[full]
                    alone = contact & ~act_close
                    contact_alone[full] |= alone
                    without = int(alone.sum())
                    contact_without += without
                    row["ContactSec"] = float(contact.sum()) / sps
                    row["ContactWithoutActivitySec"] = without / sps
            interval_rows.append(row)

        if emulate and contact_all.any():
            ## The light persists for the decay after contact ends.
            contact_near_all = near(contact_all, int(decay_by_sample.max(initial=0)) + tol,
                                    tol)
        lit_total = int(lit.sum())
        judged_lit = int(judged.sum())
        unexplained_n = int(unexplained.sum())
        fraction = unexplained_n / judged_lit if judged_lit else float("nan")
        unexplained_sec = unexplained_n / sps
        onset = float("nan")
        if judged_lit and unexplained_n and unexplained_sec >= settings.min_unexplained_sec:
            if fraction >= settings.fail_fraction:
                flags.append(UNEXPLAINED if section is not None else UNEXPLAINED_NO_PROGRAM)
            elif fraction >= settings.warn_fraction:
                flags.append(PARTLY_UNEXPLAINED)
            if flags:
                onset = onset_minute(minutes, judged, unexplained, settings.warn_fraction)
        elif judged_lit and unexplained_n and fraction >= settings.warn_fraction:
            notes.append(f"{unexplained_sec:.1f} s of unexplained light "
                         f"({100 * fraction:.0f}% of lit time) is under "
                         f"opto_unexplained_min_sec, so not judged")
        unexplained_with_contact = int((unexplained & contact_near_all).sum())
        cause = ""
        if emulate and unexplained_n and any(f in (UNEXPLAINED, PARTLY_UNEXPLAINED)
                                             for f in flags):
            cause = (CAUSE_DRIFT if unexplained_with_contact >= 0.5 * unexplained_n
                     else CAUSE_HARDWARE)
        ## A fired flag says its own numbers (the table has them as columns);
        ## notes are for what no flag says.
        if off_lit > 0:
            flags.append(LIGHT_WHILE_OFF)
        open_fraction = open_lit / open_samples if open_samples else float("nan")
        if open_samples and open_fraction < settings.open_loop_min_fraction:
            flags.append(OPEN_LOOP_DARK)
        if closed_events >= MIN_FEEDING_EVENTS and \
                unlit_events / closed_events >= settings.unlit_fraction:
            flags.append(UNLIT_FEEDING)
        elif 0 < closed_events < MIN_FEEDING_EVENTS and unlit_events:
            notes.append(f"{unlit_events} of {closed_events} closed-loop feeding bouts "
                         f"unlit; {MIN_FEEDING_EVENTS} are needed for a verdict")
        feeding_windows = any(m in op.FEEDING_PARADIGMS for m in paradigms)
        if section is not None and feeding_windows and judged_lit == 0 and feeding_events:
            flags.append(NO_LIGHT_EVENTS)
            expects_light = True
        if nonfeeding_samples:
            notes.append(f"{nonfeeding_samples / sps / 60.0:.0f} min of non-feeding "
                         f"activation, which this QC does not judge")
        ## A sample or two past the program's end is the sample clock, not
        ## news; ten seconds of light outside the schedule is.
        unscheduled = int((lit & ~covered).sum())
        if section is not None and unscheduled >= 10 * sps:
            notes.append(f"{unscheduled / sps:.1f} s of light outside the program's "
                         f"schedule, not judged")

        ## ---- this group's light events ----
        onsets, ends = light_events(lit)
        act = any_activity(tuple(sorted(triggers_seen)))
        ## The last activity sample at or before each index, for the overrun.
        last_act = (np.maximum.accumulate(np.where(act, np.arange(n), -1)) if n
                    else np.zeros(0, dtype=int))
        cum_judged = np.concatenate([[0], np.cumsum(judged, dtype=np.int64)])
        cum_explained = np.concatenate([[0], np.cumsum(judged & ~unexplained,
                                                       dtype=np.int64)])
        cum_unexplained = np.concatenate([[0], np.cumsum(unexplained, dtype=np.int64)])
        unexplained_events = 0
        for e, (a, b) in enumerate(zip(onsets.tolist(), ends.tolist()), start=1):
            k = int(window_of[a])
            mode = window_modes[k] if k >= 0 else "unscheduled"
            interval = (windows[k].interval.index
                        if k >= 0 and windows[k].interval is not None else np.nan)
            judged_event = bool(cum_judged[b] > cum_judged[a])
            un_sec = overrun = np.nan
            explained_flag = None
            if judged_event:
                un_sec = float(cum_unexplained[b] - cum_unexplained[a]) / sps
                explained_flag = bool(cum_explained[b] > cum_explained[a])
                d = int(decay_by_sample[a])
                lo, hi = max(0, a - d - tol), min(n, b + tol)
                last = int(last_act[hi - 1]) if hi > 0 else -1
                if last >= lo:
                    overrun = max(0.0, (b - (last + d + tol + 1)) / sps)
                else:
                    overrun = (b - a) / sps
                if not explained_flag:
                    unexplained_events += 1
            event_rows.append({
                "DFM": inp.dfm_id, "Group": label, "Event": e,
                "OnsetMin": float(minutes[a]),
                "EndMin": float(minutes[b - 1]) + 1.0 / sps / 60.0,
                "DurationSec": (b - a) / sps, "Interval": interval, "Mode": mode,
                "UnexplainedSec": un_sec, "Explained": explained_flag,
                "OverrunSec": overrun,
            })

        if n:
            minute_bin = np.clip(np.floor(minutes), 0, None).astype(int)
            size = int(minute_bin.max()) + 1

            def per_minute(mask: np.ndarray) -> np.ndarray:
                return np.bincount(minute_bin[mask], minlength=size) / sps

            timelines[label] = pd.DataFrame({
                "Minute": np.arange(size, dtype=float),
                "LitSec": per_minute(lit),
                "ExplainedSec": per_minute(judged & ~unexplained),
                "UnexplainedSec": per_minute(unexplained),
                "LitWhileOffSec": per_minute(off_mask),
                "NotJudgedSec": per_minute(lit & ~judged & ~off_mask),
                "ContactWithoutActivitySec": per_minute(contact_alone),
            })

        chambers = sorted({inp.chamber_of_well[w] for w in wells
                           if w in inp.chamber_of_well})
        group_rows.append({
            "DFM": inp.dfm_id, "Group": label,
            "Linkage": label if linkage_source == "program" else np.nan,
            "Wells": op.wells_label(wells),
            "TriggerWells": op.wells_label(triggers_seen),
            "Chambers": _chambers_label(chambers), "_chambers": tuple(chambers),
            "_wells": tuple(wells), "Treatment": "",
            "Program": status, "Paradigm": ", ".join(paradigms),
            "LitSec": lit_total / sps, "JudgedLitSec": judged_lit / sps,
            "UnexplainedSec": unexplained_sec,
            "UnexplainedFraction": fraction, "UnexplainedOnsetMin": onset,
            "LightEvents": int(onsets.size), "UnexplainedEvents": unexplained_events,
            "FeedingEvents": feeding_events,
            "ClosedLoopFeedingEvents": closed_events if closed_events else np.nan,
            "UnlitFeedingEvents": unlit_events if closed_events else np.nan,
            "OpenLoopLitFraction": open_fraction,
            "LitWhileOffSec": off_lit / sps if section is not None else np.nan,
            "ContactSec": contact_all.sum() / sps if emulate else np.nan,
            "ContactWithoutActivitySec": contact_without / sps if emulate else np.nan,
            "UnexplainedWithContactSec": (unexplained_with_contact / sps if emulate
                                          else np.nan),
            "EmulationApproximate": bool(approximate) if emulate else np.nan,
            "LikelyCause": cause, "_flags": flags, "_notes": notes,
            "Excluded": False,
        })

    ## ---- the DFM as a whole ----
    lit_any = int(lights.any(axis=1).sum())
    if setting == op.SETTING_YES and lit_any == 0 and (section is None or expects_light):
        dfm_flags.append(NO_LIGHT)
        dfm_notes.append("optogenetics is 'yes' but no LED was lit on this DFM for the "
                         "whole recording: an unplugged lid, or a program that did not "
                         "load")
    if not group_rows:
        ## Nothing lit and nothing programmed: one row speaks for the DFM, so a
        ## failure has somewhere to be recorded.
        chambers = sorted(set(inp.chamber_of_well.values()))
        group_rows.append({
            "DFM": inp.dfm_id, "Group": 0, "Linkage": np.nan, "Wells": "W1-W12",
            "TriggerWells": "—", "Chambers": _chambers_label(chambers),
            "_chambers": tuple(chambers), "_wells": tuple(range(1, op.N_WELLS + 1)),
            "Treatment": "", "Program": status, "Paradigm": "",
            "LitSec": 0.0, "JudgedLitSec": 0.0, "UnexplainedSec": 0.0,
            "UnexplainedFraction": np.nan, "UnexplainedOnsetMin": np.nan,
            "LightEvents": 0, "UnexplainedEvents": 0, "FeedingEvents": 0,
            "ClosedLoopFeedingEvents": np.nan,
            "UnlitFeedingEvents": np.nan, "OpenLoopLitFraction": np.nan,
            "LitWhileOffSec": np.nan, "ContactSec": np.nan,
            "ContactWithoutActivitySec": np.nan, "UnexplainedWithContactSec": np.nan,
            "EmulationApproximate": np.nan, "LikelyCause": "", "_flags": [],
            "_notes": ["no LED was lit on this DFM"], "Excluded": False,
        })
    for row in group_rows:
        own = [f for f in row.pop("_flags")
               if not (f == NO_LIGHT_EVENTS and NO_LIGHT in dfm_flags)]
        flags = list(dict.fromkeys([*own, *dfm_flags]))
        row["Flags"] = ", ".join(flags)
        row["Verdict"] = verdict_of(flags)
        row["Notes"] = "; ".join(row.pop("_notes"))
    return DFMOptoResult(dfm_id=inp.dfm_id, setting=setting, program_status=status,
                         groups=group_rows, intervals=interval_rows, events=event_rows,
                         flags=dfm_flags, notes=dfm_notes, decay_samples=decay_by_sample,
                         lit_samples=lit_any, timelines=timelines)


# ---------------------------------------------------------------------------
# The program as a table
# ---------------------------------------------------------------------------

def program_rows(program: op.OptoProgram, dfm_ids=None) -> list[dict]:
    """One row per interval of every DFM section (or of *dfm_ids*)."""
    rows: list[dict] = []
    ids = sorted(program.dfms) if dfm_ids is None else [d for d in sorted(dfm_ids)
                                                        if d in program.dfms]
    for dfm_id in ids:
        section = program.dfms[dfm_id]
        for iv in section.intervals:
            triggers = iv.trigger_wells()
            open_wells = iv.open_wells()
            if triggers:
                paradigm = iv.params.paradigm
            elif open_wells:
                paradigm = op.OPEN_LOOP
            else:
                paradigm = op.LIGHTS_OFF
            p = iv.params
            rows.append({
                "DFM": dfm_id, "Interval": iv.index,
                "Start": iv.start.strftime("%Y-%m-%d %H:%M:%S") if iv.start else "",
                "DurationMin": float(iv.duration_min),
                "ProgramType": section.program_type, "Paradigm": paradigm,
                "Dark": bool(iv.dark), "F": int(p.frequency),
                "FrequencyHz": p.frequency_hz,
                "AcclimationEvents": p.acclimation_events, "P": int(p.pulse_width),
                "PulseWidthMs": p.pulse_width_ms, "NonFeeding": bool(p.inverted),
                "DecayMs": p.decay_ms,
                "Delay": p.delay, "MaxTimeOn": p.max_time_on,
                "Linkage": op.linkage_label(section),
                "TriggerWells": op.wells_label(triggers),
                "OpenLoopWells": op.wells_label(open_wells),
            })
    return rows


def paradigms_in(program: op.OptoProgram) -> set[str]:
    """Every paradigm a Trigger Well runs under anywhere in *program*."""
    out: set[str] = set()
    for section in program.dfms.values():
        for iv in section.intervals:
            if iv.trigger_wells():
                out.add(iv.params.paradigm)
            elif iv.open_wells():
                out.add(op.OPEN_LOOP)
    return out
