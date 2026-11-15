"""Simulated DFM firmware 4.0.0 recordings for tests (generation 3).

A per-millisecond simulation of the Paradigm rules in DFMV3_V2's
``src/LEDControl.c``, writing what PyMCU 4.0.0 writes from them: the 90-column
``DFM<id>_<n>.csv`` files (``StatusPacket.py``, ``DFM.py``, ``DFMGroup.py``),
the ``Program.txt`` export (``Program.py``, ``InstructionSet.py``,
``Instruction.py``, ``Paradigm.py``) and ``Messages.csv`` (``MessagesList.py``,
``Message.py``).  ``LEDControl.c`` is the authority: where this module and it
disagree, it is right, and ``test_ledcontrol_crosscheck.py`` compiles it and
compares the two millisecond by millisecond.  Every later stream of the
firmware 4.0.0 work (``docs/v4-data-plan.md``, Phase 0a) builds its test data
here, so the public names below are a contract.

How a recording is made, per DFM:

* **Signals.**  Each Well's signal is built as a 1 ms array: a resting level,
  plus optional :class:`Ramp` drift, plus Gaussian noise drawn every
  ``noise_step_ms`` and interpolated (noise that changes every millisecond
  would make a Well near its threshold flicker faster than the Event gap and
  so look like one endless bout), plus the scripted :class:`Bout` amplitudes.
  The firmware compares the sum of 128 scans with ``threshold * 128``, so the
  array is held as that integer sum and the comparison is the firmware's,
  exactly.  The 128-scan smoothing itself is not simulated.
* **Baseline.**  PyMCU 4.0.1's mean of each Well's first ten ``W`` values,
  rounded half up (``DFM.StartingBaseline``; see ``_pymcu_baseline``), added
  to the authored thresholds
  when ``baseline`` is true.  ``Program.txt`` prints the thresholds raw, as
  PyMCU does when it writes the file at the start of the recording.
* **Schedule.**  PyMCU's ``GetPosition``: the first instruction reaches the
  DFM after ``lead_in_records`` records (PyMCU waits for the baseline), and a
  new one at every change of Interval or Cycle, flagged keep-state when it
  names the Paradigm last sent.
* **Engine.**  Each Well's state machine is ``StepWell`` transcribed, run
  event-driven: one exact step wherever something can change (a contact
  change, a timer running out, an instruction), and the quiet stretches in
  between applied in one go.  ``simulate(spec, _step_every_ms=True)`` runs
  every millisecond instead, which the tests use to check the shortcut.
* **Records.**  Record ``n`` (``Index`` n) holds steps ``(n-1)*200`` to
  ``n*200 - 1`` and is stamped ``start + n * 0.2 s``; ``OnMs`` counts its lit
  steps after Linkage, ``OptoCol1`` and the Paradigm state are read after
  its last step, and ``W`` is that step's sum shifted right by 7, as
  ``AnalogInputs.c`` does.  The lead-in rows' ``Paradigm`` is ``NA``: PyMCU
  has no instruction to name before the program (``StatusPacket``).
* **Restarts** (:class:`Restart`, PyMCU 529b926's ``DFM._StartRestart`` and
  ``_FinishRestart``).  The DFM goes down at a record boundary and writes no
  records while it is down, so ``Index`` jumps; it boots under the idle
  instruction, every Well Off, whose rows PyMCU writes with ``Interval``,
  ``Cycle`` and ``Paradigm`` ``NA``; then PyMCU sends the instruction in
  force again without keep-state, so every Well starts as under a new
  Paradigm.  ``Activations`` carries on, as PyMCU carries it.  The firmware
  has no part in a restart, so ``firmware_inputs`` refuses a DFM with one.
* **Faults** (:func:`write_v4_recording`): a ``Program.txt`` that is not the
  program that ran (``program_paradigms``) and a misreported LED
  (:class:`OnMsFault`).  They change the files only, never the truth.

Times in :class:`SimTruth` are 1 ms steps from the start of the recording; the
record holding step ``t`` has ``Index`` ``t // 200 + 1``.
"""

from __future__ import annotations

import csv
import io
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, time as dtime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ON, OFF = "On", "Off"
CHOICE_LINKAGE = (1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6)
#: Chambers 1-3 each linked to one of chambers 4-6, for Paired and Yoked roles.
YOKED_LINKAGE = (1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3)
IDLE = "Idle"
MISSING = "NA"

PARADIGM_TYPES = ("OpenLoop", "ClosedLoop", "FixedInterval", "NonFeeding",
                  "ProgressiveRatio", "FixedRatio")
PROGRAM_TYPES = ("Linear", "Repeating", "Circadian", "Constant")

#: The signal of a Well at rest, its noise and how often the noise is drawn.
DEFAULT_RESTING = 8.0
DEFAULT_NOISE_SD = 0.3
DEFAULT_NOISE_STEP_MS = 40

#: PyMCU's StatusPacket.DATA_COLUMNS and PARADIGM_COLUMNS, as tuples.
DATA_COLUMNS: tuple[str, ...] = (
    "Date", "Time", "MSec", "Sample", *(f"W{w}" for w in range(1, 13)),
    "Temp", "Humid", "LUX", "VoltsIn", "Dark", "OptoFreq", "OptoPW",
    "OptoCol1", "OptoCol2", "Error", "Index")
PARADIGM_COLUMNS: tuple[str, ...] = (
    "Interval", "Cycle", "Paradigm",
    *(f"{name}{w}" for name in ("Phase", "Activations", "Requirement", "Progress", "OnMs")
      for w in range(1, 13)))
COLUMNS: tuple[str, ...] = DATA_COLUMNS + PARADIGM_COLUMNS

#: The ``read_csv`` dtypes ``PyMCU/docs/DataFileColumns.md`` prescribes.
DATA_FILE_DTYPES: dict[str, str] = {
    "Interval": "Int64", "Cycle": "Int64", "Paradigm": "string",
    **{f"Phase{w}": "string" for w in range(1, 13)},
    **{f"{c}{w}": "Int64" for c in ("Activations", "Requirement", "Progress", "OnMs")
       for w in range(1, 13)},
}

#: PyMCU starts a new data file every 12 hours (DFMGroup.DATA_FILE_SECONDS).
SPLIT_HOURS = 12.0
VOLTS_IN = 5.01

# Wire values, LEDControl.h.
THRESHOLD_ON = 0xFFFE
THRESHOLD_OFF = 0xFFFF
_MAX_THRESHOLD = 0xFFFD            # COMM.ParadigmInstructionPayload clamps to this
_MAX_SUM = 1023 * 128              # 128 scans of a 10-bit ADC
_OPENLOOP, _CLOSEDLOOP, _FIXEDINTERVAL, _NONFEEDING, _PROGRESSIVERATIO, _FIXEDRATIO = range(6)
_RATIO_TYPES = (_PROGRESSIVERATIO, _FIXEDRATIO)
_ACCLIMATION_TYPES = (_FIXEDINTERVAL, _PROGRESSIVERATIO, _FIXEDRATIO)
_MAIN, _ACCLIMATION = 0, 1
_PARADIGM_INSTRUCTION = 0xF7
_PARADIGM_INSTRUCTION_SIZE = 63
_KEEP_STATE = 0x01
_NEVER = 1 << 62
#: PyMCU's ``ResendWindow`` for a 4.0.0 DFM (2 * ``COMM.MAX_STATUS_PACKETS_V4``):
#: a record count that goes back further than this is a restart, not a resend.
_RESEND_WINDOW = 54
#: ``_Instruction.kind``: a program instruction (or the idle one), the moment
#: the DFM goes down, and the idle instruction it boots with after a restart.
_PROGRAM, _DOWN, _BOOT = "program", "down", "boot"


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParadigmSpec:
    name: str
    type: str                      # OpenLoop | ClosedLoop | FixedInterval | NonFeeding | ProgressiveRatio | FixedRatio
    frequency: int = 40
    pulse_width: int = 8
    hold_ms: int = 0               # ClosedLoop
    max_on_ms: int = 0             # ClosedLoop, NonFeeding
    activation_ms: int = 0         # FixedInterval, ProgressiveRatio, FixedRatio (and Acclimation)
    lockout_ms: int = 0            # FixedInterval
    on_delay_ms: int = 0           # NonFeeding
    series: str = "Arithmetic"     # ProgressiveRatio: Arithmetic | Exponential
    first_requirement: int = 1
    step: int = 0
    scale: float = 5.0
    rate: float = 0.2
    cap: int = 65535
    ratio: int = 0                 # FixedRatio
    event_gap_ms: int = 100
    acclimation: int = 0           # FixedInterval and ratio types

    def __post_init__(self) -> None:
        if self.type not in PARADIGM_TYPES:
            raise ValueError(f"Paradigm {self.name}: unknown type {self.type!r}")
        if self.series not in ("Arithmetic", "Exponential"):
            raise ValueError(f"Paradigm {self.name}: unknown Series {self.series!r}")


#: PyMCU's own Paradigm (``Paradigm.Idle``): ClosedLoop at 40 Hz and 8 ms.
IDLE_PARADIGM = ParadigmSpec(IDLE, "ClosedLoop")


@dataclass(frozen=True)
class IntervalSpec:
    paradigm: str
    dark: bool
    thresholds: tuple[int | str, ...]   # 12 of int | ON | OFF
    minutes: float


@dataclass(frozen=True)
class Bout:                        # one scripted contact at a well
    start_s: float
    duration_s: float
    amplitude: float = 40.0


@dataclass(frozen=True)
class Ramp:
    """Drift: the resting level moves by *rise* linearly from *start_s* to
    *end_s* (seconds from the start of the recording) and stays there."""
    start_s: float
    end_s: float
    rise: float


@dataclass(frozen=True)
class Restart:
    """The DFM restarts on its own (a brown-out or a watchdog reset) *at_s*
    seconds from the recording start, at the record boundary nearest it.

    It is down for *down_s*: no records, no light and no contact, so
    ``Index`` and the times jump by ``round(down_s * 1000 / 200)`` records, as
    PyMCU's offset makes them.  It then runs *resend_s* under the idle
    instruction (``round(resend_s * 1000 / 200)`` rows with ``Interval``,
    ``Cycle`` and ``Paradigm`` ``NA``) until PyMCU's instruction applies again.
    PyMCU tells a restart from a resend only when the DFM had counted more
    than 54 records, so a Restart needs that many since the last one."""
    at_s: float
    down_s: float = 5.0
    resend_s: float = 1.0


@dataclass(frozen=True)
class OnMsFault:
    """A misreported LED, for :func:`write_v4_recording`: in the records
    covering ``[start_s, end_s)`` (record ``n`` covers ms ``(n-1)*200`` to
    ``n*200``), Well *well*'s ``OnMs`` is *on_ms* and its ``OptoCol1`` bit is
    set exactly when *on_ms* > 0.  Linkage is not applied."""
    well: int
    start_s: float
    end_s: float
    on_ms: int


@dataclass(frozen=True)
class DFMSpec:
    dfm_id: int
    intervals: tuple[IntervalSpec, ...]
    bouts: Mapping[int, Sequence[Bout]]   # well (1-12) -> bouts
    linkage: tuple[int, ...] = CHOICE_LINKAGE
    program_type: str = "Linear"          # Linear | Repeating | Circadian | Constant
    # Additions to the plan's contract, all optional.
    resting: Mapping[int, float] = field(default_factory=dict)       # well -> level, default DEFAULT_RESTING
    noise_sd: Mapping[int, float] = field(default_factory=dict)      # well -> sd, default DEFAULT_NOISE_SD
    drift: Mapping[int, Ramp | Sequence[Ramp]] = field(default_factory=dict)
    noise_step_ms: int = DEFAULT_NOISE_STEP_MS
    restarts: tuple[Restart, ...] = ()


@dataclass(frozen=True)
class RecordingSpec:
    start: datetime
    duration_min: float
    paradigms: Mapping[str, ParadigmSpec]
    dfms: tuple[DFMSpec, ...]
    baseline: bool = True
    samples_per_second: int = 5
    seed: int = 7
    lead_in_records: int = 25      # Interval 0 rows before the first instruction


@dataclass
class SimTruth:                    # per DFM, what the engine did, for assertions
    """Wells are 1-12; times are 1 ms steps from the recording start.

    An Event span is ``[first contact, end)`` where ``end`` is the step at
    which the gap reached EventGapMs (the last contact plus EventGapMs; a gap
    of 0 acts as 1), the step the DFM went down in a restart, or the end of
    the recording.  ``acclimation_end_ms`` is the step a Well first moved
    from Phase A to M (None if it never did).  Every list covers the whole
    recording, what happened after a restart included.
    """
    activation_onset_ms: dict[int, list[int]]        # well -> ms of each Activation start
    requirement_met: dict[int, list[int | None]]     # well -> Requirement each Activation met (None outside ratio types)
    event_spans_ms: dict[int, list[tuple[int, int]]] # well -> Firmware Events
    acclimation_end_ms: dict[int, int | None]
    on_ms_total: dict[int, int]
    # Additions to the plan's contract.
    #: well -> "A" or "M" for each Activation (Acclimation Activations meet no Requirement).
    activation_phase: dict[int, list[str]] = field(default_factory=dict)
    #: well -> [start, end) ms of the Well's own output, before Linkage.
    lit_spans_ms: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    #: well -> PyMCU's baseline (0 when the spec's baseline is off).
    baseline: dict[int, int] = field(default_factory=dict)
    #: (first ms, Interval from 1, Cycle, Paradigm name) of each instruction sent.
    #: A restart's resend of the instruction in force is not listed here.
    schedule: list[tuple[int, int, int, str]] = field(default_factory=list)
    #: (ms the DFM went down, ms the instruction in force applied again,
    #: records missing) for each :class:`Restart`.
    restarts_ms: list[tuple[int, int, int]] = field(default_factory=list)


@dataclass
class SimResult:
    rows: pd.DataFrame             # the 90 columns, typed as pandas reads them
    truth: SimTruth


@dataclass
class FirmwareInputs:
    """What the DFM itself receives, for running ``LEDControl.c`` on it."""
    signals: np.ndarray                     # (ms, 12) int32 sums of 128 scans
    linkage: tuple[int, ...]
    instructions: list[tuple[int, bytes]]   # (first ms, decoded 0xF7 instruction, 63 bytes)
    record_ms: int


# ---------------------------------------------------------------------------
# Instructions and the Requirement series
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Instruction:
    ms: int                        # the first 1 ms step it applies to
    number: int
    paradigm: ParadigmSpec
    type: int
    keep: bool                     # PyMCU's keep-state flag
    dark: bool
    gap: int
    acclimation: int
    settings: tuple[int, int, int, int, int]
    thresholds: tuple[int, ...]    # wire values
    interval: int | None           # from 0; None outside the program
    cycle: int | None
    kind: str = _PROGRAM           # _PROGRAM, or a restart's _DOWN and _BOOT


def _type_code(paradigm: ParadigmSpec) -> int:
    return PARADIGM_TYPES.index(paradigm.type)


def _settings_words(p: ParadigmSpec) -> tuple[int, int, int, int, int]:
    """The 5 type-dependent numbers of the instruction (Paradigm.SettingsWords)."""
    code = _type_code(p)
    if code == _CLOSEDLOOP:
        words = [p.hold_ms, p.max_on_ms]
    elif code == _FIXEDINTERVAL:
        words = [p.activation_ms, p.lockout_ms]
    elif code == _NONFEEDING:
        words = [p.on_delay_ms, p.max_on_ms]
    elif code == _PROGRESSIVERATIO:
        if p.series == "Exponential":
            words = [p.activation_ms, 1, round(p.scale * 1000), round(p.rate * 1000), p.cap]
        else:
            words = [p.activation_ms, 0, p.first_requirement, p.step, p.cap]
    elif code == _FIXEDRATIO:
        words = [p.activation_ms, p.ratio]
    else:
        words = []
    words = [int(w) for w in words] + [0] * (5 - len(words))
    return tuple(words)  # type: ignore[return-value]


def _wire_gap(p: ParadigmSpec) -> int:
    # Paradigm.eventGapMs: types without an EventGapMs setting send the default.
    return p.event_gap_ms if _type_code(p) in (_CLOSEDLOOP, *_ACCLIMATION_TYPES) else 100


def _wire_acclimation(p: ParadigmSpec) -> int:
    return p.acclimation if _type_code(p) in _ACCLIMATION_TYPES else 0


def _instruction(p: ParadigmSpec, *, ms: int = 0, number: int = 0, keep: bool = False,
                 dark: bool = False, thresholds: Sequence[int] = (THRESHOLD_OFF,) * 12,
                 interval: int | None = None, cycle: int | None = None) -> _Instruction:
    return _Instruction(ms, number, p, _type_code(p), keep, dark, _wire_gap(p) & 0xFFFF,
                        _wire_acclimation(p) & 0xFFFF, _settings_words(p),
                        tuple(int(t) for t in thresholds), interval, cycle)


#: InitializeLEDControl's instruction: ClosedLoop, every Well Off, number 0.
_IDLE_INSTRUCTION = _instruction(IDLE_PARADIGM)


def _requirement_for(ins: _Instruction, k: int) -> int:
    """LEDControl.c RequirementFor: Events needed for the k-th Main Activation.
    The exponential series is computed in single precision, as the firmware
    does, so the rounding at the halves matches."""
    s = ins.settings
    if ins.type == _FIXEDRATIO:
        value = s[1]
    elif ins.type == _PROGRESSIVERATIO:
        cap = s[4]
        if cap == 0 or cap > 0xFFFF:
            cap = 0xFFFF
        if s[1] == 1:
            f32 = np.float32
            scale = f32(s[2]) / f32(1000.0)
            # As in the firmware: Scale 0 skips the exponential (0 * infinity would be NaN),
            # and anything not below the cap, infinity included, gives the cap.
            x = scale * np.exp((f32(s[3]) / f32(1000.0)) * f32(k)) - scale if scale > 0 else f32(0)
            value = int(x + f32(0.5)) if x < f32(cap) else cap
        else:
            value = s[2] + s[3] * (k - 1)
        value = min(value, cap)
    else:
        return 0
    return max(1, min(int(value), 0xFFFF))


def requirement_series(paradigm: ParadigmSpec, n: int) -> list[int]:
    """The Requirements of the first *n* Main Activations under *paradigm*."""
    ins = _instruction(paradigm)
    return [_requirement_for(ins, k) for k in range(1, n + 1)]


def paradigm_frame(dfm_id: int, number: int, paradigm: ParadigmSpec, *, keep: bool, dark: bool,
                   thresholds: Sequence[int]) -> bytes:
    """The decoded 0xF7 instruction (``ParseParadigmInstruction``'s input):
    ID, command, the body of ``COMM.ParadigmInstructionPayload``, checksum."""
    body = struct.pack(">HBBBHHHH", number & 0xFFFF, _type_code(paradigm), _KEEP_STATE if keep else 0,
                       1 if dark else 0, paradigm.frequency, paradigm.pulse_width,
                       _wire_gap(paradigm), _wire_acclimation(paradigm))
    body += struct.pack(">5I", *_settings_words(paradigm))
    body += struct.pack(">12H", *thresholds)
    packet = bytes((dfm_id, _PARADIGM_INSTRUCTION)) + body
    frame = packet + ((-sum(packet)) & 0xFFFFFFFF).to_bytes(4, "big")
    assert len(frame) == _PARADIGM_INSTRUCTION_SIZE
    return frame


# ---------------------------------------------------------------------------
# The schedule (PyMCU InstructionSet.GetPosition, DFM.UpdateInstruction)
# ---------------------------------------------------------------------------

def _record_ms(spec: RecordingSpec) -> int:
    if spec.samples_per_second <= 0 or 1000 % spec.samples_per_second:
        raise ValueError("samples_per_second must divide 1000")
    return 1000 // spec.samples_per_second


def _n_records(spec: RecordingSpec) -> int:
    return int(round(spec.duration_min * 60 * spec.samples_per_second))


def _td_ms(td: timedelta) -> int:
    return int(round(td / timedelta(milliseconds=1)))


def _elapsed(dfm: DFMSpec) -> list[tuple[timedelta, timedelta]]:
    """Each Interval's elapsed start and length, summed as InstructionSet does."""
    out, start = [], timedelta(0)
    for iv in dfm.intervals:
        length = timedelta(seconds=iv.minutes * 60)
        out.append((start, length))
        start += length
    return out


def _paradigm(spec: RecordingSpec, name: str) -> ParadigmSpec:
    if name in spec.paradigms:
        return spec.paradigms[name]
    if name == IDLE:
        return IDLE_PARADIGM
    raise ValueError(f"No Paradigm named {name!r}")


def _wire_threshold(value: int | str, baseline: int) -> int:
    if isinstance(value, str):
        if value.lower() == ON.lower():
            return THRESHOLD_ON
        if value.lower() == OFF.lower():
            return THRESHOLD_OFF
        raise ValueError(f"Thresholds are On, Off or 1-1023, not {value!r}")
    if not 1 <= int(value) <= 1023:
        raise ValueError(f"Thresholds are On, Off or 1-1023, not {value!r}")
    return min(int(value) + baseline, _MAX_THRESHOLD)


def _validate(spec: RecordingSpec) -> None:
    ids = [d.dfm_id for d in spec.dfms]
    if len(set(ids)) != len(ids):
        raise ValueError("DFM ids must be unique")
    for name, p in spec.paradigms.items():
        if name != p.name:
            raise ValueError(f"Paradigm keyed {name!r} is named {p.name!r}")
    for d in spec.dfms:
        if not 1 <= d.dfm_id <= 98:
            raise ValueError("DFM ids are 1 to 98 (99 is the Environmental Monitor)")
        if not d.intervals:
            raise ValueError(f"DFM {d.dfm_id} has no Intervals")
        if d.program_type not in PROGRAM_TYPES:
            raise ValueError(f"DFM {d.dfm_id}: unknown program type {d.program_type!r}")
        if len(d.linkage) != 12:
            raise ValueError(f"DFM {d.dfm_id}: Linkage needs 12 values")
        for iv in d.intervals:
            if len(iv.thresholds) != 12:
                raise ValueError(f"DFM {d.dfm_id}: an Interval needs 12 thresholds")
            if iv.minutes <= 0:
                raise ValueError(f"DFM {d.dfm_id}: an Interval must last more than 0 minutes")
            _paradigm(spec, iv.paradigm)
            for t in iv.thresholds:
                _wire_threshold(t, 0)
        if d.program_type == "Circadian":
            total = sum((length for _, length in _elapsed(d)), timedelta(0))
            if total != timedelta(minutes=1440):
                raise ValueError(f"DFM {d.dfm_id}: Circadian Intervals must add up to 1440 minutes")
    n = _n_records(spec)
    if spec.lead_in_records < (10 if spec.baseline else 0) or spec.lead_in_records >= n:
        raise ValueError("lead_in_records must cover the 10 baseline records and end before the recording does")
    for d in spec.dfms:
        _outages(spec, d)


@dataclass(frozen=True)
class _Outage:
    """A :class:`Restart` on the record grid: records ``after + 1`` to
    ``after + missing`` are never written, and ``idle`` records run under the
    idle instruction before the program's applies again at *resume_ms*."""
    down_ms: int
    up_ms: int
    resume_ms: int
    after: int                     # Index of the last record before
    missing: int
    idle: int
    count_before: int              # the DFM's own record count then (PyMCU's lastDFMIndex)


def _outages(spec: RecordingSpec, dfm: DFMSpec) -> list[_Outage]:
    P = _record_ms(spec)
    n = _n_records(spec)
    out: list[_Outage] = []
    since = 0                      # the Index the DFM's own count began after
    for r in sorted(dfm.restarts, key=lambda r: r.at_s):
        after = int(round(r.at_s * 1000 / P))
        missing = int(round(r.down_s * 1000 / P))
        idle = int(round(r.resend_s * 1000 / P))
        where = f"DFM {dfm.dfm_id}: the Restart at {r.at_s:g} s"
        if missing < 1 or idle < 1:
            raise ValueError(f"{where} must be down for a record and idle for a record at least")
        if after <= spec.lead_in_records or (out and after < out[-1].resume_ms // P):
            raise ValueError(f"{where} must come after the first instruction and after the last Restart")
        if after - since <= _RESEND_WINDOW:
            raise ValueError(f"{where} comes {after - since} records into the DFM's count; PyMCU "
                             f"takes a count back within {_RESEND_WINDOW} for a resend")
        if after + missing + idle >= n:
            raise ValueError(f"{where} must end before the recording does")
        out.append(_Outage(after * P, (after + missing) * P, (after + missing + idle) * P,
                           after, missing, idle, after - since))
        since = after + missing
    return out


def _received(schedule: list[_Instruction], outages: Sequence[_Outage]) -> list[_Instruction]:
    """The instructions the DFM runs, in order: *schedule*, and for each
    outage the moment it goes down, the idle instruction it boots with, and
    the instruction in force then sent again (PyMCU ``_StartRestart`` forgets
    the Paradigm last sent, so without keep-state, under the same number).
    A scheduled change while the DFM is down or idle is superseded by that
    resend."""
    out: list[_Instruction] = []
    i = 0
    for o in outages:
        while i < len(schedule) and schedule[i].ms < o.down_ms:
            out.append(schedule[i])
            i += 1
        out.append(replace(_IDLE_INSTRUCTION, ms=o.down_ms, kind=_DOWN))
        out.append(replace(_IDLE_INSTRUCTION, ms=o.up_ms, kind=_BOOT))
        while i < len(schedule) and schedule[i].ms <= o.resume_ms:
            i += 1
        out.append(replace(schedule[i - 1], ms=o.resume_ms, keep=False))
    out.extend(schedule[i:])
    return out


def _position(dfm: DFMSpec, bounds: list[tuple[int, int]], start: datetime, t_ms: int) -> tuple[int, int]:
    """(Interval index from 0, Cycle) at step *t_ms*, as InstructionSet.GetPosition."""
    def index_at(x: int) -> int:
        for k, (a, b) in enumerate(bounds):
            if a <= x < b:
                return k
        return len(bounds) - 1

    if dfm.program_type == "Circadian":
        now = start + timedelta(milliseconds=t_ms)
        midnight = datetime.combine(now.date(), dtime(0), tzinfo=now.tzinfo)
        return index_at(_td_ms(now - midnight)), (now.date() - start.date()).days + 1
    if dfm.program_type == "Constant":
        return 0, 1
    if dfm.program_type == "Repeating":
        total = bounds[-1][1]
        return index_at(t_ms % total), t_ms // total + 1
    return index_at(t_ms), 1


def _schedule(spec: RecordingSpec, dfm: DFMSpec, baseline: Sequence[int], n_ms: int, P: int) -> list[_Instruction]:
    """The program instructions the DFM receives, from the end of the lead-in."""
    bounds = [(_td_ms(a), _td_ms(a + length)) for a, length in _elapsed(dfm)]
    t0 = spec.lead_in_records * P
    if dfm.program_type == "Linear":
        candidates = [a for a, _ in bounds[1:]]
    elif dfm.program_type == "Repeating":
        total = bounds[-1][1]
        candidates = [c * total + a for c in range(n_ms // total + 1) for a, _ in bounds]
    elif dfm.program_type == "Circadian":
        first = _td_ms(datetime.combine(spec.start.date(), dtime(0), tzinfo=spec.start.tzinfo) - spec.start)
        candidates = [first + day * 86_400_000 + a for day in range(n_ms // 86_400_000 + 2) for a, _ in bounds]
    else:
        candidates = []
    times = [t0] + sorted({t for t in candidates if t0 < t < n_ms})

    out: list[_Instruction] = []
    last_name, last_position = IDLE, None
    for t in times:
        position = _position(dfm, bounds, spec.start, t)
        if position == last_position:
            continue
        index, cycle = position
        iv = dfm.intervals[index]
        p = _paradigm(spec, iv.paradigm)
        wire = [_wire_threshold(v, baseline[w]) for w, v in enumerate(iv.thresholds)]
        out.append(_instruction(p, ms=t, number=len(out) + 1, keep=p.name == last_name, dark=iv.dark,
                                thresholds=wire, interval=index, cycle=cycle))
        last_name, last_position = p.name, position
    return out


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def _ramps(value: Ramp | Sequence[Ramp]) -> tuple[Ramp, ...]:
    return (value,) if isinstance(value, Ramp) else tuple(value)


def _well_sums(seed: int, dfm: DFMSpec, well: int, n_ms: int, upto: int | None = None) -> np.ndarray:
    """Well *well*'s signal per 1 ms step of an *n_ms* recording, as the sum
    of 128 scans (int32); only the first *upto* steps if given (the same
    values: the noise is drawn for the whole recording either way)."""
    rng = np.random.default_rng([seed, dfm.dfm_id, well])
    step = max(1, int(dfm.noise_step_ms))
    sd = float(dfm.noise_sd.get(well, DEFAULT_NOISE_SD))
    knots = (rng.standard_normal(n_ms // step + 2) * sd).astype(np.float32)
    n = n_ms if upto is None else min(upto, n_ms)
    knots = knots[: n // step + 2]
    frac = np.arange(step, dtype=np.float32) / np.float32(step)
    signal = (knots[:-1, None] + (knots[1:] - knots[:-1])[:, None] * frac[None, :]).reshape(-1)[:n]
    signal += np.float32(dfm.resting.get(well, DEFAULT_RESTING))
    for ramp in _ramps(dfm.drift.get(well, ())):
        a = max(int(round(ramp.start_s * 1000)), 0)
        b = max(int(round(ramp.end_s * 1000)), a)
        if b > a:
            ramp_part = np.float32(ramp.rise) * (np.arange(b - a, dtype=np.float32) / np.float32(b - a))
            signal[min(a, n):min(b, n)] += ramp_part[: max(0, min(b, n) - a)]
        signal[min(b, n):] += np.float32(ramp.rise)
    for bout in dfm.bouts.get(well, ()):
        a = int(round(bout.start_s * 1000))
        b = a + int(round(bout.duration_s * 1000))
        a, b = max(a, 0), min(b, n)
        if b > a:
            signal[a:b] += np.float32(bout.amplitude)
    # In place: a full-length recording is tens of millions of steps.
    signal *= np.float32(128)
    np.rint(signal, out=signal)
    np.clip(signal, 0, _MAX_SUM, out=signal)
    return signal.astype(np.int32)


def _pymcu_baseline(first_w: Sequence[int]) -> int:
    """PyMCU 4.0.1's ``DFM.StartingBaseline``: the readings summed and divided
    once, rounded half up.  Before 4.0.1 ``DFM.UpdateBaseline`` truncated a
    running mean at every record, up to 2 counts lower.  Under this rule the
    stress spec's DFM 2 Well 10 hides two Acclimation Activations in its first
    row of Cycle 2's second PR1 Interval, where it was On in the first: the
    case ``activation_ledger`` reads from ``ParadigmState.acclimation``."""
    values = [int(x) for x in first_w]
    return (2 * sum(values) + len(values)) // (2 * len(values))


def _baseline_of(seed: int, dfm: DFMSpec, well: int, n_ms: int, record_ms: int) -> int:
    """The baseline PyMCU takes for *well* of *dfm* in a recording of *n_ms*:
    from its first ten records, so only their steps are built."""
    sums = _well_sums(seed, dfm, well, n_ms, upto=10 * record_ms)
    return _pymcu_baseline(sums[np.arange(1, 11) * record_ms - 1] >> 7)


# ---------------------------------------------------------------------------
# The engine: LEDControl.c StepWell, per Well
# ---------------------------------------------------------------------------

class _Well:
    """A Well's WellState (LEDControl.c), plus what the simulation records."""
    __slots__ = ("phase", "in_bout", "was_lit", "cut_off", "gap_ms", "hold_left", "activation_left",
                 "lockout_left", "on_ms", "delay_left", "max_on_left", "main_activations",
                 "acclimation_done", "activations", "requirement", "progress",
                 "event_open", "event_start", "events", "onsets", "met", "onset_phase", "acclimation_end")

    def __init__(self) -> None:
        # InitializeLEDControl then ResetParadigmState under the idle instruction.
        self.phase = _MAIN
        self.in_bout = self.was_lit = self.cut_off = False
        self.gap_ms = self.hold_left = self.activation_left = self.lockout_left = 0
        self.on_ms = self.delay_left = self.max_on_left = 0
        self.main_activations = self.acclimation_done = self.activations = 0
        self.requirement = self.progress = 0
        self.event_open = False
        self.event_start = 0
        self.events: list[tuple[int, int]] = []
        self.onsets: list[int] = []
        self.met: list[int | None] = []
        self.onset_phase: list[str] = []
        self.acclimation_end: int | None = None

    def reset(self, ins: _Instruction) -> None:
        """ResetWell: the Activation total and the bout in progress are kept."""
        self.phase = _ACCLIMATION if ins.acclimation > 0 and ins.type in _ACCLIMATION_TYPES else _MAIN
        self.was_lit = self.cut_off = False
        self.hold_left = self.activation_left = self.lockout_left = self.on_ms = self.delay_left = 0
        self.max_on_left = ins.settings[1] if ins.type == _NONFEEDING else 0
        self.main_activations = self.acclimation_done = self.progress = 0
        self.requirement = _requirement_for(ins, 1)

    def power_off(self, t: int) -> None:
        """The DFM goes down at step *t*: the firmware loses everything
        (``InitializeLEDControl`` runs again when it boots), the bout in
        progress included; PyMCU keeps the Activation total."""
        if self.event_open:
            self.events.append((self.event_start, t))
            self.event_open = False
        self.in_bout = False
        self.gap_ms = 0
        self.reset(_IDLE_INSTRUCTION)

    def count(self, t: int, met: int | None) -> None:
        self.activations += 1
        self.onsets.append(t)
        self.met.append(met)
        self.onset_phase.append("A" if self.phase == _ACCLIMATION else "M")


def _step(s: _Well, ins: _Instruction, contact: bool, t: int) -> bool:
    """One 1 ms step of a Well with a level threshold (StepWell after its
    On/Off early returns)."""
    event_started = False
    if contact:
        s.gap_ms = 0
        if not s.in_bout:
            s.in_bout = True
            event_started = not s.was_lit
            if event_started:
                s.event_open, s.event_start = True, t
    elif s.in_bout:
        s.gap_ms += 1
        if s.gap_ms >= ins.gap:
            s.in_bout = s.cut_off = False
            s.gap_ms = 0
            if s.event_open:
                s.events.append((s.event_start, t))
                s.event_open = False

    a = ins.settings
    if s.phase == _ACCLIMATION:
        # StepAcclimation: closed loop with ActivationMs as the hold.
        lit = False
        if contact:
            s.hold_left = a[0]
            lit = True
        elif s.hold_left > 0:
            s.hold_left -= 1
            lit = True
        # Each on-period is an Activation, as in ClosedLoop: a fly back within the Event
        # gap after the hold has ended gets a new one, though not a new Event.
        if lit and not s.was_lit:
            s.acclimation_done += 1
            s.count(t, None)
        if not lit and s.was_lit and s.acclimation_done >= ins.acclimation:
            s.phase = _MAIN
            if s.acclimation_end is None:
                s.acclimation_end = t
    elif ins.type == _OPENLOOP:
        lit = True
    elif ins.type == _CLOSEDLOOP:
        if s.cut_off:
            s.hold_left = 0
            lit = False
        else:
            lit = False
            if contact:
                s.hold_left = a[0]
                lit = True
            elif s.hold_left > 0:
                s.hold_left -= 1
                lit = True
            if lit and not s.was_lit:
                s.count(t, None)
                s.on_ms = 0
            if lit and a[1] > 0:
                s.on_ms += 1
                if s.on_ms > a[1]:
                    lit = False
                    s.hold_left = 0
                    s.cut_off = s.in_bout
    elif ins.type == _FIXEDINTERVAL:
        if s.lockout_left > 0:
            s.lockout_left -= 1
        if contact and s.activation_left == 0 and s.lockout_left == 0:
            s.activation_left, s.lockout_left = a[0], a[1]
            s.count(t, None)
        lit = s.activation_left > 0
        if lit:
            s.activation_left -= 1
    elif ins.type == _NONFEEDING:
        lit = False
        if contact:
            s.delay_left, s.max_on_left = a[0], a[1]
        elif s.delay_left > 0:
            s.delay_left -= 1
        elif a[1] == 0:
            lit = True
        elif s.max_on_left > 0:
            s.max_on_left -= 1
            lit = True
        if lit and not s.was_lit:
            s.count(t, None)
    else:
        # StepRatio: the Event that meets the Requirement starts an Activation.
        if event_started:
            s.progress = (s.progress + 1) & 0xFFFF
            if s.progress >= s.requirement:
                met = s.requirement
                s.progress = 0
                s.main_activations += 1
                s.requirement = _requirement_for(ins, s.main_activations + 1)
                s.activation_left = a[0]
                s.count(t, met)
        lit = s.activation_left > 0
        if lit:
            s.activation_left -= 1
    s.was_lit = lit
    return lit


def _skip(s: _Well, ins: _Instruction, contact: bool, limit: int) -> int:
    """Applies the next k <= *limit* steps at once, where each would only move
    counters along (no light change, no Activation, no bout change), and
    returns k.  Called after an exact step with the same contact."""
    if contact:
        if not s.in_bout or s.gap_ms:
            return 0
        k_bout = _NEVER
    else:
        # The step at which the gap reaches EventGapMs ends the bout.
        k_bout = max(max(ins.gap, 1) - s.gap_ms - 1, 0) if s.in_bout else _NEVER

    a = ins.settings
    hold = activation = delay = max_on = on_ms = False
    if s.phase == _ACCLIMATION:
        if contact:
            lit, k = True, _NEVER
        elif s.hold_left > 0:
            lit, k, hold = True, s.hold_left, True
        else:
            lit, k = False, _NEVER
    elif ins.type == _OPENLOOP:
        lit, k = True, _NEVER
    elif ins.type == _CLOSEDLOOP:
        if s.cut_off:
            lit, k = False, _NEVER
        elif contact or s.hold_left > 0:
            lit = True
            k, hold = (_NEVER, False) if contact else (s.hold_left, True)
            if a[1] > 0:
                k, on_ms = min(k, a[1] - s.on_ms), True
        else:
            lit, k = False, _NEVER
    elif ins.type == _FIXEDINTERVAL:
        if s.activation_left > 0:
            lit, k, activation = True, s.activation_left, True
        elif contact:
            # The next start comes at the step where the lockout reaches 0.
            lit, k = False, max(s.lockout_left - 1, 0)
        else:
            lit, k = False, _NEVER
    elif ins.type == _NONFEEDING:
        if contact:
            lit, k = False, _NEVER
        elif s.delay_left > 0:
            lit, k, delay = False, s.delay_left, True
        elif a[1] == 0:
            lit, k = True, _NEVER
        elif s.max_on_left > 0:
            lit, k, max_on = True, s.max_on_left, True
        else:
            lit, k = False, _NEVER
    else:
        if s.activation_left > 0:
            lit, k, activation = True, s.activation_left, True
        else:
            lit, k = False, _NEVER

    k = min(k, k_bout, limit)
    if k <= 0 or lit != s.was_lit:
        return 0
    if hold:
        s.hold_left -= k
    if activation:
        s.activation_left -= k
    if delay:
        s.delay_left -= k
    if max_on:
        s.max_on_left -= k
    if on_ms:
        s.on_ms += k
    if ins.type == _FIXEDINTERVAL and s.phase == _MAIN:
        s.lockout_left -= min(s.lockout_left, k)
    if not contact and s.in_bout:
        s.gap_ms += k
    return k


@dataclass
class _WellRun:
    lit_starts: list[int]
    lit_ends: list[int]
    log_t: np.ndarray
    log_values: np.ndarray            # (n, 4): activations, requirement, progress, phase
    well: _Well


def _run_well(w: int, sums: np.ndarray, schedule: list[_Instruction], n_ms: int,
              step_every_ms: bool) -> _WellRun:
    """Runs Well index *w* (0-11) through the recording, under the
    instructions the DFM receives (:func:`_received`)."""
    s = _Well()
    lit_now = False
    lit_starts: list[int] = []
    lit_ends: list[int] = []
    log_t: list[int] = [-1]
    log_v: list[tuple[int, int, int, int]] = [(0, 0, 0, _MAIN)]

    def light(t: int, lit: bool) -> None:
        nonlocal lit_now
        if lit != lit_now:
            (lit_starts if lit else lit_ends).append(t)
            lit_now = lit

    def log(t: int) -> None:
        v = (s.activations, s.requirement, s.progress, s.phase)
        if v != log_v[-1]:
            log_t.append(t)
            log_v.append(v)

    previous = _IDLE_INSTRUCTION
    for i, ins in enumerate(schedule):
        s0 = ins.ms
        s1 = schedule[i + 1].ms if i + 1 < len(schedule) else n_ms
        if ins.kind == _DOWN:
            # A restart: dark, and nothing is stepped until the DFM boots.
            s.power_off(s0)
            light(s0, False)
            log(s0)
            previous = ins
            continue
        # ApplyParadigmInstruction: keep only with the flag and the same type.
        if not (ins.keep and ins.type == previous.type):
            s.reset(ins)
        previous = ins
        log(s0)
        threshold = ins.thresholds[w]
        if threshold in (THRESHOLD_ON, THRESHOLD_OFF):
            # On and Off return before the Paradigm runs: nothing changes but the light.
            light(s0, threshold == THRESHOLD_ON)
            continue
        if s1 <= s0:
            continue
        contact = sums[s0:s1] > threshold * 128
        edges = np.flatnonzero(contact[1:] != contact[:-1]) + 1
        run_starts = np.concatenate(([0], edges)).tolist()
        run_ends = np.concatenate((edges, [s1 - s0])).tolist()
        run_values = contact[run_starts].tolist()
        for b0, b1, c in zip(run_starts, run_ends, run_values):
            t, t1 = s0 + b0, s0 + b1
            while t < t1:
                light(t, _step(s, ins, c, t))
                log(t)
                t += 1
                if t < t1 and not step_every_ms:
                    t += _skip(s, ins, c, t1 - t)
    if lit_now:
        lit_ends.append(n_ms)
    if s.event_open:
        s.events.append((s.event_start, n_ms))
        s.event_open = False
    return _WellRun(lit_starts, lit_ends, np.asarray(log_t, dtype=np.int64),
                    np.asarray(log_v, dtype=np.int64), s)


# ---------------------------------------------------------------------------
# Linkage and records
# ---------------------------------------------------------------------------

def _union(spans: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    starts = np.concatenate([a for a, _ in spans]) if spans else np.zeros(0, np.int64)
    ends = np.concatenate([b for _, b in spans]) if spans else np.zeros(0, np.int64)
    if not len(starts):
        return starts, ends
    order = np.argsort(starts, kind="stable")
    starts, ends = starts[order], ends[order]
    reach = np.maximum.accumulate(ends)
    new = np.ones(len(starts), dtype=bool)
    new[1:] = starts[1:] > reach[:-1]
    first = np.flatnonzero(new)
    return starts[first], np.maximum.reduceat(ends, first)


def _lit_before(starts: np.ndarray, ends: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Lit ms in [0, t) for disjoint sorted spans."""
    done = np.concatenate(([0], np.cumsum(ends - starts)))
    i = np.searchsorted(ends, t, side="right")
    partial = np.zeros(len(t), dtype=np.int64)
    inside = i < len(starts)
    partial[inside] = np.maximum(t[inside] - starts[i[inside]], 0)
    return done[i] + partial


def _lit_at(starts: np.ndarray, ends: np.ndarray, t: np.ndarray) -> np.ndarray:
    j = np.searchsorted(starts, t, side="right") - 1
    ok = j >= 0
    out = np.zeros(len(t), dtype=bool)
    out[ok] = ends[j[ok]] > t[ok]
    return out


def _na_ints(values: np.ndarray, missing: np.ndarray) -> pd.arrays.IntegerArray:
    return pd.arrays.IntegerArray(values.astype(np.int64), np.asarray(missing, dtype=bool).copy())


def _simulate_dfm(spec: RecordingSpec, dfm: DFMSpec, step_every_ms: bool) -> SimResult:
    P = _record_ms(spec)
    n_rec = _n_records(spec)
    n_ms = n_rec * P
    rec_t = np.arange(1, n_rec + 1, dtype=np.int64) * P - 1     # each record's last step

    baseline = [_baseline_of(spec.seed, dfm, w, n_ms, P) if spec.baseline else 0 for w in range(1, 13)]
    schedule = _schedule(spec, dfm, baseline, n_ms, P)
    outages = _outages(spec, dfm)
    received = _received(schedule, outages)
    # One Well's millisecond signal at a time: the Wells only meet through Linkage, on the light.
    w_values = np.empty((n_rec, 12), dtype=np.int64)
    runs = []
    for w in range(12):
        sums = _well_sums(spec.seed, dfm, w + 1, n_ms)
        w_values[:, w] = sums[rec_t] >> 7
        runs.append(_run_well(w, sums, received, n_ms, step_every_ms))
        del sums

    # Linkage: a lit Well lights every Well with its number; OnMs counts after that.
    own = [(np.asarray(r.lit_starts, np.int64), np.asarray(r.lit_ends, np.int64)) for r in runs]
    on_ms = np.zeros((n_rec, 12), dtype=np.int64)
    snapshot = np.zeros((n_rec, 12), dtype=bool)
    edges = np.arange(0, n_rec + 1, dtype=np.int64) * P
    for number in set(dfm.linkage):
        members = [w for w in range(12) if dfm.linkage[w] == number]
        starts, ends = _union([own[w] for w in members])
        counts = np.minimum(np.diff(_lit_before(starts, ends, edges)), 255)
        at = _lit_at(starts, ends, rec_t)
        for w in members:
            on_ms[:, w] = counts
            snapshot[:, w] = at

    # The instruction in force at each record; -1 is the idle lead-in.
    ins_ms = np.asarray([ins.ms for ins in received], dtype=np.int64)
    which = np.searchsorted(ins_ms, rec_t, side="right") - 1
    lead = which < 0
    table = [_IDLE_INSTRUCTION] + received
    pick = which + 1
    kinds = np.asarray([ins.kind for ins in table], dtype=object)[pick]
    down = (kinds == _DOWN) & ~lead          # records a restarting DFM never wrote
    boot = (kinds == _BOOT) & ~lead
    # PyMCU names no instruction before the program, nor the idle one a DFM boots with after a
    # restart (StatusPacket._ParadigmColumns): no Paradigm, no Cycle, no light settings.
    unnamed = lead | boot
    numeric = np.asarray([[t not in (THRESHOLD_ON, THRESHOLD_OFF) for t in ins.thresholds] for ins in table])[pick]
    ratio = np.asarray([ins.type in _RATIO_TYPES for ins in table])[pick]

    times = pd.Timestamp(spec.start) + pd.to_timedelta(np.arange(1, n_rec + 1) * P, unit="ms")
    cols: dict[str, object] = {
        "Date": pd.array(times.strftime("%m/%d/%Y"), dtype="str"),
        "Time": pd.array(times.strftime("%H:%M:%S"), dtype="str"),
        "MSec": np.asarray(times.microsecond, dtype=np.float64) / 1000.0,
        "Sample": np.arange(1, n_rec + 1, dtype=np.int64),
    }
    for w in range(12):
        cols[f"W{w + 1}"] = w_values[:, w].astype(np.int64)
    cols["Temp"] = np.zeros(n_rec)
    cols["Humid"] = np.zeros(n_rec)
    cols["LUX"] = np.zeros(n_rec)
    cols["VoltsIn"] = np.full(n_rec, VOLTS_IN)
    # Dark, OptoFreq and OptoPW are 0 before any instruction (DataFileColumns.md).
    cols["Dark"] = np.where(unnamed, 0, np.asarray([int(i.dark) for i in table])[pick]).astype(np.int64)
    cols["OptoFreq"] = np.where(unnamed, 0, np.asarray([i.paradigm.frequency for i in table])[pick]).astype(np.int64)
    cols["OptoPW"] = np.where(unnamed, 0, np.asarray([i.paradigm.pulse_width for i in table])[pick]).astype(np.int64)
    cols["OptoCol1"] = (snapshot.astype(np.int64) << np.arange(12, dtype=np.int64)).sum(axis=1)
    cols["OptoCol2"] = np.zeros(n_rec, dtype=np.int64)
    cols["Error"] = np.zeros(n_rec, dtype=np.int64)
    cols["Index"] = np.arange(1, n_rec + 1, dtype=np.int64)
    interval = np.asarray([-1 if i.interval is None else i.interval + 1 for i in table])[pick]
    cycle = np.asarray([0 if i.cycle is None else i.cycle for i in table])[pick]
    cols["Interval"] = _na_ints(np.where(lead, 0, interval), boot)
    cols["Cycle"] = _na_ints(cycle, unnamed)
    names = np.asarray([i.paradigm.name for i in table], dtype=object)[pick]
    names[unnamed] = None
    cols["Paradigm"] = pd.array(names, dtype="string")

    state = np.empty((n_rec, 12, 4), dtype=np.int64)
    for w, run in enumerate(runs):
        state[:, w, :] = run.log_values[np.searchsorted(run.log_t, rec_t, side="right") - 1]
    in_acclimation = (state[:, :, 3] == _ACCLIMATION) & numeric
    shows_ratio = ratio[:, None] & numeric
    for w in range(12):
        cols[f"Phase{w + 1}"] = pd.array(np.where(in_acclimation[:, w], "A", "M").astype(object), dtype="string")
    for w in range(12):
        cols[f"Activations{w + 1}"] = _na_ints(state[:, w, 0], np.zeros(n_rec, dtype=bool))
    for w in range(12):
        cols[f"Requirement{w + 1}"] = _na_ints(state[:, w, 1], ~shows_ratio[:, w])
    for w in range(12):
        cols[f"Progress{w + 1}"] = _na_ints(state[:, w, 2], ~shows_ratio[:, w])
    for w in range(12):
        cols[f"OnMs{w + 1}"] = _na_ints(on_ms[:, w], np.zeros(n_rec, dtype=bool))
    rows = pd.DataFrame(cols, columns=list(COLUMNS))
    if down.any():
        # Index and the times carry on across a restart; Sample counts the rows PyMCU wrote.
        rows = rows.loc[~down].reset_index(drop=True)
        rows["Sample"] = np.arange(1, len(rows) + 1, dtype=np.int64)

    wells = range(1, 13)
    truth = SimTruth(
        activation_onset_ms={w: list(runs[w - 1].well.onsets) for w in wells},
        requirement_met={w: list(runs[w - 1].well.met) for w in wells},
        event_spans_ms={w: list(runs[w - 1].well.events) for w in wells},
        acclimation_end_ms={w: runs[w - 1].well.acclimation_end for w in wells},
        on_ms_total={w: int(on_ms[~down, w - 1].sum()) for w in wells},
        activation_phase={w: list(runs[w - 1].well.onset_phase) for w in wells},
        lit_spans_ms={w: list(zip(runs[w - 1].lit_starts, runs[w - 1].lit_ends)) for w in wells},
        baseline={w: baseline[w - 1] for w in wells},
        schedule=[(i.ms, i.interval + 1, i.cycle, i.paradigm.name) for i in schedule],
        restarts_ms=[(o.down_ms, o.resume_ms, o.missing) for o in outages],
    )
    return SimResult(rows, truth)


def simulate(spec: RecordingSpec, *, _step_every_ms: bool = False) -> dict[int, SimResult]:
    """Every DFM of *spec*, keyed by id.  ``_step_every_ms`` runs the exact
    step every millisecond instead of skipping quiet stretches (slow; tests)."""
    _validate(spec)
    return {d.dfm_id: _simulate_dfm(spec, d, _step_every_ms) for d in spec.dfms}


def firmware_inputs(spec: RecordingSpec, dfm_id: int) -> FirmwareInputs:
    """The signals, Linkage and instructions DFM *dfm_id* receives, for
    running the firmware's own ``LEDControl.c`` on the same recording.  A
    DFM with a :class:`Restart` has none: a restart is the power supply's and
    PyMCU's doing, which ``LEDControl.c`` does not model."""
    _validate(spec)
    dfm = next(d for d in spec.dfms if d.dfm_id == dfm_id)
    if dfm.restarts:
        raise ValueError(f"DFM {dfm_id} restarts; LEDControl.c cannot run a restart")
    P = _record_ms(spec)
    n_rec = _n_records(spec)
    n_ms = n_rec * P
    signals = np.empty((n_ms, 12), dtype=np.int32)
    for w in range(12):
        signals[:, w] = _well_sums(spec.seed, dfm, w + 1, n_ms)
    baseline = [_baseline_of(spec.seed, dfm, w, n_ms, P) if spec.baseline else 0 for w in range(1, 13)]
    frames = [(ins.ms, paradigm_frame(dfm.dfm_id, ins.number, ins.paradigm, keep=ins.keep, dark=ins.dark,
                                      thresholds=ins.thresholds))
              for ins in _schedule(spec, dfm, baseline, n_ms, P)]
    return FirmwareInputs(signals, tuple(dfm.linkage), frames, P)


# ---------------------------------------------------------------------------
# Program.txt and Messages.csv
# ---------------------------------------------------------------------------

def _describe(p: ParadigmSpec) -> str:
    """Paradigm.Describe: every setting of the type, in its order, with units."""
    def ms(name: str, value: int) -> str:
        return f"{name} {value} ms"

    parts = [f"Frequency {p.frequency} Hz", f"PulseWidth {p.pulse_width} ms"]
    code = _type_code(p)
    gap = ms("EventGapMs", p.event_gap_ms)
    acclimation = f"Acclimation {p.acclimation}"
    if code == _CLOSEDLOOP:
        parts += [ms("HoldMs", p.hold_ms), ms("MaxOnMs", p.max_on_ms), gap]
    elif code == _FIXEDINTERVAL:
        parts += [ms("ActivationMs", p.activation_ms), ms("LockoutMs", p.lockout_ms), gap, acclimation]
    elif code == _NONFEEDING:
        parts += [ms("OnDelayMs", p.on_delay_ms), ms("MaxOnMs", p.max_on_ms)]
    elif code == _PROGRESSIVERATIO:
        ## PyMCU 4.0.1 writes the Counter on every ratio Paradigm; the
        ## fixtures simulate Event counting.
        parts += [ms("ActivationMs", p.activation_ms), "Counter Events", f"Series {p.series}"]
        if p.series == "Exponential":
            parts += [f"Scale {p.scale:g}", f"Rate {p.rate:g}"]
        else:
            parts += [f"FirstRequirement {p.first_requirement}", f"Step {p.step}"]
        parts += [f"Cap {p.cap}", gap, acclimation]
    elif code == _FIXEDRATIO:
        parts += [ms("ActivationMs", p.activation_ms), "Counter Events", f"Ratio {p.ratio}", gap, acclimation]
    return f"{p.name} ({p.type}): {', '.join(parts)}"


def _threshold_text(value: int | str) -> str:
    if isinstance(value, str):
        return ON if value.lower() == ON.lower() else OFF
    return str(int(value))


def _common(values: Sequence[object], default: object) -> object:
    return values[0] if values and all(v == values[0] for v in values) else default


def program_text(spec: RecordingSpec) -> str:
    """The ``Program.txt`` PyMCU writes at the start of the recording
    (``MCUProgram.GetProgramDescription`` for a loaded Program).  The defaults
    are what every DFM shares (as a ``[General]`` section would give them),
    else PyMCU's own defaults."""
    _validate(spec)
    stamp = "%m/%d/%Y %H:%M:%S"
    duration = timedelta(minutes=spec.duration_min)
    linkage = _common([tuple(d.linkage) for d in spec.dfms], tuple(range(1, 13)))
    ptype = _common([d.program_type for d in spec.dfms], "Linear")
    s = f"Start Time: {spec.start.strftime(stamp)}\n"
    s += f"End Time: {(spec.start + duration).strftime(stamp)}\n"
    s += "Duration: {:g} min\n".format(duration.total_seconds() / 60)
    s += "Default Linkage: " + ",".join(str(v) for v in linkage) + "\n"  # type: ignore[union-attr]
    s += f"Default Program Type: {ptype}\n"
    s += "Baseline: " + ("Yes" if spec.baseline else "No") + "\n"
    used: dict[str, ParadigmSpec] = {}
    for d in spec.dfms:
        for iv in d.intervals:
            used.setdefault(iv.paradigm, _paradigm(spec, iv.paradigm))
    if used:
        s += "\nParadigms:\n" + "".join(_describe(p) + "\n" for p in used.values())
    for d in sorted(spec.dfms, key=lambda d: d.dfm_id):
        s += f"\n***DFM {d.dfm_id}***\n"
        s += "Linkage: " + ",".join(str(v) for v in d.linkage) + "\n"
        s += f"Program Type: {d.program_type}\n"
        for iv, (start, length) in zip(d.intervals, _elapsed(d)):
            if start < duration:
                s += "({}) {}, {}, {}, {:g} min\n".format(
                    (spec.start + start).strftime(stamp), iv.paradigm, "Dark" if iv.dark else "Lit",
                    ",".join(_threshold_text(t) for t in iv.thresholds), length.total_seconds() / 60)
    return s


def _message(dfm_id: int, when: datetime, sample: int, text: str, label: str) -> str:
    """Message.FileRow, written as MessagesList does: with the csv module, so
    a message holding a comma is quoted."""
    out = io.StringIO()
    csv.writer(out, lineterminator="\n").writerow(
        [dfm_id, when.strftime("%m/%d/%Y"), when.strftime("%H:%M:%S"), when.microsecond / 1000,
         sample, text, label])
    return out.getvalue()


def restart_messages(dfm_id: int, count_before: int, after: int, missing: int,
                     record_ms: int = 200) -> tuple[str, str]:
    """PyMCU's two errors for a restart, worded as ``DFM._StartRestart`` and
    ``DFM._FinishRestart`` word them: the DFM's own record count went back
    from *count_before* to 1, and Index ``after + 1`` to ``after + missing``
    are missing."""
    started = (f"DFM {dfm_id} restarted: its record count went back from {count_before} to 1. "
               "Its Paradigm state and Activation counts started again. Sending its Interval and "
               "linkage again.")
    finished = "DFM {} recorded nothing for about {:.0f} s while it restarted: Index {} to {} are missing.".format(
        dfm_id, missing * record_ms / 1000, after + 1, after + missing)
    return started, finished


def messages_text(spec: RecordingSpec, *,
                  index_gaps: Mapping[int, Sequence[tuple[int, int]]] | None = None) -> str:
    """``Messages.csv``: the recording's start and end from DFM 0; for each
    injected gap, the "Lost records" error PyMCU logs on the first record
    after it, dated when the gap began (``DFM._ReportLostRecords``); and for
    each :class:`Restart`, the two errors of :func:`restart_messages`, dated
    at the first record after the outage.  The first has the DFM's own count
    as its ``Sample`` (1), the second the last Index before, as PyMCU's."""
    P = _record_ms(spec)
    errors = []
    for dfm_id, gaps in (index_gaps or {}).items():
        for after, missing in gaps:
            index = after + missing + 1
            when = spec.start + timedelta(milliseconds=(index - missing) * P)
            first, last = after + 1, after + missing
            span = f"Index {first}" if first == last else f"Index {first} to {last}"
            text = (f"Lost {missing} record{'' if missing == 1 else 's'} ({missing * P / 1000:.1f} s): "
                    f"{span}, from {when:%H:%M:%S}.")
            errors.append((when, dfm_id, _message(dfm_id, when, index, text, "Error")))
    for d in spec.dfms:
        for o in _outages(spec, d):
            when = spec.start + timedelta(milliseconds=(o.after + o.missing + 1) * P)
            started, finished = restart_messages(d.dfm_id, o.count_before, o.after, o.missing, P)
            errors.append((when, d.dfm_id, _message(d.dfm_id, when, 1, started, "Error")))
            errors.append((when, d.dfm_id, _message(d.dfm_id, when, o.after, finished, "Error")))
    s = "DFM,Date,Time,Millisecond,Sample,Message,MsgType\n"
    s += _message(0, spec.start, 0, "Recording started", "Unknown")
    s += "".join(text for _, _, text in sorted(errors, key=lambda e: (e[0], e[1])))
    s += _message(0, spec.start + timedelta(minutes=spec.duration_min), 0, "Recording ended", "Unknown")
    return s


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def _without_paradigm_state(rows: pd.DataFrame) -> pd.DataFrame:
    """Generation 2: PyMCU 4.0.0 writes every Paradigm column NA for a record
    from firmware older than 4.0.0."""
    out = rows.copy()
    n = len(out)
    for c in PARADIGM_COLUMNS:
        if c == "Paradigm" or c.startswith("Phase"):
            out[c] = pd.array([pd.NA] * n, dtype="string")
        else:
            out[c] = pd.array([pd.NA] * n, dtype="Int64")
    return out


def _program_spec(spec: RecordingSpec, program_paradigms: Mapping[str, ParadigmSpec] | None) -> RecordingSpec:
    """*spec* with *program_paradigms* in place of its Paradigms of the same
    names, for a ``Program.txt`` that is not the program that ran."""
    if not program_paradigms:
        return spec
    for name, p in program_paradigms.items():
        if name not in spec.paradigms or p.name != name:
            raise ValueError(f"program_paradigms must replace the spec's own Paradigms by name, not {name!r}")
    return replace(spec, paradigms={**spec.paradigms, **program_paradigms})


def _apply_on_ms_fault(rows: pd.DataFrame, fault: OnMsFault, record_ms: int) -> None:
    if not 1 <= int(fault.well) <= 12 or not 0 <= int(fault.on_ms) <= 255:
        raise ValueError(f"an OnMsFault is a Well 1-12 and an OnMs 0-255, not {fault}")
    first_ms = (rows["Index"].to_numpy(dtype=np.int64) - 1) * record_ms
    hit = (first_ms >= round(fault.start_s * 1000)) & (first_ms < round(fault.end_s * 1000))
    rows.loc[hit, f"OnMs{fault.well}"] = int(fault.on_ms)
    bit = 1 << (int(fault.well) - 1)
    led = rows["OptoCol1"].to_numpy(dtype=np.int64, copy=True)
    led[hit] = (led[hit] | bit) if fault.on_ms > 0 else (led[hit] & ~bit)
    rows["OptoCol1"] = led


def _write(data_dir: Path, spec: RecordingSpec, results: Mapping[int, SimResult], *, gen2: bool,
           split_hours: float, index_gaps: Mapping[int, Sequence[tuple[int, int]]],
           error_bits: Mapping[int, Sequence[tuple[int, int]]], program: bool, messages: bool,
           blank_missing: bool, program_paradigms: Mapping[str, ParadigmSpec] | None = None,
           on_ms_faults: Mapping[int, Sequence[OnMsFault]] | None = None) -> dict[str, Path]:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    per_file = max(1, int(round(split_hours * 3600 * spec.samples_per_second)))
    written: dict[str, Path] = {}
    if program:
        path = data_dir / "Program.txt"
        path.write_text(program_text(_program_spec(spec, program_paradigms)), encoding="utf-8", newline="\n")
        written[path.name] = path
    for dfm_id, result in sorted(results.items()):
        rows = _without_paradigm_state(result.rows) if gen2 else result.rows.copy()
        for index, bits in error_bits.get(dfm_id, ()):
            rows.loc[rows["Index"] == index, "Error"] = int(bits)
        for fault in (on_ms_faults or {}).get(dfm_id, ()):
            _apply_on_ms_fault(rows, fault, _record_ms(spec))
        keep = np.ones(len(rows), dtype=bool)
        for after, missing in index_gaps.get(dfm_id, ()):
            keep &= ~((rows["Index"] > after) & (rows["Index"] <= after + missing)).to_numpy()
        rows = rows.loc[keep].reset_index(drop=True)
        # Sample counts the rows PyMCU wrote; Index is the DFM's own record count.
        rows["Sample"] = np.arange(1, len(rows) + 1, dtype=np.int64)
        part = ((rows["Index"].to_numpy() - 1) // per_file)
        for n, piece in enumerate(np.unique(part)):
            path = data_dir / f"DFM{dfm_id}_{n}.csv"
            rows.loc[part == piece].to_csv(path, index=False, float_format="%.2f",
                                           na_rep="" if blank_missing else MISSING, lineterminator="\n")
            written[path.name] = path
    if messages:
        path = data_dir / "Messages.csv"
        path.write_text(messages_text(spec, index_gaps=index_gaps), encoding="utf-8", newline="\n")
        written[path.name] = path
    return written


def write_v4_recording(data_dir: Path, spec: RecordingSpec, *, split_hours: float = SPLIT_HOURS,
                       index_gaps: Mapping[int, Sequence[tuple[int, int]]] = {},   # dfm -> [(after_index, missing)]
                       error_bits: Mapping[int, Sequence[tuple[int, int]]] = {},   # dfm -> [(index, bits)]
                       program: bool = True, messages: bool = True,
                       blank_missing: bool = False,
                       results: Mapping[int, SimResult] | None = None,
                       program_paradigms: Mapping[str, ParadigmSpec] | None = None,
                       on_ms_faults: Mapping[int, Sequence[OnMsFault]] | None = None) -> dict[str, Path]:
    """Writes the recording folder PyMCU 4.0.0 leaves for a 4.0.0 DFM:
    ``DFM<id>_<n>.csv`` (a new file, with its header, every *split_hours*),
    ``Program.txt`` and ``Messages.csv``.  Returns file name -> path.
    *results* reuses a :func:`simulate` of the same spec.

    Two faults, in the files only (the truth is the simulation's):

    * *program_paradigms*: ``Program.txt`` describes these Paradigms in
      place of the spec's of the same names, so it is not the program that
      ran;
    * *on_ms_faults*: DFM id -> :class:`OnMsFault` s, a corrupted or
      misreported LED.
    """
    results = simulate(spec) if results is None else results
    return _write(data_dir, spec, results, gen2=False, split_hours=split_hours, index_gaps=index_gaps,
                  error_bits=error_bits, program=program, messages=messages, blank_missing=blank_missing,
                  program_paradigms=program_paradigms, on_ms_faults=on_ms_faults)


def write_gen2_recording(data_dir: Path, spec: RecordingSpec, *, split_hours: float = SPLIT_HOURS,
                         index_gaps: Mapping[int, Sequence[tuple[int, int]]] = {},
                         error_bits: Mapping[int, Sequence[tuple[int, int]]] = {},
                         program: bool = True, messages: bool = True,
                         blank_missing: bool = False,
                         results: Mapping[int, SimResult] | None = None) -> dict[str, Path]:
    """The same recording from firmware older than 4.0.0 read by PyMCU 4.0.0:
    the same rows with every Paradigm column NA; the W values carry no
    training flag."""
    results = simulate(spec) if results is None else results
    return _write(data_dir, spec, results, gen2=True, split_hours=split_hours, index_gaps=index_gaps,
                  error_bits=error_bits, program=program, messages=messages, blank_missing=blank_missing)


def read_v4_csv(path: Path) -> pd.DataFrame:
    """A data file read as ``DataFileColumns.md`` says to."""
    return pd.read_csv(path, dtype=DATA_FILE_DTYPES)


# ---------------------------------------------------------------------------
# Ready-made specs
# ---------------------------------------------------------------------------

#: PyMCU's example start (DataFileColumns.md).
START = datetime(2026, 10, 16, 23, 34, 0)
CHOICE_THRESHOLDS: tuple[int | str, ...] = (20, OFF) * 6
#: Chamber -> what its fly does in :func:`choice_pr_spec` (both DFMs alike).
CHOICE_PR_CASES: dict[int, str] = {1: "healthy", 2: "healthy", 3: "drifting",
                                   4: "dead_led", 5: "never_training", 6: "healthy"}

OPEN = ParadigmSpec("Open", "OpenLoop")
CLOSED = ParadigmSpec("Closed", "ClosedLoop", hold_ms=500, max_on_ms=2500)
INTERVAL_1MIN = ParadigmSpec("Interval1Min", "FixedInterval", activation_ms=1000, lockout_ms=60000)
NOT_FEEDING = ParadigmSpec("NotFeeding", "NonFeeding", on_delay_ms=500, max_on_ms=0)
PR10 = ParadigmSpec("PR10", "ProgressiveRatio", activation_ms=500, series="Arithmetic",
                    first_requirement=1, step=10, cap=200, acclimation=10)
FR5 = ParadigmSpec("FR5", "FixedRatio", activation_ms=500, ratio=5)

# A healthy fly feeds about every 12 s at its trigger well and less at the other.
_TRIGGER_GAP_S = 12.0
_OTHER_GAP_S = 45.0
# Bouts start after the lead-in so the baseline sees the Well at rest.
_FIRST_BOUT_S = 6.0


def _bout_train(rng: np.random.Generator, start_s: float, end_s: float, mean_gap_s: float,
                amplitude: tuple[float, float] = (35.0, 45.0)) -> tuple[Bout, ...]:
    """Bouts of 1-4 s, at least 1 s apart (well beyond any Event gap)."""
    bouts, t = [], start_s + float(rng.exponential(mean_gap_s / 2))
    while True:
        duration = float(rng.uniform(1.0, 4.0))
        if t + duration > end_s:
            return tuple(bouts)
        bouts.append(Bout(round(t, 3), round(duration, 3), round(float(rng.uniform(*amplitude)), 1)))
        t += duration + max(1.0, float(rng.exponential(mean_gap_s)))


def _healthy_chamber(rng: np.random.Generator, chamber: int, end_s: float) -> dict[int, tuple[Bout, ...]]:
    a, b = 2 * chamber - 1, 2 * chamber
    return {a: _bout_train(rng, _FIRST_BOUT_S, end_s, _TRIGGER_GAP_S),
            b: _bout_train(rng, _FIRST_BOUT_S, end_s, _OTHER_GAP_S)}


def _rng(seed: int, dfm_id: int, chamber: int) -> np.random.Generator:
    # Bouts are drawn apart from the noise (which uses [seed, dfm, well]).
    return np.random.default_rng([seed, dfm_id, chamber, 1])


def choice_pr_spec(minutes: float = 30.0, *, closed_minutes: float | None = None,
                   dfm_ids: Sequence[int] = (1, 2), seed: int = 7) -> RecordingSpec:
    """PyMCU's example program on choice chambers: ``Closed`` then ``PR10``
    (Acclimation 10, Step 10, Cap 200), A wells 20, B wells Off, CHOICE_LINKAGE.

    ``minutes=360, closed_minutes=30`` is the full-length original; the
    default is a short version, Closed for a sixth of it.  Each DFM's chambers
    act out :data:`CHOICE_PR_CASES`:

    * ``healthy``: feeds about every 12 s at A, finishes Acclimation in PR10
      and earns Requirements 1, 11, 21, ...;
    * ``drifting``: never feeds at A, but A's resting level ramps up during
      Closed to just under its firmware threshold (its own baseline plus 20)
      and stays there, noisier than the rest (sd 1.0), so the noise carries it
      across the threshold again and again: the firmware counts Event after
      Event, lights the LED through the rest of Closed and earns Activations
      under PR10, while a running-median baseline sees a flat signal;
    * ``dead_led``: the fly sits on A while PyMCU takes the baseline (a 2.6 s
      bout from the first millisecond), so A's firmware threshold sits about
      50 above rest and no later bout reaches it: pyflic sees feeding, the
      firmware sees no Event, and the LED never lights;
    * ``never_training``: three bouts at A during Closed and six during PR10,
      fewer than Acclimation's ten, so A stays in Phase A.
    """
    closed = minutes / 6.0 if closed_minutes is None else closed_minutes
    end_s, closed_s = minutes * 60.0, closed * 60.0
    n_ms = int(round(minutes * 60 * 5)) * 200
    dfms = []
    for dfm_id in dfm_ids:
        bouts: dict[int, tuple[Bout, ...]] = {}
        noise: dict[int, float] = {}
        drifting: list[int] = []
        for chamber, case in CHOICE_PR_CASES.items():
            rng = _rng(seed, dfm_id, chamber)
            a, b = 2 * chamber - 1, 2 * chamber
            bouts.update(_healthy_chamber(rng, chamber, end_s))
            if case == "drifting":
                bouts[a] = ()
                noise[a] = 1.0
                drifting.append(a)
            elif case == "dead_led":
                bouts[a] = (Bout(0.0, 2.6, 50.0),) + bouts[a]
            elif case == "never_training":
                in_closed = np.linspace(_FIRST_BOUT_S + 10, closed_s - 10, 3)
                in_pr = np.linspace(closed_s + 30, end_s - 30, 6)
                bouts[a] = tuple(Bout(round(float(t), 3), 2.0, 40.0) for t in (*in_closed, *in_pr))
        dfm = DFMSpec(
            dfm_id=dfm_id,
            intervals=(IntervalSpec("Closed", True, CHOICE_THRESHOLDS, closed),
                       IntervalSpec("PR10", True, CHOICE_THRESHOLDS, minutes - closed)),
            bouts=bouts, linkage=CHOICE_LINKAGE, noise_sd=noise)
        # The drift starts after the baseline window, so the baseline (and so
        # the threshold the plateau sits under) comes from the drift-free signal.
        drift = {a: Ramp(0.1 * closed_s, 0.5 * closed_s,
                         _baseline_of(seed, dfm, a, n_ms, 200) + 20 - 0.25 - DEFAULT_RESTING)
                 for a in drifting}
        dfms.append(replace(dfm, drift=drift))
    return RecordingSpec(start=START, duration_min=minutes, paradigms={"Closed": CLOSED, "PR10": PR10},
                         dfms=tuple(dfms), seed=seed)


def _healthy_dfm(dfm_id: int, intervals: tuple[IntervalSpec, ...], minutes: float, seed: int, *,
                 linkage: tuple[int, ...] = CHOICE_LINKAGE, program_type: str = "Linear") -> DFMSpec:
    bouts: dict[int, tuple[Bout, ...]] = {}
    for chamber in range(1, 7):
        bouts.update(_healthy_chamber(_rng(seed, dfm_id, chamber), chamber, minutes * 60.0))
    return DFMSpec(dfm_id=dfm_id, intervals=intervals, bouts=bouts, linkage=linkage,
                   program_type=program_type)


def one_of_each_spec(minutes: float = 30.0, *, seed: int = 7) -> RecordingSpec:
    """PyMCU's ``ParadigmExamples.txt``: one DFM per Paradigm type, every fly
    healthy.  DFM 1 OpenLoop (A wells On, Lit), 2 ClosedLoop, 3 FixedInterval,
    4 NonFeeding, 5 Closed for a twelfth of the run then PR10, 6 FixedRatio;
    A wells 20 and B wells Off elsewhere."""
    def one(name: str, dark: bool = True, thresholds: tuple[int | str, ...] = CHOICE_THRESHOLDS,
            length: float = minutes) -> IntervalSpec:
        return IntervalSpec(name, dark, thresholds, length)

    closed = minutes / 12.0
    plan = {
        1: (one("Open", False, (ON, OFF) * 6),),
        2: (one("Closed"),),
        3: (one("Interval1Min"),),
        4: (one("NotFeeding"),),
        5: (one("Closed", length=closed), one("PR10", length=minutes - closed)),
        6: (one("FR5"),),
    }
    paradigms = {p.name: p for p in (OPEN, CLOSED, INTERVAL_1MIN, NOT_FEEDING, PR10, FR5)}
    return RecordingSpec(start=START, duration_min=minutes, paradigms=paradigms,
                         dfms=tuple(_healthy_dfm(d, iv, minutes, seed) for d, iv in plan.items()), seed=seed)


def fixed_ratio_spec(minutes: float = 30.0, *, dfm_ids: Sequence[int] = (1,), seed: int = 7) -> RecordingSpec:
    """``FR5`` with Acclimation 3 on choice chambers (A 20, B Off), every fly
    healthy."""
    fr5 = ParadigmSpec("FR5", "FixedRatio", activation_ms=500, ratio=5, acclimation=3)
    intervals = (IntervalSpec("FR5", True, CHOICE_THRESHOLDS, minutes),)
    return RecordingSpec(start=START, duration_min=minutes, paradigms={"FR5": fr5},
                         dfms=tuple(_healthy_dfm(d, intervals, minutes, seed) for d in dfm_ids), seed=seed)


def repeating_open_loop_spec(minutes: float = 45.0, *, seed: int = 7) -> RecordingSpec:
    """ProgramFormat.md's open-loop schedule: Paradigm ``Open`` with every Well
    On for 3 minutes, then every Well Off for 12, Repeating (keep-state across
    the Intervals and Cycles, as they share the name)."""
    intervals = (IntervalSpec("Open", True, (ON,) * 12, 3.0), IntervalSpec("Open", True, (OFF,) * 12, 12.0))
    return RecordingSpec(start=START, duration_min=minutes, paradigms={"Open": OPEN},
                         dfms=(_healthy_dfm(1, intervals, minutes, seed, program_type="Repeating"),), seed=seed)


#: The Paradigms of :func:`stress_spec`, chosen for awkward edges: holds longer
#: than the Event gap, a gap of 0, cut-offs, lockouts shorter than the
#: Activation, Activations shorter than the gap, Acclimation on each type that
#: takes it, the exponential series.
STRESS_PARADIGMS: tuple[ParadigmSpec, ...] = (
    ParadigmSpec("OL", "OpenLoop"),
    ParadigmSpec("CL1", "ClosedLoop", hold_ms=300, max_on_ms=400),
    ParadigmSpec("CL2", "ClosedLoop", hold_ms=150, event_gap_ms=0),
    ParadigmSpec("CL3", "ClosedLoop", hold_ms=250, max_on_ms=120, event_gap_ms=50),
    ParadigmSpec("FI1", "FixedInterval", activation_ms=300, lockout_ms=700, event_gap_ms=80, acclimation=2),
    ParadigmSpec("FI2", "FixedInterval", activation_ms=400, lockout_ms=150),
    ParadigmSpec("NF1", "NonFeeding", on_delay_ms=200, max_on_ms=500),
    ParadigmSpec("NF2", "NonFeeding"),
    ParadigmSpec("PR1", "ProgressiveRatio", activation_ms=60, first_requirement=2, step=1, cap=4, acclimation=2),
    ParadigmSpec("PR2", "ProgressiveRatio", activation_ms=250, series="Exponential", event_gap_ms=30),
    ParadigmSpec("FR1", "FixedRatio", activation_ms=40, ratio=3, event_gap_ms=150, acclimation=1),
)


def stress_spec(minutes: float = 8.0, *, seed: int = 7) -> RecordingSpec:
    """Not an experiment: the engine's edge cases, for the firmware cross-check.

    Four DFMs starting four minutes before midnight: DFM 1 Linear through short
    Intervals of :data:`STRESS_PARADIGMS` (often repeating the last name, so
    keep-state, with Wells moving between levels, On and Off), DFM 2 Repeating
    (a name kept across the Cycle wrap), DFM 3 Circadian (Cycle 2 after
    midnight), DFM 4 Constant with every Well linked.  Bouts are dense, with
    dips shorter and longer than every Event gap, and two Wells per DFM sit
    noisily at their threshold."""
    rng = np.random.default_rng([seed, 99])
    names = [p.name for p in STRESS_PARADIGMS]

    def thresholds() -> tuple[int | str, ...]:
        out: list[int | str] = []
        for _ in range(12):
            u = float(rng.random())
            out.append(ON if u < 0.12 else OFF if u < 0.24 else int(rng.integers(15, 26)))
        return tuple(out)

    def interval(name: str, length: float) -> IntervalSpec:
        return IntervalSpec(name, bool(rng.random() < 0.5), thresholds(), length)

    linear: list[IntervalSpec] = []
    for k in range(max(4, int(minutes * 2))):
        same = k > 0 and rng.random() < 0.35
        name = linear[-1].paradigm if same else names[int(rng.integers(len(names)))]
        linear.append(interval(name, round(float(rng.uniform(0.25, 0.75)), 3)))
    plans = {
        1: (tuple(linear), (1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 8, 8), "Linear"),
        2: ((interval("PR1", 0.4), interval("PR1", 0.3), interval("CL3", 0.5)), CHOICE_LINKAGE, "Repeating"),
        3: ((interval("FR1", 1438.0), interval("FI1", 2.0)), tuple(range(1, 13)), "Circadian"),
        4: ((interval("NF1", 1.0), interval("CL2", 5.0)), (1,) * 12, "Constant"),
    }
    dfms = []
    end_s = minutes * 60.0
    for dfm_id, (intervals, linkage, ptype) in plans.items():
        bouts: dict[int, tuple[Bout, ...]] = {}
        for w in range(1, 13):
            wr = np.random.default_rng([seed, 99, dfm_id, w])
            t, out = _FIRST_BOUT_S, []
            while t < end_s:
                duration = float(wr.uniform(0.005, 2.0)) if wr.random() < 0.7 else float(wr.uniform(0.005, 0.08))
                out.append(Bout(round(t, 3), round(duration, 3), round(float(wr.uniform(25.0, 50.0)), 1)))
                gap = float(wr.uniform(0.01, 0.09)) if wr.random() < 0.4 else float(wr.uniform(0.15, 3.0))
                t += duration + gap
            bouts[w] = tuple(out)
        # Two Wells rise to about 26 after the baseline and sit there, noisy.
        noisy = (3, 10)
        dfms.append(DFMSpec(dfm_id=dfm_id, intervals=intervals, bouts=bouts, linkage=linkage, program_type=ptype,
                            noise_sd={w: 2.0 for w in noisy}, drift={w: Ramp(3.0, 4.0, 18.0) for w in noisy},
                            noise_step_ms=10))
    return RecordingSpec(start=datetime(2026, 10, 16, 23, 56, 0), duration_min=minutes,
                         paradigms={p.name: p for p in STRESS_PARADIGMS}, dfms=tuple(dfms), seed=seed)


def yoked_controls_spec(minutes: float = 30.0, *, seed: int = 7) -> RecordingSpec:
    """ClosedLoop ``Closed`` with chambers 1-3 triggering at their A wells and
    chambers 4-6 all Off, each linked to one of them (YOKED_LINKAGE): chamber
    c + 3 receives exactly chamber c's light."""
    thresholds: tuple[int | str, ...] = (20, OFF) * 3 + (OFF, OFF) * 3
    intervals = (IntervalSpec("Closed", True, thresholds, minutes),)
    return RecordingSpec(start=START, duration_min=minutes, paradigms={"Closed": CLOSED},
                         dfms=(_healthy_dfm(1, intervals, minutes, seed, linkage=YOKED_LINKAGE),), seed=seed)


# ---------------------------------------------------------------------------
# One spec per Paradigm Family edge case (docs/paradigm-analyses-plan.md, C1)
# ---------------------------------------------------------------------------

#: The Paradigms of :func:`family_specs`.
CL500 = ParadigmSpec("CL500", "ClosedLoop", hold_ms=500, max_on_ms=2000)
FI10S = ParadigmSpec("FI10s", "FixedInterval", activation_ms=1000, lockout_ms=10000)
FI500 = ParadigmSpec("FI500", "FixedInterval", activation_ms=1500, lockout_ms=500, acclimation=3)
NF1000 = ParadigmSpec("NF1000", "NonFeeding", on_delay_ms=1000, max_on_ms=0)
NF_MAX5S = ParadigmSpec("NFMax5s", "NonFeeding", on_delay_ms=1000, max_on_ms=5000)
PR_EXP = ParadigmSpec("PRExp", "ProgressiveRatio", activation_ms=500, series="Exponential", scale=5.0,
                      rate=0.2, acclimation=2)
FR3 = ParadigmSpec("FR3", "FixedRatio", activation_ms=500, ratio=3, acclimation=2)
CL_RESTART = ParadigmSpec("CL", "ClosedLoop", hold_ms=500)

#: The keys of :func:`family_specs`, in order.
FAMILY_SPEC_KEYS: tuple[str, ...] = ("closed_max", "fixed_interval", "non_feeding", "exponential", "restart")

#: A distribution of seconds: ``(weight, low, high)`` uniform parts.
_Mixture = Sequence[tuple[float, float, float]]


def _draw(rng: np.random.Generator, mixture: _Mixture) -> float:
    weights = np.asarray([part[0] for part in mixture], dtype=float)
    _w, low, high = mixture[int(rng.choice(len(mixture), p=weights / weights.sum()))]
    return float(rng.uniform(low, high))


def _mixed_train(rng: np.random.Generator, start_s: float, end_s: float, lengths: _Mixture,
                 gaps: _Mixture, amplitude: tuple[float, float] = (35.0, 45.0)) -> tuple[Bout, ...]:
    """Bouts whose lengths and the gaps between them are drawn from *lengths*
    and *gaps*; a gap of at least 20 ms keeps the bouts apart after rounding."""
    bouts, t = [], start_s + _draw(rng, gaps)
    while True:
        length = _draw(rng, lengths)
        if t + length > end_s:
            return tuple(bouts)
        bouts.append(Bout(round(t, 3), round(length, 3), round(float(rng.uniform(*amplitude)), 1)))
        t += length + _draw(rng, gaps)


def _family_dfm(dfm_id: int, paradigm: ParadigmSpec, minutes: float, seed: int, lengths: _Mixture,
                gaps: _Mixture) -> DFMSpec:
    """Choice chambers (A 20, B Off) under *paradigm* for the whole run: the
    A wells' bouts drawn from the mixtures, the B wells' a healthy fly's."""
    end_s = minutes * 60.0
    bouts: dict[int, tuple[Bout, ...]] = {}
    for chamber in range(1, 7):
        rng = _rng(seed, dfm_id, chamber)
        bouts[2 * chamber - 1] = _mixed_train(rng, _FIRST_BOUT_S, end_s, lengths, gaps)
        bouts[2 * chamber] = _bout_train(rng, _FIRST_BOUT_S, end_s, _OTHER_GAP_S)
    return DFMSpec(dfm_id, (IntervalSpec(paradigm.name, True, CHOICE_THRESHOLDS, minutes),), bouts)


def _family_spec(dfms: Sequence[DFMSpec], paradigms: Sequence[ParadigmSpec], minutes: float,
                 seed: int) -> RecordingSpec:
    return RecordingSpec(start=START, duration_min=minutes, paradigms={p.name: p for p in paradigms},
                         dfms=tuple(dfms), seed=seed)


def family_specs(minutes: float = 20.0, *, seed: int = 7) -> dict[str, RecordingSpec]:
    """One spec per edge case of the Paradigm Families, *minutes* long, keyed
    as :data:`FAMILY_SPEC_KEYS`, on choice chambers (A wells 20, B wells Off,
    CHOICE_LINKAGE):

    * ``closed_max``: :data:`CL500`, ClosedLoop with Hold 500 and MaxOnMs
      2000.  Bouts of 2.2-6 s, which it cuts off, and of up to 1.5 s; dips
      shorter than the Event gap; returns 150-450 ms after a bout, within the
      Hold, or after a cut-off at once.
    * ``fixed_interval``: DFM 1 :data:`FI10S` (ActivationMs 1000, LockoutMs
      10000), with feeding during the lockout and contact held through
      several; DFM 2 :data:`FI500` (LockoutMs 500, shorter than ActivationMs
      1500, Acclimation 3), where held contact gives continuous light and a
      new Activation every 1.5 s.
    * ``non_feeding``: DFM 1 :data:`NF1000` (OnDelayMs 1000, MaxOnMs 0), DFM 2
      :data:`NF_MAX5S` (MaxOnMs 5000); gaps between bouts shorter than the
      delay, so it stays dark, and on DFM 2 longer than the delay and MaxOnMs
      together.
    * ``exponential``: :data:`PR_EXP`, ProgressiveRatio with the exponential
      series (Scale 5, Rate 0.2) and Acclimation 2, with a bout every few
      seconds.
    * ``restart``: DFM 1 :data:`FR3` (FixedRatio 3, Acclimation 2), every fly
      healthy, restarting halfway through, in Main; DFM 2 :data:`CL_RESTART`
      (ClosedLoop, Hold 500) restarting at 35% of the run, down 3 s and idle
      2 s, while Well 1 is in a bout that outlasts both.
    """
    end_s = minutes * 60.0
    closed_max = _family_dfm(1, CL500, minutes, seed, lengths=((0.4, 0.05, 1.5), (0.6, 2.2, 6.0)),
                             gaps=((0.1, 0.02, 0.08), (0.2, 0.15, 0.45), (0.7, 1.0, 20.0)))
    fixed_long = _family_dfm(1, FI10S, minutes, seed, lengths=((0.75, 0.3, 3.0), (0.25, 4.0, 25.0)),
                             gaps=((0.6, 0.5, 4.0), (0.4, 4.0, 20.0)))
    fixed_short = _family_dfm(2, FI500, minutes, seed, lengths=((0.5, 0.1, 1.2), (0.5, 1.5, 6.0)),
                              gaps=((0.3, 0.15, 0.8), (0.7, 1.0, 15.0)))
    non_feeding = _family_dfm(1, NF1000, minutes, seed, lengths=((1.0, 0.3, 3.0),),
                              gaps=((0.35, 0.2, 0.9), (0.65, 1.5, 20.0)))
    non_feeding_max = _family_dfm(2, NF_MAX5S, minutes, seed, lengths=((1.0, 0.3, 3.0),),
                                  gaps=((0.3, 0.2, 0.9), (0.3, 1.5, 5.5), (0.4, 6.5, 30.0)))
    exponential = _family_dfm(1, PR_EXP, minutes, seed, lengths=((1.0, 0.15, 1.0),), gaps=((1.0, 0.3, 4.0),))

    ratio = _healthy_dfm(1, (IntervalSpec(FR3.name, True, CHOICE_THRESHOLDS, minutes),), minutes, seed)
    ratio = replace(ratio, restarts=(Restart(round(0.5 * end_s, 1)),))
    closed = _healthy_dfm(2, (IntervalSpec(CL_RESTART.name, True, CHOICE_THRESHOLDS, minutes),), minutes, seed)
    restart = Restart(round(0.35 * end_s, 1), down_s=3.0, resend_s=2.0)
    held = Bout(restart.at_s - 5.0, restart.down_s + restart.resend_s + 10.0)
    well_1 = [b for b in closed.bouts[1]
              if b.start_s + b.duration_s < held.start_s - 3.0 or b.start_s > held.start_s + held.duration_s + 3.0]
    closed = replace(closed, bouts={**closed.bouts, 1: tuple(sorted([*well_1, held], key=lambda b: b.start_s))},
                     restarts=(restart,))
    return {
        "closed_max": _family_spec([closed_max], [CL500], minutes, seed),
        "fixed_interval": _family_spec([fixed_long, fixed_short], [FI10S, FI500], minutes, seed),
        "non_feeding": _family_spec([non_feeding, non_feeding_max], [NF1000, NF_MAX5S], minutes, seed),
        "exponential": _family_spec([exponential], [PR_EXP], minutes, seed),
        "restart": _family_spec([ratio, closed], [FR3, CL_RESTART], minutes, seed),
    }
