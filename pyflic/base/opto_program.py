"""The Opto Program: what the MCU ran on each DFM, read from ``Program.txt``.

The DFM firmware decides when an Optolid LED is on from settings the
stand-alone MCU loads at the start of a run: a signal **threshold** per well, a
**linkage** that ORs wells together, and the rule that turns contact into
light.  None of it is in a pre-4.0.0 DFM CSV, which records only the outcome:
``OptoCol1``, the LEDs actually lit at each sample.

The MCU writes the program back out as ``Program.txt`` and the experimenter
copies that file into ``data/`` beside the CSVs.  This module reads that export
(never the authored ``[General]`` / ``[DFM]`` program, which is the MCU's input,
not a record of what ran) into an :class:`OptoProgram`.  The export has three
shapes, told apart by content, behind one API:

* **legacy** (PyMCU 3.x): each interval line echoes five numbers — ``F:``,
  ``P:``, ``D:``, ``L:``, ``M:`` are frequency, pulse width, decay, delay and
  max-time-on — whose combination selects the paradigm, and thresholds are
  integers (0 always on, -1 never).  Semantics are the MCU/DFM version 2.0
  conventions, with two encodings decoded here: the high seven bits of the
  frequency carry an acclimation-event count, and the top bit of the pulse
  width selects non-feeding activation.  Each interval's numbers are wrapped
  into a :class:`Paradigm` with PyMCU's generated name, as PyMCU 4.0.0 itself
  converts an old program (``PyMCU/docs/ProgramFormat.md``, "Converting old
  programs");
* **v4** (PyMCU 4.0.0): a ``Paradigms:`` block names each **Paradigm** with
  its type and every setting, and interval lines read ``(stamp) NAME,
  Dark|Lit, t1..t12, N min`` (``Dark|Lit, NAME`` in earlier exports) with
  thresholds a level, ``On`` or ``Off``;
* **simple**: a recording started without a Program exports one ``***ALL
  DFM***`` block, every well ``Off``; it yields no DFM section.

Terms, fixed in ``CONTEXT.md``:

* the **Opto Program** is the parsed file;
* a **Paradigm** is a named rule for when a well's LED turns on, of one of six
  types, with its settings;
* a **Program Interval** is one timed step of a DFM's program: its Paradigm,
  its dark setting and a threshold per well;
* a **Linkage Group** is a set of wells sharing one linkage number, lit
  together whenever any member is triggered;
* a **Trigger Well** is a well whose threshold is a signal level in some
  interval, so its signal can switch its group's light on;
* the **Hold** is how long an Activation keeps the LED on after contact ends.

Pure Python, no pandas: the parser is imported by the loader, the linter and
the editors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

PROGRAM_FILENAME = "Program.txt"
N_WELLS = 12

#: Low bits of the frequency value hold the frequency in Hz; the seven above
#: them hold the number of acclimation events of an event-trained progressive
#: ratio (so ``5160`` is 10 events at 40 Hz).
FREQUENCY_BITS = 9
#: The top bit of the pulse width selects non-feeding activation (lights off
#: only while the fly feeds); the pulse width proper is the value below it.
INVERT_BIT = 0x8000

#: Firmware defaults for a parameter no line of a legacy export states.
FIRMWARE_DEFAULTS = {"frequency": 40, "pulse_width": 8, "decay": 0, "delay": 0,
                     "max_time_on": 0}
DEFAULT_LINKAGE: tuple[int, ...] = tuple(range(1, N_WELLS + 1))

#: :attr:`OptoProgram.grammar` — which shape of export the file had.
GRAMMAR_LEGACY = "legacy"
GRAMMAR_V4 = "v4"
GRAMMAR_SIMPLE = "simple"
#: The warning a ``***ALL DFM***`` export always carries.
SIMPLE_WARNING = "recorded without a Program: every well Off"

# ---------------------------------------------------------------------------
# The ``optogenetics`` setting
# ---------------------------------------------------------------------------

SETTING_AUTO = "auto"
SETTING_YES = "yes"
SETTING_NO = "no"
SETTINGS: tuple[str, ...] = (SETTING_AUTO, SETTING_YES, SETTING_NO)
SETTING_KEY = "optogenetics"


def normalize_setting(value, *, where: str = SETTING_KEY) -> str:
    """``auto``, ``yes`` or ``no`` for an ``optogenetics:`` value.

    YAML reads a bare ``yes`` / ``no`` as a boolean, so ``True`` / ``False``
    are the usual spellings on disk; ``None`` (unset) is ``auto``.  Raises
    ``ValueError`` naming *where* for anything else.
    """
    if value is None:
        return SETTING_AUTO
    if isinstance(value, bool):
        return SETTING_YES if value else SETTING_NO
    text = str(value).strip().lower()
    if text in ("", "auto", "default"):
        return SETTING_AUTO
    if text in ("yes", "y", "true", "on", "1"):
        return SETTING_YES
    if text in ("no", "n", "false", "off", "0"):
        return SETTING_NO
    raise ValueError(f"'{where}' must be auto, yes or no, got {value!r}")


def setting_problem(value, *, where: str = SETTING_KEY) -> str | None:
    """Why *value* is not a valid ``optogenetics:`` setting, or ``None``."""
    try:
        normalize_setting(value, where=where)
    except ValueError as err:
        return str(err)
    return None


def setting_to_yaml(setting: str):
    """The on-disk form of *setting*: ``True`` / ``False`` for yes / no, and
    ``None`` for auto, which is the default and is written as nothing."""
    setting = normalize_setting(setting)
    if setting == SETTING_AUTO:
        return None
    return setting == SETTING_YES


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

#: A well lit for the whole interval, whatever the Paradigm (legacy ``0``).
ON = "On"
#: A well that never lights by its own signal (legacy ``-1``); linkage can
#: still light it, and it runs no Paradigm of its own.
OFF = "Off"
#: The highest signal level a threshold can be.
MAX_LEVEL = 1023
#: One well's threshold in one interval: a signal level (1 to
#: :data:`MAX_LEVEL`), :data:`ON` or :data:`OFF`.
Threshold = int | str


def is_trigger(t: Threshold) -> bool:
    """Whether *t* is a signal level — the well's own contact decides."""
    return isinstance(t, int) and not isinstance(t, bool)


def threshold_from_legacy(value: int) -> Threshold:
    """A pre-4.0.0 threshold in the new terms, as PyMCU 4.0.0 converts it: 0
    is :data:`ON`; -1, any other negative and anything above
    :data:`MAX_LEVEL` are :data:`OFF`; the rest are levels."""
    value = int(value)
    if value == 0:
        return ON
    if value < 0 or value > MAX_LEVEL:
        return OFF
    return value


def threshold_text(t: Threshold) -> str:
    """``On``, ``Off`` or the level, as the export prints it."""
    return str(t)


# ---------------------------------------------------------------------------
# Paradigm types
# ---------------------------------------------------------------------------

OPEN_LOOP = "open loop"
CLOSED_LOOP = "closed loop"
#: Not a type of its own: a closed loop whose light is cut off after a
#: maximum on-time.  Kept as a mode because pre-4.0.0 outputs name it.
CLOSED_LOOP_MAX = "closed loop, max time on"
FIXED_INTERVAL = "fixed interval"
PROGRESSIVE_RATIO = "progressive ratio"
FIXED_RATIO = "fixed ratio"
NON_FEEDING = "non-feeding activation"
LIGHTS_OFF = "off"
UNKNOWN_PARADIGM = "unknown"

#: The six Paradigm types, in PyMCU's order.
PARADIGM_TYPES: tuple[str, ...] = (OPEN_LOOP, CLOSED_LOOP, FIXED_INTERVAL, NON_FEEDING,
                                   PROGRESSIVE_RATIO, FIXED_RATIO)
#: PyMCU's spelling of each type, as the export and the data file write it.
TYPE_NAMES: dict[str, str] = {
    OPEN_LOOP: "OpenLoop", CLOSED_LOOP: "ClosedLoop", FIXED_INTERVAL: "FixedInterval",
    NON_FEEDING: "NonFeeding", PROGRESSIVE_RATIO: "ProgressiveRatio",
    FIXED_RATIO: "FixedRatio",
}
_TYPE_BY_NAME = {name.lower(): kind for kind, name in TYPE_NAMES.items()}


def type_from_name(name: str) -> str | None:
    """The type for PyMCU's spelling *name* (case-insensitive), or ``None``."""
    return _TYPE_BY_NAME.get(str(name).strip().lower())


#: Paradigms that earn an Activation by a count of Firmware Events.
RATIO_PARADIGMS: tuple[str, ...] = (PROGRESSIVE_RATIO, FIXED_RATIO)
#: Paradigms whose light is switched on by the fly's contact with a Trigger
#: Well — the ones the light QC can explain from licks.
FEEDING_PARADIGMS: tuple[str, ...] = (CLOSED_LOOP, CLOSED_LOOP_MAX, FIXED_INTERVAL,
                                      PROGRESSIVE_RATIO, FIXED_RATIO)
#: Paradigms in which every feeding bout should light the group.
CLOSED_LOOP_PARADIGMS: tuple[str, ...] = (CLOSED_LOOP, CLOSED_LOOP_MAX)
#: Paradigms that take an ``Acclimation`` count.
ACCLIMATION_PARADIGMS: tuple[str, ...] = (FIXED_INTERVAL, PROGRESSIVE_RATIO, FIXED_RATIO)

SERIES_ARITHMETIC = "Arithmetic"
SERIES_EXPONENTIAL = "Exponential"
#: The largest Requirement the firmware holds (a 16-bit count).
MAX_REQUIREMENT = 0xFFFF


class ProgramError(ValueError):
    """A ``Program.txt`` pyflic cannot use, or more than one of them."""


@dataclass(frozen=True, slots=True)
class OptoParams:
    """One legacy interval's optogenetic parameters as the MCU held them.

    *frequency* and *pulse_width* are the raw values, encodings included;
    the properties decode them.  *delay* and *max_time_on* are milliseconds
    except under a progressive ratio, where they are the lick increment and
    the lick cap.  Reachable as :attr:`Paradigm.legacy`.
    """

    frequency: int
    pulse_width: int
    decay: int
    delay: int
    max_time_on: int

    @property
    def frequency_hz(self) -> int:
        return int(self.frequency) & ((1 << FREQUENCY_BITS) - 1)

    @property
    def acclimation_events(self) -> int:
        return int(self.frequency) >> FREQUENCY_BITS

    @property
    def inverted(self) -> bool:
        return bool(int(self.pulse_width) & INVERT_BIT)

    @property
    def pulse_width_ms(self) -> int:
        return int(self.pulse_width) & (INVERT_BIT - 1)

    @property
    def decay_ms(self) -> int:
        return int(self.decay)

    @property
    def paradigm(self) -> str:
        """The mode this parameter set selects for a well whose threshold
        is positive (the conventions' table; a zero threshold is open loop and
        a negative one is off, whatever the parameters say)."""
        if self.inverted:
            return NON_FEEDING
        if int(self.delay) == 0:
            return CLOSED_LOOP if int(self.max_time_on) == 0 else CLOSED_LOOP_MAX
        return FIXED_INTERVAL if int(self.max_time_on) == 0 else PROGRESSIVE_RATIO

    def describe(self) -> str:
        """One line for a table cell or a summary."""
        parts = [f"{self.frequency_hz} Hz", f"{self.pulse_width_ms} ms pulses",
                 f"decay {self.decay_ms} ms"]
        paradigm = self.paradigm
        if paradigm == PROGRESSIVE_RATIO:
            parts.append(f"ratio step {self.delay} licks, cap {self.max_time_on} licks")
            if self.acclimation_events:
                parts.append(f"{self.acclimation_events} acclimation events")
        elif paradigm == FIXED_INTERVAL:
            parts.append(f"delay {self.delay} ms")
        elif paradigm == CLOSED_LOOP_MAX:
            parts.append(f"max time on {self.max_time_on} ms")
        elif paradigm == NON_FEEDING:
            parts.append(f"delay {self.delay} ms")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# Paradigms
# ---------------------------------------------------------------------------

#: The export's setting names: (setting, Paradigm field, kind).  The kind
#: decides the unit printed and how a value is read.
_FREQUENCY = ("Frequency", "frequency_hz", "hz")
_PULSE_WIDTH = ("PulseWidth", "pulse_width_ms", "ms")
_HOLD = ("HoldMs", "hold_ms", "ms")
_MAX_ON = ("MaxOnMs", "max_on_ms", "ms")
_ACTIVATION = ("ActivationMs", "activation_ms", "ms")
_LOCKOUT = ("LockoutMs", "lockout_ms", "ms")
_ON_DELAY = ("OnDelayMs", "on_delay_ms", "ms")
_EVENT_GAP = ("EventGapMs", "event_gap_ms", "ms")
_ACCLIMATION = ("Acclimation", "acclimation", "count")
_SERIES = ("Series", "series", "series")
_FIRST = ("FirstRequirement", "first_requirement", "count")
_STEP = ("Step", "step", "count")
_SCALE = ("Scale", "scale", "decimal")
_RATE = ("Rate", "rate", "decimal")
_CAP = ("Cap", "cap", "count")
_RATIO = ("Ratio", "ratio", "count")

_LIGHT = (_FREQUENCY, _PULSE_WIDTH)
#: Each type's settings in PyMCU's display order (``Paradigm._SETTINGS``).
_TYPE_SETTINGS: dict[str, tuple[tuple[str, str, str], ...]] = {
    OPEN_LOOP: (),
    CLOSED_LOOP: (_HOLD, _MAX_ON, _EVENT_GAP),
    FIXED_INTERVAL: (_ACTIVATION, _LOCKOUT, _EVENT_GAP, _ACCLIMATION),
    NON_FEEDING: (_ON_DELAY, _MAX_ON),
    PROGRESSIVE_RATIO: (_ACTIVATION, _SERIES, _FIRST, _STEP, _SCALE, _RATE, _CAP,
                        _EVENT_GAP, _ACCLIMATION),
    FIXED_RATIO: (_ACTIVATION, _RATIO, _EVENT_GAP, _ACCLIMATION),
}
#: Settings that belong to one Series only; the export prints the chosen one's.
_SERIES_SETTINGS = {SERIES_ARITHMETIC: ("FirstRequirement", "Step"),
                    SERIES_EXPONENTIAL: ("Scale", "Rate")}
_ALL_SETTINGS = {s[0].lower() for specs in _TYPE_SETTINGS.values() for s in specs} \
    | {s[0].lower() for s in _LIGHT}
_UNITS = {"hz": " Hz", "ms": " ms"}


@dataclass(frozen=True, slots=True)
class Paradigm:
    """A named rule for when a well's LED turns on, with its settings.

    *type* is one of :data:`PARADIGM_TYPES` (pyflic's mode strings;
    :attr:`type_name` is PyMCU's spelling).  A setting the type does not take
    is ``None``.  An interval of a legacy export is wrapped into one with
    *generated* set and its raw numbers kept in *legacy*.
    """

    name: str
    type: str
    frequency_hz: int
    pulse_width_ms: int
    hold_ms: int | None = None            # ClosedLoop HoldMs
    max_on_ms: int | None = None          # ClosedLoop, NonFeeding
    activation_ms: int | None = None      # FixedInterval, ratio types
    lockout_ms: int | None = None         # FixedInterval
    on_delay_ms: int | None = None        # NonFeeding
    event_gap_ms: int | None = None
    acclimation: int = 0
    series: str | None = None             # ProgressiveRatio: Arithmetic | Exponential
    first_requirement: int | None = None
    step: int | None = None
    scale: float | None = None
    rate: float | None = None
    cap: int | None = None
    ratio: int | None = None              # FixedRatio
    generated: bool = False               # a name PyMCU made from old-format settings
    legacy: OptoParams | None = None      # the pre-4.0.0 numbers this was inferred from

    @property
    def type_name(self) -> str:
        """PyMCU's spelling of :attr:`type`, e.g. ``ClosedLoop``."""
        return TYPE_NAMES.get(self.type, self.type)

    @property
    def mode(self) -> str:
        """What the light QC calls this rule: the type, except that a closed
        loop with a maximum on-time is :data:`CLOSED_LOOP_MAX`."""
        if self.type == CLOSED_LOOP and (self.max_on_ms or 0) > 0:
            return CLOSED_LOOP_MAX
        return self.type

    @property
    def hold(self) -> int | None:
        """The Hold (``CONTEXT.md``): ``HoldMs`` for a closed loop,
        ``ActivationMs`` for a fixed interval and the ratio types, ``None``
        for the types without one."""
        if self.type == CLOSED_LOOP:
            return self.hold_ms
        if self.type in ACCLIMATION_PARADIGMS:
            return self.activation_ms
        return None

    def requirement_for(self, k: int) -> int | None:
        """Firmware Events needed for the *k*-th Main Activation (from 1), as
        the firmware computes it (``LEDControl.c``, ``RequirementFor``);
        ``None`` outside the ratio types and for a legacy progressive ratio,
        whose old lick counts have no Requirement."""
        k = int(k)
        if k < 1:
            raise ValueError(f"k counts Main Activations from 1, got {k}")
        if self.type == FIXED_RATIO:
            if self.ratio is None:
                return None
            value = int(self.ratio)
        elif self.type == PROGRESSIVE_RATIO and self.series is not None:
            cap = int(self.cap) if self.cap else MAX_REQUIREMENT
            if cap > MAX_REQUIREMENT:
                cap = MAX_REQUIREMENT
            if self.series == SERIES_EXPONENTIAL:
                ## The MCU sends Scale and Rate as thousandths. The firmware
                ## works in single precision (so the rounding at the halves is
                ## its own), skips the exponential when Scale is 0, and gives
                ## the cap for anything not below it, an overflow included.
                f32 = np.float32
                scale = f32(round(float(5.0 if self.scale is None else self.scale) * 1000)) / f32(1000.0)
                rate = f32(round(float(0.2 if self.rate is None else self.rate) * 1000)) / f32(1000.0)
                if scale > 0:
                    with np.errstate(over="ignore"):
                        x = scale * np.exp(rate * f32(k)) - scale
                else:
                    x = f32(0.0)
                value = int(x + f32(0.5)) if x < f32(cap) else cap
            else:
                first = 1 if self.first_requirement is None else int(self.first_requirement)
                step = 0 if self.step is None else int(self.step)
                value = first + step * (k - 1)
            value = min(value, cap)
        else:
            return None
        return max(1, min(value, MAX_REQUIREMENT))

    def describe(self) -> str:
        """The Paradigm as the export prints it, e.g. ``PR10
        (ProgressiveRatio): Frequency 40 Hz, PulseWidth 8 ms, ActivationMs 500
        ms, Series Arithmetic, …``; settings that are ``None`` are left out."""
        parts = []
        skip = ()
        if self.type == PROGRESSIVE_RATIO:
            skip = tuple(name for series, names in _SERIES_SETTINGS.items()
                         if series != self.series for name in names)
        for setting, attr, kind in _LIGHT + _TYPE_SETTINGS.get(self.type, ()):
            value = getattr(self, attr)
            if value is None or setting in skip:
                continue
            text = f"{value:g}" if kind == "decimal" else str(value)
            parts.append(f"{setting} {text}{_UNITS.get(kind, '')}")
        return f"{self.name} ({self.type_name}): {', '.join(parts)}"


def _generated_name(kind: str, frequency_hz: int, pulse_width_ms: int,
                    settings: Iterable[tuple[str, object]]) -> str:
    """PyMCU's name for a Paradigm made from old settings
    (``Paradigm.Legacy``): ``ClosedLoop-40Hz-8ms-Hold500-MaxOn0``.  Equal
    settings give equal names."""
    return f"{TYPE_NAMES[kind]}-{frequency_hz}Hz-{pulse_width_ms}ms" + "".join(
        f"-{setting.replace('Ms', '')}{value}" for setting, value in settings)


def _pymcu_generated_name(paradigm: Paradigm) -> str | None:
    """The name PyMCU 4.0.0 would give *paradigm* had it come from an old
    interval line, or ``None`` for a type it never generates."""
    p = paradigm
    if p.acclimation:
        return None
    if p.type == CLOSED_LOOP:
        settings = (("HoldMs", p.hold_ms), ("MaxOnMs", p.max_on_ms))
    elif p.type == FIXED_INTERVAL:
        settings = (("ActivationMs", p.activation_ms), ("LockoutMs", p.lockout_ms))
    elif p.type == NON_FEEDING:
        settings = (("OnDelayMs", p.on_delay_ms), ("MaxOnMs", p.max_on_ms))
    else:
        return None
    return _generated_name(p.type, p.frequency_hz, p.pulse_width_ms, settings)


def paradigm_from_legacy(params: OptoParams) -> Paradigm:
    """The :class:`Paradigm` a legacy interval's numbers meant.

    The type is the one :attr:`OptoParams.paradigm` infers; the settings
    follow PyMCU 4.0.0's conversion of an old program: delay 0 is ClosedLoop
    (``HoldMs`` = decay, ``MaxOnMs`` = max-time-on, a negative one meaning no
    limit), a delay with no max-time-on is FixedInterval (``ActivationMs`` =
    decay, ``LockoutMs`` = delay), the pulse width's top bit is NonFeeding
    (``OnDelayMs`` = delay, ``MaxOnMs`` = decay).  PyMCU refuses the old
    progressive ratio and the frequency's acclimation bits, which counted
    milliseconds of contact; pyflic still reads them, as a ProgressiveRatio
    with ``ActivationMs`` = decay and no Requirement series, and as
    ``acclimation``.  Those two extend PyMCU's naming pattern
    (``-Delay…-MaxOn…``, ``-Acclimation…``) so that different old settings
    keep different names.
    """
    mode = params.paradigm
    decay, delay, max_on = int(params.decay), int(params.delay), int(params.max_time_on)
    if mode == NON_FEEDING:
        kind = NON_FEEDING
        settings = {"on_delay_ms": delay, "max_on_ms": decay}
        named = (("OnDelayMs", delay), ("MaxOnMs", decay))
    elif mode in CLOSED_LOOP_PARADIGMS:
        kind = CLOSED_LOOP
        settings = {"hold_ms": decay, "max_on_ms": max(max_on, 0)}
        named = (("HoldMs", decay), ("MaxOnMs", max(max_on, 0)))
    elif mode == FIXED_INTERVAL:
        kind = FIXED_INTERVAL
        settings = {"activation_ms": decay, "lockout_ms": delay}
        named = (("ActivationMs", decay), ("LockoutMs", delay))
    else:
        kind = PROGRESSIVE_RATIO
        settings = {"activation_ms": decay}
        named = (("ActivationMs", decay), ("Delay", delay), ("MaxOn", max_on))
    acclimation = params.acclimation_events
    if acclimation:
        named = (*named, ("Acclimation", acclimation))
    name = _generated_name(kind, params.frequency_hz, params.pulse_width_ms, named)
    return Paradigm(name=name, type=kind, frequency_hz=params.frequency_hz,
                    pulse_width_ms=params.pulse_width_ms, acclimation=acclimation,
                    generated=True, legacy=params, **settings)


# ---------------------------------------------------------------------------
# Program Intervals and DFM sections
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ProgramInterval:
    """One interval line of a DFM section, as the MCU echoed it."""

    index: int                         # 1-based position in its DFM section
    start: datetime | None             # the timestamp the export printed
    dark: bool                         # the DFM ran dark (indicator LEDs off)
    thresholds: tuple[Threshold, ...]  # W1..W12: a level, ON or OFF
    paradigm: Paradigm                 # parsed, or synthesised from F/P/D/L/M
    duration_min: float

    def threshold(self, well: int) -> Threshold:
        return self.thresholds[int(well) - 1]

    def _wells(self, wells: Iterable[int] | None,
               keep: Callable[[Threshold], bool]) -> tuple[int, ...]:
        pool = range(1, N_WELLS + 1) if wells is None else wells
        return tuple(int(w) for w in pool if keep(self.threshold(w)))

    def trigger_wells(self, wells: Iterable[int] | None = None) -> tuple[int, ...]:
        """Wells (of *wells*, default all twelve) whose threshold is a level."""
        return self._wells(wells, is_trigger)

    def open_wells(self, wells: Iterable[int] | None = None) -> tuple[int, ...]:
        """Wells whose threshold is :data:`ON`."""
        return self._wells(wells, lambda t: t == ON)

    def off_wells(self, wells: Iterable[int] | None = None) -> tuple[int, ...]:
        """Wells whose threshold is :data:`OFF`."""
        return self._wells(wells, lambda t: t == OFF)

    def group_mode(self, wells: Iterable[int]) -> str:
        """What a Linkage Group of *wells* does during this interval.

        An :data:`ON` threshold on any member lights the whole group
        constantly (open loop); every member :data:`OFF` keeps it dark (off);
        otherwise the group answers to its Trigger Wells under the
        Paradigm's :attr:`~Paradigm.mode`.
        """
        wells = tuple(int(w) for w in wells)
        if any(self.threshold(w) == ON for w in wells):
            return OPEN_LOOP
        if all(self.threshold(w) == OFF for w in wells):
            return LIGHTS_OFF
        return self.paradigm.mode

    @property
    def hold_ms(self) -> int:
        """The Hold the light QC allows after contact, in ms.  A legacy
        interval keeps its decay whatever the paradigm, as pyflic always
        read it; a v4 one has its Paradigm's :attr:`~Paradigm.hold`, 0 for a
        type without one."""
        legacy = self.paradigm.legacy
        if legacy is not None:
            return legacy.decay_ms
        return int(self.paradigm.hold or 0)


@dataclass(frozen=True, slots=True)
class ScheduledInterval:
    """One occurrence of an interval on the run's timeline."""

    interval: ProgramInterval
    start: datetime
    end: datetime
    occurrence: int                    # 1 for the first pass, 2 for the first repeat…


@dataclass(frozen=True, slots=True)
class DFMProgram:
    """One ``***DFM n***`` section."""

    dfm_id: int
    linkage: tuple[int, ...]
    program_type: str                  # Linear | Repeating | Circadian | Constant
    intervals: tuple[ProgramInterval, ...]

    def linkage_groups(self) -> dict[int, tuple[int, ...]]:
        """``{linkage number: wells}``, in order of first appearance."""
        groups: dict[int, list[int]] = {}
        for well, label in enumerate(self.linkage, start=1):
            groups.setdefault(int(label), []).append(well)
        return {label: tuple(wells) for label, wells in groups.items()}

    def trigger_wells(self) -> tuple[int, ...]:
        """Every well whose threshold is a level in at least one interval
        whose Paradigm is not OpenLoop: under OpenLoop a level well is lit
        regardless, the firmware running no rule for it (as
        :func:`opto_roles.linkage_view` counts it).  A legacy export has no
        OpenLoop type (open loop is a threshold of 0), so its trigger wells
        are every level well, as before."""
        found = {w for iv in self.intervals if iv.paradigm.type != OPEN_LOOP
                 for w in iv.trigger_wells()}
        return tuple(sorted(found))

    def schedule(self, start: datetime, end: datetime) -> list[ScheduledInterval]:
        """The intervals laid out from *start* to *end*.

        The first pass keeps the timestamps the export printed; after it a
        Linear program holds its last interval to the end, a Repeating or
        Circadian one starts over, and a Constant one runs its first interval
        throughout.  An interval running past *end* is cut there — the run's
        duration takes precedence over the intervals, as on the MCU.
        """
        if end <= start or not self.intervals:
            return []
        kind = self.program_type.strip().lower()
        if kind == "constant":
            return [ScheduledInterval(self.intervals[0], start, end, 1)]
        out: list[ScheduledInterval] = []
        t = start
        for iv in self.intervals:
            s = iv.start if iv.start is not None else t
            s = max(s, t)
            if s >= end:
                break
            e = min(s + timedelta(minutes=float(iv.duration_min)), end)
            if e > s:
                out.append(ScheduledInterval(iv, s, e, 1))
            t = max(t, e)
        if t >= end or not out:
            return out
        cycle = sum(max(0.0, float(iv.duration_min)) for iv in self.intervals)
        if kind in ("repeating", "circadian") and cycle > 0:
            occurrence = 1
            while t < end:
                occurrence += 1
                for iv in self.intervals:
                    if t >= end:
                        break
                    e = min(t + timedelta(minutes=float(iv.duration_min)), end)
                    if e > t:
                        out.append(ScheduledInterval(iv, t, e, occurrence))
                    t = e
            return out
        last = out[-1]
        out[-1] = ScheduledInterval(last.interval, last.start, end, last.occurrence)
        return out


@dataclass(slots=True)
class OptoProgram:
    """A parsed ``Program.txt``."""

    path: Path | None
    start: datetime | None
    end: datetime | None
    duration_min: float | None
    baseline: bool                     # firmware subtracts a start-of-run baseline
    dfms: dict[int, DFMProgram] = field(default_factory=dict)
    #: DFM id -> why its section was dropped; that DFM is analysed as if it had
    #: no section, and the reason is reported.
    rejected: dict[int, str] = field(default_factory=dict)
    #: Lines skipped and sections ignored, for the summary.
    warnings: list[str] = field(default_factory=list)
    #: Name -> Paradigm: the ``Paradigms:`` block, or for a legacy export the
    #: Paradigms synthesised from its intervals.
    paradigms: dict[str, Paradigm] = field(default_factory=dict)
    #: :data:`GRAMMAR_LEGACY`, :data:`GRAMMAR_V4` or :data:`GRAMMAR_SIMPLE`.
    grammar: str = GRAMMAR_LEGACY
    #: ``Baseline: Adaptive``: PyMCU kept the baseline up to date during the
    #: recording, at most ``baseline_rate_per_hour`` counts an hour and within
    #: ``baseline_cap_counts`` of where it started; the data's Baseline
    #: columns say what was in force.  ``baseline`` is True then too.
    adaptive_baseline: bool = False
    baseline_rate_per_hour: float | None = None
    baseline_cap_counts: int | None = None

    @property
    def skipped(self) -> list[str]:
        """:attr:`warnings` without :data:`SIMPLE_WARNING`: the lines skipped
        and sections ignored.  The simple export's warning says what the
        recording is, and is not a line that was skipped."""
        return [w for w in self.warnings if w != SIMPLE_WARNING]

    def section(self, dfm_id: int) -> DFMProgram | None:
        return self.dfms.get(int(dfm_id))

    def interval_for(self, dfm_id: int, k: int) -> ProgramInterval | None:
        """The *k*-th interval line printed for *dfm_id* (1-based, as a 4.0.0
        data file's ``Interval`` column counts), or ``None``.  The export
        omits Intervals that start past the duration and a Repeating
        program's later passes, so a recorded *k* may have no line."""
        section = self.section(dfm_id)
        k = int(k)
        if section is None or not 1 <= k <= len(section.intervals):
            return None
        return section.intervals[k - 1]

    def resolved_end(self) -> datetime | None:
        """``End Time``, else ``Start Time`` plus ``Duration``."""
        if self.end is not None:
            return self.end
        if self.start is not None and self.duration_min is not None:
            return self.start + timedelta(minutes=float(self.duration_min))
        return None

    def mentions(self, dfm_id: int) -> bool:
        """Whether the file has a section for *dfm_id*, usable or not."""
        return int(dfm_id) in self.dfms or int(dfm_id) in self.rejected


# ---------------------------------------------------------------------------
# Finding and reading the file
# ---------------------------------------------------------------------------

def find_program(data_dir: str | Path) -> Path | None:
    """The one ``Program.txt`` in *data_dir* (name matched case-insensitively),
    or ``None``.  More than one is a :class:`ProgramError`: which of them
    describes this recording is not pyflic's guess to make."""
    data_dir = Path(data_dir)
    try:
        entries = sorted(data_dir.iterdir())
    except OSError:
        return None
    found = [p for p in entries
             if p.is_file() and p.name.lower() == PROGRAM_FILENAME.lower()]
    if len(found) > 1:
        names = ", ".join(p.name for p in found)
        raise ProgramError(
            f"{data_dir} holds more than one {PROGRAM_FILENAME} ({names}); keep "
            f"only the one the MCU exported for this recording")
    return found[0] if found else None


def read_program(path: str | Path) -> OptoProgram:
    """Parse the file at *path* (:func:`parse_program`)."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")
    return parse_program(text, path=path)


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(r"^\*{3,}\s*(?P<name>.*?)\s*\*{3,}$")
_DFM_SECTION_RE = re.compile(r"^dfm\s*#?\s*(?P<id>\d+)$", re.IGNORECASE)
_INTERVAL_RE = re.compile(r"^\((?P<stamp>[^)]*)\)\s*(?P<body>.+)$")
_KEYED_RE = re.compile(r"^(?P<key>[A-Za-z])\s*:\s*(?P<value>[-+]?\d+(?:\.\d+)?)\s*[A-Za-z]*$")
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_DARK_RE = re.compile(r"^dark\s*[:=]?\s*(?P<state>on|off|yes|no|true|false|\d+)$",
                      re.IGNORECASE)
_PARADIGMS_RE = re.compile(r"^paradigms\s*:\s*$", re.IGNORECASE)
#: ``NAME (Type): Setting value[ unit], …`` — PyMCU's names start with a
#: letter and hold letters, digits, ``_``, ``-`` and ``.``.
_PARADIGM_LINE_RE = re.compile(
    r"^(?P<name>[A-Za-z][\w.\-]*)\s*\(\s*(?P<type>[^()]*?)\s*\)\s*:\s*(?P<settings>.*)$")
_MINUTES_RE = re.compile(r"^(?P<value>[-+]?\d+(?:\.\d+)?)\s*(?:min)?$", re.IGNORECASE)
_TIME_FORMATS = ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M",
                 "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S")
_PROGRAM_TYPES = ("linear", "repeating", "circadian", "constant")
#: The MCU reserves this id for the Environmental Monitor.
_ENVMON_ID = 99
#: Sections a 4.0.0 export writes that are not DFMs and need no warning.
_ENVMON_SECTION = "env mon"
_ALL_DFM_SECTION = "all dfm"

_HEADER_KEYS = {
    "start time": "start", "end time": "end", "duration": "duration",
    "default linkage": "linkage",
    "default opto frequency": "frequency",
    "default opto pulsewidth": "pulse_width", "default opto pulse width": "pulse_width",
    "default opto delay": "delay", "default opto decay": "decay",
    "default max time on": "max_time_on", "default maxtimeon": "max_time_on",
    "default program type": "program_type",
    "baseline": "baseline",
    "baselinerateperhour": "baseline_rate", "baselinecapcounts": "baseline_cap",
}
_SECTION_KEYS = {
    "linkage": "linkage",
    "opto frequency": "frequency",
    "opto pulsewidth": "pulse_width", "opto pulse width": "pulse_width",
    "opto delay": "delay", "opto decay": "decay",
    "max time on": "max_time_on", "maxtimeon": "max_time_on",
    "program type": "program_type",
}
#: The keys a 4.0.0 export writes; the legacy ``Opto …`` defaults have no
#: meaning beside a ``Paradigms:`` block.
_V4_HEADER = ("start", "end", "duration", "linkage", "program_type", "baseline",
              "baseline_rate", "baseline_cap")
_V4_SECTION = ("linkage", "program_type")
#: What an ``***ALL DFM***`` block states.
_SIMPLE_KEYS = {"start time": "start", "end time": "end", "duration": "duration",
                "linkage": "linkage", "baseline": "baseline",
                "baselinerateperhour": "baseline_rate", "baselinecapcounts": "baseline_cap",
                "program type": "program_type"}
#: Interval keys the legacy export writes after the thresholds.
_INTERVAL_KEYS = {"F": "frequency", "P": "pulse_width", "D": "decay", "L": "delay",
                  "M": "max_time_on"}
_PARAM_ORDER = ("frequency", "pulse_width", "decay", "delay", "max_time_on")


def _norm(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _parse_time(text: str) -> datetime | None:
    text = " ".join(str(text).strip().split())
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _parse_int(text: str) -> int:
    """The leading integer of *text* ("40Hz", "8ms", "1000000000000")."""
    match = _NUMBER_RE.search(str(text))
    if match is None:
        raise ValueError(f"no number in {text!r}")
    value = match.group(0)
    if "." in value:
        number = float(value)
        if number != int(number):
            raise ValueError(f"{text!r} is not a whole number")
        return int(number)
    return int(value)


def _parse_float(text: str) -> float:
    match = _NUMBER_RE.search(str(text))
    if match is None:
        raise ValueError(f"no number in {text!r}")
    return float(match.group(0))


def _parse_linkage(text: str) -> tuple[int, ...]:
    values = [v.strip() for v in str(text).split(",") if v.strip()]
    if len(values) != N_WELLS:
        raise ValueError(f"linkage needs {N_WELLS} values, got {len(values)}")
    return tuple(int(v) for v in values)


def _parse_yes_no(text: str) -> bool:
    value = _norm(text)
    if value in ("yes", "y", "true", "on", "1"):
        return True
    if value in ("no", "n", "false", "off", "0"):
        return False
    raise ValueError(f"expected Yes or No, got {text!r}")


#: What a ``Baseline:`` line can say: Yes, No, or PyMCU's Adaptive (a
#: baseline that is used, and kept up to date).
BASELINE_ADAPTIVE = "adaptive"


def _parse_baseline(text: str) -> bool | str:
    """True, False or :data:`BASELINE_ADAPTIVE`."""
    if _norm(text) == BASELINE_ADAPTIVE:
        return BASELINE_ADAPTIVE
    try:
        return _parse_yes_no(text)
    except ValueError:
        raise ValueError(f"expected Yes, No or Adaptive, got {text!r}") from None


def _parse_program_type(text: str, warnings: list[str], where: str) -> str:
    value = _norm(text)
    if value in _PROGRAM_TYPES:
        return value.capitalize()
    warnings.append(f"{where}: program type {text.strip()!r} is not Linear, "
                    f"Repeating, Circadian or Constant; read as Linear")
    return "Linear"


def _parse_interval(index: int, stamp: str, body: str, defaults: dict,
                    warnings: list[str], where: str) -> ProgramInterval:
    """One legacy ``(timestamp) Dark Off,t1..t12,F:..,P:..,D:..,L:..,M:..,<min>min.``
    line.  Raises ``ValueError`` when the line cannot be a program interval."""
    text = body.strip()
    if text.endswith("."):
        text = text[:-1]
    tokens = [t.strip() for t in text.split(",")]
    if not tokens or not tokens[0]:
        raise ValueError("empty interval")
    head = tokens[0]
    dark_match = _DARK_RE.match(head)
    if dark_match:
        state = dark_match.group("state").lower()
        dark = state in ("on", "yes", "true") or (state.isdigit() and int(state) != 0)
        rest = tokens[1:]
    else:
        try:
            dark = int(head) != 0
        except ValueError:
            raise ValueError(f"dark-running field {head!r} not understood") from None
        rest = tokens[1:]
    if len(rest) < N_WELLS + 1:
        raise ValueError(f"{len(rest)} values after the dark-running field; "
                         f"{N_WELLS} thresholds and a duration are needed")
    try:
        thresholds = tuple(int(v) for v in rest[:N_WELLS])
    except ValueError:
        raise ValueError(f"thresholds {rest[:N_WELLS]} are not all integers") from None
    tail = rest[N_WELLS:]
    keyed: dict[str, int] = {}
    plain: list[str] = []
    for token in tail:
        match = _KEYED_RE.match(token)
        if match is not None:
            key = match.group("key").upper()
            if key not in _INTERVAL_KEYS:
                raise ValueError(f"unknown interval parameter {token!r}")
            keyed[_INTERVAL_KEYS[key]] = _parse_int(match.group("value"))
        else:
            plain.append(token)
    if not plain:
        raise ValueError("no interval duration")
    duration = _parse_float(plain[-1])
    if duration <= 0:
        raise ValueError(f"duration {plain[-1]!r} is not positive")
    positional = plain[:-1]
    if positional and not keyed and len(positional) == len(_PARAM_ORDER):
        ## The authored 19-value form: frequency, pulse width, decay, delay,
        ## max-time-on, then the duration.
        keyed = {k: _parse_int(v) for k, v in zip(_PARAM_ORDER, positional)}
    elif positional:
        raise ValueError(f"values {positional} between the thresholds and the "
                         f"duration are not understood")
    params = {k: keyed.get(k, defaults[k]) for k in _PARAM_ORDER}
    start = _parse_time(stamp) if stamp.strip() else None
    if stamp.strip() and start is None:
        warnings.append(f"{where}: interval {index} timestamp {stamp.strip()!r} not "
                        f"understood; the interval is placed after the one before it")
    return ProgramInterval(index=index, start=start, dark=dark,
                           thresholds=tuple(threshold_from_legacy(t) for t in thresholds),
                           paradigm=paradigm_from_legacy(OptoParams(**params)),
                           duration_min=duration)


# ---- the 4.0.0 grammar -----------------------------------------------------

def _whole(text: str) -> int:
    """A whole number written as ``500`` or ``500.0``."""
    try:
        return int(text)
    except ValueError:
        number = float(text)
        if not number.is_integer():
            raise ValueError(f"{text!r} is not a whole number") from None
        return int(number)


def _parse_setting_value(kind: str, text: str):
    text = text.strip()
    if kind in _UNITS:
        unit = _UNITS[kind].strip().lower()
        if text.lower().endswith(unit):
            text = text[:-len(unit)].strip()
    if kind == "series":
        for series in (SERIES_ARITHMETIC, SERIES_EXPONENTIAL):
            if text.lower() == series.lower():
                return series
        raise ValueError(f"Series must be Arithmetic or Exponential, not {text!r}")
    if kind == "decimal":
        return float(text)
    return _whole(text)


def _parse_paradigm_line(number: int, match: re.Match, warnings: list[str]
                         ) -> Paradigm | None:
    """One ``NAME (Type): Setting value[ unit], …`` line of the ``Paradigms:``
    block, or ``None`` (with a warning) when its type is not one of the six."""
    name = match.group("name")
    kind = type_from_name(match.group("type"))
    if kind is None:
        warnings.append(f"line {number}: Paradigm {name} has type "
                        f"{match.group('type')!r}, not one of "
                        f"{', '.join(TYPE_NAMES.values())}; it is dropped")
        return None
    allowed = {s[0].lower(): s for s in _LIGHT + _TYPE_SETTINGS[kind]}
    values: dict[str, object] = {}
    for item in match.group("settings").split(","):
        item = item.strip()
        if not item:
            continue
        key, _sep, rest = item.partition(" ")
        spec = allowed.get(key.lower())
        if spec is None:
            why = (f"{TYPE_NAMES[kind]} Paradigms have no {key}"
                   if key.lower() in _ALL_SETTINGS else f"unknown setting {key!r}")
            warnings.append(f"line {number}: Paradigm {name}: {why}; ignored")
            continue
        setting, attr, setting_kind = spec
        if attr in values:
            warnings.append(f"line {number}: Paradigm {name}: {setting} is given "
                            f"twice; the first is kept")
            continue
        try:
            values[attr] = _parse_setting_value(setting_kind, rest)
        except ValueError as err:
            warnings.append(f"line {number}: Paradigm {name}: {setting} {rest.strip()!r} "
                            f"not understood ({err}); ignored")
    ## The export prints every setting; these two are the only ones a
    ## Paradigm cannot lack, so a hand-edited file falls back on PyMCU's
    ## defaults for them.
    for attr, setting, default, unit in (("frequency_hz", "Frequency", 40, "Hz"),
                                         ("pulse_width_ms", "PulseWidth", 8, "ms")):
        if attr not in values:
            warnings.append(f"line {number}: Paradigm {name} states no {setting}; "
                            f"taken as {default} {unit}")
            values[attr] = default
    paradigm = Paradigm(name=name, type=kind, **values)
    if _pymcu_generated_name(paradigm) == name:
        paradigm = replace(paradigm, generated=True)
    return paradigm


def _parse_threshold(text: str) -> Threshold:
    value = text.strip()
    if value.lower() == ON.lower():
        return ON
    if value.lower() == OFF.lower():
        return OFF
    try:
        level = int(value)
    except ValueError:
        level = None
    if level is None or not 1 <= level <= MAX_LEVEL:
        raise ValueError(f"thresholds are On, Off or a level from 1 to {MAX_LEVEL}, "
                         f"not {value!r}")
    return level


def _lookup(paradigms: dict[str, Paradigm], name: str) -> Paradigm | None:
    """The Paradigm *name* refers to; PyMCU matches names ignoring case."""
    found = paradigms.get(name)
    if found is None:
        folded = name.lower()
        found = next((p for key, p in paradigms.items() if key.lower() == folded), None)
    return found


def _dark_position(tokens: list[str]) -> int | None:
    """Which of an interval's first two values is a bare ``Dark`` / ``Lit``:
    1 as PyMCU writes it now, after the Paradigm as in a Program file; 0 in
    earlier 4.0.0 exports; ``None`` for neither.  A Paradigm can't be named
    Dark or Lit, so the order is never ambiguous."""
    for at in (1, 0):
        if at < len(tokens) and tokens[at].strip().lower() in ("dark", "lit"):
            return at
    return None


def _parse_v4_interval(index: int, stamp: str, body: str,
                       paradigms: dict[str, Paradigm], warnings: list[str],
                       where: str) -> ProgramInterval:
    """One ``(stamp) NAME, Dark|Lit, t1..t12, N min`` line, or the earlier
    ``Dark|Lit, NAME`` order.  Raises ``ValueError`` when the line cannot be
    a program interval."""
    text = body.strip()
    if text.endswith("."):
        text = text[:-1]
    tokens = [t.strip() for t in text.split(",")]
    if len(tokens) != N_WELLS + 3:
        raise ValueError(f"{len(tokens)} values; an interval is a Paradigm, Dark or Lit, "
                         f"{N_WELLS} thresholds and its minutes")
    at = _dark_position(tokens)
    if at is None:
        raise ValueError(f"neither {tokens[0]!r} nor {tokens[1]!r} is Dark or Lit")
    head, name = tokens[at].lower(), tokens[1 - at]
    paradigm = _lookup(paradigms, name)
    if paradigm is None:
        raise ValueError(f"no Paradigm named {name!r} in the Paradigms block")
    thresholds = tuple(_parse_threshold(t) for t in tokens[2:2 + N_WELLS])
    match = _MINUTES_RE.match(tokens[-1])
    if match is None:
        raise ValueError(f"duration {tokens[-1]!r} is not a number of minutes")
    duration = float(match.group("value"))
    if duration <= 0:
        raise ValueError(f"duration {tokens[-1]!r} is not positive")
    start = _parse_time(stamp) if stamp.strip() else None
    if stamp.strip() and start is None:
        warnings.append(f"{where}: interval {index} timestamp {stamp.strip()!r} not "
                        f"understood; the interval is placed after the one before it")
    return ProgramInterval(index=index, start=start, dark=head == "dark",
                           thresholds=thresholds, paradigm=paradigm,
                           duration_min=duration)


def _grammar_of(lines: list[str]) -> str:
    """Which export *lines* are: a ``Paradigms:`` line or an interval whose
    first or second value is a bare ``Dark`` / ``Lit`` is 4.0.0 (a legacy
    line starts ``Dark Off``, then numbers); an ``***ALL DFM***`` block with
    no DFM section is the simple export; anything else is legacy."""
    v4 = has_dfm = has_all = False
    for raw in lines:
        line = raw.strip().replace("\t", " ")
        if _PARADIGMS_RE.match(line):
            v4 = True
            continue
        section = _SECTION_RE.match(line)
        if section is not None:
            name = section.group("name")
            dfm = _DFM_SECTION_RE.match(name)
            if dfm is not None and int(dfm.group("id")) != _ENVMON_ID:
                has_dfm = True
            elif _norm(name) == _ALL_DFM_SECTION:
                has_all = True
            continue
        interval = _INTERVAL_RE.match(line)
        if interval is not None:
            if _dark_position(interval.group("body").split(",", 2)[:2]) is not None:
                v4 = True
    if has_all and not has_dfm:
        return GRAMMAR_SIMPLE
    return GRAMMAR_V4 if v4 else GRAMMAR_LEGACY


@dataclass
class _Section:
    name: str
    first_line: int
    dfm_id: int | None
    values: dict = field(default_factory=dict)
    intervals: list[tuple[int, str, str]] = field(default_factory=list)
    problem: str | None = None


def parse_program(text: str, *, path: str | Path | None = None) -> OptoProgram:
    """Read an MCU-exported ``Program.txt``, of either grammar.

    Unknown lines are skipped with a warning; a DFM section that cannot be
    read (a malformed linkage or interval, an interval naming a Paradigm the
    block does not define, no interval at all) is dropped and its reason kept
    in :attr:`OptoProgram.rejected`; a file with no readable DFM section at
    all raises :class:`ProgramError` — except the simple export of a
    recording made without a Program, which has none by design and is
    returned with :data:`SIMPLE_WARNING`.
    """
    lines = str(text).splitlines()
    grammar = _grammar_of(lines)
    legacy = grammar == GRAMMAR_LEGACY
    warnings: list[str] = []
    header: dict = {}
    sections: list[_Section] = []
    paradigm_lines: list[tuple[int, re.Match]] = []
    current: _Section | None = None
    ignored: _Section | None = None
    in_simple = in_paradigms = False
    authored = False
    for number, raw in enumerate(lines, start=1):
        line = raw.strip().replace("\t", " ")
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            authored = True
            warnings.append(f"line {number}: {line} is a section of an authored MCU "
                            f"program, not of the exported Program.txt; skipped")
            current = ignored = None
            in_simple = in_paradigms = False
            continue
        section_match = _SECTION_RE.match(line)
        if section_match is not None:
            name = section_match.group("name")
            dfm_match = _DFM_SECTION_RE.match(name)
            in_simple = in_paradigms = False
            if dfm_match is not None and int(dfm_match.group("id")) != _ENVMON_ID:
                current = _Section(name=name, first_line=number,
                                   dfm_id=int(dfm_match.group("id")))
                sections.append(current)
                ignored = None
            elif grammar == GRAMMAR_SIMPLE and _norm(name) == _ALL_DFM_SECTION:
                ## Its settings describe the recording; its interval lines are
                ## every well Off, which is what no section already means.
                current = ignored = None
                in_simple = True
            else:
                current = None
                ignored = _Section(name=name, first_line=number, dfm_id=None)
                if legacy or _norm(name) != _ENVMON_SECTION:
                    warnings.append(f"line {number}: section '{name}' is not a DFM; "
                                    f"its lines are ignored")
            continue
        if ignored is not None:
            continue
        if in_simple:
            key, sep, value = line.partition(":")
            name = _SIMPLE_KEYS.get(_norm(key)) if sep else None
            if name is not None:
                header[name] = (number, value.strip())
            elif "," not in line:
                warnings.append(f"line {number}: not understood, skipped: {line!r}")
            continue
        if not legacy and current is None and _PARADIGMS_RE.match(line):
            in_paradigms = True
            continue
        if in_paradigms and current is None:
            ## Before the interval parser and the key/value split: a Paradigm
            ## line holds commas and a colon of its own.
            paradigm_match = _PARADIGM_LINE_RE.match(line)
            if paradigm_match is not None:
                paradigm_lines.append((number, paradigm_match))
                continue
        interval_match = _INTERVAL_RE.match(line)
        if interval_match is not None:
            if current is None:
                warnings.append(f"line {number}: an interval outside any DFM section; "
                                f"skipped")
            else:
                current.intervals.append((number, interval_match.group("stamp"),
                                          interval_match.group("body")))
            continue
        key, sep, value = line.partition(":")
        if not sep:
            warnings.append(f"line {number}: not understood, skipped: {line!r}")
            continue
        norm = _norm(key)
        if current is None:
            name = _HEADER_KEYS.get(norm)
            if name is not None and not legacy and name not in _V4_HEADER:
                warnings.append(f"line {number}: {key.strip()!r} belongs to the "
                                f"pre-4.0.0 export, not beside a Paradigms block; "
                                f"skipped")
                continue
            if name is None:
                warnings.append(f"line {number}: unknown setting {key.strip()!r}; "
                                f"skipped")
                continue
            header[name] = (number, value.strip())
        else:
            name = _SECTION_KEYS.get(norm)
            if name is not None and not legacy and name not in _V4_SECTION:
                warnings.append(f"line {number} (DFM {current.dfm_id}): "
                                f"{key.strip()!r} belongs to the pre-4.0.0 export; "
                                f"skipped")
                continue
            if name is None:
                warnings.append(f"line {number} (DFM {current.dfm_id}): unknown "
                                f"setting {key.strip()!r}; skipped")
                continue
            current.values[name] = (number, value.strip())

    ## ---- the header ----
    def header_value(name, parse, fallback):
        if name not in header:
            return fallback
        number, value = header[name]
        try:
            return parse(value)
        except ValueError as err:
            warnings.append(f"line {number}: {err}; {name.replace('_', ' ')} ignored")
            return fallback

    start = end = None
    if "start" in header:
        number, value = header["start"]
        start = _parse_time(value)
        if start is None:
            warnings.append(f"line {number}: Start Time {value!r} not understood")
    if "end" in header:
        number, value = header["end"]
        end = _parse_time(value)
        if end is None:
            warnings.append(f"line {number}: End Time {value!r} not understood")
    duration = header_value("duration", _parse_float, None)
    baseline = header_value("baseline", _parse_baseline, True)
    adaptive = baseline == BASELINE_ADAPTIVE
    rate = header_value("baseline_rate", _parse_float, None)
    cap = header_value("baseline_cap", _parse_int, None)
    if not adaptive and (rate is not None or cap is not None):
        warnings.append("BaselineRatePerHour and BaselineCapCounts apply only with "
                        "Baseline: Adaptive; ignored")
        rate = cap = None
    default_linkage = header_value("linkage", _parse_linkage, DEFAULT_LINKAGE)
    default_type = "Linear"
    if "program_type" in header:
        number, value = header["program_type"]
        default_type = _parse_program_type(value, warnings, f"line {number}")
    defaults = {k: header_value(k, _parse_int, FIRMWARE_DEFAULTS[k])
                for k in _PARAM_ORDER}

    program = OptoProgram(path=Path(path) if path is not None else None, start=start,
                          end=end, duration_min=duration, baseline=bool(baseline),
                          warnings=warnings, grammar=grammar, adaptive_baseline=adaptive,
                          baseline_rate_per_hour=rate, baseline_cap_counts=cap)
    if grammar == GRAMMAR_SIMPLE:
        warnings.append(SIMPLE_WARNING)
        return program

    ## ---- the Paradigms block ----
    for number, match in paradigm_lines:
        paradigm = _parse_paradigm_line(number, match, warnings)
        if paradigm is None:
            continue
        if _lookup(program.paradigms, paradigm.name) is not None:
            warnings.append(f"line {number}: a second Paradigm named {paradigm.name}; "
                            f"the first is kept")
            continue
        program.paradigms[paradigm.name] = paradigm

    ## ---- the DFM sections ----
    for section in sections:
        dfm_id = int(section.dfm_id)
        where = f"DFM {dfm_id}"
        if dfm_id in program.dfms or dfm_id in program.rejected:
            warnings.append(f"line {section.first_line}: a second section for DFM "
                            f"{dfm_id}; the first is kept")
            continue
        try:
            linkage = default_linkage
            if "linkage" in section.values:
                linkage = _parse_linkage(section.values["linkage"][1])
            program_type = default_type
            if "program_type" in section.values:
                program_type = _parse_program_type(section.values["program_type"][1],
                                                   warnings, where)
            section_defaults = dict(defaults)
            for name in _PARAM_ORDER:
                if name in section.values:
                    section_defaults[name] = _parse_int(section.values[name][1])
            if not section.intervals:
                raise ValueError("no interval lines")
            intervals = []
            for index, (number, stamp, body) in enumerate(section.intervals, start=1):
                try:
                    if legacy:
                        interval = _parse_interval(index, stamp, body, section_defaults,
                                                   warnings, where)
                    else:
                        interval = _parse_v4_interval(index, stamp, body,
                                                      program.paradigms, warnings, where)
                except ValueError as err:
                    raise ValueError(f"interval {index} (line {number}): {err}") from None
                intervals.append(interval)
        except ValueError as err:
            program.rejected[dfm_id] = str(err)
            warnings.append(f"DFM {dfm_id}: section dropped — {err}")
            continue
        program.dfms[dfm_id] = DFMProgram(dfm_id=dfm_id, linkage=tuple(linkage),
                                          program_type=program_type,
                                          intervals=tuple(intervals))
        if legacy:
            for interval in intervals:
                program.paradigms.setdefault(interval.paradigm.name, interval.paradigm)
    if not program.dfms:
        where = f"{program.path.name}: " if program.path is not None else ""
        if authored:
            raise ProgramError(
                f"{where}this looks like an authored MCU program ([General] / [DFM] "
                f"sections).  pyflic reads the Program.txt the MCU exports for a run, "
                f"whose sections read ***DFM 1***")
        detail = "; ".join(f"DFM {k}: {v}" for k, v in sorted(program.rejected.items()))
        raise ProgramError(f"{where}no DFM section could be read"
                           + (f" ({detail})" if detail else ""))
    return program


# ---------------------------------------------------------------------------
# Small helpers shared by the QC, the reports and the editors
# ---------------------------------------------------------------------------

def wells_label(wells: Iterable[int]) -> str:
    """``W1-W4`` for a contiguous run, ``W1, W3`` otherwise; ``—`` for none."""
    ws = sorted({int(w) for w in wells})
    if not ws:
        return "—"
    runs: list[tuple[int, int]] = []
    for w in ws:
        if runs and w == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], w)
        else:
            runs.append((w, w))
    return ", ".join(f"W{a}" if a == b else (f"W{a}, W{b}" if b == a + 1 else f"W{a}-W{b}")
                     for a, b in runs)


def linkage_label(dfm_program: DFMProgram) -> str:
    """The linkage as groups, e.g. ``1: W1-W4 · 2: W5-W8 · 3: W9-W12``, or
    ``none`` when every well is its own group."""
    groups = dfm_program.linkage_groups()
    if all(len(wells) == 1 for wells in groups.values()):
        return "none"
    return " · ".join(f"{label}: {wells_label(wells)}"
                      for label, wells in groups.items() if len(wells) > 1) +         (" · the rest unlinked" if any(len(w) == 1 for w in groups.values()) else "")
