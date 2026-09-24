"""The Opto Program: what the MCU ran on each DFM, read from ``Program.txt``.

The DFM firmware decides when an Optolid LED is on from settings the
stand-alone MCU loads at the start of a run: a signal **threshold** per well, a
**linkage** that ORs wells together, and five optogenetic parameters —
frequency, pulse width, decay, delay and max-time-on — whose combination
selects a paradigm (open loop, closed loop, fixed interval, progressive ratio,
non-feeding activation).  None of it is in the DFM CSVs, which record only the
outcome: ``OptoCol1``, the LEDs actually lit at each sample.

The MCU writes the program back out as ``Program.txt``, echoing every interval
as it parsed it — ``F:``, ``P:``, ``D:``, ``L:``, ``M:`` are frequency, pulse
width, decay, delay and max-time-on — and the experimenter copies that file into
``data/`` beside the CSVs.  This module reads that export (never the authored
``[General]`` / ``[DFM]`` program, which is the MCU's input, not a record of
what ran) into an :class:`OptoProgram`.  Terms, fixed in ``CONTEXT.md``:

* the **Opto Program** is the parsed file;
* a **Linkage Group** is a set of wells sharing one linkage number, lit
  together whenever any member is triggered;
* a **Trigger Well** is a well whose threshold is positive in some interval,
  so its signal can switch its group's light on.

Semantics are the MCU/DFM version 2.0 conventions ("Stand Alone MCU Programming
Conventions"), with two encodings decoded here: the high seven bits of the
frequency carry the progressive ratio's acclimation-event count, and the top
bit of the pulse width selects non-feeding activation.

Pure Python, no pandas: the parser is imported by the loader, the linter and
the editors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

PROGRAM_FILENAME = "Program.txt"
N_WELLS = 12

#: Low bits of the frequency value hold the frequency in Hz; the seven above
#: them hold the number of acclimation events of an event-trained progressive
#: ratio (so ``5160`` is 10 events at 40 Hz).
FREQUENCY_BITS = 9
#: The top bit of the pulse width selects non-feeding activation (lights off
#: only while the fly feeds); the pulse width proper is the value below it.
INVERT_BIT = 0x8000

#: Firmware defaults for a parameter no line states.
FIRMWARE_DEFAULTS = {"frequency": 40, "pulse_width": 8, "decay": 0, "delay": 0,
                     "max_time_on": 0}
DEFAULT_LINKAGE: tuple[int, ...] = tuple(range(1, N_WELLS + 1))

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
# Paradigms
# ---------------------------------------------------------------------------

OPEN_LOOP = "open loop"
CLOSED_LOOP = "closed loop"
CLOSED_LOOP_MAX = "closed loop, max time on"
FIXED_INTERVAL = "fixed interval"
PROGRESSIVE_RATIO = "progressive ratio"
NON_FEEDING = "non-feeding activation"
LIGHTS_OFF = "off"
UNKNOWN_PARADIGM = "unknown"

#: Paradigms whose light is switched on by the fly's contact with a Trigger
#: Well — the ones the light QC can explain from licks.
FEEDING_PARADIGMS: tuple[str, ...] = (CLOSED_LOOP, CLOSED_LOOP_MAX, FIXED_INTERVAL,
                                      PROGRESSIVE_RATIO)
#: Paradigms in which every feeding bout should light the group.
CLOSED_LOOP_PARADIGMS: tuple[str, ...] = (CLOSED_LOOP, CLOSED_LOOP_MAX)


class ProgramError(ValueError):
    """A ``Program.txt`` pyflic cannot use, or more than one of them."""


@dataclass(frozen=True, slots=True)
class OptoParams:
    """One interval's optogenetic parameters as the MCU holds them.

    *frequency* and *pulse_width* are the raw values, encodings included;
    the properties decode them.  *delay* and *max_time_on* are milliseconds
    except under a progressive ratio, where they are the lick increment and
    the lick cap.
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
        """The paradigm this parameter set selects for a well whose threshold
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


@dataclass(frozen=True, slots=True)
class ProgramInterval:
    """One interval line of a DFM section, as the MCU echoed it."""

    index: int                         # 1-based position in its DFM section
    start: datetime | None             # the timestamp the export printed
    dark: bool                         # the DFM ran dark (indicator LEDs off)
    thresholds: tuple[int, ...]        # W1..W12
    params: OptoParams
    duration_min: float

    def threshold(self, well: int) -> int:
        return int(self.thresholds[int(well) - 1])

    def trigger_wells(self, wells: Iterable[int] | None = None) -> tuple[int, ...]:
        """Wells (of *wells*, default all twelve) whose threshold is positive."""
        pool = range(1, N_WELLS + 1) if wells is None else wells
        return tuple(int(w) for w in pool if self.threshold(w) > 0)

    def open_wells(self, wells: Iterable[int] | None = None) -> tuple[int, ...]:
        pool = range(1, N_WELLS + 1) if wells is None else wells
        return tuple(int(w) for w in pool if self.threshold(w) == 0)

    def group_mode(self, wells: Iterable[int]) -> str:
        """What a Linkage Group of *wells* does during this interval.

        A zero threshold on any member lights the whole group constantly (open
        loop); every member negative keeps it dark (off); otherwise the group
        answers to its Trigger Wells under this interval's paradigm.
        """
        wells = tuple(int(w) for w in wells)
        if any(self.threshold(w) == 0 for w in wells):
            return OPEN_LOOP
        if all(self.threshold(w) < 0 for w in wells):
            return LIGHTS_OFF
        return self.params.paradigm


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
        """Every well whose threshold is positive in at least one interval."""
        found = {w for iv in self.intervals for w in iv.trigger_wells()}
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

    def section(self, dfm_id: int) -> DFMProgram | None:
        return self.dfms.get(int(dfm_id))

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
_TIME_FORMATS = ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M",
                 "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S")
_PROGRAM_TYPES = ("linear", "repeating", "circadian", "constant")
#: The MCU reserves this id for the Environmental Monitor.
_ENVMON_ID = 99

_HEADER_KEYS = {
    "start time": "start", "end time": "end", "duration": "duration",
    "default linkage": "linkage",
    "default opto frequency": "frequency",
    "default opto pulsewidth": "pulse_width", "default opto pulse width": "pulse_width",
    "default opto delay": "delay", "default opto decay": "decay",
    "default max time on": "max_time_on", "default maxtimeon": "max_time_on",
    "default program type": "program_type",
    "baseline": "baseline",
}
_SECTION_KEYS = {
    "linkage": "linkage",
    "opto frequency": "frequency",
    "opto pulsewidth": "pulse_width", "opto pulse width": "pulse_width",
    "opto delay": "delay", "opto decay": "decay",
    "max time on": "max_time_on", "maxtimeon": "max_time_on",
    "program type": "program_type",
}
#: Interval keys the export writes after the thresholds.
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


def _parse_program_type(text: str, warnings: list[str], where: str) -> str:
    value = _norm(text)
    if value in _PROGRAM_TYPES:
        return value.capitalize()
    warnings.append(f"{where}: program type {text.strip()!r} is not Linear, "
                    f"Repeating, Circadian or Constant; read as Linear")
    return "Linear"


def _parse_interval(index: int, stamp: str, body: str, defaults: dict,
                    warnings: list[str], where: str) -> ProgramInterval:
    """One ``(timestamp) Dark Off,t1..t12,F:..,P:..,D:..,L:..,M:..,<min>min.``
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
    return ProgramInterval(index=index, start=start, dark=dark, thresholds=thresholds,
                           params=OptoParams(**params), duration_min=duration)


@dataclass
class _Section:
    name: str
    first_line: int
    dfm_id: int | None
    values: dict = field(default_factory=dict)
    intervals: list[tuple[int, str, str]] = field(default_factory=list)
    problem: str | None = None


def parse_program(text: str, *, path: str | Path | None = None) -> OptoProgram:
    """Read an MCU-exported ``Program.txt``.

    Unknown lines are skipped with a warning; a DFM section that cannot be
    read (a malformed linkage or interval, no interval at all) is dropped and
    its reason kept in :attr:`OptoProgram.rejected`; a file with no readable
    DFM section at all raises :class:`ProgramError`.
    """
    warnings: list[str] = []
    header: dict = {}
    sections: list[_Section] = []
    current: _Section | None = None
    ignored: _Section | None = None
    authored = False
    for number, raw in enumerate(str(text).splitlines(), start=1):
        line = raw.strip().replace("\t", " ")
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            authored = True
            warnings.append(f"line {number}: {line} is a section of an authored MCU "
                            f"program, not of the exported Program.txt; skipped")
            current = ignored = None
            continue
        section_match = _SECTION_RE.match(line)
        if section_match is not None:
            name = section_match.group("name")
            dfm_match = _DFM_SECTION_RE.match(name)
            if dfm_match is not None and int(dfm_match.group("id")) != _ENVMON_ID:
                current = _Section(name=name, first_line=number,
                                   dfm_id=int(dfm_match.group("id")))
                sections.append(current)
                ignored = None
            else:
                current = None
                ignored = _Section(name=name, first_line=number, dfm_id=None)
                warnings.append(f"line {number}: section '{name}' is not a DFM; "
                                f"its lines are ignored")
            continue
        if ignored is not None:
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
            if name is None:
                warnings.append(f"line {number}: unknown setting {key.strip()!r}; "
                                f"skipped")
                continue
            header[name] = (number, value.strip())
        else:
            name = _SECTION_KEYS.get(norm)
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
    baseline = header_value("baseline", _parse_yes_no, True)
    default_linkage = header_value("linkage", _parse_linkage, DEFAULT_LINKAGE)
    default_type = "Linear"
    if "program_type" in header:
        number, value = header["program_type"]
        default_type = _parse_program_type(value, warnings, f"line {number}")
    defaults = {k: header_value(k, _parse_int, FIRMWARE_DEFAULTS[k])
                for k in _PARAM_ORDER}

    ## ---- the DFM sections ----
    program = OptoProgram(path=Path(path) if path is not None else None, start=start,
                          end=end, duration_min=duration, baseline=bool(baseline),
                          warnings=warnings)
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
                    intervals.append(_parse_interval(index, stamp, body, section_defaults,
                                                     warnings, where))
                except ValueError as err:
                    raise ValueError(f"interval {index} (line {number}): {err}") from None
        except ValueError as err:
            program.rejected[dfm_id] = str(err)
            warnings.append(f"DFM {dfm_id}: section dropped — {err}")
            continue
        program.dfms[dfm_id] = DFMProgram(dfm_id=dfm_id, linkage=tuple(linkage),
                                          program_type=program_type,
                                          intervals=tuple(intervals))
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
