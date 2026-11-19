"""The signal ceiling: a well resting at the top of its sensor's range.

A DFM reads each well through a 10-bit converter, so its signal cannot go
above 1023, and the analogue front end of many boards saturates a little
below that: the test recordings sit at 1014, 1016, 1018, 1022 or 1023,
always the same level on the same board.  A well whose **Resting Level** (the
per-minute median of its raw signal, :func:`pr_light_qc.resting_levels`)
reaches that top cannot rise any further when the fly touches it, so no lick
or touch shows: the minutes it spends there are lost from the feeding record,
however the baseline is subtracted.  A wet pad, food wicking onto it, or
condensation climbs there over hours; a board whose every well is there at
once is the DFM itself, not one pad.

A well is **at the ceiling** in a minute whose Resting Level is
:data:`CEILING_LEVEL` or more: below every board's plateau, and above any
level a well rests at and still reads a lick over ``feeding_threshold``.  A
median over a minute ignores licks, so a fly's brief contacts never count.

Every DFM's integrity report lists its wells' minutes at the ceiling, and
auto-removal takes out a chamber with a well at the ceiling longer than
``max_ceiling_minutes`` of the analysed range (default
:data:`DEFAULT_MAX_MINUTES`, every Experiment Type's; ``null`` turns it off).
The minutes are counted inside the range, so a chamber whose pad saturated at
hour 10 leaves the whole recording and stays in a Slice that ends at hour 6.

Imports nothing from pyflic but :mod:`constant_fields` and
:mod:`windowing`, so the DFM, the experiment and the editors may all use it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .constant_fields import ConstantField
from .constant_fields import constant_problems as _constant_problems

#: A Resting Level at or above this is at the ceiling.
CEILING_LEVEL = 1000.0

#: The Design constant, and its default for every Experiment Type.
KEY = "max_ceiling_minutes"
DEFAULT_MAX_MINUTES = 30
DEFAULT_CONSTANTS: dict[str, Any] = {KEY: DEFAULT_MAX_MINUTES}

#: The editors' and the validation's description of the constant.  ``null``
#: turns the removal off; the integrity report lists the minutes either way.
CEILING_CONSTANT_FIELDS: tuple[ConstantField, ...] = (
    ConstantField(
        KEY, "Max minutes at the signal ceiling",
        f"{KEY} — remove a chamber when any of its wells rests at the signal "
        f"ceiling (a per-minute median of {CEILING_LEVEL:g} or more, the top of the "
        f"sensor's range, where no lick can show) for longer than this many minutes "
        f"of the analysed range.  Default {DEFAULT_MAX_MINUTES}; unchecked is off.",
        "cutoff", minimum=0, blank="off"),
)


@dataclass(frozen=True, slots=True)
class WellCeiling:
    """One well's time at the ceiling within a range."""

    well: int
    #: Minutes whose Resting Level was at the ceiling.
    minutes: int
    #: The first and last such minute (recording minutes, the minute's start).
    first: float
    last: float
    #: The median Resting Level over those minutes: the board's plateau.
    level: float

    def describe(self) -> str:
        span = (f"minute {self.first:g}" if self.first == self.last
                else f"minutes {self.first:g}-{self.last:g}")
        return (f"W{self.well}: {self.minutes} min at the ceiling ({span}), "
                f"resting at {self.level:.0f}")


def ceiling_problems(constants: Mapping[str, Any] | None) -> list[str]:
    """What is wrong with ``max_ceiling_minutes`` in a ``constants:`` block —
    not a number, or below 0.  ``null`` is allowed: it turns the removal off.
    Never raises."""
    return _constant_problems(CEILING_CONSTANT_FIELDS, constants)


def max_minutes(constants: Mapping[str, Any] | None) -> float | None:
    """The resolved ``max_ceiling_minutes``: the default when the block does
    not state it, ``None`` (off) when it states ``null``."""
    stated = dict(constants or {})
    if KEY not in stated:
        return float(DEFAULT_MAX_MINUTES)
    value = stated[KEY]
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(DEFAULT_MAX_MINUTES)


def _in_range(levels: pd.DataFrame, range_minutes) -> pd.DataFrame:
    """The minutes of *levels* inside *range_minutes*, by each minute's
    middle (minute ``m`` holds the samples from ``m`` to ``m + 1``), the way
    the feeding summary windows its rows."""
    from . import windowing

    if levels.empty:
        return levels
    mask = windowing.range_mask(np.asarray(levels.index, dtype=float) + 0.5, range_minutes)
    return levels if mask is None else levels[mask]


def at_ceiling(levels: pd.DataFrame) -> pd.DataFrame:
    """``True`` where a per-minute Resting Level is at the ceiling."""
    return levels.apply(pd.to_numeric, errors="coerce") >= CEILING_LEVEL


def well_ceilings(levels: pd.DataFrame,
                  range_minutes: Sequence[float] = (0, 0)) -> dict[int, WellCeiling]:
    """Every well of *levels* (per-minute Resting Levels, one ``W<n>`` column
    per well) with at least one minute at the ceiling inside *range_minutes*,
    by well number."""
    levels = _in_range(levels, range_minutes)
    hit = at_ceiling(levels)
    out: dict[int, WellCeiling] = {}
    for col in levels.columns:
        rows = hit[col].to_numpy(dtype=bool)
        if not rows.any():
            continue
        minutes = np.asarray(levels.index, dtype=float)[rows]
        values = pd.to_numeric(levels[col], errors="coerce").to_numpy(dtype=float)[rows]
        well = int(str(col)[1:])
        out[well] = WellCeiling(well=well, minutes=int(rows.sum()),
                                first=float(minutes.min()), last=float(minutes.max()),
                                level=float(np.median(values)))
    return out


def board_minutes(levels: pd.DataFrame, range_minutes: Sequence[float] = (0, 0)) -> int:
    """Minutes inside *range_minutes* with every well of the DFM at the
    ceiling together: the board, not a pad."""
    levels = _in_range(levels, range_minutes)
    if levels.empty or levels.shape[1] == 0:
        return 0
    return int(at_ceiling(levels).all(axis=1).sum())


def removal_reasons(ceilings: Mapping[int, WellCeiling], wells: Sequence[int],
                    limit: float | None) -> list[str]:
    """Why a chamber of *wells* leaves under ``max_ceiling_minutes`` =
    *limit*, or ``[]``: one reason per well at the ceiling longer than it."""
    if limit is None:
        return []
    reasons = []
    for well in wells:
        c = ceilings.get(int(well))
        if c is not None and c.minutes > float(limit):
            reasons.append(f"W{c.well} at the signal ceiling {c.minutes} min "
                           f"> {KEY}={float(limit):g}")
    return reasons


def criteria_line(limit: float | None) -> str:
    """``summary.txt``'s filter-section line for the cutoff."""
    if limit is None:
        return (f"  • {KEY}: off (null) — wells at the signal ceiling are reported in "
                f"the integrity report, not removed.")
    return (f"  • {KEY} = {limit:g}\n"
            f"    Excluded if any well rests at the signal ceiling (a per-minute "
            f"median of {CEILING_LEVEL:g} or more) for more than {limit:g} min")
