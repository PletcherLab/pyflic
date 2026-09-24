"""Progressive Ratio light QC: did the Paired fly earn its light?

The DFM firmware switches a Chamber Group's light from its own reading of the
Paired chamber's Sucrose Well during the recording.  pyflic counts licks only
afterwards, from the *baselined* signal.  The two disagree exactly when the
assay fails: a Sucrose Well whose resting level creeps up looks continuously
touched to the firmware, which then fires the light on its own schedule, while
the baselined signal is flat and pyflic records no licks at all.  The light
looks earned in every table built on light-on time, and nothing was earned.

This module holds the arithmetic of that check on plain arrays; the
experiment assembles it per Chamber Group
(:meth:`ProgressiveRatioExperiment.light_qc_table`).  Terms, fixed in
``CONTEXT.md``:

* a **Light Event** is one onset of the group's light;
* a **Lick-free Light Event** has no Sucrose Well licks since the previous
  Light Event ended, and no Sucrose Well activity (a lick or a touch) within
  the light's decay window around it — the decay the Opto Program states,
  so a light the fly did touch for is never counted against it;
* **Self-triggered light** is a run of at least ``pr_lick_free_run``
  consecutive Lick-free Light Events in the Test phase — the light was
  following the sensor, not the fly;
* the **Resting Level** of a well is its raw (un-baselined) signal between
  licks, measured as the per-minute median.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

#: Design ``constants:`` read by the light QC, with their defaults.  The
#: Progressive Ratio type merges these into its ``default_constants``.
DEFAULT_CONSTANTS: dict[str, Any] = {
    "exclude_failed_pr_groups": True,
    "pr_lick_free_run": 5,
    "pr_trend_min_events": 5,
    "pr_trend_min_rho": 0.3,
    "pr_resting_level_rise": 15,
    "pr_resting_level_ratio": 3,
}

SELF_TRIGGERED = "self-triggered light"
IMPLAUSIBLE_TRAINING = "implausible training"
NO_TREND = "no increasing trend"
RESTING_RISE = "resting level rise"
RESTING_ELEVATED = "resting level elevated"

#: Flags that fail a Chamber Group.  The others are warnings: a drifting
#: Resting Level is the precursor of a failure and a flat lick trend can be a
#: fly that simply stopped trying, so neither is proof on its own.
FAILING_FLAGS: tuple[str, ...] = (SELF_TRIGGERED, IMPLAUSIBLE_TRAINING)
WARNING_FLAGS: tuple[str, ...] = (NO_TREND, RESTING_RISE, RESTING_ELEVATED)

VERDICT_OK = "ok"
VERDICT_WARNING = "warning"
VERDICT_FAILED = "failed"

#: Minutes over which the start, end and peak of a Resting Level are taken.
RESTING_WINDOW_MIN = 30

#: Test Light Events the requirement increment is estimated from.
INCREMENT_EVENTS = 8


def _truthy(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(value)


@dataclass(frozen=True, slots=True)
class LightQCSettings:
    """The light QC's thresholds, resolved from the design's ``constants:``."""

    exclude: bool = True
    lick_free_run: int = 5
    trend_min_events: int = 5
    trend_min_rho: float = 0.3
    resting_rise: float = 15.0
    resting_ratio: float = 3.0

    @classmethod
    def from_constants(cls, constants: Mapping[str, Any] | None) -> LightQCSettings:
        merged = {**DEFAULT_CONSTANTS, **dict(constants or {})}

        def number(key: str) -> float:
            try:
                return float(merged[key])
            except (TypeError, ValueError):
                return float(DEFAULT_CONSTANTS[key])

        return cls(
            exclude=_truthy(merged["exclude_failed_pr_groups"]),
            lick_free_run=max(1, int(number("pr_lick_free_run"))),
            trend_min_events=max(3, int(number("pr_trend_min_events"))),
            trend_min_rho=number("pr_trend_min_rho"),
            resting_rise=number("pr_resting_level_rise"),
            resting_ratio=number("pr_resting_level_ratio"),
        )

    def describe(self) -> str:
        return (f"exclude_failed_pr_groups={'true' if self.exclude else 'false'}, "
                f"pr_lick_free_run={self.lick_free_run}, "
                f"pr_trend_min_events={self.trend_min_events}, "
                f"pr_trend_min_rho={self.trend_min_rho:g}, "
                f"pr_resting_level_rise={self.resting_rise:g}, "
                f"pr_resting_level_ratio={self.resting_ratio:g}")


# ---------------------------------------------------------------------------
# Light Events and the licks between them
# ---------------------------------------------------------------------------

def light_events(light: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(onsets, ends)`` of every Light Event in a boolean per-sample *light*.

    ``onsets[k]`` is the first lit sample of event *k*; ``ends[k]`` the first
    dark sample after it, or ``len(light)`` for an event still lit when the
    recording stops.
    """
    lit = np.asarray(light, dtype=bool)
    if lit.size == 0:
        empty = np.zeros(0, dtype=int)
        return empty, empty
    before = np.concatenate([[False], lit[:-1]])
    onsets = np.flatnonzero(lit & ~before)
    after = np.concatenate([lit[1:], [False]])
    ends = np.flatnonzero(lit & ~after) + 1
    return onsets.astype(int), ends.astype(int)


def licks_between_events(licks: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Lick samples credited to each Light Event.

    Event *k* is credited with every lick from the end of event *k − 1* (the
    start of the recording for the first) to the end of event *k*.  The
    windows tile the recording up to the last event, so each lick counts once;
    ending at the event's *end* rather than its onset keeps the lick that
    triggered it even when the detector dates that lick to the first lit
    sample, which would otherwise read as a Lick-free Light Event.
    """
    ends = np.asarray(ends, dtype=int)
    if ends.size == 0:
        return np.zeros(0, dtype=int)
    cum = np.concatenate([[0], np.cumsum(np.asarray(licks, dtype=bool), dtype=np.int64)])
    starts = np.concatenate([[0], ends[:-1]])
    return (cum[ends] - cum[starts]).astype(int)


def longest_run(mask: np.ndarray) -> tuple[int, int | None]:
    """``(length, start)`` of the longest run of True in *mask*; ``(0, None)``
    when there is none.  Ties go to the earliest run."""
    best, best_start, cur, start = 0, None, 0, 0
    for i, value in enumerate(np.asarray(mask, dtype=bool)):
        if value:
            if cur == 0:
                start = i
            cur += 1
            if cur > best:
                best, best_start = cur, start
        else:
            cur = 0
    return best, best_start


def first_run_start(mask: np.ndarray, length: int) -> int | None:
    """Index where the first run of at least *length* True values begins."""
    cur, start = 0, 0
    for i, value in enumerate(np.asarray(mask, dtype=bool)):
        if value:
            if cur == 0:
                start = i
            cur += 1
            if cur >= length:
                return start
        else:
            cur = 0
    return None


def lick_trend(counts: np.ndarray) -> tuple[float, float]:
    """``(rho, slope)``: Spearman's rho of licks per event against event number,
    and the least-squares slope in licks per event.

    Rho is the verdict because it asks only "does the requirement keep
    rising?" — the firmware's increment is user-defined and pyflic under-counts
    brief touches, so no particular slope can be expected.  A constant series
    has no trend and gives ``rho = nan``.
    """
    y = np.asarray(counts, dtype=float)
    if y.size < 2:
        return float("nan"), float("nan")
    x = np.arange(1, y.size + 1, dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    if np.all(y == y[0]):
        return float("nan"), slope
    rho = float(pd.Series(y).corr(pd.Series(x), method="spearman"))
    return rho, slope


# ---------------------------------------------------------------------------
# Resting Level
# ---------------------------------------------------------------------------

def resting_levels(raw: pd.DataFrame, minutes: np.ndarray) -> pd.DataFrame:
    """Per-minute median of every ``W*`` column of *raw*, indexed by whole
    minute.  A median over a minute ignores licks, which are brief, so what is
    left is the level the well sits at between them."""
    wells = [c for c in raw.columns if str(c).startswith("W") and str(c)[1:].isdigit()]
    if not wells or len(raw) == 0:
        return pd.DataFrame(columns=wells)
    minute = np.floor(np.asarray(minutes, dtype=float)).astype(int)
    frame = raw[wells].apply(pd.to_numeric, errors="coerce")
    out = frame.groupby(minute).median()
    out.index.name = "Minute"
    return out


def resting_summary(level: pd.Series) -> dict[str, float]:
    """Start, peak and end of a per-minute Resting Level.

    ``start`` is the median over the first :data:`RESTING_WINDOW_MIN` minutes,
    ``end`` over the last as many, and ``max`` the highest centred rolling
    median of that width — so a single minute of sustained contact, a fly
    standing on the well, does not read as the well's level.
    """
    s = pd.to_numeric(level, errors="coerce").dropna()
    if s.empty:
        return {"start": float("nan"), "max": float("nan"), "end": float("nan"),
                "level": float("nan")}
    first, last = s.index.min(), s.index.max()
    start = float(s[s.index < first + RESTING_WINDOW_MIN].median())
    end = float(s[s.index > last - RESTING_WINDOW_MIN].median())
    rolled = s.rolling(RESTING_WINDOW_MIN, center=True,
                       min_periods=max(1, RESTING_WINDOW_MIN // 2)).median()
    peak = float(rolled.max()) if rolled.notna().any() else float(s.max())
    return {"start": start, "max": max(peak, start), "end": end,
            "level": float(s.median())}


def resting_ratio(level: float, reference: float) -> float:
    """*level* over *reference*, the reference floored at one count so a DFM
    whose wells rest near zero does not turn noise into a large ratio."""
    if not np.isfinite(level) or not np.isfinite(reference):
        return float("nan")
    return float(level) / max(float(reference), 1.0)


# ---------------------------------------------------------------------------
# The verdict for one Chamber Group
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class GroupVerdict:
    flags: tuple[str, ...]
    notes: tuple[str, ...]
    lick_free_events: int
    longest_lick_free_run: int
    lick_free_run_start: int | None     # Test event index where the failing run begins
    trend_rho: float
    trend_slope: float

    @property
    def verdict(self) -> str:
        if any(f in FAILING_FLAGS for f in self.flags):
            return VERDICT_FAILED
        return VERDICT_WARNING if self.flags else VERDICT_OK

    @property
    def failed(self) -> bool:
        return self.verdict == VERDICT_FAILED


def judge_group(
    settings: LightQCSettings,
    *,
    training_complete: bool,
    training_light_events: int,
    training_licks: int,
    test_counts: np.ndarray,
    resting: Mapping[str, float],
    reference_level: float,
    any_light: bool,
    lick_free: np.ndarray | None = None,
) -> GroupVerdict:
    """Flags and notes for one Chamber Group from its measured quantities.

    *lick_free* marks the Lick-free Light Events among the Test events; by
    default an event with no credited licks.  The experiment passes the
    decay-aware mask (``LickFree`` of the light events table)."""
    flags: list[str] = []
    notes: list[str] = []
    counts = np.asarray(test_counts, dtype=int)
    lick_free = (counts == 0) if lick_free is None else np.asarray(lick_free, dtype=bool)
    run, run_first = longest_run(lick_free)
    failing_start = first_run_start(lick_free, settings.lick_free_run)
    rho, slope = lick_trend(counts)

    if not any_light:
        notes.append("no light events recorded for this group (no OptoCol1 "
                     "light bits), so the light checks do not apply")
    elif not training_complete:
        notes.append("training never completed, so there is no Test phase to check")
    else:
        ## The firmware credits a training pairing to a touch pyflic may not
        ## count as a lick (a brief contact under the feeding threshold), so
        ## fewer licks than pairings is normal.  None at all is not: every
        ## pairing then came from the sensor alone.
        if training_light_events > 0 and training_licks == 0:
            flags.append(IMPLAUSIBLE_TRAINING)
        if counts.size == 0:
            notes.append("no Test light events — the fly stopped before the "
                         "first Test requirement (a breaking point, not a failure)")
        else:
            if failing_start is not None:
                flags.append(SELF_TRIGGERED)
            if counts.size >= settings.trend_min_events:
                if not (np.isfinite(rho) and rho >= settings.trend_min_rho):
                    flags.append(NO_TREND)
            else:
                notes.append(f"{counts.size} Test light event(s); "
                             f"{settings.trend_min_events} are needed for a "
                             f"trend verdict")

    rise = float(resting.get("max", np.nan)) - float(resting.get("start", np.nan))
    if np.isfinite(rise) and rise >= settings.resting_rise:
        flags.append(RESTING_RISE)
    level = float(resting.get("level", np.nan))
    ratio = resting_ratio(level, reference_level)
    ## Both a ratio and a margin: a DFM whose wells rest at 2-5 counts would
    ## otherwise call a well at 12 "elevated" on noise alone.
    if (np.isfinite(ratio) and ratio >= settings.resting_ratio
            and level - float(reference_level) >= settings.resting_rise):
        flags.append(RESTING_ELEVATED)

    return GroupVerdict(
        flags=tuple(flags), notes=tuple(notes),
        lick_free_events=int(lick_free.sum()),
        longest_lick_free_run=int(run),
        lick_free_run_start=failing_start,
        trend_rho=rho, trend_slope=slope,
    )


def estimate_increment(series: list[np.ndarray]) -> tuple[float, float, int] | None:
    """``(slope, intercept, n_groups)`` of the requirement, from the first
    :data:`INCREMENT_EVENTS` Test Light Events of every group that has that
    many and a rising fit; ``None`` when no group qualifies.

    Medians across groups, so one group that under-counts does not move it.
    A reference line and a summary line only — no verdict depends on it.
    """
    slopes, intercepts = [], []
    for counts in series:
        y = np.asarray(counts, dtype=float)[:INCREMENT_EVENTS]
        if y.size < INCREMENT_EVENTS:
            continue
        x = np.arange(1, y.size + 1, dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        if slope > 0:
            slopes.append(float(slope))
            intercepts.append(float(intercept))
    if not slopes:
        return None
    return float(np.median(slopes)), float(np.median(intercepts)), len(slopes)
