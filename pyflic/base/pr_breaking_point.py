"""Progressive Ratio breaking point and Sucrose Persistence (ADR-0014).

Both answer one question — when did the fly stop? — asked of two response
streams with one rule, the **first-gap rule**.  Responses are read in order
from the Chamber Group's training end, and the fly is taken to have stopped at
the first gap longer than ``pr_break_gap_min`` (Δt, default 120 minutes).  The
gaps are training end to the first response, response to response, and the
last response to the end of the Test window.  A fly with no such gap before
the window ends was still responding when the recording stopped: its value is
**Censored**, a lower bound rather than a measurement.

Terms, fixed in ``CONTEXT.md``:

* the **Breaking Point** of a Chamber Group is its Paired fly's count of
  lick-backed Test Light Events before the break.  A Lick-free Light Event is
  no evidence the fly was still responding, so it neither counts nor
  interrupts a gap.  The Yoked fly has no breaking point: its light is its
  partner's.
* **Sucrose Persistence** is, for either fly, the minutes since training end
  of its last Sucrose Well feeding event before the break.

The **Test window** is the recording end minus the group's training end,
capped at ``pr_test_window_min`` when that is set (off by default).  Nothing
here knows the time of day: a long pause at night is a gap like any other.

Plain arrays only; :class:`ProgressiveRatioExperiment` assembles them per
Chamber Group, and :mod:`pyflic.base.analytics` holds the statistics
(Kaplan-Meier, log-rank) that use the censoring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

#: Design ``constants:`` read here, with their defaults.  The Progressive
#: Ratio type merges these into its ``default_constants``.
#: ``pr_test_window_min`` is deliberately absent: unset (or 0) means off.
DEFAULT_CONSTANTS: dict[str, Any] = {
    "pr_break_gap_min": 120,
}

#: The Δt values ``summary.txt`` always tabulates the breaking point at,
#: beside the configured one — the number moves with Δt, so every run shows
#: by how much.
SENSITIVITY_GAPS_MIN: tuple[float, ...] = (60.0, 120.0, 240.0)


def _positive(value: Any) -> float | None:
    """*value* as a positive finite float, else ``None``."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


@dataclass(frozen=True, slots=True)
class BreakSettings:
    """The first-gap rule's settings, resolved from the design's ``constants:``."""

    gap_min: float = 120.0
    test_window_min: float | None = None

    @classmethod
    def from_constants(cls, constants: Mapping[str, Any] | None) -> BreakSettings:
        merged = {**DEFAULT_CONSTANTS, **dict(constants or {})}
        gap = _positive(merged.get("pr_break_gap_min"))
        if gap is None:
            gap = float(DEFAULT_CONSTANTS["pr_break_gap_min"])
        return cls(gap_min=gap,
                   test_window_min=_positive(merged.get("pr_test_window_min")))

    def with_gap(self, gap_min: float) -> BreakSettings:
        return BreakSettings(gap_min=float(gap_min),
                             test_window_min=self.test_window_min)

    def test_end(self, available_min: float) -> float:
        """The Test window, in minutes since training end, for a group whose
        recording ran *available_min* past its training end."""
        available = max(0.0, float(available_min))
        if self.test_window_min is None:
            return available
        return min(available, self.test_window_min)

    def describe(self) -> str:
        window = "off" if self.test_window_min is None else f"{self.test_window_min:g}"
        return f"pr_break_gap_min={self.gap_min:g}, pr_test_window_min={window}"


@dataclass(frozen=True, slots=True)
class GapBreak:
    """The first-gap rule's verdict on one response stream.  ``counted`` is
    aligned with the input and says which responses the count holds."""

    count: int                  # responses before the break
    last_min: float             # the last of them, minutes since training end; 0 when none
    censored: bool              # no gap over the threshold before the Test window ended
    counted: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))


def first_gap_break(times: np.ndarray, test_end_min: float, gap_min: float) -> GapBreak:
    """The first-gap rule on *times*, minutes since training end.

    Responses outside ``[0, test_end_min]`` (or not finite) are ignored.  The
    first gap longer than *gap_min* — from training end to the first
    response, between responses, or from the last response to
    *test_end_min* — ends the count; when the only such gap is the tail, the
    fly stopped inside the window and the count is observed.  No gap at all:
    censored.
    """
    t = np.asarray(times, dtype=float)
    counted = np.zeros(t.size, dtype=bool)
    end = float(test_end_min)
    gap = float(gap_min)
    usable = np.isfinite(t) & (t >= 0.0) & (t <= end)
    order = np.flatnonzero(usable)
    order = order[np.argsort(t[order], kind="stable")]
    prev, count = 0.0, 0
    for i in order:
        if t[i] - prev > gap:
            return GapBreak(count=count, last_min=prev, censored=False, counted=counted)
        counted[i] = True
        count += 1
        prev = float(t[i])
    return GapBreak(count=count, last_min=prev, censored=not (end - prev > gap),
                    counted=counted)


def breaking_point(minutes: np.ndarray, lick_backed: np.ndarray,
                   test_end_min: float, gap_min: float) -> GapBreak:
    """A Chamber Group's Breaking Point from its Test Light Events.

    *minutes* are the events' onsets since training end and *lick_backed*
    says which were credited with at least one Sucrose Well lick.  Lick-free
    events are removed before the rule runs, so they neither count nor end a
    gap; ``counted`` is aligned with the full input.
    """
    m = np.asarray(minutes, dtype=float)
    backed = np.asarray(lick_backed, dtype=bool)
    if backed.shape != m.shape:
        raise ValueError("minutes and lick_backed must have the same length")
    idx = np.flatnonzero(backed)
    sub = first_gap_break(m[idx], test_end_min, gap_min)
    counted = np.zeros(m.size, dtype=bool)
    counted[idx[sub.counted]] = True
    return GapBreak(count=sub.count, last_min=sub.last_min, censored=sub.censored,
                    counted=counted)


def persistence(event_minutes: np.ndarray, test_end_min: float,
                gap_min: float) -> tuple[float, bool]:
    """``(minutes, censored)``: Sucrose Persistence from one fly's Sucrose
    Well feeding-event onsets, minutes since training end."""
    r = first_gap_break(event_minutes, test_end_min, gap_min)
    return float(r.last_min), bool(r.censored)


def format_count(count: int, censored: bool) -> str:
    """``"12"``, or ``"12+"`` for a censored count — a lower bound."""
    return f"{int(count)}{'+' if censored else ''}"
