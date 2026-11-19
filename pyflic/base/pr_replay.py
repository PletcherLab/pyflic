"""Replaying a ratio Paradigm on a recorded well: the Simulated Yoked Breaking Point (ADR-0023).

The firmware's progressive ratio under ``Counter Contact`` (DFMV3_V2
``src/LEDControl.c``, ``StepRatio``) counts milliseconds of contact outside
the well's own Activation, added up across bouts.  The Activation starts in
the millisecond the count reaches the Requirement; the count then restarts,
and contact that carries on after the Activation counts toward the next one.
This module replays those rules on a recorded signal, so that a well the
firmware never watched — a Yoked fly's Sucrose Well — can be asked what it
would have earned.

The data hold one sample every 200 ms, but the firmware counts contact per
ms.  :class:`ContactCredit` turns a well's above-threshold mask into ms of
contact per record.  The credit depends on whether the previous, this and the
next record were above threshold, and it is fitted on wells whose ``Progress``
the firmware did record (:func:`fit_contact_credit`): a record in the middle
of a bout is worth almost the whole 200 ms, one at a bout's edge about half.

Plain arrays only.  :class:`~pyflic.base.progressive_ratio_experiment.ProgressiveRatioExperiment`
chooses the wells, the rows and the threshold, and scores the replayed lights
with :mod:`pyflic.base.pr_breaking_point`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Iterable

import numpy as np

#: The eight ``(previous, this, next)`` above-threshold patterns of a record.
PATTERNS: tuple[tuple[bool, bool, bool], ...] = tuple(product((False, True), repeat=3))

#: The constant that switches the Simulated Yoked Breaking Point on.
CONSTANT = "pr_simulated_yoked_bp"


def neighbours(above: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(previous, this, next)`` above-threshold masks, aligned with *above*;
    beyond either end counts as below."""
    a = np.asarray(above, dtype=bool)
    prev = np.concatenate([[False], a[:-1]]) if a.size else a
    nxt = np.concatenate([a[1:], [False]]) if a.size else a
    return prev, a, nxt


@dataclass(frozen=True)
class ContactCredit:
    """ms of contact credited to a record, by its ``(previous, this, next)``
    above-threshold pattern.  ``records`` says how many fitted records each
    value rests on; a pattern with none takes the flat value (the whole
    record when it is above threshold, nothing otherwise)."""

    sample_ms: float
    ms: dict[tuple[bool, bool, bool], float] = field(default_factory=dict)
    records: dict[tuple[bool, bool, bool], int] = field(default_factory=dict)

    @classmethod
    def flat(cls, sample_ms: float) -> ContactCredit:
        return cls(sample_ms=float(sample_ms),
                   ms={p: (float(sample_ms) if p[1] else 0.0) for p in PATTERNS},
                   records={p: 0 for p in PATTERNS})

    def value(self, pattern: tuple[bool, bool, bool]) -> float:
        if pattern in self.ms:
            return float(self.ms[pattern])
        return float(self.sample_ms) if pattern[1] else 0.0

    def credit(self, above: np.ndarray) -> np.ndarray:
        """ms of contact in each record of a well whose above-threshold mask
        is *above*."""
        prev, this, nxt = neighbours(above)
        out = np.zeros(this.size, dtype=float)
        for p in PATTERNS:
            out[(prev == p[0]) & (this == p[1]) & (nxt == p[2])] = self.value(p)
        return out

    @property
    def fitted(self) -> int:
        """Records the fit rests on."""
        return int(sum(self.records.values()))

    def table(self) -> list[dict]:
        """One row per pattern, for text and CSV."""
        return [{"Previous": p[0], "This": p[1], "Next": p[2],
                 "ContactMs": self.value(p), "Records": int(self.records.get(p, 0))}
                for p in PATTERNS]


@dataclass(frozen=True)
class WellRecord:
    """One Trigger Well's recorded state over the rows of a ratio run: its
    above-threshold mask and the firmware's ``Progress`` (ms of contact),
    ``Requirement``, ``Activations``, ``Phase`` (True in Acclimation) and
    ``OnMs`` on each row."""

    above: np.ndarray
    progress: np.ndarray
    requirement: np.ndarray
    activations: np.ndarray
    acclimation: np.ndarray
    on_ms: np.ndarray


def fit_contact_credit(wells: Iterable[WellRecord], *, sample_ms: float,
                       activation_ms: float) -> ContactCredit:
    """The mean ms of contact the firmware counted in a record, by pattern,
    over every Main-phase record of *wells* in which it could count freely:
    no Activation started, and no light in the record or in the ones an
    Activation of *activation_ms* could reach back from.  The contact counted
    in a record is the rise of its ``Progress`` over the record before."""
    reach = int(math.ceil(float(activation_ms) / float(sample_ms)))
    sums = {p: 0.0 for p in PATTERNS}
    counts = {p: 0 for p in PATTERNS}
    for well in wells:
        progress = np.asarray(well.progress, dtype=float)
        n = progress.size
        if n < 2:
            continue
        main = ~np.asarray(well.acclimation, dtype=bool) & ~np.isnan(
            np.asarray(well.requirement, dtype=float))
        activations = np.asarray(well.activations, dtype=float)
        lit = np.asarray(well.on_ms, dtype=float) > 0
        free = main.copy()
        free[1:] &= main[:-1] & (np.diff(activations) == 0)
        free[0] = False
        for k in range(reach + 1):
            free[k:] &= ~lit[:n - k] if k else ~lit
        rise = np.diff(progress, prepend=np.nan)
        free &= np.isfinite(rise) & (rise >= 0)
        prev, this, nxt = neighbours(well.above)
        for p in PATTERNS:
            sel = free & (prev == p[0]) & (this == p[1]) & (nxt == p[2])
            sums[p] += float(rise[sel].sum())
            counts[p] += int(sel.sum())
    flat = ContactCredit.flat(sample_ms)
    return ContactCredit(
        sample_ms=float(sample_ms),
        ms={p: (sums[p] / counts[p] if counts[p] else flat.value(p)) for p in PATTERNS},
        records=counts)


@dataclass(frozen=True)
class Replay:
    """The Activations a replay produced: the record of each onset and the
    Requirement (ms of contact) it met."""

    onsets: np.ndarray
    requirements: np.ndarray


def replay_contact_ratio(credit_ms: np.ndarray, start: int, end: int, *,
                         requirement_for: Callable[[int], int | None],
                         activation_ms: float, sample_ms: float) -> Replay:
    """Replay a ratio Paradigm counting Contact over records ``[start, end)``.

    *credit_ms* is the ms of contact in each record
    (:meth:`ContactCredit.credit`).  The replay starts in Main with the first
    Requirement and no Progress, as the firmware is at training end.
    *requirement_for(k)* is the k-th Main Activation's Requirement in ms
    (:meth:`~pyflic.base.opto_program.Paradigm.requirement_for`).  A record's
    contact is spread evenly over it.  The part an Activation covers does not
    count, and an Activation starts at the moment within the record when the
    count reaches the Requirement.
    """
    credit = np.asarray(credit_ms, dtype=float)
    sample = float(sample_ms)
    onsets: list[int] = []
    met: list[float] = []
    k = 1
    req = requirement_for(k)
    progress = 0.0
    lit_left = 0.0                      # ms of the current Activation still to run
    for j in range(max(0, int(start)), min(int(end), credit.size)):
        if req is None or req <= 0:
            break
        covered = min(lit_left, sample)
        lit_left -= covered
        usable = credit[j] * (sample - covered) / sample
        if usable <= 0:
            continue
        if progress + usable >= req:
            ## The count reaches the Requirement part-way through the record:
            ## the Activation starts there and covers the rest of it.
            at = covered + (req - progress) / usable * (sample - covered)
            onsets.append(j)
            met.append(float(req))
            k += 1
            req = requirement_for(k)
            progress = 0.0
            lit_left = float(activation_ms) - (sample - at)
        else:
            progress += usable
    return Replay(onsets=np.asarray(onsets, dtype=int), requirements=np.asarray(met, dtype=float))
