from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence


_nsre = re.compile(r"(\d+)")


def natural_key(s: str) -> list[object]:
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]


def natural_sorted(paths: Iterable[Path]) -> list[Path]:
    return sorted(list(paths), key=lambda p: natural_key(p.name))


def range_is_specified(rng: Sequence[float]) -> bool:
    return len(rng) == 2 and (float(rng[0]) + float(rng[1]) != 0.0)


def range_bounds(rng: Sequence[float]) -> tuple[float, float]:
    """``(start, end)`` for a specified range, with the open end resolved.

    ``0`` (and any non-positive end) is pyflic's "through the end of the
    recording" sentinel — ``windowing.as_range_minutes`` hands the tail Facet
    to the loaders as ``(start, 0)`` — so it becomes ``inf`` here instead of
    being read as a literal minute, which used to leave the last facet of
    every faceted experiment empty.
    """
    a, b = float(rng[0]), float(rng[1])
    if b <= 0.0:
        b = float("inf")
    return a, b

