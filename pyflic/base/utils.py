from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Sequence


_nsre = re.compile(r"(\d+)")

#: The Environmental Monitor's id.  PyMCU writes its records as
#: ``DFM99_<n>.csv`` beside the DFMs' (27 columns, ``W1``-``W6`` relay and
#: output states), so every DFM-id discovery skips it and the loader refuses
#: it (decision 5 of ``docs/v4-data-plan.md``).
ENVIRONMENTAL_MONITOR_ID = 99
ENVIRONMENTAL_MONITOR_MESSAGE = "DFM 99 is the Environmental Monitor, not a DFM"


def is_environmental_monitor(dfm_id) -> bool:
    """Whether *dfm_id* names the Environmental Monitor, not a DFM."""
    try:
        return int(dfm_id) == ENVIRONMENTAL_MONITOR_ID
    except (TypeError, ValueError):
        return False


def user_config_dir() -> Path:
    """The user's pyflic settings folder: ``$XDG_CONFIG_HOME/pyflic``, or
    ``~/.config/pyflic`` when the variable is unset, empty or relative (the
    XDG spec ignores both).  Asked on every call, so the settings file, the
    type library and ``.env`` follow the environment as it is now."""
    raw = os.environ.get("XDG_CONFIG_HOME", "")
    base = Path(raw) if raw and os.path.isabs(raw) else Path.home() / ".config"
    return base / "pyflic"


def natural_key(s: str) -> list[object]:
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]


def natural_sorted(paths: Iterable[Path]) -> list[Path]:
    return sorted(list(paths), key=lambda p: natural_key(p.name))


def range_is_specified(rng: Sequence[float]) -> bool:
    """Whether *rng* names a window rather than the whole-recording sentinel
    ``(0, 0)``.  Compares the pair, not its sum, as
    :func:`pyflic.base.windowing.is_full_range` does, so ``(-5, 5)`` is a
    window."""
    return len(rng) == 2 and not (float(rng[0]) == 0.0 and float(rng[1]) == 0.0)


#: The power of pyflic's lick transform (``transform_licks``): Licks^0.25.
LICK_TRANSFORM_POWER = 0.25


def lick_total(a, b, *, transformed: bool):
    """Two wells' Licks together.  On the fourth-root scale the total is the
    root of the summed counts, (A + B)^¼, never A^¼ + B^¼: the counts are
    recovered (``value ** 4``), summed and transformed again.  Works on
    numbers, arrays and Series alike; NaN propagates."""
    if not transformed:
        return a + b
    return (a ** 4 + b ** 4) ** LICK_TRANSFORM_POWER


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

