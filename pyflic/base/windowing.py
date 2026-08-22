"""Facet windows: a time window is a column, not a directory (ADR-0008).

pyflic used to write windowed results into ``analysis_<start>_<end>/``, so an
output path was a function of the requested window.  Facets replace that: the
Project Design fixes ``facet_cutoffs``, every Replicate is windowed
identically, and the window travels with the data as a column.

**Windows are half-open — ``[start, end)``.**  That is what makes a faceted
analysis a true partition of the recording: a lick landing exactly on a cutoff
belongs to the phase after it and to nothing else.  The final facet always runs
to ``inf``, so nothing is lost at the tail.

``(0, 0)`` remains pyflic's sentinel for "the whole recording".
"""

from __future__ import annotations

FULL_RANGE = (0.0, 0.0)


def is_full_range(range_minutes) -> bool:
    """True when *range_minutes* is the "whole recording" sentinel ``(0, 0)``.

    Compares the pair directly rather than summing it, so a genuine window such
    as ``(-5, 5)`` is not mistaken for the sentinel.
    """
    try:
        return tuple(float(v) for v in range_minutes) == FULL_RANGE
    except (TypeError, ValueError):
        return False


def normalize_cutoffs(cutoffs) -> list[float]:
    """Coerce a facet-cutoff argument into a list of floats.

    Accepts a single number or any numeric sequence.  ``None`` raises rather
    than defaulting, so a caller that forgets to pass cutoffs fails loudly
    instead of silently splitting the recording somewhere arbitrary and
    labelling the result as the user's configured facets.
    """
    if cutoffs is None:
        raise ValueError(
            "cutoffs must be a number or a sequence of numbers, not None. "
            "Set 'facet_cutoffs' under 'global:' in flic_config.yaml (or in "
            "the Project design), or pass cutoffs= explicitly."
        )
    if isinstance(cutoffs, bool):
        raise ValueError(f"Invalid cutoffs: {cutoffs!r}. Must be numeric.")
    if isinstance(cutoffs, (int, float)):
        values = [cutoffs]
    elif isinstance(cutoffs, (str, bytes)):
        raise ValueError(f"Invalid cutoffs: {cutoffs!r}. Must be a number or a sequence.")
    else:
        try:
            values = list(cutoffs)
        except TypeError as err:
            raise ValueError(
                f"Invalid cutoffs: {cutoffs!r}. Must be a number or a sequence of numbers."
            ) from err
    if not values:
        raise ValueError("cutoffs must contain at least one value.")
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError) as err:
        raise ValueError(
            f"Invalid cutoffs: {cutoffs!r}. Every value must be numeric."
        ) from err


def _tidy_bound(value):
    """Render a whole-number boundary as an int.

    ``FacetRange`` cells and plot titles show these values, so ``10`` must not
    become ``10.0`` just because it passed through a float normalisation step.
    """
    if value == float("inf") or value != int(value):
        return value
    return int(value)


def facet_windows(cutoffs) -> list[tuple]:
    """Return the ``[(start, end), ...]`` windows implied by *cutoffs*.

    Boundaries are sorted and de-duplicated, the first window opens at 0 and the
    last runs to infinity.  Combined with the half-open rule the windows tile
    the recording exactly once.
    """
    values = sorted(set(normalize_cutoffs(cutoffs)))
    bounds = [0.0] + [v for v in values if v > 0.0] + [float("inf")]
    tidied = [_tidy_bound(b) for b in bounds]
    return [(tidied[i], tidied[i + 1]) for i in range(len(tidied) - 1)]


def minute_label(window) -> str:
    """A plain minute-range label for *window*, e.g. ``"10-70 min"``."""
    start, end = window
    if end == float("inf"):
        return f"{_tidy_bound(start)}+ min"
    return f"{_tidy_bound(start)}-{_tidy_bound(end)} min"


def format_range(window) -> str:
    """The canonical ``FacetRange`` cell for *window* — round-trips through
    :func:`parse_range`."""
    start, end = window
    return f"({_tidy_bound(start)}, {_tidy_bound(end)})"


def parse_range(value) -> tuple:
    """A ``FacetRange`` cell (``"(10, 70)"`` / ``"(70, inf)"``) back to a tuple."""
    if isinstance(value, tuple):
        return value
    inner = str(value).strip().strip("()")
    lo, _, hi = inner.partition(",")
    return (float(lo), float(hi))


def as_range_minutes(window) -> tuple[float, float]:
    """A facet window as the ``range_minutes`` pair pyflic's loaders take.

    The open-ended tail becomes ``(start, 0)`` because ``0`` is pyflic's
    "through the end of the recording" sentinel — ``inf`` would be rejected by
    the range validators in ``dfm``.
    """
    start, end = window
    return (float(start), 0.0 if end == float("inf") else float(end))
