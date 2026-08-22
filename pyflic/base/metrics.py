"""Metric catalogues, shared by the UI, the Script Editor, and the runners.

These lived in ``analysis_hub`` — a PyQt module — which meant a headless Batch
Run had to import the whole GUI stack to look up a metric's default two-well
mode.  They are plain data and belong somewhere neither UI nor runner has to
apologise for importing.

Each binned-metric entry is ``(display label, metric arg, two_well_mode arg)``.
"""

from __future__ import annotations

TWO_WELL_BINNED: list[tuple[str, str, str]] = [
    ("Licks (A+B total)",       "Licks",         "total"),
    ("PI",                      "PI",            "total"),
    ("Event PI",                "EventPI",       "total"),
    ("Licks A",                 "LicksA",        "A"),
    ("Licks B",                 "LicksB",        "B"),
    ("Events (A+B total)",      "Events",        "total"),
    ("Med Duration (A+B avg)",  "MedDuration",   "mean_ab"),
    ("Med Duration A",          "MedDurationA",  "A"),
    ("Med Duration B",          "MedDurationB",  "B"),
    ("Mean Duration (A+B avg)", "MeanDuration",  "mean_ab"),
    ("Med Time Btw (A+B avg)",  "MedTimeBtw",    "mean_ab"),
]

SINGLE_WELL_BINNED: list[tuple[str, str, str]] = [
    ("Licks",         "Licks",        "total"),
    ("Events",        "Events",       "total"),
    ("Med Duration",  "MedDuration",  "total"),
    ("Mean Duration", "MeanDuration", "total"),
    ("Med Time Btw",  "MedTimeBtw",   "total"),
    ("Mean Int",      "MeanInt",      "total"),
    ("Median Int",    "MedianInt",    "total"),
]

#: Base metric names for the well A vs B comparison.
WELL_CMP_METRICS: list[str] = [
    "MedDuration", "MeanDuration", "Licks", "MedTimeBtw", "MeanTimeBtw",
    "MeanInt", "MedianInt",
]

#: Default ``two_well_mode`` for each metric, used by the script runner.
METRIC_DEFAULT_MODE: dict[str, str] = {
    metric: mode for _, metric, mode in TWO_WELL_BINNED + SINGLE_WELL_BINNED
}


def binned_metrics(chamber_layout: str) -> list[tuple[str, str, str]]:
    """The binned-metric catalogue for *chamber_layout*."""
    return list(SINGLE_WELL_BINNED if chamber_layout == "single_well"
                else TWO_WELL_BINNED)
