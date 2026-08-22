"""The Progressive Ratio Experiment Type.

A two-well operant assay ported from ``breaking_point.R``: the effort required
for a reward escalates until the animal stops working, and the point at which it
stops — the breaking point — is the result.  The escalation schedule means the
recording has a real phase structure, so this type carries facet cutoffs rather
than analysing the run as one block.
"""

from __future__ import annotations

from .base import ExperimentType


class ProgressiveRatioExperimentType(ExperimentType):
    name = "ProgressiveRatio"
    display_name = "Progressive Ratio"

    chamber_layout = "two_well"
    required_wells = ("A", "B")
    experiment_class = "pyflic.base.progressive_ratio_experiment:ProgressiveRatioExperiment"
    # A default the user may change: training, then the escalating test block.
    facet_cutoffs = (30.0,)
    facets_fixed = False
    phase_labels = ("Training", "Test")

    default_constants = {
        "min_untransformed_licks_cutoff": 20,
        "max_med_duration_cutoff": 13.0,
        "max_events_cutoff": 150000.0,
    }

    def report_intro(self) -> str:
        return ("A progressive-ratio assay: the effort required per reward "
                "escalates over the recording, and the breaking point is the "
                "ratio at which the animal stops responding. The default "
                "phases are Training (0-30 min) and Test (30+ min).")

    def report_set(self, chamber_layout: str) -> list[str]:
        return ["faceted_licks", "faceted_events", "faceted_medduration",
                "timecourse_licks", "timecourse_events"]

    def output_manifest(self) -> list[str]:
        return ["feeding_summary.csv", "feeding_summary_facet.csv",
                "summary.txt"]
