"""The Hedonic Feeding Experiment Type.

A two-well choice assay in which each chamber offers two food sources and the
readout is how strongly the animal favours one over the other.  Well A is the
reference: pyflic's Preference Index is ``(A - B) / (A + B)``, so **positive PI
means a preference for well A** — which is why the type insists both wells are
named.  Without ``well_names`` a PI axis reads "preference for A", which tells a
reader nothing.
"""

from __future__ import annotations

from .base import ExperimentType


class HedonicExperimentType(ExperimentType):
    name = "Hedonic"
    display_name = "Hedonic Feeding"

    chamber_layout = "two_well"
    required_wells = ("A", "B")
    experiment_class = "pyflic.base.hedonic_experiment:HedonicFeedingExperiment"
    # A default the user may change (facets_fixed stays False): the whole
    # recording as one facet is the common case, and a hedonic run has no
    # protocol-imposed phase structure the way an optogenetic assay does.
    facet_cutoffs = None

    default_constants = {
        "min_untransformed_licks_cutoff": 20,
        "max_med_duration_cutoff": 13.0,
        "max_events_cutoff": 150000.0,
    }

    def report_intro(self) -> str:
        return ("A hedonic feeding assay: each chamber offers a choice between "
                "two food sources. Positive PI means a preference for well A "
                "over well B; the well names below say which is which.")

    def report_set(self, chamber_layout: str) -> list[str]:
        return ["faceted_pi", "faceted_licks", "faceted_medduration",
                "timecourse_pi", "timecourse_licks"]

    def output_manifest(self) -> list[str]:
        return ["feeding_summary.csv", "feeding_summary_facet.csv",
                "summary.txt", "weighted_duration.csv"]
