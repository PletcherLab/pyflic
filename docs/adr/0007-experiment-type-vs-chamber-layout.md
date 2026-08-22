---
status: accepted
---

# Experiment Type and Chamber Layout are separate concepts

`experiment_type:` conflated two things: `hedonic` and `progressive_ratio` name
an assay, while `two_well` and `single_well` name a hardware fact (chamber_size
2 vs 1) that several different assays share — which is why `HedonicFeedingExperiment`
had to subclass `TwoWellExperiment`. Following PyTrackingAnalysis's Experiment
Type / Tracking Type split (its ADR-0001 and ADR-0002), we separate them:
**Chamber Layout** (`single_well` / `two_well`) is the low-level layer and stays
the existing `Experiment` subclass hierarchy, while **Experiment Type** becomes a
composed strategy object (Custom, Hedonic, Progressive Ratio) that selects one
Chamber Layout and constrains everything else.

## Considered options

Porting PyTrackingAnalysis's composed strategy wholesale — dissolving the
subclasses so one `Experiment` orchestrator asks its type for every decision —
was rejected. TwoWellExperiment's simultaneous-feeding matrix, bleeding check and
`facet_plot_well_durations`, plus ProgressiveRatioExperiment's breaking-point port
from `breaking_point.R`, are working analysis code; rewriting them into strategy
objects buys symmetry with PyTrackingAnalysis and nothing else. The subclasses are
a good fit for what they actually express — a data shape.

## Consequences

- A typed config writes neither `chamber_layout:` nor `params.chamber_size`; the
  type owns both and they are derived, never persisted. The load-time check that
  the two agree (yaml_config.py) is deleted, because they can no longer disagree.
  Only a Custom Experiment states `chamber_layout:`.
- The `Experiment` subclass is chosen from the Chamber Layout, *except* where a
  type brings analysis of its own: `ExperimentType.experiment_class` names one,
  and Hedonic and Progressive Ratio use it to keep `weighted_duration_summary`,
  `hedonic_feeding_plot`, and the breaking-point port reachable. So the subclass
  hierarchy is "data shape, plus the two types that outgrew it" rather than data
  shape alone — the honest description of what shipped.
- `experiment_type: two_well` and `experiment_type: single_well` stop being types.
  Such a config is a Custom Experiment with that Chamber Layout.
- The Script Editor's single `Requires` literal splits in two: `plot_well_comparison`
  and `transition_matrix` require a Chamber Layout, `plot_hedonic` and
  `plot_breaking_point` require an Experiment Type.
- An Experiment Type additionally declares an ordered **report set** — the figures
  its report and the Standard Pipeline produce with no authoring. That is a default,
  not a gate: any non-gated action stays available in the Script Editor.
