---
status: accepted
---

# Progressive Ratio Facets are per-group, data-derived Training/Test windows

A Progressive Ratio recording has a real two-phase structure — closed-loop
**Training**, then the escalating **Test** — but the boundary is
behaviour-contingent and differs for every **Chamber Group** (6 to 896 minutes
in the first real dataset). A **Facet** in pyflic (ADR-0008) is a named time
window fixed by the Design's `facet_cutoffs:` so that every Member is windowed
identically. We keep the Facet *column* and its consumers, and let this one
type fill it from the data: `feeding_summary_facet.csv` for a Progressive Ratio
experiment carries exactly two Facets, `Training` and `Test`, split at each
group's own training end (the last minute the Paired chamber's Sucrose Well is
flagged in training). The type owns its facets (`facets_fixed`), so a PR config
never states `facet_cutoffs`, and the old 30-minute default is gone.

## Considered options

- **Test phase only, no facet file.** `feeding_summary.csv` computed from each
  group's training end onward. Rejected: the whole-recording row every other
  type produces would vanish, training-phase feeding would never be summarised,
  and the Combined Analysis would have to special-case the type.
- **Keep fixed-minute facets, add PR columns.** Rejected: a Facet labelled
  "Training" that ends at minute 30 for a group that trained for 896 minutes is
  a false label, and the type's whole point is the phase boundary.
- **Per-group Facets (chosen).** A Facet is still one named window per row, so
  `faceted_*` figures, the Combined Analysis and the mixed model work with
  `Facet == "Test"` unchanged. The cost is that `StartMin`/`EndMin` now vary
  by row within one Member, which no consumer relied on being constant.

## Consequences

- The glossary's Facet entry names this as the single exception; the Design
  editor hides the cutoff controls for this type.
- A group whose Paired well never clears has no `Test` rows worth reading:
  `TrainingComplete` is false on both chambers, and the type constant
  `require_training_complete` (default true) auto-removes the group through
  the existing exclusion path, so the rule is stated once and appears in the
  exclusions table.
- Time-course figures for this type take their x axis from the group's training
  end, not from the recording start.

## Addendum (2026-09-23): the light QC, and auto-removal in the basic pipeline

The first real dataset showed a failure the Facets cannot see. The firmware
lights a Chamber Group from its own reading of the Paired chamber's Sucrose
Well during the run; pyflic counts licks afterwards from the baselined signal.
A Sucrose Well whose resting level creeps up looks continuously touched to the
firmware and flat to pyflic, so the light fires on its own and every light-on
number describes the sensor. On DFM 1 of `test_data/progressive_ratio` one
group's light ran unearned for the last few hours of the day, and another's
group "completed" Training in 13 seconds with no lick at all — and the pooled
Chr-vs-WCS difference rested on both.

Every Chamber Group now gets a **light QC** verdict (`pyflic.base.pr_light_qc`,
`pr_light_qc.csv`), computed over the whole recording:

- **Self-triggered light** — at least `pr_lick_free_run` (5) consecutive Test
  Light Events with no Sucrose Well licks — and **implausible training** —
  Training completed with light events and zero Sucrose Well licks — fail the
  group.
- **No increasing trend** (Spearman's rho of licks per Light Event against
  event number below `pr_trend_min_rho`, 0.3, once there are
  `pr_trend_min_events`, 5), a **resting level rise** of at least
  `pr_resting_level_rise` (15) counts over the first 30 minutes, and a
  **resting level elevated** to `pr_resting_level_ratio` (3) times the DFM's
  other Sucrose Wells are warnings.

A failed group leaves through the same auto-removal path as
`require_training_complete`, unless `exclude_failed_pr_groups` is switched off;
the QC figures and table keep it in view either way.

Considered and rejected:

- **Truncating the group's Test window at the first failure.** The boundary is
  fuzzy — the dataset's first group degraded for hours before its first
  lick-free run — and a silent per-group window is easy to misread downstream.
  The table reports `LickFreeRunStartMin` for a person to act on instead.
- **Fewer licks than training pairings as the training rule.** pyflic's
  feeding threshold misses the brief touches the firmware counts: a healthy
  group trained on 5 lick samples over 8 pairings. Zero licks is the rule.
- **Firmware constants (lick increment, pairings) in the design.** Both are set
  per experiment and not recorded in the data; the checks ask only whether the
  requirement keeps rising, and the increment is estimated from the data for a
  reference line.

The Consequences above said `require_training_complete` removes a group
"through the existing exclusion path". Until this addendum nothing on the
standard path called that exclusion: `auto_remove_chambers()` ran only from the
QC Viewer's Auto Filter button. `execute_basic_analysis` now runs it, once per
loaded experiment and for every Experiment Type, before it writes the summary,
so every output describes the filtered design and the statement holds.
