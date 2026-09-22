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
