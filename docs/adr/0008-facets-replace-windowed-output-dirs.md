---
status: accepted
---

# A time window is a column, not a directory

pyflic wrote windowed results to `<stem>_results/analysis_<start>_<end>/`, so an
output path was a function of the config filename and the requested window. Two
things made that untenable: fixing the output directory to `analysis/` (ADR-0005,
matching PyTrackingAnalysis and pySurvAnalysis), and pooling — a Combined Analysis
that stacks Replicates would have to guess which of a Replicate's window
directories is the one to pool. We adopt PyTrackingAnalysis's **Facet** instead:
`facet_cutoffs:` in the Project Design windows every Replicate identically, and
analysis writes `feeding_summary.csv` plus `feeding_summary_facet.csv` carrying a
Facet column.

## Consequences

- The Hub's Start/End controls select a Facet rather than filtering the load;
  load-time `range_minutes`, the script `start`/`end` step params, and the
  `.pyflic_cache/` key all change accordingly.
- Facets make the Publication Figure families expressible: a faceted metric plot
  (x = Treatment, faceted by Facet) has no meaning while windows are directories.
- Experiment Types can name their phases, the way PyTrackingAnalysis's Valence type
  names Acclimation / Experiment / Cooldown.
- Existing `analysis_<start>_<end>/` and `<stem>_results/` directories are orphaned.
  pyflic never deletes them; `pyflic lint` reports them (there is no migration tool
  — see CONTEXT.md § Migration).
