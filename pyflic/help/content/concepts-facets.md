# Facets: time windows as a column

A **facet** is a named time window within a recording. Facets are how pyflic splits a run
into phases — acclimation, test, recovery — without splitting the output into directories.

## Windows are a column, not a directory

Set `facet_cutoffs` and analysis writes two files into `analysis/`:

- `feeding_summary.csv` — one row per chamber, whole recording
- `feeding_summary_facet.csv` — one row per chamber **per facet**, with `Facet` and
  `FacetRange` columns

```yaml
global:
  facet_cutoffs: [10, 70]
```

gives three facets: `0-10 min`, `10-70 min`, `70+ min`.

Older versions wrote windowed results into `analysis_0_360/` directories, so an output path
was a function of the window you happened to ask for. That is retired. Two things made it
untenable: outputs are now always in `analysis/`, and pooling could not work — a Combined
Analysis stacking replicates would have to *guess* which of a replicate's window
directories was the one to pool.

## Half-open windows

Every window is `[start, end)` — inclusive at the start, exclusive at the end. That is what
makes a faceted analysis a true partition: a lick landing exactly on a cutoff belongs to the
phase after it and to nothing else. With inclusive bounds it would be counted in both
adjacent phases and inflate them.

The last facet always runs to the end of the recording, so nothing is lost at the tail.

## Named phases

An Experiment Type can name its phases. Progressive Ratio's default cutoff of 30 minutes
gives `Training` and `Test`.

Those names apply **only** while the cutoffs are the type's default. Move a cutoff and you
get plain minute-range labels instead — "Training" would be a lie once you have moved the
boundary that defined it. Set `facet_labels` to name them yourself:

```yaml
global:
  facet_cutoffs: [15]
  facet_labels: [Early, Late]
```

## Facets in a Project

`facet_cutoffs` is part of the design, so every replicate in a Project is windowed
identically by construction. There is nothing to reconcile at pooling time — which is the
whole point.

Facets are also what make the faceted publication figures possible: a plot with one panel
per phase has no meaning while windows are directories. See
[Plot Editor](app-plot-editor.md).
