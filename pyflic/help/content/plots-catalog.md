# Plot catalogue

What each figure shows, when it is produced, and what to look for in it.

## QC plots

Written to `<config>_results/qc/` for every DFM whenever QC runs — including as part of
basic analysis. These are diagnostics, not figures for a paper.

| Plot | Saved as | Shows |
|---|---|---|
| Raw signal | `qc*/raw_signal/DFM{n}_raw.png` | Every well's unprocessed signal |
| Baselined signal | `qc*/baselined/DFM{n}_baselined.png` | Every well after baseline subtraction, with thresholds drawn |
| Cumulative licks | `qc*/cumulative_licks/DFM{n}_cumulative_licks.png` | Running lick total per well over time |

The **baselined signal with thresholds** is the most useful one in the set. It puts your
`feeding_threshold` and `feeding_minimum` lines directly on the trace, so whether they sit
in a sensible place is a visual question rather than a guess.

**Cumulative licks** curves reveal timing at a glance: a steady slope is steady feeding, a
plateau is a fly that stopped, a step is a burst. A well whose curve never leaves zero is
inactive and should be excluded.

## Summary plots

### Feeding summary

Box-and-jitter panels grouped by treatment — or by factor combination when
[factors](config-factors.md) are declared. Written to `analysis*/feeding_summary.png` by
basic analysis.

Which metrics are panelled depends on the experiment type:

- **Single-well:** `Licks`, `Events`, `MeanDuration`, `MedDuration`, `MeanTimeBtw`,
  `MedTimeBtw`, `MeanInt`, `MedianInt`
- **Two-well, hedonic, progressive-ratio:** `PI`, `EventPI`, and A/B pairs of `Licks`,
  `Events`, `MeanDuration`, `MedDuration`, `MeanTimeBtw`, `MedTimeBtw`, `MeanInt`,
  `MedianInt`

Facet labels use your `well_names` when you have set them, so panels read `Sucrose` and
`Yeast` rather than `A` and `B`.

Because each point is a chamber, this is the plot that shows you your *spread*. A treatment
difference resting on two outlying chambers is visible here and invisible in a bar chart.

### Binned metric by treatment

A time course: per-treatment mean with an SEM band, across time bins. Use it to ask when
feeding happened rather than how much.

Any feeding-summary column works as the metric, plus base names — `Licks`, `Events`,
`MeanDuration`, `MedDuration`, `MeanTimeBtw`, `MedTimeBtw`, `MeanInt`, `MedianInt` — which
resolve automatically for the chamber size.

For two-well experiments, the combining mode matters and the sensible choice differs by
metric: **`total`** (A+B) for counts like `Licks` and `Events`; **`mean_ab`** for durations
and intervals, since summing two durations is meaningless. `A` and `B` isolate one well.

Bin width is yours to choose. Narrow bins show structure but are noisy; wide bins are
smooth but can hide a short-lived effect entirely. Try more than one before concluding
there is nothing there.

### Sliding-window plots

Three actions share this shape, each summarising a window of data and advancing by a step —
both in minutes, defaulting to 60 and 30.

| Plot | Shows |
|---|---|
| `plot_moving_window` | Treatment mean ± SEM of any metric |
| `plot_moving_median_chambers` | Median bout duration, one line per chamber, faceted by treatment |
| `plot_moving_median_treatment` | Treatment mean ± SEM of median bout duration |

Where binned plots cut time into non-overlapping slices, these slide a window across it, so
a step smaller than the window gives overlapping data and a smoother curve. That smoothing
is cosmetic: adjacent points share data and are not independent observations, so do not
treat them as such in a test.

`..._chambers` is the diagnostic version — it shows whether an effect is consistent across
flies or driven by a few. `..._treatment` is the version for a figure.

## Choice-specific plots

**Well A vs Well B duration jitter**, faceted — a direct visual comparison of the two
options within each treatment.

**Cumulative preference index**, per DFM — how a preference developed over the session. A PI
that starts near zero and separates gradually is a different phenomenon from one that is
established from the first minutes, and only a cumulative view distinguishes them.

## Hedonic plots

**Hedonic feeding plot** — Well A against Well B median durations, faceted by treatment.
Duration is the measure of interest in these designs; see
[Experiment types](concepts-experiment-types.md).

## Progressive-ratio plots

**Cumulative difference curve** (`plot_pr_cumulative_diff`, and `timecourse_pr_diff` in a
Project) — the headline figure: paired-minus-yoked cumulative sucrose-well licks against
minutes since the chamber group's training end, one mean ± SEM curve per treatment with the
individual group traces faint behind. A group contributes until its own recording ends, so
the ribbon widens late rather than every curve stopping at the shortest group.

**Training-aligned traces** (`plot_pr_cumulative_licks`) — per DFM, one panel per chamber
group: paired and yoked cumulative sucrose-well licks since training end, with the bins in
which the group's light was on drawn as points. A QC figure, not a result.

**Breaking-point plots** (`plot_breaking_point`) — per DFM, one panel per chamber:
`DeltaLicks` per light-on period against minutes since training end. Provisional; the
per-period table is the classic breaking-point readout and is kept in that spirit.

All three require light data (`OptoCol1`).

## Interactive or static

The hub's **Interactive plots** checkbox controls how figures are embedded. Interactive
gives pan, zoom and hover tooltips; static renders a PNG, which paints faster and uses less
memory. Turn it off when generating many figures at once.

## Where plots are saved

QC plots and the feeding summary are written automatically by basic analysis. Plots drawn
from the hub's Plots card open as tabs; save them from the toolbar or produce them through
a [script](scripts-actions.md) to write them to disk as part of a pipeline.

---

Related: [Summary metrics](concepts-metrics.md) · [Script actions](scripts-actions.md)
