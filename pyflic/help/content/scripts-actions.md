# Script actions

Every action a script step can take. `start` and `end` are accepted by nearly all of them
and are omitted from the tables below — they are minutes, and blank means inherit (see
[where values come from](scripts-overview.md#where-values-come-from)).

## Load actions

| Action | Parameters | What it does |
|---|---|---|
| `load` | `parallel` | Read the DFM CSVs into memory for the rest of the script |
| `remove_chambers` | `group` | Apply exclusions from `remove_chambers.csv` |
| `write_summary` | — | Write `summary.txt` with excluded chambers and experiment metadata |

`parallel: true` loads DFMs concurrently, which is worth having on any experiment with
several devices. `group` defaults to the **script's name** — see
[exclusions](config-dfms-chambers.md#excluding-chambers).

## Analyse actions

| Action | Parameters | What it does |
|---|---|---|
| `basic_analysis` | — | The standard pipeline: QC, summary, feeding summary, summary plot |
| `feeding_csv` | — | Per-chamber feeding summary to CSV |
| `binned_csv` | `binsize` | Feeding metrics binned over time, to CSV |
| `weighted_duration` | — | Weighted-duration summary (hedonic experiments) |
| `tidy_export` | `kind` | One row per bout, for downstream statistics |
| `bootstrap` | `metric`, `mode`, `n_boot`, `ci`, `seed` | Bootstrap confidence intervals |
| `compare` | `metric`, `mode`, `model`, `factors` | ANOVA or linear mixed model across treatments |
| `light_phase_summary` | — | Feeding summary split by light and dark phase |
| `param_sensitivity` | `parameter`, `values` | Re-run across a range of one detection parameter |
| `transition_matrix` | — | Transition probabilities between bout types |
| `pdf_report` | `metrics`, `binsize` | Binned plots and tables bundled into one PDF |

**`tidy_export`** takes `kind: feeding` (default) or `tasting`.

**`bootstrap`** defaults to `metric: PI`, `n_boot: 2000`, `ci: 0.95`, `seed: 0`. The fixed
seed means repeated runs give identical intervals — change it only if you want to confirm
your result is not an artefact of one resampling draw.

**`compare`** defaults to `metric: MedDuration`, `mode: A`, `model: aov`. Use `model: lmm`
for a linear mixed model when you need a random effect. `factors` names the factors to
include; see [Factorial designs](config-factors.md).

**`param_sensitivity`** is the action to reach for when you are unsure about a detection
parameter. It re-runs the analysis across the values you give and records how your metrics
move:

```yaml
- action: param_sensitivity
  parameter: feeding_event_link_gap
  values: [1, 3, 5, 10, 20]
```

`parameter` accepts `feeding_event_link_gap`, `feeding_threshold`, `feeding_minimum`,
`tasting_minimum`, `tasting_maximum`, `feeding_minevents`, `tasting_minevents`,
`baseline_window_minutes`, or `samples_per_second`. Run it **before** you commit to a
parameter for a publication — an effect that appears only at one link gap is not an effect.

**`pdf_report`** defaults to `metrics: Licks, Events, MedDuration`.

## Plot actions

| Action | Parameters | What it does |
|---|---|---|
| `plot_feeding_summary` | — | Bar/jitter plot of per-chamber metrics |
| `plot_binned` | `metric`, `mode`, `binsize` | A metric binned over time, by treatment |
| `plot_dot` | `metric`, `mode` | Jittered dot plot of a metric by treatment |
| `plot_moving_window` | `metric`, `mode`, `window`, `step` | Treatment mean ± SEM over a sliding window |
| `plot_moving_median_chambers` | `window`, `step`, `mode` | Time-dependent median bout duration, one line per chamber |
| `plot_moving_median_treatment` | `window`, `step`, `mode` | Treatment mean ± SEM of time-dependent median bout duration |
| `plot_well_comparison` | `metric` | Well A against Well B for one metric |
| `plot_hedonic` | — | Hedonic feeding plot |
| `plot_breaking_point` | `config` | Per-DFM breaking-point plots (progressive ratio) |

## `metric` and `mode`

`plot_binned`, `plot_dot`, `plot_moving_window`, `bootstrap` and `compare` all take a
metric name and a mode.

**`metric`** is any column of the feeding summary — `Licks`, `Events`, `MedDuration`,
`MeanDuration`, `MedTimeBtw`, `PI`, `EventPI`, and the rest. See
[Summary metrics](concepts-metrics.md).

**`mode`** decides how the two wells of a chamber are combined:

| `mode` | Meaning |
|---|---|
| `total` | Both wells summed — the chamber as a whole |
| `A` | Well A only |
| `B` | Well B only |
| `mean_ab` | The mean of the two wells |

Mode is meaningless for single-well experiments and for metrics that are already
chamber-level, such as `PI`.

## Sliding-window plots

The three moving-window actions share `window` and `step`, both in minutes and both
required — defaulting to `60` and `30`.

`window` is how much data each point summarises; `step` is how far the window advances
between points. A step smaller than the window means overlapping windows and a smoother
curve, at the cost that adjacent points are no longer independent — which matters if you
are tempted to run a test on them.

`plot_moving_median_chambers` draws one line per chamber, faceted by treatment; use it to
see whether a treatment effect is consistent across flies or driven by a few. 
`plot_moving_median_treatment` collapses to treatment mean ± SEM, which is the version for
a figure. Both take `mode` of `mean_ab` (default), `A`, or `B`.

## A full pipeline

```yaml
scripts:
  - name: "Standard Analysis"
    steps:
      - action: load
        start: 0
        end: 240
        parallel: true
      - action: remove_chambers
      - action: write_summary
      - action: basic_analysis
      - action: feeding_csv
      - action: binned_csv
        binsize: 30
      - action: plot_feeding_summary
      - action: plot_binned
        metric: Licks
        mode: total
        binsize: 30
      - action: plot_dot
        metric: PI
      - action: plot_moving_median_treatment
        window: 60
        step: 30
      - action: compare
        metric: MedDuration
        mode: mean_ab
        model: aov
      - action: pdf_report
```

---

Related: [What a script is](scripts-overview.md) · [Script Editor](scripts-editor.md) ·
[Plot catalogue](plots-catalog.md)
