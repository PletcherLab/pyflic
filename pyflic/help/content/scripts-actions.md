# Script actions

Every action a script step can take. `slice` is accepted by nearly all of them and is
omitted from the tables below: the [Slice](concepts-slices.md) the step works on, blank to
inherit the script's, else the whole recording (see
[Slices](scripts-overview.md#slices)). `run_qc`, `run_analysis` and `run_plots` without a
`slice` run the whole recording and every Slice the Selection keeps.

## Load actions

| Action | Parameters | What it does |
|---|---|---|
| `load` | `parallel` | Nothing. Kept so older scripts run unchanged |
| `remove_chambers` | `group` | Apply a group of exclusions from `remove_chambers.csv` |
| `write_summary` | — | Write `summary.txt` with excluded chambers and experiment metadata |

The member is already loaded when a script starts, so `load` reads nothing, and its
`parallel` changes nothing — see
[The member is already loaded](scripts-overview.md#the-member-is-already-loaded).
`group` defaults to the **script's name** — see
[exclusions](config-dfms-chambers.md#excluding-chambers).

## Analyse actions

| Action | Parameters | What it does |
|---|---|---|
| `basic_analysis` | — | The standard pipeline: auto-removal (the `constants:` cutoffs, once) and `removed_chambers.csv`, summary, feeding summary (and, with Facets, the faceted one), summary plot; for an optogenetic experiment the light QC, the light table, the yoked comparison (`yoked_comparison.csv`, a design with a yoked group only) and each DFM's Program timeline (not an older recording's); and the type's own outputs — Hedonic's plot and weighted-duration summary, Progressive Ratio's difference table, light QC tables, breaking point and figures. QC is `run_qc`'s job, and it does not read `global.omit` |
| `run_qc` | — | The Hub's **Run QC**: the QC Set minus `global.omit.qc`, into `qc/` — the integrity report, data breaks, simultaneous feeding and bleeding (two-well), the raw / baselined / cumulative-licks plots, and the optogenetic light QC, its figure and the Program timeline; Progressive Ratio's light QC and its two figures into `analysis/` |
| `run_analysis` | `binsize` | The Hub's **Run analysis**: the core (auto-removal and `removed_chambers.csv`, `summary.txt`, `feeding_summary.csv` and, with Facets, `feeding_summary_facet.csv`), then the Analysis Set minus `global.omit.analyses`, the experiment report last. `binsize` is the binned summary's and the report's |
| `feeding_csv` | — | Per-chamber feeding summary to CSV |
| `facet_csv` | — | Per-chamber summary per Facet to `feeding_summary_facet.csv`, which `run_analysis` writes as part of its core; skipped when the experiment has no Facets |
| `binned_csv` | `binsize` | Feeding metrics binned over time, to CSV |
| `weighted_duration` | — | Weighted-duration summary (hedonic experiments), to `treatment_means.csv` |
| `paired_yoked_diff` | — | Paired − yoked table, one row per chamber group per Facet (progressive ratio) |
| `breaking_point` | — | **Breaking point table**: per chamber group, the paired fly's lick-backed Test light events before its first pause longer than `pr_break_gap_min`. Writes `pr_breaking_point.csv` and logs the table with its sensitivity to the gap (progressive ratio) |
| `pr_light_qc` | — | **Light QC table**: per chamber group, did the paired fly earn its light? Writes `pr_light_qc.csv` and `pr_light_events.csv` and logs each flagged group (progressive ratio) |
| `opto_light_qc` | — | The QC panel's **Opto light QC tables**: per linkage group, was the light where the licks were? Writes `qc/opto/` (verdicts, intervals, light events, the program as read) and logs each flagged group (any optogenetic experiment; skipped otherwise) |
| `light_summary` | — | The Analyze panel's **Light tables**: what each chamber's and each well's light did — lit time, Activations, Acclimation, Firmware Events and the Paradigm type's own measures, with the Treatment, factor and `Role` columns. Writes `analysis/light_summary.csv`, `light_wells.csv` and, for a faceted experiment, their `_facet` variants (any optogenetic experiment; skipped otherwise) |
| `tidy_export` | `kind` | **Event statistics**: one row per individual event, for downstream statistics |
| `bootstrap` | `metric`, `mode`, `n_boot`, `ci`, `seed` | Bootstrap confidence intervals |
| `compare` | `metric`, `mode`, `model`, `factors` | ANOVA or linear mixed model across treatments |
| `light_phase_summary` | — | Feeding summary split by light and dark phase |
| `param_sensitivity` | `parameter`, `values` | Re-run across a range of one detection parameter |
| `transition_matrix` | — | Transition probabilities between bout types |
| `pdf_report` | `metrics`, `binsize` | The experiment report: cover, quality control, the type's inference figures with statistics, appendices — see [Reports](reports.md) |

**`tidy_export`** — the Analyze panel's **Event statistics** — takes `kind: feeding` (default) or `tasting`, and writes one row per event with its `StartMin`, `Duration`, `Licks`, `AvgIntensity` and `MaxIntensity`, labelled by chamber, well and treatment. The action key keeps its original spelling, so scripts already on disk are unaffected.

**`bootstrap`** defaults to `metric: PI` on a two-well layout and `metric: Licks` on a
single-well one, which has no PI, with `n_boot: 2000`, `ci: 0.95`, `seed: 0`. Its `mode` is
`total`, `A`, `B` or `diff` (A − B); it has no `mean_ab`. The fixed
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

**`run_qc`, `run_analysis` and `run_plots`** are the Hub's three buttons, so a script that
names them writes what a click writes. Each runs its set minus the experiment's Selection,
[`global.omit`](config-structure.md#globalomit), logs what it left out and what does not
apply, and removes no file. The other actions are the single items, and run whatever the
Selection says.

## Plot actions

| Action | Parameters | What it does |
|---|---|---|
| `run_plots` | `metric`, `binsize` | The Hub's **Generate plots**: the Plot Set minus `global.omit.plots`, each figure saved into `analysis/` and returned for a tab. `metric` is the Chosen metric's (the binned time course and the dot plot), `binsize` the binned time course's |
| `plot_feeding_summary` | — | Bar/jitter plot of per-chamber metrics |
| `plot_binned` | `metric`, `mode`, `binsize` | A metric binned over time, by treatment |
| `plot_dot` | `metric`, `mode` | Jittered dot plot of a metric by treatment; on a Progressive Ratio member, one point per chamber group — the paired-minus-yoked difference, panelled by Facet |
| `plot_moving_window` | `metric`, `mode`, `window`, `step` | Treatment mean ± SEM over a sliding window |
| `plot_moving_median_chambers` | `window`, `step`, `mode` | Time-dependent median bout duration, one line per chamber |
| `plot_moving_median_treatment` | `window`, `step`, `mode` | Treatment mean ± SEM of time-dependent median bout duration |
| `plot_well_comparison` | `metric` | Well A against Well B for one metric |
| `plot_hedonic` | — | Hedonic feeding plot |
| `plot_pr_cumulative_diff` | `binsize` | Paired − yoked cumulative licks since training end, by treatment; also writes `analysis/pr_cumulative_diff.csv` (progressive ratio) |
| `plot_pr_cumulative_licks` | `binsize` | Per-DFM training-aligned paired/yoked traces, light-on as points, lick-free light events as rings (progressive ratio, QC) |
| `plot_pr_light_events` | — | Per-DFM sucrose licks per Test light event; lick-free events as red rings (progressive ratio, QC) |
| `plot_pr_resting_level` | — | Per-DFM paired Sucrose Well resting level over the recording, light onsets as a rug (progressive ratio, QC) |
| `plot_pr_still_responding` | — | Fraction of paired flies whose breaking point reached each ratio, per treatment; censored flies as ticks (progressive ratio) |
| `plot_breaking_point` | — | Per-DFM ΔLicks per light-on period since training end, the break marked (progressive ratio) |
| `plot_opto_light` | `binsize` | Per-DFM lit time per bin, explained and unexplained by trigger-well licks, and the contact with no lick: the Activations' (on an older recording, the emulated firmware trigger's) (any optogenetic experiment, QC) |
| `plot_program_timeline` | `binsize` | **Program timeline**: per DFM, the Program Intervals coloured by Paradigm type, and per well its Acclimation, its Activations as ticks and the share of each bin its LED was on. Draws each DFM with Paradigm data or a `Program.txt` section (any optogenetic experiment; skipped otherwise, QC) |

`plot_program_timeline`'s `binsize` (10 minutes by default) is the width of the light
strip's bins. See [Program timeline](plots-catalog.md#program-timeline).

Every plot action also writes its figure into the member's `analysis/`:
`feeding_summary.png`, `binned_<metric>.png`, `dot_<metric>.png`,
`moving_window_<metric>.png`, `moving_median_duration_chambers.png`,
`moving_median_duration_treatments.png`, `well_comparison_<metric>.png`,
`hedonic_feeding_plot.png`, and for
Progressive Ratio `pr_cumulative_diff.png`, `pr_still_responding.png`, and per DFM
`pr_cumulative_licks_dfm<id>.png`, `pr_light_events_dfm<id>.png`,
`pr_resting_level_dfm<id>.png` and `breaking_point_dfm<id>.png`.  `plot_opto_light`
writes into `qc/opto/`, as `opto_light_dfm<id>.png`, and `plot_program_timeline` into
`qc/program/`, as `program_timeline_dfm<id>.png`.

## Project actions

These belong to **Project Scripts** (`project.yaml` `scripts:`) and cannot appear in an
Experiment Script, nor the ones above in a Project Script.

| Action | Parameters | What it does |
|---|---|---|
| `validate_design` | — | Fail the script early when a member contradicts the design |
| `run_in_experiments` | `script`, `only` | Run a named Experiment Script in every member, or only those listed — the one bridge between the levels. A member that fails, or has no script of that name, is logged and counted, and the step carries on |
| `run_all_analyses` | `skip_analyzed`, `reports` | Basic analysis in every member, with its QC reports, optionally skipping every member that already has a saved `feeding_summary.csv` (stale or not) and writing each member's PDF report. Like the Hub's **Analyze all**, it does not read `global.omit` |
| `build_combined_analysis` | — | Stack the members' saved summaries into the Combined Analysis, with pooled and mixed-model statistics |
| `project_report` | `ai_summary` | The Project Report; builds the Combined Analysis first when it is missing — see [Reports](reports.md#the-project-report) |
| `render_publication_figures` | `format`, `only` | Every figure the Project's data supports, each drawn by its Spec in `plot_specs.yaml` or its default, or only the ids in `only`, into `figures/` — `svg` (default) or `pdf`. Skipped with a log line when the Project has no `plot_specs.yaml` |
| `generate_ai_narrative` | `provider` | The [AI summary](concepts-ai-summary.md) of the Combined Analysis, into `analysis/ai_summary.txt`; `provider` is `anthropic` (default) or `openai` |

The Combined Analysis never analyses a member: one with no saved summary is omitted and
listed. For a Progressive Ratio Project it also stacks each member's
`paired_yoked_diff.csv`, `pr_light_qc.csv` and `pr_breaking_point.csv` into
`<project>_PairedYokedDiff.csv`, `<project>_LightQC.csv` and `<project>_BreakingPoint.csv`,
and `<project>_Stats.txt` adds the tests against zero and the breaking point's log-rank
test ([Statistics](concepts-progressive-ratio.md#statistics)). For a Project with
optogenetic members it stacks their light tables into `<project>_LightSummary.csv` and
`<project>_LightWells.csv` (and their `_Facet` variants), and their yoked comparisons into
`<project>_YokedComparison.csv`, and `<project>_Stats.txt` adds the light behaviour
comparisons of the headline metrics of the Paradigm Families the members' programs run.

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
    slice: first_4h
    steps:
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
