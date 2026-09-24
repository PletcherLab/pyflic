# Configuration file structure

`flic_config.yaml` has three top-level keys. Only `global` and `dfms` are required.

```yaml
global:
  experiment_type: ...
  transform_licks: ...                   # optional
  constants: { ... }                     # optional
  params: { ... }
  experimental_design_factors: { ... }   # optional
  well_names: { ... }                    # optional

dfms:
  1:
    params: { ... }                      # optional; overrides global for this DFM
    chambers:
      1: TreatmentA
      2: TreatmentB

scripts:                                 # optional
  - name: "My pipeline"
    steps:
      - action: load
      - action: basic_analysis
```

Validate at any time with `pyflic lint <project directory>`, which reports errors and
warnings with line numbers where it can.

> **`data_dir` no longer exists.** Data is always read from
> `<project directory>/data/`. The linter flags a leftover `data_dir` key so you can
> delete it.

## `global.experiment_type`

The Experiment Type — `Hedonic` or `ProgressiveRatio`; omit it for a Custom Experiment,
which states `chamber_layout` instead. A Progressive Ratio config also needs
`paired_chambers` on every DFM (see [DFMs and chambers](config-dfms-chambers.md)). See
[Experiment types](concepts-experiment-types.md).

## `global.transform_licks`

Whether lick counts in the feeding summary are fourth-root transformed. **Defaults to
`true`.** This changes every lick number you read, so see
[the lick transform](concepts-metrics.md#the-lick-transform) before leaving it at the
default or changing it.

```yaml
global:
  transform_licks: false
```

## `global.params`

Detection parameters — thresholds, baseline window, link gap, chamber size. Every entry is
documented in [Parameter reference](reference-parameters.md).

These are defaults for the whole experiment. Any DFM may override any of them for itself.

## `global.constants`

Cutoffs used by automatic chamber removal. They are not applied at load: they take effect
when `auto_remove_chambers()` runs, which **basic analysis does once, before it writes the
summary** — the Hub's *Basic analysis*, *Analyze all*, a Batch Run and
`execute_basic_analysis()` all apply them, and every output describes the filtered design.
The QC Viewer's *Auto Filter Chambers* and the Python API run the same removal on demand.

| Key | Applies to | Effect |
|---|---|---|
| `min_untransformed_licks_cutoff` | all experiment types | Remove a chamber if any of its wells has a lick count below this |
| `max_med_duration_cutoff` | hedonic, progressive ratio | Remove a chamber if `MedDurationA` or `MedDurationB` is above this |
| `max_events_cutoff` | hedonic, progressive ratio | Remove a chamber if `EventsA` or `EventsB` is above this |

An Experiment Type supplies its own defaults for these (Hedonic and Progressive Ratio set
all three); a yaml value always wins. A Custom Experiment has none, and there an unset key
means the check is not performed — pyflic reports it as "not configured" rather than
substituting a number.

A Progressive Ratio experiment adds its own, all with defaults:

| Key | Default | Effect |
|---|---|---|
| `require_training_complete` | `true` | Remove both chambers of a chamber group whose paired fly never finished training |
| `exclude_failed_pr_groups` | `true` | Remove both chambers of a chamber group that fails the light QC (self-triggered light, implausible training); `false` keeps them, still listed in `pr_light_qc.csv` |
| `pr_lick_free_run` | `5` | Consecutive Test light events with no sucrose licks that make a group's light self-triggered |
| `pr_trend_min_events` | `5` | Test light events needed before the licks-per-event trend is judged |
| `pr_trend_min_rho` | `0.3` | Spearman's rho below which a group gets the *no increasing trend* warning |
| `pr_resting_level_rise` | `15` | Counts the sucrose well's resting level may rise above its first 30 minutes before a warning; also the margin an *elevated* well must clear |
| `pr_resting_level_ratio` | `3` | Times the DFM's other sucrose wells a paired sucrose well may rest at before the *elevated* warning |

What the light QC checks, and why, is in
[Experiment types](concepts-experiment-types.md#progressive-ratio). The Project Design
dialog shows these as fields for a Progressive Ratio design.

One rule applies regardless of configuration: a chamber whose lick value is `NaN` or
undefined is always removed by `auto_remove_chambers()`, because it produced no usable
data at all.

```yaml
global:
  constants:
    min_untransformed_licks_cutoff: 20
```

Note the name: the cutoff is compared against **untransformed** lick counts, so use a real
lick count here even when `transform_licks` is on.

## `global.well_names`

Human-readable labels for the two wells of a chamber, used in plots and reports.

```yaml
global:
  well_names:
    A: Sucrose
    B: Yeast
```

Purely cosmetic — it does not affect detection, the preference index, or which well is
Well A. That is `pi_direction`'s job; see
[Two-well choice and the preference index](concepts-two-well-pi.md).

## `global.experimental_design_factors`

Defines a factorial design so chambers can be labelled by several crossed factors at once.
See [Factorial designs](config-factors.md).

## `dfms`

One entry per physical device, keyed by the device number used in the CSV filenames. See
[DFMs, chambers and exclusions](config-dfms-chambers.md).

## `scripts`

Named pipelines that run from the hub in one click. See
[What a script is](scripts-overview.md).

## A complete example

```yaml
global:
  experiment_type: hedonic
  transform_licks: true
  constants:
    min_untransformed_licks_cutoff: 20
    max_med_duration_cutoff: 13
    max_events_cutoff: 150000
  params:
    chamber_size: 2
    pi_direction: left
    baseline_window_minutes: 3
    samples_per_second: 5
    feeding_threshold: 20
    feeding_minimum: 10
    feeding_minevents: 1
    feeding_event_link_gap: 5
    tasting_minimum: 5
    tasting_maximum: 20
    tasting_minevents: 1
    correct_for_dual_feeding: true
  experimental_design_factors:
    paired: [Paired, Unpaired]
    genotype: [Chrim, WCS]
  well_names:
    A: Sucrose
    B: Yeast

dfms:
  1:
    params:
      pi_direction: left
    chambers:
      1: Paired,WCS
      2: Unpaired,WCS
  2:
    params:
      pi_direction: right      # counterbalanced against DFM 1
    chambers:
      1: Unpaired,Chrim
      2: Paired,Chrim
```
