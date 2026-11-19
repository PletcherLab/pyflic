# Configuration file structure

`flic_config.yaml` has four top-level keys. `dfms` is always required. A standalone
experiment states `global` too; a member of a Project takes it from the design (below).
`data_folder` is needed only when the recording is not in this folder's `data/`.

```yaml
data_folder: ...                         # optional: where the recording is, when not data/
global:
  experiment_type: ...                   # or chamber_layout: for a Custom experiment
  transform_licks: ...                   # optional
  optogenetics: ...                      # optional: auto (default), true or false
  constants: { ... }                     # optional
  params: { ... }                        # optional
  facet_cutoffs: [ ... ]                 # optional
  facet_labels: [ ... ]                  # optional: names for the facet_cutoffs windows
  facet_source: ...                      # optional: cutoffs (default), program or program_cycles
  experimental_design_factors: { ... }   # optional, unless the type names factors
  well_names: { ... }                    # required by a type that names its wells
  control_level: ...                     # required under a type with a Treatment control
  exclusion_group: ...                   # optional, in a Project's design
  omit: { ... }                          # optional: what Run QC, Run analysis and Generate plots leave out

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

Validate at any time with `pyflic lint <experiment directory>`, which reports errors and
warnings with line numbers where it can.

**Inside a Project, `global:` belongs to the design.** A member's `flic_config.yaml`
normally holds only `dfms:` and its own `scripts:`; every `global:` key comes from the
`design:` in `project.yaml`, written in the same shape. See
[Projects and members](concepts-project.md).

> **`data_dir` no longer exists.** It is never read, whatever it says: the recording is
> read from `<experiment directory>/data/`, or from the folder `data_folder` names. The
> linter flags a leftover `data_dir` key so you can delete it. Old configs carry
> `data_dir` paths to folders on someone's own PC, which is why the new key has a new name.

## `data_folder`

The folder this experiment's recording is read from, when it is not the `data/` beside
`flic_config.yaml` — for a member whose recording is kept in a run folder that may not be
changed. It holds what `data/` would hold: the `DFM<id>_<n>.csv` files and, when PyMCU
exported them, `Program.txt`, `Messages.csv` and `Baselines.csv`.

```yaml
data_folder: ../../../../../runs/flic/2026/R-20261020-asmith-01/data
dfms:
  - id: 1
    chambers: {1: Ctrl, 2: Exp}
```

- **Relative to this folder**, with forward slashes. An absolute path works too, but
  `pyflic lint` warns that it breaks when the share is mounted somewhere else.
- **Only ever read.** The analysis, the QC, the reports, the cache and every Slice are
  written in this folder, as always; nothing is written in the folder `data_folder` names,
  and a member that names one is never filed.
- **This config's alone.** It goes at the top, never under `global:` (where `pyflic lint`
  reports it as an error, because it is not read there), and never in a Project's design.
  A member scaffolded or copied from another does not take it.
- **One recording, one member.** When two members of a Project read the same folder, the
  first by name is analysed and the other is reported as a `duplicate`. Members of
  different Projects may read one recording, each with its own settings.
- **Problems stop the member, not the Project.** A folder that does not exist, a file
  where a folder belongs, or a recording of the member's own beside the key (DFM CSVs at
  its root, or anything in its own `data/`) makes the member blocked, with the resolved
  path in the reason; loading it fails with the same words. An empty `data/` beside the
  key is fine. See [Blocked members](concepts-project.md#blocked-members).

The load message, `summary.txt`, the member's report and the Project Report's member
table say where the recording was read, as the resolved path with the key as written.

## `global.experiment_type`

The Experiment Type — `Hedonic`, `ProgressiveRatio`, or a type your lab built in the
[Experiment Type Editor](app-type-editor.md); omit it for a Custom Experiment, which states
`chamber_layout: single_well` or `two_well` instead (two-well when omitted). A config whose
type fixes the layout states neither `chamber_layout` nor `params.chamber_size`: the type
owns them. A lab type is read from `experiment_type.yaml` beside the config (or beside
`project.yaml`), which the editors write.

The type also decides what else the config must say:

- the `well_names` it requires (Hedonic and Progressive Ratio name both wells);
- the design factors it names, in `experimental_design_factors`;
- a `control_level`, when it has a Treatment control;
- the light it expects: when `data/Program.txt` runs light the type does not expect, the
  recording fails to load, unless `optogenetics: false` keeps pyflic from reading the
  program.

A Progressive Ratio config also says, per DFM, how its chambers form chamber groups:
`paired_chambers` for pairs, `chamber_groups: single` for single chambers, or neither when
`data/Program.txt`'s linkage gives them (see [DFMs and chambers](config-dfms-chambers.md)).
See [Experiment types](concepts-experiment-types.md).

## `global.control_level`

Under a type whose report leads with a **Treatment control**, the level of that factor
every other level is compared with, for example `control_level: w1118`. Such a type
requires it: a config without one does not load. It must be one of the factor's levels,
and it is an error under a type without a Treatment control. See
[Control comparisons](concepts-experiment-types.md#control-comparisons).

## `global.optogenetics`

Whether the optogenetic light QC runs: `auto` (the default) on every DFM with a section in
`data/Program.txt` or a lit LED, `true` on every DFM, `false` on none (`yes` and `no` mean
the same). A DFM entry may override it for that DFM. Its thresholds are more `constants:`,
shared by every type. See [Optogenetic experiments](concepts-optogenetics.md).

```yaml
global:
  optogenetics: true
```

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

Detection parameters — thresholds, baseline window, link gap, dual-feeding correction.
Every entry is documented in [Parameter reference](reference-parameters.md).

These are defaults for the whole experiment. A standalone experiment's DFM may override any
of them for itself; inside a Project only the physical keys `pi_direction` and
`chamber_sets` may vary per DFM.

## `global.facet_cutoffs`

Minutes at which the recording is split into named phases, with optional `facet_labels`,
one name per window. A type may suggest cutoffs of its own, used when the config states
none. Progressive Ratio owns its facets (Training and Test, split at each chamber group's
training end), and so does a lab type that fixes its cutoffs: their configs must not state
them. See [Facets](concepts-facets.md).

## `global.facet_source`

Where the Facets come from:

| Value | Facets |
|---|---|
| `cutoffs` (the default) | `facet_cutoffs`, the same minute windows on every DFM |
| `program` | one per Program Interval of each DFM's `Program.txt` section, labelled by its Paradigm's name, its Cycles pooled |
| `program_cycles` | one per Program Interval and Cycle, labelled `<Paradigm> C<n>` |

A Paradigm a DFM runs in two Program Intervals is labelled `I<k> <Paradigm>`. Older
recordings lay the Facets out from the program's clock schedule, which `summary.txt` notes
as approximate.

```yaml
global:
  optogenetics: auto
  facet_source: program
```

`pyflic lint` and the loader refuse `program` or `program_cycles` without a `data/Program.txt`
section for every DFM, beside `facet_cutoffs`, or under Progressive Ratio, which owns its
Facets. See [Facets](concepts-facets.md#program-facets).

## `global.constants`

Cutoffs used by automatic chamber removal. They are not applied at load: they take effect
when `auto_remove_chambers()` runs, which **an analysis does once, before it writes the
summary** — the Hub's *Run analysis* and *Analyze all*, a Batch Run and
`execute_basic_analysis()` all apply them, and every output describes the filtered design.
The QC Viewer's **Auto Filter** and the Python API run the same removal on demand.

| Key | Applies to | Effect |
|---|---|---|
| `min_untransformed_licks_cutoff` | all experiment types | Remove a chamber if any of its wells has a lick count below this |
| `max_med_duration_cutoff` | every two-well type | Remove a chamber if `MedDurationA` or `MedDurationB` is above this |
| `max_events_cutoff` | every two-well type | Remove a chamber if `EventsA` or `EventsB` is above this |
| `max_ceiling_minutes` | every experiment type (default `30`) | Remove a chamber if any of its wells rests at the [signal ceiling](#the-signal-ceiling) for more than this many minutes of the analysed range |

An Experiment Type supplies its own defaults for the first three (Hedonic and Progressive
Ratio set all three); `max_ceiling_minutes` defaults to 30 under every type, Custom and a
lab's own included. A yaml number always wins over the default. A yaml `null` also wins and
means **that filter is off** (the Config Editor and Project Design dialog write `null` when
the cutoff's **Use** checkbox is unchecked). An omitted key still inherits the default.
A Custom Experiment has no defaults for the first three, and there an unset or null key
means the check is not performed — pyflic reports it as "not configured" rather than
substituting a number.

### The signal ceiling

A DFM reads each well through a 10-bit converter, so its signal cannot exceed 1023, and
many boards saturate a little below that (1014 to 1022, always the same level on the same
board). A well that rests up there cannot rise when the fly touches it, so its licks
vanish for as long as it stays. A wet pad, food wicking onto it or condensation usually
climbs there over hours.

pyflic counts a well **at the ceiling** in each minute whose per-minute median raw signal
(its resting level) is 1000 or more. The median ignores the fly's brief contacts, so a hard
lick that touches 1023 never counts. Every DFM's integrity report lists each well's minutes
at the ceiling, and the minutes every well spent there together, which point to the board
itself rather than one pad. `max_ceiling_minutes` then removes a chamber with a well at the
ceiling for longer than the cutoff, counting only the minutes inside the analysed range: a
pad that saturated at hour 10 leaves the whole recording and stays in a Slice that ends at
hour 6. With the cutoff off, the integrity report and the experiment report still list the
wells.

A type with the Progressive Ratio analysis (the shipped `ProgressiveRatio`, or a lab type
built with it) adds its own, all with defaults:

| Key | Default | Effect |
|---|---|---|
| `require_training_complete` | `true` | Remove every chamber of a chamber group whose paired fly never finished training (both of a pair, the one of a single chamber) |
| `exclude_failed_pr_groups` | `true` | Remove every chamber of a chamber group that fails the light QC (self-triggered light, implausible training, *requirement series wrong*); `false` keeps them, still listed in `pr_light_qc.csv` |
| `pr_lick_free_run` | `5` | Consecutive Test light events with no sucrose licks that make a group's light self-triggered |
| `pr_trend_min_events` | `5` | Test light events needed before the licks-per-event trend is judged |
| `pr_trend_min_rho` | `0.3` | Spearman's rho below which a group gets the *no increasing trend* warning |
| `pr_resting_level_rise` | `15` | Counts the sucrose well's resting level may rise above its first 30 minutes before a warning; also the margin an *elevated* well must clear |
| `pr_resting_level_ratio` | `3` | Times the DFM's other sucrose wells a paired sucrose well may rest at before the *elevated* warning |
| `pr_break_gap_min` | `120` | Minutes. A pause longer than this ends a fly's responding: the breaking point counts the paired fly's lick-backed Test light events before its first such pause, and sucrose persistence is the time to either fly's last sucrose feeding event before one |
| `pr_test_window_min` | unset (off) | Minutes. Caps every chamber group's Test window at this long after its own training end, so a group that trained late is not measured over less time; `0` or unset means no cap |
| `pr_simulated_yoked_bp` | `false` | Adds the **Simulated Yoked Breaking Point**: the Test Paradigm replayed on each yoked fly's own sucrose contact, which gives the breaking point background feeding alone would have earned. Also adds `dBreakingPoint`, paired minus simulated yoked. Needs a firmware 4 recording whose ratio Paradigm counts Contact |

What the light QC checks, and how the breaking point is defined, is in
[Progressive Ratio experiments](concepts-progressive-ratio.md). The Project Design dialog
and the [Config Editor](app-config-editor.md) show these as fields under such a type, the
type's default in grey.

Every optogenetic experiment, of any type, reads the optogenetic light QC's constants too —
`exclude_failed_opto_chambers` (off by default: failed linkage groups are flagged, not
excluded) and twenty thresholds whose names start `opto_`, thirteen of which use the
Paradigm columns and so do not apply to older recordings. They are listed in
[Optogenetic experiments](concepts-optogenetics.md#thresholds). Both editors show them as
fields too, in an *Optogenetic light QC* group, for an experiment whose optogenetics is not
off, the default in grey.

A value of the wrong kind or out of range — a switch that is not `true` or `false`, a gap
of 0, a rho above 1 — is a validation error: `pyflic lint` reports it and the experiment
does not load.

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
[Two-well choice and the preference index](concepts-two-well-pi.md). A type may still
require them: Hedonic and Progressive Ratio refuse a config that does not name both wells.
A DFM entry may state its own `well_names:`, which override these, well by well, for that
DFM.

## `global.experimental_design_factors`

Defines a factorial design so chambers can be labelled by several crossed factors at once.
An Experiment Type may name factors that every config of it declares; the config gives
their levels and may add factors of its own. See [Factorial designs](config-factors.md).

## `global.omit`

What the member's runs leave out: the boxes unticked on the Hub's QC, Analyze and Plots
panels. Three lists of item ids, one per set:

```yaml
global:
  omit:
    qc: [bleeding]                        # Run QC
    analyses: [event_statistics]          # Run analysis
    plots: [dot_plot, well_comparison]    # Generate plots
```

Leave it out, or any of its lists, and everything runs. It stores what is left *out*, so an
item a later pyflic adds is included until someone leaves it out. The Hub writes it as you
tick and untick, so you rarely write it by hand.

- **Who reads it.** The Hub's **Run QC**, **Run analysis** and **Generate plots**, and a
  script's `run_qc`, `run_analysis` and `run_plots` steps. The Project's **Analyze all**, a
  Batch Run's `run_all_analyses`, `basic_analysis` and the single-output script actions do
  not: they run what they always ran.
- **Inside a Project** it belongs to the design, like every `global:` key: `project.yaml`
  `design.global.omit`, shared by every member. Ticking a member's box writes there, and
  removes any `omit:` a member's own `global:` states, so every member inherits the
  design's.
- **What it cannot leave out.** Run analysis always writes the core — the auto-removal and
  `removed_chambers.csv`, `summary.txt`, `feeding_summary.csv` and, with Facets,
  `feeding_summary_facet.csv` — because every other item reads it.
- **Nothing is deleted.** An item left out keeps whatever an earlier run wrote; the log and
  `summary.txt` say that this run left it out.
- **`pyflic lint`** warns of an id no run produces: a misspelt id leaves nothing out.

| List | Ids, in panel order |
|---|---|
| `qc` | `integrity`, `data_breaks`, `raw_signal`, `baselined_signal`, `cumulative_licks`, `simultaneous_feeding`, `bleeding`, `opto_light_qc`, `opto_light_figures`, `program_timeline`, `pr_light_qc`, `pr_light_event_figures`, `pr_resting_level_figures` |
| `analyses` | `binned_summary`, `event_statistics`, `light_tables`, `yoked_comparison`, `paired_yoked_diff`, `cumulative_diff`, `breaking_point`, `weighted_duration`, `experiment_report` |
| `plots` | `feeding_summary_plot`, `well_comparison`, `binned_time_course`, `dot_plot`, `pr_cumulative_diff_plot`, `pr_cumulative_licks`, `pr_still_responding`, `pr_breaking_point_plots`, `hedonic_plot` |

What each item writes, and which members it is offered for, is in the Analysis Hub's
[QC panel](app-hub.md#qc-panel), [Analyze panel](app-hub.md#analyze-panel) and
[Plots panel](app-hub.md#plots-panel) tables.

## `global.exclusion_group`

The `remove_chambers.csv` group the whole recording is loaded with, and the one the QC
Viewer saves its removals to: `general` when it is left out. In a Project it belongs in
the design, where the Project Design dialog's **Exclusion group** field writes it; a
standalone experiment states it in its own `global:`, where the Config Editor writes it.
Each Slice has its own group (see [Slices](concepts-slices.md)). See
[Excluding chambers](config-dfms-chambers.md#excluding-chambers).

## `dfms`

One entry per physical device, keyed by the device number used in the CSV filenames. See
[DFMs, chambers and exclusions](config-dfms-chambers.md).

## `scripts`

Named pipelines that run from the hub in one click. See
[What a script is](scripts-overview.md).

## A complete example

```yaml
global:
  experiment_type: Hedonic
  transform_licks: true
  constants:
    min_untransformed_licks_cutoff: 20
    max_med_duration_cutoff: 13
    max_events_cutoff: 150000
  params:
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

## A Progressive Ratio example

```yaml
global:
  experiment_type: ProgressiveRatio
  well_names: {A: Sucrose, B: Yeast}     # well A is always the sucrose well
  constants:
    pr_break_gap_min: 120                # the pause that ends a breaking point
    pr_test_window_min: 720              # optional: the same Test window for every group

dfms:
  1:
    params: {pi_direction: left}         # which side well A is on
    paired_chambers: [1, 4, 5]           # one from each chamber group
    chambers: {1: Ctrl, 2: Ctrl, 3: Exp, 4: Exp, 5: Exp, 6: Exp}
  2:
    params: {pi_direction: right}
    paired_chambers: [2, 3, 6]
    chambers: {1: Exp, 2: Exp, 3: Ctrl, 4: Ctrl, 5: Exp, 6: Exp}
  3:
    params: {pi_direction: left}
    chamber_groups: single               # six chambers, no yoked controls
    chambers: {1: Ctrl, 2: Exp, 3: Ctrl, 4: Exp, 5: Ctrl, 6: Exp}
```

No `facet_cutoffs`, no `chamber_layout`: the type owns both. With `Program.txt` in `data/`,
a DFM may name neither `paired_chambers` nor `chamber_groups`, and its linkage gives the
groups. See
[Progressive Ratio experiments](concepts-progressive-ratio.md).
