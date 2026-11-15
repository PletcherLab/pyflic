# Python API

Everything the GUI does is available from Python, which is where to go when you need
something the interface does not offer — a custom figure, an analysis of your own, or a
pipeline driven from a notebook.

## Loading an experiment

```python
from pyflic import load_experiment_yaml

exp = load_experiment_yaml(
    "/path/to/experiment_dir",       # the folder holding flic_config.yaml
    range_minutes=(0, 0),            # (start, end); (0, 0) = whole recording
    parallel=True,                   # load DFMs concurrently
    eager=True,                      # pre-compute the feeding summary now
    use_disk_cache=True,             # read/write .pyflic_cache/
    exclusion_group="general",       # a remove_chambers.csv group; default the config's
                                     # exclusion_group:, None excludes nothing
)
```

`load_experiment_yaml` reads the configuration and returns the right subclass for the
experiment type — you do not choose the class yourself.

Note `exclusion_group`: it defaults to applying the `general` group. Pass `None` if you
want the unfiltered design, and be deliberate about which you want, because the two give
different numbers. See [exclusions](config-dfms-chambers.md#excluding-chambers).

To load a [Slice](concepts-slices.md) the design declares, pass its name:
`load_experiment_yaml(path, slice="first_6h")`. The recording is clipped to the Slice, its
own exclusion groups replace the whole recording's, and its outputs go to
`slices/first_6h/`. A loaded experiment's `exp.load_slice("first_6h")` does the same with the
options it was loaded with, and raises `SliceNotCovered` when the recording stops short of
the Slice's end. For a Project, `project.for_slice("first_6h")` is the Project seen through
that Slice: its `load_member`, `combined_frames` and `build_combined_analysis` work on it.

## Loading a Project's members

A member of a Project inherits the design's `global:` block, so load it through its
Project rather than on its own — the Project supplies the design and the design's
exclusion group:

```python
from pyflic import Project

project = Project("/path/to/project")
project.member_names                  # the members, in name order
exp = project.load_member("rep1")     # a loaded Experiment, design applied
project.build_combined_analysis()     # stack the members' saved summaries
```

```python
from pyflic import write_project_report
write_project_report(project)         # builds the Combined Analysis if missing
```

## Class hierarchy

```
Experiment (base)
├── SingleWellExperiment          chamber_size=1
└── TwoWellExperiment             chamber_size=2
    ├── HedonicFeedingExperiment
    └── ProgressiveRatioExperiment
```

All subclasses share the same core API; the specialised methods — weighted durations,
the Progressive Ratio tables below — live on the subclasses. The class follows the
Experiment Type: a type with the Hedonic analysis gives a `HedonicFeedingExperiment`, one
with the Progressive Ratio analysis a `ProgressiveRatioExperiment`, and any other the class
of its chamber layout. See [Experiment types](concepts-experiment-types.md).

## Working with an experiment

```python
exp.dfms                      # {dfm_id: DFM}
dfm = exp.get_dfm(1)
exp.design                    # ExperimentDesign: each treatment and its chambers
exp.design_factors            # the factor names, in declared order
exp.chamber_factors           # {(dfm_id, chamber): {factor: level}}

exp.write_qc_reports()        # write QC output to disk
exp.compute_qc_results()      # the same results as Python dicts

df = exp.feeding_summary()
df = exp.feeding_summary(range_minutes=(30, 90))
binned = exp.binned_feeding_summary(binsize_min=30)

exp.execute_basic_analysis()  # the whole standard pipeline

print(exp.summary_text())
exp.write_summary()
```

Plotting returns figure objects you can modify before saving:

```python
fig = exp.plot_feeding_summary()
p   = exp.plot_binned_metric_by_treatment(metric="Licks", binsize_min=30)
fig = exp.plot_dot_metric_by_treatment(metric="MedDuration")
fig = exp.plot_cumulative_licks_chamber(dfm_id=1, chamber=1)
```

Some return matplotlib figures and some return plotnine `ggplot` objects — save the former
with `fig.savefig(...)` and the latter with `p.save(...)`.

```python
from pyflic import write_experiment_report
write_experiment_report(exp)
```

## Progressive Ratio experiments

A `ProgressiveRatioExperiment` adds the chamber-group tables and figures. Every method
works on the loaded design, so run `exp.auto_remove_chambers()` first — as basic analysis
does — when you want the numbers the pipeline reports.

```python
exp.roles_table()                 # DFM, Chamber, Group, Role, Partner
exp.training_table()              # per chamber group: training end, complete or not
exp.paired_yoked_diff()           # paired − yoked, per group per Facet
exp.light_qc_table()              # the light QC verdict per group
exp.light_events_ledger()         # every Test light event, with Counted (and Requirement)

exp.breaking_point_summary()      # BreakingPoint, BreakMin, Censored, BreakRequirement, ...
exp.breaking_point_sensitivity(gaps=[60, 120, 240])
exp.group_break(dfm_id=1, group=2)            # the first-gap rule's verdict for one group
exp.chamber_persistence(dfm_id=1, chamber=3)  # (minutes, censored)

exp.write_paired_yoked_diff()
exp.write_light_qc()
exp.write_breaking_point()        # analysis/pr_breaking_point.csv

p = exp.plot_cumulative_diff()
p = exp.plot_still_responding()
p = exp.plot_breaking_point_dfm(1)
p = exp.plot_light_events_dfm(1)
```

The rule's settings come from the design's `constants:` (`pr_break_gap_min`,
`pr_test_window_min`); `exp.break_settings()` shows them. See
[Progressive Ratio experiments](concepts-progressive-ratio.md).

## Optogenetic experiments

Every experiment, of any type, has `exp.opto`, its optogenetic light QC. It covers the DFMs
the `optogenetics:` setting names, and `exp.is_optogenetic` says whether there are any.
`exp.opto_program` is the parsed `data/Program.txt`, or `None`.

```python
exp.is_optogenetic                # any DFM covered?
exp.opto.dfm_ids()                # which
exp.opto.qc_table()               # the verdict per linkage group, with the rule checks
exp.opto.interval_table()         # per group per scheduled interval
exp.opto.events_table()           # every light event: UnexplainedSec, OverrunSec, the ledger
exp.opto.program_table()          # the program as read
exp.opto.lines()                  # the flagged groups, in words
exp.opto.write()                  # qc/opto/*.csv
p = exp.opto.plot_dfm(1, binsize_min=10)

from pyflic.base.opto_program import read_program
program = read_program("data/Program.txt")
program.section(1).linkage_groups()          # {1: (1, 2, 3, 4), ...}
program.section(1).intervals[0].paradigm.describe()   # "PR10 (ProgressiveRatio): ..."
program.grammar                              # "legacy", "v4" or "simple"
program.paradigms["PR10"].requirement_for(3) # the 3rd Main Activation's Requirement
program.interval_for(1, 2)                   # DFM 1's Program Interval 2
```

See [Optogenetic experiments](concepts-optogenetics.md).

## Firmware data

A few modules read what the firmware recorded. Each takes a loaded experiment or DFM.

**The Paradigm state.** Every `DFM` has a `generation` (3 on firmware 4.0.1; 1 or 2 on an
older recording) and, on generation 3, a `paradigm`: the 63 firmware columns as arrays, one
row per `raw_df` row, wells numbered 1–12. `raw_df` does not hold them.

```python
dfm.generation                    # 1, 2 or 3
dfm.training_info                 # "flag", "none" or "phase"
dfm.light_source                  # "OnMs, exact" or "OptoCol1, sampled at 5 Hz"
dfm.light_seconds()               # seconds lit, wells 1-12
state = dfm.paradigm              # None before generation 3
state.interval_runs()             # runs of rows per Program Interval and Cycle
state.activation_onsets(1)        # rows where well 1's Activations rose
state.training_end(1)             # first Main row under a ratio Paradigm
state.requirement_met(1, onset)   # the Requirement that Activation met
```

**Roles** (`pyflic.base.opto_roles`): `dfm_linkage_view(dfm, section)` gives a DFM's light
circuits and trigger wells, from the program or the data, and
`chamber_roles(view, dfm_chamber_map(dfm))` its chambers' Roles.

**The light table** (`pyflic.base.light_summary`): `light_summary(exp)` and
`light_wells(exp)` return the per-chamber and per-well tables, `light_summary_facet(exp)`
and `light_wells_facet(exp)` the same per Facet, and `write(exp)` writes all four into
`analysis/`. `experiment_headline_metrics(exp)` names the metrics tested between
treatments, those of the Paradigm Families the member's program runs;
`HEADLINE_METRICS` is the four a member without `Program.txt`, or with an older
recording, keeps. `behaviour_rows(frame)` gives the rows those tests use: a chamber whose
Role does not earn light keeps its lit time and loses its Activation measures.

**Paradigm Families** (`pyflic.base.paradigm_families`): `FAMILIES`, `family_of(mode,
acclimation=False)`, `families_in(program, dfm_ids=None)`,
`headline_metrics(families, max_on=False, acclimation=False)` and `rule_sentence(paradigm)`,
the rule in words with its settings.

**The Activation ledger** (`pyflic.base.activation_ledger`): `ledger(state)` returns one row
per Activation of a generation-3 DFM's Paradigm state, timed to the millisecond
(`OnsetMs`, `EndMs`, `LitMs`, `Overlap`, `Boundary`, `K`, …), the rows
`opto_light_events.csv` carries. The rule checks over it are in `pyflic.base.rule_checks`,
one function per check, each returning a `CheckResult`.

**The yoked comparison** (`pyflic.base.analytics`): `yoked_groups(exp)` lists the linkage
groups that yoke chambers to a paired one, and `yoked_comparison(exp, metrics=None)` returns
the paired − yoked table basic analysis writes as `analysis/yoked_comparison.csv`; test its
`d<metric>` columns against zero with `zero_tests`.

**Program Facets** (`pyflic.base.program_facets`): `exp.facet_source` is `cutoffs`,
`program` or `program_cycles`; `exp.facet_windows_for(dfm_id)` gives a DFM's Facets, each a
label with one or more `(start, end)` windows.

**Messages** (`pyflic.base.messages`): `read(data_dir)` returns `Messages.csv` as a table
(`DFM, Time, Index, Message, MsgType, Repeats`), empty when there is none;
`for_dfm(frame, dfm_id)` keeps one DFM's messages and the MCU's own, and `errors(frame)`
the errors.

**The Program timeline** (`pyflic.base.program_timeline`): `plot_dfm(exp, dfm_id,
binsize_min=10)` returns the figure, `well_table(exp, dfm_id)` the state of each well at the
end of the loaded range, `activations_table(exp, dfm_id)` the DFM's Activations from the
light events, and `write(exp)` saves every DFM's figure to `qc/program/`.

**Restarts**: `dfm.restarts()` lists a generation-3 DFM's restarts, from the rows and from
`Messages.csv` (`None` before generation 3), and `dfm.paradigm.restarts()` the spans the rows
show.

A `ProgressiveRatioExperiment` adds `exp.training_information()` (the *no training
information* text, or `None`), `exp.test_paradigm(dfm_id, group)` (the name and the
Paradigm of a group's Test), `exp.chamber_groups(dfm_id)` (its Chamber Groups, pairs or solo
groups), `exp.has_yoked_controls()` and `exp.test_restart(dfm_id, group)` (a DFM restart
that ends the group's Test window, or `None`). See
[Generations](concepts-optogenetics.md#generations).

## DFM objects

A `DFM` exposes every intermediate stage of the pipeline, which makes it the right level
for checking what detection actually did:

```python
dfm = exp.get_dfm(1)

dfm.raw_df          # the raw CSV
dfm.baseline_df     # after baseline subtraction
dfm.lick_df         # boolean: True where a feeding lick was detected
dfm.event_df        # integer: run length at each event start, 0 elsewhere
dfm.tasting_df      # boolean tasting licks
dfm.lights_df       # per-well light state
dfm.durations       # per-well bout durations
dfm.intervals       # per-well inter-bout intervals

dfm.feeding_summary()
dfm.plot_raw()
dfm.plot_baselined()
dfm.plot_cumulative_licks()
```

`event_df` uses an unusual encoding worth knowing: it is zero everywhere except at the
*first sample* of each event, where it holds that event's length in samples. Counting
events means counting non-zero entries, not summing them.

## Analytics functions

Most of these are also [script actions](scripts-actions.md) — `tidy_export`, `bootstrap`,
`compare`, `light_phase_summary`, `param_sensitivity` and `transition_matrix` — and
`tidy_events` is the Hub's *Event statistics* item in the Analyze panel. All but
`compare_configs` and `lint_flic_config` take a loaded experiment.

```python
from pyflic import (
    tidy_events, bootstrap_metric, compare_treatments,
    light_phase_summary, parameter_sensitivity,
    bout_transition_matrix, compare_configs, lint_flic_config,
)
```

**`tidy_events(exp, kind="feeding")`** — one row per bout: `DFM`, `Chamber`, `Well`,
`WellLabel`, `WellName`, `Treatment`, factor columns, `StartMin`, `Licks`, `Duration`,
`AvgIntensity`, `MaxIntensity`. The right starting point for analysis in another tool.

**`bootstrap_metric(exp, metric="PI", n_boot=2000, ci=0.95, seed=0)`** — percentile
confidence intervals, resampling **at the chamber level** because chambers are the
independent biological units. Use it for `PI` and other bounded or skewed metrics, where a
parametric standard error is misleading. `res.summary` has the group column
(`res.group_col`: `Treatment`, or `_Group` when factors are declared), `n`, `mean`, `sem`,
`ci_low`, `ci_high`.

**`compare_treatments(exp, metric="MedDuration", model="aov", posthoc="tukey")`** — Type II
ANOVA, or `model="lmm"` for a linear mixed model with DFM as a random intercept. Prefer the
mixed model when chambers within a device might be correlated, which is usually the safer
assumption. Returns `res.table` and `res.posthoc`.

**`light_phase_summary(exp)`** — metrics split by light and dark phase. Normalise by
`PhaseSeconds` before comparing; see
[Light state and phase analysis](concepts-light-phase.md).

**`parameter_sensitivity(exp, parameter=..., values=[...])`** — re-runs the full pipeline at
each value and reports how `Licks`, `Events` and `MedDuration` respond. `res.grid` has the
parameter, `Group`, `n_chambers`, and mean/SEM for each metric.

**`bout_transition_matrix(exp)`** — consecutive bout transitions per chamber: `FromWell`,
`ToWell`, `Count`. Distinguishes a fly alternating between wells from one that settled on
a single option — a distinction the preference index cannot make, since both can give the
same PI.

**`compare_configs("a/", "b/", metrics=(...))`** — per-treatment means from two standalone
experiment directories side by side: `Group`, `Metric`, `mean_a`, `mean_b`, `delta`,
`pct_change`.

**`lint_flic_config("project/flic_config.yaml")`** — validation results as objects:

```python
for issue in lint_flic_config("project/flic_config.yaml"):
    print(issue.format())
```

## Batches and folder layout

A Batch is a folder with Projects anywhere beneath it; these are what the Hub's Batch panel
runs:

```python
from pyflic import apply_exclusion_sheet, discover_projects, run_batch

discover_projects("/path/to/batch")       # every Project under it, in run order
apply_exclusion_sheet("/path/to/batch")   # the batch's remove_chambers.csv into each member
run_batch("/path/to/batch")               # {project: "ok" or its error}
```

`run_batch(batch_dir, script_name=None, project_names=None, apply_exclusions=True)` runs
one Project Script in every Project, carrying on past a failure: `script_name=None` takes
the designation in `batch.yaml`, and with none each Project runs its own default. It
applies the exclusion sheet first unless `apply_exclusions=False`. See
[Running many projects at once](scripts-batch.md).

`members_in(project_dir)` says what each folder of a Project is — a healthy or a Blocked
Member — and `classify(directory)` the same for one folder; `file_recording(directory)`
moves a member's loose recording into `data/`, as the Hub's *File unfiled recordings* does.
See [Blocked members](concepts-project.md#blocked-members).

## Jupyter notebooks

A set of tutorial notebooks exists in the repository under `doc/ToBeDepricated/`:

| Notebook | Covers |
|---|---|
| `01_GettingStarted.ipynb` | Loading experiments; the detection pipeline |
| `02_GroupedAnalysis.ipynb` | Treatment groups and factorial designs |
| `03_ChoiceChamberAnalysis.ipynb` | Two-well choice experiments and PI |
| `HedonicFeeding.ipynb` | Hedonic feeding experiments |
| `ProgressiveRatio.ipynb` | Progressive-ratio experiments |

They are **deprecated** and will be removed in a future release. They exist for
continuity with the original R workflow, and they are not updated alongside the code — if
one disagrees with this help, the help is right.

`ProgressiveRatio.ipynb` in particular predates chamber groups, the light QC and the
breaking point, and uses the retired "configuration 1–4" argument.

They are also not installed: they live in the repository, not in the package. Everything
they demonstrate is covered by the hub, the [scripting system](scripts-overview.md), and
the API above. New work should start there.

---

Related: [Summary metrics](concepts-metrics.md) ·
[Performance and caching](performance.md) · [Script actions](scripts-actions.md)
