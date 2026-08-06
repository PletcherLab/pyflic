# Python API

Everything the GUI does is available from Python, which is where to go when you need
something the interface does not offer — a custom figure, a analysis of your own, or a
pipeline driven from a notebook.

## Loading an experiment

```python
from pyflic import load_experiment_yaml

exp = load_experiment_yaml(
    "/path/to/project_dir",
    config_name="flic_config.yaml",  # which YAML in the project directory
    range_minutes=(0, 0),            # (start, end); (0, 0) = whole recording
    parallel=True,                   # load DFMs concurrently
    eager=True,                      # pre-compute the feeding summary now
    use_disk_cache=True,             # read/write .pyflic_cache/
    exclusion_group="general",       # a remove_chambers.csv group; None disables
)
```

`load_experiment_yaml` reads the configuration and returns the right subclass for the
experiment type — you do not choose the class yourself.

Note `exclusion_group`: it defaults to applying the `general` group. Pass `None` if you
want the unfiltered design, and be deliberate about which you want, because the two give
different numbers. See [exclusions](config-dfms-chambers.md#excluding-chambers).

## Class hierarchy

```
Experiment (base)
├── SingleWellExperiment          chamber_size=1
└── TwoWellExperiment             chamber_size=2
    ├── HedonicFeedingExperiment
    └── ProgressiveRatioExperiment
```

All subclasses share the same core API; the specialised methods — breakpoint analysis,
weighted durations — live on the subclasses. See
[Experiment types](concepts-experiment-types.md).

## Working with an experiment

```python
exp.dfms                      # {dfm_id: DFM}
dfm = exp.get_dfm(1)
exp.design                    # ExperimentDesign: treatments and factor levels

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

Each of these is also a hub button and a [script action](scripts-actions.md). They take a
loaded experiment.

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

**`bootstrap_metric(exp, metric="PI", n_boot=5000, ci=0.95, seed=42)`** — percentile
confidence intervals, resampling **at the chamber level** because chambers are the
independent biological units. Use it for `PI` and other bounded or skewed metrics, where a
parametric standard error is misleading. `res.summary` has `group`, `n`, `mean`, `sem`,
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

**`compare_configs("a/", "b/", metrics=(...))`** — per-treatment means from two projects
side by side: `Group`, `Metric`, `mean_a`, `mean_b`, `delta`, `pct_change`.

**`lint_flic_config("project/flic_config.yaml")`** — validation results as objects:

```python
for issue in lint_flic_config("project/flic_config.yaml"):
    print(issue.format())
```

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

They are also not installed: they live in the repository, not in the package. Everything
they demonstrate is covered by the hub, the [scripting system](scripts-overview.md), and
the API above. New work should start there.

---

Related: [Summary metrics](concepts-metrics.md) ·
[Performance and caching](performance.md) · [Script actions](scripts-actions.md)
