# pyflic Usage Guide

Source code: [https://github.com/PletcherLab/pyflic](https://github.com/PletcherLab/pyflic)

---

## Table of Contents

1. [What pyflic does](#1-what-pyflic-does)
2. [Project directory layout](#2-project-directory-layout)
3. [Configuration file (`flic_config.yaml`)](#3-configuration-file-flic_configyaml)
4. [Command-line tools](#4-command-line-tools)
5. [Python API](#5-python-api)
6. [Typical workflows](#6-typical-workflows)
7. [Jupyter notebooks](#7-jupyter-notebooks)

---

## 1. What pyflic does

pyflic analyzes data from **FLIC (Fly Liquid-food Interaction Counter)** experiments, which measure licking behavior in fruit flies using electrical signal data. It detects feeding and tasting bouts, generates quality-control reports, and produces publication-ready plots and summary tables.

The main objects in pyflic map to the physical hardware:

- A **DFM** (Data File Module) is one physical FLIC device, reading up to 12 wells.
- A **chamber** is a group of wells (1 or 2) within a DFM assigned to one experimental treatment.
- An **experiment** is a collection of DFMs governed by a shared configuration.

---

## 2. Project directory layout

pyflic expects one directory per experiment with the following structure:

```
project_dir/
  flic_config.yaml        ← required; defines experiment structure and parameters
  data/                   ← DFM CSV files (one per DFM, named by device ID)
  qc/                     ← written by pyflic after running QC (do not create manually)
  analysis/               ← written by pyflic after running analysis (do not create manually)
```

---

## 3. Configuration file (`flic_config.yaml`)

The YAML config is the entry point for every experiment. It defines the experiment type, algorithm parameters, and how chambers are assigned to treatments. The easiest way to create one is with the `pyflic-config` GUI (see [Command-line tools](#4-command-line-tools)).

### Top-level structure

```yaml
global:
  experiment_type: ...
  constants: { ... }
  params: { ... }
  experimental_design_factors: { ... }   # optional
  well_names: { ... }                    # optional

dfms:
  1:
    params: { ... }
    chambers:
      1: TreatmentA
      2: TreatmentB
  2:
    ...
```

---

### `global.experiment_type`

Determines which analysis class is loaded. If omitted, pyflic selects automatically based on `chamber_size`.

| Value | Class | Use when |
|---|---|---|
| `single_well` | `SingleWellExperiment` | 12 independent wells per DFM |
| `two_well` | `TwoWellExperiment` | Two-well choice experiments |
| `hedonic` | `HedonicFeedingExperiment` | Two-well with hedonic/reward-based analysis |
| `progressive_ratio` | `ProgressiveRatioExperiment` | Progressive-ratio schedules |

---

### `global.constants`

Hard limits used during data validation. Chambers that exceed these thresholds are flagged or excluded.

| Key | Default | Description |
|---|---|---|
| `min_transform_licks_cutoff` | `0.00001` | Minimum transformed lick value; chambers below this are excluded |
| `max_med_duration_cutoff` | `13` | Maximum median bout duration (seconds); chambers above are flagged |
| `max_events_cutoff` | `150000` | Maximum number of raw events; chambers above are flagged |

---

### `global.params`

Algorithm parameters used for bout detection. These are global defaults; any DFM-level `params` block overrides them for that device only.

**Timing**

| Key | Default | Description |
|---|---|---|
| `baseline_window_minutes` | `3` | Duration (minutes) of the rolling window used to compute baseline signal |
| `samples_per_second` | `5` | Hardware sampling rate in Hz |

**Feeding detection**

| Key | Default | Description |
|---|---|---|
| `feeding_threshold` | `10` | Signal rate threshold (counts/s) above which a sample is considered feeding |
| `feeding_minimum` | `10` | Minimum duration (seconds) for a detected bout to be counted as feeding |
| `feeding_minevents` | `1` | Minimum number of raw events within a bout window |
| `feeding_event_link_gap` | `5` | Maximum gap (in samples) between events that can be bridged into one bout (~1 s at 5 Hz) |

**Tasting detection**

| Key | Default | Description |
|---|---|---|
| `tasting_minimum` | `0` | Minimum duration (seconds) for a taste bout |
| `tasting_maximum` | `10` | Maximum duration (seconds); contacts longer than this are not counted as tasting |
| `tasting_minevents` | `1` | Minimum raw events per taste bout |

**Hardware / experiment design**

| Key | Required | Description |
|---|---|---|
| `chamber_size` | Yes | Wells per chamber: `1` (single-well) or `2` (two-well choice) |
| `pi_direction` | No | Which side is the preference-index reference: `"left"` or `"right"` |
| `correct_for_dual_feeding` | No | If `true`, corrects simultaneous licks in two-well experiments |

> Many parameter names have accepted aliases (e.g. `baseline_window` is equivalent to `baseline_window_minutes`, `link_gap` to `feeding_event_link_gap`). The config editor handles these automatically.

---

### `global.experimental_design_factors` (optional)

Defines a factorial experimental design. Each key is a factor name; its value is the list of levels. When factors are defined, chamber assignments in `dfms` must be written as comma-separated factor levels in the same order as the factors are listed here.

```yaml
experimental_design_factors:
  paired: [Paired, Unpaired]
  genotype: [Chrim, WCS]
```

With this definition a chamber assigned `"Paired,Chrim"` will be tagged with `paired=Paired` and `genotype=Chrim`.

---

### `global.well_names` (optional)

Human-readable labels for the wells within each chamber, used in plots and reports.

```yaml
# Two-well (chamber_size: 2)
well_names:
  A: Sucrose
  B: Yeast

# Single-well (chamber_size: 1)
well_names:
  1: Water
  2: 10 mM Sucrose
```

---

### `dfms`

One entry per physical DFM device. The key (or `id` field) matches the device number used in the CSV filenames.

```yaml
dfms:
  1:
    params:
      pi_direction: left      # override global param for this DFM only
    chambers:
      1: Paired,WCS           # chamber 1 → Paired genotype WCS
      2: Unpaired,WCS         # chamber 2 → Unpaired genotype WCS
  2:
    params: {}                # no overrides; inherits all global params
    chambers:
      1: Unpaired,Chrim
      2: Paired,Chrim
```

- The `params` block accepts the same keys as `global.params` and takes precedence over global defaults for that DFM.
- The `chambers` block maps chamber index to treatment name (or comma-separated factor levels if `experimental_design_factors` is defined).

---

### Complete example

```yaml
global:
  experiment_type: hedonic
  constants:
    min_transform_licks_cutoff: 0.00001
    max_med_duration_cutoff: 13
    max_events_cutoff: 150000
  params:
    chamber_size: 2
    pi_direction: left
    baseline_window_minutes: 3
    samples_per_second: 5
    feeding_threshold: 10
    feeding_minimum: 10
    feeding_minevents: 1
    feeding_event_link_gap: 5
    tasting_minimum: 0
    tasting_maximum: 10
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
      pi_direction: right
    chambers:
      1: Unpaired,Chrim
      2: Paired,Chrim
```

---

## 4. Command-line tools

Three tools are installed with pyflic.

### `pyflic`

Prints a summary of available commands and the Python API. Takes no arguments.

```bash
pyflic
```

### `pyflic-config`

Opens a graphical configuration editor (PyQt6) for creating and editing `flic_config.yaml`. Automatically loads an existing config if one is present in the current directory.

```bash
pyflic-config
```

Use this before running any analysis to set up experiment structure, chamber assignments, and parameter values without editing YAML by hand.

### `pyflic-qc`

Opens an interactive QC viewer (PyQt6) for a project that has already had QC reports computed. Displays one tab per DFM with:

- **Integrity report** — per-chamber validation results
- **Data breaks** — detected time gaps in the raw signal
- **Simultaneous feeding matrix** — which wells are licked at the same time (two-well only)
- **Bleeding check** — cross-well signal contamination
- **Signal plots** — raw, baselined, and cumulative lick plots

```bash
pyflic-qc /path/to/project_dir
```

> QC reports must be computed first (via `exp.write_qc_reports()` in Python or `execute_basic_analysis()`). The viewer reads pre-computed files from `project_dir/qc/` and does not reload raw data.

---

## 5. Python API

### Loading an experiment

```python
from pyflic import load_experiment_yaml

exp = load_experiment_yaml(
    "/path/to/project_dir",
    range_minutes=(0, 0),   # (start, end) in minutes; (0, 0) means the full recording
    parallel=True,           # load DFMs concurrently
)
```

`load_experiment_yaml` reads `flic_config.yaml` and returns the appropriate subclass (`SingleWellExperiment`, `TwoWellExperiment`, `HedonicFeedingExperiment`, or `ProgressiveRatioExperiment`).

You can also load a specific subclass explicitly:

```python
from pyflic import HedonicFeedingExperiment
exp = HedonicFeedingExperiment.load("/path/to/project_dir")
```

---

### Experiment class hierarchy

```
Experiment (base)
├── SingleWellExperiment        # chamber_size=1; 12 independent wells per DFM
└── TwoWellExperiment           # chamber_size=2; two-well choice
    ├── HedonicFeedingExperiment
    └── ProgressiveRatioExperiment
```

All subclasses share the same core API. Specialized methods (breakpoint analysis, hedonic metrics) are added in the subclasses.

---

### Key methods

**Accessing data**

```python
exp.dfms                        # dict of DFM objects keyed by DFM ID
dfm = exp.get_dfm(1)            # get a single DFM
exp.design                      # ExperimentDesign (structure, treatments, factor levels)
```

**Running QC**

```python
# Compute and write QC reports to project_dir/qc/
exp.write_qc_reports()

# Or just get results as Python dicts without writing files
results = exp.compute_qc_results()
```

**Feeding summary**

```python
# Full-experiment feeding summary (returns a DataFrame)
df = exp.feeding_summary()
# Columns: Chamber, Treatment, DFM, Licks, Events, MeanDuration, MedDuration, ...

# Restrict to a time window
df = exp.feeding_summary(range_minutes=(30, 90))

# Time-binned summary (30-minute bins)
binned = exp.binned_feeding_summary(binsize_min=30)
# Extra columns: Interval, Minutes (bin midpoint)
```

**Plotting**

```python
# Faceted jitter + boxplot for all feeding metrics grouped by treatment
fig = exp.plot_feeding_summary()
exp.write_feeding_summary_plot()           # saves to project_dir/analysis/

# Cumulative licks for one chamber
fig = exp.plot_cumulative_licks_chamber(dfm_id=1, chamber=1)

# Binned time-series metrics
fig = exp.plot_binned_metric_by_treatment(binned, metric="Licks")
fig = exp.plot_binned_licks_by_treatment(binned)
```

**Full pipeline in one call**

```python
results = exp.execute_basic_analysis()
# Runs: write_qc_reports → write_summary → write_feeding_summary → write_feeding_summary_plot
# Returns a dict with paths to all output files
```

**Text summary**

```python
print(exp.summary_text())          # prints to console
exp.write_summary()                # saves to project_dir/analysis/summary.txt
```

---

### DFM objects

Individual DFM objects expose the raw and processed data:

```python
dfm = exp.get_dfm(1)

dfm.raw_df                         # raw CSV as a DataFrame
dfm.baseline_df                    # baseline-subtracted signal
dfm.feeding_summary()              # per-chamber feeding metrics for this DFM

dfm.plot_raw()                     # raw signal plot
dfm.plot_baselined()               # baseline-subtracted signal plot
dfm.plot_cumulative_licks()        # cumulative lick count plot
```

---

## 6. Typical workflows

### Workflow A — Full analysis from scratch

```bash
# 1. Create the config file with the GUI
pyflic-config

# 2. Run analysis in Python (or in a Jupyter notebook)
```

```python
from pyflic import load_experiment_yaml

exp = load_experiment_yaml("/path/to/project_dir")
exp.execute_basic_analysis()
```

```bash
# 3. Inspect QC results interactively
pyflic-qc /path/to/project_dir
```

---

### Workflow B — Custom analysis in a notebook

```python
from pyflic import load_experiment_yaml

exp = load_experiment_yaml("/path/to/project_dir")

# Explore structure
print(exp.summary_text())

# Get feeding metrics
df = exp.feeding_summary()
display(df)

# Filter to a treatment group
sucrose = df[df["Treatment"].str.contains("Sucrose")]

# Custom jitter plot
fig = exp.plot_jitter_summary(sucrose, x_col="Treatment", y_col="Licks")
fig.show()

# Time-binned analysis (60-minute bins, first 4 hours)
binned = exp.binned_feeding_summary(binsize_min=60)
fig = exp.plot_binned_licks_by_treatment(binned)
```

---

### Workflow C — QC only

If you want to inspect data quality before running a full analysis:

```python
exp = load_experiment_yaml("/path/to/project_dir")
exp.write_qc_reports()
```

```bash
pyflic-qc /path/to/project_dir
```

---

## 7. Jupyter notebooks

The `doc/` directory contains a series of tutorial notebooks. Work through them in order for a complete introduction, or jump to the one that matches your experiment type.

### `doc/01_GettingStarted.ipynb`

**Start here.** Covers:
- The Experiment class hierarchy and when to use each subclass
- Loading an experiment from a project directory
- Accessing DFMs and printing experiment summaries
- Basic CLI tool overview

### `doc/02_GroupedAnalysis.ipynb`

Covers analysis grouped by treatment and factor levels:
- Inspecting and filtering the feeding summary DataFrame
- Grouping results by experimental factors
- Customizing and exporting summary tables

### `doc/03_ChoiceChamberAnalysis.ipynb`

For **two-well choice experiments**:
- Well-by-well metrics (preference index, simultaneous feeding)
- Two-well specific QC (bleeding check, simultaneous feeding matrix)
- Plotting utilities for choice behavior

### `doc/HedonicFeeding.ipynb`

For **hedonic feeding experiments**:
- Multi-level factorial designs (e.g., concentration × genotype)
- Hedonic-specific QC and filtering
- Weighted duration summaries and specialized plots
- End-to-end example with a real dataset

### `doc/ProgressiveRatio.ipynb`

For **progressive-ratio experiments**:
- Breakpoint detection and analysis
- Time-series and transition visualization
- Schedule-specific parameter recommendations

---

The `notebooks/` directory contains additional working examples:

- **`notebooks/load_experiment.ipynb`** — detailed walkthrough of loading a project directory, inspecting raw DFM objects, and manually overriding parameters
- **`notebooks/run_flic_config_6h.ipynb`** — complete 6-hour experiment pipeline: config setup → QC → full analysis → advanced plotting
