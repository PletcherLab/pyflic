# pyflic

A Python toolkit for analyzing data from **FLIC (Fly Liquid-food Interaction Counter)** experiments. pyflic detects feeding and tasting bouts from raw electrical signal data, generates quality-control reports, computes summary statistics, and produces publication-ready plots.

pyflic is a complete port of the original R-based FLIC analysis pipeline (FLICFunctions.R) into Python, with a modern GUI, a YAML-based configuration system, and built-in statistical tools.

## Features

- **Signal processing pipeline** -- baseline subtraction (running median), dual-threshold feeding detection, event linking, tasting detection, and preference index computation
- **YAML configuration** -- define experiment structure, parameters, factorial designs, and automated analysis scripts in a single `flic_config.yaml` file
- **Graphical tools** -- config editor, QC viewer with live parameter recompute, and an analysis hub with one-click pipelines
- **Statistical analysis** -- ANOVA / linear mixed models, bootstrap confidence intervals, parameter sensitivity sweeps, light-phase summaries, and bout microstructure analysis
- **Publication outputs** -- per-treatment dot plots, binned time-course plots, PDF experiment reports, and tidy long-format CSV exports

## Quick start

### Install

Requires Python 3.13+. Install from GitHub:

```bash
pip install git+https://github.com/PletcherLab/pyflic.git
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add git+https://github.com/PletcherLab/pyflic.git
```

### Set up a project

Organize your experiment as a directory with `flic_config.yaml` and a `data/` folder containing DFM CSV files:

```
my_experiment/
  flic_config.yaml
  data/
    DFM1_0.csv
    DFM2_0.csv
    ...
```

Create the config file interactively:

```bash
pyflic config
```

Or validate an existing one:

```bash
pyflic lint my_experiment/
```

### Run an analysis

**From the GUI:**

```bash
pyflic hub my_experiment/
```

**From Python / Jupyter:**

```python
from pyflic import load_experiment_yaml

exp = load_experiment_yaml("my_experiment/")
exp.execute_basic_analysis()
```

**Generate a PDF report:**

```bash
pyflic report my_experiment/
```

## CLI commands

| Command | Description |
|---|---|
| `pyflic config` | Launch the config editor GUI |
| `pyflic hub [project]` | Launch the analysis hub GUI |
| `pyflic qc <project>` | Launch the QC viewer |
| `pyflic lint <project>` | Validate `flic_config.yaml` against the schema |
| `pyflic report <project>` | Generate a PDF experiment report |
| `pyflic clear-cache <project>` | Remove cached feeding summaries |
| `pyflic version` | Print the installed version |

## How it works

pyflic processes raw FLIC signal data through a multi-step pipeline:

1. **Baseline subtraction** -- a running median removes slow drift and inter-device variation
2. **Feeding detection** -- a dual-threshold algorithm identifies feeding bouts: candidate licks must exceed `feeding_minimum`, and at least one sample per bout must exceed `feeding_threshold`
3. **Event linking** -- short gaps between events (controlled by `feeding_event_link_gap`) are bridged so brief mid-meal interruptions don't split a single feeding bout into many
4. **Tasting detection** -- contacts that fall between the tasting thresholds and were not already classified as feeding are labelled as tasting events
5. **Summary metrics** -- per-chamber lick counts, event counts, bout durations, inter-bout intervals, preference indices, and intensity measures are computed and aggregated by treatment

See [doc/USAGE.md](doc/USAGE.md) for the full pipeline description and parameter reference.

## Documentation

| Document | Contents |
|---|---|
| [doc/INSTALL.md](doc/INSTALL.md) | Installation options (pip, uv, wheel) and verification |
| [doc/USAGE.md](doc/USAGE.md) | Complete usage guide: signal processing pipeline, YAML config reference, GUI descriptions, Python API, advanced analytics, and workflows |
| [doc/01_GettingStarted.ipynb](doc/01_GettingStarted.ipynb) | Tutorial: loading experiments, understanding the detection pipeline |
| [doc/02_GroupedAnalysis.ipynb](doc/02_GroupedAnalysis.ipynb) | Tutorial: treatment groups and factorial designs |
| [doc/03_ChoiceChamberAnalysis.ipynb](doc/03_ChoiceChamberAnalysis.ipynb) | Tutorial: two-well choice experiments and PI |
| [doc/HedonicFeeding.ipynb](doc/HedonicFeeding.ipynb) | Tutorial: hedonic feeding experiments |
| [doc/ProgressiveRatio.ipynb](doc/ProgressiveRatio.ipynb) | Tutorial: progressive-ratio experiments |

> **Note:** The Jupyter notebooks are included for consistency with the original R-based workflow and will be deprecated in a future release. The Analysis Hub GUI (`pyflic hub`) and the YAML scripting system now cover all of the same functionality with a more streamlined interface. New users should start with the GUI; the notebooks remain available as reference for users transitioning from the R pipeline.

## Experiment types

| Type | Chamber size | Use case |
|---|---|---|
| `single_well` | 1 | 12 independent wells per DFM |
| `two_well` | 2 | Two-well choice assays with preference index |
| `hedonic` | 2 | Hedonic feeding with weighted duration analysis |
| `progressive_ratio` | 2 | Progressive-ratio schedules with breakpoint detection |

## Migrating from R

pyflic reproduces the output of the R FLIC analysis pipeline. Key correspondences:

| R function / concept | pyflic equivalent |
|---|---|
| `ParametersClass.TwoWell()` | `Parameters.two_well()` |
| `SetParameter(p, Feeding.Event.Link.Gap=5)` | `params.with_updates(feeding_event_link_gap=5)` |
| `Feeding.Summary.Monitors(...)` | `exp.feeding_summary()` |
| `BinnedFeeding.Summary.Monitors(...)` | `exp.binned_feeding_summary(binsize_min=30)` |
| `RawDataPlot.DFM(DFM1)` | `dfm.plot_raw()` |
| `CalculateBaseline()` | automatic on `DFM.load()` |
| `Get.Events(z)` | `get_events(z)` |
| `Link.Events(z, thresh)` | `link_events(z, thresh)` |
| `PI.Multiplier` | `pi_direction: "left"` or `"right"` |
| `ExpDesign.csv` | `flic_config.yaml` chambers section |
| Manual R scripts | `scripts:` section in YAML, or GUI buttons |

Parameter names use underscores instead of dots (e.g. `Feeding.Threshold` becomes `feeding_threshold`). The YAML config replaces the combination of R parameter objects and `ExpDesign.csv` files.

## License

See [LICENSE](LICENSE) for details.
