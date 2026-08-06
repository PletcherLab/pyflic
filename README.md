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
| `pyflic` | Launch the analysis hub GUI (same as `pyflic hub`) |
| `pyflic config` | Launch the config editor GUI |
| `pyflic hub [project]` | Launch the analysis hub GUI |
| `pyflic qc <project>` | Launch the QC viewer |
| `pyflic help [topic]` | Open the help window |
| `pyflic lint <project>` | Validate `flic_config.yaml` against the schema |
| `pyflic report <project>` | Generate a PDF experiment report |
| `pyflic clear-cache <project>` | Remove cached feeding summaries |
| `pyflic version` | Print the installed version |

`pyflic --help` prints this list as text.

## How it works

pyflic processes raw FLIC signal data through a multi-step pipeline:

1. **Baseline subtraction** -- a running median removes slow drift and inter-device variation
2. **Feeding detection** -- a dual-threshold algorithm identifies feeding bouts: candidate licks must exceed `feeding_minimum`, and at least one sample per bout must exceed `feeding_threshold`
3. **Event linking** -- short gaps between events (controlled by `feeding_event_link_gap`) are bridged so brief mid-meal interruptions don't split a single feeding bout into many
4. **Tasting detection** -- contacts that fall between the tasting thresholds and were not already classified as feeding are labelled as tasting events
5. **Summary metrics** -- per-chamber lick counts, event counts, bout durations, inter-bout intervals, preference indices, and intensity measures are computed and aggregated by treatment

See [How feeding is detected](pyflic/help/content/concepts-licks-events.md) for the full
pipeline description, and [Parameter reference](pyflic/help/content/reference-parameters.md)
for every parameter and its effect.

## Documentation

**Documentation ships inside the application.** Press **F1** anywhere, click the **`?`**
button beside any card, tab or parameter, or open the help window on its own:

```bash
pyflic help                          # contents
pyflic help concepts-licks-events    # a specific topic
```

Every parameter control has its own `?` that opens the reference at that parameter, so
"what does this number do?" is one click from the box you are typing into.

The same topics are readable here on GitHub — they are ordinary markdown in
[pyflic/help/content/](pyflic/help/content/):

| Start here | |
|---|---|
| [Getting started](pyflic/help/content/getting-started.md) | What pyflic is and what you need |
| [Setting up a project directory](pyflic/help/content/getting-started-project.md) | Where your CSVs go |
| [Creating the configuration file](pyflic/help/content/getting-started-config.md) | Defining your experiment |
| [Running your first analysis](pyflic/help/content/getting-started-first-run.md) | From load to first plot |

| Core concepts | |
|---|---|
| [The raw signal and the baseline](pyflic/help/content/concepts-signal.md) | Baseline subtraction |
| [How feeding is detected](pyflic/help/content/concepts-licks-events.md) | Licks, events, and the link gap |
| [Summary metrics](pyflic/help/content/concepts-metrics.md) | What every output column means |
| [Two-well choice and the preference index](pyflic/help/content/concepts-two-well-pi.md) | PI, counterbalancing, crosstalk |
| [Parameter reference](pyflic/help/content/reference-parameters.md) | Every parameter, default, and effect |

| Reference | |
|---|---|
| [Configuration file structure](pyflic/help/content/config-structure.md) | The YAML format |
| [What a script is](pyflic/help/content/scripts-overview.md) · [Script actions](pyflic/help/content/scripts-actions.md) | Automated pipelines |
| [Running many projects at once](pyflic/help/content/scripts-batch.md) | Batch modes |
| [Python API](pyflic/help/content/python-api.md) | Driving pyflic from code |
| [Plot catalogue](pyflic/help/content/plots-catalog.md) | What each figure shows |
| [Troubleshooting](pyflic/help/content/troubleshooting.md) | When results look wrong |
| [Installing pyflic](pyflic/help/content/install.md) | Install, update, verify |

> **Note:** The Jupyter notebooks in `doc/ToBeDepricated/` are retained for users
> transitioning from the R pipeline and will be removed in a future release. The Analysis
> Hub (`pyflic hub`) and the YAML scripting system cover the same ground. New users should
> start with the GUI.

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

## For developers

Clone and install in editable mode with the dev dependency group:

```bash
git clone https://github.com/PletcherLab/pyflic.git
cd pyflic
uv sync
```

Run the tests:

```bash
uv run pytest tests/
```

The dev group includes `pytest` and `hypothesis`, used for property-based tests of the
event-detection algorithms.

### Editing the documentation

Help topics are markdown files in [pyflic/help/content/](pyflic/help/content/) — one topic
per file, one `# Title` heading each. They are the only copy of the user documentation and
ship inside the package.

- Add a topic by creating the file and listing it in
  [pyflic/help/toc.py](pyflic/help/toc.py).
- Link between topics as `[text](other-topic.md#heading)`, which resolves both in the app
  and on GitHub.
- `uv run pytest tests/test_help_refs.py` validates that every topic is in the table of
  contents, every cross-link and anchor resolves, and every parameter control's `?` points
  at a real heading. It fails the build on a dangling reference.

Design rationale is recorded in
[docs/adr/0003-help-topics-single-source.md](docs/adr/0003-help-topics-single-source.md)
(what help is) and
[docs/adr/0004-help-rendering-and-reference-validation.md](docs/adr/0004-help-rendering-and-reference-validation.md)
(how it renders, and why the anchor code looks the way it does).

## License

See [LICENSE](LICENSE) for details.
