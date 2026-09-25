# pyflic

A Python toolkit for analyzing data from **FLIC (Fly Liquid-food Interaction Counter)** experiments. pyflic detects feeding and tasting bouts from raw electrical signal data, generates quality-control reports, computes summary statistics, and produces publication-ready plots.

pyflic is a complete port of the original R-based FLIC analysis pipeline (FLICFunctions.R) into Python, with a modern GUI, a YAML-based configuration system, and built-in statistical tools. Related recordings are grouped into **Projects** — a Project's **members** are different experiments addressing one question, not repeats — which pool their results into combined figures, statistics, and a project report.

## Features

- **Signal processing pipeline** -- baseline subtraction (running median), dual-threshold feeding detection, event linking, tasting detection, and preference index computation
- **Batch / Project / Experiment structure** -- members of one design live in a Project whose `design:` section is the authority for their shared settings; a Batch runs one Project Script across many Projects unattended, finding them recursively so grouping folders are transparent
- **Pooled analysis** -- member summaries stacked into a Combined Analysis, with pooled per-chamber tests beside a mixed model (DFM nested within Experiment)
- **Experiment Types** -- a named assay (Hedonic, Progressive Ratio, or Custom) selects a chamber layout and constrains the facets, quality cutoffs, analyses, and report
- **Facets** -- a time window is a column, not a directory: one `analysis/` per experiment carries every phase
- **YAML configuration** -- experiment structure, parameters, factorial designs, and two levels of automated scripting
- **Graphical tools** -- a tile-strip analysis hub, config editor, QC viewer, and a project-level Plot Editor
- **Statistical analysis** -- ANOVA / linear mixed models, bootstrap confidence intervals, parameter sensitivity sweeps, light-phase summaries, and bout microstructure analysis
- **Publication outputs** -- journal-ready vector figures (faceted metric plots and binned time courses) rendered from a saved Plot Spec + Plot Style, plus PDF experiment and project reports

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

A single recording is an **Experiment Directory**: a `flic_config.yaml` and a `data/`
folder of DFM CSVs.

```
my_experiment/
  flic_config.yaml
  data/
    DFM1_0.csv
    DFM2_0.csv
```

Members of one design go in a **Project**, whose `project.yaml` holds the shared
settings. A **member** is one of the several *different* experiments that Project brings
to bear on its question — a dose series, a genotype panel, a pilot beside its follow-up —
not a repeat of its neighbours. It normally omits `global:` entirely and inherits:

```
my_project/
  project.yaml          <- design: + Project Scripts
  analysis/             <- the pooled Combined Analysis
  figures/              <- publication figures
  plot_specs.yaml
  my_project_report.pdf
  rep1/
    flic_config.yaml    <- just its dfms:
    data/
    analysis/
  rep2/
    ...
```

A folder with Projects **anywhere beneath it** is a **Batch** — nothing marks it, being
one is structural. Discovery is recursive and stops at each Project, so
`Sept2026/ProjA` and `Archive/2025/ProjC` are both found from one batch folder, and a
Project is named by its path inside it. **Run Batch** always opens a review window first,
stating what will run and offering to repair the folders it cannot use — a recording still
loose at its root, or one with no config yet.

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
pyflic hub my_project/
```

**From Python / Jupyter:**

```python
from pyflic import Project, load_experiment_yaml

# one recording
exp = load_experiment_yaml("my_experiment/")
exp.execute_basic_analysis()

# a whole project
project = Project("my_project/")
project.run_all()
project.build_combined_analysis()
```

**Generate a PDF report** — `report` decides what to write from the marker file, so the
same command serves both levels:

```bash
pyflic report my_experiment/    # experiment report
pyflic report my_project/       # pooled project report
```

**Run a Project Script across every Project in a folder:**

```bash
pyflic batch my_study/
```

## CLI commands

| Command | Description |
|---|---|
| `pyflic` | Launch the analysis hub GUI (same as `pyflic hub`) |
| `pyflic config` | Launch the config editor GUI |
| `pyflic hub [dir]` | Launch the analysis hub GUI |
| `pyflic plots <project>` | Launch the Plot Editor (project level) |
| `pyflic qc <dir>` | Launch the QC viewer |
| `pyflic help [topic]` | Open the help window |
| `pyflic lint <dir>` | Validate configs and report needed migrations |
| `pyflic report <dir>` | Write an experiment or project report |
| `pyflic batch <dir>` | Run the designated Project Script in every Project |
| `pyflic clear-cache <dir>` | Remove cached feeding summaries |
| `pyflic version` | Print the installed version |

Commands taking a directory work out what it is from its marker file — a `project.yaml`
means Project, a `flic_config.yaml` means Experiment Directory — so no command needs a
level flag.

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

| User guides | |
|---|---|
| [Getting Started](docs/GETTING_STARTED.md) | Installing and running pyflic, for someone new to Python |
| [Checking your data](docs/qc-user-guide.md) | Every quality-control check, table and figure, including the optogenetic light QC |
| [Analysing a Progressive Ratio assay](docs/progressive-ratio-user-guide.md) | The Progressive Ratio assay end to end: settings, analyses, figures and statistics |

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
