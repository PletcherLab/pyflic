# pyflic

A Python toolkit for analyzing data from **FLIC (Fly Liquid-food Interaction Counter)** experiments. pyflic detects feeding and tasting bouts from raw electrical signal data, generates quality-control reports, computes summary statistics, and produces publication-ready plots.

pyflic is a complete port of the original R-based FLIC analysis pipeline (FLICFunctions.R) into Python, with a modern GUI, a YAML-based configuration system, and built-in statistical tools. Related recordings are grouped into **Projects** — a Project's **members** are different experiments addressing one question, not repeats — which pool their results into combined figures, statistics, and a project report.

## Features

- **Signal processing pipeline** -- baseline subtraction (running median), dual-threshold feeding detection, event linking, tasting detection, and preference index computation
- **Batch / Project / Experiment structure** -- members of one design live in a Project whose `design:` section is the authority for their shared settings; a Batch runs one Project Script across many Projects unattended, finding them recursively so grouping folders are transparent
- **Pooled analysis** -- member summaries stacked into a Combined Analysis, with pooled per-chamber tests beside a mixed model (DFM nested within Experiment)
- **Experiment Types** -- a named assay (Hedonic, Progressive Ratio, Custom, or one a lab builds in the Experiment Type Editor from pyflic's Analysis Modules) selects a chamber layout, states the light it expects, and constrains the facets, quality cutoffs, analyses, and report
- **Facets** -- a time window is a column, not a directory: one `analysis/` per experiment carries every phase
- **YAML configuration** -- experiment structure, parameters, factorial designs, and two levels of automated scripting
- **Graphical tools** -- an Analysis Hub whose QC, Analyze and Plots panels are checkboxes over one button each, a config editor, a QC viewer, a project-level Plot Editor, and the Experiment Type Editor
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

A member may instead read its recording from a folder outside the Project — a run folder
that must not change — named by `data_folder:` at the top of its `flic_config.yaml`,
relative to the member and with forward slashes:

```yaml
data_folder: ../../../../../runs/flic/2026/R-20261020-asmith-01/data
dfms:
  - id: 1
    chambers: {1: Ctrl, 2: Exp}
```

pyflic only reads that folder: the analysis, QC, reports and cache stay in the member, and
two members of one Project never read the same recording. Without the key nothing changes,
and the retired `data_dir:` stays ignored. See
[`data_folder`](pyflic/help/content/config-structure.md#data-folder).

A folder with Projects **anywhere beneath it** is a **Batch** — nothing marks it, being
one is structural. Discovery is recursive and stops at each Project, so
`Sept2026/ProjA` and `Archive/2025/ProjC` are both found from one batch folder, and a
Project is named by its path inside it. **Run Batch** always opens a review window first,
stating what will run and offering to repair the folders it cannot use — a recording still
loose at its root, or one with no config yet.

Create the config file interactively:

```bash
pyflic config my_experiment/
```

An example config, with scripts, is in [examples/flic_config.yaml](examples/flic_config.yaml).

Or validate an existing one:

```bash
pyflic lint my_experiment/
```

### Run an analysis

**From the GUI:**

```bash
pyflic hub my_project/
```

The Hub's QC, Analyze and Plots panels list what a run can produce as checkboxes, over
one button each: **Run QC**, **Run analysis** and **Generate plots**. A tick is
configuration, saved at once as `global.omit:` in the Project's design (or a standalone
experiment's `flic_config.yaml`), so a script's `run_qc`, `run_analysis` and `run_plots`
steps produce the same files as a click (ADR-0018). The Project's **Analyze all** and a
Batch Run run each member's Basic analysis and its report, whatever the ticks.

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
| `pyflic config [dir]` | Launch the config editor GUI (default: the current folder's config) |
| `pyflic experiment [dir]` | Launch the Experiment Editor: one recording's `about:`, design and chambers |
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

The Experiment Type Editor has an entry point of its own, `pyflic-types`, and opens from
the Hub's Tools panel too.

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
| [The firmware 4.0.0 data columns](docs/v4-data-columns.md) | The 63 columns a firmware 4.0.1 recording adds, and where pyflic uses them |

[docs/README.md](docs/README.md) indexes these guides, the developer references, and the
plans removed once they were built.

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
| [Firmware 4.0.0 data in pyflic](docs/firmware-4-data-in-pyflic.md) | What pyflic reads from each column of a DFM data file, by firmware generation, and where it surfaces |

> **Note:** The Jupyter notebooks in `doc/ToBeDepricated/` are retained for users
> transitioning from the R pipeline and will be removed in a future release. The Analysis
> Hub (`pyflic hub`) and the YAML scripting system cover the same ground. New users should
> start with the GUI.

## Experiment types

The Experiment Type (`experiment_type:`) names the assay; the chamber layout
(`chamber_layout:`, `single_well` or `two_well`) is set apart from it (ADR-0007). A type is
a definition, not code (ADR-0019). pyflic ships three, read-only, and a lab builds its own
in the Experiment Type Editor (`pyflic-types`, or the Hub's Tools panel) by combining
Analysis Modules — Progressive Ratio, Hedonic, Paired and yoked, Light and dark — with its
layout, the light it expects, and the Control Comparison its report leads with. The lab's
types live in a library folder set in the editor (`~/.config/pyflic/experiment_types` by
default; a shared drive shares it). A lab type travels with the data as
`experiment_type.yaml` beside `project.yaml` or `flic_config.yaml`; the Project design
editor, the Hub and `pyflic lint` say when that copy differs from the library's.

| Type | Chamber layout | Light it expects | Analysis Modules | Use case |
|---|---|---|---|---|
| `Custom` (the default) | `single_well` or `two_well` | any, or none | Paired and yoked, Light and dark | Any design: feeding summary, preference index on two wells, and for an optogenetic recording the light QC and light behaviour of whatever Paradigms it ran |
| `Hedonic` | `two_well` | none | Hedonic | Hedonic feeding with weighted duration analysis, without light |
| `ProgressiveRatio` | `two_well` | ProgressiveRatio, FixedRatio (OpenLoop beside them) | Progressive Ratio, Light and dark | Progressive or fixed ratio training and test, with breaking point detection, with or without yoked controls |

A recording whose `Program.txt` runs a Paradigm type its Experiment Type does not expect
stops loading and names it. A Progressive Ratio program with a ClosedLoop Training
interval before its ratio Paradigm needs a lab type that expects ClosedLoop too.

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
| Manual R scripts | `scripts:` section in YAML, or the Analysis Hub |

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
(how it renders, and why the anchor code looks the way it does). Every other design
decision is an ADR in [docs/adr/](docs/adr/), and [docs/README.md](docs/README.md) indexes
the developer references, including where to read the plans removed once they were built.
