# Setting up an experiment directory

An **Experiment Directory** is one folder holding one FLIC recording: its configuration,
its raw data, and everything pyflic writes about it. You make the folder and its `data/`
subfolder yourself; pyflic creates the rest.

## What you create

```
my_experiment/
  flic_config.yaml        <-- you create this (next step)
  data/                   <-- you create this; put your CSVs here
    DFM1_0.csv
    DFM2_0.csv
```

Only two things are required: a configuration file named `flic_config.yaml`, and a `data/`
folder containing your raw CSVs. The folder name is yours to choose — pyflic never reads
meaning from it.

## What pyflic adds

```
my_experiment/
  flic_config.yaml
  data/
  remove_chambers.csv     <-- optional; chamber exclusions you declare
  qc/                     <-- quality-control output
  analysis/               <-- summaries, CSV exports, plots, the experiment report
  .pyflic_cache/          <-- cached feeding summaries; safe to delete
```

Results always go to `analysis/` and `qc/`, whatever time range you analysed. Time windows
within a recording are [Facets](concepts-facets.md) — a column in the output, not a separate
folder — so re-running never scatters results across differently named directories.

An Experiment Directory holds exactly **one** configuration. To analyse the same recording
two ways, make a second Experiment Directory with its own copy of `data/`.

## Naming your data files

Your CSVs must live in `data/` and follow one of these patterns:

- **v3 format:** `DFM{id}_{segment}.csv` — for example `DFM1_0.csv`, `DFM1_1.csv`
- **v2 format:** `DFM_{id}.csv` or `DFM_{id}_{segment}.csv`

The `{id}` is the device number, and it must match the DFM entry in your configuration
file. If your rig wrote `DFM3_0.csv`, your config needs a DFM numbered `3`.

Experiments recorded in several segments are stitched together automatically in filename
order, so `DFM1_0.csv`, `DFM1_1.csv`, `DFM1_2.csv` load as one continuous recording. You
do not need to concatenate them yourself.

CSVs left at the folder's root rather than in `data/` are not read. Inside a Project the Hub
shows such a folder as a **blocked member** and offers to file them for you.

> **The `data_dir` key no longer exists.** Older configurations sometimes carry one. Data
> is always read from `<experiment directory>/data/`, and `pyflic lint` will flag a leftover
> `data_dir` so you can delete it.

## More than one experiment

When several recordings address one question — a dose series, a genotype panel, a pilot
and its follow-up — put their Experiment Directories side by side inside a **Project**: a
folder with a `project.yaml` whose design every member shares, and where results are pooled.
The Hub's **Create project…** and **Initialize existing directory…** set one up. See
[Projects and members](concepts-project.md), and
[Running many projects at once](scripts-batch.md) for a folder of Projects.

---

Next: **[Creating the configuration file](getting-started-config.md)**
