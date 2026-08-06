# Setting up a project directory

A **project directory** is one folder holding one experiment: its configuration, its raw
data, and everything pyflic writes. You make the folder and the `data/` subfolder
yourself; pyflic creates the rest.

## What you create

```
my_experiment/
  flic_config.yaml        <-- you create this (next step)
  data/                   <-- you create this; put your CSVs here
    DFM1_0.csv
    DFM2_0.csv
```

Only two things are required: a configuration file, and a `data/` folder containing your
raw CSVs. The folder name is yours to choose — pyflic never reads meaning from it.

## What pyflic adds

```
my_experiment/
  flic_config.yaml
  data/
  remove_chambers.csv        <-- optional; chamber exclusions
  flic_config_results/       <-- created on first run
    qc/                      <-- quality-control output
    analysis/                <-- summaries, CSV exports, plots
  .pyflic_cache/             <-- cached feeding summaries; safe to delete
```

Output folders are **named after the configuration file**. `flic_config.yaml` writes into
`flic_config_results/`; a second config called `my_protocol.yaml` writes into
`my_protocol_results/`. That is what lets you keep several analyses of the same raw data
side by side without them overwriting each other.

When you load a restricted time range, the output subfolders are suffixed with it —
`qc_0_360/` and `analysis_0_360/` for minutes 0–360 — so ranged runs do not clobber
whole-experiment runs.

## Naming your data files

Your CSVs must live in `data/` and follow one of these patterns:

- **v3 format:** `DFM{id}_{segment}.csv` — for example `DFM1_0.csv`, `DFM1_1.csv`
- **v2 format:** `DFM_{id}.csv` or `DFM_{id}_{segment}.csv`

The `{id}` is the device number, and it must match the DFM entry in your configuration
file. If your rig wrote `DFM3_0.csv`, your config needs a DFM numbered `3`.

Experiments recorded in several segments are stitched together automatically in filename
order, so `DFM1_0.csv`, `DFM1_1.csv`, `DFM1_2.csv` load as one continuous recording. You
do not need to concatenate them yourself.

> **The `data_dir` key no longer exists.** Older configurations sometimes carry one. Data
> is always read from `<project directory>/data/`, and `pyflic lint` will flag a leftover
> `data_dir` so you can delete it.

## More than one experiment

Give each experiment its own project directory. If you keep them as sibling folders under
a common parent, pyflic can run all of them in one click — see
[Running many projects at once](scripts-batch.md).

---

Next: **[Creating the configuration file](getting-started-config.md)**
