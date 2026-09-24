# Getting started

pyflic analyses data from **FLIC** (Fly Liquid-food Interaction Counter) experiments,
which measure licking behaviour in fruit flies from electrical signal data. It finds
feeding and tasting bouts in the raw signal, produces quality-control output, computes
summary statistics, and draws publication-ready plots.

## What you need

- **Python 3.13 or newer**, with pyflic installed — see [Installing pyflic](install.md).
- **Raw CSV files** from your FLIC rig, one or more per DFM.
- Nothing else. pyflic does not need a network connection, a database, or a server.

## The three things pyflic works with

These map directly onto the hardware, so if you know your rig you already know the model.

| Term | What it is |
|---|---|
| **DFM** | One physical FLIC device (Data File Module), reading up to 12 wells. |
| **Chamber** | A group of wells within a DFM — one well, or two for a choice assay — holding one fly and assigned to one treatment. |
| **Experiment** | One recording: a set of DFMs governed by one configuration file, in one Experiment Directory. |

A single-well experiment therefore gives you 12 chambers per DFM; a two-well choice
experiment gives you 6, since each chamber uses a pair of wells.

Several experiments that address one question are pooled as the **members** of a
**Project** — see [Projects and members](concepts-project.md).

## The path from here

Four short steps take you from an empty folder to your first plot:

1. **[Set up an experiment directory](getting-started-project.md)** — where your CSVs go.
2. **[Create the configuration file](getting-started-config.md)** — what your chambers mean.
3. **[Run your first analysis](getting-started-first-run.md)** — and find the output.
4. Then read **[How feeding is detected](concepts-licks-events.md)** so you can judge
   whether the numbers are right.

That fourth step matters more than it looks. pyflic will happily produce clean-looking
plots from badly chosen detection parameters, so understanding what a *lick* and an
*event* actually are is what separates a result you can publish from one you can't.

## Getting help while you work

- Cards, panels and settings groups across the pyflic apps carry a **`?`** button that
  opens this window at the topic for that area.
- Detection parameters have their own **`?`** buttons that jump straight to the entry
  for that parameter in [Parameter reference](reference-parameters.md).
- In the Hub, the **`?`** at the right-hand end of the output area — beside **Clear
  Errors** — opens this window at the start, whatever panel is open. **Help** on the Tools
  panel does the same.
- Press **F1** in the Hub, the QC Viewer, the Config Editor, the Script Editor or the Plot
  Editor to open help for the screen you are on.
- Run `pyflic help` from a terminal to open this window without launching an app.
- Search, at the top of this window, looks through every topic; **Guides** in the topic
  list are reading paths through a whole subject.
