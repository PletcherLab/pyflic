# Analysis Hub

The main pyflic application, and where most work happens.

```bash
pyflic hub my_project/
```

The argument is optional — run `pyflic hub` and choose a folder from the Project panel.
Whatever you pass, pyflic works out what it is from its marker file: a `project.yaml` means
a Project, a `flic_config.yaml` means an Experiment Directory, and a folder of Projects is
a Batch.

## Layout

A horizontal **tile strip** runs across the top: Batch · Project · Analyze · Plots ·
Scripts · AI · Tools, with a **status readout** filling the strip to their right. Below it
is a full-width output and plots area.

Each tile shows only live status — how many replicates, which one is loaded, how many
scripts. All the *controls* live in the tile's **anchored panel**, which drops down when
you click the tile. One panel is open at a time; click the tile again, click anywhere in
the background, or press Esc to close it.

**Tiles never move or hide.** A tile that does not apply yet is *dimmed* — Analyze is dim
until a replicate is loaded — but it stays clickable, because its panel holds the control
that fixes the missing state. The strip is a map, not a menu that rearranges itself.

## Project-first

The Hub is **Project-first**. The selection names the working container — a Batch or a
Project — and does only that one job.

An experiment is loaded **only** by double-clicking its row in the Project panel's
replicates table. There is no Load tile: the load options (parallel loading, worker count)
sit in the Project panel beside the table that triggers the load, so there is exactly one
route to a loaded replicate and one place the Hub can be asked what is loaded.

## Batch panel

Lists the Projects directly beneath the chosen folder, with each one's replicate count and
whether it has been analysed and reported. Pick a Project Script and press **Run Batch** to
run it in every Project, continue-on-error.

Double-clicking a row is an ordinary selection change down to that Project — there is no
drill-in state and no up-button. See [Running many projects at once](scripts-batch.md).

## Project panel

The replicates table is the centre of the Hub: one row per replicate, with its DFM count,
chamber count, and whether it has been analysed and reported. Double-click a row to load
it.

- **Open a Project** / **New Project here** — choose an existing Project, or write a
  `project.yaml` into a folder to make it one.
- **Scaffold pending replicates** — lights up when a subfolder holds DFM CSVs but has no
  `flic_config.yaml`. See [Projects and replicates](concepts-project.md).
- **Analyze all** / **Combine** / **Create report** — the project-level actions.

## Analyze panel

Actions on the **loaded replicate**: basic analysis, the summary CSVs, the faceted summary,
binned CSVs, tidy events, and the per-replicate PDF report.

## Plots panel

Quick figures for the loaded replicate, and the entry point to the Plot Editor for the
Project's publication figures. See [Plot catalogue](plots-catalog.md) and
[Plot Editor](app-plot-editor.md).

## Scripts panel

Both script levels, kept visibly apart: Project Scripts from `project.yaml` above,
Experiment Scripts for the loaded replicate below. See [Scripts](scripts-overview.md).

## AI panel

An optional AI-written narrative of the Combined Analysis. Dimmed until an API key is
present. See [AI summary](concepts-ai-summary.md).

## Tools panel

The config editor, the QC viewer, the linter and its migration checks, cache clearing, the
theme toggle, and this help.
