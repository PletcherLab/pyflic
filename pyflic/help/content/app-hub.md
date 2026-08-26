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

Each tile shows only live status — how many members, which one is loaded, how many
scripts. All the *controls* live in the tile's **anchored panel**, which drops down when
you click the tile. One panel is open at a time; click the tile again, click anywhere in
the background, or press Esc to close it.

**Tiles never move or hide.** A tile that does not apply yet is *dimmed* — Analyze is dim
until a member is loaded — but it stays clickable, because its panel holds the control
that fixes the missing state. Its panel's cards are greyed to match, and they stay
clickable too. The strip is a map, not a menu that rearranges itself.

**Batch and Project are never dimmed.** Their panels hold "Choose batch folder…" and "Open
a Project…" — the controls that fix the empty state — so a closed-looking tile there would
be pointing away from the only way forward.

## Project-first

The Hub is **Project-first**. The selection names the working container — a Batch or a
Project — and does only that one job.

An experiment is loaded **only** by double-clicking its row in the Project panel's
members table. There is no Load tile: the load options (parallel loading, worker count)
sit in the Project panel beside the table that triggers the load, so there is exactly one
route to a loaded member and one place the Hub can be asked what is loaded.

## Batch panel

Lists every Project found **anywhere** beneath the chosen folder — the scan is recursive
and stops at each Project — with how many of its members a run can use, whether it has a
report, and whether anything in it is blocked. A row is named by its path inside the batch
folder (`Sept2026/ProjA`).

- **Check** the Projects a Batch Run should touch. One with nothing usable starts
  unchecked, because it could only produce a failure.
- **Red rows** hold blocked members; hover for the reasons, right-click to open the review
  window focused on that Project.
- **Double-click** a row to make it the working Project — the Project panel opens on it.
  There is no drill-in state and no up-button; the Batch panel keeps showing the batch it
  came from.
- **Rescan** re-walks the folder, for changes made outside the app.
- **Run Batch** opens a review window stating exactly what will run, then runs it.

See [Running many projects at once](scripts-batch.md).

## Project panel

The members table is the centre of the Hub: one row per member, with its DFM count,
chamber count, and whether it has been analysed and reported. Double-click a row to load
it — the Analyze panel opens on it, because loading is only ever a step toward doing
something with it.

**Red rows are blocked members** — folders a run cannot use as they stand, including ones
the Project cannot even see yet because they have no config. Hover for the reason. These
used to be invisible here, which is exactly where the fix lives.

The **Analyzed** column has three values, not two. **re-run needed** means the member's
`remove_chambers.csv` is newer than its saved analysis: those results describe a chamber
population you have since said was wrong.

- **Open a Project** / **New Project here** — choose an existing Project, or write a
  `project.yaml` into a folder to make it one. Creating one opens the project editor, so a
  new Project states its design from the start rather than acquiring one by accident.
- **Project design…** — the `project.yaml` editor: the Project's name and notes, and the
  **design** every member inherits — experiment type, detection parameters, well names,
  the auto-filter constants and the design factors. Reads *(none set)* when the Project
  has no `design:` block, which means its members are being validated against each other
  instead of against an authority. See [Projects and members](concepts-project.md).
- **File unfiled recordings** — moves DFM CSVs sitting at a member's root into its `data/`.
- **Member configs…** — gives a folder holding DFM CSVs a config scaffolded from an
  existing member. See [Projects and members](concepts-project.md).
- **Analyze all** / **Combine** / **Create report** — the project-level actions.
- **View reports** — opens the Project Report and each member's own report.
- **Apply exclusion sheet…** — see [Excluding chambers in bulk](concepts-exclusions.md).
- **Project Scripts** — pick one and **Run** it, or **Edit…** to open the Script Editor on
  `project.yaml`. Project Scripts live here, with the Project they act on: they need no
  loaded member, and they are available the moment a Project is open. See
  [Scripts](scripts-overview.md).

## Analyze panel

Actions on the **loaded member**: basic analysis, the summary CSVs, the faceted summary,
binned CSVs, tidy events, and the per-member PDF report.

## Plots panel

Quick figures for the loaded member, and the entry point to the Plot Editor for the
Project's publication figures. See [Plot catalogue](plots-catalog.md) and
[Plot Editor](app-plot-editor.md).

## Scripts panel

**Experiment Scripts for the loaded member**, and nothing else — the tile is dimmed and
its controls disabled until a member is loaded, like Analyze and Plots. Scripts named in
the Project's `experiment_scripts:` appear here too, so one central recipe serves every
member without being copied into each.

Project Scripts are *not* here: they are in the Project panel, because they act on the
Project and are runnable with no member loaded. See [Scripts](scripts-overview.md).

## AI panel

An optional AI-written narrative of the Combined Analysis. Dimmed until an API key is
present. See [AI summary](concepts-ai-summary.md).

## Tools panel

The config editor, the QC viewer, **Validate every YAML here** (parses every
`project.yaml`, `batch.yaml` and `flic_config.yaml` under the selection and reports what
fails — the cheap way to find a hand-edited config three folders down before an unattended
run finds it), the linter and its migration checks, opening the selected folder, cache
clearing, the theme toggle, and this help.

## Output, plots, and errors

Below the strip: an **Output** tab carrying everything a run prints, an **Errors** tab
collecting the warnings and failures (it badges itself while unread), and one tab per
figure. The buttons in the top-right corner clear each of the three.

During a long run the figure tabs pile up faster than anyone reads them. **Suppress new
plot / output tabs**, in the Batch panel, stops new ones being created — every artifact is
still written to disk.
