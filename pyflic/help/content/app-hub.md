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

A **tile ribbon** runs across the top. The top strip is the containment hierarchy read
left to right — wide **Batch · Project · Experiment** tiles — then **Tools** and a
**status readout** filling the strip to their right. The Experiment tile opens no panel
of its own: it expands a second, shorter row of five compact subtiles — **QC ·
Analyze · Plots · Scripts · AI** — the tools that act on the loaded member. Below the ribbon is a
full-width output and plots area.

Each tile shows only live status — how many members, which one is loaded, how many
scripts. All the *controls* live in the tile's **anchored panel**, which drops down when
you click the tile. One panel is open at a time; click the tile again, click anywhere in
the background, or press Esc to close it.

**Tiles never move or hide.** A tile that does not apply yet is *dimmed* — Analyze is dim
until a member is loaded — but it stays clickable, because its panel holds the control
that fixes the missing state. Its panel's cards are greyed to match, and they stay
clickable too. The strip is a map, not a menu that rearranges itself. The one exception
is the Experiment tile: it opens no panel, so with nothing loaded a click could not show
the fix — it goes inert as well as dimmed, and its hint names where the fix is
(double-click a member in the Project panel).

**Batch and Project are never dimmed.** Their panels hold "Choose batch folder…" and
"Open Project" — the controls that fix the empty state — so a closed-looking tile there
would be pointing away from the only way forward.

**One thing is open at a time.** Expanding the Experiment group closes a container
panel and vice versa; a click in the background closes the open panel *and* folds the
group; Esc closes only the panel. The subtiles' panels are narrow columns of buttons,
and their status lives in each subtile's tooltip.

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

Three cards, top to bottom, answering three questions: **Create/Load** (which Project?),
**Experiments** (which members does it hold?), and **Analysis** (what to do with them —
shown only while a Project is open).

The Experiments card's members table is the centre of the Hub: one row per member, with its DFM count,
chamber count, and whether it has been analysed and reported. Double-click a row to load
it — once the load finishes, the Experiment group expands to its sub-strip (QC · Analyze ·
Plots · Scripts · AI), because loading is only ever a step toward doing something with it.
No panel is opened: the sub-strip is the menu of what you can now do, and which of those
you want is your call.

**Red rows are blocked members** — folders a run cannot use as they stand, including ones
the Project cannot even see yet because they have no config. Hover for the reason. These
used to be invisible here, which is exactly where the fix lives.

The **Analyzed** column has three values, not two. **re-run needed** means the member's
`remove_chambers.csv` is newer than its saved analysis: those results describe a chamber
population you have since said was wrong.

### Getting a Project open — four buttons, three cases

There are three states a folder can be in before it is a Project, so there are three ways
in, plus the editor for the one already open. They are disjoint: each refuses the other
two's case and names the button that handles it.

- **Open Project** — the folder exists *and* already has a `project.yaml`. Choosing the
  one already open re-reads it from disk, so members added or analysed outside the Hub show
  up: the picker is the reload.
- **Create project…** — nothing exists yet. Name the folder in the dialog's Directory field
  (its browser will make one), fill in the design, and the `project.yaml` is written into
  it.
- **Initialize existing directory…** — the folder exists, usually with experiment folders
  already in it, but has no `project.yaml`. It keeps its own name, its subdirectories become the
  members, and the design shown is **inferred from the first one that already has a
  config**. This is the path for a study started before Projects existed. Point it at an
  Experiment Directory and it offers to initialize the *parent* instead, so that experiment
  becomes a member.
- **Project design…** — reopens the editor on the loaded Project's `project.yaml`. Disabled
  until one is open, because it edits the Project in hand. Reads *(none set)* when that
  Project has no `design:` block, which means its members are being validated against each
  other instead of against an authority.

**Validate YAMLs**, on its own full-width row, is not a fifth way in: it checks the open
Project's `project.yaml` and every member's `flic_config.yaml` and reports to the log.
The loaded-project summary sits below the buttons, in this always-visible card, so a
Project that fails to load is explained even while the sections below stay down.

See [Projects and members](concepts-project.md).
### Adding a member — the same three cases, one level down

All three inherit the design, so all three stay disabled until a Project is open: a member
with nothing to conform to is not a member.

- **Create member…** — the member does not exist yet. Give it a name; the Hub makes the
  folder and its `data/`, and scaffolds a `flic_config.yaml` from the design. It refuses a
  name that already has a config, and refuses one whose folder already exists — that is the
  next button's job. Then it offers the two ways to finish: **Edit config…** opens the
  scaffold, **Copy config from…** replaces it with a config chosen from disk, *checked
  against the design before it is written*. A copy that would not conform is not made at
  all; the mismatches are listed and the scaffold stays.
- **Initialize existing directory… (n)** — the folder is already in the Project but has no
  config. It lists the candidates with what was found in each. Loose files are **filed
  first** — the recording into `data/`, everything else into `extra_files/` — and only then
  is the config scaffolded and the config editor opened, because a recording left at the
  root would make the freshly configured member look empty. An ambiguous or unreadable
  folder is refused rather than guessed at. The count is how many candidates there are.
- **Member configs…** — the bulk view: every folder with its config status, so the missing
  ones can be created and the existing ones opened without hunting through the file system.

**Double-click a blocked row** to fix it in place: a row with no config offers to scaffold
one and open it; an unfiled recording offers to file itself.

- **File unfiled recordings** — the same filing in bulk, for every member that needs it.

### The Analysis card

Appears when a Project is open, in the order the work happens:

- **Analyze all** / **Combine** / **Create report** — the project-level actions.
- **View reports** — opens the Project Report and each member's own report.
- **Apply exclusion sheet…** — see [Excluding chambers in bulk](concepts-exclusions.md).
- **Plot editor…** / **Render figures** — the Project's publication figures.
  `plot_specs.yaml` and `figures/` live at the project root, so their buttons live here
  rather than in the per-member Plots panel. See [Plot Editor](app-plot-editor.md).
- **AI narrative…** — the project-level entry point for the AI-written narrative of the
  Combined Analysis, using the provider picked in the AI panel.
- **Script row** — pick a Project Script and **Run script**, or **Edit scripts…** to open
  the Script Editor on `project.yaml`. Project Scripts live here, with the Project they
  act on: they need no loaded member, and they are available the moment a Project is
  open. See [Scripts](scripts-overview.md).

## Analyze panel

Actions on the **loaded member**: basic analysis, the summary CSVs, the faceted
summary, binned CSVs, **Event statistics** (one row per individual feeding or tasting
event — its start minute, duration, licks and intensity — written by the `tidy_export`
action to `analysis*/tidy_<kind>_events.csv`), and the per-member PDF report. Basic
analysis deliberately skips QC — that is the QC panel's job.

## QC panel

Everything QC for the loaded member, in the order the work happens. **QC reports**
writes the per-DFM bundle into `qc/`: the integrity report, data breaks, the
simultaneous-feeding and bleeding matrices (two-well), and the **Raw Signal**,
**Baselined**, and **Cumulative Licks** signal plots. **Open QC Viewer** opens the
interactive app on the already-loaded member — tables, plots, and per-chamber
exclusions saved to `remove_chambers.csv` (see [QC Viewer](app-qc-viewer.md)).
**View QC plots** opens the saved signal plots as output-area tabs, one per DFM and
kind, reusing tabs on a second look. **Open qc folder** shows the files themselves.

## Plots panel

Quick figures for the loaded member, in three groups — because the buttons are three
kinds of thing and a flat list said they were one.

**Chosen metric** holds the **Metric** dropdown and the only two figures it steers:
**Binned time course** and **Dot plot**. Change the metric and nothing outside this group
changes.

**Standard figures** — **Feeding summary** and **Well A vs B** — draw their own metrics;
the Metric box does not reach them. **Well A vs B** is offered only on a two-well layout.

**Type-specific groups** are titled with the Experiment Type they belong to
(*Progressive Ratio only*, *Hedonic only*) and appear only while a member of that type
is loaded, so the card never offers a button whose only possible answer is "this requires
a different Experiment Type".

The panel does not repeat which member is loaded: the Experiment tile it hangs from
already says so, as does the status strip.

The Plot Editor and the publication figures are Project-level and live on the Project
panel's Analysis card. See [Plot catalogue](plots-catalog.md) and
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
present — and, like the rest of the Experiment group, until a member of an open Project
is loaded; with just a Project open, use **AI narrative…** on the Project panel's
Analysis card instead. See [AI summary](concepts-ai-summary.md).

## Tools panel

The config editor, **Validate every YAML here** (parses every
`project.yaml`, `batch.yaml` and `flic_config.yaml` under the selection and reports what
fails — the cheap way to find a hand-edited config three folders down before an unattended
run finds it), the linter and its migration checks, opening the selected folder, cache
clearing, the theme toggle, and this help.

## Output, plots, and errors

Below the strip: an **Output** tab carrying everything a run prints, an **Errors** tab
collecting the warnings and failures (it badges itself while unread), and one tab per
figure. The buttons in the top-right corner clear each of the three.

During a Batch Run the figure tabs pile up faster than anyone reads them. **Suppress
new plot / output tabs during Batch Runs**, in the Batch panel, stops new ones being
created — every artifact is still written to disk. It governs Batch Runs only: project
and experiment analyses always show their plots.
