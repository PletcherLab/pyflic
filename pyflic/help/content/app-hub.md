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

### Getting a Project open — four buttons, three cases

There are three states a folder can be in before it is a Project, so there are three ways
in, plus the editor for the one already open. They are disjoint: each refuses the other
two's case and names the button that handles it.

- **Open a Project…** — the folder exists *and* already has a `project.yaml`. Choosing the
  one already open re-reads it from disk, so members added or analysed outside the Hub show
  up: the picker is the reload.
- **Create Project…** — nothing exists yet. Name the folder in the dialog's Directory field
  (its browser will make one), fill in the design, and the `project.yaml` is written into
  it.
- **Initialize this folder…** — the folder exists, usually with experiment folders already
  in it, but has no `project.yaml`. It keeps its own name, its subdirectories become the
  members, and the design shown is **inferred from the first one that already has a
  config**. This is the path for a study started before Projects existed. Point it at an
  Experiment Directory and it offers to initialize the *parent* instead, so that experiment
  becomes a member.
- **Project design…** — reopens the editor on the loaded Project's `project.yaml`. Disabled
  until one is open, because it edits the Project in hand. Reads *(none set)* when that
  Project has no `design:` block, which means its members are being validated against each
  other instead of against an authority.

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
- **Initialize existing folder… (n)** — the folder is already in the Project but has no
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
