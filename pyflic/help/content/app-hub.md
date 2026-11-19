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

**The readout's Loaded line** names the loaded member, and the
[Slice](#the-slice-picker) it is loaded as when it is one, and adds a note when its
recording needs one:

| Note | Means |
|---|---|
| *DFM n: firmware 4.0.0 recording without Program.txt: Paradigm types and settings unknown* | copy the `Program.txt` PyMCU exported for this recording into `data/` |
| *no training information: recorded with DFM firmware older than 4.0.0* | an older recording (generation 2): PyMCU 4 recorded a DFM with older firmware, so there is no Training, and a Progressive Ratio member has one `Recording` Facet and no Test |

See [Generations](concepts-optogenetics.md#generations).

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

**Help is on every panel.** Each panel's cards carry a `?` beside their title, and each group
box of the QC, Analyze and Plots panels (*QC reports*, *Optogenetics*, *Progressive Ratio*,
*Chosen metric*, …) carries its own, opening this window at the topic for that area. A
control that needs more than its card says has a `?` beside it too: the Batch panel's
Project Script menu and AI narrative, the Experiments card's member buttons, repair row
and load options, the Analysis card's script row, and the AI panel's provider. The `?` at the right-hand end of the output area, beside **Clear Errors**, opens
help at the start whatever is open, and **F1** opens this topic. The Hub's dialogs —
*Project design*, *Member configs* and the Batch Run review — open their own topics with
**F1**.

**One thing is open at a time.** Expanding the Experiment group closes a container
panel and vice versa; a click in the background closes the open panel *and* folds the
group; Esc closes only the panel. The subtiles' panels are narrow columns — QC, Analyze and
Plots of checkboxes over one button — and their status lives in each subtile's tooltip.

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
- **Red rows** hold blocked members; hover for the reasons. Right-click a row for **Fix
  blocked members in …**, which opens the review window focused on that Project, or
  **Open … as the Project…**.
- **Double-click** a row to make it the working Project — the Project panel opens on it.
  There is no drill-in state and no up-button; the Batch panel keeps showing the batch it
  came from.
- **Rescan** re-walks the folder, for changes made outside the app.
- **Apply exclusion sheet…** writes the batch folder's `remove_chambers.csv` into the
  members of the checked Projects now; a Batch Run applies it anyway. See
  [Exclusion sheets](scripts-batch.md#exclusion-sheets).
- **Project Script** chooses what a Batch Run executes in each Project, and remembers the
  choice in `batch.yaml`; see [Which script runs](scripts-batch.md#which-script-runs).
- **Run Batch** opens a review window stating exactly what will run, then runs it.
- **AI narrative of the batch** asks an AI provider, after the run, to pool the Projects'
  own narratives; see [AI narrative of the batch](scripts-batch.md#ai-narrative-of-the-batch).
  **Suppress new plot / output tabs during Batch Runs** is described under
  [Output, plots, and errors](#output-plots-and-errors).

See [Running many projects at once](scripts-batch.md).

## Project panel

Three cards, top to bottom, answering three questions: **Create/Load** (which Project?),
**Experiments** (which members does it hold?), and **Analysis** (what to do with them —
shown only while a Project is open).

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

When the open Project uses a lab type whose copy beside `project.yaml` differs from the
library's, the card says so, with **Update the type copy from the library**; the members'
results then need a re-run. See
[Where types live](concepts-experiment-types.md#where-types-live).

See [Projects and members](concepts-project.md).

### The members table

The Experiments card's members table is the centre of the Hub: one row per member, with
its DFM count, chamber count, and whether it has been analysed and reported. Double-click a
row to load it — once the load finishes, the Experiment group expands to its sub-strip
(QC · Analyze · Plots · Scripts · AI), because loading is only ever a step toward doing
something with it. No panel is opened: the sub-strip is the menu of what you can now do,
and which of those you want is your call.

**Red rows are blocked members** — folders a run cannot use as they stand, including ones
the Project cannot even see yet because they have no config. Hover for the reason. These
used to be invisible here, which is exactly where the fix lives. See
[Blocked members](concepts-project.md#blocked-members).

The **Analyzed** column has three values, not two. **re-run needed** means the member's
saved results no longer match what its files ask for: its `remove_chambers.csv` is newer
than its saved analysis, so those results describe a chamber population you have since said
was wrong, or a setting in the project design or the member's config has changed since the
analysis ran (a detection parameter, a constant, a chamber's treatment). Hover the cell to
see each change, for example *params.feeding_threshold changed: 20 → 25*. Saving the design
or a config updates the column at once; a save that changes no setting, such as ticking a
box in the Selection or editing a script, leaves it alone. With a Slice picked, it also
means the Slice's definition changed since its run, and a fourth value, **not covered**,
means the member's recording stops short of the Slice (see
[The Slice picker](#the-slice-picker)).

**Load options**, at the foot of the card, govern every load: **Load DFMs in parallel**
(on by default) and the number of **Workers** (*auto* by default). See
[Parallel loading](performance.md#parallel-loading).

### The Slice picker

When the design declares [Slices](concepts-slices.md), a **Slice** menu sits above the
members table: **Whole recording**, then each Slice with its minutes. It chooses which one
the Hub shows, one at a time:

- the table's **Analyzed** and **Report** columns. **not covered** means the member is left
  out of the Slice; hover for the reason. See [Coverage](concepts-slices.md#coverage).
- what a double-click loads. A member that does not cover the Slice is not loaded, and the
  Hub says why.
- the loaded member. Picking another Slice reloads it as that Slice, and the status strip
  names the Slice it is loaded as.
- so what the QC Viewer opens on, what **View QC plots** and **Open qc folder** show, which
  figures **Generate plots** opens in tabs (a Slice's tabs end with `· Slice <name>`), and which
  reports **View reports** opens.

It does not choose what runs: **Run QC**, **Run analysis** and **Generate plots** run the
whole recording and every ticked Slice (see [Slices](#slices)). With no Slices declared, the
menu is hidden.

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
- **Member configs… (n missing)** — the bulk view: every folder with its config status, so
  the missing ones can be created and the existing ones opened without hunting through the
  file system. See [The Member configs dialog](concepts-project.md#the-member-configs-dialog).

**Double-click a blocked row** to fix it in place: a row with no config offers to scaffold
one and open it; an unfiled recording offers to file itself.

- **File unfiled recordings** — the same filing in bulk, for every member that needs it.

### The Analysis card

Appears when a Project is open, in the order the work happens:

- **Analyze all** runs each member's QC reports, Basic analysis and experiment report, as
  before, **whatever the QC, Analyze and Plots panels' ticks**: it does not read the
  Selection, and neither does a Batch Run's `run_all_analyses` step. It removes no earlier
  output. To run a member's ticked items, use its own panels, or a script's `run_qc`,
  `run_analysis` and `run_plots` steps. With [Slices](concepts-slices.md) it runs the whole
  recording and then every Slice the design declares.
- **Combine** / **Create report** — pool the members' summaries, and write the Project
  Report: the whole recording's, and each Slice's from the members that cover it.
- **View reports** — opens the Project Report and each member's own report
  (`analysis/experiment_report.pdf`), those of the Slice the picker shows. See
  [Reports](reports.md).
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

## The ticks are configuration

The QC, Analyze and Plots panels each list a set as checkboxes, in group boxes, over one
button: the **QC Set** over **Run QC**, the **Analysis Set** over **Run analysis**, the
**Plot Set** over **Generate plots**. A ticked box is included in every run. Every box has a
tooltip saying what the item writes.

**A tick is written at once**, and the log says where, in one line:

- for a standalone experiment, into its `flic_config.yaml`, under `global.omit`;
- for a member of a Project, into the Project's `project.yaml`, under `design.global.omit`.
  A Project's members are analysed alike, so **a tick on one member applies to every
  member**, and the log line says so. A Project with no `design: global:` section has
  nowhere to keep it: the Hub says so and puts the box back. Write the design first
  (**Project design…**).

What is stored is what is left *out*, so a config nobody touched runs everything, and an
item a later pyflic adds is included until you untick it. A script's `run_qc`,
`run_analysis` and `run_plots` steps read the same lists, so a click and a script write the
same files. See [`global.omit`](config-structure.md#globalomit).

**A missing box** is one the member cannot have: two-well items on a single-well layout,
optogenetic items on a run without light, a type's items under another type. A group with
none left is not shown. One quiet line under the groups names what is not offered; hover it
for the reasons.

**A greyed box** is one the design allows but the data cannot answer: the Program timeline
for an older recording with no `Program.txt`, the paired and yoked comparison with no
linkage group lighting a Yoked chamber, the Paired-Yoked Difference with every chamber group
solo. Its tooltip gives the reason, and a run writes nothing for it.

**A run states what it left out.** The log, and for Run analysis `summary.txt`, list each
item unticked and, separately, each that does not apply. **No run removes a file**: an item
left out keeps whatever an earlier run wrote.

The panels follow the member: they are rebuilt when a member loads, unloads or reloads. The
Project panel's **Analyze all** and a Batch Run do not read the ticks (see
[The Analysis card](#the-analysis-card)).

## Slices

When the design declares [Slices](concepts-slices.md), the QC, Analyze and Plots panels each
end with a **Slices** group: a box per Slice, all ticked to begin with. **Run QC**, **Run
analysis** and **Generate plots** run the whole recording, then every ticked Slice, each into
its own `slices/<name>/` folder. The whole recording always runs, so it has no box.

The three groups are one setting: a tick on one panel ticks the other two. Like the other
ticks it is written at once, to `global.omit.slices`, and a script's `run_qc`, `run_analysis`
and `run_plots` steps read it too. A member whose recording does not cover a Slice is left
out of that Slice and the run goes on; the log and its `slice.json` say why.

Each Slice has its own exclusion group; see [Exclusions](concepts-slices.md#exclusions).
**Analyze all**, **Combine** and **Create report** do not read these boxes: they cover every
Slice the design declares.

## QC panel

Two cards. **QC** lists the QC Set for the loaded member over **Run QC**, which writes each
ticked item into the member's `qc/` folder (the Progressive Ratio light QC into
`analysis/`). The QC is computed once per run and written item by item. The ticks are
saved, as on Analyze and Plots: see [The ticks are configuration](#the-ticks-are-configuration).

| Group | Item | Writes | Offered for |
|---|---|---|---|
| QC reports | Data integrity | `qc/integrity/DFM<n>_integrity_report.csv` and `.txt`: rows, time span, lost and repeated records, the Paradigm sequence and restarts | every member |
| | Data breaks | `qc/data_breaks/DFM<n>_data_breaks.csv`, for a DFM that has a gap | every member |
| | Raw signal plots | `qc/raw_signal/DFM<n>_raw.png` | every member |
| | Baselined signal plots | `qc/baselined/DFM<n>_baselined.png`, with the thresholds drawn | every member |
| | Cumulative licks plots | `qc/cumulative_licks/DFM<n>_cumulative_licks.png` | every member |
| | Simultaneous feeding | `qc/simultaneous_feeding/DFM<n>_simultaneous_feeding_matrix.csv` | two-well |
| | Bleeding between wells | `qc/bleeding/DFM<n>_bleeding_matrix.csv` and `DFM<n>_bleeding_alldata.csv` | two-well |
| Optogenetics | Opto light QC tables | `qc/opto/`: the verdicts, the intervals, every light event and the program as read; the log names every linkage group that failed or warned | optogenetic |
| | Light explained by licks | `qc/opto/opto_light_dfm<n>.png`, the figure behind the verdict | optogenetic |
| | Program timeline | `qc/program/program_timeline_dfm<n>.png`: the Program Intervals by Paradigm, each well's Acclimation, Activations and light | optogenetic; greyed for an older recording with no `Program.txt` section |
| Progressive Ratio | Light QC table | `analysis/pr_light_qc.csv` and `pr_light_events.csv`: did the paired fly earn its light? The log names every chamber group that failed or warned | a type with the Progressive Ratio module |
| | Licks per light event | `analysis/pr_light_events_dfm<n>.png` | Progressive Ratio module |
| | Sucrose Well resting level | `analysis/pr_resting_level_dfm<n>.png` | Progressive Ratio module |

The light figures and the Program timeline are drawn in 10-minute bins, whatever the
Analyze panel's Bin size. See [Optogenetic experiments](concepts-optogenetics.md),
[Program timeline](plots-catalog.md#program-timeline) and
[Light QC](concepts-progressive-ratio.md#light-qc).

**View** holds the three buttons that run nothing:

- **Open QC Viewer** opens the interactive app on the already-loaded member — tables,
  plots, and per-chamber exclusions saved to `remove_chambers.csv` (see
  [QC Viewer](app-qc-viewer.md)).
- **View QC plots** opens the saved signal plots, the light QC's figure and the Program
  timelines as output-area tabs, one per DFM and kind, reusing tabs on a second look.
- **Open qc folder** shows the files themselves.

## Analyze panel

At the top, the **Bin size (min)** box. The binned summary, the experiment report's time
courses and the Plots panel's Binned time course use it, and nothing else: the light figures
and the Program timeline keep 10-minute bins, and the Progressive Ratio curves 1-minute
bins. See [Progressive Ratio experiments](concepts-progressive-ratio.md#outputs).

Then the Analysis Set, over **Run analysis**. **The core always runs** and has no checkbox,
because every other item reads it: the design's auto-removal and `removed_chambers.csv`,
`summary.txt` (with the QC summary), `feeding_summary.csv` and, with Facets,
`feeding_summary_facet.csv`. Run analysis writes the core, then each ticked item, the report
last.

| Group | Item | Writes into `analysis/` | Offered for |
|---|---|---|---|
| Analyses | Binned summary | `binned_feeding_summary.csv`, in bins of the Bin size | every member |
| | Event statistics | `tidy_feeding_events.csv`: one row per feeding event — its start minute, duration, licks and intensity | every member |
| Optogenetics | Light tables | `light_summary.csv` per chamber and `light_wells.csv` per well — lit time, Activations, Acclimation, Firmware Events and the Paradigm type's own measures — and, with Facets, their `_facet` variants | optogenetic |
| | Paired and yoked comparison | `yoked_comparison.csv` | optogenetic, with a type that has the Paired and yoked module (Custom does); greyed when no linkage group lights a Yoked chamber |
| Progressive Ratio | Paired-Yoked Difference | `paired_yoked_diff.csv` | a type with the Progressive Ratio module; greyed when every chamber group is solo |
| | Cumulative difference table | `pr_cumulative_diff.csv`, in 1-minute bins | Progressive Ratio module |
| | Breaking point | `pr_breaking_point.csv`, and each group's breaking point with its sensitivity to the gap in the log | Progressive Ratio module |
| Hedonic | Weighted duration summary | `treatment_means.csv` | a type with the Hedonic module |
| Report | Experiment report (PDF) | `experiment_report.pdf` ([Reports](reports.md)) | every member |

The Analyze panel's old buttons, one per output, are gone; their script actions remain (see
[Script actions](scripts-actions.md)), and *Basic analysis* is still the `basic_analysis`
action and what **Analyze all** runs.

## Plots panel

Quick figures for the loaded member, over **Generate plots**, which saves each ticked figure
into `analysis/` and opens it in a tab, replacing the tab of the same title. With
[Slices](#slices) it saves each ticked Slice's figures too, into `slices/<name>/analysis/`,
and opens the tabs of the Slice the picker shows.

| Group | Item | Saves into `analysis/` | Offered for |
|---|---|---|---|
| Figures | Feeding summary | `feeding_summary.png` | every member |
| | Well A vs B | `well_comparison_MedDuration.png`: median bout duration at well A beside well B | two-well |
| Chosen metric | Binned time course | `binned_<metric>.png`, in bins of the Bin size | every member |
| | Dot plot | `dot_<metric>.png`, one point per chamber (per Chamber Group under Progressive Ratio) | every member |
| Progressive Ratio | Cumulative difference curve | `pr_cumulative_diff.png` | a type with the Progressive Ratio module |
| | Training-aligned traces | `pr_cumulative_licks_dfm<n>.png` | Progressive Ratio module |
| | Still-responding curve | `pr_still_responding.png` | Progressive Ratio module |
| | Breaking-point plots | `breaking_point_dfm<n>.png` | Progressive Ratio module |
| Hedonic | Hedonic feeding plot | `hedonic_feeding_plot.png` | a type with the Hedonic module |

The **Metric** menu sits inside the *Chosen metric* group because it steers those two
figures and no other. The Progressive Ratio curves keep 1-minute bins.

The panel does not repeat which member is loaded: the Experiment tile it hangs from
already says so, as does the status strip.

The Plot Editor and the publication figures are Project-level and live on the Project
panel's Analysis card. See [Plot catalogue](plots-catalog.md) and
[Plot Editor](app-plot-editor.md).

## Scripts panel

**Experiment Scripts for the loaded member**, and nothing else — the tile is dimmed and
its controls disabled until a member is loaded, like Analyze and Plots. Pick a script in
the menu and **Run** it; **Open Script Editor** edits the member's `flic_config.yaml`
scripts. Scripts named in the Project's `experiment_scripts:` appear here too, so one
central recipe serves every member without being copied into each.

Project Scripts are *not* here: they are in the Project panel, because they act on the
Project and are runnable with no member loaded. See [Scripts](scripts-overview.md).

## AI panel

An optional AI-written narrative of the Combined Analysis: pick a **Provider** and a
**Model**, then **Generate narrative**. Dimmed until an API key is present — and, like the
rest of the Experiment group, until a member of an open Project is loaded; with just a
Project open, use **AI narrative…** on the Project panel's Analysis card instead. See
[AI summary](concepts-ai-summary.md) and, for the keys,
[Providers and keys](concepts-ai-summary.md#providers-and-keys).

## Tools panel

- **Config editor** — the [Config Editor](app-config-editor.md), on the selected folder's
  `flic_config.yaml`.
- **Experiment type editor** — the [Experiment Type Editor](app-type-editor.md), for your
  lab's own types.
- **Validate every YAML here** — parses every `project.yaml`, `batch.yaml` and
  `flic_config.yaml` under the selection and reports what fails: the cheap way to find a
  hand-edited config three folders down before an unattended run finds it.
- **Lint / migration check** — the linter and its migration checks, as `pyflic lint`.
- **Open this folder** — the selected folder in the system file browser.
- **Clear cache** — removes the loaded member's cached feeding summaries or, with only a
  Project open, every member's, and the log says how many it removed from each; see
  [The disk cache](performance.md#the-disk-cache).
- **Toggle theme** — light or dark, remembered for next time.
- **Help** — this help window, at its first page.

## Output, plots, and errors

Below the strip: an **Output** tab carrying everything a run prints, an **Errors** tab
collecting the warnings and failures (it badges itself while unread), and one tab per
figure. The buttons in the top-right corner clear each of the three — **Clear Tabs**,
**Clear Output**, **Clear Errors** — and the `?` after them opens this help window at its
first page, for when you want help without opening a panel first.

During a Batch Run the figure tabs pile up faster than anyone reads them. **Suppress
new plot / output tabs during Batch Runs**, in the Batch panel, stops new ones being
created — every artifact is still written to disk. It governs Batch Runs only: project
and experiment analyses always show their plots.
