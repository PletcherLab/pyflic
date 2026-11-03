# QC Viewer

```bash
pyflic qc my_experiment/
```

The QC viewer is where you decide whether to believe your results. It shows the signal
pyflic worked from and the events it found, so you can check that detection agrees with
what you would have called by eye.

Do this before running statistics, not after. Detection parameters that are wrong for your
data produce output that looks entirely plausible.

Opened from the Hub's QC panel (**Open QC Viewer**, on its *View* card), the viewer starts
directly on the member the Hub already loaded — no second load. The **Run QC** button in the
viewer's top bar computes the whole QC bundle for that experiment (integrity, data breaks,
bleeding, and the raw / baselined / cumulative-licks plots), whatever the Hub's QC panel has
ticked — the Hub's own **Run QC** writes only the ticked items — and refreshes the DFM tabs
in place, so the plots appear without a round-trip back to the Hub; progress streams to the
Load tab's log, and unsaved exclusion checkboxes survive the refresh. Loading from the Load
tab runs the same QC as part of the load.

**Help follows the tab.** The `?` in the top bar and **F1** open the section for the tab
you are on — on a DFM tab, for its sub-tab. Each DFM tab and the Opto Light QC tab's lower
panes also carry a `?` at the right of their own tab bar.

## Tabs

| Tab | Shows |
|---|---|
| **Load** | Load the experiment and choose the Slice or the time range |
| **Feeding Summary** | The per-chamber summary table, and the exclusion controls |
| **DFM *n*** | One tab per device: its QC sub-tabs and the Exclude Wells panel |
| **Opto Light QC** | Optogenetic experiments only: was the light where the licks were? |
| **Params** | Detection parameters, with live recompute |

## The Load tab

The **Load Experiment** card names what to load and how:

- **Experiment directory** — the folder holding `flic_config.yaml`, with **Browse…**. A
  member of a Project is loaded through its Project, so it inherits the design.
- **Slice** — shown when the design declares [Slices](concepts-slices.md): **Whole
  recording**, or a Slice, which loads its own minutes in place of the Start and End times.
  See [Viewing a Slice](#viewing-a-slice).
- **Start time** and **End time** — the minutes to load; an End time of 0 loads to the end
  of the recording. See [Restricting the time range](performance.md#restricting-the-time-range).
- **Load DFMs in parallel** — read the DFMs side by side; see
  [Parallel loading](performance.md#parallel-loading).

**Load** reads the recording, writes its QC reports into `qc/` and fills the other tabs.
**Edit Config** opens the [Config Editor](app-config-editor.md) on the folder's
`flic_config.yaml`, and **Reload Config** re-reads that file and prints a short summary of
it — the type, the DFMs and their chamber counts — to the log below. The log also carries
everything the load prints, including each DFM's integrity report.

### Viewing a Slice

Choose a Slice in the **Slice** menu and **Load** to judge that part of the recording on its
own: only its minutes are loaded, its QC goes into `slices/<name>/qc/`, and **Save
Exclusions…** offers the Slice's own exclusion group, so a chamber you exclude here leaves
that Slice and nothing else. See [Exclusions](concepts-slices.md#exclusions). The window
title names the Slice. A member whose recording stops short of the Slice does not load; the
log says why ([Coverage](concepts-slices.md#coverage)).

Opened from the Hub, the viewer starts on the Slice the Hub's
[Slice picker](app-hub.md#the-slice-picker) shows, with the menu set to it.

## The Feeding Summary tab

The table is the experiment's feeding summary, one row per chamber, sorted by DFM and
chamber: the identity columns, then the metrics described in
[What the metrics mean](concepts-metrics.md). Its first column, **Excl.**, is a checkbox:
ticked means excluded. It is kept in step with the Exclude Wells panel of each DFM tab, and
it opens ticked for the chambers already excluded in `remove_chambers.csv` under the
experiment's own exclusion group.

**Click a column header** to sort the rows by it, and click it again to reverse them. Numbers
sort as numbers, and blank values stay at the bottom either way. Sorting by **Excl.** puts the
ticked chambers together. A tick stays with its chamber wherever the sort moves it, and the
sort is kept when the Params tab recomputes the table.

The buttons above the table:

- **Mark All Excluded** and **Clear All** set every box at once.
- **Auto Filter** ticks the chambers the automatic cutoffs would remove, and **View
  Criteria** shows why; see [Auto Filter](#auto-filter).
- **Export CSV…** writes the rows you have *not* excluded to a CSV file you name.
- **Save Exclusions…** records the ticks; see [Saving exclusions](#saving-exclusions).

## The DFM tabs

Each **DFM** tab has its own sub-tabs, and on its right the
[Exclude Wells panel](#the-exclude-wells-panel). Switching DFM tabs keeps the same sub-tab
open, so you can step through the DFMs comparing like with like.

- **Integrity** — data continuity and the recording's messages.
- **Sim. Feeding** and **Bleeding** — two-well experiments only, once QC has run.
- **Raw Signal**, **Baselined** and **Cumulative Licks** — the saved QC plots.
- **Program** — for a DFM with firmware 4.0.0 data or a `Program.txt` section.

### The Integrity sub-tab

The DFM's integrity report from `qc/integrity/`: the table above, the text report below —
rows, time span, gaps and data breaks, lost and repeated records, error flags and, on
firmware 4.0.0 data, the Paradigm sequence and the DFM's restarts. Below the report, the
recording's `Messages.csv` for this DFM and the MCU, when `data/` holds one.

### The messages table

PyMCU writes `Messages.csv` beside the data: what the MCU logged while it recorded. The
Integrity sub-tab lists the messages of this DFM and of the MCU itself (DFM 0), with their
time, the record `Index` they refer to, the text and the type, and counts the errors in its
title. *Lost N records* (*Skipped index* from PyMCU builds before 10/17/2026) is records
lost; *DFM n restarted* and *DFM n recorded nothing for about S s while it restarted* are a
restart; a DFM error code points at the `Error` column. The integrity report lists the same
errors, checks the lost records against the `Index` gaps, and names each restart.

### The Sim. Feeding sub-tab

One row per chamber: how often both of its wells registered feeding in the same sample.
This is your window onto electrical crosstalk; see
[dual-feeding correction](concepts-two-well-pi.md#dual-feeding-correction).

| Column | What it holds |
|---|---|
| `Licks1`, `Licks2` | samples licked at the chamber's first and second well |
| `Both` | samples licked at both at once |
| `MaxMinSignalAtBoth` | across those samples, the largest value of the weaker well's signal |
| `HigherInCol1AtBoth` | how many of them had the first well's signal higher |
| `Both_pre`, `MaxMinSignalAtBoth_pre`, `HigherInCol1AtBoth_pre` | the same, before the dual-feeding correction |

The counts without `_pre` are after the correction — the licks the analysis counts. With the
correction off, the two sets agree; with it on, the difference is what the correction
removed.

**The colours.** A chamber's row is **light red** when 0.1% or more of its lick samples
(`Licks1 + Licks2 − Both`) registered on both wells at once before the correction
(`Both_pre`). It is **dark red** at 1% or more. Hover a coloured row for its numbers. Crosstalk
is rare in good recordings, so a coloured row deserves a look at its two wells' **Baselined**
traces. Nothing is excluded for it.

### The Bleeding sub-tab

How far each well's signal follows the others' (signal bleeding between wells).

- **Bleeding matrix** — one row per well (`W1Sig` …): over the samples where that well's
  baselined signal is above 50, the mean baselined signal of every well (`W1Resp` …). A
  well whose row shows large values in other wells' columns is leaking into them.
- **All data** — each well's mean baselined signal over the whole recording, for
  comparison.

**The colours.** Off the diagonal, a cell is coloured by how it compares with the responding
well's own thresholds, the loaded DFM's (two-well defaults: feeding minimum 10, feeding
threshold 20):

- **Light red** means the response is above the feeding minimum, enough to lengthen a feeding
  event at that well.
- **Dark red** means it is above the feeding threshold, enough to register as licks. This is
  the cell the report's *Bleeding between wells* warning names.

Hover a cell for its numbers. A coloured cell in another chamber's column can be flies in the
two chambers that fed at the same time, not a leak. The responding well's **Baselined** trace,
at the moments the signalling well was fed from, tells which. Nothing is excluded for it.

### The signal plot sub-tabs

**Raw Signal**, **Baselined** and **Cumulative Licks** show the saved plots from `qc/`
for this DFM, once QC has run; until then they say the file is not found, and **Run QC**
in the top bar fills them. The baselined plot carries the feeding thresholds. How to read
them is under [What to look for](#what-to-look-for).

### The Program sub-tab

The [Program timeline](plots-catalog.md#program-timeline) fills the top: the Program
Intervals coloured by Paradigm type, and per well its Acclimation, its Activations and its
light, with its `On` and `Off` stretches and the DFM's restarts on firmware 4.0.0 data.
Below it, two tabs. **Wells** is a table with one row per well at the end of the loaded
range:

| Column | What it holds |
|---|---|
| `Well`, `Chamber`, `Role` | the well, its chamber and the chamber's [Role](concepts-optogenetics.md#roles) |
| `Paradigm` | the Paradigm in force on the last row |
| `Phase` | `A` while in Acclimation, `M` after |
| `AcclimationEndMin` | the minute the well left Acclimation |
| `Activations` | Activations so far |
| `Requirement`, `Progress` | under a ratio Paradigm: the Firmware Events the next Activation needs, and how many it has |
| `LitSec` | seconds its LED was on over the loaded range |

**Activations**, on firmware 4.0.0 data, lists every Activation of the DFM's trigger wells,
one row each, as `qc/opto/opto_light_events.csv` has them:

| Column | What it holds |
|---|---|
| `Well`, `OnsetMin` | the well, and the minute its light came on |
| `Family` | its [Paradigm Family](concepts-optogenetics.md#paradigm-families): closed loop in Acclimation, the Paradigm's own after |
| `Phase`, `K` | `A` or `M`, and its number among the well's Main Activations of its Paradigm run |
| `LitMs` | how long it was lit, in milliseconds |
| `EndedBy` | what ended it: `Hold`, `MaxOnMs`, `length`, `contact`, an Interval change (`interval`), a gap in `Index` (`gap`), the end of the data (`end`), or `unknown` |
| `RequirementMet` | under a ratio Paradigm, the Requirement it met |
| `Overlap` | true when other light merged with it, so its length is not its own and the rule checks skip it |

On a recording older than firmware 4.0.0 the timeline is laid out from the program's clock
times, only `Paradigm`, `Role` and `LitSec` can be filled, and there is no Activations tab.
The sub-tab is computed from the loaded experiment, not read from `qc/`, so a **Params**
recompute redraws it.

A Progressive Ratio experiment's own QC — the light QC table and its figures — is on the
Hub's QC panel rather than here; see
[Light QC](concepts-progressive-ratio.md#light-qc). The Feeding Summary tab shows its
`Group`, `Role` and light QC columns like any other.

### The Exclude Wells panel

On the right of every DFM tab: one checkbox per well, **Well 1** to **Well 12**, ticked
for the wells already excluded. Ticking a well marks its chamber excluded — both wells of
a two-well chamber go together — and the Feeding Summary's **Excl.** box follows, and the
other way round. Nothing is saved until **Save Exclusions…** on the Feeding Summary tab.

## The Opto Light QC tab

Shown for an optogenetic experiment of any type: one row per linkage group, with its
chambers and their `Roles` (`1 Paired, 2 Yoked`), its lit time, the share of it no
trigger-well lick or touch explains, the minute that began, the contact with no lick — the
emulated firmware trigger's, or on firmware 4.0.0 data the light of Activations with no lick
or touch at their onset — and the verdict, coloured by tone. On firmware 4.0.0 data the rule
checks' columns follow: short, over `MaxOnMs`, cut-offs, wrong length, lockout broken, missed
contacts, series wrong, non-feeding dark, relight delay, the lowest open-loop lit fraction,
the interval lag and the restarts, blank where the group's Paradigms have no such rule (see
[Each Paradigm's own rule](concepts-optogenetics.md#each-paradigms-own-rule)). The last
column, **Likely cause**, is explained under
[Drift or hardware?](concepts-optogenetics.md#drift-or-hardware). The tab opens on the
worst group. Selecting a group fills the panes below it:

- **Why** — the verdict, a line per flag with its numbers and likely cause, and the notes
  about its DFM and its program. See [The checks](concepts-optogenetics.md#the-checks).
- **Intervals**, **Light events** and **Program** — the group's rows of the tables in
  `qc/opto/` (see [Outputs](concepts-optogenetics.md#outputs)), and its DFM's program as
  read. A column empty on every row — a setting none of the DFM's Paradigms has, a firmware
  column on an older recording — is hidden.
- The **Light explained by licks** figure — the selected group's panels, or the whole DFM
  with **Selected group only** unticked, for the DFM chosen in **Figure for DFM**, at the
  **Bin (min)** you choose. See [Explained light](concepts-optogenetics.md#explained-light).

The tab is computed from the loaded experiment, not read from `qc/opto/`, so a **Params**
recompute updates its verdicts at once: a threshold that loses real licks makes healthy
light look unexplained, which is worth seeing before you commit to it.

**Mark Selected Group Excluded** and **Mark Failed Groups Excluded** tick the chambers of
those groups on the Feeding Summary and DFM tabs. They save nothing: a failed light does not
invalidate the feeding record, so whether to exclude is yours to decide, and **Save
Exclusions…** records the decision. See
[Optogenetic experiments](concepts-optogenetics.md).

## What to look for

**Does the baseline sit where it should?** It should track the background and ignore
feeding. A baseline that visibly rises during long bouts means the running median is being
pulled up by the feeding itself — see [the baseline](concepts-signal.md).

**Does detection match the visible bouts?** The plots do not mark individual events, so set
a well's **Baselined** trace, with its threshold lines, beside its **Cumulative Licks**
curve, which climbs only where licks were detected. Climbs where the trace is flat mean the
thresholds are too low. Clear deflections with no climb mean they are too high.

**Are bouts being split or merged?** The plots cannot show this, but the Feeding Summary
can: many short events per chamber suggest the link gap is too small, and a few very long
ones that it is too large. This is the single most common cause of surprising duration and
event-count results — see [event linking](concepts-licks-events.md#event-linking-the-link-gap).

**Are some wells dead?** A well with no signal at all is an empty position or a hardware
fault. Exclude it rather than letting a zero into your averages.

## The Params tab

Change a detection parameter and recompute without editing your configuration or
restarting. This is the right way to choose parameter values: try one, look at the traces,
try another.

The workflow that works:

1. Open a DFM tab and find a chamber whose behaviour you can interpret by eye.
2. Change one parameter in **Params** and **Recompute**.
3. Look at whether detection now matches your reading of the trace.
4. When it does, write that value into your configuration file — **the Params tab does not
   save it for you.**

Each parameter has a `?` button that opens
[Parameter reference](reference-parameters.md) at that parameter. The fields start at the
first DFM's values.

Changing parameters here affects only this session. It is a place to experiment, not a
place to configure.

## Saving exclusions

On the **Feeding Summary** tab, tick the chambers to exclude — or the wells, on a DFM tab's
Exclude Wells panel. **Save Exclusions…** asks for a group name, offering the experiment's
own exclusion group (its design's `exclusion_group:`, `general` when it names none; a
Slice's own group when a Slice is loaded), and writes your current selection to that group
in `remove_chambers.csv`, replacing the group's rows. Naming the own group of another
analysis — the whole recording's, or another Slice's — asks first, saying how many of its
rows would be replaced. That is the intended path from
"this chamber looks wrong" to a recorded, reusable exclusion set — see
[exclusions](config-dfms-chambers.md#excluding-chambers).

Name the group after the analysis it belongs to. A script's bare `remove_chambers` step
looks for a group named after the script.

## Auto Filter

**Auto Filter** applies the automatic cutoffs — those under `global.constants`, or the
Experiment Type's defaults — and ticks the chambers they would remove. For a Progressive
Ratio experiment that includes the chamber groups whose training never completed or whose
light QC failed. **View Criteria** shows what the last run used and why each chamber was
selected; its window opens this section with **F1**. Review the selection before saving —
automatic criteria are a starting point, not a verdict. See
[Automatic exclusion](config-dfms-chambers.md#automatic-exclusion).

---

Related: [How feeding is detected](concepts-licks-events.md) ·
[Parameter reference](reference-parameters.md) · [Analysis Hub](app-hub.md)
