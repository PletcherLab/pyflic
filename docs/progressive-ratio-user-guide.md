# Analysing a Progressive Ratio assay with pyflic

A guide for researchers running Progressive Ratio (PR) experiments on the
FLIC. It explains what pyflic measures, which settings you control and how
to choose them, how to read each figure, and what the statistics do and do
not let you conclude.

You do not need to read code to use it. For installing and starting pyflic,
see [Getting Started](GETTING_STARTED.md); if you installed it that way, type
the `pyflic …` commands below as `uv run pyflic …`, from the pyflic folder.
For the quality-control checks every experiment gets, including the
optogenetic light QC, see [Checking your data](qc-user-guide.md). The in-app
help topic *Progressive Ratio experiments* (`pyflic help`) covers the same
ground as a concise reference. Developers will find the retired design
plan for the assay listed in the documentation index, under
[Removed plans](README.md#removed-plans).

---

## 1. The assay in one page

A DFM's six chambers form **chamber groups**. The chambers of a group share
one light circuit and one treatment.

- The **paired** fly controls the light. When it feeds at the **sucrose
  well** (always well A), the light comes on, typically driving
  optogenetic stimulation.
- In a **paired design** each group is a pair: chambers 1+2, 3+4 and 5+6,
  or as the program links them. The **yoked** fly, the paired fly's
  partner, is lit at exactly the same moments, whatever it does itself. It
  receives the same light but cannot earn it.
- In a **design without yoked controls** each group is a single chamber, a
  **solo group**: a paired fly alone, whose light answers to its own feeding
  only. This is PyMCU's reference design.

The recording has two phases:

1. **Training.** Closed loop: every sucrose feed by the paired fly turns
   the light on. The firmware marks training in the data, and each group
   finishes training at its own time, when its paired fly has done so.
2. **Test.** The firmware demands progressively more feeding for each light
   event. A fly that finds the light rewarding keeps working as the price
   rises, up to a point. The price is time in contact with the food; a
   Paradigm with `Counter: Events` counts feeding bouts instead. A fixed
   ratio, the same price every time, is possible too.

The analysis asks three questions:

- **How far did the paired fly keep working as the requirement rose?** The
  **breaking point**.
- **How long did each fly keep feeding at the sucrose well?** **Sucrose
  persistence**, for every fly.
- **Does the paired fly feed differently from a fly that got the same
  light for free?** The **paired − yoked difference**, within each pair.
  Only a paired design can ask this one.

Two rules run through everything:

- **Differences are always taken within a pair, never between group
  means.** The yoked fly is the paired fly's own control: same device,
  same light, same moments.
- **Time runs from each group's training end**, not from the start of the
  recording.

---

## 2. Before you start

### Data you need

This guide describes recordings from DFM firmware 4.0.1 or later. They
record Training (each well's Phase), every light event (each well's
Activations) and the Requirement it met in their own columns, and the light
exactly. pyflic calls such a recording **generation 3**. Recordings from
older firmware hold less; see [Older recordings](#older-recordings) below.

- **Both wells named:** well A is the sucrose well, well B the yeast well.
- **The MCU's `Program.txt`.** The MCU writes the program it ran back out as
  `Program.txt`; copy it into the member's `data/` folder, beside the DFM
  files. It tells pyflic each well's threshold, the linkage and the light's
  rule — its **Paradigm**: ProgressiveRatio or FixedRatio, with its
  settings — which the light QC needs to judge the light properly, and its
  linkage can give the chamber groups for you. **It is all but required:**
  the data name each Paradigm but not its type, so without the file a fixed
  ratio cannot be told from a progressive one, and pyflic warns loudly
  (*firmware 4.0.0 recording without Program.txt*).

A program trains the flies in one of two ways: a ratio
Paradigm whose `Acclimation` setting gives each fly a set number of
closed-loop light events first, or a ClosedLoop Paradigm run before the
ratio Paradigm. Either way, pyflic reads that opening as Training. The
second needs an Experiment Type of your own, because the shipped one does
not take a ClosedLoop interval (below).

#### Older recordings

Firmware 1.0.5 never ran a progressive ratio. pyflic still reads
recordings from firmware 2.x and 3.x, which did, but they hold less:

- **Generation 1** (PyMCU 3.x): training is a flag inside the sucrose
  well's signal, and light events are onsets of the light in `OptoCol1`,
  which samples it five times a second. Without `OptoCol1` no group ever
  finishes training, and every group is removed.
- **Generation 2** (PyMCU 4 with firmware 3.x or older): no training
  information at all. Each group is kept as one `Recording` phase with no
  Test, no breaking point and no sucrose persistence (section 5.1).

Neither records Phase, Requirement or the firmware's own count, so the
checks and columns below that use them are left out. Without `Program.txt`
the analysis of an older recording still runs, with a more limited light
QC.

### The Experiment Type

Everything this guide describes is the work of pyflic's **Progressive
Ratio** Analysis Module: the chamber groups, Training and Test, the light QC
of section 5.2, the paired − yoked difference, the breaking point and
sucrose persistence. Any Experiment Type that includes the module analyses a
recording this way. pyflic ships one, `ProgressiveRatio`, listed as
*Progressive Ratio* in the Experiment Type menus of the Project design
dialog and the Config Editor. It states:

- **the chamber layout**: two-well, so a config of the type never states
  `chamber_layout`;
- **the light it expects**: the two ratio Paradigms, ProgressiveRatio and
  FixedRatio, with open-loop light (an OpenLoop interval, or wells set `On`)
  and dark intervals beside them;
- **the wells**: both must be named, well A the sucrose well;
- **its Analysis Modules**: Progressive Ratio, and Light and dark, which adds
  *Feeding in light and dark* to the report's open-loop light (section 6);
- **its default cutoffs**: those of section 4;
- **no control comparisons**: its reports lead with the breaking point and,
  in a paired design, the paired − yoked difference.

**The program is checked against the type.** A `Program.txt` that runs any
other kind of light, such as a ClosedLoop interval for Training, stops the
load. The message names the Paradigm type, the DFM and the Program Interval:
*the Opto Program runs ClosedLoop (DFM 1 interval 1), and experiment_type
'ProgressiveRatio' expects ProgressiveRatio, FixedRatio, with OpenLoop and
dark intervals beside them; choose or build a type that expects it*.

**A type of your own.** Build it once in the Experiment Type Editor
(`pyflic-types`, **Experiment type editor** on the Hub's Tools panel, or
**New type…** beside the Experiment Type menu). Select *Progressive Ratio*,
click **New from…**, give the copy a name of your own, change what your
assay needs, and **Save to library**. For a program that trains under
ClosedLoop, add ClosedLoop to its light; its Training then includes the
ClosedLoop interval (section 5.1). A lab type may also name a **control
comparison** for its reports to lead with:

- a **Treatment control**: one of its factors, such as Genotype. The Design
  names the control level with `control_level:`, and the reports compare
  every other level with it;
- a **Role comparison** of Paired against Yoked, which moves the paired −
  yoked difference to the front of the reports.

The module needs a two-well layout and ProgressiveRatio or FixedRatio among
the light the type expects, and a type with it cannot also include the
Hedonic module, or the Paired and yoked module of other optogenetic types:
it keeps its own paired − yoked difference. The editor's **Problems** list
names anything of this kind, and **Save to library** stays off until it is
empty. Saving a Project design whose type is a
lab type writes a copy, `experiment_type.yaml`, beside `project.yaml`, so
the Project analyses the same on a machine without your library. The help
topics *Experiment types and chamber layouts* and *Experiment Type Editor*
cover the editor in full.

### Setting up the experiment

For each DFM, the config states which side of the chamber well A (sucrose)
is on, and how the chambers form groups. There are three ways to say the
second.

**Pairs, with `paired_chambers`.** Name the paired chamber of each pair
1+2, 3+4 and 5+6:

```yaml
dfms:
  1:
    params: {pi_direction: left}      # which side well A (sucrose) is on
    paired_chambers: [1, 4, 5]        # exactly one chamber from each pair
    chambers: {1: Ctrl, 2: Ctrl, 3: Exp, 4: Exp, 5: Exp, 6: Exp}
```

- **`paired_chambers`** must name exactly one chamber from each of 1–2,
  3–4 and 5–6. The yoked chamber is the other one and is never written.
- **Both chambers of a pair must carry the same treatment.** pyflic
  refuses to load a config where they differ.

**Single chambers, with `chamber_groups: single`.** Every chamber is its
own group, a solo group with no yoked control:

```yaml
dfms:
  1:
    params: {pi_direction: left}
    chamber_groups: single            # six groups of one chamber each
    chambers: {1: Ctrl, 2: Exp, 3: Ctrl, 4: Exp, 5: Ctrl, 6: Exp}
```

Each chamber can have its own treatment. `chamber_groups: single` and
`paired_chambers` cannot both be stated.

**From `Program.txt`, by stating neither.** With the MCU's `Program.txt`
in `data/`, pyflic reads the groups from the program's linkage:

- a linkage group that lights two chambers is a pair, and the chamber
  holding its trigger well is the paired one;
- a linkage group that lights one chamber is a solo group, with no yoked
  control;
- pairs that are not neighbours, such as 1+4 and 2+3, are read as the
  program pairs them.

If the linkage makes no sensible groups (one light circuit over three
chambers, say), pyflic tries the pairs 1+2, 3+4 and 5+6 with the program's
trigger wells. Without `Program.txt`, or when that fails too, the DFM does
not load and asks for `paired_chambers` (or `chamber_groups: single`).

Whichever you use:

- **What the config states wins.** If you give `paired_chambers` or
  `chamber_groups` and the program disagrees, `summary.txt` says so. A
  trigger well that is well B is flagged too, because the sucrose well is
  always well A.
- **`chamber_groups` belongs to a DFM.** It is read from a DFM's entry
  only, never from `global:`, so DFMs of one recording can differ.
- **Paired and yoked are roles, not treatments.** Don't create a "paired"
  treatment or design factor; pyflic adds a `Role` column itself.
- **In the Config Editor**, under any type with the Progressive Ratio
  module, each DFM tab has a *Paired chamber per chamber group* card with a
  **Groups** choice: *Pairs*, with three pickers for the paired chambers;
  *Single chambers*; or *From Program.txt*, which writes neither key. A DFM
  entry that already names neither opens on *From Program.txt*.
- **`chamber_groups: paired`** may be written too. It means the pairs 1+2,
  3+4 and 5+6, with their paired chambers from `paired_chambers` or, without
  it, from the program's trigger wells.
- **Groups are numbered in chamber order**, so on a DFM of single chambers
  group 3 is chamber 3.

### Setting up the Project

Put the recordings that answer one question into one **Project**. Its
`project.yaml` holds the shared **design**, which every member inherits:

```yaml
# project.yaml
name: my_pr_study
design:
  global:
    experiment_type: ProgressiveRatio
    well_names: {A: Sucrose, B: Yeast}
    constants:
      pr_break_gap_min: 120
      # pr_test_window_min: 720        # optional; see section 4
    experimental_design_factors:
      Genotype: [Ctrl, Exp]
```

Edit it in the Hub with **Project panel → Project design…**. Its
**Experiment type** menu chooses *Progressive Ratio* or your own type, with
**New type…** beside it, and its sections hold every setting in section 4.
A lab type with a Treatment control adds a **Control level** field there.
**Set the analysis constants here, once.**
Every member is then analysed under the same rules, which pooling
requires.

---

## 3. Running the analysis

For each member (experiment), load it in the Hub by double-clicking its row
in the Project panel, then use the **QC**, **Analyze** and **Plots** tiles.
Each panel lists what its one button writes as checkboxes, in groups, with
every box ticked to begin with. A **Progressive Ratio** group appears on
each panel for a member whose type includes the Progressive Ratio module.

1. **Hub → QC → Run QC.** This writes the ticked items of the per-DFM QC
   bundle into `qc/` (the *QC reports* group and, for an optogenetic
   recording, the *Optogenetics* group), and those of the Progressive Ratio
   group into `analysis/`:
   - *Light QC table*: `pr_light_qc.csv` and `pr_light_events.csv`, the
     light QC of section 5.2, with every flagged group in the log;
   - *Licks per light event*: `pr_light_events_dfm<n>.png`;
   - *Sucrose Well resting level*: `pr_resting_level_dfm<n>.png`.
2. **Hub → Analyze → Run analysis.** This always writes the core first: the
   auto-removal (section 5.3), `removed_chambers.csv`, `summary.txt`, and
   the feeding summaries for the whole recording and per phase
   (`feeding_summary.csv`, `feeding_summary_facet.csv`). Then, into
   `analysis/`, each ticked item:
   - *Binned summary* and *Event statistics*, as for any member, and *Light
     tables* for an optogenetic recording;
   - *Paired-Yoked Difference*: `paired_yoked_diff.csv` (section 5.4). The
     box is greyed when every chamber group is a solo group;
   - *Cumulative difference table*: `pr_cumulative_diff.csv` (section 5.5);
   - *Breaking point*: `pr_breaking_point.csv` (section 5.6), with every
     group's breaking point and its sensitivity to the gap in the log;
   - *Experiment report (PDF)*: `experiment_report.pdf`, last (or
     `pyflic report <member folder>`).

   The *Paired and yoked comparison* that other optogenetic types offer is
   not offered here: a type with the Progressive Ratio module keeps its own
   paired − yoked difference.
3. **Hub → Plots → Generate plots.** This saves each ticked figure into
   `analysis/` and opens it in a tab (section 6):
   - *Feeding summary* and *Well A vs B*, and, for the metric chosen in the
     *Chosen metric* group, a *Binned time course* and a *Dot plot*, whose
     points are chamber groups;
   - *Cumulative difference curve*, *Training-aligned traces*,
     *Still-responding curve* and *Breaking-point plots*.

Untick what you do not need. The choice is saved at once in the Project's
design (`project.yaml`, under `design.global.omit`), so it holds for every
member; a standalone experiment keeps it in its own `flic_config.yaml`. A
script's `run_qc`, `run_analysis` and `run_plots` steps leave
out the same. A run states in its log (and Run analysis in `summary.txt`)
what it left out, and it removes no file: an item you untick keeps whatever
an earlier run wrote.

The Project panel's **Analyze all** runs every member's QC, Basic analysis
and report in one go, **whatever the ticks** (below).

For the Project:

4. **Project panel → Combine.** This stacks every member's tables and runs
   the pooled statistics (`<project>_Stats.txt`).
5. **Project panel → Create report**, or `pyflic report <project folder>`.
   This writes the Project Report.

Every step is also a Script Editor action, so a Project Script or
`pyflic batch` can run it unattended. **Re-run the analysis after changing
any setting.** Members analysed before a setting changed keep their old
numbers until you do.

**Check each member's data before its results.** Run analysis does not
write the per-DFM QC bundle (integrity, data breaks, crosstalk and the
signal plots). **Hub → QC → Run QC** writes it, and **Open QC Viewer**, on
the QC panel's View card, shows it. [Checking your data](qc-user-guide.md)
explains every check.

### What basic analysis does, in order

Basic analysis is what **Analyze all** and a script's `basic_analysis` run.
It does not read the panels' ticks. Its figures are the cumulative
difference curve, the still-responding curve and, per DFM, the
training-aligned traces, licks per light event and Sucrose Well resting
level. With every box ticked, the Hub's three buttons write everything it
writes, and more: the binned summary, the event statistics, the
breaking-point plots and the *Chosen metric* figures.

1. **Training.** It finds each group's training end (section 5.1).
2. **Quality control and exclusions.** It checks training and light for
   every group, and removes chambers that fail (sections 5.2 and 5.3).
   It also runs the optogenetic light QC and writes it into `qc/opto/`
   (section 5.2).
3. **Feeding summaries.** It computes the standard two-well metrics (licks,
   events, durations, preference index) per chamber, for the whole
   recording and separately for the Training and Test phases.
4. **Paired − yoked difference.** One row per pair per phase (section 5.4).
   A design without yoked controls has none.
5. **Cumulative difference curve** (section 5.5), over the pairs.
6. **Breaking point** and **sucrose persistence** (sections 5.6 and 5.7).
7. **Figures** and the text summary (`summary.txt`).

---

## 4. Settings you control

All of these are in the Project's design (**Project design…**; for a
standalone experiment, the Config Editor), except the figure bin size, which
is set per plot action. The defaults below are the shipped
`ProgressiveRatio` type's; a type of your own may state others, and both
editors show the type's default in grey in an empty field. The defaults are
chosen to be sensible for a typical overnight recording. Change one only for
a reason you can state.

### Study design

| Setting | What it does | Choosing it |
|---|---|---|
| `paired_chambers` (per DFM) | which chamber of each pair controls the light | from your rig setup; must match what the firmware did. Leave it out for single chambers or when `Program.txt` gives the groups |
| `chamber_groups` (per DFM) | `single`: every chamber its own group, with no yoked control; `paired` (the default): pairs | from your rig setup; leave it out when `Program.txt` gives the groups |
| `pi_direction` (per DFM) | which side well A, the sucrose well, is on | from your rig setup; may differ between DFMs |
| treatments (`chambers:`) | what each group received | the same for both chambers of a pair |
| `experimental_design_factors` | factors such as genotype and drug, crossed | the first factor is the x axis of the dot plots |
| `well_names` | labels for wells A and B in plots and reports | well A must be the sucrose well |
| `transform_licks` | transforms lick counts in the summary tables | the cumulative curves always use raw counts regardless |

### Detection parameters

`feeding_threshold`, `feeding_minimum`, `tasting_minimum` and the other
detection parameters decide which signal deflections count as licks and
feeding events. They matter more in this assay than in most, because the
**light QC** and the **breaking point** both depend on whether a light event
was backed by at least one lick. See the *Parameter reference* help topic. Change
them for the whole Project, never for one member.

### Exclusion

| Setting | Default | Effect |
|---|---|---|
| `min_untransformed_licks_cutoff` | 20 | removes a chamber whose sucrose or yeast well has fewer licks than this over the whole recording, counted before any transform |
| `max_med_duration_cutoff` | 13.0 | removes a chamber whose median feeding-event duration at either well is above this (a sign of a stuck or bleeding signal) |
| `max_events_cutoff` | 150000 | removes a chamber with an implausible number of events at either well |
| `require_training_complete` | true | removes **every** chamber of a group (both of a pair, the one of a single chamber) whose paired fly never finished training; a generation-2 group, which has no training information, is kept |
| `exclude_failed_pr_groups` | true | removes **every** chamber of a group that fails the light QC |

The optogenetic light QC has its own switch, `exclude_failed_opto_chambers`,
in the optogenetic table below. It is off by default.

### Light QC thresholds

| Setting | Default | Meaning |
|---|---|---|
| `pr_lick_free_run` | 5 | this many lick-free Test light events **in a row** fails the group as self-triggered |
| `pr_trend_min_events` | 5 | the fewest Test light events needed to judge the lick trend |
| `pr_trend_min_rho` | 0.3 | the licks-per-event trend must reach this rank correlation, or the group gets a warning |
| `pr_resting_level_rise` | 15 | a rise in the sucrose well's resting signal of this many counts is a warning |
| `pr_resting_level_ratio` | 3 | a sucrose well resting this many times higher than the DFM's other sucrose wells is a warning |

### Optogenetic light QC thresholds

Every optogenetic experiment, PR or not, also gets the general light QC of
section 5.2. Its settings, with their defaults:

| Setting | Default | Meaning |
|---|---|---|
| `optogenetics` | `auto` | run it when `Program.txt` or light data exist; `true` always, `false` never (a design key, and a per-DFM one) |
| `exclude_failed_opto_chambers` | false | remove the chambers of a failed linkage group; off, they are flagged and kept |
| `opto_unexplained_warn_fraction` | 0.10 | a group whose lit time is this fraction unexplained by licks gets a warning |
| `opto_unexplained_fail_fraction` | 0.30 | this fraction fails the group |
| `opto_unexplained_min_sec` | 30 | seconds of unexplained light needed before either is judged |
| `opto_default_decay_ms` | 1000 | the light's Hold assumed without `Program.txt` |
| `opto_decay_tolerance_samples` | 2 | slack, in samples, around the Hold |
| `opto_unlit_feeding_fraction` | 0.50 | in a closed-loop interval, this share of feeding bouts that never lit the group is a warning |
| `opto_open_loop_min_lit_fraction` | 0.95 | an open-loop interval lit for less than this share of its time is a warning |
| `opto_event_disagreement_warn_fraction` | 0.25 | a paired sucrose well whose firmware-counted feeding bouts and pyflic's differ by more than this share is a warning |
| `opto_yoking_fail_fraction` | 0.01 | a group whose yoked light differs from the paired light by more than 1 ms in more than this share of its lit records fails |
| `opto_lost_records_warn` | 25 | more records lost than this is a warning for the DFM |
| `opto_unknown_rows_warn_fraction` | 0.01 | more than this share of rows under an instruction outside the program is a warning for the DFM |
| `opto_late_records_warn` | 10 | more late records than this is a warning for the DFM |
| `opto_activation_tolerance_ms` | 25 | how far an Activation's light may differ from its Paradigm's setting and still keep the rule |
| `opto_activation_length_fail_fraction` | 0.05 | more than this share of a group's judged Activations lasting the wrong time (*wrong activation length*, *short activations*) fails it |
| `opto_lockout_fail_fraction` | 0.02 | FixedInterval: more than this share of Activations started inside the lockout fails the group |
| `opto_missed_contact_warn_fraction` | 0.25 | FixedInterval: contact that should have lit the light and did not, above this share, is a warning |
| `opto_non_feeding_lit_warn_fraction` | 0.10 | NonFeeding: lit through more than this share of the feeding is a warning |
| `opto_non_feeding_min_sec` | 30 | NonFeeding: seconds of that light needed before it is judged |
| `opto_non_feeding_dark_warn_fraction` | 0.10 | NonFeeding: dark for more than this share of the time the rule says lit is a warning |
| `opto_interval_lag_warn_sec` | 5 | a Program Interval starting more than this many seconds late is a warning for the DFM |

The last thirteen use the Paradigm columns, so they don't apply to older
recordings. Their defaults
were set on simulated recordings and will be revisited once there are
bench recordings of each Paradigm type. Of the checks of each Paradigm's
own rule, those that apply to a ratio program are *wrong activation length*
and *requirement series wrong* for the ratio Paradigm, *short activations*
for its Acclimation or a ClosedLoop interval before it, and *interval
started late* and *DFM restarted*, which apply to every program.
[Checking your data](qc-user-guide.md), sections 6 and 7, describes the
checks these settings drive.

A type with the Progressive Ratio module owns its phases, Training and Test,
so its design never states `facet_cutoffs` or `facet_source`.

### Breaking point

| Setting | Default | Meaning |
|---|---|---|
| `pr_break_gap_min` | 120 minutes | the pause that counts as "the fly stopped". It is used for both the breaking point and sucrose persistence |
| `pr_test_window_min` | off | caps every group's Test window at the same length |

**Choosing `pr_break_gap_min`.** The breaking point moves with this value,
so pyflic always shows every group's breaking point at 60, 120 and 240
minutes beside your setting (in `summary.txt`, and in the log when Run
analysis writes the *Breaking point* item). Look at that table **before** you
settle on a gap:

- If the ranking of treatments is the same at every gap, the result is
  robust.
- If it changes, report that.

Pick the gap on biological grounds, such as how long a fly normally goes
between feeding bouts, and not by choosing whichever value gives the
smallest p-value. pyflic does not know the time of day, so a long night
pause ends a count like any other.

**Choosing `pr_test_window_min`.** Each group is watched from its own
training end to the end of the ratio Paradigm's run, or to a DFM restart
inside it (section 5.6; on a generation 1 recording, to the end of the
recording), so a group that trained late gets less Test time and is more
likely to be censored (section 5.6). The
`TestMinutes` column of `pr_breaking_point.csv` shows each group's window.
If those windows differ a lot, set `pr_test_window_min` to a value no longer
than the shortest one: every group is then judged over the same span. It
must be comfortably longer than `pr_break_gap_min`, or almost every group
will be censored.

### Figures

The cumulative-curve actions (*Cumulative difference curve*,
*Training-aligned traces*) take a **bin size**, default 1 minute. It changes
only the smoothness of the curve, not any statistic. The Hub always draws
them, and writes the cumulative difference table, at 1 minute, whatever the
Analyze panel's **Bin size** box says; that box sets only the binned
summary, the report's time courses and the Plots panel's *Binned time
course*.

---

## 5. The analyses

### 5.1 Training

The firmware records each well's **Phase**: `A` while the
well is in **Acclimation**, its opening run of closed-loop light events, and
`M` (Main) after. The training end is the paired sucrose well's first row in
`M` under the ratio Paradigm. Everything before it is Training, including a
ClosedLoop interval that ran before the ratio Paradigm (which needs a type
of your own that expects ClosedLoop, section 2). The group's other
wells are `Off`, read `M` throughout and say nothing, so there are no
per-well notes. The notes say instead:

- *Training was empty*, when the ratio Paradigm had no Acclimation and no
  interval before it. Training then ended at the program's first row, and
  the group counts as having completed it.
- why Training never completed: the paired sucrose well never ran a ratio
  Paradigm, was still in Acclimation when the recording ended, or never
  reached `M` under one;
- *Paradigm restarted at Program Interval k*, when the well went back into
  Acclimation later. The firmware starts a well afresh whenever the Paradigm
  changes name, and after the DFM itself restarts; the note then adds
  *after the DFM restarted at minute m* (section 5.6).

A group whose paired fly **never finished training** has no Test phase. It
appears as `TrainingComplete = false`, and by default every chamber of the
group is removed.

**Where to look:** the report's *Progressive ratio: training* table (one row
per group, with its training end) and the training notes beneath it. The
Program timeline (section 6) shows each well's Acclimation directly.

**Older recordings** ([section 2](#older-recordings)):

- **Generation 1.** The training end is the last minute its paired fly's
  sucrose well carries the firmware's training flag. The group's other
  wells normally clear at the same moment. If they don't, the difference
  appears as a *training-flag note* in `summary.txt` and the report: QC
  information, not an error, since the analysis always follows the paired
  sucrose well.
- **Generation 2** records no Training at all. Nothing is removed for it:
  each group becomes one phase, `Recording`, with no Test, so no breaking
  point and no sucrose persistence, and it is never pooled with other
  members' Test. `TrainingComplete` is `NA`. `summary.txt`, the report's
  cover, the Hub's **Loaded** line, the Project Report's member table and
  the Stats text all say *no training information: recorded with DFM
  firmware older than 4.0.0*.

### 5.2 Light QC: did the paired fly earn its light?

The firmware switches the light on from its own live reading of the paired
sucrose well. pyflic counts licks afterwards, from a cleaned-up signal.
Usually the two agree. When a sucrose well's resting signal slowly creeps
up, however, the firmware sees a well that is constantly touched and fires
the light on its own. pyflic sees a flat signal and no licks. The light
then describes the sensor, not the fly, and every number built on it is
meaningless. The light QC catches this, per group, over the whole
recording.

A **light event** is one onset of the group's light: an **Activation** of
the paired sucrose well, the firmware's own record of each light it switched
on for that well, exact to the record. (On a generation 1 or 2 recording it
is read from `OptoCol1`, which samples the light five times a second.)

| Check | What it means | Verdict |
|---|---|---|
| **Self-triggered light** | 5 (`pr_lick_free_run`) or more Test light events in a row with no sucrose lick since the previous one | **fails** the group |
| **Implausible training** | training "completed", with light events, but without a single sucrose lick | **fails** the group |
| **Requirement series wrong** | a Test light event met a Requirement that the Test Paradigm's series in `Program.txt` does not give | **fails** the group |
| **No increasing trend** | licks per light event don't rise across the Test phase, as a working progressive ratio should; not judged under a fixed ratio | warning |
| **Resting level rise** | the sucrose well's resting signal rose during the recording | warning |
| **Resting level elevated** | the sucrose well rests well above the DFM's other sucrose wells | warning |

Some behaviour here is normal and is not flagged:

- **An occasional lick-free light event.** The firmware responds to brief
  touches below pyflic's feeding threshold. That is why only a *run* fails
  a group.
- **Fewer training licks than training light events**, for the same
  reason. Only *zero* licks is implausible.
- **A group with no Test light events at all** simply stopped before its
  first Test requirement. That is a breaking point of 0, not a failure.

A failed group is removed by default. `pr_light_qc.csv` still lists it, and
the QC figures keep it on screen with **EXCLUDED** and the reason in its
panel title, so you can check the decision. Warnings never remove a group;
read them as reasons to look at the QC figures.

If you would rather judge a failed group yourself, `LickFreeRunStartMin` in
`pr_light_qc.csv` is the minute after training end at which its failing run
began. Two ways to use it:

- set `exclude_failed_pr_groups: false` and cap the analysis with
  `pr_test_window_min`;
- exclude the chambers by hand.

Whichever you choose, apply the same rule to every group, whatever its
treatment.

**Lick-free allows for the Hold.** The firmware keeps the light on for its
*Hold* after the contact that earned it (the old programs' *decay*), and a
brief touch it counts may read below pyflic's feeding threshold. A light
event is therefore lick-free only when it has no sucrose lick since the
previous one *and* no sucrose lick or touch within its Hold — the one
`Program.txt` states, or `opto_default_decay_ms` without it. A light the fly
touched for is never counted against it, and the breaking point counts it.

**The firmware's own count.** Under `Counter: Contact`, the default, the
firmware counts milliseconds of contact, not feeding bouts, so there are no
**Firmware Events**: `TrendRhoFirmware` in `pr_light_qc.csv` is NaN and only
the trend of pyflic's licks is judged. Under `Counter: Events` the firmware
counts the feeding bouts it saw at the paired sucrose well, its Firmware
Events, and records how many each light event needed. pyflic measures their
trend across the Test, `TrendRhoFirmware`, beside the trend of pyflic's
licks. The *no increasing trend* warning stays on pyflic's licks. When the
firmware's count rises and pyflic's licks do not, the group gets a note
instead: *the firmware counted rising Events per light event; pyflic's licks
did not follow: check detection*. The fly worked to the schedule, and
pyflic's detection missed some of it. Under a fixed ratio the requirement
never rises, so no trend is judged at all.

**The requirement series.** The firmware records the
Requirement each light event met, and it keeps its program's series
exactly: under a progressive ratio each Requirement is the next one the
series gives, under a fixed ratio each is the `Ratio`. pyflic checks every
Test light event of the paired sucrose well against the Test Paradigm's
series in `Program.txt`, with the same check the general light QC uses. One
that differs fails the group as *requirement series wrong*. It cannot
happen on a working system with the right program, so the likely cause is
that `Program.txt` is not the program that ran: check that it is the one
exported with this recording. One case the data cannot settle is forgiven:
a light event counted in the very record the ratio Paradigm began may
belong to the Paradigm before it, so a run whose Requirements match the
series from the next event on is not a mismatch. The check needs
`Program.txt`, because the series is written only there.

**The general optogenetic light QC runs beside this one.** It asks a
simpler question of every linkage group — here, every chamber group — from
lit *time* rather than light events: how much of the light had no lick or
touch behind it? With `Program.txt` it also re-runs the firmware's own
trigger on the raw signal, so a well the firmware read as touched while
pyflic saw nothing is named as a drifting baseline. The two checks can
disagree. On the first real dataset, a group lit for four hours during
training passes this check, because one long light event is not a run of
lick-free events, and fails the general one, because 93% of those four
hours had no lick. The general check also confirms the yoking, in a pair: the yoked chamber's wells must count no
light events of their own and get exactly the paired well's light, or the
group fails for *broken yoking*. It checks each Paradigm against its own
rule as well, one light event at a time: each light event under the ratio
Paradigm after Acclimation must last its `ActivationMs` (*wrong activation
length*) and meet
the series (*requirement series wrong*), and each Acclimation light event
must stay lit for at least its Hold (*short activations*). These fail a
group, because the firmware keeps its rules exactly. A group the general
check fails is flagged, not removed,
unless you switch on `exclude_failed_opto_chambers`. Its results are in
`qc/opto/`, an *Optogenetic light QC* section of `summary.txt` and of the
report, and the **Light explained by licks** figure. The QC Viewer's
**Opto Light QC** tab shows the same verdicts group by group, says why each
group got its own, recomputes them when you change a detection parameter,
and can tick a failed group's chambers for exclusion. Every check it makes,
and how to read its figures, is in [Checking your data](qc-user-guide.md),
sections 6 and 7.

### 5.3 Exclusions

Chambers leave the analysis for three kinds of reason:

- **By hand**, in `remove_chambers.csv`.
- **A chamber-level cutoff:** too few licks, an implausible duration or an
  implausible event count. Only that chamber is removed.
- **A group-level failure:** training never completed, or the light QC
  failed. Every chamber of the group is removed: both of a pair, the one
  of a single chamber.

A linkage group that fails the optogenetic light QC is flagged, not
removed, unless you switch on `exclude_failed_opto_chambers`. Here a
linkage group is a chamber group, so its failure would remove every
chamber of the group too.

Every automatic removal and its reason is written to `removed_chambers.csv`
and listed in `summary.txt`. The Project gathers them all into
`<project>_Excluded.csv`.

A pair needs **both** of its chambers for anything that compares paired
with yoked. If one chamber is removed, its partner still appears in the
per-chamber tables, but the whole pair drops out of the difference table,
the difference curve and the breaking point. A single-chamber group drops
out when its one chamber is removed.

### 5.4 The paired − yoked difference

For each pair and each phase (Training, Test), pyflic subtracts the yoked
fly's value from the paired fly's, for each metric:

| Column | Paired − yoked … |
|---|---|
| `dLicksA`, `dLicksB` | sucrose licks, yeast licks |
| `dEventsA`, `dEventsB` | sucrose feeding events, yeast feeding events |
| `dPI`, `dEventPI` | preference index (by licks, by events) |
| `dMedDurationA`, `dMedDurationB` | median feeding-event duration |
| `dPersistA` | sucrose persistence (section 5.7), Test rows only |

Positive means the paired fly did more. **Zero is the null hypothesis:** the
contingency made no difference beyond the light itself. The file is
`paired_yoked_diff.csv`, one row per pair per phase. The reports use the
**Test** phase, where the requirement is rising.

**Without yoked controls there is no difference to take.** A
single-chamber group has no row, so a design made only of them writes
`paired_yoked_diff.csv` with its header and nothing else. The report says
once, where the difference would have been, that *the design has no yoked
controls*, and goes on to the breaking point; the Project Report says the
same, and the Project's Stats text reads *Paired − yoked difference: none;
the design has no yoked controls*. When a recording mixes pairs and single
chambers, the comparison uses the pairs, and the report names the
single-chamber groups it left out.

### 5.5 The cumulative difference curve

This is the difference as it builds up over time: for each pair, paired
minus yoked cumulative sucrose licks, minute by minute from training end.
Per treatment, pyflic draws the mean ± SEM across pairs, over the span of
time that **every** pair covers, so the curve never jumps when one pair's
recording ends. The individual pairs are drawn faintly behind it. A design
without yoked controls has no curve.

### 5.6 The breaking point

*How far did the paired fly keep working?*

pyflic reads the paired fly's Test light events in order from training end.
The fly is taken to have **stopped at its first pause longer than
`pr_break_gap_min`** (120 minutes by default). Pauses are counted from
training end to the first event, between events, and from the last event
to the end of the Test window.

The **breaking point** is the number of light events the fly completed
before that pause: the last ratio it met. Some details:

- **Only lick-backed events count.** A lick-free light event neither counts
  nor ends a pause, because a light with no lick is no evidence that the
  fly was still working.
- **A pause of exactly the gap does not end the count.**
- **`BreakMin`** is the minute, after training end, of the last event
  counted.
- **Censored (`n+`).** If no long pause came before the Test window ended,
  the fly was still working when the recording stopped. Its count is a
  **lower bound**, not a measurement. Tables write it as `12+`, the dot
  plot draws it as an open symbol, and the still-responding curve marks it
  with a tick.
- **The yoked fly has no breaking point.** Its light is its partner's, so
  there is nothing for it to have worked for. A single chamber's paired
  fly has one, like any paired fly.
- **The unit is light events, not licks.** The Requirement each one met
  is reported beside the count (below). On a generation 1 recording the
  firmware's schedule isn't recorded, and `LargestRequirement`, the most
  licks credited to one counted event, is given as a description only.

**The recorded Requirement.** The data state, for every light event, the
**Requirement** it met: the milliseconds of contact the firmware demanded
for it, or, under `Counter: Events`, the feeding bouts. The breaking point is still the
count of lick-backed light events, because a light with no lick is still the
drifting-sensor signature. The Requirement is reported beside it:

| Column | What it is |
|---|---|
| `BreakRequirement` | the Requirement the last counted light event met: the ratio the fly reached, in the firmware's own units |
| `FirmwareEvents` | under `Counter: Events`, the feeding bouts the firmware counted for the counted light events; empty under `Counter: Contact` |
| `ParadigmType` | the Test Paradigm: `ProgressiveRatio` or `FixedRatio` |
| `LargestRequirement` | the same as `BreakRequirement` here |

Both reports show the break requirement beside the breaking point, and the
Project Report adds its median per treatment.

**Fixed ratio.** Under a FixedRatio Test Paradigm every light event needs the
same Requirement: the same contact time, or the same number of feeding bouts. The same rule runs, and what it counts is the
activations the fly earned before its first long pause. `BreakRequirement` is
the constant ratio, no trend is judged, and the reports title the section
**Fixed ratio: activations earned** (the Project Report when every pooled
group ran a fixed ratio). What only a rising requirement explains is left
out for a fixed-ratio group: the estimated requirement increment in
`summary.txt`, the grey estimated-requirement line on *Licks per light
event* and that figure's line "a working progressive ratio climbs". A
recording that runs both shows them for its progressive groups only.

**The end of the Test window.** The Test ends where the ratio Paradigm's
run ends, not at the end of the recording (on a generation 1 recording, at
the end of the recording). A
ClosedLoop interval after it is not Test. Only the first ratio run is the
Test: a later one is noted in `summary.txt`. `pr_test_window_min` caps the
window after that, and `TestMinutes` is each group's.

**A DFM that restarts ends the Test window.** A DFM can restart on its own
in the middle of a recording, from a brief loss of power for example: a
few seconds of records are lost, a few rows run under no instruction, and
then the firmware starts every well's Paradigm afresh, the ratio from its
first Requirement. What follows is no longer the same Test. So when a DFM
restarts inside a group's Test window:

- the window ends at the last row before the restart, and the breaking
  point is judged up to there;
- a fly still responding at the restart is **censored** (`n+`), because it
  might have gone on;
- `TestMinutes` is the shortened window, and sucrose persistence is
  judged over it too;
- the group's light QC `Notes`, `summary.txt` beside the breaking point and
  the report under the breaking point all say *the DFM restarted at minute
  m, d min after training end: the Test window ends at the restart,
  because the ratio starts again after it, so a breaking point still
  running there is censored*.

`pr_breaking_point.csv` gains no column for it. The integrity report and
the general light QC (*DFM restarted*) name every restart too.

The file is `pr_breaking_point.csv`, one row per group, with `BreakingPoint`,
`BreakMin`, `Censored`, `TestMinutes`, `LargestRequirement`,
`BreakRequirement`, `FirmwareEvents`, `ParadigmType` and the group's light QC
result. A generation-2 group has no row: it has no Test. `pr_light_events.csv`
lists every Test light event, with a `Counted` column showing which ones make
up the breaking point. Each event also carries the `Requirement` it met,
the `FirmwareEvents` since the previous one and its `Phase` (not on a
generation 1 recording). Use it to check any group by hand.

### 5.7 Sucrose persistence

The same rule applied to feeding, for **every** fly, paired or yoked: the
minutes from training end to the fly's last sucrose feeding event before a
pause longer than `pr_break_gap_min`. It is censored the same way when the
fly was still feeding at the end of the window.

Because the yoked fly of a pair has it too, it has a paired − yoked
difference, `dPersistA`. `dPersistCensored` marks differences where either fly was
censored. Those differences are kept and flagged, not dropped, because in a
recording that ends during a feeding peak, dropping them would remove most
groups.

Persistence is in `PersistA` / `PersistACensored` on every per-chamber
summary row, and in `dPersistA` / `dPersistCensored` in the difference
table.

---

## 6. Reading the figures

### Results figures

**Cumulative difference curve** (`pr_cumulative_diff.png`; paired designs
only). The line is each treatment's mean paired − yoked cumulative sucrose
licks since training end, with the shaded band its SEM, over the time every
pair covers. The faint lines are individual pairs, and the dashed line is
zero. Without yoked controls the figure is empty and its title says why.

- A line climbing above zero means the paired flies worked for the light
  more than their yoked partners fed.
- A line that flattens shows when the difference stopped growing.
- A flat line near zero means no operant effect.
- Check the faint lines: one extreme pair can carry a mean.

**Paired − yoked dot plots** (reports; paired designs only). One point per
pair, for Test-phase `dLicksA`, `dPI` and, when present, `dPersistA`, with
the treatment mean ± SEM and a line at zero. Points mostly above zero
support a paired > yoked effect. Compare how far the treatments sit from
zero, not only from each other.

**Breaking point by treatment** (reports). One point per group, with mean ±
SEM. **Open symbols are censored**: the true value is at least that high.
If one treatment has many open symbols, its mean understates it. Under a
fixed ratio the section is titled *Fixed ratio: activations earned*, and
the points are the activations each fly earned before its first long pause.

**Light behaviour** (reports). One point per paired chamber, by treatment,
with the treatment comparisons; a yoked fly's light is its partner's, so it
is left out. With `Program.txt` there is one
subsection per kind of light the program ran, in this order, each with its
own measures:

- *open loop*, when the program lit wells whatever their fly did (an
  OpenLoop interval, or wells set `On`): lit time and the light feeding
  index, with the *Feeding in light and dark* figure that the Light and dark
  module adds;
- *closed loop*, when a ClosedLoop interval ran (under a type of your own
  that expects it): light events, lit time and the median light event's
  length;
- *fixed length, ratio*: light events (Activations), the last Requirement
  met and, under `Counter: Events`, the firmware's feeding bouts (Firmware
  Events);
- *Acclimation (closed loop)*: the minute each fly's Acclimation ended.

Otherwise one section shows the four measures of before: light events, lit
time, the firmware's feeding bouts and the end of Acclimation. On an older
recording (generation 1 or 2) only lit time is there.

**Still-responding curve** (`pr_still_responding.png`). For each treatment,
the fraction of paired flies that reached each ratio: 1.0 at ratio 0,
stepping down as flies stop. Ticks mark censored flies, which were still
responding when their window ended. The legend gives each treatment's *n*.

- A curve lying to the right of another means that treatment's flies kept
  working to higher ratios.
- This is the figure that treats censoring correctly. When many flies are
  censored, trust it over the dot plot.

**Breaking-point plots** (`breaking_point_dfm<id>.png`, from the Plots
panel's **Breaking-point plots** and in the experiment report). One panel per
chamber, both roles of a pair, showing the licks between successive light
onsets over time since training end.

- Blue points are the events the breaking point counts.
- Grey points came after the break, or after the Test window.
- Hollow red rings are lick-free events, never counted.
- The dashed line marks the break. A censored group has no dashed line, and
  its strip reads `BP n+`.
- Groups removed by the light QC are drawn for reference, with
  "— excluded" in the strip and no break marked.

### Quality-control figures

The per-DFM QC figures below have one panel per group, three to a row, so
a DFM of six single chambers takes two rows.

**Licks per light event** (`pr_light_events_dfm<id>.png`). For each group,
the sucrose licks credited to each Test light event, in order.

- **A working progressive ratio climbs.**
- Hollow red rings are lick-free events; a run of them along the bottom is
  self-triggered light.
- The dashed line is the group's own trend. The grey line is the
  requirement pyflic estimates across the experiment, for reference.
- Under a fixed ratio the requirement does not climb, so there is no grey
  line, and the caption does not ask for a climb.

**Sucrose Well resting level** (`pr_resting_level_dfm<id>.png`). For each
group, the paired sucrose well's resting signal over the whole recording,
against the median of the DFM's other sucrose wells. The rug along the
bottom shows light onsets, and the dashed line is training end.

- A well that creeps up while the rug gets denser is a sensor problem, not
  a motivated fly.
- This figure is usually the clearest explanation of a self-triggered
  failure.

**Training-aligned traces** (`pr_cumulative_licks_dfm<id>.png`). Paired and
yoked cumulative sucrose licks since each group's training end; a single
chamber has the paired trace only. Points mark minutes with the light on;
black rings mark lick-free light events on the paired trace. Use it to see
whether the light kept firing while the paired trace went flat. That
pattern is self-triggering.

Every QC figure's panel title gives the group's treatment and its light QC
result: *light QC ok*, *warning: …*, or *EXCLUDED: …*.

**Light explained by licks** (`qc/opto/opto_light_dfm<id>.png`). The
optogenetic light QC's figure: one row per group, with its wells and its
optogenetic verdict in the strip. The left panel is the group's lit time
per bin, 10 minutes when basic analysis draws it: grey-blue where a lick or
touch at the paired sucrose well explains it, and red where nothing does.
The right panel is orange where the firmware saw contact and pyflic saw
nothing: the light of the light events the firmware switched on with no
lick or touch at their onset. (On a generation 1 or 2 recording, drawn when
`Program.txt` is in `data/`, it is the time pyflic's re-run of the
firmware's test read the well as touched.)

- Red beside orange in the same bins is a drifting well: the light
  followed the sensor.
- Red with no orange is light the firmware had no reason to switch on.
- [Checking your data](qc-user-guide.md), section 6.7, covers the other
  colours.

**Program timeline** (`qc/program/program_timeline_dfm<id>.png`; an older
recording needs `Program.txt`). For one DFM, the
program it ran — a band of intervals coloured by Paradigm, such as the
ClosedLoop opening and then `PR10` — above one lane per well. Each paired
sucrose well's lane shows its Acclimation shaded, its light events as ticks
and its light as a blue strip; in a pair, its yoked partner's lane should
show the same strip and no ticks.

- The shading should end in the first interval: that is the training end.
  A well shaded to the end never finished training.
- Ticks thin out as the requirement rises. A dense comb of ticks while the
  paired fly's licks are flat is self-triggered light.
- A yoked lane with ticks, or a different strip, is broken yoking.
- A red band labelled *restart* marks a DFM restart. A Test window that holds one ends there (section 5.6).
- [Checking your data](qc-user-guide.md), section 6.8, covers the rest.

---

## 7. Statistics and inference

### The unit of analysis

**One observation per chamber group.** Paired and yoked flies are never
entered as separate observations in the main tests. The yoked fly is the
paired fly's control, and pooling them would average an effect with its own
control. In a design without yoked controls each group is one fly, and the
breaking point is the main result.

### The tests

| Question | Test | Where |
|---|---|---|
| **Is the paired fly different from its yoked partner?** (pairs only) | Each treatment's Test-phase differences against zero: the paired t-test (a one-sample t-test on the differences), with the Wilcoxon signed-rank test beside it; in a Project, also a mixed model's intercept | both reports; `<project>_Stats.txt` |
| **Do treatments differ in the paired − yoked difference?** (pairs only) | Welch's t-test for two treatments, Tukey HSD for three or more; in a Project, also a mixed model | both reports; `<project>_Stats.txt` |
| **Do treatments differ in breaking point?** | The same tests on `BreakingPoint`, plus a pairwise **log-rank test** on the ratio reached | both reports; `<project>_Stats.txt` |
| **What fraction of flies reached each ratio?** | Kaplan-Meier estimate (the still-responding curve) | both reports; `pr_still_responding.png` |

The per-chamber pooled tests from the standard two-well analysis are still
reported, as a secondary section.

In a Project with two or more members, the **mixed model** treats each
experiment, and each DFM within it, as a source of variation of its own
(DFM nested within Experiment). It is reported beside the pooled p-value.
**When they disagree, prefer the mixed model**: it doesn't treat groups on
the same device, or in the same recording, as independent.

### What a result lets you say

- **A difference above zero in the paired flies** (for example `dLicksA`
  in the Test phase) supports the light acting as a reinforcer. The paired
  fly fed more when feeding produced the light than a fly given the same
  light regardless. The yoked design controls for the light's direct
  effects on feeding, but only when both flies of a pair were treated
  identically in every other way. A design without yoked controls cannot
  separate the two: it measures how hard the flies worked, not whether the
  contingency itself mattered.
- **A higher breaking point** means the paired flies kept working to
  higher requirements, a measure of how much effort the light was worth.
  There is no yoked comparison for it. Compare treatments with each other,
  and keep the dot plot and the still-responding curve side by side.
- **A difference between treatments in the paired − yoked difference**
  means the treatment changed how reinforcing the light was, not merely
  how much the flies fed.

### Cautions before you conclude

1. **Censored breaking points.** The t-test, Tukey and the mixed model
   enter a censored count as if it were the true value, which **biases
   that treatment's mean downward**. The log-rank test and the
   still-responding curve treat censoring correctly. Count the censored
   groups per treatment (open symbols, `n+`). When censoring is common or
   uneven between treatments, base the conclusion on the log-rank test and
   the curve, and consider a longer recording.
2. **The gap setting.** Check the sensitivity table (60 / 120 / 240
   minutes). A conclusion that holds at only one gap is weak and should be
   reported as such.
3. **Unequal Test windows.** Groups that trained late had less Test time,
   and a DFM restart cuts a group's window short. If `TestMinutes` varies
   widely, set `pr_test_window_min` so every group is judged over the same
   span.
4. **Independence in a single experiment.** In one experiment's report, the
   groups on a DFM are treated as independent although they share a
   device. The mixed model that accounts for this runs only across two or
   more members, so replicate across recordings and pool them in a
   Project.
5. **Multiple comparisons.** The reports test several metrics
   (`dLicksA`, `dLicksB`, `dEventsA`, `dPI`, `dMedDurationA`, `dPersistA`,
   and `dEventsB` in a Project) without correction. Tukey HSD corrects
   within one metric; the log-rank tests are **not** corrected between
   treatment pairs. Decide your primary measure before you look, and treat
   the others as supporting.
6. **Exclusions can bias a comparison.** If the light QC or training
   removes more groups from one treatment than another, the remaining
   groups may not be comparable. `<project>_Excluded.csv` and the report's
   exclusion table give the counts. Report them per treatment.
7. **Lick detection matters.** Whether a light event counts depends on
   pyflic detecting at least one lick. Detection parameters that miss
   brief licks turn real responses into lick-free events. That lowers
   breaking points and can trigger the self-triggered check. Keep detection
   parameters identical across the Project, and look at the QC figures
   when many lick-free events appear.
8. **Sucrose persistence differences can be differences of lower bounds.**
   Rows with `dPersistCensored = true` are kept. If many are censored,
   interpret `dPersistA` cautiously.

### What to report

- The number of chamber groups per treatment, before and after exclusion,
  with the reasons.
- `pr_break_gap_min`, `pr_test_window_min` (if used), and the detection
  parameters.
- The number of censored groups per treatment.
- For the breaking point: the still-responding curve and the log-rank test,
  with the mean comparison as secondary.
- For the paired − yoked difference, in a paired design: the test against
  zero within each treatment, and the between-treatment comparison.
- Whether the design had yoked controls, and any DFM restart that ended a
  Test window.
- Whether the conclusion survives the breaking-point sensitivity table.
- For pooled data: the mixed-model p-values.

---

## 8. Quick troubleshooting

| You see | Likely cause | What to do |
|---|---|---|
| "firmware 4.0.0 recording without Program.txt" | the program was not copied into `data/` | copy the `Program.txt` PyMCU exported for this recording and run again; without it a fixed ratio is taken as progressive |
| Loading fails: "the Opto Program runs ClosedLoop …, and experiment_type 'ProgressiveRatio' expects ProgressiveRatio, FixedRatio" | the program trains under a ClosedLoop interval, which the shipped type does not take (section 2) | build a type of your own from *Progressive Ratio* in the Experiment Type Editor, with ClosedLoop added to its light, and choose it in the Project design |
| The *Paired-Yoked Difference* box on the Analyze panel is greyed | every chamber group is a solo group, so there is no difference to take | nothing, if the rig had no yoked partners; otherwise state `paired_chambers` or check the program's linkage |
| The Analyze panel has no *Paired and yoked comparison* | a type with the Progressive Ratio module keeps its own paired − yoked difference instead | use *Paired-Yoked Difference* |
| The Progressive Ratio groups are missing from the Hub's panels | the member's Experiment Type does not include the Progressive Ratio module | choose *Progressive Ratio*, or a type of your own with the module, in the Project design |
| "paired Sucrose Well W*n* never reached Phase M under a ratio Paradigm" | the fly never finished Acclimation, or the ratio Paradigm never ran | check the Program timeline; the group is removed like any group that never finished training |
| "Training was empty" | the ratio Paradigm had no Acclimation and nothing ran before it | nothing: Training ended at the start, and every light event is Test |
| "Paradigm restarted at Program Interval *k*" | a later interval named a different Paradigm, so the firmware started the well afresh | check the program was the one you meant; only the first ratio run is the Test |
| A group failed for **broken yoking** | the yoked chamber's light differed from the paired chamber's, or a yoked well counted light events of its own | check the lid's wiring and the program's linkage; the yoked fly did not get its partner's light |
| "the firmware counted rising Events per light event; pyflic's licks did not follow" | the fly worked to the schedule, and pyflic's detection missed some of its feeding | look at the paired sucrose well's baselined trace and the detection parameters |
| A group failed for **self-triggered light** | the sucrose well's resting level drifted, so the light fired on its own | look at *Sucrose Well resting level*; the group is correctly excluded |
| A group failed for **implausible training** | training was completed by the sensor, not the fly | check the well and the fly; the group is correctly excluded |
| A group failed for **requirement series wrong** | a Test light event met a Requirement the Test Paradigm's series does not give; the firmware keeps its series exactly | check that `Program.txt` is the one exported with this recording; the Requirement each light event met is in `pr_light_events.csv` |
| "the DFM restarted at minute *m* … the Test window ends at the restart" | the DFM restarted during the group's Test, and the ratio started again after it | nothing to fix: the breaking point is judged up to the restart, censored if the fly was still responding. Check the DFM's power and connection before the next recording |
| Almost every breaking point is censored (`n+`) | the Test window is short compared with `pr_break_gap_min`, or a DFM restart cut it short | record longer, or reconsider the gap; don't shorten the gap to get a result |
| "the design has no yoked controls" | every chamber group is a single chamber | nothing: there is no paired − yoked difference to take; the breaking point is the result. If the rig had yoked partners, state `paired_chambers` or check the program's linkage |
| A group is missing from the difference table | one of its two chambers was removed, or it is a single chamber | see `removed_chambers.csv` |
| The Project Report says a member has no breaking point table | that member was analysed with an older pyflic | re-run its analysis |
| Loading fails on `paired_chambers` | missing, two chambers from one pair, or a pair left out | name exactly one chamber from each of 1–2, 3–4, 5–6; or, for a design without yoked controls, write `chamber_groups: single`; or copy `Program.txt` into `data/` so its linkage gives the groups |
| "'paired_chambers' cannot be stated with 'chamber_groups: single'" | both keys on one DFM | keep the one that describes the rig |
| "chamber groups taken from Program.txt's linkage" in `summary.txt` | neither key was stated, and the program's linkage gave the groups | check that the groups it lists are the rig's |
| "chamber_groups single disagrees with Program.txt" | the config says single chambers, the program links chambers into pairs | the config's are used; fix whichever is wrong |
| Loading fails on treatments | the two chambers of a pair have different treatments | give both chambers the same treatment |
| A linkage group failed for **unexplained light** | most of its lit time had no lick or touch behind it | open *Light explained by licks*: orange beside the red is a drifting well, no orange is a light the firmware had no reason to switch on |
| "holds more than one Program.txt" | two program files in `data/` | keep only the one the MCU exported for this recording |
| `summary.txt` says `paired_chambers` disagrees with `Program.txt` | the config names a different paired chamber than the program's trigger well | check which chamber the program lit for; fix the config or the program |
| Older recording: every group removed, with "training never completed" | a generation 1 file with no light data: no training flags | check the DFM files are version 3 with `OptoCol1` |
| Older recording: "no training flag on any well of group *g*" | same as above, or the run never entered training | as above |
| "no training information: recorded with DFM firmware older than 4.0.0" | PyMCU 4 recorded a DFM with older firmware: there is no record of Training | nothing is removed; each group is one `Recording` phase with no breaking point. Update the DFM's firmware for the next recording |
