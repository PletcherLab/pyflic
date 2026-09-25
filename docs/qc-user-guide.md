# Checking your data: quality control in pyflic

A guide for researchers analysing FLIC recordings with pyflic. It explains
every quality-control (QC) check pyflic runs, the files and figures each one
writes, how to read them, and how to decide which chambers to leave out.

Sections 1 to 5 apply to every experiment. Section 6 adds the light QC of
optogenetic experiments, section 7 the checks of a Progressive Ratio
experiment, and section 8 what a Project does with its members' QC.
Section 9 is a checklist to work through for each recording.

You do not need to read code to use it. For installing and starting pyflic,
see [Getting Started](GETTING_STARTED.md). For the Progressive Ratio assay
itself, see [Analysing a Progressive Ratio assay](progressive-ratio-user-guide.md).
The in-app help (`pyflic help`, or **F1** in any window) covers the same
ground as a concise reference. Its reading path *Checking your data* starts
from the QC Viewer.

---

## 1. What QC is for

Every number pyflic reports rests on two things: the signal each well
recorded, and the licks pyflic detected in it. Both can go wrong in ways
that still produce plausible results. A well whose background drifts,
thresholds that miss brief licks, two wells that pick up each other's
signal: none of these stops an analysis, and each one changes its numbers.

QC asks five questions:

| Question | Where to look | Section |
|---|---|---|
| Did the recording run cleanly? | the integrity report and the data breaks | 3.1, 3.2 |
| Does detection agree with what you would call by eye? | the signal plots and the QC Viewer | 3.3, 4 |
| Are the two wells of a chamber independent? | simultaneous feeding and bleeding (two-well) | 3.4, 3.5 |
| Is each chamber a usable fly? | the exclusion cutoffs and your own inspection | 5 |
| Was the light where the licks were? | the optogenetic light QC | 6 |

Three rules run through everything:

- **Do QC before statistics.** Once you have seen a result, every exclusion
  looks like a choice made for it.
- **Most checks flag; you decide.** Only the design's exclusion cutoffs
  remove chambers on their own. Everything else is information for your
  judgement.
- **Apply one rule to every chamber, whatever its treatment.** A rule
  applied to one treatment only biases the comparison.

---

## 2. Running QC

### Where QC runs

| How | What it writes |
|---|---|
| **Hub → QC → QC reports** | the **QC bundle**, into the member's `qc/` folder: the integrity report, the data breaks, the three signal plots, the two-well crosstalk tables, and the optogenetic light QC when the member is optogenetic |
| **Hub → Analyze → Basic analysis** | auto-removal, `summary.txt` and the feeding summaries. It skips the QC bundle, which is the slow part, but it writes the optogenetic light QC and a Progressive Ratio experiment's own light QC |
| **QC Viewer** | loading a member on its **Load** tab writes the QC bundle. Opened from the Hub, the viewer starts on the member the Hub already loaded, and **Run QC** in its top bar writes the bundle |
| **PDF report** | nothing into `qc/`. The report computes its integrity and crosstalk tables from the loaded recording, never from an old `qc/` folder |
| **Scripts** | `run_qc` writes the QC bundle. `basic_analysis` skips it, as on the Hub |

From Python, `execute_basic_analysis()` writes the QC bundle too, unless you
pass `skip_qc=True`.

The QC bundle describes the detection parameters in force when it was
written. **Write it again after you change a parameter**, or the baselined
plot will show thresholds you no longer use.

### What goes where

```
my_experiment/
  remove_chambers.csv        exclusions you declare (section 5)
  qc/
    integrity/               DFM1_integrity_report.csv and .txt
    data_breaks/             DFM1_data_breaks.csv, only when there are breaks
    raw_signal/              DFM1_raw.png
    baselined/               DFM1_baselined.png
    cumulative_licks/        DFM1_cumulative_licks.png
    simultaneous_feeding/    DFM1_simultaneous_feeding_matrix.csv   (two-well)
    bleeding/                DFM1_bleeding_matrix.csv
                             DFM1_bleeding_alldata.csv               (two-well)
    opto/                    four tables and opto_light_dfm1.png     (optogenetic)
  analysis/
    removed_chambers.csv     what auto-removal took out, and why
    summary.txt              the run in text, with its QC sections
    experiment_report.pdf    the report, which opens with Quality control
    pr_light_qc.csv, ...     Progressive Ratio (section 7)
```

Every file outside `qc/opto/` has one copy per DFM, named `DFM<n>_...`.

`summary.txt` always lists the chambers excluded and the chambers
auto-removal took out, and it has a light QC section for an optogenetic or
Progressive Ratio experiment. Its *QC summary* of integrity, data
breaks and crosstalk appears only when the QC bundle ran as part of basic
analysis, as it does from Python by default. The Hub's Basic analysis
leaves that section out; the report and `qc/` hold the same information.

---

## 3. The checks every experiment gets

### 3.1 Data integrity

For each DFM, the integrity report states:

| Item | What it is |
|---|---|
| Rows | the number of samples recorded |
| Start, end and elapsed time | the first and last sample's clock time, and the minutes between them |
| Elapsed from the `Minutes` column | the same span, from the file's own minute count |
| Error column | how many samples carry a non-zero firmware error code, and how many of each kind: I2C, ID, PacketType, DMA_TX, DMA_RX, PacketSize, AIInterrupt and OInterrupt |
| Index check | whether the device's sample index rises by exactly 1 from each row to the next |

**Where to look.** `qc/integrity/DFM<n>_integrity_report.txt` holds the full
text and the `.csv` beside it one row of numbers. The QC Viewer's
**Integrity** tab shows both. The report's *Data integrity* table gives each
DFM a verdict: **ok** when it has no data breaks, no error flags and a
continuous index, and **warning** otherwise. The report's cover repeats the
verdict under *At a glance*.

**Reading it.**

- The two elapsed times should agree with each other and with how long you
  recorded. A DFM much shorter than the others stopped early.
- An index that does not rise by 1 means samples are missing or repeated.
- The error names are the firmware's. pyflic counts them but does not
  interpret them.
- **Nothing here removes a chamber.** A warning is a reason to look at the
  signal plots around the affected time, and to note it if that DFM's
  results stand out.

### 3.2 Data breaks

A **data break** is a gap between two consecutive samples longer than four
sampling intervals: more than 0.8 seconds at the default 5 samples per
second. `qc/data_breaks/DFM<n>_data_breaks.csv` lists each break with its
length in seconds (`Interval`) and the sample that ended it. The file exists
only for a DFM that has breaks. The report's integrity table counts them.

- **A few short breaks change little.** Totals lose a second or two.
- **A long break removes that stretch from every chamber on the DFM.**
  Compare the DFM's recording length with the others', and treat
  time-binned results across the gap with care.
- **Breaks on one DFM only** point at that device or its connection, not at
  the flies.

### 3.3 The signal plots

Three figures per DFM, one panel per well. They are diagnostics, not
figures for a paper.

**Raw signal** (`qc/raw_signal/DFM<n>_raw.png`). Every well's signal as
recorded.

- A well that stays flat, with no deflections at all, is an empty position
  or a dead well.
- A background that creeps upward is drift. The baseline removes slow
  drift, as the next figure shows, but a well that drifts a lot is worth a
  second look.
- A sudden step in the background is a disturbance: a moved plate, a
  refilled well, a bubble.

**Baselined signal** (`qc/baselined/DFM<n>_baselined.png`). Every well after
pyflic subtracts a running median of the signal, its **baseline**, with two
red dashed lines: `feeding_threshold` (the upper line) and `feeding_minimum`
(the lower one). A feeding bout must reach the upper line, and its licks are
the samples above the lower one.

- **This is the most useful figure in the set.** It shows whether the
  thresholds sit in a sensible place for your data.
- The trace should sit near zero between bouts. A trace that dips below
  zero around long bouts, or long bouts that look flattened, means the
  baseline is being pulled up by the feeding itself; see *The raw signal and
  the baseline* in the help.
- Bouts that barely cross the upper line, or noise that often crosses it,
  both say the threshold needs a look. Try values in the QC Viewer
  (section 4) before changing the configuration.

**Cumulative licks** (`qc/cumulative_licks/DFM<n>_cumulative_licks.png`).
Each well's running lick total over time, drawn as the fourth root of the
total whatever `transform_licks` says. A two-well experiment has one row per
chamber, with well A on the left and well B on the right.

- A steady slope is steady feeding, a plateau is a fly that stopped, and a
  step is a burst.
- A curve that never leaves zero is a well with no licks. With a fly in the
  chamber, that fly did not feed there, or the well did not work.
- One chamber far steeper than its neighbours is either a heavy feeder or a
  signal problem. The baselined plot tells you which.

### 3.4 Simultaneous feeding (two-well)

A fly cannot drink from both wells of its chamber at once. Samples that
register a lick on both wells are therefore **crosstalk** between the wells,
or a fly bridging them. `qc/simultaneous_feeding/DFM<n>_simultaneous_feeding_matrix.csv`
has one row per chamber:

| Column | Meaning |
|---|---|
| `Licks1`, `Licks2` | lick samples at the chamber's lower- and higher-numbered well |
| `Both` | lick samples on both wells at once |
| `MaxMinSignalAtBoth` | across those samples, the largest value of the weaker well's signal |
| `HigherInCol1AtBoth` | how many of them had the lower-numbered well's signal higher |

The QC Viewer's **Sim. Feeding** tab shows the table, and the report's
*Simultaneous feeding* table adds each chamber's treatment and `Both` as a
percentage of its lick samples.

**The dual-feeding correction.** `correct_for_dual_feeding`, on by default in
two-well experiments, gives each such sample to the stronger well and
detects the licks again. The table is taken after the correction, so with
the correction on its `Both` column reads zero, and the lick counts are the
ones each well kept. To see how much crosstalk the correction removed, set
`correct_for_dual_feeding: false`, write the QC bundle, read the table, and
set it back. State in your methods whether the correction was on.

Crosstalk left uncorrected pushes a chamber's preference index toward zero,
because the weaker well is credited with licks that belong to the stronger
one. The correction repairs most of that, but a chamber where crosstalk is
common deserves a look at its baselined trace.

### 3.5 Bleeding between wells (two-well)

**Bleeding** is one well's signal appearing in another well. For every well
in turn, pyflic takes the samples where that well's baselined signal is
above 50 and averages every well's baselined signal over them. The result is a 12 × 12
table in `qc/bleeding/DFM<n>_bleeding_matrix.csv`: a row per signalling well
(`W1Sig` …) and a column per responding well (`W1Resp` …).
`DFM<n>_bleeding_alldata.csv` is each well's mean baselined signal over the
whole recording, for comparison.

- The diagonal is each well responding to itself, which is large by
  construction. Ignore it.
- **Off the diagonal, every value should be near zero.** A well that rises
  while another is being fed from is picking up its signal.
- **The report's check.** For each DFM, the report's *Bleeding between wells*
  table names the largest response off the diagonal. It is a **warning** when
  that response is above `feeding_threshold`, because such a response would
  register as licks that never happened.

A bleeding warning is a reason to open the baselined plot of the responding
well and look at the moments the signalling well was fed from. Nothing is
removed automatically.

---

## 4. The QC Viewer

The QC Viewer is where you check detection against the trace. Open it from
the Hub's **QC** panel with **Open QC Viewer**, or from a terminal:

```
pyflic qc "path/to/my_experiment"
```

| Tab | Shows |
|---|---|
| **Load** | load an experiment and choose the time range; the log of every step |
| **Feeding Summary** | the per-chamber summary table, with a checkbox per chamber for exclusion |
| **DFM *n*** | one tab per device: **Integrity**, **Sim. Feeding** and **Bleeding** (two-well), **Raw Signal**, **Baselined** and **Cumulative Licks**, with an **Exclude Wells** panel beside them |
| **Opto Light QC** | optogenetic experiments only (section 6.6) |
| **Params** | the detection parameters, with a live recompute |

### What to look for

The DFM tabs show the saved QC plots of section 3.3. They do not mark
individual events. Judge detection by setting a well's **Baselined** trace,
with its threshold lines, beside its **Cumulative Licks** curve, which
climbs only where pyflic detected licks.

- **Does the baseline sit where it should?** Between bouts the baselined
  trace should sit near zero.
- **Does detection match the trace?** Deflections that clear the upper line
  should appear as climbs in the cumulative curve. Climbs where the trace is
  flat mean the thresholds are too low. Clear deflections with no climb mean
  they are too high.
- **Are bouts split or merged?** The plots cannot show this, but the
  Feeding Summary can. Many short events per chamber suggest the link gap
  (`feeding_event_link_gap`) is too small, and a few very long ones that it
  is too large. This is the most common cause of surprising duration and
  event-count results.
- **Are some wells dead?** A well with no signal is an empty position or a
  hardware fault. Exclude it rather than let a zero into the averages.

### Trying parameters

The **Params** tab changes a detection parameter and recomputes the feeding
summary, without editing your configuration. Change one value, recompute,
look at a chamber you can read by eye, and repeat. When detection matches
your reading of the trace, write the value into the configuration. The
Params tab never saves it for you.

A recompute updates the Feeding Summary and the Opto Light QC tabs. The DFM
tabs show the images saved in `qc/`, which do not change. **Run QC** after a
recompute writes the QC bundle under the parameters you are trying, not the
ones in your configuration. Write it again from the Hub once you have
settled on a value.

**Before you publish, sweep the parameter.** The Script Editor's *Parameter
sensitivity sweep* (`param_sensitivity`) re-runs the analysis over a list of
values and writes `analysis/param_sensitivity_<parameter>.csv`, with the
licks, events and median duration of each treatment at each value. An effect
that appears at only one link gap or one threshold is not a robust effect.

### Marking exclusions

On the **Feeding Summary** tab, tick the chambers to exclude, or use a DFM
tab's **Exclude Wells** panel, which ticks the chamber a well belongs to.
**Mark All Excluded** and **Clear All** set every box at once. **Auto Filter**
ticks the chambers the design's cutoffs would remove, and **View Criteria**
shows the rules it used and why each chamber was chosen. **Save Exclusions…**
writes your selection to a named group in the member's `remove_chambers.csv`.

The ticks are only a selection until you save them. Review Auto Filter's
selection before saving: automatic criteria are a starting point, not a
verdict.

---

## 5. Exclusions

An **excluded chamber** leaves every result: every summary, plot and
statistic. Chambers leave in two ways.

### By hand

Each member declares its own exclusions in a `remove_chambers.csv` at its
root:

```csv
group,dfm_id,chamber,note
general,1,3,low lick count
general,2,5,fly escaped
```

The `group` column lets one member keep several exclusion sets. The active
one is `general`, unless a Project's design names another with
`exclusion_group:`. The QC Viewer's **Save Exclusions…** writes this file
for you.

To declare many at once, put one exclusion sheet at the project or batch
folder and apply it; pyflic writes its rows into each member's file. See
*Excluding chambers in bulk* in the help.

### Automatically, by the design's cutoffs

Basic analysis applies the design's `constants:` cutoffs once, before it
writes any summary. The defaults depend on the experiment type:

| Setting | Removes a chamber when | Custom | Hedonic, Progressive Ratio |
|---|---|---|---|
| (always) | a well produced no usable lick count | yes | yes |
| `min_untransformed_licks_cutoff` | a well has fewer licks than this over the recording, counted before any transform | off | 20 |
| `max_med_duration_cutoff` | the median feeding-event duration at either well is above this, a sign of a stuck or bleeding signal | not applied | 13.0 |
| `max_events_cutoff` | either well has more feeding events than this | not applied | 150000 |
| `exclude_failed_opto_chambers` | its linkage group failed the optogenetic light QC (section 6) | off | off |

A Progressive Ratio experiment adds its own group-level rules (section 7).
Set the cutoffs in the Project's design so that every member is judged by
the same rule.

### Where exclusions are recorded

| File | Holds |
|---|---|
| `remove_chambers.csv` | the exclusions you declared |
| `analysis/removed_chambers.csv` | every automatic removal and its reason |
| `analysis/summary.txt` | both lists |
| the report's *Excluded chambers* table | both, marked *by hand* or *automatic*; the cover counts them |
| `<project>_Excluded.csv` | every member's, in a Project (section 8) |

**Declaring does not change results already on disk.** When a member's
`remove_chambers.csv` is newer than its saved results, the Hub's Analyzed
column reads **re-run needed**. Run the member's basic analysis again.

---

## 6. Optogenetic experiments: the light QC

When a DFM drives an Optolid, the firmware decides every millisecond whether
each well's light is on. It reads the well's own signal, compares it with a
threshold from the program the MCU loaded, and keeps the light on for a set
**decay** after contact ends. pyflic sees only the outcome, the light state
recorded in `OptoCol1`, and counts licks afterwards from the baselined
signal.

Usually the two agree. When a well's signal drifts upward, however, the
firmware sees a well that is constantly touched and keeps the light on,
while pyflic sees a flat trace and no licks. The light then describes the
sensor, not the fly. The **optogenetic light QC** measures that disagreement
over the whole recording, for any experiment type.

A short tail of light after the last lick is normal: that is the decay. Long
stretches of light with no lick are not.

### 6.1 Turning it on

One setting, `optogenetics`, in the design's `global:` block or the
experiment's own:

| Value | Effect |
|---|---|
| `auto` (the default) | runs on every DFM that has a section in `Program.txt` or any light on in its data |
| `true` (*yes* in the editors) | runs on every DFM. A DFM with no light on for the whole recording **fails**: an unplugged lid, or a program that did not load |
| `false` (*no*) | off |

A DFM's own entry may override it, because whether its lid was lit is a fact
about the recording:

```yaml
dfms:
- id: 3
  optogenetics: false     # this DFM ran without a lid
```

In the Config Editor the setting is on the Experiment tab, and each DFM tab
has its own *Optogenetics* picker. In the Project Design dialog it is a
design setting.

### 6.2 Program.txt

The MCU writes the program it ran back out as `Program.txt`. **Copy that file
into the member's `data/` folder**, beside the DFM files. Use the exported
file, the one with `***DFM 1***` sections, not the program you wrote for the
MCU.

From it pyflic learns, for each DFM and each interval of the program:

- **each well's threshold.** A positive threshold makes the well a
  **trigger well**. `0` keeps its light on regardless (open loop), and `-1`
  means it never triggers;
- **the linkage**: wells that share a linkage number form one **linkage
  group**, lit together whenever any of them triggers;
- **the light's settings**: frequency, pulse width, **decay**, delay and
  maximum time on, and from them the **paradigm** (closed loop, closed loop
  with a maximum time on, fixed interval, progressive ratio or non-feeding
  activation);
- **the schedule**: the program type and the interval timestamps, which lay
  the intervals out over the run.

**The file is optional, but the check is weaker without it.** pyflic then
infers the linkage from wells lit together, treats every lit well as a
trigger well, and assumes a decay of `opto_default_decay_ms`. Unexplained
light can then warn but never fail, because without the program an
open-loop schedule looks exactly like a stuck light. `summary.txt` and the
report say so.

If the file cannot be read in part, the lines pyflic skipped are listed in
`summary.txt`, and a DFM whose section cannot be read is judged as if it had
no program. Two `Program.txt` files in `data/` stop the load: keep only the
one the MCU exported for this recording.

### 6.3 What it checks

The unit of judgement is the **linkage group**, because its light is one
circuit. Every verdict passes to each chamber the group touches.

**Explained light.** A lit sample is **explained** when one of its group's
trigger wells has a feeding lick, or a tasting-level touch, from the decay
before it until just after it. Touches count because the firmware decides on
contact at millisecond resolution, and a brief touch it acts on often reads
below pyflic's feeding threshold. Two samples of slack each side
(`opto_decay_tolerance_samples`) absorb the difference in timing.

| Check | What it means | Verdict |
|---|---|---|
| **Unexplained light** | at least 30% of the group's lit time had no lick or touch behind it | **fails** the group |
| **Partly unexplained light** | at least 10% unexplained | warning |
| **Light while off** | lit while every well of the group had threshold -1 | **fails** the group |
| **No light recorded** | `optogenetics: true`, and no light on the DFM at all | **fails** the DFM |
| **Unlit feeding** | in a closed-loop interval, at least half of the feeding bouts at its trigger wells never lit the group (at least 5 bouts) | warning |
| **No light events** | feeding at its trigger wells, and never lit | warning |
| **Open loop not lit** | lit for less than 95% of an open-loop interval | warning |
| **Program mismatch** | the data's frequency, pulse width or dark setting differ from the program, or the data start more than a minute from its start time | warning |
| **No program section** | `Program.txt` has no usable section for this DFM | warning |

Neither unexplained-light check is judged until a group has at least 30
seconds of unexplained light (`opto_unexplained_min_sec`), so a few stray
samples in a group lit for seconds are not called a fault.

Some light is not judged at all:

- light at the start of an interval, for one decay, which belongs to the
  interval before it;
- light outside the program's schedule;
- **open-loop** light and **non-feeding activation**, where the light is not
  meant to follow feeding. Open-loop intervals get their own check instead.

### 6.4 The emulated trigger: drift or hardware?

With a program, pyflic also re-runs the firmware's own test on the recorded
signal. The firmware subtracts the mean of the run's first ten seconds and
compares each sample with the interval's threshold. Wherever this
**emulated trigger** reads a trigger well above threshold while pyflic sees
no lick or touch, the firmware was seeing contact that pyflic's running
baseline removed. That is the signature of a drifting baseline.

`ContactWithoutActivitySec` is how long that lasted. Each flagged group gets
a **likely cause**:

- **a drifting baseline or sustained contact**, when most of its unexplained
  light fell during such contact. The light followed the sensor.
- **hardware, linkage or program mismatch**, otherwise: light the firmware
  had no reason to switch on. Check the lid, the linkage and the program.

The emulation is marked approximate when the data begin more than two
seconds after the program started, because the firmware's first ten seconds
were not recorded.

### 6.5 Settings

The thresholds are design `constants:`, shared by every experiment type,
with fields in the Project Design dialog and the Config Editor. Change one
only for a reason you can state.

| Setting | Default | Meaning |
|---|---|---|
| `exclude_failed_opto_chambers` | false | remove every chamber of a failed linkage group; off, they are flagged and kept |
| `opto_unexplained_warn_fraction` | 0.10 | unexplained share of lit time that warns |
| `opto_unexplained_fail_fraction` | 0.30 | unexplained share that fails |
| `opto_unexplained_min_sec` | 30 | seconds of unexplained light needed before either is judged |
| `opto_default_decay_ms` | 1000 | the decay assumed without `Program.txt` |
| `opto_decay_tolerance_samples` | 2 | samples of slack around the decay |
| `opto_unlit_feeding_fraction` | 0.50 | unlit share of closed-loop feeding bouts that warns |
| `opto_open_loop_min_lit_fraction` | 0.95 | lit share an open-loop interval needs |

### 6.6 Where to look

Basic analysis and **QC reports** both write the light QC into `qc/opto/`:

| File | One row per | Holds |
|---|---|---|
| `opto_light_qc.csv` | linkage group | its wells, trigger wells, chambers and treatment; lit and unexplained time; the onset; the emulated trigger's contact; `LikelyCause`, `Flags`, `Verdict`, `Excluded` |
| `opto_light_intervals.csv` | group and scheduled interval | the interval's paradigm, decay and trigger wells; lit, unexplained and contact time; feeding bouts, and those never lit |
| `opto_light_events.csv` | light event | onset, end and duration; unexplained seconds; `OverrunSec`, the light's time past the last lick plus the decay |
| `opto_program.csv` | program interval | the program as pyflic read it, with the raw values the MCU echoed beside their decoding |
| `opto_light_dfm<n>.png` | DFM | the *Light explained by licks* figure |

Every chamber's rows in the feeding summaries carry `OptoLightQC`, its
group's flags (empty when the group is ok), and `OptoUnexplainedFraction`.
`summary.txt` has an *Optogenetic light QC* section, and the report has one
in Quality control, with the program, the verdicts and the figure.

**On the Hub**, the QC panel shows an **Optogenetics** group whenever the
loaded member is optogenetic: **Opto light QC table** writes the tables and
logs every flagged group, and **Light explained by licks (QC)** draws the
figure.

**In the QC Viewer**, the **Opto Light QC** tab lists every linkage group with
its lit time, unexplained share, onset, emulated contact and verdict, and
opens on the worst group. Selecting a group shows why it got its verdict,
its intervals, light events and program, and its panels of the figure. The
tab is computed from the loaded experiment, so a **Params** recompute
updates its verdicts at once. That is worth seeing before you settle on a
threshold: one that loses real licks makes healthy light look unexplained.

### 6.7 Reading *Light explained by licks*

One row per linkage group that was ever lit or touched, or was flagged, with
the group's wells and verdict in each panel's strip. Basic analysis and the
QC Viewer draw it in 10-minute bins; the Hub's button uses the bin size on
its Analyze panel.

- **The left panel** is the group's lit time per bin, stacked: **grey-blue**
  where a trigger-well lick or touch explains it, **red** where nothing
  does, **purple** where the group was lit while every threshold was -1, and
  **light grey** where the light is not judged by licks (open loop,
  non-feeding activation).
- **The right panel**, when `Program.txt` allows the emulated trigger, is
  **orange**: the time the firmware would have read a trigger well above
  threshold while pyflic saw no lick or touch.

How to read a row:

- **Mostly grey-blue**: the light followed the fly. A little red beside it
  is light that outlasted the last lick or touch pyflic saw by more than the
  decay. Small amounts are normal, which is why nothing is judged below 30
  seconds or the warning fraction.
- **Red beside orange** in the same bins: a drifting well. The firmware saw
  contact, pyflic did not, and the light is not evidence of feeding.
- **Red with no orange**: light the firmware had no reason to switch on.
  Check the lid, the linkage and the program.
- **Purple** anywhere: the program said the light was off. That is a
  hardware or program problem, whatever the licks did.

### 6.8 Deciding what to exclude

**A failed group is flagged, not excluded, by default.** A faulty light does
not make the feeding record wrong. It makes the light-on numbers wrong, and
whether that matters depends on your question. Three ways to act on it:

- **Keep the group.** Its flags stay on its chambers' rows, and the Project
  reports it as *retained*. Say so when you report the result.
- **Exclude it by hand.** In the QC Viewer's Opto Light QC tab, **Mark
  Selected Group Excluded** or **Mark Failed Groups Excluded** tick its
  chambers, and **Save Exclusions…** records them. The buttons save nothing
  on their own.
- **Exclude every failed group automatically**, by setting
  `exclude_failed_opto_chambers: true` in the design.

`UnexplainedOnsetMin` is the start of the first 30-minute window in which a
flagged group's unexplained share reached the warning fraction, in minutes
from the start of the recording. It is the latest point at which a cutoff
you set by hand could fall, if you analyse only the time before the fault.

Whichever you choose, apply it to every group, whatever its treatment, and
decide before you look at the results.

---

## 7. Progressive Ratio experiments

A Progressive Ratio experiment gets everything above, including the
optogenetic light QC, and a QC of its own, which asks whether the paired fly
earned each light event. It is described in full in the
[Progressive Ratio guide](progressive-ratio-user-guide.md), sections 5.1 to
5.3 and 6. In brief:

| Check | Verdict |
|---|---|
| **Training never completed** | removes both chambers of the group (`require_training_complete`) |
| **Self-triggered light**: five or more Test light events in a row with no sucrose lick since the previous one | **fails** the group |
| **Implausible training**: training completed, with light events, but without one sucrose lick | **fails** the group |
| **No increasing trend** in licks per light event | warning |
| **Resting level rise** or **resting level elevated** at the sucrose well | warning |

A group that fails leaves the analysis by default
(`exclude_failed_pr_groups: true`). A light event is lick-free only when it
has no sucrose lick since the previous one and no lick or touch within its
decay, so a light the fly touched for is never counted against it.

| Output | What it is |
|---|---|
| `analysis/pr_light_qc.csv` | one row per chamber group: flags, verdict, `Excluded`, and `LickFreeRunStartMin` |
| `analysis/pr_light_events.csv` | every Test light event, with its licks and whether it counts |
| `analysis/pr_light_events_dfm<n>.png` | **Licks per light event**: a working progressive ratio climbs |
| `analysis/pr_resting_level_dfm<n>.png` | **Sucrose Well resting level**: a well that creeps up is a sensor problem |
| `analysis/pr_cumulative_licks_dfm<n>.png` | **Training-aligned traces**: light that keeps firing while the paired trace is flat is self-triggering |

The Hub shows the checks' buttons in the same **Optogenetics** group of the
QC panel, beside the optogenetic ones.

**The two light QCs can disagree, and both are right.** The PR check counts
light *events*; the optogenetic check measures lit *time*. On the first real
dataset, one group lit for four hours during training passed the PR check,
because one long light event is not a run of lick-free events. It failed
the optogenetic check, because 93% of those four hours had no lick behind
them. Read both before deciding.

---

## 8. Projects

A Project pools its members, and so it pools their QC. **Project panel →
Build combined analysis** writes, into the Project's `analysis/` folder:

| File | Holds |
|---|---|
| `<project>_Excluded.csv` | every member's exclusions: `manual` (its `remove_chambers.csv`) and `auto` (its `removed_chambers.csv`), with the reason |
| `<project>_LightQC.csv` | every member's light QC tables stacked, with a `Source` column: `PR` for a Progressive Ratio chamber group, `Opto` for an optogenetic linkage group |
| `<project>_Stats.txt` | the pooled statistics, with every flagged group listed before the tables |

Every flagged group is listed in the Stats text and the Project Report's
Quality control section with a **status**:

- **excluded**: its member's auto-removal took its chambers out;
- **retained**: it failed, but its exclusion switch was off, so it **is** in
  the pooled numbers;
- **kept — warning only**: it was only warned.

Pooling needs **every member judged by the same rules**. A Project's design
owns the settings that decide QC: the detection parameters, the exclusion
cutoffs and the light QC thresholds. Every member inherits them. Set them
there, once, and keep a DFM's own settings to what its rig requires, such as
`pi_direction`.

Before pooling, compare the exclusion counts per treatment in
`<project>_Excluded.csv`. If one treatment lost far more chambers than
another, the chambers that remain may not be comparable. Report the counts.

---

## 9. A QC checklist

For each recording, before you look at any result:

1. **Load it and write the QC bundle**: Hub → QC → **QC reports**.
2. **Integrity**: every DFM *ok*, or every warning explained (section 3.1).
3. **Data breaks**: none long enough to matter (section 3.2).
4. **Raw signal**: no dead wells you did not expect, no wells drifting far
   (section 3.3).
5. **Baselined signal**: the thresholds sit sensibly for your data.
6. **QC Viewer**: detected bouts match the trace in a few chambers you can
   read by eye. Settle any parameter change now, and write it into the
   configuration (section 4).
7. **Two-well**: the bleeding check is *ok*, and you know whether the
   dual-feeding correction was on (sections 3.4 and 3.5).
8. **Optogenetic**: `Program.txt` is in `data/`; every linkage group is ok,
   or every flagged group has a decision (section 6).
9. **Progressive Ratio**: the training table and the PR light QC (section 7).
10. **Exclusions**: declare them, run **Basic analysis** again, and check
    `removed_chambers.csv` and the report's *Excluded chambers* table
    (section 5).
11. **Write the report** and read its Quality control section and cover
    before its results.

In a Project, finish with **Build combined analysis** and read the flagged
groups and the exclusion counts per treatment (section 8).

---

## 10. Quick troubleshooting

| You see | Likely cause | What to do |
|---|---|---|
| The QC Viewer's DFM tabs show no integrity files or plots | the QC bundle has not been written | **Run QC** in the viewer, or **QC reports** on the Hub |
| The baselined plot shows thresholds you changed | the QC bundle predates the change | write it again |
| An integrity **warning** | data breaks, error flags or a skipping index | read the integrity text for which; look at the signal plots around that time |
| A well with no deflections at all on the raw plot | an empty position or a dead well | exclude the chamber by hand if a fly was in it |
| The simultaneous-feeding table's `Both` is zero everywhere | the dual-feeding correction is on, and the table is taken after it | see section 3.4 for how to see the crosstalk it removed |
| A **bleeding** warning | one well's signal appears in another above `feeding_threshold` | look at the responding well's baselined trace while the signalling well is fed from |
| Many chambers removed for too few licks | the flies fed little, or `min_untransformed_licks_cutoff` is high for this assay | check the cumulative licks; set the cutoff in the design, for every member |
| The Hub says **re-run needed** | exclusions changed after the results were written | run the member's basic analysis again |
| A linkage group failed for **unexplained light** | most of its lit time had no lick or touch behind it | open *Light explained by licks*: orange beside the red is a drifting well; no orange is a light the firmware had no reason to switch on |
| Unexplained light only **warns**, never fails | no `Program.txt` in `data/` | copy the MCU's exported `Program.txt` into `data/` and run again |
| "holds more than one Program.txt" | two program files in `data/` | keep only the one exported for this recording |
| **No light recorded** fails a DFM | `optogenetics: true`, and the DFM's light was never on | check the lid and the program; set the DFM's own `optogenetics: false` if it ran without a lid |
| **Program mismatch** | the data's light settings or start time differ from the program | check that `Program.txt` is the one exported for this recording |
