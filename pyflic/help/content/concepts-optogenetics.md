# Optogenetic experiments and the light QC

When a DFM drives an Optolid, the firmware decides every millisecond whether each well's LED
is on. It reads the well's own signal, compares it with a threshold from the program the MCU
loaded, and applies the rule of the interval's **Paradigm**. pyflic counts licks afterwards,
from the baselined signal.

The two should agree. When they do not, the light is evidence about the sensor rather than
the fly. A well whose signal drifts upward looks continuously touched to the firmware and
flat to pyflic, so the light runs with no lick behind it, and every light-on number
describes the hardware. The **optogenetic light QC** measures that disagreement for any
experiment type, over the whole recording, whatever window a table uses.

A short tail of light after the last lick is normal: that is the **Hold**. Long stretches of
light with no lick are not.

## Turning it on

One setting, `optogenetics`, under `global:` (or the Project's design):

```yaml
global:
  optogenetics: auto      # the default; also true (yes) or false (no)
```

- **auto** runs the light QC on every DFM that has a section in `data/Program.txt` or any
  LED lit in its data. A recording with neither is not treated as optogenetic.
- **yes** (`true`) runs it on every DFM, and a DFM with no LED lit for the whole recording
  **fails** — an unplugged lid, or a program that did not load.
- **no** (`false`) turns it off.

YAML reads a bare `yes` or `no` as true or false, which is why the editors write those.

A DFM's own entry may override the setting for that DFM, because whether its lid was lit is
a fact about the recording, not about the analysis. The override is allowed inside a
Project, like the rest of `dfms:`:

```yaml
dfms:
- id: 3
  optogenetics: false     # this DFM ran without a lid
  chambers: {1: Ctrl, 2: Ctrl}
```

In the [Config Editor](app-config-editor.md) the setting is on the Experiment tab and each
DFM tab has its own *Optogenetics* picker, which inherits by default. In the Project Design
dialog it is the *Optogenetics* field, with the light QC's thresholds in the *Optogenetic
light QC* group below it.

## Generations

No data file states its firmware version. pyflic reads the shape of each DFM's file once, at
load, and calls it the recording's **Generation**:

| Generation | Written by | Columns | Training | Light |
|---|---|---|---|---|
| 1 | PyMCU 3.x, any older firmware | 27 | a flag inside the signal | `OptoCol1`, sampled five times a second |
| 2 | PyMCU 4.0.0, DFM firmware older than 4.0.0 | 90, the new ones all `NA` | **none** | `OptoCol1` |
| 3 | PyMCU 4.0.0, DFM firmware 4.0.0 | 90, filled | each well's `Phase` | `OnMs`, exact to the millisecond |

A generation-3 file records, on every row, the **Program Interval**, **Cycle** and
**Paradigm** in force, and for each well its **Phase** (`A` in Acclimation, `M` after), its
running **Activations**, its **Requirement** and **Progress** under a ratio Paradigm, and
the milliseconds its LED was on (`OnMs`). The light QC reads those decisions instead of
reconstructing them.

A generation-2 file comes from a lab with an updated MCU and an un-updated DFM. It has light
but no training information at all, and pyflic says so wherever training would show.

`summary.txt` names each DFM's generation and where its light seconds came from. The
report's cover has a **Generation** line and, for an optogenetic member, a **Lit time from**
line. `opto_light_qc.csv` and the light table (`light_summary.csv`, `light_wells.csv`)
carry a `Generation` column.

## Program.txt

The MCU writes the program it ran back out as `Program.txt`. Copy that file into the
member's `data/` folder, beside the DFM CSVs. pyflic reads the exported file only, the one
with `***DFM 1***` sections, not the program you authored for the MCU with `[General]`,
`[Paradigm …]` and `[DFM]` sections.

PyMCU writes two kinds of export. pyflic tells them apart by their content and reads both.

### The PyMCU 4.0.0 export

```
Paradigms:
Closed (ClosedLoop): Frequency 40 Hz, PulseWidth 8 ms, HoldMs 500 ms, MaxOnMs 0 ms, EventGapMs 100 ms
PR10 (ProgressiveRatio): Frequency 40 Hz, PulseWidth 8 ms, ActivationMs 500 ms, Series Arithmetic, FirstRequirement 1, Step 10, Cap 200, EventGapMs 100 ms, Acclimation 10

***DFM 1***
Linkage: 1,1,2,2,3,3,4,4,5,5,6,6
Program Type: Linear
(10/12/2026 20:00:00) Closed, Dark, 20,Off,20,Off,20,Off,20,Off,20,Off,20,Off, 30 min
(10/12/2026 20:30:00) PR10, Dark, 20,Off,20,Off,20,Off,20,Off,20,Off,20,Off, 330 min
```

- **The `Paradigms:` block** names every Paradigm, its type in brackets and every setting.
  It is the only place the type and settings appear: a data row carries only the name.
- **An interval line** gives its start, the Paradigm's name, `Dark` or `Lit`, twelve
  thresholds and its length, in the order of the Program file's Interval line. Earlier
  4.0.0 exports put `Dark` or `Lit` first; pyflic reads both. The k-th line of a DFM's
  section is its Program Interval k.
- **A threshold** is a signal level, `On` or `Off`. A level makes the well a **trigger
  well**. `On` keeps it lit regardless (open loop). `Off` never lights it by its own
  behaviour, and such a well runs no Paradigm of its own.
- **`***Env Mon***`** is the Environmental Monitor's section. pyflic ignores it.
- **`***ALL DFM***`** is what PyMCU writes for a recording started without a Program. pyflic
  reads it as every well `Off` for the whole recording, so any light is a fault.
- **An old program loaded into PyMCU 4.0.0** is exported with generated Paradigm names that
  describe its settings, such as `ClosedLoop-40Hz-8ms-Hold500-MaxOn0`.

### The older export

Before PyMCU 4.0.0 an interval line looked like this:

```
(07/28/2025 12:01:00) Dark Off,20,-1,-1,-1,20,-1,-1,-1,20,-1,-1,-1,F:5160,P:8,D:500,L:1000,M:1000000000000,1440.0min.
```

- **Thresholds** are numbers: a positive one makes a trigger well, `0` is open loop and `-1`
  never triggers.
- **F, P, D, L, M** are frequency, pulse width, decay, delay and max-time-on. A frequency
  above 512 carries the progressive ratio's acclimation-event count in its high bits (`5160`
  is 10 events at 40 Hz). A pulse width at or above 32768 selects non-feeding activation.
  Both are decoded.
- pyflic turns each interval's numbers into the Paradigm PyMCU would make of them, with the
  same generated name. The decay becomes that Paradigm's Hold.

### What pyflic takes from it

- **Linkage**: wells sharing a number form one **linkage group**, lit together whenever any
  member triggers. With the file, the groups always come from its `Linkage:` line, never from
  the data. The data only check them. On firmware 4.0.0, linked wells whose light differs fail
*broken yoking*. On older firmware, the group's notes say how long its linked wells were not
lit together.
- **Each interval's Paradigm and thresholds**: which wells trigger, which are open loop,
  which are off, and the rule the light followed.
- **Program type** and the interval timestamps. On generations 1 and 2 these lay the
  intervals out over the run: Linear holds its last interval to the end, Repeating and
  Circadian start over, and Constant runs its first interval throughout. A generation-3
  file records its own schedule row by row, so the timestamps are used only as a check.
- **Baseline**, the start time and the duration.

### Without it

**The file is optional.** Without it the light QC still runs, but it is limited:

- **Generations 1 and 2.** Linkage is inferred from wells lit together, every lit well
  counts as a trigger well, and the Hold is taken as `opto_default_decay_ms`.
- **Generation 3.** The rows name each Paradigm but not its type, settings or thresholds.
  Linkage comes from wells whose `OnMs` match exactly, and the trigger wells are the ones
  that counted Activations. Every Activation is judged against its group's licks within
  `opto_default_decay_ms`. The checks that need the Paradigm's type are not made, and every
  type-dependent column is `NA`.

Either way, unexplained light can warn but not fail, because without the program an
open-loop schedule is indistinguishable from a stuck light.

A firmware 4.0.0 recording without its program is warned about loudly, because the file it
needs was simply not copied. `pyflic lint`, the report's *At a glance* and the Hub's
**Loaded** line all say:

```
DFM 1: firmware 4.0.0 recording without Program.txt: Paradigm types and settings unknown
```

### When it is wrong

A line pyflic does not understand is skipped and listed in `summary.txt`. A DFM section
that cannot be read is dropped, and that DFM is judged as if there were no program. A file
with no readable DFM section is set aside with the reason. More than one `Program.txt` in
`data/` stops the load, because which one describes this recording is not a guess to make.

On a generation-3 recording the rows are checked against the file. A row naming a Paradigm
the file does not define means the file belongs to another recording, which fails every
group of that DFM (see [the checks](#the-checks)).

## Paradigm types

Each Paradigm is one of six types. Every type has a frequency and a pulse width. The
**Hold** is how long the light stays on after the contact that earned it; the light QC
explains light with it.

| Type | When the LED is on | The Hold | Other settings |
|---|---|---|---|
| **OpenLoop** | the whole interval, except `Off` wells | none | — |
| **ClosedLoop** | during contact, and for the Hold after it ends; one on-period is cut off at `MaxOnMs` | `HoldMs` | `MaxOnMs`, `EventGapMs` |
| **FixedInterval** | contact starts an Activation of exactly `ActivationMs`; the next cannot start until the last one has ended and `LockoutMs` has passed since it started | `ActivationMs` | `LockoutMs`, `EventGapMs`, `Acclimation` |
| **NonFeeding** | on at the start, off during contact, on again `OnDelayMs` after contact ends, for at most `MaxOnMs` | none | `OnDelayMs`, `MaxOnMs` |
| **ProgressiveRatio** | the feeding bout that meets the **Requirement** starts an Activation of exactly `ActivationMs`; the Requirement then rises along its series | `ActivationMs` | `Series`, `FirstRequirement` and `Step`, or `Scale` and `Rate`; `Cap`, `EventGapMs`, `Acclimation` |
| **FixedRatio** | every `Ratio`-th feeding bout starts an Activation of exactly `ActivationMs` | `ActivationMs` | `Ratio`, `EventGapMs`, `Acclimation` |

The firmware counts a feeding bout of its own, a **Firmware Event**: contact that ends once
the signal has stayed below the threshold for `EventGapMs`. It is a different detector from
pyflic's feeding events, on the same signal. The two agree in the main and differ at the
edges.

**Acclimation.** A FixedInterval or ratio Paradigm with `Acclimation: N` starts each well
with N closed-loop Activations: contact lights the LED, and it stays lit for `ActivationMs`
after contact ends. Each time the LED comes on is one Activation, so a fly that comes back
after the light has gone off counts again, even within the same Firmware Event. The well's
`Phase` reads `A` until its Nth Activation ends, then `M` (Main). Each well moves on
separately. Acclimation Activations earn no Firmware Event, so the light-events table leaves
their `FirmwareEvents` empty.

**Same name, same state.** When the next interval names the same Paradigm, every well keeps
its Phase, Requirement and Progress. A different name starts every well afresh.

**A DFM that restarts** on its own mid-recording (a power dip, say) starts afresh too. Its
`Index` and times carry on across the gap, then come a few rows under no instruction
(`Interval` `NA`) until PyMCU sends the instruction again, and from there every well starts
as under a new Paradigm: Acclimation again, the Requirement series from its first value.
Its Activation totals carry on. pyflic finds each restart in the rows, and `Messages.csv`
confirms it when it is in `data/`.

The firmware's own definitions are in PyMCU's `ProgramFormat.md`.

### Paradigm Families

pyflic checks and summarises the six types in four **Paradigm Families**: types the
firmware lights by the same mechanism, so they share one check and one set of measures. A
type gets a check of its own only where it has a setting its family lacks: `MaxOnMs`,
`LockoutMs` or the Requirement series.

- **Open loop**: OpenLoop, and any well set `On` under another Paradigm. Lit throughout,
  whatever the fly does, and never an Activation.
- **Closed loop**: ClosedLoop, and the Acclimation of FixedInterval, ProgressiveRatio and
  FixedRatio. Contact lights the LED, and it stays lit for the Hold after contact ends;
  each on-period is one Activation. With `MaxOnMs`, an on-period is cut off there.
- **Fixed length**: FixedInterval, ProgressiveRatio and FixedRatio after Acclimation. An
  earned Activation lasts exactly `ActivationMs`. Its *interval* branch (FixedInterval) is
  earned by contact once the lockout has passed; its *ratio* branch (the two ratio types)
  by the Firmware Event that meets the Requirement.
- **Non-feeding**: NonFeeding. Dark while the fly feeds, lit `OnDelayMs` after it stops.

A well's family follows its Phase: a FixedInterval well in Acclimation is closed loop until
its Acclimation ends. A well set `Off` belongs to no family. The families decide the
[rule checks](#each-paradigms-own-rule), the [light table](#the-light-table)'s measures
and the report's *Light behaviour* subsections, under every experiment type: the
experiment type describes the feeding design, the Paradigm the light.

## Roles

Every chamber of an optogenetic DFM gets a **Role**: what it was to the light, in any
design.

| Role | The chamber |
|---|---|
| **Paired** | holds the trigger well of a linkage group that also lights other chambers, and is the only chamber of that group that can |
| **Yoked** | shares a linkage group with a Paired (or Mutual) chamber and holds none of its trigger wells: lit when its partner earns light |
| **Mutual** | is one of several chambers of one linkage group that each hold a trigger well |
| **Self** | holds a trigger well whose group lights no other chamber |
| **Open** | is lit regardless of its fly: every well `On`, or in a group lit open loop |
| **Unlit** | none of the above |

Roles come from the program's linkage and thresholds. On a generation-3 recording without a
program they come from the data: trigger wells are the wells that counted Activations, and
wells whose `OnMs` match exactly are linked. A generation-1 or 2 recording without a program
cannot tell a trigger well from a yoked one, so its Role is left blank.

The Role is a column beside the Treatment, never a Treatment level. It appears on
`feeding_summary.csv` rows, in the [light table](#the-light-table), in the lane labels of
the [Program timeline](#the-program-timeline), and per group in `opto_light_qc.csv`'s
`Roles` column (`1 Paired, 2 Yoked`). A [Progressive Ratio](concepts-progressive-ratio.md)
experiment (a type with the Progressive Ratio module) keeps its own `paired` and `yoked`
everywhere, from its chamber groups.

## What the light QC checks

The unit of judgement is the **linkage group**: its light is one circuit. Every verdict is
passed on to each chamber the group touches.

### Windows

Each group is judged interval by interval, under that interval's Paradigm.

- **Generations 1 and 2.** The intervals are laid out by clock time from the program.
- **Generation 3.** The rows are the schedule: each run of rows under one Program Interval
  and Cycle is a window. The baseline rows before the first instruction (Program Interval 0)
  and rows under an instruction outside the program are not judged. A Program Interval
  beyond the lines `Program.txt` prints (it omits intervals past the duration and lists a
  Repeating program's first pass only) is judged by its Paradigm's name, with the trigger
  wells read from the Activations.

A group's mode in a window comes from its members' thresholds: any member `On` is open loop,
every member `Off` is off, and otherwise the Paradigm's type rules.

### Explained light

The light is judged by the rule of its Paradigm type:

| Light under | Is explained when |
|---|---|
| **ClosedLoop** | a trigger well has a feeding lick or a tasting sample from the Hold (`HoldMs`) plus `opto_decay_tolerance_samples` (2) before the lit sample to the tolerance after it |
| **FixedInterval**, **ProgressiveRatio**, **FixedRatio** | an Activation's whole `ActivationMs` of light is explained when its trigger well had a lick or touch within the tolerance of its onset. Feeding during a FixedInterval lockout that earned nothing is expected |
| **Acclimation** (Phase `A`) | the ClosedLoop rule, with `ActivationMs` as the Hold |
| **NonFeeding** | not judged by explained light; checked instead for light that stayed on through feeding |
| **OpenLoop** and `On` wells | not judged by explained light; checked for their lit fraction |
| every member `Off` | never: any light fails |

Tasting samples count because the firmware decides on contact at millisecond resolution, and
a brief touch it counts often reads below pyflic's feeding threshold. The tolerance absorbs
the difference between those decisions and the recorded samples.

**On generation 3 the light is attributed to Activations.** Lit time is `OnMs`, exact to the
millisecond. Each Activation of a trigger well owns the group's light from its onset to the
next onset, the light going out, or the end of its window plus the Hold, whichever comes
first. Unexplained light is the lit time of Activations whose onset had no pyflic lick or
touch. Light no Activation owns is judged sample by sample with the window rule.

**On generations 1 and 2** every lit sample of `OptoCol1` is judged with the window rule and
the interval's Hold (the old export's decay).

**Light already on at a Slice's first row** came on before the Slice, so the onset that
explains it is not in the Slice's rows. It is not judged for up to one Hold plus the
tolerance, on every generation; light still on after that is judged as usual, so a stuck LED
still shows.

The thresholds are the same for every type: at least `opto_unexplained_warn_fraction` (10%)
of the group's judged lit time unexplained warns, at least `opto_unexplained_fail_fraction`
(30%) fails, and neither is judged until a group has `opto_unexplained_min_sec` (30 s) of
unexplained light, so a few stray samples are not called a fault. Where a group is flagged,
`UnexplainedOnsetMin` is the start of the first 30-minute window in which its unexplained
share reached the warning fraction: the latest a hand-set cutoff could fall.

Light at the start of an interval is left to the interval before it for the length of its
Hold.

### The checks

Each check writes a flag into the group's `Flags`, in the words below.

| Check | Verdict | Generation |
|---|---|---|
| **unexplained light**: at least 30% of the group's judged lit time unexplained | **fails** the group | all |
| **partly unexplained light**: at least 10% unexplained | warning | all |
| **unexplained light (no Program.txt)**: 30% or more without a program, which cannot fail | warning | all |
| **light while off**: lit while every member's threshold was `Off` (-1), or under PyMCU's `Idle` instruction after the program ended | **fails** the group | all |
| **no light recorded**: `optogenetics: yes`, and no LED lit on the DFM at all, where the program expected light (NonFeeding expects it too) | **fails** the DFM | all |
| **broken yoking**: linked wells' `OnMs` differ by more than 1 ms in more than `opto_yoking_fail_fraction` (1%) of the group's lit records, or an `Off` well linked to a trigger well counted Activations of its own | **fails** the group | 3 |
| **paradigm not in program**: a row names a Paradigm `Program.txt` does not define, so the file belongs to another recording; those rows are not judged | **fails** every group of the DFM | 3 |
| **short activations**, **max on exceeded**, **wrong activation length**, **lockout broken**, **requirement series wrong**: a Paradigm's own rule broken ([below](#each-paradigms-own-rule)) | **fails** the group | 3 |
| **unlit feeding**: at least `opto_unlit_feeding_fraction` (50%) of the feeding bouts at its trigger wells under closed loop or in Acclimation never lit the group (5 bouts needed) | warning | all |
| **no light events**: feeding at its trigger wells, and never lit | warning | all |
| **open loop not lit**: lit for less than `opto_open_loop_min_lit_fraction` (95%) of an open-loop interval; on generation 3 the lowest of the group's open-loop windows, so one dark Program Interval or one dark pass of a Repeating program shows | warning | all |
| **non-feeding light during feeding**: under NonFeeding, lit through more than `opto_non_feeding_lit_warn_fraction` (10%) of the feeding at its trigger wells, and for at least `opto_non_feeding_min_sec` (30 s) | warning | 3 |
| **contact without activation**, **non-feeding dark**, **interval started late**, **DFM restarted**: [below](#each-paradigms-own-rule) | warning | 3 |
| **acclimation count mismatch**: a trigger well's completed Acclimation counted a different number of Activations than its Paradigm's `Acclimation` | warning | 3 |
| **firmware events disagree**: under a ratio Paradigm's Main phase, a trigger well's Firmware Events and pyflic's feeding events differ by more than `opto_event_disagreement_warn_fraction` (25%) of the Firmware Events (judged from 5 events). pyflic's count leaves out the feeding events that start while the well's own Activation is lit, after its onset row, as the firmware starts no Event then | warning | 3 |
| **lost records**: more than `opto_lost_records_warn` (25) records missing from the DFM's `Index`, not counting a restart's gap, which **DFM restarted** reports | warning, DFM | 3 |
| **unknown-instruction rows**: more than `opto_unknown_rows_warn_fraction` (1%) of the rows after the lead-in (Program Interval 0) recorded under an instruction outside the program | warning, DFM | 3 |
| **late records**: more than `opto_late_records_warn` (10) records with an `OnMs` above 200 ms | warning, DFM | 3 |
| **program mismatch**: the data's `OptoFreq`, `OptoPW` or `Dark` differ from the program, the data start more than a minute from its Start Time, or (generation 3) the rows name another Paradigm than the printed line | warning | all |
| **no program section**: `Program.txt` has no usable section for the DFM | warning | 1, 2 |
| **no Program.txt for firmware 4.0.0**: a generation-3 DFM without its program | warning | 3 |

The firmware 4.0.0 checks with constants read the file's own bookkeeping. Their defaults
were set on simulated recordings and on PyMCU's own simulator, and are to be revisited on
bench recordings of each Paradigm type. Treat a warning from one as a reason to look, and
see [Thresholds](#thresholds).

### Each Paradigm's own rule

On a generation-3 recording with its `Program.txt`, each Paradigm is also checked against
its own rule, one Activation at a time, with the program's settings. The firmware keeps
every rule to the millisecond, so a check that fails means `Program.txt` is probably not
the program that ran, or the LED or its `OnMs` report is faulty: never the fly. The checks
that rest on pyflic's own lick detection only warn, because pyflic's threshold is not the
firmware's.

The checks read the **Activation ledger**: each Activation's onset, end and lit time to the
millisecond, from the edges of its `OnMs`. An Activation is **judged** only when its light
is its own (no linked partner's light, no light already on, no next Activation run into
it) and its rule ended it (not an Interval change, a gap in `Index` or the end of the
data). The rest are counted as not judged, in the group's notes. A light period timed only
to the record gets one record more of tolerance. Windows that ran the same Paradigm are
pooled within a group, and each fraction is judged on the Paradigm's total.

**Closed loop**

- **short activations**: a closed-loop Activation (ClosedLoop, or any Acclimation) lit for
  less than its Hold, `HoldMs`, or `ActivationMs` in Acclimation (or less than `MaxOnMs`,
  when that is shorter), less `opto_activation_tolerance_ms` (25 ms). Lit time is the
  contact plus the Hold, so never less. **Fails** above `opto_activation_length_fail_fraction`
  (5%) of the judged ones.
- **max on exceeded**: a ClosedLoop Activation lit longer than its `MaxOnMs` plus the
  tolerance. **Fails** on any one. The Activations within the tolerance of `MaxOnMs` are
  reported as `CutOffs`.

**Fixed length**

- **wrong activation length**: a Main Activation of FixedInterval, ProgressiveRatio or
  FixedRatio whose light did not last `ActivationMs`, within the tolerance. **Fails** above
  `opto_activation_length_fail_fraction` (5%).
- **lockout broken** (FixedInterval): consecutive Main Activations started closer than the
  longer of `LockoutMs` and `ActivationMs`, less the tolerance. **Fails** above
  `opto_lockout_fail_fraction` (2%).
- **contact without activation** (FixedInterval): a run of pyflic licks or touches at a
  trigger well that began with the well free (its Activation over, and the longer of
  `LockoutMs` and `ActivationMs` since its last onset) and brought no Activation in its
  first record or the next. **Warns** above `opto_missed_contact_warn_fraction` (25%) of at
  least 5 such runs: contact below the firmware's threshold, or a dead sensor.
- **requirement series wrong** (ProgressiveRatio, FixedRatio): a Main Activation that met a
  Requirement its series does not give for its number in the Paradigm's run. **Fails** on
  any one. A run whose first Activation fell on its first row, which may still be the old
  Paradigm's, is matched with every number one less, and noted.

**Non-feeding**

- **non-feeding dark**: the time the light was off although the rule says lit: no pyflic
  activity at the trigger well for longer than `OnDelayMs` plus the tolerance, and with
  `MaxOnMs` within the first `MaxOnMs` after that. **Warns** above
  `opto_non_feeding_dark_warn_fraction` (10%) of the time the rule says lit, once that is
  10 s: contact the firmware read and pyflic did not (a drifting baseline), or a longer
  `OnDelayMs` than `Program.txt` states. A lamp that fails in the lid does not show in the
  data: `OnMs` is the firmware's record of driving the LED.
- **max on exceeded**, with `MaxOnMs` above 0, as for closed loop.
- `RelightDelayMs` is reported, not judged: the median time from the end of the fly's last
  activity sample to the light coming back, against `OnDelayMs`. Activity is good to a
  record, so it can read up to a record either side.
- **non-feeding light during feeding** (in the table above) asks the opposite question.

**Open loop**: **open loop not lit**, per window (in the table above).

**Every family**

- **interval started late**: the worst lag of a Program Interval's first row, on every
  Cycle, behind its `Program.txt` time, as PyMCU lays the schedule out (a Circadian program
  by the time of day). The firmware's own lag is one record. The DFM's first Interval
  follows PyMCU's baseline lead-in, so its first 5 s are allowed. **Warns** above
  `opto_interval_lag_warn_sec` (5 s): the DFM took the instruction late, and PyMCU sent it
  again.
- **DFM restarted**: each restart the rows show, with its time and the records lost.
  **Warns** on every group of that DFM. Acclimation, the Activation numbers and the
  Requirement series start again after a restart, and the checks follow them.

Without `Program.txt` none of these runs but **DFM restarted**, which reads the rows alone,
and their columns are `NA`. On generations 1 and 2 none runs: `OptoCol1`, sampled every
200 ms, cannot time a light period.

### Drift or hardware?

Each flagged group gets a `LikelyCause`: **a drifting baseline or sustained contact** when
most of its unexplained light fell where the firmware saw contact and pyflic saw no lick or
touch, otherwise **hardware, linkage or program mismatch**: light the firmware had no reason
to switch on.

**Generation 3 reads the firmware's own decisions.** An Activation, or a Firmware Event (a
rise in `Progress`), with no pyflic lick or touch at its trigger well is the drifting-baseline
signature itself. On these recordings:

- `ContactSec` is the lit time of the trigger wells' Activations: the firmware's contact
  decisions.
- `ContactWithoutActivitySec` is the lit time of the Activations whose onset had no pyflic
  lick or touch at their well.
- `FirmwareEventsWithoutActivity` counts the Firmware Events with none either.

**Generations 1 and 2 emulate the firmware.** With a program, pyflic re-runs the firmware's
own test on the recorded signal: raw minus the mean of the run's first ten seconds, against
the interval's threshold. `ContactWithoutActivitySec` is how long that **emulated trigger**
read a trigger well above threshold while pyflic saw no lick or touch.
`EmulationApproximate` marks it approximate when the data begin more than two seconds after
the program, so the baseline window was not recorded. A
[Slice](concepts-slices.md) that starts later still reads the baseline from the run's first
ten seconds, which the load keeps aside; only without them is it approximate, with a note. On generation 3 nothing is emulated and
the column is `NA`.

## Thresholds

The thresholds are design `constants:`, shared by every experiment type, with fields in the
Project Design dialog and the Config Editor. The dialog shows them unless `optogenetics` is
no; the Config Editor once the experiment is known to be optogenetic (`optogenetics: yes`,
a `Program.txt` in `data/`, or light QC files from a run).

| Constant | Default | What it does |
|---|---|---|
| `exclude_failed_opto_chambers` | false | every chamber of a failed group leaves the analysis through auto-removal |
| `opto_unexplained_warn_fraction` | 0.10 | unexplained share that warns |
| `opto_unexplained_fail_fraction` | 0.30 | unexplained share that fails |
| `opto_unexplained_min_sec` | 30 | unexplained seconds needed before either is judged |
| `opto_default_decay_ms` | 1000 | the Hold assumed without a `Program.txt` |
| `opto_decay_tolerance_samples` | 2 | samples of slack around the Hold |
| `opto_unlit_feeding_fraction` | 0.50 | unlit share of closed-loop feeding bouts that warns |
| `opto_open_loop_min_lit_fraction` | 0.95 | lit share an open-loop interval needs |
| `opto_event_disagreement_warn_fraction` | 0.25 | firmware 4.0.0: a trigger well whose Firmware Events and pyflic feeding events differ by more than this share of the Firmware Events warns |
| `opto_yoking_fail_fraction` | 0.01 | firmware 4.0.0: share of a group's lit records in which its linked wells' `OnMs` differ by more than 1 ms, above which the group fails |
| `opto_lost_records_warn` | 25 | firmware 4.0.0: records missing from `Index` (5 s at 5 Hz), outside restarts, above which the DFM warns |
| `opto_unknown_rows_warn_fraction` | 0.01 | firmware 4.0.0: share of the rows after the lead-in under an unknown instruction above which the DFM warns |
| `opto_late_records_warn` | 10 | firmware 4.0.0: records with an `OnMs` above 200 ms above which the DFM warns |
| `opto_activation_tolerance_ms` | 25 | firmware 4.0.0: how far a light period may differ from its Hold, `ActivationMs` or `MaxOnMs`, or two onsets from the lockout, and still keep the rule (one record more for a period timed to the record) |
| `opto_activation_length_fail_fraction` | 0.05 | firmware 4.0.0: share of judged Activations that break their length rule (*wrong activation length*, *short activations*) above which the group fails |
| `opto_lockout_fail_fraction` | 0.02 | firmware 4.0.0: share of judged FixedInterval Main Activations started inside the lockout above which the group fails |
| `opto_missed_contact_warn_fraction` | 0.25 | firmware 4.0.0: share of at least 5 free FixedInterval contact runs that brought no Activation above which the group warns |
| `opto_non_feeding_lit_warn_fraction` | 0.10 | firmware 4.0.0: share of the feeding under NonFeeding that was lit above which the group warns |
| `opto_non_feeding_min_sec` | 30 | firmware 4.0.0: seconds of NonFeeding light during feeding needed before it is judged |
| `opto_non_feeding_dark_warn_fraction` | 0.10 | firmware 4.0.0: share of the time a NonFeeding rule says lit that was dark above which the group warns |
| `opto_interval_lag_warn_sec` | 5 | firmware 4.0.0: seconds a Program Interval may start after its `Program.txt` time before its DFM warns |

The last thirteen apply only to firmware 4.0.0 recordings. *Non-feeding light during
feeding* has its own two constants, whose defaults are the unexplained-light values it used
to borrow, so its results are unchanged. The defaults of the eight rule-check constants
were set on simulated recordings, where clean data stayed far inside them, and are to be
revisited on bench recordings of each Paradigm type.

**Failures are flagged, not excluded, by default.** A faulty light does not invalidate the
feeding record, so a failed group stays in every result with its flags on its chambers'
rows (`OptoLightQC`, `OptoUnexplainedFraction`) until you exclude it by hand or switch
`exclude_failed_opto_chambers` on.

## Outputs

Basic analysis and the Hub's **Run QC** write, into the member's `qc/opto/` (the four
tables are Run QC's *Opto light QC tables* item, the figure its *Light explained by licks*):

| File | One row per | Holds |
|---|---|---|
| `opto_light_qc.csv` | linkage group | `Generation`, wells, trigger wells, chambers, their `Roles`, treatment, lit and unexplained time, onset, the contact evidence, `FirmwareEvents` and `PyflicEvents` under a ratio Paradigm, `AcclimationActivations` and `AcclimationExpected`, the rule checks' columns (below), `LikelyCause`, `Flags`, `Verdict`, `Excluded` |
| `opto_light_intervals.csv` | group × window | the window's printed `Interval`, and on generation 3, as the rows recorded it, `ProgramInterval`, `Cycle`, `Paradigm` and `ParadigmType`; its mode, the `HoldMs` it was judged with, its trigger wells, `LitSource`; lit, unexplained and contact time; feeding bouts and unlit ones |
| `opto_light_events.csv` | light event | its `Well`, onset, end and duration, interval, `Phase`, the `RequirementMet` and `FirmwareEvents` behind it, `UnexplainedSec`, `Explained`, `OverrunSec` (the light's time past the last lick plus the Hold), its `Source`, and on generation 3 the ledger's columns (below) |
| `opto_program.csv` | program interval | the program as read: the Paradigm's name and type (`ParadigmType`, as PyMCU spells it) with its settings (`HoldMs`, `ActivationMs`, `LockoutMs`, `Acclimation`, the Requirement series …, `NA` where the type has none), linkage, trigger wells, and for an export older than PyMCU 4.0.0 the raw `F` and `P` it echoed (`LegacyF`, `LegacyP`) |
| `opto_light_dfm<id>.png` | DFM | the QC figure, below |

A light event is a lit run of `OptoCol1` on generations 1 and 2 (`Source` = `optocol1`), and
one Activation of a trigger well on generation 3 (`activations`). The firmware columns are
`NA` before generation 3.

**The rule checks' columns** in `opto_light_qc.csv`, summed over the group's Paradigms, `NA`
outside their family, without `Program.txt` and before generation 3:

- `ShortActivations`, `OverMaxOnActivations`, `WrongLengthActivations`,
  `LockoutViolations`, `SeriesMismatches`: the judged Activations that broke the rule;
  `CutOffs`, the ones `MaxOnMs` ended; `MissedContacts`, the free contact runs that brought
  no Activation;
- `NonFeedingDarkSec`, seconds dark where the rule says lit; `RelightDelayMs`, a median;
- `IntervalLagSec`, the DFM's worst Program Interval start; `Restarts`, its restarts (with
  or without a program);
- `OpenLoopLitFraction`, on generation 3 the lowest over the group's open-loop windows.

**The ledger's columns** in `opto_light_events.csv`, on generation 3 (`NA` on the
`OptoCol1` rows):

- `OnsetMs`, `EndMs`, `LitMs`: the light period in milliseconds from the start of the first
  record (not the same origin as `OnsetMin`, which counts from the first row's time stamp);
- `Timing`: `exact` when both edges are known to the millisecond, `record` when to a record;
- `Overlap`: other light merged with this one, so its length is not its own;
- `Boundary`: `interval`, `gap` or `end` when the period was stopped by an Interval change,
  a gap in `Index` or the end of the data; empty when only its rule can have ended it;
- `SinceLastOnsetMs`: from the well's previous onset in the same Paradigm run;
- `K`: its number among the well's Main Activations of the run (a restart starts again);
- `Family`: its [Paradigm Family](#paradigm-families), from its type and Phase;
- `EndedBy`: what ended it: `Hold` or `MaxOnMs` (closed loop), `length` (fixed length),
  `contact` or `MaxOnMs` (non-feeding), `interval`, `gap` or `end`, or `unknown`.

`summary.txt` gains an *Optogenetic light QC* section, with every constant it judged by, and
the experiment report a subsection of Quality control with the program table, the verdicts
and the figure. On generation 3 with `Program.txt` that subsection goes on with one
**Rule check** subsection per Paradigm the program runs, in family order: its rule in words
with its settings, a table of its checks per linkage group with the Activations judged and
not judged, and a rule figure with the values that break the rule in red. A Paradigm with
Acclimation has a closed-loop subsection for its Acclimation as well, and wells set `On`
under another Paradigm one of their own. See [the rule figures](plots-catalog.md#rule-figures).

The figure, **Light explained by licks**, gives each group a row: its lit time per bin,
explained and unexplained, and beside it the contact with no lick — the emulated trigger's
on generations 1 and 2, the light of Activations with no lick or touch on generation 3
(titled *firmware events, no lick*). The report's cover counts the groups that failed or
warned, and how many groups raised each flag. A Project's Combined Analysis stacks every
member's verdicts into `<project>_LightQC.csv` with `Source` = `Opto`.

The Hub's QC panel has an **Optogenetics** group whenever the loaded member is
optogenetic: **Opto light QC tables**, **Light explained by licks** and **Program timeline**,
which **Run QC** writes while they are ticked. A Progressive Ratio member's own light checks
are in the panel's **Progressive Ratio** group, and the light table is the Analyze panel's
**Light tables**. The single Script Editor actions are `opto_light_qc`, `plot_opto_light`,
`plot_program_timeline` and `light_summary`. The
[QC Viewer](app-qc-viewer.md#the-opto-light-qc-tab) has an **Opto Light QC** tab: the
verdicts with the rule checks' columns, why each group got its own, the figure per group, a
Params recompute that updates them live, and buttons that tick a failed group's chambers for
exclusion.

## The light table

The light QC asks whether the light was right. The **light table** asks what it did, and
puts the answer in the feeding summary's shape so it can be pooled and tested like feeding.
Every optogenetic experiment of any type gets one. Basic analysis and the Hub's **Run
analysis** (its *Light tables* item) write it into `analysis/`:

| File | One row per |
|---|---|
| `light_summary.csv` | chamber |
| `light_wells.csv` | well |
| `light_summary_facet.csv`, `light_wells_facet.csv` | the same, per Facet, when the experiment has Facets |

Each row carries the Treatment, the design factors, the `Role` and the `Generation`, then:

| Columns | What they hold |
|---|---|
| `Paradigm`, `ParadigmType`, `ProgramInterval`, `Cycle` | what ran over the window, or `mixed` |
| `WindowMinutes`, `LitSec`, `LitFraction`, `LitSource` | the light received: from `OnMs` on generation 3, from `OptoCol1` samples before |
| `LightOnsets` | times the light came on; on generation 1 the nearest thing to an Activation count |
| `Activations`, `ActivationsPerHour` | on-periods the well's own Paradigm started |
| `AcclimationActivations`, `AcclimationEndMin`, `AcclimationComplete`, `MainActivations` | FixedInterval and ratio Paradigms with Acclimation: the Activations in Acclimation and after it, which add up to `Activations` |
| `FirmwareEvents`, `LastRequirementMet`, `ProgressAtEnd`, `EventsPerActivation` | ratio Paradigms |
| `MedianActivationSec`, `CutOffs` | closed loop (ClosedLoop, and Acclimation): the median judged Activation's lit time; with `MaxOnMs`, the Activations it cut off |
| `MaxActivations`, `RewardEfficiency`, `MedianReturnSec`, `LockedOutContactSec` | FixedInterval: the most Activations its Main time allowed (that time over the longer of `LockoutMs` and `ActivationMs`); the Main Activations earned as a share of that; the median wait, once the lockout was over, before the next Activation; and the fly's lick time while its own lockout ran and its own light was off |
| `UnlitSec`, `Suppressions`, `MedianSuppressionSec` | NonFeeding: time dark; the light periods the fly's contact ended (not those `MaxOnMs` ended); and the median dark time from one of those to the light coming back |
| `LicksPerMinLit`, `LicksPerMinDark`, `LightFeedingIndex` | open loop: the chamber's licks per minute in its own light and in the dark, over the rows a Program Interval ran, and (lit − dark) / (lit + dark) of the two |
| `PyflicEvents`, `PyflicLicks` (per well); `EventsA`, `EventsB` (per chamber) | pyflic's own count, for comparison |

A chamber row sums its wells' counts and takes its medians over all its wells' Activations.
Its `LitSec` is the chamber's own light, counted once however many of its wells were lit,
its `AcclimationEndMin` the latest of its wells', and `AcclimationComplete` true only when
all of them finished.

A column is `NA` wherever the well ran no Paradigm of a type that defines it, and every
type-dependent column is `NA` without the DFM's `Program.txt` section. On generations 1 and
2 every firmware column is `NA`. The baseline before the first instruction and rows under an
unknown instruction count toward the light but never toward `Paradigm`, and never toward the
open-loop rates. A light phase shorter than a minute has no rate: a chamber lit for the whole
program has no dark rate and no `LightFeedingIndex`. The families' columns judge an
Activation as the [rule checks](#each-paradigms-own-rule) do, with
`opto_activation_tolerance_ms`.

**Headline metrics by family.** The metrics tested between treatments follow the
[Paradigm Families](#paradigm-families) the program runs:

- open loop: `LitSec`, `LightFeedingIndex`;
- closed loop: `Activations`, `LitSec`, `MedianActivationSec`, and `CutOffs` when a
  ClosedLoop runs with `MaxOnMs`;
- fixed length, interval: `Activations`, `RewardEfficiency`, `MedianReturnSec`;
- fixed length, ratio: `Activations`, `LastRequirementMet`, `FirmwareEvents`;
- non-feeding: `LitFraction`, `Suppressions`;
- Acclimation, when a Paradigm has one: `AcclimationEndMin`.

The experiment report's **Light behaviour** section has one subsection per family the
program runs, each with its own metrics, from the chambers whose window ran that family; a
chamber that ran several shows under each. Open loop adds *Feeding in light and dark* under
a type with the Light and dark [module](concepts-experiment-types.md#analysis-modules)
(Custom and ProgressiveRatio include it). A Project pools the members' tables the same way.
A member without a firmware 4.0.0 recording or a program keeps the four metrics of before in
one section: `Activations`, `LitSec`, `FirmwareEvents` and `AcclimationEndMin`
(`light_summary.HEADLINE_METRICS`). See [Light behaviour plots](plots-catalog.md#light-behaviour-plots).
Each numeric column is also a faceted Publication Figure in the
[Plot Editor](app-plot-editor.md), offered once the Project's Combined Analysis has a light
table.

**Only chambers that earn light are compared on Activations.** A Yoked chamber's light is
its partner's, an Open one's is given regardless, and an Unlit one has none, so their
Activations are 0 by construction and would dilute the comparison. In the statistics their
rows keep their Role and their lit time (`LitSec`, `LitFraction`), so a yoked fly's light is
seen beside its partner's, and lose every Activation measure: `Activations` and the
Acclimation, ratio, closed-loop and FixedInterval columns, `Suppressions` and
`MedianSuppressionSec`. Paired, Self and Mutual chambers keep them, and so does a chamber
whose Role is blank, since nothing says its light is not its own. A Progressive Ratio
experiment compares its paired chambers only, as before. A metric that is the same in every
chamber, such as an open-loop chamber's lit time, is named once as not compared.

Every optogenetic `feeding_summary.csv` row also carries its chamber's `Role`, `LitSec` and
`Activations`. A Project's Combined Analysis stacks the members' tables into
`<project>_LightSummary.csv` and `<project>_LightWells.csv` (and their `_Facet` variants),
with `Experiment` first.

## Paired and yoked designs

A linkage group whose chambers are one **Paired** chamber and one or more **Yoked** ones is
a yoked group: the paired fly's feeding lights its partners at the same moments, whatever
they do. Under a type with the **Paired and yoked**
[module](concepts-experiment-types.md#analysis-modules) (Custom includes it), pyflic then
asks Progressive Ratio's question of it: does the paired fly feed differently from a fly
that got the same light for free? A Progressive Ratio experiment has its own
[paired − yoked difference](concepts-progressive-ratio.md#statistics) instead, and a type
without the module, such as Hedonic, has neither.

- **`analysis/yoked_comparison.csv`**, written by basic analysis and by Run analysis's
  *Paired and yoked comparison* beside the light table, only when a yoked group exists (the
  Hub greys the box otherwise, and does not offer it under a type without the module). One
  row per yoked group and window: `Facet` (`Whole recording`, then each Facet) and
  `FacetRange`, `DFM`, `Group` (the linkage group), `PairedChamber`, `YokedChambers`,
  the paired chamber's `Treatment` and factor levels, then per metric `Paired<m>`,
  `Yoked<m>` (the mean of its yoked chambers) and `d<m>`, the difference. The metrics are
  the feeding summary's licks, events and median duration, per well on a two-well layout
  (`LicksA`, `LicksB`, …). An excluded chamber is left out; a group without its paired
  chamber or any yoked one has no row.
- **`<project>_YokedComparison.csv`**: a Project's Combined Analysis stacks the members'.
- **Paired and yoked**, in both reports after the light results: the differences by
  treatment, each tested against zero with the paired t-test (a one-sample t-test on the
  differences) and the Wilcoxon signed-rank test beside it, then compared between
  treatments, one observation per group; a Project adds the mixed models. A figure plots
  each group's paired value against its yoked mean over the identity line. With one
  treatment, the one-treatment note replaces the comparison between treatments. When the
  type's [control comparison](concepts-experiment-types.md#control-comparisons) is Paired
  against Yoked, this subsection opens the Results instead, and is not repeated.

A design with no yoked group writes no such file and has no such subsection. From Python,
`pyflic.base.analytics.yoked_comparison(exp)` returns the table.

## Program Facets

With `facet_source: program` in the design, every **Program Interval** of a DFM's program is
a [Facet](concepts-facets.md#program-facets), labelled by its Paradigm's name: every metric
is split by what the program ran. `facet_source: program_cycles` separates the passes of a
Repeating program. On a generation-3 recording the Facets come from the rows themselves; on
an older one from the program's clock schedule, noted as approximate. Either way every DFM
needs its `Program.txt` section: without one, the member does not load.

## The Program timeline

One figure per DFM: what its program ran, and what each well did.

- **The top band** is the Program Intervals, coloured by Paradigm type and labelled by
  Paradigm name, with the Cycle of a repeating program. Light grey is the baseline before
  the first instruction, the palest grey PyMCU's own `Idle` (no light), hatched white an
  unknown instruction, and dark grey a Paradigm whose type is unknown, because
  `Program.txt` does not define it or is missing.
- **Below it, one lane per well**, its chamber and Role in the label: its Acclimation
  (Phase `A`) shaded, each Activation as a tick, and the share of each bin its LED was on as
  a blue strip. On generation 3 a lane is also hatched where `Program.txt` sets the well
  `On`, and grey where it sets it `Off`.
- **A red band labelled "restart"** across the band and the lanes marks each time the DFM
  restarted, from the last record before the gap to the instruction taking hold again
  (generation 3).

A generation-3 recording draws all of this from its own rows. A generation-1 or 2 recording
with a `Program.txt` section gets the band from the program's clock schedule, approximate,
and the lanes show `OptoCol1` only. With neither there is nothing to draw.

**Run QC** writes it to `qc/program/program_timeline_dfm<id>.png` for every DFM that has
one, and basic analysis for every generation-3 DFM, in an optogenetic experiment only: a
recording whose light never came on has nothing for it to show. It appears in the report's Quality
control section, behind the QC panel's **Program timeline** box (`plot_program_timeline`),
and in the [QC Viewer](app-qc-viewer.md#the-program-sub-tab)'s **Program** sub-tab, with
each well's state at the end of the loaded range and, on generation 3, the DFM's
Activations from `opto_light_events.csv`. See
[Reading the Program timeline](plots-catalog.md#program-timeline).

## Progressive Ratio

A [Progressive Ratio](concepts-progressive-ratio.md) experiment gets this light QC as well
as its own, which asks whether the paired fly earned each light event. The program serves
both:

- **Chamber groups from the program.** With a `Program.txt`, a DFM entry may leave
  `paired_chambers` out: the program's linkage makes the groups. A linkage group lighting
  two chambers is a pair, its chamber holding the trigger well the paired one; one lighting
  one chamber is a solo group, with no yoked control. When the config states the groups and
  the program disagrees, the config wins and `summary.txt` says so. A trigger well that is
  well B is flagged, because the Sucrose Well is always well A.
- **One series check.** On generation 3 the Progressive Ratio light QC fails a group with
  *requirement series wrong* when a Test light event met a Requirement the Test Paradigm's
  series does not give: it calls this light QC's own check.
- **Hold-aware lick-free events.** A light event is lick-free only when it is credited
  with no Sucrose Well licks *and* has no lick or touch within its Hold, the program's or
  `opto_default_decay_ms`. A light the fly touched for is never counted against it.
- **Both ratio Paradigms.** The Progressive Ratio module covers ProgressiveRatio and
  FixedRatio.
- **A type mismatch is reported.** A program running a ratio Paradigm under a type without
  the Progressive Ratio module (Custom, say), or a type with that module whose program runs
  no ratio Paradigm, is noted in `summary.txt` and the report. A type that lists the
  Paradigm types it expects refuses to load a program that runs another; see
  [The light a type expects](concepts-experiment-types.md#the-light-a-type-expects).

---

Related: [Light state and phase analysis](concepts-light-phase.md) ·
[Progressive Ratio experiments](concepts-progressive-ratio.md) ·
[Facets](concepts-facets.md) · [Exclusions](concepts-exclusions.md) · [Reports](reports.md)
