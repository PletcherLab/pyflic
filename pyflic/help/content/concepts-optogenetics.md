# Optogenetic experiments and the light QC

When a DFM drives an Optolid, the firmware decides every millisecond whether each well's LED
is on. It reads the well's own signal, compares it with a threshold from the program the MCU
loaded, and keeps the LED lit for a set **decay** after contact ends. pyflic sees only the
outcome, the `OptoCol1` column of LEDs actually lit, and counts licks afterwards from the
baselined signal.

The two should agree. When they do not, the light is evidence about the sensor rather than
the fly. A well whose signal drifts upward looks continuously touched to the firmware and
flat to pyflic, so the light runs with no lick behind it, and every light-on number
describes the hardware. The **optogenetic light QC** measures that disagreement for any
experiment type, over the whole recording, whatever window a table uses.

A short tail of light after the last lick is normal: that is the decay. Long stretches of
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
DFM tab has its own *Optogenetics* picker, which inherits by default.

## Program.txt

The MCU writes the program it ran back out as `Program.txt`, echoing every interval as it
read it. Copy that file into the member's `data/` folder, beside the DFM CSVs. pyflic reads
the exported file only, the one with `***DFM 1***` sections, not the program you authored
for the MCU with `[General]` and `[DFM]` sections.

An interval line in the export looks like this:

```
(07/28/2025 12:01:00) Dark Off,20,-1,-1,-1,20,-1,-1,-1,20,-1,-1,-1,F:5160,P:8,D:500,L:1000,M:1000000000000,1440.0min.
```

From it pyflic takes:

- **Thresholds**, one per well in `W1`–`W12` order. A positive threshold makes the well a
  **trigger well**. `0` keeps the light on regardless (open loop), and `-1` means the well
  never triggers.
- **F, P, D, L, M**: frequency, pulse width, decay, delay and max-time-on. A frequency above
  512 carries the progressive ratio's acclimation-event count in its high bits (`5160` is 10
  events at 40 Hz), and a pulse width at or above 32768 selects non-feeding activation.
  Both are decoded.
- **Linkage**: wells sharing a number form one **linkage group**, lit together whenever
  any member triggers.
- **Program type** and the interval timestamps, which lay the intervals out over the run:
  Linear holds its last interval to the end, Repeating and Circadian start over, and
  Constant runs its first interval throughout.
- **Baseline**, the start time and the duration.

Each interval's parameters select a **paradigm**: closed loop, closed loop with a maximum
time on, fixed interval, progressive ratio or non-feeding activation. The semantics are the
MCU/DFM version 2.0 conventions.

**The file is optional.** Without it the light QC still runs, but it is limited. Linkage is
inferred from wells lit together, every lit well counts as a trigger well, and the decay is
taken as `opto_default_decay_ms`. Unexplained light can then warn but not fail, because an
open-loop schedule is indistinguishable from a stuck light without the program.
`summary.txt` and the report say so.

**When it is wrong.** A line pyflic does not understand is skipped and listed in
`summary.txt`. A DFM section that cannot be read is dropped, and that DFM is judged as if
there were no program. A file with no readable DFM section is set aside with the reason.
More than one `Program.txt` in `data/` stops the load, because which one describes this
recording is not a guess to make.

## What the light QC checks

The unit of judgement is the **linkage group**: its light is one circuit. Every verdict is
passed on to each chamber the group touches.

### Explained light

A lit sample is **explained** when one of its group's trigger wells has a feeding lick *or*
a tasting sample from the interval's decay plus `opto_decay_tolerance_samples` (2) before it
to the tolerance after it. Tasting samples count because the firmware decides on contact at
millisecond resolution, and a brief touch it counts often reads below pyflic's feeding
threshold. The tolerance absorbs the difference between those decisions and the recorded
samples.

| Check | Verdict |
|---|---|
| **Unexplained light**: at least `opto_unexplained_fail_fraction` (30%) of the group's lit time unexplained | **fails** the group |
| **Partly unexplained light**: at least `opto_unexplained_warn_fraction` (10%) unexplained | warning |
| **Light while off**: lit while every member's threshold is -1 | **fails** the group |
| **No light recorded**: `optogenetics: yes`, and no LED lit on the DFM at all | **fails** the DFM |
| **Unlit feeding**: in a closed-loop interval, at least `opto_unlit_feeding_fraction` (50%) of the feeding bouts at its trigger wells never lit the group (5 bouts needed) | warning |
| **No light events**: feeding at its trigger wells, and never lit | warning |
| **Open loop not lit**: lit for less than `opto_open_loop_min_lit_fraction` (95%) of an open-loop interval | warning |
| **Program mismatch**: the data's `OptoFreq`, `OptoPW` or `Dark` differ from the program, or the data start more than a minute from its Start Time | warning |
| **No program section**: `Program.txt` has no usable section for the DFM | warning |

Neither unexplained-light fraction is judged until a group has at least
`opto_unexplained_min_sec` (30 s) of unexplained light, so a few stray samples in a group
lit for seconds are not called a fault. Where a group is flagged, `UnexplainedOnsetMin` is
the start of the first 30-minute window in which its unexplained share reached the warning
fraction: the latest a hand-set cutoff could fall.

Light at the start of an interval is left to the interval before it for the length of its
decay. Light outside the program's schedule is not judged, and neither is non-feeding
activation: the light is meant to be on while the fly does *not* feed.

### The emulated firmware trigger

With a program, pyflic also re-runs the firmware's own test on the recorded signal. The
firmware subtracts the mean of the run's first ten seconds and compares each sample with
the interval's threshold. Wherever that **emulated trigger** reads a trigger well above
threshold and pyflic sees no lick or touch, the firmware was seeing contact that pyflic's
running baseline removed: the drifting-baseline signature.

`ContactWithoutActivitySec` is how long that lasted. When most of a group's unexplained
light fell during such contact, `LikelyCause` reads *a drifting baseline or sustained
contact*; otherwise it reads *hardware, linkage or program mismatch*: light the firmware had
no reason to switch on. The emulation is marked approximate (`EmulationApproximate`) when
the data begin more than two seconds after the program, so the baseline window was not
recorded.

## Thresholds

The thresholds are design `constants:`, shared by every experiment type, with fields in the
Project Design dialog and the Config Editor. The Config Editor shows them once the
experiment is known to be optogenetic.

| Constant | Default | What it does |
|---|---|---|
| `exclude_failed_opto_chambers` | false | every chamber of a failed group leaves the analysis through auto-removal |
| `opto_unexplained_warn_fraction` | 0.10 | unexplained share that warns |
| `opto_unexplained_fail_fraction` | 0.30 | unexplained share that fails |
| `opto_unexplained_min_sec` | 30 | unexplained seconds needed before either is judged |
| `opto_default_decay_ms` | 1000 | the decay assumed without a `Program.txt` |
| `opto_decay_tolerance_samples` | 2 | samples of slack around the decay |
| `opto_unlit_feeding_fraction` | 0.50 | unlit share of closed-loop feeding bouts that warns |
| `opto_open_loop_min_lit_fraction` | 0.95 | lit share an open-loop interval needs |

**Failures are flagged, not excluded, by default.** A faulty light does not invalidate the
feeding record, so a failed group stays in every result with its flags on its chambers'
rows (`OptoLightQC`, `OptoUnexplainedFraction`) until you exclude it by hand or switch
`exclude_failed_opto_chambers` on.

## Outputs

Basic analysis and **QC reports** write, into the member's `qc/opto/`:

| File | One row per | Holds |
|---|---|---|
| `opto_light_qc.csv` | linkage group | wells, trigger wells, chambers, treatment, lit and unexplained time, onset, the emulated trigger's contact, `LikelyCause`, `Flags`, `Verdict`, `Excluded` |
| `opto_light_intervals.csv` | group × scheduled interval | the interval's mode, decay and trigger wells; lit, unexplained and contact time; feeding bouts and unlit ones |
| `opto_light_events.csv` | light event | onset, end, duration, interval, `UnexplainedSec`, `Explained`, and `OverrunSec`, the light's time past the last lick plus decay |
| `opto_program.csv` | program interval | the program as read: paradigm, decay, delay, max time on, frequency, acclimation events, linkage, trigger wells, and the raw `F` and `P` the MCU echoed beside their decoding |
| `opto_light_dfm<id>.png` | DFM | the QC figure, below |

`summary.txt` gains an *Optogenetic light QC* section, and the experiment report a
subsection of Quality control with the program table, the verdicts and the figure. The
figure, **Light explained by licks (QC)**, shows each group's lit time per bin, explained and
unexplained, and below it the emulated trigger's contact with no lick. A Project's Combined
Analysis stacks every member's verdicts into `<project>_LightQC.csv` with `Source` = `Opto`.

The Hub's QC panel has an **Optogenetics** group whenever the loaded member is
optogenetic: **Opto light QC table** and **Light explained by licks (QC)**, with the
Progressive Ratio light checks beside them for that type. The Script Editor's actions are
`opto_light_qc` and `plot_opto_light`.

## Progressive Ratio

A [Progressive Ratio](concepts-progressive-ratio.md) experiment gets this light QC as well
as its own, which asks whether the paired fly earned each light event. The program serves
both:

- **Paired chambers from the program.** With a `Program.txt`, a DFM entry may leave
  `paired_chambers` out: in each chamber group the chamber holding the trigger well is the
  paired one. When both are given and disagree, the config wins and `summary.txt` says so.
  A trigger well that is well B is flagged, because the Sucrose Well is always well A.
- **Decay-aware lick-free events.** A light event is lick-free only when it is credited
  with no Sucrose Well licks *and* has no lick or touch within its decay, the program's or
  `opto_default_decay_ms`. A light the fly touched for is never counted against it.
- **A type mismatch is reported.** A program running a progressive ratio under another
  experiment type, or a Progressive Ratio experiment whose program runs none, is noted in
  `summary.txt` and the report.

---

Related: [Light state and phase analysis](concepts-light-phase.md) ·
[Progressive Ratio experiments](concepts-progressive-ratio.md) ·
[Exclusions](concepts-exclusions.md) · [Reports](reports.md)
