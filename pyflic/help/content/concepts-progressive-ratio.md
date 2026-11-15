# Progressive Ratio experiments

A two-well operant assay. The paired fly's feeding at the sucrose well switches on a light
(typically driving neuronal stimulation), and the analysis asks two questions: how far did
the paired fly keep working as the requirement rose, and, where the design has yoked
controls, did it work for the light more than a control that got the same light for free?

Everything here applies to any Experiment Type that includes the **Progressive Ratio**
Analysis Module: the shipped `ProgressiveRatio`, or a type your lab builds with it (see
[Analysis Modules](concepts-experiment-types.md#analysis-modules)). "The type" below means
any of them.

This topic covers what is special about the type. The two-well basics it builds on —
`pi_direction`, the preference index, dual-feeding correction — are in
[Two-well choice and the preference index](concepts-two-well-pi.md). What the light did,
and the general light QC every optogenetic experiment gets, are in
[Optogenetic experiments](concepts-optogenetics.md).

## Chamber groups, paired and yoked

A DFM's chambers form **chamber groups**: the chambers of one light circuit, sharing one
treatment and one training end. In each group one chamber is **paired**: its feeding at the
sucrose well turns the light on. A group is one of two kinds:

- **A pair**: the other chamber is **yoked**, lit at the same moments as its partner,
  whatever it does itself. A comparison between the two is always made *within* a pair,
  never between group means.
- **A single chamber** (a solo group): the paired chamber alone, with no yoked control.
  Its light answers to its own fly only. This is PyMCU's reference design.

Groups are numbered in chamber order, so a DFM of single chambers has groups 1 to 6, each
its chamber. Each DFM entry says which design it ran, in one of three ways:

```yaml
dfms:
- id: 1
  params: {pi_direction: left}     # which side well A (sucrose) is on
  paired_chambers: [1, 4, 5]       # pairs 1+2, 3+4, 5+6: one paired chamber from each
  chambers: {1: Ctrl, 2: Ctrl, 3: Exp, 4: Exp, 5: Exp, 6: Exp}
- id: 2
  params: {pi_direction: left}
  chamber_groups: single           # six single chambers, no yoked controls
  chambers: {1: Ctrl, 2: Exp, 3: Ctrl, 4: Exp, 5: Ctrl, 6: Exp}
- id: 3                            # neither key: the groups come from Program.txt
  params: {pi_direction: left}
  chambers: {1: Ctrl, 2: Ctrl, 3: Exp, 4: Exp, 5: Exp, 6: Exp}
```

- **`paired_chambers`** names exactly one chamber from each of the pairs 1+2, 3+4 and 5+6;
  the other is yoked, and never written.
- **`chamber_groups: single`** makes every chamber its own group. It cannot be stated
  together with `paired_chambers`.
- **Neither key**, with the MCU's `Program.txt` in `data/`: the program's linkage gives the
  groups. A linkage group whose light reaches two chambers is a pair, its paired chamber
  the one holding the trigger well; one whose light reaches a single chamber is a solo
  group. Pairs that are not neighbours, such as 1+4 and 2+3, are read as the program
  pairs them. When the linkage makes no groups (one circuit over three chambers, trigger
  wells in both linked chambers, a chamber no trigger well lights), the pairs 1+2, 3+4
  and 5+6 are tried, with the chamber holding each pair's trigger well as its paired one;
  when that fails too, the DFM does not load, and the message says why.
- **`chamber_groups: paired`**, or neither key without `Program.txt`: the pairs 1+2, 3+4
  and 5+6. `paired_chambers` names their paired chambers, or else `Program.txt`'s trigger
  wells do; with neither, the DFM does not load.

What the config states wins. When `Program.txt` disagrees with it, `summary.txt` says so,
and a trigger well that is well B is flagged too, because the Sucrose Well is always well A.
`chamber_groups` is read from a DFM entry only, never from `global:`.

- **Well A is always the sucrose well** and well B the yeast well, so the type requires
  `well_names` for both. There is no separate side key: `pi_direction` places well A, as
  in any two-well experiment, and may differ per DFM.
- **Both chambers of a pair carry the same treatment.** The loader and `pyflic lint`
  refuse a config where they differ, or where `paired_chambers` names two chambers of one
  pair or leaves a pair out. A design of single chambers needs neither `paired_chambers`
  nor a shared treatment.
- **Paired and yoked are roles, not treatment levels.** Do not declare a `paired` design
  factor; the role comes from the groups and appears as the `Role` column.

In the [Config Editor](app-config-editor.md) each DFM tab has a **Groups** choice — Pairs,
Single chambers or From Program.txt — and, for pairs, three pickers, shown only for a type
with the Progressive Ratio module. See also [Optogenetic experiments](concepts-optogenetics.md#progressive-ratio). Paired
and Yoked are also [Roles](concepts-optogenetics.md#roles) in any optogenetic design; this
type keeps its own `paired` and `yoked`, read from its groups.

### Without yoked controls

A solo group has no partner, so everything that takes paired minus yoked skips it: the
paired − yoked difference and the cumulative difference curve. Its `YokedChamber` is `NA` in
the training table and `pr_light_qc.csv`. The breaking point, sucrose persistence and the
light QC run as for any group.

- **Every group single**: the report says once, where the difference would have been, that
  the design has no yoked controls, and shows the breaking point. The Project Report says
  the same, and the Project's Stats text reads *Paired − yoked difference: none; the design
  has no yoked controls*. The training table drops its empty *Yoked chamber* column.
- **Pairs and single chambers mixed** (one DFM of each, say): the comparison uses the
  pairs, and the report names the single-chamber groups it left out.
- **Exclusion**: a solo group leaves the analysis when its chamber does. A pair still needs
  both of its chambers for anything that compares them.

The per-DFM QC figures put three groups to a row, so six single chambers take two rows.

## Both ratio Paradigms

From DFM firmware 4.0.0 the program names each interval's Paradigm. This type covers both
ratio Paradigms:

| Paradigm | Each light event needs |
|---|---|
| **ProgressiveRatio** | more time in contact each time, along its series (Arithmetic or Exponential), up to its `Cap`; with `Counter: Events`, more feeding bouts |
| **FixedRatio** | the same time in contact every time, its `Ratio`; with `Counter: Events`, the same number of feeding bouts |

What a ratio Paradigm counts is its **Counter**, which `Program.txt` gives on its line:
`Contact`, milliseconds of contact (PyMCU 4.0.1's default, on DFM firmware 4.0.1), or
`Events`, feeding bouts (the only kind PyMCU 4.0.0 wrote, which leaves the Counter out).

A FixedRatio experiment is analysed with the same machinery; see
[Fixed ratio](#fixed-ratio). Without `Program.txt` the Paradigm's type is unknown, and the
Test is taken as progressive.

## Training

The recording opens with a closed-loop **training** phase in which every sucrose-well feed
of the paired fly turns the light on. Where the recording says so depends on its
[Generation](concepts-optogenetics.md#generations):

| Generation | Training is read from | A group's training end |
|---|---|---|
| 1 (older firmware, 27 columns) | a flag inside the signal: while a well is in training its raw value is offset by 65536 (any sample above 40000) | the last flagged minute of the paired chamber's sucrose well; a light event that begins on the row that clears the flag is the last Training pairing, not the first Test one |
| 2 (PyMCU 4.0.0 with an older DFM) | nowhere | none: see [No training information](#no-training-information) |
| 3 (firmware 4.0.0) | the paired sucrose well's `Phase` | its first row in Phase `M` (Main) under a ratio Paradigm |

A group's training end differs between groups, because it depends on the fly, and every
Progressive Ratio time axis runs from it rather than from the start of the recording.

**On generation 1** the group's other wells (three in a pair, one in a single chamber) are
expected to clear at the same minute. When they do not, `summary.txt` and the report list
the disagreement as a QC note, and the analysis proceeds on the paired well.

**On generation 3** Training is the paired sucrose well's Acclimation (Phase `A`), together
with any interval before the ratio Paradigm: a ClosedLoop interval that comes first is part
of Training. The group's other wells are `Off` and read `M` throughout, so they carry
nothing to disagree about. The notes speak of Phase instead:

- A ratio Paradigm with no Acclimation and no interval before it leaves nothing to call
  Training. Training is then empty and complete, and the notes say *Training was empty: the
  ratio Paradigm had no Acclimation and no Interval before it*.
- A paired sucrose well that never ran a ratio Paradigm, was still in Acclimation when the
  recording ended, or never reached `M` under a ratio Paradigm never completed Training,
  and the note says which.
- A later run of Acclimation on the same well is noted as *Paradigm restarted at Program
  Interval k*. The firmware starts every well afresh when the Paradigm's name changes, and
  after the DFM restarts on its own, when the note adds *after the DFM restarted at minute
  m*.

**A ClosedLoop interval needs a lab type.** The shipped `ProgressiveRatio` expects only the
ProgressiveRatio and FixedRatio Paradigms, with OpenLoop and dark intervals beside them, so
a program with a lit ClosedLoop interval (before the ratio Paradigm or after it) does not
load under it. For such a program, build a type in the
[Experiment Type Editor](app-type-editor.md) that expects ClosedLoop as well and includes
the Progressive Ratio module (see
[The light a type expects](concepts-experiment-types.md#the-light-a-type-expects)).

A group whose paired fly never finishes training has `TrainingComplete = false` and no
`TrainingMinutes` on every chamber of the group. With the default
`require_training_complete: true`, they leave the analysis through auto-removal.

### No training information

A generation-2 recording comes from a DFM whose firmware is older than 4.0.0, recorded by
PyMCU 4.0.0. It carries no training flag and no Phase, so it has no training information at
all. Such a DFM is analysed, not removed:

- each chamber group gets one Facet, `Recording`, which is never pooled into `Test`;
- `TrainingComplete` is `NA`, and `require_training_complete` removes nothing;
- there is no Test phase, so no Breaking Point and no Sucrose Persistence;
- *no training information: recorded with DFM firmware older than 4.0.0* is said in
  `summary.txt`, on the report's cover, on the Hub's **Loaded** line, in the Project
  Report's member table and in the Stats text's flagged block.

## Training and Test facets

This type's [Facets](concepts-facets.md) are `Training` and `Test`, split at each
group's own training end rather than at a fixed minute, so the config never states
`facet_cutoffs` or `facet_source`; the Project design dialog shows its Facet source,
cutoff and phase-name fields as derived and read-only, and the Config Editor its Facet
source. Every table and plot that reads the `Facet` column works
unchanged; `StartMin` and `EndMin` simply vary from group to group. A DFM with no training
information has the one Facet `Recording`.

## Light QC

*Did the paired fly earn its light?* The firmware lights a group from its own reading of
the paired chamber's sucrose well during the run; pyflic counts licks afterwards, from the
baselined signal. A sucrose well whose raw **resting level** creeps up looks continuously
touched to the firmware and flat to pyflic, so the light fires on its own schedule and
nothing about it was earned. Every chamber group is therefore checked, over the whole
recording whatever window a table uses.

A **light event** is one onset of the group's light. On generations 1 and 2 it is an onset
in `OptoCol1`. On generation 3 it is an **Activation** of the paired sucrose well, read from
its `Activations` count: the firmware's own record of each light it earned.

| Check | Verdict |
|---|---|
| **Self-triggered light**: at least `pr_lick_free_run` (5) consecutive Test light events with no sucrose licks since the previous one | **fails** the group |
| **Implausible training**: training completed with light events but not a single sucrose lick | **fails** the group |
| **Requirement series wrong** (generation 3): a Test light event met a Requirement the Test Paradigm's series in `Program.txt` does not give | **fails** the group |
| **No increasing trend**: with at least `pr_trend_min_events` (5) Test light events inside the Test window, Spearman's rho of licks per event against event number is below `pr_trend_min_rho` (0.3) | warning |
| **Resting level rise**: the sucrose well's resting level peaks at least `pr_resting_level_rise` (15) counts above its first 30 minutes | warning |
| **Resting level elevated**: the sucrose well rests at `pr_resting_level_ratio` (3) times the DFM's other sucrose wells, and at least `pr_resting_level_rise` counts above them | warning |

A few **lick-free light events** are normal — pyflic's feeding threshold misses brief
touches the firmware counts — which is why a *run* of them, not a single one, fails a group,
and why training fails only on zero licks rather than on fewer licks than pairings. A light
event is lick-free only when it is credited with no sucrose licks *and* has no sucrose lick
or touch within its light's Hold (the one `Program.txt` states, or `opto_default_decay_ms`):
a light the fly touched for is never counted against it. A group with no Test light events
at all has simply stopped before its first Test requirement: a breaking point of 0,
reported, never flagged.

**A second witness on generation 3.** The firmware also records the feeding bouts it
counted for each light event, its **Firmware Events**. Their trend is measured beside the
licks' (`TrendRhoFirmware`). The warning stays on pyflic's licks. When the Firmware Events
rise and the licks do not, the group gets a note instead: *the firmware counted rising
Events per light event; pyflic's licks did not follow: check detection*. Under a FixedRatio
Test the requirement never rises, so no trend is judged, and the note says so. Under
`Counter: Contact` the firmware counts milliseconds, not bouts, so there are no Firmware
Events: `TrendRhoFirmware` is NaN and only the licks' trend is judged.

**The series, on generation 3.** The firmware records the Requirement each light event
met, and it keeps its program's series exactly: under a ProgressiveRatio the Requirements
climb as the series gives them, under a FixedRatio each is the `Ratio`. So the Test
window's light events are checked against the Test Paradigm's series in `Program.txt`, with
the optogenetic light QC's own check (*requirement series wrong*). One that differs fails
the group, because it cannot happen with the right program: `Program.txt` is probably not
the program that ran. The check forgives one case the rows cannot settle, a light event
counted in the very record the Paradigm began, which may belong to the Paradigm before; a
run whose Requirements match the series from the next event is not a mismatch. It needs
`Program.txt`, and the type of the Test Paradigm.

With the default `exclude_failed_pr_groups: true`, every chamber of a failed group (both of
a pair, the one of a single chamber) leaves the analysis through auto-removal, with the
reason in `removed_chambers.csv`. Switch it off to keep them. `pr_light_qc.csv` lists every group's verdict either way, with
`LickFreeRunStartMin` — minutes after training end at which the first failing run began —
for deciding a cutoff by hand. The QC figures keep a failed group in view, its panel strip
saying why; the result figures and tables no longer contain it.

The Hub's QC panel has the **Light QC table** and the two figures behind the verdict, in its
*Progressive Ratio* group, which **Run QC** writes while they are ticked; see
[Plot catalogue](plots-catalog.md#did-the-paired-fly-earn-its-light-qc).

This check asks whether the paired fly *earned* each light event. Every optogenetic
experiment, of any type, also gets the general light QC, which asks whether the light was
*explained* by licks at all, from lit time rather than events: see
[Optogenetic experiments](concepts-optogenetics.md). In a pair, that check also confirms
the yoking: a yoked chamber's wells must count no Activations of their own and show exactly
the paired sucrose well's light.

**A DFM that restarted.** A DFM can restart on its own mid-recording: records are lost, a
few rows run under no instruction, and the firmware starts every well's Paradigm afresh,
the ratio from its first Requirement. When that happens inside a group's Test window, its
`pr_light_qc.csv` `Notes` say so: *the DFM restarted at minute m, d min after training end:
the Test window ends at the restart, because the ratio starts again after it, so a breaking
point still running there is censored*. See [The Test window](#the-test-window).

## Breaking point

*When did the paired fly stop working for the light?* Responses are read in order from the
chamber group's training end, and the fly is taken to have stopped at its **first pause
longer than `pr_break_gap_min`** (default 120 minutes). The **breaking point** is the number
of lick-backed Test light events the paired fly completed before that pause: the last ratio
it met. `BreakMin` is the minute, since training end, of the last one counted.

- **The pauses run from training end to the end of the Test window**: training end to the
  first event, event to event, and the last event to the window's end. A fly whose first
  Test event came more than `pr_break_gap_min` after training has a breaking point of 0; one
  that stopped well before the recording did is seen to stop. A pause of exactly
  `pr_break_gap_min` does not end the count.
- **Lick-free light events are ignored.** One neither counts nor ends a pause: if the fly
  did not lick, the light is no evidence it was still responding.
- **Censored.** A group with no such pause before its Test window ended was still
  responding when the recording stopped. Its count is a lower bound, not a measurement:
  written `n+` in tables, drawn open on the dot plot and as a tick on the still-responding
  curve.
- **Paired only.** In a pair the yoked fly has no breaking point, because its light is its
  partner's. Paired and yoked are compared on feeding metrics instead, sucrose persistence
  among them. A single chamber's breaking point is its paired fly's, like any other.
- **The unit is light events**, not licks.

### The recorded Requirement

A generation-3 recording states the **Requirement** each light event met: the milliseconds
of contact the firmware demanded for it under `Counter: Contact`, or how many Firmware
Events under `Counter: Events`. The breaking point is still the count of
lick-backed light events, because a light with no lick is still the drifting-sensor
signature. The Requirement is reported beside it:

| Column | What it holds |
|---|---|
| `BreakRequirement` | the Requirement the last counted light event met |
| `FirmwareEvents` | the Firmware Events credited to the counted light events |
| `ParadigmType` | the Test Paradigm's type, `ProgressiveRatio` or `FixedRatio` |
| `LargestRequirement` | on generation 1, the most sucrose licks credited to one counted event, a description only; on generation 3, the same as `BreakRequirement` |

`BreakRequirement` and `FirmwareEvents` are `NA` before generation 3, and `FirmwareEvents`
is `NA` under `Counter: Contact` too. The reports show `BreakRequirement` beside the
breaking point.

### Fixed ratio

Under a FixedRatio Test Paradigm every light event needs the same number of feeding bouts.
The same first-gap rule runs, and what it counts is the activations the fly earned before
its first long pause. `BreakRequirement` is the constant `Ratio`, the trend check is
skipped, and the report's section is titled **Fixed ratio: activations earned**. The Project
Report uses that title when every pooled group ran FixedRatio.

What only a climbing requirement makes sense of is left out for a fixed-ratio group: the
estimated requirement increment in `summary.txt`, the grey estimated-requirement line on
*Licks per light event*, and that figure's caption "a working progressive ratio climbs". A
recording that mixes the two shows them for its progressive groups only.

### The Test window

The span the breaking point is judged over runs from each group's training end to the end
of the recording, so it differs between groups — a group that trained late is watched for
less time.

On a generation-3 recording the Test window ends where the ratio Paradigm's run ends, not at
the end of the recording. Only the first ratio run is the Test. A ClosedLoop interval after
it (under a lab type that expects ClosedLoop) is no part of the Test, and a later ratio run is noted (*a ratio Paradigm ran again from
Program Interval k, after the Test window*) and left to
[Program Facets](concepts-facets.md#program-facets).

**A DFM restart ends the Test window.** After a restart the firmware starts the ratio again
from its first Requirement, so what follows is not the same Test. When a DFM restarts inside
a group's Test window, the window ends at the last row before the restart, the first-gap
rule sees it only up to there, and a fly still responding at that point is **censored**.
`TestMinutes` is the shortened window, and the note naming the restart is in
`pr_light_qc.csv`'s `Notes`, in `summary.txt` beside the breaking point, and under the
report's breaking point. `pr_breaking_point.csv` gains no column for it.

`pr_test_window_min` caps every group's window at the same length, after that. It is off by
default, and censoring is judged against the capped window. `TestMinutes` in
`pr_breaking_point.csv` is each group's window. The `Test` facet itself is never capped.

### Choosing the gap

The rule does not know the time of day: a long pause at night ends a count like any other.
The number moves with the gap, so `summary.txt`, and the log of **Run analysis** with
*Breaking point* ticked, always tabulate every group's breaking point at 60, 120 and 240
minutes beside the configured value. Read that table before settling on a gap, and set it once, in the Project's design,
so every member is analysed under one rule.

## Sucrose persistence

The breaking point's rule applied to feeding: for **either** fly, the minutes from training
end to its last sucrose-well feeding event before a pause longer than `pr_break_gap_min`,
over the same Test window. In a pair it is the one persistence measure the yoked fly has
too, so it has a paired − yoked difference.

It appears as `PersistA` and `PersistACensored` on every per-chamber summary row (blank on
Training rows), and as `dPersistA` in `paired_yoked_diff.csv`, with `dPersistCensored`
true when either fly was still feeding when its window ended. Such a difference is kept and
flagged rather than dropped: in a recording that ends during a feeding peak, dropping them
would remove most groups.

## Outputs

Basic analysis writes, into the member's `analysis/`, the files below. The Hub's **Run QC**,
**Run analysis** and **Generate plots** write the same files, each its ticked items (see
the [Analysis Hub](app-hub.md#analyze-panel)); the cumulative difference table and curves
keep one-minute bins whatever the Bin size.

| File | One row per | Holds |
|---|---|---|
| `feeding_summary.csv` | chamber | the two-well columns plus `Group`, `Role`, `TrainingMinutes`, `TrainingComplete`, `LightOn_sec`, `LitSec`, `Activations`, `LightQC`, `LickFreeLightEvents`, `PersistA`, `PersistACensored` |
| `feeding_summary_facet.csv` | chamber × Facet | the same, per group's own Training and Test windows (or `Recording`) |
| `paired_yoked_diff.csv` | pair × Facet | paired − yoked `dLicksA/B`, `dEventsA/B`, `dPI`, `dEventPI`, `dMedDurationA/B`, `dPersistA`, `dPersistCensored`, with `PairedChamber`, `YokedChamber`, `TrainingMinutes`; no row for a single chamber, so a design without yoked controls writes the header only |
| `pr_breaking_point.csv` | chamber group | `BreakingPoint`, `BreakMin`, `Censored`, `TestMinutes`, `LargestRequirement`, `BreakRequirement`, `FirmwareEvents`, `ParadigmType`, `LickFreeLightEvents`, `LightQC` |
| `pr_light_qc.csv` | chamber group | `PairedChamber`, `YokedChamber` (`NA` for a single chamber), the light QC's counts, trend (`TrendRho`, and `TrendRhoFirmware` on generation 3), resting level, `Flags`, `Verdict`, `Excluded`, and `Notes`, which name a DFM restart that ended the Test window |
| `pr_light_events.csv` | Test light event | the paired chamber's light event ledger: `MinutesSincePrev`, `LicksSincePrev`, `Explained` (a sucrose lick or touch within the light's Hold), `LickFree`, `RestingLevel`, `Counted` (whether the breaking point holds it); on generation 3 also the `Requirement` it met, the `FirmwareEvents` since the previous one and its `Phase` |
| `light_summary.csv`, `light_wells.csv` | chamber, well | the [light table](concepts-optogenetics.md#the-light-table) |
| `pr_cumulative_diff.csv` / `.png` | time bin | the cumulative difference curve |
| `pr_still_responding.png` | — | the still-responding curve |
| `pr_*_dfm<id>.png` | DFM | QC figures: training-aligned traces, licks per light event, sucrose well resting level, a panel per group, three to a row |
| `summary.txt` | — | adds the training table, the training notes, the light QC and the breaking point with its sensitivity to the gap, and any DFM restart that ended a Test window |

**Generate plots** adds `breaking_point_dfm<id>.png` while *Breaking-point plots* is
ticked: per DFM, one panel per chamber, its licks per light event against minutes since
training end, with the group's breaking point marked. Basic analysis does not write it.

`LightOn_sec` is the seconds the chamber's light was on. On generation 3 it is exact, from
the larger of its two wells' `OnMs` record by record; before, it counts the lit `OptoCol1`
samples. `summary.txt` names the source for each DFM.

In the ledger, `FirmwareEvents` counts the Firmware Events from the previous light event to
this one, so under a ratio Paradigm's Main phase it equals the `Requirement` met. An
Acclimation light event meets no Requirement, and its `Phase` is `A`.

Lick tables obey `transform_licks` as in every type, and with it the paired − yoked
`dLicksA/B` are differences on the fourth-root scale, labelled so in the reports; the
cumulative curves always use raw lick counts. A Project's Combined Analysis stacks the difference table, the light QC and the
breaking point into `<project>_PairedYokedDiff.csv`, `<project>_LightQC.csv` and
`<project>_BreakingPoint.csv`. The figures themselves are described in the
[Plot catalogue](plots-catalog.md#progressive-ratio-plots).

## Statistics

One observation per chamber group, treatment fixed and, in a Project, DFM nested within
Experiment. The difference table is primary; the per-chamber tables are secondary. A design
without yoked controls has no difference to test, and the breaking point is its result.

- **Is the paired fly different from its yoked partner?** Each treatment's paired − yoked
  differences are tested against zero: the paired t-test (a one-sample t-test on the
  differences) with the Wilcoxon signed-rank test beside it and, in a Project, the mixed
  model's intercept.
- **Do treatments differ in that difference?** Welch's t-test for two treatments, Tukey HSD
  for more, and the mixed model when members are pooled.
- **Do treatments differ in breaking point?** The same tests, which enter a censored count
  as observed, plus a pairwise **log-rank test** on the ratio reached, which treats it as the
  lower bound it is. The still-responding curve draws the same thing.

See [Reports](reports.md) for where each table appears.

## Requirements

The type needs to know where Training ends and when the light came on:

- **Generation 1** needs version-3 DFM files with light data (`OptoCol1`): the training
  flag and the light state both live there. A version-2 file still loads, but no group ever
  finishes training — `summary.txt` notes "no training flag on any well" for each — so with
  `require_training_complete` on, every group is auto-removed and there is nothing to
  analyse.
- **Generation 2** loads and is kept, with one `Recording` Facet per group and no Test (see
  [No training information](#no-training-information)).
- **Generation 3** reads everything from the file's own Paradigm columns. Copy the
  `Program.txt` PyMCU exported into `data/` as well: without it the Test Paradigm's type is
  unknown (it is taken as progressive), the series is not checked, and the chamber groups
  must be stated in the config.

A member analysed before the breaking point existed has no `pr_breaking_point.csv`; re-run
its analysis (**Run analysis** with *Breaking point* ticked, or **Analyze all**), and the
Project Report says so until you do.

---

Related: [Experiment types](concepts-experiment-types.md) ·
[Optogenetic experiments](concepts-optogenetics.md) ·
[Configuration file structure](config-structure.md#globalconstants) ·
[Plot catalogue](plots-catalog.md#progressive-ratio-plots) · [Reports](reports.md)
