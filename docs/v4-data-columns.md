# The firmware 4.0.0 data columns, and what pyflic does with them

DFM firmware 4.0.0 or 4.0.1, recorded with PyMCU 4.0.0 or later, adds 63 columns to
every DFM data file, and PyMCU 4.0.1 one more, `BaselineSet`. They record what each well's light program was doing, every 200 ms:
which step of the program was running, whether each fly was still in
Acclimation, how many light periods it had earned, and exactly how long its LED
was on. This page explains the new columns and where they show up in pyflic.

For the full column-by-column reference, see
[Firmware 4.0.0 data in pyflic](firmware-4-data-in-pyflic.md). For how to read
the QC results, see [Checking your data](qc-user-guide.md).

---

## Which kind of file do you have?

pyflic decides this from each DFM file when it loads, and names it in
`summary.txt` and on the report's cover as the file's **Generation**.

| Generation | What it is | What pyflic can do |
|---|---|---|
| **1** | 27 columns, from PyMCU 3.x | Everything it did before. Progressive Ratio training is read from a flag inside the signal. |
| **2** | 90 columns (91 from PyMCU 4.0.1), the Paradigm ones all `NA`: the new PyMCU with a DFM that still runs older firmware | Feeding analysis as usual. There is **no training information** at all, so a Progressive Ratio experiment is analysed as one "Recording" phase and says so. |
| **3** | 90 columns (91 from PyMCU 4.0.1), filled: firmware 4.0.0 or 4.0.1 | Everything below. |

Feeding itself is still measured from the well signals, `W1`–`W12`, exactly as
before. The new columns describe the light.

---

## The new columns

Each per-well column has one copy per well, numbered 1–12 like `W1`–`W12`.

| Column | What it records | What pyflic uses it for |
|---|---|---|
| `Interval` | Which step of the program was running: 1 for the first, and so on. `0` marks the few seconds of baseline before the program reached the DFM. `NA` marks rows under no program step, such as the few seconds after a DFM restarted. | Splitting the recording by program step: the light QC judges each step by its own rule, and **Program Facets** can split every result by step. Also whether each step started on time. pyflic's own tables call it `ProgramInterval`. |
| `Cycle` | Which pass through a Repeating or Circadian program. | Telling the passes apart, when you ask for one Facet per pass. |
| `Paradigm` | The name of the light rule in force, as written in the program. `NA` on the baseline rows (`Interval` 0) and on rows under no program step. | Looking up the rule's type and settings in `Program.txt`, and so which checks and measures apply. Labels the Program Facets and the Program timeline. |
| `Phase` | `A` while a well is in Acclimation, `M` (Main) after. | **Progressive Ratio training end**: the paired sucrose well's first Main row under the ratio rule. Also the light table's Acclimation columns and a QC check that each fly got the number of Acclimation light periods it should. |
| `Activations` | A running count of the light periods each well's own rule started. A well lit only through its partner never counts one. | **Light Events** in Progressive Ratio; the light table's `Activations`; the light QC, which asks whether each light period had a lick behind it and kept its rule; roles (a yoked well has light but no Activations). |
| `Requirement` | Under a ratio rule, what the well needs for its next light period: milliseconds of contact under `Counter: Contact` (PyMCU 4.0.1's default), or feeding bouts under `Counter: Events`. `Program.txt` gives each ratio Paradigm's Counter; a file without one (PyMCU 4.0.0) counted bouts. | The ratio each light period met, which a Progressive Ratio breaking point now reports as `BreakRequirement`, and which the light QC checks against the program's series. |
| `Progress` | Under a ratio rule, how much the well has done toward that Requirement, in the same units. | Under `Counter: Events`, **Firmware Events**: the feeding bouts the firmware counted, compared with pyflic's own count of feeding events as a QC check. Under `Counter: Contact` there are none, and pyflic's Firmware Events columns are `NA`. |
| `OnMs` | How many of the record's 200 ms the well's LED was on, from any cause. | **Exact light time**: `LitSec` in the light table, the `OptoOn_sec` columns of the feeding summary, and the lit time the light QC judges. With `Activations`, it times each light period to the millisecond, so pyflic can check its length against the program. Older files can only sample the light once every 200 ms. |
| `BaselineSet` | PyMCU 4.0.1, after `OnMs`: which set of baselines the row's thresholds were measured from, a row of the recording's `Baselines.csv` (copy it into `data/` beside `Program.txt`). | Each row's **Baseline in force**: the light QC's emulated trigger on older firmware, and telling a drifting baseline from sustained contact. pyflic drops the column at load and keeps the baselines on the DFM. |

A value that does not apply is written `NA`: for example `Requirement` and
`Progress` on a well that is not under a ratio rule. `NA` means "not
applicable", not lost data.

---

## Original columns that now mean slightly more

| Column | What changed |
|---|---|
| `W1`–`W12` | Always the plain signal. Older files added 65536 to a well in training; the new firmware reports training in `Phase` instead. |
| `Dark`, `OptoFreq`, `OptoPW` | Taken from the program step that was running. The light QC checks them against `Program.txt`. |
| `OptoCol1` | Still the LEDs lit at the instant of each record. On firmware 4.0.0 pyflic measures light from `OnMs` instead; `OptoCol1` remains the source for older files. |
| `Error` | Labelled by generation in the integrity report: on firmware 4.0.0, 0x01 is *No Ack* and 0x02 is *Overflow*. |
| `Index` | The DFM's record counter. A gap is records lost on the way to the MCU; the integrity report counts them as **lost records**. A gap followed by rows with `Interval` `NA` is a DFM that **restarted**: the integrity report names it, and the light QC warns. |

---

## `Program.txt` still matters

The data file names each row's light rule, but not its type, its settings, the
threshold of each well or which wells are linked. Those are only in the
`Program.txt` the MCU writes into the recording folder. **Copy it into the
experiment's `data/` folder.**

Without it, pyflic still reads Phase, Activations and light time, and works
out which wells trigger and which are linked from the data. It cannot tell
which rule each step followed, so results that depend on the rule are `NA`, and
pyflic warns in `pyflic lint`, in `summary.txt`, on the report and in the Hub.

A recording started without a program (`***ALL DFM***` at the top of
`Program.txt`, every well Off) is analysed as an ordinary feeding experiment.

---

## Where you see it

pyflic groups the six light rules into four **families**, because the
firmware lights them by the same mechanism, and checks and summarises each
family its own way:

- **open loop**: OpenLoop, and any well set `On`; lit whatever the fly does;
- **closed loop**: ClosedLoop, and the Acclimation that FixedInterval and the
  ratio rules can begin with; lit by contact, for the contact and its Hold;
- **fixed length**: FixedInterval, ProgressiveRatio and FixedRatio after
  Acclimation; each earned light period lasts exactly `ActivationMs`;
- **non-feeding**: NonFeeding; dark while the fly feeds.

How the light is checked and summarised follows the rules in `Program.txt`,
whatever the experiment type. The experiment type says only which rules it
expects: Custom takes any, Hedonic none, and Progressive Ratio its two ratio
rules, with OpenLoop allowed beside them. A recording whose `Program.txt`
runs a rule its type does not expect stops loading and names the rule; for
such a design, build a type that expects that light in the Experiment Type
Editor.

Where each column shows up:

- **The feeding summary** (`analysis/feeding_summary.csv`) of an optogenetic
  experiment gains `Role`, `LitSec` and `Activations` for each chamber.
- **The light table** (`analysis/light_summary.csv` per chamber,
  `light_wells.csv` per well) holds what the light did: lit time, Activations,
  Acclimation, Firmware Events, the last ratio met, and each family's own
  measures, such as the median light period and the cut-offs (closed loop),
  the reward efficiency and the return after the lockout (FixedInterval), the
  suppressions (NonFeeding), and feeding in the light against feeding in the
  dark (open loop). It carries the same Treatment and factor columns as the
  feeding summary, so it pools, plots and is compared between treatments the
  same way. Basic analysis writes it, as does the Hub's *Light tables* box
  under **Run analysis**. A Project stacks it into
  `<project>_LightSummary.csv`.
- **Reports** gain a *Light behaviour* section, with one part per family the
  program runs, each comparing that family's own headline measures between
  treatments. Only the chambers that earn their own light are compared on
  Activations: a yoked chamber keeps its lit time beside its partner's. The
  Plot Editor offers every measure as a publication figure.
- **Paired and yoked designs**, under an experiment type with the *Paired
  and yoked* analysis, which Custom has: where a light circuit lights a yoked
  chamber with a paired one, the paired fly's feeding minus its yoked
  partner's, by treatment, in `analysis/yoked_comparison.csv`,
  `<project>_YokedComparison.csv` and the reports' *Paired and yoked*
  subsection. Progressive Ratio has its own Paired-Yoked Difference instead.
- **Progressive Ratio** reads training, light events and the ratio met
  straight from the recording. `pr_light_events.csv` gains `Requirement`,
  `FirmwareEvents` and `Phase`; `pr_breaking_point.csv` gains
  `BreakRequirement`, `FirmwareEvents` and `ParadigmType`. The breaking point
  rule itself is unchanged. Fixed ratio programs, designs without yoked
  controls, and chamber groups taken from the program's linkage are supported
  too.
- **Program Facets.** Set `facet_source: program` in the Project design (or
  a standalone experiment's config), and every summary is split by program
  step, each labelled with its rule's name.
- **The optogenetic light QC** judges each rule by its own logic, one light
  period at a time, from the firmware's own record, with the program's
  settings: a light period's length against its Hold, `ActivationMs` or
  `MaxOnMs`, the FixedInterval lockout, the ratio series, the NonFeeding light
  after contact, each open-loop step, and each step's start time. It also
  checks for yoked wells whose light differs, a rule name missing from
  `Program.txt`, an Acclimation count that is off, Firmware Events that
  disagree with pyflic's events, a DFM that restarted, and lost, late or
  unknown records. The report has a *Rule check* subsection, with a figure,
  for each rule the program ran. `qc/opto/opto_light_events.csv` lists every
  light period with its timing to the millisecond and what ended it.
- **The Program timeline** (`qc/program/`, the report's Quality control, the
  *Program timeline* box of the Hub's QC panel, run by **Run QC**, and the QC
  Viewer's *Program* tab) draws each DFM's program
  steps, and for each well its Acclimation, its Activations, its light, and
  the steps that set it `On` or `Off`; a restart is a red band.
- **The integrity report** counts lost, repeated, late and unknown-instruction
  records, lists each DFM restart, and lists `Messages.csv`'s errors.

---

## Worth knowing

- **The baseline rows** at the start (`Interval` 0, `Paradigm` `NA`, about
  five seconds) count toward whole-recording feeding totals, but toward no
  program step, no light judgement and no light-or-dark feeding rate.
- **Activations are a well's own; `OnMs` is the light it received.** In a
  paired and yoked design, the yoked fly's wells show light with zero
  Activations. pyflic uses that to tell the roles apart, and to check the
  yoking.
- **The Requirement a light period met is on the row before it.** The firmware
  moves to the next Requirement in the same millisecond it starts the light,
  so the row where `Activations` rises already shows the next one. pyflic reads
  the row before.
- **`OnMs` above 200** means a record arrived late. A few are harmless; many
  are flagged as *late records*.
- **Nothing here changes how feeding is detected.** Licks and feeding events
  come from the well signals and your detection parameters, on every
  generation.
