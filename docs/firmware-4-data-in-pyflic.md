# Firmware 4.0.0 data in pyflic

What pyflic does with each column of a DFM data file, by **Generation**, and
where each one surfaces. For a developer or an analyst who needs to know which
output a column feeds, or why a column is `NA`.

The file formats themselves are PyMCU's to define. This document says only what
pyflic reads and how; where it and PyMCU's documents disagree about the file,
PyMCU's win:

| What | Where |
|---|---|
| Every data-file column, its missing-value rule, when it is meaningful | [`PyMCU/docs/DataFileColumns.md`](../../PyMCU/docs/DataFileColumns.md) |
| The Program file, the six Paradigm types, their settings and defaults, the `Program.txt` export | [`PyMCU/docs/ProgramFormat.md`](../../PyMCU/docs/ProgramFormat.md) |
| The decision and its reasons | [ADR-0016](adr/0016-the-data-file-is-the-authority-on-what-the-light-did.md), and the addenda to [ADR-0014](adr/0014-progressive-ratio-breaking-point-is-the-first-gap-rule.md) and [ADR-0015](adr/0015-the-opto-program-and-a-light-qc-for-every-type.md) |
| The per-Paradigm light QC, summaries and reports built on it | [ADR-0017](adr/0017-paradigm-families-not-experiment-types.md) and ADR-0015's last addendum |
| The light each Experiment Type expects | [ADR-0019](adr/0019-experiment-types-are-definitions-a-lab-builds.md) |
| The two plans, whose numbered decisions code comments and tests cite, and where the build departed from them | `v4-data-plan.md` and `paradigm-analyses-plan.md`, removed once built; their text is at commit 82aace9 ([Removed plans](README.md#removed-plans)) |
| The vocabulary (Generation, Paradigm, Paradigm Family, Program Interval, Activation, Requirement, Progress, Firmware Event, Phase, Acclimation, Hold, Role) | [../CONTEXT.md](../CONTEXT.md) |

The PyMCU links assume its checkout sits beside pyflic's, on branch `v4_Dev`.
For a short overview of the new columns and where they show up, see
[The firmware 4.0.0 data columns](v4-data-columns.md).
For the researcher's view of the same features see the help topic
*Optogenetic experiments and the light QC* and
[Checking your data](qc-user-guide.md).

- [Generations](#generations)
- [The recording folder](#the-recording-folder)
- [Loading](#loading)
- [The original 27 columns](#the-original-27-columns)
- [The 63 new columns](#the-63-new-columns)
- [Program.txt](#programtxt)
- [Messages.csv](#messagescsv)
- [DFM restarts](#dfm-restarts)
- [What pyflic derives](#what-pyflic-derives)
- [The light QC's own checks](#the-light-qcs-own-checks)
- [The Activation ledger and the rule checks](#the-activation-ledger-and-the-rule-checks)
- [Outputs by file](#outputs-by-file)
- [Where the outputs are made](#where-the-outputs-are-made)
- [To confirm on the first real recording](#to-confirm-on-the-first-real-recording)
- [Not read](#not-read)
- [Not built](#not-built)

## Generations

No file states its firmware or PyMCU version. Each DFM's concatenated files are
classified once at load (`paradigm_state.detect_generation`), from `Phase1`,
the one Paradigm column the firmware fills on every row:

| Generation | Constant | Written by | Header | `Phase1` | `DFM.training_info` |
|---|---|---|---|---|---|
| 1 | `GENERATION_FLAG` | PyMCU 3.x, any older firmware | 27 columns | absent | `flag` |
| 2 | `GENERATION_EMPTY` | PyMCU 4.0.0, DFM firmware before 4.0.0 | 90 or 102 columns | every value `NA` or blank | `none` |
| 3 | `GENERATION_PARADIGM` | PyMCU 4.0.0, DFM firmware 4.0.0 | 90 or 102 columns | any `A` or `M` | `phase` |

102 columns are the released PyMCU 4.0.0: the 90, then `Baseline1`–`Baseline12`
([The Baseline columns](#the-baseline-columns)). They don't change the
generation; 90-column files are 4.0.0 development builds from before 1 November
2026.

`DFM.generation` carries it, and `with_params` copies it. The `version` (2 or
3) that the file name gives stays as a separate axis. The generation is
written to `summary.txt` (*DFM details*), the integrity report, the report
cover's **Generation** line, the `Generation` column of the light table and of
`opto_light_qc.csv`, and the Project Report's member table.

A generation-2 DFM has no Training, and the note saying so has one wording,
`paradigm_state.NO_TRAINING_INFORMATION`: *no training information: recorded
with DFM firmware older than 4.0.0*. It appears in a Progressive Ratio member's
`summary.txt` and report, in the Progressive Ratio light QC's note
(`pr_light_qc`), on the Hub's Loaded line, in the Project Report's member table
and in the Stats text. `light_summary` recognises it by its start, "no training
information".

## The recording folder

| File | pyflic |
|---|---|
| `DFM<id>_<n>.csv` | read and concatenated in natural order of `n` (a new file every 12 hours, each with its header) |
| `Program.txt` | parsed into the Opto Program (`opto_program.read_program`); optional; more than one stops the load |
| `Messages.csv` | read by `messages.read` for the integrity report and the QC Viewer; optional |
| `EnvMon_<n>.csv` | the Environmental Monitor (id 99); never read |

DFM id 99 is the Environmental Monitor's. PyMCU writes its data as
`EnvMon_<n>.csv`, which pyflic's `DFM<id>_*.csv` discovery never matches. pyflic
also excludes id 99 everywhere a DFM id is discovered (`layout`, `yaml_config`,
`project`), so a `DFM99_<n>.csv` is never scaffolded or loaded either, and a
config naming `id: 99` is refused with *DFM 99 is the Environmental Monitor,
not a DFM* (`utils.ENVIRONMENTAL_MONITOR_MESSAGE`), at load and by `pyflic lint`.

The disk cache's key (`cache._CACHE_VERSION` `"3"`) includes the modification
times of `Program.txt` and `Messages.csv`, so either changing recomputes the
member.

## Loading

- **Missing values** are the literal `NA` in released PyMCU 4.0.0, and blank in
  development builds before 16 October 2026. pyflic accepts both. A leading run
  of missing `Interval` is the baseline, 0, as those builds wrote it.
- **The Paradigm columns leave `raw_df`.** `split_paradigm_columns` takes the
  63 new columns off the frame the rest of pyflic copies, and on generation 3
  `ParadigmState.from_frame` holds them as numpy arrays, row for row with
  `raw_df`, as `DFM.paradigm`. Nothing downstream reads them except through it.
  On generations 1 and 2 `DFM.paradigm` is `None`.
- **Repeated records are dropped.** A row whose `Index`, `Date`, `Time` and
  `MSec` an earlier row already has is a block PyMCU wrote twice after a failed
  file write; the first copy is kept, in `raw_df` and in the Paradigm state
  alike, and the count goes to `DFM.repeated_records` and a load note
  (`DFM.load_notes`). A DFM that restarted and counts its `Index` from 1 again
  with new times, as PyMCU builds before 529b926 wrote it, loses nothing; from
  529b926 a restart is an `Index` gap instead ([DFM restarts](#dfm-restarts)).
- **A range filter** applies to the Paradigm state as to `raw_df`
  (`ParadigmState.subset`).
- **The Baseline columns leave `raw_df` too.** `split_baseline_columns` takes
  `Baseline1`–`Baseline12` off, on every generation, and `DFM.baseline_in_force`
  holds them as an `(n, 12)` float array, row for row with `raw_df`, `NaN`
  where the file has `NA`; `None` for a file without them. Repeated records and
  a range filter drop the same rows from it, and `with_params` shares it.

## The Baseline columns

PyMCU writes `Baseline1`–`Baseline12` after `OnMs12`, on every DFM whatever its
firmware: the Baseline in force, which PyMCU added to the Program's threshold
before sending each instruction, so a numeric threshold T meant contact above
`Baseline + T` (PyMCU's `docs/DataFileColumns.md` and
`docs/adaptive-baseline-plan.md`). With `Baseline: Yes` they hold the Starting
baseline from Interval 1 on; with `Baseline: Adaptive` they change when PyMCU
sent new thresholds (at a new Interval or Cycle, or when a well's tracked
baseline moved 10 counts or more). They are `NA` in Interval 0, outside the
Program and with `Baseline: No`.

| pyflic reads it as | Used for | Surfaces in |
|---|---|---|
| `DFM.baseline_in_force`, passed to the light QC as `DFMInputs.baseline_in_force` | the Emulated Trigger's baseline row by row, so it is never reconstructed and never *approximate*; the drift and sustained-contact causes told apart; how each well's baseline moved | `opto_light_qc.csv`'s `LikelyCause` (`CAUSE_SUSTAINED_CONTACT`, `CAUSE_DRIFTING_BASELINE`), `qc/opto/opto_baselines.csv`, and a DFM note in `summary.txt` and the report when any baseline moved |

**The cause.** Without the columns, light the firmware's contact explains
but no pyflic lick or touch does is `CAUSE_DRIFT`, *a drifting baseline or
sustained contact*. With them, `opto_qc.contact_cause` asks whether most of
that light fell where one of the group's Trigger Wells rested 10 counts or
more (`BASELINE_DRIFT_COUNTS`, PyMCU's update step) above its Baseline in
force, its resting level being the minute's 10th percentile
(`resting_floor`): if so the cause is *a drifting baseline*
(`CAUSE_DRIFTING_BASELINE`), otherwise the well rested at its baseline and
the signal stayed up, *sustained contact* (`CAUSE_SUSTAINED_CONTACT`), such as
a fly standing on the food, which pyflic's 3-minute running median absorbs.
`CONTACT_CAUSES` holds all three.

## The original 27 columns

Unchanged in name and position since PyMCU 3.x.

| Column | Generation 1 | Generation 2 | Generation 3 | Surfaces in |
|---|---|---|---|---|
| `Date`, `Time`, `MSec` | the recording clock: `Seconds` and `Minutes` | same | same; PyMCU derives them from recording start + `Index` × 200 ms, so a lost record is a time gap | every time axis; integrity start and end; `data_breaks`; the repeated-record match |
| `Sample` | read | read | read | `data_breaks` rows; the generation-3 training end's `Sample` |
| `Index` | gaps are lost records | same | same, through `ParadigmState.index_gaps` | integrity report (`lost_records`, the gaps), the report's *Lost records*, the light QC's *lost records* (generation 3) |
| `W1`–`W12` | the signal; a value above 40000 is a well in training, its signal the value − 65536 | the signal, never offset | the signal, never offset | every lick, event and feeding metric; on generations 1 and 2 also the emulated firmware trigger |
| `Temp`, `Humid`, `LUX`, `VoltsIn` | kept in `raw_df` | same | same | nowhere |
| `Dark`, `OptoFreq`, `OptoPW` | compared with each scheduled interval's program | same | compared with each Program Interval's Paradigm; 0 on the baseline rows | the light QC's *program mismatch* |
| `OptoCol1` | the light per well (`lights_df`) | same | `lights_df` is still built, but `OnMs` is the light | generations 1 and 2: the light QC's lit samples and light events, the Progressive Ratio light events and `LightOn_sec`, `OptoOn_sec*`, the light table's `LitSec` (`LitSource` = `OptoCol1`) and `LightOnsets`, the light-phase summary, the timeline's light strip |
| `OptoCol2` | decoded with `OptoCol1` | same | always 0 | — |
| `Error` | bits 0x01 and 0x02 counted together as *No ACK / overflow* | 0x01 *No Ack*, 0x02 *Overflow* | same as 2 | the integrity report (`dfm.ERROR_BITS`), the report's error flags; bits 0x04–0x80 are labelled the same on every generation |

## The 63 new columns

Every one is `NA` on generation 2; pyflic reads them only on generation 3.

| Column | pyflic reads it as | Used for | Surfaces in |
|---|---|---|---|
| `Interval` | `ParadigmState.program_interval`; `NA` is an unknown instruction, 0 the baseline | the windows of everything per Program Interval (`interval_runs`); the lead-in (`lead_in_rows`); unknown-instruction rows; restarts (an `Index` gap into a run of `NA` rows, `restarts`); each Interval's start against `Program.txt` (*interval started late*) | the light QC's windows and `opto_light_intervals.csv`'s `ProgramInterval`; Program Facets; the light table's `ProgramInterval`; the open-loop light and dark rates (rows under a Program Interval only); the timeline's band; the integrity report's unknown-instruction rows and *DFM restarts*. Always exposed as `ProgramInterval`, never bare `Interval` |
| `Cycle` | `ParadigmState.cycle` | with `Interval`, one run of one Program Interval | `program_cycles` Facets (`C<n>`); `Cycle` in the light table and `opto_light_intervals.csv`; the timeline's band labels; *interval started late* on every pass |
| `Paradigm` | `ParadigmState.paradigm`; the join key to `Program.txt`'s `Paradigms:` block (names matched ignoring case); `NA` on the lead-in and on unknown-instruction rows, as PyMCU writes it | each row's Paradigm type and settings, and so its Paradigm Family; a change of name resets every well's state (`state_runs`) | the light QC (*paradigm not in program*, *program mismatch*, the rule checks); Program Facet labels; the light table's `Paradigm` and `ParadigmType`; the ledger's `Paradigm`; the integrity report's Paradigm sequence; the Progressive Ratio Test Paradigm (`test_paradigm`) |
| `Phase1`–`12` | `ParadigmState.phase_a`: `A` true, anything else false | Acclimation runs and ends; Training (decision 11) | Progressive Ratio training end and notes; the light QC's Acclimation rule and *acclimation count mismatch*; the light table's Acclimation columns; the ledger's `Phase`; the timeline's shading; the QC Viewer's `Phase` |
| `Activations1`–`12` | `ParadigmState.activations`, cumulative; a missing value carries the previous total; a rise marks an Activation's onset row | light events; the Activation ledger's rows; trigger wells without a program | the light QC's light events (`Source` = `activations`) and Explained Light; the rule checks; Progressive Ratio Light Events; the light table's `Activations` and the family columns; `Activations` on the feeding summary; Roles without a program; *broken yoking* (an `Off` well that counts); the timeline's ticks and the QC Viewer's *Activations* table |
| `Requirement1`–`12` | `ParadigmState.requirement`; `NA` off a ratio Paradigm | the Requirement an Activation met, read from the row before the rise (`requirement_met`) | the ledger's `RequirementMet` and *requirement series wrong* (both light QCs); `BreakRequirement`; the light table's `LastRequirementMet`; `RequirementMet` in `opto_light_events.csv`; the QC Viewer's `Requirement` |
| `Progress1`–`12` | `ParadigmState.progress` | Firmware Events: rises in Progress plus the Requirements met (`firmware_events_between`) | the ledger's and the breaking point's `FirmwareEvents`; `TrendRhoFirmware`; the light QC's `FirmwareEvents`, *firmware events disagree* and `FirmwareEventsWithoutActivity`; the light table's `FirmwareEvents`, `ProgressAtEnd`, `EventsPerActivation` |
| `OnMs1`–`12` | `ParadigmState.on_ms`; a missing value is 0 | the light, exact: `lit()`, `lit_seconds()`; each Activation's onset, end and lit time from the edges of its light period (the ledger); linkage without a program (identical columns); late records (above 200) | `OptoOn_sec*`, `LightOn_sec`, the light table's `LitSec` (`LitSource` = `OnMs`) and `LightOnsets`; the light QC's lit time, *broken yoking* and every length rule check; the ledger's `OnsetMs`, `EndMs`, `LitMs`; the light-phase summary; the timeline's light strip; the integrity report's late records |

pyflic relies on these facts from the PyMCU and firmware source, which
PyMCU's documents do not all state (the differences were reported to the
PyMCU and DFMV3_V2 repositories):

- **`OnMs` is a byte**: the milliseconds the LED was driven since the previous
  record, saturating at 255. It is nominally 200 at most but not clamped
  there, so a late record shows more.
- **`Progress` returns to 0, and `Requirement` moves to its next value, in the
  millisecond the Activation starts.** So the Requirement an Activation met
  is on the row before the row where `Activations` rises.
- **`Interval` `NA` after the program ends cannot reach the file**, because
  PyMCU sends the idle instruction only after closing it. `NA` rows come from
  a DFM restart, from outputs toggled during a recording, or from an
  instruction number older than PyMCU's 256-entry history.

## Program.txt

`opto_program.parse_program` reads three shapes behind one API, told apart by
content (`OptoProgram.grammar`):

| Grammar | Recognised by | What it gives |
|---|---|---|
| `v4` | a `Paradigms:` line, or an interval line whose head is a bare `Dark` or `Lit` | the `Paradigms:` block as `Paradigm` objects (name, one of six types, every setting); per DFM the `Linkage:`, `Program Type:` and the interval lines, thresholds a level, `On` or `Off` |
| `legacy` | `F:`, `P:`, `D:`, `L:`, `M:` interval lines | each line's numbers wrapped into a `Paradigm` with PyMCU's generated name (`ClosedLoop-40Hz-8ms-Hold500-MaxOn0`), `generated` set and the raw numbers in `legacy` |
| `simple` | a `***ALL DFM***` block | no DFM section, and the warning *recorded without a Program: every well Off*; the light QC judges every well `Off` |

`***Env Mon***` is ignored. A `NAME (Type): …` line with a type not among the
six is a warning, and that Paradigm is dropped.

`Baseline:` is `Yes`, `No` or `Adaptive` (`OptoProgram.baseline` is true for
`Yes` and `Adaptive`, and `OptoProgram.adaptive_baseline` for `Adaptive`). With
`Adaptive`, PyMCU writes `BaselineRatePerHour:` and `BaselineCapCounts:` after it,
read as `baseline_rate_per_hour` and `baseline_cap_counts`; beside `Yes` or `No`
they are a warning and ignored.

How the program meets the rows on generation 3:

- **Row `Paradigm` → the `Paradigms:` block**: the type and settings. Only the
  file has them; without it every type-dependent result is `NA` and the
  generation-3 warning *DFM n: firmware 4.0.0 recording without Program.txt:
  Paradigm types and settings unknown* is raised.
- **Row `Interval` k → the DFM section's k-th printed line**
  (`OptoProgram.interval_for`): the thresholds of that run. The export omits
  Intervals whose first-pass start falls past the duration and lists a
  Repeating program's first pass only, so a k beyond the printed lines is
  judged by its Paradigm's name with thresholds unknown.
- **`Linkage:`** gives the Linkage Groups for the whole recording.
- **The timestamps** are used only on generations 1 and 2, to lay the
  intervals out by clock time; on generation 3 the rows are the schedule.

The thresholds printed are the ones PyMCU held when recording started: raw on a
first run, baseline-adjusted when the same loaded program is run again without
reloading. pyflic does not use them as signal levels, only to tell trigger,
`On` and `Off` wells apart.

`qc/opto/opto_program.csv` (`opto_qc.OPTO_PROGRAM_COLUMNS`) is the program as
read: `DFM, Interval, Start, DurationMin, ProgramType, Paradigm, ParadigmType,
Generated, Dark, FrequencyHz, PulseWidthMs, HoldMs, MaxOnMs, ActivationMs,
LockoutMs, OnDelayMs, EventGapMs, Acclimation, Series, FirstRequirement, Step,
Scale, Rate, Cap, Ratio, Linkage, TriggerWells, OpenLoopWells, LegacyF,
LegacyP`.

### The light the Experiment Type expects

An Experiment Type's definition states the light it expects (ADR-0019):
`light:` is `none`, `any`, or a list of Paradigm types in PyMCU's spelling.
`ExperimentType.light_problems(program, dfm_ids)` judges `Program.txt`, not
the rows:

- each printed interval runs its Paradigm's type on its level wells and
  OpenLoop on its `On` wells; an interval whose wells are all `Off` runs
  nothing, whatever its Paradigm (PyMCU's `Idle` among them);
- OpenLoop is always allowed beside a list, as a control, and the `simple`
  grammar runs nothing;
- without `Program.txt`, or under `optogenetics: no`, nothing is checked.

A program running a type the Experiment Type does not expect stops
`load_experiment_yaml`, naming each DFM and interval, and `pyflic lint`
reports it as an error, adding for Hedonic that a hedonic type expecting that
light can be built in the Experiment Type Editor. The shipped types expect:
Custom `any`; Hedonic `none`; ProgressiveRatio `[ProgressiveRatio,
FixedRatio]`. A program that runs a ClosedLoop Training interval before its
ratio Paradigm therefore needs a lab type that expects ClosedLoop too, under
which Training includes that interval as before. The test fixtures that do
this (`choice_pr_spec`, and DFM 1 of `tests/fixtures/Program_v4.txt`) load
under the lab type `TrainedProgressiveRatio` in `tests/pr_fixtures.py`, which
expects ClosedLoop, ProgressiveRatio and FixedRatio.

How the light is judged does not depend on the Experiment Type: the light QC,
the light table's columns and the report's rule subsections follow the
Paradigm types the program runs (ADR-0017). The type's Analysis Modules
decide only which of the type-specific analyses run: the `ratio` module's
Progressive Ratio analysis, the `yoked` module's yoked comparison and the
`light_dark` module's *Feeding in light and dark*.

## Messages.csv

The columns are `DFM,Date,Time,Millisecond,Sample,Message,MsgType`, where
`Sample` holds the record `Index` and `Millisecond` is an unformatted float.
DFM 0 is the MCU itself. Two writers exist, and `messages.read` reads both:

- **PyMCU 4.0.0 from 10/17/2026** (PyMCU d148926) writes with the csv module,
  so a message holding a comma is quoted. A message that came again within 10
  minutes (same DFM, type and text) is one line ending "(repeated N more
  times, the last at HH:MM:SS)"; `messages.read` puts N in `Repeats`, and
  `messages.count` counts a frame with its repeats. Lost records read "Lost N
  records (S s): Index A to B, from HH:MM:SS.", dated when the gap began; a DFM
  that restarted reads "DFM n restarted: its record count went back from …"
  and "DFM n recorded nothing for about S s while it restarted: Index A to B
  are missing."; DFM error bits are notices naming the bit ("DFM 1 error: No
  Ack (…)").
- **Earlier builds** wrote every line unquoted, so a message's commas made
  extra fields: the line is split on its first five commas and its last one,
  as before. Their lost records read "(1021) Skipped index: (1000)".

`messages.read` returns `DFM, Time, Index, Message, MsgType, Repeats`; a
missing file is an empty frame.

`messages.record_messages` picks out what PyMCU said about a DFM's records,
one row each with `DFM, Time, Kind, First, Last, Records, Message`. Its kinds:

- `LOST_RECORDS` (`"lost records"`): "Lost N records (S s): Index A to B,
  from HH:MM:SS.", or an earlier build's "(n) Skipped index: (m)";
- `DFM_RESTARTED` (`"restarted"`): "DFM d restarted: its record count went
  back from m to 1. …" (`DFM._StartRestart`);
- `RESTART_GAP` (`"restart gap"`): "DFM d recorded nothing for about S s while
  it restarted: Index A to B are missing." (`DFM._FinishRestart`), written
  after the first when the restart lost records.

What reads them:

- The integrity report lists the `Error` rows for the DFM and for DFM 0. On
  generation 3 it also sums the "Lost N records" counts inside the loaded
  `Index` range and checks them against the `Index` gaps outside restarts
  ("Lost records Messages.csv names: N, as the Index gaps outside restarts
  show", or the difference), and matches each restart the rows show with its
  `RESTART_GAP` by `Index` range (next section).
- The QC Viewer's Integrity sub-tab shows every message for the DFM and DFM 0,
  and counts them with their repeats.

## DFM restarts

A DFM that restarts on its own (PyMCU 529b926) loses its Paradigm state and its
record count. PyMCU keeps `Index` and the times continuous with an offset, so
the data show:

1. **an `Index` gap**: the records the DFM did not make while it was down;
2. **rows under no instruction** (`Interval`, `Cycle` and `Paradigm` `NA`,
   Phase `M`, `Requirement` and `Progress` `NA`, `OnMs`, `Dark`, `OptoFreq`
   and `OptoPW` 0), until PyMCU has sent the Interval and linkage again;
3. **the instruction afresh**: every well as under a new Paradigm (Acclimation
   again, the Requirement series from k = 1, `Progress` 0, no lockout
   running), with the `Activations` totals carried on.

How pyflic reads it (generation 3 only; generations 1 and 2 have no Paradigm
state to restart):

- **`ParadigmState.restarts()`** finds each `Index` gap whose next row begins
  a run of unknown-instruction rows after the lead-in: a `RestartSpan(row,
  resume_row, missing, minute)`. `state_runs()` splits there, because the
  restart's `NA` rows are a run of their own.
- **`DFM.restarts()`** adds `Messages.csv`: each span is matched with its
  `RESTART_GAP` by the first `Index` missing, and a restart only the
  file names is listed too ("named in Messages.csv; the data show an Index
  gap only", or "…; it lost no records, so the data show no Index gap").
  Each is a dict with `Time`, `Minute`, `Row`, `ResumeRow`, `AfterIndex`,
  `Missing`, `Seconds`, `Source`, `Messages` and `Note`.
- **The integrity report** gains `## DFM restarts`, one line per restart with
  its clock time, minute, the records and seconds missing, the rows under no
  instruction and whether `Messages.csv` names it, and the report dict's
  `restarts`. Its *Lost records* still counts every `Index` gap, the
  restart's among them; the *No Ack* note is not given when anything was
  lost, so a recording with a restart never gets it.
- **The light QC** warns *DFM restarted* on every group of the DFM
  (`Restarts` in `opto_light_qc.csv`, with or without `Program.txt`). A
  restart raises that one warning: its *lost records* check counts only the
  gaps outside restarts (`rule_checks.gaps_outside_restarts`), and its
  *unknown-instruction rows* leave out the restart's rows under no
  instruction, before PyMCU resent it. *DFM restarted* reports both, and the
  notes give their counts.
- **The Activation ledger** starts a new state run there: `K`,
  `SinceLastOnsetMs`, the Acclimation runs and the series check start over,
  as the firmware's do. A light period cut by the gap has `Boundary` `gap`,
  and an onset right after one is `Overlap`, since it may have begun in the
  lost records.
- **The Program timeline** draws a red band labelled "restart" from the end
  of the last record before the gap to the resume row.
- **Progressive Ratio**: a restart inside a group's Test window ends the
  window there (`test_restart`): the first-gap rule sees the window up to the
  restart, a count still running is censored, and the note goes into
  `pr_light_qc.csv`'s `Notes`, `summary.txt` and the report.
  `training_end` and `acclimation_end` do not move.

## What pyflic derives

| Quantity | Generation 1 | Generation 2 | Generation 3 |
|---|---|---|---|
| **Training end** (per well) | the last flagged minute | none: every well never flagged | the first row under a ratio Paradigm and not in Phase `A` (`training_end`); `Minutes` of the row before |
| **A Progressive Ratio Light Event** | an onset of the chamber's `OptoCol1` light | same | an Activation of the paired sucrose well |
| **The Requirement a light event met** | — | — | the row before the rise in `Activations`; on the first row of a new Paradigm, the program's `requirement_for(k)`; none in Acclimation |
| **Firmware Events behind a light event** | — | — | from the previous onset, exclusive, to this one, inclusive |
| **Light seconds** | lit `OptoCol1` samples × the sample period | same | Σ `OnMs` / 1000 (`DFM.light_seconds`, `DFM.light_source`) |
| **Linkage without a program** | wells lit identically in `OptoCol1` | same | wells whose `OnMs` columns are identical |
| **Trigger wells without a program** | every lit well | same | wells with Activations |
| **Role without a program** | blank | blank | from the data |
| **Program Facets** | the program's clock schedule, approximate | same | the rows' `Interval` and `Cycle`; `Experiment.facet_windows_for(dfm_id)` gives each Facet as a list of windows, several when a `program` Facet pools a Repeating program's Cycles |
| **The light QC's contact evidence** | the emulated trigger | same | Activations and Firmware Events with no pyflic activity; nothing emulated |
| **An Activation's onset, end and lit time** | — | — | to the millisecond from the edges of its `OnMs` (the ledger), or to the record |
| **Each Paradigm's rule checks** | not run | not run | per Linkage Group and Paradigm, over the ledger |
| **A DFM restart** | — | — | an `Index` gap into `NA`-Interval rows, confirmed by `Messages.csv` |

## The light QC's own checks

On generation 3 the optogenetic light QC (`opto_qc`) windows each DFM by its
rows and judges each window under the Paradigm its rows name. Besides
Explained Light and the rule checks (next section), it reads the file's own
bookkeeping. Flags are `opto_qc` constants; `FAILING_FLAGS` fail a Linkage
Group and `WARNING_FLAGS` warn.

| Check (flag) | Verdict | Constant (default) |
|---|---|---|
| *broken yoking* | fails: within a Linkage Group, the share of lit records whose wells' `OnMs` differ by more than 1 ms is above the constant, or a linked `Off` well counts Activations | `opto_yoking_fail_fraction` (0.01) |
| *paradigm not in program* | fails: a row's Paradigm is not in `Program.txt`'s `Paradigms:` block, so the file is not the program that ran | — |
| *program mismatch* | warns: a window's `Dark`, `OptoFreq` or `OptoPW` disagree with its Paradigm, or its rows name a defined Paradigm other than the printed line's; the rows win, and are judged by their own Paradigm | — |
| *acclimation count mismatch* | warns: a well's Activations in its `A` run are not its Paradigm's `Acclimation` | — |
| *firmware events disagree* | warns, per Trigger Well: \|`FirmwareEvents` − pyflic feeding events\| / `FirmwareEvents` above the constant, judged from 5 events in either count (`MIN_FEEDING_EVENTS`). pyflic's count leaves out events that start inside the well's own Activation after its onset row, as the firmware does | `opto_event_disagreement_warn_fraction` (0.25) |
| *lost records* | warns, per DFM: more records missing from `Index`, outside restarts, than the constant | `opto_lost_records_warn` (25) |
| *unknown-instruction rows* | warns, per DFM: the share of rows after the lead-in under no instruction, outside restarts, above the constant | `opto_unknown_rows_warn_fraction` (0.01) |
| *late records* | warns, per DFM: more rows with `OnMs` above 200 than the constant | `opto_late_records_warn` (10) |
| *no Program.txt for firmware 4.0.0* (`PROGRAM_MISSING`) | warns: a generation-3 DFM with no `Program.txt` or no section in it. `NO_PROGRAM_SECTION` stays for generations 1 and 2 | — |
| *non-feeding light during feeding* | warns: a NonFeeding light that stayed on through feeding | `opto_non_feeding_lit_warn_fraction` (0.10), `opto_non_feeding_min_sec` (30) |

None of the five constants of the bookkeeping checks has bench data behind
it yet ([To confirm](#to-confirm-on-the-first-real-recording)).

The Progressive Ratio light QC (`pr_light_qc`) measures `TrendRhoFirmware`,
the trend of Firmware Events per Light Event, as a witness, not a second
judge: the *no increasing trend* warning stays on pyflic's licks, and a
firmware trend that rises while the licks do not is a note pointing at the
lick detection.

## The Activation ledger and the rule checks

Built by the Paradigm analyses (ADR-0017; the plan, `paradigm-analyses-plan.md`,
is at commit 82aace9: see [Removed plans](README.md#removed-plans)).

**The ledger** (`activation_ledger.ledger(state, wells=None, linked=None)`) has
one row per Activation of each well, read from the Paradigm state alone:
`Well, Row, EndRow, Index, OnsetMin, OnsetMs, EndMs, LitMs, Timing, Overlap,
Boundary, SinceLastOnsetMs, Phase, K, Paradigm, ProgramInterval, Cycle,
RequirementMet`. Its time base is milliseconds from the start of the record
with `Index` 1, the simulator's and PyMCU's truth's; `OnsetMin` is pyflic's
`Minutes`, which count from the first row's stamp, so the two are never
converted into each other. A row is `Overlap` when other light merges with
the period (a linked partner's onset in it, light already on, the next
Activation run into it, two onsets in its row, or a gap in `Index` right
before it), and has a `Boundary` (`interval`, `gap`, `end`) when something
other than its rule stopped it. `opto_light_events.csv` carries the ledger's
columns on its generation-3 rows, with `Family` (`paradigm_families.family_of`
of the row's mode and Phase) and `EndedBy` (`rule_checks.ended_by`).

**The Paradigm Families** (`paradigm_families`): open loop, closed loop
(ClosedLoop and every Acclimation), fixed length (FixedInterval, and the two
ratio types, in Main; an interval and a ratio branch) and non-feeding.
`FAMILIES` declares each family's light-QC columns, light-table columns,
headline metrics and rule figure; `families_in(program)` lists the families a
program runs; `rule_sentence(paradigm)` states a rule with its settings.

**The rule checks** (`rule_checks`, one function per check, each returning a
`CheckResult`) judge the ledger rows of a Linkage Group's Trigger Wells under
one Paradigm, with `Program.txt`'s settings, pooled over the group's windows
of that Paradigm. A row is judgeable for a length when it is not `Overlap`
and has no `Boundary`; `tol` is `opto_activation_tolerance_ms`, one record
wider where `Timing` is `record`.

| Check (flag) | Applies to | Verdict |
|---|---|---|
| *short activations* | ClosedLoop Main; every Acclimation | fails above `opto_activation_length_fail_fraction` |
| *max on exceeded* | ClosedLoop and NonFeeding with `MaxOnMs` | fails on any one |
| `CutOffs` | ClosedLoop with `MaxOnMs` | reported |
| *wrong activation length* | FixedInterval, PR and FR in Main | fails above `opto_activation_length_fail_fraction` |
| *lockout broken* | FixedInterval Main | fails above `opto_lockout_fail_fraction` |
| *contact without activation* | FixedInterval Main | warns above `opto_missed_contact_warn_fraction`, from 5 free runs |
| *requirement series wrong* | PR and FR Main | fails on any one; a state run that matches the series with `K` one less passes, with a note (the rows credit its first Main Activation to the old Paradigm) |
| *non-feeding dark* | NonFeeding | warns above `opto_non_feeding_dark_warn_fraction`, from 10 s expected lit |
| `RelightDelayMs` | NonFeeding | reported |
| *open loop not lit* | the lowest open-loop window | warns below `opto_open_loop_min_lit_fraction` |
| *interval started late* | every Program Interval's first row, every Cycle | warns above `opto_interval_lag_warn_sec`; the first Interval allowed `LEAD_IN_SEC` (5 s) |
| *DFM restarted* | each restart | warns |

The Progressive Ratio light QC calls `requirement_series_mismatches` on the
Test window's Main Activations of the Paired Sucrose Well. The report's rule
subsections and figures are `rule_figures.py`'s; they recompute the checks per
Paradigm with the same functions, so a group's sections sum to its
`opto_light_qc.csv` row.

**The light table's family columns** (`light_summary`, from the ledger):
`MainActivations`, `MedianActivationSec`, `CutOffs`, `MaxActivations`,
`RewardEfficiency`, `MedianReturnSec`, `LockedOutContactSec` (redefined),
`Suppressions` (redefined), `MedianSuppressionSec`, `LicksPerMinLit`,
`LicksPerMinDark` and `LightFeedingIndex`, each `NA` outside its family and
without a program. The open-loop rates use only rows under a Program Interval
(`Interval` > 0), and a phase shorter than `MIN_LIGHT_PHASE_MIN` (1 minute) has
no rate. The light table carries them under every Experiment Type; the
reports' *Feeding in light and dark* figure shows only under a type with the
Light and dark module (`light_dark`, which Custom and ProgressiveRatio
include). The headline metrics follow the families run
(`experiment_headline_metrics`; `HEADLINE_METRICS` is the four of before, for a
member with no generation-3 DFM). `behaviour_rows` gives the Activation
columns to the chambers whose Role earns light (Paired, Self, Mutual) only.

**The yoked comparison** (`analytics.yoked_groups`, `yoked_comparison`) takes,
for every Linkage Group of one Paired and one or more Yoked chambers, the
paired chamber's feeding-summary metrics minus the mean of its yoked
chambers'. It runs under a type with the Paired and yoked module (`yoked`,
which Custom includes). The `ratio` module excludes it, because Progressive
Ratio keeps its own Paired-Yoked Difference.

## Outputs by file

What each output gains from the new columns. Every firmware column is `NA`
before generation 3.

| Output | New or changed |
|---|---|
| `analysis/feeding_summary*.csv` | optogenetic members: `Role`, `LitSec`, `Activations`; `OptoOn_sec*` from `OnMs` on generation 3 |
| `analysis/light_summary.csv`, `light_wells.csv`, their `_facet` variants | new: `light_summary.LIGHT_CHAMBER_COLUMNS` and `LIGHT_WELL_COLUMNS`, with the Paradigm Families' columns (previous section) |
| `analysis/yoked_comparison.csv` | new, a design with a yoked group under a type with the `yoked` module: `analytics.YOKED_COLUMNS`, the design's factors, then `Paired<m>`, `Yoked<m>`, `d<m>` per metric |
| `analysis/light_phase_summary.csv` | the phase is each chamber's own light, on every generation (decision 24) |
| `analysis/pr_breaking_point.csv` | `BreakRequirement`, `FirmwareEvents`, `ParadigmType`; `LargestRequirement` equals `BreakRequirement` on generation 3; no row for a generation-2 group; a restart inside the Test window shortens `TestMinutes` and can set `Censored` |
| `analysis/pr_light_events.csv` | `Requirement`, `FirmwareEvents`, `Phase` |
| `analysis/pr_light_qc.csv` | `TrendRhoFirmware`; the *requirement series wrong* flag; a restart's note in `Notes`; `YokedChamber` `NA` for a solo group |
| `analysis/summary.txt` | each DFM's generation and light source; the Program Facets note; *no training information* for a generation-2 Progressive Ratio member; the rule checks' constants; a restart that ends a Test window |
| `qc/opto/opto_light_qc.csv` | `Generation`, `Roles`, `FirmwareEvents`, `PyflicEvents`, `AcclimationActivations`, `AcclimationExpected`, `FirmwareEventsWithoutActivity`; the rule checks' `ShortActivations`, `OverMaxOnActivations`, `CutOffs`, `WrongLengthActivations`, `LockoutViolations`, `MissedContacts`, `SeriesMismatches`, `NonFeedingDarkSec`, `RelightDelayMs`, `IntervalLagSec`, `Restarts` (nullable integers for the counts); on generation 3 `ContactSec` and `ContactWithoutActivitySec` read the firmware's decisions, `OpenLoopLitFraction` is the lowest window's and `EmulationApproximate` is `NA` |
| `qc/opto/opto_light_intervals.csv` | `ProgramInterval`, `Cycle`, `Paradigm`, `ParadigmType`, `HoldMs` (was `DecayMs`), `LitSource` |
| `qc/opto/opto_light_events.csv` | `Well`, `Phase`, `RequirementMet`, `FirmwareEvents`, `Source`; on generation 3 the ledger's `OnsetMs`, `EndMs`, `LitMs`, `Timing`, `Overlap`, `Boundary`, `SinceLastOnsetMs`, `K`, then `Family` and `EndedBy` |
| `qc/opto/opto_program.csv` | the columns above, on every generation; they replace `F`, `AcclimationEvents`, `P`, `NonFeeding`, `DecayMs`, `Delay` and `MaxTimeOn` |
| `qc/opto/opto_baselines.csv` | new, only when some DFM's data have the Baseline columns: `opto_qc.OPTO_BASELINE_COLUMNS`, one row per well, its Baseline in force's start, end, lowest, highest, largest move, number of changes, and with `Baseline: Adaptive` the cap and whether it was reached |
| `qc/integrity/DFM<n>_integrity_report.*` | `generation`, `lost_records`, `repeated_records`, `unknown_instruction_rows`, `late_records`, and on generation 3 `restarts`; the text adds the gaps, the Paradigm sequence, `## DFM restarts`, the No Ack note, the `Messages.csv` errors and its lost-records check |
| `qc/data_breaks/DFM<n>_data_breaks.csv` | led by `BreakIndex` and `GapSeconds` (were `Index` and `Interval`, which crashed on any version-3 file with a break) |
| `qc/simultaneous_feeding/DFM<n>_simultaneous_feeding_matrix.csv` | `Both_pre`, `MaxMinSignalAtBoth_pre`, `HigherInCol1AtBoth_pre`, before the dual-feeding correction |
| `qc/program/program_timeline_dfm<n>.png` | new; on generation 3 with the `On` and `Off` shading and the restart bands |
| the experiment report | on generation 3 with `Program.txt`, a *Rule check* subsection per Paradigm with its rule figure; *Light behaviour* by Paradigm Family, with *Feeding in light and dark* under the `light_dark` module; *Paired and yoked* under the `yoked` module; a type's Control Comparisons first in the Results |
| `<project>_LightSummary.csv`, `<project>_LightWells.csv` and `_Facet` variants | new: the members' light tables stacked, `Experiment` first |
| `<project>_YokedComparison.csv` | new: the members' yoked comparisons stacked |

## Where the outputs are made

**Basic analysis** (`Experiment.execute_basic_analysis`) writes every member
output above for an optogenetic member: the QC reports (`write_qc_reports`)
into `qc/`, the light QC into `qc/opto/`, each generation-3 DFM's Program
timeline into `qc/program/`, and the light tables and, under the `yoked`
module, the yoked comparison into `analysis/`. The Project's *Analyze all*
and a Batch Run's `run_all_analyses` step run it for every member, whatever
the Hub's ticks. The `basic_analysis` script step passes `skip_qc=True`,
which leaves out `write_qc_reports` (the integrity report, data breaks, the
two-well tables and the signal plots) but not the light QC or the Program
timeline.

**The Hub** (ADR-0018) runs a selection. Its QC, Analyze and Plots panels
are checkboxes over one button each, and each item is an id that
`global.omit:` can leave out (`pyflic/base/selections.py`). The items behind
this document's outputs sit in each panel's *Optogenetics* group, shown for
an optogenetic member, and in the *Progressive Ratio* group, shown under a
type with the `ratio` module:

- **Run QC**: `opto_light_qc` (the four tables in `qc/opto/`),
  `opto_light_figures` (*Light explained by licks*, `opto_light_dfm<id>.png`)
  and `program_timeline` (`qc/program/`, greyed with no firmware 4.0.0 data
  and no `Program.txt`); and `pr_light_qc`, `pr_light_event_figures` and
  `pr_resting_level_figures`.
- **Run analysis**: `light_tables` (`light_summary.csv`, `light_wells.csv`
  and, with Facets, their `_facet` variants) and `yoked_comparison` (under the
  `yoked` module, greyed when no Linkage Group yokes a chamber to a Paired
  one); and `paired_yoked_diff`, `cumulative_diff` and `breaking_point`.
- **Generate plots**: the `ratio` module's four figures.

The light figures and the Program timeline are drawn at 10-minute bins, and
the Progressive Ratio cumulative table and curves at 1 minute, whatever the
Analyze panel's *Bin size*. The buttons these items replaced (*Opto light QC
table*, *Light explained by licks (QC)*, *Program timeline*, *Light summary
CSV* and the rest) are gone, and their script actions remain:
`opto_light_qc`, `plot_opto_light`, `plot_program_timeline` and
`light_summary`. The script actions `run_qc`, `run_analysis` and `run_plots`
run a panel's set minus `global.omit:`, as its button does.

## To confirm on the first real recording

No recording from a 4.0.0 board existed when this was built. Everything above
comes from the PyMCU and firmware source and a simulator pinned to the
firmware's own `LEDControl.c` (`tests/v4_fixtures.py`, cross-checked by
`tests/test_ledcontrol_crosscheck.py`). The first bench recording, a quiet
one, has confirmed part of it (below). Diff each new bench recording against
the fixtures and the loader:

- The header is the 102 columns (90 before the Baseline columns); every row has
  as many fields; missing values are `NA`.
- The lead-in: how many `Interval` 0 rows, their `Paradigm` (`NA`), their
  `Dark`, `OptoFreq` and `OptoPW` (0). Confirmed below.
- `Interval` changes on the first row the DFM ran the new instruction; its lag
  against `Program.txt`'s timestamps.
- `Activations` rises in the record where the light starts; a linked `Off`
  well's `OnMs` equals the trigger well's; `Requirement` during Acclimation;
  `Progress` is 0 on the rising row.
- `OnMs` values above 200 and how often they occur; `OptoCol1` against
  `OnMs > 0`.
- `Program.txt`: thresholds raw or baseline-adjusted (a re-run program), the
  omitted Intervals, the Repeating first pass.
- `Messages.csv`: the actual field shapes, and any message with commas.
- The five constants (`opto_event_disagreement_warn_fraction`,
  `opto_yoking_fail_fraction`, `opto_lost_records_warn`,
  `opto_unknown_rows_warn_fraction`, `opto_late_records_warn`) against what a
  clean recording shows.
- The simulator's signal model against real `W` traces around Activations
  (the 128-scan lag).

Then re-run the real Progressive Ratio dataset (generation 1) and confirm its
outputs are unchanged apart from the deliberate changes in `CHANGELOG.md`.
It was not on the machine the work was built on, so the check that
generation-1 outputs are unchanged rests on the fixtures alone, and
`tests/test_opto_real_dataset.py` skips without it. Check also whether its
`Program.txt` fits the shipped ProgressiveRatio type, which expects
ProgressiveRatio and FixedRatio, or needs a lab type that expects its
closed-loop Training too (see
[The light the Experiment Type expects](#the-light-the-experiment-type-expects)).

**Confirmed since** (10/17/2026):

- The lead-in, from the first bench recording (a quiet single-well run, every
  well Off): 20 `Interval` 0 rows, `Paradigm` `NA`, `OptoFreq` and `OptoPW` 0.
- `Requirement` during Acclimation shows the first Main Requirement (PyMCU, on
  a FixedRatio bench recording).
- OpenLoop and `On` wells never add Activations (DFMV3_V2's `LEDControl.c`).
- Acclimation counts every on-period as an Activation, even within one
  Firmware Event (DFMV3_V2 024b692). The simulator follows it, and the
  cross-check passes against that firmware.
- PyMCU's own simulator (`scripts/simulate_recording.py`, PyMCU 6d1aa42) runs
  simulated flies through PyMCU's real record parser and file writer, and
  writes a truth folder. On its two-hour Progressive Ratio recording,
  pyflic's light table matched the truth exactly for all twelve triggering
  wells on Activations, Acclimation Activations, the last Requirement met,
  Progress at the end and lit time. The end of Acclimation agrees to within
  one record once the two time origins are aligned. The truth counts
  milliseconds from the start of record 1, and pyflic's `Minutes` count from
  the first row's time stamp, which PyMCU sets at the record's end; on that
  recording the two differ by 0.57 s. `tests/test_pymcu_simulator.py` checks
  all of it on every run.

Still to see on the bench: a DFM that restarts mid-recording (PyMCU 529b926
keeps `Index` and the times continuous, writes `NA` Interval rows until it has
sent the instruction again, and the Paradigm state starts over).

### Bench checkpoint for the Paradigm analyses

The rule checks, the family columns and their constants were calibrated on
pyflic's simulator and on PyMCU's simulator only (stream S1 in the "As built"
list of the Paradigm analyses plan, at 82aace9).
On real recordings of each Paradigm type, confirm:

- **One recording of each type**, ClosedLoop with and without `MaxOnMs`,
  FixedInterval with a lockout longer and shorter than `ActivationMs`,
  NonFeeding with and without `MaxOnMs`, ProgressiveRatio (Arithmetic and
  Exponential) and FixedRatio, OpenLoop and wells `On`: no rule-check flag
  fires on a clean run, and every rule figure sits on its dashed line.
- **The constants' defaults** (`opto_activation_tolerance_ms` 25,
  `opto_activation_length_fail_fraction` 0.05, `opto_lockout_fail_fraction`
  0.02, `opto_missed_contact_warn_fraction` 0.25,
  `opto_non_feeding_lit_warn_fraction` 0.10, `opto_non_feeding_min_sec` 30,
  `opto_non_feeding_dark_warn_fraction` 0.10, `opto_interval_lag_warn_sec` 5)
  against what clean bench recordings show. On simulated data the largest
  clean values were 0 for the length, lockout, missed-contact and series
  checks, 2.1% for *non-feeding dark* and 0.2 s for the interval lag.
- ***Contact without activation*** counts a run of pyflic **activity**, a lick
  or a tasting sample, as contact. A tasting sample can sit below the
  firmware's threshold (the authored level plus the well's baseline), so on a
  real FixedInterval recording the free runs that brought no Activation may be
  touches the firmware never saw. Check how many fire on a clean run before
  trusting the 0.25.
- **`LEAD_IN_SEC`** (5 s, `rule_checks`), the allowance for the first
  Interval's start: the lead-in measured 4.8 s on one bench recording and in
  PyMCU's simulator, 5.2 s in pyflic's. Re-measure it on more recordings, and
  on a DFM whose baseline takes longer.
- **A wrong `OnDelayMs`** is caught by no flag of its own. A light that came
  back sooner than `Program.txt` says breaks no check, and one that came back
  later only adds dark time to *non-feeding dark*, which warns only past its
  fraction. The setting shows in `RelightDelayMs` and the relight-delay
  figure alone: compare their median with `OnDelayMs`.
- **A DFM restart**: the `Index` gap, the `NA` rows and their count, the
  `Messages.csv` lines and their match by `Index`, Acclimation and `K`
  starting again, the restart band on the timeline, and a Progressive Ratio
  Test window ending there.
- **PyMCU's FixedRatio recording of 10/17/2026**: the series check, `K`, and
  the Acclimation counts.

**A known gap: Circadian exports.** PyMCU's `Program.txt` omits an Interval
whose start, counted from the Start Time through the lengths of the Intervals
before it, falls past the recording's duration. A Circadian program runs by
the time of day instead, so a recording that starts partway through its day
can run an Interval the export leaves out. On `stress_spec`'s DFM 3, started at
23:56 under a day of FR1 for 1438 minutes then FI1 for 2, the export lists FR1
only, though FI1 runs at minutes 2 to 4. The rows name FI1, so it is judged by
its name, but its thresholds are unknown and its wells get no Acclimation
columns (`NA`). pyflic's simulator reproduces PyMCU's export exactly, so a real
recording of such a program will show the same; only an export that prints
every Interval of the day would close it.

## Not read

- `Temp`, `Humid`, `LUX` and `VoltsIn`, and the Environmental Monitor's own
  file.
- Fine mode (firmware 4.1.0, per-scan data). `ParadigmState.index` is the key a
  fine table would join on.
- The firmware version: no file records it, which is why the generation is
  inferred. Nor does it record which of the two firmware 2.0.0 builds ran,
  so generation 1 labels error bits 0x01 and 0x02 together.

## Not built

Left out of the firmware 4.0.0 work and the Paradigm analyses by decision:

- **Acclimation as Facets outside Progressive Ratio.** Acclimation stays
  columns (`AcclimationEndMin`, `AcclimationActivations`, `MainActivations`);
  splitting every result at each chamber's own Acclimation end, as
  Progressive Ratio's Training and Test do, was deferred.
- **Lick microstructure under light**: bout structure within an Activation
  against outside it.
- **Analyses of their own for Circadian programs or for Mutual groups**,
  beyond their Roles.
- **Re-running a Paradigm offline** with other thresholds: it needs
  fine-mode data.
