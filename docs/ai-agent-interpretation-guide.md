# Interpreting pyflic results: a guide for AI agents

This guide is for an AI agent that is handed one or more folders analysed by
pyflic and asked to summarise and interpret what they show. It is written to be
included in a larger prompt. It assumes nothing about the particular
experiment: not the genotypes, not the foods, not the question. It tells you how
to find out those things from the files, how to judge whether the data can be
trusted, and how to read the statistics pyflic has already run.

Read it as instructions addressed to you. Where the surrounding prompt sets the
format or length of your answer, follow the surrounding prompt. The ground rules
in section 1 apply whatever the format.

| Section | What it gives you |
|---|---|
| 1 | Ground rules: what you may and may not claim |
| 2 | A primer: what FLIC measures and how pyflic turns signal into numbers |
| 3 | How to recognise a Batch, a Project, a Member and a standalone experiment |
| 4 | Every output file, what one row means, and its key columns |
| 5 | How to read the configuration files |
| 6 | How to recognise treatments and controls, and check the design for confounds |
| 7 | A dictionary of every metric, with units and traps |
| 8 | Quality control: a procedure, the meaning of every check, and how QC should change your conclusions |
| 9 | The statistics pyflic runs, exactly, and how to interpret them |
| 10 | Playbooks for each kind of experiment |
| 11 | A step-by-step workflow |
| 12 | How to write the summary |
| 13 | Quick reference: glossary, defaults, a question-to-file index and red flags |

---

## 1. Ground rules

1. **pyflic's outputs are the record.** Every number you state must come from a
   file you read, or be a derivation you name (for example "raising the
   transformed value to the fourth power gives about 8,490 licks"). Say which
   file each key number came from.
2. **Keep three kinds of statement apart.**
   - *Computed by pyflic*: values and p-values in its CSV, TXT and PDF outputs.
   - *Derived by you*: arithmetic on pyflic's tables, such as counts of chambers
     per treatment, per-member means, exclusion percentages, back-transformed
     lick counts, the direction of an effect in each member. These are welcome,
     and you must label them as yours.
   - *Interpretation*: what the results may mean biologically.

   Do not compute new inferential statistics (p-values, confidence intervals)
   and present them as results. If the surrounding prompt allows exploratory
   computation, label it exploratory and never let it override pyflic's own
   tests.
3. **Quality control comes before results.** Read the QC first and decide what
   is trustworthy, then read the results. A result you have already seen makes
   every later QC judgement look like a choice made for it.
4. **A missing file is not a passed check.** If a QC output does not exist, the
   check was not run, or it did not apply. Say which, when you can tell. "No
   integrity report was found" is different from "integrity was ok".
5. **One rule for every treatment.** Any QC concern, exclusion or caveat must be
   weighed for every treatment alike. When QC failures or exclusions fall
   unevenly across treatments, that is itself a finding and must be reported.
6. **Say plainly when a comparison is not significant.** Do not describe a
   "trend" the statistics do not support. You may describe the direction of a
   non-significant difference only if you say in the same sentence that it is
   not significant.
7. **Significance is not size.** Report the magnitude of a difference, in its
   units, beside its p-value. A tiny p on a large sample can describe a
   negligible difference.
8. **Do not infer what is not recorded.** Treatment names are labels a person
   typed. pyflic records no genotype biology, no sex, no age, no food
   composition and no time of day, except as far as the labels, `well_names`,
   notes files or factor names state them. When you interpret a label (for
   example reading `Ctrl` as a control), say that it is an inference from the
   name.
9. **Use pyflic's vocabulary.** A Project's subfolders are **Members**, not
   replicates: they are different experiments that address one question, not
   repeats of one experiment. Section 13.1 has the glossary.
10. **Be specific about where.** Name the member, DFM, chamber or chamber group
    when you report a problem: "Member Rep2, DFM 5, chamber group 3", not "one
    group".

---

## 2. Primer: what FLIC measures and how pyflic computes it

### 2.1 The instrument

- A **DFM** (Drosophila Feeding Monitor) is one board with twelve **wells**,
  `W1` to `W12`, each holding liquid food. When a fly's proboscis or leg touches
  the food it completes a circuit and the well's signal rises. Each well is
  sampled **5 times per second** by default (`samples_per_second`). A DFM writes
  its data as `DFM<id>_<n>.csv`; a long recording spans several `<n>` files.
- Every DFM has an integer **id**. Ids are per recording: DFM 1 of one
  experiment is a different physical board from DFM 1 of another. DFM 99 is the
  Environmental Monitor (temperature, humidity, light), never a DFM.
- A **chamber** is the unit one fly occupies, and **every metric is computed per
  chamber**. The **chamber layout** decides what a chamber is:
  - `single_well`: 12 chambers per DFM; chamber *n* is well *n*. One food per
    fly. Used for consumption assays.
  - `two_well`: 6 chambers per DFM; chamber *n* is wells 2*n*−1 and 2*n*
    (chamber 1 = W1+W2). Two foods per fly. Used for choice assays.
- On an **optogenetic** rig an Optolid puts an LED over each well. The DFM's own
  firmware switches each LED on or off, every millisecond, from its own live
  reading of the signal and the rule of the program it was given.

### 2.2 From signal to licks and events

The pipeline runs on each well separately:

1. **Baseline subtraction.** A running median over `baseline_window_minutes`
   (default 3) is subtracted from the raw signal. It removes drift and
   differences between wells and boards. It assumes feeding is rare within any
   3-minute window; a fly feeding almost continuously can pull its own baseline
   up and flatten its long bouts.
2. **Licks.** A **lick** is one *sample* whose baselined signal is above
   `feeding_minimum`, inside a run of such samples that reaches
   `feeding_threshold` at least once. **A lick is a sample, not a tongue
   extension.** At 5 Hz, one second of continuous contact is 5 licks. Raw lick
   counts divided by `samples_per_second` are seconds of contact.
3. **Events.** An **event** (a bout, a meal) is a contiguous run of licks. Runs
   shorter than `feeding_minevents` samples are discarded. Then gaps of
   `feeding_event_link_gap` samples or fewer between two events are bridged
   (default 5 samples = 1 second), merging them into one event. Linking changes
   event counts, durations and intervals; it does not change lick counts or the
   lick preference index.
4. **Tasting.** Samples between `tasting_minimum` and `tasting_maximum` that
   feeding did not claim are tasting licks, grouped into tasting events. Tasting
   is *not* in the feeding summary. It appears only in
   `tidy_tasting_events.csv`, when that export was run, and inside the
   optogenetic light QC, which counts a touch as contact.
5. **Adaptive thresholds.** If `feeding_threshold` is negative, all four
   thresholds become fractions of each well's own maximum signal. This is rare;
   check the parameters.

### 2.3 Two-well choice

- **Well A and well B** are logical labels. `pi_direction: left` makes well A the
  odd-numbered well of each chamber; `right` makes it the even one. It is set per
  DFM, so experimenters can counterbalance the side on which the test food
  sits. `well_names` gives the labels (for example `A: Sucrose`, `B: Yeast`).
- **Preference index**: `PI = (LicksA − LicksB) / (LicksA + LicksB)`, from raw
  counts, from −1 (all licks at B) to +1 (all licks at A). **Positive means a
  preference for well A.** `EventPI` is the same formula on event counts. A
  chamber with no licks has an undefined PI (`NA`).
- **Dual-feeding correction** (`correct_for_dual_feeding`, on by default for
  two-well) gives any sample licked on both wells at once to the stronger well
  and detects licks again. Uncorrected crosstalk pushes PI toward zero.

### 2.4 The lick transform

By default (`transform_licks: true`) **every lick-count column in the summary
tables is the fourth root of the count**: `Licks = count ** 0.25`. A `LicksA` of
6.2 is about 1,478 licks, not 6.2. Events, durations, intervals, intensities and
PI are never transformed. Cumulative curves (`pr_cumulative_diff.csv`, the QC
plots) always use raw counts. Section 7.10 explains how to recover raw counts.

### 2.5 Experiment Types

The **Experiment Type** (`experiment_type:`) names the assay and decides which
analyses run. pyflic ships three:

| Type | Layout | Light it expects | What it adds |
|---|---|---|---|
| `Custom` (or no `experiment_type` key) | stated by `chamber_layout:` | any, or none | the general feeding summary; for an optogenetic recording, the light QC, the light table, the paired − yoked comparison (Paired and yoked module) and feeding in light and dark (Light and dark module) |
| `Hedonic` | `two_well` | none | bout duration at each well, plain and weighted by events (`treatment_means.csv`); the hedonic feeding plot |
| `ProgressiveRatio` | `two_well` | ProgressiveRatio or FixedRatio, with open-loop and dark intervals beside them | chamber groups, Training and Test phases, the paired fly's light QC, the paired − yoked difference, the breaking point, sucrose persistence |

A lab may build its own types from **Analysis Modules** (Progressive Ratio,
Hedonic, Paired and yoked, Light and dark). A lab type's copy travels with the
data as `experiment_type.yaml` beside `project.yaml` or `flic_config.yaml`; read
it to learn the type's modules, the light it expects, its default constants and
its **control comparisons** (section 6.4).

### 2.6 Optogenetics in one page

- The MCU's program is exported as **`Program.txt`**. pyflic reads it **only from
  the member's `data/` folder**. It gives each well's threshold, the **linkage**
  (wells lit together form a **linkage group**, one light circuit) and the
  **Paradigm** of each **Program Interval**.
- **Paradigm types**: OpenLoop (lit regardless), ClosedLoop (lit during contact
  and for a Hold after), FixedInterval (contact earns a fixed-length Activation,
  then a lockout), NonFeeding (dark while the fly feeds), ProgressiveRatio and
  FixedRatio (a number of feeding bouts earns a fixed-length Activation; the
  number rises along a series, or stays fixed). They fall into four **Paradigm
  Families**: open loop, closed loop, fixed length, non-feeding.
- **Roles** say what each chamber was to the light: **Paired** (its fly's feeding
  lights other chambers too), **Yoked** (lit when its partner earns light,
  whatever it does itself), **Mutual**, **Self** (its own feeding lights only
  itself), **Open** (lit regardless) and **Unlit**. Roles are read from the
  program or the data, never assigned, and are never treatments.
- **Generations** of data file: **1** (27 columns, older firmware; light from
  `OptoCol1` sampled at 5 Hz), **2** (90 columns, new columns empty; no Training
  information at all), **3** (90 columns filled, firmware 4.0.0; the firmware's
  own decisions are recorded: Paradigm, Phase, Activations, Requirement, `OnMs`
  to the millisecond). What can be checked, and how exactly, depends on the
  generation and on whether `Program.txt` is present.

---

## 3. Recognising the folder levels

pyflic has three structural levels, each recognised by a marker file.

| Level | Marker | Its children | What it owns |
|---|---|---|---|
| **Batch** | none: any folder with a Project anywhere beneath it | Projects, at any depth | no analysis; optionally `batch.yaml`, an exclusion sheet, `batch_ai_narrative.md` |
| **Project** | `project.yaml` | its *immediate* subfolders that hold a `flic_config.yaml`: the **Members** | the **Design** every Member inherits; the pooled **Combined Analysis**; the Project Report |
| **Experiment Directory** | `flic_config.yaml` | none | one recording: `data/`, and its own `qc/` and `analysis/` |

A **Member** is an Experiment Directory inside a Project. An Experiment
Directory outside any Project is a **standalone experiment**.

### 3.1 Discovery procedure

1. Find every `project.yaml` under the root you were given. Each is a Project.
   Do not search inside a Project for more Projects: its subfolders are Members.
2. For each Project, its Members are the immediate subfolders that contain a
   `flic_config.yaml`. Nothing deeper.
3. Any `flic_config.yaml` not directly under a Project is a standalone
   experiment.
4. A folder above one or more Projects is a Batch. A Batch never pools across
   Projects: each Project is its own design with its own statistics. Compare
   Projects only qualitatively, and say that they were analysed separately.
5. Look for **Blocked Members**: subfolders of a Project that hold a recording
   pyflic could not use, and so were not analysed.
   - DFM CSVs at the folder's root rather than in `data/` (an unfiled
     recording);
   - a `data/` with `DFM*.csv` but no `flic_config.yaml`;
   - a `flic_config.yaml` with no DFM CSV in `data/`.
   Name each one. A Project with a blocked member pools only the healthy ones.

### 3.2 Files to ignore, and files worth reading for context

| Path | What it is | What to do |
|---|---|---|
| `.pyflic_cache/` | cached feeding summaries | ignore |
| `<stem>_results/`, `analysis_<start>_<end>/` | outputs of pyflic versions before Facets; orphaned, never updated | ignore for results; if they are the *only* outputs, say the analysis predates the current pyflic and was not re-run |
| `Thumbs.db`, `~$*` | operating system and editor debris | ignore |
| `extra_files/` | loose files moved out of the way when a recording was filed | may hold notes or logs; read for context |
| `Notes.rtf`, `notes.txt` and the like | the experimenter's notes | read for context: they may name genotypes, foods, dates or problems; label anything you take from them |
| `ExpDesign.csv` | the R pipeline's design file | **not read by pyflic**; compare it with `flic_config.yaml` and report disagreements |
| `Program.txt`, `Messages.csv` **outside** `data/` | the MCU's program and log | **pyflic did not read them.** If `summary.txt` says "no Program.txt in data/" while a `Program.txt` sits at the member root, the light QC ran with less knowledge than it could have. Report it |
| `LoadedREnvironFile` and other R files | leftovers of the R pipeline | ignore |

### 3.3 The full tree

```
<batch>/                                   a Batch (no marker)
  batch.yaml                               optional: the designated Project Script
  remove_chambers.csv | .xlsx              optional: an exclusion sheet (a writer, never read at analysis time)
  batch_ai_narrative.md                    optional: an LLM synthesis of the Projects' own narratives
  <grouping folders>/...                   transparent
    <project>/                             a Project
      project.yaml                         design: (the authority) and Project Scripts
      experiment_type.yaml                 copy of a lab-built type, when the design names one
      plot_specs.yaml                      Publication Figure specs and styles (presentation only)
      remove_chambers.csv                  optional exclusion sheet
      <project name>_report.pdf            the Project Report
      figures/                             Publication Figures: <plot_id>.svg or .pdf
      analysis/                            the Combined Analysis (section 4.4)
      <member>/                            a Member (an Experiment Directory)
        flic_config.yaml                   dfms: block; normally no global:
        remove_chambers.csv                exclusions declared for this recording
        data/                              DFM<id>_<n>.csv, Program.txt, Messages.csv
        qc/                                QC outputs (section 4.3)
        analysis/                          results (section 4.2)
```

A Project name can contain spaces and punctuation (one real Project is named
`pyflic - prog assay`, so its files are `pyflic - prog assay_Stats.txt` and so
on). The prefix of the Project's files is the `name:` in `project.yaml`, not
necessarily the folder name.

---

## 4. Output file inventory

"One row per" is the grain of a table. Get it right before comparing anything:
a per-chamber table and a per-chamber-group table cannot be pooled row for row.

### 4.1 A member's root

| File | What it holds |
|---|---|
| `flic_config.yaml` | the `dfms:` block: DFM ids, chamber → treatment assignments, per-DFM `pi_direction`, `paired_chambers` / `chamber_groups` (Progressive Ratio), `optogenetics`; a standalone experiment also has `global:` |
| `remove_chambers.csv` | exclusions declared by hand: columns `group, dfm_id, chamber, note`. Only rows whose `group` is the active exclusion group (default `general`, or the design's `exclusion_group:`) apply |
| `data/` | raw DFM CSVs, and `Program.txt` and `Messages.csv` when copied in |
| `experiment_type.yaml` | a standalone experiment's copy of a lab type |

### 4.2 A member's `analysis/`

The **core** is written by every analysis run: auto-removal, `removed_chambers.csv`,
`summary.txt`, `feeding_summary.csv`, and `feeding_summary_facet.csv` when the
experiment has Facets. Everything else depends on the run and the experiment
type.

| File | One row per | What it is |
|---|---|---|
| `summary.txt` | — | the run in text. Read it first. Sections: pyflic version; config path; loaded DFMs; treatments; detection parameters; the experimental design table (DFM, chamber → treatment, **after** exclusions); *Chambers excluded (group '…')*; *Chambers removed by auto_remove_chambers* with reasons; per-DFM details (file version, **generation**, where light seconds came from, raw rows, actual time range, parameters); *QC summary* (integrity, data breaks, simultaneous feeding, when QC ran with the analysis); *Optogenetic light QC*; Progressive Ratio training, light QC and breaking point tables, with the breaking point at other gaps; and *Analysis Set (Run analysis)*: what was included, left out on purpose, and not applicable |
| `feeding_summary.csv` | analysed chamber, whole recording | the main per-chamber metrics (section 7). Only chambers assigned to a treatment and not excluded have a row |
| `feeding_summary_facet.csv` | analysed chamber × Facet | the same metrics per time window, with `Facet` and `FacetRange` first |
| `removed_chambers.csv` | auto-removed chamber | `DFM, Chamber, Treatment, Reason`, for example `LicksA=0 < min_untransformed_licks_cutoff=20` |
| `binned_feeding_summary.csv` | chamber × time bin | the metrics in fixed bins: `Interval` is the bin, written half-open as `"(0,30]"`, and `Minutes` is its midpoint. Lick columns are transformed like the summary's |
| `tidy_feeding_events.csv` | feeding event | `DFM, Chamber, Well, WellLabel, WellName, Treatment, StartMin, Licks, Duration, AvgIntensity, MaxIntensity`, then factors. A bout's `Licks` includes samples bridged by linking |
| `tidy_tasting_events.csv` | tasting event | the same shape, for tasting; only when the `tidy_export` script action ran with `kind: tasting` |
| `light_summary.csv`, `light_wells.csv` | chamber; well | the **light table** of an optogenetic experiment: what the light did (section 7.8) |
| `light_summary_facet.csv`, `light_wells_facet.csv` | the same × Facet | per window |
| `yoked_comparison.csv` | yoked linkage group × window | non-Progressive-Ratio yoked designs: `Paired<m>`, `Yoked<m>` (mean of the yoked chambers), `d<m>` = paired − yoked. **Absent means no yoked group existed**: pyflic deletes a stale one |
| `treatment_means.csv` | treatment | Hedonic: per well, `MeanDuration`, `SemDuration`, `WeightedMeanDuration`, `WeightedSemDuration`, `SampleSize` |
| `paired_yoked_diff.csv` | paired chamber group × Facet | Progressive Ratio: paired − yoked for each metric (`dLicksA`, `dPI`, `dPersistA`, …). The primary table for that type's statistics |
| `pr_cumulative_diff.csv` | chamber group × minute since training end | Progressive Ratio: `DiffCumLicks`, paired − yoked cumulative sucrose-well licks, **raw counts** |
| `pr_breaking_point.csv` | chamber group | Progressive Ratio: `BreakingPoint`, `BreakMin`, `Censored`, `TestMinutes`, `LargestRequirement`, `BreakRequirement`, `FirmwareEvents`, `LickFreeLightEvents`, `LightQC`, `ParadigmType` |
| `pr_light_qc.csv` | chamber group | Progressive Ratio light QC: training, light events, lick-free runs, trend, resting level, `Flags`, `Verdict`, `Excluded`, `Notes` |
| `pr_light_events.csv` | Test light event | Progressive Ratio: each light event's `Minutes` since training end, licks since the previous one, `LickFree`, `Counted` (in the breaking point), and on generation 3 `Requirement`, `FirmwareEvents`, `Phase` |
| `bootstrap_<metric>.csv` | treatment (or factor combination) | optional script action: chamber-level bootstrap `n, mean, sem, ci_low, ci_high` (percentile interval) |
| `compare_<metric>_<model>.csv`, `…_posthoc.csv` | model term; pair | optional: ANOVA (Type II) or a linear mixed model with a DFM random intercept, over the design factors with interactions; Tukey post hoc for one factor |
| `param_sensitivity_<parameter>.csv` | parameter value × treatment | optional: `mean_Licks`, `sem_Licks`, `mean_Events`, …, `mean_MedDuration` at each value of one detection parameter |
| `light_phase_summary.csv` | chamber × phase (light, dark) | optional: licks and events in each chamber's own light and dark, with `PhaseSeconds` |
| `bout_transition_matrix.csv` | chamber × transition | optional (two-well): counts of A→A, A→B, B→A, B→B consecutive bouts |
| `experiment_report.pdf` | — | the member's report: cover with *At a glance* verdicts, Quality control, Results with statistics tables, appendices of every metric and the parameters used |
| `*.png` | — | figures: `feeding_summary.png`, `binned_<metric>.png`, `dot_<metric>.png`, `well_comparison_<metric>.png`, `hedonic_feeding_plot.png`, `moving_*`, and for Progressive Ratio `pr_cumulative_diff.png`, `pr_still_responding.png`, `breaking_point_dfm<id>.png`, `pr_cumulative_licks_dfm<id>.png`, `pr_light_events_dfm<id>.png`, `pr_resting_level_dfm<id>.png` |

**Member-level statistics exist only in `experiment_report.pdf`.** No CSV or TXT
holds a member's treatment p-values. If you cannot read the PDF, you can still
describe a member from its CSVs, but its p-values are unavailable to you; say so.

### 4.3 A member's `qc/`

Most files have one copy per DFM, named `DFM<id>_…`.

| Path | One row per | What it is |
|---|---|---|
| `integrity/DFM<id>_integrity_report.csv` | DFM | `dfm_id, n_rawdata, generation, start_time, end_time, elapsed_minutes, elapsed_minutes_from_minutes_col, error_columns, index_increments_by_one, lost_records, repeated_records, unknown_instruction_rows, late_records` |
| `integrity/DFM<id>_integrity_report.txt` | — | the full text: error-flag counts by kind, each lost-record gap, repeated records, on generation 3 the Paradigm sequence and **DFM restarts**, and `Messages.csv` errors |
| `data_breaks/DFM<id>_data_breaks.csv` | gap | exists **only** for a DFM with breaks: `BreakIndex` (row number) and `GapSeconds` |
| `raw_signal/`, `baselined/`, `cumulative_licks/` | — | per-DFM PNGs: raw signal; baselined signal with the two thresholds dashed; cumulative licks (fourth root) |
| `simultaneous_feeding/DFM<id>_simultaneous_feeding_matrix.csv` | chamber (two-well) | `Licks1`, `Licks2` (raw lick samples at the lower- and higher-numbered well, after correction), `Both`, `MaxMinSignalAtBoth`, `HigherInCol1AtBoth`, and the same three with `_pre` (before the dual-feeding correction) |
| `bleeding/DFM<id>_bleeding_matrix.csv` | signalling well (two-well) | 12 × 12: rows `W<n>Sig`, columns `W<n>Resp`; the mean baselined signal of each responding well while the signalling well is above 50 |
| `bleeding/DFM<id>_bleeding_alldata.csv` | well | each well's mean baselined signal over the whole recording |
| `opto/opto_light_qc.csv` | linkage group | the optogenetic light QC verdicts (section 8.8) |
| `opto/opto_light_intervals.csv` | linkage group × interval | lit, unexplained and contact time per Program Interval, its Paradigm and Hold |
| `opto/opto_light_events.csv` | light event (an Activation on generation 3) | onset, end, duration, unexplained seconds, `OverrunSec`; on generation 3 the light period to the millisecond, `Family`, `K`, `RequirementMet`, `EndedBy` |
| `opto/opto_program.csv` | program interval | the program as pyflic read it |
| `opto/opto_light_dfm<id>.png` | — | *Light explained by licks* |
| `program/program_timeline_dfm<id>.png` | — | the Program timeline |

The Progressive Ratio QC tables (`pr_light_qc.csv`, `pr_light_events.csv`) and
their figures live in `analysis/`, not `qc/`.

**`qc/` may be absent or out of date.** The Hub's *Run analysis* writes nothing
into `qc/`; only *Run QC*, *Analyze all*, and a script's `run_qc` or Python's
default basic analysis do. No run deletes an old file. Compare the timestamps
of `qc/` files with `analysis/feeding_summary.csv`, and the parameters printed in
`summary.txt` with the thresholds drawn in the baselined plot, before relying on
`qc/`. The experiment report computes its own integrity and crosstalk tables
fresh, so its Quality control section is current with the report.

### 4.4 A Project's `analysis/` (the Combined Analysis)

Built by stacking each Member's **saved, already-filtered** tables and adding an
`Experiment` column (the Member's folder name). The Combined Analysis never
analyses a Member; a Member with no saved summary is left out.

| File | One row per | Built from |
|---|---|---|
| `<name>_Summary.csv` | analysed chamber, across Members | each Member's `feeding_summary.csv` |
| `<name>_Summary_Facet.csv` | chamber × Facet | each `feeding_summary_facet.csv` |
| `<name>_PairedYokedDiff.csv` | paired chamber group × Facet | each `paired_yoked_diff.csv` (Progressive Ratio) |
| `<name>_BreakingPoint.csv` | chamber group | each `pr_breaking_point.csv` (Progressive Ratio) |
| `<name>_LightQC.csv` | light QC group | each `analysis/pr_light_qc.csv` (`Source` = `PR`) and `qc/opto/opto_light_qc.csv` (`Source` = `Opto`), stacked with the union of their columns, so many cells are `NA` |
| `<name>_LightSummary.csv`, `_LightSummary_Facet.csv`, `_LightWells.csv`, `_LightWells_Facet.csv` | chamber; well | each Member's light table |
| `<name>_YokedComparison.csv` | yoked group × window | each `yoked_comparison.csv` |
| `<name>_Excluded.csv` | excluded chamber | `Experiment, DFM, Chamber, Source, Note`. `Source` is `manual` (from `remove_chambers.csv`) or `auto` (from `removed_chambers.csv`). For a manual row, `Note` holds the exclusion group's name, not the reason; read the Member's `remove_chambers.csv` for the reason |
| `<name>_Stats.txt` | — | the pooled statistics (section 9.4) |
| `ai_summary.txt` | — | an optional narrative written by a language model from the report's content; it is deleted whenever the Combined Analysis is rebuilt. Treat it as secondary and check every claim in it against the tables |

Outside `analysis/`: `<name>_report.pdf` (the Project Report, which adds what
`_Stats.txt` lacks: the control comparisons, the pooled *Paired and yoked*
section, the per-Member table and the pooled figures) and `figures/` (Publication Figures
named by plot id, such as `faceted_pi.svg`, `timecourse_licks.svg`,
`dot_pr_breaking_point.svg`).

**DFM ids repeat across Members.** Always identify a DFM as (Experiment, DFM).

### 4.5 Freshness, completeness and provenance

Check each of these and report anything that fails:

- **Stale member results.** If a Member's `remove_chambers.csv` is newer than its
  `analysis/feeding_summary.csv`, its results were computed before its current
  exclusions (the Hub calls this *re-run needed*). Those numbers describe a
  chamber population nobody asked for. Likewise, `analysis/analysis_settings.json`
  records the Design and config settings the summary was computed with; any that
  differ from `project.yaml`'s `design: global:` or the Member's
  `flic_config.yaml` now (other than `omit:` and the Slice keys) mean the
  numbers came from settings nobody is asking for any more.
- **Stale Combined Analysis.** If any Member's `feeding_summary.csv` (or other
  pooled table) is newer than the Project's `<name>_Summary.csv` or
  `_Stats.txt`, the pooled results predate that Member's latest analysis.
- **Leftover files.** Unticking an output leaves the file from an earlier run.
  A file much older than `summary.txt` may describe old parameters. The
  *Analysis Set* block at the end of `summary.txt` says what the latest run
  included, left out and found not applicable.
- **Missing Members.** `<name>_Stats.txt` prints `Members pooled : N (names…)`.
  `N` counts Members with saved results; the list in brackets names every
  Member with a config, including any left out. If `N` is smaller than the list,
  compare with the `Experiment` column of `<name>_Summary.csv` to see who is
  missing.
- **pyflic version.** `summary.txt` starts with `pyflic version: …`. Members
  analysed by different versions may differ in what they contain (for example a
  Member analysed before the breaking-point table existed has none). Report
  version differences between Members.
- **Parameters.** Inside a Project the Design forces one set of detection
  parameters and constants on every Member, so they should agree; confirm from
  each `summary.txt`. Between standalone experiments nothing forces agreement:
  compare parameters before comparing results.

---

## 5. Reading the configuration

### 5.1 `project.yaml`

```yaml
name: drug_trial                 # prefix of the Project's output files
design:
  global:                        # the authority for every Member's settings
    experiment_type: Hedonic     # or Custom (default), ProgressiveRatio, a lab type
    chamber_layout: two_well     # only for Custom or a lab type with layout "either"
    transform_licks: false       # fourth-root transform of lick counts (default true)
    facet_cutoffs: [60]          # minute boundaries of the Facets
    facet_labels: [Early, Late]  # optional names for them
    facet_source: cutoffs        # or program / program_cycles (Facets from Program.txt)
    optogenetics: auto           # auto (default), true, false
    exclusion_group: general     # which rows of each remove_chambers.csv apply
    control_level: w1118         # only under a type with a Treatment control
    params: {...}                # detection parameters (section 13.2)
    well_names: {A: S5, B: S5Y5} # labels of well A and well B
    constants: {...}             # exclusion cutoffs and QC thresholds (section 13.2)
    experimental_design_factors:
      Genotype: [Ctrl, Exp]
      Drug: [Vehicle, Drug]
    omit: {qc: [...], analyses: [...], plots: [...]}   # outputs deliberately left out
scripts: [...]                   # Project Scripts (what a Batch Run runs)
```

A Member may not deviate from any `design.global` key; a Member that did would
have failed to load. A Project with no `design:` validates its Members against
each other instead.

### 5.2 A Member's `flic_config.yaml`

```yaml
dfms:
  - id: 1
    params: {pi_direction: left}       # physical keys only inside a Project
    paired_chambers: [1, 4, 5]         # Progressive Ratio: the paired chamber of each pair
    # chamber_groups: single           # Progressive Ratio: six solo groups, no yoked controls
    optogenetics: false                # this DFM ran without a lid
    chambers:
      1: Ctrl,Vehicle                  # one level per factor, in declaration order
      2: Exp,Vehicle
```

- A chamber **not listed** under `chambers:` was never part of the design (an
  empty position). It is not an exclusion and is not counted as one.
- A standalone experiment has its own `global:` block with the same keys as a
  Design.
- `excluded_chambers:` inside a DFM entry is a retired key that pyflic ignores
  with a warning. If you see it, those exclusions were **not** applied.

---

## 6. Recognising treatments and controls

### 6.1 Where treatments are declared

1. `design.global.experimental_design_factors` in `project.yaml` (or `global:` of
   a standalone `flic_config.yaml`): the factor names and their permitted
   levels.
2. `dfms: … chambers:` in each Member's `flic_config.yaml`: which chamber got
   which treatment.
3. The `Treatment` column of every per-chamber table, followed by one column per
   factor.
4. `summary.txt`: `Treatments: [...]` and the design table (after exclusions).
5. The report cover: treatments with their chamber counts.

### 6.2 How the Treatment label is formed

- **No factors declared**: the label is the free name typed for the chamber
  (`Sucrose`, `Water`, `Chrim`).
- **Factors declared**: the label is the levels **joined with an underscore**, in
  the order the factors are declared. With `Genotype: [Ctrl, Exp]` then
  `Drug: [Vehicle, Drug]`, a chamber assigned `Exp,Drug` has `Treatment =
  Exp_Drug`, `Genotype = Exp`, `Drug = Drug`. Use the factor columns to split a
  label, not string surgery: a level may itself contain an underscore.
- With one factor, the factor column repeats the Treatment.
- Some plots and the bootstrap group by a synthetic label joined with `" / "`
  (`Exp / Drug`). It is the same grouping.

### 6.3 What is *not* a treatment

| Column or concept | What it is | Why it matters |
|---|---|---|
| `Role` | what a chamber was to the light (`Paired`, `Yoked`, `Mutual`, `Self`, `Open`, `Unlit`; lowercase `paired`/`yoked` under Progressive Ratio) | read from the program; a within-treatment control, never a treatment level |
| `Group` | a Progressive Ratio chamber group (1–3 for pairs, 1–6 for solo groups), or a linkage group in the light QC | the unit of pairing, numbered within a DFM |
| `Facet`, `FacetRange` | a time window | compare treatments *within* a Facet; never pool Facets as if they were treatments |
| `Experiment` | the Member | a source of variation; DFM is nested within it |
| `DFM` | the board | a source of variation (shared device, shared position) |
| well A / well B | the two foods of a two-well chamber | the PI compares wells *within* a chamber |
| `Phase` (`A`, `M`) | a well's Acclimation state on generation 3 | firmware state, not a design factor |

### 6.4 Identifying controls

In order of reliability:

1. **Declared**: a lab type with a **Treatment control** names a factor in
   `experiment_type.yaml`, and the Design names its control level as
   `control_level:`. The reports then compare every other level with it
   (Dunnett's test). This is the only fully explicit control.
2. **Declared Role comparison**: a lab type may name two Roles to compare (for
   example Paired against Yoked, or Self against Unlit). The reports lead with
   it.
3. **Structural controls in optogenetic designs**: a **Yoked** chamber is its
   Paired partner's control (same light, no contingency); **Unlit** chambers
   and dark Program Intervals are no-light controls; an open-loop interval beside
   a closed-loop one separates the light's direct effect from its contingency.
   A Progressive Ratio paired design's **paired − yoked difference** is built on
   this.
4. **Inferred from names** (say it is an inference): `Ctrl`, `Control`, `WT`,
   `Vehicle`, `Mock`, `Sham`, `0mM`, `Unpaired`, `w1118`, `Canton-S` / `CS`,
   `Oregon-R`, `attP2`, `attP40`, empty-GAL4 or empty split-GAL4 lines, parental
   controls such as `GAL4/+` and `UAS/+`, `-ATR` or `noATR` (flies not fed the
   all-trans-retinal cofactor that channelrhodopsins such as Chrimson need).
   Labels such as `Chrim` or `CsChrimson` usually name an optogenetic effector,
   not a control. Do not assert any of this; state it as a reading of the label.

If no control can be identified, compare treatments symmetrically and say that
no control was declared.

### 6.5 Build the design table

Before reading any result, tabulate the design yourself:

1. **Assigned**: from every Member's `flic_config.yaml`, count chambers per
   treatment, per Member and per DFM. (`summary.txt`'s design table already
   omits excluded chambers, so it cannot give you the assigned count.)
2. **Analysed**: count rows per treatment in `feeding_summary.csv` (or the
   Project's `<name>_Summary.csv`, by `Experiment`). For Progressive Ratio, count
   chamber groups in `pr_breaking_point.csv` and paired groups in
   `paired_yoked_diff.csv` per Facet.
3. **Excluded**: manual (`remove_chambers.csv`, active group) and automatic
   (`removed_chambers.csv`, with reasons), per treatment.
4. **Factors**: is the factorial design complete (every combination of levels
   present) and balanced? Which combinations are missing or thin?
5. **Units**: how many chambers, chamber groups, DFMs and Members stand behind
   each treatment? The number of DFMs and Members is the real replication for
   anything that varies by device or by day.

### 6.6 Check for confounds

A treatment difference can only be attributed to the treatment when the
treatment is not tied to something else. Check each of these, and report every
one you find:

- **Treatment × Member.** If a treatment occurs in only one Member (for example
  each Member is one dose), treatment is confounded with recording day, fly
  batch and rig set-up. The pooled per-chamber test cannot tell them apart, and
  the mixed model cannot estimate a treatment effect separately from the
  Member effect. Many Projects are built exactly this way ("a dose series, a
  genotype panel"); say that the comparison is between recordings as well as
  between treatments.
- **Treatment × DFM.** If every chamber of a DFM got the same treatment (DFM 1
  all control, DFM 2 all experimental), treatment is confounded with the board
  and its position. Chambers on one DFM are not independent.
- **Treatment × side.** In a two-well design, cross-tabulate treatment with the
  DFM's `pi_direction`. If the test food sat on the same side for every chamber
  of one treatment, a side bias is indistinguishable from a preference.
- **Treatment × generation or firmware.** Members recorded with different data
  generations measure light differently (section 8.3).
- **Treatment × exclusion.** Unequal exclusion rates across treatments (section
  8.7).
- **Treatment × recording length.** Compare `actual range minutes` per DFM in
  `summary.txt`; a treatment recorded for less time has fewer licks for that
  reason alone.

### 6.7 Special designs

- **One treatment.** The reports say "One treatment, …, so nothing is compared
  between treatments", and `_Stats.txt` says `(no comparable treatment groups
  found)`. That is a design, not a failure. Describe the treatment, and for
  Progressive Ratio and yoked designs report the within-treatment tests (paired
  − yoked against zero), which do not need a second treatment.
- **Progressive Ratio.** Both chambers of a pair always carry the same treatment.
  Treatments differ *between* chamber groups. The paired fly and its yoked
  partner are compared *within* a group.
- **Factors crossed with Members.** When different Members carry different
  levels of one factor and share another, say which comparisons are within a
  Member and which are across Members.

---

## 7. Metric dictionary

Units: minutes for `StartMin`, `EndMin`, `Minutes`, `TrainingMinutes`,
`BreakMin`, `PersistA`, `TestMinutes` and Facet ranges; seconds for durations,
intervals and anything ending `_sec` or `Sec`; milliseconds for anything ending
`Ms`. Facet windows are half-open, `(start, end]`.

### 7.1 Feeding summary, every layout

| Column | Meaning | Traps |
|---|---|---|
| `Treatment`, factor columns, `DFM`, `Chamber` | identity | DFM is per Member |
| `Licks` (single-well) | lick samples | **fourth-root transformed** when `transform_licks` is true |
| `Events` | feeding events (bouts) | depends on `feeding_event_link_gap` |
| `MeanDuration`, `MedDuration` | mean and median event duration, seconds | `NA` when the chamber had no events; the median is the more robust |
| `MeanTimeBtw`, `MedTimeBtw` | mean and median length, seconds, of the stretches without feeding, including before the first event and after the last | meaningless for a chamber that never fed (it reads as the whole window, then 0) |
| `MeanInt`, `MedianInt`, `MinInt`, `MaxInt` | baselined signal inside events (contact strength, arbitrary units) | can be negative at the minimum; describe, do not over-interpret |
| `OptoOn_sec` | seconds the well's light was on | `OnMs` sums on generation 3, sampled `OptoCol1` before |
| `StartMin`, `EndMin` | the window the row covers | `0, 0` means the whole recording |

### 7.2 Two-well additions

`PI`, `EventPI` (−1 to +1, positive = well A preferred, from raw counts, never
transformed), and per-well columns suffixed `A` and `B`: `LicksA`, `LicksB`
(transformed), `EventsA`, `EventsB`, `MeanDurationA/B`, `MedDurationA/B`,
`MeanTimeBtwA/B`, `MedTimeBtwA/B`, `MeanIntA/B`, `MedianIntA/B`, `MinIntA/B`,
`MaxIntA/B`, `OptoOn_sec_A`, `OptoOn_sec_B`. There is **no** combined `Licks`
column in a two-well summary.

`PI` and `EventPI` can disagree: many short visits to one well and a few long
meals at the other give opposite signs. That disagreement is informative;
report which index a conclusion rests on.

### 7.3 Optogenetic additions to the feeding summary

`Role` (beside `Chamber`), `LitSec` (seconds the chamber's own light was on,
counted once however many of its wells were lit), `Activations` (on-periods its
wells' own Paradigms started; blank before generation 3), `OptoLightQC` (its
linkage group's flags; empty when clean) and `OptoUnexplainedFraction`.

### 7.4 Progressive Ratio additions

| Column | Meaning |
|---|---|
| `Group` | chamber group within the DFM |
| `Role` | `paired` or `yoked` (every chamber of a solo-group design is `paired`) |
| `TrainingMinutes` | the group's training end, minutes from the recording start; the same on both chambers of a pair; blank if training never completed |
| `TrainingComplete` | `True`, `False`, or `NA` when the recording has no Training information (generation 2) |
| `LightOn_sec` | seconds the group's light was on |
| `LightQC` | the group's Progressive Ratio light QC flags; empty when clean |
| `LickFreeLightEvents` | Test light events the paired fly was credited no sucrose lick for |
| `PersistA` | sucrose persistence: minutes from training end to the chamber's last sucrose feeding event before a pause longer than `pr_break_gap_min`; blank on Training rows |
| `PersistACensored` | `PersistA` is a lower bound: the fly was still feeding when its Test window ended |

In `paired_yoked_diff.csv`: `dLicksA`, `dLicksB`, `dEventsA`, `dEventsB`, `dPI`,
`dEventPI`, `dMedDurationA`, `dMedDurationB`, `dPersistA`, `dPersistCensored`,
each **paired minus yoked** within one group and one Facet. `dLicksA` is the
difference of the two flies' *transformed* lick values when the transform is on
(for example 2.659 − 2.576 = 0.084), not a difference of counts.

Facets under Progressive Ratio are `Training` and `Test`, split at each group's
own training end, so `FacetRange` differs between groups. A generation-2 group
has one Facet, `Recording`, and no Test.

### 7.5 Facet columns

`Facet` is the label (`0-60 min`, `60+ min`; a type's names such as `Training`;
`facet_labels`; or under Program Facets the Paradigm's name: `PR10`, `I2 Open`
when one Paradigm runs in two Program Intervals, `PR10 C3` under
`program_cycles`). `FacetRange` is the window, `(start, end)` in minutes; under
Program Facets of a Repeating program it can list several windows. A Program
Facet a DFM never ran has no rows for that DFM. On generations 1 and 2 Program
Facets are laid out from the clock schedule and are approximate; `summary.txt`
says so.

### 7.6 Binned summary

The same metrics in fixed bins. Use it for *when* feeding happened. Bins are
half-open and labelled in `Interval`; `Minutes` is the bin's midpoint. The bin
width is whatever the run used; read it from `Interval`. Narrow bins are noisy;
a short-lived effect can vanish in wide ones.

### 7.7 Event statistics

`tidy_feeding_events.csv` is one row per bout: use it for bout-level
distributions (durations, timing, intensity) and for raw lick totals per well
(sum `Licks` by chamber and well, remembering that a bout's `Licks` includes
bridged gap samples, so the sum slightly exceeds the summary's lick count).

### 7.8 The light table

`light_summary.csv` (per chamber) and `light_wells.csv` (per well) carry the
Treatment, factors, `Role` and `Generation`, then:

| Columns | Meaning |
|---|---|
| `Paradigm`, `ParadigmType`, `ProgramInterval`, `Cycle` | what ran over the window, or `mixed` |
| `WindowMinutes`, `LitSec`, `LitFraction`, `LitSource` | light received; `LitSource` is `OnMs` (generation 3, exact) or `OptoCol1` (sampled) |
| `LightOnsets` | times the light came on; on generation 1 the nearest thing to an Activation count |
| `Activations`, `ActivationsPerHour` | on-periods the well's own Paradigm started |
| `AcclimationActivations`, `AcclimationEndMin`, `AcclimationComplete`, `MainActivations` | Paradigms with Acclimation |
| `FirmwareEvents`, `LastRequirementMet`, `ProgressAtEnd`, `EventsPerActivation` | ratio Paradigms |
| `MedianActivationSec`, `CutOffs` | closed loop |
| `MaxActivations`, `RewardEfficiency`, `MedianReturnSec`, `LockedOutContactSec` | FixedInterval |
| `UnlitSec`, `Suppressions`, `MedianSuppressionSec` | NonFeeding |
| `LicksPerMinLit`, `LicksPerMinDark`, `LightFeedingIndex` | open loop: licks per minute in the chamber's own light and in the dark, and (lit − dark)/(lit + dark) |
| `PyflicEvents`, `PyflicLicks` (per well), `EventsA`, `EventsB` (per chamber) | pyflic's own counts, **raw**, for comparison |

A column is `NA` where the well ran no Paradigm that defines it, everywhere
without the DFM's `Program.txt` section for type-dependent columns, and for
every firmware column on generations 1 and 2. `NA` means *not defined*, never
zero. A Yoked, Open or Unlit chamber's Activations are zero by construction;
the statistics leave those chambers out of Activation measures.

**Headline metrics by Paradigm Family** (the ones the reports test): open loop
`LitSec`, `LightFeedingIndex`; closed loop `Activations`, `LitSec`,
`MedianActivationSec` (and `CutOffs` with `MaxOnMs`); fixed length, interval
`Activations`, `RewardEfficiency`, `MedianReturnSec`; fixed length, ratio
`Activations`, `LastRequirementMet`, `FirmwareEvents`; non-feeding
`LitFraction`, `Suppressions`; Acclimation `AcclimationEndMin`. Without a
generation-3 recording and a program, the four metrics `Activations`, `LitSec`,
`FirmwareEvents`, `AcclimationEndMin` are reported, and only `LitSec` has
values.

### 7.9 Empty values

`NA` is deliberate. A fly that never fed has no bout duration, not a duration of
zero; a chamber with no licks has no PI. Tests drop `NA` rows metric by metric,
so **n can differ between metrics of the same table**. Report n per metric
when it matters.

### 7.10 Recovering raw lick counts

When `transform_licks` is true (check `summary.txt` or the Design):

- A chamber's raw count is the transformed value to the fourth power: `LicksA =
  9.599` is `9.599⁴ ≈ 8,490` licks, which at 5 Hz is about 1,698 seconds (28
  minutes) of contact.
- The same raw counts are written untransformed in two other places: `Licks1` /
  `Licks2` of the simultaneous-feeding matrix (by well number, not by A/B), and
  `PyflicLicks` of `light_wells.csv` (optogenetic experiments). In the repository's
  test Project these three agree exactly (8,490).
- **Do not back-transform a mean.** The mean of fourth roots, raised to the
  fourth power, is not the mean count; it is closer to a geometric-type average
  and is smaller. To state an average count, back-transform each chamber and
  average those, and label the result as derived.
- `min_untransformed_licks_cutoff` and the counts quoted in removal reasons are
  raw counts.

### 7.11 The primary measures by experiment type

| Design | Lead with | Secondary |
|---|---|---|
| single-well consumption | `Licks`, `Events`, `MedDuration` | `MeanDuration`, intervals, time courses |
| two-well choice | `PI` (and `EventPI`) | per-well licks and events, `MedDurationA/B`, preference over time |
| Hedonic | median bout duration at each well, plain and event-weighted (`treatment_means.csv`) | PI, licks |
| Progressive Ratio, paired design | Test-phase `dLicksA`, `dPI`, `dPersistA`; the breaking point; the cumulative difference curve | the other `d` metrics; per-chamber metrics |
| Progressive Ratio, solo groups | the breaking point (still-responding curve and log-rank) | `PersistA`, per-chamber metrics |
| optogenetic, any type | the feeding metrics of the layout, by Facet or Program Facet; the light table's headline metrics | yoked comparison, light and dark feeding |

A lab type may name control comparisons, which then lead its reports.

---

## 8. Quality control

### 8.1 Principles

- QC asks whether the numbers describe the flies, or the instrument.
- **Most checks flag; people decide.** Only the design's `constants:` cutoffs,
  and the Progressive Ratio group rules, remove chambers automatically. Optogenetic
  light QC failures are, by default, flagged and **kept** in the numbers.
- QC verdicts concern the instrument and the light, not the biology. A clean QC
  says the measurement was sound; it does not make an effect real.

### 8.2 Build a QC ledger

For every Member (or standalone experiment), record:

| Item | Source |
|---|---|
| DFMs, their generation, elapsed minutes, start and end clock time | `summary.txt` *DFM details*; `qc/integrity/*.csv` |
| integrity: index continuous, lost records, repeated records, error flags, restarts | `qc/integrity/*` |
| data breaks: count and longest gap | `qc/data_breaks/*` (absent means none, if integrity ran) |
| crosstalk: `Both_pre` per chamber; largest off-diagonal bleeding | `qc/simultaneous_feeding/*`, `qc/bleeding/*` |
| exclusions: manual and automatic, with reasons, by treatment | `remove_chambers.csv`, `analysis/removed_chambers.csv`, `summary.txt` |
| `Program.txt` present in `data/`? at the root instead? | the file system; `summary.txt` *Optogenetic light QC* `Program:` line |
| optogenetic light QC: groups failed, warned, ok; flags; likely cause; excluded or retained | `qc/opto/opto_light_qc.csv`; `summary.txt` |
| rule checks (generation 3 with `Program.txt`) | `opto_light_qc.csv` rule columns; report *Rule check* subsections |
| Progressive Ratio: training completion, light QC verdicts, lick-free runs, resting level, censoring, Test window lengths, restarts | `pr_light_qc.csv`, `pr_breaking_point.csv`, `summary.txt` |
| freshness and provenance | section 4.5 |
| which checks did not run | missing files; the report's notes |

Then summarise the ledger per treatment.

### 8.3 Recording integrity

From `qc/integrity/DFM<id>_integrity_report.csv` and `.txt`:

- **Elapsed time.** `elapsed_minutes` and `elapsed_minutes_from_minutes_col`
  should agree with each other and with the other DFMs. A DFM much shorter than
  its siblings stopped early; every chamber on it has less time to feed.
- **`index_increments_by_one`.** `False` means records are missing or repeated.
- **`lost_records`.** Records the DFM could not deliver (one record per 200 ms).
  A few change little. A long gap removes that stretch from every chamber on the
  DFM; treat time-binned results across it with care. On generation 3 a lost
  record count is cross-checked against `Messages.csv`.
- **`repeated_records`.** Blocks written twice; pyflic keeps the first copy and
  drops the second. Harmless.
- **Error flags.** Counts of the firmware's error bits. pyflic counts them and
  does not interpret them. *Overflow* beside lost records is one event seen twice.
  On generation 3, *No Ack with nothing lost or repeated* is a known harmless
  artefact of older PyMCU builds.
- **DFM restarts** (generation 3). The DFM restarted on its own (power dip, loose
  cable): an `Index` gap, rows under no instruction, then the Paradigm starts
  afresh (Acclimation and the Requirement series start over). Results across a
  restart are two runs of the program. A Progressive Ratio Test window ends at a
  restart.
- **Generation.** Note each DFM's generation. A Project mixing generations has
  Members whose light was measured differently (`OnMs` versus sampled
  `OptoCol1`), and generation-2 Members have no Training information.
- The report gives each DFM **ok** (no breaks, no error flags, no lost records,
  continuous index) or **warning**. A warning removes nothing.

### 8.4 Data breaks

A break is a gap between consecutive samples longer than four sample periods
(0.8 s at 5 Hz). The CSV exists only for a DFM with breaks. Many short breaks cost
seconds; a long break costs that stretch on every chamber of the DFM. Breaks on
one DFM only point at that board or its cable, not at the flies.

### 8.5 Signal plots (when you can see images)

- **Raw signal**: a flat well with no deflections is an empty position or a dead
  well; a creeping background is drift; a step is a disturbance (moved plate,
  refilled well, bubble).
- **Baselined signal**: the most useful figure. Between bouts the trace should
  sit near zero. The upper dashed line is `feeding_threshold`, the lower
  `feeding_minimum`. Noise often crossing the upper line inflates licks; real
  bouts barely reaching it are being lost. A trace dipping below zero around
  long bouts means sustained feeding is pulling up the baseline.
- **Cumulative licks** (fourth root): a steady slope is steady feeding, a
  plateau a fly that stopped, a step a burst; a curve that never leaves zero is a
  well with no licks; one chamber far steeper than its neighbours is either a
  heavy feeder or a signal problem.

### 8.6 Two-well crosstalk

- **Simultaneous feeding.** A fly cannot drink from both wells at once, so samples
  licked on both are crosstalk or a fly bridging the wells. With the correction
  on, `Both` is normally 0 after it; read `Both_pre`. Express `Both_pre` as a
  share of the chamber's lick samples (`Licks1 + Licks2`). A chamber with common
  crosstalk deserves caution in its PI. State whether the correction was on
  (`correct_for_dual_feeding` in `summary.txt`).
- **Bleeding.** In the 12 × 12 matrix, ignore the diagonal (a well responding to
  itself, large by construction). Off the diagonal every value should be near
  zero. The report warns when the largest off-diagonal response on a DFM exceeds
  `feeding_threshold`, because such a response would register as licks that
  never happened. Off-diagonal values of a few units against a threshold of 20
  are normal.

### 8.7 Exclusions

| Source | Where | Meaning |
|---|---|---|
| manual | `remove_chambers.csv`, rows of the active group | the experimenter's judgement (`note` says why: escaped fly, dead well, noisy signal) |
| auto: no usable lick count | `removed_chambers.csv` | always applied: a well's lick value was `NaN` |
| auto: `min_untransformed_licks_cutoff` | `removed_chambers.csv` | a well had fewer raw licks than the cutoff over the recording (20 under Hedonic and Progressive Ratio; off under Custom unless set) |
| auto: `max_med_duration_cutoff` | `removed_chambers.csv` | two-well: a median event duration above the cutoff at either well (13 s by default where set): a stuck or bleeding signal |
| auto: `max_events_cutoff` | `removed_chambers.csv` | two-well: an implausible number of events |
| auto: `exclude_failed_opto_chambers` | `removed_chambers.csv` | every chamber of a linkage group that failed the optogenetic light QC; **off by default** |
| auto: `require_training_complete` | `removed_chambers.csv` | Progressive Ratio: every chamber of a group whose paired fly never finished training |
| auto: `exclude_failed_pr_groups` | `removed_chambers.csv` | Progressive Ratio: every chamber of a group that failed the Progressive Ratio light QC |

What to do with them:

- **Count exclusions per treatment**, manual and automatic separately, as a share
  of the chambers assigned. If one treatment lost far more chambers than another,
  the chambers that remain may not be comparable (for example, a treatment that
  suppresses feeding loses its weakest feeders to the lick cutoff and its
  remaining mean is biased upward). Report the counts whatever you conclude.
- **Read the reasons.** Removals for zero licks at both wells are usually empty or
  dead positions. Removals for too few licks in one treatment only may be the
  treatment's effect, not a fault, and excluding them changes the question.
- **A pair needs both chambers.** Under Progressive Ratio a removed chamber takes
  its whole pair out of the difference table, the difference curve and the
  breaking point, while its partner may remain in the per-chamber tables.
- Inside a Project the active exclusion group and the cutoffs are the Design's,
  so every Member is filtered by one rule. Between standalone experiments, check.

### 8.8 The optogenetic light QC

**The question.** The firmware lights a linkage group from its own live reading
of the group's trigger wells; pyflic counts licks afterwards. When a well's
resting signal drifts up, the firmware sees constant contact and keeps the light
on while pyflic sees no licks. The light then describes the sensor, not the fly.
The light QC measures the share of judged lit time with no lick or touch at a
trigger well behind it (*unexplained light*), per linkage group, over the whole
recording.

**What it can know depends on the inputs.**

| Situation | Consequence |
|---|---|
| generation 3 with `Program.txt` | the firmware's decisions are recorded; full QC plus every rule check |
| generation 3 without `Program.txt` | Paradigm types and settings unknown; rule checks do not run; loud warning *firmware 4.0.0 recording without Program.txt*; unexplained light can warn but never fail |
| generations 1–2 with `Program.txt` | thresholds, linkage and Hold known; pyflic emulates the firmware's trigger to tell drift from hardware (`EmulationApproximate` when the data begin late) |
| generations 1–2 without `Program.txt` | linkage inferred from wells lit together, every lit well treated as a trigger well, Hold assumed (`opto_default_decay_ms`); **unexplained light can warn but never fail**; Roles are blank |

**Verdicts and flags** (`opto_light_qc.csv`: `Flags`, `Verdict` ∈ `ok` /
`warning` / `failed`, `Excluded`, `LikelyCause`, `Notes`):

| Flag | Verdict | Meaning |
|---|---|---|
| unexplained light | **fail** | ≥ 30 % of the group's judged lit time had no lick or touch behind it (judged only once ≥ 30 s are unexplained) |
| partly unexplained light | warning | ≥ 10 % unexplained |
| light while off | **fail** | lit while every well of the group was `Off`, or under the idle Paradigm |
| no light recorded | **fail** (DFM) | `optogenetics: true` and the DFM was never lit: unplugged lid or unloaded program |
| broken yoking | **fail** | generation 3: linked wells' light differs, or a yoked (`Off`) well counted its own Activations; the yoked fly did not get its partner's light |
| Paradigm not in program | **fail** (all groups of the DFM) | the rows name a Paradigm `Program.txt` does not define: the file is another recording's |
| unlit feeding | warning | ≥ half of closed-loop feeding bouts never lit the group |
| no light events | warning | feeding at trigger wells, never lit |
| open loop not lit | warning | an open-loop window lit < 95 % of its time |
| non-feeding light during feeding | warning | NonFeeding light on through > 10 % of feeding |
| Acclimation count mismatch, firmware events disagree, lost records, unknown-instruction rows, late records, program mismatch, no program section, no Program.txt for firmware 4.0.0 | warning | bookkeeping disagreements; thresholds set before bench data existed, so treat as reasons to look |
| rule checks (section 8.9) | fail or warning | each Paradigm checked against its own rule |

**`LikelyCause`** for a flagged group: *drifting baseline or sustained contact*
(most unexplained light fell where the firmware saw contact and pyflic did not:
the light followed the sensor) or *hardware, linkage or program mismatch* (light
the firmware had no reason to switch on). `UnexplainedOnsetMin` is the start of
the first 30-minute window in which the unexplained share reached the warning
level.

**In a Project**, every flagged group appears in `_Stats.txt` and the Project
Report with a **status**:

- **excluded**: its Member's auto-removal took its chambers out;
- **retained (exclude_failed_…: false)**: it **failed but is in the pooled
  numbers**, because the exclusion switch was off;
- **kept — warning only**: it was only warned.

**How it should change your interpretation.** A failed light makes the light-on
numbers wrong; it does not make the feeding record wrong. So:

- For conclusions about **feeding** (licks, PI, durations), a retained failed group
  is a caveat; check whether results hold without it if you can (count how many
  such groups each treatment has).
- For conclusions about **the light or its contingency** (light-table metrics,
  paired − yoked differences, light and dark feeding, anything that assumes the
  light followed the fly), a retained failed group is a serious threat. Name it
  and say the affected numbers include it.
- Failures concentrated in one treatment bias the comparison.

### 8.9 Rule checks (generation 3 with `Program.txt`)

The firmware keeps each Paradigm's rule to the millisecond, so a group that
breaks one is **first a question about the file**: most often the `Program.txt`
in `data/` is not the program that ran; otherwise the LED or its record is
faulty.

| Check (column) | Verdict |
|---|---|
| short activations (`ShortActivations`): closed-loop light shorter than its Hold | **fail** above 5 % of judged Activations |
| max on exceeded (`OverMaxOnActivations`) | **fail** on any |
| wrong activation length (`WrongLengthActivations`): fixed-length light not `ActivationMs` | **fail** above 5 % |
| lockout broken (`LockoutViolations`): FixedInterval onsets closer than the lockout | **fail** above 2 % |
| requirement series wrong (`SeriesMismatches`): a ratio Requirement off its series | **fail** on any |
| contact without activation (`MissedContacts`) | warning above 25 % of free contact runs |
| non-feeding dark (`NonFeedingDarkSec`) | warning above 10 % |
| interval started late (`IntervalLagSec`) | warning above 5 s |
| DFM restarted (`Restarts`) | warning |
| `CutOffs`, `RelightDelayMs` | reported, no verdict |

All are `NA` on generations 1–2 and without `Program.txt` (except `Restarts`).
`NA` means *not checked*, not *passed*. Activations merged with other light or
cut short by a boundary are not judged; the group's `Notes` give the count.

### 8.10 Progressive Ratio QC

| Check | Verdict | Where |
|---|---|---|
| training never completed | removes the group (`require_training_complete`) | `TrainingComplete = False` |
| no training information (generation 2) | nothing removed; one `Recording` Facet, no Test, no breaking point, no persistence; never pooled with other Members' Test | `TrainingComplete = NA`; notes |
| self-triggered light: ≥ `pr_lick_free_run` (5) consecutive lick-free Test light events | **fail**, removed by default | `LongestLickFreeRun`, `LickFreeRunStartMin` |
| implausible training: training completed with light events and no sucrose lick | **fail**, removed by default | `TrainingLicks = 0` |
| requirement series wrong (generation 3) | **fail**, removed by default | `Notes` |
| no increasing trend in licks per light event (not judged under a fixed ratio) | warning | `TrendRho` < `pr_trend_min_rho` |
| resting level rise / elevated at the sucrose well | warning | `RestingRise`, `RestingRatio` |

Normal, not flagged: an occasional lick-free light event (the firmware responds
to touches below pyflic's threshold); fewer training licks than training light
events; a group with no Test light events (a breaking point of 0). On generation
1, *training-flag notes* that the other wells of a group stayed flagged to the
end are QC information, not errors; the analysis follows the paired sucrose well.

**Two light QCs can disagree, and both can be right.** The Progressive Ratio QC
counts light *events*; the optogenetic QC measures lit *time*. A group lit for
hours by one long light event passes the first and fails the second. Read both.

**Censoring and Test windows.** Count censored groups per treatment
(`Censored = True`, written `n+`). Compare `TestMinutes` across groups: late
training, the end of the ratio Paradigm or a DFM restart shortens a group's
window and makes censoring likelier. The breaking point at other gaps (60, 120,
240 minutes) is in `summary.txt`; a ranking of treatments that changes with the
gap is not robust.

**Detection matters more here than anywhere.** Whether a light event counts
depends on pyflic detecting at least one lick. Detection that misses brief licks
turns real responses into lick-free events, lowers breaking points and can fail
groups as self-triggered. A note *the firmware counted rising Events per light
event; pyflic's licks did not follow: check detection* means exactly that.

### 8.11 Severity and propagation

Grade each finding and carry it into your conclusions.

| Severity | Examples | What you must do |
|---|---|---|
| **Critical**: may invalidate a conclusion | treatment confounded with Member or DFM; unequal exclusion across treatments; failed light groups retained in light-dependent conclusions; rule-check failures (wrong `Program.txt`); stale results (re-run needed, or a Combined Analysis older than its Members); a Member missing from the pooled analysis; a DFM that stopped early in one treatment only | state it before the result it threatens; qualify or withhold the conclusion |
| **Major**: qualifies a conclusion | integrity warnings with long gaps; bleeding warnings; high pre-correction crosstalk; many censored breaking points; widely unequal Test windows; a result that changes with the gap or with the link gap; `Program.txt` missing or outside `data/`; generation-2 Members in a Progressive Ratio Project; approximate Program Facets | mention it beside the affected result |
| **Minor**: note only | *No Ack* note; repeated records dropped; generation-1 training-flag notes; a few lost records; warning-only light QC groups spread evenly | list in a QC note |

Map each finding to what it threatens:

| Finding | Threatens |
|---|---|
| lost records, data breaks, early stop, restart | totals and time courses on that DFM; Progressive Ratio Test windows |
| drift, dead wells | that chamber's licks and events; the light QC |
| crosstalk, bleeding | PI (toward zero), per-well licks |
| detection parameters poorly placed | everything event-based; Progressive Ratio breaking points |
| failed optogenetic light QC | light-table metrics; yoked comparisons; any claim about the contingency |
| self-triggered Progressive Ratio groups | breaking points and differences (already removed by default) |
| unequal exclusion | every between-treatment comparison |
| confounding | causal attribution to the treatment |

### 8.12 Things that look alarming but are not

- `Both = 0` everywhere in the simultaneous-feeding matrix: the correction is on;
  read `Both_pre`.
- Negative `MinInt` values: the baselined signal dips below zero at the edge of a
  bout.
- A Yoked chamber with zero Activations: its light comes from its partner.
- Generation-1 *training-flag notes* that three wells of a group stayed flagged to
  the end.
- *No Ack with nothing lost or repeated*; *repeated records dropped at load*.
- `NA` in firmware columns of a generation-1 or 2 recording.
- A `yoked_comparison.csv` that does not exist: there was no yoked group.
- `(no comparable treatment groups found)` in a one-treatment design.

---

## 9. Statistics

### 9.1 What pyflic computes, and where

| Analysis | Unit | Test | Where |
|---|---|---|---|
| treatment comparisons of feeding metrics, per Facet | chamber | Welch's t (2 treatments) or Tukey HSD (≥ 3); a mixed model beside it in a Project | member report; `_Stats.txt`; Project Report |
| paired − yoked difference against zero (Progressive Ratio) | chamber group | one-sample t-test on the differences (the paired t-test) and Wilcoxon signed-rank; mixed-model intercept in a Project | both reports; `_Stats.txt` |
| treatments compared on the paired − yoked difference | chamber group | Welch / Tukey; mixed model in a Project | both reports; `_Stats.txt` |
| breaking point between treatments | chamber group | Welch / Tukey (censored entered as observed), mixed model in a Project, and a pairwise **log-rank** test that treats censoring correctly | both reports; `_Stats.txt` |
| still-responding curve | chamber group | Kaplan–Meier estimate | `pr_still_responding.png`; reports |
| yoked comparison (non-Progressive-Ratio) | yoked group | zero tests and between-treatment tests; mixed models in a Project | both reports (PDF only) |
| light behaviour | chamber (Activation measures: chambers that earn light) | Welch / Tukey per Facet; mixed model in a Project | both reports; `_Stats.txt` |
| Role comparison (lab type) | chamber | Welch's t between the two Roles within each treatment | reports (PDF only) |
| Treatment control (lab type) | chamber | Dunnett's test against `control_level`, other factors held fixed | reports (PDF only) |
| ANOVA / mixed model with factors (optional) | chamber | Type II ANOVA, or a mixed model with a DFM random intercept | `compare_<metric>_<model>.csv` |
| bootstrap CIs (optional) | chamber | percentile interval of the mean | `bootstrap_<metric>.csv` |
| parameter sensitivity (optional) | chamber | means and SEMs at each parameter value; no test | `param_sensitivity_<parameter>.csv` |

### 9.2 The per-chamber treatment comparison, exactly

For each metric and each Facet (or the whole recording when there are no
Facets):

- Rows with a missing value for that metric are dropped; treatments with fewer
  than two observations are left out.
- **Two treatments**: Welch's unequal-variance t-test.
- **Three or more**: Tukey HSD on every pair, which controls the familywise error
  *within that one metric and Facet*.
- A pair whose p-value is not finite (no variance, typically a tail window with no
  feeding) is silently dropped from the table.
- The unit is the chamber (the row). Chambers on the same DFM and in the same
  Member are treated as independent.
- `significant` means p < 0.05, unadjusted beyond Tukey.

**Which values are tested** (this changes what a "Licks" result means):

| Layout | Metrics | How a two-well metric is formed |
|---|---|---|
| two-well | `PI`, `EventPI`, `Licks`, `Events`, `MedDuration` | `Licks` = `LicksA + LicksB` (a sum of **fourth roots** when transformed, not the root of the total); `Events` = `EventsA + EventsB`; `MedDuration` = **well A only** |
| single-well | `Licks`, `Events`, `MedDuration`, `MeanDuration` | as stored |

The reports label these "Licks (A + B)" and "Median duration, A (s)" with the
well names substituted.

### 9.3 The linear mixed model, exactly

In a Project with two or more Members, each pairwise comparison also gets
`p_mixed`:

- Model: `Value ~ is_b`, fitted on the two treatments' chambers only, with
  **Experiment** (Member) as the grouping factor and a **DFM variance component
  nested within Experiment**; REML.
- If that model fails to converge it falls back, without saying so, to a random
  intercept for Experiment alone.
- `p_mixed` is `n/a` when the subset spans only one Member or the fit fails.
- It is pairwise and unadjusted: no Tukey-style correction.
- For the paired − yoked tests against zero the model is `Value ~ 1` with the same
  random structure, and `p_mixed` is the intercept's p-value.

**How to weigh `p_pooled` against `p_mixed`.**

- `p_pooled` treats every chamber as independent. Chambers share boards and
  recording days, so it is optimistic when those vary.
- `p_mixed` allows for between-Member and between-DFM variation, which is the
  honest question when the claim is "this treatment does this", not "these flies
  on these boards did this". pyflic's guidance is to prefer `p_mixed` when they
  disagree.
- **With only two or three Members the mixed model is fragile.** It estimates a
  variance from two or three groups, and its p-value can swing in either
  direction (see 9.15). When the two disagree, do not stop at "between-member
  variation": look at the data. Compute the treatment difference (or the mean
  difference against zero) **separately in each Member**, and the direction per
  DFM. If every Member and nearly every DFM agree in direction and size, the
  effect is consistent, and the disagreement is a property of the model;
  report both p-values and the consistency. If the Members disagree, the pooled
  result rests on one Member and must be reported as such.
- When treatment is confounded with Member (section 6.6), neither p-value
  separates treatment from recording.

### 9.4 Reading `<name>_Stats.txt`

Layout, top to bottom:

1. Header: `Members pooled : N (names)` (section 4.5 on the count),
   `Chambers pooled` (rows of `_Summary.csv`), experiment type, layout, and for
   Program Facets a note that Facets are pooled by Paradigm label.
2. The definitions of `p_pooled` and `p_mixed`.
3. **Flagged Chamber Groups**: every light QC group (Progressive Ratio and
   optogenetic) with flags, its status (excluded, retained, warning only), its
   flags and onset, and a count of failed groups retained in the numbers; then any
   generation-2 Member.
4. Progressive Ratio only:
   - *Paired − yoked difference (primary)*. `Chamber groups pooled` counts **rows
     across both Facets** (Training and Test), so it is about twice the number of
     groups; use the `n` columns. Then the between-treatment table and the
     **against zero** table: `Metric, Facet, Treatment, n, mean, sem, p_t,
     p_wilcoxon, p_mixed`, starred when `p_t < 0.05`.
   - *Breaking point*: groups pooled and how many are censored, median break
     requirement on generation 3, then the table with `p_logrank`.
   - *Per-chamber metrics (secondary)*: these pool paired and yoked flies
     together and are not the type's result.
   - A design of solo groups reads *Paired − yoked difference: none; the design
     has no yoked controls*.
5. The per-chamber table: `Metric, Facet, A, B, nA, nB, diff, p_pooled, p_mixed`,
   with `*` when **`p_pooled`** < 0.05 (the star never reflects `p_mixed`).
6. *Light behaviour* for optogenetic Members (paired chambers only under
   Progressive Ratio).

**Sign convention: `diff` is the mean of B minus the mean of A**, where A and B
are the treatment names in the row's own `A` and `B` columns. With two treatments
A is the one that appears first in the data; under Tukey the pairs are in
alphabetical order. Always read the labels on the row; never assume which is the
control.

`_Stats.txt` holds no treatment means, only their difference. Take means and
n from the CSVs (or the reports) when you need them, and label them as derived.

`_Stats.txt` does **not** include the yoked comparison of non-Progressive-Ratio
designs or the control comparisons; those are only in the Project Report PDF.

### 9.5 Reading the PDF reports' statistics tables

- **Comparison** reads `X vs Y`; **n** and **Mean** read `X / Y`;
  **Difference** is `X − Y` (the first named minus the second). This is the same
  arithmetic as `_Stats.txt`'s `diff`, written the other way round: X is
  `_Stats.txt`'s B.
- Highlighted p-values are < 0.05. `p (mixed)` appears in the Project Report;
  `p (log-rank)` beside the breaking point.
- The against-zero table reads `Mean ± SEM`, `p (paired t)`, `p (Wilcoxon)`.
- Role comparisons: `n` and `Mean` read as the first Role / the second;
  Difference is the first minus the second.
- A grey note "No treatment comparison was possible here" means fewer than two
  treatments with two observations each.
- The *At a glance* callouts on each cover summarise integrity, exclusions,
  missing `Program.txt`, and the light QC verdicts; they are a fast QC index.

### 9.6 Facets in the statistics

- When an experiment has Facets, the pooled tests run **per Facet only**; the whole
  recording is not tested in `_Stats.txt`. Program Facets and Progressive Ratio
  Facets are split by label, cutoff Facets by window.
- A difference in one Facet and not another is a time-dependent effect, or noise
  spread over several tests. Say how many Facets were tested.
- The Progressive Ratio reports use the **Test** Facet for the paired − yoked
  difference; the Training rows exist but are not the result.

### 9.7 Progressive Ratio statistics

- **Unit**: one observation per chamber group. Paired and yoked flies are never
  separate observations in the primary tests.
- **Is the paired fly different from its yoked partner?** The Test-phase
  differences against zero, per treatment. A positive `dLicksA` supports the light
  acting as a reinforcer: the paired fly fed more at the sucrose well when
  feeding produced the light than a fly given the same light regardless. This
  needs both flies treated identically in every other way. It can be asked even
  with a single treatment.
- **Do treatments differ in it?** The between-treatment test on the differences:
  a treatment changed how reinforcing the light was, not merely how much flies
  fed.
- **Breaking point**: how far the paired fly kept working as the requirement rose.
  The yoked fly has none. Welch, Tukey and the mixed model enter a censored count
  as if observed, which **biases that treatment's mean downward**; the log-rank
  test and the still-responding curve treat it as the lower bound it is. When
  censoring is common or uneven, base the conclusion on the log-rank test and the
  curve. Log-rank tests are **not** corrected across treatment pairs.
- **Fixed ratio**: the same rule counts activations earned before the first long
  pause; the reports title it *Fixed ratio: activations earned*; no trend is
  expected or judged.
- **Sucrose persistence**: `dPersistCensored` rows are kept, not dropped; many
  censored rows make `dPersistA` a difference of lower bounds.
- **Generation 2 Members** contribute no Test, breaking point or persistence.
- **Within one Member**, groups on a DFM are treated as independent; only a Project
  of two or more Members gets the mixed model.

### 9.8 Yoked comparisons (non-Progressive-Ratio optogenetic designs)

`d<m>` = the paired chamber's value minus the mean of its yoked chambers', per
yoked linkage group and window (`Whole recording`, then each Facet). The reports
test each treatment's differences against zero (paired t, Wilcoxon) and compare
treatments; a Project adds mixed models. The figure plots paired against yoked
over the identity line: points above it are groups where the paired fly's value
is larger.

### 9.9 Control comparisons

- **Treatment control**: each level of the control factor against
  `control_level`, within each combination of the other factors (a *stratum*),
  by Dunnett's test on chambers. Difference = level − control. Levels or controls
  with fewer than two chambers are untested. No mixed model, even in a Project.
- **Role comparison**: within each treatment, the chambers of one Role against the
  other's, Welch's t; untested with fewer than two chambers on a side.

### 9.10 Light behaviour comparisons

Per Paradigm Family, the headline metrics (section 7.8) are compared between
treatments per chamber, per Facet. Activation measures use only chambers that
earn light (Paired, Self, Mutual, or a blank Role); Yoked chambers keep their lit
time. A metric identical in every chamber (open-loop lit time, for instance) is
named as not compared. Under Progressive Ratio only paired chambers enter.

### 9.11 Optional analyses

- **`compare_<metric>_<model>.csv`**: the only output that fits **factors as
  separate terms with interactions** (`Y ~ C(f1) * C(f2)`), as a Type II ANOVA, or
  as a mixed model with a DFM random intercept. Use it for main effects and
  interactions. The pooled per-chamber tables, by contrast, compare every
  combination of levels (each `Treatment` label) pairwise, so they never test a
  main effect or an interaction directly. Tukey post hoc is written only for a
  single factor.
- **`bootstrap_<metric>.csv`**: chamber-level resampling, percentile intervals, a
  fixed seed (repeatable). Non-overlap of two intervals is a rough guide, not a
  test.
- **`param_sensitivity_<parameter>.csv`**: each treatment's mean at several values
  of one detection parameter. An effect that appears at only one link gap or
  one threshold is not robust. If present, use it.
- **`light_phase_summary.csv`**: divide counts by `PhaseSeconds` before comparing
  light and dark; phases are rarely equal in length.
- **`bout_transition_matrix.csv`**: how often a fly switched between wells
  between consecutive bouts.

### 9.12 Interpretation rules and pitfalls

1. **Pseudo-replication.** Chambers on one DFM share a board, a position and
   often a treatment; chambers in one Member share a day and a batch of flies. The
   per-chamber n overstates independent replication. State the number of DFMs and
   Members behind each treatment.
2. **Multiplicity.** pyflic tests several metrics, in every Facet, for every
   pair of treatments, and (Progressive Ratio) several difference metrics and
   log-rank pairs, with no correction beyond Tukey within one metric and Facet.
   Expect some p < 0.05 by chance when many tests are run. Lead with the
   design's primary measure (section 7.11); treat isolated significant results
   among many tests as tentative; note when several related measures agree.
3. **Transformed scales.** Lick differences are on the fourth-root scale. A
   difference of 1 in transformed licks is large at low counts and smaller in
   relative terms at high counts. Give back-transformed magnitudes per chamber
   when you describe size, and say how you computed them.
4. **What a two-well "Licks" result means.** A sum of per-well fourth roots, so it
   is not proportional to total licks; and "MedDuration" is well A only.
5. **Missing values shrink n.** Duration metrics and PI drop chambers that never
   fed; compare `nA`/`nB` across metrics.
6. **Small samples.** With n of 3–5 per treatment, Welch's test has little power
   and is sensitive to one chamber. A non-significant result is weak evidence of
   no effect. Look at the spread (per-chamber dot plots, CSV rows).
7. **Absence of evidence.** "Not significant" means the data do not show a
   difference at this sample size, not that there is none. Say so when n is small
   or the variance large.
8. **Effect size.** Report the difference in natural units: PI points (on −1 to
   +1), seconds of median bout duration, back-transformed licks, minutes of
   persistence, light events of breaking point. Relate it to the spread (the
   SEM in the zero tests; the per-chamber SD from the CSV, labelled as derived).
9. **Consistency.** A robust effect usually has the same direction in every
   Member, on most DFMs, on both sides (`pi_direction`) in a counterbalanced
   design, and across adjacent Facets. Check what you can; report
   inconsistencies.
10. **Detection dependence.** Event counts, durations, intervals and the event PI
    depend on `feeding_event_link_gap`; lick counts and the lick PI do not.
    Thresholds move everything. Compare results only between analyses with the
    same parameters.
11. **Time of day.** pyflic does not know it. A long night-time pause ends a
    breaking point like any other; circadian structure shows in time courses.
    Consider the recording start time (integrity reports) when interpreting
    time courses.
12. **Exclusion bias.** Section 8.7: a comparison of the survivors may not be a
    comparison of the treatments.
13. **Data-derived windows.** Progressive Ratio Facets are split at each group's
    own training end, so Training and Test lengths differ between groups;
    Test-phase totals partly reflect how long each group's Test lasted.
14. **Two indices of one thing.** `PI` and `EventPI`, `dLicksA` and `dEventsA`, the
    breaking point and the break requirement: when they agree, say so; when they
    disagree, report both and the reason (meal pattern, detection, units).
15. **No over-reach.** A preference index says which food was chosen, not why. A
    paired − yoked difference says the contingency mattered, not which neurons
    did what.

### 9.13 Hedonic specifics

The weighted duration summary weights each chamber's median bout duration by its
event count at that well, so a chamber whose median rests on 300 events counts
more than one resting on 3. Compare `WeightedMeanDuration` with `MeanDuration`:
a large gap means a few low-event chambers pull the plain mean. `SampleSize` is
the chambers with events at that well. `treatment_means.csv` is descriptive and
holds no p-values; the treatment tests of median duration at each well are the
per-chamber Welch or Tukey tests in the reports, like any other metric.

### 9.14 Bootstrap and other descriptive intervals

Treat a bootstrap interval as a description of uncertainty in a treatment mean
over chambers, with the same pseudo-replication caveat as the tests.

### 9.15 Worked example: when `p_pooled` and `p_mixed` disagree

From the repository's test Project `test_data/NPF` (Progressive Ratio, one
treatment `Chrim`, two Members, 47 paired chamber groups), the against-zero table
in `pyflic - prog assay_Stats.txt` reads, for the Test Facet:

```
Metric        Facet     Treatment        n       mean       sem       p_t  p_wilcoxon   p_mixed
dLicksA       Test      Chrim           47      1.709    0.2131  2.75e-10   3.504e-10 6.265e-15 *
dEventsA      Test      Chrim           47        145     22.18 4.551e-08   5.269e-07    0.2854 *
dPI           Test      Chrim           47     0.1686   0.03909 8.438e-05   2.426e-06    0.4472 *
```

For `dEventsA` the paired t-test gives p = 4.6 × 10⁻⁸ and the mixed model p =
0.29. Computing the mean per Member from `_PairedYokedDiff.csv` (Test rows) gives
149.6 (n = 23) and 140.5 (n = 24): the Members agree closely. Per-DFM means are
positive on 15 of 16 DFMs. The disagreement therefore does not come from
inconsistent data; with only two Members the mixed model's estimate is unstable
(for `dLicksA` it gives a *smaller* p than the t-test). A faithful summary says:
paired flies made about 145 more sucrose feeding events than their yoked partners
during Test (mean ± SEM 145 ± 22, n = 47 groups; paired t p = 4.6 × 10⁻⁸;
Wilcoxon p = 5.3 × 10⁻⁷); the effect had the same direction and similar size in
both Members and on 15 of 16 DFMs; the mixed model did not reach significance
(p = 0.29), which with two Members reflects limited ability to estimate
between-member variation rather than disagreement between them. It also says
there was one treatment, so no between-treatment comparison was possible
(`_Stats.txt` reads `(no comparable treatment groups found)`), and that `dLicksA`
is on the fourth-root scale.

---

## 10. Playbooks

### 10.1 Single-well consumption (Custom, `chamber_layout: single_well`)

- Treatments and n per treatment, DFM and Member.
- QC: integrity, breaks, dead wells, exclusions by treatment. No crosstalk checks
  apply.
- Results: `Licks` (transformed), `Events`, `MedDuration`, `MeanDuration` by
  treatment and Facet; time courses from the binned summary.
- Statistics: per-chamber Welch or Tukey; mixed model in a Project.
- Caveats: lick transform; empty chambers; recording length per DFM.

### 10.2 Two-well choice (Custom two-well, or a lab type)

- Identify wells A and B from `well_names` and the side from `pi_direction` per
  DFM; check counterbalancing across treatments.
- QC: add simultaneous feeding and bleeding; state the dual-feeding correction.
- Results: lead with `PI`; report `EventPI` beside it; then per-well licks and
  events, `MedDurationA/B`; preference over time.
- Interpretation: positive PI is a preference for well A. A PI near zero can be
  indifference, crosstalk, or two populations of opposite preference; check the
  per-chamber spread.

### 10.3 Hedonic

- Two-well, no light (a lab hedonic type may have light).
- Lead with median bout duration at each well, plain and event-weighted
  (`treatment_means.csv`, the hedonic plot); then PI and licks.
- The idea is that palatability shows in how long bouts last, even when totals
  are similar. A treatment that lengthens bouts at one food relative to the other
  supports a hedonic difference; equal durations with different PI suggest a
  difference in choice, not palatability.
- Default cutoffs apply: 20 raw licks per well, median duration 13 s, 150,000
  events.

### 10.4 Progressive Ratio

- Read the design: pairs (`paired_chambers`), solo groups (`chamber_groups:
  single`), or from `Program.txt`; per-DFM sucrose side (`pi_direction`; well A is
  always sucrose, well B yeast).
- QC: training table; both light QCs; resting level; lick-free runs; censoring;
  Test windows; restarts; generation (2 means no Test).
- Results, paired design: Test-phase `dLicksA`, `dPI`, `dPersistA` against zero and
  between treatments; cumulative difference curve; breaking point
  (still-responding curve and log-rank first, mean comparisons second); break
  requirement on generation 3.
- Results, solo groups: no paired − yoked difference exists ("the design has no
  yoked controls"); the breaking point is the result; it measures effort, not
  whether the contingency mattered.
- Report: groups per treatment before and after exclusion with reasons;
  `pr_break_gap_min` and `pr_test_window_min`; censored groups per treatment;
  whether conclusions survive the 60/120/240 minute sensitivity table; any restart
  that ended a Test window.

### 10.5 Optogenetic designs under Custom or a lab type

- Identify the Paradigms that ran (`opto_program.csv`, `opto_light_intervals.csv`,
  the report's program table, Program Facet labels) and the Roles of the chambers.
- Facets: under `facet_source: program`, every metric is split by Paradigm;
  compare treatments within each Program Facet, and compare Program Facets
  within a treatment only descriptively unless a test addresses it.
- QC: the optogenetic light QC and, on generation 3 with `Program.txt`, the rule
  checks. Without `Program.txt` the QC is limited and cannot fail unexplained
  light.
- Results: the layout's feeding metrics per Facet; the light table's headline
  metrics; *Feeding in light and dark* (open loop): lines falling from light to
  dark are flies that fed more in the light; `LightFeedingIndex` > 0 means more
  licks per minute lit than dark.
- Yoked groups: `yoked_comparison.csv`, the paired − yoked logic of 9.8.
- Closed loop: `Activations` and `LitSec` describe how much light flies earned;
  more light earned with more feeding may simply be the contingency, not a
  preference for the light.
- NonFeeding: `Suppressions` count light periods the fly ended by feeding.

### 10.6 Lab types with control comparisons

Read `experiment_type.yaml`: its modules, the light it expects, its default
constants and its `control:` block. Lead with its declared comparison: Dunnett
against `control_level`, or the Role comparison. Report the declared comparison
first, then the general results.

---

## 11. Workflow

1. **Map the tree** (section 3): Batch, Projects, Members, standalone experiments,
   blocked members, files to ignore, context files.
2. **Check provenance and freshness** (4.5): versions, staleness, missing Members,
   leftover files, `Program.txt` location.
3. **Read the design** (5): type, layout, factors and levels, well names,
   transform, Facets, optogenetics, constants, exclusion group, control level,
   omitted outputs.
4. **Tabulate the design** (6.5) and **check confounds** (6.6).
5. **Build the QC ledger** (8.2) per Member, then per treatment; grade each finding
   (8.11).
6. **Read the results** in this order: the Project's `_Stats.txt` (or the member's
   report), then the CSVs for means, n and spread, then the figures if available.
7. **Check consistency** across Members, DFMs, sides and Facets (9.12, item 9).
8. **Weigh** each candidate conclusion against the QC ledger and the statistical
   cautions.
9. **Write** (section 12).

Final checklist before you answer:

- [ ] Every number traces to a file, or is labelled as derived.
- [ ] n per treatment stated (chambers or chamber groups; DFMs; Members).
- [ ] Exclusions per treatment stated, with reasons.
- [ ] Every critical QC finding stated before the result it affects.
- [ ] Non-significant comparisons stated as such.
- [ ] `p_pooled` and `p_mixed` both reported where they exist, with their
      disagreement explained from the data.
- [ ] Censoring reported for breaking points.
- [ ] Transform and well identities stated wherever licks or PI are quoted.
- [ ] Inferences from labels marked as inferences.
- [ ] Checks that did not run are named.

---

## 12. Writing the summary

### 12.1 A structure that works

1. **What was analysed**: Project(s), Members, experiment type, layout, factors
   and treatments, foods (well names), recording lengths, generations, whether
   optogenetic, and the n behind each treatment.
2. **Data quality**: the QC verdict in a few sentences, critical and major findings
   first, exclusions per treatment, checks that did not run.
3. **Results**: the primary measure first, with magnitude, n and p-values (pooled
   and mixed), then secondary measures; within-treatment tests (paired − yoked)
   before between-treatment tests for Progressive Ratio and yoked designs.
4. **Robustness**: consistency across Members and DFMs, sensitivity tables,
   censoring, Facet dependence.
5. **Interpretation and limits**: what the results support, what they do not, and
   what would strengthen them.

### 12.2 Confidence language

| Situation | Wording |
|---|---|
| significant on the primary measure, consistent across Members, clean QC, `p_mixed` agrees | "shows", "clear evidence" |
| significant pooled, mixed model fragile or disagreeing, data consistent | "evidence for … , consistent across members; the mixed model did not confirm it" |
| significant in one Member only, or among many uncorrected tests | "suggests", "tentative" |
| not significant | "no significant difference was detected (p = …)" with n |
| QC threatens the measure | "cannot be interpreted until …", or report with the caveat first |

### 12.3 Never

- invent a number, a p-value or a test pyflic did not run;
- call a non-significant difference a trend;
- treat Roles, Facets, Members or DFMs as treatments;
- quote a transformed lick value as a count;
- describe a breaking point mean without its censoring;
- hide a QC failure, a retained failed group or unequal exclusions;
- pool across Projects of a Batch;
- treat an AI narrative (`ai_summary.txt`, `batch_ai_narrative.md`) as a primary
  source.

---

## 13. Quick reference

### 13.1 Glossary

| Term | Meaning |
|---|---|
| Batch | a folder with Projects anywhere below it; runs Projects unattended; never pools them |
| Project | a folder with `project.yaml`; its Members are its immediate subfolders with a `flic_config.yaml`; pools them |
| Member | an Experiment Directory in a Project; a different experiment addressing the Project's question, not a repeat |
| Experiment Directory | one recording: `flic_config.yaml`, `data/`, `qc/`, `analysis/` |
| Design | `project.yaml`'s `design:`; the authority over every Member's `global:` settings |
| Experiment Type | the assay definition (Custom, Hedonic, ProgressiveRatio, lab types) |
| Analysis Module | an analysis a type includes: Progressive Ratio, Hedonic, Paired and yoked, Light and dark |
| Chamber Layout | `single_well` (12 chambers per DFM) or `two_well` (6) |
| DFM | one 12-well board, identified by id within a recording |
| Chamber | the unit a fly occupies and the unit of every metric |
| Treatment | what a chamber was assigned; factor levels joined by `_` when factors exist |
| Experimental Design Factor | a declared independent variable with levels; positional in assignments |
| Facet | a named time window; a column, not a folder |
| Program Facet | a Facet from the Opto Program: one per Program Interval (or per Interval and Cycle) |
| Excluded Chamber | removed from every result: by hand (`remove_chambers.csv`) or automatically (cutoffs) |
| Combined Analysis | the Project's stacked, filtered Member tables plus pooled statistics |
| lick | one sample above threshold (5 per second of contact at 5 Hz) |
| event | a bout: a run of licks, with short gaps linked |
| PI, EventPI | (A − B)/(A + B) of licks, of events; positive favours well A |
| Opto Program | the MCU's exported `Program.txt`, read only from `data/` |
| Linkage Group | wells lit together; the unit of the optogenetic light QC |
| Trigger Well | a well whose own signal can switch its group's light |
| Role | Paired, Yoked, Mutual, Self, Open, Unlit: what a chamber was to the light |
| Paradigm, Paradigm type, Paradigm Family | the light rule of an interval; its type; one of four families |
| Program Interval, Cycle | a timed step of the program; a pass through a repeating program |
| Generation | 1, 2 or 3: the shape of a DFM's data file and what it records |
| Activation | one LED on-period a well's own Paradigm started |
| Acclimation, Phase | the firmware's opening closed-loop run (`A`), then Main (`M`) |
| Requirement, Progress | under a ratio Paradigm, the bouts needed for the next Activation, and the count so far |
| Firmware Event | a feeding bout as the firmware counts it |
| Hold | how long closed-loop light stays on after contact ends |
| Explained light | lit time with a trigger-well lick or touch behind it |
| Chamber Group | Progressive Ratio: a pair (paired + yoked) or a solo group, sharing light and treatment |
| Training, Test | Progressive Ratio phases, split at each group's training end |
| Light Event | Progressive Ratio: one onset of a group's light |
| Lick-free Light Event | a light event with no sucrose lick since the previous one and none within its Hold |
| Breaking Point | lick-backed Test light events before the first pause longer than `pr_break_gap_min` |
| Sucrose Persistence | minutes to the last sucrose feeding event before such a pause, for every fly |
| Censored | still responding when the Test window ended: a lower bound (`n+`) |
| Test window | training end to the end of the recording, the ratio run, a restart, or `pr_test_window_min` |
| Paired-Yoked Difference | paired minus yoked within a group and Facet |
| Resting Level | a well's per-minute median raw signal; its rise is sensor drift |

### 13.2 Defaults

Detection parameters (two-well; single-well is the same except
`correct_for_dual_feeding: false`):

| Parameter | Default |
|---|---|
| `baseline_window_minutes` | 3 |
| `feeding_threshold` | 20 |
| `feeding_minimum` | 10 |
| `tasting_minimum` | 5 |
| `tasting_maximum` | 20 |
| `feeding_minevents` | 1 |
| `tasting_minevents` | 1 |
| `feeding_event_link_gap` | 5 samples (1 s) |
| `samples_per_second` | 5 |
| `correct_for_dual_feeding` | true |
| `pi_direction` | left |
| `transform_licks` | true |

Exclusion cutoffs:

| Constant | Custom | Hedonic, ProgressiveRatio |
|---|---|---|
| `min_untransformed_licks_cutoff` | off | 20 |
| `max_med_duration_cutoff` | off | 13.0 |
| `max_events_cutoff` | off | 150000 |
| `exclude_failed_opto_chambers` | false | false |
| `require_training_complete` | — | true (Progressive Ratio) |
| `exclude_failed_pr_groups` | — | true (Progressive Ratio) |

A lab type may set others; the Design's values win; `summary.txt` prints the
values used.

Optogenetic light QC: `opto_unexplained_warn_fraction` 0.10,
`opto_unexplained_fail_fraction` 0.30, `opto_unexplained_min_sec` 30,
`opto_default_decay_ms` 1000, `opto_decay_tolerance_samples` 2,
`opto_unlit_feeding_fraction` 0.50, `opto_open_loop_min_lit_fraction` 0.95;
generation 3: `opto_event_disagreement_warn_fraction` 0.25,
`opto_yoking_fail_fraction` 0.01, `opto_lost_records_warn` 25,
`opto_unknown_rows_warn_fraction` 0.01, `opto_late_records_warn` 10,
`opto_activation_tolerance_ms` 25, `opto_activation_length_fail_fraction` 0.05,
`opto_lockout_fail_fraction` 0.02, `opto_missed_contact_warn_fraction` 0.25,
`opto_non_feeding_lit_warn_fraction` 0.10, `opto_non_feeding_min_sec` 30,
`opto_non_feeding_dark_warn_fraction` 0.10, `opto_interval_lag_warn_sec` 5.

Progressive Ratio: `pr_lick_free_run` 5, `pr_trend_min_events` 5,
`pr_trend_min_rho` 0.3, `pr_resting_level_rise` 15, `pr_resting_level_ratio` 3,
`pr_break_gap_min` 120 minutes, `pr_test_window_min` off.

### 13.3 Question → file

| Question | File |
|---|---|
| What was the design? | `project.yaml`, `flic_config.yaml`, `experiment_type.yaml`, `summary.txt` |
| Which treatments, how many chambers? | `flic_config.yaml` (assigned); `feeding_summary.csv` or `<name>_Summary.csv` (analysed) |
| What was excluded and why? | `remove_chambers.csv`, `analysis/removed_chambers.csv`, `<name>_Excluded.csv`, `summary.txt` |
| Did the recording run cleanly? | `qc/integrity/`, `qc/data_breaks/`, `summary.txt` *DFM details* and *QC summary* |
| Crosstalk? | `qc/simultaneous_feeding/`, `qc/bleeding/` |
| Was the light right? | `qc/opto/opto_light_qc.csv`, `opto_light_intervals.csv`, `opto_light_events.csv`; `<name>_LightQC.csv` |
| What did the light do? | `light_summary.csv`, `light_wells.csv`; `<name>_LightSummary.csv` |
| Progressive Ratio training and light QC | `pr_light_qc.csv`, `pr_light_events.csv`, `summary.txt` |
| Breaking points | `pr_breaking_point.csv`, `<name>_BreakingPoint.csv`, `summary.txt` (gap sensitivity) |
| Paired − yoked | `paired_yoked_diff.csv`, `<name>_PairedYokedDiff.csv`, `yoked_comparison.csv` |
| Pooled p-values | `<name>_Stats.txt`; `<name>_report.pdf` |
| One member's p-values | `analysis/experiment_report.pdf` only |
| Time course | `binned_feeding_summary.csv`; `pr_cumulative_diff.csv` |
| Bout-level detail | `tidy_feeding_events.csv`, `tidy_tasting_events.csv` |
| Raw lick counts | `light_wells.csv` `PyflicLicks`; simultaneous-feeding `Licks1`/`Licks2`; or back-transform |
| Robustness to parameters | `param_sensitivity_<parameter>.csv`; breaking point at other gaps in `summary.txt` |
| What did the latest run include or skip? | *Analysis Set* block at the end of `summary.txt` |

### 13.4 Red flags

- `summary.txt` says "no Program.txt in data/" while a `Program.txt` sits elsewhere
  in the member.
- A Member's `remove_chambers.csv` is newer than its `feeding_summary.csv`.
- A Member's results are newer than the Project's `_Stats.txt`.
- `Members pooled : N` smaller than the list of names beside it.
- One treatment loses a much larger share of chambers than another.
- One treatment lives in one Member, or on one DFM, or on one side.
- `retained` failed light QC groups in `_Stats.txt`.
- Any rule-check failure (*wrong activation length*, *lockout broken*,
  *requirement series wrong*, *short activations*, *max on exceeded*).
- *Paradigm not in program*, *broken yoking*, *light while off*.
- A DFM whose elapsed minutes fall well short of its siblings'.
- Most breaking points censored, or censoring concentrated in one treatment.
- A Progressive Ratio Member of generation 2 (no Test) pooled with others.
- `p_pooled` tiny and `p_mixed` large, or the reverse, without a per-Member check.
- Licks quoted without saying whether they are transformed.
- Outputs only in `analysis_<start>_<end>/` or `<stem>_results/` folders.
