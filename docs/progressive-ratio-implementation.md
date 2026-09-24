# Progressive Ratio — what is currently implemented

> **Status (2026-09-22):** the "Decided direction" at the end of this file has
> been implemented — type object, loader validation, per-group training end,
> augmented summaries, the Paired-Yoked Difference table, both figures, the
> pooled statistics and `timecourse_pr_diff` spec, script actions, linter,
> Config/Design editor support and help topics. Tests:
> `tests/test_progressive_ratio.py` (fixtures in `tests/pr_fixtures.py`).
> Two pre-existing bugs found on the way are fixed and noted in the CHANGELOG:
> the empty tail Facet (`(start, 0)` read literally) and the never-written
> `removed_chambers.csv`. The survey below describes the code **before** this
> work.
>
> **Breaking point (2026-09-24):** defined by a second grilling session and
> implemented — the first-gap rule with censoring, Sucrose Persistence, the
> log-rank test and the tests against zero (ADR-0014; *Breaking point* at the
> end of this file). Tests: `tests/test_pr_breaking_point.py`.
>
> **As implemented (2026-09-24):** the last part of this file, *As
> implemented*, describes the code as it now stands: the module map, the
> pipeline order, the QC protocols, the exclusion criteria, the breaking point
> and Sucrose Persistence, and the figures and statistics built on them.

A survey of the Progressive Ratio experiment type as it stood at commit
`8000615`, kept as the "before" to the decisions at the end.

## The type object

`ProgressiveRatioExperimentType` in
[`pyflic/base/experiment_types/progressive_ratio.py`](../pyflic/base/experiment_types/progressive_ratio.py)
is thin — 46 lines. It declares:

| Field | Value | Effect |
|---|---|---|
| `name` / `display_name` | `ProgressiveRatio` / "Progressive Ratio" | yaml key + menu label |
| `chamber_layout` | `"two_well"` (owned) | derives `params.chamber_size: 2`; config must **not** state either key |
| `required_wells` | `("A", "B")` | lint error if `well_names` omits either |
| `experiment_class` | `pyflic.base.progressive_ratio_experiment:ProgressiveRatioExperiment` | overrides the layout→class map in `yaml_config.py:37-50` |
| `facet_cutoffs` | `(30.0,)`, `facets_fixed = False` | default only — the user may change it |
| `phase_labels` | `("Training", "Test")` | applied **only** while cutoffs still equal `(30.0,)` (`experiment_types/base.py:113-135`) |
| `default_constants` | `min_untransformed_licks_cutoff: 20`, `max_med_duration_cutoff: 13.0`, `max_events_cutoff: 150000.0` | identical to Hedonic — no PR-specific QC |
| `report_set` | `faceted_licks`, `faceted_events`, `faceted_medduration`, `timecourse_licks`, `timecourse_events` | all generic specs from `pubfigures.py`; **no breaking-point plot** |
| `output_manifest` | `feeding_summary.csv`, `feeding_summary_facet.csv`, `summary.txt` | identical to the base class; no BP outputs |

## The experiment class

[`pyflic/base/progressive_ratio_experiment.py`](../pyflic/base/progressive_ratio_experiment.py)
derives `TwoWellExperiment` (which derives `Experiment`); it is a
`@dataclass(slots=True)`. It adds exactly five members, all a direct port of
`breaking_point.R`:

- **`load()`** (`:52-84`) wraps `load_experiment_yaml` and hard-fails if any DFM
  has `chamber_size != 2`. This is a redundant second gate — the type already
  owns the layout, so a typed config cannot reach here with the wrong size.
- **`breaking_point_well(dfm, well, *, end_training=None)`** (`:87-170`), port of
  R `breaking.test()`. Takes `lights_df["Minutes", "Wn"]` plus the cumsum of
  `lick_df["Wn"]`, filters to `Minutes > end_training`, keeps rows where
  `Lights` *transitions*, diffs `Minutes` and `CumLicks`, then keeps only
  lights-**on** rows. Returns `Minutes, Lights, CumLicks, DeltaMinutes,
  DeltaLicks`. The first row always has `DeltaMinutes = DeltaLicks = 0.0` — the
  `np.concatenate([[0.0], np.diff(...)])` seed.
- **`breaking_point_dfm(dfm, configuration)`** (`:172-224`), all 12 wells →
  `dict[str, DataFrame]`. `configuration` ∈ 1–4 selects which well carried the
  training signal, per `_BREAKING_POINT_CONFIGS` (`:17-22`):

  | configuration | training wells |
  |---|---|
  | 1 | W1 (wells 1–4), W5 (5–8), W9 (9–12) |
  | 2 | W3, W7, W11 |
  | 3 | W2, W6, W10 |
  | 4 | W4, W8, W12 |

  Wells 1–4 inherit the first reference well's end-of-training, 5–8 the second,
  9–12 the third.
- **`breaking_point_summary(configuration)`** (`:226-239`), the same across all
  DFMs → `dict[dfm_id, dict[well, DataFrame]]`.
- **`plot_breaking_point_dfm_gg(dfm, configuration, ...)`** (`:241-335`),
  a plotnine `ΔLicks vs Minutes` plot, `facet_wrap(~Well, ncol=4)`,
  `coord_cartesian(ylim=(0, 500))` by default.

**No breakpoint scalar is ever computed.** The help text says "the breakpoint is
where `DeltaLicks` falls away"
(`pyflic/help/content/concepts-experiment-types.md:114-123`), but nothing reduces
the per-period table to a per-well number. That reduction is left to the
reader's eye on the facet plot.

## Data prerequisites

Two pieces of DFM preprocessing, both in [`pyflic/base/dfm.py`](../pyflic/base/dfm.py),
and neither gated on experiment type:

1. **`_calculate_progressive_ratio_training()`** (`dfm.py:204-236`), a port of R
   `CalculateProgressiveRatioTraining()` + `GetDoneTrainingInfo()`. Runs **only
   when `version == 3`** (`:135-136`, `:153-154`). It reads the in-training flag
   steganographically: a well sample `> 40000` means "in training", and the raw
   value is corrected by `- 65536`. It then populates `dfm.in_training_data` —
   one row per well with `well, Minutes, Sample` = the **max** (last)
   in-training sample, `NaN` if that well never trained.
2. **`_calculate_lights()`** (`dfm.py:254-281`). Bit-unpacks `OptoCol1` into 12
   boolean well columns. Without an `OptoCol1` column it emits an all-zero
   frame, so breaking-point analysis on a non-opto recording silently returns
   empty tables rather than erroring.

In practice: **version-3 DFM files with `OptoCol1` present**. A v2 file leaves
`in_training_data = None`, so `end_training` falls back to `0.0` and no training
filter is applied at all.

## Design file

Nothing PR-specific. `ExperimentDesign`
([`experiment_design.py:20`](../pyflic/base/experiment_design.py)) carries a
free-text `experiment_type: str | None` field — its comment still lists the
retired `two_well`/`single_well` spellings — and that is the whole of it.

Crucially, **`configuration` (1–4) has no home in the config or the design**. It
is passed per call, and in the GUI it is a per-step script parameter defaulting
to `1`. There is no way to record "this plate ran config 3" alongside the
chamber assignments, and no validation that the chosen config matches the wells
that actually show training data.

## GUI and scripting surface

One action, `plot_breaking_point`
([`script_editor/actions.py:441-451`](../pyflic/base/script_editor/actions.py)):
category PLOTS, `requires="progressive_ratio"`, params `config` (int, default 1)
plus start/end. The runner
([`script_editor/runner.py:239-249`](../pyflic/base/script_editor/runner.py))
loops over DFMs and writes `analysis/breaking_point_dfm{id}_config{n}.png`. It
skips with a log line if the experiment is not a `ProgressiveRatioExperiment`
(`:109-113`).

The action ignores its own `start`/`end` params — `rm` is computed but never
passed to `plot_breaking_point_dfm_gg`, which has no range argument.

Nothing else. There is no tabular BP export action (`breaking_point_summary` is
Python-API-only), no project-level action
(`script_editor/project_actions.py` has none), and no BP entry in
`pubfigures.py` — so BP never reaches the project report or the AI narrative
payload, both of which iterate `report_set` (`project_report.py:135`,
`ai/payload.py:75`).

## Tests

Effectively none. Four assertions across `tests/test_experiment_types.py:20,76,86`
and `tests/test_config_editor_ui.py:90` check that the type registers, owns
`chamber_layout`, and appears in the pick-list. **There is zero coverage of the
breaking-point maths**, including in `tests/test_parity_r.py`, which covers other
R ports.

## Gaps

1. No breakpoint scalar or metric — only the per-period delta table.
2. `configuration` is unrecorded and unvalidated; it belongs in the design or config.
3. BP outputs are absent from `report_set` and `output_manifest`, so they are
   invisible to the report, pipeline, and AI layers.
4. `default_constants` are copy-pasted from Hedonic; there is no PR-appropriate QC.
5. No R parity test for the port.
6. `plot_breaking_point` silently drops its range params.
7. Missing `OptoCol1` yields a silent empty result rather than a clear error.

---

# Decided direction (grilling session, 2026-09-22)

Vocabulary is fixed in `CONTEXT.md` under **Progressive Ratio**; the facet
decision is ADR-0013. This section lists what was decided so the survey above
can be read as "before" and this as "after".

## Config

- **Roles.** Each `dfms:` entry gains `paired_chambers: [c, c, c]` — exactly one
  chamber from each Chamber Group (1–2, 3–4, 5–6). Yoked is the other member,
  never written. Missing key, wrong count, or two from one group is a lint and
  load error. Both chambers of a group must carry the same Treatment.
- **Side.** No new key. The type requires `well_names` A (sucrose) and B
  (yeast); the existing per-DFM physical key `pi_direction` places well A.
  The Config Editor relabels the field for this type ("Sucrose (well A) side")
  and shows a Paired marker per chamber in the DFM tab.
- **Facets.** Owned by the type (`facets_fixed = True`): Training and Test,
  split at each group's own training end. `facet_cutoffs` must not appear; the
  Design editor hides the cutoff controls. The 30-minute default is gone.
- **Constants.** Hedonic's three cutoffs stay as defaults, plus
  `require_training_complete: true`, the light QC's thresholds, and the
  breaking point's `pr_break_gap_min: 120`; `pr_test_window_min` has no
  default (unset means no cap).
- The R `configuration` 1–4 parameter is retired everywhere.

## Loading and training

- Training flag read per well from its own column (`> 40000`, value − 65536),
  as `_calculate_progressive_ratio_training` already does; it will additionally
  record per-well "cleared at" vs "never cleared".
- A group's training end = last flagged minute of well A of its paired
  chamber. Other wells of the group disagreeing or never clearing → per-well QC
  warning in `summary.txt` and the QC output, never an error.
- Paired well A never clears → both chambers keep their rows with
  `TrainingComplete = false`, `TrainingMinutes = NA`, no Test-facet data;
  `require_training_complete` auto-removes the group via the existing
  exclusion path and it appears in the exclusions table.  (Until the light QC
  change that path never ran in the pipeline; `execute_basic_analysis` now
  runs `auto_remove_chambers()` once before writing the summary.)
- **Light QC** (ADR-0013 addendum, `pr_light_qc.py`): per group, self-triggered
  light (a run of `pr_lick_free_run` lick-free Test light events) and
  implausible training (light events, zero sucrose licks) fail the group and
  remove it through auto-removal while `exclude_failed_pr_groups` is on; no
  increasing licks-per-event trend and a rising or elevated Sucrose Well
  resting level are warnings. Computed over the whole recording; the QC
  figures and `pr_light_qc.csv` keep an auto-removed group in view.
- Light on for a chamber = OR of its two wells' `OptoCol1` bits (the data
  shows all four bits of a group set together).
- Version-2 data (no flags, no `OptoCol1`) is a clear load error for this type.

## Outputs (`analysis/`)

| File | Grain | Notes |
|---|---|---|
| `feeding_summary.csv` | chamber, whole recording | standard two-well columns + `Group`, `Role`, `TrainingMinutes` (the group's training end, on both its rows), `TrainingComplete`, `LightOn_sec`, `LightQC` (the group's light QC flags), `LickFreeLightEvents`, `PersistA`, `PersistACensored` (Sucrose Persistence over the Test phase) |
| `feeding_summary_facet.csv` | chamber × Facet (Training, Test) | per-group windows; `StartMin`/`EndMin` vary by row; `PersistA` blank on Training rows |
| `paired_yoked_diff.csv` | Chamber Group × Facet | Paired − Yoked for LicksA/B, EventsA/B, PI, MedDurationA/B, PersistA; plus `PairedChamber`, `YokedChamber`, `TrainingMinutes`, `LightOn_sec`, `dPersistCensored`; no row if either chamber is missing |
| `pr_breaking_point.csv` | Chamber Group | `BreakingPoint`, `BreakMin`, `Censored`, `TestMinutes`, `LargestRequirement`, `LickFreeLightEvents`, `LightQC`; groups with both chambers and complete training |
| `pr_still_responding.png` | experiment | Kaplan-Meier fraction of paired flies reaching each ratio, per Treatment, censored flies as ticks |
| `pr_cumulative_diff.png` | experiment | Cumulative Difference Curve: mean ± SEM per Treatment over groups, truncated to the range every group covers, faint group traces to each group's end, x = min since training end, 1-min bins, raw licks |
| `pr_cumulative_licks_dfm<id>.png` | DFM, panel per group | QC: paired and yoked cumulative well-A licks, light-on samples as points, lick-free light events as rings, x from 0 at the group's training end |
| `pr_light_qc.csv` | Chamber Group | light QC verdict: training/Test light events, training licks, lick-free events and longest run (`LickFreeRunStartMin`, min since training end), licks-per-event trend (`TrendRho`, `TrendSlope`), Sucrose Well resting level (start/max/end/rise, ratio to the DFM's other Sucrose Wells), `Flags`, `Verdict`, `Excluded`, `Notes` |
| `pr_light_events.csv` | Test Light Event | the Paired chamber's breaking-point table stacked with `DFM, Group, PairedChamber, Event`: `MinutesSincePrev`, `LicksSincePrev` (licks from the previous event's end to this one's), `LickFree`, `RestingLevel`, `Counted` (the Breaking Point holds it) |
| `pr_light_events_dfm<id>.png` | DFM, panel per group | QC: licks per Test light event, lick-free events as hollow red rings, the group's trend and the estimated requirement |
| `pr_resting_level_dfm<id>.png` | DFM, panel per group | QC: the paired Sucrose Well's per-minute median raw level against the DFM's other Sucrose Wells, light onsets as a rug |
| `summary.txt` | — | gains per-group training table, flag-disagreement warnings, the light QC section and the breaking point with its sensitivity to the gap (60, 120, 240 minutes and the configured value) |

Cumulative curves always use raw lick counts; the tables obey
`transform_licks` as every type does.

## Report set and statistics

- Report set: `timecourse_pr_diff` (new Plot Spec, timecourse family, data
  source `paired_yoked_diff` binned), then `faceted_licks`, `faceted_events`,
  `faceted_pi` defaulting to Facet = Test. Generic `timecourse_licks` is out of
  the default set (recording-time x axis) but stays available in the Script
  Editor.
- Combined Analysis stacks `paired_yoked_diff.csv` with an `Experiment` column.
  Statistics: primary = pooled tests and mixed model on the difference table,
  Facet = Test, Treatment fixed, DFM nested in Experiment random, one
  observation per Chamber Group. Per-chamber tables run as today, with `Role`
  available for splitting, as the secondary section.

## Breaking point

Rewritten on the new model, spirit preserved: one method returning the
per-light-on-period table (minutes since training end, `DeltaMinutes`,
`DeltaLicks`) for a chamber, driven by the group's training end and the
chamber's Role. `plot_breaking_point` becomes a per-DFM figure of that table
with the dead `start`/`end` params removed.

### Defined (grilling session, 2026-09-24; ADR-0014)

- **Definition.** The first-gap rule: the Paired fly's lick-backed Test Light
  Events before its first pause longer than `pr_break_gap_min` (Δt, default
  120 minutes). Pauses run from training end to the first event, between
  events, and from the last event to the Test end; a pause of exactly Δt does
  not break. `BreakMin` is the minute of the last event counted, 0 when none.
- **Paired only.** The Yoked fly's light is its partner's, so it has no
  breaking point; paired versus yoked is compared on feeding metrics.
- **Lick-free Light Events** are removed before the rule: they neither count
  nor end a pause.
- **Censored** when no pause over Δt comes before the Test window ends: the
  count is a lower bound, `n+` in text, an open symbol or a tick in figures.
- **Test window** = recording end − training end, capped at
  `pr_test_window_min` (design constant, off by default) before the rule runs.
- **No clock awareness**: the data carry no photoperiod.
- **Sucrose Persistence**, for both roles: minutes since training end of the
  last Sucrose Well feeding event (the events `EventsA` counts) before the
  first pause over Δt, sharing Δt and the Test window. `PersistA` and
  `PersistACensored` on summary rows, `dPersistA` and `dPersistCensored` in the
  difference table; a censored difference is kept and flagged.
- **Statistics**, one observation per Chamber Group: Welch / Tukey and the
  mixed model with censored counts entered as observed, plus a pairwise
  log-rank test (`p_logrank`) and the Kaplan-Meier still-responding curve. The
  Paired-Yoked Difference is also tested against zero per treatment: paired t,
  Wilcoxon signed-rank, and the mixed model's intercept in a Project.
- **Figures**: the report dot plot with censored groups open, the
  still-responding curve (experiment report, `pr_still_responding.png`, and
  the Project Report's pooled breaking point section), and the per-DFM ΔLicks
  figure with the break marked and onsets past it grey.
- **Surface**: `pr_breaking_point.csv` (stacked as
  `<project>_BreakingPoint.csv`), `Counted` in the ledger, the summary's
  sensitivity table, the `breaking_point` and `plot_pr_still_responding`
  script actions with Hub buttons, and both constants in the Project Design
  dialog. The Project Report no longer derives a number from
  `pr_light_qc.csv`.

---

# As implemented (2026-09-24)

What the code does now, module by module. The decisions above say *why*;
this part says *where* and *how*, so a reader can go from a column in a CSV
to the line that computes it. Line numbers are those of the working tree on
this date and will drift.

## Module map

| Module | Holds |
|---|---|
| [`experiment_types/progressive_ratio.py`](../pyflic/base/experiment_types/progressive_ratio.py) | The type object: Chamber Groups, `paired_chambers` parsing and validation, `default_constants` (merging in the light QC and breaking point defaults), the report set, the Project Report's pooled results (`project_results_blocks`), the `output_manifest` |
| [`pr_light_qc.py`](../pyflic/base/pr_light_qc.py) | Light QC arithmetic on plain arrays: Light Events, lick crediting, runs, the lick trend, Resting Levels, the per-group verdict (`judge_group`), the requirement estimate |
| [`pr_breaking_point.py`](../pyflic/base/pr_breaking_point.py) | The first-gap rule on plain arrays: `BreakSettings`, `first_gap_break`, `breaking_point`, `persistence`, `format_count` |
| [`progressive_ratio_experiment.py`](../pyflic/base/progressive_ratio_experiment.py) | `ProgressiveRatioExperiment`: roles, training end, light QC per group, auto-removal, summaries and the difference table, the breaking point per group, every figure, `summary.txt` sections, the experiment-report hooks, the pipeline |
| [`analytics.py`](../pyflic/base/analytics.py#L444) | Statistics: `treatment_comparisons`, `kaplan_meier`, `still_responding`, `logrank_p`, `breaking_point_comparisons`, `zero_tests` |
| [`report_content.py`](../pyflic/base/report_content.py#L112) | Report figures and tables shared by both reports: `censored_dot_plot`, `still_responding_plot`, `dot_plot`, `stats_table`, `zero_test_table` |
| [`project.py`](../pyflic/base/project.py#L812) | Pooling (`combined_diff_frame`, `combined_light_qc_frame`, `combined_breaking_point_frame`), the Combined Analysis files, the pooled tests and the two mixed models |
| [`dfm.py`](../pyflic/base/dfm.py#L271) | Per-well training flags: `training_end_minutes`, `training_flag_state` (`cleared` / `never_cleared` / `never_flagged`) |

The two plain-array modules import nothing from the experiment, so their
rules are tested directly on numbers (`tests/test_pr_light_qc.py`,
`tests/test_pr_breaking_point.py`).

## Pipeline order

[`ProgressiveRatioExperiment.execute_basic_analysis`](../pyflic/base/progressive_ratio_experiment.py#L2376)
runs the two-well pipeline first, then the type's own steps:

1. **Two-well basic analysis.** This includes
   [`apply_auto_removal`](../pyflic/base/experiment.py#L2406), which runs
   `auto_remove_chambers` once, before any summary is written. The
   experiment report calls it too, and a second call keeps the first
   record, so both describe the same filtered design.
2. `paired_yoked_diff.csv`
3. `pr_cumulative_diff.csv`
4. `pr_light_qc.csv` and `pr_light_events.csv`, with the light QC lines
   echoed to the log
5. `pr_breaking_point.csv`
6. The figures (`write_pr_figures`): `pr_cumulative_diff.png`,
   `pr_still_responding.png`, and per DFM `pr_cumulative_licks_dfm<id>.png`,
   `pr_light_events_dfm<id>.png` and `pr_resting_level_dfm<id>.png`.

The breaking-point ΔLicks figure (`breaking_point_dfm<id>.png`) is **not**
written by the pipeline. It comes from the `plot_breaking_point` script
action (Hub: **Breaking-point plots**) and appears in the experiment report.

## QC protocols

Two checks run per Chamber Group: training, then the light QC. Both
describe the hardware and the fly over the **whole recording**, whatever
window a table uses. Neither ever raises. Each yields notes, flags and a
verdict, and a failure leaves through auto-removal (next section).

### Training

[`group_training`](../pyflic/base/progressive_ratio_experiment.py#L217)
reads the group's training end from the paired chamber's Sucrose Well
(`dfm.training_end_minutes`). Every other well of the group is then
classified with `dfm.training_flag_state` and reported in `notes`, never as
an error:

| Other-well state | Note |
|---|---|
| cleared more than 0.1 min (`_FLAG_TOLERANCE_MIN`) from the paired Sucrose Well | "`W<n>` (role ch A/B) cleared at *t*, paired sucrose well at *t′*" |
| cleared while the paired Sucrose Well never did | "… cleared at *t* while the paired sucrose well never cleared" |
| never cleared | grouped into one line: "… stayed flagged in training to the end of the recording" |
| never flagged (while some well of the group was) | grouped into one line: "… never carried the training flag" |
| no well of the group ever flagged | one note: "no training flag on any well of group *g*" (a v2 file, or a run that never trained) |

If the paired Sucrose Well itself never cleared, the group's
`training_end` is `None` (`TrainingComplete = false`) and a leading note
says whether it stayed flagged or was never flagged. Such a group has only a
Training window spanning the whole recording
([`group_windows`](../pyflic/base/progressive_ratio_experiment.py#L853)),
no Test rows, no breaking point and no persistence.

Outputs: `training_table()` (in `summary.txt` and the report's
*Progressive ratio: training* table) and `training_warnings()` (the
*Training-flag notes* list).

### Light QC: did the paired fly earn its light?

The firmware lights a group from its own reading of the paired Sucrose Well
during the run, while pyflic counts licks afterwards from the baselined
signal. A Sucrose Well whose resting level creeps up looks touched to the
firmware and flat to pyflic, so the light fires on its own. The light QC
measures that disagreement.

**Light Events and lick crediting.** The group's light is the OR of the
paired chamber's two `OptoCol1` well bits.
[`light_events`](../pyflic/base/pr_light_qc.py#L117) turns it into
`(onsets, ends)` sample indices. Then
[`licks_between_events`](../pyflic/base/pr_light_qc.py#L135) credits event
*k* with every Sucrose Well lick from the end of event *k − 1* (the start of
the recording, for the first) to the **end** of event *k*. Ending at the
event's end rather than its onset keeps the lick that triggered it, which
the detector may date to the first lit sample. The windows tile the
recording, so each lick counts once, and the first Test event counts from
the end of the last Training event.
[`light_events_table`](../pyflic/base/progressive_ratio_experiment.py#L418)
assembles this per group as `Phase, RecordingMinute, Minutes, CumLicks,
MinutesSincePrev, LicksSincePrev, LickFree, RestingLevel`. An event is
**Test** when its onset falls after training end.

**Resting Level.** [`resting_levels`](../pyflic/base/pr_light_qc.py#L208)
is the per-minute median of every well's raw (un-baselined) signal. The
median ignores brief licks, so what remains is the level the well rests at.
[`resting_summary`](../pyflic/base/pr_light_qc.py#L222) takes `start` (the
median over the first 30 minutes), `end` (over the last 30), `max` (the
highest centred 30-minute rolling median, at least 15 minutes of data,
floored at `start`, so one minute of a fly standing on the well is not the
level) and `level` (the median of the whole series). The reference
([`resting_reference`](../pyflic/base/progressive_ratio_experiment.py#L403))
is the per-minute median of the DFM's **other Sucrose Wells** only; the
yeast wells drift by hundreds of counts over a long run and would hide a
slow Sucrose Well rise. [`resting_ratio`](../pyflic/base/pr_light_qc.py#L244)
is `level / max(reference, 1)`.

**The checks** ([`judge_group`](../pyflic/base/pr_light_qc.py#L277)).
Settings come from `LightQCSettings.from_constants`; a bad value falls back
to its default.

| Flag | Fires when | Constant (default) | Severity |
|---|---|---|---|
| `self-triggered light` | the Test events contain a run of at least `pr_lick_free_run` consecutive Lick-free Light Events (`LicksSincePrev == 0`) | `pr_lick_free_run` (5, min 1) | **fails** |
| `implausible training` | the group had training light events and **zero** Sucrose Well licks up to training end. Fewer licks than pairings is normal (the firmware credits touches below the feeding threshold); none at all is not | — | **fails** |
| `no increasing trend` | at least `pr_trend_min_events` Test events, and Spearman's rho of licks per event against event number is below `pr_trend_min_rho` or undefined (a constant series). With fewer events, a note says a verdict needs more | `pr_trend_min_events` (5, min 3), `pr_trend_min_rho` (0.3) | warning |
| `resting level rise` | `max − start ≥ pr_resting_level_rise` | `pr_resting_level_rise` (15 counts) | warning |
| `resting level elevated` | `ratio ≥ pr_resting_level_ratio` **and** `level − reference ≥ pr_resting_level_rise`. The margin stops a DFM resting at 2–5 counts from calling 12 "elevated" | `pr_resting_level_ratio` (3) | warning |

Rho is the verdict rather than the slope because the firmware's increment is
user-defined and pyflic under-counts brief touches, so no particular slope
can be expected. The slope (`TrendSlope`) is reported beside it.

The first three checks need light and a Test phase. With no light at all,
a note says the light checks do not apply. With training incomplete, a
note says there is no Test phase to check. A group with training complete
but no Test events gets the note "the fly stopped before the first Test
requirement (a breaking point, not a failure)". The two resting-level checks
run for **every** group, whatever its training.

**Verdict.** `failed` if any failing flag fired, `warning` if only warnings
fired, `ok` otherwise. `Excluded = failed and exclude_failed_pr_groups`
(default true). `LickFreeRunStartMin` is the Test minute (since training
end) where the first failing run begins, the latest point a hand-set cutoff
could fall.

**Requirement estimate** ([`estimate_increment`](../pyflic/base/pr_light_qc.py#L343)).
For every trained group with at least 8 Test events (`INCREMENT_EVENTS`),
pyflic fits a line to licks per event over those first 8 and keeps it if
the slope is positive. The estimate is the median slope and intercept
across the kept groups. It feeds only the grey reference line and a summary
line; no verdict depends on it.

**Keeping failed groups in view.** The light QC reads the design **as
loaded**. `_design_snapshot` records `{(dfm, chamber): treatment}` before
auto-removal first thins it. The light QC table and every QC figure use
that snapshot, so an auto-removed group stays visible, its strip saying
`EXCLUDED: <flags>`. A chamber excluded by hand in `remove_chambers.csv`
never enters the design and stays out of both.

**Caching.** Light QC results are cached per DFM object (`_light_cache`)
and dropped when that object is replaced, as a QC Viewer recompute or a
parameter sweep does. `LightQC`, `LickFreeLightEvents` and the persistence
columns are recomputed on every summary, even one read back from the disk
cache, because their thresholds live in the design and the cache key does
not cover them.

**Outputs.** `pr_light_qc.csv` (`LIGHT_QC_COLUMNS`, one row per group);
`pr_light_events.csv` (the Test ledger); `LightQC` and
`LickFreeLightEvents` on every summary row and in `paired_yoked_diff.csv`;
the *Progressive ratio light QC* section of `summary.txt` (settings,
table, estimated increment, one explained line per flagged group, and the
excluded and failed-but-kept lists); the report's glance callout and QC
pages; `<project>_LightQC.csv` and the Stats text's *Flagged Chamber
Groups* block in a Project.

## Exclusion criteria

A chamber leaves the analysis in one of two ways:

- **By hand:** `remove_chambers.csv`, per exclusion group (ADR-0010). The
  chamber never enters the design, so it is in no table, not even the
  light QC table, and in no QC figure. The one exception is the
  breaking-point plots, which draw all six chambers of a DFM whatever the
  design; there a hand-excluded group is drawn with its break marked and
  no "— excluded" label, since that label only recognises auto-removal.
- **Automatically:**
  [`ProgressiveRatioExperiment.auto_remove_chambers`](../pyflic/base/progressive_ratio_experiment.py#L1159),
  run once per load by `apply_auto_removal`, on the whole-recording
  summary.

The automatic criteria, in the order they run:

| # | Criterion | Constant (default) | Removes | Where |
|---|---|---|---|---|
| — | light QC verdicts computed on the unthinned design and cached | — | nothing yet | `light_qc_failed_groups()` |
| 1 | `LicksA` or `LicksB` is NaN | always on | the chamber | [`Experiment.auto_remove_chambers`](../pyflic/base/experiment.py#L91), untransformed licks |
| 2 | `LicksA` or `LicksB` < cutoff | `min_untransformed_licks_cutoff` (20) | the chamber | same, untransformed licks |
| 3 | `MedDurationA` or `MedDurationB` > cutoff | `max_med_duration_cutoff` (13.0) | the chamber | PR override |
| 4 | `EventsA` or `EventsB` > cutoff | `max_events_cutoff` (150000) | the chamber | PR override |
| 5 | `TrainingComplete` is false | `require_training_complete` (true) | **both chambers** of the group | PR override |
| 6 | the group's light QC failed (`self-triggered light`, `implausible training`) | `exclude_failed_pr_groups` (true) | **both chambers** of the group | PR override |

Criteria 3–6 are checked only for chambers that criteria 1–2 kept. All the
reasons that fire for a chamber are joined into one `Reason`, for example
`light QC failed in chamber group 2: self-triggered light
(exclude_failed_pr_groups)`. Criteria 5 and 6 are facts about the group, so
they fire on both of its rows.

**Where the record goes.** `removed_chambers.csv` (written by
`write_removed_chambers`) holds every removed chamber with its reason. The
experiment's `filter_criteria_summary` gains a *Progressive Ratio filter
(additional criteria)* block stating each rule as configured, and
`summary.txt` lists the removed chambers. A Project reads both sources into
`<project>_Excluded.csv`, with `Source` = `manual` or `auto`.

**What a removal does downstream.** Criteria 1–4 remove a single chamber;
its partner stays in the per-chamber tables. Every within-group output
needs both chambers, though, so the group drops out of
`paired_yoked_diff.csv`, `paired_yoked_delta`, the cumulative difference
curve and `pr_breaking_point.csv`. A group with incomplete training is
absent from all Test outputs whether or not criterion 5 is on. With
`require_training_complete: false` its rows stay, with `TrainingComplete =
false`. With `exclude_failed_pr_groups: false` a failed group stays in
every table, whole, with its flags. The summary and the report say
"FAILED but retained", and the Project's Stats text warns that such groups
are in the pooled numbers.

## Breaking point and Sucrose Persistence

### The rule

[`first_gap_break(times, test_end_min, gap_min)`](../pyflic/base/pr_breaking_point.py#L107)
takes response times in minutes since training end:

1. Ignore times that are not finite or fall outside `[0, test_end_min]`,
   and sort the rest (stably).
2. Walk the sorted times with `prev = 0` (training end). At the first
   response with `t − prev > gap_min`, stop: `count` is the responses so
   far, `last_min = prev`, `censored = false`.
3. If no gap inside the window broke the count, check the tail: `censored =
   not (test_end_min − prev > gap_min)`. A long tail means the fly stopped
   inside the window, so the count is observed; a short one means the fly
   was still responding when the window ended.

The comparison is strict, so a gap of exactly `gap_min` does not break.
`last_min` is 0 when nothing was counted, so a group with no responses is
either a break at 0 (window longer than Δt) or censored at 0 (window
shorter). `counted` is a boolean mask aligned with the input, whatever its
order. The result is a `GapBreak(count, last_min, censored, counted)`.

[`breaking_point(minutes, lick_backed, …)`](../pyflic/base/pr_breaking_point.py#L135)
runs the rule on the lick-backed events only and maps `counted` back to the
full input. Lick-free events therefore neither count nor end a gap.
[`persistence(event_minutes, …)`](../pyflic/base/pr_breaking_point.py#L156)
runs it on feeding-event onsets and returns `(last_min, censored)`.

### Settings

[`BreakSettings.from_constants`](../pyflic/base/pr_breaking_point.py#L64)
reads two constants:

- **`pr_break_gap_min`** (default 120). A missing, non-numeric,
  non-positive or non-finite value falls back to 120.
- **`pr_test_window_min`** (no default). Unset, empty, 0 or invalid means
  no cap.

`test_end(available)` is `max(0, available)`, capped at
`pr_test_window_min` when that is set. `describe()` is the settings line in
`summary.txt`.

### Per Chamber Group

On [`ProgressiveRatioExperiment`](../pyflic/base/progressive_ratio_experiment.py#L1704):

- **`_test_window(dfm, group)`**: `settings.test_end(recording end −
  training end)`, where recording end is the DFM's last raw `Minutes`.
  `None` when training never completed.
- **[`group_break`](../pyflic/base/progressive_ratio_experiment.py#L1725)**:
  `breaking_point` on the group's Test Light Events. The inputs are
  `Minutes` from `light_events_table` and lick-backed = `not LickFree`,
  where `LickFree` means `LicksSincePrev == 0`. Returns `None` for an
  untrained group. **Paired only:** the light and the licks are the paired
  chamber's.
- **[`chamber_persistence`](../pyflic/base/progressive_ratio_experiment.py#L1746)**:
  for either role, `persistence` on the onsets of the chamber's Sucrose
  Well feeding events after training end, shifted to minutes since
  training end. The onsets are the rows of `dfm.event_df` with a positive
  value in the well-A column, the same events `EventsA` counts. It shares
  the group's Test window and Δt, and is cached per `(chamber, settings)`.
- **[`breaking_point_table`](../pyflic/base/progressive_ratio_experiment.py#L1772)**:
  one row per light onset after training end, with `Minutes, CumLicks,
  DeltaMinutes, DeltaLicks` (the first row's deltas are 0). For the paired
  chamber the rows are the Test ledger plus `LEDGER_COLUMNS` and `Counted`.
  For the yoked chamber they are the same onsets with its own well-A licks.
  An event lit across training end belongs to training.
- **[`breaking_point_summary`](../pyflic/base/progressive_ratio_experiment.py#L1963)**:
  one row per group with **both chambers in the design and training
  complete**. Columns: `Treatment` (the paired chamber's), the factors,
  `DFM, Chamber, Group, PairedChamber, BreakingPoint, BreakMin, Censored,
  TestMinutes, LargestRequirement` (the most licks credited to one counted
  event, descriptive only), `LickFreeLightEvents, LightQC`. This is
  `pr_breaking_point.csv`.
- **[`breaking_point_sensitivity`](../pyflic/base/progressive_ratio_experiment.py#L2011)**:
  every summary group re-run at Δt = 60, 120, 240
  (`SENSITIVITY_GAPS_MIN`) and the configured value, as `BP_<gap>` and
  `Censored_<gap>`.
  [`breaking_point_lines`](../pyflic/base/progressive_ratio_experiment.py#L2045)
  prints both tables (the configured gap starred, censored counts as `n+`)
  and a line counting the censored groups. That text is the *Progressive
  ratio breaking point* section of `summary.txt` and the `breaking_point`
  action's log.

### Where persistence lands

`_with_persistence_columns` writes `PersistA` and `PersistACensored` on
every per-chamber summary row. Like `TrainingMinutes`, they are a fact about
the group's Test phase, whatever window the row covers.
`feeding_summary_facet` blanks both on Training rows.
[`paired_yoked_diff`](../pyflic/base/progressive_ratio_experiment.py#L904)
adds `dPersistA` (paired − yoked) and `dPersistCensored`: true if either
fly is censored, and missing if either value is missing. A censored
difference is kept and flagged, never dropped.

## Figures

| Figure | Built by | File | How to make it | Shows |
|---|---|---|---|---|
| Cumulative Difference Curve | [`plot_cumulative_diff`](../pyflic/base/progressive_ratio_experiment.py#L1387) | `pr_cumulative_diff.png` | pipeline; `plot_pr_cumulative_diff` (bin size) | per group, paired − yoked cumulative raw well-A licks since training end in 1-min bins (faint lines); mean ± SEM per Treatment over the range **every** group covers ([`cumulative_diff_stat`](../pyflic/base/progressive_ratio_experiment.py#L1351)); dashed zero. Excluded groups absent |
| Still responding | [`rc.still_responding_plot`](../pyflic/base/report_content.py#L196) via `plot_still_responding` | `pr_still_responding.png` | pipeline; `plot_pr_still_responding` | per Treatment, the Kaplan-Meier fraction of paired flies reaching each ratio, as an `hv` step from (0, 1); a fly that broke after *k* drops the curve at *k* + 1; censored flies as `\|` ticks; legend carries *n* |
| Breaking-point plots | [`plot_breaking_point_dfm`](../pyflic/base/progressive_ratio_experiment.py#L1831) | `breaking_point_dfm<id>.png` | `plot_breaking_point` only (not the pipeline); experiment report | ΔLicks per light-on period against minutes since training end, one panel per chamber (both roles), BP `n` / `n+` in the paired strip. Counted events blue, events past the break or the Test window grey, lick-free events hollow red rings; dashed line at `BreakMin` unless censored. Auto-removed groups drawn for reference with no break, strip "— excluded" |
| Breaking point by treatment | [`rc.censored_dot_plot`](../pyflic/base/report_content.py#L154) | report only | experiment report; Project Report | `BreakingPoint`, one point per group, jittered, mean ± SEM; censored points open |
| Paired − yoked dot plots | `rc.dot_plot(…, hline_at=0)` | report only | experiment report; Project Report | Test-phase `dLicksA`, `dPI` and (when present) `dPersistA`, one point per group, zero drawn |
| Training-aligned traces (QC) | [`plot_cumulative_licks_dfm`](../pyflic/base/progressive_ratio_experiment.py#L1439) | `pr_cumulative_licks_dfm<id>.png` | pipeline; `plot_pr_cumulative_licks` | per group, paired and yoked cumulative well-A licks since training end; light-on bins as points; Test lick-free events as black rings on the paired trace; strip = training end and light QC verdict. Uses the snapshot |
| Licks per light event (QC) | [`plot_light_events_dfm`](../pyflic/base/progressive_ratio_experiment.py#L1502) | `pr_light_events_dfm<id>.png` | pipeline; `plot_pr_light_events` | per group, `LicksSincePrev` against Test event number; lick-free events as hollow red rings; the group's least-squares trend dashed; the experiment's estimated requirement grey, stopped just above the panel's data |
| Sucrose Well resting level (QC) | [`plot_resting_level_dfm`](../pyflic/base/progressive_ratio_experiment.py#L1592) | `pr_resting_level_dfm<id>.png` | pipeline; `plot_pr_resting_level` | per group, the paired Sucrose Well's per-minute median raw level against the median of the DFM's other Sucrose Wells; light onsets as a rug; training end dashed |
| `timecourse_pr_diff` | Plot Spec in `pubfigures.py` | Plot Editor / Project Report | Project | the pooled difference curve, from the stacked `pr_cumulative_diff.csv` |

The inherited per-treatment plots are made role-aware. The time courses and
multi-metric summaries split each Treatment into `<treatment> · paired` and
`<treatment> · yoked` (`_resolve_group_col`).
[`plot_dot_metric_by_treatment`](../pyflic/base/progressive_ratio_experiment.py#L1103)
plots the within-group difference itself
([`paired_yoked_delta`](../pyflic/base/progressive_ratio_experiment.py#L1034):
any metric, resolved per chamber and then differenced), panelled by Facet,
with zero drawn.

## Statistics

Every test here has **one observation per Chamber Group**.

| Question | Function | Test | Censoring |
|---|---|---|---|
| Do treatments differ in the Paired-Yoked Difference? | [`treatment_comparisons`](../pyflic/base/analytics.py#L444) | two treatments: Welch's t; three or more: Tukey HSD on every pair | `dPersistA` entered as observed; `dPersistCensored` is only a flag |
| Is the difference non-zero within a treatment? | [`zero_tests`](../pyflic/base/analytics.py#L723) | one-sample t on the differences (= paired t of paired against yoked), `significant` = `p_t < 0.05`, Wilcoxon signed-rank beside it | as above |
| Do treatments differ in breaking point? | [`breaking_point_comparisons`](../pyflic/base/analytics.py#L687) | the `treatment_comparisons` rows on `BreakingPoint`, plus `p_logrank` per pair | t / Tukey / mixed model enter censored counts as observed; the log-rank test treats them as censored |
| How many flies reach each ratio? | [`still_responding`](../pyflic/base/analytics.py#L604) over [`kaplan_meier`](../pyflic/base/analytics.py#L578) | Kaplan-Meier on `BreakingPoint`, event = not censored | a censored fly is at risk at its own count |

Details that affect the numbers:

- **Dropped observations.** `treatment_comparisons` leaves out a treatment
  with fewer than two observations, and any pair whose p is not finite.
  `zero_tests` needs two differences that are not all equal; a failed
  Wilcoxon becomes `None`.
- **Log-rank.** [`logrank_p`](../pyflic/base/analytics.py#L660) is the
  two-sample chi-square (1 df) with ties, run pairwise and **unadjusted**
  when there are three or more treatments. It returns NaN, reported as
  `None`, when there is no event or no variance.
- **Mixed models, Projects only**, and only when the frame spans two or
  more Members. [`Project._mixed_p`](../pyflic/base/project.py#L1064) fits
  `Value ~ is_b` for a treatment pair, and
  [`Project._mixed_p0`](../pyflic/base/project.py#L1105) fits `Value ~ 1`
  for one treatment's differences, reporting the intercept's p. Both group
  on `Experiment` with a DFM variance component (DFM nested in Experiment),
  fall back to Experiment alone if that fails to converge, and return
  `None` rather than raise.
- **Metrics tested.** The experiment report tests `DIFF_REPORT_METRICS`
  (`dLicksA, dLicksB, dEventsA, dPI, dMedDurationA, dPersistA`) on Test
  rows. A Project tests `DIFF_TEST_METRICS` (the same plus `dEventsB`) on
  the type's report Facets (`["Test"]`). Per-chamber pooled tests still run
  as the secondary section.

**Where the statistics appear.**

- **Experiment report**
  ([`report_results_blocks`](../pyflic/base/progressive_ratio_experiment.py#L2253),
  when comparisons are enabled): the between-treatment and against-zero
  tables for the difference, and the breaking point comparisons (no mixed
  model, since there is one member).
- **Project Report**
  ([`project_results_blocks`](../pyflic/base/experiment_types/progressive_ratio.py#L158)):
  the pooled breaking point with the mixed model and the log-rank test, and
  the pooled difference.
- **`<project>_Stats.txt`**: the *Flagged Chamber Groups* block, then the
  difference as the primary section (between treatments, then against zero
  via [`diff_zero_rows`](../pyflic/base/project.py#L1029)), then the
  breaking point ([`breaking_point_rows`](../pyflic/base/project.py#L1046)).

`summary.txt` carries the breaking point and sensitivity tables but no
tests.

**Pooling.** [`build_combined_analysis`](../pyflic/base/project.py#L896)
stacks each Member's `paired_yoked_diff.csv`, `pr_light_qc.csv` and
`pr_breaking_point.csv`, each with an `Experiment` column, into
`<project>_PairedYokedDiff.csv`, `<project>_LightQC.csv` and
`<project>_BreakingPoint.csv`. A Member analysed before
`pr_breaking_point.csv` existed has none and is not guessed at; the Project
Report says to re-run it.

## Surfaces

Script actions, all gated `requires: progressive_ratio` (`_PR_GATED` in
[`runner.py`](../pyflic/base/script_editor/runner.py#L22)):

| Action | Writes | Hub button |
|---|---|---|
| `paired_yoked_diff` | `paired_yoked_diff.csv` | Analyze → *Paired − yoked difference CSV* |
| `breaking_point` | `pr_breaking_point.csv`, logs `breaking_point_lines()` | Analyze → *Breaking point CSV* |
| `pr_light_qc` | `pr_light_qc.csv`, `pr_light_events.csv`, logs the flagged groups | QC → *Light QC table* |
| `plot_pr_light_events` | `pr_light_events_dfm<id>.png` | QC → *Licks per light event (QC)* |
| `plot_pr_resting_level` | `pr_resting_level_dfm<id>.png` | QC → *Sucrose Well resting level (QC)* |
| `plot_pr_cumulative_diff` (`binsize`) | `pr_cumulative_diff.csv` + `.png` | Plots → *Cumulative difference curve* |
| `plot_pr_cumulative_licks` (`binsize`) | `pr_cumulative_licks_dfm<id>.png` | Plots → *Training-aligned traces (QC)* |
| `plot_pr_still_responding` | `pr_still_responding.png` | Plots → *Still-responding curve* |
| `plot_breaking_point` | `breaking_point_dfm<id>.png` | Plots → *Breaking-point plots* |

The Hub shows these groups only while a Progressive Ratio member is loaded.
All the constants above have fields in the Project Design dialog (the
Design, light QC and breaking point sections).

## Tests

Counts are test functions; parametrised cases add more.

| File | Covers |
|---|---|
| `tests/test_pr_light_qc.py` (28) | light events and crediting, runs, trend, resting-level summaries, each flag with its margins, settings, the increment estimate, the table and ledger, fresh QC columns on a cached summary, exclusion through auto-removal and switching it off, the pipeline, the QC figures, the report, script actions, a parameter recompute, and the Project's flagged groups |
| `tests/test_pr_breaking_point.py` (33) | the first-gap rule's edge cases (exact-Δt gap, the tail, empty input, the window cap, unsorted input, lick-free events), the real-dataset group the rule was designed on, persistence, settings, Kaplan-Meier against the textbook, log-rank against statsmodels, zero tests, the mixed intercept only across members, CSV booleans, the summary, ledger and sensitivity, the pipeline, script actions, both reports, and Project pooling |
| `tests/test_progressive_ratio.py` (37) | groups and roles, `paired_chambers` validation in the loader and linter, training end and flag notes, per-group facets, the summary columns, the difference table and delta, incomplete-training removal and switching it off, the cumulative curves and their truncation, the role-aware plots, the breaking-point table, the pipeline outputs, the `timecourse_pr_diff` spec, Project pooling, and headless figure rendering |

Shared fixtures are in `tests/pr_fixtures.py`.
- **Config Editor** (follow-up, same day): a *Progressive Ratio* section shows
  all nine of the type's constants under ADR-0011's rule (default as
  placeholder, only typed values written). `PR_CONSTANT_FIELDS` in the type
  module is the one description the Config Editor, the Project Design dialog
  and `ProgressiveRatioExperimentType.validate()` read, so a bad value is
  refused everywhere, including at load.
