# Progressive Ratio — what is currently implemented

> **Status (2026-09-22):** the "Decided direction" at the end of this file has
> been implemented — type object, loader validation, per-group training end,
> augmented summaries, the Paired-Yoked Difference table, both figures, the
> pooled statistics and `timecourse_pr_diff` spec, script actions, linter,
> Config/Design editor support and help topics. Tests:
> `tests/test_progressive_ratio.py` (fixtures in `tests/pr_fixtures.py`).
> Two pre-existing bugs found on the way are fixed and noted in the CHANGELOG:
> the empty tail Facet (`(start, 0)` read literally) and the never-written
> `removed_chambers.csv`. The breaking-point details remain provisional. The
> survey below describes the code **before** this work.

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
  `require_training_complete: true`.
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
  exclusion path and it appears in the exclusions table.
- Light on for a chamber = OR of its two wells' `OptoCol1` bits (the data
  shows all four bits of a group set together).
- Version-2 data (no flags, no `OptoCol1`) is a clear load error for this type.

## Outputs (`analysis/`)

| File | Grain | Notes |
|---|---|---|
| `feeding_summary.csv` | chamber, whole recording | standard two-well columns + `Group`, `Role`, `TrainingMinutes` (the group's training end, on both its rows), `TrainingComplete`, `LightOn_sec` |
| `feeding_summary_facet.csv` | chamber × Facet (Training, Test) | per-group windows; `StartMin`/`EndMin` vary by row |
| `paired_yoked_diff.csv` | Chamber Group × Facet | Paired − Yoked for LicksA/B, EventsA/B, PI, MedDurationA/B; plus `PairedChamber`, `YokedChamber`, `TrainingMinutes`, `LightOn_sec`; no row if either chamber is missing |
| `pr_cumulative_diff.png` | experiment | Cumulative Difference Curve: mean ± SEM per Treatment over groups, truncated to the range every group covers, faint group traces to each group's end, x = min since training end, 1-min bins, raw licks |
| `pr_cumulative_licks_dfm<id>.png` | DFM, panel per group | QC: paired and yoked cumulative well-A licks, light-on samples as points, x from 0 at the group's training end |
| `summary.txt` | — | gains per-group training table and flag-disagreement warnings |

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
with the dead `start`/`end` params removed. No breakpoint scalar yet; the
details are to be revisited once the structure above is in place.
