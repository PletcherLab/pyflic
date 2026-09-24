# Experiment types and chamber layouts

Two things used to be one setting, and separating them is the biggest change in how a
config is written.

- An **Experiment Type** is the *assay* — the top-level thing a scientist chooses. It
  selects a chamber layout and constrains everything else: the required `well_names`, the
  facet cutoffs and phase names, the default quality cutoffs, which analyses run, which
  plots the report contains.
- A **Chamber Layout** is the *hardware* — how many wells a chamber has. `single_well` or
  `two_well`. Several different assays share one layout.

`two_well` was never an assay. It described a plate, which is why a hedonic experiment had
to be "a kind of two-well experiment" to exist at all. Now it is what it always was.

## What a config says

A **typed** config names its type and says nothing about the layout — the type owns it,
and the value is derived rather than written to disk:

```yaml
global:
  experiment_type: Hedonic
  well_names: {A: S5, B: S5Y5}
```

Stating `chamber_layout` or `params.chamber_size` in a typed config is an error. There is no
longer a check that the two agree, because they can no longer disagree — the config does not
get a vote.

A **Custom** experiment is the absence of a chosen type. It states its layout directly:

```yaml
global:
  chamber_layout: two_well
```

A config with no `experiment_type` key **is** a Custom Experiment. That is today's freeform
mode, with no type-level constraints applied.

## The shipped types

| `experiment_type` | Layout | Use it when |
|---|---|---|
| *(omitted)* / `Custom` | as stated | No type-level constraints wanted |
| `Hedonic` | `two_well` | Two-well choice designs where bout duration matters |
| `ProgressiveRatio` | `two_well` | Progressive-ratio schedules with a breakpoint |

## Migrating

`experiment_type: two_well` and `experiment_type: single_well` are no longer experiment
types. A config using either fails to load with the replacement spelled out:

```yaml
# before
global:
  experiment_type: two_well
  params: {chamber_size: 2}

# after
global:
  chamber_layout: two_well
```

Run `pyflic lint` on a folder to see every config that needs this.

## The layouts

### `single_well`



The simplest case. Each of the 12 wells is its own chamber with its own fly and its own
treatment, giving 12 chambers per DFM. There is no preference index — with one food source
there is nothing to prefer — and `correct_for_dual_feeding` defaults to `false`, since
there is no neighbouring well to bleed in.

Use it for consumption assays: how much did each treatment group eat.

### `two_well`

Wells are paired into 6 chambers per DFM, each holding one fly with two options.
Everything single-well reports is reported per well, plus the **preference index**. This
is the workhorse for choice experiments.

The details that matter here — `pi_direction`, counterbalancing, and dual-feeding
correction — are covered in
[Two-well choice and the preference index](concepts-two-well-pi.md).

### Hedonic

A two-well design analysed with attention to **how long** bouts last rather than how many
there are. The reasoning is that palatability shows up in bout duration: a fly presented
with something it finds more rewarding sustains longer feeding bouts, even when total lick
counts are similar.

Its distinctive output is the **weighted duration summary**, which computes, per treatment
and per well:

- `WeightedMeanDurationA` / `WeightedMeanDurationB` — the mean of each chamber's
  `MedDuration`, **weighted by that chamber's event count**
- `WeightedStdDurationA` / `WeightedStdDurationB` — the corresponding weighted standard
  deviations
- `N` — the number of chambers that contributed

The weighting is the point. A chamber whose median bout duration rests on 3 events is much
weaker evidence than one resting on 300, and an unweighted mean across chambers would
treat them as equal. Weighting by event count gives each chamber influence proportional to
how much it actually observed.

Hedonic experiments also get a dedicated plot contrasting Well A and Well B median
durations, faceted by treatment.

### Progressive Ratio

A two-well operant assay built on **chamber groups**. Each DFM's six chambers form three
groups — chambers 1+2, 3+4 and 5+6 — that share one light circuit and one treatment. In
each group one chamber is **paired**: its feeding at the sucrose well (always **well A**)
turns the light on. The other is **yoked**: lit at the same moments as its partner,
regardless of its own behaviour. The config names the paired chamber per DFM and the
yoked one is derived:

```yaml
dfms:
- id: 1
  params: {pi_direction: left}     # which side well A (sucrose) is on
  paired_chambers: [1, 4, 5]       # exactly one chamber from each group
  chambers: {1: Ctrl, 2: Ctrl, 3: Exp, 4: Exp, 5: Exp, 6: Exp}
```

Both chambers of a group must carry the same treatment; the loader refuses a config where
they differ, or where `paired_chambers` is missing, names two chambers of one group, or
none.

**Training.** The recording opens with a closed-loop training phase whose end the firmware
marks in the data itself: while a well is in training its raw value is offset by 65536
(any sample above 40000), and the offset disappears when training completes. Training end
is read per group from the paired chamber's sucrose well, and it differs between groups
because it depends on the fly. The other three wells of a group are expected to clear at
the same minute; when they do not, the summary lists the disagreement as a QC note and the
analysis proceeds on the paired well. A group whose paired fly never finishes training has
`TrainingComplete = false` on both chambers and, with the default constant
`require_training_complete: true`, both leave the analysis through the ordinary
auto-removal path.

**Facets** for this type are `Training` and `Test`, split at each group's own training end
rather than at a fixed minute, so the config never states `facet_cutoffs`. See
[Facets](concepts-facets.md).

**Light QC — did the paired fly earn its light?** The firmware lights a group from its own
reading of the paired chamber's sucrose well during the run; pyflic counts licks
afterwards, from the baselined signal. A sucrose well whose raw **resting level** creeps up
looks continuously touched to the firmware and flat to pyflic, so the light fires on its
own schedule and nothing about it was earned. Every chamber group is therefore checked,
over the whole recording whatever window a table uses:

| Check | Verdict |
|---|---|
| **Self-triggered light**: at least `pr_lick_free_run` (5) consecutive Test light events with no sucrose licks since the previous one | **fails** the group |
| **Implausible training**: training completed with light events but not a single sucrose lick | **fails** the group |
| **No increasing trend**: with at least `pr_trend_min_events` (5) Test light events, Spearman's rho of licks per event against event number is below `pr_trend_min_rho` (0.3) | warning |
| **Resting level rise**: the sucrose well's resting level peaks at least `pr_resting_level_rise` (15) counts above its first 30 minutes | warning |
| **Resting level elevated**: the sucrose well rests at `pr_resting_level_ratio` (3) times the DFM's other sucrose wells, and at least `pr_resting_level_rise` counts above them | warning |

A few lick-free light events are normal — pyflic's feeding threshold misses brief touches
the firmware counts — which is why a *run* of them, not a single one, fails a group, and
why training fails only on zero licks rather than on fewer licks than pairings. A group
with no Test light events at all has simply stopped before its first Test requirement: a
breaking point, reported, never flagged.

With the default `exclude_failed_pr_groups: true`, both chambers of a failed group leave
the analysis through auto-removal, with the reason in `removed_chambers.csv`. Switch it off
to keep them; `pr_light_qc.csv` lists every group's verdict either way, with
`LickFreeRunStartMin` — minutes after training end at which the first failing run began —
for deciding a cutoff by hand. The thresholds are design constants
([Parameters](reference-parameters.md)).

**Outputs** beyond the standard two-well summary: `Group`, `Role`, `TrainingMinutes`
(the chamber group's training end, carried on both its chambers' rows and in every Facet —
only a group that never finished training has none), `TrainingComplete`, `LightOn_sec`,
`LightQC` (the group's light QC flags; empty when clean) and `LickFreeLightEvents`
columns; `paired_yoked_diff.csv`, one
row per chamber group per Facet with paired-minus-yoked differences (`dLicksA`, `dPI`, …);
`pr_cumulative_diff.csv` and its figure, the cumulative difference curve;
`pr_light_qc.csv` (one row per chamber group) and `pr_light_events.csv` (one row per Test
light event, with the licks credited to it); and per-DFM QC figures — training-aligned
traces, licks per light event, and the sucrose well resting level. See
[Plots](plots-catalog.md).

**Statistics** in a Project treat the difference table as primary — one observation per
chamber group, treatment fixed, DFM nested within Experiment — with the per-chamber tables
as the secondary section.

Light data (`OptoCol1`) and version-3 files are required: the training flag and the light
state both live there.

## Changing type later

The experiment type is a property of the configuration, not the data, so you can analyse
the same raw CSVs both ways — put a second configuration file in the project directory with
a different `experiment_type` and its results land in their own `<config>_results/` folder.
This is a normal thing to do; it is what the per-config output namespacing exists for.

What you cannot do is change `chamber_size`, since that changes what a chamber *is* and
therefore which wells belong together.

---

Related: [Configuration file structure](config-structure.md) ·
[Plot catalogue](plots-catalog.md)
