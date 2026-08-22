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

A schedule in which the effort required for each reward increases over the session. The
measure of interest is the **breakpoint** — the point at which the fly stops working for
the reward, taken as an index of motivation.

pyflic identifies this from lights-on periods after training ends. For each well, it
produces one row per lights-on period at the moment the lights switch on, with
`CumLicks`, `DeltaMinutes` and `DeltaLicks` — the change in licking across successive
periods. The breakpoint is where `DeltaLicks` falls away.

Two requirements apply:

- **Every DFM must use `chamber_size: 2`**; loading validates this and fails otherwise.
- **Light data is required.** The analysis is defined in terms of lights-on periods, so a
  recording without an `OptoCol1` column cannot be analysed this way. See
  [Light state and phase analysis](concepts-light-phase.md).

The end of training is read from the DFM's training data when present, and otherwise
defaults to minute 0. You can pass it explicitly, and passing `0.0` skips the training
filter entirely.

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
