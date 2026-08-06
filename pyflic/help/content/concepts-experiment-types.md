# Experiment types

`global.experiment_type` selects which analysis pyflic runs. All four share the same
signal processing and bout detection; they differ in what they compute *afterwards* and
which plots they can draw.

```yaml
global:
  experiment_type: two_well
```

If you omit it, pyflic chooses based on `chamber_size` — `1` gives single-well, `2` gives
two-well. Set it explicitly anyway: hedonic and progressive-ratio experiments both use
`chamber_size: 2` and cannot be inferred.

## The four types

| Value | Chamber size | Use it when |
|---|---|---|
| `single_well` | 1 | 12 independent wells per DFM, one food source each |
| `two_well` | 2 | Choice assays — two options per fly, preference index |
| `hedonic` | 2 | Two-well designs where bout *duration* is the measure of interest |
| `progressive_ratio` | 2 | Progressive-ratio schedules with a breakpoint |

## `single_well`

The simplest case. Each of the 12 wells is its own chamber with its own fly and its own
treatment, giving 12 chambers per DFM. There is no preference index — with one food source
there is nothing to prefer — and `correct_for_dual_feeding` defaults to `false`, since
there is no neighbouring well to bleed in.

Use it for consumption assays: how much did each treatment group eat.

## `two_well`

Wells are paired into 6 chambers per DFM, each holding one fly with two options.
Everything single-well reports is reported per well, plus the **preference index**. This
is the workhorse for choice experiments.

The details that matter here — `pi_direction`, counterbalancing, and dual-feeding
correction — are covered in
[Two-well choice and the preference index](concepts-two-well-pi.md).

## `hedonic`

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

## `progressive_ratio`

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
