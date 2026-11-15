# Experiment types and chamber layouts

Two things used to be one setting, and separating them is the biggest change in how a
config is written.

- An **Experiment Type** is the *assay* — the top-level thing a scientist chooses. It is a
  definition that selects a chamber layout and constrains everything else: the light the
  recording should have, the required `well_names`, the facet cutoffs and phase names, the
  default quality cutoffs, which analyses run, which plots the report contains, and which
  comparison the report leads with.
- A **Chamber Layout** is the *hardware* — how many wells a chamber has. `single_well` or
  `two_well`. Several different assays share one layout.

`two_well` was never an assay. It described a plate, which is why a hedonic experiment had
to be "a kind of two-well experiment" to exist at all. Now it is what it always was.

pyflic ships three types. A lab defines the rest itself, in the
[Experiment Type Editor](app-type-editor.md), from the analyses pyflic provides.

## What a config says

A **typed** config names its type. When the type fixes the layout, the config says nothing
about it — the type owns it, and the value is derived rather than written to disk:

```yaml
global:
  experiment_type: Hedonic
  well_names: {A: S5, B: S5Y5}
```

Stating `chamber_layout` or `params.chamber_size` in such a config is an error. There is no
check that the two agree, because they can no longer disagree — the config does not get a
vote.

A **Custom** experiment is the type with no constraints. It states its layout directly:

```yaml
global:
  chamber_layout: two_well
```

A config with no `experiment_type` key **is** a Custom Experiment. A lab type that leaves
the layout open (`chamber_layout: either`) states it the same way, beside its
`experiment_type`. A config of either kind that states no layout is `two_well`.

## The shipped types

| `experiment_type` | Layout | Light it expects | Use it when |
|---|---|---|---|
| *(omitted)* / `Custom` | as stated | any, or none | No type-level constraints wanted: a recording with no light, or any light program you do not want to declare |
| `Hedonic` | `two_well` | none | A two-well hedonic design, where bout duration matters, recorded without light |
| `ProgressiveRatio` | `two_well` | ProgressiveRatio or FixedRatio | A ratio program, with or without yoked controls: a light earned by feeding, and a breaking point |

Custom includes the Paired and yoked and the Light and dark modules, Hedonic the Hedonic
module, and ProgressiveRatio the Progressive Ratio and Light and dark modules (see
[Analysis Modules](#analysis-modules)). Hedonic and ProgressiveRatio also set default
quality cutoffs: `min_untransformed_licks_cutoff: 20`, `max_med_duration_cutoff: 13` and
`max_events_cutoff: 150000`.

They are read-only. To change one, build a type of your own starting from it (*New from…*
in the editor).

## Building your own type

A lab type is a definition you build once and reuse in every Project that runs the assay.
It states:

- its **name** — what `experiment_type:` holds, so letters, digits and underscores,
  starting with a letter, and never a shipped type's name (names that differ only in case,
  spaces, hyphens or underscores count as the same) — and a display name and a sentence for
  the report cover;
- its **chamber layout**: `single_well`, `two_well` or `either`;
- the **light it expects**: none, any, or a list of Paradigm types (OpenLoop, ClosedLoop,
  FixedInterval, NonFeeding, ProgressiveRatio, FixedRatio);
- the **wells** that must be named, and the **factor names** every Design of the type
  declares — a Design may add factors of its own, and it always supplies the levels;
- its **Analysis Modules** (below);
- its **control comparisons** (below);
- the **plots** its report shows after its modules' own, picked from the built-in Plot
  Specs, and Plot Specs of its own, imported from a Project's `plot_specs.yaml`; a new
  Project of the type starts with them;
- default facet cutoffs and their names, and whether a config may change them (a module
  whose Facets come from the data, Progressive Ratio, overrides them);
- default `constants:`, such as the quality cutoffs; a config's own value wins.

A type combines analyses pyflic provides; it never creates one. A new kind of analysis
needs code, and then appears in the editor for every type.

### Analysis Modules

| Module | What it adds | What it needs |
|---|---|---|
| **Progressive Ratio** (`ratio`) | Chamber groups, Training and Test split at each group's training end, the paired fly's light QC, the paired − yoked difference, the breaking point and sucrose persistence | two wells, both named, and a ProgressiveRatio or FixedRatio Paradigm among the light the type expects |
| **Hedonic** (`hedonic`) | Median bout duration at each well, plain and weighted by each chamber's events (the weighted duration summary), and the hedonic feeding plot | two wells, both named |
| **Paired and yoked** (`yoked`) | Per yoked linkage group, the paired chamber's feeding, its yoked chambers' mean and the difference, tested against zero (`yoked_comparison.csv`) | light; not beside Progressive Ratio, which keeps its own |
| **Light and dark** (`light_dark`) | Under open-loop light, each chamber's licks per minute lit and dark and its light feeding index, in the reports | light |

The Paired and yoked comparison is described in
[Paired and yoked designs](concepts-optogenetics.md#paired-and-yoked-designs), and the lit
and dark columns in [the light table](concepts-optogenetics.md#the-light-table).

The editor refuses a combination that cannot run — Progressive Ratio on one well, say, or
the Paired and yoked module on a type that expects no light — and a type may include only
one of Progressive Ratio and Hedonic, since each brings an analysis class of its own.

The QC, Analyze and Plots panels of the Hub offer a module's items only to a member whose
type includes it.

## The light a type expects

When a recording has a `Program.txt`, pyflic checks it against the light the type expects,
and the recording fails to load if the program runs a Paradigm type the type does not take.
The message names the Paradigm type and the DFMs and Program Intervals that ran it.

- A type that expects **no light** takes no lit interval at all.
- A type that expects **a list of Paradigm types** takes those, with OpenLoop light beside
  them as a control — a whole DFM run open loop, or wells set `On` — and dark intervals.
- A type that expects **any** light takes everything. Custom does.

An interval whose wells are all `Off` runs nothing, whatever its Paradigm, and a recording
made without a program (`***ALL DFM***`, every well `Off`) has no light. A config that
says `optogenetics: no` keeps pyflic from reading the program, so nothing is checked.

How the light is *judged* — the light QC, the light table, the rule figures — always
follows the Paradigms that actually ran, never the type (see
[Optogenetic experiments](concepts-optogenetics.md)). The type only says what should have
run.

A program that runs ClosedLoop Training before a ProgressiveRatio Paradigm does not fit the
shipped `ProgressiveRatio`. Build a type that expects both, with the Progressive Ratio
module: its Training then includes the ClosedLoop interval, as before.

## Control comparisons

A type may name what its reports lead with, before the general results. It may name both;
the Role comparison then comes first.

- a **Role comparison** — two [Roles](concepts-optogenetics.md#roles) from the program's
  linkage, such as Paired against Yoked, or Self against Unlit. A Role comes from the light,
  so the type must expect some. Paired against Yoked is the paired − yoked comparison moved
  to the front: it needs the Paired and yoked module (or Progressive Ratio, whose own
  paired − yoked difference it then is). Any other pair compares the two Roles' chambers
  within each treatment and window, with Welch's t-test;
- a **Treatment control** — one of the type's factors. The Design names the control level
  with `control_level:`, one of that factor's levels, and every other level is compared
  with it by Dunnett's test, the other factors held at the same levels. The Project design
  editor and the Config Editor show a **Control level** field for such a type. A Design of
  the type that names no control level fails to load, and so does a `control_level` under
  a type with no Treatment control.

```yaml
global:
  experiment_type: HedonicClosedLoop
  experimental_design_factors:
    Genotype: [w1118, Gr5a]
  control_level: w1118
```

A type with neither leads with the general results, as before.

## Where types live

- **Shipped** types are part of pyflic and are named, never copied.
- A lab's types are kept in its **type library**, a folder set in the editor. It is
  `~/.config/pyflic/experiment_types` by default; point it at a shared drive and everyone in
  the lab builds on the same types.
- The data carries its own **copy**: saving a Design (or a standalone config) whose type
  is a lab type writes `experiment_type.yaml` beside `project.yaml` (or beside
  `flic_config.yaml`). A Project therefore analyses the same on any machine, with or
  without the library. When the copy and the library's version differ, the Design editor
  and the Hub say so and offer to update the copy; until you do, the copy decides.

`pyflic lint` warns about a lab type the data carries no copy of, and about a copy that
differs from the library's.

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

`Hedonic` now means a recording without light. A Hedonic config whose recording has a
lit program fails to load: build a hedonic type that expects that light (for example one
expecting ClosedLoop, with the Hedonic module) and name it instead.

Run `pyflic lint` on a folder to see every config that needs either change.

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
is the workhorse for choice experiments. Every two-well type applies the
`max_med_duration_cutoff` and `max_events_cutoff` quality cutoffs when they are set; the
shipped Hedonic and Progressive Ratio types set them by default.

The details that matter here — `pi_direction`, counterbalancing, and dual-feeding
correction — are covered in
[Two-well choice and the preference index](concepts-two-well-pi.md).

### Hedonic

A two-well design analysed with attention to **how long** bouts last rather than how many
there are. The reasoning is that palatability shows up in bout duration: a fly presented
with something it finds more rewarding sustains longer feeding bouts, even when total lick
counts are similar. The shipped `Hedonic` type is this design without light; the Hedonic
module brings the same analysis to a lab's own type under any light.

Its distinctive output is the **weighted duration summary**, `analysis/treatment_means.csv`,
with one row per treatment and these columns for each well (ending in `A` or `B`):

- `MeanDuration` and `SemDuration` — the plain mean of the chambers' `MedDuration`, and
  its standard error
- `WeightedMeanDuration` and `WeightedSemDuration` — the mean of the chambers'
  `MedDuration` **weighted by each chamber's event count** at that well, and its weighted
  standard error
- `SampleSize` — the number of chambers that contributed: those with events at that well

The weighting is the point. A chamber whose median bout duration rests on 3 events is much
weaker evidence than one resting on 300, and an unweighted mean across chambers would
treat them as equal. Weighting by event count gives each chamber influence proportional to
how much it actually observed.

Hedonic experiments also get the **hedonic feeding plot**, `hedonic_feeding_plot.png`:
Well A's median bout durations beside Well B's, one point per chamber sized by its events,
with the mean and standard error, and each treatment in a panel of its own.

### Progressive Ratio

A two-well operant assay built on **chamber groups**, each lit by one light circuit and
sharing one treatment. In each group the **paired** fly's feeding at the sucrose well
(always well A) turns the light on. A group is a pair, whose **yoked** fly is lit at the
same moments whatever it does, or a single chamber with no yoked control. Per DFM, the
config names the paired chamber of each pair with `paired_chambers:`, or says
`chamber_groups: single`, or leaves both out and the linkage of `Program.txt` gives the
groups. The shipped type covers both ratio Paradigms, ProgressiveRatio and FixedRatio.
Other optogenetic designs — open loop, closed loop, fixed interval, non-feeding — are
analysed as a Custom experiment or under a lab type that
expects their light, with the [light table](concepts-optogenetics.md#the-light-table) beside
the feeding results. A yoked design among them gets a paired − yoked comparison of feeding
too, in the report's *Paired and yoked* results, under a type with the Paired and yoked
module, as Custom has.

What the Progressive Ratio module adds:

- **Training and Test facets**, split at each group's own training end, which the firmware
  marks in the data — so a Progressive Ratio config never states `facet_cutoffs`.
- **Paired − yoked differences** per pair, the primary input to its statistics when the
  design has yoked controls.
- A **light QC** that fails a group whose light followed the sensor rather than the fly,
  or met a Requirement its program's series does not give.
- A **breaking point** — the lick-backed light events the paired fly completed before its
  first long pause — and **sucrose persistence**, the same question asked of feeding.
- Its own constants (`require_training_complete`, `exclude_failed_pr_groups`, the light QC
  thresholds, `pr_break_gap_min`, `pr_test_window_min`), figures and report sections.

All of it is explained in [Progressive Ratio experiments](concepts-progressive-ratio.md).

## Changing type later

The experiment type is a property of the configuration, not the data. An Experiment
Directory holds exactly one `flic_config.yaml`, and its results always go to `analysis/`,
so analysing the same raw CSVs as a different type means a second Experiment Directory:
copy the `data/` folder beside a config that names the other type. Changing the type in
place overwrites the previous results the next time basic analysis runs.

A type that fixes the Chamber Layout fixes what a chamber *is* — which wells belong
together — so changing to a type with a different layout means every chamber assignment
has to be redone. The Config Editor names the assignments that would be lost and asks
first.

---

Related: [Experiment Type Editor](app-type-editor.md) ·
[Configuration file structure](config-structure.md) ·
[Progressive Ratio experiments](concepts-progressive-ratio.md) ·
[Plot catalogue](plots-catalog.md)
