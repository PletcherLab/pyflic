# Facets: time windows as a column

A **facet** is a named time window within a recording. Facets are how pyflic splits a run
into phases — acclimation, test, recovery — without splitting the output into directories.

## Windows are a column, not a directory

Set `facet_cutoffs` and analysis writes two files into `analysis/`:

- `feeding_summary.csv` — one row per chamber, whole recording
- `feeding_summary_facet.csv` — one row per chamber **per facet**, with `Facet` and
  `FacetRange` columns

```yaml
global:
  facet_cutoffs: [10, 70]
```

gives three facets: `0-10 min`, `10-70 min`, `70+ min`.

Older versions wrote windowed results into `analysis_0_360/` directories, so an output path
was a function of the window you happened to ask for, and pooling could not work — a
Combined Analysis stacking members would have to *guess* which of a member's window
directories was the one to pool. Facets cover the windows of *one* analysis. When a part of
the recording needs to be analysed on its own, with its own excluded chambers, declare a
[Slice](concepts-slices.md): a named, separate analysis that every member runs and the
Project pools by name. Facets then split each Slice, each window cut to the Slice and one
left empty dropped.

## Half-open windows

Every window is `(start, end]` — exclusive at the start, inclusive at the end. That is what
makes a faceted analysis a true partition: a sample landing exactly on a cutoff belongs to
the phase before it and to nothing else. With inclusive bounds at both ends it would be
counted in both adjacent phases and inflate them.

The last facet always runs to the end of the recording, so nothing is lost at the tail.

## Named phases

An Experiment Type can name its phases. Progressive Ratio's Facets are `Training` and
`Test` — and for that type the boundary is not a minute in the config but each chamber
group's own training end, read from the data. Its config never states `facet_cutoffs` or
`facet_source`; the `Facet` column still carries one label per row, so every plot and
statistic that reads Facets works unchanged. See
[Experiment types](concepts-experiment-types.md).

For every other type, those names apply **only** while the cutoffs are the type's default. Move a cutoff and you
get plain minute-range labels instead — "Training" would be a lie once you have moved the
boundary that defined it. Set `facet_labels` to name them yourself, one name per window:

```yaml
global:
  facet_cutoffs: [15]
  facet_labels: [Early, Late]
```

A list whose length does not match the windows is not used: the Facets keep the type's
names or the minute ranges.

## Program Facets

An optogenetic recording can take its Facets from its program instead of from minutes. Set
`facet_source` in the design:

```yaml
global:
  facet_source: program
```

| `facet_source` | Facets |
|---|---|
| `cutoffs` (the default) | `facet_cutoffs`: the same minute windows on every DFM |
| `program` | one per **Program Interval** of each DFM's program, its Cycles pooled |
| `program_cycles` | one per Program Interval and Cycle |

Every metric is then split by what the program ran: the feeding summary, the
[light table](concepts-optogenetics.md#the-light-table), the reports and the faceted
figures.

**Labels.** A Facet is named after its Paradigm, such as `Closed` or `PR10`. When any DFM
runs the same Paradigm in two Program Intervals, that Paradigm's labels add the interval,
`I<k>`, on every DFM: a program of open-loop light on, then off, gives `I1 Open` and
`I2 Open`. Under `program_cycles` the
label adds the Cycle, `C<n>`, as in `PR10 C1`. A label depends on the Paradigm and the
interval only, never on the DFM, so pooling a Facet across DFMs, or across the members of a
Project, pools chambers that ran the same Paradigm.

**Pooled Cycles.** A Repeating program runs its intervals again and again. Under `program`
each interval's Cycles form one Facet of several separate windows, and `FacetRange` lists
them all. `program_cycles` keeps each pass apart.

**Each DFM has its own windows.** Two DFMs running different programs have different Facets.
A Facet a DFM's program never ran has no rows for that DFM. The Facet list is every label
any DFM has, in the order they began.

**What belongs to no Facet.** The baseline before the program's first instruction reached
the DFM, and rows recorded under an instruction outside the program. Whole-recording tables
still include them.

**Exact or approximate.** The recording states its Program Interval and Cycle on every
row, so its Facets are exact. An older recording (generation 1 or 2) does not, so its
Facets are laid out from the program's clock schedule. `summary.txt` notes each such DFM as
*Program Facets from the clock schedule: approximate*, and the report's cover says so.

**Errors.** `pyflic lint` reports each of these, and the loader refuses the config:

- `facet_source: program` or `program_cycles` without a `data/Program.txt`, or with a
  `Program.txt` that has no section for a DFM in the config;
- `facet_cutoffs` stated beside `facet_source: program`: the two cannot both decide the
  Facets;
- a `facet_source` that is not one of the three values;
- `facet_source` under the Progressive Ratio type, which owns its Facets.

`facet_labels` is ignored under Program Facets, with a warning: the Paradigm names the
Facets.

The Project Design dialog and the Config Editor have a **Facet source** picker. See
[Optogenetic experiments](concepts-optogenetics.md#program-facets).

## Facets in a Project

`facet_cutoffs` (with `facet_labels`) and `facet_source` sit in the design's `global:` block
— the Project Design dialog's **Facet source**, **Facet cutoffs** and **Phase names** fields
— and are part of the design, so every member in a Project is windowed by the same rule by
construction. There is nothing to reconcile at pooling time — which is the whole point.
Program Facets pool by label: every member's `PR10` Facet is one Facet in the Combined
Analysis.

Facets are also what make the faceted publication figures possible: a plot with one panel
per phase has no meaning while windows are directories. See
[Plot Editor](app-plot-editor.md).
