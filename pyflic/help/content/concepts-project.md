# Projects and members

pyflic has three structural levels, each recognised by a marker file:

| Level | Marker | Its children are | Found by |
|---|---|---|---|
| **Batch** | none (structural) | Projects | a recursive walk that stops at each Project |
| **Project** | `project.yaml` | Members | its *immediate* subdirectories |
| **Experiment Directory** | `flic_config.yaml` | — | — |

```
my_study/                     ← a Batch
  drug_trial/                 ← a Project
    project.yaml
    analysis/                 ← the Combined Analysis
    figures/                  ← publication figures
    plot_specs.yaml
    drug_trial_report.pdf
    rep1/                     ← a Member
      flic_config.yaml
      data/                   ← DFM CSVs
      analysis/
      qc/
    rep2/
```

A Project's members may sit at any depth *below the batch folder*, because grouping
folders like `Sept2026/` are transparent to the scan — but a Project's own members are its
immediate children and nothing deeper. See
[Running many projects at once](scripts-batch.md).

An **Experiment Directory** is one FLIC recording. A **Member** is an Experiment
Directory inside a Project. If you are working on a single recording with no siblings, you
never need a Project at all — open the Experiment Directory directly.

## Members are not replicates

A Project's members are **different experiments that address one question** — a dose
series, a genotype panel, a pilot beside its follow-up — not repeats of one another. That
is the difference from the sister application PyTrackingAnalysis, whose projects hold
replicates.

It matters in two places. The Design still forces one detection rule across every member,
because comparing them otherwise would be meaningless. But the Combined Analysis keeps
`Experiment` as a column and the mixed model nests DFM *within* Experiment, because the
rows are not interchangeable: DFM 1 in one member is a different physical device from DFM 1
in another, and the members themselves are different experiments.

## The design is an authority

The `design:` section of `project.yaml` owns **every** key under a member's `global:` —
the experiment type, the detection parameters, `well_names`, `transform_licks`, the
`constants:` cutoffs, the design factors, and the facet cutoffs.

A member that contradicts any of them **fails to load**. This is stricter than it might
seem necessary, and deliberately so: if `well_names` says well A is S5 in one member and
S5Y5 in another, a pooled licks figure is not noisy, it is *wrong*, and nothing in the
output would show it.

```yaml
# project.yaml
name: drug_trial
design:
  global:
    experiment_type: Hedonic
    transform_licks: false
    facet_cutoffs: [60]
    params:
      feeding_threshold: 20
      samples_per_second: 5
    well_names: {A: S5, B: S5Y5}
    experimental_design_factors:
      TreatmentNew: [Ctrl, Exp]
```

## Members inherit, they do not repeat

Because the design owns all of `global:`, a member's `flic_config.yaml` normally holds
only the parts that genuinely differ — its `dfms:` block and its own `scripts:`:

```yaml
# rep1/flic_config.yaml
dfms:
  - id: 1
    params: {pi_direction: left}
    chambers: {1: Ctrl, 2: Ctrl, 3: Exp}
```

If a `global:` block *is* present — typically a standalone experiment moved into a Project —
it is validated key by key, and any deviation is an error. The trade-off is that a
member's config is no longer readable on its own; you need its parent to know what it
means.

## What stays free

- the whole `dfms:` block — DFM count, chamber → treatment assignment
- per-DFM `params:` overrides, but **only** the physical keys `pi_direction` and
  `chamber_sets`

`pi_direction` genuinely varies between DFMs of one recording — it says which side of the
chamber the reference well sits on. An override of an *analysis* key is rejected inside a
Project, because it would reintroduce the divergence the design outlaws, one level lower
and much harder to see.

## Blocked members

A folder inside a Project can hold a perfectly good recording and still be unusable. The
loader reads `experiment_dir/data` and nothing else, so DFM CSVs sitting at the folder's
root are invisible — not broken, not warned about, simply absent from the Project.

The Project panel now lists these as **blocked members**, in red, with the reason in the
tooltip:

| Reason | Fix |
|---|---|
| `unfiled recording` — DFM CSVs at the root, not in `data/` | **File unfiled recordings** |
| `no config` — `data/` holds DFM CSVs but there is no `flic_config.yaml` | **Member configs…** |
| `no recording` — a config, but no DFM CSV the loader can find | by hand |
| `ambiguous` — the same DFM id both loose and in `data/` | by hand: decide which copy is the experiment |

Blocked belongs to the **member**, never to the Project: a Project with four healthy
members and one blocked one analyses the four.

**Filing** moves `DFM*.csv` into `data/` and every other loose file into `extra_files/`.
Every `.yaml` file and any `remove_chambers.csv` stays exactly where it is — at a member
root those are configuration and declaration, never data. Nothing is ever overwritten.

## Adding a member

Drop a folder with a `data/` directory of DFM CSVs into the Project. It appears in the
Project panel as a blocked member with `no config`; **Member configs…** gives it one.

Scaffolding copies the `dfms:` block from the first existing member — members of one
design almost always reuse the plate layout, and retyping forty-eight chamber assignments is
the work worth avoiding — then reconciles it against the DFM ids actually in `data/`:

- an id in the data with no entry is **added**, chambers unassigned
- an entry with no data is **flagged**, never silently dropped, because a missing CSV is
  usually a copy that has not finished

An existing config is never overwritten.

If the folder's DFM CSVs are still loose at its root, file them first — scaffolding
reconciles against what is in `data/`, and doing it beforehand would write a config
reconciled against nothing.
