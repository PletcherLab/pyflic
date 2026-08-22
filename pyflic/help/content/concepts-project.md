# Projects and replicates

pyflic has three structural levels, each recognised by a marker file and each looking only
at its *immediate* children:

| Level | Marker | Its children are |
|---|---|---|
| **Batch** | none (structural) | Projects |
| **Project** | `project.yaml` | Replicates |
| **Experiment Directory** | `flic_config.yaml` | — |

```
my_study/                     ← a Batch
  drug_trial/                 ← a Project
    project.yaml
    analysis/                 ← the Combined Analysis
    figures/                  ← publication figures
    plot_specs.yaml
    drug_trial_report.pdf
    rep1/                     ← a Replicate
      flic_config.yaml
      data/                   ← DFM CSVs
      analysis/
      qc/
    rep2/
```

An **Experiment Directory** is one FLIC recording. A **Replicate** is an Experiment
Directory inside a Project. If you are working on a single recording with no siblings, you
never need a Project at all — open the Experiment Directory directly.

## The design is an authority

The `design:` section of `project.yaml` owns **every** key under a replicate's `global:` —
the experiment type, the detection parameters, `well_names`, `transform_licks`, the
`constants:` cutoffs, the design factors, and the facet cutoffs.

A replicate that contradicts any of them **fails to load**. This is stricter than it might
seem necessary, and deliberately so: if `well_names` says well A is S5 in one replicate and
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

## Replicates inherit, they do not repeat

Because the design owns all of `global:`, a replicate's `flic_config.yaml` normally holds
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
replicate's config is no longer readable on its own; you need its parent to know what it
means.

## What stays free

- the whole `dfms:` block — DFM count, chamber → treatment assignment
- per-DFM `params:` overrides, but **only** the physical keys `pi_direction` and
  `chamber_sets`

`pi_direction` genuinely varies between DFMs of one recording — it says which side of the
chamber the reference well sits on. An override of an *analysis* key is rejected inside a
Project, because it would reintroduce the divergence the design outlaws, one level lower
and much harder to see.

## Adding a replicate

Drop a folder with a `data/` directory of DFM CSVs into the Project. It appears in the
Project panel as a scaffolding candidate; **Scaffold pending replicates** gives it a
config.

Scaffolding copies the `dfms:` block from the first existing replicate — replicates of one
design almost always reuse the plate layout, and retyping forty-eight chamber assignments is
the work worth avoiding — then reconciles it against the DFM ids actually in `data/`:

- an id in the data with no entry is **added**, chambers unassigned
- an entry with no data is **flagged**, never silently dropped, because a missing CSV is
  usually a copy that has not finished

An existing config is never overwritten.
