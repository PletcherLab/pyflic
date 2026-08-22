# Running many projects at once

A **Batch** is a directory whose *immediate* subdirectories holding a `project.yaml` are
its Projects. That is the whole definition — nothing marks a Batch, and being one is
structural: a folder becomes a Batch by containing Projects and stops being one by not.

A **Batch Run** executes one designated Project Script in every Project.

```bash
pyflic batch my_study/
```

## What a Batch is not

- It is **not a Project**. It holds no analysis of its own and never pools results across
  Projects — each Project has its own design, and there is no cross-Project analysis. Its
  only product is a per-Project run summary.
- It has **no script level of its own**. What a Batch Run executes IS a Project Script.
  There is no third registry to learn.

## Immediate children only

The scan looks one level down and no further. If your tree is deeper than
Batch → Project → Experiment, run a Batch per parent.

This is a deliberate change from older versions, which walked the whole tree looking for a
script literally named `batch`. That recursion existed because pyflic had no structural
level for "study → cohort → experiment"; Projects supply it now, so unbounded recursion has
no remaining job — and it invited running one Project twice from two different ancestors.

A child that is not a Project is skipped and reported, so a forgotten `project.yaml` shows
up at the top of the run rather than as a quietly short project list.

## Which script runs

Resolution order, per Project:

1. the Project's own script of that name, in its `project.yaml`
2. the Batch's central `project_scripts:` section
3. the built-in pipelines

The Project's own copy wins, so a Project can specialise the designated run without the
Batch knowing.

By default the name is `batch` — every new `project.yaml` is seeded with a Project Script
under that name, holding the Report Pipeline's steps. So zero authoring means "create a
report on every Project".

## batch.yaml

Optional, and it appears only once you want one of two things:

```yaml
script: Report Pipeline        # which Project Script a Batch Run executes
project_scripts:               # one recipe serving every Project, not copied
  - name: nightly
    steps:
      - action: run_in_experiments
        script: standard
      - action: project_report
```

Unlike a Project, a Batch has no authority to declare — there is no `design:` here.

## Errors

Continue-on-error, with per-Project log prefixes. One bad Project never aborts an
unattended overnight run; the summary at the end says how many succeeded.

## Migrating from the old batch modes

Both of the old modes are retired:

- **Subdir-batch mode** (the recursive walk keyed on a script named `batch`) — replaced by
  the Batch level described here. Experiment Scripts still named `batch` no longer have any
  special meaning; `pyflic lint` reports them so you can rename them.
- **YAML-batch mode** (running one script across every YAML in a directory) — retired with
  multi-YAML directories. An Experiment Directory now holds exactly one `flic_config.yaml`.

Run `pyflic lint` on an old folder to see exactly what needs changing.
