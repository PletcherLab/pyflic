# Running many projects at once

A **Batch** is a directory with at least one Project **anywhere beneath it**. That is the
whole definition — nothing marks a Batch, and being one is structural: a folder becomes a
Batch by containing Projects and stops being one by not.

A **Batch Run** executes one designated Project Script in every checked Project.

```bash
pyflic batch my_study/
```

## What a Batch is not

- It is **not a Project**. It holds no analysis of its own and never pools results across
  Projects — each Project has its own design, and there is no cross-Project analysis. Its
  only product is a per-Project run summary.
- It has **no script level of its own**. What a Batch Run executes IS a Project Script.
  There is no third registry to learn.

## Where Projects are found

The scan is **recursive**, and it **stops at each Project**. It walks down until it finds
one — a `project.yaml` plus at least one member the run could actually use — and never
looks inside, because a Project's subdirectories are its members by definition.

That has three consequences worth knowing:

- **Grouping folders are transparent.** `Sept2026/ProjA` and `Archive/2025/ProjC` are both
  found from one batch folder. You do not have to flatten your tree for the tool.
- **Nothing runs twice.** An archived copy carrying its own `project.yaml` *inside* a
  Project cannot become a second target.
- **A project is named by its path.** The Batch table shows `Sept2026/ProjA`, not `ProjA`.
  A Project directly under the batch folder still shows its bare name, so anything you
  wrote before — a `batch.yaml` designation, an exclusion sheet row — still resolves.

A folder that looks like it was meant to be a Project but is not — a `project.yaml` with no
member directory, an unreadable folder, a symlink — is skipped and **reported in the
Output tab**, so a forgotten file shows up at the top of the run rather than as a quietly
short project list. If the folder you picked is enormous, the scan stops early and says so
rather than hanging.

The list is read once when you choose the folder. Press **Rescan** after changing things
outside the app.

## Blocked members

A **Blocked Member** is a folder inside a Project that a run cannot use as it stands:

| What you see | What it means | Fix |
|---|---|---|
| `unfiled recording` | DFM CSVs sit at the folder's root, not in its `data/` | **File data…** |
| `no config` | `data/` holds DFM CSVs but there is no `flic_config.yaml` | **Member configs…** |
| `no recording` | there is a config but no DFM CSV the loader can find | by hand |
| `ambiguous` | the same DFM id appears both loose and in `data/` | by hand — decide which copy is the experiment |
| `unreadable` | the folder cannot be listed | by hand — permissions |

Blocked belongs to the **member**, never to the Project. A Project with four healthy
members and one blocked one runs the four, and a run is **never refused** because of one —
a stale folder must not stop ten Projects at 2am. A Project with *nothing* usable starts
unchecked, because it can only produce a failure.

## The review window

**Run Batch** always opens a review window first. It lists every Project that was found,
with its path, how many of its members the run can use, and each blocked member with the
reason and the button that clears it. Uncheck any Project you do not want to run.

It is shown even when nothing is wrong: with a recursive scan, the folder you picked no
longer tells you what will run, and this list is the only place that does.

Nothing in it is a gate. Repair what you like, uncheck what you like, then Run or Cancel.

**Filing** moves `DFM*.csv` into `data/` and every other loose file into `extra_files/`.
Every `.yaml` file and any `remove_chambers.csv` stays exactly where it is — at a member
root those are configuration and declaration, never data. Nothing is ever overwritten: a
name that already exists is skipped and reported.

## Which script runs

Resolution order, per Project:

1. the Project's own script of that name, in its `project.yaml`
2. the Batch's central `project_scripts:` section
3. the built-in pipelines

The Project's own copy wins, so a Project can specialise the designated run without the
Batch knowing.

With **no designation at all** — the default — each Project runs its **own** script named
`batch`. Every new `project.yaml` is seeded with one, holding the Report Pipeline's steps,
so zero authoring means "create a report on every Project". A Project whose `scripts:`
section is empty is reported and skipped rather than having something substituted for it.

Only the **selected** batch folder's `batch.yaml` governs. A grouping folder deeper in the
tree may be a Batch in its own right and carry its own designation; it is ignored, and the
run log says so.

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

## Exclusion sheets

If the batch folder holds a `remove_chambers.csv`, a Batch Run applies it before anything
else — writing its rows into each member's own declaration. The review window previews
exactly what it would do and lets you decline it for this run. See
[Excluding chambers in bulk](concepts-exclusions.md).

## Errors

Continue-on-error, with per-Project log prefixes. One bad Project never aborts an
unattended overnight run.

The summary at the end names every Project and how many of its members ran (`3/5`), so
"succeeded" can never be read as "analysed everything". Warnings and failures are also
copied to the **Errors** tab, which badges itself while unread — in a run of several
thousand lines, the four that matter are otherwise unfindable.

## Too many tabs

A Batch Run touches every member of every Project, so the figure tabs it would open run
into the hundreds. **Suppress new plot / output tabs** (in the Batch panel, on by default)
stops new tabs being created. Every artifact is still written to disk, and the Output and
Errors tabs keep streaming — only the tabs are skipped. The switch applies to every run
while it is checked, not just Batch Runs.

## Migrating from the old batch modes

Both of the old modes are retired:

- **Subdir-batch mode** (the recursive walk keyed on a script named `batch`) — replaced by
  the Batch level described here. Experiment Scripts still named `batch` no longer have any
  special meaning; `pyflic lint` reports them so you can rename them.
- **YAML-batch mode** (running one script across every YAML in a directory) — retired with
  multi-YAML directories. An Experiment Directory now holds exactly one `flic_config.yaml`.

Run `pyflic lint` on an old folder to see exactly what needs changing.
