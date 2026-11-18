# Running many projects at once

A **Batch** is a directory with at least one Project **anywhere beneath it**. That is the
whole definition — nothing marks a Batch, and being one is structural: a folder becomes a
Batch by containing Projects and stops being one by not.

A **Batch Run** executes one designated Project Script in every checked Project.

```bash
pyflic batch my_study/                    # each Project's designated script
pyflic batch my_study/ "Report Pipeline"  # or the one you name
```

From the command line there is no review window: every Project found runs, and an
exclusion sheet at the batch folder is applied. The command exits with an error status when
any Project fails. In the Hub, the [Batch panel](app-hub.md#batch-panel) lists the
Projects and runs them.

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
short project list. Hidden folders (a name starting with `.`) are never entered. If the
folder you picked is enormous, the scan stops early and says so rather than hanging.

The list is read once when you choose the folder. Press **Rescan** after changing things
outside the app.

## Blocked members

A **Blocked Member** is a folder inside a Project that a run cannot use as it stands:

| What you see | What it means | Fix |
|---|---|---|
| `unfiled recording` | DFM CSVs sit at the folder's root, not in its `data/` | **File data…** in the review window, or **File unfiled recordings** on the Project panel |
| `no config` | `data/` holds DFM CSVs but there is no `flic_config.yaml` | double-click it in the review window, or **Member configs…** on the Project panel |
| `no recording` | there is a config but no DFM CSV the loader can find, or its `data_folder` does not exist or is a file | by hand |
| `ambiguous` | the same DFM id appears both loose and in `data/`, or a `data_folder` sits beside a recording of the member's own | by hand — decide which copy is the experiment |
| `duplicate` | it reads the same recording as a member before it by name (see [`data_folder`](config-structure.md#data-folder)) | by hand — decide which member reads it |
| `unreadable` | the folder cannot be listed | by hand — permissions |

Blocked belongs to the **member**, never to the Project. A Project with four healthy
members and one blocked one runs the four, and a run is **never refused** because of one —
a stale folder must not stop ten Projects at 2am. A Project with *nothing* usable starts
unchecked, because it can only produce a failure; it is checked again the moment a repair
in the review window makes it runnable.

## The review window

**Run Batch** always opens a review window first, **Batch Run — review**. Right-clicking a
row of the Batch panel's table and choosing **Fix blocked members in …** opens the same
window on that Project. It is shown even when nothing is wrong: with a recursive scan, the
folder you picked no longer tells you what will run, and this list is the only place that
does.

Nothing in it is a gate. Repair what you like, uncheck what you like, then **Run batch** or
**Cancel**.

### The project list

The line at the top says how many Projects and members will run, and how many members are
blocked. Below it, one row per Project found:

- **Project** — its path inside the batch folder, with a checkbox. The checks start as the
  Batch panel's; uncheck any Project you do not want to run.
- **Members** — how many of its members the run can use, out of how many it has (`3/5`).
- **Status** — `ok`, `no members`, or how many members are blocked. Each blocked member is
  listed under its Project with its reason; hover it for the detail.

### Repairing blocked members

- **File data…** files the selected member's recording.
- **File every unfiled recording** files every unfiled member, except in a Project you
  unchecked. It asks first.
- **Double-click** a blocked member to repair it: an unfiled one is filed, and one with no
  config opens its Project's **Member configs…** window. The others have to be fixed by
  hand, and the window says so.
- **Rescan** walks the batch folder again, for changes made outside the app.

**Filing** moves `DFM*.csv`, `Program.txt` and `Messages.csv` into `data/` and every
other loose file into `extra_files/`.
Every `.yaml` file and any `remove_chambers.csv` stays exactly where it is — at a member
root those are configuration and declaration, never data. Nothing is ever overwritten: a
name that already exists is skipped and reported.

### The exclusion sheet switch

When the batch folder holds an exclusion sheet, **Apply the exclusion sheet before running**
is ticked, with a line counting what its rows would do; hover the line for each row. Untick
it to run without the sheet this time: the sheet and every member's own declarations are
left as they are. A sheet that cannot be read is reported there, and the switch is off. See
[Exclusion sheets](#exclusion-sheets).

## Which script runs

The Batch panel's **Project Script** menu picks the designated script: each Project's own
`batch` script (the default), one of the built-in pipelines, or one of `batch.yaml`'s
`project_scripts:`. The choice is saved as `batch.yaml`'s `script:`.

A designated name is resolved per Project, in this order:

1. the Batch's central `project_scripts:` section, in `batch.yaml`
2. the Project's own script of that name, in its `project.yaml`
3. the built-in pipelines

The central copy wins, so every Project runs the same recipe; a Project's own script of
that name is used only when `batch.yaml` has none. A Project where the name resolves
nowhere is reported and skipped.

With **no designation at all** — the default — each Project runs its **own** script named
`batch`, or, when it has none by that name, the first script in its `scripts:`. Every new
`project.yaml` is seeded with a `batch` script that analyzes every member, builds the
Combined Analysis, builds the Project Report and renders the figures (see
[What a script is](scripts-overview.md#the-batch-script-every-project-starts-with)), so zero
authoring means "analyze and report every Project". A Project whose `scripts:` section is
empty is reported and skipped rather than having something substituted for it.

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
      - action: build_combined_analysis
      - action: project_report
```

Unlike a Project, a Batch has no authority to declare — there is no `design:` here.

## Exclusion sheets

If the batch folder holds a `remove_chambers.csv`, a Batch Run applies it before anything
else — writing its rows into each member's own declaration. Only the rows for the checked
Projects are written; the others are skipped and counted. The review window previews
exactly what it would do and lets you decline it for this run. **Apply exclusion sheet…**
on the Batch panel applies it now, to the checked Projects. See
[Excluding chambers in bulk](concepts-exclusions.md).

## Errors

Continue-on-error, with per-Project log prefixes. One bad Project never aborts an
unattended overnight run. Within a Project, its Project Script stops at its first failing
step, and the run moves on to the next Project.

The run opens by listing every Project with how many of its members it can use
(`3/5 members, 1 blocked`), so "succeeded" can never be read as "analysed everything". It
ends with how many Projects succeeded and the first line of each failure. Warnings and
failures are also copied to the **Errors** tab, which badges itself while unread — in a run
of several thousand lines, the four that matter are otherwise unfindable.

## AI narrative of the batch

Tick **AI narrative of the batch** on the Batch panel, and after the run an AI provider
writes `batch_ai_narrative.md` at the batch folder: one synthesis of the Projects that
succeeded — where they agree and disagree, design problems, and Projects that lost many
chambers. It reads the narrative each Project already has (`analysis/ai_summary.txt`, which
the `generate_ai_narrative` step writes), so a Project with none is left out. The provider
is the one picked in the AI panel, checked before the run starts: with no API key, the run
does not start. A narrative that fails does not fail the run. See [AI summary](concepts-ai-summary.md).

## Too many tabs

A Batch Run touches every member of every Project, so the figure tabs it would open run
into the hundreds. **Suppress new plot / output tabs** (in the Batch panel, on by default)
stops new tabs being created during a Batch Run. Every artifact is still written to
disk, and the Output and Errors tabs keep streaming — only the tabs are skipped. The
switch governs Batch Runs only: a project or experiment analysis you run by hand always
shows its plots.

## Migrating from the old batch modes

Both of the old modes are retired:

- **Subdir-batch mode** (the recursive walk keyed on a script named `batch`) — replaced by
  the Batch level described here. Experiment Scripts still named `batch` no longer have any
  special meaning; `pyflic lint` reports them so you can rename them.
- **YAML-batch mode** (running one script across every YAML in a directory) — retired with
  multi-YAML directories. An Experiment Directory now holds exactly one `flic_config.yaml`.

Run `pyflic lint` on an old folder to see exactly what needs changing.
