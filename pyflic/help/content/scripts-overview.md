# What a script is

A **script** is a named recipe: a list of steps under one name, run from the Hub. Instead
of clicking Load, then Basic analysis, then three plots every time you revisit an
experiment, you define that sequence once and run it in a click.

Scripts also make an analysis reproducible. The sequence lives in the config alongside the
parameters that produced the results, so re-running months later gives the same output
without anyone remembering which buttons were pressed in which order.

## Two levels

There are two script levels with **separate action registries**. They cannot mix: a
project-level step in an Experiment Script is an error, and vice versa.

| Level | Lives in | Acts on | Run from |
|---|---|---|---|
| **Experiment Script** | a member's `flic_config.yaml` `scripts:`, or a Project's `experiment_scripts:` | one loaded member | the **Scripts** panel |
| **Project Script** | `project.yaml` `scripts:` | the Project as a whole | the **Project** panel |

Each level is run from the panel of the thing it acts on, so the level a script belongs to
is never a question of which combo box you happened to click. The Scripts tile is dimmed
until a member is loaded, for the same reason Analyze and Plots are: with nothing loaded
there is nothing for an Experiment Script to run on. Project Scripts, needing no member,
stay available the moment a Project is open.

There is no third level. What a [Batch Run](scripts-batch.md) executes IS a Project Script.

The only bridge between the two is the `run_in_experiments` project action, which runs a
named Experiment Script in every member — or, with its optional `only:` list, in just
some of them.

```yaml
# project.yaml
experiment_scripts:              # one recipe, serving every member
  - name: standard
    steps:
      - action: load
      - action: basic_analysis
      - action: binned_csv
        binsize: 30

scripts:                         # Project Scripts
  - name: batch
    steps:
      - action: run_in_experiments
        script: standard
      - action: project_report
      - action: render_publication_figures
```

Putting a recipe in `experiment_scripts:` means one copy serves every member instead of
being pasted into each of their configs. A member's own `scripts:` is the fallback when
the name is not found centrally.

### Built-in Project Scripts

Two pipelines every Project can run without authoring anything. Neither is written to
`project.yaml`, so they track the shipped default:

- **Standard Pipeline** — validate design → project report → render figures
- **Report Pipeline** — project report → render figures

A Batch Run prefers the Report Pipeline's steps because they do not gate on
`validate_design`, which would fail Projects mid-migration. That is what every new
`project.yaml` is seeded with, under the name `batch`.

## Defining an Experiment Script

Scripts live under a `scripts:` key:

```yaml
scripts:
  - name: "Standard Analysis"
    steps:
      - action: load
        start: 0
        end: 240
      - action: remove_chambers
      - action: write_summary
      - action: basic_analysis
      - action: feeding_csv
      - action: binned_csv
        binsize: 30
      - action: plot_feeding_summary
      - action: plot_binned
        metric: Licks
        mode: total
        binsize: 30
      - action: pdf_report

  - name: "Quick Overview"
    steps:
      - action: load
      - action: basic_analysis
      - action: plot_feeding_summary
```

Each script needs a `name` and a `steps` list. Each step needs an `action`; everything
else is optional. Define as many scripts as you like — each appears in the Scripts panel's
dropdown.

You do not have to write this by hand. The [Script Editor](scripts-editor.md) builds it
visually.

## Start with `load`

Every script should begin with a `load` step. It reads the CSVs, applies the time window,
and prepares the experiment for every step that follows.

```yaml
- action: load
  start: 0        # minutes; omit to inherit from the hub
  end: 240        # 0 means "through the end of the recording"
  parallel: true  # load DFMs concurrently
```

If a script has no `load` step, the hub loads with its current settings before the first
step runs. Relying on that makes the script's behaviour depend on what the hub happened to
be set to, which is exactly the reproducibility you were trying to buy. Be explicit.

## Follow it with `remove_chambers` and `write_summary`

```yaml
- action: remove_chambers
  group: "Standard Analysis"   # optional; defaults to the script's name
- action: write_summary
```

`remove_chambers` applies exclusions from `remove_chambers.csv`. **When `group` is omitted
it defaults to the script's name**, which is what lets one CSV serve several scripts with
different exclusion sets. It also means a script name that does not match any group in the
CSV applies no exclusions at all, silently — so if exclusions seem not to be taking
effect, check that spelling first.

`write_summary` records which chambers were excluded and what the experiment looked like
afterwards. Putting it immediately after exclusions means your summary describes the data
your results were actually computed from.

## Where values come from

For each step, a parameter is resolved in this order:

1. **The step itself** — `binsize: 30` written on that step
2. **The script level** — `start:` / `end:` set on the script
3. **The hub** — whatever the corresponding control currently shows

So a step-level value always wins, and anything you leave blank falls back to the hub. That
is deliberate: it lets one script be re-run over different time windows by changing a
spinbox, while the parameters you *did* pin stay pinned.

## Time windows

`start` and `end` are minutes, and `end: 0` means "through the end of the recording".
Outputs always go to `analysis/`, whatever the window, so a second run over a different
window overwrites the first. To compare phases of one recording, declare
[Facets](concepts-facets.md) instead: every phase lands in one table, side by side.

## Running them

- **Run** on the Scripts panel — the selected Experiment Script, on the loaded member
- **Run script** on the Project panel's Analysis card — the selected Project Script
- **Run Batch** on the Batch panel — one Project Script in every checked Project; see
  [Running many projects at once](scripts-batch.md)

Progressive Ratio members have their own actions — the difference table, the light QC,
the breaking point and their figures; see
[Script actions](scripts-actions.md#analyse-actions). A step whose action needs a different
Experiment Type is skipped with a log line, not an error.

---

Next: **[Script actions](scripts-actions.md)** — every action and its parameters.
