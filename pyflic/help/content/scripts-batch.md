# Running many projects at once

pyflic has two batch modes, and they do different things. Say which one you mean — "batch"
alone is ambiguous.

| Mode | Runs across | Toggle |
|---|---|---|
| **Subdir-batch mode** | many **project directories** under a chosen root | *Run 'batch' script in every directory under here* |
| **YAML-batch mode** | many **configurations** in one project directory | *Run action for every YAML config* |

The two are mutually exclusive; enabling one disables the other.

## Subdir-batch mode

Use this when you have many experiments and want to run the same pipeline over all of
them.

### What a batch target is

A **batch target** is a directory containing at least one YAML that defines a script named
`batch`. That is the entire definition — it has nothing to do with depth.

Turn the toggle on and click **Run Script**, and pyflic walks the **whole tree** beneath
the folder you chose, running the `batch` script in every batch target it finds.

### The walk

- **Unbounded depth.** Real research trees are deeper than one level — study → cohort →
  experiment — and the walk descends all of it. This is not a scan of immediate children.
- **The root counts.** If the folder you chose itself contains a YAML defining `batch`, it
  runs alongside its descendants.
- **Every qualifier runs.** A parent and its batch-bearing descendants all run; there is no
  leaf-only filter and no parent suppression. `batch` scripts at several levels are treated
  as intentional aggregation.
- **Several YAMLs, several runs.** Each YAML in a directory that defines `batch`
  contributes its own run, so a directory with three such configurations runs three times.
- **The name is case-insensitive.** `batch`, `Batch` and `BATCH` all qualify.

These directories are pruned from the walk and never searched:

```
any name starting with "."      analysis/    plots/    qc/
__pycache__/                    node_modules/
```

Symbolic links are not followed.

### What gets skipped, and whether you hear about it

- A directory with **no YAML files** is skipped silently.
- A directory with YAMLs but **no `batch` script** is logged as a **near-miss skip**.

That distinction exists for one reason: near-misses are almost always a forgotten script,
not a deliberate exclusion. If you expected a folder to run and it did not, look for it in
the near-miss log before looking anywhere else.

### Each target is its own project

A batch target is treated as an independent project directory, so its outputs land inside
it. Nothing is written to the root you started from, and targets cannot overwrite each
other. Run banners and figure titles label each target by its path relative to the root, so
nested targets are unambiguous; the root itself labels as `./<yaml name>`.

**Outputs are always overwritten.** There is no skip-if-results-exist short-circuit — a
re-run regenerates everything. The input-keyed cache in `.pyflic_cache/` is left alone and
stays correct by construction.

### Setting it up

Give the pipeline script the name `batch` in each experiment's configuration:

```yaml
scripts:
  - name: "batch"
    steps:
      - action: load
        start: 0
        end: 0
        parallel: true
      - action: remove_chambers
      - action: basic_analysis
      - action: feeding_csv
      - action: pdf_report
```

Then point the hub at the common parent, tick the toggle, and click **Run Script**. The
button label shows how many targets were found, so check that number matches your
expectation before you start a long run.

Remember that a bare `remove_chambers` step uses the **script name** as its exclusion
group — so in a `batch` script it looks for a group called `batch` in each project's
`remove_chambers.csv`. Name the group accordingly, or set `group:` explicitly.

In this mode the Config dropdown and Script picker are ignored; each target uses its own
YAML and its own `batch` script.

## YAML-batch mode

Use this when one project directory holds several configurations of the *same* raw data —
different parameters, different time windows, different experiment types — and you want to
run one action across all of them.

Tick *Run action for every YAML config* and the chosen action runs once per configuration,
each writing into its own `<config>_results/` folder. Nothing collides, because output
paths are namespaced by configuration name.

This is the mode for parameter comparisons: put `link_gap_1.yaml`, `link_gap_5.yaml` and
`link_gap_20.yaml` in one project and analyse all three in a click.

---

Related: [What a script is](scripts-overview.md) · [Analysis Hub](app-hub.md)
