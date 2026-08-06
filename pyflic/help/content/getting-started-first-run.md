# Running your first analysis

With a project directory and a configuration file in place, you are ready to produce
results. Start with the graphical hub — it is the fastest way to see whether your data
loaded correctly.

## Open the hub

```bash
pyflic hub my_experiment/
```

You can also run `pyflic hub` with no argument and choose the folder from the **Project**
card.

## Load, then analyse

1. **Load.** In the **Load** card, click **Load**. pyflic reads every CSV in `data/`,
   subtracts the baseline, and runs bout detection. The output panel reports each DFM as
   it finishes. This is the slow step; later steps reuse the result.
2. **Analyze.** In the **Analyze** card, click **Basic analysis**. This runs the standard
   pipeline in one go — quality-control reports, a text summary, a feeding-summary table,
   and the feeding-summary plot.

If loading fails, the message in the output panel usually names the problem directly: a
DFM in the configuration with no matching CSV, or a chamber number outside the range your
`chamber_size` allows. [Troubleshooting](troubleshooting.md) covers the common ones.

## Where the results go

Basic analysis writes four things into your project directory:

| Output | Path | What it is |
|---|---|---|
| QC reports | `<config>_results/qc/` | Per-DFM signal plots and integrity checks |
| Text summary | `<config>_results/analysis/summary.txt` | Human-readable overview of the run |
| Feeding summary | `<config>_results/analysis/feeding_summary.csv` | One row per chamber: licks, events, durations, intervals |
| Summary plot | `<config>_results/analysis/feeding_summary.png` | The same data by treatment |

`<config>` is the stem of your configuration file, so `flic_config.yaml` gives you
`flic_config_results/`.

## Look at the QC output before you believe anything

This is the step that is easiest to skip and most expensive to skip. Open the QC viewer:

```bash
pyflic qc my_experiment/
```

It shows you the raw and baselined signal per well, with detected events marked. What you
are checking is whether the events pyflic found are the events *you* would have called by
eye. If detection looks too eager or too conservative, that is a parameter question, not
a data question — see [How feeding is detected](concepts-licks-events.md) and
[Parameter reference](reference-parameters.md).

The QC viewer can recompute with different parameters live, so you can try a value before
committing it to your configuration file. See [QC Viewer](app-qc-viewer.md).

## From here

- Draw more figures from the **Plots** card — [Plot catalogue](plots-catalog.md).
- Stop clicking the same buttons every time by defining a script —
  [What a script is](scripts-overview.md).
- Understand the numbers you just produced — [Summary metrics](concepts-metrics.md).
