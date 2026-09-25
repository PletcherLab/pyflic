# Running your first analysis

With an experiment directory and a configuration file in place, you are ready to produce
results. Start with the graphical Hub — it is the fastest way to see whether your data
loaded correctly.

## Open the Hub

```bash
pyflic hub my_experiment/
```

Pointed at an Experiment Directory, the Hub loads it straight away; the Output tab reports
each DFM as it finishes. This is the slow step, and every later step reuses the result.

Inside a Project, open the Project instead (`pyflic hub my_project/`, or **Open Project**
in the Project panel) and **double-click a member's row** in the Experiments table to load
it. Either way, once a member is loaded the Experiment tile unfolds its tools — **QC ·
Analyze · Plots · Scripts · AI**. See [Analysis Hub](app-hub.md).

If loading fails, the message in the Output and Errors tabs usually names the problem
directly: a DFM in the configuration with no matching CSV, or a chamber number outside the
range the chamber layout allows. [Troubleshooting](troubleshooting.md) covers the common
ones.

## Analyse

Open **Analyze** and click **Basic analysis**. It applies the design's automatic chamber
removal once, then writes the standard outputs into the member's folder:

| Output | Path | What it is |
|---|---|---|
| Removed chambers | `analysis/removed_chambers.csv` | What auto-removal took out, and why |
| Text summary | `analysis/summary.txt` | Human-readable overview of the run |
| Feeding summary | `analysis/feeding_summary.csv` | One row per chamber: licks, events, durations, intervals |
| Faceted summary | `analysis/feeding_summary_facet.csv` | The same per [Facet](concepts-facets.md), when the experiment has them |
| Summary plot | `analysis/feeding_summary.png` | The same data by treatment |

An Experiment Type adds its own — a Progressive Ratio experiment writes its difference
table, light QC, breaking point and figures too
([Progressive Ratio experiments](concepts-progressive-ratio.md#outputs)). An optogenetic
experiment of any type also gets the optogenetic light QC, in `qc/opto/`: was the light
where the licks were? See [Optogenetic experiments](concepts-optogenetics.md).

**PDF report** on the same panel writes `analysis/experiment_report.pdf`: quality control,
the type's results with statistics, and the parameters used. See [Reports](reports.md).

## Look at the QC output before you believe anything

This is the step that is easiest to skip and most expensive to skip. Basic analysis skips
the per-DFM QC, which is the slow part: open **QC** and click **QC reports** to write the
integrity checks, the two-well crosstalk tables and the signal plots into `qc/`, then
**Open QC Viewer**.

The light QC is the exception. For an optogenetic experiment, basic analysis writes it
every time, and the QC panel's **Optogenetics** group and the viewer's **Opto Light QC**
tab show its verdicts. A group whose light had no licks behind it is flagged, not
excluded; deciding is yours.

The viewer shows you the raw and baselined signal per well, with the detection thresholds
drawn on the baselined trace, and each well's cumulative licks, which climb only where
pyflic detected licks. What you are checking is whether the licks pyflic found are the
ones *you* would have called by eye. If detection looks too eager or too conservative, that is a parameter
question, not a data question — see [How feeding is detected](concepts-licks-events.md)
and [Parameter reference](reference-parameters.md).

The QC viewer can recompute with different parameters live, so you can try a value before
committing it to your configuration file. See [QC Viewer](app-qc-viewer.md), and the
Guides list's *Checking your data* for every check in turn.

## From here

- Draw more figures from the **Plots** panel — [Plot catalogue](plots-catalog.md).
- Stop clicking the same buttons every time by defining a script —
  [What a script is](scripts-overview.md).
- Understand the numbers you just produced — [Summary metrics](concepts-metrics.md).
- Pool several recordings — [Projects and members](concepts-project.md).
