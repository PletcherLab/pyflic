# Analysis Hub

The main pyflic application, and where most work happens.

```bash
pyflic hub my_experiment/
```

The project argument is optional — run `pyflic hub` and choose a folder from the Project
card.

## Layout

A navigation rail on the left jumps to six cards on the right; the output panel and plot
tabs sit alongside. Cards appear only when they apply, so a single-well experiment shows no
preference-index controls and only a hedonic experiment shows the hedonic plot.

| Card | What it holds |
|---|---|
| **Project** | Choose the project directory, pick the active configuration, batch toggles |
| **Load** | Load the experiment, remove chambers |
| **Analyze** | Summaries, CSV exports, statistics |
| **Plots** | Interactive figures |
| **Scripts** | Run a script, run all scripts |
| **Tools** | Lint, compare configurations, clear the cache |

## Project card

Sets the project directory and the **active configuration** — the YAML currently driving
the hub. If your project holds several configurations, this dropdown is how you switch
between them.

- **YAML info** — a summary of every configuration in the folder: experiment type, chamber
  size, DFM count, factors, and the scripts each one defines. Useful for finding out what
  a project contains without opening files.
- **Reload config** — re-read the YAML after editing it elsewhere.
- **Edit config** / **QC viewer** — launch the other applications on this project.

The two batch toggles live here. They are mutually exclusive, and both are described in
[Running many projects at once](scripts-batch.md).

## Load card

**Load experiment** reads the CSVs, subtracts baselines, and runs detection. This is the
slow step; everything downstream reuses its result.

Start and end minute controls restrict the analysis to a time window. `end: 0` means
through the end of the recording. Ranged loads write into their own suffixed output folders
so they do not overwrite whole-experiment results.

**Remove chambers** applies the `general` exclusion group from `remove_chambers.csv` — see
[exclusions](config-dfms-chambers.md#excluding-chambers).

## Analyze card

| Button | Produces |
|---|---|
| Run full basic analysis | QC reports, `summary.txt`, feeding summary, summary plot |
| Write feeding summary CSV | Per-chamber metrics |
| Write binned feeding summary CSV | Metrics in time bins |
| Write weighted duration summary | Hedonic experiments only |
| Tidy events CSV | One row per bout |
| Bootstrap CIs (metric)… | Bootstrap confidence intervals |
| Compare treatments (ANOVA / LMM)… | Statistical comparison |
| Light-phase summary CSV | Split by light and dark phase |
| Parameter sensitivity sweep… | Metrics across a range of one parameter |
| Bout transition matrix | Transition probabilities |
| Write PDF report | Everything bundled into one PDF |

These are the same operations available as [script actions](scripts-actions.md). Use the
buttons while exploring; move to a script once you know the sequence you want.

## Plots card

Each plot row has its own metric and mode selectors. **Mode** decides how the two wells of
a chamber combine — total, Well A, Well B, or their mean; see
[metric and mode](scripts-actions.md#metric-and-mode).

The **Moving window** section has its own window and step controls, both in minutes, for
the sliding-window plots. Everything drawn is catalogued in
[Plot catalogue](plots-catalog.md).

The **Interactive plots** checkbox in the top bar controls how figures are embedded.
Interactive gives you pan, zoom and hover tooltips; unchecked renders a static image, which
is faster to paint and lighter on memory. Turn it off when generating many figures at once.

## Scripts card

**Run Script** runs the script chosen in the dropdown; **Run All Scripts** runs every
script in the active configuration in sequence. In subdir-batch mode the button changes to
show how many batch targets were found — check that count before starting a long run.

## Tools card

- **Lint config** — validate the active configuration and report problems with line numbers
- **Compare two configs…** — diff two configurations, for tracking down why two analyses of
  the same data differ
- **Clear disk cache** — remove `.pyflic_cache/`

The cache is keyed by input, so clearing it is never required for correctness — it is a
disk-space operation.

## Output panel

Everything a run prints appears here in real time, and figures open as additional tabs
beside it. When something fails, the message here is the first place to look; it usually
names the problem directly.

---

Related: [QC Viewer](app-qc-viewer.md) · [Config Editor](app-config-editor.md) ·
[Troubleshooting](troubleshooting.md)
