# Changelog

## 2026-04-24

### Breaking changes

- **`plot_binned_metric_by_treatment` and `plot_binned_metrics_by_treatment` now return plotnine `ggplot`** instead of a matplotlib `Figure`. Any code that calls `.savefig(...)` on the result must be updated to use `p.save("out.png", dpi=200)` instead.

- **`ProgressiveRatioExperiment.plot_breaking_point_dfm` removed.** The matplotlib version of the per-DFM breaking-point plot has been deleted. Use `plot_breaking_point_dfm_gg()` (returns a plotnine `ggplot`) instead.

### Improvements

- **Higher-resolution plot outputs.** Default DPI raised across several outputs:
  - QC report PNGs (`write_qc_reports()`): 150 → 200 DPI
  - `write_feeding_summary_plot()` default: 150 → 200 DPI
  - `execute_basic_analysis()` saved plots: 150 → 200 DPI
  - `HedonicFeedingExperiment.hedonic_feeding_plot()` default: 150 → 200 DPI
  - Analysis Hub in-app plot rendering: 120 → 150 DPI

- **QC Viewer redesign.** The viewer now has a themed top bar with a **light/dark mode toggle** button. All panels use Card widgets with category-tinted styling consistent with the Analysis Hub and Script Editor. The **Params** tab now appears after all DFM tabs.

- **Config Editor layout.** Experiment Settings and Global Parameters are now displayed side by side, making better use of horizontal space.

- **Analysis Hub button placement.** The **Edit config…** and **QC viewer…** launch buttons have moved from the Load card to the Project card.
