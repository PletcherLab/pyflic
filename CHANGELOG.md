# Changelog

## Unreleased

### New features

- **Project design editor.** The Project panel gains **Project design…**, an editor for `project.yaml`: the Project's name, its notes, and the `design:` block every member inherits — experiment type, chamber layout, detection parameters, well names, `transform_licks`, facet cutoffs and phase names, the auto-filter constants, the exclusion group and the design factors. **New Project here…** now opens the same editor, so a new Project states its design at creation instead of acquiring one by accident; the button reads *(none set)* for a Project whose members are still being validated against each other. Opening it on a folder that is not a Project yet infers the design from the first member found. Saving a design lists any member carrying a `global:` block of its own and offers to delete those blocks so the members inherit — their `dfms:` and `scripts:` are untouched.

### Changed

- **The design is reinforced in member configs.** Opening a member's `flic_config.yaml` in the config editor now reads the parent Project's design: the global settings are filled in from it and shown read-only behind a banner naming the `project.yaml`, the chamber table is split by the design's factors, and the per-DFM override checkboxes are limited to the physical keys (`pi_direction`, `chamber_sets`). Saving writes no `global:` block at all, so editing a member can no longer produce a config that stops the Project loading. A standalone experiment is governed by nothing and keeps every field editable.
- **The two script levels sit with what they act on.** Project Scripts moved to the Project panel (pick, **Run**, and **Edit…** which opens the Script Editor on `project.yaml`); the Scripts panel is now the loaded member's Experiment Scripts only, and its tile dims — like Analyze and Plots — until a member is loaded. The Hub's Script Editor button no longer changes which file it opens depending on what happens to be selected.

### Fixed

- **The config editor opened from Member configs… was frozen.** That dialog ran application-modal, so the config editor it launched painted correctly but ignored clicks, keystrokes and the scroll wheel. It is window-modal now, and the editor is raised and focused when it opens. Editing a second member no longer discards the first editor's window.
- **The config editor's DFM pane could not be reached.** The top pane handed its (tall) minimum height to the splitter, squeezing the DFM tabs to a couple of hundred pixels with no way to scroll to the rest. Both panes scroll now, and the window sizes itself to the screen rather than to a fixed 1020px.
- **Saving a member config no longer deletes its `scripts:`.** The editor rebuilt the file from its widgets, dropping every key it does not itself edit; it now preserves them.
- **Scaffolding a single-well member gave it six chambers** instead of twelve when the Project had no member to copy from — the blank chamber block follows the design's chamber layout.

### Visual

- **Inline `?` help buttons are quieter.** They rest in a warm grey and take the amber accent (and a faint amber wash) only under the pointer, instead of a column of amber dots down every parameter form.

## 2026-05-04

### New features

- **Subdir batch mode in the Analysis Hub.** A new **"Run 'batch' script in every subdirectory"** checkbox appears in the Project card. When enabled, clicking **Run Script** walks every immediate subdirectory of the project directory, finds YAMLs in each one, and runs the script named `batch` from every YAML that defines it. Each subdirectory is treated as its own independent project, so outputs land inside that subdir. Subdirectories whose YAMLs do not define a `batch` script are skipped with a log message. The Config dropdown, Script picker, and single-project cards (Load, Analyze, Plots, Tools) are hidden in this mode. The two batch modes ("Run action for every YAML config" and "Run 'batch' script in every subdirectory") are mutually exclusive.

## 2026-04-28

### Internal changes

- **Feeding-summary cache invalidated on upgrade.** The on-disk cache version has been bumped from 1 to 2. Any `.pyflic_cache/` entries written by a prior release will be automatically discarded and recomputed on the first run after upgrading — no manual action is required. This ensures the corrected lick-count values from the lick-gap fix are reflected in all cached feeding summaries.

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
