# Changelog

## Unreleased

### New features

- **The Hub is a two-row ribbon** (ADR-0012, mirroring PyTrackingAnalysis). The top strip is the containment hierarchy read left to right — wide **Batch · Project · Experiment** container tiles, a regular **Tools** chip, and the status readout. The four experiment-level tiles — **Analyze · Plots · Scripts · AI** — moved into a collapsible sub-strip of compact, title-only subtiles that the Experiment tile expands, enabled while a member is loaded; their panels are narrow button columns and their status is a hover away. The Experiment tile opens no panel of its own and is the one tile that goes inert as well as dimmed when nothing is loaded. One thing is open at a time: expanding the group closes a container panel and vice versa; click-away folds the group, Esc closes only the panel; unloading folds the group. Immediate feedback on a double-click load now ends with the Analyze panel opening *after* the load completes, since the group it hangs from needs the loaded member. Three container levels get three colors — Batch tints LOAD-blue, Project stays neutral, Experiment is QC-red — `Category.AI` joins the palette, and the chrome surfaces adopt upstream's values so the sibling Hubs read as one family.
- **The Project panel is three cards** — *Create/Load* (**Open Project** / **Create project…** / **Initialize existing directory…** / **Project design…**, a full-width **Validate YAMLs**, and the loaded-project summary in the always-visible card), *Experiments* (the members table with its load options and the member-level Create/Initialize/Member-configs trio), and *Analysis* (the pooled actions and the Project Script row), the last built hidden until a Project is open. The Project-level publication-figure buttons — **Plot editor…** and **Render figures** — moved from the per-member Plots panel onto the Analysis card, where `plot_specs.yaml` and `figures/` actually live.
- **The Plot Editor saves on close.** Closing the editor flushes every named style and every defined plot's spec into `plot_specs.yaml`, so nothing designed in the dialog is lost to a forgotten Save; explicit Save and Render still write eagerly.
- **The seeded `batch` Project Script is the whole unattended run, and explains itself.** `create_project_file` now writes it with the full pipeline spelled out — `run_all_analyses` → `build_combined_analysis` → `project_report` → `render_publication_figures` — because pyflic's `project_report` builds the Combined Analysis only when it is *missing* and never analyzes a member, so the old report+figures seed failed a fresh Project outright and pooled stale results on an old one. It also carries a `notes:` line stating what it runs, when a Batch Run uses it, and where to edit it — and the Script Editor now preserves `notes:` (and any other key it does not understand) instead of stripping it on the first save.

- **Three ways into a Project, three into a member** — mirroring PyTrackingAnalysis's Create/Load and Experiments cards, because a folder can be in exactly three states and a single "New Project here…" answered only one of them. The Project panel now offers **Open a Project…** (it exists and has a `project.yaml`), **Create Project…** (nothing exists yet — the folder is created for you), **Initialize this folder…** (the folder exists without a `project.yaml`: it keeps its name, its subdirectories become the members, and the design is inferred from the first one that has a config), and **Project design…** for the Project already open. Each refuses the other cases and names the button that handles them.
- **Member creation, one level down.** **Create member…** makes the folder, its `data/`, and a design-scaffolded config, then offers **Edit config…** or **Copy config from…**; a copied config is validated against the design *before* it is written, so a non-conforming copy is never made rather than discovered when the Project stops loading. **Initialize existing folder…** adopts a folder already in the Project: loose files are filed into `data/` / `extra_files/` **first**, then the config is scaffolded and opened. **Member configs…** remains the bulk view. Double-clicking a blocked row now offers the repair in place instead of naming the button that would do it.
- **`Project.design_problems_for(config)`** — the load-time conformance test, run against a config that is not in the Project yet. **`layout.initializable_dirs()`** lists the folders that have no config, including ones that are not experiment-shaped yet.

- **Project design editor.** The Project panel gains **Project design…**, an editor for `project.yaml`: the Project's name, its notes, and the `design:` block every member inherits — experiment type, chamber layout, detection parameters, well names, `transform_licks`, facet cutoffs and phase names, the auto-filter constants, the exclusion group and the design factors. **New Project here…** now opens the same editor, so a new Project states its design at creation instead of acquiring one by accident; the button reads *(none set)* for a Project whose members are still being validated against each other. Opening it on a folder that is not a Project yet infers the design from the first member found. Saving a design lists any member carrying a `global:` block of its own and offers to delete those blocks so the members inherit — their `dfms:` and `scripts:` are untouched.

### Changed

- **The config editor is two tabs.** **Experiment** holds everything that applies to the recording as a whole — experiment type, chamber layout, well names, detection parameters and the design factors; **DFMs & Chambers** holds the per-device configuration, one tab down the left side per DFM. The DFM configuration used to be the bottom half of a vertical splitter, competing for height with cards that did not need it. The "Number of DFMs" spinner is gone: **Add DFM** and **Remove DFM** sit under the DFM list, and Remove acts on the DFM you have selected rather than always dropping the last tab — which, with freely-editable DFM ids, was rarely the one anybody meant. The window opens at 1000×850 instead of 960×1020, which overflowed a 1080p screen.
- **The config editor caught up with ADR-0007.** The Experiment Type menu is built from the type registry, so it offers Custom Experiment, Hedonic Feeding and Progressive Ratio, and no longer offers `two_well` and `single_well` — those stopped being experiment types and the editor was writing configs its own linter reported. A typed experiment owns its chamber layout: the layout is shown but not editable, and neither `chamber_layout:` nor `params.chamber_size` is written. A Custom Experiment states `chamber_layout:`. Pre-ADR-0007 configs still open — `experiment_type: two_well` reads as a Custom Experiment with that layout, a bare `params.chamber_size` still sizes the chamber tables — and saving migrates them.
- **Auto-filter thresholds show the type's default instead of inventing one.** The placeholders come from the selected type's `default_constants`, so a Hedonic experiment reads *20 (default for Hedonic Feeding)* rather than a hardcoded example, and a blank field inherits rather than skipping — which is what `resolve_constants` has always done. Only a value you type is written to `constants:`, so a config never freezes against the defaults it happened to be created under. Fixes the max-events placeholder, which suggested 150 against a real default of 150000.
- **Destructive edits say what they will discard.** Reducing the chamber count, or removing a DFM, now names the assignments that would be lost and asks first — and only when something would actually be lost, so building a fresh config is never interrupted. Declining a chamber-layout change also undoes the experiment-type change that asked for it.
- **Problems are counted on the tab they live on.** Each tab label carries a ⚠ and a count sourced from `ExperimentType.validate()` — the loader's own function, so the editor and the loader cannot disagree about what is valid — plus the chamber tables' own factor checks. Saving with problems outstanding lists them first; it is never blocked.
- **The design is reinforced in member configs.** Opening a member's `flic_config.yaml` in the config editor now reads the parent Project's design: the global settings are filled in from it and shown read-only behind a banner naming the `project.yaml`, the chamber table is split by the design's factors, and the per-DFM override checkboxes are limited to the physical keys (`pi_direction`, `chamber_sets`). Saving writes no `global:` block at all, so editing a member can no longer produce a config that stops the Project loading. A standalone experiment is governed by nothing and keeps every field editable.
- **The two script levels sit with what they act on.** Project Scripts moved to the Project panel (pick, **Run**, and **Edit…** which opens the Script Editor on `project.yaml`); the Scripts panel is now the loaded member's Experiment Scripts only, and its tile dims — like Analyze and Plots — until a member is loaded. The Hub's Script Editor button no longer changes which file it opens depending on what happens to be selected.

### Fixed

- **`experiment_type` is compared canonically.** A member saying `hedonic` under a design saying `Hedonic` names the same type — the registry is case-insensitive — but design validation compared the strings and called it a fatal deviation.
- **Scaffolding refuses a member name with a path separator**, which would otherwise write a config and a `data/` folder outside the Project.
- **The config editor opened from Member configs… was frozen.** That dialog ran application-modal, so the config editor it launched painted correctly but ignored clicks, keystrokes and the scroll wheel. It is window-modal now, and the editor is raised and focused when it opens. Editing a second member no longer discards the first editor's window.
- **The config editor silently rewrote factorial designs.** A chamber row with some factor columns filled and some blank was compacted by dropping the blanks, so `(blank, Chrim)` was written as `Chrim` — which reads back as the *first* factor, not the second. Factor assignments are positional; a half-filled row is an incomplete assignment, not a shorter one. Such a row is now flagged in the table and left out of the file instead.
- **The twelfth chamber of a single-well DFM could not be reached.** The chamber table neither scrolls nor clips by design, but its minimum height assumed 26px rows against a theme that draws 30, so the last row fell outside the widget. It is measured now.
- **Loading a single-well config left the chamber tables at six rows.** The chamber count was applied only to DFM widgets created during the load, not to the ones already on screen, so the last six assignments of every existing DFM were unreachable until the layout was toggled by hand.
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
