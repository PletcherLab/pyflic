# Mirrored files

pyflic duplicates generic UI and scripting code from
[PyTrackingAnalysis](../PyTrackingAnalysis) rather than sharing a package. The
three sibling apps — pyflic, PyTrackingAnalysis, pySurvAnalysis — keep their own
copies of the Hub shell, the Script Editor, the Plot Editor, `ui/`, and `help/`,
because their dependency floors and their domains have diverged. Duplication is
the cheaper option today, but its cost is drift, and drift that nobody can see
is drift nobody fixes.

This file makes it visible. Each row records what was copied, from where, and at
which upstream commit, so a future extraction into a shared package is
mechanical rather than archaeological.

**Format note.** This file's shape and conventions are owned by pySurvAnalysis,
which introduced the ledger; pyflic follows the convention, it does not define
it.

**Upstream baseline:** `PyTrackingAnalysis@067e38e` (2026-08-30), raised from
`8d71a42` by the ribbon pass below.

| File here | Upstream path | Relationship |
|---|---|---|
| `pyflic/base/ui/tiles.py` | `pytrackinganalysis/apps/_hub_tiles.py` | **structure ported**, docstrings rewritten; `StatusReadout` here is upstream's `StatusPanel`. Now carries the full dim treatment (background, title *and* icon), `TilePanel._content_height`, and `TilePanel.cards()` |
| `pyflic/base/ui/widgets.py` | `pytrackinganalysis/ui/widgets.py` | pyflic is the **origin** of this file; upstream vendored it and grew. Taken back: `Card.set_dimmed`/`restyle`, `OutputLog.append_stream`/`clear_log`/`line_appended`, `PlotDock`'s Errors tab, clear bar, and `add_widget`. `Card.add_title_widget` keeps pyflic's insert-after-title placement (upstream right-aligns it); `add_figure` keeps returning its content `QSize`. Since 2026-09-24 `CardGroup.add_title_widget` puts the widget at the right of the group's first member when the group has no note, rather than on a row of its own, which opened an empty band under the title; `CardGroup.reflow_title_widget()` moves it beside the first member not hidden by hand, for groups that show and hide members |
| `pyflic/base/ui/zoom.py` | `pytrackinganalysis/ui/zoom.py` | **verbatim** |
| `pyflic/base/ui/textformat.py` | `pytrackinganalysis/ui/textformat.py` | **verbatim** |
| `pyflic/base/ui/theme.py` | `pytrackinganalysis/ui/theme.py` | pyflic's own; + `blocked_color()`, so the Batch table, the Project table, and the preflight tree paint one fact one way |
| `pyflic/base/gui_env.py` | `pytrackinganalysis/gui_env.py` | **verbatim** — same toolkit, same Wayland/IBus complaint at startup |
| `pyflic/base/layout.py` | `pytrackinganalysis/layout.py` | **structure ported, body rewritten** — same `classify` / members-in contract, the same "decide with the loader's own rule" principle, and the same filing allowlist. The recording is many `DFM*.csv` rather than one workbook, so "several recordings here" is normal and the ambiguity is "which copy", not "which file" |
| `pyflic/base/batch.py` | `pytrackinganalysis/batch.py` | **reimplemented** on the same contract — recursive walk with pruning, `project_kind`, relative-path keys, scoped `run_batch`, coverage-aware summary, lazy `batch.yaml`, `resolve_designated_script` central→own→built-in with no implicit fallback. Upstream's `BatchMember` is `BatchProject` here (see the divergence below) |
| `pyflic/base/batch_preflight.py` | `pytrackinganalysis/apps/batch_preflight.py` | **structure ported** — same always-shown review modal, tree of Projects → blocked children, reason-matched repair, Rescan, derive-don't-remember check state. Its sheet section previews an Exclusion Sheet rather than a Removal Sheet |
| `pyflic/base/exclusion_sheet.py` | `pytrackinganalysis/removals.py` (sheet half only) | **reimplemented** on the same contract — `find_sheet` / `read_sheet` / `plan_sheet` / `apply_sheet`, standing-declaration-wins, conflict reporting, root-escape refusal. The unit is a chamber, and the target is the Member's existing `remove_chambers.csv` rather than a new sidecar |
| `pyflic/base/hub.py` | `pytrackinganalysis/apps/hub.py` | **structure ported, body rewritten** — same tile strip and panel model, FLIC domain. Now also the cached recursive scan, the batch table with keys/status/red rows/context menu/check column, card dimming, always-lit Batch and Project tiles, tab suppression, stream logging, "View reports", project-wide YAML validation, and the panel-flow changes (double-click a Batch row → Project, double-click a member → Analyze) |
| `pyflic/base/plot_editor.py` | `pytrackinganalysis/apps/plot_editor.py` | **structure ported, body rewritten** — same Spec/Style split, same one-plot-at-a-time model (a **Plot combo** in the toolbar, never a checkable list: only one figure can be previewed, and a Project's figure set is not something the editor curates), same lazily-created Specs and `Restore defaults` reset. Now also the single controls column of four group boxes (Style / This plot / Facets / Treatments), the per-facet width and height, the three-column Treatments table, and the fit-to-pane preview. pyflic keeps its explicit **Save** button and its *skip the write when nothing changed* close, where upstream saves unconditionally on close and on each vector export |
| `pyflic/base/ai/__init__.py` | `pytrackinganalysis/batch.py` (narrative half) | **reimplemented** — `generate_batch_narrative` synthesizes the Projects' own narratives into one at the Batch root, on the same "synthesis, not pooling" contract |

### The 2026-08-30 pass (the ribbon)

Ported (ADR-0012 here, mirroring upstream's ADR-0012 and its 2026-08-29
follow-ups): the two-row ribbon — wide container tiles (Batch · Project ·
Experiment, `StatusTile(wide=True)` at 1.75×¾) over a collapsible sub-strip of
compact title-only subtiles (Analyze · Plots · Scripts · AI, 38px, status in
the tooltip) that the Experiment group tile expands; the group tile as the one
dimmed-*and*-inert tile (`set_clickable`); one-thing-open-at-a-time between the
group and a container panel; narrow button-column subtile panels; the
three-card Project panel (Create/Load with the three disjoint ways in plus
full-width Validate YAMLs and the always-visible project summary, Experiments,
Analysis built hidden until a Project opens); the Project-level
publication-figure buttons on the Analysis card rather than the Plots panel;
the tile/card color pairing (Batch=LOAD blue, Project=neutral,
Experiment=QC red) with `Category.AI` and upstream's surface palette; the Plot
Editor flushing every style and spec to `plot_specs.yaml` on close (skipping
the write when nothing changed — a look-and-close must not rewrite the file);
and the default `batch` Project Script carrying its explanatory `notes:`, with
its steps spelling out the whole unattended run (analyze → pool → report →
figures) because pyflic's `project_report`, unlike upstream's, does not analyze
members itself.  Tools keeps its chip and panel for now — upstream removed
theirs, and this one is expected to follow.

### The 2026-08-24 pass

Ported: recursive Batch discovery with pruning and relative-path keys, the
Blocked-Member concept and its layout classifier, filing an Unfiled Recording,
the always-shown preflight, the scoped run and its coverage-aware summary,
`StatusTile`'s full dim treatment, `TilePanel._content_height` and `cards()`,
`Card.set_dimmed`/`restyle`, `OutputLog.append_stream`, the `PlotDock` clear bar
and Errors tab, the suppress-tabs switch, "View reports", project-wide YAML
validation, the batch AI narrative, and the sheet-preview/decline switch.

## Not mirrored — deliberate divergences

* **A fifth subtile: QC.** Upstream's Experiment sub-strip is Analyze · Plots ·
  Scripts · AI; pyflic adds **QC** ahead of Analyze (ADR-0012 here).
  FLIC QC is heavyweight and central — per-DFM integrity/bleeding reports and
  the raw / baselined / cumulative-licks signal plots, plus an interactive
  viewer that decides exclusions — where upstream folds QC into its Load + QC
  step and a viewer launched from elsewhere.  The subtile gives the reports,
  the viewer, the saved plots, and the folder one home beside the analysis
  they qualify.

* **A Project's children are Members, not replicates** (ADR-0009). Upstream's
  Project holds the same experiment repeated; pyflic's holds different
  experiments addressing one question. Every mirrored surface renames the word,
  and the Combined Analysis keeps `Experiment` as a column because the rows are
  not interchangeable. This is the single biggest reason a shared package would
  need a vocabulary layer.

* **A Project inside a Batch is a `BatchProject`, never a "member".** Upstream
  calls it `BatchMember`, which is free there because its Project's children are
  replicates. Here it would collide head-on: "a member with four members" is a
  sentence this codebase must not be able to write. `normalize_member_key` is
  kept as an alias of `normalize_key` so upstream-shaped calls still resolve.

* **No Removed Regions and no `removed_regions.yaml`.** Upstream's unit is a
  tracking region and its sidecar is new machinery. Here the **Excluded
  Chamber** (`remove_chambers.csv`, active group named by the Design) already
  removes chambers from the analysis population and is already stamped on every
  output, so only the bulk-authoring half was needed — the **Exclusion Sheet**
  (ADR-0010) — plus upstream ADR-0010's **stale rule**, implemented here by
  comparing the declaration's mtime against the saved `feeding_summary.csv`
  rather than by stamping provenance into an on-disk format that R scripts read.

* **"Several recordings here" is not ambiguous.** Upstream refuses to file when
  a directory holds two workbooks, because it cannot tell which is the
  experiment. A FLIC recording is many files — one or more per DFM — so the
  refusal here is narrower and sharper: the same DFM id present both loose and
  in `data/`, which is "which copy", not "which file".

* **No case-insensitive `data/` match.** Upstream's `Experiment._find_subdir`
  takes the first case-insensitive match, so its classifier has to match the
  same way. `load_experiment_yaml` builds `experiment_dir / "data"` and nothing
  else, so a `Data/` here is genuinely invisible to the loader and the
  classifier says so.

* **No `install_desktop.py` and no artifact tabs in the QC viewer.** Not
  ported; nothing here needs them yet.

* **`script_editor/` is not part of this pass.** Upstream's palette height fix
  and its `textformat`-driven inspector changes were left alone; only
  `textformat.py` itself came across, for the Output log.

## Keeping this honest

When you copy something new across, add a row. When you edit a file marked
*verbatim*, change its relationship to describe the edit — a row that claims
"verbatim" while the file has diverged is worse than no row at all.
