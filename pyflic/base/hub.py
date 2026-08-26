"""The pyflic Analysis Hub — a tile strip over a full-width output area.

Replaces the pre-overhaul card column.  The strip across the top is
Batch · Project · Analyze · Plots · Scripts · AI · Tools plus a status readout
filling the remaining width; every control lives in a tile's anchored panel,
one open at a time.

Two rules shape it:

* **Tiles never move or hide.**  An inapplicable tile dims and its panel holds
  the control that fixes the missing state, so the strip is a stable map rather
  than a shifting menu.
* **Project-first.**  The selection names the working container — a Batch or a
  Project — and an experiment is loaded *only* by double-clicking its row in the
  Project panel's members table.  There is no Load tile: the load options
  live in the Project panel beside the table that triggers the load.
  Double-clicking a Batch row opens the Project panel and double-clicking a
  member opens the Analyze panel: selecting is only ever a step toward doing
  something.
* **Every folder that could run is visible.**  Batch discovery is recursive and
  prunes at each Project, and a Blocked Member — an unfiled recording, a folder
  with no config — is listed in red where the button that fixes it lives, rather
  than failing at load an hour into an unattended run (ADR-0009).

The Batch and Project tiles are never dimmed: their panels hold the controls
that fix the empty state, so a closed-looking tile there would point away from
the only way forward.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

from PyQt6.QtCore import QObject, QSize, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import batch as batch_mod
from . import layout as layout_mod
from . import project as project_mod
from .gui_env import sanitize_input_method_environment
from .ui import (
    Category,
    OutputLog,
    PlotDock,
    apply_theme,
    blocked_color,
    icon,
    surface_colors,
)
from .ui import settings as ui_settings
from .ui.tiles import TILE_HEIGHT, ClickAwayFilter, StatusReadout, StatusTile, TilePanel
from .ui.widgets import ActionButton, Card

#: (key, title, icon, category, panel width).  Order is strip order.
TILE_SPECS: list[tuple[str, str, str, Category, int]] = [
    ("batch",   "Batch",   "batch",    Category.NEUTRAL, 640),
    ("project", "Project", "project",  Category.LOAD,    720),
    ("analyze", "Analyze", "analyze",  Category.ANALYZE, 520),
    ("plots",   "Plots",   "plots",    Category.PLOTS,   520),
    ("scripts", "Scripts", "scripts",  Category.SCRIPTS, 560),
    ("ai",      "AI",      "ai",       Category.TOOLS,   480),
    ("tools",   "Tools",   "tools",    Category.TOOLS,   480),
]


# ---------------------------------------------------------------------------
# Worker plumbing
# ---------------------------------------------------------------------------

class _Stream(QObject):
    """A file-like that forwards writes to the GUI thread as a signal."""

    text = pyqtSignal(str)

    def write(self, data) -> int:
        if data:
            self.text.emit(str(data))
        return len(str(data))

    def flush(self) -> None:
        pass


class Worker(QThread):
    """Runs one callable off the GUI thread, streaming its stdout to the log.

    Figures come back through :attr:`finished_ok` rather than being drawn in the
    worker: touching a Qt widget off the GUI thread is a hard abort, and a
    matplotlib figure built in a worker is safe only while nothing paints it.
    """

    line = pyqtSignal(str)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, task: Callable[[], Any], parent=None) -> None:
        super().__init__(parent)
        self._task = task

    def run(self) -> None:
        stream = _Stream()
        stream.text.connect(self.line, Qt.ConnectionType.QueuedConnection)
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = stream  # type: ignore[assignment]
        try:
            result = self._task()
            self.finished_ok.emit(result)
        except Exception:  # noqa: BLE001
            self.failed.emit(traceback.format_exc())
        finally:
            sys.stdout, sys.stderr = old_out, old_err


# ---------------------------------------------------------------------------
# Hub
# ---------------------------------------------------------------------------

class AnalysisHubWindow(QMainWindow):
    """The Hub window: tile strip, anchored panels, output/plot area."""

    def __init__(self, target: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("pyflic Analysis Hub")
        self.resize(1280, 860)

        #: The working container — a Batch or a Project — and what is loaded.
        self.batch_root: str | None = None
        self.project: project_mod.Project | None = None
        self.experiment_name: str | None = None
        self.experiment = None
        self._worker: Worker | None = None
        self._open_key: str | None = None
        #: One recursive walk per selection, reused by the table, the tiles,
        #: and the run.  Discovery is recursive now, so the walk is far more
        #: expensive than the single ``listdir`` it replaced — and ``refresh``
        #: runs on every checkbox toggle and every finished task (ADR-0009).
        self._batch_scan_cache: tuple | None = None
        #: The Batch the panel is *showing*.  The app selection still names
        #: exactly one container, but the Batch panel is allowed to stay open
        #: while a row double-click selects one of its Projects.
        self._batch_panel_root: str | None = None
        self._noted_sheet: str | None = None
        self._noted_truncation: str | None = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 10)
        root.setSpacing(8)

        # ---- strip ----------------------------------------------------
        strip_host = QWidget()
        self._strip = QHBoxLayout(strip_host)
        self._strip.setContentsMargins(0, 0, 0, 0)
        self._strip.setSpacing(2)
        strip_host.setFixedHeight(TILE_HEIGHT + 2)
        self.tiles: dict[str, StatusTile] = {}
        for index, (key, title, icon_name, category, _w) in enumerate(TILE_SPECS):
            tile = StatusTile(key, title, icon_name, category)
            tile.clicked.connect(self._on_tile_clicked)
            left = 5 if index == 0 else 0
            right = 0
            tile.set_rounding(left, right)
            self._strip.addWidget(tile)
            self.tiles[key] = tile
        self.readout = StatusReadout()
        self._strip.addWidget(self.readout, 1)
        root.addWidget(strip_host)

        # ---- output / plots -------------------------------------------
        ## Two logs, not one: a Batch Run's ordinary output runs to thousands
        ## of lines, and the four that say a Project failed are the only ones
        ## anybody needs.  The Errors tab badges itself while it is unread.
        self.log = OutputLog()
        self.errors = OutputLog()
        self.dock = PlotDock(self.log, self.errors)
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.dock)
        root.addWidget(splitter, 1)

        # ---- panels ----------------------------------------------------
        self.panels: dict[str, TilePanel] = {}
        for key, _t, _i, _c, width in TILE_SPECS:
            self.panels[key] = TilePanel(key, width, central)
        self._build_panels()
        self._install_panel_help()
        for panel in self.panels.values():
            panel.finish()

        self._click_filter = ClickAwayFilter(self)
        QApplication.instance().installEventFilter(self._click_filter)

        self._restyle()
        if target:
            self.open_target(target)
        else:
            self.refresh()

    # ------------------------------------------------------------------
    # Panel construction
    # ------------------------------------------------------------------

    #: Project keys are relative paths in a recursive batch.  Keep the first
    #: column bounded so a deep Project cannot force the Batch table wider
    #: than its panel; Qt paints the hidden tail with an ellipsis.
    BATCH_KEY_COLUMN_WIDTH = 260

    #: Tile key → help topic reference. Adding a tile means adding a line here;
    #: ``tests/test_help_refs.py`` asserts every value resolves.
    _TILE_HELP: dict[str, str] = {
        "batch": "scripts-batch#the-review-window",
        "project": "concepts-project#blocked-members",
        "analyze": "app-hub#analyze-panel",
        "plots": "plots-catalog",
        "scripts": "scripts-overview",
        "ai": "concepts-ai-summary",
        "tools": "app-hub#tools-panel",
    }

    def _install_panel_help(self) -> None:
        """Put a ``?`` in each panel's card title row.

        Imported here rather than at module scope so a failure in the help
        package degrades to a Hub without help buttons instead of no Hub.
        """
        try:
            from pyflic.help.button import HelpButton
        except Exception:  # noqa: BLE001
            return
        for key, ref in self._TILE_HELP.items():
            panel = self.panels.get(key)
            if panel is None:
                continue
            for card in panel.findChildren(Card):
                card.add_title_widget(HelpButton(ref, card))
                break

    def _build_panels(self) -> None:
        self._build_batch_panel()
        self._build_project_panel()
        self._build_analyze_panel()
        self._build_plots_panel()
        self._build_scripts_panel()
        self._build_ai_panel()
        self._build_tools_panel()

    @staticmethod
    def _table(headers: list[str]) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().setVisible(False)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        table.setMinimumHeight(160)
        return table

    def _build_batch_panel(self) -> None:
        """The Batch panel (ADR-0006, ADR-0009): a Batch is a directory with
        Projects anywhere beneath it, found by a walk that prunes at each one.
        A Batch Run executes one designated Project Script in every checked
        Project — there is no third script level, and a Batch never pools
        results across Projects."""
        card = Card("Batch", Category.NEUTRAL, icon_name="batch",
                    subtitle="Run a Project Script in every Project of this "
                             "folder.  Projects are found recursively, so "
                             "grouping folders are transparent.")
        pick_row = QHBoxLayout()
        open_btn = ActionButton("Choose batch folder…", Category.NEUTRAL,
                                "open", primary=True)
        open_btn.setToolTip(
            "Pick the folder that holds your Projects.  They are found "
            "recursively, so they can sit at any depth inside it — every one "
            "found is listed below for the run.")
        open_btn.clicked.connect(self._choose_batch)
        pick_row.addWidget(open_btn)
        self.batch_rescan_btn = ActionButton("Rescan", Category.TOOLS,
                                             "refresh")
        self.batch_rescan_btn.setToolTip(
            "Walk the batch folder again.  The project list is read once when "
            "the folder is selected; rescan after adding or fixing projects "
            "outside the app.")
        self.batch_rescan_btn.setSizePolicy(QSizePolicy.Policy.Fixed,
                                            QSizePolicy.Policy.Fixed)
        self.batch_rescan_btn.clicked.connect(self._rescan_batch)
        pick_row.addWidget(self.batch_rescan_btn)
        ## A secondary action on the folder you just chose, not a peer of Run
        ## batch.  Enabled only when that folder actually holds a sheet.
        self.batch_sheet_btn = ActionButton("Apply exclusion sheet…",
                                            Category.TOOLS, "clear")
        self.batch_sheet_btn.setToolTip(
            "Read remove_chambers.csv at the batch folder and write its rows "
            "into each member's own remove_chambers.csv.  A Batch Run applies "
            "it automatically before running; this is for applying it now.  "
            "Declarations already in place are kept.")
        self.batch_sheet_btn.setSizePolicy(QSizePolicy.Policy.Fixed,
                                           QSizePolicy.Policy.Fixed)
        self.batch_sheet_btn.clicked.connect(
            lambda: self._apply_exclusion_sheet(
                self._batch_view_root(), "batch folder",
                projects=self._batch_checked_keys()))
        pick_row.addWidget(self.batch_sheet_btn)
        pick_row.addStretch(1)
        card.add_body(pick_row)

        self.batch_empty = QLabel(
            "Choose a batch folder — one with Projects anywhere inside it — "
            "and every Project found is listed here for the run.  A Project "
            "is a folder with a project.yaml and at least one member "
            "directory.")
        self.batch_empty.setStyleSheet(
            "color: palette(mid); font-style: italic;")
        self.batch_empty.setWordWrap(True)
        card.add_body(self.batch_empty)

        self.batch_table = self._table(["Project", "Members", "Report",
                                        "Status"])
        self.batch_table.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.batch_table.setColumnWidth(0, self.BATCH_KEY_COLUMN_WIDTH)
        self.batch_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Fixed)
        self.batch_table.setToolTip(
            "Checked Projects join the next Batch Run.  Projects are found "
            "recursively, so a row's name is its path inside the batch "
            "folder.  Double-click a row to open that Project; right-click "
            "for its blocked members.")
        self.batch_table.itemDoubleClicked.connect(self._batch_row_activated)
        ## Right-click, not double-click: double-click already means "open
        ## this Project", so the repair entry takes the gesture that is free.
        self.batch_table.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self.batch_table.customContextMenuRequested.connect(
            self._batch_table_menu)
        self.batch_table.itemChanged.connect(self._on_batch_item_changed)
        card.add_body(self.batch_table)

        run_row = QHBoxLayout()
        run_row.addWidget(QLabel("Project Script:"))
        self.batch_script = QComboBox()
        self.batch_script.setMinimumWidth(200)
        self.batch_script.setToolTip(
            "The designated Project Script — resolved per Project from "
            "batch.yaml project_scripts, then the project's own scripts, then "
            "the built-ins.  The default runs each project's own 'batch' "
            "script; a project with none is reported and skipped.  Changing "
            "it is remembered in batch.yaml.")
        self.batch_script.currentIndexChanged.connect(
            self._on_batch_script_changed)
        run_row.addWidget(self.batch_script, 1)
        self.batch_run_btn = ActionButton("Run Batch", Category.SCRIPTS, "play",
                                          primary=True)
        self.batch_run_btn.setToolTip(
            "Run the designated Project Script in every checked Project — "
            "continue-on-error, per-Project summary at the end.  A review "
            "window states the target list first.")
        self.batch_run_btn.clicked.connect(lambda: self._run_batch())
        run_row.addWidget(self.batch_run_btn)
        card.add_body(run_row)

        self.chk_batch_narrative = QCheckBox("AI narrative of the batch")
        self.chk_batch_narrative.setToolTip(
            "After the run, ask an AI provider to synthesize the Projects' "
            "own narratives into batch_ai_narrative.md at the batch folder — "
            "results across the batch, design problems, and Projects that "
            "lost a lot of chambers.")
        card.add_body(self.chk_batch_narrative)

        ## A Batch Run touches every member of every Project, so the figure
        ## and artifact tabs it would open run into the hundreds and bury the
        ## Output tab the user is actually reading.  Checked, new tabs stop
        ## being created; Output and Errors keep streaming and every artifact
        ## is still written to disk.
        self.chk_suppress_tabs = QCheckBox("Suppress new plot / output tabs")
        self.chk_suppress_tabs.setToolTip(
            "Stop opening a tab for every figure.  The Output and Errors tabs "
            "keep updating, and every artifact is still written to disk — "
            "only the tabs are skipped.  Applies to all runs while it is "
            "checked, not just Batch Runs.")
        self.chk_suppress_tabs.setChecked(True)
        card.add_body(self.chk_suppress_tabs)

        card.add_section_label(
            "Double-click a project to make it the working container.")
        self.panels["batch"].add_card(card)

    def _build_project_panel(self) -> None:
        card = Card("Project", Category.LOAD, icon_name="project",
                    subtitle="Members of one Project — different experiments "
                             "addressing one question, pooled by the Combined "
                             "Analysis.  Double-click a member to load it.")
        row = QHBoxLayout()
        open_btn = ActionButton("Open a Project…", Category.LOAD, "open",
                                primary=True)
        open_btn.clicked.connect(self._choose_project)
        row.addWidget(open_btn)
        new_btn = ActionButton("New Project here…", Category.LOAD, "new")
        new_btn.clicked.connect(self._create_project)
        row.addWidget(new_btn)
        card.add_body(row)

        self.project_table = self._table(["Member", "DFMs", "Chambers",
                                          "Analyzed", "Report"])
        self.project_table.setToolTip(
            "Double-click a member to load it.  A red row is a Blocked "
            "Member — a run cannot use it as it stands; its reason is in the "
            "tooltip and the repair buttons below clear it.")
        self.project_table.itemDoubleClicked.connect(self._project_row_activated)
        card.add_body(self.project_table)

        ## Repairs, in the panel that lists what needs them (ADR-0009).  Both
        ## stay visible and simply disable: a button that vanishes when there
        ## is nothing to fix teaches nobody that it exists.
        self.repair_row = QHBoxLayout()
        self.file_btn = ActionButton("File unfiled recordings",
                                     Category.TOOLS, "file")
        self.file_btn.setToolTip(
            "Move DFM CSVs sitting loose at a member's root into its data/ "
            "folder (everything else loose goes to extra_files/).  YAML files "
            "and remove_chambers.csv stay where they are, and nothing is "
            "overwritten.")
        self.file_btn.clicked.connect(self._file_unfiled)
        self.repair_row.addWidget(self.file_btn)
        self.scaffold_btn = ActionButton("Member configs…", Category.LOAD,
                                         "new")
        self.scaffold_btn.setToolTip(
            "Give a folder that holds DFM CSVs a design-conformant "
            "flic_config.yaml, scaffolded from an existing member and "
            "reconciled against the DFMs actually in its data/.")
        self.scaffold_btn.clicked.connect(self._open_member_configs)
        self.repair_row.addWidget(self.scaffold_btn)
        self.repair_row.addStretch(1)
        card.add_body(self.repair_row)

        card.add_section_label("Load options")
        opts = QHBoxLayout()
        self.chk_parallel = QCheckBox("Load DFMs in parallel")
        self.chk_parallel.setChecked(True)
        opts.addWidget(self.chk_parallel)
        opts.addWidget(QLabel("Workers:"))
        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(0, 64)
        self.spin_workers.setSpecialValueText("auto")
        opts.addWidget(self.spin_workers)
        opts.addStretch(1)
        card.add_body(opts)

        card.add_section_label("Project actions")
        acts = QHBoxLayout()
        for label, icon_name, handler in (
            ("Analyze all", "basic", self._run_all_members),
            ("Combine", "csv", self._build_combined),
            ("Create report", "pdf", self._project_report),
        ):
            button = ActionButton(label, Category.ANALYZE, icon_name)
            button.clicked.connect(handler)
            acts.addWidget(button)
        card.add_body(acts)

        views = QHBoxLayout()
        self.view_reports_btn = ActionButton("View reports", Category.ANALYZE,
                                             "report")
        self.view_reports_btn.setToolTip(
            "Open the Project Report, and each member's own report, in the "
            "system PDF viewer.")
        self.view_reports_btn.clicked.connect(self._view_reports)
        views.addWidget(self.view_reports_btn)
        sheet_btn = ActionButton("Apply exclusion sheet…", Category.TOOLS,
                                 "clear")
        sheet_btn.setToolTip(
            "Read remove_chambers.csv at this Project's root and write its "
            "rows into each member's own file.  Standing declarations win.")
        sheet_btn.clicked.connect(
            lambda: self._apply_exclusion_sheet(
                self.project.project_directory if self.project else None,
                "project"))
        views.addWidget(sheet_btn)
        self.project_sheet_btn = sheet_btn
        views.addStretch(1)
        card.add_body(views)
        self.panels["project"].add_card(card)

    def _build_analyze_panel(self) -> None:
        card = Card("Analyze", Category.ANALYZE, icon_name="analyze",
                    subtitle="Actions on the loaded member.")
        self.analyze_hint = QLabel("")
        self.analyze_hint.setWordWrap(True)
        card.add_body(self.analyze_hint)

        grid = QVBoxLayout()
        for label, icon_name, action in (
            ("Basic analysis", "basic", {"action": "basic_analysis"}),
            ("Feeding summary CSV", "csv", {"action": "feeding_csv"}),
            ("Faceted summary CSV", "csv", {"action": "facet_csv"}),
            ("Binned CSV", "binned", {"action": "binned_csv"}),
            ("Tidy events CSV", "tidy", {"action": "tidy_export"}),
            ("PDF report", "pdf", {"action": "pdf_report"}),
        ):
            button = ActionButton(label, Category.ANALYZE, icon_name)
            button.clicked.connect(
                lambda _c=False, s=action: self._run_experiment_action(s))
            grid.addWidget(button)
        card.add_body(grid)

        row = QHBoxLayout()
        row.addWidget(QLabel("Bin size (min):"))
        self.spin_binsize = QDoubleSpinBox()
        self.spin_binsize.setRange(0.5, 600.0)
        self.spin_binsize.setValue(30.0)
        row.addWidget(self.spin_binsize)
        row.addStretch(1)
        card.add_body(row)
        self.panels["analyze"].add_card(card)

    def _build_plots_panel(self) -> None:
        card = Card("Plots", Category.PLOTS, icon_name="plots",
                    subtitle="Quick figures for the loaded member, and the "
                             "Plot Editor for the Project's publication "
                             "figures.")
        self.plots_hint = QLabel("")
        self.plots_hint.setWordWrap(True)
        card.add_body(self.plots_hint)

        row = QHBoxLayout()
        row.addWidget(QLabel("Metric:"))
        self.plot_metric = QComboBox()
        self.plot_metric.setMinimumWidth(180)
        row.addWidget(self.plot_metric, 1)
        card.add_body(row)

        for label, icon_name, action in (
            ("Feeding summary", "feeding", "plot_feeding_summary"),
            ("Binned time course", "binned", "plot_binned"),
            ("Dot plot", "dot", "plot_dot"),
            ("Well A vs B", "well", "plot_well_comparison"),
        ):
            button = ActionButton(label, Category.PLOTS, icon_name)
            button.clicked.connect(
                lambda _c=False, a=action: self._run_plot_action(a))
            card.add_body(button)

        card.add_section_label("Publication figures (project level)")
        editor_btn = ActionButton("Open Plot Editor", Category.PLOTS, "plot",
                                  primary=True)
        editor_btn.clicked.connect(self._open_plot_editor)
        card.add_body(editor_btn)
        render_btn = ActionButton("Render figures from plot_specs.yaml",
                                  Category.PLOTS, "plot")
        render_btn.clicked.connect(self._render_figures)
        card.add_body(render_btn)
        self.panels["plots"].add_card(card)

    def _build_scripts_panel(self) -> None:
        card = Card("Scripts", Category.SCRIPTS, icon_name="scripts",
                    subtitle="Two levels, separate registries. The only bridge "
                             "is run_in_experiments.")
        card.add_section_label("Project Scripts (project.yaml)")
        prow = QHBoxLayout()
        self.project_script = QComboBox()
        self.project_script.setMinimumWidth(200)
        prow.addWidget(self.project_script, 1)
        run_p = ActionButton("Run", Category.SCRIPTS, "play", primary=True)
        run_p.clicked.connect(self._run_project_script)
        prow.addWidget(run_p)
        card.add_body(prow)

        card.add_section_label("Experiment Scripts (loaded member)")
        erow = QHBoxLayout()
        self.experiment_script = QComboBox()
        self.experiment_script.setMinimumWidth(200)
        erow.addWidget(self.experiment_script, 1)
        run_e = ActionButton("Run", Category.SCRIPTS, "play")
        run_e.clicked.connect(self._run_experiment_script)
        erow.addWidget(run_e)
        card.add_body(erow)

        edit_btn = ActionButton("Open Script Editor", Category.SCRIPTS, "script")
        edit_btn.clicked.connect(self._open_script_editor)
        card.add_body(edit_btn)
        self.panels["scripts"].add_card(card)

    def _build_ai_panel(self) -> None:
        card = Card("AI summary", Category.TOOLS, icon_name="ai",
                    subtitle="An AI-written narrative of the Combined "
                             "Analysis, generated from the report's own "
                             "content. It summarizes; it never analyzes.")
        self.ai_hint = QLabel("")
        self.ai_hint.setWordWrap(True)
        card.add_body(self.ai_hint)

        row = QHBoxLayout()
        row.addWidget(QLabel("Provider:"))
        self.ai_provider = QComboBox()
        self.ai_provider.currentTextChanged.connect(self._refresh_ai_models)
        row.addWidget(self.ai_provider, 1)
        card.add_body(row)

        mrow = QHBoxLayout()
        mrow.addWidget(QLabel("Model:"))
        self.ai_model = QComboBox()
        self.ai_model.setEditable(True)
        mrow.addWidget(self.ai_model, 1)
        card.add_body(mrow)

        gen = ActionButton("Generate narrative", Category.TOOLS, "ai",
                           primary=True)
        gen.clicked.connect(self._generate_narrative)
        card.add_body(gen)
        self.panels["ai"].add_card(card)

    def _build_tools_panel(self) -> None:
        card = Card("Tools", Category.TOOLS, icon_name="tools")
        for label, icon_name, handler in (
            ("Config editor", "config", self._open_config_editor),
            ("QC viewer", "qc", self._open_qc_viewer),
            ("Validate every YAML here", "lint", self._validate_yaml),
            ("Lint / migration check", "lint", self._run_lint),
            ("Open this folder", "open", self._open_folder),
            ("Clear cache", "clear", self._clear_cache),
            ("Toggle theme", "theme_dark", self._toggle_theme),
            ("Help", "help", self._open_help),
        ):
            button = ActionButton(label, Category.TOOLS, icon_name)
            button.clicked.connect(handler)
            card.add_body(button)
        self.panels["tools"].add_card(card)

    # ------------------------------------------------------------------
    # Panel open / close
    # ------------------------------------------------------------------

    def _on_tile_clicked(self, key: str) -> None:
        if self._open_key == key:
            self._close_panel()
        else:
            self._open_panel(key)

    def _open_panel(self, key: str) -> None:
        self._close_panel()
        tile = self.tiles[key]
        panel = self.panels[key]
        central = self.centralWidget()
        top_left = tile.mapTo(central, tile.rect().bottomLeft())
        panel.open_at(top_left.x(), top_left.y() + 4, central.height() - 8)
        tile.set_active(True)
        self._open_key = key

    def _close_panel(self) -> None:
        if self._open_key is None:
            return
        self.panels[self._open_key].hide()
        self.tiles[self._open_key].set_active(False)
        self._open_key = None

    def _handle_click_away(self, event) -> None:
        """Close the open panel on a click outside it and outside the strip."""
        if self._open_key is None:
            return
        widget = QApplication.widgetAt(event.globalPosition().toPoint())
        if widget is None:
            return
        panel = self.panels[self._open_key]
        node = widget
        while node is not None:
            if node is panel or isinstance(node, StatusTile):
                return
            node = node.parentWidget()
        self._close_panel()

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.key() == Qt.Key.Key_Escape and self._open_key is not None:
            self._close_panel()
            return
        super().keyPressEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        if self._open_key is not None:
            self._open_panel(self._open_key)

    def _restyle(self) -> None:
        c = surface_colors()
        self.centralWidget().setStyleSheet(f"background: {c['band']};")
        for tile in self.tiles.values():
            tile.restyle()
        self.readout.restyle()
        for panel in self.panels.values():
            panel.restyle()

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def open_target(self, path: str) -> None:
        """Point the Hub at *path*, deciding what it is from its marker file."""
        path = str(Path(path).expanduser().resolve())
        try:
            if project_mod.is_project_dir(path):
                self._set_project(project_mod.Project(path))
            elif batch_mod.is_batch_dir(path):
                self._set_batch(path)
            elif project_mod.is_experiment_dir(path):
                ## A standalone Experiment Directory has no Project above it.
                ## Rather than refuse, treat its parent as the container and
                ## load it directly — the Hub is Project-first, not
                ## Project-only.
                self._set_batch(None)
                self._set_project(None)
                self._load_standalone(path)
            else:
                self._log_issue(
                    f"{path} is not a Batch, a Project, or an Experiment "
                    f"Directory.")
        except Exception as err:  # noqa: BLE001
            QMessageBox.critical(self, "Could not open", str(err))
            self._log_issue(f"ERROR: {err}")
        self.refresh()

    def _choose_batch(self) -> None:
        """The Batch panel's own way in: pick the parent directory, and every
        Project inside it loads into the table below."""
        path = QFileDialog.getExistingDirectory(self, "Choose a Batch folder")
        if not path:
            return
        self._set_batch(path)
        self._set_project(None)
        if self.batch_root is None:
            ## Not a Batch after all — say why, about the folder the user
            ## picked and using what the walk actually saw.  "No Project
            ## subdirectories" is the wrong answer now the search is recursive.
            found = self._scan_batch(path)
            if project_mod.is_project_dir(path) and not found["skipped"]:
                self._log_issue(
                    f"[batch] {os.path.basename(path)} is a single Project — "
                    "to batch it, choose a folder that contains it.")
            else:
                self._log_issue(
                    f"[batch] no Project found anywhere in {path} — a Project "
                    "is a folder with a project.yaml and at least one member "
                    "directory inside it.")
                for key, why in found["skipped"][:10]:
                    self._log_issue(f"[batch]   {key} — {why}")
                if found.get("truncated"):
                    self._log_issue(
                        "[batch]   the scan stopped early: this folder is "
                        "larger than a batch should be.  Choose one closer to "
                        "the projects.")
        self.refresh()

    def _choose_project(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose a Project folder")
        if not path:
            return
        try:
            self._set_project(project_mod.Project(path))
        except Exception as err:  # noqa: BLE001
            QMessageBox.critical(self, "Not a Project", str(err))
            self._log_issue(f"ERROR: {err}")
            return
        self.refresh()

    def _create_project(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Choose a folder to become a Project")
        if not path:
            return
        try:
            created = project_mod.create_project_file(path)
            self.log.append_line(f"Wrote {created}")
            self._set_project(project_mod.Project(path))
        except Exception as err:  # noqa: BLE001
            QMessageBox.critical(self, "Could not create Project", str(err))
            return
        self._invalidate_batch_scan()
        self.refresh()

    # ---- the cached recursive walk (ADR-0009) ------------------------

    def _scan_batch(self, path, refresh: bool = False) -> dict:
        """The recursive walk of *path*, cached.

        One walk per selection is kept and reused;
        :meth:`_invalidate_batch_scan` drops it whenever the tree may have
        changed underneath (a new selection, a filed recording, a scaffolded
        config, a finished run), and Rescan covers changes made outside the
        app entirely.
        """
        key = str(path)
        cached = self._batch_scan_cache
        if refresh or cached is None or cached[0] != key:
            cached = (key, batch_mod.discover(key))
            self._batch_scan_cache = cached
        return cached[1]

    def _invalidate_batch_scan(self) -> None:
        self._batch_scan_cache = None

    def _set_batch(self, path) -> None:
        """Select *path* as the working Batch — or clear the selection."""
        self._invalidate_batch_scan()
        self._noted_sheet = None
        self._noted_truncation = None
        if path is None:
            self.batch_root = None
            self._batch_panel_root = None
            return
        path = os.path.abspath(str(path))
        found = self._scan_batch(path)
        if not found["projects"]:
            ## Gated on the walk alone.  A short-circuit asking is_project_dir
            ## would disagree with the walk's stricter test: a batch root
            ## carrying a stray or legacy project.yaml would enumerate its
            ## Projects fine and show an empty, dead Batch panel.
            self.batch_root = None
            return
        self.batch_root = path
        self._batch_panel_root = path
        blocked = sum(len(p.blocked) for p in found["projects"])
        self.log.append_line(
            f"Batch: {path} — {len(found['projects'])} project(s)"
            + (f", {blocked} blocked member(s)" if blocked else ""))

    def _set_project(self, project) -> None:
        """Make *project* the working Project (or clear it).

        Unloads whatever member was loaded: a Project change makes the loaded
        experiment a stale copy of results from somewhere else, and a Hub that
        keeps showing it invites analysing one Project's member under another
        Project's design.
        """
        self.project = project
        self.experiment = None
        self.experiment_name = None
        if project is None:
            return
        ## The Hub holds a Batch and a Project at once — the readout answers
        ## "which batch, and which project inside it?" — but only while the
        ## Project is actually inside that Batch.  Opening an unrelated Project
        ## while a stale Batch table sat beside it was a standing invitation to
        ## run a batch nobody was looking at.
        if self.batch_root is not None:
            root = os.path.realpath(self.batch_root)
            here = os.path.realpath(project.project_directory)
            if os.path.commonpath([root, here]) != root or root == here:
                self.batch_root = None
        layouts = project.member_layouts()
        blocked = [item for item in layouts if item.blocked]
        self.log.append_line(
            f"Project: {project.project_directory} — "
            f"{len(project.member_names)} member(s)"
            + (f", {len(blocked)} blocked" if blocked else ""))
        for item in blocked:
            ## Named on selection, not only when a run fails on them: a Blocked
            ## Member is invisible to the Project's own membership test, so
            ## nothing else would ever mention it.
            self._log_issue(f"  blocked: {item.describe()}")
        for warning in project.warnings:
            self._log_issue(f"  note: {warning}")

    def _batch_view_root(self) -> str | None:
        """The Batch whose table is currently being shown.

        The app selection names exactly one container, but the Batch panel is
        allowed to stay open while a row double-click selects one of its
        Projects.  In that state the panel keeps displaying the Batch it came
        from instead of rebuilding itself as an empty Batch card.
        """
        if self.batch_root is not None:
            return self.batch_root
        remembered = self._batch_panel_root
        if remembered is None or self._open_key != "batch":
            return None
        if self._scan_batch(remembered)["projects"]:
            return remembered
        self._batch_panel_root = None
        return None

    def _batch_projects(self) -> list:
        root = self._batch_view_root()
        return self._scan_batch(root)["projects"] if root is not None else []

    def _batch_project(self, key):
        for item in self._batch_projects():
            if item.key == key:
                return item
        return None

    def _rescan_batch(self) -> None:
        root = self._batch_view_root()
        if root is None:
            return
        found = self._scan_batch(root, refresh=True)
        blocked = sum(len(p.blocked) for p in found["projects"])
        self.log.append_line(
            f"[batch] rescanned {root}: {len(found['projects'])} project(s)"
            + (f", {blocked} blocked member(s)" if blocked else ""))
        for key, why in found["skipped"]:
            self._log_issue(f"[batch] {key} skipped — {why}")
        self.refresh()

    def _batch_row_activated(self, item) -> None:
        """Double-clicking a project row is an ordinary selection change down to
        that Project — no drill-in state, no up-button.

        The Batch panel gives way to the Project panel, because the next thing
        anyone does after picking a Project is look at its members.
        """
        root = self._batch_view_root()
        cell = self.batch_table.item(item.row(), 0)
        if root is None or cell is None:
            return
        ## The row's text is the Project key — a path relative to the batch
        ## root, which may hold separators (ADR-0009).
        directory = batch_mod.project_directory(root, cell.text())
        try:
            self._set_project(project_mod.Project(directory))
        except Exception as err:  # noqa: BLE001
            QMessageBox.critical(self, "Not a Project", str(err))
            return
        self.refresh()
        self._open_panel("project")

    def _batch_table_menu(self, point) -> None:
        """Right-click on a project row: fix its blocked members (ADR-0009).

        It lives here because the other two gestures are taken — double-click
        selects the Project and a check runs it — and it opens the same review
        window Run batch does, so its buttons mean the same thing either way.
        """
        root = self._batch_view_root()
        if root is None:
            return
        item = self.batch_table.itemAt(point)
        if item is None:
            return
        cell = self.batch_table.item(item.row(), 0)
        if cell is None:
            return
        key = cell.text().strip()
        project = self._batch_project(key)
        menu = QMenu(self)
        fix = menu.addAction(f"Fix blocked members in {key}…")
        fix.setEnabled(bool(project is not None and project.blocked))
        if project is not None and project.blocked:
            fix.setToolTip("\n".join(m.describe() for m in project.blocked))
        open_action = menu.addAction(f"Open {key} as the Project…")
        chosen = menu.exec(self.batch_table.viewport().mapToGlobal(point))
        if chosen is fix:
            self._run_batch(focus=key)
        elif chosen is open_action:
            self._batch_row_activated(cell)

    def _on_batch_item_changed(self, item) -> None:
        """A check toggle changes what a Batch Run and the sheet would touch."""
        if item.column() == 0 and not getattr(self, "_filling_batch", False):
            self._refresh_batch_tile()

    def _batch_checked_keys(self) -> list[str]:
        keys = []
        for row in range(self.batch_table.rowCount()):
            item = self.batch_table.item(row, 0)
            if item is not None \
                    and item.checkState() == Qt.CheckState.Checked:
                keys.append(item.text())
        return keys

    def _project_row_activated(self, item) -> None:
        """Double-clicking a member loads it and shows the Analyze panel —
        loading is only ever a step toward doing something with it."""
        if self.project is None:
            return
        cell = self.project_table.item(item.row(), 0)
        if cell is None:
            return
        name = cell.text()
        blocked = {m.name: m for m in self.project.blocked_members()}
        if name in blocked:
            member = blocked[name]
            QMessageBox.information(
                self, "Blocked member",
                f"{member.name}: {member.detail or member.status}\n\n"
                + ("Use 'File unfiled recordings' below to move its DFM CSVs "
                   "into data/." if member.fix == "file"
                   else "Use 'Member configs…' below to give it a config."
                   if member.fix == "config"
                   else "Nothing here can be fixed automatically."))
            return
        self._load_member(name)
        self._open_panel("analyze")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_kwargs(self) -> dict:
        workers = self.spin_workers.value()
        return {"parallel": self.chk_parallel.isChecked(),
                "max_workers": workers or None}

    def _load_member(self, name: str) -> None:
        project = self.project
        if project is None:
            return

        def task():
            return project.load_experiment(name, **self._load_kwargs())

        def done(exp):
            self.experiment = exp
            self.experiment_name = name
            self.refresh()

        self._start(task, f"Loading member '{name}'", done)

    def _load_standalone(self, path: str) -> None:
        from .yaml_config import load_experiment_yaml

        def task():
            return load_experiment_yaml(path, **self._load_kwargs())

        def done(exp):
            self.experiment = exp
            self.experiment_name = os.path.basename(path)
            self.refresh()

        self._start(task, f"Loading {path}", done)

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    def _start(self, task, label: str, on_done=None) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "Busy",
                                    "A task is already running.")
            return
        self.log.append_line(f"\n=== {label} ===")
        self._set_running(True)
        worker = Worker(task, self)
        worker.line.connect(self._on_worker_line)
        worker.failed.connect(self._on_worker_failed)

        def finished(result):
            self._set_running(False)
            if on_done is not None:
                on_done(result)
            else:
                self._show_figures(result)
                self.refresh()

        worker.finished_ok.connect(finished)
        self._worker = worker
        worker.start()

    def _on_worker_line(self, text: str) -> None:
        """Stream a raw stdout chunk into the log.

        ``print`` writes its text and its terminator separately, so a chunk can
        carry several lines, a bare newline, or the front half of a line.
        Splitting on newlines and dropping the blanks — the old behaviour —
        threw away the vertical spacing every printed table relies on, and a
        progress line written without a newline did not appear until whatever
        came next flushed it.  :meth:`OutputLog.append_stream` keeps the line
        discipline instead.
        """
        chunk = str(text)
        self.log.append_stream(chunk)
        for line in chunk.splitlines():
            if any(mark in line for mark in
                   ("FAILED", "ERROR", "Traceback", "WARNING", "SKIPPED")):
                self.errors.append_line(line)

    def _on_worker_failed(self, message: str) -> None:
        self._set_running(False)
        self.log.append_line(message)
        self.errors.append_line(message)
        first = message.strip().splitlines()[-1] if message.strip() else "failed"
        QMessageBox.critical(self, "Task failed", first)

    def _log_issue(self, message: str) -> None:
        """Say it in both places: the Output log keeps the narrative in order,
        and the Errors tab keeps it findable after two thousand lines."""
        self.log.append_line(message)
        self.errors.append_line(message)

    def _set_running(self, running: bool) -> None:
        """Grey the open panel's cards in place rather than closing it — a task
        finishing should not move the UI out from under the user."""
        if self._open_key is not None:
            self.panels[self._open_key].setEnabled(not running)

    def _tabs_suppressed(self) -> bool:
        """Whether new figure/artifact tabs are being skipped for this run.

        The switch lives in the Batch panel because a Batch Run is what makes
        the tabs unbearable, but it governs every run: a 40-member "Analyze
        all" buries the Output tab just as thoroughly.
        """
        return bool(getattr(self, "chk_suppress_tabs", None)
                    and self.chk_suppress_tabs.isChecked())

    def _show_figures(self, result) -> None:
        if not isinstance(result, list):
            return
        if self._tabs_suppressed():
            shown = sum(1 for entry in result
                        if isinstance(entry, tuple) and len(entry) == 2)
            if shown:
                ## Never silent: "where did my plots go" is the question this
                ## switch creates, and the answer belongs in the log.
                self.log.append_line(
                    f"{shown} figure(s) not shown — 'Suppress new plot / "
                    "output tabs' is checked in the Batch panel.")
            return
        for entry in result:
            if isinstance(entry, tuple) and len(entry) == 2:
                title, figure = entry
                try:
                    self.dock.add_figure(str(title), figure)
                except Exception as err:  # noqa: BLE001
                    self._log_issue(f"could not show '{title}': {err}")

    # ---- experiment-level -------------------------------------------

    def _require_experiment(self) -> bool:
        if self.experiment is None:
            QMessageBox.information(
                self, "No member loaded",
                "Double-click a member in the Project panel to load it.")
            return False
        return True

    def _run_experiment_action(self, step: dict) -> None:
        if not self._require_experiment():
            return
        from .script_editor.runner import ScriptContext, run_experiment_script

        exp = self.experiment
        context = ScriptContext(binsize=self.spin_binsize.value())
        script = {"name": "hub", "steps": [dict(step)]}

        def task():
            return run_experiment_script(exp, script, context=context)

        self._start(task, step["action"])

    def _run_plot_action(self, action: str) -> None:
        if not self._require_experiment():
            return
        step = {"action": action}
        if action in ("plot_binned", "plot_dot"):
            metric = self.plot_metric.currentData()
            if metric:
                step["metric"] = metric
        self._run_experiment_action(step)

    def _run_experiment_script(self) -> None:
        if not self._require_experiment():
            return
        name = self.experiment_script.currentText()
        if not name:
            return
        recipe = self._find_experiment_script(name)
        if recipe is None:
            QMessageBox.information(self, "Not found",
                                    f"No Experiment Script named '{name}'.")
            return
        from .script_editor.runner import ScriptContext, run_experiment_script

        exp = self.experiment
        context = ScriptContext(binsize=self.spin_binsize.value())

        def task():
            return run_experiment_script(exp, recipe, context=context)

        self._start(task, f"Experiment Script '{name}'")

    def _find_experiment_script(self, name: str) -> dict | None:
        if self.project is not None:
            central = self.project.find_experiment_script(name)
            if central is not None:
                return central
        config = getattr(self.experiment, "config", None) or {}
        for script in (config.get("scripts") or []):
            if isinstance(script, dict) and script.get("name") == name:
                return script
        return None

    # ---- project-level ----------------------------------------------

    def _require_project(self) -> bool:
        if self.project is None:
            QMessageBox.information(
                self, "No Project", "Open a Project first.")
            return False
        return True

    def _run_all_members(self) -> None:
        if not self._require_project():
            return
        project = self.project

        def task():
            failures = project.run_all()
            if failures:
                raise RuntimeError("; ".join(failures))
            return None

        self._start(task, "Analyze all members")

    def _build_combined(self) -> None:
        if not self._require_project():
            return
        project = self.project
        self._start(lambda: project.build_combined_analysis(),
                    "Build combined analysis", lambda _r: self.refresh())

    def _project_report(self) -> None:
        if not self._require_project():
            return
        from .project_report import write_project_report

        project = self.project
        self._start(lambda: write_project_report(project),
                    "Create project report", lambda _r: self.refresh())

    def _render_figures(self) -> None:
        if not self._require_project():
            return
        from . import pubfigures

        project = self.project
        self._start(lambda: pubfigures.render_all(project),
                    "Render publication figures", lambda _r: self.refresh())

    def _run_project_script(self) -> None:
        if not self._require_project():
            return
        from .script_editor.project_actions import builtin_project_script
        from .script_editor.project_runner import run_project_script

        name = self.project_script.currentText()
        project = self.project
        script = project.find_script(name) or builtin_project_script(name)
        if script is None:
            QMessageBox.information(self, "Not found",
                                    f"No Project Script named '{name}'.")
            return
        self._start(lambda: run_project_script(project, script),
                    f"Project Script '{name}'", lambda _r: self.refresh())

    def _reload_project(self) -> None:
        """Re-read the Project from disk after something changed its members."""
        if self.project is None:
            return
        directory = self.project.project_directory
        try:
            self._set_project(project_mod.Project(directory))
        except Exception as err:  # noqa: BLE001
            self._log_issue(f"could not reload {directory}: {err}")
        self._invalidate_batch_scan()
        self.refresh()

    def _file_unfiled(self) -> None:
        """File every Unfiled Recording in this Project (ADR-0009).

        The one repair that is safe in bulk: it only ever *moves* files inside
        the member directory that already holds them, never overwrites, and
        refuses outright where the answer is ambiguous.
        """
        if not self._require_project():
            return
        targets = [item for item in self.project.member_layouts()
                   if item.fix == "file"]
        if not targets:
            QMessageBox.information(
                self, "Nothing to file",
                "Every member's DFM CSVs are already in its data/ folder.")
            return
        confirm = QMessageBox.question(
            self, "File unfiled recordings",
            f"Move the DFM CSVs into data/ in {len(targets)} member "
            "director(ies)?\n\nEvery other loose file goes to extra_files/.  "
            "YAML files and remove_chambers.csv stay where they are, and "
            "nothing is overwritten.")
        if confirm != QMessageBox.StandardButton.Yes:
            return
        for member in targets:
            plan = layout_mod.file_recording(member.directory,
                                             log=self.log.append_line)
            if plan.refused:
                self._log_issue(f"[file] {member.name}: {plan.refused}")
            else:
                self.log.append_line(f"[file] {member.name}: {plan.describe()}")
            for name, why in plan.skipped:
                self._log_issue(f"[file] {member.name}: {name} skipped — {why}")
        self._reload_project()

    def _open_member_configs(self) -> None:
        """The one design-aware scaffolding path (ADR-0009).

        A dialog rather than a bulk button: scaffolding copies one member's
        ``dfms:`` block into another, and which member it copies from is a
        decision worth showing before it is made.
        """
        if not self._require_project():
            return
        MemberConfigsDialog(self, self.project).exec()
        self._reload_project()

    def _view_reports(self) -> None:
        """Open the Project Report and each member's own report."""
        if not self._require_project():
            return
        project = self.project
        paths = [Path(project.project_directory) / f"{project.name}_report.pdf"]
        paths += [Path(project.member_dir(name)) / f"{name}_report.pdf"
                  for name in project.member_names]
        found = [path for path in paths if path.is_file()]
        if not found:
            QMessageBox.information(
                self, "No reports yet",
                "Nothing to open — run 'Create report' first.")
            return
        for path in found:
            self._open_externally(path)

    def _open_externally(self, path: Path) -> bool:
        """Hand *path* to the system viewer or file browser; report rather than
        raise — a missing xdg-open must not take a report button down with it."""
        import subprocess

        try:
            if sys.platform.startswith("darwin"):
                subprocess.Popen(["open", str(path)])
            elif os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as err:  # noqa: BLE001
            self._log_issue(f"could not open {path}: {err}")
            return False
        self.log.append_line(f"Opened {path}")
        return True

    # ---- batch -------------------------------------------------------

    def _run_batch(self, focus=None) -> None:
        root = self._batch_view_root()
        if root is None:
            QMessageBox.information(self, "No Batch",
                                    "Open a Batch folder first.")
            return
        ## The preflight is where the target list is confirmed and blocked
        ## members are repaired (ADR-0009).  Always shown: with recursive
        ## discovery the folder you picked no longer says what will run.
        confirmed = self._open_batch_preflight(root, focus=focus)
        if confirmed is None:
            return
        checked, apply_exclusions = confirmed
        if not checked:
            QMessageBox.information(self, "Nothing checked",
                                    "Check at least one Project row.")
            return
        name = self.batch_script.currentData()

        ## The provider is chosen BEFORE the run: the narrative is written from
        ## the worker thread, which cannot raise a dialog, and finding out
        ## there is no API key after an hour of analysis is no use.
        provider = None
        if self.chk_batch_narrative.isChecked():
            provider = self._choose_ai_provider()
            if provider is None:
                return

        ## A Batch Run rewrites every member's analysis in every Project — a
        ## loaded experiment would survive as a stale copy of results that no
        ## longer exist.
        self.experiment = None
        self.experiment_name = None

        def task():
            results = batch_mod.run_batch(
                root, script_name=name, project_names=checked, log=print,
                apply_exclusions=apply_exclusions)
            if provider is not None:
                ## Only the Projects that actually ran: summarizing one that
                ## just failed would describe stale numbers as fresh ones.
                ran = [k for k, v in results.items() if v == "ok"]
                self._write_batch_narrative(root, provider, ran)
            failed = [k for k, v in results.items() if v != "ok"]
            if failed:
                raise RuntimeError(
                    f"{len(failed)} of {len(results)} Project(s) failed: "
                    + ", ".join(failed[:6]))
            return None

        def done(_result):
            self._invalidate_batch_scan()
            self.refresh()

        self._start(task, f"Batch Run in {os.path.basename(root)}", done)

    def _open_batch_preflight(self, root, focus=None):
        """Show the preflight for *root*; returns ``(keys, apply_exclusions)``
        when the user chose to run, or None when they cancelled.

        Opened from Run Batch and from the table's right-click fix entry — the
        same dialog either way, so there is one place that states what a Batch
        Run is about to do, and its Run button means the same thing from both.
        """
        from .batch_preflight import BatchPreflightDialog

        dialog = BatchPreflightDialog(self, root,
                                      checked=self._batch_checked_keys(),
                                      log=self.log.append_line)
        if focus is not None:
            dialog.focus_project(focus)
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        ## Filing or scaffolding inside the dialog changes the tree, so the
        ## cached walk is stale either way.
        self._invalidate_batch_scan()
        self.refresh()
        if not accepted:
            return None
        return dialog.selected_keys, dialog.apply_exclusions

    def _on_batch_script_changed(self, _index: int) -> None:
        root = self._batch_view_root()
        if root is None or getattr(self, "_filling_batch", False):
            return
        try:
            batch_mod.save_batch_designation(
                root, self.batch_script.currentData())
        except Exception as err:  # noqa: BLE001
            self._log_issue(f"[batch] could not save the designation: {err}")

    def _choose_ai_provider(self) -> str | None:
        """Ask which configured provider to use, or None when unavailable or
        cancelled (the caller then does nothing)."""
        from . import ai

        available = ai.available_providers()
        if not available:
            QMessageBox.information(
                self, "No AI provider",
                "No API key found.  Set ANTHROPIC_API_KEY or OPENAI_API_KEY "
                "in your environment or a .env file.")
            return None
        label = self.ai_provider.currentText()
        for provider in available:
            if provider.display_name == label:
                return provider.provider_name
        return available[0].provider_name

    def _write_batch_narrative(self, root, provider: str,
                               ran: list[str]) -> None:
        """Synthesize the ran Projects' narratives into one at the Batch root.

        Runs on the worker thread, so it reports through ``print`` and never
        raises: a failed narrative must not turn a successful overnight batch
        into a failed one.
        """
        from . import ai

        try:
            ai.generate_batch_narrative(root, provider, project_keys=ran)
        except Exception as err:  # noqa: BLE001
            print(f"[ai] batch narrative failed: {err}")

    def _apply_exclusion_sheet(self, root, label: str,
                               projects=None) -> None:
        """Write the Exclusion Sheet at *root* into each member's own file.

        Explicit, never automatic outside a Batch Run: selecting a folder
        reports its sheet, it never applies one — browsing to a colleague's
        batch must not rewrite eighty directories (ADR-0010).
        """
        from . import exclusion_sheet

        if root is None:
            QMessageBox.information(self, "Nothing selected",
                                    f"Open a {label} first.")
            return
        sheet = exclusion_sheet.find_sheet(str(root))
        if sheet is None:
            QMessageBox.information(
                self, "No exclusion sheet",
                f"No remove_chambers.csv at the {label}.\n\nColumns: "
                "project, member, dfm, chamber, group, reason.")
            return
        confirm = QMessageBox.question(
            self, "Apply exclusion sheet",
            f"Write the rows of {os.path.basename(sheet)} into each member's "
            "own remove_chambers.csv?\n\nDeclarations already in place are "
            "kept; a differing note is reported as a conflict rather than "
            "overwritten.")
        if confirm != QMessageBox.StandardButton.Yes:
            return
        result = batch_mod.apply_exclusion_sheet(
            str(root), log=self.log.append_line, projects=projects)
        for note in result.get("failed") or []:
            self._log_issue(f"[exclusions] could not write {note}")
        self.refresh()

    # ---- AI ----------------------------------------------------------

    def _refresh_ai_models(self, provider_name: str) -> None:
        from . import ai

        self.ai_model.clear()
        for provider in ai.PROVIDERS:
            if provider.display_name == provider_name:
                self.ai_model.addItems(list(provider.models))
                break

    def _generate_narrative(self) -> None:
        if not self._require_project():
            return
        from . import ai

        label = self.ai_provider.currentText()
        provider_name = ""
        for provider in ai.PROVIDERS:
            if provider.display_name == label:
                provider_name = provider.provider_name
        if not provider_name:
            QMessageBox.information(
                self, "No provider",
                "No AI provider is configured. Add an API key to .env.")
            return
        project = self.project
        model = self.ai_model.currentText() or None
        self._start(
            lambda: ai.generate_project_narrative(project, provider_name, model),
            "Generate AI narrative", lambda _r: self.refresh())

    # ---- tools -------------------------------------------------------

    def _current_dir(self) -> str | None:
        if self.experiment is not None and self.experiment.experiment_dir:
            return str(self.experiment.experiment_dir)
        if self.project is not None:
            return self.project.project_directory
        if self.batch_root is not None:
            return self.batch_root
        return None

    def _open_config_editor(self) -> None:
        from .config_editor import FLICConfigEditor

        directory = self._current_dir()
        config = os.path.join(directory, "flic_config.yaml") if directory else None
        self._config_editor = FLICConfigEditor(
            config if config and os.path.isfile(config) else None)
        self._config_editor.show()

    def _open_qc_viewer(self) -> None:
        if not self._require_experiment():
            return
        from .qc_viewer import MainWindow as QCViewerWindow

        self._qc = QCViewerWindow(Path(self.experiment.experiment_dir))
        self._qc.show()

    def _open_script_editor(self) -> None:
        from .script_editor import ScriptEditorWindow

        ## The editor edits one yaml. A Project's own scripts live in
        ## project.yaml, a member's in its flic_config.yaml — so the file to
        ## open follows what is loaded, not what is merely selected.
        if self.experiment is not None:
            config = os.path.join(str(self.experiment.experiment_dir),
                                  "flic_config.yaml")
        elif self.project is not None:
            config = os.path.join(self.project.project_directory,
                                  "project.yaml")
        else:
            QMessageBox.information(
                self, "Nothing to edit",
                "Open a Project, or load a member, first.")
            return
        self._script_editor = ScriptEditorWindow(config)
        self._script_editor.show()

    def _open_plot_editor(self) -> None:
        if not self._require_project():
            return
        from .plot_editor import PlotEditorWindow

        self._plot_editor = PlotEditorWindow(self.project.project_directory)
        self._plot_editor.show()

    def _run_lint(self) -> None:
        directory = self._current_dir()
        if not directory:
            QMessageBox.information(self, "Nothing selected",
                                    "Open a Batch, Project, or member first.")
            return
        from .migration_lint import check_tree
        from .yaml_lint import lint_flic_config

        def task():
            for config in sorted(Path(directory).rglob("flic_config.yaml")):
                for issue in lint_flic_config(config):
                    print(issue.format(config))
            issues = check_tree(Path(directory))
            if issues:
                print("\n--- migration (ADR-0005..0008) ---")
                for issue in issues:
                    print(issue.format())
            else:
                print("No migration problems found.")
            return None

        self._start(task, f"Lint {directory}")

    def _yaml_validation_targets(self) -> list[Path]:
        """Every YAML this selection is responsible for, widest first.

        A Batch has one per Project plus one per member; a Project has its own
        plus its members'.  The point is to find the one bad file *before* an
        unattended run does — which means checking the ones the user never
        opens, not only the one they are looking at.
        """
        root = self._current_dir()
        if not root:
            return []
        base = Path(root)
        found: list[Path] = []
        for name in (project_mod.PROJECT_FILENAME, batch_mod.BATCH_FILENAME,
                     project_mod.CONFIG_FILENAME):
            candidate = base / name
            if candidate.is_file():
                found.append(candidate)
        ## rglob, because a Batch's Projects sit at arbitrary depth (ADR-0009).
        for pattern in (f"*/**/{project_mod.PROJECT_FILENAME}",
                        f"*/**/{project_mod.CONFIG_FILENAME}",
                        f"*/**/{batch_mod.BATCH_FILENAME}"):
            found.extend(sorted(base.glob(pattern)))
        seen: list[Path] = []
        for path in found:
            if path not in seen:
                seen.append(path)
        return seen

    def _validate_yaml(self) -> None:
        """Parse every YAML under the selection and report what fails.

        Cheap, read-only, and the only way to learn that a hand-edited config
        three folders down is unparseable without waiting for hour three of a
        Batch Run to say so.
        """
        import yaml as _yaml

        targets = self._yaml_validation_targets()
        if not targets:
            QMessageBox.information(self, "Nothing selected",
                                    "Open a Batch, Project, or member first.")
            return
        root = Path(self._current_dir())

        def task():
            bad = 0
            for path in targets:
                label = path.relative_to(root) if path.is_relative_to(root) \
                    else path
                try:
                    with open(path, encoding="utf-8") as handle:
                        loaded = _yaml.safe_load(handle)
                except Exception as err:  # noqa: BLE001
                    bad += 1
                    print(f"ERROR {label}: {err}")
                    continue
                if loaded is not None and not isinstance(loaded, dict):
                    bad += 1
                    print(f"ERROR {label}: top level is a "
                          f"{type(loaded).__name__}, not a mapping")
            print(f"\nChecked {len(targets)} YAML file(s) — "
                  + (f"{bad} problem(s)." if bad else "all parse cleanly."))
            return None

        self._start(task, f"Validate YAML under {root.name}")

    def _open_folder(self) -> None:
        """Open the selected directory in the system file browser."""
        directory = self._current_dir()
        if not directory:
            QMessageBox.information(self, "Nothing selected",
                                    "Open a Batch, Project, or member first.")
            return
        self._open_externally(Path(directory))

    def _clear_cache(self) -> None:
        directory = self._current_dir()
        if not directory:
            return
        from . import cache

        removed = cache.clear(Path(directory))
        self.log.append_line(f"Removed {removed} cache file(s) from {directory}")

    def _toggle_theme(self) -> None:
        from .ui import resolved_mode

        new_mode = "light" if resolved_mode() == "dark" else "dark"
        apply_theme(QApplication.instance(), mode=new_mode)
        ui_settings.set_value("theme", new_mode)
        self._restyle()

    def _open_help(self) -> None:
        try:
            from pyflic.help import open_help

            open_help(None)
        except Exception as err:  # noqa: BLE001
            QMessageBox.information(self, "Help unavailable", str(err))

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        self._refresh_batch()
        self._refresh_project()
        self._refresh_experiment_tiles()
        self._refresh_ai()
        self._refresh_tools_tile()
        self._refresh_card_dimming()
        self._refresh_readout()

    # ---- batch -------------------------------------------------------

    def _refresh_batch(self) -> None:
        from . import exclusion_sheet
        from .script_editor.project_actions import BUILTIN_PROJECT_SCRIPTS

        root = self._batch_view_root()
        live = root is not None
        self.batch_empty.setVisible(not live)
        ## Deliberately NOT in this list: the suppress-tabs box applies to
        ## every run, not only Batch Runs, so it stays usable with no Batch
        ## selected.
        for widget in (self.batch_table, self.batch_script,
                       self.batch_run_btn, self.chk_batch_narrative,
                       self.batch_rescan_btn):
            widget.setEnabled(live)

        if not live:
            self._filling_batch = True
            try:
                self.batch_table.setRowCount(0)
                self.batch_script.clear()
            finally:
                self._filling_batch = False
            self.batch_sheet_btn.setEnabled(False)
            self._refresh_batch_tile()
            return

        ## Selecting a Batch REPORTS its Exclusion Sheet; it never applies one
        ## (ADR-0010).
        sheet = exclusion_sheet.find_sheet(root)
        self.batch_sheet_btn.setEnabled(sheet is not None)
        if sheet is not None and self._noted_sheet != sheet:
            self._noted_sheet = sheet
            try:
                rows = len(exclusion_sheet.read_sheet(sheet))
                self.log.append_line(
                    f"[exclusions] {os.path.basename(sheet)} found: {rows} "
                    "row(s).  'Apply exclusion sheet…' writes them into the "
                    "members; a Batch Run applies it automatically.")
            except Exception as err:  # noqa: BLE001
                self._log_issue(f"[exclusions] {os.path.basename(sheet)} "
                                f"could not be read: {err}")

        found = self._scan_batch(root)
        projects = found["projects"]
        if found.get("truncated") and self._noted_truncation != str(root):
            ## Once per selection: a partial scan reported as a complete one is
            ## how an unattended run silently skips half a batch.
            self._noted_truncation = str(root)
            self._log_issue(
                f"[batch] the scan of {root} stopped early — this folder is "
                "larger than a batch should be, and projects deeper in it were "
                "not found.  Choose a folder closer to the projects.")

        ## Rebuilding must not silently re-check a Project the user unchecked;
        ## a new row defaults to checked unless nothing in it can run, which
        ## can only produce a failure (ADR-0009).
        previous: dict[str, Qt.CheckState] = {}
        for row in range(self.batch_table.rowCount()):
            cell = self.batch_table.item(row, 0)
            if cell is not None:
                previous[cell.text()] = cell.checkState()

        self._filling_batch = True
        try:
            self.batch_table.setRowCount(0)
            for row, project in enumerate(projects):
                self.batch_table.insertRow(row)
                cell = QTableWidgetItem(project.key)
                cell.setFlags(cell.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                default = (Qt.CheckState.Checked if project.runnable
                           else Qt.CheckState.Unchecked)
                cell.setCheckState(previous.get(project.key, default))
                cell.setToolTip(project.key)
                self.batch_table.setItem(row, 0, cell)
                blocked = project.blocked
                for column, text in enumerate(
                        [f"{len(project.usable)}/{len(project.members)}",
                         "yes" if project.has_report else "no",
                         f"{len(blocked)} blocked" if blocked else "ok"],
                        start=1):
                    self.batch_table.setItem(row, column,
                                             QTableWidgetItem(text))
                if blocked:
                    ## Red, and the reasons in the tooltip: the strip has no
                    ## room for them and the preflight is a click away.
                    brush = QBrush(QColor(blocked_color()))
                    detail = "\n".join(m.describe() for m in blocked)
                    for column in range(self.batch_table.columnCount()):
                        item = self.batch_table.item(row, column)
                        if item is not None:
                            item.setForeground(brush)
                            item.setToolTip(f"{project.key}\n\n{detail}"
                                            if column == 0 else detail)

            ## Picker: the per-Project default, the built-ins, then batch.yaml's
            ## central Project Scripts.  A designation naming a script only some
            ## project.yaml defines gets an entry rather than vanishing.
            meta = batch_mod.load_batch_file(root)
            self.batch_script.clear()
            self.batch_script.addItem(
                f"Each project's own '{batch_mod.DEFAULT_SCRIPT_NAME}' script "
                "(default)", None)
            for name in BUILTIN_PROJECT_SCRIPTS:
                self.batch_script.addItem(f"{name} (built-in)", name)
            for script in meta["project_scripts"]:
                name = str(script.get("name"))
                self.batch_script.addItem(f"{name} (batch.yaml)", name)
            want = meta["script"]
            index = 0
            if want:
                index = self.batch_script.findData(want)
                if index < 0:
                    self.batch_script.addItem(f"{want} (from each project)",
                                              want)
                    index = self.batch_script.count() - 1
            self.batch_script.setCurrentIndex(index)
        finally:
            self._filling_batch = False
        self._refresh_batch_tile()

    def _refresh_batch_tile(self) -> None:
        """The Batch tile is always lit (ADR-0009).

        It used to dim with no Batch open, which read as "unavailable" — but
        choosing a batch folder is precisely what its panel is for, so the one
        tile that could fix the empty state was the one that looked closed.
        """
        tile = self.tiles["batch"]
        tile.set_dimmed(False)
        root = self._batch_view_root()
        if root is None:
            tile.set_summary(["no batch open", "choose a folder of Projects"])
            return
        projects = self._batch_projects()
        checked = len(self._batch_checked_keys())
        blocked = sum(len(p.blocked) for p in projects)
        second = f"{checked} checked"
        if blocked:
            second += f" · {blocked} blocked"
        tile.set_summary([f"{len(projects)} project(s)", second])

    # ---- project -----------------------------------------------------

    def _refresh_project(self) -> None:
        from . import exclusion_sheet
        from .script_editor.project_actions import BUILTIN_PROJECT_SCRIPTS

        tile = self.tiles["project"]
        table = self.project_table
        table.setRowCount(0)
        self.project_script.clear()
        ## Always lit, for the same reason the Batch tile is: its panel holds
        ## "Open a Project…", the control that fixes the missing state.
        tile.set_dimmed(False)
        if self.project is None:
            tile.set_summary(["no project open", "open one to begin"])
            for widget in (self.file_btn, self.scaffold_btn,
                           self.view_reports_btn, self.project_sheet_btn):
                widget.setEnabled(False)
            return
        project = self.project

        ## One classification pass feeds the table, the buttons and the tile:
        ## the Project's membership test asks only "is there a config", and a
        ## run asks the harder question (ADR-0009).
        layouts = {item.name: item for item in project.member_layouts()}
        analyzed = stale = 0
        row = 0
        for name in project.member_names:
            status = project.member_status(name)
            analyzed += 1 if status["analyzed"] else 0
            stale += 1 if status["stale"] else 0
            table.insertRow(row)
            table.setItem(row, 0, QTableWidgetItem(name))
            analyzed_text = "no"
            if status["analyzed"]:
                ## The stale rule (ADR-0010): results computed before the
                ## current exclusion declaration describe a population nobody
                ## asked for, so they read as needing a re-run rather than as
                ## an unqualified yes.
                analyzed_text = "re-run needed" if status["stale"] else "yes"
            for column, text in enumerate(
                    [str(status["dfms"]),
                     str(status["chambers"])
                     if status["chambers"] is not None else "—",
                     analyzed_text,
                     "yes" if status["report"] else "no"], start=1):
                table.setItem(row, column, QTableWidgetItem(text))
            item = layouts.get(name)
            if item is not None and item.blocked:
                self._paint_blocked_row(table, row, item)
            row += 1

        ## Blocked folders that are not Members yet (no config, or an unfiled
        ## recording) are listed too: they are invisible to the Project and
        ## would otherwise fail silently at run time.
        for item in layouts.values():
            if item.name in project.member_names or not item.blocked:
                continue
            table.insertRow(row)
            table.setItem(row, 0, QTableWidgetItem(item.name))
            for column, text in enumerate(
                    [str(len(item.dfm_ids) or "—"), "—", "—", "—"], start=1):
                table.setItem(row, column, QTableWidgetItem(text))
            self._paint_blocked_row(table, row, item)
            row += 1

        unfiled = [i for i in layouts.values() if i.fix == "file"]
        pending = project.unconfigured_dirs()
        self.file_btn.setEnabled(bool(unfiled))
        self.file_btn.setText(
            f"File {len(unfiled)} unfiled recording(s)" if unfiled
            else "File unfiled recordings")
        self.scaffold_btn.setEnabled(True)
        self.scaffold_btn.setText(
            f"Member configs… ({len(pending)} missing)" if pending
            else "Member configs…")
        self.view_reports_btn.setEnabled(True)
        self.project_sheet_btn.setEnabled(
            exclusion_sheet.find_sheet(project.project_directory) is not None)

        blocked = sum(1 for i in layouts.values() if i.blocked)
        summary = [f"{len(project.member_names)} member(s)",
                   f"{analyzed} analyzed"]
        if blocked:
            summary[1] = f"{analyzed} analyzed · {blocked} blocked"
        elif stale:
            summary[1] = f"{analyzed} analyzed · {stale} stale"
        if self.experiment_name:
            summary[1] = f"loaded: {self.experiment_name}"
        tile.set_summary(summary)

        names = [s["name"] for s in project.scripts]
        names += [n for n in BUILTIN_PROJECT_SCRIPTS if n not in names]
        self.project_script.addItems(names)

    @staticmethod
    def _paint_blocked_row(table, row: int, item) -> None:
        """Red row + the reason in every cell's tooltip.

        Colour alone would say "something is wrong" without saying what; the
        reason is the part that lets someone fix it, and the row has no space
        for it.
        """
        brush = QBrush(QColor(blocked_color()))
        detail = item.detail or item.status
        for column in range(table.columnCount()):
            cell = table.item(row, column)
            if cell is not None:
                cell.setForeground(brush)
                cell.setToolTip(f"{item.name}: {item.status}\n{detail}")

    def _refresh_experiment_tiles(self) -> None:
        loaded = self.experiment is not None
        for key in ("analyze", "plots"):
            self.tiles[key].set_dimmed(not loaded)
        if not loaded:
            for tile_key, hint in (("analyze", self.analyze_hint),
                                   ("plots", self.plots_hint)):
                hint.setText("No member is loaded. Double-click a member "
                             "row in the Project panel to load one.")
                self.tiles[tile_key].set_summary(["no member loaded", ""])
            self.plot_metric.clear()
            self.experiment_script.clear()
            self.tiles["scripts"].set_dimmed(self.project is None)
            self._refresh_scripts_tile()
            return

        exp = self.experiment
        layout = getattr(exp, "chamber_layout", None) or "two_well"
        type_name = getattr(getattr(exp, "experiment_type", None), "name", "Custom")
        self.analyze_hint.setText(
            f"Loaded: {self.experiment_name} — {type_name} / {layout}")
        self.plots_hint.setText(self.analyze_hint.text())
        self.tiles["analyze"].set_summary([str(self.experiment_name),
                                           f"{type_name} · {layout}"])
        facets = len(exp.facet_windows())
        self.tiles["plots"].set_summary(
            [str(self.experiment_name),
             f"{facets} facet(s)" if facets else "no facets"])

        from .metrics import binned_metrics

        current = self.plot_metric.currentData()
        self.plot_metric.clear()
        for label, metric, _mode in binned_metrics(layout):
            self.plot_metric.addItem(label, metric)
        if current:
            index = self.plot_metric.findData(current)
            if index >= 0:
                self.plot_metric.setCurrentIndex(index)

        self.experiment_script.clear()
        names: list[str] = []
        if self.project is not None:
            names += [s["name"] for s in self.project.experiment_scripts]
        for script in ((getattr(exp, "config", None) or {}).get("scripts") or []):
            if isinstance(script, dict) and script.get("name") not in names:
                names.append(script.get("name"))
        self.experiment_script.addItems([n for n in names if n])
        self.tiles["scripts"].set_dimmed(False)
        self._refresh_scripts_tile()

    def _refresh_scripts_tile(self) -> None:
        project_count = self.project_script.count()
        experiment_count = self.experiment_script.count()
        self.tiles["scripts"].set_summary([
            f"{project_count} project script(s)",
            f"{experiment_count} experiment script(s)",
        ])

    def _refresh_tools_tile(self) -> None:
        from .ui import resolved_mode

        where = self._current_dir()
        self.tiles["tools"].set_summary([
            f"theme: {resolved_mode()}",
            os.path.basename(where) if where else "nothing selected",
        ])

    def _refresh_ai(self) -> None:
        from . import ai

        available = ai.available_providers()
        tile = self.tiles["ai"]
        current = self.ai_provider.currentText()
        self.ai_provider.clear()
        self.ai_provider.addItems([p.display_name for p in available])
        if current:
            self.ai_provider.setCurrentText(current)
        if not available:
            tile.set_dimmed(True)
            tile.set_summary(["no provider key", "add one to .env"])
            self.ai_hint.setText(
                "No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in "
                "your environment or a .env file, then reopen this panel.")
            return
        tile.set_dimmed(self.project is None)
        saved = (ai.read_project_narrative(self.project)
                 if self.project is not None else None)
        tile.set_summary([f"{len(available)} provider(s)",
                          "narrative saved" if saved else "no narrative yet"])
        self.ai_hint.setText(
            "The narrative is a derivative of one Combined Analysis: "
            "rebuilding the analysis deletes it.")

    def _refresh_card_dimming(self) -> None:
        """Dim the cards whose actions have no subject yet.

        The strip already says which tiles are inapplicable; a panel that opens
        looking exactly as live as a working one undoes that the moment it is
        opened.  Dimming is presentation only — every card stays clickable,
        because a dimmed card is precisely the one holding the control that
        fixes the missing state (ADR-0007).
        """
        has_project = self.project is not None
        has_experiment = self.experiment is not None
        dim = {
            "batch": False,          # its panel holds "Choose batch folder…"
            "project": False,        # its panel holds "Open a Project…"
            "analyze": not has_experiment,
            "plots": not has_experiment,
            "scripts": not (has_project or has_experiment),
            "ai": not has_project,
            "tools": False,
        }
        for key, panel in self.panels.items():
            for card in panel.cards():
                card.set_dimmed(dim.get(key, False))

    def _refresh_readout(self) -> None:
        rows: list[tuple[str, str]] = []
        root = self._batch_view_root()
        if root is not None:
            rows.append(("Batch", f"{os.path.basename(root)} "
                                  f"({len(self._batch_projects())} projects)"))
        if self.project is not None:
            rows.append(("Project", f"{self.project.name} "
                                    f"({len(self.project.experiment_names)} members)"))
            rows.append(("Design", f"{self.project.experiment_type.display_name} · "
                                   f"{self.project.chamber_layout}"))
        else:
            rows.append(("Project", "none open"))
        rows.append(("Loaded", self.experiment_name or "no member loaded"))
        self.readout.set_rows(rows)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self._click_filter)
        super().closeEvent(event)


class MemberConfigsDialog(QDialog):
    """Give a Project's unconfigured folders a design-conformant config.

    The **one** design-aware scaffolding path (ADR-0009): the Project panel
    opens it, and so does the Batch preflight when a Blocked Member's reason is
    "no config".  A second, batch-only scaffolder would drift from this one, and
    the thing it writes — a member's ``dfms:`` block — is hand-made work nobody
    wants written twice differently.

    Scaffolding never overwrites: a folder that already has a config is listed
    as such and offers Edit instead.
    """

    def __init__(self, hub, project) -> None:
        super().__init__(hub)
        self._hub = hub
        self._project = project
        self.setWindowTitle(f"Member configs — {project.name}")
        self.setMinimumSize(560, 420)

        outer = QVBoxLayout(self)
        outer.setSpacing(10)
        heading = QLabel(
            "Every folder in this Project that holds DFM CSVs.  A folder with "
            "no flic_config.yaml is scaffolded from an existing member's "
            "dfms: block and reconciled against the DFMs actually in its "
            "data/ — ids with no entry are added with chambers unassigned, "
            "and entries with no data are flagged rather than dropped.")
        heading.setWordWrap(True)
        outer.addWidget(heading)

        self._list = QListWidget()
        self._list.itemDoubleClicked.connect(self._on_double_click)
        self._list.currentItemChanged.connect(lambda *_a: self._sync())
        outer.addWidget(self._list, 1)

        row = QHBoxLayout()
        self._btn_create = ActionButton("Create config", Category.LOAD, "new",
                                        primary=True)
        self._btn_create.clicked.connect(self._create_selected)
        row.addWidget(self._btn_create)
        self._btn_all = ActionButton("Create every missing config",
                                     Category.LOAD, "new")
        self._btn_all.clicked.connect(self._create_all_missing)
        row.addWidget(self._btn_all)
        self._btn_edit = ActionButton("Edit config…", Category.TOOLS, "config")
        self._btn_edit.clicked.connect(self._edit_selected)
        row.addWidget(self._btn_edit)
        row.addStretch(1)
        outer.addLayout(row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        outer.addWidget(buttons)

        self._reload()

    # ------------------------------------------------------------------

    def _reload(self) -> None:
        self._list.clear()
        for item in self._project.member_layouts():
            configured = item.configured
            label = f"{item.name} — " + (
                "configured" if configured else "no flic_config.yaml")
            if item.blocked:
                label += f" ({item.status})"
            entry = QListWidgetItem(label)
            entry.setData(Qt.ItemDataRole.UserRole, item.name)
            entry.setData(Qt.ItemDataRole.UserRole + 1, configured)
            if item.blocked:
                entry.setForeground(QBrush(QColor(blocked_color())))
                entry.setToolTip(item.detail or item.status)
            self._list.addItem(entry)
        self._sync()

    def _selected(self) -> tuple[str, bool] | None:
        item = self._list.currentItem()
        if item is None:
            return None
        return (item.data(Qt.ItemDataRole.UserRole),
                bool(item.data(Qt.ItemDataRole.UserRole + 1)))

    def _missing(self) -> list[str]:
        return self._project.unconfigured_dirs()

    def _sync(self) -> None:
        chosen = self._selected()
        self._btn_create.setEnabled(bool(chosen) and not chosen[1])
        self._btn_edit.setEnabled(bool(chosen) and chosen[1])
        missing = self._missing()
        self._btn_all.setEnabled(bool(missing))
        self._btn_all.setText(
            f"Create {len(missing)} missing config(s)" if missing
            else "Create every missing config")

    def _on_double_click(self, _item) -> None:
        chosen = self._selected()
        if chosen is None:
            return
        self._edit_selected() if chosen[1] else self._create_selected()

    def _create_selected(self) -> None:
        chosen = self._selected()
        if chosen is not None and not chosen[1]:
            self._create(chosen[0])
            self._reload()

    def _create_all_missing(self) -> None:
        for name in self._missing():
            self._create(name)
        self._reload()

    def _create(self, name: str) -> None:
        try:
            path, notes = self._project.scaffold_member(name)
        except Exception as err:  # noqa: BLE001
            self._hub._log_issue(f"[configs] {name}: FAILED — {err}")
            QMessageBox.warning(self, "Could not scaffold", f"{name}: {err}")
            return
        self._hub.log.append_line(f"[configs] scaffolded {name}: {path}")
        for note in notes:
            ## Every reconciliation decision is logged: a DFM added with
            ## chambers unassigned, or one kept and flagged, is a thing
            ## somebody has to finish by hand.
            self._hub.log.append_line(f"[configs]   {note}")

    def _edit_selected(self) -> None:
        chosen = self._selected()
        if chosen is None:
            return
        from .config_editor import FLICConfigEditor

        path = os.path.join(self._project.member_dir(chosen[0]),
                            project_mod.CONFIG_FILENAME)
        self._hub._config_editor = FLICConfigEditor(path)
        self._hub._config_editor.show()


def main() -> None:
    sanitize_input_method_environment()
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("pyflic Analysis Hub")
    apply_theme(app, mode=ui_settings.get("theme", "auto"))
    target = sys.argv[1] if len(sys.argv) > 1 else None
    window = AnalysisHubWindow(target=target)
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
