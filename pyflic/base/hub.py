"""The pyflic Analysis Hub — a two-row tile ribbon over a full-width output
area.

Mirrors PyTrackingAnalysis's Hub.  The top strip is the containment hierarchy
read left to right — Batch · Project · Experiment — as wide tiles, plus Tools
and a status readout filling the remaining width.  The Experiment tile opens
no panel of its own: it expands a sub-strip of four compact, title-only
subtiles — Analyze · Plots · Scripts · AI — the tools that act on the loaded
member, enabled only while one is loaded.  Every control lives in a tile's
anchored panel, one open at a time.

Two rules shape it:

* **Tiles never move or hide.**  An inapplicable tile dims and its panel holds
  the control that fixes the missing state, so the strip is a stable map rather
  than a shifting menu.  The one exception is the Experiment group tile, which
  opens no panel: with nothing loaded it is inert as well as dimmed.
* **Project-first.**  The selection names the working container — a Batch or a
  Project — and an experiment is loaded *only* by double-clicking its row in the
  Project panel's members table.  There is no Load tile: the load options
  live in the Project panel beside the table that triggers the load.
  Double-clicking a Batch row opens the Project panel and double-clicking a
  member opens the QC panel: selecting is only ever a step toward doing
  something, and QC comes first.
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
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
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
from .ui.widgets import ActionButton, Card, CardGroup

#: Top strip: (key, title, icon, category, panel width).  Order is strip
#: order — the containment hierarchy reads left to right (a Batch holds
#: Projects, a Project holds members, and the loaded member holds the tools
#: that act on it), then Tools.  Three container levels want three colors:
#: Batch tints LOAD-blue, Project stays neutral, Experiment is QC-red —
#: otherwise unused on the ribbon.  The Experiment tile opens no panel
#: (width 0): it expands the sub-strip.
TILE_SPECS: list[tuple[str, str, str, Category, int]] = [
    ("batch",      "Batch",      "batch",   Category.LOAD,    620),
    ("project",    "Project",    "project", Category.NEUTRAL, 640),
    ("experiment", "Experiment", "member",  Category.QC,      0),
    ("tools",      "Tools",      "tools",   Category.TOOLS,   480),
]

#: The Experiment sub-strip: compact, title-only subtiles for the tools that
#: act on the loaded member.  Their panels are deliberately narrow — a column
#: of buttons, and a button only needs its label.
## QC before Analyze: checking the recording and deciding exclusions is what
## happens before the analysis that depends on them.
SUBTILE_SPECS: list[tuple[str, str, str, Category, int]] = [
    ("qc",      "QC",      "qc",      Category.QC,      310),
    ("analyze", "Analyze", "analyze", Category.ANALYZE, 310),
    ("plots",   "Plots",   "plots",   Category.PLOTS,   370),
    ("scripts", "Scripts", "scripts", Category.SCRIPTS, 390),
    ("ai",      "AI",      "ai",      Category.AI,      300),
]

#: The subtile keys, for "is this an experiment-level panel" checks.
EXPERIMENT_SUBTILES = tuple(spec[0] for spec in SUBTILE_SPECS)


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
        #: Whether the running task is one the suppress-tabs switch governs
        #: (a Batch Run).  Set by _start, read by _tabs_suppressed.
        self._suppress_tabs_task = False

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 10)
        root.setSpacing(8)

        # ---- ribbon: strip + collapsible sub-strip --------------------
        ## Distinct chips with a hairline seam, mirroring PyTrackingAnalysis:
        ## every tile keeps its own rounded corners and the 2px spacing is the
        ## seam.
        self._strip_host = QWidget()
        self._strip = QHBoxLayout(self._strip_host)
        self._strip.setContentsMargins(0, 0, 0, 0)
        self._strip.setSpacing(2)
        self._strip_host.setFixedHeight(TILE_HEIGHT + 2)
        self.tiles: dict[str, StatusTile] = {}
        for key, title, icon_name, category, _w in TILE_SPECS:
            ## Container tiles are wide; Tools stays a regular chip (it will
            ## probably go away, as it did in PyTrackingAnalysis).
            tile = StatusTile(key, title, icon_name, category,
                              wide=(key != "tools"))
            if key == "experiment":
                ## The Experiment tile opens no panel: it expands the
                ## sub-strip of experiment-level subtiles.
                tile.clicked.connect(lambda _k: self._toggle_experiment())
            else:
                tile.clicked.connect(self._on_tile_clicked)
            self._strip.addWidget(tile)
            self.tiles[key] = tile
        self.readout = StatusReadout()
        self._strip.addWidget(self.readout, 1)
        root.addWidget(self._strip_host)

        ## The sub-strip is a second, shorter row of title-only chips that the
        ## Experiment tile expands and collapses.  Hidden, it takes no space.
        self._experiment_expanded = False
        self._sub_strip_host = QWidget()
        self._sub_strip = QHBoxLayout(self._sub_strip_host)
        self._sub_strip.setContentsMargins(0, 0, 0, 0)
        self._sub_strip.setSpacing(2)
        self._sub_strip_host.setFixedHeight(StatusTile.COMPACT_HEIGHT + 2)
        for key, title, icon_name, category, _w in SUBTILE_SPECS:
            tile = StatusTile(key, title, icon_name, category, compact=True)
            tile.clicked.connect(self._on_tile_clicked)
            self._sub_strip.addWidget(tile)
            self.tiles[key] = tile
        self._sub_strip.addStretch(1)
        self._sub_strip_host.hide()
        root.addWidget(self._sub_strip_host)

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
        ## Every tile except the Experiment group tile owns a panel.
        self.panels: dict[str, TilePanel] = {}
        for key, _t, _i, _c, width in TILE_SPECS + SUBTILE_SPECS:
            if width:
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
        "project": "app-hub#getting-a-project-open-four-buttons-three-cases",
        "analyze": "app-hub#analyze-panel",
        "qc": "app-hub#qc-panel",
        "plots": "plots-catalog",
        "scripts": "scripts-overview",
        "ai": "concepts-ai-summary",
        "tools": "app-hub#tools-panel",
    }

    #: (tile key, card title) → help topic, for a panel's cards after its
    #: first, which takes the tile's entry above.
    _CARD_HELP: dict[tuple[str, str], str] = {
        ("project", "Experiments"): "concepts-project#blocked-members",
        ("project", "Analysis"): "app-hub#the-analysis-card",
    }

    #: (tile key, Experiment Type key) → help topic, for the type-specific
    #: groups of the Analyze, QC and Plots cards.
    _TYPE_GROUP_HELP: dict[tuple[str, str], str] = {
        ("analyze", "progressive_ratio"): "concepts-progressive-ratio#outputs",
        ("qc", "optogenetics"): "concepts-optogenetics",
        ("plots", "progressive_ratio"): "plots-catalog#progressive-ratio-plots",
        ("plots", "hedonic"): "plots-catalog#hedonic-plots",
    }

    def _type_groups(self, key: str) -> dict[str, CardGroup]:
        """The Experiment-Type groups on the *key* tile's card."""
        return {"analyze": self._type_analyze_groups, "qc": self._type_qc_groups,
                "plots": self._type_plot_groups}[key]

    def _install_panel_help(self) -> None:
        """Put a ``?`` in each panel's card title rows and type groups, one
        after the dock's clear buttons, and bind F1.

        Imported here rather than at module scope so a failure in the help
        package degrades to a Hub without help buttons instead of no Hub.
        """
        try:
            from pyflic.help.button import HelpButton, install_help_shortcut
        except Exception:  # noqa: BLE001
            return
        for key, ref in self._TILE_HELP.items():
            panel = self.panels.get(key)
            if panel is None:
                continue
            for i, card in enumerate(panel.findChildren(Card)):
                card_ref = ref if i == 0 else self._CARD_HELP.get((key, card.title()))
                if card_ref:
                    card.add_title_widget(HelpButton(card_ref, card))
        for (key, requires), ref in self._TYPE_GROUP_HELP.items():
            self._type_groups(key)[requires].add_title_widget(HelpButton(ref))
        ## General help, for when no panel is open: the dock's corner, after
        ## Clear Errors.  The first topic in reading order is the way in.
        self.dock.add_corner_widget(HelpButton(
            "getting-started", self.dock,
            tooltip="Open pyflic help — the start page, the topic list and search"))
        install_help_shortcut(self, "app-hub")

    def _build_panels(self) -> None:
        self._build_batch_panel()
        self._build_project_panel()
        self._build_analyze_panel()
        self._build_qc_panel()
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
        card = Card("Batch", Category.LOAD, icon_name="batch",
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
        self.chk_suppress_tabs = QCheckBox(
            "Suppress new plot / output tabs during Batch Runs")
        self.chk_suppress_tabs.setToolTip(
            "During a Batch Run, stop opening a tab for every figure.  The "
            "Output and Errors tabs keep updating, and every artifact is "
            "still written to disk — only the tabs are skipped.  Project and "
            "experiment analyses always show their plots.")
        self.chk_suppress_tabs.setChecked(True)
        card.add_body(self.chk_suppress_tabs)

        card.add_section_label(
            "Double-click a project to make it the working container.")
        self.panels["batch"].add_card(card)

    def _build_project_panel(self) -> None:
        """Three sections, top to bottom, mirroring PyTrackingAnalysis's
        Project panel: project identity (Create/Load), the members
        (Experiments), then what to do with them (Analysis)."""
        self._build_project_create_card()
        self._build_project_experiments_card()
        self._build_project_analysis_card()

    def _build_project_create_card(self) -> None:
        card = Card("Create/Load", Category.NEUTRAL, icon_name="project",
                    subtitle="Open a Project directory and edit its "
                             "project.yaml.")
        self.project_create_card = card
        ## Three ways in and the editor for the one that is open, in a 2×2
        ## grid: the folder already is a Project / there is no folder yet /
        ## the folder exists but has no project.yaml.  They are disjoint on
        ## purpose — each refuses the other two's case and names the button
        ## that handles it, so nobody has to guess which one their situation
        ## is.
        open_btn = ActionButton("Open Project", Category.NEUTRAL, "browse",
                                primary=True)
        open_btn.setToolTip(
            "Open a folder that already holds a project.yaml.  Choosing the "
            "one already open re-reads it from disk — members added or "
            "analyzed outside the Hub show up.")
        open_btn.clicked.connect(self._choose_project)
        new_btn = ActionButton("Create project…", Category.NEUTRAL, "new")
        new_btn.setToolTip(
            "Make a Project that does not exist yet: choose where it goes, "
            "name it, and state the design every member inherits.  The folder "
            "is created for you.")
        new_btn.clicked.connect(self._create_project)
        init_btn = ActionButton("Initialize existing directory…",
                                Category.NEUTRAL, "project")
        init_btn.setToolTip(
            "Turn a folder you already have into a Project: it keeps its own "
            "name, the experiment folders already inside it become its "
            "members, and the design is inferred from the first one that has "
            "a config.  This is the path for a study started before Projects.")
        init_btn.clicked.connect(self._initialize_project_folder)
        self.design_btn = ActionButton("Project design…", Category.NEUTRAL,
                                       "settings")
        self.design_btn.setToolTip(
            "Edit the design in project.yaml: experiment type, detection "
            "parameters, well names, the auto-filter constants and the design "
            "factors.  Every member inherits it, and a member that "
            "contradicts it fails to load.")
        self.design_btn.clicked.connect(self._edit_project_design)
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        for i, button in enumerate((open_btn, new_btn, init_btn,
                                    self.design_btn)):
            grid.addWidget(button, i // 2, i % 2)
            grid.setColumnStretch(i % 2, 1)
        ## Not a fifth way into a Project — the check you run over the one
        ## that is open, so it spans its own full-width row.
        validate_btn = ActionButton("Validate YAMLs", Category.NEUTRAL, "lint")
        validate_btn.setToolTip(
            "Check this Project's project.yaml and every member's "
            "flic_config.yaml — parse errors and semantic problems alike.  "
            "Results go to the log.  (With no Project open, the current "
            "selection is checked instead.)")
        ## Explicitly the Project's scope: the Tools-panel validator follows
        ## the selection, which prefers the loaded member's folder — a
        ## Project-panel button that skipped every other member would claim a
        ## validation it never ran.
        validate_btn.clicked.connect(
            lambda: self._validate_yaml(
                self.project.project_directory if self.project else None))
        grid.addWidget(validate_btn, 2, 0, 1, 2)
        card.add_body(grid)

        ## The loaded-project description lives in this always-visible card so
        ## a Project-load failure is readable even while the sections below
        ## stay down.
        self.project_summary = QLabel("")
        self.project_summary.setWordWrap(True)
        card.add_body(self.project_summary)
        self.panels["project"].add_card(card)

    def _build_project_experiments_card(self) -> None:
        card = Card("Experiments", Category.NEUTRAL, icon_name="member")
        self.project_experiments_card = card
        self.project_table = self._table(["Member", "DFMs", "Chambers",
                                          "Analyzed", "Report"])
        self.project_table.setToolTip(
            "Double-click a member to load it.  A red row is a Blocked "
            "Member — a run cannot use it as it stands; its reason is in the "
            "tooltip and the repair buttons below clear it.")
        self.project_table.itemDoubleClicked.connect(self._project_row_activated)
        card.add_body(self.project_table)
        hint = QLabel("Double-click a member to load it as the current "
                      "experiment.")
        hint.setStyleSheet("color: palette(mid); font-style: italic;")
        card.add_body(hint)

        ## The same three cases as the Project one level up: the member exists
        ## (the table), it does not exist at all (Create), or its folder does
        ## but its flic_config.yaml does not (Initialize) — with the bulk view
        ## for doing the third in quantity.  All three inherit the design, so
        ## all three need a Project open.
        self.create_member_btn = ActionButton("Create member…",
                                              Category.NEUTRAL, "new")
        self.create_member_btn.setToolTip(
            "Make a member folder that does not exist yet: it gets a data/ "
            "folder and a flic_config.yaml scaffolded from the design, so the "
            "design holds by construction.  Everything but the name and the "
            "chamber assignments is inherited.")
        self.create_member_btn.clicked.connect(self._create_member)
        self.init_member_btn = ActionButton("Initialize existing directory…",
                                            Category.NEUTRAL, "project")
        self.init_member_btn.setToolTip(
            "Adopt a folder already sitting in the Project that has no "
            "flic_config.yaml: any loose recording is filed into data/ (and "
            "everything else loose into extra_files/), the config is "
            "scaffolded from the design, and the config editor opens on it.")
        self.init_member_btn.clicked.connect(self._initialize_member_folder)
        self.scaffold_btn = ActionButton("Member configs…", Category.NEUTRAL,
                                         "config")
        self.scaffold_btn.setToolTip(
            "The bulk view: every folder in the Project with its config "
            "status, so the missing ones can be created and the existing ones "
            "opened without hunting through the file system.")
        self.scaffold_btn.clicked.connect(self._open_member_configs)
        member_grid = QGridLayout()
        member_grid.setHorizontalSpacing(8)
        member_grid.setVerticalSpacing(8)
        for i, button in enumerate((self.create_member_btn,
                                    self.init_member_btn, self.scaffold_btn)):
            member_grid.addWidget(button, 0, i)
            member_grid.setColumnStretch(i, 1)
        card.add_body(member_grid)

        ## The one repair that is safe in bulk keeps its own row, and stays
        ## visible while disabled: a button that vanishes when there is
        ## nothing to fix teaches nobody that it exists (ADR-0009).
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
        self.panels["project"].add_card(card)

    def _build_project_analysis_card(self) -> None:
        """What to do with the members: pool, report, publication figures,
        and the Project Scripts that do it unattended.

        Built hidden — the whole card appears only when a Project is open,
        unlike the Experiments card, which stays visible-but-gated so an empty
        Project still shows where members will appear.
        """
        card = Card("Analysis", Category.NEUTRAL, icon_name="analyze")
        self.project_analysis_card = card
        ## In the order the work happens: analyze the members, pool them,
        ## build the report — then review what came out and shape the figures.
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        buttons: list[ActionButton] = []
        for label, icon_name, handler in (
            ("Analyze all", "basic", self._run_all_members),
            ("Combine", "csv", self._build_combined),
            ("Create report", "pdf", self._project_report),
        ):
            button = ActionButton(label, Category.NEUTRAL, icon_name)
            button.clicked.connect(handler)
            buttons.append(button)
        self.view_reports_btn = ActionButton("View reports", Category.NEUTRAL,
                                             "report")
        self.view_reports_btn.setToolTip(
            "Open the Project Report, and each member's own report, in the "
            "system PDF viewer.")
        self.view_reports_btn.clicked.connect(self._view_reports)
        buttons.append(self.view_reports_btn)
        sheet_btn = ActionButton("Apply exclusion sheet…", Category.NEUTRAL,
                                 "clear")
        sheet_btn.setToolTip(
            "Read remove_chambers.csv at this Project's root and write its "
            "rows into each member's own file.  Standing declarations win.")
        sheet_btn.clicked.connect(
            lambda: self._apply_exclusion_sheet(
                self.project.project_directory if self.project else None,
                "project"))
        self.project_sheet_btn = sheet_btn
        buttons.append(sheet_btn)
        ## Publication figures are Project-level (plot_specs.yaml and figures/
        ## live at the project root), so their buttons live here rather than
        ## in the per-member Plots panel — mirroring PyTrackingAnalysis.
        editor_btn = ActionButton("Plot editor…", Category.NEUTRAL, "plot")
        editor_btn.setToolTip(
            "Design the Project's publication figures.  Specs and styles are "
            "saved to plot_specs.yaml at the project root.")
        editor_btn.clicked.connect(self._open_plot_editor)
        buttons.append(editor_btn)
        render_btn = ActionButton("Render figures", Category.NEUTRAL, "plot")
        render_btn.setToolTip(
            "Re-render the curated figures straight from plot_specs.yaml "
            "into figures/ — the headless equivalent of the Plot editor's "
            "save buttons.")
        render_btn.clicked.connect(self._render_figures)
        buttons.append(render_btn)
        ## The project-level entry point for the narrative, so it stays
        ## reachable with just a Project open — the AI subtile follows the
        ## Experiment group's gate and needs a loaded member first.
        ai_btn = ActionButton("AI narrative…", Category.NEUTRAL, "ai")
        ai_btn.setToolTip(
            "Generate an AI-written narrative of the Combined Analysis with "
            "the provider picked in the AI panel.")
        ai_btn.clicked.connect(self._generate_narrative)
        buttons.append(ai_btn)
        for i, button in enumerate(buttons):
            grid.addWidget(button, i // 3, i % 3)
        for column in range(3):
            grid.setColumnStretch(column, 1)
        card.add_body(grid)

        ## Project Scripts live with the Project, not in the Scripts panel:
        ## they act on the Project (pool, report, run_in_experiments) and are
        ## available the moment one is open, with no member loaded.  The
        ## Scripts panel is the *member* level and nothing else — the same
        ## split PyTrackingAnalysis draws.
        srow = QHBoxLayout()
        srow.addWidget(QLabel("Script:"))
        self.project_script = QComboBox()
        self.project_script.setMinimumWidth(200)
        srow.addWidget(self.project_script, 1)
        self.run_project_script_btn = ActionButton("Run script",
                                                   Category.NEUTRAL,
                                                   "play", primary=True)
        self.run_project_script_btn.clicked.connect(self._run_project_script)
        srow.addWidget(self.run_project_script_btn)
        self.edit_project_scripts_btn = ActionButton(
            "Edit scripts…", Category.NEUTRAL, "scripts")
        self.edit_project_scripts_btn.setToolTip(
            "Open the Script Editor on this Project's project.yaml.")
        self.edit_project_scripts_btn.clicked.connect(
            self._open_project_script_editor)
        srow.addWidget(self.edit_project_scripts_btn)
        card.add_body(srow)
        self.panels["project"].add_card(card)
        ## add_card force-shows reparented cards; this one starts hidden.
        card.setVisible(False)

    def _build_analyze_panel(self) -> None:
        card = Card("Analyze", Category.ANALYZE, icon_name="analyze",
                    subtitle="Actions on the loaded member.")
        grid = QVBoxLayout()
        for label, icon_name, action in (
            ("Basic analysis", "basic", {"action": "basic_analysis"}),
            ("Feeding summary CSV", "csv", {"action": "feeding_csv"}),
            ("Faceted summary CSV", "csv", {"action": "facet_csv"}),
            ("Binned CSV", "binned", {"action": "binned_csv"}),
            ("Event statistics", "tidy", {"action": "tidy_export"}),
            ("PDF report", "pdf", {"action": "pdf_report"}),
        ):
            button = ActionButton(label, Category.ANALYZE, icon_name)
            button.clicked.connect(
                lambda _c=False, s=action: self._run_experiment_action(s))
            grid.addWidget(button)
        card.add_body(grid)

        ## Experiment-Type-specific analyses, shown only while a member of that
        ## type is loaded — the same gating, and the same keys, as the Plots
        ## card's type groups.  (The type's QC lives on the QC card.)
        self._type_analyze_groups: dict[str, CardGroup] = {}
        for requires, title, buttons in (
            ("progressive_ratio", "Progressive Ratio only",
             (("Paired − yoked difference CSV", "csv", "paired_yoked_diff"),
              ("Breaking point CSV", "csv", "breaking_point"))),
        ):
            group = CardGroup(title)
            for label, icon_name, action in buttons:
                button = ActionButton(label, Category.ANALYZE, icon_name)
                button.clicked.connect(
                    lambda _c=False, a=action: self._run_experiment_action({"action": a}))
                group.add(button)
            group.setVisible(False)
            card.add_body(group)
            self._type_analyze_groups[requires] = group

        row = QHBoxLayout()
        row.addWidget(QLabel("Bin size (min):"))
        self.spin_binsize = QDoubleSpinBox()
        self.spin_binsize.setRange(0.5, 600.0)
        self.spin_binsize.setValue(30.0)
        row.addWidget(self.spin_binsize)
        row.addStretch(1)
        card.add_body(row)
        self.panels["analyze"].add_card(card)

    def _build_qc_panel(self) -> None:
        """Everything QC for the loaded member, in the order the work
        happens: write the reports, inspect them, decide exclusions."""
        card = Card("QC", Category.QC, icon_name="qc",
                    subtitle="Per-DFM quality control for the loaded member: "
                             "integrity, bleeding, and the raw / baselined / "
                             "cumulative-licks signal plots.")
        ## Not "which member is loaded" — the Experiment tile this panel
        ## hangs from already says that, and so does the status strip.  Only
        ## what is true of THIS card: whether there is anything to look at.
        self.qc_hint = QLabel("")
        self.qc_hint.setWordWrap(True)
        card.add_body(self.qc_hint)

        run_btn = ActionButton("QC reports", Category.QC, "qc", primary=True)
        run_btn.setToolTip(
            "Write the QC bundle into qc/: integrity report, data breaks, "
            "simultaneous-feeding and bleeding matrices (two-well), and the "
            "Raw Signal, Baselined, and Cumulative Licks plots.  Basic "
            "analysis deliberately skips this slow half.")
        run_btn.clicked.connect(
            lambda: self._run_experiment_action({"action": "run_qc"}))
        card.add_body(run_btn)

        viewer_btn = ActionButton("Open QC Viewer", Category.QC, "qc")
        viewer_btn.setToolTip(
            "The interactive QC app on the loaded member — integrity and "
            "bleeding tables, the signal plots, and per-chamber exclusions "
            "saved to remove_chambers.csv.")
        viewer_btn.clicked.connect(self._open_qc_viewer)
        card.add_body(viewer_btn)

        plots_btn = ActionButton("View QC plots", Category.QC, "plot")
        plots_btn.setToolTip(
            "Open the saved Raw Signal, Baselined, and Cumulative Licks "
            "plots — one tab per DFM and kind — in the output area.")
        plots_btn.clicked.connect(self._view_qc_plots)
        card.add_body(plots_btn)

        folder_btn = ActionButton("Open qc folder", Category.TOOLS, "open")
        folder_btn.setToolTip(
            "Open the member's qc/ folder in the system file browser.")
        folder_btn.clicked.connect(self._open_qc_folder)
        card.add_body(folder_btn)

        ## The light QC, in one group: every optogenetic member's (was the light
        ## where the licks were?), shown whenever the member is optogenetic,
        ## and Progressive Ratio's own (did the paired fly earn its light?),
        ## shown for that type.
        self._type_qc_groups: dict[str, CardGroup] = {}
        opto = CardGroup("Optogenetics")
        self._opto_qc_buttons: list[ActionButton] = []
        for label, icon_name, run, tip in (
            ("Opto light QC table", "qc",
             lambda: self._run_experiment_action({"action": "opto_light_qc"}),
             "Per linkage group: was the light where the licks were?  Writes "
             "qc/opto/ (the verdicts, the intervals, every light event and the "
             "Program.txt as read) and logs every flagged group.  Any "
             "Experiment Type; a failed group's chambers leave the analysis only "
             "when exclude_failed_opto_chambers is on."),
            ("Light explained by licks (QC)", "plot",
             lambda: self._run_plot_action("plot_opto_light"),
             "Per DFM: each linkage group's lit time, explained by a trigger-well "
             "lick or touch and unexplained, and the time the emulated firmware "
             "trigger saw contact that pyflic did not."),
        ):
            button = ActionButton(label, Category.QC, icon_name)
            button.setToolTip(tip)
            button.clicked.connect(lambda _c=False, r=run: r())
            opto.add(button)
            self._opto_qc_buttons.append(button)
        self._pr_qc_buttons: list[ActionButton] = []
        for label, icon_name, run, tip in (
            ("Light QC table", "qc",
             lambda: self._run_experiment_action({"action": "pr_light_qc"}),
             "Per chamber group: did the paired fly earn its light?  Writes "
             "pr_light_qc.csv and pr_light_events.csv and logs every flagged "
             "group.  A failed group leaves the analysis at the next Basic "
             "analysis unless exclude_failed_pr_groups is off."),
            ("Licks per light event (QC)", "plot",
             lambda: self._run_plot_action("plot_pr_light_events"),
             "Per DFM: the sucrose licks credited to each Test light event.  A "
             "working progressive ratio climbs; hollow red rings are lick-free "
             "light events."),
            ("Sucrose Well resting level (QC)", "plot",
             lambda: self._run_plot_action("plot_pr_resting_level"),
             "Per DFM: the paired Sucrose Well's per-minute median raw signal "
             "against the DFM's other Sucrose Wells, light onsets as a rug."),
        ):
            button = ActionButton(label, Category.QC, icon_name)
            button.setToolTip(tip)
            button.clicked.connect(lambda _c=False, r=run: r())
            opto.add(button)
            self._pr_qc_buttons.append(button)
        opto.setVisible(False)
        card.add_body(opto)
        self._type_qc_groups["optogenetics"] = opto
        self.panels["qc"].add_card(card)

    def _build_plots_panel(self) -> None:
        """Quick figures for the loaded member — and only that.

        Publication figures are Project-level, so the Plot editor and the
        headless render live on the Project panel's Analysis card instead
        (mirroring PyTrackingAnalysis).

        The card is three groups, because its buttons are three kinds of
        thing and a flat list said they were one.  The Metric dropdown steers
        exactly two of them, so it lives *inside* their group rather than
        above the lot; the figures that fix their own metrics sit apart; and
        anything an Experiment Type adds is grouped under that type's name and
        shown only while such a member is loaded.
        """
        card = Card("Plots", Category.PLOTS, icon_name="plots",
                    subtitle="Quick figures for the loaded member.")
        ## The one note on the card: which buttons the dropdown reaches is
        ## the whole point of the grouping, and a box cannot say "only".
        metric_group = CardGroup(
            "Chosen metric", note="Applies to these two figures only.")
        row = QHBoxLayout()
        row.addWidget(QLabel("Metric:"))
        self.plot_metric = QComboBox()
        self.plot_metric.setMinimumWidth(180)
        row.addWidget(self.plot_metric, 1)
        metric_group.add(row)
        for label, icon_name, action in (
            ("Binned time course", "binned", "plot_binned"),
            ("Dot plot", "dot", "plot_dot"),
        ):
            metric_group.add(self._plot_button(label, icon_name, action))
        card.add_body(metric_group)

        fixed_group = CardGroup("Standard figures")
        for label, icon_name, action in (
            ("Feeding summary", "feeding", "plot_feeding_summary"),
            ("Well A vs B", "well", "plot_well_comparison"),
        ):
            button = self._plot_button(label, icon_name, action)
            fixed_group.add(button)
            if action == "plot_well_comparison":
                ## Two-well only — a single-well member has no B to compare.
                self._two_well_plot_buttons = [button]
        card.add_body(fixed_group)

        ## Experiment-Type-specific figures.  One group per type, titled with
        ## the type, hidden unless a member of that type is loaded — so the
        ## card never offers a button whose only possible answer is "this
        ## action requires a different Experiment Type".
        self._type_plot_groups: dict[str, CardGroup] = {}
        for requires, title, buttons in (
            ("progressive_ratio", "Progressive Ratio only",
             (("Cumulative difference curve", "plot",
               "plot_pr_cumulative_diff"),
              ("Training-aligned traces (QC)", "plot",
               "plot_pr_cumulative_licks"),
              ("Still-responding curve", "plot", "plot_pr_still_responding"),
              ("Breaking-point plots", "plot", "plot_breaking_point"))),
            ("hedonic", "Hedonic only",
             (("Hedonic feeding plot", "feeding", "plot_hedonic"),)),
        ):
            group = CardGroup(title)
            for label, icon_name, action in buttons:
                group.add(self._plot_button(label, icon_name, action))
            group.setVisible(False)
            card.add_body(group)
            self._type_plot_groups[requires] = group
        self.panels["plots"].add_card(card)

    def _plot_button(self, label: str, icon_name: str,
                     action: str) -> ActionButton:
        """One Plots-card button, wired to its plot action."""
        button = ActionButton(label, Category.PLOTS, icon_name)
        button.clicked.connect(
            lambda _c=False, a=action: self._run_plot_action(a))
        return button

    def _build_scripts_panel(self) -> None:
        """The **member** script level, and only that.

        Project Scripts are authored and run from the Project panel: they act
        on the Project itself and need no member.  Keeping both levels here
        made the panel readable while a Project was open with nothing loaded,
        which is precisely when half of it could not run.
        """
        card = Card("Scripts", Category.SCRIPTS, icon_name="scripts",
                    subtitle="Experiment Scripts for the loaded member.  "
                             "Project Scripts live in the Project panel — the "
                             "two registries never mix; the only bridge is "
                             "run_in_experiments.")
        self.scripts_hint = QLabel("")
        self.scripts_hint.setWordWrap(True)
        card.add_body(self.scripts_hint)

        erow = QHBoxLayout()
        self.experiment_script = QComboBox()
        self.experiment_script.setMinimumWidth(200)
        erow.addWidget(self.experiment_script, 1)
        self.run_experiment_script_btn = ActionButton("Run", Category.SCRIPTS,
                                                      "play", primary=True)
        self.run_experiment_script_btn.clicked.connect(
            self._run_experiment_script)
        erow.addWidget(self.run_experiment_script_btn)
        card.add_body(erow)

        self.edit_scripts_btn = ActionButton("Open Script Editor",
                                             Category.SCRIPTS, "scripts")
        self.edit_scripts_btn.setToolTip(
            "Open the Script Editor on the loaded member's flic_config.yaml.")
        self.edit_scripts_btn.clicked.connect(self._open_script_editor)
        card.add_body(self.edit_scripts_btn)
        self.panels["scripts"].add_card(card)

    def _build_ai_panel(self) -> None:
        card = Card("AI summary", Category.AI, icon_name="ai",
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

        gen = ActionButton("Generate narrative", Category.AI, "ai",
                           primary=True)
        gen.clicked.connect(self._generate_narrative)
        card.add_body(gen)
        self.panels["ai"].add_card(card)

    def _build_tools_panel(self) -> None:
        card = Card("Tools", Category.TOOLS, icon_name="tools")
        ## The QC viewer moved to the Experiment group's QC subtile — its
        ## home beside the button that writes what it shows.
        for label, icon_name, handler in (
            ("Config editor", "config", self._open_config_editor),
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
        ## A subtile panel needs the sub-strip showing to have an anchor; a
        ## container panel and the expanded group are never open together —
        ## one thing is open at a time.
        if key in EXPERIMENT_SUBTILES:
            self._expand_experiment()
            if not self._experiment_expanded:
                ## No member loaded, so the group cannot expand — a panel
                ## anchored to a hidden subtile would float in space.
                return
        else:
            self._collapse_experiment()
        tile = self.tiles[key]
        panel = self.panels[key]
        central = self.centralWidget()
        self._settle_ribbon_layout()
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

    # ---- the Experiment group (sub-strip) -----------------------------

    def _toggle_experiment(self) -> None:
        if self._experiment_expanded:
            self._collapse_experiment()
        else:
            self._expand_experiment()

    def _expand_experiment(self) -> None:
        if self.experiment is None or self._experiment_expanded:
            return
        ## The group and a container panel are never open together.
        if self._open_key is not None \
                and self._open_key not in EXPERIMENT_SUBTILES:
            self._close_panel()
        self._experiment_expanded = True
        self._place_sub_strip()
        self._sub_strip_host.show()
        self.tiles["experiment"].set_active(True)
        self._settle_ribbon_layout()

    def _collapse_experiment(self) -> None:
        if not self._experiment_expanded:
            return
        if self._open_key in EXPERIMENT_SUBTILES:
            self._close_panel()
        self._experiment_expanded = False
        self._sub_strip_host.hide()
        self.tiles["experiment"].set_active(False)
        self._settle_ribbon_layout()
        self._reanchor_open_panel()

    def _place_sub_strip(self) -> None:
        """Left-indent the sub-strip to the Experiment tile's left edge, so
        the subtiles read as that tile's contents rather than a second,
        unrelated row."""
        x = self.tiles["experiment"].geometry().x()
        self._sub_strip.setContentsMargins(max(0, x), 0, 0, 0)

    def _settle_ribbon_layout(self) -> None:
        """Force layout on a just-shown/hidden sub-strip so tile coordinates
        are real before a panel anchors to them."""
        for widget in (self.centralWidget(), self._strip_host,
                       self._sub_strip_host):
            layout = widget.layout() if widget is not None else None
            if layout is not None:
                layout.activate()

    def _reanchor_open_panel(self) -> None:
        if self._open_key is not None:
            self._open_panel(self._open_key)

    def _handle_click_away(self, event) -> None:
        """Close the open panel — and fold the sub-strip — on a click outside
        them and outside the ribbon."""
        if self._open_key is None and not self._experiment_expanded:
            return
        widget = QApplication.widgetAt(event.globalPosition().toPoint())
        if widget is None:
            return
        panel = self.panels.get(self._open_key) if self._open_key else None
        node = widget
        while node is not None:
            if node is panel or isinstance(node, StatusTile) \
                    or node is self._strip_host or node is self._sub_strip_host:
                return
            node = node.parentWidget()
        self._close_panel()
        self._collapse_experiment()

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        ## Esc closes only the panel; the expanded group stays.
        if event.key() == Qt.Key.Key_Escape and self._open_key is not None:
            self._close_panel()
            return
        super().keyPressEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._place_sub_strip()
        self._reanchor_open_panel()

    def _restyle(self) -> None:
        c = surface_colors()
        self.centralWidget().setStyleSheet(f"background: {c['band']};")
        for tile in self.tiles.values():
            tile.restyle()
        self.readout.restyle()
        for panel in self.panels.values():
            panel.restyle()
            ## Cards paint their own surfaces (and their groups'), so the
            ## theme toggle has to reach them or half the panel stays light.
            for card in panel.cards():
                card.restyle()

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
        """Write a project.yaml — through the design editor, always.

        A Project created with no ``design:`` validates its members against
        *each other* instead (the legacy mode), which means the first member
        edited silently becomes the standard.  Stating the design up front is
        the whole point of the level.
        """
        ## No pre-picker: the dialog's own Directory field is where the new
        ## folder is named, and its file browser will create one.  Asking for
        ## an existing folder first is the *other* button's question.
        start = self.project.project_directory if self.project is not None \
            else (self._current_dir() or os.getcwd())
        self._run_design_editor(start)

    def _initialize_project_folder(self) -> None:
        """The third way in: a folder that exists but has no project.yaml.

        Open a Project wants the marker already there and Create Project makes
        the folder itself, so this one is for the study that was under way
        before Projects existed — its subdirectories become the members and
        its design is read off the first of them that has a config.
        """
        ## Prefilling a folder that is already a Project would only earn the
        ## dialog's "that one is a Project already" refusal.
        start = ""
        if self.project is None:
            current = self._current_dir()
            if current and not project_mod.is_project_dir(current):
                start = current
        self._run_design_editor(start, initialize=True)

    def _edit_project_design(self) -> None:
        directory = (self.project.project_directory
                     if self.project is not None else self._current_dir())
        if directory is None:
            QMessageBox.information(
                self, "No Project",
                "Open a Project first, or use 'Create project…' to make "
                "one.")
            return
        self._run_design_editor(directory)

    def _run_design_editor(self, directory: str, *,
                           initialize: bool = False) -> None:
        from .design_editor import ProjectDesignDialog

        dialog = ProjectDesignDialog(self, start_dir=directory,
                                     initialize_existing=initialize)
        if not dialog.exec() or not dialog.saved_dir:
            return
        saved = dialog.saved_dir
        self.log.append_line(
            f"Wrote {os.path.join(saved, project_mod.PROJECT_FILENAME)}")
        for name in getattr(dialog, "adopted", []) or []:
            self.log.append_line(
                f"[design] {name}: removed its own global: — it now inherits "
                f"the project design")
        try:
            self._set_project(project_mod.Project(saved))
        except Exception as err:  # noqa: BLE001
            ## A Project whose members contradict the new design refuses to
            ## load.  Say so here rather than leaving the old one selected as
            ## though nothing had happened.
            self._set_project(None)
            QMessageBox.critical(self, "Project does not load", str(err))
            self._log_issue(f"ERROR: {err}")
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
        """Double-clicking a member loads it — and, once loaded, reveals the
        QC panel: loading is only ever a step toward doing something with
        it, and checking the recording comes first.  The reveal waits for
        the load because the Experiment group the panel hangs from does not
        exist until a member is loaded."""
        if self.project is None:
            return
        cell = self.project_table.item(item.row(), 0)
        if cell is None:
            return
        name = cell.text()
        blocked = {m.name: m for m in self.project.blocked_members()}
        if name in blocked:
            self._offer_blocked_fix(blocked[name])
            return
        ## Close the panel only when the load actually starts: _start refuses
        ## while a task is running, and yanking the panel away would then be
        ## the click's only effect.
        if self._load_member(name):
            self._close_panel()

    def _offer_blocked_fix(self, member) -> None:
        """Double-clicking a Blocked Member offers the repair, rather than
        naming the button that would have done it.

        The row is where somebody notices the problem, so it is where the fix
        belongs; the buttons below stay, because a repair nobody discovers by
        double-clicking is still one that has to be findable.
        """
        detail = f"{member.name}: {member.detail or member.status}"
        if member.fix == "config":
            answer = QMessageBox.question(
                self, "Blocked member",
                f"{detail}\n\nGive it a flic_config.yaml scaffolded from the "
                "project design, and open it in the config editor?")
            if answer != QMessageBox.StandardButton.Yes:
                return
            if self._scaffold_member(member.name) is None:
                return
            self._reload_project()
            self._open_config_editor_on(
                Path(self.project.member_dir(member.name))
                / project_mod.CONFIG_FILENAME)
            return
        if member.fix == "file":
            answer = QMessageBox.question(
                self, "Blocked member",
                f"{detail}\n\nFile its DFM CSVs into data/ now?  Everything "
                "else loose goes to extra_files/; YAML files and "
                "remove_chambers.csv stay where they are.")
            if answer == QMessageBox.StandardButton.Yes:
                self._file_unfiled()
            return
        QMessageBox.information(
            self, "Blocked member",
            f"{detail}\n\nNothing here can be fixed automatically.")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    #: Logged the moment a load starts.  Parsing the first DFM CSV can take a
    #: while and prints nothing until it finishes, so without this the log sat
    #: silent and a double-click looked like it had done nothing.
    _LOADING_NOTE = ("Reading DFM data from disk — this can take a while for "
                     "large recordings.  Progress appears here as each DFM "
                     "finishes.")

    def _load_kwargs(self) -> dict:
        workers = self.spin_workers.value()
        return {"parallel": self.chk_parallel.isChecked(),
                "max_workers": workers or None}

    def _load_member(self, name: str) -> bool:
        project = self.project
        if project is None:
            return False

        def task():
            return project.load_experiment(name, **self._load_kwargs())

        def done(exp):
            ## The selection can change while the load runs; attaching the
            ## old Project's member under the new one would show a member
            ## built from a different design.
            if self.project is not project:
                self.log.append_line(
                    f"[load] '{name}' finished loading after the selection "
                    "changed — discarded.")
                return
            self.experiment = exp
            self.experiment_name = name
            self.refresh()
            self._reveal_experiment_group()

        return self._start(task, f"Loading member '{name}'", done,
                           note=self._LOADING_NOTE)

    def _load_standalone(self, path: str) -> bool:
        from .yaml_config import load_experiment_yaml

        def task():
            return load_experiment_yaml(path, **self._load_kwargs())

        def done(exp):
            if self.project is not None:
                self.log.append_line(
                    f"[load] {path} finished loading after a Project was "
                    "opened — discarded.")
                return
            self.experiment = exp
            self.experiment_name = os.path.basename(path)
            self.refresh()
            self._reveal_experiment_group()

        return self._start(task, f"Loading {path}", done,
                           note=self._LOADING_NOTE)

    def _reveal_experiment_group(self) -> None:
        """Expand the experiment sub-strip after a load — QC · Analyze ·
        Plots · Scripts · AI — without opening any of them.

        A load used to open the QC panel outright.  That answered a question
        nobody had asked yet: it put one panel in front of the member before
        anyone said which one they wanted, and the panel it opened covered
        the rest of the strip.  The sub-strip *is* the menu of what can now
        be done; showing it is the feedback, choosing from it is the user's
        move.  Unless the user opened a panel while the load ran — a reveal
        must not yank that away.
        """
        if self._open_key is None:
            self._expand_experiment()

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    def _start(self, task, label: str, on_done=None, *,
               note: str | None = None, suppress_tabs: bool = False) -> bool:
        """Run *task* in the worker.  Returns whether it actually started —
        callers with a side effect tied to the run (closing the panel a
        double-click came from) must not perform it on a refusal.

        *suppress_tabs* marks the task as one the Batch panel's suppress-tabs
        switch governs (a Batch Run); every other task shows its figures
        regardless of the switch.
        """
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "Busy",
                                    "A task is already running.")
            return False
        self._suppress_tabs_task = suppress_tabs
        self.log.append_line(f"\n=== {label} ===")
        if note:
            self.log.append_line(note)
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
        return True

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
        ## The task may have changed state before failing — a Batch Run
        ## unloads the member up front — and without this, tiles keep
        ## describing state that no longer exists until the next success.
        self.refresh()
        first = message.strip().splitlines()[-1] if message.strip() else "failed"
        QMessageBox.critical(self, "Task failed", first)

    def _log_issue(self, message: str) -> None:
        """Say it in both places: the Output log keeps the narrative in order,
        and the Errors tab keeps it findable after two thousand lines."""
        self.log.append_line(message)
        self.errors.append_line(message)

    def _set_running(self, running: bool) -> None:
        """Grey the open panel's cards in place rather than closing it — a task
        finishing should not move the UI out from under the user.

        Re-enabling walks *every* panel, not just the open one: the open panel
        can change while a task runs (double-clicking a member greys the
        Project panel, then switches to Analyze), and re-enabling only the
        current panel left the one greyed at start disabled for good."""
        if running:
            if self._open_key is not None:
                self.panels[self._open_key].setEnabled(False)
        else:
            for panel in self.panels.values():
                panel.setEnabled(True)

    def _tabs_suppressed(self) -> bool:
        """Whether new figure/artifact tabs are being skipped for this run.

        Scoped to Batch Runs only: the switch exists because a Batch Run's
        tabs run into the hundreds, but plots are the point of a project or
        experiment analysis someone ran by hand, so those always show.
        """
        return bool(self._suppress_tabs_task
                    and getattr(self, "chk_suppress_tabs", None)
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
                    "output tabs during Batch Runs' is checked in the Batch "
                    "panel.")
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
        if action in ("plot_pr_cumulative_diff", "plot_pr_cumulative_licks",
                      "plot_opto_light"):
            ## These read the bin from the step, not the context, so a script
            ## keeps its own default; from the Hub the spinbox rules.
            step["binsize"] = self.spin_binsize.value()
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

    def _create_member(self) -> None:
        """The member does not exist yet: make its folder and scaffold it.

        Everything but the name and the chamber assignments comes from the
        Project's design, so the only thing worth asking for is the name.
        """
        if not self._require_project():
            return
        project = self.project
        name, ok = QInputDialog.getText(
            self, "Create member", "New member folder name:")
        name = (name or "").strip()
        if not ok or not name:
            return
        directory = Path(project.member_dir(name))
        if project_mod.is_experiment_dir(directory):
            QMessageBox.warning(self, "Create member",
                                f"'{name}' already exists and has a config.")
            return
        if directory.exists():
            ## The other scenario, and it has its own button: initializing
            ## files what is already in there instead of assuming an empty
            ## folder.
            QMessageBox.warning(
                self, "Create member",
                f"'{name}' already exists.\n\nUse 'Initialize existing "
                "directory…' to give the folder you already have a config.")
            return
        if self._scaffold_member(name) is None:
            return
        self._reload_project()
        ## The scaffold is a starting point, not the config: its chamber
        ## assignments are blank (or the first member's).  Both ways of
        ## finishing it are one click away rather than left to be found.
        self._finish_new_member_config(name, directory)

    def _scaffold_member(self, name: str) -> str | None:
        """Scaffold *name*'s flic_config.yaml from the design, with its notes.

        Shared by Create member, Initialize existing directory, and the blocked
        row's offer — one place decides what a scaffolded member looks like.
        """
        if self.project is None:
            return None
        try:
            path, notes = self.project.scaffold_member(name)
        except Exception as err:  # noqa: BLE001
            self._log_issue(f"[member] {name}: FAILED — {err}")
            QMessageBox.warning(self, "Could not scaffold", f"{name}: {err}")
            return None
        self.log.append_line(f"[member] scaffolded {name}: {path}")
        for note in notes:
            self.log.append_line(f"[member]   {note}")
        return path

    def _finish_new_member_config(self, name: str, directory: Path) -> None:
        """Offer the two ways to finish a just-created member's config.

        Copying is the common case for the second and later members of a run —
        the same plate layout as one that already works.  It is checked
        against the design *before* it is written, and anything that would not
        conform sends the user to the editor instead, scaffold still in place.
        """
        box = QMessageBox(self)
        box.setWindowTitle("Create member")
        box.setIcon(QMessageBox.Icon.Question)
        box.setText(f"'{name}' is ready, with a flic_config.yaml scaffolded "
                    "from the project design.")
        box.setInformativeText(
            "Edit it now, or replace it with a config copied from a member "
            "that is already set up.")
        edit_btn = box.addButton("Edit config…",
                                 QMessageBox.ButtonRole.AcceptRole)
        copy_btn = box.addButton("Copy config from…",
                                 QMessageBox.ButtonRole.ActionRole)
        box.setDefaultButton(edit_btn)
        box.exec()
        if box.clickedButton() is copy_btn \
                and self._copy_member_config(name, directory):
            self._reload_project()
            return
        ## Either they chose to edit, or the copy did not happen — and a
        ## member left on its scaffold is one nobody has assigned chambers in.
        self._open_config_editor_on(directory / project_mod.CONFIG_FILENAME)

    def _copy_member_config(self, name: str, directory: Path) -> bool:
        """Replace *name*'s scaffold with a config chosen from elsewhere.

        True when the copy was made.  False means nothing was written — the
        user cancelled, or the chosen file would not be a conforming member —
        and the caller opens the editor on the scaffold that is still there.
        """
        import shutil

        project = self.project
        if project is None:
            return False
        chosen, _ = QFileDialog.getOpenFileName(
            self, f"Choose a flic_config.yaml to copy into '{name}'",
            project.project_directory,
            "FLIC config (flic_config.yaml);;YAML files (*.yaml *.yml);;"
            "All files (*)")
        if not chosen:
            QMessageBox.information(
                self, "Copy config",
                "No file chosen — opening the scaffolded config in the config "
                "editor instead.")
            return False
        source = Path(chosen)
        target = directory / project_mod.CONFIG_FILENAME
        if source.resolve() == target.resolve():
            QMessageBox.warning(
                self, "Copy config",
                f"That is '{name}'s own config.\n\nOpening it in the config "
                "editor instead.")
            return False
        import yaml as _yaml

        try:
            config = _yaml.safe_load(source.read_text(encoding="utf-8"))
        except Exception as err:  # noqa: BLE001
            QMessageBox.warning(
                self, "Copy config",
                f"'{source.name}' could not be read:\n{err}\n\nOpening the "
                "scaffolded config in the config editor instead.")
            return False
        ## Checked BEFORE it is written: a non-conforming member makes the
        ## whole Project refuse to load, and the copy would then have to be
        ## found and undone by hand.
        problems = project.design_problems_for(config, source.name)
        if problems:
            QMessageBox.warning(
                self, "Copy config",
                f"'{source.name}' does not fit this Project's design:\n  - "
                + "\n  - ".join(problems[:6])
                + ("\n  - …" if len(problems) > 6 else "")
                + "\n\nNothing was copied.  Opening the scaffolded config in "
                "the config editor instead.")
            return False
        try:
            shutil.copyfile(source, target)
        except Exception as err:  # noqa: BLE001
            QMessageBox.warning(self, "Copy config",
                                f"Could not copy '{source.name}':\n{err}")
            return False
        self.log.append_line(f"[member] {name}: copied {source} to {target}")
        ## Conforming is not the same as ready: a copied config carries the
        ## other member's chamber assignments, which are about that plate.
        self.log.append_line(
            f"[member]   check {name}'s chamber → treatment assignments — "
            f"they came from {source.parent.name}.")
        return True

    def _initialize_member_folder(self) -> None:
        """The member's folder exists but its config does not: file what is
        loose in it, scaffold the config, and open the editor.

        Filing first, config second: the config editor's view of the member
        (and every analysis after it) reads ``data/``, so a recording still
        sitting at the root would make the freshly configured member look
        empty.
        """
        if not self._require_project():
            return
        project = self.project
        candidates = layout_mod.initializable_dirs(project.project_directory)
        if not candidates:
            QMessageBox.information(
                self, "Initialize existing directory",
                f"Every folder in '{project.name}' already has a "
                "flic_config.yaml.\n\nUse 'Create member…' to make a new "
                "one.")
            return
        labels = [f"{item.name}  —  {item.status or 'empty'}"
                  for item in candidates]
        choice, ok = QInputDialog.getItem(
            self, "Initialize existing directory",
            f"Folder to make a member of '{project.name}':", labels, 0, False)
        if not ok or not choice:
            return
        item = candidates[labels.index(choice)]

        ## Re-classify rather than trusting the listing: it was built before
        ## the user had a chance to change anything on disk.
        state = layout_mod.classify(item.directory)
        if state.status in (layout_mod.AMBIGUOUS, layout_mod.UNREADABLE):
            QMessageBox.warning(
                self, "Initialize existing directory",
                f"'{state.name}': {state.detail or state.status}\n\nThis one "
                "has to be sorted out by hand.")
            return
        if state.status == layout_mod.UNFILED:
            plan = layout_mod.file_recording(item.directory,
                                             log=self.log.append_line)
            if plan.refused:
                ## A same-id collision or an unwritable target: stop before
                ## writing the config, so a retry after the fix does the whole
                ## job rather than half of it.
                QMessageBox.warning(
                    self, "Initialize existing directory",
                    f"Could not file '{state.name}': {plan.refused}")
                return
            self.log.append_line(f"[file] {state.name}: {plan.describe()}")
            for skipped, why in plan.skipped:
                self.log.append_line(
                    f"[file] {state.name}: {skipped} skipped — {why}")

        if self._scaffold_member(item.name) is None:
            return
        self._reload_project()
        ## The point of this button is the editor: a scaffolded config still
        ## needs its chamber assignments before the member means anything.
        self._open_config_editor_on(
            Path(item.directory) / project_mod.CONFIG_FILENAME)

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
        paths += [Path(project.member_report_path(name))
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
        ## longer exist.  Refresh the experiment-level tiles now: the run's
        ## own refresh only comes when it finishes, and a lit Experiment tile
        ## naming an unloaded member is a click that silently does nothing.
        self.experiment = None
        self.experiment_name = None
        self._refresh_experiment_tiles()
        self._refresh_ai()
        self._refresh_card_dimming()
        self._refresh_readout()

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

        self._start(task, f"Batch Run in {os.path.basename(root)}", done,
                    suppress_tabs=True)

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
        directory = self._current_dir()
        config = os.path.join(directory, "flic_config.yaml") if directory else None
        self._open_config_editor_on(
            config if config and os.path.isfile(config) else None)

    def _open_config_editor_on(self, path) -> None:
        """Open a config editor window on *path* and keep it alive.

        Held in a list: one editor per member, so opening a second does not
        drop the reference to the first and take its window (and any unsaved
        edits) with it.
        """
        from .config_editor import FLICConfigEditor

        editors = getattr(self, "_config_editors", None)
        if editors is None:
            editors = self._config_editors = []
        ## Drop the ones the user has already closed, so the list does not
        ## grow for the life of the session.  A window whose C++ side has gone
        ## raises from isVisible() rather than answering.
        kept = []
        for old_editor in editors:
            try:
                if old_editor.isVisible():
                    kept.append(old_editor)
            except RuntimeError:
                pass
        editors[:] = kept
        editor = FLICConfigEditor(str(path) if path is not None else None)
        editors.append(editor)
        self._config_editor = editor
        editor.show()
        editor.raise_()
        editor.activateWindow()
        return editor

    def _open_qc_viewer(self) -> None:
        if not self._require_experiment():
            return
        from .qc_viewer import MainWindow as QCViewerWindow

        ## Hand over the loaded experiment: re-parsing every DFM CSV in the
        ## viewer's own Load tab is minutes of work the Hub already did.
        self._qc = QCViewerWindow(Path(self.experiment.experiment_dir),
                                  experiment=self.experiment)
        self._qc.show()

    def _member_qc_dir(self) -> Path | None:
        """The loaded member's ``qc/`` directory, loaded member permitting."""
        if self.experiment is None:
            return None
        qc_dir = getattr(self.experiment, "qc_dir", None)
        return Path(qc_dir) if qc_dir is not None \
            else Path(self.experiment.experiment_dir) / "qc"

    #: The saved QC signal plots, as (subdirectory, filename suffix, label).
    _QC_PLOT_KINDS = (("raw_signal", "raw", "raw"),
                      ("baselined", "baselined", "baselined"),
                      ("cumulative_licks", "cumulative_licks", "licks"))

    def _view_qc_plots(self) -> None:
        """Open the saved QC signal PNGs as output-area tabs.

        Reuses a tab per (DFM, kind) so viewing twice does not grow the tab
        bar; the images are the ones ``write_qc_reports`` saved, so a change
        of parameters needs a re-run to show.
        """
        if not self._require_experiment():
            return
        from .ui import ZoomableImageView

        qc_dir = self._member_qc_dir()
        shown = 0
        for dfm_id in sorted(self.experiment.dfms):
            pngs = [(qc_dir / subdir / f"DFM{dfm_id}_{suffix}.png", label)
                    for subdir, suffix, label in self._QC_PLOT_KINDS]
            ## The optogenetic light QC's figure, when a run wrote one.
            pngs.append((qc_dir / "opto" / f"opto_light_dfm{dfm_id}.png",
                         "Light explained"))
            for png, label in pngs:
                if not png.is_file():
                    continue
                self.dock.add_widget(f"DFM{dfm_id} {label}",
                                     ZoomableImageView(png),
                                     icon("qc", Category.QC),
                                     replace_existing=True)
                shown += 1
        if shown:
            self.log.append_line(
                f"[qc] Opened {shown} QC plot(s) from {qc_dir}.")
        else:
            QMessageBox.information(
                self, "No QC plots",
                "No QC plots on disk yet — run 'QC reports' first.")

    def _open_qc_folder(self) -> None:
        if not self._require_experiment():
            return
        qc_dir = self._member_qc_dir()
        if not qc_dir.is_dir():
            QMessageBox.information(
                self, "No QC folder",
                "This member has no qc/ folder yet — run 'QC reports' "
                "first.")
            return
        self._open_externally(qc_dir)

    def _open_script_editor(self) -> None:
        """The Script Editor on the **loaded member's** flic_config.yaml.

        One button, one file.  Project Scripts have their own button in the
        Project panel, because "whichever level happens to be selected" is not
        something a Save should depend on.
        """
        if not self._require_experiment():
            return
        config = os.path.join(str(self.experiment.experiment_dir),
                              "flic_config.yaml")
        self._show_script_editor(config)

    def _open_project_script_editor(self) -> None:
        if not self._require_project():
            return
        self._show_script_editor(
            os.path.join(self.project.project_directory,
                         project_mod.PROJECT_FILENAME))

    def _show_script_editor(self, config: str) -> None:
        from .script_editor import ScriptEditorWindow

        self._script_editor = ScriptEditorWindow(config)
        self._script_editor.show()
        self._script_editor.raise_()
        self._script_editor.activateWindow()

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

    def _yaml_validation_targets(self, root: str | None = None) -> list[Path]:
        """Every YAML *root* (default: the selection) is responsible for,
        widest first.

        A Batch has one per Project plus one per member; a Project has its own
        plus its members'.  The point is to find the one bad file *before* an
        unattended run does — which means checking the ones the user never
        opens, not only the one they are looking at.
        """
        root = root or self._current_dir()
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

    def _validate_yaml(self, root: str | None = None) -> None:
        """Parse every YAML under *root* (default: the selection) and report
        what fails.

        Cheap, read-only, and the only way to learn that a hand-edited config
        three folders down is unparseable without waiting for hour three of a
        Batch Run to say so.
        """
        import yaml as _yaml

        root = root or self._current_dir()
        targets = self._yaml_validation_targets(root)
        if not targets:
            QMessageBox.information(self, "Nothing selected",
                                    "Open a Batch, Project, or member first.")
            return
        root = Path(root)

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
        ## "Open Project", the control that fixes the missing state.
        tile.set_dimmed(False)
        ## The whole Analysis card appears only with a Project open; the
        ## Experiments card stays visible-but-gated so an empty state still
        ## shows where members will appear.
        self.project_analysis_card.setVisible(self.project is not None)
        if self.project is None:
            tile.set_summary(["no project open", "open one to begin"])
            self.project_summary.setText("")
            for widget in (self.file_btn, self.scaffold_btn,
                           self.view_reports_btn, self.project_sheet_btn,
                           self.project_script, self.run_project_script_btn,
                           self.edit_project_scripts_btn,
                           self.create_member_btn, self.init_member_btn,
                           ## The editor for the Project that is open, so it
                           ## waits for one.  The two buttons beside it are
                           ## the ways to make a Project when there is none.
                           self.design_btn):
                widget.setEnabled(False)
            self.design_btn.setText("Project design…")
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
        for widget in (self.project_script, self.run_project_script_btn,
                       self.edit_project_scripts_btn, self.design_btn,
                       self.create_member_btn, self.init_member_btn):
            widget.setEnabled(True)
        ## Initializing needs a candidate: a folder in the Project with no
        ## config of its own.  Disabled but present, so the button still says
        ## the case exists.
        pending_dirs = layout_mod.initializable_dirs(project.project_directory)
        self.init_member_btn.setEnabled(bool(pending_dirs))
        self.init_member_btn.setText(
            f"Initialize existing directory… ({len(pending_dirs)})"
            if pending_dirs else "Initialize existing directory…")
        ## A Project with no design: validates its members against each other
        ## rather than against an authority — say so on the button that fixes
        ## it, since nothing else in the Hub would ever mention it.
        declared = bool(project.design_global)
        self.design_btn.setText(
            "Project design…" if declared else "Project design… (none set)")
        self.design_btn.setToolTip(
            self.design_btn.toolTip().split("\n\n")[0]
            + ("" if declared else
               "\n\nThis Project declares no design: — its members are "
               "validated against each other instead."))

        ## The loaded-project description, in the always-visible Create/Load
        ## card, so it reads even while the sections below stay down.
        parts = [f"<b>{project.name}</b> — "
                 f"{project.experiment_type.display_name} · "
                 f"{project.chamber_layout}",
                 f"{len(project.member_names)} member(s), {analyzed} analyzed"]
        if not declared:
            parts.append("no design declared — members are validated against "
                         "each other")
        for warning in getattr(project, "warnings", []) or []:
            parts.append(f"⚠ {warning}")
        self.project_summary.setText("<br>".join(parts))

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
        self._refresh_experiment_group_tile()
        for key in ("analyze", "qc", "plots"):
            self.tiles[key].set_dimmed(not loaded)
        if not loaded:
            self.qc_hint.setText("")
            for tile_key in ("analyze", "qc", "plots"):
                self.tiles[tile_key].set_summary(["no member loaded", ""])
            self.plot_metric.clear()
            for group in (*getattr(self, "_type_plot_groups", {}).values(),
                          *getattr(self, "_type_analyze_groups", {}).values(),
                          *getattr(self, "_type_qc_groups", {}).values()):
                group.setVisible(False)
            self.experiment_script.clear()
            self.scripts_hint.setText(
                "No member is loaded — Experiment Scripts run on one.  "
                "Double-click a member row in the Project panel.  A Project's "
                "own scripts are in the Project panel.")
            self.tiles["scripts"].set_dimmed(True)
            self._refresh_scripts_tile()
            return

        exp = self.experiment
        layout = getattr(exp, "chamber_layout", None) or "two_well"
        type_name = getattr(getattr(exp, "experiment_type", None), "name", "Custom")
        self.tiles["analyze"].set_summary([str(self.experiment_name),
                                           f"{type_name} · {layout}"])
        qc_dir = self._member_qc_dir()
        qc_on_disk = qc_dir is not None and qc_dir.is_dir()
        self.qc_hint.setText(
            "" if qc_on_disk else
            "No QC reports on disk yet; 'QC reports' writes them.")
        self.tiles["qc"].set_summary(
            [str(self.experiment_name),
             "reports on disk" if qc_on_disk else "no reports yet"])
        ## Labels, not windows: a Progressive Ratio member has two Facets
        ## with no fixed windows at all (ADR-0013).
        labels_of = getattr(exp, "facet_labels", None)
        facets = len(labels_of()) if callable(labels_of) else len(exp.facet_windows())
        self.tiles["plots"].set_summary(
            [str(self.experiment_name),
             f"{facets} facet(s)" if facets else "no facets"])
        ## Type-specific plot groups, keyed by the same ``requires`` string
        ## the Script Editor's action catalogue gates on — one spelling of
        ## "this belongs to that Experiment Type", not two.
        from .script_editor.actions import requires_key_for

        active = requires_key_for(type_name)
        for key, group in (*getattr(self, "_type_plot_groups", {}).items(),
                           *getattr(self, "_type_analyze_groups", {}).items(),
                           *getattr(self, "_type_qc_groups", {}).items()):
            group.setVisible(key == active)
        ## The Optogenetics QC group follows the member, not the type: the
        ## light QC is every optogenetic experiment's, and Progressive Ratio's
        ## own light checks sit in it for that type.
        opto_group = getattr(self, "_type_qc_groups", {}).get("optogenetics")
        if opto_group is not None:
            try:
                optogenetic = bool(getattr(exp, "is_optogenetic", False))
            except Exception:  # noqa: BLE001 - a status display must not fail
                optogenetic = False
            is_pr = active == "progressive_ratio"
            opto_group.setVisible(optogenetic or is_pr)
            for button in getattr(self, "_opto_qc_buttons", []):
                button.setVisible(optogenetic)
            for button in getattr(self, "_pr_qc_buttons", []):
                button.setVisible(is_pr)
            ## Its help button sits beside the first button still shown.
            opto_group.reflow_title_widget()
        for button in getattr(self, "_two_well_plot_buttons", []):
            button.setVisible(layout != "single_well")

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
        central = len(self.project.experiment_scripts) \
            if self.project is not None else 0
        self.scripts_hint.setText(
            f"Experiment Scripts for '{self.experiment_name}'"
            + (f" — {central} of them served centrally from the Project's "
               f"experiment_scripts:." if central else "."))
        self.tiles["scripts"].set_dimmed(False)
        self._refresh_scripts_tile()

    def _refresh_experiment_group_tile(self) -> None:
        """The gate on the four subtiles: dimmed AND inert with nothing
        loaded — it opens no panel, so a click could not show the fix; the
        hint names where it is instead."""
        tile = self.tiles["experiment"]
        if self.experiment is None:
            tile.set_summary(["no member loaded",
                              "double-click one in Project"])
            ## Collapse only on the lit→dimmed unload transition, not on
            ## every refresh — a programmatically opened subtile panel must
            ## survive the refresh that follows every finished task.
            if not tile.is_dimmed():
                self._collapse_experiment()
            tile.set_dimmed(True)
            tile.set_clickable(False)
            return
        exp = self.experiment
        layout = getattr(exp, "chamber_layout", None) or "two_well"
        type_name = getattr(getattr(exp, "experiment_type", None), "name",
                            "Custom")
        tile.set_summary([str(self.experiment_name),
                          f"{type_name} · {layout}"])
        tile.set_dimmed(False)
        tile.set_clickable(True)

    def _refresh_scripts_tile(self) -> None:
        count = self.experiment_script.count()
        has_member = self.experiment is not None
        self.tiles["scripts"].set_summary([
            str(self.experiment_name) if has_member else "no member loaded",
            f"{count} experiment script(s)" if has_member
            else "load one to run scripts",
        ])
        for widget in (self.experiment_script, self.run_experiment_script_btn,
                       self.edit_scripts_btn):
            widget.setEnabled(has_member)

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
        ## Grouped under the Experiment tile, so it follows the group's gate —
        ## AND the Project's, because the narrative is about the Project's
        ## Combined Analysis and generating one refuses without a Project.  A
        ## standalone experiment (loaded with no Project) must not light a
        ## tile whose only action is a dead end; the Analysis card carries the
        ## project-level entry point.
        tile.set_dimmed(self.experiment is None or self.project is None)
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
        has_experiment = self.experiment is not None
        dim = {
            "batch": False,          # its panel holds "Choose batch folder…"
            "project": False,        # its panel holds "Open Project"
            "analyze": not has_experiment,
            "qc": not has_experiment,
            "plots": not has_experiment,
            ## Scripts is the member level now: with no member loaded there
            ## is no script to pick and no config to edit.
            "scripts": not has_experiment,
            ## AI follows the Experiment group's gate, like its subtile — and
            ## the Project's, because generating a narrative needs one.
            "ai": not (has_experiment and self.project is not None),
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
        ## Window-modal, not application-modal.  Edit opens a config editor as
        ## a separate top-level window; an application-modal dialog blocks
        ## input to every other window in the app, so that editor came up
        ## looking alive but ignoring clicks, keys and the scroll wheel.
        ## Window modality blocks only this dialog's parent chain (the hub),
        ## leaving the parentless editor interactive.
        self.setWindowModality(Qt.WindowModality.WindowModal)

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
        ## File a loose recording first, the same order Initialize existing
        ## directory… uses: the scaffold reconciles its dfms: against data/,
        ## and a recording still at the root is invisible to it — the config
        ## would list no DFMs with the data sitting right there.
        directory = self._project.member_dir(name)
        state = layout_mod.classify(directory)
        if state.status == layout_mod.UNFILED:
            plan = layout_mod.file_recording(directory,
                                             log=self._hub.log.append_line)
            if plan.refused:
                QMessageBox.warning(
                    self, "Could not scaffold",
                    f"Could not file '{name}': {plan.refused}")
                return
            self._hub.log.append_line(f"[configs] {name}: {plan.describe()}")
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
        path = os.path.join(self._project.member_dir(chosen[0]),
                            project_mod.CONFIG_FILENAME)
        self._hub._open_config_editor_on(path)


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
