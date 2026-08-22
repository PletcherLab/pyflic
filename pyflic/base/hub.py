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
  Project panel's replicates table.  There is no Load tile: the load options
  live in the Project panel beside the table that triggers the load.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import batch as batch_mod
from . import project as project_mod
from .ui import Category, OutputLog, PlotDock, apply_theme, icon, surface_colors
from .ui import settings as ui_settings
from .ui.tiles import TILE_HEIGHT, ClickAwayFilter, StatusReadout, StatusTile, TilePanel
from .ui.widgets import ActionButton, Card

#: (key, title, icon, category, panel width).  Order is strip order.
TILE_SPECS: list[tuple[str, str, str, Category, int]] = [
    ("batch",   "Batch",   "batch",    Category.NEUTRAL, 560),
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
        self.batch: batch_mod.Batch | None = None
        self.project: project_mod.Project | None = None
        self.experiment_name: str | None = None
        self.experiment = None
        self._worker: Worker | None = None
        self._open_key: str | None = None

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
        self.log = OutputLog()
        self.dock = PlotDock(self.log)
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

    #: Tile key → help topic reference. Adding a tile means adding a line here;
    #: ``tests/test_help_refs.py`` asserts every value resolves.
    _TILE_HELP: dict[str, str] = {
        "batch": "scripts-batch",
        "project": "concepts-project",
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
        card = Card("Batch", Category.NEUTRAL, icon_name="batch",
                    subtitle="A directory whose immediate subdirectories "
                             "holding a project.yaml are its Projects. A Batch "
                             "Run executes one designated Project Script in "
                             "each, continue-on-error.")
        row = QHBoxLayout()
        row.addWidget(ActionButton("Open a Batch folder…", Category.NEUTRAL,
                                   "open", primary=True))
        row.itemAt(0).widget().clicked.connect(self._choose_batch)
        card.add_body(row)

        self.batch_table = self._table(["Project", "Replicates", "Analyzed",
                                        "Report"])
        self.batch_table.itemDoubleClicked.connect(self._batch_row_activated)
        card.add_body(self.batch_table)

        run_row = QHBoxLayout()
        run_row.addWidget(QLabel("Project Script:"))
        self.batch_script = QComboBox()
        self.batch_script.setMinimumWidth(200)
        run_row.addWidget(self.batch_script, 1)
        self.batch_run_btn = ActionButton("Run Batch", Category.SCRIPTS, "play",
                                          primary=True)
        self.batch_run_btn.clicked.connect(self._run_batch)
        run_row.addWidget(self.batch_run_btn)
        card.add_body(run_row)
        card.add_section_label(
            "Double-click a project to make it the working container.")
        self.panels["batch"].add_card(card)

    def _build_project_panel(self) -> None:
        card = Card("Project", Category.LOAD, icon_name="project",
                    subtitle="Replicates of one design, pooled. "
                             "Double-click a replicate to load it.")
        row = QHBoxLayout()
        open_btn = ActionButton("Open a Project…", Category.LOAD, "open",
                                primary=True)
        open_btn.clicked.connect(self._choose_project)
        row.addWidget(open_btn)
        new_btn = ActionButton("New Project here…", Category.LOAD, "new")
        new_btn.clicked.connect(self._create_project)
        row.addWidget(new_btn)
        card.add_body(row)

        self.project_table = self._table(["Replicate", "DFMs", "Chambers",
                                          "Analyzed", "Report"])
        self.project_table.itemDoubleClicked.connect(self._project_row_activated)
        card.add_body(self.project_table)

        self.scaffold_row = QHBoxLayout()
        self.scaffold_btn = ActionButton("Scaffold pending replicates",
                                         Category.LOAD, "new")
        self.scaffold_btn.clicked.connect(self._scaffold_pending)
        self.scaffold_row.addWidget(self.scaffold_btn)
        card.add_body(self.scaffold_row)

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
            ("Analyze all", "basic", self._run_all_replicates),
            ("Combine", "csv", self._build_combined),
            ("Create report", "pdf", self._project_report),
        ):
            button = ActionButton(label, Category.ANALYZE, icon_name)
            button.clicked.connect(handler)
            acts.addWidget(button)
        card.add_body(acts)
        self.panels["project"].add_card(card)

    def _build_analyze_panel(self) -> None:
        card = Card("Analyze", Category.ANALYZE, icon_name="analyze",
                    subtitle="Actions on the loaded replicate.")
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
                    subtitle="Quick figures for the loaded replicate, and the "
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

        card.add_section_label("Experiment Scripts (loaded replicate)")
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
            ("Lint / migration check", "lint", self._run_lint),
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
                self._set_batch(batch_mod.Batch(path))
            elif project_mod.is_experiment_dir(path):
                ## A standalone Experiment Directory has no Project above it.
                ## Rather than refuse, treat its parent as the container and
                ## load it directly — the Hub is Project-first, not
                ## Project-only.
                self._set_batch(None)
                self._set_project(None)
                self._load_standalone(path)
            else:
                self.log.append_line(
                    f"{path} is not a Batch, a Project, or an Experiment "
                    f"Directory.")
        except Exception as err:  # noqa: BLE001
            QMessageBox.critical(self, "Could not open", str(err))
            self.log.append_line(f"ERROR: {err}")
        self.refresh()

    def _choose_batch(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose a Batch folder")
        if path:
            self._set_batch(batch_mod.Batch(path))
            self._set_project(None)
            self.refresh()

    def _choose_project(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose a Project folder")
        if not path:
            return
        try:
            self._set_project(project_mod.Project(path))
        except Exception as err:  # noqa: BLE001
            QMessageBox.critical(self, "Not a Project", str(err))
            self.log.append_line(f"ERROR: {err}")
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
        self.refresh()

    def _set_batch(self, batch) -> None:
        self.batch = batch
        if batch is not None:
            self.log.append_line(
                f"Batch: {batch.batch_directory} — {len(batch)} project(s)")

    def _set_project(self, project) -> None:
        self.project = project
        self.experiment = None
        self.experiment_name = None
        if project is not None:
            self.log.append_line(
                f"Project: {project.project_directory} — "
                f"{len(project.experiment_names)} replicate(s)")
            for warning in project.warnings:
                self.log.append_line(f"  note: {warning}")

    def _batch_row_activated(self, item) -> None:
        """Double-clicking a project row is an ordinary selection change down to
        that Project — no drill-in state, no up-button."""
        if self.batch is None:
            return
        name = self.batch_table.item(item.row(), 0).text()
        try:
            self._set_project(project_mod.Project(
                os.path.join(self.batch.batch_directory, name)))
        except Exception as err:  # noqa: BLE001
            QMessageBox.critical(self, "Not a Project", str(err))
            return
        self.refresh()

    def _project_row_activated(self, item) -> None:
        if self.project is None:
            return
        name = self.project_table.item(item.row(), 0).text()
        self._load_replicate(name)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_kwargs(self) -> dict:
        workers = self.spin_workers.value()
        return {"parallel": self.chk_parallel.isChecked(),
                "max_workers": workers or None}

    def _load_replicate(self, name: str) -> None:
        project = self.project
        if project is None:
            return

        def task():
            return project.load_experiment(name, **self._load_kwargs())

        def done(exp):
            self.experiment = exp
            self.experiment_name = name
            self.refresh()

        self._start(task, f"Loading replicate '{name}'", done)

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
        for line in str(text).splitlines():
            if line.strip():
                self.log.append_line(line)

    def _on_worker_failed(self, message: str) -> None:
        self._set_running(False)
        self.log.append_line(message)
        first = message.strip().splitlines()[-1] if message.strip() else "failed"
        QMessageBox.critical(self, "Task failed", first)

    def _set_running(self, running: bool) -> None:
        """Grey the open panel's cards in place rather than closing it — a task
        finishing should not move the UI out from under the user."""
        if self._open_key is not None:
            self.panels[self._open_key].setEnabled(not running)

    def _show_figures(self, result) -> None:
        if not isinstance(result, list):
            return
        for entry in result:
            if isinstance(entry, tuple) and len(entry) == 2:
                title, figure = entry
                try:
                    self.dock.add_figure(str(title), figure)
                except Exception as err:  # noqa: BLE001
                    self.log.append_line(f"could not show '{title}': {err}")

    # ---- experiment-level -------------------------------------------

    def _require_experiment(self) -> bool:
        if self.experiment is None:
            QMessageBox.information(
                self, "No replicate loaded",
                "Double-click a replicate in the Project panel to load it.")
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

    def _run_all_replicates(self) -> None:
        if not self._require_project():
            return
        project = self.project

        def task():
            failures = project.run_all()
            if failures:
                raise RuntimeError("; ".join(failures))
            return None

        self._start(task, "Analyze all replicates")

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

    def _scaffold_pending(self) -> None:
        if not self._require_project():
            return
        project = self.project
        pending = project.unconfigured_dirs()
        if not pending:
            QMessageBox.information(
                self, "Nothing to scaffold",
                "Every folder holding DFM CSVs already has a "
                "flic_config.yaml.")
            return
        for name in pending:
            try:
                path, notes = project.scaffold_replicate(name)
                self.log.append_line(f"Scaffolded {name}: {path}")
                for note in notes:
                    self.log.append_line(f"  {note}")
            except Exception as err:  # noqa: BLE001
                self.log.append_line(f"{name}: FAILED — {err}")
        self._set_project(project_mod.Project(project.project_directory))
        self.refresh()

    # ---- batch -------------------------------------------------------

    def _run_batch(self) -> None:
        if self.batch is None:
            QMessageBox.information(self, "No Batch", "Open a Batch folder first.")
            return
        batch = self.batch
        name = self.batch_script.currentText() or None
        self._start(lambda: batch.run(name),
                    f"Batch Run '{name or batch.script_name}'",
                    lambda _r: self.refresh())

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
        if self.batch is not None:
            return self.batch.batch_directory
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
        ## project.yaml, a replicate's in its flic_config.yaml — so the file to
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
                "Open a Project, or load a replicate, first.")
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
                                    "Open a Batch, Project, or replicate first.")
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
        self._refresh_readout()

    def _refresh_batch(self) -> None:
        tile = self.tiles["batch"]
        table = self.batch_table
        table.setRowCount(0)
        if self.batch is None:
            tile.set_dimmed(True)
            tile.set_summary(["no batch open", "open a folder of Projects"])
            self.batch_script.clear()
            return
        tile.set_dimmed(False)
        names = self.batch.project_names
        for row, name in enumerate(names):
            table.insertRow(row)
            table.setItem(row, 0, QTableWidgetItem(name))
            path = os.path.join(self.batch.batch_directory, name)
            try:
                project = project_mod.Project(path)
                reps = len(project.experiment_names)
                analyzed = sum(1 for n in project.experiment_names
                               if project.experiment_status(n)["analyzed"])
                report = os.path.isfile(
                    os.path.join(path, f"{project.name}_report.pdf"))
                cells = [str(reps), f"{analyzed}/{reps}",
                         "yes" if report else "no"]
            except Exception as err:  # noqa: BLE001
                cells = ["—", "—", f"error: {type(err).__name__}"]
            for col, text in enumerate(cells, start=1):
                table.setItem(row, col, QTableWidgetItem(text))
        tile.set_summary([f"{len(names)} project(s)",
                          os.path.basename(self.batch.batch_directory)])
        current = self.batch_script.currentText()
        self.batch_script.clear()
        from .script_editor.project_actions import BUILTIN_PROJECT_SCRIPTS

        options = [self.batch.script_name]
        options += [s["name"] for s in self.batch.project_scripts]
        options += list(BUILTIN_PROJECT_SCRIPTS)
        seen: list[str] = []
        for option in options:
            if option not in seen:
                seen.append(option)
        self.batch_script.addItems(seen)
        if current in seen:
            self.batch_script.setCurrentText(current)

    def _refresh_project(self) -> None:
        tile = self.tiles["project"]
        table = self.project_table
        table.setRowCount(0)
        self.project_script.clear()
        if self.project is None:
            tile.set_dimmed(True)
            tile.set_summary(["no project open", "open one to begin"])
            self.scaffold_btn.setEnabled(False)
            return
        tile.set_dimmed(False)
        project = self.project
        analyzed = 0
        for row, name in enumerate(project.experiment_names):
            status = project.experiment_status(name)
            analyzed += 1 if status["analyzed"] else 0
            table.insertRow(row)
            table.setItem(row, 0, QTableWidgetItem(name))
            for col, text in enumerate(
                    [str(status["dfms"]),
                     str(status["chambers"]) if status["chambers"] is not None else "—",
                     "yes" if status["analyzed"] else "no",
                     "yes" if status["report"] else "no"], start=1):
                table.setItem(row, col, QTableWidgetItem(text))
        pending = project.unconfigured_dirs()
        self.scaffold_btn.setEnabled(bool(pending))
        self.scaffold_btn.setText(
            f"Scaffold {len(pending)} pending replicate(s)" if pending
            else "Scaffold pending replicates")
        summary = [f"{len(project.experiment_names)} replicate(s)",
                   f"{analyzed} analyzed"]
        if self.experiment_name:
            summary[1] = f"loaded: {self.experiment_name}"
        tile.set_summary(summary)

        from .script_editor.project_actions import BUILTIN_PROJECT_SCRIPTS

        names = [s["name"] for s in project.scripts]
        names += [n for n in BUILTIN_PROJECT_SCRIPTS if n not in names]
        self.project_script.addItems(names)

    def _refresh_experiment_tiles(self) -> None:
        loaded = self.experiment is not None
        for key in ("analyze", "plots"):
            self.tiles[key].set_dimmed(not loaded)
        if not loaded:
            for tile_key, hint in (("analyze", self.analyze_hint),
                                   ("plots", self.plots_hint)):
                hint.setText("No replicate is loaded. Double-click a replicate "
                             "row in the Project panel to load one.")
                self.tiles[tile_key].set_summary(["no replicate loaded", ""])
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

    def _refresh_readout(self) -> None:
        rows: list[tuple[str, str]] = []
        if self.batch is not None:
            rows.append(("Batch", f"{os.path.basename(self.batch.batch_directory)} "
                                  f"({len(self.batch)} projects)"))
        if self.project is not None:
            rows.append(("Project", f"{self.project.name} "
                                    f"({len(self.project.experiment_names)} replicates)"))
            rows.append(("Design", f"{self.project.experiment_type.display_name} · "
                                   f"{self.project.chamber_layout}"))
        else:
            rows.append(("Project", "none open"))
        rows.append(("Loaded", self.experiment_name or "no replicate loaded"))
        self.readout.set_rows(rows)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self._click_filter)
        super().closeEvent(event)


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("pyflic Analysis Hub")
    apply_theme(app, mode=ui_settings.get("theme", "auto"))
    target = sys.argv[1] if len(sys.argv) > 1 else None
    window = AnalysisHubWindow(target=target)
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
