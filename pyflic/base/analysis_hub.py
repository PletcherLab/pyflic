"""
FLIC Analysis Hub
=================
PyQt6 launcher for common pyflic analysis workflows. Opens the config editor and
QC viewer in separate processes. Runs analysis in a background thread.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import yaml
from PyQt6.QtCore import QEvent, QObject, QSize, QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .ui import (
    ActionButton,
    Card,
    Category,
    OutputLog,
    PlotDock,
    SidebarNav,
    TopBar,
    apply_theme,
    icon,
    resolved_mode,
)
from .ui import settings as ui_settings

# ---------------------------------------------------------------------------
# Metric definitions for plot controls
# Each entry: (display label, metric arg, two_well_mode arg)
# ---------------------------------------------------------------------------
_TWO_WELL_BINNED: list[tuple[str, str, str]] = [
    ("Licks (A+B total)",      "Licks",         "total"),
    ("PI",                     "PI",            "total"),
    ("Event PI",               "EventPI",       "total"),
    ("Licks A",                "LicksA",        "A"),
    ("Licks B",                "LicksB",        "B"),
    ("Events (A+B total)",     "Events",        "total"),
    ("Med Duration (A+B avg)", "MedDuration",   "mean_ab"),
    ("Med Duration A",         "MedDurationA",  "A"),
    ("Med Duration B",         "MedDurationB",  "B"),
    ("Mean Duration (A+B avg)","MeanDuration",  "mean_ab"),
    ("Med Time Btw (A+B avg)", "MedTimeBtw",    "mean_ab"),
]

_SINGLE_WELL_BINNED: list[tuple[str, str, str]] = [
    ("Licks",         "Licks",       "total"),
    ("Events",        "Events",      "total"),
    ("Med Duration",  "MedDuration", "total"),
    ("Mean Duration", "MeanDuration","total"),
    ("Med Time Btw",  "MedTimeBtw",  "total"),
    ("Mean Int",      "MeanInt",     "total"),
    ("Median Int",    "MedianInt",   "total"),
]

# Base metric names for the well A vs B comparison (facet_plot_well_durations)
_WELL_CMP_METRICS: list[str] = [
    "MedDuration",
    "MeanDuration",
    "Licks",
    "MedTimeBtw",
    "MeanTimeBtw",
    "MeanInt",
    "MedianInt",
]

# Default two_well_mode for each metric (used by script runner)
_METRIC_DEFAULT_MODE: dict[str, str] = {
    metric: mode for _, metric, mode in _TWO_WELL_BINNED + _SINGLE_WELL_BINNED
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_deleted(widget) -> bool:
    """Return True if the underlying C++ object has been destroyed."""
    try:
        widget.objectName()
        return False
    except RuntimeError:
        return True


def _is_child_of(widget, parent) -> bool:
    """Walk up the widget tree to check parentage."""
    try:
        w = widget.parentWidget()
        while w is not None:
            if w is parent:
                return True
            w = w.parentWidget()
    except RuntimeError:
        return True
    return False


# Directory names pruned during the recursive batch-target walk, in addition
# to any name starting with ``.``. Output trees pyflic creates inside a
# project (`analysis/`, `plots/`, `qc/`) plus common tooling clutter never
# contain valid batch YAMLs and would only slow the walk and pollute logs.
_BATCH_SKIP_DIR_NAMES = frozenset({
    "__pycache__", "node_modules", "analysis", "plots", "qc",
})


def _parse_scripts(cfg: dict) -> list[dict]:
    """Return list of script dicts from YAML config, or [] if none defined."""
    raw = cfg.get("scripts") or []
    if not isinstance(raw, list):
        return []
    result = []
    for item in raw:
        if isinstance(item, dict) and item.get("name") and isinstance(item.get("steps"), list):
            result.append(item)
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    return data if isinstance(data, dict) else {}


def _chamber_size_from_cfg(cfg: dict[str, Any]) -> int | None:
    g = cfg.get("global")
    if isinstance(g, dict):
        params = g.get("params") or g.get("parameters")
        if isinstance(params, dict) and params.get("chamber_size") is not None:
            return int(params["chamber_size"])
    dfms = cfg.get("dfms") or cfg.get("DFMs")
    nodes: list[Any]
    if isinstance(dfms, dict):
        nodes = list(dfms.values())
    elif isinstance(dfms, list):
        nodes = dfms
    else:
        nodes = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        p = node.get("params") or node.get("parameters")
        if isinstance(p, dict) and p.get("chamber_size") is not None:
            return int(p["chamber_size"])
    return None


def _norm_et(raw: Any) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    return s or None


def _list_yaml_configs(project_dir: Path) -> list[str]:
    """Return sorted list of ``*.yaml`` / ``*.yml`` filenames in *project_dir*.

    ``flic_config.yaml`` is placed first when present.
    """
    if not project_dir.is_dir():
        return []
    names = sorted(
        {p.name for p in project_dir.iterdir()
         if p.is_file() and p.suffix.lower() in (".yaml", ".yml")}
    )
    if "flic_config.yaml" in names:
        names.remove("flic_config.yaml")
        names.insert(0, "flic_config.yaml")
    return names


def _list_subdir_projects(project_dir: Path, config_name: str) -> list[Path]:
    """Return immediate subdirectories of *project_dir* that contain *config_name*."""
    if not project_dir.is_dir():
        return []
    return sorted(
        d for d in project_dir.iterdir()
        if d.is_dir() and (d / config_name).is_file()
    )


def read_project_meta(project_dir: Path, config_name: str = "flic_config.yaml") -> dict[str, Any]:
    """Lightweight parse of the selected YAML config for status display (no DFM load)."""
    cfg_path = project_dir / config_name
    if not cfg_path.is_file():
        return {"ok": False, "error": f"No {config_name} in {project_dir}"}
    try:
        cfg = _read_yaml(cfg_path)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"Could not read config: {e}"}
    g = cfg.get("global")
    et = _norm_et(g.get("experiment_type") if isinstance(g, dict) else None)
    cs = _chamber_size_from_cfg(cfg)
    inferred = et
    if inferred is None and cs == 1:
        inferred = "single_well"
    elif inferred is None and cs == 2:
        inferred = "two_well"
    return {
        "ok": True,
        "experiment_type": et,
        "inferred_type": inferred,
        "chamber_size": cs,
        "config_path": cfg_path,
    }


def _resolve_cli(name: str, module: str) -> list[str]:
    exe = shutil.which(name)
    if exe:
        return [exe]
    return [sys.executable, "-m", module]


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class _SignalWriter:
    """File-like object that emits a Qt signal on each line written."""

    def __init__(self, signal: pyqtSignal) -> None:
        self._signal = signal
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._signal.emit(line)
        return len(text)

    def flush(self) -> None:
        if self._buf:
            self._signal.emit(self._buf)
            self._buf = ""


class _WheelToScroll(QObject):
    """Event filter that redirects wheel events to a scroll area.

    Closed combo boxes and spin boxes accept the mouse wheel by default and
    change their value on scroll, which makes scrolling a tall control panel
    jump values under the cursor.  Installed on those widgets, this filter
    forwards their wheel events to the scroll area's vertical scrollbar and
    swallows them at the widget, so the wheel always scrolls the panel no
    matter where the cursor sits.  An open combo-box popup is a separate
    top-level widget and is not affected, so list scrolling still works.
    """

    def __init__(self, scroll_area: QScrollArea) -> None:
        super().__init__(scroll_area)
        self._scroll = scroll_area

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.Wheel:
            QApplication.sendEvent(self._scroll.verticalScrollBar(), event)
            return True
        return False


class AnalysisWorker(QObject):
    """Runs a callable in a worker thread; streams stdout/stderr to the log.

    Task callables may return one of:
    - ``None``                                      — no figure to display
    - ``(title: str, fig: object)``                 — single figure
    - ``[(title, fig), ...]``                       — multiple figures

    *fig* may be a matplotlib ``Figure`` or a plotnine ``ggplot``.  Plotnine
    objects are drawn here on the worker thread so the GUI thread receives a
    ready-to-embed matplotlib ``Figure``.
    """

    log = pyqtSignal(str)
    failed = pyqtSignal(str)
    finished = pyqtSignal()
    figure_ready = pyqtSignal(str, object)

    def __init__(self, task: Callable[[], Any]) -> None:
        super().__init__()
        self._task = task

    def run(self) -> None:
        writer = _SignalWriter(self.log)
        result = None
        try:
            from contextlib import redirect_stderr, redirect_stdout

            with redirect_stdout(writer), redirect_stderr(writer):
                result = self._task()
            writer.flush()
        except Exception as e:  # noqa: BLE001
            writer.flush()
            self.failed.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        finally:
            # Normalize result → list of (title, figure_or_ggplot) tuples.
            # Plotnine ggplot objects are passed through untouched; the GUI
            # thread calls ``.draw()`` to obtain a matplotlib Figure for
            # embedding (calling it here would warn about creating a GUI
            # canvas off the main thread).
            figures: list[tuple[str, Any]] = []
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, tuple) and len(item) == 2 and item[1] is not None:
                        figures.append((str(item[0]), item[1]))
            elif isinstance(result, tuple) and len(result) == 2 and result[1] is not None:
                figures.append((str(result[0]), result[1]))
            for title, fig in figures:
                self.figure_ready.emit(title, fig)
            self.finished.emit()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class AnalysisHubWindow(QMainWindow):
    _range_updated = pyqtSignal(float, float)   # emitted from script thread after load

    def __init__(self, project_dir: str | Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("pyflic — Analysis Hub")
        self.resize(1350, 860)
        self._initial_dir = Path(project_dir).expanduser().resolve() if project_dir else None

        self._thread: QThread | None = None
        self._worker: AnalysisWorker | None = None
        self._busy = False
        self._cached_exp: Any = None
        self._cached_exp_key: tuple | None = None
        self._exp_loaded: bool = False
        self._load_buttons: list[QPushButton] = []   # always enabled when not busy
        self._data_buttons: list[QPushButton] = []   # enabled only when exp loaded
        self._scripts: list[dict] = []
        self._active_config: str = "flic_config.yaml"
        self._cards: dict[str, Card] = {}
        self._sidebar_keys: list[str] = []

        # ── Top bar ───────────────────────────────────────────────────────
        self._top_bar = TopBar("pyflic — Analysis Hub")
        # Interactive-plot toggle (off by default — static PNG tabs).
        self._chk_interactive = QCheckBox("Interactive plots")
        self._chk_interactive.setToolTip(
            "When checked, plot tabs embed a live matplotlib canvas with "
            "pan/zoom/save toolbar and hover tooltips. Off: plot tabs show a "
            "static PNG snapshot (faster, less memory)."
        )
        self._chk_interactive.setChecked(False)
        self._top_bar.add_right(self._chk_interactive)

        # Theme toggle (right side)
        self._btn_theme = QToolButton()
        self._btn_theme.setIcon(icon("theme_dark" if resolved_mode() == "light" else "theme_light"))
        self._btn_theme.setIconSize(QSize(18, 18))
        self._btn_theme.setToolTip("Toggle light / dark theme")
        self._btn_theme.setAutoRaise(True)
        self._btn_theme.clicked.connect(self._toggle_theme)
        self._top_bar.add_right(self._btn_theme)

        # ── Sidebar nav ───────────────────────────────────────────────────
        self._sidebar = SidebarNav()
        for key, label, icon_name, cat in [
            ("project",  "Project",  "project",  Category.NEUTRAL),
            ("load",     "Load",     "load",     Category.LOAD),
            ("analyze",  "Analyze",  "basic",    Category.ANALYZE),
            ("plots",    "Plots",    "plots",    Category.PLOTS),
            ("scripts",  "Scripts",  "scripts",  Category.SCRIPTS),
            ("tools",    "Tools",    "tools",    Category.TOOLS),
        ]:
            self._sidebar.add_item(key, label, icon_name, category=cat)
            self._sidebar_keys.append(key)
        self._sidebar.add_stretch()
        self._sidebar.add_separator()
        _help_item = self._sidebar.add_item(
            "help", "Help", "help",
            category=Category.NEUTRAL,
            tooltip="Open the pyflic help window  (F1)",
        )
        # Amber, matching the [?] buttons on the cards.
        from .ui.icons import help_color as _help_color

        _help_item.setIcon(icon("help", color=_help_color()))
        # Help opens a window rather than scrolling to a card, so it must not
        # take the sidebar's checked state away from the current section.
        _help_item.setCheckable(False)
        self._sidebar.itemSelected.connect(self._on_sidebar_selected)

        # ── Cards (built once; Analyze / Plots get rebuilt on meta refresh)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        # Button labels shrink (min-width 0), so they never push the cards wider
        # than the column.  Show a horizontal scrollbar only if some control
        # genuinely overflows, so nothing is clipped out of reach.
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # Redirect wheel events from combo boxes / spin boxes to the panel so
        # the wheel always scrolls instead of changing the hovered control.
        self._wheel_guard = _WheelToScroll(self._scroll)
        cards_host = QWidget()
        cards_host.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self._cards_lay = QVBoxLayout(cards_host)
        self._cards_lay.setContentsMargins(16, 12, 16, 12)
        self._cards_lay.setSpacing(12)
        self._scroll.setWidget(cards_host)

        self._build_card_project()
        self._build_card_load()
        # Analyze, Plots, Scripts, Tools cards are placeholders until meta
        # refresh populates them with experiment-aware buttons.
        self._cards["analyze"] = Card("Analyze", category=Category.ANALYZE,
                                      subtitle="Summaries, CSV exports, statistics",
                                      icon_name="basic")
        self._cards["plots"] = Card("Plots", category=Category.PLOTS,
                                    subtitle="Interactive figures (zoom / pan / hover)",
                                    icon_name="plot")
        self._cards["scripts"] = Card("Scripts", category=Category.SCRIPTS,
                                      subtitle="Multi-step recipes defined in YAML",
                                      icon_name="scripts")
        self._cards["tools"] = Card("Tools", category=Category.TOOLS,
                                    subtitle="Lint, compare configs, clear cache, edit YAML",
                                    icon_name="tools")
        for k in ("analyze", "plots", "scripts", "tools"):
            self._cards_lay.addWidget(self._cards[k])
        self._cards_lay.addStretch(1)

        # Scripts card is populated now (static); its visibility is managed
        # later by ``_refresh_script_dropdown``.  Analyze / Plots / Tools are
        # rebuilt dynamically after meta is known.
        self._build_card_scripts()
        self._install_wheel_guards()
        self._install_card_help()

        # ── Output / Plot dock ────────────────────────────────────────────
        self._log = OutputLog()
        self._plot_dock = PlotDock(self._log)

        # ── Compose central widget ────────────────────────────────────────
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._top_bar)

        # Cards on the left, plot dock on the right (2× wider, full height).
        # Wrap in a QSplitter so the user can resize the divide.
        from PyQt6.QtWidgets import QSplitter

        self._main_split = QSplitter(Qt.Orientation.Horizontal)
        self._main_split.setChildrenCollapsible(False)
        self._main_split.setHandleWidth(4)
        self._main_split.addWidget(self._scroll)
        self._main_split.addWidget(self._plot_dock)
        # Initial 1:2 ratio (cards : dock); user can drag.
        self._main_split.setStretchFactor(0, 1)
        self._main_split.setStretchFactor(1, 2)
        self._scroll.setMinimumWidth(320)
        self._plot_dock.setMinimumWidth(420)

        body_row = QHBoxLayout()
        body_row.setContentsMargins(0, 0, 0, 0)
        body_row.setSpacing(0)
        body_row.addWidget(self._sidebar)
        body_row.addWidget(self._main_split, 1)
        body_widget = QWidget()
        body_widget.setLayout(body_row)
        outer.addWidget(body_widget, 1)

        # ── Final wiring ──────────────────────────────────────────────────
        self._path_edit.editingFinished.connect(self._refresh_meta)
        self._spin_start.valueChanged.connect(self._invalidate_exp_cache)
        self._spin_end.valueChanged.connect(self._invalidate_exp_cache)
        self._range_updated.connect(self._on_range_updated)

        start_dir = self._initial_dir or Path.cwd()
        self._path_edit.setText(str(start_dir))
        self._refresh_meta()

        # Apply the 1:2 ratio for the cards/dock split based on the window's
        # initial width.  Stretch factors alone don't seed initial sizes.
        avail = max(900, self.width() - 180)  # subtract sidebar
        cards_w = avail // 3
        self._main_split.setSizes([cards_w, avail - cards_w])

    # ------------------------------------------------------------------
    # Help
    # ------------------------------------------------------------------

    #: Card key → help topic reference.  Adding a card means adding a line
    #: here; ``tests/test_help_refs.py`` fails if a reference goes stale.
    _CARD_HELP: dict[str, str] = {
        "project": "app-hub#project-card",
        "load": "app-hub#load-card",
        "analyze": "app-hub#analyze-card",
        "plots": "plots-catalog",
        "scripts": "scripts-overview",
        "tools": "app-hub#tools-card",
    }

    def _install_card_help(self) -> None:
        """Put a ``?`` in each card's title row and bind F1.

        Help is imported here rather than at module scope so a failure in the
        help package can never stop the hub from starting.
        """
        try:
            from ..help import HelpButton, install_help_shortcut
        except Exception:  # noqa: BLE001 - help is optional, the hub is not
            return
        for key, ref in self._CARD_HELP.items():
            card = self._cards.get(key)
            if card is not None:
                card.add_title_widget(HelpButton(ref, card))
        install_help_shortcut(self, "app-hub")

    def _open_help(self) -> None:
        from ..help import open_help

        open_help("getting-started")

    # ------------------------------------------------------------------
    # Card builders
    # ------------------------------------------------------------------

    def _build_card_project(self) -> None:
        card = Card("Project", category=Category.NEUTRAL,
                    subtitle="Choose the project directory and YAML config.",
                    icon_name="project")

        # Path row
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Folder:"))
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Path to project directory")
        self._path_edit.setMinimumWidth(80)
        self._path_edit.setSizePolicy(QSizePolicy.Policy.Expanding,
                                      QSizePolicy.Policy.Fixed)
        path_row.addWidget(self._path_edit, 1)
        browse = QToolButton()
        browse.setIcon(icon("browse"))
        browse.setToolTip("Browse…")
        browse.clicked.connect(self._browse_project)
        path_row.addWidget(browse)
        card.add_body(path_row)

        # Config row — full width now; no other widget competes for space.
        cfg_row = QHBoxLayout()
        cfg_row.addWidget(QLabel("Config:"))
        self._cmb_config = QComboBox()
        self._cmb_config.setMinimumWidth(80)
        self._cmb_config.setSizePolicy(QSizePolicy.Policy.Expanding,
                                       QSizePolicy.Policy.Fixed)
        self._cmb_config.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._cmb_config.currentTextChanged.connect(self._on_config_changed)
        cfg_row.addWidget(self._cmb_config, 1)
        card.add_body(cfg_row)

        # Batch-mode toggle on its own row.
        self._chk_all_yamls = QCheckBox("Run action for every YAML config")
        self._chk_all_yamls.setToolTip(
            "When checked, the next action button you press runs once for every "
            "YAML config in the project directory. Each run writes into its own "
            "subdirectory named after the YAML file's stem.\n\n"
            "Run Script: the dropdown shows the union of script names across "
            "all YAMLs; clicking Run Script executes each YAML's own script "
            "with that name (skipping YAMLs that don't define it)."
        )
        self._chk_all_yamls.toggled.connect(self._on_all_yamls_toggled)
        card.add_body(self._chk_all_yamls)

        self._chk_all_subdirs = QCheckBox(
            "Run 'batch' script in every directory under here"
        )
        self._chk_all_subdirs.setToolTip(
            "When checked, the Run Script button walks the project "
            "directory recursively (root included, arbitrary depth) and "
            "runs the script named 'batch' from every YAML that defines "
            "one. Each batch target is treated as its own project, so "
            "outputs land inside that target. Outputs are always "
            "overwritten.\n\n"
            "Directories with no YAMLs are silent; directories with "
            "YAMLs but no 'batch' script are logged as near-miss skips. "
            "Hidden dirs (starting with '.') and known output/tooling "
            "dirs (analysis, plots, qc, __pycache__, node_modules) are "
            "pruned from the walk; symlinks are not followed.\n\n"
            "The Config dropdown and Script picker are ignored in this "
            "mode."
        )
        self._chk_all_subdirs.toggled.connect(self._on_all_subdirs_toggled)
        card.add_body(self._chk_all_subdirs)

        # Status text is stored and revealed in a popup dialog.
        self._status_html = "No project loaded."
        info_row = QHBoxLayout()
        self._btn_yaml_info = ActionButton("YAML info…", category=Category.NEUTRAL,
                                           icon_name="info")
        self._btn_yaml_info.setToolTip("Show experiment type, chamber size, and "
                                       "per-YAML details for this project.")
        self._btn_yaml_info.clicked.connect(self._show_yaml_info)
        info_row.addWidget(self._btn_yaml_info)
        self._btn_qc = ActionButton("QC viewer…", category=Category.QC,
                                    icon_name="qc")
        self._btn_qc.clicked.connect(self._launch_qc_viewer)
        info_row.addWidget(self._btn_qc)
        info_row.addStretch(1)
        card.add_body(info_row)

        # External tool launchers
        tools_row = QHBoxLayout()
        self._btn_config = ActionButton("Edit config…", category=Category.TOOLS,
                                        icon_name="config")
        self._btn_config.clicked.connect(self._launch_config_editor)
        tools_row.addWidget(self._btn_config)
        self._btn_reload = ActionButton("Reload config", category=Category.TOOLS,
                                        icon_name="reload")
        self._btn_reload.setToolTip(
            "Drop the cached experiment and re-read project metadata so the "
            "next action picks up edits made to the YAML."
        )
        self._btn_reload.clicked.connect(self._action_reload_config)
        tools_row.addWidget(self._btn_reload)
        tools_row.addStretch(1)
        card.add_body(tools_row)

        self._cards["project"] = card
        self._cards_lay.addWidget(card)

    def _build_card_load(self) -> None:
        card = Card("Load", category=Category.LOAD,
                    subtitle="Load the experiment, set time window, run a script.",
                    icon_name="load")

        # Load options form — wrap when narrow so labels don't push width.
        opt_form = QFormLayout()
        opt_form.setHorizontalSpacing(10)
        opt_form.setVerticalSpacing(6)
        opt_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        def _shrinky(spin):
            spin.setMinimumWidth(70)
            spin.setMaximumWidth(140)
            return spin

        self._spin_start = _shrinky(QDoubleSpinBox())
        self._spin_start.setRange(0, 1_000_000)
        self._spin_start.setSpecialValueText("0")
        self._spin_start.setValue(0)
        opt_form.addRow("Start min:", self._spin_start)

        self._spin_end = _shrinky(QDoubleSpinBox())
        self._spin_end.setRange(0, 1_000_000)
        self._spin_end.setSpecialValueText("0")
        self._spin_end.setValue(0)
        opt_form.addRow("End min:", self._spin_end)

        self._chk_parallel = QCheckBox("Load DFMs in parallel")
        self._chk_parallel.setChecked(True)
        opt_form.addRow("", self._chk_parallel)

        self._spin_binsize = _shrinky(QSpinBox())
        self._spin_binsize.setRange(1, 10_000)
        self._spin_binsize.setValue(30)
        opt_form.addRow("Bin size (min):", self._spin_binsize)

        card.add_body(opt_form)

        # Action row: Load Experiment + Remove chambers.
        actions = QHBoxLayout()
        btn_load = ActionButton("Load experiment", category=Category.LOAD,
                                icon_name="load", primary=True)
        btn_load.clicked.connect(self._action_load_experiment)
        actions.addWidget(btn_load)
        self._load_buttons.append(btn_load)

        self._btn_remove_chambers = ActionButton(
            "Remove chambers", category=Category.LOAD, icon_name="remove"
        )
        self._btn_remove_chambers.setToolTip(
            "Remove chambers listed in remove_chambers.csv (group 'general') "
            "from the loaded experiment."
        )
        self._btn_remove_chambers.clicked.connect(self._action_remove_chambers)
        actions.addWidget(self._btn_remove_chambers)
        self._data_buttons.append(self._btn_remove_chambers)

        actions.addStretch(1)
        card.add_body(actions)

        self._cards["load"] = card
        self._cards_lay.addWidget(card)

    def _build_card_scripts(self) -> None:
        """Populate the Scripts card with a name dropdown + Run button.

        The dropdown + button are hidden until the active YAML (or — in
        batch mode — any YAML in the project) actually defines a
        ``scripts:`` section.  ``_refresh_script_dropdown`` manages the
        visibility; this just creates the widgets.
        """
        card = self._cards["scripts"]
        lay = card.body_layout()
        self._clear_layout(lay)

        self._script_empty_lbl = QLabel(
            "No scripts defined in the active YAML. "
            "Add a <code>scripts:</code> section to your config "
            "to run multi-step recipes."
        )
        self._script_empty_lbl.setWordWrap(True)
        self._script_empty_lbl.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self._script_empty_lbl)

        picker_row = QHBoxLayout()
        picker_row.addWidget(QLabel("Script:"))
        self._cmb_script = QComboBox()
        self._cmb_script.setMinimumWidth(80)
        self._cmb_script.setSizePolicy(QSizePolicy.Policy.Expanding,
                                       QSizePolicy.Policy.Fixed)
        self._cmb_script.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        picker_row.addWidget(self._cmb_script, 1)
        lay.addLayout(picker_row)

        self._btn_run_script = ActionButton("Run Script", category=Category.SCRIPTS,
                                            icon_name="script", primary=True)
        self._btn_run_script.clicked.connect(self._action_run_script)
        lay.addWidget(self._btn_run_script)
        self._load_buttons.append(self._btn_run_script)

        self._btn_run_all_scripts = ActionButton("Run All Scripts", category=Category.SCRIPTS,
                                                 icon_name="run")
        self._btn_run_all_scripts.setToolTip("Run every script in the active YAML in sequence.")
        self._btn_run_all_scripts.clicked.connect(self._action_run_all_scripts)
        lay.addWidget(self._btn_run_all_scripts)
        self._load_buttons.append(self._btn_run_all_scripts)

        # Hidden until the active YAML has scripts (or batch mode is on).
        self._cmb_script.setVisible(False)
        self._btn_run_script.setVisible(False)
        self._btn_run_all_scripts.setVisible(False)

    # ------------------------------------------------------------------
    # Dynamic group box rebuilding (Analyze / Plots)
    # ------------------------------------------------------------------

    @staticmethod
    def _clear_layout(lay) -> None:
        while lay.count():
            item = lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
            sub = item.layout()
            if sub is not None:
                AnalysisHubWindow._clear_layout(sub)

    def _refresh_config_list(self) -> None:
        """Repopulate the config-file dropdown from *.yaml in project dir."""
        p = self._project_dir()
        names = _list_yaml_configs(p)
        # Preserve current selection if still present; otherwise prefer
        # flic_config.yaml, then the first entry.
        prev = self._cmb_config.currentText()
        self._cmb_config.blockSignals(True)
        self._cmb_config.clear()
        self._cmb_config.addItems(names)
        if prev in names:
            self._cmb_config.setCurrentText(prev)
        elif "flic_config.yaml" in names:
            self._cmb_config.setCurrentText("flic_config.yaml")
        elif names:
            self._cmb_config.setCurrentIndex(0)
        self._cmb_config.blockSignals(False)
        self._active_config = self._cmb_config.currentText() or "flic_config.yaml"

    def _on_config_changed(self, name: str) -> None:
        if not name:
            return
        self._active_config = name
        self._invalidate_exp_cache()
        self._refresh_meta(refresh_config_list=False)

    def _on_all_yamls_toggled(self, checked: bool) -> None:
        # The script dropdown's contents depend on whether we're in batch mode.
        if checked:
            chk = getattr(self, "_chk_all_subdirs", None)
            if chk is not None and chk.isChecked():
                chk.setChecked(False)
        self._refresh_script_dropdown()

    def _on_all_subdirs_toggled(self, checked: bool) -> None:
        if checked and self._chk_all_yamls.isChecked():
            self._chk_all_yamls.setChecked(False)
        # Switching modes changes what counts as a valid Config and meta:
        # rebuild both so the dropdown lists subdir YAMLs (or parent YAMLs
        # again when leaving subdir mode) and the status bar reflects the
        # batch summary.
        self._refresh_meta(refresh_config_list=True)
        self._apply_subdir_mode_visibility()

    def _apply_subdir_mode_visibility(self) -> None:
        """Hide single-project UI when subdir-batch mode is on.

        Subdir-batch mode walks the tree under the project root and runs
        the 'batch' script in every batch target it finds, so the only
        enabled action is the tree-wide Run Script. Hide the Load /
        Analyze / Plots / Tools cards and the 'Run All Scripts' button.
        (The root may itself be a batch target — see ADR-0001 — but
        single-project UI is still hidden in this mode.)
        """
        chk = getattr(self, "_chk_all_subdirs", None)
        on = bool(chk is not None and chk.isChecked())
        for key in ("load", "analyze", "plots", "tools"):
            card = self._cards.get(key)
            if card is not None:
                card.setVisible(not on)
        btn = getattr(self, "_btn_run_all_scripts", None)
        if btn is not None and on:
            btn.setVisible(False)

    def _refresh_meta(self, refresh_config_list: bool = True) -> None:
        self._invalidate_exp_cache()
        p = self._project_dir()
        if refresh_config_list:
            self._refresh_config_list()
        cfg_name = self._active_config or "flic_config.yaml"
        subdirs_mode = (getattr(self, "_chk_all_subdirs", None) is not None
                        and self._chk_all_subdirs.isChecked())
        if subdirs_mode:
            targets, _near = self._collect_batch_targets(p)
            n_targets = len({t[0] for t in targets})
            self._status_html = (
                f"<b>{p}</b> — subdir batch: "
                f"<code>{n_targets}</code> batch target(s) "
                f"under tree"
            )
            self._rebuild_dynamic_groups(None, None)
            self._scripts = []
            self._refresh_script_dropdown()
            return
        meta = read_project_meta(p, cfg_name)
        if not meta.get("ok"):
            self._status_html = meta.get("error", "Invalid project.")
            self._rebuild_dynamic_groups(None, None)
            return
        et = meta.get("experiment_type")
        inf = meta.get("inferred_type")
        cs = meta.get("chamber_size")
        parts = [f"<b>{p}</b>", f"config: <code>{cfg_name}</code>"]
        if et:
            parts.append(f"experiment_type: <code>{et}</code>")
        elif inf:
            parts.append(f"experiment_type: <i>unspecified</i> (loads as <code>{inf}</code>)")
        else:
            parts.append("experiment_type: <i>unknown</i>")
        if cs is not None:
            parts.append(f"chamber_size: <code>{cs}</code>")
        data_ok = (p / "data").is_dir()
        parts.append("data/: " + ("found" if data_ok else "<span style='color:#a50'>missing</span>"))
        self._status_html = " — ".join(parts)
        self._rebuild_dynamic_groups(et or inf, cs)

        # Capture the active YAML's scripts (for single-yaml mode), then
        # populate the dropdown according to the current batch-mode state.
        cfg = _read_yaml(p / cfg_name)
        self._scripts = _parse_scripts(cfg)
        self._refresh_script_dropdown()

    def _refresh_script_dropdown(self) -> None:
        """Populate the Script dropdown based on the current batch-mode state.

        - Both batch modes off: list scripts defined in the active YAML.
        - YAML batch mode on: list the union of script names across every
          YAML in the project directory (deduped, preserving first-seen
          order).
        - Subdir batch mode on: hide the dropdown entirely; the Run Script
          button executes the script named ``batch`` in every batch target
          found under the project root (recursive walk, root included).
        """
        subdirs_mode = (getattr(self, "_chk_all_subdirs", None) is not None
                        and self._chk_all_subdirs.isChecked())

        if subdirs_mode:
            self._cmb_script.blockSignals(True)
            self._cmb_script.clear()
            self._cmb_script.blockSignals(False)
            self._cmb_script.setVisible(False)
            targets, _near = self._collect_batch_targets(self._project_dir())
            n = len({t[0] for t in targets})
            self._btn_run_script.setText(
                f"Run 'batch' ({n} target{'s' if n != 1 else ''})"
                if n else "Run 'batch' (no targets)"
            )
            self._btn_run_script.setVisible(True)
            self._btn_run_script.setEnabled(n > 0)
            self._btn_run_all_scripts.setVisible(False)
            empty_lbl = getattr(self, "_script_empty_lbl", None)
            if empty_lbl is not None:
                if n == 0:
                    empty_lbl.setText(
                        "No directory under this root contains a YAML with a "
                        "script named <code>batch</code>. Add one to enable "
                        "the run."
                    )
                    empty_lbl.setVisible(True)
                else:
                    empty_lbl.setVisible(False)
            return

        # Reset Run Script label any time we leave subdir mode.
        self._btn_run_script.setText("Run Script")
        self._btn_run_script.setEnabled(True)

        prev = self._cmb_script.currentText()
        self._cmb_script.blockSignals(True)
        self._cmb_script.clear()
        if self._chk_all_yamls.isChecked():
            configs = _list_yaml_configs(self._project_dir())
            names = self._union_script_names(configs)
        else:
            names = [s.get("name", "(unnamed)") for s in self._scripts]
        for n in names:
            self._cmb_script.addItem(n)
        if prev in names:
            self._cmb_script.setCurrentText(prev)
        self._cmb_script.blockSignals(False)
        has = bool(names)
        self._cmb_script.setVisible(has)
        self._btn_run_script.setVisible(has)
        self._btn_run_all_scripts.setVisible(len(names) > 1)
        # Empty-state hint lives next to the widgets in the Scripts card.
        empty_lbl = getattr(self, "_script_empty_lbl", None)
        if empty_lbl is not None:
            empty_lbl.setVisible(not has)
            if has:
                empty_lbl.setText("")
            else:
                if self._chk_all_yamls.isChecked():
                    empty_lbl.setText(
                        "No <code>scripts:</code> section found in any YAML "
                        "in this project."
                    )
                else:
                    empty_lbl.setText(
                        f"No <code>scripts:</code> section in "
                        f"<code>{self._active_config}</code>. Add one to the "
                        f"YAML or switch to another config above."
                    )

    def _rebuild_dynamic_groups(self, exp_type: str | None, chamber_size: int | None) -> None:
        self._data_buttons.clear()
        # Re-register persistent data-gated buttons that live in static cards
        # (only Remove chambers, built once in the Load card). Without this they
        # are orphaned by the clear above and never re-enabled on load.
        btn_remove = getattr(self, "_btn_remove_chambers", None)
        if btn_remove is not None and not _is_deleted(btn_remove):
            self._data_buttons.append(btn_remove)
        # Drop dead/dynamic-card buttons; static buttons (Load, Run Script,
        # Edit Config, QC viewer) stay in self._load_buttons.
        analyze_card = self._cards.get("analyze")
        tools_card = self._cards.get("tools")
        self._load_buttons = [
            b for b in self._load_buttons
            if not _is_deleted(b)
               and not _is_child_of(b, analyze_card)
               and not _is_child_of(b, tools_card)
        ]
        self._rebuild_card_analyze(exp_type, chamber_size)
        self._rebuild_card_plots(exp_type, chamber_size)
        self._rebuild_card_tools()
        self._update_data_buttons()
        self._install_wheel_guards()

    def _install_wheel_guards(self) -> None:
        """Install the wheel-redirect filter on every combo box / spin box in
        the scrollable panel.

        Idempotent — Qt ignores a filter that is already installed — so it is
        safe to call after each dynamic card rebuild to cover freshly created
        controls.
        """
        for w in self._scroll.findChildren((QComboBox, QAbstractSpinBox)):
            w.installEventFilter(self._wheel_guard)

    def _rebuild_card_analyze(self, exp_type: str | None, chamber_size: int | None) -> None:
        card = self._cards["analyze"]
        lay = card.body_layout()
        self._clear_layout(lay)

        def add(text: str, slot, *, icon_name: str = "basic") -> None:
            b = ActionButton(text, category=Category.ANALYZE, icon_name=icon_name)
            b.clicked.connect(slot)
            lay.addWidget(b)
            self._data_buttons.append(b)

        add("Run full basic analysis", self._action_basic_full, icon_name="basic")
        add("Write feeding summary CSV", self._action_write_feeding_csv, icon_name="csv")
        add("Write binned feeding summary CSV", self._action_binned_csv, icon_name="binned")
        if exp_type == "hedonic":
            add("Write weighted duration summary", self._action_weighted_duration, icon_name="weighted")

        card.add_section_label("— Advanced —")
        add("Tidy events CSV", self._action_tidy_export, icon_name="tidy")
        add("Bootstrap CIs (metric)…", self._action_bootstrap, icon_name="bootstrap")
        add("Compare treatments (ANOVA / LMM)…", self._action_compare, icon_name="compare")
        add("Light-phase summary CSV", self._action_light_phase, icon_name="lightphase")
        add("Parameter sensitivity sweep…", self._action_param_sensitivity, icon_name="sensitivity")
        is_two_well = (chamber_size == 2
                       or exp_type in {"two_well", "hedonic", "progressive_ratio"})
        if is_two_well:
            add("Bout transition matrix", self._action_transition_matrix, icon_name="transition")
        add("Write PDF report", self._action_pdf_report, icon_name="pdf")

    def _rebuild_card_tools(self) -> None:
        card = self._cards["tools"]
        lay = card.body_layout()
        self._clear_layout(lay)

        def add(text: str, slot, *, icon_name: str) -> None:
            b = ActionButton(text, category=Category.TOOLS, icon_name=icon_name)
            b.clicked.connect(slot)
            lay.addWidget(b)
            self._load_buttons.append(b)

        add("Lint config", self._action_lint_config, icon_name="lint")
        add("Compare two configs…", self._action_compare_configs, icon_name="compare_cfg")
        add("Clear disk cache", self._action_clear_cache, icon_name="clear")

    def _rebuild_card_plots(self, exp_type: str | None, chamber_size: int | None) -> None:
        card = self._cards["plots"]
        lay = card.body_layout()
        self._clear_layout(lay)

        # Feeding summary
        b = ActionButton("Feeding summary", category=Category.PLOTS, icon_name="feeding")
        b.clicked.connect(self._action_plot_feeding_summary)
        lay.addWidget(b)
        self._data_buttons.append(b)

        is_two_well = (chamber_size == 2
                       or exp_type in {"two_well", "hedonic", "progressive_ratio"})
        metrics = _TWO_WELL_BINNED if is_two_well else _SINGLE_WELL_BINNED

        def _shrinky_combo() -> QComboBox:
            cmb = QComboBox()
            cmb.setMinimumWidth(80)
            cmb.setSizePolicy(QSizePolicy.Policy.Expanding,
                              QSizePolicy.Policy.Fixed)
            cmb.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            return cmb

        def _shrinky(spin):
            spin.setMinimumWidth(70)
            spin.setMaximumWidth(140)
            return spin

        # Dot
        dot_row = QHBoxLayout()
        dot_row.addWidget(QLabel("Dot plot:"))
        self._cmb_dot_metric = _shrinky_combo()
        for label, metric, mode in metrics:
            self._cmb_dot_metric.addItem(label, userData=(metric, mode))
        dot_row.addWidget(self._cmb_dot_metric, 1)
        b_dot = ActionButton("Plot", category=Category.PLOTS, icon_name="dot")
        b_dot.clicked.connect(self._action_plot_dot)
        dot_row.addWidget(b_dot)
        lay.addLayout(dot_row)
        self._data_buttons.append(b_dot)

        # Well A vs B (two-well only) — sits directly under the dot row.
        if is_two_well:
            cmp_row = QHBoxLayout()
            cmp_row.addWidget(QLabel("Well A vs B:"))
            self._cmb_well_cmp = _shrinky_combo()
            for m in _WELL_CMP_METRICS:
                self._cmb_well_cmp.addItem(m)
            cmp_row.addWidget(self._cmb_well_cmp, 1)
            b_cmp = ActionButton("Plot", category=Category.PLOTS, icon_name="well")
            b_cmp.clicked.connect(self._action_plot_well_cmp)
            cmp_row.addWidget(b_cmp)
            lay.addLayout(cmp_row)
            self._data_buttons.append(b_cmp)

        # Binned
        binned_row = QHBoxLayout()
        binned_row.addWidget(QLabel("Binned:"))
        self._cmb_binned_metric = _shrinky_combo()
        for label, metric, mode in metrics:
            self._cmb_binned_metric.addItem(label, userData=(metric, mode))
        binned_row.addWidget(self._cmb_binned_metric, 1)
        b_binned = ActionButton("Plot", category=Category.PLOTS, icon_name="binned")
        b_binned.clicked.connect(self._action_plot_binned)
        binned_row.addWidget(b_binned)
        lay.addLayout(binned_row)
        self._data_buttons.append(b_binned)

        # Moving-window metric (sliding window) — optional, non-standard.
        # Mirrors the binned row: pick any feeding metric and plot its
        # treatment mean ± SEM over an overlapping sliding window.  Every row is
        # built from shrinkable widgets (short labels, expanding spinboxes,
        # single-button rows) so the controls can always collapse to the column
        # width instead of forcing the left panel wider.
        card.add_section_label("— Moving window —")

        def _flex_spin(value: float, tip: str) -> QDoubleSpinBox:
            sp = QDoubleSpinBox()
            sp.setRange(0.1, 1_000_000)
            sp.setDecimals(1)
            sp.setValue(value)
            sp.setToolTip(tip)
            sp.setMinimumWidth(44)
            sp.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            return sp

        mm_ctrl = QHBoxLayout()
        mm_ctrl.addWidget(QLabel("Win"))
        self._spin_mm_window = _flex_spin(60.0, "Sliding window width (minutes).")
        mm_ctrl.addWidget(self._spin_mm_window, 1)
        mm_ctrl.addWidget(QLabel("Step"))
        self._spin_mm_step = _flex_spin(30.0, "Window step / stride (minutes).")
        mm_ctrl.addWidget(self._spin_mm_step, 1)
        lay.addLayout(mm_ctrl)

        mm_row = QHBoxLayout()
        mm_row.addWidget(QLabel("Moving:"))
        self._cmb_mm_metric = _shrinky_combo()
        for label, metric, mode in metrics:
            self._cmb_mm_metric.addItem(label, userData=(metric, mode))
        mm_row.addWidget(self._cmb_mm_metric, 1)
        b_mm_tr = ActionButton("Plot", category=Category.PLOTS, icon_name="plot")
        b_mm_tr.setToolTip("Treatment mean ± SEM of the chosen metric over a sliding window.")
        b_mm_tr.clicked.connect(self._action_plot_moving_window)
        mm_row.addWidget(b_mm_tr)
        lay.addLayout(mm_row)
        self._data_buttons.append(b_mm_tr)

        # Closing rule so the moving-window block reads as its own group,
        # matching the dim section-label styling used for its header.
        card.add_section_label("─" * 24)

        # Hedonic
        if exp_type == "hedonic":
            b = ActionButton("Hedonic feeding plot", category=Category.PLOTS, icon_name="feeding")
            b.clicked.connect(self._action_plot_hedonic)
            lay.addWidget(b)
            self._data_buttons.append(b)

        # Progressive-ratio
        if exp_type == "progressive_ratio":
            pr_row = QHBoxLayout()
            pr_row.addWidget(QLabel("BP config:"))
            self._spin_pr_cfg = QSpinBox()
            self._spin_pr_cfg.setRange(1, 4)
            self._spin_pr_cfg.setValue(1)
            pr_row.addWidget(self._spin_pr_cfg)
            pr_row.addStretch()
            b_pr = ActionButton("Breaking-point plots", category=Category.PLOTS,
                                icon_name="plot")
            b_pr.clicked.connect(self._action_plot_pr)
            pr_row.addWidget(b_pr)
            lay.addLayout(pr_row)
            self._data_buttons.append(b_pr)

    # ------------------------------------------------------------------
    # Busy / worker helpers
    # ------------------------------------------------------------------

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        for w in self._load_buttons:
            w.setEnabled(not busy)
        self._update_data_buttons()

    def _update_data_buttons(self) -> None:
        enabled = self._exp_loaded and not self._busy
        for w in self._data_buttons:
            w.setEnabled(enabled)

    def _clear_worker_refs(self) -> None:
        self._thread = None
        self._worker = None

    def _start_worker(self, task: Callable[[], Any], force_single: bool = False) -> None:
        if self._busy:
            QMessageBox.information(self, "Busy", "An analysis task is already running.")
            return
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "Busy", "An analysis task is already running.")
            return
        p = self._project_dir()
        subdirs_mode = (getattr(self, "_chk_all_subdirs", None) is not None
                        and self._chk_all_subdirs.isChecked())
        # In subdir-batch mode the root needn't itself be a batch target —
        # any qualifying directory under the tree is enough. Skip the
        # root-config validation.
        if not subdirs_mode:
            configs = _list_yaml_configs(p)
            if not configs:
                QMessageBox.warning(self, "No config", f"No YAML config files found in {p}.")
                return
            if not (p / self._active_config).is_file():
                QMessageBox.warning(
                    self, "No config",
                    f"Selected config {self._active_config!r} not found in {p}.",
                )
                return

            if (not force_single
                    and self._chk_all_yamls.isChecked()
                    and len(configs) > 1):
                task = self._wrap_task_for_all_yamls(task, configs)

        self._thread = QThread()
        self._worker = AnalysisWorker(task)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._clear_worker_refs)
        self._worker.log.connect(self._append_log)
        self._worker.failed.connect(self._on_failed)
        self._worker.figure_ready.connect(self._show_figure)
        self._worker.finished.connect(lambda: self._set_busy(False))

        self._set_busy(True)
        self._thread.start()

    def _wrap_task_for_all_yamls(
        self, task: Callable[[], Any], configs: list[str]
    ) -> Callable[[], list[tuple[str, bytes]]]:
        """Return a task that runs *task* once for each YAML config.

        Per-iteration: update ``self._active_config``, clear the cached
        experiment, run the original task, collect any figures, and tag
        figure titles with the config stem.
        """
        def wrapped() -> list[tuple[str, bytes]]:
            all_figures: list[tuple[str, bytes]] = []
            original_config = self._active_config
            try:
                for cfg in configs:
                    stem = Path(cfg).stem
                    print(f"\n=== [{stem}] running action for {cfg} ===", flush=True)
                    self._active_config = cfg
                    # Force a fresh load per config since parameters, ranges,
                    # and excluded chambers differ between YAMLs.
                    self._cached_exp = None
                    self._cached_exp_key = None
                    try:
                        result = task()
                    except Exception as e:  # noqa: BLE001
                        print(f"[{stem}] FAILED: {type(e).__name__}: {e}", flush=True)
                        continue
                    if isinstance(result, tuple) and len(result) == 2 \
                            and isinstance(result[1], (bytes, bytearray)):
                        all_figures.append((f"[{stem}] {result[0]}", bytes(result[1])))
                    elif isinstance(result, list):
                        for item in result:
                            if (isinstance(item, tuple) and len(item) == 2
                                    and isinstance(item[1], (bytes, bytearray))):
                                all_figures.append((f"[{stem}] {item[0]}", bytes(item[1])))
            finally:
                self._active_config = original_config
                self._cached_exp = None
                self._cached_exp_key = None
            return all_figures

        return wrapped

    def _on_failed(self, msg: str) -> None:
        self._append_log(msg)
        QMessageBox.critical(self, "Analysis error", msg[:1200])

    def _append_log(self, text: str) -> None:
        self._log.append_line(text)

    def _show_figure(self, title: str, figure: Any) -> None:
        if figure is None:
            return
        try:
            size = self._plot_dock.add_figure(
                title, figure, interactive=self._chk_interactive.isChecked()
            )
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"[plot] Failed to embed {title!r}: {exc}")
            return
        if isinstance(size, QSize) and size.isValid():
            self._fit_window_to_plot(size.width(), size.height())

    def _fit_window_to_plot(self, content_w: int, content_h: int) -> None:
        """Grow the window so the plot pane can show a *content_w*×*content_h*
        figure without scrolling, capped at the monitor's available area.

        The window only ever grows (so a later, smaller plot never shrinks it),
        the extra width is given to the plot pane rather than the cards column,
        and the window is nudged back on-screen if growth pushed it past an
        edge.
        """
        if content_w <= 0 or content_h <= 0:
            return
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        split = self._main_split
        # Chrome = the window area outside the splitter (menus, margins, log).
        chrome_w = max(0, self.width() - split.width())
        chrome_h = max(0, self.height() - split.height())
        sizes = split.sizes()
        left_w = sizes[0] if len(sizes) == 2 else self._scroll.width()
        handle = split.handleWidth()
        # Padding for the dock's tab bar and the scroll-area viewport.
        PAD_W, PAD_H = 30, 72
        want_w = chrome_w + left_w + handle + content_w + PAD_W
        want_h = chrome_h + content_h + PAD_H
        new_w = min(max(self.width(), want_w), avail.width())
        new_h = min(max(self.height(), want_h), avail.height())
        if new_w != self.width() or new_h != self.height():
            self.resize(new_w, new_h)
        # Hand the extra width to the plot pane; keep the cards column as-is.
        inner_w = max(0, new_w - chrome_w - handle)
        min_plot = self._plot_dock.minimumWidth()
        min_left = self._scroll.minimumWidth()
        left_keep = max(min_left, min(left_w, inner_w - min_plot))
        plot_w = max(min_plot, inner_w - left_keep)
        split.setSizes([left_keep, plot_w])
        # Keep the (possibly grown) window fully on screen.
        fg = self.frameGeometry()
        nx, ny = self.x(), self.y()
        if fg.right() > avail.right():
            nx = max(avail.left(), avail.right() - fg.width())
        if fg.bottom() > avail.bottom():
            ny = max(avail.top(), avail.bottom() - fg.height())
        if (nx, ny) != (self.x(), self.y()):
            self.move(nx, ny)

    def _toggle_theme(self) -> None:
        from .ui import theme as _theme

        new_mode = "light" if _theme.resolved_mode() == "dark" else "dark"
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, mode=new_mode)
        ui_settings.set_value("theme", new_mode)
        self._btn_theme.setIcon(
            icon("theme_dark" if _theme.resolved_mode() == "light" else "theme_light")
        )

    def _on_sidebar_selected(self, key: str) -> None:
        if key == "help":
            self._open_help()
            return
        self._scroll_to_card(key)

    def _scroll_to_card(self, key: str) -> None:
        card = self._cards.get(key)
        if card is None:
            return
        self._scroll.ensureWidgetVisible(card, 0, 8)

    def _show_yaml_info(self) -> None:
        """Open a pop-out dialog summarising every YAML in the project dir."""
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QTextBrowser

        p = self._project_dir()
        yamls = _list_yaml_configs(p)

        rows: list[str] = []
        for name in yamls:
            meta = read_project_meta(p, name)
            active_marker = (" <span style='color:#2563eb'>(active)</span>"
                             if name == self._active_config else "")
            if not meta.get("ok"):
                rows.append(
                    f"<li><b>{name}</b>{active_marker} "
                    f"<span style='color:#c33'>[error: {meta.get('error')}]</span></li>"
                )
                continue
            et = meta.get("experiment_type") or \
                 (f"<i>(inferred: {meta.get('inferred_type')})</i>"
                  if meta.get("inferred_type") else "<i>unknown</i>")
            cs = meta.get("chamber_size")

            details: list[str] = []
            details.append(f"experiment_type: <code>{et}</code>")
            if cs is not None:
                details.append(f"chamber_size: <code>{cs}</code>")

            try:
                cfg = _read_yaml(p / name)
            except Exception:  # noqa: BLE001
                rows.append(
                    f"<li><b>{name}</b>{active_marker}<br>"
                    f"{' &middot; '.join(details)}</li>"
                )
                continue

            # DFM count + excluded-chamber roll-up.
            dfms_raw = cfg.get("dfms") or cfg.get("DFMs") or []
            if isinstance(dfms_raw, dict):
                dfm_items: list[tuple[Any, dict]] = [
                    (k, v) for k, v in dfms_raw.items() if isinstance(v, dict)
                ]
            elif isinstance(dfms_raw, list):
                dfm_items = [
                    (node.get("id", node.get("ID", "?")), node)
                    for node in dfms_raw if isinstance(node, dict)
                ]
            else:
                dfm_items = []
            details.append(f"DFMs: {len(dfm_items)}")

            excl_parts: list[str] = []
            excl_total = 0
            for dfm_id, node in dfm_items:
                excluded = node.get("excluded_chambers") or []
                if excluded:
                    excl_total += len(excluded)
                    ids = ", ".join(str(c) for c in excluded)
                    excl_parts.append(f"DFM {dfm_id}: [{ids}]")
            if excl_total:
                details.append(
                    f"excluded chambers: <code>{excl_total}</code> "
                    f"({'; '.join(excl_parts)})"
                )
            else:
                details.append("excluded chambers: <i>none</i>")

            # Experimental design factors.
            global_cfg = cfg.get("global") or {}
            factors = (global_cfg.get("experimental_design_factors") or {}
                       if isinstance(global_cfg, dict) else {})
            if isinstance(factors, dict) and factors:
                factor_parts = []
                for fname, levels in factors.items():
                    if isinstance(levels, list):
                        factor_parts.append(f"{fname}=[{', '.join(str(x) for x in levels)}]")
                    else:
                        factor_parts.append(str(fname))
                details.append("factors: " + "; ".join(factor_parts))
            else:
                details.append("factors: <i>none</i>")

            # Scripts: count and names.
            scripts = _parse_scripts(cfg)
            if scripts:
                names = ", ".join(f"<code>{s.get('name', '(unnamed)')}</code>"
                                  for s in scripts)
                details.append(f"scripts ({len(scripts)}): {names}")
            else:
                details.append("scripts: <i>none</i>")

            rows.append(
                f"<li><b>{name}</b>{active_marker}<br>"
                f"{' &middot; '.join(details)}</li>"
            )

        list_html = "<ul>" + "".join(rows) + "</ul>" if rows else "<i>No YAML files found.</i>"
        body = (
            f"<p><b>Project:</b> <code>{p}</code></p>"
            f"<h3 style='margin-top:12px;margin-bottom:4px;'>YAML configs</h3>"
            f"{list_html}"
        )

        dlg = QDialog(self)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dlg.setWindowTitle("YAML info")
        dlg.resize(680, 500)
        lay = QVBoxLayout(dlg)
        view = QTextBrowser()
        view.setOpenExternalLinks(False)
        view.setHtml(body)
        lay.addWidget(view, 1)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(dlg.reject)
        btns.accepted.connect(dlg.accept)
        btns.button(QDialogButtonBox.StandardButton.Close).clicked.connect(dlg.accept)
        lay.addWidget(btns)
        dlg.exec()

    # ------------------------------------------------------------------
    # Project / experiment helpers
    # ------------------------------------------------------------------

    def _range_minutes(self) -> tuple[float, float]:
        return float(self._spin_start.value()), float(self._spin_end.value())

    def _project_dir(self) -> Path:
        return Path(self._path_edit.text().strip()).expanduser().resolve()

    def _browse_project(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select project directory", str(self._project_dir()))
        if d:
            self._path_edit.setText(d)
            self._refresh_meta()

    def _exp_cache_key(self, exclusion_group: str | None = None) -> tuple:
        return (
            str(self._project_dir()),
            self._active_config,
            self._range_minutes(),
            self._chk_parallel.isChecked(),
            exclusion_group,
        )

    def _load_exp(self, exclusion_group: str | None = None):
        key = self._exp_cache_key(exclusion_group)
        if self._cached_exp is not None and self._cached_exp_key == key:
            print("Using cached experiment (same project / options).", flush=True)
            return self._cached_exp

        from pyflic import load_experiment_yaml

        exp = load_experiment_yaml(
            self._project_dir(),
            config_name=self._active_config,
            range_minutes=self._range_minutes(),
            parallel=self._chk_parallel.isChecked(),
            exclusion_group=exclusion_group,
        )
        self._cached_exp = exp
        self._cached_exp_key = key
        return exp

    def _invalidate_exp_cache(self) -> None:
        self._cached_exp = None
        self._cached_exp_key = None
        self._exp_loaded = False
        self._update_data_buttons()

    def _try_write_summary(self, exp) -> None:
        """Write summary.txt to the analysis dir; silent on failure."""
        try:
            p = exp.write_summary()
            print(f"Summary: {p}", flush=True)
        except Exception as exc:
            print(f"  (summary.txt not written: {exc})", flush=True)

    def _on_range_updated(self, start: float, end: float) -> None:
        """Update the start/end spinboxes to reflect the range actually loaded by a script.

        Blocks the spinboxes' valueChanged signals so the exp cache is not invalidated.
        """
        self._spin_start.blockSignals(True)
        self._spin_end.blockSignals(True)
        self._spin_start.setValue(start)
        self._spin_end.setValue(end)
        self._spin_start.blockSignals(False)
        self._spin_end.blockSignals(False)

    def _action_reload_config(self) -> None:
        """Drop the cached experiment and re-read project metadata.

        Called after the user edits the YAML externally (e.g. via the config
        editor subprocess) so that the next analysis action sees the updated
        config without changing project / range / parallel selections.
        """
        p = self._project_dir()
        if not p.is_dir():
            QMessageBox.warning(self, "Invalid path", "Choose a valid project directory.")
            return
        self._refresh_meta()
        print(f"Reloaded config: {self._active_config}", flush=True)

    def _launch_config_editor(self) -> None:
        p = self._project_dir()
        if not p.is_dir():
            QMessageBox.warning(self, "Invalid path", "Choose a valid project directory.")
            return
        cmd = _resolve_cli("pyflic-config", "pyflic.base.config_editor")
        # Prefer opening the currently-selected config file directly; fall back
        # to the project directory so the editor's auto-load can find a default.
        target = p / self._active_config
        arg = str(target if target.is_file() else p)
        try:
            subprocess.Popen([*cmd, arg], cwd=str(p))  # noqa: S603
        except OSError as e:
            QMessageBox.critical(self, "Could not start", str(e))

    def _qc_dir_for_range(self) -> Path:
        """
        Resolve the QC output directory for the current range under the active
        config's output subfolder.

        Mirrors ``Experiment._range_suffix``: ``(0, 0)`` → ``qc``; finite
        ``(a, b)`` → ``qc_{int(a)}_{int(b)}``; ``b`` of ``inf`` or ``0`` is
        written as ``end`` by the producer, so we resolve that by matching the
        existing ``qc_{int(a)}_*`` directory on disk.
        """
        p = self._project_dir()
        stem = Path(self._active_config or "flic_config.yaml").stem
        base = p / f"{stem}_results"
        a, b = self._range_minutes()
        if a == 0.0 and b == 0.0:
            return base / "qc"
        a_lbl = str(int(a))
        if b == float("inf") or b == 0.0:
            exact = base / f"qc_{a_lbl}_end"
            if exact.is_dir():
                return exact
            matches = sorted(base.glob(f"qc_{a_lbl}_*"))
            if matches:
                return matches[0]
            return base / "qc"
        ranged = base / f"qc_{a_lbl}_{int(b)}"
        if ranged.is_dir():
            return ranged
        return base / "qc"

    def _launch_qc_viewer(self) -> None:
        p = self._project_dir()
        if not p.is_dir():
            QMessageBox.warning(self, "Invalid path", "Choose a valid project directory.")
            return
        cmd = _resolve_cli("pyflic-qc", "pyflic.base.qc_viewer")
        cmd = [*cmd, str(p), str(self._qc_dir_for_range())]
        try:
            subprocess.Popen(cmd, cwd=str(p))  # noqa: S603
        except OSError as e:
            QMessageBox.critical(self, "Could not start", str(e))

    # ------------------------------------------------------------------
    # Analyze actions  (return None — no figure display)
    # ------------------------------------------------------------------

    def _action_load_experiment(self) -> None:
        def task() -> None:
            exp = self._load_exp()
            print(exp.summary_text(include_qc=False), flush=True)

        self._start_worker(task)
        if self._worker is not None:
            self._worker.finished.connect(self._on_load_finished)

    def _on_load_finished(self) -> None:
        if self._cached_exp is not None:
            self._exp_loaded = True
            self._update_data_buttons()

    def _action_remove_chambers(self) -> None:
        """Apply the 'general' exclusion group from remove_chambers.csv in-place."""
        project_dir = self._project_dir()

        def task() -> None:
            from .exclusions import read_exclusions
            exp = self._load_exp()
            all_excl = read_exclusions(project_dir)
            group_excl = all_excl.get("general", {})
            if not group_excl:
                print(
                    "No chambers listed for group 'general' in remove_chambers.csv.",
                    flush=True,
                )
                return
            remove_set = {
                (dfm_id, ch)
                for dfm_id, chambers in group_excl.items()
                for ch in chambers
            }
            exp._remove_chambers_from_design(remove_set)
            total = len(remove_set)
            print("Chambers removed (group 'general'):", flush=True)
            for dfm_id, chambers in sorted(group_excl.items()):
                print(f"  DFM {dfm_id}: chamber(s) {sorted(chambers)}", flush=True)
            print(f"  Total: {total} chamber(s) removed.", flush=True)
            exp.excluded_chambers = {k: sorted(v) for k, v in group_excl.items()}
            exp.exclusion_group = "general"
            self._try_write_summary(exp)

        self._start_worker(task)

    def _action_basic_full(self) -> None:
        rm = self._range_minutes()

        def task() -> None:
            exp = self._load_exp()
            paths = exp.execute_basic_analysis(range_minutes=rm, skip_qc=False)
            for k, v in paths.items():
                if v is not None:
                    print(f"{k}: {v}", flush=True)

        self._start_worker(task)

    def _action_write_feeding_csv(self) -> None:
        rm = self._range_minutes()

        def task() -> None:
            exp = self._load_exp()
            self._try_write_summary(exp)
            p = exp.write_feeding_summary(range_minutes=rm)
            print(f"Wrote: {p}", flush=True)

        self._start_worker(task)

    def _action_binned_csv(self) -> None:
        rm = self._range_minutes()
        bs = float(self._spin_binsize.value())

        def task() -> None:
            exp = self._load_exp()
            self._try_write_summary(exp)
            p = exp.binned_feeding_summary(binsize_min=bs, range_minutes=rm, save=True)
            print(f"Binned rows: {len(p)}", flush=True)

        self._start_worker(task)

    def _action_weighted_duration(self) -> None:
        rm = self._range_minutes()

        def task() -> None:
            from pyflic import HedonicFeedingExperiment

            exp = self._load_exp()
            if not isinstance(exp, HedonicFeedingExperiment):
                raise TypeError(
                    "Set experiment_type: hedonic in flic_config.yaml. "
                    f"Got {type(exp).__name__}."
                )
            self._try_write_summary(exp)
            p = exp.weighted_duration_summary(save=True, range_minutes=rm)
            print(f"Wrote: {p}", flush=True)

        self._start_worker(task)

    # ------------------------------------------------------------------
    # Plot actions  (return (title, png_bytes) to trigger figure display)
    # ------------------------------------------------------------------

    def _action_plot_feeding_summary(self) -> None:
        rm = self._range_minutes()

        def task() -> tuple[str, bytes]:
            exp = self._load_exp()
            fig = exp.plot_feeding_summary(range_minutes=rm)
            out = exp.analysis_dir / "feeding_summary.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            if hasattr(fig, "save"):
                fig.save(str(out), dpi=300)
            else:
                fig.savefig(str(out), dpi=300, bbox_inches="tight")
            print(f"Wrote: {out}", flush=True)
            return "Feeding Summary", fig

        self._start_worker(task)

    def _action_plot_moving_window(self) -> None:
        rm = self._range_minutes()
        win = float(self._spin_mm_window.value())
        step = float(self._spin_mm_step.value())
        metric, mode = self._cmb_mm_metric.currentData()

        def task() -> tuple[str, Any]:
            exp = self._load_exp()
            fig = exp.plot_moving_window_metric_by_treatment(
                metric=metric,
                two_well_mode=mode,
                window_min=win,
                step_min=step,
                range_minutes=rm,
            )
            safe = metric.replace("/", "_")
            out = exp.analysis_dir / f"moving_window_{safe}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            if hasattr(fig, "save"):
                fig.save(str(out), dpi=300)
            else:
                fig.savefig(str(out), dpi=300, bbox_inches="tight")
            print(f"Wrote: {out}", flush=True)
            return f"Moving window: {metric}", fig

        self._start_worker(task)

    def _action_plot_binned(self) -> None:
        rm = self._range_minutes()
        bs = float(self._spin_binsize.value())
        metric, mode = self._cmb_binned_metric.currentData()

        def task() -> tuple[str, bytes]:
            exp = self._load_exp()
            fig = exp.plot_binned_metric_by_treatment(
                metric=metric,
                two_well_mode=mode,
                binsize_min=bs,
                range_minutes=rm,
            )
            safe = metric.replace("/", "_")
            out = exp.analysis_dir / f"binned_{safe}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            if hasattr(fig, "save"):
                fig.save(str(out), dpi=300)
            else:
                fig.savefig(str(out), dpi=300, bbox_inches="tight")
            print(f"Wrote: {out}", flush=True)
            return f"Binned: {metric}", fig

        self._start_worker(task)

    def _action_plot_dot(self) -> None:
        rm = self._range_minutes()
        metric, mode = self._cmb_dot_metric.currentData()

        def task() -> tuple[str, bytes]:
            exp = self._load_exp()
            fig = exp.plot_dot_metric_by_treatment(
                metric=metric,
                two_well_mode=mode,
                range_minutes=rm,
            )
            safe = metric.replace("/", "_")
            out = exp.analysis_dir / f"dot_{safe}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            if hasattr(fig, "save"):
                fig.save(str(out), dpi=300)
            else:
                fig.savefig(str(out), dpi=300, bbox_inches="tight")
            print(f"Wrote: {out}", flush=True)
            return f"Dot: {metric}", fig

        self._start_worker(task)

    def _action_plot_well_cmp(self) -> None:
        rm = self._range_minutes()
        metric = self._cmb_well_cmp.currentText()

        def task() -> tuple[str, bytes]:
            from pyflic import TwoWellExperiment

            exp = self._load_exp()
            if not isinstance(exp, TwoWellExperiment):
                raise TypeError(
                    "Well A vs B comparison requires a two-well experiment. "
                    f"Got {type(exp).__name__}."
                )
            fig = exp.facet_plot_well_durations(metric=metric, range_minutes=rm)
            out = exp.analysis_dir / f"well_comparison_{metric}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.save(str(out), dpi=300)
            print(f"Wrote: {out}", flush=True)
            return f"Well A vs B: {metric}", fig

        self._start_worker(task)

    def _action_plot_hedonic(self) -> None:
        rm = self._range_minutes()

        def task() -> tuple[str, bytes]:
            from pyflic import HedonicFeedingExperiment

            exp = self._load_exp()
            if not isinstance(exp, HedonicFeedingExperiment):
                raise TypeError(
                    "Set experiment_type: hedonic in flic_config.yaml. "
                    f"Got {type(exp).__name__}."
                )
            fig = exp.hedonic_feeding_plot(save=True, range_minutes=rm)
            print("Wrote hedonic feeding plot.", flush=True)
            return "Hedonic Feeding Plot", fig

        self._start_worker(task)

    def _action_plot_pr(self) -> None:
        cfg = int(self._spin_pr_cfg.value())

        def task() -> list[tuple[str, bytes]]:
            from pyflic import ProgressiveRatioExperiment

            exp = self._load_exp()
            if not isinstance(exp, ProgressiveRatioExperiment):
                raise TypeError(
                    "Set experiment_type: progressive_ratio in flic_config.yaml. "
                    f"Got {type(exp).__name__}."
                )
            ad = exp.analysis_dir
            if ad is None:
                raise ValueError("No project_dir on experiment.")
            ad.mkdir(parents=True, exist_ok=True)
            results: list[tuple[str, bytes]] = []
            for dfm_id, dfm in sorted(exp.dfms.items()):
                fig = exp.plot_breaking_point_dfm_gg(dfm, cfg)
                out = ad / f"breaking_point_dfm{dfm_id}_config{cfg}.png"
                fig.save(str(out), dpi=300)
                print(f"Wrote: {out}", flush=True)
                results.append((
                    f"Breaking Point — DFM {dfm_id} (config {cfg})",
                    fig,
                ))
            return results

        self._start_worker(task)

    # ------------------------------------------------------------------
    # Advanced analytics actions
    # ------------------------------------------------------------------

    def _action_tidy_export(self) -> None:
        rm = self._range_minutes()

        def task() -> list[tuple[str, bytes]]:
            from .analytics import tidy_events
            exp = self._load_exp()
            df = tidy_events(exp, kind="feeding")
            out = exp.analysis_dir / "tidy_feeding_events.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            print(f"Wrote: {out}  ({len(df)} bouts)", flush=True)
            return []

        self._start_worker(task)

    def _action_bootstrap(self) -> None:
        from PyQt6.QtWidgets import QInputDialog
        rm = self._range_minutes()
        metric, ok = QInputDialog.getText(
            self, "Bootstrap CIs",
            "Metric (e.g. PI, MedDuration, Events):", text="PI",
        )
        if not ok or not metric.strip():
            return
        n_boot, ok = QInputDialog.getInt(
            self, "Bootstrap CIs", "Bootstrap iterations:", 2000, 100, 100000, 100,
        )
        if not ok:
            return

        def task() -> list[tuple[str, bytes]]:
            from .analytics import bootstrap_metric
            exp = self._load_exp()
            res = bootstrap_metric(
                exp, metric=metric.strip(),
                two_well_mode="total",
                n_boot=int(n_boot),
                range_minutes=rm,
            )
            out = exp.analysis_dir / f"bootstrap_{metric.strip()}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            res.summary.to_csv(out, index=False)
            print(f"Wrote: {out}\n{res.summary.to_string(index=False)}", flush=True)
            return []

        self._start_worker(task)

    def _action_compare(self) -> None:
        from PyQt6.QtWidgets import QInputDialog
        rm = self._range_minutes()
        metric, ok = QInputDialog.getText(
            self, "Compare treatments",
            "Metric (e.g. MedDuration, PI, Events):", text="MedDuration",
        )
        if not ok or not metric.strip():
            return
        model, ok = QInputDialog.getItem(
            self, "Compare treatments", "Model:", ["aov", "lmm"], 0, False,
        )
        if not ok:
            return

        def task() -> list[tuple[str, bytes]]:
            from .analytics import compare_treatments
            exp = self._load_exp()
            res = compare_treatments(
                exp, metric=metric.strip(),
                two_well_mode="A", model=model,
                range_minutes=rm,
            )
            out = exp.analysis_dir / f"compare_{metric.strip()}_{model}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            res.table.to_csv(out, index=False)
            if res.posthoc is not None:
                res.posthoc.to_csv(out.with_name(out.stem + "_posthoc.csv"), index=False)
            print(f"Wrote: {out}\n{res.table.to_string(index=False)}", flush=True)
            return []

        self._start_worker(task)

    def _action_light_phase(self) -> None:
        def task() -> list[tuple[str, bytes]]:
            from .analytics import light_phase_summary
            exp = self._load_exp()
            df = light_phase_summary(exp)
            out = exp.analysis_dir / "light_phase_summary.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            print(f"Wrote: {out}  ({len(df)} rows)", flush=True)
            return []

        self._start_worker(task)

    def _action_param_sensitivity(self) -> None:
        from PyQt6.QtWidgets import QInputDialog
        rm = self._range_minutes()
        param, ok = QInputDialog.getItem(
            self, "Parameter sensitivity",
            "Parameter to sweep:",
            ["feeding_event_link_gap", "feeding_threshold", "feeding_minimum",
             "tasting_minimum", "tasting_maximum", "feeding_minevents",
             "tasting_minevents", "baseline_window_minutes", "samples_per_second"],
            0, False,
        )
        if not ok:
            return
        text, ok = QInputDialog.getText(
            self, "Parameter sensitivity",
            f"Comma-separated values for {param}:", text="2,5,10,15,20",
        )
        if not ok or not text.strip():
            return
        try:
            values = [float(s.strip()) for s in text.split(",") if s.strip()]
        except ValueError:
            QMessageBox.warning(self, "Bad input", "Values must be numeric")
            return

        def task() -> list[tuple[str, bytes]]:
            from .analytics import parameter_sensitivity
            exp = self._load_exp()
            res = parameter_sensitivity(
                exp, parameter=param, values=values,
                range_minutes=rm,
            )
            out = exp.analysis_dir / f"param_sensitivity_{param}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            res.grid.to_csv(out, index=False)
            print(f"Wrote: {out}\n{res.grid.to_string(index=False)}", flush=True)
            return []

        self._start_worker(task)

    def _action_transition_matrix(self) -> None:
        def task() -> list[tuple[str, bytes]]:
            from .analytics import bout_transition_matrix
            exp = self._load_exp()
            df = bout_transition_matrix(exp)
            out = exp.analysis_dir / "bout_transition_matrix.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            print(f"Wrote: {out}  ({len(df)} rows)", flush=True)
            return []

        self._start_worker(task)

    def _action_pdf_report(self) -> None:
        rm = self._range_minutes()

        def task() -> list[tuple[str, bytes]]:
            from .pdf_report import write_experiment_report
            exp = self._load_exp()
            p = write_experiment_report(exp, range_minutes=rm)
            print(f"Wrote PDF report: {p}", flush=True)
            return []

        self._start_worker(task)

    # ------------------------------------------------------------------
    # Tools (no loaded experiment required)
    # ------------------------------------------------------------------

    def _action_lint_config(self) -> None:
        from .yaml_lint import lint_flic_config
        cfg = self._project_dir() / self._active_config
        if not cfg.is_file():
            QMessageBox.warning(self, "No config", f"No {self._active_config} in {self._project_dir()}")
            return
        issues = lint_flic_config(cfg)
        n_err = sum(1 for i in issues if i.severity == "error")
        n_warn = sum(1 for i in issues if i.severity == "warning")
        if not issues:
            QMessageBox.information(self, "Lint", f"{cfg.name}: clean (0 issues).")
            return
        body = "\n".join(i.format(cfg) for i in issues[:50])
        body += f"\n\n{n_err} error(s), {n_warn} warning(s)"
        if n_err:
            QMessageBox.critical(self, "Lint errors", body)
        else:
            QMessageBox.warning(self, "Lint warnings", body)

    def _action_compare_configs(self) -> None:
        from PyQt6.QtWidgets import QInputDialog
        d = QFileDialog.getExistingDirectory(
            self, "Select 2nd project to compare against", str(self._project_dir()),
        )
        if not d:
            return
        metric, ok = QInputDialog.getText(
            self, "Compare configs", "Metric (e.g. Licks, Events, MedDuration):",
            text="Events",
        )
        if not ok or not metric.strip():
            return
        rm = self._range_minutes()

        stem = Path(self._active_config or "flic_config.yaml").stem

        def task() -> list[tuple[str, bytes]]:
            from .analytics import compare_configs
            df = compare_configs(
                self._project_dir(), Path(d),
                metrics=(metric.strip(),), two_well_mode="total",
                range_minutes=rm,
            )
            out = self._project_dir() / stem / "analysis" / f"compare_configs_{metric.strip()}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            print(f"Wrote: {out}\n{df.to_string(index=False)}", flush=True)
            return []

        self._start_worker(task)

    def _action_clear_cache(self) -> None:
        from . import cache as _cache
        n = _cache.clear(self._project_dir())
        QMessageBox.information(self, "Cache cleared", f"Removed {n} cached file(s).")
        self._invalidate_exp_cache()

    # ------------------------------------------------------------------
    # Script execution
    # ------------------------------------------------------------------

    def _action_run_script(self) -> None:
        if (getattr(self, "_chk_all_subdirs", None) is not None
                and self._chk_all_subdirs.isChecked()):
            self._run_batch_targets()
            return
        name = self._cmb_script.currentText().strip()
        if not name:
            return
        if self._chk_all_yamls.isChecked():
            self._run_named_script_batch(name)
            return
        # Single-yaml mode: find the script by name in the active yaml.
        for s in self._scripts:
            if s.get("name") == name:
                self._run_script(s)
                return
        # Fallback by index in case names duplicate within one yaml.
        idx = self._cmb_script.currentIndex()
        if 0 <= idx < len(self._scripts):
            self._run_script(self._scripts[idx])

    def _action_run_all_scripts(self) -> None:
        """Run every script in the active YAML in sequence as a single worker task."""
        scripts = self._scripts
        if not scripts:
            return
        ui_binsize = float(self._spin_binsize.value())
        ui_parallel = self._chk_parallel.isChecked()
        project_dir = self._project_dir()
        # Build each script's task closure on the main thread so UI values are captured.
        named_tasks = [
            (str(s.get("name", f"Script {i + 1}")),
             self._build_script_task(s, project_dir, ui_binsize, ui_parallel))
            for i, s in enumerate(scripts)
        ]
        total = len(named_tasks)

        def combined_task() -> list[tuple[str, bytes]]:
            all_figures: list[tuple[str, bytes]] = []
            for i, (name, task) in enumerate(named_tasks):
                print(f"\n{'=' * 50}", flush=True)
                print(f"Running script {i + 1}/{total}: {name}", flush=True)
                print(f"{'=' * 50}", flush=True)
                figs = task()
                if figs:
                    all_figures.extend(figs)
            print(f"\nAll {total} script(s) complete.", flush=True)
            return all_figures

        self._start_worker(combined_task)
        if self._worker is not None:
            self._worker.finished.connect(self._on_load_finished)

    def _collect_scripts_by_yaml(self, configs: list[str]) -> dict[str, dict[str, dict]]:
        """Return ``{yaml_filename: {script_name: script_dict}}`` for *configs*."""
        p = self._project_dir()
        out: dict[str, dict[str, dict]] = {}
        for cfg in configs:
            try:
                data = _read_yaml(p / cfg)
            except Exception:  # noqa: BLE001
                continue
            named: dict[str, dict] = {}
            for s in _parse_scripts(data):
                n = s.get("name")
                if n and n not in named:
                    named[n] = s
            out[cfg] = named
        return out

    def _union_script_names(self, configs: list[str]) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        per_yaml = self._collect_scripts_by_yaml(configs)
        for cfg in configs:
            for n in per_yaml.get(cfg, {}):
                if n not in seen:
                    seen.add(n)
                    names.append(n)
        return names

    def _run_named_script_batch(self, name: str) -> None:
        """Run, in one worker, the script named *name* in every yaml that has it."""
        project_dir = self._project_dir()
        configs = _list_yaml_configs(project_dir)
        per_yaml = self._collect_scripts_by_yaml(configs)
        matches: list[tuple[str, dict]] = [
            (cfg, per_yaml[cfg][name])
            for cfg in configs
            if name in per_yaml.get(cfg, {})
        ]
        if not matches:
            QMessageBox.information(
                self, "No matches",
                f"No YAML in {project_dir} defines a script named {name!r}.",
            )
            return
        ui_binsize = float(self._spin_binsize.value())
        ui_parallel = self._chk_parallel.isChecked()

        def task() -> list[tuple[str, bytes]]:
            all_figures: list[tuple[str, bytes]] = []
            original_config = self._active_config
            try:
                for cfg, script in matches:
                    stem = Path(cfg).stem
                    print(
                        f"\n=== [{stem}] running script {name!r} from {cfg} "
                        f"({len(script.get('steps') or [])} step(s)) ===",
                        flush=True,
                    )
                    self._active_config = cfg
                    self._cached_exp = None
                    self._cached_exp_key = None
                    inner = self._build_script_task(
                        script, project_dir, ui_binsize, ui_parallel,
                    )
                    try:
                        figs = inner() or []
                    except Exception as e:  # noqa: BLE001
                        print(f"[{stem}] FAILED: {type(e).__name__}: {e}", flush=True)
                        continue
                    for t, b in figs:
                        all_figures.append((f"[{stem}] {t}", b))
            finally:
                self._active_config = original_config
                self._cached_exp = None
                self._cached_exp_key = None
            return all_figures

        self._start_worker(task, force_single=True)
        if self._worker is not None:
            self._worker.finished.connect(self._on_load_finished)

    def _collect_batch_targets(
        self, root: Path,
    ) -> tuple[list[tuple[Path, str, dict]], list[Path]]:
        """Walk *root* (root included) and classify every directory.

        Returns ``(targets, near_misses)`` where:

        * ``targets`` — list of ``(target_dir, yaml_name, script_dict)``
          for every YAML in every directory under *root* that defines a
          script named ``batch``. Multiple YAMLs in one directory each
          contribute one entry.
        * ``near_misses`` — directories that contain at least one
          parseable YAML but none defines ``batch``. Used for diagnostic
          skip logging; directories with no YAMLs at all are silent.

        Skips directory names starting with ``.`` and the entries in
        ``_BATCH_SKIP_DIR_NAMES``; does not follow symlinks.
        """
        if not root.is_dir():
            return [], []
        targets: list[tuple[Path, str, dict]] = []
        near_misses: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(
            root, topdown=True, followlinks=False,
        ):
            # Prune in-place: dot-dirs and known output/tooling dirs.
            dirnames[:] = sorted(
                d for d in dirnames
                if not d.startswith(".")
                and d not in _BATCH_SKIP_DIR_NAMES
            )
            cur = Path(dirpath)
            yamls = sorted(
                f for f in filenames
                if Path(f).suffix.lower() in (".yaml", ".yml")
            )
            if not yamls:
                continue
            had_yaml_parse = False
            added_for_dir = False
            for fname in yamls:
                try:
                    data = _read_yaml(cur / fname)
                except Exception:  # noqa: BLE001
                    continue
                had_yaml_parse = True
                for s in _parse_scripts(data):
                    if str(s.get("name", "")).strip().lower() == "batch":
                        targets.append((cur, fname, s))
                        added_for_dir = True
                        break  # one batch script per YAML
            if had_yaml_parse and not added_for_dir:
                near_misses.append(cur)
        return targets, near_misses

    def _run_batch_targets(self) -> None:
        """Walk the tree under the chosen root and run the ``batch`` script
        in every batch target. The root is included as a candidate.

        Each target directory is treated as its own project, so script
        output lands inside the target. Outputs are always overwritten;
        no skip-if-results-exist guard.
        """
        root = self._project_dir()
        targets, near_misses = self._collect_batch_targets(root)
        if not targets:
            QMessageBox.information(
                self, "No 'batch' scripts found",
                f"No YAML under {root} defines a script named 'batch'.",
            )
            return
        ui_binsize = float(self._spin_binsize.value())
        ui_parallel = self._chk_parallel.isChecked()

        def _rel(p: Path) -> str:
            try:
                rel = p.relative_to(root).as_posix()
            except ValueError:
                return p.as_posix()
            return rel if rel != "." else "."

        def task() -> list[tuple[str, bytes]]:
            all_figures: list[tuple[str, bytes]] = []
            if near_misses:
                for d in near_misses:
                    print(
                        f"Skipped: {_rel(d)} (YAML present but no "
                        f"'batch' script defined)",
                        flush=True,
                    )
            original_config = self._active_config
            try:
                for target_dir, yaml_name, script in targets:
                    rel = _rel(target_dir)
                    label = f"./{yaml_name}" if rel == "." else f"{rel}/{yaml_name}"
                    print(
                        f"\n=== [{label}] running 'batch' "
                        f"({len(script.get('steps') or [])} step(s)) ===",
                        flush=True,
                    )
                    # _build_script_task reads self._active_config inside the
                    # closure; swap it to the per-target YAML for the run.
                    self._active_config = yaml_name
                    self._cached_exp = None
                    self._cached_exp_key = None
                    inner = self._build_script_task(
                        script, target_dir, ui_binsize, ui_parallel,
                    )
                    try:
                        figs = inner() or []
                    except Exception as e:  # noqa: BLE001
                        print(f"[{label}] FAILED: {type(e).__name__}: {e}", flush=True)
                        continue
                    for t, b in figs:
                        all_figures.append((f"[{label}] {t}", b))
            finally:
                self._active_config = original_config
                self._cached_exp = None
                self._cached_exp_key = None
            return all_figures

        self._start_worker(task, force_single=True)
        if self._worker is not None:
            self._worker.finished.connect(self._on_load_finished)

    def _run_script(self, script: dict) -> None:
        ui_binsize = float(self._spin_binsize.value())
        ui_parallel = self._chk_parallel.isChecked()
        project_dir = self._project_dir()
        task = self._build_script_task(script, project_dir, ui_binsize, ui_parallel)
        self._start_worker(task)
        if self._worker is not None:
            self._worker.finished.connect(self._on_load_finished)

    def _build_script_task(
        self,
        script: dict,
        project_dir: Path,
        ui_binsize: float,
        ui_parallel: bool,
    ) -> Callable[[], list[tuple[str, bytes]]]:
        """Build a worker task closure for *script*.

        The closure reads ``self._active_config`` at execution time so the
        per-yaml batch wrapper can swap configs between iterations.
        """
        steps = script.get("steps") or []
        script_name = str(script.get("name") or "general")
        ui_start, ui_end = self._range_minutes()
        script_start = float(script.get("start", ui_start))
        script_end = float(script.get("end", ui_end))

        def task() -> list[tuple[str, bytes]]:
            exp = None
            figures: list[tuple[str, bytes]] = []

            def _get_rm(step: dict) -> tuple[float, float]:
                return (
                    float(step.get("start", script_start)),
                    float(step.get("end", script_end)),
                )

            def _make_cache_key(rm, parallel):
                return (str(project_dir), self._active_config, rm, parallel, script_name)

            def _ensure_exp(rm):
                nonlocal exp
                if exp is None:
                    cache_key = _make_cache_key(rm, ui_parallel)
                    if (self._cached_exp is not None
                            and self._cached_exp_key == cache_key):
                        print("Using cached experiment (same project / options).", flush=True)
                        exp = self._cached_exp
                        return exp
                    from pyflic import load_experiment_yaml
                    exp = load_experiment_yaml(
                        project_dir, config_name=self._active_config,
                        range_minutes=rm, parallel=ui_parallel,
                        exclusion_group=None,
                    )
                    self._cached_exp = exp
                    self._cached_exp_key = cache_key
                return exp

            for i, step in enumerate(steps):
                action = str(step.get("action", "")).strip().lower()
                rm = _get_rm(step)
                print(f"\n[Script step {i + 1}/{len(steps)}] {action}", flush=True)

                if action == "load":
                    parallel = bool(step.get("parallel", ui_parallel))
                    cache_key = _make_cache_key(rm, parallel)
                    if (self._cached_exp is not None
                            and self._cached_exp_key == cache_key):
                        exp = self._cached_exp
                        print("Using cached experiment (same project / options).", flush=True)
                    else:
                        from pyflic import load_experiment_yaml
                        exp = load_experiment_yaml(
                            project_dir, config_name=self._active_config,
                            range_minutes=rm, parallel=parallel,
                            exclusion_group=None,
                        )
                        self._cached_exp = exp
                        self._cached_exp_key = cache_key
                        print("Loaded.", flush=True)
                    self._range_updated.emit(rm[0], rm[1])

                elif action == "remove_chambers":
                    from .exclusions import read_exclusions
                    e = _ensure_exp(rm)
                    group = str(step.get("group", script_name))
                    all_excl = read_exclusions(project_dir)
                    group_excl = all_excl.get(group, {})
                    if not group_excl:
                        print(
                            f"  No chambers listed for group '{group}' in remove_chambers.csv.",
                            flush=True,
                        )
                    else:
                        remove_set = {
                            (dfm_id, ch)
                            for dfm_id, chambers in group_excl.items()
                            for ch in chambers
                        }
                        e._remove_chambers_from_design(remove_set)
                        total = len(remove_set)
                        for dfm_id, chambers in sorted(group_excl.items()):
                            print(
                                f"  DFM {dfm_id}: removed chamber(s) {sorted(chambers)}",
                                flush=True,
                            )
                        print(
                            f"  Total: {total} chamber(s) removed (group '{group}').",
                            flush=True,
                        )
                        e.excluded_chambers = {
                            k: sorted(v) for k, v in group_excl.items()
                        }
                        e.exclusion_group = group
                    self._try_write_summary(e)

                elif action == "basic_analysis":
                    e = _ensure_exp(rm)
                    e.execute_basic_analysis(range_minutes=rm, skip_qc=True)

                elif action == "feeding_csv":
                    e = _ensure_exp(rm)
                    p = e.write_feeding_summary(range_minutes=rm)
                    print(f"Wrote: {p}", flush=True)

                elif action == "binned_csv":
                    bs = float(step.get("binsize", ui_binsize))
                    e = _ensure_exp(rm)
                    df = e.binned_feeding_summary(binsize_min=bs, range_minutes=rm, save=True)
                    print(f"Binned rows: {len(df)}", flush=True)

                elif action == "weighted_duration":
                    from pyflic import HedonicFeedingExperiment
                    e = _ensure_exp(rm)
                    if not isinstance(e, HedonicFeedingExperiment):
                        print("[Skip] weighted_duration requires hedonic experiment.", flush=True)
                        continue
                    p = e.weighted_duration_summary(save=True, range_minutes=rm)
                    print(f"Wrote: {p}", flush=True)

                elif action == "plot_feeding_summary":
                    e = _ensure_exp(rm)
                    fig = e.plot_feeding_summary(range_minutes=rm)
                    out = e.analysis_dir / "feeding_summary.png"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    if hasattr(fig, "save"):
                        fig.save(str(out), dpi=300)
                    else:
                        fig.savefig(str(out), dpi=300, bbox_inches="tight")
                    print(f"Wrote: {out}", flush=True)
                    figures.append(("Feeding Summary", fig))

                elif action in ("plot_binned", "plot_dot"):
                    metric = str(step.get("metric", "Licks"))
                    mode = str(step.get("mode", _METRIC_DEFAULT_MODE.get(metric, "total")))
                    bs = float(step.get("binsize", ui_binsize))
                    e = _ensure_exp(rm)
                    if action == "plot_binned":
                        fig = e.plot_binned_metric_by_treatment(
                            metric=metric, two_well_mode=mode, binsize_min=bs, range_minutes=rm
                        )
                        safe = metric.replace("/", "_")
                        out = e.analysis_dir / f"binned_{safe}.png"
                        out.parent.mkdir(parents=True, exist_ok=True)
                        if hasattr(fig, "save"):
                            fig.save(str(out), dpi=300)
                        else:
                            fig.savefig(str(out), dpi=300, bbox_inches="tight")
                        print(f"Wrote: {out}", flush=True)
                        figures.append((f"Binned: {metric}", fig))
                    else:
                        fig = e.plot_dot_metric_by_treatment(
                            metric=metric, two_well_mode=mode, range_minutes=rm
                        )
                        safe = metric.replace("/", "_")
                        out = e.analysis_dir / f"dot_{safe}.png"
                        out.parent.mkdir(parents=True, exist_ok=True)
                        if hasattr(fig, "save"):
                            fig.save(str(out), dpi=300)
                        else:
                            fig.savefig(str(out), dpi=300, bbox_inches="tight")
                        print(f"Wrote: {out}", flush=True)
                        figures.append((f"Dot: {metric}", fig))

                elif action == "plot_moving_window":
                    metric = str(step.get("metric", "MedDuration"))
                    mode = str(step.get("mode", _METRIC_DEFAULT_MODE.get(metric, "mean_ab")))
                    win = float(step.get("window", 60.0))
                    mm_step = float(step.get("step", 30.0))
                    e = _ensure_exp(rm)
                    fig = e.plot_moving_window_metric_by_treatment(
                        metric=metric, two_well_mode=mode,
                        window_min=win, step_min=mm_step, range_minutes=rm,
                    )
                    safe = metric.replace("/", "_")
                    out = e.analysis_dir / f"moving_window_{safe}.png"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    if hasattr(fig, "save"):
                        fig.save(str(out), dpi=300)
                    else:
                        fig.savefig(str(out), dpi=300, bbox_inches="tight")
                    print(f"Wrote: {out}", flush=True)
                    figures.append((f"Moving window: {metric}", fig))

                elif action in ("plot_moving_median_chambers", "plot_moving_median_treatment"):
                    win = float(step.get("window", 60.0))
                    mm_step = float(step.get("step", 30.0))
                    mode = str(step.get("mode", "mean_ab"))
                    e = _ensure_exp(rm)
                    if action == "plot_moving_median_chambers":
                        fig = e.plot_moving_median_duration_by_chamber(
                            window_min=win, step_min=mm_step,
                            two_well_mode=mode, range_minutes=rm,
                        )
                        out = e.analysis_dir / "moving_median_duration_chambers.png"
                        title = "Moving MedDuration: chambers"
                    else:
                        fig = e.plot_moving_median_duration_by_treatment(
                            window_min=win, step_min=mm_step,
                            two_well_mode=mode, range_minutes=rm,
                        )
                        out = e.analysis_dir / "moving_median_duration_treatments.png"
                        title = "Moving MedDuration: treatments"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    if hasattr(fig, "save"):
                        fig.save(str(out), dpi=300)
                    else:
                        fig.savefig(str(out), dpi=300, bbox_inches="tight")
                    print(f"Wrote: {out}", flush=True)
                    figures.append((title, fig))

                elif action == "plot_well_comparison":
                    from pyflic import TwoWellExperiment
                    metric = str(step.get("metric", "MedDuration"))
                    e = _ensure_exp(rm)
                    if not isinstance(e, TwoWellExperiment):
                        print("[Skip] plot_well_comparison requires two-well experiment.", flush=True)
                        continue
                    fig = e.facet_plot_well_durations(metric=metric, range_minutes=rm)
                    out = e.analysis_dir / f"well_comparison_{metric}.png"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    fig.save(str(out), dpi=300)
                    print(f"Wrote: {out}", flush=True)
                    figures.append((f"Well A vs B: {metric}", fig))

                elif action == "plot_hedonic":
                    from pyflic import HedonicFeedingExperiment
                    e = _ensure_exp(rm)
                    if not isinstance(e, HedonicFeedingExperiment):
                        print("[Skip] plot_hedonic requires hedonic experiment.", flush=True)
                        continue
                    fig = e.hedonic_feeding_plot(save=True, range_minutes=rm)
                    print("Wrote hedonic feeding plot.", flush=True)
                    figures.append(("Hedonic Feeding Plot", fig))

                elif action == "plot_breaking_point":
                    from pyflic import ProgressiveRatioExperiment
                    cfg_idx = int(step.get("config", 1))
                    e = _ensure_exp(rm)
                    if not isinstance(e, ProgressiveRatioExperiment):
                        print("[Skip] plot_breaking_point requires progressive_ratio experiment.", flush=True)
                        continue
                    ad = e.analysis_dir
                    ad.mkdir(parents=True, exist_ok=True)
                    for dfm_id, dfm in sorted(e.dfms.items()):
                        fig = e.plot_breaking_point_dfm_gg(dfm, cfg_idx)
                        out = ad / f"breaking_point_dfm{dfm_id}_config{cfg_idx}.png"
                        fig.save(str(out), dpi=300)
                        print(f"Wrote: {out}", flush=True)
                        figures.append((
                            f"Breaking Point — DFM {dfm_id} (config {cfg_idx})",
                            fig,
                        ))

                elif action == "tidy_export":
                    e = _ensure_exp(rm)
                    from .analytics import tidy_events
                    kind = str(step.get("kind", "feeding")).strip().lower()
                    df = tidy_events(e, kind=kind)
                    out = e.analysis_dir / f"tidy_{kind}_events.csv"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(out, index=False)
                    print(f"Wrote: {out}  ({len(df)} bouts)", flush=True)

                elif action == "bootstrap":
                    e = _ensure_exp(rm)
                    from .analytics import bootstrap_metric
                    metric = str(step.get("metric", "PI"))
                    mode = str(step.get("mode", "total"))
                    n_boot = int(step.get("n_boot", 2000))
                    ci_lvl = float(step.get("ci", 0.95))
                    seed = step.get("seed", 0)
                    res = bootstrap_metric(
                        e, metric=metric, two_well_mode=mode,
                        n_boot=n_boot, ci=ci_lvl,
                        range_minutes=rm,
                        seed=int(seed) if seed is not None else None,
                    )
                    out = e.analysis_dir / f"bootstrap_{metric}.csv"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    res.summary.to_csv(out, index=False)
                    print(f"Wrote: {out}\n{res.summary.to_string(index=False)}", flush=True)

                elif action == "compare":
                    e = _ensure_exp(rm)
                    from .analytics import compare_treatments
                    metric = str(step.get("metric", "MedDuration"))
                    mode = str(step.get("mode", "A"))
                    model = str(step.get("model", "aov"))
                    factors = step.get("factors")
                    res = compare_treatments(
                        e, metric=metric, two_well_mode=mode, model=model,
                        factors=tuple(factors) if isinstance(factors, list) else None,
                        range_minutes=rm,
                    )
                    out = e.analysis_dir / f"compare_{metric}_{model}.csv"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    res.table.to_csv(out, index=False)
                    if res.posthoc is not None:
                        res.posthoc.to_csv(out.with_name(out.stem + "_posthoc.csv"), index=False)
                    print(f"Wrote: {out}\n{res.table.to_string(index=False)}", flush=True)

                elif action == "light_phase_summary":
                    e = _ensure_exp(rm)
                    from .analytics import light_phase_summary
                    df = light_phase_summary(e)
                    out = e.analysis_dir / "light_phase_summary.csv"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(out, index=False)
                    print(f"Wrote: {out}  ({len(df)} rows)", flush=True)

                elif action == "param_sensitivity":
                    e = _ensure_exp(rm)
                    from .analytics import parameter_sensitivity
                    param = str(step.get("parameter", "feeding_event_link_gap"))
                    values = step.get("values") or []
                    res = parameter_sensitivity(
                        e, parameter=param,
                        values=[float(v) for v in values],
                        range_minutes=rm,
                    )
                    out = e.analysis_dir / f"param_sensitivity_{param}.csv"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    res.grid.to_csv(out, index=False)
                    print(f"Wrote: {out}\n{res.grid.to_string(index=False)}", flush=True)

                elif action == "transition_matrix":
                    e = _ensure_exp(rm)
                    from .analytics import bout_transition_matrix
                    from pyflic import TwoWellExperiment
                    if not isinstance(e, TwoWellExperiment):
                        print("[Skip] transition_matrix requires two-well experiment.", flush=True)
                        continue
                    df = bout_transition_matrix(e)
                    out = e.analysis_dir / "bout_transition_matrix.csv"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(out, index=False)
                    print(f"Wrote: {out}  ({len(df)} rows)", flush=True)

                elif action == "pdf_report":
                    e = _ensure_exp(rm)
                    from .pdf_report import write_experiment_report
                    metrics = step.get("metrics") or ("Licks", "Events", "MedDuration")
                    bs = float(step.get("binsize", ui_binsize))
                    p = write_experiment_report(
                        e,
                        metrics=tuple(metrics),
                        binsize_min=bs,
                        range_minutes=rm,
                    )
                    print(f"Wrote: {p}", flush=True)

                elif action == "write_summary":
                    e = _ensure_exp(rm)
                    self._try_write_summary(e)

                else:
                    print(f"[Skip] Unknown action: {action!r}", flush=True)

            return figures

        return task


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("pyflic Analysis Hub")
    apply_theme(app, mode=ui_settings.get("theme", "auto"))
    project_dir = sys.argv[1] if len(sys.argv) > 1 else None
    win = AnalysisHubWindow(project_dir=project_dir)
    win.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
