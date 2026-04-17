"""
FLIC Analysis Hub
=================
PyQt6 launcher for common pyflic analysis workflows. Opens the config editor and
QC viewer in separate processes. Runs analysis in a background thread.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import yaml
from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

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


def read_project_meta(project_dir: Path) -> dict[str, Any]:
    """Lightweight parse of ``flic_config.yaml`` for status display (no DFM load)."""
    cfg_path = project_dir / "flic_config.yaml"
    if not cfg_path.is_file():
        return {"ok": False, "error": f"No flic_config.yaml in {project_dir}"}
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


def _figure_to_bytes(fig_or_gg, *, dpi: int = 100) -> bytes:
    """Render a matplotlib Figure or plotnine ggplot to PNG bytes."""
    import io as _io

    import matplotlib.pyplot as _plt

    buf = _io.BytesIO()
    if hasattr(fig_or_gg, "savefig"):          # matplotlib Figure
        fig_or_gg.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        _plt.close(fig_or_gg)
    else:                                       # plotnine ggplot
        mpl_fig = fig_or_gg.draw()
        mpl_fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        _plt.close(mpl_fig)
    buf.seek(0)
    return buf.read()



# ---------------------------------------------------------------------------
# Plot display window
# ---------------------------------------------------------------------------

class _PlotWindow(QDialog):
    """Scrollable window for displaying a rendered plot image."""

    def __init__(self, title: str, png_bytes: bytes, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(title)
        self.resize(980, 760)

        lay = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label = QLabel()
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pix = QPixmap()
        pix.loadFromData(png_bytes)
        label.setPixmap(pix)
        scroll.setWidget(label)
        scroll.setWidgetResizable(False)
        lay.addWidget(scroll, stretch=1)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        lay.addWidget(close_btn)


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


class AnalysisWorker(QObject):
    """Runs a callable in a worker thread; streams stdout/stderr to the log.

    Task callables may return one of:
    - ``None``                                      — no figure to display
    - ``(title: str, png_bytes: bytes)``            — single figure
    - ``[(title, png_bytes), ...]``                 — multiple figures
    """

    log = pyqtSignal(str)
    failed = pyqtSignal(str)
    finished = pyqtSignal()
    figure_ready = pyqtSignal(str, bytes)

    def __init__(self, task: Callable[[], Any]) -> None:
        super().__init__()
        self._task = task

    def run(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
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
            # Emit any figures returned by the task.
            figures: list[tuple[str, bytes]] = []
            if isinstance(result, list):
                figures = [(str(t), bytes(b)) for t, b in result
                           if isinstance(b, (bytes, bytearray))]
            elif (isinstance(result, tuple) and len(result) == 2
                  and isinstance(result[1], (bytes, bytearray))):
                figures = [(str(result[0]), bytes(result[1]))]
            for title, png_bytes in figures:
                self.figure_ready.emit(title, png_bytes)
            self.finished.emit()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class AnalysisHubWindow(QMainWindow):
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
        self._plot_windows: list[_PlotWindow] = []
        self._scripts: list[dict] = []

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)

        # ── Project directory row ────────────────────────────────────────
        proj_row = QHBoxLayout()
        proj_row.addWidget(QLabel("Project directory:"))
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Select a folder containing flic_config.yaml and data/")
        proj_row.addWidget(self._path_edit, stretch=1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_project)
        proj_row.addWidget(browse)
        outer.addLayout(proj_row)

        # ── Status label ─────────────────────────────────────────────────
        self._status = QLabel("No project loaded.")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

        # ── Load options ─────────────────────────────────────────────────
        opt_box = QGroupBox("Load options")
        opt_form = QFormLayout(opt_box)

        self._spin_start = QDoubleSpinBox()
        self._spin_start.setRange(0, 1_000_000)
        self._spin_start.setSpecialValueText("0")
        self._spin_start.setValue(0)
        opt_form.addRow("Start minute (0 = from beginning):", self._spin_start)

        self._spin_end = QDoubleSpinBox()
        self._spin_end.setRange(0, 1_000_000)
        self._spin_end.setSpecialValueText("0")
        self._spin_end.setValue(0)
        opt_form.addRow("End minute (0 = through end of recording):", self._spin_end)

        self._chk_parallel = QCheckBox("Load DFMs in parallel")
        self._chk_parallel.setChecked(True)
        opt_form.addRow(self._chk_parallel)

        self._spin_binsize = QSpinBox()
        self._spin_binsize.setRange(1, 10_000)
        self._spin_binsize.setValue(30)
        opt_form.addRow("Bin size (minutes):", self._spin_binsize)

        outer.addWidget(opt_box)

        # ── Three action group boxes ──────────────────────────────────────
        self._grp_load = QGroupBox("Load")
        QVBoxLayout(self._grp_load)
        self._build_grp_load()

        self._grp_analyze = QGroupBox("Analyze")
        QVBoxLayout(self._grp_analyze)

        self._grp_plots = QGroupBox("Plots")
        QVBoxLayout(self._grp_plots)

        mid = QHBoxLayout()
        mid.addWidget(self._grp_load, stretch=1)
        mid.addWidget(self._grp_analyze, stretch=2)
        mid.addWidget(self._grp_plots, stretch=2)
        outer.addLayout(mid)

        # ── Output log ───────────────────────────────────────────────────
        outer.addWidget(QLabel("Output"))
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(5000)
        outer.addWidget(self._log, stretch=1)

        self._path_edit.editingFinished.connect(self._refresh_meta)
        self._spin_start.valueChanged.connect(self._invalidate_exp_cache)
        self._spin_end.valueChanged.connect(self._invalidate_exp_cache)

        start_dir = self._initial_dir or Path.cwd()
        self._path_edit.setText(str(start_dir))
        self._refresh_meta()

    # ------------------------------------------------------------------
    # Load group box (static)
    # ------------------------------------------------------------------

    def _build_grp_load(self) -> None:
        lay = self._grp_load.layout()

        btn_load = QPushButton("Load experiment")
        btn_load.clicked.connect(self._action_load_experiment)
        lay.addWidget(btn_load)
        self._load_buttons.append(btn_load)

        # Script selector row — shown only when flic_config.yaml has a scripts: section
        script_row = QHBoxLayout()
        self._cmb_script = QComboBox()
        self._btn_run_script = QPushButton("Run Script")
        self._btn_run_script.clicked.connect(self._action_run_script)
        script_row.addWidget(self._cmb_script, stretch=1)
        script_row.addWidget(self._btn_run_script)
        lay.addLayout(script_row)
        self._load_buttons.append(self._btn_run_script)
        self._cmb_script.setVisible(False)
        self._btn_run_script.setVisible(False)

        self._btn_config = QPushButton("Edit config (pyflic-config)…")
        self._btn_config.clicked.connect(self._launch_config_editor)
        lay.addWidget(self._btn_config)

        self._btn_qc = QPushButton("QC viewer (pyflic-qc)…")
        self._btn_qc.clicked.connect(self._launch_qc_viewer)
        lay.addWidget(self._btn_qc)

        lay.addStretch()

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

    def _refresh_meta(self) -> None:
        self._invalidate_exp_cache()
        p = self._project_dir()
        meta = read_project_meta(p)
        if not meta.get("ok"):
            self._status.setText(meta.get("error", "Invalid project."))
            self._rebuild_dynamic_groups(None, None)
            return
        et = meta.get("experiment_type")
        inf = meta.get("inferred_type")
        cs = meta.get("chamber_size")
        parts = [f"<b>{p}</b>"]
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
        self._status.setText(" — ".join(parts))
        self._rebuild_dynamic_groups(et or inf, cs)

        # Populate script dropdown from YAML
        cfg = _read_yaml(p / "flic_config.yaml")
        self._scripts = _parse_scripts(cfg)
        self._cmb_script.clear()
        for s in self._scripts:
            self._cmb_script.addItem(s.get("name", "(unnamed)"))
        has_scripts = bool(self._scripts)
        self._cmb_script.setVisible(has_scripts)
        self._btn_run_script.setVisible(has_scripts)

    def _rebuild_dynamic_groups(self, exp_type: str | None, chamber_size: int | None) -> None:
        self._data_buttons.clear()
        # Purge load_buttons that live inside the dynamic groups (they'll be
        # re-created below).  Keep only buttons whose C++ side is still alive
        # AND whose parent is NOT the Analyze or Plots group box.
        self._load_buttons = [
            b for b in self._load_buttons
            if not _is_deleted(b) and not _is_child_of(b, self._grp_analyze)
        ]
        self._rebuild_grp_analyze(exp_type, chamber_size)
        self._rebuild_grp_plots(exp_type, chamber_size)
        self._update_data_buttons()

    def _rebuild_grp_analyze(self, exp_type: str | None, chamber_size: int | None) -> None:
        lay = self._grp_analyze.layout()
        self._clear_layout(lay)

        b = QPushButton("Run full basic analysis")
        b.clicked.connect(self._action_basic_full)
        lay.addWidget(b)
        self._data_buttons.append(b)

        b = QPushButton("Write feeding summary CSV")
        b.clicked.connect(self._action_write_feeding_csv)
        lay.addWidget(b)
        self._data_buttons.append(b)

        b = QPushButton("Write binned feeding summary CSV")
        b.clicked.connect(self._action_binned_csv)
        lay.addWidget(b)
        self._data_buttons.append(b)

        if exp_type == "hedonic":
            b = QPushButton("Write weighted duration summary")
            b.clicked.connect(self._action_weighted_duration)
            lay.addWidget(b)
            self._data_buttons.append(b)

        # ── Advanced analytics ────────────────────────────────────────────
        sep = QLabel("— Advanced —")
        sep.setStyleSheet("color: #888; font-size: 10px;")
        lay.addWidget(sep)

        b = QPushButton("Tidy events CSV")
        b.clicked.connect(self._action_tidy_export)
        lay.addWidget(b)
        self._data_buttons.append(b)

        b = QPushButton("Bootstrap CIs (metric)…")
        b.clicked.connect(self._action_bootstrap)
        lay.addWidget(b)
        self._data_buttons.append(b)

        b = QPushButton("Compare treatments (ANOVA / LMM)…")
        b.clicked.connect(self._action_compare)
        lay.addWidget(b)
        self._data_buttons.append(b)

        b = QPushButton("Light-phase summary CSV")
        b.clicked.connect(self._action_light_phase)
        lay.addWidget(b)
        self._data_buttons.append(b)

        b = QPushButton("Parameter sensitivity sweep…")
        b.clicked.connect(self._action_param_sensitivity)
        lay.addWidget(b)
        self._data_buttons.append(b)

        is_two_well = (chamber_size == 2
                       or exp_type in {"two_well", "hedonic", "progressive_ratio"})
        if is_two_well:
            b = QPushButton("Bout transition matrix")
            b.clicked.connect(self._action_transition_matrix)
            lay.addWidget(b)
            self._data_buttons.append(b)

        b = QPushButton("Write PDF report")
        b.clicked.connect(self._action_pdf_report)
        lay.addWidget(b)
        self._data_buttons.append(b)

        # ── Tools (config-level, no loaded experiment needed) ────────────
        sep2 = QLabel("— Tools —")
        sep2.setStyleSheet("color: #888; font-size: 10px;")
        lay.addWidget(sep2)

        b_lint = QPushButton("Lint flic_config.yaml")
        b_lint.clicked.connect(self._action_lint_config)
        lay.addWidget(b_lint)
        self._load_buttons.append(b_lint)

        b_diff = QPushButton("Compare two configs…")
        b_diff.clicked.connect(self._action_compare_configs)
        lay.addWidget(b_diff)
        self._load_buttons.append(b_diff)

        b_cache = QPushButton("Clear disk cache")
        b_cache.clicked.connect(self._action_clear_cache)
        lay.addWidget(b_cache)
        self._load_buttons.append(b_cache)

        lay.addStretch()

    def _rebuild_grp_plots(self, exp_type: str | None, chamber_size: int | None) -> None:
        lay = self._grp_plots.layout()
        self._clear_layout(lay)

        # ── Feeding summary ──────────────────────────────────────────────
        b = QPushButton("Feeding summary")
        b.clicked.connect(self._action_plot_feeding_summary)
        lay.addWidget(b)
        self._data_buttons.append(b)

        # ── Binned time-course ───────────────────────────────────────────
        is_two_well = (chamber_size == 2
                       or exp_type in {"two_well", "hedonic", "progressive_ratio"})
        metrics = _TWO_WELL_BINNED if is_two_well else _SINGLE_WELL_BINNED

        binned_row = QHBoxLayout()
        binned_row.addWidget(QLabel("Binned:"))
        self._cmb_binned_metric = QComboBox()
        for label, metric, mode in metrics:
            self._cmb_binned_metric.addItem(label, userData=(metric, mode))
        binned_row.addWidget(self._cmb_binned_metric, stretch=1)
        b_binned = QPushButton("Plot")
        b_binned.clicked.connect(self._action_plot_binned)
        binned_row.addWidget(b_binned)
        lay.addLayout(binned_row)
        self._data_buttons.append(b_binned)

        # ── Dot plot (same metrics, non-binned) ──────────────────────────
        dot_row = QHBoxLayout()
        dot_row.addWidget(QLabel("Dot plot:"))
        self._cmb_dot_metric = QComboBox()
        for label, metric, mode in metrics:
            self._cmb_dot_metric.addItem(label, userData=(metric, mode))
        dot_row.addWidget(self._cmb_dot_metric, stretch=1)
        b_dot = QPushButton("Plot")
        b_dot.clicked.connect(self._action_plot_dot)
        dot_row.addWidget(b_dot)
        lay.addLayout(dot_row)
        self._data_buttons.append(b_dot)

        # ── Well A vs B comparison (two-well only) ───────────────────────
        if is_two_well:
            cmp_row = QHBoxLayout()
            cmp_row.addWidget(QLabel("Well A vs B:"))
            self._cmb_well_cmp = QComboBox()
            for m in _WELL_CMP_METRICS:
                self._cmb_well_cmp.addItem(m)
            cmp_row.addWidget(self._cmb_well_cmp, stretch=1)
            b_cmp = QPushButton("Plot")
            b_cmp.clicked.connect(self._action_plot_well_cmp)
            cmp_row.addWidget(b_cmp)
            lay.addLayout(cmp_row)
            self._data_buttons.append(b_cmp)

        # ── Hedonic-specific ─────────────────────────────────────────────
        if exp_type == "hedonic":
            b = QPushButton("Hedonic feeding plot")
            b.clicked.connect(self._action_plot_hedonic)
            lay.addWidget(b)
            self._data_buttons.append(b)

        # ── Progressive-ratio breaking-point ────────────────────────────
        if exp_type == "progressive_ratio":
            pr_row = QHBoxLayout()
            pr_row.addWidget(QLabel("BP config:"))
            self._spin_pr_cfg = QSpinBox()
            self._spin_pr_cfg.setRange(1, 4)
            self._spin_pr_cfg.setValue(1)
            pr_row.addWidget(self._spin_pr_cfg)
            pr_row.addStretch()
            b_pr = QPushButton("Breaking-point plots")
            b_pr.clicked.connect(self._action_plot_pr)
            pr_row.addWidget(b_pr)
            lay.addLayout(pr_row)
            self._data_buttons.append(b_pr)

        lay.addStretch()

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

    def _start_worker(self, task: Callable[[], Any]) -> None:
        if self._busy:
            QMessageBox.information(self, "Busy", "An analysis task is already running.")
            return
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "Busy", "An analysis task is already running.")
            return
        p = self._project_dir()
        if not (p / "flic_config.yaml").is_file():
            QMessageBox.warning(self, "No config", "Select a directory containing flic_config.yaml.")
            return

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

    def _on_failed(self, msg: str) -> None:
        self._append_log(msg)
        QMessageBox.critical(self, "Analysis error", msg[:1200])

    def _append_log(self, text: str) -> None:
        self._log.appendPlainText(text.rstrip())
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _show_figure(self, title: str, png_bytes: bytes) -> None:
        win = _PlotWindow(title, png_bytes, parent=self)
        win.show()
        self._plot_windows.append(win)
        win.finished.connect(
            lambda: self._plot_windows.remove(win)
            if win in self._plot_windows else None
        )

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

    def _exp_cache_key(self) -> tuple:
        return (str(self._project_dir()), self._range_minutes(), self._chk_parallel.isChecked())

    def _load_exp(self):
        key = self._exp_cache_key()
        if self._cached_exp is not None and self._cached_exp_key == key:
            print("Using cached experiment (same project / options).", flush=True)
            return self._cached_exp

        from pyflic import load_experiment_yaml

        exp = load_experiment_yaml(
            self._project_dir(),
            range_minutes=self._range_minutes(),
            parallel=self._chk_parallel.isChecked(),
        )
        self._cached_exp = exp
        self._cached_exp_key = key
        return exp

    def _invalidate_exp_cache(self) -> None:
        self._cached_exp = None
        self._cached_exp_key = None
        self._exp_loaded = False
        self._update_data_buttons()

    def _launch_config_editor(self) -> None:
        p = self._project_dir()
        if not p.is_dir():
            QMessageBox.warning(self, "Invalid path", "Choose a valid project directory.")
            return
        cmd = _resolve_cli("pyflic-config", "pyflic.base.config_editor")
        try:
            subprocess.Popen(cmd, cwd=str(p))  # noqa: S603
        except OSError as e:
            QMessageBox.critical(self, "Could not start", str(e))

    def _qc_dir_for_range(self) -> Path:
        """
        Resolve the QC output directory for the current range.

        Mirrors ``Experiment._range_suffix``: ``(0, 0)`` → ``qc``; finite
        ``(a, b)`` → ``qc_{int(a)}_{int(b)}``; ``b`` of ``inf`` or ``0`` is
        written as ``end`` by the producer, so we resolve that by matching the
        existing ``qc_{int(a)}_*`` directory on disk.
        """
        p = self._project_dir()
        a, b = self._range_minutes()
        if a == 0.0 and b == 0.0:
            return p / "qc"
        a_lbl = str(int(a))
        if b == float("inf") or b == 0.0:
            exact = p / f"qc_{a_lbl}_end"
            if exact.is_dir():
                return exact
            matches = sorted(p.glob(f"qc_{a_lbl}_*"))
            if matches:
                return matches[0]
            return p / "qc"
        ranged = p / f"qc_{a_lbl}_{int(b)}"
        if ranged.is_dir():
            return ranged
        return p / "qc"

    def _launch_qc_viewer(self) -> None:
        p = self._project_dir()
        if not p.is_dir():
            QMessageBox.warning(self, "Invalid path", "Choose a valid project directory.")
            return
        qc_dir = self._qc_dir_for_range()
        if not qc_dir.is_dir():
            ans = QMessageBox.question(
                self,
                "No QC found",
                f"No QC directory found at:\n{qc_dir}\n\n"
                "Run 'Run full basic analysis' first to generate QC reports, or "
                "open the viewer anyway?",
                QMessageBox.StandardButton.Open | QMessageBox.StandardButton.Cancel,
            )
            if ans != QMessageBox.StandardButton.Open:
                return
        cmd = _resolve_cli("pyflic-qc", "pyflic.base.qc_viewer")
        cmd = [*cmd, str(p), str(qc_dir)]
        try:
            subprocess.Popen(cmd)  # noqa: S603
        except OSError as e:
            QMessageBox.critical(self, "Could not start", str(e))

    # ------------------------------------------------------------------
    # Analyze actions  (return None — no figure display)
    # ------------------------------------------------------------------

    def _action_load_experiment(self) -> None:
        def task() -> None:
            exp = self._load_exp()

            yaml_excl = exp.yaml_excluded_chambers
            print("Chambers excluded in YAML", flush=True)
            print("-------------------------", flush=True)
            if yaml_excl:
                total = 0
                for dfm_id, chambers in sorted(yaml_excl.items()):
                    print(f"  DFM {dfm_id}: chamber(s) {chambers}", flush=True)
                    total += len(chambers)
                print(f"  Total: {total} chamber(s).", flush=True)
            else:
                print("  (none)", flush=True)

            print(flush=True)
            print(exp.summary_text(include_qc=False), flush=True)

        self._start_worker(task)
        if self._worker is not None:
            self._worker.finished.connect(self._on_load_finished)

    def _on_load_finished(self) -> None:
        if self._cached_exp is not None:
            self._exp_loaded = True
            self._update_data_buttons()

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
            p = exp.write_feeding_summary(range_minutes=rm)
            print(f"Wrote: {p}", flush=True)

        self._start_worker(task)

    def _action_binned_csv(self) -> None:
        rm = self._range_minutes()
        bs = float(self._spin_binsize.value())

        def task() -> None:
            exp = self._load_exp()
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
                fig.save(str(out), dpi=150)
            else:
                fig.savefig(str(out), dpi=150, bbox_inches="tight")
            print(f"Wrote: {out}", flush=True)
            return "Feeding Summary", _figure_to_bytes(fig, dpi=100)

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
            fig.savefig(str(out), dpi=150, bbox_inches="tight")
            print(f"Wrote: {out}", flush=True)
            return f"Binned: {metric}", _figure_to_bytes(fig, dpi=100)

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
                fig.save(str(out), dpi=150)
            else:
                fig.savefig(str(out), dpi=150, bbox_inches="tight")
            print(f"Wrote: {out}", flush=True)
            return f"Dot: {metric}", _figure_to_bytes(fig, dpi=100)

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
            fig.save(str(out), dpi=150)
            print(f"Wrote: {out}", flush=True)
            return f"Well A vs B: {metric}", _figure_to_bytes(fig, dpi=100)

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
            return "Hedonic Feeding Plot", _figure_to_bytes(fig, dpi=100)

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
                fig.save(str(out), dpi=150)
                print(f"Wrote: {out}", flush=True)
                results.append((
                    f"Breaking Point — DFM {dfm_id} (config {cfg})",
                    _figure_to_bytes(fig, dpi=100),
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
        cfg = self._project_dir() / "flic_config.yaml"
        if not cfg.is_file():
            QMessageBox.warning(self, "No config", f"No flic_config.yaml in {self._project_dir()}")
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

        def task() -> list[tuple[str, bytes]]:
            from .analytics import compare_configs
            df = compare_configs(
                self._project_dir(), Path(d),
                metrics=(metric.strip(),), two_well_mode="total",
                range_minutes=rm,
            )
            out = self._project_dir() / "analysis" / f"compare_configs_{metric.strip()}.csv"
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
        idx = self._cmb_script.currentIndex()
        if idx < 0 or idx >= len(self._scripts):
            return
        self._run_script(self._scripts[idx])

    def _run_script(self, script: dict) -> None:
        steps = script.get("steps") or []
        ui_start, ui_end = self._range_minutes()
        script_start = float(script.get("start", ui_start))
        script_end = float(script.get("end", ui_end))
        ui_binsize = float(self._spin_binsize.value())
        ui_parallel = self._chk_parallel.isChecked()
        project_dir = self._project_dir()

        def task() -> list[tuple[str, bytes]]:
            exp = None
            figures: list[tuple[str, bytes]] = []

            def _get_rm(step: dict) -> tuple[float, float]:
                return (
                    float(step.get("start", script_start)),
                    float(step.get("end", script_end)),
                )

            def _make_cache_key(rm, parallel):
                return (str(project_dir), rm, parallel)

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
                        project_dir, range_minutes=rm, parallel=ui_parallel
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
                            project_dir, range_minutes=rm, parallel=parallel
                        )
                        self._cached_exp = exp
                        self._cached_exp_key = cache_key
                        print("Loaded.", flush=True)

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
                        fig.save(str(out), dpi=150)
                    else:
                        fig.savefig(str(out), dpi=150, bbox_inches="tight")
                    print(f"Wrote: {out}", flush=True)
                    figures.append(("Feeding Summary", _figure_to_bytes(fig)))

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
                        fig.savefig(str(out), dpi=150, bbox_inches="tight")
                        print(f"Wrote: {out}", flush=True)
                        figures.append((f"Binned: {metric}", _figure_to_bytes(fig)))
                    else:
                        fig = e.plot_dot_metric_by_treatment(
                            metric=metric, two_well_mode=mode, range_minutes=rm
                        )
                        safe = metric.replace("/", "_")
                        out = e.analysis_dir / f"dot_{safe}.png"
                        out.parent.mkdir(parents=True, exist_ok=True)
                        if hasattr(fig, "save"):
                            fig.save(str(out), dpi=150)
                        else:
                            fig.savefig(str(out), dpi=150, bbox_inches="tight")
                        print(f"Wrote: {out}", flush=True)
                        figures.append((f"Dot: {metric}", _figure_to_bytes(fig)))

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
                    fig.save(str(out), dpi=150)
                    print(f"Wrote: {out}", flush=True)
                    figures.append((f"Well A vs B: {metric}", _figure_to_bytes(fig)))

                elif action == "plot_hedonic":
                    from pyflic import HedonicFeedingExperiment
                    e = _ensure_exp(rm)
                    if not isinstance(e, HedonicFeedingExperiment):
                        print("[Skip] plot_hedonic requires hedonic experiment.", flush=True)
                        continue
                    fig = e.hedonic_feeding_plot(save=True, range_minutes=rm)
                    print("Wrote hedonic feeding plot.", flush=True)
                    figures.append(("Hedonic Feeding Plot", _figure_to_bytes(fig)))

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
                        fig.save(str(out), dpi=150)
                        print(f"Wrote: {out}", flush=True)
                        figures.append((
                            f"Breaking Point — DFM {dfm_id} (config {cfg_idx})",
                            _figure_to_bytes(fig),
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

                else:
                    print(f"[Skip] Unknown action: {action!r}", flush=True)

            return figures

        self._start_worker(task)
        if self._worker is not None:
            self._worker.finished.connect(self._on_load_finished)


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("pyflic Analysis Hub")
    project_dir = sys.argv[1] if len(sys.argv) > 1 else None
    win = AnalysisHubWindow(project_dir=project_dir)
    win.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
