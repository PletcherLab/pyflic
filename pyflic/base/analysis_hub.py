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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

    def _rebuild_dynamic_groups(self, exp_type: str | None, chamber_size: int | None) -> None:
        self._data_buttons.clear()
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
        p = self._project_dir()
        a, b = self._range_minutes()
        if a == 0.0 and b == 0.0:
            return p / "qc"
        ranged = p / f"qc_{int(a)}_{int(b)}"
        if ranged.is_dir():
            return ranged
        return p / "qc"

    def _launch_qc_viewer(self) -> None:
        p = self._project_dir()
        if not p.is_dir():
            QMessageBox.warning(self, "Invalid path", "Choose a valid project directory.")
            return
        qc_dir = self._qc_dir_for_range()
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
            paths = exp.execute_basic_analysis(range_minutes=rm, skip_qc=True)
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


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("pyflic Analysis Hub")
    project_dir = sys.argv[1] if len(sys.argv) > 1 else None
    win = AnalysisHubWindow(project_dir=project_dir)
    win.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
