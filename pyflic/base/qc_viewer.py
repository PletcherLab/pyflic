#!/usr/bin/env python3
"""
FLIC QC Viewer
==============
Interactive PyQt dashboard for loading experiments, viewing QC results, and
managing per-chamber exclusions that are persisted back to ``flic_config.yaml``.

Usage
-----
    pyflic-qc [project_dir]

Workflow
--------
1. **Load tab** — enter the project directory, time range, and parallelism
   options, then click "Load".  Loading runs in a background thread and its
   progress (stdout from pyflic) is streamed into the log window.
2. **Feeding Summary tab** — after loading a table of all chambers is shown.
   The "Excl." checkbox in column 0 marks a chamber for exclusion.
3. **DFM N tabs** — one tab per DFM.  The left side shows the usual QC
   sub-tabs (Integrity, Sim. Feeding, Bleeding, Raw Signal …).  The right
   side shows a narrow "Exclude Wells" panel with one checkbox per well (1-12).

Exclusion checkboxes are **bidirectionally synchronised**: toggling a chamber
in the Feeding Summary tab updates the corresponding well checkbox in the DFM
tab and vice versa.

"Save to YAML" writes the current exclusion state back to
``project_dir/flic_config.yaml`` under ``excluded_chambers:`` for each DFM.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# ── matplotlib backend must be set before any pyplot import ────────────────
import matplotlib
matplotlib.use("QtAgg")

import matplotlib.image as mpimg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# ── PyQt — try 6 first, fall back to 5 ─────────────────────────────────────
try:
    from PyQt6 import QtCore, QtWidgets
    from PyQt6.QtCore import Qt, pyqtSignal
    _PYQT = 6
    _Checked   = Qt.CheckState.Checked
    _Unchecked = Qt.CheckState.Unchecked
except ImportError:
    from PyQt5 import QtCore, QtWidgets          # type: ignore[no-redef]
    from PyQt5.QtCore import Qt, pyqtSignal      # type: ignore[no-redef]
    _PYQT = 5
    _Checked   = Qt.Checked                      # type: ignore[attr-defined]
    _Unchecked = Qt.Unchecked                    # type: ignore[attr-defined]

import shutil
import subprocess

import pandas as pd
import yaml


# ───────────────────────────────────────────────────────────────────────────
# Utility
# ───────────────────────────────────────────────────────────────────────────

def _resolve_cli(name: str, module: str) -> list[str]:
    """Return a command list for *name*, falling back to ``python -m module``."""
    exe = shutil.which(name)
    if exe:
        return [exe]
    return [sys.executable, "-m", module]


# ───────────────────────────────────────────────────────────────────────────
# Qt helpers (display-only table, CSV widget, text widget, PNG widget)
# ───────────────────────────────────────────────────────────────────────────

class _DfTableWidget(QtWidgets.QTableWidget):
    """Display a pandas DataFrame as a read-only QTableWidget."""

    def __init__(self, df: pd.DataFrame, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
            if _PYQT == 6 else
            QtWidgets.QAbstractItemView.NoEditTriggers  # type: ignore[attr-defined]
        )
        self.setAlternatingRowColors(True)
        self._populate(df)

    def _populate(self, df: pd.DataFrame) -> None:
        self.setRowCount(len(df))
        self.setColumnCount(len(df.columns))
        self.setHorizontalHeaderLabels([str(c) for c in df.columns])
        self.setVerticalHeaderLabels([str(i) for i in df.index])
        for r in range(len(df)):
            for c, col in enumerate(df.columns):
                val = df.iat[r, c]
                text = f"{val:.4g}" if isinstance(val, float) else str(val)
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    if _PYQT == 6 else
                    Qt.AlignRight | Qt.AlignVCenter  # type: ignore[attr-defined]
                )
                self.setItem(r, c, item)
        self.resizeColumnsToContents()
        self.resizeRowsToContents()


def _csv_widget(path: Path) -> QtWidgets.QWidget:
    """Load a CSV and return a table widget, or an error label."""
    w = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(w)
    layout.setContentsMargins(8, 8, 8, 8)
    if path.exists():
        try:
            df = pd.read_csv(path, index_col=0)
            layout.addWidget(_DfTableWidget(df))
        except Exception as exc:
            layout.addWidget(QtWidgets.QLabel(f"Could not load {path.name}:\n{exc}"))
    else:
        layout.addWidget(QtWidgets.QLabel(f"File not found:\n{path.name}"))
    return w


def _text_widget(path: Path) -> QtWidgets.QWidget:
    """Display a plain-text file in a read-only text area."""
    w = QtWidgets.QTextEdit()
    w.setReadOnly(True)
    if path.exists():
        w.setPlainText(path.read_text(encoding="utf-8", errors="replace"))
    else:
        w.setPlainText(f"File not found: {path.name}")
    return w


def _png_widget(path: Path) -> QtWidgets.QWidget:
    """Embed a PNG as a matplotlib image canvas."""
    w = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(w)
    layout.setContentsMargins(0, 0, 0, 0)
    if not path.exists():
        layout.addWidget(QtWidgets.QLabel(f"Image not found:\n{path.name}"))
        return w
    try:
        img = mpimg.imread(str(path))
        fig = Figure(tight_layout=True)
        ax = fig.add_subplot(111)
        ax.imshow(img)
        ax.axis("off")
        canvas = FigureCanvasQTAgg(fig)
        canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding if _PYQT == 6 else QtWidgets.QSizePolicy.Expanding,  # type: ignore[attr-defined]
            QtWidgets.QSizePolicy.Policy.Expanding if _PYQT == 6 else QtWidgets.QSizePolicy.Expanding,  # type: ignore[attr-defined]
        )
        layout.addWidget(canvas)
    except Exception as exc:
        layout.addWidget(QtWidgets.QLabel(f"Could not render image:\n{exc}"))
    return w


# ───────────────────────────────────────────────────────────────────────────
# Selectable table widget (feeding summary + exclusion checkboxes)
# ───────────────────────────────────────────────────────────────────────────

class _SelectableTableWidget(QtWidgets.QTableWidget):
    """
    A read-only ``QTableWidget`` that prepends an "Excl." checkbox column.

    Checked = excluded.  Rows listed in *pre_checked* start checked.
    Emits ``row_exclusion_changed(row, is_excluded)`` when the user toggles a
    checkbox.  The programmatic setter :meth:`set_row_excluded` is signal-
    silent (uses ``blockSignals``).
    """

    row_exclusion_changed = pyqtSignal(int, bool)

    def __init__(
        self,
        df: pd.DataFrame,
        pre_checked: list[int] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._df = df
        self._pre_checked: set[int] = set(pre_checked or [])
        self.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
            if _PYQT == 6 else
            QtWidgets.QAbstractItemView.NoEditTriggers  # type: ignore[attr-defined]
        )
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
            if _PYQT == 6 else
            QtWidgets.QAbstractItemView.SelectRows  # type: ignore[attr-defined]
        )
        self._populate()

    def _populate(self) -> None:
        cols = ["Excl."] + [str(c) for c in self._df.columns]
        self.setColumnCount(len(cols))
        self.setRowCount(len(self._df))
        self.setHorizontalHeaderLabels(cols)

        if _PYQT == 6:
            _checkable = Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
            _readonly  = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            _align_r   = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        else:
            _checkable = Qt.ItemIsUserCheckable | Qt.ItemIsEnabled           # type: ignore[attr-defined]
            _readonly  = Qt.ItemIsEnabled | Qt.ItemIsSelectable              # type: ignore[attr-defined]
            _align_r   = Qt.AlignRight | Qt.AlignVCenter                     # type: ignore[attr-defined]

        self.blockSignals(True)
        for r in range(len(self._df)):
            # Col 0: exclusion checkbox
            chk = QtWidgets.QTableWidgetItem()
            chk.setFlags(_checkable)
            chk.setCheckState(_Checked if r in self._pre_checked else _Unchecked)
            self.setItem(r, 0, chk)

            # Remaining cols: data (read-only)
            for c, col in enumerate(self._df.columns):
                val = self._df.iat[r, c]
                text = f"{val:.4g}" if isinstance(val, float) else str(val)
                item = QtWidgets.QTableWidgetItem(text)
                item.setFlags(_readonly)
                item.setTextAlignment(_align_r)
                self.setItem(r, c + 1, item)
        self.blockSignals(False)

        self.resizeColumnsToContents()
        self.resizeRowsToContents()
        self.itemChanged.connect(self._on_item_changed)

    # ------------------------------------------------------------------

    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if item.column() == 0:
            self.row_exclusion_changed.emit(item.row(), item.checkState() == _Checked)

    def set_row_excluded(self, row: int, excluded: bool) -> None:
        """Set the exclusion state of *row* without emitting ``row_exclusion_changed``."""
        item = self.item(row, 0)
        if item is not None:
            self.blockSignals(True)
            item.setCheckState(_Checked if excluded else _Unchecked)
            self.blockSignals(False)

    def set_all_checked(self, checked: bool) -> None:
        """Mark all rows as excluded (``True``) or included (``False``) silently."""
        state = _Checked if checked else _Unchecked
        self.blockSignals(True)
        for r in range(self.rowCount()):
            item = self.item(r, 0)
            if item is not None:
                item.setCheckState(state)
        self.blockSignals(False)

    def excluded_rows(self) -> list[int]:
        return [
            r for r in range(self.rowCount())
            if self.item(r, 0) is not None and self.item(r, 0).checkState() == _Checked
        ]

    def excluded_dataframe(self) -> pd.DataFrame:
        return self._df.iloc[self.excluded_rows()].copy()

    def included_dataframe(self) -> pd.DataFrame:
        excl = set(self.excluded_rows())
        keep = [r for r in range(len(self._df)) if r not in excl]
        return self._df.iloc[keep].copy()


# ───────────────────────────────────────────────────────────────────────────
# Stdout redirect — forwards print() output from a worker thread to a Qt signal
# ───────────────────────────────────────────────────────────────────────────

class _LogRedirect:
    """Context manager: replaces ``sys.stdout`` with a callback-based writer."""

    def __init__(self, callback):
        self._callback = callback

    def write(self, text: str) -> int:
        if text:
            self._callback(text)
        return len(text)

    def flush(self) -> None:
        pass

    def __enter__(self):
        self._orig = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *_):
        sys.stdout = self._orig


# ───────────────────────────────────────────────────────────────────────────
# Background load worker
# ───────────────────────────────────────────────────────────────────────────

class _LoadWorker(QtCore.QObject):
    """Loads an experiment on a background thread, emitting progress and result signals."""

    finished = pyqtSignal(object)   # experiment
    errored  = pyqtSignal(str)      # error message
    logged   = pyqtSignal(str)      # stdout text

    def __init__(
        self,
        project_dir: Path,
        range_minutes: tuple[float, float],
        parallel: bool,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._project_dir  = project_dir
        self._range_minutes = range_minutes
        self._parallel     = parallel

    def run(self) -> None:
        from .yaml_config import load_experiment_yaml

        with _LogRedirect(self.logged.emit):
            try:
                exp = load_experiment_yaml(
                    self._project_dir,
                    range_minutes=self._range_minutes,
                    parallel=self._parallel,
                )
                print("Writing QC reports...", flush=True)
                exp.write_qc_reports()
                # Print integrity report text for each DFM
                qc = exp.qc_results or {}
                for dfm_id in sorted(qc.keys()):
                    text = qc[dfm_id].get("integrity_text")
                    if text:
                        print(f"\n--- DFM {dfm_id} integrity report ---", flush=True)
                        print(text, flush=True)
                self.finished.emit(exp)
            except Exception as exc:
                self.errored.emit(str(exc))


# ───────────────────────────────────────────────────────────────────────────
# Load tab
# ───────────────────────────────────────────────────────────────────────────

class LoadTab(QtWidgets.QWidget):
    """
    Form for loading an experiment.

    Emits ``experiment_loaded(exp)`` on the main thread after a successful
    background load.
    """

    experiment_loaded = pyqtSignal(object)

    def __init__(self, project_dir: Path, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._project_dir = project_dir
        self._thread: QtCore.QThread | None = None
        self._worker: _LoadWorker | None = None
        self._build()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        # ── Project directory ──────────────────────────────────────────
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 8)

        self._project_edit = QtWidgets.QLineEdit(str(self._project_dir))
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_project)
        dir_row = QtWidgets.QHBoxLayout()
        dir_row.addWidget(self._project_edit, stretch=1)
        dir_row.addWidget(browse_btn)
        form.addRow("Project directory:", dir_row)

        # ── Time range ────────────────────────────────────────────────
        self._start_spin = QtWidgets.QDoubleSpinBox()
        self._start_spin.setRange(0.0, 100_000.0)
        self._start_spin.setDecimals(1)
        self._start_spin.setValue(0.0)
        self._start_spin.setSuffix(" min")

        self._end_spin = QtWidgets.QDoubleSpinBox()
        self._end_spin.setRange(0.0, 100_000.0)
        self._end_spin.setDecimals(1)
        self._end_spin.setValue(0.0)
        self._end_spin.setSuffix(" min  (0 = load all)")

        form.addRow("Start time:", self._start_spin)
        form.addRow("End time:",   self._end_spin)

        # ── Parallel ──────────────────────────────────────────────────
        self._parallel_cb = QtWidgets.QCheckBox("Load DFMs in parallel")
        self._parallel_cb.setChecked(True)
        form.addRow("", self._parallel_cb)

        root.addLayout(form)

        # ── Action buttons row ────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()

        self._load_btn = QtWidgets.QPushButton("Load")
        self._load_btn.setFixedWidth(120)
        self._load_btn.clicked.connect(self._on_load_clicked)
        btn_row.addWidget(self._load_btn)

        edit_cfg_btn = QtWidgets.QPushButton("Edit Config")
        edit_cfg_btn.setToolTip("Open flic_config.yaml in the pyflic-config editor")
        edit_cfg_btn.clicked.connect(self._on_edit_config)
        btn_row.addWidget(edit_cfg_btn)

        reload_cfg_btn = QtWidgets.QPushButton("Reload Config")
        reload_cfg_btn.setToolTip("Re-read flic_config.yaml and show a summary in the log")
        reload_cfg_btn.clicked.connect(self._on_reload_config)
        btn_row.addWidget(reload_cfg_btn)

        self._status_label = QtWidgets.QLabel("Ready.")
        btn_row.addWidget(self._status_label, stretch=1)
        root.addLayout(btn_row)

        # ── Progress bar ──────────────────────────────────────────────
        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 0)   # indeterminate
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        # ── Log window ────────────────────────────────────────────────
        self._log = QtWidgets.QPlainTextEdit()
        self._log.setReadOnly(True)
        font = self._log.font()
        font.setFamily("Monospace")
        self._log.setFont(font)
        self._log.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        root.addWidget(self._log, stretch=1)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _browse_project(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select project directory", str(self._project_dir)
        )
        if d:
            self._project_edit.setText(d)

    def _current_project_dir(self) -> Path:
        return Path(self._project_edit.text().strip()).expanduser().resolve()

    def _on_edit_config(self) -> None:
        """Launch pyflic-config in the current project directory."""
        p = self._current_project_dir()
        if not p.is_dir():
            self._status_label.setText(f"Not a directory: {p}")
            return
        cmd = _resolve_cli("pyflic-config", "pyflic.base.config_editor")
        try:
            subprocess.Popen(cmd, cwd=str(p))   # noqa: S603
            self._status_label.setText(f"Launched pyflic-config in {p.name}/")
        except OSError as exc:
            self._status_label.setText(f"Could not launch pyflic-config: {exc}")

    def _on_reload_config(self) -> None:
        """Re-read flic_config.yaml and print a summary to the log."""
        p = self._current_project_dir()
        config_path = p / "flic_config.yaml"
        if not config_path.exists():
            self._append_log(f"[Reload] flic_config.yaml not found in {p}")
            return
        try:
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._append_log(f"[Reload] Could not parse flic_config.yaml: {exc}")
            return

        g      = cfg.get("global", {}) or {}
        params = g.get("params", {}) or {}
        dfms   = cfg.get("dfms", []) or []
        lines  = [
            f"[Reload] {config_path}",
            f"  experiment_type : {g.get('experiment_type', '(not set)')}",
            f"  chamber_size    : {params.get('chamber_size', '(not set)')}",
            f"  pi_direction    : {params.get('pi_direction', '(not set)')}",
            f"  DFMs defined    : {len(dfms)}",
        ]
        for dfm_node in dfms:
            if not isinstance(dfm_node, dict):
                continue
            dfm_id   = dfm_node.get("id", "?")
            chambers = dfm_node.get("chambers", {}) or {}
            excluded = dfm_node.get("excluded_chambers", []) or []
            n_ch     = len(chambers) if isinstance(chambers, dict) else len(chambers)
            excl_str = f"  excluded: {excluded}" if excluded else ""
            lines.append(f"    DFM {dfm_id}: {n_ch} chamber(s){excl_str}")

        self._append_log("\n".join(lines))
        self._status_label.setText("Config reloaded — see log.")

    def _on_load_clicked(self) -> None:
        project_dir = self._current_project_dir()
        if not project_dir.is_dir():
            self._status_label.setText(f"Not a directory: {project_dir}")
            return

        self._project_dir = project_dir
        range_minutes = (float(self._start_spin.value()), float(self._end_spin.value()))
        parallel = self._parallel_cb.isChecked()

        self._log.clear()
        self._status_label.setText("Loading…")
        self._load_btn.setEnabled(False)
        self._progress.setVisible(True)

        self._worker = _LoadWorker(project_dir, range_minutes, parallel)
        self._thread = QtCore.QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.logged.connect(self._append_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.errored.connect(self._on_errored)
        self._worker.finished.connect(self._thread.quit)
        self._worker.errored.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _append_log(self, text: str) -> None:
        text = text.rstrip("\n")
        if text:
            self._log.appendPlainText(text)

    def _on_finished(self, exp) -> None:
        self._progress.setVisible(False)
        self._load_btn.setEnabled(True)
        self._status_label.setText("Loaded successfully.")
        self.experiment_loaded.emit(exp)

    def _on_errored(self, msg: str) -> None:
        self._progress.setVisible(False)
        self._load_btn.setEnabled(True)
        self._status_label.setText(f"Error: {msg}")
        self._log.appendPlainText(f"\n[ERROR] {msg}")


# ───────────────────────────────────────────────────────────────────────────
# Feeding Summary tab
# ───────────────────────────────────────────────────────────────────────────

class FeedingSummaryTab(QtWidgets.QWidget):
    """
    Displays the feeding summary as a table with an "Excl." checkbox column.

    Checked = excluded.  Pre-checks rows whose ``(DFM, Chamber)`` pair appears
    in *excluded_by_dfm*.

    Emits ``exclusion_changed(dfm_id, chamber, is_excluded)`` when the user
    toggles a row.  Provides :meth:`set_chamber_excluded` for the reverse
    (DFM-tab → Feeding Summary) sync; that setter is signal-silent.
    """

    exclusion_changed      = pyqtSignal(int, int, bool)  # dfm_id, chamber, excluded
    auto_filter_requested  = pyqtSignal()
    view_criteria_requested = pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._df: pd.DataFrame | None = None
        self._table: _SelectableTableWidget | None = None
        self._root = QtWidgets.QVBoxLayout(self)
        self._root.setContentsMargins(8, 8, 8, 8)

    # ------------------------------------------------------------------

    def populate(
        self,
        df: pd.DataFrame,
        excluded_by_dfm: dict[int, list[int]] | None = None,
    ) -> None:
        """Populate (or re-populate) the table from *df*."""
        # Clear any existing content
        while self._root.count():
            item = self._root.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Sort by DFM then Chamber before display
        sort_cols = [c for c in ("DFM", "Chamber") if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, ascending=True)
        self._df = df.reset_index(drop=True)

        # Determine pre-checked rows
        pre_checked: list[int] = []
        if excluded_by_dfm:
            for r in range(len(self._df)):
                row = self._df.iloc[r]
                dfm_id  = int(row.get("DFM",     -1)) if "DFM"     in self._df.columns else -1
                chamber = int(row.get("Chamber", -1)) if "Chamber" in self._df.columns else -1
                if dfm_id >= 0 and chamber in (excluded_by_dfm.get(dfm_id) or []):
                    pre_checked.append(r)

        self._table = _SelectableTableWidget(self._df, pre_checked=pre_checked)
        self._table.row_exclusion_changed.connect(self._on_row_changed)

        # Button bar
        btn_bar = QtWidgets.QHBoxLayout()
        mark_all_btn = QtWidgets.QPushButton("Mark All Excluded")
        mark_all_btn.clicked.connect(lambda: self._set_all_and_sync(True))
        clear_all_btn = QtWidgets.QPushButton("Clear All")
        clear_all_btn.clicked.connect(lambda: self._set_all_and_sync(False))
        auto_filter_btn = QtWidgets.QPushButton("Auto Filter Chambers")
        auto_filter_btn.setToolTip(
            "Apply thresholds from global_constants in flic_config.yaml to automatically "
            "exclude chambers that fail QC criteria."
        )
        auto_filter_btn.clicked.connect(self.auto_filter_requested)
        criteria_btn = QtWidgets.QPushButton("View Filter Criteria")
        criteria_btn.setToolTip("Show the criteria used by the last Auto Filter run.")
        criteria_btn.clicked.connect(self.view_criteria_requested)
        export_btn = QtWidgets.QPushButton("Export Included to CSV…")
        export_btn.clicked.connect(self._export_included)
        btn_bar.addWidget(mark_all_btn)
        btn_bar.addWidget(clear_all_btn)
        btn_bar.addWidget(auto_filter_btn)
        btn_bar.addWidget(criteria_btn)
        btn_bar.addWidget(export_btn)
        btn_bar.addStretch()

        self._root.addLayout(btn_bar)
        self._root.addWidget(self._table, stretch=1)

    def _set_all_and_sync(self, excluded: bool) -> None:
        """Set all rows to *excluded* and propagate the change to DFM tabs."""
        if self._table is None or self._df is None:
            return
        self._table.set_all_checked(excluded)
        for r in range(len(self._df)):
            row     = self._df.iloc[r]
            dfm_id  = int(row.get("DFM",     -1)) if "DFM"     in self._df.columns else -1
            chamber = int(row.get("Chamber", -1)) if "Chamber" in self._df.columns else -1
            if dfm_id >= 0 and chamber >= 0:
                self.exclusion_changed.emit(dfm_id, chamber, excluded)

    def set_chamber_excluded(self, dfm_id: int, chamber: int, excluded: bool) -> None:
        """Programmatically set the exclusion state of a (DFM, Chamber) row (signal-silent)."""
        if self._df is None or self._table is None:
            return
        for r in range(len(self._df)):
            row = self._df.iloc[r]
            rd  = int(row.get("DFM",     -1)) if "DFM"     in self._df.columns else -1
            rc  = int(row.get("Chamber", -1)) if "Chamber" in self._df.columns else -1
            if rd == dfm_id and rc == chamber:
                self._table.set_row_excluded(r, excluded)
                return

    def _on_row_changed(self, row: int, excluded: bool) -> None:
        if self._df is None:
            return
        r       = self._df.iloc[row]
        dfm_id  = int(r.get("DFM",     -1)) if "DFM"     in self._df.columns else -1
        chamber = int(r.get("Chamber", -1)) if "Chamber" in self._df.columns else -1
        if dfm_id >= 0 and chamber >= 0:
            self.exclusion_changed.emit(dfm_id, chamber, excluded)

    def _export_included(self) -> None:
        if self._table is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export included chambers", "", "CSV files (*.csv)"
        )
        if path:
            self._table.included_dataframe().to_csv(path, index=False)


# ───────────────────────────────────────────────────────────────────────────
# Per-DFM tab  (QC sub-tabs + exclusion sidebar)
# ───────────────────────────────────────────────────────────────────────────

class DfmTab(QtWidgets.QWidget):
    """
    Displays QC sub-tabs for a single DFM, plus an "Exclude Wells" sidebar.

    The QC side shows whatever pre-computed files are present in *qc_dir*
    (integrity, simultaneous feeding, bleeding, signal plots).

    The sidebar has checkboxes for wells 1-12 (checked = excluded).

    Emits ``exclusion_changed(dfm_id, well_num, is_excluded)`` when a checkbox
    is toggled by the user.  Provides :meth:`set_well_excluded` for the
    reverse (Feeding Summary → DFM) sync; that setter is signal-silent.
    """

    exclusion_changed = pyqtSignal(int, int, bool)   # dfm_id, well_num, excluded

    def __init__(
        self,
        dfm_id: int,
        qc_dir: Path,
        excluded_wells: list[int] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._dfm_id = dfm_id
        self._checkboxes: dict[int, QtWidgets.QCheckBox] = {}
        self._build(dfm_id, qc_dir, excluded_wells or [])

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build(self, dfm_id: int, qc_dir: Path, excluded_wells: list[int]) -> None:
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        # Left: QC sub-tabs
        self._tabs = QtWidgets.QTabWidget()
        outer.addWidget(self._tabs, stretch=1)
        self._build_qc_tabs(dfm_id, qc_dir)

        # Right: exclusion panel
        excl_panel = self._build_exclusion_panel(excluded_wells)
        outer.addWidget(excl_panel, stretch=0)

    @staticmethod
    def _resolve(qc_dir: Path, subdir: str, filename: str) -> Path:
        p = qc_dir / subdir / filename
        if p.exists():
            return p
        return qc_dir / filename

    def _build_qc_tabs(self, dfm_id: int, qc_dir: Path) -> None:
        tabs = self._tabs

        # ── Integrity ─────────────────────────────────────────────────
        integ_csv = self._resolve(qc_dir, "integrity", f"DFM{dfm_id}_integrity_report.csv")
        integ_txt = self._resolve(qc_dir, "integrity", f"DFM{dfm_id}_integrity_report.txt")

        integ_widget = QtWidgets.QWidget()
        integ_outer = QtWidgets.QVBoxLayout(integ_widget)
        integ_outer.setContentsMargins(8, 8, 8, 8)

        if not integ_csv.exists() and not integ_txt.exists():
            integ_outer.addWidget(QtWidgets.QLabel("No integrity files found."))
        else:
            splitter = QtWidgets.QSplitter(
                Qt.Orientation.Vertical if _PYQT == 6 else Qt.Vertical  # type: ignore[attr-defined]
            )
            integ_outer.addWidget(splitter)

            top = QtWidgets.QWidget()
            top_layout = QtWidgets.QVBoxLayout(top)
            top_layout.setContentsMargins(0, 0, 0, 4)
            if integ_csv.exists():
                try:
                    df = pd.read_csv(integ_csv)
                    top_layout.addWidget(_DfTableWidget(df))
                except Exception as exc:
                    top_layout.addWidget(QtWidgets.QLabel(f"Error reading CSV: {exc}"))
            else:
                top_layout.addWidget(QtWidgets.QLabel("Integrity CSV not found."))
            splitter.addWidget(top)

            txt = QtWidgets.QTextEdit()
            txt.setReadOnly(True)
            if integ_txt.exists():
                txt.setPlainText(integ_txt.read_text(encoding="utf-8", errors="replace"))
            else:
                txt.setPlainText("Integrity text report not found.")
            splitter.addWidget(txt)
            splitter.setSizes([100, 330])

        tabs.addTab(integ_widget, "Integrity")

        # ── Simultaneous Feeding + Bleeding (two-well only) ───────────
        sim_path = self._resolve(
            qc_dir, "simultaneous_feeding", f"DFM{dfm_id}_simultaneous_feeding_matrix.csv"
        )
        if sim_path.exists():
            sim_widget = QtWidgets.QWidget()
            sim_layout = QtWidgets.QVBoxLayout(sim_widget)
            sim_layout.setContentsMargins(8, 8, 8, 8)
            try:
                df = pd.read_csv(sim_path, index_col=0)
                sim_layout.addWidget(_DfTableWidget(df))
            except Exception as exc:
                sim_layout.addWidget(QtWidgets.QLabel(f"Error reading CSV: {exc}"))
            sim_layout.addStretch()
            tabs.addTab(sim_widget, "Sim. Feeding")

            bleed_mat  = self._resolve(qc_dir, "bleeding", f"DFM{dfm_id}_bleeding_matrix.csv")
            bleed_all  = self._resolve(qc_dir, "bleeding", f"DFM{dfm_id}_bleeding_alldata.csv")
            bleed_w    = QtWidgets.QWidget()
            bleed_outer = QtWidgets.QVBoxLayout(bleed_w)
            bleed_outer.setContentsMargins(8, 8, 8, 8)
            found = False

            splitter = QtWidgets.QSplitter(
                Qt.Orientation.Vertical if _PYQT == 6 else Qt.Vertical  # type: ignore[attr-defined]
            )

            if bleed_mat.exists():
                found = True
                top = QtWidgets.QWidget()
                top_lay = QtWidgets.QVBoxLayout(top)
                top_lay.setContentsMargins(0, 0, 0, 0)
                top_lay.addWidget(QtWidgets.QLabel("<b>Bleeding matrix:</b>"))
                try:
                    df = pd.read_csv(bleed_mat, index_col=0)
                    top_lay.addWidget(_DfTableWidget(df))
                except Exception as exc:
                    top_lay.addWidget(QtWidgets.QLabel(f"Error: {exc}"))
                splitter.addWidget(top)

            if bleed_all.exists():
                found = True
                bot = QtWidgets.QWidget()
                bot_lay = QtWidgets.QVBoxLayout(bot)
                bot_lay.setContentsMargins(0, 0, 0, 0)
                bot_lay.addWidget(QtWidgets.QLabel("<b>All data:</b>"))
                try:
                    df = pd.read_csv(bleed_all, index_col=0)
                    bot_lay.addWidget(_DfTableWidget(df))
                except Exception as exc:
                    bot_lay.addWidget(QtWidgets.QLabel(f"Error: {exc}"))
                splitter.addWidget(bot)

            if found:
                bleed_outer.addWidget(splitter)
            else:
                bleed_outer.addWidget(QtWidgets.QLabel("No bleeding data found."))
                bleed_outer.addStretch()
            tabs.addTab(bleed_w, "Bleeding")

        # ── Signal plots ──────────────────────────────────────────────
        for label, subdir, suffix in [
            ("Raw Signal",       "raw_signal",       f"DFM{dfm_id}_raw.png"),
            ("Baselined",        "baselined",        f"DFM{dfm_id}_baselined.png"),
            ("Cumulative Licks", "cumulative_licks", f"DFM{dfm_id}_cumulative_licks.png"),
        ]:
            tabs.addTab(_png_widget(self._resolve(qc_dir, subdir, suffix)), label)

    def _build_exclusion_panel(self, excluded_wells: list[int]) -> QtWidgets.QGroupBox:
        panel = QtWidgets.QGroupBox("Exclude Wells")
        panel.setFixedWidth(140)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        excluded_set = set(excluded_wells)
        for well_num in range(1, 13):
            cb = QtWidgets.QCheckBox(f"Well {well_num}")
            cb.setChecked(well_num in excluded_set)
            cb.toggled.connect(
                lambda checked, w=well_num: self.exclusion_changed.emit(self._dfm_id, w, checked)
            )
            self._checkboxes[well_num] = cb
            layout.addWidget(cb)

        layout.addStretch()
        return panel

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_well_excluded(self, well_num: int, excluded: bool) -> None:
        """Set the exclusion state of a well checkbox without emitting a signal."""
        cb = self._checkboxes.get(well_num)
        if cb is not None:
            cb.blockSignals(True)
            cb.setChecked(excluded)
            cb.blockSignals(False)

    def get_excluded_wells(self) -> list[int]:
        """Return the sorted list of currently checked (excluded) well numbers."""
        return sorted(n for n, cb in self._checkboxes.items() if cb.isChecked())


# ───────────────────────────────────────────────────────────────────────────
# Main window
# ───────────────────────────────────────────────────────────────────────────

class MainWindow(QtWidgets.QMainWindow):
    """
    Top-level window.

    Tab order after loading: Load | Feeding Summary | DFM 1 | DFM 2 | …

    Exclusion checkboxes in the Feeding Summary tab and DFM tabs are kept in
    sync via the ``_syncing`` flag and two cross-wired signal handlers.
    """

    def __init__(self, project_dir: Path) -> None:
        super().__init__()
        self.resize(1380, 900)
        self.setWindowTitle(f"FLIC QC Viewer  —  {project_dir}")

        self._project_dir: Path = project_dir
        self._exp = None
        self._syncing = False                        # prevents exclusion sync feedback loops
        self._dfm_tab_widgets: dict[int, DfmTab] = {}
        self._dfm_chamber_sizes: dict[int, int] = {}  # dfm_id → chamber_size
        self._feeding_tab: FeedingSummaryTab | None = None
        self._active_subtab_idx: int = 0

        # Auto-save timer: coalesces rapid changes (e.g. "Mark All") into one write
        self._autosave_timer = QtCore.QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(150)        # ms — wait for burst to settle
        self._autosave_timer.timeout.connect(self._save_exclusions_to_yaml)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root_layout = QtWidgets.QVBoxLayout(central)
        root_layout.setContentsMargins(4, 4, 4, 4)

        self._tabs = QtWidgets.QTabWidget()
        root_layout.addWidget(self._tabs)

        # ── Load tab (always first) ───────────────────────────────────
        self._load_tab = LoadTab(project_dir)
        self._load_tab.experiment_loaded.connect(self._on_experiment_loaded)
        self._tabs.addTab(self._load_tab, "Load")

    # ------------------------------------------------------------------
    # After experiment is loaded
    # ------------------------------------------------------------------

    def _on_experiment_loaded(self, exp) -> None:
        self._exp = exp
        self._project_dir = exp.project_dir
        self.setWindowTitle(f"FLIC QC Viewer  —  {self._project_dir}")

        # Remove any previously-loaded tabs (keep only Load tab at index 0)
        while self._tabs.count() > 1:
            w = self._tabs.widget(1)
            self._tabs.removeTab(1)
            if w:
                w.deleteLater()
        self._dfm_tab_widgets.clear()
        self._dfm_chamber_sizes.clear()
        self._feeding_tab = None

        # Cache chamber_size per DFM
        for dfm_id, dfm in exp.dfms.items():
            self._dfm_chamber_sizes[dfm_id] = int(dfm.params.chamber_size)

        # Read excluded_chambers from YAML (chamber numbers)
        excluded_by_dfm = self._read_excluded_from_yaml()

        # Feeding summary tab
        try:
            fs = exp.feeding_summary()
        except Exception as exc:
            fs = pd.DataFrame()
            self.statusBar().showMessage(f"Could not compute feeding summary: {exc}")

        self._feeding_tab = FeedingSummaryTab()
        self._feeding_tab.populate(fs, excluded_by_dfm=excluded_by_dfm)
        self._feeding_tab.exclusion_changed.connect(self._on_feeding_exclusion_changed)
        self._feeding_tab.auto_filter_requested.connect(self._on_auto_filter)
        self._feeding_tab.view_criteria_requested.connect(self._on_view_filter_criteria)
        self._tabs.addTab(self._feeding_tab, "Feeding Summary")

        # DFM tabs — convert excluded chamber numbers to well numbers
        qc_dir = self._project_dir / "qc"
        for dfm_id in sorted(exp.dfms.keys()):
            excl_chambers = excluded_by_dfm.get(dfm_id, [])
            excl_wells = [
                w
                for c in excl_chambers
                for w in self._wells_for_chamber(dfm_id, c)
            ]
            tab = DfmTab(dfm_id, qc_dir, excluded_wells=excl_wells)
            tab.exclusion_changed.connect(self._on_dfm_exclusion_changed)
            tab._tabs.currentChanged.connect(self._on_subtab_changed)
            self._dfm_tab_widgets[dfm_id] = tab
            self._tabs.addTab(tab, f"DFM {dfm_id}")

        self._tabs.currentChanged.connect(self._on_top_tab_changed)

        n_dfms = len(self._dfm_tab_widgets)
        self.statusBar().showMessage(
            f"Loaded {n_dfms} DFM(s) from {self._project_dir}"
        )
        self._tabs.setCurrentIndex(1)   # switch to Feeding Summary

    # ------------------------------------------------------------------
    # Subtab synchronisation (same as original viewer)
    # ------------------------------------------------------------------

    def _on_subtab_changed(self, idx: int) -> None:
        self._active_subtab_idx = idx

    def _on_top_tab_changed(self, new_idx: int) -> None:
        tab = self._tabs.widget(new_idx)
        if isinstance(tab, DfmTab):
            target = min(self._active_subtab_idx, tab._tabs.count() - 1)
            tab._tabs.blockSignals(True)
            tab._tabs.setCurrentIndex(target)
            tab._tabs.blockSignals(False)

    # ------------------------------------------------------------------
    # Chamber ↔ well mapping helpers
    # ------------------------------------------------------------------

    def _wells_for_chamber(self, dfm_id: int, chamber: int) -> list[int]:
        """Return the well numbers that belong to *chamber* in *dfm_id*."""
        cs = self._dfm_chamber_sizes.get(dfm_id, 1)
        start = cs * (chamber - 1) + 1
        return list(range(start, start + cs))

    def _chamber_for_well(self, dfm_id: int, well_num: int) -> int:
        """Return the chamber number that contains *well_num* in *dfm_id*."""
        cs = self._dfm_chamber_sizes.get(dfm_id, 1)
        return (well_num - 1) // cs + 1

    # ------------------------------------------------------------------
    # Bidirectional exclusion sync
    # ------------------------------------------------------------------

    def _on_feeding_exclusion_changed(self, dfm_id: int, chamber: int, excluded: bool) -> None:
        """Feeding Summary checkbox toggled → check all wells for that chamber in the DFM tab."""
        if self._syncing:
            return
        self._syncing = True
        try:
            tab = self._dfm_tab_widgets.get(dfm_id)
            if tab is not None:
                for w in self._wells_for_chamber(dfm_id, chamber):
                    tab.set_well_excluded(w, excluded)
        finally:
            self._syncing = False
        self._autosave_timer.start()

    def _on_dfm_exclusion_changed(self, dfm_id: int, well_num: int, excluded: bool) -> None:
        """DFM tab well checkbox toggled → exclude the parent chamber in Feeding Summary
        and sync any sibling wells in the same chamber."""
        if self._syncing:
            return
        self._syncing = True
        try:
            chamber = self._chamber_for_well(dfm_id, well_num)
            # Sync the Feeding Summary row for this chamber
            if self._feeding_tab is not None:
                self._feeding_tab.set_chamber_excluded(dfm_id, chamber, excluded)
            # Sync sibling wells in the same chamber (e.g. well 2 when well 1 is toggled)
            tab = self._dfm_tab_widgets.get(dfm_id)
            if tab is not None:
                for w in self._wells_for_chamber(dfm_id, chamber):
                    if w != well_num:
                        tab.set_well_excluded(w, excluded)
        finally:
            self._syncing = False
        self._autosave_timer.start()

    # ------------------------------------------------------------------
    # Auto filter
    # ------------------------------------------------------------------

    def _apply_exclusions(self, excluded_by_dfm: dict[int, list[int]]) -> None:
        """
        Silently set the exclusion state for all chambers to match *excluded_by_dfm*
        (a mapping of dfm_id → list of excluded chamber numbers), then trigger
        one auto-save.  No signals are emitted — updates are applied directly.
        """
        # Clear all checkboxes in feeding summary and DFM tabs silently
        if self._feeding_tab is not None and self._feeding_tab._table is not None:
            self._feeding_tab._table.set_all_checked(False)
        for tab in self._dfm_tab_widgets.values():
            for well_num in range(1, 13):
                tab.set_well_excluded(well_num, False)

        # Set excluded chambers
        for dfm_id, chambers in excluded_by_dfm.items():
            for chamber in chambers:
                if self._feeding_tab is not None:
                    self._feeding_tab.set_chamber_excluded(dfm_id, chamber, True)
                tab = self._dfm_tab_widgets.get(dfm_id)
                if tab is not None:
                    for w in self._wells_for_chamber(dfm_id, chamber):
                        tab.set_well_excluded(w, True)

        self._autosave_timer.start()

    def _on_view_filter_criteria(self) -> None:
        """Show the filter criteria summary from the last auto_remove_chambers run."""
        summary = (
            getattr(self._exp, "filter_criteria_summary", "") if self._exp is not None else ""
        )
        if not summary:
            summary = (
                "Auto filter has not been run yet.\n\n"
                "Click 'Auto Filter Chambers' to apply the thresholds defined\n"
                "under global_constants in flic_config.yaml."
            )

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Auto Filter Criteria")
        dlg.resize(520, 320)
        layout = QtWidgets.QVBoxLayout(dlg)

        text = QtWidgets.QPlainTextEdit()
        text.setReadOnly(True)
        font = text.font()
        font.setFamily("Monospace")
        text.setFont(font)
        text.setPlainText(summary)
        layout.addWidget(text)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        dlg.exec()

    def _on_auto_filter(self) -> None:
        """Run ``auto_remove_chambers`` on the loaded experiment and apply results to the UI."""
        if self._exp is None:
            return

        self.statusBar().showMessage("Running auto filter…")
        QtWidgets.QApplication.processEvents()  # let the status bar repaint

        try:
            removed = self._exp.auto_remove_chambers()
        except Exception as exc:
            self.statusBar().showMessage(f"Auto filter failed: {exc}")
            return

        # Build excluded_by_dfm from the returned DataFrame
        excluded_by_dfm: dict[int, list[int]] = {}
        if removed is not None and not removed.empty:
            for _, row in removed.iterrows():
                dfm_id  = int(row["DFM"])
                chamber = int(row["Chamber"])
                excluded_by_dfm.setdefault(dfm_id, []).append(chamber)

        self._apply_exclusions(excluded_by_dfm)

        n = sum(len(v) for v in excluded_by_dfm.values())
        if n == 0:
            constants = (self._exp.global_constants or {})
            if not constants:
                self.statusBar().showMessage(
                    "Auto filter: no global_constants defined in flic_config.yaml — "
                    "no chambers removed."
                )
            else:
                self.statusBar().showMessage("Auto filter: all chambers passed — none excluded.")
        else:
            self.statusBar().showMessage(
                f"Auto filter: {n} chamber(s) excluded and saved to flic_config.yaml."
            )

    # ------------------------------------------------------------------
    # YAML persistence
    # ------------------------------------------------------------------

    def _read_excluded_from_yaml(self) -> dict[int, list[int]]:
        """Read ``excluded_chambers`` entries from ``flic_config.yaml``."""
        config_path = self._project_dir / "flic_config.yaml"
        if not config_path.exists():
            return {}
        try:
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        result: dict[int, list[int]] = {}
        for dfm_node in cfg.get("dfms", []):
            if not isinstance(dfm_node, dict):
                continue
            dfm_id = int(dfm_node.get("id", 0))
            excl   = dfm_node.get("excluded_chambers") or []
            if excl:
                result[dfm_id] = [int(x) for x in excl]
        return result

    def _save_exclusions_to_yaml(self, *, _quiet: bool = False) -> None:
        """Write the current exclusion state back to ``flic_config.yaml``."""
        config_path = self._project_dir / "flic_config.yaml"
        if not config_path.exists():
            self.statusBar().showMessage(f"flic_config.yaml not found in {self._project_dir}")
            return
        try:
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.statusBar().showMessage(f"Could not read YAML: {exc}")
            return

        for dfm_node in cfg.get("dfms", []):
            if not isinstance(dfm_node, dict):
                continue
            dfm_id = int(dfm_node.get("id", 0))
            tab = self._dfm_tab_widgets.get(dfm_id)
            if tab is None:
                continue
            excl_wells = tab.get_excluded_wells()
            # Convert well numbers back to chamber numbers
            cs = self._dfm_chamber_sizes.get(dfm_id, 1)
            excl_chambers = sorted({(w - 1) // cs + 1 for w in excl_wells})
            if excl_chambers:
                dfm_node["excluded_chambers"] = excl_chambers
            else:
                dfm_node.pop("excluded_chambers", None)

        try:
            config_path.write_text(
                yaml.dump(cfg, sort_keys=False, default_flow_style=False),
                encoding="utf-8",
            )
            total_chambers = sum(
                len({(w - 1) // self._dfm_chamber_sizes.get(dfm_id, 1) + 1
                     for w in tab.get_excluded_wells()})
                for dfm_id, tab in self._dfm_tab_widgets.items()
            )
            self.statusBar().showMessage(
                f"Exclusions saved to {config_path.name}  "
                f"({total_chambers} chamber(s) excluded)"
            )
        except Exception as exc:
            self.statusBar().showMessage(f"Could not write YAML: {exc}")

    def closeEvent(self, event) -> None:
        """On close, flush any pending auto-save before exiting."""
        if self._autosave_timer.isActive():
            self._autosave_timer.stop()
            self._save_exclusions_to_yaml()
        super().closeEvent(event)


# ───────────────────────────────────────────────────────────────────────────
# Entry point
# ───────────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) > 2:
        print("Usage: pyflic-qc [project_dir]", file=sys.stderr)
        sys.exit(1)

    project_dir = Path(sys.argv[1] if len(sys.argv) >= 2 else ".").expanduser().resolve()
    if not project_dir.is_dir():
        print(f"Error: not a directory: {project_dir}", file=sys.stderr)
        sys.exit(1)

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("FLIC QC Viewer")

    win = MainWindow(project_dir)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
