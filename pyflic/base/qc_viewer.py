#!/usr/bin/env python3
"""
FLIC QC Viewer
==============
Interactive PyQt dashboard that displays pre-computed QC results from a
project's ``qc/`` directory — no raw data loading required.

Usage
-----
    python base/qc_viewer.py /path/to/project

The script scans ``project_dir/qc/`` for files written by
``Experiment.write_qc_reports()`` and displays them in a tabbed window —
one tab per DFM.

Each DFM tab contains:
  • Integrity       — CSV table + raw text report
  • Data Breaks     — CSV table (or "none detected")
  • Sim. Feeding    — CSV table
  • Bleeding        — Matrix + AllData CSV tables
  • Raw Signal      — DFM{id}_raw.png embedded as an image
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
    from PyQt6.QtCore import Qt
    _PYQT = 6
except ImportError:
    from PyQt5 import QtCore, QtWidgets  # type: ignore[no-redef]
    from PyQt5.QtCore import Qt          # type: ignore[no-redef]
    _PYQT = 5

import pandas as pd


# ───────────────────────────────────────────────────────────────────────────
# Helpers
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
# Per-DFM tab
# ───────────────────────────────────────────────────────────────────────────

class DfmTab(QtWidgets.QWidget):
    """Displays pre-computed QC files for a single DFM."""

    def __init__(self, dfm_id: int, qc_dir: Path, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._build(dfm_id, qc_dir)

    def _build(self, dfm_id: int, qc_dir: Path) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        self._tabs = QtWidgets.QTabWidget()
        outer.addWidget(self._tabs)
        tabs = self._tabs  # local alias for the rest of _build

        prefix = qc_dir / f"DFM{dfm_id}"

        # ── Integrity ──────────────────────────────────────────────────────
        integ_csv = Path(f"{prefix}_integrity_report.csv")
        integ_txt = Path(f"{prefix}_integrity_report.txt")

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

            # Top: CSV table (small — normally one row)
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

            # Bottom: text report (large)
            txt = QtWidgets.QTextEdit()
            txt.setReadOnly(True)
            if integ_txt.exists():
                txt.setPlainText(integ_txt.read_text(encoding="utf-8", errors="replace"))
            else:
                txt.setPlainText("Integrity text report not found.")
            splitter.addWidget(txt)

            splitter.setSizes([100, 330])  # ~30% / ~70%

        tabs.addTab(integ_widget, "Integrity")

        # ── Simultaneous Feeding + Bleeding (two-well only) ────────────────
        # The sim-feeding CSV is only written for two-well DFMs, so its
        # presence is used as the chamber_size=2 indicator.
        sim_path = Path(f"{prefix}_simultaneous_feeding_matrix.csv")
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

            bleed_mat_path = Path(f"{prefix}_bleeding_matrix.csv")
            bleed_all_path = Path(f"{prefix}_bleeding_alldata.csv")
            bleed_widget = QtWidgets.QWidget()
            bleed_layout = QtWidgets.QVBoxLayout(bleed_widget)
            bleed_layout.setContentsMargins(8, 8, 8, 8)
            found_bleed = False
            if bleed_mat_path.exists():
                found_bleed = True
                try:
                    df = pd.read_csv(bleed_mat_path, index_col=0)
                    bleed_layout.addWidget(QtWidgets.QLabel("<b>Bleeding matrix (max cross-well response):</b>"))
                    bleed_layout.addWidget(_DfTableWidget(df))
                except Exception as exc:
                    bleed_layout.addWidget(QtWidgets.QLabel(f"Error reading matrix: {exc}"))
            if bleed_all_path.exists():
                found_bleed = True
                try:
                    df = pd.read_csv(bleed_all_path, index_col=0)
                    bleed_layout.addWidget(QtWidgets.QLabel("<b>All data (mean signal per well):</b>"))
                    bleed_layout.addWidget(_DfTableWidget(df))
                except Exception as exc:
                    bleed_layout.addWidget(QtWidgets.QLabel(f"Error reading alldata: {exc}"))
            if not found_bleed:
                bleed_layout.addWidget(QtWidgets.QLabel("No bleeding data found."))
            bleed_layout.addStretch()
            tabs.addTab(bleed_widget, "Bleeding")

        # ── Signal plots (always shown; message if file absent) ────────────
        for label, suffix in [
            ("Raw Signal",       "_raw.png"),
            ("Baselined",        "_baselined.png"),
            ("Cumulative Licks", "_cumulative_licks.png"),
        ]:
            tabs.addTab(_png_widget(Path(f"{prefix}{suffix}")), label)


# ───────────────────────────────────────────────────────────────────────────
# Main window
# ───────────────────────────────────────────────────────────────────────────

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, project_dir: Path) -> None:
        super().__init__()
        self.setWindowTitle(f"FLIC QC Viewer  —  {project_dir}")
        self.resize(1280, 860)

        qc_dir = project_dir / "qc"

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        self._dfm_tabs = QtWidgets.QTabWidget()
        layout.addWidget(self._dfm_tabs)

        status = self.statusBar()

        if not qc_dir.is_dir():
            status.showMessage(f"QC directory not found: {qc_dir}")
            err = QtWidgets.QLabel(
                f"No QC directory found at:\n{qc_dir}\n\n"
                "Run Experiment.write_qc_reports() first to generate QC output."
            )
            err.setAlignment(
                Qt.AlignmentFlag.AlignCenter if _PYQT == 6 else Qt.AlignCenter  # type: ignore[attr-defined]
            )
            self._dfm_tabs.addTab(err, "Error")
            return

        # Discover DFM IDs from filenames like DFM{id}_*.
        dfm_ids = sorted({
            int(m.group(1))
            for f in qc_dir.iterdir()
            if (m := re.match(r"DFM(\d+)_", f.name))
        })

        if not dfm_ids:
            status.showMessage(f"No DFM files found in {qc_dir}")
            self._dfm_tabs.addTab(
                QtWidgets.QLabel("No DFM QC files found in the qc/ directory."),
                "Error",
            )
            return

        self._active_subtab_idx: int = 0

        for dfm_id in dfm_ids:
            tab = DfmTab(dfm_id, qc_dir)
            self._dfm_tabs.addTab(tab, f"DFM {dfm_id}")
            tab._tabs.currentChanged.connect(self._on_subtab_changed)

        self._dfm_tabs.currentChanged.connect(self._on_dfm_tab_changed)

        status.showMessage(
            f"Loaded {len(dfm_ids)} DFM(s) from {qc_dir}"
        )


    def _on_subtab_changed(self, idx: int) -> None:
        """Remember the subtab the user just switched to."""
        self._active_subtab_idx = idx

    def _on_dfm_tab_changed(self, new_idx: int) -> None:
        """When switching DFM tabs, restore the previously active subtab."""
        tab = self._dfm_tabs.widget(new_idx)
        if isinstance(tab, DfmTab):
            target = min(self._active_subtab_idx, tab._tabs.count() - 1)
            tab._tabs.blockSignals(True)
            tab._tabs.setCurrentIndex(target)
            tab._tabs.blockSignals(False)


# ───────────────────────────────────────────────────────────────────────────
# Entry point
# ───────────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) > 2:
        print("Usage: pyflic-qc [project_dir]", file=sys.stderr)
        sys.exit(1)

    project_dir = Path(sys.argv[1] if len(sys.argv) == 2 else ".").expanduser().resolve()
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
