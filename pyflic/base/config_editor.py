"""
FLIC Config Editor
==================
PyQt6 GUI for creating and editing ``flic_config.yaml`` experiment configuration files.

Usage (command line)::

    python -m pyflic

Usage (Python)::

    from pyflic.base.config_editor import launch
    launch()
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import yaml
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Parameter metadata
# ---------------------------------------------------------------------------

_PARAM_DEFAULTS: dict[int, dict[str, Any]] = {
    1: {  # single_well — mirrors Parameters.single_well()
        "baseline_window_minutes": 3,
        "feeding_threshold": 20,
        "feeding_minimum": 10,
        "tasting_minimum": 5,
        "tasting_maximum": 20,
        "feeding_minevents": 1,
        "tasting_minevents": 1,
        "samples_per_second": 5,
        "feeding_event_link_gap": 5,
        "pi_direction": "left",
        "correct_for_dual_feeding": False,
    },
    2: {  # two_well — mirrors Parameters.two_well()
        "baseline_window_minutes": 3,
        "feeding_threshold": 20,
        "feeding_minimum": 10,
        "tasting_minimum": 5,
        "tasting_maximum": 20,
        "feeding_minevents": 1,
        "tasting_minevents": 1,
        "samples_per_second": 5,
        "feeding_event_link_gap": 5,
        "pi_direction": "left",
        "correct_for_dual_feeding": True,
    },
}

_PARAM_LABELS: dict[str, str] = {
    "baseline_window_minutes": "Baseline Window (min)",
    "feeding_threshold": "Feeding Threshold",
    "feeding_minimum": "Feeding Minimum",
    "tasting_minimum": "Tasting Minimum",
    "tasting_maximum": "Tasting Maximum",
    "feeding_minevents": "Feeding Min Events",
    "tasting_minevents": "Tasting Min Events",
    "samples_per_second": "Samples / Second",
    "feeding_event_link_gap": "Event Link Gap (samples)",
    "pi_direction": "PI Direction (side with PI = 1)",
    "correct_for_dual_feeding": "Correct for Dual Feeding",
}

_PARAM_ORDER: list[str] = [
    "baseline_window_minutes",
    "feeding_threshold",
    "feeding_minimum",
    "tasting_minimum",
    "tasting_maximum",
    "feeding_minevents",
    "tasting_minevents",
    "samples_per_second",
    "feeding_event_link_gap",
    "pi_direction",
    "correct_for_dual_feeding",
]

# ---------------------------------------------------------------------------
# Widget helpers
# ---------------------------------------------------------------------------


def _make_param_widget(key: str, default: Any) -> QWidget:
    """Return an appropriate input widget for the given parameter key."""
    if key == "pi_direction":
        w = QComboBox()
        w.addItems(["left", "right"])
        w.setCurrentText(str(default))
        w.setFixedWidth(75)
        return w
    if key == "correct_for_dual_feeding":
        w = QCheckBox()
        w.setChecked(bool(default))
        return w
    w = QSpinBox()
    w.setFixedWidth(75)
    if key == "samples_per_second":
        w.setRange(1, 1000)
    elif key == "baseline_window_minutes":
        w.setRange(1, 60)
    elif key in ("feeding_threshold", "feeding_minimum", "tasting_minimum", "tasting_maximum"):
        w.setRange(0, 100000)
    else:
        w.setRange(0, 10000)
    w.setValue(int(default))
    return w


def _get_param_value(widget: QWidget) -> Any:
    if isinstance(widget, QComboBox):
        return widget.currentText()
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    if isinstance(widget, QSpinBox):
        return widget.value()
    return None


def _set_param_value(widget: QWidget, value: Any) -> None:
    if isinstance(widget, QComboBox):
        idx = widget.findText(str(value))
        if idx >= 0:
            widget.setCurrentIndex(idx)
    elif isinstance(widget, QCheckBox):
        widget.setChecked(bool(value))
    elif isinstance(widget, QSpinBox):
        widget.setValue(int(round(float(value))))


# ---------------------------------------------------------------------------
# ParamsForm
# ---------------------------------------------------------------------------


class ParamsForm(QWidget):
    """
    A QFormLayout-based widget for all non-chamber_size Parameters fields.

    In *global* mode (``override_mode=False``): every field is shown enabled
    and always included in ``get_values()``.

    In *override* mode (``override_mode=True``): each row has an enable
    checkbox; only checked rows are returned by ``get_values()``.
    """

    def __init__(
        self,
        *,
        override_mode: bool = False,
        chamber_size: int = 2,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._override_mode = override_mode
        self._input_widgets: dict[str, QWidget] = {}
        self._enable_checks: dict[str, QCheckBox] = {}
        self._two_well_rows: list[tuple[QFormLayout, int]] = []

        defaults = _PARAM_DEFAULTS.get(chamber_size, _PARAM_DEFAULTS[2])

        n = len(_PARAM_ORDER)
        s1 = (n + 2) // 3
        columns = [_PARAM_ORDER[:s1], _PARAM_ORDER[s1 : 2 * s1], _PARAM_ORDER[2 * s1 :]]

        outer = QHBoxLayout(self)
        outer.setSpacing(20)

        for col_keys in columns:
            form = QFormLayout()
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
            form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
            form.setHorizontalSpacing(8)
            form.setVerticalSpacing(2)

            for key in col_keys:
                label = _PARAM_LABELS[key]
                widget = _make_param_widget(key, defaults[key])
                self._input_widgets[key] = widget

                row_idx = form.rowCount()
                if key in ("pi_direction", "correct_for_dual_feeding"):
                    self._two_well_rows.append((form, row_idx))

                if override_mode:
                    cb = QCheckBox()
                    cb.setChecked(False)
                    cb.toggled.connect(lambda checked, w=widget: w.setEnabled(checked))
                    widget.setEnabled(False)
                    self._enable_checks[key] = cb

                    row = QWidget()
                    rl = QHBoxLayout(row)
                    rl.setContentsMargins(0, 0, 0, 0)
                    rl.setSpacing(4)
                    rl.addWidget(cb)
                    rl.addWidget(widget)
                    rl.addStretch()
                    form.addRow(label, row)
                else:
                    form.addRow(label, widget)

            outer.addLayout(form, stretch=1)

        self.set_chamber_size(chamber_size)

    def get_values(self, *, include_disabled: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, w in self._input_widgets.items():
            if self._override_mode and not include_disabled:
                cb = self._enable_checks.get(key)
                if cb is not None and not cb.isChecked():
                    continue
            out[key] = _get_param_value(w)
        return out

    def load_values(self, values: dict[str, Any], chamber_size: int = 2) -> None:
        defaults = _PARAM_DEFAULTS.get(chamber_size, _PARAM_DEFAULTS[2])
        for key in _PARAM_ORDER:
            w = self._input_widgets.get(key)
            if w is None:
                continue
            cb = self._enable_checks.get(key)
            if key in values:
                _set_param_value(w, values[key])
                if cb is not None:
                    cb.setChecked(True)
                    w.setEnabled(True)
            else:
                _set_param_value(w, defaults[key])
                if cb is not None:
                    cb.setChecked(False)
                    w.setEnabled(False)

    def reset_defaults(self, chamber_size: int) -> None:
        defaults = _PARAM_DEFAULTS.get(chamber_size, _PARAM_DEFAULTS[2])
        for key in _PARAM_ORDER:
            w = self._input_widgets.get(key)
            if w is not None:
                _set_param_value(w, defaults[key])
            cb = self._enable_checks.get(key)
            if cb is not None:
                cb.setChecked(False)
                w.setEnabled(False)

    def set_chamber_size(self, chamber_size: int) -> None:
        show = chamber_size == 2
        for form, row_idx in self._two_well_rows:
            form.setRowVisible(row_idx, show)


# ---------------------------------------------------------------------------
# FactorsWidget
# ---------------------------------------------------------------------------


class FactorsWidget(QWidget):
    """Editable table for defining experimental design factors and their levels."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(4)

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Factor Name", "Levels (comma-separated)"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.setMinimumHeight(80)
        layout.addWidget(self._table, stretch=1)

        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(6)
        add_btn = QPushButton("+ Factor")
        add_btn.setMaximumWidth(90)
        add_btn.clicked.connect(self._add_row)
        remove_btn = QPushButton("− Remove")
        remove_btn.setMaximumWidth(90)
        remove_btn.clicked.connect(self._remove_row)
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(remove_btn)
        btn_layout.addStretch()
        layout.addWidget(btn_row)

    def _add_row(self) -> None:
        r = self._table.rowCount()
        self._table.setRowCount(r + 1)
        self._table.setItem(r, 0, QTableWidgetItem(""))
        self._table.setItem(r, 1, QTableWidgetItem(""))
        self._table.editItem(self._table.item(r, 0))

    def _remove_row(self) -> None:
        row = self._table.currentRow()
        if row >= 0:
            self._table.removeRow(row)

    def get_factors(self) -> dict[str, list[str]]:
        """Return {factor_name: [level, ...]} for all non-empty rows."""
        result: dict[str, list[str]] = {}
        for r in range(self._table.rowCount()):
            name_item = self._table.item(r, 0)
            levels_item = self._table.item(r, 1)
            name = name_item.text().strip() if name_item else ""
            levels_raw = levels_item.text().strip() if levels_item else ""
            if name:
                result[name] = [lv.strip() for lv in levels_raw.split(",") if lv.strip()]
        return result

    def get_factor_names(self) -> list[str]:
        return list(self.get_factors().keys())

    def load_factors(self, factors: dict) -> None:
        self._table.setRowCount(0)
        for name, levels in factors.items():
            r = self._table.rowCount()
            self._table.setRowCount(r + 1)
            self._table.setItem(r, 0, QTableWidgetItem(str(name)))
            if isinstance(levels, list):
                self._table.setItem(r, 1, QTableWidgetItem(", ".join(str(lv) for lv in levels)))
            else:
                self._table.setItem(r, 1, QTableWidgetItem(str(levels)))


# ---------------------------------------------------------------------------
# DFMWidget
# ---------------------------------------------------------------------------


def _sanitize_treatment(text: str) -> str:
    """Spaces → underscores; strip everything that isn't alphanumeric or underscore."""
    return re.sub(r"[^A-Za-z0-9_]", "", text.replace(" ", "_"))


def _chamber_table_min_height(n_chambers: int) -> int:
    return n_chambers * 26 + 32


def _build_chamber_table(n_chambers: int) -> QTableWidget:
    """Build a bare chamber table (Chamber + Treatment columns, no factor logic)."""
    table = QTableWidget(n_chambers, 2)
    table.setHorizontalHeaderLabels(["Chamber", "Treatment"])
    table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
    table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
    table.verticalHeader().setVisible(False)
    table.setAlternatingRowColors(True)
    table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    table.setMinimumHeight(_chamber_table_min_height(n_chambers))
    for i in range(n_chambers):
        ch_item = QTableWidgetItem(str(i + 1))
        ch_item.setFlags(ch_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        table.setItem(i, 0, ch_item)
        table.setItem(i, 1, QTableWidgetItem(""))
    return table


class DFMWidget(QWidget):
    """Configuration widget for a single DFM (one tab in the DFM tab widget)."""

    def __init__(
        self,
        dfm_id: int,
        chamber_size: int = 2,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._chamber_size = chamber_size
        self._factor_levels: dict[str, list[str]] = {}

        outer = QVBoxLayout(self)
        outer.setAlignment(Qt.AlignmentFlag.AlignTop)

        # -- DFM ID --------------------------------------------------------
        id_group = QGroupBox("DFM Identity")
        id_layout = QHBoxLayout(id_group)
        id_layout.addWidget(QLabel("DFM ID:"))
        self._id_spin = QSpinBox()
        self._id_spin.setRange(1, 99)
        self._id_spin.setValue(dfm_id)
        self._id_spin.setMaximumWidth(80)
        id_layout.addWidget(self._id_spin)
        id_layout.addStretch()
        outer.addWidget(id_group)

        # -- Chamber table -------------------------------------------------
        n_chambers = 12 if chamber_size == 1 else 6
        self._ch_group = QGroupBox("Chamber → Treatment Assignments")
        ch_layout = QVBoxLayout(self._ch_group)
        self._ch_hint = QLabel(
            "Leave a Treatment cell blank to omit that chamber from the config."
        )
        self._ch_hint.setStyleSheet("color: gray; font-size: 11px;")
        ch_layout.addWidget(self._ch_hint)
        self._chamber_table = _build_chamber_table(n_chambers)
        self._chamber_table.itemChanged.connect(self._on_chamber_cell_changed)
        ch_layout.addWidget(self._chamber_table)
        outer.addWidget(self._ch_group)

        # -- Parameter overrides -------------------------------------------
        over_group = QGroupBox(
            "Parameter Overrides  (check a box to override the global value)"
        )
        over_layout = QVBoxLayout(over_group)
        self._params_form = ParamsForm(override_mode=True, chamber_size=chamber_size)
        over_layout.addWidget(self._params_form)
        outer.addWidget(over_group)

    # ------------------------------------------------------------------

    def _on_chamber_cell_changed(self, item: QTableWidgetItem) -> None:
        col = item.column()
        if col == 0:
            return
        raw = item.text()
        clean = _sanitize_treatment(raw)
        if clean != raw:
            self._chamber_table.blockSignals(True)
            item.setText(clean)
            self._chamber_table.blockSignals(False)
            raw = clean
        self._validate_chamber_item(item, col)

    def _validate_chamber_item(self, item: QTableWidgetItem, col: int) -> None:
        """Validate a factor-column cell against its factor's allowed levels."""
        factor_names = list(self._factor_levels.keys())
        factor_col = col - 1  # 0-based factor index
        if factor_names and 0 <= factor_col < len(factor_names):
            fname = factor_names[factor_col]
            allowed = self._factor_levels.get(fname, [])
            text = item.text().strip()
            if text and allowed and text not in allowed:
                item.setBackground(QColor("#ffcccc"))
                item.setToolTip(f"'{text}' is not a valid level for '{fname}'. Allowed: {allowed}")
                return
        item.setData(Qt.ItemDataRole.BackgroundRole, None)
        item.setToolTip("")

    def update_chamber_size(self, chamber_size: int) -> None:
        self._chamber_size = chamber_size
        n_new = 12 if chamber_size == 1 else 6
        n_old = self._chamber_table.rowCount()
        n_cols = self._chamber_table.columnCount()

        # Preserve existing row data across all factor columns
        old_data: dict[int, list[str]] = {}
        for i in range(n_old):
            row = []
            for c in range(1, n_cols):
                it = self._chamber_table.item(i, c)
                row.append(it.text() if it else "")
            old_data[i + 1] = row

        self._chamber_table.blockSignals(True)
        self._chamber_table.setRowCount(n_new)
        self._chamber_table.setMinimumHeight(_chamber_table_min_height(n_new))
        for i in range(n_new):
            ch_num = i + 1
            ch_item = QTableWidgetItem(str(ch_num))
            ch_item.setFlags(ch_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._chamber_table.setItem(i, 0, ch_item)
            saved = old_data.get(ch_num, [""] * (n_cols - 1))
            for c in range(1, n_cols):
                v = saved[c - 1] if c - 1 < len(saved) else ""
                self._chamber_table.setItem(i, c, QTableWidgetItem(v))
        self._chamber_table.blockSignals(False)

        self._params_form.set_chamber_size(chamber_size)

    def update_factors(self, factor_levels: dict[str, list[str]]) -> None:
        """Restructure the chamber table to show one column per factor."""
        self._factor_levels = factor_levels
        factor_names = list(factor_levels.keys())
        n_factors = len(factor_names)

        # Snapshot existing data: join all current factor cols into a list per row
        n_old_cols = self._chamber_table.columnCount()
        old_data: dict[int, list[str]] = {}
        for i in range(self._chamber_table.rowCount()):
            parts = []
            for c in range(1, n_old_cols):
                it = self._chamber_table.item(i, c)
                parts.append(it.text().strip() if it else "")
            old_data[i] = parts  # list of per-factor values (may be shorter/longer than new)

        # Determine new layout
        n_new_cols = 1 + max(1, n_factors)  # Chamber col + factor cols (or Treatment)

        self._chamber_table.blockSignals(True)
        self._chamber_table.setColumnCount(n_new_cols)

        if factor_names:
            headers = ["Chamber"] + factor_names
            self._ch_group.setTitle("Chamber → Factor Level Assignments")
            self._ch_hint.setText(
                f"Enter one level per column in the order: {', '.join(factor_names)}.  Leave blank to omit."
            )
        else:
            headers = ["Chamber", "Treatment"]
            self._ch_group.setTitle("Chamber → Treatment Assignments")
            self._ch_hint.setText("Leave a Treatment cell blank to omit that chamber from the config.")

        self._chamber_table.setHorizontalHeaderLabels(headers)
        self._chamber_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for c in range(1, n_new_cols):
            self._chamber_table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeMode.Stretch)

        # Re-populate rows, distributing old values into new columns
        for i in range(self._chamber_table.rowCount()):
            ch_item = QTableWidgetItem(str(i + 1))
            ch_item.setFlags(ch_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._chamber_table.setItem(i, 0, ch_item)
            prev = old_data.get(i, [])
            for c in range(1, n_new_cols):
                val = prev[c - 1] if c - 1 < len(prev) else ""
                self._chamber_table.setItem(i, c, QTableWidgetItem(val))

        self._chamber_table.blockSignals(False)

        # Revalidate all data cells
        for i in range(self._chamber_table.rowCount()):
            for c in range(1, n_new_cols):
                it = self._chamber_table.item(i, c)
                if it is not None:
                    self._validate_chamber_item(it, c)

    def get_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"id": self._id_spin.value()}

        overrides = self._params_form.get_values()
        if overrides:
            result["params"] = overrides

        n_cols = self._chamber_table.columnCount()
        chambers: dict[int, str] = {}
        for i in range(self._chamber_table.rowCount()):
            if n_cols == 2:
                it = self._chamber_table.item(i, 1)
                val = it.text().strip() if it else ""
            else:
                parts = []
                for c in range(1, n_cols):
                    it = self._chamber_table.item(i, c)
                    parts.append(it.text().strip() if it else "")
                val = ", ".join(p for p in parts if p)
            if val:
                chambers[i + 1] = val
        if chambers:
            result["chambers"] = chambers

        return result

    def load_dict(self, data: dict[str, Any], chamber_size: int) -> None:
        self._id_spin.setValue(int(data.get("id", self._id_spin.value())))

        params_raw = data.get("params", data.get("parameters", {})) or {}
        self._params_form.load_values(dict(params_raw), chamber_size)

        chambers_raw = data.get("chambers", data.get("Chambers", {})) or {}
        if isinstance(chambers_raw, dict):
            assignments = {int(k): str(v) for k, v in chambers_raw.items()}
        elif isinstance(chambers_raw, list):
            assignments = {int(it["index"]): str(it.get("treatment", it.get("levels", ""))) for it in chambers_raw}
        else:
            assignments = {}

        n_cols = self._chamber_table.columnCount()
        for i in range(self._chamber_table.rowCount()):
            val = assignments.get(i + 1, "")
            if n_cols == 2:
                self._chamber_table.setItem(i, 1, QTableWidgetItem(val))
            else:
                parts = [p.strip() for p in val.split(",")]
                for c in range(1, n_cols):
                    v = parts[c - 1] if c - 1 < len(parts) else ""
                    self._chamber_table.setItem(i, c, QTableWidgetItem(v))


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class FLICConfigEditor(QMainWindow):
    """Main window for the FLIC Config Editor."""

    def __init__(self) -> None:
        super().__init__()
        self._current_path: Path | None = None
        self._dfm_widgets: list[DFMWidget] = []

        self.setWindowTitle("FLIC Config Editor")
        self.resize(960, 1020)

        self._build_menu()
        self._build_ui()
        self._auto_load()

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")

        new_act = QAction("&New", self)
        new_act.setShortcut("Ctrl+N")
        new_act.triggered.connect(self._new)
        file_menu.addAction(new_act)

        open_act = QAction("&Open…", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self._open)
        file_menu.addAction(open_act)

        file_menu.addSeparator()

        save_act = QAction("&Save", self)
        save_act.setShortcut("Ctrl+S")
        save_act.triggered.connect(self._save)
        file_menu.addAction(save_act)

        saveas_act = QAction("Save &As…", self)
        saveas_act.setShortcut("Ctrl+Shift+S")
        saveas_act.triggered.connect(self._save_as)
        file_menu.addAction(saveas_act)

        file_menu.addSeparator()

        exit_act = QAction("E&xit", self)
        exit_act.setShortcut("Ctrl+Q")
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.setCentralWidget(splitter)

        # ---- Top pane ---------------------------------------------------
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        top_layout.setSpacing(8)
        top_layout.setContentsMargins(6, 6, 6, 6)

        # Experiment Settings
        exp_group = QGroupBox("Experiment Settings")
        self._exp_form = QFormLayout(exp_group)
        self._exp_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self._exp_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        dir_row = QWidget()
        dir_layout = QHBoxLayout(dir_row)
        dir_layout.setContentsMargins(0, 0, 0, 0)
        self._data_dir_edit = QLineEdit()
        self._data_dir_edit.setPlaceholderText("e.g. ./flic  (relative to config file)")
        browse_btn = QPushButton("Browse…")
        browse_btn.setMaximumWidth(90)
        browse_btn.clicked.connect(self._browse_data_dir)
        dir_layout.addWidget(self._data_dir_edit, stretch=1)
        dir_layout.addWidget(browse_btn)
        self._exp_form.addRow("Data Directory:", dir_row)

        self._chamber_size_combo = QComboBox()
        self._chamber_size_combo.addItems(["1  (single-well, 12 chambers)", "2  (two-well, 6 chambers)"])
        self._chamber_size_combo.setCurrentIndex(1)
        self._chamber_size_combo.setMaximumWidth(240)
        self._chamber_size_combo.currentIndexChanged.connect(self._on_chamber_size_changed)
        self._exp_form.addRow("Chamber Size:", self._chamber_size_combo)

        self._num_dfms_spin = QSpinBox()
        self._num_dfms_spin.setRange(1, 20)
        self._num_dfms_spin.setValue(1)
        self._num_dfms_spin.setMaximumWidth(80)
        self._num_dfms_spin.valueChanged.connect(self._on_num_dfms_changed)
        self._exp_form.addRow("Number of DFMs:", self._num_dfms_spin)

        # Well Names (two-well only) — inline in experiment settings
        self._well_a_edit = QLineEdit()
        self._well_a_edit.setPlaceholderText("e.g. Sucrose")
        self._well_a_edit.setMaximumWidth(200)
        self._well_b_edit = QLineEdit()
        self._well_b_edit.setPlaceholderText("e.g. Yeast")
        self._well_b_edit.setMaximumWidth(200)
        self._well_a_row = self._exp_form.rowCount()
        self._exp_form.addRow("Well A:", self._well_a_edit)
        self._well_b_row = self._exp_form.rowCount()
        self._exp_form.addRow("Well B:", self._well_b_edit)

        top_layout.addWidget(exp_group)

        # Global Parameters
        global_group = QGroupBox("Global Parameters  (applied to all DFMs unless overridden)")
        global_inner = QVBoxLayout(global_group)
        global_inner.setContentsMargins(8, 8, 8, 8)
        self._global_params = ParamsForm(override_mode=False, chamber_size=2)
        global_inner.addWidget(self._global_params)
        top_layout.addWidget(global_group)

        # Experimental Design Factors
        factors_group = QGroupBox("Experimental Design Factors  (optional)")
        factors_inner = QVBoxLayout(factors_group)
        factors_inner.setContentsMargins(8, 4, 8, 8)
        hint = QLabel(
            "Define factors here. Chamber assignments will use comma-separated level values "
            "in the same order as the factors listed below. Leave empty for simple treatment names."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray; font-size: 11px;")
        factors_inner.addWidget(hint, stretch=1)
        self._factors_widget = FactorsWidget()
        factors_inner.addWidget(self._factors_widget, stretch=4)
        top_layout.addWidget(factors_group)

        # Wire factor table changes → update DFM chamber column headers
        self._factors_widget._table.itemChanged.connect(self._on_factors_changed)
        self._factors_widget._table.model().rowsInserted.connect(self._on_factors_changed)
        self._factors_widget._table.model().rowsRemoved.connect(self._on_factors_changed)

        splitter.addWidget(top_widget)

        # ---- Bottom pane: DFM Tabs --------------------------------------
        dfm_group = QGroupBox("DFM Configuration")
        dfm_group_layout = QVBoxLayout(dfm_group)
        dfm_group_layout.setContentsMargins(6, 6, 6, 6)
        self._dfm_tabs = QTabWidget()
        dfm_group_layout.addWidget(self._dfm_tabs)
        splitter.addWidget(dfm_group)

        splitter.setSizes([440, 580])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self._sync_dfm_tabs(1, 2)
        self._update_well_names_visibility()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _chamber_size(self) -> int:
        return int(self._chamber_size_combo.currentText().split()[0])

    def _on_chamber_size_changed(self) -> None:
        cs = self._chamber_size()
        self._global_params.reset_defaults(cs)
        self._global_params.set_chamber_size(cs)
        for w in self._dfm_widgets:
            w.update_chamber_size(cs)
        self._update_well_names_visibility()

    def _update_well_names_visibility(self) -> None:
        show = self._chamber_size() == 2
        self._exp_form.setRowVisible(self._well_a_row, show)
        self._exp_form.setRowVisible(self._well_b_row, show)

    def _on_num_dfms_changed(self, n: int) -> None:
        self._sync_dfm_tabs(n, self._chamber_size())

    def _on_factors_changed(self, *_args) -> None:
        factors = self._factors_widget.get_factors()
        for w in self._dfm_widgets:
            w.update_factors(factors)

    def _on_dfm_id_changed(self, changed_widget: DFMWidget, new_id: int) -> None:
        """Update tab label and resolve ID conflicts when a DFM ID spinner changes."""
        try:
            changed_idx = self._dfm_widgets.index(changed_widget)
        except ValueError:
            return
        self._dfm_tabs.setTabText(changed_idx, f"DFM {new_id}")
        # Resolve conflict: if another widget has the same ID, reassign it the
        # lowest positive integer not currently used by any widget.
        for i, w in enumerate(self._dfm_widgets):
            if i == changed_idx:
                continue
            if w._id_spin.value() == new_id:
                used = {ow._id_spin.value() for ow in self._dfm_widgets if ow is not w}
                free = next(n for n in range(1, 200) if n not in used)
                w._id_spin.blockSignals(True)
                w._id_spin.setValue(free)
                w._id_spin.blockSignals(False)
                self._dfm_tabs.setTabText(i, f"DFM {free}")
                break

    def _sync_dfm_tabs(self, n: int, chamber_size: int) -> None:
        current = len(self._dfm_widgets)
        if n < current:
            for _ in range(current - n):
                self._dfm_tabs.removeTab(self._dfm_tabs.count() - 1)
                w = self._dfm_widgets.pop()
                w.deleteLater()
        elif n > current:
            factors = self._factors_widget.get_factors() if hasattr(self, "_factors_widget") else {}
            for i in range(current, n):
                dfm_id = i + 1
                w = DFMWidget(dfm_id=dfm_id, chamber_size=chamber_size)
                w.update_factors(factors)
                w._id_spin.valueChanged.connect(lambda val, _w=w: self._on_dfm_id_changed(_w, val))
                self._dfm_widgets.append(w)
                tab_scroll = QScrollArea()
                tab_scroll.setWidgetResizable(True)
                tab_scroll.setWidget(w)
                self._dfm_tabs.addTab(tab_scroll, f"DFM {dfm_id}")

    def _auto_load(self) -> None:
        """Load flic_config.yaml from the cwd automatically on startup, if present."""
        for name in ("flic_config.yaml", "flic_config.yml"):
            candidate = Path.cwd() / name
            if candidate.exists():
                try:
                    cfg = yaml.safe_load(candidate.read_text(encoding="utf-8"))
                    if isinstance(cfg, dict):
                        self._current_path = candidate
                        self.setWindowTitle(f"FLIC Config Editor — {candidate.name}")
                        self._populate_from_yaml(cfg)
                except Exception:
                    pass  # silently ignore; user can open manually
                break

    def _browse_data_dir(self) -> None:
        start = str(Path.cwd())
        d = QFileDialog.getExistingDirectory(self, "Select Data Directory", start)
        if d:
            self._data_dir_edit.setText(d)

    # ------------------------------------------------------------------
    # YAML serialisation / deserialisation
    # ------------------------------------------------------------------

    def _collect_yaml(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {}

        data_dir = self._data_dir_edit.text().strip()
        if data_dir:
            cfg["data_dir"] = data_dir

        global_params = self._global_params.get_values()
        global_params["chamber_size"] = self._chamber_size()
        global_section: dict[str, Any] = {"params": global_params}

        # Well names
        if self._chamber_size() == 2:
            wa = self._well_a_edit.text().strip()
            wb = self._well_b_edit.text().strip()
            if wa or wb:
                global_section["well_names"] = {
                    **({"A": wa} if wa else {}),
                    **({"B": wb} if wb else {}),
                }

        # Experimental design factors
        factors = self._factors_widget.get_factors()
        if factors:
            global_section["experimental_design_factors"] = factors

        cfg["global"] = global_section
        cfg["dfms"] = [w.get_dict() for w in self._dfm_widgets]
        return cfg

    def _populate_from_yaml(self, cfg: dict[str, Any]) -> None:
        self._data_dir_edit.setText(str(cfg.get("data_dir", "")))

        global_cfg = cfg.get("global", {}) or {}
        global_params_raw = global_cfg.get("params", global_cfg.get("parameters", {})) or {}
        chamber_size = int(global_params_raw.get("chamber_size", 2))

        idx = 0 if chamber_size == 1 else 1
        self._chamber_size_combo.blockSignals(True)
        self._chamber_size_combo.setCurrentIndex(idx)
        self._chamber_size_combo.blockSignals(False)
        self._update_well_names_visibility()

        params_to_load = {k: v for k, v in global_params_raw.items() if k != "chamber_size"}
        self._global_params.load_values(params_to_load, chamber_size)
        self._global_params.set_chamber_size(chamber_size)

        # Well names
        well_names = global_cfg.get("well_names") or {}
        self._well_a_edit.setText(str(well_names.get("A", "")))
        self._well_b_edit.setText(str(well_names.get("B", "")))

        # Experimental design factors
        factors_node = global_cfg.get("experimental_design_factors") or {}
        self._factors_widget._table.blockSignals(True)
        self._factors_widget.load_factors(factors_node)
        self._factors_widget._table.blockSignals(False)
        factors = self._factors_widget.get_factors()

        # DFM nodes
        dfm_nodes = cfg.get("dfms", cfg.get("DFMs", [])) or []
        if isinstance(dfm_nodes, dict):
            items: list[dict] = []
            for k, v in dfm_nodes.items():
                node = dict(v)
                node.setdefault("id", int(k))
                items.append(node)
            dfm_nodes = items

        n = max(1, len(dfm_nodes))
        self._num_dfms_spin.blockSignals(True)
        self._num_dfms_spin.setValue(n)
        self._num_dfms_spin.blockSignals(False)
        self._sync_dfm_tabs(n, chamber_size)

        for i, node in enumerate(dfm_nodes):
            if i < len(self._dfm_widgets):
                self._dfm_widgets[i].update_factors(factors)
                self._dfm_widgets[i]._id_spin.blockSignals(True)
                self._dfm_widgets[i].load_dict(node, chamber_size)
                self._dfm_widgets[i]._id_spin.blockSignals(False)
                dfm_id = int(node.get("id", i + 1))
                self._dfm_tabs.setTabText(i, f"DFM {dfm_id}")

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def _new(self) -> None:
        self._current_path = None
        self.setWindowTitle("FLIC Config Editor")
        self._data_dir_edit.clear()
        self._well_a_edit.clear()
        self._well_b_edit.clear()

        self._chamber_size_combo.blockSignals(True)
        self._num_dfms_spin.blockSignals(True)
        self._chamber_size_combo.setCurrentIndex(1)
        self._num_dfms_spin.setValue(1)
        self._chamber_size_combo.blockSignals(False)
        self._num_dfms_spin.blockSignals(False)

        self._factors_widget._table.blockSignals(True)
        self._factors_widget._table.setRowCount(0)
        self._factors_widget._table.blockSignals(False)

        self._dfm_tabs.clear()
        for w in self._dfm_widgets:
            w.deleteLater()
        self._dfm_widgets.clear()

        self._global_params.reset_defaults(2)
        self._update_well_names_visibility()
        self._sync_dfm_tabs(1, 2)

    def _open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Config",
            str(Path.cwd()),
            "YAML files (*.yaml *.yml);;All files (*)",
        )
        if not path:
            return
        try:
            cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
            if not isinstance(cfg, dict):
                raise ValueError("File does not contain a YAML mapping.")
            self._current_path = Path(path)
            self.setWindowTitle(f"FLIC Config Editor — {self._current_path.name}")
            self._populate_from_yaml(cfg)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to load config:\n{exc}")

    def _save(self) -> None:
        if self._current_path is None:
            self._save_as()
        else:
            self._write_yaml(self._current_path)

    def _save_as(self) -> None:
        default = str(self._current_path or (Path.cwd() / "flic_config.yaml"))
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Config As",
            default,
            "YAML files (*.yaml *.yml);;All files (*)",
        )
        if not path:
            return
        self._current_path = Path(path)
        self.setWindowTitle(f"FLIC Config Editor — {self._current_path.name}")
        self._write_yaml(self._current_path)

    def _write_yaml(self, path: Path) -> None:
        try:
            cfg = self._collect_yaml()
            path.write_text(
                yaml.dump(cfg, default_flow_style=False, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            QMessageBox.information(self, "Saved", f"Config saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to save config:\n{exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def launch() -> None:
    """Launch the FLIC Config Editor GUI."""
    app = QApplication.instance() or QApplication(sys.argv)
    win = FLICConfigEditor()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch()
