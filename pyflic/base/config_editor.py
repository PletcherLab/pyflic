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
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QAction, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
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
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .ui import ActionButton, Card, Category, TopBar, apply_theme, icon, resolved_mode
from .ui import settings as ui_settings

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
        add_btn = ActionButton("Add Factor", Category.LOAD, icon_name="new")
        add_btn.setMaximumWidth(130)
        add_btn.clicked.connect(self._add_row)
        remove_btn = ActionButton("Remove", Category.QC, icon_name="clear")
        remove_btn.setMaximumWidth(110)
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
        outer.setSpacing(8)
        outer.setContentsMargins(8, 8, 8, 8)

        # -- DFM ID --------------------------------------------------------
        id_card = Card("DFM Identity", Category.NEUTRAL)
        id_inner = QHBoxLayout()
        id_inner.setContentsMargins(0, 0, 0, 0)
        id_inner.addWidget(QLabel("DFM ID:"))
        self._id_spin = QSpinBox()
        self._id_spin.setRange(1, 99)
        self._id_spin.setValue(dfm_id)
        self._id_spin.setMaximumWidth(80)
        id_inner.addWidget(self._id_spin)
        id_inner.addStretch()
        id_card.add_body(id_inner)
        outer.addWidget(id_card)

        # -- Chamber table -------------------------------------------------
        n_chambers = 12 if chamber_size == 1 else 6
        self._ch_card = Card("Chamber → Treatment Assignments", Category.LOAD)
        self._ch_hint = QLabel(
            "Leave a Treatment cell blank to omit that chamber from the config."
        )
        self._ch_hint.setObjectName("PyflicCardSubtitle")
        self._ch_hint.setWordWrap(True)
        self._ch_card.add_body(self._ch_hint)
        self._chamber_table = _build_chamber_table(n_chambers)
        self._chamber_table.itemChanged.connect(self._on_chamber_cell_changed)
        self._ch_card.add_body(self._chamber_table)
        outer.addWidget(self._ch_card)

        # -- Parameter overrides -------------------------------------------
        over_card = Card(
            "Parameter Overrides",
            Category.ANALYZE,
            subtitle="Check a box to override the global value for this DFM.",
        )
        self._params_form = ParamsForm(override_mode=True, chamber_size=chamber_size)
        over_card.add_body(self._params_form)
        outer.addWidget(over_card)

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
            self._ch_card.set_title("Chamber → Factor Level Assignments")
            self._ch_hint.setText(
                f"Enter one level per column in the order: {', '.join(factor_names)}.  Leave blank to omit."
            )
        else:
            headers = ["Chamber", "Treatment"]
            self._ch_card.set_title("Chamber → Treatment Assignments")
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

    def __init__(self, initial_path: str | Path | None = None) -> None:
        super().__init__()
        self._current_path: Path | None = None
        self._dfm_widgets: list[DFMWidget] = []
        self._script_editor_window: Any | None = None

        self.setWindowTitle("FLIC Config Editor")
        self.resize(960, 1020)

        self._build_menu()
        self._build_ui()
        self._auto_load(initial_path)

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")

        new_act = QAction(icon("new"), "&New", self)
        new_act.setShortcut("Ctrl+N")
        new_act.triggered.connect(self._new)
        file_menu.addAction(new_act)

        open_act = QAction(icon("open"), "&Open…", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self._open)
        file_menu.addAction(open_act)

        file_menu.addSeparator()

        save_act = QAction(icon("save"), "&Save", self)
        save_act.setShortcut("Ctrl+S")
        save_act.triggered.connect(self._save)
        file_menu.addAction(save_act)

        saveas_act = QAction(icon("save_as"), "Save &As…", self)
        saveas_act.setShortcut("Ctrl+Shift+S")
        saveas_act.triggered.connect(self._save_as)
        file_menu.addAction(saveas_act)

        file_menu.addSeparator()

        script_act = QAction(icon("scripts", category=Category.SCRIPTS),
                             "Script &Editor…", self)
        script_act.setShortcut("Ctrl+E")
        script_act.setToolTip(
            "Open a visual editor for the YAML's scripts: section. "
            "Requires the config to have been saved first."
        )
        script_act.triggered.connect(self._open_script_editor)
        file_menu.addAction(script_act)

        file_menu.addSeparator()

        exit_act = QAction("E&xit", self)
        exit_act.setShortcut("Ctrl+Q")
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # Compose a top-bar + splitter shell.
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._top_bar = TopBar("FLIC Config Editor")
        self._modified_pill = QLabel("")
        self._modified_pill.setStyleSheet(
            "color: #f59e0b; font-weight: 600; padding: 2px 8px;"
            " border: 1px solid #f59e0b; border-radius: 10px;"
        )
        self._modified_pill.setVisible(False)
        self._top_bar.add_right(self._modified_pill)

        self._btn_theme = QToolButton()
        self._btn_theme.setIcon(icon("theme_dark" if resolved_mode() == "light" else "theme_light"))
        self._btn_theme.setIconSize(QSize(18, 18))
        self._btn_theme.setToolTip("Toggle light / dark theme")
        self._btn_theme.setAutoRaise(True)
        self._btn_theme.clicked.connect(self._toggle_theme)
        self._top_bar.add_right(self._btn_theme)

        self._btn_script_editor = QToolButton()
        self._btn_script_editor.setIcon(icon("script", category=Category.SCRIPTS))
        self._btn_script_editor.setIconSize(QSize(18, 18))
        self._btn_script_editor.setToolTip("Open script editor")
        self._btn_script_editor.setAutoRaise(True)
        self._btn_script_editor.clicked.connect(self._open_script_editor)
        self._top_bar.add_right(self._btn_script_editor)

        outer.addWidget(self._top_bar)

        splitter = QSplitter(Qt.Orientation.Vertical)
        outer.addWidget(splitter, 1)
        self.setCentralWidget(central)

        # ---- Top pane ---------------------------------------------------
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        top_layout.setSpacing(8)
        top_layout.setContentsMargins(6, 6, 6, 6)

        # Experiment Settings
        exp_card = Card("Experiment Settings", Category.LOAD, icon_name="settings")
        self._exp_form = QFormLayout()
        self._exp_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self._exp_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._chamber_size_combo = QComboBox()
        self._chamber_size_combo.addItems(["1  (single-well, 12 chambers)", "2  (two-well, 6 chambers)"])
        self._chamber_size_combo.setCurrentIndex(1)
        self._chamber_size_combo.setMaximumWidth(240)
        self._chamber_size_combo.currentIndexChanged.connect(self._on_chamber_size_changed)
        self._exp_form.addRow("Chamber Size:", self._chamber_size_combo)

        self._experiment_type_combo = QComboBox()
        self._experiment_type_combo.addItems(
            ["(auto)", "hedonic", "progressive_ratio", "two_well", "single_well"]
        )
        self._experiment_type_combo.setMaximumWidth(240)
        self._exp_form.addRow("Experiment Type:", self._experiment_type_combo)

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

        # Auto-filter thresholds (used by auto_remove_chambers)
        filter_header = QLabel("Auto-filter Thresholds  (used by auto_remove_chambers)")
        filter_header.setObjectName("PyflicSectionDivider")
        self._filter_header_row = self._exp_form.rowCount()
        self._exp_form.addRow(filter_header)

        self._min_raw_licks_edit = QLineEdit()
        self._min_raw_licks_edit.setPlaceholderText("e.g. 20  (leave blank to skip)")
        self._min_raw_licks_edit.setMaximumWidth(220)
        self._min_raw_licks_row = self._exp_form.rowCount()
        self._exp_form.addRow("Min Untransformed Licks:", self._min_raw_licks_edit)

        self._max_dur_edit = QLineEdit()
        self._max_dur_edit.setPlaceholderText("e.g. 13  (leave blank to skip)")
        self._max_dur_edit.setMaximumWidth(220)
        self._max_dur_row = self._exp_form.rowCount()
        self._exp_form.addRow("Max Median Duration:", self._max_dur_edit)

        self._max_events_edit = QLineEdit()
        self._max_events_edit.setPlaceholderText("e.g. 150  (leave blank to skip)")
        self._max_events_edit.setMaximumWidth(220)
        self._max_events_row = self._exp_form.rowCount()
        self._exp_form.addRow("Max Events:", self._max_events_edit)

        exp_card.add_body(self._exp_form)
        top_layout.addWidget(exp_card)

        # Global Parameters
        global_card = Card(
            "Global Parameters",
            Category.ANALYZE,
            subtitle="Applied to all DFMs unless overridden per-DFM.",
        )
        self._global_params = ParamsForm(override_mode=False, chamber_size=2)
        global_card.add_body(self._global_params)
        top_layout.addWidget(global_card)

        # Experimental Design Factors
        factors_card = Card(
            "Experimental Design Factors",
            Category.TOOLS,
            subtitle=(
                "Optional. Define factors so chamber assignments use comma-separated "
                "level values. Leave empty for simple treatment names."
            ),
        )
        self._factors_widget = FactorsWidget()
        factors_card.add_body(self._factors_widget)
        top_layout.addWidget(factors_card)

        # Wire factor table changes → update DFM chamber column headers
        self._factors_widget._table.itemChanged.connect(self._on_factors_changed)
        self._factors_widget._table.model().rowsInserted.connect(self._on_factors_changed)
        self._factors_widget._table.model().rowsRemoved.connect(self._on_factors_changed)

        splitter.addWidget(top_widget)

        # ---- Bottom pane: DFM Tabs --------------------------------------
        dfm_card = Card("DFM Configuration", Category.NEUTRAL)
        self._dfm_tabs = QTabWidget()
        dfm_card.add_body(self._dfm_tabs)
        splitter.addWidget(dfm_card)

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
        self._global_params.set_chamber_size(cs)
        for w in self._dfm_widgets:
            w.update_chamber_size(cs)
        self._update_well_names_visibility()

    def _update_well_names_visibility(self) -> None:
        show = self._chamber_size() == 2
        self._exp_form.setRowVisible(self._well_a_row, show)
        self._exp_form.setRowVisible(self._well_b_row, show)
        # The "min untransformed licks" threshold applies to all experiment types.
        self._exp_form.setRowVisible(self._filter_header_row, True)
        self._exp_form.setRowVisible(self._min_raw_licks_row, True)
        # The remaining thresholds are currently hedonic/two-well specific.
        self._exp_form.setRowVisible(self._max_dur_row, show)
        self._exp_form.setRowVisible(self._max_events_row, show)

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

    def _auto_load(self, initial_path: str | Path | None = None) -> None:
        """Load a YAML config on startup.

        If *initial_path* is given:
          • a file → loaded directly;
          • a directory → search for ``flic_config.yaml`` / ``flic_config.yml``.
        Otherwise: search the current working directory for the defaults.
        """
        candidates: list[Path] = []
        if initial_path is not None:
            p = Path(initial_path).expanduser()
            if p.is_file():
                candidates.append(p)
            elif p.is_dir():
                candidates.extend(p / name for name in ("flic_config.yaml", "flic_config.yml"))
        if not candidates:
            candidates.extend(Path.cwd() / name for name in ("flic_config.yaml", "flic_config.yml"))

        for candidate in candidates:
            if candidate.exists():
                try:
                    cfg = yaml.safe_load(candidate.read_text(encoding="utf-8"))
                    if isinstance(cfg, dict):
                        self._current_path = candidate
                        self.setWindowTitle(f"FLIC Config Editor — {candidate.name}")
                        self._populate_from_yaml(cfg)
                except Exception as exc:
                    self.statusBar().showMessage(
                        f"Could not auto-load {candidate.name}: {exc}.  Use File → Open to load manually."
                    )
                break

    # ------------------------------------------------------------------
    # YAML serialisation / deserialisation
    # ------------------------------------------------------------------

    def _collect_yaml(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {}

        global_params = self._global_params.get_values()
        global_params["chamber_size"] = self._chamber_size()
        global_section: dict[str, Any] = {"params": global_params}

        et = self._experiment_type_combo.currentText()
        if et != "(auto)":
            global_section["experiment_type"] = et

        # Well names
        if self._chamber_size() == 2:
            wa = self._well_a_edit.text().strip()
            wb = self._well_b_edit.text().strip()
            if wa or wb:
                global_section["well_names"] = {
                    **({"A": wa} if wa else {}),
                    **({"B": wb} if wb else {}),
                }

        # Filter thresholds → global.constants
        constants: dict[str, Any] = {}
        # Applies to all experiment types
        text = self._min_raw_licks_edit.text().strip()
        if text:
            try:
                constants["min_untransformed_licks_cutoff"] = float(text)
            except ValueError:
                pass
        # Two-well / hedonic extras
        if self._chamber_size() == 2:
            for attr, key in (
                ("_max_dur_edit", "max_med_duration_cutoff"),
                ("_max_events_edit", "max_events_cutoff"),
            ):
                text = getattr(self, attr).text().strip()
                if text:
                    try:
                        constants[key] = float(text)
                    except ValueError:
                        pass
        if constants:
            global_section["constants"] = constants

        # Experimental design factors
        factors = self._factors_widget.get_factors()
        if factors:
            global_section["experimental_design_factors"] = factors

        cfg["global"] = global_section
        cfg["dfms"] = [w.get_dict() for w in self._dfm_widgets]
        return cfg

    def _populate_from_yaml(self, cfg: dict[str, Any]) -> None:
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

        # Experiment type
        et = str(global_cfg.get("experiment_type") or "").strip().lower().replace("-", "_").replace(" ", "_")
        et_items = [self._experiment_type_combo.itemText(i) for i in range(self._experiment_type_combo.count())]
        self._experiment_type_combo.setCurrentIndex(et_items.index(et) if et in et_items else 0)

        # Well names
        well_names = global_cfg.get("well_names") or {}
        self._well_a_edit.setText(str(well_names.get("A", "")))
        self._well_b_edit.setText(str(well_names.get("B", "")))

        # Filter thresholds
        constants = global_cfg.get("constants") or {}
        val = constants.get("min_untransformed_licks_cutoff")
        self._min_raw_licks_edit.setText("" if val is None else str(val))
        for attr, key in (
            ("_max_dur_edit", "max_med_duration_cutoff"),
            ("_max_events_edit", "max_events_cutoff"),
        ):
            val = constants.get(key)
            getattr(self, attr).setText("" if val is None else str(val))

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
        self._well_a_edit.clear()
        self._well_b_edit.clear()
        self._min_raw_licks_edit.clear()
        self._max_dur_edit.clear()
        self._max_events_edit.clear()
        self._experiment_type_combo.setCurrentIndex(0)

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
        chosen = Path(path)
        if chosen.suffix.lower() not in (".yaml", ".yml"):
            chosen = chosen.with_suffix(".yaml")
        self._current_path = chosen
        self.setWindowTitle(f"FLIC Config Editor — {self._current_path.name}")
        self._write_yaml(self._current_path)

    def _open_script_editor(self) -> None:
        """Launch the graphical script editor as a non-modal companion window."""
        if self._current_path is None:
            QMessageBox.information(
                self,
                "Save config first",
                "The script editor writes to the YAML file on disk. "
                "Save your config (File → Save or Save As) first, then try "
                "File → Script Editor again.",
            )
            return

        # If a script editor is already open for any path, bring it forward
        # (and retarget it if the user has since opened a different yaml).
        existing = self._script_editor_window
        if existing is not None:
            try:
                existing.raise_()
                existing.activateWindow()
                return
            except RuntimeError:
                # Underlying C++ object has been destroyed — fall through.
                self._script_editor_window = None

        from .script_editor import ScriptEditorWindow

        win = ScriptEditorWindow(self._current_path, parent=self)
        win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        win.destroyed.connect(lambda: setattr(self, "_script_editor_window", None))
        self._script_editor_window = win
        win.show()

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
    """Launch the FLIC Config Editor GUI.

    Optional CLI argument: a YAML config file (or a directory containing
    ``flic_config.yaml``).  When omitted, looks for ``flic_config.yaml`` in
    the current working directory.
    """
    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app, mode=ui_settings.get("theme", "auto"))
    initial_path = sys.argv[1] if len(sys.argv) > 1 else None
    win = FLICConfigEditor(initial_path=initial_path)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch()
