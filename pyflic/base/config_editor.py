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
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionTab,
    QStylePainter,
    QTabBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .gui_env import sanitize_input_method_environment
from .ui import ActionButton, Card, Category, TopBar, apply_theme, icon, resolved_mode
from .ui.icons import help_color
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


def _param_help_button(key: str, parent: QWidget | None = None):
    """A ``?`` opening the parameter reference at *key*, or ``None``.

    Returns ``None`` if the help package is unavailable, so the config editor
    keeps working with no help installed.
    """
    try:
        from ..help import HelpButton
    except Exception:  # noqa: BLE001 - help is optional, the editor is not
        return None
    return HelpButton(
        f"reference-parameters#{key}", parent,
        tooltip=f"What does {key} do?",
    )


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
        num_columns: int = 3,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._override_mode = override_mode
        self._input_widgets: dict[str, QWidget] = {}
        self._enable_checks: dict[str, QCheckBox] = {}
        self._two_well_rows: list[tuple[QFormLayout, int]] = []

        defaults = _PARAM_DEFAULTS.get(chamber_size, _PARAM_DEFAULTS[2])

        n = len(_PARAM_ORDER)
        c = max(1, num_columns)
        col_size = (n + c - 1) // c
        columns = [_PARAM_ORDER[i * col_size : (i + 1) * col_size] for i in range(c)]

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

                help_btn = _param_help_button(key, self)

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
                    if help_btn is not None:
                        rl.addWidget(help_btn)
                    form.addRow(label, row)
                elif help_btn is not None:
                    row = QWidget()
                    rl = QHBoxLayout(row)
                    rl.setContentsMargins(0, 0, 0, 0)
                    rl.setSpacing(4)
                    rl.addWidget(widget, 1)
                    rl.addWidget(help_btn)
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

    def set_read_only(self, read_only: bool, *, reason: str = "") -> None:
        """Show the values but refuse edits — for a Member of a Project, whose
        detection parameters belong to the Design (ADR-0005)."""
        for widget in self._input_widgets.values():
            widget.setEnabled(not read_only)
            widget.setToolTip(reason if read_only else "")
        for check in self._enable_checks.values():
            check.setEnabled(not read_only)
            check.setToolTip(reason if read_only else "")

    def restrict_overrides(self, allowed: set[str] | None, *,
                           reason: str = "") -> None:
        """Allow only *allowed* keys to be overridden (override mode only).

        Inside a Project only the *physical* keys may vary per DFM; an analysis
        key overridden on one DFM reintroduces exactly the divergence the
        Design outlaws, one level lower and much harder to see.  ``None``
        lifts the restriction.
        """
        if not self._override_mode:
            return
        for key, check in self._enable_checks.items():
            permitted = allowed is None or key in allowed
            if not permitted and check.isChecked():
                check.setChecked(False)
            check.setEnabled(permitted)
            check.setToolTip("" if permitted else reason)
            if not permitted:
                self._input_widgets[key].setEnabled(False)
                self._input_widgets[key].setToolTip(reason)


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
        self._buttons = (add_btn, remove_btn)

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

    def set_read_only(self, read_only: bool, *, reason: str = "") -> None:
        """Show the factors but refuse edits — a Member inherits them."""
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers if read_only
            else QAbstractItemView.EditTrigger.AllEditTriggers)
        self._table.setToolTip(reason if read_only else "")
        for button in self._buttons:
            button.setEnabled(not read_only)
            button.setToolTip(reason if read_only else "")

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


#: Display labels for the two Chamber Layouts.  The layout is the domain term
#: (ADR-0007); ``chamber_size`` is the number it implies, and is derived.
_LAYOUT_LABELS: dict[str, str] = {
    "single_well": "Single-well  (12 chambers)",
    "two_well": "Two-well  (6 chambers)",
}


def _layout_chamber_size(layout: str) -> int:
    """The ``chamber_size`` a Chamber Layout implies, from the one table."""
    from .experiment_types import LAYOUT_CHAMBER_SIZE

    return LAYOUT_CHAMBER_SIZE.get(layout, 2)


def _normalise_layout(raw: Any) -> str | None:
    key = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    return key if key in _LAYOUT_LABELS else None


def _fmt_number(value: Any) -> str:
    """``13.0`` -> ``13``; anything else unchanged.  For placeholder text."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(f)) if f == int(f) else str(f)


class _WestTabBar(QTabBar):
    """A left-hand tab bar whose labels read horizontally.

    Qt draws a West bar's text rotated ninety degrees.  For a stack of "DFM 1"
    … "DFM 12" that is harder to scan than the horizontal strip it replaced,
    which would defeat the point of moving it to the side.
    """

    def tabSizeHint(self, index: int):
        size = super().tabSizeHint(index)
        size.transpose()
        size.setWidth(max(size.width(), 84))
        return size

    def paintEvent(self, _event) -> None:
        painter = QStylePainter(self)
        option = QStyleOptionTab()
        for index in range(self.count()):
            self.initStyleOption(option, index)
            painter.drawControl(QStyle.ControlElement.CE_TabBarTabShape, option)
            painter.drawText(
                self.tabRect(index),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextDontClip,
                self.tabText(index))


#: Tab labels.  The ampersand is doubled because Qt reads a single one as a
#: mnemonic marker and swallows it — "DFMs & Chambers" renders "DFMs_Chambers".
_TAB_EXPERIMENT = "Experiment"
_TAB_DFMS = "DFMs && Chambers"


def _fit_chamber_table(table: QTableWidget) -> None:
    """Make *table* exactly tall enough for every row it holds.

    The table scrolls neither way on purpose — a chamber you cannot see is a
    chamber you will not assign — so its minimum height has to be right.  It
    used to assume 26px rows against a theme that draws 30, which left the
    twelfth chamber of a single-well DFM clipped and unreachable.
    """
    rows = table.rowCount()
    row_h = table.rowHeight(0) if rows else table.verticalHeader().defaultSectionSize()
    header_h = table.horizontalHeader().sizeHint().height()
    table.setMinimumHeight(header_h + rows * row_h + 2 * table.frameWidth() + 2)


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
    for i in range(n_chambers):
        ch_item = QTableWidgetItem(str(i + 1))
        ch_item.setFlags(ch_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        table.setItem(i, 0, ch_item)
        table.setItem(i, 1, QTableWidgetItem(""))
    _fit_chamber_table(table)
    return table


class _LevelDelegate(QStyledItemDelegate):
    """Dropdown editor for the chamber table's factor columns.

    Typing levels invited typos and values the design never declared; the
    dropdown offers exactly what the column's factor defines, plus a blank
    entry to clear the cell (omitting a chamber is "clear the whole row").
    The Treatment column of a config with no factors keeps free text — there
    is nothing defined to pick from.  A level already in the cell that the
    design does not declare is still shown and offered, so opening the editor
    cannot silently rewrite a loaded config; validation paints it red instead.
    """

    def __init__(self, owner: "DFMWidget", parent=None) -> None:
        super().__init__(parent)
        self._owner = owner

    def createEditor(self, parent, option, index):  # noqa: N802 (Qt override)
        levels = self._owner.levels_for_column(index.column())
        if not levels:
            return super().createEditor(parent, option, index)
        combo = QComboBox(parent)
        combo.addItem("")
        combo.addItems(levels)
        current = str(index.data() or "")
        if current and current not in levels:
            combo.addItem(current)
        ## Commit the pick immediately — the default waits for a focus-out,
        ## which reads as the choice not taking.
        combo.activated.connect(lambda _i, c=combo: self._commit(c))
        return combo

    def _commit(self, combo: QComboBox) -> None:
        self.commitData.emit(combo)
        self.closeEditor.emit(combo)

    def setEditorData(self, editor, index):  # noqa: N802 (Qt override)
        if isinstance(editor, QComboBox):
            editor.setCurrentText(str(index.data() or ""))
            ## Deliberately no auto-showPopup: popping the list on a timer
            ## during the click sequence raced the mouse release, which then
            ## landed on the popup and dismissed it — a sporadic dead
            ## dropdown, worst under Wayland.  The single-click path shows a
            ## QMenu instead (see DFMWidget._maybe_edit_cell); this editor
            ## serves double-click and keyboard edits.
        else:
            super().setEditorData(editor, index)

    def setModelData(self, editor, model, index):  # noqa: N802 (Qt override)
        if isinstance(editor, QComboBox):
            model.setData(index, editor.currentText(),
                          Qt.ItemDataRole.EditRole)
        else:
            super().setModelData(editor, model, index)


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
        ## Factor columns edit through a dropdown of the declared levels; a
        ## single click opens it, so assigning a chamber is a pick, not a re-
        ## typing of the design.
        self._chamber_table.setItemDelegate(
            _LevelDelegate(self, self._chamber_table))
        self._chamber_table.cellClicked.connect(self._maybe_edit_cell)
        self._ch_card.add_body(self._chamber_table)
        outer.addWidget(self._ch_card)

        # -- Progressive Ratio: paired chamber per chamber group -----------
        ## Shown only for that type.  The other chamber of each group is
        ## yoked and never written; well A is the sucrose well and the
        ## pi_direction override above says which side it is on.
        from .experiment_types.progressive_ratio import CHAMBER_GROUPS
        self._pr_enabled = False
        self._pr_card = Card(
            "Paired chamber per chamber group", Category.LOAD,
            subtitle="Progressive Ratio: the other chamber of each group is "
                     "yoked. Well A is the sucrose well; PI Direction is its side.")
        pr_row = QHBoxLayout()
        pr_row.setContentsMargins(0, 0, 0, 0)
        self._paired_combos: list[QComboBox] = []
        for g, (a, b) in CHAMBER_GROUPS.items():
            pr_row.addWidget(QLabel(f"Group {g} ({a}+{b}):"))
            combo = QComboBox()
            combo.addItem(f"Chamber {a}", a)
            combo.addItem(f"Chamber {b}", b)
            combo.setFixedWidth(110)
            pr_row.addWidget(combo)
            self._paired_combos.append(combo)
        pr_row.addStretch()
        self._pr_card.add_body(pr_row)
        self._pr_card.setVisible(False)
        outer.addWidget(self._pr_card)

        # -- Parameter overrides -------------------------------------------
        over_card = Card(
            "Parameter Overrides",
            Category.ANALYZE,
            subtitle="Check a box to override the global value for this DFM.",
        )
        ## Two columns, not three.  Three needed 888px inside a page that has
        ## ~877 once the West tab bar and a scrollbar are taken out, so the
        ## whole tab scrolled sideways — much worse than the 54px of extra
        ## height two columns cost.
        self._params_form = ParamsForm(override_mode=True, chamber_size=chamber_size,
                                       num_columns=2)
        over_card.add_body(self._params_form)
        outer.addWidget(over_card)

    # ------------------------------------------------------------------

    def set_progressive_ratio(self, enabled: bool) -> None:
        """Show the paired-chamber pickers (Progressive Ratio) or hide them."""
        self._pr_enabled = bool(enabled)
        self._pr_card.setVisible(self._pr_enabled)

    def paired_chambers(self) -> list[int]:
        return [int(c.currentData()) for c in self._paired_combos]

    def set_paired_chambers(self, chambers) -> None:
        for combo in self._paired_combos:
            for i in range(combo.count()):
                if int(combo.itemData(i)) in {int(c) for c in (chambers or [])}:
                    combo.setCurrentIndex(i)
                    break

    def set_override_restriction(self, allowed: set[str] | None, *,
                                 reason: str = "") -> None:
        """Limit which parameters this DFM may override (Project rule)."""
        self._params_form.restrict_overrides(allowed, reason=reason)

    def levels_for_column(self, column: int) -> list[str]:
        """The declared levels behind chamber-table *column*, or ``[]`` when
        the column is free text (the Chamber column, a Treatment column with
        no factors, or a factor that declares no levels)."""
        names = list(self._factor_levels.keys())
        if not names or column < 1 or column - 1 >= len(names):
            return []
        return list(self._factor_levels.get(names[column - 1]) or [])

    def _level_menu(self, row: int, column: int) -> QMenu | None:
        """The pick-a-level menu for a factor cell, or ``None`` for free
        text.  Blank clears; a current value the factor does not declare is
        offered too, so the menu cannot silently rewrite a loaded config."""
        levels = self.levels_for_column(column)
        item = self._chamber_table.item(row, column)
        if not levels or item is None:
            return None
        current = item.text().strip()
        entries = [""] + levels
        if current and current not in levels:
            entries.append(current)
        menu = QMenu(self._chamber_table)
        for level in entries:
            action = menu.addAction(level or "(clear)")
            action.setData(level)
            action.setCheckable(True)
            action.setChecked(level == current)
        return menu

    def _maybe_edit_cell(self, row: int, column: int) -> None:
        """Single click picks the level from a menu at the cell.

        A QMenu rather than the delegate's combo popup: opening a combo's
        list mid-click raced the mouse release, which landed on the popup
        and dismissed it — a sporadic dead dropdown, worst under Wayland.
        The menu runs its own event loop after the click completes, so it
        cannot lose that race; the delegate still serves double-click and
        keyboard edits.
        """
        menu = self._level_menu(row, column)
        if menu is None:
            return
        rect = self._chamber_table.visualRect(
            self._chamber_table.model().index(row, column))
        chosen = menu.exec(
            self._chamber_table.viewport().mapToGlobal(rect.bottomLeft()))
        if chosen is not None:
            self._chamber_table.item(row, column).setText(chosen.data())

    def _on_chamber_cell_changed(self, item: QTableWidgetItem) -> None:
        col = item.column()
        if col == 0:
            return
        raw = item.text()
        ## A declared level is written as declared: sanitizing one (say, a
        ## hyphenated level) would rewrite the very value the dropdown just
        ## offered, and the mangled text then fails validation.
        if raw not in self.levels_for_column(col):
            clean = _sanitize_treatment(raw)
            if clean != raw:
                self._chamber_table.blockSignals(True)
                item.setText(clean)
                self._chamber_table.blockSignals(False)
        self.revalidate_chambers()

    def _row_levels(self, row: int) -> list[str]:
        """The factor-column texts of *row*, one entry per column, blanks kept.

        Blanks are kept because a factor assignment is *positional*: dropping
        them is what turns "no level for genotype" into "genotype = Paired".
        """
        n_cols = self._chamber_table.columnCount()
        out: list[str] = []
        for c in range(1, n_cols):
            it = self._chamber_table.item(row, c)
            out.append(it.text().strip() if it else "")
        return out

    def revalidate_chambers(self) -> None:
        """Mark every bad cell in the chamber table.

        Two kinds of bad: a level that is not one the factor declares, and a
        *blank* in a row that is otherwise filled.  The second only exists
        because assignments are positional — a half-filled row is an
        incomplete assignment, never a shorter one.
        """
        names = list(self._factor_levels.keys())
        n_cols = self._chamber_table.columnCount()
        self._chamber_table.blockSignals(True)
        for i in range(self._chamber_table.rowCount()):
            parts = self._row_levels(i)
            incomplete = bool(names) and any(parts) and not all(parts)
            for c in range(1, n_cols):
                it = self._chamber_table.item(i, c)
                if it is None:
                    continue
                text = parts[c - 1]
                problem = ""
                if names and c - 1 < len(names):
                    fname = names[c - 1]
                    allowed = self._factor_levels.get(fname) or []
                    if text and allowed and text not in allowed:
                        problem = (f"'{text}' is not a valid level for "
                                   f"'{fname}'. Allowed: {allowed}")
                    elif not text and incomplete:
                        problem = (f"'{fname}' has no level. Factor assignments "
                                   f"are positional — fill every column, or "
                                   f"clear the whole row to omit chamber "
                                   f"{i + 1}.")
                if problem:
                    it.setBackground(QColor("#ffcccc"))
                    it.setToolTip(problem)
                else:
                    it.setData(Qt.ItemDataRole.BackgroundRole, None)
                    it.setToolTip("")
        self._chamber_table.blockSignals(False)

    def chamber_problems(self) -> list[str]:
        """Human-readable problems with this DFM's chamber assignments."""
        dfm_id = self._id_spin.value()
        out: list[str] = []
        if self._pr_enabled and self._chamber_table.rowCount() >= 6:
            ## Both chambers of a chamber group share one treatment.
            from .experiment_types.progressive_ratio import CHAMBER_GROUPS
            for g, (a, b) in CHAMBER_GROUPS.items():
                ta = ", ".join(p for p in self._row_levels(a - 1) if p)
                tb = ", ".join(p for p in self._row_levels(b - 1) if p)
                if ta and tb and ta != tb:
                    out.append(f"DFM {dfm_id} chambers {a} and {b} form one "
                               f"chamber group and must share a treatment "
                               f"('{ta}' vs '{tb}')")
        names = list(self._factor_levels.keys())
        if not names:
            return out
        for i in range(self._chamber_table.rowCount()):
            parts = self._row_levels(i)
            if any(parts) and not all(parts):
                missing = [names[c] for c in range(len(parts))
                           if c < len(names) and not parts[c]]
                out.append(f"DFM {dfm_id} chamber {i + 1}: no level for "
                           f"{', '.join(missing)}")
            for c, text in enumerate(parts):
                if not text or c >= len(names):
                    continue
                allowed = self._factor_levels.get(names[c]) or []
                if allowed and text not in allowed:
                    out.append(f"DFM {dfm_id} chamber {i + 1}: '{text}' is not "
                               f"a level of '{names[c]}'")
        return out

    def assignments_beyond(self, n_keep: int) -> list[tuple[int, str]]:
        """``(chamber number, text)`` for assigned chambers past *n_keep*.

        ``n_keep=0`` asks "what is assigned at all" — which is how removing a
        whole DFM finds out whether it is throwing anything away.
        """
        out: list[tuple[int, str]] = []
        for i in range(n_keep, self._chamber_table.rowCount()):
            parts = self._row_levels(i)
            if any(parts):
                out.append((i + 1, ", ".join(p for p in parts if p)))
        return out

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
        _fit_chamber_table(self._chamber_table)

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
                f"Click a cell and pick the level from its dropdown, one per "
                f"column in the order: {', '.join(factor_names)}.  Every "
                f"column must be filled — pick the blank entry in every "
                f"column to omit a chamber."
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

        self.revalidate_chambers()

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
                parts = self._row_levels(i)
                ## Positional, so a half-filled row is not a shorter tuple.
                ## Compacting it used to write "Chrim" for (blank, Chrim),
                ## which reads back as paired=Chrim — the design silently
                ## rewritten.  Omit the chamber; chamber_problems() says why.
                if not any(parts):
                    val = ""
                elif not all(parts):
                    continue
                else:
                    val = ", ".join(parts)
            if val:
                chambers[i + 1] = val
        if self._pr_enabled:
            result["paired_chambers"] = self.paired_chambers()
        if chambers:
            result["chambers"] = chambers

        return result

    def load_dict(self, data: dict[str, Any], chamber_size: int) -> None:
        self._id_spin.setValue(int(data.get("id", self._id_spin.value())))
        if data.get("paired_chambers") is not None:
            self.set_paired_chambers(data.get("paired_chambers"))

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
        self.revalidate_chambers()


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
        #: The config as read from disk.  Saving rewrites ``global:`` and
        #: ``dfms:`` and leaves every other key — ``scripts:`` above all —
        #: exactly as it was found; rebuilding the file from the widgets alone
        #: deleted a member's Experiment Scripts on the first save.
        self._loaded_raw: dict[str, Any] = {}
        #: The Design governing this file, when it is a Member of a Project,
        #: and the project.yaml it came from.  ``None`` for a standalone
        #: experiment, which is governed by nothing.
        self._design: dict[str, Any] | None = None
        self._design_source: str | None = None

        self.setWindowTitle("FLIC Config Editor")
        ## Either tab fits in 850px — the tallest case is a single-well DFM
        ## with twelve chambers — and 850 still leaves room for the window
        ## chrome on a 1080p display, which 1020 did not.
        self.resize(1000, self._preferred_height(850))

        self._build_menu()
        self._build_ui()
        self._install_help()
        self._auto_load(initial_path)

    @staticmethod
    def _preferred_height(wanted: int) -> int:
        screen = QApplication.primaryScreen()
        if screen is None:
            return wanted
        return max(600, min(wanted, screen.availableGeometry().height() - 80))

    def _install_help(self) -> None:
        """Add the Help menu and the F1 shortcut.

        Guarded so the editor still starts if the help package is missing.
        """
        try:
            from ..help import install_help_shortcut, open_help
        except Exception:  # noqa: BLE001 - help is optional, the editor is not
            return
        install_help_shortcut(self, "app-config-editor")
        help_menu = self.menuBar().addMenu("&Help")
        act_this = QAction(icon("info"), "Config Editor &help", self)
        act_this.setShortcut("F1")
        act_this.triggered.connect(lambda: open_help("app-config-editor"))
        help_menu.addAction(act_this)
        act_params = QAction(icon("sensitivity", category=Category.ANALYZE),
                             "&Parameter reference", self)
        act_params.triggered.connect(lambda: open_help("reference-parameters"))
        help_menu.addAction(act_params)
        help_menu.addSeparator()
        act_start = QAction(icon("home"), "&Getting started", self)
        act_start.triggered.connect(lambda: open_help("getting-started"))
        help_menu.addAction(act_start)

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

        ## Two tabs rather than one splitter.  The DFM configuration used to
        ## live in the bottom pane of a vertical splitter, where it competed
        ## for height with three cards that never needed it and lost.
        self._tabs = QTabWidget()
        outer.addWidget(self._tabs, 1)
        self.setCentralWidget(central)

        # ==== Tab 1: Experiment ==========================================
        exp_page = QWidget()
        top_layout = QVBoxLayout(exp_page)
        top_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        top_layout.setSpacing(8)
        top_layout.setContentsMargins(6, 6, 6, 6)

        ## Says, once and at the top, why the whole left/right row below is
        ## read-only.  Every alternative (a tooltip per field, a greyed form
        ## with no explanation) leaves someone clicking a field that will
        ## never take a keystroke.
        self._design_banner = QLabel("")
        self._design_banner.setObjectName("PyflicDesignBanner")
        self._design_banner.setWordWrap(True)
        self._design_banner.setVisible(False)
        self._design_banner.setStyleSheet(
            f"QLabel#PyflicDesignBanner {{"
            f"  color: {help_color()};"
            f"  border: 1px solid {help_color()};"
            f"  border-radius: 6px;"
            f"  padding: 6px 10px;"
            f"}}")
        top_layout.addWidget(self._design_banner)

        # Side-by-side row: Experiment Settings (left) + Global Parameters (right)
        side_row = QHBoxLayout()
        side_row.setContentsMargins(0, 0, 0, 0)
        side_row.setSpacing(8)

        # Experiment Settings
        exp_card = Card("Experiment Settings", Category.LOAD, icon_name="settings")
        self._exp_form = QFormLayout()
        self._exp_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self._exp_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        ## The Experiment Type comes from the registry, so adding a type
        ## reaches this menu with no edit here — and so the retired layout
        ## names (two_well, single_well) cannot reappear in it (ADR-0007).
        from . import experiment_types as _et_mod

        self._experiment_type_combo = QComboBox()
        for item in _et_mod.available_experiment_types():
            self._experiment_type_combo.addItem(item.display_name, item.name)
        self._experiment_type_combo.setMaximumWidth(240)
        self._last_type_index = self._experiment_type_combo.currentIndex()
        self._experiment_type_combo.currentIndexChanged.connect(
            self._on_experiment_type_changed)
        self._exp_form.addRow("Experiment Type:", self._experiment_type_combo)

        ## Shown even when the type owns it: the layout is what decides
        ## whether the other tab has 6 chambers or 12, and a derived value
        ## that drives visible structure elsewhere should be readable.
        self._chamber_layout_combo = QComboBox()
        for key, label in _LAYOUT_LABELS.items():
            self._chamber_layout_combo.addItem(label, key)
        self._chamber_layout_combo.setCurrentIndex(
            self._chamber_layout_combo.findData("two_well"))
        self._chamber_layout_combo.setMaximumWidth(240)
        self._last_layout_index = self._chamber_layout_combo.currentIndex()
        self._chamber_layout_combo.currentIndexChanged.connect(
            self._on_chamber_layout_changed)
        self._layout_hint = QLabel("")
        self._layout_hint.setObjectName("PyflicCardSubtitle")
        self._layout_hint.setVisible(False)
        layout_row = QWidget()
        layout_row_l = QHBoxLayout(layout_row)
        layout_row_l.setContentsMargins(0, 0, 0, 0)
        layout_row_l.setSpacing(6)
        layout_row_l.addWidget(self._chamber_layout_combo)
        layout_row_l.addWidget(self._layout_hint)
        layout_row_l.addStretch()
        self._exp_form.addRow("Chamber Layout:", layout_row)

        # Well Names (two-well only) — inline in experiment settings
        self._well_a_edit = QLineEdit()
        self._well_a_edit.setPlaceholderText("e.g. Sucrose")
        self._well_a_edit.setMaximumWidth(200)
        self._well_a_edit.textChanged.connect(self._refresh_badges)
        self._well_b_edit = QLineEdit()
        self._well_b_edit.setPlaceholderText("e.g. Yeast")
        self._well_b_edit.setMaximumWidth(200)
        self._well_b_edit.textChanged.connect(self._refresh_badges)
        self._well_a_row = self._exp_form.rowCount()
        self._exp_form.addRow("Well A:", self._well_a_edit)
        self._well_b_row = self._exp_form.rowCount()
        self._exp_form.addRow("Well B:", self._well_b_edit)

        # Lick transformation toggle — applies experiment-wide.
        self._transform_licks_check = QCheckBox(
            "Apply 0.25-power transform to Licks (uncheck for raw counts)"
        )
        self._transform_licks_check.setChecked(True)
        self._exp_form.addRow("Transform Licks:", self._transform_licks_check)

        # Auto-filter thresholds (used by auto_remove_chambers)
        filter_header = QLabel("Auto-filter Thresholds  (used by auto_remove_chambers)")
        filter_header.setObjectName("PyflicSectionDivider")
        self._filter_header_row = self._exp_form.rowCount()
        self._exp_form.addRow(filter_header)

        ## Placeholders are filled from the type's default_constants, never
        ## hardcoded: resolve_constants() merges those *under* the yaml, so a
        ## blank field inherits rather than skips, and saying otherwise (as
        ## "leave blank to skip" did) is simply false for a typed experiment.
        self._min_raw_licks_edit = QLineEdit()
        self._min_raw_licks_edit.setMaximumWidth(260)
        self._min_raw_licks_row = self._exp_form.rowCount()
        self._exp_form.addRow("Min Untransformed Licks:", self._min_raw_licks_edit)

        self._max_dur_edit = QLineEdit()
        self._max_dur_edit.setMaximumWidth(260)
        self._max_dur_row = self._exp_form.rowCount()
        self._exp_form.addRow("Max Median Duration:", self._max_dur_edit)

        self._max_events_edit = QLineEdit()
        self._max_events_edit.setMaximumWidth(260)
        self._max_events_row = self._exp_form.rowCount()
        self._exp_form.addRow("Max Events:", self._max_events_edit)

        exp_card.add_body(self._exp_form)
        ## Top-aligned: in a plain hbox the shorter of the two cards is
        ## stretched to the taller one's height, which spreads its title away
        ## from its own fields.
        side_row.addWidget(exp_card, 1, Qt.AlignmentFlag.AlignTop)

        # Global Parameters
        global_card = Card(
            "Global Parameters",
            Category.ANALYZE,
            subtitle="Applied to all DFMs unless overridden per-DFM.",
        )
        ## One column.  Two put "PI Direction (side with PI = 1)" past the
        ## right edge of a half-width card and gave the whole page a
        ## horizontal scrollbar; the tab has vertical room to spare instead.
        self._global_params = ParamsForm(override_mode=False, chamber_size=2, num_columns=1)
        global_card.add_body(self._global_params)
        side_row.addWidget(global_card, 1, Qt.AlignmentFlag.AlignTop)

        top_layout.addLayout(side_row)

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
        ## The cards keep their natural height; the slack goes here rather
        ## than stretching a three-row factor table down the window.
        top_layout.addStretch(1)

        # Wire factor table changes → update DFM chamber column headers
        self._factors_widget._table.itemChanged.connect(self._on_factors_changed)
        self._factors_widget._table.model().rowsInserted.connect(self._on_factors_changed)
        self._factors_widget._table.model().rowsRemoved.connect(self._on_factors_changed)

        exp_scroll = QScrollArea()
        exp_scroll.setWidgetResizable(True)
        exp_scroll.setFrameShape(QFrame.Shape.NoFrame)
        exp_scroll.setWidget(exp_page)
        self._tabs.addTab(exp_scroll, _TAB_EXPERIMENT)

        # ==== Tab 2: DFMs & Chambers =====================================
        dfm_page = QWidget()
        dfm_layout = QVBoxLayout(dfm_page)
        dfm_layout.setSpacing(6)
        dfm_layout.setContentsMargins(6, 6, 6, 6)

        ## A West tab bar, not a second horizontal one: two stacked horizontal
        ## strips read as one confused bar, and twenty DFMs scroll sideways
        ## where they stack down the side for free.
        self._dfm_tabs = QTabWidget()
        self._dfm_tabs.setTabBar(_WestTabBar())
        self._dfm_tabs.setTabPosition(QTabWidget.TabPosition.West)
        dfm_layout.addWidget(self._dfm_tabs, 1)

        dfm_btn_row = QHBoxLayout()
        dfm_btn_row.setContentsMargins(0, 0, 0, 0)
        dfm_btn_row.setSpacing(6)
        self._btn_add_dfm = ActionButton("Add DFM", Category.LOAD, icon_name="new")
        self._btn_add_dfm.setMaximumWidth(140)
        self._btn_add_dfm.clicked.connect(self._add_dfm)
        self._btn_remove_dfm = ActionButton("Remove DFM", Category.QC, icon_name="clear")
        self._btn_remove_dfm.setMaximumWidth(150)
        self._btn_remove_dfm.clicked.connect(self._remove_dfm)
        dfm_btn_row.addWidget(self._btn_add_dfm)
        dfm_btn_row.addWidget(self._btn_remove_dfm)
        dfm_btn_row.addStretch()
        dfm_layout.addLayout(dfm_btn_row)

        self._tabs.addTab(dfm_page, _TAB_DFMS)

        self._sync_dfm_tabs(1, 2)
        self._sync_layout_control()
        self._refresh_threshold_hints()
        self._update_well_names_visibility()
        self._update_dfm_buttons()
        self._refresh_badges()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _chamber_layout(self) -> str:
        return str(self._chamber_layout_combo.currentData() or "two_well")

    def _chamber_size(self) -> int:
        return _layout_chamber_size(self._chamber_layout())

    def _n_chambers(self, layout: str | None = None) -> int:
        return 12 if _layout_chamber_size(layout or self._chamber_layout()) == 1 else 6

    def _current_type(self):
        """The selected ``ExperimentType``, Custom if anything is amiss."""
        from . import experiment_types

        try:
            return experiment_types.get_experiment_type(
                self._experiment_type_combo.currentData())
        except Exception:  # noqa: BLE001 - an unknown name is a Custom Experiment
            return experiment_types.get_experiment_type(None)

    # ---- Experiment Type / Chamber Layout ----------------------------

    def _sync_layout_control(self) -> None:
        """Enable the layout combo only where it is genuinely free.

        Two authorities can take it away: the Experiment Type (ADR-0007) and,
        one level up, the Project Design (ADR-0005).
        """
        item = self._current_type()
        fixed = item.chamber_layout is not None
        self._chamber_layout_combo.setEnabled(not fixed and not self._design)
        self._layout_hint.setVisible(fixed)
        self._layout_hint.setText(f"set by {item.display_name}" if fixed else "")
        self._chamber_layout_combo.setToolTip(
            f"The {item.display_name} experiment type fixes the chamber layout."
            if fixed else "")

    def _on_experiment_type_changed(self, idx: int) -> None:
        item = self._current_type()
        fixed = item.chamber_layout
        if fixed is not None and not self._set_chamber_layout(fixed):
            ## The layout change was declined, so the type change that asked
            ## for it never happened either — anything else leaves a Hedonic
            ## experiment sitting on a single-well plate.
            self._experiment_type_combo.blockSignals(True)
            self._experiment_type_combo.setCurrentIndex(self._last_type_index)
            self._experiment_type_combo.blockSignals(False)
            return
        self._last_type_index = idx
        self._sync_layout_control()
        self._refresh_threshold_hints()
        self._update_well_names_visibility()
        self._sync_pr_widgets()
        self._refresh_badges()

    def _sync_pr_widgets(self) -> None:
        """Show the paired-chamber pickers on every DFM tab for Progressive
        Ratio and hide them otherwise."""
        enabled = self._current_type().name == "ProgressiveRatio"
        for w in getattr(self, "_dfm_widgets", []):
            w.set_progressive_ratio(enabled)

    def _on_chamber_layout_changed(self, idx: int) -> None:
        ## Roll the combo back first, then go through the one guarded path —
        ## so a user change and a type-driven change cannot diverge.
        layout = str(self._chamber_layout_combo.itemData(idx))
        self._chamber_layout_combo.blockSignals(True)
        self._chamber_layout_combo.setCurrentIndex(self._last_layout_index)
        self._chamber_layout_combo.blockSignals(False)
        self._set_chamber_layout(layout)

    def _set_chamber_layout(self, layout: str) -> bool:
        """Move to *layout*, asking first if chambers would be discarded.

        Returns False when the user declined, so the caller can undo whatever
        asked for the change.
        """
        idx = self._chamber_layout_combo.findData(layout)
        if idx < 0 or idx == self._chamber_layout_combo.currentIndex():
            return True
        losing = self._assignments_beyond(self._n_chambers(layout))
        if losing and not self._confirm_discard(
                f"Switching to {_LAYOUT_LABELS[layout].split('  ')[0].lower()} "
                f"chambers", losing):
            return False
        self._chamber_layout_combo.blockSignals(True)
        self._chamber_layout_combo.setCurrentIndex(idx)
        self._chamber_layout_combo.blockSignals(False)
        self._last_layout_index = idx
        self._apply_chamber_size(_layout_chamber_size(layout))
        return True

    def _apply_chamber_size(self, cs: int) -> None:
        self._global_params.set_chamber_size(cs)
        for w in self._dfm_widgets:
            w.update_chamber_size(cs)
        self._update_well_names_visibility()
        self._refresh_badges()

    # ---- Destructive-edit guard --------------------------------------

    def _assignments_beyond(self, n_keep: int) -> list[str]:
        """Assigned chambers that a shrink to *n_keep* rows would discard."""
        out: list[str] = []
        for w in self._dfm_widgets:
            dfm_id = w._id_spin.value()
            for chamber, text in w.assignments_beyond(n_keep):
                out.append(f"DFM {dfm_id} chamber {chamber}: {text}")
        return out

    def _confirm_discard(self, action: str, items: list[str]) -> bool:
        """Ask before throwing away work that is not on screen.

        Only ever called when something would actually be lost — building a
        fresh config, where every cell is blank, must never be interrupted.
        """
        shown = items[:12]
        detail = "\n".join(f"  \u2022 {t}" for t in shown)
        if len(items) > len(shown):
            detail += f"\n  \u2026 and {len(items) - len(shown)} more"
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Discard chamber assignments?")
        box.setText(f"{action} will discard chamber assignments that have "
                    f"already been made.")
        box.setInformativeText(detail + "\n\nThis cannot be undone.")
        box.setStandardButtons(QMessageBox.StandardButton.Discard
                               | QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        return box.exec() == QMessageBox.StandardButton.Discard

    # ---- Auto-filter thresholds --------------------------------------

    def _threshold_fields(self) -> tuple[tuple[QLineEdit, str], ...]:
        return (
            (self._min_raw_licks_edit, "min_untransformed_licks_cutoff"),
            (self._max_dur_edit, "max_med_duration_cutoff"),
            (self._max_events_edit, "max_events_cutoff"),
        )

    def _refresh_threshold_hints(self) -> None:
        """Show the type's default as placeholder text, never as a value.

        Writing the number into the field would freeze this config against
        today's defaults; leaving it blank keeps the type the owner, which is
        the whole point of it having them (ADR-0011).
        """
        item = self._current_type()
        defaults = item.default_constants or {}
        for widget, key in self._threshold_fields():
            value = defaults.get(key)
            if value is None:
                widget.setPlaceholderText("no default \u2014 blank skips this filter")
            else:
                widget.setPlaceholderText(
                    f"{_fmt_number(value)}  (default for {item.display_name})")

    def _update_well_names_visibility(self) -> None:
        two_well = self._chamber_size() == 2
        self._exp_form.setRowVisible(self._well_a_row, two_well)
        self._exp_form.setRowVisible(self._well_b_row, two_well)
        ## A threshold row is shown when the layout has the metric or the type
        ## carries a default for it — not because the layout happens to be
        ## two-well, which is what this used to key off.
        defaults = self._current_type().default_constants or {}
        self._exp_form.setRowVisible(self._filter_header_row, True)
        self._exp_form.setRowVisible(self._min_raw_licks_row, True)
        self._exp_form.setRowVisible(
            self._max_dur_row, two_well or "max_med_duration_cutoff" in defaults)
        self._exp_form.setRowVisible(
            self._max_events_row, two_well or "max_events_cutoff" in defaults)

    def _threshold_row_visible(self, key: str) -> bool:
        two_well = self._chamber_size() == 2
        defaults = self._current_type().default_constants or {}
        if key == "min_untransformed_licks_cutoff":
            return True
        return two_well or key in defaults

    # ---- DFMs ---------------------------------------------------------

    def _on_factors_changed(self, *_args) -> None:
        factors = self._factors_widget.get_factors()
        for w in self._dfm_widgets:
            w.update_factors(factors)
        self._refresh_badges()

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
        self._refresh_badges()

    def _append_dfm(self, dfm_id: int, chamber_size: int) -> DFMWidget:
        factors = self._factors_widget.get_factors() if hasattr(self, "_factors_widget") else {}
        w = DFMWidget(dfm_id=dfm_id, chamber_size=chamber_size)
        w.update_factors(factors)
        w.set_progressive_ratio(self._current_type().name == "ProgressiveRatio")
        for combo in w._paired_combos:
            combo.currentIndexChanged.connect(self._refresh_badges)
        w._id_spin.valueChanged.connect(lambda val, _w=w: self._on_dfm_id_changed(_w, val))
        w._chamber_table.itemChanged.connect(self._refresh_badges)
        self._dfm_widgets.append(w)
        tab_scroll = QScrollArea()
        tab_scroll.setWidgetResizable(True)
        tab_scroll.setFrameShape(QFrame.Shape.NoFrame)
        tab_scroll.setWidget(w)
        self._dfm_tabs.addTab(tab_scroll, f"DFM {dfm_id}")
        return w

    def _sync_dfm_tabs(self, n: int, chamber_size: int) -> None:
        """Force the tab count to *n* — used when loading a file, not by hand.

        Interactive add and remove go through :meth:`_add_dfm` and
        :meth:`_remove_dfm`, which say which DFM they are about to discard.
        """
        current = len(self._dfm_widgets)
        if n < current:
            for _ in range(current - n):
                self._dfm_tabs.removeTab(self._dfm_tabs.count() - 1)
                w = self._dfm_widgets.pop()
                w.deleteLater()
        elif n > current:
            used = {w._id_spin.value() for w in self._dfm_widgets}
            for _ in range(n - current):
                dfm_id = next(k for k in range(1, 200) if k not in used)
                used.add(dfm_id)
                self._append_dfm(dfm_id, chamber_size)
        self._update_dfm_buttons()

    def _add_dfm(self) -> None:
        if len(self._dfm_widgets) >= 20:
            return
        used = {w._id_spin.value() for w in self._dfm_widgets}
        dfm_id = next(k for k in range(1, 200) if k not in used)
        self._append_dfm(dfm_id, self._chamber_size())
        self._dfm_tabs.setCurrentIndex(len(self._dfm_widgets) - 1)
        self._update_dfm_buttons()
        self._refresh_badges()

    def _remove_dfm(self) -> None:
        """Remove the *selected* DFM.

        The count spinner this replaced always dropped the last tab, which
        with freely-editable DFM ids meant it dropped whichever DFM happened
        to sit at the end — rarely the one anybody meant.
        """
        idx = self._dfm_tabs.currentIndex()
        if idx < 0 or len(self._dfm_widgets) <= 1:
            return
        w = self._dfm_widgets[idx]
        dfm_id = w._id_spin.value()
        losing = [f"chamber {c}: {t}" for c, t in w.assignments_beyond(0)]
        if losing and not self._confirm_discard(f"Removing DFM {dfm_id}", losing):
            return
        self._dfm_tabs.removeTab(idx)
        self._dfm_widgets.pop(idx)
        w.deleteLater()
        self._update_dfm_buttons()
        self._refresh_badges()

    def _update_dfm_buttons(self) -> None:
        if not hasattr(self, "_btn_add_dfm"):
            return
        n = len(self._dfm_widgets)
        self._btn_add_dfm.setEnabled(n < 20)
        self._btn_remove_dfm.setEnabled(n > 1)
        self._btn_remove_dfm.setToolTip(
            "" if n > 1 else "An experiment needs at least one DFM.")

    # ---- Validation ---------------------------------------------------

    def _problems(self) -> tuple[list[str], list[str]]:
        """Problems on the Experiment tab and on the DFM tab, separately.

        The Experiment half is ``ExperimentType.validate()`` — the loader's
        own function, so the editor and the loader cannot drift about what
        counts as valid.
        """
        experiment: list[str] = []
        if not self._design:
            try:
                experiment = list(self._current_type().validate(
                    self._global_section()))
            except Exception:  # noqa: BLE001 - validation must never block the UI
                experiment = []
        dfms: list[str] = []
        for w in self._dfm_widgets:
            dfms.extend(w.chamber_problems())
        return experiment, dfms

    def _refresh_badges(self, *_args) -> None:
        if not hasattr(self, "_tabs"):
            return
        experiment, dfms = self._problems()
        self._tabs.setTabText(
            0, _TAB_EXPERIMENT + (f"  \u26a0 {len(experiment)}" if experiment else ""))
        self._tabs.setTabToolTip(0, "\n".join(experiment))
        self._tabs.setTabText(
            1, _TAB_DFMS + (f"  \u26a0 {len(dfms)}" if dfms else ""))
        self._tabs.setTabToolTip(1, "\n".join(dfms))

    def _auto_load(self, initial_path: str | Path | None = None) -> None:
        """Load a YAML config on startup.

        If *initial_path* is given:
          • a file → loaded directly;
          • a directory → search for ``flic_config.yaml`` / ``flic_config.yml``;
            with no config there, DFM CSVs found in the directory (loose, or
            already filed into ``data/``) preload one DFM tab per id — the
            "initialize an existing recording" case, where the plate already
            says which DFMs exist and retyping their ids is pure error surface.
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
                        self._load_config(cfg)
                except Exception as exc:
                    self.statusBar().showMessage(
                        f"Could not auto-load {candidate.name}: {exc}.  Use File → Open to load manually."
                    )
                break
        else:
            self._preload_from_data(initial_path)

    def _preload_from_data(self, initial_path: str | Path | None) -> None:
        """No config to load — preload the DFMs the data files name.

        Goes through :meth:`_load_config` like a real file, so a directory
        inside a Project also picks up the Design's ``global:`` and its
        field locks.  Saving writes the directory's ``flic_config.yaml``.
        """
        if initial_path is None:
            return
        directory = Path(initial_path).expanduser()
        if not directory.is_dir():
            return
        from . import layout as layout_mod
        from .project import dfm_ids_in_data

        ids = dfm_ids_in_data(directory) or layout_mod.dfm_ids(directory)
        if not ids:
            return
        self._current_path = directory / "flic_config.yaml"
        self.setWindowTitle(f"FLIC Config Editor — {self._current_path.name}")
        self._load_config(
            {"dfms": [{"id": dfm_id, "chambers": {}} for dfm_id in ids]})
        id_list = ", ".join(str(i) for i in ids)
        self.statusBar().showMessage(
            f"No flic_config.yaml here yet — preloaded DFM(s) {id_list} "
            "found in the data files.  Assign chambers and save.")

    # ------------------------------------------------------------------
    # YAML serialisation / deserialisation
    # ------------------------------------------------------------------

    def _global_section(self) -> dict[str, Any]:
        """The ``global:`` block these widgets describe.

        Separate from :meth:`_collect_yaml` because validation needs it too,
        and validating something other than what gets written is how an editor
        comes to bless a config the loader rejects.
        """
        item = self._current_type()
        global_section: dict[str, Any] = {}

        ## ADR-0007 / ADR-0011.  A typed config writes neither
        ## ``chamber_layout`` nor ``params.chamber_size`` — the type owns both
        ## and they are derived.  A Custom Experiment states the layout,
        ## because single-well versus two-well silently reinterprets the
        ## whole plate and is not a thing to leave implicit.
        if item.is_custom:
            global_section["chamber_layout"] = self._chamber_layout()
        else:
            global_section["experiment_type"] = item.name

        global_params = self._global_params.get_values()
        global_params.pop("chamber_size", None)
        global_section["params"] = global_params

        # Lick transformation — only emit the key when it differs from the
        # historical default (True) to keep YAML output minimal.
        if not self._transform_licks_check.isChecked():
            global_section["transform_licks"] = False

        # Well names
        if self._chamber_size() == 2:
            wa = self._well_a_edit.text().strip()
            wb = self._well_b_edit.text().strip()
            if wa or wb:
                global_section["well_names"] = {
                    **({"A": wa} if wa else {}),
                    **({"B": wb} if wb else {}),
                }

        ## Only what the experimenter actually typed reaches ``constants:``.
        ## A blank field inherits the type's default, so materialising it here
        ## would freeze this config against today's numbers (ADR-0011).
        constants: dict[str, Any] = {}
        for widget, key in self._threshold_fields():
            if not self._threshold_row_visible(key):
                continue
            text = widget.text().strip()
            if not text:
                continue
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

        return global_section

    def _collect_yaml(self) -> dict[str, Any]:
        ## Start from the file as it was read so keys this editor knows
        ## nothing about survive — a member's scripts: above all, which a
        ## rebuilt-from-widgets config used to delete on the first save.
        cfg = {k: v for k, v in self._loaded_raw.items()
               if k not in ("global", "dfms", "DFMs")}
        if self._design:
            ## A Member inherits global: from the Design.  Writing one here
            ## would at best duplicate the authority and at worst contradict
            ## it, and a contradiction stops the whole Project loading.
            cfg.pop("global", None)
        else:
            cfg["global"] = self._global_section()
        cfg["dfms"] = [w.get_dict() for w in self._dfm_widgets]
        return cfg

    def _load_config(self, cfg: dict[str, Any]) -> None:
        """Populate from *cfg* — with the Project Design standing in for
        ``global:`` when the file is a Member of one.

        The substitution happens *before* the widgets are filled rather than
        after, because the design's factors decide how a chamber assignment is
        read: "w1118, M" is two factor levels under a two-factor design and a
        single treatment name under none, and the second reading rewrites the
        cell.
        """
        self._loaded_raw = dict(cfg)
        self._read_design()
        effective = dict(cfg)
        if self._design:
            effective["global"] = self._design_as_global()
        self._populate_from_yaml(effective)
        self._apply_design()
        ## The DFM tabs may have been rebuilt before the type combo settled;
        ## the pickers follow the type as it stands after the whole load.
        self._sync_pr_widgets()

    def _design_as_global(self) -> dict[str, Any]:
        """The Design shaped like a ``global:`` block the editor can load.

        The one addition is ``params.chamber_size``: the Design never states it
        (the Experiment Type owns the layout it comes from), but every widget
        here is sized by it.
        """
        from . import experiment_types

        design = dict(self._design or {})
        item = experiment_types.get_experiment_type(
            design.get("experiment_type"))
        params = dict(design.get("params") or {})
        params["chamber_size"] = item.resolve_chamber_size(design)
        design["params"] = params
        return design

    def _read_design(self) -> None:
        """Find the Design governing the file being edited, if any."""
        self._design = None
        self._design_source = None
        if self._current_path is None:
            return
        from . import project as project_mod

        try:
            design, source = project_mod.design_for_member(
                self._current_path.parent)
        except Exception:  # noqa: BLE001 - a bad project.yaml governs nothing
            return
        if design:
            self._design = design
            self._design_source = source

    def _apply_design(self) -> None:
        """Lock the fields the Project Design owns, and say why.

        A Member of a Project does not own ``global:`` — the Design does, and a
        Member that states anything different **fails to load** (ADR-0005).
        The editor therefore shows what is actually in force and refuses the
        edit here, rather than letting someone type a value that will break
        the Project the next time it is opened.
        """
        governed = bool(self._design)
        reason = (f"Owned by the Project design in {self._design_source}"
                  if governed else "")

        for widget in (self._chamber_layout_combo, self._experiment_type_combo,
                       self._well_a_edit, self._well_b_edit,
                       self._transform_licks_check, self._min_raw_licks_edit,
                       self._max_dur_edit, self._max_events_edit):
            widget.setEnabled(not governed)
            widget.setToolTip(reason)
        self._global_params.set_read_only(governed, reason=reason)
        self._factors_widget.set_read_only(governed, reason=reason)
        ## Re-assert the Experiment Type's claim on the layout: the loop above
        ## has just re-enabled it on the Design's say-so, and the type's is the
        ## narrower authority of the two.
        self._sync_layout_control()

        from .yaml_config import PHYSICAL_DFM_KEYS

        for widget in self._dfm_widgets:
            widget.set_override_restriction(
                set(PHYSICAL_DFM_KEYS) if governed else None,
                reason=("Inside a Project only the physical keys may vary per "
                        "DFM — an analysis key overridden here would "
                        "reintroduce the divergence the design outlaws."))

        self._design_banner.setVisible(governed)
        if governed:
            self._design_banner.setText(
                "<b>These settings belong to the Project design.</b>  This "
                "experiment is a member of the Project at "
                f"<code>{self._design_source}</code> and inherits every "
                "global setting from it — shown here, edited there (Hub → "
                "Project → Project design…).  Saving leaves this file's "
                "<code>global:</code> out entirely, which is what makes the "
                "inheritance work.")

    def _resolve_type_and_layout(self, global_cfg: dict[str, Any]):
        """The Experiment Type and Chamber Layout *this* config means.

        The read path stays deliberately forgiving where the write path is
        strict: a pre-ADR-0007 config naming ``experiment_type: two_well``, or
        carrying only ``params.chamber_size``, still opens — and the next save
        writes it in the new form.
        """
        from . import experiment_types

        raw_type = global_cfg.get("experiment_type")
        legacy_layout: str | None = None
        try:
            item = experiment_types.get_experiment_type(raw_type)
        except ValueError:
            ## Either a retired layout name or one this build does not know.
            ## Both are a Custom Experiment; a retired name also tells us the
            ## layout, which is the whole content of that migration.
            key = str(raw_type or "").strip().lower().replace("-", "_")
            item = experiment_types.get_experiment_type(None)
            if "single" in key:
                legacy_layout = "single_well"
            elif "two" in key:
                legacy_layout = "two_well"

        if item.chamber_layout is not None:
            layout = item.chamber_layout
        elif legacy_layout is not None:
            layout = legacy_layout
        elif _normalise_layout(global_cfg.get("chamber_layout")):
            layout = _normalise_layout(global_cfg.get("chamber_layout"))
        else:
            params = global_cfg.get("params", global_cfg.get("parameters", {})) or {}
            layout = "single_well" if int(params.get("chamber_size", 2)) == 1 else "two_well"
        return item, layout

    def _populate_from_yaml(self, cfg: dict[str, Any]) -> None:
        global_cfg = cfg.get("global", {}) or {}
        global_params_raw = global_cfg.get("params", global_cfg.get("parameters", {})) or {}

        item, layout = self._resolve_type_and_layout(global_cfg)
        chamber_size = _layout_chamber_size(layout)

        type_idx = self._experiment_type_combo.findData(item.name)
        self._experiment_type_combo.blockSignals(True)
        self._experiment_type_combo.setCurrentIndex(max(0, type_idx))
        self._experiment_type_combo.blockSignals(False)
        self._last_type_index = self._experiment_type_combo.currentIndex()

        layout_idx = self._chamber_layout_combo.findData(layout)
        self._chamber_layout_combo.blockSignals(True)
        self._chamber_layout_combo.setCurrentIndex(max(0, layout_idx))
        self._chamber_layout_combo.blockSignals(False)
        self._last_layout_index = self._chamber_layout_combo.currentIndex()

        self._sync_layout_control()
        self._refresh_threshold_hints()
        self._update_well_names_visibility()

        params_to_load = {k: v for k, v in global_params_raw.items() if k != "chamber_size"}
        self._global_params.load_values(params_to_load, chamber_size)
        self._global_params.set_chamber_size(chamber_size)

        # Well names
        well_names = global_cfg.get("well_names") or {}
        self._well_a_edit.setText(str(well_names.get("A", "")))
        self._well_b_edit.setText(str(well_names.get("B", "")))

        # Lick transformation toggle
        self._transform_licks_check.setChecked(bool(global_cfg.get("transform_licks", True)))

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

        if not dfm_nodes and self._current_path is not None:
            ## A config that lists no DFMs, sitting beside data that does —
            ## a just-initialized directory.  Preload the ids the files name
            ## rather than a lone default DFM 1 that would be retyped from
            ## the filenames.
            from . import layout as layout_mod
            from .project import dfm_ids_in_data

            directory = self._current_path.parent
            ids = dfm_ids_in_data(directory) or layout_mod.dfm_ids(directory)
            if ids:
                dfm_nodes = [{"id": dfm_id, "chambers": {}} for dfm_id in ids]
                self.statusBar().showMessage(
                    "Config lists no DFMs — preloaded DFM(s) "
                    f"{', '.join(str(i) for i in ids)} found in the data "
                    "files.  Assign chambers and save.")

        n = max(1, len(dfm_nodes))
        ## Resize the widgets that already exist before adjusting the count:
        ## _sync_dfm_tabs only sizes the ones it creates, so loading a
        ## single-well config into a freshly-opened editor used to leave the
        ## chamber tables at six rows and the last six assignments unreachable.
        for widget in self._dfm_widgets:
            widget.update_chamber_size(chamber_size)
        self._sync_dfm_tabs(n, chamber_size)

        for i, node in enumerate(dfm_nodes):
            if i < len(self._dfm_widgets):
                self._dfm_widgets[i].update_factors(factors)
                self._dfm_widgets[i]._id_spin.blockSignals(True)
                self._dfm_widgets[i].load_dict(node, chamber_size)
                self._dfm_widgets[i]._id_spin.blockSignals(False)
                dfm_id = int(node.get("id", i + 1))
                self._dfm_tabs.setTabText(i, f"DFM {dfm_id}")

        self._update_dfm_buttons()
        self._refresh_badges()

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def _new(self) -> None:
        self._current_path = None
        self._loaded_raw = {}
        self._design = None
        self._design_source = None
        self._apply_design()
        self.setWindowTitle("FLIC Config Editor")
        self._well_a_edit.clear()
        self._well_b_edit.clear()
        self._min_raw_licks_edit.clear()
        self._max_dur_edit.clear()
        self._max_events_edit.clear()
        self._transform_licks_check.setChecked(True)

        self._experiment_type_combo.blockSignals(True)
        self._experiment_type_combo.setCurrentIndex(
            max(0, self._experiment_type_combo.findData("Custom")))
        self._experiment_type_combo.blockSignals(False)
        self._last_type_index = self._experiment_type_combo.currentIndex()

        self._chamber_layout_combo.blockSignals(True)
        self._chamber_layout_combo.setCurrentIndex(
            max(0, self._chamber_layout_combo.findData("two_well")))
        self._chamber_layout_combo.blockSignals(False)
        self._last_layout_index = self._chamber_layout_combo.currentIndex()

        self._factors_widget._table.blockSignals(True)
        self._factors_widget._table.setRowCount(0)
        self._factors_widget._table.blockSignals(False)

        self._dfm_tabs.clear()
        for w in self._dfm_widgets:
            w.deleteLater()
        self._dfm_widgets.clear()

        self._global_params.reset_defaults(2)
        self._sync_layout_control()
        self._refresh_threshold_hints()
        self._update_well_names_visibility()
        self._sync_dfm_tabs(1, 2)
        ## A new config starts at the top of the dependency chain: the type
        ## decides the layout, and the layout decides the other tab's shape.
        self._tabs.setCurrentIndex(0)
        self._refresh_badges()

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
            self._load_config(cfg)
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

    def _confirm_save_with_problems(self, problems: list[str]) -> bool:
        """Name what is wrong before writing it anyway.

        Never refuses: a half-finished config is a legitimate thing to save.
        What it will not do is let a problem stay invisible because it lives
        on the tab you are not looking at.
        """
        shown = problems[:12]
        detail = "\n".join(f"  \u2022 {t}" for t in shown)
        if len(problems) > len(shown):
            detail += f"\n  \u2026 and {len(problems) - len(shown)} more"
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Save with problems?")
        box.setText(f"This configuration has {len(problems)} problem"
                    f"{'' if len(problems) == 1 else 's'}.")
        box.setInformativeText(detail + "\n\nSave it anyway?")
        box.setStandardButtons(QMessageBox.StandardButton.Save
                               | QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        return box.exec() == QMessageBox.StandardButton.Save

    def _write_yaml(self, path: Path) -> None:
        experiment, dfms = self._problems()
        problems = experiment + dfms
        if problems and not self._confirm_save_with_problems(problems):
            return
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
    sanitize_input_method_environment()
    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app, mode=ui_settings.get("theme", "auto"))
    initial_path = sys.argv[1] if len(sys.argv) > 1 else None
    win = FLICConfigEditor(initial_path=initial_path)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch()
