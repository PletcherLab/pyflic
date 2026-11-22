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

import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml
from PyQt6.QtCore import QSize, Qt, pyqtSignal
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

from . import slices as slices_mod
from .gui_env import sanitize_input_method_environment
from .ui import ActionButton, Card, Category, TopBar, apply_theme, icon, resolved_mode
from .ui.icons import help_color
from .ui import settings as ui_settings
from .ui.widgets import NoWheelComboBox, NoWheelSpinBox

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
    """Return an appropriate input widget for the given parameter key.

    Both editors host the form in a scrolling page, so the wheel never
    changes a value: it scrolls the page."""
    if key == "pi_direction":
        w = NoWheelComboBox()
        w.addItems(["left", "right"])
        w.setCurrentText(str(default))
        w.setFixedWidth(75)
        return w
    if key == "correct_for_dual_feeding":
        w = QCheckBox()
        w.setChecked(bool(default))
        return w
    w = NoWheelSpinBox()
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


def _help_button(ref: str, tooltip: str, parent: QWidget | None = None):
    """A ``?`` opening *ref*, or ``None`` when help is unavailable."""
    try:
        from ..help import HelpButton
    except Exception:  # noqa: BLE001 - help is optional, the editor is not
        return None
    return HelpButton(ref, parent, tooltip=tooltip)


def _add_help(card: Card, ref: str, tooltip: str) -> None:
    """Put a ``?`` opening *ref* in *card*'s title row, if help is available."""
    button = _help_button(ref, tooltip, card)
    if button is not None:
        card.add_title_widget(button)


def _with_help(widget: QWidget, ref: str, tooltip: str) -> QWidget:
    """*widget* with a ``?`` opening *ref* right after it, as one row — for a
    form field or a section label — or *widget* alone without help."""
    button = _help_button(ref, tooltip)
    if button is None:
        return widget
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(6)
    row.addWidget(widget)
    row.addWidget(button, 0, Qt.AlignmentFlag.AlignVCenter)
    row.addStretch(1)
    return holder


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

    def set_override(self, key: str, value: Any) -> None:
        """Override *key* with *value*, or with ``None`` stop overriding it
        (override mode).  The other overrides are left as they are."""
        widget = self._input_widgets[key]
        if value is not None:
            _set_param_value(widget, value)
        check = self._enable_checks.get(key)
        if check is not None:
            check.setChecked(value is not None)
            widget.setEnabled(value is not None)

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
# SlicesWidget
# ---------------------------------------------------------------------------

#: A Slice's own columns: (header, tooltip).
_SLICE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("Name", "The Slice's name: letters, digits, _ and -.  It names the folder "
             "slices/<name>/ its QC and results are written to."),
    ("Start (min)", "The minute the Slice starts; blank is 0."),
    ("End (min)", "The minute the Slice ends, or end (or blank) for the end of the "
                  "recording."),
    ("Own group", "exclusion_group — the remove_chambers.csv group this Slice's "
                  "removals are read from and written to.  It is the Slice's name "
                  "unless you change it, and a blank cell becomes the name.  No two "
                  "analyses share one."),
    ("Also exclude", "also_exclude — further groups whose chambers this Slice also "
                     "leaves out, comma-separated.  Nothing is inherited from the "
                     "whole recording, and nothing is written to these."),
)
#: The auto-removal cutoffs a Slice may override (``slices.AUTO_REMOVAL_CONSTANTS``),
#: one column each: (key, header, kind, tooltip).  *kind* is ``cutoff`` (a
#: number, or ``off`` for YAML null: the filter off), ``number`` or ``switch``
#: (yes / no).  Blank takes the design's own; a key with no column here is
#: kept as the file says.
_SLICE_CONSTANT_COLUMNS: tuple[tuple[str, str, str, str], ...] = (
    ("min_untransformed_licks_cutoff", "Min licks", "cutoff",
     "min_untransformed_licks_cutoff for this Slice — blank takes the design's, a "
     "number overrides it, off turns the filter off.  A Slice is shorter than the "
     "recording, so a lower cutoff is often the point."),
    ("exclude_failed_opto_chambers", "Opto failed", "switch",
     "exclude_failed_opto_chambers for this Slice — blank takes the design's; yes "
     "excludes the chambers whose optogenetic light QC fails, no keeps and flags "
     "them."),
    ("opto_unexplained_fail_fraction", "Fail fraction", "number",
     "opto_unexplained_fail_fraction for this Slice — blank takes the design's; the "
     "fraction of a linkage group's lit time, 0 to 1, that unexplained light must "
     "reach to fail the group.  Not below the design's warning fraction."),
    ("opto_unexplained_min_sec", "Unexplained (s)", "number",
     "opto_unexplained_min_sec for this Slice — blank takes the design's; the "
     "seconds of unexplained light before the optogenetic light QC judges a group."),
)
#: The optogenetic light QC's columns, shown only where the design's are.
_SLICE_OPTO_KEYS = frozenset({"exclude_failed_opto_chambers",
                              "opto_unexplained_fail_fraction", "opto_unexplained_min_sec"})
#: A Slice's keys that have a column; any other is kept as the file says, so
#: the check names it rather than the editor dropping it.
_SLICE_BODY_KEYS = frozenset({"start", "end", "exclusion_group", "also_exclude",
                              "constants"})
_SLICE_KIND_WORDS = {"cutoff": "a number, or off", "number": "a number",
                     "switch": "yes or no"}
_TOLERANCE_TIP = (
    "slice_tolerance_min — how far short of a Slice's end a recording may stop and "
    "still cover it; a member that does not cover a Slice is left out of it.  A "
    "nominal 24-hour run often ends at minute 1439.8.")


def _number_text(value: Any) -> str:
    """A number as a cell shows it, and anything else (``.inf``, a word) as
    written: ``_fmt_number`` cannot take an infinity."""
    try:
        if math.isfinite(float(value)):
            return _fmt_number(value)
    except (TypeError, ValueError):
        pass
    return str(value)


def _group_list(text: str) -> list[str]:
    """Comma-separated exclusion group names, blanks and repeats dropped."""
    out: list[str] = []
    for part in str(text or "").split(","):
        name = part.strip()
        if name and name not in out:
            out.append(name)
    return out


def _group_text(value: Any) -> str:
    """An ``also_exclude:`` value (a list, or one name) as a field shows it."""
    if value is None or value == "":
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)


def _slice_entries(node: Any) -> list[tuple[str, Any]]:
    """``slices:`` as (name, body) pairs in declared order: a mapping keyed by
    name, or the list of ``{name: ...}`` entries the loader also accepts."""
    if isinstance(node, Mapping):
        return [(str(k), v) for k, v in node.items()]
    if isinstance(node, list):
        out = []
        for item in node:
            body = dict(item) if isinstance(item, Mapping) else {}
            out.append((str(body.pop("name", "") or ""), body))
        return out
    return []


def _slice_constant_text(kind: str, value: Any) -> str:
    """A Slice's constant as its cell shows it."""
    if kind == "switch":
        state = _switch_value(value)
        if state is None:
            return "" if value is None else str(value)
        return "yes" if state else "no"
    if value is None:
        return "off" if kind == "cutoff" else ""
    return _number_text(value)


def _slice_constant(kind: str, text: str) -> tuple[Any, bool]:
    """A constant cell's text as its YAML value, and whether it could be read."""
    word = text.strip().lower()
    if kind == "switch":
        state = _switch_value(word)
        return state, state is not None
    if kind == "cutoff" and word in ("off", "none", "null"):
        return None, True
    try:
        value = float(text)
    except ValueError:
        return None, False
    if not math.isfinite(value):
        return None, False
    return slices_mod.tidy_minute(value), True


#: On a row's Name item: the name it had before the edit being made, so an
#: own group that equalled it can follow a rename.
_PREVIOUS_NAME_ROLE = Qt.ItemDataRole.UserRole + 1
#: On a row's Name item: the name the row was loaded with ("" for a row
#: added here), so the Selection's ``omit.slices`` can follow a rename.
_LOADED_NAME_ROLE = Qt.ItemDataRole.UserRole + 2


class _SliceKept:
    """A Slice's keys and constants that have no column, as the file stated
    them.  An object rather than a dict: Qt would turn a dict stored on an
    item into a sorted map and lose what it cannot convert."""

    __slots__ = ("body", "constants")

    def __init__(self, body: dict, constants: dict) -> None:
        self.body = body
        self.constants = constants


class SlicesWidget(QWidget):
    """The design's Slices (ADR-0020): one row per Slice in declared order,
    and how far short of a Slice's end a recording may stop and still cover it.

    Shared by the Project Design dialog and the Config Editor, so a Slice is
    stated in one vocabulary wherever a design is edited.  The rows are text;
    :meth:`get_values` turns them into ``slices:`` and names what it could not
    read, and ``slices.slice_problems`` — the Project's and the linter's own
    check — judges the rest.
    """

    #: Any edit: a row added, removed or typed into, or the tolerance.
    changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._locked = False
        self._lock_reason = ""
        self._refusal: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(4)

        ## Says why the table is greyed under a type that has no Slices.
        self._reason = QLabel("")
        self._reason.setObjectName("PyflicCardSubtitle")
        self._reason.setWordWrap(True)
        self._reason.setVisible(False)
        layout.addWidget(self._reason)

        headers = [*_SLICE_COLUMNS,
                   *((header, tip) for _k, header, _kind, tip in _SLICE_CONSTANT_COLUMNS)]
        self._table = QTableWidget(0, len(headers))
        for column, (text, tip) in enumerate(headers):
            item = QTableWidgetItem(text)
            item.setToolTip(tip)
            self._table.setHorizontalHeaderItem(column, item)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        for column in (0, 3, 4):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.setMinimumHeight(90)
        self._table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._table, stretch=1)

        self._controls = QWidget()
        row = QHBoxLayout(self._controls)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self._add_btn = ActionButton("Add Slice", Category.LOAD, icon_name="new")
        self._add_btn.setMaximumWidth(130)
        self._add_btn.clicked.connect(self._add_row)
        self._remove_btn = ActionButton("Remove", Category.QC, icon_name="clear")
        self._remove_btn.setMaximumWidth(110)
        self._remove_btn.clicked.connect(self._remove_row)
        row.addWidget(self._add_btn)
        row.addWidget(self._remove_btn)
        row.addStretch(1)
        ## Beside the buttons rather than a row of its own: it is one number,
        ## and it only matters once there are Slices.
        row.addWidget(QLabel("Coverage tolerance (min):"))
        self._tolerance_edit = QLineEdit()
        self._tolerance_edit.setMaximumWidth(110)
        self._tolerance_edit.setPlaceholderText(
            f"{_fmt_number(slices_mod.DEFAULT_TOLERANCE_MIN)}  (default)")
        self._tolerance_edit.textChanged.connect(lambda _text: self.changed.emit())
        row.addWidget(self._tolerance_edit)
        button = _help_button("concepts-slices#coverage",
                              "When a recording covers a Slice, and the tolerance")
        if button is not None:
            row.addWidget(button, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._controls)
        self._apply_state()

    # ---- editing ------------------------------------------------------

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """Any edit; a renamed Slice whose own group was its old name takes
        the new one too.  Left behind, the old group would still collect the
        Slice's removals under a name nothing shows, and a later Slice of the
        old name would be refused for sharing it."""
        if item.column() == 0:
            row = item.row()
            name = item.text().strip()
            previous = item.data(_PREVIOUS_NAME_ROLE) or ""
            own = self._table.item(row, 3)
            blocked = self._table.blockSignals(True)
            try:
                if own is not None and name != previous \
                        and own.text().strip() == previous:
                    own.setText(name)
                item.setData(_PREVIOUS_NAME_ROLE, name)
            finally:
                self._table.blockSignals(blocked)
        self.changed.emit()

    def renamed(self) -> dict[str, str]:
        """``{name loaded: name now}`` for every loaded Slice renamed here."""
        out: dict[str, str] = {}
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item is None:
                continue
            loaded = item.data(_LOADED_NAME_ROLE) or ""
            name = item.text().strip()
            if loaded and name and name != loaded:
                out[loaded] = name
        return out

    def follow_renames(self, omit: Any) -> Any:
        """*omit* (``global.omit``, the Selection) with each renamed Slice's
        entry in ``slices`` renamed too: a box left unticked stays unticked,
        and no entry is left naming a Slice that is no longer declared."""
        renames = self.renamed()
        if not renames or not isinstance(omit, Mapping) \
                or not isinstance(omit.get("slices"), list):
            return omit
        out = dict(omit)
        out["slices"] = [renames.get(name, name) if isinstance(name, str) else name
                         for name in omit["slices"]]
        return out

    def _add_row(self) -> None:
        row = self._append("", {})
        self._apply_state()
        self.changed.emit()
        self._table.setCurrentCell(row, 0)
        self._table.editItem(self._table.item(row, 0))

    def _remove_row(self) -> None:
        ## The last row when none is selected: under a type that has no
        ## Slices the table is greyed and cannot take a selection, and
        ## removing is then the only way to a design that saves.
        row = self._table.currentRow()
        if row < 0:
            row = self._table.rowCount() - 1
        if row >= 0:
            self._table.removeRow(row)
            self._apply_state()
            self.changed.emit()

    def _append(self, name: str, body: Any) -> int:
        """Add a row for the Slice *name* declared as *body*; return its index."""
        body = dict(body) if isinstance(body, Mapping) else {}
        constants = body.get("constants")
        stated = dict(constants) if isinstance(constants, Mapping) else {}
        columns = {key for key, *_rest in _SLICE_CONSTANT_COLUMNS}
        extra_body = {k: v for k, v in body.items() if k not in _SLICE_BODY_KEYS}
        if constants is not None and not isinstance(constants, Mapping):
            extra_body["constants"] = constants
        extra_constants = {k: v for k, v in stated.items() if k not in columns}
        start, end = body.get("start"), body.get("end")
        open_end = (end is None
                    or (isinstance(end, str) and end.strip().lower() in slices_mod.OPEN_END_WORDS)
                    or (isinstance(end, float) and math.isinf(end)))
        cells = [name,
                 "" if start is None else _number_text(start),
                 "end" if open_end else _number_text(end),
                 ## The group in effect, never a blank that silently means the
                 ## name: the own group is where this Slice's removals go.
                 str(body.get("exclusion_group") or name),
                 _group_text(body.get("also_exclude")),
                 ## Absent is blank (the design's own); a stated null is off.
                 *(_slice_constant_text(kind, stated[key]) if key in stated else ""
                   for key, _header, kind, _tip in _SLICE_CONSTANT_COLUMNS)]
        row = self._table.rowCount()
        blocked = self._table.blockSignals(True)
        self._table.setRowCount(row + 1)
        for column, text in enumerate(cells):
            self._table.setItem(row, column, QTableWidgetItem(text))
        first = self._table.item(row, 0)
        first.setData(Qt.ItemDataRole.UserRole, _SliceKept(extra_body, extra_constants))
        first.setData(_PREVIOUS_NAME_ROLE, name)
        first.setData(_LOADED_NAME_ROLE, name)
        self._table.blockSignals(blocked)
        return row

    def _cell(self, row: int, column: int) -> str:
        item = self._table.item(row, column)
        return item.text().strip() if item is not None else ""

    def _extra(self, row: int) -> "_SliceKept":
        """What row *row* was loaded with that has no column: kept on save."""
        item = self._table.item(row, 0)
        kept = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        return kept if isinstance(kept, _SliceKept) else _SliceKept({}, {})

    # ---- loading and saving -------------------------------------------

    def load_values(self, global_cfg: Mapping[str, Any] | None) -> None:
        """Fill the rows and the tolerance from a ``global:`` block."""
        g = dict(global_cfg or {})
        tolerance = g.get("slice_tolerance_min")
        self._tolerance_edit.blockSignals(True)
        self._tolerance_edit.setText("" if tolerance is None else _number_text(tolerance))
        self._tolerance_edit.blockSignals(False)
        self._table.setRowCount(0)
        for name, body in _slice_entries(g.get("slices")):
            self._append(name, body)
        self._apply_state()

    def get_values(self) -> tuple[dict[str, Any], list[str]]:
        """The ``slice_tolerance_min`` and ``slices`` keys the widget states
        (each only when it states one), and what it could not read.

        ``slices`` is a mapping keyed by name in row order: ``start`` always,
        ``end: end`` for an open end, and the optional keys only when filled.
        """
        out: dict[str, Any] = {}
        problems: list[str] = []
        text = self._tolerance_edit.text().strip()
        if text:
            value, ok = _slice_constant("number", text)
            if ok and value >= 0:
                out["slice_tolerance_min"] = value
            else:
                problems.append(f"slice_tolerance_min: the coverage tolerance is a number "
                                f"of minutes, 0 or more, not {text!r}")
        from .opto_qc import opto_constant_problems

        declared: dict[str, Any] = {}
        for row in range(self._table.rowCount()):
            cells = [self._cell(row, column) for column in range(self._table.columnCount())]
            name, start, end, own, others = cells[:5]
            extra = self._extra(row)
            if not name:
                if any(cells) or extra.body or extra.constants:
                    problems.append(f"slices: row {row + 1} has no name")
                continue
            where = f"slices.{name}"
            if name in declared:
                problems.append(f"{where}: two Slices are named '{name}'")
                continue
            ## A number stays a number, anything else is written as typed so
            ## the check names it ("start must be a minute, not 'six'").
            body: dict[str, Any] = {
                "start": _number_or_text(start) if start else 0,
                "end": "end" if end.lower() in slices_mod.OPEN_END_WORDS
                       else _number_or_text(end),
            }
            ## Always written: a Slice's own group is stated, not implied.
            body["exclusion_group"] = own or name
            groups = _group_list(others)
            if groups:
                body["also_exclude"] = groups
            constants = dict(extra.constants)
            for (key, header, kind, _tip), text in zip(_SLICE_CONSTANT_COLUMNS, cells[5:]):
                if not text:
                    continue
                value, ok = _slice_constant(kind, text)
                if not ok:
                    problems.append(f"{where}: {header} is {_SLICE_KIND_WORDS[kind]}, "
                                    f"not {text!r}")
                    continue
                constants[key] = value
            problems += [f"{where}: {p}" for p in opto_constant_problems(
                {k: v for k, v in constants.items() if k in _SLICE_OPTO_KEYS})]
            if constants:
                body["constants"] = constants
            for key, value in extra.body.items():
                body.setdefault(key, value)
            declared[name] = body
        if declared:
            out["slices"] = declared
        return out, problems

    # ---- what the editor around it decides ----------------------------

    def set_read_only(self, read_only: bool, *, reason: str = "") -> None:
        """Show the Slices but refuse edits — a Member inherits them."""
        self._locked = bool(read_only)
        self._lock_reason = reason if read_only else ""
        self._apply_state()

    def set_refusal(self, reason: str | None) -> None:
        """Grey the Slices out with *reason* (a type that has none), or
        ``None`` to allow them."""
        self._refusal = reason
        self._apply_state()

    def set_opto_columns_visible(self, visible: bool) -> None:
        """Show the optogenetic light QC's columns only where the design's
        own optogenetic settings are shown.  Presentation only: a hidden
        column still keeps what it holds."""
        for column, (key, *_rest) in enumerate(_SLICE_CONSTANT_COLUMNS, len(_SLICE_COLUMNS)):
            if key in _SLICE_OPTO_KEYS:
                self._table.setColumnHidden(column, not visible)

    def _apply_state(self) -> None:
        refused = self._refusal is not None
        has_rows = self._table.rowCount() > 0
        editable = not self._locked and not refused
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.AllEditTriggers if editable
            else QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setEnabled(not refused)
        ## Under a type with no Slices an empty table is only noise; one that
        ## still holds Slices stays, greyed, so they can be removed.
        self._table.setVisible(not refused or has_rows)
        self._controls.setVisible(not refused or has_rows)
        self._add_btn.setEnabled(editable)
        self._remove_btn.setEnabled(not self._locked and has_rows)
        self._tolerance_edit.setEnabled(editable)
        for widget in (self._table, self._add_btn, self._remove_btn):
            widget.setToolTip(self._lock_reason)
        self._tolerance_edit.setToolTip(self._lock_reason or _TOLERANCE_TIP)
        self._reason.setVisible(refused)
        if refused:
            self._reason.setText(
                f"{self._refusal}." + ("  Remove these Slices to save." if has_rows else ""))


# ---------------------------------------------------------------------------
# DFMWidget
# ---------------------------------------------------------------------------

#: The ``dfms:`` entry keys a DFM tab writes (or, cleared, leaves out); any
#: other key it loaded is kept as the file says.
_MANAGED_DFM_KEYS = frozenset({"id", "optogenetics", "params", "parameters",
                               "chambers", "Chambers", "paired_chambers",
                               "chamber_groups"})


def _same_path(a, b) -> bool:
    """Whether two paths name one file or folder, on a share that ignores
    case as on one that does not."""
    return os.path.normcase(os.path.abspath(str(a))) == \
        os.path.normcase(os.path.abspath(str(b)))


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


def _number_or_text(text: str) -> Any:
    """*text* as an int when it is a whole number and a float otherwise — or
    the text itself when it is no number at all, so the type's validation
    names the mistake instead of the value silently vanishing."""
    try:
        value = float(text)
    except ValueError:
        return text
    return int(value) if value.is_integer() else value


#: The ``optogenetics:`` choices: (label, the value written to the yaml).
#: ``auto`` is the default and is written as nothing.
_OPTO_CHOICES: tuple[tuple[str, Any], ...] = (
    ("auto — Program.txt or light data", "auto"),
    ("yes — every DFM", True),
    ("no — none", False),
)


def _opto_index(value: Any, *, inherit: bool = False) -> int:
    """The combo index for an ``optogenetics:`` value; with *inherit* the first
    entry is "inherit" and an unset value selects it."""
    from .opto_program import SETTING_AUTO, SETTING_YES, normalize_setting

    offset = 1 if inherit else 0
    if value is None:
        return 0
    try:
        setting = normalize_setting(value)
    except ValueError:
        return 0
    if setting == SETTING_AUTO:
        return offset
    return offset + (1 if setting == SETTING_YES else 2)


def _switch_value(value: Any) -> bool | None:
    """A yaml switch as True / False, or None when unset or unreadable."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "yes", "on", "1"):
            return True
        if s in ("false", "no", "off", "0"):
            return False
    return None


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

#: A Progressive Ratio DFM's Chamber Groups, as its tab picks them: pairs
#: (``paired_chambers``), single chambers (``chamber_groups: single``), or
#: Program.txt's Linkage (neither key written).
PR_GROUPS_PAIRED, PR_GROUPS_SINGLE, PR_GROUPS_PROGRAM = "paired", "single", "program"


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
        *,
        simple: bool = False,
    ) -> None:
        """*simple* shows only what the design is — the DFM's id and its
        chamber assignments (and, for Progressive Ratio, its paired chambers)
        — for the Experiment Editor.  The optogenetics and parameter
        overrides it hides are still loaded and saved as the file has them."""
        super().__init__(parent)
        self._chamber_size = chamber_size
        self._factor_levels: dict[str, list[str]] = {}
        #: The keys of the loaded ``dfms:`` entry this tab has no field for.
        self._unmanaged: dict[str, Any] = {}

        outer = QVBoxLayout(self)
        outer.setAlignment(Qt.AlignmentFlag.AlignTop)
        outer.setSpacing(8)
        outer.setContentsMargins(8, 8, 8, 8)

        # -- DFM ID --------------------------------------------------------
        id_card = Card("DFM Identity", Category.NEUTRAL)
        _add_help(id_card, "app-config-editor#dfm-identity",
                  "The DFM's id, and its own optogenetics setting")
        id_inner = QHBoxLayout()
        id_inner.setContentsMargins(0, 0, 0, 0)
        id_inner.addWidget(QLabel("DFM ID:"))
        ## Each DFM tab scrolls: the wheel scrolls it, never a choice on it.
        self._id_spin = NoWheelSpinBox()
        self._id_spin.setRange(1, 99)
        self._id_spin.setValue(dfm_id)
        self._id_spin.setMaximumWidth(80)
        id_inner.addWidget(self._id_spin)
        id_inner.addSpacing(18)
        ## Per-DFM optogenetics: a physical fact about this DFM (was its lid
        ## lit?), so it stays free inside a Project, like the rest of dfms:.
        self._opto_label = QLabel("Optogenetics:")
        id_inner.addWidget(self._opto_label)
        self._opto_combo = NoWheelComboBox()
        self._opto_combo.addItem("inherit", None)
        for label, value in _OPTO_CHOICES:
            self._opto_combo.addItem(label.split(" —")[0], value)
        self._opto_combo.setToolTip(
            "optogenetics for this DFM alone — inherit takes the experiment's "
            "setting; auto runs the light QC when data/Program.txt has a section "
            "for this DFM or its LEDs were ever lit; yes always; no never.")
        self._opto_combo.setMaximumWidth(110)
        id_inner.addWidget(self._opto_combo)
        id_inner.addStretch()
        id_card.add_body(id_inner)
        outer.addWidget(id_card)

        # -- Chamber table -------------------------------------------------
        n_chambers = 12 if chamber_size == 1 else 6
        self._ch_card = Card("Chamber → Treatment Assignments", Category.LOAD)
        _add_help(self._ch_card, "app-config-editor#chamber-assignments",
                  "Assigning treatments, and factor levels, to chambers")
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
        ## pi_direction override above says which side it is on.  The groups
        ## themselves are pairs, single chambers (chamber_groups: single), or
        ## whatever Program.txt's Linkage makes (neither key written).
        from .experiment_types.progressive_ratio import CHAMBER_GROUPS
        self._pr_enabled = False
        self._pr_card = Card(
            "Paired chamber per chamber group", Category.LOAD,
            subtitle="Progressive Ratio: the other chamber of each group is "
                     "yoked. Well A is the sucrose well; PI Direction is its side.")
        try:
            from ..help import HelpButton

            self._pr_card.add_title_widget(HelpButton(
                "concepts-progressive-ratio#chamber-groups-paired-and-yoked",
                self._pr_card, tooltip="Chamber groups, paired and yoked"))
        except Exception:  # noqa: BLE001 - help is optional, the editor is not
            pass
        pr_row = QHBoxLayout()
        pr_row.setContentsMargins(0, 0, 0, 0)
        self._groups_combo = NoWheelComboBox()
        for label, value in (("Pairs", PR_GROUPS_PAIRED), ("Single chambers", PR_GROUPS_SINGLE),
                             ("From Program.txt", PR_GROUPS_PROGRAM)):
            self._groups_combo.addItem(label, value)
        self._groups_combo.setToolTip(
            "Pairs: chambers 1+2, 3+4, 5+6, one paired and one yoked (paired_chambers).  "
            "Single chambers: each chamber its own group, with no yoked control "
            "(chamber_groups: single).  From Program.txt: the groups its linkage makes.")
        self._groups_combo.currentIndexChanged.connect(self._sync_groups_mode)
        pr_row.addWidget(QLabel("Groups:"))
        pr_row.addWidget(self._groups_combo)
        self._paired_combos: list[QComboBox] = []
        for g, (a, b) in CHAMBER_GROUPS.items():
            pr_row.addWidget(QLabel(f"Group {g} ({a}+{b}):"))
            combo = NoWheelComboBox()
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
        _add_help(over_card, "app-config-editor#parameter-overrides",
                  "Overriding a detection parameter for this DFM alone")
        ## Two columns, not three.  Three needed 888px inside a page that has
        ## ~877 once the West tab bar and a scrollbar are taken out, so the
        ## whole tab scrolled sideways — much worse than the 54px of extra
        ## height two columns cost.
        self._params_form = ParamsForm(override_mode=True, chamber_size=chamber_size,
                                       num_columns=2)
        over_card.add_body(self._params_form)
        outer.addWidget(over_card)
        if simple:
            for widget in (self._opto_label, self._opto_combo, over_card):
                widget.setVisible(False)

        # -- PI Direction (the Experiment Editor) ----------------------------
        ## The one per-DFM parameter the Experiment Editor shows.  Which side
        ## well A is on is a fact of how this DFM was set up, so it may differ
        ## from DFM to DFM even inside a Project (PHYSICAL_DFM_KEYS).  It is
        ## the override form's pi_direction, so loading and saving are the
        ## form's.  The Config Editor shows the whole form instead.
        self._simple = simple
        self.side_card = Card(
            "PI Direction", Category.LOAD,
            subtitle="The side well A is on in this DFM's chambers, whose "
                     "preference index is 1. inherit takes the experiment's "
                     "setting: left, unless it states another.")
        side_row = QHBoxLayout()
        side_row.setContentsMargins(0, 0, 0, 0)
        side_row.addWidget(QLabel(_PARAM_LABELS["pi_direction"] + ":"))
        self.side_combo = NoWheelComboBox()
        self.side_combo.addItem("inherit", None)
        self.side_combo.addItem("left", "left")
        self.side_combo.addItem("right", "right")
        self.side_combo.setFixedWidth(90)
        self.side_combo.currentIndexChanged.connect(self._side_chosen)
        side_row.addWidget(self.side_combo)
        side_help = _param_help_button("pi_direction", self)
        if side_help is not None:
            side_row.addWidget(side_help)
        side_row.addStretch()
        self.side_card.add_body(side_row)
        outer.addWidget(self.side_card)
        self._show_side_card()

    # ------------------------------------------------------------------

    def pi_direction(self) -> str | None:
        """This DFM's own ``pi_direction``, or ``None`` when it inherits."""
        return self._params_form.get_values().get("pi_direction")

    def set_pi_direction(self, side: str | None) -> None:
        """Give this DFM its own ``pi_direction`` (``left`` or ``right``), or
        ``None`` to inherit the experiment's."""
        self._params_form.set_override("pi_direction", side)
        self._sync_side_combo()

    def _side_chosen(self, *_args) -> None:
        self._params_form.set_override("pi_direction", self.side_combo.currentData())

    def _sync_side_combo(self) -> None:
        side = self.pi_direction()
        index = 0 if side is None else self.side_combo.findData(str(side))
        self.side_combo.blockSignals(True)
        self.side_combo.setCurrentIndex(max(index, 0))
        self.side_combo.blockSignals(False)

    def _show_side_card(self) -> None:
        """Shown in the Experiment Editor, for two wells: one well has no side."""
        self.side_card.setVisible(self._simple and self._chamber_size == 2)

    def set_progressive_ratio(self, enabled: bool) -> None:
        """Show the paired-chamber pickers (Progressive Ratio) or hide them."""
        self._pr_enabled = bool(enabled)
        self._pr_card.setVisible(self._pr_enabled)

    def paired_chambers(self) -> list[int]:
        return [int(c.currentData()) for c in self._paired_combos]

    def set_paired_chambers(self, chambers) -> None:
        self.set_groups_mode(PR_GROUPS_PAIRED)
        for combo in self._paired_combos:
            for i in range(combo.count()):
                if int(combo.itemData(i)) in {int(c) for c in (chambers or [])}:
                    combo.setCurrentIndex(i)
                    break

    def groups_mode(self) -> str:
        """``paired``, ``single`` or ``program``: what the Chamber Groups are."""
        return str(self._groups_combo.currentData())

    def set_groups_mode(self, mode: str) -> None:
        index = self._groups_combo.findData(mode)
        if index >= 0:
            self._groups_combo.setCurrentIndex(index)
        self._sync_groups_mode()

    def _sync_groups_mode(self, *_args) -> None:
        """The paired-chamber pickers mean something only for pairs."""
        for combo in self._paired_combos:
            combo.setEnabled(self.groups_mode() == PR_GROUPS_PAIRED)

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
        if (self._pr_enabled and self._chamber_table.rowCount() >= 6
                and self.groups_mode() == PR_GROUPS_PAIRED):
            ## Both chambers of a chamber group share one treatment.  Single
            ## chambers share nothing, and Program.txt's pairs are checked
            ## against it at load.
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
        self._show_side_card()

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

    def optogenetics(self) -> Any:
        """This DFM's ``optogenetics:`` value, or ``None`` to inherit."""
        return self._opto_combo.currentData()

    def get_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"id": self._id_spin.value()}
        opto = self.optogenetics()
        if opto is not None:
            result["optogenetics"] = opto

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
        if self._pr_enabled and self.groups_mode() == PR_GROUPS_PAIRED:
            result["paired_chambers"] = self.paired_chambers()
        elif self._pr_enabled and self.groups_mode() == PR_GROUPS_SINGLE:
            result["chamber_groups"] = PR_GROUPS_SINGLE
        if chambers:
            result["chambers"] = chambers
        ## A key this tab has no field for (well_names, a note) was written
        ## by hand; keep it.  A key it does manage is written above or, when
        ## cleared, not at all.
        for key, value in self._unmanaged.items():
            result.setdefault(key, value)

        return result

    def load_dict(self, data: dict[str, Any], chamber_size: int) -> None:
        self._unmanaged = {k: v for k, v in data.items() if k not in _MANAGED_DFM_KEYS}
        self._id_spin.setValue(int(data.get("id", self._id_spin.value())))
        self._opto_combo.setCurrentIndex(_opto_index(data.get("optogenetics"),
                                                     inherit=True))
        if data.get("paired_chambers") is not None:
            self.set_paired_chambers(data.get("paired_chambers"))
        elif str(data.get("chamber_groups") or "").strip().lower() == PR_GROUPS_SINGLE:
            self.set_groups_mode(PR_GROUPS_SINGLE)
        else:
            ## Neither key: the groups are Program.txt's (as the loader takes
            ## them), so saving must not write the pickers' defaults.
            self.set_groups_mode(PR_GROUPS_PROGRAM)

        params_raw = data.get("params", data.get("parameters", {})) or {}
        self._params_form.load_values(dict(params_raw), chamber_size)
        self._sync_side_combo()

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


#: The ``global:`` keys the Experiment tab writes (or, cleared, leaves out).
#: ``well_names`` is one only for a two-well layout, the one with fields for
#: it; every other key the file states is kept on save.
_MANAGED_GLOBAL_KEYS = frozenset({
    "experiment_type", "chamber_layout", "params", "parameters", "transform_licks",
    "optogenetics", "facet_source", "constants", "experimental_design_factors",
    "exclusion_group", "also_exclude", "slice_tolerance_min", "slices", "omit",
    "control_level",
})


class FLICConfigEditor(QMainWindow):
    """Main window for the FLIC Config Editor."""

    #: The path of a config just written, so the Hub can re-judge the
    #: Member's saved results against it (ADR-0022).
    saved = pyqtSignal(str)

    def __init__(self, initial_path: str | Path | None = None) -> None:
        super().__init__()
        self._current_path: Path | None = None
        #: The file ``_loaded_raw`` was read from: Save As may write elsewhere.
        self._loaded_path: Path | None = None
        #: Where Save As starts when nothing was loaded: the file or folder
        #: the editor was opened on.
        self._save_hint: Path | None = None
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
        ## F1 is the shortcut above.  Binding it on this action as well made
        ## the key ambiguous, and Qt then fires neither; the menu only shows it.
        act_this = QAction(icon("info"), "Config Editor &help\tF1", self)
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
        _add_help(exp_card, "app-config-editor#experiment-settings",
                  "The Experiment tab's settings, field by field")
        self._exp_form = QFormLayout()
        self._exp_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self._exp_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        ## Where the recording is read (ADR-0021): the top-level data_folder:
        ## key, this experiment's own, so a Project's design never locks it.
        ## Blank is the experiment's own data/, and then no key is written.
        self._data_folder_edit = QLineEdit()
        self._data_folder_edit.setPlaceholderText("data/  (this folder's own)")
        self._data_folder_edit.setToolTip(
            "data_folder — the folder this experiment's recording is read from, "
            "when it is not the data/ beside this file: relative to this folder, "
            "with forward slashes, for example "
            "../../../../../runs/flic/2026/R-20261020-asmith-01/data.  pyflic only "
            "reads it; every output still goes in this folder.  Leave it blank to "
            "read data/.")
        self._data_folder_edit.textChanged.connect(self._refresh_opto_rows)
        self._exp_form.addRow("Data Folder:", _with_help(
            self._data_folder_edit, "config-structure#data-folder",
            "Reading the recording from another folder"))

        ## The Experiment Type comes from the registry, so adding a type
        ## reaches this menu with no edit here — and so the retired layout
        ## names (two_well, single_well) cannot reappear in it (ADR-0007).
        ## A lab's own types are listed too, and New type… builds one in the
        ## Experiment Type Editor (ADR-0019).
        ## The page scrolls; the wheel scrolls it and never changes a choice
        ## here — a type or layout changed by a passing pointer reshapes the
        ## whole config.
        self._experiment_type_combo = NoWheelComboBox()
        self._fill_type_combo()
        self._experiment_type_combo.setMaximumWidth(240)
        self._last_type_index = self._experiment_type_combo.currentIndex()
        self._experiment_type_combo.currentIndexChanged.connect(
            self._on_experiment_type_changed)
        self._new_type_btn = QPushButton("New type…")
        self._new_type_btn.setIcon(icon("add", category=Category.LOAD))
        self._new_type_btn.setToolTip(
            "Open the Experiment Type Editor to build a type for your lab; it is "
            "listed here once saved.")
        self._new_type_btn.clicked.connect(lambda _checked=False: self._open_type_editor())
        type_row = QHBoxLayout()
        type_row.setContentsMargins(0, 0, 0, 0)
        type_row.addWidget(self._experiment_type_combo)
        type_row.addWidget(self._new_type_btn)
        type_help = _help_button("concepts-experiment-types",
                                 "What an Experiment Type decides")
        if type_help is not None:
            type_row.addWidget(type_help)
        type_row.addStretch(1)
        self._exp_form.addRow("Experiment Type:", type_row)

        ## Shown even when the type owns it: the layout is what decides
        ## whether the other tab has 6 chambers or 12, and a derived value
        ## that drives visible structure elsewhere should be readable.
        self._chamber_layout_combo = NoWheelComboBox()
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

        ## The Treatment control's level (ADR-0019), shown for a type that
        ## compares every level of a factor with a control.
        self._control_level_edit = QLineEdit()
        self._control_level_edit.setMaximumWidth(200)
        self._control_level_edit.setToolTip(
            "control_level — the level of the type's control factor every other "
            "level is compared with.")
        self._control_level_row = self._exp_form.rowCount()
        self._exp_form.addRow("Control Level:", _with_help(
            self._control_level_edit, "concepts-experiment-types#control-comparisons",
            "The Treatment control every other level is compared with"))

        ## Whether the optogenetic light QC runs (auto | yes | no); a DFM tab
        ## can override it for that DFM.
        self._opto_combo = NoWheelComboBox()
        for label, value in _OPTO_CHOICES:
            self._opto_combo.addItem(label, value)
        self._opto_combo.setMaximumWidth(260)
        self._opto_combo.setToolTip(
            "optogenetics — auto (the default) runs the optogenetic light QC on "
            "every DFM with a section in data/Program.txt or a lit LED in its "
            "data; yes on every DFM, and a DFM with no light at all fails; no "
            "never.  Any experiment type.")
        self._opto_combo.currentIndexChanged.connect(self._refresh_opto_rows)
        self._opto_combo.currentIndexChanged.connect(self._refresh_badges)
        self._exp_form.addRow("Optogenetics:", _with_help(
            self._opto_combo, "config-structure#globaloptogenetics",
            "auto, yes or no: when the optogenetic light QC runs"))

        ## Where the Facets come from (facet_source, decision 7): minute
        ## cutoffs, or each DFM's Program Intervals from data/Program.txt.  A
        ## type that owns its Facets owns this too (_sync_facet_source).
        from .program_facets import FACET_SOURCES, SOURCE_LABELS

        self._facet_source_combo = NoWheelComboBox()
        for value in FACET_SOURCES:
            self._facet_source_combo.addItem(SOURCE_LABELS[value], value)
        self._facet_source_combo.setMaximumWidth(260)
        self._facet_source_combo.setToolTip(
            "facet_source — minute cutoffs (the default) split every DFM at the "
            "same minutes; Program Intervals give one Facet per Program Interval "
            "of each DFM's data/Program.txt section, labelled by its Paradigm "
            "(Cycles pooled, or one per Cycle).")
        self._facet_source_combo.currentIndexChanged.connect(self._refresh_badges)
        self._exp_form.addRow("Facet source:", _with_help(
            self._facet_source_combo, "concepts-facets#program-facets",
            "Facets from minute cutoffs, or from Program.txt"))

        # Auto-filter thresholds (used by auto_remove_chambers)
        filter_header = QLabel("Auto-filter Thresholds  (used by auto_remove_chambers)")
        filter_header.setObjectName("PyflicSectionDivider")
        self._filter_header_row = self._exp_form.rowCount()
        self._exp_form.addRow(_with_help(
            filter_header, "app-config-editor#auto-filter-thresholds",
            "The Use box and value of each auto-removal cutoff"))

        ## Each cutoff is optional: unchecked writes YAML null (filter off);
        ## checked + blank inherits the type's default (ADR-0011); checked +
        ## a number overrides it.  Placeholders come from default_constants.
        self._min_raw_licks_check, self._min_raw_licks_edit, min_row = (
            self._make_threshold_control())
        self._min_raw_licks_row = self._exp_form.rowCount()
        self._exp_form.addRow("Min Untransformed Licks:", min_row)

        self._max_dur_check, self._max_dur_edit, dur_row = (
            self._make_threshold_control())
        self._max_dur_row = self._exp_form.rowCount()
        self._exp_form.addRow("Max Median Duration:", dur_row)

        self._max_events_check, self._max_events_edit, events_row = (
            self._make_threshold_control())
        self._max_events_row = self._exp_form.rowCount()
        self._exp_form.addRow("Max Events:", events_row)

        ## Every type's, both layouts (pyflic.base.ceiling).
        from .ceiling import CEILING_CONSTANT_FIELDS

        self._ceiling_check, self._ceiling_edit, ceiling_row = (
            self._make_threshold_control())
        self._ceiling_edit.setToolTip(CEILING_CONSTANT_FIELDS[0].tooltip)
        self._ceiling_row = self._exp_form.rowCount()
        self._exp_form.addRow("Max Minutes at Ceiling:", ceiling_row)

        ## Progressive Ratio's own constants — the switches that decide which
        ## failed chamber groups leave, the light QC's thresholds and the
        ## breaking point's rule — shown only for a type that carries them.
        ## The same contract as the cutoffs above (ADR-0011): the type's
        ## default is the placeholder, and only what the experimenter types or
        ## picks reaches ``constants:``.  The rows come from the description
        ## the Project Design dialog and the type's validation also read.
        from .experiment_types.progressive_ratio import PR_CONSTANT_FIELDS

        pr_header = QLabel("Progressive Ratio  (light QC and breaking point)")
        pr_header.setObjectName("PyflicSectionDivider")
        self._pr_header_row = self._exp_form.rowCount()
        self._exp_form.addRow(_with_help(
            pr_header, "app-config-editor#progressive-ratio-settings",
            "The Progressive Ratio light QC and breaking point settings"))
        self._pr_constant_widgets: dict[str, QWidget] = {}
        self._pr_constant_rows: dict[str, int] = {}
        for field in PR_CONSTANT_FIELDS:
            if field.group == "switch":
                widget = NoWheelComboBox()
                widget.addItem("default", None)
                widget.addItem(field.choices[0], True)
                widget.addItem(field.choices[1], False)
                widget.currentIndexChanged.connect(self._refresh_badges)
                label = field.short_label or field.label
            else:
                widget = QLineEdit()
                widget.textChanged.connect(self._refresh_badges)
                label = field.label
            widget.setMaximumWidth(260)
            widget.setToolTip(field.tooltip)
            self._pr_constant_rows[field.key] = self._exp_form.rowCount()
            self._exp_form.addRow(f"{label}:", widget)
            self._pr_constant_widgets[field.key] = widget

        ## The optogenetic light QC's constants, every type's: shown when the
        ## experiment is optogenetic (optogenetics yes, or a Program.txt in
        ## data/, or a run that already found light), with the same contract
        ## as the rows above — defaults as placeholders, typed values saved.
        from .opto_qc import DEFAULT_CONSTANTS as _OPTO_DEFAULTS
        from .opto_qc import OPTO_CONSTANT_FIELDS

        opto_header = QLabel("Optogenetic light QC  (every experiment type)")
        opto_header.setObjectName("PyflicSectionDivider")
        self._opto_header_row = self._exp_form.rowCount()
        self._exp_form.addRow(_with_help(
            opto_header, "concepts-optogenetics#thresholds",
            "What each optogenetic light QC threshold does"))
        self._opto_constant_widgets: dict[str, QWidget] = {}
        self._opto_constant_rows: dict[str, int] = {}
        for field in OPTO_CONSTANT_FIELDS:
            default = _OPTO_DEFAULTS.get(field.key)
            if field.group == "switch":
                widget = NoWheelComboBox()
                widget.addItem("default: " + field.choices[0 if default else 1], None)
                widget.addItem(field.choices[0], True)
                widget.addItem(field.choices[1], False)
                widget.currentIndexChanged.connect(self._refresh_badges)
                label = field.short_label or field.label
            else:
                widget = QLineEdit()
                widget.setPlaceholderText(f"{_fmt_number(default)}  (default)")
                widget.textChanged.connect(self._refresh_badges)
                label = field.label
            widget.setMaximumWidth(260)
            widget.setToolTip(field.tooltip)
            self._opto_constant_rows[field.key] = self._exp_form.rowCount()
            self._exp_form.addRow(f"{label}:", widget)
            self._opto_constant_widgets[field.key] = widget

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
        _add_help(global_card, "reference-parameters",
                  "Every detection parameter, and which way it moves results")
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
        _add_help(factors_card, "config-factors",
                  "Factorial designs: factors, levels and chamber assignments")
        self._factors_widget = FactorsWidget()
        factors_card.add_body(self._factors_widget)
        top_layout.addWidget(factors_card)

        ## Which remove_chambers.csv groups the whole recording reads, and the
        ## Slices (ADR-0020) with their own.  One card: a Slice's own group
        ## may not be the whole recording's, so the two are edited together.
        slices_card = Card(
            "Exclusion Groups and Slices",
            Category.ANALYZE,
            subtitle=(
                "Optional. A Slice is a separate analysis of part of the recording, "
                "with its own exclusions, QC and results; the whole recording is "
                "always analysed as well."
            ),
        )
        _add_help(slices_card, "concepts-slices#slices-in-the-design",
                  "Slices: separate analyses of part of the recording")
        exclusion_form = QFormLayout()
        exclusion_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        exclusion_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._exclusion_group_edit = QLineEdit()
        self._exclusion_group_edit.setPlaceholderText("general")
        self._exclusion_group_edit.setMaximumWidth(200)
        self._exclusion_group_edit.setToolTip(
            "exclusion_group — the whole recording's own remove_chambers.csv group: "
            "the one its removals are read from and written to.")
        self._exclusion_group_edit.textChanged.connect(self._refresh_badges)
        exclusion_form.addRow("Exclusion Group:", _with_help(
            self._exclusion_group_edit, "config-structure#globalexclusion-group",
            "The remove_chambers.csv group the whole recording is read under"))
        self._also_exclude_edit = QLineEdit()
        self._also_exclude_edit.setPlaceholderText("e.g. dead  (comma-separated)")
        self._also_exclude_edit.setMaximumWidth(200)
        self._also_exclude_edit.setToolTip(
            "also_exclude — further groups whose chambers the whole recording also "
            "leaves out; it never writes to them.  A fly dead from the start, listed "
            "once in such a group, is left out of every analysis that names it.")
        self._also_exclude_edit.textChanged.connect(self._refresh_badges)
        exclusion_form.addRow("Also Exclude:", _with_help(
            self._also_exclude_edit, "concepts-slices#exclusions",
            "Each analysis's own group, and the groups it also excludes"))
        slices_card.add_body(exclusion_form)
        #: Their own tooltips, given back when a Project's design stops owning them.
        self._exclusion_tips = {edit: edit.toolTip() for edit in (
            self._exclusion_group_edit, self._also_exclude_edit)}
        self._slices_widget = SlicesWidget()
        self._slices_widget.changed.connect(self._refresh_badges)
        slices_card.add_body(self._slices_widget)
        top_layout.addWidget(slices_card)
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
        self._sync_control_level()
        self._sync_slices()
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

    def _type_where(self) -> Path | None:
        """The directory of the file being edited: a lab type is read from
        the copy beside it (or beside its project.yaml) first (ADR-0019)."""
        return self._current_path.parent if self._current_path is not None else None

    def _fill_type_combo(self, select: str | None = None) -> None:
        """List the types this config can name, keeping *select* (or the
        current choice) selected when it is still there."""
        from . import experiment_types

        combo = self._experiment_type_combo
        current = select or combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for item in experiment_types.available_experiment_types(self._type_where()):
            label = item.display_name if item.is_shipped else f"{item.display_name}  (lab)"
            combo.addItem(label, item.name)
        index = combo.findData(current) if current else -1
        combo.setCurrentIndex(max(index, 0))
        combo.blockSignals(False)

    def _reload_types(self, select: str | None = None) -> None:
        """Re-read the type list after the Experiment Type Editor saved or
        deleted one; a type it just saved is chosen."""
        before = self._experiment_type_combo.currentData()
        self._fill_type_combo(select or None)
        if self._experiment_type_combo.currentData() != before:
            self._on_experiment_type_changed(self._experiment_type_combo.currentIndex())

    def _open_type_editor(self):
        """The Experiment Type Editor on a new type."""
        from .type_editor import open_type_editor

        editor = open_type_editor(self)
        editor.typesChanged.connect(self._reload_types)
        editor.closed.connect(lambda: self._reload_types())
        return editor

    def _sync_control_level(self) -> None:
        """Show the control level only for a type with a Treatment control."""
        item = self._current_type()
        factor = item.definition.treatment_control if item.definition is not None else None
        self._exp_form.setRowVisible(self._control_level_row, factor is not None)
        if factor is not None:
            self._control_level_edit.setPlaceholderText(f"a level of {factor}")

    def _sync_slices(self) -> None:
        """Grey the Slices out, with the reason, under a type whose Facets
        come from the data (the Ratio analysis): it has no Slices."""
        item = self._current_type()
        self._slices_widget.set_refusal(
            None if slices_mod.allows_slices(item) else slices_mod.refusal(item))

    def _current_type(self):
        """The selected ``ExperimentType``, Custom if anything is amiss."""
        from . import experiment_types

        try:
            return experiment_types.get_experiment_type(
                self._experiment_type_combo.currentData(), self._type_where())
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
        ## The Facet source answers to the same two authorities.
        self._sync_facet_source()

    def _sync_facet_source(self) -> None:
        """Fix the Facet source at cutoffs for a type that owns its Facets
        (Progressive Ratio), and lock it under a Project Design."""
        item = self._current_type()
        owned = item.owns_facet_source
        if owned:
            self._facet_source_combo.blockSignals(True)
            self._facet_source_combo.setCurrentIndex(
                max(self._facet_source_combo.findData("cutoffs"), 0))
            self._facet_source_combo.blockSignals(False)
        self._facet_source_combo.setEnabled(not owned and not self._design)
        if owned:
            self._facet_source_combo.setToolTip(
                f"The {item.display_name} experiment type owns its Facets.")
        elif self._design:
            self._facet_source_combo.setToolTip(
                f"Owned by the Project design in {self._design_source}")

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
        self._sync_control_level()
        self._sync_slices()
        self._refresh_badges()

    def _sync_pr_widgets(self) -> None:
        """Show the paired-chamber pickers on every DFM tab for Progressive
        Ratio and hide them otherwise."""
        enabled = self._current_type().has_module("ratio")
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

    def _make_threshold_control(self) -> tuple[QCheckBox, QLineEdit, QWidget]:
        """Checkbox + value field for one optional auto-filter cutoff."""
        check = QCheckBox("Use")
        check.setToolTip(
            "When checked, auto-removal applies this cutoff (blank value = "
            "type default).  When unchecked, this filter is off.")
        edit = QLineEdit()
        edit.setMaximumWidth(200)
        edit.setEnabled(False)
        check.toggled.connect(edit.setEnabled)
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(check)
        layout.addWidget(edit, 1)
        return check, edit, row

    def _threshold_controls(
            self) -> tuple[tuple[QCheckBox, QLineEdit, str], ...]:
        return (
            (self._min_raw_licks_check, self._min_raw_licks_edit,
             "min_untransformed_licks_cutoff"),
            (self._max_dur_check, self._max_dur_edit,
             "max_med_duration_cutoff"),
            (self._max_events_check, self._max_events_edit,
             "max_events_cutoff"),
            (self._ceiling_check, self._ceiling_edit,
             "max_ceiling_minutes"),
        )

    def _threshold_fields(self) -> tuple[tuple[QLineEdit, str], ...]:
        return tuple((edit, key) for _check, edit, key in self._threshold_controls())

    def _load_threshold_controls(self, constants: dict[str, Any]) -> None:
        """Fill the optional cutoff rows from a ``constants:`` block.

        ``null`` means the filter is off (checkbox unchecked).  An omitted key
        means inherit the type's default when it has one (checkbox checked,
        value blank).  A number is an override (checked, value shown).
        """
        defaults = self._current_type().default_constants or {}
        for check, edit, key in self._threshold_controls():
            if key in constants:
                value = constants[key]
                enabled = value is not None
                text = "" if value is None else _fmt_number(value)
            else:
                enabled = defaults.get(key) is not None
                text = ""
            check.blockSignals(True)
            check.setChecked(enabled)
            check.blockSignals(False)
            edit.setText(text)
            edit.setEnabled(enabled and check.isEnabled())

    def _refresh_threshold_hints(self) -> None:
        """Show the type's default as placeholder text, never as a value.

        Writing the number into the field would freeze this config against
        today's defaults; leaving it blank keeps the type the owner, which is
        the whole point of it having them (ADR-0011).  On a type change, a
        blank field follows the type: checked when the type has a default,
        unchecked when it does not.
        """
        item = self._current_type()
        defaults = item.default_constants or {}
        for check, edit, key in self._threshold_controls():
            value = defaults.get(key)
            if value is None:
                edit.setPlaceholderText(
                    "no default \u2014 check and type a number to filter")
            else:
                edit.setPlaceholderText(
                    f"{_fmt_number(value)}  (default for {item.display_name})")
            if not edit.text().strip():
                check.blockSignals(True)
                check.setChecked(value is not None)
                check.blockSignals(False)
                edit.setEnabled(check.isChecked() and check.isEnabled())
        self._refresh_pr_constant_rows()

    def _pr_constants_apply(self) -> bool:
        """Whether the selected type carries Progressive Ratio's constants."""
        from .experiment_types.progressive_ratio import PR_CONSTANT_FIELDS

        defaults = set(self._current_type().default_constants or {})
        return bool(defaults & {f.key for f in PR_CONSTANT_FIELDS})

    def _refresh_pr_constant_rows(self) -> None:
        """Show the Progressive Ratio rows for a type that has them, with the
        type's defaults as placeholders and as the switches' "default"
        entry — never as values (ADR-0011)."""
        if not hasattr(self, "_pr_constant_widgets"):
            return
        from .experiment_types.progressive_ratio import PR_CONSTANT_FIELDS

        item = self._current_type()
        shown = self._pr_constants_apply()
        defaults = item.default_constants or {}
        self._exp_form.setRowVisible(self._pr_header_row, shown)
        for field in PR_CONSTANT_FIELDS:
            self._exp_form.setRowVisible(self._pr_constant_rows[field.key], shown)
            widget = self._pr_constant_widgets[field.key]
            default = defaults.get(field.key)
            if field.group == "switch":
                widget.setItemText(0, "default" if default is None else
                                   "default: " + field.choices[0 if default else 1])
            elif default is None:
                widget.setPlaceholderText(f"{field.blank or 'no default'}  (default)")
            else:
                widget.setPlaceholderText(
                    f"{_fmt_number(default)}  (default for {item.display_name})")
        self._refresh_opto_rows()

    def _opto_rows_apply(self) -> bool:
        """Whether the optogenetic light QC's rows are shown: optogenetics is
        yes, or it is auto and the experiment has shown itself to be
        optogenetic — a Program.txt in data/, or light QC files from a run."""
        value = self._opto_combo.currentData() if hasattr(self, "_opto_combo") else "auto"
        if value is False:
            return False
        if value is True:
            return True
        if self._current_path is None:
            return False
        from .layout import DATA_FOLDER_KEY, data_location
        from .opto_program import find_program

        directory = self._current_path.parent
        ## Where the loader would read Program.txt: data/, or the Data Folder
        ## as it stands in the field (ADR-0021).
        location = data_location(
            directory, config={DATA_FOLDER_KEY: self._data_folder_edit.text().strip()})
        try:
            if not location.problem and find_program(location.path) is not None:
                return True
        except ValueError:
            return True             # more than one Program.txt: still optogenetic
        return (directory / "qc" / "opto" / "opto_light_qc.csv").is_file()

    def _refresh_opto_rows(self, *_args) -> None:
        if not hasattr(self, "_opto_constant_widgets"):
            return
        shown = self._opto_rows_apply()
        self._exp_form.setRowVisible(self._opto_header_row, shown)
        for row in self._opto_constant_rows.values():
            self._exp_form.setRowVisible(row, shown)
        if hasattr(self, "_slices_widget"):
            self._slices_widget.set_opto_columns_visible(shown)

    def _load_opto_constants(self, constants: dict[str, Any]) -> None:
        from .opto_qc import OPTO_CONSTANT_FIELDS

        for field in OPTO_CONSTANT_FIELDS:
            widget = self._opto_constant_widgets[field.key]
            value = constants.get(field.key)
            if field.group == "switch":
                state = _switch_value(value)
                widget.blockSignals(True)
                widget.setCurrentIndex(0 if state is None else (1 if state else 2))
                widget.blockSignals(False)
            else:
                widget.setText("" if value is None else _fmt_number(value))

    def _load_pr_constants(self, constants: dict[str, Any]) -> None:
        """Fill the Progressive Ratio rows from a ``constants:`` block: what it
        states shows as a value, what it leaves out stays blank (the type's
        default).  A value that is no number is shown as written, so saving
        keeps it and the validation names it."""
        from .experiment_types.progressive_ratio import PR_CONSTANT_FIELDS

        for field in PR_CONSTANT_FIELDS:
            widget = self._pr_constant_widgets[field.key]
            value = constants.get(field.key)
            if field.group == "switch":
                state = _switch_value(value)
                widget.blockSignals(True)
                widget.setCurrentIndex(0 if state is None else (1 if state else 2))
                widget.blockSignals(False)
            else:
                widget.setText("" if value is None else _fmt_number(value))

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
        self._exp_form.setRowVisible(self._ceiling_row, True)
        self._exp_form.setRowVisible(
            self._max_dur_row, two_well or "max_med_duration_cutoff" in defaults)
        self._exp_form.setRowVisible(
            self._max_events_row, two_well or "max_events_cutoff" in defaults)

    def _threshold_row_visible(self, key: str) -> bool:
        two_well = self._chamber_size() == 2
        defaults = self._current_type().default_constants or {}
        if key in ("min_untransformed_licks_cutoff", "max_ceiling_minutes"):
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
        w.set_progressive_ratio(self._current_type().has_module("ratio"))
        for combo in (*w._paired_combos, w._groups_combo):
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
            experiment += self._slice_problems()
        dfms: list[str] = []
        for w in self._dfm_widgets:
            dfms.extend(w.chamber_problems())
        return experiment, dfms

    def _slice_problems(self) -> list[str]:
        """What keeps this experiment's Slices from being saved: the cells
        the table could not read, then ``slices.slice_problems`` — the
        linter's own check — on the ``global:`` that would be written.  None
        for a Member, whose Slices are its Project's."""
        if self._design:
            return []
        try:
            _declared, problems = self._slices_widget.get_values()
            return problems + slices_mod.slice_problems(
                self._global_section(), self._current_type())
        except Exception:  # noqa: BLE001 - validation must never block the UI
            return []

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
          • a path that does not exist → nothing is loaded.
        Otherwise: the current working directory's config, if it has one —
        the Experiment Directory the editor was started in.

        A named path is never swapped for the working directory's config:
        Save would then overwrite a file nobody named.  Where nothing is
        loaded, Save asks where to write, starting from what was named.
        """
        names = ("flic_config.yaml", "flic_config.yml")
        candidates: list[Path] = []
        if initial_path is None:
            candidates.extend(Path.cwd() / name for name in names)
        else:
            p = Path(initial_path).expanduser()
            if p.is_file():
                candidates.append(p)
            elif p.is_dir():
                candidates.extend(p / name for name in names)
                self._save_hint = p / names[0]
            else:
                if p.suffix.lower() in (".yaml", ".yml"):
                    self._save_hint = p
                self.statusBar().showMessage(
                    f"{p} does not exist: nothing loaded.  Save asks where to write.")
                return

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
        ## and they are derived.  A type that leaves the layout open (Custom,
        ## or a lab type with chamber_layout: either) states it, because
        ## single-well versus two-well silently reinterprets the whole plate
        ## and is not a thing to leave implicit.
        if not item.is_custom:
            global_section["experiment_type"] = item.name
        if item.layout_free:
            global_section["chamber_layout"] = self._chamber_layout()

        global_params = self._global_params.get_values()
        global_params.pop("chamber_size", None)
        global_section["params"] = global_params

        # Lick transformation — only emit the key when it differs from the
        # historical default (True) to keep YAML output minimal.
        if not self._transform_licks_check.isChecked():
            global_section["transform_licks"] = False

        ## optogenetics: only yes / no are written; auto is the default.
        opto = self._opto_combo.currentData()
        if isinstance(opto, bool):
            global_section["optogenetics"] = opto

        ## facet_source: only program / program_cycles are written (cutoffs is
        ## the default), never under a type that owns its Facets.
        from .program_facets import FACET_SOURCE_KEY, is_program_source

        facet_source = self._facet_source_combo.currentData()
        if not item.owns_facet_source and is_program_source(facet_source):
            global_section[FACET_SOURCE_KEY] = facet_source

        # Well names
        if self._chamber_size() == 2:
            wa = self._well_a_edit.text().strip()
            wb = self._well_b_edit.text().strip()
            if wa or wb:
                global_section["well_names"] = {
                    **({"A": wa} if wa else {}),
                    **({"B": wb} if wb else {}),
                }

        ## Optional cutoffs: unchecked → YAML null (filter off, overrides the
        ## type default); checked + blank → omit (inherit); checked + number
        ## → override.  Materialising the type default would freeze this
        ## config against today's numbers (ADR-0011).  A cutoff the type has
        ## no default for is off unchecked already, so its null is written
        ## only when the file stated it: a look-and-save adds nothing.
        constants: dict[str, Any] = {}
        type_defaults = item.default_constants or {}
        loaded = (self._loaded_raw.get("global") or {}).get("constants") or {}
        for check, widget, key in self._threshold_controls():
            if not self._threshold_row_visible(key):
                continue
            if not check.isChecked():
                if type_defaults.get(key) is not None or key in loaded:
                    constants[key] = None
                continue
            text = widget.text().strip()
            if not text:
                continue
            try:
                constants[key] = float(text)
            except ValueError:
                pass
        from .experiment_types.progressive_ratio import PR_CONSTANT_FIELDS

        if self._pr_constants_apply():
            for field in PR_CONSTANT_FIELDS:
                widget = self._pr_constant_widgets[field.key]
                if field.group == "switch":
                    value = widget.currentData()
                    if value is not None:
                        constants[field.key] = bool(value)
                    continue
                text = widget.text().strip()
                if text:
                    constants[field.key] = _number_or_text(text)
        from .opto_qc import OPTO_CONSTANT_FIELDS

        opto_managed: set[str] = set()
        if self._opto_rows_apply():
            opto_managed = {field.key for field in OPTO_CONSTANT_FIELDS}
            for field in OPTO_CONSTANT_FIELDS:
                widget = self._opto_constant_widgets[field.key]
                if field.group == "switch":
                    value = widget.currentData()
                    if value is not None:
                        constants[field.key] = bool(value)
                    continue
                text = widget.text().strip()
                if text:
                    constants[field.key] = _number_or_text(text)
        ## A constant this form has no field for was typed into the yaml by
        ## hand; keep it on save.  The Progressive Ratio rows are managed like
        ## the cutoffs: a cleared row, or another type, drops its key.  The
        ## optogenetic rows are managed only while they are shown, so a hidden
        ## section keeps what the file says.
        managed = ({key for _widget, key in self._threshold_fields()}
                   | {field.key for field in PR_CONSTANT_FIELDS} | opto_managed)
        for key, value in loaded.items():
            if key not in managed:
                constants.setdefault(key, value)
        if constants:
            global_section["constants"] = constants

        # Experimental design factors
        factors = self._factors_widget.get_factors()
        if factors:
            global_section["experimental_design_factors"] = factors

        ## The whole recording's own exclusion group and the groups it also
        ## excludes, then the Slices (ADR-0020); each only when stated.
        group = self._exclusion_group_edit.text().strip()
        if group:
            global_section["exclusion_group"] = group
        others = _group_list(self._also_exclude_edit.text())
        if others:
            global_section["also_exclude"] = others
        declared, _problems = self._slices_widget.get_values()
        global_section.update(declared)

        ## The Selection (ADR-0018) has no field here: the Hub's ticks write
        ## it.  Kept as the file says, or a save would undo every untick —
        ## with a renamed Slice's entry renamed, so its box stays unticked.
        loaded_global = self._loaded_raw.get("global") or {}
        omit = self._slices_widget.follow_renames(loaded_global.get("omit"))
        if omit:
            global_section["omit"] = omit

        ## The Treatment control's level, under a type that has one (ADR-0019).
        level = self._control_level_edit.text().strip()
        if level and item.definition is not None and item.definition.treatment_control:
            global_section["control_level"] = level

        ## A key this form has no field for — facet_cutoffs, facet_labels, a
        ## single-well design's well_names — was written by hand or by
        ## another editor; keep it.  A key the form manages is written above
        ## or, cleared, not at all: keeping the file's would undo the clear.
        ## Nor is a key kept that the choices made here rule out: one the
        ## type now owns, or minute cutoffs beside Program Facets.
        managed = set(_MANAGED_GLOBAL_KEYS) | set(item.owned_keys())
        if self._chamber_size() == 2:
            managed.add("well_names")
        if is_program_source(facet_source):
            managed.add("facet_cutoffs")
        for key, value in loaded_global.items():
            if key not in managed:
                global_section.setdefault(key, value)

        return global_section

    def _collect_yaml(self) -> dict[str, Any]:
        ## Start from the file as it was read so keys this editor knows
        ## nothing about survive — a member's scripts: above all, which a
        ## rebuilt-from-widgets config used to delete on the first save.
        cfg = {k: v for k, v in self._loaded_raw.items()
               if k not in ("global", "dfms", "DFMs")}
        if self._design:
            ## A Member carries the Design Copy — what its recording is — and
            ## inherits the rest (ADR-0025): writing the analysis settings here
            ## would at best duplicate the authority and at worst contradict
            ## it, and a contradiction stops the whole Project loading.  A
            ## Member whose own description conflicts with the Design keeps
            ## its global: exactly as written: which is right is a person's
            ## call, never the editor's.
            from . import project as project_mod

            wanted = project_mod.member_global(
                self._loaded_raw, self._design, self._type_where())
            if wanted is None:
                if self._loaded_raw.get("global") is not None:
                    cfg["global"] = self._loaded_raw["global"]
            elif wanted:
                cfg["global"] = wanted
        else:
            cfg["global"] = self._global_section()
        cfg["dfms"] = [w.get_dict() for w in self._dfm_widgets]
        cfg = self._with_unedited_on_disk(cfg)
        return self._with_about_on_disk(self._with_data_folder(cfg))

    def _with_unedited_on_disk(self, cfg: dict[str, Any]) -> dict[str, Any]:
        """*cfg* with what this editor has no field for as the file holds it
        **now**: every top-level key but ``global:``, ``dfms:``, ``about:``
        (:meth:`_with_about_on_disk`) and ``data_folder:``, and a standalone
        config's Selection (``global.omit``).

        The Script Editor writes ``scripts:`` and a Hub tick writes the
        Selection, each perhaps while this editor has the file open, and a
        save wrote back the copy read when it was opened, undoing them.  Only
        a save over the file that was opened reads it again; a file that is
        gone or no longer parses keeps what was loaded.
        """
        if (self._current_path is None or self._loaded_path is None
                or not _same_path(self._current_path, self._loaded_path)):
            return cfg
        try:
            disk = yaml.safe_load(self._current_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError):
            return cfg
        if not isinstance(disk, dict):
            return cfg
        from .about import ABOUT_KEY
        from .layout import DATA_FOLDER_KEY

        edited = {"global", "dfms", "DFMs", ABOUT_KEY, DATA_FOLDER_KEY}
        out: dict[str, Any] = {}
        for key, value in cfg.items():
            if key in edited:
                out[key] = value
            elif key in disk:
                out[key] = disk[key]
        for key, value in disk.items():
            if key not in edited and key not in out:
                out[key] = value
        if not self._design and isinstance(out.get("global"), dict):
            section = dict(out["global"])
            on_disk = disk.get("global") if isinstance(disk.get("global"), dict) else {}
            omit = self._slices_widget.follow_renames(on_disk.get("omit"))
            if omit:
                section["omit"] = omit
            else:
                section.pop("omit", None)
            out["global"] = section
        return out

    def _with_about_on_disk(self, cfg: dict[str, Any]) -> dict[str, Any]:
        """*cfg* with the ``about:`` its file holds now (ADR-0026).

        This editor does not edit ``about:``, and a catalogue that files the
        recording may rewrite the section while the file is open here.  So
        the section written is the one on disk at this moment, in the file
        being written, not the one read when it was opened: added, changed
        or removed there, it is the same after this save.  A file that is
        gone or no longer parses keeps the section that was loaded, without
        ``run:`` when Save As writes into another recording's folder.
        """
        if self._current_path is None:
            return cfg
        from . import project as project_mod

        found, about = project_mod.about_on_disk(self._current_path)
        if found:
            return project_mod.with_about(cfg, about)
        if self._loaded_path is not None and not _same_path(
                self._current_path.parent, self._loaded_path.parent):
            return project_mod.without_run(cfg)
        return cfg

    def _with_data_folder(self, cfg: dict[str, Any]) -> dict[str, Any]:
        """*cfg* with the Data Folder field's ``data_folder:`` (ADR-0021).

        A value typed where there was none goes first, where labdata writes
        it; a cleared field removes the key; a blank field never adds one, so
        a config that did not name a data folder is saved without the key.  A
        value that is not text was never shown in the field, and is kept as
        the file has it unless something is typed over it.
        """
        from .layout import DATA_FOLDER_KEY

        text = self._data_folder_edit.text().strip()
        raw = cfg.get(DATA_FOLDER_KEY)
        if text:
            if raw != text:
                rest = {k: v for k, v in cfg.items() if k != DATA_FOLDER_KEY}
                cfg = {DATA_FOLDER_KEY: text, **rest} if raw is None else {
                    k: (text if k == DATA_FOLDER_KEY else v) for k, v in cfg.items()}
        elif raw is None or isinstance(raw, str):
            cfg.pop(DATA_FOLDER_KEY, None)
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
        self._loaded_path = self._current_path
        from .layout import DATA_FOLDER_KEY

        folder = cfg.get(DATA_FOLDER_KEY)
        self._data_folder_edit.blockSignals(True)
        self._data_folder_edit.setText(folder if isinstance(folder, str) else "")
        self._data_folder_edit.blockSignals(False)
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
        try:
            item = experiment_types.get_experiment_type(
                design.get("experiment_type"), self._type_where())
        except ValueError:
            item = experiment_types.get_experiment_type(None)
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
                       self._new_type_btn, self._control_level_edit,
                       self._well_a_edit, self._well_b_edit,
                       self._transform_licks_check):
            widget.setEnabled(not governed)
            widget.setToolTip(reason)
        for widget, tip in self._exclusion_tips.items():
            widget.setEnabled(not governed)
            widget.setToolTip(reason or tip)
        self._slices_widget.set_read_only(governed, reason=reason)
        for check, edit, _key in self._threshold_controls():
            check.setEnabled(not governed)
            check.setToolTip(reason or check.toolTip())
            edit.setEnabled(not governed and check.isChecked())
            edit.setToolTip(reason)
        from .experiment_types.progressive_ratio import PR_CONSTANT_FIELDS

        for field in PR_CONSTANT_FIELDS:
            widget = self._pr_constant_widgets[field.key]
            widget.setEnabled(not governed)
            widget.setToolTip(reason or field.tooltip)
        from .opto_qc import OPTO_CONSTANT_FIELDS

        for field in OPTO_CONSTANT_FIELDS:
            widget = self._opto_constant_widgets[field.key]
            widget.setEnabled(not governed)
            widget.setToolTip(reason or field.tooltip)
        self._opto_combo.setEnabled(not governed)
        if governed:
            self._opto_combo.setToolTip(
                reason + " — a DFM tab can still override optogenetics for its DFM")
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
                "Project → Project design…).  Saving writes only the design's "
                "description of the recording (type, layout, well names, "
                "factors) into this file's <code>global:</code>, so the file "
                "reads on its own; every other setting stays the design's.")

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
            item = experiment_types.get_experiment_type(raw_type, self._type_where())
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

        ## The types beside this file (a lab type's copy) join the list.
        self._fill_type_combo()
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
        self._control_level_edit.setText(str(global_cfg.get("control_level") or ""))
        self._sync_control_level()

        params_to_load = {k: v for k, v in global_params_raw.items() if k != "chamber_size"}
        self._global_params.load_values(params_to_load, chamber_size)
        self._global_params.set_chamber_size(chamber_size)

        # Well names
        well_names = global_cfg.get("well_names") or {}
        self._well_a_edit.setText(str(well_names.get("A", "")))
        self._well_b_edit.setText(str(well_names.get("B", "")))

        # Lick transformation toggle
        self._transform_licks_check.setChecked(bool(global_cfg.get("transform_licks", True)))
        self._opto_combo.blockSignals(True)
        self._opto_combo.setCurrentIndex(_opto_index(global_cfg.get("optogenetics")))
        self._opto_combo.blockSignals(False)
        self._facet_source_combo.blockSignals(True)
        self._facet_source_combo.setCurrentIndex(max(0, self._facet_source_combo.findData(
            item.resolve_facet_source(global_cfg))))
        self._facet_source_combo.blockSignals(False)

        # Filter thresholds — null means off; omit means inherit (checked).
        constants = global_cfg.get("constants") or {}
        self._load_threshold_controls(constants)
        self._load_pr_constants(constants)
        self._load_opto_constants(constants)
        self._refresh_opto_rows()

        # Experimental design factors
        factors_node = global_cfg.get("experimental_design_factors") or {}
        self._factors_widget._table.blockSignals(True)
        self._factors_widget.load_factors(factors_node)
        self._factors_widget._table.blockSignals(False)
        factors = self._factors_widget.get_factors()

        # Exclusion groups and Slices
        self._exclusion_group_edit.setText(str(global_cfg.get("exclusion_group") or ""))
        self._also_exclude_edit.setText(_group_text(global_cfg.get("also_exclude")))
        self._slices_widget.load_values(global_cfg)
        self._sync_slices()

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

        for widget in self._dfm_widgets[len(dfm_nodes):]:
            widget._unmanaged = {}
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
        self._loaded_path = None
        self._loaded_raw = {}
        self._data_folder_edit.clear()
        self._design = None
        self._design_source = None
        self._apply_design()
        self.setWindowTitle("FLIC Config Editor")
        self._well_a_edit.clear()
        self._well_b_edit.clear()
        for check, edit, _key in self._threshold_controls():
            check.blockSignals(True)
            check.setChecked(False)
            check.blockSignals(False)
            edit.clear()
            edit.setEnabled(False)
        self._load_pr_constants({})
        self._load_opto_constants({})
        self._opto_combo.setCurrentIndex(0)
        self._facet_source_combo.setCurrentIndex(0)
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
        self._exclusion_group_edit.clear()
        self._also_exclude_edit.clear()
        self._slices_widget.load_values({})

        self._dfm_tabs.clear()
        for w in self._dfm_widgets:
            w.deleteLater()
        self._dfm_widgets.clear()

        self._global_params.reset_defaults(2)
        self._fill_type_combo('Custom')
        self._control_level_edit.clear()
        self._sync_layout_control()
        self._refresh_threshold_hints()
        self._update_well_names_visibility()
        self._sync_control_level()
        self._sync_slices()
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
        default = str(self._current_path or self._save_hint
                      or (Path.cwd() / "flic_config.yaml"))
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

    def _refuse_slices(self, problems: list[str]) -> None:
        """Say why the Slices stop this save.  The one refusal here: a Slice
        that cannot be used is skipped by every run, so writing it would
        promise an analysis that never happens."""
        shown = problems[:12]
        detail = "\n".join(f"  • {t}" for t in shown)
        if len(problems) > len(shown):
            detail += f"\n  … and {len(problems) - len(shown)} more"
        QMessageBox.warning(
            self, "Slices cannot be saved",
            f"Not saved — the Slices cannot be used as they stand:\n\n{detail}\n\n"
            "Correct or remove them on the Experiment tab, then save again.")

    def _write_yaml(self, path: Path) -> None:
        slice_problems = self._slice_problems()
        if slice_problems:
            self._refuse_slices(slice_problems)
            return
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
            self._write_type_copy(path)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to save config:\n{exc}")
            return
        self.saved.emit(str(path))
        QMessageBox.information(self, "Saved", f"Config saved to:\n{path}")

    def _write_type_copy(self, path: Path) -> None:
        """Carry a standalone experiment's type with it (ADR-0019): a lab type
        is copied beside *path*, a shipped one leaves no copy behind.  A
        Member's type is its Project's, copied by the Design editor."""
        from .experiment_types import library

        directory = Path(path).parent
        if self._design or library.copy_dir(directory) != directory:
            return
        item = self._current_type()
        if item.is_shipped:
            library.remove_copy(directory)
        else:
            library.write_copy(item.definition, directory)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def launch() -> None:
    """Launch the FLIC Config Editor GUI.

    Optional CLI argument: a YAML config file (or a directory containing
    ``flic_config.yaml``).  When omitted, opens the current working
    directory's ``flic_config.yaml`` if it has one, and otherwise a blank
    editor whose Save asks where to write.
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
