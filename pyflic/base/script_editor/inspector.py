"""Step inspector — renders a parameter form for the selected step.

Re-builds itself from the action catalogue every time a new step is
selected.  Emits :pyattr:`stepEdited(dict)` with a fresh copy of the step
whenever any field changes.
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..ui import Category, category_color, icon
from .actions import Action, Param, get_action, metric_choices, default_mode_for_metric


class Inspector(QWidget):
    """Right-hand parameter form for the currently-selected step."""

    stepEdited = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._experiment_type: str | None = None
        self._current_step: dict[str, Any] | None = None
        self._current_action: Action | None = None
        self._widgets: dict[str, dict[str, Any]] = {}
        self._is_rebuilding = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        title = QLabel("Step parameters", self)
        title.setStyleSheet("font-size: 12pt; font-weight: 600;")
        outer.addWidget(title)

        # Header card (shows the action name + blurb)
        self._header = QFrame(self)
        self._header.setObjectName("InspectorHeader")
        self._header.setFrameShape(QFrame.Shape.NoFrame)
        hlay = QHBoxLayout(self._header)
        hlay.setContentsMargins(10, 8, 10, 8)
        hlay.setSpacing(8)
        self._hdr_icon = QLabel(self._header)
        self._hdr_icon.setFixedSize(24, 24)
        hlay.addWidget(self._hdr_icon)
        hdr_text_col = QVBoxLayout()
        hdr_text_col.setContentsMargins(0, 0, 0, 0)
        self._hdr_title = QLabel(self._header)
        self._hdr_title.setStyleSheet("font-weight: 600;")
        self._hdr_blurb = QLabel(self._header)
        self._hdr_blurb.setStyleSheet("color: palette(mid); font-size: 9pt;")
        self._hdr_blurb.setWordWrap(True)
        hdr_text_col.addWidget(self._hdr_title)
        hdr_text_col.addWidget(self._hdr_blurb)
        hlay.addLayout(hdr_text_col, 1)
        outer.addWidget(self._header)

        # Scrollable form area
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._form_host = QWidget()
        self._form_lay = QVBoxLayout(self._form_host)
        self._form_lay.setContentsMargins(0, 0, 0, 0)
        self._form_lay.setSpacing(10)
        scroll.setWidget(self._form_host)
        outer.addWidget(scroll, 1)

        # Validation banner
        self._val_lbl = QLabel(self)
        self._val_lbl.setWordWrap(True)
        self._val_lbl.setStyleSheet(
            f"color: {category_color(Category.QC)}; font-size: 9pt;"
        )
        self._val_lbl.setVisible(False)
        outer.addWidget(self._val_lbl)

        self.setMinimumWidth(280)
        self._clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_experiment_type(self, et: str | None) -> None:
        self._experiment_type = et
        # Metric dropdowns may depend on experiment type; rebuild.
        if self._current_step is not None:
            self.show_step(self._current_step)

    def show_step(self, step: dict[str, Any] | None) -> None:
        self._current_step = step
        self._current_action = get_action(step.get("action", "")) if step else None
        self._rebuild()

    def clear(self) -> None:
        self._clear()

    # ------------------------------------------------------------------
    # Rebuild
    # ------------------------------------------------------------------

    def _clear_form(self) -> None:
        while self._form_lay.count():
            item = self._form_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
            sub = item.layout()
            if sub is not None:
                self._drain_layout(sub)

    @staticmethod
    def _drain_layout(lay) -> None:
        while lay.count():
            item = lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
            sub = item.layout()
            if sub is not None:
                Inspector._drain_layout(sub)

    def _clear(self) -> None:
        self._widgets.clear()
        self._clear_form()
        self._hdr_icon.clear()
        self._hdr_title.setText("—")
        self._hdr_blurb.setText("Select a step in the canvas to edit its parameters.")
        self._header.setStyleSheet(
            "QFrame#InspectorHeader { border-left: 4px solid palette(mid);"
            " border-radius: 6px; background: palette(alternate-base); }"
        )
        self._val_lbl.setVisible(False)

    def _rebuild(self) -> None:
        self._widgets.clear()
        self._clear_form()
        step = self._current_step
        act = self._current_action
        if step is None or act is None:
            self._clear()
            return

        # Header
        self._hdr_icon.setPixmap(icon(act.icon, category=act.category).pixmap(24, 24))
        self._hdr_title.setText(act.label)
        self._hdr_blurb.setText(act.blurb)
        col = category_color(act.category)
        self._header.setStyleSheet(
            f"QFrame#InspectorHeader {{ border-left: 4px solid {col};"
            f" border-radius: 6px; background: palette(alternate-base); }}"
        )

        self._is_rebuilding = True
        try:
            for p in act.params:
                self._form_lay.addWidget(self._build_param_widget(p, step))
        finally:
            self._is_rebuilding = False

        self._form_lay.addStretch(1)

        # Validation
        issues = _validate(step, act, self._experiment_type)
        if issues:
            self._val_lbl.setText("⚠  " + "<br>⚠  ".join(issues))
            self._val_lbl.setVisible(True)
        else:
            self._val_lbl.setVisible(False)

    def _build_param_widget(self, p: Param, step: dict[str, Any]) -> QWidget:
        container = QFrame()
        container.setFrameShape(QFrame.Shape.NoFrame)
        lay = QVBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        lbl = QLabel(p.label)
        lbl.setStyleSheet("font-weight: 500;")
        if p.required:
            lbl.setText(p.label + "  *")
            lbl.setToolTip("Required parameter")
        lay.addWidget(lbl)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        widget, getter, setter = self._build_field(p, step)
        row.addWidget(widget, 1)
        if p.unit:
            unit_lbl = QLabel(p.unit)
            unit_lbl.setStyleSheet("color: palette(mid);")
            row.addWidget(unit_lbl)
        lay.addLayout(row)

        inherit_cb: QCheckBox | None = None
        if p.inheritable:
            inherit_cb = QCheckBox("inherit from UI / script")
            inherit_cb.setChecked(p.key not in step)
            widget.setEnabled(not inherit_cb.isChecked())
            inherit_cb.toggled.connect(
                lambda checked, w=widget, key=p.key: self._on_inherit_toggled(
                    checked, w, key
                )
            )
            lay.addWidget(inherit_cb)

        if p.note:
            note = QLabel(p.note)
            note.setStyleSheet("color: palette(mid); font-size: 9pt;")
            note.setWordWrap(True)
            lay.addWidget(note)

        self._widgets[p.key] = {
            "param": p,
            "widget": widget,
            "get": getter,
            "set": setter,
            "inherit": inherit_cb,
        }
        return container

    def _build_field(self, p: Param, step: dict[str, Any]):
        """Return (widget, getter, setter) for parameter *p*."""
        value = step.get(p.key, p.default)

        if p.type == "bool":
            w = QCheckBox()
            if value is not None:
                w.setChecked(bool(value))
            w.toggled.connect(lambda _s: self._on_field_changed())
            return w, (lambda: w.isChecked()), (lambda v: w.setChecked(bool(v)))

        if p.type == "int":
            w = QSpinBox()
            w.setRange(-1_000_000, 1_000_000)
            if value is not None:
                try:
                    w.setValue(int(value))
                except (TypeError, ValueError):
                    pass
            w.valueChanged.connect(lambda _v: self._on_field_changed())
            return w, (lambda: int(w.value())), (lambda v: w.setValue(int(v)))

        if p.type == "float":
            w = QDoubleSpinBox()
            w.setRange(-1_000_000, 1_000_000)
            w.setDecimals(3)
            if value is not None:
                try:
                    w.setValue(float(value))
                except (TypeError, ValueError):
                    pass
            w.valueChanged.connect(lambda _v: self._on_field_changed())
            return w, (lambda: float(w.value())), (lambda v: w.setValue(float(v)))

        if p.type == "choice":
            w = QComboBox()
            for choice in (p.choices or []):
                w.addItem(choice)
            if value is None and p.derived_from:
                # compute derived default from another step field
                src = step.get(p.derived_from)
                if isinstance(src, str):
                    derived = default_mode_for_metric(src)
                    if derived:
                        idx = w.findText(derived)
                        if idx >= 0:
                            w.setCurrentIndex(idx)
            elif value is not None:
                idx = w.findText(str(value))
                if idx >= 0:
                    w.setCurrentIndex(idx)
            w.currentTextChanged.connect(lambda _t: self._on_field_changed())
            return w, (lambda: w.currentText()), (lambda v: w.setCurrentText(str(v)))

        if p.type in ("metric", "well_metric"):
            w = QComboBox()
            is_two_well = self._experiment_type in {"two_well", "hedonic", "progressive_ratio"}
            for label, key in metric_choices(p.type, is_two_well):
                w.addItem(label, userData=key)
            if value is not None:
                for i in range(w.count()):
                    if w.itemData(i) == value:
                        w.setCurrentIndex(i)
                        break
            w.currentIndexChanged.connect(lambda _i: self._on_field_changed())
            return (
                w,
                (lambda: w.currentData() or ""),
                (lambda v: _select_by_data(w, v)),
            )

        if p.type == "list_str":
            w = QLineEdit()
            if isinstance(value, list):
                w.setText(", ".join(str(x) for x in value))
            elif isinstance(value, str):
                w.setText(value)
            w.setPlaceholderText("item1, item2, item3")
            w.textChanged.connect(lambda _t: self._on_field_changed())
            return (
                w,
                (lambda: _parse_list_str(w.text())),
                (lambda v: w.setText(", ".join(str(x) for x in (v or [])))),
            )

        if p.type == "list_float":
            w = QLineEdit()
            if isinstance(value, list):
                w.setText(", ".join(str(x) for x in value))
            w.setPlaceholderText("1.0, 2.5, 5.0")
            w.textChanged.connect(lambda _t: self._on_field_changed())
            return (
                w,
                (lambda: _parse_list_float(w.text())),
                (lambda v: w.setText(", ".join(str(x) for x in (v or [])))),
            )

        # string fallback
        w = QLineEdit()
        if value is not None:
            w.setText(str(value))
        w.textChanged.connect(lambda _t: self._on_field_changed())
        return w, (lambda: w.text().strip()), (lambda v: w.setText(str(v)))

    # ------------------------------------------------------------------
    # Field change handling
    # ------------------------------------------------------------------

    def _on_inherit_toggled(self, inherited: bool, widget: QWidget, _key: str) -> None:
        widget.setEnabled(not inherited)
        self._on_field_changed()

    def _on_field_changed(self) -> None:
        if self._is_rebuilding or self._current_step is None or self._current_action is None:
            return
        new_step: dict[str, Any] = {"action": self._current_action.action}
        for key, entry in self._widgets.items():
            p = entry["param"]
            inherit_cb = entry["inherit"]
            if inherit_cb is not None and inherit_cb.isChecked():
                # Omit field entirely ⇒ inherit from script / UI.
                continue
            value = entry["get"]()
            if value == "" or value is None:
                continue
            if isinstance(value, list) and not value:
                continue
            new_step[key] = value
        self._current_step = new_step
        self.stepEdited.emit(new_step)
        # Re-validate.
        issues = _validate(new_step, self._current_action, self._experiment_type)
        if issues:
            self._val_lbl.setText("⚠  " + "<br>⚠  ".join(issues))
            self._val_lbl.setVisible(True)
        else:
            self._val_lbl.setVisible(False)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _select_by_data(combo: QComboBox, value: Any) -> None:
    for i in range(combo.count()):
        if combo.itemData(i) == value:
            combo.setCurrentIndex(i)
            return


def _parse_list_str(text: str) -> list[str]:
    return [tok.strip() for tok in text.split(",") if tok.strip()]


def _parse_list_float(text: str) -> list[float]:
    out: list[float] = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def _validate(step: dict[str, Any], act: Action, experiment_type: str | None) -> list[str]:
    msgs: list[str] = []
    if act.requires and act.requires != experiment_type:
        msgs.append(
            f"This action requires experiment_type={act.requires!r}; current is "
            f"{experiment_type or 'unspecified'}. The step will be skipped at "
            f"runtime."
        )
    for p in act.params:
        if p.required and p.key not in step:
            msgs.append(f"Missing required parameter: {p.label} ({p.key}).")
    return msgs
