"""Script canvas — step list with drag-to-reorder + script-level header.

The canvas is the editor's single source of truth for the currently-being-
edited script.  It exposes:

* :pyattr:`stepsChanged` — emitted whenever the script structure changes
  (add/remove/reorder/rename/start-end edits) so the window can update the
  YAML preview and flag unsaved changes.
* :pyattr:`stepSelected(int)` — emitted when the user clicks a step card.
  ``-1`` means "no selection".
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QSizePolicy,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..ui import Category, category_color, icon
from .actions import Action, describe_step, get_action, validation_issues


# ---------------------------------------------------------------------------
# Step card widget (embedded inside each QListWidgetItem)
# ---------------------------------------------------------------------------

class _StepCard(QFrame):
    """Compact visual for one step.

    Step state lives in the QListWidgetItem's user role; this widget is a
    pure view.  The card has no click handlers — selection is managed by
    the parent QListWidget.
    """

    deleteRequested = pyqtSignal(object)        # emits self
    menuRequested = pyqtSignal(object)          # emits self

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ScriptStepCard")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAutoFillBackground(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 8, 10, 8)
        lay.setSpacing(10)

        # Drag handle (purely visual — drag is driven by QListWidget)
        self._handle = QLabel("☰", self)
        self._handle.setStyleSheet("color: palette(mid); font-size: 14pt;")
        self._handle.setFixedWidth(18)
        lay.addWidget(self._handle)

        # Number + icon
        self._num_lbl = QLabel("1.", self)
        self._num_lbl.setStyleSheet("color: palette(mid); font-weight: 600;")
        self._num_lbl.setFixedWidth(28)
        lay.addWidget(self._num_lbl)

        self._icon_lbl = QLabel(self)
        self._icon_lbl.setFixedSize(QSize(22, 22))
        lay.addWidget(self._icon_lbl)

        # Title + params chip
        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(2)
        self._title_lbl = QLabel("", self)
        self._title_lbl.setStyleSheet("font-weight: 600;")
        text_col.addWidget(self._title_lbl)
        self._chip_lbl = QLabel("", self)
        self._chip_lbl.setStyleSheet("color: palette(mid); font-size: 9pt;")
        self._chip_lbl.setWordWrap(True)
        text_col.addWidget(self._chip_lbl)
        lay.addLayout(text_col, 1)

        # Warning indicator (shown when validation returns issues)
        self._warn_lbl = QLabel(self)
        self._warn_lbl.setFixedSize(QSize(18, 18))
        self._warn_lbl.setPixmap(icon("warning", category=Category.QC).pixmap(16, 16))
        self._warn_lbl.setVisible(False)
        lay.addWidget(self._warn_lbl)

        # More menu
        self._btn_menu = QToolButton(self)
        self._btn_menu.setText("⋯")
        self._btn_menu.setAutoRaise(True)
        self._btn_menu.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_menu.clicked.connect(lambda: self.menuRequested.emit(self))
        lay.addWidget(self._btn_menu)

        # Delete
        self._btn_del = QToolButton(self)
        self._btn_del.setText("✕")
        self._btn_del.setAutoRaise(True)
        self._btn_del.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_del.setToolTip("Delete this step")
        self._btn_del.clicked.connect(lambda: self.deleteRequested.emit(self))
        lay.addWidget(self._btn_del)

    def update_from(self, index: int, step: dict[str, Any], *, experiment_type: str | None) -> None:
        act = get_action(step.get("action", ""))
        if act is None:
            self._title_lbl.setText(step.get("action", "(unknown)"))
            self._chip_lbl.setText("")
            self._icon_lbl.clear()
            self._warn_lbl.setVisible(True)
            self._warn_lbl.setToolTip(f"Unknown action {step.get('action')!r}")
            self._num_lbl.setText(f"{index + 1}.")
            self._apply_category_color(Category.NEUTRAL)
            return
        self._title_lbl.setText(act.label)
        chip = describe_step(step)
        self._chip_lbl.setText(chip)
        self._icon_lbl.setPixmap(icon(act.icon, category=act.category).pixmap(20, 20))
        self._num_lbl.setText(f"{index + 1}.")
        self._apply_category_color(act.category)
        issues = validation_issues(step, experiment_type=experiment_type)
        if issues:
            self._warn_lbl.setVisible(True)
            self._warn_lbl.setToolTip("\n".join(f"⚠ {msg}" for msg in issues))
        else:
            self._warn_lbl.setVisible(False)
            self._warn_lbl.setToolTip("")

    def _apply_category_color(self, category: Category) -> None:
        col = category_color(category)
        self.setStyleSheet(
            f"QFrame#ScriptStepCard {{"
            f"  border-left: 4px solid {col};"
            f"  border-top: 1px solid palette(mid);"
            f"  border-right: 1px solid palette(mid);"
            f"  border-bottom: 1px solid palette(mid);"
            f"  border-radius: 6px;"
            f"  background: palette(base);"
            f"}}"
        )


# ---------------------------------------------------------------------------
# Delegate: disables focus rectangle (cards handle their own appearance).
# ---------------------------------------------------------------------------

class _CardDelegate(QStyledItemDelegate):
    """Paint the default item background without a focus rectangle."""

    def paint(self, painter, option: QStyleOptionViewItem, index) -> None:
        from PyQt6.QtWidgets import QStyle

        opt = QStyleOptionViewItem(option)
        opt.state &= ~QStyle.StateFlag.State_HasFocus
        super().paint(painter, opt, index)


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------

_STEP_ROLE = Qt.ItemDataRole.UserRole + 1


class Canvas(QWidget):
    """Script editor's center pane — script-level header + step list."""

    stepsChanged = pyqtSignal()
    stepSelected = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._experiment_type: str | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        # Script-level header (name + start/end with inherit checkboxes)
        self._header = QFrame(self)
        self._header.setObjectName("ScriptHeader")
        self._header.setFrameShape(QFrame.Shape.NoFrame)
        self._header.setStyleSheet(
            "QFrame#ScriptHeader { border-radius: 6px; "
            "background: palette(alternate-base); padding: 8px; }"
        )
        header_lay = QFormLayout(self._header)
        header_lay.setHorizontalSpacing(10)
        header_lay.setVerticalSpacing(6)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("script name (e.g. quick_overview)")
        self._name_edit.textChanged.connect(self._on_name_changed)
        header_lay.addRow("Name:", self._name_edit)

        # Start with inherit checkbox
        start_row = QHBoxLayout()
        self._start_inherit = QCheckBox("inherit from UI")
        self._start_inherit.setChecked(True)
        self._start_spin = QDoubleSpinBox()
        self._start_spin.setRange(0, 1_000_000)
        self._start_spin.setSuffix(" min")
        self._start_spin.setEnabled(False)
        self._start_inherit.toggled.connect(self._on_start_inherit_toggled)
        self._start_spin.valueChanged.connect(self._emit_changed)
        start_row.addWidget(self._start_spin)
        start_row.addWidget(self._start_inherit)
        header_lay.addRow("Start:", start_row)

        end_row = QHBoxLayout()
        self._end_inherit = QCheckBox("inherit from UI")
        self._end_inherit.setChecked(True)
        self._end_spin = QDoubleSpinBox()
        self._end_spin.setRange(0, 1_000_000)
        self._end_spin.setSuffix(" min")
        self._end_spin.setEnabled(False)
        self._end_inherit.toggled.connect(self._on_end_inherit_toggled)
        self._end_spin.valueChanged.connect(self._emit_changed)
        end_row.addWidget(self._end_spin)
        end_row.addWidget(self._end_inherit)
        header_lay.addRow("End:", end_row)

        outer.addWidget(self._header)

        # Step list
        self._list = QListWidget(self)
        self._list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self._list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._list.setMovement(QListWidget.Movement.Snap)
        self._list.setSpacing(4)
        self._list.setItemDelegate(_CardDelegate(self._list))
        self._list.setStyleSheet(
            "QListWidget { background: transparent; border: none; }"
            "QListWidget::item { padding: 0; }"
            "QListWidget::item:selected { background: transparent; }"
        )
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
        self._list.model().rowsMoved.connect(self._on_rows_moved)
        outer.addWidget(self._list, 1)

        # Empty-state hint
        self._empty_hint = QLabel(
            "No steps yet. Pick an action from the palette on the left to add "
            "your first step.",
            self,
        )
        self._empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_hint.setStyleSheet(
            "color: palette(mid); font-style: italic; padding: 20px;"
        )
        self._empty_hint.setWordWrap(True)
        outer.addWidget(self._empty_hint)

        self._update_empty_hint()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_experiment_type(self, et: str | None) -> None:
        self._experiment_type = et
        # Re-render each step so warnings update.
        for i in range(self._list.count()):
            self._rerender_row(i)

    def load_script(self, script: dict[str, Any] | None) -> None:
        """Populate the canvas from *script* (a dict like ``{name, start, end, steps}``).

        Passing ``None`` clears the canvas.
        """
        self._list.blockSignals(True)
        self._list.clear()
        if script is None:
            self._name_edit.blockSignals(True)
            self._name_edit.setText("")
            self._name_edit.blockSignals(False)
            self._set_range_from_script({})
            self._list.blockSignals(False)
            self._update_empty_hint()
            self.stepSelected.emit(-1)
            return
        self._name_edit.blockSignals(True)
        self._name_edit.setText(str(script.get("name", "")))
        self._name_edit.blockSignals(False)
        self._set_range_from_script(script)
        for step in (script.get("steps") or []):
            if isinstance(step, dict):
                self._append_item(dict(step))
        self._list.blockSignals(False)
        self._update_empty_hint()
        if self._list.count() > 0:
            self._list.setCurrentRow(0)
        else:
            self.stepSelected.emit(-1)

    def current_script(self) -> dict[str, Any]:
        """Return the current in-memory script dict (fresh copy)."""
        out: dict[str, Any] = {"name": self._name_edit.text().strip()}
        if not self._start_inherit.isChecked():
            out["start"] = float(self._start_spin.value())
        if not self._end_inherit.isChecked():
            out["end"] = float(self._end_spin.value())
        out["steps"] = [self._step_at(i) for i in range(self._list.count())]
        return out

    def selected_index(self) -> int:
        row = self._list.currentRow()
        return row if row >= 0 else -1

    def selected_step(self) -> dict[str, Any] | None:
        row = self._list.currentRow()
        return self._step_at(row) if row >= 0 else None

    def update_selected_step(self, step: dict[str, Any]) -> None:
        row = self._list.currentRow()
        if row < 0:
            return
        item = self._list.item(row)
        item.setData(_STEP_ROLE, dict(step))
        self._rerender_row(row)
        self._emit_changed()

    def append_step(self, action_name: str) -> None:
        """Add a new step with default values for *action_name*."""
        act = get_action(action_name)
        if act is None:
            return
        step: dict[str, Any] = {"action": action_name}
        for p in act.params:
            if p.default is not None and not p.inheritable:
                step[p.key] = p.default
        self._append_item(step)
        self._list.setCurrentRow(self._list.count() - 1)
        self._emit_changed()
        self._update_empty_hint()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _append_item(self, step: dict[str, Any]) -> None:
        card = _StepCard(self._list)
        item = QListWidgetItem(self._list)
        item.setData(_STEP_ROLE, step)
        item.setSizeHint(QSize(0, 60))
        self._list.addItem(item)
        self._list.setItemWidget(item, card)
        card.deleteRequested.connect(lambda _c=card, i=item: self._delete_item(i))
        card.menuRequested.connect(lambda _c=card, i=item: self._show_context_menu(i))
        card.update_from(self._list.row(item), step, experiment_type=self._experiment_type)

    def _rerender_row(self, row: int) -> None:
        if row < 0 or row >= self._list.count():
            return
        item = self._list.item(row)
        card = self._list.itemWidget(item)
        step = item.data(_STEP_ROLE)
        if isinstance(card, _StepCard) and isinstance(step, dict):
            card.update_from(row, step, experiment_type=self._experiment_type)

    def _rerender_all(self) -> None:
        for i in range(self._list.count()):
            self._rerender_row(i)

    def _step_at(self, row: int) -> dict[str, Any]:
        if row < 0 or row >= self._list.count():
            return {}
        step = self._list.item(row).data(_STEP_ROLE)
        return dict(step) if isinstance(step, dict) else {}

    def _delete_item(self, item: QListWidgetItem) -> None:
        row = self._list.row(item)
        self._list.takeItem(row)
        self._rerender_all()
        self._update_empty_hint()
        self._emit_changed()

    def _show_context_menu(self, item: QListWidgetItem) -> None:
        row = self._list.row(item)
        menu = QMenu(self)
        act_dup = menu.addAction(icon("save_as"), "Duplicate")
        act_up = menu.addAction("Move up")
        act_down = menu.addAction("Move down")
        if row == 0:
            act_up.setEnabled(False)
        if row == self._list.count() - 1:
            act_down.setEnabled(False)
        chosen = menu.exec(self._list.mapToGlobal(
            self._list.visualItemRect(item).bottomLeft()
        ))
        if chosen is act_dup:
            clone = dict(self._step_at(row))
            self._append_item(clone)
            self._list.setCurrentRow(self._list.count() - 1)
            self._emit_changed()
        elif chosen is act_up:
            self._move_row(row, row - 1)
        elif chosen is act_down:
            self._move_row(row, row + 1)

    def _move_row(self, src: int, dst: int) -> None:
        if dst < 0 or dst >= self._list.count():
            return
        step = self._step_at(src)
        self._list.takeItem(src)
        self._insert_item_at(dst, step)
        self._list.setCurrentRow(dst)
        self._rerender_all()
        self._emit_changed()

    def _insert_item_at(self, row: int, step: dict[str, Any]) -> None:
        card = _StepCard(self._list)
        item = QListWidgetItem()
        item.setData(_STEP_ROLE, step)
        item.setSizeHint(QSize(0, 60))
        self._list.insertItem(row, item)
        self._list.setItemWidget(item, card)
        card.deleteRequested.connect(lambda _c=card, i=item: self._delete_item(i))
        card.menuRequested.connect(lambda _c=card, i=item: self._show_context_menu(i))
        card.update_from(row, step, experiment_type=self._experiment_type)

    def _update_empty_hint(self) -> None:
        self._empty_hint.setVisible(self._list.count() == 0)

    def _on_selection_changed(self) -> None:
        self.stepSelected.emit(self.selected_index())

    def _on_rows_moved(self, *_: Any) -> None:
        self._rerender_all()
        self._emit_changed()

    def _on_name_changed(self, _text: str) -> None:
        self._emit_changed()

    def _on_start_inherit_toggled(self, inherited: bool) -> None:
        self._start_spin.setEnabled(not inherited)
        self._emit_changed()

    def _on_end_inherit_toggled(self, inherited: bool) -> None:
        self._end_spin.setEnabled(not inherited)
        self._emit_changed()

    def _set_range_from_script(self, script: dict[str, Any]) -> None:
        for key, inherit_cb, spin in (
            ("start", self._start_inherit, self._start_spin),
            ("end", self._end_inherit, self._end_spin),
        ):
            inherit_cb.blockSignals(True)
            spin.blockSignals(True)
            if key in script and script[key] is not None:
                inherit_cb.setChecked(False)
                try:
                    spin.setValue(float(script[key]))
                except (TypeError, ValueError):
                    spin.setValue(0)
                spin.setEnabled(True)
            else:
                inherit_cb.setChecked(True)
                spin.setValue(0)
                spin.setEnabled(False)
            inherit_cb.blockSignals(False)
            spin.blockSignals(False)

    def _emit_changed(self) -> None:
        self.stepsChanged.emit()
