"""The Experiment Editor: one recording's description and design.

``pyflic-experiment [folder | flic_config.yaml]`` (or ``pyflic experiment``).

A short form for the moment a recording is filed, usually before any Project
exists.  It edits three things in ``flic_config.yaml`` and nothing else:

* ``about:`` — what the recording is: title, purpose, experimenter, series,
  the flies, the protocol, notes, and what each design factor and level means
  (ADR-0024).  Every field is optional.
* the design, under ``global:`` — the Experiment Type, the Chamber Layout,
  the well names, the Experimental Design Factors and the control level: the
  Design Copy keys of ADR-0025.
* ``dfms:`` — each DFM's id and which levels each chamber had (and, for
  Progressive Ratio, its paired chambers).

Everything else the file holds — detection ``params:``, ``constants:``,
Slices, the Selection, optogenetics, per-DFM overrides, ``scripts:``,
``data_folder:`` — belongs to the Config Editor and is written back exactly
as it was read.  Inside a Project with a Design the design is shown, not
edited, and saving writes the Design Copy, as the Config Editor does.

The ``global:`` it writes for a new recording is the smallest one the loader
accepts: every analysis setting is the Experiment Type's default until the
Config Editor or a Project Design says otherwise (ADR-0011).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import about as about_mod
from .config_editor import (
    _LAYOUT_LABELS,
    DFMWidget,
    _layout_chamber_size,
    _normalise_layout,
    _WestTabBar,
)
from .gui_env import sanitize_input_method_environment
from .ui import ActionButton, Card, Category, TopBar, apply_theme
from .ui import settings as ui_settings
from .ui.widgets import NoWheelComboBox

CONFIG_NAME = "flic_config.yaml"
WINDOW_TITLE = "FLIC Experiment Editor"
#: The ``global:`` keys this editor writes; every other one is kept as read.
_DESIGN_KEYS = ("experiment_type", "chamber_layout", "well_names",
                "experimental_design_factors", "control_level")


def _text(widget) -> str:
    if isinstance(widget, QPlainTextEdit):
        return widget.toPlainText().strip()
    return widget.text().strip()


# ---------------------------------------------------------------------------
# Factors: name, kind and levels
# ---------------------------------------------------------------------------


class FactorTable(QWidget):
    """The design factors, one row each: name, kind (``about:``) and levels
    (``experimental_design_factors``).  Under a Project Design the names and
    levels are the Design's and cannot be edited; the kind, which describes
    this recording, still can."""

    changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Factor", "Kind", "Levels (comma-separated)"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(110)
        self.table.itemChanged.connect(lambda _item: self.changed.emit())
        layout.addWidget(self.table)
        row = QHBoxLayout()
        self.add_button = ActionButton("Add factor", Category.LOAD, icon_name="new")
        self.add_button.clicked.connect(lambda _checked=False: self.add_row())
        self.remove_button = ActionButton("Remove", Category.QC, icon_name="clear")
        self.remove_button.clicked.connect(self._remove_row)
        row.addWidget(self.add_button)
        row.addWidget(self.remove_button)
        row.addStretch(1)
        layout.addLayout(row)
        self._read_only = False

    def _kind_combo(self, kind: str) -> NoWheelComboBox:
        combo = NoWheelComboBox()
        combo.addItem("", "")
        for value in about_mod.FACTOR_KINDS:
            combo.addItem(value, value)
        if kind and combo.findData(kind) < 0:
            combo.addItem(kind, kind)          # kept as the file has it
        combo.setCurrentIndex(max(combo.findData(kind), 0))
        combo.setToolTip("What kind of variable this factor is (about.factors)")
        combo.currentIndexChanged.connect(lambda _i: self.changed.emit())
        return combo

    def add_row(self, name: str = "", kind: str = "", levels: list[str] | None = None) -> None:
        self.table.blockSignals(True)
        r = self.table.rowCount()
        self.table.insertRow(r)
        name_item = QTableWidgetItem(name)
        levels_item = QTableWidgetItem(", ".join(levels or []))
        if self._read_only:
            for item in (name_item, levels_item):
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table.setItem(r, 0, name_item)
        self.table.setCellWidget(r, 1, self._kind_combo(kind))
        self.table.setItem(r, 2, levels_item)
        self.table.blockSignals(False)
        self.changed.emit()

    def _remove_row(self) -> None:
        row = self.table.currentRow()
        if row >= 0 and not self._read_only:
            self.table.removeRow(row)
            self.changed.emit()

    def load(self, factors: dict, kinds: dict[str, str]) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        self.table.blockSignals(False)
        for name, levels in (factors or {}).items():
            values = [str(v) for v in levels] if isinstance(levels, list) else \
                [part.strip() for part in str(levels or "").split(",") if part.strip()]
            self.add_row(str(name), kinds.get(str(name), ""), values)

    def factors(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for r in range(self.table.rowCount()):
            name = (self.table.item(r, 0).text() if self.table.item(r, 0) else "").strip()
            levels = (self.table.item(r, 2).text() if self.table.item(r, 2) else "")
            if name:
                out[name] = [v.strip() for v in levels.split(",") if v.strip()]
        return out

    def kinds(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for r in range(self.table.rowCount()):
            name = (self.table.item(r, 0).text() if self.table.item(r, 0) else "").strip()
            combo = self.table.cellWidget(r, 1)
            kind = combo.currentData() if combo is not None else ""
            if name and kind:
                out[name] = kind
        return out

    def set_read_only(self, read_only: bool, reason: str = "") -> None:
        self._read_only = read_only
        for r in range(self.table.rowCount()):
            for c in (0, 2):
                item = self.table.item(r, c)
                if item is None:
                    continue
                flags = item.flags()
                item.setFlags(flags & ~Qt.ItemFlag.ItemIsEditable if read_only
                              else flags | Qt.ItemFlag.ItemIsEditable)
                item.setToolTip(reason if read_only else "")
        for button in (self.add_button, self.remove_button):
            button.setEnabled(not read_only)
            button.setToolTip(reason if read_only else button.text())


class MeaningTable(QTableWidget):
    """What each level means (``about.factors.<factor>.levels``): one row per
    declared level, rebuilt when the factors change, keeping what was typed."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(0, 3, parent)
        self.setHorizontalHeaderLabels(["Factor", "Level", "What it means"])
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().setVisible(False)
        self.setMinimumHeight(140)
        self.setToolTip("A few words per level: the cross, the dose, the diet.  "
                        "Optional; written under about.factors.")

    def refresh(self, factors: dict[str, list[str]],
                meanings: dict[str, dict[str, str]] | None = None) -> None:
        kept = self.meanings() if meanings is None else meanings
        self.setRowCount(0)
        for factor, levels in factors.items():
            for level in levels:
                r = self.rowCount()
                self.insertRow(r)
                for c, text in ((0, factor), (1, level)):
                    item = QTableWidgetItem(text)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.setItem(r, c, item)
                self.setItem(r, 2, QTableWidgetItem(kept.get(factor, {}).get(level, "")))

    def meanings(self) -> dict[str, dict[str, str]]:
        out: dict[str, dict[str, str]] = {}
        for r in range(self.rowCount()):
            factor, level, meaning = (self.item(r, c).text() if self.item(r, c) else ""
                                      for c in range(3))
            if meaning.strip():
                out.setdefault(factor, {})[level] = meaning.strip()
        return out


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------


class ExperimentEditor(QMainWindow):
    """One recording's ``about:``, design and chamber assignments."""

    #: The path of a config just written.
    saved = pyqtSignal(str)

    def __init__(self, initial_path: str | Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self._current_path: Path | None = None
        self._loaded_raw: dict[str, Any] = {}
        self._design: dict | None = None
        self._design_source: str | None = None
        self._dfm_widgets: list[DFMWidget] = []
        self._last_layout = "two_well"
        self._last_type: str | None = None
        #: Sex values the file holds that have no checkbox; written back.
        self._other_sex: list[str] = []
        self._build_menu()
        self._build_ui()
        self._fill_type_combo()
        self._on_type_changed()
        if initial_path is not None:
            self.open_path(initial_path)
        self.resize(980, 760)

    # ---- construction --------------------------------------------------

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("&File")
        for text, shortcut, slot in (
                ("&Open…", "Ctrl+O", self._open),
                ("&Save", "Ctrl+S", self._save),
                ("Save &As…", "Ctrl+Shift+S", self._save_as),
                ("Open in the &Config Editor", None, self._open_config_editor)):
            action = QAction(text, self)
            if shortcut:
                action.setShortcut(shortcut)
            action.triggered.connect(lambda _checked=False, s=slot: s())
            menu.addAction(action)

    def _build_ui(self) -> None:
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(TopBar(WINDOW_TITLE))
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.tabs.addTab(self._scrolled(self._build_about_page()), "About")
        self.tabs.addTab(self._scrolled(self._build_design_page()), "Design")
        self.tabs.addTab(self._build_chambers_page(), "Chambers")

    @staticmethod
    def _scrolled(page: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(page)
        return scroll

    @staticmethod
    def _form() -> QFormLayout:
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        return form

    @staticmethod
    def _notes_edit(placeholder: str, height: int = 70) -> QPlainTextEdit:
        edit = QPlainTextEdit()
        edit.setPlaceholderText(placeholder)
        edit.setFixedHeight(height)
        return edit

    def _build_about_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        card = Card("About this recording", Category.LOAD,
                    subtitle="Every field is optional and is saved under about:. "
                             "pyflic never uses it to analyse; it is for whoever "
                             "reads this config later.")
        form = self._form()
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("One line, such as: Progressive ratio, sucrose, replicate 1")
        self.title_count = QLabel("")
        self.title_count.setObjectName("PyflicCardSubtitle")
        self.title_edit.textChanged.connect(self._on_title_changed)
        title_row = QHBoxLayout()
        title_row.addWidget(self.title_edit, 1)
        title_row.addWidget(self.title_count)
        form.addRow("Title:", title_row)
        self.purpose_edit = self._notes_edit(
            "One or two sentences: what was varied, in which flies, and what was measured.")
        form.addRow("Purpose:", self.purpose_edit)
        self.experimenter_edit = QLineEdit()
        self.experimenter_edit.setPlaceholderText("Who ran it, such as your lab handle")
        form.addRow("Experimenter:", self.experimenter_edit)
        self.series_edit = QLineEdit()
        self.series_edit.setPlaceholderText(
            "Shared by replicate recordings, such as pr-sucrose-2026")
        form.addRow("Series:", self.series_edit)
        self.started_label = QLabel("")
        self.started_label.setObjectName("PyflicCardSubtitle")
        self.started_label.setWordWrap(True)
        form.addRow("Recorded:", self.started_label)
        card.add_body(form)
        layout.addWidget(card)

        flies = Card("The flies", Category.LOAD)
        form = self._form()
        self.organism_edit = QLineEdit()
        self.organism_edit.setPlaceholderText("Drosophila melanogaster")
        form.addRow("Organism:", self.organism_edit)
        sex_row = QHBoxLayout()
        self.sex_checks: dict[str, QCheckBox] = {}
        for value in about_mod.SEX_CHOICES:
            check = QCheckBox(value)
            self.sex_checks[value] = check
            sex_row.addWidget(check)
        self.other_sex_label = QLabel("")
        self.other_sex_label.setObjectName("PyflicCardSubtitle")
        sex_row.addWidget(self.other_sex_label)
        sex_row.addStretch(1)
        form.addRow("Sex:", sex_row)
        self.age_edit = QLineEdit()
        self.age_edit.setPlaceholderText("At the start of the recording, such as 5-7 days")
        form.addRow("Age at start:", self.age_edit)
        flies.add_body(form)
        layout.addWidget(flies)

        protocol = Card("Protocol and notes", Category.LOAD)
        form = self._form()
        self.protocol_edits = {field: QLineEdit() for field in about_mod.PROTOCOL_FIELDS}
        self.protocol_edits["name"].setPlaceholderText("Such as FLIC progressive ratio")
        self.protocol_edits["version"].setPlaceholderText("Such as 2")
        self.protocol_edits["link"].setPlaceholderText("Where the protocol is written down")
        form.addRow("Protocol:", self.protocol_edits["name"])
        form.addRow("Version:", self.protocol_edits["version"])
        form.addRow("Link:", self.protocol_edits["link"])
        self.notes_edit = self._notes_edit("Anything else worth knowing about this recording.", 90)
        form.addRow("Notes:", self.notes_edit)
        protocol.add_body(form)
        layout.addWidget(protocol)
        return page

    def _build_design_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.design_banner = QLabel("")
        self.design_banner.setWordWrap(True)
        self.design_banner.setVisible(False)
        self.design_banner.setStyleSheet(
            "border: 1px solid palette(mid); border-radius: 6px; padding: 6px 10px;")
        layout.addWidget(self.design_banner)

        card = Card("Experiment", Category.TOOLS,
                    subtitle="What kind of experiment this is, and what each chamber held.")
        self.design_form = self._form()
        self.type_combo = NoWheelComboBox()
        self.type_combo.setMaximumWidth(260)
        self.type_combo.currentIndexChanged.connect(lambda _i: self._on_type_changed())
        self.design_form.addRow("Experiment Type:", self.type_combo)
        self.layout_combo = NoWheelComboBox()
        for key, label in _LAYOUT_LABELS.items():
            self.layout_combo.addItem(label, key)
        self.layout_combo.setCurrentIndex(self.layout_combo.findData("two_well"))
        self.layout_combo.setMaximumWidth(260)
        self.layout_combo.currentIndexChanged.connect(lambda _i: self._on_layout_changed())
        self.layout_hint = QLabel("")
        self.layout_hint.setObjectName("PyflicCardSubtitle")
        layout_row = QHBoxLayout()
        layout_row.addWidget(self.layout_combo)
        layout_row.addWidget(self.layout_hint)
        layout_row.addStretch(1)
        self.design_form.addRow("Chamber Layout:", layout_row)
        self.well_a_edit = QLineEdit()
        self.well_a_edit.setPlaceholderText("What well A held, such as Sucrose")
        self.well_b_edit = QLineEdit()
        self.well_b_edit.setPlaceholderText("What well B held, such as Yeast")
        self._well_a_row = self.design_form.rowCount()
        self.design_form.addRow("Well A:", self.well_a_edit)
        self._well_b_row = self.design_form.rowCount()
        self.design_form.addRow("Well B:", self.well_b_edit)
        self.control_edit = QLineEdit()
        self.control_edit.setMaximumWidth(220)
        self._control_row = self.design_form.rowCount()
        self.design_form.addRow("Control Level:", self.control_edit)
        self.type_needs = QLabel("")
        self.type_needs.setObjectName("PyflicCardSubtitle")
        self.type_needs.setWordWrap(True)
        self.design_form.addRow("", self.type_needs)
        card.add_body(self.design_form)
        layout.addWidget(card)

        factors = Card("Design factors", Category.TOOLS,
                       subtitle="Each factor and its levels.  A chamber is assigned "
                                "one level of every factor, in this order.  Leave the "
                                "table empty to name each chamber's treatment directly.")
        self.factor_table = FactorTable()
        self.factor_table.changed.connect(self._on_factors_changed)
        factors.add_body(self.factor_table)
        layout.addWidget(factors)

        meanings = Card("What each level means", Category.TOOLS,
                        subtitle="Optional: a few words per level, such as the cross "
                                 "or the dose.")
        self.meaning_table = MeaningTable()
        meanings.add_body(self.meaning_table)
        layout.addWidget(meanings)
        return page

    def _build_chambers_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        self.data_label = QLabel("")
        self.data_label.setWordWrap(True)
        self.data_label.setObjectName("PyflicCardSubtitle")
        layout.addWidget(self.data_label)
        self.dfm_tabs = QTabWidget()
        self.dfm_tabs.setTabBar(_WestTabBar())
        self.dfm_tabs.setTabPosition(QTabWidget.TabPosition.West)
        layout.addWidget(self.dfm_tabs, 1)
        row = QHBoxLayout()
        self.add_dfm_button = ActionButton("Add DFM", Category.LOAD, icon_name="new")
        self.add_dfm_button.clicked.connect(lambda _checked=False: self._add_dfm())
        self.remove_dfm_button = ActionButton("Remove DFM", Category.QC, icon_name="clear")
        self.remove_dfm_button.clicked.connect(lambda _checked=False: self._remove_dfm())
        self.add_found_button = ActionButton("Add the DFMs found in the data", Category.LOAD)
        self.add_found_button.clicked.connect(lambda _checked=False: self._add_found_dfms())
        for button in (self.add_dfm_button, self.remove_dfm_button, self.add_found_button):
            row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)
        return page

    # ---- the Experiment Type and the layout ----------------------------

    def _where(self) -> Path | None:
        return self._current_path.parent if self._current_path is not None else None

    def _fill_type_combo(self, select: str | None = None) -> None:
        from . import experiment_types

        current = select or self.type_combo.currentData()
        self.type_combo.blockSignals(True)
        self.type_combo.clear()
        for item in experiment_types.available_experiment_types(self._where()):
            label = item.display_name if item.is_shipped else f"{item.display_name}  (lab)"
            self.type_combo.addItem(label, item.name)
        index = self.type_combo.findData(current) if current else -1
        self.type_combo.setCurrentIndex(max(index, 0))
        self.type_combo.blockSignals(False)

    def current_type(self):
        from . import experiment_types

        try:
            return experiment_types.get_experiment_type(self.type_combo.currentData(),
                                                        self._where())
        except Exception:  # noqa: BLE001 - an unknown name is a Custom Experiment
            return experiment_types.get_experiment_type(None)

    def chamber_layout(self) -> str:
        item = self.current_type()
        return item.chamber_layout or str(self.layout_combo.currentData() or "two_well")

    def _on_type_changed(self) -> None:
        item = self.current_type()
        if item.chamber_layout is not None and not self._set_layout(item.chamber_layout):
            ## Declined: the type that asked for the layout is not taken either.
            self.type_combo.blockSignals(True)
            self.type_combo.setCurrentIndex(max(self.type_combo.findData(self._last_type), 0))
            self.type_combo.blockSignals(False)
            return
        self._last_type = item.name
        governed = bool(self._design)
        self.layout_combo.setEnabled(item.layout_free and not governed)
        self.layout_hint.setText("" if item.layout_free else f"set by {item.display_name}")
        factor = item.definition.treatment_control if item.definition is not None else None
        self.design_form.setRowVisible(self._control_row, factor is not None)
        if factor is not None:
            self.control_edit.setPlaceholderText(f"the level of {factor} others are compared with")
        needs = list(item.definition.factors) if item.definition is not None else []
        self.type_needs.setText(
            f"{item.display_name} needs the factor{'s' if len(needs) > 1 else ''} "
            f"{', '.join(needs)} declared below." if needs else "")
        self.type_needs.setVisible(bool(needs))
        for widget in self._dfm_widgets:
            widget.set_progressive_ratio(item.has_module("ratio"))
        self._sync_wells()

    def _set_layout(self, layout: str) -> bool:
        """Move to *layout*, asking first when chambers would be discarded;
        False when declined."""
        if layout == self._last_layout:
            index = self.layout_combo.findData(layout)
            if index != self.layout_combo.currentIndex():
                self.layout_combo.blockSignals(True)
                self.layout_combo.setCurrentIndex(index)
                self.layout_combo.blockSignals(False)
            return True
        keep = 12 if layout == "single_well" else 6
        losing = [f"DFM {w._id_spin.value()} chamber {c}: {text}"
                  for w in self._dfm_widgets for c, text in w.assignments_beyond(keep)]
        if losing:
            answer = QMessageBox.question(
                self, "Discard chamber assignments?",
                "Six chambers per DFM leaves out these assignments:\n\n"
                + "\n".join(f"  • {line}" for line in losing[:12])
                + ("\n  • …" if len(losing) > 12 else "") + "\n\nSwitch anyway?")
            if answer != QMessageBox.StandardButton.Yes:
                self.layout_combo.blockSignals(True)
                self.layout_combo.setCurrentIndex(self.layout_combo.findData(self._last_layout))
                self.layout_combo.blockSignals(False)
                return False
        self.layout_combo.blockSignals(True)
        self.layout_combo.setCurrentIndex(self.layout_combo.findData(layout))
        self.layout_combo.blockSignals(False)
        self._last_layout = layout
        for widget in self._dfm_widgets:
            widget.update_chamber_size(_layout_chamber_size(layout))
        self._sync_wells()
        return True

    def _on_layout_changed(self) -> None:
        self._set_layout(str(self.layout_combo.currentData() or "two_well"))

    def _sync_wells(self) -> None:
        two_well = self.chamber_layout() == "two_well"
        self.design_form.setRowVisible(self._well_a_row, two_well)
        self.design_form.setRowVisible(self._well_b_row, two_well)

    def _on_factors_changed(self) -> None:
        factors = self.factor_table.factors()
        self.meaning_table.refresh(factors)
        for widget in self._dfm_widgets:
            widget.update_factors(factors)

    def _on_title_changed(self, text: str) -> None:
        n = len(text.strip())
        self.title_count.setText(f"{n}/{about_mod.TITLE_LIMIT}" if n else "")
        self.title_count.setStyleSheet("color: #d97706;" if n > about_mod.TITLE_LIMIT else "")

    # ---- DFMs ----------------------------------------------------------

    def _append_dfm(self, dfm_id: int) -> DFMWidget:
        widget = DFMWidget(dfm_id=dfm_id, chamber_size=_layout_chamber_size(self.chamber_layout()),
                           simple=True)
        widget.update_factors(self.factor_table.factors())
        widget.set_progressive_ratio(self.current_type().has_module("ratio"))
        widget._id_spin.valueChanged.connect(
            lambda value, w=widget: self._on_dfm_id_changed(w, value))
        self._dfm_widgets.append(widget)
        self.dfm_tabs.addTab(self._scrolled(widget), f"DFM {dfm_id}")
        return widget

    def _on_dfm_id_changed(self, widget: DFMWidget, value: int) -> None:
        if widget in self._dfm_widgets:
            self.dfm_tabs.setTabText(self._dfm_widgets.index(widget), f"DFM {value}")
        self._refresh_data_label()

    def _clear_dfms(self) -> None:
        while self._dfm_widgets:
            self.dfm_tabs.removeTab(self.dfm_tabs.count() - 1)
            self._dfm_widgets.pop().deleteLater()

    def _add_dfm(self, dfm_id: int | None = None) -> DFMWidget:
        used = {w._id_spin.value() for w in self._dfm_widgets}
        if dfm_id is None:
            dfm_id = next(k for k in range(1, 200) if k not in used)
        widget = self._append_dfm(dfm_id)
        self.dfm_tabs.setCurrentIndex(self.dfm_tabs.count() - 1)
        self._refresh_data_label()
        return widget

    def _remove_dfm(self) -> None:
        index = self.dfm_tabs.currentIndex()
        if index < 0:
            return
        widget = self._dfm_widgets[index]
        if any(widget.get_dict().get("chambers", {}).values()):
            answer = QMessageBox.question(
                self, "Remove DFM?",
                f"DFM {widget._id_spin.value()} has chamber assignments.  Remove it?")
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.dfm_tabs.removeTab(index)
        self._dfm_widgets.pop(index).deleteLater()
        self._refresh_data_label()

    def _found_dfms(self) -> list[int]:
        """The DFM ids the recording's files name: in the folder its config
        reads (data/, or data_folder:), else loose beside the config."""
        directory = self._where()
        if directory is None or not directory.is_dir():
            return []
        from . import layout as layout_mod
        from .project import dfm_ids_in_data, dfm_ids_in_folder

        try:
            if self._loaded_raw.get(layout_mod.DATA_FOLDER_KEY):
                location = layout_mod.data_location(str(directory), config=self._loaded_raw)
                return [] if location.problem else dfm_ids_in_folder(location.path)
            return dfm_ids_in_data(directory) or layout_mod.dfm_ids(directory)
        except Exception:  # noqa: BLE001 - finding DFMs is a convenience
            return []

    def _add_found_dfms(self) -> None:
        have = {w._id_spin.value() for w in self._dfm_widgets}
        for dfm_id in self._found_dfms():
            if dfm_id not in have:
                self._append_dfm(dfm_id)
        self._refresh_data_label()

    def _refresh_data_label(self) -> None:
        found = self._found_dfms()
        have = sorted(w._id_spin.value() for w in self._dfm_widgets)
        if not found:
            self.data_label.setText("No DFM files found for this recording yet.")
            self.add_found_button.setVisible(False)
            return
        missing = [i for i in found if i not in have]
        extra = [i for i in have if i not in found]
        parts = [f"The data has DFM {', '.join(map(str, found))}."]
        if missing:
            parts.append(f"Not here yet: {', '.join(map(str, missing))}.")
        if extra:
            parts.append(f"No data for: {', '.join(map(str, extra))}.")
        self.data_label.setText("  ".join(parts))
        self.add_found_button.setVisible(bool(missing))

    def _refresh_started(self) -> None:
        directory = self._where()
        text = "Program.txt not found: the start is read from it when it is there."
        if directory is not None and directory.is_dir():
            try:
                from . import layout as layout_mod
                from .opto_program import find_program, read_program

                location = layout_mod.data_location(str(directory), config=self._loaded_raw)
                program = None if location.problem else find_program(location.path)
                if program is not None:
                    start = read_program(program).start
                    if start is not None:
                        text = f"Started {start:%Y-%m-%d %H:%M}, from Program.txt"
            except Exception:  # noqa: BLE001 - a date shown is a convenience
                pass
        self.started_label.setText(text)

    # ---- loading -------------------------------------------------------

    def open_path(self, path: str | Path) -> None:
        """Open a ``flic_config.yaml``, or a folder: its config if it has one,
        and otherwise a new config for it, one DFM tab per DFM in its data."""
        p = Path(path).expanduser()
        config = p if p.is_file() else p / CONFIG_NAME
        if p.is_dir() and not config.exists() and (p / "flic_config.yml").exists():
            config = p / "flic_config.yml"
        if config.is_file():
            try:
                raw = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
            except Exception as err:  # noqa: BLE001
                QMessageBox.critical(self, WINDOW_TITLE, f"Could not read {config}:\n{err}")
                return
            if not isinstance(raw, dict):
                QMessageBox.critical(self, WINDOW_TITLE, f"{config} is not a YAML mapping.")
                return
            self._current_path = config
            self.load_config(raw)
            self.statusBar().showMessage(f"Opened {config}")
        else:
            self._current_path = config
            self.load_config({})
            found = self._found_dfms()
            for dfm_id in found:
                self._append_dfm(dfm_id)
            self._refresh_data_label()
            self.statusBar().showMessage(
                f"No {CONFIG_NAME} here yet" + (f": DFM {', '.join(map(str, found))} found "
                                                f"in the data" if found else "")
                + ".  Save writes it.")
        self.setWindowTitle(f"{WINDOW_TITLE} — {self._current_path}")

    def load_config(self, raw: dict) -> None:
        """Fill the form from *raw*, a parsed ``flic_config.yaml``."""
        self._loaded_raw = dict(raw)
        self._read_design()
        about = about_mod.read_about(raw)
        self._load_about(about)
        own = raw.get("global") if isinstance(raw.get("global"), dict) else {}
        shown = dict(self._design) if self._design else dict(own)
        self._load_design(shown, about)
        self._clear_dfms()
        chamber_size = _layout_chamber_size(self.chamber_layout())
        for node in (raw.get("dfms") or raw.get("DFMs") or []):
            if not isinstance(node, dict):
                continue
            try:
                dfm_id = int(node.get("id", node.get("ID")))
            except (TypeError, ValueError):
                continue
            widget = self._append_dfm(dfm_id)
            widget.load_dict(node, chamber_size)
        self._apply_governance()
        self._refresh_data_label()
        self._refresh_started()

    def _read_design(self) -> None:
        self._design, self._design_source = None, None
        if self._current_path is None:
            return
        from . import project as project_mod

        try:
            design, source = project_mod.design_for_member(self._current_path.parent)
        except Exception:  # noqa: BLE001 - a bad project.yaml governs nothing
            return
        if design:
            self._design, self._design_source = design, source

    def _load_about(self, about: dict) -> None:
        self.title_edit.setText(about_mod._text(about.get("title")))
        self.purpose_edit.setPlainText(about_mod._text(about.get("purpose")))
        self.experimenter_edit.setText(about_mod._text(about.get("experimenter")))
        self.series_edit.setText(about_mod._text(about.get("series")))
        self.organism_edit.setText(about_mod._text(about.get("organism")))
        self.age_edit.setText(about_mod._text(about.get("age_at_start")))
        self.notes_edit.setPlainText(about_mod._text(about.get("notes")))
        sexes = about_mod.sex_values(about.get(about_mod.SEX_KEY))
        lowered = [s.lower() for s in sexes]
        for value, check in self.sex_checks.items():
            check.setChecked(value in lowered)
        self._other_sex = [s for s in sexes if s.lower() not in self.sex_checks]
        self.other_sex_label.setText(
            f"also: {', '.join(self._other_sex)}" if self._other_sex else "")
        protocol = about.get(about_mod.PROTOCOL_KEY)
        if isinstance(protocol, str):
            protocol = {"name": protocol}
        protocol = protocol if isinstance(protocol, dict) else {}
        for field, edit in self.protocol_edits.items():
            edit.setText(about_mod._text(protocol.get(field)))

    def _load_design(self, g: dict, about: dict) -> None:
        from . import experiment_types

        raw_type = g.get("experiment_type")
        try:
            item = experiment_types.get_experiment_type(raw_type, self._where())
            legacy = None
        except ValueError:
            item = experiment_types.get_experiment_type(None)
            key = str(raw_type or "").lower()
            legacy = "single_well" if "single" in key else "two_well" if "two" in key else None
        self._fill_type_combo(item.name)
        if item.chamber_layout is not None:
            layout = item.chamber_layout
        else:
            params = g.get("params", g.get("parameters")) or {}
            size = params.get("chamber_size") if isinstance(params, dict) else None
            layout = (legacy or _normalise_layout(g.get("chamber_layout"))
                      or ("single_well" if str(size) == "1" else "two_well"))
        self._last_layout = layout
        self.layout_combo.blockSignals(True)
        self.layout_combo.setCurrentIndex(self.layout_combo.findData(layout))
        self.layout_combo.blockSignals(False)
        wells = g.get("well_names") if isinstance(g.get("well_names"), dict) else {}
        self.well_a_edit.setText(about_mod._text(wells.get("A")))
        self.well_b_edit.setText(about_mod._text(wells.get("B")))
        self.control_edit.setText(about_mod._text(g.get("control_level")))
        notes = about_mod.factor_notes(about)
        factors = g.get("experimental_design_factors")
        factors = factors if isinstance(factors, dict) else {}
        self.factor_table.load(factors, {k: v["kind"] for k, v in notes.items()})
        self.meaning_table.refresh(self.factor_table.factors(),
                                   {k: v["levels"] for k, v in notes.items()})
        self._on_type_changed()

    def _apply_governance(self) -> None:
        governed = bool(self._design)
        reason = f"Owned by the Project design in {self._design_source}" if governed else ""
        for widget in (self.type_combo, self.well_a_edit, self.well_b_edit, self.control_edit):
            widget.setEnabled(not governed)
            widget.setToolTip(reason)
        self.layout_combo.setEnabled(not governed and self.current_type().layout_free)
        self.factor_table.set_read_only(governed, reason)
        self.design_banner.setVisible(governed)
        if governed:
            self.design_banner.setText(
                f"<b>This recording is a member of the Project at "
                f"<code>{self._design_source}</code>.</b>  Its design is the "
                "Project's, shown here and edited there.  Saving writes a copy of it "
                "into this file, so the file reads on its own.  The About tab, the "
                "factor kinds and meanings, and the chambers are this recording's.")

    # ---- what is written -----------------------------------------------

    def about_section(self) -> dict:
        """``about:`` as the form describes it; keys the form has no field
        for are kept as read."""
        loaded = about_mod.read_about(self._loaded_raw)
        known = set(about_mod.KEY_ORDER)
        section: dict[str, Any] = {k: v for k, v in loaded.items() if k not in known}
        for key, widget in (("title", self.title_edit), ("purpose", self.purpose_edit),
                            ("experimenter", self.experimenter_edit),
                            ("series", self.series_edit), ("organism", self.organism_edit),
                            ("age_at_start", self.age_edit), ("notes", self.notes_edit)):
            section[key] = _text(widget)
        section[about_mod.SEX_KEY] = [v for v, c in self.sex_checks.items() if c.isChecked()] \
            + self._other_sex
        loaded_protocol = loaded.get(about_mod.PROTOCOL_KEY)
        protocol = {k: v for k, v in loaded_protocol.items()
                    if k not in about_mod.PROTOCOL_FIELDS} \
            if isinstance(loaded_protocol, dict) else {}
        protocol.update({field: _text(edit) for field, edit in self.protocol_edits.items()})
        section[about_mod.PROTOCOL_KEY] = protocol
        kinds = self.factor_table.kinds()
        meanings = self.meaning_table.meanings()
        factors: dict[str, Any] = {}
        for name in self.factor_table.factors():
            entry: dict[str, Any] = {}
            if kinds.get(name):
                entry["kind"] = kinds[name]
            if meanings.get(name):
                entry["levels"] = meanings[name]
            if entry:
                factors[name] = entry
        section[about_mod.FACTORS_KEY] = factors
        return about_mod.clean_about(section)

    def design_section(self) -> dict:
        """The design keys of ``global:`` as the form describes them."""
        item = self.current_type()
        g: dict[str, Any] = {}
        if not item.is_custom:
            g["experiment_type"] = item.name
        if item.layout_free:
            g["chamber_layout"] = self.chamber_layout()
        if self.chamber_layout() == "two_well":
            wells = {k: _text(e) for k, e in (("A", self.well_a_edit), ("B", self.well_b_edit))}
            wells = {k: v for k, v in wells.items() if v}
            if wells:
                g["well_names"] = wells
        factors = self.factor_table.factors()
        if factors:
            g["experimental_design_factors"] = factors
        has_control = item.definition is not None and item.definition.treatment_control
        if has_control and _text(self.control_edit):
            g["control_level"] = _text(self.control_edit)
        return g

    def global_section(self, dfms: list[dict]) -> dict | None:
        """The ``global:`` to write, or ``None`` to write none."""
        own = self._loaded_raw.get("global")
        own = dict(own) if isinstance(own, dict) else {}
        if self._design:
            from . import project as project_mod

            config = {**self._loaded_raw, "dfms": dfms}
            wanted = project_mod.member_global(config, self._design, self._where())
            if wanted is None:
                return own or None             # a conflict is a person's to settle
            return wanted or None
        rest = {k: v for k, v in own.items() if k not in _DESIGN_KEYS}
        params = rest.get("params")
        if isinstance(params, dict) and "chamber_size" in params:
            ## The layout says it now (ADR-0007); two statements could differ.
            params = {k: v for k, v in params.items() if k != "chamber_size"}
            rest["params"] = params
        if "params" not in rest and "parameters" not in rest and not any(
                isinstance(node, dict) and ("params" in node or "parameters" in node)
                for node in dfms):
            ## The loader needs a params: section somewhere; empty is every
            ## default, until the Config Editor or a Design says otherwise.
            rest["params"] = {}
        return {**self.design_section(), **rest}

    def collect(self) -> dict:
        """The whole config to write: the file as read, with ``about:``, the
        design and ``dfms:`` replaced by the form's."""
        from .layout import DATA_FOLDER_KEY

        raw = self._loaded_raw
        dfms = [w.get_dict() for w in self._dfm_widgets]
        about = self.about_section()
        g = self.global_section(dfms)
        out: dict[str, Any] = {}
        if DATA_FOLDER_KEY in raw:
            out[DATA_FOLDER_KEY] = raw[DATA_FOLDER_KEY]
        if about:
            out[about_mod.ABOUT_KEY] = about
        for key, value in raw.items():
            if key in (DATA_FOLDER_KEY, about_mod.ABOUT_KEY):
                continue
            if key == "global":
                if g:
                    out["global"] = g
            elif key in ("dfms", "DFMs"):
                if g and "global" not in out:
                    out["global"] = g
                out["dfms"] = dfms
            else:
                out[key] = value
        if g and "global" not in out:
            out["global"] = g
        out.setdefault("dfms", dfms)
        return out

    def problems(self) -> list[str]:
        """What would make the saved config wrong or refused: unassigned or
        half-assigned chambers, and what the Experiment Type requires."""
        found: list[str] = []
        for widget in self._dfm_widgets:
            found += [f"DFM {widget._id_spin.value()}: {p}" for p in widget.chamber_problems()]
        ids = [w._id_spin.value() for w in self._dfm_widgets]
        if len(ids) != len(set(ids)):
            found.append("two DFM tabs have the same id")
        if not self._design:
            found += self.current_type().validate(self.design_section())
        for _where, message in about_mod.about_problems(
                self.about_section(), (self._design or self.design_section()).get(
                    "experimental_design_factors") or {}):
            found.append(message)
        return found

    # ---- file actions ----------------------------------------------------

    def _open(self) -> None:
        start = str(self._where() or Path.cwd())
        path, _ = QFileDialog.getOpenFileName(self, "Open flic_config.yaml", start,
                                              "FLIC config (*.yaml *.yml)")
        if path:
            self.open_path(path)

    def _save(self) -> None:
        if self._current_path is None:
            self._save_as()
        else:
            self.write(self._current_path)

    def _save_as(self) -> None:
        start = str(self._current_path or Path.cwd() / CONFIG_NAME)
        path, _ = QFileDialog.getSaveFileName(self, "Save flic_config.yaml", start,
                                              "FLIC config (*.yaml *.yml)")
        if path:
            self._current_path = Path(path)
            self.write(self._current_path)

    def write(self, path: Path, *, confirm: bool = True) -> bool:
        """Write the config to *path*.  With *confirm*, problems are shown
        first and the save may be declined."""
        problems = self.problems()
        if problems and confirm:
            answer = QMessageBox.question(
                self, "Save with problems?",
                "This config has problems:\n\n"
                + "\n".join(f"  • {p}" for p in problems[:15])
                + ("\n  • …" if len(problems) > 15 else "") + "\n\nSave anyway?")
            if answer != QMessageBox.StandardButton.Yes:
                return False
        try:
            cfg = self.collect()
            Path(path).write_text(
                yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
            self._write_type_copy(Path(path))
        except Exception as err:  # noqa: BLE001
            QMessageBox.critical(self, WINDOW_TITLE, f"Could not save {path}:\n{err}")
            return False
        self._loaded_raw = cfg
        self.setWindowTitle(f"{WINDOW_TITLE} — {path}")
        self.statusBar().showMessage(f"Saved {path}")
        self.saved.emit(str(path))
        return True

    def _write_type_copy(self, path: Path) -> None:
        """A lab type travels with a standalone recording (ADR-0019), as the
        Config Editor carries it; a Member's is its Project's."""
        from .experiment_types import library

        directory = path.parent
        if self._design or library.copy_dir(directory) != directory:
            return
        item = self.current_type()
        if item.is_shipped:
            library.remove_copy(directory)
        else:
            library.write_copy(item.definition, directory)

    def _open_config_editor(self) -> None:
        """Every other setting is the Config Editor's: open it on this file."""
        if self._current_path is None or not self._current_path.is_file():
            QMessageBox.information(self, WINDOW_TITLE,
                                    "Save this config first, then open it in the Config Editor.")
            return
        from .config_editor import FLICConfigEditor

        self._config_editor = FLICConfigEditor(initial_path=self._current_path)
        self._config_editor.saved.connect(lambda _p: self.open_path(self._current_path))
        self._config_editor.show()


def launch() -> None:
    """``pyflic-experiment [folder | flic_config.yaml]``: with no argument, the
    current folder."""
    sanitize_input_method_environment()
    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app, mode=ui_settings.get("theme", "auto"))
    target = sys.argv[1] if len(sys.argv) > 1 else str(Path.cwd())
    window = ExperimentEditor(initial_path=target)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch()
