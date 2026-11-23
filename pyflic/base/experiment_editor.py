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
  Progressive Ratio, its paired chambers), and on a two-well layout the side
  well A is on, left or right, which every DFM states (``params:
  pi_direction``, ADR-0029).

Everything else the file holds — detection ``params:``, ``constants:``,
Slices, the Selection, optogenetics, a DFM's other overrides, ``scripts:``,
``data_folder:`` — belongs to the Config Editor and is written back exactly
as it was read.  Inside a Project with a Design the design is shown, not
edited, and saving writes the Design Copy, as the Config Editor does.

The ``global:`` it writes holds the design and no analysis setting: every
one is the Experiment Type's default until the Config Editor or a Project
Design says otherwise (ADR-0011).  It writes no ``params:`` there and drops an
empty one it reads, which said nothing (ADR-0030).

**Another program may write the file** (ADR-0026).  A catalogue that files
the recording rewrites ``about:`` to match its record, adding ``run:`` and,
in a level's mapping, ``term:``, ``label:`` and ``role:``.  So the editor:

* shows ``run:`` and never edits it, and edits only a level's
  ``definition:``, keeping every other key of the section as read;
* never saves silently over a file that changed on disk since it was read:
  it asks, and overwriting writes only what it edits over the file as it
  then is (:meth:`ExperimentEditor.write`);
* fills the form from an earlier recording (ADR-0027): its whole About but
  ``run:``, its design and its chamber map, asking first when that replaces
  what this recording already says (:meth:`ExperimentEditor.start_from`).

It also shows what ``Program.txt`` says the MCU ran, and which PyMCU wrote
it, beside the protocol, and copies them into the blank Protocol and Version
fields when asked (:meth:`ExperimentEditor.use_program_for_protocol`).
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import yaml
from PyQt6.QtCore import QEvent, QObject, Qt, pyqtSignal
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
    QSizePolicy,
    QStackedWidget,
    QTabBar,
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
    _same_path,
    _WestTabBar,
)
from .gui_env import sanitize_input_method_environment
from .ui import ActionButton, Card, Category, TopBar, apply_theme
from .ui import settings as ui_settings
from .ui.widgets import NoWheelComboBox

CONFIG_NAME = "flic_config.yaml"
WINDOW_TITLE = "FLIC Experiment Editor"
#: What the Organism box holds when a recording names none: nearly every study in
#: the lab (2026-11-22).  A value, which Save writes, not a hint; a person who
#: empties the box states no organism.
DEFAULT_ORGANISM = "Drosophila melanogaster"
#: The ``global:`` keys this editor writes; every other one is kept as read.
_DESIGN_KEYS = ("experiment_type", "chamber_layout", "well_names",
                "experimental_design_factors", "control_level")
#: The keys of a ``dfms:`` entry this editor has a field for.
_DFM_KEYS = ("id", "chambers", "paired_chambers", "chamber_groups")
#: What a person may answer when the file changed on disk since it was read.
RELOAD, OVERWRITE, CANCEL = "reload", "overwrite", "cancel"


def _text(widget) -> str:
    if isinstance(widget, QPlainTextEdit):
        return widget.toPlainText().strip()
    return widget.text().strip()


def _parse_config(data: bytes) -> dict | None:
    """*data* as a config, or ``None`` when it does not read as one."""
    try:
        raw = yaml.safe_load(data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError):
        return None
    if raw is None:
        return {}
    return raw if isinstance(raw, dict) else None


def _dfms_over(dfms: list[dict], on_disk: dict) -> list[dict]:
    """*dfms*, as the tabs give them, with every key this editor has no field
    for (a DFM's overrides, its optogenetics, a note) taken from *on_disk*'s
    entry for that DFM, when it has one."""
    theirs = {node["id"]: node for node in dfm_nodes(on_disk)}
    out: list[dict] = []
    for node in dfms:
        other = theirs.get(node.get("id"))
        if other is None:
            out.append(node)
            continue
        kept = {k: v for k, v in other.items() if k not in (*_DFM_KEYS, "Chambers")}
        out.append({"id": node["id"], **kept,
                    **{k: v for k, v in node.items() if k in _DFM_KEYS[1:]}})
    return out


def dfm_nodes(config: Any) -> list[dict]:
    """The ``dfms:`` entries of *config*, one mapping per DFM, each with its
    ``id``.  The section is a list of entries or a mapping keyed by DFM id, as
    the loader reads it; an entry with no integer id is left out."""
    section = config.get("dfms", config.get("DFMs")) if isinstance(config, dict) else None
    if isinstance(section, dict):
        ## Keyed by id: the key is the id unless the entry states its own.
        entries = [{"id": key, **(node if isinstance(node, dict) else {})}
                   for key, node in section.items()]
    else:
        entries = [node for node in (section or []) if isinstance(node, dict)]
    out: list[dict] = []
    for node in entries:
        try:
            dfm_id = int(node.get("id", node.get("ID")))
        except (TypeError, ValueError):
            continue
        out.append({**{k: v for k, v in node.items() if k != "ID"}, "id": dfm_id})
    return out


# ---------------------------------------------------------------------------
# Starting from another recording (ADR-0027)
# ---------------------------------------------------------------------------

#: The About fields of one line or a paragraph, as the replace question names them.
_ABOUT_LABELS = (("title", "Title"), ("purpose", "Purpose"),
                 ("experimenter", "Experimenter"), ("series", "Series"),
                 ("organism", "Organism"), ("age_at_start", "Age at start"),
                 ("notes", "Notes"))


def _states(value: Any) -> bool:
    """Whether *value* says anything: not nothing, a blank or an empty
    list or mapping."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return True


def _stated_side(section: Any) -> str | None:
    """The ``pi_direction`` that *section*, a DFM's entry or a ``global:``,
    states in its ``params:`` (legacy ``pi_multiplier`` included), or
    ``None``."""
    from .yaml_config import _normalize_param_overrides

    params = section.get("params", section.get("parameters")) \
        if isinstance(section, dict) else None
    if not isinstance(params, dict):
        return None
    try:
        return _normalize_param_overrides(params).get("pi_direction")
    except (TypeError, ValueError):
        return None                            # not a side the loader takes


def _folder_config(folder: Path) -> Path:
    """The config a folder's experiment reads: its own ``flic_config.yaml``,
    or, for a Project Member holding ``flic_member.yaml``, the recording's own
    config that it names (ADR-0031), so that Save writes the one config the
    recording has and never a second one beside the Member file."""
    from .layout import config_path

    return Path(config_path(folder))


def _states_design(design: dict) -> bool:
    """Whether *design* (a ``global:``) states any key of the design, so
    that taking it describes something rather than resetting the form."""
    if any(_states(design.get(key)) for key in _DESIGN_KEYS):
        return True
    params = design.get("params", design.get("parameters"))
    return isinstance(params, dict) and params.get("chamber_size") is not None


def _declared(design: dict | None) -> dict[str, list[str]]:
    """The factors *design* declares, as the factor table gives them."""
    factors = (design or {}).get("experimental_design_factors")
    if not isinstance(factors, dict):
        return {}
    return {str(name).strip(): about_mod.declared_levels(levels)
            for name, levels in factors.items() if str(name).strip()}


def _about_taken(mine: dict, earlier: dict, factors: dict | None) -> dict:
    """The About *mine* with every field *earlier* states: its shared About
    (``about.start_from``, limited to *factors* when given), and its title,
    purpose, notes and any key pyflic does not know.  Never its ``run:``:
    *mine*'s is kept exactly as read."""
    merged = about_mod.start_from(mine, earlier, factors)
    rest = {key: copy.deepcopy(value) for key, value in earlier.items()
            if key not in about_mod.SHARED_KEYS and key != about_mod.RUN_KEY
            and _states(value)}
    merged = about_mod.clean_about({**merged, **rest})
    if about_mod.RUN_KEY in mine:
        ## The catalogue's, as read: cleaning the section must not reword it.
        merged = {about_mod.RUN_KEY: mine[about_mod.RUN_KEY],
                  **{k: v for k, v in merged.items() if k != about_mod.RUN_KEY}}
    return merged


def _short(text: str, limit: int = 60) -> str:
    line = " ".join(str(text).split())
    return line if len(line) <= limit else line[:limit - 1].rstrip() + "…"


def _protocol_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, dict):
        return ""
    return ", ".join(about_mod._text(value.get(field)) for field in about_mod.PROTOCOL_FIELDS
                     if about_mod._text(value.get(field)))


def _factors_text(factors: dict[str, list[str]]) -> str:
    return "; ".join(f"{name} ({', '.join(levels)})" for name, levels in factors.items())


def _chamber_text(chambers: Any) -> dict[int, str]:
    """A ``chambers:`` value as ``{chamber: "level, level"}``, for comparing
    two maps however each was written."""
    if isinstance(chambers, list):
        chambers = {item.get("index"): item.get("treatment", item.get("levels", ""))
                    for item in chambers if isinstance(item, dict)}
    if not isinstance(chambers, dict):
        return {}
    out: dict[int, str] = {}
    for key, value in chambers.items():
        try:
            chamber = int(key)
        except (TypeError, ValueError):
            continue
        parts = value if isinstance(value, (list, tuple)) else str(value).split(",")
        text = ", ".join(str(part).strip() for part in parts if str(part).strip())
        if text:
            out[chamber] = text
    return out


# ---------------------------------------------------------------------------
# Factors: name, kind and levels
# ---------------------------------------------------------------------------

class _ColumnShares(QObject):
    """Keep *table*'s columns at fixed shares of its width.

    A table has no such resize mode of its own: sized to their contents, a
    factor's name and kind got a few letters each and the levels the rest.
    A person may still drag a column; the next resize shares them out again.
    """

    def __init__(self, table: QTableWidget, shares: tuple[float, ...]) -> None:
        super().__init__(table)
        self._table, self._shares = table, shares
        header = table.horizontalHeader()
        for column in range(len(shares)):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        table.viewport().installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt's name
        if event.type() == QEvent.Type.Resize:
            self.apply()
        return False

    def apply(self) -> None:
        width, total = self._table.viewport().width(), sum(self._shares)
        for column, share in enumerate(self._shares[:-1]):
            self._table.setColumnWidth(column, int(width * share / total))


class _DfmTabs(QWidget):
    """The Chambers page's DFMs: their list down the left, with the buttons
    that add and remove one under it, and the chosen DFM's page to its right.

    A ``QTabWidget`` cannot hold a button in a West tab bar's column, so this
    is that bar beside a stack of pages, answering the few calls the editor
    makes of a tab widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.bar = _WestTabBar()
        self.bar.setShape(QTabBar.Shape.RoundedWest)
        self.bar.setExpanding(False)
        ## As tall as its DFMs, so the buttons sit right under the last one.
        self.bar.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.stack = QStackedWidget()
        self.stack.setFrameShape(QFrame.Shape.StyledPanel)
        self.bar.currentChanged.connect(self.stack.setCurrentIndex)
        self.nav_column = QWidget()
        nav = QVBoxLayout(self.nav_column)
        nav.setContentsMargins(0, 0, 4, 0)
        nav.setSpacing(6)
        nav.addWidget(self.bar)
        self._buttons = QVBoxLayout()
        nav.addLayout(self._buttons)
        nav.addStretch(1)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self.nav_column)
        row.addWidget(self.stack, 1)

    def add_button(self, button: QWidget) -> None:
        self._buttons.addWidget(button)

    def insertTab(self, index: int, page: QWidget, text: str) -> int:  # noqa: N802
        self.stack.insertWidget(index, page)
        index = self.bar.insertTab(index, text)
        self.stack.setCurrentIndex(self.bar.currentIndex())
        return index

    def addTab(self, page: QWidget, text: str) -> int:  # noqa: N802
        return self.insertTab(self.count(), page, text)

    def removeTab(self, index: int) -> None:  # noqa: N802
        page = self.stack.widget(index)
        self.bar.removeTab(index)
        self.stack.removeWidget(page)
        page.deleteLater()
        self.stack.setCurrentIndex(self.bar.currentIndex())

    def count(self) -> int:
        return self.bar.count()

    def currentIndex(self) -> int:  # noqa: N802
        return self.bar.currentIndex()

    def setCurrentIndex(self, index: int) -> None:  # noqa: N802
        self.bar.setCurrentIndex(index)

    def tabText(self, index: int) -> str:  # noqa: N802
        return self.bar.tabText(index)

    def setTabText(self, index: int, text: str) -> None:  # noqa: N802
        self.bar.setTabText(index, text)


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
        self._shares = _ColumnShares(self.table, (3, 2, 4))
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(110)
        self.table.itemChanged.connect(self._on_item_changed)
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
        #: The kind the file gives each factor it describes, by name.
        self._loaded_kinds: dict[str, str] = {}

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() == 0:
            ## A factor named here that the file already describes takes the
            ## kind the file gives it, unless one is chosen.
            combo = self.table.cellWidget(item.row(), 1)
            kind = self._loaded_kinds.get(item.text().strip())
            if combo is not None and kind and not combo.currentData():
                self._set_kind(combo, kind)
        self.changed.emit()

    @staticmethod
    def _set_kind(combo, kind: str) -> None:
        if kind and combo.findData(kind) < 0:
            combo.addItem(kind, kind)          # kept as the file has it
        combo.setCurrentIndex(max(combo.findData(kind), 0))

    def _kind_combo(self, kind: str) -> NoWheelComboBox:
        combo = NoWheelComboBox()
        combo.addItem("", "")
        for value in about_mod.FACTOR_KINDS:
            combo.addItem(value, value)
        self._set_kind(combo, kind)
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
        """Show *factors*; *kinds* is the kind the file gives each factor it
        describes, declared here or not."""
        self._loaded_kinds = dict(kinds)
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

    def set_kinds(self, kinds: dict[str, str]) -> None:
        """Give every factor shown the kind *kinds* states for it, or none."""
        self._loaded_kinds = dict(kinds)
        for r in range(self.table.rowCount()):
            name = (self.table.item(r, 0).text() if self.table.item(r, 0) else "").strip()
            combo = self.table.cellWidget(r, 1)
            if combo is not None:
                self._set_kind(combo, kinds.get(name, ""))

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
    declared level, rebuilt when the factors change.  A row that stays keeps
    what is typed in it; a level declared here shows what the file already
    says about it."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(0, 3, parent)
        self.setHorizontalHeaderLabels(["Factor", "Level", "What it means"])
        self._shares = _ColumnShares(self, (3, 3, 4))
        self.verticalHeader().setVisible(False)
        self.setMinimumHeight(140)
        self.setToolTip("A few words per level: the cross, the dose, the diet.  "
                        "Optional; written under about.factors.")
        #: What the file says each level means, declared here or not.
        self._loaded: dict[str, dict[str, str]] = {}

    def refresh(self, factors: dict[str, list[str]],
                meanings: dict[str, dict[str, str]] | None = None) -> None:
        """Rebuild the rows for *factors*.  With *meanings* (a load) each row
        shows what they give its level.  Without, a row that was here keeps
        its text, a blank included, and a new one shows the loaded meaning."""
        here: dict[tuple[str, str], str] = {}
        if meanings is None:
            for r in range(self.rowCount()):
                factor, level, text = (self.item(r, c).text() if self.item(r, c) else ""
                                       for c in range(3))
                here[(factor, level)] = text
        else:
            self._loaded = {factor: dict(levels) for factor, levels in meanings.items()}
        self.setRowCount(0)
        for factor, levels in factors.items():
            for level in levels:
                r = self.rowCount()
                self.insertRow(r)
                for c, text in ((0, factor), (1, level)):
                    item = QTableWidgetItem(text)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.setItem(r, c, item)
                text = here.get((factor, level),
                                self._loaded.get(factor, {}).get(level, ""))
                self.setItem(r, 2, QTableWidgetItem(text))

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
        #: The file ``_loaded_raw`` was read from and its bytes as read
        #: (``None``: it did not exist), to notice a change on disk at save.
        self._read_path: Path | None = None
        self._read_bytes: bytes | None = None
        self._design: dict | None = None
        self._design_source: str | None = None
        self._dfm_widgets: list[DFMWidget] = []
        self._last_layout = "two_well"
        self._last_type: str | None = None
        #: Sex values the file holds that have no checkbox; written back.
        self._other_sex: list[str] = []
        #: The factors and levels the design declared when ``about:`` was
        #: read.  What ``about.factors`` says of any other was never shown
        #: here, so it is written back as read.
        self._declared_at_load: dict[str, set[str]] = {}
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
                ("Open &folder…", "Ctrl+Shift+O", self._open_folder),
                ("&Save", "Ctrl+S", self._save),
                ("Save &As…", "Ctrl+Shift+S", self._save_as),
                ("Start &from another recording…", None, self._start_from),
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
        ## about.run: the recording's id in the catalogue that filed it.  The
        ## catalogue writes it; it is shown here and written back as read.
        self.run_label = QLabel("")
        self.run_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.run_label.setToolTip(
            "about.run: written by the catalogue that filed this recording.  "
            "Shown here, never edited, and never copied to another recording.")
        self.run_label.setVisible(False)
        card.add_body(self.run_label)
        ## Replicates share most of what is on this tab (ADR-0026).
        self.start_from_button = ActionButton("Start from another recording…",
                                              Category.LOAD, icon_name="open")
        self.start_from_button.setToolTip(
            "Fill in who ran it, the series, the flies, the protocol and what each "
            "factor and level means from an earlier recording's flic_config.yaml.  "
            "The title, purpose and notes stay as they are.  A recording with no "
            "design yet takes the earlier one's design and chamber map too.  "
            "Nothing is saved until you save.")
        self.start_from_button.clicked.connect(lambda _checked=False: self._start_from())
        start_row = QHBoxLayout()
        start_row.addWidget(self.start_from_button)
        start_hint = QLabel("A replicate shares most of this with the recording before it.")
        start_hint.setObjectName("PyflicCardSubtitle")
        start_row.addWidget(start_hint)
        start_row.addStretch(1)
        card.add_body(start_row)
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
        self.organism_edit = QLineEdit(DEFAULT_ORGANISM)
        self.organism_edit.setPlaceholderText("Species, such as Drosophila melanogaster")
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
        ## What the MCU ran, read from Program.txt each time and never saved
        ## unless the person copies it into the fields above (ADR-0027).
        self.program_label = QLabel("")
        self.program_label.setObjectName("PyflicCardSubtitle")
        self.program_label.setWordWrap(True)
        self.program_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.use_program_button = ActionButton("Use for protocol", Category.LOAD,
                                               icon_name="new")
        self.use_program_button.setToolTip(
            "Copy what Program.txt says into Protocol and Version where they are "
            "blank.  Program.txt names no protocol: this is what the MCU ran, and "
            "which PyMCU wrote the file.  Edit it before you save.")
        self.use_program_button.clicked.connect(
            lambda _checked=False: self.use_program_for_protocol())
        program_row = QHBoxLayout()
        program_row.addWidget(self.program_label, 1)
        program_row.addWidget(self.use_program_button, 0, Qt.AlignmentFlag.AlignTop)
        form.addRow("From Program.txt:", program_row)
        self._program_form, self._program_row = form, form.rowCount() - 1
        form.setRowVisible(self._program_row, False)
        self._program_text: tuple[str, str | None] | None = None
        for field in ("name", "version"):
            self.protocol_edits[field].textChanged.connect(
                lambda _text: self._refresh_use_program())
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
        self.add_found_button = ActionButton("Add the DFMs found in the data", Category.LOAD)
        self.add_found_button.clicked.connect(lambda _checked=False: self._add_found_dfms())
        top = QHBoxLayout()
        top.addWidget(self.data_label, 1)
        top.addWidget(self.add_found_button)
        layout.addLayout(top)
        ## A DFM's number is what its data files say: the list names each one,
        ## and Add DFM adds the next, so no tab has a box to renumber it.
        self.dfm_tabs = _DfmTabs()
        self.add_dfm_button = ActionButton("Add DFM", Category.LOAD, icon_name="new")
        self.add_dfm_button.setToolTip(
            "Add the next DFM: the first the data holds that has no tab yet, "
            "else the one after the highest listed.")
        self.add_dfm_button.clicked.connect(lambda _checked=False: self._add_dfm())
        self.remove_dfm_button = ActionButton("Remove DFM", Category.QC, icon_name="clear")
        self.remove_dfm_button.setToolTip("Remove the DFM shown, and its chamber assignments.")
        self.remove_dfm_button.clicked.connect(lambda _checked=False: self._remove_dfm())
        for button in (self.add_dfm_button, self.remove_dfm_button):
            self.dfm_tabs.add_button(button)
        layout.addWidget(self.dfm_tabs, 1)
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

    def _append_dfm(self, dfm_id: int, index: int | None = None) -> DFMWidget:
        """A tab for DFM *dfm_id*, at *index* in the list (last when not
        given).  Its number is fixed: the tab has no box to change it."""
        index = len(self._dfm_widgets) if index is None else index
        widget = DFMWidget(dfm_id=dfm_id, chamber_size=_layout_chamber_size(self.chamber_layout()),
                           simple=True)
        widget.set_side_default(self._experiment_side())
        widget.update_factors(self.factor_table.factors())
        widget.set_progressive_ratio(self.current_type().has_module("ratio"))
        widget._id_spin.valueChanged.connect(
            lambda value, w=widget: self._on_dfm_id_changed(w, value))
        self._dfm_widgets.insert(index, widget)
        self.dfm_tabs.insertTab(index, self._scrolled(widget), f"DFM {dfm_id}")
        return widget

    def _on_dfm_id_changed(self, widget: DFMWidget, value: int) -> None:
        if widget in self._dfm_widgets:
            self.dfm_tabs.setTabText(self._dfm_widgets.index(widget), f"DFM {value}")
        self._refresh_data_label()

    def _clear_dfms(self) -> None:
        while self._dfm_widgets:
            self.dfm_tabs.removeTab(self.dfm_tabs.count() - 1)
            self._dfm_widgets.pop().deleteLater()

    def _next_dfm(self) -> int | None:
        """The DFM *Add DFM* adds: the first the data holds that has no tab
        yet, else the one after the highest listed (the lowest free number
        past 99, a DFM's highest).  ``None`` when every number is taken."""
        used = {w._id_spin.value() for w in self._dfm_widgets}
        missing = [i for i in self._found_dfms() if i not in used]
        if missing:
            return missing[0]
        after = max(used, default=0) + 1
        if after <= 99:
            return after
        return next((k for k in range(1, 100) if k not in used), None)

    def _add_dfm(self, dfm_id: int | None = None) -> DFMWidget | None:
        """Add DFM *dfm_id*, or the next one (:meth:`_next_dfm`), in number
        order in the list, and show it."""
        dfm_id = self._next_dfm() if dfm_id is None else dfm_id
        if dfm_id is None:
            return None
        index = next((i for i, w in enumerate(self._dfm_widgets)
                      if w._id_spin.value() > dfm_id), len(self._dfm_widgets))
        widget = self._append_dfm(dfm_id, index)
        self.dfm_tabs.setCurrentIndex(index)
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

    def _program_path(self) -> Path | None:
        """The recording's ``Program.txt``: in the folder its config reads
        (``data/``, or ``data_folder:``), else, with no ``data_folder:``,
        beside the config, where an instrument folder keeps it -- the same two
        places :meth:`_found_dfms` looks for DFM files."""
        directory = self._where()
        if directory is None or not directory.is_dir():
            return None
        from . import layout as layout_mod
        from .opto_program import find_program

        location = layout_mod.data_location(str(directory), config=self._loaded_raw)
        found = None if location.problem else find_program(location.path)
        if found is None and not self._loaded_raw.get(layout_mod.DATA_FOLDER_KEY):
            found = find_program(directory)
        return found

    def _refresh_started(self) -> None:
        """The start, and what the MCU ran, from ``Program.txt``: shown, never
        saved (ADR-0024, ADR-0027)."""
        text = "Program.txt not found: the start is read from it when it is there."
        self._program_text = None
        try:
            from .opto_program import program_summary, pymcu_version, read_program

            path = self._program_path()
            if path is not None:
                program = read_program(path)
                if program.start is not None:
                    text = f"Started {program.start:%Y-%m-%d %H:%M}, from Program.txt"
                self._program_text = (program_summary(program), pymcu_version(program))
        except Exception:  # noqa: BLE001 - what is shown here is a convenience
            pass
        self.started_label.setText(text)
        if self._program_text is not None:
            summary, version = self._program_text
            self.program_label.setText(summary + (f"  ·  written by {version}" if version else ""))
        self._program_form.setRowVisible(self._program_row, self._program_text is not None)
        self._refresh_use_program()

    def _refresh_use_program(self) -> None:
        self.use_program_button.setEnabled(bool(self._program_fills()))

    def _program_fills(self) -> dict[str, str]:
        """What *Use for protocol* would write: each of Protocol and Version
        that is blank and that ``Program.txt`` has something for."""
        if self._program_text is None:
            return {}
        summary, version = self._program_text
        return {field: value for field, value in (("name", summary), ("version", version))
                if value and not self.protocol_edits[field].text().strip()}

    def use_program_for_protocol(self) -> list[str]:
        """Copy what ``Program.txt`` says into the blank Protocol and Version
        fields; a field with text in it is left alone.  Returns the fields
        filled.  Nothing is saved until the person saves."""
        fills = self._program_fills()
        for field, value in fills.items():
            self.protocol_edits[field].setText(value)
        self.statusBar().showMessage(
            "Filled " + " and ".join("Protocol" if f == "name" else "Version" for f in fills)
            + " from Program.txt.  Edit them as you like before you save."
            if fills else "Protocol and Version already say something; nothing was changed.")
        return list(fills)

    # ---- loading -------------------------------------------------------

    def open_path(self, path: str | Path) -> None:
        """Open a ``flic_config.yaml``, or a folder: its config if it has one,
        and otherwise a new config for it, one DFM tab per DFM in its data."""
        p = Path(path).expanduser()
        config = p if p.is_file() else _folder_config(p)
        if p.is_dir() and not config.exists() and (p / "flic_config.yml").exists():
            config = p / "flic_config.yml"
        if config.is_file():
            try:
                data = config.read_bytes()
                raw = yaml.safe_load(data.decode("utf-8")) or {}
            except Exception as err:  # noqa: BLE001
                QMessageBox.critical(self, WINDOW_TITLE, f"Could not read {config}:\n{err}")
                return
            if not isinstance(raw, dict):
                QMessageBox.critical(self, WINDOW_TITLE, f"{config} is not a YAML mapping.")
                return
            self._current_path = config
            self._read_path, self._read_bytes = config, data
            self.load_config(raw)
            self.statusBar().showMessage(f"Opened {config}")
        else:
            self._current_path = config
            self._read_path, self._read_bytes = config, None
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
        self._note_declared()
        self._clear_dfms()
        chamber_size = _layout_chamber_size(self.chamber_layout())
        for node in dfm_nodes(raw):
            widget = self._append_dfm(node["id"])
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

    def _note_declared(self) -> None:
        """Remember what the design declares now: the rows ``about.factors``
        has in this form."""
        self._declared_at_load = {name: set(levels)
                                  for name, levels in self.factor_table.factors().items()}

    def _show_run(self, about: dict) -> None:
        run = about_mod._text(about.get(about_mod.RUN_KEY))
        self.run_label.setText(f"Filed as {run}" if run else "")
        self.run_label.setVisible(bool(run))

    def _load_about(self, about: dict) -> None:
        self._show_run(about)
        self.title_edit.setText(about_mod._text(about.get("title")))
        self.purpose_edit.setPlainText(about_mod._text(about.get("purpose")))
        self.experimenter_edit.setText(about_mod._text(about.get("experimenter")))
        self.series_edit.setText(about_mod._text(about.get("series")))
        self.organism_edit.setText(about_mod._text(about.get("organism")) or DEFAULT_ORGANISM)
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

    def _type_and_layout(self, g: dict):
        """The Experiment Type and the Chamber Layout the design *g* means
        for this recording.  A type unknown here is a Custom Experiment."""
        from . import experiment_types

        raw_type = g.get("experiment_type")
        try:
            item = experiment_types.get_experiment_type(raw_type, self._where())
            legacy = None
        except ValueError:
            item = experiment_types.get_experiment_type(None)
            key = str(raw_type or "").lower()
            legacy = "single_well" if "single" in key else "two_well" if "two" in key else None
        if item.chamber_layout is not None:
            return item, item.chamber_layout
        params = g.get("params", g.get("parameters")) or {}
        size = params.get("chamber_size") if isinstance(params, dict) else None
        return item, (legacy or _normalise_layout(g.get("chamber_layout"))
                      or ("single_well" if str(size) == "1" else "two_well"))

    def _load_design(self, g: dict, about: dict) -> None:
        item, layout = self._type_and_layout(g)
        self._fill_type_combo(item.name)
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

    # ---- starting from another recording ---------------------------------

    def _start_from(self) -> None:
        """Ask for another recording's config and fill the form from it."""
        here = self._where()
        start = str(here.parent if here is not None else Path.cwd())
        path, _ = QFileDialog.getOpenFileName(
            self, "Start from another recording's flic_config.yaml", start,
            "FLIC config (*.yaml *.yml)")
        if path:
            self.start_from(path)

    def start_from(self, source: str | Path) -> bool:
        """Fill the form from another recording: *source* is its
        ``flic_config.yaml``, or the folder that holds one (ADR-0027, which
        amends ADR-0026).

        Replicates share nearly everything, so the form takes everything it
        shows that this recording can take:

        * the other ``about:``: every field it states, the title, the purpose
          and the notes included, and each factor's kind and each level's
          whole entry.  Never ``run:``, the catalogue's id of that recording.
          A field it does not state stays as it is here;
        * its design (type, layout, well names, factors, control level),
          unless a Project design governs this recording, it states none, or
          its type is one this recording cannot use;
        * its chamber map, for the DFMs this recording's data holds (its tabs,
          when no data is found): each chamber's levels and, for Progressive
          Ratio, the chamber groups.  Taken with its design, or without it
          when its factors are the ones this recording declares.  When the
          factors change, a DFM it has no map for is cleared here;
        * each DFM's PI Direction, the side well A is on, where it states one
          for a DFM this recording's data holds and the layout has two wells
          (ADR-0028).  It says how the DFM was set up, not what the design is,
          so it is taken with or without the map.  A DFM's other overrides,
          its optogenetics and notes are that recording's own.

        Whatever that replaces of what this recording already says is listed
        in one question first, and No changes nothing.  Nothing is written:
        the form is filled, and the person saves.  False, with the reason
        said, when *source* is not another recording's config or the person
        declines.
        """
        found = self._other_config(source)
        if found is None:
            return False
        config, other = found
        their_design = self._design_of(other, config.parent)
        theirs = None
        if not self._design and _states_design(their_design) \
                and self._type_usable(their_design, config):
            theirs = their_design
        factors_now = self.factor_table.factors()
        factors = _declared(theirs) if theirs is not None else factors_now
        takes_map = theirs is not None or list(_declared(their_design)) == list(factors_now)
        load, clear = self._map_plan(other, clear_rest=theirs is not None
                                     and factors != factors_now) if takes_map else ({}, [])
        sides = self._side_plan(other)
        mine = self.about_section()
        earlier = about_mod.read_about(other)
        merged = _about_taken(mine, earlier, factors or None)

        changes = self._about_changes(mine, merged)
        if theirs is not None:
            changes += self._design_changes(theirs)
        changes += self._map_changes(load, clear)
        changes += self._side_changes(sides)
        if changes and not self._confirm_replace(config, changes):
            self.statusBar().showMessage(f"Nothing was taken from {config}.")
            return False

        took: list[str] = []
        if theirs is not None:
            self._show_design(theirs)
            took.append("the design")
        mapped = self._take_map(load, clear)
        if mapped:
            took.append(f"the chamber map of DFM {', '.join(map(str, mapped))}")
        sided = self._take_sides(sides)
        if sided:
            took.append(f"the PI Direction of DFM {', '.join(map(str, sided))}")
        raw = {k: v for k, v in self._loaded_raw.items() if k != about_mod.ABOUT_KEY}
        self._loaded_raw = {**raw, about_mod.ABOUT_KEY: merged} if merged else raw
        self._load_about(merged)
        notes = about_mod.factor_notes(merged)
        self.factor_table.set_kinds({k: v["kind"] for k, v in notes.items()})
        self.meaning_table.refresh(self.factor_table.factors(),
                                   {k: v["levels"] for k, v in notes.items()})
        self._note_declared()
        self._refresh_data_label()
        stated = [key for key, value in earlier.items()
                  if key != about_mod.RUN_KEY and _states(value)]
        if stated:
            took.insert(0, f"the About ({', '.join(map(str, stated))})")
        self.statusBar().showMessage(
            f"Filled in from {config}: {'; '.join(took)}.  Nothing is saved until you save."
            if took else f"{config} had nothing to start from.")
        return True

    def _confirm_replace(self, config: Path, changes: list[str]) -> bool:
        answer = QMessageBox.question(
            self, "Replace what this recording says?",
            f"Starting from {config} replaces what this recording already says:\n\n"
            + "\n".join(f"  • {line}" for line in changes)
            + "\n\nReplace it?  Nothing is saved until you save.\n\n"
              "No leaves this recording as it is.")
        return answer == QMessageBox.StandardButton.Yes

    def _about_changes(self, mine: dict, merged: dict) -> list[str]:
        """What taking *merged* would replace of the About *mine*, one line
        each: a field this recording states that the other states otherwise."""
        lines: list[str] = []
        for key, label in _ABOUT_LABELS:
            here, there = about_mod._text(mine.get(key)), about_mod._text(merged.get(key))
            if here and here != there:
                lines.append(f"{label}: {_short(here)} → {_short(there)}")
        here = about_mod.sex_values(mine.get(about_mod.SEX_KEY))
        there = about_mod.sex_values(merged.get(about_mod.SEX_KEY))
        if here and here != there:
            lines.append(f"Sex: {', '.join(here)} → {', '.join(there) or 'none'}")
        here = _protocol_text(mine.get(about_mod.PROTOCOL_KEY))
        there = _protocol_text(merged.get(about_mod.PROTOCOL_KEY))
        if here and here != there:
            lines.append(f"Protocol: {_short(here)} → {_short(there) or 'none'}")
        before, after = about_mod.factor_notes(mine), about_mod.factor_notes(merged)
        differ = 0
        for factor, note in before.items():
            now = after.get(factor, {"kind": "", "levels": {}})
            differ += bool(note["kind"]) and note["kind"] != now["kind"]
            differ += sum(1 for level, text in note["levels"].items()
                          if text and text != now["levels"].get(level, ""))
        if differ:
            lines.append(f"Factor kinds and level meanings: {differ} "
                         f"{'entry' if differ == 1 else 'entries'}")
        return lines

    def _other_config(self, source: str | Path) -> tuple[Path, dict] | None:
        """Another recording's config and where it is, or ``None`` (and the
        reason said) when *source* is not one."""
        given = Path(source).expanduser()
        config = _folder_config(given) if given.is_dir() else given
        if given.is_dir() and not config.exists() and (given / "flic_config.yml").exists():
            config = given / "flic_config.yml"
        if not config.is_file():
            QMessageBox.warning(self, WINDOW_TITLE, f"No {CONFIG_NAME} found at {given}.")
            return None
        if self._current_path is not None and _same_path(config, self._current_path):
            QMessageBox.information(
                self, WINDOW_TITLE,
                "That is this recording's own config.  Choose an earlier recording's.")
            return None
        data = self._bytes_on_disk(config)
        other = _parse_config(data) if data is not None else None
        if other is None:
            QMessageBox.warning(self, WINDOW_TITLE, f"{config} does not read as a config.")
            return None
        return config, other

    @staticmethod
    def _design_of(config: dict, directory: Path) -> dict:
        """The design the recording in *directory* was run under: its
        Project's, for a Member, else its own ``global:``."""
        from . import project as project_mod

        try:
            design, _source = project_mod.design_for_member(directory)
        except Exception:  # noqa: BLE001 - a bad project.yaml governs nothing
            design = None
        own = config.get("global")
        return dict(design) if design else dict(own) if isinstance(own, dict) else {}

    def _type_usable(self, theirs: dict, config: Path) -> bool:
        """Whether the design *theirs* names a type this recording can use.
        When it does not, it is said, and the design is not taken."""
        from . import experiment_types
        from .experiment_types import library, registry

        name = theirs.get("experiment_type")
        retired = {library.name_key(r) for r in registry.RETIRED_AS_LAYOUT}
        if name not in (None, "") and library.name_key(name) not in retired:
            try:
                experiment_types.get_experiment_type(name, self._where())
            except ValueError:
                QMessageBox.warning(
                    self, WINDOW_TITLE,
                    f"{config} names the Experiment Type {name!r}, which this recording "
                    "cannot use: it is neither in the lab's library nor beside this "
                    "config.  Its design was not taken; its About was.")
                return False
        return True

    def _design_changes(self, theirs: dict) -> list[str]:
        """What showing the design *theirs* would replace of a design already
        stated here, one line each: a type other than Custom, a single-well
        layout, a well name, a control level or the factors.  Empty when it
        only fills in."""
        other, layout = self._type_and_layout(theirs)
        mine = self.current_type()
        changes: list[str] = []
        if not mine.is_custom and other.name != mine.name:
            changes.append(f"Experiment Type: {mine.display_name} → {other.display_name}")
        elif mine.layout_free and self.chamber_layout() == "single_well" \
                and layout != "single_well":
            changes.append(f"Chamber Layout: {_LAYOUT_LABELS['single_well']} → "
                           f"{_LAYOUT_LABELS.get(layout, layout)}")
        wells = theirs.get("well_names") if isinstance(theirs.get("well_names"), dict) else {}
        for label, edit, value in (
                ("Well A", self.well_a_edit, wells.get("A")),
                ("Well B", self.well_b_edit, wells.get("B")),
                ("Control Level", self.control_edit, theirs.get("control_level"))):
            here, there = _text(edit), about_mod._text(value)
            if here and here != there:
                changes.append(f"{label}: {here} → {there or 'none'}")
        here_factors, there_factors = self.factor_table.factors(), _declared(theirs)
        if here_factors and here_factors != there_factors:
            changes.append(f"Design factors: {_factors_text(here_factors)} → "
                           f"{_factors_text(there_factors) or 'none'}")
        return changes

    def _map_plan(self, other: dict, *, clear_rest: bool) -> tuple[dict[int, dict], list[int]]:
        """What taking the config *other*'s chamber map would do: for each
        DFM this recording's data holds (its tabs, when no data is found) that
        *other* maps, the map it would show; and, with *clear_rest*, the DFMs
        here with chambers assigned that *other* has no map for, whose levels
        the new factors would not know.  A DFM's overrides, optogenetics and
        notes are that recording's own, and are never part of it; its PI
        Direction is taken apart from the map (:meth:`_side_plan`)."""
        theirs_by_id = {node["id"]: node for node in dfm_nodes(other)}
        tabs = {w._id_spin.value(): w for w in self._dfm_widgets}
        load: dict[int, dict] = {}
        for dfm_id in (self._found_dfms() or sorted(tabs)):
            node = theirs_by_id.get(dfm_id)
            if node is None:
                continue                       # this recording's alone
            shown = {k: node[k] for k in _DFM_KEYS[1:] if k in node}
            shown.setdefault("chambers", node.get("Chambers") or {})
            load[dfm_id] = shown
        clear = [dfm_id for dfm_id, widget in tabs.items()
                 if clear_rest and dfm_id not in load and widget.assignments_beyond(0)]
        return load, clear

    def _map_changes(self, load: dict[int, dict], clear: list[int]) -> list[str]:
        tabs = {w._id_spin.value(): w for w in self._dfm_widgets}
        replaced = [dfm_id for dfm_id, shown in load.items()
                    if dfm_id in tabs and tabs[dfm_id].assignments_beyond(0)
                    and _chamber_text(tabs[dfm_id].get_dict().get("chambers"))
                    != _chamber_text(shown.get("chambers"))]
        lines: list[str] = []
        if replaced:
            lines.append(f"Chambers of DFM {', '.join(map(str, replaced))}: "
                         "replaced by that recording's")
        if clear:
            lines.append(f"Chambers of DFM {', '.join(map(str, clear))}: cleared, since "
                         "that recording maps none and the factors change")
        return lines

    def _side_plan(self, other: dict) -> dict[int, str]:
        """Each DFM's PI Direction to take from the config *other*
        (ADR-0028): for each DFM this recording's data holds (its tabs, when
        no data is found), the side well A is on, where *other* states one for
        that DFM, legacy ``pi_multiplier`` included, or in its ``global:``
        for a DFM it lists with none of its own (ADR-0029), as the loader
        reads it.  A DFM it states none for keeps its own here, as an About
        field it leaves blank does."""
        theirs_by_id = {node["id"]: node for node in dfm_nodes(other)}
        their_side = _stated_side(other.get("global"))
        tabs = {w._id_spin.value(): w for w in self._dfm_widgets}
        sides: dict[int, str] = {}
        for dfm_id in (self._found_dfms() or sorted(tabs)):
            if dfm_id not in theirs_by_id:
                continue
            side = _stated_side(theirs_by_id[dfm_id]) or their_side
            if side is not None:
                sides[dfm_id] = side
        return sides

    def _experiment_side(self) -> str:
        """The side a DFM that states none is read with: the Design's
        ``params:`` inside a Project with one, else this file's ``global:``
        ``params:``, else ``left``, the loader's default.  Each two-well DFM
        that states none is given it as its own (ADR-0029)."""
        return _stated_side(self._design or self._loaded_raw.get("global")) or "left"

    def _side_changes(self, sides: dict[int, str]) -> list[str]:
        """The sides stated here that *sides* would replace.  A side filled in
        with the experiment's is not one stated here (ADR-0029)."""
        tabs = {w._id_spin.value(): w for w in self._dfm_widgets}
        replaced = []
        for dfm_id, side in sides.items():
            widget = tabs.get(dfm_id)
            here = widget.pi_direction() if widget and widget.side_stated() else None
            if here is not None and here != side:
                replaced.append(f"DFM {dfm_id} {here} → {side}")
        return [f"PI Direction: {', '.join(replaced)}"] if replaced else []

    def _take_sides(self, sides: dict[int, str]) -> list[int]:
        """Give each listed DFM its side from *sides*; none on a one-well
        layout, whose chambers have no side.  Returns the DFMs given one."""
        if _layout_chamber_size(self.chamber_layout()) != 2:
            return []
        tabs = {w._id_spin.value(): w for w in self._dfm_widgets}
        taken = [dfm_id for dfm_id in sides if dfm_id in tabs]
        for dfm_id in taken:
            tabs[dfm_id].set_pi_direction(sides[dfm_id])
        return taken

    def _show_design(self, theirs: dict) -> None:
        """Show the design *theirs*, and fit every DFM tab to it."""
        self._load_design(theirs, {})
        size = _layout_chamber_size(self.chamber_layout())
        factors = self.factor_table.factors()
        for widget in self._dfm_widgets:
            widget.update_chamber_size(size)
            widget.update_factors(factors)

    def _take_map(self, load: dict[int, dict], clear: list[int]) -> list[int]:
        """Show the chamber maps *load* (adding a tab where there is none) and
        clear the chambers of *clear*, keeping each tab's other keys.
        Returns the DFMs whose map was taken."""
        size = _layout_chamber_size(self.chamber_layout())
        tabs = {w._id_spin.value(): w for w in self._dfm_widgets}
        for dfm_id, shown in load.items():
            widget = tabs.get(dfm_id) or self._append_dfm(dfm_id)
            kept = {k: v for k, v in widget.get_dict().items() if k not in _DFM_KEYS[1:]}
            widget.load_dict({**kept, **shown}, size)
        for dfm_id in clear:
            widget = tabs[dfm_id]
            widget.load_dict({k: v for k, v in widget.get_dict().items()
                              if k not in _DFM_KEYS[1:]}, size)
        return list(load)

    # ---- what is written -----------------------------------------------

    def about_section(self) -> dict:
        """``about:`` as the form describes it.  What the form has no field
        for is kept as read: ``run:``, a key pyflic does not know, the keys
        of a level's mapping beside its ``definition:``, and what the file
        says of a factor or level this form never showed (ADR-0026)."""
        loaded = about_mod.read_about(self._loaded_raw)
        known = set(about_mod.KEY_ORDER)
        section: dict[str, Any] = {k: v for k, v in loaded.items() if k not in known}
        for key, widget in (("title", self.title_edit), ("purpose", self.purpose_edit),
                            ("experimenter", self.experimenter_edit),
                            ("series", self.series_edit), ("organism", self.organism_edit),
                            ("age_at_start", self.age_edit), ("notes", self.notes_edit)):
            section[key] = _text(widget)
        ticked = [v for v, c in self.sex_checks.items() if c.isChecked()] + self._other_sex
        was = about_mod.sex_values(loaded.get(about_mod.SEX_KEY))
        ## Unchanged, it is written as the file had it: its order and spelling.
        same = sorted(v.lower() for v in was) == sorted(v.lower() for v in ticked)
        section[about_mod.SEX_KEY] = was if same else ticked
        loaded_protocol = loaded.get(about_mod.PROTOCOL_KEY)
        protocol = {k: v for k, v in loaded_protocol.items()
                    if k not in about_mod.PROTOCOL_FIELDS} \
            if isinstance(loaded_protocol, dict) else {}
        protocol.update({field: _text(edit) for field, edit in self.protocol_edits.items()})
        section[about_mod.PROTOCOL_KEY] = protocol
        section[about_mod.FACTORS_KEY] = self._factors_section(
            loaded.get(about_mod.FACTORS_KEY))
        cleaned = about_mod.clean_about(section)
        run = loaded.get(about_mod.RUN_KEY)
        if run not in (None, ""):
            ## The catalogue's, never the form's: as read, and first.
            cleaned = {about_mod.RUN_KEY: run, **cleaned}
        return cleaned

    def _factors_section(self, loaded: Any) -> dict:
        """``about.factors`` from the two tables, over what the file held.

        A level's entry keeps its shape: text stays text, and a mapping keeps
        its other keys with its ``definition:`` set to the row's text
        (``about.with_meaning``).  A factor's own keys beside ``kind:`` and
        ``levels:`` are kept.  A factor or level the design did not declare
        when the file was read has no row here, so its entry is written back
        as read; one this editor removed or renamed is dropped with its row.
        Entries stay in the file's order, with new ones after them.
        """
        loaded = {str(name).strip(): body for name, body in loaded.items()} \
            if isinstance(loaded, dict) else {}
        declared = self.factor_table.factors()
        kinds = self.factor_table.kinds()
        meanings = self.meaning_table.meanings()
        out: dict[str, Any] = {}
        for name in [*loaded, *(n for n in declared if n not in loaded)]:
            body = loaded.get(name)
            if name not in declared:
                if name not in self._declared_at_load:
                    out[name] = copy.deepcopy(body)        # never shown: as read
                continue
            body = body if isinstance(body, dict) else {}
            entries = body.get("levels")
            entries = {str(level).strip(): entry for level, entry in entries.items()} \
                if isinstance(entries, dict) else {}
            shown_before = self._declared_at_load.get(name, set())
            levels: dict[str, Any] = {}
            for level in [*entries, *(v for v in declared[name] if v not in entries)]:
                if level in declared[name]:
                    value = about_mod.with_meaning(entries.get(level),
                                                   meanings.get(name, {}).get(level, ""))
                elif level not in shown_before:
                    value = copy.deepcopy(entries[level])  # never shown: as read
                else:
                    continue                               # removed or renamed here
                if value is not None:
                    levels[level] = value
            ## Blanks are left out when the section is cleaned.
            out[name] = {"kind": kinds.get(name, ""), "levels": levels,
                         **{k: copy.deepcopy(v) for k, v in body.items()
                            if k not in ("kind", "levels")}}
        return out

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
        for key in ("params", "parameters"):
            if key in rest and not rest[key]:
                ## Empty: every default, which no section says as well
                ## (ADR-0030).  A setting the Config Editor wrote stays.
                del rest[key]
        return {**self.design_section(), **rest}

    def collect(self) -> dict:
        """The whole config to write: the file as read, with ``about:``, the
        design and ``dfms:`` replaced by the form's."""
        from .layout import DATA_FOLDER_KEY

        raw = self._loaded_raw
        dfms = [w.get_dict() for w in self._dfm_widgets]
        about = self.about_section()
        unread = raw.get(about_mod.ABOUT_KEY)
        if unread is not None and not isinstance(unread, dict) \
                and about == {"organism": DEFAULT_ORGANISM}:
            ## The organism's default is not typing: a section the form could not
            ## read stays as written until the person types something.
            about = {}
        g = self.global_section(dfms)
        out: dict[str, Any] = {}
        if DATA_FOLDER_KEY in raw:
            out[DATA_FOLDER_KEY] = raw[DATA_FOLDER_KEY]
        if about:
            out[about_mod.ABOUT_KEY] = about
        elif raw.get(about_mod.ABOUT_KEY) is not None and not isinstance(
                raw[about_mod.ABOUT_KEY], dict):
            ## Not a section the form could read, and nothing typed to put in
            ## its place: left as written, for the linter to name.
            out[about_mod.ABOUT_KEY] = raw[about_mod.ABOUT_KEY]
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

    def _open_folder(self) -> None:
        """Pick a recording's folder -- an MCU's output as it wrote it, or one
        with its data in ``data/`` -- and open it: its ``flic_config.yaml``
        when it has one, else a new config for it with one DFM tab per DFM
        its data holds (:meth:`open_path`).  The dialog starts beside the
        recording open now, where the next one usually is."""
        here = self._where()
        start = str(here.parent if here is not None else Path.cwd())
        folder = QFileDialog.getExistingDirectory(self, "Open a recording's folder", start)
        if folder:
            self.open_path(folder)

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

    def write(self, path: Path, *, confirm: bool = True,
              if_changed: str | None = None) -> bool:
        """Write the config to *path*.

        A file that changed on disk since this editor read it is never
        overwritten silently (ADR-0026).  The person is asked first, unless
        *if_changed* answers for them: :data:`RELOAD` shows the file as it
        now is and writes nothing; :data:`OVERWRITE` writes what this editor
        edits (the About fields, the design and the chambers) over the file
        as it now is, so everything else it holds, ``run:`` and a level's
        ``term:`` included, stays as the other program left it;
        :data:`CANCEL` does neither.

        With *confirm*, problems are shown first and the save may be declined.
        """
        path = Path(path)
        on_disk: dict | None = None
        if self._read_path is not None and _same_path(path, self._read_path):
            now = self._bytes_on_disk(path)
            if now is not None and now != self._read_bytes:
                on_disk = _parse_config(now)
                answer = if_changed or self._ask_changed_on_disk(path, on_disk is not None)
                if answer == RELOAD and on_disk is not None:
                    self.open_path(path)
                    self.statusBar().showMessage(
                        f"Read {path} again as it is on disk.  Nothing was saved.")
                    return False
                if answer != OVERWRITE:
                    return False
        problems = self.problems()
        if problems and confirm:
            answer = QMessageBox.question(
                self, "Save with problems?",
                "This config has problems:\n\n"
                + "\n".join(f"  • {p}" for p in problems[:15])
                + ("\n  • …" if len(problems) > 15 else "") + "\n\nSave anyway?")
            if answer != QMessageBox.StandardButton.Yes:
                return False
        ## run: names the recording in the folder this config was read from.
        elsewhere = self._read_path is not None and not _same_path(
            path.parent, self._read_path.parent)
        left_out = None
        before = (self._loaded_raw, self._design, self._design_source)
        try:
            if on_disk is not None:
                ## What this editor does not edit is the file's, as it is now.
                self._loaded_raw = on_disk
                self._read_design()
            cfg = self.collect()
            if on_disk is not None:
                cfg["dfms"] = _dfms_over(cfg["dfms"], on_disk)
            if elsewhere:
                cfg, left_out = self._for_another_folder(cfg, path)
            path.write_text(
                yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
            self._write_type_copy(path)
        except Exception as err:  # noqa: BLE001
            self._loaded_raw, self._design, self._design_source = before
            QMessageBox.critical(self, WINDOW_TITLE, f"Could not save {path}:\n{err}")
            return False
        if on_disk is not None:
            ## Show the file just written: the tabs still held the DFMs'
            ## hidden keys as they were first read.
            self.open_path(path)
        else:
            self._set_base(cfg)
            self._read_path, self._read_bytes = path, self._bytes_on_disk(path)
        self.setWindowTitle(f"{WINDOW_TITLE} — {path}")
        self.statusBar().showMessage(
            f"Saved {path}" + (f".  run: {left_out} was not written there: it names the "
                               f"recording this config was read from." if left_out else ""))
        self.saved.emit(str(path))
        return True

    @staticmethod
    def _for_another_folder(cfg: dict, path: Path) -> tuple[dict, Any]:
        """*cfg* as it is written into another recording's folder: without
        this recording's ``run:``, and with the ``run:`` a file written over
        there already holds, which is the id of the recording in that folder.
        Returns the config, and the ``run:`` left out, if any."""
        from . import project as project_mod

        left_out = about_mod.read_about(cfg).get(about_mod.RUN_KEY)
        cfg = project_mod.without_run(cfg)
        _found, theirs = project_mod.about_on_disk(path)
        run = theirs.get(about_mod.RUN_KEY) if isinstance(theirs, dict) else None
        if run not in (None, ""):
            cfg = project_mod.with_about(
                cfg, {about_mod.RUN_KEY: run, **about_mod.read_about(cfg)})
        return cfg, left_out

    @staticmethod
    def _bytes_on_disk(path: Path) -> bytes | None:
        try:
            return Path(path).read_bytes()
        except OSError:
            return None

    def _ask_changed_on_disk(self, path: Path, readable: bool) -> str:
        """Say that *path* changed on disk since it was read, and ask what
        to do: :data:`RELOAD`, :data:`OVERWRITE` or :data:`CANCEL`.
        *readable* is whether the file still reads as a config."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Changed on disk")
        box.setText(f"{path.name} has changed on disk since this editor read it.")
        if readable:
            box.setInformativeText(
                "Another program wrote to it: the Config Editor, the Hub, or a "
                "catalogue that filed this recording.\n\n"
                "Reload reads it again.  What you changed here is lost.\n\n"
                "Overwrite saves this editor's About, design and chambers over "
                "the file as it is now.  Everything else in it is kept as it is now.")
            reload_button = box.addButton("Reload", QMessageBox.ButtonRole.AcceptRole)
        else:
            box.setInformativeText(
                "It no longer reads as a config, so there is nothing in it to "
                "keep.\n\nOverwrite replaces it with what this editor holds.")
            reload_button = None
        overwrite_button = box.addButton("Overwrite", QMessageBox.ButtonRole.DestructiveRole)
        box.setDefaultButton(box.addButton(QMessageBox.StandardButton.Cancel))
        box.exec()
        clicked = box.clickedButton()
        if clicked is overwrite_button:
            return OVERWRITE
        return RELOAD if reload_button is not None and clicked is reload_button else CANCEL

    def _set_base(self, cfg: dict) -> None:
        """*cfg*, just written, is now what the file holds: what the form
        does not show is kept from it at the next save."""
        self._loaded_raw = cfg
        self._show_run(about_mod.read_about(cfg))
        notes = about_mod.factor_notes(about_mod.read_about(cfg))
        self.factor_table._loaded_kinds = {k: v["kind"] for k, v in notes.items()}
        self.meaning_table._loaded = {k: dict(v["levels"]) for k, v in notes.items()}
        self._note_declared()

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
