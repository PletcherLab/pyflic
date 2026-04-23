"""Action palette — scrollable library of action tiles.

Emits :pyattr:`Palette.actionRequested` with the action string when the user
picks a tile (click or double-click on the keyboard-navigable list).
"""

from __future__ import annotations

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..ui import Category, category_color, icon
from .actions import ACTIONS, Action, actions_by_category


# ---------------------------------------------------------------------------
# Tile
# ---------------------------------------------------------------------------

class _ActionTile(QPushButton):
    """Single tile in the palette — icon + label + one-line blurb."""

    def __init__(self, action: Action, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.action = action
        self.setCheckable(False)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(56)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(10)

        ico = QLabel(self)
        ico.setPixmap(icon(action.icon, category=action.category).pixmap(22, 22))
        lay.addWidget(ico, 0, Qt.AlignmentFlag.AlignTop)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(0)
        lbl = QLabel(action.label, self)
        lbl.setStyleSheet("font-weight: 600;")
        text_col.addWidget(lbl)
        blurb = QLabel(action.blurb, self)
        blurb.setStyleSheet("color: palette(mid); font-size: 9pt;")
        blurb.setWordWrap(True)
        text_col.addWidget(blurb)
        lay.addLayout(text_col, 1)

        col = category_color(action.category)
        self.setStyleSheet(
            f"QPushButton {{"
            f"  border-left: 3px solid {col};"
            f"  border-top: 1px solid palette(mid);"
            f"  border-right: 1px solid palette(mid);"
            f"  border-bottom: 1px solid palette(mid);"
            f"  border-radius: 6px;"
            f"  padding: 0;"
            f"  text-align: left;"
            f"  background: palette(base);"
            f"}}"
            f"QPushButton:hover {{ background: palette(midlight); }}"
            f"QPushButton:disabled {{ color: palette(mid); }}"
        )
        self.setToolTip(action.blurb)

    def set_dimmed(self, dimmed: bool, reason: str = "") -> None:
        """Visually fade the tile when it doesn't match the experiment type."""
        self.setEnabled(not dimmed)  # keep clickable via re-enabling below
        # We still want the user to be able to add it (with a warning); so
        # re-enable but keep the faded look via stylesheet.
        self.setEnabled(True)
        if dimmed:
            self.setGraphicsEffect(None)
            self.setStyleSheet(
                self.styleSheet()
                + " QPushButton { color: palette(mid); }"
            )
            if reason:
                self.setToolTip(f"{self.action.blurb}\n\n⚠ {reason}")


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

class Palette(QWidget):
    """Scrollable action library with a search box and category headers."""

    actionRequested = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tiles: list[tuple[_ActionTile, Action]] = []
        self._category_headers: dict[Category, QLabel] = {}
        self._experiment_type: str | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        # Header
        title = QLabel("Actions", self)
        title.setStyleSheet("font-size: 12pt; font-weight: 600;")
        outer.addWidget(title)

        # Search
        self._search = QLineEdit(self)
        self._search.setPlaceholderText("Search actions…")
        self._search.textChanged.connect(self._refilter)
        outer.addWidget(self._search)

        # Scrollable list of tiles
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        host = QWidget()
        host.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self._tiles_lay = QVBoxLayout(host)
        self._tiles_lay.setContentsMargins(0, 0, 0, 0)
        self._tiles_lay.setSpacing(6)

        for cat, actions in actions_by_category().items():
            header = QLabel(cat.name, host)
            header.setStyleSheet(
                f"color: {category_color(cat)}; font-size: 10pt; "
                f"font-weight: 700; letter-spacing: 0.08em; padding-top: 6px;"
            )
            self._tiles_lay.addWidget(header)
            self._category_headers[cat] = header
            for act in actions:
                tile = _ActionTile(act, host)
                tile.clicked.connect(
                    lambda _checked, a=act.action: self.actionRequested.emit(a)
                )
                self._tiles_lay.addWidget(tile)
                self._tiles.append((tile, act))
        self._tiles_lay.addStretch(1)

        scroll.setWidget(host)
        outer.addWidget(scroll, 1)

        self.setMinimumWidth(260)

    def set_experiment_type(self, et: str | None) -> None:
        """Re-tint tiles so ones that don't match *et* appear dimmed."""
        self._experiment_type = et
        for tile, act in self._tiles:
            if act.requires and act.requires != et:
                tile.set_dimmed(
                    True,
                    reason=f"Requires experiment_type={act.requires!r}; "
                           f"current is {et or 'unspecified'}.",
                )
            else:
                tile.setEnabled(True)
                tile.setToolTip(act.blurb)

    # ------------------------------------------------------------------
    # Search filter
    # ------------------------------------------------------------------

    def _refilter(self, query: str) -> None:
        q = query.strip().lower()
        any_visible_in_cat: dict[Category, bool] = {}
        for tile, act in self._tiles:
            match = (
                not q
                or q in act.action.lower()
                or q in act.label.lower()
                or q in act.blurb.lower()
            )
            tile.setVisible(match)
            any_visible_in_cat[act.category] = any_visible_in_cat.get(
                act.category, False
            ) or match
        # Hide category headers for which no tile matches.
        for cat, header in self._category_headers.items():
            header.setVisible(any_visible_in_cat.get(cat, False))
