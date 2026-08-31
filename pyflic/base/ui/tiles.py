"""Tile-strip widgets for the Analysis Hub.

A :class:`StatusTile` is a compact, clickable live-status chip in the strip
across the top of the Hub; all of a tile's controls live in its
:class:`TilePanel` — an anchored overlay that drops down under the tile, hosting
the full :class:`~pyflic.base.ui.widgets.Card` widgets.  Panels are persistent
hidden children of the Hub's central widget (state survives open/close, and
``findChildren`` keeps working for tests), one open at a time.

Tiles never hide or move: an inapplicable tile is *dimmed* but stays clickable,
because its panel holds the control that fixes the missing state.
"""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject, QSize, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from . import Category, category_color, icon, resolved_mode, surface_colors

#: Tile and readout height — the strip is one band, so these must match.
TILE_HEIGHT = 84


def chrome() -> dict:
    return surface_colors()


class StatusTile(QFrame):
    """One strip tile: an icon + title row and up to two live summary lines.

    Three geometries (mirroring PyTrackingAnalysis's tile ribbon):

    * regular — the default chip;
    * ``wide=True`` — a container tile (Batch, Project, Experiment), the same
      height but scaled wider so the three levels read as the ribbon's anchors;
    * ``compact=True`` — a title-only chip for the Experiment sub-strip: no
      summary lines, the status lives in the tooltip instead.
    """

    clicked = pyqtSignal(str)

    #: Hard cap so a chatty summary can never widen the strip.
    _MAX_LINE_CHARS = 26

    MIN_WIDTH = 118
    MAX_WIDTH = 196
    #: Container tiles' width relative to a regular tile: 1.75×, then reduced
    #: by a quarter so the status readout keeps most of the strip.
    WIDE_SCALE = 1.75 * 0.75
    COMPACT_HEIGHT = 38
    COMPACT_MIN_WIDTH = 96
    COMPACT_MAX_WIDTH = 150

    def __init__(self, key: str, title: str, icon_name: str,
                 category: Category, parent: QWidget | None = None, *,
                 wide: bool = False, compact: bool = False) -> None:
        super().__init__(parent)
        self.key = key
        self._category = category
        self._dimmed = False
        self._active = False
        self._clickable = True
        self._compact = compact
        #: (left, right) corner radii — distinct chips keep all corners
        #: rounded; a caller can flatten interior seams.
        self._radii = (5, 5)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        ## A width RANGE, not a fixed size: fixed-width tiles force a minimum
        ## window width that small laptops cannot show.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        #: The summary cap scales with the tile: a wide tile's extra width
        #: must buy extra text, not just extra padding.
        self._max_line_chars = self._MAX_LINE_CHARS
        if compact:
            self.setMinimumWidth(self.COMPACT_MIN_WIDTH)
            self.setMaximumWidth(self.COMPACT_MAX_WIDTH)
            self.setFixedHeight(self.COMPACT_HEIGHT)
        else:
            scale = self.WIDE_SCALE if wide else 1.0
            self._max_line_chars = round(self._MAX_LINE_CHARS * scale)
            self.setMinimumWidth(round(self.MIN_WIDTH * scale))
            self.setMaximumWidth(round(self.MAX_WIDTH * scale))
            self.setFixedHeight(TILE_HEIGHT)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(2)

        head = QHBoxLayout()
        head.setSpacing(6)
        self._icon = icon(icon_name, category)
        self._icon_lbl = QLabel()
        head.addWidget(self._icon_lbl)
        self._title_lbl = QLabel(title.upper())
        head.addWidget(self._title_lbl)
        head.addStretch(1)
        lay.addLayout(head)

        self._summary_lbl = QLabel("")
        self._summary_lbl.setTextFormat(Qt.TextFormat.PlainText)
        lay.addWidget(self._summary_lbl, 1, Qt.AlignmentFlag.AlignTop)
        if compact:
            ## Title-only chip: the tile above / beside it already names the
            ## subject, so the status is a hover away rather than a line.
            self._summary_lbl.hide()
        self.restyle()

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        """Prefer the top of the width range; the layout may still compress."""
        hint = super().sizeHint()
        hint.setWidth(self.maximumWidth())
        return hint

    def set_summary(self, lines: list[str]) -> None:
        full = [str(line) for line in lines]
        if self._compact:
            ## No summary lines on a compact chip — tooltip only.
            self.setToolTip("\n".join(full))
            return
        clipped = []
        for line in full[:2]:
            if len(line) > self._max_line_chars:
                line = line[: self._max_line_chars - 1] + "…"
            clipped.append(line)
        self._summary_lbl.setText("\n".join(clipped))
        ## The untruncated summary is always one hover away.
        self.setToolTip("\n".join(full))

    def summary_text(self) -> str:
        return self._summary_lbl.text()

    def set_dimmed(self, dimmed: bool) -> None:
        if dimmed != self._dimmed:
            self._dimmed = dimmed
            self.restyle()

    def is_dimmed(self) -> bool:
        return self._dimmed

    def set_active(self, active: bool) -> None:
        if active != self._active:
            self._active = active
            self.restyle()

    def is_active(self) -> bool:
        return self._active

    def set_clickable(self, clickable: bool) -> None:
        """Tiles are normally dimmed-but-clickable (their panel holds the
        fix).  The one exception is a tile that opens no panel of its own —
        the Experiment group tile: with nothing loaded it is inert as well as
        dimmed, and its hint names where the fix is."""
        self._clickable = clickable
        self.setCursor(Qt.CursorShape.PointingHandCursor if clickable
                       else Qt.CursorShape.ArrowCursor)

    def is_clickable(self) -> bool:
        return self._clickable

    def set_rounding(self, left: int, right: int) -> None:
        self._radii = (left, right)
        self.restyle()

    def restyle(self) -> None:
        c = chrome()
        color = category_color(self._category)
        border = (f"2px solid {color}" if self._active
                  else f"1px solid {c['border']}")
        left, right = self._radii
        ## Dimming has to be *visible* — an inapplicable tile stays clickable
        ## (its panel holds the fix), so the only thing telling the user it is
        ## not ready is how it looks. The background recedes to the strip band
        ## and the summary text mutes; the title keeps its category color so
        ## the strip is still readable as a map at a glance.
        background = c["band"] if self._dimmed else c["hover"]
        if self._dimmed:
            body = c["muted"]
        else:
            body = "#ffffff" if resolved_mode() == "dark" else "#000000"
        self.setStyleSheet(
            f"StatusTile {{ background: {background}; border: {border}; "
            f"border-top-left-radius: {left}px; "
            f"border-bottom-left-radius: {left}px; "
            f"border-top-right-radius: {right}px; "
            f"border-bottom-right-radius: {right}px; }} "
            f"QLabel {{ color: {body}; background: transparent; border: none; "
            f"font-size: 9pt; }}")
        ## Every element mutes together — title, summary AND icon.  Words
        ## alone said it too quietly: seven equally bright chips read as seven
        ## equally available ones, and the tile that is dim is precisely the
        ## one whose panel holds the fix.
        title_color = c["muted"] if self._dimmed else color
        self._title_lbl.setStyleSheet(
            f"color: {title_color}; font-weight: 700; font-size: 9pt; "
            "letter-spacing: 0.06em; background: transparent; border: none;")
        mode = QIcon.Mode.Disabled if self._dimmed else QIcon.Mode.Normal
        self._icon_lbl.setPixmap(self._icon.pixmap(QSize(16, 16), mode))

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.MouseButton.LeftButton and self._clickable:
            self.clicked.emit(self.key)
        super().mousePressEvent(event)


class StatusReadout(QFrame):
    """The strip's right-hand readout: what is loaded right now.

    Not a tile — it opens nothing and is never dimmed.  It fills the strip's
    leftover width so the Hub can always answer "which project, and which
    member inside it?" without opening a panel first.
    """

    _MAX_ROWS = 4

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("StatusReadout")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(TILE_HEIGHT)
        self.setMinimumWidth(160)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(0)
        self._label = QLabel("")
        self._label.setTextFormat(Qt.TextFormat.RichText)
        self._label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self._label, 1, Qt.AlignmentFlag.AlignTop)
        self._plain = ""
        self.restyle()

    def set_rows(self, rows: list[tuple[str, str]]) -> None:
        """Render ``(label, value)`` pairs, one per line.  Extra rows go to the
        tooltip rather than growing the strip."""
        import html as _html

        c = chrome()
        parts = []
        for label, value in rows[: self._MAX_ROWS]:
            parts.append(
                f"<div style='margin:0'>"
                f"<span style='color:{c['muted']}'>{_html.escape(str(label))}:"
                f"</span> {_html.escape(str(value))}</div>")
        self._label.setText("".join(parts))
        self._plain = "\n".join(f"{lab}: {val}" for lab, val in rows)
        self.setToolTip(self._plain)

    def status_text(self) -> str:
        """The rendered rows as plain text — every row, not just the shown."""
        return self._plain

    def restyle(self) -> None:
        c = chrome()
        pop = "#ffffff" if resolved_mode() == "dark" else "#000000"
        self.setStyleSheet(
            f"QFrame#StatusReadout {{ background: {c['hover']}; "
            f"border: 1px solid {c['border']}; border-radius: 5px; }} "
            f"QLabel {{ color: {pop}; background: transparent; border: none; "
            f"font-size: 9pt; }}")


class TilePanel(QFrame):
    """The anchored overlay under a tile: a framed, scrollable host for Cards.

    Created once and shown/hidden — never destroyed — so a panel's widget state
    survives being closed, and a running task can grey its cards in place
    rather than the panel vanishing under the user.
    """

    def __init__(self, key: str, width: int,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.key = key
        self._panel_width = width
        self.setObjectName("TilePanel")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.restyle()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        host = QWidget()
        self._content_lay = QVBoxLayout(host)
        self._content_lay.setContentsMargins(8, 8, 8, 8)
        self._content_lay.setSpacing(12)
        self._scroll.setWidget(host)
        outer.addWidget(self._scroll)
        self.hide()

    def restyle(self) -> None:
        c = chrome()
        self.setStyleSheet(
            f"QFrame#TilePanel {{ background: {c['band']}; "
            f"border: 1px solid {c['border']}; border-radius: 10px; }}")

    def add_card(self, card: QWidget) -> None:
        """Reparent an existing Card into this panel — its handlers, children,
        and findChildren visibility all come along."""
        self._content_lay.addWidget(card)
        card.setVisible(True)

    def cards(self) -> list:
        """The Cards this panel hosts, in strip order.

        The Hub dims a whole panel by dimming its cards, and it must not reach
        past the panel into an unrelated Card that happens to share the
        window: ``findChildren`` from the Hub would.
        """
        from .widgets import Card

        host = self._scroll.widget()
        return [child for child in host.findChildren(Card)]

    def finish(self) -> None:
        self._content_lay.addStretch(1)

    def _content_height(self) -> int:
        """The height the panel's content wants right now.

        Recomputed rather than read from the cached hint: widgets rebuilt while
        the panel was hidden leave a stale hint, which opens the panel far too
        short on its first click.
        """
        host = self._scroll.widget()
        layout = host.layout()
        if layout is not None:
            layout.invalidate()
            layout.activate()
        return host.sizeHint().height() + 24

    def open_at(self, x: int, y: int, max_bottom: int) -> None:
        """Show anchored at (*x*, *y*), clamped so the panel never runs past
        *max_bottom* or the parent's right edge."""
        parent = self.parentWidget()
        width = min(self._panel_width, parent.width() - 16)
        height = max(140, min(self._content_height(), max_bottom - y - 8))
        x = max(8, min(x, parent.width() - width - 8))
        self.setGeometry(x, y, width, height)
        self.raise_()
        self.show()


class ClickAwayFilter(QObject):
    """App-level filter that ONLY forwards GUI-thread mouse presses.

    Installing a QWidget itself as an application event filter is fatal:
    application filters run in the receiving object's thread, so worker-thread
    events would call into GUI-widget machinery off-thread — a hard Qt abort.
    This plain QObject early-outs on everything except a main-thread
    MouseButtonPress and never swallows the event.
    """

    def __init__(self, owner) -> None:
        super().__init__(owner)   # parented: auto-removed when owner dies
        self._owner = owner

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt override)
        try:
            if (event.type() == QEvent.Type.MouseButtonPress
                    and QThread.currentThread() is self.thread()):
                self._owner._handle_click_away(event)
        except RuntimeError:
            ## The owner was destroyed between the event and this call.
            pass
        return False
