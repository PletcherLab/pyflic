"""Reusable themed widgets shared across the pyflic Qt apps.

* :class:`SidebarNav`  — vertical navigation rail with category-tinted items
* :class:`TopBar`      — app title + arbitrary right-aligned controls
* :class:`Card`        — rounded panel with title, optional subtitle, and a body layout
* :class:`CardGroup`   — titled box grouping the controls inside a Card that belong together
* :class:`ActionButton`— QPushButton with category-coloured left border + icon
* :class:`PlotDock`    — tabbed interactive plot dock (matplotlib + nav toolbar)
* :class:`OutputLog`   — monospaced log panel that grows scrollback
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPalette, QPixmap, QTextCursor
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .icons import icon
from .theme import Category, category_color, resolved_mode


# ---------------------------------------------------------------------------
# Sidebar navigation rail
# ---------------------------------------------------------------------------

class SidebarNav(QWidget):
    """Vertical navigation rail.

    Items emit :pyattr:`itemSelected` with the item's *key*.  Use
    :meth:`add_item` to register entries; the first added item is selected
    by default.
    """

    itemSelected = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None, *, width: int = 180) -> None:
        super().__init__(parent)
        self.setObjectName("PyflicSidebar")
        self.setFixedWidth(width)
        self.setAutoFillBackground(True)
        # Subtle alternate-row background distinct from main content.
        pal = self.palette()
        bg = pal.color(QPalette.ColorRole.Window).darker(105) \
            if resolved_mode() == "light" \
            else pal.color(QPalette.ColorRole.Window).lighter(110)
        pal.setColor(QPalette.ColorRole.Window, bg)
        self.setPalette(pal)

        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(8, 12, 8, 12)
        self._lay.setSpacing(2)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QPushButton] = {}

    def add_item(
        self,
        key: str,
        label: str,
        icon_name: str,
        *,
        category: Category | None = None,
        tooltip: str | None = None,
    ) -> QPushButton:
        btn = QPushButton(label, self)
        btn.setObjectName("PyflicSidebarItem")
        btn.setCheckable(True)
        btn.setIcon(icon(icon_name, category=category))
        btn.setIconSize(QSize(16, 16))
        if tooltip:
            btn.setToolTip(tooltip)
        btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        btn.clicked.connect(lambda _checked, k=key: self.itemSelected.emit(k))
        self._lay.addWidget(btn)
        self._group.addButton(btn)
        self._buttons[key] = btn
        if len(self._buttons) == 1:
            btn.setChecked(True)
        return btn

    def add_separator(self) -> None:
        line = QFrame(self)
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        self._lay.addWidget(line)

    def add_stretch(self) -> None:
        self._lay.addStretch(1)

    def select(self, key: str) -> None:
        btn = self._buttons.get(key)
        if btn is not None:
            btn.setChecked(True)
            self.itemSelected.emit(key)


# ---------------------------------------------------------------------------
# Top bar
# ---------------------------------------------------------------------------

class TopBar(QFrame):
    """Slim top bar with an app title on the left and slots on the right."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("PyflicTopBar")
        self.setFixedHeight(56)
        self.setFrameShape(QFrame.Shape.NoFrame)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 6, 12, 6)
        lay.setSpacing(10)

        self._title = QLabel(title, self)
        self._title.setObjectName("PyflicAppTitle")
        lay.addWidget(self._title)

        lay.addStretch(1)

        self._right_lay = QHBoxLayout()
        self._right_lay.setContentsMargins(0, 0, 0, 0)
        self._right_lay.setSpacing(8)
        right_host = QWidget(self)
        right_host.setLayout(self._right_lay)
        lay.addWidget(right_host)

    def add_right(self, widget: QWidget) -> None:
        self._right_lay.addWidget(widget)

    def set_title(self, title: str) -> None:
        self._title.setText(title)


# ---------------------------------------------------------------------------
# Card
# ---------------------------------------------------------------------------

class Card(QFrame):
    """Rounded panel with a category-tinted left border + title row.

    Use :meth:`body_layout` to add content widgets.
    """

    def __init__(
        self,
        title: str,
        category: Category = Category.NEUTRAL,
        subtitle: str | None = None,
        icon_name: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("PyflicCard")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._category = category

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(8)

        # Title row
        title_row = QHBoxLayout()
        title_row.setSpacing(8)

        self._icon = icon(icon_name, category=category) if icon_name else None
        self._icon_lbl: QLabel | None = None
        if self._icon is not None:
            self._icon_lbl = QLabel(self)
            title_row.addWidget(self._icon_lbl)

        self._title_lbl = QLabel(title, self)
        self._title_lbl.setObjectName("PyflicCardTitle")
        title_row.addWidget(self._title_lbl, 0)
        # Extra title-row widgets (the help button) are inserted here, right
        # after the title text, so they stay visible when the card is narrow.
        self._title_insert_at = title_row.count()
        title_row.addStretch(1)
        self._title_row = title_row

        outer.addLayout(title_row)

        self._subtitle_lbl: QLabel | None = None
        if subtitle:
            sub = QLabel(subtitle, self)
            sub.setObjectName("PyflicCardSubtitle")
            sub.setWordWrap(True)
            outer.addWidget(sub)
            self._subtitle_lbl = sub

        self._body = QVBoxLayout()
        self._body.setSpacing(8)
        outer.addLayout(self._body)

        self._dimmed = False
        self.setAutoFillBackground(True)
        self.restyle()

    def set_dimmed(self, dimmed: bool) -> None:
        """Grey the card's surface to show its actions have no subject yet.

        Dimming is presentation only — the card stays live so the control that
        fixes the missing state (an Open button, a checkbox) keeps working;
        the actions themselves are gated with ``setEnabled`` as before.
        """
        if dimmed != self._dimmed:
            self._dimmed = dimmed
            self.restyle()

    def is_dimmed(self) -> bool:
        return self._dimmed

    def restyle(self) -> None:
        """Repaint the card for the CURRENT theme and dim state.

        The colors come from ``surface_colors`` rather than palette roles,
        which qdarktheme leaves at the platform's light values, and are applied
        as this widget's own stylesheet — the app stylesheet's
        ``QFrame#PyflicCard`` background rule wins over a palette color.

        Every visible piece is repainted rather than fading the whole card with
        a ``QGraphicsOpacityEffect``: an effect composites the card over
        whatever is behind it, and behind it is a panel still painting the
        platform's LIGHT base, so on the dark theme the "dim" comes out
        brighter than the live card.
        """
        from .theme import surface_colors

        c = surface_colors()
        base = QColor(c["base"])
        if self._dimmed:
            ## Away from the live surface in the direction the theme reads as
            ## recessed, and far enough to survive a glance: on the dark theme
            ## a few points of lightness is invisible.
            bg = base.darker(112) if resolved_mode() == "light" \
                else base.darker(150)
            accent = text = c["muted"]
            border = f"1px solid {c['border']}"
        else:
            bg, accent, text = base, category_color(self._category), c["text"]
            border = "none"
        self.setStyleSheet(
            f"QFrame#PyflicCard {{ border-radius: 10px; "
            f"background: {bg.name()}; border: {border}; }}"
        )
        self._title_lbl.setStyleSheet(
            f"QLabel#PyflicCardTitle {{"
            f"  border-left: 4px solid {accent};"
            f"  padding-left: 8px;"
            f"  color: {text};"
            f"}}"
        )
        if self._subtitle_lbl is not None:
            self._subtitle_lbl.setStyleSheet(
                f"QLabel#PyflicCardSubtitle {{ color: {c['muted']}; }}"
            )
        if self._icon_lbl is not None and self._icon is not None:
            ## Qt's own greyed rendering — the category tint at full strength
            ## is the loudest thing left on a dimmed card.
            mode = QIcon.Mode.Disabled if self._dimmed else QIcon.Mode.Normal
            self._icon_lbl.setPixmap(self._icon.pixmap(QSize(20, 20), mode))
        ## The groups inside paint from the same surfaces and are repainted
        ## with the card, so a theme toggle reaches them too.
        for group in self.findChildren(CardGroup):
            group.restyle()

    def add_title_widget(self, widget: QWidget) -> None:
        """Add *widget* to the title row, immediately after the title text.

        Used for the card's help button.  Placed beside the title rather than
        right-aligned so it cannot be clipped when the cards column is narrow.
        The card itself knows nothing about help; it just offers the slot.
        """
        self._title_row.insertWidget(
            self._title_insert_at, widget, 0, Qt.AlignmentFlag.AlignVCenter
        )
        self._title_insert_at += 1

    def body_layout(self) -> QVBoxLayout:
        return self._body

    def add_body(self, widget_or_layout: QWidget | Any) -> None:
        if isinstance(widget_or_layout, QWidget):
            self._body.addWidget(widget_or_layout)
        else:
            self._body.addLayout(widget_or_layout)

    def set_title(self, title: str) -> None:
        self._title_lbl.setText(title)

    def add_section_label(self, text: str) -> None:
        lbl = QLabel(text, self)
        lbl.setObjectName("PyflicSectionDivider")
        self._body.addWidget(lbl)


# ---------------------------------------------------------------------------
# Card group
# ---------------------------------------------------------------------------

class CardGroup(QGroupBox):
    """A titled box for the controls inside a :class:`Card` that belong together.

    A Card lists its actions one under the other, which reads as "these are all
    the same kind of thing".  When some of them are not — a control that steers
    only two of the buttons, a set of buttons that exists only for one
    Experiment Type — the flat list actively misleads, so those go in a group
    whose title says what they share.

    Presentation only: the group owns no state, just the layout its members
    sit in.  Use :meth:`add` for each member and :meth:`add_note` for the
    one-line explanation of what the group is.
    """

    def __init__(self, title: str, parent: QWidget | None = None, *,
                 note: str | None = None) -> None:
        super().__init__(title, parent)
        self.setObjectName("PyflicCardGroup")
        self._body = QVBoxLayout(self)
        ## The top margin is the title's clearance and nothing more: the
        ## title is drawn in the frame's own margin box, so any padding here
        ## opens a band of empty group under its own name.
        self._body.setContentsMargins(8, 7, 8, 8)
        self._body.setSpacing(6)
        self._notes: list[QLabel] = []
        if note:
            self.add_note(note)
        self.restyle()

    def add_note(self, text: str) -> None:
        label = QLabel(text, self)
        label.setObjectName("PyflicCardGroupNote")
        label.setWordWrap(True)
        self._body.addWidget(label)
        self._notes.append(label)
        self.restyle()

    def restyle(self) -> None:
        """Repaint for the CURRENT theme, from ``surface_colors``.

        Not ``palette(mid)`` from the app stylesheet: qdarktheme leaves the
        palette roles at values that come out all but invisible on the dark
        theme, which turned the title into a ghost and the notes into blank
        vertical space — the group looked like a gap rather than a group.
        The Card beneath does the same for the same reason.
        """
        from .theme import surface_colors

        c = surface_colors()
        self.setStyleSheet(
            f"QGroupBox#PyflicCardGroup {{"
            f"  border: 1px solid {c['border']};"
            f"  border-radius: 8px;"
            f"  margin-top: 9px;"
            f"  padding: 0;"
            f"  font-size: 9pt;"
            f"  font-weight: 600;"
            f"  color: {c['text']};"
            f"}}"
            f"QGroupBox#PyflicCardGroup::title {{"
            f"  subcontrol-origin: margin;"
            f"  subcontrol-position: top left;"
            f"  left: 10px;"
            f"  padding: 0 4px;"
            f"  color: {c['text']};"
            f"}}"
        )
        for label in self._notes:
            label.setStyleSheet(
                f"QLabel#PyflicCardGroupNote {{ color: {c['muted']}; "
                f"font-size: 9pt; font-weight: 400; }}")
        ## The stylesheet carries the box model, so the height this group
        ## asks for changes with it.
        self.updateGeometry()

    def add(self, widget_or_layout: QWidget | Any) -> None:
        if isinstance(widget_or_layout, QWidget):
            self._body.addWidget(widget_or_layout)
        else:
            self._body.addLayout(widget_or_layout)


# ---------------------------------------------------------------------------
# Action button
# ---------------------------------------------------------------------------

class ActionButton(QPushButton):
    """QPushButton with a category-coloured left accent and themed icon.

    The button is willing to shrink below its natural text width — long
    labels are clipped (the full text remains readable as a tooltip) so it
    never forces the surrounding card to overflow.
    """

    def __init__(
        self,
        text: str,
        category: Category = Category.NEUTRAL,
        icon_name: str | None = None,
        *,
        primary: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(text, parent)
        self._category = category
        # Tooltip echoes the full label so it stays discoverable when the
        # button is narrower than its text.
        if not self.toolTip():
            self.setToolTip(text)
        # Allow horizontal compression so a long label cannot push the parent
        # card wider than its column.
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        if icon_name is not None:
            self.setIcon(icon(icon_name, category=category))
            self.setIconSize(QSize(16, 16))
        col = category_color(category)
        weight = "600" if primary else "500"
        bg = "palette(highlight)" if primary else "palette(button)"
        fg = "palette(highlighted-text)" if primary else "palette(button-text)"
        self.setStyleSheet(
            f"QPushButton {{"
            f"  border-left: 3px solid {col};"
            f"  border-radius: 6px;"
            f"  padding: 6px 12px;"
            f"  font-weight: {weight};"
            f"  background: {bg};"
            f"  color: {fg};"
            f"}}"
            f"QPushButton:hover {{ background: {col}; color: white; }}"
            ## The background too, not just the text: a disabled *primary*
            ## button kept the highlight fill and read as the live thing to
            ## click in a panel where nothing was clickable.
            f"QPushButton:disabled {{"
            f"  background: palette(button);"
            f"  color: palette(mid);"
            f"  border-left-color: palette(mid);"
            f"}}"
        )


# ---------------------------------------------------------------------------
# Output log
# ---------------------------------------------------------------------------

class OutputLog(QPlainTextEdit):
    """Read-only log panel with a capped scrollback.

    Lines render as rich text — proportional prose with muted ``[prefix]``
    tags, accents for failures and warnings, and the monospace font only when
    a line's spacing is tabular (see :mod:`pyflic.base.ui.textformat`).
    """

    line_appended = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None, *,
                 max_lines: int = 5000) -> None:
        super().__init__(parent)
        self.setObjectName("PyflicLog")
        self.setReadOnly(True)
        self.setMaximumBlockCount(max_lines)
        #: Text written without a closing newline, waiting for the rest of its
        #: line.  It is displayed immediately (as its own block) and that block
        #: is rewritten when the remainder arrives.
        self._pending = ""
        self._pending_shown = False

    def append_line(self, text: str) -> None:
        """Append *text* as one or more COMPLETE lines.

        For callers that hand over a finished message — most of the app.  A
        trailing newline is optional and never treated as "more to come", so
        two consecutive messages cannot run together.  Embedded newlines still
        split into separate blocks: ``appendHtml`` renders its argument as a
        single HTML fragment, where a newline is mere whitespace — which is
        what runs whole tables together on one line.
        """
        if not text:
            return
        self._flush_pending()
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()          # a trailing newline closes, it does not add
        for line in lines:
            self._append_one(line)
        self._after_append(text)

    def append_stream(self, chunk: str) -> None:
        """Append a raw chunk from a redirected ``stdout``.

        Unlike :meth:`append_line` a chunk has no line discipline: ``print``
        writes its text and its terminator separately, so one call can carry
        several lines, a bare newline, or the front half of a line.  The
        trailing fragment is shown immediately and rewritten in place when the
        rest of it arrives, so nothing appears twice.
        """
        if not chunk:
            return
        lines = (self._pending + chunk).split("\n")
        self._pending = lines.pop()
        if self._pending_shown:
            self._drop_last_block()
            self._pending_shown = False
        for line in lines:
            self._append_one(line)
        if self._pending:
            self._append_one(self._pending)
            self._pending_shown = True
        self._after_append(chunk)

    def clear_log(self) -> None:
        """Erase the scrollback, including any partially-streamed line."""
        self.clear()
        self._flush_pending()

    def _flush_pending(self) -> None:
        """Close off a partial streamed line so a complete message from
        somewhere else cannot be glued onto its end."""
        self._pending = ""
        self._pending_shown = False

    def _after_append(self, text: str) -> None:
        self.moveCursor(QTextCursor.MoveOperation.End)
        self.ensureCursorVisible()
        self.line_appended.emit(text)

    def _append_one(self, line: str) -> None:
        from .textformat import log_line_to_html

        stripped = line.rstrip()
        if stripped:
            self.appendHtml(log_line_to_html(stripped))
        else:
            self.appendPlainText("")

    def _drop_last_block(self) -> None:
        """Remove the block holding the partial line, so the completed line
        replaces it rather than appearing twice."""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
        cursor.removeSelectedText()


# ---------------------------------------------------------------------------
# Plot dock
# ---------------------------------------------------------------------------

def _close_figure(figure: Any) -> None:
    """Unregister *figure* from pyplot, ignoring anything that goes wrong."""
    try:
        import matplotlib.pyplot as _plt

        _plt.close(figure)
    except Exception:  # noqa: BLE001
        pass


def _close_figure_when_destroyed(widget: QWidget, figure: Any) -> None:
    """Close *figure* once *widget*'s C++ object goes away.

    ``destroyed`` fires after ``deleteLater`` has run, which is exactly when
    the embedded canvas stops needing the figure.  The slot touches no Qt
    state, so it is safe at that point in the object's life.
    """
    # Bind both the figure and the helper as default arguments: the signal can
    # fire from the garbage collector or at interpreter shutdown, when free
    # variables and module globals are no longer reachable, and an exception
    # raised inside a Qt slot takes the process down.
    widget.destroyed.connect(
        lambda *_args, fig=figure, close=_close_figure: close(fig)
    )


class PlotDock(QTabWidget):
    """Tabbed dock for matplotlib figures and saved artifacts.

    The first tab is always *Output* (the supplied :class:`OutputLog`); an
    optional second permanent *Errors* tab (``error_log``) collects warnings
    and failures so they are not lost in the normal output.  Subsequent tabs
    are added by :meth:`add_figure` / :meth:`add_widget` and are individually
    closable.
    """

    def __init__(
        self,
        output_log: OutputLog,
        error_log: OutputLog | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setTabsClosable(True)
        self.setMovable(True)
        self.setDocumentMode(True)
        self.tabCloseRequested.connect(self._on_close)

        no_btn = self.tabBar().ButtonPosition.RightSide
        self._output_log = output_log
        self._error_log = error_log
        self._unseen_issues = 0
        # Output tab is always present and not closable.
        self.addTab(output_log, icon("info"), "Output")
        self.tabBar().setTabButton(0, no_btn, None)
        if error_log is not None:
            idx = self.addTab(error_log, icon("warning"), "Errors")
            self.tabBar().setTabButton(idx, no_btn, None)
            # Badge the Errors tab when lines arrive while it isn't visible.
            error_log.line_appended.connect(self._on_issue_logged)
            self.currentChanged.connect(self._on_tab_changed)

        self.setCornerWidget(self._build_clear_bar(), Qt.Corner.TopRightCorner)

    def _build_clear_bar(self) -> QWidget:
        """Row of clear buttons shown in the dock's top-right corner.

        One per thing that accumulates: the analysis tabs, the Output log, and
        (when present) the Errors log.  A long run fills all three, and a
        "close every tab" gesture that lives on each tab's own X is no gesture
        at all once there are forty of them.
        """
        buttons: list[tuple[str, str, Any]] = [
            ("Clear Tabs",
             "Close all figure and artifact tabs and show the Output tab.",
             self.clear_figures),
            ("Clear Output", "Erase the contents of the Output tab.",
             self.clear_output),
        ]
        if self._error_log is not None:
            buttons.append(("Clear Errors",
                            "Erase the contents of the Errors tab.",
                            self.clear_errors))

        bar = QWidget(self)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(0, 0, 4, 0)
        lay.setSpacing(2)
        for text, tip, slot in buttons:
            btn = QToolButton(bar)
            btn.setText(text)
            btn.setIcon(icon("clear"))
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            btn.setAutoRaise(True)
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            lay.addWidget(btn)
        return bar

    def _on_issue_logged(self, _text: str) -> None:
        idx = self.indexOf(self._error_log)
        if idx < 0 or self.currentWidget() is self._error_log:
            return
        self._unseen_issues += 1
        self.setTabText(idx, f"Errors ({self._unseen_issues})")

    def _on_tab_changed(self, _idx: int) -> None:
        if self.currentWidget() is self._error_log:
            self._unseen_issues = 0
            self.setTabText(self.indexOf(self._error_log), "Errors")

    def clear_figures(self) -> None:
        """Close every added tab (everything but Output/Errors), show Output."""
        fixed = (self._output_log, self._error_log)
        for idx in range(self.count() - 1, -1, -1):
            w = self.widget(idx)
            if w in fixed:
                continue
            self.removeTab(idx)
            if w is not None:
                w.deleteLater()
        self.setCurrentWidget(self._output_log)

    def clear_output(self) -> None:
        """Erase everything in the Output log."""
        self._output_log.clear_log()

    def clear_errors(self) -> None:
        """Erase everything in the Errors log and drop its unseen badge."""
        if self._error_log is None:
            return
        self._error_log.clear_log()
        self._unseen_issues = 0
        idx = self.indexOf(self._error_log)
        if idx >= 0:
            self.setTabText(idx, "Errors")

    def _on_close(self, idx: int) -> None:
        if self.widget(idx) in (self._output_log, self._error_log):
            return
        w = self.widget(idx)
        self.removeTab(idx)
        if w is not None:
            w.deleteLater()

    def add_figure(self, title: str, figure: Any, *,
                   interactive: bool = False,
                   replace_existing: bool = False) -> QSize:
        """Embed *figure* (a matplotlib ``Figure``) as a tab.

        Plotnine ggplot objects are accepted too — they are drawn first.
        Returns the natural pixel size of the embedded content, so callers can
        grow the window to show it without scrolling.

        ``interactive=True`` uses matplotlib's native Qt canvas + navigation
        toolbar; ``interactive=False`` (the default) renders the figure to a
        PNG and shows it inside :class:`ZoomableImageView`, so the user can
        pan, wheel-zoom, and use the +/-/Fit buttons exactly as in the
        saved-artifact tabs.

        ``replace_existing=True`` reuses the tab that already carries *title*
        instead of opening another one — for views that re-render the same
        panel as the user clicks around, so tabs do not pile up unboundedly.
        """
        if not hasattr(figure, "savefig") and hasattr(figure, "draw"):
            figure = figure.draw()

        if interactive:
            # Lazy imports so headless smoke tests don't pull matplotlib
            # backends until needed.
            from matplotlib.backends.backend_qtagg import (
                FigureCanvasQTAgg,
                NavigationToolbar2QT,
            )

            host = QWidget(self)
            lay = QVBoxLayout(host)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(0)
            canvas = FigureCanvasQTAgg(figure)
            toolbar = NavigationToolbar2QT(canvas, host)
            lay.addWidget(toolbar)
            lay.addWidget(canvas, 1)
            try:
                import mplcursors

                host._mpl_cursor = mplcursors.cursor(figure, hover=True)
            except Exception:  # noqa: BLE001
                pass
            w_in, h_in = figure.get_size_inches()
            content_size = QSize(
                int(round(w_in * figure.dpi)),
                int(round(h_in * figure.dpi)) + toolbar.sizeHint().height(),
            )
            ## The tab owns the figure from here on.  Without this an
            ## interactive figure stays registered with pyplot forever —
            ## closing the tab frees the widget but leaves the full RGBA
            ## buffer and the source frames alive.
            _close_figure_when_destroyed(host, figure)
            widget: QWidget = host
        else:
            import io as _io

            from .zoom import ZoomableImageView

            buf = _io.BytesIO()
            try:
                figure.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            finally:
                # Also close on a savefig failure, which used to leak.
                _close_figure(figure)
            buf.seek(0)
            pix = QPixmap()
            pix.loadFromData(buf.getvalue())
            widget = ZoomableImageView(pix)
            content_size = QSize(pix.width(), pix.height())

        self.add_widget(title, widget, replace_existing=replace_existing)
        return content_size

    def add_widget(self, title: str, widget: QWidget, tab_icon: Any = None, *,
                   replace_existing: bool = False) -> int:
        """Add *widget* as a tab titled *title* and make it current.

        ``replace_existing=True`` reuses the tab that already carries *title*
        (deleting the widget it held) instead of opening a second one, so
        re-rendering the same panel does not grow the tab bar without bound.
        Returns the tab index.
        """
        if tab_icon is None:
            tab_icon = icon("plots", category=Category.PLOTS)
        if replace_existing:
            for existing in range(self.count()):
                if self.tabText(existing) != title:
                    continue
                old = self.widget(existing)
                if old in (self._output_log, self._error_log):
                    break
                self.removeTab(existing)
                if old is not None:
                    old.deleteLater()
                idx = self.insertTab(existing, widget, tab_icon, title)
                self.setCurrentIndex(idx)
                return idx

        idx = self.addTab(widget, tab_icon, title)
        self.setCurrentIndex(idx)
        return idx
