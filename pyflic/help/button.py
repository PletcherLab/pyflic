"""The ``[?]`` help button and the F1 shortcut.

Both are thin: they hold a topic reference and call :func:`open_help`.  Nothing
here knows anything about the apps that use it — the dependency runs one way,
GUI → help, so removing this package breaks only its call sites.
"""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QSize, Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import QToolButton, QWidget

from ..base.ui.icons import help_color, icon
from . import topics as _topics
from .window import open_help

#: Default button edge in pixels.  Help buttons sit beside dense form rows, so
#: they are sized to be obviously clickable without pushing the row taller than
#: the spin boxes they annotate.
DEFAULT_SIZE = 24


class HelpButton(QToolButton):
    """A ``[?]`` that opens the help window at one topic.

    ``ref`` is a topic id, optionally with an anchor:
    ``"reference-parameters#feeding_event_link_gap"``.

    Rendered in amber rather than a category colour: help is an affordance that
    belongs to no analysis category, and the warm tint makes it findable
    against the blue/green/orange controls it sits beside.
    """

    def __init__(
        self,
        ref: str,
        parent: QWidget | None = None,
        *,
        tooltip: str | None = None,
        size: int = DEFAULT_SIZE,
    ) -> None:
        super().__init__(parent)
        self._topic_id, self._anchor = _topics.parse_ref(ref)
        self._size = size
        self._color: str | None = None
        self._restyling = False
        self.setIconSize(QSize(size - 4, size - 4))
        self.setFixedSize(size, size)
        self.setAutoRaise(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setToolTip(tooltip or "Open help for this section")
        self._apply_theme()
        self.clicked.connect(self._open)

    def _apply_theme(self) -> None:
        """Re-resolve the amber for the theme currently in force."""
        if self._restyling:
            return
        col = help_color()
        if col == self._color:
            return
        self._color = col
        self._restyling = True
        try:
            self.setIcon(icon("help", color=col))
            self.setStyleSheet(
                f"QToolButton {{"
                f"  border: none;"
                f"  border-radius: {self._size // 2}px;"
                f"  background: transparent;"
                f"}}"
                f"QToolButton:hover {{ background: {col}; }}"
            )
        finally:
            self._restyling = False

    def changeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        """Re-tint when the application theme changes.

        The hub and QC viewer both toggle light/dark at runtime.  Resolving
        the colour once in ``__init__`` left every help button showing the
        previous theme's amber until the app was restarted.

        Only palette changes are watched.  ``StyleChange`` would recurse —
        :meth:`_apply_theme` calls ``setStyleSheet``, which emits it.
        """
        if event.type() in (
            QEvent.Type.PaletteChange,
            QEvent.Type.ApplicationPaletteChange,
        ):
            self._apply_theme()
        super().changeEvent(event)

    @property
    def ref(self) -> str:
        return f"{self._topic_id}#{self._anchor}" if self._anchor else self._topic_id

    def _open(self) -> None:
        open_help(self._topic_id, self._anchor)


def install_help_shortcut(widget: QWidget, ref: str) -> QShortcut:
    """Bind F1 on *widget* to open help at *ref*."""
    topic_id, anchor = _topics.parse_ref(ref)
    sc = QShortcut(QKeySequence(Qt.Key.Key_F1), widget)
    sc.setContext(Qt.ShortcutContext.WindowShortcut)
    sc.activated.connect(lambda: open_help(topic_id, anchor))
    return sc
