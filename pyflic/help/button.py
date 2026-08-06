"""The ``[?]`` help button and the F1 shortcut.

Both are thin: they hold a topic reference and call :func:`open_help`.  Nothing
here knows anything about the apps that use it — the dependency runs one way,
GUI → help, so removing this package breaks only its call sites.
"""

from __future__ import annotations

from PyQt6.QtCore import QSize, Qt
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
        col = help_color()
        self.setIcon(icon("help", color=col))
        self.setIconSize(QSize(size - 4, size - 4))
        self.setFixedSize(size, size)
        self.setAutoRaise(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setToolTip(tooltip or "Open help for this section")
        self.setStyleSheet(
            f"QToolButton {{"
            f"  border: none;"
            f"  border-radius: {size // 2}px;"
            f"  background: transparent;"
            f"}}"
            f"QToolButton:hover {{ background: {col}; }}"
        )
        self.clicked.connect(self._open)

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
