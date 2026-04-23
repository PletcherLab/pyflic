"""Live YAML preview of the active script (or the full ``scripts:`` block).

Read-only text widget; the parent window calls :meth:`Preview.set_script` (or
:meth:`Preview.set_scripts_block`) whenever the model changes.
"""

from __future__ import annotations

from typing import Any

import yaml
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class Preview(QWidget):
    """Collapsible YAML preview panel."""

    showFullToggled = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(10)
        title = QLabel("YAML preview", self)
        title.setStyleSheet("font-size: 11pt; font-weight: 600;")
        header_row.addWidget(title)

        self._full_cb = QCheckBox("Show full scripts: block", self)
        self._full_cb.setToolTip(
            "When checked, preview the full scripts: list as it will be "
            "written to the YAML.  When unchecked, preview just the current "
            "script."
        )
        self._full_cb.toggled.connect(self.showFullToggled)
        header_row.addWidget(self._full_cb)

        header_row.addStretch(1)
        outer.addLayout(header_row)

        self._text = QPlainTextEdit(self)
        self._text.setReadOnly(True)
        mono = QFont("JetBrains Mono")
        if not mono.exactMatch():
            mono = QFont("Menlo")
        if not mono.exactMatch():
            mono = QFont("Consolas")
        self._text.setFont(mono)
        self._text.setStyleSheet(
            "QPlainTextEdit { background: palette(alternate-base); "
            "border: 1px solid palette(mid); border-radius: 6px; padding: 6px; }"
        )
        self._text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._text.setMinimumHeight(120)
        outer.addWidget(self._text, 1)

    def is_show_full(self) -> bool:
        return self._full_cb.isChecked()

    def set_script(self, script: dict[str, Any]) -> None:
        self._text.setPlainText(self._dump_script(script))

    def set_scripts_block(self, scripts: list[dict[str, Any]]) -> None:
        block = {"scripts": scripts}
        self._text.setPlainText(
            yaml.dump(block, default_flow_style=False, allow_unicode=True, sort_keys=False)
        )

    @staticmethod
    def _dump_script(script: dict[str, Any]) -> str:
        return yaml.dump(
            [script],
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
