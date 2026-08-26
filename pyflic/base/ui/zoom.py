"""Zoomable view widgets used as PlotDock tabs for saved analysis/QC artifacts.

* :class:`ZoomableImageView` displays a PNG with wheel-zoom and a
  +/−/100%/Fit toolbar; image is anchored at the cursor when zooming.
* :class:`ZoomableTextView` displays a UTF-8 text file (txt/csv) in a
  read-only monospaced editor; the wheel zooms the font size with
  +/−/100% buttons.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class ZoomableImageView(QWidget):
    """Scrollable image viewer with wheel-zoom, +/−/100%/Fit toolbar.

    Starts in fit-to-window mode; the wheel switches to free zoom (anchored
    at the cursor).
    """

    _ZOOM_STEP = 1.15
    _ZOOM_MIN = 0.05
    _ZOOM_MAX = 8.0

    def __init__(
        self,
        source: "Path | str | QPixmap",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if isinstance(source, QPixmap):
            self._original = source
        else:
            self._original = QPixmap(str(source))
        self._zoom = 1.0
        self._fit = True

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(6, 4, 6, 4)
        toolbar.setSpacing(4)

        toolbar.addWidget(_mk_btn("−", "Zoom out", lambda: self.zoom_by(1.0 / self._ZOOM_STEP)))
        toolbar.addWidget(_mk_btn("+", "Zoom in", lambda: self.zoom_by(self._ZOOM_STEP)))
        toolbar.addWidget(_mk_btn("100%", "Actual size", self.reset_zoom))
        toolbar.addWidget(_mk_btn("Fit", "Fit to window", self.fit_window))
        self._zoom_label = QLabel("Fit")
        self._zoom_label.setStyleSheet("color: palette(mid); padding-left: 6px;")
        toolbar.addWidget(self._zoom_label)
        toolbar.addStretch(1)
        outer.addLayout(toolbar)

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll = QScrollArea()
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll.setWidget(self._label)
        self._scroll.setWidgetResizable(False)
        self._scroll.viewport().installEventFilter(self)
        outer.addWidget(self._scroll, 1)

        self._render()

    def is_empty(self) -> bool:
        return self._original.isNull()

    def zoom_by(self, factor: float, anchor=None) -> None:
        if self._original.isNull():
            return
        self._fit = False
        new_zoom = max(self._ZOOM_MIN, min(self._ZOOM_MAX, self._zoom * factor))
        if new_zoom == self._zoom:
            return
        if anchor is None:
            anchor = self._scroll.viewport().rect().center()
        h = self._scroll.horizontalScrollBar()
        v = self._scroll.verticalScrollBar()
        # Divide by the *current* zoom to get image coordinates. Clamping this
        # to 1.0 broke the anchor for everything below 100% (fit-to-window is
        # routinely ~0.3), scrolling the image hundreds of pixels off target.
        current = self._zoom if self._zoom > 0 else 1.0
        img_x = (h.value() + anchor.x()) / current
        img_y = (v.value() + anchor.y()) / current
        self._zoom = new_zoom
        self._render()
        h.setValue(int(img_x * self._zoom - anchor.x()))
        v.setValue(int(img_y * self._zoom - anchor.y()))

    def reset_zoom(self) -> None:
        self._fit = False
        self._zoom = 1.0
        self._render()

    def fit_window(self) -> None:
        self._fit = True
        self._render()

    def eventFilter(self, obj, event):  # noqa: N802 — Qt API
        if obj is self._scroll.viewport() and event.type() == QEvent.Type.Wheel:
            delta = event.angleDelta().y()
            if delta:
                factor = self._ZOOM_STEP if delta > 0 else 1.0 / self._ZOOM_STEP
                self.zoom_by(factor, anchor=event.position().toPoint())
            return True
        return super().eventFilter(obj, event)

    def resizeEvent(self, event) -> None:  # noqa: N802 — Qt API
        super().resizeEvent(event)
        if self._fit:
            self._render()

    def _render(self) -> None:
        if self._original.isNull():
            return
        if self._fit:
            vp = self._scroll.viewport().size()
            if vp.width() <= 0 or vp.height() <= 0:
                return
            scaled = self._original.scaled(
                vp,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._zoom = scaled.width() / max(1, self._original.width())
            self._zoom_label.setText("Fit")
        else:
            new_w = max(1, int(self._original.width() * self._zoom))
            new_h = max(1, int(self._original.height() * self._zoom))
            scaled = self._original.scaled(
                new_w, new_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._zoom_label.setText(f"{self._zoom * 100:.0f}%")
        self._label.setPixmap(scaled)
        self._label.resize(scaled.size())


class ZoomableTextView(QWidget):
    """Read-only rich-text view with +/−/100% font-size buttons.

    Content renders through :mod:`..ui.textformat`: CSVs become real tables,
    prose gets the proportional font with headings, and monospace appears
    only on tabular blocks. All generated sizes are relative, so scaling the
    default font zooms everything together. The mouse wheel is left to the
    underlying editor so it scrolls the text the way users expect; use the
    toolbar buttons to change size.
    """

    _ZOOM_STEP = 1.15
    _MIN_PT = 6.0
    _MAX_PT = 36.0

    def __init__(self, text_path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._path = Path(text_path)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(6, 4, 6, 4)
        toolbar.setSpacing(4)

        toolbar.addWidget(_mk_btn("−", "Smaller text", lambda: self.zoom_by(1.0 / self._ZOOM_STEP)))
        toolbar.addWidget(_mk_btn("+", "Larger text", lambda: self.zoom_by(self._ZOOM_STEP)))
        toolbar.addWidget(_mk_btn("100%", "Default size", self.reset_zoom))
        self._zoom_label = QLabel("")
        self._zoom_label.setStyleSheet("color: palette(mid); padding-left: 6px;")
        toolbar.addWidget(self._zoom_label)
        toolbar.addStretch(1)
        outer.addLayout(toolbar)

        self._editor = QTextEdit()
        self._editor.setReadOnly(True)
        font = self._editor.font()
        font.setPointSizeF(10.0)
        self._editor.setFont(font)
        self._default_pt = font.pointSizeF()
        outer.addWidget(self._editor, 1)

        from .textformat import file_to_html

        try:
            raw = self._path.read_text(encoding="utf-8", errors="replace")
            self._editor.setHtml(file_to_html(self._path, raw))
        except Exception as err:  # noqa: BLE001
            self._editor.setPlainText(f"(could not read {self._path}: {err})")
        self._update_label()

    def zoom_by(self, factor: float) -> None:
        font = self._editor.font()
        new_pt = max(self._MIN_PT, min(self._MAX_PT, font.pointSizeF() * factor))
        font.setPointSizeF(new_pt)
        self._editor.setFont(font)
        self._update_label()

    def reset_zoom(self) -> None:
        font = self._editor.font()
        font.setPointSizeF(self._default_pt)
        self._editor.setFont(font)
        self._update_label()

    def _update_label(self) -> None:
        ratio = self._editor.font().pointSizeF() / self._default_pt
        self._zoom_label.setText(f"{ratio * 100:.0f}%")


def _mk_btn(text: str, tooltip: str, slot) -> QToolButton:
    b = QToolButton()
    b.setText(text)
    b.setToolTip(tooltip)
    b.setAutoRaise(True)
    b.clicked.connect(slot)
    return b
