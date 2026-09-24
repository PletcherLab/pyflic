"""Page layout for pyflic's PDF reports: US Letter, one look, no pyplot pages.

Both reports — a member's ``analysis/experiment_report.pdf`` and the Project
Report — are built from the same blocks: a cover, headings, paragraphs,
key–value lists, callouts, bullet lists, tables and plots.  A
:class:`ReportDocument` lays them out top to bottom on fixed 8.5 × 11 in pages
with a running header and a "Page n of N" footer, then writes the PDF.

Three rules the old reports broke, kept here by construction:

* **Every page is the same size.**  Pages are never cropped to their content.
* **The report owns the heading above a plot.**  A plotnine figure is drawn
  into a box of known size with its own title removed, so the two can never
  overlap; the plot's caption, if it has one, stays.
* **Tables flow.**  Column widths come from the text, long text columns wrap,
  and a table longer than the page continues on the next with its header
  repeated.

Everything is plain matplotlib: text is measured with the same font the PDF
embeds, and the pages are ``matplotlib.figure.Figure`` objects built without
pyplot, so a report written on the Hub's worker thread never asks for a GUI
canvas.  Plots are embedded as 200 dpi images.
"""

from __future__ import annotations

import datetime as _dt
import io
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Geometry, type and colour
# ---------------------------------------------------------------------------

PAGE_W, PAGE_H = 8.5, 11.0            # US Letter, inches
MARGIN_X = 0.75
TOP = 0.95                            # content starts this far below the top edge
BOTTOM = 0.85                         # ...and ends this far above the bottom edge
CONTENT_W = PAGE_W - 2 * MARGIN_X
CONTENT_H = PAGE_H - TOP - BOTTOM

SIZE_TITLE = 24.0
SIZE_SUBTITLE = 12.0
SIZE_H1 = 14.5
SIZE_H2 = 11.0
SIZE_BODY = 9.5
SIZE_SMALL = 8.0
SIZE_TABLE = 7.8
SIZE_TABLE_MIN = 6.6
LEADING = 1.38                        # line height as a multiple of the font size

INK = "#1f2933"
MUTED = "#5f6b7a"
RULE = "#d5dbe3"
HEAD_FILL = "#e9eef5"
ZEBRA = "#f6f8fa"
ACCENT = "#1f5fbf"

#: tone -> (text/bar colour, fill colour)
TONES: dict[str, tuple[str, str]] = {
    "ok": ("#1a7f37", "#e7f5ec"),
    "warning": ("#8a5a00", "#fff4d1"),
    "failed": ("#b42318", "#fdecec"),
    "info": ("#1f5fbf", "#eaf1fb"),
    "neutral": ("#5f6b7a", "#f1f3f6"),
}

FIGURE_DPI = 200


@lru_cache(maxsize=1)
def font_family() -> str:
    """The report's font: the first of Arial, Helvetica, Liberation Sans and
    Segoe UI that matplotlib can find, else its own DejaVu Sans — the same
    family the Plot Editor's default style names, so figures and pages
    match."""
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in ("Arial", "Helvetica", "Liberation Sans", "Segoe UI"):
        if candidate in names:
            return candidate
    return "DejaVu Sans"


def _props(size: float, bold: bool = False, italic: bool = False):
    from matplotlib.font_manager import FontProperties

    return FontProperties(family=font_family(), size=size,
                          weight="bold" if bold else "normal",
                          style="italic" if italic else "normal")


_TEXT_TO_PATH = None
_WIDTHS: dict[tuple[str, float, bool], float] = {}


def text_width(text: str, size: float, bold: bool = False) -> float:
    """Width of *text* in inches, measured with the report's font."""
    key = (text, float(size), bool(bold))
    cached = _WIDTHS.get(key)
    if cached is not None:
        return cached
    global _TEXT_TO_PATH
    if _TEXT_TO_PATH is None:
        from matplotlib.textpath import TextToPath

        _TEXT_TO_PATH = TextToPath()
    if not text:
        width = 0.0
    else:
        w, _h, _d = _TEXT_TO_PATH.get_text_width_height_descent(
            text, _props(size, bold), ismath=False)
        width = float(w) / 72.0
    _WIDTHS[key] = width
    return width


def line_height(size: float) -> float:
    return size * LEADING / 72.0


def wrap_text(text: str, width: float, size: float, bold: bool = False) -> list[str]:
    """*text* broken into lines no wider than *width* inches.  Explicit
    newlines are kept; a word longer than the line is broken by character."""
    lines: list[str] = []
    for para in str(text).split("\n"):
        words = para.split(" ")
        current = ""
        for word in words:
            trial = word if not current else f"{current} {word}"
            if text_width(trial, size, bold) <= width:
                current = trial
                continue
            if current:
                lines.append(current)
            ## A single word wider than the line: break it where it must.
            while text_width(word, size, bold) > width and len(word) > 1:
                cut = len(word)
                while cut > 1 and text_width(word[:cut], size, bold) > width:
                    cut -= 1
                lines.append(word[:cut])
                word = word[cut:]
            current = word
        lines.append(current)
    return lines or [""]


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

@dataclass
class Cover:
    """The top of page one: a small kind label, the subject, the lines under it."""
    kind: str                       # e.g. "Experiment report"
    subject: str                    # e.g. the member's name
    lines: Sequence[str] = ()       # e.g. type · layout, date


@dataclass
class Heading:
    text: str
    level: int = 1                  # 1 = section, 2 = subsection
    numbered: bool = True


@dataclass
class Paragraph:
    text: str
    size: float = SIZE_BODY
    color: str = INK
    italic: bool = False


@dataclass
class Bullets:
    items: Sequence[str]
    size: float = SIZE_BODY
    color: str = INK


@dataclass
class KeyValues:
    rows: Sequence[tuple[str, str]]
    key_width: float = 1.75


@dataclass
class Callout:
    text: str
    tone: str = "info"              # ok | warning | failed | info | neutral
    title: str | None = None


@dataclass
class Table:
    frame: pd.DataFrame
    caption: str | None = None
    formats: dict[str, Any] = field(default_factory=dict)   # column -> fmt str or callable
    status: dict[str, Callable[[Any], str | None]] = field(default_factory=dict)
    max_rows: int | None = None
    font_size: float = SIZE_TABLE
    stretch: bool | None = None     # None: stretch when the table is already wide


@dataclass
class Plot:
    """A figure in a box of known size.  *plot* is a plotnine ``ggplot``, a
    matplotlib ``Figure``, or a zero-argument callable returning either — a
    callable is only called while the report is laid out, so a figure that
    fails costs that figure, not the report."""
    plot: Any
    height: float = 3.3
    title: str | None = None
    caption: str | None = None
    width: float | None = None      # default: the content width


@dataclass
class PlotRow:
    """Plots side by side, sharing one height."""
    plots: Sequence[Plot]
    height: float = 3.1
    gap: float = 0.25


@dataclass
class Contents:
    title: str = "Contents"


@dataclass
class PageBreak:
    pass


@dataclass
class Spacer:
    height: float = 0.12


# ---------------------------------------------------------------------------
# Formatting helpers the content builders share
# ---------------------------------------------------------------------------

def fmt_value(value: Any, fmt: Any = None) -> str:
    """One table cell as text.  *fmt* is a format string (``"{:.2f}"``), a
    callable, or ``None`` for pyflic's defaults: integers plain, other numbers
    to three significant figures (four when larger), missing as an en dash."""
    if callable(fmt):
        return str(fmt(value))
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    try:
        if pd.isna(value):
            return "–"
    except (TypeError, ValueError):
        pass
    if isinstance(fmt, str):
        try:
            return fmt.format(value)
        except (ValueError, TypeError):
            return str(value)
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if not math.isfinite(v):
            return "∞" if v > 0 else "−∞"
        if v == int(v) and abs(v) < 1e7:
            return f"{int(v)}"
        a = abs(v)
        if a >= 1000:
            return f"{v:,.0f}"
        if a >= 100:
            return f"{v:.1f}"
        if a >= 1:
            return f"{v:.2f}"
        if a >= 0.001:
            return f"{v:.3f}"
        return f"{v:.2g}"
    return str(value)


def fmt_p(value: Any) -> str:
    """A p-value: three significant figures, ``< 0.001`` below that."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "–"
    if not math.isfinite(v):
        return "–"
    if v < 0.001:
        return "< 0.001"
    return f"{v:.3f}"


def tone_of(value: Any) -> str | None:
    """A verdict cell's tone from its first word: *ok*/*pass* → ok,
    *warning*/*kept* → warning, *failed*/*excluded*/*retained* → failed."""
    s = str(value).strip().lower()
    if not s or s in ("nan", "–", "-"):
        return None
    if s.startswith(("fail", "exclud", "retain")):
        return "failed"
    if s.startswith(("warn", "kept")):
        return "warning"
    if s.startswith(("ok", "pass")):
        return "ok"
    return None


# ---------------------------------------------------------------------------
# Plots to images
# ---------------------------------------------------------------------------

def render_plot(plot: Any, width: float, height: float, *,
                drop_title: bool) -> tuple[np.ndarray, float, float]:
    """``(rgba pixels, width, height)`` of *plot* drawn for a box of
    *width* × *height* inches.  A ggplot is drawn at exactly that size (with
    the report's font, and its title removed when the report prints one); a
    matplotlib Figure keeps its own layout and is scaled to fit."""
    from PIL import Image

    if callable(plot) and not _is_figure_like(plot):
        plot = plot()
    buf = io.BytesIO()
    try:
        from plotnine import element_blank, ggplot, theme
    except ImportError:  # pragma: no cover
        ggplot = None  # type: ignore[assignment]
    if ggplot is not None and isinstance(plot, ggplot):
        ## The report's font reaches the figure through rcParams (see _rc):
        ## a theme(text=...) here would re-enable text a plot has blanked.
        extra = theme(figure_size=(width, height))
        if drop_title:
            extra = extra + theme(plot_title=element_blank())
        fig = (plot + extra).draw()
        fig.savefig(buf, format="png", dpi=FIGURE_DPI, facecolor="white")
        import matplotlib.pyplot as plt

        plt.close(fig)
    elif hasattr(plot, "savefig"):
        plot.savefig(buf, format="png", dpi=FIGURE_DPI, facecolor="white",
                     bbox_inches="tight")
    else:
        raise TypeError(f"cannot draw a {type(plot).__name__} as a figure")
    buf.seek(0)
    image = np.asarray(Image.open(buf).convert("RGBA"))
    h_px, w_px = image.shape[:2]
    ## Fit inside the box, keeping the aspect ratio.
    scale = min(width / (w_px / FIGURE_DPI), height / (h_px / FIGURE_DPI))
    return image, (w_px / FIGURE_DPI) * scale, (h_px / FIGURE_DPI) * scale


def _is_figure_like(obj: Any) -> bool:
    if hasattr(obj, "savefig"):
        return True
    try:
        from plotnine import ggplot

        return isinstance(obj, ggplot)
    except ImportError:  # pragma: no cover
        return False


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------

Draw = Callable[[Any], None]            # draws onto a page Figure


class ReportDocument:
    """Blocks in, a PDF out.  ``add`` blocks in reading order, then ``save``.

    Layout is two passes: the first measures every block, breaks pages and
    renders the plots; the second draws each page — header, footer with the
    final page count, contents with the final page numbers — and writes it.
    """

    def __init__(self, kind: str, subject: str, *, generated: str | None = None,
                 version: str | None = None) -> None:
        self.kind = kind
        self.subject = subject
        self.generated = generated or _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.version = version if version is not None else _pyflic_version()
        self.blocks: list[Any] = []
        self._pages: list[list[Draw]] = []
        self._toc: list[tuple[int, str, int]] = []
        self._y = TOP
        self._numbers = [0, 0]

    # ---- building -----------------------------------------------------

    def add(self, *blocks: Any) -> ReportDocument:
        for block in blocks:
            if block is None:
                continue
            if isinstance(block, (list, tuple)):
                self.add(*block)
            else:
                self.blocks.append(block)
        return self

    def save(self, path: str | Path) -> Path:
        """Lay out and write the PDF to *path*."""
        from matplotlib.backends.backend_pdf import PdfPages

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"Title": f"{self.kind} — {self.subject}",
                    "Author": "pyflic", "Subject": self.kind,
                    "Creator": f"pyflic {self.version}".strip()}
        with self._rc():
            with PdfPages(path, metadata=metadata) as pdf:
                for fig in self._render_pages():
                    pdf.savefig(fig)
        return path

    def save_pngs(self, directory: str | Path, *, dpi: int = 110,
                  stem: str = "page") -> list[Path]:
        """Every page as a PNG — the same pages ``save`` writes, for looking
        at a report where no PDF viewer is to hand (tests, previews)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        out: list[Path] = []
        with self._rc():
            for index, fig in enumerate(self._render_pages(), start=1):
                path = directory / f"{stem}_{index:02d}.png"
                fig.savefig(path, dpi=dpi, facecolor="white")
                out.append(path)
        return out

    def _rc(self):
        import matplotlib

        return matplotlib.rc_context({
            "pdf.fonttype": 42, "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.sans-serif": [font_family(), "DejaVu Sans"],
        })

    def _render_pages(self):
        from matplotlib.figure import Figure

        self._layout()
        total = len(self._pages)
        for index, ops in enumerate(self._pages):
            fig = Figure(figsize=(PAGE_W, PAGE_H))
            fig.patch.set_facecolor("white")
            if index > 0:
                self._draw_header(fig)
            self._draw_footer(fig, index + 1, total)
            for op in ops:
                op(fig)
            yield fig

    def page_count(self) -> int:
        """Pages after layout (``save`` lays out; this does too if needed)."""
        if not self._pages:
            self._layout()
        return len(self._pages)

    # ---- pass one: layout ---------------------------------------------

    def _layout(self) -> None:
        self._pages = [[]]
        self._toc = []
        self._y = TOP
        self._numbers = [0, 0]
        blocks = self.blocks
        for i, block in enumerate(blocks):
            if isinstance(block, PageBreak):
                self._new_page()
            elif isinstance(block, Cover):
                self._place_cover(block)
            elif isinstance(block, Heading):
                self._place_heading(block, blocks[i + 1:i + 5])
            elif isinstance(block, Paragraph):
                self._place_lines(wrap_text(block.text, CONTENT_W, block.size),
                                  block.size, color=block.color, italic=block.italic)
                self._y += 0.08
            elif isinstance(block, Bullets):
                self._place_bullets(block)
            elif isinstance(block, KeyValues):
                self._place_key_values(block)
            elif isinstance(block, Callout):
                self._place_callout(block)
            elif isinstance(block, Table):
                self._place_table(block)
            elif isinstance(block, Plot):
                self._place_plot_row(PlotRow([block], height=block.height))
            elif isinstance(block, PlotRow):
                self._place_plot_row(block)
            elif isinstance(block, Contents):
                self._place_contents(block)
            elif isinstance(block, Spacer):
                self._y += block.height
            else:  # pragma: no cover - a programming error, shown not hidden
                self._place_callout(Callout(f"(unknown block {type(block).__name__})",
                                            tone="failed"))

    def _remaining(self) -> float:
        return PAGE_H - BOTTOM - self._y

    def _new_page(self) -> None:
        if self._pages[-1] or self._y > TOP:
            self._pages.append([])
        self._y = TOP

    def _ensure(self, height: float) -> None:
        if height > self._remaining() and self._y > TOP + 1e-6:
            self._new_page()

    def _op(self, draw: Draw) -> None:
        self._pages[-1].append(draw)

    def _page_index(self) -> int:
        return len(self._pages) - 1

    # -- cover ------------------------------------------------------------

    def _place_cover(self, cover: Cover) -> None:
        y0 = self._y
        kind = cover.kind.upper()
        subject_lines = wrap_text(cover.subject, CONTENT_W, SIZE_TITLE, bold=True)
        extra = [line for text in cover.lines
                 for line in wrap_text(text, CONTENT_W, SIZE_SUBTITLE)]

        def draw(fig, y0=y0):
            y = y0 - 0.2
            _rect(fig, MARGIN_X, y - 0.02, 0.9, 0.06, ACCENT)
            _text(fig, MARGIN_X, y + 0.14, kind, SIZE_SMALL, bold=True, color=ACCENT)
            y += 0.42
            for line in subject_lines:
                _text(fig, MARGIN_X, y, line, SIZE_TITLE, bold=True)
                y += line_height(SIZE_TITLE) * 0.95
            y += 0.05
            for line in extra:
                _text(fig, MARGIN_X, y, line, SIZE_SUBTITLE, color=MUTED)
                y += line_height(SIZE_SUBTITLE)
            _hline(fig, MARGIN_X, PAGE_W - MARGIN_X, y + 0.1, RULE, 1.0)

        self._op(draw)
        self._y = (y0 + 0.22 + len(subject_lines) * line_height(SIZE_TITLE) * 0.95
                   + 0.05 + len(extra) * line_height(SIZE_SUBTITLE) + 0.35)

    # -- headings and text ------------------------------------------------

    def _heading_height(self, heading: Heading) -> float:
        size = SIZE_H1 if heading.level == 1 else SIZE_H2
        before = 0.28 if heading.level == 1 else 0.16
        return before + line_height(size) + (0.12 if heading.level == 1 else 0.06)

    def _min_height(self, block: Any) -> float:
        """How much of the following block must fit beside its heading."""
        if block is None:
            return 0.0
        if isinstance(block, (Plot, PlotRow)):
            ## Title, box and caption: a plot cannot split, so all of it.
            return block.height + 0.85
        if isinstance(block, Table):
            return 0.9
        if isinstance(block, Callout):
            return 0.6
        return 0.45

    def _keep_with(self, following: Sequence[Any]) -> float:
        """How much must fit under a heading: any intro paragraphs, then the
        first block after them — all of a plot (it cannot split), the start
        of anything else."""
        need = 0.0
        for block in following:
            if isinstance(block, Paragraph):
                need += len(wrap_text(block.text, CONTENT_W, block.size))                     * line_height(block.size) + 0.08
                continue
            if isinstance(block, Heading):
                ## A section heading straight over its first subsection
                ## keeps that one too.
                need += self._heading_height(block)
                continue
            if isinstance(block, PageBreak) or block is None:
                break
            need += (self._min_height(block) if isinstance(block, (Plot, PlotRow))
                     else min(self._min_height(block), 3.6))
            break
        return need

    def _place_heading(self, heading: Heading, following: Sequence[Any]) -> None:
        need = self._heading_height(heading) + min(self._keep_with(following),
                                                   CONTENT_H - 1.0)
        if self._remaining() < need:
            self._new_page()
        level = 1 if heading.level <= 1 else 2
        label = heading.text
        if heading.numbered:
            if level == 1:
                self._numbers = [self._numbers[0] + 1, 0]
                label = f"{self._numbers[0]}  {heading.text}"
            else:
                self._numbers[1] += 1
                label = f"{self._numbers[0]}.{self._numbers[1]}  {heading.text}"
        self._toc.append((level, label, self._page_index() + 1))
        size = SIZE_H1 if level == 1 else SIZE_H2
        before = 0.28 if level == 1 else 0.16
        if self._y <= TOP + 1e-6:
            before = 0.0
        y = self._y + before

        def draw(fig, y=y, label=label, size=size, level=level):
            _text(fig, MARGIN_X, y, label, size, bold=True,
                  color=INK if level == 1 else ACCENT)
            if level == 1:
                _hline(fig, MARGIN_X, PAGE_W - MARGIN_X,
                       y + line_height(size) + 0.03, ACCENT, 1.1)

        self._op(draw)
        self._y = y + line_height(size) + (0.14 if level == 1 else 0.06)

    def _place_lines(self, lines: list[str], size: float, *, color: str = INK,
                     italic: bool = False, indent: float = 0.0,
                     bold: bool = False) -> None:
        lh = line_height(size)
        for line in lines:
            if lh > self._remaining():
                self._new_page()
            y = self._y

            def draw(fig, y=y, line=line):
                _text(fig, MARGIN_X + indent, y, line, size, color=color,
                      italic=italic, bold=bold)

            self._op(draw)
            self._y += lh

    def _place_bullets(self, bullets: Bullets) -> None:
        lh = line_height(bullets.size)
        for item in bullets.items:
            lines = wrap_text(item, CONTENT_W - 0.2, bullets.size)
            self._ensure(min(len(lines), 3) * lh)
            first = True
            for line in lines:
                if lh > self._remaining():
                    self._new_page()
                y = self._y

                def draw(fig, y=y, line=line, first=first):
                    if first:
                        _text(fig, MARGIN_X + 0.05, y, "•", bullets.size,
                              color=ACCENT, bold=True)
                    _text(fig, MARGIN_X + 0.2, y, line, bullets.size,
                          color=bullets.color)

                self._op(draw)
                self._y += lh
                first = False
            self._y += 0.03
        self._y += 0.06

    def _place_key_values(self, kv: KeyValues) -> None:
        lh = line_height(SIZE_BODY)
        value_w = CONTENT_W - kv.key_width
        for key, value in kv.rows:
            lines = wrap_text(str(value), value_w, SIZE_BODY)
            height = len(lines) * lh + 0.04
            self._ensure(height)
            y = self._y

            def draw(fig, y=y, key=key, lines=lines):
                _text(fig, MARGIN_X, y + 0.01, str(key), SIZE_SMALL, bold=True,
                      color=MUTED)
                for j, line in enumerate(lines):
                    _text(fig, MARGIN_X + kv.key_width, y + j * lh, line, SIZE_BODY)

            self._op(draw)
            self._y += height
        self._y += 0.1

    def _place_callout(self, callout: Callout) -> None:
        bar, fill = TONES.get(callout.tone, TONES["info"])
        pad = 0.1
        inner_w = CONTENT_W - 2 * pad - 0.06
        lines = wrap_text(callout.text, inner_w, SIZE_BODY)
        title_h = line_height(SIZE_BODY) if callout.title else 0.0
        height = 2 * pad + title_h + len(lines) * line_height(SIZE_BODY) - 0.03
        self._ensure(height + 0.1)
        y = self._y

        def draw(fig, y=y):
            _rect(fig, MARGIN_X, y, CONTENT_W, height, fill)
            _rect(fig, MARGIN_X, y, 0.06, height, bar)
            yy = y + pad
            if callout.title:
                _text(fig, MARGIN_X + 0.06 + pad, yy, callout.title, SIZE_BODY,
                      bold=True, color=bar)
                yy += title_h
            for line in lines:
                _text(fig, MARGIN_X + 0.06 + pad, yy, line, SIZE_BODY)
                yy += line_height(SIZE_BODY)

        self._op(draw)
        self._y += height + 0.12

    def _place_contents(self, contents: Contents) -> None:
        ## The entries are known only after layout; their count is known now.
        entries = [b for b in self.blocks if isinstance(b, Heading)]
        lh = line_height(SIZE_BODY) * 1.12
        height = line_height(SIZE_H2) + 0.08 + len(entries) * lh
        self._ensure(min(height, CONTENT_H))
        y = self._y
        page = self._page_index()

        def draw(fig, y=y, page=page):
            _text(fig, MARGIN_X, y, contents.title, SIZE_H2, bold=True, color=ACCENT)
            yy = y + line_height(SIZE_H2) + 0.08
            ## Only the entries that land on this page's remaining space;
            ## a very long list is cut rather than overprinting the footer.
            for level, label, number in self._toc:
                if yy + lh > PAGE_H - BOTTOM:
                    break
                indent = 0.0 if level == 1 else 0.28
                size = SIZE_BODY if level == 1 else SIZE_SMALL + 0.5
                _text(fig, MARGIN_X + indent, yy, label, size, bold=(level == 1))
                right = PAGE_W - MARGIN_X
                num = str(number)
                _text(fig, right, yy, num, size, ha="right", bold=(level == 1))
                x0 = MARGIN_X + indent + text_width(label, size, level == 1) + 0.08
                x1 = right - text_width(num, size, level == 1) - 0.08
                if x1 > x0:
                    _hline(fig, x0, x1, yy + line_height(size) * 0.72, RULE, 0.8,
                           style=(0, (1, 2)))
                yy += lh

        self._op(draw)
        self._y += height + 0.15

    # -- tables ---------------------------------------------------------------

    def _place_table(self, table: Table) -> None:
        frame = table.frame
        if frame is None or len(frame.columns) == 0:
            return
        shown = frame if table.max_rows is None else frame.head(table.max_rows)
        columns = [str(c) for c in shown.columns]
        cells = [[fmt_value(v, table.formats.get(col)) for col, v in zip(columns, row)]
                 for row in shown.itertuples(index=False, name=None)]
        numeric = [pd.api.types.is_numeric_dtype(shown[c]) and
                   not pd.api.types.is_bool_dtype(shown[c]) for c in shown.columns]
        size = table.font_size
        pad = 0.06

        def natural(size: float) -> list[float]:
            widths = []
            for j, col in enumerate(columns):
                w = text_width(col, size, bold=True)
                for row in cells:
                    w = max(w, text_width(row[j], size))
                widths.append(w + 2 * pad)
            return widths

        widths = natural(size)
        while sum(widths) > CONTENT_W and size > SIZE_TABLE_MIN:
            size = round(size - 0.3, 2)
            widths = natural(size)
        if sum(widths) > CONTENT_W:
            ## Still too wide: wrap the widest text columns, widest first.
            excess = sum(widths) - CONTENT_W
            order = sorted(range(len(columns)), key=lambda j: -widths[j])
            for j in order:
                if excess <= 0:
                    break
                longest_word = max(columns[j].split(" "),
                                   key=lambda w: text_width(w, size, True))
                floor = max(0.7, text_width(longest_word, size, True) + 2 * pad + 0.02)
                cut = min(excess, widths[j] - floor)
                if cut > 0:
                    widths[j] -= cut
                    excess -= cut
        stretch = table.stretch if table.stretch is not None else sum(widths) > 0.6 * CONTENT_W
        if stretch and sum(widths) < CONTENT_W:
            extra = (CONTENT_W - sum(widths)) / len(widths)
            widths = [w + extra for w in widths]

        lh = line_height(size)
        header = [wrap_text(col, widths[j] - 2 * pad, size, bold=True)
                  for j, col in enumerate(columns)]
        header_h = max(len(h) for h in header) * lh + 2 * 0.035
        wrapped_rows = [[wrap_text(cell, widths[j] - 2 * pad, size)
                         for j, cell in enumerate(row)] for row in cells]
        row_heights = [max(len(c) for c in row) * lh + 2 * 0.035 for row in wrapped_rows]
        tones = [[(table.status[col](shown.iloc[i, j]) if col in table.status else None)
                  for j, col in enumerate(columns)] for i in range(len(shown))]
        caption_lines = (wrap_text(table.caption, CONTENT_W, SIZE_SMALL, bold=True)
                         if table.caption else [])
        caption_h = len(caption_lines) * line_height(SIZE_SMALL) + (0.04 if caption_lines else 0)

        first_rows = min(len(row_heights), 3)
        self._ensure(caption_h + header_h + sum(row_heights[:first_rows]) + 0.05)
        if caption_lines:
            self._place_lines(caption_lines, SIZE_SMALL, color=MUTED, bold=True)
            self._y += 0.04

        i = 0
        while True:
            y = self._y
            rows_here: list[int] = []
            height = header_h
            while i < len(row_heights) and height + row_heights[i] <= self._remaining():
                rows_here.append(i)
                height += row_heights[i]
                i += 1
            if not rows_here and i < len(row_heights):
                ## Not even one row fits under a header here.
                self._new_page()
                continue

            def draw(fig, y=y, rows=tuple(rows_here), size=size):
                x0 = MARGIN_X
                total_w = sum(widths)
                _rect(fig, x0, y, total_w, header_h, HEAD_FILL)
                xx = x0
                for j, lines in enumerate(header):
                    for k, line in enumerate(lines):
                        if numeric[j]:
                            _text(fig, xx + widths[j] - pad, y + 0.035 + k * lh, line,
                                  size, bold=True, ha="right")
                        else:
                            _text(fig, xx + pad, y + 0.035 + k * lh, line, size, bold=True)
                    xx += widths[j]
                yy = y + header_h
                for n, r in enumerate(rows):
                    rh = row_heights[r]
                    if n % 2 == 1:
                        _rect(fig, x0, yy, total_w, rh, ZEBRA)
                    xx = x0
                    for j, lines in enumerate(wrapped_rows[r]):
                        tone = tones[r][j]
                        color = INK
                        if tone:
                            bar, fill = TONES[tone]
                            _rect(fig, xx, yy, widths[j], rh, fill)
                            color = bar
                        for k, line in enumerate(lines):
                            if numeric[j]:
                                _text(fig, xx + widths[j] - pad, yy + 0.035 + k * lh,
                                      line, size, ha="right", color=color,
                                      bold=bool(tone))
                            else:
                                _text(fig, xx + pad, yy + 0.035 + k * lh, line, size,
                                      color=color, bold=bool(tone))
                        xx += widths[j]
                    yy += rh
                _hline(fig, x0, x0 + total_w, y, "#9aa5b1", 0.8)
                _hline(fig, x0, x0 + total_w, y + header_h, "#9aa5b1", 0.6)
                _hline(fig, x0, x0 + total_w, yy, "#9aa5b1", 0.8)

            self._op(draw)
            self._y = y + height
            if i >= len(row_heights):
                break
            self._new_page()
        if table.max_rows is not None and len(frame) > table.max_rows:
            self._place_lines([f"… {len(frame) - table.max_rows} more row(s) in the CSV."],
                              SIZE_SMALL, color=MUTED, italic=True)
        self._y += 0.16

    # -- plots ----------------------------------------------------------------

    def _place_plot_row(self, row: PlotRow) -> None:
        plots = list(row.plots)
        if not plots:
            return
        n = len(plots)
        gap = row.gap if n > 1 else 0.0
        box_w = (CONTENT_W - gap * (n - 1)) / n
        title_h = max((len(wrap_text(p.title, box_w, SIZE_BODY, bold=True))
                       * line_height(SIZE_BODY) + 0.05) if p.title else 0.0
                      for p in plots)
        captions = [wrap_text(p.caption, box_w, SIZE_SMALL) if p.caption else []
                    for p in plots]
        caption_h = max((len(c) * line_height(SIZE_SMALL) + 0.05) if c else 0.0
                        for c in captions)
        box_h = min(row.height, CONTENT_H - title_h - caption_h - 0.1)
        rendered = []
        for p in plots:
            width = min(p.width or box_w, box_w)
            try:
                image, w, h = render_plot(p.plot, width, box_h, drop_title=bool(p.title))
                rendered.append((image, w, h, None))
            except Exception as exc:  # noqa: BLE001 - one figure, not the report
                rendered.append((None, width, min(box_h, 0.6),
                                 f"This figure could not be drawn: {exc}"))
        body_h = max(h for _img, _w, h, _err in rendered)
        self._ensure(title_h + body_h + caption_h + 0.1)
        y = self._y

        def draw(fig, y=y):
            for k, (p, (image, w, h, err)) in enumerate(zip(plots, rendered)):
                x = MARGIN_X + k * (box_w + gap)
                yy = y
                if p.title:
                    for line in wrap_text(p.title, box_w, SIZE_BODY, bold=True):
                        _text(fig, x, yy, line, SIZE_BODY, bold=True)
                        yy += line_height(SIZE_BODY)
                    yy += 0.05
                    yy = y + title_h
                if err:
                    bar, fill = TONES["failed"]
                    _rect(fig, x, yy, box_w, h, fill)
                    for j, line in enumerate(wrap_text(err, box_w - 0.2, SIZE_SMALL)):
                        _text(fig, x + 0.1, yy + 0.08 + j * line_height(SIZE_SMALL),
                              line, SIZE_SMALL, color=bar)
                else:
                    xi = x + (box_w - w) / 2
                    ax = fig.add_axes([xi / PAGE_W, 1 - (yy + h) / PAGE_H,
                                       w / PAGE_W, h / PAGE_H])
                    ax.imshow(image, interpolation="none", aspect="auto")
                    ax.set_axis_off()
                cy = y + title_h + body_h + 0.05
                for line in captions[k]:
                    _text(fig, x, cy, line, SIZE_SMALL, color=MUTED)
                    cy += line_height(SIZE_SMALL)

        self._op(draw)
        self._y = y + title_h + body_h + caption_h + 0.22

    # ---- pass two: chrome -------------------------------------------------

    def _draw_header(self, fig) -> None:
        y = 0.5
        _text(fig, MARGIN_X, y, f"pyflic · {self.kind}", SIZE_SMALL, color=MUTED)
        _text(fig, PAGE_W - MARGIN_X, y, self.subject, SIZE_SMALL, color=MUTED,
              ha="right", bold=True)
        _hline(fig, MARGIN_X, PAGE_W - MARGIN_X, y + line_height(SIZE_SMALL) + 0.02,
               RULE, 0.8)

    def _draw_footer(self, fig, page: int, total: int) -> None:
        y = PAGE_H - 0.55
        _hline(fig, MARGIN_X, PAGE_W - MARGIN_X, y - 0.06, RULE, 0.8)
        left = f"Generated {self.generated}"
        if self.version:
            left += f" · pyflic {self.version}"
        _text(fig, MARGIN_X, y, left, SIZE_SMALL - 0.5, color=MUTED)
        _text(fig, PAGE_W - MARGIN_X, y, f"Page {page} of {total}", SIZE_SMALL - 0.5,
              color=MUTED, ha="right")


# ---------------------------------------------------------------------------
# Drawing primitives (page coordinates: inches from the top-left corner)
# ---------------------------------------------------------------------------

def _text(fig, x: float, y: float, s: str, size: float, *, bold: bool = False,
          italic: bool = False, color: str = INK, ha: str = "left") -> None:
    fig.text(x / PAGE_W, 1 - y / PAGE_H, s, fontproperties=_props(size, bold, italic),
             color=color, ha=ha, va="top", parse_math=False)


def _rect(fig, x: float, y: float, w: float, h: float, color: str) -> None:
    from matplotlib.patches import Rectangle

    fig.add_artist(Rectangle((x / PAGE_W, 1 - (y + h) / PAGE_H), w / PAGE_W, h / PAGE_H,
                             transform=fig.transFigure, facecolor=color,
                             edgecolor="none", zorder=0))


def _hline(fig, x0: float, x1: float, y: float, color: str, width: float,
           style: Any = "-") -> None:
    from matplotlib.lines import Line2D

    fig.add_artist(Line2D([x0 / PAGE_W, x1 / PAGE_W], [1 - y / PAGE_H] * 2,
                          transform=fig.transFigure, color=color, linewidth=width,
                          linestyle=style, zorder=1))


def _pyflic_version() -> str:
    try:
        from importlib.metadata import version

        return version("pyflic")
    except Exception:  # noqa: BLE001
        return ""
