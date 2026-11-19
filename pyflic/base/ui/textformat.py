"""Rich rendering for the predominantly-text surfaces (Output log, text tabs).

The Output and Errors logs are wholly in a fixed-width font (the widget's,
:mod:`pyflic.base.ui.fonts`): every line keeps its spacing exactly, so what an
analysis prints as a table lines up and its indentation shows.  The text tabs
give prose the app's proportional font with real headings, and keep the
fixed-width font for TABULAR blocks, where alignment carries meaning.
On-screen numbers show at most four decimal places (0.0000).

Every size in the generated HTML is relative (named sizes, no absolute pt),
so the viewers' font-based zoom keeps scaling everything together.
"""

from __future__ import annotations

import csv
import html
import io
import re

from .fonts import PREFERRED_FIXED_FAMILIES

#: The fixed-width stack for tabular blocks in the text tabs, in the logs'
#: order of preference. Single-quoted: these land inside double-quoted
#: style="" attributes, where inner double quotes would terminate the
#: attribute and silently drop the span.
MONO_FAMILY = ", ".join([f"'{f}'" for f in PREFERRED_FIXED_FAMILIES] + ["monospace"])

#: Columns between tab stops, as a terminal has them: text printed with tabs
#: was laid out against these.
_TAB_STOP = 8

#: Muted gray that reads on both light and dark surfaces.
_MUTED = "#8a8a8a"

# Plain-notation floats with five or more decimals. Scientific notation is
# deliberately excluded: rounding a 1e-8 p-value to 0.0000 would erase it.
_DECIMALS = re.compile(r"(?<![\w.])(-?\d+\.\d{5,})(?![\d.eE])")

_TABULAR = re.compile(r"\S(\t+| {2,})\S")
_RULE = re.compile(r"^[=\-─═_]{3,}\s*$")
_LOG_PREFIX = re.compile(r"^\[([^\]\n]{1,40})\]\s?")
## "Error" alone is a word, not a failure: the DFM's firmware column is named
## Error, and the integrity report names it on a clean recording.  An error
## *message* opens with it ("Error: ...", "ValueError: ...").
_ERRISH = re.compile(r"\b(FAILED|ERROR|Traceback)\b|\b\w*Error:")
_WARNISH = re.compile(r"\b(SKIPPED|WARNING|skipped)\b")


def shorten_decimals(text: str) -> str:
    """Display floats with at most four decimal places (0.0000)."""
    return _DECIMALS.sub(lambda m: f"{float(m.group(1)):.4f}", text)


def _shorten_padded(line: str) -> str:
    """``shorten_decimals`` with each replacement right-padded back to the
    original width, so pre-aligned table columns stay aligned."""
    return _DECIMALS.sub(
        lambda m: f"{float(m.group(1)):.4f}".ljust(len(m.group(1))), line)


def looks_tabular(line: str) -> bool:
    """Interior alignment padding — tabs, or runs of 2+ spaces between
    words — marks a line whose layout carries meaning."""
    return bool(_TABULAR.search(line.strip()))


def _esc(text: str) -> str:
    return html.escape(text, quote=False)


def _fixed(text: str) -> str:
    """Escape *text* for fixed-width display, preserving its exact spacing:
    tabs expanded to their stops, and leading spaces and runs of two or more
    kept rather than collapsed as HTML would.  Single interior spaces stay
    ordinary, so a long line can still wrap there."""
    out = _esc(text.expandtabs(_TAB_STOP))
    return re.sub(r"^ +| {2,}", lambda m: "&nbsp;" * len(m.group(0)), out)


# ---------------------------------------------------------------------------
# Output-log lines
# ---------------------------------------------------------------------------

def log_line_to_html(text: str) -> str:
    """One log line as rich text for a fixed-width log: its spacing kept, a
    muted ``[prefix]``, and error/warning/success accents on prose.

    A line whose spacing does tabular work (a table row, a traceback, a
    block indented four spaces) is shown exactly as printed, with long
    decimals shortened in place so its columns stay aligned."""
    if looks_tabular(text) or text.startswith(("Traceback", '  File "', "    ")):
        return _fixed(_shorten_padded(text))
    line = shorten_decimals(text)
    if line.startswith("── "):
        return f'<b style="font-size:large">{_esc(line[3:].strip())}</b>'

    color = None
    if _ERRISH.search(line):
        color = "#d9534f"
    elif _WARNISH.search(line):
        color = "#c77700"
    elif line.startswith("✓"):
        color = "#2e8b57"

    prefix_html = ""
    body = line
    m = _LOG_PREFIX.match(line)
    if m:
        prefix_html = (f'<span style="color:{_MUTED}">'
                       f"[{_esc(m.group(1))}]</span> ")
        body = line[m.end():]
    inner = _fixed(body)
    if color:
        inner = f'<span style="color:{color}">{inner}</span>'
    return prefix_html + inner


# ---------------------------------------------------------------------------
# Whole text files (artifact tabs)
# ---------------------------------------------------------------------------

def _split_blocks(text: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for raw in text.splitlines():
        if raw.strip():
            current.append(raw.rstrip())
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def _heading_text(block: list[str]) -> str | None:
    if len(block) == 2 and _RULE.match(block[1]) and block[0].strip() \
            and not _RULE.match(block[0]):
        return block[0].strip()
    if len(block) == 1:
        line = block[0].strip()
        if 0 < len(line) <= 60 and line.endswith(":") \
                and not looks_tabular(line):
            return line.rstrip(":")
    return None


def _block_kind(block: list[str]) -> str:
    if all(_RULE.match(line) for line in block):
        return "rule"
    if _heading_text(block) is not None:
        return "heading"
    tabular = sum(1 for line in block if looks_tabular(line))
    if any("\t" in line for line in block) \
            or (len(block) >= 2 and tabular >= max(2, int(0.6 * len(block)))):
        return "table"
    return "prose"


def text_to_html(text: str) -> str:
    parts: list[str] = []
    for block in _split_blocks(text):
        kind = _block_kind(block)
        if kind == "rule":
            parts.append("<hr>")
        elif kind == "heading":
            parts.append(f'<p><b style="font-size:large">'
                         f"{_esc(_heading_text(block))}</b></p>")
        elif kind == "table":
            body = "<br>".join(_fixed(_shorten_padded(line))
                               for line in block)
            parts.append(f'<p style="font-family:{MONO_FAMILY}">{body}</p>')
        else:
            body = "<br>".join(_esc(shorten_decimals(line))
                               for line in block)
            parts.append(f"<p>{body}</p>")
    return "".join(parts) or "<p></p>"


def _format_cell(cell: str) -> tuple[str, bool]:
    """(display text, is_numeric) — numeric cells show ≤4 decimals."""
    raw = cell.strip()
    try:
        value = float(raw)
    except ValueError:
        return raw, False
    if "." in raw and "e" not in raw.lower() \
            and len(raw.split(".", 1)[1]) > 4:
        return f"{value:.4f}", True
    return raw, True


def csv_to_html(text: str, max_rows: int = 1500) -> str:
    try:
        dialect = csv.Sniffer().sniff(text[:2048], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.reader(io.StringIO(text), dialect))
    if not rows:
        return "<p>(empty file)</p>"
    header, body = rows[0], rows[1:]
    parts = ['<table cellspacing="0" cellpadding="4" '
             'style="border-collapse:collapse">']
    parts.append("<tr>" + "".join(
        f'<th style="border-bottom:1px solid {_MUTED}; '
        f'text-align:left; padding:4px 10px 4px 4px">{_esc(h.strip())}</th>'
        for h in header) + "</tr>")
    for row in body[:max_rows]:
        cells = []
        for cell in row:
            shown, numeric = _format_cell(cell)
            align = "right" if numeric else "left"
            cells.append(f'<td align="{align}" '
                         f'style="padding:2px 10px 2px 4px">'
                         f"{_esc(shown)}</td>")
        parts.append("<tr>" + "".join(cells) + "</tr>")
    parts.append("</table>")
    if len(body) > max_rows:
        parts.append(f'<p style="color:{_MUTED}">… {len(body) - max_rows} '
                     "more row(s) — open the file for the full data.</p>")
    return "".join(parts)


def file_to_html(path, text: str) -> str:
    """Render a text artifact for display: CSVs as a real table, everything
    else through the block classifier."""
    suffix = str(path).lower().rsplit(".", 1)
    if len(suffix) == 2 and suffix[1] == "csv":
        return csv_to_html(text)
    return text_to_html(text)
