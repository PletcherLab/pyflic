"""The fixed-width font for text whose spacing carries meaning.

The Output and Errors logs show what analyses print: tables padded with
spaces, indented steps, tracebacks.  They need a font whose every character is
one width, and a font named but not installed must never fall back to a
proportional one — which is what a stylesheet's ``"JetBrains Mono", …,
monospace`` did wherever JetBrains Mono was missing.  So the family is chosen
here, from the fonts this machine actually has.
"""

from __future__ import annotations

from functools import lru_cache

from PyQt6.QtGui import QFont, QFontDatabase

#: Most preferred first: well-hinted coding fonts with distinct 0/O and 1/l/I,
#: then each platform's own.  The first one installed is used.
PREFERRED_FIXED_FAMILIES = (
    "JetBrains Mono",
    "Cascadia Mono",        # Windows 11
    "SF Mono",              # macOS, where it is exposed
    "IBM Plex Mono",
    "Fira Mono",
    "Fira Code",
    "Source Code Pro",
    "Menlo",                # macOS
    "Consolas",             # Windows
    "Ubuntu Sans Mono",
    "Ubuntu Mono",
    "Noto Sans Mono",
    "DejaVu Sans Mono",
    "Liberation Mono",
)

#: The log's size, in points, before any zoom.
LOG_POINT_SIZE = 10


def _bare(family: str) -> str:
    """``"IBM Plex Mono [IBM]"`` → ``"ibm plex mono"``: Qt names a family
    by its foundry too when two foundries ship it."""
    return family.split(" [", 1)[0].strip().casefold()


@lru_cache(maxsize=1)
def fixed_family() -> str:
    """The first of :data:`PREFERRED_FIXED_FAMILIES` installed here, else the
    system's own fixed font.  Needs a running ``QGuiApplication``."""
    installed: dict[str, str] = {}
    for family in QFontDatabase.families():
        installed.setdefault(_bare(family), family)
    for wanted in PREFERRED_FIXED_FAMILIES:
        found = installed.get(wanted.casefold())
        if found is not None and QFontDatabase.isFixedPitch(found):
            return found
    return QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont).family()


def fixed_font(point_size: int = LOG_POINT_SIZE) -> QFont:
    """:func:`fixed_family` at *point_size*, hinted as monospace."""
    font = QFont(fixed_family(), point_size)
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setFixedPitch(True)
    return font
