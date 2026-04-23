"""Centralised icon factory.

Wraps ``qtawesome`` so glyph names live in one file and tinting follows
the active theme + category color.  Use ``icon("load")`` for a category
icon, or ``icon("fa5s.folder")`` for an explicit qtawesome name.
"""

from __future__ import annotations

import qtawesome as qta
from PyQt6.QtGui import QIcon

from .theme import Category, category_color, resolved_mode

# Logical name → (qtawesome glyph, default Category for tinting).
_GLYPHS: dict[str, tuple[str, Category | None]] = {
    # Navigation / sidebar
    "home":       ("fa5s.home",                Category.NEUTRAL),
    "project":    ("fa5s.folder-open",         Category.NEUTRAL),
    "run":        ("fa5s.bolt",                Category.LOAD),
    "plots":      ("fa5s.chart-bar",           Category.PLOTS),
    "qc":         ("fa5s.search",              Category.QC),
    "scripts":    ("fa5s.scroll",              Category.SCRIPTS),
    "tools":      ("fa5s.tools",               Category.TOOLS),
    "settings":   ("fa5s.cog",                 Category.NEUTRAL),
    "theme_dark": ("fa5s.moon",                Category.NEUTRAL),
    "theme_light":("fa5s.sun",                 Category.NEUTRAL),
    # Actions
    "load":       ("fa5s.download",            Category.LOAD),
    "script":     ("fa5s.play",                Category.SCRIPTS),
    "basic":      ("fa5s.chart-area",          Category.ANALYZE),
    "csv":        ("fa5s.file-csv",            Category.ANALYZE),
    "binned":     ("fa5s.stream",              Category.ANALYZE),
    "weighted":   ("fa5s.balance-scale",       Category.ANALYZE),
    "tidy":       ("fa5s.table",               Category.ANALYZE),
    "bootstrap":  ("fa5s.dice",                Category.ANALYZE),
    "compare":    ("fa5s.code-branch",         Category.ANALYZE),
    "lightphase": ("fa5s.adjust",              Category.ANALYZE),
    "sensitivity":("fa5s.sliders-h",           Category.ANALYZE),
    "transition": ("fa5s.exchange-alt",        Category.ANALYZE),
    "pdf":        ("fa5s.file-pdf",            Category.ANALYZE),
    "plot":       ("fa5s.chart-line",          Category.PLOTS),
    "feeding":    ("fa5s.utensils",            Category.PLOTS),
    "dot":        ("fa5s.braille",             Category.PLOTS),
    "well":       ("fa5s.vials",               Category.PLOTS),
    "lint":       ("fa5s.spell-check",         Category.TOOLS),
    "compare_cfg":("fa5s.exchange-alt",        Category.TOOLS),
    "clear":      ("fa5s.trash-alt",           Category.TOOLS),
    "config":     ("fa5s.sliders-h",           Category.TOOLS),
    # File menu
    "open":       ("fa5s.folder-open",         Category.NEUTRAL),
    "save":       ("fa5s.save",                Category.LOAD),
    "save_as":    ("fa5s.file-export",         Category.LOAD),
    "new":        ("fa5s.file",                Category.NEUTRAL),
    # Misc
    "warning":    ("fa5s.exclamation-triangle",Category.QC),
    "info":       ("fa5s.info-circle",         Category.NEUTRAL),
    "play":       ("fa5s.play",                Category.LOAD),
    "stop":       ("fa5s.stop",                Category.QC),
    "browse":     ("fa5s.ellipsis-h",          Category.NEUTRAL),
}


def _tint_for(category: Category | None) -> str:
    if category is None:
        # Default tint = a neutral foreground that reads on either theme.
        return "#cbd5e1" if resolved_mode() == "dark" else "#334155"
    return category_color(category)


def icon(name: str, category: Category | None = None) -> QIcon:
    """Return a themed QIcon for *name*.

    *name* is either a logical key (``"load"``) or an explicit qtawesome
    glyph (``"fa5s.folder-open"``).  *category* overrides the default
    tint registered for that key.
    """
    if name in _GLYPHS:
        glyph, default_category = _GLYPHS[name]
    else:
        glyph, default_category = name, None
    color = _tint_for(category if category is not None else default_category)
    return qta.icon(glyph, color=color)
