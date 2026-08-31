"""Shared theming for pyflic Qt apps.

A single ``apply_theme()`` is the entry point.  Categories define the
semantic color of an action and are consumed by ``icons.py`` and
``widgets.py`` to tint icons, button borders, and section headers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import qdarktheme
from PyQt6.QtWidgets import QApplication

ThemeMode = Literal["light", "dark", "auto"]


class Category(str, Enum):
    LOAD = "load"
    ANALYZE = "analyze"
    PLOTS = "plots"
    QC = "qc"
    SCRIPTS = "scripts"
    AI = "ai"
    TOOLS = "tools"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class CategoryColors:
    light: str
    dark: str

    def for_mode(self, mode: ThemeMode) -> str:
        return self.dark if mode == "dark" else self.light


PALETTE: dict[Category, CategoryColors] = {
    Category.LOAD:    CategoryColors("#2563eb", "#3b82f6"),
    Category.ANALYZE: CategoryColors("#16a34a", "#22c55e"),
    Category.PLOTS:   CategoryColors("#ea580c", "#fb923c"),
    Category.QC:      CategoryColors("#dc2626", "#f87171"),
    Category.SCRIPTS: CategoryColors("#9333ea", "#a855f7"),
    Category.AI:      CategoryColors("#0d9488", "#2dd4bf"),
    Category.TOOLS:   CategoryColors("#475569", "#94a3b8"),
    Category.NEUTRAL: CategoryColors("#64748b", "#94a3b8"),
}

_current_mode: ThemeMode = "auto"
_resolved_mode: Literal["light", "dark"] = "light"


def current_mode() -> ThemeMode:
    return _current_mode


def resolved_mode() -> Literal["light", "dark"]:
    """The actually-applied light/dark mode (``"auto"`` resolves via OS)."""
    return _resolved_mode


def category_color(category: Category, mode: ThemeMode | None = None) -> str:
    use_mode: ThemeMode = mode or _resolved_mode
    if use_mode == "auto":
        use_mode = _resolved_mode
    return PALETTE[category].for_mode(use_mode)


#: Chrome surfaces for the tile strip, tiles, and anchored panels.  Kept here
#: rather than derived from the Qt palette so the strip reads as one deliberate
#: band in both themes instead of inheriting whatever qdarktheme picks.
## Values mirror PyTrackingAnalysis's ui/theme.py so the two Hubs read as one
## family of apps.
_SURFACES: dict[str, dict[str, str]] = {
    "light": {
        "band":   "#f4f5f7",
        "base":   "#ffffff",
        "hover":  "#e4e7ea",
        "border": "#c4c8cc",
        "text":   "#0f172a",
        "muted":  "#64748b",
    },
    "dark": {
        "band":   "#26292d",
        "base":   "#1f2226",
        "hover":  "#33383e",
        "border": "#3f444b",
        "text":   "#e1e5e9",
        "muted":  "#8b949e",
    },
}


def surface_colors(mode: ThemeMode | None = None) -> dict[str, str]:
    """The chrome surface colors for *mode* (default: the applied mode)."""
    use_mode: ThemeMode = mode or _resolved_mode
    if use_mode == "auto":
        use_mode = _resolved_mode
    return dict(_SURFACES["dark" if use_mode == "dark" else "light"])


def blocked_color() -> str:
    """Red for a Blocked Member, legible on both themes.

    The light theme's red is unreadable on the dark surface and vice versa, so
    this is a pair rather than a constant.  One definition, because the Batch
    table, the Project table, and the preflight tree all paint the same fact
    and must not drift apart.
    """
    return "#f87171" if _resolved_mode == "dark" else "#b91c1c"


def _resolve_auto() -> Literal["light", "dark"]:
    try:
        import darkdetect

        ans = (darkdetect.theme() or "Light").strip().lower()
        return "dark" if ans == "dark" else "light"
    except Exception:  # noqa: BLE001
        return "light"


def _additional_qss() -> str:
    """QSS appended to qdarktheme's stylesheet for pyflic-specific widgets."""
    return """
    QPushButton#PyflicSidebarItem {
        text-align: left;
        padding: 8px 12px;
        border: none;
        border-radius: 6px;
        font-weight: 500;
    }
    QPushButton#PyflicSidebarItem:hover {
        background: palette(midlight);
    }
    QPushButton#PyflicSidebarItem:checked {
        background: palette(highlight);
        color: palette(highlighted-text);
    }
    QFrame#PyflicCard {
        border-radius: 10px;
        background: palette(base);
    }
    QLabel#PyflicCardTitle {
        font-size: 13pt;
        font-weight: 600;
        padding-bottom: 2px;
    }
    QLabel#PyflicCardSubtitle {
        color: palette(mid);
        font-size: 9pt;
    }
    QLabel#PyflicSectionDivider {
        color: palette(mid);
        font-size: 10px;
        padding-top: 6px;
    }
    QFrame#PyflicTopBar {
        background: palette(base);
        border-bottom: 1px solid palette(midlight);
    }
    QLabel#PyflicAppTitle {
        font-size: 14pt;
        font-weight: 600;
    }
    QPlainTextEdit#PyflicLog {
        font-family: "JetBrains Mono", "Menlo", "Consolas", monospace;
        font-size: 10pt;
    }
    """


def apply_theme(app: QApplication, mode: ThemeMode = "auto") -> None:
    """Apply the chosen theme to *app* and remember it."""
    global _current_mode, _resolved_mode
    _current_mode = mode
    _resolved_mode = _resolve_auto() if mode == "auto" else mode  # type: ignore[assignment]
    qdarktheme.setup_theme(mode, additional_qss=_additional_qss())
