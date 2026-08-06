"""In-app help for pyflic.

Help topics are markdown files in :mod:`pyflic.help.content`, shipped with the
package so they are available to every install — including one made with
``pip install`` where no repository checkout exists.

This package depends on the shared UI primitives in ``pyflic.base.ui`` and on
nothing else in pyflic.  In particular it never imports ``analysis_hub``,
``qc_viewer``, ``config_editor``, ``script_editor``, or any analysis module, so
it can be updated or replaced without touching the analysis pipeline.  See
``docs/adr/0004-help-rendering-and-reference-validation.md``.

Typical use from a GUI::

    from pyflic.help import HelpButton, install_help_shortcut

    card.add_title_widget(HelpButton("app-hub#load"))
    install_help_shortcut(self, "app-hub")

The Qt-dependent names are imported lazily so that ``import pyflic.help`` from a
headless context (tests, CLI parsing) does not require a display.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .toc import (
    GUIDES,
    TOC,
    Guide,
    Section,
    all_topic_ids,
    guide,
    guide_neighbours,
    neighbours,
    section_for,
)
from .topics import Topic, available, load, parse_ref, search, slugify

if TYPE_CHECKING:  # pragma: no cover
    from .button import HelpButton, install_help_shortcut
    from .window import HelpWindow, help_window, open_help

_LAZY: dict[str, str] = {
    "HelpButton": ".button",
    "install_help_shortcut": ".button",
    "HelpWindow": ".window",
    "help_window": ".window",
    "open_help": ".window",
}

__all__ = [
    "GUIDES",
    "TOC",
    "Guide",
    "HelpButton",
    "HelpWindow",
    "Section",
    "Topic",
    "all_topic_ids",
    "available",
    "guide",
    "guide_neighbours",
    "help_window",
    "install_help_shortcut",
    "load",
    "neighbours",
    "open_help",
    "parse_ref",
    "search",
    "section_for",
    "slugify",
]


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module, __name__), name)
