"""Every test runs against a throwaway pyflic settings folder, and every Qt
test in one QApplication.

The apps save preferences to ``~/.config/pyflic/ui.json``: the theme a
toggle chooses, the help window's geometry, the lab's type library folder.
A test that toggles a theme or closes the help window would otherwise
rewrite the user's own file (a theme-toggle test once left a developer's
pyflic in light mode).  The type library's default folder lives there too,
so isolating it also keeps a lab's real types out of the tests' type lists.

Each Qt test module's ``app`` fixture returns ``QApplication.instance()`` or
a new one.  When the module ended, its fixture dropped the last reference,
PyQt destroyed the application while that module's widgets still existed,
and the next Qt module made another.  Widgets outliving their application is
undefined behaviour, and is the likeliest cause of the intermittent
segmentation fault in ``test_hub_ribbon_ui``'s ``hub`` fixture a quarter of
the way through a full run (the module passes alone).  The session holds one
application from the first test to the last.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _one_qapplication():
    try:
        import pyflic.base  # noqa: F401  (matplotlib before Qt: see its docstring)
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        yield None
        return
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    yield QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path_factory, monkeypatch):
    ## Every reader of the settings folder asks
    ## pyflic.base.utils.user_config_dir on each call, so the variable is
    ## enough; it must be absolute, or the folder is ~/.config/pyflic again.
    folder = tmp_path_factory.mktemp("config")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(folder.resolve()))
    monkeypatch.delenv("PYFLIC_TYPE_LIBRARY", raising=False)
