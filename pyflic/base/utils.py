from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable, Sequence


def config_dir() -> Path:
    """Per-user directory for pyflic's own state.

    Deliberately the same root for settings and logs.  Nothing here is written
    inside the installation directory: a frozen bundle is effectively read-only
    and the installer requires no administrator rights, so it may land
    somewhere the user cannot write to.
    """
    root = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(root) / "pyflic"


def resolve_app_command(name: str, module: str, subcommand: str) -> list[str]:
    """Command list that launches one of pyflic's GUI apps as a subprocess.

    Three deployments have to work, and they need three different answers:

    * **Frozen bundle.**  ``sys.executable`` is the pyflic binary itself rather
      than a Python interpreter, and no console scripts exist anywhere on
      ``PATH``.  Both of the other branches therefore produce something that
      silently does nothing — ``pyflic.exe -m pyflic.base.config_editor`` is
      parsed by :mod:`pyflic.__main__` as the unknown subcommand ``-m``.  The
      binary dispatches on ``argv[1]``, so re-exec it with *subcommand*.
    * **Installed package.**  pip generated a ``pyflic-config``-style console
      script from ``[project.scripts]``; prefer it, so the child process picks
      up the same environment the parent was launched from.
    * **Source checkout.**  No console script exists; fall back to ``-m``.
    """
    if getattr(sys, "frozen", False):
        return [sys.executable, subcommand]
    exe = shutil.which(name)
    if exe:
        return [exe]
    return [sys.executable, "-m", module]


_nsre = re.compile(r"(\d+)")


def natural_key(s: str) -> list[object]:
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]


def natural_sorted(paths: Iterable[Path]) -> list[Path]:
    return sorted(list(paths), key=lambda p: natural_key(p.name))


def range_is_specified(rng: Sequence[float]) -> bool:
    return len(rng) == 2 and (float(rng[0]) + float(rng[1]) != 0.0)

