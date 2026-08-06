"""
Unified ``pyflic`` CLI entry.

Dispatches to subcommands::

    pyflic config [project_dir]   -- launch the config editor GUI
    pyflic qc <project_dir>       -- launch the QC viewer
    pyflic hub  [project_dir]     -- launch the analysis hub GUI
    pyflic help [topic]           -- open the help window
    pyflic lint <project_or_yaml> -- schema-lint a flic_config.yaml
    pyflic clear-cache <project>  -- remove project_dir/.pyflic_cache
    pyflic report <project_dir>   -- write a PDF experiment report
    pyflic version                -- print the installed version

Use ``pyflic --help`` (or ``-h``) for this list; ``pyflic help`` opens the
graphical help.

Existing entry points (``pyflic-config``, ``pyflic-qc``, ``pyflic-hub``)
remain available.
"""

from __future__ import annotations

import sys
from pathlib import Path


_COMMANDS = ("config", "qc", "hub", "lint", "clear-cache", "report", "version", "help")


def _print_help() -> None:
    print(__doc__ or "pyflic CLI")


def _launch_help(topic: str | None) -> None:
    """Open the help window as a standalone application."""
    from PyQt6.QtWidgets import QApplication

    from pyflic.base.ui import apply_theme
    from pyflic.base.ui import settings as ui_settings
    from pyflic.help import open_help

    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app, mode=ui_settings.get("theme", "auto"))
    win = open_help(topic)
    if win is None:
        print("could not open the help window", file=sys.stderr)
        raise SystemExit(1)
    app.exec()


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        _print_help()
        return
    cmd, *rest = argv

    if cmd == "help":
        _launch_help(rest[0] if rest else None)
        return

    if cmd == "version":
        from pyflic import __version__
        print(__version__)
        return

    if cmd == "config":
        from pyflic.base.config_editor import launch
        sys.argv = ["pyflic-config", *rest]
        launch()
        return

    if cmd == "qc":
        from pyflic.base.qc_viewer import main as qc_main
        sys.argv = ["pyflic-qc", *rest]
        qc_main()
        return

    if cmd == "hub":
        from pyflic.base.analysis_hub import main as hub_main
        sys.argv = ["pyflic-hub", *rest]
        hub_main()
        return

    if cmd == "lint":
        from pyflic.base.yaml_lint import main_cli
        sys.argv = ["pyflic-lint", *rest]
        main_cli()
        return

    if cmd == "clear-cache":
        from pyflic.base import cache as _cache
        if not rest:
            print("usage: pyflic clear-cache <project_dir>", file=sys.stderr)
            raise SystemExit(2)
        n = _cache.clear(Path(rest[0]))
        print(f"removed {n} cache file(s) from {rest[0]}")
        return

    if cmd == "report":
        if not rest:
            print("usage: pyflic report <project_dir>", file=sys.stderr)
            raise SystemExit(2)
        from pyflic import load_experiment_yaml
        from pyflic.base.pdf_report import write_experiment_report
        exp = load_experiment_yaml(rest[0])
        out = write_experiment_report(exp)
        print(f"wrote {out}")
        return

    print(f"unknown command: {cmd!r}\n", file=sys.stderr)
    _print_help()
    raise SystemExit(2)


if __name__ == "__main__":
    main()
