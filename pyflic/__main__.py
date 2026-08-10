"""
Unified ``pyflic`` CLI entry.

Run with no arguments to launch the analysis hub::

    pyflic                        -- launch the analysis hub GUI

Or dispatch to a subcommand::

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


_COMMANDS = ("config", "qc", "hub", "lint", "clear-cache", "report", "version",
             "help", "selftest")


def _print_help() -> None:
    print(__doc__ or "pyflic CLI")


_NO_DISPLAY_HINT = (
    "pyflic help needs a graphical display.\n"
    "On a headless machine, read the topics as plain text instead:\n"
    "  python -c \"from pyflic.help import load; print(load('getting-started').source)\"\n"
    "Topic names are listed by:\n"
    "  python -c \"from pyflic.help import all_topic_ids; print(*all_topic_ids(), sep=chr(10))\""
)


def _launch_help(topic: str | None) -> None:
    """Open the help window as a standalone application.

    Note that a missing or unusable Qt platform plugin aborts inside Qt
    itself (``qFatal``) before Python regains control, so that particular
    failure cannot be turned into a friendly message here.  Everything that
    *is* catchable gets one.
    """
    try:
        from PyQt6.QtWidgets import QApplication
    except Exception as exc:  # noqa: BLE001
        print(f"could not load the Qt GUI toolkit: {exc}", file=sys.stderr)
        print(_NO_DISPLAY_HINT, file=sys.stderr)
        raise SystemExit(1) from exc

    from pyflic.base.ui import apply_theme
    from pyflic.base.ui import settings as ui_settings
    from pyflic.help import available, open_help

    if topic is not None and topic not in available():
        print(f"unknown help topic: {topic!r}", file=sys.stderr)
        print("available topics:", file=sys.stderr)
        for name in available():
            print(f"  {name}", file=sys.stderr)
        raise SystemExit(2)

    try:
        app = QApplication.instance() or QApplication(sys.argv)
    except Exception as exc:  # noqa: BLE001
        print(f"could not start a graphical session: {exc}", file=sys.stderr)
        print(_NO_DISPLAY_HINT, file=sys.stderr)
        raise SystemExit(1) from exc

    apply_theme(app, mode=ui_settings.get("theme", "auto"))
    win = open_help(topic)
    if win is None:
        print("could not open the help window", file=sys.stderr)
        print(_NO_DISPLAY_HINT, file=sys.stderr)
        raise SystemExit(1)
    app.exec()


def main() -> None:
    # Before anything else: a windowed frozen build has no console, and on
    # Windows ``sys.stdout`` is ``None``, so the first stray ``print()``
    # anywhere below would raise.  Cheap, and imports no Qt.
    from pyflic.base.diagnostics import install as _install_diagnostics
    _install_diagnostics(gui=True)

    argv = sys.argv[1:]
    if argv and argv[0] in ("-h", "--help"):
        _print_help()
        return
    if not argv:
        # Bare ``pyflic`` launches the analysis hub — the same thing
        # ``pyflic-hub`` does.  Subcommands below are unaffected, and
        # ``pyflic --help`` still prints the command list.
        from pyflic.base.analysis_hub import main as hub_main
        sys.argv = ["pyflic-hub"]
        hub_main()
        return
    cmd, *rest = argv

    if cmd == "help":
        _launch_help(rest[0] if rest else None)
        return

    if cmd == "selftest":
        from pyflic.base.diagnostics import selftest
        ok, lines = selftest()
        for line in lines:
            print(line)
        print("\nPASS" if ok else "\nFAIL — send this output to the maintainers")
        raise SystemExit(0 if ok else 1)

    if cmd == "version":
        from pyflic import __version__
        if rest and rest[0] in ("-v", "--verbose"):
            from pyflic.base.diagnostics import env_summary_for_cli
            print(env_summary_for_cli())
        else:
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
