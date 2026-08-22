"""
Unified ``pyflic`` CLI entry.

Run with no arguments to launch the analysis hub::

    pyflic                        -- launch the analysis hub GUI

Or dispatch to a subcommand::

    pyflic config [dir]        -- launch the config editor GUI
    pyflic qc <dir>            -- launch the QC viewer
    pyflic hub [dir]           -- launch the analysis hub GUI
    pyflic plots <project>     -- launch the Plot Editor (project level)
    pyflic help [topic]        -- open the help window
    pyflic lint <dir>          -- validate configs and report migrations
    pyflic clear-cache <dir>   -- remove <dir>/.pyflic_cache
    pyflic report <dir>        -- write a PDF report
    pyflic batch <dir>         -- run the designated Project Script in
                                  every Project under <dir>
    pyflic version             -- print the installed version

Commands taking a directory decide what to do from its marker file: a
``project.yaml`` means Project, a ``flic_config.yaml`` means Experiment
Directory. No command needs a level flag.

Use ``pyflic --help`` (or ``-h``) for this list; ``pyflic help`` opens the
graphical help.

Existing entry points (``pyflic-config``, ``pyflic-qc``, ``pyflic-hub``)
remain available.
"""

from __future__ import annotations

import sys
from pathlib import Path


_COMMANDS = ("config", "qc", "hub", "plots", "lint", "clear-cache", "report",
             "batch", "version", "help")


def _classify(path) -> str:
    """What *path* is: ``"project"``, ``"experiment"``, ``"batch"``, or ``""``.

    The marker file is the whole test (ADR-0005/0006), which is why no
    level-aware command needs a flag.
    """
    from pyflic.base import batch as _batch
    from pyflic.base import project as _project

    if _project.is_project_dir(path):
        return "project"
    if _project.is_experiment_dir(path):
        return "experiment"
    if _batch.is_batch_dir(path):
        return "batch"
    return ""


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
    argv = sys.argv[1:]
    if argv and argv[0] in ("-h", "--help"):
        _print_help()
        return
    if not argv:
        # Bare ``pyflic`` launches the analysis hub — the same thing
        # ``pyflic-hub`` does.  Subcommands below are unaffected, and
        # ``pyflic --help`` still prints the command list.
        from pyflic.base.hub import main as hub_main
        sys.argv = ["pyflic-hub"]
        hub_main()
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
        from pyflic.base.hub import main as hub_main
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
            print("usage: pyflic clear-cache <experiment_dir>", file=sys.stderr)
            raise SystemExit(2)
        n = _cache.clear(Path(rest[0]))
        print(f"removed {n} cache file(s) from {rest[0]}")
        return

    if cmd == "plots":
        from pyflic.base.plot_editor import main as plots_main
        sys.argv = ["pyflic-plots", *rest]
        plots_main()
        return

    if cmd == "batch":
        if not rest:
            print("usage: pyflic batch <dir> [script]", file=sys.stderr)
            raise SystemExit(2)
        from pyflic.base.batch import Batch
        b = Batch(rest[0])
        if not len(b):
            print(f"no Projects directly under {rest[0]} "
                  f"(a Batch scans immediate children only)", file=sys.stderr)
            raise SystemExit(1)
        summary = b.run(rest[1] if len(rest) > 1 else None)
        raise SystemExit(1 if summary["failed"] else 0)

    if cmd == "report":
        if not rest:
            print("usage: pyflic report <dir>", file=sys.stderr)
            raise SystemExit(2)
        target = rest[0]
        kind = _classify(target)
        if kind == "project":
            from pyflic.base.project import Project
            from pyflic.base.project_report import write_project_report
            out = write_project_report(Project(target))
        elif kind == "experiment":
            from pyflic import load_experiment_yaml
            from pyflic.base.pdf_report import write_experiment_report
            out = write_experiment_report(load_experiment_yaml(target))
        else:
            print(f"{target!r} is neither a Project (project.yaml) nor an "
                  f"Experiment Directory (flic_config.yaml).", file=sys.stderr)
            raise SystemExit(2)
        print(f"wrote {out}")
        return

    print(f"unknown command: {cmd!r}\n", file=sys.stderr)
    _print_help()
    raise SystemExit(2)


if __name__ == "__main__":
    main()
