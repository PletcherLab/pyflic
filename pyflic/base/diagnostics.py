"""Stream guards, logging, and crash reporting for frozen builds.

Three problems this solves, all of which only appear once pyflic is packaged as
a windowed executable and none of which are visible during development:

1. **There is no console.**  On Windows a windowed frozen build has
   ``sys.stdout is None`` outright.  There are ~150 bare ``print()`` calls
   across ``pyflic.base``; the ones inside a worker are fine, because
   ``_SignalWriter`` redirects ``sys.stdout`` while it runs, but any that fire
   outside that window raise ``AttributeError: 'NoneType' object has no
   attribute 'write'``.  Guarding the two streams once neutralises the whole
   class without auditing every call site.

2. **Crashes are silent.**  An exception escaping the worker thread closes the
   window with no message and no trace, so the user's report is "it
   disappeared".

3. **There is no update mechanism**, so a bug report is the only channel back.
   It has to carry the version, or it is unactionable.

This module deliberately imports **no Qt at module level**.  ``pyflic --help``,
``pyflic version`` and ``pyflic help`` were made to start in 0.2s rather than
5.8s by keeping the heavy imports lazy (see ADR-0004); importing Qt here would
put a large part of that back.
"""

from __future__ import annotations

import datetime as _dt
import logging
import logging.handlers
import os
import platform
import sys
import traceback
from pathlib import Path

from .utils import config_dir

_LOG_DIR_NAME = "logs"
_LOG_FILE_NAME = "pyflic.log"
_MAX_BYTES = 1_000_000
_BACKUP_COUNT = 3

_installed = False


# ---------------------------------------------------------------------------
# 1. Stream guards
# ---------------------------------------------------------------------------

class _NullWriter:
    """Absorbs writes when there is no console to write to.

    Not ``io.StringIO``: that would grow without bound over a long session,
    holding every line any ``print()`` ever produced.
    """

    def write(self, _text: str) -> int:
        return 0

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        raise OSError("no file descriptor")


def install_stream_guards() -> None:
    """Replace ``sys.stdout``/``sys.stderr`` when the platform gave us none."""
    if sys.stdout is None:
        sys.stdout = _NullWriter()  # type: ignore[assignment]
    if sys.stderr is None:
        sys.stderr = _NullWriter()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 2. Logging
# ---------------------------------------------------------------------------

def log_dir() -> Path:
    return config_dir() / _LOG_DIR_NAME


def log_path() -> Path:
    return log_dir() / _LOG_FILE_NAME


def configure_logging(*, level: int = logging.INFO) -> Path | None:
    """Attach a rotating file handler to the root logger.

    Returns the log path, or ``None`` when the location is unwritable — which
    must never be fatal.  A read-only home directory is unusual but a locked
    down lab machine is exactly where it would happen, and losing logging is a
    far better outcome than refusing to start.
    """
    try:
        log_dir().mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            log_path(), maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
    except OSError:
        return None

    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    ))
    root = logging.getLogger()
    root.setLevel(level)
    # Re-running an entry point in the same process (the tests do) must not
    # stack duplicate handlers writing every line several times over.
    for existing in list(root.handlers):
        if isinstance(existing, logging.handlers.RotatingFileHandler):
            root.removeHandler(existing)
    root.addHandler(handler)
    return log_path()


# ---------------------------------------------------------------------------
# 3. Crash reports
# ---------------------------------------------------------------------------

def environment_report() -> str:
    """Everything needed to identify what the user was running."""
    from pyflic import __version__

    frozen = getattr(sys, "frozen", False)
    lines = [
        f"pyflic       : {__version__}",
        f"frozen       : {bool(frozen)}",
        f"executable   : {sys.executable}",
        f"python       : {platform.python_version()}",
        f"platform     : {platform.platform()}",
        f"machine      : {platform.machine()}",
        f"time (UTC)   : {_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')}",
    ]
    if __version__ == "unknown":
        lines.append(
            "WARNING      : version is 'unknown' — this build lost its metadata "
            "and cannot be identified.  See DEPLOYMENT-PLAN.md 0.2."
        )
    return "\n".join(lines)


def build_report(exc_type: type[BaseException], exc: BaseException,
                 tb: object, *, context: str | None = None) -> str:
    parts = [
        "pyflic crash report",
        "===================",
        "",
        environment_report(),
        "",
    ]
    if context:
        parts += ["context", "-------", context, ""]
    parts += [
        "traceback",
        "---------",
        "".join(traceback.format_exception(exc_type, exc, tb)),  # type: ignore[arg-type]
    ]

    tail = _log_tail()
    if tail:
        parts += ["recent log", "----------", tail]
    return "\n".join(parts)


def _log_tail(max_lines: int = 200) -> str:
    try:
        text = log_path().read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-max_lines:])


def write_report(text: str) -> Path | None:
    """Persist the report next to the log, so it survives a dismissed dialog."""
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    try:
        log_dir().mkdir(parents=True, exist_ok=True)
        path = log_dir() / f"crash-{stamp}.txt"
        path.write_text(text, encoding="utf-8")
        return path
    except OSError:
        return None


def _show_dialog(report: str, saved: Path | None) -> None:
    """Best-effort GUI dialog.  Never raises — it runs while already crashing."""
    try:
        from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

        if QApplication.instance() is None:
            return

        box = QMessageBox()
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("pyflic")
        box.setText("pyflic hit an unexpected error.")
        box.setInformativeText(
            "The analysis you were running has stopped. Your data files have "
            "not been modified.\n\nPlease send the report below to the pyflic "
            "maintainers — it contains the version and the technical details "
            "needed to diagnose this."
            + (f"\n\nA copy was saved to:\n{saved}" if saved else "")
        )
        box.setDetailedText(report)
        save_btn = box.addButton("Save report…", QMessageBox.ButtonRole.ActionRole)
        box.addButton("Close", QMessageBox.ButtonRole.RejectRole)
        box.exec()

        if box.clickedButton() is save_btn:
            target, _ = QFileDialog.getSaveFileName(
                None, "Save pyflic crash report",
                str(Path.home() / (saved.name if saved else "pyflic-crash.txt")),
                "Text files (*.txt)",
            )
            if target:
                Path(target).write_text(report, encoding="utf-8")
    except Exception:  # noqa: BLE001 - a failed dialog must not mask the crash
        pass


def install_excepthook() -> None:
    """Route uncaught exceptions to the log, a saved report, and a dialog."""
    previous = sys.excepthook

    def _hook(exc_type, exc, tb):  # type: ignore[no-untyped-def]
        if issubclass(exc_type, KeyboardInterrupt):
            previous(exc_type, exc, tb)
            return
        report = build_report(exc_type, exc, tb)
        logging.getLogger("pyflic").critical("uncaught exception\n%s", report)
        saved = write_report(report)
        _show_dialog(report, saved)
        previous(exc_type, exc, tb)

    sys.excepthook = _hook

    # Worker threads do not go through sys.excepthook.  The analysis worker
    # already catches broadly, but anything else that spawns a thread would
    # otherwise fail invisibly.
    import threading

    def _thread_hook(args: threading.ExceptHookArgs) -> None:
        if issubclass(args.exc_type, SystemExit):
            return
        report = build_report(
            args.exc_type, args.exc_value, args.exc_traceback,
            context=f"thread: {args.thread.name if args.thread else '?'}",
        )
        logging.getLogger("pyflic").critical("uncaught thread exception\n%s", report)
        write_report(report)

    threading.excepthook = _thread_hook


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def install(*, gui: bool = True) -> None:
    """Install every diagnostic facility.  Safe to call more than once.

    Called from each GUI entry point as well as from :mod:`pyflic.__main__`,
    because the standalone ``pyflic-hub``-style console scripts bypass the
    dispatcher entirely.
    """
    global _installed
    install_stream_guards()          # cheap, and wanted even if already installed
    if _installed:
        return
    _installed = True

    configure_logging()
    logging.getLogger("pyflic").info("start: %s", environment_report().replace("\n", " | "))
    if gui:
        install_excepthook()


def env_summary_for_cli() -> str:
    """Text for ``pyflic version --verbose`` style output."""
    path = log_path()
    return environment_report() + f"\nlog          : {path if path.exists() else '(none yet)'}"


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def selftest() -> tuple[bool, list[str]]:
    """Check that this installation is complete and working.

    Written for the two audiences a frozen, unsigned, never-auto-updating build
    creates:

    * **The build.**  Several packaging defects are silent — the app starts, the
      window opens, and the damage only shows when a user clicks something.
      Missing help content is the worst of them, because the help system
      degrades to a placeholder page by design.  Verifying inside the frozen
      build is the only check that means anything.
    * **Support.**  With no update mechanism and no code signing, "run
      ``pyflic selftest`` and send me the output" is the fastest way to find out
      what a user actually has.

    Returns ``(ok, lines)`` where each line is ``STATUS key=value``.
    """
    lines: list[str] = []
    ok = True

    def record(good: bool, key: str, value: object, note: str = "") -> None:
        nonlocal ok
        if not good:
            ok = False
        status = "ok  " if good else "FAIL"
        lines.append(f"{status} {key}={value}" + (f"  # {note}" if note else ""))

    from pyflic import __version__
    record(__version__ != "unknown", "version", __version__,
           "" if __version__ != "unknown" else "build lost its metadata")
    record(True, "frozen", bool(getattr(sys, "frozen", False)))

    # Help content — package data PyInstaller does not collect by default.
    try:
        from pyflic.help.topics import available
        topics = available()
        record(len(topics) > 0, "help_topics", len(topics),
               "" if topics else "help content missing from the bundle")
    except Exception as exc:  # noqa: BLE001
        record(False, "help_topics", f"error:{exc.__class__.__name__}", str(exc))

    # Lazily-imported public API — invisible to static analysis.
    try:
        from pyflic import load_experiment_yaml  # noqa: F401
        from pyflic import write_experiment_report  # noqa: F401
        record(True, "lazy_api", "resolved")
    except Exception as exc:  # noqa: BLE001
        record(False, "lazy_api", f"error:{exc.__class__.__name__}", str(exc))

    # Plotting stack — lazy submodule imports that fail on first plot, not build.
    for mod in ("matplotlib", "plotnine", "statsmodels.api", "pandas", "numpy"):
        try:
            __import__(mod)
            record(True, mod.replace(".", "_"), "import ok")
        except Exception as exc:  # noqa: BLE001
            record(False, mod.replace(".", "_"), "import failed", str(exc))

    # Icon fonts, loaded by filename rather than imported.
    try:
        import qtawesome
        font_dir = Path(qtawesome.__file__).parent / "fonts"
        fonts = list(font_dir.glob("*.ttf")) if font_dir.is_dir() else []
        record(bool(fonts), "qtawesome_fonts", len(fonts),
               "" if fonts else "icons will be blank")
    except Exception as exc:  # noqa: BLE001
        record(False, "qtawesome_fonts", f"error:{exc.__class__.__name__}", str(exc))

    # Somewhere writable for settings and logs.
    try:
        config_dir().mkdir(parents=True, exist_ok=True)
        probe = config_dir() / ".write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        record(True, "config_dir", config_dir())
    except OSError as exc:
        record(False, "config_dir", config_dir(), str(exc))

    return ok, lines
