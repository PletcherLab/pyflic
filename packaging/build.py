#!/usr/bin/env python
"""Build the pyflic desktop bundle for the current platform.

    python packaging/build.py [--clean] [--skip-smoke]

Produces ``dist/pyflic/`` (Windows, Linux) or ``dist/pyflic.app`` (macOS).
Wrapping that folder in an installer is a separate step — see
``packaging/pyflic.iss`` for Windows.

The smoke check at the end is not optional politeness.  Several of the failure
modes this build guards against are **silent**: the app starts, the window
opens, and the damage only shows when a user clicks something.  Verifying in the
frozen build is the entire point; verifying in the source tree proves nothing,
because there every file obviously exists.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "packaging" / "pyflic.spec"
DIST = ROOT / "dist"
BUILD = ROOT / "build"
VERSION_FILE = ROOT / "pyflic" / "_version.py"


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=True, **kw)


# ---------------------------------------------------------------------------


def resolve_version() -> str:
    """Version from installed metadata, falling back to git."""
    try:
        from importlib.metadata import version
        return version("pyflic")
    except Exception:  # noqa: BLE001
        pass
    try:
        out = subprocess.run(
            ["git", "describe", "--tags", "--dirty", "--always"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip().lstrip("v")
    except Exception:  # noqa: BLE001
        return "0.0.0+unknown"


def write_version_file(version: str) -> None:
    """Bake the version in, so the bundle never reports 'unknown'.

    There is no update mechanism, so the version a user reads back to you is the
    only thing identifying what they are running.  ``copy_metadata`` in the spec
    covers this too; having both means a change to either one cannot silently
    break bug reports.
    """
    VERSION_FILE.write_text(
        '"""Written by packaging/build.py. Not tracked in git."""\n\n'
        f'__version__ = "{version}"\n',
        encoding="utf-8",
    )
    print(f"wrote {VERSION_FILE.relative_to(ROOT)} -> {version}")


# ---------------------------------------------------------------------------


def generate_icon() -> None:
    """Render an application icon from qtawesome.

    pyflic has no icon assets — every icon in the UI is a qtawesome glyph
    rendered at runtime — so rather than committing a binary that drifts from
    the UI, draw the same glyph the hub uses. Failure is non-fatal: PyInstaller
    falls back to its default icon and the build still produces a working app.
    """
    assets = ROOT / "packaging" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    target = assets / ("pyflic.ico" if sys.platform == "win32" else "pyflic.icns")
    if target.exists():
        print(f"icon present: {target.name}")
        return

    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import qtawesome as qta
        from PyQt6.QtCore import QSize, Qt
        from PyQt6.QtGui import QColor, QPainter, QPixmap
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        sizes = [256, 128, 64, 48, 32, 16]
        pngs = []
        for size in sizes:
            canvas = QPixmap(QSize(size, size))
            canvas.fill(QColor("#1f6feb"))
            glyph = qta.icon("fa5s.bolt", color="white").pixmap(
                QSize(int(size * 0.62), int(size * 0.62))
            )
            painter = QPainter(canvas)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawPixmap(
                (size - glyph.width()) // 2, (size - glyph.height()) // 2, glyph
            )
            painter.end()
            png = assets / f"_icon_{size}.png"
            canvas.save(str(png), "PNG")
            pngs.append(png)
        del app

        from PIL import Image

        images = [Image.open(p) for p in pngs]
        if sys.platform == "win32":
            images[0].save(target, format="ICO",
                           sizes=[(s, s) for s in sizes])
        else:
            images[0].save(target.with_suffix(".png"), format="PNG")
            if sys.platform == "darwin":
                _icns_from_pngs(assets, target)
        for p in pngs:
            p.unlink(missing_ok=True)
        print(f"generated {target.name}")
    except Exception as exc:  # noqa: BLE001
        print(f"icon generation skipped ({exc}); using PyInstaller default")


def _icns_from_pngs(assets: Path, target: Path) -> None:
    """macOS only: iconutil turns an .iconset directory into an .icns."""
    iconset = assets / "pyflic.iconset"
    iconset.mkdir(exist_ok=True)
    src = target.with_suffix(".png")
    from PIL import Image
    base = Image.open(src)
    for size in (16, 32, 128, 256, 512):
        base.resize((size, size)).save(iconset / f"icon_{size}x{size}.png")
        base.resize((size * 2, size * 2)).save(iconset / f"icon_{size}x{size}@2x.png")
    _run(["iconutil", "-c", "icns", str(iconset), "-o", str(target)])
    shutil.rmtree(iconset, ignore_errors=True)


# ---------------------------------------------------------------------------


def freeze(clean: bool) -> None:
    cmd = [sys.executable, "-m", "PyInstaller", str(SPEC), "--noconfirm"]
    if clean:
        cmd.append("--clean")
    _run(cmd, cwd=ROOT)


def adhoc_sign_macos() -> None:
    """Ad-hoc sign the .app.  Free, no Apple account, and mandatory on arm64.

    An entirely unsigned arm64 binary will not execute at all on Apple Silicon.
    Ad-hoc signing also downgrades the alarming "pyflic is damaged and can't be
    opened. You should move it to the Trash." to the milder unidentified-
    developer path, which the download page walks users through.

    This is *not* a substitute for notarisation, which was deliberately declined
    on cost — see ADR-0002.
    """
    if sys.platform != "darwin":
        return
    app = DIST / "pyflic.app"
    if not app.exists():
        print("no .app to sign")
        return
    _run(["codesign", "--force", "--deep", "--sign", "-", str(app)])
    _run(["codesign", "--verify", "--verbose=2", str(app)])


# ---------------------------------------------------------------------------


def frozen_executable() -> Path:
    if sys.platform == "darwin":
        return DIST / "pyflic.app" / "Contents" / "MacOS" / "pyflic"
    name = "pyflic.exe" if sys.platform == "win32" else "pyflic"
    return DIST / "pyflic" / name


def smoke_check(expected_version: str) -> bool:
    """Verify the frozen build, not the source tree.

    Covers exactly the Phase 0 defects that produce a bundle which builds
    cleanly and fails on someone else's machine.
    """
    exe = frozen_executable()
    if not exe.exists():
        print(f"FAIL: no executable at {exe}")
        return False

    # `pyflic selftest` runs the checks from inside the bundle, which is the
    # only place they mean anything.
    try:
        out = subprocess.run([str(exe), "selftest"], capture_output=True,
                             text=True, timeout=300)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: could not run `pyflic selftest`: {exc}")
        return False

    print(out.stdout.rstrip() or "(no output)")
    if out.stderr.strip():
        print("--- stderr ---")
        print(out.stderr.rstrip())

    ok = out.returncode == 0

    # The bundle reports how many topics it can see; compare that against the
    # source tree so the check cannot rot as topics are added or removed.
    expected = len(list((ROOT / "pyflic" / "help" / "content").glob("*.md")))
    found = _parse_kv(out.stdout, "help_topics")
    if found is None:
        print("FAIL: selftest did not report help_topics")
        ok = False
    elif found != expected:
        print(f"FAIL 0.5: bundle sees {found} help topics, source has {expected}"
              " — help content was not fully collected")
        ok = False
    else:
        print(f"  ok: help topic count matches source ({expected})")

    return ok


def _parse_kv(stdout: str, key: str) -> int | None:
    """Pull an integer out of a ``ok  key=value`` selftest line."""
    for line in stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[1].startswith(f"{key}="):
            value = parts[1].split("=", 1)[1].split("#")[0].strip()
            try:
                return int(value)
            except ValueError:
                return None
    return None


def _count_bundled_topics() -> int:
    """Count help topics inside the built bundle, wherever PyInstaller put them."""
    roots = [DIST / "pyflic", DIST / "pyflic.app"]
    for root in roots:
        if not root.exists():
            continue
        hits = list(root.rglob("help/content/*.md"))
        if hits:
            return len(hits)
    return 0


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true",
                        help="discard PyInstaller's caches first")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="skip post-build verification (not recommended)")
    args = parser.parse_args()

    if args.clean:
        shutil.rmtree(BUILD, ignore_errors=True)
        shutil.rmtree(DIST, ignore_errors=True)

    version = resolve_version()
    print(f"building pyflic {version} for {sys.platform}")
    write_version_file(version)
    generate_icon()
    freeze(args.clean)
    adhoc_sign_macos()

    if args.skip_smoke:
        print("\nsmoke check skipped")
        return 0

    print("\nsmoke check (frozen build)")
    if not smoke_check(version):
        print("\nBUILD FAILED VERIFICATION — see DEPLOYMENT-PLAN.md Phase 0")
        return 1
    print(f"\nOK: {frozen_executable().parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
