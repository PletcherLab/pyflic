# Frozen per-OS installers are the primary distribution channel

## Context

pyflic's users are bench researchers running FLIC rigs, not developers. The existing install path (`uv add git+https://…`, or `pip install` a wheel) assumes Python, a terminal, and a working mental model of virtual environments. That path loses most of the intended audience before they ever see the hub.

The dependency set makes this harder than usual: PyQt6, pandas, numpy, matplotlib, plotnine and statsmodels on Python 3.13 — a large graph of compiled extensions with no pure-Python fallback. Institutional machines are assumed to be IT-managed, so nothing may require administrator rights.

## Decision

pyflic is distributed as a downloadable, per-user installer for each supported platform. The wheel remains available but is no longer the user-facing route.

- **Freezer.** PyInstaller, **one-folder** mode. One-file mode is rejected outright: it unpacks the whole bundle to a temp directory on every launch, which for this dependency set is a multi-second blank stare before the window appears.
- **One binary, five behaviours.** `pyflic.__main__:main` is the single frozen entry point and dispatches on `argv[1]`, as it already does for the CLI. The hub continues to launch the config editor and QC viewer as subprocesses, but re-execs *itself* with a subcommand rather than resolving a console script. `_resolve_cli` must branch on `sys.frozen`: neither of its current strategies survives freezing — there is no `pyflic-config` on `PATH` in a bundle, and `sys.executable` is the frozen binary, not a Python interpreter.
- **Wrappers.** Inno Setup with `PrivilegesRequired=lowest` → per-user `.exe` (Windows); `.app` in a `.dmg` (macOS); AppImage (Linux). Nothing installs to a system location; nothing prompts for elevation.
- **Build targets.** Four artifacts — Windows x64, macOS arm64, macOS x86_64, Linux AppImage (built against the oldest available glibc). PyInstaller cannot cross-compile, so all four are built on CI runners of the matching platform. universal2 is not attempted: it would require every compiled dependency to ship universal2 wheels, which they do not.
- **Acquisition.** A download page on the lab site is the user-facing surface; the artifacts themselves are hosted as GitHub Release assets. Users never encounter a repository, an account, or a command. The page detects the visitor's platform and offers one primary button, which is what resolves the otherwise-unanswerable "do I have Intel or Apple Silicon?".
- **Unsigned.** Neither the macOS nor Windows build is signed with a paid certificate. Users will hit Gatekeeper and SmartScreen warnings on first launch and are walked through them on the download page. The macOS build **is** ad-hoc signed (`codesign -s -`) — free, and mandatory on Apple Silicon, where an entirely unsigned arm64 binary will not execute at all. Ad-hoc signing also downgrades the alarming *"pyflic is damaged and can't be opened"* message to the milder unidentified-developer path.
- **No update mechanism.** The app makes no network calls and never modifies itself. Users learn about releases from the lab, and reinstall over the top.
- **Two channels, split by audience.** The download page carries installers only, with no mention of Python, pip or git. A separate advanced page retains the wheel install for anyone wanting `import pyflic` in a notebook.

## Considered alternatives

- *Briefcase* — produces native installers for all three platforms from one config, far less wrapper plumbing. Rejected: tuned for its own Toga toolkit, and PyQt6 plus a heavy scientific stack is a sparsely-travelled combination with correspondingly thin help when it breaks.
- *conda `constructor`* — the strongest story for scientific binaries, and what napari and Anaconda use. Rejected: requires restructuring how pyflic is built and released around a conda recipe, and the macOS `.pkg` needs coaxing to avoid requiring admin.
- *Bundle a standalone interpreter without freezing* — sidesteps every hidden-import pathology because nothing is analysed or rewritten. Rejected: hand-rolling the launcher, layout and shortcut creation on three platforms is more bespoke work than PyInstaller's known failure modes.
- *Paid code signing (Apple Developer ~$99/yr, Windows OV cert ~$120–400/yr)* — would remove the security warnings entirely. Deferred on cost and on the identity-validation overhead of a Windows certificate. Revisit if the warnings prove to be where users actually give up.
- *In-app auto-update (Sparkle / WinSparkle / AppImageUpdate)* — rejected on three grounds: three separate mechanisms to maintain, no way to verify a downloaded replacement without code signing, and — the deciding one — analysis software that silently changes itself between two runs of "the same" version is a reproducibility hazard in a research setting.

## Consequences

- **Version reporting must be fixed before the first build ships.** `__version__` comes from `importlib.metadata.version("pyflic")`, and PyInstaller does not bundle dist-info metadata by default, so every frozen build would report `"unknown"`. With no update mechanism, the version a user reads back to you is the only handle you have on a bug report — so `--copy-metadata pyflic` (or a baked `_version.py`) is load-bearing, not cosmetic. The version should also be visible in the hub's title bar and stamped into analysis outputs.
- **Crashes are silent unless made otherwise.** A windowed frozen build has no console; on Windows `sys.stdout` is `None`. Any exception escaping the worker thread closes the window with no trace. A rotating log file under `~/.config/pyflic/logs/` and a global excepthook producing a saveable report are therefore required, not optional. The same `None` stdout means a stray `print()` outside `_SignalWriter`'s redirect raises in a frozen build while working fine in development.
- **`qtawesome` and `plotnine` need explicit collection.** qtawesome loads icon fonts as package data and plotnine/statsmodels import lazily; both fail silently at runtime rather than at build time.
- **Every release is a manual reinstall for every user.** Accepted as the cost of the no-update decision.
- **Nothing writes to the install directory**, which the current code already satisfies — settings go to `~/.config/pyflic/`, caches to `project_dir/.pyflic_cache/`. This must stay true.
