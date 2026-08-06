# Deployment plan — shipping pyflic to non-technical users

**Status:** designed, not started. Written 2026-08-05.
**Rationale lives in** [docs/adr/0002-frozen-per-os-installers.md](docs/adr/0002-frozen-per-os-installers.md) and [docs/adr/0003-help-topics-single-source.md](docs/adr/0003-help-topics-single-source.md). This file is the *work plan*; the ADRs are the *why*. Read them first if a decision below looks arbitrary.

---

## Goal

A bench researcher with no Python, no terminal and no admin rights downloads one file, double-clicks it, and ends up with a working pyflic they can get help inside. No git, no repos, no command lines.

## Constraints that shaped everything

- **No admin rights.** IT-managed institutional machines. Everything installs into the user's home directory; nothing prompts for elevation.
- **Three platforms.** Windows, macOS, Linux.
- **Heavy dependency graph.** PyQt6 + pandas 3 + numpy 2.4 + matplotlib + plotnine + statsmodels on Python 3.13. All compiled, no pure-Python fallback.
- **Research software.** Reproducibility matters more than being on the newest version.

## Decisions (all settled)

| Area | Decision |
|---|---|
| Install | Per-user, no elevation, home directory only |
| Acquisition | Lab download page fronting GitHub Release assets |
| Trust | Unsigned; ad-hoc Mac signing; bypass documented on the web page |
| Front door | One icon → hub; one frozen binary re-execing itself for subcommands |
| Packaging | PyInstaller **one-folder** → Inno Setup / .dmg / AppImage |
| Updates | None. Version *visibility* is the compensating control |
| Help topics | One topic = one file under `pyflic/help/`; long guides become derived |
| Help UI | `QTextBrowser` window; HTML generated from markdown at build time |
| First run | Hub empty-state panel; example project copied on demand |
| Build targets | Win x64, macOS arm64, macOS x86_64, Linux AppImage; page auto-detects |
| Channels | Installers for users; wheel on a separate "advanced" page |
| Docs job | `.github/workflows/daily_docs.yaml` retired |
| Support | Rotating log file + crash dialog with "Save report…" |

Asserted without discussion, revisit if you disagree: don't modify `PATH`; uninstall leaves `~/.config/pyflic` intact; releases trigger on a git tag (`hatch-vcs` makes the tag *be* the version).

---

## Phase 0 — Blocking fixes

**Do these before any freezing work.** Each produces a build that compiles, launches, and fails later on a user's machine. None is caught by "does the app open?".

### 0.1 — `_resolve_cli` breaks under `sys.frozen`

**Files:** [pyflic/base/analysis_hub.py:238-242](pyflic/base/analysis_hub.py#L238-L242), and the duplicate at [pyflic/base/qc_viewer.py:74](pyflic/base/qc_viewer.py#L74).

**What breaks:** both strategies fail in a bundle. `shutil.which("pyflic-config")` finds nothing, because that executable exists only when pip generates it from `[project.scripts]`. The fallback `[sys.executable, "-m", module]` is worse — `sys.executable` is *the frozen binary*, not a Python interpreter, so the command becomes `pyflic.exe -m pyflic.base.config_editor`. The dispatcher in `__main__.py` sees `-m` as the subcommand, doesn't recognise it, and exits. **Symptom: clicking "Edit config…" or "QC viewer…" does nothing.**

**Fix:** branch on frozen state and use the subcommand dispatcher that already exists in [pyflic/__main__.py](pyflic/__main__.py):

```python
def _resolve_cli(name: str, module: str, subcommand: str) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, subcommand]   # pyflic.exe config
    exe = shutil.which(name)
    if exe:
        return [exe]
    return [sys.executable, "-m", module]
```

Callers pass `"config"` / `"qc"`. Fix both copies — `qc_viewer.py` has its own.

### 0.2 — `__version__` collapses to `"unknown"`

**File:** [pyflic/__init__.py:1-6](pyflic/__init__.py#L1-L6).

**What breaks:** `importlib.metadata.version("pyflic")` reads the `.dist-info` directory pip writes at install time. PyInstaller does not bundle it, so the lookup raises `PackageNotFoundError` and falls through to `"unknown"`.

**Why it matters more than it looks:** there is no update mechanism, so the version a user reads back to you is the only handle you have on any bug report.

**Fix:** add `--copy-metadata pyflic` to the PyInstaller build (or have `hatch-vcs` write a `_version.py` at build time and fall back to it). Then surface it: hub title bar, and stamped into analysis outputs and PDF reports.

### 0.3 — Crashes are silent; `sys.stdout` is `None`

**What breaks:** a frozen windowed app has no console. On Windows `sys.stdout` is literally `None`. Any exception escaping the worker thread ([analysis_hub.py:325](pyflic/base/analysis_hub.py#L325) catches only inside the worker) closes the window with no message and no trace. The user's report is "it disappeared."

Second trap in the same place: any bare `print()` on a path not wrapped by `_SignalWriter` raises `AttributeError: 'NoneType' object has no attribute 'write'` when frozen, while working fine in development. Audit for stray prints.

**Fix:**
- Rotating log file at `~/.config/pyflic/logs/pyflic.log` (matches the existing settings location convention in [pyflic/base/ui/settings.py:14](pyflic/base/ui/settings.py#L14)).
- Global `sys.excepthook` showing a dialog: "pyflic hit an unexpected error", with a **Save report…** button writing one file containing version, OS, architecture, traceback, and the active config.
- Guard stdout/stderr being `None` early in `main()`.

### 0.4 — qtawesome and plotnine need explicit collection

**What breaks:** PyInstaller decides what to include by statically reading `import` statements. It cannot see qtawesome's icon **fonts** (data files opened by filename — used via [pyflic/base/ui/icons.py](pyflic/base/ui/icons.py)) or plotnine's and statsmodels' lazy string-based imports. Build succeeds, hub opens, **failure appears the first time a user clicks a plot button**.

**Fix:** in the spec file — `--collect-data qtawesome`, `--collect-all plotnine`, hidden imports for `statsmodels`, plus matplotlib backend data. Verify by running a plot in the *frozen* build, not the source tree.

---

## Phase 1 — Prove it freezes (Windows only)

Cheapest possible answer to the question that could invalidate the whole plan: *does this dependency graph freeze at all?* You're on Windows, so iterate locally in minutes instead of waiting on CI.

1. PyInstaller spec for `pyflic/__main__.py`, **one-folder**, windowed.
   - One-file mode is rejected: it unpacks the whole bundle to temp on every launch — a multi-second blank stare for this dependency set.
2. Get it to build, then verify **in the frozen build**: hub opens → "Edit config…" launches the editor → "QC viewer…" launches → a plot renders → icons appear → `pyflic.exe version` prints a real version.
3. Expect to iterate on hidden imports. Note each one in the spec with a comment saying why.

**Exit criterion:** a Windows folder you can zip, send to someone, and have them run.

## Phase 2 — The other three targets

PyInstaller cannot cross-compile, so each artifact is built on a CI runner of its own platform.

- Windows x64 (`windows-latest`)
- macOS arm64 (Apple Silicon runner)
- macOS x86_64 (Intel runner)
- Linux AppImage (**oldest available glibc runner** — build on new glibc and it won't run on older institutional distros)

macOS builds must be **ad-hoc signed**: `codesign --force --deep -s - pyflic.app`. Free, no account. Mandatory on Apple Silicon, where a wholly unsigned arm64 binary will not execute at all, and it downgrades the alarming *"pyflic is damaged and can't be opened"* message to the milder unidentified-developer path.

Universal2 is **not** attempted — it needs every compiled dependency shipped universal2, which numpy/pandas/statsmodels wheels generally aren't.

## Phase 3 — Wrappers and the download page

- **Windows:** Inno Setup, `PrivilegesRequired=lowest`, install to `%LOCALAPPDATA%`, Start Menu + desktop shortcut, uninstaller entry. Do **not** modify `PATH`.
- **macOS:** `.app` inside a `.dmg` with a drag-to-Applications backdrop. `~/Applications` works without admin.
- **Linux:** AppImage — a single double-clickable file.
- **Download page** on the lab site:
  - Detects the visitor's platform, shows **one** primary button, others behind "other downloads". This is what answers the otherwise-unanswerable "do I have Intel or Apple Silicon?".
  - Shows the version it's serving, and a changelog.
  - Carries the **Gatekeeper and SmartScreen walkthroughs with screenshots.** This is the riskiest prose in the product — if it's unclear, users conclude pyflic is malware and stop. macOS changes this flow between releases; re-check the screenshots yearly.
  - Installers only. No mention of Python, pip, uv or git.
  - Separate, plainly-linked **advanced page** keeps the wheel install for `import pyflic` in notebooks.

## Phase 4 — Help system

Blocks nothing above; the largest single chunk. Can proceed in parallel.

1. Build the `QTextBrowser` help window — topic list beside topic body, one window shared by all apps ([CONTEXT.md:37](CONTEXT.md#L37)).
2. Migrate ~2,000 lines from [doc/](doc/) (`USAGE.md` 1007, `SCRIPTS.md` 608, `PLOTS.md` 205, `INSTALL.md` 152) into one-topic-per-file under `pyflic/help/`, organised concepts / screens / tasks.
3. Ordering manifest so guides and the PDFs can be reassembled from topics. **No prose is authored at guide level.**
4. Markdown → HTML build step feeding *both* the bundle and the download page. Install and bypass topics **must** render on the web — the user needs them before the app will open.
5. Add `[?]` buttons across the apps, each naming one topic.
6. Delete `.github/workflows/daily_docs.yaml`. It was aimed at `docs/` (ADRs) and never touched `doc/` anyway. Replacement discipline: **a topic edit ships in the same change as the behaviour it describes.**

## Phase 5 — First-run experience

Currently a new user gets "No project loaded" ([analysis_hub.py:576](pyflic/base/analysis_hub.py#L576)) and no route forward — `File → New` lives in a *different* app ([config_editor.py:654](pyflic/base/config_editor.py#L654)).

Add a hub empty-state panel with three large buttons:
- **Open a project folder…**
- **Create a new project…** → launches the config editor at File → New
- **Open the example project** → copies a bundled trimmed example into `~/Documents/pyflic-example` on first click

Plus a link to the Getting Started topic. [settings.py:57](pyflic/base/ui/settings.py#L57) already has `recent_projects` to hang this off.

Build a **purpose-built trimmed example** (one DFM, short run, a few MB) rather than shipping `test_experiment_small` — 93MB raw, and confusing to read.

---

## Open questions

- **Topic inventory.** We settled the help system's shape, not a word of its contents. Which topics exist, and which controls get a `[?]`.
- **Test fixtures.** `test_experiment` (366MB), `test_experiment_small` (93MB), `SubdirTest` (43MB) are committed to the repo and **nothing in [tests/](tests/) references them** — verified. ~500MB of manual scratch data. Decide whether they stay now that a trimmed example ships.
- **Where the version is displayed.** Hub title bar was asserted; an About dialog or status-bar corner are equally reasonable.

## Things already verified as fine — don't re-investigate

- **Nothing writes to the install directory.** Settings go to `~/.config/pyflic/ui.json` with silent failure ([settings.py:14](pyflic/base/ui/settings.py#L14)); cache goes to `project_dir/.pyflic_cache/` ([cache.py:24](pyflic/base/cache.py#L24)). A read-only frozen bundle is fine. **Keep it that way.**
- `~/.config` is unconventional on Windows and macOS but works and is consistent. Not worth churning.
- The hub already launches its siblings as subprocesses, which survives freezing once 0.1 is fixed. No need to rearchitect into in-process windows.
