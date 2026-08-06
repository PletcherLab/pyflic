# Deployment plan — shipping pyflic to non-technical users

**Status:** designed; help system built, packaging not started.
**Written** 2026-08-05. **Revised** 2026-08-06 after the help-system implementation.
**Rationale lives in** [docs/adr/0002-frozen-per-os-installers.md](docs/adr/0002-frozen-per-os-installers.md), [docs/adr/0003-help-topics-single-source.md](docs/adr/0003-help-topics-single-source.md) and [docs/adr/0004-help-rendering-and-reference-validation.md](docs/adr/0004-help-rendering-and-reference-validation.md). This file is the *work plan*; the ADRs are the *why*. Read them first if a decision below looks arbitrary.

---

## What changed since the first draft

The help system landed, and it is substantially more complete than this plan assumed. `pyflic/help/` now holds 26 topics, a `QTextBrowser` window with search and in-topic outlines, help buttons wired at ~25 call sites, guide orderings in `toc.py`, and two test modules — including [tests/test_help_refs.py](tests/test_help_refs.py), which fails the build on any dangling topic id or anchor. `doc/`'s four long guides and their stale PDFs were deleted; the README now links directly into `pyflic/help/content/`, which GitHub renders.

**Phase 4 is therefore mostly done.** What remains of it is listed below and is small.

Two consequences pull *into* this plan rather than out of it:

- **[ADR-0004](docs/adr/0004-help-rendering-and-reference-validation.md) explicitly defers two divergences to the deployment plan** (its §98–123). Both are resolved below in Phase 3.
- **The help implementation introduced two new freezing hazards** — lazily-imported modules and markdown content as package data. Both are now Phase 0 items, and one of them fails *silently*.

Everything in Phase 0 from the original plan is **still outstanding**. No packaging work has begun.

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
| Help topics | One topic = one file in `pyflic/help/content/` — **built** |
| Help UI | `QTextBrowser`, markdown rendered **at runtime**, no build step — **built** |
| Web help | HTML generated for install topics only, as an *additional* output — **Phase 3** |
| Guides | Orderings in `toc.py`; `doc/` deleted, not regenerated — **settled** |
| First run | Hub empty-state panel; example project copied on demand — **not started** |
| Build targets | Win x64, macOS arm64, macOS x86_64, Linux AppImage; page auto-detects |
| Channels | Installers for users; pip/uv on a separate "advanced" page |
| Docs job | `.github/workflows/daily_docs.yaml` retired — **still present** |
| Support | Rotating log file + crash dialog with "Save report…" — **not started** |

Asserted without discussion, revisit if you disagree: don't modify `PATH`; uninstall leaves `~/.config/pyflic` intact; releases trigger on a git tag (`hatch-vcs` makes the tag *be* the version).

---

## Phase 0 — Blocking fixes

**Do these before any freezing work.** Each produces a build that compiles, launches, and fails later on a user's machine. None is caught by "does the app open?".

### 0.1 — `_resolve_cli` breaks under `sys.frozen` — *outstanding*

**Files:** [pyflic/base/analysis_hub.py:251](pyflic/base/analysis_hub.py#L251) and [pyflic/base/qc_viewer.py:69](pyflic/base/qc_viewer.py#L69) — two separate definitions. **Three** call sites now: [analysis_hub.py:1629](pyflic/base/analysis_hub.py#L1629), [analysis_hub.py:1674](pyflic/base/analysis_hub.py#L1674), and [qc_viewer.py:462](pyflic/base/qc_viewer.py#L462) (the QC viewer launches the config editor too).

**What breaks:** both strategies fail in a bundle. `shutil.which("pyflic-config")` finds nothing, because that executable exists only when pip generates it from `[project.scripts]`. The fallback `[sys.executable, "-m", module]` is worse — `sys.executable` is *the frozen binary*, not a Python interpreter, so the command becomes `pyflic.exe -m pyflic.base.config_editor`. The dispatcher in `__main__.py` sees `-m` as the subcommand, doesn't recognise it, and exits. **Symptom: clicking "Edit config…" or "QC viewer…" does nothing.**

**Fix:** branch on frozen state and use the subcommand dispatcher that already exists in [pyflic/__main__.py:32](pyflic/__main__.py#L32):

```python
def _resolve_cli(name: str, module: str, subcommand: str) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, subcommand]   # pyflic.exe config
    exe = shutil.which(name)
    if exe:
        return [exe]
    return [sys.executable, "-m", module]
```

Callers pass `"config"` / `"qc"`. Fix both copies — consider consolidating them into `pyflic.base.utils` while you're there, since they've now drifted into three call sites across two modules.

### 0.2 — `__version__` collapses to `"unknown"` — *outstanding*

**File:** [pyflic/__init__.py:20-23](pyflic/__init__.py#L20-L23).

**What breaks:** `importlib.metadata.version("pyflic")` reads the `.dist-info` directory pip writes at install time. PyInstaller does not bundle it, so the lookup raises `PackageNotFoundError` and falls through to `"unknown"`.

**Why it matters more than it looks:** there is no update mechanism, so the version a user reads back to you is the only handle you have on any bug report. [install.md:79-82](pyflic/help/content/install.md#L79-L82) already tells users to record `pyflic version` alongside their results — which silently becomes worthless advice in a frozen build.

**Fix:** add `--copy-metadata pyflic` to the PyInstaller build (or have `hatch-vcs` write a `_version.py` at build time and fall back to it). Then surface it: hub title bar, and stamped into analysis outputs and PDF reports.

### 0.3 — Crashes are silent; `sys.stdout` is `None` — *outstanding*

**What breaks:** a frozen windowed app has no console. On Windows `sys.stdout` is literally `None`. Any exception escaping the worker thread closes the window with no message and no trace. The user's report is "it disappeared."

Second trap in the same place: there are **151 bare `print()` calls** across `pyflic/base/`. Those inside a worker are fine — `_SignalWriter` redirects `sys.stdout` while it runs — but any that fire outside that window raise `AttributeError: 'NoneType' object has no attribute 'write'` when frozen, while working perfectly in development.

**Fix:** do **not** audit 151 print sites. Instead, at the very top of `main()`, replace `sys.stdout`/`sys.stderr` with a null writer when they are `None`. That neutralises the entire class in three lines. Then add:

- Rotating log file at `~/.config/pyflic/logs/pyflic.log` (matches the existing settings location in [pyflic/base/ui/settings.py:14](pyflic/base/ui/settings.py#L14)).
- Global `sys.excepthook` showing a dialog: "pyflic hit an unexpected error", with a **Save report…** button writing one file containing version, OS, architecture, traceback, and the active config.

### 0.4 — qtawesome and plotnine need explicit collection — *outstanding*

**What breaks:** PyInstaller decides what to include by statically reading `import` statements. It cannot see qtawesome's icon **fonts** (data files opened by filename, used via [pyflic/base/ui/icons.py](pyflic/base/ui/icons.py)) or plotnine's and statsmodels' lazy string-based imports. Build succeeds, hub opens, **failure appears the first time a user clicks a plot button**.

**Fix:** in the spec — `--collect-data qtawesome`, `--collect-all plotnine`, hidden imports for `statsmodels`, plus matplotlib backend data. Verify by running a plot in the *frozen* build, not the source tree.

### 0.5 — Help content is package data PyInstaller will not collect — *new, and fails silently*

**File:** [pyflic/help/topics.py:21](pyflic/help/topics.py#L21) — `CONTENT_DIR = Path(__file__).parent / "content"`.

**What breaks:** PyInstaller collects `.py`, not `.md`. Without an explicit datas entry, `content/` is absent from the bundle, `available()` returns `[]`, and **every help topic in the app renders "topic not available."**

**This is the most dangerous item in Phase 0**, because the help system is deliberately built to fail quietly: `load()` never raises ([topics.py:98-115](pyflic/help/topics.py#L98-L115)), a missing topic degrades to a placeholder page rather than an exception, and the hub checks `_help_available()` before even showing the Help control. Every one of those is correct design — and together they mean a bundle with no help content looks like a working app with empty help. Nothing fails, nothing logs, no test catches it, because [tests/test_help_refs.py](tests/test_help_refs.py) runs against the source tree where the files obviously exist.

**Note the trap in the reasoning:** [ADR-0004:11-14](docs/adr/0004-help-rendering-and-reference-validation.md#L11-L14) verified that *hatchling* ships these files in a wheel with no `[tool.hatch.build]` section. That verification is real but does **not** carry over — PyInstaller is an entirely different mechanism and honours none of hatchling's defaults.

**Fix:** `--collect-data pyflic` (or an explicit `datas=[('pyflic/help/content', 'pyflic/help/content')]`), preserving the relative path so `Path(__file__).parent / "content"` still resolves under `_internal`. Then add a **smoke check that runs against the frozen build**, not the source tree: assert `len(available()) == 26` (or whatever the count is at the time). Without that assertion this bug will recur every time the spec is touched.

### 0.6 — PEP 562 lazy imports are invisible to static analysis — *new*

**File:** [pyflic/__init__.py:50-73](pyflic/__init__.py#L50-L73) — the `_LAZY` dict maps public names to module paths, resolved through `importlib.import_module` on attribute access.

**What breaks:** PyInstaller's module graph follows literal `import` statements, including function-level ones. It cannot follow `importlib.import_module(_LAZY[name])`, because the module name is a runtime value.

**Mitigating factor, but don't lean on it:** the same modules are listed as real imports inside the `if TYPE_CHECKING:` block at [__init__.py:25-46](pyflic/__init__.py#L25-L46), and those statements do compile to `IMPORT_NAME` opcodes that the analyser generally does pick up. So this may well work by accident. "Works by accident" is not a property you want in a build that only fails on someone else's machine.

**Fix:** list the `_LAZY` values explicitly as `hiddenimports` in the spec, with a comment pointing back at `_LAZY` so the two stay in step. Verify in the frozen build that `from pyflic import load_experiment_yaml` and `write_experiment_report` both resolve.

This change was worth making regardless — [ADR-0004:42-49](docs/adr/0004-help-rendering-and-reference-validation.md#L42-L49) records it cutting cold CLI startup from 5.8s to 0.2s. It just needs declaring to the freezer.

---

## Phase 1 — Prove it freezes (Windows only)

Cheapest possible answer to the question that could invalidate the whole plan: *does this dependency graph freeze at all?* You're on Windows, so iterate locally in minutes instead of waiting on CI.

1. PyInstaller spec for `pyflic/__main__.py`, **one-folder**, windowed.
   - One-file mode is rejected: it unpacks the whole bundle to temp on every launch — a multi-second blank stare for this dependency set.
2. Verify **in the frozen build**, not the source tree:
   - hub opens → "Edit config…" launches the editor → "QC viewer…" launches (0.1)
   - `pyflic.exe version` prints a real version (0.2)
   - a plot renders; icons appear (0.4)
   - **help opens and topics have content** (0.5) — check an actual topic body, not just that the window appears
   - `from pyflic import load_experiment_yaml` resolves (0.6)
3. Expect to iterate on hidden imports. Note each in the spec with a comment saying why.

**Exit criterion:** a Windows folder you can zip, send to someone, and have them run.

## Phase 2 — The other three targets

PyInstaller cannot cross-compile, so each artifact is built on a CI runner of its own platform.

- Windows x64 (`windows-latest`)
- macOS arm64 (Apple Silicon runner)
- macOS x86_64 (Intel runner)
- Linux AppImage (**oldest available glibc runner** — build on new glibc and it won't run on older institutional distros)

macOS builds must be **ad-hoc signed**: `codesign --force --deep -s - pyflic.app`. Free, no account. Mandatory on Apple Silicon, where a wholly unsigned arm64 binary will not execute at all, and it downgrades the alarming *"pyflic is damaged and can't be opened"* message to the milder unidentified-developer path.

Universal2 is **not** attempted — it needs every compiled dependency shipped universal2, which numpy/pandas/statsmodels wheels generally aren't.

## Phase 3 — Wrappers, web output, and the download page

### 3.1 Installers

- **Windows:** Inno Setup, `PrivilegesRequired=lowest`, install to `%LOCALAPPDATA%`, Start Menu + desktop shortcut, uninstaller entry. Do **not** modify `PATH`.
- **macOS:** `.app` inside a `.dmg` with a drag-to-Applications backdrop. `~/Applications` works without admin.
- **Linux:** AppImage — a single double-clickable file.

### 3.2 Resolving the two divergences ADR-0004 handed over

**Divergence 1 — runtime markdown vs build-time HTML. Resolved: keep runtime rendering.** [ADR-0003](docs/adr/0003-help-topics-single-source.md) specified HTML generated at build time; the implementation renders markdown directly through `setMarkdown` and it works, tables included. There is no reason to reintroduce a build step on the app's critical path. The HTML generator below is an **additional output for the web**, never the app's rendering path.

**Divergence 3 — no web output. Resolved: generate HTML for the web-needed topics.** This requirement stands and is the reason the generator exists at all: install and security-bypass instructions must be readable *before* the app will open, which in-app help cannot satisfy by definition. Emit HTML for `install`-family topics plus `getting-started` and `troubleshooting`. Reuse `pyflic.help.topics` for parsing so slugs and anchors match the app exactly, and let [tests/test_help_refs.py](tests/test_help_refs.py) cover the generated set too.

**Divergence 2 — `doc/` deleted rather than regenerated. Accept as settled, no work.** Guides live as orderings in `toc.py`; the README links into `pyflic/help/content/`, which GitHub renders. The hand-maintained PDFs had already drifted a month behind their sources, which is the argument against bringing them back.

### 3.3 The install topic has to split

[pyflic/help/content/install.md](pyflic/help/content/install.md) is currently **entirely the pip/uv/GitHub story** — `uv add git+https://…`, wheel files, `uv add --upgrade`. Under the two-channel decision that content is the *advanced* page, not the user-facing one. So:

- Keep the existing `install.md` content as the **advanced/scripting** channel documentation.
- Author a **new** installer topic: download, double-click, and — the part that matters most — the Gatekeeper and SmartScreen walkthroughs with screenshots.
- Its "Updating" guidance changes completely: no `uv add --upgrade`, just "download the new installer and run it over the top". Note that [install.md:79-82](pyflic/help/content/install.md#L79-L82)'s reproducibility argument for pinning survives and gets *stronger* under the no-update decision — keep it, reworded for installers.

Because both topics ship in-app, add both to `toc.py`, and `test_help_refs.py` will hold the references honest.

### 3.4 Download page

- Detects the visitor's platform, shows **one** primary button, others behind "other downloads". This is what answers the otherwise-unanswerable "do I have Intel or Apple Silicon?".
- Shows the version it's serving, and a changelog.
- Carries the **Gatekeeper and SmartScreen walkthroughs**, generated from the new install topic in 3.3. This is the riskiest prose in the product — if it's unclear, users conclude pyflic is malware and stop. macOS changes this flow between releases; re-check the screenshots yearly.
- Installers only. No mention of Python, pip, uv or git.
- Separate, plainly-linked **advanced page** carrying the old `install.md` and [python-api.md](pyflic/help/content/python-api.md).

## Phase 4 — Help system — *mostly done*

Built: `pyflic/help/` with 26 topics, the `QTextBrowser` window, search, in-topic outlines, guide orderings, ~25 wired help-button call sites, and reference validation by test. See [ADR-0004](docs/adr/0004-help-rendering-and-reference-validation.md) for the mechanics — several of which are deliberately non-obvious and carry "do not simplify this" warnings.

**Remaining:**

1. **Retire [.github/workflows/daily_docs.yaml](.github/workflows/daily_docs.yaml).** Still present, still aimed at `docs/` (your ADR directory), and now aimed at a `doc/` tree that no longer exists. It has been opening PRs against a target that was deleted. Delete the workflow. Replacement discipline: **a topic edit ships in the same change as the behaviour it describes.**
2. The new installer topic (3.3) and the web generator (3.2).
3. Nothing else. Do not re-plan this phase.

## Phase 5 — First-run experience — *not started*

Still unchanged: a new user gets `"No project loaded."` ([analysis_hub.py:648](pyflic/base/analysis_hub.py#L648)) and no route forward — `File → New` lives in a *different* app.

Add a hub empty-state panel with three large buttons:
- **Open a project folder…**
- **Create a new project…** → launches the config editor at File → New
- **Open the example project** → copies a bundled trimmed example into `~/Documents/pyflic-example` on first click

Plus a help button pointing at `getting-started-first-run` — which now exists, so this is just a call site. [settings.py:57](pyflic/base/ui/settings.py#L57) already has `recent_projects` to hang this off.

Build a **purpose-built trimmed example** (one DFM, short run, a few MB) rather than shipping `test_experiment_small` — 93MB raw, and confusing to read. Note it becomes bundled data with the same collection problem as 0.5.

---

## Open questions

- **Test fixtures.** `test_experiment` (366MB), `test_experiment_small` (93MB), `SubdirTest` (43MB) are committed and **nothing in [tests/](tests/) references them** — verified twice. ~500MB of manual scratch data. Decide whether they stay.
- **Where the version is displayed.** Hub title bar was asserted; an About dialog or status-bar corner are equally reasonable.
- **Whether the frozen build ships the CLI usefully at all.** `pyflic lint` and `pyflic report` exist and are documented in `install.md`, but with `PATH` untouched a GUI user cannot reach them. Either surface both in the GUI or accept they are advanced-channel only.

## Things already verified as fine — don't re-investigate

- **Nothing writes to the install directory.** Settings go to `~/.config/pyflic/ui.json` with silent failure ([settings.py:14](pyflic/base/ui/settings.py#L14)); cache goes to `project_dir/.pyflic_cache/` ([cache.py:24](pyflic/base/cache.py#L24)). A read-only frozen bundle is fine. **Keep it that way.**
- `~/.config` is unconventional on Windows and macOS but works and is consistent. Not worth churning.
- The hub launches its siblings as subprocesses, which survives freezing once 0.1 is fixed. No need to rearchitect into in-process windows.
- Help is one-way coupled (GUI → help) and degrades safely when absent — see [ADR-0004:34-40](docs/adr/0004-help-rendering-and-reference-validation.md#L34-L40). Good design; just be aware it is exactly what hides 0.5.
