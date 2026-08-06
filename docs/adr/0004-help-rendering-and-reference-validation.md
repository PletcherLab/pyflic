# Help rendering, coupling, and reference validation

Implements the topic model decided in
[ADR-0003](0003-help-topics-single-source.md). That ADR settles *what* help is —
topics as the single source, guides derived, one offline in-app window. This one records
the mechanics that are surprising enough to be "corrected" by a future reader, and flags
three points where the implementation currently diverges from 0003.

## Decision

- **Topics live in `pyflic/help/content/`,** one markdown file per topic, id = filename
  stem. Verified that hatchling ships non-Python files under the package directory with no
  `[tool.hatch.build]` section, so they are present in a `pip install` — which the old
  `doc/` tree was not.

- **Markdown is rendered at runtime, not converted to HTML at build time.**
  `QTextBrowser.setMarkdown(..., MarkdownDialectGitHub)` renders the topics directly,
  tables included (measured on Qt 6.10). There is no build step. See *Divergences* below.

- **Anchors resolve by walking heading blocks.** Qt's `setMarkdown` emits **no anchors**
  for headings — `scrollToAnchor("some-heading")` is measurably a no-op (scrollbar
  unchanged at 0). Anchor navigation therefore walks the document with
  `QTextBlock.blockFormat().headingLevel()` and matches heading text by slug. The same
  walk builds the in-topic outline. **Do not "simplify" this back to `scrollToAnchor`.**

- **Heading sizes are merged in block by block,** for the same underlying reason:
  `setMarkdown` builds the document directly rather than parsing HTML, so
  `setDefaultStyleSheet` never applies to it.

- **Anchor scrolling is deferred one event-loop turn.** Help is often opened before the
  window is shown, and an unlaid-out document reports the wrong cursor rect — landing the
  reader a couple of sections off target.

- **One-way dependency: GUI → help.** `pyflic/help/` imports the shared UI primitives in
  `pyflic.base.ui` and nothing else from pyflic — never `analysis_hub`, `qc_viewer`,
  `config_editor`, `script_editor`, or any analysis module. GUI modules import
  `HelpButton` / `install_help_shortcut` and name topics explicitly at each call site, each
  behind a guarded import. Deleting `pyflic/help/` breaks only those call sites; the
  analysis pipeline cannot be affected. A missing topic renders a "topic not available"
  page rather than raising into the host app.

- **`pyflic/__init__.py` imports its public API lazily (PEP 562).** The isolation above is
  otherwise defeated by Python itself: importing *any* subpackage runs the parent
  `__init__`, so `import pyflic.help` used to load the entire analysis stack — measured,
  `pyflic.base.dfm`, `pyflic.base.algorithms.*`, pandas, numpy, statsmodels and plotnine
  were all imported. `from pyflic import load_experiment_yaml` is unchanged; the modules
  load on first attribute access. This also cut cold CLI startup from **5.8 s to 0.2 s**,
  since `pyflic --help`, `pyflic version` and `pyflic help` never needed the numerical
  stack.

- **Help affordances are built only when help imports.** The hub queries
  `_help_available()` before adding the sidebar Help item, so a broken or absent help
  package leaves no dead control behind; `_open_help` stays guarded regardless.

- **Deferred scrolls carry a navigation generation.** The anchor scroll is queued, so
  navigating away before it fires would otherwise scroll the *newly opened* topic to a
  heading it happens to share with the old one — `scripts-actions` and `plots-catalog`
  both have "Sliding-window plots". The queued call is discarded if the generation
  moved on.

- **Heading formatting selects start-of-block to end-of-block explicitly.**
  `QTextCursor.SelectionType.BlockUnderCursor` also spans the *preceding* block
  separator, which propagated the heading's block format onto the paragraph above it.
  Measured on `reference-parameters`: 16 real headings became 31 blocks reporting a
  non-zero `headingLevel()` — corrupting the very metadata anchor resolution matches
  against. `tests/test_help_window.py` asserts the document's heading set equals the
  markdown's, for every topic.

- **Help buttons re-tint on palette change.** `changeEvent` watches `PaletteChange` and
  `ApplicationPaletteChange` only — never `StyleChange`, which `setStyleSheet` itself
  emits and which recurses.

- **Topic content is cached against file size and mtime,** so editing a topic with the app
  open is picked up without a restart.

- **A declarative auto-injecting registry was rejected** — matching on `objectName`
  strings that nothing validates fails silently when a widget is renamed, and is far
  harder to trace than an explicit call site.

- **References are validated by test.** Topic ids and anchors are plain strings at every
  call site. `tests/test_help_refs.py` resolves all of them — TOC entries, guide members,
  cross-topic links, same-page anchors, the hub's `_CARD_HELP` map, the QC viewer's
  per-tab `_help_ref` assignments, and the parameter key lists behind the runtime-built
  refs — and fails on any dangling reference or any topic missing from the TOC. This is
  what makes the content safe to rewrite. Note the shape of the gap this closed: a
  literal-scanning regex silently sees *fewer* call sites than exist, and reports success.
  Any new indirection needs its own check.

- **`pyflic help` opens the help window.** It previously printed CLI usage; `-h` and
  `--help` still do, per the convention that flags carry usage and subcommands carry
  actions.

- **Help buttons are amber, 24px.** Help belongs to no analysis category, and the warm
  tint makes it findable against the category-coloured controls it sits beside. Placed
  immediately after a card's title text rather than right-aligned, so it cannot be clipped
  when the cards column is narrow.

## Divergences from ADR-0003, open for decision

1. **Runtime markdown vs. build-time HTML.** 0003 specifies "HTML generated from the
   markdown at build time" and lists a markdown → HTML build step as a release consequence.
   This implementation renders markdown at runtime instead: no build step, no generated
   artefact to fall out of date, and tables/anchors were verified to work. If the download
   page needs the same HTML (see 3), a build step returns — but as an *additional* output,
   not as the app's rendering path.

2. **`doc/` deleted rather than repopulated with derived guides.** 0003 §34 assigns `doc/`
   the role of holding derived guides, and §13 says the long documents and PDFs become
   generated output. Here `doc/*.md` and the stale PDFs were deleted outright and the README
   links into `pyflic/help/content/`, which GitHub renders directly — chosen because the
   hand-maintained PDFs had already drifted a month behind their markdown sources. Guides
   currently exist only as orderings in `toc.py`, consumed by the help window. Nothing
   regenerates `doc/`.

3. **No web output yet.** 0003 §15 argues the same generated HTML must also feed the lab
   download page, because install and security-warning-bypass instructions are needed
   *before* the app will open — a requirement in-app help cannot satisfy. That argument
   stands and is unaddressed here: `install.md` is currently an in-app topic only. Resolving
   it means adding a generator that emits HTML for at least the install topics.

Divergences 1 and 3 are coupled: adding the download page brings back a build step, and at
that point emitting HTML for the app too becomes cheap. They were left open rather than
guessed at, because they belong to the deployment plan rather than to this change.
