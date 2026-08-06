# Help topics are the single source of user prose; guides are derived

## Context

User-facing prose lives in `doc/` as four long documents — `USAGE.md` (1,007 lines), `SCRIPTS.md`, `PLOTS.md`, `INSTALL.md` — plus generated PDFs. `CONTEXT.md` already defines a **help topic** as *"one addressable, self-contained piece of user-facing explanation"* and requires that *"the topic is the only copy of the text"*, but no help system exists in the code and none of the four documents satisfies that definition. `USAGE.md` alone spans the ten-step event-detection algorithm, the full YAML schema, the script reference and the CLI.

Shipping pyflic as a frozen installer to non-technical researchers makes this urgent: help has to be available offline, inside the app, addressable from a `[?]` button beside the control it explains.

## Decision

One help topic is one markdown file, authored under `pyflic/help/` and shipped inside the bundle. Guides are assembled from topics; nothing is authored at guide level.

- **Topics are the only source.** The four long documents stop being written and become derived output — a guide is topics concatenated in a chosen order. The existing PDFs are generated the same way. No sentence a user reads exists in two places.
- **Rendering.** A single in-app help window built on `QTextBrowser`, with a topic list and the topic beside it, displaying HTML generated from the markdown at build time. Fully offline, negligible bundle cost.
- **One source, two outputs.** The same generated HTML feeds the lab download page. This is not merely convenient — it is required. Install and security-warning-bypass instructions cannot be an in-app topic, because the user needs them *before* the app will open. Those topics must render on the web.
- **Topics version with the app.** They are inside the bundle, so a user's help always describes the build they are running. With no update mechanism, help fetched from the web would drift out of sync with installed versions indefinitely.
- **Topics are product copy.** A topic is written when the behaviour it describes is written, in the same change. The nightly documentation job is retired.

## Considered alternatives

- *Link help buttons at anchors in the existing long documents* — no migration cost and nothing to keep in sync. Rejected: the reader lands mid-manual with a scrollbar showing they are a fifth of the way through something enormous, which is exactly what "self-contained" was written to prevent.
- *Generate topics by splitting the long files on headings at build time* — no migration, no duplicated prose. Rejected: heading text silently becomes an API, so renaming a heading breaks a help button, and the current headings are uneven (`# Two-well (chamber_size: 2)` appears as a top-level heading inside a subsection).
- *Keep long guides for end-to-end reading and write separate short topics for help buttons* — the best reading experience for both audiences. Rejected: two bodies of prose describing the same behaviour, guaranteed to drift, and forbidden by `CONTEXT.md`'s "the topic is the only copy of the text".
- *`QWebEngineView` instead of `QTextBrowser`* — real CSS, real search, in-app. Rejected: adds roughly 150–250MB to an already-large bundle and is among the hardest components to freeze reliably across three platforms.
- *Online-only help* — always current, nothing shipped, editable without a release. Rejected: useless on an offline rig machine, and with no update mechanism a user on an old build would read documentation for a version they do not have.
- *Retarget the nightly documentation job at the new topic directory* — rejected along with the job itself. Help topics are now shipped product surface; plausible-sounding generated prose about threshold computation is precisely the failure a PR review is likely to wave through.

## Consequences

- **A one-time migration of roughly 2,000 lines** of existing prose into individual topics, plus an ordering manifest to reassemble guides.
- **Nothing now watches for stale topics.** Retiring the nightly job removes the only automated check that a behaviour change contradicted the documentation. The discipline that replaces it is that a topic edit ships in the same change as the behaviour it describes.
- **A markdown → HTML build step becomes part of the release**, feeding both the bundle and the download page.
- **`QTextBrowser` renders a limited HTML/CSS subset.** Topics will look plain, and the topic list and any search must be built by hand.
- **The `doc/` versus `docs/` ambiguity is finally closed**: `pyflic/help/` holds authored topics, `doc/` holds derived guides, `docs/` holds ADRs and nothing else.
