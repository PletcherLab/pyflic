# Excluded chambers are authored in bulk from a sheet, and results that predate a declaration say so

Extends the **Excluded Chamber** concept (per-Member `remove_chambers.csv`,
active group named by the Design) with a bulk-authoring surface and a staleness
rule. Depends on ADR-0009 for the Batch keys the sheet's `project` column uses.

Adapted from PyTrackingAnalysis's ADR-0010, whose unit is a tracking region and
whose sidecar was new machinery. Here the per-Member declaration already exists
and is already stamped on every output, so only the authoring half and the stale
rule were needed.

## Context

An excluded chamber is a per-Member fact and belongs in that Member's own
`remove_chambers.csv` — that file is the audit trail, and the analysis already
reads and reports it. What was missing is the *authoring*: an experimenter who
has watched forty recordings has forty directories to edit, one row at a time,
in forty files. Work that tedious does not get done, and chambers that should
have been excluded quietly stay in the analysis.

The second problem is quieter. Declaring a chamber does nothing to results
already on disk. `feeding_summary.csv` from last week describes a chamber
population that today's declaration says was wrong, and the Hub reported it as
"analyzed: yes" — a true statement about a number that is not.

## Decision

- **An Exclusion Sheet is a `remove_chambers.csv` (or `.xlsx`) at a Batch root
  or a Project root** — one level *above* the per-Member files it writes into.
  Columns: `project`, `member`, `dfm`, `chamber`, `group`, `reason`. Headers are
  matched case- and punctuation-insensitively, and `project`, `group` and
  `reason` are optional (a Project-root sheet needs no project column; a missing
  group is `general`; a missing reason is `Undefined`). Same stem as the file it
  writes into, deliberately: location says which is which, and one word covers
  the whole concept.

- **It is a writer, never an overlay.** Applying it stamps rows into each
  Member's own `remove_chambers.csv`; nothing reads the sheet at analysis time.
  A sheet that is deleted, moved, or never applied changes no result, and every
  exclusion that reaches an analysis is visible in the file the analysis already
  reports from. An overlay would have created a second source of truth that only
  the Hub could see.

- **The standing declaration wins.** A chamber already declared is never
  rewritten, and a differing reason is reported as a **conflict** rather than
  applied. A Batch Run re-applies the sheet every time, so letting the sheet win
  would keep resetting reasons refined by hand. This makes re-application
  idempotent, which in turn means the reader must be able to see the note it
  itself wrote — reading only "which chambers" and discarding the note would
  report a conflict on every subsequent run, forever, on rows that are already
  exactly right.

- **Selecting reports; it never applies.** Choosing a batch folder logs that a
  sheet is there and how many rows it has. Writing happens only on an explicit
  button or as the first act of a Batch Run. Browsing to a colleague's batch
  folder must not silently rewrite eighty member directories.

- **A Batch Run previews the sheet and lets it be declined.** The preflight
  (ADR-0009) shows every row's outcome — applied / already declared / conflict /
  unknown project / unknown member / unknown chamber / incomplete — and one
  switch, checked by default. Declining skips the write for that run and never
  edits the sheet or any standing declaration. The preview runs the *same*
  evaluation as the write against an in-memory copy, so what the user was shown
  cannot disagree with what happens.

- **Rows are scoped to the Projects actually running.** A row naming a Project
  that is not checked is skipped and reported rather than written. Unchecking a
  Project means "do not touch this Project", and recursive discovery surfaces
  Projects the user may never have known were there. Scoping matches the row's
  *whole* path — `project=Sept, member=ProjB/rep1` names the same Member as
  `project=Sept/ProjB, member=rep1` — against the longest Project key that
  prefixes it, because the two columns are one path split where the author chose.
  At a Batch root a blank `project` cell scopes to nothing; the "blank means the
  root itself" contract belongs to a Project-root sheet, where no scoping is
  asked for at all.

- **A sheet cell cannot escape its root.** The sheet is a hand-edited
  spreadsheet and `os.path.join` honours both `../` and an absolute path.
  Without the guard, a stray cell writes a `remove_chambers.csv` anywhere on
  disk that happens to hold a `flic_config.yaml`.

- **Nothing about the sheet is fatal.** An unreadable sheet is reported and the
  run continues; a row naming a project, member or DFM that does not exist is
  counted in the summary; one unwritable member directory does not discard the
  others' declarations, and is named rather than reported as "nothing written" —
  the one thing an audit trail must never do.

- **A Member whose declaration is newer than its saved analysis reads "re-run
  needed", not "yes".** Compared by modification time against
  `analysis/feeding_summary.csv`. Deliberately not a stamp inside the outputs:
  the file's timestamp already answers the question, and adding a provenance key
  to every summary would change an on-disk format that R scripts and notebooks
  read. The rule is conservative in the right direction — touching the
  declaration without changing it costs one re-run, while the alternative
  reports numbers for a population nobody asked for.

## Consequences

- `remove_chambers.csv` now means two things by location: at a member root it is
  the declaration; at a Project or Batch root it is a sheet. The two are
  distinguishable by their headers, and `layout.file_recording` exempts the stem
  at every level, so filing can never sweep either into `data/` or
  `extra_files/`.
- A sheet is not required. Nothing in the app creates one, and a Batch with no
  sheet behaves exactly as before.
- The Hub's "Analyzed" column has a third value. A user who has never touched a
  declaration will never see it.
