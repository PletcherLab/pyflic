# Batch discovery is recursive, a Project's children are Members, and a Batch Run is confirmed in a preflight

Supersedes the structural half of ADR-0006 (Projects need not be immediate
children of the Batch). Renames the Project→child relationship established by
ADR-0005.

Originated in PyTrackingAnalysis (its ADR-0011). This is the pyflic
counterpart; see `MIRRORED.md` for what was taken and what deliberately
diverges.

## Context

Three problems, one shape.

**Experimenters do not keep Projects in one flat folder.** They keep
`Sept2026/ProjA`, `Archive/2025/ProjC`, a `pilot/` beside a `final/` — and
ADR-0006's Batch is one `os.listdir`, so pointing the Hub at the folder that
actually holds the work finds nothing. Splitting a tree into flat batch folders
to satisfy the tool is the wrong direction of accommodation.

**A FLIC recording arrives as loose files.** The rig writes `DFM<id>_<n>.csv`
wherever the acquisition PC was pointed, and the loader reads `experiment_dir/
data` and nothing else. A folder full of a perfectly good recording is
therefore invisible to its Project — not broken, not warned about, simply
absent from `member_names` — and stays that way until somebody moves the files
by hand. The same is true of a folder whose config was never scaffolded: it
fails at load, an hour into an unattended run.

**"Replicate" was the wrong word.** pyflic borrowed the sibling app's term for
a Project's children along with its structure. But a PyTrackingAnalysis Project
holds replicates — the same experiment repeated, where pooling is a bigger n —
and a pyflic Project holds *different experiments addressing one question*: a
dose series, a genotype panel, a pilot beside its follow-up. Every surface that
said "replicate" was making a claim about the data that is not true, and the
Combined Analysis has always kept `Experiment` as a column precisely because
those rows are not interchangeable.

## Decision

- **A Project's children are Members.** `Project.member_names`,
  `member_dir`, `load_member`, `member_status`, `scaffold_member`, and every
  user-facing string. The old spellings remain as thin aliases — notebooks
  written against pyflic before this exist and breaking them buys nothing. The
  word *replicate* survives in exactly one place: the Exclusion Sheet's accepted
  header spellings, for sheets already written.

- **A Project inside a Batch is not called a member.** It is a
  `BatchProject`. "A member with four members" is a sentence this codebase must
  not be able to write, and the collision is the sibling app's, not ours to
  inherit.

- **Discovery is recursive and prunes at each Project.** A **Batch** is a
  directory with at least one Project anywhere beneath it. The walk descends
  until it finds one — `project.yaml` plus at least one Member the run could
  *use* — and never looks inside, because a Project's subdirectories are its
  Members by definition (ADR-0005). An archived copy carrying its own
  `project.yaml` inside a Project therefore cannot become a second target, and
  no Member is analyzed twice in one run. Grouping folders are transparent. The
  walk does not follow symlinks, skips dot-directories, ignores unreadable ones,
  and stops after 20 000 directories so a mis-clicked home directory cannot hang
  the Hub.

- **The prune bar is a *usable* Member, not "evidence of a recording".** A
  grouping folder holding one `template/flic_config.yaml` looks exactly like a
  Project whose Members are all blocked, and pruning there would hide every real
  Project beneath it. A `project.yaml` directory with experiment-shaped children
  but nothing usable is descended into first and only called a Project if no
  real Project turns up below. The same rule applies to a stray
  `flic_config.yaml` at a grouping level: stopping unconditionally at a config
  marker hid every Project beneath one file.

- **A Project is keyed by its path relative to the Batch root.**
  `Sept2026/ProjA`; a top-level Project is still just `ProjA`, so every existing
  `batch.yaml`, Exclusion Sheet row, and `run_batch(project_names=...)` call
  keeps working untouched. Leaf names were rejected: two `ProjA`s under
  different parents collide, and disambiguating only on collision makes a key
  change when an unrelated Project is added elsewhere in the tree — silently
  invalidating a designation and every sheet row that named it. A key that
  escapes its root (`../`, an absolute path) is refused, not resolved.

- **Blocked is a property of the Member, not the Project.** A **Blocked
  Member** is one a run cannot use: an **Unfiled Recording** (DFM CSVs at the
  root rather than in `data/`), a recording with no `flic_config.yaml`, a
  configured directory with no DFM CSV at all, an ambiguous one, or one nobody
  can list. A Project with four healthy Members and one blocked one runs the
  four. Blocked Members are reported when the Project is selected, before the
  run, and again in its summary; a run is **never refused** because of one — a
  stale folder must not stop ten Projects at 2am. A Project with *no* usable
  Member starts unchecked, since it can only produce a failure.

- **Blocked status is decided with the loader's own test.** The same
  `data/DFM<id>_*.csv` spellings `dfm.py` globs for, the same exact-case
  `data/` that `load_experiment_yaml` builds, the same exact-case
  `flic_config.yaml` that `is_experiment_dir` asks for. A classifier that says
  "healthy" where the loader says "no DFMs to load" is worse than no classifier:
  it moves the failure from the preflight, where someone is looking, into hour
  three of an unattended run. Layout only — no YAML parsed, no data read — so
  it is cheap enough to run over every subdirectory of every Project in a Batch.

- **Filing is an allowlist, and the sidecars are exempt by rule.** Filing an
  Unfiled Recording moves `DFM*.csv` — exactly what the loader globs for — into
  `data/`, and every other loose file into `extra_files/`. **Every `.yaml`/`.yml`
  stays at the member root**, and so does `remove_chambers.csv`: at an
  experiment root those are configuration or declaration, never data. That
  protects `flic_config.yaml` (moving it un-makes the Experiment Directory) and
  the Exclusion Sheet (moving it silently returns excluded chambers to the
  analysis — the precise failure ADR-0010 exists to prevent), and it protects
  any sidecar added later. Subdirectories are never touched.

- **Filing never overwrites and never guesses.** A destination that already
  exists is skipped and reported, leaving both files where they are. The same
  DFM id present both loose and in `data/` refuses to file at all and says
  which id is doubled. Unlike the sibling app, "several recordings here" is
  *not* ambiguous — a FLIC experiment is many files, one or more per DFM — so
  the ambiguity is never "which file" but "which copy".

- **Run batch opens a preflight, always.** One modal listing the discovered
  Projects with their relative-path keys, usable-member counts, and
  per-Member block reasons; a reason-matched action on each (file the recording;
  "Member configs…" for a missing config — reusing the one design-aware
  scaffolding path rather than writing a second); the Exclusion Sheet preview;
  then Run or Cancel. Shown even when nothing is wrong, because with recursive
  discovery the target list is no longer obvious from the folder you picked, and
  that list is the one thing no other surface states.

- **A Project repaired inside the preflight joins the run.** Check state is
  derived every rebuild from what the user actually said plus what the Project
  can do *now*, never from the previous check column: filing is what *makes* a
  Project runnable, so re-deriving "unchecked" from the pre-repair state would
  exclude the very Project just fixed.

- **The walk is cached per selection.** It runs when a batch folder is chosen
  and is invalidated by choosing another, filing a recording, scaffolding a
  config, or finishing a run; an explicit **Rescan** covers changes made outside
  the app. `is_batch_dir` is `bool(discover(...)["projects"])` — deliberately the
  same predicate the table and the run use, because a short-circuit asking
  `is_project_dir` at the root disagrees with the walk on every batch folder
  carrying a stray or legacy `project.yaml`, and shows an empty, dead panel.

- **Only the selected Batch's `batch.yaml` governs.** A nested grouping folder
  may be a Batch in its own right and carry its own designation; it is ignored
  and named in the run log. Resolution is already three steps (batch
  `project_scripts:` → the Project's `scripts:` → built-ins) and a fourth that
  depended on where the user clicked would be unmemorable.

- **No designation means each Project runs its OWN default script.** Every
  `project.yaml` is created with one, so there is no built-in fallback: a
  Project whose `scripts:` is empty does not run, and says so. `batch.yaml` is a
  lazy marker — choosing that default never creates the file.

- **Two symlinked Member directories are one Member.** Both
  `Project.__init__` and `layout.members_in` de-duplicate by real path.
  Without it the same chambers were analyzed twice and stacked into the Combined
  Analysis under two labels.

## Consequences

- `batch.yaml` files and Exclusion Sheets written after this change may contain
  path-shaped Project keys; older flat ones keep working unchanged, and a key is
  stable as long as the folder is not moved within the Batch.
- `extra_files/` starts appearing in member directories. It is inert — nothing
  reads it — and exists so filing never has to decide that an unrecognised file
  is disposable.
- A Batch can now contain Batches. Whichever one is selected is the one that
  runs; nesting has no other meaning.
- The run summary gains a per-Project member ratio (`3/5`) and a blocked list,
  so "succeeded" can no longer be read as "analyzed everything".
- The Project panel lists folders that are not Members yet. That is the point:
  they were previously invisible in the one place that offers the fix.
