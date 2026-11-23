---
status: accepted
---

> Extended by ADR-0031: a Member may also read its recording's own config,
> named by its `flic_member.yaml`, with the data beside it.

# A Member can read its recording from a folder its config names

Amends ADR-0009: its "one recording, one Member" rule now compares data
locations too, its classifier reads a config that names one, and its filing
never touches a Member that does.  Amends ADR-0015 in one word: `Program.txt`
is read from the data location, which is `data/` unless the config names
another.

Originated in PyTrackingAnalysis (its ADR-0018), whose `layout.py` and
`project.py` this ports.  See `MIRRORED.md` for what was taken and what
differs.

## Context

The lab's `labdata` system gives every data collection a permanent, locked
folder: a **run** (`runs/flic/<year>/<run ID>/`, recording in `data/`),
checksummed a week after registration, so nothing under `data/` may change
after that.  A study's analysis lives elsewhere
(`studies/<study>/analyses/<analysis ID>/`) and corresponds to a Project here.
Several studies analyse the same run with different settings.

pyflic could not do that.  A Project's Members are its subdirectories
(ADR-0005); each Member read its recording from its own `data/`; each wrote its
output into its own folder.  The only way to pool runs was to copy their raw
data into the Project, and `labdata` forbids raw-data copies.

pyflic had a key for this once.  `data_dir:` stopped being read on 2026-04-16
(commit `5769e9a`), and `pyflic lint` has warned since that it is unused.

## Decision

- **One optional key, `data_folder:`, at the top of a Member's
  `flic_config.yaml`.**  It names the folder that holds what `data/` holds
  today: the DFM CSVs, and the `Program.txt`, `Messages.csv` and
  `Baselines.csv` PyMCU exports beside them.  It belongs to that config alone:
  never under `global:` (where `pyflic lint` reports it as an error, since it
  is not read there), never in a Project's `design:`, never compared across
  Members, never inherited.  Scaffolding a Member copies only `dfms:`, and the
  Hub's **Copy config from…** drops the key, because it names *that* Member's
  recording.  A blank value is no key.  It works the same for a standalone
  Experiment Directory.

- **Not `data_dir:`.**  Old configs on the lab drive still carry `data_dir:`
  set to absolute paths on someone's own PC.  Honouring that key again would
  send their reads to the wrong place without a word, so it stays dead.  Its
  lint warning stays word for word for a config without `data_folder:`; beside
  that key, where "always read from experiment_dir/data/" would be untrue, it
  says data is read from the folder `data_folder:` names.  The new meaning gets
  a new name.

- **Relative to the Member folder, forward slashes.**
  `../../../../../runs/flic/2026/R-20261020-asmith-01/data` is what labdata
  writes.  Resolution is the operating system's own (`realpath` of the joined
  path, so `..` is taken after any symlink, as `open()` would take it).
  Windows accepts backslashes natively.  An absolute value works, but
  `pyflic lint` warns that it is not portable — as it does for a backslash —
  and a warning never blocks a load.

- **No key, no change.**  Other labs use pyflic.  A config whose text never
  mentions `data_folder` is not even parsed by the resolver; every code path,
  output, message and cache key for it is what it was.  So the data-location
  lines — in the load message, `summary.txt`, the Member's report and the
  Project Report's member table — appear only when a Member uses the key.
  `Experiment.data_dir`, which nothing used to set, is now set by every load,
  and `summary.txt` still says nothing about it for a Member reading `data/`.

- **One resolver: `layout.data_location`.**  It answers "where is this
  Member's recording?" from the Member folder and its config: today's
  exact-case `data/` (`layout.data_dir`), or the resolved `data_folder:` with
  any problem the loader would refuse it for.  The loader
  (`load_experiment_yaml`, and through it the Opto Program, Messages,
  Baselines and the DFM-ID check), the disk cache, the classifier
  (`layout.classify`), `layout.members_in` (and through it Batch discovery,
  the preflight and the Hub's Project panel), `Project`,
  `has_experiment_data`, `dfm_ids_in_data`, `pyflic lint`, the Project
  Report's program lookup and the Config Editor all call it, so the
  classifier and the loader cannot disagree — the rule ADR-0009 already
  stated.  The loader resolves before it reads anything, so a reference it
  refuses reads nothing.

- **A referenced recording is only ever read.**  Everything written for the
  Member stays in the Member folder — `analysis/`, `qc/`, `slices/<name>/`,
  the report, `remove_chambers.csv`, `.pyflic_cache/` — or in the Project
  folder, exactly where it went before.  Filing refuses a referencing Member
  and moves nothing.  Scaffolding refuses any folder with a config before it
  creates `data/`, so a referencing Member never gets an empty one.

- **Conflicts are errors, not guesses.**  A referencing Member is blocked —
  and the loader raises `DataLocationError`, naming the key, the value as
  written and the resolved path — when the folder does not exist or is a file
  (**no recording**); when the Member folder also holds a recording of its own
  (**ambiguous**: which is the experiment?); or when the named folder holds the
  Member folder itself (**ambiguous**: the analysis would be written inside the
  recording).  "A recording of its own" is pyflic's recording, not every
  file: in the Member's own `data/`, anything shaped like part of one, each on
  its own — a DFM export in any spelling, or a lone `Program.txt`,
  `Messages.csv` or `Baselines.csv`, since which program goes with the
  experiment is not pyflic's guess to make; at
  its root, only what the ordinary classifier counts as a loose recording, the
  DFM exports the loader would find.  So a `units.csv` or `notes.txt` beside
  the config is just a file.  Hidden files, Excel lock files, the Exclusion
  Sheet and the named folder itself never count, and an empty `data/` is not a
  conflict.  A referencing Member is never an **Unfiled Recording**: DFM files
  in the named folder are where the loader looks, and DFM files beside the key
  are a conflict to settle by hand.

- **One recording, one Member — by data location too.**  ADR-0009 drops a
  symlinked copy of a Member by real path.  Two Members whose data locations
  resolve to one folder (`normcase(realpath(...))`, for the case-insensitive
  share) are now one as well: the first in sorted order is kept and the other
  is reported by name — a Blocked Member with the new status **duplicate** in
  `members_in`, so the preflight and the Batch Run summary list it with the
  other Blocked Members, and `Project.duplicates` plus a Project warning,
  which the Hub and the Project Report show.  `Project.__init__` and
  `members_in` share the rule (`layout.DataLocations`), so they cannot disagree
  about who the duplicate is.  The rule applies only when one of the two uses
  the key: two plain Members whose `data/` folders are one folder by symlink
  are left exactly as they were, because "no key, no change" outranks tidying
  an old corner.  Across Projects nothing is compared: two Projects reading
  one recording with different settings is the point.

- **The cache follows the recording.**  The data half of a cached summary's
  key hashes the files in the folder the loader read, and, for a referenced
  folder, where that folder is.  Without it a referencing Member hashed a
  `data/` that does not exist, and a recording replaced under an unchanged
  config was served from the cache.  A plain Member's key is computed exactly
  as before, so its cached summaries stay valid and the cache version is not
  bumped.

- **The Config Editor shows the key.**  A **Data Folder** field, first on the
  Experiment Settings card, with a tooltip and a help button.  A Project's
  design never locks it, because it is not a `global:` key.  Blank writes no
  key; clearing it removes one.

Rejected:

- **Honouring `data_dir:` again.**  See above: it would misdirect every old
  config that still carries one.
- **A Project-level data root**, or a registry mapping Members to
  recordings.  It would make the Design carry per-Member facts and split one
  Member's definition across two files.
- **Symlinking `data/`.**  Symlinks on the lab's Windows share need
  privileges most users lack, and a symlinked `data/` is indistinguishable
  from a local one — filing would happily move a loose DFM file into the
  locked run.
- **Honouring the key by changing `layout.data_dir`.**  That function is also
  where filing puts a recording, and a referencing Member must never be filed
  into, so "where is the recording" and "where would filing put one" stay two
  functions.
- **The strict duplicate rule**, comparing every pair of Members.  It would
  change what a Project without the key analyses; it is a one-line change in
  `DataLocations.claim` if it is ever wanted.

## Consequences

- A Project no longer always travels with its data.  Identity still does —
  Members are still the configured subdirectories — but a copy of the
  analysis folder without the `runs/` tree finds its referencing Members
  blocked, each naming the path it resolved.  Moving the managed tree as a
  whole keeps every relative reference working.
- `layout.classify` now parses YAML for a config that mentions `data_folder`;
  every other config is still only listed.  A config that mentions the key
  but cannot be parsed counts as having none, so the classifier keeps today's
  answer and the loader reports the broken YAML.
- With an alternate config (`load_experiment_yaml(..., config_name=...)`) the
  loader reads the key from that file, and the cache's data half follows it;
  the classifier, the Project and Batch discovery always read
  `flic_config.yaml`.  The Hub only ever loads the canonical file.
- On Windows, `realpath` may return a path longer than 260 characters with a
  `\\?\` prefix, or a mapped drive in `\\server\share` form; that is the
  resolved path the messages, `summary.txt` and the reports then show.  It
  still reads correctly.  Untested: the suite runs on Linux only.
- `layout.DUPLICATE` is a new Blocked Member status with no fix button — the
  fix is deciding which Member should read that recording.
- `.pyflic_cache/` stays in the Member folder.  labdata's layout rule against
  caches in study folders is a separate question.
- The retired key's lint warning has two wordings.  A config without
  `data_folder:` (or with a blank one) gets the one it always got, so nothing
  about an old config's messages changes.  A config with both keys is told that
  data is read from the folder `data_folder:` names, and the warning about
  `data_dir:` stays a warning.
- `layout.py` and `hub.py` are mirrored files; `MIRRORED.md` records the port.
