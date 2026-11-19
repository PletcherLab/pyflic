---
status: accepted
---

# Results record the settings they were computed with, and a changed setting reads "re-run needed"

Extends ADR-0010's stale rule from the exclusion declaration to the settings in
the Design and the Member's config.  Leaves ADR-0020's Slice record in charge of
each Slice's own definition.

## Context

ADR-0010 made a Member whose `remove_chambers.csv` is newer than its saved
`analysis/feeding_summary.csv` read "re-run needed" instead of "analyzed: yes".
Nothing did the same for the settings the numbers come from.  Raise the
Design's `feeding_threshold`, change a constant, or reassign a chamber in a
Member's config, and every Member analysed before the change still read "yes"
in the Hub and in the Project Report.  The date beside it was true and the
numbers were not, which is exactly what ADR-0010 set out to stop.

ADR-0010's mtime comparison cannot be reused as it stands.  `project.yaml` and
`flic_config.yaml` are rewritten by saves that change no number: a Selection
tick in the Hub writes `global.omit:` (ADR-0018), the Script Editor writes
`scripts:`, the Design editor writes the notes, and either editor may write the
same values back.  Judged by file times, every one of those would mark every
Member stale, and a status that cries wolf on most saves gets ignored.

## Decision

- **Every `feeding_summary.csv` an analysis writes in its own `analysis/`
  folder gets `analysis_settings.json` beside it.**  It holds the settings the
  load was made from, as the files stated them: the Design's `global:` merged
  with the Member's own (the Member's keys win, as at load), the `dfms:` block
  keyed by DFM id, and `data_folder:`.  It is taken from the YAML as read,
  before the loader resolves anything (the Opto Program fills in paired
  chambers), so the comparison is like for like.  A summary written to any
  other path — the retired ranged files — leaves no record.

- **The status compares the record with the files as they stand.**
  `Project.member_status` reads `project.yaml` and the Member's
  `flic_config.yaml` from disk at each call, behind a modification-time cache,
  so a save from the Design editor, the Config Editor, or a text editor is
  seen at the next refresh with no reload.  Any difference makes the Member
  stale, and `stale_reasons` names each one: `params.feeding_threshold
  changed: 20 → 25`, `DFM 2 chambers.3 changed: Exp → Ctrl`.

- **What changes no number is left out of the comparison.**  `omit:` (which
  outputs a run writes, not how — ADR-0018), and `slices:` and
  `slice_tolerance_min:`, which each Slice's `slice.json` already answers for
  (ADR-0020).  Changing a Slice therefore stales that Slice, not the whole
  recording.  Everything else in `global:` counts, including keys that only
  rename things (`well_names`, `facet_labels`): they change what the outputs
  say, and a key left out wrongly would hide a real change.  Numbers compare
  as numbers, so `20` and `20.0` are one setting, and the `dfms:` list and
  mapping forms compare equal.

- **The record is a separate file, not a column or a header.**  ADR-0010
  declined to stamp provenance into `feeding_summary.csv` because R scripts and
  notebooks read that format.  A file beside it changes nothing they read.

- **Results from before records existed are judged by file times.**  With no
  `analysis_settings.json`, a `flic_config.yaml` or `project.yaml` modified
  after the summary reads stale, with a reason that says the analysis predates
  the record.  Conservative in the right direction, as ADR-0010 is: a save
  that changed nothing costs one re-run, after which the record decides.

- **"Skip already-analyzed" no longer skips a stale Member.**  `run_all`'s
  `skip_analyzed` (and the `run_all_analyses` step's) skips only results that
  are current, and logs each re-run's reasons.  Skipping a Member whose results
  are known to be wrong was never what the option was for.

- **The Hub re-judges on every save.**  The Design editor already reloaded the
  Project.  The Config Editor now emits `saved`, and the Hub refreshes on it.
  The Analyzed column's "re-run needed" cell lists the reasons in its tooltip.

## Consequences

- One new file in each `analysis/` folder (and each `slices/<name>/analysis/`).
  It is JSON, never pooled, and nothing but the stale rule reads it.
- An edit to a lab Experiment Type's own definition (its default constants,
  modules or facets) is not compared: the type is a separate file
  (ADR-0019), and only the Design and config are recorded.  A changed type
  default that the Design does not state is not seen.
- The Combined Analysis still pools a stale Member's summary, as it always
  has for ADR-0010's staleness; the Project Report's member table says
  "re-run needed" beside it.
