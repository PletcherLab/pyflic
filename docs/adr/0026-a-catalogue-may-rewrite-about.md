---
status: accepted
---

# A catalogue may rewrite `about:`, and pyflic keeps what it does not edit

Amends ADR-0024 in one sentence: "Scaffolding a Member does not copy another
Member's" no longer holds.  A scaffolded Member now takes the first Member's
**shared About**.  The rest of ADR-0024 stands: the section is optional,
belongs to the recording, and is never read for an analysis.

## Context

ADR-0024 gave a recording a place to say what it is.  On 2026-11-21 the lab
decided how its catalogue, `labdata`, and the analysis packages meet (labdata
decisions AH to AL; the contract both packages build to is labdata's
`docs/package_briefs/P10-about-contract.md`):

- **Describe, then file.**  A person describes a recording in its own folder,
  usually before any Project exists.  The catalogue files it from that folder
  and types nothing again.
- **After filing, the catalogue's record is the truth,** and the catalogue
  rewrites `about:` to match it, in the filed copy of the config and in every
  copy a Project holds as a Member.  It writes `run:`, the recording's id
  there, and inside a level's mapping `term:`, `label:` and `role:`, its
  registry's name for the level.
- **A new recording starts from an earlier one.**  Replicates share most of
  their description, so nobody should type it twice.

So `about:` can change on disk while a pyflic window has the file open, and it
holds keys pyflic neither wrote nor reads.  As ADR-0024 left it, pyflic lost
them.  The Experiment Editor rebuilt `about.factors` from its tables as plain
text and had no place for `run:`.  The Config Editor wrote back the copy of
the section it read when the file was opened.  The Hub's *Copy config from…*
copied one recording's `run:` and title into another.

## Decision

- **The shape gains what a catalogue writes.**  `about.run` is text.  A level
  is a line of text, or a mapping whose `definition:` is that text and which
  may carry `term:`, `label:`, `role:` and any other key.  The section is
  written in the order `run`, `title`, `purpose`, `experimenter`, `series`,
  `organism`, `sex`, `age_at_start`, `protocol`, `notes`, `factors`, then any
  key pyflic does not know, as it was (`about.KEY_ORDER`).

- **pyflic never writes `run:`, `term:`, `label:` or `role:`.**  It shows
  `run:` ("Filed as …" on the Experiment Editor's About tab), keeps all four,
  and edits only a level's `definition` (`about.with_meaning`): text stays
  text, a mapping keeps its other keys, and a mapping left with no definition
  keeps the rest.

- **Every writer keeps what it does not edit.**  That covers a key inside
  `about:` pyflic does not know, a factor's keys beside `kind:` and `levels:`,
  and what the section says of a factor or level the design did not declare
  when the file was read.  The Experiment Editor has no row for those, so it
  writes them back as read, and they stay warned of.  A level renamed or
  removed in the editor still loses its description: that is the person's
  edit.

- **A writer that does not edit `about:` writes back the section now on
  disk,** not the one it read when the file was opened.  The Config Editor
  reads it again at save, from the file it writes (`project.about_on_disk`).
  The Script Editor, a Selection tick and `project.adopt_design` re-read the
  whole file at save, as they always did.  A file that is gone or no longer
  parses keeps the section that was loaded.

- **The Experiment Editor never saves silently over a file that changed.**  It
  remembers the file's bytes as read.  When they differ at save, it says so
  and offers to reload, to overwrite or to cancel.  Overwriting writes what
  the editor edits (the About fields, the design, the chambers) over the file
  as it then is, so `run:`, a level's `term:`, `params:`, `scripts:` and every
  other key another program wrote stay as that program left them.

- **A recording starts from an earlier one.**  *Start from another
  recording…*, on the About tab and in the File menu, fills the form with the
  other config's shared About (`about.SHARED_KEYS`, `about.start_from`):
  experimenter, series, organism, sex, age at start, protocol, and each
  factor's kind and each level's whole entry.  It is limited to the factors
  and levels this recording declares when it declares any.  The title, the
  purpose, the notes and `run:` are one recording's alone and are never
  taken.  A recording with no design factors and no chamber assignments yet,
  and no Project design over it, also takes the other one's design (type,
  layout, well names, factors, control level) and, for the DFMs its own data
  holds, its chamber map; it asks first when that would replace a type, a
  layout, a well name or a control level stated here.  Nothing is written
  until the person saves.

- **Scaffolding a Member copies the first Member's shared About,** limited to
  the factors and levels the Design declares, and read from the first
  Member's file as it then is (`Project.shared_about_of`).  The Hub's *Copy
  config from…* takes the same part of the config it copies.

- **`run:` is never copied to another recording.**  Not by scaffolding, not by
  *Copy config from…*, not by *Start from another recording…*, and not by
  Save As into another folder (`project.without_run`), where a config written
  over keeps the `run:` it already had.

- **The Hub opens the Experiment Editor** on the loaded experiment or Member,
  beside the Config Editor in the Tools panel, and refreshes when it saves.

- **Still warnings, never errors.**  `pyflic lint` says nothing of `run:`,
  `term:`, `label:`, `role:` or a key it does not know.  Everything ADR-0024
  lists still warns, and nothing in `about:` stops a load, a save or a run.

- **pyflic stays unaware of the catalogue.**  It reads none of its files and
  gains no dependency.  The cache key is unchanged: a config holding `about:`
  is hashed without it.

Rejected:

- **Overwriting the whole file with the editor's copy.**  The plain meaning of
  "overwrite" would put back the `about:`, `params:` and `scripts:` the editor
  read earlier, and so undo the catalogue's filing and any save made in the
  Config Editor meanwhile.  The editor keeps what it does not edit; that rule
  has no exception for a person in a hurry.
- **Dropping what the design does not declare.**  The Experiment Editor used
  to write `about.factors` for the declared factors only.  A catalogue may
  describe a level a Project's Design leaves out, and deleting its entry on a
  save nobody made to it is the loss this ADR exists to stop.
- **Limiting *Start from another recording…* to nothing when this recording
  declares no factors.**  The contract takes the other recording's factors
  then.  When the design is not taken with them, they are kept, warned of as
  undeclared, and shown as soon as the person declares the factor.

## Consequences

- `pyflic/base/about.py` is the origin of a module PyTrackingAnalysis keeps
  verbatim (`MIRRORED.md`).  Change it here first, then copy.
- A Member's `about:` is no longer only what was typed in that Member: a
  scaffolded one starts with the first Member's shared part.  It is a copy
  made once, never inherited, and each Member's is its own from then on.
- "Overwrite" in the Experiment Editor is narrower than the word suggests, on
  purpose.  The question says what it will write.  After it, the form is read
  again from the file written.
- Only `about:` is read again by the Config Editor.  The other keys it has no
  field for (`scripts:`, the Selection's `omit:`) and the ones it edits are
  still written from the copy it loaded, with no question, as before.  Two
  editors open on one file can still undo each other outside `about:`.
- An analysis from before ADR-0022's records is judged by file times, so a
  catalogue rewriting `about:` makes such a Member read "re-run needed" once.
  pyflic cannot tell what changed in a file it has no record for.  Every
  analysis since has a record, and for those a rewritten `about:` changes
  nothing.
- A recording whose data holds no DFM file yet takes the chamber map for the
  DFM tabs the person has added; with neither, the design is taken alone.
- The chamber map taken is what the Chambers tab shows: each chamber's
  levels and, for a Progressive Ratio type, the chamber groups.  A DFM's
  parameter overrides, its optogenetics and its notes are the other
  recording's own.
