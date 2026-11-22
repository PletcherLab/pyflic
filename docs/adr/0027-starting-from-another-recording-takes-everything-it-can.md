---
status: accepted
---

> Amended by ADR-0028: each DFM's PI Direction, the side well A is on, is taken
> too, where the other recording states one.  A DFM's other overrides,
> optogenetics and notes are still never taken.

# Starting from another recording takes everything it can, and asks before replacing

Amends ADR-0026's *Start from another recording…*, which took only the
**shared About**, and the design and chamber map only for a recording that had
neither yet.  Scaffolding a Member and the Hub's *Copy config from…* are
unchanged: they still take the shared About alone.

## Context

ADR-0026 let a new recording start from an earlier one, on the reasoning that
replicates share most of their description.  It held back the title, the
purpose and the notes, as one recording's alone, and it left a design stated
here untouched, taking the other one's only into a form with no factors and
no chamber assignments.

In use (2026-11-22) that read as an action that did nothing.  A folder already
holding an old `flic_config.yaml` kept its old design and chamber map, and the
other recording's factor kinds and level meanings, limited to the factors this
one declared, mostly did not appear.  The title, purpose and notes, which a
replicate repeats with a word changed, had to be typed again.  The lab head
asked for the action to fill in as much as possible, replacing what is there
after a warning.

`Program.txt` came up at the same time: the About's protocol fields were
typed by hand, though the file records what the MCU ran.  It holds no protocol
name and no version line, though: only the program, and in its shape which
PyMCU wrote it.

## Decision

- **Everything the form shows is taken that this recording can take.**
  - **The whole About but `run:`**: the title, the purpose and the notes, the
    shared About, and any key pyflic does not know.  `run:` is the catalogue's
    id of that recording and is never taken (ADR-0026).  A field the other
    recording does not state, or states as a blank, stays as it is here.
  - **The design** (type, layout, well names, factors, control level), unless
    a Project design governs this recording, the other states none, or its type
    is one this recording cannot use (said, as before).
  - **The chamber map**, for the DFMs this recording's data holds (its tabs,
    when no data is found), with the chamber groups.  Taken with the design, or
    without it when the other recording's factors are, by name and order, the
    ones this recording declares: so a Member takes another Member's map.  When
    the design taken changes the factors, a DFM here that the other does not
    map has its chambers cleared, since its levels would name nothing.  A DFM's
    overrides, optogenetics and notes are never taken.
- **One question before anything stated here is replaced.**  It lists each
  thing: a title, purpose, notes, experimenter, series, organism, sex, age or
  protocol that says something else; a factor kind or level meaning; the
  type, a single-well layout, a well name, the control level or the factors;
  the chambers of a DFM replaced or cleared.  **No changes nothing**, the About
  included; ADR-0026's No took the shared About and kept the design.  Nothing
  stated here, nothing asked.  Nothing is written until the person saves.
- **`Program.txt` beside the protocol.**  The About tab shows, under Protocol,
  what each DFM ran in order and for how long (`opto_program.program_summary`)
  and which PyMCU wrote the export, as far as its shape tells
  (`opto_program.pymcu_version`: `PyMCU 3.x` or `PyMCU 4.x`; the simple export
  claims none).  It is read each time, like the start, and never saved.  **Use
  for protocol** copies them into Protocol and Version where those are blank;
  a field with text in it is left alone.
- **`Program.txt` beside the config.**  With no `data_folder:`, the editor
  looks for it beside the config when the data location has none, as it
  already did for DFM files, so an instrument folder with a config put in it
  shows its start and program.  This is the editor's display only; the loader
  still reads the data location.
- **`about.py` is unchanged.**  The mirrored module (`MIRRORED.md`) keeps
  `SHARED_KEYS` and `start_from` as the contract defines them; the editor adds
  the rest on top of them.

Rejected:

- **Taking the program summary into `about:` on its own.**  labdata files
  `protocol` as the lab protocol's name and version.  A generated line is a
  suggestion for a person to edit, not a name, and PyMCU's version is not the
  protocol's; it goes in only when asked.
- **A third answer, "take the About and keep the design".**  It was ADR-0026's
  No, and is what made the action look broken.  A person who wants the design
  kept answers No and copies what they want by hand, or edits after Yes.
- **Taking the analysis settings** (`params:`, `constants:`, Slices, a DFM's
  overrides).  The Experiment Editor shows none of them, and the Hub's *Copy
  config from…* is the action for those.

## Consequences

- This departs from labdata's `about:` contract (`P10-about-contract.md`, item
  8 and test A6), which has the action leave the title, purpose and notes as
  they are.  labdata reads the config and is not affected; PyTrackingAnalysis's
  equivalent action still follows the contract until it is told otherwise.
- A replicate's title starts as the earlier one's.  Nothing marks it as copied:
  the person edits it, as they edit "replicate 1" to "replicate 2".
- Scaffolding and *Copy config from…* now take less than *Start from another
  recording…*.  Whether they should take the title, purpose and notes too is
  open.
