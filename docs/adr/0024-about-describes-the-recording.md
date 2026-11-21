---
status: accepted
---

# `about:` says what a recording is, and pyflic never analyses it

## Context

A FLIC recording is filed — in the lab's `labdata` system, or just a folder on
the share — one at a time, usually before any Project exists.  Whoever files
it knows things no file records: why it was run, who ran it, the flies' sex
and age, the protocol, and what each design factor and level means (`MBON10`
is which cross; `2` is 2% sucrose).  A catalogue, a colleague a year later, or
an agent reading the config cannot recover them from the DFM files.

`flic_config.yaml` is already where a recording's design lives (factors,
levels, chamber map), and `labdata` reads it there rather than asking pyflic
to read labdata's own records (labdata decision A).  The description belongs
beside the design.

## Decision

- **One optional top-level section, `about:`.**  Its fields are all optional:
  `title`, `purpose`, `experimenter`, `series`, `organism`, `sex` (a list),
  `age_at_start`, `protocol` (`name`, `version`, `link`), `notes`, and
  `factors`, which gives each design factor a `kind` (genotype, treatment,
  diet, condition, sex, other) and each level a few words of meaning
  (`pyflic.base.about`).  A key pyflic does not know is kept.
- **pyflic never reads it for an analysis.**  No number, cache key
  (`cache._config_hash` hashes a config holding `about:` without it) or
  "re-run needed" verdict depends on it.  A config without it loads, lints and
  caches exactly as before.
- **It belongs to the recording,** like `data_folder:` (ADR-0021): never under
  `global:`, never in a Project Design, never inherited, never compared across
  Members for errors.  Scaffolding a Member does not copy another Member's.
- **One warning, never an error:** a Project names factors or levels that its
  Members describe differently (`about.meaning_spread`), because the Combined
  Analysis pools same-named levels as one group.
- **`pyflic lint`** accepts the section and warns, never errors, on a malformed
  field, an unknown factor kind, a title over 60 characters, or a factor or
  level `experimental_design_factors` does not declare.
- **The Experiment Editor** (`pyflic-experiment`, `pyflic experiment`) edits
  it with the design and the chamber map, and nothing else.  Every other key
  the file holds — detection `params:`, `constants:`, Slices, the Selection,
  optogenetics, per-DFM overrides, `scripts:`, `data_folder:` — is the Config
  Editor's and is written back as read.  It writes `params: {}` when a
  standalone config has none, since the loader needs the section and empty is
  every default (ADR-0011).

## Consequences

- The fields follow `labdata`'s run record (`title`, `purpose`, `series`,
  `organism`, `biology.sex`, `biology.age_at_start`, `protocol`, factor
  `kind`, level `definition`), so labdata can fill a new run from them.
  pyflic does not depend on labdata and knows none of its files; `experimenter`
  is free text, which labdata matches to a lab member.
- The start date is not stored: it is in `Program.txt`, and the editor shows
  it from there.  Two places for one date would disagree eventually.
