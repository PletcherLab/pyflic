---
status: accepted
---

# The Config Editor derives from the Experiment Type rather than asking for everything

The Config Editor predates ADR-0007 and never caught up: it offered **Chamber
Size** and **Experiment Type** as two independent controls, listed `two_well`
and `single_well` among the types, always wrote `params.chamber_size`, and
never wrote `chamber_layout:`. That is the language ADR-0007 retired, and the
editor was emitting configs its own `pyflic lint` reports as needing
migration. The editor now defers to the selected `ExperimentType` for
everything the type owns, and asks the experimenter only for what is genuinely
free.

## Context

Three separate things follow from the Experiment Type, and the editor was
guessing at all three independently.

**The Chamber Layout.** `HedonicExperimentType` and
`ProgressiveRatioExperimentType` both fix `chamber_layout = "two_well"`; only a
Custom Experiment reads it from the yaml. A form that lets you pick *Hedonic*
and *single-well* offers a combination the loader rejects.

**The auto-filter cutoffs.** `resolve_constants()` merges the type's
`default_constants` *under* the yaml's `constants:` block, so for a typed
experiment a blank field does not mean "no filtering" — it means 20 / 13.0 /
150000.0. The form said "leave blank to skip", and its max-events placeholder
read `e.g. 150` against a real default of `150000.0`.

**What counts as valid.** `ExperimentType.validate()` already encodes the
type's requirements — Hedonic and Progressive Ratio both declare
`required_wells = ("A", "B")` — and the loader calls it. The editor validated
nothing at all and wrote whatever the form held.

## Decision

The Experiment Type combo is populated from `available_experiment_types()`, so
Custom Experiment / Hedonic Feeding / Progressive Ratio are the only choices
and the retired layout names are gone. Chamber Layout is shown but disabled
whenever the type fixes it. Threshold fields carry the type's
`default_constants` as *placeholder* text and write to `constants:` only when
the experimenter types a value. Validation is `ExperimentType.validate()` —
the loader's own function — surfaced as a count on each tab and a listing
before save.

On disk: a typed config writes neither `chamber_layout:` nor
`params.chamber_size`; a Custom Experiment writes `chamber_layout:`. The read
path is unchanged and still accepts legacy `params.chamber_size` and
`experiment_type: two_well`, so an old config opens, and re-saving migrates it.

## Considered options

**Materialising every effective value into the yaml** — writing 20 / 13.0 /
150000.0 explicitly on every save — was rejected, and the rejection is not
obvious, because reproducibility argues for it: a config that states its
cutoffs is a record of exactly what ran. It loses because it freezes each
config against whatever the type's defaults happened to be the day it was
written. Correcting a cutoff in `default_constants` would then reach no
existing experiment, and the type would own the value in name only. The
provenance argument is real but belongs to the outputs, not the input: the
report and `summary.txt` are where the effective values should be stamped.

**Leaving the editor's own validation rules** was rejected for the reason the
rules were absent in the first place — two implementations of "valid" drift,
and the one in the editor is the one nobody runs in CI.

## Consequences

- `config_editor.py` now imports `experiment_types`. The editor cannot be
  built without the registry, which is the point: adding a type reaches the
  editor with no further edits.
- A blank threshold field is no longer a null. Reading one requires knowing
  the type, so any future surface showing "the cutoffs in force" must resolve
  through `resolve_constants()` rather than reading `constants:` directly.
- `app-config-editor.md`'s "`chamber_size` is required and cannot be inferred"
  is false and is rewritten. The remaining eight help topics that still speak
  of `chamber_size` are pre-existing ADR-0007 drift, tracked separately.
