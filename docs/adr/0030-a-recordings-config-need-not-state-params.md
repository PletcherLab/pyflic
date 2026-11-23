---
status: accepted
---

# A recording's config need not state `params:`

## Context

The loader refused a DFM that had no `params:` section, under `global:` or its
own ("DFM 3 must define a `params` section").  The rule predates the
Experiment Types: since ADR-0011 every analysis setting a config leaves out is
the type's default, so an empty `params: {}` states exactly what no section
does.  The Experiment Editor wrote one into every new config only to get past
the check.

That empty section then got in the way.  A recording filed in the lab's
catalogue keeps its config in the run's folder, and an analysis of it is a
Project whose design states the analysis settings.  The Project read the
member's `params: {}` as a member stating settings the design owns: *Write the
design into the members* offered to remove it, and declining made the Project
fail to load ("global.params is {} but the project design requires {...}").
On 2026-11-22 the lab head asked that a recording's config hold no software
settings: they belong in the Project, and may differ from one analysis to the
next.

## Decision

- **No `params:` section is the Experiment Type's defaults.**  The loader no
  longer requires one.
- **The Experiment Editor writes no `params:` under `global:`**, and drops an
  empty one it reads.  A setting the Config Editor wrote is kept as read.  A
  DFM's own `params:` holds what was recorded: its PI Direction (ADR-0029).
- **An empty `params:` (or `parameters:`) says nothing.**  Inside a Project it
  is not a member stating the design's settings, and when the member loads it
  does not hide them: the design's `params:` apply
  (`analysis_settings.says_nothing`, `stated_global`).

## Consequences

- A config written before this keeps its `params: {}` until the Experiment
  Editor saves it again, and it loads either way, alone or in a Project.
- The Config Editor still writes settings into a config, for a recording
  analysed on its own.  For a recording that belongs to a Project, the
  Project's design is where its settings go.
