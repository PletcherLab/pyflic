---
status: accepted
---

> Amended by ADR-0029: every two-well DFM states its side, with no *inherit*.
> A side filled in with the experiment's is not one stated here, and a side
> the other recording's `global:` states is taken too.

# Start from another recording takes each DFM's PI Direction

Amends ADR-0027, under which *Start from another recording…* never took a
DFM's overrides, optogenetics or notes.

## Context

ADR-0027 left a DFM's parameter overrides behind because the Experiment Editor
showed none of them; the Hub's *Copy config from…* was the action for those.
On 2026-11-22 the Experiment Editor began to show one: each two-well DFM's
**PI Direction**, the side well A is on (`params: {pi_direction: ...}` on the
DFM).

The same day, two replicate FLIC runs pooled in one Project disagreed: one set
`pi_direction: right` on DFMs 3, 4, 7 and 8 and the other set nothing, though
their chamber maps were identical.  A replicate started from the other takes
the map and, until now, not the sides, so it silently read well A on the left
for every DFM.  The lab head asked that the side be carried over.

Which side well A is on describes how a DFM was set up, not how it is analysed.
pyflic already lets it vary per DFM inside a Project (`PHYSICAL_DFM_KEYS`,
ADR-0005), and replicates on one rig share it.

## Decision

- **Each DFM's PI Direction is taken** where the other recording states one
  (`pi_direction`, or the legacy `pi_multiplier`) for a DFM this recording's
  data holds (its tabs, when no data is found).  It is taken with or without
  the chamber map, since it does not depend on the factors.
- **A DFM the other recording states none for keeps its own here**, as an
  About field the other leaves blank does (ADR-0027).
- **None is taken on a one-well layout**, whose chambers have no side.
- **A side stated here that would change is listed in the one question**
  (`PI Direction: DFM 3 left → right`), and No changes nothing.
- **Nothing else of a DFM's is taken**: its other overrides, its optogenetics
  and its notes are still that recording's own.

## Consequences

- A replicate made with *Start from another recording…* reads its PI
  Directions from the earlier recording.  A person who rewired a DFM changes
  that DFM's choice after taking.
- The Experiment Editor now writes per-DFM `params:` where it did not before.
  The loader needs a `params:` section for every DFM, under `global:` or its
  own, so the editor keeps the empty `params: {}` under `global:` unless every
  DFM states its own.  It used to leave it out when any DFM did, which made
  such a config fail to load.
