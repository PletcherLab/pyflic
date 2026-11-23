---
status: accepted
---

# Every two-well DFM states its PI Direction

Amends ADR-0028 and the Experiment Editor's PI Direction choice of the same day
(2026-11-22).

## Context

The Experiment Editor's **PI Direction** offered *inherit*, *left* and *right*
for each two-well DFM.  *inherit* wrote no side, so the DFM was read with the
experiment's (`params:` under `global:`, or a Project design's), else the
loader's default, left.

That is how two replicate runs came to disagree: one stated `right` for DFMs 3,
4, 7 and 8, the other stated nothing and was read as left, and nothing in its
config said whether anyone had looked.  Which side well A is on is fixed when
the recording is set up, DFM by DFM.  The lab head asked that the Experiment
Editor take it as left or right for every DFM, with no inheriting.

## Decision

- **No *inherit*.**  Each two-well DFM's PI Direction is *left* or *right*,
  and Save writes it as that DFM's own `params: {pi_direction: ...}`.
- **A DFM whose config states none is shown the side it is read with**: the
  Design's `params:` inside a Project with one, else the file's `global:`
  `params:`, else left.  That side is written, so opening and saving a config
  changes how no DFM is analysed; the file only says it.
- **A side filled in this way is not one stated here.**  *Start from another
  recording…* replaces it without asking; only a side read from the file or
  chosen is listed in its question (ADR-0028).
- **Start from another recording… also takes a side the other recording's
  `global:` states**, for each DFM it lists with none of its own, as the
  loader reads it.
- **A one-well DFM states none.**  Changing the layout from two wells to one
  drops each DFM's side.
- **A DFM's legacy `pi_multiplier` is read as its PI Direction** (1 left,
  anything else right), in the Experiment Editor and the Config Editor.
  Neither had a field for it, so saving dropped it and the DFM was then read as
  left.
- The Config Editor is otherwise unchanged: its override form still lets a DFM
  state no side.

## Consequences

- A two-well config saved by the Experiment Editor states every DFM's side.
  Every DFM then has its own `params:`, so a new config gets no empty
  `params: {}` under `global:`.
- Opening an older config and saving it adds `params: {pi_direction: left}`
  (or the experiment's side) to each DFM.
- A side stated on a DFM wins over `global:` and over a Project design's
  `params:` (defaults < global < DFM).  A later change to the experiment-wide
  side no longer reaches a DFM saved by the Experiment Editor; each DFM is
  changed on its own page.
- A person can still save the side a DFM was given without looking at it.  The
  help asks them to check each DFM before saving.
