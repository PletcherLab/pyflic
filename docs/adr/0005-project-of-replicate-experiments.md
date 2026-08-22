---
status: accepted
---

# A Project pools replicate Experiment Directories

pyflic previously had no level above a single recording: its "project directory"
held one experiment, and the only way to work across many was subdir-batch mode.
We are adopting PyTrackingAnalysis's structure so the two apps share one
vocabulary and one directory model: a **Project** is a directory with a
`project.yaml` whose immediate subdirectories holding a `flic_config.yaml` are
its **Replicates**, and the Project owns the pooled results (Combined Analysis,
pooled figures, Project Report). What was called a project directory is now an
**Experiment Directory**, and `Experiment.project_dir` is renamed accordingly.

## Consequences

- The `design:` section of `project.yaml` is the authority for **every** key
  under a Replicate's `global:` — deviation is a load error, not a warning.
  This is stricter than PyTrackingAnalysis, which enforces only experiment type
  and design factors. Chosen because `well_names` and `transform_licks`
  divergence produces a pooled figure that is *wrong* rather than merely noisy,
  and drawing the line anywhere short of "all of `global:`" left an arbitrary
  boundary to defend.
- A Replicate's `flic_config.yaml` normally omits `global:` and inherits it, so
  full authority does not mean ~25 duplicated keys per Replicate. When a
  `global:` block *is* present (a standalone experiment moved into a Project) it
  is validated key-by-key. A Replicate's config is therefore no longer readable
  in isolation — the accepted price of not duplicating the design.
- Per-DFM `params:` overrides survive inside a Project only for the *physical*
  keys `pi_direction` and `chamber_sets`; an override of an analysis key is
  rejected, since it would reintroduce the divergence the authority outlaws one
  level lower and less visibly. Standalone Experiment Directories keep
  unrestricted overrides.
- DFM ids repeat across Replicates (1..8 in each), so DFM is only ever
  interpreted within an Experiment. The pooled mixed model nests DFM within
  Experiment (`groups=Experiment`, `vc_formula={"DFM": "0 + C(DFM)"}`) rather
  than using DFM as a bare random intercept the way the per-experiment model
  does — grouping the stacked frame on DFM alone would silently merge devices
  from different Replicates.
