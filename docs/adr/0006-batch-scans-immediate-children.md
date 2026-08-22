---
status: accepted
supersedes: ADR-0001
---

# Batch scans immediate children for Projects

ADR-0001 made subdir-batch mode walk the tree at unbounded depth, treating any
directory with a YAML defining a script named `batch` as a target. Its stated
motivation was that "real research trees are deeper than one level (study →
cohort → experiment)" — the recursion was compensating for a structural layer
pyflic did not have. ADR-0005 supplies that layer, so we adopt
PyTrackingAnalysis's definition instead: a **Batch** is a directory whose
*immediate* subdirectories holding a `project.yaml` are its Projects, and a
Batch Run executes one designated Project Script in each.

## Consequences

- The magic script name `batch`, the recursive walk, the directory skip-list,
  the near-miss logging, and "batch target" as a term are all removed.
- A Batch is not itself a Project: it holds no analysis of its own and never
  pools across Projects. Its optional `batch.yaml` carries only the designated
  `script:` and a central `project_scripts:` section.
- Trees deeper than Batch → Project → Experiment now require invoking a Batch
  Run per parent. Accepted: unbounded recursion over a hierarchy that already
  has named levels invites running the same Project twice from two ancestors.
