---
status: superseded by ADR-0006
---

# Subdir-batch traverses arbitrary depth, all qualifiers run

## Context

Subdir-batch mode previously walked only the *immediate* children of the chosen project directory and ran the script named `batch` from any YAML in each child. Real research trees are deeper than one level (study → cohort → experiment), so users had to invoke batch separately at each parent. This ADR records the design that replaces that flat scan.

## Decision

A **batch target** is a directory containing at least one YAML that defines a script named `batch`. Subdir-batch mode now walks the full tree under the chosen root (root included) and runs the `batch` script in every batch target it finds.

- **Recursion.** Unbounded depth, no target-count warning. Skip dirs whose name starts with `.` or equals `__pycache__`, `node_modules`, `analysis`, `plots`, `qc`. Symlinks not followed.
- **Every qualifier runs.** Parents and their batch-bearing descendants both run; no leaf-filter, no parent-suppression. If a user puts `batch` at multiple levels, that's intentional aggregation.
- **Root counts.** If the chosen root contains a YAML defining `batch`, it runs alongside its descendants.
- **Diagnostics.** Dirs with no YAMLs are silent; dirs with YAMLs but no `batch` script are logged as near-miss skips so users notice forgotten scripts.
- **No output guards.** Outputs are always overwritten; no skip-if-results-exist short-circuit. The input-keyed feeding-summary cache in `.pyflic_cache/` is left untouched (correct by hash construction).
- **Labels.** Run banners and figure titles use `target.relative_to(root).as_posix()` so nested targets are unambiguous; the root labels as `./<yaml_name>`.

## Considered alternatives

- *Leaf-only* (skip parents when a descendant qualifies) — rejected: silently drops user-authored parent scripts.
- *Hard depth cap or N-targets confirm dialog* — rejected: trust the user; skip-list and no-symlink-follow are sufficient guards.
- *Special-case `data/` in skip list* — rejected: hidden domain assumption that breaks if anyone ever puts a YAML there.

## Consequences

The card-hiding rule when subdir-batch is on (single-project Load/Analyze/Plots cards hidden) is now inconsistent — the root might itself be a target with data the user wants to interact with directly. Deferred; revisit if it bites.
