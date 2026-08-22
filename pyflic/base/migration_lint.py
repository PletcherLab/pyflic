"""Migration checks for the Project overhaul (ADR-0005 through ADR-0008).

There is no migration tool: the overhaul is a hard break, and this module is
the whole crossing.  It names each construct that no longer works and the form
that replaces it, so a config can be fixed by hand with the error in front of
you.  Nothing here writes to disk — orphaned output directories in particular
are only ever *reported*, because they may hold figures already published.

Consumed by ``pyflic lint``, which runs it beside the schema linter.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class MigrationIssue:
    severity: str      # "error" | "warning"
    message: str
    fix: str
    path: str | None = None

    def format(self) -> str:
        where = f"{self.path}: " if self.path else ""
        return f"{where}{self.severity}: {self.message}\n    fix: {self.fix}"


def _load(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception:  # noqa: BLE001
        return {}


def check_config(config_path: Path, *, in_project: bool = False
                 ) -> list[MigrationIssue]:
    """Migration problems in one ``flic_config.yaml``."""
    from . import experiment_types
    from .yaml_config import PHYSICAL_DFM_KEYS, _normalize_param_overrides

    issues: list[MigrationIssue] = []
    rel = str(config_path)
    cfg = _load(config_path)
    global_cfg = cfg.get("global") or {}

    # ---- experiment_type that is now a Chamber Layout (ADR-0007) --------
    raw_type = str(global_cfg.get("experiment_type") or "").strip()
    key = raw_type.lower().replace("-", "_").replace(" ", "_")
    if key in experiment_types.RETIRED_AS_LAYOUT:
        layout = "single_well" if "single" in key else "two_well"
        issues.append(MigrationIssue(
            "error",
            f"experiment_type: {raw_type} is no longer an experiment type — "
            f"it names a chamber layout",
            f"remove 'experiment_type' and set 'chamber_layout: {layout}' "
            f"under global: (this makes it a Custom Experiment, which is what "
            f"it always was)",
            rel))
        exp_type = experiment_types.CustomExperimentType()
    else:
        try:
            exp_type = experiment_types.get_experiment_type(raw_type or None)
        except ValueError as err:
            issues.append(MigrationIssue(
                "error", str(err),
                "use one of the known experiment types, or remove the key for "
                "a Custom Experiment", rel))
            return issues
        for problem in exp_type.validate(global_cfg):
            issues.append(MigrationIssue(
                "error", problem,
                "delete the key — the experiment type owns it and derives its "
                "value (ADR-0007)", rel))

    # ---- chamber_size on an untyped config -----------------------------
    params = global_cfg.get("params") or global_cfg.get("parameters") or {}
    if exp_type.is_custom and params.get("chamber_size") is not None:
        layout = ("single_well" if int(params["chamber_size"]) == 1
                  else "two_well")
        issues.append(MigrationIssue(
            "warning",
            "params.chamber_size is derived from the chamber layout now",
            f"replace it with 'chamber_layout: {layout}' under global:", rel))

    # ---- per-DFM analysis overrides inside a Project (ADR-0005) ---------
    if in_project:
        if global_cfg:
            issues.append(MigrationIssue(
                "warning",
                "this replicate states its own global: block",
                "delete it and inherit the project design, or make sure every "
                "key matches the design exactly — a mismatch is a load error",
                rel))
        for node in (cfg.get("dfms") or []):
            if not isinstance(node, dict):
                continue
            over = node.get("params") or node.get("parameters") or {}
            illegal = sorted(set(_normalize_param_overrides(over))
                             - PHYSICAL_DFM_KEYS)
            if illegal:
                issues.append(MigrationIssue(
                    "error",
                    f"DFM {node.get('id')} overrides analysis params {illegal}",
                    f"move them to the project design; only "
                    f"{sorted(PHYSICAL_DFM_KEYS)} may vary per DFM", rel))

    # ---- Experiment Scripts named 'batch' (ADR-0006) -------------------
    for script in (cfg.get("scripts") or []):
        if isinstance(script, dict) and \
                str(script.get("name", "")).strip().lower() == "batch":
            issues.append(MigrationIssue(
                "warning",
                "an Experiment Script named 'batch' no longer has special "
                "meaning — subdir-batch mode is retired",
                "rename it (a Batch Run now executes a *Project* Script named "
                "by batch.yaml, one level up)", rel))
    return issues


def check_directory(directory: Path, *, in_project: bool = False
                    ) -> list[MigrationIssue]:
    """Migration problems in one Experiment Directory: its config, its extra
    YAMLs, and its orphaned output directories."""
    issues: list[MigrationIssue] = []
    rel = str(directory)

    configs = sorted(p for p in directory.glob("*.yaml")
                     if p.name not in ("project.yaml", "batch.yaml",
                                       "plot_specs.yaml"))
    configs += sorted(directory.glob("*.yml"))
    canonical = directory / "flic_config.yaml"

    if len(configs) > 1:
        others = ", ".join(p.name for p in configs if p != canonical)
        issues.append(MigrationIssue(
            "error",
            f"{len(configs)} config YAMLs in one directory ({others} beside "
            f"flic_config.yaml) — an Experiment Directory holds exactly one",
            "keep one as flic_config.yaml and move the others into their own "
            "Experiment Directories; this cannot be automated because only "
            "you know which is canonical", rel))
    elif configs and not canonical.exists():
        issues.append(MigrationIssue(
            "error",
            f"the config is named {configs[0].name}, not flic_config.yaml",
            "rename it to flic_config.yaml", rel))

    for entry in sorted(directory.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.endswith("_results"):
            issues.append(MigrationIssue(
                "warning",
                f"{entry.name}/ is an orphaned output directory — outputs go "
                f"to analysis/ now",
                "move anything you still need, then delete it; pyflic will "
                "never touch it", rel))
        elif entry.name.startswith(("analysis_", "qc_")):
            issues.append(MigrationIssue(
                "warning",
                f"{entry.name}/ is an orphaned windowed output directory — a "
                f"time window is a Facet column now (ADR-0008)",
                "set 'facet_cutoffs' under global: to get the same windows as "
                "columns in feeding_summary_facet.csv, then delete it", rel))
        elif entry.name == ".pyflic_cache":
            issues.append(MigrationIssue(
                "warning",
                ".pyflic_cache/ entries predate the facet change and will not "
                "be reused",
                "run 'pyflic clear-cache' on this directory to reclaim the "
                "space (correctness is unaffected — the keys simply miss)",
                rel))

    if canonical.exists():
        issues += check_config(canonical, in_project=in_project)
    return issues


def check_tree(root: Path) -> list[MigrationIssue]:
    """Migration problems anywhere under *root*, dispatching on marker files."""
    from .project import is_experiment_dir, is_project_dir

    issues: list[MigrationIssue] = []
    root = Path(root)

    if is_project_dir(root):
        meta = _load(root / "project.yaml")
        if not (meta.get("design") or {}).get("global"):
            issues.append(MigrationIssue(
                "warning",
                "project.yaml has no design.global section, so replicates are "
                "validated against each other instead of against an authority",
                "add a design: global: block naming the shared settings "
                "(ADR-0005)", str(root)))
        for entry in sorted(root.iterdir()):
            if entry.is_dir() and is_experiment_dir(entry):
                issues += check_directory(entry, in_project=True)
        return issues

    if is_experiment_dir(root) or any(root.glob("*.yaml")):
        return check_directory(root)

    ## Not itself an experiment: treat it as a Batch and check the Projects
    ## under it, so `pyflic lint` on a study folder does something useful.
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and is_project_dir(entry):
            issues += check_tree(entry)
    return issues
