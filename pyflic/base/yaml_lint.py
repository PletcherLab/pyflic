"""
Schema linter for ``flic_config.yaml``.

Reports problems with line/column information when possible.  Use as:

    issues = lint_flic_config(Path("project/flic_config.yaml"))
    for i in issues:
        print(i.format())
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import yaml


_KNOWN_GLOBAL_KEYS = {
    "params", "parameters", "well_names", "constants",
    "experimental_design_factors", "experiment_type",
}
_KNOWN_PARAM_KEYS = {
    "baseline_window_minutes", "baseline_window_min", "baseline_window",
    "feeding_threshold", "feeding_minimum", "feeding_minevents",
    "feeding_event_link_gap", "link_gap",
    "tasting_minimum", "tasting_maximum", "tasting_low", "tasting_high",
    "tasting_minevents",
    "samples_per_second", "samples_per_sec",
    "chamber_size", "chamber_sets",
    "correct_for_dual_feeding", "pi_direction", "pi_multiplier",
}
_KNOWN_DFM_KEYS = {
    "id", "ID", "params", "parameters", "chambers", "Chambers",
    "excluded_chambers", "well_names",
}


@dataclass(frozen=True, slots=True)
class LintIssue:
    severity: str   # "error" | "warning"
    message: str
    line: int | None = None
    column: int | None = None
    path: str | None = None

    def format(self, file: Path | None = None) -> str:
        loc = ""
        if file is not None:
            loc = str(file)
        if self.line is not None:
            loc += f":{self.line}"
            if self.column is not None:
                loc += f":{self.column}"
        prefix = f"{loc} " if loc else ""
        return f"{prefix}{self.severity}: {self.message}"


def _emit(
    issues: list[LintIssue],
    severity: str,
    message: str,
    node: Any | None = None,
    path: str | None = None,
) -> None:
    line = column = None
    if node is not None and hasattr(node, "start_mark"):
        line = node.start_mark.line + 1
        column = node.start_mark.column + 1
    issues.append(LintIssue(severity, message, line, column, path))


def lint_flic_config(path: str | Path) -> list[LintIssue]:
    """Validate *path* against the flic_config.yaml schema.

    Returns a list of issues; an empty list means the file is well-formed.
    """
    path = Path(path)
    issues: list[LintIssue] = []

    if not path.is_file():
        issues.append(LintIssue("error", f"file does not exist: {path}"))
        return issues

    text = path.read_text(encoding="utf-8")
    try:
        # Use compose to keep node line numbers
        node = yaml.compose(text)
    except yaml.YAMLError as exc:
        line = column = None
        if hasattr(exc, "problem_mark") and exc.problem_mark is not None:
            line = exc.problem_mark.line + 1
            column = exc.problem_mark.column + 1
        issues.append(LintIssue("error", f"YAML parse error: {exc.problem}", line, column))
        return issues

    if node is None:
        issues.append(LintIssue("error", "file is empty"))
        return issues

    cfg = yaml.safe_load(text)
    if not isinstance(cfg, dict):
        _emit(issues, "error", "root must be a mapping", node)
        return issues

    # Top-level keys
    expected_top = {"global", "dfms", "DFMs", "scripts", "data_dir"}
    for k in cfg.keys():
        if k not in expected_top:
            _emit(issues, "warning", f"unknown top-level key {k!r}", path=k)
    if "data_dir" in cfg:
        _emit(
            issues,
            "warning",
            "'data_dir' is ignored by the loader; data is always read from project_dir/data",
            path="data_dir",
        )

    g = cfg.get("global") or {}
    if g and not isinstance(g, dict):
        _emit(issues, "error", "global: must be a mapping", path="global")
        g = {}
    for k in g.keys():
        if k not in _KNOWN_GLOBAL_KEYS:
            _emit(issues, "warning", f"unknown global key {k!r}", path=f"global.{k}")

    g_params = g.get("params") or g.get("parameters") or {}
    chamber_size_global = None
    if isinstance(g_params, dict):
        for k in g_params.keys():
            if k not in _KNOWN_PARAM_KEYS:
                _emit(
                    issues, "warning",
                    f"unknown param {k!r} (will be ignored by loader)",
                    path=f"global.params.{k}",
                )
        chamber_size_global = g_params.get("chamber_size")
        pi = g_params.get("pi_direction")
        if pi is not None and str(pi).lower() not in ("left", "right"):
            _emit(issues, "error",
                  f"global.params.pi_direction must be 'left' or 'right', got {pi!r}",
                  path="global.params.pi_direction")

    factors = g.get("experimental_design_factors") or {}
    factor_names = list(factors.keys()) if isinstance(factors, dict) else []
    factor_levels: dict[str, list[str]] = {}
    if factor_names:
        for fname, lvls in factors.items():
            if not isinstance(lvls, list) or not lvls:
                _emit(
                    issues, "error",
                    f"factor {fname!r}: levels must be a non-empty list",
                    path=f"global.experimental_design_factors.{fname}",
                )
                factor_levels[fname] = []
            else:
                factor_levels[fname] = [str(x) for x in lvls]

    dfms_node = cfg.get("dfms", cfg.get("DFMs"))
    if dfms_node is None:
        _emit(issues, "error", "missing required 'dfms' section")
        return issues

    if isinstance(dfms_node, dict):
        dfm_items = list(dfms_node.items())
    elif isinstance(dfms_node, list):
        dfm_items = [(node.get("id", node.get("ID")), node) for node in dfms_node if isinstance(node, dict)]
    else:
        _emit(issues, "error", "dfms must be a mapping or list", path="dfms")
        return issues

    seen_ids: set[int] = set()
    for raw_id, dfm in dfm_items:
        if not isinstance(dfm, dict):
            _emit(issues, "error", f"DFM {raw_id!r}: entry must be a mapping")
            continue
        try:
            dfm_id = int(raw_id if raw_id is not None else dfm.get("id"))
        except (TypeError, ValueError):
            _emit(issues, "error", f"DFM id is not an integer: {raw_id!r}")
            continue
        if dfm_id in seen_ids:
            _emit(issues, "error", f"DFM {dfm_id}: duplicate id")
        seen_ids.add(dfm_id)

        for k in dfm.keys():
            if k not in _KNOWN_DFM_KEYS:
                _emit(issues, "warning", f"DFM {dfm_id}: unknown key {k!r}",
                      path=f"dfms.{dfm_id}.{k}")

        d_params = dfm.get("params") or dfm.get("parameters") or {}
        if isinstance(d_params, dict):
            for k in d_params.keys():
                if k not in _KNOWN_PARAM_KEYS:
                    _emit(issues, "warning",
                          f"DFM {dfm_id}: unknown param {k!r}",
                          path=f"dfms.{dfm_id}.params.{k}")
            pi = d_params.get("pi_direction")
            if pi is not None and str(pi).lower() not in ("left", "right"):
                _emit(issues, "error",
                      f"DFM {dfm_id}: pi_direction must be 'left' or 'right', got {pi!r}",
                      path=f"dfms.{dfm_id}.params.pi_direction")

        chamber_size = (d_params.get("chamber_size") if isinstance(d_params, dict) else None)
        if chamber_size is None:
            chamber_size = chamber_size_global
        if chamber_size is None:
            _emit(issues, "error",
                  f"DFM {dfm_id}: chamber_size must be set in global.params or this DFM's params",
                  path=f"dfms.{dfm_id}.params.chamber_size")

        chambers = dfm.get("chambers") or dfm.get("Chambers")
        if chambers is None:
            _emit(issues, "warning", f"DFM {dfm_id}: no 'chambers' assignments",
                  path=f"dfms.{dfm_id}.chambers")
        elif factor_names and isinstance(chambers, dict):
            for ch_idx, raw in chambers.items():
                levels = [s.strip() for s in str(raw).split(",")]
                if len(levels) != len(factor_names):
                    _emit(
                        issues, "error",
                        f"DFM {dfm_id} chamber {ch_idx}: expected {len(factor_names)} factor "
                        f"levels ({', '.join(factor_names)}), got {len(levels)}: {raw!r}",
                        path=f"dfms.{dfm_id}.chambers.{ch_idx}",
                    )
                    continue
                for fname, lvl in zip(factor_names, levels, strict=True):
                    allowed = factor_levels.get(fname) or []
                    if allowed and lvl not in allowed:
                        _emit(
                            issues, "error",
                            f"DFM {dfm_id} chamber {ch_idx}: level {lvl!r} not in "
                            f"declared {fname} levels {allowed}",
                            path=f"dfms.{dfm_id}.chambers.{ch_idx}",
                        )

    return issues


def main_cli() -> None:
    """Command-line entry: ``python -m pyflic.base.yaml_lint <project_or_yaml>``."""
    import sys
    if len(sys.argv) != 2:
        print("usage: pyflic-lint <project_dir | flic_config.yaml>", file=sys.stderr)
        raise SystemExit(2)
    target = Path(sys.argv[1])
    cfg = target if target.is_file() else target / "flic_config.yaml"
    issues = lint_flic_config(cfg)
    for i in issues:
        print(i.format(cfg))
    n_err = sum(1 for i in issues if i.severity == "error")
    n_warn = sum(1 for i in issues if i.severity == "warning")
    print(f"\n{n_err} error(s), {n_warn} warning(s)")
    raise SystemExit(1 if n_err else 0)
