"""The settings an analysis ran with, recorded beside its results (ADR-0022).

Every ``analysis/feeding_summary.csv`` gets an ``analysis_settings.json`` next to
it: the Design's ``global:`` merged with the Member's own, the ``dfms:`` block
and ``data_folder:`` — what the numbers were computed from.  The stale rule
compares it with what the Design and the config say now, so saving a changed
detection parameter, a constant or a chamber assignment reads "re-run needed",
while a save that changes no setting (a Selection tick, a Project Script edit,
the same values written back) does not.

A record is separate from the summary on purpose: ADR-0010 kept provenance out
of ``feeding_summary.csv`` because R scripts and notebooks read that format.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

RECORD_FILENAME = "analysis_settings.json"

#: ``global:`` keys that change no number in the results: the Selection says
#: which outputs a run writes, not how they are computed (ADR-0018), and each
#: Slice's own ``slice.json`` answers for its definition (ADR-0020).
IGNORED_GLOBAL_KEYS = frozenset({"omit", "slices", "slice_tolerance_min"})

#: Top-level config keys an analysis reads besides ``global:`` and ``dfms:``.
_OTHER_CONFIG_KEYS = ("data_folder",)


def says_nothing(key: Any, value: Any) -> bool:
    """An empty ``params:`` (or ``parameters:``) in a config's ``global:``:
    every default, which states no setting (ADR-0030).  The Experiment Editor
    once wrote one into every new config, only to get past a loader check
    that is gone."""
    return key in ("params", "parameters") and isinstance(value, Mapping) and not value


def stated_global(own: Any) -> dict:
    """A config's own ``global:`` without what :func:`says_nothing`, so that
    over a Project Design's it never hides the Design's settings."""
    if not isinstance(own, Mapping):
        return {}
    return {k: v for k, v in own.items() if not says_nothing(k, v)}


def _comparable(value: Any) -> Any:
    """JSON-ready comparison form: numbers as floats (``20`` and ``20.0`` in
    two hand-edited files are one setting), mapping keys as strings."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Mapping):
        return {str(k): _comparable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_comparable(v) for v in value]
    return str(value)


def _dfms_by_id(nodes: Any) -> Any:
    """``dfms:`` keyed by DFM id, in list or mapping form alike, so a
    difference names the DFM rather than its position in the file."""
    if isinstance(nodes, Mapping):
        pairs = list(nodes.items())
    elif isinstance(nodes, list):
        pairs = [((node.get("id", node.get("ID", index))
                   if isinstance(node, Mapping) else index), node)
                 for index, node in enumerate(nodes)]
    else:
        return nodes
    out: dict[str, Any] = {}
    for key, node in pairs:
        if isinstance(key, float) and key.is_integer():
            key = int(key)
        if isinstance(node, Mapping):
            node = {k: v for k, v in node.items() if k not in ("id", "ID")}
        out[str(key)] = node
    return out


def settings(design_global: Mapping[str, Any] | None,
             config: Mapping[str, Any] | None) -> dict:
    """The settings an analysis of *config* depends on, in comparison form.

    *design_global* is the Project Design's ``global:`` (``None`` for a
    standalone experiment); the config's own ``global:`` keys win, as they do
    at load.
    """
    config = config if isinstance(config, Mapping) else {}
    merged = dict(design_global or {})
    merged.update(stated_global(config.get("global")))
    for key in IGNORED_GLOBAL_KEYS:
        merged.pop(key, None)
    out: dict[str, Any] = {"global": merged,
                           "dfms": _dfms_by_id(config.get("dfms", config.get("DFMs")))}
    for key in _OTHER_CONFIG_KEYS:
        if config.get(key) is not None:
            out[key] = config[key]
    return _comparable(out)


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

def write(analysis_dir: str | Path, recorded: Mapping[str, Any]) -> Path:
    """Write ``<analysis_dir>/analysis_settings.json``."""
    path = Path(analysis_dir) / RECORD_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"about": "The settings the results in this folder were computed "
                        "with; pyflic compares them with the config to say "
                        "whether a re-run is needed.",
               "settings": dict(recorded)}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read(analysis_dir: str | Path) -> dict | None:
    """The settings recorded in *analysis_dir*, or ``None`` when there is no
    readable record (results written before records existed)."""
    path = Path(analysis_dir) / RECORD_FILENAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    recorded = payload.get("settings") if isinstance(payload, dict) else None
    return recorded if isinstance(recorded, dict) else None


# ---------------------------------------------------------------------------
# What the settings are now
# ---------------------------------------------------------------------------

#: path -> ((mtime_ns, size), parsed): the Hub asks for every Member's status
#: on each refresh, and the files rarely change between two.
_YAML_CACHE: dict[str, tuple[tuple[int, int], Any]] = {}


def _read_yaml(path: str) -> Any:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    key = (stat.st_mtime_ns, stat.st_size)
    hit = _YAML_CACHE.get(path)
    if hit is not None and hit[0] == key:
        return hit[1]
    try:
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except Exception:  # noqa: BLE001 - an unreadable file has no settings
        data = None
    _YAML_CACHE[path] = (key, data)
    return data


def current(project_yaml: str | None, config_path: str) -> dict | None:
    """The settings the files on disk state now, or ``None`` when the
    config cannot be read.  Read fresh (behind a modification-time cache),
    so a save made anywhere, in an editor or by hand, is seen at once."""
    config = _read_yaml(config_path)
    if not isinstance(config, Mapping):
        return None
    design_global = None
    if project_yaml is not None:
        meta = _read_yaml(project_yaml)
        design = meta.get("design") if isinstance(meta, Mapping) else None
        found = design.get("global") if isinstance(design, Mapping) else None
        design_global = found if isinstance(found, Mapping) else None
    return settings(design_global, config)


# ---------------------------------------------------------------------------
# Differences, for people
# ---------------------------------------------------------------------------

def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _flatten(value: Any, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    if isinstance(value, Mapping) and value:
        out: dict[tuple[str, ...], Any] = {}
        for key, item in value.items():
            out.update(_flatten(item, (*prefix, str(key))))
        return out
    return {prefix: value}


def _label(path: tuple[str, ...]) -> str:
    if path and path[0] == "global":
        return ".".join(path[1:]) or "global"
    if len(path) >= 2 and path[0] == "dfms":
        rest = ".".join(path[2:])
        return f"DFM {path[1]}" + (f" {rest}" if rest else "")
    return ".".join(path)


def _show(value: Any) -> str:
    if value is None:
        return "(not set)"
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return str(int(value))
        return f"{value:g}"
    if isinstance(value, list):
        return "[" + ", ".join(_show(v) for v in value) + "]"
    if isinstance(value, Mapping) and not value:
        return "(none)"
    return str(value)


def differences(recorded: Mapping[str, Any], now: Mapping[str, Any]) -> list[str]:
    """Each setting that differs, as ``"params.feeding_threshold changed:
    20 → 25"``; empty when they agree."""
    if _canonical(recorded) == _canonical(now):
        return []
    before, after = _flatten(recorded), _flatten(now)
    out = []
    ## The Design's settings first, then each DFM's.
    for path in sorted(set(before) | set(after),
                       key=lambda p: (p[:1] != ("global",), p)):
        old, new = before.get(path), after.get(path)
        if _canonical(old) == _canonical(new):
            continue
        out.append(f"{_label(path)} changed: {_show(old)} → {_show(new)}")
    return out
