"""
Tiny on-disk feeding-summary cache for ``Experiment``.

Cache entries live in ``experiment_dir/.pyflic_cache/`` and are keyed by:

  - SHA-256 of the canonicalised ``flic_config.yaml`` text
  - SHA-256 of the sorted ``(filename, mtime_ns, size)`` tuple of every
    DFM CSV under ``experiment_dir/data/``, and of ``Program.txt``,
    ``Messages.csv`` and ``Baselines.csv`` there when present — or, when the
    config names a ``data_folder:`` (ADR-0021), of those files in that folder
    together with where it is, so a summary of one recording is never served
    for another
  - The (range_minutes, transform_licks) request
  - What else shaped the summary as loaded (``extra``): the design's
    ``global:`` block a Member inherits from ``project.yaml``, the chambers
    its exclusion groups removed, and the Slice (ADR-0020).  Without it an
    exclusion edit or a Design change returned the summary from before it.

Cache hits skip the per-DFM feeding/tasting recomputation entirely.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import pandas as pd


_CACHE_DIR_NAME = ".pyflic_cache"
# Bump this when feeding-summary computation logic changes so stale caches
# from prior releases are automatically invalidated.  "3": firmware 4.0.0
# recordings, whose light seconds come from OnMs (docs/v4-data-plan.md, 0b).
# "4": the key carries the exclusions, the design and the Slice (ADR-0020).
_CACHE_VERSION = "4"

#: Files beside the DFM CSVs that change what an analysis reports: the Opto
#: Program, PyMCU's message log and its Baseline sets.  Matched
#: case-insensitively, as ``opto_program.find_program`` finds the program.
_SIDECAR_NAMES = ("program.txt", "messages.csv", "baselines.csv")


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _config_hash(experiment_dir: Path) -> str:
    """The config's half of the key.  ``about:`` describes the recording and
    changes no number (ADR-0024), so a config holding one is hashed without
    it, and editing a title keeps the cached summaries.  A config without one
    is hashed byte for byte, exactly as it always was."""
    cfg = experiment_dir / "flic_config.yaml"
    if not cfg.is_file():
        return "noconfig"
    data = cfg.read_bytes()
    if b"about" in data:
        import yaml

        from .about import ABOUT_KEY

        try:
            parsed = yaml.safe_load(data)
        except yaml.YAMLError:
            parsed = None
        if isinstance(parsed, dict) and ABOUT_KEY in parsed:
            rest = {k: v for k, v in parsed.items() if k != ABOUT_KEY}
            try:
                return _hash_bytes(json.dumps(rest, sort_keys=True, default=str)
                                   .encode("utf-8"))
            except TypeError:      # keys of mixed types cannot be sorted
                pass
    return _hash_bytes(data)


def _data_hash(experiment_dir: Path, data_location=None) -> str:
    """The recording's half of the key.  *data_location* is where the loader
    read it (:func:`pyflic.base.layout.data_location`, asked here when not
    given).  Without ``data_folder:`` it is ``data/`` and the hash is exactly
    what it always was, so a plain experiment's cached summaries stay valid."""
    if data_location is None:
        from .layout import data_location as _data_location

        data_location = _data_location(experiment_dir)
    data_dir = Path(data_location.path)
    if not data_dir.is_dir():
        return "nodata"
    rows: list[tuple[str, int, int]] = []
    for f in sorted(data_dir.glob("DFM*.csv")):
        st = f.stat()
        rows.append((f.name, st.st_mtime_ns, st.st_size))
    for f in sorted(data_dir.iterdir()):
        if f.name.lower() in _SIDECAR_NAMES and f.is_file():
            st = f.stat()
            rows.append((f.name, st.st_mtime_ns, st.st_size))
    if data_location.referenced:
        return _hash_bytes(json.dumps({"folder": data_location.identity,
                                       "files": rows}).encode("utf-8"))
    return _hash_bytes(json.dumps(rows).encode("utf-8"))


def feeding_summary_key(
    experiment_dir: Path,
    *,
    range_minutes: Sequence[float],
    transform_licks: bool,
    extra: str = "",
    data_location=None,
) -> str:
    a, b = float(range_minutes[0]), float(range_minutes[1])
    parts = [
        f"v{_CACHE_VERSION}",
        _config_hash(experiment_dir),
        _data_hash(experiment_dir, data_location),
        f"r{a:g}_{b:g}",
        f"tx{int(bool(transform_licks))}",
        _hash_bytes(str(extra).encode("utf-8")),
    ]
    return "_".join(parts)


def cache_dir(experiment_dir: Path) -> Path:
    d = experiment_dir / _CACHE_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def feeding_summary_path(
    experiment_dir: Path,
    *,
    range_minutes: Sequence[float],
    transform_licks: bool,
    extra: str = "",
    data_location=None,
) -> Path:
    key = feeding_summary_key(
        experiment_dir, range_minutes=range_minutes, transform_licks=transform_licks,
        extra=extra, data_location=data_location,
    )
    return cache_dir(experiment_dir) / f"feeding_summary_{key}.csv"


def load_feeding_summary(
    experiment_dir: Path,
    *,
    range_minutes: Sequence[float],
    transform_licks: bool,
    extra: str = "",
    data_location=None,
) -> pd.DataFrame | None:
    """Return cached feeding summary, or ``None`` if no entry."""
    p = feeding_summary_path(
        experiment_dir,
        range_minutes=range_minutes,
        transform_licks=transform_licks,
        extra=extra,
        data_location=data_location,
    )
    if not p.is_file():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        try:
            p.unlink()
        except OSError:
            pass
        return None


def save_feeding_summary(
    df: pd.DataFrame,
    experiment_dir: Path,
    *,
    range_minutes: Sequence[float],
    transform_licks: bool,
    extra: str = "",
    data_location=None,
) -> Path:
    p = feeding_summary_path(
        experiment_dir,
        range_minutes=range_minutes,
        transform_licks=transform_licks,
        extra=extra,
        data_location=data_location,
    )
    df.to_csv(p, index=False)
    return p


def clear(experiment_dir: Path) -> int:
    """Remove all cache files under *experiment_dir*.  Returns count removed."""
    d = experiment_dir / _CACHE_DIR_NAME
    if not d.is_dir():
        return 0
    n = 0
    for f in d.iterdir():
        if f.is_file():
            f.unlink()
            n += 1
    return n
