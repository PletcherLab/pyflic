"""
Tiny on-disk feeding-summary cache for ``Experiment``.

Cache entries live in ``experiment_dir/.pyflic_cache/`` and are keyed by:

  - SHA-256 of the canonicalised ``flic_config.yaml`` text
  - SHA-256 of the sorted ``(filename, mtime_ns, size)`` tuple of every
    DFM CSV under ``experiment_dir/data/``
  - The (range_minutes, transform_licks) request

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
# from prior releases are automatically invalidated.
_CACHE_VERSION = "2"


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _config_hash(experiment_dir: Path) -> str:
    cfg = experiment_dir / "flic_config.yaml"
    if not cfg.is_file():
        return "noconfig"
    return _hash_bytes(cfg.read_bytes())


def _data_hash(experiment_dir: Path) -> str:
    data_dir = experiment_dir / "data"
    if not data_dir.is_dir():
        return "nodata"
    rows: list[tuple[str, int, int]] = []
    for f in sorted(data_dir.glob("DFM*.csv")):
        st = f.stat()
        rows.append((f.name, st.st_mtime_ns, st.st_size))
    return _hash_bytes(json.dumps(rows).encode("utf-8"))


def feeding_summary_key(
    experiment_dir: Path,
    *,
    range_minutes: Sequence[float],
    transform_licks: bool,
) -> str:
    a, b = float(range_minutes[0]), float(range_minutes[1])
    parts = [
        f"v{_CACHE_VERSION}",
        _config_hash(experiment_dir),
        _data_hash(experiment_dir),
        f"r{a:g}_{b:g}",
        f"tx{int(bool(transform_licks))}",
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
) -> Path:
    key = feeding_summary_key(
        experiment_dir, range_minutes=range_minutes, transform_licks=transform_licks,
    )
    return cache_dir(experiment_dir) / f"feeding_summary_{key}.csv"


def load_feeding_summary(
    experiment_dir: Path,
    *,
    range_minutes: Sequence[float],
    transform_licks: bool,
) -> pd.DataFrame | None:
    """Return cached feeding summary, or ``None`` if no entry."""
    p = feeding_summary_path(
        experiment_dir,
        range_minutes=range_minutes,
        transform_licks=transform_licks,
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
) -> Path:
    p = feeding_summary_path(
        experiment_dir,
        range_minutes=range_minutes,
        transform_licks=transform_licks,
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
