"""Persisted UI preferences (theme mode, recent projects).

Stored as JSON at ``~/.config/pyflic/ui.json``.  No external deps; failures
are silent so the apps keep working even when the config dir is unwritable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "pyflic"
_CONFIG_FILE = _CONFIG_DIR / "ui.json"

_DEFAULTS: dict[str, Any] = {
    "theme": "auto",
    "recent_projects": [],
}


def load() -> dict[str, Any]:
    if not _CONFIG_FILE.is_file():
        return dict(_DEFAULTS)
    try:
        data = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return dict(_DEFAULTS)
        merged = dict(_DEFAULTS)
        merged.update(data)
        return merged
    except Exception:  # noqa: BLE001
        return dict(_DEFAULTS)


def save(data: dict[str, Any]) -> None:
    try:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        _CONFIG_FILE.write_text(
            json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
        )
    except Exception:  # noqa: BLE001
        pass


def get(key: str, default: Any = None) -> Any:
    return load().get(key, default if default is not None else _DEFAULTS.get(key))


def set_value(key: str, value: Any) -> None:
    data = load()
    data[key] = value
    save(data)


def add_recent_project(path: str | Path, *, max_items: int = 8) -> None:
    p = str(Path(path).expanduser().resolve())
    data = load()
    recents = [r for r in data.get("recent_projects", []) if r != p]
    recents.insert(0, p)
    data["recent_projects"] = recents[:max_items]
    save(data)
