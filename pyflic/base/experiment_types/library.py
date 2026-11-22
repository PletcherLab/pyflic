"""Where Experiment Type definitions live (ADR-0019).

Three places, read in this order for a name:

* **Shipped**: ``experiment_types/shipped/*.yaml`` — Custom, Hedonic and
  ProgressiveRatio, read-only and referenced by name.
* **The copy**: ``experiment_type.yaml`` beside ``project.yaml`` (for a
  Project and every Member) or beside a standalone ``flic_config.yaml``.  A
  lab type travels with the data this way, so the data analyses the same on
  any machine.
* **The lab library**: a folder of ``<name>.yaml`` files set in pyflic's
  settings (``type_library`` in ``ui.json``), ``~/.config/pyflic/
  experiment_types`` by default; point it at a shared drive to share a lab's
  types.  The library holds the masters; a copy that differs from its
  master is flagged (:func:`copy_drift`) and the editors offer to update it.

Names are unique across shipped and lab types, compared without case,
spaces, hyphens or underscores.  Pure: no Qt.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from .definition import (TypeDefinition, read_definition, same_definition,
                         write_definition)

#: The shipped definitions.
SHIPPED_DIR = Path(__file__).resolve().parent / "shipped"
#: The copy's file name, beside ``project.yaml`` or ``flic_config.yaml``.
COPY_NAME = "experiment_type.yaml"
#: The ``ui.json`` key holding the lab library's folder.
LIBRARY_KEY = "type_library"
#: An environment variable that overrides the setting (tests, shared setups).
ENV_LIBRARY = "PYFLIC_TYPE_LIBRARY"
PROJECT_FILE = "project.yaml"


def name_key(name) -> str:
    """How names compare: without case, spaces, hyphens or underscores."""
    return "".join(ch for ch in str(name).strip().lower() if ch not in " -_")


# ---------------------------------------------------------------------------
# Shipped
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _shipped() -> dict[str, TypeDefinition]:
    out: dict[str, TypeDefinition] = {}
    for path in sorted(SHIPPED_DIR.glob("*.yaml")):
        definition = read_definition(path, shipped=True)
        out[name_key(definition.name)] = definition
    return out


def shipped_definitions() -> list[TypeDefinition]:
    """The shipped types, Custom first."""
    return sorted(_shipped().values(), key=lambda d: (d.name != "Custom", d.name))


def shipped_definition(name) -> TypeDefinition | None:
    return _shipped().get(name_key(name))


def is_shipped_name(name) -> bool:
    return name_key(name) in _shipped()


# ---------------------------------------------------------------------------
# The lab library
# ---------------------------------------------------------------------------

def _config_dir() -> Path:
    from ..utils import user_config_dir

    return user_config_dir()


def default_library_dir() -> Path:
    """The library when nothing sets one: the user's pyflic settings folder."""
    return _config_dir() / "experiment_types"


def _settings() -> dict:
    path = _config_dir() / "ui.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def library_dir() -> Path:
    """The lab library's folder: :data:`ENV_LIBRARY`, else the
    ``type_library`` setting, else :func:`default_library_dir`."""
    raw = os.environ.get(ENV_LIBRARY) or _settings().get(LIBRARY_KEY)
    return Path(raw).expanduser() if raw else default_library_dir()


def set_library_dir(path: str | Path | None) -> None:
    """Set the library's folder in ``ui.json`` (``None`` for the default)."""
    data = _settings()
    if path is None or str(path).strip() == "":
        data.pop(LIBRARY_KEY, None)
    else:
        data[LIBRARY_KEY] = str(Path(path).expanduser())
    target = _config_dir() / "ui.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def library_definitions(directory: str | Path | None = None
                        ) -> tuple[list[TypeDefinition], list[str]]:
    """``(definitions, problems)`` for every ``*.yaml`` in the library.  A
    file that cannot be used, or that takes a shipped type's name, is a
    problem line, never an exception, so one bad file hides no other."""
    folder = Path(directory) if directory is not None else library_dir()
    found: list[TypeDefinition] = []
    problems: list[str] = []
    if not folder.is_dir():
        return found, problems
    seen: set[str] = set()
    for path in sorted(folder.glob("*.yaml")):
        try:
            definition = read_definition(path)
        except ValueError as err:
            problems.append(str(err))
            continue
        key = name_key(definition.name)
        if key in _shipped():
            problems.append(f"{path}: '{definition.name}' is a shipped type's name; "
                            f"rename the lab type")
            continue
        if key in seen:
            problems.append(f"{path}: another library file already defines "
                            f"'{definition.name}'")
            continue
        seen.add(key)
        found.append(definition)
    return found, problems


def library_definition(name, directory: str | Path | None = None) -> TypeDefinition | None:
    key = name_key(name)
    for definition in library_definitions(directory)[0]:
        if name_key(definition.name) == key:
            return definition
    return None


def library_path(name, directory: str | Path | None = None) -> Path:
    folder = Path(directory) if directory is not None else library_dir()
    return folder / f"{name}.yaml"


def save_to_library(definition: TypeDefinition,
                    directory: str | Path | None = None) -> Path:
    """Write *definition* to the library as ``<name>.yaml``, replacing the
    master of the same name.  ``ValueError`` for a shipped type's name."""
    if is_shipped_name(definition.name):
        raise ValueError(f"'{definition.name}' is a shipped type's name; a lab type "
                         f"needs a name of its own")
    folder = Path(directory) if directory is not None else library_dir()
    existing = library_definition(definition.name, folder)
    if existing is not None and existing.source is not None \
            and existing.source.name != f"{definition.name}.yaml":
        existing.source.unlink()
    return write_definition(definition, library_path(definition.name, folder))


def delete_from_library(name, directory: str | Path | None = None) -> bool:
    """Remove the library's master of *name*; ``False`` when there is none."""
    definition = library_definition(name, directory)
    if definition is None or definition.source is None:
        return False
    definition.source.unlink()
    return True


# ---------------------------------------------------------------------------
# The copy beside the data
# ---------------------------------------------------------------------------

def copy_dir(start: str | Path) -> Path:
    """Where the copy for *start* lives: the Project's root when *start* is
    a Project or one of its Members, else *start* (a standalone experiment)."""
    start = Path(start)
    if (start / PROJECT_FILE).is_file():
        return start
    if (start.parent / PROJECT_FILE).is_file():
        return start.parent
    return start


def copy_path(start: str | Path) -> Path:
    return copy_dir(start) / COPY_NAME


def read_copy(start: str | Path | None) -> TypeDefinition | None:
    """The copy that applies to *start*, or ``None`` when there is none.
    ``ValueError`` when there is one and it cannot be used."""
    if start is None:
        return None
    path = copy_path(start)
    if not path.is_file():
        return None
    definition = read_definition(path)
    if is_shipped_name(definition.name):
        raise ValueError(f"{path}: '{definition.name}' is a shipped type's name; a "
                         f"copy holds a lab type")
    return definition


def write_copy(definition: TypeDefinition, start: str | Path) -> Path:
    """Copy *definition* beside the data at *start* (ADR-0019).  A shipped
    type needs no copy: ``ValueError``."""
    if is_shipped_name(definition.name):
        raise ValueError(f"'{definition.name}' is shipped with pyflic and is referenced "
                         f"by name; it needs no copy")
    return write_definition(definition, copy_path(start))


def remove_copy(start: str | Path) -> bool:
    path = copy_path(start)
    if path.is_file():
        path.unlink()
        return True
    return False


def copy_drift(start: str | Path, directory: str | Path | None = None) -> str | None:
    """Why the copy at *start* differs from the library's master of its
    name, or ``None`` when they agree, there is no copy, or the library has
    no master (a copy from another lab is not drift)."""
    try:
        copy = read_copy(start)
    except ValueError:
        return None
    if copy is None:
        return None
    master = library_definition(copy.name, directory)
    if master is None or same_definition(copy, master):
        return None
    return (f"the copy of '{copy.name}' in {copy_path(start)} differs from the "
            f"library's ({master.source}); update the copy to analyse with the "
            f"library's version")


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def resolve(name, start: str | Path | None = None,
            directory: str | Path | None = None) -> tuple[TypeDefinition, str]:
    """``(definition, where)`` for *name*: shipped, else the copy at
    *start*, else the library.  *where* is ``"shipped"``, ``"copy"`` or
    ``"library"``.  ``ValueError`` naming the known types when none has it."""
    shipped = shipped_definition(name)
    if shipped is not None:
        return shipped, "shipped"
    copy = read_copy(start)
    if copy is not None and name_key(copy.name) == name_key(name):
        return copy, "copy"
    master = library_definition(name, directory)
    if master is not None:
        return master, "library"
    known = [d.name for d in shipped_definitions()]
    known += [d.name for d in library_definitions(directory)[0]]
    if copy is not None:
        known.append(copy.name)
    where = (f" There is no {COPY_NAME} for it beside the data, and the lab library "
             f"({Path(directory) if directory is not None else library_dir()}) has none.")
    raise ValueError(f"Unknown experiment_type '{name}'. Known types: "
                     f"{', '.join(dict.fromkeys(known))}.{where}")
