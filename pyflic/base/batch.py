"""Batch: run one designated Project Script across many Projects (ADR-0006).

A **Batch** is a directory whose *immediate* subdirectories holding a
``project.yaml`` are its Projects.  It is not itself a Project: it holds no
analysis of its own and never pools across Projects — each Project has its own
design, and there is no cross-Project analysis.  Its only product is a
per-Project run summary.

Nothing marks a Batch — being one is structural.  An optional ``batch.yaml``
appears only to name the designated Project Script and to hold central
``project_scripts:``; unlike a Project, a Batch has no authority to declare.

This replaces the recursive, ``batch``-script-keyed discovery of ADR-0001.  That
recursion existed because pyflic had no structural level for "study → cohort →
experiment"; ADR-0005 supplies it, so depth-unbounded scanning has no remaining
job and would only invite running one Project twice from two ancestors.
"""

from __future__ import annotations

import os

import yaml

from .project import PROJECT_FILENAME, Project, is_project_dir

BATCH_FILENAME = "batch.yaml"

#: The Project Script a Batch Run executes when ``batch.yaml`` names none.
#: Every new ``project.yaml`` is seeded with a Project Script under this name,
#: so zero authoring means "create a report on every Project".
DEFAULT_SCRIPT_NAME = "batch"


def project_dirs(batch_dir) -> list[str]:
    """Immediate subdirectories of *batch_dir* that are Projects, sorted.

    Only existing Projects qualify: a Batch Run never creates or upgrades a
    ``project.yaml``.  Children that are not Projects are simply not returned;
    :func:`skipped_dirs` reports them so a forgotten ``project.yaml`` is
    visible rather than silent.
    """
    root = str(batch_dir)
    try:
        entries = sorted(os.listdir(root))
    except OSError:
        return []
    return [os.path.join(root, e) for e in entries
            if os.path.isdir(os.path.join(root, e))
            and is_project_dir(os.path.join(root, e))]


def skipped_dirs(batch_dir) -> list[str]:
    """Immediate subdirectories that are *not* Projects but look like they were
    meant to be — they contain at least one subdirectory of their own.

    Reported so "I forgot the project.yaml" surfaces at the top of a Batch Run
    instead of as a silently short project list.
    """
    root = str(batch_dir)
    out: list[str] = []
    try:
        entries = sorted(os.listdir(root))
    except OSError:
        return []
    for entry in entries:
        path = os.path.join(root, entry)
        if not os.path.isdir(path) or is_project_dir(path):
            continue
        if entry.startswith(".") or entry in {"analysis", "figures", "qc"}:
            continue
        try:
            if any(os.path.isdir(os.path.join(path, c))
                   for c in os.listdir(path)):
                out.append(path)
        except OSError:
            continue
    return out


def is_batch_dir(path) -> bool:
    """True when *path* has at least one immediate Project child.

    Structural, as ADR-0006 requires: no marker file is consulted, so a
    directory becomes a Batch by containing Projects and stops being one by not.
    """
    return bool(project_dirs(path))


class Batch:
    """A loaded Batch: its Projects, its designation, and its central scripts."""

    def __init__(self, batch_dir):
        self.batch_directory = os.path.abspath(str(batch_dir))
        self.project_paths = project_dirs(self.batch_directory)
        self.skipped = skipped_dirs(self.batch_directory)

        self.meta: dict = {}
        marker = os.path.join(self.batch_directory, BATCH_FILENAME)
        if os.path.isfile(marker):
            with open(marker, encoding="utf-8") as handle:
                self.meta = yaml.safe_load(handle) or {}
        self.name = str(self.meta.get("name")
                        or os.path.basename(self.batch_directory))
        #: The designated Project Script — the one a Batch Run executes in
        #: every Project.
        self.script_name = str(self.meta.get("script") or DEFAULT_SCRIPT_NAME)
        #: Central Project Scripts: one recipe serving every Project without
        #: being copied into each ``project.yaml``.
        raw = self.meta.get("project_scripts") or []
        self.project_scripts: list[dict] = [
            dict(item) for item in raw
            if isinstance(item, dict) and item.get("name")]

    def __len__(self) -> int:
        return len(self.project_paths)

    @property
    def project_names(self) -> list[str]:
        return [os.path.basename(p) for p in self.project_paths]

    def find_project_script(self, name: str) -> dict | None:
        for script in self.project_scripts:
            if script.get("name") == name:
                return script
        return None

    def resolve_script(self, project: Project, name: str | None = None):
        """The script to run in *project*: the Project's own copy first, then
        the Batch's central one, then the built-in pipelines.

        The Project's own copy wins so a Project can specialise the designated
        run without the Batch knowing.
        """
        from .script_editor.project_actions import builtin_project_script

        wanted = name or self.script_name
        return (project.find_script(wanted)
                or self.find_project_script(wanted)
                or builtin_project_script(wanted))

    def run(self, script_name: str | None = None, log=print) -> dict:
        """Execute the designated Project Script in every Project.

        Continue-on-error with per-Project log prefixes: one bad Project must
        not abort an unattended overnight run.  Returns a run summary.
        """
        from .script_editor.project_runner import run_project_script

        wanted = script_name or self.script_name
        results: list[dict] = []
        for path in self.project_paths:
            label = os.path.basename(path)
            log(f"\n=== [{label}] {wanted} ===")
            entry = {"project": label, "ok": False, "error": None,
                     "steps": 0}
            try:
                project = Project(path)
                script = self.resolve_script(project, wanted)
                if script is None:
                    raise ValueError(
                        f"no Project Script named '{wanted}' in "
                        f"{PROJECT_FILENAME}, in {BATCH_FILENAME}, or among "
                        f"the built-in pipelines")
                steps = script.get("steps") or []
                run_project_script(project, script,
                                   log=lambda m, p=label: log(f"[{p}] {m}"))
                entry.update(ok=True, steps=len(steps))
            except Exception as err:  # noqa: BLE001
                entry["error"] = f"{type(err).__name__}: {err}"
                log(f"[{label}] FAILED: {entry['error']}")
            results.append(entry)

        ok = sum(1 for r in results if r["ok"])
        log(f"\n=== Batch complete: {ok}/{len(results)} project(s) succeeded ===")
        for path in self.skipped:
            log(f"  skipped (no {PROJECT_FILENAME}): {os.path.basename(path)}")
        return {"script": wanted, "results": results,
                "succeeded": ok, "failed": len(results) - ok,
                "skipped": [os.path.basename(p) for p in self.skipped]}
