"""The Batch level (ADR-0006, ADR-0009): many Projects run unattended.

A Batch is structural — a directory with at least one Project anywhere beneath
it.  Discovery is recursive and **prunes at each Project**: the walk descends
until it finds one — ``project.yaml`` plus at least one Member the run could
use — and never looks inside, because a Project's subdirectories are its
Members by definition (ADR-0005).  Grouping folders (``Sept2026/``,
``Archive/2025/``) are therefore transparent, and a Project is identified by
its POSIX path relative to the Batch root (``Sept2026/ProjA``; a top-level
Project is just ``ProjA``, so existing ``batch.yaml`` files, Exclusion Sheets,
and API calls keep working unchanged).

Nothing marks a Batch: ``batch.yaml`` at its root appears only once batch-level
scripting is authored, because unlike a Project a Batch has no authority to
declare.  A Batch Run executes one designated Project Script in every Project
(continue-on-error, per-Project log prefixes) and its only product is the
per-Project run summary — a Batch never pools results across Projects.

There is no third script level: the thing a Batch Run runs IS a Project Script,
resolved per Project as ``batch.yaml`` ``project_scripts:`` → the Project's own
``scripts:`` → the built-ins.  With no designation at all each Project runs its
OWN default script — every ``project.yaml`` is created with one, so nothing is
silently substituted for a Project that has none.

**Naming.** A pyflic Project holds **Members** (different experiments
addressing one question), so the Projects inside a Batch are deliberately NOT
called members: "a member with four members" is a sentence this codebase must
not be able to write.  They are :class:`BatchProject`, keyed by relative path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import yaml

from . import exclusion_sheet
from . import layout
from . import project as prj
from .project import PROJECT_FILENAME, Project, is_project_dir

#: The lazy Batch file: the ``script:`` designation plus centrally-held Project
#: Scripts (the ``experiment_scripts:`` idea one level up).
BATCH_FILENAME = "batch.yaml"

#: The Batch AI narrative, written at the Batch root.  Named distinctly from a
#: Project's own narrative so a recursive glob can tell the levels apart.
BATCH_NARRATIVE_FILENAME = "batch_ai_narrative.md"

#: The Project Script a Batch Run executes when ``batch.yaml`` names none and
#: the caller asks for the historical default by name.
DEFAULT_SCRIPT_NAME = "batch"

#: What the batch-level agent is asked for.  It reads the Projects' own
#: narratives, so it must synthesize across them rather than restate each: the
#: per-Project detail is already one file down.
BATCH_NARRATIVE_PROMPT = """\
You are given the AI narratives of several independent Projects from one batch
of FLIC feeding-behaviour experiments. Each Project is its own design with its
own members; results are NOT pooled across Projects.

Write one synthesis for a researcher reviewing the whole batch:

1. **Results across the batch, in reasonable detail.** What each Project found,
   and — more importantly — where the Projects agree, where they disagree, and
   what the batch shows taken together. Give effect directions and magnitudes
   where the narratives state them.
2. **Experimental design problems.** Call out specific Projects whose design
   looks compromised: too few members, unbalanced or missing treatment groups,
   inconsistent factors or well assignments between members, members that were
   never analyzed.
3. **Heavy chamber loss.** Name any Project or member where a large share of
   chambers was excluded or flagged (low lick counts, quality cutoffs), give
   the numbers the narratives report, and say what it implies for trusting that
   Project's result.

Do not itemize minor per-Project or per-member detail — that lives in the
Project narratives themselves. Prefer plain prose over bullet soup. Where the
source narratives are silent on something, say so rather than inferring it. Do
not perform new analysis: you are summarizing what the pipeline already
computed."""

#: A runaway walk is a mis-clicked home directory, not a Batch.  The cap is far
#: above any real batch (a 200-Project batch visits a few hundred directories)
#: and exists so pointing the Hub at ``/`` cannot hang it.
MAX_WALKED_DIRECTORIES = 20000

#: Never descended into: caches and anything hidden.  Everything else is
#: decided structurally — a denylist of output-directory names would misfire on
#: a grouping folder that happens to be called ``figures``.
_SKIP_DIRNAMES = {"__pycache__"}


@dataclass(frozen=True)
class BatchProject:
    """One Project a Batch Run can target, with its layout already read.

    *key* is the identity (POSIX path relative to the Batch root); *members*
    holds every experiment-shaped subdirectory, healthy or Blocked — blocked is
    a property of the Member, never of the Project, so a Project with four
    healthy members and one blocked one runs the four.
    """

    key: str
    directory: str
    members: tuple = ()
    has_report: bool = False

    @property
    def usable(self) -> tuple:
        return tuple(m for m in self.members if m.usable)

    @property
    def blocked(self) -> tuple:
        return tuple(m for m in self.members if m.blocked)

    @property
    def runnable(self) -> bool:
        """False when nothing in it can be analyzed — the run would only
        produce a failure, so such a Project starts unchecked."""
        return bool(self.usable)

    def summary(self) -> str:
        text = f"{len(self.usable)}/{len(self.members)} members"
        if self.blocked:
            text += f", {len(self.blocked)} blocked"
        return text


def project_kind(directory) -> tuple[str, tuple]:
    """Classify *directory* as a Batch Project candidate.

    Returns ``(kind, members)`` where kind is:

    ``"project"``
        ``project.yaml`` and at least one *usable* Member — certainly a
        Project.  The walk prunes here.
    ``"unconfirmed"``
        ``project.yaml`` and experiment-shaped children, but not one the run
        could use — every Member is blocked.  It is probably a Project needing
        repair, but a grouping folder holding one junk subdirectory (a
        ``template/`` with a config, an exported spreadsheet) looks identical,
        so the walk descends first and only calls it a Project if no real
        Project turns up below.
    ``"marker"``
        ``project.yaml`` and nothing experiment-shaped at all: a grouping
        folder someone dropped a marker into.  Walk through it.
    ``""``
        not a Project.
    """
    if not is_project_dir(directory):
        return "", ()
    members = tuple(layout.members_in(directory))
    if any(m.usable for m in members):
        return "project", members
    if members:
        ## Configured-but-unusable is not strong enough to prune on: a grouping
        ## folder with one ``template/flic_config.yaml`` looks exactly like a
        ## Project whose members are all blocked, and pruning there hides every
        ## real Project beneath it.
        return "unconfirmed", members
    return "marker", ()


def _batch_project(path, root, members) -> BatchProject:
    return BatchProject(
        key=os.path.relpath(path, root).replace(os.sep, "/"),
        directory=path,
        members=tuple(members),
        has_report=_has_report(path),
    )


class _Walk:
    """One recursive discovery pass, with its budget and cycle guard.

    Depth-first and *decide-after-descending* for the ambiguous case, so a
    stray ``project.yaml`` at a grouping level can never hide the Projects
    beneath it — the exact mistake recursion exists to tolerate.
    """

    def __init__(self, root) -> None:
        self.root = os.path.abspath(str(root))
        self.projects: list = []
        self.skipped: list = []
        self.truncated = False
        self._seen: set[str] = set()
        self._budget = MAX_WALKED_DIRECTORIES

    def key(self, path) -> str:
        return os.path.relpath(path, self.root).replace(os.sep, "/")

    def note(self, path, why: str) -> None:
        self.skipped.append((self.key(path), why))

    def run(self) -> dict:
        kind, _members = project_kind(self.root)
        if kind == "project":
            ## A Project is never also a Batch: its subdirectories are its
            ## Members, not Projects.
            return self.result()
        ## The root's own flic_config.yaml does not stop the walk either: a
        ## folder can be both a stray Experiment Directory and the place the
        ## user keeps their projects.
        self.descend(self.root)
        return self.result()

    def result(self) -> dict:
        self.projects.sort(key=lambda p: p.key)
        return {"projects": self.projects, "skipped": sorted(self.skipped),
                "truncated": self.truncated}

    def descend(self, directory) -> int:
        """Walk *directory*'s children; returns how many Projects were found.

        A caller deciding "no Projects below, so this IS the Project" must
        check :attr:`truncated` first — a budget-exhausted descent found
        nothing because it never looked.
        """
        if self._budget <= 0:
            self.truncated = True
            return 0
        self._budget -= 1
        try:
            real = os.path.realpath(directory)
        except OSError:
            return 0
        if real in self._seen:
            return 0
        self._seen.add(real)

        try:
            entries = sorted(os.scandir(directory), key=lambda e: e.name)
        except OSError as err:
            ## A directory nobody can read may hold a whole Project.  Silently
            ## pruning it would report the batch as smaller than it is.
            self.note(directory, f"cannot be listed ({err.strerror or err})")
            return 0

        found = 0
        for entry in entries:
            name = entry.name
            if name.startswith(".") or name in _SKIP_DIRNAMES:
                continue
            try:
                if not entry.is_dir(follow_symlinks=False):
                    if entry.is_symlink() and entry.is_dir():
                        ## A link into an archive share would double-run its
                        ## members, and a link to an ancestor is a cycle.
                        self.note(entry.path,
                                  "symlinked directory — not followed")
                    continue
            except OSError:
                self.note(entry.path, "cannot be read")
                continue
            found += self.visit(entry.path)
        return found

    def visit(self, path) -> int:
        kind, members = project_kind(path)
        if kind == "project":
            self.projects.append(_batch_project(path, self.root, members))
            return 1                              # prune: a Project is a leaf
        if kind == "unconfirmed":
            below = self.descend(path)
            if below or self.truncated:
                ## Real Projects live under it, so the marker was a grouping
                ## folder with a junk subdirectory — say so and keep them.
                self.note(path, f"has {PROJECT_FILENAME} and {len(members)} "
                                "folder(s) nothing can run; treated as a "
                                "folder of projects")
                return below
            self.projects.append(_batch_project(path, self.root, members))
            return 1
        if kind == "marker":
            below = self.descend(path)
            ## Always reported, either way: "I marked that folder as a Project
            ## — why isn't it in the list?" is the question this rule creates,
            ## and it deserves an answer in the log.
            self.note(path, f"has {PROJECT_FILENAME} but no member directory"
                            + (f"; {below} project(s) found inside it instead"
                               if below else ""))
            return below
        if layout.has_config(path):
            ## An Experiment Directory: its children are data/, analysis/, qc/
            ## — never Projects.  But a stray flic_config.yaml at a grouping
            ## level looks identical, so descend first and stop only if nothing
            ## turns up: an unconditional stop here hid every Project below one
            ## stray file, exactly as the project.yaml marker did.
            below = self.descend(path)
            if below:
                self.note(path, f"has {prj.CONFIG_FILENAME} and {below} "
                                "project(s) inside it")
            return below
        return self.descend(path)


def _has_report(project_dir) -> bool:
    try:
        return any(name.lower().endswith("_report.pdf")
                   for name in os.listdir(project_dir))
    except OSError:
        return False


def discover(root) -> dict:
    """Everything one walk of *root* found.

    ``{"projects": [...], "skipped": [(key, why)], "truncated": bool}`` — the
    Projects in run order, every directory the walk dropped and why, and
    whether the walk hit its budget.  The single source of truth for what a
    Batch contains: the table, the preflight, and the run all read it, so what
    the user was shown is what runs.  Cheap by construction — directory
    listings only, no Project loads and no YAML parsing.
    """
    return _Walk(root).run()


def discover_projects(root) -> list:
    """Every Project under *root*, in run order (by relative-path key)."""
    return discover(root)["projects"]


def batch_project_names(path) -> list[str]:
    """The Projects' keys, in the order a Batch Run visits them.

    A key is the Project's path relative to the Batch root, so a top-level
    Project is still its bare folder name and every stored designation, sheet
    row, and API call written before recursive discovery still resolves.
    """
    return [item.key for item in discover_projects(path)]


def project_directory(root, key) -> str:
    """Absolute directory of Project *key* in Batch *root*.

    A key comes from the walk today, but this is the public key→directory
    resolver and the Exclusion Sheet's project column is hand-typed, so an
    escaping key is refused rather than resolved.
    """
    parts = [part for part in str(key).split("/") if part not in ("", ".")]
    if any(part == ".." for part in parts) or os.path.isabs(str(key)):
        raise ValueError(f"{key!r} is not a project of this batch")
    return os.path.join(os.path.abspath(str(root)), *parts)


def is_batch_dir(path) -> bool:
    """A Batch is structural: at least one Project lies somewhere beneath.

    A Project is never also a Batch — its subdirectories are its Members, so a
    nested ``project.yaml`` does not turn it into one.  Deliberately the SAME
    walk the table and the run use: a short-circuit that asked
    ``is_project_dir`` at the root would disagree with the walk's stricter
    test, and one stray ``project.yaml`` at a batch root would empty the whole
    panel.
    """
    return bool(discover(path)["projects"])


def nested_batch_files(root) -> list[str]:
    """``batch.yaml`` files below *root* that this run ignores.

    Recursive discovery means a grouping folder can be a Batch in its own right
    and carry its own designation.  Only the selected Batch's file governs — the
    resolution order is already three steps and a fourth that depended on where
    the user clicked would be unmemorable — so these are named rather than
    silently overridden.
    """
    root = os.path.abspath(str(root))
    found: list[str] = []
    for item in discover_projects(root):
        directory = os.path.dirname(item.directory)
        while len(directory) > len(root):
            candidate = os.path.join(directory, BATCH_FILENAME)
            relative = os.path.relpath(candidate, root).replace(os.sep, "/")
            if os.path.isfile(candidate) and relative not in found:
                found.append(relative)
            directory = os.path.dirname(directory)
    return sorted(found)


def load_batch_file(path) -> dict:
    """The parsed ``batch.yaml`` of Batch *path* as
    ``{"script": str | None, "project_scripts": list[dict]}``.

    Lenient like ``Project.__init__``'s scripts parse: a missing file, a
    malformed document, or a bad block yields empty sections rather than an
    exception — one bad block must not take down a whole Batch Run.
    """
    result: dict = {"script": None, "project_scripts": []}
    file_path = os.path.join(str(path), BATCH_FILENAME)
    if not os.path.isfile(file_path):
        return result
    try:
        with open(file_path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception:  # noqa: BLE001
        return result
    if not isinstance(data, dict):
        return result
    name = data.get("script")
    if isinstance(name, str) and name.strip():
        result["script"] = name.strip()
    raw = data.get("project_scripts") or []
    if isinstance(raw, list):
        result["project_scripts"] = [
            dict(item) for item in raw
            if isinstance(item, dict) and item.get("name")]
    return result


def save_batch_designation(path, script_name: str | None) -> None:
    """Persist the designated Project Script name in ``batch.yaml``.

    The lazy-marker rule: ``None`` (each Project's own default script) never
    CREATES the file — it only clears the ``script:`` key of an existing one.
    Unknown keys are preserved.
    """
    file_path = os.path.join(str(path), BATCH_FILENAME)
    exists = os.path.isfile(file_path)
    if script_name is None and not exists:
        return
    payload: dict = {}
    if exists:
        try:
            with open(file_path, encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
        except Exception:  # noqa: BLE001
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
    if script_name is None:
        payload.pop("script", None)
    else:
        payload["script"] = script_name
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    temporary = file_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(temporary, file_path)


def resolve_designated_script(name: str | None, central_scripts: list[dict],
                              project) -> tuple[dict | None, str]:
    """Resolve the designated Project Script *name* for *project*.

    Order: the Batch's central ``project_scripts:``, then the Project's own
    ``scripts:``, then the built-ins.  Returns ``(script, source)``;
    ``(None, "")`` when the name resolves nowhere.

    ``None`` means "no designation": each Project runs its OWN default script.
    Every ``project.yaml`` is created with one, so there is no built-in
    fallback here — a Project whose ``scripts:`` is empty does not run, and
    says so.
    """
    from .script_editor.project_actions import (
        BUILTIN_PROJECT_SCRIPTS,
        builtin_project_script,
    )

    if not name:
        ## The seeded default by name, else the first authored script — a
        ## Project whose default was renamed still runs its own pipeline.
        own = project.find_script(DEFAULT_SCRIPT_NAME)
        if own is None:
            own = project.scripts[0] if project.scripts else None
        if own is None:
            return None, ""
        return own, f"{PROJECT_FILENAME} scripts"
    for script in central_scripts:
        if str(script.get("name", "")).strip() == name:
            return script, f"{BATCH_FILENAME} project_scripts"
    own = project.find_script(name)
    if own is not None:
        return own, f"{PROJECT_FILENAME} scripts"
    if name in BUILTIN_PROJECT_SCRIPTS:
        return builtin_project_script(name), "built-in"
    return None, ""


# ---------------------------------------------------------------------------
# Keys, and scoping an Exclusion Sheet to the run
# ---------------------------------------------------------------------------


def normalize_key(value) -> str:
    """A hand-typed path cell as a Project key.

    Sheets are written by hand, so a cell may carry a backslash separator, a
    leading ``./``, a trailing slash, a doubled separator, or an interior
    ``/./`` — all of them name the same Project, and the scope test and the
    write have to agree about which.
    """
    text = str(value or "").strip().replace("\\", "/")
    parts = [part for part in text.split("/") if part not in ("", ".")]
    return "/".join(parts)


#: Historical spelling kept for callers written against the sibling app's API.
normalize_member_key = normalize_key


def normalize_sheet_rows(rows) -> list:
    """Rewrite each row's ``project`` cell into a Project key.

    The scope test and the write must resolve the same cell the same way: a
    Windows-authored ``Sept2026\\ProjA`` that scopes in and then resolves to
    nothing is worse than one that never scoped in at all.
    """
    for row in rows:
        if getattr(row, "project", ""):
            row.project = normalize_key(row.project)
    return list(rows)


def row_project_key(row, keys) -> str:
    """Which Project a sheet row addresses, given the known *keys*.

    A hand-written sheet does not always put the whole path in the ``project``
    cell — ``project=Sept2026, member=ProjA/Rep1`` names the same Member as
    ``project=Sept2026/ProjA, member=Rep1``.  Scoping on the project cell alone
    lets the second spelling slip past the filter and write into a Project that
    was not running, so the row's full path is matched against the longest
    Project key that prefixes it.
    """
    whole = normalize_key(
        "/".join(part for part in (getattr(row, "project", ""),
                                   getattr(row, "member", "")) if part))
    folded = os.path.normcase(whole)
    best = ""
    for key in keys:
        candidate = os.path.normcase(normalize_key(key))
        if (folded == candidate or folded.startswith(candidate + "/")) \
                and len(candidate) > len(best):
            best = candidate
    return best


def scope_sheet_rows(rows, projects) -> tuple[list, list]:
    """Split *rows* into (for these Projects, for anyone else).

    Unchecking a Project means "do not touch this Project" — and with recursive
    discovery the walk surfaces Projects the user may never have known were
    there, so writing into one they excluded is the audit failure one step
    removed.
    """
    if projects is None:
        return list(rows), []
    ## normcase, because on Windows and macOS 'sept2026/proja' opens the same
    ## directory as 'Sept2026/ProjA': scoping a row out on the very filesystem
    ## where its path resolves would silently drop a declaration.
    wanted = {os.path.normcase(normalize_key(p)) for p in projects}
    keep, skip = [], []
    for row in rows:
        ## Matched on the row's whole path, not its project cell: the two
        ## columns are one path split at a place the author chose.
        key = row_project_key(row, wanted)
        ## Scoping is asked for, so a row must name one of them.  A blank
        ## project cell is "the root itself" — the Project-root sheet contract
        ## — but at a Batch root that contract does not apply, and honouring it
        ## would let a row write outside every checked Project.
        (keep if key else skip).append(row)
    return keep, skip


def apply_exclusion_sheet(batch_dir, log=print, projects=None) -> dict:
    """Apply the Batch's Exclusion Sheet, if it has one (ADR-0010).

    *projects* restricts the write to those keys — the ones actually running;
    ``None`` applies every row.  Never fatal: an unreadable sheet is reported
    and the run continues, and a row naming a project, member or chamber that
    does not exist is counted in the summary rather than aborting ten Projects'
    worth of work.
    """
    root = os.path.abspath(str(batch_dir))
    path = exclusion_sheet.find_sheet(root)
    if path is None:
        return {"results": [], "written": [], "counts": {}, "sheet": None}
    try:
        rows = normalize_sheet_rows(exclusion_sheet.read_sheet(path))
        rows, out_of_scope = scope_sheet_rows(rows, projects)
        if out_of_scope:
            log(f"[exclusions] {len(out_of_scope)} row(s) skipped — they name "
                "projects that are not in this run")
        if not rows:
            return {"results": [], "written": [], "counts": {}, "sheet": path,
                    "skipped": len(out_of_scope)}
        result = exclusion_sheet.apply_sheet(root, rows, sheet_path=path,
                                             log=log)
        result["skipped"] = len(out_of_scope)
        return result
    except Exception as err:  # noqa: BLE001
        log(f"[exclusions] {os.path.basename(path)} could not be applied: {err}")
        return {"results": [], "written": [], "counts": {}, "sheet": path,
                "error": str(err)}


def preview_exclusion_sheet(batch_dir, projects=None) -> dict:
    """What applying the sheet WOULD do — read-only (the preflight's preview).

    Runs the same matching as :func:`apply_exclusion_sheet` against a copy of
    each Member's declaration, so the preview and the write cannot disagree,
    and writes nothing: selecting a Batch reports, it never applies.
    """
    root = os.path.abspath(str(batch_dir))
    path = exclusion_sheet.find_sheet(root)
    if path is None:
        return {"sheet": None, "results": [], "counts": {}, "skipped": 0}
    try:
        rows = normalize_sheet_rows(exclusion_sheet.read_sheet(path))
    except Exception as err:  # noqa: BLE001
        return {"sheet": path, "results": [], "counts": {}, "skipped": 0,
                "error": str(err)}
    rows, out_of_scope = scope_sheet_rows(rows, projects)
    results = exclusion_sheet.plan_sheet(root, rows)
    counts: dict = {}
    for item in results:
        counts[item.status] = counts.get(item.status, 0) + 1
    return {"sheet": path, "results": results, "counts": counts,
            "skipped": len(out_of_scope)}


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run_batch(batch_dir, script_name: str | None = None,
              project_names: list[str] | None = None,
              log=print, apply_exclusions: bool = True) -> dict:
    """One Batch Run: the designated Project Script in every Project of
    *batch_dir* — continue-on-error, per-Project log prefixes.

    Returns ``{project_key: 'ok' | error message}``, keyed by each Project's
    path relative to *batch_dir*.  A Batch Run never creates or upgrades a
    ``project.yaml``.  *script_name* ``None`` falls back to the ``batch.yaml``
    designation, and with no designation each Project runs its own default
    script.  *project_names* restricts the run to that (checked) subset;
    *apply_exclusions* False declines the Exclusion Sheet for this run without
    touching it or any standing declaration.
    """
    from .script_editor.project_runner import run_project_script

    root = os.path.abspath(str(batch_dir))
    meta = load_batch_file(root)
    if script_name is None:
        script_name = meta["script"]

    found = discover(root)
    projects = found["projects"]
    by_key = {item.key: item for item in projects}

    if project_names is None:
        targets = [item.key for item in projects]
    else:
        wanted = {os.path.normcase(normalize_key(n)) for n in project_names}
        targets = [item.key for item in projects
                   if os.path.normcase(normalize_key(item.key)) in wanted]
        for missing in sorted(wanted - {os.path.normcase(normalize_key(k))
                                        for k in by_key}):
            ## An API caller's stale key must not vanish silently — the Hub
            ## table only offers real Projects, so this is for scripts.
            log(f"[{missing}] not a Project in this Batch — ignored")

    results: dict[str, str] = {}
    if not targets:
        log(f"No Projects to run in {root}")
        return results

    log(f"Batch Run: {len(targets)} Project(s) in {root}")

    ## Discovery is recursive, so say what it found: with Projects at arbitrary
    ## depth the target list is no longer obvious from the folder that was
    ## picked, and a per-Project member ratio stops "succeeded" being read as
    ## "analyzed everything".
    for key in targets:
        item = by_key[key]
        log(f"  {key} — {item.summary()}")
        for member in item.blocked:
            log(f"      {member.describe()}")

    for key, why in found["skipped"]:
        log(f"[{key}] skipped — {why}")
    if found.get("truncated"):
        ## A truncated walk that said nothing would report a partial run as a
        ## complete one.
        log(f"[batch] WARNING: stopped after {MAX_WALKED_DIRECTORIES} "
            "directories — this folder is larger than a batch should be, and "
            "projects deeper in it were not found.")

    for note in nested_batch_files(root):
        ## Only the selected Batch's designation governs; a sub-batch's own
        ## batch.yaml is ignored, and saying so beats wondering why the wrong
        ## script ran.
        log(f"[batch] {note} ignored — only the selected batch folder's "
            "designation applies")

    ## An Exclusion Sheet at the Batch root is applied before anything runs, so
    ## an unattended run honours the experimenter's notes (ADR-0010).  It is a
    ## writer, never an overlay: the rows are stamped into each Member's
    ## remove_chambers.csv and nothing reads the sheet at analysis time.
    ## Scoped to the Projects actually running: unchecking one means "do not
    ## touch this Project".
    if apply_exclusions:
        apply_exclusion_sheet(root, log=log, projects=targets)
    elif exclusion_sheet.find_sheet(root) is not None:
        log("[exclusions] sheet found but declined for this run — nothing "
            "written")

    for index, key in enumerate(targets, 1):
        directory = by_key[key].directory

        def plog(message, _k=key):
            log(f"[{_k}] {message}")

        log(f"[{index}/{len(targets)}] {key}")
        try:
            project = Project(directory)
        except Exception as err:  # noqa: BLE001
            ## A raising constructor (design mismatch, empty designless
            ## Project) is that Project's failure, never the run's.
            results[key] = f"{type(err).__name__}: {err}"
            plog(f"FAILED to load: {err}")
            continue
        script, source = resolve_designated_script(
            script_name, meta["project_scripts"], project)
        if script is None:
            results[key] = (
                f"no Project Script named '{script_name}' ({BATCH_FILENAME} "
                f"project_scripts, {PROJECT_FILENAME} scripts, or built-ins)"
                if script_name else
                f"{PROJECT_FILENAME} defines no Project Script — add one in "
                "the Script Editor (new Projects are created with a default "
                "one)")
            plog(f"SKIPPED — {results[key]}")
            continue
        plog(f"running '{script.get('name')}' (from {source})…")
        try:
            run_project_script(project, script, log=plog)
            results[key] = "ok"
        except Exception as err:  # noqa: BLE001
            ## str() of a bare exception can be empty — keep the type name so
            ## the summary always has something to say.
            results[key] = str(err) or type(err).__name__
            plog(f"FAILED: {results[key]}")

    ok = sum(1 for value in results.values() if value == "ok")
    log(f"Batch Run complete: {ok}/{len(targets)} Project(s) succeeded.")
    if ok < len(targets):
        log("Failed:")
        for key, message in results.items():
            if message != "ok":
                headline = message.splitlines()[0] if message else "<no message>"
                log(f"  {key}: {headline}")
    return results


# ---------------------------------------------------------------------------
# The object form the CLI and the Hub keep using
# ---------------------------------------------------------------------------


class Batch:
    """A loaded Batch: its Projects, its designation, and its central scripts.

    A thin object over :func:`discover` for callers that want one — the Hub
    uses the functions directly so it can cache one walk per selection.
    """

    def __init__(self, batch_dir):
        self.batch_directory = os.path.abspath(str(batch_dir))
        found = discover(self.batch_directory)
        self.projects: list = found["projects"]
        self.skipped: list = found["skipped"]
        self.truncated: bool = found["truncated"]
        self.project_paths = [item.directory for item in self.projects]

        self.meta = load_batch_file(self.batch_directory)
        self.name = os.path.basename(self.batch_directory)
        #: The designated Project Script, or None for "each Project's own".
        self.script_name: str | None = self.meta["script"]
        #: Central Project Scripts: one recipe serving every Project without
        #: being copied into each ``project.yaml``.
        self.project_scripts: list[dict] = self.meta["project_scripts"]

    def __len__(self) -> int:
        return len(self.projects)

    @property
    def project_names(self) -> list[str]:
        """Each Project's key — a path relative to the Batch root."""
        return [item.key for item in self.projects]

    def find_project_script(self, name: str) -> dict | None:
        for script in self.project_scripts:
            if script.get("name") == name:
                return script
        return None

    def resolve_script(self, project: Project, name: str | None = None):
        script, _source = resolve_designated_script(
            name or self.script_name, self.project_scripts, project)
        return script

    def run(self, script_name: str | None = None, log=print,
            project_names: list[str] | None = None,
            apply_exclusions: bool = True) -> dict:
        """Execute the designated Project Script in every Project."""
        results = run_batch(self.batch_directory,
                            script_name=script_name or self.script_name,
                            project_names=project_names, log=log,
                            apply_exclusions=apply_exclusions)
        ok = sum(1 for value in results.values() if value == "ok")
        return {"script": script_name or self.script_name, "results": results,
                "succeeded": ok, "failed": len(results) - ok,
                "skipped": [key for key, _why in self.skipped]}
