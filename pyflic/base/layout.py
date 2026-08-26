"""Experiment Directory layout: what a run can use, and filing what it cannot.

An Experiment Directory is loadable when its DFM exports sit in ``data/``.
Recordings do not arrive that way — the FLIC rig writes ``DFM<id>_<n>.csv``
wherever the acquisition PC was pointed, and somebody has to move them — so a
directory can be perfectly well-intentioned and still unusable.  This module
names those states and repairs the ones that are repairable:

* :func:`classify` — what a directory is, and why a run cannot use it.
* :func:`plan_filing` / :func:`file_recording` — move an **Unfiled Recording**
  into ``data/``, everything else loose into ``extra_files/``, and never a
  ``.yaml`` or an Exclusion Sheet: at an experiment root those are
  configuration or declaration, never data.  ``flic_config.yaml`` moved is an
  Experiment Directory un-made, and ``remove_chambers.csv`` moved silently
  returns excluded chambers to the analysis (ADR-0010).

"Loadable" is decided with the loader's own test — the same
``data/DFM<id>_*.csv`` spellings :mod:`pyflic.base.dfm` globs for — rather than
a lookalike.  A classifier that says "healthy" where the loader says "no DFM
data file(s) found" is worse than no classifier: it moves the failure from the
preflight, where someone is looking, into hour three of an unattended run.

Nothing here overwrites and nothing here guesses: a destination that already
exists is skipped, and a directory holding the same DFM id both loose and filed
refuses to file at all rather than deciding which copy the analysis is of.

Unlike PyTrackingAnalysis, whose experiment is one workbook, a FLIC recording
is *many* files — one or more per DFM — so "more than one recording here" is
the normal case and is never ambiguous on its own.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field

from .project import CONFIG_FILENAME

#: Where a recording belongs, and where everything else loose is parked.
DATA_DIRNAME = "data"
EXTRA_DIRNAME = "extra_files"

#: The DFM export spellings the loader accepts: ``DFM3_1.csv`` (v3),
#: ``DFM_3.csv`` and ``DFM_3_1.csv`` (v2).  Matched case-insensitively here so
#: a ``dfm3_1.csv`` is *reported* — the loader's glob is case-sensitive on
#: Linux, and "no data here" is baffling when the files are plainly visible.
_DFM_RE = re.compile(r"^DFM_?(\d+)(_\d+)?\.csv$", re.IGNORECASE)

#: The exact-case form the loader will actually find.
_DFM_RE_STRICT = re.compile(r"^DFM_?(\d+)(_\d+)?\.csv$")

#: Never moved.  At an experiment root a YAML file is the config or a
#: declaration sidecar; both are read from the root by definition.
KEEP_SUFFIXES = (".yaml", ".yml")

#: Excel writes ``~$Book.xlsx`` beside an open workbook, and dotfiles are the
#: filesystem's business, not ours.
_IGNORED_PREFIXES = ("~$", ".")

#: Subdirectories of a Project that are never Members.  Without this a
#: Project's own ``data/`` (one that also holds DFM CSVs, or a stray copy)
#: reads as an Unfiled Recording, and filing it would nest ``data/data/``.
_NOT_MEMBERS = {DATA_DIRNAME, EXTRA_DIRNAME, "analysis", "qc", "figures",
                "__pycache__"}

# Status values.  The empty string is "not experiment-shaped at all", which is
# not a problem to report — most subdirectories of anything are not.
NOT_AN_EXPERIMENT = ""
OK = "ok"
UNFILED = "unfiled recording"
NO_CONFIG = "no config"
NO_RECORDING = "no recording"
AMBIGUOUS = "ambiguous"
UNREADABLE = "unreadable"

#: Which action clears each status.  ``None`` means no button can fix it.
_FIX = {UNFILED: "file", NO_CONFIG: "config"}


@dataclass(frozen=True)
class MemberLayout:
    """What one directory is, from its layout alone — no config parsing.

    Named for the Project level it serves: a Project's subdirectories are its
    **Members**, and this is the per-Member fact the Batch walk, the Project
    panel, and the preflight all read.
    """

    directory: str
    name: str
    status: str
    detail: str = ""
    #: Loose recording files that filing would move into ``data/``.
    loose: tuple[str, ...] = ()
    #: Whether ``flic_config.yaml`` is present — the Project's own membership
    #: test, which is not the same question as "can it run".
    configured: bool = False
    #: DFM ids the loader would find in ``data/``.
    dfm_ids: tuple[int, ...] = ()

    @property
    def blocked(self) -> bool:
        """A run cannot use this directory as it stands."""
        return self.status not in (OK, NOT_AN_EXPERIMENT)

    @property
    def usable(self) -> bool:
        return self.status == OK

    @property
    def fix(self) -> str | None:
        """``"file"``, ``"config"``, or None when nothing can repair it."""
        return _FIX.get(self.status)

    def describe(self) -> str:
        return (f"{self.name}: {self.status}"
                + (f" — {self.detail}" if self.detail else ""))


# ---------------------------------------------------------------------------
# Reading a directory
# ---------------------------------------------------------------------------


def _entries(directory) -> list[str]:
    try:
        return sorted(os.listdir(directory))
    except OSError:
        return []


def _readable(directory) -> bool:
    try:
        os.listdir(directory)
    except OSError:
        return False
    return True


def data_dir(directory) -> str:
    """The directory's ``data/`` folder.

    Exact-case, deliberately: :func:`pyflic.base.yaml_config.load_experiment_yaml`
    builds ``experiment_dir / "data"`` and nothing else, so a ``Data/`` here is
    a directory the loader cannot see — and calling it ``data/`` would make the
    classifier say "healthy" where the loader says "no DFMs to load".
    """
    return os.path.join(str(directory), DATA_DIRNAME)


def dfm_files(directory) -> list[str]:
    """DFM export filenames in *directory* that the loader would actually find."""
    return sorted(name for name in _entries(directory)
                  if _DFM_RE_STRICT.match(name)
                  and os.path.isfile(os.path.join(str(directory), name)))


def dfm_ids(directory) -> list[int]:
    """DFM ids present in *directory*, parsed from the export filenames."""
    ids: set[int] = set()
    for name in dfm_files(directory):
        match = _DFM_RE_STRICT.match(name)
        if match:
            ids.add(int(match.group(1)))
    return sorted(ids)


def _dfm_lookalikes(directory) -> list[str]:
    """Files that read as DFM exports to a human but not to the loader —
    ``dfm3_1.csv`` on a case-sensitive filesystem."""
    strict = set(dfm_files(directory))
    return [name for name in _entries(directory)
            if _DFM_RE.match(name) and name not in strict
            and os.path.isfile(os.path.join(str(directory), name))]


def has_config(directory) -> bool:
    """``flic_config.yaml`` at the root — the Experiment membership test.

    Deliberately the same exact-case test as
    :func:`pyflic.base.project.is_experiment_dir`: a directory this says has no
    config is one the Project genuinely cannot see, which is what makes
    "scaffold a config" the right offer.
    """
    return os.path.isfile(os.path.join(str(directory), CONFIG_FILENAME))


def _is_sidecar(name: str) -> bool:
    """An Exclusion Sheet at an experiment root is a declaration, not data.

    ``remove_chambers.csv`` is a ``.csv`` like every DFM export, so without
    this filing would sweep the sheet into ``extra_files/`` and disarm it —
    the ADR-0010 failure the YAML exemption exists to prevent, reached through
    the one extension the allowlist actively moves.
    """
    from .exclusion_sheet import SHEET_STEM

    stem, _dot, _suffix = name.lower().partition(".")
    return stem == SHEET_STEM.lower()


def _loose_recording_files(directory) -> tuple[str, ...]:
    return tuple(dfm_files(directory))


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify(directory) -> MemberLayout:
    """What *directory* is: healthy Member, Blocked Member, or neither.

    Layout only — no YAML is parsed and no data is read, so this stays cheap
    enough to run over every subdirectory of every Project in a Batch.
    """
    directory = str(directory)
    name = os.path.basename(os.path.normpath(directory))
    if not os.path.isdir(directory):
        return MemberLayout(directory, name, NOT_AN_EXPERIMENT)
    if not _readable(directory):
        ## Unknown is not the same as absent: a directory nobody can list may
        ## hold a Member, and silently dropping it would overstate the
        ## Project's coverage in both the preflight and the run summary.
        return MemberLayout(directory, name, UNREADABLE,
                            "cannot be listed (permissions?)")

    configured = has_config(directory)
    data = data_dir(directory)
    filed = dfm_files(data)
    loose = dfm_files(directory)

    def _blocked(status, detail, loose_files=()):
        return MemberLayout(directory, name, status, detail, loose_files,
                            configured)

    if os.path.exists(data) and not os.path.isdir(data):
        return _blocked(AMBIGUOUS,
                        f"{DATA_DIRNAME} is a file, not a directory")

    if filed and loose:
        ## Many DFM files is the normal case here, so the ambiguity is never
        ## "which file" — it is "which COPY", and filing on top of a partly
        ## filed directory would merge two recordings into one experiment.
        both = sorted(set(dfm_ids(data)) & set(dfm_ids(directory)))
        if both:
            return _blocked(
                AMBIGUOUS,
                f"DFM {both[0]} appears both at the root and in "
                f"{DATA_DIRNAME}/ — which copy is the experiment?")
        return _blocked(
            UNFILED,
            f"{len(loose)} DFM file(s) sit at the root while "
            f"{len(filed)} are already in {DATA_DIRNAME}/",
            _loose_recording_files(directory))

    if filed:
        if configured:
            return MemberLayout(directory, name, OK, configured=True,
                                dfm_ids=tuple(dfm_ids(data)))
        return _blocked(
            NO_CONFIG,
            f"holds {len(filed)} DFM file(s) but no {CONFIG_FILENAME}")

    if loose:
        detail = (f"{len(loose)} DFM file(s) sit at the root, not in "
                  f"{DATA_DIRNAME}/")
        if not configured:
            detail += f"; no {CONFIG_FILENAME} either"
        return _blocked(UNFILED, detail, _loose_recording_files(directory))

    ## No DFM export the loader would find.  A lookalike is worth saying out
    ## loud: "no data here" is baffling when the files are plainly visible.
    lookalikes = _dfm_lookalikes(data) or _dfm_lookalikes(directory)
    if lookalikes and configured:
        return _blocked(
            NO_RECORDING,
            f"{lookalikes[0]} is not loadable — the loader looks for "
            "DFM<id>_<n>.csv with that exact spelling")

    if configured:
        return _blocked(NO_RECORDING,
                        f"{CONFIG_FILENAME} but no DFM CSV in {DATA_DIRNAME}/")
    return MemberLayout(directory, name, NOT_AN_EXPERIMENT)


def members_in(project_dir) -> list[MemberLayout]:
    """Classify every immediate subdirectory of *project_dir* that could be a
    Member, in name order.

    Symlinked directories are followed here — :attr:`Project.member_names`
    counts them, so refusing to would make a Project of symlinked Members look
    empty — while the Batch walk never follows one.  Output directories
    (``data/``, ``analysis/``, ``qc/``, ``figures/``) are excluded by name:
    they are the one thing that is reliably not a Member, and a Project root
    that is also an Experiment Directory would otherwise have its own ``data/``
    listed as an unfiled Member.
    """
    found: list[MemberLayout] = []
    #: Real paths already taken: a symlinked copy of a Member must not be
    #: counted (and analyzed, and pooled) twice — the same rule the Batch walk
    #: enforces one level up, and the one ``Project.__init__`` applies.
    seen: set[str] = set()
    for name in _entries(project_dir):
        if name.startswith("."):
            continue
        path = os.path.join(str(project_dir), name)
        if not os.path.isdir(path):
            continue
        ## Output directories are excluded by name — but only while they have
        ## no config of their own.  A Member someone named "data" is a Member:
        ## Project.member_names counts it, and discovery disagreeing with the
        ## Project about its own membership is worse than the odd name.
        if name.lower() in _NOT_MEMBERS and not has_config(path):
            continue
        real = os.path.realpath(path)
        if real in seen:
            continue
        seen.add(real)
        item = classify(path)
        if item.status != NOT_AN_EXPERIMENT:
            found.append(item)
    return found


# ---------------------------------------------------------------------------
# Filing an Unfiled Recording
# ---------------------------------------------------------------------------


@dataclass
class FilingPlan:
    """What filing *would* do — computed before anything moves, so the same
    plan can be shown to the user and then executed."""

    directory: str
    moves: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    refused: str = ""

    @property
    def possible(self) -> bool:
        return not self.refused and bool(self.moves)

    def describe(self) -> str:
        if self.refused:
            return f"cannot file: {self.refused}"
        if not self.moves:
            return "nothing to file"
        into_data = sum(
            1 for _s, d in self.moves
            if os.path.basename(os.path.dirname(d)) == DATA_DIRNAME)
        parts = [f"{into_data} file(s) → {DATA_DIRNAME}/"]
        extra = len(self.moves) - into_data
        if extra:
            parts.append(f"{extra} → {EXTRA_DIRNAME}/")
        if self.skipped:
            parts.append(f"{len(self.skipped)} skipped")
        return ", ".join(parts)


def _target_problem(directory, target) -> str:
    """Why *target* cannot receive files, or ``""``."""
    name = os.path.basename(target)
    if os.path.exists(target) and not os.path.isdir(target):
        return f"{name} exists as a file, not a directory"
    if os.path.islink(target):
        ## Following it would move the recording out of the Project tree, and
        ## two Members pointing at one target would contaminate each other.
        return f"{name} is a symlink"
    if not os.access(directory, os.W_OK):
        return "the experiment directory is not writable"
    return ""


def plan_filing(directory) -> FilingPlan:
    """Plan the moves that would make *directory* loadable.

    ``DFM*.csv`` goes to ``data/``; every other loose file goes to
    ``extra_files/``; ``.yaml``/``.yml``, an Exclusion Sheet, hidden files,
    Excel lock files, symlinks, and every subdirectory stay exactly where they
    are.  A destination that already exists is never overwritten — it is
    skipped and reported.
    """
    directory = str(directory)
    plan = FilingPlan(directory)
    item = classify(directory)
    if item.status != UNFILED:
        plan.refused = ("already filed" if item.status == OK else
                        item.detail or item.status or "not an experiment")
        return plan

    data = data_dir(directory)
    extra = os.path.join(directory, EXTRA_DIRNAME)
    problem = _target_problem(directory, data)
    if problem:
        plan.refused = problem
        return plan

    linked = [name for name in _loose_recording_files(directory)
              if os.path.islink(os.path.join(directory, name))]
    if linked:
        ## Filing some DFM files while refusing a symlinked one would leave
        ## the directory in a worse state than it started: a partial
        ## recording that loads and silently analyzes fewer DFMs.
        plan.refused = (f"{linked[0]} is a symlink — file this experiment by "
                        "hand")
        return plan

    for name in _entries(directory):
        source = os.path.join(directory, name)
        if not os.path.isfile(source):
            continue
        if os.path.islink(source):
            plan.skipped.append((name, "symlink — left where it is"))
            continue
        if name.startswith(_IGNORED_PREFIXES):
            continue
        if name.lower().endswith(KEEP_SUFFIXES) or _is_sidecar(name):
            continue
        recording = bool(_DFM_RE_STRICT.match(name))
        if not recording:
            ## Checked lazily: a stray file named extra_files/ must not refuse
            ## a filing whose only job is moving the recording.
            problem = _target_problem(directory, extra)
            if problem:
                plan.skipped.append((name, problem))
                continue
        target = data if recording else extra
        destination = os.path.join(target, name)
        if os.path.exists(destination):
            plan.skipped.append(
                (name, f"{os.path.basename(target)}/{name} already exists"))
            continue
        plan.moves.append((source, destination))

    if not plan.moves and not plan.refused:
        plan.refused = "nothing here can be moved"
    return plan


def file_recording(directory, log=None) -> FilingPlan:
    """Execute :func:`plan_filing` for *directory*; returns the plan as run.

    Moves are attempted one at a time and a failure is recorded against that
    file rather than raised: a half-filed directory that says which half is
    recoverable, and one unwritable file must not strand the rest.  A move that
    leaves the source behind (a cross-filesystem copy that failed partway) is
    reported as such rather than counted as done — that state reads as
    "ambiguous" afterwards, and silently converting a one-click fix into a
    permanent block is the worst outcome available.
    """
    plan = plan_filing(directory)
    if not plan.possible:
        if log and plan.refused:
            log(f"[file] {os.path.basename(str(directory))}: {plan.refused}")
        return plan

    done: list[tuple[str, str]] = []
    for source, destination in plan.moves:
        parent = os.path.dirname(destination)
        name = os.path.basename(source)
        try:
            os.makedirs(parent, exist_ok=True)
            ## Re-check under the directory we just made: between planning and
            ## here, the destination may have appeared.
            if os.path.exists(destination):
                plan.skipped.append((name, "already exists — not overwritten"))
                continue
            shutil.move(source, destination)
        except OSError as err:
            plan.skipped.append((name, str(err)))
            continue
        if os.path.exists(source):
            plan.skipped.append(
                (name, "copied but the original is still here — move it by "
                       "hand before running"))
            continue
        done.append((source, destination))
        if log:
            log(f"[file] {name} → {os.path.basename(parent)}/")
    if not done and plan.skipped:
        ## Otherwise describe() reports "nothing to file" for a directory
        ## where every single move was refused by the filesystem.
        plan.refused = f"nothing could be moved ({plan.skipped[0][1]})"
    plan.moves = done
    return plan
