"""Experiment Directory layout: what a run can use, and filing what it cannot.

An Experiment Directory is loadable when its DFM exports sit in ``data/``.
Recordings do not arrive that way — the FLIC rig writes ``DFM<id>_<n>.csv``
wherever the acquisition PC was pointed, and somebody has to move them — so a
directory can be perfectly well-intentioned and still unusable.  This module
names those states and repairs the ones that are repairable:

* :func:`classify` — what a directory is, and why a run cannot use it.
* :func:`plan_filing` / :func:`file_recording` — move an **Unfiled Recording**
  into ``data/`` (its ``Program.txt`` and ``Messages.csv`` with it),
  everything else loose into ``extra_files/``, and never a
  ``.yaml`` or an Exclusion Sheet: at an experiment root those are
  configuration or declaration, never data.  ``flic_config.yaml`` moved is an
  Experiment Directory un-made, and ``remove_chambers.csv`` moved silently
  returns excluded chambers to the analysis (ADR-0010).

"Loadable" is decided with the loader's own test — the same
``data/DFM<id>_<n>.csv`` spellings :mod:`pyflic.base.dfm` reads
(:data:`pyflic.base.utils.DFM_FILE_RE`) — rather than a lookalike.  A classifier that says "healthy" where the loader says "no DFM
data file(s) found" is worse than no classifier: it moves the failure from the
preflight, where someone is looking, into hour three of an unattended run.

Nothing here overwrites and nothing here guesses: a destination that already
exists is skipped, and a directory holding the same DFM id both loose and filed
refuses to file at all rather than deciding which copy the analysis is of.

Unlike PyTrackingAnalysis, whose experiment is one workbook, a FLIC recording
is *many* files — one or more per DFM — so "more than one recording here" is
the normal case and is never ambiguous on its own.

**Where the recording is** is answered once, by :func:`data_location`
(ADR-0021): the Member's own ``data/``, or the folder its ``data_folder:`` key
names.  The loader, the disk cache, this classifier, the Project, Batch
discovery, the linter, the Project Report and the Config Editor all ask it,
so none of them can read a different folder from the others.  A referenced
recording is only ever read: it is never filed, moved, or written beside.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field

import yaml

from .project import CONFIG_FILENAME
from .utils import DFM_FILE_RE, dfm_file_id, is_environmental_monitor

#: Where a recording belongs, and where everything else loose is parked.
DATA_DIRNAME = "data"
EXTRA_DIRNAME = "extra_files"

#: The top-level ``flic_config.yaml`` key naming the folder a Member reads its
#: recording from instead of its own ``data/`` (ADR-0021).  It belongs to that
#: one config: never under ``global:``, never in a Project design, never
#: inherited by a scaffolded Member.
DATA_FOLDER_KEY = "data_folder"

#: The DFM export spellings the loader accepts: ``DFM3_1.csv`` (v3),
#: ``DFM_3.csv`` and ``DFM_3_1.csv`` (v2).  Matched case-insensitively here,
#: and with an optional part, so a ``dfm3_1.csv`` or a ``DFM3.csv`` is
#: *reported* — the loader reads the exact spellings only, and "no data here"
#: is baffling when the files are plainly visible.
_DFM_RE = re.compile(r"^DFM_?(\d+)(_\d+)?\.csv$", re.IGNORECASE)

#: The exact-case forms the loader will actually find: its own rule.
_DFM_RE_STRICT = DFM_FILE_RE

#: Never moved.  At an experiment root a YAML file is the config or a
#: declaration sidecar; both are read from the root by definition.
KEEP_SUFFIXES = (".yaml", ".yml")

#: What PyMCU exports beside the DFM files, filed into ``data/`` with them:
#: the loader reads the Opto Program (``opto_program.find_program``), the
#: message log (``messages.find_messages``) and the Baseline sets
#: (``paradigm_state.read_baseline_sets``) from ``data/`` only, so any of them
#: parked in ``extra_files/`` is silently lost.  Lower case: all are found
#: case-insensitively.
_RECORDING_COMPANIONS = ("program.txt", "messages.csv", "baselines.csv")

#: Excel writes ``~$Book.xlsx`` beside an open workbook, and dotfiles are the
#: filesystem's business, not ours.
_IGNORED_PREFIXES = ("~$", ".")

#: Subdirectories of a Project that are never Members.  Without this a
#: Project's own ``data/`` (one that also holds DFM CSVs, or a stray copy)
#: reads as an Unfiled Recording, and filing it would nest ``data/data/``.
_NOT_MEMBERS = {DATA_DIRNAME, EXTRA_DIRNAME, "analysis", "qc", "figures",
                "slices", "__pycache__"}

# Status values.  The empty string is "not experiment-shaped at all", which is
# not a problem to report — most subdirectories of anything are not.
NOT_AN_EXPERIMENT = ""
OK = "ok"
UNFILED = "unfiled recording"
NO_CONFIG = "no config"
NO_RECORDING = "no recording"
AMBIGUOUS = "ambiguous"
UNREADABLE = "unreadable"
#: A second Member reading a recording an earlier one already reads (ADR-0021).
DUPLICATE = "duplicate"

#: Which action clears each status.  ``None`` means no button can fix it.
_FIX = {UNFILED: "file", NO_CONFIG: "config"}


@dataclass(frozen=True)
class MemberLayout:
    """What one directory is, from its layout alone — no data read, and the
    config parsed only when it names a ``data_folder:``.

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
    """The directory's own ``data/`` folder.

    Exact-case, deliberately: the loader reads ``experiment_dir / "data"`` and
    nothing else, so a ``Data/`` here is a directory the loader cannot see —
    and calling it ``data/`` would make the classifier say "healthy" where the
    loader says "no DFMs to load".

    This is the folder a Member reads when its config names no
    ``data_folder:``, and the one filing moves a recording into.  "Where is the
    recording?" is :func:`data_location`'s question — the loader asks it too,
    so the classifier and the loader cannot end up reading different folders.
    """
    return os.path.join(str(directory), DATA_DIRNAME)


def _recording_files(directory) -> list[str]:
    """Every file of the recording PyMCU names ``DFM…csv``, the Environmental
    Monitor's ``DFM99_<n>.csv`` included: filing moves them together."""
    return sorted(name for name in _entries(directory)
                  if _DFM_RE_STRICT.match(name)
                  and os.path.isfile(os.path.join(str(directory), name)))


def _is_monitor_file(name: str) -> bool:
    match = _DFM_RE.match(name)
    return bool(match) and is_environmental_monitor(match.group(1))


def dfm_files(directory) -> list[str]:
    """DFM export filenames in *directory* that the loader would actually
    find.  The Environmental Monitor's ``DFM99_<n>.csv`` is not a DFM."""
    return [name for name in _recording_files(directory) if not _is_monitor_file(name)]


def dfm_ids(directory) -> list[int]:
    """DFM ids present in *directory*, parsed from the export filenames."""
    return sorted({dfm_file_id(name) for name in dfm_files(directory)})


def _dfm_lookalikes(directory) -> list[str]:
    """Files that read as DFM exports to a human but not to the loader —
    ``dfm3_1.csv`` on a case-sensitive filesystem."""
    return [name for name in _entries(directory)
            if _DFM_RE.match(name) and not _DFM_RE_STRICT.match(name)
            and not _is_monitor_file(name)
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
    return tuple(_recording_files(directory))


# ---------------------------------------------------------------------------
# Where a Member's recording is (ADR-0021)
# ---------------------------------------------------------------------------


class DataLocationError(ValueError):
    """A Member's ``data_folder:`` cannot be read as it stands — the folder is
    missing, is a file, holds the Member itself, or competes with a recording
    of the Member's own.  The message names the key, the value as written and
    the resolved path."""


#: Why a referenced location cannot be used; the classifier maps each to a
#: Blocked Member status.
_MISSING = "missing"
_NOT_A_FOLDER = "not a folder"
_NOT_A_PATH = "not a path"
_CONFLICT = "conflict"
_HOLDS_MEMBER = "holds the member"


@dataclass(frozen=True)
class DataLocation:
    """Where one Member's recording is read from.

    *path* is the folder the loader reads.  *written* is the ``data_folder:``
    value exactly as the config spells it, or None when the Member reads its
    own ``data/`` — in which case *path* is :func:`data_dir`'s answer and
    nothing else about the Member has changed.  *problem* says why the loader
    cannot read a referenced location; it is empty when it can.
    """

    path: str
    written: str | None = None
    problem: str = ""
    kind: str = ""

    @property
    def referenced(self) -> bool:
        """The Member names its data location in its config."""
        return self.written is not None

    @property
    def identity(self) -> str:
        """What two locations are compared by: one recording, one Member.
        ``normcase`` because the lab's share is case-insensitive."""
        return os.path.normcase(os.path.realpath(self.path))

    def describe(self) -> str:
        """The location said both ways, for a message or a report: the
        resolved path, then the key as written.  Just the path without the
        key."""
        if not self.referenced:
            return str(self.path)
        return f"{self.path} ({DATA_FOLDER_KEY}: {self.written})"


def configured_data_folder(directory, config=None, config_path=None):
    """The ``data_folder:`` value for *directory*, as written, or None.

    *config* is a config already parsed; otherwise the file at *config_path*
    (default ``<directory>/flic_config.yaml``) is read.  A config that never
    mentions the key is not parsed at all, so a Member without one costs a
    file read and nothing it did before can change.  A config that cannot be
    read or parsed counts as having no key: the loader reports a broken
    config itself, and the classifier keeps the answer it always gave.  A
    blank value is no key.
    """
    if config is None:
        path = config_path or os.path.join(str(directory), CONFIG_FILENAME)
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                text = handle.read()
        except OSError:
            return None
        if DATA_FOLDER_KEY not in text:
            return None
        try:
            config = yaml.safe_load(text)
        except yaml.YAMLError:
            return None
    if not isinstance(config, dict):
        return None
    value = config.get(DATA_FOLDER_KEY)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return value


def resolve_data_folder(directory, value) -> str:
    """*value* resolved against the Member folder *directory*.

    An absolute value is kept as it is.  Resolution is the operating system's
    own — ``realpath``, so ``..`` is taken after any symlink, exactly as an
    ``open()`` of the joined path would take it.  Windows accepts backslashes
    as separators natively; elsewhere a backslash is part of a name, which is
    why the linter asks for forward slashes.
    """
    return os.path.realpath(os.path.join(os.path.abspath(str(directory)), str(value)))


def _contains(folder, path) -> bool:
    """*path* is *folder* or somewhere beneath it."""
    folder = os.path.normcase(os.path.realpath(folder))
    path = os.path.normcase(os.path.realpath(path))
    try:
        return os.path.commonpath([folder, path]) == folder
    except ValueError:                      # different drives
        return False


def _recording_parts(directory) -> list[str]:
    """Files in *directory* that would be part of a recording: a DFM export in
    any spelling (the Environmental Monitor's too) or one of the files PyMCU
    exports beside them, never a hidden file or an Excel lock file.  Used for a
    Member's own ``data/``, where anything shaped like part of a recording is
    one."""
    return [name for name in _entries(directory)
            if (_DFM_RE.match(name) or name.lower() in _RECORDING_COMPANIONS)
            and not name.startswith(_IGNORED_PREFIXES)
            and os.path.isfile(os.path.join(str(directory), name))]


def _own_recording(directory, target) -> str:
    """A recording the Member folder holds itself, as a path relative to it,
    or ``""``.

    In its own ``data/``, anything shaped like part of a recording counts
    (:func:`_recording_parts`).  At its root, only what the ordinary
    classifier counts as a loose recording — the DFM exports the loader would
    find (:func:`dfm_files`) — so a ``units.csv`` or a ``notes.txt`` kept
    beside the config is not mistaken for a second recording.

    A folder that IS the target does not count: ``data_folder: data`` names
    the Member's own ``data/``, and that is one recording, not two.  An empty
    ``data/`` holds nothing, so it is not a conflict either.
    """
    places = []
    local = data_dir(directory)
    if os.path.isdir(local):
        places.append((local, DATA_DIRNAME + "/", _recording_parts))
    places.append((str(directory), "", dfm_files))
    target = os.path.normcase(os.path.realpath(target))
    for folder, prefix, recording in places:
        if os.path.normcase(os.path.realpath(folder)) == target:
            continue
        found = recording(folder)
        if found:
            return prefix + found[0]
    return ""


def data_location(directory, *, config=None, config_path=None) -> DataLocation:
    """Where the Member in *directory* reads its recording from.

    Without ``data_folder:`` this is :func:`data_dir` — today's exact-case
    ``data/`` — and nothing more is checked.  With it, the value is resolved
    against *directory* and checked, in this order, for the problems that make
    the loader refuse it: a value that is not a path; a location holding the
    Member folder itself (everything pyflic writes for the Member goes there,
    and nothing is ever written at a data location); a recording of the
    Member's own beside the key ("which is the experiment?"); a location that
    does not exist; and a file where a folder belongs.  Each problem names the
    key, the value as written and the resolved path.

    *config* (already parsed) or *config_path* (another config file) are for
    a caller that has one; the default reads ``flic_config.yaml``.
    """
    directory = str(directory)
    value = configured_data_folder(directory, config, config_path)
    if value is None:
        return DataLocation(data_dir(directory))

    written = value if isinstance(value, str) else repr(value)
    path = resolve_data_folder(directory, written)
    where = f"{DATA_FOLDER_KEY}: {written} resolves to {path}"

    def refused(kind, problem):
        return DataLocation(path, written, problem, kind)

    if not isinstance(value, str):
        return refused(_NOT_A_PATH, f"{DATA_FOLDER_KEY}: {written} is not a folder path")
    if _contains(path, directory):
        return refused(
            _HOLDS_MEMBER,
            f"{where}, which holds the experiment folder "
            f"{os.path.realpath(directory)} itself — the analysis is written "
            "there, and nothing is ever written at a data location")
    own = _own_recording(directory, path)
    if own:
        return refused(
            _CONFLICT,
            f"{where}, but {directory} also holds a recording of its own "
            f"({own}) — which is the experiment?  Remove {DATA_FOLDER_KEY}: "
            "or move the local recording out")
    if not os.path.exists(path):
        return refused(_MISSING, f"{where}, which does not exist")
    if not os.path.isdir(path):
        return refused(_NOT_A_FOLDER, f"{where}, which is a file, not a folder")
    return DataLocation(path, written)


def require_data_location(directory, *, config=None, config_path=None) -> DataLocation:
    """:func:`data_location`, raising :class:`DataLocationError` for a
    referenced location the loader cannot read.  The loader's entry point."""
    location = data_location(directory, config=config, config_path=config_path)
    if location.problem:
        source = config_path or os.path.join(str(directory), CONFIG_FILENAME)
        raise DataLocationError(f"{location.problem} (set in {source})")
    return location


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify(directory) -> MemberLayout:
    """What *directory* is: healthy Member, Blocked Member, or neither.

    Layout only — no data is read, and the config is parsed only when it
    mentions ``data_folder:`` (:func:`data_location`), so this stays cheap
    enough to run over every subdirectory of every Project in a Batch.
    """
    directory = str(directory)
    if not os.path.isdir(directory):
        return MemberLayout(directory, _name(directory), NOT_AN_EXPERIMENT)
    return _classify(directory, data_location(directory))


def _name(directory) -> str:
    return os.path.basename(os.path.normpath(str(directory)))


def _classify(directory, location: DataLocation) -> MemberLayout:
    """:func:`classify` for a directory whose data location is already known
    — the same answer, so :func:`members_in` can compare locations without
    resolving each one twice."""
    name = _name(directory)
    if not _readable(directory):
        ## Unknown is not the same as absent: a directory nobody can list may
        ## hold a Member, and silently dropping it would overstate the
        ## Project's coverage in both the preflight and the run summary.
        return MemberLayout(directory, name, UNREADABLE,
                            "cannot be listed (permissions?)")
    if location.referenced:
        return _classify_referenced(directory, name, location)

    configured = has_config(directory)
    data = location.path
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


def _classify_referenced(directory, name, location: DataLocation) -> MemberLayout:
    """A Member whose config names its data folder (ADR-0021).

    Decided by the loader's rule, applied to the folder the loader will read:
    healthy when that folder holds a DFM export the loader would find,
    blocked otherwise with the resolved path in the reason.  Never an
    Unfiled Recording — DFM files in a referenced folder are where the loader
    looks, and DFM files at the Member root beside the key are a conflict to
    settle by hand, not something to file.
    """
    def blocked(status, detail):
        return MemberLayout(directory, name, status, detail, (), True)

    if location.problem:
        status = (AMBIGUOUS if location.kind in (_CONFLICT, _HOLDS_MEMBER)
                  else NO_RECORDING)
        return blocked(status, location.problem)

    where = f"{DATA_FOLDER_KEY}: {location.written} ({location.path})"
    if dfm_files(location.path):
        return MemberLayout(directory, name, OK, configured=True,
                            dfm_ids=tuple(dfm_ids(location.path)))
    lookalikes = _dfm_lookalikes(location.path)
    if lookalikes:
        return blocked(NO_RECORDING,
                       f"{lookalikes[0]} in {where} is not loadable — the loader "
                       "looks for DFM<id>_<n>.csv with that exact spelling")
    detail = f"no DFM CSV in {where}"
    nested = data_dir(location.path)
    inside = dfm_files(nested) if os.path.isdir(nested) else []
    if inside:
        ## The usual slip: the key names the run folder, one level above the
        ## data/ that holds the recording.
        detail += (f"; its {DATA_DIRNAME}/ holds {inside[0]} — point "
                   f"{DATA_FOLDER_KEY}: at that folder")
    return blocked(NO_RECORDING, detail)


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

    A configured Member that reads a recording an earlier one already reads
    is reported as a :data:`DUPLICATE` (ADR-0021), by the rule
    ``Project.__init__`` applies (:class:`DataLocations`).
    """
    found: list[MemberLayout] = []
    #: Real paths already taken: a symlinked copy of a Member must not be
    #: counted (and analyzed, and pooled) twice — the same rule the Batch walk
    #: enforces one level up, and the one ``Project.__init__`` applies.
    seen: set[str] = set()
    #: Data locations already read, for the same rule one step further: two
    #: Members naming one recording are one Member (ADR-0021).
    reads = DataLocations()
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
        location = data_location(path)
        if has_config(path):
            first = reads.claim(name, location)
            if first is not None:
                found.append(MemberLayout(path, name, DUPLICATE,
                                          duplicate_detail(first, location),
                                          configured=True))
                continue
        item = _classify(path, location)
        if item.status != NOT_AN_EXPERIMENT:
            found.append(item)
    return found


class DataLocations:
    """The data locations a Project's Members read, first come first served.

    One recording is analyzed once per Project (ADR-0009, extended by
    ADR-0021): a Member whose data location an earlier one — in sorted order
    — already reads is a duplicate.  Locations compare by
    ``normcase(realpath(...))``.  The rule applies only when one of the two
    names its location with ``data_folder:``; two plain Members whose
    ``data/`` folders happen to be one folder are left as they always were.
    """

    def __init__(self) -> None:
        self._taken: dict[str, tuple[str, DataLocation]] = {}

    def claim(self, name: str, location: DataLocation) -> str | None:
        """Record *name* reading *location*; return the earlier Member it
        duplicates, or None when it is the first to read it."""
        first = self._taken.get(location.identity)
        if first is not None and (location.referenced or first[1].referenced):
            return first[0]
        self._taken.setdefault(location.identity, (name, location))
        return None


def duplicate_detail(first: str, location: DataLocation) -> str:
    """Why a Member is a duplicate, naming the one that is kept."""
    return (f"reads the same recording as {first} ({location.path}) — only "
            f"{first} is analyzed")


def initializable_dirs(project_dir) -> list[MemberLayout]:
    """Immediate subdirectories of *project_dir* with no ``flic_config.yaml``.

    The candidates for "initialize this existing folder as a Member".
    :func:`members_in` answers the narrower question — what is already
    experiment-*shaped* — and a folder somebody made and dropped a recording
    into loose, or has not filled yet, is exactly the one that needs
    initializing.  It must not be invisible for not being shaped like a Member
    yet.

    Output directories are excluded by name, as there; a directory that already
    has a config is a Member and has nothing to initialize.
    """
    found: list[MemberLayout] = []
    seen: set[str] = set()
    for name in _entries(project_dir):
        if name.startswith("."):
            continue
        path = os.path.join(str(project_dir), name)
        if not os.path.isdir(path) or has_config(path):
            continue
        if name.lower() in _NOT_MEMBERS:
            continue
        real = os.path.realpath(path)
        if real in seen:
            continue
        seen.add(real)
        found.append(classify(path))
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

    ``DFM*.csv``, ``Program.txt`` and ``Messages.csv`` go to ``data/``, where
    the loader reads them; every other loose file goes to
    ``extra_files/``; ``.yaml``/``.yml``, an Exclusion Sheet, hidden files,
    Excel lock files, symlinks, and every subdirectory stay exactly where they
    are.  A destination that already exists is never overwritten — it is
    skipped and reported.
    """
    directory = str(directory)
    plan = FilingPlan(directory)
    location = data_location(directory)
    if location.referenced:
        ## Its recording lives where data_folder: says, and is only ever read
        ## (ADR-0021).  Whatever is loose here is a conflict to settle by hand,
        ## not a recording to move.
        plan.refused = (f"its {CONFIG_FILENAME} reads the recording from "
                        f"{DATA_FOLDER_KEY}: {location.written} — a member that "
                        "names its data folder is never filed")
        return plan
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
        recording = (bool(_DFM_RE_STRICT.match(name))
                     or name.lower() in _RECORDING_COMPANIONS)
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
