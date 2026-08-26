"""The Exclusion Sheet: one spreadsheet declaring excluded chambers in bulk.

An **Excluded Chamber** is already a per-Member fact — a row in that Member's
``remove_chambers.csv`` under the active exclusion group.  That file is the
audit trail, and nothing here changes it.  What was missing is the *authoring*
half: an experimenter who watched forty recordings has forty directories to
edit, and doing it forty times by hand is how chambers stop getting declared.

An **Exclusion Sheet** is a ``remove_chambers.csv`` (or ``.xlsx``) at a **Batch
root** or a **Project root** — one level *above* the per-Member files — whose
rows name a Member and a chamber.  Applying it writes those rows down into each
Member's own file.  It is a **writer, never an overlay**: nothing reads the
sheet at analysis time, so a sheet that is deleted, moved, or never applied
changes no result, and every removal that reaches an analysis is visible in the
Member's own file where the analysis already stamps it.

Three rules keep it from becoming a second source of truth:

* **The standing declaration wins.**  A chamber already declared is never
  rewritten; a differing note is reported as a **conflict** rather than
  applied.  A Batch Run re-applies the sheet every time, so letting the sheet
  win would keep resetting notes refined by hand.
* **Selecting a Batch reports; it never applies.**  Browsing to a colleague's
  batch folder must not rewrite eighty directories.
* **Rows are scoped to the run.**  A row naming a Project that is not in the
  run is skipped and reported (see :func:`pyflic.base.batch.scope_sheet_rows`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

#: The sheet: same stem as the per-Member file it writes into, at a Batch or
#: Project root.  ``.csv`` wins when both spellings are present.
SHEET_STEM = "remove_chambers"
SHEET_SUFFIXES = (".csv", ".xlsx", ".xls")

#: What a declaration without a note means.  The experimenter can overwrite it;
#: the point is that every exclusion carries *something* into the audit.
DEFAULT_REASON = "Undefined"

#: The exclusion group a row lands in when it names none.  Matches
#: ``Project.exclusion_group``'s own default.
DEFAULT_GROUP = "general"

#: Header spellings accepted for each field (compared case- and
#: space-insensitively), so a sheet written by hand does not have to guess.
_SHEET_COLUMNS = {
    "project": ("project", "projectname", "projectdir", "projectdirectory"),
    "member": ("member", "membername", "experiment", "experimentname",
               "replicate", "replicatename", "experimentdir",
               "experimentdirectory"),
    "dfm": ("dfm", "dfmid", "dfmnumber", "monitor"),
    "chamber": ("chamber", "chamberindex", "chamberid", "well", "tube"),
    "group": ("group", "exclusiongroup", "set"),
    "reason": ("reason", "note", "notes", "comment", "comments"),
}


@dataclass
class SheetRow:
    """One row of an Exclusion Sheet, as read."""

    index: int          # 1-based row number as a spreadsheet shows it
    project: str
    member: str
    dfm: str
    chamber: str
    group: str
    reason: str

    @property
    def where(self) -> str:
        parts = [p for p in (self.project, self.member) if p]
        cell = f"DFM {self.dfm} chamber {self.chamber}" \
            if (self.dfm or self.chamber) else ""
        return "/".join(parts) + (f" {cell}" if cell else "")


@dataclass
class RowResult:
    """What applying one :class:`SheetRow` did."""

    row: SheetRow
    #: applied | already declared | conflict | unknown project |
    #: unknown member | unknown chamber | incomplete
    status: str
    detail: str = ""

    @property
    def wrote(self) -> bool:
        return self.status == "applied"

    def describe(self) -> str:
        line = f"row {self.row.index} {self.row.where or '(blank)'}: {self.status}"
        return f"{line} — {self.detail}" if self.detail else line


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def find_sheet(root) -> str | None:
    """The Exclusion Sheet at *root*, or ``None``.  ``.csv`` beats ``.xlsx``.

    Matched case-insensitively: "Save As" in Excel routinely produces
    ``Remove_Chambers.csv``, and an exact-case test makes that sheet invisible
    on Linux — the preflight then reports that the batch simply has no sheet,
    and the run applies nothing with no line anywhere saying so.
    """
    try:
        entries = sorted(os.listdir(str(root)))
    except OSError:
        return None
    for suffix in SHEET_SUFFIXES:
        wanted = (SHEET_STEM + suffix).lower()
        for entry in entries:
            path = os.path.join(str(root), entry)
            if entry.lower() == wanted and os.path.isfile(path):
                return path
    return None


def looks_like_sheet(path) -> bool:
    """True when *path* carries sheet headers rather than per-Member ones.

    The two files share a stem, so a ``remove_chambers.csv`` that someone
    copied *up* from a Member (``group,dfm_id,chamber,note``) would otherwise
    read as a sheet with no member column and report every row incomplete.
    """
    try:
        headers = _read_frame(path).columns
    except Exception:  # noqa: BLE001
        return False
    canon = {_canon(h) for h in headers}
    return bool(canon & set(_SHEET_COLUMNS["member"]))


def _canon(header) -> str:
    return "".join(ch for ch in str(header).lower() if ch.isalnum())


def _read_frame(path):
    import pandas as pd

    if str(path).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path, dtype=str)
    return pd.read_csv(path, dtype=str)


def read_sheet(path) -> list[SheetRow]:
    """Parse an Exclusion Sheet into rows.

    Headers are matched case-insensitively and ignoring spaces/underscores;
    ``project``, ``group`` and ``reason`` are optional (a Project-root sheet
    needs no project column, a missing group is :data:`DEFAULT_GROUP`, and a
    missing reason becomes :data:`DEFAULT_REASON`).  Raises ``ValueError`` when
    the required columns are absent — an unparseable sheet is a mistake worth
    stopping for, unlike an unmatched row.
    """
    import pandas as pd

    frame = _read_frame(path)
    mapping: dict[str, str] = {}
    for column in frame.columns:
        key = _canon(column)
        for field, spellings in _SHEET_COLUMNS.items():
            if key in spellings and field not in mapping:
                mapping[field] = column
    missing = [f for f in ("member", "dfm", "chamber") if f not in mapping]
    if missing:
        raise ValueError(
            f"{os.path.basename(str(path))} is missing the "
            f"{', '.join(missing)} column(s).  Expected headers: "
            "project, member, dfm, chamber, group, reason.")

    def cell(row, field) -> str:
        column = mapping.get(field)
        if column is None:
            return ""
        value = row.get(column)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        text = str(value).strip()
        return "" if text.lower() == "nan" else text

    rows: list[SheetRow] = []
    for offset, (_, raw) in enumerate(frame.iterrows(), start=2):
        row = SheetRow(index=offset,
                       project=cell(raw, "project"),
                       member=cell(raw, "member"),
                       dfm=cell(raw, "dfm"),
                       chamber=cell(raw, "chamber"),
                       group=cell(raw, "group") or DEFAULT_GROUP,
                       reason=cell(raw, "reason") or DEFAULT_REASON)
        if not (row.project or row.member or row.dfm or row.chamber):
            continue                      # a blank spacer row
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Matching and writing
# ---------------------------------------------------------------------------


def _canonical(path) -> str:
    """One spelling per directory: resolved, and case-folded where the
    filesystem is (Windows/macOS), so the same folder is never two keys."""
    return os.path.normcase(os.path.realpath(str(path)))


def _inside(root, path) -> bool:
    """Is *path* within *root*?  Guards a hand-edited sheet from writing
    outside the tree it was found in."""
    root_real = os.path.realpath(str(root))
    target = os.path.realpath(str(path))
    try:
        return os.path.commonpath([root_real, target]) == root_real
    except ValueError:              # different drives on Windows
        return False


def _int(text) -> int | None:
    try:
        return int(str(text).strip())
    except (TypeError, ValueError):
        return None


def plan_sheet(root, rows) -> list:
    """What applying *rows* under *root* would do — writes nothing.

    The preflight's preview and the run's write share this one evaluation, so
    what the user was shown cannot disagree with what happens.
    """
    results, _pending = _evaluate_sheet(root, rows)
    return results


def _evaluate_sheet(root, rows) -> tuple:
    """Match every row against the tree; returns ``(results, pending)``.

    Pure: per-Member declarations accumulate in memory and are handed back for
    the caller to write — or to throw away, for a preview.
    """
    from . import project as prj
    from .batch import normalize_key
    from .exclusions import read_exclusion_notes

    root = str(root)
    results: list[RowResult] = []
    #: ``{member_dir: {group: {dfm: {chamber: note}}}}``, written once at the
    #: end so a sheet touching one Member twenty times writes one file.
    pending: dict[str, dict] = {}
    known_dfms: dict[str, set] = {}

    for row in rows:
        dfm = _int(row.dfm)
        chamber = _int(row.chamber)
        if not row.member or dfm is None or chamber is None:
            results.append(RowResult(
                row, "incomplete",
                "member, dfm and chamber are required, and dfm/chamber must "
                "be whole numbers"))
            continue

        ## One seam normalizes the cell, so scoping and resolution cannot
        ## disagree: a Windows-authored 'Sept2026\\ProjA' that resolves fine on
        ## Windows must not read "unknown project" on the Linux box that
        ## actually runs the batch.
        project = normalize_key(row.project) if row.project else ""
        project_dir = os.path.join(root, project) if project else root
        if project and not _inside(root, project_dir):
            ## A sheet is a hand-edited spreadsheet, and os.path.join honours
            ## both '../' and an absolute path: without this a stray cell
            ## writes remove_chambers.csv anywhere on disk that happens to
            ## hold a flic_config.yaml.
            results.append(RowResult(
                row, "unknown project",
                f"{row.project!r} is outside {os.path.basename(root)}"))
            continue
        if project and not os.path.isdir(project_dir):
            results.append(RowResult(
                row, "unknown project",
                f"no directory {row.project!r} under {root}"))
            continue

        member_dir = os.path.join(project_dir, normalize_key(row.member))
        if not _inside(root, member_dir):
            results.append(RowResult(
                row, "unknown member",
                f"{row.member!r} is outside {os.path.basename(root)}"))
            continue
        if not prj.is_experiment_dir(member_dir):
            results.append(RowResult(
                row, "unknown member",
                f"no {prj.CONFIG_FILENAME} in "
                f"{os.path.relpath(member_dir, root)}"))
            continue

        declared_dfms = known_dfms.get(member_dir)
        if declared_dfms is None:
            declared_dfms = _config_dfms(member_dir)
            known_dfms[member_dir] = declared_dfms
        if declared_dfms and dfm not in declared_dfms:
            results.append(RowResult(
                row, "unknown chamber",
                f"DFM {dfm} is not configured in {row.member}"))
            continue

        ## Keyed by the resolved path: two spellings of one directory
        ## ('P1/Rep1' and './P1/Rep1') would otherwise accumulate two
        ## independent declaration sets, and the second write would throw away
        ## the first's chambers while both rows reported "applied".
        key = _canonical(member_dir)
        declared = pending.get(key)
        if declared is None:
            declared = {"dir": member_dir,
                        "groups": _read_groups(
                            read_exclusion_notes(member_dir))}
            pending[key] = declared
        group_map = declared["groups"].setdefault(row.group, {})
        existing = group_map.get((dfm, chamber))
        if existing is None:
            group_map[(dfm, chamber)] = row.reason
            results.append(RowResult(row, "applied"))
        elif existing == row.reason:
            results.append(RowResult(row, "already declared"))
        else:
            results.append(RowResult(
                row, "conflict",
                f"kept {existing!r}, sheet says {row.reason!r} "
                "(edit the Member's remove_chambers.csv to change it)"))

    return results, pending


def _read_groups(groups) -> dict:
    """Normalize :func:`read_exclusion_notes` output for comparison.

    Notes matter here, not just chambers: a Batch Run re-applies the sheet
    every time, so a reader that could not see the note it had just written
    would report a conflict on every subsequent run — forever, on rows that
    are already exactly right.  A declaration with a blank note reads as
    :data:`DEFAULT_REASON`, which is what a sheet row with no reason writes.
    """
    out: dict[str, dict] = {}
    for group, entries in (groups or {}).items():
        target = out.setdefault(str(group), {})
        for (dfm_id, chamber), note in (entries or {}).items():
            target[(int(dfm_id), int(chamber))] = note or DEFAULT_REASON
    return out


def _config_dfms(member_dir) -> set:
    """DFM ids declared in a Member's ``flic_config.yaml``, or an empty set
    when the file cannot be read (then nothing is rejected — the loader warns
    about an unmatched declaration instead)."""
    import yaml

    from . import project as prj

    path = os.path.join(str(member_dir), prj.CONFIG_FILENAME)
    try:
        with open(path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    except Exception:  # noqa: BLE001
        return set()
    found: set[int] = set()
    for entry in (config.get("dfms") or []):
        if isinstance(entry, dict):
            value = _int(entry.get("id", entry.get("dfm_id")))
            if value is not None:
                found.add(value)
    return found


def apply_sheet(root, rows=None, *, sheet_path=None, log=None) -> dict:
    """Write an Exclusion Sheet's rows into each Member's ``remove_chambers.csv``.

    *root* is a Batch root (rows name a project) or a Project root (they do
    not).  Merging is additive and the standing declaration wins.  Returns
    ``{"results": [RowResult], "written": [paths], "counts": {...}}``.  Nothing
    here raises for a row that matches nothing: it is reported.
    """
    from .exclusions import read_exclusion_notes, write_exclusions

    if rows is None:
        sheet_path = sheet_path or find_sheet(root)
        if sheet_path is None:
            return {"results": [], "written": [], "counts": {}, "sheet": None}
        rows = read_sheet(sheet_path)

    results, pending = _evaluate_sheet(root, rows)

    written: list[str] = []
    failed: list[str] = []
    for declared in pending.values():
        member_dir = declared["dir"]
        before = _read_groups(read_exclusion_notes(member_dir))
        if declared["groups"] == before:
            continue
        try:
            for group, entries in declared["groups"].items():
                if entries == before.get(group, {}):
                    continue
                by_dfm: dict[int, list[int]] = {}
                notes: dict[tuple[int, int], str] = {}
                for (dfm, chamber), reason in sorted(entries.items()):
                    by_dfm.setdefault(dfm, []).append(chamber)
                    notes[(dfm, chamber)] = reason
                path = write_exclusions(member_dir, group, by_dfm, notes)
                written.append(str(path))
        except OSError as err:
            ## One read-only Member in a batch of forty must not discard the
            ## other thirty-nine's declarations — and reporting "nothing
            ## written" when files HAVE been rewritten is the one thing an
            ## audit trail must never do.  Name the Project too: "Rep1" alone
            ## is ambiguous across a batch where every Project has one.
            parent = os.path.basename(os.path.dirname(member_dir))
            failed.append(
                f"{os.path.join(parent, os.path.basename(member_dir))}: {err}")
            continue

    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1

    if log is not None:
        _log_results(results, counts, written, sheet_path, log)
        for note in failed:
            log(f"[exclusions] could not write {note}")
    return {"results": results, "written": written, "counts": counts,
            "sheet": sheet_path, "failed": failed}


def _log_results(results, counts, written, sheet_path, log) -> None:
    """Report an application: the summary first (what a batch reader sees),
    then every row that did not simply apply."""
    name = os.path.basename(str(sheet_path)) if sheet_path else "exclusion sheet"
    log(f"[exclusions] {name}: {len(results)} row(s), "
        + ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
        + f" — {len(written)} file(s) written")
    for result in results:
        if result.status != "applied":
            log(f"[exclusions] {result.describe()}")
