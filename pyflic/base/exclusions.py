"""
Chamber exclusion groups for pyflic experiments.

Exclusions are stored in ``remove_chambers.csv`` in the project directory.
Each row names a *group*, a DFM id, a chamber index, and an optional note.
Different groups let different scripts (or manual hub runs) apply different
exclusion sets from the same ``flic_config.yaml``.

CSV format::

    group,dfm_id,chamber,note
    general,1,3,low lick count
    general,2,5,
    Standard Analysis,1,4,noisy signal
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

_FILENAME = "remove_chambers.csv"
_FIELDNAMES = ["group", "dfm_id", "chamber", "note"]


def read_csv_text(path: str | Path) -> str:
    """The text of a CSV a person may have saved from Excel.

    Excel's "CSV UTF-8" starts the file with a byte-order mark, which as
    plain UTF-8 makes the first header ``\ufeffgroup``, so no row had a group;
    its "CSV (Comma delimited)" writes Windows-1252, where one ``°`` or ``é``
    is not UTF-8 and the whole file failed to decode.  Either way every
    declaration in it vanished without a word.  UTF-8 (with or without the
    mark) is tried first, then Windows-1252, then Latin-1, which reads any
    bytes.  ``OSError`` is the caller's.
    """
    data = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def read_exclusions(experiment_dir: str | Path) -> dict[str, dict[int, list[int]]]:
    """Read ``remove_chambers.csv`` and return ``{group: {dfm_id: [chamber, ...]}}``.

    Returns an empty dict if the file does not exist or cannot be parsed.
    Chamber lists within each group/DFM are sorted ascending.
    """
    path = Path(experiment_dir) / _FILENAME
    if not path.exists():
        return {}
    result: dict[str, dict[int, list[int]]] = {}
    try:
        reader = csv.DictReader(io.StringIO(read_csv_text(path), newline=""))
        for row in reader:
            group = str(row.get("group", "") or "").strip()
            try:
                dfm_id = int(row["dfm_id"])
                chamber = int(row["chamber"])
            except (KeyError, ValueError, TypeError):
                continue
            if not group:
                continue
            group_dict = result.setdefault(group, {})
            chambers = group_dict.setdefault(dfm_id, [])
            if chamber not in chambers:
                chambers.append(chamber)
    except Exception:  # noqa: BLE001
        return {}
    for group_dict in result.values():
        for dfm_id in group_dict:
            group_dict[dfm_id] = sorted(group_dict[dfm_id])
    return result


def read_exclusion_notes(
    experiment_dir: str | Path,
) -> dict[str, dict[tuple[int, int], str]]:
    """Read ``remove_chambers.csv`` as ``{group: {(dfm, chamber): note}}``.

    :func:`read_exclusions` answers "which chambers" and drops the note.  The
    Exclusion Sheet needs the note as well: without it a standing declaration
    reads back as a blank reason, and re-applying the very sheet that wrote it
    reports a conflict — on every Batch Run, forever.
    """
    path = Path(experiment_dir) / _FILENAME
    if not path.exists():
        return {}
    result: dict[str, dict[tuple[int, int], str]] = {}
    try:
        for row in csv.DictReader(io.StringIO(read_csv_text(path), newline="")):
            group = str(row.get("group", "") or "").strip()
            try:
                key = (int(row["dfm_id"]), int(row["chamber"]))
            except (KeyError, ValueError, TypeError):
                continue
            if not group:
                continue
            result.setdefault(group, {})[key] = str(
                row.get("note", "") or "").strip()
    except Exception:  # noqa: BLE001
        return {}
    return result


def write_exclusions(
    experiment_dir: str | Path,
    group: str,
    exclusions_by_dfm: dict[int, list[int]],
    notes: dict[tuple[int, int], str] | None = None,
) -> Path:
    """Write/update one named group in ``remove_chambers.csv``.

    All rows for *group* are replaced with the entries in *exclusions_by_dfm*.
    Rows for all other groups are preserved unchanged.  If *exclusions_by_dfm*
    is empty, all rows for *group* are removed.

    Parameters
    ----------
    experiment_dir:
        Project root directory (``remove_chambers.csv`` lives here).
    group:
        Name of the exclusion group to update (e.g. ``"general"``).
    exclusions_by_dfm:
        ``{dfm_id: [chamber, ...]}`` — the complete desired exclusion set for
        this group.  Pass an empty dict to clear the group entirely.
    notes:
        Optional ``{(dfm_id, chamber): note_text}`` for per-entry notes.

    Returns
    -------
    Path
        Absolute path to the written ``remove_chambers.csv``.
    """
    path = Path(experiment_dir) / _FILENAME
    notes = notes or {}

    # Preserve rows that belong to other groups.  A file that cannot be read
    # raises rather than being replaced by this one group: the rows it holds
    # are other analyses' declarations, and the audit trail besides.
    existing_rows: list[dict] = []
    if path.exists():
        reader = csv.DictReader(io.StringIO(read_csv_text(path), newline=""))
        for row in reader:
            if str(row.get("group", "") or "").strip() != group:
                existing_rows.append(dict(row))

    # Build new rows for the updated group
    new_rows: list[dict] = []
    for dfm_id in sorted(exclusions_by_dfm):
        for chamber in sorted(exclusions_by_dfm[dfm_id]):
            new_rows.append({
                "group": group,
                "dfm_id": dfm_id,
                "chamber": chamber,
                "note": notes.get((dfm_id, chamber), ""),
            })

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_rows + new_rows)

    return path
