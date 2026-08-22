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
from pathlib import Path

_FILENAME = "remove_chambers.csv"
_FIELDNAMES = ["group", "dfm_id", "chamber", "note"]


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
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
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

    # Preserve rows that belong to other groups
    existing_rows: list[dict] = []
    if path.exists():
        try:
            with path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if str(row.get("group", "") or "").strip() != group:
                        existing_rows.append(dict(row))
        except Exception:  # noqa: BLE001
            existing_rows = []

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
