"""Blocked Members, filing, and the Exclusion Sheet (ADR-0009, ADR-0010)."""

from __future__ import annotations

import ntpath
from pathlib import Path

import pytest
import yaml

from pyflic.base import batch as batch_mod
from pyflic.base import exclusion_sheet as sheet_mod
from pyflic.base import layout
from pyflic.base.exclusions import read_exclusions

CONFIG = {"dfms": [{"id": 1, "chambers": {1: "Ctrl", 2: "Exp"}},
                   {"id": 2, "chambers": {1: "Ctrl", 2: "Exp"}}]}


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(payload, sort_keys=False),
                        encoding="utf-8")


def _member(root: Path, *, dfms=(1, 2), configured=True, filed=True) -> Path:
    data = root / "data" if filed else root
    for dfm in dfms:
        _write(data / f"DFM{dfm}_0.csv", "Sample,Seconds\n")
    if configured:
        _write(root / "flic_config.yaml", CONFIG)
    return root


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------

def test_a_filed_configured_member_is_usable(tmp_path: Path):
    item = layout.classify(_member(tmp_path / "rep1"))
    assert item.status == layout.OK and item.usable and not item.blocked
    assert item.dfm_ids == (1, 2)


def test_data_at_the_root_is_an_unfiled_recording(tmp_path: Path):
    """The loader reads data/ and nothing else, so a recording loose at the
    root is invisible rather than merely untidy."""
    item = layout.classify(_member(tmp_path / "rep1", filed=False))
    assert item.status == layout.UNFILED and item.fix == "file"


def test_data_without_a_config_offers_a_config(tmp_path: Path):
    item = layout.classify(_member(tmp_path / "rep1", configured=False))
    assert item.status == layout.NO_CONFIG and item.fix == "config"


def test_a_config_with_no_recording_cannot_be_repaired(tmp_path: Path):
    root = tmp_path / "rep1"
    _write(root / "flic_config.yaml", CONFIG)
    item = layout.classify(root)
    assert item.status == layout.NO_RECORDING and item.fix is None


def test_the_same_dfm_loose_and_filed_refuses_rather_than_guessing(tmp_path: Path):
    """Many DFM files is normal here, so the question is never "which file" —
    it is "which COPY", and filing on top would merge two recordings."""
    root = _member(tmp_path / "rep1")
    _write(root / "DFM1_0.csv", "Sample,Seconds\n")
    item = layout.classify(root)
    assert item.status == layout.AMBIGUOUS
    assert not layout.plan_filing(root).possible


def test_a_case_wrong_csv_is_reported_rather_than_silently_missing(tmp_path: Path):
    root = tmp_path / "rep1"
    _write(root / "flic_config.yaml", CONFIG)
    _write(root / "data" / "dfm1_0.csv", "Sample,Seconds\n")
    item = layout.classify(root)
    assert item.status == layout.NO_RECORDING
    assert "dfm1_0.csv" in item.detail


def test_output_directories_are_not_members(tmp_path: Path):
    _write(tmp_path / "project.yaml", {"name": "p"})
    _member(tmp_path / "rep1")
    for name in ("analysis", "qc", "figures", "data"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "DFM1_0.csv").write_text("Sample,Seconds\n")
    assert [i.name for i in layout.members_in(tmp_path)] == ["rep1"]


# ---------------------------------------------------------------------------
# filing
# ---------------------------------------------------------------------------

def test_filing_moves_the_recording_and_parks_everything_else(tmp_path: Path):
    root = _member(tmp_path / "rep1", filed=False, configured=True)
    _write(root / "notes.txt", "hello")
    plan = layout.file_recording(root)
    assert plan.possible or plan.moves
    assert (root / "data" / "DFM1_0.csv").is_file()
    assert (root / "extra_files" / "notes.txt").is_file()
    assert layout.classify(root).status == layout.OK


def test_filing_moves_the_program_and_the_message_log_with_the_recording(tmp_path: Path):
    """The loader reads Program.txt and Messages.csv from data/ only, so
    parking them in extra_files/ silently lost the program."""
    from pyflic.base.messages import find_messages
    from pyflic.base.opto_program import find_program

    root = _member(tmp_path / "rep1", filed=False, configured=True)
    _write(root / "PROGRAM.TXT", "Start Time: 1/1/2026 9:00:00 AM\n")
    _write(root / "Messages.csv", "DFM,Date,Time\n")
    layout.file_recording(root)
    assert find_program(root / "data") == root / "data" / "PROGRAM.TXT"
    assert find_messages(root / "data") == root / "data" / "Messages.csv"
    assert not (root / "extra_files").exists()


def test_filing_never_moves_a_yaml_or_the_exclusion_sheet(tmp_path: Path):
    """Moving flic_config.yaml un-makes the Experiment Directory, and moving
    remove_chambers.csv silently returns excluded chambers to the analysis —
    the ADR-0010 failure the exemption exists to prevent."""
    root = _member(tmp_path / "rep1", filed=False, configured=True)
    _write(root / "remove_chambers.csv", "group,dfm_id,chamber,note\n")
    _write(root / "sidecar.yml", {"x": 1})
    layout.file_recording(root)
    assert (root / "flic_config.yaml").is_file()
    assert (root / "remove_chambers.csv").is_file()
    assert (root / "sidecar.yml").is_file()


def test_filing_never_overwrites(tmp_path: Path):
    root = _member(tmp_path / "rep1", dfms=(1,), filed=False, configured=True)
    _write(root / "data" / "DFM9_0.csv", "Sample,Seconds\n")
    _write(root / "notes.txt", "new")
    _write(root / "extra_files" / "notes.txt", "old")
    layout.file_recording(root)
    assert (root / "extra_files" / "notes.txt").read_text() == "old"
    assert (root / "notes.txt").is_file()


# ---------------------------------------------------------------------------
# the Exclusion Sheet
# ---------------------------------------------------------------------------

def _batch_with_sheet(tmp_path: Path, rows: str) -> Path:
    root = tmp_path / "batch"
    for key in ("ProjA", "Sept/ProjB"):
        _write(root / key / "project.yaml", {"name": Path(key).name})
        _member(root / key / "rep1")
    _write(root / "remove_chambers.csv",
           "project,member,dfm,chamber,group,reason\n" + rows)
    return root


def test_the_sheet_writes_into_each_members_own_file(tmp_path: Path):
    root = _batch_with_sheet(tmp_path, "ProjA,rep1,1,2,general,noisy\n")
    result = batch_mod.apply_exclusion_sheet(root, log=lambda _m: None)
    assert result["counts"] == {"applied": 1}
    assert read_exclusions(root / "ProjA" / "rep1") == {"general": {1: [2]}}


def test_re_applying_the_same_sheet_is_idempotent(tmp_path: Path):
    """A Batch Run applies the sheet every time; a reader that could not see
    the note it had just written would report a conflict forever."""
    root = _batch_with_sheet(tmp_path, "ProjA,rep1,1,2,general,noisy\n")
    batch_mod.apply_exclusion_sheet(root, log=lambda _m: None)
    again = batch_mod.apply_exclusion_sheet(root, log=lambda _m: None)
    assert again["counts"] == {"already declared": 1}


def test_a_standing_declaration_wins_and_the_difference_is_reported(tmp_path: Path):
    root = _batch_with_sheet(tmp_path, "ProjA,rep1,1,2,general,noisy\n")
    batch_mod.apply_exclusion_sheet(root, log=lambda _m: None)
    _write(root / "remove_chambers.csv",
           "project,member,dfm,chamber,group,reason\n"
           "ProjA,rep1,1,2,general,something else\n")
    result = batch_mod.apply_exclusion_sheet(root, log=lambda _m: None)
    assert result["counts"] == {"conflict": 1}
    notes = (root / "ProjA" / "rep1" / "remove_chambers.csv").read_text()
    assert "noisy" in notes and "something else" not in notes


def test_a_blank_group_lands_in_the_designs_own_and_an_unread_one_is_named(tmp_path: Path):
    """A row naming no group goes where a load naming none reads; a group no
    analysis excludes is still written, and said so."""
    root = _batch_with_sheet(tmp_path, "ProjA,rep1,1,2,,noisy\nProjA,rep1,2,1,mian,typo\n")
    _write(root / "ProjA" / "project.yaml",
           {"name": "ProjA", "design": {"global": {"exclusion_group": "main"}}})
    result = batch_mod.apply_exclusion_sheet(root, log=lambda _m: None)
    assert result["counts"] == {"applied": 1, sheet_mod.UNREAD_GROUP: 1}
    unread = next(r for r in result["results"] if r.status == sheet_mod.UNREAD_GROUP)
    assert unread.wrote and "'mian'" in unread.detail and "'main'" in unread.detail
    assert read_exclusions(root / "ProjA" / "rep1") == {"main": {1: [2]}, "mian": {2: [1]}}


def test_rows_are_scoped_to_the_projects_actually_running(tmp_path: Path):
    """Unchecking a Project means "do not touch this Project", and recursion
    surfaces Projects the user may never have known were there."""
    root = _batch_with_sheet(
        tmp_path,
        "ProjA,rep1,1,2,general,a\nSept/ProjB,rep1,1,2,general,b\n")
    result = batch_mod.apply_exclusion_sheet(
        root, log=lambda _m: None, projects=["ProjA"])
    assert result["skipped"] == 1
    assert read_exclusions(root / "Sept" / "ProjB" / "rep1") == {}


def test_a_split_path_still_scopes_to_the_right_project(tmp_path: Path):
    """``project=Sept, member=ProjB/rep1`` names the same member as
    ``project=Sept/ProjB, member=rep1``; scoping on the project cell alone let
    the second spelling write into a Project that was not running."""
    root = _batch_with_sheet(tmp_path, "Sept,ProjB/rep1,1,2,general,b\n")
    result = batch_mod.apply_exclusion_sheet(
        root, log=lambda _m: None, projects=["ProjA"])
    assert result["skipped"] == 1
    assert read_exclusions(root / "Sept" / "ProjB" / "rep1") == {}


def _row(project: str, member: str) -> sheet_mod.SheetRow:
    return sheet_mod.SheetRow(index=2, project=project, member=member, dfm="1",
                              chamber="2", group="general", reason="r")


def test_scoping_holds_under_windows_case_folding(monkeypatch):
    """Windows' normcase lowers the case and turns '/' into '\\': the scope
    test then never saw 'proja\\rep1' start with 'proja/', and skipped every
    row of a running Project.  Here on every platform."""
    monkeypatch.setattr(batch_mod.os.path, "normcase", ntpath.normcase)
    rows = [_row("ProjA", "rep1"), _row("sept/projb", "rep1"), _row("Sept", "ProjB/rep1"),
            _row("ProjC", "rep1")]
    keep, skip = batch_mod.scope_sheet_rows(rows, ["ProjA", "Sept/ProjB"])
    assert [(r.project, r.member) for r in keep] == [
        ("ProjA", "rep1"), ("sept/projb", "rep1"), ("Sept", "ProjB/rep1")]
    assert [(r.project, r.member) for r in skip] == [("ProjC", "rep1")]


def test_the_preview_writes_nothing(tmp_path: Path):
    """Selecting a Batch reports; it never applies."""
    root = _batch_with_sheet(tmp_path, "ProjA,rep1,1,2,general,noisy\n")
    preview = batch_mod.preview_exclusion_sheet(root)
    assert preview["counts"] == {"applied": 1}
    assert read_exclusions(root / "ProjA" / "rep1") == {}


def test_a_row_naming_an_unconfigured_dfm_is_reported_not_written(tmp_path: Path):
    root = _batch_with_sheet(tmp_path, "ProjA,rep1,9,2,general,noisy\n")
    result = batch_mod.apply_exclusion_sheet(root, log=lambda _m: None)
    assert result["counts"] == {"unknown chamber": 1}
    assert read_exclusions(root / "ProjA" / "rep1") == {}


def test_a_sheet_cell_cannot_escape_its_root(tmp_path: Path):
    root = _batch_with_sheet(tmp_path, "../elsewhere,rep1,1,2,general,x\n")
    result = batch_mod.apply_exclusion_sheet(root, log=lambda _m: None)
    assert set(result["counts"]) <= {"unknown project", "unknown member"}


def test_a_sheet_missing_a_required_column_is_a_stopping_mistake(tmp_path: Path):
    path = tmp_path / "remove_chambers.csv"
    path.write_text("project,reason\nProjA,noisy\n", encoding="utf-8")
    with pytest.raises(ValueError, match="member"):
        sheet_mod.read_sheet(path)


def test_the_sheet_is_found_whatever_case_excel_saved_it_in(tmp_path: Path):
    (tmp_path / "Remove_Chambers.csv").write_text(
        "project,member,dfm,chamber\n", encoding="utf-8")
    assert sheet_mod.find_sheet(tmp_path) is not None


# ---------------------------------------------------------------------------
# Files Excel saved: a byte-order mark, or Windows' own encoding
# ---------------------------------------------------------------------------

#: Excel's "CSV UTF-8" starts the file with a byte-order mark; its plain
#: "CSV (Comma delimited)" writes Windows-1252, where ° and é are one byte
#: each and not UTF-8.
_EXCEL_ENCODINGS = [("utf-8-sig", "CSV UTF-8"), ("cp1252", "CSV (Comma delimited)")]


@pytest.mark.parametrize("encoding", [e for e, _label in _EXCEL_ENCODINGS])
def test_a_members_file_excel_saved_keeps_its_exclusions(tmp_path: Path, encoding):
    from pyflic.base.exclusions import read_exclusion_notes

    (tmp_path / "remove_chambers.csv").write_bytes(
        "group,dfm_id,chamber,note\ngeneral,1,3,dead at 25°C\ndead,2,5,décédée\n"
        .encode(encoding))
    assert read_exclusions(tmp_path) == {"general": {1: [3]}, "dead": {2: [5]}}
    assert read_exclusion_notes(tmp_path) == {
        "general": {(1, 3): "dead at 25°C"}, "dead": {(2, 5): "décédée"}}


@pytest.mark.parametrize("encoding", [e for e, _label in _EXCEL_ENCODINGS])
def test_writing_one_group_keeps_the_others_of_a_file_excel_saved(tmp_path: Path, encoding):
    from pyflic.base.exclusions import read_exclusion_notes, write_exclusions

    (tmp_path / "remove_chambers.csv").write_bytes(
        "group,dfm_id,chamber,note\ngeneral,1,3,dead at 25°C\ndead,2,5,\n"
        .encode(encoding))
    write_exclusions(tmp_path, "early", {1: [6]})
    assert read_exclusions(tmp_path) == {
        "general": {1: [3]}, "dead": {2: [5]}, "early": {1: [6]}}
    assert read_exclusion_notes(tmp_path)["general"] == {(1, 3): "dead at 25°C"}


def test_a_file_that_cannot_be_read_is_never_written_over(tmp_path: Path, monkeypatch):
    """Rewriting one group keeps every other group's rows, so a file whose
    rows cannot be read is refused rather than replaced by the one group."""
    from pyflic.base import exclusions

    path = tmp_path / "remove_chambers.csv"
    path.write_text("group,dfm_id,chamber,note\ngeneral,1,3,x\n", encoding="utf-8")

    def broken(_path):
        raise OSError("locked by another program")

    monkeypatch.setattr(exclusions, "read_csv_text", broken)
    with pytest.raises(OSError, match="locked"):
        exclusions.write_exclusions(tmp_path, "early", {1: [6]})
    assert path.read_text(encoding="utf-8") == "group,dfm_id,chamber,note\ngeneral,1,3,x\n"


@pytest.mark.parametrize("encoding", [e for e, _label in _EXCEL_ENCODINGS])
def test_a_sheet_excel_saved_is_read_and_applied(tmp_path: Path, encoding):
    root = _batch_with_sheet(tmp_path, "")
    (root / "remove_chambers.csv").write_bytes(
        "project,member,dfm,chamber,group,reason\nProjA,rep1,1,2,general,mort à 25°C\n"
        .encode(encoding))
    assert sheet_mod.looks_like_sheet(root / "remove_chambers.csv")
    result = batch_mod.apply_exclusion_sheet(root, log=lambda _m: None)
    assert result["counts"] == {"applied": 1}
    assert read_exclusions(root / "ProjA" / "rep1") == {"general": {1: [2]}}


def test_a_dotdot_in_a_cell_cannot_reach_a_project_that_is_not_running(tmp_path: Path):
    """``ProjA/../Sept/ProjB`` opens Sept/ProjB: the scope test must see
    the Project the write will reach, not the one the cell starts with."""
    root = _batch_with_sheet(tmp_path, "ProjA,../Sept/ProjB/rep1,1,2,general,b\n")
    result = batch_mod.apply_exclusion_sheet(
        root, log=lambda _m: None, projects=["ProjA"])
    assert result["skipped"] == 1
    assert read_exclusions(root / "Sept" / "ProjB" / "rep1") == {}
    assert batch_mod.normalize_key("ProjA/../Sept/./ProjB/") == "Sept/ProjB"
    assert batch_mod.normalize_key("../elsewhere") == "../elsewhere"
