"""Editing Slices (ADR-0020) where a design is edited: the Project Design
dialog for a Project, the Config Editor for a standalone experiment.

Headless — the widgets are driven directly.  What is asserted is the
contract: a design with Slices loads into the table and saves back in the
same shape (a mapping keyed by name, declared order kept, ``end: end`` for an
open end, empty optional keys left out); a Slice that cannot be used stops
the save and is named; a type with the Ratio analysis has no Slices.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

pytest.importorskip("PyQt6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QAbstractItemView, QApplication, QMessageBox  # noqa: E402

from pyflic.base import project as project_mod  # noqa: E402
from pyflic.base import slices  # noqa: E402
from pyflic.base.config_editor import (  # noqa: E402
    _SLICE_COLUMNS,
    _SLICE_CONSTANT_COLUMNS,
    FLICConfigEditor,
    SlicesWidget,
)
from pyflic.base.design_editor import ProjectDesignDialog  # noqa: E402

SLICES = {
    "early": {"start": 0, "end": 360, "exclusion_group": "early",
              "also_exclude": ["dead"],
              "constants": {"min_untransformed_licks_cutoff": 20}},
    "late": {"start": 360, "end": "end"},
}
DESIGN = {
    "experiment_type": "Hedonic",
    "well_names": {"A": "S5", "B": "S5Y5"},
    "exclusion_group": "general",
    "also_exclude": ["dead"],
    "slice_tolerance_min": 10,
    "slices": SLICES,
}
#: Column index of each constant a Slice may override.
CONSTANT_COLUMN = {key: len(_SLICE_COLUMNS) + i
                   for i, (key, *_rest) in enumerate(_SLICE_CONSTANT_COLUMNS)}


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _project(root: Path, design: dict) -> Path:
    _write(root / "project.yaml", {"name": root.name, "design": {"global": design},
                                   "scripts": [{"name": "batch", "steps": []}]})
    return root


def _cells(widget: SlicesWidget, row: int) -> list[str]:
    table = widget._table
    return [table.item(row, c).text() for c in range(table.columnCount())]


def _type_row(widget: SlicesWidget, *cells: str) -> int:
    """Add a row the way the user does, and type *cells* into it."""
    widget._add_btn.click()
    row = widget._table.rowCount() - 1
    widget._table.closePersistentEditor(widget._table.item(row, 0))
    for column, text in enumerate(cells):
        widget._table.item(row, column).setText(text)
    return row


def _no_dialogs(monkeypatch) -> list[str]:
    """Record the Config Editor's message boxes instead of showing them."""
    said: list[str] = []
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda _parent, _title, text, *a, **k: said.append(text)))
    return said


def test_a_column_for_every_cutoff_a_slice_may_override():
    """A cutoff added to the core without a column here would be kept as
    written but never editable; this names it."""
    assert {key for key, *_rest in _SLICE_CONSTANT_COLUMNS} == slices.AUTO_REMOVAL_CONSTANTS


# ---------------------------------------------------------------------------
# The Project Design dialog
# ---------------------------------------------------------------------------

def test_the_design_dialog_round_trips_slices(app, tmp_path):
    root = _project(tmp_path / "Proj", DESIGN)
    dialog = ProjectDesignDialog(start_dir=str(root))
    widget = dialog.slices_widget

    assert dialog.also_exclude_edit.text() == "dead"
    assert widget._tolerance_edit.text() == "10"
    assert widget._table.rowCount() == 2
    early, late = _cells(widget, 0), _cells(widget, 1)
    assert early[:5] == ["early", "0", "360", "early", "dead"]
    assert early[CONSTANT_COLUMN["min_untransformed_licks_cutoff"]] == "20"
    ## Open-ended reads "end"; the own group in effect is shown, the Slice's
    ## name when the file states none; a constant the Slice does not state is
    ## blank.
    assert late[:5] == ["late", "360", "end", "late", ""]
    assert late[CONSTANT_COLUMN["min_untransformed_licks_cutoff"]] == ""
    assert widget._add_btn.isEnabled() and widget._reason.isHidden()

    dialog._save()

    assert dialog.saved_dir == str(root)
    text = (root / "project.yaml").read_text(encoding="utf-8")
    g = yaml.safe_load(text)["design"]["global"]
    ## The own group is written, the default stated rather than implied.
    assert g["slices"] == {**SLICES, "late": {**SLICES["late"], "exclusion_group": "late"}}
    assert list(g["slices"]) == ["early", "late"]
    assert "end: end" in text
    assert g["also_exclude"] == ["dead"]
    assert g["slice_tolerance_min"] == 10
    assert g["exclusion_group"] == "general"
    ## And the Project reads them back as the Slices it runs.
    assert [s.name for s in project_mod.Project(root).slices] == ["early", "late"]


def test_a_slice_typed_into_the_dialog_is_written_in_the_yaml_shape(app, tmp_path):
    root = tmp_path / "fresh"
    root.mkdir()
    dialog = ProjectDesignDialog(start_dir=str(root))
    dialog.well_a_edit.setText("S5")
    dialog.well_b_edit.setText("S5Y5")
    widget = dialog.slices_widget
    row = _type_row(widget, "first", "", "120", "", "dead, sick")
    widget._table.item(row, CONSTANT_COLUMN["min_untransformed_licks_cutoff"]).setText("off")
    _type_row(widget, "rest", "120", "")

    dialog._save()

    g = yaml.safe_load((root / "project.yaml").read_text(encoding="utf-8"))["design"]["global"]
    assert g["slices"] == {
        "first": {"start": 0, "end": 120, "exclusion_group": "first",
                  "also_exclude": ["dead", "sick"],
                  "constants": {"min_untransformed_licks_cutoff": None}},
        "rest": {"start": 120, "end": "end", "exclusion_group": "rest"},
    }
    ## Nothing stated, nothing written.
    assert "also_exclude" not in g and "slice_tolerance_min" not in g


def test_the_design_dialog_refuses_a_slice_that_cannot_be_used(app, tmp_path, monkeypatch):
    root = _project(tmp_path / "Proj", DESIGN)
    before = (root / "project.yaml").read_text(encoding="utf-8")
    dialog = ProjectDesignDialog(start_dir=str(root))
    warned: list[str] = []
    monkeypatch.setattr(dialog, "_warn", warned.append)
    widget = dialog.slices_widget
    widget._table.item(0, 2).setText("0")                      # end before start
    widget._table.item(1, 3).setText("general")                # the whole recording's group
    widget._table.item(1, CONSTANT_COLUMN["min_untransformed_licks_cutoff"]).setText("lots")

    dialog._save()

    assert dialog.saved_dir is None
    assert (root / "project.yaml").read_text(encoding="utf-8") == before
    assert len(warned) == 1
    assert "slices.early: end (0) must be after start (0)" in warned[0]
    assert "exclusion_group 'general' is already the own group of the whole recording" \
        in warned[0]
    assert "slices.late: Min licks is a number, or off, not 'lots'" in warned[0]


def test_two_slices_of_one_name_and_a_nameless_row_are_refused(app, tmp_path, monkeypatch):
    root = _project(tmp_path / "Proj", DESIGN)
    dialog = ProjectDesignDialog(start_dir=str(root))
    warned: list[str] = []
    monkeypatch.setattr(dialog, "_warn", warned.append)
    _type_row(dialog.slices_widget, "early", "400", "500")
    _type_row(dialog.slices_widget, "", "500")

    assert dialog._build_design() is None
    assert "slices.early: two Slices are named 'early'" in warned[0]
    assert "slices: row 4 has no name" in warned[0]


def test_a_progressive_ratio_design_has_no_slices(app, tmp_path, monkeypatch):
    root = tmp_path / "pr"
    root.mkdir()
    dialog = ProjectDesignDialog(start_dir=str(root))
    widget = dialog.slices_widget
    dialog.type_combo.setCurrentIndex(dialog.type_combo.findData("ProgressiveRatio"))

    assert not widget._reason.isHidden()
    assert "has no Slices" in widget._reason.text()
    assert not widget._add_btn.isEnabled()
    ## Nothing to show and nothing to add: the empty table goes.
    assert widget._table.isHidden()

    dialog.type_combo.setCurrentIndex(dialog.type_combo.findData("Hedonic"))
    assert widget._reason.isHidden()
    assert widget._add_btn.isEnabled() and not widget._table.isHidden()


def test_a_progressive_ratio_design_with_slices_saves_once_they_are_removed(
        app, tmp_path, monkeypatch):
    root = _project(tmp_path / "pr", {**DESIGN, "experiment_type": "ProgressiveRatio",
                                      "well_names": {"A": "Sucrose", "B": "Yeast"}})
    dialog = ProjectDesignDialog(start_dir=str(root))
    warned: list[str] = []
    monkeypatch.setattr(dialog, "_warn", warned.append)
    widget = dialog.slices_widget

    ## Shown, greyed and uneditable, with the way out still live.
    assert widget._table.rowCount() == 2
    assert not widget._table.isHidden() and not widget._table.isEnabled()
    assert widget._table.editTriggers() == QAbstractItemView.EditTrigger.NoEditTriggers
    assert "Remove these Slices to save" in widget._reason.text()
    assert widget._remove_btn.isEnabled() and not widget._add_btn.isEnabled()

    dialog._save()
    assert dialog.saved_dir is None
    assert "has no Slices" in warned[0]

    widget._remove_btn.click()
    widget._remove_btn.click()
    assert widget._table.rowCount() == 0 and not widget._remove_btn.isEnabled()
    dialog._save()
    assert dialog.saved_dir == str(root)
    g = yaml.safe_load((root / "project.yaml").read_text(encoding="utf-8"))["design"]["global"]
    assert "slices" not in g
    ## The whole recording's own exclusions are not the Slices'.
    assert g["also_exclude"] == ["dead"]


def test_a_slice_key_with_no_column_is_kept_and_named(app, tmp_path, monkeypatch):
    """A typo in a hand-written Slice is not silently dropped — that would
    turn ``ends: 300`` into a Slice running to the end of the recording."""
    root = _project(tmp_path / "Proj", {**DESIGN, "slices": {
        "early": {"start": 0, "ends": 300, "constants": {"max_events_cutoff": 5}}}})
    dialog = ProjectDesignDialog(start_dir=str(root))
    warned: list[str] = []
    monkeypatch.setattr(dialog, "_warn", warned.append)

    declared, problems = dialog.slices_widget.get_values()
    assert declared["slices"]["early"]["ends"] == 300
    assert declared["slices"]["early"]["constants"] == {"max_events_cutoff": 5}
    assert dialog._build_design() is None
    assert "unknown key(s) ['ends']" in warned[0]


def test_an_infinite_end_reads_as_the_end_of_the_recording(app, tmp_path):
    """``end: .inf`` is an open end to the loader; it must not stop the
    dialog opening, and it saves in the one spelling, ``end: end``."""
    root = _project(tmp_path / "Proj", {**DESIGN, "slices": {
        "late": {"start": 360, "end": float("inf")}}})
    dialog = ProjectDesignDialog(start_dir=str(root))

    assert _cells(dialog.slices_widget, 0)[:3] == ["late", "360", "end"]
    assert dialog._build_design()["global"]["slices"] == {
        "late": {"start": 360, "end": "end", "exclusion_group": "late"}}


def test_the_opto_columns_follow_the_design_optogenetics(app, tmp_path):
    dialog = ProjectDesignDialog(start_dir=str(tmp_path / "new"))
    table = dialog.slices_widget._table
    opto = CONSTANT_COLUMN["opto_unexplained_min_sec"]
    licks = CONSTANT_COLUMN["min_untransformed_licks_cutoff"]

    assert not table.isColumnHidden(opto)
    dialog.opto_combo.setCurrentIndex(dialog.opto_combo.findData(False))
    assert table.isColumnHidden(opto) and not table.isColumnHidden(licks)


# ---------------------------------------------------------------------------
# The Config Editor: a standalone experiment's design, and a Member's
# ---------------------------------------------------------------------------

def _standalone(tmp_path: Path, global_cfg: dict) -> Path:
    path = tmp_path / "solo" / "flic_config.yaml"
    _write(path, {"global": global_cfg, "dfms": [{"id": 1, "chambers": {1: "Ctrl"}}]})
    return path


def test_the_config_editor_round_trips_slices(app, tmp_path, monkeypatch):
    _no_dialogs(monkeypatch)
    path = _standalone(tmp_path, DESIGN)
    editor = FLICConfigEditor(path)
    try:
        assert editor._exclusion_group_edit.text() == "general"
        assert editor._also_exclude_edit.text() == "dead"
        assert editor._slices_widget._table.rowCount() == 2
        assert "⚠" not in editor._tabs.tabText(0)

        editor._write_yaml(path)

        text = path.read_text(encoding="utf-8")
        g = yaml.safe_load(text)["global"]
        assert g["slices"] == {**SLICES, "late": {**SLICES["late"], "exclusion_group": "late"}}
        assert list(g["slices"]) == ["early", "late"]
        assert "end: end" in text
        assert (g["exclusion_group"], g["also_exclude"], g["slice_tolerance_min"]) == \
            ("general", ["dead"], 10)
    finally:
        editor.close()


def test_the_config_editor_refuses_a_slice_that_cannot_be_used(app, tmp_path, monkeypatch):
    said = _no_dialogs(monkeypatch)
    path = _standalone(tmp_path, DESIGN)
    before = path.read_text(encoding="utf-8")
    editor = FLICConfigEditor(path)
    try:
        editor._slices_widget._table.item(0, 0).setText("all")    # a reserved name

        ## Counted on the tab it lives on before anyone saves...
        assert "⚠" in editor._tabs.tabText(0)
        assert "'all' is reserved" in editor._tabs.tabToolTip(0)
        editor._write_yaml(path)

        ## ...and the save is refused, not merely confirmed.
        assert path.read_text(encoding="utf-8") == before
        assert len(said) == 1 and "slices.all: 'all' is reserved" in said[0]
    finally:
        editor.close()


def test_the_config_editor_has_no_slices_for_progressive_ratio(app, tmp_path):
    path = _standalone(tmp_path, {"experiment_type": "ProgressiveRatio",
                                  "well_names": {"A": "Sucrose", "B": "Yeast"}})
    editor = FLICConfigEditor(path)
    try:
        widget = editor._slices_widget
        assert "has no Slices" in widget._reason.text() and not widget._reason.isHidden()
        assert not widget._add_btn.isEnabled() and widget._table.isHidden()
        ## The whole recording's exclusion groups stay editable.
        assert editor._also_exclude_edit.isEnabled()

        editor._experiment_type_combo.setCurrentIndex(
            editor._experiment_type_combo.findData("Hedonic"))
        assert widget._reason.isHidden() and widget._add_btn.isEnabled()
    finally:
        editor.close()


def test_a_members_slices_are_the_projects_and_read_only(app, tmp_path, monkeypatch):
    _no_dialogs(monkeypatch)
    root = _project(tmp_path / "Proj", DESIGN)
    member = root / "rep1" / "flic_config.yaml"
    _write(member, {"dfms": [{"id": 1, "chambers": {1: "Ctrl"}}]})
    editor = FLICConfigEditor(member)
    try:
        widget = editor._slices_widget
        assert widget._table.rowCount() == 2
        assert _cells(widget, 1)[:3] == ["late", "360", "end"]
        assert widget._table.editTriggers() == QAbstractItemView.EditTrigger.NoEditTriggers
        assert not widget._add_btn.isEnabled() and not widget._remove_btn.isEnabled()
        assert "Project design" in widget._table.toolTip()
        assert not editor._also_exclude_edit.isEnabled()

        editor._write_yaml(member)
        ## Only the Design Copy is written: the Slices stay the design's (ADR-0025).
        assert yaml.safe_load(member.read_text(encoding="utf-8"))["global"] == \
            {"experiment_type": "Hedonic", "well_names": {"A": "S5", "B": "S5Y5"}}
    finally:
        editor.close()


# ---------------------------------------------------------------------------
# Renaming a Slice, and the constants a Slice may override
# ---------------------------------------------------------------------------

def test_renaming_a_slice_renames_its_own_group_and_its_unticked_box(app, tmp_path,
                                                                     monkeypatch):
    """An own group that was the Slice's name follows a rename, and so does
    the Slice's entry in omit.slices; an own group named otherwise stays."""
    _no_dialogs(monkeypatch)
    path = _standalone(tmp_path, {**DESIGN, "omit": {"slices": ["late"]}, "slices": {
        **SLICES, "mid": {"start": 100, "end": 200, "exclusion_group": "middle"}}})
    editor = FLICConfigEditor(path)
    try:
        table = editor._slices_widget._table
        table.item(1, 0).setText("tail")
        table.item(2, 0).setText("centre")
        assert table.item(1, 3).text() == "tail"
        assert table.item(2, 3).text() == "middle"
        assert editor._slices_widget.renamed() == {"late": "tail", "mid": "centre"}

        editor._write_yaml(path)

        g = yaml.safe_load(path.read_text(encoding="utf-8"))["global"]
        assert list(g["slices"]) == ["early", "tail", "centre"]
        assert g["slices"]["tail"]["exclusion_group"] == "tail"
        assert g["slices"]["centre"]["exclusion_group"] == "middle"
        assert g["omit"] == {"slices": ["tail"]}
    finally:
        editor.close()


def test_a_new_slices_own_group_follows_its_name_until_changed(app):
    widget = SlicesWidget()
    row = _type_row(widget, "first")
    assert widget._table.item(row, 3).text() == "first"
    widget._table.item(row, 3).setText("mine")
    widget._table.item(row, 0).setText("second")
    assert widget._table.item(row, 3).text() == "mine"
    ## Added here, so nothing loaded was renamed.
    assert widget.renamed() == {}


def test_the_design_dialog_follows_a_rename_into_the_selection(app, tmp_path):
    root = _project(tmp_path / "Proj", {**DESIGN, "omit": {"slices": ["early"]}})
    dialog = ProjectDesignDialog(start_dir=str(root))
    dialog.slices_widget._table.item(0, 0).setText("first")

    dialog._save()

    assert dialog.saved_dir == str(root)
    g = yaml.safe_load((root / "project.yaml").read_text(encoding="utf-8"))["design"]["global"]
    assert g["slices"]["first"]["exclusion_group"] == "first"
    assert g["omit"] == {"slices": ["first"]}


def test_a_slice_may_override_the_unexplained_light_failure_fraction(app, tmp_path):
    """The fraction that decides the FAIL verdict exclude_failed_opto_chambers
    acts on, beside the seconds before it is judged."""
    root = _project(tmp_path / "Proj", {**DESIGN, "slices": {
        "early": {"start": 0, "end": 360,
                  "constants": {"opto_unexplained_fail_fraction": 0.5}}}})
    dialog = ProjectDesignDialog(start_dir=str(root))
    widget = dialog.slices_widget
    column = CONSTANT_COLUMN["opto_unexplained_fail_fraction"]
    assert _cells(widget, 0)[column] == "0.5"

    widget._table.item(0, column).setText("1.5")
    _declared, problems = widget.get_values()
    assert any("opto_unexplained_fail_fraction" in p for p in problems)

    widget._table.item(0, column).setText("0.4")
    assert dialog._build_design()["global"]["slices"]["early"]["constants"] == {
        "opto_unexplained_fail_fraction": 0.4}
    ## An optogenetic setting, shown where the design's are.
    dialog.opto_combo.setCurrentIndex(dialog.opto_combo.findData(False))
    assert widget._table.isColumnHidden(column)
