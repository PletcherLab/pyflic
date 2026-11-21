"""The Experiment Editor: a recording's about:, design and chambers.

Headless — the widgets are driven directly.  What is asserted is the
contract: it writes about:, the design keys and dfms:, keeps everything else
the file holds exactly as read, writes a config the loader accepts, and
inside a Project writes the Design Copy (ADR-0024, ADR-0025).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
import yaml

pytest.importorskip("PyQt6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from pyflic.base import project as project_mod  # noqa: E402
from pyflic.base.experiment_editor import ExperimentEditor  # noqa: E402
from pyflic.base.yaml_config import load_experiment_yaml  # noqa: E402
from pyflic.base.yaml_lint import lint_flic_config  # noqa: E402

FULL = {
    "data_folder": "../elsewhere/data",
    "global": {
        "experiment_type": "Hedonic",
        "params": {"feeding_threshold": 22, "chamber_size": 2},
        "transform_licks": False,
        "well_names": {"A": "S5", "B": "S5Y5"},
        "constants": {"min_untransformed_licks_cutoff": 15},
        "slices": {"early": {"start": 0, "end": 60}},
        "experimental_design_factors": {"Genotype": ["Ctrl", "Exp"]},
    },
    "dfms": [{"id": 1, "optogenetics": False, "params": {"pi_direction": "right"},
              "note": "lid cracked",
              "chambers": {1: "Ctrl", 2: "Exp"}}],
    "scripts": [{"name": "mine", "steps": [{"action": "load"}]}],
}


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def no_dialogs(monkeypatch):
    """Every confirmation answers Yes, so nothing blocks headless."""
    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _read(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _dfm_csv(folder: Path, dfm_id: int, n: int = 600) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    cols = {"Sample": list(range(1, n + 1)), "Seconds": [i / 5.0 for i in range(n)]}
    for w in range(1, 13):
        cols[f"W{w}"] = [0] * n
    pd.DataFrame(cols).to_csv(folder / f"DFM{dfm_id}_0.csv", index=False)


def _set_kind(editor: ExperimentEditor, row: int, kind: str) -> None:
    combo = editor.factor_table.table.cellWidget(row, 1)
    combo.setCurrentIndex(combo.findData(kind))


def test_saving_changes_only_about_the_design_and_chambers(app, tmp_path):
    path = tmp_path / "exp" / "flic_config.yaml"
    _write(path, FULL)
    editor = ExperimentEditor(path)
    try:
        assert editor.type_combo.currentData() == "Hedonic"
        assert not editor.layout_combo.isEnabled()           # Hedonic owns it
        assert editor.factor_table.factors() == {"Genotype": ["Ctrl", "Exp"]}
        editor.title_edit.setText("Hedonic, replicate 1")
        editor.experimenter_edit.setText("scott")
        editor.sex_checks["female"].setChecked(True)
        editor.protocol_edits["name"].setText("FLIC hedonic")
        _set_kind(editor, 0, "genotype")
        editor.meaning_table.item(1, 2).setText("R12C11 > GtACR1")
        assert editor.write(path, confirm=False)
    finally:
        editor.close()

    saved = _read(path)
    assert list(saved) == ["data_folder", "about", "global", "dfms", "scripts"]
    assert saved["about"] == {
        "title": "Hedonic, replicate 1", "experimenter": "scott", "sex": ["female"],
        "protocol": {"name": "FLIC hedonic"},
        "factors": {"Genotype": {"kind": "genotype", "levels": {"Exp": "R12C11 > GtACR1"}}},
    }
    expected_global = {k: v for k, v in FULL["global"].items()}
    expected_global["params"] = {"feeding_threshold": 22}   # the layout says the size
    assert saved["global"] == expected_global
    assert saved["dfms"] == FULL["dfms"]                    # hidden keys kept
    assert saved["scripts"] == FULL["scripts"]
    assert saved["data_folder"] == FULL["data_folder"]


def test_reopening_shows_what_was_saved(app, tmp_path):
    path = tmp_path / "exp" / "flic_config.yaml"
    _write(path, {**FULL, "about": {"title": "T", "sex": ["female", "mixed"],
                                    "lab_book": "p. 12",
                                    "factors": {"Genotype": {"kind": "genotype",
                                                             "levels": {"Ctrl": "w1118"}}}}})
    editor = ExperimentEditor(path)
    try:
        assert editor.title_edit.text() == "T"
        assert editor.sex_checks["female"].isChecked()
        assert "mixed" in editor.other_sex_label.text()
        assert editor.factor_table.kinds() == {"Genotype": "genotype"}
        assert editor.meaning_table.meanings() == {"Genotype": {"Ctrl": "w1118"}}
        about = editor.about_section()
        assert about["sex"] == ["female", "mixed"]
        assert about["lab_book"] == "p. 12"                 # no field, kept
    finally:
        editor.close()


def test_renaming_a_level_drops_its_meaning_and_keeps_the_others(app, tmp_path):
    path = tmp_path / "exp" / "flic_config.yaml"
    _write(path, FULL)
    editor = ExperimentEditor(path)
    try:
        editor.meaning_table.item(0, 2).setText("control cross")
        editor.meaning_table.item(1, 2).setText("experimental cross")
        editor.factor_table.table.item(0, 2).setText("Ctrl, Exp2")
        assert editor.meaning_table.meanings() == {"Genotype": {"Ctrl": "control cross"}}
        ## The chamber columns follow the factors too.
        assert editor._dfm_widgets[0].levels_for_column(1) == ["Ctrl", "Exp2"]
    finally:
        editor.close()


def test_a_new_recording_gets_a_config_the_loader_and_linter_accept(app, tmp_path):
    folder = tmp_path / "FLICMCU_11_21_2026_12_01"
    for dfm_id in (1, 2):
        _dfm_csv(folder / "data", dfm_id)
    (folder / "data" / "Program.txt").write_text(
        (Path(__file__).parent / "fixtures" / "Program_simple.txt").read_text())
    editor = ExperimentEditor(folder)
    try:
        assert [w._id_spin.value() for w in editor._dfm_widgets] == [1, 2]
        assert "DFM 1, 2" in editor.data_label.text()
        assert editor.started_label.text().startswith("Started ")
        editor.title_edit.setText("New recording")
        editor.factor_table.add_row("Treatment", "treatment", ["Ctrl", "Drug"])
        table = editor._dfm_widgets[0]._chamber_table
        from PyQt6.QtWidgets import QTableWidgetItem
        table.setItem(0, 1, QTableWidgetItem("Ctrl"))
        table.setItem(1, 1, QTableWidgetItem("Drug"))
        assert editor.write(folder / "flic_config.yaml", confirm=False)
    finally:
        editor.close()

    saved = _read(folder / "flic_config.yaml")
    assert saved["about"]["title"] == "New recording"
    assert saved["global"] == {
        "chamber_layout": "two_well",
        "experimental_design_factors": {"Treatment": ["Ctrl", "Drug"]},
        "params": {},
    }
    assert saved["dfms"][0]["chambers"] == {1: "Ctrl", 2: "Drug"}
    assert not [i for i in lint_flic_config(folder / "flic_config.yaml")
                if i.severity == "error"]
    exp = load_experiment_yaml(folder, eager=False, use_disk_cache=False)
    assert sorted(exp.dfms) == [1, 2]


def test_choosing_a_type_that_shrinks_the_plate_can_be_declined(app, tmp_path, monkeypatch):
    path = tmp_path / "exp" / "flic_config.yaml"
    _write(path, {"global": {"chamber_layout": "single_well", "params": {}},
                  "dfms": [{"id": 1, "chambers": {10: "Ctrl"}}]})
    editor = ExperimentEditor(path)
    try:
        assert editor.chamber_layout() == "single_well"
        monkeypatch.setattr(QMessageBox, "question",
                            staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))
        editor.type_combo.setCurrentIndex(editor.type_combo.findData("Hedonic"))
        assert editor.type_combo.currentData() == "Custom"
        assert editor.chamber_layout() == "single_well"
        assert editor._dfm_widgets[0].get_dict()["chambers"] == {10: "Ctrl"}
    finally:
        editor.close()


# ---------------------------------------------------------------------------
# Inside a Project
# ---------------------------------------------------------------------------

DESIGN = {"experiment_type": "Hedonic", "params": {"feeding_threshold": 20},
          "well_names": {"A": "S5", "B": "S5Y5"},
          "experimental_design_factors": {"Genotype": ["Ctrl", "Exp"]}}


def _member(tmp_path: Path, config: dict) -> Path:
    root = tmp_path / "Proj"
    _write(root / "project.yaml", {"name": "Proj", "design": {"global": DESIGN}})
    _write(root / "rep1" / "flic_config.yaml", config)
    _dfm_csv(root / "rep1" / "data", 1)
    return root / "rep1" / "flic_config.yaml"


def test_a_member_shows_the_design_and_saves_the_design_copy(app, tmp_path):
    path = _member(tmp_path, {"dfms": [{"id": 1, "chambers": {1: "Ctrl"}}]})
    editor = ExperimentEditor(path)
    try:
        assert editor.design_banner.isVisible() or editor.design_banner.text()
        assert not editor.type_combo.isEnabled()
        assert not editor.factor_table.add_button.isEnabled()
        assert editor.factor_table.factors() == DESIGN["experimental_design_factors"]
        _set_kind(editor, 0, "genotype")             # the kind is the recording's
        editor.title_edit.setText("rep1")
        assert editor.write(path, confirm=False)
    finally:
        editor.close()
    saved = _read(path)
    assert saved["global"] == project_mod.design_copy(DESIGN)
    assert saved["about"]["factors"] == {"Genotype": {"kind": "genotype"}}
    assert project_mod.Project(path.parent.parent).member_names == ["rep1"]


def test_a_conflicting_member_keeps_its_description(app, tmp_path):
    own = {"experiment_type": "Hedonic", "well_names": {"A": "S5", "B": "Yeast"}}
    path = _member(tmp_path, {"global": own, "dfms": [{"id": 1, "chambers": {1: "Ctrl"}}]})
    editor = ExperimentEditor(path)
    try:
        editor.title_edit.setText("rep1")
        assert editor.write(path, confirm=False)
    finally:
        editor.close()
    assert _read(path)["global"] == own


# ---------------------------------------------------------------------------
# dfms: in its mapping form
# ---------------------------------------------------------------------------

def test_a_config_whose_dfms_are_keyed_by_id_keeps_its_chambers(app, tmp_path):
    """``dfms:`` may be a mapping keyed by DFM id, as the loader and the Config
    Editor read it: the editor must show those DFMs, not save an empty list."""
    path = tmp_path / "exp" / "flic_config.yaml"
    _write(path, {"global": {"chamber_layout": "two_well", "params": {}},
                  "dfms": {1: {"chambers": {1: "Ctrl", 2: "Exp"}},
                           4: {"params": {"pi_direction": "right"},
                               "chambers": {3: "Exp"}}}})
    editor = ExperimentEditor(path)
    try:
        assert [w._id_spin.value() for w in editor._dfm_widgets] == [1, 4]
        assert editor.write(path, confirm=False)
    finally:
        editor.close()
    saved = _read(path)
    assert saved["dfms"] == [{"id": 1, "chambers": {1: "Ctrl", 2: "Exp"}},
                             {"id": 4, "params": {"pi_direction": "right"},
                              "chambers": {3: "Exp"}}]


# ---------------------------------------------------------------------------
# about: as a catalogue leaves it (ADR-0026)
# ---------------------------------------------------------------------------
#
# A catalogue that files the recording rewrites about: to match its record:
# it writes run:, and inside a level's mapping term:, label: and role:.  The
# editor shows run:, edits only a level's definition, and writes back
# everything it does not show.

ABOUT = {
    "run": "R-20261121-scott-01",
    "title": "Hedonic, replicate 1",
    "purpose": "Whether silencing MBON10 changes hedonic feeding.",
    "experimenter": "scott",
    "series": "hedonic-2026",
    "organism": "Drosophila melanogaster",
    "sex": ["female"],
    "age_at_start": "5-7 days",
    "protocol": {"name": "FLIC hedonic", "version": "2", "link": "https://example.org/h",
                 "x_doi": "kept"},
    "notes": "Lid 3 was replaced.",
    "factors": {
        "Genotype": {
            "kind": "genotype",
            "levels": {
                "Exp": "R12C11-GAL4 > UAS-GtACR1",
                "Ctrl": {"definition": "w1118 > UAS-GtACR1", "term": "GEN-0012",
                         "label": "w1118 x UAS-GtACR1", "role": "reference",
                         "x_other": "kept"},
            },
            "x_factor": "kept",
        },
    },
    "x_lab": {"a": 1},
}
ABOUT_ORDER = ["run", "title", "purpose", "experimenter", "series", "organism", "sex",
               "age_at_start", "protocol", "notes", "factors", "x_lab"]


def _filed(tmp_path: Path, about: dict = ABOUT, name: str = "exp") -> Path:
    path = tmp_path / name / "flic_config.yaml"
    _write(path, {**FULL, "about": about})
    return path


def _meaning(editor: ExperimentEditor, factor: str, level: str):
    table = editor.meaning_table
    for row in range(table.rowCount()):
        if (table.item(row, 0).text(), table.item(row, 1).text()) == (factor, level):
            return table.item(row, 2)
    raise AssertionError(f"no row for {factor} {level}")


def test_a_save_that_edits_nothing_keeps_about_as_a_catalogue_left_it(app, tmp_path):
    path = _filed(tmp_path)
    editor = ExperimentEditor(path)
    try:
        assert editor.write(path, confirm=False)
    finally:
        editor.close()
    about = _read(path)["about"]
    assert about == ABOUT
    assert list(about) == ABOUT_ORDER
    ## Unknown keys beside kind: and levels: are a factor's too, and the levels
    ## stay in the order the file had them.
    assert about["factors"]["Genotype"]["x_factor"] == "kept"
    assert list(about["factors"]["Genotype"]["levels"]) == ["Exp", "Ctrl"]


def test_editing_a_meaning_changes_only_the_definition(app, tmp_path):
    path = _filed(tmp_path)
    editor = ExperimentEditor(path)
    try:
        assert _meaning(editor, "Genotype", "Ctrl").text() == "w1118 > UAS-GtACR1"
        _meaning(editor, "Genotype", "Ctrl").setText("the control cross")
        _meaning(editor, "Genotype", "Exp").setText("the silenced cross")
        assert editor.write(path, confirm=False)
    finally:
        editor.close()
    about = _read(path)["about"]
    levels = about["factors"]["Genotype"]["levels"]
    was = ABOUT["factors"]["Genotype"]["levels"]["Ctrl"]
    assert levels["Ctrl"] == {**was, "definition": "the control cross"}
    assert list(levels["Ctrl"])[0] == "definition"
    assert levels["Exp"] == "the silenced cross"             # text stays text
    about["factors"]["Genotype"]["levels"] = ABOUT["factors"]["Genotype"]["levels"]
    assert about == ABOUT                                    # and nothing else changed


def test_a_meaning_cleared_from_a_mapping_keeps_its_other_keys(app, tmp_path):
    path = _filed(tmp_path)
    editor = ExperimentEditor(path)
    try:
        _meaning(editor, "Genotype", "Ctrl").setText("")
        _meaning(editor, "Genotype", "Exp").setText("  ")
        assert editor.write(path, confirm=False)
    finally:
        editor.close()
    levels = _read(path)["about"]["factors"]["Genotype"]["levels"]
    assert levels == {"Ctrl": {"term": "GEN-0012", "label": "w1118 x UAS-GtACR1",
                               "role": "reference", "x_other": "kept"}}


def test_run_is_shown_never_edited_and_written_back_first(app, tmp_path):
    from PyQt6.QtWidgets import QLabel

    path = _filed(tmp_path)
    editor = ExperimentEditor(path)
    try:
        assert editor.run_label.text() == "Filed as R-20261121-scott-01"
        assert not editor.run_label.isHidden()
        assert type(editor.run_label) is QLabel              # nothing to type into
        editor.title_edit.setText("A better title")
        assert editor.write(path, confirm=False)
    finally:
        editor.close()
    about = _read(path)["about"]
    assert list(about)[0] == "run"
    assert about["run"] == "R-20261121-scott-01"
    assert about["title"] == "A better title"

    plain = tmp_path / "unfiled" / "flic_config.yaml"
    _write(plain, {**FULL, "about": {"title": "Not filed yet"}})
    editor = ExperimentEditor(plain)
    try:
        assert editor.run_label.isHidden()
        assert editor.run_label.text() == ""
        assert editor.write(plain, confirm=False)
    finally:
        editor.close()
    assert "run" not in _read(plain)["about"]


def test_what_the_form_never_showed_is_kept_and_what_a_rename_removes_is_dropped(
        app, tmp_path):
    """The design here declares Genotype: Ctrl, Exp.  The file also describes
    a level and a factor it does not declare: they have no row, so they are
    written back as read (and still warned of).  A level renamed in this
    editor loses its description, as it always has."""
    old = {"definition": "retired cross", "term": "GEN-0009"}
    dose = {"kind": "treatment", "levels": {"2": "2% sucrose"}}
    path = _filed(tmp_path, {"title": "T", "factors": {
        "Genotype": {"kind": "genotype",
                     "levels": {"Ctrl": "w1118", "Old": old,
                                "Exp": {"definition": "x", "term": "GEN-0002"}}},
        "Dose": dose}})
    editor = ExperimentEditor(path)
    try:
        editor.factor_table.table.item(0, 2).setText("Ctrl, Exp2")
        assert any("Old" in problem for problem in editor.problems())
        assert editor.write(path, confirm=False)
    finally:
        editor.close()
    factors = _read(path)["about"]["factors"]
    assert factors == {"Genotype": {"kind": "genotype",
                                    "levels": {"Ctrl": "w1118", "Old": old}},
                       "Dose": dose}


def test_a_level_declared_here_shows_what_the_file_already_says_about_it(app, tmp_path):
    old = {"definition": "retired cross", "term": "GEN-0009"}
    path = _filed(tmp_path, {"factors": {"Genotype": {"levels": {"Ctrl": "w1118",
                                                                   "Old": old}}}})
    editor = ExperimentEditor(path)
    try:
        editor.factor_table.table.item(0, 2).setText("Ctrl, Exp, Old")
        assert _meaning(editor, "Genotype", "Old").text() == "retired cross"
        ## A rename and back: the file's words return, not a blank.
        editor.factor_table.table.item(0, 2).setText("Control, Exp, Old")
        editor.factor_table.table.item(0, 2).setText("Ctrl, Exp, Old")
        assert _meaning(editor, "Genotype", "Ctrl").text() == "w1118"
        ## What is typed in a row that stays is kept, a blank included.
        _meaning(editor, "Genotype", "Ctrl").setText("")
        editor.factor_table.table.item(0, 2).setText("Ctrl, Exp, Old, New")
        assert _meaning(editor, "Genotype", "Ctrl").text() == ""
        assert editor.write(path, confirm=False)
    finally:
        editor.close()
    assert _read(path)["about"]["factors"] == {"Genotype": {"levels": {"Old": old}}}


def test_a_factor_named_here_takes_the_kind_the_file_gives_it(app, tmp_path):
    path = tmp_path / "exp" / "flic_config.yaml"
    _write(path, {"about": {"factors": {"Dose": {"kind": "treatment",
                                                 "levels": {"2": "2% sucrose"}}}},
                  "global": {"chamber_layout": "two_well", "params": {}},
                  "dfms": [{"id": 1}]})
    editor = ExperimentEditor(path)
    try:
        assert editor.factor_table.factors() == {}
        editor.factor_table.add_row()
        editor.factor_table.table.item(0, 0).setText("Dose")
        editor.factor_table.table.item(0, 2).setText("2, 4")
        assert editor.factor_table.kinds() == {"Dose": "treatment"}
        assert _meaning(editor, "Dose", "2").text() == "2% sucrose"
        assert editor.write(path, confirm=False)
    finally:
        editor.close()
    assert _read(path)["about"]["factors"] == {
        "Dose": {"kind": "treatment", "levels": {"2": "2% sucrose"}}}


def test_sex_is_written_as_the_file_had_it_until_it_is_changed(app, tmp_path):
    path = _filed(tmp_path, {"sex": ["Male", "female", "mixed"]})
    editor = ExperimentEditor(path)
    try:
        assert editor.write(path, confirm=False)
        assert _read(path)["about"]["sex"] == ["Male", "female", "mixed"]
        editor.sex_checks["male"].setChecked(False)
        assert editor.write(path, confirm=False)
    finally:
        editor.close()
    assert _read(path)["about"]["sex"] == ["female", "mixed"]


def test_an_about_that_is_not_a_mapping_is_kept_until_something_is_typed(app, tmp_path):
    path = _filed(tmp_path, "see the lab book")
    editor = ExperimentEditor(path)
    try:
        assert editor.write(path, confirm=False)
        assert _read(path)["about"] == "see the lab book"
        editor.title_edit.setText("A title")
        assert editor.write(path, confirm=False)
    finally:
        editor.close()
    assert _read(path)["about"] == {"title": "A title"}


# ---------------------------------------------------------------------------
# A file that changed on disk since the editor read it (ADR-0026)
# ---------------------------------------------------------------------------

def _rewritten(path: Path) -> dict:
    """What a catalogue, the Config Editor and the Script Editor did to the
    file while this editor had it open."""
    config = _read(path)
    config["about"] = {**config["about"], "run": "R-20261121-scott-02",
                       "title": "Rewritten elsewhere", "x_lab": {"a": 2}}
    config["about"]["factors"]["Genotype"]["levels"]["Ctrl"]["term"] = "GEN-0099"
    config["global"]["params"]["feeding_threshold"] = 30
    config["dfms"][0]["params"] = {"pi_direction": "left"}
    config["dfms"][0]["chambers"] = {1: "Exp", 2: "Exp"}
    config["scripts"] = [{"name": "theirs", "steps": [{"action": "load"}]}]
    _write(path, config)
    return config


def _never_asked(*_args):
    pytest.fail("asked about a file that did not change on disk")


def test_a_file_changed_on_disk_is_not_written_until_the_person_chooses(app, tmp_path):
    path = _filed(tmp_path)
    editor = ExperimentEditor(path)
    try:
        editor.notes_edit.setPlainText("Typed here.")
        theirs = _rewritten(path)
        asked: list = []

        def cancel(file, readable):
            asked.append((Path(file), readable))
            return "cancel"

        editor._ask_changed_on_disk = cancel
        assert editor.write(path, confirm=False) is False
        assert asked == [(path, True)]                       # it said so
        assert _read(path) == theirs                         # and wrote nothing
        assert editor.notes_edit.toPlainText() == "Typed here."

        ## Reload: the form shows the file as it now is.  Still nothing written.
        assert editor.write(path, confirm=False, if_changed="reload") is False
        assert _read(path) == theirs
        assert editor.title_edit.text() == "Rewritten elsewhere"
        assert editor.notes_edit.toPlainText() == ABOUT["notes"]
        assert editor.run_label.text() == "Filed as R-20261121-scott-02"

        ## Read again, the file is this editor's to save without a question.
        editor._ask_changed_on_disk = _never_asked
        assert editor.write(path, confirm=False)
        assert _read(path)["about"] == theirs["about"]
        assert editor.write(path, confirm=False)             # nor after its own save
    finally:
        editor.close()


def test_overwriting_writes_what_this_editor_edits_over_the_file_as_it_now_is(
        app, tmp_path):
    path = _filed(tmp_path)
    editor = ExperimentEditor(path)
    try:
        editor.notes_edit.setPlainText("Typed here.")
        _meaning(editor, "Genotype", "Ctrl").setText("the control cross")
        theirs = _rewritten(path)
        assert editor.write(path, confirm=False, if_changed="overwrite")
        saved = _read(path)
        about = saved["about"]
        ctrl = about["factors"]["Genotype"]["levels"]["Ctrl"]
        ## What the form has a field for is this editor's ...
        assert about["notes"] == "Typed here."
        assert about["title"] == ABOUT["title"]
        assert ctrl["definition"] == "the control cross"
        assert saved["dfms"][0]["chambers"] == {1: "Ctrl", 2: "Exp"}
        ## ... and everything it has none for is as the file now has it.
        assert about["run"] == "R-20261121-scott-02"
        assert ctrl["term"] == "GEN-0099"
        assert about["x_lab"] == {"a": 2}
        assert saved["global"]["params"] == {"feeding_threshold": 30}
        assert saved["global"]["constants"] == theirs["global"]["constants"]
        assert saved["scripts"] == theirs["scripts"]
        assert saved["dfms"][0]["params"] == {"pi_direction": "left"}
        assert saved["dfms"][0]["note"] == "lid cracked"
        assert saved["data_folder"] == theirs["data_folder"]
        ## The form now shows the file it wrote, and saves it again unasked.
        assert editor.run_label.text() == "Filed as R-20261121-scott-02"
        assert editor.notes_edit.toPlainText() == "Typed here."
        editor._ask_changed_on_disk = _never_asked
        assert editor.write(path, confirm=False)
        assert _read(path) == saved
    finally:
        editor.close()


def test_a_file_that_no_longer_reads_is_replaced_only_when_the_person_says_so(
        app, tmp_path):
    path = _filed(tmp_path)
    editor = ExperimentEditor(path)
    try:
        path.write_text("about: [not closed", encoding="utf-8")
        seen: list = []

        def answer(_file, readable):
            seen.append(readable)
            return "cancel" if len(seen) == 1 else "overwrite"

        editor._ask_changed_on_disk = answer
        assert editor.write(path, confirm=False) is False
        assert path.read_text(encoding="utf-8") == "about: [not closed"
        assert editor.write(path, confirm=False)
        assert seen == [False, False]                        # nothing there to keep
    finally:
        editor.close()
    assert _read(path)["about"] == ABOUT


def test_a_file_that_vanished_is_written_again_without_a_question(app, tmp_path):
    path = _filed(tmp_path)
    editor = ExperimentEditor(path)
    try:
        path.unlink()
        editor._ask_changed_on_disk = _never_asked
        assert editor.write(path, confirm=False)
    finally:
        editor.close()
    assert _read(path)["about"] == ABOUT


def test_a_config_written_while_a_new_one_was_being_described_is_noticed(app, tmp_path):
    folder = tmp_path / "new"
    _dfm_csv(folder / "data", 1)
    editor = ExperimentEditor(folder)
    try:
        editor.title_edit.setText("Described here")
        theirs = {"about": {"title": "Described elsewhere"},
                  "global": {"chamber_layout": "two_well", "params": {}},
                  "dfms": [{"id": 1, "chambers": {1: "Ctrl"}}]}
        _write(folder / "flic_config.yaml", theirs)
        assert editor.write(folder / "flic_config.yaml", confirm=False,
                            if_changed="cancel") is False
        assert _read(folder / "flic_config.yaml") == theirs
    finally:
        editor.close()


def test_saving_as_into_another_folder_does_not_take_run_along(app, tmp_path, monkeypatch):
    from pyflic.base import experiment_editor as module

    path = _filed(tmp_path)
    other = tmp_path / "rep2" / "flic_config.yaml"
    other.parent.mkdir()
    monkeypatch.setattr(module.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(other), "")))
    editor = ExperimentEditor(path)
    try:
        editor._save_as()
        assert editor.run_label.isHidden()                   # this file is not filed
        assert editor.write(other, confirm=False)
    finally:
        editor.close()
    assert _read(other)["about"] == {k: v for k, v in ABOUT.items() if k != "run"}
    assert _read(path)["about"] == ABOUT                     # the one it was read from

    ## A copy beside the config is the same recording's, and keeps it.
    beside = path.with_name("flic_config_copy.yaml")
    editor = ExperimentEditor(path)
    try:
        assert editor.write(beside, confirm=False)
    finally:
        editor.close()
    assert _read(beside)["about"] == ABOUT


# ---------------------------------------------------------------------------
# Start from another recording (ADR-0026)
# ---------------------------------------------------------------------------
#
# Replicate recordings share most of their description.  The action takes an
# earlier recording's config and fills the form with its shared About: who
# ran it, the series, the flies, the protocol, and each factor's kind and each
# level's whole entry.  The title, the purpose, the notes and run: stay this
# recording's own.  A recording with no design yet takes the earlier one's
# design and chamber map too.  Nothing is written until the person saves.

SHARED = ("experimenter", "series", "organism", "sex", "age_at_start", "protocol", "factors")
EARLIER = {
    "about": ABOUT,
    "global": FULL["global"],
    "dfms": [{"id": 1, "optogenetics": False, "params": {"pi_direction": "right"},
              "note": "lid cracked", "chambers": {1: "Ctrl", 2: "Exp"}},
             {"id": 2, "chambers": {1: "Exp"}},
             {"id": 3, "chambers": {3: "Ctrl", 4: "Exp"}}],
    "scripts": FULL["scripts"],
}


def _earlier(tmp_path: Path, config: dict = EARLIER) -> Path:
    path = tmp_path / "rep1" / "flic_config.yaml"
    _write(path, config)
    return path


def _questions(monkeypatch, answer=QMessageBox.StandardButton.Yes) -> list[str]:
    """Record what the editor asks, answering *answer*."""
    asked: list[str] = []

    def question(*args, **_kwargs):
        asked.append(" ".join(str(a) for a in args if isinstance(a, str)))
        return answer

    monkeypatch.setattr(QMessageBox, "question", staticmethod(question))
    return asked


def test_starting_from_another_recording_fills_only_the_shared_about(
        app, tmp_path, monkeypatch):
    mine = {"run": "R-20261128-scott-01", "title": "Replicate 2", "purpose": "Mine.",
            "notes": "My notes.", "experimenter": "someone", "organism": "D. simulans",
            "x_mine": True}
    path = _filed(tmp_path, mine, name="rep2")
    earlier = _earlier(tmp_path)
    before = path.read_bytes()
    asked = _questions(monkeypatch)
    editor = ExperimentEditor(path)
    try:
        editor.purpose_edit.setPlainText("Mine, as just typed.")
        assert editor.start_from(earlier.parent)             # a folder holding one
        assert path.read_bytes() == before                   # nothing is written yet
        assert asked == []
        ## The form shows what was taken, and what stayed.
        assert editor.title_edit.text() == "Replicate 2"
        assert editor.purpose_edit.toPlainText() == "Mine, as just typed."
        assert editor.notes_edit.toPlainText() == "My notes."
        assert editor.run_label.text() == "Filed as R-20261128-scott-01"
        assert editor.experimenter_edit.text() == "scott"
        assert editor.series_edit.text() == "hedonic-2026"
        assert editor.organism_edit.text() == "Drosophila melanogaster"
        assert editor.sex_checks["female"].isChecked()
        assert editor.age_edit.text() == "5-7 days"
        assert editor.protocol_edits["link"].text() == "https://example.org/h"
        assert editor.factor_table.kinds() == {"Genotype": "genotype"}
        assert _meaning(editor, "Genotype", "Ctrl").text() == "w1118 > UAS-GtACR1"
        assert editor.write(path, confirm=False)
    finally:
        editor.close()
    about = _read(path)["about"]
    for key in SHARED:
        assert about[key] == ABOUT[key]                      # mapping keys and all
    assert about["run"] == "R-20261128-scott-01"
    assert about["title"] == "Replicate 2"
    assert about["purpose"] == "Mine, as just typed."
    assert about["notes"] == "My notes."
    assert about["x_mine"] is True
    assert "x_lab" not in about                              # the earlier one's own
    assert _read(earlier) == EARLIER                         # and it is only read


def test_with_a_design_here_only_the_shared_about_is_taken_and_nothing_is_asked(
        app, tmp_path, monkeypatch):
    path = _filed(tmp_path, {"title": "Replicate 2"}, name="rep2")
    other = {"about": {**ABOUT, "factors": {
                 "Genotype": {"kind": "genotype",
                              "levels": {"Ctrl": "w1118", "Exp": "the cross",
                                         "Other": "not used here"}},
                 "Dose": {"kind": "treatment", "levels": {"1": "1 mM"}}}},
             "global": {"chamber_layout": "single_well", "params": {},
                        "experimental_design_factors": {
                            "Genotype": ["Ctrl", "Exp", "Other"], "Dose": ["1"]}},
             "dfms": [{"id": 1, "chambers": {1: "Other, 1", 9: "Ctrl, 1"}}]}
    earlier = _earlier(tmp_path, other)
    asked = _questions(monkeypatch)
    editor = ExperimentEditor(path)
    try:
        design = editor.design_section()
        chambers = [w.get_dict() for w in editor._dfm_widgets]
        assert editor.start_from(earlier)
        assert asked == []
        assert editor.design_section() == design
        assert [w.get_dict() for w in editor._dfm_widgets] == chambers
        about = editor.about_section()
    finally:
        editor.close()
    assert about["experimenter"] == "scott"
    assert about["title"] == "Replicate 2"
    ## Limited to the factors and levels this recording declares.
    assert about["factors"] == {"Genotype": {"kind": "genotype",
                                             "levels": {"Ctrl": "w1118", "Exp": "the cross"}}}


def test_a_new_recording_takes_the_design_and_the_chamber_map_for_its_dfms(
        app, tmp_path, monkeypatch):
    folder = tmp_path / "FLICMCU_11_28_2026_09_30"
    for dfm_id in (1, 3, 5):
        _dfm_csv(folder / "data", dfm_id)
    earlier = _earlier(tmp_path)
    asked = _questions(monkeypatch)
    editor = ExperimentEditor(folder)
    try:
        assert editor.start_from(earlier)
        assert asked == []                                   # nothing here to replace
        assert not (folder / "flic_config.yaml").exists()    # nothing written yet
        assert editor.type_combo.currentData() == "Hedonic"
        assert editor.factor_table.factors() == {"Genotype": ["Ctrl", "Exp"]}
        assert editor.well_a_edit.text() == "S5"
        ## The chamber map of the DFMs this recording's data holds: DFM 2 is
        ## the earlier recording's alone, and DFM 5 this one's.  A DFM's
        ## overrides and notes are the earlier recording's own.
        assert [w.get_dict() for w in editor._dfm_widgets] == [
            {"id": 1, "chambers": {1: "Ctrl", 2: "Exp"}},
            {"id": 3, "chambers": {3: "Ctrl", 4: "Exp"}},
            {"id": 5}]
        editor.title_edit.setText("Hedonic, replicate 2")
        assert editor.write(folder / "flic_config.yaml", confirm=False)
    finally:
        editor.close()
    saved = _read(folder / "flic_config.yaml")
    assert list(saved) == ["about", "global", "dfms"]
    assert saved["about"] == {"title": "Hedonic, replicate 2",
                              **{key: ABOUT[key] for key in SHARED}}
    ## The design, and none of the earlier recording's analysis settings.
    assert saved["global"] == {
        "experiment_type": "Hedonic", "well_names": {"A": "S5", "B": "S5Y5"},
        "experimental_design_factors": {"Genotype": ["Ctrl", "Exp"]}, "params": {}}
    assert not [i for i in lint_flic_config(folder / "flic_config.yaml")
                if i.severity == "error"]
    assert sorted(load_experiment_yaml(folder, eager=False, use_disk_cache=False).dfms) \
        == [1, 3, 5]


def test_a_progressive_ratio_replicate_takes_its_chamber_groups_with_the_map(
        app, tmp_path, monkeypatch):
    six = {1: "Ctrl", 2: "Ctrl", 3: "Exp", 4: "Exp", 5: "Exp", 6: "Exp"}
    other = {"global": {"experiment_type": "ProgressiveRatio",
                        "well_names": {"A": "Sucrose", "B": "Yeast"}},
             "dfms": {1: {"params": {"pi_direction": "left"},
                          "paired_chambers": [1, 4, 5], "chambers": six},
                      2: {"chamber_groups": "single", "chambers": six},
                      3: {"chambers": six}}}
    folder = tmp_path / "pr2"
    for dfm_id in (1, 2, 3):
        _dfm_csv(folder / "data", dfm_id)
    earlier = _earlier(tmp_path, other)
    editor = ExperimentEditor(folder)
    try:
        assert editor.start_from(earlier)
        assert editor.type_combo.currentData() == "ProgressiveRatio"
        assert [w.get_dict() for w in editor._dfm_widgets] == [
            {"id": 1, "paired_chambers": [1, 4, 5], "chambers": six},
            {"id": 2, "chamber_groups": "single", "chambers": six},
            {"id": 3, "chambers": six}]                      # its groups: Program.txt's
    finally:
        editor.close()


def test_a_design_stated_here_is_replaced_only_after_asking(app, tmp_path, monkeypatch):
    stated = {"global": {"experiment_type": "Hedonic", "params": {},
                         "well_names": {"A": "Sucrose", "B": "Yeast"}},
              "dfms": [{"id": 1}]}
    path = tmp_path / "rep2" / "flic_config.yaml"
    _write(path, stated)
    earlier = _earlier(tmp_path)

    asked = _questions(monkeypatch, QMessageBox.StandardButton.No)
    editor = ExperimentEditor(path)
    try:
        assert editor.start_from(earlier)
        assert len(asked) == 1 and "Well A" in asked[0] and "Sucrose" in asked[0]
        ## Declined: the design and the chambers stay, the shared About is taken.
        assert editor.well_a_edit.text() == "Sucrose"
        assert editor.factor_table.factors() == {}
        assert editor._dfm_widgets[0].get_dict() == {"id": 1}
        assert editor.experimenter_edit.text() == "scott"
        ## It declares no factors to limit to, so the earlier recording's
        ## descriptions come along, and the editor says they are undeclared.
        assert editor.about_section()["factors"] == ABOUT["factors"]
        assert any("Genotype" in problem for problem in editor.problems())
    finally:
        editor.close()

    asked = _questions(monkeypatch, QMessageBox.StandardButton.Yes)
    editor = ExperimentEditor(path)
    try:
        assert editor.start_from(earlier)
        assert len(asked) == 1
        assert editor.well_a_edit.text() == "S5"
        assert editor.factor_table.factors() == {"Genotype": ["Ctrl", "Exp"]}
        assert editor._dfm_widgets[0].get_dict() == {"id": 1,
                                                     "chambers": {1: "Ctrl", 2: "Exp"}}
    finally:
        editor.close()
    assert _read(path) == stated                             # still nothing written


def test_a_design_that_agrees_with_what_is_stated_here_is_taken_unasked(
        app, tmp_path, monkeypatch):
    path = tmp_path / "rep2" / "flic_config.yaml"
    _write(path, {"global": {"experiment_type": "Hedonic", "params": {},
                             "well_names": {"A": "S5"}},
                  "dfms": [{"id": 1}]})
    asked = _questions(monkeypatch)
    editor = ExperimentEditor(path)
    try:
        assert editor.start_from(_earlier(tmp_path))
        assert asked == []
        assert editor.well_b_edit.text() == "S5Y5"
        assert editor.factor_table.factors() == {"Genotype": ["Ctrl", "Exp"]}
    finally:
        editor.close()


def test_a_member_takes_only_the_shared_about_its_design_is_the_projects(
        app, tmp_path, monkeypatch):
    path = _member(tmp_path, {"dfms": [{"id": 1}]})
    other = {**EARLIER, "global": {**EARLIER["global"],
                                   "well_names": {"A": "Water", "B": "Yeast"}}}
    earlier = tmp_path / "elsewhere" / "flic_config.yaml"
    _write(earlier, other)
    asked = _questions(monkeypatch)
    editor = ExperimentEditor(path)
    try:
        assert editor.start_from(earlier)
        assert asked == []
        assert editor.well_a_edit.text() == "S5"             # the Project design's
        assert editor._dfm_widgets[0].get_dict() == {"id": 1}
        assert editor.factor_table.kinds() == {"Genotype": "genotype"}
        assert editor.write(path, confirm=False)
    finally:
        editor.close()
    saved = _read(path)
    assert saved["global"] == project_mod.design_copy(DESIGN)
    assert saved["about"] == {key: ABOUT[key] for key in SHARED}


def test_the_earlier_recording_may_be_a_member_whose_design_is_its_projects(
        app, tmp_path, monkeypatch):
    member = _member(tmp_path, {"about": ABOUT,
                                "dfms": [{"id": 1, "chambers": {1: "Ctrl", 2: "Exp"}}]})
    folder = tmp_path / "new"
    _dfm_csv(folder / "data", 1)
    editor = ExperimentEditor(folder)
    try:
        assert editor.start_from(member)
        assert editor.type_combo.currentData() == "Hedonic"
        assert editor.factor_table.factors() == DESIGN["experimental_design_factors"]
        assert editor._dfm_widgets[0].get_dict() == {"id": 1,
                                                     "chambers": {1: "Ctrl", 2: "Exp"}}
    finally:
        editor.close()


def test_starting_from_nothing_usable_changes_nothing(app, tmp_path, monkeypatch):
    path = _filed(tmp_path)
    said: list[str] = []
    for name in ("warning", "information"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(
            lambda *a, **k: said.append(" ".join(str(x) for x in a if isinstance(x, str)))))
    broken = tmp_path / "broken" / "flic_config.yaml"
    broken.parent.mkdir()
    broken.write_text("about: [not closed", encoding="utf-8")
    editor = ExperimentEditor(path)
    try:
        before = editor.collect()
        assert editor.start_from(path) is False              # its own config
        assert editor.start_from(tmp_path / "no_such_folder") is False
        assert editor.start_from(broken) is False
        assert len(said) == 3
        assert editor.collect() == before
    finally:
        editor.close()


def test_the_action_is_on_the_about_tab_and_in_the_file_menu(app, tmp_path, monkeypatch):
    from pyflic.base import experiment_editor as module

    path = _filed(tmp_path, {"title": "Replicate 2"}, name="rep2")
    earlier = _earlier(tmp_path)
    chosen: list[str] = []

    def pick(*args, **_kwargs):
        chosen.append(args[1])
        return str(earlier), ""

    monkeypatch.setattr(module.QFileDialog, "getOpenFileName", staticmethod(pick))
    editor = ExperimentEditor(path)
    try:
        actions = [a.text() for a in editor.menuBar().actions()[0].menu().actions()]
        assert "Start &from another recording…" in actions
        editor.start_from_button.click()
        assert len(chosen) == 1
        assert editor.experimenter_edit.text() == "scott"
    finally:
        editor.close()


def _lab_type(folder: Path, monkeypatch) -> None:
    """A lab Experiment Type with a Treatment control, in a library of its own."""
    from pyflic.base.experiment_types import library
    from pyflic.base.experiment_types.definition import parse_definition

    folder.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(library.ENV_LIBRARY, str(folder))
    definition, problems = parse_definition(
        {"name": "GenoPanel", "chamber_layout": "two_well", "required_wells": ["A", "B"],
         "factors": ["Genotype"], "control": {"treatment": "Genotype"}})
    assert problems == []
    library.save_to_library(definition)


PANEL = {"global": {"experiment_type": "GenoPanel",
                    "well_names": {"A": "Sucrose", "B": "Yeast"},
                    "experimental_design_factors": {"Genotype": ["w1118", "Gr5a"]},
                    "control_level": "w1118"},
         "dfms": [{"id": 1, "chambers": {1: "w1118", 2: "Gr5a"}}]}


def test_a_lab_types_control_level_is_part_of_the_design_taken(app, tmp_path, monkeypatch):
    _lab_type(tmp_path / "library", monkeypatch)
    folder = tmp_path / "new"
    _dfm_csv(folder / "data", 1)
    earlier = _earlier(tmp_path, PANEL)
    editor = ExperimentEditor(folder)
    try:
        assert editor.start_from(earlier)
        assert editor.type_combo.currentData() == "GenoPanel"
        assert editor.design_section() == PANEL["global"]

        ## Another control level stated here is something to ask about.
        editor.control_edit.setText("Gr5a")
        editor.factor_table.table.setRowCount(0)
        for widget in editor._dfm_widgets:
            widget.load_dict({"id": 1}, 2)
        asked = _questions(monkeypatch, QMessageBox.StandardButton.No)
        assert editor.start_from(earlier)
        assert len(asked) == 1 and "Control Level: Gr5a → w1118" in asked[0]
        assert editor.control_edit.text() == "Gr5a"
    finally:
        editor.close()


def test_a_design_of_a_type_unknown_here_is_not_taken(app, tmp_path, monkeypatch):
    folder = tmp_path / "new"
    _dfm_csv(folder / "data", 1)
    earlier = _earlier(tmp_path, {**PANEL, "about": ABOUT})
    said: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(
        lambda *a, **k: said.append(" ".join(str(x) for x in a if isinstance(x, str)))))
    editor = ExperimentEditor(folder)
    try:
        assert editor.start_from(earlier)
        assert len(said) == 1 and "GenoPanel" in said[0]
        ## The design is not guessed at: only the shared About is taken.
        assert editor.type_combo.currentData() == "Custom"
        assert editor.factor_table.factors() == {}
        assert editor._dfm_widgets[0].get_dict() == {"id": 1}
        assert editor.experimenter_edit.text() == "scott"
    finally:
        editor.close()
