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
