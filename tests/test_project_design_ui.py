"""The Design as an authority the user can actually state and every Member
carries: the project.yaml editor, the two script levels, and the config
editor's refusal to let a Member contradict its Project (ADR-0005).

Headless — the widgets are driven directly.  What is asserted is the contract:
where a setting may be *edited*, what a save *writes*, and which tile is live.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

pytest.importorskip("PyQt6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from pyflic.base import project as project_mod  # noqa: E402
from pyflic.base.config_editor import FLICConfigEditor  # noqa: E402
from pyflic.base.design_editor import ProjectDesignDialog  # noqa: E402
from pyflic.base.hub import AnalysisHubWindow  # noqa: E402

DESIGN_GLOBAL = {
    "experiment_type": "hedonic",
    "transform_licks": False,
    "params": {"feeding_threshold": 42, "samples_per_second": 5},
    "well_names": {"A": "S5", "B": "S5Y5"},
    "experimental_design_factors": {"Genotype": ["w1118", "CS"],
                                    "Sex": ["M", "F"]},
    "constants": {"min_untransformed_licks_cutoff": 20},
}


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "Proj"
    _write(root / "project.yaml",
           {"name": "Proj", "design": {"global": DESIGN_GLOBAL},
            "scripts": [{"name": "batch", "steps": []}]})
    for dfm in (1, 2):
        (root / "rep1" / "data").mkdir(parents=True, exist_ok=True)
        (root / "rep1" / "data" / f"DFM{dfm}_0.csv").write_text(
            "Sample,Seconds\n", encoding="utf-8")
    _write(root / "rep1" / "flic_config.yaml",
           {"dfms": [{"id": 1, "chambers": {1: "w1118, M", 2: "CS, F"}}],
            "scripts": [{"name": "mine", "steps": [{"action": "load"}]}]})
    return root


# ---------------------------------------------------------------------------
# The project.yaml editor
# ---------------------------------------------------------------------------

def test_design_dialog_round_trips_an_existing_design(app, project):
    dialog = ProjectDesignDialog(start_dir=str(project))

    assert dialog.type_combo.currentData() == "Hedonic"
    assert dialog.well_a_edit.text() == "S5"
    assert dialog.transform_check.isChecked() is False
    assert dialog.factors_widget.get_factors() == \
        DESIGN_GLOBAL["experimental_design_factors"]

    built = dialog._build_design()["global"]
    assert built["params"]["feeding_threshold"] == 42
    assert built["experimental_design_factors"] == \
        DESIGN_GLOBAL["experimental_design_factors"]
    ## The type owns the layout: it is derived, never written (ADR-0007).
    assert "chamber_layout" not in built


def test_design_dialog_writes_project_yaml(app, tmp_path):
    root = tmp_path / "fresh"
    root.mkdir()
    dialog = ProjectDesignDialog(start_dir=str(root))
    dialog.name_edit.setText("Fresh")
    dialog.factors_widget.load_factors({"Treatment": ["Ctrl", "Exp"]})
    dialog.well_a_edit.setText("S5")
    dialog.well_b_edit.setText("S5Y5")
    dialog._save()

    assert dialog.saved_dir == str(root)
    written = yaml.safe_load((root / "project.yaml").read_text())
    assert written["name"] == "Fresh"
    assert written["design"]["global"]["experimental_design_factors"] == \
        {"Treatment": ["Ctrl", "Exp"]}
    ## Every Project ships a visible default Project Script.
    assert [s["name"] for s in written["scripts"]] == ["batch"]


def test_design_dialog_refuses_an_experiment_directory(app, tmp_path,
                                                       monkeypatch):
    root = tmp_path / "standalone"
    _write(root / "flic_config.yaml", {"dfms": []})
    dialog = ProjectDesignDialog(start_dir=str(root))
    warned: list[str] = []
    monkeypatch.setattr(dialog, "_warn", warned.append)
    dialog._save()

    assert dialog.saved_dir is None
    assert not (root / "project.yaml").exists()
    assert warned and "Experiment Directory" in warned[0]


def test_phase_names_must_match_the_cutoffs(app, tmp_path, monkeypatch):
    dialog = ProjectDesignDialog(start_dir=str(tmp_path))
    dialog.cutoffs_edit.setText("10, 70")
    dialog.labels_edit.setText("Only, Two")
    warned: list[str] = []
    monkeypatch.setattr(dialog, "_warn", warned.append)

    assert dialog._build_design() is None
    assert warned and "3 names" in warned[0]


# ---------------------------------------------------------------------------
# Members inherit it
# ---------------------------------------------------------------------------

def test_member_config_shows_the_design_and_refuses_to_edit_it(app, project):
    editor = FLICConfigEditor(project / "rep1" / "flic_config.yaml")

    assert editor._design is not None
    assert editor._design_banner.text().startswith("<b>These settings")
    ## Shown — the member's editor is where someone looks for what is in force.
    assert editor._global_params.get_values()["feeding_threshold"] == 42
    assert editor._factors_widget.get_factors() == \
        DESIGN_GLOBAL["experimental_design_factors"]
    ## …but not edited here.
    assert not editor._experiment_type_combo.isEnabled()
    assert not editor._well_a_edit.isEnabled()
    assert not editor._global_params._input_widgets[
        "feeding_threshold"].isEnabled()


def test_member_save_omits_global_and_keeps_its_scripts(app, project,
                                                       monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    ## The editor confirms a save with a modal box; headless it would block.
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    path = project / "rep1" / "flic_config.yaml"
    editor = FLICConfigEditor(path)
    editor._write_yaml(path)

    written = yaml.safe_load(path.read_text())
    ## Inheriting is what keeps the design an authority rather than a copy.
    assert "global" not in written
    ## The member's own scripts are not this editor's to delete.
    assert [s["name"] for s in written["scripts"]] == ["mine"]
    assert written["dfms"][0]["chambers"] == {1: "w1118, M", 2: "CS, F"}
    ## And the Project still loads, which is the whole point of the refusal.
    assert project_mod.Project(project).member_names == ["rep1"]


def test_design_factors_split_a_members_chamber_assignments(app, project):
    editor = FLICConfigEditor(project / "rep1" / "flic_config.yaml")
    table = editor._dfm_widgets[0]._chamber_table

    headers = [table.horizontalHeaderItem(i).text()
               for i in range(table.columnCount())]
    assert headers == ["Chamber", "Genotype", "Sex"]
    assert [table.item(0, c).text() for c in (1, 2)] == ["w1118", "M"]


def test_only_physical_keys_may_be_overridden_inside_a_project(app, project):
    editor = FLICConfigEditor(project / "rep1" / "flic_config.yaml")
    checks = editor._dfm_widgets[0]._params_form._enable_checks

    assert checks["pi_direction"].isEnabled()
    assert not checks["feeding_threshold"].isEnabled()


def test_a_standalone_experiment_stays_free(app, tmp_path):
    path = tmp_path / "solo" / "flic_config.yaml"
    _write(path, {"global": {"params": {"feeding_threshold": 7}},
                  "dfms": [{"id": 1, "chambers": {1: "Ctrl"}}]})
    editor = FLICConfigEditor(path)

    assert editor._design is None
    assert editor._global_params._input_widgets["feeding_threshold"].isEnabled()
    assert "global" in editor._collect_yaml()


# ---------------------------------------------------------------------------
# Reinforcing the design in members already on disk
# ---------------------------------------------------------------------------

def test_adopt_design_strips_only_the_global_block(tmp_path, project):
    member = project / "rep1" / "flic_config.yaml"
    payload = yaml.safe_load(member.read_text())
    payload["global"] = {"params": {"feeding_threshold": 999}}
    _write(member, payload)

    stating = project_mod.members_stating_global(project)
    assert stating == [("rep1", False)]
    with pytest.raises(ValueError):
        project_mod.Project(project)   # the deviation is fatal, as designed

    assert project_mod.adopt_design(project) == ["rep1"]
    after = yaml.safe_load(member.read_text())
    assert "global" not in after
    assert [s["name"] for s in after["scripts"]] == ["mine"]
    assert project_mod.Project(project).member_names == ["rep1"]


def test_adopt_design_refuses_without_a_design(tmp_path):
    root = tmp_path / "NoDesign"
    _write(root / "project.yaml", {"name": "NoDesign"})
    _write(root / "rep1" / "flic_config.yaml",
           {"global": {"params": {}}, "dfms": []})

    with pytest.raises(ValueError):
        project_mod.adopt_design(root)


# ---------------------------------------------------------------------------
# The two script levels sit in their own tiles
# ---------------------------------------------------------------------------

def test_scripts_tile_waits_for_a_loaded_member(app, project):
    hub = AnalysisHubWindow(target=str(project))
    try:
        assert hub.project is not None
        assert hub.experiment is None
        ## Project Scripts are the Project's, and live in its panel.
        assert hub.project_script.count()
        assert hub.run_project_script_btn.isEnabled()
        ## The Scripts tile is the member level: nothing loaded, nothing to run.
        assert hub.tiles["scripts"].is_dimmed()
        assert not hub.run_experiment_script_btn.isEnabled()
        assert not hub.edit_scripts_btn.isEnabled()
        for card in hub.panels["scripts"].cards():
            assert card.is_dimmed()
    finally:
        hub.close()


def test_project_design_button_says_when_there_is_no_design(app, tmp_path):
    root = tmp_path / "Bare"
    _write(root / "project.yaml", {"name": "Bare"})
    _write(root / "rep1" / "flic_config.yaml", {"dfms": []})
    hub = AnalysisHubWindow(target=str(root))
    try:
        assert "(none set)" in hub.design_btn.text()
        assert hub.design_btn.isEnabled()
    finally:
        hub.close()


def test_a_progressive_ratio_design_saves_with_its_fixed_phase_names(app, tmp_path,
                                                                    monkeypatch):
    """The type names Training and Test itself and has no cutoffs to count
    them against; the dialog must neither refuse the save nor write
    facet_cutoffs / facet_labels (ADR-0013)."""
    root = tmp_path / "pr"
    root.mkdir()
    dialog = ProjectDesignDialog(start_dir=str(root))
    dialog.type_combo.setCurrentIndex(dialog.type_combo.findData("ProgressiveRatio"))
    dialog.name_edit.setText("PR")
    dialog.well_a_edit.setText("Sucrose")
    dialog.well_b_edit.setText("Yeast")
    warned: list[str] = []
    monkeypatch.setattr(dialog, "_warn", warned.append)

    assert dialog.labels_edit.text() == "Training, Test"
    assert not dialog.labels_edit.isEnabled()
    assert not dialog.cutoffs_edit.isEnabled()
    dialog._save()

    assert warned == []
    assert dialog.saved_dir == str(root)
    g = yaml.safe_load((root / "project.yaml").read_text())["design"]["global"]
    assert g["experiment_type"] == "ProgressiveRatio"
    assert "facet_cutoffs" not in g and "facet_labels" not in g
    assert g["constants"]["require_training_complete"] is True


def test_the_progressive_ratio_light_qc_settings_round_trip(app, tmp_path):
    """The light QC's switches and thresholds are fields for a Progressive
    Ratio design, hidden for any other type, and a constant the form has no
    field for survives the save instead of being dropped."""
    root = tmp_path / "pr"
    _write(root / "project.yaml", {"name": "PR", "design": {"global": {
        "experiment_type": "ProgressiveRatio",
        "well_names": {"A": "Sucrose", "B": "Yeast"},
        "constants": {"exclude_failed_pr_groups": False,
                      "require_training_complete": False,
                      "pr_lick_free_run": 8, "my_own_note": 3},
    }}})
    dialog = ProjectDesignDialog(start_dir=str(root))
    assert not dialog.pr_qc_group.isHidden()
    assert dialog.pr_switch_checks["exclude_failed_pr_groups"].isChecked() is False
    assert dialog.pr_number_edits["pr_lick_free_run"].text() == "8"
    assert dialog.pr_number_edits["pr_trend_min_rho"].text() == "0.3"
    assert "pr_lick_free_run" in dialog.pr_number_edits["pr_lick_free_run"].toolTip()

    constants = dialog._build_design()["global"]["constants"]
    assert constants["exclude_failed_pr_groups"] is False
    assert constants["require_training_complete"] is False
    assert constants["pr_lick_free_run"] == 8
    assert constants["my_own_note"] == 3

    dialog.type_combo.setCurrentIndex(dialog.type_combo.findData("Hedonic"))
    assert dialog.pr_qc_group.isHidden()


def test_a_standalone_config_keeps_constants_it_has_no_field_for(app, tmp_path):
    path = tmp_path / "solo" / "flic_config.yaml"
    _write(path, {"global": {"experiment_type": "ProgressiveRatio",
                             "well_names": {"A": "Sucrose", "B": "Yeast"},
                             "constants": {"exclude_failed_pr_groups": False,
                                           "pr_lick_free_run": 9}},
                  "dfms": [{"id": 1, "paired_chambers": [1, 3, 5],
                            "chambers": {1: "Ctrl", 2: "Ctrl"}}]})
    editor = FLICConfigEditor(path)
    constants = editor._collect_yaml()["global"]["constants"]
    assert constants["exclude_failed_pr_groups"] is False
    assert constants["pr_lick_free_run"] == 9
