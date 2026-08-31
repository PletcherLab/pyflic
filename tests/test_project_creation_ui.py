"""Three ways into a Project, three ways into a Member.

The cases are deliberately disjoint — the thing exists / it does not exist at
all / its directory exists but its config does not — and each button refuses
the other two's case rather than guessing.  These tests hold that line, and the
one rule that makes the trio safe: a config is checked against the design
*before* it is written, never after.

Mirrors PyTrackingAnalysis's Create/Load and Experiments cards.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

pytest.importorskip("PyQt6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import (  # noqa: E402
    QApplication,
    QInputDialog,
    QMessageBox,
)

from pyflic.base import layout as layout_mod  # noqa: E402
from pyflic.base import project as project_mod  # noqa: E402
from pyflic.base.design_editor import ProjectDesignDialog  # noqa: E402
from pyflic.base.hub import AnalysisHubWindow  # noqa: E402

DESIGN_GLOBAL = {
    "experiment_type": "hedonic",
    "params": {"feeding_threshold": 20},
    "well_names": {"A": "S5", "B": "S5Y5"},
    "experimental_design_factors": {"Treatment": ["Ctrl", "Exp"]},
}
MEMBER = {"dfms": [{"id": 1, "chambers": {i: "Ctrl" for i in range(1, 7)}}]}


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _recording(directory: Path, dfm: int = 1, *, filed: bool = True) -> None:
    target = directory / "data" if filed else directory
    target.mkdir(parents=True, exist_ok=True)
    (target / f"DFM{dfm}_0.csv").write_text("Sample,Seconds\n", encoding="utf-8")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "Proj"
    _write(root / "project.yaml",
           {"name": "Proj", "design": {"global": DESIGN_GLOBAL}})
    _recording(root / "rep1")
    _write(root / "rep1" / "flic_config.yaml", MEMBER)
    return root


@pytest.fixture
def hub(app, project):
    window = AnalysisHubWindow(target=str(project))
    yield window
    window.close()


def _silence(monkeypatch) -> list[str]:
    """Swallow the modal boxes, keeping their text for assertions."""
    said: list[str] = []

    def _capture(*args, **kwargs):
        for arg in args:
            if isinstance(arg, str):
                said.append(arg)
        return QMessageBox.StandardButton.No

    for name in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(_capture))
    return said


# ---------------------------------------------------------------------------
# Into a Project: exists / does not exist / exists without a project.yaml
# ---------------------------------------------------------------------------

def test_initialize_keeps_the_folders_own_name_and_infers_the_design(
        app, tmp_path, monkeypatch):
    ## Saving offers to make the existing member inherit the design just
    ## inferred from it — accept, which is the whole point of the flow: the
    ## design it wrote is *richer* than the partial global: the member states
    ## (every parameter, not the two that were set), so the member would
    ## otherwise be a deviation from a design taken off itself.
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k:
                                     QMessageBox.StandardButton.Yes))
    root = tmp_path / "OldStudy"
    _recording(root / "rep1")
    _write(root / "rep1" / "flic_config.yaml",
           {"global": DESIGN_GLOBAL, **MEMBER})

    dialog = ProjectDesignDialog(start_dir=str(root), initialize_existing=True)
    ## The folder names the Project — initializing in place does not rename it.
    assert dialog.name_edit.text() == "OldStudy"
    assert dialog.name_edit.isReadOnly()
    ## …and the design is read off the member that is already there.
    assert dialog.factors_widget.get_factors() == \
        DESIGN_GLOBAL["experimental_design_factors"]
    assert dialog.well_a_edit.text() == "S5"

    dialog._save()
    written = yaml.safe_load((root / "project.yaml").read_text())
    assert written["name"] == "OldStudy"
    assert written["design"]["global"]["well_names"] == {"A": "S5", "B": "S5Y5"}
    ## And the Project it just made loads, with the folder as its member.
    assert dialog.adopted == ["rep1"]
    assert "global" not in yaml.safe_load(
        (root / "rep1" / "flic_config.yaml").read_text())
    loaded = project_mod.Project(root)
    assert loaded.member_names == ["rep1"]
    ## Inheriting leaves the member with exactly what it had before.
    assert loaded.resolved_global("rep1")["well_names"] == \
        {"A": "S5", "B": "S5Y5"}


def test_initialize_refuses_a_folder_that_is_already_a_project(
        app, project, monkeypatch):
    said = _silence(monkeypatch)
    dialog = ProjectDesignDialog(start_dir=str(project),
                                 initialize_existing=True)
    dialog._save()

    assert dialog.saved_dir is None
    assert any("is a Project already" in text for text in said)


def test_initialize_refuses_a_missing_folder(app, tmp_path, monkeypatch):
    said: list[str] = []
    dialog = ProjectDesignDialog(start_dir=str(tmp_path / "nope"),
                                 initialize_existing=True)
    monkeypatch.setattr(dialog, "_warn", said.append)
    dialog._save()

    assert dialog.saved_dir is None
    assert any("Create Project" in text for text in said)


def test_the_design_editor_waits_for_a_project(app, tmp_path):
    """The two ways to *make* one are live with nothing open; the editor for
    the one that is open is not."""
    _write(tmp_path / "loose" / "keep.txt", {})
    window = AnalysisHubWindow(target=None)
    try:
        assert window.project is None
        assert not window.design_btn.isEnabled()
        assert not window.create_member_btn.isEnabled()
        assert not window.init_member_btn.isEnabled()
    finally:
        window.close()


# ---------------------------------------------------------------------------
# Into a Member: the same three, one level down
# ---------------------------------------------------------------------------

def test_initializable_dirs_sees_folders_that_are_not_members_yet(project):
    (project / "rep2").mkdir()                    # empty
    _recording(project / "rep3", dfm=3, filed=False)   # loose recording
    (project / "analysis").mkdir()                # an output directory

    names = [item.name for item in
             layout_mod.initializable_dirs(project)]
    assert names == ["rep2", "rep3"]
    ## members_in answers the narrower question and misses the empty folder.
    assert "rep2" not in [i.name for i in layout_mod.members_in(project)]


def test_create_member_scaffolds_from_the_design(hub, project, monkeypatch):
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("rep2", True)))
    finished: list[tuple] = []
    monkeypatch.setattr(hub, "_finish_new_member_config",
                        lambda *args: finished.append(args))
    hub._create_member()

    config = project / "rep2" / "flic_config.yaml"
    assert config.exists()
    assert (project / "rep2" / "data").is_dir()
    ## Inheriting, not repeating: the scaffold states no global: of its own.
    assert "global" not in yaml.safe_load(config.read_text())
    assert finished and finished[0][0] == "rep2"


def test_create_member_refuses_a_folder_that_already_exists(hub, project,
                                                           monkeypatch):
    (project / "rep2").mkdir()
    said = _silence(monkeypatch)
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("rep2", True)))
    hub._create_member()

    assert not (project / "rep2" / "flic_config.yaml").exists()
    assert any("Initialize existing directory" in text for text in said)


def test_create_member_refuses_a_name_that_escapes_the_project(project):
    loaded = project_mod.Project(project)
    with pytest.raises(ValueError):
        loaded.scaffold_member("../elsewhere")


def test_initialize_member_files_the_recording_before_configuring(
        hub, project, monkeypatch):
    _recording(project / "rep2", dfm=2, filed=False)
    hub._reload_project()
    monkeypatch.setattr(
        QInputDialog, "getItem",
        staticmethod(lambda *a, **k: (a[3][0] if len(a) > 3 else "", True)))
    opened: list[Path] = []
    monkeypatch.setattr(hub, "_open_config_editor_on", opened.append)
    hub._initialize_member_folder()

    ## Filing first: a recording left at the root would make the freshly
    ## configured member look empty to every analysis that follows.
    assert (project / "rep2" / "data" / "DFM2_0.csv").exists()
    assert not (project / "rep2" / "DFM2_0.csv").exists()
    assert (project / "rep2" / "flic_config.yaml").exists()
    assert opened and opened[0].parent.name == "rep2"
    ## The reconciliation used the data actually present: DFM 2 was added
    ## from data/, and the copied layout's DFM 1 is kept and flagged rather
    ## than dropped — a missing CSV is usually a copy that has not finished.
    config = yaml.safe_load((project / "rep2" / "flic_config.yaml").read_text())
    assert [node["id"] for node in config["dfms"]] == [1, 2]


def test_initialize_member_says_when_there_is_nothing_to_adopt(
        hub, monkeypatch):
    said = _silence(monkeypatch)
    hub._initialize_member_folder()

    assert any("Create member" in text for text in said)


# ---------------------------------------------------------------------------
# A copied config is checked before it is written
# ---------------------------------------------------------------------------

def test_a_conforming_config_is_accepted(project):
    loaded = project_mod.Project(project)
    assert loaded.design_problems_for(dict(MEMBER), "rep1") == []


def test_a_deviating_config_is_refused_with_reasons(project):
    loaded = project_mod.Project(project)
    problems = loaded.design_problems_for(
        {"global": {"params": {"feeding_threshold": 999}}, **MEMBER}, "other")

    assert problems and "feeding_threshold" in problems[0]


def test_a_config_that_is_not_a_config_is_refused(project):
    loaded = project_mod.Project(project)
    assert loaded.design_problems_for({"name": "not a config"}, "x")
    assert loaded.design_problems_for(["nope"], "x")


def test_an_illegal_per_dfm_override_is_refused(project):
    loaded = project_mod.Project(project)
    problems = loaded.design_problems_for(
        {"dfms": [{"id": 1, "params": {"feeding_threshold": 5},
                   "chambers": {1: "Ctrl"}}]}, "other")

    assert problems and "only" in problems[0]


def test_copy_refuses_the_members_own_config(hub, project, monkeypatch):
    said = _silence(monkeypatch)
    target = project / "rep1" / "flic_config.yaml"
    monkeypatch.setattr(
        "PyQt6.QtWidgets.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: (str(target), "")))

    assert hub._copy_member_config("rep1", project / "rep1") is False
    assert any("own config" in text for text in said)


def test_a_deviating_copy_is_not_written(hub, project, monkeypatch):
    said = _silence(monkeypatch)
    source = project.parent / "elsewhere.yaml"
    _write(source, {"global": {"params": {"feeding_threshold": 999}}, **MEMBER})
    hub._scaffold_member("rep2")
    hub._reload_project()
    scaffold = (project / "rep2" / "flic_config.yaml").read_text()
    monkeypatch.setattr(
        "PyQt6.QtWidgets.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: (str(source), "")))

    assert hub._copy_member_config("rep2", project / "rep2") is False
    ## The scaffold is still there, so the editor has something to open.
    assert (project / "rep2" / "flic_config.yaml").read_text() == scaffold
    assert any("does not fit" in text for text in said)


def test_a_conforming_copy_replaces_the_scaffold(hub, project, monkeypatch):
    _silence(monkeypatch)
    source = project / "rep1" / "flic_config.yaml"
    hub._scaffold_member("rep2")
    hub._reload_project()
    monkeypatch.setattr(
        "PyQt6.QtWidgets.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: (str(source), "")))

    assert hub._copy_member_config("rep2", project / "rep2") is True
    assert (project / "rep2" / "flic_config.yaml").read_text() == \
        source.read_text()
    assert project_mod.Project(project).member_names == ["rep1", "rep2"]


# ---------------------------------------------------------------------------
# The blocked row offers its own repair
# ---------------------------------------------------------------------------

def test_double_clicking_a_config_less_member_offers_to_make_one(
        hub, project, monkeypatch):
    _recording(project / "rep2", dfm=2)
    hub._reload_project()
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    opened: list[Path] = []
    monkeypatch.setattr(hub, "_open_config_editor_on", opened.append)

    blocked = {m.name: m for m in hub.project.blocked_members()}
    hub._offer_blocked_fix(blocked["rep2"])

    assert (project / "rep2" / "flic_config.yaml").exists()
    assert opened and opened[0].parent.name == "rep2"
