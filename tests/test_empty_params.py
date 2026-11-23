"""An empty ``params: {}`` says nothing (ADR-0030).

The Experiment Editor once wrote one into every new config, only to get past a
loader check that is now gone.  Inside a Project it must neither count as a
member stating the Design's settings nor hide them when the member loads.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from pyflic.base import analysis_settings
from pyflic.base import project as project_mod
from pyflic.base.project import Project

DESIGN = {"global": {
    "experiment_type": "Hedonic",
    "params": {"feeding_threshold": 33, "samples_per_second": 5},
    "well_names": {"A": "S5", "B": "S5Y5"},
    "experimental_design_factors": {"Treatment": ["Ctrl", "Exp"]},
}}


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _project(tmp_path: Path, own_global: dict) -> Path:
    root = tmp_path / "proj"
    _write(root / "project.yaml", {"name": "proj", "design": DESIGN})
    _write(root / "rep1" / "flic_config.yaml",
           {"global": own_global,
            "dfms": [{"id": 1, "chambers": {1: "Ctrl", 2: "Exp"}}]})
    (root / "rep1" / "data").mkdir()
    (root / "rep1" / "data" / "DFM1_0.csv").write_text("Sample,Seconds\n")
    return root


def test_a_member_with_an_empty_params_loads_with_the_designs(tmp_path):
    described = {k: v for k, v in DESIGN["global"].items() if k != "params"}
    root = _project(tmp_path, {**described, "params": {}})
    project = Project(root)                      # no "global.params is {}" error
    assert project.resolved_global("rep1")["params"] == DESIGN["global"]["params"]
    states = project_mod.member_design_states(root)
    assert [(s.name, s.state) for s in states] == [("rep1", "in_step")]


def test_an_empty_params_does_not_hide_the_designs_settings():
    config = {"global": {"params": {}}}
    recorded = analysis_settings.settings(DESIGN["global"], config)
    assert recorded["global"]["params"]["feeding_threshold"] == 33
    assert analysis_settings.stated_global({"params": {}, "parameters": {},
                                            "well_names": {"A": "x"}}) == {
        "well_names": {"A": "x"}}
    assert not analysis_settings.says_nothing("params", {"feeding_threshold": 1})
    assert not analysis_settings.says_nothing("constants", {})


def test_a_member_stating_real_params_is_still_the_designs_to_settle(tmp_path):
    root = _project(tmp_path, {"params": {"feeding_threshold": 10}})
    states = project_mod.member_design_states(root)
    assert states[0].state == "write"
    assert "states params, which the design owns" in states[0].reasons


def test_the_experiment_editor_drops_an_empty_params_and_keeps_settings(tmp_path, monkeypatch):
    import os

    import pytest

    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication, QMessageBox

    from pyflic.base.experiment_editor import ExperimentEditor

    QApplication.instance() or QApplication([])
    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    path = tmp_path / "exp" / "flic_config.yaml"
    for params, kept in (({}, None), ({"feeding_threshold": 12}, {"feeding_threshold": 12})):
        _write(path, {"global": {"chamber_layout": "single_well", "params": params},
                      "dfms": [{"id": 1, "chambers": {1: "Ctrl"}}]})
        editor = ExperimentEditor(path)
        try:
            assert editor.write(path, confirm=False)
        finally:
            editor.close()
        saved = yaml.safe_load(path.read_text(encoding="utf-8"))["global"]
        assert saved.get("params") == kept        # a setting the Config Editor wrote stays
