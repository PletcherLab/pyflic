"""Each DFM's PI Direction in the Experiment Editor (2026-11-22).

Which side well A sits on is a fact of how a DFM was set up, so it may differ
from DFM to DFM, even inside a Project (``PHYSICAL_DFM_KEYS``).  The Experiment
Editor hid every per-DFM parameter, so a config made there could not say that
DFMs 3, 4, 7 and 8 had well A on the right.  Each two-well DFM page now offers
inherit, left or right, written as that DFM's ``params: {pi_direction: ...}``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

pytest.importorskip("PyQt6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from pyflic.base.config_editor import DFMWidget  # noqa: E402
from pyflic.base.experiment_editor import ExperimentEditor  # noqa: E402

CONFIG = {
    "global": {
        "experiment_type": "Hedonic",
        "well_names": {"A": "S5", "B": "S5Y5"},
        "experimental_design_factors": {"Genotype": ["Ctrl", "Exp"]},
    },
    "dfms": [
        {"id": 1, "chambers": {1: "Ctrl", 2: "Exp"}},
        {"id": 2, "params": {"pi_direction": "right", "feeding_threshold": 30},
         "chambers": {1: "Exp", 2: "Ctrl"}},
    ],
}


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def no_dialogs(monkeypatch):
    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "exp" / "flic_config.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(CONFIG, sort_keys=False), encoding="utf-8")
    return path


def _choose(widget: DFMWidget, side: str | None) -> None:
    combo = widget.side_combo
    combo.setCurrentIndex(0 if side is None else combo.findData(side))


def test_each_dfm_shows_its_own_pi_direction(app, tmp_path):
    editor = ExperimentEditor(_config(tmp_path))
    try:
        first, second = editor._dfm_widgets
        assert not first.side_card.isHidden()
        assert first.pi_direction() is None and first.side_combo.currentText() == "inherit"
        assert second.pi_direction() == "right"
        assert second.side_combo.currentData() == "right"
    finally:
        editor.close()


def test_a_dfms_pi_direction_is_saved_and_its_other_settings_kept(app, tmp_path):
    path = _config(tmp_path)
    editor = ExperimentEditor(path)
    try:
        first, second = editor._dfm_widgets
        _choose(first, "right")
        _choose(second, None)  # inherit: the experiment's side
        assert editor.write(path, confirm=False)
    finally:
        editor.close()
    dfms = yaml.safe_load(path.read_text(encoding="utf-8"))["dfms"]
    assert dfms[0]["params"] == {"pi_direction": "right"}
    assert dfms[1]["params"] == {"feeding_threshold": 30}  # a setting it shows no field for


def test_inherit_on_a_dfm_with_nothing_else_writes_no_params(app, tmp_path):
    path = _config(tmp_path)
    editor = ExperimentEditor(path)
    try:
        first, _second = editor._dfm_widgets
        _choose(first, "left")
        _choose(first, None)
        assert "params" not in first.get_dict()
    finally:
        editor.close()


def test_the_choice_shows_only_for_two_wells_and_only_in_the_experiment_editor(app):
    page = DFMWidget(1, chamber_size=1, simple=True)
    try:
        assert page.side_card.isHidden()       # one well: no side to choose
        page.update_chamber_size(2)
        assert not page.side_card.isHidden()
    finally:
        page.deleteLater()
    full = DFMWidget(1, chamber_size=2)
    try:
        assert full.side_card.isHidden()       # the Config Editor has its override form
    finally:
        full.deleteLater()
