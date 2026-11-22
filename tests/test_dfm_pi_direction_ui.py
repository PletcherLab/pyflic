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


# ---------------------------------------------------------------------------
# Start from another recording takes each DFM's PI Direction (ADR-0028).
# ---------------------------------------------------------------------------

EARLIER = {
    "global": CONFIG["global"],
    "dfms": [{"id": 1, "chambers": {1: "Ctrl", 2: "Exp"}},
             {"id": 3, "params": {"pi_direction": "right", "feeding_threshold": 30},
              "chambers": {1: "Exp", 2: "Ctrl"}},
             {"id": 4, "params": {"pi_multiplier": 2}, "chambers": {1: "Exp"}}],
}


def _dfm_csv(folder: Path, dfm_id: int, n: int = 600) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    header = ",".join(["Sample", "Seconds"] + [f"W{w}" for w in range(1, 13)])
    rows = [",".join([str(i + 1), str(i / 5.0)] + ["0"] * 12) for i in range(n)]
    (folder / f"DFM{dfm_id}_0.csv").write_text("\n".join([header, *rows]) + "\n")


def _earlier(tmp_path: Path, config: dict = EARLIER) -> Path:
    path = tmp_path / "rep1" / "flic_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _questions(monkeypatch, answer=QMessageBox.StandardButton.Yes) -> list[str]:
    asked: list[str] = []

    def question(*args, **_kwargs):
        asked.append(" ".join(str(a) for a in args if isinstance(a, str)))
        return answer

    monkeypatch.setattr(QMessageBox, "question", staticmethod(question))
    return asked


def test_a_new_recording_takes_each_dfms_side_from_the_earlier_one(app, tmp_path, monkeypatch):
    folder = tmp_path / "rep2"
    for dfm_id in (1, 3, 4):
        _dfm_csv(folder / "data", dfm_id)
    asked = _questions(monkeypatch)
    editor = ExperimentEditor(folder)
    try:
        assert editor.start_from(_earlier(tmp_path))
        assert asked == []                       # nothing stated here is replaced
        assert [w.pi_direction() for w in editor._dfm_widgets] == [None, "right", "right"]
        assert editor._dfm_widgets[1].side_combo.currentData() == "right"
        ## Only the side: the earlier DFM's other overrides are its own.
        assert editor._dfm_widgets[1].get_dict()["params"] == {"pi_direction": "right"}
        assert "the PI Direction of DFM 3, 4" in editor.statusBar().currentMessage()
    finally:
        editor.close()


def test_a_side_stated_here_is_replaced_only_after_asking(app, tmp_path, monkeypatch):
    path = tmp_path / "rep2" / "flic_config.yaml"
    path.parent.mkdir(parents=True)
    mine = {**CONFIG, "dfms": [
        {"id": 1, "params": {"pi_direction": "left"}, "chambers": {1: "Ctrl", 2: "Exp"}},
        {"id": 3, "params": {"pi_direction": "left"}, "chambers": {1: "Exp", 2: "Ctrl"}}]}
    path.write_text(yaml.safe_dump(mine, sort_keys=False), encoding="utf-8")
    earlier = _earlier(tmp_path)

    asked = _questions(monkeypatch, QMessageBox.StandardButton.No)
    editor = ExperimentEditor(path)
    try:
        assert editor.start_from(earlier) is False
        assert [w.pi_direction() for w in editor._dfm_widgets] == ["left", "left"]
    finally:
        editor.close()
    assert "PI Direction: DFM 3 left → right" in asked[0]
    assert "DFM 1" not in asked[0].split("PI Direction:")[1]  # the earlier states none for 1

    _questions(monkeypatch)
    editor = ExperimentEditor(path)
    try:
        assert editor.start_from(earlier)
        assert [w.pi_direction() for w in editor._dfm_widgets] == ["left", "right"]
    finally:
        editor.close()


def test_a_one_well_recording_takes_no_side(app, tmp_path, monkeypatch):
    folder = tmp_path / "rep2"
    _dfm_csv(folder / "data", 1)
    earlier = _earlier(tmp_path, {
        "global": {"chamber_layout": "single_well", "params": {},
                   "experimental_design_factors": {"Genotype": ["Ctrl", "Exp"]}},
        "dfms": [{"id": 1, "params": {"pi_direction": "right"}, "chambers": {1: "Ctrl"}}]})
    _questions(monkeypatch)
    editor = ExperimentEditor(folder)
    try:
        assert editor.start_from(earlier)
        assert editor.chamber_layout() == "single_well"
        assert editor._dfm_widgets[0].pi_direction() is None
    finally:
        editor.close()


def test_a_new_config_with_one_dfms_side_still_loads(app, tmp_path):
    """The loader needs a params: section for every DFM, under global: or its
    own.  One DFM's side must not take the empty global one away from the rest."""
    from pyflic.base.yaml_config import load_experiment_yaml

    folder = tmp_path / "rep2"
    for dfm_id in (1, 2):
        _dfm_csv(folder / "data", dfm_id)
    editor = ExperimentEditor(folder)
    try:
        editor._dfm_widgets[1].set_pi_direction("right")
        assert editor.write(folder / "flic_config.yaml", confirm=False)
    finally:
        editor.close()
    saved = yaml.safe_load((folder / "flic_config.yaml").read_text(encoding="utf-8"))
    assert saved["global"]["params"] == {}
    assert saved["dfms"][1]["params"] == {"pi_direction": "right"}
    assert sorted(load_experiment_yaml(folder, eager=False, use_disk_cache=False).dfms) == [1, 2]
