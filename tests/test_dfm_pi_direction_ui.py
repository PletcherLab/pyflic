"""Each DFM's PI Direction in the Experiment Editor (2026-11-22).

Which side well A sits on is a fact of how a DFM was set up, so it may differ
from DFM to DFM, even inside a Project (``PHYSICAL_DFM_KEYS``).  The Experiment
Editor hid every per-DFM parameter, so a config made there could not say that
DFMs 3, 4, 7 and 8 had well A on the right.  Each two-well DFM page now offers
left or right, written as that DFM's ``params: {pi_direction: ...}``.  There is
no inherit (ADR-0029): every two-well DFM states its side, and one whose config
states none is given the side it is read with.
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


def _choose(widget: DFMWidget, side: str) -> None:
    combo = widget.side_combo
    combo.setCurrentIndex(combo.findData(side))


def test_each_dfm_shows_its_own_pi_direction(app, tmp_path):
    editor = ExperimentEditor(_config(tmp_path))
    try:
        first, second = editor._dfm_widgets
        assert not first.side_card.isHidden()
        assert [first.side_combo.itemText(i)
                for i in range(first.side_combo.count())] == ["left", "right"]  # no inherit
        assert second.pi_direction() == "right" and second.side_stated()
        assert second.side_combo.currentData() == "right"
        ## DFM 1 states none: it is given the side it is read with, and says so.
        assert first.pi_direction() == "left" and not first.side_stated()
        assert first.side_combo.currentData() == "left"
    finally:
        editor.close()


def test_a_dfms_pi_direction_is_saved_and_its_other_settings_kept(app, tmp_path):
    path = _config(tmp_path)
    editor = ExperimentEditor(path)
    try:
        first, second = editor._dfm_widgets
        _choose(first, "right")
        _choose(second, "left")
        assert first.side_stated() and second.side_stated()
        assert editor.write(path, confirm=False)
    finally:
        editor.close()
    dfms = yaml.safe_load(path.read_text(encoding="utf-8"))["dfms"]
    assert dfms[0]["params"] == {"pi_direction": "right"}
    assert dfms[1]["params"] == {"pi_direction": "left",
                                 "feeding_threshold": 30}  # a setting it shows no field for


def test_every_two_well_dfm_saves_a_side(app, tmp_path):
    """No inherit (ADR-0029): a DFM whose config states no side is saved with
    the one it is read with, so the file says every DFM's side."""
    path = _config(tmp_path)
    editor = ExperimentEditor(path)
    try:
        assert editor.write(path, confirm=False)
    finally:
        editor.close()
    dfms = yaml.safe_load(path.read_text(encoding="utf-8"))["dfms"]
    assert [d["params"]["pi_direction"] for d in dfms] == ["left", "right"]


def test_a_dfm_that_states_none_is_given_the_experiments_side(app, tmp_path):
    path = tmp_path / "exp" / "flic_config.yaml"
    path.parent.mkdir(parents=True)
    config = {**CONFIG, "global": {**CONFIG["global"], "params": {"pi_direction": "right"}}}
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    editor = ExperimentEditor(path)
    try:
        assert [w.pi_direction() for w in editor._dfm_widgets] == ["right", "right"]
        assert editor._dfm_widgets[0].side_combo.currentData() == "right"
        added = editor._add_dfm(5)
        assert added.pi_direction() == "right" and not added.side_stated()
    finally:
        editor.close()


def test_a_legacy_pi_multiplier_is_read_as_the_dfms_side(app, tmp_path):
    path = tmp_path / "exp" / "flic_config.yaml"
    path.parent.mkdir(parents=True)
    config = {**CONFIG, "dfms": [
        {"id": 1, "params": {"pi_multiplier": 2}, "chambers": {1: "Ctrl", 2: "Exp"}},
        {"id": 2, "params": {"pi_multiplier": 1}, "chambers": {1: "Exp", 2: "Ctrl"}}]}
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    editor = ExperimentEditor(path)
    try:
        assert [w.pi_direction() for w in editor._dfm_widgets] == ["right", "left"]
        assert all(w.side_stated() for w in editor._dfm_widgets)
        assert editor.write(path, confirm=False)
    finally:
        editor.close()
    dfms = yaml.safe_load(path.read_text(encoding="utf-8"))["dfms"]
    assert [d["params"] for d in dfms] == [{"pi_direction": "right"}, {"pi_direction": "left"}]
    ## The Config Editor's DFM page reads it the same way.
    page = DFMWidget(1, chamber_size=2)
    try:
        page.load_dict(config["dfms"][0], 2)
        assert page.get_dict()["params"] == {"pi_direction": "right"}
    finally:
        page.deleteLater()


def test_the_choice_shows_only_for_two_wells_and_only_in_the_experiment_editor(app):
    page = DFMWidget(1, chamber_size=1, simple=True)
    try:
        assert page.side_card.isHidden()       # one well: no side to choose
        assert page.pi_direction() is None
        page.update_chamber_size(2)
        assert not page.side_card.isHidden()
        assert page.pi_direction() == "left"   # two wells: a side, always
        _choose(page, "right")
        page.update_chamber_size(1)
        assert page.pi_direction() is None     # one-well chambers have none
        assert "params" not in page.get_dict()
    finally:
        page.deleteLater()
    full = DFMWidget(1, chamber_size=2)
    try:
        assert full.side_card.isHidden()       # the Config Editor has its override form
        assert full.pi_direction() is None     # ...in which a DFM may state none
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
        ## A side filled in with the experiment's is not one stated here.
        assert asked == []                       # nothing stated here is replaced
        assert [w.pi_direction() for w in editor._dfm_widgets] == ["left", "right", "right"]
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


def test_the_earlier_recordings_global_side_is_taken(app, tmp_path, monkeypatch):
    """A side the other recording states in its global: is the side of each
    DFM it lists with none of its own, as the loader reads it (ADR-0029)."""
    folder = tmp_path / "rep2"
    for dfm_id in (1, 3, 5):
        _dfm_csv(folder / "data", dfm_id)
    earlier = _earlier(tmp_path, {
        "global": {**CONFIG["global"], "params": {"pi_direction": "right"}},
        "dfms": [{"id": 1, "chambers": {1: "Ctrl", 2: "Exp"}},
                 {"id": 3, "params": {"pi_direction": "left"},
                  "chambers": {1: "Exp", 2: "Ctrl"}}]})
    _questions(monkeypatch)
    editor = ExperimentEditor(folder)
    try:
        assert editor.start_from(earlier)
        ## DFM 5 is not the other recording's: it keeps the side it was given.
        assert [w.pi_direction() for w in editor._dfm_widgets] == ["right", "left", "left"]
    finally:
        editor.close()


def test_a_new_config_states_every_side_and_loads(app, tmp_path):
    """Every DFM has its own params: (its side), so the empty global one is
    not needed; the loader takes the config and reads each DFM's side."""
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
    assert [d["params"] for d in saved["dfms"]] == [{"pi_direction": "left"},
                                                    {"pi_direction": "right"}]
    dfms = load_experiment_yaml(folder, eager=False, use_disk_cache=False).dfms
    assert {i: dfms[i].params.pi_direction for i in sorted(dfms)} == {1: "left", 2: "right"}
