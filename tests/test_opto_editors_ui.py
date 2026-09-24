"""The optogenetics setting and the light QC's constants in the two editors.

Headless: the widgets are driven directly, and what is asserted is what a save
writes.  The same contract as every other design constant (ADR-0011): defaults
are placeholders, only a value typed or picked reaches the yaml, and a Member
shows its Design's values locked — except the per-DFM setting, which describes
the recording and stays free inside a Project.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

pytest.importorskip("PyQt6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from pyflic.base.config_editor import FLICConfigEditor  # noqa: E402
from pyflic.base.design_editor import ProjectDesignDialog  # noqa: E402

GOOD_PROGRAM = (
    "Start Time: 01/01/2025 08:00:00\n***DFM 1***\n"
    "(01/01/2025 08:00:00) Dark Off,20,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"
    "F:40,P:8,D:1000,L:0,M:0,60.0min.\n")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _cfg(**global_extra) -> dict:
    g = {"chamber_layout": "single_well", "params": {"feeding_threshold": 20}}
    g.update(global_extra)
    return {"global": g, "dfms": [{"id": 1, "chambers": {1: "Ctrl"}}]}


def _open(tmp_path: Path, cfg: dict, *, program: bool = False) -> FLICConfigEditor:
    (tmp_path / "data").mkdir(exist_ok=True)
    if program:
        (tmp_path / "data" / "Program.txt").write_text(GOOD_PROGRAM, encoding="utf-8")
    (tmp_path / "flic_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False),
                                               encoding="utf-8")
    return FLICConfigEditor(initial_path=tmp_path)


def _shown(win: FLICConfigEditor) -> bool:
    return win._exp_form.isRowVisible(win._opto_header_row)


def test_the_setting_is_written_only_when_it_is_not_auto(app, tmp_path):
    win = _open(tmp_path, _cfg())
    try:
        assert win._opto_combo.currentData() == "auto"
        assert "optogenetics" not in win._collect_yaml()["global"]
        win._opto_combo.setCurrentIndex(1)
        assert win._collect_yaml()["global"]["optogenetics"] is True
        win._opto_combo.setCurrentIndex(2)
        assert win._collect_yaml()["global"]["optogenetics"] is False
    finally:
        win.close()


def test_the_setting_round_trips(app, tmp_path):
    win = _open(tmp_path, _cfg(optogenetics=False))
    try:
        assert win._opto_combo.currentData() is False
        assert win._collect_yaml()["global"]["optogenetics"] is False
    finally:
        win.close()


def test_the_constants_show_once_the_experiment_is_optogenetic(app, tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    win = _open(plain, _cfg())
    try:
        assert not _shown(win)                        # auto, no Program.txt, no run
        win._opto_combo.setCurrentIndex(1)            # yes
        assert _shown(win)
        widgets = win._opto_constant_widgets
        assert "0.3" in widgets["opto_unexplained_fail_fraction"].placeholderText()
        assert widgets["exclude_failed_opto_chambers"].itemText(0) == \
            "default: keep and flag"
        win._opto_combo.setCurrentIndex(2)            # no
        assert not _shown(win)
    finally:
        win.close()
    other = tmp_path / "with_program"
    other.mkdir()
    win = _open(other, _cfg(), program=True)
    try:
        assert _shown(win)                            # auto, and a Program.txt
    finally:
        win.close()


def test_the_constants_are_written_only_when_set(app, tmp_path):
    win = _open(tmp_path, _cfg(optogenetics=True))
    try:
        w = win._opto_constant_widgets
        assert "constants" not in win._collect_yaml()["global"]
        w["opto_unexplained_fail_fraction"].setText("0.4")
        w["exclude_failed_opto_chambers"].setCurrentIndex(1)
        assert win._collect_yaml()["global"]["constants"] == {
            "opto_unexplained_fail_fraction": 0.4, "exclude_failed_opto_chambers": True}
        w["opto_unexplained_warn_fraction"].setText("0.6")
        experiment, _dfms = win._problems()
        assert any("must not exceed" in p for p in experiment)
    finally:
        win.close()


def test_a_hidden_section_keeps_what_the_file_says(app, tmp_path):
    win = _open(tmp_path, _cfg(optogenetics=False,
                               constants={"opto_default_decay_ms": 500}))
    try:
        assert not _shown(win)
        assert win._collect_yaml()["global"]["constants"] == {"opto_default_decay_ms": 500}
    finally:
        win.close()


def test_each_dfm_can_override_the_setting(app, tmp_path):
    cfg = _cfg()
    cfg["dfms"] = [{"id": 1, "optogenetics": False, "chambers": {1: "Ctrl"}},
                   {"id": 2, "chambers": {1: "Ctrl"}}]
    win = _open(tmp_path, cfg)
    try:
        first, second = win._dfm_widgets
        assert first.optogenetics() is False and second.optogenetics() is None
        dfms = win._collect_yaml()["dfms"]
        assert dfms[0]["optogenetics"] is False and "optogenetics" not in dfms[1]
        second._opto_combo.setCurrentIndex(2)         # yes
        assert win._collect_yaml()["dfms"][1]["optogenetics"] is True
    finally:
        win.close()


def test_a_member_shows_the_designs_setting_locked_but_its_dfms_free(app, tmp_path):
    (tmp_path / "project.yaml").write_text(yaml.safe_dump(
        {"design": {"global": {"chamber_layout": "single_well", "optogenetics": True,
                               "constants": {"opto_unexplained_min_sec": 60}}}}),
        encoding="utf-8")
    member = tmp_path / "memberA"
    (member / "data").mkdir(parents=True)
    (member / "data" / "DFM1_0.csv").write_text("x", encoding="utf-8")
    (member / "flic_config.yaml").write_text(yaml.safe_dump(
        {"dfms": [{"id": 1, "optogenetics": False, "chambers": {1: "Ctrl"}}]}),
        encoding="utf-8")
    win = FLICConfigEditor(initial_path=member)
    try:
        assert win._opto_combo.currentData() is True
        assert not win._opto_combo.isEnabled()
        assert "DFM tab" in win._opto_combo.toolTip()
        widget = win._opto_constant_widgets["opto_unexplained_min_sec"]
        assert widget.text() == "60" and not widget.isEnabled()
        assert win._dfm_widgets[0]._opto_combo.isEnabled()
        saved = win._collect_yaml()
        assert "global" not in saved and saved["dfms"][0]["optogenetics"] is False
    finally:
        win.close()


def test_the_design_dialog_round_trips_the_setting_and_constants(app, tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "project.yaml").write_text(yaml.safe_dump({"name": "P", "design": {"global": {
        "chamber_layout": "two_well", "optogenetics": True,
        "constants": {"opto_unexplained_fail_fraction": 0.5, "my_own_note": 1}}}}),
        encoding="utf-8")
    dialog = ProjectDesignDialog(start_dir=str(root))
    assert dialog.opto_combo.currentData() is True
    assert not dialog.opto_group.isHidden()
    assert dialog.opto_number_edits["opto_unexplained_fail_fraction"].text() == "0.5"
    assert "0.1" in dialog.opto_number_edits["opto_unexplained_warn_fraction"] \
        .placeholderText()
    built = dialog._build_design()["global"]
    assert built["optogenetics"] is True
    assert built["constants"]["opto_unexplained_fail_fraction"] == 0.5
    assert built["constants"]["my_own_note"] == 1
    assert "exclude_failed_opto_chambers" not in built["constants"]    # the default
    dialog.opto_switch_checks["exclude_failed_opto_chambers"].setChecked(True)
    assert dialog._build_design()["global"]["constants"][
        "exclude_failed_opto_chambers"] is True
    dialog.opto_combo.setCurrentIndex(dialog.opto_combo.findData(False))
    assert dialog.opto_group.isHidden()
    built = dialog._build_design()["global"]
    assert built["optogenetics"] is False
    ## Hidden, the section keeps the file's values rather than writing its own.
    assert built["constants"]["opto_unexplained_fail_fraction"] == 0.5
    dialog.opto_combo.setCurrentIndex(0)
    assert "optogenetics" not in dialog._build_design()["global"]
