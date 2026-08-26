"""The Config Editor's write contract and its two destructive-edit guards.

Headless: every test drives the widgets directly.  The point is what reaches
disk and what the editor refuses to do quietly — a typed config that writes no
``chamber_size`` (ADR-0007/ADR-0011), a partial factor row that is never
compacted into a wrong one, and a shrink that says what it is about to throw
away.  Not pixels.

The editor gained tabs and lost its "Number of DFMs" spinner in the same
change; the tests address the widgets, not the layout, so a further rearrange
does not break them.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

pytest.importorskip("PyQt6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QTableWidgetItem  # noqa: E402

from pyflic.base.config_editor import FLICConfigEditor  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def editor(app, tmp_path):
    """An editor with nothing to auto-load — a blank Custom Experiment."""
    win = FLICConfigEditor(initial_path=tmp_path)
    yield win
    win.close()


def _open(app, tmp_path, cfg: dict) -> FLICConfigEditor:
    """An editor with *cfg* written to disk and auto-loaded."""
    (tmp_path / "flic_config.yaml").write_text(
        yaml.dump(cfg, sort_keys=False), encoding="utf-8")
    return FLICConfigEditor(initial_path=tmp_path)


def _select_type(win: FLICConfigEditor, name: str) -> None:
    win._experiment_type_combo.setCurrentIndex(
        win._experiment_type_combo.findData(name))


def _select_layout(win: FLICConfigEditor, layout: str) -> None:
    win._chamber_layout_combo.setCurrentIndex(
        win._chamber_layout_combo.findData(layout))


def _allow_discard(win: FLICConfigEditor, allow: bool) -> list:
    """Answer the discard dialog without showing it; record what it asked."""
    asked: list = []

    def _fake(action, items):
        asked.append((action, list(items)))
        return allow

    win._confirm_discard = _fake
    return asked


# ---------------------------------------------------------------------------
# The type menu
# ---------------------------------------------------------------------------

def test_type_menu_offers_the_registry_and_not_the_retired_layouts(editor):
    """``two_well`` and ``single_well`` stopped being types (ADR-0007).

    They were still in this menu long after the loader started rejecting them,
    so the editor wrote configs its own linter reported.
    """
    names = {editor._experiment_type_combo.itemData(i)
             for i in range(editor._experiment_type_combo.count())}
    assert names == {"Custom", "Hedonic", "ProgressiveRatio"}
    assert "two_well" not in names and "single_well" not in names


def test_custom_experiment_is_named_not_spelled_auto(editor):
    """A config with no ``experiment_type`` IS a Custom Experiment, not a
    request for the editor to guess one."""
    labels = [editor._experiment_type_combo.itemText(i)
              for i in range(editor._experiment_type_combo.count())]
    assert "Custom Experiment" in labels
    assert not any(label.startswith("(auto") for label in labels)


# ---------------------------------------------------------------------------
# The write contract (ADR-0007 / ADR-0011)
# ---------------------------------------------------------------------------

def test_custom_experiment_writes_the_layout_and_no_chamber_size(editor):
    _select_type(editor, "Custom")
    _select_layout(editor, "single_well")
    global_cfg = editor._collect_yaml()["global"]

    assert global_cfg["chamber_layout"] == "single_well"
    assert "chamber_size" not in global_cfg["params"]
    assert "experiment_type" not in global_cfg


def test_typed_config_writes_neither_layout_nor_chamber_size(editor):
    _select_type(editor, "Hedonic")
    global_cfg = editor._collect_yaml()["global"]

    assert global_cfg["experiment_type"] == "Hedonic"
    assert "chamber_layout" not in global_cfg
    assert "chamber_size" not in global_cfg["params"]


def test_what_the_editor_writes_is_what_the_type_accepts(editor):
    """The editor validates with the loader's own function, so a config it
    considers clean cannot be one the loader rejects."""
    _select_type(editor, "Hedonic")
    editor._well_a_edit.setText("Sucrose")
    editor._well_b_edit.setText("Yeast")

    assert editor._current_type().validate(editor._collect_yaml()["global"]) == []


def test_a_hedonic_config_without_well_names_is_reported(editor):
    _select_type(editor, "Hedonic")
    experiment, _ = editor._problems()

    assert any("well_names" in p for p in experiment)
    assert "⚠" in editor._tabs.tabText(0)

    editor._well_a_edit.setText("Sucrose")
    editor._well_b_edit.setText("Yeast")
    assert editor._problems()[0] == []
    assert "⚠" not in editor._tabs.tabText(0)


# ---------------------------------------------------------------------------
# Thresholds inherit rather than materialise
# ---------------------------------------------------------------------------

def test_blank_thresholds_inherit_and_are_not_written(editor):
    """``resolve_constants`` merges the type's defaults *under* the yaml, so a
    blank field means "use the type's", not "no filtering".  Writing the number
    out would freeze this config against today's defaults."""
    _select_type(editor, "Hedonic")

    assert "20" in editor._min_raw_licks_edit.placeholderText()
    assert "150000" in editor._max_events_edit.placeholderText()
    assert "constants" not in editor._collect_yaml()["global"]


def test_a_typed_threshold_is_written_as_an_override(editor):
    _select_type(editor, "Hedonic")
    editor._min_raw_licks_edit.setText("35")

    constants = editor._collect_yaml()["global"]["constants"]
    assert constants == {"min_untransformed_licks_cutoff": 35.0}


# ---------------------------------------------------------------------------
# Reading what earlier versions wrote
# ---------------------------------------------------------------------------

def test_a_pre_adr0007_config_still_opens_and_migrates_on_save(app, tmp_path):
    """``experiment_type: two_well`` is a Custom Experiment with that layout.

    The read path stays forgiving where the write path is strict — otherwise
    the only way to migrate a config would be to retype it.
    """
    win = _open(app, tmp_path, {
        "global": {"experiment_type": "two_well",
                   "params": {"chamber_size": 2, "feeding_threshold": 25}},
        "dfms": [{"id": 1, "chambers": {1: "Ctrl"}}],
    })
    try:
        assert win._experiment_type_combo.currentData() == "Custom"
        assert win._chamber_layout() == "two_well"
        assert win._global_params.get_values()["feeding_threshold"] == 25

        global_cfg = win._collect_yaml()["global"]
        assert global_cfg["chamber_layout"] == "two_well"
        assert "experiment_type" not in global_cfg
        assert "chamber_size" not in global_cfg["params"]
    finally:
        win.close()


def test_a_legacy_chamber_size_alone_still_sizes_the_chamber_tables(app, tmp_path):
    win = _open(app, tmp_path, {
        "global": {"params": {"chamber_size": 1}},
        "dfms": [{"id": 1, "chambers": {1: "Ctrl", 12: "Exp"}}],
    })
    try:
        assert win._chamber_layout() == "single_well"
        assert win._dfm_widgets[0]._chamber_table.rowCount() == 12
        assert win._collect_yaml()["dfms"][0]["chambers"] == {1: "Ctrl", 12: "Exp"}
    finally:
        win.close()


def test_keys_the_editor_knows_nothing_about_survive_a_save(app, tmp_path):
    win = _open(app, tmp_path, {
        "global": {"chamber_layout": "two_well"},
        "scripts": {"analyze": ["load", "plot_licks"]},
        "dfms": [{"id": 1, "chambers": {1: "Ctrl"}}],
    })
    try:
        assert win._collect_yaml()["scripts"] == {"analyze": ["load", "plot_licks"]}
    finally:
        win.close()


def test_a_factor_round_trip_keeps_the_declaration_order(app, tmp_path):
    """Assignments are positional, so a reordered declaration is a rewritten
    design, not a cosmetic difference."""
    win = _open(app, tmp_path, {
        "global": {"chamber_layout": "two_well",
                   "experimental_design_factors": {
                       "paired": ["Paired", "Unpaired"],
                       "genotype": ["Chrim", "WCS"]}},
        "dfms": [{"id": 1, "chambers": {1: "Paired, Chrim", 2: "Unpaired, WCS"}}],
    })
    try:
        out = win._collect_yaml()["global"]
        assert list(out["experimental_design_factors"]) == ["paired", "genotype"]
        assert win._collect_yaml()["dfms"][0]["chambers"] == {
            1: "Paired, Chrim", 2: "Unpaired, WCS"}
    finally:
        win.close()


# ---------------------------------------------------------------------------
# Partial factor rows
# ---------------------------------------------------------------------------

def test_a_partial_factor_row_is_reported_and_never_compacted(app, tmp_path):
    """The old join-the-non-empty rule turned ``(blank, Chrim)`` into
    ``"Chrim"``, which reads back as ``paired=Chrim``: the experimental design
    silently rewritten."""
    win = _open(app, tmp_path, {
        "global": {"chamber_layout": "two_well",
                   "experimental_design_factors": {
                       "paired": ["Paired", "Unpaired"],
                       "genotype": ["Chrim", "WCS"]}},
        "dfms": [{"id": 1, "chambers": {1: "Paired, Chrim"}}],
    })
    try:
        table = win._dfm_widgets[0]._chamber_table
        table.setItem(2, 2, QTableWidgetItem("Chrim"))   # chamber 3, genotype only

        _, dfm_problems = win._problems()
        assert dfm_problems == ["DFM 1 chamber 3: no level for paired"]
        assert "⚠" in win._tabs.tabText(1)

        chambers = win._collect_yaml()["dfms"][0]["chambers"]
        assert 3 not in chambers
        assert chambers == {1: "Paired, Chrim"}
    finally:
        win.close()


def test_a_level_the_factor_does_not_declare_is_reported(app, tmp_path):
    win = _open(app, tmp_path, {
        "global": {"chamber_layout": "two_well",
                   "experimental_design_factors": {"genotype": ["Chrim", "WCS"]}},
        "dfms": [{"id": 1, "chambers": {}}],
    })
    try:
        win._dfm_widgets[0]._chamber_table.setItem(0, 1, QTableWidgetItem("Nonsense"))
        _, dfm_problems = win._problems()
        assert dfm_problems == [
            "DFM 1 chamber 1: 'Nonsense' is not a level of 'genotype'"]
    finally:
        win.close()


def test_a_wholly_blank_row_omits_its_chamber_without_complaint(app, tmp_path):
    win = _open(app, tmp_path, {
        "global": {"chamber_layout": "two_well",
                   "experimental_design_factors": {
                       "paired": ["Paired", "Unpaired"],
                       "genotype": ["Chrim", "WCS"]}},
        "dfms": [{"id": 1, "chambers": {1: "Paired, Chrim"}}],
    })
    try:
        assert win._problems()[1] == []
        assert win._collect_yaml()["dfms"][0]["chambers"] == {1: "Paired, Chrim"}
    finally:
        win.close()


# ---------------------------------------------------------------------------
# The destructive-edit guard
# ---------------------------------------------------------------------------

def test_shrinking_an_empty_plate_asks_nothing(editor):
    """Building a fresh config must never be interrupted — there is nothing
    to lose until a cell holds something."""
    asked = _allow_discard(editor, False)
    _select_layout(editor, "single_well")
    _select_layout(editor, "two_well")

    assert asked == []
    assert editor._chamber_layout() == "two_well"


def test_shrinking_over_an_assignment_asks_and_can_be_declined(editor):
    asked = _allow_discard(editor, False)
    _select_layout(editor, "single_well")
    editor._dfm_widgets[0]._chamber_table.setItem(8, 1, QTableWidgetItem("Sucrose"))

    _select_layout(editor, "two_well")

    assert asked and asked[-1][1] == ["DFM 1 chamber 9: Sucrose"]
    assert editor._chamber_layout() == "single_well"
    assert editor._chamber_layout_combo.currentData() == "single_well"


def test_accepting_the_discard_actually_shrinks(editor):
    _allow_discard(editor, False)
    _select_layout(editor, "single_well")
    editor._dfm_widgets[0]._chamber_table.setItem(8, 1, QTableWidgetItem("Sucrose"))

    _allow_discard(editor, True)
    _select_layout(editor, "two_well")

    assert editor._chamber_layout() == "two_well"
    assert editor._dfm_widgets[0]._chamber_table.rowCount() == 6


def test_declining_the_layout_change_undoes_the_type_change_too(editor):
    """Anything else leaves a Hedonic experiment on a single-well plate."""
    _allow_discard(editor, False)
    _select_layout(editor, "single_well")
    editor._dfm_widgets[0]._chamber_table.setItem(10, 1, QTableWidgetItem("Yeast"))

    _select_type(editor, "Hedonic")

    assert editor._experiment_type_combo.currentData() == "Custom"
    assert editor._chamber_layout() == "single_well"


# ---------------------------------------------------------------------------
# The Experiment Type owns the Chamber Layout
# ---------------------------------------------------------------------------

def test_a_typed_experiment_shows_its_layout_but_refuses_to_let_you_change_it(editor):
    """Shown rather than hidden: the layout is what decides whether the other
    tab has six chambers or twelve."""
    _select_type(editor, "Hedonic")

    assert editor._chamber_layout() == "two_well"
    assert not editor._chamber_layout_combo.isEnabled()
    assert "Hedonic Feeding" in editor._layout_hint.text()

    _select_type(editor, "Custom")
    assert editor._chamber_layout_combo.isEnabled()
    assert not editor._layout_hint.isVisible()


def test_the_layout_drives_the_chamber_count_on_every_dfm(editor):
    editor._add_dfm()
    _select_layout(editor, "single_well")
    assert all(w._chamber_table.rowCount() == 12 for w in editor._dfm_widgets)

    _select_layout(editor, "two_well")
    assert all(w._chamber_table.rowCount() == 6 for w in editor._dfm_widgets)


# ---------------------------------------------------------------------------
# Adding and removing DFMs
# ---------------------------------------------------------------------------

def test_remove_takes_the_selected_dfm_not_the_last_one(editor):
    """The count spinner this replaced always dropped the trailing tab, which
    with freely-editable ids was rarely the DFM anybody meant."""
    editor._add_dfm()
    editor._add_dfm()
    assert [w._id_spin.value() for w in editor._dfm_widgets] == [1, 2, 3]

    editor._dfm_tabs.setCurrentIndex(1)
    editor._remove_dfm()

    assert [w._id_spin.value() for w in editor._dfm_widgets] == [1, 3]
    assert [editor._dfm_tabs.tabText(i) for i in range(editor._dfm_tabs.count())] \
        == ["DFM 1", "DFM 3"]


def test_removing_a_dfm_with_assignments_asks_first(editor):
    editor._add_dfm()
    editor._dfm_tabs.setCurrentIndex(1)
    editor._dfm_widgets[1]._chamber_table.setItem(0, 1, QTableWidgetItem("Ctrl"))

    asked = _allow_discard(editor, False)
    editor._remove_dfm()

    assert asked[-1][0] == "Removing DFM 2"
    assert asked[-1][1] == ["chamber 1: Ctrl"]
    assert len(editor._dfm_widgets) == 2


def test_an_experiment_keeps_at_least_one_dfm(editor):
    assert not editor._btn_remove_dfm.isEnabled()
    editor._remove_dfm()
    assert len(editor._dfm_widgets) == 1

    editor._add_dfm()
    assert editor._btn_remove_dfm.isEnabled()


# ---------------------------------------------------------------------------
# A Member of a Project
# ---------------------------------------------------------------------------

def _project_with_member(tmp_path, design: dict, member_cfg: dict) -> Path:
    (tmp_path / "project.yaml").write_text(
        yaml.dump({"design": {"global": design}}, sort_keys=False), encoding="utf-8")
    member = tmp_path / "memberA"
    (member / "data").mkdir(parents=True)
    (member / "data" / "DFM1_1.csv").write_text("x", encoding="utf-8")
    (member / "flic_config.yaml").write_text(
        yaml.dump(member_cfg, sort_keys=False), encoding="utf-8")
    return member


def test_a_member_shows_the_design_and_refuses_to_edit_it(app, tmp_path):
    """A Member that states anything different from its Design fails to load
    (ADR-0005), so the editor shows what is in force and locks it."""
    member = _project_with_member(
        tmp_path,
        {"experiment_type": "Hedonic",
         "well_names": {"A": "Sucrose", "B": "Yeast"},
         "params": {"feeding_threshold": 22}},
        {"dfms": [{"id": 1, "chambers": {1: "Ctrl"}}], "scripts": {"analyze": ["load"]}})
    win = FLICConfigEditor(initial_path=member)
    try:
        assert win._experiment_type_combo.currentData() == "Hedonic"
        assert win._chamber_layout() == "two_well"
        assert win._global_params.get_values()["feeding_threshold"] == 22
        assert not win._experiment_type_combo.isEnabled()
        assert not win._chamber_layout_combo.isEnabled()

        out = win._collect_yaml()
        assert "global" not in out
        assert out["scripts"] == {"analyze": ["load"]}
    finally:
        win.close()


def test_a_custom_design_still_locks_the_layout_for_its_members(app, tmp_path):
    """The Design's claim survives the Experiment Type having none of its own —
    the two authorities compose rather than one clearing the other."""
    member = _project_with_member(
        tmp_path,
        {"chamber_layout": "single_well"},
        {"dfms": [{"id": 1, "chambers": {1: "Ctrl"}}]})
    win = FLICConfigEditor(initial_path=member)
    try:
        assert win._chamber_layout() == "single_well"
        assert not win._chamber_layout_combo.isEnabled()
    finally:
        win.close()


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------

def _rendered(label: str) -> str:
    """A tab label as Qt draws it — a doubled ampersand is one on screen."""
    return label.replace("&&", "&")


def test_the_editor_opens_on_the_experiment_tab(editor):
    """Type decides layout decides chamber count — reading order follows the
    dependency order."""
    assert editor._tabs.count() == 2
    assert _rendered(editor._tabs.tabText(0)).startswith("Experiment")
    assert _rendered(editor._tabs.tabText(1)).startswith("DFMs & Chambers")
    assert editor._tabs.currentIndex() == 0


def test_the_dfm_tab_label_escapes_its_ampersand(editor):
    """Qt reads a lone '&' as a mnemonic marker and swallows it, which drew
    the tab as "DFMs_Chambers"."""
    assert "&&" in editor._tabs.tabText(1)
    assert _rendered(editor._tabs.tabText(1)) == "DFMs & Chambers"


def test_new_returns_to_a_blank_custom_experiment(editor):
    _select_type(editor, "Hedonic")
    editor._well_a_edit.setText("Sucrose")
    editor._add_dfm()

    editor._new()

    assert editor._experiment_type_combo.currentData() == "Custom"
    assert editor._chamber_layout() == "two_well"
    assert editor._well_a_edit.text() == ""
    assert len(editor._dfm_widgets) == 1
    assert editor._tabs.currentIndex() == 0
