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

from PyQt6.QtWidgets import (  # noqa: E402
    QApplication,
    QComboBox,
    QLineEdit,
    QStyleOptionViewItem,
    QTableWidgetItem,
)

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


def test_a_configless_directory_preloads_the_dfms_its_data_names(app, tmp_path):
    """Opening the editor on a recording with no config yet — the initialize
    case — preloads one DFM tab per id found in data/, instead of a blank
    editor whose ids have to be retyped from the filenames."""
    (tmp_path / "data").mkdir()
    for name in ("DFM3_0.csv", "DFM3_1.csv", "DFM7_0.csv"):
        (tmp_path / "data" / name).write_text("x")
    win = FLICConfigEditor(initial_path=tmp_path)
    try:
        assert [w._id_spin.value() for w in win._dfm_widgets] == [3, 7]
        ## Save lands in the directory the DFMs came from.
        assert win._current_path == tmp_path / "flic_config.yaml"
    finally:
        win.close()


def test_loose_recordings_at_the_root_preload_too(app, tmp_path):
    """A recording not yet filed into data/ still names its DFMs."""
    (tmp_path / "DFM2_0.csv").write_text("x")
    win = FLICConfigEditor(initial_path=tmp_path)
    try:
        assert [w._id_spin.value() for w in win._dfm_widgets] == [2]
    finally:
        win.close()


def test_a_config_listing_no_dfms_preloads_them_from_the_data(app, tmp_path):
    """A scaffolded-but-empty config beside real data must not open as a
    lone default DFM 1 — the data already says which DFMs exist."""
    (tmp_path / "data").mkdir()
    for name in ("DFM4_0.csv", "DFM6_0.csv"):
        (tmp_path / "data" / name).write_text("x")
    (tmp_path / "flic_config.yaml").write_text("dfms: []\n")
    win = FLICConfigEditor(initial_path=tmp_path)
    try:
        assert [w._id_spin.value() for w in win._dfm_widgets] == [4, 6]
        ## Saving writes the entries themselves, chambers still unassigned —
        ## the initial yaml names every DFM the data holds.
        out = win._collect_yaml()
        assert [d["id"] for d in out["dfms"]] == [4, 6]
    finally:
        win.close()


def test_an_empty_directory_still_opens_a_blank_editor(app, tmp_path):
    win = FLICConfigEditor(initial_path=tmp_path)
    try:
        assert len(win._dfm_widgets) == 1
        assert win._current_path is None
    finally:
        win.close()


def test_a_factor_column_edits_through_a_dropdown_of_its_levels(app, tmp_path):
    """Typed levels invited typos and undeclared values; the cell editor is a
    combo of exactly what the factor declares, plus a blank to clear."""
    win = _open(app, tmp_path, {
        "global": {"chamber_layout": "two_well",
                   "experimental_design_factors": {
                       "paired": ["Paired", "Unpaired"],
                       "genotype": ["Chrim", "WCS"]}},
        "dfms": [{"id": 1, "chambers": {}}],
    })
    try:
        table = win._dfm_widgets[0]._chamber_table
        delegate = table.itemDelegate()
        option = QStyleOptionViewItem()
        for column, levels in ((1, ["Paired", "Unpaired"]),
                               (2, ["Chrim", "WCS"])):
            index = table.model().index(0, column)
            editor = delegate.createEditor(table, option, index)
            assert isinstance(editor, QComboBox)
            assert [editor.itemText(i) for i in range(editor.count())] \
                == [""] + levels
            editor.setCurrentText(levels[0])
            delegate.setModelData(editor, table.model(), index)
        assert win._collect_yaml()["dfms"][0]["chambers"][1] == "Paired, Chrim"
        assert win._problems()[1] == []
    finally:
        win.close()


def test_a_single_click_offers_the_levels_as_a_menu(app, tmp_path):
    """The click path is a QMenu, not the delegate's combo popup: popping a
    combo's list mid-click raced the mouse release, which dismissed it — a
    sporadic dead dropdown, worst under Wayland."""
    win = _open(app, tmp_path, {
        "global": {"chamber_layout": "two_well",
                   "experimental_design_factors": {
                       "genotype": ["Chrim", "WCS"]}},
        "dfms": [{"id": 1, "chambers": {}}],
    })
    try:
        widget = win._dfm_widgets[0]
        menu = widget._level_menu(0, 1)
        assert [a.text() for a in menu.actions()] == \
            ["(clear)", "Chrim", "WCS"]
        menu.actions()[1].trigger()   # what exec()'s return path does
        widget._chamber_table.item(0, 1).setText(menu.actions()[1].data())
        assert win._collect_yaml()["dfms"][0]["chambers"][1] == "Chrim"
        ## The current pick is marked, and clearing is always offered.
        menu2 = widget._level_menu(0, 1)
        assert [a.isChecked() for a in menu2.actions()] == \
            [False, True, False]
        ## An undeclared saved value appears rather than being rewritten.
        widget._chamber_table.item(1, 1).setText("Mystery")
        menu3 = widget._level_menu(1, 1)
        assert [a.text() for a in menu3.actions()] == \
            ["(clear)", "Chrim", "WCS", "Mystery"]
        ## The chamber column and factor-less tables get no menu.
        assert widget._level_menu(0, 0) is None
    finally:
        win.close()


def test_an_undeclared_level_from_disk_is_offered_not_rewritten(app, tmp_path):
    """Opening the dropdown on a cell whose saved level the design no longer
    declares must not silently replace it — it is offered, and validation
    reports it."""
    win = _open(app, tmp_path, {
        "global": {"chamber_layout": "two_well",
                   "experimental_design_factors": {"genotype": ["Chrim", "WCS"]}},
        "dfms": [{"id": 1, "chambers": {1: "Mystery"}}],
    })
    try:
        table = win._dfm_widgets[0]._chamber_table
        editor = table.itemDelegate().createEditor(
            table, QStyleOptionViewItem(), table.model().index(0, 1))
        assert "Mystery" in [editor.itemText(i) for i in range(editor.count())]
        assert win._problems()[1] == [
            "DFM 1 chamber 1: 'Mystery' is not a level of 'genotype'"]
    finally:
        win.close()


def test_without_factors_the_treatment_column_stays_free_text(app, tmp_path):
    """With no factors declared there is nothing to pick from."""
    win = _open(app, tmp_path, {
        "global": {"chamber_layout": "two_well"},
        "dfms": [{"id": 1, "chambers": {}}],
    })
    try:
        table = win._dfm_widgets[0]._chamber_table
        editor = table.itemDelegate().createEditor(
            table, QStyleOptionViewItem(), table.model().index(0, 1))
        assert isinstance(editor, QLineEdit)
    finally:
        win.close()


def test_a_declared_level_is_not_mangled_by_the_sanitizer(app, tmp_path):
    """The sanitizer exists for typed text; a level the design declares is
    written as declared, hyphens and all — otherwise the dropdown offers a
    value that turns invalid the moment it lands."""
    win = _open(app, tmp_path, {
        "global": {"chamber_layout": "two_well",
                   "experimental_design_factors": {
                       "line": ["UAS-x1", "w1118"]}},
        "dfms": [{"id": 1, "chambers": {}}],
    })
    try:
        table = win._dfm_widgets[0]._chamber_table
        table.setItem(0, 1, QTableWidgetItem("UAS-x1"))
        assert table.item(0, 1).text() == "UAS-x1"
        assert win._problems()[1] == []
        assert win._collect_yaml()["dfms"][0]["chambers"][1] == "UAS-x1"
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


# ---------------------------------------------------------------------------
# Progressive Ratio: paired chambers per chamber group
# ---------------------------------------------------------------------------

def test_progressive_ratio_shows_paired_pickers_and_writes_them(editor):
    _select_type(editor, "ProgressiveRatio")
    w = editor._dfm_widgets[0]
    assert not w._pr_card.isHidden()
    w.set_paired_chambers([2, 3, 6])
    assert w.get_dict()["paired_chambers"] == [2, 3, 6]

    _select_type(editor, "Hedonic")
    assert w._pr_card.isHidden()
    assert "paired_chambers" not in w.get_dict()


def test_progressive_ratio_reports_a_split_chamber_group(editor):
    from PyQt6.QtWidgets import QTableWidgetItem

    _select_type(editor, "ProgressiveRatio")
    w = editor._dfm_widgets[0]
    w._chamber_table.setItem(0, 1, QTableWidgetItem("Ctrl"))
    w._chamber_table.setItem(1, 1, QTableWidgetItem("Exp"))
    problems = w.chamber_problems()
    assert any("chambers 1 and 2" in p for p in problems)
    w._chamber_table.setItem(1, 1, QTableWidgetItem("Ctrl"))
    assert not any("chambers 1 and 2" in p for p in w.chamber_problems())


def test_progressive_ratio_paired_chambers_round_trip(app, tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from pr_fixtures import pr_config

    cfg = pr_config([{"id": 4, "pi_direction": "right", "paired_chambers": [2, 3, 6],
                      "chambers": {1: "A", 2: "A", 3: "B", 4: "B", 5: "A", 6: "A"}}])
    win = _open(app, tmp_path, cfg)
    try:
        w = win._dfm_widgets[0]
        assert not w._pr_card.isHidden()
        assert w.paired_chambers() == [2, 3, 6]
        out = win._collect_yaml()
        assert out["dfms"][0]["paired_chambers"] == [2, 3, 6]
        assert out["dfms"][0]["params"]["pi_direction"] == "right"
    finally:
        win.close()


# ---------------------------------------------------------------------------
# Progressive Ratio: the type's own constants (light QC, breaking point)
# ---------------------------------------------------------------------------

def _pr_cfg(constants: dict | None = None) -> dict:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from pr_fixtures import pr_config

    return pr_config([{"id": 1, "pi_direction": "left", "paired_chambers": [1, 3, 5],
                       "chambers": {1: "A", 2: "A", 3: "B", 4: "B", 5: "A", 6: "A"}}],
                     constants=constants)


def test_progressive_ratio_constants_show_for_that_type_only(editor):
    _select_type(editor, "ProgressiveRatio")
    form, w = editor._exp_form, editor._pr_constant_widgets
    assert form.isRowVisible(editor._pr_header_row)
    assert set(w) == {"require_training_complete", "exclude_failed_pr_groups",
                      "pr_lick_free_run", "pr_trend_min_events", "pr_trend_min_rho",
                      "pr_resting_level_rise", "pr_resting_level_ratio",
                      "pr_break_gap_min", "pr_test_window_min"}
    ## The type's defaults are placeholders, never values (ADR-0011).
    assert "120" in w["pr_break_gap_min"].placeholderText()
    assert "no cap" in w["pr_test_window_min"].placeholderText()
    assert w["exclude_failed_pr_groups"].itemText(0) == "default: exclude the group"
    assert "pr_break_gap_min" in w["pr_break_gap_min"].toolTip()
    assert "constants" not in editor._collect_yaml()["global"]

    _select_type(editor, "Hedonic")
    assert not form.isRowVisible(editor._pr_header_row)
    assert not form.isRowVisible(editor._pr_constant_rows["pr_break_gap_min"])


def test_progressive_ratio_constants_are_written_only_when_set(editor):
    _select_type(editor, "ProgressiveRatio")
    w = editor._pr_constant_widgets
    w["pr_break_gap_min"].setText("90")
    w["pr_trend_min_rho"].setText("0.25")
    w["exclude_failed_pr_groups"].setCurrentIndex(2)          # keep the group
    assert editor._collect_yaml()["global"]["constants"] == {
        "exclude_failed_pr_groups": False, "pr_trend_min_rho": 0.25,
        "pr_break_gap_min": 90}
    ## Another type drops them: they mean nothing to a Hedonic experiment.
    _select_type(editor, "Hedonic")
    assert "constants" not in editor._collect_yaml()["global"]


def test_a_bad_progressive_ratio_value_is_reported_not_dropped(editor):
    _select_type(editor, "ProgressiveRatio")
    editor._pr_constant_widgets["pr_break_gap_min"].setText("soon")
    editor._pr_constant_widgets["pr_trend_min_rho"].setText("2")
    assert editor._collect_yaml()["global"]["constants"]["pr_break_gap_min"] == "soon"
    experiment, _dfms = editor._problems()
    assert any("pr_break_gap_min" in p and "number" in p for p in experiment)
    assert any("pr_trend_min_rho" in p and "at most 1" in p for p in experiment)


def test_progressive_ratio_constants_round_trip(app, tmp_path):
    win = _open(app, tmp_path, _pr_cfg({"pr_break_gap_min": 180,
                                        "pr_test_window_min": 600,
                                        "require_training_complete": False,
                                        "my_own_note": 3}))
    try:
        w = win._pr_constant_widgets
        assert w["pr_break_gap_min"].text() == "180"
        assert w["pr_test_window_min"].text() == "600"
        assert w["require_training_complete"].currentData() is False
        assert w["exclude_failed_pr_groups"].currentData() is None
        assert w["pr_lick_free_run"].text() == ""
        assert win._collect_yaml()["global"]["constants"] == {
            "require_training_complete": False, "pr_break_gap_min": 180,
            "pr_test_window_min": 600, "my_own_note": 3}
        ## A cleared row goes back to the type's default: its key leaves.
        w["pr_test_window_min"].clear()
        assert "pr_test_window_min" not in win._collect_yaml()["global"]["constants"]
    finally:
        win.close()


def test_a_member_shows_the_designs_progressive_ratio_constants_locked(app, tmp_path):
    member = _project_with_member(
        tmp_path,
        {"experiment_type": "ProgressiveRatio",
         "well_names": {"A": "Sucrose", "B": "Yeast"},
         "constants": {"pr_break_gap_min": 240, "exclude_failed_pr_groups": False}},
        {"dfms": [{"id": 1, "paired_chambers": [1, 3, 5],
                   "chambers": {1: "Ctrl", 2: "Ctrl"}}]})
    win = FLICConfigEditor(initial_path=member)
    try:
        w = win._pr_constant_widgets
        assert w["pr_break_gap_min"].text() == "240"
        assert w["exclude_failed_pr_groups"].currentData() is False
        assert not w["pr_break_gap_min"].isEnabled()
        assert "Project design" in w["pr_break_gap_min"].toolTip()
        assert "global" not in win._collect_yaml()
    finally:
        win.close()


def test_new_clears_the_progressive_ratio_constants(app, tmp_path):
    win = _open(app, tmp_path, _pr_cfg({"pr_break_gap_min": 180,
                                        "require_training_complete": False}))
    try:
        win._new()
        w = win._pr_constant_widgets
        assert w["pr_break_gap_min"].text() == ""
        assert w["require_training_complete"].currentData() is None
    finally:
        win.close()
