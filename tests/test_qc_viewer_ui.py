"""The QC Viewer's Project awareness and its handoff from the Hub.

A member of a Project must load *through* the Project so the Design's
``global:`` is inherited (ADR-0005) — a member config with no ``global:`` of
its own is the normal case, and a direct ``load_experiment_yaml`` on it
fails.  Opened from the Hub, the viewer starts on the already-loaded
experiment instead of re-parsing the DFM CSVs, and its exclusion group is
the one the experiment analyzes with (ADR-0010), not a hardcoded
``general``.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

pytest.importorskip("PyQt6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import matplotlib

matplotlib.use("Agg")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from pyflic.base import qc_viewer  # noqa: E402
from pyflic.base.parameters import Parameters  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def member(tmp_path):
    proj = tmp_path / "P"
    member = proj / "Rep1"
    member.mkdir(parents=True)
    (proj / "project.yaml").write_text("name: P\n")
    (member / "flic_config.yaml").write_text("dfms: []\n")
    return member


def test_a_member_loads_through_its_project(app, member):
    fake_exp = mock.MagicMock()
    fake_exp.qc_results = {}
    with mock.patch("pyflic.base.project.Project") as FakeProject:
        FakeProject.return_value.load_member.return_value = fake_exp
        worker = qc_viewer._LoadWorker(member, (0.0, 0.0), True)
        got: list = []
        worker.finished.connect(got.append)
        errors: list = []
        worker.errored.connect(errors.append)
        worker.run()
    assert errors == []
    FakeProject.assert_called_once_with(str(member.parent))
    call = FakeProject.return_value.load_member.call_args
    assert call.args[0] == "Rep1"
    ## QC views everything: exclusions are what it exists to decide.
    assert call.kwargs["exclusion_group"] is None
    assert got == [fake_exp]
    fake_exp.write_qc_reports.assert_called_once()


def test_a_standalone_experiment_still_loads_directly(app, tmp_path):
    alone = tmp_path / "Alone"
    alone.mkdir()
    (alone / "flic_config.yaml").write_text("dfms: []\n")
    fake_exp = mock.MagicMock()
    fake_exp.qc_results = {}
    with mock.patch("pyflic.base.yaml_config.load_experiment_yaml",
                    return_value=fake_exp) as loader:
        worker = qc_viewer._LoadWorker(alone, (0.0, 0.0), True)
        errors: list = []
        worker.errored.connect(errors.append)
        worker.run()
    assert errors == []
    loader.assert_called_once()


class _FakeDfm:
    params = Parameters()


class _FakeExp:
    """Just enough of a loaded Experiment for the handoff path."""

    def __init__(self, member: Path) -> None:
        self.dfms = {1: _FakeDfm()}
        self.experiment_dir = str(member)      # str on purpose: coercion
        self.exclusion_group = "design_group"
        self._output_root = member
        self.qc_dir = member / "qc"            # never written

    def feeding_summary(self):
        raise RuntimeError("no data in this fake")


def test_the_hub_handoff_skips_the_duplicate_load(app, member):
    win = qc_viewer.MainWindow(member, experiment=_FakeExp(member))
    try:
        tabs = [win._tabs.tabText(i) for i in range(win._tabs.count())]
        assert tabs == ["Load", "Feeding Summary", "DFM 1", "Params"]
        assert win._tabs.currentIndex() == 1
        assert isinstance(win._experiment_dir, Path)
        ## The design-named group, not a hardcoded 'general'.
        assert win._active_group() == "design_group"
        ## The signal-plot tabs will be empty; the in-viewer fix is named
        ## and armed.
        assert "Run QC" in win.statusBar().currentMessage()
        assert win._btn_run_qc.isEnabled()
    finally:
        win.close()


def test_a_load_naming_no_group_saves_to_the_designs(app, member):
    """No group on the experiment: the design's ``exclusion_group``, not a
    hardcoded ``general`` that no run reads (P3)."""
    exp = _FakeExp(member)
    exp.exclusion_group = None
    exp.global_config = {"exclusion_group": "main"}
    win = qc_viewer.MainWindow(member, experiment=exp)
    try:
        assert win._active_group() == "main"
    finally:
        win.close()


def test_saving_to_another_analysis_own_group_asks_first(app, member, monkeypatch):
    """A save replaces every row of the group it names.  Typed over the
    loaded Slice's group, another analysis's own group would lose its
    declarations to ticks read from somewhere else: asked, with the count."""
    from PyQt6.QtWidgets import QInputDialog, QMessageBox

    from pyflic.base.exclusions import read_exclusions, write_exclusions

    exp = _FakeExp(member)
    exp.exclusion_group = "early"
    exp.global_config = {"exclusion_group": "main", "slices": {
        "early": {"start": 0, "end": 10}, "late": {"start": 10, "end": "end"}}}
    write_exclusions(member, "main", {1: [1, 2, 3]})
    asked: list[str] = []
    answer = [QMessageBox.StandardButton.Cancel]
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **_k: asked.append(a[2]) or answer[0]))
    win = qc_viewer.MainWindow(member, experiment=exp)

    def save_as(group: str) -> None:
        monkeypatch.setattr(QInputDialog, "getText",
                            staticmethod(lambda *_a, **_k: (group, True)))
        win._on_save_exclusions()

    try:
        save_as("main")
        save_as("late")
        assert len(asked) == 2
        assert "own group of the whole recording" in asked[0]
        assert "replaces the 3 row(s) of 'main'" in asked[0]
        assert "own group of Slice late" in asked[1]
        ## Cancelled: nothing written.
        assert read_exclusions(member) == {"main": {1: [1, 2, 3]}}

        ## The loaded analysis's own group, and a group no analysis owns,
        ## save without asking.
        save_as("early")
        save_as("spare")
        assert len(asked) == 2

        answer[0] = QMessageBox.StandardButton.Save
        save_as("main")
        assert len(asked) == 3 and "main" not in read_exclusions(member)
    finally:
        win.close()


def test_run_qc_waits_for_an_experiment(app, member):
    win = qc_viewer.MainWindow(member)
    try:
        assert not win._btn_run_qc.isEnabled()
    finally:
        win.close()


def test_reload_qc_picks_up_fresh_artifacts_and_keeps_exclusions(app, tmp_path):
    """A QC run written after the tabs were built must show without a
    reload — and must not reset the (possibly unsaved) well checkboxes."""
    empty = tmp_path / "qc"
    tab = qc_viewer.DfmTab(1, empty, excluded_wells=[3])
    try:
        assert tab._tabs.count() == 4     # Integrity + the 3 signal plots
        tab.set_well_excluded(5, True)
        ## A fresh run wrote the two-well artifacts.
        (empty / "simultaneous_feeding").mkdir(parents=True)
        (empty / "simultaneous_feeding" /
         "DFM1_simultaneous_feeding_matrix.csv").write_text("a,b\n1,2\n")
        tab.reload_qc(empty)
        labels = [tab._tabs.tabText(i) for i in range(tab._tabs.count())]
        assert "Sim. Feeding" in labels and "Bleeding" in labels
        assert tab.get_excluded_wells() == [3, 5]
    finally:
        tab.deleteLater()


def test_qc_written_refreshes_and_rearms(app, member):
    win = qc_viewer.MainWindow(member, experiment=_FakeExp(member))
    try:
        win._btn_run_qc.setEnabled(False)   # as _on_run_qc leaves it
        win._on_qc_written(member / "qc")
        assert win._btn_run_qc.isEnabled()
        assert "refreshed" in win.statusBar().currentMessage()
    finally:
        win.close()


# ---------------------------------------------------------------------------
# The Opto Light QC tab
# ---------------------------------------------------------------------------

@pytest.fixture
def opto_window(app, tmp_path):
    """The viewer on the synthetic optogenetic rig (``opto_fixtures``), handed
    over as the Hub hands over a loaded member."""
    import contextlib
    import io

    from opto_fixtures import make_opto_dir
    from pyflic.base.yaml_config import load_experiment_yaml

    root = make_opto_dir(tmp_path / "rig")
    with contextlib.redirect_stdout(io.StringIO()):
        exp = load_experiment_yaml(root, parallel=False, use_disk_cache=False)
    win = qc_viewer.MainWindow(root, experiment=exp)
    yield win
    win.close()


def _column(tab, header: str) -> int:
    headers = [tab._table.horizontalHeaderItem(c).text()
               for c in range(tab._table.columnCount())]
    return headers.index(header)


def test_an_optogenetic_experiment_gets_the_tab(opto_window):
    win = opto_window
    tabs = [win._tabs.tabText(i) for i in range(win._tabs.count())]
    assert tabs == ["Load", "Feeding Summary", "DFM 1", "Opto Light QC", "Params"]
    win._tabs.setCurrentIndex(tabs.index("Opto Light QC"))
    assert win._help_ref == "app-qc-viewer#the-opto-light-qc-tab"


def test_the_tab_opens_on_the_worst_group_and_says_why(opto_window):
    from pyflic.base.ui import ZoomableImageView

    tab = opto_window._opto_tab
    assert tab._table.rowCount() == 12
    verdict = _column(tab, "Verdict")
    assert tab._table.item(1, verdict).text() == "failed: unexplained light"
    assert tab._table.item(0, verdict).text() == "ok"
    ## The first failed group is selected, and the detail pane explains it.
    selected = tab._selected()
    assert selected["Verdict"] == "failed" and selected["Group"] == 2
    why = tab._why.toPlainText()
    assert "DFM 1 linkage group 2" in why and "drifting baseline" in why
    assert tab._details.count() == 4
    assert tab._hosts["Intervals"].count() == 1 and tab._hosts["Light events"].count() == 1
    ## The figure is drawn for the selected group alone.
    assert tab._only_selected.isChecked()
    assert isinstance(tab._figure_host.itemAt(0).widget(), ZoomableImageView)
    assert tab._figure_key[1] == (2,)
    tab._only_selected.setChecked(False)
    assert tab._figure_key[1] is None                  # the whole DFM


def _detail_headers(tab, label: str) -> list[str]:
    table = tab._hosts[label].itemAt(0).widget()
    return [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]


def test_the_detail_tables_drop_columns_with_nothing_in_them(opto_window):
    """Generation 1 records none of the firmware's own columns: the
    Intervals and Light events tables leave them out, as Program does."""
    tab = opto_window._opto_tab
    opto = opto_window._opto_tab._exp.opto
    selected = tab._selected()
    for label, frame in (("Intervals", opto.interval_table()),
                         ("Light events", opto.events_table())):
        rows = frame[(frame["DFM"] == selected["DFM"]) & (frame["Group"] == selected["Group"])]
        kept = [c for c in rows.columns if c not in ("DFM", "Group") and rows[c].notna().any()]
        headers = _detail_headers(tab, label)
        assert headers == kept, label
        assert len(kept) < len(rows.columns) - 2, label
    assert "ProgramInterval" not in _detail_headers(tab, "Intervals")
    assert not {"Phase", "RequirementMet", "FirmwareEvents"} & \
        set(_detail_headers(tab, "Light events"))


def test_the_buttons_tick_chambers_but_save_nothing(opto_window, tmp_path):
    win = opto_window
    tab = win._opto_tab
    feeding = win._feeding_tab
    tab._btn_exclude_failed.click()
    excluded = feeding._table.excluded_dataframe()
    assert sorted(excluded["Chamber"].astype(int)) == [2, 3, 4]
    assert win._dfm_tab_widgets[1].get_excluded_wells() == [2, 3, 4]
    assert "Save Exclusions" in win.statusBar().currentMessage()
    assert not (Path(win._experiment_dir) / "remove_chambers.csv").exists()
    ## The selected group only: pick the open-loop warning (group 6).
    tab._table.selectRow(5)
    tab._btn_exclude_selected.click()
    assert 6 in set(feeding._table.excluded_dataframe()["Chamber"].astype(int))


def test_a_params_recompute_refreshes_the_verdicts(opto_window):
    win = opto_window
    tab = win._opto_tab
    verdict = _column(tab, "Verdict")
    assert tab._table.item(0, verdict).text() == "ok"
    ## Thresholds no burst reaches: the healthy group's light loses its licks.
    win._on_params_recompute({"feeding_threshold": 500.0, "feeding_minimum": 400.0,
                              "tasting_minimum": 300.0, "tasting_maximum": 400.0})
    assert tab._table.item(0, verdict).text() == "failed: unexplained light"
    assert "opto light QC" in win.statusBar().currentMessage()


# ---------------------------------------------------------------------------
# The Program sub-tab and Messages.csv (docs/v4-data-plan.md, stream 6)
# ---------------------------------------------------------------------------

def _load_quietly(root):
    import contextlib
    import io

    from pyflic.base.yaml_config import load_experiment_yaml

    with contextlib.redirect_stdout(io.StringIO()):
        return load_experiment_yaml(root, parallel=False, use_disk_cache=False)


@pytest.fixture
def v4_window(app, tmp_path):
    """The viewer on a firmware 4.0.0 recording of two DFMs, DFM 1 with a gap
    PyMCU logged in Messages.csv."""
    import yaml

    from v4_fixtures import fixed_ratio_spec, write_v4_recording

    root = tmp_path / "v4"
    spec = fixed_ratio_spec(minutes=6.0, dfm_ids=(1, 2))
    write_v4_recording(root / "data", spec, index_gaps={1: [(500, 4)]})
    params = {"feeding_threshold": 20, "feeding_minimum": 10, "tasting_minimum": 5,
              "tasting_maximum": 20, "samples_per_second": 5, "pi_direction": "left"}
    cfg = {"global": {"chamber_layout": "two_well", "transform_licks": False,
                      "params": params},
           "dfms": [{"id": d, "chambers": {c: "Ctrl" for c in range(1, 7)}} for d in (1, 2)]}
    (root / "flic_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    win = qc_viewer.MainWindow(root, experiment=_load_quietly(root))
    yield win
    win.close()


def _sub_tabs(tab) -> list[str]:
    return [tab._tabs.tabText(i) for i in range(tab._tabs.count())]


def test_a_generation_3_member_gets_the_program_sub_tab(v4_window):
    from pyflic.base.program_timeline import WELL_TABLE_COLUMNS
    from pyflic.base.ui import ZoomableImageView

    win = v4_window
    for dfm_id in (1, 2):
        assert _sub_tabs(win._dfm_tab_widgets[dfm_id])[-1] == "Program"
    ## Drawn the first time it is shown, not at load.
    panel = win._dfm_tab_widgets[1]._program_panel
    assert panel.table is None
    win.show()
    win._tabs.setCurrentWidget(win._dfm_tab_widgets[1])
    win._dfm_tab_widgets[1]._tabs.setCurrentWidget(panel)
    QApplication.processEvents()
    table = panel.table
    assert table is not None and table.rowCount() == 12
    assert [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())] == \
        list(WELL_TABLE_COLUMNS)
    assert isinstance(panel._figure_host.itemAt(0).widget(), ZoomableImageView)
    ## A Params recompute redraws it from the DFMs the recompute put in place.
    win._on_params_recompute({"feeding_threshold": 25.0})
    QApplication.processEvents()
    assert panel.table is not None and panel.table is not table


def test_the_program_sub_tab_lists_the_dfms_activations(v4_window):
    """Stream S3: beside the wells, the DFM's Activations, the ledger's rows
    of opto_light_events.csv."""
    from pyflic.base.program_timeline import ACTIVATION_TABLE_COLUMNS, activations_table

    win = v4_window
    panel = win._dfm_tab_widgets[1]._program_panel
    win.show()
    win._tabs.setCurrentWidget(win._dfm_tab_widgets[1])
    win._dfm_tab_widgets[1]._tabs.setCurrentWidget(panel)
    QApplication.processEvents()
    table = panel.activations
    expected = activations_table(panel._exp, 1)
    assert table is not None and table.rowCount() == len(expected) > 0
    assert [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())] == \
        list(ACTIVATION_TABLE_COLUMNS)


def test_a_generation_3_group_keeps_the_firmwares_columns(v4_window):
    tab = v4_window._opto_tab
    assert {"ProgramInterval", "Cycle", "Paradigm"} <= set(_detail_headers(tab, "Intervals"))
    assert {"Phase", "RequirementMet", "FirmwareEvents"} <= \
        set(_detail_headers(tab, "Light events"))


def test_the_integrity_sub_tab_lists_the_dfms_messages(v4_window):
    from PyQt6.QtWidgets import QLabel

    tab = v4_window._dfm_tab_widgets[1]
    integrity = tab._tabs.widget(_sub_tabs(tab).index("Integrity"))
    heads = [w.text() for w in integrity.findChildren(QLabel) if "Messages.csv" in w.text()]
    ## DFM 1's gap error and the MCU's start and end; nothing of DFM 2's.
    assert heads == ["<b>Messages.csv</b> — DFM 1 and the MCU (DFM 0): 3 message(s), "
                     "1 error(s)"]
    table = integrity.findChildren(qc_viewer._DfTableWidget)[-1]
    assert table.rowCount() == 3
    other = v4_window._dfm_tab_widgets[2]
    integrity = other._tabs.widget(_sub_tabs(other).index("Integrity"))
    assert any("2 message(s), 0 error(s)" in w.text() for w in integrity.findChildren(QLabel))


def test_a_generation_1_member_has_a_program_sub_tab_only_with_a_program(app, tmp_path):
    from opto_fixtures import make_opto_dir

    bare = make_opto_dir(tmp_path / "bare", program=None)
    win = qc_viewer.MainWindow(bare, experiment=_load_quietly(bare))
    try:
        assert "Program" not in _sub_tabs(win._dfm_tab_widgets[1])
    finally:
        win.close()
    rig = make_opto_dir(tmp_path / "rig")
    win = qc_viewer.MainWindow(rig, experiment=_load_quietly(rig))
    try:
        assert _sub_tabs(win._dfm_tab_widgets[1])[-1] == "Program"
    finally:
        win.close()


# ---------------------------------------------------------------------------
# Sorting the Feeding Summary; colouring Sim. Feeding and Bleeding
# ---------------------------------------------------------------------------

def _summary():
    import numpy as np
    import pandas as pd

    return pd.DataFrame({"DFM": [1, 1, 1, 2], "Chamber": [1, 2, 3, 1],
                         "Licks": [9.0, 100.0, np.nan, 10.0]})


def _column_text(table, header: str) -> list[str]:
    c = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())].index(header)
    return [table.item(r, c).text() for r in range(table.rowCount())]


def test_the_feeding_summary_sorts_by_value_with_blanks_last(app):
    from PyQt6.QtCore import Qt

    tab = qc_viewer.FeedingSummaryTab()
    try:
        tab.populate(_summary())
        table = tab._table
        assert _column_text(table, "Chamber") == ["1", "2", "3", "1"]   # as given
        c = [table.horizontalHeaderItem(i).text()
             for i in range(table.columnCount())].index("Licks")
        table.sortByColumn(c, Qt.SortOrder.AscendingOrder)
        assert _column_text(table, "Licks") == ["9", "10", "100", "nan"]
        table.sortByColumn(c, Qt.SortOrder.DescendingOrder)
        assert _column_text(table, "Licks") == ["100", "10", "9", "nan"]
        ## A recompute re-populates the tab; the sort stays.
        tab.populate(_summary())
        assert _column_text(tab._table, "Licks") == ["100", "10", "9", "nan"]
    finally:
        tab.deleteLater()


def test_a_sorted_feeding_summary_excludes_the_chamber_ticked(app):
    from PyQt6.QtCore import Qt

    tab = qc_viewer.FeedingSummaryTab()
    try:
        tab.populate(_summary(), excluded_by_dfm={2: [1]})
        table = tab._table
        seen = []
        tab.exclusion_changed.connect(lambda d, ch, e: seen.append((d, ch, e)))
        c = [table.horizontalHeaderItem(i).text()
             for i in range(table.columnCount())].index("Licks")
        table.sortByColumn(c, Qt.SortOrder.DescendingOrder)
        table.item(0, 0).setCheckState(Qt.CheckState.Checked)      # 100 licks
        assert seen == [(1, 2, True)]
        tab.set_chamber_excluded(1, 3, True)
        assert table.excluded_dataframe()[["DFM", "Chamber"]].values.tolist() == [
            [1, 2], [1, 3], [2, 1]]
        assert table.included_dataframe()[["DFM", "Chamber"]].values.tolist() == [[1, 1]]
    finally:
        tab.deleteLater()


def test_crosstalk_is_judged_by_its_share_of_the_chambers_lick_samples():
    import pandas as pd

    frame = pd.DataFrame(
        {"Licks1": [1000, 1000, 1000, 1000, 1000], "Licks2": [1000, 1000, 1000, 1000, 0],
         "Both": [0, 0, 0, 30, 0], "Both_pre": [0, 1, 3, 30, 0]},
        index=[f"Chamber{i}" for i in range(1, 6)])
    tiers = qc_viewer.crosstalk_tiers(frame)
    ## 1 of 2000 lick samples is 0.05%, 3 is 0.15%; 30 of 1970 is 1.5%.
    assert {r: t for r, (t, _why) in tiers.items()} == {2: "light", 3: "dark"}
    assert "before the dual-feeding correction" in tiers[2][1]
    ## Without the _pre columns, Both is what the wells registered.
    assert qc_viewer.crosstalk_tiers(frame.drop(columns="Both_pre"))[3][0] == "dark"


def test_bleeding_is_judged_against_the_responding_wells_thresholds():
    import pandas as pd

    labels = [f"W{i}" for i in range(1, 4)]
    matrix = pd.DataFrame([[90.0, 15.0, 25.0], [12.0, 80.0, 5.0], [9.0, 30.0, 70.0]],
                          index=[f"{w}Sig" for w in labels],
                          columns=[f"{w}Resp" for w in labels])
    feeding_max = {1: 20.0, 2: 20.0, 3: 20.0}
    tiers = qc_viewer.bleeding_tiers(matrix, feeding_max, {1: 10.0, 2: 10.0, 3: 10.0})
    assert {k: t for k, (t, _why) in tiers.items()} == {
        (0, 1): "light", (0, 2): "dark", (1, 0): "light", (2, 1): "dark"}
    assert "W1's signal" in tiers[(0, 2)][1] and "at W3" in tiers[(0, 2)][1]
    ## A minimum at the threshold has no light tier; an unknown well is not judged.
    tiers = qc_viewer.bleeding_tiers(matrix, {1: 20.0, 3: 20.0}, {1: 20.0, 3: 20.0})
    assert {k: t for k, (t, _why) in tiers.items()} == {(0, 2): "dark"}


def test_the_dfm_tab_colours_the_problem_cells(app, tmp_path):
    from types import SimpleNamespace

    import pandas as pd

    qc = tmp_path / "qc"
    (qc / "simultaneous_feeding").mkdir(parents=True)
    (qc / "bleeding").mkdir()
    pd.DataFrame({"Licks1": [1000, 1000], "Licks2": [1000, 1000], "Both": [0, 0],
                  "Both_pre": [0, 40]}, index=["Chamber1", "Chamber2"]).to_csv(
        qc / "simultaneous_feeding" / "DFM1_simultaneous_feeding_matrix.csv")
    wells = range(1, 13)
    bleed = pd.DataFrame(0.0, index=[f"W{i}Sig" for i in wells],
                         columns=[f"W{i}Resp" for i in wells])
    bleed.iloc[0, 0], bleed.iloc[0, 1], bleed.iloc[1, 0] = 100.0, 25.0, 12.0
    bleed.to_csv(qc / "bleeding" / "DFM1_bleeding_matrix.csv")
    pd.Series(1.0, index=[f"W{i}" for i in wells]).to_csv(
        qc / "bleeding" / "DFM1_bleeding_alldata.csv", header=True)

    tab = qc_viewer.DfmTab(1, qc)
    try:
        labels = [tab._tabs.tabText(i) for i in range(tab._tabs.count())]
        dark, light = (qc_viewer._TIER_COLORS[t][0] for t in ("dark", "light"))

        def colour(table, r, c):
            from PyQt6.QtCore import Qt

            brush = table.item(r, c).data(Qt.ItemDataRole.BackgroundRole)
            return None if brush is None else brush.color().name()

        sim = tab._tabs.widget(labels.index("Sim. Feeding")).findChildren(
            qc_viewer._DfTableWidget)[0]
        assert [colour(sim, r, 0) for r in range(2)] == [None, dark]    # 2% of 2000
        matrix, = (t for t in tab._tabs.widget(labels.index("Bleeding")).findChildren(
            qc_viewer._DfTableWidget) if t.columnCount() == 12)
        ## The two-well defaults without an experiment: threshold 20, minimum 10.
        assert (colour(matrix, 0, 0), colour(matrix, 0, 1), colour(matrix, 1, 0),
                colour(matrix, 1, 2)) == (None, dark, light, None)

        ## A loaded DFM's own thresholds, per well, win.
        frame = pd.DataFrame({"FeedingMax": [30.0], "FeedingMin": [15.0]})
        tab._exp = SimpleNamespace(dfms={1: SimpleNamespace(
            thresholds={f"W{w}": frame for w in wells}, params=Parameters.two_well())})
        assert tab._feeding_thresholds(1) == ({w: 30.0 for w in wells},
                                              {w: 15.0 for w in wells})
    finally:
        tab.deleteLater()
