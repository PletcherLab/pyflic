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
