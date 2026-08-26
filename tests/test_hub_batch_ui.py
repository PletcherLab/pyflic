"""The Hub's Batch and Project surfaces (ADR-0007, ADR-0009, ADR-0010).

Headless: every test drives the widgets directly.  The point is the contracts
the strip and the tables promise — tiles that never go dark, blocked rows that
say why, a check column that survives a rebuild — not pixels.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

pytest.importorskip("PyQt6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from pyflic.base.batch_preflight import BatchPreflightDialog  # noqa: E402
from pyflic.base.hub import AnalysisHubWindow, MemberConfigsDialog  # noqa: E402

DESIGN = {"global": {"chamber_layout": "two_well", "well_names": ["A", "B"]}}
CONFIG = {"dfms": [{"id": 1, "chambers": {1: "Ctrl", 2: "Exp"}},
                   {"id": 2, "chambers": {1: "Ctrl", 2: "Exp"}}]}


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(payload, sort_keys=False),
                        encoding="utf-8")


def _member(root: Path, *, configured=True, filed=True) -> None:
    data = root / "data" if filed else root
    for dfm in (1, 2):
        _write(data / f"DFM{dfm}_0.csv", "Sample,Seconds\n")
    if configured:
        _write(root / "flic_config.yaml", CONFIG)


@pytest.fixture
def batch_root(tmp_path: Path) -> Path:
    root = tmp_path / "batch"
    _write(root / "Sept2026" / "ProjA" / "project.yaml",
           {"name": "ProjA", "design": DESIGN,
            "scripts": [{"name": "batch", "steps": []}]})
    _member(root / "Sept2026" / "ProjA" / "rep1")
    _member(root / "Sept2026" / "ProjA" / "rep2")
    _member(root / "Sept2026" / "ProjA" / "rep3_loose", configured=False,
            filed=False)
    _member(root / "Sept2026" / "ProjA" / "rep4_nocfg", configured=False)
    _write(root / "ProjB" / "project.yaml",
           {"name": "ProjB", "design": DESIGN,
            "scripts": [{"name": "batch", "steps": []}]})
    _member(root / "ProjB" / "rep1")
    return root


@pytest.fixture
def hub(app, batch_root, monkeypatch):
    window = AnalysisHubWindow(target=str(batch_root))
    yield window
    window.close()


def _column(table, column: int) -> list[str]:
    return [table.item(row, column).text() for row in range(table.rowCount())]


# ---------------------------------------------------------------------------
# The batch table
# ---------------------------------------------------------------------------

def test_the_batch_table_lists_every_project_by_relative_key(hub):
    assert _column(hub.batch_table, 0) == ["ProjB", "Sept2026/ProjA"]


def test_a_project_with_blocked_members_is_red_and_says_why(hub):
    row = _column(hub.batch_table, 0).index("Sept2026/ProjA")
    assert hub.batch_table.item(row, 3).text() == "2 blocked"
    tip = hub.batch_table.item(row, 0).toolTip()
    assert "unfiled recording" in tip and "no config" in tip


def test_a_runnable_project_starts_checked(hub):
    for row in range(hub.batch_table.rowCount()):
        assert hub.batch_table.item(row, 0).checkState() == Qt.CheckState.Checked


def test_a_project_with_nothing_usable_starts_unchecked(app, tmp_path):
    root = tmp_path / "batch"
    _write(root / "ProjA" / "project.yaml", {"name": "ProjA", "design": DESIGN})
    _member(root / "ProjA" / "rep1", configured=False, filed=False)
    window = AnalysisHubWindow(target=str(root))
    try:
        assert window.batch_table.item(0, 0).checkState() \
            == Qt.CheckState.Unchecked
        assert window._batch_checked_keys() == []
    finally:
        window.close()


def test_unchecking_survives_a_rebuild(hub):
    hub.batch_table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    dropped = hub.batch_table.item(0, 0).text()
    hub.refresh()
    assert dropped not in hub._batch_checked_keys()


def test_the_batch_tile_is_never_dark(app, tmp_path):
    """Its panel holds "Choose batch folder…" — the control that fixes the
    empty state.  A dimmed tile there reads as unavailable."""
    window = AnalysisHubWindow()
    try:
        assert not window.tiles["batch"].is_dimmed()
        assert not window.tiles["project"].is_dimmed()
        assert "no batch open" in window.tiles["batch"].summary_text()
    finally:
        window.close()


# ---------------------------------------------------------------------------
# Drilling in
# ---------------------------------------------------------------------------

def test_double_clicking_a_project_selects_it_and_shows_its_panel(hub):
    row = _column(hub.batch_table, 0).index("Sept2026/ProjA")
    hub._batch_row_activated(hub.batch_table.item(row, 0))
    assert hub.project is not None and hub.project.name == "ProjA"
    assert hub._open_key == "project"
    ## The Batch panel keeps showing the Batch it came from.
    assert hub._batch_view_root() is not None


def test_the_project_table_lists_blocked_folders_that_are_not_members_yet(hub):
    row = _column(hub.batch_table, 0).index("Sept2026/ProjA")
    hub._batch_row_activated(hub.batch_table.item(row, 0))
    names = _column(hub.project_table, 0)
    assert names == ["rep1", "rep2", "rep3_loose", "rep4_nocfg"]
    tip = hub.project_table.item(2, 0).toolTip()
    assert "unfiled recording" in tip


def test_the_repair_buttons_count_what_they_would_fix(hub):
    row = _column(hub.batch_table, 0).index("Sept2026/ProjA")
    hub._batch_row_activated(hub.batch_table.item(row, 0))
    assert hub.file_btn.isEnabled()
    assert "1 unfiled" in hub.file_btn.text()
    assert "1 missing" in hub.scaffold_btn.text()


# ---------------------------------------------------------------------------
# Card dimming
# ---------------------------------------------------------------------------

def test_cards_with_no_subject_are_dimmed_but_still_live(hub):
    for key in ("analyze", "plots"):
        cards = hub.panels[key].cards()
        assert cards and all(card.is_dimmed() for card in cards)
        assert all(card.isEnabled() for card in cards)
    for key in ("batch", "project", "tools"):
        assert all(not card.is_dimmed() for card in hub.panels[key].cards())


# ---------------------------------------------------------------------------
# The preflight
# ---------------------------------------------------------------------------

def test_the_preflight_states_the_target_list(app, hub, batch_root):
    dialog = BatchPreflightDialog(hub, batch_root, checked=["Sept2026/ProjA"],
                                  log=lambda _m: None)
    try:
        assert dialog.selected_keys == ["Sept2026/ProjA"]
        assert "1 of 2 project(s)" in dialog._heading.text()
        assert "2 blocked member(s)" in dialog._heading.text()
    finally:
        dialog.close()


def test_filing_inside_the_preflight_re_checks_the_repaired_project(app, tmp_path):
    """Filing is what MAKES a Project runnable, so re-deriving "unchecked" from
    the pre-repair state excluded the very Project the user had just fixed."""
    root = tmp_path / "batch"
    _write(root / "ProjA" / "project.yaml", {"name": "ProjA", "design": DESIGN})
    _member(root / "ProjA" / "rep1", filed=False)
    dialog = BatchPreflightDialog(None, root, checked=[], log=lambda _m: None)
    try:
        assert dialog.selected_keys == []
        dialog._file(str(root / "ProjA" / "rep1"))
        assert dialog.selected_keys == ["ProjA"]
    finally:
        dialog.close()


def test_the_preflight_previews_the_sheet_without_writing_it(app, hub, batch_root):
    _write(batch_root / "remove_chambers.csv",
           "project,member,dfm,chamber,group,reason\n"
           "Sept2026/ProjA,rep1,1,2,general,noisy\n")
    dialog = BatchPreflightDialog(hub, batch_root, log=lambda _m: None)
    try:
        assert "applied: 1" in dialog._sheet_label.text()
        assert dialog.apply_exclusions
        assert not (batch_root / "Sept2026" / "ProjA" / "rep1"
                    / "remove_chambers.csv").exists()
    finally:
        dialog.close()


# ---------------------------------------------------------------------------
# Member configs
# ---------------------------------------------------------------------------

def test_member_configs_lists_every_folder_and_scaffolds_the_missing(app, hub):
    row = _column(hub.batch_table, 0).index("Sept2026/ProjA")
    hub._batch_row_activated(hub.batch_table.item(row, 0))
    dialog = MemberConfigsDialog(hub, hub.project)
    try:
        labels = [dialog._list.item(i).text()
                  for i in range(dialog._list.count())]
        assert any("rep4_nocfg — no flic_config.yaml" in text
                   for text in labels)
        dialog._create_all_missing()
        directory = Path(hub.project.project_directory) / "rep4_nocfg"
        assert (directory / "flic_config.yaml").is_file()
    finally:
        dialog.close()


# ---------------------------------------------------------------------------
# The output log
# ---------------------------------------------------------------------------

def test_a_streamed_line_is_not_shown_twice(hub):
    hub.log.clear_log()
    hub._on_worker_line("[ProjA] runn")
    hub._on_worker_line("ing\n")
    assert hub.log.toPlainText().strip() == "[ProjA] running"


def test_failures_are_copied_to_the_errors_tab(hub):
    hub.log.clear_log()
    hub.errors.clear_log()
    hub._on_worker_line("[ProjA] ok\n[ProjB] FAILED: boom\n")
    assert "FAILED: boom" in hub.errors.toPlainText()
    assert "[ProjA] ok" not in hub.errors.toPlainText()


def test_opening_an_unrelated_project_clears_the_batch(app, hub, tmp_path):
    """The readout answers "which batch, and which project inside it?" — a
    stale batch table beside an unrelated Project is an invitation to run a
    batch nobody is looking at."""
    other = tmp_path / "elsewhere"
    _write(other / "project.yaml", {"name": "other", "design": DESIGN})
    _member(other / "rep1")
    from pyflic.base.project import Project

    hub._set_project(Project(other))
    hub.refresh()
    assert hub._batch_view_root() is None
    assert hub.batch_table.rowCount() == 0


def test_drilling_into_a_project_keeps_its_batch(hub):
    row = _column(hub.batch_table, 0).index("Sept2026/ProjA")
    hub._batch_row_activated(hub.batch_table.item(row, 0))
    hub.refresh()
    assert hub._batch_view_root() is not None
    assert "Batch" in hub.readout.status_text()
    assert "Project" in hub.readout.status_text()
