"""The two-row tile ribbon and the three-card Project panel.

Mirrors PyTrackingAnalysis's Hub (their ADR-0012): wide container tiles on
top — Batch · Project · Experiment · Tools — and a collapsible sub-strip of
compact experiment-level subtiles the Experiment tile expands.  The Project
panel stacks three cards: Create/Load, Experiments, Analysis.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import matplotlib  # noqa: E402

matplotlib.use("Agg")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from pyflic.base.hub import (  # noqa: E402
    EXPERIMENT_SUBTILES,
    AnalysisHubWindow,
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class _FakeExperiment:
    """Just enough of an Experiment for the refresh paths the ribbon uses."""

    chamber_layout = "two_well"
    experiment_dir = "."
    experiment_type = None
    config: dict = {}

    def facet_windows(self):
        return []


@pytest.fixture
def hub(app):
    window = AnalysisHubWindow()
    window.show()
    yield window
    window.close()


def _load_fake(hub) -> None:
    hub.experiment = _FakeExperiment()
    hub.experiment_name = "Rep1"
    hub.refresh()


def test_the_ribbon_has_container_tiles_and_subtiles(hub):
    assert set(hub.tiles) == {"batch", "project", "experiment", "tools",
                              "analyze", "qc", "plots", "scripts", "ai"}
    ## The Experiment group tile opens no panel of its own.
    assert set(hub.panels) == {"batch", "project", "tools",
                               "analyze", "qc", "plots", "scripts", "ai"}
    assert EXPERIMENT_SUBTILES == ("qc", "analyze", "plots", "scripts", "ai")
    ## Container tiles are wide; Tools stays a regular chip.
    assert hub.tiles["batch"].maximumWidth() > hub.tiles["tools"].maximumWidth()


def test_the_sub_strip_starts_hidden_and_the_group_tile_inert(hub):
    assert not hub._sub_strip_host.isVisible()
    assert hub.tiles["experiment"].is_dimmed()
    assert not hub.tiles["experiment"].is_clickable()
    hub._toggle_experiment()
    assert not hub._experiment_expanded


def test_a_subtile_panel_is_refused_with_nothing_loaded(hub):
    hub._open_panel("analyze")
    assert hub._open_key is None
    assert not hub._experiment_expanded


def test_the_project_panel_stacks_three_cards(hub):
    titles = [card._title_lbl.text() for card in hub.panels["project"].cards()]
    assert titles == ["Create/Load", "Experiments", "Analysis"]


def test_the_analysis_card_appears_only_with_a_project(hub):
    from pathlib import Path

    assert hub.project_analysis_card.isHidden()
    hub.open_target(str(Path(__file__).parent.parent / "test_experiment"))
    assert hub.project is not None
    assert not hub.project_analysis_card.isHidden()
    assert "test_experiment" in hub.project_summary.text()


def test_loading_a_member_arms_and_expands_the_group(hub):
    _load_fake(hub)
    tile = hub.tiles["experiment"]
    assert not tile.is_dimmed()
    assert tile.is_clickable()
    hub._toggle_experiment()
    assert hub._experiment_expanded
    assert hub._sub_strip_host.isVisible()


def test_the_group_and_a_container_panel_are_never_open_together(hub):
    _load_fake(hub)
    hub._open_panel("project")
    hub._toggle_experiment()
    assert hub._experiment_expanded
    assert hub._open_key is None
    hub._open_panel("analyze")
    assert hub._open_key == "analyze"
    hub._open_panel("project")
    assert hub._open_key == "project"
    assert not hub._experiment_expanded


def test_unloading_folds_the_group_and_closes_its_panel(hub):
    _load_fake(hub)
    hub._open_panel("plots")
    assert hub._experiment_expanded
    hub.experiment = None
    hub.experiment_name = None
    hub.refresh()
    assert not hub._experiment_expanded
    assert hub._open_key is None
    assert hub.tiles["experiment"].is_dimmed()
    assert not hub.tiles["experiment"].is_clickable()


def test_a_finished_load_reveals_the_qc_panel(hub):
    """Loading is a step toward doing something — and QC comes first."""
    _load_fake(hub)
    hub._reveal_qc()
    assert hub._open_key == "qc"
    assert hub._experiment_expanded


def test_the_reveal_never_yanks_away_a_panel_the_user_opened(hub):
    _load_fake(hub)
    hub._open_panel("project")
    hub._reveal_qc()
    assert hub._open_key == "project"


def test_suppress_tabs_governs_only_batch_tasks(hub):
    """Plots are the point of an analysis someone ran by hand; the switch
    exists for Batch Runs, whose tabs run into the hundreds."""
    hub.chk_suppress_tabs.setChecked(True)
    assert not hub._tabs_suppressed()
    hub._suppress_tabs_task = True          # what a Batch Run's _start sets
    assert hub._tabs_suppressed()
    hub.chk_suppress_tabs.setChecked(False)
    assert not hub._tabs_suppressed()


def test_subtiles_are_compact_and_route_summaries_to_the_tooltip(hub):
    tile = hub.tiles["analyze"]
    assert tile.height() == tile.COMPACT_HEIGHT
    tile.set_summary(["ready", "faceted"])
    assert tile.summary_text() == ""          # no summary lines on the chip
    assert "ready" in tile.toolTip()
