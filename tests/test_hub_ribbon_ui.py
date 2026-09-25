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
from pyflic.base.ui.widgets import ActionButton  # noqa: E402


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


def test_a_finished_load_reveals_the_sub_strip_and_opens_nothing(hub):
    """Loading is a step toward doing something, so the strip of what can now
    be done appears — but which of those to open is the user's move."""
    _load_fake(hub)
    hub._reveal_experiment_group()
    assert hub._experiment_expanded
    assert hub._sub_strip_host.isVisible()
    assert hub._open_key is None


def test_the_reveal_never_yanks_away_a_panel_the_user_opened(hub):
    _load_fake(hub)
    hub._open_panel("project")
    hub._reveal_experiment_group()
    assert hub._open_key == "project"
    assert not hub._experiment_expanded


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


# ---------------------------------------------------------------------------
# The Plots card's groups
# ---------------------------------------------------------------------------
#
# A flat column of buttons said every figure on the card was the same kind of
# thing.  Three of them are not: two follow the Metric dropdown, one exists
# only on a two-well layout, and some exist only for one Experiment Type.


class _NamedType:
    def __init__(self, name: str) -> None:
        self.name = name


def _load_fake_of_type(hub, type_name: str | None,
                       layout: str = "two_well") -> None:
    exp = _FakeExperiment()
    exp.experiment_type = None if type_name is None else _NamedType(type_name)
    exp.chamber_layout = layout
    hub.experiment = exp
    hub.experiment_name = "Rep1"
    hub.refresh()


def _group_titled(hub, fragment: str):
    from pyflic.base.ui.widgets import CardGroup

    for group in hub.panels["plots"].findChildren(CardGroup):
        if fragment.lower() in group.title().lower():
            return group
    raise AssertionError(f"no CardGroup titled like {fragment!r}")


def test_the_metric_dropdown_sits_with_the_buttons_it_steers(hub):
    """The Metric box changes two of the card's figures and not the rest, so
    it lives inside their group rather than above the lot."""
    group = _group_titled(hub, "Chosen metric")
    assert hub.plot_metric in group.findChildren(type(hub.plot_metric))
    labels = {b.text() for b in group.findChildren(ActionButton)}
    assert labels == {"Binned time course", "Dot plot"}
    fixed = {b.text() for b in
             _group_titled(hub, "Standard figures").findChildren(ActionButton)}
    assert fixed == {"Feeding summary", "Well A vs B"}


def test_type_specific_figures_are_grouped_and_hidden_for_other_types(hub):
    _load_fake_of_type(hub, "ProgressiveRatio")
    pr = _group_titled(hub, "Progressive Ratio")
    hedonic = _group_titled(hub, "Hedonic")
    assert not pr.isHidden() and hedonic.isHidden()

    _load_fake_of_type(hub, "Hedonic")
    assert not hedonic.isHidden() and pr.isHidden()

    _load_fake_of_type(hub, None)               # a Custom experiment
    assert pr.isHidden() and hedonic.isHidden()


def test_unloading_hides_every_type_specific_group(hub):
    _load_fake_of_type(hub, "ProgressiveRatio")
    hub.experiment = None
    hub.experiment_name = None
    hub.refresh()
    assert _group_titled(hub, "Progressive Ratio").isHidden()
    assert hub._type_analyze_groups["progressive_ratio"].isHidden()
    assert hub._type_qc_groups["optogenetics"].isHidden()


_PR_QC = {"Light QC table", "Licks per light event (QC)",
          "Sucrose Well resting level (QC)"}
_OPTO_QC = {"Opto light QC table", "Light explained by licks (QC)"}


def _shown(group) -> set[str]:
    return {b.text() for b in group.findChildren(ActionButton) if not b.isHidden()}


def test_progressive_ratio_light_qc_lives_on_the_qc_card(hub):
    """The light QC table and its two figures are QC, so they sit in the QC
    card's Optogenetics group; Analyze keeps the result table, Plots the
    result figures."""
    qc = hub._type_qc_groups["optogenetics"]
    assert qc.title() == "Optogenetics"
    assert qc in hub.panels["qc"].findChildren(type(qc))
    assert {b.text() for b in qc.findChildren(ActionButton)} == _PR_QC | _OPTO_QC
    analyze = hub._type_analyze_groups["progressive_ratio"]
    assert {b.text() for b in analyze.findChildren(ActionButton)} == {
        "Paired − yoked difference CSV", "Breaking point CSV"}
    plots = {b.text() for b in _group_titled(hub, "Progressive Ratio")
             .findChildren(ActionButton)}
    assert plots.isdisjoint({"Light QC table", "Licks per light event (QC)",
                             "Sucrose Well resting level (QC)"})

    _load_fake_of_type(hub, "ProgressiveRatio")
    assert not qc.isHidden() and not analyze.isHidden()
    assert _shown(qc) == _PR_QC            # not optogenetic: no light data here
    _load_fake_of_type(hub, "Hedonic")
    assert qc.isHidden() and analyze.isHidden()


def test_the_optogenetics_group_follows_the_member_not_the_type(hub):
    """Any optogenetic member gets the opto light QC; Progressive Ratio adds
    its own light checks to the same group."""
    qc = hub._type_qc_groups["optogenetics"]
    _load_fake_of_type(hub, None)
    assert qc.isHidden()
    hub.experiment.is_optogenetic = True
    hub.refresh()
    assert not qc.isHidden() and _shown(qc) == _OPTO_QC
    _load_fake_of_type(hub, "ProgressiveRatio")
    hub.experiment.is_optogenetic = True
    hub.refresh()
    assert _shown(qc) == _OPTO_QC | _PR_QC


def test_the_breaking_point_buttons_run_their_actions(hub, monkeypatch):
    """The breaking point table sits with the difference table on Analyze;
    its still-responding curve sits with the other result figures (ADR-0014)."""
    sent: list[dict] = []
    monkeypatch.setattr(hub, "_run_experiment_action", sent.append)
    _load_fake_of_type(hub, "ProgressiveRatio")
    analyze = {b.text(): b for b in
               hub._type_analyze_groups["progressive_ratio"].findChildren(ActionButton)}
    analyze["Breaking point CSV"].click()
    plots = {b.text(): b for b in
             hub._type_plot_groups["progressive_ratio"].findChildren(ActionButton)}
    plots["Still-responding curve"].click()
    assert [step["action"] for step in sent] == ["breaking_point",
                                                 "plot_pr_still_responding"]


def test_the_qc_card_buttons_run_their_actions(hub, monkeypatch):
    sent: list[dict] = []
    monkeypatch.setattr(hub, "_run_experiment_action", sent.append)
    _load_fake_of_type(hub, "ProgressiveRatio")
    buttons = {b.text(): b for b in
               hub._type_qc_groups["optogenetics"].findChildren(ActionButton)}
    for label in ("Light QC table", "Licks per light event (QC)",
                  "Sucrose Well resting level (QC)", "Opto light QC table",
                  "Light explained by licks (QC)"):
        buttons[label].click()
    assert [step["action"] for step in sent] == [
        "pr_light_qc", "plot_pr_light_events", "plot_pr_resting_level",
        "opto_light_qc", "plot_opto_light"]


def test_the_pr_curve_buttons_use_the_bin_size_spinbox(hub, monkeypatch):
    sent: list[dict] = []
    monkeypatch.setattr(hub, "_run_experiment_action", sent.append)
    _load_fake_of_type(hub, "ProgressiveRatio")
    hub.spin_binsize.setValue(5.0)
    hub._run_plot_action("plot_pr_cumulative_licks")
    hub._run_plot_action("plot_pr_light_events")
    assert sent[0] == {"action": "plot_pr_cumulative_licks", "binsize": 5.0}
    assert sent[1] == {"action": "plot_pr_light_events"}


def test_well_comparison_is_offered_only_on_a_two_well_layout(hub):
    _load_fake_of_type(hub, None, layout="two_well")
    button, = hub._two_well_plot_buttons
    assert not button.isHidden()
    _load_fake_of_type(hub, None, layout="single_well")
    assert button.isHidden()


def test_the_subtile_panels_do_not_repeat_the_loaded_member(hub):
    """The Experiment tile these panels hang from already names the member,
    and so does the status strip; the QC line keeps only what is true of that
    card — whether there is anything on disk to look at."""
    _load_fake_of_type(hub, "ProgressiveRatio")
    assert not hasattr(hub, "analyze_hint")
    assert not hasattr(hub, "plots_hint")
    assert "Rep1" not in hub.qc_hint.text()
    assert "QC reports" in hub.qc_hint.text()


def _help_refs(widget) -> list[str]:
    from pyflic.help.button import HelpButton

    return [b.ref for b in widget.findChildren(HelpButton)]


def test_general_help_sits_after_clear_errors(hub):
    """The dock's corner ends with a help button opening the start page, so
    help is reachable without opening a panel first."""
    corner = hub.dock.cornerWidget()
    lay = corner.layout()
    last = lay.itemAt(lay.count() - 1).widget()
    before = lay.itemAt(lay.count() - 2).widget()
    assert _help_refs(corner) == ["getting-started"]
    assert last.ref == "getting-started"
    assert before.text() == "Clear Errors"


def test_every_project_panel_card_has_help(hub):
    from pyflic.base.ui.widgets import Card

    cards = hub.panels["project"].findChildren(Card)
    assert [c.title() for c in cards][:3] == ["Create/Load", "Experiments", "Analysis"]
    for card in cards[:3]:
        assert _help_refs(card), f"no help button on the {card.title()} card"


def test_type_groups_carry_their_own_help(hub):
    for key, requires in AnalysisHubWindow._TYPE_GROUP_HELP:
        group = hub._type_groups(key)[requires]
        assert _help_refs(group) == [AnalysisHubWindow._TYPE_GROUP_HELP[(key, requires)]]


def _beside(group):
    """``(button, help)`` sharing the group's help row, else ``None``."""
    from pyflic.help.button import HelpButton

    helps = group.findChildren(HelpButton)
    if len(helps) != 1:
        return None
    row = helps[0].parentWidget()
    buttons = [w for w in row.findChildren(ActionButton) if w.parentWidget() is row]
    return (buttons[0], helps[0]) if len(buttons) == 1 else None


def test_type_group_help_sits_beside_the_first_button(hub):
    """A help row of its own opened an empty band under each group's title;
    the ``?`` shares the first button's row instead, at its right."""
    _load_fake_of_type(hub, "ProgressiveRatio")
    for key, requires in AnalysisHubWindow._TYPE_GROUP_HELP:
        group = hub._type_groups(key)[requires]
        pair = _beside(group)
        assert pair is not None, (key, requires)
        button, _help = pair
        ## In layout order, the first button not hidden on purpose.
        from pyflic.base.ui.widgets import _hidden_by_hand

        shown = [m for m in group._members if not _hidden_by_hand(m)]
        assert button is shown[0], (key, requires, button.text())


def test_the_optogenetics_help_follows_the_first_button_shown(hub):
    qc = hub._type_qc_groups["optogenetics"]
    _load_fake_of_type(hub, "ProgressiveRatio")          # not optogenetic
    assert _beside(qc)[0].text() == "Light QC table"
    hub.experiment.is_optogenetic = True
    hub.refresh()
    assert _beside(qc)[0].text() == "Opto light QC table"
    _load_fake_of_type(hub, "ProgressiveRatio")
    assert _beside(qc)[0].text() == "Light QC table"
    ## Every button kept the visibility the refresh gave it.
    assert _shown(qc) == _PR_QC


def test_a_card_group_places_its_title_widget_without_a_row_of_its_own(app):
    from PyQt6.QtWidgets import QLabel, QWidget

    from pyflic.base.ui.widgets import CardGroup

    noted = CardGroup("Noted", note="a note")
    noted.add(ActionButton("first"))
    help_ = QLabel("?")
    noted.add_title_widget(help_)
    ## With a note, the widget shares the note's row, as before.
    assert noted._body.itemAt(0).layout() is not None
    assert noted._title_member is None
    plain = CardGroup("Plain")
    first, second = ActionButton("first"), ActionButton("second")
    plain.add(first)
    plain.add(second)
    plain.add_title_widget(QLabel("?"))
    assert plain._title_member is first
    assert isinstance(plain._body.itemAt(0).widget(), QWidget)
    ## Hide the first member: the widget moves to the second, and the first
    ## stays hidden where it was.
    first.setVisible(False)
    plain.reflow_title_widget()
    assert plain._title_member is second
    assert plain._body.itemAt(0).widget() is first and first.isHidden()
    from pyflic.base.ui.widgets import _hidden_by_hand

    assert _hidden_by_hand(first) and not _hidden_by_hand(second)
    empty = CardGroup("Empty")
    empty.add_title_widget(QLabel("?"))              # nothing to sit beside yet
    assert empty._body.itemAt(0).layout() is not None
