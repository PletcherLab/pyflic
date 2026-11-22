"""The two-row tile ribbon and the three-card Project panel.

Mirrors PyTrackingAnalysis's Hub (their ADR-0012): wide container tiles on
top — Batch · Project · Experiment · Tools — and a collapsible sub-strip of
compact experiment-level subtiles the Experiment tile expands.  The Project
panel stacks three cards: Create/Load, Experiments, Analysis.  The QC, Analyze
and Plots panels are a Selection over one button each (ADR-0018).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import matplotlib  # noqa: E402

matplotlib.use("Agg")

from PyQt6.QtWidgets import QApplication, QCheckBox, QLabel, QMessageBox  # noqa: E402

from pyflic.base import experiment_types as et  # noqa: E402
from pyflic.base import selections as sel  # noqa: E402
from pyflic.base.hedonic_experiment import HedonicFeedingExperiment  # noqa: E402
from pyflic.base.hub import (  # noqa: E402
    EXPERIMENT_SUBTILES,
    AnalysisHubWindow,
)
from pyflic.base.progressive_ratio_experiment import ProgressiveRatioExperiment  # noqa: E402
from pyflic.base.single_well_experiment import SingleWellExperiment  # noqa: E402
from pyflic.base.two_well_experiment import TwoWellExperiment  # noqa: E402
from pyflic.base.ui.widgets import ActionButton, CardGroup  # noqa: E402


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


def test_the_analysis_card_appears_only_with_a_project(hub, tmp_path):
    ## A Project of its own: the repo's test_experiment/ is local sample data,
    ## ignored by git, so a fresh checkout or worktree has none, and a test
    ## must not open the developer's own copy in a window either.
    project = tmp_path / "SampleProject"
    _write_yaml(project / "project.yaml", {"name": "SampleProject"})
    _write_yaml(project / "Rep1" / "flic_config.yaml",
                {"global": {"chamber_layout": "two_well"},
                 "dfms": [{"id": 1, "chambers": {1: "Ctrl"}}]})

    assert hub.project_analysis_card.isHidden()
    hub.open_target(str(project))
    assert hub.project is not None
    assert not hub.project_analysis_card.isHidden()
    assert "SampleProject" in hub.project_summary.text()


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
# The QC, Analyze and Plots panels: a Selection over one button (ADR-0018)
# ---------------------------------------------------------------------------
#
# Each panel is a checkbox per item of its set, in a group box per
# selections.groups(kind), over the one button that runs them.  The fakes
# below are of the real classes, because the sets' requirements test the
# layout and the type by class, as the script runner gates the actions.


class _MemberBase:
    """Just enough of a loaded member for the refresh paths, on the class a
    member of that layout and type is loaded as."""

    chamber_layout = "two_well"
    experiment_dir = "."
    experiment_type = None
    is_optogenetic = False
    qc_dir = None
    #: What the data can answer, read by the stand-ins for
    #: analytics.yoked_groups and program_timeline.timeline_dfms below.
    yoked = True
    stub_yoked_groups: list = []
    stub_timeline_dfms: list = []

    def __init__(self) -> None:
        self.config = {}
        self.global_config = {}

    def facet_windows(self):
        return []

    def facet_labels(self):
        return []

    def has_yoked_controls(self):
        return self.yoked


_CLASSES = {"two_well": TwoWellExperiment, "single_well": SingleWellExperiment,
            "progressive_ratio": ProgressiveRatioExperiment,
            "hedonic": HedonicFeedingExperiment}
_FAKES = {key: type(f"_Fake{cls.__name__}", (_MemberBase, cls), {})
          for key, cls in _CLASSES.items()}
_TYPE_NAMES = {"progressive_ratio": "ProgressiveRatio", "hedonic": "Hedonic"}


@pytest.fixture(autouse=True)
def _stub_data(monkeypatch):
    """What the data can answer comes from the fake: whether a DFM has a
    Program timeline and whether a linkage group lights a Yoked chamber."""
    from pyflic.base import analytics, program_timeline

    monkeypatch.setattr(analytics, "yoked_groups",
                        lambda exp: list(getattr(exp, "stub_yoked_groups", [])))
    monkeypatch.setattr(program_timeline, "timeline_dfms",
                        lambda exp: list(getattr(exp, "stub_timeline_dfms", [])))


def _member(hub, kind: str = "two_well", *, optogenetic: bool = False,
            directory=".", name: str = "Rep1", omit: dict | None = None,
            yoked: bool = True, yoked_groups=(), timeline=()):
    """Load a fake member of *kind* (a layout, or a type on two wells)."""
    exp = _FAKES[kind]()
    exp.chamber_layout = "single_well" if kind == "single_well" else "two_well"
    ## The shipped type its modules come from (ADR-0019): Custom for a layout.
    exp.experiment_type = et.get_experiment_type(_TYPE_NAMES.get(kind))
    exp.is_optogenetic = optogenetic
    exp.experiment_dir = str(directory)
    exp.yoked = yoked
    exp.stub_yoked_groups = list(yoked_groups)
    exp.stub_timeline_dfms = list(timeline)
    if omit:
        exp.global_config[sel.OMIT_KEY] = {k: list(v) for k, v in omit.items()}
    hub.experiment = exp
    hub.experiment_name = name
    hub.refresh()
    return exp


def _load_fake_of_type(hub, type_name: str | None,
                       layout: str = "two_well") -> None:
    kind = {"ProgressiveRatio": "progressive_ratio", "Hedonic": "hedonic"}.get(
        type_name, layout)
    _member(hub, kind)


def _groups(hub, kind: str) -> dict[str, list[str]]:
    """``{group title: [item ids]}`` as the panel lays them out."""
    layout = hub._selection_hosts[kind].layout()
    shown = [layout.itemAt(i).widget() for i in range(layout.count())]
    boxes = hub._selection_boxes[kind]
    titles = {id(g): t for t, g in hub._selection_groups[kind].items()}
    return {titles[id(g)]: [i for i, box in boxes.items() if g.isAncestorOf(box)]
            for g in shown if isinstance(g, CardGroup)}


def _box(hub, kind: str, item_id: str) -> QCheckBox:
    return hub._selection_boxes[kind][item_id]


def _not_offered(hub, kind: str) -> QLabel | None:
    for label in hub._selection_hosts[kind].findChildren(QLabel):
        if label.text().startswith("Not offered"):
            return label
    return None


_QC_REPORTS = ["integrity", "data_breaks", "raw_signal", "baselined_signal",
               "cumulative_licks", "simultaneous_feeding", "bleeding"]
_ONE_WELL_QC = [i for i in _QC_REPORTS if i not in ("simultaneous_feeding", "bleeding")]
_OPTO_QC = ["opto_light_qc", "opto_light_figures", "program_timeline"]
_PR_QC = ["pr_light_qc", "pr_light_event_figures", "pr_resting_level_figures"]
_PR_ANALYSES = ["paired_yoked_diff", "cumulative_diff", "breaking_point"]
_PR_PLOTS = ["pr_cumulative_diff_plot", "pr_cumulative_licks", "pr_still_responding",
             "pr_breaking_point_plots"]
_CHOSEN = ["binned_time_course", "dot_plot"]

#: (kind, optogenetic) → what each panel offers, group by group, in order.
_OFFERED = {
    ("two_well", False): {
        sel.QC: {"QC reports": _QC_REPORTS},
        sel.ANALYSES: {"Analyses": ["binned_summary", "event_statistics"],
                       "Report": ["experiment_report"]},
        sel.PLOTS: {"Figures": ["feeding_summary_plot", "well_comparison"],
                    "Chosen metric": _CHOSEN},
    },
    ("single_well", False): {
        sel.QC: {"QC reports": _ONE_WELL_QC},
        sel.ANALYSES: {"Analyses": ["binned_summary", "event_statistics"],
                       "Report": ["experiment_report"]},
        sel.PLOTS: {"Figures": ["feeding_summary_plot"], "Chosen metric": _CHOSEN},
    },
    ("single_well", True): {
        sel.QC: {"QC reports": _ONE_WELL_QC, "Optogenetics": _OPTO_QC},
        sel.ANALYSES: {"Analyses": ["binned_summary", "event_statistics"],
                       "Optogenetics": ["light_tables", "yoked_comparison"],
                       "Report": ["experiment_report"]},
        sel.PLOTS: {"Figures": ["feeding_summary_plot"], "Chosen metric": _CHOSEN},
    },
    ("progressive_ratio", False): {
        sel.QC: {"QC reports": _QC_REPORTS, "Progressive Ratio": _PR_QC},
        sel.ANALYSES: {"Analyses": ["binned_summary", "event_statistics"],
                       "Progressive Ratio": _PR_ANALYSES,
                       "Report": ["experiment_report"]},
        sel.PLOTS: {"Figures": ["feeding_summary_plot", "well_comparison"],
                    "Chosen metric": _CHOSEN, "Progressive Ratio": _PR_PLOTS},
    },
    ## Progressive Ratio keeps its own Paired-Yoked Difference, so the
    ## optogenetic yoked comparison is not offered under it.
    ("progressive_ratio", True): {
        sel.QC: {"QC reports": _QC_REPORTS, "Optogenetics": _OPTO_QC,
                 "Progressive Ratio": _PR_QC},
        sel.ANALYSES: {"Analyses": ["binned_summary", "event_statistics"],
                       "Optogenetics": ["light_tables"],
                       "Progressive Ratio": _PR_ANALYSES,
                       "Report": ["experiment_report"]},
        sel.PLOTS: {"Figures": ["feeding_summary_plot", "well_comparison"],
                    "Chosen metric": _CHOSEN, "Progressive Ratio": _PR_PLOTS},
    },
    ("hedonic", False): {
        sel.QC: {"QC reports": _QC_REPORTS},
        sel.ANALYSES: {"Analyses": ["binned_summary", "event_statistics"],
                       "Hedonic": ["weighted_duration"],
                       "Report": ["experiment_report"]},
        sel.PLOTS: {"Figures": ["feeding_summary_plot", "well_comparison"],
                    "Chosen metric": _CHOSEN, "Hedonic": ["hedonic_plot"]},
    },
}


@pytest.mark.parametrize("kind,optogenetic", list(_OFFERED))
def test_each_panel_offers_the_groups_and_boxes_its_member_can_have(hub, kind, optogenetic):
    _member(hub, kind, optogenetic=optogenetic)
    for set_kind, expected in _OFFERED[(kind, optogenetic)].items():
        assert _groups(hub, set_kind) == expected, set_kind
        for item_id, box in hub._selection_boxes[set_kind].items():
            item = sel.item(set_kind, item_id)
            assert box.text() == item.label
            assert box.toolTip().startswith(item.caption)


def test_each_panel_runs_its_set_with_one_primary_button(hub):
    _member(hub)
    for key, kind, button in (("qc", sel.QC, hub.run_qc_btn),
                              ("analyze", sel.ANALYSES, hub.run_analysis_btn),
                              ("plots", sel.PLOTS, hub.generate_plots_btn)):
        card = hub.panels[key].cards()[0]
        assert [b.text() for b in card.findChildren(ActionButton)] == [sel.RUN_NAMES[kind]]
        assert button.text() == sel.RUN_NAMES[kind]
        ## Underneath the groups: the last thing on the card.
        body = card.body_layout()
        assert body.itemAt(body.count() - 1).widget() is button
        assert body.indexOf(hub._selection_hosts[kind]) == body.count() - 2


def test_the_analyze_panel_puts_the_bin_size_first_and_says_the_core_always_runs(hub):
    body = hub.panels["analyze"].cards()[0].body_layout()
    row = body.itemAt(0).layout()
    assert any(row.itemAt(i).widget() is hub.spin_binsize for i in range(row.count()))
    note, core = (body.itemAt(i).widget().text() for i in (1, 2))
    assert "binned summary" in note and "report" in note and "only" in note
    assert core.startswith("The core always runs")
    assert "feeding summary" in core and "summary.txt" in core


def test_an_item_the_member_cannot_have_is_not_offered_and_one_line_says_so(hub):
    _member(hub, "single_well")
    assert "simultaneous_feeding" not in hub._selection_boxes[sel.QC]
    note = _not_offered(hub, sel.QC)
    assert note.text() == ("Not offered for this member: Simultaneous feeding, "
                           "Bleeding between wells, Optogenetics, Progressive Ratio.")
    assert ("Simultaneous feeding, Bleeding between wells: needs a two-well chamber "
            "layout.") in note.toolTip().splitlines()
    _member(hub)                                # two wells, Custom, no light
    assert _not_offered(hub, sel.PLOTS).text() == (
        "Not offered for this member: Progressive Ratio, Hedonic.")
    _member(hub, "progressive_ratio", optogenetic=True)
    assert _not_offered(hub, sel.ANALYSES).text() == (
        "Not offered for this member: Paired and yoked comparison, Hedonic.")


def test_an_item_the_data_cannot_answer_is_greyed_with_its_reason(hub):
    exp = _member(hub, optogenetic=True)        # no timeline, no yoked group
    for kind, item_id, start in ((sel.QC, "program_timeline", "No timeline:"),
                                 (sel.ANALYSES, "yoked_comparison", "No yoked group:")):
        item = sel.item(kind, item_id)
        reason = sel.unavailable(item, exp)
        assert reason.startswith(start)
        box = _box(hub, kind, item_id)
        assert not box.isEnabled()
        assert box.toolTip() == f"{item.caption}\n\nNot available for this member: {reason}."
    ## The ticks still show the Selection: greyed is not unticked.
    assert _box(hub, sel.QC, "program_timeline").isChecked()

    _member(hub, optogenetic=True, timeline=[1], yoked_groups=[object()])
    for kind, item_id in ((sel.QC, "program_timeline"), (sel.ANALYSES, "yoked_comparison")):
        box = _box(hub, kind, item_id)
        assert box.isEnabled() and box.toolTip() == sel.item(kind, item_id).caption

    _member(hub, "progressive_ratio", yoked=False)
    box = _box(hub, sel.ANALYSES, "paired_yoked_diff")
    assert not box.isEnabled()
    assert box.toolTip().endswith(
        "No yoked chamber: every chamber group is a solo group.")
    assert all(b.isEnabled() for i, b in hub._selection_boxes[sel.ANALYSES].items()
               if i != "paired_yoked_diff")


def test_a_box_reflects_the_configs_omit_on_load(hub):
    _member(hub, omit={sel.ANALYSES: ["event_statistics"], sel.PLOTS: ["dot_plot"],
                       sel.QC: ["bleeding"]})
    assert not _box(hub, sel.ANALYSES, "event_statistics").isChecked()
    assert _box(hub, sel.ANALYSES, "binned_summary").isChecked()
    assert not _box(hub, sel.PLOTS, "dot_plot").isChecked()
    assert not _box(hub, sel.QC, "bleeding").isChecked()
    assert sum(not b.isChecked() for kind in sel.OMIT_KINDS
               for b in hub._selection_boxes[kind].values()) == 3
    ## Another member, another Selection: the panels are rebuilt from it.
    _member(hub)
    assert all(b.isChecked() for kind in sel.OMIT_KINDS
               for b in hub._selection_boxes[kind].values())


def test_a_refresh_keeps_the_same_members_boxes(hub):
    """Refresh runs after every task; only a load, an unload or a reload
    (a new experiment object) rebuilds the panels."""
    _member(hub)
    box = _box(hub, sel.QC, "integrity")
    hub.refresh()
    assert _box(hub, sel.QC, "integrity") is box


def test_a_panel_rebuilt_while_hidden_measures_its_new_groups(hub):
    """The panels are hidden while a load rebuilds them; a panel that opened
    at the previous member's height had its space spread through the cards."""
    panel = hub.panels["qc"]
    _member(hub, "progressive_ratio", optogenetic=True)
    tall = panel._content_height()
    _member(hub, "single_well")
    short = panel._content_height()
    _member(hub, "progressive_ratio", optogenetic=True)
    assert short < tall == panel._content_height()


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _last_line(log) -> str:
    return log.toPlainText().splitlines()[-1]


def test_ticking_writes_a_standalone_experiments_flic_config(hub, tmp_path):
    member = tmp_path / "rep1"
    config = member / "flic_config.yaml"
    _write_yaml(config, {"global": {"chamber_layout": "two_well"}, "dfms": []})
    exp = _member(hub, directory=member, name="rep1")

    _box(hub, sel.ANALYSES, "event_statistics").setChecked(False)
    written = _read_yaml(config)
    assert written["global"] == {"chamber_layout": "two_well",
                                 "omit": {"analyses": ["event_statistics"]}}
    assert sel.omitted(exp, sel.ANALYSES) == {"event_statistics"}
    assert _last_line(hub.log) == (
        "[selection] Event statistics: left out of every run of rep1 "
        f"(global.omit.analyses in {config})")

    _box(hub, sel.ANALYSES, "event_statistics").setChecked(True)
    assert _read_yaml(config)["global"] == {"chamber_layout": "two_well"}
    assert sel.omitted(exp, sel.ANALYSES) == frozenset()
    assert _last_line(hub.log).startswith(
        "[selection] Event statistics: included in every run of rep1 ")


def test_ticking_a_members_box_writes_its_projects_design(hub, tmp_path):
    project = tmp_path / "ProjA"
    _write_yaml(project / "project.yaml",
                {"name": "ProjA", "design": {"global": {"chamber_layout": "two_well"}}})
    member_config = project / "rep1" / "flic_config.yaml"
    _write_yaml(member_config, {"dfms": []})
    _member(hub, directory=project / "rep1", name="rep1")

    _box(hub, sel.QC, "bleeding").setChecked(False)
    design = _read_yaml(project / "project.yaml")["design"]
    assert design["global"]["omit"] == {"qc": ["bleeding"]}
    assert "global" not in _read_yaml(member_config)
    line = _last_line(hub.log)
    assert line == ("[selection] Bleeding between wells: left out of every member's "
                    "runs in Project ProjA, not only rep1's (design.global.omit.qc in "
                    f"{project / 'project.yaml'})")


def test_a_member_of_a_project_without_a_design_is_told_and_its_box_put_back(
        hub, tmp_path, monkeypatch):
    project = tmp_path / "ProjB"
    _write_yaml(project / "project.yaml", {"name": "ProjB"})
    _write_yaml(project / "rep1" / "flic_config.yaml", {"dfms": []})
    warned: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *args, **_kw: warned.append(args[2]))
    _member(hub, directory=project / "rep1", name="rep1")

    _box(hub, sel.QC, "bleeding").setChecked(False)          # must not raise
    assert len(warned) == 1 and "design: global:" in warned[0]
    assert "Bleeding between wells: not saved" in hub.errors.toPlainText()
    assert "design" not in _read_yaml(project / "project.yaml")
    for _ in range(5):                    # the deferred rebuild
        QApplication.processEvents()
    assert _box(hub, sel.QC, "bleeding").isChecked()


def test_each_run_button_runs_its_step_with_the_panels_bin_size_and_metric(
        hub, monkeypatch):
    sent: list[tuple[dict, dict]] = []
    monkeypatch.setattr(hub, "_run_experiment_action",
                        lambda step, **kw: sent.append((step, kw)))
    _member(hub)
    hub.spin_binsize.setValue(5.0)
    hub.plot_metric.setCurrentIndex(hub.plot_metric.findData("Events"))
    for button in (hub.run_qc_btn, hub.run_analysis_btn, hub.generate_plots_btn):
        button.click()
    assert [step for step, _kw in sent] == [
        {"action": "run_qc"},
        {"action": "run_analysis", "binsize": 5.0},
        {"action": "run_plots", "metric": "Events", "binsize": 5.0},
    ]
    assert [kw["label"] for _s, kw in sent] == ["Run QC", "Run analysis", "Generate plots"]
    assert [kw["replace_tabs"] for _s, kw in sent] == [False, False, True]
    from pyflic.base.script_editor.actions import ACTIONS

    known = {a.action for a in ACTIONS}
    assert {step["action"] for step, _kw in sent} <= known


def _finish(hub) -> None:
    """Wait for the worker, then deliver the result it queued."""
    worker = hub._worker
    assert worker is not None and worker.wait(30000)
    for _ in range(20):
        QApplication.processEvents()


def test_generate_plots_opens_its_figures_replacing_tabs_of_the_same_title(
        hub, monkeypatch):
    from matplotlib.figure import Figure

    from pyflic.base.script_editor import runner

    scripts: list[dict] = []

    def figure() -> Figure:
        fig = Figure(figsize=(2, 2))
        fig.add_subplot().plot([0, 1], [0, 1])
        return fig

    def run(exp, script, *, context=None):
        scripts.append(script)
        return [("Feeding Summary", figure()), ("Dot: Licks", figure())]

    monkeypatch.setattr(runner, "run_experiment_script", run)
    _member(hub)
    for _ in range(2):
        hub.generate_plots_btn.click()
        _finish(hub)
    titles = [hub.dock.tabText(i) for i in range(hub.dock.count())]
    assert titles.count("Feeding Summary") == 1 and titles.count("Dot: Licks") == 1
    assert scripts[0]["steps"] == [{"action": "run_plots",
                                    "metric": hub.plot_metric.currentData(),
                                    "binsize": hub.spin_binsize.value()}]


def test_the_metric_menu_sits_in_the_chosen_metric_group_and_outlives_a_rebuild(hub):
    """The Metric menu steers the Chosen metric group's two figures and no
    other, so it lives inside that group; the group is rebuilt for each
    member, and the menu, and the metric chosen, carry over."""
    _member(hub)
    group = hub._selection_groups[sel.PLOTS]["Chosen metric"]
    combo = hub.plot_metric
    assert group.isAncestorOf(combo)
    assert not any(g.isAncestorOf(combo) for t, g in
                   hub._selection_groups[sel.PLOTS].items() if t != "Chosen metric")
    combo.setCurrentIndex(combo.findData("Events"))
    _member(hub, "hedonic")
    rebuilt = hub._selection_groups[sel.PLOTS]["Chosen metric"]
    assert rebuilt is not group and rebuilt.isAncestorOf(combo)
    assert hub.plot_metric is combo and combo.currentData() == "Events"
    assert not hub._metric_row.isHidden()


def test_unloading_empties_the_three_panels(hub):
    _member(hub, "progressive_ratio", optogenetic=True)
    hub.experiment = None
    hub.experiment_name = None
    hub.refresh()
    for kind in sel.OMIT_KINDS:
        assert hub._selection_groups[kind] == {} and hub._selection_boxes[kind] == {}
        texts = [label.text() for label in
                 hub._selection_hosts[kind].findChildren(QLabel)]
        assert texts == [f"Load a member to see what {sel.RUN_NAMES[kind]} "
                         "can write for it."]


def test_the_view_card_keeps_the_buttons_that_run_nothing(hub, tmp_path, monkeypatch):
    view = hub.qc_view_card
    assert [c.title() for c in hub.panels["qc"].cards()] == ["QC", "View"]
    assert [b.text() for b in view.findChildren(ActionButton)] == [
        "Open QC Viewer", "View QC plots", "Open qc folder"]
    assert hub.qc_hint in view.findChildren(QLabel)

    import pyflic.base.qc_viewer as qc_viewer

    opened: list = []

    class _Viewer:
        def __init__(self, directory, experiment=None):
            opened.append((directory, experiment))

        def show(self):
            pass

    monkeypatch.setattr(qc_viewer, "MainWindow", _Viewer)
    monkeypatch.setattr(hub, "_open_externally", opened.append)
    exp = _member(hub, directory=tmp_path)
    (tmp_path / "qc").mkdir()
    buttons = {b.text(): b for b in view.findChildren(ActionButton)}
    buttons["Open QC Viewer"].click()
    buttons["Open qc folder"].click()
    assert opened == [(tmp_path, exp), tmp_path / "qc"]


def test_analyze_all_says_it_keeps_basic_analysis_and_ignores_the_ticks(hub):
    button, = [b for b in hub.project_analysis_card.findChildren(ActionButton)
               if b.text() == "Analyze all"]
    tip = button.toolTip()
    assert "Basic analysis" in tip and "does not read" in tip and "removes no" in tip


def test_the_readout_says_when_a_member_has_no_training_information(hub):
    """A member recorded from DFM firmware older than 4.0.0 (generation 2)
    says so on the readout's member line (decision 15)."""
    from types import SimpleNamespace

    from pyflic.base.light_summary import NO_TRAINING_INFORMATION

    _load_fake(hub)
    assert NO_TRAINING_INFORMATION not in hub.readout.status_text()
    hub.experiment.dfms = {1: SimpleNamespace(training_info="phase"),
                           2: SimpleNamespace(training_info="none")}
    hub.refresh()
    assert f"Loaded: Rep1 · {NO_TRAINING_INFORMATION}" in hub.readout.status_text()


def test_view_qc_plots_opens_the_program_timeline(hub, tmp_path):
    """The Program timeline a run saved in qc/program/ opens beside the
    signal plots and the light QC's figure."""
    from PyQt6.QtGui import QColor, QPixmap

    from pyflic.base.program_timeline import timeline_path

    _load_fake(hub)
    hub.experiment.experiment_dir = str(tmp_path)
    hub.experiment.dfms = {1: None, 2: None}
    pixmap = QPixmap(8, 8)
    pixmap.fill(QColor("white"))
    for png in (timeline_path(tmp_path / "qc", 1),
                tmp_path / "qc" / "opto" / "opto_light_dfm1.png"):
        png.parent.mkdir(parents=True, exist_ok=True)
        assert pixmap.save(str(png), "PNG")
    button, = [b for b in hub.qc_view_card.findChildren(ActionButton)
               if b.text() == "View QC plots"]
    button.click()
    titles = [hub.dock.tabText(i) for i in range(hub.dock.count())]
    assert "DFM1 Program timeline" in titles and "DFM1 Light explained" in titles
    assert not any(t.startswith("DFM2") for t in titles)


def test_the_subtile_panels_do_not_repeat_the_loaded_member(hub):
    """The Experiment tile these panels hang from already names the member,
    and so does the status strip; the QC line keeps only what is true of that
    card — whether there is anything on disk to look at."""
    _load_fake_of_type(hub, "ProgressiveRatio")
    assert not hasattr(hub, "analyze_hint")
    assert not hasattr(hub, "plots_hint")
    assert "Rep1" not in hub.qc_hint.text()
    assert "Run QC" in hub.qc_hint.text()


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


def test_every_selection_card_has_help(hub):
    for key in ("qc", "analyze", "plots"):
        for card in hub.panels[key].cards():
            assert _help_refs(card)[:1], f"no help button on the {card.title()} card"
    assert _help_refs(hub.qc_view_card) == ["app-qc-viewer"]


def test_every_group_of_the_three_sets_has_a_help_topic():
    assert set(AnalysisHubWindow._GROUP_HELP) == {
        (kind, title) for kind in sel.OMIT_KINDS for title in sel.groups(kind)}


def _beside(group):
    """``(member, help)`` sharing the group's help row, else ``None``."""
    from PyQt6.QtWidgets import QWidget

    from pyflic.help.button import HelpButton

    helps = group.findChildren(HelpButton)
    if len(helps) != 1:
        return None
    row = helps[0].parentWidget()
    members = [w for w in row.children()
               if isinstance(w, QWidget) and w is not helps[0]]
    return (members[0], helps[0]) if len(members) == 1 else None


@pytest.mark.parametrize("kind,optogenetic", [("progressive_ratio", True),
                                              ("hedonic", False),
                                              ("single_well", True)])
def test_each_group_box_carries_its_help_beside_its_first_member(hub, kind, optogenetic):
    """A help row of its own opened an empty band under each group's title;
    the ``?`` shares the first box's row instead, at its right (the metric
    menu's, in the Chosen metric group)."""
    _member(hub, kind, optogenetic=optogenetic)
    for set_kind in sel.OMIT_KINDS:
        assert hub._selection_groups[set_kind]
        for title, group in hub._selection_groups[set_kind].items():
            assert _help_refs(group) == [AnalysisHubWindow._GROUP_HELP[(set_kind, title)]]
            pair = _beside(group)
            assert pair is not None, (set_kind, title)
            assert pair[0] is group._members[0], (set_kind, title)
    assert _beside(hub._selection_groups[sel.PLOTS]["Chosen metric"])[0] is hub._metric_row


def test_a_card_group_places_its_title_widget_without_a_row_of_its_own(app):
    from PyQt6.QtWidgets import QWidget

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


# ---------------------------------------------------------------------------
# The panels' muted notes stay readable in either theme
# ---------------------------------------------------------------------------

def _contrast(a, b) -> float:
    def lum(rgb):
        def ch(c):
            c = c / 255.0
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
        return 0.2126 * ch(rgb[0]) + 0.7152 * ch(rgb[1]) + 0.0722 * ch(rgb[2])
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _note_contrasts(hub) -> dict[str, float]:
    """Each muted note of the open Analyze panel: the contrast between its
    text and the box behind it, read from the rendered pixels."""
    from collections import Counter

    notes = ("Used by the binned summary", "The core always runs",
             "Not offered for this member")
    out = {}
    for label in hub.panels["analyze"].findChildren(QLabel):
        if not label.isVisible() or not label.text().startswith(notes):
            continue
        image = label.grab().toImage()
        counts = Counter((image.pixelColor(x, y).red(), image.pixelColor(x, y).green(),
                          image.pixelColor(x, y).blue())
                         for x in range(image.width()) for y in range(image.height()))
        box = counts.most_common(1)[0][0]
        out[label.text()[:40]] = max(_contrast(rgb, box) for rgb in counts)
    return out


def test_the_muted_notes_are_readable_in_the_light_theme_and_after_a_toggle(app, hub):
    """The Analyze panel's notes ("Used by the binned summary…", "The core
    always runs…", "Not offered…") were palette(mid), which qdarktheme leaves
    at one grey for both themes: 1.8:1 on the light theme's band, so they
    read as empty boxes.  They take the family's muted colour now, and the
    theme toggle re-tints them."""
    from pyflic.base.ui import apply_theme

    apply_theme(app, mode="light")
    hub._restyle()
    _member(hub, "progressive_ratio", optogenetic=True)
    hub._open_panel("analyze")
    app.processEvents()
    light = _note_contrasts(hub)
    assert len(light) >= 3, light
    assert min(light.values()) >= 3.0, light
    hub._toggle_theme()                       # to dark
    app.processEvents()
    dark = _note_contrasts(hub)
    assert min(dark.values()) >= 3.0, dark
    hub._toggle_theme()                       # and back
    app.processEvents()
    assert min(_note_contrasts(hub).values()) >= 3.0


# ---------------------------------------------------------------------------
# The Experiment Editor, beside the Config Editor in the Tools panel (ADR-0026)
# ---------------------------------------------------------------------------

def _tool(hub, label: str) -> ActionButton:
    return next(b for b in hub.panels["tools"].findChildren(ActionButton)
                if b.text() == label)


def test_the_tools_panel_lists_the_experiment_editor_beside_the_config_editor(hub):
    labels = [b.text() for b in hub.panels["tools"].findChildren(ActionButton)]
    assert labels.index("Experiment editor") == labels.index("Config editor") + 1


def test_the_experiment_editor_opens_on_the_loaded_members_config(hub, tmp_path,
                                                                 monkeypatch):
    member = tmp_path / "Proj" / "rep1"
    config = member / "flic_config.yaml"
    _write_yaml(config, {"about": {"title": "Replicate 1"},
                         "global": {"chamber_layout": "two_well", "params": {}},
                         "dfms": [{"id": 1, "chambers": {1: "Ctrl"}}]})
    _member(hub, directory=member)
    _tool(hub, "Experiment editor").click()
    editor = hub._experiment_editor
    try:
        assert editor._current_path == config
        assert editor.title_edit.text() == "Replicate 1"
        ## A save there is one the Hub hears, as it hears the Config Editor's:
        ## it re-reads what it shows.
        refreshed: list[bool] = []
        monkeypatch.setattr(hub, "refresh", lambda: refreshed.append(True))
        editor.title_edit.setText("Replicate 1, renamed")
        assert editor.write(config, confirm=False)
        assert refreshed == [True]
        assert _last_line(hub.log) == f"Wrote {config}"
        assert _read_yaml(config)["about"]["title"] == "Replicate 1, renamed"
    finally:
        editor.close()


def test_two_experiment_editors_opened_from_the_hub_both_stay_open(hub, tmp_path):
    first = tmp_path / "rep1"
    second = tmp_path / "rep2"
    for member in (first, second):
        _write_yaml(member / "flic_config.yaml", {"dfms": [{"id": 1}]})
    _member(hub, directory=first)
    one = hub._open_experiment_editor()
    _member(hub, directory=second, name="Rep2")
    two = hub._open_experiment_editor()
    try:
        assert one._current_path == first / "flic_config.yaml"
        assert two._current_path == second / "flic_config.yaml"
        assert one.isVisible() and two.isVisible()
    finally:
        one.close()
        two.close()


def test_with_nothing_loaded_the_experiment_editor_opens_empty(hub):
    editor = hub._open_experiment_editor()
    try:
        assert editor._current_path is None
    finally:
        editor.close()
