"""Every window opens its own help with F1, and its sections carry a ``?``.

One test per window and dialog: F1 opens that window's topic, exactly one F1
is bound in it (two bindings make the key ambiguous, and Qt then fires
neither), and its main sections each have a help button.  A last test
collects every help button the windows build and resolves its reference —
the runtime counterpart of ``tests/test_help_refs.py``'s source scan, which
cannot see a reference built from a map or passed through a helper.

No test here opens the help window or writes the user's settings: help
requests are recorded instead of opened, and the settings and the lab's type
library are pointed at a temporary folder for every test, with a guard that
the real settings file is untouched.
"""

from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path

import pytest
import yaml

pytest.importorskip("PyQt6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import matplotlib  # noqa: E402

matplotlib.use("Agg")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtGui import QAction, QKeySequence, QShortcut  # noqa: E402
from PyQt6.QtWidgets import QApplication, QDialog, QWidget  # noqa: E402

from pyflic.base.ui import settings as _settings  # noqa: E402
from pyflic.help import topics as _topics  # noqa: E402

#: The settings file this process would write without the patch below.
_REAL_SETTINGS = Path(_settings.config_file())


def _stamp(path: Path):
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_size, stat.st_mtime_ns


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    """Settings and the type library in a temporary folder; the real
    settings file must come out of every test as it went in."""
    from pyflic.base.experiment_types import library

    before = _stamp(_REAL_SETTINGS)
    folder = tmp_path / "config"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(folder.resolve()))
    (tmp_path / "types").mkdir()
    monkeypatch.setenv(library.ENV_LIBRARY, str(tmp_path / "types"))
    assert _settings.config_file() == folder.resolve() / "pyflic" / "ui.json"
    yield
    assert _stamp(_REAL_SETTINGS) == before, "a test wrote the user's ui.json"


@pytest.fixture
def opened(monkeypatch) -> list[str]:
    """The help references asked for, in order.  No help window opens."""
    import pyflic.help.button as button_mod
    import pyflic.help.window as window_mod

    calls: list[str] = []

    def record(topic_id=None, anchor=None, guide_id=None):
        if topic_id and anchor is None:
            topic_id, anchor = _topics.parse_ref(topic_id)
        calls.append(f"{topic_id}#{anchor}" if anchor else str(topic_id))

    monkeypatch.setattr(window_mod, "open_help", record)
    monkeypatch.setattr(button_mod, "open_help", record)
    return calls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_F1 = QKeySequence(Qt.Key.Key_F1)


def _own(window: QWidget, widgets) -> list:
    """The objects among *widgets* that belong to *window* itself, not to a
    child window or dialog of it."""
    def owner(obj):
        parent = obj.parent()
        while parent is not None and not (isinstance(parent, QWidget)
                                          and parent.isWindow()):
            parent = parent.parent()
        return parent

    return [w for w in widgets if owner(w) is window]


def press_f1(window: QWidget, opened: list[str]) -> str:
    """Press F1 in *window* and return what it asked help for.

    A real key press when the window can be made active, which is what shows
    an ambiguous binding (Qt then fires nothing); the binding is counted as
    well, so the check holds where the platform will not activate a window.
    """
    from PyQt6.QtTest import QTest

    shortcuts = _own(window, [s for s in window.findChildren(QShortcut)
                              if s.key() == _F1])
    actions = _own(window, [a for a in window.findChildren(QAction)
                            if _F1 in a.shortcuts()])
    assert len(shortcuts) + len(actions) == 1, (
        f"F1 is bound {len(shortcuts) + len(actions)} times in "
        f"{type(window).__name__}; Qt fires none of an ambiguous shortcut")
    assert shortcuts, "F1 should come from install_help_shortcut"
    count = len(opened)
    window.show()
    window.activateWindow()
    if QTest.qWaitForWindowActive(window, 1000):
        QTest.keyClick(window, Qt.Key.Key_F1)
        QApplication.processEvents()
    else:  # pragma: no cover - a platform that will not activate the window
        shortcuts[0].activated.emit()
    assert len(opened) == count + 1, "F1 opened no help"
    return opened[-1]


def help_refs(widget: QWidget) -> list[str]:
    from pyflic.help.button import HelpButton

    return [b.ref for b in widget.findChildren(HelpButton)]


def assert_resolves(refs) -> None:
    problems = []
    for ref in refs:
        topic_id, anchor = _topics.parse_ref(ref)
        topic = _topics.load(topic_id)
        if topic is None:
            problems.append(f"{ref}: no such topic")
        elif anchor and not topic.has_anchor(anchor):
            problems.append(f"{ref}: no such heading")
    assert not problems, "dangling help references:\n  " + "\n  ".join(problems)


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "Proj"
    _write(root / "project.yaml",
           {"name": "Proj",
            "design": {"global": {"chamber_layout": "two_well",
                                  "well_names": ["A", "B"]}}})
    _write(root / "rep1" / "data" / "DFM1_0.csv", "Sample,Seconds\n")
    _write(root / "rep1" / "flic_config.yaml",
           {"dfms": [{"id": 1, "chambers": {1: "Ctrl", 2: "Exp"}}]})
    _write(root / "rep2" / "data" / "DFM1_0.csv", "Sample,Seconds\n")
    return root


# ---------------------------------------------------------------------------
# The Hub and its dialogs
# ---------------------------------------------------------------------------

@pytest.fixture
def hub(app, opened):
    from pyflic.base.hub import AnalysisHubWindow

    window = AnalysisHubWindow()
    yield window
    window.close()


def test_the_hubs_f1_opens_its_topic(hub, opened):
    assert press_f1(hub, opened) == "app-hub"


def test_the_hubs_controls_that_need_more_than_their_card_have_help(hub):
    batch = help_refs(hub.panels["batch"])
    assert {"app-hub#batch-panel", "scripts-batch#which-script-runs",
            "scripts-batch#ai-narrative-of-the-batch"} <= set(batch)
    experiments = help_refs(hub.project_experiments_card)
    assert {"app-hub#the-members-table",
            "app-hub#adding-a-member-the-same-three-cases-one-level-down",
            "concepts-project#blocked-members",
            "performance#parallel-loading"} <= set(experiments)
    assert "scripts-overview#two-levels" in help_refs(hub.project_analysis_card)
    assert "concepts-ai-summary#providers-and-keys" in help_refs(hub.panels["ai"])
    for key, panel in hub.panels.items():
        for card in panel.cards():
            assert help_refs(card), f"the {key} panel's {card.title()} card has no ?"


def test_the_type_drift_help_shows_only_with_the_drift(hub):
    assert hub._type_drift_help is not None
    assert hub._type_drift_help.isHidden()
    assert hub._type_drift_help.ref == "concepts-experiment-types#where-types-live"


def test_the_member_configs_dialog_opens_its_topic(app, opened, project):
    from pyflic.base.hub import AnalysisHubWindow, MemberConfigsDialog

    window = AnalysisHubWindow(target=str(project))
    dialog = MemberConfigsDialog(window, window.project)
    try:
        assert press_f1(dialog, opened) == "concepts-project#the-member-configs-dialog"
        assert help_refs(dialog) == ["concepts-project#the-member-configs-dialog"]
        ## Window-modal, so the help it opens stays usable beside it.
        assert dialog.windowModality() == Qt.WindowModality.WindowModal
    finally:
        dialog.close()
        window.close()


def test_the_batch_review_window_opens_its_topic(app, opened, tmp_path):
    from pyflic.base.batch_preflight import BatchPreflightDialog

    root = tmp_path / "batch"
    root.mkdir()
    dialog = BatchPreflightDialog(None, root, checked=[], log=lambda _m: None)
    try:
        assert press_f1(dialog, opened) == "scripts-batch#the-review-window"
        assert help_refs(dialog) == ["scripts-batch#the-project-list",
                                     "scripts-batch#repairing-blocked-members",
                                     "scripts-batch#the-exclusion-sheet-switch"]
        assert dialog.windowModality() == Qt.WindowModality.WindowModal
        ## No sheet in the folder, so no switch, and no ? beside nothing.
        assert dialog._sheet_box.isHidden() and dialog._sheet_help.isHidden()
    finally:
        dialog.close()


def test_the_project_design_dialog_opens_its_topic(app, opened, tmp_path):
    from pyflic.base.design_editor import ProjectDesignDialog

    dialog = ProjectDesignDialog(start_dir=str(tmp_path / "new"))
    try:
        assert press_f1(dialog, opened) == "concepts-project#the-project-design-dialog"
        from pyflic.base.ui.widgets import Card

        cards = {c.title(): help_refs(c) for c in dialog.findChildren(Card)}
        assert cards["Project"][0] == "concepts-project#the-project-design-dialog"
        assert cards["Design"][0] == "concepts-project#the-design-is-an-authority"
        assert {"concepts-experiment-types",
                "concepts-experiment-types#control-comparisons",
                "concepts-facets", "config-structure#globalexclusion-group",
                "config-factors"} <= set(cards["Design"])
        assert cards["Detection parameters"][0] == "reference-parameters"
        assert {"config-dfms-chambers#automatic-exclusion",
                "concepts-optogenetics#thresholds"} <= set(cards["Detection parameters"])
    finally:
        dialog.close()


# ---------------------------------------------------------------------------
# The editors
# ---------------------------------------------------------------------------

def test_the_config_editors_f1_is_bound_once_and_opens_its_topic(app, opened):
    from pyflic.base.config_editor import FLICConfigEditor
    from pyflic.base.ui.widgets import Card

    editor = FLICConfigEditor()
    try:
        assert press_f1(editor, opened) == "app-config-editor"
        refs = {c.title(): help_refs(c) for c in editor.findChildren(Card)}
        assert refs["Experiment Settings"][0] == "app-config-editor#experiment-settings"
        assert {"concepts-experiment-types",
                "concepts-experiment-types#control-comparisons",
                "config-structure#globaloptogenetics",
                "concepts-facets#program-facets",
                "app-config-editor#auto-filter-thresholds",
                "app-config-editor#progressive-ratio-settings",
                "concepts-optogenetics#thresholds"} <= set(refs["Experiment Settings"])
        assert refs["Global Parameters"][0] == "reference-parameters"
        assert refs["Experimental Design Factors"] == ["config-factors"]
        assert refs["DFM Identity"] == ["app-config-editor#dfm-identity"]
        ## Titled by what is assigned: treatments, or factor levels.
        assert help_refs(editor._dfm_widgets[0]._ch_card) == [
            "app-config-editor#chamber-assignments"]
        assert refs["Parameter Overrides"][0] == "app-config-editor#parameter-overrides"
        assert refs["Paired chamber per chamber group"] == [
            "concepts-progressive-ratio#chamber-groups-paired-and-yoked"]
    finally:
        editor._dirty = False
        editor.close()


def test_the_type_editor_opens_its_topic_and_every_card_has_help(app, opened):
    from pyflic.base.type_editor import ExperimentTypeEditor
    from pyflic.base.ui.widgets import Card

    editor = ExperimentTypeEditor()
    try:
        assert press_f1(editor, opened) == "app-type-editor"
        for card in editor.findChildren(Card):
            assert help_refs(card), f"the {card.title()} card has no ?"
        refs = {c.title(): help_refs(c)[0] for c in editor.findChildren(Card)}
        assert refs["Type"] == "app-type-editor#type"
        assert refs["Analysis Modules"] == "concepts-experiment-types#analysis-modules"
        assert refs["Light"] == "concepts-experiment-types#the-light-a-type-expects"
    finally:
        editor._dirty = False
        editor.close()


def test_the_plot_editor_opens_its_topic_and_every_group_has_help(app, opened):
    from pyflic.base.plot_editor import PlotEditorWindow
    from pyflic.base.ui.widgets import Card, CardGroup

    window = PlotEditorWindow()
    try:
        assert press_f1(window, opened) == "app-plot-editor"
        for group in window.findChildren(CardGroup):
            assert help_refs(group), f"the {group.title()} group has no ?"
        preview, = window.findChildren(Card)
        assert help_refs(preview) == ["app-plot-editor#preview-and-output"]
    finally:
        window.close()


def test_the_script_editor_opens_its_topic_and_each_pane_has_help(app, opened, tmp_path):
    from pyflic.base.script_editor.window import ScriptEditorWindow

    config = tmp_path / "flic_config.yaml"
    _write(config, {"dfms": [{"id": 1, "chambers": {1: "Ctrl"}}],
                    "scripts": [{"name": "quick", "steps": []}]})
    window = ScriptEditorWindow(config)
    try:
        assert press_f1(window, opened) == "scripts-editor"
        assert help_refs(window._palette) == ["scripts-editor#the-palette"]
        assert help_refs(window._canvas) == ["scripts-editor#the-canvas"]
        assert help_refs(window._inspector) == ["scripts-editor#the-inspector"]
        assert help_refs(window._preview) == ["scripts-editor#the-yaml-preview"]
    finally:
        window._dirty = False
        window.close()


# ---------------------------------------------------------------------------
# The QC Viewer
# ---------------------------------------------------------------------------

@pytest.fixture
def viewer(app, opened, tmp_path):
    """The viewer on the synthetic optogenetic rig, handed over as the Hub
    hands over a loaded member, with its QC written so every sub-tab shows."""
    from opto_fixtures import make_opto_dir
    from pyflic.base import qc_viewer
    from pyflic.base.yaml_config import load_experiment_yaml

    root = make_opto_dir(tmp_path / "rig")
    with contextlib.redirect_stdout(io.StringIO()):
        exp = load_experiment_yaml(root, parallel=False, use_disk_cache=False)
        exp.write_qc_reports()
    win = qc_viewer.MainWindow(root, experiment=exp)
    yield win
    win.close()


def _tab(win, label: str) -> int:
    labels = [win._tabs.tabText(i) for i in range(win._tabs.count())]
    return labels.index(label)


def test_the_viewers_f1_follows_the_tab_and_the_dfm_sub_tab(viewer, opened):
    win = viewer
    expected = {
        "Load": "app-qc-viewer#the-load-tab",
        "Feeding Summary": "app-qc-viewer#the-feeding-summary-tab",
        "Opto Light QC": "app-qc-viewer#the-opto-light-qc-tab",
        "Params": "app-qc-viewer#the-params-tab",
    }
    for label, ref in expected.items():
        win._tabs.setCurrentIndex(_tab(win, label))
        assert press_f1(win, opened) == ref, label
    win._tabs.setCurrentIndex(_tab(win, "DFM 1"))
    dfm = win._dfm_tab_widgets[1]
    sub = {dfm._tabs.tabText(i): i for i in range(dfm._tabs.count())}
    for label, ref in (("Integrity", "app-qc-viewer#the-integrity-sub-tab"),
                       ("Raw Signal", "app-qc-viewer#the-signal-plot-sub-tabs"),
                       ("Program", "app-qc-viewer#the-program-sub-tab")):
        dfm._tabs.setCurrentIndex(sub[label])
        assert press_f1(win, opened) == ref, label
        ## The ? in the sub-tab bar's corner follows too.
        assert dfm._help_corner.ref == ref, label
    ## Leaving and coming back restores the sub-tab with its signals
    ## blocked; the corner and F1 still follow it.
    dfm._tabs.setCurrentIndex(sub["Integrity"])
    win._tabs.setCurrentIndex(_tab(win, "Params"))
    win._active_subtab_idx = sub["Raw Signal"]
    win._tabs.setCurrentIndex(_tab(win, "DFM 1"))
    assert dfm._help_corner.ref == "app-qc-viewer#the-signal-plot-sub-tabs"
    assert press_f1(win, opened) == "app-qc-viewer#the-signal-plot-sub-tabs"


def test_each_part_of_the_viewer_has_help(viewer):
    win = viewer
    assert help_refs(win._load_tab) == ["app-qc-viewer#the-load-tab"]
    assert help_refs(win._feeding_tab) == ["app-qc-viewer#the-feeding-summary-tab"]
    dfm = win._dfm_tab_widgets[1]
    assert "app-qc-viewer#the-exclude-wells-panel" in help_refs(dfm)
    assert dfm._tabs.cornerWidget(Qt.Corner.TopRightCorner) is dfm._help_corner
    opto = help_refs(win._opto_tab)
    assert {"app-qc-viewer#the-opto-light-qc-tab", "concepts-optogenetics#explained-light",
            "concepts-optogenetics#the-checks"} <= set(opto)
    params = help_refs(win._params_tab)
    assert params[0] == "reference-parameters"
    assert len(params) == 1 + len(type(win._params_tab)._PARAM_FIELDS)


def test_the_opto_detail_panes_help_follows_the_pane(viewer):
    from pyflic.base.qc_viewer import _OPTO_DETAIL_HELP

    details = viewer._opto_tab._details
    corner = details.cornerWidget(Qt.Corner.TopRightCorner)
    for index in range(details.count()):
        details.setCurrentIndex(index)
        assert corner.ref == _OPTO_DETAIL_HELP[details.tabText(index)]


def test_the_auto_filter_criteria_window_opens_its_section(viewer, opened, monkeypatch):
    shown: list[QDialog] = []

    def fake_exec(dialog):
        shown.append(dialog)
        return 0

    monkeypatch.setattr(QDialog, "exec", fake_exec)
    viewer._on_view_filter_criteria()
    dialog, = shown
    assert press_f1(dialog, opened) == "app-qc-viewer#auto-filter"
    assert help_refs(dialog) == ["app-qc-viewer#auto-filter"]
    assert dialog.windowModality() == Qt.WindowModality.WindowModal


# ---------------------------------------------------------------------------
# Every reference the windows build resolves
# ---------------------------------------------------------------------------

def test_every_help_button_the_windows_build_resolves(app, opened, tmp_path, project):
    from pyflic.base.batch_preflight import BatchPreflightDialog
    from pyflic.base.config_editor import FLICConfigEditor
    from pyflic.base.design_editor import ProjectDesignDialog
    from pyflic.base.hub import AnalysisHubWindow, MemberConfigsDialog
    from pyflic.base.plot_editor import PlotEditorWindow
    from pyflic.base.script_editor.window import ScriptEditorWindow
    from pyflic.base.type_editor import ExperimentTypeEditor

    config = tmp_path / "solo" / "flic_config.yaml"
    _write(config, {"dfms": [{"id": 1, "chambers": {1: "Ctrl"}}]})
    hub = AnalysisHubWindow(target=str(project))
    windows = [
        hub, MemberConfigsDialog(hub, hub.project),
        BatchPreflightDialog(None, tmp_path, checked=[], log=lambda _m: None),
        ProjectDesignDialog(start_dir=str(tmp_path / "new")),
        FLICConfigEditor(), ExperimentTypeEditor(), PlotEditorWindow(),
        ScriptEditorWindow(config),
    ]
    try:
        refs = [ref for window in windows for ref in help_refs(window)]
        assert len(refs) > 60
        assert_resolves(refs)
    finally:
        for window in reversed(windows):
            for flag in ("_dirty",):
                if hasattr(window, flag):
                    setattr(window, flag, False)
            window.close()


def test_every_help_button_the_viewer_builds_resolves(viewer):
    from pyflic.base.qc_viewer import _DFM_SUBTAB_HELP, _OPTO_DETAIL_HELP, MainWindow

    assert_resolves(help_refs(viewer))
    assert_resolves([*_DFM_SUBTAB_HELP.values(), *_OPTO_DETAIL_HELP.values(),
                     *MainWindow._TAB_HELP.values()])


# ---------------------------------------------------------------------------
# Help opened over a modal dialog
# ---------------------------------------------------------------------------

@pytest.fixture
def real_help(app):
    """The real help window, discarded afterwards without being closed:
    closing is what saves its geometry, and these tests must save nothing
    (the settings fixture would catch it in a temporary folder anyway)."""
    import pyflic.help.window as window_mod

    yield
    win, window_mod._window = window_mod._window, None
    if win is not None:
        win.hide()
        win.deleteLater()


def test_help_over_an_application_modal_dialog_is_usable(real_help):
    """An application-modal dialog blocks every other window, so help opened
    from one was visible but took no input; it is modal itself then, and
    modeless again when opened from anywhere else."""
    from PyQt6.QtCore import QTimer

    from pyflic.help.window import help_window, open_help

    seen: list = []
    dialog = QDialog()

    def inside():
        open_help("app-hub")
        seen.append(help_window().windowModality())
        ## Modal over the dialog, help is put away before the dialog can be.
        help_window().hide()
        dialog.accept()

    QTimer.singleShot(0, inside)
    dialog.exec()
    open_help("app-hub")
    seen.append(help_window().windowModality())
    help_window().hide()
    assert seen == [Qt.WindowModality.ApplicationModal, Qt.WindowModality.NonModal]


def test_help_over_a_window_modal_dialog_stays_modeless(real_help):
    from PyQt6.QtCore import QTimer

    from pyflic.help.window import help_window, open_help

    seen: list = []
    dialog = QDialog()
    dialog.setWindowModality(Qt.WindowModality.WindowModal)

    def inside():
        open_help("app-hub")
        seen.append(help_window().windowModality())
        dialog.accept()

    QTimer.singleShot(0, inside)
    dialog.exec()
    help_window().hide()
    assert seen == [Qt.WindowModality.NonModal]
