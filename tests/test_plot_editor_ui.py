"""The Plot Editor's preview: it renders, it says why when it cannot, and
opening it changes nothing.

Every one of these is a regression.  The editor opened showing a 94x94
thumbnail of every figure and a blank rectangle for the rest, and closing it
wrote that over the Project's ``plot_specs.yaml``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import matplotlib  # noqa: E402

matplotlib.use("Agg")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from pyflic.base import pubfigures  # noqa: E402
from pyflic.base.plot_editor import PlotEditorWindow  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    """Answer every message box instead of showing one.

    A modal dialog under the offscreen platform waits for a click that will
    never come, so one unstubbed ``QMessageBox`` hangs the whole run rather
    than failing it.
    """
    from PyQt6.QtWidgets import QMessageBox

    for name in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok))


def _pooled_facet_rows() -> pd.DataFrame:
    rows = []
    for experiment in ("rep1", "rep2"):
        for facet in ("0-60 min", "60+ min"):
            for treatment, base in (("Ctrl", 0.1), ("Exp", 0.6)):
                for index in range(4):
                    rows.append({
                        "Experiment": experiment, "Facet": facet,
                        "Treatment": treatment, "DFM": 1, "Chamber": index + 1,
                        "PI": base + index * 0.01,
                        "LicksA": 100 + index, "LicksB": 40 + index,
                    })
    return pd.DataFrame(rows)


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """A Project with a pooled facet table on disk and nothing else — enough
    for the faceted figures, deliberately not enough for the time courses."""
    root = tmp_path / "proj"
    ## A Custom Experiment states its Chamber Layout and no experiment_type
    ## (ADR-0007); getting that wrong makes Project() raise, and the editor
    ## answers a raise with a MODAL dialog that hangs an offscreen test.
    design = {
        "chamber_layout": "two_well",
        "well_names": {"A": "Sucrose", "B": "Yeast"},
    }
    for name in ("rep1", "rep2"):
        member = root / name
        member.mkdir(parents=True)
        (member / "flic_config.yaml").write_text(
            yaml.safe_dump({"dfms": [{"id": 1, "chambers": {1: "Ctrl"}}]},
                           sort_keys=False), encoding="utf-8")
    (root / "project.yaml").write_text(
        yaml.safe_dump({"name": "proj", "design": {"global": design}},
                       sort_keys=False), encoding="utf-8")
    analysis = root / "analysis"
    analysis.mkdir()
    _pooled_facet_rows().to_csv(analysis / "proj_Summary_Facet.csv", index=False)
    return root


@pytest.fixture
def editor(app, project_dir):
    window = PlotEditorWindow(str(project_dir))
    window.show()
    app.processEvents()
    yield window
    window.close()


def _select(editor, app, plot_id: str) -> None:
    index = editor.plot_combo.findData(plot_id)
    assert index >= 0, f"{plot_id} is not offered for this Project"
    editor.plot_combo.setCurrentIndex(index)
    app.processEvents()


def test_opening_does_not_overwrite_the_style_with_empty_widgets(editor):
    """The Style tab is where ``_apply_style`` reads the style back FROM, so
    it has to be loaded before anything harvests it.  It was not, and opening
    the editor flattened the style to the spin boxes' minimums."""
    style = editor.specs.styles[editor.style_combo.currentText()]
    defaults = pubfigures.PlotStyle()
    assert (style.width_mm, style.height_mm) == (defaults.width_mm,
                                                 defaults.height_mm)
    assert style.base_pt == defaults.base_pt
    assert style.font_family == defaults.font_family
    assert style.point_size == defaults.point_size
    ## And the widgets agree with it, which is what made the round-trip safe.
    assert editor.width_mm.value() == defaults.width_mm
    assert editor.height_mm.value() == defaults.height_mm


def test_a_figure_with_data_previews_at_the_styles_size(editor, app):
    _select(editor, app, "faceted_pi")
    pixmap = editor.preview_label.pixmap()
    assert not pixmap.isNull()
    ## 180 mm at 120 dpi, not the 20 mm minimum the empty widgets produced.
    assert pixmap.width() > 600
    assert editor.preview_label.text() == ""


def test_a_figure_without_data_says_so_instead_of_going_blank(editor, app):
    """A QLabel holds text or a pixmap and each setter clears the other, so
    the order mattered: the message was set and then wiped, leaving a blank
    rectangle — indistinguishable from a broken editor."""
    _select(editor, app, "timecourse_pi")
    assert editor.preview_label.pixmap().isNull()
    assert "binned" in editor.preview_label.text().lower()


def test_editing_the_style_re_renders_the_preview(editor, app):
    _select(editor, app, "faceted_pi")
    before = editor.preview_label.pixmap().width()
    editor.width_mm.setValue(90.0)
    app.processEvents()
    after = editor.preview_label.pixmap().width()
    assert after < before
    assert editor.specs.styles[editor.style_combo.currentText()].width_mm == 90.0


def test_editing_the_content_re_renders_the_preview(editor, app):
    _select(editor, app, "faceted_pi")
    editor.title_edit.setText("A title")
    editor._apply_content()
    app.processEvents()
    assert editor.specs.plots["faceted_pi"].title == "A title"
    assert not editor.preview_label.pixmap().isNull()


def test_opening_and_closing_untouched_writes_nothing(app, project_dir):
    specs_file = project_dir / pubfigures.SPECS_FILENAME
    assert not specs_file.exists()
    window = PlotEditorWindow(str(project_dir))
    window.show()
    app.processEvents()
    window.close()
    app.processEvents()
    assert not specs_file.exists()


# ---------------------------------------------------------------------------
# One plot at a time
# ---------------------------------------------------------------------------
#
# The picker used to be a checkable list, so "which plot am I editing" and
# "which plots belong to the figure set" were two meanings of one widget —
# and ticking several implied the preview could show several, which it never
# could.  The set is no longer a thing the editor curates.

def test_the_plot_picker_is_single_choice(editor):
    from PyQt6.QtWidgets import QComboBox, QListWidget

    assert isinstance(editor.plot_combo, QComboBox)
    assert not hasattr(editor, "plot_list")
    ## Every figure this Project can draw is offered, none of them ticked.
    offered = {editor.plot_combo.itemData(i)
               for i in range(editor.plot_combo.count())}
    assert offered == set(pubfigures.plots_for_layout(
        editor.project.chamber_layout, editor.project.experiment_type.name))
    ## The lists that remain are inside one plot: its facets and treatments.
    assert isinstance(editor.facet_list, QListWidget)


def test_switching_plots_keeps_each_ones_edits(editor, app):
    _select(editor, app, "faceted_pi")
    editor.title_edit.setText("PI title")
    editor._apply_content()
    _select(editor, app, "faceted_licks")
    editor.title_edit.setText("Licks title")
    editor._apply_content()
    _select(editor, app, "faceted_pi")
    assert editor.title_edit.text() == "PI title"
    assert editor.specs.plots["faceted_licks"].title == "Licks title"


def test_saving_writes_the_edited_plot_into_the_file(editor, app, project_dir):
    _select(editor, app, "faceted_pi")
    editor.title_edit.setText("Preference")
    editor._apply_content()
    editor._save_specs()
    saved = pubfigures.load_project_specs(str(project_dir))
    assert saved.plots["faceted_pi"].title == "Preference"


def test_restore_defaults_resets_only_this_plots_content(editor, app):
    _select(editor, app, "faceted_pi")
    editor.title_edit.setText("Temporary")
    editor._apply_content()
    style_name = editor.style_combo.currentText()
    editor.specs.styles[style_name].width_mm = 111.0

    editor._restore_defaults()
    assert editor.specs.plots["faceted_pi"].title == (
        pubfigures.default_spec("faceted_pi").title)
    ## Styles are shared; resetting one figure must not repaint the set.
    assert editor.specs.styles[style_name].width_mm == 111.0


# ---------------------------------------------------------------------------
# One panel, four groups
# ---------------------------------------------------------------------------

def test_every_control_is_on_one_scrolling_panel(editor):
    """Content and Style were two tabs, so half the knobs shaping the one
    figure in the preview were always behind a click."""
    from pyflic.base.ui.widgets import CardGroup

    assert not hasattr(editor, "tabs")
    titles = [g.title() for g in editor.controls_scroll.findChildren(CardGroup)]
    assert titles == ["Style (shared across plots)", "This plot",
                      "Facets", "Treatments"]


def test_the_facets_group_is_hidden_for_a_time_course(editor, app):
    _select(editor, app, "faceted_pi")
    assert not editor.facets_group.isHidden()
    _select(editor, app, "timecourse_pi")
    assert editor.facets_group.isHidden()
    ## ...and the binning row it makes way for is shown instead.
    assert not editor.binsize.isHidden()


def test_treatments_are_one_table_not_two_lists(editor, app):
    _select(editor, app, "faceted_pi")
    table = editor.treatment_table
    assert not hasattr(editor, "treatment_list")
    assert not hasattr(editor, "color_list")
    assert [table.horizontalHeaderItem(i).text()
            for i in range(table.columnCount())] == ["Treatment", "Label",
                                                     "Colour"]
    assert table.rowCount() == 2
    ## The colour cell shows the colour the figure will actually use.
    style = editor.specs.style_for(editor.current_spec())
    names = [table.item(r, 0).data(Qt.ItemDataRole.UserRole)
             for r in range(table.rowCount())]
    for row, name in enumerate(names):
        assert table.cellWidget(row, 2).text() == style.color_for(name, row)


def test_the_table_writes_labels_and_visibility_back_to_the_spec(editor, app):
    _select(editor, app, "faceted_pi")
    table = editor.treatment_table
    first = table.item(0, 0).data(Qt.ItemDataRole.UserRole)
    table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    table.item(1, 1).setText("Renamed")
    editor._apply_content()
    treatments = editor.current_spec().treatments
    assert treatments[str(first)]["show"] is False
    assert "Renamed" in [e["label"] for e in treatments.values()]


def test_facet_size_is_per_panel_and_off_at_zero(editor, app):
    """A three-facet figure should not be three squeezed panels in the width
    of a one-facet one."""
    _select(editor, app, "faceted_pi")
    assert editor.facet_width_mm.value() == 0.0
    assert editor.facet_width_mm.text() == "off"

    editor.facet_width_mm.setValue(40.0)
    editor.facet_height_mm.setValue(50.0)
    app.processEvents()
    style = editor.specs.styles[editor.style_combo.currentText()]
    assert (style.facet_width_mm, style.facet_height_mm) == (40.0, 50.0)
    ## Two facets at 40 mm each, plus the axis margin — not the 180 mm figure.
    assert pubfigures.effective_width_mm(style, 2) < style.width_mm
    assert pubfigures.effective_width_mm(style, 2) > 2 * 40.0


def test_the_preview_is_fitted_to_its_pane_and_never_upscaled(editor, app):
    _select(editor, app, "faceted_pi")
    full = editor._preview_pixmap
    shown = editor.preview_label.pixmap()
    assert not full.isNull()
    viewport = editor.preview_scroll.viewport().width()
    assert shown.width() <= viewport
    ## Fitting means scaling down only — the preview claims to be the file.
    assert shown.width() <= full.width()
    assert (editor.preview_scroll.horizontalScrollBarPolicy()
            == Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
