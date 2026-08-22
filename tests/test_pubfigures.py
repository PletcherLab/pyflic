"""Tests for the Plot Spec / Plot Style system and both figure families."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pyflic.base import pubfigures


def _facet_frame() -> pd.DataFrame:
    rows = []
    for experiment in ("rep1", "rep2"):
        for facet in ("0-60 min", "60+ min"):
            for treatment, base in (("Ctrl", 0.1), ("Exp", 0.6)):
                for index in range(4):
                    rows.append({
                        "Experiment": experiment, "Facet": facet,
                        "FacetRange": "(0, 60)" if facet.startswith("0") else "(60, inf)",
                        "Treatment": treatment, "DFM": 1 + index % 2,
                        "Chamber": index + 1,
                        "PI": base + index * 0.01,
                        "LicksA": 100 + index, "LicksB": 40 + index,
                    })
    return pd.DataFrame(rows)


def _binned_frame() -> pd.DataFrame:
    rows = []
    for experiment in ("rep1", "rep2"):
        for minutes in (15, 45, 75):
            for treatment, base in (("Ctrl", 10.0), ("Exp", 40.0)):
                for chamber in range(3):
                    rows.append({
                        "Experiment": experiment, "Minutes": minutes,
                        "Treatment": treatment, "DFM": 1, "Chamber": chamber,
                        "LicksA": base + chamber, "LicksB": 5.0,
                        "PI": 0.2 if treatment == "Ctrl" else 0.8,
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Spec / style persistence
# ---------------------------------------------------------------------------

def test_specs_round_trip_through_yaml(tmp_path: Path):
    specs = pubfigures.ProjectSpecs()
    specs.ensure_default_style()
    specs.styles["default"].point_size = 3.3
    specs.styles["default"].colors = {"Ctrl": "#123456"}
    spec = pubfigures.default_spec("faceted_pi", well_a="S5")
    spec.title = "Preference"
    spec.treatments = {"Ctrl": {"label": "Control", "show": True}}
    specs.plots["faceted_pi"] = spec

    pubfigures.save_project_specs(tmp_path, specs)
    loaded = pubfigures.load_project_specs(tmp_path)
    assert loaded.styles["default"].point_size == 3.3
    assert loaded.styles["default"].colors == {"Ctrl": "#123456"}
    assert loaded.plots["faceted_pi"].title == "Preference"
    assert loaded.plots["faceted_pi"].treatments["Ctrl"]["label"] == "Control"


def test_unknown_plot_ids_are_dropped_on_load(tmp_path: Path):
    (tmp_path / pubfigures.SPECS_FILENAME).write_text(
        "plots:\n  not_a_plot:\n    title: x\n", encoding="utf-8")
    assert pubfigures.load_project_specs(tmp_path).plots == {}


def test_an_invalid_style_field_falls_back_rather_than_raising():
    style = pubfigures.PlotStyle.from_dict({"theme": "nonsense", "geom": "blob"})
    assert style.theme == "classic"
    assert style.geom == "dots"


def test_color_is_keyed_by_the_original_name_not_the_label():
    """Renaming a treatment for one figure must not change its color."""
    style = pubfigures.PlotStyle.from_dict({"colors": {"Ctrl": "#abcdef"}})
    assert style.color_for("Ctrl", 0) == "#abcdef"
    assert style.color_for("Exp", 1) == pubfigures.DEFAULT_PALETTE[1]


def test_default_spec_substitutes_the_well_name():
    spec = pubfigures.default_spec("faceted_pi", well_a="S5")
    assert "S5" in spec.y_label
    assert spec.y_limits == [-1.0, 1.0]
    assert spec.ref_line == 0.0


# ---------------------------------------------------------------------------
# Layout gating
# ---------------------------------------------------------------------------

def test_pi_plots_are_offered_only_for_two_well():
    two = pubfigures.plots_for_layout("two_well")
    single = pubfigures.plots_for_layout("single_well")
    assert "faceted_pi" in two and "timecourse_pi" in two
    assert "faceted_pi" not in single and "timecourse_pi" not in single
    assert "faceted_licks" in single      # layout-neutral plots survive


# ---------------------------------------------------------------------------
# Data shaping
# ---------------------------------------------------------------------------

def test_faceted_data_resolves_two_well_metrics():
    """A two-well summary has LicksA/LicksB, never a bare Licks column."""
    df = pubfigures.faceted_data(_facet_frame(), "Licks")
    assert not df.empty
    assert set(df.columns) >= {"Treatment", "Phase", "Value", "Experiment"}
    assert (df["Value"] > 100).all()      # A + B, not one well


def test_faceted_data_of_an_unfaceted_frame_is_one_phase():
    frame = _facet_frame().drop(columns=["Facet"])
    df = pubfigures.faceted_data(frame, "PI")
    assert list(df["Phase"].cat.categories) == ["Whole recording"]


def test_timecourse_data_keeps_the_time_axis():
    df = pubfigures.timecourse_data(_binned_frame(), "Licks")
    assert set(df["Minutes"]) == {15, 45, 75}
    assert "Experiment" in df.columns


def test_merged_treatments_picks_up_a_treatment_absent_from_the_spec():
    spec = pubfigures.PlotSpec(treatments={"Ctrl": {"label": "C", "show": True}})
    df = pubfigures.faceted_data(_facet_frame(), "PI")
    merged = pubfigures.merged_treatments(spec, df)
    assert set(merged) == {"Ctrl", "Exp"}     # Exp appears rather than vanishing


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_faceted_figure_renders(tmp_path: Path):
    df = pubfigures.faceted_data(_facet_frame(), "PI")
    spec = pubfigures.default_spec("faceted_pi")
    figure = pubfigures.build_figure("faceted_pi", df, spec,
                                     pubfigures.PlotStyle())
    out = pubfigures.save_figure(figure, str(tmp_path / "f.svg"), "svg")
    assert Path(out).stat().st_size > 1000


def test_timecourse_figure_renders(tmp_path: Path):
    df = pubfigures.timecourse_data(_binned_frame(), "Licks")
    spec = pubfigures.default_spec("timecourse_licks")
    figure = pubfigures.build_figure("timecourse_licks", df, spec,
                                     pubfigures.PlotStyle())
    out = pubfigures.save_figure(figure, str(tmp_path / "t.svg"), "svg")
    assert Path(out).stat().st_size > 1000


def test_svg_keeps_text_editable(tmp_path: Path):
    """svg.fonttype='none' is what makes labels live text in Illustrator."""
    df = pubfigures.faceted_data(_facet_frame(), "PI")
    spec = pubfigures.default_spec("faceted_pi")
    spec.y_label = "UNIQUEYLABEL"
    figure = pubfigures.build_figure("faceted_pi", df, spec,
                                     pubfigures.PlotStyle())
    out = pubfigures.save_figure(figure, str(tmp_path / "f.svg"), "svg")
    body = Path(out).read_text(encoding="utf-8")
    assert "UNIQUEYLABEL" in body          # text, not outlined paths


def test_hidden_treatment_is_excluded_from_the_figure():
    df = pubfigures.faceted_data(_facet_frame(), "PI")
    spec = pubfigures.default_spec("faceted_pi")
    spec.treatments = {"Ctrl": {"label": "Ctrl", "show": True},
                       "Exp": {"label": "Exp", "show": False}}
    data, labels, colors = pubfigures._apply_treatment_order(
        df, spec, pubfigures.PlotStyle())
    assert labels == ["Ctrl"]
    assert set(data["Treatment"].astype(str)) == {"Ctrl"}


def test_preview_png_is_returned_as_bytes():
    df = pubfigures.faceted_data(_facet_frame(), "PI")
    spec = pubfigures.default_spec("faceted_pi")
    figure = pubfigures.build_figure("faceted_pi", df, spec,
                                     pubfigures.PlotStyle())
    png = pubfigures.render_png_bytes(figure, pubfigures.PlotStyle())
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
