"""Tests for the Plot Spec / Plot Style system and both figure families."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pyflic.base import experiment_types as et
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


def _pr_facet_frame() -> pd.DataFrame:
    """Progressive Ratio facet summary: Training/Test × paired/yoked."""
    rows = []
    for experiment in ("rep1", "rep2"):
        for facet in ("Training", "Test"):
            for role in ("paired", "yoked"):
                for treatment, base in (("Ctrl", 10.0), ("Exp", 40.0)):
                    for index in range(3):
                        rows.append({
                            "Experiment": experiment, "Facet": facet,
                            "FacetRange": "(0, 30)" if facet == "Training"
                            else "(30, inf)",
                            "Treatment": treatment, "Role": role,
                            "DFM": 1, "Chamber": index + 1,
                            "Group": 1,
                            "LicksA": base + index + (5 if role == "paired" else 0),
                            "LicksB": 5.0,
                            "EventsA": 2 + index, "EventsB": 1,
                            "MedDurationA": 1.0 + index * 0.1,
                            "MedDurationB": 0.5,
                            "PI": 0.2 if treatment == "Ctrl" else 0.7,
                        })
    return pd.DataFrame(rows)


def test_faceted_data_keeps_role_for_progressive_ratio():
    df = pubfigures.faceted_data(_pr_facet_frame(), "Licks")
    assert "Role" in df.columns
    ## Yoked (control) left of paired along the top of the facet grid.
    assert list(df["Role"].cat.categories) == ["yoked", "paired"]
    assert set(df["Phase"].astype(str)) == {"Training", "Test"}


def test_order_roles_puts_yoked_before_paired():
    assert pubfigures.order_roles(["paired", "yoked"]) == ["yoked", "paired"]
    assert pubfigures.order_roles(["paired"]) == ["paired"]


def test_role_grid_rows_follow_the_experiment_order():
    df = pubfigures.faceted_data(_pr_facet_frame(), "Licks")
    spec = pubfigures.default_spec("faceted_licks")
    ## Training ran first, so its row is drawn above Test's.
    figure = pubfigures.build_faceted(df, spec, pubfigures.PlotStyle())
    assert list(figure.data["Phase"].cat.categories) == ["Training", "Test"]


def test_role_display_labels_are_capitalized():
    assert pubfigures.role_display_label("yoked") == "Yoked"
    assert pubfigures.role_display_label("paired") == "Paired"
    assert pubfigures.role_display_label("YOKED") == "Yoked"


def test_faceted_data_without_role_stays_phase_only():
    df = pubfigures.faceted_data(_facet_frame(), "PI")
    assert "Role" not in df.columns


def test_build_faceted_filters_phase_and_role_independently(tmp_path: Path):
    df = pubfigures.faceted_data(_pr_facet_frame(), "Licks")
    spec = pubfigures.default_spec("faceted_licks")
    spec.facets = ["Test"]
    spec.roles = ["paired"]
    figure = pubfigures.build_figure("faceted_licks", df, spec,
                                     pubfigures.PlotStyle())
    out = pubfigures.save_figure(figure, str(tmp_path / "pr.svg"), "svg")
    assert Path(out).stat().st_size > 1000
    ## Filtering happens before draw: only the chosen cells remain.
    data, _labels, _colors = pubfigures._apply_treatment_order(
        df, spec, pubfigures.PlotStyle())
    include_phases = [p for p in (spec.facets or []) if p in data["Phase"].cat.categories]
    filtered = data[data["Phase"].isin(include_phases)]
    filtered = filtered[filtered["Role"].astype(str).isin(spec.roles)]
    assert set(filtered["Phase"].astype(str)) == {"Test"}
    assert set(filtered["Role"].astype(str)) == {"paired"}


def test_roles_round_trip_through_yaml(tmp_path: Path):
    specs = pubfigures.ProjectSpecs()
    specs.ensure_default_style()
    spec = pubfigures.default_spec("faceted_licks")
    spec.facets = ["Test"]
    spec.roles = ["paired", "yoked"]
    specs.plots["faceted_licks"] = spec
    pubfigures.save_project_specs(tmp_path, specs)
    loaded = pubfigures.load_project_specs(tmp_path)
    assert loaded.plots["faceted_licks"].facets == ["Test"]
    assert loaded.plots["faceted_licks"].roles == ["paired", "yoked"]


def test_effective_height_scales_with_role_rows():
    style = pubfigures.PlotStyle(facet_height_mm=40.0, height_mm=180.0)
    one = pubfigures.effective_height_mm(style, n_rows=1)
    two = pubfigures.effective_height_mm(style, n_rows=2)
    assert two > one
    assert two - one == pytest.approx(40.0)


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


def test_two_treatments_given_one_label_stay_two():
    """A label typed twice, or one that is another treatment's name, made the
    display categories repeat, and pd.Categorical refused them, so the figure
    failed; merging them would draw two treatments as one.  Each keeps its
    points, the repeated label told apart by the treatment's name."""
    df = pubfigures.faceted_data(_facet_frame(), "PI")
    spec = pubfigures.default_spec("faceted_pi")
    spec.treatments = {"Ctrl": {"label": "Fed", "show": True},
                       "Exp": {"label": "Fed", "show": True}}
    data, labels, colors = pubfigures._apply_treatment_order(
        df, spec, pubfigures.PlotStyle())
    assert labels == ["Fed (Ctrl)", "Fed (Exp)"] and len(colors) == 2
    counts = data["Treatment"].astype(str).value_counts()
    assert counts["Fed (Ctrl)"] == counts["Fed (Exp)"] == len(df) // 2
    figure = pubfigures.build_figure("faceted_pi", df, spec, pubfigures.PlotStyle())
    assert pubfigures.render_png_bytes(figure, pubfigures.PlotStyle())[:4] == b"\x89PNG"


def test_preview_png_is_returned_as_bytes():
    df = pubfigures.faceted_data(_facet_frame(), "PI")
    spec = pubfigures.default_spec("faceted_pi")
    figure = pubfigures.build_figure("faceted_pi", df, spec,
                                     pubfigures.PlotStyle())
    png = pubfigures.render_png_bytes(figure, pubfigures.PlotStyle())
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# Progressive Ratio breaking-point scatter
# ---------------------------------------------------------------------------

def _breaking_point_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "Experiment": ["rep1", "rep1", "rep2", "rep2"],
        "Treatment": ["Ctrl", "Exp", "Ctrl", "Exp"],
        "DFM": [1, 1, 2, 2],
        "Group": [1, 2, 1, 2],
        "PairedChamber": [1, 3, 1, 3],
        "BreakingPoint": [4, 8, 3, 10],
        "BreakMin": [40.0, 90.0, 55.0, 120.0],
        "Censored": [False, True, False, False],
        "TestMinutes": [200.0, 200.0, 180.0, 180.0],
    })


def test_breaking_point_scatter_is_the_ratio_modules_only():
    assert "scatter_pr_breaking_point" in pubfigures.plots_for_layout(
        "two_well", et.get_experiment_type("ProgressiveRatio"))
    assert "scatter_pr_breaking_point" not in pubfigures.plots_for_layout(
        "two_well", et.get_experiment_type("Hedonic"))
    assert "scatter_pr_breaking_point" not in pubfigures.plots_for_layout(
        "two_well")
    assert pubfigures.source_of("scatter_pr_breaking_point") == "pr_breaking"
    assert pubfigures.family_of("scatter_pr_breaking_point") == (
        pubfigures.FAMILY_SCATTER)


def test_breaking_point_data_tidies_and_keeps_censoring():
    df = pubfigures.breaking_point_data(_breaking_point_frame())
    assert set(df.columns) >= {"Treatment", "BreakMin", "BreakingPoint",
                               "Censored", "Experiment"}
    assert df["Censored"].tolist() == [False, True, False, False]
    assert list(df["BreakMin"]) == [40.0, 90.0, 55.0, 120.0]


def test_breaking_point_scatter_renders(tmp_path: Path):
    df = pubfigures.breaking_point_data(_breaking_point_frame())
    spec = pubfigures.default_spec("scatter_pr_breaking_point")
    figure = pubfigures.build_figure("scatter_pr_breaking_point", df, spec,
                                     pubfigures.PlotStyle())
    out = pubfigures.save_figure(figure, str(tmp_path / "bp.svg"), "svg")
    assert Path(out).stat().st_size > 1000


def test_breaking_point_by_treatment_is_the_ratio_modules_only():
    assert "dot_pr_breaking_point" in pubfigures.plots_for_layout("two_well", ("ratio",))
    assert "dot_pr_breaking_point" not in pubfigures.plots_for_layout(
        "two_well", ("hedonic",))
    assert pubfigures.source_of("dot_pr_breaking_point") == "pr_breaking"
    spec = pubfigures.default_spec("dot_pr_breaking_point")
    assert spec.x_label == ""
    assert spec.y_label == "Breaking point"


def test_breaking_point_by_treatment_renders(tmp_path: Path):
    df = pubfigures.breaking_point_data(_breaking_point_frame())
    spec = pubfigures.default_spec("dot_pr_breaking_point")
    figure = pubfigures.build_figure("dot_pr_breaking_point", df, spec,
                                     pubfigures.PlotStyle())
    out = pubfigures.save_figure(figure, str(tmp_path / "bp_tx.svg"), "svg")
    assert Path(out).stat().st_size > 1000
    ## Dropping a treatment leaves the others on the shared categorical axis.
    spec.treatments = {"Ctrl": {"label": "Ctrl", "show": True},
                       "Exp": {"label": "Exp", "show": False}}
    data, labels, _colors = pubfigures._apply_treatment_order(
        df, spec, pubfigures.PlotStyle())
    assert labels == ["Ctrl"]
    assert set(data["Treatment"].astype(str)) == {"Ctrl"}


def test_breaking_point_data_of_an_empty_frame_is_empty():
    df = pubfigures.breaking_point_data(pd.DataFrame())
    assert df.empty
    assert list(df.columns) == ["Treatment", "BreakMin", "BreakingPoint",
                                "Censored"]


def test_a_two_well_licks_figure_roots_the_summed_counts():
    """Review C2: on the fourth-root scale the two wells' Licks total is
    (A + B)^0.25, never A^0.25 + B^0.25; light and difference tables are left
    alone."""
    from types import SimpleNamespace

    frame = pd.DataFrame({"Treatment": ["Ctrl"], "LicksA": [16 ** 0.25],
                          "LicksB": [65 ** 0.25]})
    assert pubfigures.faceted_data(frame, "Licks", transformed=True)["Value"].iloc[0] \
        == pytest.approx(3.0)
    assert pubfigures.faceted_data(frame, "Licks")["Value"].iloc[0] \
        == pytest.approx(16 ** 0.25 + 65 ** 0.25)
    transformed, raw = SimpleNamespace(transform_licks=True), SimpleNamespace(transform_licks=False)
    assert pubfigures.licks_transformed(transformed, "faceted_licks")
    assert pubfigures.licks_transformed(transformed, "timecourse_licks")
    assert not pubfigures.licks_transformed(transformed, "faceted_licksperminlit")
    assert not pubfigures.licks_transformed(raw, "faceted_licks")
