"""Progressive Ratio: paired/yoked Chamber Groups, data-derived Training/Test
Facets, the Paired-Yoked Difference and its figures (ADR-0013)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from pyflic.base import experiment_types as et
from pyflic.base import pubfigures
from pyflic.base.experiment_types.progressive_ratio import (
    group_of,
    parse_paired_chambers,
    partner_of,
)
from pyflic.base.progressive_ratio_experiment import ProgressiveRatioExperiment
from pyflic.base.yaml_config import load_experiment_yaml
from pyflic.base.yaml_lint import lint_flic_config

from pr_fixtures import make_pr_experiment_dir, pr_config, write_pr_csv


# ---------------------------------------------------------------------------
# Type object and pure helpers
# ---------------------------------------------------------------------------

def test_groups_and_partners():
    assert [group_of(c) for c in range(1, 7)] == [1, 1, 2, 2, 3, 3]
    assert [partner_of(c) for c in range(1, 7)] == [2, 1, 4, 3, 6, 5]


@pytest.mark.parametrize("raw,fragment", [
    (None, "required"),
    ([1, 2, 5], "both chambers of group 1"),
    ([1, 3], "no chamber from group 3"),
    ([1, 3, 7], "outside 1-6"),
    ("1,3,5", "must be a list"),
])
def test_parse_paired_chambers_reports_each_problem(raw, fragment):
    paired, problems = parse_paired_chambers(raw)
    assert problems and any(fragment in p for p in problems)


def test_parse_paired_chambers_accepts_one_per_group():
    assert parse_paired_chambers([5, 1, 4]) == ([1, 4, 5], [])


def test_type_owns_data_derived_facets():
    pr = et.get_experiment_type("ProgressiveRatio")
    assert pr.data_derived_facets and pr.facets_fixed
    assert "facet_cutoffs" in pr.owned_keys()
    assert pr.resolve_facet_cutoffs({"facet_cutoffs": [30]}) is None
    problems = pr.validate({"facet_cutoffs": [30],
                            "well_names": {"A": "Sucrose", "B": "Yeast"}})
    assert any("facet_cutoffs" in p for p in problems)
    assert "facet_cutoffs" not in pr.build_global(well_names={"A": "S", "B": "Y"})
    assert pr.default_constants["require_training_complete"] is True
    assert pr.report_set("two_well")[0] == "timecourse_pr_diff"
    assert pr.report_facets() == ["Test"]


def test_validate_dfm_requires_shared_treatment_in_a_group():
    pr = et.get_experiment_type("ProgressiveRatio")
    problems = pr.validate_dfm(3, {"paired_chambers": [1, 3, 5]},
                               {1: "Ctrl", 2: "Exp", 3: "Ctrl", 4: "Ctrl"})
    assert len(problems) == 1
    assert "chambers 1 and 2" in problems[0] and "DFM 3" in problems[0]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pr_dir(tmp_path_factory) -> Path:
    return make_pr_experiment_dir(tmp_path_factory.mktemp("pr") / "exp")


@pytest.fixture(scope="module")
def exp(pr_dir) -> ProgressiveRatioExperiment:
    return load_experiment_yaml(pr_dir, parallel=False, use_disk_cache=False)


def test_loader_returns_the_pr_class(exp):
    assert isinstance(exp, ProgressiveRatioExperiment)
    assert exp.paired_chambers_by_dfm() == {1: [1, 4, 5], 2: [2, 3, 6]}
    roles = exp.roles_table()
    assert roles[(roles.DFM == 1) & (roles.Chamber == 4)].Role.item() == "paired"
    assert roles[(roles.DFM == 1) & (roles.Chamber == 3)].Role.item() == "yoked"
    assert roles[(roles.DFM == 2) & (roles.Chamber == 1)].Partner.item() == 2


def _write_config(root: Path, cfg: dict) -> None:
    (root / "flic_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def test_loader_refuses_missing_paired_chambers(tmp_path):
    root = tmp_path / "exp"
    write_pr_csv(root / "data", 1)
    cfg = pr_config([{"id": 1, "paired_chambers": [1, 3, 5],
                      "chambers": {1: "A", 2: "A", 3: "A", 4: "A", 5: "A", 6: "A"}}])
    del cfg["dfms"][0]["paired_chambers"]
    _write_config(root, cfg)
    with pytest.raises(ValueError, match="paired_chambers"):
        load_experiment_yaml(root, parallel=False, use_disk_cache=False)


def test_loader_refuses_split_treatment_in_a_group(tmp_path):
    root = tmp_path / "exp"
    write_pr_csv(root / "data", 1)
    cfg = pr_config([{"id": 1, "paired_chambers": [1, 3, 5],
                      "chambers": {1: "A", 2: "B", 3: "A", 4: "A", 5: "A", 6: "A"}}])
    _write_config(root, cfg)
    with pytest.raises(ValueError, match="chambers 1 and 2"):
        load_experiment_yaml(root, parallel=False, use_disk_cache=False)


def test_lint_reports_the_same_problems(tmp_path):
    root = tmp_path / "exp"
    root.mkdir()
    cfg = pr_config([{"id": 1, "paired_chambers": [1, 2, 5],
                      "chambers": {1: "A", 2: "B", 3: "A", 4: "A", 5: "A", 6: "A"}}])
    _write_config(root, cfg)
    issues = lint_flic_config(root / "flic_config.yaml")
    messages = [i.message for i in issues if i.severity == "error"]
    assert any("both chambers of group 1" in m for m in messages)
    assert any("chambers 1 and 2" in m for m in messages)
    assert not any("unknown key 'paired_chambers'" in i.message for i in issues)


# ---------------------------------------------------------------------------
# Training end and flag notes
# ---------------------------------------------------------------------------

def test_training_end_is_read_from_the_paired_sucrose_well(exp):
    table = exp.training_table().set_index(["DFM", "Group"])
    assert table.loc[(1, 1), "PairedChamber"] == 1
    assert table.loc[(1, 1), "SucroseWell"] == 1           # left well, pi left
    assert table.loc[(2, 1), "PairedChamber"] == 2
    assert table.loc[(2, 1), "SucroseWell"] == 4           # right well, pi right
    assert table.loc[(1, 1), "TrainingEndMin"] == pytest.approx(6.0, abs=0.02)
    assert table.loc[(1, 2), "TrainingEndMin"] == pytest.approx(9.0, abs=0.02)
    assert table.loc[(2, 2), "TrainingEndMin"] == pytest.approx(10.0, abs=0.02)
    assert table.loc[(2, 3), "TrainingComplete"] == False  # noqa: E712
    assert pd.isna(table.loc[(2, 3), "TrainingEndMin"])


def test_wells_that_never_clear_are_notes_not_errors(exp):
    notes = exp.training_warnings()
    assert any("DFM 1 group 1" in n and "stayed flagged" in n for n in notes)
    assert any("never completed" in n for n in notes)
    summary = exp.summary_text(include_qc=False)
    assert "training by chamber group" in summary
    assert "Training-flag notes" in summary


def test_all_four_wells_clearing_together_gives_no_notes(tmp_path):
    root = make_pr_experiment_dir(tmp_path / "exp", flag_mode="all_clear",
                                  incomplete_group_on_dfm2=False)
    e = load_experiment_yaml(root, parallel=False, use_disk_cache=False)
    assert e.training_warnings() == []
    assert e.training_table()["TrainingComplete"].all()


# ---------------------------------------------------------------------------
# Feeding summary, facets, difference
# ---------------------------------------------------------------------------

def test_feeding_summary_carries_the_pr_columns(exp):
    fs = exp.feeding_summary()
    for col in ("Group", "Role", "TrainingMinutes", "TrainingComplete", "LightOn_sec"):
        assert col in fs.columns
    paired = fs[fs.Role == "paired"]
    yoked = fs[fs.Role == "yoked"]
    assert len(paired) == 6 and len(yoked) == 6
    ## Training end belongs to the Chamber Group, so BOTH flies carry it —
    ## the yoked cell was NA and read as missing data everywhere downstream.
    complete = fs[fs.TrainingComplete]
    assert complete["TrainingMinutes"].notna().all()
    assert set(complete.Role) == {"paired", "yoked"}
    ## A group that never finished has no such moment; neither row invents one.
    assert fs[~fs.TrainingComplete]["TrainingMinutes"].isna().all()
    assert (fs["LightOn_sec"] > 0).all()
    ## Both chambers of a group share the light and the training end.
    for (_, _), sub in fs.groupby(["DFM", "Group"]):
        assert sub["LightOn_sec"].nunique() == 1
        assert sub["TrainingMinutes"].nunique(dropna=False) == 1
    ## Idempotent on the cache: a second call returns the same augmented frame.
    assert list(exp.feeding_summary().columns) == list(fs.columns)


def test_facets_are_split_per_group_at_training_end(exp):
    ff = exp.feeding_summary_facet()
    assert set(ff["Facet"]) == {"Training", "Test"}
    ## Two windows per complete group (5 groups), one for the incomplete group.
    assert len(ff) == 5 * 2 * 2 + 2
    g11 = ff[(ff.DFM == 1) & (ff.Group == 1)]
    assert g11[g11.Facet == "Training"]["EndMin"].iloc[0] == pytest.approx(6.0, abs=0.02)
    assert g11[g11.Facet == "Test"]["StartMin"].iloc[0] == pytest.approx(6.0, abs=0.02)
    g12 = ff[(ff.DFM == 1) & (ff.Group == 2)]
    assert g12[g12.Facet == "Test"]["StartMin"].iloc[0] == pytest.approx(9.0, abs=0.02)
    incomplete = ff[(ff.DFM == 2) & (ff.Group == 3)]
    assert set(incomplete["Facet"]) == {"Training"}
    ## Every faceted row of a complete group carries its training end, in
    ## both Facets and for both roles.
    done = ff[ff.TrainingComplete]
    assert done["TrainingMinutes"].notna().all()
    for (_, _), sub in done.groupby(["DFM", "Group"]):
        assert sub["TrainingMinutes"].nunique() == 1
    ## The tail window is not empty (regression: open-ended range read as 0).
    assert (ff[ff.Facet == "Test"]["LicksA"] > 0).all()


def test_paired_yoked_diff_is_one_row_per_group_per_facet(exp):
    d = exp.paired_yoked_diff()
    assert len(d) == 5 * 2 + 1
    assert {"dLicksA", "dPI", "PairedChamber", "YokedChamber", "TrainingMinutes",
            "LightOn_sec", "Genotype"}.issubset(d.columns)
    test = d[d.Facet == "Test"]
    ## Paired flies feed more at sucrose in the fixture, so the difference is
    ## positive within every group.
    assert (test["dLicksA"] > 0).all()
    assert (test["dLicksB"] == 0).all()
    row = test[(test.DFM == 2) & (test.Group == 1)].iloc[0]
    assert row.PairedChamber == 2 and row.YokedChamber == 1


def test_diff_skips_a_group_missing_a_chamber(tmp_path):
    root = make_pr_experiment_dir(tmp_path / "exp", incomplete_group_on_dfm2=False)
    cfg = yaml.safe_load((root / "flic_config.yaml").read_text())
    del cfg["dfms"][0]["chambers"][2]      # yoked chamber of DFM 1 group 1 unassigned
    _write_config(root, cfg)
    e = load_experiment_yaml(root, parallel=False, use_disk_cache=False)
    d = e.paired_yoked_diff()
    assert d[(d.DFM == 1) & (d.Group == 1)].empty
    assert len(d) == 5 * 2


# ---------------------------------------------------------------------------
# Auto-removal
# ---------------------------------------------------------------------------

def test_incomplete_training_group_is_auto_removed_by_default(tmp_path):
    root = make_pr_experiment_dir(tmp_path / "exp")
    e = load_experiment_yaml(root, parallel=False, use_disk_cache=False)
    removed = e.auto_remove_chambers()
    keys = set(zip(removed.DFM.astype(int), removed.Chamber.astype(int)))
    assert keys == {(2, 5), (2, 6)}
    assert "require_training_complete" in removed.Reason.iloc[0]
    assert (root / "analysis" / "removed_chambers.csv").is_file()
    assert len(e.feeding_summary()) == 10


def test_require_training_complete_can_be_switched_off(tmp_path):
    root = make_pr_experiment_dir(tmp_path / "exp",
                                  constants={"require_training_complete": False})
    e = load_experiment_yaml(root, parallel=False, use_disk_cache=False)
    removed = e.auto_remove_chambers()
    assert removed.empty
    assert len(e.feeding_summary()) == 12


# ---------------------------------------------------------------------------
# Curves, figures, breaking point, pipeline
# ---------------------------------------------------------------------------

def test_cumulative_curves_start_at_training_end(exp):
    curves = exp.cumulative_curve_data(binsize_min=1.0)
    assert {"Role", "LightOn", "CumLicks", "Minutes"}.issubset(curves.columns)
    assert curves[(curves.DFM == 2) & (curves.Group == 3)].empty   # never trained
    g = curves[(curves.DFM == 1) & (curves.Group == 1)]
    assert g["Minutes"].min() == pytest.approx(1.0)
    assert g["LightOn"].any()
    assert (g.groupby("Role")["CumLicks"].apply(lambda s: s.is_monotonic_increasing)).all()
    diff = exp.cumulative_diff_data()
    assert set(diff.columns) >= {"Treatment", "Genotype", "DFM", "Group", "Minutes",
                                 "DiffCumLicks"}
    assert diff.groupby(["DFM", "Group"]).ngroups == 5
    assert diff.groupby(["DFM", "Group"])["DiffCumLicks"].last().gt(0).all()


def test_figures_build(exp):
    exp.plot_cumulative_diff().draw()
    exp.plot_cumulative_licks_dfm(1).draw()
    exp.plot_breaking_point_dfm(1).draw()


# ---------------------------------------------------------------------------
# The standard per-treatment plots never pool paired with yoked
# ---------------------------------------------------------------------------

def test_treatment_grouping_carries_the_role(exp):
    """A Treatment names both flies of a Chamber Group, so grouping by
    Treatment alone draws one cloud holding an effect and its own control."""
    summary = exp.feeding_summary()
    grouped, col = exp._resolve_group_col(summary)
    assert col == "_RoleGroup"
    assert set(grouped[col]) == {"w1118 · paired", "w1118 · yoked",
                                 "mut · paired", "mut · yoked"}


def test_the_binned_treatment_table_carries_the_role(exp):
    """The time courses group through the same column, so the split has to
    reach the binned table too — not just the per-chamber summary."""
    binned = exp._binned_licks_table_by_treatment(binsize_min=10.0)
    assert {"Group", "Role"}.issubset(binned.columns)
    assert set(binned["Role"]) == {"paired", "yoked"}
    _, col = exp._resolve_group_col(binned)
    assert col == "_RoleGroup"


def test_delta_is_one_row_per_group_per_facet_and_matches_the_diff_table(exp):
    delta = exp.paired_yoked_delta(metric="PI")
    diff = exp.paired_yoked_diff()
    keys = ["DFM", "Group", "Facet"]
    assert len(delta) == len(diff)
    merged = delta.merge(diff[[*keys, "dPI"]], on=keys)
    assert len(merged) == len(delta)
    assert (merged["Delta"] - merged["dPI"]).abs().max() < 1e-12
    ## Composite metrics have no stored d-column; they are differenced here.
    licks = exp.paired_yoked_delta(metric="Licks", two_well_mode="total")
    assert (licks["Delta"] - (licks["Paired"] - licks["Yoked"])).abs().max() == 0


def test_an_explicit_window_replaces_the_facets(exp):
    """The Facets are per Chamber Group, so one shared window is a different
    question rather than a filter on the same one."""
    windowed = exp.paired_yoked_delta(metric="PI", range_minutes=(0, 20))
    assert set(windowed["Facet"]) == {"Custom"}
    assert windowed.groupby(["DFM", "Group"]).size().max() == 1


def test_the_dot_plot_is_the_within_group_difference(exp):
    """One point is one Chamber Group's paired-minus-yoked value — the unit
    CONTEXT.md fixes for this type — not one fly."""
    plot = exp.plot_dot_metric_by_treatment(metric="PI")
    assert plot.mapping["y"] == "Delta"
    drawn = plot.data
    expected = exp.paired_yoked_delta(metric="PI")
    assert len(drawn) == len(expected)
    assert list(drawn["_Panel"].cat.categories) == ["Training", "Test"]
    plot.draw()


def test_breaking_point_table_is_per_light_period_after_training(exp):
    bp = exp.breaking_point_table(1, 1)
    ## Chamber 1 is paired: its rows are the group's Light Event Ledger.
    assert list(bp.columns) == ["Minutes", "CumLicks", "DeltaMinutes", "DeltaLicks",
                                "MinutesSincePrev", "LicksSincePrev", "LickFree",
                                "RestingLevel"]
    assert (bp["Minutes"] > 0).all()
    assert bp["DeltaLicks"].iloc[0] == 0.0 and (bp["DeltaLicks"].iloc[1:] > 0).all()
    assert (bp["LicksSincePrev"] > 0).all() and not bp["LickFree"].any()
    ## The yoked chamber keeps the plain breaking-point table.
    assert list(exp.breaking_point_table(1, 2).columns) == [
        "Minutes", "CumLicks", "DeltaMinutes", "DeltaLicks"]
    assert exp.breaking_point_table(2, 5).empty       # group never trained


def test_basic_analysis_writes_the_pr_outputs(tmp_path):
    root = make_pr_experiment_dir(tmp_path / "exp")
    e = load_experiment_yaml(root, parallel=False, use_disk_cache=False)
    e.execute_basic_analysis(skip_qc=True)
    analysis = root / "analysis"
    for name in e.experiment_type.output_manifest():
        assert (analysis / name).is_file(), name
    assert (analysis / "pr_cumulative_licks_dfm1.png").is_file()
    diff = pd.read_csv(analysis / "paired_yoked_diff.csv")
    assert "dLicksA" in diff.columns
    curve = pd.read_csv(analysis / "pr_cumulative_diff.csv")
    assert {"Minutes", "DiffCumLicks", "Treatment"}.issubset(curve.columns)


# ---------------------------------------------------------------------------
# Publication figures and gating
# ---------------------------------------------------------------------------

def test_pr_diff_plot_spec_is_type_gated():
    assert "timecourse_pr_diff" in pubfigures.plots_for_layout("two_well", "ProgressiveRatio")
    assert "timecourse_pr_diff" not in pubfigures.plots_for_layout("two_well", "Hedonic")
    assert "timecourse_pr_diff" not in pubfigures.plots_for_layout("two_well")
    assert pubfigures.source_of("timecourse_pr_diff") == "pr_diff"
    assert pubfigures.source_of("timecourse_licks") == "binned"
    assert pubfigures.source_of("faceted_licks") == "facet"
    spec = pubfigures.default_spec("timecourse_pr_diff")
    assert "training end" in spec.x_label


def test_pr_diff_figure_builds_from_the_curve_frame(tmp_path):
    root = make_pr_experiment_dir(tmp_path / "exp")
    e = load_experiment_yaml(root, parallel=False, use_disk_cache=False)
    frame = e.cumulative_diff_data()
    frame.insert(0, "Experiment", "rep1")
    df = pubfigures.timecourse_data(frame, "DiffCumLicks")
    assert not df.empty
    spec = pubfigures.default_spec("timecourse_pr_diff")
    style = pubfigures.PlotStyle()
    pubfigures.build_figure("timecourse_pr_diff", df, spec, style).draw()


# ---------------------------------------------------------------------------
# Project level: pooling, statistics, report
# ---------------------------------------------------------------------------

def test_project_pools_the_difference_and_makes_it_primary(tmp_path):
    from pyflic.base.project import Project
    from pyflic.base.project_report import write_project_report

    root = tmp_path / "proj"
    root.mkdir()
    members = {}
    for name in ("rep1", "rep2"):
        make_pr_experiment_dir(root / name, incomplete_group_on_dfm2=(name == "rep1"))
        cfg = yaml.safe_load((root / name / "flic_config.yaml").read_text())
        members[name] = cfg.pop("global")          # a Member inherits global:
        (root / name / "flic_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (root / "project.yaml").write_text(yaml.safe_dump(
        {"name": "proj", "design": {"global": members["rep1"]}}, sort_keys=False))

    project = Project(root)
    assert project.experiment_type.name == "ProgressiveRatio"
    assert project.shared_windows() == (None, None)
    for name in project.member_names:
        e = project.load_member(name, parallel=False, use_disk_cache=False)
        assert isinstance(e, ProgressiveRatioExperiment)
        e.execute_basic_analysis(skip_qc=True)

    result = project.build_combined_analysis()
    written = {Path(p).name for p in result["written"]}
    assert "proj_PairedYokedDiff.csv" in written
    diff = pd.read_csv(root / "analysis" / "proj_PairedYokedDiff.csv")
    assert set(diff["Experiment"]) == {"rep1", "rep2"}
    ## rep1's never-trained group leaves through the pipeline's auto-removal
    ## (require_training_complete), so it contributes no Training row.
    assert len(diff) == (5 * 2) + (6 * 2)
    assert "proj_LightQC.csv" in written

    summary, facet, _missing = project.combined_frames()
    labels = [label for label, _ in project._facet_frames(summary, facet)]
    assert labels == ["Training", "Test"]           # by label, not by FacetRange

    rows = project.diff_comparison_rows(diff)
    assert rows and all(r["phase"] == "Test" for r in rows)
    assert {r["metric"] for r in rows} >= {"dLicksA"}
    text = (root / "analysis" / "proj_Stats.txt").read_text(encoding="utf-8")
    assert text.index("Paired − yoked difference (primary") < text.index("Per-chamber metrics (secondary)")

    pdf = write_project_report(project, log=lambda *_a, **_k: None)
    assert Path(pdf).is_file()


# ---------------------------------------------------------------------------
# Common-range truncation and the Script Editor's type key
# ---------------------------------------------------------------------------

def test_pooled_pr_diff_mean_stops_at_the_shortest_series():
    rows = []
    for series, last in (("a", 5), ("b", 5), ("c", 3)):     # c ends early
        for minute in range(1, last + 1):
            rows.append({"Experiment": "rep1", "Treatment": "Ctrl", "DFM": 1,
                         "Group": series, "Minutes": float(minute),
                         "DiffCumLicks": 10.0 * minute})
    frame = pd.DataFrame(rows)
    tidy = pubfigures.timecourse_data(frame, "DiffCumLicks")
    kept = pubfigures.common_range_rows(tidy)
    assert kept["Minutes"].max() == 3.0
    spec = pubfigures.default_spec("timecourse_pr_diff")
    assert spec.common_range is True
    assert pubfigures.default_spec("timecourse_licks").common_range is False
    pubfigures.build_figure("timecourse_pr_diff", tidy, spec, pubfigures.PlotStyle()).draw()


def test_member_diff_curve_mean_is_truncated_but_traces_are_not(exp):
    data = exp.cumulative_diff_data()
    ends = data.groupby(["DFM", "Group"])["Minutes"].max()
    assert ends.nunique() > 1                     # groups really do end apart
    stat = exp.cumulative_diff_stat()
    assert stat["Minutes"].max() == pytest.approx(ends.min())
    ## Every averaged point has every group of its treatment in it.
    n_groups = data.groupby("Treatment")[["DFM", "Group"]].nunique().max(axis=1)
    for treatment, sub in stat.groupby("Treatment"):
        assert (sub["count"] == n_groups[treatment]).all()
    exp.plot_cumulative_diff().draw()


@pytest.mark.parametrize("spelling,key", [
    ("ProgressiveRatio", "progressive_ratio"),
    ("progressive_ratio", "progressive_ratio"),
    ("Progressive-Ratio", "progressive_ratio"),
    ("Hedonic", "hedonic"),
    ("", None),
    (None, None),
    ("Custom", None),
])
def test_script_editor_type_key_matches_the_registry_spelling(spelling, key):
    from pyflic.base.script_editor.actions import requires_key_for

    assert requires_key_for(spelling) == key


def test_dfm_qc_figures_are_built_without_pyplot_and_save_in_a_thread(exp):
    """The Hub writes QC reports in a worker thread; a pyplot-managed figure
    there is a Qt canvas off the GUI thread and fails in FreeType."""
    import threading

    import matplotlib.pyplot as plt

    dfm = exp.dfms[1]
    before = set(plt.get_fignums())
    figs = [dfm.plot_raw(), dfm.plot_baselined(include_thresholds=True),
            dfm.plot_cumulative_licks(transform_licks=True)]
    assert set(plt.get_fignums()) == before          # nothing registered with pyplot
    errors: list[BaseException] = []

    def work():
        try:
            import io
            for fig in figs:
                fig.savefig(io.BytesIO(), format="png", dpi=100, bbox_inches="tight")
        except BaseException as err:  # noqa: BLE001
            errors.append(err)

    t = threading.Thread(target=work)
    t.start()
    t.join()
    assert errors == []


def test_every_app_draws_through_agg_not_a_gui_backend():
    """Left to choose, matplotlib picks QtAgg once PyQt6 is imported, and then
    every pyplot figure built on a worker thread — the PDF report's pages,
    plotnine's own — warns that it is starting a GUI off the main thread.  No
    pyflic app shows a figure through pyplot (both Hubs embed their own
    ``FigureCanvasQTAgg``), so Agg is the honest backend and has no GUI to
    start."""
    import os
    import subprocess
    import sys

    code = (
        "import pyflic.base.hub, pyflic.base.qc_viewer, pyflic.base.plot_editor\n"
        "import matplotlib\n"
        "print(matplotlib.get_backend().lower())\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True,
                         env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})
    assert out.returncode == 0, out.stderr[-800:]
    assert out.stdout.strip().endswith("agg"), out.stdout
    assert "qtagg" not in out.stdout


def test_a_report_page_is_built_without_pyplot(tmp_path):
    """Report pages are built straight from ``Figure`` and register with
    nothing, so the Hub's worker thread never asks matplotlib for a canvas it
    cannot make."""
    import matplotlib.pyplot as plt

    from pyflic.base import report_layout as rl

    before = set(plt.get_fignums())
    doc = rl.ReportDocument("Experiment report", "x")
    doc.add(rl.Cover("Experiment report", "x"), rl.Heading("One"), rl.Paragraph("text"))
    doc.save(tmp_path / "r.pdf")
    assert set(plt.get_fignums()) == before


def test_matplotlib_loads_before_qt_in_every_app_module():
    """Two FreeType copies (matplotlib's bundled one, Qt's system one) bind
    each other's calls by load order; Qt first breaks matplotlib's text
    renderer (``FT_Render_Glyph … raster overflow``).  ``pyflic.base`` imports
    matplotlib first, and every app module imports ``pyflic.base`` first."""
    import subprocess
    import sys

    code = (
        "import sys, pyflic.base.hub, pyflic.base.qc_viewer, pyflic.base.config_editor, "
        "pyflic.base.plot_editor\n"
        "order = list(sys.modules)\n"
        "print(order.index('matplotlib.ft2font') < order.index('PyQt6.QtCore'))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**__import__('os').environ, "QT_QPA_PLATFORM": "offscreen"})
    assert out.returncode == 0, out.stderr[-800:]
    assert out.stdout.strip().endswith("True"), out.stdout
