"""The yoked comparison for every Experiment Type (docs/paradigm-analyses-plan.md,
stream S5; decision 12), and the Acclimation count of the light table.

Generation 3 from the firmware 4.0.0 simulator (``v4_fixtures``):
``yoked_controls_spec`` (chambers 1-3 trigger at their A wells, chamber c + 3
gets chamber c's light) loaded as a Custom experiment, then variants with
two yoked chambers in a group, Program Facets, two DFMs and one treatment.
Progressive Ratio keeps its own comparison and its own rows.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import stats

from pyflic.base import analytics as an
from pyflic.base import light_summary as ls
from pyflic.base import report_layout as rl
from pyflic.base import yoked_report as yr
from pyflic.base.yaml_config import load_experiment_yaml

from pr_fixtures import make_pr_experiment_dir
import v4_fixtures as vf

MINUTES = 12.0
#: The two-well feeding-summary columns the comparison takes by default.
TWO_WELL_METRICS = ("LicksA", "LicksB", "EventsA", "EventsB", "MedDurationA", "MedDurationB")
#: yoked_controls_spec's Linkage Groups: label -> (Paired, Yoked chambers).
YOKED_GROUPS = {1: (1, (4,)), 2: (2, (5,)), 3: (3, (6,))}


def _treatments(dfm_ids, chambers=None) -> list[dict]:
    chambers = chambers or {c: ("Ctrl" if c % 2 else "Exp") for c in range(1, 7)}
    return [{"id": int(i), "chambers": dict(chambers)} for i in dfm_ids]


def _config(dfm_ids, chambers=None, **extra) -> dict:
    params = {"feeding_threshold": 20, "feeding_minimum": 10, "tasting_minimum": 5,
              "tasting_maximum": 20, "samples_per_second": 5, "pi_direction": "left"}
    return {"global": {"chamber_layout": "two_well", "transform_licks": False,
                       "params": params, **extra},
            "dfms": _treatments(dfm_ids, chambers)}


def _write(root: Path, spec, results, cfg: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    vf.write_v4_recording(root / "data", spec, results=results)
    (root / "flic_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return root


def _load(root: Path):
    return load_experiment_yaml(root, parallel=False, use_disk_cache=False)


@lru_cache(maxsize=None)
def _simulated(name: str):
    spec = SPECS[name]()
    return spec, vf.simulate(spec)


def _two_yoked_spec():
    """Group 1 lights chambers 1, 4 and 6 (Paired 1, Yoked 4 and 6), group 2
    chambers 2 and 5; chamber 3 is alone in group 3 (Self)."""
    spec = vf.yoked_controls_spec(minutes=MINUTES)
    (dfm,) = spec.dfms
    return replace(spec, dfms=(replace(dfm, linkage=(1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 1, 1)),))


def _two_paradigm_spec():
    """The yoked design under two Paradigms of six minutes each, so Program
    Facets split it in two."""
    spec = vf.yoked_controls_spec(minutes=MINUTES)
    (dfm,) = spec.dfms
    thresholds = dfm.intervals[0].thresholds
    second = replace(vf.CLOSED, name="Closed2", hold_ms=300)
    intervals = (vf.IntervalSpec("Closed", True, thresholds, MINUTES / 2),
                 vf.IntervalSpec("Closed2", True, thresholds, MINUTES / 2))
    return replace(spec, paradigms={**spec.paradigms, "Closed2": second},
                   dfms=(replace(dfm, intervals=intervals),))


def _two_dfm_spec():
    """yoked_controls_spec on two DFMs, so each treatment has two Paired
    chambers or more for the light statistics."""
    spec = vf.yoked_controls_spec(minutes=MINUTES)
    (dfm,) = spec.dfms
    return replace(spec, dfms=(dfm, replace(dfm, dfm_id=2)))


SPECS = {
    "yoked": lambda: vf.yoked_controls_spec(minutes=MINUTES),
    "two_yoked": _two_yoked_spec,
    "two_paradigms": _two_paradigm_spec,
    "two_dfms": _two_dfm_spec,
    "choice": vf.choice_pr_spec,
}


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    """``get(name, **config) -> experiment``, each loaded once."""
    cache: dict = {}

    def get(name: str, chambers=None, **extra):
        key = (name, repr(chambers), repr(sorted(extra.items())))
        if key not in cache:
            spec, results = _simulated(name)
            cfg = _config([d.dfm_id for d in spec.dfms], chambers, **extra)
            root = _write(tmp_path_factory.mktemp(name), spec, results, cfg)
            cache[key] = _load(root)
        return cache[key]

    return get


# ---------------------------------------------------------------------------
# Which groups
# ---------------------------------------------------------------------------

def test_the_yoked_groups_are_the_linkage_groups(loaded):
    e = loaded("yoked")
    assert e.experiment_type.name == "Custom"
    assert an.yoked_groups(e) == [an.YokedGroup(1, label, paired, yoked)
                                  for label, (paired, yoked) in YOKED_GROUPS.items()]
    ## Two Yoked chambers in one group; a Self chamber is in none.
    assert an.yoked_groups(loaded("two_yoked")) == [an.YokedGroup(1, 1, 1, (4, 6)),
                                                    an.YokedGroup(1, 2, 2, (5,))]


# ---------------------------------------------------------------------------
# The hand computation
# ---------------------------------------------------------------------------

def _by_hand(summary: pd.DataFrame, groups: dict) -> pd.DataFrame:
    """Paired minus the mean of the yoked chambers, per group, from the
    feeding summary's rows."""
    rows = []
    for label, (paired, yoked) in groups.items():
        p = summary[summary["Chamber"] == paired].iloc[0]
        y = summary[summary["Chamber"].isin(yoked)]
        row = {"Group": label, "Treatment": p["Treatment"]}
        for m in TWO_WELL_METRICS:
            row[f"Paired{m}"] = float(p[m])
            row[f"Yoked{m}"] = float(y[m].mean())
            row[f"d{m}"] = float(p[m]) - float(y[m].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _assert_rows(ours: pd.DataFrame, expected: pd.DataFrame) -> None:
    ours = ours.reset_index(drop=True)
    assert len(ours) == len(expected)
    for column in expected.columns:
        if column in ("Group", "Treatment"):
            assert ours[column].tolist() == expected[column].tolist(), column
        else:
            np.testing.assert_allclose(ours[column].to_numpy(float),
                                       expected[column].to_numpy(float), rtol=0, atol=1e-12,
                                       err_msg=column)


def test_the_comparison_is_a_hand_computation_from_the_feeding_summary(loaded):
    """The acceptance: on yoked_controls_spec under Custom, per group, the
    paired chamber's feeding-summary value, its yoked partner's, and the
    difference, for Licks, Events and MedDuration of each well."""
    e = loaded("yoked")
    table = an.yoked_comparison(e)
    summary = e.feeding_summary()
    assert list(table.columns[:7]) == list(an.YOKED_COLUMNS)
    assert [c for c in table.columns if c.startswith("d")] == [f"d{m}" for m in TWO_WELL_METRICS]
    assert set(table["Facet"]) == {an.WHOLE_RECORDING}
    assert table["PairedChamber"].tolist() == [1, 2, 3]
    assert table["YokedChambers"].tolist() == ["4", "5", "6"]
    assert set(summary.loc[summary["Chamber"].isin([1, 2, 3]), "Role"]) == {"Paired"}
    _assert_rows(table, _by_hand(summary, YOKED_GROUPS))
    ## The treatment is the paired chamber's (chamber 1 Ctrl, its partner Exp).
    assert table["Treatment"].tolist() == ["Ctrl", "Exp", "Ctrl"]


def test_two_yoked_chambers_are_averaged(loaded):
    e = loaded("two_yoked")
    table = an.yoked_comparison(e)
    groups = {1: (1, (4, 6)), 2: (2, (5,))}
    assert table["YokedChambers"].tolist() == ["4, 6", "5"]
    _assert_rows(table, _by_hand(e.feeding_summary(), groups))


@pytest.mark.parametrize("facets", [{"facet_cutoffs": [5]}, {"facet_source": "program"}])
def test_each_facet_is_a_window_of_its_own(loaded, facets):
    """Cutoff Facets and Program Facets: after the whole recording, one row
    per group and Facet, each a hand computation from that Facet's rows of
    the faceted feeding summary."""
    e = loaded("two_paradigms", **facets)
    table = an.yoked_comparison(e)
    facet = e.feeding_summary_facet()
    labels = list(dict.fromkeys(facet["Facet"].astype(str)))
    assert len(labels) == 2
    assert list(dict.fromkeys(table["Facet"])) == [an.WHOLE_RECORDING, *labels]
    _assert_rows(table[table["Facet"] == an.WHOLE_RECORDING],
                 _by_hand(e.feeding_summary(), YOKED_GROUPS))
    for label in labels:
        rows = facet[facet["Facet"] == label]
        ours = table[table["Facet"] == label]
        assert set(ours["FacetRange"]) == set(rows["FacetRange"])
        _assert_rows(ours, _by_hand(rows, YOKED_GROUPS))


def test_a_metric_can_be_named(loaded):
    e = loaded("yoked")
    table = an.yoked_comparison(e, metrics=["PI", "Licks"])
    assert [c for c in table.columns if c.startswith("d")] == ["dPI", "dLicksA", "dLicksB"]
    with pytest.raises(ValueError, match="not in the feeding summary"):
        an.yoked_comparison(e, metrics=["NoSuchMetric"])


def test_an_excluded_partner_takes_its_group_out(loaded, monkeypatch):
    e = loaded("yoked")
    summary = e.feeding_summary()
    monkeypatch.setattr(type(e), "feeding_summary",
                        lambda self, **kw: summary[summary["Chamber"] != 5].reset_index(drop=True))
    table = an.yoked_comparison(e)
    assert table["Group"].tolist() == [1, 3]


# ---------------------------------------------------------------------------
# The tests: Progressive Ratio's
# ---------------------------------------------------------------------------

def test_the_table_tests_with_progressive_ratios_function(loaded):
    """The table's p-values are analytics.zero_tests' (the one-sample t-test
    of the differences, the Wilcoxon beside it), on the groups of each
    treatment; a treatment with one group is shown, untested."""
    e = loaded("yoked")
    table = an.yoked_comparison(e)
    frames = yr.yoked_windows(table)
    assert [label for label, _rows in frames] == [an.WHOLE_RECORDING]
    rows = yr.difference_rows(frames, yr.difference_columns(table))
    tested = {(r["metric"], r["treatment"]): r
              for r in an.zero_tests(frames, yr.difference_columns(table))}
    for r in rows:
        values = table.loc[table["Treatment"] == r["treatment"], r["metric"]].to_numpy(float)
        assert r["n"] == values.size and r["mean"] == pytest.approx(values.mean())
        found = tested.get((r["metric"], r["treatment"]))
        if r["treatment"] == "Ctrl" and found is not None:
            assert r == found
            assert r["p_t"] == pytest.approx(stats.ttest_1samp(values, 0.0).pvalue)
        else:
            assert np.isnan(r["p_t"]), r
    assert {r["treatment"] for r in rows} == {"Ctrl", "Exp"}


def _tables(blocks) -> list[rl.Table]:
    return [b for b in blocks if isinstance(b, rl.Table)]


def _texts(blocks) -> list[str]:
    return [b.text for b in blocks if isinstance(b, rl.Paragraph)]


def test_the_blocks_compare_treatments_and_draw_the_identity_line(loaded):
    e = loaded("two_dfms")
    table = an.yoked_comparison(e)
    assert len(table) == 6
    blocks = yr.yoked_table_blocks(table, well_names=e.well_names)
    captions = [t.caption for t in _tables(blocks)]
    assert captions == ["Paired − yoked difference by treatment (one observation per group)",
                        "Treatment comparisons: paired − yoked difference"]
    by_treatment = _tables(blocks)[0].frame
    assert set(by_treatment["Treatment"]) == {"Ctrl", "Exp"}
    assert not any(t.startswith("One treatment") for t in _texts(blocks))
    figures = yr.yoked_figure_blocks(table, well_names=e.well_names)
    assert len(figures) == 1 and isinstance(figures[0], rl.Plot)
    figure = figures[0].plot()
    layers = [type(layer.geom).__name__ for layer in figure.layers]
    assert "geom_abline" in layers and "geom_point" in layers
    image, _w, _h = rl.render_plot(figures[0].plot, 6.0, 3.0, drop_title=True)
    assert image.size


def test_the_whole_recording_rows_read_as_the_slices_module_names_them(loaded):
    """Review C14: one label for the unfaceted rows (the Slices module's),
    and their FacetRange an open window, not the (0, 0) sentinel."""
    from pyflic.base import slices

    assert an.WHOLE_RECORDING == slices.WHOLE_RECORDING_LABEL
    table = an.yoked_comparison(loaded("two_dfms"))
    whole = table[table["Facet"] == slices.WHOLE_RECORDING_LABEL]
    assert len(whole) and set(whole["FacetRange"]) == {"(0, inf)"}


def test_transformed_lick_differences_say_they_are_on_the_root_scale(loaded):
    """Review C2 (owner decision): the yoked comparison keeps paired − yoked
    of transformed licks on the ¼-root scale, and its labels say so — only
    when the licks were transformed."""
    e = loaded("two_dfms")
    table = an.yoked_comparison(e, transform_licks=True)
    names = e.well_names
    measures = set(_tables(yr.yoked_table_blocks(table, well_names=names,
                                                 transformed=True))[0].frame["Measure"])
    licks = {m for m in measures if "licks" in m}
    assert licks and all("¼-root scale" in m for m in licks)
    assert not any("¼-root" in m for m in measures - licks)
    plain = set(_tables(yr.yoked_table_blocks(table, well_names=names))[0].frame["Measure"])
    assert not any("¼-root" in m for m in plain)
    figure = yr.yoked_figure(table, well_names=names, transformed=True)
    panels = set(figure.data["Metric"].astype(str))
    assert any("licks" in p and "¼-root scale" in p for p in panels)
    assert not any("¼-root" in p for p in panels if "licks" not in p)


def test_a_single_treatment_gets_the_note_not_a_comparison(loaded):
    e = loaded("yoked", chambers={c: "Ctrl" for c in range(1, 7)})
    table = an.yoked_comparison(e)
    assert set(table["Treatment"]) == {"Ctrl"}
    blocks = yr.yoked_table_blocks(table, well_names=e.well_names)
    assert any(t.startswith("One treatment, Ctrl, so nothing is compared")
               for t in _texts(blocks))
    ## Nothing is compared between treatments; the difference is still
    ## tested against zero, which is the design's question.
    assert [t.caption for t in _tables(blocks)] == [
        "Paired − yoked difference by treatment (one observation per group)"]
    assert not any("Welch" in t or "Tukey" in t for t in _texts(blocks))


def test_nothing_to_draw_without_a_value():
    frame = pd.DataFrame({"Facet": [an.WHOLE_RECORDING], "Treatment": ["Ctrl"],
                          "PairedLicks": [np.nan], "YokedLicks": [3.0], "dLicks": [np.nan]})
    with pytest.raises(rl.NothingToDraw):
        yr.yoked_figure(frame)


# ---------------------------------------------------------------------------
# The CSV, basic analysis and the Project
# ---------------------------------------------------------------------------

def test_basic_analysis_writes_the_csv(tmp_path):
    spec, results = _simulated("yoked")
    e = _load(_write(tmp_path / "exp", spec, results, _config([1])))
    result = e.execute_basic_analysis(skip_qc=True)
    path = Path(e.analysis_dir) / an.YOKED_COMPARISON_FILE
    assert result["yoked_comparison"] == path.resolve()
    written = pd.read_csv(path)
    expected = an.yoked_comparison(e)
    assert list(written.columns) == list(expected.columns)
    np.testing.assert_allclose(written["dLicksA"].to_numpy(float),
                               expected["dLicksA"].to_numpy(float))


def test_without_yoking_there_is_no_csv_and_no_block(loaded, tmp_path):
    """choice_pr_spec lights each chamber alone (Self): no yoked group, no
    table, no block, no file; one left by an earlier run is removed."""
    e = loaded("choice")
    assert e.is_optogenetic and an.yoked_groups(e) == []
    table = an.yoked_comparison(e)
    assert table.empty
    assert yr.yoked_table_blocks(table) == [] and yr.yoked_figure_blocks(table) == []
    spec, results = _simulated("choice")
    other = _load(_write(tmp_path / "exp", spec, results, _config([1])))
    stale = Path(other.analysis_dir) / an.YOKED_COMPARISON_FILE
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("old")
    result = other.execute_basic_analysis(skip_qc=True)
    assert "yoked_comparison" not in result and not stale.exists()


def _project(root: Path, members: dict):
    from pyflic.base.project import Project

    design = None
    for name, (spec_name, chambers) in members.items():
        spec, results = _simulated(spec_name)
        cfg = _config([d.dfm_id for d in spec.dfms], chambers)
        design = cfg.pop("global")
        _write(root / name, spec, results, cfg)
    (root / "project.yaml").write_text(yaml.safe_dump(
        {"name": "proj", "design": {"global": design}}, sort_keys=False))
    project = Project(root)
    for name in project.member_names:
        project.load_member(name, parallel=False,
                            use_disk_cache=False).execute_basic_analysis(skip_qc=True)
    return project


def test_the_project_stacks_the_yoked_comparison(tmp_path):
    project = _project(tmp_path / "proj", {"rep1": ("yoked", None),
                                           "rep2": ("two_yoked", None),
                                           "solo": ("choice", None)})
    written = {Path(p).name for p in project.build_combined_analysis()["written"]}
    assert "proj_YokedComparison.csv" in written
    stacked = pd.read_csv(Path(project.analysis_path) / "proj_YokedComparison.csv")
    assert stacked.columns[0] == "Experiment"
    assert stacked.groupby("Experiment").size().to_dict() == {"rep1": 3, "rep2": 2}
    frame = project.combined_light_frames()["YokedComparison"]
    blocks = yr.yoked_table_blocks(frame, mixed_p=project._mixed_p,
                                   mixed_p0=project._mixed_p0)
    assert _tables(blocks)


# ---------------------------------------------------------------------------
# Light behaviour by Role (decision 12)
# ---------------------------------------------------------------------------

def test_yoked_chambers_leave_the_activation_statistics_and_stay_in_lit_time(loaded):
    """The acceptance: behaviour_rows keeps every chamber, with its Role;
    the yoked chambers' Activation-based columns are missing, so the
    statistics of Activations compare the Paired chambers alone, while
    LitSec's take every chamber."""
    e = loaded("two_dfms")
    table = ls.light_summary(e)
    kept = ls.behaviour_rows(table)
    assert len(kept) == len(table) == 12 and kept["Role"].tolist() == table["Role"].tolist()
    yoked = kept["Role"] == "Yoked"
    assert yoked.sum() == 6
    columns = [c for c in ls.ACTIVATION_COLUMNS if c in kept.columns]
    assert kept.loc[yoked, columns].isna().all().all()
    assert kept["LitSec"].notna().all() and kept["LitFraction"].notna().all()
    ## The Paired rows are untouched.
    pd.testing.assert_frame_equal(kept[~yoked], table[~yoked])
    rows = {r["metric"]: r for r in an.treatment_comparisons(
        [("Whole recording", kept)], ["Activations", "LitSec"])}
    assert rows["Activations"]["n_a"] + rows["Activations"]["n_b"] == 6
    assert rows["LitSec"]["n_a"] + rows["LitSec"]["n_b"] == 12


def test_the_report_compares_the_paired_chambers_activations(loaded):
    from pyflic.base.pdf_report import build_experiment_report

    e = loaded("two_dfms")
    doc = build_experiment_report(e, metrics=())
    ## Light behaviour by family (stream S3): the design runs closed loop.
    table = next(b for b in doc.blocks if isinstance(b, rl.Table)
                 and b.caption == "Treatment comparisons: light behaviour, closed loop").frame
    n = dict(zip(table["Measure"], table["n"]))
    ## Ctrl has Paired chambers 1 and 3 of each DFM, Exp chamber 2; each has
    ## six chambers in all.
    assert n["Activations"] in ("4 / 2", "2 / 4")
    assert n["Lit time (s)"] == "6 / 6"


def test_the_other_roles_are_read_by_meaning():
    """Self and Mutual earn light, as Paired does; Open and Unlit do not;
    an unknown Role (no source) keeps its values."""
    frame = pd.DataFrame({"Role": ["Paired", "Self", "Mutual", "Yoked", "Open", "Unlit", ""],
                          "LitSec": np.arange(7.0), "Activations": np.arange(7.0),
                          "AcclimationComplete": [True] * 7})
    kept = ls.behaviour_rows(frame)
    assert kept["LitSec"].tolist() == list(range(7))
    assert kept["Activations"].isna().tolist() == [False, False, False, True, True, True, False]
    assert kept["AcclimationComplete"].tolist() == [True, True, True, None, None, None, True]


def test_progressive_ratio_keeps_its_own(tmp_path):
    """Progressive Ratio keeps its paired-minus-yoked table and its paired
    rows: no yoked group, no file, and behaviour_rows as before."""
    pr = _load(make_pr_experiment_dir(tmp_path / "pr"))
    assert an.yoked_groups(pr) == [] and an.yoked_comparison(pr).empty
    table = ls.light_summary(pr)
    pd.testing.assert_frame_equal(ls.behaviour_rows(table), table[table["Role"] == "paired"])
    result = pr.execute_basic_analysis(skip_qc=True)
    assert "yoked_comparison" not in result
    assert not (Path(pr.analysis_dir) / an.YOKED_COMPARISON_FILE).exists()
    assert "paired_yoked_diff" in result


# ---------------------------------------------------------------------------
# AcclimationActivations from the ledger's Phase
# ---------------------------------------------------------------------------

ACCLIMATION_SPECS = {
    **{f"family:{k}": (lambda k=k: vf.family_specs(MINUTES)[k]) for k in vf.FAMILY_SPEC_KEYS},
    "fixed_ratio": lambda: vf.fixed_ratio_spec(minutes=MINUTES),
    "one_of_each": vf.one_of_each_spec,
    "choice": vf.choice_pr_spec,
    "stress": vf.stress_spec,
}


@pytest.fixture(scope="module")
def acclimation_wells(tmp_path_factory):
    """``get(name) -> (light_wells, results)``."""
    cache: dict = {}

    def get(name: str):
        if name not in cache:
            spec = ACCLIMATION_SPECS[name]()
            results = vf.simulate(spec)
            root = _write(tmp_path_factory.mktemp(name.replace(":", "_")), spec, results,
                          _config([d.dfm_id for d in spec.dfms]))
            cache[name] = (ls.light_wells(_load(root)), results)
        return cache[name]

    return get


@pytest.mark.parametrize("name", list(ACCLIMATION_SPECS))
def test_acclimation_activations_are_the_truths(acclimation_wells, name):
    """Wherever Acclimation applies, Activations = AcclimationActivations +
    MainActivations, and AcclimationActivations is the truth's count of
    Phase-A Activations; AcclimationEndMin is the truth's first Acclimation
    end to within one record.  Counted from the rows' Phase, as before, it
    missed the truth on 8 of stress_spec's 20 such wells, and the sum on 7:
    an Acclimation Activation shorter than a record can start and end in a
    row that reads Main, the first row of a state run among them.  stress_spec's
    Circadian DFM 3 counts too: its Program.txt lists FI1, which wells 9 and 10
    run from minute 2, though the recording ends before FI1's 1438 minutes
    from midnight (PyMCU 4.0.1's export)."""
    wells, results = acclimation_wells(name)
    defined = 0
    for row in wells.itertuples(index=False):
        truth = results[row.DFM].truth
        phases = list(truth.activation_phase[row.Well])
        if pd.isna(row.AcclimationActivations):
            assert "A" not in phases, (row.DFM, row.Well)
            continue
        defined += 1
        assert row.Activations == row.AcclimationActivations + row.MainActivations, row
        assert row.AcclimationActivations == phases.count("A"), (row.DFM, row.Well)
        end = truth.acclimation_end_ms.get(row.Well)
        if end is None:
            assert pd.isna(row.AcclimationEndMin), (row.DFM, row.Well)
        else:
            assert abs(row.AcclimationEndMin * 60000.0 - end) <= 200 + 1e-6, (row.DFM, row.Well)
    assert (defined > 0) == (name not in ("family:closed_max", "family:non_feeding"))
