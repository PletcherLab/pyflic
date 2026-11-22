"""The QC Set, the Analysis Set and the Plot Set, the Selection, and the three
runs (ADR-0018, docs/hub-selection-plan.md, stream 1).

Every layout, type, light and Generation goes through the same three runs:

* single-well Custom, generation 1, optogenetic (``opto_fixtures``) and not;
* two-well Custom, generation 2 (``v4_fixtures``' older-firmware writer) and
  generation 3 with a lost-record gap (so a data break), and generation 3
  with a yoked linkage (so the yoked comparison);
* Progressive Ratio, generation 1 (``pr_fixtures``) and generation 3;
* Hedonic, generation 2, without light.

What is asserted is the contract: each item runs where it applies and writes
what its table says, a left-out item is not run and is said to be left out
(nothing is removed), an item that does not apply is reported and not run,
*Run QC* with nothing left out writes the file set ``write_qc_reports`` wrote
before the split, the Selection lives in the right file and is read back, and
Analyze all is unchanged.
"""

from __future__ import annotations

import contextlib
import io
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest
import yaml

from pyflic.base import run_selection as rs
from pyflic.base import selections as sel
from pyflic.base.experiment import QC_REPORT_ITEMS
from pyflic.base.project import DESIGN_KEYS, Project
from pyflic.base.yaml_config import load_experiment_yaml
from pyflic.base.yaml_lint import lint_flic_config

import v4_fixtures as vf
from opto_fixtures import make_opto_dir, opto_config, write_opto_csv
from pr_fixtures import make_pr_experiment_dir, pr_config

# ---------------------------------------------------------------------------
# Fixtures: one Experiment Directory per layout, type, light and Generation
# ---------------------------------------------------------------------------

_PARAMS = {"feeding_threshold": 20, "feeding_minimum": 10, "tasting_minimum": 5,
           "tasting_maximum": 20, "samples_per_second": 5, "pi_direction": "left"}


def _two_well_config(dfm_ids, **extra) -> dict:
    return {"global": {"chamber_layout": "two_well", "transform_licks": False,
                       "params": dict(_PARAMS), **extra},
            "dfms": [{"id": int(i), "chambers": {c: ("Ctrl" if c % 2 else "Exp")
                                                 for c in range(1, 7)}}
                     for i in dfm_ids]}


def _hedonic_config(dfm_ids) -> dict:
    cfg = _two_well_config(dfm_ids, optogenetics=False)
    g = cfg["global"]
    g.pop("chamber_layout")
    g["experiment_type"] = "Hedonic"
    g["well_names"] = {"A": "Sucrose", "B": "Yeast"}
    ## The simulated flies never feed at well B: no lick cutoff, or none stays.
    g["constants"] = {"min_untransformed_licks_cutoff": None}
    return cfg


def _pr_config(dfm_ids) -> dict:
    return pr_config([{"id": int(d), "paired_chambers": [1, 3, 5],
                       "chambers": {1: "Ctrl", 2: "Ctrl", 3: "Exp", 4: "Exp",
                                    5: "Third", 6: "Third"}} for d in dfm_ids])


@lru_cache(maxsize=None)
def _simulated(name: str):
    spec = {"short": lambda: vf.fixed_ratio_spec(minutes=6.0),
            "yoked": lambda: vf.yoked_controls_spec(minutes=12.0),
            "pr": lambda: vf.fixed_ratio_spec(minutes=12.0)}[name]()
    return spec, vf.simulate(spec)


def _v4_dir(root: Path, name: str, cfg: dict, *, writer=vf.write_v4_recording, **kw) -> Path:
    spec, results = _simulated(name)
    root.mkdir(parents=True, exist_ok=True)
    writer(root / "data", spec, results=results, **kw)
    (root / "flic_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return root


BUILDERS = {
    "single_opto_g1": lambda root: make_opto_dir(root),
    "single_plain_g1": lambda root: make_opto_dir(root, program=None, light_cases=False),
    "two_well_g2": lambda root: _v4_dir(root, "short", _two_well_config([1]),
                                        writer=vf.write_gen2_recording),
    "two_well_g3": lambda root: _v4_dir(root, "short", _two_well_config([1]),
                                        index_gaps={1: [(600, 20)]}),
    "yoked_g3": lambda root: _v4_dir(root, "yoked", _two_well_config([1])),
    "pr_g1": lambda root: make_pr_experiment_dir(root),
    "pr_g3": lambda root: _v4_dir(root, "pr", _pr_config([1])),
    "hedonic_g2": lambda root: _v4_dir(root, "short", _hedonic_config([1]),
                                       writer=vf.write_gen2_recording),
}
#: The fixtures whose Run analysis writes the experiment report (slow); the
#: others leave it out by the caller's omit.
WITH_REPORT = {"single_opto_g1", "pr_g1"}
#: The core Run analysis always writes, the feeding summary with the record of
#: the settings it was computed from (ADR-0022).
CORE = {"analysis/removed_chambers.csv", "analysis/summary.txt",
        "analysis/feeding_summary.csv", "analysis/analysis_settings.json",
        "analysis/feeding_summary_facet.csv"}


def _load(root: Path):
    with contextlib.redirect_stdout(io.StringIO()):
        return load_experiment_yaml(root, parallel=False, use_disk_cache=False)


def _quiet(call, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return call(*args, **kwargs)


@dataclass
class Ran:
    root: Path
    exp: object
    qc: rs.SelectionRun
    analysis: rs.SelectionRun
    plots: rs.SelectionRun
    log: list[str]

    @property
    def runs(self) -> tuple[rs.SelectionRun, ...]:
        return self.qc, self.analysis, self.plots

    def files(self) -> set[str]:
        return {p.relative_to(self.root).as_posix() for p in self.root.rglob("*")
                if p.is_file() and p.parts[len(self.root.parts)] in ("qc", "analysis")}


def _run_everything(root: Path, name: str) -> Ran:
    exp = _load(root)
    log: list[str] = []
    omit_report = None if name in WITH_REPORT else (
        sel.omitted(exp, sel.ANALYSES) | {"experiment_report"})
    qc = _quiet(rs.run_qc, exp, log=log.append)
    analysis = _quiet(rs.run_analysis, exp, omit=omit_report, log=log.append)
    plots = _quiet(rs.run_plots, exp, log=log.append, keep_figures=False)
    return Ran(root, exp, qc, analysis, plots, log)


@pytest.fixture(scope="module")
def ran(tmp_path_factory) -> dict[str, Ran]:
    base = tmp_path_factory.mktemp("selections")
    return {name: _run_everything(build(base / name), name)
            for name, build in BUILDERS.items()}


def _matches(item: sel.Item, rel: str) -> bool:
    return any(Path(rel).match(pattern) and len(Path(rel).parts) == len(Path(pattern).parts)
               for pattern in item.outputs)


# ---------------------------------------------------------------------------
# The sets
# ---------------------------------------------------------------------------

def test_the_baselined_caption_names_only_the_thresholds_the_plot_draws():
    """``DFM.plot_baselined`` draws FeedingMax and FeedingMin, no tasting
    threshold."""
    caption = sel.item(sel.QC, "baselined_signal").caption
    assert "feeding_threshold and feeding_minimum" in caption
    assert "tasting" not in caption


def test_the_three_sets_hold_the_plans_items_in_panel_order():
    assert [i.id for i in sel.QC_SET] == [
        "integrity", "data_breaks", "raw_signal", "baselined_signal", "cumulative_licks",
        "simultaneous_feeding", "bleeding", "opto_light_qc", "opto_light_figures",
        "program_timeline", "pr_light_qc", "pr_light_event_figures",
        "pr_resting_level_figures"]
    assert [i.id for i in sel.ANALYSIS_SET] == [
        "binned_summary", "event_statistics", "light_tables", "yoked_comparison",
        "paired_yoked_diff", "cumulative_diff", "breaking_point", "weighted_duration",
        "experiment_report"]
    assert [i.id for i in sel.PLOT_SET] == [
        "feeding_summary_plot", "well_comparison", "binned_time_course", "dot_plot",
        "pr_cumulative_diff_plot", "pr_cumulative_licks", "pr_still_responding",
        "pr_breaking_point_plots", "hedonic_plot"]
    assert sel.groups(sel.QC) == ["QC reports", "Optogenetics", "Progressive Ratio"]
    assert sel.groups(sel.ANALYSES) == ["Analyses", "Optogenetics", "Progressive Ratio",
                                        "Hedonic", "Report"]
    assert sel.groups(sel.PLOTS) == ["Figures", "Chosen metric", "Progressive Ratio",
                                     "Hedonic"]
    for kind in sel.OMIT_KINDS:
        assert all(i.kind == kind and i.outputs and i.caption for i in sel.items(kind))
        assert all(set(i.requirements) <= set(sel.REQUIREMENTS) for i in sel.items(kind))
    ## Every item has a run, and write_qc_reports writes the QC reports in
    ## the set's order.
    assert set(rs._RUNNERS) == {i.id for i in sel.all_items()}
    qc_ids = [i.id for i in sel.QC_SET]
    assert list(QC_REPORT_ITEMS) == [i for i in qc_ids if i in QC_REPORT_ITEMS]
    with pytest.raises(ValueError):
        sel.items("figures")


def test_every_item_runs_on_a_fixture_where_it_applies(ran):
    done = {(run.kind, i.id) for r in ran.values() for run in r.runs for i in run.ran}
    assert done == {(i.kind, i.id) for i in sel.all_items()}


def test_each_item_writes_what_its_table_says(ran):
    """An item that ran wrote files, and only files its outputs name; across
    the fixtures every output it names was written somewhere."""
    seen: dict[str, set[str]] = {}
    for name, r in ran.items():
        for run in r.runs:
            for item in run.ran:
                written = [p.relative_to(r.root).as_posix()
                           for p in run.item_outputs[item.id]]
                ## Data breaks: a folder and no file when there is no break.
                if item.id != "data_breaks":
                    assert written, f"{name}: {item.id} wrote nothing"
                for rel in written:
                    assert _matches(item, rel), f"{name}: {item.id} wrote {rel}"
                    assert (r.root / rel).is_file()
                seen.setdefault(item.id, set()).update(written)
    for item in sel.all_items():
        for pattern in item.outputs:
            assert any(Path(rel).match(pattern) for rel in seen[item.id]), \
                f"{item.id}: nothing ever matched {pattern}"
    assert seen["data_breaks"] == {"qc/data_breaks/DFM1_data_breaks.csv"}


def test_every_file_the_runs_write_belongs_to_the_core_or_to_one_item(ran):
    """So each item's outputs name all it writes, and no two items claim a file."""
    for name, r in ran.items():
        for rel in r.files():
            owners = [i.id for i in sel.all_items() if _matches(i, rel)]
            assert (rel in CORE) != bool(owners), f"{name}: {rel} owned by {owners}"
            assert len(owners) <= 1, f"{name}: {rel} owned by {owners}"


@pytest.mark.parametrize("name,generation,layout,optogenetic", [
    ("single_opto_g1", 1, "single_well", True), ("single_plain_g1", 1, "single_well", False),
    ("two_well_g2", 2, "two_well", True), ("two_well_g3", 3, "two_well", True),
    ("yoked_g3", 3, "two_well", True), ("pr_g1", 1, "two_well", True),
    ("pr_g3", 3, "two_well", True), ("hedonic_g2", 2, "two_well", False)])
def test_the_fixtures_cover_every_layout_type_light_and_generation(
        ran, name, generation, layout, optogenetic):
    exp = ran[name].exp
    assert {d.generation for d in exp.dfms.values()} == {generation}
    assert exp.chamber_layout == layout and exp.is_optogenetic is optogenetic


def test_run_analysis_writes_the_core_first_and_the_report_last(ran):
    r = ran["pr_g1"]
    written = [p.relative_to(r.root).as_posix() for p in r.analysis.written]
    assert set(written[:len(CORE)]) == CORE
    assert written[-1] == "analysis/experiment_report.pdf"
    assert r.analysis.ran[-1].id == "experiment_report"
    assert r.log.index(next(line for line in r.log if line.startswith("Core"))) \
        < r.log.index("[1/7] Binned summary")


def test_the_summary_states_what_was_left_out_and_what_did_not_apply(ran):
    text = (ran["two_well_g3"].root / "analysis" / "summary.txt").read_text(encoding="utf-8")
    block = text[text.index("Analysis Set (Run analysis)"):]
    assert "Left out: Experiment report (PDF) (experiment_report) — left out by this run" \
        in block
    assert ("Not applicable: Paired and yoked comparison (yoked_comparison) — No yoked "
            "group: no linkage group lights a Yoked chamber") in block
    assert "Not applicable: Weighted duration summary (weighted_duration) — needs an " \
           "experiment type with the Hedonic module" in block
    assert "Included: Binned summary (binned_summary), Event statistics" in block


def test_generate_plots_keeps_the_figures_for_tabs_only_when_asked(ran, tmp_path):
    assert all(run.figures == [] for r in ran.values() for run in r.runs)
    root = BUILDERS["single_plain_g1"](tmp_path / "plain")
    exp = _load(root)
    run = _quiet(rs.run_plots, exp, metric="Events", binsize_min=10.0, log=lambda m: None)
    assert [title for title, _fig in run.figures] == ["Feeding Summary", "Binned: Events",
                                                      "Dot: Events"]
    assert (root / "analysis" / "binned_Events.png").is_file()


# ---------------------------------------------------------------------------
# What does not apply
# ---------------------------------------------------------------------------

def _not_applicable(run: rs.SelectionRun) -> dict[str, str]:
    return {item.id: why for item, why in run.not_applicable}


def test_items_that_do_not_apply_are_reported_and_not_run(ran):
    r = ran["single_plain_g1"]
    qc = _not_applicable(r.qc)
    assert qc == {
        "simultaneous_feeding": "needs a two-well chamber layout",
        "bleeding": "needs a two-well chamber layout",
        "opto_light_qc": sel._NOT_MET[sel.OPTOGENETIC],
        "opto_light_figures": sel._NOT_MET[sel.OPTOGENETIC],
        "program_timeline": sel._NOT_MET[sel.OPTOGENETIC],
        "pr_light_qc": sel._NOT_MET[sel.RATIO],
        "pr_light_event_figures": sel._NOT_MET[sel.RATIO],
        "pr_resting_level_figures": sel._NOT_MET[sel.RATIO]}
    assert set(_not_applicable(r.analysis)) == {
        "light_tables", "yoked_comparison", "paired_yoked_diff", "cumulative_diff",
        "breaking_point", "weighted_duration"}
    assert set(_not_applicable(r.plots)) == {
        "well_comparison", "pr_cumulative_diff_plot", "pr_cumulative_licks",
        "pr_still_responding", "pr_breaking_point_plots", "hedonic_plot"}
    files = r.files()
    for run in r.runs:
        for item, _why in run.not_applicable:
            assert not any(_matches(item, rel) for rel in files), item.id
    assert not (r.root / "qc" / "opto").exists() and not (r.root / "qc" / "bleeding").exists()
    assert "  Not applicable — Bleeding between wells: needs a two-well chamber layout" \
        in r.log


def test_an_item_the_data_cannot_answer_is_reported_with_its_reason(ran):
    pr = ran["pr_g1"]
    assert _not_applicable(pr.qc) == {"program_timeline": sel._no_timeline(pr.exp)}
    assert not (pr.root / "qc" / "program").exists()
    assert sel.unavailable(sel.item(sel.QC, "program_timeline"), pr.exp).startswith(
        "No timeline")
    ## The yoked comparison: greyed without a yoked group, hidden under
    ## a type without the Paired and yoked module, such as Progressive Ratio,
    ## whose Ratio module keeps its own (ADR-0019).
    yoked = sel.item(sel.ANALYSES, "yoked_comparison")
    assert sel.unavailable(yoked, ran["single_opto_g1"].exp) == \
        "No yoked group: no linkage group lights a Yoked chamber"
    assert sel.unavailable(yoked, ran["yoked_g3"].exp) is None
    assert not sel.applies(yoked, pr.exp) and sel.unavailable(yoked, pr.exp) is None
    assert sel.not_applicable_reason(yoked, pr.exp) == sel._NOT_MET[sel.YOKED]
    ## Hedonic and Progressive Ratio items apply to their own type only.
    assert sel.applies(sel.item(sel.PLOTS, "hedonic_plot"), ran["hedonic_g2"].exp)
    assert not sel.applies(sel.item(sel.PLOTS, "hedonic_plot"), pr.exp)
    assert sel.applies(sel.item(sel.PLOTS, "well_comparison"), ran["hedonic_g2"].exp)


def test_the_hedonic_manifest_names_the_files_its_runs_write(ran):
    """The Hedonic module once named a weighted_duration.csv nothing writes;
    the weighted duration summary is treatment_means.csv."""
    from pyflic.base.analysis_modules.hedonic import HEDONIC

    hedonic = ran["hedonic_g2"]
    manifest = hedonic.exp.experiment_type.output_manifest()
    assert set(HEDONIC.outputs) == {"hedonic_feeding_plot.png", "treatment_means.csv"}
    assert set(HEDONIC.outputs) <= set(manifest)
    assert "weighted_duration.csv" not in manifest
    for name in HEDONIC.outputs:
        assert (hedonic.root / "analysis" / name).is_file(), name
    table = pd.read_csv(hedonic.root / "analysis" / "treatment_means.csv")
    assert list(table.columns) == [
        "Treatment",
        "MeanDurationA", "SemDurationA", "WeightedMeanDurationA", "WeightedSemDurationA",
        "SampleSizeA",
        "MeanDurationB", "SemDurationB", "WeightedMeanDurationB", "WeightedSemDurationB",
        "SampleSizeB"]
    ## The script step names the file it wrote, not the table.
    from pyflic.base.script_editor.runner import run_experiment_script

    lines: list[str] = []
    run_experiment_script(hedonic.exp, {"steps": [{"action": "weighted_duration"}]},
                          log=lines.append)
    assert f"Wrote: {Path(hedonic.exp.analysis_dir) / 'treatment_means.csv'}" in lines


def test_the_paired_yoked_difference_is_greyed_without_a_yoked_chamber(ran, monkeypatch):
    from pyflic.base.progressive_ratio_experiment import ProgressiveRatioExperiment

    item = sel.item(sel.ANALYSES, "paired_yoked_diff")
    exp = ran["pr_g3"].exp
    assert sel.unavailable(item, exp) is None
    monkeypatch.setattr(ProgressiveRatioExperiment, "has_yoked_controls", lambda self: False)
    assert sel.unavailable(item, exp) == "No yoked chamber: every chamber group is a solo group"


def test_an_omitted_item_that_does_not_apply_is_reported_as_not_applicable(tmp_path):
    root = BUILDERS["single_plain_g1"](tmp_path / "plain")
    exp = _load(root)
    run, todo = rs._plan(exp, sel.QC, {"bleeding", "raw_signal"})
    assert [i.id for i in run.left_out] == ["raw_signal"]
    assert "bleeding" in _not_applicable(run)
    assert "raw_signal" not in [i.id for i in todo]


# ---------------------------------------------------------------------------
# Run QC with nothing left out is write_qc_reports
# ---------------------------------------------------------------------------

#: ``qc/`` as write_qc_reports wrote it at 9582254 (before the split), on
#: make_opto_dir() and make_pr_experiment_dir().
QC_BEFORE = {
    "single_opto_g1": {
        "baselined/", "baselined/DFM1_baselined.png", "cumulative_licks/",
        "cumulative_licks/DFM1_cumulative_licks.png", "data_breaks/", "integrity/",
        "integrity/DFM1_integrity_report.csv", "integrity/DFM1_integrity_report.txt",
        "opto/", "opto/opto_light_dfm1.png", "opto/opto_light_events.csv",
        "opto/opto_light_intervals.csv", "opto/opto_light_qc.csv", "opto/opto_program.csv",
        "program/", "program/program_timeline_dfm1.png", "raw_signal/",
        "raw_signal/DFM1_raw.png"},
    "pr_g1": {
        "baselined/", "baselined/DFM1_baselined.png", "baselined/DFM2_baselined.png",
        "bleeding/", "bleeding/DFM1_bleeding_alldata.csv", "bleeding/DFM1_bleeding_matrix.csv",
        "bleeding/DFM2_bleeding_alldata.csv", "bleeding/DFM2_bleeding_matrix.csv",
        "cumulative_licks/", "cumulative_licks/DFM1_cumulative_licks.png",
        "cumulative_licks/DFM2_cumulative_licks.png", "data_breaks/", "integrity/",
        "integrity/DFM1_integrity_report.csv", "integrity/DFM1_integrity_report.txt",
        "integrity/DFM2_integrity_report.csv", "integrity/DFM2_integrity_report.txt",
        "opto/", "opto/opto_light_dfm1.png", "opto/opto_light_dfm2.png",
        "opto/opto_light_events.csv", "opto/opto_light_intervals.csv",
        "opto/opto_light_qc.csv", "opto/opto_program.csv", "raw_signal/",
        "raw_signal/DFM1_raw.png", "raw_signal/DFM2_raw.png", "simultaneous_feeding/",
        "simultaneous_feeding/DFM1_simultaneous_feeding_matrix.csv",
        "simultaneous_feeding/DFM2_simultaneous_feeding_matrix.csv"},
}


def _listing(directory: Path) -> set[str]:
    return {p.relative_to(directory).as_posix() + ("/" if p.is_dir() else "")
            for p in directory.rglob("*")}


def _text_of(path: Path, root: Path) -> str:
    """A CSV or text file with its own Experiment Directory taken out, so
    two copies of one recording compare equal."""
    return path.read_text(encoding="utf-8").replace(str(root), "<root>")


@pytest.mark.parametrize("name", ["single_opto_g1", "pr_g1"])
def test_run_qc_with_nothing_omitted_writes_what_write_qc_reports_wrote(tmp_path, name):
    before = BUILDERS[name](tmp_path / "write_qc_reports")
    after = BUILDERS[name](tmp_path / "run_qc")
    _quiet(_load(before).write_qc_reports)
    exp = _load(after)
    run = _quiet(rs.run_qc, exp, log=lambda m: None)
    assert run.left_out == []

    assert _listing(before / "qc") == QC_BEFORE[name]
    assert _listing(after / "qc") == QC_BEFORE[name]
    for rel in QC_BEFORE[name]:
        if rel.endswith((".csv", ".txt")):
            assert _text_of(before / "qc" / rel, before) == _text_of(after / "qc" / rel, after), rel
    assert exp.qc_report_dir == after / "qc"
    ## Beyond qc/, only the Progressive Ratio's light QC, which its action
    ## writes into analysis/.
    outside = {p.relative_to(after).as_posix() for p in run.written
               if p.parts[len(after.parts)] != "qc"}
    if name == "pr_g1":
        assert outside == {"analysis/pr_light_qc.csv", "analysis/pr_light_events.csv",
                           "analysis/pr_light_events_dfm1.png",
                           "analysis/pr_light_events_dfm2.png",
                           "analysis/pr_resting_level_dfm1.png",
                           "analysis/pr_resting_level_dfm2.png"}
        assert [i.id for i in run.ran][-3:] == [
            "pr_light_qc", "pr_light_event_figures", "pr_resting_level_figures"]
    else:
        assert outside == set() and not (after / "analysis").exists()


# ---------------------------------------------------------------------------
# Leaving an item out
# ---------------------------------------------------------------------------

#: What the omit tests leave out, by kind.
LEFT_OUT = ((sel.QC, "raw_signal"), (sel.QC, "data_breaks"),
            (sel.ANALYSES, "event_statistics"), (sel.ANALYSES, "experiment_report"),
            (sel.PLOTS, "dot_plot"))


def _run_with_selection(root: Path) -> tuple[list[rs.SelectionRun], list[str]]:
    exp = _load(root)
    for kind, item_id in LEFT_OUT:
        sel.set_included(exp, kind, item_id, False)
    exp = _load(root)                       # the Selection as read back from the file
    log: list[str] = []
    runs = [_quiet(rs.run_qc, exp, log=log.append),
            _quiet(rs.run_analysis, exp, log=log.append),
            _quiet(rs.run_plots, exp, log=log.append, keep_figures=False)]
    return runs, log


def test_an_omitted_item_is_left_out_and_says_so(tmp_path):
    root = BUILDERS["single_plain_g1"](tmp_path / "plain")
    (qc, analysis, plots), log = _run_with_selection(root)
    assert [i.id for i in qc.left_out] == ["data_breaks", "raw_signal"]
    assert [i.id for i in analysis.left_out] == ["event_statistics", "experiment_report"]
    assert [i.id for i in plots.left_out] == ["dot_plot"]
    for run in (qc, analysis, plots):
        assert not {i.id for i in run.left_out} & {i.id for i in run.ran}
    ## Nothing of theirs was written; the rest was.
    assert not (root / "qc" / "raw_signal").exists()
    assert not (root / "qc" / "data_breaks").exists()
    for name in ("tidy_feeding_events.csv", "experiment_report.pdf", "dot_Licks.png"):
        assert not (root / "analysis" / name).exists()
    assert (root / "qc" / "baselined" / "DFM1_baselined.png").is_file()
    assert (root / "analysis" / "binned_Licks.png").is_file()
    assert ("  Left out — Event statistics: unticked (flic_config.yaml "
            "global.omit.analyses)") in log
    assert "  Left out — Raw signal plots: unticked (flic_config.yaml global.omit.qc)" in log
    text = (root / "analysis" / "summary.txt").read_text(encoding="utf-8")
    assert ("Left out: Event statistics (event_statistics) — unticked (flic_config.yaml "
            "global.omit.analyses)") in text
    assert ("Left out: Experiment report (PDF) (experiment_report) — unticked "
            "(flic_config.yaml global.omit.analyses)") in text


def test_leaving_an_item_out_removes_nothing(tmp_path):
    """An earlier run's outputs of an item now left out stay as they were
    (removing them was considered and dropped)."""
    root = BUILDERS["single_plain_g1"](tmp_path / "plain")
    _run_everything(root, "single_plain_g1")
    (root / "analysis" / "experiment_report.pdf").write_bytes(b"%PDF an earlier report")
    earlier = {rel: (root / rel).stat().st_mtime_ns for rel in (
        "qc/raw_signal/DFM1_raw.png", "analysis/tidy_feeding_events.csv",
        "analysis/dot_Licks.png", "analysis/experiment_report.pdf")}
    _runs, log = _run_with_selection(root)
    for rel, mtime in earlier.items():
        assert (root / rel).stat().st_mtime_ns == mtime, rel
    assert (root / "qc" / "data_breaks").is_dir()
    assert not any("Removed" in line for line in log)
    assert not hasattr(rs.SelectionRun(), "removed")


# ---------------------------------------------------------------------------
# Where the Selection lives
# ---------------------------------------------------------------------------

def test_omitted_reads_an_experiment_a_config_or_a_global_block():
    g = {"omit": {"qc": ["bleeding"], "analyses": "event_statistics", "plots": None}}
    for source in (g, {"global": g, "dfms": []}):
        assert sel.omitted(source, sel.QC) == {"bleeding"}
        assert sel.omitted(source, sel.ANALYSES) == {"event_statistics"}
        assert sel.omitted(source, sel.PLOTS) == frozenset()
    assert sel.omitted({"global": {}}, sel.QC) == frozenset()
    assert sel.omitted({"omit": "bleeding"}, sel.QC) == frozenset()
    with pytest.raises(ValueError):
        sel.omitted(g, "figures")


def test_set_included_writes_a_standalone_experiments_config(tmp_path):
    root = BUILDERS["single_plain_g1"](tmp_path / "solo")
    exp = _load(root)
    path = sel.set_included(exp, sel.QC, "bleeding", False)
    assert path == root / "flic_config.yaml"
    sel.set_included(exp, sel.QC, "integrity", False)
    sel.set_included(exp, sel.PLOTS, "dot_plot", False)
    written = yaml.safe_load(path.read_text())
    ## Panel order, whatever order they were unticked in.
    assert written["global"]["omit"] == {"qc": ["integrity", "bleeding"], "plots": ["dot_plot"]}
    assert written["dfms"] and written["global"]["params"]
    ## The loaded experiment honours it at once, and a reload reads it.
    assert sel.omitted(exp, sel.QC) == {"integrity", "bleeding"}
    assert sel.omitted(_load(root), sel.PLOTS) == {"dot_plot"}

    for kind, item_id in ((sel.QC, "integrity"), (sel.QC, "bleeding"),
                          (sel.PLOTS, "dot_plot")):
        sel.set_included(exp, kind, item_id, True)
    assert "omit" not in yaml.safe_load(path.read_text())["global"]
    assert sel.omitted(exp, sel.QC) == frozenset()
    with pytest.raises(KeyError):
        sel.set_included(exp, sel.QC, "bleding", False)


def _project(root: Path, members=("rep1", "rep2"), *, design_extra=None) -> Path:
    """A Project of single-well, light-free members (generation 1)."""
    g = dict(opto_config()["global"])
    g.update(design_extra or {})
    root.mkdir(parents=True)
    (root / "project.yaml").write_text(yaml.safe_dump(
        {"name": "P", "design": {"global": g}, "scripts": []}, sort_keys=False))
    for seed, name in enumerate(members, start=3):
        write_opto_csv(root / name / "data", seed=seed, light_cases=False)
        (root / name / "flic_config.yaml").write_text(yaml.safe_dump(
            {"dfms": [opto_config()["dfms"][0]]}, sort_keys=False))
    return root


def test_set_included_writes_the_project_design_for_a_member(tmp_path):
    root = _project(tmp_path / "P")
    project = Project(root)
    exp = _quiet(project.load_member, "rep1")
    member_config = (root / "rep1" / "flic_config.yaml").read_text()

    path = sel.set_included(exp, sel.ANALYSES, "event_statistics", False, project=project)
    assert path == root / "project.yaml"
    meta = yaml.safe_load(path.read_text())
    assert meta["design"]["global"]["omit"] == {"analyses": ["event_statistics"]}
    assert meta["name"] == "P" and meta["design"]["global"]["params"]
    assert (root / "rep1" / "flic_config.yaml").read_text() == member_config
    assert sel.selection_where(exp, sel.ANALYSES) == "project.yaml design.global.omit.analyses"
    ## Every Member reads it, from the file and from the Project in memory.
    assert project.design_global["omit"] == {"analyses": ["event_statistics"]}
    for name in ("rep1", "rep2"):
        assert sel.omitted(_quiet(Project(root).load_member, name), sel.ANALYSES) == \
            {"event_statistics"}
        assert sel.omitted(_quiet(project.load_member, name), sel.ANALYSES) == \
            {"event_statistics"}

    sel.set_included(exp, sel.ANALYSES, "event_statistics", True, project=project)
    assert "omit" not in yaml.safe_load(path.read_text())["design"]["global"]
    assert sel.omitted(exp, sel.ANALYSES) == frozenset()


def test_a_member_stating_its_own_omit_inherits_the_new_one(tmp_path):
    root = _project(tmp_path / "P", design_extra={"omit": {"qc": ["raw_signal"]}})
    cfg_path = root / "rep2" / "flic_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["global"] = {"omit": {"qc": ["raw_signal"]}}         # agrees with the design
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    project = Project(root)
    exp = _quiet(project.load_member, "rep1")
    sel.set_included(exp, sel.QC, "data_breaks", False, project=project)
    assert "global" not in yaml.safe_load(cfg_path.read_text())
    ## So the Project still loads, and rep2 reads the design's new Selection.
    assert sel.omitted(_quiet(Project(root).load_member, "rep2"), sel.QC) == \
        {"raw_signal", "data_breaks"}


def test_a_member_of_a_project_without_a_design_has_nowhere_to_hold_it(tmp_path):
    root = BUILDERS["single_plain_g1"](tmp_path / "Legacy" / "rep1")
    (tmp_path / "Legacy" / "project.yaml").write_text("name: Legacy\n")
    exp = _load(root)
    with pytest.raises(ValueError, match="no design"):
        sel.set_included(exp, sel.QC, "bleeding", False)
    assert sel.selection_where(exp, sel.QC) == "global.omit.qc"


def test_omit_is_a_design_key():
    assert "omit" in DESIGN_KEYS


# ---------------------------------------------------------------------------
# pyflic lint
# ---------------------------------------------------------------------------

def test_validate_omit_names_what_no_run_produces():
    assert sel.validate_omit({"global": {}}) == []
    assert sel.validate_omit({"omit": {"qc": ["bleeding"], "plots": "dot_plot"}}) == []
    problems = sel.validate_omit({"omit": {"qc": ["bleding"], "figures": ["x"],
                                           "plots": {"a": 1}}})
    assert problems[0].startswith("global.omit.qc names 'bleding', which no Run QC produces")
    assert problems[1].startswith("global.omit.figures is not something a run can leave out")
    assert problems[2] == "global.omit.plots must be a list of item ids"
    assert sel.validate_omit({"omit": ["bleeding"]})[0].startswith("global.omit must be a mapping")


def test_lint_warns_on_an_unknown_id(tmp_path):
    root = BUILDERS["single_plain_g1"](tmp_path / "solo")
    cfg = yaml.safe_load((root / "flic_config.yaml").read_text())
    cfg["global"]["omit"] = {"analyses": ["binned_summary", "event_stats"]}
    (root / "flic_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    issues = lint_flic_config(root / "flic_config.yaml")
    messages = [i.message for i in issues if i.severity == "warning"]
    assert not any("unknown global key 'omit'" in m for m in messages)
    assert [m for m in messages if "omit" in m] == [
        "global.omit.analyses names 'event_stats', which no Run analysis produces (known: "
        + ", ".join(i.id for i in sel.ANALYSIS_SET) + ")"]
    assert not any(i.severity == "error" for i in issues if "omit" in i.message)


def test_lint_warns_on_an_unknown_id_in_the_projects_design(tmp_path):
    root = _project(tmp_path / "P", design_extra={"omit": {"plots": ["dot"]}})
    messages = [i.message for i in lint_flic_config(root / "rep1" / "flic_config.yaml")]
    assert any(m.startswith("project.yaml design: global.omit.plots names 'dot'")
               for m in messages)


# ---------------------------------------------------------------------------
# Analyze all is unchanged; the script actions
# ---------------------------------------------------------------------------

def test_analyze_all_keeps_basic_analysis_and_does_not_read_the_selection(
        tmp_path, monkeypatch):
    """Analyze all and a Batch Run's run_all_analyses running the three runs
    was considered and dropped: each member still gets Basic analysis."""
    from pyflic.base.experiment import Experiment

    root = _project(tmp_path / "P", design_extra={"omit": {"analyses": ["binned_summary"]}})
    basic: list[dict] = []
    monkeypatch.setattr(Experiment, "execute_basic_analysis",
                        lambda self, **kw: basic.append(kw) or {})
    for name in ("run_qc", "run_analysis", "run_plots"):
        monkeypatch.setattr(rs, name, lambda *a, _n=name, **k: pytest.fail(f"{_n} ran"))
    assert _quiet(Project(root).run_all, make_reports=False, log=lambda m: None) == []
    assert basic == [{}, {}]


#: The script actions before this stream; every one stays.
EXISTING_ACTIONS = {
    "load", "write_summary", "remove_chambers", "basic_analysis", "run_qc", "feeding_csv",
    "facet_csv", "binned_csv", "weighted_duration", "tidy_export", "bootstrap", "compare",
    "light_phase_summary", "param_sensitivity", "transition_matrix", "pdf_report",
    "plot_feeding_summary", "plot_binned", "plot_dot", "plot_moving_window",
    "plot_moving_median_chambers", "plot_moving_median_treatment", "plot_well_comparison",
    "plot_hedonic", "paired_yoked_diff", "breaking_point", "pr_light_qc", "opto_light_qc",
    "plot_opto_light", "plot_pr_cumulative_diff", "plot_pr_cumulative_licks",
    "plot_pr_light_events", "plot_pr_resting_level", "plot_breaking_point",
    "plot_pr_still_responding", "plot_program_timeline", "light_summary"}


def test_the_three_actions_are_in_the_catalogue():
    from pyflic.base.script_editor.actions import ACTIONS, get_action

    names = {a.action for a in ACTIONS}
    assert EXISTING_ACTIONS | {"run_analysis", "run_plots"} == names
    ## A run names its Slice, or covers the whole recording and every Slice
    ## the Selection keeps (ADR-0020).
    assert [p.key for p in get_action("run_analysis").params] == ["binsize", "slice"]
    assert [p.key for p in get_action("run_plots").params] == ["metric", "binsize", "slice"]
    assert [p.key for p in get_action("run_qc").params] == ["slice"]
    assert get_action("run_plots").produces == "figure"


def test_the_three_script_actions_run_through_run_experiment_script(tmp_path):
    from pyflic.base.script_editor.runner import ScriptContext, run_experiment_script

    root = BUILDERS["single_plain_g1"](tmp_path / "plain")
    exp = _load(root)
    sel.set_included(exp, sel.ANALYSES, "experiment_report", False)
    lines: list[str] = []
    script = {"name": "t", "steps": [
        {"action": "run_qc"}, {"action": "run_analysis", "binsize": 10},
        {"action": "run_plots", "metric": "Events"}]}
    figures = _quiet(run_experiment_script, exp, script,
                     context=ScriptContext(binsize=20.0), log=lines.append)
    assert [title for title, _fig in figures] == ["Feeding Summary", "Binned: Events",
                                                  "Dot: Events"]
    assert (root / "qc" / "raw_signal" / "DFM1_raw.png").is_file()
    assert f"QC reports → {root / 'qc'}" in lines
    ## The step's bin for the binned summary; the context's for the plot.
    binned = pd.read_csv(root / "analysis" / "binned_feeding_summary.csv")
    assert binned["Interval"].iloc[0] == "(0,10]"
    assert (root / "analysis" / "binned_Events.png").is_file()
    ## The Selection holds for a script as for the Hub.
    assert "  Left out — Experiment report (PDF): unticked (flic_config.yaml " \
           "global.omit.analyses)" in lines
    assert not (root / "analysis" / "experiment_report.pdf").exists()


# ---------------------------------------------------------------------------
# The editors keep the Selection
# ---------------------------------------------------------------------------

_QT: list = []


def _qt_app():
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    ## Held, or it is collected and the first widget brings the process down.
    if not _QT:
        _QT.append(QApplication.instance() or QApplication([]))
    return _QT[0]


def test_the_design_editor_keeps_the_selection(tmp_path):
    _qt_app()
    from pyflic.base.design_editor import ProjectDesignDialog

    omit = {"qc": ["raw_signal"], "plots": ["dot_plot"]}
    root = _project(tmp_path / "P", design_extra={"omit": omit})
    dialog = ProjectDesignDialog(start_dir=str(root))
    assert dialog._build_design()["global"]["omit"] == omit


def test_the_config_editor_keeps_a_standalone_selection(tmp_path):
    _qt_app()
    from pyflic.base.config_editor import FLICConfigEditor

    root = BUILDERS["single_plain_g1"](tmp_path / "solo")
    exp = _load(root)
    sel.set_included(exp, sel.QC, "bleeding", False)
    editor = FLICConfigEditor(root / "flic_config.yaml")
    assert editor._collect_yaml()["global"]["omit"] == {"qc": ["bleeding"]}


def test_the_breaking_point_item_writes_the_simulated_lights_when_the_design_asks():
    """Basic analysis writes pr_simulated_yoked_lights.csv when
    pr_simulated_yoked_bp is on (ADR-0023); the Hub's Breaking point item and
    a script's breaking_point step wrote pr_breaking_point.csv alone."""
    from types import SimpleNamespace

    written: list[str] = []

    def exp(enabled: bool):
        return SimpleNamespace(
            write_breaking_point=lambda: written.append("bp") or Path("pr_breaking_point.csv"),
            simulated_yoked_enabled=lambda: enabled,
            write_simulated_yoked_lights=lambda: (written.append("sim")
                                                  or Path("pr_simulated_yoked_lights.csv")),
            breaking_point_lines=lambda: [])

    on = rs._breaking_point(exp(True), rs._Context(None, lambda _m: None))
    assert [p.name for p in on] == ["pr_breaking_point.csv", "pr_simulated_yoked_lights.csv"]
    written.clear()
    off = rs._breaking_point(exp(False), rs._Context(None, lambda _m: None))
    assert [p.name for p in off] == ["pr_breaking_point.csv"] and written == ["bp"]
