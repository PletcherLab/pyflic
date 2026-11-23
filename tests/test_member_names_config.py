"""A Member that reads its recording's own config (ADR-0031).

A recording keeps one ``flic_config.yaml``, in its own folder, however many
Projects analyse it.  A Member of such a Project holds ``flic_member.yaml``
instead of a config: ``config:`` names the recording's config and, optionally,
``data_folder:`` names its data, both resolved against the Member folder.
Without ``data_folder:`` the data are the ``data/`` beside the named config.

pyflic reads the named config and never writes beside it: everything it writes
for the Member stays in the Member folder or the Project folder.  Inside a
Project the analysis settings are the design's alone: the recording's config
describes the recording (type, layout, wells, factors), which must agree with
the design, and any settings it states are not used, with a warning, so two
Projects may analyse one recording with different settings.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from pyflic.base import batch as batch_mod
from pyflic.base import cache, layout
from pyflic.base import project as prj
from pyflic.base.project import Project
from pyflic.base.project_report import report_path
from test_referenced_data import (
    ANALYSIS,
    ASSAY,
    IS_ROOT,
    PIPELINE_PARAMS,
    RUN1,
    RUN2,
    changed,
    copy_recording,
    load,
    lock,
    managed_root,
    pipeline_dfms,
    pipeline_global,
    quiet,
    run_dir,
    simple_dfms,
    simple_global,
    snapshot,
    summary_tables,
    unlock,
    write_config,
    write_simple,
    write_v4,
)

MEMBER = layout.MEMBER_FILENAME
#: The analysis settings a recording's config states and a Project's design
#: overrides: a feeding threshold the design does not use.
RUN_PARAMS = {**PIPELINE_PARAMS, "feeding_threshold": 99}


def config_value(run) -> str:
    """``config:`` as labdata writes it: relative, forward slashes."""
    return f"../../../../../runs/{ASSAY}/2026/{run}/{prj.CONFIG_FILENAME}"


def data_value(run) -> str:
    return f"../../../../../runs/{ASSAY}/2026/{run}/data"


def write_member(member: Path, pointer: dict) -> Path:
    member = Path(member)
    member.mkdir(parents=True, exist_ok=True)
    path = member / MEMBER
    path.write_text(yaml.safe_dump(pointer, sort_keys=False), encoding="utf-8")
    return path


def make_run(managed, run, seed, *, params=RUN_PARAMS) -> Path:
    """A run folder: its recording in ``data/`` and its one config beside it,
    which states analysis settings of its own."""
    folder = run_dir(managed, run)
    write_v4(folder / "data", seed=seed)
    write_config(folder, {"about": {"title": f"Recording {run}"},
                          "global": pipeline_global(params=dict(params)),
                          "dfms": pipeline_dfms()})
    return folder


def make_analysis(managed, name, runs, *, design_global=None, with_data=True) -> Path:
    analysis = Path(managed) / "studies" / "test-study" / "analyses" / name
    analysis.mkdir(parents=True)
    prj.create_project_file(str(analysis),
                            design={"global": design_global or pipeline_global()})
    meta = yaml.safe_load((analysis / prj.PROJECT_FILENAME).read_text(encoding="utf-8"))
    meta["scripts"][0]["steps"][0]["reports"] = True
    (analysis / prj.PROJECT_FILENAME).write_text(
        yaml.safe_dump(meta, sort_keys=False, allow_unicode=True), encoding="utf-8")
    for run in runs:
        pointer = {"config": config_value(run)}
        if with_data:
            pointer["data_folder"] = data_value(run)
        write_member(analysis / run, pointer)
    return analysis


# ---------------------------------------------------------------------------
# The whole pipeline over a read-only runs/ tree
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("named-config")
    managed = managed_root(tmp)
    make_run(managed, RUN1, seed=7)
    make_run(managed, RUN2, seed=11)
    analysis = make_analysis(managed, ANALYSIS, [RUN1, RUN2])
    (analysis.parent / "remove_chambers.csv").write_text(
        f"project,member,dfm,chamber,reason\n{ANALYSIS},{RUN1},1,1,test\n", encoding="utf-8")
    runs = managed / "runs"
    runs_before = snapshot(runs)
    tree_before = snapshot(tmp)
    locked = lock(runs)
    try:
        logs: list = []
        results = quiet(batch_mod.run_batch, str(analysis.parent), log=logs.append)
        project = Project(analysis)
        yield SimpleNamespace(
            tmp=tmp, managed=managed, runs=runs, analysis=analysis, results=results,
            logs=logs, project=project, layouts=layout.members_in(analysis),
            runs_before=runs_before, runs_after=snapshot(runs),
            tree_before=tree_before, tree_after=snapshot(tmp), locked=locked)
    finally:
        unlock(runs)


def test_the_pipeline_runs_through_the_named_config(pipeline):
    assert pipeline.results == {ANALYSIS: "ok"}, "\n".join(pipeline.logs)
    project = pipeline.project
    assert project.member_names == [RUN1, RUN2]
    for name in (RUN1, RUN2):
        member = pipeline.analysis / name
        assert (member / "analysis" / "feeding_summary.csv").is_file()
        assert os.path.isfile(project.member_report_path(name))
        assert sorted(p.name for p in member.iterdir() if p.is_file()) == sorted(
            [MEMBER, "remove_chambers.csv"] if name == RUN1 else [MEMBER])
    assert os.path.isfile(report_path(project))
    assert "test" in (pipeline.analysis / RUN1 / "remove_chambers.csv").read_text(encoding="utf-8")


def test_nothing_is_written_beside_the_recording_or_its_config(pipeline):
    assert pipeline.runs_after == pipeline.runs_before
    inside = os.path.relpath(pipeline.analysis, pipeline.tmp)
    new = changed(pipeline.tree_before, pipeline.tree_after)
    assert new and all(path == inside + "/" or path.startswith(inside + os.sep) for path in new)


@pytest.mark.skipif(IS_ROOT, reason="permissions do not stop root")
def test_the_runs_were_read_only_throughout(pipeline):
    assert pipeline.locked


def test_the_classifier_sees_two_healthy_members_reading_their_runs(pipeline):
    assert [(m.name, m.status) for m in pipeline.layouts] == [(RUN1, layout.OK), (RUN2, layout.OK)]
    for name, run in ((RUN1, RUN1), (RUN2, RUN2)):
        found = layout.config_location(pipeline.analysis / name)
        assert found.referenced and not found.problem
        assert Path(found.path).resolve() == (run_dir(pipeline.managed, run)
                                              / prj.CONFIG_FILENAME).resolve()
        where = layout.data_location(pipeline.analysis / name)
        assert where.referenced and Path(where.path).resolve() == (
            run_dir(pipeline.managed, run) / "data").resolve()


def test_the_designs_settings_are_used_and_the_runs_are_named_in_a_warning(pipeline, tmp_path):
    """The run's config states feeding_threshold 99; the design's 20 is what the
    analysis uses, exactly as a Member holding a copy of the recording."""
    plain = tmp_path / "plain"
    prj.create_project_file(str(plain), design={"global": pipeline_global()})
    copy_recording(run_dir(pipeline.managed, RUN1) / "data", plain / RUN1 / "data")
    write_config(plain / RUN1, {"dfms": pipeline_dfms()})
    shutil.copyfile(pipeline.analysis / RUN1 / "remove_chambers.csv",
                    plain / RUN1 / "remove_chambers.csv")
    assert quiet(Project(plain).run_all, make_reports=False) == []
    named, copied = summary_tables(pipeline.analysis / RUN1), summary_tables(plain / RUN1)
    assert set(named) == set(copied) and "analysis/feeding_summary.csv" in named
    for name, frame in named.items():
        pd.testing.assert_frame_equal(frame, copied[name], obj=name)
    assert pipeline.project.resolved_global(RUN1)["params"] == PIPELINE_PARAMS
    (warning,) = [w for w in pipeline.project.warnings if RUN1 in w and "params" in w]
    assert "not used" in warning


def test_two_projects_analyse_one_run_with_their_own_settings(pipeline):
    second = make_analysis(pipeline.managed, "A-20261105-test-02", [RUN1],
                           design_global=pipeline_global(
                               params={**PIPELINE_PARAMS, "feeding_threshold": 40}))
    project = Project(second)
    assert quiet(project.run_all, make_reports=False) == []
    own = pd.read_csv(second / RUN1 / "analysis" / "feeding_summary.csv")
    theirs = pd.read_csv(pipeline.analysis / RUN1 / "analysis" / "feeding_summary.csv")
    assert not own["LicksA"].equals(theirs["LicksA"])
    assert snapshot(pipeline.runs) == pipeline.runs_before


# ---------------------------------------------------------------------------
# Where the config and the data are, and what blocks a Member
# ---------------------------------------------------------------------------

def _recording(tmp_path, *, config=None) -> Path:
    run = tmp_path / "runs" / "R1"
    write_simple(run / "data")
    write_config(run, config or {"global": simple_global(), "dfms": simple_dfms()})
    return run


def test_without_a_member_file_the_config_is_the_members_own(tmp_path):
    write_config(tmp_path / "m", {"dfms": simple_dfms()})
    found = layout.config_location(tmp_path / "m")
    assert not found.referenced and found.path == str(tmp_path / "m" / prj.CONFIG_FILENAME)


def test_without_data_folder_the_data_are_beside_the_named_config(tmp_path):
    run = _recording(tmp_path)
    write_member(tmp_path / "p" / "m", {"config": "../../runs/R1/flic_config.yaml"})
    where = layout.data_location(tmp_path / "p" / "m")
    assert not where.problem and Path(where.path) == (run / "data").resolve()
    assert where.via == "../../runs/R1/flic_config.yaml" and where.written is None
    assert layout.classify(tmp_path / "p" / "m").status == layout.OK
    exp = load(tmp_path / "p" / "m")
    assert sorted(exp.dfms) == [1]


def test_the_member_files_data_folder_names_the_data_and_wins(tmp_path):
    """The lab's catalogue writes both keys: the data may be a clock-repaired
    copy, or kept in place elsewhere, not the data/ beside the config."""
    run = _recording(tmp_path)
    elsewhere = write_simple(tmp_path / "repaired", seed=3)
    write_member(tmp_path / "p" / "m", {"config": "../../runs/R1/flic_config.yaml",
                                        "data_folder": "../../repaired"})
    where = layout.data_location(tmp_path / "p" / "m")
    assert not where.problem and Path(where.path) == elsewhere.resolve()
    assert where.written == "../../repaired" and where.via is not None
    assert Path(where.path) != (run / "data").resolve()


def test_the_named_configs_own_data_folder_is_read_from_its_folder(tmp_path):
    run = tmp_path / "runs" / "R1"
    write_simple(run / "raw")
    write_config(run, {"data_folder": "raw", "global": simple_global(), "dfms": simple_dfms()})
    write_member(tmp_path / "p" / "m", {"config": "../../runs/R1/flic_config.yaml"})
    where = layout.data_location(tmp_path / "p" / "m")
    assert not where.problem and Path(where.path) == (run / "raw").resolve()


@pytest.mark.parametrize("pointer, kind, words", [
    ({"config": "../../runs/R9/flic_config.yaml"}, "missing", "does not exist"),
    ({"config": "../../runs/R1"}, "folder", "is a folder, not a file"),
    ({"config": ""}, "blank", "names no config"),
    ({"config": 3}, "number", "names no config"),
    ({}, "none", "names no config"),
])
def test_a_config_that_cannot_be_read_blocks_the_member_and_the_loader_names_it(
        tmp_path, pointer, kind, words):
    _recording(tmp_path)
    write_member(tmp_path / "p" / "m", pointer)
    item = layout.classify(tmp_path / "p" / "m")
    assert item.blocked and words in item.detail, (kind, item)
    with pytest.raises((layout.DataLocationError, ValueError), match=words):
        load(tmp_path / "p" / "m")


def test_a_member_holding_its_own_config_too_is_a_conflict(tmp_path):
    _recording(tmp_path)
    write_member(tmp_path / "p" / "m", {"config": "../../runs/R1/flic_config.yaml"})
    write_config(tmp_path / "p" / "m", {"dfms": simple_dfms()})
    item = layout.classify(tmp_path / "p" / "m")
    assert item.status == layout.AMBIGUOUS and "which is the experiment's config" in item.detail


def test_a_member_inside_the_recordings_folder_is_refused(tmp_path):
    run = _recording(tmp_path)
    write_member(run / "member", {"config": "../flic_config.yaml"})
    item = layout.classify(run / "member")
    assert item.blocked and "nothing is ever written beside a recording's config" in item.detail


def test_a_recording_of_the_members_own_beside_the_member_file_is_a_conflict(tmp_path):
    _recording(tmp_path)
    write_member(tmp_path / "p" / "m", {"config": "../../runs/R1/flic_config.yaml"})
    write_simple(tmp_path / "p" / "m" / "data")
    item = layout.classify(tmp_path / "p" / "m")
    assert item.status == layout.AMBIGUOUS and "which is the experiment?" in item.detail


def test_filing_a_member_that_names_its_config_is_refused(tmp_path):
    _recording(tmp_path)
    write_member(tmp_path / "p" / "m", {"config": "../../runs/R1/flic_config.yaml"})
    write_simple(tmp_path / "p" / "m")                         # loose, as the MCU wrote it
    plan = layout.plan_filing(tmp_path / "p" / "m")
    assert not plan.possible and "never filed" in plan.refused


def test_two_members_naming_one_config_are_one_member(tmp_path):
    _recording(tmp_path)
    root = tmp_path / "p"
    prj.create_project_file(str(root), design={"global": simple_global()})
    for name in ("a", "b"):
        write_member(root / name, {"config": "../../runs/R1/flic_config.yaml"})
    project = Project(root)
    assert project.member_names == ["a"] and project.duplicates == {"b": "a"}


# ---------------------------------------------------------------------------
# The design and the recording's config
# ---------------------------------------------------------------------------

def _project_reading(tmp_path, run_global: dict, design_global=None) -> Path:
    _recording(tmp_path, config={"global": run_global, "dfms": simple_dfms()})
    root = tmp_path / "p"
    prj.create_project_file(str(root), design={"global": design_global or simple_global()})
    write_member(root / "m", {"config": "../../runs/R1/flic_config.yaml"})
    return root


def test_a_recording_described_otherwise_than_the_design_does_not_load(tmp_path):
    root = _project_reading(tmp_path, simple_global(well_names={"A": "Sucrose", "B": "Water"}),
                            simple_global(well_names={"A": "S5", "B": "S5Y5"}))
    with pytest.raises(ValueError, match="well_names"):
        Project(root)


def test_settings_the_recordings_config_states_are_not_used_and_never_removed(tmp_path):
    run_global = simple_global(params={**simple_global()["params"], "feeding_threshold": 99},
                               constants={"min_untransformed_licks_cutoff": 3})
    root = _project_reading(tmp_path, run_global)
    config = tmp_path / "runs" / "R1" / prj.CONFIG_FILENAME
    before = config.read_bytes()
    project = Project(root)
    assert project.resolved_global("m")["params"] == simple_global()["params"]
    assert "constants" not in project.resolved_global("m")
    (warning,) = [w for w in project.warnings if w.startswith("m:")]
    assert "params" in warning and "constants" in warning and "not used" in warning
    states = prj.member_design_states(root)
    assert [(s.name, s.state) for s in states] == [("m", "in_step")]
    assert prj.adopt_design(root) == []
    assert config.read_bytes() == before
    exp = project.load_member("m", parallel=False, use_disk_cache=False, eager=False)
    assert exp.dfms[1].params.feeding_threshold == simple_global()["params"]["feeding_threshold"]


def test_a_dfm_setting_other_than_the_physical_ones_is_not_used_either(tmp_path):
    root = _project_reading(tmp_path, simple_global())
    config = tmp_path / "runs" / "R1" / prj.CONFIG_FILENAME
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    raw["dfms"][0]["params"] = {"pi_direction": "right", "feeding_threshold": 77}
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    project = Project(root)                                   # loads: not an error here
    exp = quiet(project.load_member, "m", parallel=False, use_disk_cache=False, eager=False)
    params = exp.dfms[1].params
    assert params.pi_direction == "right"                     # how the DFM was set up
    assert params.feeding_threshold == simple_global()["params"]["feeding_threshold"]


def test_a_first_design_is_inferred_from_the_recordings_whole_config(tmp_path):
    """Initialize existing directory…: a folder of Members naming their
    recordings' configs becomes a Project whose first design is the first
    recording's global:, settings included, for the person to check."""
    run_global = simple_global(well_names={"A": "S5", "B": "S5Y5"},
                               experimental_design_factors={"Genotype": ["Ctrl", "Exp"]})
    _recording(tmp_path, config={"global": run_global, "dfms": simple_dfms()})
    folder = tmp_path / "analysis"
    write_member(folder / "m", {"config": "../../runs/R1/flic_config.yaml"})
    design = prj.design_from_members(folder)
    assert design["well_names"] == {"A": "S5", "B": "S5Y5"}
    assert design["params"] == simple_global()["params"]
    assert design["experimental_design_factors"] == {"Genotype": ["Ctrl", "Exp"]}


def test_the_cache_key_follows_the_named_config_but_not_its_about(tmp_path):
    root = _project_reading(tmp_path, simple_global())
    config = tmp_path / "runs" / "R1" / prj.CONFIG_FILENAME
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    config.write_text(yaml.safe_dump({"about": {"title": "First"}, **raw}, sort_keys=False),
                      encoding="utf-8")
    key = cache._config_hash(root / "m")
    config.write_text(yaml.safe_dump({"about": {"title": "Retitled"}, **raw}, sort_keys=False),
                      encoding="utf-8")
    assert cache._config_hash(root / "m") == key
    raw["dfms"][0]["chambers"] = {1: "Exp"}
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    assert cache._config_hash(root / "m") != key


def test_a_member_naming_its_config_outside_a_project_has_no_selection_home(tmp_path):
    from pyflic.base.selections import selection_file

    _recording(tmp_path)
    write_member(tmp_path / "p" / "m", {"config": "../../runs/R1/flic_config.yaml"})
    with pytest.raises(ValueError, match="names its recording's config"):
        selection_file(SimpleNamespace(experiment_dir=str(tmp_path / "p" / "m")))



# ---------------------------------------------------------------------------
# The editors open the one config
# ---------------------------------------------------------------------------

@pytest.fixture
def qt(monkeypatch):
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication, QMessageBox

    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    return QApplication.instance() or QApplication([])


def test_the_experiment_editor_on_a_member_folder_edits_the_recordings_config(tmp_path, qt):
    from pyflic.base.experiment_editor import ExperimentEditor

    run = _recording(tmp_path)
    member = tmp_path / "p" / "m"
    write_member(member, {"config": "../../runs/R1/flic_config.yaml"})
    editor = ExperimentEditor(member)
    try:
        assert Path(editor._current_path).resolve() == (run / prj.CONFIG_FILENAME).resolve()
        editor.title_edit.setText("Edited from the analysis")
        assert editor.write(Path(editor._current_path), confirm=False)
    finally:
        editor.close()
    assert not (member / prj.CONFIG_FILENAME).exists()          # never a second config
    saved = yaml.safe_load((run / prj.CONFIG_FILENAME).read_text(encoding="utf-8"))
    assert saved["about"]["title"] == "Edited from the analysis"


def test_the_config_editor_on_a_member_folder_opens_the_recordings_config(tmp_path, qt):
    from pyflic.base.config_editor import FLICConfigEditor

    run = _recording(tmp_path)
    member = tmp_path / "p" / "m"
    write_member(member, {"config": "../../runs/R1/flic_config.yaml"})
    editor = FLICConfigEditor(str(member))
    try:
        assert Path(editor._current_path).resolve() == (run / prj.CONFIG_FILENAME).resolve()
    finally:
        editor.close()
