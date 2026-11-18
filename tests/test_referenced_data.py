"""A Member that reads its recording from somewhere else (ADR-0021).

``data_folder:`` at the top of a Member's ``flic_config.yaml`` names the folder
that holds what ``data/`` holds today — the DFM CSVs, ``Program.txt``,
``Messages.csv`` and ``Baselines.csv`` — resolved against the Member folder.
pyflic only ever *reads* that folder: everything it writes for the Member stays
in the Member folder or the Project folder.

The shared tests (S1–S8) build the layout a study analysis has — a Project and
its Members under ``studies/``, recordings under ``runs/`` — beneath a prefix
with spaces, an apostrophe, an ampersand and non-ASCII characters, and make
``runs/`` read-only while the whole pipeline runs over it.  The pyflic tests
(F1–F6) pin the loader, the cache, the Design's inheritance, Slices, the
classifier and the dead ``data_dir:`` key.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
import shutil
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from pyflic.base import batch as batch_mod
from pyflic.base import cache, layout
from pyflic.base import project as prj
from pyflic.base.exclusion_sheet import apply_sheet
from pyflic.base.paradigm_state import BASELINE_COLUMNS
from pyflic.base.project import Project
from pyflic.base.project_report import member_table, report_path
from pyflic.base.yaml_config import load_experiment_yaml
from pyflic.base.yaml_lint import lint_flic_config

from v4_fixtures import choice_pr_spec, simulate, write_v4_recording

#: Fixed by the shared contract, so spelled out here rather than imported.
KEY = "data_folder"
ASSAY = "flic"
RUN1 = "R-20261020-test-01"
RUN2 = "R-20261021-test-02"
ANALYSIS = "A-20261105-test-01"
ANALYSIS2 = "A-20261105-test-02"
#: The member folders' own names sort RUN1, RUN2, then this one — so the third
#: member is the one reported as the duplicate (contract item 7).
DUPLICATE = "Z-also-test-01"

IS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0

#: The linter's warning about the retired key, exactly as it has read since
#: 2026-04-16 (commit 5769e9a).  It must not change.
DATA_DIR_WARNING = ("'data_dir' is no longer used and should be removed; "
                    "data is always read from experiment_dir/data/")


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def printed(fn, *args, **kwargs):
    """``(result, what fn printed)``."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = fn(*args, **kwargs)
    return result, out.getvalue()


# ---------------------------------------------------------------------------
# Recordings
# ---------------------------------------------------------------------------

#: The firmware 4.0.0 simulator's two-DFM choice recording: DFM CSVs,
#: Program.txt and Messages.csv, the files a run's data/ holds.
PIPELINE_PARAMS = {"feeding_threshold": 20, "feeding_minimum": 10, "tasting_minimum": 5,
                   "tasting_maximum": 20, "samples_per_second": 5, "pi_direction": "left"}
PIPELINE_DFMS = (1, 2)


@lru_cache(maxsize=None)
def _simulated(seed: int):
    spec = choice_pr_spec(dfm_ids=PIPELINE_DFMS, seed=seed)
    return spec, simulate(spec)


def write_v4(folder: Path, seed: int = 7, **kw) -> Path:
    spec, results = _simulated(seed)
    write_v4_recording(Path(folder), spec, results=results, **kw)
    return Path(folder)


def pipeline_global(**extra) -> dict:
    g = {"chamber_layout": "two_well", "transform_licks": False,
         "params": dict(PIPELINE_PARAMS)}
    g.update(extra)
    return g


def pipeline_dfms() -> list[dict]:
    return [{"id": int(i), "chambers": {c: ("Ctrl" if c % 2 else "Exp") for c in range(1, 7)}}
            for i in PIPELINE_DFMS]


#: A plain synthetic recording, for tests that need one quickly: five samples a
#: second, a feeding bout on every well every minute.
SIMPLE_PARAMS = {"feeding_threshold": 5, "feeding_minimum": 5, "tasting_minimum": 1,
                 "tasting_maximum": 4, "feeding_event_link_gap": 2, "samples_per_second": 5,
                 "correct_for_dual_feeding": False}
SIMPLE_CHAMBERS = {1: "Ctrl", 2: "Ctrl", 3: "Exp", 4: "Exp", 5: "Ctrl", 6: "Exp"}


def simple_global(**extra) -> dict:
    g = {"chamber_layout": "two_well", "transform_licks": False,
         "params": dict(SIMPLE_PARAMS)}
    g.update(extra)
    return g


def simple_dfms() -> list[dict]:
    return [{"id": 1, "chambers": dict(SIMPLE_CHAMBERS)}]


def write_simple(folder: Path, *, seed: int = 1, minutes: float = 20.0,
                 bout: int = 8, dfm_id: int = 1) -> Path:
    folder = Path(folder)
    n = int(minutes * 60 * 5)
    rng = np.random.default_rng(seed)
    cols: dict[str, object] = {"Sample": np.arange(1, n + 1), "Seconds": np.arange(n) / 5.0}
    for w in range(1, 13):
        sig = rng.normal(0.0, 0.3, size=n)
        for k in range(int(minutes)):
            start = 300 * k + 40 + (w * 7 + seed) % 30
            sig[start:start + bout] += 10.0 + rng.normal(0, 0.5, size=bout)
        cols[f"W{w}"] = np.round(sig, 3)
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cols).to_csv(folder / f"DFM{dfm_id}_0.csv", index=False)
    return folder


def write_config(member: Path, config: dict) -> Path:
    member = Path(member)
    member.mkdir(parents=True, exist_ok=True)
    path = member / prj.CONFIG_FILENAME
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")
    return path


def referencing(value, config: dict) -> dict:
    """*config* with the key first, as labdata writes it."""
    return {KEY: value, **copy.deepcopy(config)}


def load(member, **kwargs):
    kwargs.setdefault("parallel", False)
    kwargs.setdefault("use_disk_cache", False)
    kwargs.setdefault("eager", False)
    return quiet(load_experiment_yaml, member, **kwargs)


# ---------------------------------------------------------------------------
# The shared layout
# ---------------------------------------------------------------------------

def managed_root(tmp) -> Path:
    return Path(tmp) / "Experiments & Data" / "MBa'b'ap Ünï" / "managed"


def run_dir(managed, run) -> Path:
    return Path(managed) / "runs" / ASSAY / "2026" / run


def reference(run) -> str:
    """The value labdata writes: relative, forward slashes, five levels up."""
    return f"../../../../../runs/{ASSAY}/2026/{run}/data"


def make_run(managed, run, seed) -> Path:
    """A run folder: its recording in ``data/`` and, beside it, the config it
    was recorded with (an ordinary standalone Experiment Directory)."""
    folder = run_dir(managed, run)
    write_v4(folder / "data", seed=seed)
    write_config(folder, {"global": pipeline_global(), "dfms": pipeline_dfms()})
    return folder


def make_analysis(managed, name, members: dict, *, design_global=None) -> Path:
    """A Project under ``studies/`` whose Members reference runs.

    *members* maps a Member folder name to the run it reads.  Each Member's
    config is what labdata writes: the key and the run's ``dfms:``, inheriting
    ``global:`` from the Design.  The Project's ``batch`` script writes each
    Member's report too, so one Batch Run is the whole pipeline.
    """
    analysis = Path(managed) / "studies" / "test-study" / "analyses" / name
    analysis.mkdir(parents=True)
    prj.create_project_file(str(analysis),
                            design={"global": design_global or pipeline_global()})
    meta = yaml.safe_load((analysis / prj.PROJECT_FILENAME).read_text(encoding="utf-8"))
    meta["scripts"][0]["steps"][0]["reports"] = True
    (analysis / prj.PROJECT_FILENAME).write_text(
        yaml.safe_dump(meta, sort_keys=False, allow_unicode=True), encoding="utf-8")
    for member, run in members.items():
        write_config(analysis / member, {KEY: reference(run), "dfms": pipeline_dfms()})
    return analysis


def snapshot(root) -> dict:
    """Every path under *root* and, for files, its SHA-256."""
    root = Path(root)
    found = {}
    for directory, _dirs, files in os.walk(root):
        rel = os.path.relpath(directory, root)
        found[rel + "/"] = None
        for name in files:
            path = Path(directory) / name
            found[os.path.join(rel, name)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return found


def changed(before, after) -> set:
    return {path for path in set(before) | set(after)
            if before.get(path, "missing") != after.get(path, "missing")}


def lock(root) -> bool:
    """Make *root* read-only: files 0o444, folders 0o555.  False as root,
    where permissions do not stop a write."""
    if IS_ROOT:
        return False
    for directory, _dirs, files in os.walk(root):
        for name in files:
            os.chmod(os.path.join(directory, name), 0o444)
    for directory, _dirs, _files in os.walk(root, topdown=False):
        os.chmod(directory, 0o555)
    return True


def unlock(root) -> None:
    for directory, _dirs, files in os.walk(root):
        os.chmod(directory, 0o755)
        for name in files:
            os.chmod(os.path.join(directory, name), 0o644)


def copy_recording(source: Path, destination: Path) -> Path:
    """The files of a flat recording folder, contents only, never modes — a
    copy of a locked run is an ordinary writable folder."""
    destination.mkdir(parents=True, exist_ok=True)
    for path in Path(source).iterdir():
        if path.is_file():
            shutil.copyfile(path, destination / path.name)
    return destination


def _holds_paths(column: pd.Series) -> bool:
    """A column of file paths, which differ between a referencing Member and
    a copy for reasons that are not the analysis.  Nothing else is ignored:
    the recording's own timestamps are the same in both."""
    return column.dtype == object and column.astype(str).str.contains(os.sep, regex=False).any()


def summary_tables(member) -> dict:
    """A Member's per-member summary tables (``analysis/`` and ``qc/``)."""
    tables = {}
    member = Path(member)
    for path in sorted([*member.glob("analysis/**/*.csv"), *member.glob("qc/**/*.csv")]):
        frame = pd.read_csv(path)
        drop = [c for c in frame.columns if _holds_paths(frame[c])]
        tables[path.relative_to(member).as_posix()] = frame.drop(columns=drop)
    return tables


# ---------------------------------------------------------------------------
# One full pipeline over a read-only runs/ tree (S2, S3, S4, S7)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Build the shared layout, hash and lock ``runs/``, then run everything
    that reads it: a Batch Run (the Exclusion Sheet, each Member's QC,
    analysis and report, the Combined Analysis and the Project Report), the
    layout classification, Batch discovery from the analyses folder and from
    the managed root, the preflight dialog and the linter."""
    tmp = tmp_path_factory.mktemp("referenced")
    managed = managed_root(tmp)
    make_run(managed, RUN1, seed=7)
    make_run(managed, RUN2, seed=11)
    analysis = make_analysis(managed, ANALYSIS, {RUN1: RUN1, RUN2: RUN2})
    batch_root = analysis.parent
    ## An Exclusion Sheet at the Batch root: a Batch Run writes its rows into
    ## each Member's own remove_chambers.csv (ADR-0010).
    (batch_root / "remove_chambers.csv").write_text(
        f"project,member,dfm,chamber,reason\n{ANALYSIS},{RUN1},1,1,test\n", encoding="utf-8")
    runs = managed / "runs"

    runs_before = snapshot(runs)
    tree_before = snapshot(tmp)
    locked = lock(runs)
    try:
        logs: list = []
        results = quiet(batch_mod.run_batch, str(batch_root), log=logs.append)
        layouts = layout.members_in(analysis)
        classified = {name: layout.classify(analysis / name) for name in (RUN1, RUN2)}
        found = batch_mod.discover(batch_root)
        found_managed = batch_mod.discover(managed)
        lint = {name: lint_flic_config(analysis / name / prj.CONFIG_FILENAME)
                for name in (RUN1, RUN2)}

        pytest.importorskip("PyQt6")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication

        from pyflic.base.batch_preflight import BatchPreflightDialog

        app = QApplication.instance() or QApplication([])
        dialog = BatchPreflightDialog(None, str(batch_root), log=lambda _m: None)
        dialog.close()
        app.processEvents()

        yield SimpleNamespace(
            tmp=tmp, managed=managed, runs=runs, analysis=analysis,
            batch_root=batch_root, results=results, logs=logs,
            layouts=layouts, classified=classified, found=found,
            found_managed=found_managed, lint=lint,
            runs_before=runs_before, runs_after=snapshot(runs),
            tree_before=tree_before, tree_after=snapshot(tmp), locked=locked)
    finally:
        unlock(runs)


def test_s2_the_full_pipeline_runs_through_the_key(pipeline):
    assert pipeline.results == {ANALYSIS: "ok"}, "\n".join(pipeline.logs)
    project = Project(pipeline.analysis)
    assert project.member_names == [RUN1, RUN2]
    for name in (RUN1, RUN2):
        member = pipeline.analysis / name
        assert (member / "analysis" / "feeding_summary.csv").is_file()
        assert (member / "analysis" / "summary.txt").is_file()
        assert any((member / "qc").iterdir())
        assert os.path.isfile(project.member_report_path(name))
        assert not (member / "data").exists()
    assert (Path(project.analysis_path) / f"{ANALYSIS}_Summary.csv").is_file()
    assert os.path.isfile(report_path(project))
    ## The sheet's row was written into the Member, not beside the recording.
    declared = (pipeline.analysis / RUN1 / "remove_chambers.csv").read_text(encoding="utf-8")
    assert "test" in declared


def test_s2_every_output_is_under_a_member_or_the_project(pipeline):
    inside = os.path.relpath(pipeline.analysis, pipeline.tmp)
    new = changed(pipeline.tree_before, pipeline.tree_after)
    assert new, "the pipeline wrote nothing"
    assert all(path == inside + "/" or path.startswith(inside + os.sep)
               for path in new), sorted(p for p in new if not p.startswith(inside))


def test_s2_the_classifier_and_discovery_see_two_healthy_members(pipeline):
    assert all(item.status == layout.OK for item in pipeline.classified.values())
    assert [(m.name, m.status) for m in pipeline.layouts] == [(RUN1, layout.OK),
                                                              (RUN2, layout.OK)]
    assert {item.dfm_ids for item in pipeline.layouts} == {PIPELINE_DFMS}
    (found,) = pipeline.found["projects"]
    assert found.key == ANALYSIS and not found.blocked
    keys = [item.key for item in pipeline.found_managed["projects"]]
    assert any(key.endswith(ANALYSIS) for key in keys)


def test_s2_the_linter_finds_nothing_wrong_with_labdatas_configs(pipeline):
    for name, issues in pipeline.lint.items():
        assert not [i for i in issues if i.severity == "error"], (name, issues)
        assert not [i for i in issues if KEY in i.message], (name, issues)


def test_s3_the_recordings_are_untouched(pipeline):
    assert pipeline.runs_after == pipeline.runs_before


@pytest.mark.skipif(IS_ROOT, reason="running as root: file permissions do not stop "
                                    "a write, so only the hash comparison is meaningful")
def test_s3_the_recordings_were_read_only_throughout(pipeline):
    assert pipeline.locked
    for directory, _dirs, files in os.walk(pipeline.runs):
        assert not os.access(directory, os.W_OK)
        for name in files:
            assert not os.access(os.path.join(directory, name), os.W_OK)


def test_s4_a_referencing_member_gets_the_same_numbers_as_a_copy(pipeline, tmp_path):
    plain = tmp_path / "plain"
    prj.create_project_file(str(plain), design={"global": pipeline_global()})
    copy_recording(run_dir(pipeline.managed, RUN1) / "data", plain / RUN1 / "data")
    write_config(plain / RUN1, {"dfms": pipeline_dfms()})
    ## The same exclusion the Batch Run wrote into the referencing Member.
    shutil.copyfile(pipeline.analysis / RUN1 / "remove_chambers.csv",
                    plain / RUN1 / "remove_chambers.csv")
    assert quiet(Project(plain).run_all, make_reports=False) == []

    referenced = summary_tables(pipeline.analysis / RUN1)
    copied = summary_tables(plain / RUN1)
    assert {"analysis/feeding_summary.csv", "analysis/light_summary.csv"} <= set(referenced)
    assert any(name.startswith("qc/opto/") for name in referenced)   # Program.txt was read
    assert set(referenced) == set(copied)
    for name, frame in referenced.items():
        pd.testing.assert_frame_equal(frame, copied[name], obj=name)


def test_s7_two_projects_can_read_one_recording(pipeline):
    first = snapshot(pipeline.analysis)
    second = make_analysis(
        pipeline.managed, ANALYSIS2, {RUN1: RUN1},
        design_global=pipeline_global(params={**PIPELINE_PARAMS, "feeding_threshold": 40}))
    logs: list = []
    results = quiet(batch_mod.run_batch, str(second.parent), project_names=[ANALYSIS2],
                    log=logs.append)
    assert results == {ANALYSIS2: "ok"}, "\n".join(logs)
    assert snapshot(pipeline.analysis) == first
    assert snapshot(pipeline.runs) == pipeline.runs_before
    own = pd.read_csv(second / RUN1 / "analysis" / "feeding_summary.csv")
    theirs = pd.read_csv(pipeline.analysis / RUN1 / "analysis" / "feeding_summary.csv")
    assert not own["LicksA"].equals(theirs["LicksA"])


# ---------------------------------------------------------------------------
# Duplicates (S5)
# ---------------------------------------------------------------------------

def _simple_run(managed, run, seed) -> Path:
    folder = run_dir(managed, run)
    write_simple(folder / "data", seed=seed)
    return folder


def _simple_analysis(managed, name, members: dict, *, design_global=None) -> Path:
    analysis = Path(managed) / "studies" / "test-study" / "analyses" / name
    prj.create_project_file(str(analysis), design={"global": design_global or simple_global()})
    for member, run in members.items():
        write_config(analysis / member, {KEY: reference(run), "dfms": simple_dfms()})
    return analysis


@pytest.fixture
def three_members(tmp_path):
    managed = managed_root(tmp_path)
    _simple_run(managed, RUN1, seed=1)
    _simple_run(managed, RUN2, seed=2)
    return _simple_analysis(managed, ANALYSIS, {RUN1: RUN1, RUN2: RUN2, DUPLICATE: RUN1})


def test_s5_the_project_keeps_the_first_and_names_the_duplicate(three_members):
    project = Project(three_members)
    assert project.member_names == [RUN1, RUN2]
    assert project.duplicates == {DUPLICATE: RUN1}
    assert any(DUPLICATE in w and RUN1 in w and "duplicate" in w for w in project.warnings)


def test_s5_layout_and_discovery_report_the_duplicate_as_blocked(three_members):
    found = {item.name: item for item in layout.members_in(three_members)}
    assert found[RUN1].status == layout.OK and found[RUN2].status == layout.OK
    duplicate = found[DUPLICATE]
    assert duplicate.status == layout.DUPLICATE and duplicate.blocked
    assert duplicate.fix is None
    assert RUN1 in duplicate.detail
    (item,) = batch_mod.discover(three_members.parent)["projects"]
    assert [m.name for m in item.blocked] == [DUPLICATE]
    assert len(item.usable) == 2


def test_s5_a_batch_run_names_the_duplicate(three_members):
    logs: list = []
    quiet(batch_mod.run_batch, str(three_members.parent), log=logs.append)
    assert any(DUPLICATE in line and layout.DUPLICATE in line for line in logs)


def test_s5_the_duplicates_chambers_appear_once_in_project_output(three_members):
    project = Project(three_members)
    assert quiet(project.run_all, make_reports=False) == []
    assert not (three_members / DUPLICATE / "analysis").exists()
    quiet(project.build_combined_analysis)
    pooled = pd.read_csv(Path(project.analysis_path) / f"{project.name}_Summary.csv")
    assert set(pooled["Experiment"]) == {RUN1, RUN2}
    one = pd.read_csv(three_members / RUN1 / "analysis" / "feeding_summary.csv")
    assert (pooled["Experiment"] == RUN1).sum() == len(one)
    assert DUPLICATE not in set(member_table(project)["Member"])


def test_s5_whichever_member_sorts_first_is_the_one_kept(three_members):
    write_config(three_members / "A-first", {KEY: reference(RUN1), "dfms": simple_dfms()})
    project = Project(three_members)
    assert project.member_names == ["A-first", RUN2]
    assert project.duplicates == {RUN1: "A-first", DUPLICATE: "A-first"}
    blocked = {m.name for m in layout.members_in(three_members) if m.blocked}
    assert blocked == {RUN1, DUPLICATE}


def test_s5_a_plain_member_and_a_reference_to_its_data_are_one_member(tmp_path):
    root = tmp_path / "proj"
    prj.create_project_file(str(root), design={"global": simple_global()})
    write_simple(root / "M1" / "data")
    write_config(root / "M1", {"dfms": simple_dfms()})
    write_config(root / "M2", {KEY: "../M1/data", "dfms": simple_dfms()})
    project = Project(root)
    assert project.member_names == ["M1"] and project.duplicates == {"M2": "M1"}


def test_without_the_key_symlinked_data_folders_keep_todays_behaviour(tmp_path):
    root = tmp_path / "proj"
    prj.create_project_file(str(root), design={"global": simple_global()})
    shared = write_simple(tmp_path / "shared")
    for name in ("M1", "M2"):
        (root / name).mkdir(parents=True)
        os.symlink(shared, root / name / "data")
        write_config(root / name, {"dfms": simple_dfms()})
    project = Project(root)
    assert project.member_names == ["M1", "M2"] and project.duplicates == {}
    assert [m.status for m in layout.members_in(root)] == [layout.OK, layout.OK]


# ---------------------------------------------------------------------------
# One member, one location: the classifier and the loader (S6, F5)
# ---------------------------------------------------------------------------

def referencing_member(tmp_path, value="../../runs/R1/data", name="M1", *,
                       recording=True, **extra) -> Path:
    """A standalone Member folder whose config names *value*, beside a run
    holding a recording at ``runs/R1/data``."""
    if recording:
        write_simple(Path(tmp_path) / "runs" / "R1" / "data")
    member = Path(tmp_path) / "proj" / name
    write_config(member, {KEY: value, "global": simple_global(), "dfms": simple_dfms(),
                          **extra})
    return member


def outcome(member):
    """``(the classifier's answer, the loader's refusal or None)``.  A
    reference the loader cannot use is a :class:`layout.DataLocationError`;
    a usable folder with no recording in it is the loader's ordinary
    "No DFMs to load"."""
    item = layout.classify(member)
    try:
        load(member)
    except ValueError as err:
        return item, str(err)
    return item, None


def test_f5_a_good_folder_is_healthy_and_loads_from_there(tmp_path):
    member = referencing_member(tmp_path)
    item, error = outcome(member)
    assert item.status == layout.OK and item.dfm_ids == (1,)
    assert error is None
    exp = load(member)
    target = os.path.realpath(tmp_path / "runs" / "R1" / "data")
    assert os.path.realpath(exp.data_dir) == target
    assert all(os.path.dirname(os.path.realpath(p)) == target
               for p in exp.dfms[1].source_files)


def test_f5_a_missing_folder_blocks_and_the_loader_names_it(tmp_path):
    member = referencing_member(tmp_path, "../../runs/R9/data")
    resolved = os.path.realpath(tmp_path / "runs" / "R9" / "data")
    item, error = outcome(member)
    assert item.status == layout.NO_RECORDING and item.blocked
    assert resolved in item.detail
    assert error is not None
    assert KEY in error and "../../runs/R9/data" in error and resolved in error


def test_f5_a_file_where_a_folder_belongs_blocks_and_the_loader_names_it(tmp_path):
    member = referencing_member(tmp_path, "../../runs/R1/data/DFM1_0.csv")
    item, error = outcome(member)
    assert item.status == layout.NO_RECORDING
    assert "file" in item.detail and "DFM1_0.csv" in item.detail
    assert error is not None and KEY in error and "not a folder" in error


def test_f5_the_key_and_a_local_recording_conflict(tmp_path):
    member = referencing_member(tmp_path)
    write_simple(member / "data")
    item, error = outcome(member)
    assert item.status == layout.AMBIGUOUS
    resolved = os.path.realpath(tmp_path / "runs" / "R1" / "data")
    assert resolved in item.detail and "data/DFM1_0.csv" in item.detail
    assert error is not None and resolved in error and "data/DFM1_0.csv" in error


def test_f5_an_empty_local_data_folder_is_not_a_conflict(tmp_path):
    member = referencing_member(tmp_path)
    (member / "data").mkdir()
    item, error = outcome(member)
    assert item.status == layout.OK and error is None


def test_f5_a_loose_recording_beside_the_key_is_a_conflict_not_unfiled(tmp_path):
    member = referencing_member(tmp_path)
    write_simple(member)                          # DFM1_0.csv at the Member root
    item, error = outcome(member)
    assert item.status == layout.AMBIGUOUS and error is not None


def test_f5_a_referenced_folder_one_level_too_high_says_where_the_recording_is(tmp_path):
    member = referencing_member(tmp_path, "../../runs/R1")
    item, error = outcome(member)
    assert item.status == layout.NO_RECORDING and item.status != layout.UNFILED
    assert "DFM1_0.csv" in item.detail and KEY in item.detail
    assert error is not None and "No DFMs to load" in error


@pytest.mark.parametrize("arrange", ["loose at the member root", "in the referenced folder"])
def test_f5_filing_a_referencing_member_is_refused_and_moves_nothing(tmp_path, arrange):
    member = referencing_member(tmp_path)
    if arrange == "loose at the member root":
        write_simple(member, dfm_id=2)
    before = snapshot(tmp_path)
    plan = layout.plan_filing(member)
    assert plan.refused and KEY in plan.refused and not plan.moves
    assert layout.file_recording(member, log=lambda _m: None).refused
    assert snapshot(tmp_path) == before


def test_a_csv_that_is_not_a_recording_beside_the_key_leaves_the_member_healthy(tmp_path):
    member = referencing_member(tmp_path)
    (member / "notes.txt").write_text("the run's notebook page", encoding="utf-8")
    (member / "units.csv").write_text("dfm,chamber,unit\n1,1,fly-1\n", encoding="utf-8")
    item, error = outcome(member)
    assert item.status == layout.OK and error is None
    assert load(member).dfms


@pytest.mark.parametrize("name", ["DFM1_0.csv", "dfm1_0.csv", "Program.txt", "Messages.csv"])
def test_a_recording_file_in_the_local_data_folder_still_conflicts(tmp_path, name):
    member = referencing_member(tmp_path)
    (member / "data").mkdir()
    (member / "data" / name).write_text("x\n", encoding="utf-8")
    item, error = outcome(member)
    assert item.status == layout.AMBIGUOUS and name in item.detail
    assert error is not None and name in error


def test_hidden_and_lock_files_in_the_local_data_folder_are_not_a_recording(tmp_path):
    member = referencing_member(tmp_path)
    (member / "data").mkdir()
    for name in (".DS_Store", "~$DFM1_0.csv", "readme.txt"):
        (member / "data" / name).write_text("x\n", encoding="utf-8")
    item, error = outcome(member)
    assert item.status == layout.OK and error is None


@pytest.mark.parametrize("value", [".", "..", "./"])
def test_the_key_may_not_name_a_folder_holding_the_member(tmp_path, value):
    member = referencing_member(tmp_path, value)
    item, error = outcome(member)
    assert item.status == layout.AMBIGUOUS and "nothing is ever written" in item.detail
    assert error is not None


def test_the_key_may_name_the_members_own_data_folder(tmp_path):
    member = referencing_member(tmp_path, "data", recording=False)
    write_simple(member / "data")
    item, error = outcome(member)
    assert item.status == layout.OK and error is None


def test_an_absolute_value_works(tmp_path):
    target = write_simple(tmp_path / "runs" / "R1" / "data")
    member = referencing_member(tmp_path, str(target), recording=False)
    item, error = outcome(member)
    assert item.status == layout.OK and error is None


@pytest.mark.parametrize("value", ["", "   ", None])
def test_a_blank_value_is_no_key_at_all(tmp_path, value):
    member = referencing_member(tmp_path, value, recording=False)
    write_simple(member / "data")
    location = layout.data_location(member)
    assert not location.referenced
    assert location.path == os.path.join(str(member), "data")
    assert layout.classify(member).status == layout.OK


def test_a_value_that_is_not_a_path_blocks_and_fails_to_load(tmp_path):
    member = referencing_member(tmp_path, ["../../runs/R1/data"])
    item, error = outcome(member)
    assert item.status == layout.NO_RECORDING and error is not None and KEY in error


def test_without_the_key_the_resolver_is_todays_data_folder(tmp_path):
    member = tmp_path / "exp"
    write_simple(member / "data")
    write_config(member, {"global": simple_global(), "dfms": simple_dfms()})
    location = layout.data_location(member)
    assert not location.referenced and location.problem == ""
    assert location.path == layout.data_dir(member) == os.path.join(str(member), "data")


def test_data_discovery_reads_the_referenced_folder(tmp_path):
    member = referencing_member(tmp_path)
    assert prj.has_experiment_data(member)
    assert prj.dfm_ids_in_data(member) == [1]
    broken = referencing_member(tmp_path / "b", "../../runs/R9/data", recording=False)
    assert not prj.has_experiment_data(broken) and prj.dfm_ids_in_data(broken) == []


# ---------------------------------------------------------------------------
# Never written: scaffolding, the Exclusion Sheet, the Project
# ---------------------------------------------------------------------------

def _project_of(tmp_path, members: dict) -> Path:
    """A Project at ``tmp/proj`` whose Members reference ``tmp/runs/<run>/data``."""
    root = tmp_path / "proj"
    prj.create_project_file(str(root), design={"global": simple_global()})
    for member, run in members.items():
        if not (tmp_path / "runs" / run / "data").is_dir():
            write_simple(tmp_path / "runs" / run / "data", seed=len(run))
        write_config(root / member, {KEY: f"../../runs/{run}/data", "dfms": simple_dfms()})
    return root


def test_scaffolding_never_makes_a_data_folder_in_a_referencing_member(tmp_path):
    root = _project_of(tmp_path, {"M1": "R1"})
    with pytest.raises(FileExistsError):
        Project(root).scaffold_member("M1")
    assert not (root / "M1" / "data").exists()


def test_a_scaffolded_member_does_not_inherit_the_key(tmp_path):
    root = _project_of(tmp_path, {"M1": "R1"})
    write_simple(root / "M2" / "data")
    path, _notes = quiet(Project(root).scaffold_member, "M2")
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assert KEY not in config and [d["id"] for d in config["dfms"]] == [1]


def test_an_exclusion_sheet_writes_only_into_members(tmp_path):
    root = _project_of(tmp_path, {"M1": "R1", "M2": "R2"})
    (root / "remove_chambers.csv").write_text(
        "member,dfm,chamber,reason\nM1,1,2,dead\nM2,1,3,dead\n", encoding="utf-8")
    runs = snapshot(tmp_path / "runs")
    locked = lock(tmp_path / "runs")
    try:
        quiet(apply_sheet, str(root), log=lambda _m: None)
    finally:
        if locked:
            unlock(tmp_path / "runs")
    assert snapshot(tmp_path / "runs") == runs
    for member in ("M1", "M2"):
        assert (root / member / "remove_chambers.csv").is_file()


def test_copying_a_referencing_config_into_a_new_member_leaves_the_key(qt_app, tmp_path,
                                                                       monkeypatch):
    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    from pyflic.base.hub import AnalysisHubWindow

    root = _project_of(tmp_path, {"M1": "R1"})
    for name in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, name,
                            staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))
    source = root / "M1" / prj.CONFIG_FILENAME
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(source), "")))
    hub = AnalysisHubWindow(target=str(root))
    try:
        quiet(Project(root).scaffold_member, "M2")
        hub._reload_project()
        assert hub._copy_member_config("M2", root / "M2") is True
    finally:
        hub.close()
    copied = yaml.safe_load((root / "M2" / prj.CONFIG_FILENAME).read_text(encoding="utf-8"))
    assert KEY not in copied and copied["dfms"] == simple_dfms()
    assert Project(root).duplicates == {}


def test_a_broken_reference_is_a_project_warning_and_a_named_failure(tmp_path):
    root = _project_of(tmp_path, {"M1": "R1"})
    write_config(root / "M2", {KEY: "../../runs/R9/data", "dfms": simple_dfms()})
    project = Project(root)
    resolved = os.path.realpath(tmp_path / "runs" / "R9" / "data")
    assert project.member_names == ["M1", "M2"]
    assert any(w.startswith("M2:") and resolved in w for w in project.warnings)
    failures = quiet(project.run_all, make_reports=False)
    assert len(failures) == 1 and failures[0].startswith("M2:") and KEY in failures[0]
    assert not (tmp_path / "runs" / "R9").exists()


# ---------------------------------------------------------------------------
# F1: every recording file comes from the referenced folder
# ---------------------------------------------------------------------------

def _add_baseline_sets(folder: Path, dfm_ids) -> None:
    """PyMCU 4.0.1's 91st column, ``BaselineSet``, naming set 1 on every
    row, and the ``Baselines.csv`` that lists it."""
    for path in sorted(folder.glob("DFM*_*.csv")):
        lines = path.read_text(encoding="utf-8").splitlines()
        lines = [lines[0] + ",BaselineSet"] + [line + ",1" for line in lines[1:]]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = ["DFM,Set,Date,Time,Reason," + ",".join(BASELINE_COLUMNS)]
    rows += [f"{d},1,11/15/2026,10:00:00,Start," + ",".join(["10"] * 12) for d in dfm_ids]
    (folder / "Baselines.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_f1_the_loader_reads_every_recording_file_from_the_referenced_folder(tmp_path):
    folder = write_v4(tmp_path / "runs" / "R1" / "data")
    _add_baseline_sets(folder, PIPELINE_DFMS)
    assert {"Program.txt", "Messages.csv", "Baselines.csv"} <= {p.name for p in folder.iterdir()}
    member = tmp_path / "proj" / "M1"
    write_config(member, {KEY: "../../runs/R1/data", "global": pipeline_global(),
                          "dfms": pipeline_dfms()})
    exp = load(member)
    target = os.path.realpath(folder)
    assert sorted(exp.dfms) == list(PIPELINE_DFMS)
    for dfm in exp.dfms.values():
        assert dfm.source_files
        assert {os.path.dirname(os.path.realpath(p)) for p in dfm.source_files} == {target}
        assert dfm.baseline_in_force is not None                    # Baselines.csv
        assert dfm._messages() is not None                          # Messages.csv
    assert exp.opto_program is not None                             # Program.txt
    assert sorted(exp.opto_program.dfms) == list(PIPELINE_DFMS)
    assert not (member / "data").exists()


def test_the_load_message_and_the_summary_say_where_the_data_came_from(tmp_path):
    member = referencing_member(tmp_path)
    resolved = os.path.realpath(tmp_path / "runs" / "R1" / "data")
    exp, out = printed(load_experiment_yaml, member, parallel=False, use_disk_cache=False,
                       eager=False)
    line = next(l for l in out.splitlines() if l.startswith("Loading "))
    assert resolved in line and f"{KEY}: ../../runs/R1/data" in line
    text = exp.summary_text(include_qc=False)
    assert resolved in text and "../../runs/R1/data" in text


def test_without_the_key_the_load_message_and_the_summary_are_todays(tmp_path):
    member = tmp_path / "exp"
    write_simple(member / "data")
    write_config(member, {"global": simple_global(), "dfms": simple_dfms()})
    exp, out = printed(load_experiment_yaml, member, parallel=False, use_disk_cache=False,
                       eager=False)
    data = member.resolve() / "data"
    assert f"Loading 1 DFM(s) [1] from {data}\n" in out
    assert KEY not in out
    text = exp.summary_text(include_qc=False)
    assert "Data dir" not in text and "Data folder" not in text and KEY not in text
    assert exp.data_dir is not None and Path(exp.data_dir) == data


def test_the_project_reports_member_table_names_the_location(tmp_path):
    root = _project_of(tmp_path, {"M1": "R1"})
    project = Project(root)
    table = member_table(project)
    resolved = os.path.realpath(tmp_path / "runs" / "R1" / "data")
    (cell,) = list(table["Data folder"])
    assert "../../runs/R1/data" in cell and resolved in cell


def test_without_the_key_the_member_table_is_todays(tmp_path):
    root = tmp_path / "proj"
    prj.create_project_file(str(root), design={"global": simple_global()})
    write_simple(root / "M1" / "data")
    write_config(root / "M1", {"dfms": simple_dfms()})
    assert "Data folder" not in member_table(Project(root)).columns


# ---------------------------------------------------------------------------
# F2: the cache follows the referenced folder
# ---------------------------------------------------------------------------

def _summary(member, *, cached: bool):
    exp, out = printed(load_experiment_yaml, member, parallel=False, use_disk_cache=cached)
    return exp.feeding_summary(), out


def test_f2_repointing_a_member_never_reuses_the_other_recordings_summary(tmp_path):
    write_simple(tmp_path / "runs" / "A" / "data", seed=1, bout=8)
    write_simple(tmp_path / "runs" / "B" / "data", seed=2, bout=14)
    member = tmp_path / "proj" / "M1"
    config = {"global": simple_global(), "dfms": simple_dfms()}
    write_config(member, referencing("../../runs/A/data", config))
    a, _out = _summary(member, cached=True)
    write_config(member, referencing("../../runs/B/data", config))
    b, out = _summary(member, cached=True)
    assert "from disk cache" not in out
    fresh, _out = _summary(member, cached=False)
    pd.testing.assert_frame_equal(b, fresh)
    assert not b["LicksA"].equals(a["LicksA"])


def test_f2_new_files_at_an_unchanged_reference_are_not_served_from_the_cache(tmp_path):
    write_simple(tmp_path / "A", seed=1, bout=8)
    write_simple(tmp_path / "B", seed=2, bout=14)
    current = copy_recording(tmp_path / "A", tmp_path / "runs" / "current")
    member = tmp_path / "proj" / "M1"
    write_config(member, referencing("../../runs/current",
                                     {"global": simple_global(), "dfms": simple_dfms()}))
    a, _out = _summary(member, cached=True)
    _again, out = _summary(member, cached=True)
    assert "from disk cache" in out                       # the cache does work
    shutil.copyfile(tmp_path / "B" / "DFM1_0.csv", current / "DFM1_0.csv")
    b, out = _summary(member, cached=True)
    assert "from disk cache" not in out
    assert not b["LicksA"].equals(a["LicksA"])
    assert not (member / "data").exists()


def test_without_the_key_the_cache_key_is_todays(tmp_path):
    """The data half of the key, computed the way it was before the key
    existed: a plain Member's cached summaries stay valid."""
    member = tmp_path / "exp"
    data = write_simple(member / "data")
    (data / "Program.txt").write_text("x\n", encoding="utf-8")
    write_config(member, {"global": simple_global(), "dfms": simple_dfms()})
    rows = []
    for f in sorted(data.glob("DFM*.csv")):
        st = f.stat()
        rows.append((f.name, st.st_mtime_ns, st.st_size))
    for f in sorted(data.iterdir()):
        if f.name.lower() in ("program.txt", "messages.csv", "baselines.csv") and f.is_file():
            st = f.stat()
            rows.append((f.name, st.st_mtime_ns, st.st_size))
    expected = hashlib.sha256(json.dumps(rows).encode("utf-8")).hexdigest()[:16]
    assert cache._data_hash(member) == expected


# ---------------------------------------------------------------------------
# F3: the Design is inherited exactly as before
# ---------------------------------------------------------------------------

def test_f3_a_referencing_member_without_global_inherits_the_design(tmp_path):
    root = tmp_path / "proj"
    design = simple_global(params={**SIMPLE_PARAMS, "feeding_threshold": 6})
    prj.create_project_file(str(root), design={"global": design})
    write_simple(tmp_path / "runs" / "R1" / "data")
    write_config(root / "M1", {KEY: "../../runs/R1/data", "dfms": simple_dfms()})
    exp = quiet(Project(root).load_member, "M1", parallel=False, use_disk_cache=False,
                eager=False)
    assert exp.global_config["params"]["feeding_threshold"] == 6
    assert exp.dfms[1].params.feeding_threshold == 6


def test_f3_a_stated_global_is_checked_key_by_key(tmp_path):
    root = tmp_path / "proj"
    prj.create_project_file(str(root), design={"global": simple_global()})
    write_simple(tmp_path / "runs" / "R1" / "data")
    write_config(root / "M1", {KEY: "../../runs/R1/data",
                               "global": {"transform_licks": False}, "dfms": simple_dfms()})
    assert Project(root).member_names == ["M1"]
    write_config(root / "M1", {KEY: "../../runs/R1/data",
                               "global": {"transform_licks": True}, "dfms": simple_dfms()})
    with pytest.raises(ValueError, match="global.transform_licks"):
        Project(root)


# ---------------------------------------------------------------------------
# F4: Slices
# ---------------------------------------------------------------------------

SLICES = {"early": {"start": 0, "end": 10}, "late": {"start": 10, "end": "end"}}


def test_f4_slice_output_lands_in_the_member_and_never_in_the_referenced_folder(tmp_path):
    root = tmp_path / "proj"
    prj.create_project_file(str(root), design={"global": simple_global(slices=SLICES)})
    for member, seed in (("M1", 1), ("M2", 2)):
        write_simple(tmp_path / "runs" / member / "data", seed=seed)
        write_config(root / member, {KEY: f"../../runs/{member}/data", "dfms": simple_dfms()})
    runs = tmp_path / "runs"
    before = snapshot(runs)
    locked = lock(runs)
    try:
        project = Project(root)
        assert quiet(project.run_all, make_reports=False) == []
        results = quiet(project.build_all_combined_analyses)
    finally:
        if locked:
            unlock(runs)
    assert snapshot(runs) == before
    for member in ("M1", "M2"):
        for name in SLICES:
            assert (root / member / "slices" / name / "analysis"
                    / "feeding_summary.csv").is_file()
            assert (root / member / "slices" / name / "slice.json").is_file()
    assert set(results) == {None, "early", "late"}
    pooled = pd.read_csv(Path(project.for_slice("early").analysis_path) / "proj_Summary.csv")
    assert set(pooled["Experiment"]) == {"M1", "M2"}


# ---------------------------------------------------------------------------
# F6: data_dir stays dead
# ---------------------------------------------------------------------------

def test_f6_data_dir_still_reads_the_members_own_data_folder(tmp_path):
    other = write_simple(tmp_path / "other", seed=99, bout=14)
    member = tmp_path / "exp"
    write_simple(member / "data", seed=1)
    write_config(member, {"data_dir": str(other), "global": simple_global(),
                          "dfms": simple_dfms()})
    assert not layout.data_location(member).referenced
    exp = load(member)
    assert {Path(p).parent for p in exp.dfms[1].source_files} == {(member / "data").resolve()}
    issues = lint_flic_config(member / prj.CONFIG_FILENAME)
    assert [i.message for i in issues if i.path == "data_dir"] == [DATA_DIR_WARNING]
    assert all(i.severity == "warning" for i in issues if i.path == "data_dir")


# ---------------------------------------------------------------------------
# S8: the linter
# ---------------------------------------------------------------------------

def _issues(member) -> list:
    return lint_flic_config(Path(member) / prj.CONFIG_FILENAME)


def test_s8_the_key_is_accepted(tmp_path):
    issues = _issues(referencing_member(tmp_path))
    assert not [i for i in issues if KEY in i.message]
    assert not [i for i in issues if i.severity == "error"]


def test_s8_an_absolute_value_warns_that_it_is_not_portable(tmp_path):
    target = write_simple(tmp_path / "runs" / "R1" / "data")
    issues = _issues(referencing_member(tmp_path, str(target), recording=False))
    (warning,) = [i for i in issues if i.path == KEY]
    assert warning.severity == "warning" and "absolute" in warning.message
    assert "portable" in warning.message


def test_s8_a_backslash_warns_to_write_forward_slashes(tmp_path):
    issues = _issues(referencing_member(tmp_path, "..\\..\\runs\\R1\\data"))
    assert any(i.path == KEY and i.severity == "warning" and "forward slash" in i.message
               for i in issues)


@pytest.mark.parametrize("value,words", [
    ("../../runs/R9/data", "does not exist"),
    ("../../runs/R1/data/DFM1_0.csv", "not a folder"),
    (["a"], "not a folder path"),
])
def test_s8_a_target_the_loader_would_refuse_is_an_error(tmp_path, value, words):
    issues = _issues(referencing_member(tmp_path, value))
    errors = [i for i in issues if i.path == KEY and i.severity == "error"]
    assert errors and words in errors[0].message


def test_s8_a_local_recording_beside_the_key_is_an_error(tmp_path):
    member = referencing_member(tmp_path)
    write_simple(member / "data")
    assert any(i.path == KEY and i.severity == "error" for i in _issues(member))


def test_the_linter_reads_the_recording_where_the_loader_does(tmp_path):
    """A generation-3 recording without its Program.txt is warned of — which
    the linter can only know by reading the referenced folder."""
    folder = write_v4(tmp_path / "runs" / "R1" / "data", program=False)
    member = tmp_path / "proj" / "M1"
    write_config(member, {KEY: "../../runs/R1/data", "global": pipeline_global(),
                          "dfms": pipeline_dfms()})
    assert any("Program.txt" in i.message for i in _issues(member))
    write_v4(folder, program=True)
    assert not any("Program.txt" in i.message and i.severity == "warning"
                   for i in _issues(member))


# ---------------------------------------------------------------------------
# The Config Editor
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qt_app():
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _editor(member):
    from pyflic.base.config_editor import FLICConfigEditor

    return FLICConfigEditor(initial_path=member)


def test_the_config_editor_round_trips_and_explains_the_key(qt_app, tmp_path):
    member = referencing_member(tmp_path)
    win = _editor(member)
    try:
        field = win._data_folder_edit
        assert field.text() == "../../runs/R1/data"
        assert KEY in field.toolTip() and "data/" in field.toolTip()
        assert win._collect_yaml()[KEY] == "../../runs/R1/data"
        field.setText("../../runs/R2/data")
        assert win._collect_yaml()[KEY] == "../../runs/R2/data"
        field.setText("")
        assert KEY not in win._collect_yaml()
    finally:
        win.close()


def test_the_config_editor_adds_no_key_to_a_plain_config(qt_app, tmp_path):
    member = tmp_path / "exp"
    write_simple(member / "data")
    write_config(member, {"global": simple_global(), "dfms": simple_dfms()})
    win = _editor(member)
    try:
        assert win._data_folder_edit.text() == ""
        assert KEY not in win._collect_yaml()
    finally:
        win.close()


def test_the_config_editor_finds_program_txt_where_the_loader_reads(qt_app, tmp_path):
    member = referencing_member(tmp_path)
    (tmp_path / "runs" / "R1" / "data" / "Program.txt").write_text(
        (Path(__file__).parent / "fixtures" / "Program.txt").read_text(encoding="utf-8"),
        encoding="utf-8")
    win = _editor(member)
    try:
        assert win._opto_rows_apply()
    finally:
        win.close()
