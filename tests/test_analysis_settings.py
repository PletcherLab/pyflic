"""The settings an analysis ran with, and the stale rule that compares them
with the Design and the config as they stand (ADR-0022)."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pyflic.base import analysis_settings
from pyflic.base.project import Project
from pyflic.base.yaml_config import load_experiment_yaml

PARAMS = {"feeding_threshold": 5, "feeding_minimum": 5, "tasting_minimum": 1,
          "tasting_maximum": 4, "feeding_event_link_gap": 2, "samples_per_second": 5,
          "correct_for_dual_feeding": False}
CHAMBERS = {1: "Ctrl", 2: "Ctrl", 3: "Exp", 4: "Exp", 5: "Ctrl", 6: "Exp"}
SLICES = {"early": {"start": 0, "end": 5}}


def _write_csv(directory: Path, *, minutes: float = 10.0) -> None:
    """Five samples a second with a feeding bout on every well every minute."""
    n = int(minutes * 60 * 5)
    rng = np.random.default_rng(1)
    cols: dict[str, object] = {"Sample": np.arange(1, n + 1),
                               "Seconds": np.arange(n) / 5.0}
    for w in range(1, 13):
        sig = rng.normal(0.0, 0.3, size=n)
        for k in range(int(minutes)):
            start = 300 * k + 40 + (w * 7) % 30
            sig[start:start + 8] += 10.0 + rng.normal(0, 0.5, size=8)
        cols[f"W{w}"] = np.round(sig, 3)
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cols).to_csv(directory / "DFM1_0.csv", index=False)


def _design(**extra) -> dict:
    g = {"transform_licks": False, "params": dict(PARAMS)}
    g.update(extra)
    return g


def _project(tmp_path: Path, **extra) -> Path:
    root = tmp_path / "proj"
    root.mkdir()
    _write_meta(root, {"name": "proj", "design": {"global": _design(**extra)}})
    for name in ("rep1", "rep2"):
        _write_csv(root / name / "data")
        _write_config(root / name, {"dfms": [{"id": 1, "chambers": dict(CHAMBERS)}]})
    return root


def _write_meta(root: Path, meta: dict) -> None:
    (root / "project.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))


def _read_meta(root: Path) -> dict:
    return yaml.safe_load((root / "project.yaml").read_text())


def _write_config(member: Path, cfg: dict) -> None:
    (member / "flic_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def _later(path: Path, than: Path) -> None:
    """Give *path* a modification time after *than*'s, whatever the clock's
    resolution."""
    stamp = os.path.getmtime(than) + 5
    os.utime(path, (stamp, stamp))


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _analysed(tmp_path: Path, **extra) -> Path:
    root = _project(tmp_path, **extra)
    assert _quiet(Project(root).run_all, make_reports=False) == []
    return root


# ---------------------------------------------------------------------------
# The comparison form
# ---------------------------------------------------------------------------

def test_the_comparison_form_ignores_how_a_value_is_written():
    as_list = {"dfms": [{"id": 1, "chambers": {1: "A"}}]}
    as_mapping = {"dfms": {1: {"chambers": {1: "A"}}}}
    assert (analysis_settings.settings({"params": {"feeding_threshold": 20}}, as_list)
            == analysis_settings.settings({"params": {"feeding_threshold": 20.0}},
                                          as_mapping))


def test_the_comparison_form_leaves_out_what_changes_no_number():
    base = analysis_settings.settings(_design(), {"dfms": []})
    noisy = analysis_settings.settings(
        _design(omit={"plots": ["dot_licks"]}, slices=SLICES, slice_tolerance_min=2),
        {"dfms": [], "scripts": [{"name": "mine", "steps": []}]})
    assert base == noisy


def test_the_members_own_global_wins_as_it_does_at_load():
    out = analysis_settings.settings({"transform_licks": True},
                                     {"global": {"transform_licks": False}, "dfms": []})
    assert out["global"]["transform_licks"] is False


def test_differences_name_the_setting_and_both_values():
    before = analysis_settings.settings(_design(), {"dfms": [{"id": 2, "chambers": {3: "Exp"}}]})
    params = dict(PARAMS, feeding_threshold=8)
    after = analysis_settings.settings(_design(params=params),
                                       {"dfms": [{"id": 2, "chambers": {3: "Ctrl"}}]})
    assert analysis_settings.differences(before, after) == [
        "params.feeding_threshold changed: 5 → 8",
        "DFM 2 chambers.3 changed: Exp → Ctrl",
    ]
    assert analysis_settings.differences(before, before) == []


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

def test_an_analysis_records_the_settings_it_ran_with(tmp_path):
    root = _analysed(tmp_path)
    record = root / "rep1" / "analysis" / analysis_settings.RECORD_FILENAME
    payload = json.loads(record.read_text(encoding="utf-8"))
    assert payload["settings"]["global"]["params"]["feeding_threshold"] == 5.0
    assert payload["settings"]["dfms"]["1"]["chambers"]["3"] == "Exp"


def test_a_standalone_experiment_records_its_settings_too(tmp_path):
    root = tmp_path / "exp"
    _write_csv(root / "data")
    _write_config(root, {"global": _design(),
                         "dfms": [{"id": 1, "chambers": dict(CHAMBERS)}]})
    exp = _quiet(load_experiment_yaml, root)
    _quiet(exp.write_feeding_summary)
    recorded = analysis_settings.read(root / "analysis")
    assert recorded == analysis_settings.current(None, str(root / "flic_config.yaml"))


def test_a_summary_written_elsewhere_leaves_no_record(tmp_path):
    root = tmp_path / "exp"
    _write_csv(root / "data")
    _write_config(root, {"global": _design(),
                         "dfms": [{"id": 1, "chambers": dict(CHAMBERS)}]})
    exp = _quiet(load_experiment_yaml, root)
    _quiet(exp.write_feeding_summary, path=root / "analysis" / "feeding_summary_0_5.csv")
    assert analysis_settings.read(root / "analysis") is None


# ---------------------------------------------------------------------------
# The stale rule
# ---------------------------------------------------------------------------

def test_a_fresh_analysis_is_not_stale(tmp_path):
    status = Project(_analysed(tmp_path)).member_status("rep1")
    assert status["analyzed"] is True
    assert status["stale"] is False and status["stale_reasons"] == []


def test_a_changed_design_parameter_reads_re_run_needed_for_every_member(tmp_path):
    root = _analysed(tmp_path)
    meta = _read_meta(root)
    meta["design"]["global"]["params"]["feeding_threshold"] = 7
    _write_meta(root, meta)
    project = Project(root)
    for name in ("rep1", "rep2"):
        status = project.member_status(name)
        assert status["analyzed"] is True and status["stale"] is True
        assert status["stale_reasons"] == ["params.feeding_threshold changed: 5 → 7"]


def test_the_status_reads_the_files_as_they_stand_not_as_the_project_loaded_them(tmp_path):
    """The Hub keeps one Project open while its design and configs are edited:
    a refresh must see the edit without reloading it."""
    root = _analysed(tmp_path)
    project = Project(root)
    meta = _read_meta(root)
    meta["design"]["global"]["transform_licks"] = True
    _write_meta(root, meta)
    assert project.member_status("rep1")["stale_reasons"] == [
        "transform_licks changed: False → True"]


def test_a_changed_member_config_stales_only_that_member(tmp_path):
    root = _analysed(tmp_path)
    chambers = {**CHAMBERS, 3: "Ctrl"}
    _write_config(root / "rep2", {"dfms": [{"id": 1, "chambers": chambers}]})
    project = Project(root)
    assert project.member_status("rep1")["stale"] is False
    assert project.member_status("rep2")["stale_reasons"] == [
        "DFM 1 chambers.3 changed: Exp → Ctrl"]


def test_a_save_that_changes_no_setting_stays_analyzed(tmp_path):
    """A Selection tick, a Project Script, notes, or the same value written
    again rewrite the file and change no number in the results."""
    root = _analysed(tmp_path)
    meta = _read_meta(root)
    meta["notes"] = "rep2 had a leaky well"
    meta["scripts"] = [{"name": "mine", "steps": [{"action": "project_report"}]}]
    meta["design"]["global"]["omit"] = {"plots": ["dot_licks"]}
    meta["design"]["global"]["params"]["feeding_threshold"] = 5.0
    _write_meta(root, meta)
    _write_config(root / "rep1", {"dfms": {1: {"chambers": dict(CHAMBERS)}},
                                  "scripts": [{"name": "x", "steps": []}]})
    project = Project(root)
    for name in ("rep1", "rep2"):
        assert project.member_status(name)["stale"] is False


def test_a_changed_slice_leaves_the_whole_recording_analyzed(tmp_path):
    root = _analysed(tmp_path, slices=SLICES)
    meta = _read_meta(root)
    meta["design"]["global"]["slices"]["early"]["end"] = 6
    _write_meta(root, meta)
    project = Project(root)
    assert project.member_status("rep1")["stale"] is False
    assert project.for_slice("early").member_status("rep1")["stale_reasons"] == [
        "Slice early has changed since it was run"]


def test_a_design_change_stales_each_slice_as_well(tmp_path):
    root = _analysed(tmp_path, slices=SLICES)
    meta = _read_meta(root)
    meta["design"]["global"]["params"]["feeding_minimum"] = 6
    _write_meta(root, meta)
    view = Project(root).for_slice("early")
    assert view.member_status("rep1")["stale_reasons"] == [
        "params.feeding_minimum changed: 5 → 6"]


def test_exclusions_declared_since_still_read_stale_with_their_reason(tmp_path):
    root = _analysed(tmp_path)
    declaration = root / "rep1" / "remove_chambers.csv"
    declaration.write_text("group,dfm_id,chamber,note\ngeneral,1,2,\n", encoding="utf-8")
    _later(declaration, root / "rep1" / "analysis" / "feeding_summary.csv")
    assert Project(root).member_status("rep1")["stale_reasons"] == [
        "remove_chambers.csv was saved after this analysis"]


def test_an_analysis_from_before_records_is_judged_by_file_times(tmp_path):
    """Results written before pyflic recorded settings: a config or
    project.yaml saved since reads stale, as the exclusions always have."""
    root = _analysed(tmp_path)
    for name in ("rep1", "rep2"):
        (root / name / "analysis" / analysis_settings.RECORD_FILENAME).unlink()
    summary = root / "rep1" / "analysis" / "feeding_summary.csv"
    for name in ("rep1", "rep2"):
        stamp = os.path.getmtime(root / name / "analysis" / "feeding_summary.csv") - 5
        os.utime(root / name / "flic_config.yaml", (stamp, stamp))
    stamp = os.path.getmtime(summary) - 5
    os.utime(root / "project.yaml", (stamp, stamp))
    project = Project(root)
    assert project.member_status("rep1")["stale"] is False

    _later(root / "project.yaml", summary)
    reasons = Project(root).member_status("rep1")["stale_reasons"]
    assert len(reasons) == 1 and reasons[0].startswith("project.yaml was saved after")


def test_skip_analyzed_runs_the_stale_members_again(tmp_path):
    root = _analysed(tmp_path)
    _write_config(root / "rep2", {"dfms": [{"id": 1, "chambers": {**CHAMBERS, 3: "Ctrl"}}]})
    lines: list[str] = []
    assert _quiet(Project(root).run_all, make_reports=False, skip_analyzed=True,
                  log=lines.append) == []
    assert "[rep1] already analyzed — skipped" in lines
    assert any(line.startswith("[rep2] re-run needed: DFM 1 chambers.3") for line in lines)
    assert "[rep2] running analysis..." in lines
    assert Project(root).member_status("rep2")["stale"] is False
