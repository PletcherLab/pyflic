"""Tests for two-level scripting: separate registries, one bridge (ADR-0006)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pyflic.base.project import Project
from pyflic.base.script_editor import project_actions
from pyflic.base.script_editor.project_runner import (
    run_project_script,
    validate_project_script,
)

DESIGN = {"global": {"experiment_type": "Hedonic",
                     "well_names": {"A": "S5", "B": "S5Y5"},
                     "params": {"feeding_threshold": 20}}}


def _make_project(tmp_path: Path, **extra) -> Project:
    root = tmp_path / "proj"
    payload = {"name": "proj", "design": DESIGN}
    payload.update(extra)
    (root).mkdir(parents=True)
    (root / "project.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")
    for name in ("rep1", "rep2"):
        (root / name / "data").mkdir(parents=True)
        (root / name / "flic_config.yaml").write_text(
            yaml.safe_dump({"dfms": [{"id": 1, "chambers": {1: "Ctrl"}}]}),
            encoding="utf-8")
    return Project(root)


# ---------------------------------------------------------------------------
# Registry separation
# ---------------------------------------------------------------------------

def test_the_two_registries_do_not_overlap():
    from pyflic.base.script_editor.actions import ACTIONS

    experiment_names = {a.action for a in ACTIONS}
    project_names = set(project_actions.PROJECT_ACTIONS_BY_NAME)
    assert not (experiment_names & project_names)


def test_an_experiment_action_in_a_project_script_is_rejected_by_name(tmp_path: Path):
    project = _make_project(tmp_path)
    problems = validate_project_script(
        project, {"name": "x", "steps": [{"action": "basic_analysis"}]})
    assert any("experiment-level action" in p for p in problems)
    assert any("run_in_experiments" in p for p in problems)


def test_an_unknown_action_is_reported(tmp_path: Path):
    project = _make_project(tmp_path)
    problems = validate_project_script(
        project, {"name": "x", "steps": [{"action": "teleport"}]})
    assert any("unknown project action" in p for p in problems)


# ---------------------------------------------------------------------------
# The bridge
# ---------------------------------------------------------------------------

def test_run_in_experiments_needs_a_script_name(tmp_path: Path):
    project = _make_project(tmp_path)
    problems = validate_project_script(
        project, {"name": "x", "steps": [{"action": "run_in_experiments"}]})
    assert any("needs a 'script' name" in p for p in problems)


def test_an_unresolvable_script_name_is_caught_before_running(tmp_path: Path):
    project = _make_project(tmp_path)
    problems = validate_project_script(
        project,
        {"name": "x", "steps": [{"action": "run_in_experiments",
                                 "script": "nope"}]})
    assert any("no Experiment Script named 'nope'" in p for p in problems)


def test_a_central_experiment_script_resolves(tmp_path: Path):
    project = _make_project(
        tmp_path,
        experiment_scripts=[{"name": "standard",
                             "steps": [{"action": "load"}]}])
    assert validate_project_script(
        project,
        {"name": "x", "steps": [{"action": "run_in_experiments",
                                 "script": "standard"}]}) == []


def test_a_replicates_own_script_resolves_as_the_fallback(tmp_path: Path):
    project = _make_project(tmp_path)
    config = project.experiment_dir("rep1") + "/flic_config.yaml"
    payload = yaml.safe_load(Path(config).read_text())
    payload["scripts"] = [{"name": "local", "steps": [{"action": "load"}]}]
    Path(config).write_text(yaml.safe_dump(payload), encoding="utf-8")
    project = Project(project.project_directory)
    assert validate_project_script(
        project,
        {"name": "x", "steps": [{"action": "run_in_experiments",
                                 "script": "local"}]}) == []


def test_an_unknown_replicate_in_only_is_caught(tmp_path: Path):
    project = _make_project(
        tmp_path,
        experiment_scripts=[{"name": "standard",
                             "steps": [{"action": "load"}]}])
    problems = validate_project_script(
        project,
        {"name": "x", "steps": [{"action": "run_in_experiments",
                                 "script": "standard",
                                 "only": ["rep1", "ghost"]}]})
    assert any("'ghost'" in p for p in problems)


def test_running_an_invalid_script_raises_before_any_step(tmp_path: Path):
    project = _make_project(tmp_path)
    with pytest.raises(ValueError, match=r"is not runnable"):
        run_project_script(project,
                           {"name": "x", "steps": [{"action": "teleport"}]},
                           log=lambda _m: None)


# ---------------------------------------------------------------------------
# Built-in pipelines
# ---------------------------------------------------------------------------

def test_report_pipeline_does_not_gate_on_validate_design():
    """It must not fail Projects mid-migration — that is why a Batch Run
    prefers it over the Standard Pipeline."""
    steps = [s["action"] for s in
             project_actions.builtin_project_script("Report Pipeline")["steps"]]
    assert "validate_design" not in steps
    assert steps == ["project_report", "render_publication_figures"]


def test_standard_pipeline_validates_first():
    steps = [s["action"] for s in
             project_actions.builtin_project_script("Standard Pipeline")["steps"]]
    assert steps[0] == "validate_design"


def test_the_seeded_project_script_is_named_batch_and_matches_report_pipeline():
    default = project_actions.default_project_script()
    report = project_actions.builtin_project_script("Report Pipeline")
    assert default["name"] == "batch"
    assert default["steps"] == report["steps"]


def test_every_builtin_pipeline_is_runnable_as_written(tmp_path: Path):
    project = _make_project(tmp_path)
    for name in project_actions.BUILTIN_PROJECT_SCRIPTS:
        script = project_actions.builtin_project_script(name)
        assert validate_project_script(project, script) == [], name
