"""Tests for the migration checks — the whole crossing, since there is no tool."""

from __future__ import annotations

from pathlib import Path

import yaml

from pyflic.base.migration_lint import check_directory, check_tree


def _experiment(root: Path, config: dict, name: str = "flic_config.yaml") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(yaml.safe_dump(config), encoding="utf-8")
    return root


def _messages(issues) -> str:
    return "\n".join(f"{i.severity}: {i.message} | {i.fix}" for i in issues)


def test_retired_experiment_type_names_the_replacement(tmp_path: Path):
    root = _experiment(tmp_path / "exp", {
        "global": {"experiment_type": "two_well",
                   "params": {"chamber_size": 2}},
        "dfms": []})
    text = _messages(check_directory(root))
    assert "no longer an experiment type" in text
    assert "chamber_layout: two_well" in text


def test_chamber_size_on_an_untyped_config_suggests_the_layout(tmp_path: Path):
    root = _experiment(tmp_path / "exp", {
        "global": {"params": {"chamber_size": 1}}, "dfms": []})
    text = _messages(check_directory(root))
    assert "chamber_layout: single_well" in text


def test_typed_config_stating_an_owned_key_is_an_error(tmp_path: Path):
    root = _experiment(tmp_path / "exp", {
        "global": {"experiment_type": "Hedonic",
                   "well_names": {"A": "S5", "B": "S5Y5"},
                   "params": {"chamber_size": 2}},
        "dfms": []})
    issues = check_directory(root)
    assert any(i.severity == "error" and "chamber_size" in i.message
               for i in issues)


def test_multiple_config_yamls_are_an_error_that_refuses_to_guess(tmp_path: Path):
    root = tmp_path / "exp"
    _experiment(root, {"global": {}, "dfms": []})
    _experiment(root, {"global": {}, "dfms": []}, name="variant_b.yaml")
    issues = check_directory(root)
    text = _messages(issues)
    assert any(i.severity == "error" for i in issues)
    assert "exactly one" in text
    assert "cannot be automated" in text


def test_a_differently_named_single_config_is_an_error(tmp_path: Path):
    root = _experiment(tmp_path / "exp", {"global": {}, "dfms": []},
                       name="my_config.yaml")
    text = _messages(check_directory(root))
    assert "rename it to flic_config.yaml" in text


def test_orphaned_results_directories_are_reported_never_deleted(tmp_path: Path):
    root = _experiment(tmp_path / "exp", {"global": {}, "dfms": []})
    (root / "flic_config_results").mkdir()
    (root / "analysis_0_360").mkdir()
    text = _messages(check_directory(root))
    assert "flic_config_results/" in text
    assert "analysis_0_360/" in text
    assert "never touch it" in text
    # Nothing was removed.
    assert (root / "flic_config_results").is_dir()
    assert (root / "analysis_0_360").is_dir()


def test_windowed_output_points_at_facets(tmp_path: Path):
    root = _experiment(tmp_path / "exp", {"global": {}, "dfms": []})
    (root / "qc_0_360").mkdir()
    assert "facet_cutoffs" in _messages(check_directory(root))


def test_experiment_script_named_batch_is_flagged(tmp_path: Path):
    root = _experiment(tmp_path / "exp", {
        "global": {}, "dfms": [],
        "scripts": [{"name": "batch", "steps": []}]})
    text = _messages(check_directory(root))
    assert "no longer has special meaning" in text


def test_analysis_override_inside_a_project_is_an_error(tmp_path: Path):
    root = tmp_path / "proj"
    (root).mkdir()
    (root / "project.yaml").write_text(yaml.safe_dump(
        {"name": "p", "design": {"global": {"params": {"feeding_threshold": 20}}}}),
        encoding="utf-8")
    _experiment(root / "rep1", {
        "dfms": [{"id": 1, "params": {"feeding_threshold": 33}}]})
    issues = check_tree(root)
    assert any(i.severity == "error" and "feeding_threshold" in i.message
               for i in issues)


def test_physical_override_inside_a_project_is_fine(tmp_path: Path):
    root = tmp_path / "proj"
    (root).mkdir()
    (root / "project.yaml").write_text(yaml.safe_dump(
        {"name": "p", "design": {"global": {"params": {"feeding_threshold": 20}}}}),
        encoding="utf-8")
    _experiment(root / "rep1", {
        "dfms": [{"id": 1, "params": {"pi_direction": "left"}}]})
    assert not any(i.severity == "error" for i in check_tree(root))


def test_a_project_without_a_design_is_warned(tmp_path: Path):
    root = tmp_path / "proj"
    (root).mkdir()
    (root / "project.yaml").write_text(yaml.safe_dump({"name": "p"}),
                                       encoding="utf-8")
    assert "no design.global" in _messages(check_tree(root))


def test_check_tree_descends_a_batch(tmp_path: Path):
    batch = tmp_path / "batch"
    project = batch / "proj"
    project.mkdir(parents=True)
    (project / "project.yaml").write_text(yaml.safe_dump({"name": "p"}),
                                          encoding="utf-8")
    _experiment(project / "rep1", {"dfms": []})
    assert check_tree(batch)          # found the project below
