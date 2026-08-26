"""Tests for the Project / Batch layer (ADR-0005 .. ADR-0009)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from pyflic.base import batch as batch_mod
from pyflic.base import windowing
from pyflic.base.project import Project, create_project_file, dfm_ids_in_data

DESIGN = {
    "global": {
        "experiment_type": "Hedonic",
        "transform_licks": False,
        "facet_cutoffs": [60],
        "params": {"feeding_threshold": 20, "samples_per_second": 5},
        "well_names": {"A": "S5", "B": "S5Y5"},
        "experimental_design_factors": {"TreatmentNew": ["Ctrl", "Exp"]},
    }
}

REPLICATE = {
    "dfms": [
        {"id": 1, "params": {"pi_direction": "left"},
         "chambers": {1: "Ctrl", 2: "Exp"}},
    ]
}


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _make_project(tmp_path: Path, members=("rep1", "rep2"),
                  design: dict | None = None) -> Path:
    root = tmp_path / "proj"
    _write(root / "project.yaml",
           {"name": "proj", "design": design if design is not None else DESIGN})
    for name in members:
        _write(root / name / "flic_config.yaml", REPLICATE)
        (root / name / "data").mkdir(parents=True, exist_ok=True)
        (root / name / "data" / "DFM1_0.csv").write_text("Sample,Seconds\n")
    return root


# ---------------------------------------------------------------------------
# Design authority and inheritance
# ---------------------------------------------------------------------------

def test_member_inherits_global_from_the_design(tmp_path: Path):
    project = Project(_make_project(tmp_path))
    assert project.experiment_names == ["rep1", "rep2"]
    assert project.experiment_type.name == "Hedonic"
    assert project.chamber_layout == "two_well"
    # The member's own file states no global: at all.
    assert project._own_global("rep1") == {}
    assert project.resolved_global("rep1")["well_names"] == {"A": "S5", "B": "S5Y5"}


def test_matching_global_in_a_member_is_accepted(tmp_path: Path):
    root = _make_project(tmp_path)
    payload = dict(REPLICATE)
    payload["global"] = {"transform_licks": False}
    _write(root / "rep1" / "flic_config.yaml", payload)
    project = Project(root)          # must not raise
    assert project.experiment_names == ["rep1", "rep2"]


def test_deviating_global_is_a_load_error(tmp_path: Path):
    root = _make_project(tmp_path)
    payload = dict(REPLICATE)
    payload["global"] = {"transform_licks": True}
    _write(root / "rep1" / "flic_config.yaml", payload)
    with pytest.raises(ValueError, match=r"global.transform_licks is True"):
        Project(root)


def test_deviating_params_is_a_load_error(tmp_path: Path):
    root = _make_project(tmp_path)
    payload = dict(REPLICATE)
    payload["global"] = {"params": {"feeding_threshold": 22,
                                    "samples_per_second": 5}}
    _write(root / "rep1" / "flic_config.yaml", payload)
    with pytest.raises(ValueError, match=r"global.params"):
        Project(root)


def test_numeric_spelling_is_not_a_deviation(tmp_path: Path):
    """20 and 20.0 are the same threshold; an error there would be indefensible."""
    root = _make_project(tmp_path)
    payload = dict(REPLICATE)
    payload["global"] = {"params": {"feeding_threshold": 20.0,
                                    "samples_per_second": 5}}
    _write(root / "rep1" / "flic_config.yaml", payload)
    Project(root)                    # must not raise


def test_unknown_global_key_in_a_member_is_rejected(tmp_path: Path):
    root = _make_project(tmp_path)
    payload = dict(REPLICATE)
    payload["global"] = {"nonsense": 1}
    _write(root / "rep1" / "flic_config.yaml", payload)
    with pytest.raises(ValueError, match=r"not part of the project design"):
        Project(root)


def test_physical_dfm_override_is_allowed(tmp_path: Path):
    root = _make_project(tmp_path)
    payload = {"dfms": [{"id": 1, "params": {"pi_direction": "right"},
                         "chambers": {1: "Ctrl"}}]}
    _write(root / "rep1" / "flic_config.yaml", payload)
    Project(root)                    # pi_direction describes hardware


def test_analysis_dfm_override_is_rejected_inside_a_project(tmp_path: Path):
    root = _make_project(tmp_path)
    payload = {"dfms": [{"id": 1, "params": {"feeding_threshold": 33},
                         "chambers": {1: "Ctrl"}}]}
    _write(root / "rep1" / "flic_config.yaml", payload)
    with pytest.raises(ValueError, match=r"only \['chamber_sets', 'pi_direction'\]"):
        Project(root)


def test_project_without_a_design_falls_back_to_agreement(tmp_path: Path):
    root = _make_project(tmp_path, design={})
    payload = dict(REPLICATE)
    payload["global"] = {"transform_licks": True}
    _write(root / "rep1" / "flic_config.yaml", payload)
    with pytest.raises(ValueError, match=r"add a design: section"):
        Project(root)


# ---------------------------------------------------------------------------
# Facets
# ---------------------------------------------------------------------------

def test_facet_windows_are_half_open_and_tile_the_recording():
    windows = windowing.facet_windows([10, 70])
    assert windows == [(0, 10), (10, 70), (70, float("inf"))]
    # Each window's end is the next window's start: no sample is in two facets.
    assert [w[1] for w in windows[:-1]] == [w[0] for w in windows[1:]]


def test_facet_range_round_trips():
    for window in windowing.facet_windows([10, 70]):
        assert windowing.parse_range(windowing.format_range(window)) == \
            tuple(float(v) for v in window)


def test_open_ended_facet_maps_to_the_whole_recording_sentinel():
    """`inf` would be rejected by the range validators; 0 means 'to the end'."""
    assert windowing.as_range_minutes((70, float("inf"))) == (70.0, 0.0)


def test_shared_windows_come_from_the_design(tmp_path: Path):
    project = Project(_make_project(tmp_path))
    windows, labels = project.shared_windows()
    assert windows == [(0, 60), (60, float("inf"))]
    assert labels == ["0-60 min", "60+ min"]


# ---------------------------------------------------------------------------
# Scaffolding
# ---------------------------------------------------------------------------

def test_dfm_ids_are_discovered_from_data_filenames(tmp_path: Path):
    data = tmp_path / "rep" / "data"
    data.mkdir(parents=True)
    for name in ("DFM1_0.csv", "DFM1_1.csv", "DFM4_0.csv", "notes.txt"):
        (data / name).write_text("")
    assert dfm_ids_in_data(tmp_path / "rep") == [1, 4]


def test_scaffold_copies_the_layout_and_reconciles_against_data(tmp_path: Path):
    root = _make_project(tmp_path, members=("rep1",))
    # A new folder holding DFMs 1 and 3 — the copied layout only knows DFM 1.
    data = root / "rep2" / "data"
    data.mkdir(parents=True)
    (data / "DFM1_0.csv").write_text("")
    (data / "DFM3_0.csv").write_text("")

    project = Project(root)
    assert project.unconfigured_dirs() == ["rep2"]
    path, notes = project.scaffold_member("rep2")

    written = yaml.safe_load(Path(path).read_text())
    assert [node["id"] for node in written["dfms"]] == [1, 3]
    # DFM 3 came from the data, so its chambers are unassigned.
    assert set(written["dfms"][1]["chambers"].values()) == {""}
    assert "global" not in written          # inherits the design
    assert any("DFM 3" in note for note in notes)


def test_scaffold_flags_a_dfm_with_no_data_rather_than_dropping_it(tmp_path: Path):
    root = _make_project(tmp_path, members=("rep1",))
    data = root / "rep2" / "data"
    data.mkdir(parents=True)
    (data / "DFM9_0.csv").write_text("")

    project = Project(root)
    _path, notes = project.scaffold_member("rep2")
    assert any("DFM 1" in n and "absent from data" in n for n in notes)


def test_scaffold_never_overwrites_an_existing_config(tmp_path: Path):
    project = Project(_make_project(tmp_path))
    with pytest.raises(FileExistsError):
        project.scaffold_member("rep1")


# ---------------------------------------------------------------------------
# Combined Analysis
# ---------------------------------------------------------------------------

def _fake_summary(experiment: str, treatment_values: dict) -> pd.DataFrame:
    rows = []
    for treatment, values in treatment_values.items():
        for index, value in enumerate(values):
            rows.append({"Treatment": treatment, "DFM": 1 + index % 2,
                         "Chamber": index + 1, "PI": value,
                         "LicksA": 100 + value, "LicksB": 50})
    return pd.DataFrame(rows)


def _save_analysis(root: Path, name: str, frame: pd.DataFrame) -> None:
    out = root / name / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "feeding_summary.csv", index=False)


def test_combined_frames_stack_with_an_experiment_column(tmp_path: Path):
    root = _make_project(tmp_path)
    _save_analysis(root, "rep1", _fake_summary("rep1", {"Ctrl": [0.1, 0.2],
                                                        "Exp": [0.6, 0.7]}))
    _save_analysis(root, "rep2", _fake_summary("rep2", {"Ctrl": [0.15, 0.25],
                                                        "Exp": [0.65, 0.75]}))
    project = Project(root)
    summary, facet, missing = project.combined_frames()
    assert missing == []
    assert list(summary.columns)[0] == "Experiment"
    assert set(summary["Experiment"]) == {"rep1", "rep2"}
    assert len(summary) == 8


def test_a_member_without_a_saved_analysis_is_reported_not_analyzed(tmp_path: Path):
    root = _make_project(tmp_path)
    _save_analysis(root, "rep1", _fake_summary("rep1", {"Ctrl": [0.1],
                                                        "Exp": [0.6]}))
    project = Project(root)
    summary, _facet, missing = project.combined_frames()
    assert missing == ["rep2"]
    assert set(summary["Experiment"]) == {"rep1"}


def test_build_combined_analysis_refuses_when_nothing_is_analyzed(tmp_path: Path):
    project = Project(_make_project(tmp_path))
    with pytest.raises(ValueError, match=r"No member has a saved analysis"):
        project.build_combined_analysis()


def test_stats_name_the_two_models(tmp_path: Path):
    root = _make_project(tmp_path)
    _save_analysis(root, "rep1", _fake_summary("rep1", {"Ctrl": [0.1, 0.2, 0.15],
                                                        "Exp": [0.6, 0.7, 0.65]}))
    _save_analysis(root, "rep2", _fake_summary("rep2", {"Ctrl": [0.12, 0.22, 0.17],
                                                        "Exp": [0.62, 0.72, 0.67]}))
    project = Project(root)
    result = project.build_combined_analysis()
    stats = (root / "analysis" / "proj_Stats.txt").read_text()
    assert "DFM nested within Experiment" in stats
    assert "p_pooled" in stats and "p_mixed" in stats
    assert any(p.endswith("proj_Excluded.csv") for p in result["written"])


def test_mixed_model_needs_more_than_one_member(tmp_path: Path):
    """With one member there is no between-member variance to model, so
    the mixed column must be absent rather than a duplicate of the pooled one."""
    root = _make_project(tmp_path, members=("rep1",))
    _save_analysis(root, "rep1", _fake_summary("rep1", {"Ctrl": [0.1, 0.2, 0.15],
                                                        "Exp": [0.6, 0.7, 0.65]}))
    project = Project(root)
    summary, facet, _missing = project.combined_frames()
    rows = project.comparison_rows(summary, facet)
    assert rows and all(row["p_mixed"] is None for row in rows)


# ---------------------------------------------------------------------------
# Batch (ADR-0006)
# ---------------------------------------------------------------------------

def test_batch_discovery_is_recursive_and_keys_by_relative_path(tmp_path: Path):
    """ADR-0009: Projects sit at any depth, and grouping folders are
    transparent.  A key is the Project's path relative to the Batch root, so a
    top-level Project keeps its bare name."""
    root = tmp_path / "batch"
    _make_project(root, members=("rep1",))                      # batch/proj
    _make_project(root / "Sept2026" / "deeper", members=("rep1",))
    assert batch_mod.batch_project_names(root) == [
        "Sept2026/deeper/proj", "proj"]
    assert batch_mod.is_batch_dir(root)


def test_batch_prunes_at_a_project_so_nothing_runs_twice(tmp_path: Path):
    """A Project's subdirectories are its Members by definition, so an
    archived copy carrying its own project.yaml inside one cannot become a
    second target."""
    root = tmp_path / "batch"
    proj = _make_project(root, members=("rep1",))
    _make_project(proj / "archive", members=("rep1",))
    assert batch_mod.batch_project_names(root) == ["proj"]


def test_a_stray_project_yaml_does_not_hide_the_projects_beneath_it(tmp_path: Path):
    """The mistake recursion exists to tolerate: a marker dropped at a
    grouping level used to stop the walk dead."""
    root = tmp_path / "batch"
    _write(root / "Archive" / "project.yaml", {"name": "stray"})
    _make_project(root / "Archive" / "2025", members=("rep1",))
    found = batch_mod.discover(root)
    assert [p.key for p in found["projects"]] == ["Archive/2025/proj"]
    assert any(key == "Archive" for key, _why in found["skipped"])


def test_a_project_is_never_also_a_batch(tmp_path: Path):
    root = _make_project(tmp_path, members=("rep1",))
    assert not batch_mod.is_batch_dir(root)


def test_a_directory_with_no_project_children_is_not_a_batch(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    assert not batch_mod.is_batch_dir(tmp_path / "empty")


def test_a_blocked_member_does_not_block_its_project(tmp_path: Path):
    """Blocked is a property of the Member, never of the Project: four healthy
    members and one blocked one runs the four (ADR-0009)."""
    root = tmp_path / "batch"
    proj = _make_project(root, members=("rep1", "rep2"))
    loose = proj / "rep3"
    loose.mkdir()
    (loose / "DFM1_0.csv").write_text("Sample,Seconds\n")
    found = batch_mod.discover(root)
    item = found["projects"][0]
    assert item.runnable and len(item.usable) == 2
    assert [m.name for m in item.blocked] == ["rep3"]
    assert item.summary() == "2/3 members, 1 blocked"


def test_a_project_with_nothing_usable_starts_unrunnable(tmp_path: Path):
    root = tmp_path / "batch"
    _write(root / "proj" / "project.yaml", {"name": "proj", "design": DESIGN})
    loose = root / "proj" / "rep1"
    loose.mkdir(parents=True)
    (loose / "DFM1_0.csv").write_text("Sample,Seconds\n")
    item = batch_mod.discover(root)["projects"][0]
    assert not item.runnable


def test_batch_default_designation_is_each_projects_own_script(tmp_path: Path):
    """No designation means every Project runs its OWN default script — there
    is no silent built-in substitution for a Project that has none."""
    root = tmp_path / "batch"
    _make_project(root, members=("rep1",))
    assert batch_mod.Batch(root).script_name is None
    project = Project(root / "proj")
    script, source = batch_mod.resolve_designated_script(None, [], project)
    assert script is None and source == ""


def test_batch_yaml_designates_the_script(tmp_path: Path):
    root = tmp_path / "batch"
    _make_project(root, members=("rep1",))
    _write(root / "batch.yaml", {"script": "Report Pipeline"})
    assert batch_mod.Batch(root).script_name == "Report Pipeline"


def test_saving_the_default_designation_never_creates_batch_yaml(tmp_path: Path):
    """The lazy-marker rule: a Batch has no authority to declare, so the file
    appears only once there is something to say."""
    root = tmp_path / "batch"
    _make_project(root, members=("rep1",))
    batch_mod.save_batch_designation(root, None)
    assert not (root / "batch.yaml").exists()
    batch_mod.save_batch_designation(root, "Report Pipeline")
    assert batch_mod.load_batch_file(root)["script"] == "Report Pipeline"
    batch_mod.save_batch_designation(root, None)
    assert batch_mod.load_batch_file(root)["script"] is None


def test_a_project_key_cannot_escape_its_batch(tmp_path: Path):
    """The public key→directory resolver, and Exclusion Sheet cells are typed
    by hand."""
    root = tmp_path / "batch"
    _make_project(root, members=("rep1",))
    for bad in ("../etc", "/etc", "proj/../../etc"):
        with pytest.raises(ValueError):
            batch_mod.project_directory(root, bad)


def test_new_project_yaml_is_seeded_with_a_batch_script(tmp_path: Path):
    root = tmp_path / "fresh"
    root.mkdir()
    create_project_file(root, name="fresh")
    payload = yaml.safe_load((root / "project.yaml").read_text())
    assert [s["name"] for s in payload["scripts"]] == ["batch"]


def test_an_existing_scripts_block_is_never_reseeded(tmp_path: Path):
    """An empty list is a deliberate deletion; re-seeding would undo an edit."""
    root = tmp_path / "fresh"
    root.mkdir()
    _write(root / "project.yaml", {"name": "fresh", "scripts": []})
    create_project_file(root, name="fresh")
    assert yaml.safe_load((root / "project.yaml").read_text())["scripts"] == []
