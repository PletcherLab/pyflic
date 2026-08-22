from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pyflic import load_experiment_yaml


def _make_project(tmp_path: Path, yaml_text: str) -> Path:
    """Create a minimal project directory containing flic_config.yaml + data/."""
    proj = tmp_path / "project"
    (proj / "data").mkdir(parents=True)
    (proj / "flic_config.yaml").write_text(yaml_text)
    return proj


def _write_minimal_dfm_csv(experiment_dir: Path, dfm_id: int, n_samples: int = 1200) -> None:
    """Write a minimal DFM CSV (Sample, Seconds, W1..W12) under experiment_dir/data."""
    cols: dict[str, list[float] | list[int]] = {
        "Sample": list(range(1, n_samples + 1)),
        "Seconds": [i / 5.0 for i in range(n_samples)],
    }
    for w in range(1, 13):
        cols[f"W{w}"] = [0] * n_samples
    pd.DataFrame(cols).to_csv(
        experiment_dir / "data" / f"DFM{dfm_id}_test.csv", index=False
    )


def test_yaml_requires_params_somewhere(tmp_path: Path):
    proj = _make_project(
        tmp_path,
        """
global: {}
dfms:
  1:
    chambers: {1: A}
""".lstrip(),
    )
    with pytest.raises(ValueError, match=r"must define a `params` section"):
        load_experiment_yaml(proj)


def test_chamber_size_is_derived_from_the_chamber_layout(tmp_path: Path):
    """chamber_size is owned by the Experiment Type via the Chamber Layout
    (ADR-0007), so a config that omits it loads instead of failing."""
    proj = _make_project(
        tmp_path,
        """
global:
  chamber_layout: single_well
  params: {feeding_threshold: 10}
dfms:
  1:
    params: {pi_multiplier: 1}
    chambers: {1: A}
""".lstrip(),
    )
    _write_minimal_dfm_csv(proj, 1)
    exp = load_experiment_yaml(proj, eager=False, use_disk_cache=False)
    assert exp.chamber_layout == "single_well"
    assert exp.dfms[1].params.chamber_size == 1


def test_typed_config_must_not_state_chamber_size(tmp_path: Path):
    """A typed config that states a key the type owns is rejected — the two
    can no longer disagree because the config no longer gets a vote."""
    proj = _make_project(
        tmp_path,
        """
global:
  experiment_type: Hedonic
  well_names: {A: S5, B: S5Y5}
  params: {feeding_threshold: 10, chamber_size: 1}
dfms:
  1:
    chambers: {1: A}
""".lstrip(),
    )
    _write_minimal_dfm_csv(proj, 1)
    with pytest.raises(ValueError, match=r"chamber_size.*owned by experiment_type"):
        load_experiment_yaml(proj, eager=False, use_disk_cache=False)


def test_retired_experiment_type_names_the_migration(tmp_path: Path):
    """`experiment_type: two_well` was always a layout, not an assay. It must
    fail loudly with the migration rather than silently become Custom."""
    proj = _make_project(
        tmp_path,
        """
global:
  experiment_type: two_well
  params: {feeding_threshold: 10}
dfms:
  1:
    chambers: {1: A}
""".lstrip(),
    )
    _write_minimal_dfm_csv(proj, 1)
    with pytest.raises(ValueError, match=r"chamber_layout: two_well"):
        load_experiment_yaml(proj, eager=False, use_disk_cache=False)


def test_yaml_global_params_applied_and_dfm_overrides(tmp_path: Path):
    proj = _make_project(
        tmp_path,
        """
global:
  params:
    chamber_size: 2
    feeding_threshold: 10
dfms:
  1:
    params:
      pi_direction: right
    chambers:
      1: A
      2: B
""".lstrip(),
    )
    _write_minimal_dfm_csv(proj, 1)
    exp = load_experiment_yaml(proj, parallel=False)
    assert 1 in exp.design.dfms
    dfm1 = exp.design.dfms[1]
    assert dfm1.params.chamber_size == 2
    assert dfm1.params.feeding_threshold == 10
    assert dfm1.params.pi_direction == "right"


def test_yaml_pi_direction_validation(tmp_path: Path):
    proj = _make_project(
        tmp_path,
        """
global:
  params:
    chamber_size: 2
dfms:
  1:
    params: {pi_direction: sideways}
    chambers: {1: A}
""".lstrip(),
    )
    with pytest.raises(ValueError, match=r"pi_direction must be 'left' or 'right'"):
        load_experiment_yaml(proj)


def test_yaml_pi_multiplier_backcompat_numeric(tmp_path: Path):
    proj = _make_project(
        tmp_path,
        """
global:
  params:
    chamber_size: 2
dfms:
  1:
    params: {pi_multiplier: 2}
    chambers: {1: A}
""".lstrip(),
    )
    _write_minimal_dfm_csv(proj, 1)
    exp = load_experiment_yaml(proj, parallel=False)
    assert exp.design.dfms[1].params.pi_direction == "right"


def test_yaml_missing_config_file_raises(tmp_path: Path):
    proj = tmp_path / "empty_project"
    proj.mkdir()
    with pytest.raises(FileNotFoundError, match=r"flic_config\.yaml not found"):
        load_experiment_yaml(proj)
