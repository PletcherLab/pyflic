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


def _write_minimal_dfm_csv(project_dir: Path, dfm_id: int, n_samples: int = 1200) -> None:
    """Write a minimal DFM CSV (Sample, Seconds, W1..W12) under project_dir/data."""
    cols: dict[str, list[float] | list[int]] = {
        "Sample": list(range(1, n_samples + 1)),
        "Seconds": [i / 5.0 for i in range(n_samples)],
    }
    for w in range(1, 13):
        cols[f"W{w}"] = [0] * n_samples
    pd.DataFrame(cols).to_csv(
        project_dir / "data" / f"DFM{dfm_id}_test.csv", index=False
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


def test_yaml_requires_chamber_size(tmp_path: Path):
    proj = _make_project(
        tmp_path,
        """
global:
  params: {feeding_threshold: 10}
dfms:
  1:
    params: {pi_multiplier: 1}
    chambers: {1: A}
""".lstrip(),
    )
    with pytest.raises(ValueError, match=r"chamber_size.*explicitly specified"):
        load_experiment_yaml(proj)


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
