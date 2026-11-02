"""pyflic against PyMCU's simulator under ``Baseline: Adaptive`` (PyMCU
``docs/adaptive-baseline-plan.md``, stage S5).

``simulate_recording.py --adaptive`` gives DFM 1 a slow drift on well 1, a
60-minute plateau on well 3 and a step on well 6, under
``BaselineRatePerHour: 30`` and ``BaselineCapCounts: 40``, and stops unless
PyMCU's Baseline columns are the baselines its simulated DFM ran.  Its truth
folder's ``Baselines.csv`` lists each update.  Here the recording goes through
basic analysis with QC, as ``test_pymcu_simulator.py`` does for the default
run, and pyflic's reading of the columns is checked against that truth.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyflic.base import opto_light
from pyflic.base import opto_program as op
from pyflic.base.paradigm_state import BASELINE_COLUMNS

import test_pymcu_simulator as sim

pytestmark = pytest.mark.slow

STEP_WELL = 6          # the simulator's yeast-style step, on DFM 1
DRIFT_WELL = 1         # its slow drift
CAP = 40               # its BaselineCapCounts


def _updates_in_data(truth: Path, dfm) -> pd.DataFrame:
    """DFM 1's updates from the truth that some record ran: one sent at the
    last check can take effect after the recording's last record."""
    updates = pd.read_csv(truth / "Baselines.csv")
    updates = updates[updates["FromRecord"].notna() & (updates["DFM"] == 1)]
    last = pd.to_numeric(dfm.raw_df["Index"], errors="coerce").max()
    return updates[updates["FromRecord"] <= last]


@pytest.fixture(scope="module")
def adaptive(tmp_path_factory: pytest.TempPathFactory):
    if not sim.SIMULATOR.is_file():
        pytest.skip(f"PyMCU's simulator not found at {sim.SIMULATOR} (set PYMCU_DIR)")
    out = tmp_path_factory.mktemp("pymcu_sim_adaptive")
    done = subprocess.run(
        [sys.executable, "-I", str(sim.RUNNER), str(sim.PYMCU_DIR), "--out", str(out),
         "--seed", str(sim.SEED), "--adaptive"],
        capture_output=True, text=True, errors="replace", timeout=600)
    assert done.returncode == 0, (f"PyMCU's simulator failed (exit {done.returncode}):\n{done.stderr}"
                                  f"\n--- its output ---\n{done.stdout[-2000:]}")
    truth = next(out.glob("SIMMCU_*_truth"))
    recording = truth.with_name(truth.name[:-len("_truth")])
    root = tmp_path_factory.mktemp("adaptive") / "pymcu_sim"
    project = sim.build_project(root, recording, sim.CUSTOM_DESIGN)
    exp = project.load_member(recording.name, parallel=False, use_disk_cache=False)
    exp.execute_basic_analysis(skip_qc=False)
    return exp, recording, truth


def test_the_program_says_adaptive(adaptive) -> None:
    _, recording, _ = adaptive
    program = op.read_program(recording / "Program.txt")
    assert program.adaptive_baseline and program.baseline
    assert (program.baseline_rate_per_hour, program.baseline_cap_counts) == (30.0, CAP)
    assert not program.skipped


def test_the_columns_load_beside_raw_df(adaptive) -> None:
    exp, _, _ = adaptive
    dfm = exp.dfms[1]
    assert not set(BASELINE_COLUMNS) & set(dfm.raw_df.columns)
    assert dfm.baseline_in_force.shape == (len(dfm.raw_df), 12)
    ## Interval 0, before the first instruction, has no Baseline.
    assert np.isnan(dfm.baseline_in_force[0]).all()


def test_the_baseline_in_force_is_the_simulated_one(adaptive) -> None:
    exp, _, truth = adaptive
    dfm = exp.dfms[1]
    updates = _updates_in_data(truth, dfm)
    assert len(updates) >= 3
    index = pd.to_numeric(dfm.raw_df["Index"], errors="coerce").to_numpy()
    for _, update in updates.iterrows():
        row = int(np.flatnonzero(index == int(update["FromRecord"]))[0])
        expected = [update[f"Baseline{w}"] for w in range(1, 13)]
        assert list(dfm.baseline_in_force[row]) == expected, int(update["FromRecord"])


def test_opto_baselines_csv(adaptive) -> None:
    exp, _, truth = adaptive
    path = Path(exp.experiment_dir) / "qc" / opto_light.QC_SUBDIR / opto_light.QC_FILES["opto_baselines"]
    table = pd.read_csv(path)
    dfm1 = table[table["DFM"] == 1].set_index("Well")
    assert bool(dfm1.loc[STEP_WELL, "ReachedCap"]) and dfm1.loc[STEP_WELL, "CapCounts"] == CAP
    assert dfm1.loc[DRIFT_WELL, "BaselineEnd"] > dfm1.loc[DRIFT_WELL, "BaselineStart"]
    values = _updates_in_data(truth, exp.dfms[1])[f"Baseline{DRIFT_WELL}"].to_numpy()
    assert dfm1.loc[DRIFT_WELL, "Updates"] == int(np.count_nonzero(
        np.diff(np.concatenate(([dfm1.loc[DRIFT_WELL, "BaselineStart"]], values)))))
