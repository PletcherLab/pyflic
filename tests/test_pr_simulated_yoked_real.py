"""The Simulated Yoked Breaking Point on the first real dataset (ADR-0023).

``test_data/Jove/Males`` (git-ignored) is a Project of two Members recorded
under DFM firmware 4.0.1 with ``Counter: Contact``: Chr and W males in
paired and yoked chamber groups.  It is where the replay was first checked
against the firmware (25 of 34 groups exact, 32 within one light, with each
Paired fly replayed from its own training end), and this test keeps it there.
Skipped unless the folder is present.  Slow: each Member loads in about 45 s.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent / "test_data" / "Jove" / "Males"
MEMBER = "Christi_11_17_26"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not (ROOT / MEMBER / "data").is_dir(),
                       reason=f"real dataset not found at {ROOT}"),
]


@pytest.fixture(scope="module")
def summary():
    from pyflic.base.project import Project

    exp = Project(ROOT).load_member(MEMBER)
    exp.global_constants = {**exp.global_constants, "pr_simulated_yoked_bp": True}
    removed = pd.read_csv(ROOT / MEMBER / "analysis" / "removed_chambers.csv")
    exp._remove_chambers_from_design({(int(d), int(c))
                                      for d, c in zip(removed.DFM, removed.Chamber)})
    return exp, exp.breaking_point_summary()


def test_the_paired_replay_matches_the_firmware(summary):
    _exp, s = summary
    real = s["BreakingPoint"].astype(int)
    replay = s["ReplayBreakingPoint"].astype(int)
    assert len(s) == 21
    assert int((replay == real).sum()) >= 13
    assert int(((replay - real).abs() <= 1).sum()) >= 19


def test_every_group_whose_yoked_fly_is_in_the_analysis_has_one(summary):
    _exp, s = summary
    ## DFM 1 group 1's yoked chamber (2) was removed: its pad sat at the ceiling.
    missing = s[s["SimulatedYokedBreakingPoint"].isna()]
    assert [(int(r.DFM), int(r.Group)) for r in missing.itertuples()] == [(1, 1)]
    have = s.dropna(subset=["SimulatedYokedBreakingPoint"])
    assert (have["dBreakingPoint"] == have["BreakingPoint"]
            - have["SimulatedYokedBreakingPoint"]).all()


def test_background_feeding_does_not_make_the_chr_yoked_flies_break_later(summary):
    _exp, s = summary
    by = s.dropna(subset=["SimulatedYokedBreakingPoint"]).groupby("Treatment")
    medians = by["SimulatedYokedBreakingPoint"].median()
    assert medians["Chr"] <= medians["W"] + 2
    assert by["dBreakingPoint"].median()["Chr"] > 30


def test_the_contact_credit_is_fitted_on_the_paired_wells(summary):
    exp, _s = summary
    credit = exp.replay_contact_credit()
    assert credit is not None and credit.fitted > 1_000_000
    assert credit.value((True, True, True)) == pytest.approx(199.4, abs=1.0)
    assert credit.value((False, False, False)) < 1.0
    lines = exp.simulated_yoked_lines()
    assert any(line.startswith("Replay check") for line in lines)
    assert not any(line.startswith("WARNING") for line in lines)
