"""The optogenetic light QC on the first real Progressive Ratio dataset.

``C:/Users/scott/Desktop/test_data/progressive_ratio`` is a Project with one
Member recorded 2025-07-28 from 12:01 for 24 h, with the MCU's own export of
the program it ran copied into ``data/Program.txt``.  Skipped wherever that
dataset is not on disk.

What the data show, and so what is pinned here:

* DFM 1 groups 1 and 2 are the two drifting Sucrose Wells the Progressive
  Ratio light QC already fails; the emulated firmware trigger reads both as
  touched for hours with no lick.
* DFM 2 group 2 lit for about four hours in training while pyflic saw no lick
  — the Progressive Ratio check passes it (one long event is not a run of
  lick-free events), the optogenetic check does not.
* The program's trigger wells make exactly the configured paired chambers.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest

ROOT = Path("C:/Users/scott/Desktop/test_data/progressive_ratio")
MEMBER = "test_experiment"

pytestmark = pytest.mark.skipif(
    not (ROOT / MEMBER / "data" / "Program.txt").is_file(),
    reason="the real Progressive Ratio dataset (with data/Program.txt) is not on disk")


@pytest.fixture(scope="module")
def exp():
    from pyflic.base.project import Project

    with contextlib.redirect_stdout(io.StringIO()):
        return Project(ROOT).load_member(MEMBER, use_disk_cache=False)


def test_the_program_is_read_and_agrees_with_the_config(exp):
    assert exp.opto_program is not None and exp.opto_program_error is None
    assert sorted(exp.opto_program.dfms) == list(range(1, 9))
    assert exp.paired_chambers_by_dfm() == {1: [1, 3, 5], 2: [2, 4, 6]}
    assert not any("disagrees" in note for note in exp.program_notes)
    assert exp.opto.dfm_ids() == [1, 2]
    program = exp.opto.program_table()
    assert list(program["DFM"]) == [1, 2]            # the loaded DFMs' sections only
    row = program.iloc[0]
    assert (row.F, row.FrequencyHz, row.AcclimationEvents) == (5160, 40, 10)
    assert (row.DecayMs, row.Delay, row.Paradigm) == (500, 1000, "progressive ratio")


def test_the_optogenetic_verdicts(exp):
    table = exp.opto.qc_table()
    verdicts = {(int(r.DFM), int(r.Group)): r.Verdict for r in table.itertuples()}
    assert verdicts == {(1, 1): "failed", (1, 2): "failed", (1, 3): "ok",
                        (2, 1): "ok", (2, 2): "failed", (2, 3): "ok"}
    failed = table[table["Verdict"] == "failed"]
    assert (failed["Flags"] == "unexplained light").all()
    assert failed["LikelyCause"].str.contains("drifting baseline").all()
    assert not table["EmulationApproximate"].astype(bool).any()
    assert not table["Excluded"].any()              # flagged, not excluded
    long_light = table[(table.DFM == 2) & (table.Group == 2)].iloc[0]
    assert long_light["LitSec"] > 3.5 * 3600


def test_the_progressive_ratio_verdicts_are_unchanged(exp):
    table = exp.light_qc_table()
    verdicts = {(int(r.DFM), int(r.Group)): r.Verdict for r in table.itertuples()}
    assert verdicts == {(1, 1): "failed", (1, 2): "failed", (1, 3): "ok",
                        (2, 1): "ok", (2, 2): "warning", (2, 3): "ok"}
    ## The decay-aware rule explains light events a touch backed.
    healthy = table[(table.DFM == 1) & (table.Group == 3)].iloc[0]
    assert healthy.LickFreeEvents == 0
    events = exp.light_events_table(1, 1)
    assert int(exp.opto.decay_samples(1)[0]) == 3          # 500 ms at 5 Hz
    assert events["Explained"].any()
