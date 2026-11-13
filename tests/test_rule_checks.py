"""The rule checks: each Paradigm checked against its own rule (stream S1).

``docs/paradigm-analyses-plan.md``, stream S1, and ADR-0015's last addendum.
The checks live in :mod:`pyflic.base.rule_checks`, one function per check,
and ``opto_qc`` runs them per Linkage Group on generation 3.

The proof is fault injection on the simulator (``v4_fixtures``, contract C1):
a ``Program.txt`` that is not the program that ran (``program_paradigms``),
``OnMs`` shortened, blanked or stretched (``on_ms_faults``), a restart, and an
Interval taken up late.  Each flag fires with a reason naming the setting and
the value measured.  Then no new flag fires on clean data: every clean
simulator spec, PyMCU's simulated recording and the quiet bench run.
"""

from __future__ import annotations

import contextlib
import io
import os
import shutil
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from pyflic.base import activation_ledger as al
from pyflic.base import opto_program as op
from pyflic.base import opto_qc as oq
from pyflic.base import paradigm_families as pfam
from pyflic.base import rule_checks as rc
from pyflic.base.yaml_config import load_experiment_yaml

import v4_fixtures as vf

#: The new flags; on clean data none fires (but *DFM restarted* where one did).
NEW_FLAGS = (*rc.FAILING_RULE_FLAGS, *rc.WARNING_RULE_FLAGS)
#: The real quiet bench run, in the checkout's git-ignored ``test_data``
#: folder (or ``PYFLIC_BENCH_DIR``'s; a worktree has none, so the main
#: checkout's is tried last).  Its test skips without it.
BENCH_NAME = "FLICMCU_10_17_2026_00_17"
_BENCH_ROOTS = (os.environ.get("PYFLIC_BENCH_DIR"),
                Path(__file__).resolve().parents[1] / "test_data",
                "C:/Users/scott/Desktop/GithubLocal/pyflic/test_data")
BENCH = next((Path(r) / BENCH_NAME for r in _BENCH_ROOTS
              if r and (Path(r) / BENCH_NAME).is_dir()), None)


@lru_cache(maxsize=None)
def _family() -> dict:
    return vf.family_specs()


SPECS = {
    **{key: (lambda key=key: _family()[key]) for key in vf.FAMILY_SPEC_KEYS},
    "one_of_each": vf.one_of_each_spec,
    "choice": vf.choice_pr_spec,
    "choice1": lambda: vf.choice_pr_spec(dfm_ids=(1,)),
    "yoked": vf.yoked_controls_spec,
    "stress": vf.stress_spec,
    "fixed_ratio": vf.fixed_ratio_spec,
    "repeating": vf.repeating_open_loop_spec,
}


@lru_cache(maxsize=None)
def sim(name: str):
    spec = SPECS[name]()
    return spec, vf.simulate(spec)


def _config(dfm_ids) -> dict:
    """A Custom two-well config, A wells on the left, every chamber assigned."""
    params = {"feeding_threshold": 20, "feeding_minimum": 10, "tasting_minimum": 5,
              "tasting_maximum": 20, "samples_per_second": 5, "pi_direction": "left"}
    return {"global": {"chamber_layout": "two_well", "transform_licks": False,
                       "params": params},
            "dfms": [{"id": int(i), "chambers": {c: ("Ctrl" if c % 2 else "Exp")
                                                 for c in range(1, 7)}}
                     for i in dfm_ids]}


def load(root: Path, name: str, *, edit=None, spec=None, **kw):
    """The recording *name* (or *spec*) written to *root* with the faults
    of *kw* and loaded; *edit* changes the simulated rows first."""
    if spec is None:
        spec, results = sim(name)
    else:
        results = vf.simulate(spec)
    if edit is not None:
        results = {k: vf.SimResult(v.rows.copy(), v.truth) for k, v in results.items()}
        edit({k: v.rows for k, v in results.items()})
    root.mkdir(parents=True, exist_ok=True)
    vf.write_v4_recording(root / "data", spec, results=results, **kw)
    (root / "flic_config.yaml").write_text(
        yaml.safe_dump(_config([d.dfm_id for d in spec.dfms]), sort_keys=False))
    with contextlib.redirect_stdout(io.StringIO()):
        return load_experiment_yaml(root, parallel=False, use_disk_cache=False)


def _flags(row) -> list[str]:
    return [f for f in str(row["Flags"]).split(", ") if f and f != "nan"]


def _groups(e, dfm_id: int) -> pd.DataFrame:
    table = e.opto.qc_table()
    return table[table["DFM"] == dfm_id].set_index("Group")


def _text(e) -> str:
    return "\n".join(e.opto.lines())


def _new_flags(e) -> dict[str, int]:
    out: dict[str, int] = {}
    for flags in e.opto.qc_table()["Flags"]:
        for f in str(flags).split(", "):
            if f in NEW_FLAGS:
                out[f] = out.get(f, 0) + 1
    return out


# ---------------------------------------------------------------------------
# The module: names, columns, the ledger's rows
# ---------------------------------------------------------------------------

def test_the_flags_sit_where_decision_6_puts_them():
    assert set(rc.FAILING_RULE_FLAGS) == {
        "wrong activation length", "lockout broken", "requirement series wrong",
        "max on exceeded", "short activations"}
    assert set(rc.WARNING_RULE_FLAGS) == {
        "contact without activation", "non-feeding dark", "interval started late",
        "DFM restarted"}
    assert set(rc.FAILING_RULE_FLAGS) <= set(oq.FAILING_FLAGS)
    assert set(rc.WARNING_RULE_FLAGS) <= set(oq.WARNING_FLAGS)
    assert not set(oq.FAILING_FLAGS) & set(oq.WARNING_FLAGS)
    assert oq.verdict_of([oq.LOCKOUT_BROKEN]) == "failed"
    assert oq.verdict_of([oq.DFM_RESTARTED]) == "warning"


def test_the_columns_are_the_registrys():
    """Every family's QC columns are in the QC table, named as the
    registry declares them (contract C5)."""
    declared = {c for f in pfam.FAMILIES.values() for c in f.qc_columns} \
        | set(pfam.COMMON_QC_COLUMNS)
    assert declared <= set(oq.OPTO_QC_COLUMNS)
    assert set(rc.RULE_QC_COLUMNS) == declared - {"OpenLoopLitFraction"}
    assert oq.OPTO_EVENT_COLUMNS[-1] == "EndedBy"
    ## The series check is importable on its own, and from opto_qc (S4).
    from pyflic.base.rule_checks import requirement_series_mismatches
    assert oq.requirement_series_mismatches is requirement_series_mismatches


def _ledger(**columns) -> pd.DataFrame:
    """Ledger rows: the given columns, the rest at clean values."""
    n = len(next(iter(columns.values())))
    base = {"Well": [1] * n, "Row": list(range(10, 10 + 10 * n, 10)),
            "EndRow": list(range(12, 12 + 10 * n, 10)), "Index": list(range(11, 11 + 10 * n, 10)),
            "OnsetMin": [0.0] * n, "OnsetMs": [1000 * i for i in range(n)],
            "EndMs": [1000 * i + 500 for i in range(n)], "LitMs": [500] * n,
            "Timing": [al.TIMING_EXACT] * n, "Overlap": [False] * n,
            "Boundary": [al.BOUNDARY_NONE] * n,
            "SinceLastOnsetMs": [np.nan] + [1000.0] * (n - 1), "Phase": ["M"] * n,
            "K": list(range(1, n + 1)), "Paradigm": ["P"] * n,
            "ProgramInterval": [1] * n, "Cycle": [1] * n, "RequirementMet": [np.nan] * n}
    base.update(columns)
    frame = pd.DataFrame(base)[list(al.LEDGER_COLUMNS)]
    for column in ("SinceLastOnsetMs", "K", "RequirementMet"):
        frame[column] = pd.array(frame[column], dtype="Int64")
    return frame


def _paradigm(type_: str, **settings) -> op.Paradigm:
    return op.Paradigm(name="P", type=type_, frequency_hz=40, pulse_width_ms=8, **settings)


def test_only_unoverlapped_rows_the_rule_stopped_are_judged():
    rows = _ledger(LitMs=[500, 100, 100, 100],
                   Overlap=[False, True, False, False],
                   Boundary=["", "", al.BOUNDARY_INTERVAL, ""],
                   Timing=[al.TIMING_EXACT, al.TIMING_EXACT, al.TIMING_EXACT, al.TIMING_RECORD])
    assert rc.judgeable(rows).tolist() == [True, False, False, True]
    assert rc.tolerances(rows, 25).tolist() == [25, 25, 25, 225]
    p = _paradigm(op.CLOSED_LOOP, hold_ms=500, max_on_ms=0)
    result = rc.short_activations(rows, p, tol_ms=25, fail_fraction=0.05)
    ## The last is 400 ms short, but timed only to the record: 100 >= 500 - 225 fails.
    assert (result.value, result.judged, result.not_judged) == (1, 2, 2)
    assert result.fired and "HoldMs of 500 ms" in result.reason
    assert "the shortest for 100 ms" in result.reason


def test_each_length_check_reads_its_own_setting():
    lit = [2000, 2010, 2100, 900]
    closed = _paradigm(op.CLOSED_LOOP, hold_ms=500, max_on_ms=2000)
    over = rc.max_on_exceeded(_ledger(LitMs=lit), closed, tol_ms=25)
    assert (over.value, over.fired) == (1, True)
    assert "MaxOnMs of 2000 ms" in over.reason and "2100 ms" in over.reason
    assert rc.cut_offs(_ledger(LitMs=lit), closed, tol_ms=25).value == 2
    fixed = _paradigm(op.FIXED_RATIO, activation_ms=2000, ratio=3)
    wrong = rc.wrong_activation_length(_ledger(LitMs=lit), fixed, tol_ms=25, fail_fraction=0.05)
    assert (wrong.value, wrong.judged, wrong.fired) == (2, 4, True)
    assert "ActivationMs of 2000 ms" in wrong.reason and "900-2100 ms" in wrong.reason
    ## Phase A rows of a fixed-length type are closed loop: their Hold is ActivationMs.
    accl = _paradigm(op.FIXED_RATIO, activation_ms=2000, ratio=3, acclimation=2)
    rows = _ledger(LitMs=lit, Phase=["A", "A", "M", "M"], K=[None, None, 1, 2])
    assert rc.wrong_activation_length(rows, accl, tol_ms=25, fail_fraction=0.05).judged == 2
    assert rc.short_activations(rows, accl, tol_ms=25, fail_fraction=0.05).judged == 2
    ## Not their rule: not applicable, so the column is NA.
    assert not rc.max_on_exceeded(_ledger(LitMs=lit), fixed, tol_ms=25).applies
    assert not rc.short_activations(_ledger(LitMs=lit), fixed, tol_ms=25,
                                    fail_fraction=0.05).applies
    assert rc.combine([rc.not_applicable(rc.SHORT_ACTIVATIONS)]) is None


def test_the_lockout_is_judged_between_readable_main_onsets():
    p = _paradigm(op.FIXED_INTERVAL, activation_ms=1000, lockout_ms=10000)
    onsets = [0, 10000, 19000, 30000, 31000]
    rows = _ledger(OnsetMs=onsets, LitMs=[1000] * 5,
                   SinceLastOnsetMs=[np.nan, 10000, 9000, 11000, 1000],
                   Overlap=[False, False, False, False, True])
    result = rc.lockout_violations(rows, p, tol_ms=25, fail_fraction=0.02)
    ## The overlapped one's onset cannot be read: 3 pairs judged, 1 broken.
    assert (result.value, result.judged, result.not_judged) == (1, 3, 1)
    assert "the closest 9000 ms after" in result.reason
    assert "the longer of LockoutMs 10000 and ActivationMs 1000" in result.reason


def test_contact_counts_as_missed_only_when_the_well_was_free():
    """Activity runs at rows 5 (free: nothing before), 20 (free, no
    Activation: missed), 60 (within the lockout of the onset at row 21)."""
    p = _paradigm(op.FIXED_INTERVAL, activation_ms=1000, lockout_ms=10000)
    n = 200
    index = np.arange(1, n + 1)
    activity = np.zeros(n, dtype=bool)
    activity[[5, 6, 50, 51, 60]] = True
    rows = _ledger(Row=[5], EndRow=[10], OnsetMs=[5 * 200 + 50], EndMs=[5 * 200 + 1050],
                   LitMs=[1000], SinceLastOnsetMs=[np.nan], K=[1])
    eligible = np.ones(n, dtype=bool)
    result = rc.missed_contacts(rows, p, {1: activity}, index, {1: eligible}, tol_ms=25,
                                warn_fraction=0.25, min_runs=1)
    ## Row 50 starts 9 s after the onset at row 5: not free.  Row 60, 11 s: free, missed.
    assert (result.value, result.judged) == (1, 2)
    assert result.fired and "1 of 2 runs" in result.reason
    assert not rc.missed_contacts(rows, p, {1: activity}, index, {1: eligible}, tol_ms=25,
                                  warn_fraction=0.25).fired          # fewer than 5 free runs


def _series_rows(met, k, phase=None, rows=None):
    n = len(met)
    return _ledger(RequirementMet=met, K=k, Phase=phase or ["M"] * n,
                   Row=rows or list(range(10, 10 + 10 * n, 10)),
                   OnsetMin=[float(i) for i in range(n)])


def test_the_series_check_compares_each_main_activation_with_requirement_for_k():
    pr = op.Paradigm(name="PR10", type=op.PROGRESSIVE_RATIO, frequency_hz=40, pulse_width_ms=8,
                     activation_ms=500, series=op.SERIES_ARITHMETIC, first_requirement=1,
                     step=10, cap=200, acclimation=0)
    assert rc.requirement_series_mismatches(_series_rows([1, 11, 21], [1, 2, 3]), pr) == []
    bad = rc.requirement_series_mismatches(_series_rows([1, 6, 11], [1, 2, 3]), pr)
    assert [(m.k, m.met, m.expected) for m in bad] == [(2, 6, 11), (3, 11, 21)]
    ## Two onsets in one row share the Requirement on the row before: not compared.
    rows = _series_rows([1, 11, 11], [1, 2, 3], rows=[10, 20, 20])
    assert rc.requirement_series_mismatches(rows, pr) == []
    ## Unknown on either side, or Acclimation: not compared.
    rows = _series_rows([None, 11, 5], [1, 2, None], phase=["M", "M", "A"])
    assert rc.requirement_series_mismatches(rows, pr) == []
    result = rc.requirement_series(_series_rows([1, 6, 11], [1, 2, 3]), pr)
    assert result.fired and result.value == 2 and "W1's 2nd met 6, the series gives 11" in \
        result.reason
    ## Other types have no series.
    assert rc.requirement_series_mismatches(_series_rows([1], [1]),
                                            _paradigm(op.CLOSED_LOOP, hold_ms=500)) == []


def test_a_run_whose_first_row_holds_the_old_paradigms_activation_is_resolved():
    """The ledger counts an Activation on a new Paradigm's first row in the
    new state run, where no Requirement shows; when it was the old Paradigm's,
    every K of the run is one too many.  Matched with K one less, the run
    is resolved, not failed, and a note says so."""
    pr = op.Paradigm(name="PR10", type=op.PROGRESSIVE_RATIO, frequency_hz=40, pulse_width_ms=8,
                     activation_ms=500, series=op.SERIES_ARITHMETIC, first_requirement=1,
                     step=10, cap=200)
    shifted = _series_rows([None, 1, 11, 21], [1, 2, 3, 4])
    assert rc.requirement_series_mismatches(shifted, pr) == []
    assert rc.series_shifted_runs(shifted, pr) == 1
    result = rc.requirement_series(shifted, pr)
    assert not result.fired and "taken as the previous Paradigm's" in result.note
    ## A wrong program is not excused: off by one, but not by the series' step.
    wrong = _series_rows([None, 2, 12, 22], [1, 2, 3, 4])
    assert len(rc.requirement_series_mismatches(wrong, pr)) == 3
    ## Nor a run whose first Main Activation shows its Requirement.
    shown = _series_rows([5, 1, 11, 21], [1, 2, 3, 4])
    assert len(rc.requirement_series_mismatches(shown, pr)) == 4


def test_the_non_feeding_rule_says_lit_after_the_delay_and_within_max_on():
    p = _paradigm(op.NON_FEEDING, on_delay_ms=1000, max_on_ms=0)
    n = 100
    index = np.arange(1, n + 1)
    activity = np.zeros(n, dtype=bool)
    activity[[20, 21]] = True                     # contact ends by 4400 ms
    starts = np.zeros(n, dtype=bool)
    starts[0] = True
    expected = rc.expected_lit_ms(activity, index, np.ones(n, dtype=bool), starts, p, tol_ms=25)
    ## From the Paradigm's start: from the end of its first row on.
    assert expected[0] == 0 and expected[1] == 175 and (expected[2:20] == 200).all()
    ## After contact: from 4400 + 200 (a record pyflic may miss) + 1000 + 25 = 5625 ms.
    assert (expected[20:28] == 0).all() and expected[28] == 175 and expected[29] == 200
    capped = _paradigm(op.NON_FEEDING, on_delay_ms=1000, max_on_ms=2000)
    expected = rc.expected_lit_ms(activity, index, np.ones(n, dtype=bool), starts, capped,
                                  tol_ms=25)
    ## ... and to 4200 + 1000 + 2000 - 25 = 7175 ms with MaxOnMs.
    assert expected[35] == 175 and (expected[36:] == 0).all()
    ## Without a Paradigm start, nothing is expected before the first contact.
    expected = rc.expected_lit_ms(activity, index, np.ones(n, dtype=bool),
                                  np.zeros(n, dtype=bool), p, tol_ms=25)
    assert (expected[:28] == 0).all()


def test_ended_by_names_what_ended_each_activation():
    rows = _ledger(LitMs=[2000, 700, 500, 500], Overlap=[False, False, True, False],
                   Boundary=["", "", "", al.BOUNDARY_GAP])
    closed = _paradigm(op.CLOSED_LOOP, hold_ms=500, max_on_ms=2000)
    got = rc.ended_by(rows, closed, activity={}, index=np.arange(1, 50), tol_ms=25)
    assert got.tolist() == [rc.ENDED_MAX_ON, rc.ENDED_HOLD, rc.ENDED_UNKNOWN, al.BOUNDARY_GAP]
    fixed = _paradigm(op.FIXED_INTERVAL, activation_ms=500, lockout_ms=1000)
    got = rc.ended_by(rows, fixed, activity={}, index=np.arange(1, 50), tol_ms=25)
    assert got.tolist() == [rc.ENDED_UNKNOWN, rc.ENDED_UNKNOWN, rc.ENDED_UNKNOWN,
                            al.BOUNDARY_GAP]
    assert rc.ended_by(rows.iloc[[3]], fixed, activity={}, index=np.arange(1, 50),
                       tol_ms=25).tolist() == [al.BOUNDARY_GAP]
    ## Non-feeding: contact at the well where the light went out.
    nf = _paradigm(op.NON_FEEDING, on_delay_ms=500, max_on_ms=0)
    activity = np.zeros(60, dtype=bool)
    activity[[2]] = True                          # 400-600 ms
    one = _ledger(OnsetMs=[0], EndMs=[450], LitMs=[450], Row=[0], EndRow=[2])
    assert rc.ended_by(one, nf, activity={1: activity}, index=np.arange(1, 61),
                       tol_ms=25).tolist() == [rc.ENDED_CONTACT]
    assert rc.ended_by(one, nf, activity={1: np.zeros(60, bool)}, index=np.arange(1, 61),
                       tol_ms=25).tolist() == [rc.ENDED_UNKNOWN]
    assert rc.ended_by(one, None, activity={}, index=np.arange(1, 61),
                       tol_ms=25).tolist() == [rc.ENDED_UNKNOWN]


# ---------------------------------------------------------------------------
# Fault injection: a Program.txt that is not the program that ran
# ---------------------------------------------------------------------------

def test_a_wrong_activation_ms_fails_wrong_activation_length(tmp_path):
    e = load(tmp_path / "e", "fixed_interval",
             program_paradigms={"FI10s": replace(vf.FI10S, activation_ms=1500)})
    groups = _groups(e, 1)
    assert (groups["Verdict"] == "failed").all()
    assert all(oq.WRONG_ACTIVATION_LENGTH in _flags(r) for _, r in groups.iterrows())
    assert (groups["WrongLengthActivations"] > 0).all()
    text = _text(e)
    assert "did not last its ActivationMs of 1500 ms within 25 ms" in text
    assert "they lasted 1000 ms" in text and "Program.txt is probably not the program" in text
    ## DFM 2's FI500 is right, and stays ok.
    assert (_groups(e, 2)["Verdict"] == "ok").all()


def test_a_longer_lockout_in_the_file_fails_lockout_broken(tmp_path):
    e = load(tmp_path / "e", "fixed_interval",
             program_paradigms={"FI10s": replace(vf.FI10S, lockout_ms=20000)})
    groups = _groups(e, 1)
    assert all(oq.LOCKOUT_BROKEN in _flags(r) for _, r in groups.iterrows())
    assert (groups["LockoutViolations"] > 0).all()
    text = _text(e)
    assert "the longer of LockoutMs 20000 and ActivationMs 1000" in text
    assert "the closest 10000 ms after" in text


def test_a_shorter_lockout_in_the_file_warns_contact_without_activation(tmp_path):
    """The file says the well is free 2 s after an onset; the DFM ran 10 s,
    so contact in between lit nothing."""
    e = load(tmp_path / "e", "fixed_interval",
             program_paradigms={"FI10s": replace(vf.FI10S, lockout_ms=2000)})
    groups = _groups(e, 1)
    assert all(_flags(r) == [oq.CONTACT_WITHOUT_ACTIVATION] for _, r in groups.iterrows())
    assert (groups["Verdict"] == "warning").all()
    assert "began with the well free under FI10s" in _text(e)
    assert "opto_missed_contact_warn_fraction is 0.25" in _text(e)


def test_a_longer_hold_in_the_file_fails_short_activations(tmp_path):
    e = load(tmp_path / "e", "closed_max",
             program_paradigms={"CL500": replace(vf.CL500, hold_ms=1500)})
    groups = _groups(e, 1)
    assert all(oq.SHORT_ACTIVATIONS in _flags(r) for _, r in groups.iterrows())
    assert "were lit for less than its HoldMs of 1500 ms less 25 ms" in _text(e)


def test_a_longer_acclimation_hold_fails_short_activations(tmp_path):
    e = load(tmp_path / "e", "fixed_interval",
             program_paradigms={"FI500": replace(vf.FI500, activation_ms=3000)})
    flags = [f for _, r in _groups(e, 2).iterrows() for f in _flags(r)]
    assert oq.SHORT_ACTIVATIONS in flags and oq.WRONG_ACTIVATION_LENGTH in flags
    assert "its Acclimation Hold, ActivationMs 3000 ms" in _text(e)


def test_a_shorter_max_on_in_the_file_fails_max_on_exceeded(tmp_path):
    e = load(tmp_path / "e", "closed_max",
             program_paradigms={"CL500": replace(vf.CL500, max_on_ms=1000)})
    groups = _groups(e, 1)
    assert all(oq.MAX_ON_EXCEEDED in _flags(r) for _, r in groups.iterrows())
    assert (groups["OverMaxOnActivations"] > 0).all()
    assert "lit for more than its MaxOnMs of 1000 ms plus 25 ms" in _text(e)
    nf = load(tmp_path / "nf", "non_feeding",
              program_paradigms={"NFMax5s": replace(vf.NF_MAX5S, max_on_ms=2000)})
    assert all(oq.MAX_ON_EXCEEDED in _flags(r) for _, r in _groups(nf, 2).iterrows())
    assert (_groups(nf, 1)["Verdict"] == "ok").all()
    assert "its MaxOnMs of 2000 ms plus 25 ms" in _text(nf)


@pytest.mark.parametrize("name, paradigm, words", [
    ("choice1", replace(vf.PR10, step=5), "Arithmetic, FirstRequirement 1, Step 5"),
    ("fixed_ratio", replace(vf.fixed_ratio_spec().paradigms["FR5"], ratio=4), "Ratio 4"),
    ("exponential", replace(vf.PR_EXP, rate=0.3), "Exponential, Scale 5, Rate 0.3"),
])
def test_another_series_in_the_file_fails_requirement_series_wrong(tmp_path, name, paradigm,
                                                                   words):
    e = load(tmp_path / "e", name, program_paradigms={paradigm.name: paradigm})
    table = e.opto.qc_table()
    hit = table[table["SeriesMismatches"].fillna(0) > 0]
    assert len(hit) and all(oq.REQUIREMENT_SERIES_WRONG in _flags(r)
                            for _, r in hit.iterrows())
    text = _text(e)
    assert f"met a Requirement its series does not give ({words}" in text
    assert "the series gives" in text


# ---------------------------------------------------------------------------
# Fault injection: OnMs shortened, blanked, stretched
# ---------------------------------------------------------------------------

def _onsets(name: str, dfm_id: int, well: int) -> np.ndarray:
    return np.asarray(sim(name)[1][dfm_id].truth.activation_onset_ms[well])


def test_activations_cut_short_fail_wrong_activation_length(tmp_path):
    """Ten of W1's FixedInterval Activations go dark 400 ms in (OnMs 0 from
    there, on W1 and its linked W2): a light cut short."""
    onsets = _onsets("fixed_interval", 1, 1)[2:12]
    faults = [vf.OnMsFault(w, t / 1000 + 0.4, t / 1000 + 1.2, 0) for t in onsets for w in (1, 2)]
    e = load(tmp_path / "e", "fixed_interval", on_ms_faults={1: faults})
    row = _groups(e, 1).loc[1]
    assert oq.WRONG_ACTIVATION_LENGTH in _flags(row) and row["Verdict"] == "failed"
    assert row["WrongLengthActivations"] == 10
    assert "did not last its ActivationMs of 1000 ms" in _text(e)
    assert (_groups(e, 1).drop(index=1)["Verdict"] == "ok").all()


def test_closed_loop_light_cut_short_fails_short_activations(tmp_path):
    onsets = _onsets("closed_max", 1, 1)[2:22]
    faults = [vf.OnMsFault(w, t / 1000 + 0.2, t / 1000 + 0.6, 0) for t in onsets for w in (1, 2)]
    e = load(tmp_path / "e", "closed_max", on_ms_faults={1: faults})
    row = _groups(e, 1).loc[1]
    assert oq.SHORT_ACTIVATIONS in _flags(row) and row["ShortActivations"] >= 10
    assert "were lit for less than its HoldMs of 500 ms" in _text(e)


def test_a_light_stuck_on_fails_max_on_exceeded(tmp_path):
    """W1's LED (and its linked W2's) stays lit 6 s from an Activation's
    onset, past CL500's MaxOnMs of 2000 ms."""
    onsets = _onsets("closed_max", 1, 1)
    gap = np.diff(onsets)
    t0 = int(onsets[:-1][(gap > 9000) & (onsets[:-1] > 60000)][0]) / 1000
    e = load(tmp_path / "e", "closed_max", on_ms_faults={
        1: [vf.OnMsFault(w, t0 + 0.2, t0 + 6.2, 200) for w in (1, 2)]})
    row = _groups(e, 1).loc[1]
    assert _flags(row) == [oq.MAX_ON_EXCEEDED] and row["OverMaxOnActivations"] == 1
    assert "lit for more than its MaxOnMs of 2000 ms plus 25 ms" in _text(e)
    events = e.opto.events_table()
    stuck = events[(events["DFM"] == 1) & (events["Well"] == 1) & (events["LitMs"] > 6000)]
    assert stuck["EndedBy"].tolist() == [rc.ENDED_HOLD]


def test_a_dark_non_feeding_led_warns_non_feeding_dark(tmp_path):
    e = load(tmp_path / "e", "non_feeding", on_ms_faults={
        1: [vf.OnMsFault(w, 120, 1100, 0) for w in (1, 2)]})
    row = _groups(e, 1).loc[1]
    assert _flags(row) == [oq.NON_FEEDING_DARK] and row["Verdict"] == "warning"
    assert row["NonFeedingDarkSec"] > 500
    text = _text(e)
    assert "rule says lit" in text and "its OnDelayMs of 1000 ms plus 25 ms" in text
    assert "opto_non_feeding_dark_warn_fraction is 0.1" in text
    assert (_groups(e, 1).drop(index=1)["Verdict"] == "ok").all()


def test_one_dark_pass_of_an_open_loop_warns(tmp_path):
    """25 s dark in the second pass of a Repeating open loop: 86% lit there,
    though over all three passes it would be 95%, the old verdict's ok."""
    e = load(tmp_path / "e", "repeating", on_ms_faults={
        1: [vf.OnMsFault(w, 960, 985, 0) for w in (1, 2)]})
    row = _groups(e, 1).loc[1]
    assert _flags(row) == [oq.OPEN_LOOP_DARK]
    assert row["OpenLoopLitFraction"] == pytest.approx(155 / 180, abs=0.01)
    assert "lit for 86% of Program Interval 1 (Cycle 2), the lowest of its 3 open-loop " \
           "window(s)" in _text(e)
    intervals = e.opto.interval_table()
    on = intervals[(intervals["Group"] == 1) & (intervals["Mode"] == op.OPEN_LOOP)]
    assert on["LitSec"].sum() / (on["Samples"].sum() * 0.2) > 0.95


# ---------------------------------------------------------------------------
# Every family: an Interval taken up late, a restart
# ---------------------------------------------------------------------------

def test_an_interval_taken_up_late_warns(tmp_path):
    def late(rows):
        frame = rows[1]
        interval = frame["Interval"].to_numpy(dtype=float, na_value=np.nan)
        cycle = frame["Cycle"].to_numpy(dtype=float, na_value=np.nan)
        first = int(np.flatnonzero((interval == 2) & (cycle == 1))[0])
        ## 50 more records under Interval 1's instruction (On), as if the
        ## DFM had not taken up Interval 2's.
        frame.loc[first:first + 49, "Interval"] = 1
        for w in range(1, 13):
            frame.loc[first:first + 49, f"OnMs{w}"] = 200

    e = load(tmp_path / "e", "repeating", edit=late)
    result = e.opto.result(1)
    assert result.flags == [oq.INTERVAL_STARTED_LATE]
    assert np.allclose(e.opto.qc_table()["IntervalLagSec"], 10.2)
    assert any("Program Interval 2 (Cycle 1) began 10.2 s after its Program.txt time "
               "23:37:00" in n and "opto_interval_lag_warn_sec is 5" in n
               for n in result.notes)
    clean = load(tmp_path / "clean", "repeating")
    assert np.allclose(clean.opto.qc_table()["IntervalLagSec"], 0.2)


def test_a_recording_started_after_the_start_time_is_not_late(tmp_path):
    """PyMCU runs a Program on the clock (``InstructionSet.GetPosition``), so
    recording started 3 min after the Start Time joins its Interval partway:
    the DFM takes it up after the usual lead-in, which is not late, and the
    rows record what ran, so the offset is a note and not a mismatch."""
    root = tmp_path / "e"
    load(root, "", spec=vf.fixed_ratio_spec(minutes=10.0))
    path = root / "data" / "Program.txt"
    text = path.read_text(encoding="utf-8")
    text = text.replace("Start Time: 10/16/2026 23:34:00", "Start Time: 10/16/2026 23:31:00")
    text = text.replace("Duration: 10 min", "Duration: 13 min")
    text = text.replace("(10/16/2026 23:34:00) FR5", "(10/16/2026 23:31:00) FR5")
    path.write_text(text.replace("20,Off, 10 min", "20,Off, 13 min"), encoding="utf-8")
    with contextlib.redirect_stdout(io.StringIO()):
        e = load_experiment_yaml(root, parallel=False, use_disk_cache=False)
    result = e.opto.result(1)
    assert oq.INTERVAL_STARTED_LATE not in result.flags
    assert oq.PROGRAM_MISMATCH not in result.flags
    assert any("3.0 min after the program's Start Time" in n for n in result.notes)
    assert (e.opto.qc_table()["IntervalLagSec"] < 1.0).all()


def test_the_lag_is_counted_against_each_cycles_time_and_past_the_lead_in():
    """Repeating: Interval k of Cycle c is due a cycle's length later each
    pass; Circadian: on the time of day, Cycle n the n-th calendar day."""
    spec = vf.stress_spec()
    program = op.parse_program(vf.program_text(spec))
    repeating, circadian = program.section(2), program.section(3)
    start = program.start
    assert rc.scheduled_start(repeating, start, 1, 2) == start + pd.Timedelta(minutes=1.2)
    assert rc.scheduled_start(repeating, start, 3, 1) == start + pd.Timedelta(minutes=0.7)
    ## The Circadian program started at 23:56, inside Interval 1: no start then.
    assert rc.scheduled_start(circadian, start, 1, 1) is None
    assert rc.scheduled_start(circadian, start, 1, 2) == pd.Timestamp("2026-10-17 00:00:00")
    assert rc.scheduled_start(circadian, start, 9, 1) is None       # not printed


def test_a_restart_warns_every_group_of_its_dfm_with_its_time(tmp_path):
    e = load(tmp_path / "e", "restart")
    for dfm_id, minute in ((1, "10.1"), (2, "7.1")):
        result = e.opto.result(dfm_id)
        assert result.flags == [oq.DFM_RESTARTED]
        assert any(f"1 restart(s): at minute {minute}" in n for n in result.notes)
        groups = _groups(e, dfm_id)
        assert (groups["Restarts"] == 1).all() and (groups["Verdict"] == "warning").all()
    assert "25 record(s) (5.0 s) missing from Index, then 5 row(s) under no instruction" in \
        "\n".join(e.opto.result(1).notes)
    ## K and the series start again after it: no series check fails.
    assert (_groups(e, 1)["SeriesMismatches"] == 0).all()
    ## The Activation cut by the restart ended at the gap.
    events = e.opto.events_table()
    assert al.BOUNDARY_GAP in set(events.loc[events["DFM"] == 2, "EndedBy"])


def test_a_restarts_gap_is_not_lost_records(tmp_path):
    """A 10 s restart loses 50 records, more than opto_lost_records_warn's
    25, but DFM restarted reports it; a gap of lost records still warns, and
    says the restart's records are not counted."""
    base = vf.fixed_ratio_spec(minutes=10.0)
    spec = replace(base, dfms=(replace(base.dfms[0], restarts=(vf.Restart(300.0, down_s=10.0),)),))
    e = load(tmp_path / "restart", "", spec=spec)
    assert e.opto.result(1).flags == [oq.DFM_RESTARTED]
    lost = load(tmp_path / "lost", "", spec=spec, index_gaps={1: [(1000, 40)]})
    result = lost.opto.result(1)
    assert result.flags == [oq.DFM_RESTARTED, oq.LOST_RECORDS]
    note = next(n for n in result.notes if n.startswith(oq.LOST_RECORDS))
    assert "40 record(s) (8.0 s) missing from Index in 1 gap(s)" in note
    assert "the 50 record(s) the DFM's restart(s) lost are not counted" in note


def test_a_restarts_rows_before_the_resend_are_not_unknown_instruction_rows(tmp_path):
    """After a restart the DFM records under no instruction until PyMCU
    resends it.  Those rows are the restart's, which DFM restarted reports,
    so even with no unknown-instruction rows allowed they raise nothing
    more; rows under an unknown instruction elsewhere still warn."""
    base = vf.fixed_ratio_spec(minutes=10.0)
    spec = replace(base, dfms=(replace(base.dfms[0],
                                       restarts=(vf.Restart(300.0, resend_s=2.0),)),))
    strict = oq.OptoQCSettings.from_constants({"opto_unknown_rows_warn_fraction": 0})
    e = load(tmp_path / "restart", "", spec=spec)
    state = e.dfms[1].paradigm
    assert int(state.unknown_rows().sum()) == 10              # 2 s at 5 records a second
    flags: list[str] = []
    notes: list[str] = []
    oq._record_checks(state, strict, 5.0, flags, notes)
    assert flags == [oq.DFM_RESTARTED]

    def unknown_elsewhere(rows):
        frame = rows[1]
        at = frame.index[2000:2005]
        frame.loc[at, "Interval"] = pd.NA
        frame.loc[at, "Paradigm"] = pd.NA

    e = load(tmp_path / "elsewhere", "", spec=spec, edit=unknown_elsewhere)
    flags, notes = [], []
    oq._record_checks(e.dfms[1].paradigm, strict, 5.0, flags, notes)
    assert flags == [oq.DFM_RESTARTED, oq.UNKNOWN_INSTRUCTION_ROWS]
    note = next(n for n in notes if n.startswith(oq.UNKNOWN_INSTRUCTION_ROWS))
    assert note.startswith(f"{oq.UNKNOWN_INSTRUCTION_ROWS}: 5 of the ")
    assert "the 10 row(s) after the DFM's restart(s)" in note


# ---------------------------------------------------------------------------
# No new flag on clean data
# ---------------------------------------------------------------------------

#: Each clean spec's designed faults: the flags it is built to raise.
CLEAN = {
    "closed_max": {}, "fixed_interval": {}, "non_feeding": {}, "exponential": {},
    "restart": {oq.DFM_RESTARTED: 12}, "one_of_each": {}, "choice": {}, "yoked": {},
    "stress": {}, "fixed_ratio": {}, "repeating": {},
}


@pytest.mark.parametrize("name", list(CLEAN))
def test_no_new_flag_fires_on_clean_data(tmp_path, name):
    e = load(tmp_path / "e", name)
    assert _new_flags(e) == CLEAN[name]
    table = e.opto.qc_table()
    ## Filled where the program has the rule, NA where not.
    assert table["Restarts"].notna().all()
    assert table["IntervalLagSec"].notna().all()
    assert (table["IntervalLagSec"] < 1.0).all()
    events = e.opto.events_table()
    assert events["EndedBy"].notna().all()
    assert set(events["EndedBy"]) <= set(rc.ENDED_BY_VALUES)


def test_the_rule_columns_follow_the_families(tmp_path):
    """one_of_each runs one type per DFM: each column is filled exactly on
    its family's DFMs, and EndedBy names each family's own ending."""
    e = load(tmp_path / "e", "one_of_each")
    table = e.opto.qc_table().set_index("DFM")
    filled = {c: sorted({int(d) for d in table.index[table[c].notna()]})
              for c in rc.RULE_QC_COLUMNS}
    ## DFM 5 runs Closed, then PR10 with Acclimation; DFM 6's FR5 has none, so
    ## it has no closed-loop rule.
    assert filled == {
        "ShortActivations": [2, 5], "OverMaxOnActivations": [2, 5], "CutOffs": [2, 5],
        "WrongLengthActivations": [3, 5, 6], "LockoutViolations": [3], "MissedContacts": [3],
        "SeriesMismatches": [5, 6], "NonFeedingDarkSec": [4], "RelightDelayMs": [4],
        "IntervalLagSec": [1, 2, 3, 4, 5, 6], "Restarts": [1, 2, 3, 4, 5, 6]}
    assert (table.loc[2, "CutOffs"] > 0).all()
    ## NotFeeding's OnDelayMs is 500; a delay reads up to a record over it.
    assert table.loc[4, "RelightDelayMs"].between(500, 700).all()
    events = e.opto.events_table()
    ended = events.groupby("DFM")["EndedBy"].agg(lambda s: set(s))
    assert ended[2] == {rc.ENDED_MAX_ON, rc.ENDED_HOLD}
    assert ended[3] == {rc.ENDED_LENGTH} and ended[6] == {rc.ENDED_LENGTH}
    assert rc.ENDED_CONTACT in ended[4]


def test_generation_2_has_the_new_columns_na(tmp_path):
    e = load(tmp_path / "e", "fixed_interval", )
    gen2 = tmp_path / "gen2"
    spec, results = sim("fixed_interval")
    vf.write_gen2_recording(gen2 / "data", spec, results=results)
    (gen2 / "flic_config.yaml").write_text(yaml.safe_dump(_config([1, 2]), sort_keys=False))
    with contextlib.redirect_stdout(io.StringIO()):
        g2 = load_experiment_yaml(gen2, parallel=False, use_disk_cache=False)
    table = g2.opto.qc_table()
    assert set(table["Generation"]) == {2}
    assert table[list(rc.RULE_QC_COLUMNS)].isna().all().all()
    for column in rc.RULE_QC_INTEGER_COLUMNS:
        assert str(table[column].dtype) == "Int64"
    assert g2.opto.events_table()["EndedBy"].isna().all()
    assert not any(f in NEW_FLAGS for f in ", ".join(table["Flags"]).split(", "))
    ## Generation 3 fills them.
    assert e.opto.qc_table()["Restarts"].notna().all()


# ---------------------------------------------------------------------------
# PyMCU's simulated recording and the quiet bench run
# ---------------------------------------------------------------------------

from test_pymcu_simulator import analysed, simulated  # noqa: E402,F401  (fixtures)


@pytest.mark.slow
def test_pymcu_s_recording_keeps_every_verdict_ok_and_fills_the_new_columns(analysed):
    qc = analysed.opto_qc
    assert (qc["Verdict"] == "ok").all(), qc[["DFM", "Group", "Flags"]]
    ## PR5 with Acclimation 10: the closed-loop, fixed-length and ratio columns.
    for column in ("ShortActivations", "WrongLengthActivations", "SeriesMismatches",
                   "IntervalLagSec", "Restarts"):
        assert qc[column].notna().all(), column
        assert (qc[column] == 0).all(), column
    for column in ("OverMaxOnActivations", "CutOffs", "LockoutViolations", "MissedContacts",
                   "NonFeedingDarkSec", "RelightDelayMs"):
        assert qc[column].isna().all(), column
    events = analysed.experiment.opto.events_table()
    assert set(events["EndedBy"]) <= {rc.ENDED_HOLD, rc.ENDED_LENGTH, rc.ENDED_UNKNOWN,
                                      al.BOUNDARY_END}
    assert (events.loc[events["Family"] == pfam.FIXED_LENGTH, "EndedBy"]
            == rc.ENDED_LENGTH).mean() > 0.9


from test_one_treatment_report import quiet  # noqa: E402,F401  (fixture)


def _single_well(root: Path):
    """*root*'s recording as twelve single-well Canton-S chambers, judged
    with optogenetics yes (the run has no light, so auto would skip it)."""
    with contextlib.redirect_stdout(io.StringIO()):
        e = load_experiment_yaml(root, parallel=False, use_disk_cache=False)
    e.optogenetics = "yes"
    return e


def test_the_simulated_quiet_bench_run_raises_no_new_flag(quiet):
    """test_one_treatment_report's quiet run: every well Off, no fly feeding,
    no Program.txt."""
    e = _single_well(quiet)
    assert e.opto.dfm_ids() == [1]
    assert _new_flags(e) == {}
    table = e.opto.qc_table()
    assert (table["Restarts"] == 0).all()
    ## Without Program.txt the rule checks do not run.
    assert table[[c for c in rc.RULE_QC_COLUMNS if c != "Restarts"]].isna().all().all()


@pytest.mark.slow
def test_the_real_quiet_bench_run_raises_no_new_flag(tmp_path):
    if BENCH is None:
        pytest.skip(f"the bench recording {BENCH_NAME} is not here (git-ignored test_data)")
    root = tmp_path / "bench"
    shutil.copytree(BENCH, root / "data")
    cfg = {"global": {"chamber_layout": "single_well",
                      "params": {"feeding_threshold": 20, "feeding_minimum": 10,
                                 "samples_per_second": 5}},
           "dfms": [{"id": 1, "chambers": {c: "Canton-S" for c in range(1, 13)}}]}
    (root / "flic_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    e = _single_well(root)
    assert _new_flags(e) == {}
    table = e.opto.qc_table()
    assert (table["Restarts"] == 0).all()
    ## The ***ALL DFM*** export: every well Off, no Interval times to lag.
    assert table["IntervalLagSec"].isna().all()
