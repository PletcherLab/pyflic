from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyflic import DFM, Parameters


def _make_dfm(params, durations: dict[str, object]) -> DFM:
    """Build a DFM with hand-set durations, bypassing file loading."""
    d = object.__new__(DFM)
    d.id = 7
    d.params = params
    d.durations = durations
    d.raw_df = pd.DataFrame({"Minutes": np.linspace(0.0, 60.0, 100)})
    return d


def _durations() -> dict[str, object]:
    durs: dict[str, object] = {f"W{i}": 0 for i in range(1, 13)}
    # Well 1: three bouts in (0,10], one in (20,30]; supplied out of order.
    durs["W1"] = pd.DataFrame(
        {"Minutes": [3.0, 1.0, 2.0, 25.0], "Duration": [30.0, 10.0, 20.0, 99.0]}
    )
    return durs


def test_two_well_has_separate_ab_columns():
    mm = _make_dfm(Parameters.two_well(), _durations()).moving_median_duration(
        window_min=10, step_min=5
    )
    assert list(mm.columns) == [
        "DFM", "Chamber", "Minutes",
        "MedDurationA", "MedDurationB", "EventsA", "EventsB",
    ]
    assert sorted(mm["Chamber"].unique()) == [1, 2, 3, 4, 5, 6]

    # Data lives only in W1, which is one of chamber 1's two wells. Windows are
    # labelled by endpoint, so window (0,10] -> Minutes 10. Exactly one of A/B
    # carries the W1 median (20, n=3) and the other well is empty (NaN, n=0).
    # A/B assignment depends on pi_direction.
    c1 = mm[(mm["Chamber"] == 1) & np.isclose(mm["Minutes"], 10.0)].iloc[0]
    meds = {float(c1["MedDurationA"]) if not np.isnan(c1["MedDurationA"]) else None,
            float(c1["MedDurationB"]) if not np.isnan(c1["MedDurationB"]) else None}
    assert 20.0 in meds and None in meds
    assert {int(c1["EventsA"]), int(c1["EventsB"])} == {3, 0}

    # window (20,30] -> endpoint 30: single bout 99 -> median 99, n=1 in one well.
    c1b = mm[(mm["Chamber"] == 1) & np.isclose(mm["Minutes"], 30.0)].iloc[0]
    meds_b = {float(c1b["MedDurationA"]) if not np.isnan(c1b["MedDurationA"]) else None,
              float(c1b["MedDurationB"]) if not np.isnan(c1b["MedDurationB"]) else None}
    assert 99.0 in meds_b
    assert {int(c1b["EventsA"]), int(c1b["EventsB"])} == {1, 0}


def test_single_well_window_math_and_columns():
    mm = _make_dfm(Parameters.single_well(), _durations()).moving_median_duration(
        window_min=10, step_min=5
    )
    assert list(mm.columns) == ["DFM", "Chamber", "Minutes", "MedDuration", "Events"]
    assert sorted(mm["Chamber"].unique()) == list(range(1, 13))
    c1 = mm[mm["Chamber"] == 1]

    # window (0,10] -> endpoint 10: bouts 10,20,30 -> median 20, n=3
    r = c1[np.isclose(c1["Minutes"], 10.0)]
    assert float(r["MedDuration"].iloc[0]) == 20.0
    assert int(r["Events"].iloc[0]) == 3

    # window (10,20] -> endpoint 20: empty -> NaN, n=0
    r = c1[np.isclose(c1["Minutes"], 20.0)]
    assert np.isnan(r["MedDuration"].iloc[0])
    assert int(r["Events"].iloc[0]) == 0


def test_empty_chamber_is_all_nan_two_well():
    mm = _make_dfm(Parameters.two_well(), _durations()).moving_median_duration(
        window_min=10, step_min=5
    )
    # Chambers 4-6 have no data in either well.
    empty = mm[mm["Chamber"].isin([4, 5, 6])]
    assert empty[["MedDurationA", "MedDurationB"]].isna().all().all()
    assert (empty[["EventsA", "EventsB"]] == 0).all().all()


def test_invalid_arguments_raise():
    dfm = _make_dfm(Parameters.two_well(), _durations())
    for bad in ({"window_min": 0}, {"window_min": 10, "step_min": 0}):
        try:
            dfm.moving_median_duration(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")


# ---------------------------------------------------------------------------
# Experiment-level plotting (per-chamber and per-treatment).
# ---------------------------------------------------------------------------

from types import SimpleNamespace  # noqa: E402

from pyflic.base.experiment import Experiment  # noqa: E402
from pyflic.base.experiment_design import ExperimentDesign  # noqa: E402
from pyflic.base.treatment import Treatment, TreatmentChamber  # noqa: E402


def _seeded_dfm(did: int, params) -> DFM:
    d = object.__new__(DFM)
    d.id = did
    d.params = params
    d.raw_df = pd.DataFrame({"Minutes": np.linspace(0.0, 120.0, 200)})
    rng = np.random.default_rng(did)
    durs: dict[str, object] = {f"W{i}": 0 for i in range(1, 13)}
    for w in range(1, 13):
        mins = np.sort(rng.uniform(0.0, 120.0, 40))
        durs[f"W{w}"] = pd.DataFrame(
            {"Minutes": mins, "Duration": rng.uniform(1.0, 8.0, 40) + did}
        )
    d.durations = durs
    return d


def _build_exp(params, etype, assignments):
    dfms = {did: _seeded_dfm(did, params) for did in {a[0] for a in assignments}}
    design = ExperimentDesign(dfms=dfms, experiment_type=etype)
    trts: dict[str, Treatment] = {}
    for did, ch, name in assignments:
        trts.setdefault(name, Treatment(name=name)).chambers.append(
            TreatmentChamber(dfm=dfms[did], chamber=SimpleNamespace(index=ch))
        )
    design.treatments = trts
    exp = object.__new__(Experiment)
    exp.dfms = dfms
    exp.design = design
    exp.design_factors = None
    exp.chamber_factors = None
    exp.transform_licks = True
    exp.parallel = False
    return exp


def _two_well_exp():
    return _build_exp(
        Parameters.two_well(),
        "two_well",
        [(1, 1, "Ctrl"), (1, 2, "Ctrl"), (1, 3, "Ctrl"),
         (2, 4, "Drug"), (2, 5, "Drug"), (2, 6, "Drug")],
    )


def test_moving_median_table_collapses_per_chamber():
    exp = _two_well_exp()
    tbl, gcol = exp._moving_median_table(
        window_min=30, step_min=10, range_minutes=(0, 0), two_well_mode="mean_ab"
    )
    assert gcol == "Treatment"
    assert list(tbl.columns) == ["Treatment", "DFM", "Chamber", "Minutes", "MedDuration"]
    counts = {
        t: int(tbl[tbl[gcol] == t][["DFM", "Chamber"]].drop_duplicates().shape[0])
        for t in tbl[gcol].unique()
    }
    assert counts == {"Ctrl": 3, "Drug": 3}
    # mean_ab keeps one value per (chamber, time), same as a single-well pick.
    tbl_a, _ = exp._moving_median_table(
        window_min=30, step_min=10, range_minutes=(0, 0), two_well_mode="A"
    )
    assert len(tbl_a) == len(tbl)


def test_plot_methods_return_ggplot():
    plotnine = pytest.importorskip("plotnine")
    exp = _two_well_exp()
    p_ch = exp.plot_moving_median_duration_by_chamber(window_min=30, step_min=10)
    p_tr = exp.plot_moving_median_duration_by_treatment(window_min=30, step_min=10)
    assert isinstance(p_ch, plotnine.ggplot)
    assert isinstance(p_tr, plotnine.ggplot)


def test_single_well_exp_uses_twelve_chambers():
    exp = _build_exp(
        Parameters.single_well(), "single_well", [(1, c, "All") for c in range(1, 13)]
    )
    tbl, _ = exp._moving_median_table(
        window_min=30, step_min=10, range_minutes=(0, 0), two_well_mode="mean_ab"
    )
    assert sorted(int(c) for c in tbl["Chamber"].unique()) == list(range(1, 13))
