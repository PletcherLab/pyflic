"""The light table (docs/v4-data-plan.md, stream 5: decisions 6, 9, 16, 17,
24 and 31; docs/paradigm-analyses-plan.md, stream S2: decisions 8 to 11).

Generation 3 from the firmware 4.0.0 simulator (``v4_fixtures``), whose
truth the table must reproduce; generation 1 from the existing optogenetic
and Progressive Ratio fixtures; generation 2 from the simulator's
un-updated-firmware writer.  Then everything that shows the table: the
feeding summary's Role / LitSec / Activations, basic analysis and the
script action, the experiment report, the Combined Analysis and the
Project Report, and the per-chamber light-phase summary.  Last, the
Paradigm Families' own columns, against the simulator's truth on each
family's spec and against PyMCU's simulator.
"""

from __future__ import annotations

import math
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from pyflic.base import activation_ledger as al
from pyflic.base import light_summary as ls
from pyflic.base import report_layout as rl
from pyflic.base.yaml_config import load_experiment_yaml

from opto_fixtures import make_opto_dir
from pr_fixtures import make_pr_experiment_dir
from test_pymcu_simulator import analysed, simulated  # noqa: F401 - fixtures
from v4_fixtures import (
    CHOICE_LINKAGE,
    IntervalSpec,
    choice_pr_spec,
    family_specs,
    fixed_ratio_spec,
    one_of_each_spec,
    repeating_open_loop_spec,
    simulate,
    write_gen2_recording,
    write_v4_recording,
    yoked_controls_spec,
)

RECORD_S = 0.2


def _config(dfm_ids, **extra) -> dict:
    params = {"feeding_threshold": 20, "feeding_minimum": 10, "tasting_minimum": 5,
              "tasting_maximum": 20, "samples_per_second": 5, "pi_direction": "left"}
    return {"global": {"chamber_layout": "two_well", "transform_licks": False,
                       "params": params, **extra},
            "dfms": [{"id": int(i), "chambers": {c: ("Ctrl" if c % 2 else "Exp")
                                                 for c in range(1, 7)}}
                     for i in dfm_ids]}


def _recording(root: Path, spec, results=None, *, writer=write_v4_recording,
               config_extra: dict | None = None, **kw) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    writer(root / "data", spec, results=results, **kw)
    cfg = _config([d.dfm_id for d in spec.dfms], **(config_extra or {}))
    (root / "flic_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return root


def _load(root: Path):
    return load_experiment_yaml(root, parallel=False, use_disk_cache=False)


@lru_cache(maxsize=None)
def _one_of_each():
    spec = one_of_each_spec()
    return spec, simulate(spec)


@lru_cache(maxsize=None)
def _choice():
    spec = choice_pr_spec()
    return spec, simulate(spec)


@pytest.fixture(scope="module")
def each(tmp_path_factory):
    spec, results = _one_of_each()
    return _load(_recording(tmp_path_factory.mktemp("each"), spec, results)), results


#: one_of_each_spec: DFM -> its Paradigm type over the whole run.
EACH_TYPES = {1: "OpenLoop", 2: "ClosedLoop", 3: "FixedInterval", 4: "NonFeeding",
              5: ls.MIXED, 6: "FixedRatio"}
ACCLIMATION = ("AcclimationActivations", "AcclimationEndMin", "AcclimationComplete")
RATIO = ("FirmwareEvents", "LastRequirementMet", "ProgressAtEnd", "EventsPerActivation")


# ---------------------------------------------------------------------------
# Generation 3: the firmware's own counts
# ---------------------------------------------------------------------------

def test_one_row_per_chamber_with_the_truths_light_and_activations(each):
    """The plan's acceptance on one_of_each_spec(): one row per chamber, the
    right ParadigmType, LitSec within one record of the simulator's OnMs
    total, Activations equal to the truth."""
    e, results = each
    table = ls.light_summary(e)
    assert list(table.columns) == list(ls.LIGHT_CHAMBER_COLUMNS)
    assert len(table) == 36 and not table.duplicated(["DFM", "Chamber"]).any()
    for row in table.itertuples(index=False):
        truth = results[row.DFM].truth
        a, b = 2 * row.Chamber - 1, 2 * row.Chamber
        assert row.ParadigmType == EACH_TYPES[row.DFM], row
        assert row.Generation == 3 and row.LitSource == ls.SOURCE_ON_MS
        ## Linked wells share one light: the chamber's is the larger total.
        lit = max(truth.on_ms_total[a], truth.on_ms_total[b]) / 1000.0
        assert abs(row.LitSec - lit) <= RECORD_S, row
        assert row.Activations == len(truth.activation_onset_ms[a]) + \
            len(truth.activation_onset_ms[b]), row
    assert table.loc[table.DFM == 5, "Paradigm"].eq(ls.MIXED).all()
    assert table.loc[table.DFM == 2, "Paradigm"].eq("Closed").all()


def test_the_type_specific_columns_are_na_off_type(each):
    e, _results = each
    table = ls.light_summary(e).set_index(["DFM", "Chamber"])
    na = table.isna() | table.apply(lambda c: c.map(lambda v: v is None))
    for dfm in range(1, 7):
        rows = na.xs(dfm, level="DFM")
        ratio = dfm in (5, 6)
        acclimation = dfm == 5                   # PR10 has Acclimation 10; FR5 none
        assert rows[list(RATIO)].any().any() == (not ratio), dfm
        assert rows[list(ACCLIMATION)].any().any() == (not acclimation), dfm
        assert rows["LockedOutContactSec"].all() == (dfm != 3), dfm
        assert rows["UnlitSec"].all() == (dfm != 4) and rows["Suppressions"].all() == (dfm != 4)
        ## Read for every type: Activations, light and pyflic's events.
        assert not rows[["Activations", "LitSec", "LightOnsets", "EventsA"]].any().any()


def test_each_well_matches_the_firmware(each):
    e, results = each
    wells = ls.light_wells(e)
    assert list(wells.columns) == list(ls.LIGHT_WELL_COLUMNS)
    assert len(wells) == 72
    for row in wells.itertuples(index=False):
        truth = results[row.DFM].truth
        assert row.LitSec == pytest.approx(truth.on_ms_total[row.Well] / 1000.0)
        assert row.Activations == len(truth.activation_onset_ms[row.Well])
        if row.DFM == 5 and row.Well % 2:
            ## PR10's trigger wells: Acclimation from the Phase, the last
            ## Requirement from the row before the Activation.
            end = truth.acclimation_end_ms[row.Well]
            assert end is not None and row.AcclimationComplete is True
            assert abs(row.AcclimationEndMin * 60000.0 - end) <= 2 * 200
            assert row.AcclimationActivations == 10
            met = [m for m in truth.requirement_met[row.Well] if m is not None]
            assert row.LastRequirementMet == met[-1]
            assert row.FirmwareEvents >= sum(met)
            assert row.EventsPerActivation == pytest.approx(row.FirmwareEvents / len(met))
        if row.DFM == 6 and row.Well % 2:
            assert row.LastRequirementMet == 5      # FR5: a constant Requirement


def test_without_a_program_the_type_dependent_columns_are_na(tmp_path):
    spec, results = _choice()
    e = _load(_recording(tmp_path / "exp", spec, results, program=False))
    assert e.opto_program is None and e.is_optogenetic
    table = ls.light_summary(e)
    assert table["ParadigmType"].isna().all()
    for column in (*RATIO, *ACCLIMATION, "LockedOutContactSec", "UnlitSec", "Suppressions"):
        assert table[column].isna().all(), column
    ## The rows still name their Paradigm, and the firmware still counted.
    assert set(table["Paradigm"]) == {ls.MIXED}
    truth = results[1].truth
    first = table[(table.DFM == 1) & (table.Chamber == 1)].iloc[0]
    assert first.Activations == len(truth.activation_onset_ms[1])
    ## Roles from the data: the dead-LED chamber never lit.
    assert first.Role == "Self"
    assert table[(table.DFM == 1) & (table.Chamber == 4)].Role.item() == "Unlit"


def test_a_window_holds_its_own_paradigm(tmp_path):
    """choice_pr_spec runs Closed for five minutes, then PR10: a window
    inside PR10 names it alone, the whole run is mixed."""
    spec, results = _choice()
    e = _load(_recording(tmp_path / "exp", spec, results))
    late = ls.light_summary(e, range_minutes=(10, 30))
    assert set(late["Paradigm"]) == {"PR10"} and set(late["ParadigmType"]) == {
        "ProgressiveRatio"}
    assert set(late["ProgramInterval"]) == {2}
    early = ls.light_summary(e, range_minutes=(1, 4))
    assert set(early["ParadigmType"]) == {"ClosedLoop"}
    assert early[list(RATIO)].isna().all().all()
    ## The never-training chamber stays in Acclimation: no end, not complete.
    never = late[(late.DFM == 1) & (late.Chamber == 5)].iloc[0]
    assert never.AcclimationComplete is False and pd.isna(never.AcclimationEndMin)


# ---------------------------------------------------------------------------
# Generations 1 and 2: OptoCol1 only
# ---------------------------------------------------------------------------

def test_generation_1_reads_optocol1(tmp_path):
    e = _load(make_opto_dir(tmp_path / "exp"))
    wells = ls.light_wells(e).set_index("Well")
    assert set(wells["LitSource"]) == {ls.SOURCE_OPTOCOL1}
    assert set(wells["Generation"]) == {1}
    for column in ("Activations", "ActivationsPerHour", *RATIO, *ACCLIMATION):
        assert wells[column].isna().all(), column
    lights = e.dfms[1].lights_df
    for well in range(1, 13):
        assert wells.loc[well, "LitSec"] == pytest.approx(lights[f"W{well}"].sum() / 5.0)
    ## Well 5 is lit throughout, well 6 the first half: one onset each.
    assert wells.loc[5, "LitFraction"] == pytest.approx(1.0, abs=1e-3)
    assert wells.loc[5, "LightOnsets"] == 1 and wells.loc[6, "LightOnsets"] == 1
    assert wells.loc[1, "LightOnsets"] == 20         # twenty closed-loop bouts
    assert wells.loc[8, "LitSec"] == 0 and wells.loc[8, "LightOnsets"] == 0
    ## The program's clock schedule names the Paradigm on generation 1.
    assert set(wells["ParadigmType"]) == {"ClosedLoop"}
    assert set(wells["ProgramInterval"]) == {1}


def test_generation_2_has_light_and_no_firmware_columns(tmp_path):
    spec, results = _choice()
    e = _load(_recording(tmp_path / "exp", spec, results, writer=write_gen2_recording))
    table = ls.light_summary(e)
    assert set(table["Generation"]) == {2} and set(table["LitSource"]) == {"OptoCol1"}
    assert table["Activations"].isna().all() and table["FirmwareEvents"].isna().all()
    assert (table["LitSec"] > 0).any()
    assert ls.has_generation_2(table)
    assert ls.no_training_information(e)


# ---------------------------------------------------------------------------
# Facets
# ---------------------------------------------------------------------------

def test_the_facet_tables_partition_the_whole(tmp_path):
    spec, results = _choice()
    e = _load(_recording(tmp_path / "exp", spec, results,
                         config_extra={"facet_cutoffs": [10]}))
    whole = ls.light_summary(e).set_index(["DFM", "Chamber"])
    facet = ls.light_summary_facet(e)
    assert list(facet.columns[:2]) == ["Facet", "FacetRange"]
    assert list(dict.fromkeys(facet["Facet"])) == e.facet_labels()
    summed = facet.groupby(["DFM", "Chamber"])[["LitSec", "Activations"]].sum()
    pd.testing.assert_frame_equal(summed, whole[["LitSec", "Activations"]],
                                  check_exact=False, atol=1e-9)
    wells = ls.light_wells_facet(e)
    assert len(wells) == 2 * len(ls.light_wells(e))
    ## Unfaceted: nothing, as feeding_summary_facet.
    e.facet_cutoffs = None
    e.config["global"].pop("facet_cutoffs", None)
    assert ls.light_summary_facet(e).empty


def test_per_dfm_windows_are_used_when_the_experiment_has_them(tmp_path, monkeypatch):
    """Program Facets (stream 5b) give each DFM its own windows through
    ``facet_windows_for``; the light table takes them when it exists."""
    spec, results = _choice()
    e = _load(_recording(tmp_path / "exp", spec, results))
    seen: list[int] = []

    def facet_windows_for(self, dfm_id):
        seen.append(dfm_id)
        return [("Closed", (0.0, 5.0)), ("PR10", (5.0, float("inf")))]

    monkeypatch.setattr(type(e), "facet_windows_for", facet_windows_for, raising=False)
    table = ls.light_summary_facet(e)
    assert seen == [1, 2]
    assert list(dict.fromkeys(table["Facet"])) == ["Closed", "PR10"]
    assert set(table.loc[table.Facet == "Closed", "ParadigmType"]) == {"ClosedLoop"}
    assert set(table.loc[table.Facet == "PR10", "ParadigmType"]) == {"ProgressiveRatio"}


def test_a_facet_of_several_windows_pools_them(tmp_path, monkeypatch):
    """A Program Facet pooling a Repeating program's Cycles is a label with
    a list of (a, b] windows (stream 5b's ``facet_windows_for``); the light
    table takes the list as one window, as the feeding summary does."""
    spec, results = _choice()
    e = _load(_recording(tmp_path / "exp", spec, results))
    windows = [(1.0, 3.0), (6.0, 9.0), (20.0, float("inf"))]
    pooled = ls.light_summary(e, range_minutes=windows).set_index(["DFM", "Chamber"])
    parts = [ls.light_summary(e, range_minutes=(a, 0.0 if b == float("inf") else b))
             .set_index(["DFM", "Chamber"]) for a, b in windows]
    for column in ("LitSec", "Activations", "LightOnsets", "FirmwareEvents", "WindowMinutes",
                   "EventsA"):
        total = sum(p[column].fillna(0) for p in parts)
        assert (pooled[column].fillna(0) - total).abs().max() < 1e-9, column
    assert set(pooled["Paradigm"]) == {ls.MIXED}
    ## The feeding summary's light columns take the same list.
    fs = e.feeding_summary()
    redone = ls.with_light_columns(e, fs, windows).set_index(["DFM", "Chamber"])
    for column in ("LitSec", "Activations"):
        assert (redone[column] - pooled.loc[redone.index, column]).abs().max() < 1e-9

    def facet_windows_for(self, dfm_id):
        return [("Early", [(1.0, 3.0), (6.0, 9.0)]), ("Late", [(20.0, float("inf"))])]

    monkeypatch.setattr(type(e), "facet_windows_for", facet_windows_for, raising=False)
    facet = ls.light_summary_facet(e)
    assert list(dict.fromkeys(facet["Facet"])) == ["Early", "Late"]
    early = facet[facet.Facet == "Early"].set_index(["DFM", "Chamber"])
    assert early["FacetRange"].iloc[0].count("(") == 2
    expected = (parts[0]["LitSec"] + parts[1]["LitSec"])
    assert (early["LitSec"] - expected.loc[early.index]).abs().max() < 1e-9


def _clipped_load(root: Path, start: float):
    return load_experiment_yaml(root, parallel=False, use_disk_cache=False,
                                range_minutes=(start, 0))


def test_a_light_on_at_a_slices_first_row_is_no_onset_of_it(tmp_path):
    """Review C11, generations 1 and 2: a range's first row compares with
    the row the clip cut off, so a light already on is no onset; the counts
    are those of the same window of the whole recording."""
    root = make_opto_dir(tmp_path / "exp")
    whole = ls.light_wells(_load(root), range_minutes=(10, 0)).set_index("Well")
    part = ls.light_wells(_clipped_load(root, 10)).set_index("Well")
    ## Well 5 is lit throughout and well 6 the first half: on at minute 10.
    assert part.loc[5, "LightOnsets"] == whole.loc[5, "LightOnsets"] == 0
    assert part.loc[6, "LightOnsets"] == whole.loc[6, "LightOnsets"] == 0
    pd.testing.assert_series_equal(part["LightOnsets"], whole["LightOnsets"])


def test_an_activation_on_a_slices_first_row_is_counted(tmp_path):
    """Review C11, generation 3: an Activation counted on a range's first row
    is the range's, as in the same window of the whole recording."""
    spec, results = _choice()
    root = _recording(tmp_path / "exp", spec, results)
    e = _load(root)
    state = e.dfms[1].paradigm
    onsets = state.activation_onsets(1)
    row = int(onsets[len(onsets) // 2])
    start = float(state.minutes[row - 1])               # the range starts on the onset
    columns = ["Activations", "LightOnsets", "LitSec"]
    whole = ls.light_wells(e, range_minutes=(start, 0)).set_index(["DFM", "Well"])[columns]
    part = ls.light_wells(_clipped_load(root, start)).set_index(["DFM", "Well"])[columns]
    pd.testing.assert_frame_equal(part, whole, check_exact=False, atol=1e-9)
    assert part.loc[(1, 1), "Activations"] >= 1


# ---------------------------------------------------------------------------
# Progressive Ratio keeps its own Role and windows
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pr(tmp_path_factory):
    return _load(make_pr_experiment_dir(tmp_path_factory.mktemp("pr")))


def test_progressive_ratio_keeps_its_role_and_group_windows(pr):
    assert ls.type_owns_role(pr)
    table = ls.light_summary(pr)
    assert set(table["Role"]) == {"paired", "yoked"}
    for row in table.itertuples(index=False):
        assert row.Role == pr.role_of(row.DFM, row.Chamber).role
    facet = ls.light_summary_facet(pr)
    assert set(facet["Facet"]) == {"Training", "Test"}
    assert list(dict.fromkeys(facet["Facet"])) == ["Training", "Test"]
    ## Each Chamber Group splits at its own training end.
    for (dfm, chamber), rows in facet.groupby(["DFM", "Chamber"]):
        group = pr.role_of(dfm, chamber).group
        windows = dict(pr.group_windows(dfm, group))
        assert set(rows["Facet"]) == set(windows)
    fs = pr.feeding_summary()
    assert {"Group", "Role", "TrainingMinutes", "LitSec", "Activations"} <= set(fs.columns)
    assert set(fs["Role"]) == {"paired", "yoked"}
    assert fs["Activations"].isna().all()           # generation 1: no Activations
    ## For comparing light behaviour, a yoked fly's light is its partner's.
    kept = ls.behaviour_rows(table)
    assert set(kept["Role"]) == {"paired"} and len(kept) == len(table) // 2


def test_progressive_ratio_facet_rows_carry_their_windows_light(pr):
    """The Training / Test rows have LitSec and Activations where the
    whole-recording summary has them, each over the row's own window."""
    facet = pr.feeding_summary_facet()
    assert list(facet.columns[:2]) == ["Facet", "FacetRange"]
    assert list(facet.columns[2:]) == list(pr.feeding_summary().columns)
    light = ls.light_summary_facet(pr).set_index(["Facet", "DFM", "Chamber"])
    for row in facet.itertuples(index=False):
        assert row.LitSec == pytest.approx(light.loc[(row.Facet, row.DFM, row.Chamber),
                                                     "LitSec"])
    assert facet["Activations"].isna().all()          # generation 1: no Activations
    assert set(facet["Role"]) == {"paired", "yoked"}


# ---------------------------------------------------------------------------
# The feeding summary's light columns
# ---------------------------------------------------------------------------

def test_the_feeding_summary_carries_role_litsec_and_activations(each):
    e, results = each
    fs = e.feeding_summary()
    columns = list(fs.columns)
    assert columns.index("Role") == columns.index("Chamber") + 1
    assert columns.index("LitSec") == columns.index("OptoOn_sec_B") + 1
    assert columns.index("Activations") == columns.index("LitSec") + 1
    table = ls.light_summary(e).set_index(["DFM", "Chamber"])
    for row in fs.itertuples(index=False):
        assert row.Role == table.loc[(row.DFM, row.Chamber), "Role"]
        assert row.LitSec == pytest.approx(table.loc[(row.DFM, row.Chamber), "LitSec"])
        assert row.Activations == table.loc[(row.DFM, row.Chamber), "Activations"]
    ## A window computes its own.
    window = e.feeding_summary(range_minutes=(0, 10))
    assert (window["Activations"] <= fs["Activations"].to_numpy()).all()


def test_the_light_columns_are_redone_on_a_cache_hit(each):
    """A summary read back from the disk cache carries whatever its light
    columns said when it was written."""
    e, _results = each
    fresh = e.feeding_summary()
    key = ((0.0, 0.0), bool(e.transform_licks))
    stale = e._feeding_summary_cache[key].copy()
    stale["Role"] = "Stale"
    stale["LitSec"] = -1.0
    e._feeding_summary_cache[key] = stale
    try:
        again = e.feeding_summary()
    finally:
        e._feeding_summary_cache.pop(key)
    pd.testing.assert_frame_equal(again, fresh)


def test_a_dark_experiment_gains_nothing(tmp_path):
    e = _load(make_opto_dir(tmp_path / "dark", program=None, light_cases=False))
    assert not e.is_optogenetic
    assert not set(ls.FEEDING_COLUMNS) & set(e.feeding_summary().columns)
    assert ls.light_summary(e).empty


# ---------------------------------------------------------------------------
# Outputs: basic analysis, the script action, the experiment report
# ---------------------------------------------------------------------------

def test_basic_analysis_writes_the_four_tables(tmp_path):
    root = make_opto_dir(tmp_path / "exp")
    cfg = yaml.safe_load((root / "flic_config.yaml").read_text())
    cfg["global"]["facet_cutoffs"] = [30]
    (root / "flic_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    e = _load(root)
    result = e.execute_basic_analysis(skip_qc=True)
    analysis = root / "analysis"
    for key, name in ls.FILES.items():
        assert result[key] == (analysis / name).resolve(), key
    chambers = pd.read_csv(analysis / "light_summary.csv")
    assert list(chambers.columns) == list(ls.LIGHT_CHAMBER_COLUMNS)
    facet = pd.read_csv(analysis / "light_summary_facet.csv")
    assert set(facet["Facet"]) == set(e.facet_labels())
    fs = pd.read_csv(analysis / "feeding_summary.csv")
    assert {"Role", "LitSec", "Activations"} <= set(fs.columns)


def test_the_script_action_writes_and_is_gated(tmp_path):
    from pyflic.base.script_editor.actions import get_action
    from pyflic.base.script_editor.runner import run_experiment_script

    assert get_action("light_summary") is not None
    e = _load(make_opto_dir(tmp_path / "exp"))
    logs: list[str] = []
    run_experiment_script(e, {"name": "t", "steps": [{"action": "light_summary"}]},
                          log=logs.append)
    assert (Path(e.analysis_dir) / "light_summary.csv").is_file()
    assert (Path(e.analysis_dir) / "light_wells.csv").is_file()
    assert not (Path(e.analysis_dir) / "light_summary_facet.csv").exists()
    dark = _load(make_opto_dir(tmp_path / "dark", program=None, light_cases=False))
    logs.clear()
    run_experiment_script(dark, {"name": "t", "steps": [{"action": "light_summary"}]},
                          log=logs.append)
    assert any("[Skip] light_summary needs an optogenetic experiment" in m for m in logs)


def _headings(doc) -> list[str]:
    return [b.text for b in doc.blocks if isinstance(b, rl.Heading)]


def test_the_experiment_report_has_light_behaviour(each):
    from pyflic.base.pdf_report import build_experiment_report

    e, _results = each
    doc = build_experiment_report(e, metrics=())
    headings = _headings(doc)
    assert "Light behaviour" in headings
    assert headings.index("Light behaviour") > headings.index("Consumption")
    cover = next(b for b in doc.blocks if isinstance(b, rl.KeyValues))
    rows = dict(cover.rows)
    assert rows["Generation"].startswith("3 — 90 or 102 columns, firmware 4.0.0")
    assert rows["Lit time from"] == "OnMs, exact"


def test_a_dark_member_report_has_no_light_section(tmp_path):
    from pyflic.base.pdf_report import build_experiment_report

    e = _load(make_opto_dir(tmp_path / "dark", program=None, light_cases=False))
    doc = build_experiment_report(e, metrics=())
    assert "Light behaviour" not in _headings(doc)
    rows = dict(next(b for b in doc.blocks if isinstance(b, rl.KeyValues)).rows)
    assert rows["Generation"].startswith("1 — 27 columns")
    assert "Lit time from" not in rows


# ---------------------------------------------------------------------------
# Decision 24: the light-phase summary is per chamber
# ---------------------------------------------------------------------------

def test_the_light_phase_is_each_chambers_own(tmp_path):
    """Changed on purpose (decision 24): a chamber is in "light" when its own
    wells are lit, not whenever any LED of the DFM is.  In the generation-1
    opto fixture well 5 is lit throughout, so before this every chamber was
    in "light" for the whole hour."""
    from pyflic.base.analytics import light_phase_summary

    e = _load(make_opto_dir(tmp_path / "exp"))
    df = light_phase_summary(e)
    light = df[df.Phase == "light"].set_index("Chamber")["PhaseSeconds"]
    dark = df[df.Phase == "dark"].set_index("Chamber")["PhaseSeconds"]
    assert light.loc[5] == pytest.approx(3600.0)
    assert light.loc[6] == pytest.approx(1800.0)
    assert light.loc[8] == 0.0 and dark.loc[8] == pytest.approx(3600.0)
    assert light.loc[1] == pytest.approx(60.0)       # twenty bouts of 15 samples
    lights = e.dfms[1].lights_df
    for chamber in range(1, 13):
        assert light.loc[chamber] == pytest.approx(lights[f"W{chamber}"].sum() / 5.0)


def test_the_light_phase_reads_onms_on_generation_3(tmp_path):
    from pyflic.base.analytics import light_phase_summary

    spec = fixed_ratio_spec(minutes=10.0)
    e = _load(_recording(tmp_path / "exp", spec))
    df = light_phase_summary(e)
    on_ms = e.dfms[1].paradigm.on_ms
    light = df[df.Phase == "light"].set_index("Chamber")["PhaseSeconds"]
    for chamber in range(1, 7):
        wells = [2 * chamber - 2, 2 * chamber - 1]
        assert light.loc[chamber] == pytest.approx((on_ms[:, wells] > 0).any(axis=1).sum() / 5.0)


# ---------------------------------------------------------------------------
# The Project
# ---------------------------------------------------------------------------

def _project(root: Path, members: dict, *, design_extra: dict | None = None):
    """A Project of one-DFM members, each recorded by ``write(data_dir)``,
    every member's basic analysis run."""
    from pyflic.base.project import Project

    design = None
    for name, write in members.items():
        member = root / name
        member.mkdir(parents=True)
        write(member / "data")
        cfg = _config([1])
        design = cfg.pop("global")
        (member / "flic_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    design.update(design_extra or {})
    (root / "project.yaml").write_text(yaml.safe_dump(
        {"name": "proj", "design": {"global": design}}, sort_keys=False))
    project = Project(root)
    for name in project.member_names:
        project.load_member(name, parallel=False,
                            use_disk_cache=False).execute_basic_analysis(skip_qc=True)
    return project


@pytest.fixture(scope="module")
def gen3_project(tmp_path_factory):
    root = tmp_path_factory.mktemp("proj") / "proj"
    return _project(root, {
        "rep1": lambda data: write_v4_recording(data, fixed_ratio_spec(minutes=10.0, seed=7)),
        "rep2": lambda data: write_v4_recording(data, fixed_ratio_spec(minutes=10.0, seed=8)),
    }, design_extra={"facet_cutoffs": [5]})


def test_the_combined_analysis_stacks_the_light_tables(gen3_project):
    project = gen3_project
    written = {Path(p).name for p in project.build_combined_analysis()["written"]}
    assert {"proj_LightSummary.csv", "proj_LightWells.csv", "proj_LightSummary_Facet.csv",
            "proj_LightWells_Facet.csv"} <= written
    stacked = pd.read_csv(Path(project.analysis_path) / "proj_LightSummary.csv")
    assert stacked.columns[0] == "Experiment"
    assert set(stacked["Experiment"]) == {"rep1", "rep2"} and len(stacked) == 12
    text = (Path(project.analysis_path) / "proj_Stats.txt").read_text(encoding="utf-8")
    assert "Light behaviour (optogenetic members; one observation per chamber)" in text
    block = text[text.index("Light behaviour"):]
    for metric in ("Activations", "LitSec", "FirmwareEvents"):
        assert metric in block
    rows = project.light_comparison_rows()
    assert {r["phase"] for r in rows} == {"0-5 min", "5+ min"}
    assert {"Activations", "LitSec", "FirmwareEvents"} <= {r["metric"] for r in rows}
    assert {r["n_a"] + r["n_b"] for r in rows} == {12}


def test_the_project_report_has_light_behaviour(gen3_project):
    from pyflic.base.project_report import build_project_report, member_table

    project = gen3_project
    project.build_combined_analysis()
    doc = build_project_report(project)
    headings = _headings(doc)
    assert "Light behaviour" in headings
    assert headings.index("Light behaviour") > headings.index("Statistics")
    members = member_table(project)
    assert list(members["Generation"]) == ["3", "3"] and "Note" not in members.columns


def test_a_generation_2_member_is_named(tmp_path):
    from pyflic.base.project_report import member_table

    project = _project(tmp_path / "proj", {
        "new": lambda data: write_v4_recording(data, fixed_ratio_spec(minutes=6.0)),
        "old": lambda data: write_gen2_recording(data, fixed_ratio_spec(minutes=6.0)),
    })
    assert project.generation_2_members() == ["old"]
    assert project.member_generations("new") == [3]
    project.build_combined_analysis()
    text = (Path(project.analysis_path) / "proj_Stats.txt").read_text(encoding="utf-8")
    flagged = text[text.index("Flagged Chamber Groups"):text.index("Metric")]
    assert f"old [generation 2]: {ls.NO_TRAINING_INFORMATION}" in flagged
    assert "new [generation 2]" not in flagged
    members = member_table(project).set_index("Member")
    assert members.loc["old", "Generation"] == "2"
    assert members.loc["old", "Note"] == ls.NO_TRAINING_INFORMATION
    assert members.loc["new", "Note"] == ""


def test_a_progressive_ratio_member_on_old_firmware_is_named_by_its_facet(tmp_path):
    """A generation-2 Progressive Ratio member has one Facet, Recording
    (decision 15), whether or not it wrote a light table."""
    from pyflic.base.project import Project

    root = tmp_path / "proj"
    for name in ("a", "b"):
        (root / name / "analysis").mkdir(parents=True)
        (root / name / "flic_config.yaml").write_text(
            yaml.safe_dump({"dfms": [{"id": 1, "chambers": {1: "Ctrl"}}]}))
    pd.DataFrame({"Facet": ["Recording"], "DFM": [1]}).to_csv(
        root / "b" / "analysis" / "feeding_summary_facet.csv", index=False)
    (root / "project.yaml").write_text(yaml.safe_dump(
        {"name": "proj", "design": {"global": {"chamber_layout": "two_well"}}}))
    assert Project(root).generation_2_members() == ["b"]


def test_no_training_information_asks_the_type_first():
    class Asks:
        dfms: dict = {}

        def __init__(self, answer):
            self.answer = answer

        def training_information(self):
            return self.answer

    assert ls.no_training_information(Asks("none"))
    assert ls.no_training_information(Asks("no training information: recorded with ..."))
    assert not ls.no_training_information(Asks("phase"))

    class Dfm:
        training_info = "none"

    class Plain:
        dfms = {1: Dfm()}

    assert ls.no_training_information(Plain())


def test_the_light_figures_draw_from_the_stacked_table(gen3_project):
    from pyflic.base import pubfigures

    project = gen3_project
    project.build_combined_analysis()
    assert pubfigures.has_light_table(project)
    offered = pubfigures.plots_for_layout("two_well", None, light=True)
    for plot_id in ("faceted_activations", "faceted_litsec", "faceted_firmwareevents",
                    "faceted_acclimationendmin"):
        assert plot_id in offered and plot_id not in pubfigures.plots_for_layout("two_well")
        assert pubfigures.source_of(plot_id) == pubfigures.SOURCE_LIGHT
    source = pubfigures.frame_for("faceted_activations", None, None, project)
    data = pubfigures.faceted_data(source, "Activations", pubfigures.facet_order(source))
    assert set(data["Phase"].astype(str)) == {"0-5 min", "5+ min"}
    ## The general Role is a filter, never a panel.
    assert "Role" not in data.columns
    spec = pubfigures.default_spec("faceted_activations")
    pubfigures.build_figure("faceted_activations", data, spec, pubfigures.PlotStyle())


# ---------------------------------------------------------------------------
# Stream S2: the Paradigm Families' columns (docs/paradigm-analyses-plan.md)
# ---------------------------------------------------------------------------
# Each column against a computation from the simulator's truth: its own
# Activations (onset, the Well's own light, Phase, the Paradigm in force),
# its Acclimation end and its schedule.  Which Activations are judgeable
# (not Overlap, no Boundary) is the ledger's call, as the plan defines it;
# the truth gives their times.  pyflic's licks are the one input the truth
# does not have (LockedOutContactSec).

#: The new columns, and the two decision 10 redefines.
FAMILY_NEW = ("MainActivations", "MedianActivationSec", "CutOffs", "MaxActivations",
              "RewardEfficiency", "MedianReturnSec", "MedianSuppressionSec",
              "LicksPerMinLit", "LicksPerMinDark", "LightFeedingIndex")
REDEFINED = ("LockedOutContactSec", "Suppressions")
#: The columns computed per well from the truth.
TRUTH_COLUMNS = ("MainActivations", "MedianActivationSec", "CutOffs", "MaxActivations",
                 "RewardEfficiency", "MedianReturnSec", "LockedOutContactSec",
                 "Suppressions", "MedianSuppressionSec")
#: The chamber's medians pool its wells' values.
POOLED = ("MedianActivationSec", "MedianReturnSec", "MedianSuppressionSec")
TOL_MS = ls.DEFAULT_ACTIVATION_TOLERANCE_MS
RECORD_MS = 200
S2_MINUTES = 12.0
ACCLIMATION_TYPES = ("FixedInterval", "ProgressiveRatio", "FixedRatio")


def _mutual_fixed_interval_spec():
    """family_specs' FI10s DFM with both wells of each chamber Trigger
    Wells, linked: a well's partner lights it, during its own lockout too,
    which the old LockedOutContactSec could not see (decision 10)."""
    spec = family_specs(S2_MINUTES)["fixed_interval"]
    dfm = spec.dfms[0]
    both = IntervalSpec(dfm.intervals[0].paradigm, True, (20,) * 12, S2_MINUTES)
    return replace(spec, dfms=(replace(dfm, intervals=(both,), linkage=CHOICE_LINKAGE),))


#: The specs the truth is computed on: every family_specs entry,
#: fixed_ratio_spec and yoked_controls_spec, and the linked FixedInterval.
S2_SPECS = {
    **{key: (lambda key=key: family_specs(S2_MINUTES)[key])
       for key in ("closed_max", "fixed_interval", "non_feeding", "exponential", "restart")},
    "fixed_ratio": lambda: fixed_ratio_spec(minutes=S2_MINUTES),
    "yoked_controls": lambda: yoked_controls_spec(minutes=S2_MINUTES),
    "mutual_fixed_interval": _mutual_fixed_interval_spec,
}


@lru_cache(maxsize=None)
def _s2_simulated(name: str):
    spec = S2_SPECS[name]()
    return spec, simulate(spec)


@pytest.fixture(scope="module")
def s2_loaded(tmp_path_factory):
    """``get(name) -> (spec, results, experiment)``, each loaded once."""
    cache: dict[str, tuple] = {}

    def get(name: str):
        if name not in cache:
            spec, results = _s2_simulated(name)
            root = _recording(tmp_path_factory.mktemp(name), spec, results)
            cache[name] = (spec, results, _load(root))
        return cache[name]

    return get


def _linked(linkage) -> dict[int, tuple[int, ...]]:
    groups: dict[int, list[int]] = {}
    for well, label in enumerate(linkage, start=1):
        groups.setdefault(label, []).append(well)
    return {well: tuple(groups[label]) for well, label in enumerate(linkage, start=1)}


def _resets(truth) -> list[int]:
    """The ms the firmware reset every Well: an instruction naming another
    Paradigm than the last, and a restart."""
    resets, last = [], None
    for ms, _interval, _cycle, name in truth.schedule:
        if name != last:
            resets.append(ms)
        last = name
    for down, resume, _missing in truth.restarts_ms:
        resets += [down, resume]
    return sorted(resets)


def _truth_activations(spec, truth, well: int) -> pd.DataFrame:
    """The truth's Activations of *well*, in order: onset, the end of its
    own light (its lit span, cut at the next onset), its length, Phase,
    state run and the Paradigm in force at the onset."""
    onsets = list(truth.activation_onset_ms[well])
    starts = np.array([a for a, _b in truth.lit_spans_ms[well]], dtype=np.int64)
    ends = np.array([b for _a, b in truth.lit_spans_ms[well]], dtype=np.int64)
    resets = _resets(truth)
    rows = []
    for k, t0 in enumerate(onsets):
        end = int(ends[np.searchsorted(starts, t0, side="right") - 1])
        if k + 1 < len(onsets):
            end = min(end, onsets[k + 1])
        name = [n for ms, _i, _c, n in truth.schedule if ms <= t0][-1]
        rows.append({"onset": t0, "end": end, "length": end - t0,
                     "phase": truth.activation_phase[well][k],
                     "run": int(np.searchsorted(resets, t0, side="right")),
                     "paradigm": spec.paradigms[name]})
    return pd.DataFrame(rows, columns=["onset", "end", "length", "phase", "run", "paradigm"])


def _truth_columns(spec, dfm_spec, result, e, book, well: int) -> tuple[dict, dict, dict]:
    """``(values, pools, slack)``: :data:`TRUTH_COLUMNS` of *well* over the
    whole recording from the truth, ``NaN`` off its family; the values its
    chamber's medians pool (ms); and how far pyflic may be from each.

    pyflic is exact but where the ledger can time an onset only to the
    record: the LED already lit when the Activation began, by the well's own
    last one (held contact under a lockout shorter than ``ActivationMs``) or
    by its partner's.  Then a return time is good to one record, and the
    ends of its lockout and its own light, so LockedOutContactSec to two
    lick samples per such Activation."""
    truth = result.truth
    acts = _truth_activations(spec, truth, well)
    mine = book[book["Well"] == well].reset_index(drop=True)
    assert len(mine) == len(acts), (dfm_spec.dfm_id, well)
    judged = (~mine["Overlap"].to_numpy(bool)) & (mine["Boundary"] == al.BOUNDARY_NONE).to_numpy(bool)
    slack = TOL_MS + np.where(mine["Timing"] == al.TIMING_RECORD, RECORD_MS, 0)
    types = np.array([p.type for p in acts["paradigm"]], dtype=object)
    max_on = np.array([p.max_on_ms for p in acts["paradigm"]], dtype=float)
    length = acts["length"].to_numpy(float)
    onset = acts["onset"].to_numpy(float)
    end = acts["end"].to_numpy(float)
    main = (acts["phase"] == "M").to_numpy(bool)
    runs = acts["run"].to_numpy()
    levels = [spec.paradigms[iv.paradigm] for iv in dfm_spec.intervals
              if isinstance(iv.thresholds[well - 1], int)]
    acclimating = any(p.acclimation > 0 and p.type in ACCLIMATION_TYPES for p in levels)
    out = dict.fromkeys(TRUTH_COLUMNS, np.nan)
    pools = {c: np.zeros(0) for c in POOLED}
    slack_of = dict.fromkeys(TRUTH_COLUMNS, 0.0)
    record_timed = mine["OnsetMs"].to_numpy(float) != acts["onset"].to_numpy(float)

    def median_s(values) -> float:
        return float(np.median(values)) / 1000.0 if len(values) else np.nan

    if acclimating or any(p.type == "ClosedLoop" for p in levels):
        pools["MedianActivationSec"] = length[judged & ((types == "ClosedLoop") | ~main)]
        out["MedianActivationSec"] = median_s(pools["MedianActivationSec"])
    if any(p.type == "ClosedLoop" and p.max_on_ms > 0 for p in levels):
        cut = judged & main & (types == "ClosedLoop") & (max_on > 0) \
            & (np.abs(length - max_on) <= slack)
        out["CutOffs"] = float(cut.sum())
    if acclimating:
        out["MainActivations"] = float(main.sum())
    fixed = [p for p in levels if p.type == "FixedInterval"]
    if fixed:
        (p,) = fixed
        spacing = max(p.lockout_ms, p.activation_ms)
        n_rec = int(round(spec.duration_min * 60 * 1000 / RECORD_MS))
        ## Main from the instruction, or the well's Acclimation end, as the
        ## record's last millisecond shows it (no restart on these specs, so
        ## record n is row n - 1).
        start = truth.acclimation_end_ms[well] if p.acclimation else truth.schedule[0][0]
        main_records = np.arange(1, n_rec + 1) * RECORD_MS - 1 >= start
        out["MaxActivations"] = float(math.floor(main_records.sum() * RECORD_MS / spacing))
        earned = main & (types == "FixedInterval")
        out["RewardEfficiency"] = earned.sum() / out["MaxActivations"]
        pairs = earned[1:] & earned[:-1] & (runs[1:] == runs[:-1])
        pools["MedianReturnSec"] = (np.diff(onset) - spacing)[pairs]
        out["MedianReturnSec"] = median_s(pools["MedianReturnSec"])
        locked = np.zeros(n_rec, dtype=bool)
        for t0 in onset[earned]:
            locked[int(t0) // RECORD_MS: math.ceil((t0 + p.lockout_ms) / RECORD_MS)] = True
        own = np.zeros(n_rec, dtype=bool)
        for a, b in truth.lit_spans_ms[well]:
            own[a // RECORD_MS: math.ceil(b / RECORD_MS)] = True
        licks = e.dfms[dfm_spec.dfm_id].lick_df[f"W{well}"].to_numpy(bool)
        out["LockedOutContactSec"] = float((licks & locked & main_records & ~own).sum()) / 5.0
        untimed = int((earned & record_timed).sum())
        slack_of["MedianReturnSec"] = RECORD_MS / 1000.0 if untimed else 0.0
        slack_of["LockedOutContactSec"] = 2 * untimed / 5.0
    if any(p.type == "NonFeeding" for p in levels):
        by_max = (max_on > 0) & (np.abs(length - max_on) <= slack)
        suppressed = judged & (types == "NonFeeding") & ~by_max
        out["Suppressions"] = float(suppressed.sum())
        then = np.append(runs[1:] == runs[:-1], False)
        pools["MedianSuppressionSec"] = (np.append(onset[1:], np.nan) - end)[suppressed & then]
        out["MedianSuppressionSec"] = median_s(pools["MedianSuppressionSec"])
    return out, pools, slack_of


def _close(ours, expected, tolerance: float) -> bool:
    if pd.isna(ours) or pd.isna(expected):
        return bool(pd.isna(ours) and pd.isna(expected))
    return abs(float(ours) - float(expected)) <= tolerance + 1e-9


#: The columns each spec defines somewhere (its family's), so that a spec
#: that stopped exercising its family shows.
S2_DEFINED = {
    "closed_max": {"MedianActivationSec", "CutOffs"},
    "fixed_interval": {"MainActivations", "MedianActivationSec", "MaxActivations",
                       "RewardEfficiency", "MedianReturnSec", "LockedOutContactSec"},
    "non_feeding": {"Suppressions", "MedianSuppressionSec"},
    "exponential": {"MainActivations", "MedianActivationSec"},
    "restart": {"MainActivations", "MedianActivationSec"},
    "fixed_ratio": {"MainActivations", "MedianActivationSec"},
    "yoked_controls": {"MedianActivationSec", "CutOffs"},
    "mutual_fixed_interval": {"MaxActivations", "RewardEfficiency", "MedianReturnSec",
                              "LockedOutContactSec"},
}


@pytest.mark.parametrize("name", list(S2_SPECS))
def test_the_family_columns_are_the_truths(s2_loaded, name):
    """Per well over the whole recording, then per chamber: its wells'
    counts summed and their values pooled for the medians.  Exact, but for
    the onsets the ledger times to the record (:func:`_truth_columns`)."""
    spec, results, e = s2_loaded(name)
    wells = ls.light_wells(e).set_index(["DFM", "Well"])
    chambers = ls.light_summary(e).set_index(["DFM", "Chamber"])
    misses, defined = [], set()
    loose = 0
    for dfm_spec in spec.dfms:
        d = dfm_spec.dfm_id
        book = al.ledger(e.dfms[d].paradigm, linked=_linked(dfm_spec.linkage))
        per_chamber: dict[int, list] = {}
        for well in range(1, 13):
            values, pools, slack = _truth_columns(spec, dfm_spec, results[d], e, book, well)
            per_chamber.setdefault((well + 1) // 2, []).append((values, pools, slack))
            loose += sum(v > 0 for v in slack.values())
            for column, expected in values.items():
                ours = wells.loc[(d, well), column]
                if not _close(ours, expected, slack[column]):
                    misses.append(f"DFM {d} W{well} {column}: pyflic {ours}, truth {expected}")
                if pd.notna(expected):
                    defined.add(column)
        for chamber, parts in per_chamber.items():
            row = chambers.loc[(d, chamber)]
            for column in ("MainActivations", "CutOffs", "MaxActivations",
                           "LockedOutContactSec", "Suppressions"):
                present = [v[column] for v, _p, _s in parts if pd.notna(v[column])]
                expected = float(sum(present)) if present else np.nan
                tolerance = sum(s[column] for _v, _p, s in parts)
                if not _close(row[column], expected, tolerance):
                    misses.append(f"DFM {d} chamber {chamber} {column}: {row[column]}, {expected}")
            for column in POOLED:
                pooled = np.concatenate([p[column] for _v, p, _s in parts])
                expected = float(np.median(pooled)) / 1000.0 if pooled.size else np.nan
                tolerance = max(s[column] for _v, _p, s in parts)
                if not _close(row[column], expected, tolerance):
                    misses.append(f"DFM {d} chamber {chamber} {column}: {row[column]}, {expected}")
    assert not misses, "\n".join(misses)
    assert defined == S2_DEFINED[name]
    ## Only the specs built to hold the light on time onsets to the record.
    assert (loose > 0) == (name in ("fixed_interval", "mutual_fixed_interval")), loose


def test_most_activations_are_judgeable(s2_loaded):
    """The judgeable Activations the medians and counts rest on are most of
    them where the light goes out between Activations, so the comparison
    with the truth is not of a few.  Under FI500 held contact keeps the
    light on through Activation after Activation, as designed."""
    for name, dfm_id in (("closed_max", 1), ("fixed_interval", 1), ("non_feeding", 1),
                         ("non_feeding", 2), ("yoked_controls", 1)):
        spec, _results, e = s2_loaded(name)
        dfm_spec = next(d for d in spec.dfms if d.dfm_id == dfm_id)
        book = al.ledger(e.dfms[dfm_id].paradigm, linked=_linked(dfm_spec.linkage))
        judged = ~book["Overlap"] & (book["Boundary"] == al.BOUNDARY_NONE)
        assert judged.mean() > 0.5, (name, dfm_id, judged.mean())


# -- decision 10: the two redefinitions --------------------------------------

def test_linked_light_no_longer_hides_lockout_feeding(s2_loaded):
    """Before, LockedOutContactSec was lick time in FixedInterval Main rows
    with OnMs 0, so a linked partner's light hid the well's feeding during
    its own lockout.  With both wells of each chamber triggering and
    linked, the old definition misses feeding the new one counts."""
    _spec, _results, e = s2_loaded("mutual_fixed_interval")
    state = e.dfms[1].paradigm
    licks = e.dfms[1].lick_df
    main = (state.program_interval.to_numpy(dtype=float, na_value=0) > 0)[:, None] \
        & ~state.phase_a
    new = ls.light_wells(e).set_index("Well")["LockedOutContactSec"]
    old = pd.Series({w: float((main[:, w - 1] & licks[f"W{w}"].to_numpy(bool)
                               & (state.on_ms[:, w - 1] == 0)).sum()) / 5.0
                     for w in range(1, 13)})
    assert (new >= old).all() and (new > old).sum() >= 6, pd.DataFrame({"old": old, "new": new})
    ## On a well its partner never lights the two agree (fixed_interval's FI10s).
    _spec, _results, solo = s2_loaded("fixed_interval")
    state = solo.dfms[1].paradigm
    licks = solo.dfms[1].lick_df
    wells = ls.light_wells(solo).set_index(["DFM", "Well"])
    for w in (1, 3, 5, 7, 9, 11):
        before = float((~state.phase_a[:, w - 1] & licks[f"W{w}"].to_numpy(bool)
                        & (state.on_ms[:, w - 1] == 0)).sum()) / 5.0
        assert wells.loc[(1, w), "LockedOutContactSec"] == before


def test_suppressions_no_longer_count_max_on_ends(s2_loaded):
    """Before, Suppressions counted the records whose light went out under
    NonFeeding, which counts each MaxOnMs end too.  Now they are the light
    periods contact ended: on NFMax5s (MaxOnMs 5000), fewer by the periods
    MaxOnMs ended."""
    spec, results, e = s2_loaded("non_feeding")
    wells = ls.light_wells(e).set_index(["DFM", "Well"])
    state = e.dfms[2].paradigm
    for w in (1, 3, 5, 7, 9, 11):
        lit = state.on_ms[:, w - 1] > 0
        before = int((~lit[1:] & lit[:-1]).sum())
        acts = _truth_activations(spec, results[2].truth, w)
        at_max = int((acts["length"] == 5000).sum())
        assert at_max > 0
        assert wells.loc[(2, w), "Suppressions"] <= before - at_max, w


# -- decision 11: feeding in light and dark, open loop only --------------------

def _index(lit, dark):
    return np.nan if pd.isna(lit) or pd.isna(dark) or lit + dark == 0 else (lit - dark) / (lit + dark)


def _program_rates(e, dfm: int, chamber: int) -> dict:
    """Licks per minute of the chamber's own light and dark, ``Both`` wells
    and each, over the rows a Program Interval ran: the computation
    light_phase_summary shares, without its baseline lead-in, and NA for a
    phase shorter than ``MIN_LIGHT_PHASE_MIN``."""
    from pyflic.base.analytics import chamber_light_phases

    d = e.dfms[dfm]
    ch = next(c for c in d.chambers if int(c.index) == chamber)
    a, b = int(ch.well_a), int(ch.well_b)
    rows = np.flatnonzero(d.paradigm.program_interval.to_numpy(dtype=np.int64, na_value=-1) > 0)
    phases = chamber_light_phases(d, (a, b), rows=rows)
    out = {}
    for phase in ("light", "dark"):
        minutes = phases[phase].samples / float(d.params.samples_per_second) / 60.0
        for name, wells in (("Both", (a, b)), ("A", (a,)), ("B", (b,))):
            licks = sum(phases[phase].licks[w] for w in wells)
            out[(name, phase)] = (licks / minutes if minutes >= ls.MIN_LIGHT_PHASE_MIN
                                  else np.nan)
    return out


def test_the_open_loop_rates_are_the_programs_light_and_dark(each, tmp_path):
    """LicksPerMinLit and LicksPerMinDark are the licks per minute of the
    chamber's own light and dark, light_phase_summary's computation, over
    the rows a Program Interval ran, for an Open chamber only.

    The baseline lead-in is left out: its few dark seconds once gave
    one_of_each's OpenLoop DFM, whose A wells are On throughout, a dark
    rate of 0 and an index of 1 on every chamber.  It has no dark phase, so
    both are NA; repeating_open_loop_spec (On for 3 minutes, Off for 12)
    has a real one."""
    spec = repeating_open_loop_spec(minutes=20.0)
    repeating = _load(_recording(tmp_path / "rep", spec, simulate(spec)))
    for e, open_dfms in ((each[0], {1}), (repeating, {1})):
        chambers = ls.light_summary(e).set_index(["DFM", "Chamber"])
        wells = ls.light_wells(e).set_index(["DFM", "Well"])
        for (dfm, chamber), row in chambers.iterrows():
            if dfm not in open_dfms:
                assert row[list(ls.OPEN_LOOP_COLUMNS)].isna().all(), (dfm, chamber)
                continue
            assert row.Role == "Open"
            rates = _program_rates(e, dfm, chamber)
            lit, dark = rates[("Both", "light")], rates[("Both", "dark")]
            assert row.LicksPerMinLit == pytest.approx(lit, nan_ok=True)
            assert row.LicksPerMinDark == pytest.approx(dark, nan_ok=True)
            assert row.LightFeedingIndex == pytest.approx(_index(lit, dark), nan_ok=True)
            ch = next(c for c in e.dfms[dfm].chambers if int(c.index) == chamber)
            for well, side in ((int(ch.well_a), "A"), (int(ch.well_b), "B")):
                assert wells.loc[(dfm, well), "LicksPerMinLit"] == pytest.approx(
                    rates[(side, "light")], nan_ok=True)
                assert wells.loc[(dfm, well), "LicksPerMinDark"] == pytest.approx(
                    rates[(side, "dark")], nan_ok=True)
    ## Lit for the whole program: no dark phase, so no index.
    always_on = ls.light_summary(each[0])
    always_on = always_on[always_on["DFM"] == 1]
    assert always_on["LicksPerMinLit"].notna().all()
    assert always_on["LicksPerMinDark"].isna().all() and always_on["LightFeedingIndex"].isna().all()
    ## The Repeating program's dark is its Off Intervals: a real comparison.
    table = ls.light_summary(repeating)
    assert (table["LicksPerMinDark"] > 0).all() and table["LightFeedingIndex"].notna().all()
    ## A Program Facet on an Off Interval alone is all dark: no index.
    off = ls.light_summary(repeating, range_minutes=(4.0, 14.0))
    assert off["LicksPerMinLit"].isna().all() and off["LightFeedingIndex"].isna().all()


# -- NA off the family, and on generations 1 and 2 ---------------------------

#: one_of_each_spec: each DFM's family columns (decision 8), per chamber.
EACH_FAMILY_COLUMNS = {
    ## The OpenLoop DFM is lit for the whole program: no dark phase, so no
    ## LicksPerMinDark and no LightFeedingIndex.
    1: {"LicksPerMinLit"},
    2: {"MedianActivationSec", "CutOffs"},
    3: {"MaxActivations", "RewardEfficiency", "MedianReturnSec", "LockedOutContactSec"},
    4: {"Suppressions", "MedianSuppressionSec"},
    5: {"MedianActivationSec", "CutOffs", "MainActivations"},   # Closed (MaxOnMs), PR10
    6: set(),                                                    # FR5, no Acclimation
}


def test_each_family_column_is_na_off_its_family(each):
    e, _results = each
    columns = (*FAMILY_NEW, *REDEFINED)
    for row in ls.light_summary(e).itertuples(index=False):
        defined = {c for c in columns if pd.notna(getattr(row, c))}
        assert defined == EACH_FAMILY_COLUMNS[row.DFM], (row.DFM, row.Chamber)
    for row in ls.light_wells(e).itertuples(index=False):
        defined = {c for c in columns if pd.notna(getattr(row, c))}
        ## The B wells are Off: only the chamber's open-loop rates reach them.
        expected = EACH_FAMILY_COLUMNS[row.DFM] if row.Well % 2 or row.DFM == 1 else set()
        assert defined == expected, (row.DFM, row.Well)


def test_a_yoked_chamber_has_no_family_columns(s2_loaded):
    _spec, _results, e = s2_loaded("yoked_controls")
    table = ls.light_summary(e).set_index("Chamber")
    for chamber in (4, 5, 6):
        assert table.loc[chamber, "Role"] == "Yoked"
        assert table.loc[chamber, list(FAMILY_NEW + REDEFINED)].isna().all()
    assert table.loc[[1, 2, 3], "CutOffs"].notna().all()


def test_the_family_columns_are_na_on_generations_1_and_2_and_without_a_program(tmp_path):
    spec, results = _choice()
    experiments = {
        "generation 1": _load(make_opto_dir(tmp_path / "g1")),
        "generation 2": _load(_recording(tmp_path / "g2", spec, results,
                                         writer=write_gen2_recording)),
        "no program": _load(_recording(tmp_path / "np", spec, results, program=False)),
    }
    for label, e in experiments.items():
        for table in (ls.light_wells(e), ls.light_summary(e)):
            assert not table.empty
            assert table[list(FAMILY_NEW + REDEFINED)].isna().all().all(), label
        ## Without generation 3 or a program, the light table's old four.
        assert ls.experiment_headline_metrics(e) == list(ls.HEADLINE_METRICS), label


# -- windows, Facets and the Project -----------------------------------------

@pytest.mark.parametrize("name", ["fixed_interval", "non_feeding", "closed_max"])
def test_the_family_counts_partition_over_facets(s2_loaded, monkeypatch, name):
    """An Activation belongs to the window holding its onset, and lick time
    to its row: the counts over Facets that partition the recording sum to
    the whole."""
    _spec, _results, e = s2_loaded(name)
    whole = ls.light_summary(e).set_index(["DFM", "Chamber"])

    def facet_windows_for(self, dfm_id):
        ## (a, b] windows: from below 0, so the first row (minute 0) is in.
        return [("Early", (-1.0, 5.0)), ("Late", (5.0, float("inf")))]

    monkeypatch.setattr(type(e), "facet_windows_for", facet_windows_for, raising=False)
    facet = ls.light_summary_facet(e)
    assert list(dict.fromkeys(facet["Facet"])) == ["Early", "Late"]
    for column in ("MainActivations", "CutOffs", "LockedOutContactSec", "Suppressions"):
        if whole[column].isna().all():
            assert facet[column].isna().all(), column
            continue
        summed = facet.groupby(["DFM", "Chamber"])[column].sum(min_count=1)
        assert (summed - whole[column]).abs().max() < 1e-9, column
    for column in ls.FIXED_INTERVAL_COLUMNS[:1]:            # MaxActivations: floored per window
        if whole[column].notna().any():
            summed = facet.groupby(["DFM", "Chamber"])[column].sum()
            assert ((whole[column] - summed) >= 0).all() and ((whole[column] - summed) <= 1).all()


def test_the_project_stacks_carry_every_family_column(gen3_project):
    """Decision 8's columns reach the Combined Analysis' stacked light
    tables as each member wrote them, and the Project's headline metrics
    are its families' (decision 9)."""
    project = gen3_project
    project.build_combined_analysis()
    analysis = Path(project.analysis_path)
    columns = [*FAMILY_NEW, *REDEFINED]
    for stacked, own, expected in (("proj_LightSummary.csv", "light_summary.csv",
                                    ls.LIGHT_CHAMBER_COLUMNS),
                                   ("proj_LightWells.csv", "light_wells.csv",
                                    ls.LIGHT_WELL_COLUMNS),
                                   ("proj_LightSummary_Facet.csv", "light_summary_facet.csv",
                                    ls.LIGHT_CHAMBER_COLUMNS),
                                   ("proj_LightWells_Facet.csv", "light_wells_facet.csv",
                                    ls.LIGHT_WELL_COLUMNS)):
        frame = pd.read_csv(analysis / stacked)
        assert set(expected) <= set(frame.columns), stacked
        for name in project.member_names:
            mine = frame[frame["Experiment"] == name].reset_index(drop=True)
            theirs = pd.read_csv(Path(project.member_dir(name)) / "analysis" / own)
            pd.testing.assert_frame_equal(mine[columns], theirs[columns], check_dtype=False)
        ## FR5 with Acclimation 3: closed loop in Acclimation, then ratio.
        assert frame["MainActivations"].notna().any() and \
            frame["MedianActivationSec"].notna().any(), stacked
    assert project.light_headline_metrics() == [
        "Activations", "LitSec", "MedianActivationSec", "LastRequirementMet", "FirmwareEvents",
        "AcclimationEndMin"]
    rows = project.light_comparison_rows()
    assert "MedianActivationSec" in {r["metric"] for r in rows}
    ## Members whose light tables have no generation-3 row keep the old four.
    old = pd.DataFrame({"Experiment": ["rep1", "rep2"], "DFM": [1, 1], "Generation": [1, 2]})
    assert project.light_headline_metrics(old) == list(ls.HEADLINE_METRICS)


# -- the headline metrics and the figures ------------------------------------

def test_the_headline_metrics_follow_the_families(each, tmp_path):
    """Decision 9: one_of_each runs every family and branch, ClosedLoop with
    MaxOnMs and PR10 with Acclimation; a generation-1 member keeps the old
    four."""
    from pyflic.base import report_content as rc
    from pyflic.base.pdf_report import build_experiment_report

    e, _results = each
    headline = ["LitSec", "LightFeedingIndex", "Activations", "MedianActivationSec", "CutOffs",
                "RewardEfficiency", "MedianReturnSec", "LastRequirementMet", "FirmwareEvents",
                "LitFraction", "Suppressions", "AcclimationEndMin"]
    assert ls.experiment_headline_metrics(e) == headline
    doc = build_experiment_report(e, metrics=())
    blocks = list(doc.blocks)
    start = next(i for i, b in enumerate(blocks)
                 if isinstance(b, rl.Heading) and b.text == "Light behaviour")
    titles = [p.title for b in blocks[start:] if isinstance(b, rl.PlotRow) for p in b.plots]
    ## Stream S3: one subsection per family, each with its own headline set,
    ## fixed length by its branch and the Acclimation apart; together they
    ## are the headline metrics.
    families = rc.light_families([(e.opto_program, e.opto.dfm_ids())])
    assert [f.key for f in families] == ["open loop", "closed loop", "interval", "ratio",
                                         "non-feeding", "acclimation"]
    ## A metric no chamber has a value for is named, not drawn: one_of_each's
    ## OpenLoop DFM is lit for the whole program, so it has no dark phase and
    ## no LightFeedingIndex.
    table = ls.light_summary(e)
    defined = {m for m in headline if table[m].notna().any()}
    assert "LightFeedingIndex" not in defined
    assert titles == [rc.metric_label(m) for f in families for m in f.metrics if m in defined]
    assert list(dict.fromkeys(m for f in families for m in f.metrics)) == headline
    assert ls.experiment_headline_metrics(_load(make_opto_dir(tmp_path / "g1"))) == \
        list(ls.HEADLINE_METRICS)
    assert ls.program_headline_metrics([]) == list(ls.HEADLINE_METRICS)


def test_every_family_column_is_a_light_figure():
    from pyflic.base import paradigm_families as pfam
    from pyflic.base import pubfigures

    labels = dict(pubfigures.LIGHT_METRICS)
    for column in (*FAMILY_NEW, *REDEFINED):
        assert labels.get(column), column
        info = pubfigures.PLOT_TYPES[f"faceted_{column.lower()}"]
        assert info["metric"] == column and info["source"] == pubfigures.SOURCE_LIGHT
    ## Every headline metric of every family can be drawn by the Project report.
    every = pfam.headline_metrics(pfam.FAMILY_ORDER, max_on=True, acclimation=True)
    assert all(f"faceted_{m.lower()}" in pubfigures.PLOT_TYPES for m in every)
    assert pubfigures.PLOT_TYPES["faceted_lightfeedingindex"]["y_limits"] == (-1.0, 1.0)
    ## And every family column is declared in the registry's summary columns.
    declared = {c for f in pfam.FAMILIES.values() for c in f.summary_columns}
    assert declared <= set(ls.LIGHT_CHAMBER_COLUMNS) and declared <= set(ls.LIGHT_WELL_COLUMNS)


def test_the_tolerance_is_the_design_constant(s2_loaded, monkeypatch):
    class Constants:
        def __init__(self, constants):
            self.global_constants = constants

    assert ls.activation_tolerance_ms(Constants({})) == 25
    assert ls.activation_tolerance_ms(Constants({"opto_activation_tolerance_ms": 60})) == 60
    assert ls.activation_tolerance_ms(Constants({"opto_activation_tolerance_ms": "x"})) == 25
    ## A tolerance wider than every Activation makes each judgeable one a cut-off.
    _spec, _results, e = s2_loaded("closed_max")
    before = ls.light_summary(e)["CutOffs"].sum()
    monkeypatch.setitem(e.global_constants, "opto_activation_tolerance_ms", 10 ** 6)
    after = ls.light_summary(e)["CutOffs"].sum()
    book = al.ledger(e.dfms[1].paradigm, linked=_linked(CHOICE_LINKAGE))
    assert before < after == (~book["Overlap"] & (book["Boundary"] == al.BOUNDARY_NONE)).sum()


# -- PyMCU's simulator ---------------------------------------------------------

@pytest.mark.slow
def test_main_activations_are_pymcus(analysed):  # noqa: F811 - the imported fixture
    """On PyMCU's simulator recording (``test_pymcu_simulator``'s fixtures),
    MainActivations is the truth's on every Trigger Well."""
    truth = analysed.truth
    merged = truth.merge(analysed.light_wells, on=["DFM", "Well"], how="left",
                         suffixes=("_truth", ""), validate="one_to_one")
    assert len(merged) == 12
    assert merged["MainActivations"].tolist() == merged["MainActivations_truth"].astype(float).tolist()
    ## Some flies never finish Acclimation on seed 1; most earn Main ones.
    assert (merged["MainActivations"] > 0).sum() >= 8


def test_a_light_metric_the_same_in_every_chamber_is_named_not_compared():
    """An open-loop chamber lit for the whole program has the same lit time
    as every other: the comparison names it once instead of saying two
    treatments are needed, which they were."""
    from pyflic.base import report_content as rc

    rows = pd.DataFrame({"Treatment": ["Ctrl"] * 3 + ["Exp"] * 3,
                         "LitSec": [475.0] * 6, "Activations": [3, 4, 5, 9, 10, 12]})
    asked: list = []

    def compare(keep):
        asked.append(list(keep))
        return []

    blocks = rc.light_comparison_blocks(["LitSec", "Activations"], rows, compare)
    texts = [b.text for b in blocks if isinstance(b, rl.Paragraph)]
    assert asked == [["Activations"]]
    assert texts[0] == "Lit time (s) is the same in every chamber, so not compared between treatments."
    ## Nothing varies: no table at all, only the note.
    asked.clear()
    blocks = rc.light_comparison_blocks(["LitSec"], rows, compare)
    assert asked == [] and len(blocks) == 1
