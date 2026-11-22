"""Loading the three Generations (docs/v4-data-plan.md, Phase 0b).

Generation 1 comes from the existing fixtures (``pr_fixtures``,
``opto_fixtures``: 27 columns, the training flag in the signal), generations
2 and 3 from the firmware 4.0.0 simulator (``v4_fixtures``).  Each goes
through the normal config path.  Also here: the data-break fix, the
Environmental Monitor (DFM 99) refused everywhere, the disk cache's key, and
the round trip of every ready-made spec's ``Program.txt`` through the parser.
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import replace
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from pyflic.base import cache
from pyflic.base import layout
from pyflic.base import opto_program as op
from pyflic.base import paradigm_state as ps
from pyflic.base.dfm import DFM
from pyflic.base.parameters import Parameters
from pyflic.base.project import dfm_ids_in_data, has_experiment_data
from pyflic.base.yaml_config import load_experiment_yaml
from pyflic.base.yaml_lint import lint_flic_config

from opto_fixtures import make_opto_dir, write_opto_csv
from pr_fixtures import make_pr_experiment_dir
from v4_fixtures import (
    CHOICE_PR_CASES,
    PR10,
    choice_pr_spec,
    fixed_ratio_spec,
    one_of_each_spec,
    program_text,
    repeating_open_loop_spec,
    simulate,
    stress_spec,
    write_gen2_recording,
    write_v4_recording,
    yoked_controls_spec,
)

RECORD_MIN = 0.2 / 60.0
ENVMON = "DFM 99 is the Environmental Monitor, not a DFM"


def two_well_config(dfm_ids, *, chamber_layout: str = "two_well") -> dict:
    """A minimal Custom config for the simulator's recordings: A wells on
    the left, every chamber assigned."""
    n = 6 if chamber_layout == "two_well" else 12
    params = {"feeding_threshold": 20, "feeding_minimum": 10, "tasting_minimum": 5,
              "tasting_maximum": 20, "samples_per_second": 5}
    if chamber_layout == "two_well":
        params["pi_direction"] = "left"
    return {"global": {"chamber_layout": chamber_layout, "transform_licks": False,
                       "params": params},
            "dfms": [{"id": int(i), "chambers": {c: ("Ctrl" if c % 2 else "Exp")
                                                 for c in range(1, n + 1)}}
                     for i in dfm_ids]}


def recording_dir(root: Path, spec, *, writer=write_v4_recording, results=None,
                  chamber_layout: str = "two_well", **kw) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    writer(root / "data", spec, results=results, **kw)
    cfg = two_well_config([d.dfm_id for d in spec.dfms], chamber_layout=chamber_layout)
    (root / "flic_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return root


def load(root: Path):
    return load_experiment_yaml(root, parallel=False, use_disk_cache=False)


@lru_cache(maxsize=None)
def choice():
    spec = choice_pr_spec()
    return spec, simulate(spec)


@pytest.fixture(scope="module")
def gen3(tmp_path_factory):
    spec, results = choice()
    return load(recording_dir(tmp_path_factory.mktemp("gen3"), spec, results=results)), results


@pytest.fixture(scope="module")
def gen2(tmp_path_factory):
    spec, results = choice()
    return load(recording_dir(tmp_path_factory.mktemp("gen2"), spec, results=results,
                              writer=write_gen2_recording)), results


# ---------------------------------------------------------------------------
# Every Generation loads and keeps its Generation
# ---------------------------------------------------------------------------

def _generation_dir(kind: str, root: Path) -> Path:
    if kind == "gen1_pr":
        return make_pr_experiment_dir(root)
    if kind == "gen1_opto":
        return make_opto_dir(root)
    spec, results = choice()
    writer = write_gen2_recording if kind == "gen2" else write_v4_recording
    return recording_dir(root, spec, results=results, writer=writer)


@pytest.mark.parametrize("kind,generation,training_info", [
    ("gen1_pr", 1, "flag"), ("gen1_opto", 1, "flag"), ("gen2", 2, "none"), ("gen3", 3, "phase")])
def test_each_generation_loads_and_keeps_its_generation(tmp_path, kind, generation, training_info):
    e = load(_generation_dir(kind, tmp_path))
    assert e.dfms
    for dfm in e.dfms.values():
        assert dfm.generation == generation and dfm.training_info == training_info
        ## The 63 Paradigm columns are never in raw_df (or anything copied from it).
        assert not set(ps.PARADIGM_COLUMNS) & set(dfm.raw_df.columns)
        assert not set(ps.PARADIGM_COLUMNS) & set(dfm.baseline_df.columns)
        if generation == 3:
            assert isinstance(dfm.paradigm, ps.ParadigmState)
            assert len(dfm.paradigm) == len(dfm.raw_df)
            assert np.allclose(dfm.paradigm.minutes, dfm.raw_df["Minutes"])
            assert np.array_equal(dfm.paradigm.index, dfm.raw_df["Index"])
        else:
            assert dfm.paradigm is None
        copy = dfm.with_params(dfm.params)
        assert (copy.generation, copy.training_info) == (generation, training_info)
        assert copy.paradigm is dfm.paradigm
        assert [copy.training_flag_state(w) for w in range(1, 13)] == \
            [dfm.training_flag_state(w) for w in range(1, 13)]
        assert [copy.training_end_minutes(w) for w in range(1, 13)] == \
            [dfm.training_end_minutes(w) for w in range(1, 13)]
        report = dfm.integrity_report()
        assert report["generation"] == generation
    summary = e.summary_text(include_qc=False)
    assert summary.count(f"- generation: {generation} (") == len(e.dfms)
    assert ("OnMs, exact" in summary) == (generation == 3)
    assert ("OptoCol1, sampled at 5 Hz" in summary) == (generation != 3)


def test_the_range_filter_applies_to_the_paradigm_state(tmp_path):
    spec, results = choice()
    root = recording_dir(tmp_path, spec, results=results)
    e = load_experiment_yaml(root, parallel=False, use_disk_cache=False, range_minutes=(10, 20))
    dfm = e.dfms[1]
    assert len(dfm.paradigm) == len(dfm.raw_df) > 0
    assert np.array_equal(dfm.paradigm.index, dfm.raw_df["Index"])
    ## (10, 20]: the end minute is kept, as in every Facet window (ADR-0020).
    assert dfm.raw_df["Minutes"].min() > 10 and dfm.raw_df["Minutes"].max() <= 20
    full = results[1].rows.set_index("Index")
    assert np.array_equal(dfm.paradigm.on_ms[:, 0],
                          full.loc[dfm.raw_df["Index"], "OnMs1"].to_numpy(dtype=np.int64))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def test_training_ends_where_the_paired_sucrose_well_leaves_acclimation(gen3):
    """Generation 3: the last Training row is the one before the first row
    in Phase M under PR10, as generation 1 records the last flagged minute."""
    e, results = gen3
    seen = set()
    for dfm_id, dfm in e.dfms.items():
        truth = results[dfm_id].truth
        minutes = dfm.raw_df["Minutes"].to_numpy()
        for chamber, case in CHOICE_PR_CASES.items():
            sucrose = 2 * chamber - 1
            end_ms = truth.acclimation_end_ms[sucrose]
            seen.add((case, end_ms is None))
            if end_ms is None:
                assert dfm.training_flag_state(sucrose) == "never_cleared", (dfm_id, case)
                assert dfm.training_end_minutes(sucrose) is None
                continue
            first_main = end_ms // 200          # the record that ran the step
            assert dfm.training_flag_state(sucrose) == "cleared"
            assert dfm.training_end_minutes(sucrose) == minutes[first_main - 1]
            assert dfm.training_end_minutes(sucrose) == pytest.approx(end_ms / 60000.0, abs=RECORD_MIN)
            ## The (0, end] / (end, inf) split puts the first Main row in Test.
            assert minutes[first_main] > dfm.training_end_minutes(sucrose)
            ## The Off wells never run a ratio Paradigm: no Training.
            assert dfm.training_flag_state(sucrose + 1) == "never_flagged"
    assert ("never_training", True) in seen and ("healthy", False) in seen


def test_a_ratio_paradigm_without_acclimation_ends_training_at_its_start(tmp_path):
    spec = choice_pr_spec(dfm_ids=(1,))
    spec = replace(spec, paradigms={**spec.paradigms, "PR10": replace(PR10, acclimation=0)})
    results = simulate(spec)
    e = load(recording_dir(tmp_path, spec, results=results))
    dfm = e.dfms[1]
    pr10_ms = next(ms for ms, _k, _c, name in results[1].truth.schedule if name == "PR10")
    for sucrose in range(1, 13, 2):
        assert dfm.training_flag_state(sucrose) == "cleared"     # the Closed rows were Training
        assert dfm.training_end_minutes(sucrose) == pytest.approx(pr10_ms / 60000.0, abs=RECORD_MIN)


def test_generation_2_has_no_training_information(gen2):
    e, _results = gen2
    for dfm in e.dfms.values():
        assert dfm.training_info == "none"
        assert all(dfm.training_flag_state(w) == "never_flagged" for w in range(1, 13))
        assert all(dfm.training_end_minutes(w) is None for w in range(1, 13))


def test_generation_1_training_is_unchanged(tmp_path):
    e = load(make_pr_experiment_dir(tmp_path))
    dfm = e.dfms[1]
    assert dfm.training_info == "flag"
    assert dfm.training_flag_state(1) == "cleared"
    assert dfm.training_end_minutes(1) == pytest.approx(6.0, abs=0.01)
    ## The flag is stripped from the signal as before.
    assert dfm.raw_df[[f"W{w}" for w in range(1, 13)]].max().max() < 40000


def test_a_range_after_training_end_keeps_generation_1_s_training(tmp_path):
    """Review C5: the training flags are read from the whole recording before
    a range clips it, so a range starting after a group's training end does
    not make the group look untrained (and auto-remove it)."""
    root = make_pr_experiment_dir(tmp_path)
    whole = load(root)
    part = load_experiment_yaml(root, parallel=False, use_disk_cache=False,
                                range_minutes=(8, 0))
    for well in range(1, 13):
        assert part.dfms[1].training_flag_state(well) == whole.dfms[1].training_flag_state(well)
        assert part.dfms[1].training_end_minutes(well) == whole.dfms[1].training_end_minutes(well)
    gt = part.group_training(1, 1)
    assert gt.complete and gt.training_end == pytest.approx(6.0, abs=0.01)
    assert not any("never carried the training flag" in n for n in gt.notes)
    assert part.dfms[1].raw_df[[f"W{w}" for w in range(1, 13)]].max().max() < 40000
    ## What the clip cut from the front is kept for the rows it left: the
    ## recording's first minutes, the flag taken off, and the light before.
    before = part.dfms[1].before_range
    assert before.start_min == 8.0 and before.start_minutes[0] == whole.dfms[1].raw_df["Minutes"].iloc[0]
    assert np.nanmax(before.start_raw) < 40000 and before.lit.shape == (12,)
    assert whole.dfms[1].before_range is None


# ---------------------------------------------------------------------------
# Light seconds
# ---------------------------------------------------------------------------

def test_generation_3_light_seconds_are_the_on_ms(gen3):
    e, results = gen3
    summary = e.feeding_summary()
    for _, row in summary.iterrows():
        truth = results[int(row["DFM"])].truth
        a, b = 2 * int(row["Chamber"]) - 1, 2 * int(row["Chamber"])
        assert row["OptoOn_sec_A"] == pytest.approx(truth.on_ms_total[a] / 1000.0, abs=0.2)
        assert row["OptoOn_sec_B"] == pytest.approx(truth.on_ms_total[b] / 1000.0, abs=0.2)
    ## Over a window, the sum of OnMs in its rows.
    dfm = e.dfms[1]
    window = dfm.feeding_summary(range_minutes=(5.0, 12.0))
    mins = dfm.raw_df["Minutes"].to_numpy()
    rows = (mins > 5.0) & (mins <= 12.0)
    on = results[1].rows.set_index("Index").loc[dfm.raw_df["Index"][rows], "OnMs1"].sum()
    assert window.loc[window["Chamber"] == 1, "OptoOn_sec_A"].iloc[0] == pytest.approx(on / 1000.0)


def test_single_well_light_seconds_on_generation_3(tmp_path):
    spec = yoked_controls_spec(minutes=10.0)
    results = simulate(spec)
    e = load(recording_dir(tmp_path, spec, results=results, chamber_layout="single_well"))
    summary = e.feeding_summary()
    for _, row in summary.iterrows():
        assert row["OptoOn_sec"] == pytest.approx(
            results[1].truth.on_ms_total[int(row["Chamber"])] / 1000.0, abs=0.2)


def test_generation_2_light_seconds_are_sampled(gen2):
    e, results = gen2
    summary = e.feeding_summary()
    for _, row in summary.iterrows():
        rows = results[int(row["DFM"])].rows
        a = 2 * int(row["Chamber"]) - 1
        lit = ((rows["OptoCol1"].to_numpy() >> (a - 1)) & 1).sum()
        assert row["OptoOn_sec_A"] == pytest.approx(lit / 5.0)


# ---------------------------------------------------------------------------
# Data breaks (decision 22)
# ---------------------------------------------------------------------------

def test_a_data_break_in_a_27_column_file(tmp_path):
    data = tmp_path / "data"
    path = write_opto_csv(data, 1)
    frame = pd.read_csv(path)
    frame.drop(index=range(1000, 1020)).to_csv(path, index=False)
    dfm = DFM.load(1, Parameters.single_well(), data_dir=data)
    breaks = dfm.data_breaks()
    assert list(breaks.columns[:2]) == ["BreakIndex", "GapSeconds"]
    assert "Index" in breaks.columns
    assert breaks["BreakIndex"].tolist() == [1001]
    assert breaks["Index"].tolist() == [1021]
    assert breaks["GapSeconds"].iloc[0] == pytest.approx(21 * 0.2, abs=1e-6)


def test_a_data_break_in_a_90_column_file(tmp_path):
    spec, results = choice()
    root = recording_dir(tmp_path, spec, results=results, index_gaps={1: [(1000, 20)]})
    dfm = DFM.load(1, Parameters.two_well(), data_dir=root / "data")
    assert dfm.generation == 3
    breaks = dfm.data_breaks()
    assert list(breaks.columns[:2]) == ["BreakIndex", "GapSeconds"]
    assert breaks["BreakIndex"].tolist() == [1001] and breaks["Index"].tolist() == [1021]
    assert breaks["GapSeconds"].iloc[0] == pytest.approx(21 * 0.2, abs=1e-6)
    assert dfm.paradigm.index_gaps() == [(1000, 20, pytest.approx(1000 * RECORD_MIN))]
    ## The QC summary that used to raise on it.
    e = load(root)
    qc = e.compute_qc_results()
    assert qc[1]["data_breaks_count"] == 1 and qc[2]["data_breaks_count"] == 0
    assert qc[1]["integrity"]["generation"] == 3


# ---------------------------------------------------------------------------
# The Environmental Monitor (decision 5)
# ---------------------------------------------------------------------------

def write_monitor_csv(directory: Path, rows: int = 50) -> Path:
    """The Environmental Monitor's ``DFM99_0.csv``: the old 27 columns,
    W1-W6 relay and output states."""
    from v4_fixtures import DATA_COLUMNS

    frame = pd.DataFrame(0, index=range(rows), columns=list(DATA_COLUMNS))
    frame["Date"], frame["Time"] = "10/16/2026", "23:34:00"
    frame["MSec"] = (np.arange(rows) % 5) * 200.0
    frame["Sample"] = frame["Index"] = np.arange(1, rows + 1)
    frame["Temp"], frame["Humid"] = 24.5, 51.0
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "DFM99_0.csv"
    frame.to_csv(path, index=False)
    return path


def _with_monitor(root: Path) -> Path:
    """A generation-3 recording with the Environmental Monitor's file."""
    spec, results = choice()
    recording_dir(root, spec, results=results)
    write_monitor_csv(root / "data")
    return root


def test_dfm_99_is_never_discovered(tmp_path):
    root = _with_monitor(tmp_path / "exp")
    data = root / "data"
    assert layout.dfm_ids(data) == [1, 2]
    assert "DFM99_0.csv" not in layout.dfm_files(data)
    assert dfm_ids_in_data(root) == [1, 2]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        e = load(root)
    assert sorted(e.dfms) == [1, 2]
    assert not [w for w in caught if "99" in str(w.message)]

    ## A data/ holding only the monitor's file is no recording.
    lone = tmp_path / "lone"
    write_monitor_csv(lone / "data")
    assert not has_experiment_data(lone) and dfm_ids_in_data(lone) == []
    (lone / "flic_config.yaml").write_text("dfms: []\n")
    assert layout.classify(lone).status == layout.NO_RECORDING


def test_filing_moves_the_monitor_with_the_recording(tmp_path):
    root = tmp_path / "unfiled"
    write_opto_csv(root, 1)
    write_monitor_csv(root)
    item = layout.classify(root)
    assert item.status == layout.UNFILED
    plan = layout.plan_filing(root)
    moved = {os.path.basename(src): os.path.basename(os.path.dirname(dst)) for src, dst in plan.moves}
    assert moved == {"DFM1_0.csv": "data", "DFM99_0.csv": "data"}


def test_a_config_naming_dfm_99_is_refused(tmp_path):
    root = _with_monitor(tmp_path)
    cfg = yaml.safe_load((root / "flic_config.yaml").read_text())
    cfg["dfms"].append({"id": 99, "chambers": {c: "Ctrl" for c in range(1, 7)}})
    (root / "flic_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    with pytest.raises(ValueError, match=ENVMON):
        load(root)
    issues = lint_flic_config(root / "flic_config.yaml")
    assert any(i.severity == "error" and ENVMON in i.message for i in issues)
    with pytest.raises(ValueError, match=ENVMON):
        DFM.load(99, Parameters.two_well(), data_dir=root / "data")


# ---------------------------------------------------------------------------
# The disk cache (decision 29)
# ---------------------------------------------------------------------------

def test_the_cache_key_follows_program_and_messages(tmp_path):
    spec, results = choice()
    root = recording_dir(tmp_path, spec, results=results, program=False, messages=False)
    key = lambda: cache.feeding_summary_key(root, range_minutes=(0, 0), transform_licks=False)  # noqa: E731
    assert key().startswith(f"v{cache._CACHE_VERSION}_")
    bare = key()
    program = root / "data" / "Program.txt"
    program.write_text(program_text(spec), encoding="utf-8")
    with_program = key()
    assert with_program != bare
    messages = root / "data" / "Messages.csv"
    messages.write_text("DFM,Date,Time,Millisecond,Sample,Message,MsgType\n", encoding="utf-8")
    with_messages = key()
    assert with_messages != with_program
    ## A changed Program.txt changes the key; a file that is not part of the
    ## recording does not.
    stamp = time.time_ns() + 5_000_000_000
    os.utime(program, ns=(stamp, stamp))
    assert key() != with_messages
    unrelated = root / "data" / "notes.txt"
    before = key()
    unrelated.write_text("not part of the recording")
    assert key() == before


# ---------------------------------------------------------------------------
# Program.txt round trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("make", [choice_pr_spec, one_of_each_spec, fixed_ratio_spec,
                                  repeating_open_loop_spec, yoked_controls_spec, stress_spec],
                         ids=lambda f: f.__name__)
def test_every_ready_made_program_parses_as_v4(make):
    spec = make()
    program = op.parse_program(program_text(spec))
    assert program.grammar == op.GRAMMAR_V4
    assert program.rejected == {}
    assert sorted(program.dfms) == sorted(d.dfm_id for d in spec.dfms)
    duration = timedelta(minutes=spec.duration_min)
    for d in spec.dfms:
        section = program.section(d.dfm_id)
        assert section.linkage == tuple(d.linkage)
        assert section.program_type == d.program_type
        ## PyMCU 4.0.1: a Circadian block lists the whole day, stamped from
        ## midnight; the others the Intervals that start before the end.
        circadian = d.program_type == "Circadian"
        origin = (datetime.combine(spec.start.date(), datetime.min.time())
                  if circadian else spec.start)
        printed, starts, start = [], [], timedelta(0)
        for iv in d.intervals:
            if circadian or start < duration:
                printed.append(iv)
                starts.append((origin + start).replace(microsecond=0))   # printed to the second
            start += timedelta(minutes=iv.minutes)
        assert len(section.intervals) == len(printed)
        assert [got.start for got in section.intervals] == starts
        for got, want in zip(section.intervals, printed):
            assert got.paradigm.name == want.paradigm
            assert got.paradigm.type_name == spec.paradigms[want.paradigm].type
            assert got.thresholds == tuple(want.thresholds)
            assert got.dark == want.dark
