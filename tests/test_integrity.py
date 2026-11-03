"""Integrity by Generation, the crosstalk table and the Program timeline
(docs/v4-data-plan.md, stream 6).

* The integrity report labels the Error byte's bits by Generation, reports
  gaps in ``Index`` as lost records and repeated records as dropped at load,
  and on generation 3 counts unknown-instruction rows and late records and
  lists the Paradigm sequence.  A generation-1 file's text gains only the
  two zero counts (the Generation line came with Phase 0).
* The simultaneous-feeding table carries the counts before the dual-feeding
  correction beside the ones after it (decision 23).
* The Program timeline renders for every ready-made spec and for a
  generation-1 recording with a program, and QC reports and basic analysis
  write it.
"""

from __future__ import annotations

import contextlib
import io
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from pyflic.base import pdf_report
from pyflic.base import program_timeline as pt
from pyflic.base.dfm import DFM, NO_ACK_NOTE
from pyflic.base.parameters import Parameters
from pyflic.base.script_editor.runner import run_experiment_script
from pyflic.base.yaml_config import load_experiment_yaml

from opto_fixtures import make_opto_dir, write_opto_csv
from pr_fixtures import make_pr_experiment_dir
from v4_fixtures import (
    CHOICE_PR_CASES,
    choice_pr_spec,
    family_specs,
    fixed_ratio_spec,
    one_of_each_spec,
    repeating_open_loop_spec,
    restart_messages,
    simulate,
    stress_spec,
    write_gen2_recording,
    write_v4_recording,
    yoked_controls_spec,
)

RECORD_MIN = 0.2 / 60.0


def _config(dfm_ids) -> dict:
    params = {"feeding_threshold": 20, "feeding_minimum": 10, "tasting_minimum": 5,
              "tasting_maximum": 20, "samples_per_second": 5, "pi_direction": "left"}
    return {"global": {"chamber_layout": "two_well", "transform_licks": False,
                       "params": params},
            "dfms": [{"id": int(i), "chambers": {c: ("Ctrl" if c % 2 else "Exp")
                                                 for c in range(1, 7)}} for i in dfm_ids]}


def _recording(root: Path, spec, *, writer=write_v4_recording, results=None, **kw) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    writer(root / "data", spec, results=results, **kw)
    (root / "flic_config.yaml").write_text(
        yaml.safe_dump(_config([d.dfm_id for d in spec.dfms]), sort_keys=False))
    return root


def _load(root: Path):
    with contextlib.redirect_stdout(io.StringIO()):
        return load_experiment_yaml(root, parallel=False, use_disk_cache=False)


def _report(dfm: DFM) -> tuple[dict, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report = dfm.integrity_report()
    return report, buf.getvalue()


@lru_cache(maxsize=None)
def _choice():
    spec = choice_pr_spec(minutes=12.0, dfm_ids=(1,))
    return spec, simulate(spec)


# ---------------------------------------------------------------------------
# Generation 3: lost records, error labels, the Paradigm state, messages
# ---------------------------------------------------------------------------

def test_a_generation_3_recording_reports_lost_records_and_its_error_labels(tmp_path):
    spec, results = _choice()
    _recording(tmp_path, spec, results=results, index_gaps={1: [(1000, 20), (3000, 7)]},
               error_bits={1: [(500, 0x01), (600, 0x03), (700, 0x82), (800, 0x01)]})
    dfm = DFM.load(1, Parameters.two_well(), data_dir=tmp_path / "data")
    report, text = _report(dfm)
    assert report["generation"] == 3
    assert report["lost_records"] == 27
    assert report["lost_record_gaps"] == [(1000, 20, pytest.approx(1000 * RECORD_MIN)),
                                          (3000, 7, pytest.approx(3000 * RECORD_MIN))]
    assert report["index_increments_by_one"] is False
    assert report["repeated_records"] == 0
    assert "Lost records: 27 in 2 gap(s)" in text
    assert "  after Index 1000 (minute 3.33): 20 record(s), 4.0 s" in text
    assert "  after Index 3000 (minute 10.00): 7 record(s), 1.4 s" in text
    ## Firmware 3.5.0 and later: 0x01 is No Ack, 0x02 Overflow.
    assert "    No Ack: 3\n" in text and "    Overflow: 2\n" in text
    assert "    OInterrupt Error: 1\n" in text and "No ACK / overflow" not in text
    ## The Paradigm state.
    assert report["unknown_instruction_rows"] == 0 and report["late_records"] == 0
    sequence = report["paradigm_sequence"]
    ## The lead-in names no Paradigm: PyMCU writes NA before the program.
    assert [s["Paradigm"] for s in sequence] == ["NA", "Closed", "PR10"]
    assert sum(s["Rows"] for s in sequence) == len(dfm.raw_df)
    assert sequence[0]["Rows"] == dfm.paradigm.lead_in_rows()
    assert "Lead-in rows (Interval 0, before the first instruction): 25" in text
    ## Messages.csv: the gap errors PyMCU logged, for this DFM.
    assert "## Messages.csv errors (DFM 1 and the MCU)" in text
    assert "DFM 1, Index 1021: Lost 20 records (4.0 s): Index 1001 to 1020, from " in text
    assert "DFM 1, Index 3008: Lost 7 records (1.4 s): Index 3001 to 3007, from 23:44:00." in text


def test_unknown_instruction_rows_and_late_records(tmp_path):
    spec, results = _choice()
    _recording(tmp_path, spec, results=results)
    path = tmp_path / "data" / "DFM1_0.csv"
    frame = pd.read_csv(path, keep_default_na=False, dtype=str)
    frame.loc[2000:2009, "Interval"] = "NA"
    frame.loc[3000:3002, "OnMs5"] = "230"
    frame.to_csv(path, index=False)
    dfm = DFM.load(1, Parameters.two_well(), data_dir=tmp_path / "data")
    report, text = _report(dfm)
    assert report["unknown_instruction_rows"] == 10
    assert report["late_records"] == 3
    assert "Rows under an unknown instruction (after the lead-in): 10" in text
    assert "Late records (some OnMs above 200 ms): 3" in text


def test_generation_2_has_no_paradigm_state_and_the_new_labels(tmp_path):
    spec, results = _choice()
    _recording(tmp_path, spec, results=results, writer=write_gen2_recording,
               error_bits={1: [(500, 0x02)]})
    dfm = DFM.load(1, Parameters.two_well(), data_dir=tmp_path / "data")
    report, text = _report(dfm)
    assert report["generation"] == 2
    assert report["unknown_instruction_rows"] is None and report["late_records"] is None
    assert report["paradigm_sequence"] is None and report["lost_records"] == 0
    assert "None: every Paradigm column is NA" in text
    assert "    Overflow: 1\n" in text


# ---------------------------------------------------------------------------
# Generation 3: restarts, the No Ack note and the lost-records cross-check
# (docs/paradigm-analyses-plan.md, C6)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _restart():
    spec = family_specs(minutes=6.0)["restart"]
    return spec, simulate(spec)


def _restart_lines(spec, dfm_id: int) -> tuple[dict, str]:
    """The report entry and the text line the restart spec's DFM should get,
    named in Messages.csv."""
    (r,) = next(d for d in spec.dfms if d.dfm_id == dfm_id).restarts
    after, missing, idle = round(r.at_s * 5), round(r.down_s * 5), round(r.resend_s * 5)
    when = spec.start + timedelta(milliseconds=(after + missing + 1) * 200)
    minute = (after + missing + 1) * RECORD_MIN
    entry = {"Time": pd.Timestamp(when), "Minute": pytest.approx(minute), "Row": after,
             "ResumeRow": after + idle, "AfterIndex": after, "Missing": missing,
             "Seconds": pytest.approx(missing * 0.2), "Source": "data", "Messages": True,
             "Note": "named in Messages.csv"}
    line = (f"  {when:%m/%d/%Y %H:%M:%S}.{when.microsecond // 1000:03d} (minute {minute:.2f}): "
            f"{missing} record(s), {missing * 0.2:.1f} s, missing after Index {after}, then {idle} "
            f"row(s) under no instruction; named in Messages.csv\n")
    return entry, line


def _drop_messages(path: Path, *texts: str) -> None:
    """Messages.csv without its lines holding any of *texts*."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(line for line in lines if not any(t in line for t in texts)),
                    encoding="utf-8")


def test_the_report_lists_each_restart_and_whether_messages_names_it(tmp_path):
    spec, results = _restart()
    _recording(tmp_path, spec, results=results)
    for dfm_id in (1, 2):
        report, text = _report(DFM.load(dfm_id, Parameters.two_well(), data_dir=tmp_path / "data"))
        entry, line = _restart_lines(spec, dfm_id)
        assert report["restarts"] == [entry]
        assert f"## DFM restarts\nRestarts: 1\n{line}\n" in text
        ## The restart's idle rows are the unknown-instruction rows, its gap the only lost records.
        assert report["unknown_instruction_rows"] == entry["ResumeRow"] - entry["Row"]
        assert report["lost_record_gaps"] == [(entry["AfterIndex"], entry["Missing"],
                                               pytest.approx(entry["AfterIndex"] * RECORD_MIN))]
        assert "Lost records Messages.csv names: 0, as the Index gaps outside restarts show.\n" in text
    ## A clean recording has none.
    clean, clean_results = _choice()
    _recording(tmp_path / "clean", clean, results=clean_results)
    report, text = _report(DFM.load(1, Parameters.two_well(), data_dir=tmp_path / "clean" / "data"))
    assert report["restarts"] == [] and "## DFM restarts\nNone.\n" in text


def test_a_restart_messages_does_not_name_says_so(tmp_path):
    spec, results = _restart()
    _recording(tmp_path / "none", spec, results=results, messages=False)
    _recording(tmp_path / "silent", spec, results=results)
    _drop_messages(tmp_path / "silent" / "data" / "Messages.csv", "restarted", "recorded nothing")
    for folder, named, note in (("none", None, "no Messages.csv to confirm it"),
                                ("silent", False, "not named in Messages.csv")):
        report, text = _report(DFM.load(1, Parameters.two_well(), data_dir=tmp_path / folder / "data"))
        (entry,) = report["restarts"]
        assert entry["Messages"] is named and entry["Note"] == note and entry["Source"] == "data"
        assert f"row(s) under no instruction; {note}\n" in text


def test_a_restart_only_messages_names_is_listed_too(tmp_path):
    """The data show an Index gap but no rows under no instruction (PyMCU's
    resend came at once), or nothing at all."""
    spec, results = _choice()
    _recording(tmp_path, spec, results=results, index_gaps={1: [(1000, 25)]})
    path = tmp_path / "data" / "Messages.csv"
    _drop_messages(path, "Lost 25 records")
    started, finished = restart_messages(1, 1000, 1000, 25)
    gone = restart_messages(1, 2000, 2000, 10)          # a range the data show no gap at
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[1:1] = [f"1,10/16/2026,23:37:25,200.0,1,{started},Error\n",
                  f"1,10/16/2026,23:37:25,200.0,1000,{finished},Error\n",
                  f"1,10/16/2026,23:40:00,200.0,1,{gone[0]},Error\n",
                  f"1,10/16/2026,23:40:00,200.0,2000,{gone[1]},Error\n",
                  f"1,10/16/2026,23:42:00,0.0,1,{restart_messages(1, 600, 2400, 1)[0]},Error\n"]
    path.write_text("".join(lines), encoding="utf-8")
    report, text = _report(DFM.load(1, Parameters.two_well(), data_dir=tmp_path / "data"))
    gap, elsewhere, no_gap = report["restarts"]
    assert (gap["Source"], gap["AfterIndex"], gap["Missing"], gap["Row"]) == ("Messages.csv", 1000, 25, None)
    assert gap["Minute"] == pytest.approx(1026 * RECORD_MIN)
    assert gap["Note"] == "named in Messages.csv; the data show an Index gap only"
    assert (elsewhere["AfterIndex"], elsewhere["Missing"], elsewhere["Minute"]) == (2000, 10, None)
    assert elsewhere["Note"] == "named in Messages.csv; the rows loaded show no Index gap there"
    assert no_gap["AfterIndex"] is None and no_gap["Time"] == pd.Timestamp(2026, 10, 16, 23, 42)
    assert no_gap["Note"] == "named in Messages.csv; it lost no records, so the data show no Index gap"
    assert "missing after Index 1000; named in Messages.csv; the data show an Index gap only\n" in text
    assert "10/16/2026 23:42:00.000; named in Messages.csv; it lost no records" in text
    ## That gap was the restart's, so no "Lost N records" is due for it.
    assert "Lost records Messages.csv names: 0, as the Index gaps outside restarts show." in text


def _no_ack(tmp_path: Path, **kw) -> str:
    spec, results = _choice()
    _recording(tmp_path, spec, results=results, **kw)
    return _report(DFM.load(1, Parameters.two_well(), data_dir=tmp_path / "data"))[1]


def test_the_no_ack_note_appears_only_when_nothing_was_lost_or_repeated(tmp_path):
    bits = {1: [(500, 0x01), (900, 0x01), (1300, 0x03)]}
    quiet = _no_ack(tmp_path / "quiet", error_bits=bits)
    assert "    No Ack: 3\n    Overflow: 1\n" in quiet
    assert f"    OInterrupt Error: 0\n  {NO_ACK_NOTE}\n\n## Index increment check" in quiet
    assert NO_ACK_NOTE == ("No Ack with nothing lost or repeated: PyMCU builds before 10/17/2026 sent "
                           "a partial ACK the firmware took as missed. No data were lost.")
    ## Records lost, or written twice, or no No Ack at all: no note.
    assert NO_ACK_NOTE not in _no_ack(tmp_path / "lost", error_bits=bits, index_gaps={1: [(2000, 3)]})
    assert NO_ACK_NOTE not in _no_ack(tmp_path / "overflow", error_bits={1: [(500, 0x02)]})
    spec, results = _choice()
    _recording(tmp_path / "repeated", spec, results=results, error_bits=bits)
    _repeat_block(tmp_path / "repeated" / "data" / "DFM1_0.csv", 2000, 2010)
    _report_, text = _report(DFM.load(1, Parameters.two_well(), data_dir=tmp_path / "repeated" / "data"))
    assert "Repeated records: 10" in text and NO_ACK_NOTE not in text
    ## Firmware 4.0.0's own: generation 2 (older firmware) has no note.
    assert NO_ACK_NOTE not in _no_ack(tmp_path / "gen2", error_bits=bits, writer=write_gen2_recording)


def test_lost_record_messages_are_checked_against_the_index_gaps(tmp_path):
    spec, results = _choice()
    _recording(tmp_path, spec, results=results, index_gaps={1: [(1000, 20), (3000, 7)]})
    data = tmp_path / "data"
    _report_, text = _report(DFM.load(1, Parameters.two_well(), data_dir=data))
    assert "Lost records Messages.csv names: 27, as the Index gaps outside restarts show.\n" in text
    ## Over the rows loaded: the first gap is before minute 5.
    _report_, text = _report(DFM.load(1, Parameters.two_well(), data_dir=data, range_minutes=(5, 12)))
    assert "Lost records Messages.csv names: 7, as the Index gaps outside restarts show.\n" in text
    ## A gap Messages.csv does not name is a difference.
    _drop_messages(data / "Messages.csv", "Lost 20 records")
    _report_, text = _report(DFM.load(1, Parameters.two_well(), data_dir=data))
    assert ("Lost records Messages.csv names: 7, but the Index gaps outside restarts hold 27 "
            "(20 more in the data).\n") in text


def test_lost_records_beside_a_restart_are_checked_apart_from_it(tmp_path):
    spec, results = _restart()
    _recording(tmp_path, spec, results=results, index_gaps={1: [(1200, 7)]})
    report, text = _report(DFM.load(1, Parameters.two_well(), data_dir=tmp_path / "data"))
    assert report["lost_records"] == 25 + 7 and len(report["restarts"]) == 1
    assert "Lost records Messages.csv names: 7, as the Index gaps outside restarts show.\n" in text


def test_generations_1_and_2_get_no_restart_section(tmp_path):
    spec, results = _choice()
    _recording(tmp_path, spec, results=results, writer=write_gen2_recording, index_gaps={1: [(1000, 5)]})
    report, text = _report(DFM.load(1, Parameters.two_well(), data_dir=tmp_path / "data"))
    assert report["restarts"] is None
    assert "## DFM restarts" not in text and "Lost records Messages.csv names" not in text
    write_opto_csv(tmp_path / "gen1")
    report, text = _report(DFM.load(1, Parameters.single_well(), data_dir=tmp_path / "gen1"))
    assert report["restarts"] is None and "## DFM restarts" not in text


def test_the_qc_reports_list_restarts_in_the_text_not_the_csv(tmp_path):
    spec, results = _restart()
    exp = _load(_recording(tmp_path, spec, results=results))
    with contextlib.redirect_stdout(io.StringIO()):
        out = exp.write_qc_reports()
    integrity = pd.read_csv(out / "integrity" / "DFM2_integrity_report.csv")
    assert "restarts" not in integrity.columns and integrity["lost_records"].iloc[0] == 15
    assert "## DFM restarts\nRestarts: 1\n" in (out / "integrity" / "DFM2_integrity_report.txt").read_text()
    assert [r["Missing"] for r in exp.dfms[2].restarts()] == [15]


# ---------------------------------------------------------------------------
# Generation 1: the text barely changes
# ---------------------------------------------------------------------------

#: ``integrity_report``'s text for ``write_opto_csv``'s recording before this
#: stream, Phase 0's Generation line included.
_GENERATION_1_BEFORE = (
    "\n==============================\nDFM Integrity Report\nDFM ID: 1\nRows in RawData: 18000\n"
    "Generation: 1\n==============================\n\n## Experiment timing\n"
    "Start time: 2025-07-28 12:00:00.500000+00:00\nEnd time:   2025-07-28 13:00:00.300000+00:00\n"
    "Elapsed:    59.997 minutes\nElapsed (from Minutes column): 59.997 minutes\n\n"
    "## Error column details\nFound error-like column(s): Error\n\nColumn: Error\n"
    "  Non-NA: 18000  NA: 0\n  Flagged entries: 0\n\n## Index increment check\n"
    "Index increments by exactly 1 for all consecutive rows.\n\n")


def test_a_generation_1_text_gains_only_the_zero_counts(tmp_path):
    write_opto_csv(tmp_path)
    dfm = DFM.load(1, Parameters.single_well(), data_dir=tmp_path)
    report, text = _report(dfm)
    expected = _GENERATION_1_BEFORE.replace(
        "consecutive rows.\n\n", "consecutive rows.\nLost records: 0\nRepeated records: 0\n\n")
    assert text == expected
    assert report["lost_records"] == 0 and report["repeated_records"] == 0
    assert report["unknown_instruction_rows"] is None
    assert report["late_records"] is None and report["paradigm_sequence"] is None


def test_generation_1_counts_the_two_low_error_bits_together(tmp_path):
    path = write_opto_csv(tmp_path)
    frame = pd.read_csv(path)
    frame.loc[[100, 200, 300, 400], "Error"] = [0x01, 0x02, 0x03, 0x04]
    frame.to_csv(path, index=False)
    _report_, text = _report(DFM.load(1, Parameters.single_well(), data_dir=tmp_path))
    assert "    No ACK / overflow: 3\n    PacketType Error: 1\n" in text
    assert "No Ack:" not in text and "Overflow:" not in text


def test_generation_1_lost_records_come_from_the_index(tmp_path):
    path = write_opto_csv(tmp_path)
    frame = pd.read_csv(path)
    frame.drop(index=range(1000, 1020)).to_csv(path, index=False)
    dfm = DFM.load(1, Parameters.single_well(), data_dir=tmp_path)
    report, text = _report(dfm)
    minute = float(dfm.raw_df["Minutes"].iloc[999])
    assert report["lost_record_gaps"] == [(1000, 20, pytest.approx(minute))]
    assert report["lost_records"] == 20
    assert "Lost records: 20 in 1 gap(s)" in text


# ---------------------------------------------------------------------------
# Repeated records: dropped at load, with a note
# ---------------------------------------------------------------------------

def _repeat_block(path: Path, start: int, stop: int) -> None:
    """Write rows ``[start, stop)`` a second time right after themselves, as a
    retried file write leaves them."""
    frame = pd.read_csv(path, keep_default_na=False, dtype=str)
    pd.concat([frame.iloc[:stop], frame.iloc[start:stop], frame.iloc[stop:]]) \
        .to_csv(path, index=False)


def test_a_repeated_block_is_dropped_at_load_with_a_note(tmp_path):
    clean_root = make_opto_dir(tmp_path / "clean")
    root = make_opto_dir(tmp_path / "repeated")
    _repeat_block(root / "data" / "DFM1_0.csv", 1000, 1020)
    clean, exp = _load(clean_root), _load(root)
    dfm = exp.dfms[1]
    pd.testing.assert_frame_equal(dfm.raw_df, clean.dfms[1].raw_df)
    assert dfm.repeated_records == 20
    (note,) = dfm.load_notes
    assert note.startswith("20 repeated record(s) dropped at load (Index 1001-1020)")
    assert dfm.with_params(dfm.params).load_notes == dfm.load_notes
    report, text = _report(dfm)
    assert report["repeated_records"] == 20 and report["index_increments_by_one"] is True
    assert report["lost_records"] == 0
    assert f"Repeated records: 20\n  {note}\n" in text
    assert f"- load note: {note}" in exp.summary_text(include_qc=False)


def test_an_index_counted_again_with_new_records_is_kept(tmp_path):
    """A DFM that restarts counts Index from 1 again: those records are new
    (their time stamps say so), not repeats."""
    path = write_opto_csv(tmp_path)
    frame = pd.read_csv(path)
    frame.loc[2000:, "Index"] = np.arange(1, len(frame) - 2000 + 1)
    frame.to_csv(path, index=False)
    dfm = DFM.load(1, Parameters.single_well(), data_dir=tmp_path)
    assert len(dfm.raw_df) == len(frame)
    assert dfm.repeated_records == 0 and dfm.load_notes == ()
    report, _text = _report(dfm)
    assert report["index_increments_by_one"] is False and report["lost_records"] == 0


def test_a_repeated_block_leaves_the_paradigm_state_row_for_row(tmp_path):
    spec, results = _choice()
    _recording(tmp_path / "clean", spec, results=results)
    _recording(tmp_path / "repeated", spec, results=results)
    _repeat_block(tmp_path / "repeated" / "data" / "DFM1_0.csv", 2000, 2050)
    clean = DFM.load(1, Parameters.two_well(), data_dir=tmp_path / "clean" / "data")
    dfm = DFM.load(1, Parameters.two_well(), data_dir=tmp_path / "repeated" / "data")
    assert dfm.repeated_records == 50
    assert len(dfm.paradigm) == len(dfm.raw_df) == len(clean.raw_df)
    assert np.array_equal(dfm.paradigm.index, dfm.raw_df["Index"])
    for name in ("index", "activations", "on_ms", "phase_a"):
        assert np.array_equal(getattr(dfm.paradigm, name), getattr(clean.paradigm, name))
    assert dfm.paradigm.index_gaps() == []


# ---------------------------------------------------------------------------
# Crosstalk before and after the dual-feeding correction (decision 23)
# ---------------------------------------------------------------------------

def _crosstalk_dir(root: Path, *, corrected: bool) -> Path:
    """The Progressive Ratio fixture with chamber 1 of DFM 1 licked at both
    wells at once for 16 samples, the left well the stronger."""
    make_pr_experiment_dir(root)
    path = root / "data" / "DFM1_0.csv"
    frame = pd.read_csv(path)
    frame.loc[3000:3015, "W1"] += 12.0
    frame.loc[3000:3015, "W2"] += 9.0
    frame.to_csv(path, index=False)
    cfg = yaml.safe_load((root / "flic_config.yaml").read_text())
    cfg["global"]["params"]["correct_for_dual_feeding"] = corrected
    (root / "flic_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return root


@pytest.mark.parametrize("corrected", [False, True])
def test_the_crosstalk_table_carries_both_counts(tmp_path, corrected):
    exp = _load(_crosstalk_dir(tmp_path, corrected=corrected))
    with contextlib.redirect_stdout(io.StringIO()):
        qc = exp.compute_qc_results()
    matrix = qc[1]["simultaneous_feeding_matrix"]
    assert list(matrix.columns) == ["Licks1", "Licks2", "Both", "MaxMinSignalAtBoth",
                                    "HigherInCol1AtBoth", "Both_pre",
                                    "MaxMinSignalAtBoth_pre", "HigherInCol1AtBoth_pre"]
    pre = exp.dfms[1].lick_matrix
    assert np.array_equal(matrix[["Both_pre", "MaxMinSignalAtBoth_pre",
                                  "HigherInCol1AtBoth_pre"]].to_numpy(), pre[:, 2:5])
    both, both_pre = matrix.loc["Chamber1", "Both"], matrix.loc["Chamber1", "Both_pre"]
    assert both_pre >= 10
    if corrected:
        ## The correction gave every such sample to the stronger well.
        assert both == 0
    else:
        assert np.array_equal(matrix[["Both", "MaxMinSignalAtBoth", "HigherInCol1AtBoth"]]
                              .to_numpy(), pre[:, 2:5])
    ## The report's table shows both.
    blocks = pdf_report._crosstalk_blocks(exp, qc)
    table = next(b for b in blocks if isinstance(b, pdf_report.rl.Table)).frame
    first = table[(table["DFM"] == 1) & (table["Chamber"] == 1)].iloc[0]
    assert first["Both at once, before correction"] == both_pre
    assert first["Both at once, after"] == both


# ---------------------------------------------------------------------------
# The report's integrity table
# ---------------------------------------------------------------------------

def test_the_report_counts_lost_records(tmp_path):
    spec, results = _choice()
    exp = _load(_recording(tmp_path / "gaps", spec, results=results,
                           index_gaps={1: [(1000, 20)]}))
    with contextlib.redirect_stdout(io.StringIO()):
        qc = exp.compute_qc_results()
    frame = pdf_report._integrity_frame(exp, qc)
    row = frame.iloc[0]
    assert row["Lost records"] == 20 and row["Verdict"] == "warning"
    assert "20 lost record(s)" in pdf_report._integrity_callout(exp, qc).text

    clean = _load(make_opto_dir(tmp_path / "opto"))
    with contextlib.redirect_stdout(io.StringIO()):
        qc = clean.compute_qc_results()
    row = pdf_report._integrity_frame(clean, qc).iloc[0]
    assert row["Lost records"] == 0 and row["Verdict"] == "ok"


# ---------------------------------------------------------------------------
# The Program timeline
# ---------------------------------------------------------------------------

_SPECS = {
    "choice_pr": lambda: choice_pr_spec(minutes=8.0, dfm_ids=(1,)),
    "one_of_each": lambda: one_of_each_spec(minutes=6.0),
    "fixed_ratio": lambda: fixed_ratio_spec(minutes=6.0),
    "repeating_open_loop": lambda: repeating_open_loop_spec(minutes=20.0),
    "yoked_controls": lambda: yoked_controls_spec(minutes=6.0),
    "stress": lambda: stress_spec(minutes=4.0),
}


def _lane_labels(fig) -> list[str]:
    lanes = fig.axes[1]
    return [t.get_text() for t in lanes.get_yticklabels()]


@pytest.mark.parametrize("name", list(_SPECS))
def test_the_timeline_renders_for_every_spec(tmp_path, name):
    spec = _SPECS[name]()
    exp = _load(_recording(tmp_path, spec))
    assert pt.timeline_dfms(exp) == sorted(d.dfm_id for d in spec.dfms)
    for dfm_id in pt.timeline_dfms(exp):
        fig = pt.plot_dfm(exp, dfm_id, binsize_min=2.0)
        band = fig.axes[0]
        runs = exp.dfms[dfm_id].paradigm.interval_runs()
        assert len(band.patches) == len(runs)
        labels = _lane_labels(fig)
        assert len(labels) == 12 and labels[0].startswith("W1  ch 1")
        fig.savefig(io.BytesIO(), format="png", dpi=40)
    if name == "yoked_controls":
        fig = pt.plot_dfm(exp, 1)
        labels = _lane_labels(fig)
        ## A trigger well says so, at its threshold, in bold; its yoked partner does not.
        assert labels[0] == "W1  ch 1  Paired  trigger 20" and labels[6] == "W7  ch 4  Yoked"
        weights = [t.get_fontweight() for t in fig.axes[1].get_yticklabels()]
        assert weights[0] == "bold" and weights[6] == "normal"
    if name == "fixed_ratio":
        ## No well lit regardless: the strip is scaled to the DFM's brightest bin.
        assert "scaled to this DFM" in pt.plot_dfm(exp, 1).axes[-1].get_ylabel()
    if name == "repeating_open_loop":
        texts = [t.get_text() for t in pt.plot_dfm(exp, 1).axes[0].texts]
        assert "Open · C1" in texts and "Open · C2" in texts


def test_the_timeline_of_a_generation_1_recording_with_a_program(tmp_path):
    exp = _load(make_opto_dir(tmp_path / "opto"))
    assert pt.timeline_dfms(exp) == [1]
    fig = pt.plot_dfm(exp, 1)
    band, lanes = fig.axes[0], fig.axes[1]
    assert len(band.patches) == 1                       # one interval, the whole hour
    assert "clock schedule" in band.get_title(loc="left")
    ## OptoCol1: a heat strip per lane, no Acclimation, and a tick at each
    ## light onset of a trigger well (the wells with a threshold), whose
    ## label is bold and names it.
    from matplotlib.collections import LineCollection, QuadMesh

    triggers, open_wells = pt.well_thresholds(pt._section(exp, 1))
    assert triggers == {1: (20,), 2: (20,), 3: (20,), 7: (20,)} and open_wells == {5, 6}
    assert sum(isinstance(c, QuadMesh) for c in lanes.collections) == 12
    lights = exp.dfms[1].lights_df
    onsets = sum(len(pt._onsets(lights[f"W{w}"].to_numpy())) for w in triggers)
    ticks = sum(len(c.get_segments()) for c in lanes.collections if isinstance(c, LineCollection))
    assert ticks == onsets > 0
    labels = _lane_labels(fig)
    assert labels[0] == "W1  ch 1  Self  trigger 20" and labels[4] == "W5  ch 5  Open  On"
    assert [t.get_fontweight() == "bold" for t in lanes.get_yticklabels()] == \
        [w in triggers for w in range(1, 13)]
    ## Wells On: the strip runs to a whole bin lit.
    assert fig.axes[-1].get_ylabel() == "% of each 10-min bin lit"
    fig.savefig(io.BytesIO(), format="png", dpi=40)
    table = pt.well_table(exp, 1)
    assert list(table.columns) == list(pt.WELL_TABLE_COLUMNS)
    assert table["Activations"].isna().all() and table["Phase"].isna().all()
    assert table["LitSec"].tolist() == pytest.approx(exp.dfms[1].light_seconds().tolist())

    bare = _load(make_opto_dir(tmp_path / "bare", program=None))
    assert pt.timeline_dfms(bare) == []
    placeholder = pt.plot_dfm(bare, 1)
    assert "no Program timeline" in placeholder.axes[0].texts[0].get_text()


def test_the_well_table_is_the_state_at_the_end_of_the_range(tmp_path):
    spec, results = _choice()
    exp = _load(_recording(tmp_path, spec, results=results))
    table = pt.well_table(exp, 1)
    state = exp.dfms[1].paradigm
    assert table["Well"].tolist() == list(range(1, 13))
    assert table["Paradigm"].unique().tolist() == ["PR10"]
    assert table["Activations"].tolist() == state.activations[-1].tolist()
    truth = results[1].truth
    for w in range(1, 13, 2):
        assert table.loc[w - 1, "Activations"] == len(truth.activation_onset_ms[w])
    never = 2 * next(c for c, case in CHOICE_PR_CASES.items() if case == "never_training") - 1
    assert table.loc[never - 1, "Phase"] == "A"
    assert pd.isna(table.loc[never - 1, "AcclimationEndMin"])
    healthy = table.loc[0]
    assert healthy["Phase"] == "M" and healthy["AcclimationEndMin"] > 2.0
    assert healthy["Requirement"] == state.requirement[-1, 0]
    assert table.loc[1, "Requirement"] is pd.NA
    assert table["LitSec"].tolist() == pytest.approx(exp.dfms[1].light_seconds().tolist())


def test_qc_reports_and_basic_analysis_write_the_timeline(tmp_path):
    spec = fixed_ratio_spec(minutes=6.0)
    exp = _load(_recording(tmp_path / "qc", spec))
    with contextlib.redirect_stdout(io.StringIO()):
        out = exp.write_qc_reports()
    assert (out / "program" / "program_timeline_dfm1.png").is_file()
    integrity = pd.read_csv(out / "integrity" / "DFM1_integrity_report.csv")
    assert {"lost_records", "repeated_records", "unknown_instruction_rows",
            "late_records"} <= set(integrity.columns)
    assert "paradigm_sequence" not in integrity.columns
    assert "Paradigm sequence:" in (out / "integrity" / "DFM1_integrity_report.txt").read_text()

    exp = _load(_recording(tmp_path / "basic", spec))
    with contextlib.redirect_stdout(io.StringIO()):
        paths = exp.execute_basic_analysis(skip_qc=True)
    assert paths["program_timeline_dfm1"].is_file()
    assert paths["program_timeline_dfm1"] == pt.timeline_path(exp.qc_dir, 1)


def test_the_script_action_is_gated_on_optogenetic(tmp_path):
    spec = fixed_ratio_spec(minutes=6.0)
    exp = _load(_recording(tmp_path / "v4", spec))
    lines: list[str] = []
    figures = run_experiment_script(
        exp, {"name": "t", "steps": [{"action": "plot_program_timeline", "binsize": 2}]},
        log=lines.append)
    assert [title for title, _fig in figures] == ["Program timeline — DFM 1"]
    assert pt.timeline_path(exp.qc_dir, 1).is_file()

    dark = _load(make_opto_dir(tmp_path / "dark", program=None, light_cases=False))
    lines = []
    figures = run_experiment_script(
        dark, {"name": "t", "steps": [{"action": "plot_program_timeline"}]}, log=lines.append)
    assert figures == [] and any("[Skip] plot_program_timeline" in line for line in lines)


def test_the_light_strip_fits_sparse_light_and_keeps_open_loop_whole():
    import numpy as np

    ## Feeding-driven light fills a few percent of a bin: rounded up to 1, 2 or 5.
    assert pt._scale_top(np.array([[0.0, 0.031]]), full=False) == pytest.approx(0.05)
    assert pt._scale_top(np.array([[0.0, 0.0012]]), full=False) == pytest.approx(0.002)
    assert pt._scale_top(np.array([[0.0, 0.02]]), full=False) == pytest.approx(0.02)
    ## Open-loop light, or none at all: the whole bin.
    assert pt._scale_top(np.array([[0.0, 0.031]]), full=True) == 1.0
    assert pt._scale_top(np.zeros((3, 12)), full=False) == 1.0
    assert pt._onsets(np.array([0, 1, 1, 0, 1, 0, 0, 1])).tolist() == [1, 4, 7]
