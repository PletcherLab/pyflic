"""Synthetic optogenetic recordings, and the Program.txt the MCU would export.

A single-well Custom experiment on one DFM whose twelve wells each act out
one case the optogenetic light QC must tell apart (see :data:`WELL_CASES`):
a healthy closed loop, a drifting sensor, a stuck LED, light on a well the
program switched off, open loop lit and unlit, and a dead LED under feeding.
The CSV carries real ``Date``/``Time``/``MSec`` stamps and the firmware's
``OptoFreq``/``OptoPW``/``Dark`` columns, so the program is aligned by clock
time and cross-checked, as on a rig.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SPS = 5
START = datetime(2025, 7, 28, 12, 0, 0)
DECAY_MS = 1000
DECAY_SAMPLES = 5            # 1000 ms at 5 Hz

#: What each well does.  Threshold is the program's, per well.
WELL_CASES: dict[int, tuple[str, int]] = {
    1: ("healthy closed loop", 20),
    2: ("drifting sensor", 20),
    3: ("stuck LED", 20),
    4: ("lit while off", -1),
    5: ("open loop, lit", 0),
    6: ("open loop, half lit", 0),
    7: ("dead LED under feeding", 20),
    8: ("quiet", -1),
    9: ("quiet", -1),
    10: ("quiet", -1),
    11: ("quiet", -1),
    12: ("quiet", -1),
}


def _stamps(n: int, start: datetime = START, *, offset_ms: float = 500.0):
    """``Date``, ``Time`` and ``MSec`` columns for *n* samples at 5 Hz."""
    base = start + timedelta(milliseconds=offset_ms)
    times = [base + timedelta(seconds=k / SPS) for k in range(n)]
    date = [t.strftime("%m/%d/%Y") for t in times]
    clock = [t.strftime("%H:%M:%S") for t in times]
    msec = [t.microsecond / 1000.0 for t in times]
    return date, clock, msec


def write_opto_csv(data_dir: Path, dfm_id: int = 1, *, minutes: float = 60.0,
                   seed: int = 7, freq: int = 40, pulse_width: int = 8,
                   start: datetime = START, offset_ms: float = 500.0,
                   light_cases: bool = True) -> Path:
    """Write ``DFM<id>_0.csv`` acting out :data:`WELL_CASES`.

    With *light_cases* False every LED stays dark (``OptoCol1`` all zero)
    while the signals are unchanged — a recording without a lid."""
    rng = np.random.default_rng(seed)
    n = int(minutes * 60 * SPS)
    wells = {w: 8.0 + rng.normal(0.0, 0.3, size=n) for w in range(1, 13)}
    lit = {w: np.zeros(n, dtype=bool) for w in range(1, 13)}

    def burst(w: int, start_idx: int, length: int = 10, amp: float = 40.0) -> None:
        stop = min(start_idx + length, n)
        wells[w][start_idx:stop] += amp + rng.normal(0, 0.5, size=stop - start_idx)

    def light(w: int, a: int, b: int) -> None:
        lit[w][max(0, a):min(n, b)] = True

    ## 1: a healthy closed loop — every feeding bout lights the well, and the
    ## light decays one second after it.
    for k in range(20):
        a = 600 + k * 800
        burst(1, a)
        light(1, a, a + 10 + DECAY_SAMPLES)
    ## 2: a drifting sensor — from minute 10 the resting level climbs 60
    ## counts, the firmware reads it as contact and lights the well for long
    ## stretches; the running baseline follows the drift, so no lick shows.
    ramp_from = 10 * 60 * SPS
    wells[2][ramp_from:] += np.linspace(0.0, 60.0, n - ramp_from)
    for k in range(3):
        a = 300 + k * 1500
        burst(2, a)
        light(2, a, a + 10 + DECAY_SAMPLES)
    above = (wells[2] - wells[2][:50].mean()) > 20
    light_mask = above.copy()
    light_mask[:ramp_from] = False
    lit[2] |= light_mask
    ## 3: a stuck LED — lit for five minutes with no contact at all.
    for k in range(4):
        a = 900 + k * 2500
        burst(3, a)
        light(3, a, a + 10 + DECAY_SAMPLES)
    light(3, 30 * 60 * SPS, 35 * 60 * SPS)
    ## 4: threshold -1, yet lit for ten seconds.
    light(4, 20 * 60 * SPS, 20 * 60 * SPS + 50)
    ## 5: open loop, lit throughout; 6: open loop, lit only the first half.
    light(5, 0, n)
    light(6, 0, n // 2)
    ## 7: a dead LED — the fly feeds, the well never lights.
    for k in range(8):
        burst(7, 1000 + k * 1600)

    opto = np.zeros(n, dtype=np.int64)
    if light_cases:
        for w in range(1, 13):
            opto |= lit[w].astype(np.int64) << (w - 1)
    date, clock, msec = _stamps(n, start, offset_ms=offset_ms)
    cols: dict[str, object] = {"Date": date, "Time": clock, "MSec": msec,
                               "Sample": np.arange(1, n + 1)}
    for w in range(1, 13):
        cols[f"W{w}"] = np.round(wells[w], 2)
    cols.update({"Temp": 25.0, "Humid": 50.0, "LUX": 0.0, "VoltsIn": 5.0, "Dark": 0,
                 "OptoFreq": freq, "OptoPW": pulse_width, "OptoCol1": opto,
                 "OptoCol2": 0, "Error": 0})
    cols["Index"] = np.arange(1, n + 1)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"DFM{dfm_id}_0.csv"
    pd.DataFrame(cols).to_csv(path, index=False)
    return path


def interval_line(thresholds, *, start: datetime = START, duration_min: float = 60.0,
                  freq: int = 40, pw: int = 8, decay: int = DECAY_MS, delay: int = 0,
                  max_on: int = 0, dark: bool = False) -> str:
    stamp = start.strftime("%m/%d/%Y %H:%M:%S")
    values = ",".join(str(int(t)) for t in thresholds)
    return (f"({stamp}) Dark {'On' if dark else 'Off'},{values},F:{freq},P:{pw},"
            f"D:{decay},L:{delay},M:{max_on},{duration_min:.1f}min.")


def program_text(sections: dict[int, dict], *, start: datetime = START,
                 minutes: float = 60.0, baseline: str = "Yes") -> str:
    """An MCU-exported Program.txt.  Each section is ``{"linkage": [...12],
    "type": "Linear", "intervals": [line, ...]}``."""
    end = start + timedelta(minutes=minutes)
    fmt = "%m/%d/%Y %H:%M:%S"
    lines = [f"Start Time: {start.strftime(fmt)}", f"End Time: {end.strftime(fmt)}",
             f"Duration: {minutes:.1f} min",
             "Default Linkage: 1,2,3,4,5,6,7,8,9,10,11,12",
             "Default Opto Frequency: 40Hz", "Default Opto Pulsewidth: 8ms",
             "Default Opto Delay: 0ms", "Default Opto Decay: 0ms",
             "Default Max Time On: 0ms", "Default Program Type: Linear",
             f"Baseline: {baseline}", ""]
    for dfm_id, section in sections.items():
        linkage = section.get("linkage", list(range(1, 13)))
        lines += [f"***DFM {dfm_id}***",
                  "Linkage: " + ",".join(str(v) for v in linkage),
                  "Opto Frequency: 40Hz", "Opto Pulsewidth: 8ms", "Opto Delay: 0ms",
                  "Opto Decay: 0ms", "Max Time On: 0ms",
                  f"Program Type: {section.get('type', 'Linear')}",
                  *section["intervals"], ""]
    return "\n".join(lines) + "\n"


def case_program(minutes: float = 60.0, **interval) -> str:
    """The program behind :func:`write_opto_csv`: closed loop, one second of
    decay, every well its own linkage group."""
    thresholds = [WELL_CASES[w][1] for w in range(1, 13)]
    return program_text({1: {"intervals": [
        interval_line(thresholds, duration_min=minutes, **interval)]}},
        minutes=minutes)


def opto_config(*, optogenetics=None, constants: dict | None = None,
                dfm_extra: dict | None = None) -> dict:
    g: dict = {
        "chamber_layout": "single_well",
        "transform_licks": False,
        "params": {"feeding_threshold": 20, "feeding_minimum": 10,
                   "tasting_minimum": 5, "tasting_maximum": 20,
                   "samples_per_second": SPS},
    }
    if optogenetics is not None:
        g["optogenetics"] = optogenetics
    if constants:
        g["constants"] = dict(constants)
    node = {"id": 1, "chambers": {w: ("Ctrl" if w % 2 else "Exp") for w in range(1, 13)}}
    node.update(dfm_extra or {})
    return {"global": g, "dfms": [node]}


def make_opto_dir(root: Path, *, program: str | None = "default", optogenetics=None,
                  constants: dict | None = None, dfm_extra: dict | None = None,
                  **csv) -> Path:
    """An Experiment Directory: :func:`write_opto_csv`'s recording, its
    config, and — unless *program* is ``None`` — ``data/Program.txt``."""
    root.mkdir(parents=True, exist_ok=True)
    data = root / "data"
    write_opto_csv(data, **csv)
    if program is not None:
        text = case_program() if program == "default" else program
        (data / "Program.txt").write_text(text, encoding="utf-8")
    cfg = opto_config(optogenetics=optogenetics, constants=constants,
                      dfm_extra=dfm_extra)
    (root / "flic_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return root


def pr_program(paired: dict[int, tuple[int, int, int]], *, pi_direction: str = "left",
               minutes: float = 40.0, decay: int = 500) -> str:
    """A progressive-ratio Program.txt for :mod:`pr_fixtures` recordings: the
    trigger well of each chamber group is its paired chamber's well A, and the
    group's four wells share one linkage number."""
    sections = {}
    for dfm_id, chambers in paired.items():
        thresholds = [-1] * 12
        for chamber in chambers:
            left, right = 2 * chamber - 1, 2 * chamber
            thresholds[(left if pi_direction == "left" else right) - 1] = 20
        sections[dfm_id] = {
            "linkage": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], "type": "Repeating",
            "intervals": [interval_line(thresholds, duration_min=minutes, freq=5160,
                                        decay=decay, delay=1000,
                                        max_on=1000000000000)]}
    return program_text(sections, minutes=minutes)
