"""Synthetic Progressive Ratio recordings for tests.

A v3-shaped DFM CSV (``Sample``, ``Seconds``, ``W1..W12``, ``OptoCol1``) whose
wells carry feeding bursts, whose training flag (raw value + 65536) clears per
Chamber Group at a chosen minute, and whose ``OptoCol1`` bits light all four
wells of a group whenever its paired chamber feeds at the sucrose well.  Small
enough to load in well under a second.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SPS = 5
GROUP_WELLS = {1: (1, 2, 3, 4), 2: (5, 6, 7, 8), 3: (9, 10, 11, 12)}
CHAMBER_WELLS = {c: (2 * c - 1, 2 * c) for c in range(1, 7)}


def sucrose_well(chamber: int, pi_direction: str) -> int:
    left, right = CHAMBER_WELLS[chamber]
    return left if pi_direction == "left" else right


def write_pr_csv(
    data_dir: Path,
    dfm_id: int,
    *,
    minutes: float = 40.0,
    paired: tuple[int, int, int] = (1, 4, 5),
    pi_direction: str = "left",
    training_end: dict[int, float | None] | None = None,
    flag_mode: str = "paired_only",
    seed: int | None = None,
    group_modes: dict[int, str] | None = None,
) -> Path:
    """Write ``DFM<id>_0.csv``.

    *training_end* maps group -> minute the group finished training (``None``
    = never).  *flag_mode* ``"paired_only"`` mimics the real firmware seen so
    far (only the paired sucrose well clears; the other three stay flagged);
    ``"all_clear"`` clears all four wells together; ``"none"`` writes no flag.

    *group_modes* rewrites a group's paired Sucrose Well and light (light QC
    fixtures; every other well keeps the default pattern):

    * ``"ratio"`` — a working progressive ratio: three training pairings, then
      Test light events each earned by a longer burst (4k + 4 lick samples);
    * ``"stops"`` — the same, but the fly gives up after three Test events;
    * ``"self_triggered"`` — trains normally, then the well's resting level
      climbs 80 counts over the Test phase and the light fires on its own
      clock with no licks, the failure seen in real data;
    * ``"bad_training"`` — the well sits 60 counts high from the first sample,
      nine training light events fire within 20 s without a lick, and the
      Test light runs on its own clock.
    """
    rng = np.random.default_rng(dfm_id if seed is None else seed)
    n = int(minutes * 60 * SPS)
    seconds = np.arange(n) / SPS
    mins = seconds / 60.0
    if training_end is None:
        training_end = {1: 6.0, 2: 9.0, 3: 12.0}

    wells = {w: rng.normal(0.0, 0.3, size=n) for w in range(1, 13)}
    opto = np.zeros(n, dtype=int)

    def burst(w: int, start: int, length: int = 8, amp: float = 10.0) -> None:
        stop = min(start + length, n)
        wells[w][start:stop] += amp + rng.normal(0, 0.5, size=stop - start)

    paired_set = set(paired)
    for chamber in range(1, 7):
        group = (chamber + 1) // 2
        sw = sucrose_well(chamber, pi_direction)
        other = [w for w in CHAMBER_WELLS[chamber] if w != sw][0]
        is_paired = chamber in paired_set
        ## Paired flies feed more at sucrose than yoked ones, so the diff is
        ## positive and not trivially zero.
        n_sucrose = 14 if is_paired else 7
        step = n // (n_sucrose + 1)
        for k in range(n_sucrose):
            start = step * (k + 1) + (chamber * 11) % 40
            burst(sw, start)
            if is_paired:
                ## The light follows the paired fly's sucrose feeding, for the
                ## whole group (all four bits), for ~3 s.
                bits = sum(1 << (w - 1) for w in GROUP_WELLS[group])
                opto[start:min(start + 15, n)] |= bits
        for k in range(4):
            burst(other, (n // 5) * (k + 1) + (chamber * 17) % 50)

    for group, mode in (group_modes or {}).items():
        if mode in (None, "normal"):
            continue
        chamber = next(c for c in paired if (c + 1) // 2 == group)
        sw = sucrose_well(chamber, pi_direction)
        bits = sum(1 << (w - 1) for w in GROUP_WELLS[group])
        wells[sw] = rng.normal(0.0, 0.3, size=n)
        opto &= ~bits
        end = training_end.get(group)
        end_idx = n if end is None else int(end * 60 * SPS)

        def light_at(start: int, length: int = 15, _bits: int = bits) -> None:
            opto[start:min(start + length, n)] |= _bits

        def clock_light(start: int) -> None:
            ## The firmware counting every sample of a "touched" well: a light
            ## event at intervals that grow by a fixed step, and no licks.
            t, k = start, 1
            while t + 3 < n:
                light_at(t, 3)
                t += 60 + 5 * k
                k += 1

        if mode in ("ratio", "stops", "self_triggered"):
            for k in range(3):
                start = int(end_idx * (k + 1) / 4)
                burst(sw, start)
                light_at(start + 8)
        if mode in ("ratio", "stops"):
            t = end_idx + 200
            for k in range(1, (3 if mode == "stops" else 12) + 1):
                length = 4 * k + 4
                if t + length + 15 >= n:
                    break
                burst(sw, t, length=length)
                light_at(t + length)
                t += length + 15 + 250
        elif mode == "self_triggered":
            wells[sw][end_idx:] += np.linspace(0.0, 80.0, n - end_idx)
            clock_light(end_idx + 100)
        elif mode == "bad_training":
            wells[sw] += 60.0
            for k in range(9):
                light_at(max(0, end_idx - 100) + 11 * k, 3)
            clock_light(end_idx + 100)
        else:
            raise ValueError(f"unknown group mode {mode!r}")

    ## Training flag: raw value + 65536 while in training.
    if flag_mode != "none":
        for group, ws in GROUP_WELLS.items():
            end = training_end.get(group)
            end_idx = n if end is None else int(end * 60 * SPS)
            for chamber in [c for c in range(1, 7) if (c + 1) // 2 == group]:
                sw = sucrose_well(chamber, pi_direction)
                for w in CHAMBER_WELLS[chamber]:
                    clears = (flag_mode == "all_clear"
                              or (chamber in paired_set and w == sw))
                    stop = end_idx if clears else n
                    wells[w][:stop] += 65536.0

    cols: dict[str, np.ndarray] = {
        "Sample": np.arange(1, n + 1),
        "Seconds": seconds,
    }
    for w in range(1, 13):
        cols[f"W{w}"] = np.round(wells[w], 3)
    cols["OptoCol1"] = opto
    cols["OptoCol2"] = np.zeros(n, dtype=int)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"DFM{dfm_id}_0.csv"
    pd.DataFrame(cols).to_csv(path, index=False)
    return path


def pr_config(dfms: list[dict], *, factors: dict | None = None,
              constants: dict | None = None) -> dict:
    """A Progressive Ratio ``flic_config.yaml`` mapping.  Each *dfms* entry
    is ``{"id", "pi_direction", "paired_chambers", "chambers"}``."""
    g: dict = {
        "experiment_type": "ProgressiveRatio",
        "transform_licks": False,
        "params": {
            "feeding_threshold": 5, "feeding_minimum": 5,
            "tasting_minimum": 1, "tasting_maximum": 4,
            "feeding_event_link_gap": 2, "samples_per_second": SPS,
            "correct_for_dual_feeding": False,
        },
        "well_names": {"A": "Sucrose", "B": "Yeast"},
    }
    if factors is not None:
        g["experimental_design_factors"] = factors
    if constants:
        g["constants"] = constants
    nodes = []
    for d in dfms:
        nodes.append({
            "id": int(d["id"]),
            "params": {"pi_direction": d.get("pi_direction", "left")},
            "paired_chambers": list(d["paired_chambers"]),
            "chambers": {int(k): v for k, v in d["chambers"].items()},
        })
    return {"global": g, "dfms": nodes}


def make_pr_experiment_dir(
    root: Path,
    *,
    flag_mode: str = "paired_only",
    incomplete_group_on_dfm2: bool = True,
    constants: dict | None = None,
) -> Path:
    """Two DFMs, factor Genotype (w1118 / mut), one group per treatment
    spread so both treatments appear on both DFMs."""
    root.mkdir(parents=True, exist_ok=True)
    data = root / "data"
    write_pr_csv(data, 1, paired=(1, 4, 5), pi_direction="left",
                 training_end={1: 6.0, 2: 9.0, 3: 12.0}, flag_mode=flag_mode)
    write_pr_csv(data, 2, paired=(2, 3, 6), pi_direction="right",
                 training_end={1: 7.0, 2: 10.0,
                               3: None if incomplete_group_on_dfm2 else 11.0},
                 flag_mode=flag_mode)
    cfg = pr_config(
        [
            {"id": 1, "pi_direction": "left", "paired_chambers": [1, 4, 5],
             "chambers": {1: "w1118", 2: "w1118", 3: "mut", 4: "mut",
                          5: "w1118", 6: "w1118"}},
            {"id": 2, "pi_direction": "right", "paired_chambers": [2, 3, 6],
             "chambers": {1: "mut", 2: "mut", 3: "w1118", 4: "w1118",
                          5: "mut", 6: "mut"}},
        ],
        factors={"Genotype": ["w1118", "mut"]},
        constants=constants,
    )
    (root / "flic_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return root


def make_pr_failure_dir(root: Path, *, constants: dict | None = None) -> Path:
    """Light QC fixture: one DFM whose three groups are a working progressive
    ratio (1), a self-triggering sensor (2) and implausible training (3), and
    a second DFM whose group 1 stops after three Test events while groups 2
    and 3 work."""
    root.mkdir(parents=True, exist_ok=True)
    data = root / "data"
    ## An hour, so the Resting Level's 30-minute start window ends well
    ## before the recording does.
    write_pr_csv(data, 1, minutes=60.0, paired=(1, 3, 5), pi_direction="left",
                 training_end={1: 6.0, 2: 6.0, 3: 3.0},
                 group_modes={1: "ratio", 2: "self_triggered", 3: "bad_training"})
    write_pr_csv(data, 2, minutes=60.0, paired=(1, 3, 5), pi_direction="left",
                 training_end={1: 6.0, 2: 7.0, 3: 8.0},
                 group_modes={1: "stops", 2: "ratio", 3: "ratio"})
    chambers = {1: "w1118", 2: "w1118", 3: "mut", 4: "mut", 5: "mut", 6: "mut"}
    cfg = pr_config(
        [{"id": 1, "pi_direction": "left", "paired_chambers": [1, 3, 5],
          "chambers": dict(chambers)},
         {"id": 2, "pi_direction": "left", "paired_chambers": [1, 3, 5],
          "chambers": dict(chambers)}],
        factors={"Genotype": ["w1118", "mut"]},
        constants=constants,
    )
    (root / "flic_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return root
