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
) -> Path:
    """Write ``DFM<id>_0.csv``.

    *training_end* maps group -> minute the group finished training (``None``
    = never).  *flag_mode* ``"paired_only"`` mimics the real firmware seen so
    far (only the paired sucrose well clears; the other three stay flagged);
    ``"all_clear"`` clears all four wells together; ``"none"`` writes no flag.
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
