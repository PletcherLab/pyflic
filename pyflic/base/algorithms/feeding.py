from __future__ import annotations

import numpy as np
import pandas as pd

from .events import expand_events, get_intervals
from ..parameters import Parameters


def bout_durations_and_intervals(
    baselined: pd.DataFrame,
    event_df: pd.DataFrame,
    *,
    params: Parameters,
) -> tuple[dict[str, pd.DataFrame | int], dict[str, pd.DataFrame | int]]:
    """
    Port of `Set.Durations.And.Intervals()` returning dicts of per-well DataFrames.
    """

    durations: dict[str, pd.DataFrame | int] = {}
    intervals: dict[str, pd.DataFrame | int] = {}

    spm = float(params.samples_per_second)
    for well in range(1, 13):
        cname = f"W{well}"
        data = baselined[cname].to_numpy()
        events = event_df[cname].to_numpy()

        starts = np.flatnonzero(events > 0)
        bout_durs = events[starts].astype(int)

        if starts.size == 0:
            durations[cname] = 0
        else:
            mins = baselined.loc[starts, "Minutes"].to_numpy()
            max_int = np.zeros_like(starts, dtype=float)
            min_int = np.zeros_like(starts, dtype=float)
            sum_int = np.zeros_like(starts, dtype=float)
            avg_int = np.zeros_like(starts, dtype=float)
            var_int = np.zeros_like(starts, dtype=float)

            n = len(data)
            actual_lengths = np.zeros_like(starts, dtype=int)
            for i, (idx, L) in enumerate(zip(starts, bout_durs, strict=False)):
                lo = int(idx)
                hi = min(n, lo + int(L))
                actual_lengths[i] = hi - lo
                seg = data[lo:hi]
                max_int[i] = float(np.max(seg))
                min_int[i] = float(np.min(seg))
                sum_int[i] = float(np.sum(seg))
                avg_int[i] = float(np.mean(seg))
                var_int[i] = float(np.var(seg, ddof=1)) if seg.size > 1 else float("nan")

            dur_df = pd.DataFrame(
                {
                    "Minutes": mins,
                    "Licks": actual_lengths,
                    "Duration": actual_lengths.astype(float) / spm,
                    "TotalIntensity": sum_int,
                    "AvgIntensity": avg_int,
                    "MinIntensity": min_int,
                    "MaxIntensity": max_int,
                    "VarIntensity": var_int,
                }
            )
            durations[cname] = dur_df

        licks_bool = expand_events(events)
        bout_int = get_intervals(licks_bool)
        idxs = np.flatnonzero(bout_int > 0)
        if idxs.size == 0:
            intervals[cname] = 0
        else:
            tmp = baselined.iloc[idxs]
            ints_df = pd.DataFrame(
                {
                    "Minutes": tmp["Minutes"].to_numpy(),
                    "Sample": tmp["Sample"].to_numpy() if "Sample" in tmp.columns else idxs + 1,
                    "IntervalSec": bout_int[idxs].astype(float) / spm,
                }
            )
            intervals[cname] = ints_df

    return durations, intervals
