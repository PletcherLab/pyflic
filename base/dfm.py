from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .algorithms.baseline import baseline_subtract
from .algorithms.feeding import bout_durations_and_intervals
from .algorithms.tasting import tasting_bout_durations_and_intervals
from .algorithms.thresholds import build_thresholds_table
from .chamber import OneWellChamber, TwoWellChamber, compute_feeding_for_well, compute_tasting_for_well
from .io import load_dfm_csvs
from .parameters import Parameters
import re
from .utils import range_is_specified


@dataclass(slots=True)
class DFM:
    id: int
    params: Parameters
    raw_df: pd.DataFrame
    version: int = 2
    source_files: list[Path] | None = None

    # Computed
    baseline_df: pd.DataFrame | None = None
    thresholds: dict[str, pd.DataFrame] | None = None
    lick_df: pd.DataFrame | None = None
    event_df: pd.DataFrame | None = None
    tasting_df: pd.DataFrame | None = None
    tasting_event_df: pd.DataFrame | None = None
    durations: dict[str, pd.DataFrame | int] | None = None
    intervals: dict[str, pd.DataFrame | int] | None = None
    tasting_durations: dict[str, pd.DataFrame | int] | None = None
    tasting_intervals: dict[str, pd.DataFrame | int] | None = None
    lick_matrix: np.ndarray | None = None
    dual_feeding_data: pd.DataFrame | None = None
    in_training_data: pd.DataFrame | None = None
    chambers: list[OneWellChamber | TwoWellChamber] | None = None

    @classmethod
    def load(
        cls,
        dfm_id: int,
        params: Parameters,
        *,
        data_dir: str | Path = ".",
        range_minutes: Sequence[float] = (0, 0),
        correct_for_dual_feeding: bool | None = None,
    ) -> DFM:
        loaded = load_dfm_csvs(dfm_id, data_dir=data_dir, range_minutes=range_minutes)
        df = loaded.df

        obj = cls(
            id=int(dfm_id),
            params=params,
            raw_df=df,
            version=int(loaded.version),
            source_files=loaded.source_files,
        )

        if correct_for_dual_feeding is None:
            correct_for_dual_feeding = bool(params.correct_for_dual_feeding)

        if obj.version == 3:
            obj._calculate_progressive_ratio_training()

        obj.recompute_all(correct_for_dual_feeding=bool(correct_for_dual_feeding))
        return obj

    def with_params(self, new_params: Parameters) -> DFM:
        """
        Equivalent of R's `ChangeParameterObject()` but returns a new object (no hidden global state).
        """

        new = DFM(
            id=self.id,
            params=new_params,
            raw_df=self.raw_df.copy(),
            version=self.version,
            source_files=self.source_files,
        )
        if new.version == 3:
            new._calculate_progressive_ratio_training()
        new.recompute_all(correct_for_dual_feeding=bool(new_params.correct_for_dual_feeding))
        return new

    def recompute_all(self, *, correct_for_dual_feeding: bool = False) -> None:
        self._calculate_baseline()
        self.thresholds = build_thresholds_table(self.baseline_df, self.params)
        self._build_chambers()
        self._calculate_feeding(correct_for_dual_feeding=correct_for_dual_feeding)
        self._calculate_tasting()

    def _build_chambers(self) -> None:
        """
        Build the chamber objects that compose this DFM.

        - chamber_size==1: 12 OneWellChamber instances (one per well)
        - chamber_size==2: one TwoWellChamber per row of params.chamber_sets
        """

        if self.params.chamber_size == 1:
            self.chambers = [
                OneWellChamber(index=i, dfm_id=self.id, params=self.params, well=i) for i in range(1, 13)
            ]
        elif self.params.chamber_size == 2:
            chambers: list[TwoWellChamber] = []
            for i, (w1, w2) in enumerate(self.params.chamber_sets.tolist()):
                left = int(w1)
                right = int(w2)
                if self.params.pi_direction == "left":
                    well_a, well_b = left, right
                elif self.params.pi_direction == "right":
                    well_a, well_b = right, left
                else:
                    raise ValueError(f"Invalid pi_direction: {self.params.pi_direction!r}")
                chambers.append(
                    TwoWellChamber(
                        index=i + 1,
                        dfm_id=self.id,
                        params=self.params,
                        left_well=left,
                        right_well=right,
                        well_a=well_a,
                        well_b=well_b,
                    )
                )
            self.chambers = chambers
        else:
            raise NotImplementedError("Unsupported chamber_size")

    def _calculate_progressive_ratio_training(self) -> None:
        # Port of `CalculateProgressiveRatioTraining()`.
        df = self.raw_df
        wcols = [f"W{i}" for i in range(1, 13) if f"W{i}" in df.columns]
        in_training_df = df[["Minutes", "Sample", *wcols]].copy() if "Sample" in df.columns else df[
            ["Minutes", *wcols]
        ].copy()
        for col in wcols:
            intraining = pd.to_numeric(df[col], errors="coerce").fillna(0) > 40000
            in_training_df[col] = intraining.to_numpy(dtype=bool)
            df.loc[intraining, col] = pd.to_numeric(df.loc[intraining, col], errors="coerce").fillna(0) - 65536
        self.raw_df = df

        # Port of `GetDoneTrainingInfo()`
        rows = []
        for i in range(1, 13):
            col = f"W{i}"
            if col not in in_training_df.columns:
                continue
            mask = in_training_df[col].to_numpy(dtype=bool)
            if not np.any(mask):
                rows.append({"well": col, "Minutes": np.nan, "Sample": np.nan})
            else:
                rows.append(
                    {
                        "well": col,
                        "Minutes": float(in_training_df.loc[mask, "Minutes"].max()),
                        "Sample": float(in_training_df.loc[mask, "Sample"].max())
                        if "Sample" in in_training_df.columns
                        else float(np.flatnonzero(mask).max() + 1),
                    }
                )
        self.in_training_data = pd.DataFrame(rows)

    def _calculate_baseline(self) -> None:
        # Port of `CalculateBaseline()` with the same window logic.
        df = self.raw_df.copy()
        window_min = float(self.params.baseline_window_minutes)
        # R code uses a fixed 5 samples/sec here.
        window = int(round(window_min * 60 * 5))
        if window % 2 == 0:
            window += 1

        for i in range(1, 13):
            col = f"W{i}"
            if col not in df.columns:
                continue
            df[col] = baseline_subtract(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float), window)

        self.baseline_df = df

    def _calculate_feeding(self, *, correct_for_dual_feeding: bool) -> None:
        if self.chambers is None:
            self._build_chambers()

        lick_df = self.baseline_df.copy()
        event_df = self.baseline_df.copy()

        for chamber in self.chambers:
            for well in chamber.wells:
                cname = f"W{well}"
                res = compute_feeding_for_well(
                    baselined_well=lick_df[cname].to_numpy(dtype=float),
                    thresholds_well=self.thresholds[cname],
                    params=self.params,
                )
                lick_df[cname] = res.licks
                event_df[cname] = res.events

        lick_matrix = None
        dual_feeding_data = None

        if self.params.chamber_size == 2:
            lick_matrix = self._check_for_simultaneous_feeding(lick_df, self.baseline_df)
            if correct_for_dual_feeding:
                adj = self._adjust_baseline_for_dual_feeding(lick_df, self.baseline_df)
                dual_feeding_data = adj["dual_feeding_data"]
                adj_baselined = adj["baselined"]

                lick_df2 = adj_baselined.copy()
                event_df2 = adj_baselined.copy()
                for chamber in self.chambers:
                    for well in chamber.wells:
                        cname = f"W{well}"
                        res = compute_feeding_for_well(
                            baselined_well=adj_baselined[cname].to_numpy(dtype=float),
                            thresholds_well=self.thresholds[cname],
                            params=self.params,
                        )
                        lick_df2[cname] = res.licks
                        event_df2[cname] = res.events

                lick_df = lick_df2
                event_df = event_df2

        self.lick_df = lick_df
        self.event_df = event_df
        self.lick_matrix = lick_matrix
        self.dual_feeding_data = dual_feeding_data

        self.durations, self.intervals = bout_durations_and_intervals(
            self.baseline_df, self.event_df, params=self.params
        )

    def _calculate_tasting(self) -> None:
        if self.chambers is None:
            self._build_chambers()

        tasting_df = self.baseline_df.copy()
        tasting_event_df = self.baseline_df.copy()

        for chamber in self.chambers:
            for well in chamber.wells:
                cname = f"W{well}"
                res = compute_tasting_for_well(
                    baselined_well=self.baseline_df[cname].to_numpy(dtype=float),
                    thresholds_well=self.thresholds[cname],
                    feeding_licks_well=self.lick_df[cname].to_numpy(dtype=bool),
                    params=self.params,
                )
                tasting_df[cname] = res.licks
                tasting_event_df[cname] = res.events

        self.tasting_df = tasting_df
        self.tasting_event_df = tasting_event_df
        self.tasting_durations, self.tasting_intervals = tasting_bout_durations_and_intervals(
            self.baseline_df,
            self.tasting_event_df,
            params=self.params,
            interval_source_lick_df=self.lick_df,  # matches R behavior
        )

    # ---- Accessors ----
    def raw(self, *, range_minutes: Sequence[float] = (0, 0)) -> pd.DataFrame:
        if not range_is_specified(range_minutes):
            return self.raw_df
        a, b = float(range_minutes[0]), float(range_minutes[1])
        return self.raw_df[(self.raw_df["Minutes"] > a) & (self.raw_df["Minutes"] <= b)]

    def baselined(self, *, range_minutes: Sequence[float] = (0, 0)) -> pd.DataFrame:
        if not range_is_specified(range_minutes):
            return self.baseline_df
        a, b = float(range_minutes[0]), float(range_minutes[1])
        df = self.baseline_df
        return df[(df["Minutes"] > a) & (df["Minutes"] <= b)]

    # ---- QC ----
    def data_breaks(self, *, multiplier: float = 4.0) -> pd.DataFrame | None:
        """Port of FindDataBreaks() from QC.R."""
        if "Seconds" not in self.raw_df.columns:
            return None
        seconds = pd.to_numeric(self.raw_df["Seconds"], errors="coerce").to_numpy(dtype=float)
        interval = np.diff(seconds, prepend=seconds[0])
        thresh = (1.0 / float(self.params.samples_per_second)) * float(multiplier)
        idx = np.flatnonzero(interval > thresh)
        if idx.size == 0:
            return None
        out = self.raw_df.iloc[idx].copy()
        out.insert(0, "Interval", interval[idx])
        out.insert(0, "Index", idx + 1)
        return out.reset_index(drop=True)

    def integrity_report(self) -> dict[str, object]:
        """Port of ReportDFMIntegrity() from QC.R."""
        _ERROR_BIT_NAMES = [
            "I2C Error", "ID Error", "PacketType Error", "DMA_TX Error",
            "DMA_RX Error", "PacketSize Error", "AIInterrupt Error", "OInterrupt Error",
        ]
        raw = self.raw_df
        n = len(raw)
        print("\n==============================")
        print("DFM Integrity Report")
        print(f"DFM ID: {self.id}")
        print(f"Rows in RawData: {n}")
        print("==============================\n")

        start_time = end_time = elapsed_minutes = elapsed_minutes_from_minutes_col = None

        if all(c in raw.columns for c in ("Date", "Time")) and n > 0:
            tmp = str(raw["Time"].iloc[0])
            is_ampm = ("AM" in tmp.upper()) or ("PM" in tmp.upper()) or (re.search(r".M$", tmp) is not None)
            fmt = "%m/%d/%Y %I:%M:%S %p" if is_ampm else "%m/%d/%Y %H:%M:%S"
            ts = pd.to_datetime(raw["Date"].astype(str) + " " + raw["Time"].astype(str), format=fmt, utc=True, errors="coerce")
            if "MSec" in raw.columns:
                ts = ts + pd.to_timedelta(pd.to_numeric(raw["MSec"], errors="coerce").fillna(0.0) / 1000.0, unit="s")
            if ts.notna().any():
                start_time = ts.iloc[0]
                end_time = ts.iloc[-1]
                elapsed_minutes = float((end_time - start_time).total_seconds() / 60.0)

        if "Minutes" in raw.columns and n > 1:
            mins = pd.to_numeric(raw["Minutes"], errors="coerce")
            if mins.notna().any():
                elapsed_minutes_from_minutes_col = float(mins.max() - mins.min())

        print("## Experiment timing")
        if start_time is not None and pd.notna(start_time) and pd.notna(end_time):
            print(f"Start time: {start_time}")
            print(f"End time:   {end_time}")
            print(f"Elapsed:    {elapsed_minutes:.3f} minutes")
        else:
            print("Start/End time: (Date/Time/MSec not found or not parseable)")
        if elapsed_minutes_from_minutes_col is not None:
            print(f"Elapsed (from Minutes column): {elapsed_minutes_from_minutes_col:.3f} minutes")
        print()

        err_cols = [c for c in raw.columns if re.search("error", c, flags=re.IGNORECASE)]
        print("## Error column details")
        if not err_cols:
            print("No column with name matching /error/i found in RawData.\n")
        else:
            print(f"Found error-like column(s): {', '.join(err_cols)}\n")
            for ec in err_cols:
                v = raw[ec]
                if pd.api.types.is_numeric_dtype(v) or pd.api.types.is_bool_dtype(v):
                    vv = pd.to_numeric(v, errors="coerce")
                    flagged = vv.notna() & (vv != 0)
                else:
                    vs = v.astype(str).fillna("")
                    flagged = (vs != "") & (vs != "0") & (vs.str.lower() != "na")
                print(f"Column: {ec}")
                print(f"  Non-NA: {int(v.notna().sum())}  NA: {int(v.isna().sum())}")
                print(f"  Flagged entries: {int(flagged.sum())}")
                if int(flagged.sum()) > 0:
                    codes = pd.to_numeric(v, errors="coerce").astype("Int64")
                    codes = codes[flagged].dropna().astype(int)
                    if len(codes) > 0:
                        bit_mat = np.vstack([((codes.to_numpy() & (1 << i)) > 0) for i in range(8)]).T
                        type_counts = bit_mat.sum(axis=0)
                        print("  Error-type counts (among flagged rows):")
                        for name, cnt in zip(_ERROR_BIT_NAMES, type_counts, strict=False):
                            print(f"    {name}: {int(cnt)}")
                print()

        print("## Index increment check")
        index_ok = None
        if "Index" not in raw.columns:
            print("No `Index` column found in RawData.\nSkipping increment-by-one analysis.\n")
        else:
            ix = pd.to_numeric(raw["Index"], errors="coerce")
            if ix.notna().sum() < 2:
                print("Index column has <2 valid entries; cannot evaluate increments.\n")
            else:
                d = np.diff(ix.to_numpy(dtype=float))
                index_ok = bool(np.all(d == 1))
                print("Index increments by exactly 1 for all consecutive rows.\n" if index_ok else "Index does NOT always increment by 1.\n")

        return {
            "dfm_id": self.id,
            "n_rawdata": n,
            "start_time": start_time,
            "end_time": end_time,
            "elapsed_minutes": elapsed_minutes,
            "elapsed_minutes_from_minutes_col": elapsed_minutes_from_minutes_col,
            "error_columns": err_cols,
            "index_increments_by_one": index_ok,
        }

    def simultaneous_feeding_matrix(self) -> np.ndarray:
        """Port of CheckForSimultaneousFeeding.DFM() from QC.R (two-well only)."""
        if self.params.chamber_size != 2:
            raise ValueError("simultaneous_feeding_matrix is for chamber_size==2 only")
        if self.lick_df is None or self.baseline_df is None:
            raise ValueError("DFM must have computed lick/baseline data first")
        return self._check_for_simultaneous_feeding(self.lick_df, self.baseline_df)

    def bleeding_check(self, cutoff: float) -> dict[str, object]:
        """Port of CheckForBleeding.DFM() from QC.R (data return only; no plot)."""
        if self.baseline_df is None:
            raise ValueError("DFM must have baseline computed first")
        wcols = [f"W{i}" for i in range(1, 13)]
        missing = [c for c in wcols if c not in self.baseline_df.columns]
        if missing:
            raise ValueError(f"baseline_df missing required well columns: {missing}")
        rd = self.baseline_df[wcols].apply(pd.to_numeric, errors="coerce")
        mat = np.full((12, 12), np.nan, dtype=float)
        for i in range(12):
            tmp = rd.loc[rd.iloc[:, i] > float(cutoff)]
            mat[i, :] = 0.0 if len(tmp) == 0 else tmp.mean(axis=0, skipna=True).to_numpy(dtype=float)
        mat = np.nan_to_num(mat, nan=0.0)
        matrix_df = pd.DataFrame(mat, index=[f"W{i}Sig" for i in range(1, 13)], columns=[f"W{i}Resp" for i in range(1, 13)])
        all_data = rd.mean(axis=0, skipna=True)
        all_data.index = wcols
        return {"Matrix": matrix_df, "AllData": all_data}

    def _check_for_simultaneous_feeding(self, lick_df: pd.DataFrame, baselined: pd.DataFrame) -> np.ndarray:
        """Port of CheckForSimultaneousFeeding.DFM() from QC.R."""
        n = int(self.params.chamber_sets.shape[0])
        mat = np.full((n, 5), np.nan, dtype=float)
        for i in range(n):
            w1, w2 = self.params.chamber_sets[i, 0], self.params.chamber_sets[i, 1]
            c1, c2 = f"W{w1}", f"W{w2}"
            l1 = lick_df[c1].to_numpy(dtype=bool)
            l2 = lick_df[c2].to_numpy(dtype=bool)
            both = l1 & l2
            if np.sum(both) > 0:
                sig1 = baselined[c1].to_numpy(dtype=float)
                sig2 = baselined[c2].to_numpy(dtype=float)
                mat[i, :] = [float(np.sum(l1)), float(np.sum(l2)), float(np.sum(both)),
                             float(np.max(np.minimum(sig1[both], sig2[both]))),
                             float(np.sum(sig1[both] > sig2[both]))]
            else:
                mat[i, :] = [float(np.sum(l1)), float(np.sum(l2)), 0.0, 0.0, 0.0]
        return mat

    def _adjust_baseline_for_dual_feeding(self, lick_df: pd.DataFrame, baselined: pd.DataFrame) -> dict[str, object]:
        """Port of Adjust.Baseline.For.Dual.Feeding.DFM() from PrivateFunctions.R."""
        wcols = [f"W{i}" for i in range(1, 13)]
        rd = lick_df[wcols].to_numpy(dtype=bool)
        rd2 = baselined[wcols].to_numpy(dtype=float).copy()
        dual_rows: list[pd.DataFrame] = []
        n = int(self.params.chamber_sets.shape[0])
        for i in range(n):
            w1, w2 = self.params.chamber_sets[i, 0], self.params.chamber_sets[i, 1]
            l1, l2 = rd[:, w1 - 1], rd[:, w2 - 1]
            both = l1 & l2
            if not np.any(both):
                continue
            sig1, sig2 = rd2[:, w1 - 1], rd2[:, w2 - 1]
            smaller_1 = both & (sig1 < sig2)
            smaller_2 = both & (sig2 < sig1)
            same = both & (sig2 == sig1)
            rd2[smaller_1, w1 - 1] = 0.0
            rd2[smaller_2, w2 - 1] = 0.0
            rd2[same, w1 - 1] = 0.0
            rd2[same, w2 - 1] = 0.0
            tmp = baselined.loc[both].copy()
            tmp.insert(0, "Chamber", i + 1)
            dual_rows.append(tmp)
        adjusted = baselined.copy()
        adjusted[wcols] = rd2
        return {"baselined": adjusted, "dual_feeding_data": pd.concat(dual_rows, ignore_index=True) if dual_rows else pd.DataFrame()}

    # ---- Stats ----
    @staticmethod
    def _chamber_from_well(well: int, params) -> int:
        if params.chamber_size == 1:
            return well
        return (well - 1) // 2 + 1

    @staticmethod
    def _tcwell_from_well(well: int, params) -> str | None:
        if params.chamber_size != 2:
            return None
        even = (well % 2 == 0)
        if params.pi_direction == "left":
            return "WellA" if not even else "WellB"
        if params.pi_direction == "right":
            return "WellB" if not even else "WellA"
        raise ValueError(f"Invalid pi_direction: {params.pi_direction!r}")

    @staticmethod
    def _interval_summary(interval_df, range_minutes: Sequence[float]) -> tuple[float, float]:
        if not isinstance(interval_df, pd.DataFrame) or interval_df.empty:
            return 0.0, 0.0
        tmp = interval_df if not range_is_specified(range_minutes) else interval_df[
            (interval_df["Minutes"] > float(range_minutes[0])) & (interval_df["Minutes"] <= float(range_minutes[1]))
        ]
        if tmp.empty:
            return 0.0, 0.0
        return float(tmp["IntervalSec"].mean()), float(tmp["IntervalSec"].median())

    @staticmethod
    def _duration_summary(dur_df, range_minutes: Sequence[float]) -> tuple[float, float]:
        if not isinstance(dur_df, pd.DataFrame) or dur_df.empty:
            return np.nan, np.nan
        tmp = dur_df if not range_is_specified(range_minutes) else dur_df[
            (dur_df["Minutes"] > float(range_minutes[0])) & (dur_df["Minutes"] <= float(range_minutes[1]))
        ]
        if tmp.empty:
            return np.nan, np.nan
        return float(tmp["Duration"].mean()), float(tmp["Duration"].median())

    @staticmethod
    def _intensity_summary(baselined_well: np.ndarray, event_vec: np.ndarray, *, range_mask=None) -> tuple[float, float, float, float]:
        from .algorithms.events import expand_events
        licks = expand_events(event_vec.astype(int))
        if range_mask is not None:
            licks = licks & range_mask
        da = baselined_well[licks]
        if da.size == 0:
            return np.nan, np.nan, np.nan, np.nan
        return float(np.mean(da)), float(np.median(da)), float(np.min(da)), float(np.max(da))

    def feeding_summary(
        self,
        *,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool = True,
    ) -> pd.DataFrame:
        from .lights import get_lights_info

        raw_r = self._apply_range(self.raw_df, range_minutes)
        lights = get_lights_info(raw_r)

        if self.params.chamber_size == 1:
            lights_sec = lights[[f"W{i}" for i in range(1, 13)]].sum(axis=0).to_numpy() / float(self.params.samples_per_second) if "W1" in lights.columns else np.zeros(12, dtype=float)
            rows: list[dict] = []
            for well in range(1, 13):
                cname = f"W{well}"
                mean_dur, med_dur = self._duration_summary(self.durations[cname], range_minutes)
                mean_intv, med_intv = self._interval_summary(self.intervals[cname], range_minutes)
                base = self.baseline_df[cname].to_numpy(dtype=float)
                ev = self.event_df[cname].to_numpy(dtype=int)
                mask = None
                if range_is_specified(range_minutes):
                    a, b = float(range_minutes[0]), float(range_minutes[1])
                    mins = self.baseline_df["Minutes"].to_numpy(dtype=float)
                    mask = (mins > a) & (mins <= b)
                mean_int, med_int, min_int, max_int = self._intensity_summary(base, ev, range_mask=mask)
                licks = int(self._apply_range(self.lick_df, range_minutes)[cname].sum())
                events = int((self._apply_range(self.event_df, range_minutes)[cname].to_numpy(dtype=int) > 0).sum())
                rows.append({
                    "DFM": self.id, "Chamber": well,
                    "Licks": float(licks), "Events": float(events),
                    "MeanDuration": mean_dur, "MedDuration": med_dur,
                    "MeanTimeBtw": mean_intv, "MedTimeBtw": med_intv,
                    "MeanInt": mean_int, "MedianInt": med_int, "MinInt": min_int, "MaxInt": max_int,
                    "OptoOn_sec": float(lights_sec[well - 1]) if len(lights_sec) == 12 else 0.0,
                    "StartMin": float(range_minutes[0]), "EndMin": float(range_minutes[1]),
                })
            out = pd.DataFrame(rows)
            if transform_licks:
                out["Licks"] = np.power(out["Licks"], 0.25)
            return out

        if self.params.chamber_size == 2:
            lights_sec = lights[[f"W{i}" for i in range(1, 13)]].sum(axis=0).to_numpy() / float(self.params.samples_per_second)
            rows = []
            for chamber in range(self.params.chamber_sets.shape[0]):
                w1, w2 = int(self.params.chamber_sets[chamber, 0]), int(self.params.chamber_sets[chamber, 1])
                if self.params.pi_direction == "left":
                    well_a, well_b = w1, w2
                elif self.params.pi_direction == "right":
                    well_a, well_b = w2, w1
                else:
                    raise ValueError(f"Invalid pi_direction: {self.params.pi_direction!r}")
                ca, cb = f"W{well_a}", f"W{well_b}"
                licks_a = float(self._apply_range(self.lick_df, range_minutes)[ca].sum())
                licks_b = float(self._apply_range(self.lick_df, range_minutes)[cb].sum())
                events_a = float((self._apply_range(self.event_df, range_minutes)[ca].to_numpy(dtype=int) > 0).sum())
                events_b = float((self._apply_range(self.event_df, range_minutes)[cb].to_numpy(dtype=int) > 0).sum())
                mean_dur_a, med_dur_a = self._duration_summary(self.durations[ca], range_minutes)
                mean_dur_b, med_dur_b = self._duration_summary(self.durations[cb], range_minutes)
                mean_intv_a, med_intv_a = self._interval_summary(self.intervals[ca], range_minutes)
                mean_intv_b, med_intv_b = self._interval_summary(self.intervals[cb], range_minutes)
                base_a = self.baseline_df[ca].to_numpy(dtype=float)
                base_b = self.baseline_df[cb].to_numpy(dtype=float)
                ev_a = self.event_df[ca].to_numpy(dtype=int)
                ev_b = self.event_df[cb].to_numpy(dtype=int)
                mask = None
                if range_is_specified(range_minutes):
                    a, b = float(range_minutes[0]), float(range_minutes[1])
                    mins = self.baseline_df["Minutes"].to_numpy(dtype=float)
                    mask = (mins > a) & (mins <= b)
                mean_int_a, med_int_a, min_int_a, max_int_a = self._intensity_summary(base_a, ev_a, range_mask=mask)
                mean_int_b, med_int_b, min_int_b, max_int_b = self._intensity_summary(base_b, ev_b, range_mask=mask)
                pi = (licks_a - licks_b) / (licks_a + licks_b) if (licks_a + licks_b) > 0 else np.nan
                event_pi = (events_a - events_b) / (events_a + events_b) if (events_a + events_b) > 0 else np.nan
                rows.append({
                    "DFM": self.id, "Chamber": chamber + 1,
                    "PI": pi, "EventPI": event_pi,
                    "LicksA": licks_a, "LicksB": licks_b,
                    "EventsA": events_a, "EventsB": events_b,
                    "MeanDurationA": mean_dur_a, "MedDurationA": med_dur_a,
                    "MeanDurationB": mean_dur_b, "MedDurationB": med_dur_b,
                    "MeanTimeBtwA": mean_intv_a, "MedTimeBtwA": med_intv_a,
                    "MeanTimeBtwB": mean_intv_b, "MedTimeBtwB": med_intv_b,
                    "MeanIntA": mean_int_a, "MedianIntA": med_int_a, "MinIntA": min_int_a, "MaxIntA": max_int_a,
                    "MeanIntB": mean_int_b, "MedianIntB": med_int_b, "MinIntB": min_int_b, "MaxIntB": max_int_b,
                    "OptoOn_sec_A": float(lights_sec[well_a - 1]),
                    "OptoOn_sec_B": float(lights_sec[well_b - 1]),
                    "StartMin": float(range_minutes[0]), "EndMin": float(range_minutes[1]),
                })
            out = pd.DataFrame(rows)
            if transform_licks:
                out["LicksA"] = np.power(out["LicksA"], 0.25)
                out["LicksB"] = np.power(out["LicksB"], 0.25)
            return out

        raise NotImplementedError("Feeding summary not implemented for this DFM type.")

    def binned_feeding_summary(
        self,
        *,
        binsize_min: float = 30.0,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool = True,
    ) -> pd.DataFrame:
        if range_is_specified(range_minutes):
            m_min, m_max = float(range_minutes[0]), float(range_minutes[1])
        else:
            m_min, m_max = 0.0, float(self.raw_df["Minutes"].max())
        if m_min > m_max:
            m_max = m_min + 1.0
        y = np.arange(m_min, m_max + 1e-9, binsize_min, dtype=float)
        if y.size == 0 or y[-1] < m_max:
            y = np.append(y, m_max)
        bins = np.column_stack([y[:-1], y[1:]])
        labels = [f"({a},{b}]" for a, b in bins]
        parts: list[pd.DataFrame] = []
        for (a, b), label in zip(bins, labels, strict=False):
            summ = self.feeding_summary(range_minutes=(float(a), float(b)), transform_licks=transform_licks)
            summ.insert(0, "Minutes", float((a + b) / 2.0))
            summ.insert(0, "Interval", label)
            parts.append(summ)
        return pd.concat(parts, ignore_index=True)

    def interval_data(self, *, range_minutes: Sequence[float] = (0, 0)) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for well in range(1, 13):
            cname = f"W{well}"
            the_data = self.intervals.get(cname, 0)
            if not isinstance(the_data, pd.DataFrame) or the_data.empty:
                continue
            tmp = the_data.copy()
            if range_is_specified(range_minutes):
                tmp = self._apply_range(tmp, range_minutes)
            if tmp.empty:
                continue
            tmp.insert(0, "Well", well)
            tmp.insert(0, "TCWell", self._tcwell_from_well(well, self.params) or "")
            tmp.insert(0, "Chamber", self._chamber_from_well(well, self.params))
            tmp.insert(0, "DFM", self.id)
            frames.append(tmp)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def duration_data(self, *, range_minutes: Sequence[float] = (0, 0)) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for well in range(1, 13):
            cname = f"W{well}"
            the_data = self.durations.get(cname, 0)
            if not isinstance(the_data, pd.DataFrame) or the_data.empty:
                continue
            tmp = the_data.copy()
            if range_is_specified(range_minutes):
                tmp = self._apply_range(tmp, range_minutes)
            if tmp.empty:
                continue
            tmp.insert(0, "Well", well)
            tmp.insert(0, "TCWell", self._tcwell_from_well(well, self.params) or "")
            tmp.insert(0, "Chamber", self._chamber_from_well(well, self.params))
            tmp.insert(0, "DFM", self.id)
            frames.append(tmp)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ---- Plots ----
    def _apply_range(self, df: pd.DataFrame, range_minutes: Sequence[float]) -> pd.DataFrame:
        if not range_is_specified(range_minutes):
            return df
        a, b = float(range_minutes[0]), float(range_minutes[1])
        return df[(df["Minutes"] > a) & (df["Minutes"] <= b)]

    def plot_raw(self, *, range_minutes: Sequence[float] = (0, 0)):
        import matplotlib.pyplot as plt

        df = self._apply_range(self.raw_df, range_minutes)
        wcols = [f"W{i}" for i in range(1, 13) if f"W{i}" in df.columns]
        n = len(wcols)
        fig, axes = plt.subplots(n, 1, sharex=True, figsize=(10, max(6, n * 1.3)))
        if n == 1:
            axes = [axes]
        for ax, col in zip(axes, wcols, strict=False):
            ax.plot(df["Minutes"], df[col], linewidth=0.8)
            ax.set_ylabel(col)
        axes[-1].set_xlabel("Minutes")
        fig.suptitle(f"DFM {self.id} Raw")
        fig.tight_layout()
        return fig

    def plot_baselined(self, *, range_minutes: Sequence[float] = (0, 0), include_thresholds: bool = False):
        import matplotlib.pyplot as plt

        df = self._apply_range(self.baseline_df, range_minutes)
        wcols = [f"W{i}" for i in range(1, 13) if f"W{i}" in df.columns]
        n = len(wcols)
        fig, axes = plt.subplots(n, 1, sharex=True, figsize=(10, max(6, n * 1.3)))
        if n == 1:
            axes = [axes]
        for ax, col in zip(axes, wcols, strict=False):
            ax.plot(df["Minutes"], df[col], linewidth=0.8)
            if include_thresholds:
                thr = self.thresholds[col]
                ax.axhline(float(thr["FeedingMax"].iloc[0]), linestyle="--", color="red", linewidth=0.6)
                ax.axhline(float(thr["FeedingMin"].iloc[0]), linestyle="--", color="red", linewidth=0.6)
            ax.set_ylabel(col)
        axes[-1].set_xlabel("Minutes")
        fig.suptitle(f"DFM {self.id} Baselined")
        fig.tight_layout()
        return fig

    def plot_raw_well(self, well: int, *, range_minutes: Sequence[float] = (0, 0)):
        """Plot the raw signal for a single well."""
        import matplotlib.pyplot as plt

        col = f"W{well}"
        df = self._apply_range(self.raw_df, range_minutes)
        if col not in df.columns:
            raise ValueError(f"Well {well} not found in DFM {self.id} raw data (expected column '{col}').")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(df["Minutes"], df[col], linewidth=0.7, color="steelblue")
        ax.set_xlabel("Minutes")
        ax.set_ylabel("Signal")
        ax.set_title(f"DFM {self.id}  W{well}  Raw")
        fig.tight_layout()
        return fig

    def plot_baselined_well(self, well: int, *, range_minutes: Sequence[float] = (0, 0), include_thresholds: bool = False):
        """Plot the baseline-subtracted signal for a single well."""
        import matplotlib.pyplot as plt

        col = f"W{well}"
        df = self._apply_range(self.baseline_df, range_minutes)
        if col not in df.columns:
            raise ValueError(f"Well {well} not found in DFM {self.id} baselined data (expected column '{col}').")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(df["Minutes"], df[col], linewidth=0.7, color="steelblue")
        if include_thresholds and hasattr(self, "thresholds") and col in self.thresholds:
            thr = self.thresholds[col]
            feed_max = float(thr["FeedingMax"].iloc[0])
            feed_min = float(thr["FeedingMin"].iloc[0])
            ax.axhline(feed_max, linestyle="--", color="red", linewidth=0.8, label=f"FeedingMax ({feed_max:.0f})")
            ax.axhline(feed_min, linestyle="--", color="red", linewidth=0.8, label=f"FeedingMin ({feed_min:.0f})")
            ax.legend(fontsize=8, loc="upper right")
        ax.set_xlabel("Minutes")
        ax.set_ylabel("Baselined Signal")
        ax.set_title(f"DFM {self.id}  W{well}  Baselined")
        fig.tight_layout()
        return fig

    def plot_binned_licks(
        self,
        *,
        binsize_min: float = 30.0,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool = True,
    ):
        import matplotlib.pyplot as plt

        binned = self.binned_feeding_summary(binsize_min=binsize_min, range_minutes=range_minutes, transform_licks=transform_licks)
        if binned.empty:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.set_title(f"DFM {self.id} (no data)")
            return fig
        fig, ax = plt.subplots(figsize=(10, 4))
        if "LicksA" in binned.columns:
            binned = binned.copy()
            binned["LicksTotal"] = binned["LicksA"] + binned["LicksB"]
            for chamber, grp in binned.groupby("Chamber"):
                ax.plot(grp["Minutes"], grp["LicksTotal"], marker="o", linewidth=1.0, label=f"Ch{int(chamber)}")
            ax.set_ylabel("Transformed Licks" if transform_licks else "Licks")
            ax.legend(title="Chamber", ncol=3, fontsize=8)
        else:
            for chamber, grp in binned.groupby("Chamber"):
                ax.plot(grp["Minutes"], grp["Licks"], marker="o", linewidth=1.0, label=f"Ch{int(chamber)}")
            ax.set_ylabel("Transformed Licks" if transform_licks else "Licks")
            ax.legend(title="Chamber", ncol=3, fontsize=8)
        ax.set_xlabel("Minutes")
        ax.set_title(f"DFM {self.id} binned feeding")
        fig.tight_layout()
        return fig

    def cumulative_pi_data(self, *, range_minutes: Sequence[float] = (0, 0)) -> pd.DataFrame:
        """Port of CumulativePI.DFM (data part). Two-well only."""
        import numpy as np

        if self.params.chamber_size != 2:
            raise ValueError("cumulative_pi_data is only defined for two-well chambers")
        licks = self._apply_range(self.lick_df, range_minutes)
        minutes = self._apply_range(self.baseline_df, range_minutes)["Minutes"].to_numpy(dtype=float)
        parts: list[pd.DataFrame] = []
        for ch in self.chambers:
            if hasattr(ch, "well_a") and hasattr(ch, "well_b"):
                well_a, well_b = int(ch.well_a), int(ch.well_b)
            else:
                w1, w2 = ch.wells
                if self.params.pi_direction == "left":
                    well_a, well_b = int(w1), int(w2)
                elif self.params.pi_direction == "right":
                    well_a, well_b = int(w2), int(w1)
                else:
                    raise ValueError(f"Invalid pi_direction: {self.params.pi_direction!r}")
            a = licks[f"W{well_a}"].to_numpy(dtype=bool)
            b = licks[f"W{well_b}"].to_numpy(dtype=bool)
            mask = a | b
            if not np.any(mask):
                continue
            ca = np.cumsum(a.astype(int))
            cb = np.cumsum(b.astype(int))
            denom = ca + cb
            pi = np.full(denom.shape, np.nan, dtype=float)
            nz = denom > 0
            pi[nz] = (ca[nz] - cb[nz]) / denom[nz]
            parts.append(pd.DataFrame({
                "Minutes": minutes[mask],
                "PI": pi[mask],
                "Licks": denom[mask].astype(int),
                "Chamber": int(ch.index),
            }))
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["Minutes", "PI", "Licks", "Chamber"])

    def cumulative_event_pi_data(
        self, *, events_limit: int | None = None, range_minutes: Sequence[float] = (0, 0)
    ) -> pd.DataFrame:
        """Port of CumulativeEventPI.DFM (data part). Two-well only."""
        import numpy as np

        if self.params.chamber_size != 2:
            raise ValueError("cumulative_event_pi_data is only defined for two-well chambers")
        events = self._apply_range(self.event_df, range_minutes)
        minutes = self._apply_range(self.baseline_df, range_minutes)["Minutes"].to_numpy(dtype=float)
        parts: list[pd.DataFrame] = []
        for ch in self.chambers:
            if hasattr(ch, "well_a") and hasattr(ch, "well_b"):
                well_a, well_b = int(ch.well_a), int(ch.well_b)
            else:
                w1, w2 = ch.wells
                if self.params.pi_direction == "left":
                    well_a, well_b = int(w1), int(w2)
                elif self.params.pi_direction == "right":
                    well_a, well_b = int(w2), int(w1)
                else:
                    raise ValueError(f"Invalid pi_direction: {self.params.pi_direction!r}")
            a = (events[f"W{well_a}"].to_numpy(dtype=int) > 0).astype(int)
            b = (events[f"W{well_b}"].to_numpy(dtype=int) > 0).astype(int)
            mask = (a + b) > 0
            if not np.any(mask):
                continue
            ca = np.cumsum(a)
            cb = np.cumsum(b)
            denom = ca + cb
            pi = np.full(denom.shape, np.nan, dtype=float)
            nz = denom > 0
            pi[nz] = (ca[nz] - cb[nz]) / denom[nz]
            out = pd.DataFrame({
                "Minutes": minutes[mask],
                "PI": pi[mask],
                "Chamber": int(ch.index),
            })
            out["EventNum"] = np.arange(1, len(out) + 1, dtype=int)
            if events_limit is not None and len(out) > int(events_limit):
                out = out.iloc[: int(events_limit)].reset_index(drop=True)
            parts.append(out)
        cols = ["Minutes", "PI", "Chamber", "EventNum"]
        return pd.concat(parts, ignore_index=True)[cols] if parts else pd.DataFrame(columns=cols)

    def plot_cumulative_pi(self, *, range_minutes: Sequence[float] = (0, 0), single_plot: bool = False):
        import matplotlib.pyplot as plt

        data = self.cumulative_pi_data(range_minutes=range_minutes)
        if data.empty:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.set_title(f"DFM {self.id} cumulative PI (no data)")
            return fig
        chambers = sorted(data["Chamber"].unique().tolist())
        if single_plot:
            fig, ax = plt.subplots(figsize=(10, 4))
            for ch in chambers:
                tmp = data[data["Chamber"] == ch]
                ax.plot(tmp["Minutes"], tmp["PI"], marker=".", linewidth=1.0, label=f"Ch{int(ch)}")
            ax.set_ylim(-1, 1)
            ax.set_xlabel("Minutes")
            ax.set_ylabel("PI (Licks)")
            ax.legend(ncol=3, fontsize=8)
            ax.set_title(f"DFM {self.id} cumulative PI")
            fig.tight_layout()
            return fig
        fig, axes = plt.subplots(len(chambers), 1, sharex=True, figsize=(10, max(4, len(chambers) * 1.6)))
        if len(chambers) == 1:
            axes = [axes]
        for ax, ch in zip(axes, chambers, strict=False):
            tmp = data[data["Chamber"] == ch]
            ax.plot(tmp["Minutes"], tmp["PI"], marker=".", linewidth=1.0)
            ax.set_ylim(-1, 1)
            ax.set_ylabel(f"Ch{int(ch)}")
        axes[-1].set_xlabel("Minutes")
        fig.suptitle(f"DFM {self.id} cumulative PI (Licks)")
        fig.tight_layout()
        return fig

    def plot_cumulative_event_pi(
        self,
        *,
        events_limit: int | None = None,
        range_minutes: Sequence[float] = (0, 0),
        single_plot: bool = False,
        by_bout: bool = False,
    ):
        import matplotlib.pyplot as plt

        data = self.cumulative_event_pi_data(events_limit=events_limit, range_minutes=range_minutes)
        if data.empty:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.set_title(f"DFM {self.id} cumulative event PI (no data)")
            return fig
        chambers = sorted(data["Chamber"].unique().tolist())
        xcol = "EventNum" if by_bout else "Minutes"
        xlabel = "Event Number" if by_bout else "Minutes"
        if single_plot:
            fig, ax = plt.subplots(figsize=(10, 4))
            for ch in chambers:
                tmp = data[data["Chamber"] == ch]
                ax.plot(tmp[xcol], tmp["PI"], marker=".", linewidth=1.0, label=f"Ch{int(ch)}")
            ax.set_ylim(-1, 1)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("PI (Events)")
            ax.legend(ncol=3, fontsize=8)
            ax.set_title(f"DFM {self.id} cumulative EventPI")
            fig.tight_layout()
            return fig
        fig, axes = plt.subplots(len(chambers), 1, sharex=True, figsize=(10, max(4, len(chambers) * 1.6)))
        if len(chambers) == 1:
            axes = [axes]
        for ax, ch in zip(axes, chambers, strict=False):
            tmp = data[data["Chamber"] == ch]
            ax.plot(tmp[xcol], tmp["PI"], marker=".", linewidth=1.0)
            ax.set_ylim(-1, 1)
            ax.set_ylabel(f"Ch{int(ch)}")
        axes[-1].set_xlabel(xlabel)
        fig.suptitle(f"DFM {self.id} cumulative EventPI")
        fig.tight_layout()
        return fig

    def plot_cumulative_licks(self, *, single_plot: bool = False, transform_licks: bool = True):
        import matplotlib.pyplot as plt
        import numpy as np

        first_min = float(self.baseline_df["Minutes"].iloc[0]) if len(self.baseline_df) else 0.0
        last_min = float(self.baseline_df["Minutes"].iloc[-1]) if len(self.baseline_df) else 0.0

        def cumulative_for_well(well: int) -> tuple[np.ndarray, np.ndarray]:
            d = self.durations.get(f"W{well}", 0)
            if not isinstance(d, pd.DataFrame) or d.empty:
                x = np.array([first_min, last_min], dtype=float)
                y = np.array([0.0, 0.0], dtype=float)
            else:
                x = d["Minutes"].to_numpy(dtype=float)
                y = np.cumsum(d["Licks"].to_numpy(dtype=float))
            if transform_licks:
                y = np.power(y, 0.25)
            return x, y

        if single_plot:
            fig, ax = plt.subplots(figsize=(10, 5))
            for well in range(1, 13):
                x, y = cumulative_for_well(well)
                ax.plot(x, y, linewidth=1.0, label=f"W{well}")
            ax.set_xlabel("Minutes")
            ax.set_ylabel("Transformed Cumulative Licks" if transform_licks else "Cumulative Licks")
            ax.set_title(f"DFM {self.id} cumulative licks")
            ax.legend(ncol=6, fontsize=7)
            fig.tight_layout()
            return fig

        if self.params.chamber_size == 1:
            fig, axes = plt.subplots(6, 2, sharex=True, figsize=(12, 10))
            for well in range(1, 13):
                r = (well - 1) // 2
                c = (well - 1) % 2
                ax = axes[r][c]
                x, y = cumulative_for_well(well)
                ax.plot(x, y, linewidth=1.0)
                ax.set_title(f"W{well}", fontsize=9)
            for ax in axes[-1]:
                ax.set_xlabel("Minutes")
            fig.suptitle(f"DFM {self.id} cumulative licks")
            fig.tight_layout()
            return fig

        fig, axes = plt.subplots(len(self.chambers), 2, sharex=True, figsize=(12, max(8, len(self.chambers) * 1.4)))
        if len(self.chambers) == 1:
            axes = [axes]
        for row_idx, ch in enumerate(self.chambers):
            if hasattr(ch, "well_a") and hasattr(ch, "well_b"):
                well_a, well_b = int(ch.well_a), int(ch.well_b)
            else:
                w1, w2 = ch.wells
                if self.params.pi_direction == "left":
                    well_a, well_b = int(w1), int(w2)
                elif self.params.pi_direction == "right":
                    well_a, well_b = int(w2), int(w1)
                else:
                    raise ValueError(f"Invalid pi_direction: {self.params.pi_direction!r}")
            for col_idx, well in enumerate([well_a, well_b]):
                ax = axes[row_idx][col_idx]
                x, y = cumulative_for_well(well)
                ax.plot(x, y, linewidth=1.0)
                ax.set_title(f"Ch{int(ch.index)} {'WellA' if col_idx==0 else 'WellB'} (W{well})", fontsize=9)
        for ax in axes[-1]:
            ax.set_xlabel("Minutes")
        fig.suptitle(f"DFM {self.id} cumulative licks")
        fig.tight_layout()
        return fig

