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
from .paradigm_state import (
    GENERATION_EMPTY,
    GENERATION_FLAG,
    GENERATION_PARADIGM,
    RECORD_MS,
    ParadigmState,
    detect_generation,
    split_baseline_columns,
    split_paradigm_columns,
)
from .parameters import Parameters
import re
from .utils import (
    ENVIRONMENTAL_MONITOR_ID,
    ENVIRONMENTAL_MONITOR_MESSAGE,
    natural_sorted,
    range_bounds,
    range_is_specified,
)

#: ``DFM.training_info`` by Generation: where a recording keeps its Training.
TRAINING_INFO = {GENERATION_FLAG: "flag", GENERATION_EMPTY: "none", GENERATION_PARADIGM: "phase"}

#: Error bits 2-7 mean the same on every V3 firmware (``DFMV3_V2/README.md``,
#: "DFM error codes"); the names are the ones the integrity report has always
#: printed.
_ERROR_BITS_COMMON: tuple[tuple[int, str], ...] = (
    (0x04, "PacketType Error"), (0x08, "DMA_TX Error"), (0x10, "DMA_RX Error"),
    (0x20, "PacketSize Error"), (0x40, "AIInterrupt Error"), (0x80, "OInterrupt Error"),
)
#: ``(mask, label)`` of the ``Error`` byte's bits by Generation, as the
#: integrity report counts them.  Bits 0 and 1 changed meaning: the two
#: firmware 2.0.0 builds put the missing ACK on one or the other and nothing
#: in a generation-1 file says which build wrote it, so generation 1 counts
#: them together; from 3.5.0 (PyMCU's ``DFMErrors.V3_ERROR_DISPLAY``) 0x01 is
#: the missing ACK and 0x02 a status-buffer overflow.
ERROR_BITS: dict[int, tuple[tuple[int, str], ...]] = {
    GENERATION_FLAG: ((0x03, "No ACK / overflow"), *_ERROR_BITS_COMMON),
    GENERATION_EMPTY: ((0x01, "No Ack"), (0x02, "Overflow"), *_ERROR_BITS_COMMON),
    GENERATION_PARADIGM: ((0x01, "No Ack"), (0x02, "Overflow"), *_ERROR_BITS_COMMON),
}
#: The integrity report's note on generation 3 when *No Ack* is counted with
#: nothing lost or repeated: firmware 4.0.0 took PyMCU's partial ACK naming
#: every record as a missed ACK until PyMCU 19efddd sent a plain one.
NO_ACK_NOTE = ("No Ack with nothing lost or repeated: PyMCU builds before 10/17/2026 sent a partial "
               "ACK the firmware took as missed. No data were lost.")


# ── DFM CSV loading (formerly io.py) ────────────────────────────────────────

def _subplots(*args, figsize=None, **kwargs):
    """``plt.subplots`` without pyplot.

    A :class:`~matplotlib.figure.Figure` built directly has no canvas widget
    and touches no pyplot state, so it is safe to build in a worker thread
    (the Hub's QC task) under any backend — through pyplot under QtAgg the
    same call would create a Qt canvas off the GUI thread, which matplotlib
    documents as likely to fail.  ``savefig`` and ``FigureCanvasQTAgg(fig)``
    both work on such a figure.
    """
    from matplotlib.figure import Figure

    fig = Figure(figsize=figsize)
    axes = fig.subplots(*args, **kwargs)
    return fig, axes


#: Minutes of the recording's start a range's clip keeps aside
#: (:class:`BeforeRange`): the firmware captures its baseline in the first ten
#: seconds of the run, which starts with the recording or within a minute of
#: it (a later start is a program mismatch the light QC flags).
START_KEPT_MIN = 5.0

#: A raw sample above this carries the generation-1 training flag: its true
#: value is the raw value minus :data:`_TRAINING_FLAG`.
_TRAINING_FLAG_ABOVE = 40000
_TRAINING_FLAG = 65536


@dataclass(frozen=True, slots=True)
class BeforeRange:
    """What a range's clip cut from the front of the recording that the rows
    it kept still need (ADR-0020).  A Slice that starts after minute 0 is
    analysed from its own rows, but the firmware's baseline was captured at
    the start of the run, and a light on at the Slice's first row may have
    come on before it.

    ``start_min`` is the range's start ``a`` of ``(a, b]``; ``start_minutes``
    and ``start_raw`` the recording's first rows (up to
    :data:`START_KEPT_MIN` minutes) and their raw signal, ``(k, 12)`` with
    wells 1-12 as columns and the training flag removed; ``lit`` each well's
    ``OptoCol1`` light on the last row before the range, ``(12,)``.  On
    generation 3 the Paradigm state keeps its own row before
    (:attr:`ParadigmState.activations_before`).
    """

    start_min: float
    start_minutes: np.ndarray
    start_raw: np.ndarray
    lit: np.ndarray


def _without_training_flag(values: np.ndarray) -> np.ndarray:
    """Raw values with the generation-1 training flag taken off."""
    values = np.asarray(values, dtype=float).copy()
    flagged = np.nan_to_num(values, nan=0.0) > _TRAINING_FLAG_ABOVE
    values[flagged] -= _TRAINING_FLAG
    return values


def _flag_training_table(df: pd.DataFrame) -> pd.DataFrame:
    """Port of ``GetDoneTrainingInfo()``, extended with a per-well verdict:
    one row per well of *df* (the raw, still-flagged signal).

    ``Minutes`` is the last minute the well's own column carried the flag
    (NaN when it never did).  ``Cleared`` says whether the flag went away
    before the recording ended — the firmware clears it on a well whose
    training completed, so a well flagged through the last sample never
    finished (or, for a well that is not the paired sucrose well, simply
    never reports).  Only a cleared well has a training end."""
    minutes = pd.to_numeric(df["Minutes"], errors="coerce").to_numpy(dtype=float)
    samples = (pd.to_numeric(df["Sample"], errors="coerce").to_numpy(dtype=float)
               if "Sample" in df.columns else None)
    rows = []
    for i in range(1, 13):
        col = f"W{i}"
        if col not in df.columns:
            continue
        mask = (pd.to_numeric(df[col], errors="coerce").fillna(0)
                > _TRAINING_FLAG_ABOVE).to_numpy(dtype=bool)
        if not np.any(mask):
            rows.append({"well": col, "Minutes": np.nan, "Sample": np.nan,
                         "EverFlagged": False, "Cleared": False})
            continue
        rows.append({
            "well": col,
            "Minutes": float(minutes[mask].max()),
            "Sample": (float(samples[mask].max()) if samples is not None
                       else float(np.flatnonzero(mask).max() + 1)),
            "EverFlagged": True,
            "Cleared": not bool(mask[-1]),
        })
    return pd.DataFrame(rows)


def _before_range(df: pd.DataFrame, a: float, *, flagged: bool) -> BeforeRange | None:
    """:class:`BeforeRange` of the whole-recording frame *df* for a range
    starting at *a*; ``None`` when the clip cuts nothing from the front."""
    minutes = pd.to_numeric(df["Minutes"], errors="coerce").to_numpy(dtype=float)
    cut = np.flatnonzero(minutes <= a)
    if not cut.size or a <= 0.0:
        return None
    head = minutes <= (minutes[0] + START_KEPT_MIN)
    raw = np.full((int(head.sum()), 12), np.nan)
    for w in range(1, 13):
        if f"W{w}" in df.columns:
            values = pd.to_numeric(df[f"W{w}"], errors="coerce").to_numpy(dtype=float)[head]
            raw[:, w - 1] = _without_training_flag(values) if flagged else values
    lit = np.zeros(12, dtype=bool)
    if "OptoCol1" in df.columns:
        last = pd.to_numeric(df["OptoCol1"], errors="coerce").iloc[int(cut[-1])]
        bits = 0 if pd.isna(last) else int(last)
        lit = (bits & (1 << np.arange(12, dtype=np.int64))) > 0
    return BeforeRange(start_min=float(a), start_minutes=minutes[head].copy(),
                       start_raw=raw, lit=lit)


@dataclass(frozen=True, slots=True)
class _LoadedDFM:
    df: pd.DataFrame
    version: int
    source_files: list[Path]
    generation: int = GENERATION_FLAG
    #: The Paradigm state on generation 3, row for row with *df*; else None.
    paradigm: ParadigmState | None = None
    #: PyMCU's Baseline columns, row for row with *df*; None without them.
    baseline_in_force: np.ndarray | None = None
    #: Rows dropped because their ``Index`` was already in the file.
    repeated_records: int = 0
    #: What the loader changed in the data, for the integrity text and
    #: ``summary.txt``.
    notes: tuple[str, ...] = ()
    #: The last minute of the recording before any range clipped it: whether
    #: a Slice is covered is read from this (ADR-0020).
    recording_end: float = float("nan")
    #: What the clip cut from the front that the kept rows still need.
    before_range: BeforeRange | None = None
    #: Generation 1's per-well training table (:func:`_flag_training_table`)
    #: read from the whole recording before a range clipped it: a range that
    #: starts after a group's training end must not make that group look as
    #: if it never trained.  ``None`` when nothing was clipped.
    training: pd.DataFrame | None = None


class EmptyRangeError(ValueError):
    """The requested range holds no rows of this DFM's recording: it starts
    after the recording ended.  Carries where the recording ended, so a
    Slice load can say why the Member does not cover it."""

    def __init__(self, dfm_id: int, recording_end: float):
        super().__init__(f"DFM {dfm_id}: no data in the requested range (the recording "
                         f"ends at minute {recording_end:.1f})")
        self.dfm_id = int(dfm_id)
        self.recording_end = float(recording_end)

    def __reduce__(self):
        ## A process pool pickles the error back to the parent.
        return (type(self), (self.dfm_id, self.recording_end))


def _index_blocks(values: np.ndarray, limit: int = 3) -> str:
    """``1001-1020, 4000-4003`` for the sorted unique *values*, the first
    *limit* runs and a count of the rest."""
    values = np.unique(values.astype(np.int64))
    if not values.size:
        return ""
    breaks = np.flatnonzero(np.diff(values) != 1) + 1
    runs = [(int(r[0]), int(r[-1])) for r in np.split(values, breaks)]
    text = ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs[:limit])
    return text + (f" and {len(runs) - limit} more block(s)" if len(runs) > limit else "")


def _repeated_index_rows(df: pd.DataFrame) -> np.ndarray:
    """(n,) bool: rows whose ``Index`` and time stamp an earlier row already
    has.

    ``Index`` counts the DFM's records from 1 and rises by 1, so a record
    seen before is one PyMCU wrote again when it retried a file write; the
    first copy is the one kept.  The time stamp (``Date``, ``Time``,
    ``MSec``) is part of the match so that a DFM that restarted, and counts
    its ``Index`` from 1 again with new records, loses nothing.  A missing
    ``Index`` is never a repeat."""
    if "Index" not in df.columns or len(df) < 2:
        return np.zeros(len(df), dtype=bool)
    index = pd.to_numeric(df["Index"], errors="coerce")
    if bool((np.diff(index.to_numpy(dtype=float)) > 0).all()):
        return np.zeros(len(df), dtype=bool)          # the usual file: nothing repeats
    key = [c for c in ("Index", "Date", "Time", "MSec") if c in df.columns]
    return (df.duplicated(subset=key, keep="first") & index.notna()).to_numpy(dtype=bool)


def _clock_times(raw: pd.DataFrame) -> pd.Series | None:
    """Each row's clock time from ``Date``, ``Time`` and ``MSec``, labelled
    UTC as the integrity report prints them (``NaT`` where one does not
    parse); ``None`` without ``Date`` and ``Time`` or without rows."""
    if not all(c in raw.columns for c in ("Date", "Time")) or len(raw) == 0:
        return None
    tmp = str(raw["Time"].iloc[0])
    is_ampm = ("AM" in tmp.upper()) or ("PM" in tmp.upper()) or (re.search(r".M$", tmp) is not None)
    fmt = "%m/%d/%Y %I:%M:%S %p" if is_ampm else "%m/%d/%Y %H:%M:%S"
    ts = pd.to_datetime(raw["Date"].astype(str) + " " + raw["Time"].astype(str), format=fmt, utc=True,
                        errors="coerce")
    if "MSec" in raw.columns:
        ts = ts + pd.to_timedelta(pd.to_numeric(raw["MSec"], errors="coerce").fillna(0.0) / 1000.0, unit="s")
    return ts


def _elapsed_seconds_from_date_time(df: pd.DataFrame) -> np.ndarray:
    dates = df["Date"].astype(str)
    times = df["Time"].astype(str)
    ms = pd.to_numeric(df["MSec"], errors="coerce").fillna(0.0).to_numpy()
    first = times.iloc[0]
    is_ampm = ("AM" in first) or ("PM" in first) or (first.upper().endswith("M"))
    if is_ampm:
        ts = pd.to_datetime(dates + " " + times, format="%m/%d/%Y %I:%M:%S %p", utc=True)
    else:
        ts = pd.to_datetime(dates + " " + times, format="%m/%d/%Y %H:%M:%S", utc=True)
    diffs = (ts - ts.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    return diffs + (ms / 1000.0)


def _ensure_minutes_seconds(df: pd.DataFrame) -> pd.DataFrame:
    if "Seconds" in df.columns:
        seconds = pd.to_numeric(df["Seconds"], errors="coerce").to_numpy(dtype=float)
        out = df.copy()
        out.insert(0, "Minutes", seconds / 60.0)
        return out
    if all(c in df.columns for c in ("Date", "Time", "MSec")):
        seconds = _elapsed_seconds_from_date_time(df)
        out = df.copy()
        out.insert(0, "Seconds", seconds)
        out.insert(0, "Minutes", seconds / 60.0)
        return out
    raise ValueError("Time information missing from DFM data (need Seconds or Date/Time/MSec).")


def _load_dfm_csvs(dfm_id: int, data_dir: str | Path = ".", range_minutes: Sequence[float] = (0, 0)) -> _LoadedDFM:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")
    if int(dfm_id) == ENVIRONMENTAL_MONITOR_ID:
        raise ValueError(ENVIRONMENTAL_MONITOR_MESSAGE)
    v3 = natural_sorted(data_dir.glob(f"DFM{dfm_id}_*.csv"))
    if v3:
        files, version = v3, 3
    else:
        v2_single = data_dir / f"DFM_{dfm_id}.csv"
        if v2_single.exists():
            files, version = [v2_single], 2
        else:
            v2_link = natural_sorted(data_dir.glob(f"DFM_{dfm_id}_*.csv"))
            if v2_link:
                files, version = v2_link, 2
            else:
                raise FileNotFoundError(f"DFM data file(s) not found for id={dfm_id} in {data_dir}")
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    ## The 63 Paradigm columns of a 90-column file come off before anything
    ## else: every derived frame (baseline, licks, events, tasting) starts as
    ## a copy of raw_df, and none of them reads these.  They live on in the
    ## ParadigmState, which takes the same rows as raw_df below.
    generation = detect_generation(df)
    df, paradigm_columns = split_paradigm_columns(df)
    ## The Baseline in force (PyMCU's last 12 columns) comes off too, kept
    ## row for row as an array, like the Paradigm state.
    df, baseline_in_force = split_baseline_columns(df)
    df = _ensure_minutes_seconds(df)
    state = None
    if generation == GENERATION_PARADIGM and paradigm_columns is not None:
        state = ParadigmState.from_frame(
            paradigm_columns, minutes=df["Minutes"].to_numpy(dtype=float),
            index=df["Index"] if "Index" in df.columns else None)
    ## A retried file write leaves records in the file twice: keep the first
    ## copy of each Index, in the data and in the Paradigm state alike.
    repeated = _repeated_index_rows(df)
    n_repeated = int(repeated.sum())
    notes: tuple[str, ...] = ()
    if n_repeated:
        blocks = _index_blocks(pd.to_numeric(df["Index"], errors="coerce")
                               .to_numpy(dtype=float)[repeated])
        notes = (f"{n_repeated} repeated record(s) dropped at load (Index {blocks}): rows a "
                 f"retried file write wrote again; the first copy of each was kept.",)
        df = df[~repeated]
        if state is not None:
            state = state.subset(~repeated)
        if baseline_in_force is not None:
            baseline_in_force = baseline_in_force[~repeated]
    minutes = pd.to_numeric(df["Minutes"], errors="coerce")
    recording_end = float(minutes.max()) if len(minutes) else float("nan")
    before = None
    training = None
    if range_is_specified(range_minutes):
        ## (a, b], the rule every Facet window follows, so adjacent Slices
        ## share no row and lose none at their common boundary.
        a, b = range_bounds(range_minutes)
        ## Read before the clip, as recording_end is: the firmware's baseline,
        ## the light on the row before, and generation 1's training flags all
        ## live in rows the clip removes.
        flagged = version == 3 and generation == GENERATION_FLAG
        before = _before_range(df, a, flagged=flagged)
        if flagged:
            training = _flag_training_table(df)
        keep = ((minutes > a) & (minutes <= b)).to_numpy(dtype=bool)
        df = df[keep]
        if state is not None:
            state = state.subset(keep)
        if baseline_in_force is not None:
            baseline_in_force = baseline_in_force[keep]
        if df.empty:
            raise EmptyRangeError(dfm_id, recording_end)
    return _LoadedDFM(df=df.reset_index(drop=True), version=version, source_files=files,
                      generation=generation, paradigm=state, baseline_in_force=baseline_in_force,
                      repeated_records=n_repeated,
                      notes=notes, recording_end=recording_end, before_range=before,
                      training=training)


@dataclass(slots=True)
class DFM:
    id: int
    params: Parameters
    raw_df: pd.DataFrame
    version: int = 2
    source_files: list[Path] | None = None
    #: The recording's Generation (``CONTEXT.md``): 1 for a 27-column file, 2
    #: for a 90-column one with every Paradigm column NA, 3 when they are
    #: filled.  A second axis beside ``version``, which the file name gives.
    generation: int = GENERATION_FLAG
    #: The firmware's per-row Paradigm state on generation 3, row for row with
    #: ``raw_df``; ``None`` on generations 1 and 2.
    paradigm: ParadigmState | None = None
    #: The Baseline in force behind the DFM's thresholds, ``(n, 12)`` float
    #: row for row with ``raw_df``, Wells 1-12 as columns, NaN where PyMCU
    #: wrote ``NA``: PyMCU's ``Baseline1``-``Baseline12``.  ``None`` for files
    #: without them (PyMCU 3.x and 4.0.0 development builds).
    baseline_in_force: np.ndarray | None = None
    #: Where Training is read from: ``"flag"`` (generation 1, the 65536 flag
    #: in the signal), ``"none"`` (generation 2, no Training information) or
    #: ``"phase"`` (generation 3, each well's Phase).
    training_info: str = "flag"
    #: Records the loader dropped because their ``Index`` was already in the
    #: file (a retried file write), and what it said about them: the
    #: integrity report and ``summary.txt`` show the notes.
    repeated_records: int = 0
    load_notes: tuple[str, ...] = ()
    #: The recording's last minute before a range clipped it (ADR-0020).
    recording_end_minutes: float | None = None
    #: What a range's clip cut from the front that the kept rows still need
    #: (:class:`BeforeRange`); ``None`` when the rows start with the recording.
    before_range: BeforeRange | None = None

    well_names: dict[str, str] | None = None  # e.g. {"A": "Sucrose", "B": "Yeast"}

    # Computed
    baseline_df: pd.DataFrame | None = None
    thresholds: dict[str, pd.DataFrame] | None = None
    lights_df: pd.DataFrame | None = None
    lick_df: pd.DataFrame | None = None
    event_df: pd.DataFrame | None = None
    tasting_df: pd.DataFrame | None = None
    tasting_event_df: pd.DataFrame | None = None
    durations: dict[str, pd.DataFrame | int] | None = None
    intervals: dict[str, pd.DataFrame | int] | None = None
    tasting_durations: dict[str, pd.DataFrame | int] | None = None
    tasting_intervals: dict[str, pd.DataFrame | int] | None = None
    #: The simultaneous-feeding matrix (:meth:`_check_for_simultaneous_feeding`)
    #: of the licks **before** the dual-feeding correction; two-well only.
    #: :meth:`simultaneous_feeding_matrix` is the one after it.
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
        loaded = _load_dfm_csvs(dfm_id, data_dir=data_dir, range_minutes=range_minutes)
        df = loaded.df

        obj = cls(
            id=int(dfm_id),
            params=params,
            raw_df=df,
            version=int(loaded.version),
            source_files=loaded.source_files,
            generation=int(loaded.generation),
            paradigm=loaded.paradigm,
            baseline_in_force=loaded.baseline_in_force,
            training_info=TRAINING_INFO[int(loaded.generation)],
            repeated_records=int(loaded.repeated_records),
            load_notes=tuple(loaded.notes),
            recording_end_minutes=float(loaded.recording_end),
            before_range=loaded.before_range,
        )

        if correct_for_dual_feeding is None:
            correct_for_dual_feeding = bool(params.correct_for_dual_feeding)

        ## Training: a flag inside the signal on generation 1 (only the v3
        ## file name's firmware set it), each well's Phase on generation 3,
        ## and nothing at all on generation 2.
        if obj.generation == GENERATION_PARADIGM:
            obj._calculate_paradigm_training()
        elif obj.generation == GENERATION_EMPTY:
            obj._calculate_no_training()
        elif obj.version == 3:
            obj._calculate_progressive_ratio_training()
            ## A range clipped the rows: the training ends come from the whole
            ## recording, read before the clip, not from the rows kept.
            if loaded.training is not None:
                obj.in_training_data = loaded.training

        obj.recompute_all(correct_for_dual_feeding=bool(correct_for_dual_feeding))
        return obj

    def with_params(self, new_params: Parameters) -> DFM:
        """
        Equivalent of R's `ChangeParameterObject()` but returns a new object (no hidden global state).

        The training flags travel with the copy rather than being read again:
        ``raw_df`` had them stripped when this DFM loaded, so re-reading it
        found no flag on any well and every Progressive Ratio group of a
        QC Viewer recompute or a parameter sweep "never trained".
        """

        new = DFM(
            id=self.id,
            params=new_params,
            raw_df=self.raw_df.copy(),
            version=self.version,
            source_files=self.source_files,
            generation=self.generation,
            ## Shared, not copied: nothing writes to a ParadigmState or to the
            ## Baseline in force.
            paradigm=self.paradigm,
            baseline_in_force=self.baseline_in_force,
            training_info=self.training_info,
            repeated_records=self.repeated_records,
            load_notes=self.load_notes,
            recording_end_minutes=self.recording_end_minutes,
            before_range=self.before_range,
            well_names=self.well_names,
            in_training_data=(None if self.in_training_data is None
                              else self.in_training_data.copy()),
        )
        new.recompute_all(correct_for_dual_feeding=bool(new_params.correct_for_dual_feeding))
        return new

    def recompute_all(self, *, correct_for_dual_feeding: bool = False) -> None:
        self._calculate_baseline()
        self.thresholds = build_thresholds_table(self.baseline_df, self.params)
        self._calculate_lights()
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
        # Port of `CalculateProgressiveRatioTraining()`: the per-well training
        # table (:func:`_flag_training_table`), then the flag taken off.
        df = self.raw_df
        self.in_training_data = _flag_training_table(df)
        for col in [f"W{i}" for i in range(1, 13) if f"W{i}" in df.columns]:
            values = pd.to_numeric(df[col], errors="coerce")
            intraining = values.fillna(0) > _TRAINING_FLAG_ABOVE
            df.loc[intraining, col] = values[intraining].fillna(0) - _TRAINING_FLAG
        self.raw_df = df

    def _calculate_paradigm_training(self) -> None:
        """Training on generation 3, from each well's Phase: the same
        ``in_training_data`` table generation 1 builds, so
        :meth:`training_end_minutes` and :meth:`training_flag_state` read
        both alike.  The ``W`` values carry no flag and are left alone.

        A well's training end is its first row under a ratio Paradigm and
        out of Phase ``A`` (:meth:`ParadigmState.training_end`, decision 11).
        ``Minutes`` and ``Sample`` are those of the row before it, the last
        Training row, as generation 1 records the last flagged minute: the
        Progressive Ratio windows split at (0, end] / (end, inf), which puts
        the first Main row in Test.  ``EverFlagged`` is whether the well had
        any Training to report (a row in ``A``, or a row of the Program
        before its training end), and is False for a well that never runs a
        ratio Paradigm; ``Cleared`` is whether a training end exists.
        """
        state = self.paradigm
        n = len(self.raw_df)
        minutes = pd.to_numeric(self.raw_df["Minutes"], errors="coerce").to_numpy(dtype=float)
        samples = (pd.to_numeric(self.raw_df["Sample"], errors="coerce").to_numpy(dtype=float)
                   if "Sample" in self.raw_df.columns else np.arange(1, n + 1, dtype=float))
        ## Rows of the Program (not the baseline lead-in, Interval 0); an
        ## unknown instruction counts as Program, not as baseline.
        program_rows = state.program_interval.to_numpy(dtype=np.int64, na_value=-1) != 0
        period_min = 1.0 / (60.0 * float(self.params.samples_per_second))
        rows = []
        for well in range(1, 13):
            ratio = state.ratio_rows(well)
            end = state.training_end(well)
            if not ratio.any():
                rows.append({"well": f"W{well}", "Minutes": np.nan, "Sample": np.nan,
                             "EverFlagged": False, "Cleared": False})
                continue
            phase_a = state.phase_a[:, well - 1]
            if end is None:
                rows.append({"well": f"W{well}", "Minutes": np.nan, "Sample": np.nan,
                             "EverFlagged": bool(phase_a.any()), "Cleared": False})
                continue
            if end > 0:
                last_minute, last_sample = float(minutes[end - 1]), float(samples[end - 1])
            else:
                ## Training ended before the first row loaded (a range that
                ## starts after it): the last Training row is the one before.
                last_minute, last_sample = float(minutes[0]) - period_min, float(samples[0]) - 1.0
            rows.append({
                "well": f"W{well}", "Minutes": last_minute, "Sample": last_sample,
                "EverFlagged": bool(phase_a.any() or program_rows[:end].any()),
                "Cleared": True,
            })
        self.in_training_data = pd.DataFrame(rows)

    def _calculate_no_training(self) -> None:
        """Generation 2: the firmware recorded no Training at all, so every
        well is never flagged (``training_info`` says why)."""
        self.in_training_data = pd.DataFrame([
            {"well": f"W{well}", "Minutes": np.nan, "Sample": np.nan,
             "EverFlagged": False, "Cleared": False} for well in range(1, 13)])

    def training_end_minutes(self, well: int) -> float | None:
        """Minute at which *well*'s training flag cleared, or ``None`` when the
        well was never flagged or never cleared (see
        :meth:`_calculate_progressive_ratio_training`)."""
        itd = self.in_training_data
        if itd is None or itd.empty or "Cleared" not in itd.columns:
            return None
        row = itd[itd["well"] == f"W{int(well)}"]
        if row.empty or not bool(row["Cleared"].iloc[0]):
            return None
        val = float(row["Minutes"].iloc[0])
        return val if np.isfinite(val) else None

    def training_flag_state(self, well: int) -> str:
        """``"cleared"``, ``"never_cleared"`` or ``"never_flagged"`` for *well*."""
        itd = self.in_training_data
        if itd is None or itd.empty or "Cleared" not in itd.columns:
            return "never_flagged"
        row = itd[itd["well"] == f"W{int(well)}"]
        if row.empty or not bool(row["EverFlagged"].iloc[0]):
            return "never_flagged"
        return "cleared" if bool(row["Cleared"].iloc[0]) else "never_cleared"

    def _calculate_baseline(self) -> None:
        # Port of `CalculateBaseline()` with the same window logic.
        df = self.raw_df.copy()
        window_min = float(self.params.baseline_window_minutes)
        window = int(round(window_min * 60 * self.params.samples_per_second))
        if window % 2 == 0:
            window += 1

        for i in range(1, 13):
            col = f"W{i}"
            if col not in df.columns:
                continue
            df[col] = baseline_subtract(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float), window)

        self.baseline_df = df

    def _calculate_lights(self) -> None:
        raw = self.raw_df
        wcols = [f"W{i}" for i in range(1, 13)]
        # Empty-frame or missing-column fall-through: emit a zero-row/zero-light
        # frame with the expected schema so downstream code never sees NaN vs
        # boolean mismatches.
        if "OptoCol1" not in raw.columns or len(raw) == 0:
            out = pd.DataFrame(0, index=raw.index, columns=wcols)
            out.insert(0, "Minutes", raw["Minutes"].to_numpy() if "Minutes" in raw.columns else [])
            out["OptoCol1"] = np.nan
            out["OptoCol2"] = np.nan
            self.lights_df = out
            return
        opto1 = pd.to_numeric(raw["OptoCol1"], errors="coerce").fillna(0).astype(int).to_numpy()
        opto2 = (
            pd.to_numeric(raw["OptoCol2"], errors="coerce").fillna(0).astype(int).to_numpy()
            if "OptoCol2" in raw.columns
            else np.zeros(len(raw), dtype=int)
        )
        # Vectorised bit-unpack — one shot over all rows, no list-comp, no
        # empty-input vstack hazard.
        bit_masks = 1 << np.arange(12, dtype=np.int64)
        lights = (opto1[:, None] & bit_masks) > 0
        out = pd.DataFrame(lights, columns=wcols)
        out.insert(0, "Minutes", raw["Minutes"].to_numpy())
        out["OptoCol1"] = opto1
        out["OptoCol2"] = opto2
        self.lights_df = out

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
    def _resolve_well(self, chamber: int, well: str | None) -> int:
        """
        Return the 1-based well number for a given chamber index and well label.

        For one-well designs, `well` is ignored.
        For two-well designs, `well` must be "a" or "b".
        """
        ch = self.chambers[chamber - 1]
        if isinstance(ch, OneWellChamber):
            return ch.well
        # TwoWellChamber — well is required
        if well is None:
            raise ValueError("well must be specified ('a' or 'b') for two-well designs")
        w = well.lower()
        if w == "a":
            return ch.well_a
        if w == "b":
            return ch.well_b
        raise ValueError(f"well must be 'a' or 'b', got {well!r}")

    def licks_for(self, chamber: int, well: str | None = None) -> pd.Series:
        """Lick boolean series for the given chamber (and well 'a'/'b' for two-well designs)."""
        col = f"W{self._resolve_well(chamber, well)}"
        return self.lick_df[col]

    def events_for(self, chamber: int, well: str | None = None) -> pd.Series:
        """Event integer series for the given chamber (and well 'a'/'b' for two-well designs)."""
        col = f"W{self._resolve_well(chamber, well)}"
        return self.event_df[col]

    def baseline_for(self, chamber: int, well: str | None = None) -> pd.Series:
        """Baselined signal series for the given chamber (and well 'a'/'b' for two-well designs)."""
        col = f"W{self._resolve_well(chamber, well)}"
        return self.baseline_df[col]

    def durations_for(self, chamber: int, well: str | None = None) -> pd.DataFrame:
        """Bout duration DataFrame for the given chamber (and well 'a'/'b' for two-well designs)."""
        col = f"W{self._resolve_well(chamber, well)}"
        return self.durations[col]

    def intervals_for(self, chamber: int, well: str | None = None) -> pd.DataFrame:
        """Inter-bout interval DataFrame for the given chamber (and well 'a'/'b' for two-well designs)."""
        col = f"W{self._resolve_well(chamber, well)}"
        return self.intervals[col]

    def lights_for(self, chamber: int, well: str | None = None) -> pd.Series:
        """Boolean lights-on series for the given chamber (and well 'a'/'b' for two-well designs)."""
        col = f"W{self._resolve_well(chamber, well)}"
        return self.lights_df[col]

    def raw(self, *, range_minutes: Sequence[float] = (0, 0)) -> pd.DataFrame:
        if not range_is_specified(range_minutes):
            return self.raw_df
        a, b = range_bounds(range_minutes)
        return self.raw_df[(self.raw_df["Minutes"] > a) & (self.raw_df["Minutes"] <= b)]

    def baselined(self, *, range_minutes: Sequence[float] = (0, 0)) -> pd.DataFrame:
        if not range_is_specified(range_minutes):
            return self.baseline_df
        a, b = range_bounds(range_minutes)
        df = self.baseline_df
        return df[(df["Minutes"] > a) & (df["Minutes"] <= b)]

    # ---- QC ----
    def data_breaks(self, *, multiplier: float = 4.0) -> pd.DataFrame | None:
        """Port of FindDataBreaks() from QC.R: the rows that follow a gap in
        time longer than *multiplier* sample periods.

        Each row of ``raw_df`` after a break, led by ``BreakIndex`` (its
        1-based row position) and ``GapSeconds`` (the time since the row
        before).  R named these ``Index`` and ``Interval``, which a version-3
        file already has as columns (and a 90-column one had ``Interval``
        too), so the insert raised on every recording with a break.
        """
        if "Seconds" not in self.raw_df.columns:
            return None
        seconds = pd.to_numeric(self.raw_df["Seconds"], errors="coerce").to_numpy(dtype=float)
        if seconds.size == 0:
            return None
        gap = np.diff(seconds, prepend=seconds[0])
        thresh = (1.0 / float(self.params.samples_per_second)) * float(multiplier)
        idx = np.flatnonzero(gap > thresh)
        if idx.size == 0:
            return None
        out = self.raw_df.iloc[idx].copy()
        out.insert(0, "GapSeconds", gap[idx])
        out.insert(0, "BreakIndex", idx + 1)
        return out.reset_index(drop=True)

    def lost_record_gaps(self) -> list[tuple[int, int, float]] | None:
        """``(last Index before, records missing, minute)`` for each gap in
        ``Index``, which starts at 1 and rises by 1 a record: the records
        lost between the DFM and the file.  The minute is that of the last
        row before the gap.  :meth:`ParadigmState.index_gaps` on generation
        3, a plain diff of ``Index`` otherwise; ``None`` without an ``Index``
        column (a v2 file) or with fewer than two values."""
        if self.paradigm is not None:
            return self.paradigm.index_gaps()
        raw = self.raw_df
        if "Index" not in raw.columns:
            return None
        index = pd.to_numeric(raw["Index"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(index)
        if valid.sum() < 2:
            return None
        index = index[valid]
        minutes = pd.to_numeric(raw["Minutes"], errors="coerce").to_numpy(dtype=float)[valid]
        step = np.diff(index)
        return [(int(index[i]), int(step[i]) - 1, float(minutes[i]))
                for i in np.flatnonzero(step > 1)]

    def paradigm_sequence(self) -> list[dict[str, object]] | None:
        """The Paradigms the rows were recorded under, in order: one entry
        per run of rows with the same name (``Paradigm``, ``FirstMin``,
        ``LastMin``, ``Rows``), an unknown name as ``NA``: the lead-in, which
        PyMCU writes before any instruction, and a restart's idle rows.
        ``None`` without a Paradigm state (generations 1 and 2)."""
        state = self.paradigm
        if state is None:
            return None
        n = len(state)
        if not n:
            return []
        codes, _ = pd.factorize(state.paradigm, use_na_sentinel=True)
        change = (np.flatnonzero(codes[1:] != codes[:-1]) + 1).tolist()
        names = state.paradigm.to_numpy(dtype=object, na_value=None)
        return [{"Paradigm": "NA" if names[a] is None else str(names[a]),
                 "FirstMin": float(state.minutes[a]), "LastMin": float(state.minutes[b - 1]),
                 "Rows": int(b - a)}
                for a, b in zip([0, *change], [*change, n])]

    def integrity_report(self) -> dict[str, object]:
        """Port of ReportDFMIntegrity() from QC.R, by Generation.

        Prints the report and returns its numbers.  Beyond R's: the Error
        byte's bits under the labels of the file's Generation
        (:data:`ERROR_BITS`); records lost (gaps in ``Index``) and records
        repeated (dropped at load, :attr:`load_notes`); on generation 3 the
        rows under an unknown instruction after the lead-in, the late
        records (some ``OnMs`` above 200) and the Paradigm sequence; and the
        ``Error`` rows of the recording's ``Messages.csv`` for this DFM and
        the MCU.  A count that does not apply to the Generation is ``None``.

        Generation 3 also gets the DFM's restarts (``## DFM restarts``,
        :meth:`restarts`, the dict's ``restarts``), :data:`NO_ACK_NOTE` when
        *No Ack* is counted with nothing lost or repeated, and the
        ``Messages.csv`` "Lost N records" counts checked against the
        ``Index`` gaps outside restarts.  Generations 1 and 2 are unchanged.
        """
        raw = self.raw_df
        n = len(raw)
        print("\n==============================")
        print("DFM Integrity Report")
        print(f"DFM ID: {self.id}")
        print(f"Rows in RawData: {n}")
        print(f"Generation: {self.generation}")
        print("==============================\n")

        start_time = end_time = elapsed_minutes = elapsed_minutes_from_minutes_col = None

        ts = _clock_times(raw)
        if ts is not None and ts.notna().any():
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

        gaps = self.lost_record_gaps()
        lost = None if gaps is None else int(sum(missing for _, missing, _ in gaps))
        ## Firmware 4.0.0's own No Ack: counted after every read, with nothing lost or resent.
        explain_no_ack = (self.generation == GENERATION_PARADIGM and lost == 0
                          and int(self.repeated_records) == 0)

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
                        values = codes.to_numpy()
                        print("  Error-type counts (among flagged rows):")
                        no_ack = 0
                        for mask, name in ERROR_BITS.get(self.generation, ERROR_BITS[GENERATION_PARADIGM]):
                            count = int(((values & mask) != 0).sum())
                            print(f"    {name}: {count}")
                            no_ack += count if name == "No Ack" else 0
                        if explain_no_ack and no_ack > 0:
                            print(f"  {NO_ACK_NOTE}")
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
                print("Index increments by exactly 1 for all consecutive rows." if index_ok else "Index does NOT always increment by 1.")
                self._print_lost_and_repeated(gaps or [])

        state = self.paradigm
        unknown = late = sequence = restarts = None
        if state is not None:
            unknown = int(state.unknown_rows().sum())
            late = int(state.late_rows().sum())
            sequence = self.paradigm_sequence()
            print("## Paradigm state")
            print(f"Lead-in rows (Interval 0, before the first instruction): {state.lead_in_rows()}")
            print(f"Rows under an unknown instruction (after the lead-in): {unknown}")
            print(f"Late records (some OnMs above {RECORD_MS} ms): {late}")
            print("Paradigm sequence:")
            width = max((len(s["Paradigm"]) for s in sequence), default=0)
            for s in sequence:
                print(f"  {s['Paradigm']:<{width}}  minutes {s['FirstMin']:.2f}-{s['LastMin']:.2f}"
                      f"  {s['Rows']} rows")
            print()
        elif self.generation == GENERATION_EMPTY:
            print("## Paradigm state")
            print("None: every Paradigm column is NA (DFM firmware older than 4.0.0).\n")
        found = self._messages()
        if state is not None:
            restarts = self._restarts(found)
            self._print_restarts(restarts)
        self._print_message_errors(found, restarts=restarts)

        return {
            "dfm_id": self.id,
            "n_rawdata": n,
            "generation": self.generation,
            "start_time": start_time,
            "end_time": end_time,
            "elapsed_minutes": elapsed_minutes,
            "elapsed_minutes_from_minutes_col": elapsed_minutes_from_minutes_col,
            "error_columns": err_cols,
            "index_increments_by_one": index_ok,
            "lost_records": lost,
            "lost_record_gaps": gaps,
            "repeated_records": int(self.repeated_records),
            "unknown_instruction_rows": unknown,
            "late_records": late,
            "paradigm_sequence": sequence,
            "restarts": restarts,
        }

    def _print_lost_and_repeated(self, gaps: list[tuple[int, int, float]]) -> None:
        """The integrity text's lost and repeated records, then a blank line."""
        lost = sum(missing for _, missing, _ in gaps)
        print(f"Lost records: {lost}" + (f" in {len(gaps)} gap(s)" if gaps else ""))
        period = 1.0 / float(self.params.samples_per_second)
        for after, missing, minute in gaps:
            print(f"  after Index {after} (minute {minute:.2f}): {missing} record(s), "
                  f"{missing * period:.1f} s")
        print(f"Repeated records: {int(self.repeated_records)}")
        for note in self.load_notes:
            print(f"  {note}")
        print()

    def _messages(self) -> pd.DataFrame | None:
        """The recording's ``Messages.csv`` rows for this DFM and the MCU
        (DFM 0), or ``None`` when the recording has no such file."""
        from . import messages

        if not self.source_files:
            return None
        data_dir = Path(self.source_files[0]).parent
        if messages.find_messages(data_dir) is None:
            return None
        return messages.for_dfm(messages.read(data_dir), self.id)

    def restarts(self) -> list[dict[str, object]] | None:
        """Each restart of this DFM (PyMCU 529b926), from the rows and from
        ``Messages.csv``, in ``Index`` order; ``None`` without a Paradigm
        state (generations 1 and 2).  See :meth:`_restarts` for the keys."""
        return self._restarts(self._messages())

    def _restarts(self, found: pd.DataFrame | None) -> list[dict[str, object]] | None:
        """The restarts the rows show (:meth:`ParadigmState.restarts`: an
        ``Index`` gap into rows under no instruction), each matched with
        ``Messages.csv``'s "recorded nothing … Index a to b are missing" by
        its ``Index`` range, then the ones only *found* (the file's rows,
        :meth:`_messages`) names.  Each is a dict:

        * ``Time``: the clock time of the first row after the gap, or the
          message's own when the rows show none;
        * ``Minute``, ``Row`` and ``ResumeRow`` (the first row after the
          rows under no instruction), ``None`` when the rows show none;
        * ``AfterIndex``, the last ``Index`` before the gap, ``Missing`` and
          ``Seconds``, what it lost;
        * ``Source``: ``"data"``, or ``"Messages.csv"`` when only the file
          names it;
        * ``Messages``: whether ``Messages.csv`` names it, ``None`` without
          the file;
        * ``Note``: the integrity text's words for that.
        """
        from . import messages

        state = self.paradigm
        if state is None:
            return None
        n = len(state)
        period = 1.0 / float(self.params.samples_per_second)
        times = _clock_times(self.raw_df)

        def clock(row: int):
            if times is None or not 0 <= row < len(times) or pd.isna(times.iloc[row]):
                return None
            return times.iloc[row].tz_localize(None)

        ## PyMCU's messages, a "restarted" with the "recorded nothing" that follows it.
        named: list[tuple[object | None, object | None]] = []
        if found is not None:
            records = messages.record_messages(found)
            mine = (records["DFM"] == int(self.id)).fillna(False).to_numpy(dtype=bool)
            started = None
            for row in records[mine].itertuples(index=False):
                if row.Kind == messages.DFM_RESTARTED:
                    if started is not None:
                        named.append((started, None))
                    started = row
                elif row.Kind == messages.RESTART_GAP:
                    named.append((started, row))
                    started = None
            if started is not None:
                named.append((started, None))

        def entry(**values) -> dict[str, object]:
            out: dict[str, object] = dict(Time=None, Minute=None, Row=None, ResumeRow=None,
                                          AfterIndex=None, Missing=None, Seconds=None,
                                          Source="data", Messages=None, Note="")
            out.update(values)
            if out["Missing"] is not None:
                out["Seconds"] = float(out["Missing"]) * period
            return out

        used: set[int] = set()
        out: list[dict[str, object]] = []
        for span in state.restarts():
            after = int(state.index[span.row - 1])
            match = next((k for k, (_start, gap) in enumerate(named)
                          if k not in used and gap is not None and int(gap.First) == after + 1), None)
            if match is not None:
                used.add(match)
            confirmed = None if found is None else match is not None
            if confirmed is None:
                note = f"no {messages.MESSAGES_FILENAME} to confirm it"
            else:
                note = f"{'named' if confirmed else 'not named'} in {messages.MESSAGES_FILENAME}"
            out.append(entry(Time=clock(span.row), Minute=float(span.minute), Row=int(span.row),
                             ResumeRow=int(span.resume_row), AfterIndex=after, Missing=int(span.missing),
                             Messages=confirmed, Note=note))
        for k, (start, gap) in enumerate(named):
            if k in used:
                continue
            source = dict(Source=messages.MESSAGES_FILENAME, Messages=True)
            if gap is None:
                out.append(entry(Time=start.Time, **source,
                                 Note=f"named in {messages.MESSAGES_FILENAME}; it lost no records, "
                                      f"so the data show no Index gap"))
                continue
            after = int(gap.First) - 1
            before = np.flatnonzero(state.index == after)
            row = int(before[0]) + 1 if before.size else n
            if row < n and int(state.index[row]) > after + 1:
                out.append(entry(Time=clock(row), Minute=float(state.minutes[row]), AfterIndex=after,
                                 Missing=int(state.index[row]) - after - 1, **source,
                                 Note=f"named in {messages.MESSAGES_FILENAME}; the data show an "
                                      f"Index gap only"))
            else:
                out.append(entry(Time=gap.Time, AfterIndex=after, Missing=int(gap.Records), **source,
                                 Note=f"named in {messages.MESSAGES_FILENAME}; the rows loaded show "
                                      f"no Index gap there"))
        out.sort(key=lambda r: (r["AfterIndex"] is None, r["AfterIndex"] or 0))
        return out

    def _print_restarts(self, restarts: list[dict[str, object]]) -> None:
        """The integrity text's ``## DFM restarts``, then a blank line."""
        print("## DFM restarts")
        if not restarts:
            print("None.\n")
            return
        print(f"Restarts: {len(restarts)}")
        for r in restarts:
            when = r["Time"]
            text = when.strftime("%m/%d/%Y %H:%M:%S.%f")[:-3] if when is not None else "(no time)"
            if r["Minute"] is not None:
                text += f" (minute {r['Minute']:.2f})"
            if r["Missing"] is not None:
                text += (f": {r['Missing']} record(s), {r['Seconds']:.1f} s, missing after Index "
                         f"{r['AfterIndex']}")
            if r["ResumeRow"] is not None:
                text += f", then {r['ResumeRow'] - r['Row']} row(s) under no instruction"
            print(f"  {text}; {r['Note']}")
        print()

    def _lost_records_check(self, found: pd.DataFrame,
                            restarts: list[dict[str, object]]) -> str | None:
        """``Messages.csv``'s "Lost N records" against the ``Index`` gaps
        outside restarts (a restart's records are named apart, by the
        "recorded nothing" error), over the ``Index`` range loaded; ``None``
        without an ``Index``."""
        from . import messages

        gaps = self.lost_record_gaps()
        if gaps is None:
            return None
        records = messages.record_messages(found)
        mine = ((records["DFM"] == int(self.id)) & (records["Kind"] == messages.LOST_RECORDS)) \
            .fillna(False).to_numpy(dtype=bool)
        records = records[mine]
        index = pd.to_numeric(self.raw_df["Index"], errors="coerce").dropna()
        lo, hi = float(index.min()), float(index.max())
        inside = (records["First"].isna() | ((records["First"] > lo) & (records["Last"] < hi))) \
            .fillna(False).to_numpy(dtype=bool)
        named = int(records.loc[inside, "Records"].fillna(0).sum())
        restart_gaps = {r["AfterIndex"] for r in restarts if r["AfterIndex"] is not None}
        in_gaps = int(sum(missing for after, missing, _ in gaps if after not in restart_gaps))
        text = f"Lost records {messages.MESSAGES_FILENAME} names: {named}"
        if named == in_gaps:
            return f"{text}, as the Index gaps outside restarts show."
        more = "more" if in_gaps > named else "fewer"
        return (f"{text}, but the Index gaps outside restarts hold {in_gaps} "
                f"({abs(in_gaps - named)} {more} in the data).")

    def _print_message_errors(self, found: pd.DataFrame | None, *,
                              restarts: list[dict[str, object]] | None = None) -> None:
        """The ``Error`` rows of the recording's ``Messages.csv`` (*found*,
        :meth:`_messages`) for this DFM and the MCU (DFM 0), and on
        generation 3 :meth:`_lost_records_check`; nothing when the recording
        has no such file."""
        from . import messages

        if found is None:
            return
        errors = messages.errors(found)
        print(f"## {messages.MESSAGES_FILENAME} errors (DFM {self.id} and the MCU)")
        if errors.empty:
            print("None.")
        for _, row in errors.iterrows():
            print(f"  {messages.describe(row)}")
        if self.generation == GENERATION_PARADIGM:
            check = self._lost_records_check(found, restarts or [])
            if check is not None:
                print(check)
        print()

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
    def _chamber_from_well(self, well: int) -> int:
        """The Chamber *well* belongs to, read from :attr:`chambers` — so a
        ``chamber_sets`` other than ``[[1, 2], [3, 4], ...]`` labels it right,
        where well parity would not."""
        if self.chambers is None:
            self._build_chambers()
        for ch in self.chambers:
            if int(well) in (int(w) for w in ch.wells):
                return int(ch.index)
        return int(well) if self.params.chamber_size == 1 else (int(well) - 1) // 2 + 1

    def _tcwell_from_well(self, well: int) -> str | None:
        """``WellA`` / ``WellB`` for *well* in a two-well Chamber (A/B follow
        ``pi_direction``), from :attr:`chambers`; ``None`` single-well."""
        if self.params.chamber_size != 2:
            return None
        if self.chambers is None:
            self._build_chambers()
        for ch in self.chambers:
            if int(well) == int(ch.well_a):
                return "WellA"
            if int(well) == int(ch.well_b):
                return "WellB"
        return None

    @staticmethod
    def _range_rows(minutes, range_minutes: Sequence[float]) -> np.ndarray | None:
        """``(n,)`` bool: the rows (by their ``Minutes``) inside the ``(a, b]``
        window, or inside any window of a list of them (a Program Facet
        pooling its Cycles); ``None`` for the whole recording.  Every
        feeding-summary statistic is then taken over the events inside the
        union, so a median is never an average of per-window medians."""
        from . import windowing

        return windowing.range_mask(minutes, range_minutes)

    @staticmethod
    def _interval_summary(interval_df, range_minutes: Sequence[float]) -> tuple[float, float]:
        """``(MeanTimeBtw, MedTimeBtw)`` of the intervals that start inside
        the window.  NaN, as for the bout durations, when none does or the
        well has none at all: 0 s would read as feeding without pause, and
        it entered treatment means and tests as if measured.  The intervals
        themselves keep R's ``Get.Intervals`` rule, lead-in and tail gaps
        included."""
        from . import windowing

        if not isinstance(interval_df, pd.DataFrame) or interval_df.empty:
            return np.nan, np.nan
        if windowing.is_window_list(range_minutes):
            tmp = interval_df[DFM._range_rows(interval_df["Minutes"], range_minutes)]
        elif range_is_specified(range_minutes):
            a, b = range_bounds(range_minutes)
            tmp = interval_df[(interval_df["Minutes"] > a) & (interval_df["Minutes"] <= b)]
        else:
            tmp = interval_df
        if tmp.empty:
            return np.nan, np.nan
        return float(tmp["IntervalSec"].mean()), float(tmp["IntervalSec"].median())

    @staticmethod
    def _duration_summary(dur_df, range_minutes: Sequence[float]) -> tuple[float, float]:
        from . import windowing

        if not isinstance(dur_df, pd.DataFrame) or dur_df.empty:
            return np.nan, np.nan
        if windowing.is_window_list(range_minutes):
            tmp = dur_df[DFM._range_rows(dur_df["Minutes"], range_minutes)]
        elif range_is_specified(range_minutes):
            a, b = range_bounds(range_minutes)
            tmp = dur_df[(dur_df["Minutes"] > a) & (dur_df["Minutes"] <= b)]
        else:
            tmp = dur_df
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

    @property
    def light_source(self) -> str:
        """Where the light seconds of the feeding summary come from:
        ``OnMs`` on generation 3 (exact, decision 31), else ``OptoCol1``
        sampled once a record."""
        if self.generation == GENERATION_PARADIGM and self.paradigm is not None:
            return "OnMs, exact"
        return f"OptoCol1, sampled at {float(self.params.samples_per_second):g} Hz"

    def light_seconds(self, range_minutes: Sequence[float] = (0, 0)) -> np.ndarray:
        """Seconds each well's LED was on over the (a, b] window, wells 1-12.

        Generation 3 sums ``OnMs`` (:meth:`ParadigmState.lit_seconds`), the
        milliseconds the LED was driven, light from Linkage included; the
        other generations count the ``OptoCol1`` snapshots lit, one sample
        period each, which misses on-periods shorter than a record.  A list
        of windows sums over the rows inside any of them.
        """
        from . import windowing

        if self.generation == GENERATION_PARADIGM and self.paradigm is not None \
                and windowing.is_window_list(range_minutes):
            rows = self._range_rows(self.paradigm.minutes, range_minutes)
            return np.array([float(self.paradigm.on_ms[rows, w - 1].sum()) / 1000.0
                             for w in range(1, 13)], dtype=float)
        if self.generation == GENERATION_PARADIGM and self.paradigm is not None:
            i0, i1 = self._state_rows(range_minutes)
            return np.array([self.paradigm.lit_seconds(w, i0, i1) for w in range(1, 13)], dtype=float)
        lights = self._apply_range(self.lights_df, range_minutes)
        if "W1" not in lights.columns:
            return np.zeros(12, dtype=float)
        return lights[[f"W{i}" for i in range(1, 13)]].sum(axis=0).to_numpy() / float(self.params.samples_per_second)

    def _state_rows(self, range_minutes: Sequence[float]) -> tuple[int, int]:
        """``[i0, i1)`` of the Paradigm state's rows in the (a, b] window
        (``Minutes`` rises with the row, so the window is one block)."""
        n = len(self.paradigm)
        if not range_is_specified(range_minutes):
            return 0, n
        a, b = range_bounds(range_minutes)
        minutes = self.paradigm.minutes
        inside = np.flatnonzero((minutes > a) & (minutes <= b))
        if inside.size == 0:
            return 0, 0
        return int(inside[0]), int(inside[-1]) + 1

    def feeding_summary(
        self,
        *,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool = True,
    ) -> pd.DataFrame:
        """Per chamber over ``(a, b]``, or over the union of a list of such
        windows (a Program Facet; see :func:`windowing.is_window_list`)."""
        from . import windowing

        span = windowing.range_span(range_minutes)
        if self.params.chamber_size == 1:
            lights_sec = self.light_seconds(range_minutes)
            rows: list[dict] = []
            for well in range(1, 13):
                cname = f"W{well}"
                mean_dur, med_dur = self._duration_summary(self.durations[cname], range_minutes)
                mean_intv, med_intv = self._interval_summary(self.intervals[cname], range_minutes)
                base = self.baseline_df[cname].to_numpy(dtype=float)
                ev = self.event_df[cname].to_numpy(dtype=int)
                mask = self._range_rows(self.baseline_df["Minutes"], range_minutes)
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
                    "StartMin": span[0], "EndMin": span[1],
                })
            out = pd.DataFrame(rows)
            if transform_licks:
                out["Licks"] = np.power(out["Licks"], 0.25)
            return out

        if self.params.chamber_size == 2:
            lights_sec = self.light_seconds(range_minutes)
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
                mask = self._range_rows(self.baseline_df["Minutes"], range_minutes)
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
                    "StartMin": span[0], "EndMin": span[1],
                })
            out = pd.DataFrame(rows)
            if transform_licks:
                out["LicksA"] = np.power(out["LicksA"], 0.25)
                out["LicksB"] = np.power(out["LicksB"], 0.25)
            return out

        raise NotImplementedError("Feeding summary not implemented for this DFM type.")

    def _sweep_bounds(self, range_minutes: Sequence[float]) -> tuple[float, float]:
        """``(m_min, m_max)`` a binned or moving-window sweep covers: the
        range given, an open end at the last row; for the whole rows, from
        the start of the range they were loaded with (a Slice's start, not
        minute 0, whose bins before it would hold no row) to the last row."""
        last = float(self.raw_df["Minutes"].max())
        if range_is_specified(range_minutes):
            m_min, m_max = range_bounds(range_minutes)
            return m_min, (last if m_max == float("inf") else m_max)
        ## getattr: a DFM built by hand (tests, ``object.__new__``) has no slot set.
        before = getattr(self, "before_range", None)
        return float(before.start_min if before is not None else 0.0), last

    def _sample_minutes(self) -> float:
        """One sample period, in minutes."""
        return 1.0 / (60.0 * float(self.params.samples_per_second))

    def _window_starts(self, m_min: float, m_max: float, window_min: float,
                       step_min: float) -> np.ndarray:
        """Starts of the moving windows ``(a, a + window_min]`` that fit inside
        ``(m_min, m_max]``: a window running past the last row would be
        labelled with a minute the recording never reached and hold part of
        its width.  The last row's minute may fall a sample short of a whole
        minute, so a window may end up to one sample period past it.  A sweep
        narrower than one window is that one window."""
        last_start = m_max - float(window_min) + self._sample_minutes() + 1e-9
        starts = np.arange(m_min, last_start, step_min, dtype=float)
        if starts.size == 0:
            starts = np.array([m_min], dtype=float)
        return starts

    def binned_feeding_summary(
        self,
        *,
        binsize_min: float = 30.0,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool = True,
    ) -> pd.DataFrame:
        m_min, m_max = self._sweep_bounds(range_minutes)
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
        ## A range that starts on the last row's minute has no bin to hold.
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def moving_window_feeding_summary(
        self,
        *,
        window_min: float = 60.0,
        step_min: float = 30.0,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool = True,
    ) -> pd.DataFrame:
        """
        Full feeding-summary metrics computed over a sliding window.

        Identical in column layout to :meth:`binned_feeding_summary`, but the
        timeline is swept with a window *window_min* minutes wide advanced in
        *step_min* increments, so successive windows may overlap.  Each window
        spans ``(a, a + window_min]`` and is labelled by its right edge
        (``a + window_min``) in the ``Minutes`` column — matching the
        convention used by :meth:`moving_median_duration`.  Only windows that
        end by the last row are swept (:meth:`_window_starts`).

        Because every metric is produced by the same per-window
        :meth:`feeding_summary` call, downstream code can extract any feeding
        metric (Licks, PI, MedDuration, intervals, …) for a moving-window plot
        exactly as it does for binned plots.

        Parameters
        ----------
        window_min:
            Width of the moving window in minutes (default 60.0).
        step_min:
            Distance the window advances each step, in minutes (default 30.0).
        range_minutes:
            ``(start, end)`` window the sweep is bounded to.  ``(0, 0)`` (the
            default) sweeps the rows loaded: from 0, or a Slice's start, to
            the end of this DFM's recording.
        transform_licks:
            Apply the fourth-root lick transform (see :meth:`feeding_summary`).
        """
        if window_min <= 0:
            raise ValueError("window_min must be positive.")
        if step_min <= 0:
            raise ValueError("step_min must be positive.")
        m_min, m_max = self._sweep_bounds(range_minutes)
        if m_min >= m_max:
            raise ValueError(f"range start ({m_min}) must be less than end ({m_max}).")

        starts = self._window_starts(m_min, m_max, window_min, step_min)

        parts: list[pd.DataFrame] = []
        for a in starts:
            b = a + float(window_min)
            summ = self.feeding_summary(
                range_minutes=(float(a), float(b)), transform_licks=transform_licks
            )
            summ.insert(0, "Minutes", float(b))  # label by the window's right edge
            summ.insert(0, "Interval", f"({a:g},{b:g}]")
            parts.append(summ)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

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
            tmp.insert(0, "TCWell", self._tcwell_from_well(well) or "")
            tmp.insert(0, "Chamber", self._chamber_from_well(well))
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
            tmp.insert(0, "TCWell", self._tcwell_from_well(well) or "")
            tmp.insert(0, "Chamber", self._chamber_from_well(well))
            tmp.insert(0, "DFM", self.id)
            frames.append(tmp)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def moving_median_duration(
        self,
        *,
        window_min: float = 60.0,
        step_min: float = 30.0,
        range_minutes: Sequence[float] = (0, 0),
    ) -> pd.DataFrame:
        """
        Median feeding-bout duration (seconds) over a sliding window.

        The timeline is swept with a window *window_min* minutes wide, advanced
        in *step_min* increments.  Each window spans ``(a, a + window_min]`` and
        is labelled by its endpoint (``a + window_min``) in the ``Minutes``
        column.  ``MedDuration`` columns are ``NaN`` (and the matching
        ``Events`` column is 0) for windows containing no feeding events.

        The column layout mirrors :meth:`feeding_summary` for the DFM's chamber
        type, with one row per (chamber, window):

        * **single-well (12-well)** — ``Chamber`` is the 1-based well::

              DFM, Chamber, Minutes, MedDuration, Events

        * **two-well (6-well)** — well A and well B are reported separately
          (A/B follow ``pi_direction``), just like ``MedDurationA`` /
          ``MedDurationB`` in the feeding summary::

              DFM, Chamber, Minutes, MedDurationA, MedDurationB, EventsA, EventsB

        Parameters
        ----------
        window_min:
            Width of the moving window in minutes (default 60.0).
        step_min:
            Distance the window advances each step, in minutes (default 30.0).
        range_minutes:
            ``(start, end)`` window the sweep is bounded to.  ``(0, 0)`` (the
            default) sweeps the rows loaded: from 0, or a Slice's start, to
            the end of this DFM's recording.  Only windows that end by the
            last row are swept (:meth:`_window_starts`).
        """
        if window_min <= 0:
            raise ValueError("window_min must be positive.")
        if step_min <= 0:
            raise ValueError("step_min must be positive.")
        m_min, m_max = self._sweep_bounds(range_minutes)
        if m_min >= m_max:
            raise ValueError(f"range start ({m_min}) must be less than end ({m_max}).")

        starts = self._window_starts(m_min, m_max, window_min, step_min)
        ends = starts + float(window_min)
        # Each window is labelled by its endpoint (right edge).
        labels = ends

        def _well_series(well: int) -> tuple[np.ndarray, np.ndarray]:
            """Per-window (median duration, event count) for one physical well."""
            dur = self.durations.get(f"W{well}", 0)
            med = np.full(starts.shape, np.nan, dtype=float)
            cnt = np.zeros(starts.shape, dtype=int)
            if not isinstance(dur, pd.DataFrame) or dur.empty:
                return med, cnt
            mins = dur["Minutes"].to_numpy(dtype=float)
            secs = dur["Duration"].to_numpy(dtype=float)
            order = np.argsort(mins, kind="stable")
            mins, secs = mins[order], secs[order]
            # Events with Minutes in (a, b]: count(<= b) excluding count(<= a).
            lo = np.searchsorted(mins, starts, side="right")
            hi = np.searchsorted(mins, ends, side="right")
            for i in range(starts.size):
                if hi[i] > lo[i]:
                    seg = secs[lo[i]:hi[i]]
                    med[i] = float(np.median(seg))
                    cnt[i] = int(seg.size)
            return med, cnt

        if self.params.chamber_size == 1:
            frames: list[pd.DataFrame] = []
            for well in range(1, 13):
                med, cnt = _well_series(well)
                frames.append(pd.DataFrame({
                    "DFM": self.id,
                    "Chamber": well,
                    "Minutes": labels,
                    "MedDuration": med,
                    "Events": cnt,
                }))
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        if self.params.chamber_size == 2:
            frames = []
            for chamber in range(self.params.chamber_sets.shape[0]):
                w1 = int(self.params.chamber_sets[chamber, 0])
                w2 = int(self.params.chamber_sets[chamber, 1])
                if self.params.pi_direction == "left":
                    well_a, well_b = w1, w2
                elif self.params.pi_direction == "right":
                    well_a, well_b = w2, w1
                else:
                    raise ValueError(f"Invalid pi_direction: {self.params.pi_direction!r}")
                med_a, cnt_a = _well_series(well_a)
                med_b, cnt_b = _well_series(well_b)
                frames.append(pd.DataFrame({
                    "DFM": self.id,
                    "Chamber": chamber + 1,
                    "Minutes": labels,
                    "MedDurationA": med_a,
                    "MedDurationB": med_b,
                    "EventsA": cnt_a,
                    "EventsB": cnt_b,
                }))
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        raise NotImplementedError("Moving median duration not implemented for this DFM type.")

    # ---- Plots ----
    def _well_label(self, side: str) -> str:
        """Return display name for well 'A' or 'B', falling back to 'WellA'/'WellB'."""
        if self.well_names and side in self.well_names:
            return self.well_names[side]
        return f"Well{side}"

    def _apply_range(self, df: pd.DataFrame, range_minutes: Sequence[float]) -> pd.DataFrame:
        from . import windowing

        if windowing.is_window_list(range_minutes):
            return df[self._range_rows(df["Minutes"], range_minutes)]
        if not range_is_specified(range_minutes):
            return df
        a, b = range_bounds(range_minutes)
        return df[(df["Minutes"] > a) & (df["Minutes"] <= b)]

    def plot_raw(self, *, range_minutes: Sequence[float] = (0, 0)):

        df = self._apply_range(self.raw_df, range_minutes)
        wcols = [f"W{i}" for i in range(1, 13) if f"W{i}" in df.columns]
        n = len(wcols)
        fig, axes = _subplots(n, 1, sharex=True, figsize=(10, max(6, n * 1.3)))
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

        df = self._apply_range(self.baseline_df, range_minutes)
        wcols = [f"W{i}" for i in range(1, 13) if f"W{i}" in df.columns]
        n = len(wcols)
        fig, axes = _subplots(n, 1, sharex=True, figsize=(10, max(6, n * 1.3)))
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

        col = f"W{well}"
        df = self._apply_range(self.raw_df, range_minutes)
        if col not in df.columns:
            raise ValueError(f"Well {well} not found in DFM {self.id} raw data (expected column '{col}').")
        fig, ax = _subplots(figsize=(10, 3))
        ax.plot(df["Minutes"], df[col], linewidth=0.7, color="steelblue")
        ax.set_xlabel("Minutes")
        ax.set_ylabel("Signal")
        ax.set_title(f"DFM {self.id}  W{well}  Raw")
        fig.tight_layout()
        return fig

    def plot_baselined_well(self, well: int, *, range_minutes: Sequence[float] = (0, 0), include_thresholds: bool = False):
        """Plot the baseline-subtracted signal for a single well."""

        col = f"W{well}"
        df = self._apply_range(self.baseline_df, range_minutes)
        if col not in df.columns:
            raise ValueError(f"Well {well} not found in DFM {self.id} baselined data (expected column '{col}').")
        fig, ax = _subplots(figsize=(10, 3))
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

        binned = self.binned_feeding_summary(binsize_min=binsize_min, range_minutes=range_minutes, transform_licks=transform_licks)
        if binned.empty:
            fig, ax = _subplots(figsize=(8, 3))
            ax.set_title(f"DFM {self.id} (no data)")
            return fig
        fig, ax = _subplots(figsize=(10, 4))
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

        data = self.cumulative_pi_data(range_minutes=range_minutes)
        if data.empty:
            fig, ax = _subplots(figsize=(8, 3))
            ax.set_title(f"DFM {self.id} cumulative PI (no data)")
            return fig
        chambers = sorted(data["Chamber"].unique().tolist())
        if single_plot:
            fig, ax = _subplots(figsize=(10, 4))
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
        fig, axes = _subplots(len(chambers), 1, sharex=True, figsize=(10, max(4, len(chambers) * 1.6)))
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

        data = self.cumulative_event_pi_data(events_limit=events_limit, range_minutes=range_minutes)
        if data.empty:
            fig, ax = _subplots(figsize=(8, 3))
            ax.set_title(f"DFM {self.id} cumulative event PI (no data)")
            return fig
        chambers = sorted(data["Chamber"].unique().tolist())
        xcol = "EventNum" if by_bout else "Minutes"
        xlabel = "Event Number" if by_bout else "Minutes"
        if single_plot:
            fig, ax = _subplots(figsize=(10, 4))
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
        fig, axes = _subplots(len(chambers), 1, sharex=True, figsize=(10, max(4, len(chambers) * 1.6)))
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
            fig, ax = _subplots(figsize=(10, 5))
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
            fig, axes = _subplots(6, 2, sharex=True, figsize=(12, 10))
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

        fig, axes = _subplots(len(self.chambers), 2, sharex=True, figsize=(12, max(8, len(self.chambers) * 1.4)))
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
                ax.set_title(f"Ch{int(ch.index)} {self._well_label('A') if col_idx == 0 else self._well_label('B')} (W{well})", fontsize=9)
        for ax in axes[-1]:
            ax.set_xlabel("Minutes")
        fig.suptitle(f"DFM {self.id} cumulative licks")
        fig.tight_layout()
        return fig

    def plot_cumulative_licks_chamber(
        self,
        chamber: int,
        *,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool = True,
        single_plot: bool = True,
        treatment: str | None = None,
    ):
        """
        Plot cumulative licks for a single chamber.

        One-well chambers: one line.
        Two-well chambers: both wells on one axes when single_plot=True (default),
        or stacked subplots when single_plot=False.

        Every sample where the light for that well is on is marked with a
        gold diamond symbol overlaid on the cumulative licks line.
        If *treatment* is provided it is included in the plot title.
        """

        ch = self.chambers[chamber - 1]
        baseline_r = self._apply_range(self.baseline_df, range_minutes)
        lights_r = self._apply_range(self.lights_df, range_minutes)
        first_min = float(baseline_r["Minutes"].iloc[0]) if len(baseline_r) else 0.0
        last_min = float(baseline_r["Minutes"].iloc[-1]) if len(baseline_r) else 0.0

        light_minutes = lights_r["Minutes"].to_numpy(dtype=float)

        trt_str = f"  [{treatment}]" if treatment else ""
        title = f"DFM {self.id}  Chamber {chamber}{trt_str}  Cumulative Licks"

        def cumulative_for_well(well: int) -> tuple[np.ndarray, np.ndarray]:
            d = self.durations.get(f"W{well}", 0)
            if not isinstance(d, pd.DataFrame) or d.empty:
                return np.array([first_min, last_min], dtype=float), np.array([0.0, 0.0], dtype=float)
            d_r = self._apply_range(d, range_minutes) if range_is_specified(range_minutes) else d
            if d_r.empty:
                return np.array([first_min, last_min], dtype=float), np.array([0.0, 0.0], dtype=float)
            x = d_r["Minutes"].to_numpy(dtype=float)
            y = np.cumsum(d_r["Licks"].to_numpy(dtype=float))
            if transform_licks:
                y = np.power(y, 0.25)
            return x, y

        def light_on_times(well: int) -> np.ndarray:
            col = f"W{well}"
            if col not in lights_r.columns or len(light_minutes) == 0:
                return np.array([], dtype=float)
            mask = lights_r[col].to_numpy(dtype=bool)
            return light_minutes[mask]

        def plot_well(ax, well: int, label: str, color: str) -> None:
            x, y = cumulative_for_well(well)
            ax.plot(x, y, linewidth=1.2, color=color, label=label)
            lx = light_on_times(well)
            if len(lx) > 0 and len(x) >= 2:
                ly = np.interp(lx, x, y, left=y[0], right=y[-1])
                ax.scatter(lx, ly, s=30, marker="D", color="gold",
                           edgecolors="darkorange", linewidths=0.5, zorder=3, label="Light on")
            ax.set_ylabel("Transformed Cum. Licks" if transform_licks else "Cum. Licks")
            ax.legend(fontsize=8)

        if isinstance(ch, OneWellChamber):
            fig, ax = _subplots(figsize=(10, 4))
            plot_well(ax, ch.well, f"W{ch.well}", "steelblue")
            ax.set_xlabel("Minutes")
            ax.set_title(title)
        elif single_plot:
            fig, ax = _subplots(figsize=(10, 4))
            plot_well(ax, ch.well_a, f"{self._well_label('A')} (W{ch.well_a})", "steelblue")
            plot_well(ax, ch.well_b, f"{self._well_label('B')} (W{ch.well_b})", "tomato")
            ax.set_xlabel("Minutes")
            ax.set_title(title)
        else:
            fig, axes = _subplots(2, 1, sharex=True, figsize=(10, 6))
            plot_well(axes[0], ch.well_a, f"{self._well_label('A')} (W{ch.well_a})", "steelblue")
            plot_well(axes[1], ch.well_b, f"{self._well_label('B')} (W{ch.well_b})", "tomato")
            axes[0].set_title(title)
            axes[-1].set_xlabel("Minutes")

        fig.tight_layout()
        return fig

