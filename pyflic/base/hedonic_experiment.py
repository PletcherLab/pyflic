from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Literal, Sequence
import pandas as pd

from .experiment import Experiment


@dataclass(slots=True)
class HedonicFeedingExperiment(Experiment):
    filtered_chambers: pd.DataFrame | None = None
    """
    A two-well (choice) specialisation of :class:`Experiment`.

    ``HedonicFeedingExperiment.load()`` behaves identically to
    ``Experiment.load()`` but raises :class:`ValueError` if any DFM in the
    config has ``chamber_size != 2``.  All methods inherited from
    :class:`Experiment` are available unchanged.
    """

    @classmethod
    def load(
        cls,
        project_dir: str | Path,
        *,
        range_minutes: Sequence[float] = (0, 0),
        parallel: bool = True,
        max_workers: int | None = None,
        executor: Literal["threads", "processes"] = "threads",
    ) -> HedonicFeedingExperiment:
        """
        Load a hedonic feeding experiment from a project directory.

        Identical to :meth:`Experiment.load` but enforces that every DFM
        uses ``chamber_size=2`` (two-well / choice design).  Raises
        :class:`ValueError` if any DFM violates this requirement.

        Parameters
        ----------
        project_dir:
            Root directory for the experiment project.  Must contain
            ``flic_config.yaml`` with ``chamber_size: 2`` in the global or
            per-DFM params section.
        range_minutes:
            ``(start, end)`` time window in minutes.  ``(0, 0)`` loads all.
        parallel:
            Whether to load DFMs in parallel (default ``True``).
        max_workers:
            Maximum number of parallel workers; ``None`` for the default.
        executor:
            ``"threads"`` (default) or ``"processes"``.
        """
        from .yaml_config import load_experiment_yaml

        base = load_experiment_yaml(
            project_dir,
            range_minutes=range_minutes,
            parallel=parallel,
            max_workers=max_workers,
            executor=executor,
        )

        # Validate before constructing — gives a clear error before any state
        # is stored.
        bad = [
            dfm_id
            for dfm_id, dfm in base.dfms.items()
            if dfm.params.chamber_size != 2
        ]
        if bad:
            raise ValueError(
                f"HedonicFeedingExperiment requires chamber_size=2 for every DFM, "
                f"but DFM(s) {sorted(bad)} have chamber_size != 2.  "
                f"Set chamber_size: 2 in flic_config.yaml."
            )

        # Re-construct as HedonicFeedingExperiment using the same field values.
        return cls(**{f.name: getattr(base, f.name) for f in dataclasses.fields(base)})

    def hedonic_feeding_plot(
        self,
        *,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool = True,
        title: str = "",
        y_label: str | None = None,
        ylim: tuple[float, float] | None = None,
        annotation: str | None = None,
        annotation_x: float = 1.0,
        annotation_y: float = 0.0,
        jitter_width: float = 0.25,
        base_font_size: float = 20.0,
        save: bool = True,
        path: str | Path | None = None,
        format: str = "png",
        dpi: int = 150,
    ):
        """
        Jitter + mean + SE plot comparing WellA vs WellB durations faceted by treatment.

        Pulls the feeding summary from the experiment, melts WellA/WellB into a
        single ``Well`` column, and facets by treatment level.

        Well axis labels are taken from ``self.well_names`` (set via ``well_names``
        in ``flic_config.yaml``), falling back to ``"WellA"`` / ``"WellB"``.

        Parameters
        ----------
        save : bool
            Save the plot to disk (default ``False``).  When ``True`` and
            *path* is not given, writes to
            ``project_dir/analysis/hedonic_feeding_plot.{format}``.
        path : str | Path | None
            Explicit output path.  Ignored when ``save=False``.
        format : str
            File format passed to the plotnine ``save()`` call, e.g.
            ``"png"`` (default) or ``"pdf"``.
        dpi : int
            Resolution in dots per inch (default 150).
        """
        metric = "MedDuration"
        col_a = f"{metric}A"
        col_b = f"{metric}B"

        fs = self.feeding_summary(range_minutes=range_minutes, transform_licks=transform_licks)

        if fs.empty and len(fs.columns) == 0:
            n_treatments = len(self.design.treatments)
            n_chambers = sum(len(t.chambers) for t in self.design.treatments.values())
            raise ValueError(
                "Feeding summary is empty — no data was collected. "
                f"Design has {n_treatments} treatment(s) with {n_chambers} total chamber(s). "
                "Check that your flic_config.yaml has valid 'chambers:' assignments under each DFM, "
                "or that filter_flies() has not removed all chambers."
            )

        if col_a not in fs.columns or col_b not in fs.columns:
            raise ValueError(
                f"Feeding summary does not contain '{col_a}' and '{col_b}'. "
                f"Available columns: {list(fs.columns)}"
            )

        keep = [c for c in ("DFM", "Chamber", "Treatment") if c in fs.columns]
        df_a = fs[keep + [col_a, "EventsA"]].copy() if "EventsA" in fs.columns else fs[keep + [col_a]].copy()
        df_a["Well"] = "WellA"
        df_a = df_a.rename(columns={col_a: metric, "EventsA": "Events"})

        df_b = fs[keep + [col_b, "EventsB"]].copy() if "EventsB" in fs.columns else fs[keep + [col_b]].copy()
        df_b["Well"] = "WellB"
        df_b = df_b.rename(columns={col_b: metric, "EventsB": "Events"})

        df_long = pd.concat([df_a, df_b], ignore_index=True)

        wn = self.well_names or {}
        well_labels = {
            "WellA": wn.get("A", "WellA"),
            "WellB": wn.get("B", "WellB"),
        }

        size_col = "Events" if "Events" in df_long.columns else None

        p = self.plot_jitter_summary(
            df_long,
            x_col="Well",
            y_col=metric,
            facet_col="Treatment",
            title=title,
            x_label="Well",
            y_label=y_label or metric,
            ylim=ylim,
            x_order=["WellA", "WellB"],
            x_labels=well_labels,
            colors={"WellA": "steelblue", "WellB": "tomato"},
            annotation=annotation,
            annotation_x=annotation_x,
            annotation_y=annotation_y,
            jitter_width=jitter_width,
            size_col=size_col,
            base_font_size=base_font_size,
        )

        if save:
            if path is None:
                out = self.analysis_dir / f"hedonic_feeding_plot.{format}"
            else:
                out = Path(path).expanduser().resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            p.save(str(out), dpi=dpi)

        return p


    
    def filter_flies(
        self,
        *,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool = True,
    ) -> pd.DataFrame:
        """
        Remove chambers that fail QC thresholds defined in ``global_constants``.

        Thresholds (all read from ``self.global_constants``):

        - ``min_transform_licks_cutoff`` — remove if ``LicksA`` is **below** this value.
        - ``max_med_duration_cutoff``    — remove if ``MedDurationA`` is **above** this value.
        - ``max_events_cutoff``          — remove if ``EventsB`` is **above** this value.

        A chamber is removed if it fails *any* of the applicable thresholds.
        Missing constants are silently skipped.  The feeding-summary cache is
        cleared after filtering so subsequent calls reflect the updated design.

        Parameters
        ----------
        range_minutes:
            Time window used to compute the feeding summary for evaluation.
            Defaults to the full experiment range.
        transform_licks:
            Whether to apply lick transformation before evaluating thresholds.

        Returns
        -------
        pd.DataFrame
            Record of every removed chamber with columns
            ``DFM``, ``Chamber``, ``Treatment``, and ``Reason``.
        """
        constants = self.global_constants
        min_licks = constants.get("min_transform_licks_cutoff")
        max_dur = constants.get("max_med_duration_cutoff")
        max_events = constants.get("max_events_cutoff")

        fs = self.feeding_summary(range_minutes=range_minutes, transform_licks=transform_licks)

        to_remove: dict[tuple[int, int], list[str]] = {}
        removed_rows: list[dict] = []

        for _, row in fs.iterrows():
            dfm_id = int(row["DFM"])
            chamber_idx = int(row["Chamber"])
            treatment_name = str(row["Treatment"])
            reasons: list[str] = []

            if min_licks is not None and "LicksA" in row.index:
                val = row["LicksA"]
                if pd.notna(val) and float(val) < float(min_licks):
                    reasons.append(
                        f"LicksA={val:.6g} < min_transform_licks_cutoff={min_licks}"
                    )

            if max_dur is not None and "MedDurationA" in row.index:
                val = row["MedDurationA"]
                if pd.notna(val) and float(val) > float(max_dur):
                    reasons.append(
                        f"MedDurationA={val:.6g} > max_med_duration_cutoff={max_dur}"
                    )

            if max_events is not None and "EventsB" in row.index:
                val = row["EventsB"]
                if pd.notna(val) and float(val) > float(max_events):
                    reasons.append(
                        f"EventsB={val:.6g} > max_events_cutoff={max_events}"
                    )

            if reasons:
                key = (dfm_id, chamber_idx)
                if key not in to_remove:
                    to_remove[key] = reasons
                    reason_str = "; ".join(reasons)
                    removed_rows.append(
                        {
                            "DFM": dfm_id,
                            "Chamber": chamber_idx,
                            "Treatment": treatment_name,
                            "Reason": reason_str,
                        }
                    )
                    print(
                        f"  [filter_flies] Removing DFM {dfm_id} Chamber {chamber_idx}"
                        f" ({treatment_name}): {reason_str}",
                        flush=True,
                    )

        # Remove failing chambers from every treatment in the design.
        if to_remove:
            remove_set = set(to_remove.keys())
            for treatment in self.design.treatments.values():
                treatment.chambers = [
                    tc
                    for tc in treatment.chambers
                    if (tc.dfm_id, tc.chamber_index) not in remove_set
                ]
            # Invalidate cache — design has changed.
            self._feeding_summary_cache.clear()

        result = pd.DataFrame(
            removed_rows, columns=["DFM", "Chamber", "Treatment", "Reason"]
        )
        if result.empty:
            print("filter_flies: no chambers removed.", flush=True)
        else:
            print(f"filter_flies: done — {len(result)} chamber(s) removed.", flush=True)

        self.filtered_chambers = result
        return result

    def summary_text(
        self,
        *,
        include_qc: bool = True,
        qc_data_breaks_multiplier: float = 4.0,
        qc_bleeding_cutoff: float = 50.0,
    ) -> str:
        """
        Return the experiment summary, appending a filtered-chambers section
        when :meth:`filter_flies` has been called and removed any chambers.
        """
        text = Experiment.summary_text(
            self,
            include_qc=include_qc,
            qc_data_breaks_multiplier=qc_data_breaks_multiplier,
            qc_bleeding_cutoff=qc_bleeding_cutoff,
        )

        if self.filtered_chambers is not None and not self.filtered_chambers.empty:
            buf = StringIO()
            buf.write("\nFiltered chambers (removed by filter_flies)\n")
            buf.write("-------------------------------------------\n")
            for _, row in self.filtered_chambers.iterrows():
                buf.write(
                    f"DFM {int(row['DFM'])} Chamber {int(row['Chamber'])}"
                    f" Treatment={row['Treatment']}: {row['Reason']}\n"
                )
            buf.write("\n")
            text += buf.getvalue()

        return text

    def execute_basic_analysis(self) -> None:
        Experiment.execute_basic_analysis(self)
        self.hedonic_feeding_plot(save=True)
        self.weighted_duration_summary(save=True)

    def weighted_duration_summary(
        self,
        *,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool = True,
        save: bool = True,
        path: str | Path | None = None,
    ) -> pd.DataFrame:
        """
        Compute weighted mean and weighted standard deviation of median bout
        duration for each treatment, using event counts as weights.

        For each treatment the following are calculated per well:

        - **WeightedMeanDurationA** — weighted mean of ``MedDurationA``
          using ``EventsA`` as weights.
        - **WeightedStdDurationA**  — weighted population std of ``MedDurationA``
          using ``EventsA`` as weights.
        - **WeightedMeanDurationB** / **WeightedStdDurationB** — same for Well B.
        - **N** — number of chambers that contributed (rows with finite weights
          and values for both wells).

        Parameters
        ----------
        range_minutes:
            Time window used to compute the feeding summary.
        transform_licks:
            Whether to apply lick transformation before computing the summary.

        Returns
        -------
        pd.DataFrame
            One row per treatment with columns ``Treatment``,
            ``WeightedMeanDurationA``, ``WeightedStdDurationA``,
            ``WeightedMeanDurationB``, ``WeightedStdDurationB``, ``N``.
        """
        import numpy as np

        fs = self.feeding_summary(range_minutes=range_minutes, transform_licks=transform_licks)

        required = {"Treatment", "MedDurationA", "MedDurationB", "EventsA", "EventsB"}
        missing = required - set(fs.columns)
        if missing:
            raise ValueError(
                f"Feeding summary is missing columns required for weighted duration summary: {missing}"
            )

        def _weighted_stats(values: np.ndarray, weights: np.ndarray):
            """Return (weighted_mean, weighted_std, weighted_sem, n) ignoring NaN/non-positive weights."""
            mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
            v, w = values[mask], weights[mask]
            n = int(mask.sum())
            if n == 0 or w.sum() == 0:
                return float("nan"), float("nan"), float("nan"), 0
            wmean = np.sum(w * v) / np.sum(w)
            wstd = float(np.sqrt(np.sum(w * (v - wmean) ** 2) / np.sum(w)))
            wsem = wstd / np.sqrt(n)
            return float(wmean), wstd, float(wsem), n

        def _unweighted_stats(values: np.ndarray):
            """Return (mean, sem, n) ignoring NaN values."""
            v = values[np.isfinite(values)]
            n = len(v)
            if n == 0:
                return float("nan"), float("nan"), 0
            mean = float(np.mean(v))
            sem = float(np.std(v, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
            return mean, sem, n

        rows: list[dict] = []
        for treatment_name, group in fs.groupby("Treatment", sort=True):
            dur_a = group["MedDurationA"].to_numpy(dtype=float)
            dur_b = group["MedDurationB"].to_numpy(dtype=float)
            ev_a = group["EventsA"].to_numpy(dtype=float)
            ev_b = group["EventsB"].to_numpy(dtype=float)

            wmean_a, _, wsem_a, n_a = _weighted_stats(dur_a, ev_a)
            wmean_b, _, wsem_b, n_b = _weighted_stats(dur_b, ev_b)
            mean_a, sem_a, _ = _unweighted_stats(dur_a)
            mean_b, sem_b, _ = _unweighted_stats(dur_b)

            rows.append(
                {
                    "Treatment": treatment_name,
                    "MeanDurationA": mean_a,
                    "SemDurationA": sem_a,
                    "WeightedMeanDurationA": wmean_a,
                    "WeightedSemDurationA": wsem_a,
                    "SampleSizeA": n_a,
                    "MeanDurationB": mean_b,
                    "SemDurationB": sem_b,
                    "WeightedMeanDurationB": wmean_b,
                    "WeightedSemDurationB": wsem_b,
                    "SampleSizeB": n_b,
                }
            )

        df = pd.DataFrame(rows, columns=[
            "Treatment",
            "MeanDurationA", "SemDurationA",
            "WeightedMeanDurationA", "WeightedSemDurationA", "SampleSizeA",
            "MeanDurationB", "SemDurationB",
            "WeightedMeanDurationB", "WeightedSemDurationB", "SampleSizeB",
        ])

        if save:
            if path is None:
                out = self.analysis_dir / "treatment_means.csv"
            else:
                out = Path(path).expanduser().resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)

        return df

    def validate(self) -> None:
        """
        Re-check that all loaded DFMs use ``chamber_size=2``.

        Useful after manually modifying DFM parameters.  Raises
        :class:`ValueError` if any DFM is not two-well.
        """
        bad = [
            dfm_id
            for dfm_id, dfm in self.dfms.items()
            if dfm.params.chamber_size != 2
        ]
        if bad:
            raise ValueError(
                f"HedonicFeedingExperiment requires chamber_size=2, "
                f"but DFM(s) {sorted(bad)} have chamber_size != 2."
            )
