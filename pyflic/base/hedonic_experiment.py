from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence
import pandas as pd

from .two_well_experiment import TwoWellExperiment


@dataclass(slots=True)
class HedonicFeedingExperiment(TwoWellExperiment):
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
        experiment_dir: str | Path,
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
        experiment_dir:
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

        # load_experiment_yaml returns a TwoWellExperiment when chamber_size=2,
        # and raises ValueError if any DFM uses a different chamber_size.
        base = load_experiment_yaml(
            experiment_dir,
            range_minutes=range_minutes,
            parallel=parallel,
            max_workers=max_workers,
            executor=executor,
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
        dpi: int = 200,
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
            ``experiment_dir/analysis/hedonic_feeding_plot.{format}``.
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

        if fs.empty or len(fs.columns) == 0:
            n_treatments = len(self.design.treatments)
            n_chambers = sum(len(t.chambers) for t in self.design.treatments.values())
            raise ValueError(
                "Feeding summary is empty — no data was collected. "
                f"Design has {n_treatments} treatment(s) with {n_chambers} total chamber(s). "
                "Check that your flic_config.yaml has valid 'chambers:' assignments under each DFM, "
                "or that auto_remove_chambers() has not removed all chambers."
            )

        if col_a not in fs.columns or col_b not in fs.columns:
            raise ValueError(
                f"Feeding summary does not contain '{col_a}' and '{col_b}'. "
                f"Available columns: {list(fs.columns)}"
            )

        fs, group_col = self._resolve_group_col(fs)
        keep = [c for c in dict.fromkeys([group_col, "DFM", "Chamber"]) if c in fs.columns]
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
            facet_col=group_col,
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
            p.save(str(out), dpi=dpi, verbose=False)

        return p

    def execute_basic_analysis(
        self,
        *,
        data_breaks_multiplier: float = 4.0,
        bleeding_cutoff: float = 50.0,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool | None = None,
        plot_format: str = "png",
        dpi: int = 200,
        skip_qc: bool = False,
    ) -> dict[str, Path]:
        result = TwoWellExperiment.execute_basic_analysis(
            self,
            data_breaks_multiplier=data_breaks_multiplier,
            bleeding_cutoff=bleeding_cutoff,
            range_minutes=range_minutes,
            transform_licks=transform_licks,
            plot_format=plot_format,
            dpi=dpi,
            skip_qc=skip_qc,
        )
        self.hedonic_feeding_plot(save=True, range_minutes=range_minutes)
        self.weighted_duration_summary(save=True, range_minutes=range_minutes)
        return result

    def report_results_blocks(self, generic: list, options) -> list:
        """The two-well results, then the type's own: bout duration at each
        well by treatment, and its event-weighted summary."""
        from . import report_layout as rl

        blocks = list(generic)
        blocks += [
            rl.Heading("Hedonic feeding", level=2),
            rl.Paragraph("Duration is the measure of interest in a hedonic design: median "
                         "feeding-bout duration at each well, one point per chamber, "
                         "panelled by treatment."),
            rl.Plot(lambda: self.hedonic_feeding_plot(
                range_minutes=options.range_minutes, base_font_size=10.0, save=False),
                height=3.3),
        ]
        weighted = self.weighted_duration_summary(range_minutes=options.range_minutes,
                                                  save=False)
        if weighted is not None and not weighted.empty:
            blocks.append(rl.Table(weighted, caption="Median bout duration by treatment, "
                                                     "weighted by each chamber's events"))
        return blocks

    def weighted_duration_summary(
        self,
        *,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool = True,
        save: bool = True,
        path: str | Path | None = None,
    ) -> pd.DataFrame:
        """
        Median bout duration at each well, by treatment: the plain mean over
        chambers and the mean weighted by each chamber's events.

        For each treatment, per well (shown for A; B is the same with
        ``MedDurationB`` and ``EventsB``):

        - **MeanDurationA** — mean of ``MedDurationA`` over the chambers
          that have one.
        - **SemDurationA** — its standard error (sample std / √n; NaN for a
          single chamber).
        - **WeightedMeanDurationA** — mean of ``MedDurationA`` weighted by
          ``EventsA``, over chambers with a value and positive events.
        - **WeightedSemDurationA** — the weighted population std / √n.
        - **SampleSizeA** — how many chambers the weighted figures use.

        Parameters
        ----------
        range_minutes:
            Time window used to compute the feeding summary.
        transform_licks:
            Whether to apply lick transformation before computing the summary.
        save:
            Write the table as CSV (default ``True``).
        path:
            Where to write it; by default ``analysis/treatment_means.csv``.

        Returns
        -------
        pd.DataFrame
            One row per treatment with columns ``Treatment``, then
            ``MeanDurationA``, ``SemDurationA``, ``WeightedMeanDurationA``,
            ``WeightedSemDurationA``, ``SampleSizeA`` and the same five for B.
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

