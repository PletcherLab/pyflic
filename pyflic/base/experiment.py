from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Literal, Sequence

import pandas as pd

from .dfm import DFM
from .experiment_design import ExperimentDesign


def _fmt_min(v: float) -> str:
    """Format a minute value for use in a filename.  ``inf`` → ``'end'``."""
    if v == float("inf"):
        return "end"
    return str(int(v)) if v == int(v) else str(v)


@dataclass(slots=True)
class Experiment:
    """
    Concrete experiment instance created from configuration.
    """

    dfms: dict[int, DFM]
    design: ExperimentDesign
    global_config: dict
    global_constants: dict = field(default_factory=dict)
    well_names: dict[str, str] | None = None   # e.g. {"A": "Sucrose", "B": "Yeast"}
    design_factors: list[str] | None = None    # ordered factor names from experimental_design_factors
    chamber_factors: dict | None = None         # {(dfm_id, chamber_idx): {factor: level}}
    config_path: Path | None = None
    data_dir: Path | None = None
    project_dir: Path | None = None
    range_minutes: tuple[float, float] | None = None
    parallel: bool | None = None
    executor: Literal["threads", "processes"] | None = None
    max_workers: int | None = None
    qc_report_dir: Path | None = None
    qc_results: dict[int, dict[str, Any]] | None = None
    _feeding_summary_cache: dict = field(default_factory=dict)

    def get_dfm(self, dfm_id: int) -> DFM | None:
        """Return the DFM with the given id, or None if it does not exist."""
        return self.dfms.get(int(dfm_id))

    @property
    def analysis_dir(self) -> Path | None:
        """Return ``project_dir/analysis``, or ``None`` if no ``project_dir`` is set."""
        if self.project_dir is None:
            return None
        return self.project_dir / "analysis"

    @property
    def qc_dir(self) -> Path | None:
        """Return ``project_dir/qc``, or ``None`` if no ``project_dir`` is set."""
        if self.project_dir is None:
            return None
        return self.project_dir / "qc"

    def _auto_save_fig(self, fig: Any, default_name: str) -> None:
        """Save fig to analysis_dir/default_name when project_dir is set."""
        d = self.analysis_dir
        if d is not None:
            d.mkdir(parents=True, exist_ok=True)
            fig.savefig(d / default_name, bbox_inches="tight")

    @classmethod
    def load(
        cls,
        project_dir: str | Path,
        *,
        range_minutes: Sequence[float] = (0, 0),
        parallel: bool = True,
        max_workers: int | None = None,
        executor: Literal["threads", "processes"] = "threads",
    ) -> Experiment:
        """
        Load an experiment from a project directory.

        ``project_dir`` is the single required argument and is the root of the
        project layout::

            project_dir/
              flic_config.yaml   ← required; loaded automatically
              data/              ← DFM CSV files
              qc/                ← QC report output (write_qc_reports)
              analysis/          ← figure and summary output (_auto_save_fig,
                                    write_summary)

        Parameters
        ----------
        project_dir:
            Root directory for the experiment project.  Must contain
            ``flic_config.yaml``.  Data is always read from ``project_dir/data``.
        range_minutes:
            ``(start, end)`` time window in minutes.  ``(0, 0)`` means load all.
        parallel:
            Whether to load DFMs in parallel.
        max_workers:
            Maximum number of parallel workers; ``None`` for the default.
        executor:
            ``"threads"`` (default) or ``"processes"``.
        """

        from .yaml_config import load_experiment_yaml

        return load_experiment_yaml(
            project_dir,
            range_minutes=range_minutes,
            parallel=parallel,
            max_workers=max_workers,
            executor=executor,
        )

    def _simultaneous_feeding_matrix_df(self, dfm: DFM) -> pd.DataFrame:
        mat = dfm.simultaneous_feeding_matrix()
        n = int(dfm.params.chamber_sets.shape[0])
        return pd.DataFrame(
            mat,
            index=[f"Chamber{i}" for i in range(1, n + 1)],
            columns=["Licks1", "Licks2", "Both", "MaxMinSignalAtBoth", "HigherInCol1AtBoth"],
        )

    def _dfm_actual_range_minutes(self, dfm: DFM) -> tuple[float, float] | None:
        """
        Best-effort actual time range in minutes, based on the loaded raw data.
        """

        for col in ("Minutes", "minutes", "Minute", "minute"):
            if col in dfm.raw_df.columns:
                s = pd.to_numeric(dfm.raw_df[col], errors="coerce").dropna()
                if s.empty:
                    return None
                return float(s.min()), float(s.max())
        return None

    def _max_experiment_minutes(self) -> float | None:
        """Return the actual maximum ``Minutes`` value across all loaded DFMs."""
        vals = [
            r[1]
            for dfm in self.dfms.values()
            if (r := self._dfm_actual_range_minutes(dfm)) is not None
        ]
        return max(vals) if vals else None

    def compute_qc_results(
        self,
        *,
        data_breaks_multiplier: float = 4.0,
        bleeding_cutoff: float = 50.0,
        include_integrity_text: bool = False,
    ) -> dict[int, dict[str, Any]]:
        """
        Compute QC results for each DFM (does not write any files).
        """

        results: dict[int, dict[str, Any]] = {}
        for dfm_id in sorted(int(k) for k in self.dfms.keys()):
            dfm = self.dfms[dfm_id]

            # Integrity report prints; optionally capture it.
            integrity_text = None
            if include_integrity_text:
                buf = StringIO()
                with redirect_stdout(buf):
                    integrity = dfm.integrity_report()
                integrity_text = buf.getvalue()
            else:
                integrity = dfm.integrity_report()

            breaks = dfm.data_breaks(multiplier=float(data_breaks_multiplier))
            breaks_count = 0 if breaks is None else int(len(breaks))

            sim_df = None
            bleed = None
            if dfm.params.chamber_size == 2:
                sim_df = self._simultaneous_feeding_matrix_df(dfm)
                bleed = dfm.bleeding_check(cutoff=float(bleeding_cutoff))

            results[dfm_id] = {
                "integrity": integrity,
                "integrity_text": integrity_text,
                "data_breaks_multiplier": float(data_breaks_multiplier),
                "data_breaks_count": breaks_count,
                "data_breaks_head": None if breaks is None else breaks.head(10),
                "simultaneous_feeding_matrix": sim_df,
                "bleeding_cutoff": float(bleeding_cutoff),
                "bleeding": bleed,
            }

        self.qc_results = results
        return results

    def write_qc_reports(
        self,
        out_dir: str | Path | None = None,
        *,
        data_breaks_multiplier: float = 4.0,
        bleeding_cutoff: float = 50.0,
    ) -> Path:
        """
        Write per-DFM QC reports (CSV/TXT) to *out_dir* and return the resolved directory path.

        If *out_dir* is not given, defaults to ``project_dir/qc``.  Raises
        ``ValueError`` if neither *out_dir* nor ``project_dir`` is set.
        """

        if out_dir is None:
            if self.qc_dir is None:
                raise ValueError(
                    "out_dir must be provided when no project_dir is set on the Experiment."
                )
            out_dir = self.qc_dir
        out = Path(out_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)

        n_total = len(self.dfms)
        print(f"  Computing QC for {n_total} DFM(s)...", flush=True)
        qc = self.compute_qc_results(
            data_breaks_multiplier=data_breaks_multiplier,
            bleeding_cutoff=bleeding_cutoff,
            include_integrity_text=True,
        )
        print(f"  Writing QC files to {out}", flush=True)

        import matplotlib.pyplot as plt
        for n, (dfm_id, r) in enumerate(qc.items(), 1):
            print(f"  DFM {dfm_id}  ({n}/{n_total})...", flush=True)

            integrity = r["integrity"]
            pd.DataFrame([integrity]).to_csv(out / f"DFM{dfm_id}_integrity_report.csv", index=False)
            if r.get("integrity_text"):
                (out / f"DFM{dfm_id}_integrity_report.txt").write_text(
                    str(r["integrity_text"]), encoding="utf-8"
                )

            # Data breaks
            breaks_head = r.get("data_breaks_head")
            breaks_count = int(r.get("data_breaks_count", 0))
            if breaks_count > 0 and isinstance(breaks_head, pd.DataFrame):
                print(f"    {breaks_count} data break(s) found — saving CSV", flush=True)
                dfm = self.dfms[dfm_id]
                breaks = dfm.data_breaks(multiplier=float(data_breaks_multiplier))
                if breaks is not None:
                    breaks.to_csv(out / f"DFM{dfm_id}_data_breaks.csv", index=False)

            # Two-well only
            sim_df = r.get("simultaneous_feeding_matrix")
            if isinstance(sim_df, pd.DataFrame):
                sim_df.to_csv(out / f"DFM{dfm_id}_simultaneous_feeding_matrix.csv")

            bleed = r.get("bleeding")
            if isinstance(bleed, dict) and "Matrix" in bleed and "AllData" in bleed:
                bleed["Matrix"].to_csv(out / f"DFM{dfm_id}_bleeding_matrix.csv")
                bleed["AllData"].to_csv(out / f"DFM{dfm_id}_bleeding_alldata.csv", header=True)

            # Signal plots
            dfm = self.dfms[dfm_id]
            fig = dfm.plot_raw()
            fig.savefig(out / f"DFM{dfm_id}_raw.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            fig = dfm.plot_baselined(include_thresholds=True)
            fig.savefig(out / f"DFM{dfm_id}_baselined.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            fig = dfm.plot_cumulative_licks(transform_licks=True)
            fig.savefig(out / f"DFM{dfm_id}_cumulative_licks.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        print(f"  QC complete — {n_total} DFM(s) processed.", flush=True)
        self.qc_report_dir = out
        return out

    def summary_text(
        self,
        *,
        include_qc: bool = True,
        qc_data_breaks_multiplier: float = 4.0,
        qc_bleeding_cutoff: float = 50.0,
    ) -> str:
        """
        Return a human-readable summary of what was loaded and how it was configured.
        """

        buf = StringIO()
        buf.write("FLIC experiment summary\n")
        buf.write("======================\n\n")

        if self.config_path is not None:
            buf.write(f"Config: {self.config_path}\n")
        if self.data_dir is not None:
            buf.write(f"Data dir: {self.data_dir}\n")
        if self.range_minutes is not None:
            buf.write(f"Requested range minutes: {self.range_minutes}\n")
        if self.parallel is not None:
            buf.write(f"Parallel: {self.parallel}\n")
        if self.executor is not None:
            buf.write(f"Executor: {self.executor}\n")
        if self.max_workers is not None:
            buf.write(f"Max workers: {self.max_workers}\n")
        if self.qc_report_dir is not None:
            buf.write(f"QC report dir: {self.qc_report_dir}\n")

        dfm_ids = sorted(int(k) for k in self.dfms.keys())
        buf.write(f"Loaded DFMs: {dfm_ids}\n")
        buf.write(f"Treatments: {sorted(self.design.treatments.keys())}\n\n")

        # Global params (if any)
        global_params = None
        if isinstance(self.global_config, dict):
            global_params = self.global_config.get("params", self.global_config.get("parameters", None))
        if global_params is not None:
            buf.write("Global params\n")
            buf.write("------------\n")
            if isinstance(global_params, dict) and global_params:
                for k in sorted(global_params.keys()):
                    buf.write(f"{k}: {global_params[k]!r}\n")
            else:
                buf.write(repr(global_params) + "\n")
            buf.write("\n")

        # Experimental design
        design_df = self.design.design_table()
        if not design_df.empty and all(c in design_df.columns for c in ("Treatment", "DFM", "Chamber")):
            design_df = design_df.sort_values(["Treatment", "DFM", "Chamber"])

        buf.write("Experimental design (DFM/Chamber -> Treatment)\n")
        buf.write("--------------------------------------------\n")
        if design_df.empty:
            buf.write("(empty)\n\n")
        else:
            buf.write(design_df.to_string(index=False))
            buf.write("\n\n")

        # Per-DFM details
        overall_min: float | None = None
        overall_max: float | None = None
        buf.write("DFM details\n")
        buf.write("-----------\n")
        for dfm_id in dfm_ids:
            dfm = self.dfms[dfm_id]
            buf.write(f"DFM {dfm_id}\n")
            buf.write(f"- version: {dfm.version}\n")
            buf.write(f"- raw rows: {len(dfm.raw_df)}\n")
            if dfm.source_files:
                buf.write("- source_files:\n")
                for p in dfm.source_files:
                    buf.write(f"  - {p}\n")
            actual = self._dfm_actual_range_minutes(dfm)
            if actual is not None:
                buf.write(f"- actual range minutes: ({actual[0]}, {actual[1]})\n")
                overall_min = actual[0] if overall_min is None else min(overall_min, actual[0])
                overall_max = actual[1] if overall_max is None else max(overall_max, actual[1])

            params = dfm.params
            buf.write("- params:\n")
            buf.write(f"  - chamber_size: {params.chamber_size!r}\n")
            buf.write(f"  - pi_direction: {params.pi_direction!r}\n")
            buf.write(f"  - correct_for_dual_feeding: {params.correct_for_dual_feeding!r}\n")
            buf.write(f"  - baseline_window_minutes: {params.baseline_window_minutes!r}\n")
            buf.write(f"  - feeding_threshold: {params.feeding_threshold!r}\n")
            buf.write(f"  - feeding_minimum: {params.feeding_minimum!r}\n")
            buf.write(f"  - tasting_minimum: {params.tasting_minimum!r}\n")
            buf.write(f"  - tasting_maximum: {params.tasting_maximum!r}\n")
            buf.write(f"  - feeding_minevents: {params.feeding_minevents!r}\n")
            buf.write(f"  - tasting_minevents: {params.tasting_minevents!r}\n")
            buf.write(f"  - samples_per_second: {params.samples_per_second!r}\n")
            buf.write(f"  - feeding_event_link_gap: {params.feeding_event_link_gap!r}\n")
            buf.write(f"  - chamber_sets: {params.chamber_sets.tolist()!r}\n")
            buf.write("\n")

        if overall_min is not None and overall_max is not None:
            buf.write(f"Overall actual loaded range (minutes): ({overall_min}, {overall_max})\n")

        if include_qc:
            buf.write("\nQC summary\n")
            buf.write("----------\n")
            qc = self.qc_results
            if qc is None:
                qc = self.compute_qc_results(
                    data_breaks_multiplier=qc_data_breaks_multiplier,
                    bleeding_cutoff=qc_bleeding_cutoff,
                    include_integrity_text=False,
                )

            for dfm_id in dfm_ids:
                r = qc.get(dfm_id, {})
                integrity = r.get("integrity", {})
                buf.write(f"DFM {dfm_id}\n")
                if isinstance(integrity, dict) and integrity:
                    buf.write(
                        f"- integrity: rows={integrity.get('n_rawdata')}, "
                        f"elapsed_min={integrity.get('elapsed_minutes_from_minutes_col')}, "
                        f"index_increments_by_one={integrity.get('index_increments_by_one')}\n"
                    )

                buf.write(
                    f"- data_breaks: count={int(r.get('data_breaks_count', 0))} "
                    f"(multiplier={r.get('data_breaks_multiplier')})\n"
                )
                head = r.get("data_breaks_head")
                if isinstance(head, pd.DataFrame) and not head.empty:
                    buf.write("  head:\n")
                    for line in head.to_string(index=False).splitlines():
                        buf.write(f"  {line}\n")

                sim_df = r.get("simultaneous_feeding_matrix")
                if isinstance(sim_df, pd.DataFrame):
                    buf.write("- simultaneous feeding (two-well):\n")
                    for line in sim_df.to_string().splitlines():
                        buf.write(f"  {line}\n")

                bleed = r.get("bleeding")
                if isinstance(bleed, dict) and "Matrix" in bleed and "AllData" in bleed:
                    mat_df = bleed["Matrix"]
                    all_data = bleed["AllData"]
                    max_resp = float(getattr(mat_df.to_numpy(), "max", lambda: 0.0)())
                    buf.write(f"- bleeding check (cutoff={r.get('bleeding_cutoff')}): max_matrix={max_resp}\n")
                    buf.write("  all_data (mean signal per well):\n")
                    for line in all_data.to_string().splitlines():
                        buf.write(f"  {line}\n")

                buf.write("\n")

        return buf.getvalue()

    def plot_cumulative_licks_chamber(
        self,
        dfm_id: int,
        chamber: int,
        *,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool = True,
        single_plot: bool = True,
    ):
        """
        Plot cumulative licks for a single DFM chamber, with the treatment name
        in the title if that chamber is assigned to one.
        """
        dfm = self.dfms[int(dfm_id)]
        treatment = self.design.treatment_for(dfm_id, chamber)
        return dfm.plot_cumulative_licks_chamber(
            chamber,
            range_minutes=range_minutes,
            transform_licks=transform_licks,
            single_plot=single_plot,
            treatment=treatment,
        )

    def plot_jitter_summary(
        self,
        df: pd.DataFrame,
        *,
        x_col: str,
        y_col: str,
        facet_col: str = "Treatment",
        title: str = "",
        x_label: str | None = None,
        y_label: str | None = None,
        ylim: tuple[float, float] | None = None,
        x_order: list[str] | None = None,
        x_labels: dict[str, str] | None = None,
        colors: dict[str, str] | None = None,
        annotation: str | None = None,
        annotation_x: float = 1.0,
        annotation_y: float = 0.0,
        jitter_width: float = 0.25,
        point_size: float = 3.0,
        size_col: str | None = None,
        base_font_size: float = 20.0,
    ):
        """
        Jitter + mean + SE error bar plot faceted by a grouping column.

        Mirrors the R pattern::

            ggplot(df, aes(x_col, y_col, color=x_col))
              + geom_jitter(width=0.25, size=3)
              + stat_summary(fun=mean)
              + stat_summary(fun.data=mean_se, geom="errorbar")
              + scale_color_brewer(palette="RdGy")
              + facet_wrap(facet_col)
              + theme_bw(base_size=20)

        Parameters
        ----------
        df : pd.DataFrame
            Feeding summary (or any DataFrame) with the relevant columns.
            Pass ``exp.feeding_summary()`` or a version augmented with
            extra metadata columns (e.g. a "Sucrose" concentration column).
        x_col : str
            Column for the x-axis grouping (e.g. ``"Sucrose"``).
        y_col : str
            Column for the y-axis metric (e.g. ``"MedDurationA"``).
        facet_col : str
            Column to facet by (default ``"Treatment"``).
        x_order : list[str] | None
            Explicit ordering of x-axis categories.
        annotation : str | None
            Optional text label to draw on every facet panel.
        annotation_x / annotation_y
            Data coordinates for the annotation.
        """
        from plotnine import (
            ggplot, aes, geom_jitter, stat_summary,
            scale_color_brewer, scale_color_manual, scale_x_discrete, coord_cartesian,
            facet_wrap, theme_bw, labs, annotate as p9_annotate,
        )
        import numpy as np

        jitter_aes = aes(size=size_col) if size_col is not None else {}
        p = (
            ggplot(df, aes(x=x_col, y=y_col, color=x_col))
            + geom_jitter(jitter_aes, width=jitter_width, **({} if size_col else {"size": point_size}))
            + stat_summary(fun_y=np.mean, geom="point", color="#333333", shape="x", size=4)
            + stat_summary(
                fun_ymin=lambda x: x.mean() - x.sem(),
                fun_ymax=lambda x: x.mean() + x.sem(),
                geom="errorbar",
                color="#333333",
                size=0.35,
            )
            + facet_wrap(f"~ {facet_col}")
            + theme_bw(base_size=base_font_size)
            + labs(title=title, x=x_label or x_col, y=y_label or y_col)
        )

        if colors is not None:
            p = p + scale_color_manual(values=colors)
        else:
            p = p + scale_color_brewer(type="div", palette="RdGy")

        if x_order is not None or x_labels is not None:
            limits = x_order  # may be None, scale_x_discrete handles that
            labels = ([x_labels.get(v, v) for v in x_order] if (x_labels and x_order) else None)
            p = p + scale_x_discrete(limits=limits, labels=labels)

        if ylim is not None:
            p = p + coord_cartesian(ylim=ylim)

        if annotation is not None:
            p = p + p9_annotate("text", x=annotation_x, y=annotation_y, label=annotation)

        return p

    def plot_feeding_summary(
        self,
        *,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool = True,
        ncols: int | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> Any:
        """
        Produce a faceted jitter+box plot of feeding summary metrics grouped by treatment.

        Treatments appear on the x-axis; each facet shows one metric with free y scales.
        Individual chamber values are overlaid as jittered coloured dots on grey box plots.

        For chamber_size=1 (8 metrics) the default grid is 3 columns.
        For chamber_size=2 (18 metrics) the default grid is 4 columns.

        Returns a plotnine ggplot object.
        """
        import math

        from plotnine import (
            aes,
            element_text,
            facet_wrap,
            geom_boxplot,
            geom_jitter,
            ggplot,
            theme,
            theme_bw,
        )

        df = self.feeding_summary(
            range_minutes=range_minutes, transform_licks=transform_licks
        )
        if df.empty:
            from plotnine import annotate
            return (
                ggplot()
                + annotate("text", x=0, y=0, label="No feeding summary data")
                + theme_bw()
            )

        chamber_size = next(iter(self.dfms.values())).params.chamber_size

        one_well_metrics = [
            ("Licks", "Licks"),
            ("Events", "Events"),
            ("MeanDuration", "Mean Duration (s)"),
            ("MedDuration", "Median Duration (s)"),
            ("MeanTimeBtw", "Mean Time Btw (s)"),
            ("MedTimeBtw", "Median Time Btw (s)"),
            ("MeanInt", "Mean Interval (s)"),
            ("MedianInt", "Median Interval (s)"),
        ]

        wn = self.well_names or {}
        na = wn.get("A", "A")
        nb = wn.get("B", "B")
        two_well_metrics = [
            ("PI", "PI"),
            ("EventPI", "Event PI"),
            ("LicksA", f"Licks ({na})"),
            ("LicksB", f"Licks ({nb})"),
            ("EventsA", f"Events ({na})"),
            ("EventsB", f"Events ({nb})"),
            ("MeanDurationA", f"Mean Duration {na} (s)"),
            ("MeanDurationB", f"Mean Duration {nb} (s)"),
            ("MedDurationA", f"Median Duration {na} (s)"),
            ("MedDurationB", f"Median Duration {nb} (s)"),
            ("MeanTimeBtwA", f"Mean Time Btw {na} (s)"),
            ("MeanTimeBtwB", f"Mean Time Btw {nb} (s)"),
            ("MedTimeBtwA", f"Median Time Btw {na} (s)"),
            ("MedTimeBtwB", f"Median Time Btw {nb} (s)"),
            ("MeanIntA", f"Mean Interval {na} (s)"),
            ("MeanIntB", f"Mean Interval {nb} (s)"),
            ("MedianIntA", f"Median Interval {na} (s)"),
            ("MedianIntB", f"Median Interval {nb} (s)"),
        ]

        all_metrics = two_well_metrics if chamber_size == 2 else one_well_metrics
        metrics = [(col, lbl) for col, lbl in all_metrics if col in df.columns]

        if not metrics:
            from plotnine import annotate
            return (
                ggplot()
                + annotate("text", x=0, y=0, label="No matching columns in feeding summary")
                + theme_bw()
            )

        id_cols = [c for c in ("Treatment", "DFM", "Chamber") if c in df.columns]
        metric_cols = [col for col, _ in metrics]
        label_map = {col: lbl for col, lbl in metrics}

        df_long = df[id_cols + metric_cols].melt(
            id_vars=id_cols,
            value_vars=metric_cols,
            var_name="_MetricCol",
            value_name="Value",
        )
        df_long["Metric"] = df_long["_MetricCol"].map(label_map)
        df_long["Metric"] = pd.Categorical(
            df_long["Metric"],
            categories=[lbl for _, lbl in metrics],
            ordered=True,
        )
        df_long = df_long.drop(columns=["_MetricCol"])

        if ncols is None:
            ncols = 4 if chamber_size == 2 else 3
        nrows = math.ceil(len(metrics) / ncols)

        if figsize is None:
            figsize = (ncols * 3.5, nrows * 3.2)

        treatments = sorted(df["Treatment"].unique().tolist())

        p = (
            ggplot(df_long, aes(x="Treatment", y="Value"))
            + geom_boxplot(
                aes(group="Treatment"),
                fill="white",
                color="#666666",
                alpha=0.5,
                outlier_alpha=0,
                width=0.4,
            )
            + geom_jitter(aes(color="Treatment"), width=0.15, size=2, alpha=0.8)
            + facet_wrap("~ Metric", ncol=ncols, scales="free_y")
            + theme_bw()
            + theme(
                axis_text_x=element_text(rotation=30, hjust=1, size=8),
                strip_text=element_text(size=8),
                legend_position="none",
                figure_size=figsize,
            )
        )

        return p

    def _binned_licks_table_by_treatment(
        self,
        *,
        binsize_min: float = 30.0,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool = True,
    ) -> pd.DataFrame:
        """
        Build a long table of binned feeding-summary rows for chambers assigned to treatments.

        Output columns:
        - Treatment, DFM, Chamber, Interval, Minutes, ... (feeding summary metrics)
        """

        def range_is_specified_local(r: Sequence[float]) -> bool:
            try:
                return not (float(r[0]) == 0.0 and float(r[1]) == 0.0)
            except Exception:  # noqa: BLE001
                return False

        # If caller didn't specify a range but the experiment was loaded with one, use it.
        effective_range = range_minutes
        if (not range_is_specified_local(range_minutes)) and self.range_minutes is not None:
            if range_is_specified_local(self.range_minutes):
                effective_range = self.range_minutes

        # Cache binned summary per DFM so we don't recompute repeatedly per treatment.
        binned_by_dfm: dict[int, pd.DataFrame] = {}
        for dfm_id, dfm in self.dfms.items():
            binned_by_dfm[int(dfm_id)] = dfm.binned_feeding_summary(
                binsize_min=float(binsize_min),
                range_minutes=effective_range,
                transform_licks=bool(transform_licks),
            )

        parts: list[pd.DataFrame] = []
        for trt_name, trt in self.design.treatments.items():
            if not trt.chambers:
                continue

            # Group selection by DFM
            by_dfm: dict[int, set[int]] = {}
            for tc in trt.chambers:
                by_dfm.setdefault(int(tc.dfm_id), set()).add(int(tc.chamber_index))

            for dfm_id, chamber_idxs in by_dfm.items():
                binned = binned_by_dfm.get(int(dfm_id))
                if binned is None or binned.empty or "Chamber" not in binned.columns:
                    continue

                tmp = binned[binned["Chamber"].astype(int).isin(chamber_idxs)].copy()
                if tmp.empty:
                    continue

                tmp.insert(0, "Treatment", str(trt_name))

                # Ensure DFM exists for grouping even if upstream summary didn't include it.
                if "DFM" not in tmp.columns:
                    tmp.insert(tmp.columns.get_loc("Treatment") + 1, "DFM", int(dfm_id))
                parts.append(tmp)

        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def _metric_series_from_binned_rows(
        self,
        binned_rows: pd.DataFrame,
        *,
        metric: str,
        two_well_mode: Literal["total", "mean_ab", "A", "B"] = "total",
    ) -> pd.Series:
        """
        Given binned feeding-summary rows, compute a 1D series of metric values.

        `metric` can be:
        - a column name present in the binned feeding summary (e.g. 'PI', 'EventPI', 'LicksA', 'MedDurationB', ...)
        - one of the common base names: 'Licks', 'Events', 'MeanDuration', 'MedDuration', 'MeanTimeBtw', 'MedTimeBtw',
          'MeanInt', 'MedianInt'

        For two-well summaries and base names, `two_well_mode` controls how A/B are combined:
        - 'A' / 'B': use the A or B column
        - 'total': sum A+B (best for Licks/Events)
        - 'mean_ab': mean of A and B (best for duration/interval/time-between metrics)
        """

        m = str(metric).strip()
        if m in binned_rows.columns:
            return pd.to_numeric(binned_rows[m], errors="coerce")

        base = m
        base_map = {
            "Licks": ("LicksA", "LicksB", "Licks"),
            "Events": ("EventsA", "EventsB", "Events"),
            "MeanDuration": ("MeanDurationA", "MeanDurationB", "MeanDuration"),
            "MedDuration": ("MedDurationA", "MedDurationB", "MedDuration"),
            "MeanTimeBtw": ("MeanTimeBtwA", "MeanTimeBtwB", "MeanTimeBtw"),
            "MedTimeBtw": ("MedTimeBtwA", "MedTimeBtwB", "MedTimeBtw"),
            "MeanInt": ("MeanIntA", "MeanIntB", "MeanInt"),
            "MedianInt": ("MedianIntA", "MedianIntB", "MedianInt"),
        }
        if base not in base_map:
            raise ValueError(
                f"Unknown metric {metric!r}. Provide an existing column name, or one of: {sorted(base_map.keys())}"
            )

        col_a, col_b, col_single = base_map[base]
        if col_a in binned_rows.columns and col_b in binned_rows.columns:
            a = pd.to_numeric(binned_rows[col_a], errors="coerce")
            b = pd.to_numeric(binned_rows[col_b], errors="coerce")
            mode = str(two_well_mode)
            if mode == "A":
                return a
            if mode == "B":
                return b
            if mode == "mean_ab":
                return (a + b) / 2.0
            # total
            return a.fillna(0.0) + b.fillna(0.0)

        if col_single in binned_rows.columns:
            return pd.to_numeric(binned_rows[col_single], errors="coerce")

        raise ValueError(
            f"Metric {metric!r} not available in binned summary columns. "
            f"Missing expected columns: {col_single!r} or ({col_a!r},{col_b!r})."
        )

    def plot_binned_metric_by_treatment(
        self,
        *,
        metric: str = "Licks",
        two_well_mode: Literal["total", "mean_ab", "A", "B"] = "total",
        binsize_min: float = 30.0,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool = True,
        show_sem: bool = True,
        show_individual_chambers: bool = False,
        figsize: tuple[float, float] = (10, 4),
    ) -> Any:
        """
        Plot mean ± SEM of a binned feeding-summary metric for each treatment across the experiment.

        - Uses the chamber→treatment assignments in `Experiment.design`.
        - Uses per-DFM `binned_feeding_summary()` and aggregates across chambers for each treatment.

        Returns a matplotlib Figure.
        """

        import matplotlib.pyplot as plt
        import numpy as np

        df = self._binned_licks_table_by_treatment(
            binsize_min=binsize_min, range_minutes=range_minutes, transform_licks=transform_licks
        )
        if df.empty:
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_title("Binned metric by treatment (no data)")
            return fig

        df = df.copy()
        df["MetricValue"] = self._metric_series_from_binned_rows(df, metric=metric, two_well_mode=two_well_mode)

        df = df.dropna(subset=["Minutes", "MetricValue"]).copy()
        df["Minutes"] = pd.to_numeric(df["Minutes"], errors="coerce")
        df["MetricValue"] = pd.to_numeric(df["MetricValue"], errors="coerce")
        df = df.dropna(subset=["Minutes", "MetricValue"])

        treatments = sorted(df["Treatment"].unique().tolist())

        agg = (
            df.groupby(["Treatment", "Minutes"], as_index=False)["MetricValue"]
            .agg(mean="mean", std="std", n="count")
            .rename(columns={"mean": "Mean", "std": "Std", "n": "N"})
        )
        agg["SEM"] = (agg["Std"] / np.sqrt(agg["N"].clip(lower=1))).fillna(0.0)

        fig, ax = plt.subplots(figsize=figsize)
        colors = [plt.cm.tab10(i % 10) for i in range(len(treatments))]

        if show_individual_chambers:
            for i, trt in enumerate(treatments):
                tmp = df[df["Treatment"] == trt]
                for (_, _), g in tmp.groupby(["DFM", "Chamber"]):
                    g = g.sort_values("Minutes")
                    ax.plot(
                        g["Minutes"],
                        g["MetricValue"],
                        color=colors[i],
                        alpha=0.18,
                        linewidth=0.8,
                    )

        for i, trt in enumerate(treatments):
            tmp = agg[agg["Treatment"] == trt].sort_values("Minutes")
            x = tmp["Minutes"].to_numpy(dtype=float)
            y = tmp["Mean"].to_numpy(dtype=float)
            e = tmp["SEM"].to_numpy(dtype=float)

            ax.plot(x, y, color=colors[i], linewidth=2.0, label=f"{trt} (n={int(tmp['N'].max())})")
            if show_sem:
                ax.fill_between(x, y - e, y + e, color=colors[i], alpha=0.2, linewidth=0)

        ax.set_xlabel("Minutes")
        ylabel = metric
        if metric in ("Licks", "Events") and two_well_mode == "total":
            ylabel = f"{metric} (total)"
        ax.set_ylabel(("Transformed " if transform_licks and metric == "Licks" else "") + ylabel)
        ax.set_title(f"Binned {metric} by treatment (mean ± SEM)")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        return fig

    def plot_binned_metrics_by_treatment(
        self,
        *,
        metrics: Sequence[str] = ("Licks", "Events", "MedDuration"),
        two_well_mode: Literal["total", "mean_ab", "A", "B"] = "total",
        binsize_min: float = 30.0,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool = True,
        show_sem: bool = True,
        show_individual_chambers: bool = False,
        ncols: int = 2,
        figsize: tuple[float, float] | None = None,
    ) -> Any:
        """
        Plot multiple binned treatment timecourses (mean ± SEM), one subplot per metric.
        """

        import math

        import matplotlib.pyplot as plt
        import numpy as np

        metrics = [str(m).strip() for m in metrics if str(m).strip()]
        if not metrics:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.set_title("No metrics specified")
            return fig

        df = self._binned_licks_table_by_treatment(
            binsize_min=binsize_min, range_minutes=range_minutes, transform_licks=transform_licks
        )
        if df.empty:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.set_title("Binned metrics by treatment (no data)")
            return fig

        df = df.copy()
        df["Minutes"] = pd.to_numeric(df["Minutes"], errors="coerce")
        df = df.dropna(subset=["Minutes"])
        treatments = sorted(df["Treatment"].unique().tolist())
        colors = [plt.cm.tab10(i % 10) for i in range(len(treatments))]

        nrows = math.ceil(len(metrics) / int(ncols))
        if figsize is None:
            figsize = (ncols * 6, nrows * 3.8)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
        axes_flat = np.array(axes).flatten()

        for i, metric in enumerate(metrics):
            ax = axes_flat[i]
            tmp = df.copy()
            tmp["MetricValue"] = self._metric_series_from_binned_rows(
                tmp, metric=metric, two_well_mode=two_well_mode
            )
            tmp["MetricValue"] = pd.to_numeric(tmp["MetricValue"], errors="coerce")
            tmp = tmp.dropna(subset=["MetricValue"])

            agg = (
                tmp.groupby(["Treatment", "Minutes"], as_index=False)["MetricValue"]
                .agg(mean="mean", std="std", n="count")
                .rename(columns={"mean": "Mean", "std": "Std", "n": "N"})
            )
            agg["SEM"] = (agg["Std"] / np.sqrt(agg["N"].clip(lower=1))).fillna(0.0)

            if show_individual_chambers:
                for t_i, trt in enumerate(treatments):
                    chamber_rows = tmp[tmp["Treatment"] == trt]
                    for (_dfm_id, _ch), g in chamber_rows.groupby(["DFM", "Chamber"]):
                        g = g.sort_values("Minutes")
                        ax.plot(
                            g["Minutes"],
                            g["MetricValue"],
                            color=colors[t_i],
                            alpha=0.18,
                            linewidth=0.8,
                        )

            for t_i, trt in enumerate(treatments):
                g = agg[agg["Treatment"] == trt].sort_values("Minutes")
                x = g["Minutes"].to_numpy(dtype=float)
                y = g["Mean"].to_numpy(dtype=float)
                e = g["SEM"].to_numpy(dtype=float)
                ax.plot(x, y, color=colors[t_i], linewidth=2.0, label=trt)
                if show_sem:
                    ax.fill_between(x, y - e, y + e, color=colors[t_i], alpha=0.2, linewidth=0)

            ylabel = metric
            if metric in ("Licks", "Events") and two_well_mode == "total":
                ylabel = f"{metric} (total)"
            ax.set_ylabel(("Transformed " if transform_licks and metric == "Licks" else "") + ylabel)
            ax.set_title(metric)
            ax.legend(fontsize=7, ncol=1)

        for j in range(len(metrics), len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle("Binned metrics by treatment (mean ± SEM)", fontsize=12)
        fig.tight_layout()
        return fig

    def plot_binned_licks_by_treatment(
        self,
        *,
        binsize_min: float = 30.0,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool = True,
        show_sem: bool = True,
        show_individual_chambers: bool = False,
        figsize: tuple[float, float] = (10, 4),
    ) -> Any:
        """
        Backward-compatible wrapper: plot binned licks by treatment (mean ± SEM).
        """

        return self.plot_binned_metric_by_treatment(
            metric="Licks",
            two_well_mode="total",
            binsize_min=binsize_min,
            range_minutes=range_minutes,
            transform_licks=transform_licks,
            show_sem=show_sem,
            show_individual_chambers=show_individual_chambers,
            figsize=figsize,
        )

    def _append_factor_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Insert per-factor columns after the Treatment column when design_factors is set."""
        if not self.design_factors or not self.chamber_factors or df.empty:
            return df
        if "DFM" not in df.columns or "Chamber" not in df.columns:
            return df
        df = df.copy()
        for factor in self.design_factors:
            df[factor] = [
                self.chamber_factors.get((int(row["DFM"]), int(row["Chamber"])), {}).get(factor, "")
                for _, row in df.iterrows()
            ]
        # Move factor columns to sit immediately after Treatment
        if "Treatment" in df.columns:
            cols = list(df.columns)
            trt_pos = cols.index("Treatment")
            for i, factor in enumerate(self.design_factors):
                cols.remove(factor)
                cols.insert(trt_pos + 1 + i, factor)
            df = df[cols]
        return df

    def feeding_summary(
        self,
        *,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool = True,
    ) -> pd.DataFrame:
        """
        Return the feeding summary for all treatment-assigned DFM chambers as a single DataFrame.

        Results are cached by ``(range_minutes, transform_licks)`` so repeated
        calls with identical arguments do not recompute the summary.  The full-
        range summary ``(0, 0)`` is pre-computed when the experiment is first
        loaded.
        """
        key = (float(range_minutes[0]), float(range_minutes[1])), bool(transform_licks)
        if key not in self._feeding_summary_cache:
            self._feeding_summary_cache[key] = self._append_factor_columns(
                self.design.feeding_summary(
                    range_minutes=range_minutes,
                    transform_licks=transform_licks,
                )
            )
        return self._feeding_summary_cache[key]

    def binned_feeding_summary(
        self,
        *,
        bins: Sequence[float] | None = None,
        binsize_min: float | None = None,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool = True,
        path: str | Path | None = None,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Return a stacked feeding summary across time bins with an ``Interval``
        and ``Minutes`` (bin midpoint) column prepended to each row.

        By default the result is also saved to
        ``project_dir/analysis/binned_feeding_summary.csv`` (or *path* if
        provided).  Pass ``save=False`` to suppress file output.

        Bins can be specified in one of two mutually exclusive ways:

        **Explicit bin edges** — pass a sequence of at least two monotonically
        increasing minute values as *bins*.  Each consecutive pair defines one
        bin::

            exp.binned_feeding_summary(bins=(0, 10, 50, 100, 200))
            # → bins (0,10], (10,50], (50,100], (100,200]

        **Regular bin size** — pass *binsize_min* and optionally *range_minutes*
        to divide a time range into equal-width windows::

            exp.binned_feeding_summary(binsize_min=30, range_minutes=(0, 120))
            # → bins (0,30], (30,60], (60,90], (90,120]

        When ``range_minutes=(0, 0)`` (the default) with *binsize_min*, the
        upper bound is taken from the maximum ``Minutes`` value across all
        loaded DFMs.

        Parameters
        ----------
        bins:
            Explicit, strictly increasing bin edges (at least 2 values).
            Mutually exclusive with *binsize_min*.
        binsize_min:
            Width of each bin in minutes.  Mutually exclusive with *bins*.
        range_minutes:
            ``(start, end)`` window used only with *binsize_min*.
            ``(0, 0)`` means from 0 to the end of the experiment.
        transform_licks:
            Whether to apply the 0.25-power lick transformation (default
            ``True``).
        path:
            Explicit output path for the CSV.  When ``None`` (the default)
            the file is written to
            ``project_dir/analysis/binned_feeding_summary.csv``.
            Ignored when ``save=False``.
        save:
            Write the result to a CSV file (default ``True``).  Set to
            ``False`` to return the DataFrame without touching the filesystem.
        """
        import numpy as np

        if bins is not None and binsize_min is not None:
            raise ValueError("Specify either 'bins' or 'binsize_min', not both.")
        if bins is None and binsize_min is None:
            raise ValueError("Specify either 'bins' or 'binsize_min'.")

        if bins is not None:
            edges = [float(v) for v in bins]
            if len(edges) < 2:
                raise ValueError("'bins' must contain at least two values.")
            if edges != sorted(edges) or len(edges) != len(set(edges)):
                raise ValueError("'bins' must be strictly increasing.")
            bin_pairs = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
        else:
            bsize = float(binsize_min)
            if bsize <= 0:
                raise ValueError("'binsize_min' must be positive.")
            if float(range_minutes[0]) == 0.0 and float(range_minutes[1]) == 0.0:
                m_min = 0.0
                m_max = max(
                    float(dfm.raw_df["Minutes"].max()) for dfm in self.dfms.values()
                )
            else:
                m_min, m_max = float(range_minutes[0]), float(range_minutes[1])
            if m_min >= m_max:
                raise ValueError(
                    f"range_minutes start ({m_min}) must be less than end ({m_max})."
                )
            y = np.arange(m_min, m_max + 1e-9, bsize, dtype=float)
            if y.size == 0 or y[-1] < m_max:
                y = np.append(y, m_max)
            bin_pairs = [(float(y[i]), float(y[i + 1])) for i in range(len(y) - 1)]

        parts: list[pd.DataFrame] = []
        for a, b in bin_pairs:
            label = f"({_fmt_min(a)},{_fmt_min(b)}]"
            midpoint = (a + b) / 2.0
            summ = self.feeding_summary(
                range_minutes=(a, b), transform_licks=transform_licks
            )
            if summ.empty:
                continue
            summ = summ.copy()
            summ.insert(0, "Minutes", midpoint)
            summ.insert(0, "Interval", label)
            parts.append(summ)

        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

        if save:
            if path is None:
                if self.analysis_dir is None:
                    raise ValueError(
                        "path must be provided when no project_dir is set on the Experiment."
                    )
                path = self.analysis_dir / "binned_feeding_summary.csv"
            out = Path(path).expanduser().resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)

        return df

    def write_feeding_summary(
        self,
        path: str | Path | None = None,
        *,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool = True,
    ) -> Path:
        """
        Save the feeding summary CSV to disk and return the resolved output path.

        If *path* is not given, defaults to
        ``project_dir/analysis/feeding_summary.csv`` for the full range, or
        ``project_dir/analysis/feeding_summary_{a}_{b}.csv`` when
        ``range_minutes=(a, b)`` is specified.
        Raises ``ValueError`` if neither *path* nor ``project_dir`` is set.
        Only includes DFM chambers assigned to a treatment.
        """
        if path is None:
            if self.analysis_dir is None:
                raise ValueError(
                    "path must be provided when no project_dir is set on the Experiment."
                )
            a, b = float(range_minutes[0]), float(range_minutes[1])
            if a == 0.0 and b == 0.0:
                stem = "feeding_summary"
            else:
                actual_max = self._max_experiment_minutes()
                b_name = b
                if actual_max is not None and (b == float("inf") or b > actual_max):
                    b_name = actual_max
                stem = f"feeding_summary_{_fmt_min(a)}_{_fmt_min(b_name)}"
            path = self.analysis_dir / f"{stem}.csv"
        out = Path(path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        df = self.feeding_summary(range_minutes=range_minutes, transform_licks=transform_licks)
        df.to_csv(out, index=False)
        return out

    def write_parsed_feeding_summary(
        self,
        breakpoints: Sequence[float],
        *,
        transform_licks: bool = True,
    ) -> list[Path]:
        """
        Write one feeding-summary CSV per time segment defined by *breakpoints*.

        *breakpoints* is a sorted sequence of minute values that partition the
        experiment into consecutive, non-overlapping windows.  For example::

            exp.write_parsed_feeding_summary((100, 400, 1000))

        produces four files:

        * ``feeding_summary_0_100.csv``   — minutes   0 → 100
        * ``feeding_summary_100_400.csv`` — minutes 100 → 400
        * ``feeding_summary_400_1000.csv``— minutes 400 → 1000
        * ``feeding_summary_1000_end.csv``— minutes 1000 → end of experiment

        All files are written to ``project_dir/analysis/`` using the same
        naming convention as :meth:`write_feeding_summary`.  Raises
        :class:`ValueError` if *breakpoints* is empty, contains non-positive
        values, or is not strictly increasing.

        Returns a list of the written file paths in segment order.
        """
        bps = [float(v) for v in breakpoints]
        if not bps:
            raise ValueError("breakpoints must contain at least one value.")
        if any(v <= 0 for v in bps):
            raise ValueError("All breakpoints must be positive (greater than 0).")
        if bps != sorted(bps) or len(bps) != len(set(bps)):
            raise ValueError("breakpoints must be strictly increasing.")

        edges = [0.0, *bps, float("inf")]
        ranges = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

        paths: list[Path] = []
        for rng in ranges:
            paths.append(self.write_feeding_summary(range_minutes=rng, transform_licks=transform_licks))
        return paths

    def write_summary(
        self,
        path: str | Path | None = None,
        *,
        include_qc: bool = True,
        qc_data_breaks_multiplier: float = 4.0,
        qc_bleeding_cutoff: float = 50.0,
    ) -> Path:
        """
        Write ``summary_text()`` to disk and return the resolved output path.

        If *path* is not given, defaults to ``project_dir/analysis/summary.txt``.
        Raises ``ValueError`` if neither *path* nor ``project_dir`` is set.
        """

        if path is None:
            if self.analysis_dir is None:
                raise ValueError(
                    "path must be provided when no project_dir is set on the Experiment."
                )
            path = self.analysis_dir / "summary.txt"
        out = Path(path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            self.summary_text(
                include_qc=include_qc,
                qc_data_breaks_multiplier=qc_data_breaks_multiplier,
                qc_bleeding_cutoff=qc_bleeding_cutoff,
            ),
            encoding="utf-8",
        )
        return out

    def write_feeding_summary_plot(
        self,
        path: str | Path | None = None,
        *,
        format: str = "png",
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool = True,
        ncols: int | None = None,
        figsize: tuple[float, float] | None = None,
        dpi: int = 150,
    ) -> Path:
        """
        Save the feeding summary plot to disk and return the resolved output path.

        If *path* is not given, defaults to
        ``project_dir/analysis/feeding_summary.{format}``.
        Raises ``ValueError`` if neither *path* nor ``project_dir`` is set.
        *format* may be ``"png"`` (default) or ``"pdf"``.
        """
        if path is None:
            if self.analysis_dir is None:
                raise ValueError(
                    "path must be provided when no project_dir is set on the Experiment."
                )
            path = self.analysis_dir / f"feeding_summary.{format}"
        out = Path(path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        p = self.plot_feeding_summary(
            range_minutes=range_minutes,
            transform_licks=transform_licks,
            ncols=ncols,
            figsize=figsize,
        )
        p.save(str(out), dpi=dpi)
        return out

    def execute_basic_analysis(
        self,
        *,
        data_breaks_multiplier: float = 4.0,
        bleeding_cutoff: float = 50.0,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool = True,
        plot_format: str = "png",
        dpi: int = 150,
    ) -> dict[str, Path]:
        """
        Run the standard analysis pipeline and write all outputs to disk.

        Calls, in order:
          1. ``write_qc_reports()``           → ``project_dir/qc/``
          2. ``write_summary()``              → ``project_dir/analysis/summary.txt``
          3. ``write_feeding_summary()``      → ``project_dir/analysis/feeding_summary.csv``
          4. ``write_feeding_summary_plot()`` → ``project_dir/analysis/feeding_summary.{plot_format}``

        Returns a dict with keys ``"qc_dir"``, ``"summary"``,
        ``"feeding_summary"`` and ``"feeding_summary_plot"`` pointing to the
        written paths.
        """
        n_dfms = len(self.dfms)
        print("=" * 50, flush=True)
        print("FLIC Basic Analysis", flush=True)
        print(f"  Project : {self.project_dir}", flush=True)
        print(f"  DFMs    : {sorted(self.dfms.keys())}", flush=True)
        print("=" * 50, flush=True)

        print("\n[1/4] QC reports...", flush=True)
        qc_dir = self.write_qc_reports(
            data_breaks_multiplier=data_breaks_multiplier,
            bleeding_cutoff=bleeding_cutoff,
        )
        print(f"  Done — {n_dfms} DFM(s) → {qc_dir}", flush=True)

        print("\n[2/4] Experiment summary...", flush=True)
        summary_path = self.write_summary(
            include_qc=True,
            qc_data_breaks_multiplier=data_breaks_multiplier,
            qc_bleeding_cutoff=bleeding_cutoff,
        )
        print(f"  Done → {summary_path}", flush=True)

        print("\n[3/4] Feeding summary CSV...", flush=True)
        feeding_csv_path = self.write_feeding_summary(
            range_minutes=range_minutes,
            transform_licks=transform_licks,
        )
        print(f"  Done → {feeding_csv_path}", flush=True)

        print("\n[4/4] Feeding summary plot...", flush=True)
        plot_path = self.write_feeding_summary_plot(
            format=plot_format,
            range_minutes=range_minutes,
            transform_licks=transform_licks,
            dpi=dpi,
        )
        print(f"  Done → {plot_path}", flush=True)

        print("\n" + "=" * 50, flush=True)
        print("Analysis complete.", flush=True)
        print("=" * 50, flush=True)

        return {
            "qc_dir": qc_dir,
            "summary": summary_path,
            "feeding_summary": feeding_csv_path,
            "feeding_summary_plot": plot_path,
        }

