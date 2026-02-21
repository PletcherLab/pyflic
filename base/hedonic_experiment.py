from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence
import pandas as pd

from .experiment import Experiment


@dataclass(slots=True)
class HedonicFeedingExperiment(Experiment):
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
        x_labels: dict[str, str] | None = None,
        annotation: str | None = None,
        annotation_x: float = 1.0,
        annotation_y: float = 0.0,
        jitter_width: float = 0.25,
        point_size: float = 3.0,
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

        Parameters
        ----------
        x_labels : dict[str, str] | None
            Optional mapping from well key to display label, e.g.
            ``{"WellA": "Sucrose", "WellB": "Yeast"}``.  Keys are
            case-insensitive (``"wella"`` and ``"WellA"`` both work).
            When omitted the tick labels default to ``"WellA"`` / ``"WellB"``.
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

        if col_a not in fs.columns or col_b not in fs.columns:
            raise ValueError(
                f"Feeding summary does not contain '{col_a}' and '{col_b}'. "
                f"Available columns: {list(fs.columns)}"
            )

        keep = [c for c in ("DFM", "Chamber", "Treatment") if c in fs.columns]
        df_a = fs[keep + [col_a]].copy()
        df_a["Well"] = "WellA"
        df_a = df_a.rename(columns={col_a: metric})

        df_b = fs[keep + [col_b]].copy()
        df_b["Well"] = "WellB"
        df_b = df_b.rename(columns={col_b: metric})

        df_long = pd.concat([df_a, df_b], ignore_index=True)

        # Normalise x_labels keys to title-case so "wella"/"WELLA"/"WellA" all work.
        norm_labels: dict[str, str] | None = None
        if x_labels is not None:
            norm_labels = {k.lower().replace(" ", ""): v for k, v in x_labels.items()}
            norm_labels = {
                "WellA": norm_labels.get("wella", "WellA"),
                "WellB": norm_labels.get("wellb", "WellB"),
            }

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
            x_labels=norm_labels,
            colors={"WellA": "steelblue", "WellB": "tomato"},
            annotation=annotation,
            annotation_x=annotation_x,
            annotation_y=annotation_y,
            jitter_width=jitter_width,
            point_size=point_size,
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


    
    def execute_basic_analysis(self) -> None:
        super().execute_basic_analysis()
        self.hedonic_feeding_plot(save=True)

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
