from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from .dfm import DFM
from .two_well_experiment import TwoWellExperiment

# Port of the configuration comments in breaking_point.R:
#   1 = training well lights on wells 1, 5, 9  (used as end-of-training reference for wells 1-4, 5-8, 9-12)
#   2 = training well lights on wells 3, 7, 11
#   3 = training well lights on wells 2, 6, 10
#   4 = training well lights on wells 4, 8, 12
_BREAKING_POINT_CONFIGS: dict[int, tuple[str, str, str]] = {
    1: ("W1",  "W5",  "W9"),
    2: ("W3",  "W7",  "W11"),
    3: ("W2",  "W6",  "W10"),
    4: ("W4",  "W8",  "W12"),
}


def _find_transition_rows(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return only the rows where *column* (bool) changes value from the previous row.

    Port of R ``find_transition_rows(data, column_name)``.
    """
    if df.empty:
        # Avoid length mismatches (R's logic effectively returns empty here).
        return df.copy()
    vals = df[column].to_numpy(dtype=bool)
    transitions = np.concatenate([[False], vals[1:] != vals[:-1]])
    return df[transitions]


@dataclass(slots=True)
class ProgressiveRatioExperiment(TwoWellExperiment):
    """
    A progressive-ratio specialisation of :class:`TwoWellExperiment`.

    All DFMs in the config must use ``chamber_size=2``.
    Set ``experiment_type: progressive_ratio`` in ``flic_config.yaml`` under
    the ``global:`` section to have :func:`load_experiment_yaml` return this
    class automatically.
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
    ) -> ProgressiveRatioExperiment:
        """Load a progressive-ratio experiment, validating that every DFM uses ``chamber_size=2``."""
        from .yaml_config import load_experiment_yaml

        base = load_experiment_yaml(
            project_dir,
            range_minutes=range_minutes,
            parallel=parallel,
            max_workers=max_workers,
            executor=executor,
        )
        bad = [
            dfm_id
            for dfm_id, dfm in base.dfms.items()
            if dfm.params.chamber_size != 2
        ]
        if bad:
            raise ValueError(
                f"ProgressiveRatioExperiment requires chamber_size=2 for every DFM, "
                f"but DFM(s) {sorted(bad)} have chamber_size != 2.  "
                f"Set chamber_size: 2 in flic_config.yaml."
            )
        return cls(**{f.name: getattr(base, f.name) for f in dataclasses.fields(base)})

    # ------------------------------------------------------------------
    # Breaking-point analysis  (port of breaking_point.R)
    # ------------------------------------------------------------------

    def breaking_point_well(
        self,
        dfm: DFM,
        well: int,
        *,
        end_training: float | None = None,
    ) -> pd.DataFrame:
        """Per-well breaking-point analysis.  Port of R ``breaking.test()``.

        Parameters
        ----------
        dfm :
            The DFM to analyse.
        well :
            Well number (1–12).
        end_training :
            Minute at which training ended.  If ``None`` the value is read
            from ``dfm.in_training_data`` when available; otherwise it falls
            back to ``0.0``. Samples with ``Minutes == end_training`` are excluded.
            Pass ``0.0`` to skip the training filter.

        Returns
        -------
        DataFrame
            Columns: ``Minutes``, ``Lights``, ``CumLicks``,
            ``DeltaMinutes``, ``DeltaLicks``.
            Contains one row per lights-on period (after training),
            at the moment the lights switch on.
        """
        if not (1 <= int(well) <= 12):
            raise ValueError(f"well must be an integer in 1..12, got {well!r}")

        cname = f"W{int(well)}"

        if dfm.lights_df is None or dfm.lick_df is None:
            raise ValueError(
                f"DFM {dfm.id} must have computed lights/lick data before breaking-point analysis."
            )
        if "Minutes" not in dfm.lights_df.columns:
            raise ValueError(f"DFM {dfm.id} lights_df is missing required column 'Minutes'.")
        if cname not in dfm.lights_df.columns:
            raise ValueError(f"DFM {dfm.id} lights_df is missing required well column {cname!r}.")
        if cname not in dfm.lick_df.columns:
            raise ValueError(f"DFM {dfm.id} lick_df is missing required well column {cname!r}.")

        cum_licks = dfm.lick_df[cname].to_numpy(dtype=float).cumsum()

        result = dfm.lights_df[["Minutes", cname]].copy().rename(columns={cname: "Lights"})
        result["CumLicks"] = cum_licks

        if end_training is None:
            itd = dfm.in_training_data
            if itd is not None:
                row = itd[itd["well"] == cname]
                val = float(row["Minutes"].iloc[0]) if len(row) > 0 else 0.0
            else:
                val = 0.0
            end_training = val if np.isfinite(val) else 0.0

        result = result[result["Minutes"] > end_training].copy()
        if result.empty:
            return pd.DataFrame(
                columns=["Minutes", "Lights", "CumLicks", "DeltaMinutes", "DeltaLicks"]
            )
        result = _find_transition_rows(result, "Lights")
        if result.empty:
            return pd.DataFrame(
                columns=["Minutes", "Lights", "CumLicks", "DeltaMinutes", "DeltaLicks"]
            )

        result = result.copy()
        result["DeltaMinutes"] = np.concatenate(
            [[0.0], np.diff(result["Minutes"].to_numpy())]
        )
        result["DeltaLicks"] = np.concatenate(
            [[0.0], np.diff(result["CumLicks"].to_numpy())]
        )

        result = result[result["Lights"]].copy()
        # Keep output column order stable (matches R output).
        result = result[["Minutes", "Lights", "CumLicks", "DeltaMinutes", "DeltaLicks"]]
        return result.reset_index(drop=True)

    def breaking_point_dfm(
        self,
        dfm: DFM,
        configuration: int,
    ) -> dict[str, pd.DataFrame]:
        """Per-DFM breaking-point analysis across all 12 wells.
        Port of R ``breaking.test.dfm()``.

        Parameters
        ----------
        dfm :
            The DFM to analyse.
        configuration : {1, 2, 3, 4}
            Which well carried the progressive-ratio training signal:

            ============  ===============================
            configuration  training well (by group)
            ============  ===============================
            1              W1 (wells 1–4), W5 (5–8), W9 (9–12)
            2              W3 (wells 1–4), W7 (5–8), W11 (9–12)
            3              W2 (wells 1–4), W6 (5–8), W10 (9–12)
            4              W4 (wells 1–4), W8 (5–8), W12 (9–12)
            ============  ===============================

        Returns
        -------
        dict
            Maps well name (``"W1"`` … ``"W12"``) to the per-well
            breaking-point DataFrame produced by :meth:`breaking_point_well`.
        """
        if configuration not in _BREAKING_POINT_CONFIGS:
            raise ValueError(f"configuration must be 1–4, got {configuration!r}")

        ref_wells = _BREAKING_POINT_CONFIGS[configuration]
        itd = dfm.in_training_data

        def _end_training(ref: str) -> float:
            if itd is None:
                return 0.0
            row = itd[itd["well"] == ref]
            val = float(row["Minutes"].iloc[0]) if len(row) > 0 else 0.0
            return val if np.isfinite(val) else 0.0

        group_ends = [_end_training(ref_wells[0]),
                      _end_training(ref_wells[1]),
                      _end_training(ref_wells[2])]

        out: dict[str, pd.DataFrame] = {}
        for i in range(1, 13):
            if i <= 4:
                end_t = group_ends[0]
            elif i <= 8:
                end_t = group_ends[1]
            else:
                end_t = group_ends[2]
            out[f"W{i}"] = self.breaking_point_well(dfm, i, end_training=end_t)
        return out

    def breaking_point_summary(
        self,
        configuration: int,
    ) -> dict[int, dict[str, pd.DataFrame]]:
        """Run :meth:`breaking_point_dfm` for every DFM in the experiment.

        Returns
        -------
        dict
            Maps DFM id → per-well breaking-point dict
            (see :meth:`breaking_point_dfm`).
        """
        return {
            int(dfm_id): self.breaking_point_dfm(dfm, configuration)
            for dfm_id, dfm in self.dfms.items()
        }

    def plot_breaking_point_dfm(
        self,
        dfm: DFM,
        configuration: int,
        *,
        ylim: tuple[float, float] | None = (0, 500),
        ncols: int = 4,
    ):
        """Plot ``DeltaLicks`` vs ``Minutes`` for every well of *dfm* using matplotlib.
        Port of the per-well plotting loop at the bottom of ``breaking_point.R``.

        Parameters
        ----------
        dfm :
            The DFM to plot.
        configuration :
            Passed directly to :meth:`breaking_point_dfm`.
        ylim :
            Y-axis limits.  ``None`` uses matplotlib auto-scaling.
        ncols :
            Number of columns in the facet grid.
        """
        import matplotlib.pyplot as plt

        data = self.breaking_point_dfm(dfm, configuration)
        nwells = len(data)
        nrows = (nwells + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
        axes_flat = np.array(axes).flatten()

        for ax, (name, df) in zip(axes_flat, data.items()):
            if len(df) > 0:
                ax.plot(df["Minutes"], df["DeltaLicks"], marker="o", linewidth=1)
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.set_title(f"DFM {dfm.id} – {name}")
            ax.set_xlabel("Minutes")
            ax.set_ylabel("ΔLicks")

        for ax in axes_flat[nwells:]:
            ax.set_visible(False)

        fig.tight_layout()
        return fig

    def plot_breaking_point_dfm_gg(
        self,
        dfm: DFM,
        configuration: int,
        *,
        ylim: tuple[float, float] | None = (0, 500),
        ncols: int = 4,
        point_size: float = 2.0,
        line_size: float = 0.6,
        base_font_size: float = 10.0,
        figsize: tuple[float, float] | None = None,
    ):
        """Plot ``DeltaLicks`` vs ``Minutes`` for every well of *dfm* using plotnine.

        Produces a faceted line+point plot with one panel per well.

        Parameters
        ----------
        dfm :
            The DFM to plot.
        configuration :
            Passed directly to :meth:`breaking_point_dfm`.
        ylim :
            Y-axis limits passed to ``coord_cartesian``.  ``None`` uses auto-scaling.
        ncols :
            Number of columns in the facet grid.
        point_size, line_size :
            Sizes for ``geom_point`` and ``geom_line``.
        base_font_size :
            Base font size for ``theme_bw``.
        figsize :
            ``(width, height)`` in inches.  Defaults to ``(ncols * 3.5, nrows * 3)``.
        """
        from plotnine import (
            aes,
            coord_cartesian,
            element_text,
            facet_wrap,
            geom_line,
            geom_point,
            ggplot,
            labs,
            theme,
            theme_bw,
        )

        data = self.breaking_point_dfm(dfm, configuration)

        frames = []
        for well_name, df in data.items():
            if len(df) > 0:
                tmp = df[["Minutes", "DeltaLicks"]].copy()
                tmp["Well"] = well_name
                frames.append(tmp)

        if not frames:
            return ggplot() + labs(title=f"DFM {dfm.id} – no breaking-point data")

        df_all = pd.concat(frames, ignore_index=True)
        df_all["Well"] = pd.Categorical(
            df_all["Well"],
            categories=[f"W{i}" for i in range(1, 13)],
            ordered=True,
        )

        nrows = (len(data) + ncols - 1) // ncols
        if figsize is None:
            figsize = (ncols * 3.5, nrows * 3.0)

        p = (
            ggplot(df_all, aes(x="Minutes", y="DeltaLicks"))
            + geom_line(size=line_size, color="#2166ac")
            + geom_point(size=point_size, color="#2166ac")
            + facet_wrap("~ Well", ncol=ncols)
            + labs(
                title=f"DFM {dfm.id} – Breaking Point (config {configuration})",
                x="Minutes",
                y="ΔLicks",
            )
            + theme_bw(base_size=base_font_size)
            + theme(
                strip_text=element_text(size=base_font_size),
                figure_size=figsize,
            )
        )

        if ylim is not None:
            p = p + coord_cartesian(ylim=ylim)

        return p
