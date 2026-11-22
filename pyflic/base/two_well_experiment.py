from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from .dfm import DFM
from .experiment import Experiment

#: The simultaneous-feeding columns also reported before the dual-feeding
#: correction, under :data:`PRE_SUFFIX` (decision 23 of ``docs/v4-data-plan.md``).
SIMULTANEOUS_PRE_COLUMNS: tuple[str, ...] = ("Both", "MaxMinSignalAtBoth", "HigherInCol1AtBoth")
PRE_SUFFIX = "_pre"


@dataclass(slots=True)
class TwoWellExperiment(Experiment):
    """
    A two-well (choice) specialisation of :class:`Experiment`.

    All DFMs in the config must use ``chamber_size=2``.
    Enforced at load time; raises :class:`ValueError` otherwise.

    Adds two-well-specific QC (simultaneous feeding matrix, bleeding check),
    plotting utilities (:meth:`facet_plot_well_durations`), and a richer
    feeding-summary plot metric list.
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
    ) -> TwoWellExperiment:
        """Load a two-well experiment, validating that every DFM uses ``chamber_size=2``."""
        from .yaml_config import load_experiment_yaml

        base = load_experiment_yaml(
            experiment_dir,
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
                f"TwoWellExperiment requires chamber_size=2 for every DFM, "
                f"but DFM(s) {sorted(bad)} have chamber_size != 2.  "
                f"Set chamber_size: 2 in flic_config.yaml."
            )
        return cls(**{f.name: getattr(base, f.name) for f in dataclasses.fields(base)})

    # ------------------------------------------------------------------
    # QC — adds simultaneous feeding matrix and bleeding check
    # ------------------------------------------------------------------

    def _simultaneous_feeding_matrix_df(self, dfm: DFM) -> pd.DataFrame:
        """Per chamber, the samples licked at each well and at both at once,
        after the dual-feeding correction (the licks the analysis counts),
        then the at-both columns before it, from ``dfm.lick_matrix``.  With
        the correction off the two agree; with it on, the ``_pre`` columns
        are the cross-talk the correction removed."""
        mat = dfm.simultaneous_feeding_matrix()
        n = int(dfm.params.chamber_sets.shape[0])
        frame = pd.DataFrame(
            mat,
            index=[f"Chamber{i}" for i in range(1, n + 1)],
            columns=["Licks1", "Licks2", "Both", "MaxMinSignalAtBoth", "HigherInCol1AtBoth"],
        )
        pre = dfm.lick_matrix
        for k, column in enumerate(SIMULTANEOUS_PRE_COLUMNS, start=2):
            frame[column + PRE_SUFFIX] = (np.nan if pre is None
                                          else np.asarray(pre, dtype=float)[:n, k])
        return frame

    def compute_qc_results(
        self,
        *,
        data_breaks_multiplier: float = 4.0,
        bleeding_cutoff: float = 50.0,
        include_integrity_text: bool = False,
    ) -> dict[int, dict[str, Any]]:
        results = Experiment.compute_qc_results(
            self,
            data_breaks_multiplier=data_breaks_multiplier,
            bleeding_cutoff=bleeding_cutoff,
            include_integrity_text=include_integrity_text,
        )
        for dfm_id, dfm in self.dfms.items():
            results[dfm_id]["simultaneous_feeding_matrix"] = self._simultaneous_feeding_matrix_df(dfm)
            results[dfm_id]["bleeding"] = dfm.bleeding_check(cutoff=float(bleeding_cutoff))
        self.qc_results = results
        return results

    # ------------------------------------------------------------------
    # Two-well-specific plotting
    # ------------------------------------------------------------------

    def facet_plot_well_durations(
        self,
        *,
        metric: str = "MedDuration",
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool = True,
        title: str = "",
        y_label: str | None = None,
        ylim: tuple[float, float] | None = None,
        x_labels: dict[str, str] | None = None,
        annotation: str | None = None,
        jitter_width: float = 0.25,
        point_size: float = 3.0,
        base_font_size: float = 20.0,
    ):
        """
        Jitter plot comparing WellA vs WellB for a given feeding metric, faceted by treatment.

        Parameters
        ----------
        metric : str
            Base column name without the A/B suffix (e.g. ``"MedDuration"`` looks for
            ``"MedDurationA"`` and ``"MedDurationB"`` in the feeding summary).
        x_labels : dict[str, str] | None
            Optional mapping from ``"WellA"`` / ``"WellB"`` to descriptive food names,
            e.g. ``{"WellA": "Sucrose", "WellB": "Yeast"}``.  Keys are case-insensitive.
        """
        col_a = f"{metric}A"
        col_b = f"{metric}B"

        df = self.feeding_summary(range_minutes=range_minutes, transform_licks=transform_licks)

        missing = [c for c in (col_a, col_b) if c not in df.columns]
        if missing:
            raise ValueError(
                f"facet_plot_well_durations: columns {missing} not found in feeding summary. "
                f"Available columns: {list(df.columns)}"
            )

        df, group_col = self._resolve_group_col(df)
        id_cols = [c for c in dict.fromkeys([group_col, "DFM", "Chamber"]) if c in df.columns]
        df_long = df[id_cols + [col_a, col_b]].melt(
            id_vars=id_cols,
            value_vars=[col_a, col_b],
            var_name="_WellCol",
            value_name=metric,
        )
        df_long["Well"] = df_long["_WellCol"].map({col_a: "WellA", col_b: "WellB"})
        df_long = df_long.drop(columns=["_WellCol"])

        normalised_labels: dict[str, str] | None = None
        if x_labels:
            normalised_labels = {}
            for k, v in x_labels.items():
                key = k.strip().lower().replace(" ", "")
                if key in ("wella", "a"):
                    normalised_labels["WellA"] = v
                elif key in ("wellb", "b"):
                    normalised_labels["WellB"] = v

        return self.plot_jitter_summary(
            df_long,
            x_col="Well",
            y_col=metric,
            facet_col=group_col,
            title=title,
            y_label=y_label or metric,
            ylim=ylim,
            x_order=["WellA", "WellB"],
            x_labels=normalised_labels,
            annotation=annotation,
            jitter_width=jitter_width,
            point_size=point_size,
            base_font_size=base_font_size,
        )

    # ------------------------------------------------------------------
    # Auto-removal: the cutoffs every two-well type applies
    # ------------------------------------------------------------------

    def two_well_removal_reasons(self, row) -> list[str]:
        """Why the feeding-summary *row* fails the two-well cutoffs —
        ``max_med_duration_cutoff`` on ``MedDurationA``/``B`` and
        ``max_events_cutoff`` on ``EventsA``/``B``, from ``global_constants``
        — or ``[]``.  Every two-well type applies them (ADR-0019); an unset
        cutoff is off."""
        constants = self.global_constants or {}
        reasons: list[str] = []
        for key, columns in (("max_med_duration_cutoff", ("MedDurationA", "MedDurationB")),
                             ("max_events_cutoff", ("EventsA", "EventsB"))):
            cutoff = constants.get(key)
            if cutoff is None:
                continue
            for col in columns:
                value = row.get(col) if hasattr(row, "get") else None
                if value is not None and pd.notna(value) and float(value) > float(cutoff):
                    reasons.append(f"{col}={float(value):.6g} > {key}={cutoff}")
        return reasons

    def two_well_criteria_lines(self) -> list[str]:
        """The two-well cutoffs as ``summary.txt``'s filter section states them."""
        constants = self.global_constants or {}
        max_dur = constants.get("max_med_duration_cutoff")
        max_events = constants.get("max_events_cutoff")
        return [
            f"  • max_med_duration_cutoff = {float(max_dur):g}: excluded if "
            f"MedDurationA or MedDurationB > {float(max_dur):g}"
            if max_dur is not None else "  • max_med_duration_cutoff: not configured.",
            f"  • max_events_cutoff = {float(max_events):g}: excluded if "
            f"EventsA or EventsB > {float(max_events):g}"
            if max_events is not None else "  • max_events_cutoff: not configured.",
        ]

    def auto_remove_chambers(
        self,
        *,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool = True,
        min_untransformed_licks_cutoff: float | None = None,
    ) -> pd.DataFrame:
        """The general lick filter (:meth:`Experiment.auto_remove_chambers`),
        then the two-well cutoffs (:meth:`two_well_removal_reasons`) on what
        it kept.  With neither cutoff set — a Custom two-well design that
        states none — this is the general filter alone."""
        base = Experiment.auto_remove_chambers(
            self, range_minutes=range_minutes,
            min_untransformed_licks_cutoff=min_untransformed_licks_cutoff)
        constants = self.global_constants or {}
        if constants.get("max_med_duration_cutoff") is None \
                and constants.get("max_events_cutoff") is None:
            return base
        fs = self.feeding_summary(range_minutes=range_minutes,
                                  transform_licks=transform_licks)
        already: set[tuple[int, int]] = set()
        if base is not None and not base.empty:
            already = {(int(r["DFM"]), int(r["Chamber"])) for _, r in base.iterrows()}
        to_remove: set[tuple[int, int]] = set()
        removed_rows: list[dict[str, object]] = []
        for _, row in fs.iterrows():
            key = (int(row["DFM"]), int(row["Chamber"]))
            if key in already or key in to_remove:
                continue
            reasons = self.two_well_removal_reasons(row)
            if reasons:
                to_remove.add(key)
                removed_rows.append({"DFM": key[0], "Chamber": key[1],
                                     "Treatment": str(row["Treatment"]),
                                     "Reason": "; ".join(reasons)})
                print(f"  [auto_remove_chambers] Removing DFM {key[0]} Chamber {key[1]}"
                      f" ({row['Treatment']}): {'; '.join(reasons)}", flush=True)
        if to_remove:
            self._remove_chambers_from_design(to_remove)
        extra = pd.DataFrame(removed_rows, columns=["DFM", "Chamber", "Treatment", "Reason"])
        if extra.empty:
            print("auto_remove_chambers (two-well cutoffs): no additional chambers removed.",
                  flush=True)
        else:
            print(f"auto_remove_chambers (two-well cutoffs): done — {len(extra)} additional "
                  f"chamber(s) removed.", flush=True)
        if base is None or base.empty:
            combined = extra
        elif extra.empty:
            combined = base
        else:
            combined = pd.concat([base, extra], ignore_index=True)
        self.filtered_chambers = combined
        lines = ["", "Two-well filter (additional criteria)", "",
                 *self.two_well_criteria_lines()]
        self.filter_criteria_summary = self.filter_criteria_summary + "\n" + "\n".join(lines)
        self.write_removed_chambers()
        return combined

    # ------------------------------------------------------------------
    # Feeding summary plot
    # ------------------------------------------------------------------

    def _feeding_plot_metrics(self) -> list[tuple[str, str]]:
        wn = self.well_names or {}
        na = wn.get("A", "A")
        nb = wn.get("B", "B")
        return [
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
            ## Intensity of the baselined signal within events, not a time.
            ("MeanIntA", f"Mean Intensity {na}"),
            ("MeanIntB", f"Mean Intensity {nb}"),
            ("MedianIntA", f"Median Intensity {na}"),
            ("MedianIntB", f"Median Intensity {nb}"),
        ]

    def _feeding_summary_default_ncols(self) -> int:
        return 4
