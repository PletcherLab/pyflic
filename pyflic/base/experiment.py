from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from .dfm import DFM
from .experiment_design import ExperimentDesign

# Okabe-Ito color-blind-safe palette (8 colors).
_OKABE_ITO = [
    "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00",
    "#CC79A7", "#999999",
]


def _fmt_min(v: float) -> str:
    """Format a minute value for use in a filename.  ``inf`` → ``'end'``."""
    if v == float("inf"):
        return "end"
    return str(int(v))


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
    experiment_dir: Path | None = None
    range_minutes: tuple[float, float] | None = None
    facet_cutoffs: tuple[float, ...] | None = None
    experiment_type: Any = None            # ExperimentType strategy (ADR-0007)
    chamber_layout: str | None = None      # 'single_well' | 'two_well'
    config: dict = field(default_factory=dict)   # the resolved yaml
    transform_licks: bool = True
    parallel: bool | None = None
    executor: Literal["threads", "processes"] | None = None
    max_workers: int | None = None
    qc_report_dir: Path | None = None
    qc_results: dict[int, dict[str, Any]] | None = None
    _feeding_summary_cache: dict = field(default_factory=dict)
    filtered_chambers: pd.DataFrame | None = None
    filter_criteria_summary: str = field(default="")
    excluded_chambers: dict[int, list[int]] = field(default_factory=dict)
    exclusion_group: str | None = None

    def get_dfm(self, dfm_id: int) -> DFM | None:
        """Return the DFM with the given id, or None if it does not exist."""
        return self.dfms.get(int(dfm_id))

    def _remove_chambers_from_design(self, remove_set: set[tuple[int, int]]) -> None:
        """Remove (DFM,Chamber) pairs from all treatments and clear caches."""
        if not remove_set:
            return
        for treatment in self.design.treatments.values():
            treatment.chambers = [
                tc
                for tc in treatment.chambers
                if (int(tc.dfm_id), int(tc.chamber_index)) not in remove_set
            ]
        self._feeding_summary_cache.clear()

    def write_removed_chambers(self) -> Path | None:
        """Write ``analysis/removed_chambers.csv`` — the auto-removal table the
        Project's aggregated exclusions read (Source = ``auto``).  Nothing is
        written when no experiment directory is set or nothing has been
        filtered yet."""
        if self.analysis_dir is None or self.filtered_chambers is None:
            return None
        out = self.analysis_dir / "removed_chambers.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        self.filtered_chambers.to_csv(out, index=False)
        return out

    def auto_remove_chambers(
        self,
        *,
        range_minutes: Sequence[float] = (0, 0),
        min_untransformed_licks_cutoff: float | None = None,
    ) -> pd.DataFrame:
        """
        Automatically remove chambers from the analysis design based on lick-count thresholds.

        This method updates ``self.design`` in-place by removing (DFM,Chamber) assignments from
        all treatments, then clears the feeding-summary cache so downstream summaries/plots
        reflect the filtered design.

        Threshold source
        ----------------
        ``min_untransformed_licks_cutoff`` is taken from:

        - the explicit argument if provided, otherwise
        - ``global.constants.min_untransformed_licks_cutoff`` from ``flic_config.yaml``.

        Filtering rules (all experiment types)
        ---------------------------------------
        A chamber is removed if **any** of the following apply to at least one
        well in the chamber:

        1. The lick value is ``NaN`` / ``None`` / undefined — the well produced
           no usable data regardless of any threshold.
        2. The lick value is below ``min_untransformed_licks_cutoff`` (when that
           threshold is configured).

        - chamber_size=1: uses ``Licks``
        - chamber_size=2: uses ``LicksA`` / ``LicksB`` (removes chamber if either fails)

        Returns
        -------
        pd.DataFrame
            Record of every removed chamber with columns ``DFM``, ``Chamber``,
            ``Treatment``, and ``Reason``.
        """
        constants = self.global_constants or {}
        if min_untransformed_licks_cutoff is None:
            raw = constants.get("min_untransformed_licks_cutoff")
            if raw is not None and str(raw).strip() != "":
                min_untransformed_licks_cutoff = float(raw)
        # None means "no cutoff configured" — NaN licks are still caught below.
        cutoff: float | None = (
            float(min_untransformed_licks_cutoff)
            if min_untransformed_licks_cutoff is not None
            else None
        )

        fs = self.feeding_summary(range_minutes=range_minutes, transform_licks=False)
        if fs.empty:
            result = pd.DataFrame(columns=["DFM", "Chamber", "Treatment", "Reason"])
            self.filtered_chambers = result
            return result

        # Identify which lick columns exist for the chamber size.
        has_single = "Licks" in fs.columns
        has_two = ("LicksA" in fs.columns) and ("LicksB" in fs.columns)
        if not (has_single or has_two):
            result = pd.DataFrame(columns=["DFM", "Chamber", "Treatment", "Reason"])
            self.filtered_chambers = result
            return result

        removed_rows: list[dict[str, object]] = []
        remove_set: set[tuple[int, int]] = set()

        for _, row in fs.iterrows():
            dfm_id = int(row["DFM"])
            chamber_idx = int(row["Chamber"])
            treatment_name = str(row.get("Treatment", ""))
            key = (dfm_id, chamber_idx)
            if key in remove_set:
                continue

            reasons: list[str] = []
            if has_single:
                val = row.get("Licks")
                if pd.isna(val):
                    reasons.append("Licks is NaN/undefined")
                elif cutoff is not None and float(val) < cutoff:
                    reasons.append(f"Licks={float(val):.6g} < min_untransformed_licks_cutoff={cutoff:.6g}")
            else:
                va = row.get("LicksA")
                vb = row.get("LicksB")
                if pd.isna(va):
                    reasons.append("LicksA is NaN/undefined")
                elif cutoff is not None and float(va) < cutoff:
                    reasons.append(f"LicksA={float(va):.6g} < min_untransformed_licks_cutoff={cutoff:.6g}")
                if pd.isna(vb):
                    reasons.append("LicksB is NaN/undefined")
                elif cutoff is not None and float(vb) < cutoff:
                    reasons.append(f"LicksB={float(vb):.6g} < min_untransformed_licks_cutoff={cutoff:.6g}")

            if reasons:
                remove_set.add(key)
                removed_rows.append(
                    {
                        "DFM": dfm_id,
                        "Chamber": chamber_idx,
                        "Treatment": treatment_name,
                        "Reason": "; ".join(reasons),
                    }
                )
                print(
                    f"  [auto_remove_chambers] Removing DFM {dfm_id} Chamber {chamber_idx}"
                    f" ({treatment_name}): {'; '.join(reasons)}",
                    flush=True,
                )

        if remove_set:
            self._remove_chambers_from_design(remove_set)

        result = pd.DataFrame(removed_rows, columns=["DFM", "Chamber", "Treatment", "Reason"])
        if result.empty:
            print("auto_remove_chambers: no chambers removed.", flush=True)
        else:
            print(f"auto_remove_chambers: done — {len(result)} chamber(s) removed.", flush=True)

        self.filtered_chambers = result

        # Build human-readable criteria summary
        lines = ["Lick filter (applied to all experiment types)", ""]
        lines.append("  • Chambers with NaN/undefined lick values are always excluded.")
        if cutoff is not None:
            lick_col = "Licks" if has_single else "LicksA or LicksB"
            lines.append(
                f"  • min_untransformed_licks_cutoff = {cutoff:g}\n"
                f"    Excluded if {lick_col} < {cutoff:g}"
            )
        else:
            lines.append("  • No lick-count cutoff configured — only NaN check applied.")
        self.filter_criteria_summary = "\n".join(lines)
        self.write_removed_chambers()

        return result

    @property
    def _output_root(self) -> Path | None:
        """The Experiment Directory itself.

        Kept as a property because callers reference it, but there is no longer
        anything between the Experiment Directory and its outputs: ADR-0005
        retired ``<stem>_results/`` along with multi-YAML directories.
        """
        return self.experiment_dir

    @property
    def analysis_dir(self) -> Path | None:
        """``experiment_dir/analysis``, or ``None`` if no ``experiment_dir`` is set.

        Fixed, never suffixed by the time window (ADR-0008): a window is a Facet
        column now, so one analysis directory holds the whole faceted result.
        """
        root = self._output_root
        return None if root is None else root / "analysis"

    @property
    def qc_dir(self) -> Path | None:
        """``experiment_dir/qc``, or ``None`` if no ``experiment_dir`` is set."""
        root = self._output_root
        return None if root is None else root / "qc"

    @property
    def figures_dir(self) -> Path | None:
        """``experiment_dir/figures`` — vector Publication Figures rendered for a
        standalone Experiment Directory.  Inside a Project the Plot Editor
        writes to the Project's own ``figures/`` instead."""
        root = self._output_root
        return None if root is None else root / "figures"

    @classmethod
    def load(
        cls,
        experiment_dir: str | Path,
        *,
        config_name: str = "flic_config.yaml",
        range_minutes: Sequence[float] = (0, 0),
        parallel: bool = True,
        max_workers: int | None = None,
        executor: Literal["threads", "processes"] = "threads",
    ) -> Experiment:
        """
        Load an experiment from a project directory.

        ``experiment_dir`` is the single required argument and is the root of the
        project layout::

            experiment_dir/
              flic_config.yaml   ← required; loaded automatically
              data/              ← DFM CSV files
              qc/                ← QC report output (write_qc_reports)
              analysis/          ← figure and summary output (write_summary)

        Parameters
        ----------
        experiment_dir:
            Root directory for the experiment project.  Must contain
            ``flic_config.yaml``.  Data is always read from ``experiment_dir/data``.
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
            experiment_dir,
            config_name=config_name,
            range_minutes=range_minutes,
            parallel=parallel,
            max_workers=max_workers,
            executor=executor,
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

            results[dfm_id] = {
                "integrity": integrity,
                "integrity_text": integrity_text,
                "data_breaks_multiplier": float(data_breaks_multiplier),
                "data_breaks_count": breaks_count,
                "data_breaks_head": None if breaks is None else breaks.head(10),
                "simultaneous_feeding_matrix": None,
                "bleeding_cutoff": float(bleeding_cutoff),
                "bleeding": None,
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

        If *out_dir* is not given, defaults to ``experiment_dir/qc``.  Raises
        ``ValueError`` if neither *out_dir* nor ``experiment_dir`` is set.
        """

        if out_dir is None:
            if self.qc_dir is None:
                raise ValueError(
                    "out_dir must be provided when no experiment_dir is set on the Experiment."
                )
            out_dir = self.qc_dir
        out = Path(out_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)

        # Subdirectories for each output type
        dir_integrity = out / "integrity"
        dir_data_breaks = out / "data_breaks"
        dir_raw = out / "raw_signal"
        dir_baselined = out / "baselined"
        dir_cumulative = out / "cumulative_licks"
        dir_sim = out / "simultaneous_feeding"
        dir_bleed = out / "bleeding"

        for d in (dir_integrity, dir_data_breaks, dir_raw, dir_baselined, dir_cumulative):
            d.mkdir(parents=True, exist_ok=True)

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
            pd.DataFrame([integrity]).to_csv(
                dir_integrity / f"DFM{dfm_id}_integrity_report.csv", index=False
            )
            if r.get("integrity_text"):
                (dir_integrity / f"DFM{dfm_id}_integrity_report.txt").write_text(
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
                    breaks.to_csv(
                        dir_data_breaks / f"DFM{dfm_id}_data_breaks.csv", index=False
                    )

            # Two-well only
            sim_df = r.get("simultaneous_feeding_matrix")
            if isinstance(sim_df, pd.DataFrame):
                dir_sim.mkdir(parents=True, exist_ok=True)
                sim_df.to_csv(dir_sim / f"DFM{dfm_id}_simultaneous_feeding_matrix.csv")

            bleed = r.get("bleeding")
            if isinstance(bleed, dict) and "Matrix" in bleed and "AllData" in bleed:
                dir_bleed.mkdir(parents=True, exist_ok=True)
                bleed["Matrix"].to_csv(dir_bleed / f"DFM{dfm_id}_bleeding_matrix.csv")
                bleed["AllData"].to_csv(
                    dir_bleed / f"DFM{dfm_id}_bleeding_alldata.csv", header=True
                )

            # Signal plots
            dfm = self.dfms[dfm_id]
            fig = dfm.plot_raw()
            fig.savefig(dir_raw / f"DFM{dfm_id}_raw.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

            fig = dfm.plot_baselined(include_thresholds=True)
            fig.savefig(dir_baselined / f"DFM{dfm_id}_baselined.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

            fig = dfm.plot_cumulative_licks(transform_licks=True)
            fig.savefig(dir_cumulative / f"DFM{dfm_id}_cumulative_licks.png", dpi=200, bbox_inches="tight")
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

        try:
            from importlib.metadata import version as _v
            _pyflic_version = _v("pyflic")
        except Exception:
            _pyflic_version = "unknown"

        buf = StringIO()
        buf.write("FLIC experiment summary\n")
        buf.write("======================\n")
        buf.write(f"pyflic version: {_pyflic_version}\n\n")

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

        group_label = f"'{self.exclusion_group}'" if self.exclusion_group else "file"
        buf.write(f"Chambers excluded (group {group_label})\n")
        buf.write("-" * 40 + "\n")
        if self.excluded_chambers:
            total = 0
            for dfm_id, chambers in sorted(self.excluded_chambers.items()):
                buf.write(f"  DFM {dfm_id}: chamber(s) {chambers}\n")
                total += len(chambers)
            buf.write(f"  Total: {total} chamber(s).\n\n")
        else:
            buf.write("  (none)\n\n")

        if self.filtered_chambers is not None and not self.filtered_chambers.empty:
            buf.write("Chambers removed by auto_remove_chambers\n")
            buf.write("----------------------------------------\n")
            for _, row in self.filtered_chambers.iterrows():
                buf.write(
                    f"  DFM {int(row['DFM'])} Chamber {int(row['Chamber'])}"
                    f" ({row['Treatment']}): {row['Reason']}\n"
                )
            buf.write(f"  Total: {len(self.filtered_chambers)} chamber(s).\n\n")

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
        transform_licks: bool | None = None,
        single_plot: bool = True,
    ):
        """
        Plot cumulative licks for a single DFM chamber, with the treatment name
        in the title if that chamber is assigned to one.
        """
        if transform_licks is None:
            transform_licks = self.transform_licks
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
        Publication-quality jitter + mean ± SEM plot faceted by a grouping column.

        Points are jittered over the x-axis grouping; a filled diamond marks the
        group mean and capped error bars show ± 1 SEM.  Colors follow the
        Okabe-Ito color-blind-safe palette unless *colors* is provided.

        Parameters
        ----------
        df : pd.DataFrame
            Feeding summary (or any DataFrame) with the relevant columns.
        x_col : str
            Column for the x-axis grouping (e.g. ``"TreatmentNew"``).
        y_col : str
            Column for the y-axis metric (e.g. ``"_MetricValue"``).
        facet_col : str
            Column to facet by.  Pass ``"_Facet"`` (synthetic constant column)
            for a single-panel plot — the strip label is hidden automatically.
        x_order : list[str] | None
            Explicit ordering of x-axis categories.
        annotation : str | None
            Optional text label drawn on every facet panel.
        annotation_x / annotation_y
            Data coordinates for the annotation.
        """
        from plotnine import (
            aes,
            annotate as p9_annotate,
            coord_cartesian,
            element_blank,
            element_line,
            element_rect,
            element_text,
            facet_wrap,
            geom_jitter,
            ggplot,
            guides,
            labs,
            scale_color_manual,
            scale_shape_manual,
            scale_x_discrete,
            stat_summary,
            theme,
            theme_classic,
        )

        # Derive figure width from number of groups × facets.
        n_groups = df[x_col].nunique() if x_col in df.columns else 2
        n_facets = df[facet_col].nunique() if facet_col in df.columns else 1
        fig_w = max(3.0, n_groups * 1.4) * n_facets
        fig_h = 4.5

        palette = list(colors.values()) if colors is not None else _OKABE_ITO
        # Build a shape cycle using string marker codes (avoids numpy int issues).
        _shape_strs = ["o", "s", "D", "^", "v", ">", "<", "p"]
        group_levels = (
            x_order if x_order is not None
            else sorted(df[x_col].unique()) if x_col in df.columns else []
        )
        shape_map = {g: _shape_strs[i % len(_shape_strs)] for i, g in enumerate(group_levels)}

        if size_col is not None:
            jitter_kwargs: dict = {
                "mapping": aes(size=size_col),
                "width": jitter_width,
                "alpha": 0.65,
                "stroke": 0.3,
            }
        else:
            jitter_kwargs = {
                "width": jitter_width,
                "size": point_size,
                "alpha": 0.65,
                "stroke": 0.3,
            }

        p = (
            ggplot(df, aes(x=x_col, y=y_col, color=x_col, shape=x_col))
            + geom_jitter(**jitter_kwargs)
            + stat_summary(
                fun_y=np.mean,
                geom="point",
                color="black",
                shape="D",
                size=4,
                fill="black",
            )
            + stat_summary(
                fun_ymin=lambda x: x.mean() - x.sem(),
                fun_ymax=lambda x: x.mean() + x.sem(),
                geom="errorbar",
                color="black",
                size=0.7,
                width=0.15,
            )
            + facet_wrap(f"~ {facet_col}")
            + scale_color_manual(values=palette)
            + scale_shape_manual(values=shape_map)
            + guides(color=False, shape=False)
            + theme_classic(base_size=base_font_size)
            + theme(
                axis_line=element_line(color="black", size=0.8),
                axis_ticks=element_line(color="black", size=0.6),
                strip_background=element_rect(fill="#f0f0f0", color="black", size=0.5),
                strip_text=element_text(size=base_font_size * 0.85, face="bold"),
                figure_size=(fig_w, fig_h),
            )
            + labs(title=title, x=x_label or x_col, y=y_label or y_col)
        )

        # Hide the facet strip entirely when it carries no information.
        if facet_col == "_Facet":
            p = p + theme(
                strip_background=element_blank(),
                strip_text=element_blank(),
            )

        if x_order is not None or x_labels is not None:
            limits = x_order
            labels = ([x_labels.get(v, v) for v in x_order] if (x_labels and x_order) else None)
            p = p + scale_x_discrete(limits=limits, labels=labels)

        if ylim is not None:
            p = p + coord_cartesian(ylim=ylim)

        if annotation is not None:
            p = p + p9_annotate("text", x=annotation_x, y=annotation_y, label=annotation)

        return p

    def _feeding_plot_metrics(self) -> list[tuple[str, str]]:
        """Return ``(column_name, display_label)`` pairs for :meth:`plot_feeding_summary`.

        Override in subclasses to provide chamber-size-specific metric lists.
        """
        return []

    def _feeding_summary_default_ncols(self) -> int:
        """Default number of columns for :meth:`plot_feeding_summary` facet grid."""
        return 3

    def plot_feeding_summary(
        self,
        *,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool | None = None,
        ncols: int | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> Any:
        """
        Produce a publication-quality faceted jitter + mean ± SEM plot of feeding
        summary metrics grouped by treatment.

        Treatments appear on the x-axis; each facet shows one metric with free y
        scales.  Individual chamber values are jittered over the group; a filled
        diamond marks the group mean and capped error bars show ± 1 SEM.

        For chamber_size=1 (8 metrics) the default grid is 3 columns.
        For chamber_size=2 (18 metrics) the default grid is 4 columns.

        Returns a plotnine ggplot object.
        """
        import math

        from plotnine import (
            aes,
            element_blank,
            element_line,
            element_rect,
            element_text,
            facet_wrap,
            geom_jitter,
            ggplot,
            guides,
            scale_color_manual,
            scale_shape_manual,
            stat_summary,
            theme,
            theme_classic,
        )

        df = self.feeding_summary(
            range_minutes=range_minutes, transform_licks=transform_licks
        )
        if df.empty:
            from plotnine import annotate, theme_classic as _tc
            return (
                ggplot()
                + annotate("text", x=0, y=0, label="No feeding summary data")
                + _tc()
            )

        metrics = [(col, lbl) for col, lbl in self._feeding_plot_metrics() if col in df.columns]

        if not metrics:
            from plotnine import annotate, theme_classic as _tc
            return (
                ggplot()
                + annotate("text", x=0, y=0, label="No matching columns in feeding summary")
                + _tc()
            )

        df, group_col = self._resolve_group_col(df)
        id_cols = [c for c in dict.fromkeys([group_col, "DFM", "Chamber"]) if c in df.columns]
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
            ncols = self._feeding_summary_default_ncols()
        nrows = math.ceil(len(metrics) / ncols)

        if figsize is None:
            figsize = (ncols * 3.5, nrows * 3.5)

        group_levels = sorted(df_long[group_col].unique())
        _shape_strs = ["o", "s", "D", "^", "v", ">", "<", "p"]
        shape_map = {g: _shape_strs[i % len(_shape_strs)] for i, g in enumerate(group_levels)}

        p = (
            ggplot(df_long, aes(x=group_col, y="Value", color=group_col, shape=group_col))
            + geom_jitter(width=0.15, size=1.8, alpha=0.65, stroke=0.3)
            + stat_summary(
                fun_y=np.mean,
                geom="point",
                color="black",
                shape="D",
                size=3,
                fill="black",
            )
            + stat_summary(
                fun_ymin=lambda x: x.mean() - x.sem(),
                fun_ymax=lambda x: x.mean() + x.sem(),
                geom="errorbar",
                color="black",
                size=0.6,
                width=0.15,
            )
            + facet_wrap("~ Metric", ncol=ncols, scales="free_y")
            + scale_color_manual(values=_OKABE_ITO)
            + scale_shape_manual(values=shape_map)
            + guides(color=False, shape=False)
            + theme_classic(base_size=9)
            + theme(
                axis_line=element_line(color="black", size=0.6),
                axis_ticks=element_line(color="black", size=0.5),
                axis_text_x=element_text(rotation=30, hjust=1),
                strip_background=element_rect(fill="#f0f0f0", color="black", size=0.5),
                strip_text=element_text(size=8, face="bold"),
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
        exp_range = getattr(self, "range_minutes", None)
        if (not range_is_specified_local(range_minutes)) and exp_range is not None:
            if range_is_specified_local(exp_range):
                effective_range = exp_range

        # Cache binned summary per DFM so we don't recompute repeatedly per treatment.
        binned_by_dfm: dict[int, pd.DataFrame] = {}
        for dfm_id, dfm in self.dfms.items():
            binned_by_dfm[int(dfm_id)] = dfm.binned_feeding_summary(
                binsize_min=float(binsize_min),
                range_minutes=effective_range,
                transform_licks=bool(transform_licks),
            )

        return self._assemble_treatment_table(binned_by_dfm)

    def _moving_window_table_by_treatment(
        self,
        *,
        window_min: float = 60.0,
        step_min: float = 30.0,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool = True,
    ) -> pd.DataFrame:
        """
        Long table of moving-window feeding-summary rows for treatment chambers.

        The moving-window analogue of :meth:`_binned_licks_table_by_treatment`;
        each DFM's :meth:`DFM.moving_window_feeding_summary` is computed once and
        the rows for treatment-assigned chambers are stacked.  Output columns
        match the binned table (Treatment, DFM, Chamber, Interval, Minutes, …
        feeding-summary metrics), so the same metric-extraction path applies.
        """
        def range_is_specified_local(r: Sequence[float]) -> bool:
            try:
                return not (float(r[0]) == 0.0 and float(r[1]) == 0.0)
            except Exception:  # noqa: BLE001
                return False

        effective_range = range_minutes
        exp_range = getattr(self, "range_minutes", None)
        if (not range_is_specified_local(range_minutes)) and exp_range is not None:
            if range_is_specified_local(exp_range):
                effective_range = exp_range

        summary_by_dfm: dict[int, pd.DataFrame] = {}
        for dfm_id, dfm in self.dfms.items():
            summary_by_dfm[int(dfm_id)] = dfm.moving_window_feeding_summary(
                window_min=float(window_min),
                step_min=float(step_min),
                range_minutes=effective_range,
                transform_licks=bool(transform_licks),
            )

        return self._assemble_treatment_table(summary_by_dfm)

    def _assemble_treatment_table(
        self, summary_by_dfm: dict[int, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Stack per-DFM feeding-summary rows into a treatment-labelled long table.

        Given a mapping of ``dfm_id -> per-chamber summary rows`` (binned or
        moving-window), select the chambers assigned to each treatment, tag them
        with the treatment name, and append design-factor columns.  Shared by
        the binned and moving-window by-treatment table builders.
        """
        parts: list[pd.DataFrame] = []
        for trt_name, trt in self.design.treatments.items():
            if not trt.chambers:
                continue

            # Group selection by DFM
            by_dfm: dict[int, set[int]] = {}
            for tc in trt.chambers:
                by_dfm.setdefault(int(tc.dfm_id), set()).add(int(tc.chamber_index))

            for dfm_id, chamber_idxs in by_dfm.items():
                summary = summary_by_dfm.get(int(dfm_id))
                if summary is None or summary.empty or "Chamber" not in summary.columns:
                    continue

                tmp = summary[summary["Chamber"].astype(int).isin(chamber_idxs)].copy()
                if tmp.empty:
                    continue

                tmp.insert(0, "Treatment", str(trt_name))

                # Ensure DFM exists for grouping even if upstream summary didn't include it.
                if "DFM" not in tmp.columns:
                    tmp.insert(tmp.columns.get_loc("Treatment") + 1, "DFM", int(dfm_id))
                parts.append(tmp)

        result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        return self._append_factor_columns(result)

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
            # total — preserve NaN when *both* wells have no data for a bin
            result = a.fillna(0.0) + b.fillna(0.0)
            result[a.isna() & b.isna()] = np.nan
            return result

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
        transform_licks: bool | None = None,
        show_sem: bool = True,
        show_individual_chambers: bool = False,
        figsize: tuple[float, float] = (10, 4),
    ) -> Any:
        """
        Plot mean ± SEM of a binned feeding-summary metric for each treatment across the experiment.

        - Uses the chamber→treatment assignments in `Experiment.design`.
        - Uses per-DFM `binned_feeding_summary()` and aggregates across chambers for each treatment.

        Returns a plotnine ggplot object.
        """
        if transform_licks is None:
            transform_licks = self.transform_licks

        df = self._binned_licks_table_by_treatment(
            binsize_min=binsize_min, range_minutes=range_minutes, transform_licks=transform_licks
        )
        return self._metric_lines_by_treatment(
            df,
            metric=metric,
            two_well_mode=two_well_mode,
            transform_licks=bool(transform_licks),
            title=f"Binned {metric} by treatment (mean ± SEM)",
            empty_label="Binned metric by treatment (no data)",
            show_sem=show_sem,
            show_individual_chambers=show_individual_chambers,
            figsize=figsize,
        )

    def plot_moving_window_metric_by_treatment(
        self,
        *,
        metric: str = "MedDuration",
        two_well_mode: Literal["total", "mean_ab", "A", "B"] = "mean_ab",
        window_min: float = 60.0,
        step_min: float = 30.0,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool | None = None,
        show_sem: bool = True,
        show_individual_chambers: bool = False,
        figsize: tuple[float, float] = (10, 4),
    ) -> Any:
        """
        Plot mean ± SEM of a feeding-summary metric by treatment over a sliding
        window.

        The moving-window counterpart to :meth:`plot_binned_metric_by_treatment`:
        the same metric set is available, but values are computed over
        overlapping windows (*window_min* wide, advanced by *step_min*) rather
        than disjoint bins, giving a smoother time course.  Each window is
        plotted at its right edge.  Returns a plotnine ggplot object.
        """
        if transform_licks is None:
            transform_licks = self.transform_licks

        df = self._moving_window_table_by_treatment(
            window_min=window_min,
            step_min=step_min,
            range_minutes=range_minutes,
            transform_licks=transform_licks,
        )
        return self._metric_lines_by_treatment(
            df,
            metric=metric,
            two_well_mode=two_well_mode,
            transform_licks=bool(transform_licks),
            title=f"Moving-window {metric} by treatment (mean ± SEM)",
            empty_label="Moving-window metric by treatment (no data)",
            show_sem=show_sem,
            show_individual_chambers=show_individual_chambers,
            figsize=figsize,
        )

    def _metric_lines_by_treatment(
        self,
        df: pd.DataFrame,
        *,
        metric: str,
        two_well_mode: str,
        transform_licks: bool,
        title: str,
        empty_label: str,
        show_sem: bool = True,
        show_individual_chambers: bool = False,
        figsize: tuple[float, float] = (10, 4),
    ) -> Any:
        """
        Render a per-treatment mean ± SEM line plot of a metric over time.

        Takes a long table with feeding-summary columns plus a ``Minutes``
        column (as produced by the binned or moving-window by-treatment table
        builders), extracts *metric* via :meth:`_metric_series_from_binned_rows`,
        and draws one mean line (with SEM ribbon) per treatment.  Shared by the
        binned and moving-window by-treatment plots so both stay visually
        identical.
        """
        from plotnine import (
            aes,
            annotate,
            element_line,
            geom_line,
            geom_ribbon,
            ggplot,
            labs,
            scale_color_manual,
            scale_fill_manual,
            theme,
            theme_classic,
        )

        if df.empty:
            return (
                ggplot()
                + annotate("text", x=0, y=0, label=empty_label)
                + theme_classic()
            )

        df = df.copy()
        df["MetricValue"] = self._metric_series_from_binned_rows(df, metric=metric, two_well_mode=two_well_mode)
        df = df.dropna(subset=["Minutes", "MetricValue"]).copy()
        df["Minutes"] = pd.to_numeric(df["Minutes"], errors="coerce")
        df["MetricValue"] = pd.to_numeric(df["MetricValue"], errors="coerce")
        df = df.dropna(subset=["Minutes", "MetricValue"])
        if df.empty:
            return (
                ggplot()
                + annotate("text", x=0, y=0, label=empty_label)
                + theme_classic()
            )

        df, group_col = self._resolve_group_col(df)
        treatments = sorted(df[group_col].unique().tolist())
        n_t = len(treatments)
        palette = (_OKABE_ITO * ((n_t // len(_OKABE_ITO)) + 1))[:n_t]
        color_map = {t: palette[i] for i, t in enumerate(treatments)}

        agg = (
            df.groupby([group_col, "Minutes"], as_index=False)["MetricValue"]
            .agg(mean="mean", std="std", n="count")
            .rename(columns={"mean": "Mean", "std": "Std", "n": "N"})
        )
        agg["SEM"] = agg["Std"] / np.sqrt(agg["N"].clip(lower=1))
        agg["Lower"] = agg["Mean"] - agg["SEM"]
        agg["Upper"] = agg["Mean"] + agg["SEM"]

        # Annotate group labels with sample size
        max_n = agg.groupby(group_col)["N"].transform("max")
        agg["_Label"] = agg[group_col] + " (n=" + max_n.astype(int).astype(str) + ")"
        label_map = {
            t: f"{t} (n={int(agg.loc[agg[group_col] == t, 'N'].max())})"
            for t in treatments
        }
        label_palette = {label_map[t]: color_map[t] for t in treatments}

        ylabel = metric
        if metric in ("Licks", "Events") and two_well_mode == "total":
            ylabel = f"{metric} (total)"
        if transform_licks and metric == "Licks":
            ylabel = f"Transformed {ylabel}"

        p = ggplot(agg, aes(x="Minutes", y="Mean", color="_Label", fill="_Label"))

        if show_individual_chambers:
            df_ind = df.copy()
            df_ind["_ChamberKey"] = df_ind["DFM"].astype(str) + "_" + df_ind["Chamber"].astype(str)
            df_ind["_Label"] = df_ind[group_col].map(label_map)
            p = p + geom_line(
                data=df_ind,
                mapping=aes(x="Minutes", y="MetricValue", group="_ChamberKey", color="_Label"),
                alpha=0.18,
                size=0.6,
                show_legend=False,
            )

        p = p + geom_line(size=1.2)
        if show_sem:
            p = p + geom_ribbon(aes(ymin="Lower", ymax="Upper"), alpha=0.2, color=None)

        p = (
            p
            + scale_color_manual(values=label_palette)
            + scale_fill_manual(values=label_palette)
            + labs(
                title=title,
                x="Minutes",
                y=ylabel,
                color="",
                fill="",
            )
            + theme_classic(base_size=11)
            + theme(
                axis_line=element_line(color="black", size=0.7),
                figure_size=figsize,
            )
        )
        return p

    def _moving_median_table(
        self,
        *,
        window_min: float,
        step_min: float,
        range_minutes: Sequence[float],
        two_well_mode: Literal["mean_ab", "A", "B"],
    ) -> tuple[pd.DataFrame, str]:
        """
        Build a per-chamber, per-time table of moving-window median bout
        duration for treatment-assigned chambers.

        Returns ``(df, group_col)`` where *df* has one row per
        (group, DFM, Chamber, Minutes) with a single ``MedDuration`` column,
        and *group_col* is the treatment/factor grouping column (see
        :meth:`_resolve_group_col`).  For two-well designs the per-chamber
        ``MedDurationA``/``MedDurationB`` columns are reduced according to
        *two_well_mode*: ``mean_ab`` averages wells A and B (NaN-skipping),
        ``A``/``B`` keep a single well.
        """
        raw = self.moving_median_duration(
            window_min=window_min,
            step_min=step_min,
            range_minutes=range_minutes,
            save=False,
        )
        if raw.empty:
            return raw, "Treatment"
        df = raw.copy()
        # Two-well summaries carry MedDurationA/MedDurationB; single-well a
        # single MedDuration column.
        if "MedDurationA" in df.columns:
            mode = (two_well_mode or "mean_ab").lower()
            if mode in ("a", "wella"):
                df["MedDuration"] = df["MedDurationA"]
            elif mode in ("b", "wellb"):
                df["MedDuration"] = df["MedDurationB"]
            else:  # mean_ab — average the two wells, skipping NaN.
                df["MedDuration"] = df[["MedDurationA", "MedDurationB"]].mean(axis=1)
        df, group_col = self._resolve_group_col(df)
        keys = [k for k in (group_col, "DFM", "Chamber", "Minutes") if k in df.columns]
        collapsed = df.groupby(keys, as_index=False)["MedDuration"].mean()
        return collapsed, group_col

    def plot_moving_median_duration_by_chamber(
        self,
        *,
        window_min: float = 60.0,
        step_min: float = 30.0,
        range_minutes: Sequence[float] = (0, 0),
        two_well_mode: Literal["mean_ab", "A", "B"] = "mean_ab",
        figsize: tuple[float, float] = (10, 4),
    ) -> Any:
        """
        Plot the time-dependent moving-window median bout duration of every
        treatment-assigned chamber as a separate line, coloured by treatment.

        Each chamber's median bout duration is computed over a sliding window
        of *window_min* minutes advanced in *step_min* increments (see
        :meth:`moving_median_duration`).  Returns a plotnine ggplot object.
        """
        from plotnine import (
            aes,
            annotate,
            element_line,
            geom_line,
            ggplot,
            labs,
            scale_color_manual,
            theme,
            theme_classic,
        )

        df, group_col = self._moving_median_table(
            window_min=window_min,
            step_min=step_min,
            range_minutes=range_minutes,
            two_well_mode=two_well_mode,
        )
        if df.empty:
            return (
                ggplot()
                + annotate("text", x=0, y=0, label="Moving median duration (no data)")
                + theme_classic()
            )

        df = df.copy()
        df["Minutes"] = pd.to_numeric(df["Minutes"], errors="coerce")
        df["MedDuration"] = pd.to_numeric(df["MedDuration"], errors="coerce")
        df = df.dropna(subset=["Minutes", "MedDuration"])
        if df.empty:
            return (
                ggplot()
                + annotate("text", x=0, y=0, label="Moving median duration (no data)")
                + theme_classic()
            )

        treatments = sorted(df[group_col].unique().tolist())
        n_t = len(treatments)
        palette = (_OKABE_ITO * ((n_t // len(_OKABE_ITO)) + 1))[:n_t]
        color_map = {t: palette[i] for i, t in enumerate(treatments)}
        n_cham = {
            t: int(df.loc[df[group_col] == t, ["DFM", "Chamber"]].drop_duplicates().shape[0])
            for t in treatments
        }
        label_map = {t: f"{t} (n={n_cham[t]})" for t in treatments}
        label_palette = {label_map[t]: color_map[t] for t in treatments}

        df["_ChamberKey"] = df["DFM"].astype(str) + "_" + df["Chamber"].astype(str)
        df["_Label"] = df[group_col].map(label_map)

        p = (
            ggplot(df, aes(x="Minutes", y="MedDuration", color="_Label", group="_ChamberKey"))
            + geom_line(alpha=0.7, size=0.8)
            + scale_color_manual(values=label_palette)
            + labs(
                title="Time-dependent median bout duration by chamber",
                x="Minutes",
                y="Median bout duration (s)",
                color="",
            )
            + theme_classic(base_size=11)
            + theme(
                axis_line=element_line(color="black", size=0.7),
                figure_size=figsize,
            )
        )
        return p

    def plot_moving_median_duration_by_treatment(
        self,
        *,
        window_min: float = 60.0,
        step_min: float = 30.0,
        range_minutes: Sequence[float] = (0, 0),
        two_well_mode: Literal["mean_ab", "A", "B"] = "mean_ab",
        show_sem: bool = True,
        figsize: tuple[float, float] = (10, 4),
    ) -> Any:
        """
        Plot the treatment-level mean (± SEM across chambers) of the
        time-dependent moving-window median bout duration.

        At each window position the per-chamber median bout durations are
        averaged within each treatment; the shaded ribbon shows ± 1 standard
        error of the mean across chambers.  Returns a plotnine ggplot object.
        """
        from plotnine import (
            aes,
            annotate,
            element_line,
            geom_line,
            geom_ribbon,
            ggplot,
            labs,
            scale_color_manual,
            scale_fill_manual,
            theme,
            theme_classic,
        )

        df, group_col = self._moving_median_table(
            window_min=window_min,
            step_min=step_min,
            range_minutes=range_minutes,
            two_well_mode=two_well_mode,
        )
        if df.empty:
            return (
                ggplot()
                + annotate("text", x=0, y=0, label="Moving median duration (no data)")
                + theme_classic()
            )

        df = df.copy()
        df["Minutes"] = pd.to_numeric(df["Minutes"], errors="coerce")
        df["MedDuration"] = pd.to_numeric(df["MedDuration"], errors="coerce")
        df = df.dropna(subset=["Minutes", "MedDuration"])
        if df.empty:
            return (
                ggplot()
                + annotate("text", x=0, y=0, label="Moving median duration (no data)")
                + theme_classic()
            )

        treatments = sorted(df[group_col].unique().tolist())
        n_t = len(treatments)
        palette = (_OKABE_ITO * ((n_t // len(_OKABE_ITO)) + 1))[:n_t]
        color_map = {t: palette[i] for i, t in enumerate(treatments)}

        agg = (
            df.groupby([group_col, "Minutes"], as_index=False)["MedDuration"]
            .agg(mean="mean", std="std", n="count")
            .rename(columns={"mean": "Mean", "std": "Std", "n": "N"})
        )
        agg["SEM"] = agg["Std"] / np.sqrt(agg["N"].clip(lower=1))
        agg["Lower"] = agg["Mean"] - agg["SEM"]
        agg["Upper"] = agg["Mean"] + agg["SEM"]

        label_map = {
            t: f"{t} (n={int(agg.loc[agg[group_col] == t, 'N'].max())})"
            for t in treatments
        }
        label_palette = {label_map[t]: color_map[t] for t in treatments}
        agg["_Label"] = agg[group_col].map(label_map)

        p = ggplot(agg, aes(x="Minutes", y="Mean", color="_Label", fill="_Label"))
        p = p + geom_line(size=1.2)
        if show_sem:
            p = p + geom_ribbon(aes(ymin="Lower", ymax="Upper"), alpha=0.2, color=None)
        p = (
            p
            + scale_color_manual(values=label_palette)
            + scale_fill_manual(values=label_palette)
            + labs(
                title="Time-dependent median bout duration by treatment (mean ± SEM)",
                x="Minutes",
                y="Median bout duration (s)",
                color="",
                fill="",
            )
            + theme_classic(base_size=11)
            + theme(
                axis_line=element_line(color="black", size=0.7),
                figure_size=figsize,
            )
        )
        return p

    def plot_dot_metric_by_treatment(
        self,
        *,
        metric: str = "Licks",
        two_well_mode: Literal["total", "mean_ab", "A", "B"] = "total",
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool | None = None,
    ) -> Any:
        """
        Jitter + mean ± SE dot plot of a single feeding-summary metric by treatment.

        When ``design_factors`` has two factors (e.g. TreatmentNew × Sex), the
        first factor is placed on the x-axis and the second used as a facet so
        each factor level gets its own panel.  With one or no factors all
        treatments appear on a single x-axis with no extra faceting.
        """
        from plotnine import annotate, theme_bw

        df = self.feeding_summary(range_minutes=range_minutes, transform_licks=transform_licks)
        if df.empty:
            from plotnine import ggplot
            return ggplot() + annotate("text", x=0, y=0, label="No data") + theme_bw()

        df = df.copy()
        df["_MetricValue"] = self._metric_series_from_binned_rows(
            df, metric=metric, two_well_mode=two_well_mode
        )
        df = df.dropna(subset=["_MetricValue"])

        label = metric
        if metric in ("Licks", "Events") and two_well_mode == "total":
            label = f"{metric} (total)"

        factors = self.design_factors or []
        factor_cols = [f for f in factors if f in df.columns]

        if len(factor_cols) >= 2:
            x_col = factor_cols[0]
            facet_col = factor_cols[1]
        elif len(factor_cols) == 1:
            x_col = factor_cols[0]
            df["_Facet"] = "All"
            facet_col = "_Facet"
        else:
            df, x_col = self._resolve_group_col(df)
            df["_Facet"] = "All"
            facet_col = "_Facet"

        return self.plot_jitter_summary(
            df,
            x_col=x_col,
            y_col="_MetricValue",
            facet_col=facet_col,
            title=f"{label} by treatment",
            y_label=label,
        )

    def plot_binned_metrics_by_treatment(
        self,
        *,
        metrics: Sequence[str] = ("Licks", "Events", "MedDuration"),
        two_well_mode: Literal["total", "mean_ab", "A", "B"] = "total",
        binsize_min: float = 30.0,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool | None = None,
        show_sem: bool = True,
        show_individual_chambers: bool = False,
        ncols: int = 2,
        figsize: tuple[float, float] | None = None,
    ) -> Any:
        """
        Plot multiple binned treatment timecourses (mean ± SEM), one panel per metric.

        Returns a plotnine ggplot object with ``facet_wrap`` over metrics.
        """
        if transform_licks is None:
            transform_licks = self.transform_licks
        import math

        from plotnine import (
            aes,
            annotate,
            element_line,
            element_text,
            facet_wrap,
            geom_line,
            geom_ribbon,
            ggplot,
            labs,
            scale_color_manual,
            scale_fill_manual,
            theme,
            theme_classic,
        )

        metrics_list = [str(m).strip() for m in metrics if str(m).strip()]
        if not metrics_list:
            return (
                ggplot()
                + annotate("text", x=0, y=0, label="No metrics specified")
                + theme_classic()
            )

        df = self._binned_licks_table_by_treatment(
            binsize_min=binsize_min, range_minutes=range_minutes, transform_licks=transform_licks
        )
        if df.empty:
            return (
                ggplot()
                + annotate("text", x=0, y=0, label="Binned metrics by treatment (no data)")
                + theme_classic()
            )

        df = df.copy()
        df["Minutes"] = pd.to_numeric(df["Minutes"], errors="coerce")
        df = df.dropna(subset=["Minutes"])
        df, group_col = self._resolve_group_col(df)
        treatments = sorted(df[group_col].unique().tolist())
        n_t = len(treatments)
        palette = (_OKABE_ITO * ((n_t // len(_OKABE_ITO)) + 1))[:n_t]
        color_map = {t: palette[i] for i, t in enumerate(treatments)}

        # Build per-metric labels (including transform prefix)
        def _ylabel(m: str) -> str:
            lbl = m
            if m in ("Licks", "Events") and two_well_mode == "total":
                lbl = f"{m} (total)"
            if transform_licks and m == "Licks":
                lbl = f"Transformed {lbl}"
            return lbl

        agg_parts: list[pd.DataFrame] = []
        ind_parts: list[pd.DataFrame] = []
        metric_order: list[str] = []

        for metric in metrics_list:
            tmp = df.copy()
            try:
                tmp["MetricValue"] = self._metric_series_from_binned_rows(
                    tmp, metric=metric, two_well_mode=two_well_mode
                )
            except ValueError:
                continue
            tmp["MetricValue"] = pd.to_numeric(tmp["MetricValue"], errors="coerce")
            tmp = tmp.dropna(subset=["MetricValue"])
            if tmp.empty:
                continue

            lbl = _ylabel(metric)
            agg = (
                tmp.groupby([group_col, "Minutes"], as_index=False)["MetricValue"]
                .agg(mean="mean", std="std", n="count")
                .rename(columns={"mean": "Mean", "std": "Std", "n": "N"})
            )
            agg["SEM"] = agg["Std"] / np.sqrt(agg["N"].clip(lower=1))
            agg["Lower"] = agg["Mean"] - agg["SEM"]
            agg["Upper"] = agg["Mean"] + agg["SEM"]
            agg["Metric"] = lbl
            agg_parts.append(agg)
            metric_order.append(lbl)

            if show_individual_chambers:
                tmp_ind = tmp.copy()
                tmp_ind["_ChamberKey"] = (
                    tmp_ind["DFM"].astype(str) + "_" + tmp_ind["Chamber"].astype(str)
                )
                tmp_ind["Metric"] = lbl
                ind_parts.append(tmp_ind)

        if not agg_parts:
            return (
                ggplot()
                + annotate("text", x=0, y=0, label="No valid metrics found")
                + theme_classic()
            )

        df_agg = pd.concat(agg_parts, ignore_index=True)
        df_agg["Metric"] = pd.Categorical(df_agg["Metric"], categories=metric_order, ordered=True)

        nrows = math.ceil(len(metric_order) / ncols)
        if figsize is None:
            figsize = (ncols * 5.5, nrows * 3.5)

        p = ggplot(df_agg, aes(x="Minutes", y="Mean", color=group_col, fill=group_col))

        if show_individual_chambers and ind_parts:
            df_ind_all = pd.concat(ind_parts, ignore_index=True)
            df_ind_all["Metric"] = pd.Categorical(
                df_ind_all["Metric"], categories=metric_order, ordered=True
            )
            p = p + geom_line(
                data=df_ind_all,
                mapping=aes(x="Minutes", y="MetricValue", group="_ChamberKey"),
                alpha=0.18,
                size=0.5,
                show_legend=False,
            )

        p = p + geom_line(size=1.0)
        if show_sem:
            p = p + geom_ribbon(aes(ymin="Lower", ymax="Upper"), alpha=0.2, color=None)

        p = (
            p
            + facet_wrap("~ Metric", ncol=ncols, scales="free_y")
            + scale_color_manual(values=color_map)
            + scale_fill_manual(values=color_map)
            + labs(
                title="Binned metrics by treatment (mean ± SEM)",
                x="Minutes",
                y="",
                color="Treatment",
                fill="Treatment",
            )
            + theme_classic(base_size=9)
            + theme(
                axis_line=element_line(color="black", size=0.6),
                strip_text=element_text(size=8, face="bold"),
                figure_size=figsize,
            )
        )
        return p

    def plot_binned_licks_by_treatment(
        self,
        *,
        binsize_min: float = 30.0,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool | None = None,
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

    def _resolve_group_col(self, df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        """Return ``(df, group_col)`` where *group_col* encodes all factor combinations.

        When ``design_factors`` are set and those columns are present in *df*, a
        synthetic ``_Group`` column is created by joining the factor values
        (e.g. ``"Ctrl / Male"``).  Otherwise ``"Treatment"`` is returned unchanged.
        """
        factor_cols = [f for f in (self.design_factors or []) if f in df.columns]
        if not factor_cols:
            return df, "Treatment"
        df = df.copy()
        df["_Group"] = df[factor_cols].astype(str).agg(" / ".join, axis=1)
        return df, "_Group"

    def _append_factor_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Insert per-factor columns after the Treatment column when design_factors is set."""
        if not self.design_factors or not self.chamber_factors or df.empty:
            return df
        if "DFM" not in df.columns or "Chamber" not in df.columns:
            return df
        df = df.copy()
        keys = list(zip(df["DFM"].astype(int), df["Chamber"].astype(int)))
        for factor in self.design_factors:
            df[factor] = [
                self.chamber_factors.get(k, {}).get(factor, "") for k in keys
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

    def _assemble_from_dfm_summaries(
        self, dfm_summaries: dict[int, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Reconstruct the treatment-level feeding summary from pre-computed per-DFM DataFrames.

        Mirrors the logic in ``Treatment.feeding_summary()`` and
        ``ExperimentDesign.feeding_summary()`` but operates on an already-computed
        dict so that DFM summaries computed in parallel can be assembled without
        re-entering the sequential call chain.
        """
        frames: list[pd.DataFrame] = []
        for trt_name, trt in self.design.treatments.items():
            by_dfm: dict[int, list[int]] = {}
            for tc in trt.chambers:
                by_dfm.setdefault(tc.dfm_id, []).append(tc.chamber_index)
            trt_frames: list[pd.DataFrame] = []
            for dfm_id, chamber_idxs in by_dfm.items():
                summ = dfm_summaries.get(dfm_id)
                if summ is None or summ.empty:
                    continue
                if "Chamber" in summ.columns:
                    summ = summ[summ["Chamber"].astype(int).isin(
                        {int(x) for x in chamber_idxs}
                    )]
                trt_frames.append(summ)
            if trt_frames:
                trt_df = pd.concat(trt_frames, ignore_index=True)
                trt_df.insert(0, "Treatment", trt_name)
                frames.append(trt_df)
        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return self._append_factor_columns(result)

    def feeding_summary(
        self,
        *,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool | None = None,
    ) -> pd.DataFrame:
        """
        Return the feeding summary for all treatment-assigned DFM chambers as a single DataFrame.

        Results are cached by ``(range_minutes, transform_licks)`` so repeated
        calls with identical arguments do not recompute the summary.  The full-
        range summary ``(0, 0)`` is pre-computed when the experiment is first
        loaded.

        When *transform_licks* is ``None`` (the default), the experiment-wide
        setting from ``self.transform_licks`` is used (driven by
        ``global.transform_licks`` in ``flic_config.yaml``).
        """
        if transform_licks is None:
            transform_licks = self.transform_licks
        key = (float(range_minutes[0]), float(range_minutes[1])), bool(transform_licks)
        if key in self._feeding_summary_cache:
            return self._feeding_summary_cache[key]

        if self.parallel:
            from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
            Exec = ProcessPoolExecutor if self.executor == "processes" else ThreadPoolExecutor
            dfm_ids = sorted({
                tc.dfm_id
                for trt in self.design.treatments.values()
                for tc in trt.chambers
            })
            dfm_summaries: dict[int, pd.DataFrame] = {}
            with Exec(max_workers=self.max_workers) as pool:
                futs = {
                    pool.submit(
                        self.dfms[did].feeding_summary,
                        range_minutes=range_minutes,
                        transform_licks=transform_licks,
                    ): did
                    for did in dfm_ids
                }
                for fut in as_completed(futs):
                    dfm_summaries[futs[fut]] = fut.result()
            result = self._assemble_from_dfm_summaries(dfm_summaries)
        else:
            result = self._append_factor_columns(
                self.design.feeding_summary(
                    range_minutes=range_minutes,
                    transform_licks=transform_licks,
                )
            )

        self._feeding_summary_cache[key] = result
        return result

    def binned_feeding_summary(
        self,
        *,
        bins: Sequence[float] | None = None,
        binsize_min: float | None = None,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool | None = None,
        path: str | Path | None = None,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Return a stacked feeding summary across time bins with an ``Interval``
        and ``Minutes`` (bin midpoint) column prepended to each row.

        By default the result is also saved to
        ``experiment_dir/analysis/binned_feeding_summary.csv`` (or *path* if
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
            Whether to apply the 0.25-power lick transformation.  When
            ``None`` (the default), uses ``self.transform_licks`` from the
            experiment (driven by ``global.transform_licks`` in
            ``flic_config.yaml``; defaults to ``True``).
        path:
            Explicit output path for the CSV.  When ``None`` (the default)
            the file is written to
            ``experiment_dir/analysis/binned_feeding_summary.csv``.
            Ignored when ``save=False``.
        save:
            Write the result to a CSV file (default ``True``).  Set to
            ``False`` to return the DataFrame without touching the filesystem.
        """
        if transform_licks is None:
            transform_licks = self.transform_licks
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
                if m_max <= 0.0 or m_max == float("inf"):
                    ## An open end runs to the last sample of the recording.
                    m_max = max(
                        float(dfm.raw_df["Minutes"].max()) for dfm in self.dfms.values()
                    )
            if m_min >= m_max:
                raise ValueError(
                    f"range_minutes start ({m_min}) must be less than end ({m_max})."
                )
            y = np.arange(m_min, m_max + 1e-9, bsize, dtype=float)
            if y.size == 0 or y[-1] < m_max:
                y = np.append(y, m_max)
            bin_pairs = [(float(y[i]), float(y[i + 1])) for i in range(len(y) - 1)]

        parts: list[pd.DataFrame] = []
        if self.parallel and len(bin_pairs) > 1:
            from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
            Exec = ProcessPoolExecutor if self.executor == "processes" else ThreadPoolExecutor
            dfm_ids = sorted({
                tc.dfm_id
                for trt in self.design.treatments.values()
                for tc in trt.chambers
            })
            # Submit one task per (dfm, bin) combination — n_dfms × n_bins tasks.
            dfm_bin_results: dict[tuple[int, float, float], pd.DataFrame] = {}
            with Exec(max_workers=self.max_workers) as pool:
                futs = {
                    pool.submit(
                        self.dfms[did].feeding_summary,
                        range_minutes=(a, b),
                        transform_licks=transform_licks,
                    ): (did, a, b)
                    for did in dfm_ids
                    for a, b in bin_pairs
                }
                for fut in as_completed(futs):
                    dfm_bin_results[futs[fut]] = fut.result()
            for a, b in bin_pairs:
                label = f"({_fmt_min(a)},{_fmt_min(b)}]"
                midpoint = (a + b) / 2.0
                bin_dfm = {did: dfm_bin_results[(did, a, b)] for did in dfm_ids}
                summ = self._assemble_from_dfm_summaries(bin_dfm)
                # Cache this per-bin result so subsequent single-bin calls are free.
                self._feeding_summary_cache[((float(a), float(b)), bool(transform_licks))] = summ
                if summ.empty:
                    continue
                summ = summ.copy()
                summ.insert(0, "Minutes", midpoint)
                summ.insert(0, "Interval", label)
                parts.append(summ)
        else:
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
                        "path must be provided when no experiment_dir is set on the Experiment."
                    )
                path = self.analysis_dir / "binned_feeding_summary.csv"
            out = Path(path).expanduser().resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)

        return df

    def moving_median_duration(
        self,
        *,
        window_min: float = 60.0,
        step_min: float = 30.0,
        range_minutes: Sequence[float] = (0, 0),
        path: str | Path | None = None,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Median feeding-bout duration over a moving window, for every treatment-
        assigned chamber across all DFMs.

        Each DFM is swept with an *window_min*-minute window advanced in
        *step_min* increments (see :meth:`DFM.moving_median_duration`); the
        per-DFM results are stacked into a single frame whose columns follow the
        feeding-summary convention for the chamber type, with a ``Treatment``
        column (and any factor columns) prepended::

            single-well: Treatment, DFM, Chamber, Minutes, MedDuration, Events
            two-well:    Treatment, DFM, Chamber, Minutes,
                         MedDurationA, MedDurationB, EventsA, EventsB

        For two-well (6-well) designs well A and well B are reported separately,
        just like ``MedDurationA``/``MedDurationB`` in :meth:`feeding_summary`.
        ``Minutes`` is the window endpoint (right edge) and the ``MedDuration*``
        columns are ``NaN`` for windows with no feeding events.  Only chambers
        assigned to a treatment are included.

        By default the result is also saved to
        ``experiment_dir/analysis/moving_median_duration.csv`` (or *path* if
        provided).  Pass ``save=False`` to suppress file output.

        Parameters
        ----------
        window_min:
            Width of the moving window in minutes (default 60.0).
        step_min:
            Distance the window advances each step, in minutes (default 30.0).
        range_minutes:
            ``(start, end)`` window the sweep is bounded to.  ``(0, 0)`` (the
            default) sweeps from 0 to the end of each DFM's recording.
        path:
            Explicit output CSV path.  When ``None`` (the default) the file is
            written to ``experiment_dir/analysis/moving_median_duration.csv``.
            Ignored when ``save=False``.
        save:
            Write the result to a CSV file (default ``True``).
        """
        dfm_ids = sorted({
            tc.dfm_id
            for trt in self.design.treatments.values()
            for tc in trt.chambers
        })
        per_dfm: dict[int, pd.DataFrame] = {}
        for did in dfm_ids:
            per_dfm[did] = self.dfms[did].moving_median_duration(
                window_min=window_min,
                step_min=step_min,
                range_minutes=range_minutes,
            )
        df = self._assemble_from_dfm_summaries(per_dfm)

        if save:
            if path is None:
                if self.analysis_dir is None:
                    raise ValueError(
                        "path must be provided when no experiment_dir is set on the Experiment."
                    )
                path = self.analysis_dir / "moving_median_duration.csv"
            out = Path(path).expanduser().resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)

        return df

    def write_feeding_summary(
        self,
        path: str | Path | None = None,
        *,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool | None = None,
    ) -> Path:
        """
        Save the feeding summary CSV to disk and return the resolved output path.

        If *path* is not given, defaults to
        ``experiment_dir/analysis/feeding_summary.csv``.  The analysis
        directory is fixed: a time window is a Facet column, not a directory
        (ADR-0008).
        Raises ``ValueError`` if neither *path* nor ``experiment_dir`` is set.
        Only includes DFM chambers assigned to a treatment.
        """
        if path is None:
            if self.analysis_dir is None:
                raise ValueError(
                    "path must be provided when no experiment_dir is set on the Experiment."
                )
            path = self.analysis_dir / "feeding_summary.csv"
        out = Path(path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        df = self.feeding_summary(range_minutes=range_minutes, transform_licks=transform_licks)
        df.to_csv(out, index=False)
        return out

    # ------------------------------------------------------------------
    # Facets (ADR-0008): a time window is a column, not a directory
    # ------------------------------------------------------------------

    def resolved_facet_cutoffs(self):
        """The facet cutoffs in effect, or ``None`` when the experiment is not
        faceted.  The Experiment Type may fix or default them; the config's
        ``facet_cutoffs`` wins otherwise."""
        if self.experiment_type is not None:
            return self.experiment_type.resolve_facet_cutoffs(
                (self.config or {}).get("global") or {})
        return tuple(self.facet_cutoffs) if self.facet_cutoffs else None

    def facet_windows(self):
        """``[(start, end), ...]`` for this experiment, or ``[]`` when unfaceted."""
        from . import windowing

        cutoffs = self.resolved_facet_cutoffs()
        return list(windowing.facet_windows(cutoffs)) if cutoffs else []

    def facet_labels(self):
        """Display labels parallel to :meth:`facet_windows`."""
        from . import windowing

        windows = self.facet_windows()
        if not windows:
            return []
        if self.experiment_type is not None:
            return self.experiment_type.phase_labels_for(
                windows, (self.config or {}).get("global") or {})
        return [windowing.minute_label(w) for w in windows]

    def feeding_summary_facet(
        self,
        *,
        transform_licks: bool | None = None,
    ) -> pd.DataFrame:
        """The feeding summary computed once per Facet and stacked.

        Adds two leading columns — ``Facet`` (the display label) and
        ``FacetRange`` (the canonical ``(start, end)`` cell) — so the window
        travels with the data instead of with the output path.  Returns an
        empty frame when the experiment is not faceted; callers should fall back
        to :meth:`feeding_summary`.
        """
        from . import windowing

        windows = self.facet_windows()
        if not windows:
            return pd.DataFrame()
        labels = self.facet_labels()
        frames: list[pd.DataFrame] = []
        for window, label in zip(windows, labels):
            df = self.feeding_summary(
                range_minutes=windowing.as_range_minutes(window),
                transform_licks=transform_licks,
            )
            if df is None or df.empty:
                continue
            df = df.copy()
            df.insert(0, "FacetRange", windowing.format_range(window))
            df.insert(0, "Facet", label)
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def write_feeding_summary_facet(
        self,
        path: str | Path | None = None,
        *,
        transform_licks: bool | None = None,
    ) -> Path | None:
        """Write ``analysis/feeding_summary_facet.csv``.

        Returns ``None`` (writing nothing) when the experiment is not faceted —
        an absent file then means "no facets", which is why it is never written
        empty.
        """
        df = self.feeding_summary_facet(transform_licks=transform_licks)
        if df.empty:
            return None
        if path is None:
            if self.analysis_dir is None:
                raise ValueError(
                    "path must be provided when no experiment_dir is set on the Experiment."
                )
            path = self.analysis_dir / "feeding_summary_facet.csv"
        out = Path(path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, na_rep="NA")
        return out

    def write_parsed_feeding_summary(
        self,
        breakpoints: Sequence[float],
        *,
        transform_licks: bool | None = None,
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

        All files are written to ``experiment_dir/analysis/`` using the same
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
        if self.analysis_dir is None:
            raise ValueError("experiment_dir must be set to write parsed feeding summaries.")
        for a, b in ranges:
            b_eff = b
            if b == float("inf"):
                actual_max = self._max_experiment_minutes()
                if actual_max is not None:
                    b_eff = actual_max
            fname = f"feeding_summary_{_fmt_min(a)}_{_fmt_min(b_eff)}.csv"
            paths.append(
                self.write_feeding_summary(
                    path=self.analysis_dir / fname,
                    range_minutes=(a, b),
                    transform_licks=transform_licks,
                )
            )
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

        If *path* is not given, defaults to ``experiment_dir/analysis/summary.txt``.
        Raises ``ValueError`` if neither *path* nor ``experiment_dir`` is set.
        """

        if path is None:
            if self.analysis_dir is None:
                raise ValueError(
                    "path must be provided when no experiment_dir is set on the Experiment."
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
        transform_licks: bool | None = None,
        ncols: int | None = None,
        figsize: tuple[float, float] | None = None,
        dpi: int = 200,
    ) -> Path:
        """
        Save the feeding summary plot to disk and return the resolved output path.

        If *path* is not given, defaults to
        ``experiment_dir/analysis/feeding_summary.{format}``.
        Raises ``ValueError`` if neither *path* nor ``experiment_dir`` is set.
        *format* may be ``"png"`` (default) or ``"pdf"``.
        """
        if path is None:
            if self.analysis_dir is None:
                raise ValueError(
                    "path must be provided when no experiment_dir is set on the Experiment."
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
        transform_licks: bool | None = None,
        plot_format: str = "png",
        dpi: int = 200,
        skip_qc: bool = False,
    ) -> dict[str, Path]:
        """
        Run the standard analysis pipeline and write all outputs to disk.

        Calls, in order:
          1. ``write_qc_reports()``           → ``experiment_dir/qc/``  (skipped when skip_qc=True)
          2. ``write_summary()``              → ``experiment_dir/analysis/summary.txt``
          3. ``write_feeding_summary()``       → ``experiment_dir/analysis/feeding_summary.csv``
          4. ``write_feeding_summary_facet()`` → ``.../feeding_summary_facet.csv`` (faceted experiments only)
          5. ``write_feeding_summary_plot()``  → ``.../feeding_summary.{plot_format}``

        Returns a dict with keys ``"qc_dir"`` (``None`` when skipped),
        ``"summary"``, ``"feeding_summary"``, ``"feeding_summary_facet"``
        (``None`` when the experiment is not faceted) and
        ``"feeding_summary_plot"`` pointing to the written paths.
        """
        n_dfms = len(self.dfms)
        n_steps = 4 if skip_qc else 5
        print("=" * 50, flush=True)
        print("FLIC Basic Analysis", flush=True)
        print(f"  Project : {self.experiment_dir}", flush=True)
        print(f"  DFMs    : {sorted(self.dfms.keys())}", flush=True)
        print("=" * 50, flush=True)

        group_label = f"'{self.exclusion_group}'" if self.exclusion_group else "file"
        print(f"\nChambers excluded (group {group_label})", flush=True)
        print("-" * 40, flush=True)
        if self.excluded_chambers:
            total = 0
            for dfm_id, chambers in sorted(self.excluded_chambers.items()):
                print(f"  DFM {dfm_id}: chamber(s) {chambers}", flush=True)
                total += len(chambers)
            print(f"  Total: {total} chamber(s).", flush=True)
        else:
            print("  (none)", flush=True)

        qc_dir: Path | None = None
        step = 0

        if not skip_qc:
            step += 1
            print(f"\n[{step}/{n_steps}] QC reports...", flush=True)
            qc_dir = self.write_qc_reports(
                data_breaks_multiplier=data_breaks_multiplier,
                bleeding_cutoff=bleeding_cutoff,
            )
            print(f"  Done — {n_dfms} DFM(s) → {qc_dir}", flush=True)

        step += 1
        print(f"\n[{step}/{n_steps}] Experiment summary...", flush=True)
        summary_path = self.write_summary(
            include_qc=not skip_qc,
            qc_data_breaks_multiplier=data_breaks_multiplier,
            qc_bleeding_cutoff=bleeding_cutoff,
        )
        print(f"  Done → {summary_path}", flush=True)

        step += 1
        print(f"\n[{step}/{n_steps}] Feeding summary CSV...", flush=True)
        feeding_csv_path = self.write_feeding_summary(
            range_minutes=range_minutes,
            transform_licks=transform_licks,
        )
        print(f"  Done → {feeding_csv_path}", flush=True)

        step += 1
        print(f"\n[{step}/{n_steps}] Faceted feeding summary CSV...", flush=True)
        facet_csv_path = self.write_feeding_summary_facet(
            transform_licks=transform_licks,
        )
        if facet_csv_path is None:
            print("  Skipped — experiment is not faceted.", flush=True)
        else:
            print(f"  Done → {facet_csv_path}", flush=True)

        step += 1
        print(f"\n[{step}/{n_steps}] Feeding summary plot...", flush=True)
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
            "feeding_summary_facet": facet_csv_path,
            "feeding_summary_plot": plot_path,
        }

