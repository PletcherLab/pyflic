"""
Single-file PDF report for an Experiment.

Combines:
  - Summary text (experiment type, DFMs, range, filters)
  - Treatment-level feeding summary table (key metrics only)
  - Feeding summary plot(s)
  - Binned metric plots
  - ANOVA / Tukey HSD output (when requested)

Uses matplotlib's ``PdfPages`` so no extra dependency is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from .experiment import Experiment


# Key columns to show in the PDF table (in display order).
# Columns not present in the DataFrame are silently skipped.
_TWO_WELL_COLS = [
    "DFM", "Chamber", "Treatment",
    "PI", "EventPI",
    "LicksA", "LicksB",
    "EventsA", "EventsB",
    "MedDurationA", "MedDurationB",
]
_SINGLE_WELL_COLS = [
    "DFM", "Chamber", "Treatment",
    "Licks", "Events",
    "MedDuration", "MeanInt",
    "MedTimeBtw",
]


def _slim_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Select only the most useful columns for PDF display."""
    # Try two-well columns first, fall back to single-well
    for candidate in (_TWO_WELL_COLS, _SINGLE_WELL_COLS):
        cols = [c for c in candidate if c in df.columns]
        if len(cols) >= 4:
            # Also include any design-factor columns that come after Treatment
            if "Treatment" in df.columns:
                trt_idx = list(df.columns).index("Treatment")
                extra = [
                    c for c in df.columns[trt_idx + 1:]
                    if c not in cols and df[c].dtype == object
                ]
                cols = cols[:3] + extra + cols[3:]
            return df[cols]
    return df


def _text_page(pdf: PdfPages, title: str, body: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ax.text(0.02, 0.98, title, fontsize=16, fontweight="bold", va="top")
    ax.text(0.02, 0.94, body, fontsize=9, family="monospace", va="top", wrap=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _table_page(
    pdf: PdfPages,
    title: str,
    df: pd.DataFrame,
    max_rows: int = 50,
) -> None:
    if df.empty:
        _text_page(pdf, title, "(no rows)")
        return

    show = df.head(max_rows).copy()
    # Round numeric columns for readability
    for c in show.select_dtypes(include="number").columns:
        show[c] = show[c].round(3)

    n_cols = len(show.columns)
    # Scale font and page orientation based on column count
    landscape = n_cols > 8
    figsize = (11, 8.5) if landscape else (8.5, 11)
    fontsize = max(5.0, min(8.0, 80.0 / n_cols))

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=12)

    tbl = ax.table(
        cellText=show.astype(str).values,
        colLabels=list(show.columns),
        loc="upper center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.auto_set_column_width(list(range(n_cols)))
    tbl.scale(1.0, 1.2)

    # Style header row
    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_text_props(fontweight="bold")
        cell.set_facecolor("#d9e2f3")

    if len(df) > max_rows:
        ax.text(
            0.5, 0.01,
            f"... {len(df) - max_rows} more row(s) omitted — see CSV for full data",
            fontsize=7, style="italic", ha="center", transform=ax.transAxes,
        )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _ggplot_to_mpl(fig_or_gg):
    """Convert a plotnine ggplot to a matplotlib Figure, or return as-is."""
    try:
        from plotnine import ggplot
        if isinstance(fig_or_gg, ggplot):
            return fig_or_gg.draw()
    except ImportError:
        pass
    return fig_or_gg


def _figure_page(pdf: PdfPages, title: str, fig) -> None:
    fig = _ggplot_to_mpl(fig)
    if fig is None:
        return
    fig.suptitle(title, fontsize=14, fontweight="bold")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_experiment_report(
    experiment: Experiment,
    path: str | Path | None = None,
    *,
    metrics: Sequence[str] = ("Licks", "Events", "MedDuration"),
    binsize_min: float = 30.0,
    range_minutes: Sequence[float] = (0, 0),
    transform_licks: bool = True,
    include_comparison: bool = True,
) -> Path:
    """
    Write a PDF report summarising *experiment* to *path*.

    If *path* is None and ``experiment.analysis_dir`` is set, the report is
    written there as ``experiment_report.pdf``.
    """
    if path is None:
        out_dir = experiment.analysis_dir
        if out_dir is None:
            raise ValueError("path must be given when experiment has no project_dir")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "experiment_report.pdf"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(path) as pdf:
        _text_page(pdf, "Experiment summary", experiment.summary_text(include_qc=False))

        summary_df = experiment.feeding_summary(
            range_minutes=range_minutes, transform_licks=transform_licks,
        )
        _table_page(pdf, "Feeding summary (key metrics)", _slim_summary(summary_df))

        # Feeding summary plot
        try:
            fig = experiment.plot_feeding_summary(
                range_minutes=range_minutes, transform_licks=transform_licks,
            )
            _figure_page(pdf, "Feeding summary", fig)
        except Exception as exc:  # pragma: no cover
            _text_page(pdf, "Feeding summary plot failed", str(exc))

        # Per-metric binned plots
        _count_metrics = {"Licks", "Events", "PI", "EventPI"}
        for metric in metrics:
            try:
                two_well_mode = "total" if metric in _count_metrics else "A"
                fig = experiment.plot_binned_metric_by_treatment(
                    metric=metric,
                    two_well_mode=two_well_mode,
                    binsize_min=binsize_min,
                    range_minutes=range_minutes,
                    transform_licks=transform_licks,
                )
                _figure_page(pdf, f"Binned {metric}", fig)
            except Exception as exc:  # pragma: no cover
                _text_page(pdf, f"Binned {metric} failed", str(exc))

        # Optional: ANOVA-style comparison on a representative metric
        if include_comparison:
            try:
                from .analytics import compare_treatments
                res = compare_treatments(
                    experiment,
                    metric=metrics[0] if metrics else "MedDuration",
                    range_minutes=range_minutes,
                    transform_licks=transform_licks,
                )
                _text_page(
                    pdf,
                    f"{res.model} — {res.metric}",
                    f"formula: {res.formula}\nn={res.n_observations}",
                )
                _table_page(pdf, f"{res.model} table — {res.metric}", res.table)
                if res.posthoc is not None:
                    _table_page(pdf, f"Tukey HSD — {res.metric}", res.posthoc)
            except Exception as exc:  # pragma: no cover
                _text_page(pdf, "Statistical comparison failed", str(exc))

    return path
