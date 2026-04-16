"""
Single-file PDF report for an Experiment.

Combines:
  - Summary text (experiment type, DFMs, range, filters)
  - Treatment-level feeding summary table
  - Feeding summary plot(s)
  - Binned licks plot
  - Tukey HSD output (if statsmodels present and comparison requested)

Uses matplotlib's ``PdfPages`` so no extra dependency is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from .experiment import Experiment


def _text_page(pdf: PdfPages, title: str, body: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ax.text(0.02, 0.98, title, fontsize=16, fontweight="bold", va="top")
    ax.text(0.02, 0.94, body, fontsize=9, family="monospace", va="top", wrap=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _table_page(pdf: PdfPages, title: str, df: pd.DataFrame, max_rows: int = 45) -> None:
    if df.empty:
        _text_page(pdf, title, "(no rows)")
        return
    show = df.head(max_rows)
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left")
    tbl = ax.table(
        cellText=show.round(3).astype(str).values,
        colLabels=list(show.columns),
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.0, 1.25)
    if len(df) > max_rows:
        ax.text(
            0.02, 0.02,
            f"... {len(df) - max_rows} more row(s) omitted",
            fontsize=8, style="italic",
        )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _figure_page(pdf: PdfPages, title: str, fig) -> None:
    if hasattr(fig, "draw"):
        fig = fig.draw()
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
        _table_page(pdf, "Feeding summary (per chamber)", summary_df)

        # Feeding summary plot
        try:
            fig = experiment.plot_feeding_summary(
                range_minutes=range_minutes, transform_licks=transform_licks,
            )
            _figure_page(pdf, "Feeding summary", fig)
        except Exception as exc:  # pragma: no cover
            _text_page(pdf, "Feeding summary plot failed", str(exc))

        # Per-metric binned plots
        for metric in metrics:
            try:
                fig = experiment.plot_binned_metric_by_treatment(
                    metric=metric,
                    two_well_mode="total",
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
