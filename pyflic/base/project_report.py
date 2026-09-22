"""The Project Report: pooled figures, pooled + mixed statistics, and a
per-Member summary table (ADR-0005).

Members never get their own figure sets here — that is what a Member's own
report is for.  The whole point of the Project level is that the figures are
*pooled*: one figure per metric drawn from the Combined Analysis, with the
per-Member detail collapsed into a single table.

The report renders with matplotlib, the same as the per-experiment report; the
vector Publication Figures are a separate path through plotnine.  The two share
the summarized, filtered data and nothing else.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from . import pubfigures
from .pdf_report import _figure_page, _table_page, _text_page


def member_table(project) -> pd.DataFrame:
    """One row per Member: what went in, and what was left out."""
    rows = []
    for name in project.member_names:
        status = project.member_status(name)
        rows.append({
            "Member": name,
            "DFMs": status["dfms"],
            "Chambers": status["chambers"] if status["chambers"] is not None else "—",
            "Analyzed": "yes" if status["analyzed"] else "NO",
            "Faceted": "yes" if status["faceted"] else "no",
            "Report": "yes" if status["report"] else "no",
        })
    return pd.DataFrame(rows)


def cover_text(project, summary: pd.DataFrame | None,
               missing: list[str]) -> str:
    lines = [
        f"Project      : {project.name}",
        f"Directory    : {project.project_directory}",
        f"Experiment   : {project.experiment_type.display_name}",
        f"Layout       : {project.chamber_layout}",
        f"Members   : {len(project.member_names)}",
    ]
    if summary is not None:
        lines.append(f"Chambers     : {len(summary)} pooled")
        if "Treatment" in summary.columns:
            treatments = sorted(set(summary["Treatment"].astype(str)))
            lines.append(f"Treatments   : {', '.join(treatments)}")
    windows, labels = project.shared_windows()
    if windows:
        lines.append(f"Facets       : {', '.join(labels)}")
    else:
        lines.append("Facets       : none (whole recording)")
    if project.notes:
        lines.append("")
        lines.append(f"Notes: {project.notes}")
    lines.append("")
    lines.append(project.experiment_type.report_intro())
    if missing:
        lines.append("")
        lines.append(
            "OMITTED — no saved analysis, so these members are NOT part of "
            "any pooled number in this report:")
        for name in missing:
            lines.append(f"  - {name}")
    if project.warnings:
        lines.append("")
        lines.append("Notes on member differences:")
        for warning in project.warnings:
            lines.append(f"  - {warning}")
    return "\n".join(lines)


def _pooled_figure(project, plot_id: str, facet, binned, specs):
    """A pooled figure for *plot_id* from the Combined Analysis, or ``None``.

    Uses the same Spec+Style the Plot Editor saves, so the report's figure and
    the curated vector figure cannot drift apart in content — only in renderer.
    """
    info = pubfigures.PLOT_TYPES[plot_id]
    family = info["family"]
    source = pubfigures.frame_for(plot_id, facet, binned, project)
    if source is None or source.empty:
        return None
    label_order = None
    if facet is not None and "Facet" in facet.columns:
        label_order = list(dict.fromkeys(facet["Facet"].astype(str)))
    df = (pubfigures.timecourse_data(source, info["metric"])
          if family == pubfigures.FAMILY_TIMECOURSE
          else pubfigures.faceted_data(source, info["metric"], label_order))
    if df.empty:
        return None
    spec = specs.plots.get(plot_id)
    if spec is None:
        spec = pubfigures.default_spec(plot_id, well_a=_well_a_name(project))
        ## A type may say which Facets its report figures show by default
        ## (Progressive Ratio: Test only) — a default, never a gate.
        wanted = project.experiment_type.report_facets()
        if wanted and family == pubfigures.FAMILY_FACETED and "Phase" in df.columns:
            present = [w for w in wanted if w in set(df["Phase"].astype(str))]
            if present:
                spec.facets = list(present)
    ## Pooled figures mark their members by default: seeing the batch
    ## structure inside a pooled cloud is most of why one pools at all.
    if "Experiment" in df.columns and plot_id not in specs.plots:
        spec.mark_experiments = True
    return pubfigures.build_figure(plot_id, df, spec, specs.style_for(spec))


def _well_a_name(project) -> str:
    names = (project.design_global.get("well_names") or {})
    return str(names.get("A") or names.get("a") or "well A")


def write_project_report(project, path: str | Path | None = None, *,
                         ai_summary: bool = False, log=print) -> str:
    """Write the Project Report and return its path.

    Builds the Combined Analysis first when it is missing, so this one call is
    the whole Create-report button — and so a Batch Run's default script does
    the right thing on a Project nobody has combined yet.
    """
    combined_summary = os.path.join(project.analysis_path,
                                    f"{project.name}_Summary.csv")
    if not os.path.isfile(combined_summary):
        log("    building combined analysis first...")
        project.build_combined_analysis()

    summary, facet, missing = project.combined_frames()
    if summary is None:
        raise ValueError(
            f"No member in '{project.name}' has a saved analysis to pool.")

    specs = pubfigures.load_project_specs(project.project_directory)
    _facet_frame, binned = pubfigures.project_frames(project)
    report_set = project.experiment_type.report_set(project.chamber_layout)

    if path is None:
        path = os.path.join(project.project_directory,
                            f"{project.name}_report.pdf")
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(path) as pdf:
        _text_page(pdf, f"Project report — {project.name}",
                   cover_text(project, summary, missing))
        _table_page(pdf, "Members", member_table(project))

        for plot_id in report_set:
            if not pubfigures.plot_allowed(plot_id, project.chamber_layout,
                                           project.experiment_type.name):
                continue
            try:
                figure = _pooled_figure(project, plot_id, facet, binned, specs)
            except Exception as exc:  # noqa: BLE001
                _text_page(pdf, f"{plot_id} failed", str(exc))
                continue
            if figure is None:
                continue
            _figure_page(pdf, pubfigures.PLOT_TYPES[plot_id]["display"], figure)

        _text_page(pdf, "Statistics",
                   project.stats_text(summary, facet, project.combined_diff_frame()))

        exclusions = project.aggregated_exclusions()
        if len(exclusions):
            _table_page(pdf, "Excluded chambers (all members)", exclusions)
        else:
            _text_page(pdf, "Excluded chambers",
                       "No chambers were excluded in any member.")

        if ai_summary:
            try:
                from .ai import read_project_narrative

                text = read_project_narrative(project)
                if text:
                    _text_page(pdf, "AI summary", text)
            except Exception as exc:  # noqa: BLE001
                _text_page(pdf, "AI summary unavailable", str(exc))

    return path
