"""The Project Report: pooled figures, pooled + mixed statistics, and a
per-Member summary table (ADR-0005) — ``<project>/<name>_report.pdf``.

Members never get their own figure sets here — that is what a Member's own
report is for.  The whole point of the Project level is that the figures are
*pooled*: one figure per metric drawn from the Combined Analysis, with the
per-Member detail collapsed into tables.

Laid out by :mod:`pyflic.base.report_layout`, the same pages as a member's
experiment report: a cover with the verdicts at a glance and the contents,
then **Members**, **Quality control** (every exclusion, and the Experiment
Type's own flags — Progressive Ratio: the light QC), **Results** (the type's
report set of pooled figures, its own pooled figures from
:meth:`ExperimentType.project_results_blocks`, and the statistics as tables)
and an appendix of the Design's settings.  The report reads saved results
only; it never analyses a member.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from . import pubfigures
from . import report_content as rc
from . import report_layout as rl


def member_table(project) -> pd.DataFrame:
    """One row per Member: what went in, and what was left out."""
    exclusions = project.aggregated_exclusions()
    excluded = (exclusions.groupby("Experiment").size().to_dict()
                if not exclusions.empty else {})
    rows = []
    for name in project.member_names:
        status = project.member_status(name)
        rows.append({
            "Member": name,
            "DFMs": status["dfms"],
            "Chambers analysed": status["chambers"] if status["chambers"] is not None else "—",
            "Excluded": int(excluded.get(name, 0)),
            "Analysed": ("re-run needed" if status.get("stale") else
                         "yes" if status["analyzed"] else "NO"),
            "Faceted": "yes" if status["faceted"] else "no",
            "Report": "yes" if status["report"] else "no",
        })
    return pd.DataFrame(rows)


def _facets_line(project) -> str:
    etype = project.experiment_type
    if getattr(etype, "data_derived_facets", False):
        return (f"{' / '.join(etype.phase_labels)} — split at each chamber group's "
                f"own training end")
    windows, labels = project.shared_windows()
    return ", ".join(labels) if windows else "none (whole recording)"


def cover_text(project, summary: pd.DataFrame | None,
               missing: list[str]) -> str:
    """The cover as plain text (the AI payload reads this)."""
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
    lines.append(f"Facets       : {_facets_line(project)}")
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
    the curated vector figure cannot drift apart in content — only in size.
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
    if path is None:
        path = os.path.join(project.project_directory, f"{project.name}_report.pdf")
    path = str(path)
    build_project_report(project, ai_summary=ai_summary).save(path)
    return path


def build_project_report(project, *, ai_summary: bool = False) -> rl.ReportDocument:
    """The Project Report as a :class:`~pyflic.base.report_layout.ReportDocument`."""
    summary, facet, missing = project.combined_frames()
    if summary is None:
        raise ValueError(
            f"No member in '{project.name}' has a saved analysis to pool.")
    doc = rl.ReportDocument("Project report", project.name)
    doc.add(_cover(project, summary, missing))
    doc.add(rl.PageBreak(), rl.Heading("Members"),
            rl.Paragraph("Each member is one experiment directory; the pooled figures and "
                         "statistics below stack their saved, filtered summaries."),
            rl.Table(member_table(project), caption="Members",
                     status={"Analysed": _analysed_tone}))
    doc.add(rl.Heading("Quality control"), _qc_blocks(project))
    doc.add(rl.PageBreak(), rl.Heading("Results"), _results_blocks(project, summary, facet))
    if ai_summary:
        doc.add(_ai_blocks(project))
    doc.add(rl.Heading("Appendix — Analysis settings", numbered=False),
            rc.settings_blocks(project.design_global,
                               project.experiment_type.resolve_constants(
                                   project.design_global)))
    return doc


def _analysed_tone(value: Any) -> str | None:
    s = str(value)
    if s == "yes":
        return "ok"
    if s.startswith("re-run"):
        return "warning"
    if s == "NO":
        return "failed"
    return None


# ---------------------------------------------------------------------------
# Cover
# ---------------------------------------------------------------------------

def _cover(project, summary: pd.DataFrame, missing: list[str]) -> list[Any]:
    etype = project.experiment_type
    n_members = len(project.member_names)
    lines = [f"{etype.display_name} · {project.chamber_layout.replace('_', '-')} · "
             f"{n_members} member{'s' if n_members != 1 else ''}"]
    rows: list[tuple[str, str]] = [
        ("Directory", str(project.project_directory)),
        ("Members", ", ".join(project.member_names) or "none"),
        ("Chambers pooled", str(len(summary))),
    ]
    if "Treatment" in summary.columns:
        counts = summary["Treatment"].astype(str).value_counts()
        rows.append(("Treatments", " · ".join(f"{t}: {n} chambers"
                                              for t, n in counts.items())))
    factors = project.design_global.get("experimental_design_factors") or {}
    if factors:
        rows.append(("Design factors", " · ".join(
            f"{f} ({', '.join(str(v) for v in levels or [])})"
            for f, levels in factors.items())))
    names = project.design_global.get("well_names") or {}
    if project.chamber_layout == "two_well" and names:
        rows.append(("Wells", f"A = {names.get('A', 'A')}, B = {names.get('B', 'B')}"))
    rows.append(("Facets", _facets_line(project)))
    if project.notes:
        rows.append(("Notes", str(project.notes)))
    blocks: list[Any] = [rl.Cover("Project report", project.name, lines),
                         rl.KeyValues(rows)]
    intro = etype.report_intro()
    if intro and not getattr(etype, "is_custom", False):
        blocks.append(rl.Paragraph(intro, size=rl.SIZE_SMALL + 0.5, color=rl.MUTED))
    blocks.append(rl.Heading("At a glance", level=2, numbered=False))
    if missing:
        blocks.append(rl.Callout(
            f"{len(missing)} member(s) have no saved analysis and are NOT in any pooled "
            f"number here: {', '.join(missing)}.", tone="failed", title="Members"))
    else:
        blocks.append(rl.Callout(f"All {n_members} member(s) are analysed and pooled.",
                                 tone="ok", title="Members"))
    exclusions = project.aggregated_exclusions()
    if exclusions.empty:
        blocks.append(rl.Callout("No chamber was excluded in any member.", tone="ok",
                                 title="Exclusions"))
    else:
        manual = int((exclusions["Source"] == "manual").sum())
        auto = int((exclusions["Source"] == "auto").sum())
        blocks.append(rl.Callout(
            f"{len(exclusions)} chamber(s) excluded across members — {manual} by hand, "
            f"{auto} by the design's cutoffs.", tone="info", title="Exclusions"))
    blocks.extend(_flagged_callouts(project))
    if project.warnings:
        blocks.append(rl.Callout("; ".join(project.warnings), tone="warning",
                                 title="Member differences"))
    blocks.append(rl.Contents())
    return blocks


def _flagged_callouts(project) -> list[Any]:
    flagged = project.flagged_groups()
    if flagged is None:
        return []
    if flagged.empty:
        return [rl.Callout("Every chamber group's light followed its paired fly.",
                           tone="ok", title="Light QC")]
    excluded = flagged[flagged["Status"] == "excluded"]
    retained = flagged[flagged["Status"].str.startswith("retained")]
    warned = flagged[flagged["Status"].str.startswith("kept")]
    parts = []
    if len(excluded):
        parts.append(f"{len(excluded)} chamber group(s) failed and were excluded")
    if len(retained):
        parts.append(f"{len(retained)} failed but are in the pooled numbers "
                     f"(exclude_failed_pr_groups off)")
    if len(warned):
        parts.append(f"{len(warned)} carry warnings")
    tone = "failed" if len(excluded) or len(retained) else "warning"
    return [rl.Callout("; ".join(parts) + ".  See Quality control.", tone=tone,
                       title="Light QC")]


# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------

def _qc_blocks(project) -> list[Any]:
    blocks: list[Any] = [rl.Heading("Excluded chambers", level=2)]
    exclusions = project.aggregated_exclusions()
    if exclusions.empty:
        blocks.append(rl.Callout("No chamber was excluded in any member.", tone="ok"))
    else:
        view = exclusions.copy()
        view["Source"] = view["Source"].map({"manual": "by hand",
                                             "auto": "automatic"}).fillna(view["Source"])
        blocks += [
            rl.Paragraph("By hand: each member's remove_chambers.csv for the design's "
                         "exclusion group.  Automatic: the design's constants: cutoffs, "
                         "applied by each member's basic analysis."),
            rl.Table(view, caption="Chambers left out of the pooled results"),
        ]
    flagged = project.flagged_groups()
    if flagged is not None:
        blocks.append(rl.Heading("Light QC — did the paired fly earn its light?", level=2))
        if flagged.empty:
            blocks.append(rl.Callout("No chamber group was flagged by the light QC.",
                                     tone="ok"))
        else:
            view = flagged.rename(columns={"LickFreeRunStartMin":
                                           "Lick-free from (min after training)"})
            blocks += [
                rl.Paragraph("Chamber groups whose light the firmware fired without the "
                             "paired fly's licks (failed), or whose Sucrose Well or lick "
                             "trend looks wrong (warnings), from every member's "
                             "pr_light_qc.csv."),
                rl.Table(view, caption="Flagged chamber groups",
                         formats={"Lick-free from (min after training)": "{:.0f}"},
                         status={"Status": rl.tone_of, "Verdict": rl.tone_of}),
            ]
    return blocks


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def _results_blocks(project, summary: pd.DataFrame, facet) -> list[Any]:
    specs = pubfigures.load_project_specs(project.project_directory)
    _facet_frame, binned = pubfigures.project_frames(project)
    report_set = project.experiment_type.report_set(project.chamber_layout)
    names = project.design_global.get("well_names") or {}
    two_well = project.chamber_layout == "two_well"

    blocks: list[Any] = [rl.Heading("Pooled figures", level=2),
                         rl.Paragraph("From the Combined Analysis, with the Plot Editor's "
                                      "saved styles; members are marked within each "
                                      "pooled cloud.")]
    pending_pair: list[rl.Plot] = []
    for plot_id in report_set:
        if not pubfigures.plot_allowed(plot_id, project.chamber_layout,
                                       project.experiment_type.name):
            continue
        info = pubfigures.PLOT_TYPES[plot_id]
        try:
            figure = _pooled_figure(project, plot_id, facet, binned, specs)
        except Exception as exc:  # noqa: BLE001
            blocks.append(rl.Callout(f"{info['display']}: could not be drawn ({exc}).",
                                     tone="failed"))
            continue
        if figure is None:
            continue
        if info["family"] == pubfigures.FAMILY_TIMECOURSE:
            blocks.append(rl.Plot(figure, title=info["display"], height=3.2))
        else:
            ## Faceted figures are publication-sized: two to a row.
            pending_pair.append(rl.Plot(figure, title=info["display"],
                                        width=(rl.CONTENT_W - 0.25) / 2))
            if len(pending_pair) == 2:
                blocks.append(rl.PlotRow(pending_pair, height=3.2))
                pending_pair = []
    if pending_pair:
        blocks.append(rl.PlotRow(pending_pair, height=3.2))
    blocks.extend(project.experiment_type.project_results_blocks(project))

    blocks.append(rl.Heading("Statistics", level=2))
    diff = project.combined_diff_frame()
    diff_rows = project.diff_comparison_rows(diff)
    if diff is not None:
        blocks += rc.stats_table(diff_rows, caption="Primary: paired − yoked difference "
                                                    "(one observation per chamber group)",
                                 well_names=names, show_phase=True)
        blocks += rc.zero_test_table(project.diff_zero_rows(diff),
                                     caption="Paired − yoked difference against zero, "
                                             "per treatment",
                                     well_names=names, show_phase=True)
    rows = project.comparison_rows(summary, facet)
    caption = ("Per-chamber metrics (secondary)" if diff is not None
               else "Treatment comparisons, per chamber")
    blocks += rc.stats_table(rows, caption=caption, well_names=names, show_phase=True,
                             two_well=two_well)
    return blocks


def _ai_blocks(project) -> list[Any]:
    try:
        from .ai import read_project_narrative

        text = read_project_narrative(project)
    except Exception as exc:  # noqa: BLE001
        return [rl.Heading("AI summary"),
                rl.Callout(f"The AI summary is unavailable: {exc}", tone="warning")]
    if not text:
        return []
    return [rl.PageBreak(), rl.Heading("AI summary"),
            rl.Callout("Written by a language model from the Combined Analysis; check "
                       "every claim against the tables above.", tone="warning"),
            rl.Paragraph(text)]
