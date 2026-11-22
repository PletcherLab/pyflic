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
from .paradigm_families import OPEN_LOOP


def member_table(project) -> pd.DataFrame:
    """One row per Member: what went in, and what was left out, and the
    Generation of its recordings, read from its saved outputs.  A Member
    without Training information (generation 2, decision 15) says so in a
    ``Note`` column, present when any Member needs one.  A ``Data folder``
    column, present when any Member names its ``data_folder:`` (ADR-0021),
    gives where each recording was read: the resolved path and the key as
    written, or ``data/``."""
    from .light_summary import NO_TRAINING_INFORMATION

    exclusions = project.aggregated_exclusions()
    excluded = (exclusions.groupby("Experiment").size().to_dict()
                if not exclusions.empty else {})
    no_training = set(project.generation_2_members())
    locations = {name: project.member_data_location(name) for name in project.member_names}
    referenced = any(location.referenced for location in locations.values())
    rows = []
    for name in project.member_names:
        status = project.member_status(name)
        generations = project.member_generations(name) or ([2] if name in no_training else [])
        row = {
            "Member": name,
            "DFMs": status["dfms"],
            "Generation": ", ".join(str(g) for g in generations) or "—",
            "Chambers analysed": status["chambers"] if status["chambers"] is not None else "—",
            "Excluded": int(excluded.get(name, 0)),
            "Analysed": ("not covered" if status.get("covered") is False else
                         "re-run needed" if status.get("stale") else
                         "yes" if status["analyzed"] else "NO"),
            "Faceted": "yes" if status["faceted"] else "no",
            "Report": "yes" if status["report"] else "no",
        }
        if referenced:
            location = locations[name]
            row["Data folder"] = location.describe() if location.referenced else "data/"
        if no_training:
            row["Note"] = NO_TRAINING_INFORMATION if name in no_training else ""
        rows.append(row)
    return pd.DataFrame(rows)


def _facets_line(project) -> str:
    etype = project.experiment_type
    if getattr(etype, "data_derived_facets", False):
        return (f"{' / '.join(etype.phase_labels)} — split at each chamber group's "
                f"own training end")
    from .program_facets import SOURCE_LABELS, is_program_source

    source = getattr(project, "facet_source", "cutoffs")
    if is_program_source(source):
        ## Each member DFM's own Program Intervals; the labels are its Paradigms.
        return (f"{SOURCE_LABELS[source]} (facet_source: {source}) — each DFM's own "
                f"program, pooled by Paradigm label")
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
    ## Not project.warnings: some of those quote what members typed in
    ## about:, which never goes on a report or into a prompt (ADR-0024).
    if project.report_warnings:
        lines.append("")
        lines.append("Notes on member differences:")
        for warning in project.report_warnings:
            lines.append(f"  - {warning}")
    return "\n".join(lines)


def _pooled_figure(project, plot_id: str, facet, binned, specs, *, source=None):
    """A pooled figure for *plot_id* from the Combined Analysis, or ``None``.

    Uses the same Spec+Style the Plot Editor saves, so the report's figure and
    the curated vector figure cannot drift apart in content — only in size.
    *source* replaces the frame the plot draws from (the light table's rows
    of one Paradigm Family).  A breaking-point figure takes its tidy rows
    (:func:`pubfigures.breaking_point_data`), as the Plot Editor draws it.
    """
    info = pubfigures.PLOT_TYPES[plot_id]
    family = info["family"]
    if source is None:
        source = pubfigures.frame_for(plot_id, facet, binned, project)
    if source is None or source.empty:
        return None
    ## A figure from another table (the light table) orders its own Facets.
    label_order = pubfigures.facet_order(
        facet if pubfigures.source_of(plot_id) == "facet" else source)
    transformed = pubfigures.licks_transformed(project, plot_id)
    if family == pubfigures.FAMILY_TIMECOURSE:
        df = pubfigures.timecourse_data(source, info["metric"], transformed=transformed)
    elif family == pubfigures.FAMILY_SCATTER:
        df = pubfigures.breaking_point_data(source)
    else:
        df = pubfigures.faceted_data(source, info["metric"], label_order,
                                     transformed=transformed)
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
        path = report_path(project)
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    build_project_report(project, ai_summary=ai_summary).save(path)
    return path


def report_path(project) -> str:
    """``<project>/<name>_report.pdf``; a Slice's is
    ``<project>/slices/<slice>/<name>_<slice>_report.pdf`` (ADR-0020)."""
    if project.slice is None:
        return os.path.join(project.project_directory, f"{project.name}_report.pdf")
    from . import slices

    root = slices.output_dir(project.project_directory, project.slice.name)
    return str(root / f"{project.name}_{project.slice.name}_report.pdf")


def write_all_project_reports(project, *, ai_summary: bool = False,
                              log=print) -> list[str]:
    """The Project Report of the whole recording and of every Slice
    (ADR-0020).  A Slice no Member has been run for is skipped and logged."""
    paths = [write_project_report(project.whole, ai_summary=ai_summary, log=log)]
    for view in project.slice_views()[1:]:
        try:
            paths.append(write_project_report(view, ai_summary=ai_summary, log=log))
        except ValueError as err:
            log(f"[{view.view_label()}] report skipped: {err}")
    return paths


def build_project_report(project, *, ai_summary: bool = False) -> rl.ReportDocument:
    """The Project Report as a :class:`~pyflic.base.report_layout.ReportDocument`."""
    summary, facet, missing = project.combined_frames()
    if summary is None:
        raise ValueError(
            f"No member in '{project.name}' has a saved analysis to pool.")
    doc = rl.ReportDocument("Project report", _title(project))
    doc.add(_cover(project, summary, missing, project.unfaceted_members(summary, facet)))
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

def _title(project) -> str:
    if project.slice is None:
        return project.name
    return f"{project.name} · Slice {project.slice.name}"


def _cover(project, summary: pd.DataFrame, missing: list[str],
           unfaceted: list[str] = ()) -> list[Any]:
    etype = project.experiment_type
    n_members = len(project.member_names)
    lines = [f"{etype.display_name} · {project.chamber_layout.replace('_', '-')} · "
             f"{n_members} member{'s' if n_members != 1 else ''}"]
    rows: list[tuple[str, str]] = [
        ("Directory", str(project.project_directory)),
        ("Members", ", ".join(project.member_names) or "none"),
        ("Chambers pooled", str(len(summary))),
    ]
    left_out = project.not_covered()
    if project.slice is not None:
        rows.insert(1, ("Slice", f"{project.slice.name}, minutes "
                                 f"{project.slice.bounds_label()}: each Member analysed as if "
                                 f"its recording were only these minutes, with the Slice's "
                                 f"own exclusions"))
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
    blocks: list[Any] = [rl.Cover("Project report", _title(project), lines),
                         rl.KeyValues(rows)]
    intro = etype.report_intro()
    if intro and not getattr(etype, "is_custom", False):
        blocks.append(rl.Paragraph(intro, size=rl.SIZE_SMALL + 0.5, color=rl.MUTED))
    blocks.append(rl.Heading("At a glance", level=2, numbered=False))
    if missing:
        blocks.append(rl.Callout(
            f"{len(missing)} member(s) have no saved analysis and are NOT in any pooled "
            f"number here: {', '.join(missing)}.", tone="failed", title="Members"))
    elif not left_out:
        blocks.append(rl.Callout(f"All {n_members} member(s) are analysed and pooled.",
                                 tone="ok", title="Members"))
    if left_out:
        detail = "; ".join(f"{name}: {reason}" for name, reason in left_out.items())
        blocks.append(rl.Callout(
            f"{len(left_out)} member(s) do not cover this Slice and are left out of it: "
            f"{detail}.", tone="info", title="Not covered"))
    if unfaceted:
        blocks.append(rl.Callout(
            f"{len(unfaceted)} member(s) have no Facet table and are missing from every "
            f"Facet's numbers: {', '.join(unfaceted)}.  Re-run the analysis of any that "
            f"should have one.",
            tone="warning", title="Not faceted"))
    spread = project.open_end_spread_note()
    if spread:
        blocks.append(rl.Callout(spread + ".", tone="warning", title="Unequal spans"))
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
    if project.report_warnings:
        blocks.append(rl.Callout("; ".join(project.report_warnings), tone="warning",
                                 title="Member differences"))
    blocks.append(rl.Contents())
    return blocks


def _flagged_callouts(project) -> list[Any]:
    flagged = project.flagged_groups()
    if flagged is None:
        return []
    if flagged.empty:
        return [rl.Callout("Every group's light was earned by its licks.",
                           tone="ok", title="Light QC")]
    excluded = flagged[flagged["Status"] == "excluded"]
    retained = flagged[flagged["Status"].str.startswith("retained")]
    warned = flagged[flagged["Status"].str.startswith("kept")]
    parts = []
    if len(excluded):
        parts.append(f"{len(excluded)} group(s) failed and were excluded")
    if len(retained):
        parts.append(f"{len(retained)} failed but are in the pooled numbers "
                     f"(their exclusion switch is off)")
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
        blocks.append(rl.Heading("Light QC — was the light earned by licks?", level=2))
        if flagged.empty:
            blocks.append(rl.Callout("No group was flagged by the light QC.", tone="ok"))
        else:
            view = flagged.rename(columns={
                "LickFreeRunStartMin": "Lick-free from (min after training)",
                "UnexplainedOnsetMin": "Unexplained from (min)"})
            if not (view["Source"] == "Opto").any():
                view = view.drop(columns=["Wells", "Unexplained from (min)"])
            if not (view["Source"] == "PR").any():
                view = view.drop(columns=["Lick-free from (min after training)"])
            blocks += [
                rl.Paragraph("Groups whose light the firmware fired without the licks to "
                             "explain it (failed), or whose sensor, program or lick trend "
                             "looks wrong (warnings).  PR: a progressive-ratio chamber "
                             "group, from every member's pr_light_qc.csv.  Opto: a "
                             "linkage group of any optogenetic member, from "
                             "qc/opto/opto_light_qc.csv."),
                rl.Table(view, caption="Flagged groups",
                         formats={"Lick-free from (min after training)": "{:.0f}",
                                  "Unexplained from (min)": "{:.0f}"},
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

    ## The type's Control Comparisons lead the Results (ADR-0019); what led
    ## is not repeated below.
    lead, led = _control_blocks(project, summary, facet)
    blocks: list[Any] = [*lead, rl.Heading("Pooled figures", level=2),
                         rl.Paragraph("From the Combined Analysis, with the Plot Editor's "
                                      "saved styles; members are marked within each "
                                      "pooled cloud.")]
    pending_pair: list[rl.Plot] = []
    for plot_id in report_set:
        if not pubfigures.plot_allowed(plot_id, project.chamber_layout,
                                       project.experiment_type):
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
    ## One treatment pooled: nothing is compared between treatments, said
    ## once.  A paired − yoked difference is still tested against zero.
    levels = rc.treatment_levels(summary)
    single = len(levels) == 1
    if single:
        blocks.append(rc.single_treatment_note(levels[0]))
    diff = project.combined_diff_frame()
    if diff is not None and diff.empty:
        ## A design without yoked controls writes an empty table (decision
        ## 13); the type's own results say so once.
        diff = None
    diff_rows = project.diff_comparison_rows(diff)
    if diff is not None and led != LED_DIFFERENCE:
        if not single:
            blocks += rc.stats_table(diff_rows, caption="Primary: paired − yoked "
                                                        "difference (one observation "
                                                        "per chamber group)",
                                     well_names=names, show_phase=True,
                                     transformed=project.transform_licks)
        blocks += rc.zero_test_table(project.diff_zero_rows(diff),
                                     caption="Paired − yoked difference against zero, "
                                             "per treatment",
                                     well_names=names, show_phase=True,
                                     transformed=project.transform_licks)
    if not single:
        rows = project.comparison_rows(summary, facet)
        caption = ("Per-chamber metrics (secondary)" if diff is not None
                   else "Treatment comparisons, per chamber")
        blocks += rc.stats_table(rows, caption=caption, well_names=names, show_phase=True,
                                 two_well=two_well)
    blocks += _light_blocks(project, specs, compare=not single)
    if led != LED_YOKED:
        blocks += _yoked_blocks(project)
    return blocks


#: What :func:`_control_blocks` moved to the front: the Paired and yoked
#: module's comparison, or the Ratio module's Paired-Yoked Difference tables.
LED_YOKED, LED_DIFFERENCE = "yoked", "difference"


def _control_blocks(project, summary: pd.DataFrame, facet) -> tuple[list[Any], str | None]:
    """``(blocks, led)``: the type's Control Comparisons
    (:mod:`control_comparisons`), the Role comparison first, pooled across
    members, for the front of the Results; nothing under a type with no
    ``control:`` block.  A Role comparison of Paired and Yoked is the
    paired − yoked comparison the report already makes, moved here: the
    Paired and yoked module's (*led* :data:`LED_YOKED`) or the Ratio
    module's Paired-Yoked Difference tables (:data:`LED_DIFFERENCE`)."""
    from . import control_comparisons as cc

    etype = project.experiment_type
    roles, factor = cc.role_pair(etype), cc.treatment_control(etype)
    if roles is None and factor is None:
        return [], None
    frames = project._facet_frames(summary, facet)
    if facet is not None and not facet.empty and "Facet" in facet.columns:
        plot_frame, facet_col = facet, "Facet"
    else:
        plot_frame, facet_col = summary, None
    names = project.design_global.get("well_names") or {}
    two_well = project.chamber_layout == "two_well"
    factors = list((project.design_global.get("experimental_design_factors") or {}))
    blocks: list[Any] = []
    led: str | None = None
    if roles is not None and cc.is_paired_yoked(roles):
        if etype.has_module("yoked"):
            led = LED_YOKED
            blocks += _yoked_blocks(project) or [
                rl.Heading("Paired and yoked", level=2),
                rl.Paragraph("The type compares Paired with Yoked chambers, and no member's "
                             "linkage group lights a Yoked chamber with a Paired one.",
                             color=rl.MUTED, italic=True)]
        elif etype.has_module("ratio"):
            diff = project.combined_diff_frame()
            if diff is not None and not diff.empty:
                led = LED_DIFFERENCE
                blocks += [rl.Heading("Paired and yoked", level=2),
                           rl.Paragraph("The type compares Paired with Yoked chambers: the "
                                        "Paired-Yoked Difference, one observation per "
                                        "chamber group, pooled across members.")]
                if len(rc.treatment_levels(summary)) != 1:
                    blocks += rc.stats_table(project.diff_comparison_rows(diff),
                                             caption="Primary: paired − yoked difference "
                                                     "(one observation per chamber group)",
                                             well_names=names, show_phase=True,
                                             transformed=project.transform_licks)
                blocks += rc.zero_test_table(project.diff_zero_rows(diff),
                                             caption="Paired − yoked difference against "
                                                     "zero, per treatment",
                                             well_names=names, show_phase=True,
                                             transformed=project.transform_licks)
    elif roles is not None:
        blocks += cc.role_blocks(frames, roles, plot_frame=plot_frame, facet_col=facet_col,
                                 well_names=names, two_well=two_well, pooled=True)
    if factor is not None:
        blocks += cc.control_blocks(
            frames, factor, etype.control_level(project.design_global), factors=factors,
            plot_frame=plot_frame, facet_col=facet_col, well_names=names,
            two_well=two_well, pooled=True)
    return blocks, led


def _member_programs(project, light: pd.DataFrame | None) -> tuple[list, dict]:
    """``([(program, DFMs)], {(member, DFM): light keys})``: each member's
    ``data/Program.txt`` with the DFMs its saved light table records as
    generation 3, as :meth:`Project.light_headline_metrics` reads them, for
    the *Light behaviour* subsections (:func:`report_content.light_families`)."""
    from . import opto_program as op
    from .paradigm_state import GENERATION_PARADIGM

    if light is None or not {"Experiment", "DFM", "Generation"} <= set(light.columns):
        return [], {}
    generation = pd.to_numeric(light["Generation"], errors="coerce")
    programs: list = []
    keys: dict = {}
    for name, rows in light[generation == GENERATION_PARADIGM].groupby("Experiment", sort=False):
        try:
            path = op.find_program(project.member_data_location(str(name)).path)
            program = op.read_program(path) if path is not None else None
        except (op.ProgramError, OSError, ValueError):
            program = None
        ids = sorted({int(d) for d in pd.to_numeric(rows["DFM"], errors="coerce").dropna()})
        programs.append((program, ids))
        for d in ids:
            keys[(str(name), d)] = rc.program_light_keys(program, [d])
    return programs, keys


def _light_blocks(project, specs, *, compare: bool = True) -> list[Any]:
    """"Light behaviour": the stacked light table pooled as the feeding
    metrics are, one subsection per Paradigm Family the members' programs run
    (decision 9; fixed length by its branch, and the Acclimation) — a faceted
    figure per headline metric, open loop's *Feeding in light and dark*, and
    the treatment comparisons, per chamber — for a Project with optogenetic
    Members; nothing otherwise.  A Project whose members' programs run no
    family (recorded before firmware 4.0.0) keeps the four metrics of before
    in one section."""
    from .light_summary import behaviour_rows

    frames = project.combined_light_frames()
    light = frames.get("LightSummary")
    if light is None:
        return []
    programs, keys = _member_programs(project, light)
    families = rc.light_families(programs)
    if not families:
        return _light_headline_blocks(project, specs, frames, compare=compare)
    kept = behaviour_rows(light)
    kept_facet = behaviour_rows(frames.get("LightSummary_Facet"))
    paired_only = len(kept) < len(light)
    yoked = "Role" in kept.columns and (kept["Role"].astype(str) == "Yoked").any()
    blocks: list[Any] = [
        rl.Heading("Light behaviour", level=2),
        rl.Paragraph(
            "What the light did, from every optogenetic member's light table, one point per "
            "chamber, under each Paradigm Family the members' programs run: each family's "
            "own measures, for the chambers that ran it.  A chamber whose window ran "
            "Paradigms of several families shows under each.  Members recorded before DFM "
            "firmware 4.0.0 have lit time only (from OptoCol1)."
            + ("  The statistics compare paired chambers: a yoked fly's light is its "
               "partner's." if paired_only else "")
            + ("  Yoked chambers keep their lit time, beside their partners'; the statistics "
               "of Activations and the other Activation measures compare the chambers that "
               "earn light (Paired, Self, Mutual)." if yoked and not paired_only else "")),
    ]
    for family in families:
        rows = family.rows(kept, keys)
        facet_rows = family.rows(kept_facet, keys)
        blocks += _light_family_blocks(project, specs, family, rows, facet_rows,
                                       compare=compare)
    return blocks


def _light_family_blocks(project, specs, family, rows: pd.DataFrame,
                         facet_rows: pd.DataFrame | None, *, compare: bool) -> list[Any]:
    """One family's subsection of the pooled *Light behaviour*."""
    blocks: list[Any] = [rl.Heading(f"Light behaviour: {family.title}", level=2)]
    if rows is None or rows.empty:
        return blocks + [rl.Paragraph(f"No pooled chamber ran it: {family.about}.",
                                      color=rl.MUTED, italic=True)]
    ## The figures' source, as the Plot Editor's: the faceted table when
    ## there is one.
    source = facet_rows if facet_rows is not None and not facet_rows.empty else rows
    metrics = [m for m in family.metrics
               if m in rows.columns and pd.to_numeric(rows[m], errors="coerce").notna().any()]
    missing = [m for m in family.metrics if m not in metrics]
    blocks.append(rl.Paragraph(
        f"{family.about[0].upper()}{family.about[1:]}."
        + (f"  Not defined here: {', '.join(rc.metric_label(m) for m in missing)}."
           if missing else "")))
    pending: list[rl.Plot] = []
    for metric in metrics:
        plot_id = f"faceted_{metric.lower()}"
        info = pubfigures.PLOT_TYPES[plot_id]
        try:
            figure = _pooled_figure(project, plot_id, None, None, specs, source=source)
        except Exception as exc:  # noqa: BLE001
            blocks.append(rl.Callout(f"{info['display']}: could not be drawn ({exc}).",
                                     tone="failed"))
            continue
        if figure is None:
            continue
        pending.append(rl.Plot(figure, title=info["y_label"],
                               width=(rl.CONTENT_W - 0.25) / 2))
        if len(pending) == 2:
            blocks.append(rl.PlotRow(pending, height=3.2))
            pending = []
    if pending:
        blocks.append(rl.PlotRow(pending, height=3.2))
    if family.key == OPEN_LOOP and project.experiment_type.has_module("light_dark"):
        facet_col = "Facet" if "Facet" in source.columns else None
        blocks.append(rl.Plot(lambda: rc.light_dark_plot(source, facet_col=facet_col),
                              height=3.2, title="Feeding in light and dark",
                              caption="Each open-loop chamber's licks per minute in its own "
                                      "light and in the dark, joined by a line; the diamond "
                                      "is the treatment mean.  The light feeding index is "
                                      "(lit − dark) / (lit + dark) of the two."))
    if compare and metrics:
        blocks += rc.light_comparison_blocks(
            metrics, rows,
            lambda keep: project._compare_frames(project._facet_frames(rows, facet_rows), keep),
            caption=f"Light behaviour, {family.title}: treatment comparisons, per chamber",
            show_phase=True)
    return blocks


def _light_headline_blocks(project, specs, frames: dict, *, compare: bool) -> list[Any]:
    """*Light behaviour* as before the Paradigm Families: the four metrics of
    before, for a Project whose members' programs run no family."""
    from .light_summary import behaviour_rows

    light = frames.get("LightSummary")
    headline = project.light_headline_metrics(light)
    paired_only = len(behaviour_rows(light)) < len(light)
    blocks: list[Any] = [
        rl.Heading("Light behaviour", level=2),
        rl.Paragraph(
            "What the light did, from every optogenetic member's light table: "
            "Activations (LED on-periods a well's own Paradigm started), lit time, "
            "Firmware Events and the minute Acclimation ended, one point per chamber.  "
            "Members recorded before DFM firmware 4.0.0 have lit time only (from "
            "OptoCol1); a measure no Paradigm in the Project defines is left out."
            + ("  The statistics compare paired chambers: a yoked fly's light is its "
               "partner's." if paired_only else "")),
    ]
    pending: list[rl.Plot] = []
    for metric in headline:
        plot_id = f"faceted_{metric.lower()}"
        info = pubfigures.PLOT_TYPES[plot_id]
        try:
            figure = _pooled_figure(project, plot_id, None, None, specs)
        except Exception as exc:  # noqa: BLE001
            blocks.append(rl.Callout(f"{info['display']}: could not be drawn ({exc}).",
                                     tone="failed"))
            continue
        if figure is None:
            continue
        pending.append(rl.Plot(figure, title=info["y_label"],
                               width=(rl.CONTENT_W - 0.25) / 2))
        if len(pending) == 2:
            blocks.append(rl.PlotRow(pending, height=3.2))
            pending = []
    if pending:
        blocks.append(rl.PlotRow(pending, height=3.2))
    if compare:
        rows = project.light_comparison_rows(light, frames.get("LightSummary_Facet"))
        blocks += rc.stats_table(rows, caption="Light behaviour: treatment comparisons, "
                                               "per chamber", show_phase=True)
    return blocks


def _yoked_blocks(project) -> list[Any]:
    """*Paired and yoked* (decision 12): the members' stacked yoked
    comparison (``<project>_YokedComparison.csv``), the paired fly's feeding
    minus its yoked partners' by treatment, with the mixed models beside the
    pooled tests; nothing without a yoked group, or under a type without the
    Paired and yoked module (ADR-0019)."""
    from . import yoked_report as yr

    if not project.experiment_type.has_module("yoked"):
        return []
    frame = project.combined_light_frames().get("YokedComparison")
    if frame is None or frame.empty:
        return []
    names = project.design_global.get("well_names") or {}
    tables = yr.yoked_table_blocks(frame, well_names=names,
                                   two_well=project.chamber_layout == "two_well",
                                   mixed_p=project._mixed_p, mixed_p0=project._mixed_p0,
                                   transformed=project.transform_licks)
    if not tables:
        return []
    factors = list((project.design_global.get("experimental_design_factors") or {}))
    return [rl.Heading("Paired and yoked", level=2),
            rl.Paragraph("A Linkage Group that lights a Paired chamber's fly by its own "
                         "feeding lights its Yoked chambers at the same moments, whatever "
                         "their flies do.  Per group, pooled across members: the paired "
                         "fly's feeding minus the mean of its yoked partners', by treatment "
                         "— is the paired fly different (against zero), and do treatments "
                         "differ?"),
            *tables,
            *yr.yoked_figure_blocks(frame, well_names=names, factors=factors,
                                    transformed=project.transform_licks)]


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
