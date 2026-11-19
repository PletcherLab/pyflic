"""The Ratio analysis: Training, Test, breaking point and Sucrose Persistence.

What the Progressive Ratio type ran, as a module any type expecting a ratio
Paradigm (ProgressiveRatio or FixedRatio) may include: Chamber Groups from
the config or the Opto Program's Linkage, Facets split at each group's own
training end (ADR-0013), the light QC of the paired fly, the Paired-Yoked
Difference, the breaking point (ADR-0014) and Sucrose Persistence
(``ProgressiveRatioExperiment``).  The Chamber Group helpers it reads are in
:mod:`pyflic.base.experiment_types.progressive_ratio`.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..experiment_types import progressive_ratio as pr
from ..pr_breaking_point import DEFAULT_CONSTANTS as BREAK_CONSTANTS
from ..pr_light_qc import DEFAULT_CONSTANTS as LIGHT_QC_CONSTANTS
from .base import AnalysisModule


class RatioModule(AnalysisModule):
    id = "ratio"
    title = "Progressive Ratio"
    summary = ("Chamber Groups, Training and Test split at each group's training end, "
               "the paired fly's light QC, the Paired-Yoked Difference, the breaking "
               "point and Sucrose Persistence.")

    needs_layout = "two_well"
    needs_light = True
    needs_paradigms = ("ProgressiveRatio", "FixedRatio")
    conflicts = ("yoked",)
    conflict_reason = "the Ratio analysis keeps its own Paired-Yoked Difference"
    experiment_class = "pyflic.base.progressive_ratio_experiment:ProgressiveRatioExperiment"
    required_wells = ("A", "B")
    default_constants = {
        ## A Chamber Group whose paired fly never finished training has no
        ## Test phase; by default both its chambers leave the analysis through
        ## the ordinary auto-removal path so the rule is stated once.
        "require_training_complete": True,
        ## Light QC (did the paired fly earn its light?): a group whose light
        ## followed the sensor rather than the fly leaves the same way, unless
        ## exclude_failed_pr_groups is switched off.  See pr_light_qc.
        **LIGHT_QC_CONSTANTS,
        ## Breaking point and Sucrose Persistence (ADR-0014): the pause, in
        ## minutes, that ends a fly's responding.  pr_test_window_min (a cap
        ## on every group's Test window) has no default: unset means off.
        **BREAK_CONSTANTS,
        ## The Simulated Yoked Breaking Point (ADR-0023), opt-in.
        "pr_simulated_yoked_bp": False,
    }
    plots = ("timecourse_pr_diff", "scatter_pr_breaking_point", "dot_pr_breaking_point")
    outputs = ("paired_yoked_diff.csv", "pr_cumulative_diff.csv",
               "pr_cumulative_diff.png", "pr_light_qc.csv", "pr_light_events.csv",
               "pr_breaking_point.csv", "pr_still_responding.png")

    ## Facets are Training and Test, split at each Chamber Group's own training
    ## end (ADR-0013).  No cutoff list exists.
    data_derived_facets = True
    phase_labels = ("Training", "Test")
    report_facets = ("Test",)

    def constant_problems(self, constants: Mapping[str, Any] | None) -> list[str]:
        """The Progressive Ratio constants' values (:func:`pr.constant_problems`),
        so the loader, ``pyflic lint``, the Config Editor and the Project Design
        dialog refuse the same mistakes."""
        return pr.constant_problems(constants)

    def complete_dfm_node(self, dfm_id: int, node: dict, *, params=None,
                          dfm_program=None) -> tuple[dict, list[str]]:
        """The Chamber Groups the Opto Program gives when the entry leaves
        them out (:func:`pr.resolve_chamber_groups`): ``paired_chambers`` for the
        paired layout, as before solo groups, and the groups themselves
        (:data:`pr.RESOLVED_GROUPS_KEY`) whenever the program decided them.
        When the config states them and the program disagrees, say so and
        keep the config's."""
        if dfm_program is None:
            return node, []
        result = pr.resolve_chamber_groups(
            node, dfm_program, chamber_sets=getattr(params, "chamber_sets", None),
            pi_direction=getattr(params, "pi_direction", "left"), dfm_id=dfm_id)
        if result.source == "program" and result.groups is not None:
            node = {**node, pr.RESOLVED_GROUPS_KEY: [[g.paired, g.yoked]
                                                     for g in result.groups]}
            if sorted(g.chambers for g in result.groups) == sorted(pr.CHAMBER_GROUPS.values()):
                ## The paired layout: written as the config would state it.
                node[pr.PAIRED_KEY] = sorted(g.paired for g in result.groups)
        return node, list(result.notes)

    def validate_dfm(self, dfm_id: int, node: Mapping | None,
                     chamber_assignments: Mapping | None) -> list[str]:
        result = pr.resolve_chamber_groups(node, dfm_id=dfm_id)
        out = [f"DFM {dfm_id}: {p}" for p in result.problems]
        assignments = {int(k): str(v) for k, v in (chamber_assignments or {}).items()}
        ## Both chambers of a pair share a treatment; the paired layout's
        ## pairs are checked even when paired_chambers is wrong or missing.
        groups = (result.groups if result.groups is not None
                  else pr.paired_layout([a for a, _b in pr.CHAMBER_GROUPS.values()])
                  if pr.parse_layout((node or {}).get(pr.GROUPS_KEY))[0] != pr.GROUPS_SINGLE
                  else ())
        for g in groups:
            if g.solo:
                continue
            a, b = g.chambers
            ta, tb = assignments.get(a), assignments.get(b)
            if ta is not None and tb is not None and ta != tb:
                out.append(
                    f"DFM {dfm_id}: chambers {a} and {b} form one chamber group "
                    f"and must share a treatment, got {ta!r} and {tb!r}")
        return out

    def project_results_blocks(self, experiment_type, project) -> list:
        """The pooled breaking point (ADR-0014) — the still-responding curve
        first, then one point per Chamber Group with the mixed model and the
        log-rank test beside the pooled test — and the pooled Paired-Yoked
        Difference in the Test phase."""
        from .. import report_content as rc
        from .. import report_layout as rl
        from ..analytics import as_bool, breaking_point_comparisons
        from ..pr_breaking_point import BreakSettings

        diff = project.combined_diff_frame()
        bp = project.combined_breaking_point_frame()
        if (diff is None or diff.empty) and bp is None:
            return []
        names = project.design_global.get("well_names") or {}
        well_a = names.get("A") or "well A"
        factors = list((project.design_global.get("experimental_design_factors") or {}))
        settings = BreakSettings.from_constants(
            experiment_type.resolve_constants(project.design_global))
        half = (rl.CONTENT_W - 0.25) / 2

        def dot(frame, column, label, hline=None):
            return lambda: rc.dot_plot(frame, column, y_label=label, factors=factors,
                                       hline_at=hline)

        blocks: list = [rl.Heading("Fixed ratio: activations earned" if pr.all_fixed_ratio(bp)
                                   else "Breaking point", level=2)]
        if bp is None:
            blocks.append(rl.Callout(
                "No member has a breaking point table (pr_breaking_point.csv): their "
                "analyses predate it.  Re-run the members' basic analysis.",
                tone="warning"))
        elif bp.empty:
            blocks.append(rl.Callout("No chamber group in the pooled analysis has a Test "
                                     "phase.", tone="warning"))
        else:
            censored = int(as_bool(bp["Censored"]).sum()) if "Censored" in bp.columns else 0
            window = ("" if settings.test_window_min is None else
                      f"  Every Test window was capped at {settings.test_window_min:g} "
                      f"minutes (pr_test_window_min).")
            blocks += [
                rl.Paragraph(
                    f"The number of lick-backed Test light events a paired fly completed "
                    f"before its first pause longer than {settings.gap_min:g} minutes "
                    f"(pr_break_gap_min), pooled across members: one observation per "
                    f"chamber group.  {censored} of {len(bp)} groups were still responding "
                    f"when their Test window ended, so their counts are lower bounds "
                    f"(censored); the curve and the log-rank test treat them as such, the "
                    f"t-tests and the mixed model enter them as observed.{window}"),
                rl.Plot(lambda: rc.still_responding_plot(bp, base_font_size=9.5),
                        height=3.4, title="Still responding"),
                rl.PlotRow([rl.Plot(lambda: rc.censored_dot_plot(
                    bp, "BreakingPoint", y_label="Light events earned", factors=factors),
                    title="Breaking point by treatment", width=half)], height=3.0),
                *rc.stats_table(breaking_point_comparisons(bp, mixed_p=project._mixed_p),
                                caption="Treatment comparisons: breaking point",
                                well_names=names, show_phase=False),
            ]
            requirement = pr.break_requirement_view(bp)
            if requirement is not None:
                blocks.append(rl.Table(
                    requirement, caption="Breaking point and the Requirement the last "
                                         "counted light event met, as the firmware "
                                         "recorded it (medians per treatment)",
                    formats={"Breaking point": "{:g}", "Break requirement": "{:g}"}))
            blocks += _simulated_yoked_blocks(project, bp, settings, factors, names, half)

        test = (diff[diff["Facet"].astype(str) == "Test"]
                if diff is not None and not diff.empty else None)
        if (test is None or test.empty) and not pr.pooled_yoked_controls(project):
            ## Decision 13: said once, where the difference would have been.
            blocks += [rl.Heading("Paired − yoked difference", level=2),
                       rl.Paragraph("The design has no yoked controls: every chamber "
                                    "group is one chamber, lit by its own fly alone, so "
                                    "there is no paired − yoked difference.")]
        if test is not None and not test.empty:
            ## A lick difference of transformed licks is on the fourth-root scale.
            licks = rc.metric_label("dLicksA", names, transformed=project.transform_licks)
            blocks += [
                rl.Heading("Paired − yoked difference, Test phase", level=2),
                rl.Paragraph("One point per chamber group, pooled across members: the "
                             "paired fly's value minus its yoked partner's over the Test "
                             f"phase.  Zero is the null; the statistics below test it, "
                             f"against zero within each treatment and between treatments.  "
                             f"Sucrose persistence is the time from training end to a fly's "
                             f"last {well_a} feeding event before a pause longer than "
                             f"{settings.gap_min:g} minutes."),
                rl.PlotRow([
                    rl.Plot(dot(test, "dLicksA", licks, 0.0), title=licks),
                    rl.Plot(dot(test, "dPI", rc.metric_label("dPI", names), 0.0),
                            title=rc.metric_label("dPI", names)),
                ], height=3.0),
            ]
            if "dPersistA" in test.columns and test["dPersistA"].notna().any():
                blocks.append(rl.PlotRow([
                    rl.Plot(dot(test, "dPersistA", rc.metric_label("dPersistA", names), 0.0),
                            title=rc.metric_label("dPersistA", names), width=half),
                ], height=3.0))
        return blocks


def _simulated_yoked_blocks(project, bp, settings, factors, names, half) -> list:
    """The pooled Simulated Yoked Breaking Point (ADR-0023): one point per
    Chamber Group for it and for ``dBreakingPoint``, with their treatment
    comparisons and ``dBreakingPoint`` tested against zero.  Nothing when no
    Member computed it (``pr_simulated_yoked_bp`` off)."""
    from .. import report_content as rc
    from .. import report_layout as rl

    sim = project.simulated_yoked_frame(bp)
    if sim is None:
        if "SimulatedYokedBreakingPoint" in bp.columns:
            return [rl.Heading("Simulated yoked breaking point", level=2),
                    rl.Callout("No chamber group could be replayed: the replay needs a "
                               "firmware 4 recording whose ratio Paradigm counts Contact "
                               "(see each member's summary.txt).", tone="warning")]
        return []
    simulated, difference, zero = project.simulated_yoked_rows(bp)
    return [
        rl.Heading("Simulated yoked breaking point", level=2),
        rl.Paragraph(
            f"The test paradigm replayed on each yoked fly's own sucrose-well contact, from "
            f"its group's training end and over the same Test window, and scored like the "
            f"paired fly's lights (pause longer than {settings.gap_min:g} minutes): the "
            f"breaking point background feeding alone would have earned.  The paired "
            f"fly's breaking point minus it is the contingent part of the paired fly's "
            f"number.  One point per chamber group ({len(sim)} groups)."),
        rl.PlotRow([
            rl.Plot(lambda: rc.censored_dot_plot(
                sim, "SimulatedYokedBreakingPoint", y_label="Simulated light events",
                censored_col="SimulatedYokedCensored", factors=factors),
                title="Simulated yoked breaking point", width=half),
            rl.Plot(lambda: rc.dot_plot(sim, "dBreakingPoint",
                                        y_label="Paired − simulated yoked",
                                        factors=factors, hline_at=0.0),
                    title="Paired − simulated yoked breaking point", width=half),
        ], height=3.0),
        *rc.stats_table(simulated, caption="Treatment comparisons: simulated yoked "
                                           "breaking point", well_names=names,
                        show_phase=False),
        *rc.stats_table(difference, caption="Treatment comparisons: paired − simulated "
                                            "yoked breaking point", well_names=names,
                        show_phase=False),
        *rc.zero_test_table(zero, caption="Paired − simulated yoked breaking point "
                                          "against zero", well_names=names),
    ]


RATIO = RatioModule()
