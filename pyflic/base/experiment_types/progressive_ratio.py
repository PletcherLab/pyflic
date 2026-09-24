"""The Progressive Ratio Experiment Type.

A two-well operant assay.  Each DFM's six chambers form three **Chamber
Groups** (1+2, 3+4, 5+6) that share one light circuit and one Treatment; in
each group one chamber is **Paired** (light contingent on its own feeding at
the sucrose well, well A) and the other **Yoked** (lit at the same moments,
regardless of its own behaviour).  The recording opens with a closed-loop
**Training** phase whose end the firmware marks per well in the data, and
differs for every group — which is why this type's Facets are data-derived
(ADR-0013) rather than minute cutoffs.

The config states the design with one key per DFM::

    dfms:
    - id: 1
      params: {pi_direction: left}     # side of the sucrose well (well A)
      paired_chambers: [1, 4, 5]       # exactly one chamber from each group
      chambers: {1: Ctrl, 2: Ctrl, 3: Exp, 4: Exp, 5: Exp, 6: Exp}

Yoked is the other chamber of the group and is never written.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from ..pr_breaking_point import DEFAULT_CONSTANTS as BREAK_CONSTANTS
from ..pr_light_qc import DEFAULT_CONSTANTS as LIGHT_QC_CONSTANTS
from .base import ExperimentType

#: Chamber Group -> its two chambers (two-well layout, chambers 1..6).
CHAMBER_GROUPS: dict[int, tuple[int, int]] = {1: (1, 2), 2: (3, 4), 3: (5, 6)}
ROLE_PAIRED = "paired"
ROLE_YOKED = "yoked"
PAIRED_KEY = "paired_chambers"


def group_of(chamber: int) -> int:
    """The Chamber Group (1-3) a two-well chamber (1-6) belongs to."""
    c = int(chamber)
    if not 1 <= c <= 6:
        raise ValueError(f"chamber must be 1..6 for a two-well DFM, got {chamber!r}")
    return (c + 1) // 2


def partner_of(chamber: int) -> int:
    """The other chamber in *chamber*'s Chamber Group."""
    c = int(chamber)
    group_of(c)
    return c + 1 if c % 2 == 1 else c - 1


def parse_paired_chambers(raw: Any) -> tuple[list[int], list[str]]:
    """``(paired_chambers, problems)`` for one DFM's ``paired_chambers`` value.

    Valid: a list of exactly three chamber indices, one from each Chamber
    Group.  Every problem is reported, none raised, so the linter and the
    loader share one description of what is wrong.
    """
    problems: list[str] = []
    if raw is None:
        return [], [f"'{PAIRED_KEY}' is required for experiment_type "
                    f"'ProgressiveRatio': one paired chamber per chamber group "
                    f"(one of 1-2, one of 3-4, one of 5-6)"]
    if isinstance(raw, (str, bytes)) or not hasattr(raw, "__iter__"):
        return [], [f"'{PAIRED_KEY}' must be a list of three chamber indices, got {raw!r}"]
    values: list[int] = []
    for v in raw:
        try:
            values.append(int(v))
        except (TypeError, ValueError):
            problems.append(f"'{PAIRED_KEY}' contains a non-integer chamber {v!r}")
    if problems:
        return [], problems
    bad = [v for v in values if not 1 <= v <= 6]
    if bad:
        problems.append(f"'{PAIRED_KEY}' names chamber(s) {bad} outside 1-6")
        return [], problems
    by_group: dict[int, list[int]] = {}
    for v in values:
        by_group.setdefault(group_of(v), []).append(v)
    for g, (a, b) in CHAMBER_GROUPS.items():
        got = by_group.get(g, [])
        if not got:
            problems.append(f"'{PAIRED_KEY}' names no chamber from group {g} "
                            f"(chambers {a}-{b})")
        elif len(got) > 1:
            problems.append(f"'{PAIRED_KEY}' names both chambers of group {g} "
                            f"({sorted(got)}); exactly one is paired, the other is yoked")
    if len(values) != len(set(values)):
        problems.append(f"'{PAIRED_KEY}' repeats a chamber: {values}")
    return sorted(set(values)), problems


@dataclass(frozen=True)
class ConstantField:
    """One Progressive Ratio design constant: how the Project Design dialog and
    the Config Editor show it, and what the type accepts for it.

    *group* is ``"switch"`` (true / false), ``"light_qc"`` or ``"break"``;
    *short_label* and *choices* are the Config Editor's words for a switch,
    whose row label must stay short and whose picker names what each value
    does.  *blank* says what an unset value means when the type has no
    default for it.
    """

    key: str
    label: str
    tooltip: str
    group: str
    integer: bool = False
    minimum: float | None = None
    maximum: float | None = None
    minimum_exclusive: bool = False
    blank: str | None = None
    short_label: str | None = None
    choices: tuple[str, str] = ("yes", "no")


#: Every Progressive Ratio constant beyond the three auto-filter cutoffs, in
#: the order the editors show them.  One list, so the two editors and the
#: validation cannot disagree about what exists or what is allowed.
PR_CONSTANT_FIELDS: tuple[ConstantField, ...] = (
    ConstantField(
        "require_training_complete", "Exclude groups whose training never completed",
        "require_training_complete — both chambers of a chamber group whose paired "
        "fly never finished training leave the analysis.",
        "switch", short_label="Training never completed",
        choices=("exclude the group", "keep the group")),
    ConstantField(
        "exclude_failed_pr_groups", "Exclude groups that fail the light QC",
        "exclude_failed_pr_groups — both chambers of a chamber group whose light "
        "followed the sensor rather than the fly (self-triggered light, "
        "implausible training) leave the analysis.  Off keeps them; "
        "pr_light_qc.csv lists them either way.",
        "switch", short_label="Light QC failed",
        choices=("exclude the group", "keep the group")),
    ConstantField(
        "pr_lick_free_run", "Lick-free run (light events)",
        "pr_lick_free_run — this many consecutive Test light events with no "
        "Sucrose Well licks between them make a group's light self-triggered "
        "(fails the group).",
        "light_qc", integer=True, minimum=1),
    ConstantField(
        "pr_trend_min_events", "Trend needs (light events)",
        "pr_trend_min_events — Test light events needed before the "
        "licks-per-event trend is judged; fewer is reported, never flagged.",
        "light_qc", integer=True, minimum=3),
    ConstantField(
        "pr_trend_min_rho", "Trend minimum rho",
        "pr_trend_min_rho — Spearman's rho of licks per light event against "
        "event number below which a group gets the 'no increasing trend' "
        "warning.",
        "light_qc", minimum=-1, maximum=1),
    ConstantField(
        "pr_resting_level_rise", "Resting level rise (counts)",
        "pr_resting_level_rise — a rise of the paired Sucrose Well's resting "
        "level above its first 30 minutes by this many counts is a warning; it "
        "is also the margin an 'elevated' well must clear.",
        "light_qc", minimum=0),
    ConstantField(
        "pr_resting_level_ratio", "Resting level ratio (×)",
        "pr_resting_level_ratio — a paired Sucrose Well resting at this many "
        "times the DFM's other Sucrose Wells is 'elevated' (a warning).",
        "light_qc", minimum=0, minimum_exclusive=True),
    ConstantField(
        "pr_break_gap_min", "Break gap (min)",
        "pr_break_gap_min — a pause longer than this many minutes ends a fly's "
        "responding.  The breaking point counts the paired fly's lick-backed Test "
        "light events before its first such pause; Sucrose Persistence is the "
        "time to either fly's last sucrose feeding event before one.  Default 120.",
        "break", minimum=0, minimum_exclusive=True),
    ConstantField(
        "pr_test_window_min", "Test window cap (min)",
        "pr_test_window_min — caps every chamber group's Test window at this many "
        "minutes after its own training end, so a group that trained late is not "
        "measured over less time than the rest.  Blank (the default) or 0: no cap.",
        "break", minimum=0, blank="no cap"),
)


def constant_problems(constants: Mapping[str, Any] | None) -> list[str]:
    """What is wrong with the Progressive Ratio constants a ``constants:`` block
    states — a switch that is not true or false, a threshold that is not a
    number or falls outside its range.  Keys it does not state are fine: the
    type's defaults fill them in.  Never raises."""
    problems: list[str] = []
    stated = dict(constants or {})
    for field in PR_CONSTANT_FIELDS:
        if field.key not in stated:
            continue
        value = stated[field.key]
        where = f"'constants.{field.key}'"
        if value is None:
            if field.blank is None:
                problems.append(f"{where} is empty; give a value or remove the key "
                                f"to use the type's default")
            continue
        if field.group == "switch":
            if not isinstance(value, bool):
                problems.append(f"{where} must be true or false, got {value!r}")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)):
            problems.append(f"{where} must be a number, got {value!r}")
            continue
        v = float(value)
        if field.integer and v != int(v):
            problems.append(f"{where} must be a whole number, got {value!r}")
        if field.minimum is not None and (
                v < field.minimum or (field.minimum_exclusive and v == field.minimum)):
            bound = "greater than" if field.minimum_exclusive else "at least"
            problems.append(f"{where} must be {bound} {field.minimum:g}, got {value!r}")
        if field.maximum is not None and v > field.maximum:
            problems.append(f"{where} must be at most {field.maximum:g}, got {value!r}")
    return problems


class ProgressiveRatioExperimentType(ExperimentType):
    name = "ProgressiveRatio"
    display_name = "Progressive Ratio"

    chamber_layout = "two_well"
    required_wells = ("A", "B")
    experiment_class = "pyflic.base.progressive_ratio_experiment:ProgressiveRatioExperiment"
    ## Facets are Training and Test, split at each Chamber Group's own training
    ## end (ADR-0013).  No cutoff list exists; ``facet_cutoffs`` is owned and
    ## must not appear in the config.
    facet_cutoffs = None
    facets_fixed = True
    data_derived_facets = True
    phase_labels = ("Training", "Test")

    default_constants = {
        "min_untransformed_licks_cutoff": 20,
        "max_med_duration_cutoff": 13.0,
        "max_events_cutoff": 150000.0,
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
    }

    def validate(self, global_cfg: dict | None) -> list[str]:
        """The base checks plus the Progressive Ratio constants' values
        (:func:`constant_problems`), so the loader, ``pyflic lint``, the Config
        Editor and the Project Design dialog refuse the same mistakes."""
        problems = super().validate(global_cfg)
        problems += constant_problems((global_cfg or {}).get("constants"))
        return problems

    def validate_dfm(self, dfm_id: int, node: dict | None,
                     chamber_assignments: dict | None) -> list[str]:
        node = node or {}
        paired, problems = parse_paired_chambers(node.get(PAIRED_KEY))
        out = [f"DFM {dfm_id}: {p}" for p in problems]
        assignments = {int(k): str(v) for k, v in (chamber_assignments or {}).items()}
        for g, (a, b) in CHAMBER_GROUPS.items():
            ta, tb = assignments.get(a), assignments.get(b)
            if ta is not None and tb is not None and ta != tb:
                out.append(
                    f"DFM {dfm_id}: chambers {a} and {b} form one chamber group "
                    f"and must share a treatment, got {ta!r} and {tb!r}")
        return out

    def report_facets(self) -> list[str] | None:
        return ["Test"]

    def report_intro(self) -> str:
        return ("A progressive-ratio assay: in each chamber group the paired fly "
                "earns light-driven stimulation by feeding at the sucrose well "
                "(well A) while its yoked partner is lit at the same moments "
                "regardless of behaviour. Training is closed-loop and ends at a "
                "time the data defines per group; the Test phase follows. The "
                "headline result is the paired-minus-yoked difference within "
                "each group.")

    def report_set(self, chamber_layout: str) -> list[str]:
        return ["timecourse_pr_diff", "faceted_licks", "faceted_events",
                "faceted_pi"]

    def project_results_blocks(self, project) -> list:
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
        settings = BreakSettings.from_constants(self.resolve_constants(project.design_global))
        half = (rl.CONTENT_W - 0.25) / 2

        def dot(frame, column, label, hline=None):
            return lambda: rc.dot_plot(frame, column, y_label=label, factors=factors,
                                       hline_at=hline)

        blocks: list = [rl.Heading("Breaking point", level=2)]
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

        test = (diff[diff["Facet"].astype(str) == "Test"]
                if diff is not None and not diff.empty else None)
        if test is not None and not test.empty:
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
                    rl.Plot(dot(test, "dLicksA", rc.metric_label("dLicksA", names), 0.0),
                            title=rc.metric_label("dLicksA", names)),
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

    def output_manifest(self) -> list[str]:
        return ["feeding_summary.csv", "feeding_summary_facet.csv",
                "paired_yoked_diff.csv", "pr_cumulative_diff.csv",
                "pr_cumulative_diff.png", "pr_light_qc.csv",
                "pr_light_events.csv", "pr_breaking_point.csv",
                "pr_still_responding.png", "summary.txt"]
