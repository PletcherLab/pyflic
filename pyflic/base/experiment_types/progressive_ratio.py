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

from typing import Any

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
    }

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

    def output_manifest(self) -> list[str]:
        return ["feeding_summary.csv", "feeding_summary_facet.csv",
                "paired_yoked_diff.csv", "pr_cumulative_diff.csv",
                "pr_cumulative_diff.png", "summary.txt"]
