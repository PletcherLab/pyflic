"""The Chamber Groups of the Ratio analysis (ADR-0019's ``ratio`` module).

The Progressive Ratio design, which the shipped ProgressiveRatio type and any
type including the ``ratio`` Analysis Module
(:mod:`pyflic.base.analysis_modules.ratio`) run: a two-well operant assay.
A DFM's six chambers form **Chamber Groups** that share one light circuit
and one Treatment.  In a paired design each group is
two chambers: one **Paired** (light contingent on its own feeding at the
sucrose well, well A) and the other **Yoked** (lit at the same moments,
regardless of its own behaviour).  A **solo** group is one chamber, Paired,
with no yoked control: PyMCU's reference design, where each chamber's light
answers to its own fly alone (decision 13 of
``docs/paradigm-analyses-plan.md``).  The recording opens with a closed-loop
**Training** phase whose end the firmware marks per well in the data (a flag
in the signal before firmware 4.0.0, the Sucrose Well's Phase from it), and
differs for every group — which is why the module's Facets are data-derived
(ADR-0013) rather than minute cutoffs.  The module covers both ratio
Paradigms of firmware 4.0.0, ProgressiveRatio and FixedRatio (decision 6 of
``docs/v4-data-plan.md``).

Where the groups come from (:func:`resolve_chamber_groups`):

* ``paired_chambers`` in the DFM's entry: the paired layout, chambers 1+2,
  3+4 and 5+6, with exactly one Paired chamber from each::

      dfms:
      - id: 1
        params: {pi_direction: left}     # side of the sucrose well (well A)
        paired_chambers: [1, 4, 5]       # exactly one chamber from each group
        chambers: {1: Ctrl, 2: Ctrl, 3: Exp, 4: Exp, 5: Exp, 6: Exp}

  Yoked is the other chamber of the group and is never written.
* ``chamber_groups: single`` in the DFM's entry: six solo groups.
* Neither, with ``Program.txt``: its Linkage.  A Linkage Group lighting two
  chambers is a pair whose chamber holding the Trigger Well is Paired; one
  lighting one chamber is a solo group.
* Neither, without ``Program.txt`` (or ``chamber_groups: paired``): the
  paired layout, whose ``paired_chambers`` is then required.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..constant_fields import ConstantField
from ..constant_fields import constant_problems as _constant_problems

#: The paired layout's Chamber Groups -> their two chambers (two-well layout,
#: chambers 1..6): what ``paired_chambers`` and ``chamber_groups: paired``
#: name.  Program.txt's Linkage may group chambers otherwise
#: (:func:`groups_from_program`).
CHAMBER_GROUPS: dict[int, tuple[int, int]] = {1: (1, 2), 2: (3, 4), 3: (5, 6)}
ROLE_PAIRED = "paired"
ROLE_YOKED = "yoked"
PAIRED_KEY = "paired_chambers"
#: The DFM entry's ``chamber_groups``: the layout when Program.txt does not
#: give the groups, ``paired`` (the default) or ``single``.
GROUPS_KEY = "chamber_groups"
GROUPS_PAIRED, GROUPS_SINGLE = "paired", "single"
GROUP_LAYOUTS: tuple[str, ...] = (GROUPS_PAIRED, GROUPS_SINGLE)
#: Where the Ratio module's ``complete_dfm_node`` leaves the
#: groups Program.txt gave, in the resolved DFM entry the loader keeps (never
#: in a file): ``[[paired, yoked or None], ...]`` in group order.
RESOLVED_GROUPS_KEY = "_chamber_groups"
#: The two-well layout's chambers.
N_CHAMBERS = 6


@dataclass(frozen=True, slots=True)
class ChamberGroup:
    """One Chamber Group of a DFM: its number (groups are numbered in
    chamber order), its Paired chamber and its Yoked chamber, ``None`` for a
    solo group."""

    number: int
    paired: int
    yoked: int | None = None

    @property
    def chambers(self) -> tuple[int, ...]:
        """The group's chambers, in order."""
        return tuple(sorted(c for c in (self.paired, self.yoked) if c is not None))

    @property
    def solo(self) -> bool:
        """A group of one chamber, with no yoked control."""
        return self.yoked is None


def paired_layout(paired: Sequence[int]) -> tuple[ChamberGroup, ...]:
    """The paired layout's groups with *paired* (one chamber of each of
    :data:`CHAMBER_GROUPS`, as :func:`parse_paired_chambers` accepts) as
    their Paired chambers."""
    by_group = {group_of(c): int(c) for c in paired}
    return tuple(ChamberGroup(g, by_group[g], partner_of(by_group[g]))
                 for g in sorted(CHAMBER_GROUPS))


def single_layout(n_chambers: int = N_CHAMBERS) -> tuple[ChamberGroup, ...]:
    """Every chamber its own solo group, numbered as the chambers are."""
    return tuple(ChamberGroup(c, c, None) for c in range(1, int(n_chambers) + 1))


def describe_groups(groups: Sequence[ChamberGroup]) -> str:
    """``1 (paired 1, yoked 4), 2 (chamber 2)``: what each group holds."""
    return ", ".join(f"{g.number} (chamber {g.paired})" if g.solo else
                     f"{g.number} (paired {g.paired}, yoked {g.yoked})" for g in groups)


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


def break_requirement_view(frame) -> Any:
    """Per treatment, the median ``BreakingPoint`` and beside it the median
    ``BreakRequirement`` of a stacked ``pr_breaking_point.csv``, with the
    groups each rests on — ``None`` when no group has a recorded Requirement
    (no generation-3 member), so a project of older recordings shows nothing
    new."""
    import pandas as pd

    if frame is None or frame.empty or "BreakRequirement" not in frame.columns:
        return None
    requirement = pd.to_numeric(frame["BreakRequirement"], errors="coerce")
    if not requirement.notna().any():
        return None
    data = pd.DataFrame({
        "Treatment": frame["Treatment"].astype(str),
        "BreakingPoint": pd.to_numeric(frame["BreakingPoint"], errors="coerce"),
        "BreakRequirement": requirement,
    })
    rows = []
    for treatment, sub in data.groupby("Treatment", sort=False):
        rows.append({"Treatment": treatment, "Groups": int(len(sub)),
                     "Breaking point": float(sub["BreakingPoint"].median()),
                     "Break requirement": (float(sub["BreakRequirement"].median())
                                           if sub["BreakRequirement"].notna().any()
                                           else float("nan")),
                     "With a requirement": int(sub["BreakRequirement"].notna().sum())})
    return pd.DataFrame(rows)


def pooled_yoked_controls(project) -> bool:
    """Whether a Project's pooled Progressive Ratio light QC
    (``<project>_LightQC.csv``, ``Source`` PR) has a group with a Yoked
    chamber; ``True`` when it cannot tell (no such table), so nothing is
    said of a design whose groups are unknown.  A solo group's
    ``YokedChamber`` is NA (decision 13)."""
    import pandas as pd

    frame = getattr(project, "combined_light_qc_frame", lambda: None)()
    if frame is None or frame.empty or "YokedChamber" not in frame.columns:
        return True
    if "Source" in frame.columns:
        frame = frame[frame["Source"].astype(str) == "PR"]
    if frame.empty:
        return True
    return bool(pd.to_numeric(frame["YokedChamber"], errors="coerce").notna().any())


def all_fixed_ratio(frame) -> bool:
    """Whether every row of a stacked ``pr_breaking_point.csv`` ran a
    FixedRatio Test Paradigm (``ParadigmType``), so its count is the
    activations earned at a constant requirement (decision 14).  False for
    an empty table and for one written before the column existed."""
    if frame is None or frame.empty or "ParadigmType" not in frame.columns:
        return False
    return bool((frame["ParadigmType"].astype("string") == "FixedRatio").fillna(False).all())


#: The two-well chamber_sets default: chamber c is wells (2c-1, 2c).
_DEFAULT_CHAMBER_SETS: tuple[tuple[int, int], ...] = tuple(
    (2 * c - 1, 2 * c) for c in range(1, 7))


def paired_from_program(dfm_program, chamber_sets=None, pi_direction: str = "left"
                        ) -> tuple[list[int] | None, list[str], list[str]]:
    """``(paired_chambers, problems, notes)`` read from one DFM section of the
    Opto Program: in each Chamber Group, the chamber holding the program's
    Trigger Well is the Paired one.

    The trigger chambers are the ones :func:`~pyflic.base.opto_roles.chamber_roles`
    gives a trigger Role (Paired, Mutual or Self) from the section's
    :class:`~pyflic.base.opto_roles.LinkageView`; with the usual linkage (a
    Chamber Group's four wells as one Linkage Group) that Role is Paired and
    the partner's is Yoked.

    *chamber_sets* and *pi_direction* are the DFM's (its wells per chamber,
    left first, and the side of well A).  *problems* say why no answer could be
    given — a group with no trigger well, or one in both chambers; *notes* flag
    a trigger well that is well B, since the Sucrose Well is always well A.
    """
    from ..opto_program import wells_label
    from ..opto_roles import TRIGGER_ROLES, chamber_map, chamber_roles, linkage_view

    sets = [tuple(int(w) for w in row)
            for row in (chamber_sets if chamber_sets is not None else _DEFAULT_CHAMBER_SETS)]
    chamber_of = chamber_map(sets)
    view = linkage_view(dfm_program, None, None)
    roles = chamber_roles(view, chamber_of)
    triggers = set(view.trigger_wells)
    paired: list[int] = []
    problems: list[str] = []
    notes: list[str] = []
    for g, (a, b) in CHAMBER_GROUPS.items():
        hit = sorted(c for c in (a, b) if roles.get(c) in TRIGGER_ROLES)
        if len(hit) == 1:
            paired.append(hit[0])
        elif not hit:
            problems.append(f"Program.txt has no trigger well in chamber group {g} "
                            f"(chambers {a}-{b})")
        else:
            own = sorted(w for w in triggers if chamber_of.get(w) in (a, b))
            problems.append(f"Program.txt has trigger wells in both chambers of group {g} "
                            f"({wells_label(own)})")
    if problems:
        return None, problems, notes
    return paired, problems, _well_b_notes(paired, sets, chamber_of, triggers, pi_direction)


def _well_b_notes(paired: Sequence[int], sets, chamber_of: Mapping[int, int],
                  triggers, pi_direction: str) -> list[str]:
    """A note for each Paired chamber whose Trigger Well is not its well A,
    the Sucrose Well, under *pi_direction*."""
    from ..opto_program import wells_label

    notes: list[str] = []
    for chamber in paired:
        if chamber > len(sets):
            continue
        left, right = sets[chamber - 1]
        well_a = left if str(pi_direction).lower() == "left" else right
        own = sorted(w for w in triggers if chamber_of.get(w) == chamber)
        if well_a not in own:
            notes.append(f"the trigger well of paired chamber {chamber} is "
                         f"{wells_label(own)}, which is well B under pi_direction "
                         f"{pi_direction}; the Sucrose Well is well A")
    return notes


def groups_from_program(dfm_program, chamber_sets=None, pi_direction: str = "left"
                        ) -> tuple[tuple[ChamberGroup, ...] | None, list[str], list[str]]:
    """``(groups, problems, notes)``: the Chamber Groups one DFM section of
    the Opto Program's Linkage makes (decision 13 of
    ``docs/paradigm-analyses-plan.md``).

    A Linkage Group with a Trigger Well lights the chambers its wells sit
    in: two chambers are a pair, the one holding the Trigger Well Paired and
    the other Yoked; one chamber is a solo group, Paired.  A Linkage Group
    with no Trigger Well lights nothing of its own.  *problems* say why the
    Linkage makes no groups — a Linkage Group over more than two chambers,
    Trigger Wells in both chambers of one, a chamber no Trigger Well lights,
    or one in two lit groups; *notes* flag a Trigger Well that is well B
    (:func:`paired_from_program`'s note).  Groups are numbered in chamber
    order.
    """
    from ..opto_roles import chamber_map, linkage_view

    sets = [tuple(int(w) for w in row)
            for row in (chamber_sets if chamber_sets is not None else _DEFAULT_CHAMBER_SETS)]
    chamber_of = chamber_map(sets)
    view = linkage_view(dfm_program, None, None)
    triggers = set(view.trigger_wells)
    ## A trigger well no Linkage Group lists lights only itself, as the
    ## Roles read it (opto_roles.chamber_roles).
    grouped = {w for wells in view.groups.values() for w in wells}
    alone = sorted(triggers - grouped)
    found: list[tuple[int, int | None]] = []
    problems: list[str] = []
    for wells in (*view.groups.values(), *((w,) for w in alone)):
        members = sorted({chamber_of[w] for w in wells if w in chamber_of})
        lit_by = sorted({chamber_of[w] for w in wells if w in chamber_of and w in triggers})
        if not members or not lit_by:
            continue
        if len(members) > 2:
            problems.append(f"Program.txt links chambers {members} into one light circuit; "
                            f"a chamber group is one chamber or two")
        elif len(lit_by) > 1:
            problems.append(f"Program.txt has trigger wells in both linked chambers "
                            f"{members[0]} and {members[1]}")
        else:
            paired = lit_by[0]
            found.append((paired, next((c for c in members if c != paired), None)))
    count = Counter(c for pair in found for c in pair if c is not None)
    for chamber in sorted(set(chamber_of.values())):
        if not count[chamber]:
            problems.append(f"Program.txt lights chamber {chamber} from no trigger well")
        elif count[chamber] > 1:
            problems.append(f"Program.txt puts chamber {chamber} in more than one lit "
                            f"linkage group")
    if problems:
        return None, problems, []
    found.sort(key=lambda pair: min(c for c in pair if c is not None))
    groups = tuple(ChamberGroup(k, p, y) for k, (p, y) in enumerate(found, start=1))
    notes = _well_b_notes([g.paired for g in groups], sets, chamber_of, triggers, pi_direction)
    return groups, [], notes


def parse_layout(raw: Any) -> tuple[str | None, list[str]]:
    """``(layout, problems)`` for a DFM entry's ``chamber_groups``: ``paired``,
    ``single`` or ``None`` when it is not stated."""
    if raw is None:
        return None, []
    text = str(raw).strip().lower()
    if text in GROUP_LAYOUTS:
        return text, []
    return None, [f"'{GROUPS_KEY}' must be one of {', '.join(GROUP_LAYOUTS)}, got {raw!r}"]


@dataclass(frozen=True, slots=True)
class GroupResolution:
    """What :func:`resolve_chamber_groups` decided for one DFM: its groups
    (``None`` when *problems* leave none), what says so (``"config"`` or
    ``"program"``), and notes for the loader to print."""

    groups: tuple[ChamberGroup, ...] | None
    source: str
    problems: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


def _stored_groups(raw: Any) -> tuple[ChamberGroup, ...]:
    return tuple(ChamberGroup(k, int(p), None if y is None else int(y))
                 for k, (p, y) in enumerate(raw, start=1))


def resolve_chamber_groups(node: Mapping[str, Any] | None, dfm_program=None, *,
                           chamber_sets=None, pi_direction: str = "left",
                           dfm_id: int | None = None) -> GroupResolution:
    """One DFM's Chamber Groups, from its ``dfms:`` entry and its Opto Program
    section (decision 13 of ``docs/paradigm-analyses-plan.md``).

    What the config states wins, as ``paired_chambers`` always has:

    * ``paired_chambers``: the paired layout with those Paired chambers; a
      section whose Trigger Wells make others is noted;
    * ``chamber_groups: single``: six solo groups; a section whose Linkage
      makes pairs is noted;
    * neither, with a section: its Linkage (:func:`groups_from_program`), or,
      when the Linkage makes no groups, the paired layout with the section's
      trigger chambers as the Paired ones (:func:`paired_from_program`, as
      before solo groups);
    * otherwise (no section, or ``chamber_groups: paired`` and a section
      that names no Paired chambers): ``paired_chambers`` is required.

    A node the loader has completed carries the groups the section gave
    (:data:`RESOLVED_GROUPS_KEY`), so the result is the same without it.
    Never raises.
    """
    from ..opto_program import wells_label

    node = node or {}
    where = f"DFM {dfm_id}: " if dfm_id is not None else ""
    if node.get(RESOLVED_GROUPS_KEY) is not None:
        return GroupResolution(_stored_groups(node[RESOLVED_GROUPS_KEY]), "program")
    layout, problems = parse_layout(node.get(GROUPS_KEY))
    stated = node.get(PAIRED_KEY)
    if problems:
        return GroupResolution(None, "config", tuple(problems))
    notes: list[str] = []
    if stated is not None:
        if layout == GROUPS_SINGLE:
            return GroupResolution(None, "config", (
                f"'{PAIRED_KEY}' cannot be stated with '{GROUPS_KEY}: single', where "
                f"every chamber is its own group and Paired (remove one)",))
        given, problems = parse_paired_chambers(stated)
        if dfm_program is not None:
            derived, _problems, derived_notes = paired_from_program(dfm_program, chamber_sets,
                                                                    pi_direction)
            notes.extend(f"{where}{note}" for note in derived_notes)
            if derived is not None and not problems and sorted(given) != sorted(derived):
                notes.append(f"{where}paired_chambers {sorted(given)} disagrees with "
                             f"Program.txt, whose trigger wells make chambers {derived} "
                             f"paired; the config's are used")
        if problems:
            return GroupResolution(None, "config", tuple(problems), tuple(notes))
        return GroupResolution(paired_layout(given), "config", (), tuple(notes))
    if layout == GROUPS_SINGLE:
        if dfm_program is not None:
            linked, _problems, _notes = groups_from_program(dfm_program, chamber_sets,
                                                            pi_direction)
            if linked is not None and not all(g.solo for g in linked):
                notes.append(f"{where}chamber_groups single disagrees with Program.txt, "
                             f"whose linkage makes chamber groups {describe_groups(linked)}; "
                             f"the config's are used")
        return GroupResolution(single_layout(), "config", (), tuple(notes))
    if dfm_program is not None:
        linked, linked_problems, linked_notes = (
            groups_from_program(dfm_program, chamber_sets, pi_direction)
            if layout is None else (None, [], []))
        derived, derived_problems, derived_notes = paired_from_program(
            dfm_program, chamber_sets, pi_direction)
        if derived is not None and (linked is None or linked == paired_layout(derived)):
            ## The paired layout, as before solo groups, in the same words.
            return GroupResolution(paired_layout(derived), "program", (), tuple(
                [f"{where}paired_chambers {derived} taken from Program.txt (trigger wells "
                 f"{wells_label(dfm_program.trigger_wells())})",
                 *(f"{where}{note}" for note in derived_notes)]))
        if linked is not None:
            kind = ("each a single chamber, with no yoked control"
                    if all(g.solo for g in linked) else describe_groups(linked))
            return GroupResolution(linked, "program", (), tuple(
                [f"{where}chamber groups taken from Program.txt's linkage: "
                 + (f"{len(linked)} groups, {kind}" if all(g.solo for g in linked) else kind),
                 *(f"{where}{note}" for note in linked_notes)]))
        notes.extend(f"{where}paired_chambers cannot be taken from Program.txt: {p}"
                     for p in derived_problems)
        notes.extend(f"{where}chamber groups cannot be taken from Program.txt's linkage: {p}"
                     for p in linked_problems)
    _given, problems = parse_paired_chambers(None)
    problems = [f"{p}; or {GROUPS_KEY}: single, for a design without yoked controls"
                for p in problems]
    return GroupResolution(None, "config", tuple(problems), tuple(notes))


#: Every Progressive Ratio constant beyond the three auto-filter cutoffs, in
#: the order the editors show them.  One list, so the two editors and the
#: validation cannot disagree about what exists or what is allowed.
PR_CONSTANT_FIELDS: tuple[ConstantField, ...] = (
    ConstantField(
        "require_training_complete", "Exclude groups whose training never completed",
        "require_training_complete — a chamber group whose paired fly never "
        "finished training leaves the analysis, with its yoked chamber when it has "
        "one.",
        "switch", short_label="Training never completed",
        choices=("exclude the group", "keep the group")),
    ConstantField(
        "exclude_failed_pr_groups", "Exclude groups that fail the light QC",
        "exclude_failed_pr_groups — a chamber group whose light QC fails leaves the "
        "analysis, with its yoked chamber when it has one: light that followed the "
        "sensor rather than the fly (self-triggered light, implausible training), or "
        "a requirement series the program does not give.  Off keeps them; "
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
    ConstantField(
        "pr_simulated_yoked_bp", "Simulate the yoked fly's breaking point",
        "pr_simulated_yoked_bp — replays the Test Paradigm on each yoked fly's own "
        "sucrose-well contact from its group's training end, and scores the "
        "simulated lights like the paired fly's: the Simulated Yoked Breaking "
        "Point, what background feeding alone would have earned.  Adds it and "
        "dBreakingPoint (paired minus simulated yoked) to pr_breaking_point.csv.  "
        "Needs a firmware 4 recording whose ratio Paradigm counts Contact.  Off "
        "by default (ADR-0023).",
        "switch", short_label="Simulated yoked BP",
        choices=("simulate it", "off")),
)


def constant_problems(constants: Mapping[str, Any] | None) -> list[str]:
    """What is wrong with the Progressive Ratio constants a ``constants:`` block
    states — a switch that is not true or false, a threshold that is not a
    number or falls outside its range.  Keys it does not state are fine: the
    type's defaults fill them in.  Never raises."""
    return _constant_problems(PR_CONSTANT_FIELDS, constants,
                              default="the type's default")
