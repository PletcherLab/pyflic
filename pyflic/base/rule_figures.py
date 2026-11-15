"""The light QC's rule subsections: each Paradigm's rule checks, and its rule figure.

Stream S3 of ``docs/paradigm-analyses-plan.md``.  On a firmware 4.0.0
recording (generation 3) with a ``Program.txt``, the report's *Optogenetic
light QC* gains one subsection per Paradigm the member's program runs, in
:data:`~pyflic.base.paradigm_families.FAMILY_ORDER`, then by Paradigm name
(:func:`rule_sections`, :func:`report_blocks`).  Each has:

* the rule in words, with its settings (:func:`paradigm_families.rule_sentence
  <pyflic.base.paradigm_families.rule_sentence>`);
* a table of its family's light-QC columns per Linkage Group that ran it,
  with the Activations judged and not judged and the verdict;
* one figure, one panel per DFM, that puts the measured light against the
  rule's setting:

  ======================  ====================================================
  Family (figure)         What it draws
  ======================  ====================================================
  closed loop             a histogram of the judged Activations' lit time
  (activation_length)     (``LitMs``), dashed at the Hold and at ``MaxOnMs``
  fixed length, interval  a histogram of the time between consecutive Main
  (onset_spacing)         onsets (``SinceLastOnsetMs``), dashed at
                          max(``LockoutMs``, ``ActivationMs``), and beside it
                          the lit time, dashed at ``ActivationMs``
  fixed length, ratio     the Requirement each Main Activation met against its
  (requirement_series)    number K, over the line of ``requirement_for(k)``,
                          and beside it the lit time, dashed at
                          ``ActivationMs``
  non-feeding             a histogram of each Activation's relight delay
  (relight_delay)         (:func:`rule_checks.relight_delays
                          <pyflic.base.rule_checks.relight_delays>`), dashed at
                          ``OnDelayMs``; with ``MaxOnMs``, beside it the lit
                          time, dashed there
  open loop               none: the Program timeline shows the schedule
  ======================  ====================================================

  Light that breaks the rule is drawn in the failure colour, as bars and as a
  rug of ticks under the axis, so that one failing Activation among hundreds
  still shows.  A section with nothing judged gets a
  :class:`report_layout.NothingToDraw <pyflic.base.report_layout.NothingToDraw>`
  note instead.

A Paradigm with Acclimation has two subsections: its Acclimation, under
closed loop, and its Main, under its own family, so a section appears exactly
when its family is in the program (:func:`paradigm_families.families_in
<pyflic.base.paradigm_families.families_in>`).  Wells set ``On`` under
another Paradigm are open loop: one section of their own.  A Program
Interval beyond the lines ``Program.txt`` prints (one past the duration, or
the rest of a Circadian day before PyMCU 4.0.1) is judged by its Paradigm by
name, as the light QC judges it.

**The numbers are the light QC's.**  The checks are stream S1's
(:mod:`pyflic.base.rule_checks`), made with the inputs the light QC gives
them (``opto_qc``'s windows, each group's mode and Trigger Wells in them and
the Paradigm starts, read through its helpers; the Activation ledger with
the program's Linkage, whose rows ``opto_light_events.csv`` carries), but on
one Paradigm of one Linkage Group at a time.  So the checks' columns, summed
over a group's sections, are its ``opto_light_qc.csv`` values, and a group
that ran two Paradigms of a family shows each apart.  The per-Activation
marks the figures draw (:func:`mark_activations`) use the checks' own rules,
and add up to their counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import activation_ledger as al
from . import opto_program as op
from . import paradigm_families as pfam
from . import rule_checks as rc
from .paradigm_state import GENERATION_PARADIGM, RECORD_MS

#: The colours the figures share with the light QC's own figure: light that
#: keeps the rule (its "explained" grey-blue) and light that breaks it (its
#: "unexplained" red).
KEEPS_COLOR = "#9DB4C8"
BREAKS_COLOR = "#D62728"
LINE_COLOR = "#333333"
KEEPS, BREAKS = "keeps the rule", "breaks the rule"

#: The name of the open-loop section of Wells set ``On`` under a Paradigm
#: that is not OpenLoop.
WELLS_ON = "Wells set On"

#: The columns :func:`mark_activations` adds to the ledger's rows.
MARK_COLUMNS: tuple[str, ...] = ("Judged", "Breaks", "SpacingJudged", "SpacingBreaks",
                                 "SeriesComparable", "SeriesK", "SeriesBreaks",
                                 "RelightDelayMs")


# ---------------------------------------------------------------------------
# One section
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class RuleSection:
    """One Paradigm's subsection of the light QC.

    *family* is a :data:`~pyflic.base.paradigm_families.FAMILY_ORDER` key;
    *paradigm* is ``None`` for the Wells set ``On`` under another Paradigm
    (:data:`WELLS_ON`); *acclimation* marks a fixed-length Paradigm's
    Acclimation, which is closed loop.  *groups* has one row per Linkage
    Group that ran it (``DFM``, ``Group``, ``Wells``, ``Treatment``, the
    family's light-QC columns this Paradigm has, ``Judged``, ``NotJudged``,
    ``Flags`` and ``Verdict``); *activations* its Activations, the ledger's
    rows with ``DFM`` and ``Group`` and :data:`MARK_COLUMNS`."""

    family: str
    paradigm: op.Paradigm | None
    acclimation: bool
    dfms: tuple[int, ...]
    groups: pd.DataFrame
    activations: pd.DataFrame
    tol_ms: float = 25.0
    record_ms: int = RECORD_MS

    @property
    def name(self) -> str:
        return self.paradigm.name if self.paradigm is not None else WELLS_ON

    @property
    def branch(self) -> str | None:
        if self.family != pfam.FIXED_LENGTH or self.paradigm is None:
            return None
        return pfam.branch_of(self.paradigm.mode)

    @property
    def figure(self) -> str | None:
        """Its rule figure (``paradigm_families.FIGURE_*``), ``None`` for open loop."""
        if self.paradigm is None or self.family == pfam.OPEN_LOOP:
            return None
        return pfam.rule_figure(self.paradigm.mode, acclimation=self.acclimation)

    @property
    def title(self) -> str:
        """``FR3 Acclimation (closed loop; DFM 1)``."""
        what = self.name + (" Acclimation" if self.acclimation else "")
        family = self.family + (f", {self.branch}" if self.branch else "")
        dfms = ", ".join(str(d) for d in self.dfms)
        return f"{what} ({family}; DFM {dfms})"

    def sentence(self) -> str:
        """The rule in words, with its settings (``rule_sentence``, whose
        Acclimation clause an Acclimation section is checked against)."""
        if self.paradigm is None:
            return (f"{WELLS_ON} (open loop): a Well whose threshold is On lights its Linkage "
                    f"Group throughout, whatever the fly does; no Activations")
        return pfam.rule_sentence(self.paradigm)


# ---------------------------------------------------------------------------
# One DFM, read once
# ---------------------------------------------------------------------------

class _DFMRules:
    """What every section of one DFM reads: its Activation ledger with the
    program's Linkage, the light QC's windows and each Linkage Group's mode
    in them, the rows where the Paradigm state started, and pyflic's
    activity per well."""

    def __init__(self, opto, dfm_id: int, settings) -> None:
        from . import opto_light as ol
        from . import opto_qc as oq

        exp = opto.experiment
        dfm = exp.dfms[int(dfm_id)]
        program = opto.program
        state = dfm.paradigm
        self.dfm_id = int(dfm_id)
        self.state = state
        n = len(state)
        self.n = n
        self.record_ms = int(round(1000.0 / float(dfm.params.samples_per_second)))
        self.groups = program.section(self.dfm_id).linkage_groups()
        ## The light QC's ledger: linked by the program's Linkage Groups.
        self.book = al.ledger(state, linked={w: tuple(g) for g in self.groups.values()
                                             for w in g}, record_ms=self.record_ms)
        self.windows = oq._paradigm_windows(state, program, self.dfm_id)[0]
        self.window_of = np.full(n, -1, dtype=int)
        for k, w in enumerate(self.windows):
            self.window_of[w.i0:w.i1] = k
        self.book_window = (self.window_of[self.book["Row"].to_numpy(dtype=np.int64)]
                            if len(self.book) else np.zeros(0, dtype=int))
        self.starts = oq._paradigm_starts(state)
        self.activity = np.asarray(ol.dfm_inputs(dfm).activity, dtype=bool).reshape(
            n, op.N_WELLS)
        self.index = np.asarray(state.index)
        self.on_ms = np.asarray(state.on_ms, dtype=float).reshape(n, op.N_WELLS)
        triggers = state.wells_with_activations()
        self.modes = {label: [oq._run_mode(w, tuple(wells), triggers, settings)
                              for w in self.windows]
                      for label, wells in self.groups.items()}

    def windows_of(self, label: int, family: str, name: str | None) -> list[int]:
        """The windows in which group *label* ran *name* (``None``: Wells set
        ``On`` under another Paradigm) as *family*: with Trigger Wells, or for
        open loop lit regardless of them."""
        modes = self.modes[label]
        out: list[int] = []
        for k, w in enumerate(self.windows):
            if w.kind != "judged" or w.paradigm is None:
                continue
            mode, trig, _hold = modes[k]
            if family == pfam.OPEN_LOOP:
                if mode != op.OPEN_LOOP:
                    continue
                if (w.paradigm.type != op.OPEN_LOOP) if name is None \
                        else w.paradigm.name == name:
                    out.append(k)
            elif w.paradigm.name == name and trig and mode == w.paradigm.mode:
                out.append(k)
        return out

    def rows_of(self, label: int, ks: list[int]) -> pd.DataFrame:
        """The ledger's rows of the group's Activations in windows *ks*: a
        Trigger Well's onsets in a window where it is one, in the ledger's
        order (by Well, then Row), as the light QC takes them."""
        mask = np.zeros(len(self.book), dtype=bool)
        wells = self.book["Well"].to_numpy(dtype=np.int64) if len(self.book) else np.zeros(0)
        modes = self.modes[label]
        for k in ks:
            mask |= (self.book_window == k) & np.isin(wells, list(modes[k][1]))
        return self.book[mask]

    def ran(self, label: int, ks: list[int]) -> dict[int, np.ndarray]:
        """Per Trigger Well, the rows where it ran windows *ks* as one."""
        modes = self.modes[label]
        triggers = sorted({t for k in ks for t in modes[k][1]})
        out = {t: np.zeros(self.n, dtype=bool) for t in triggers}
        for k in ks:
            for t in modes[k][1]:
                out[t][self.windows[k].i0:self.windows[k].i1] = True
        return out


# ---------------------------------------------------------------------------
# Which sections a program has
# ---------------------------------------------------------------------------

def _section_keys(program: op.OptoProgram, dfm_ids) -> dict[tuple, tuple[op.Paradigm | None,
                                                                       set[int]]]:
    """``{(family, name, acclimation): (Paradigm, DFMs)}`` from *program*'s
    printed interval lines, as :func:`paradigm_families.families_in` reads
    them: a line's Paradigm where a Well has a level threshold, its
    Acclimation as closed loop, and open loop for an ``On`` Well."""
    out: dict[tuple, tuple[op.Paradigm | None, set[int]]] = {}

    def add(key: tuple, paradigm: op.Paradigm | None, dfm_id: int) -> None:
        out.setdefault(key, (paradigm, set()))[1].add(int(dfm_id))

    for dfm_id in dfm_ids:
        section = program.section(dfm_id)
        if section is None:
            continue
        for iv in section.intervals:
            p = iv.paradigm
            if iv.open_wells():
                if p.type == op.OPEN_LOOP:
                    add((pfam.OPEN_LOOP, p.name, False), p, dfm_id)
                else:
                    add((pfam.OPEN_LOOP, None, False), None, dfm_id)
            if not iv.trigger_wells():
                continue
            if p.type == op.OPEN_LOOP:
                add((pfam.OPEN_LOOP, p.name, False), p, dfm_id)
                continue
            family = pfam.family_of(p.mode)
            if family is not None:
                add((family, p.name, False), p, dfm_id)
            if p.acclimation and p.type in op.ACCLIMATION_PARADIGMS:
                add((pfam.CLOSED_LOOP, p.name, True), p, dfm_id)
    return out


def _order(key: tuple) -> tuple:
    family, name, acclimation = key
    return (pfam.FAMILY_ORDER.index(family), name is None, (name or "").lower(), acclimation)


def rule_dfms(opto) -> list[int]:
    """The DFMs whose Paradigms get rule sections: covered by the light QC,
    recorded on generation 3, with a ``Program.txt`` section (the checks
    read the Paradigm's settings)."""
    program = opto.program
    if program is None or program.grammar == op.GRAMMAR_SIMPLE:
        return []
    exp = opto.experiment
    return [d for d in opto.dfm_ids()
            if int(getattr(exp.dfms[d], "generation", 0) or 0) == GENERATION_PARADIGM
            and getattr(exp.dfms[d], "paradigm", None) is not None
            and program.section(d) is not None]


# ---------------------------------------------------------------------------
# The checks, per Paradigm and Linkage Group
# ---------------------------------------------------------------------------

def _applies(rows: pd.DataFrame, paradigm: op.Paradigm, family: str,
             acclimation: bool) -> np.ndarray:
    """The Activations a section judges: ClosedLoop's Main ones, a
    Paradigm's Acclimation ones, a fixed-length Paradigm's Main ones, every
    NonFeeding one."""
    phase = rows["Phase"].astype(object).to_numpy()
    if family == pfam.CLOSED_LOOP:
        return phase == ("A" if acclimation else "M")
    if family == pfam.FIXED_LENGTH:
        return phase == "M"
    return np.ones(len(rows), dtype=bool)


def _checks(ctx: _DFMRules, label: int, ks: list[int], rows: pd.DataFrame,
            paradigm: op.Paradigm, family: str, acclimation: bool,
            settings) -> list[rc.CheckResult]:
    """The section's checks on one Linkage Group, as the light QC makes them
    (``opto_qc._group_rule_checks``) on this Paradigm alone."""
    tol = int(settings.activation_tolerance_ms)
    rm = ctx.record_ms
    p = paradigm
    if family == pfam.CLOSED_LOOP:
        out = [rc.short_activations(rows, p, tol_ms=tol, record_ms=rm,
                                    fail_fraction=settings.activation_length_fail_fraction)]
        if not acclimation:
            out += [rc.max_on_exceeded(rows, p, tol_ms=tol, record_ms=rm),
                    rc.cut_offs(rows, p, tol_ms=tol, record_ms=rm)]
        return out
    ran = ctx.ran(label, ks)
    acts = {t: ctx.activity[:, t - 1] for t in ran}
    if family == pfam.FIXED_LENGTH:
        out = [rc.wrong_activation_length(rows, p, tol_ms=tol, record_ms=rm,
                                          fail_fraction=settings.activation_length_fail_fraction),
               rc.lockout_violations(rows, p, tol_ms=tol, record_ms=rm,
                                     fail_fraction=settings.lockout_fail_fraction),
               rc.requirement_series(rows, p)]
        if p.type == op.FIXED_INTERVAL:
            ## Free means free of any of the well's Activations, so all of them.
            mine = ctx.book[ctx.book["Well"].isin(list(ran))]
            main = {t: ran[t] & ~ctx.state.phase_a[:, t - 1] for t in ran}
            out.append(rc.missed_contacts(mine, p, acts, ctx.index, main, tol_ms=tol,
                                          warn_fraction=settings.missed_contact_fraction,
                                          record_ms=rm))
        return out
    if family == pfam.NON_FEEDING:
        return [rc.non_feeding_dark({t: ctx.on_ms[:, t - 1] for t in ran}, acts, ctx.index,
                                    ran, ctx.starts, p, tol_ms=tol,
                                    warn_fraction=settings.non_feeding_dark_fraction,
                                    record_ms=rm),
                rc.relight_delay(rc.relight_delays(rows, acts, ctx.index, record_ms=rm)),
                rc.max_on_exceeded(rows, p, tol_ms=tol, record_ms=rm)]
    return []


#: Each check's column in ``opto_light_qc.csv``.
_COLUMN_OF: dict[str, str] = {
    rc.SHORT_ACTIVATIONS: rc.SHORT_COLUMN, rc.MAX_ON_EXCEEDED: rc.OVER_MAX_ON_COLUMN,
    rc.CUT_OFFS: rc.CUT_OFFS_COLUMN, rc.WRONG_ACTIVATION_LENGTH: rc.WRONG_LENGTH_COLUMN,
    rc.LOCKOUT_BROKEN: rc.LOCKOUT_COLUMN, rc.CONTACT_WITHOUT_ACTIVATION: rc.MISSED_CONTACTS_COLUMN,
    rc.REQUIREMENT_SERIES_WRONG: rc.SERIES_COLUMN, rc.NON_FEEDING_DARK: rc.NON_FEEDING_DARK_COLUMN,
    rc.RELIGHT_DELAY: rc.RELIGHT_DELAY_COLUMN,
}


# ---------------------------------------------------------------------------
# Per Activation: what the figures draw
# ---------------------------------------------------------------------------

def _numbers(rows: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def mark_activations(rows: pd.DataFrame, paradigm: op.Paradigm, *, family: str,
                     acclimation: bool = False, tol_ms: float, record_ms: int = RECORD_MS,
                     activity: dict[int, np.ndarray] | None = None,
                     index: np.ndarray | None = None) -> pd.DataFrame:
    """*rows* (the ledger's rows of one Paradigm's Activations of one Linkage
    Group, by Well then Row) with :data:`MARK_COLUMNS`, each check's verdict
    on each Activation, by the check's own rule (:mod:`pyflic.base.rule_checks`):

    * ``Judged``: judgeable and judged for its length by the section's
      family; ``Breaks``: the flag its length breaks (*short activations*,
      *max on exceeded*, *wrong activation length*), or ``""``;
    * FixedInterval: ``SpacingJudged``, a Main onset whose distance from the
      previous one is judged, and ``SpacingBreaks``, *lockout broken*;
    * ratio types: ``SeriesComparable`` and ``SeriesBreaks``, *requirement
      series wrong*, and ``SeriesK``, the number K the figure draws it at: K,
      or K - 1 in a state run the check takes as off by one (its first Main
      Activation the previous Paradigm's);
    * NonFeeding: ``RelightDelayMs`` (with *activity* and *index*)."""
    out = rows.copy()
    m = len(rows)
    out["Judged"] = False
    out["Breaks"] = ""
    out["SpacingJudged"] = False
    out["SpacingBreaks"] = False
    out["SeriesComparable"] = False
    out["SeriesK"] = np.nan
    out["SeriesBreaks"] = False
    out["RelightDelayMs"] = np.nan
    if not m:
        return out
    p = paradigm
    ok = rc.judgeable(rows)
    tol = rc.tolerances(rows, tol_ms, record_ms)
    lit = _numbers(rows, "LitMs")
    phase = rows["Phase"].astype(object).to_numpy()
    breaks = np.full(m, "", dtype=object)
    judged = np.zeros(m, dtype=bool)
    over_applies, max_on = rc._max_on_rows(p, rows)
    if max_on > 0:
        over = over_applies & ok & (lit > max_on + tol)
    else:
        over = np.zeros(m, dtype=bool)
    if family == pfam.CLOSED_LOOP:
        applies, limit, _words = rc._closed_loop_rule(p, rows)
        judged = applies & ok
        if limit is not None:
            breaks[judged & (lit < limit - tol)] = rc.SHORT_ACTIVATIONS
        if not acclimation:
            breaks[over] = rc.MAX_ON_EXCEEDED
    elif family == pfam.FIXED_LENGTH:
        main = phase == "M"
        judged = main & ok
        if p.activation_ms is not None:
            wrong = judged & (np.abs(lit - float(p.activation_ms)) > tol)
            breaks[wrong] = rc.WRONG_ACTIVATION_LENGTH
        if p.type == op.FIXED_INTERVAL and p.activation_ms is not None:
            overlap = rc._overlap(rows)
            prev = rc._previous(rows)
            at = np.maximum(prev, 0)
            pairs = main & (prev >= 0) & (phase[at] == "M")
            spaced = pairs & ~overlap & ~overlap[at]
            since = _numbers(rows, "SinceLastOnsetMs")
            out["SpacingJudged"] = spaced
            out["SpacingBreaks"] = spaced & (since < rc._spacing(p) - np.maximum(tol, tol[at]))
        if p.type in op.RATIO_PARADIGMS and p.requirement_for(1) is not None:
            _series_marks(out, rows, p)
    elif family == pfam.NON_FEEDING:
        judged = ok.copy()
        breaks[over] = rc.MAX_ON_EXCEEDED
        if activity is not None and index is not None:
            out["RelightDelayMs"] = rc.relight_delays(rows, activity, index,
                                                      record_ms=record_ms)
    out["Judged"] = judged
    out["Breaks"] = breaks
    return out


def _series_marks(out: pd.DataFrame, rows: pd.DataFrame, paradigm: op.Paradigm) -> None:
    """``SeriesComparable``, ``SeriesK`` and ``SeriesBreaks``, as
    :func:`rule_checks.requirement_series_mismatches` scans the state runs."""
    reset = rows.reset_index(drop=True)
    comparable, k, met = rc._comparable(reset)
    runs = rc._state_run_ids(reset)
    shown_k = k.copy()
    bad_rows = np.zeros(len(reset), dtype=bool)
    for r in np.unique(runs[comparable]).tolist():
        run = runs == r
        sel = np.flatnonzero(run & comparable)
        expected = np.array([paradigm.requirement_for(int(k[i])) for i in sel], dtype=float)
        bad = sel[met[sel] != expected]
        if not bad.size:
            continue
        if rc._series_shifted(paradigm, reset, run, comparable, k, met):
            shown_k[run] = k[run] - 1
            continue
        bad_rows[bad] = True
    out["SeriesComparable"] = comparable
    out["SeriesK"] = shown_k
    out["SeriesBreaks"] = bad_rows


# ---------------------------------------------------------------------------
# The sections of one experiment
# ---------------------------------------------------------------------------

def _verdict(flags: list[str]) -> str:
    from .opto_light import verdict_text

    return verdict_text(", ".join(flags))


def rule_sections(opto) -> list[RuleSection]:
    """The rule subsections of *opto* (an
    :class:`~pyflic.base.opto_light.OptoLightQC`): one per Paradigm of its
    program on the DFMs :func:`rule_dfms` gives (a fixed-length Paradigm's
    Acclimation apart, and the Wells set ``On`` under another Paradigm as
    one), in :data:`~pyflic.base.paradigm_families.FAMILY_ORDER`, then by
    name.  Empty without such a DFM."""
    ids = rule_dfms(opto)
    if not ids:
        return []
    program = opto.program
    settings = opto.settings()
    tol = float(settings.activation_tolerance_ms)
    keys = _section_keys(program, ids)
    qc = opto.qc_table()
    treatment = {(int(r.DFM), int(r.Group)): str(r.Treatment or "")
                 for r in qc.itertuples(index=False)}
    intervals = opto.interval_table()
    contexts: dict[int, _DFMRules] = {}

    def context(dfm_id: int) -> _DFMRules:
        if dfm_id not in contexts:
            contexts[dfm_id] = _DFMRules(opto, dfm_id, settings)
        return contexts[dfm_id]

    ## Program Intervals beyond the lines Program.txt prints (a Repeating
    ## program's later passes are printed once; a Circadian one that starts
    ## partway through its day omits the rest): the light QC judges them by
    ## their Paradigm by name, and so do these sections.
    for dfm_id in ids:
        for w in context(dfm_id).windows:
            p = w.paradigm
            if w.kind != "judged" or p is None or w.interval is not None \
                    or p.type == op.OPEN_LOOP or pfam.family_of(p.mode) is None:
                continue
            keys.setdefault((pfam.family_of(p.mode), p.name, False), (p, set()))[1].add(dfm_id)
            if p.acclimation and p.type in op.ACCLIMATION_PARADIGMS:
                keys.setdefault((pfam.CLOSED_LOOP, p.name, True), (p, set()))[1].add(dfm_id)
    sections: list[RuleSection] = []
    for key in sorted(keys, key=_order):
        family, name, acclimation = key
        paradigm, dfms = keys[key]
        group_rows: list[dict[str, Any]] = []
        marked: list[pd.DataFrame] = []
        for dfm_id in sorted(dfms):
            ctx = context(dfm_id)
            for label, wells in ctx.groups.items():
                ks = ctx.windows_of(label, family, name)
                if not ks:
                    continue
                row: dict[str, Any] = {"DFM": dfm_id, "Group": label,
                                       "Wells": op.wells_label(wells),
                                       "Treatment": treatment.get((dfm_id, int(label)), "")}
                if family == pfam.OPEN_LOOP:
                    row.update(_open_loop_row(intervals, dfm_id, label, ctx, ks, settings))
                    group_rows.append(row)
                    continue
                rows = ctx.rows_of(label, ks)
                results = _checks(ctx, label, ks, rows, paradigm, family, acclimation, settings)
                for result in results:
                    if result.applies:
                        row[_COLUMN_OF[result.flag]] = result.value
                applies = _applies(rows, paradigm, family, acclimation)
                ok = rc.judgeable(rows)
                row["Judged"] = int((applies & ok).sum())
                row["NotJudged"] = int((applies & ~ok).sum())
                fired = [r.flag for r in results if r.applies and r.fired]
                row["Flags"] = ", ".join(fired)
                row["Verdict"] = _verdict(fired)
                group_rows.append(row)
                acts = {t: ctx.activity[:, t - 1] for t in ctx.ran(label, ks)}
                marks = mark_activations(rows, paradigm, family=family,
                                         acclimation=acclimation, tol_ms=tol,
                                         record_ms=ctx.record_ms, activity=acts,
                                         index=ctx.index)
                marks.insert(0, "Group", label)
                marks.insert(0, "DFM", dfm_id)
                marked.append(marks[_applies(rows, paradigm, family, acclimation)])
        columns = ["DFM", "Group", "Wells", "Treatment", *pfam.FAMILIES[family].qc_columns,
                   "Judged", "NotJudged", "Flags", "Verdict"]
        groups = pd.DataFrame(group_rows)
        groups = groups.reindex(columns=[c for c in columns if c in groups.columns
                                         or c in ("DFM", "Group", "Wells", "Treatment")])
        activations = (pd.concat(marked, ignore_index=True) if marked
                       else pd.DataFrame(columns=["DFM", "Group", *al.LEDGER_COLUMNS,
                                                  *MARK_COLUMNS]))
        record_ms = next(iter(contexts.values())).record_ms if contexts else RECORD_MS
        sections.append(RuleSection(family=family, paradigm=paradigm, acclimation=acclimation,
                                    dfms=tuple(sorted(dfms)), groups=groups,
                                    activations=activations, tol_ms=tol, record_ms=record_ms))
    return sections


def _open_loop_row(intervals: pd.DataFrame, dfm_id: int, label: int, ctx: _DFMRules,
                   ks: list[int], settings) -> dict[str, Any]:
    """An open-loop section's row for one group: the lowest lit fraction of
    its windows in the section (``opto_light_intervals.csv``'s), against
    ``opto_open_loop_min_lit_fraction``, as *open loop not lit* judges it."""
    from .opto_qc import OPEN_LOOP_DARK

    windows = {ctx.windows[k].index for k in ks}
    rows = intervals[(intervals["DFM"] == dfm_id) & (intervals["Group"] == label)
                     & intervals["Window"].isin(windows)]
    fractions = pd.to_numeric(rows["LitFraction"], errors="coerce").dropna()
    lowest = float(fractions.min()) if len(fractions) else np.nan
    fired = bool(np.isfinite(lowest) and lowest < settings.open_loop_min_fraction)
    flags = [OPEN_LOOP_DARK] if fired else []
    return {rc.OPEN_LOOP_COLUMN: lowest, "Windows": int(len(fractions)),
            "Flags": ", ".join(flags), "Verdict": _verdict(flags)}


# ---------------------------------------------------------------------------
# The figures
# ---------------------------------------------------------------------------

def _nothing(section: RuleSection, what: str) -> None:
    from .report_layout import NothingToDraw

    raise NothingToDraw(f"No {what} of {section.name}"
                        + (" in Acclimation" if section.acclimation else "")
                        + " could be judged: each was merged with other light, or cut short "
                          "by an Interval change, a gap in Index or the end of the data.")


def _panels(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    order = [f"DFM {d}" for d in sorted({int(d) for d in frame["DFM"]})]
    frame["Panel"] = pd.Categorical([f"DFM {int(d)}" for d in frame["DFM"]],
                                    categories=order, ordered=True)
    return frame


def _histogram(values: pd.DataFrame, x: str, lines: list[tuple[float, str]], *,
               x_label: str, binwidth: float | None = None, base_size: float = 9.0):
    """A histogram of *values*' column *x* per DFM panel, stacked by
    ``Status`` (keeps / breaks the rule), with a rug of the breaking values
    and dashed lines at *lines* (``(x, label)``)."""
    import plotnine as p9

    data = _panels(values)
    data["Status"] = pd.Categorical(np.where(data["_breaks"], BREAKS, KEEPS),
                                    categories=[KEEPS, BREAKS], ordered=True)
    span_values = [*data[x].tolist(), *[v for v, _ in lines]]
    lo, hi = float(np.nanmin(span_values)), float(np.nanmax(span_values))
    if binwidth is None:
        binwidth = max((hi - lo) / 40.0, 1.0) if hi > lo else 1.0
    ## Bars centred on the rule's setting (the first line), so a light that
    ## keeps it exactly sits on its dashed line, and a few bars either side
    ## even when every value is the same.
    anchor = float(lines[0][0]) if lines else lo
    k0 = int(np.floor((lo - anchor) / binwidth)) - 1
    k1 = int(np.ceil((hi - anchor) / binwidth)) + 1
    breaks = anchor - binwidth / 2.0 + binwidth * np.arange(k0, k1 + 2)
    g = (p9.ggplot(data, p9.aes(x, fill="Status"))
         + p9.geom_histogram(breaks=breaks.tolist(), position="stack", color="white",
                             size=0.15)
         + p9.scale_fill_manual(values={KEEPS: KEEPS_COLOR, BREAKS: BREAKS_COLOR}, drop=False)
         + p9.facet_wrap("~ Panel", ncol=1, scales="free_y")
         + p9.labs(x=x_label, y="Activations", fill=""))
    bad = data[data["_breaks"]]
    if len(bad):
        g += p9.geom_rug(data=bad, mapping=p9.aes(x=x), color=BREAKS_COLOR, sides="b",
                         inherit_aes=False, size=0.8)
    if lines:
        marks = pd.DataFrame({"x": [v for v, _ in lines], "label": [t for _, t in lines]})
        g += p9.geom_vline(data=marks, mapping=p9.aes(xintercept="x"), linetype="dashed",
                           color=LINE_COLOR, size=0.6, inherit_aes=False)
        g += p9.expand_limits(x=[float(marks["x"].min()), float(marks["x"].max())])
    return (g + p9.theme_bw(base_size=base_size)
            + p9.theme(legend_position="bottom", strip_background=p9.element_rect(
                fill="#f0f0f0")))


def lit_time_figure(section: RuleSection):
    """The judged Activations' lit time (``LitMs``) per DFM, dashed at the
    rule's settings: the Hold and ``MaxOnMs`` (closed loop), ``ActivationMs``
    (fixed length), ``MaxOnMs`` (non-feeding); bars and ticks in the failure
    colour for the lengths that break the rule."""
    p = section.paradigm
    data = section.activations
    data = data[data["Judged"].astype(bool)] if len(data) else data
    if not len(data):
        _nothing(section, "Activation's length")
    values = pd.DataFrame({"DFM": data["DFM"].to_numpy(),
                           "LitMs": _numbers(data, "LitMs"),
                           "_breaks": data["Breaks"].astype(str).to_numpy() != ""})
    lines: list[tuple[float, str]] = []
    if section.family == pfam.CLOSED_LOOP:
        hold = p.activation_ms if section.acclimation else p.hold_ms
        if hold is not None:
            lines.append((float(hold), "Hold"))
        if not section.acclimation and (p.max_on_ms or 0) > 0:
            lines.append((float(p.max_on_ms), "MaxOnMs"))
    elif section.family == pfam.FIXED_LENGTH and p.activation_ms is not None:
        lines.append((float(p.activation_ms), "ActivationMs"))
    elif section.family == pfam.NON_FEEDING and (p.max_on_ms or 0) > 0:
        lines.append((float(p.max_on_ms), "MaxOnMs"))
    span = float(np.nanmax(values["LitMs"])) - float(np.nanmin(values["LitMs"]))
    binwidth = max(span / 40.0, float(section.record_ms) / 20.0)
    return _histogram(values, "LitMs", lines, x_label="Lit time per Activation (ms)",
                      binwidth=binwidth)


def onset_spacing_figure(section: RuleSection):
    """The time between consecutive Main onsets of a FixedInterval Paradigm,
    per DFM, dashed at max(``LockoutMs``, ``ActivationMs``); the onsets that
    came sooner, *lockout broken*, in the failure colour.  Longer gaps than
    four times that spacing (or the 90th percentile) are counted in the last
    bar."""
    p = section.paradigm
    data = section.activations
    data = data[data["SpacingJudged"].astype(bool)] if len(data) else data
    if not len(data):
        _nothing(section, "pair of consecutive Main onsets")
    spacing = rc._spacing(p) / 1000.0
    seconds = _numbers(data, "SinceLastOnsetMs") / 1000.0
    cap = max(4.0 * spacing, float(np.nanpercentile(seconds, 90)) if seconds.size else 0.0)
    values = pd.DataFrame({"DFM": data["DFM"].to_numpy(),
                           "Seconds": np.minimum(seconds, cap),
                           "_breaks": data["SpacingBreaks"].astype(bool).to_numpy()})
    binwidth = max(cap / 40.0, 0.02)
    return _histogram(values, "Seconds", [(spacing, "spacing")],
                      x_label="Time from the previous Main onset (s)", binwidth=binwidth)


def requirement_series_figure(section: RuleSection):
    """The Requirement each Main Activation met against its number K, one
    point per Activation of each Trigger Well (spread a little across K),
    over the line of ``requirement_for(k)``; the points the series does not
    give, *requirement series wrong*, larger and in the failure colour."""
    import plotnine as p9

    p = section.paradigm
    data = section.activations
    data = data[data["SeriesComparable"].astype(bool)] if len(data) else data
    if not len(data):
        _nothing(section, "Main Activation's Requirement")
    points = _panels(pd.DataFrame({"DFM": data["DFM"].to_numpy(),
                                   "K": _numbers(data, "SeriesK"),
                                   "Met": _numbers(data, "RequirementMet"),
                                   "_breaks": data["SeriesBreaks"].astype(bool).to_numpy()}))
    top = int(np.nanmax(points["K"])) if len(points) else 1
    series = []
    for panel in points["Panel"].cat.categories:
        for k in range(1, max(top, 1) + 1):
            value = p.requirement_for(k)
            if value is not None:
                series.append({"Panel": panel, "K": float(k), "Met": float(value)})
    line = pd.DataFrame(series)
    if len(line):
        line["Panel"] = pd.Categorical(line["Panel"], categories=points["Panel"].cat.categories,
                                       ordered=True)
    keeps, bad = points[~points["_breaks"]], points[points["_breaks"]]
    g = p9.ggplot(points, p9.aes("K", "Met"))
    if len(line):
        g += p9.geom_line(data=line, color=LINE_COLOR, linetype="dashed", size=0.6)
    if len(keeps):
        g += p9.geom_point(data=keeps, color="#4C78A8", size=1.3, alpha=0.7,
                           position=p9.position_jitter(width=0.18, height=0, random_state=1))
    if len(bad):
        g += p9.geom_point(data=bad, color=BREAKS_COLOR, size=2.6)
    return (g + p9.facet_wrap("~ Panel", ncol=1, scales="free_y")
            + p9.labs(x="Main Activation number K", y="Requirement met")
            + p9.theme_bw(base_size=9.0)
            + p9.theme(strip_background=p9.element_rect(fill="#f0f0f0")))


def relight_delay_figure(section: RuleSection):
    """Each NonFeeding Activation's relight delay that follows pyflic
    activity (its onset less the end of the last activity sample before it),
    per DFM, dashed at ``OnDelayMs``.  Activity is per record, so a delay
    reads up to a record either side of the rule's."""
    p = section.paradigm
    data = section.activations
    delays = _numbers(data, "RelightDelayMs") if len(data) else np.zeros(0)
    keep = np.isfinite(delays)
    if not keep.any():
        _nothing(section, "relight delay")
    values = pd.DataFrame({"DFM": data["DFM"].to_numpy()[keep], "Delay": delays[keep],
                           "_breaks": False})
    span = float(values["Delay"].max() - values["Delay"].min())
    binwidth = max(span / 40.0, float(section.record_ms) / 10.0)
    return _histogram(values, "Delay", [(float(p.on_delay_ms or 0), "OnDelayMs")],
                      x_label="Relight delay after activity ends (ms)", binwidth=binwidth)


_BUILDERS: dict[str, Callable[[RuleSection], Any]] = {
    "lit": lit_time_figure, "spacing": onset_spacing_figure,
    "series": requirement_series_figure, "relight": relight_delay_figure,
}


def figure_parts(section: RuleSection) -> list[str]:
    """The plots of *section*'s rule figure, side by side: ``lit``,
    ``spacing``, ``series`` or ``relight`` (:func:`build_part`).  Fixed
    length draws its lit time beside its spacing or its series; non-feeding
    beside its relight delay only with ``MaxOnMs``; open loop draws none."""
    kind = section.figure
    if kind == pfam.FIGURE_ACTIVATION_LENGTH:
        return ["lit"]
    if kind == pfam.FIGURE_ONSET_SPACING:
        return ["spacing", "lit"]
    if kind == pfam.FIGURE_REQUIREMENT_SERIES:
        return ["series", "lit"]
    if kind == pfam.FIGURE_RELIGHT_DELAY:
        return ["relight", "lit"] if (section.paradigm.max_on_ms or 0) > 0 else ["relight"]
    return []


def build_part(section: RuleSection, part: str):
    """One plot of *section*'s rule figure (a plotnine ``ggplot``); raises
    :class:`~pyflic.base.report_layout.NothingToDraw` with nothing judged."""
    return _BUILDERS[part](section)


def _lines_words(section: RuleSection, part: str) -> str:
    """The dashed lines of one plot, in words."""
    p = section.paradigm
    if part == "spacing":
        return (f"max(LockoutMs, ActivationMs), {rc._spacing(p) / 1000.0:g} s; red, the onsets "
                f"that came sooner (lockout broken)")
    if part == "series":
        return "the series, requirement_for(k); red, the Requirements it does not give"
    if part == "relight":
        return f"OnDelayMs, {int(p.on_delay_ms or 0)} ms"
    if section.family == pfam.CLOSED_LOOP:
        hold = p.activation_ms if section.acclimation else p.hold_ms
        words = f"the Hold, {int(hold or 0)} ms"
        if not section.acclimation and (p.max_on_ms or 0) > 0:
            words += f", and MaxOnMs, {int(p.max_on_ms)} ms"
    elif section.family == pfam.FIXED_LENGTH:
        words = f"ActivationMs, {int(p.activation_ms or 0)} ms"
    else:
        words = f"MaxOnMs, {int(p.max_on_ms or 0)} ms"
    return words + "; red, the lengths that break the rule"


#: What each plot shows.
_WHAT: dict[str, str] = {
    "lit": "Lit time per judged Activation",
    "spacing": "Time from the previous Main onset, longer gaps in the last bar",
    "series": "The Requirement each Main Activation met, by its number K (a little spread "
              "across K)",
    "relight": "Each Activation's delay after pyflic's last activity sample, good to a "
               "record either way",
}
#: Each plot's title.
_TITLES: dict[str, str] = {"lit": "lit time", "spacing": "time between onsets",
                           "series": "Requirement met", "relight": "relight delay"}


def part_caption(section: RuleSection, part: str) -> str:
    """One plot's caption: what it shows, one panel per DFM, its dashed
    lines and what is red.  Red is beyond ``opto_activation_tolerance_ms``
    (a record more for an Activation timed to the record), as the checks
    judge."""
    text = f"{_WHAT[part]}, one panel per DFM.  Dashed: {_lines_words(section, part)}"
    if part in ("lit", "spacing"):
        text += (f", beyond opto_activation_tolerance_ms ({section.tol_ms:g} ms), as bars "
                 f"and as ticks under the axis")
    return text + "."


# ---------------------------------------------------------------------------
# Report blocks
# ---------------------------------------------------------------------------

#: How the table names the light-QC columns.
_HEADERS: dict[str, str] = {
    rc.SHORT_COLUMN: "Short", rc.OVER_MAX_ON_COLUMN: "Over MaxOnMs",
    rc.CUT_OFFS_COLUMN: "Cut-offs", rc.WRONG_LENGTH_COLUMN: "Wrong length",
    rc.LOCKOUT_COLUMN: "Lockout broken", rc.MISSED_CONTACTS_COLUMN: "Missed contacts",
    rc.SERIES_COLUMN: "Series mismatches", rc.NON_FEEDING_DARK_COLUMN: "Dark where lit (s)",
    rc.RELIGHT_DELAY_COLUMN: "Relight delay (ms)",
    rc.OPEN_LOOP_COLUMN: "Lowest lit fraction", "Windows": "Windows",
    "Judged": "Judged", "NotJudged": "Not judged",
}


def section_table(section: RuleSection) -> pd.DataFrame:
    """*section*'s groups as the report shows them: the columns this
    Paradigm has (a check its type lacks is left out), named for a reader."""
    frame = section.groups.drop(columns=["Flags"], errors="ignore")
    keep = [c for c in frame.columns
            if c in ("DFM", "Group", "Wells", "Treatment", "Verdict", "Judged", "NotJudged")
            or pd.to_numeric(frame[c], errors="coerce").notna().any()]
    return frame[keep].rename(columns=_HEADERS)


def _judged_any(section: RuleSection) -> bool:
    data = section.activations
    if not len(data):
        return False
    return bool(data["Judged"].astype(bool).any() or data["SpacingJudged"].astype(bool).any()
                or data["SeriesComparable"].astype(bool).any()
                or np.isfinite(_numbers(data, "RelightDelayMs")).any())


def report_blocks(opto, sections: list[RuleSection] | None = None) -> list[Any]:
    """The rule subsections for the report's *Optogenetic light QC*
    (*sections*, else :func:`rule_sections`): per Paradigm a heading, its
    rule, its table and its figure.  Empty without a DFM the rule checks
    judge."""
    from . import report_layout as rl

    blocks: list[Any] = []
    for section in (rule_sections(opto) if sections is None else sections):
        what = section.name + (" Acclimation" if section.acclimation else "")
        blocks += [rl.Heading(f"Rule check: {section.title}", level=2),
                   rl.Paragraph(section.sentence() + ".")]
        if section.groups.empty:
            blocks.append(rl.Paragraph("No linkage group ran it in this recording.",
                                       color=rl.MUTED, italic=True))
            continue
        how = ("the lowest lit fraction of its open-loop windows" if section.paradigm is None
               or section.family == pfam.OPEN_LOOP else
               "Judged: Activations whose length is the rule's to read")
        blocks.append(rl.Table(section_table(section),
                               caption=f"{what}: {section.family} checks by linkage group "
                                       f"({how})",
                               formats={"Lowest lit fraction": "{:.0%}",
                                        "Dark where lit (s)": "{:.1f}",
                                        "Relight delay (ms)": "{:.0f}"},
                               status={"Verdict": rl.tone_of}))
        parts = figure_parts(section)
        if not parts:
            continue
        ## Each DFM a panel, stacked.
        height = min(8.4, 1.55 * max(1, len(section.dfms)) + 1.0)
        if not _judged_any(section):
            ## Nothing judged: one grey note, not a figure (S3).
            blocks.append(rl.Plot(lambda s=section: _nothing(s, "Activation"), height=height,
                                  title=f"{what} against its rule"))
            continue
        plots = [rl.Plot(lambda s=section, part=part: build_part(s, part),
                         title=f"{what}: {_TITLES[part]}", caption=part_caption(section, part))
                 for part in parts]
        if len(plots) == 1:
            plots[0].height = height
            blocks.append(plots[0])
        else:
            blocks.append(rl.PlotRow(plots, height=height))
    return blocks
