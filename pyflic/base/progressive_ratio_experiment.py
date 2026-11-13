"""The Progressive Ratio experiment: paired/yoked Chamber Groups, data-derived
Training/Test Facets, and the within-group difference (ADR-0013).

Vocabulary is fixed in ``CONTEXT.md`` under *Progressive Ratio*:

* a **Chamber Group** is the chambers of one light circuit, sharing one
  Treatment and one training end: a pair (1+2, 3+4, 5+6, or as Program.txt's
  Linkage makes them), or a **solo** group of one chamber with no yoked
  control (decision 13 of ``docs/paradigm-analyses-plan.md``;
  :func:`~pyflic.base.experiment_types.progressive_ratio.resolve_chamber_groups`);
* the **Paired** chamber's feeding at the **Sucrose Well** (always well A)
  drives the light; the **Yoked** chamber is lit at the same moments;
* **Training** ends, per group, at the paired chamber's Sucrose Well's
  training end; the **Test** phase is everything after.

How a recording says where Training ends depends on its **Generation**
(``CONTEXT.md``; ``docs/v4-data-plan.md``, decisions 11-15):

* generation 1: the last minute the firmware flags the Sucrose Well in
  training (raw sample > 40000); a Light Event is an onset of the chamber's
  ``OptoCol1`` light, the OR of its two wells' bits (the data sets all four
  bits of a group together);
* generation 2 (a DFM older than firmware 4.0.0 recorded by PyMCU 4.0.0):
  nowhere.  Each group has one Facet, ``Recording``, ``TrainingComplete`` is
  NA, and there is no Test phase, Breaking Point or Sucrose Persistence;
* generation 3: the Sucrose Well's first row in Phase ``M`` under a ratio
  Paradigm.  A Light Event is the Sucrose Well's own **Activation**, which
  carries the Requirement it met and the Firmware Events it took, light
  seconds come from ``OnMs``, and the Test window ends where the ratio
  Paradigm's run ends.

Everything here is derived from the ordinary two-well machinery plus the
Chamber Groups: ``Program.txt``'s Linkage, or one config key per DFM,
``paired_chambers`` (or ``chamber_groups: single``).  The paired-minus-yoked
parts skip a solo group, and the report says so once.

A DFM that restarts on its own inside a group's Test window ends that
window: the ratio starts again after a restart, so the Breaking Point is
judged up to it, censored when the fly was still responding there
(decision 15).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from . import pr_breaking_point as pbp
from . import pr_light_qc as lqc
from . import windowing
from .dfm import DFM
from .experiment import Experiment
from .experiment_types.progressive_ratio import (
    PAIRED_KEY,
    ROLE_PAIRED,
    ROLE_YOKED,
    ChamberGroup,
    resolve_chamber_groups,
)
from .opto_program import (
    FIXED_RATIO,
    GRAMMAR_LEGACY,
    PROGRESSIVE_RATIO,
    RATIO_PARADIGMS,
    TYPE_NAMES,
)
from .paradigm_state import GENERATION_PARADIGM, NO_TRAINING_INFORMATION, ParadigmState
from .two_well_experiment import TwoWellExperiment
from .utils import range_bounds, range_is_specified

TRAINING_LABEL = "Training"
TEST_LABEL = "Test"
FACET_LABELS = (TRAINING_LABEL, TEST_LABEL)
#: The one Facet of a generation-2 DFM, which records no Training: the whole
#: recording, never pooled into Test (decision 15).
RECORDING_LABEL = "Recording"
#: Every Facet this type makes, in the order its tables list them.
_FACET_ORDER = (TRAINING_LABEL, TEST_LABEL, RECORDING_LABEL)

## What summary.txt and the report say of a generation-2 DFM
## (ProgressiveRatioExperiment.training_information) is
## paradigm_state.NO_TRAINING_INFORMATION, imported above: the one wording
## the Hub and the Project Report use too.
#: Generation 3: a ratio Paradigm that is the first Interval and has no
#: Acclimation leaves nothing to call Training (decision 11).
EMPTY_TRAINING_NOTE = ("Training was empty: the ratio Paradigm had no Acclimation "
                       "and no Interval before it")
#: What the report says, once, of a design whose Chamber Groups are all solo
#: (decision 13 of ``docs/paradigm-analyses-plan.md``).
NO_YOKED_CONTROLS = "the design has no yoked controls"

#: Per-chamber metrics the Paired-Yoked Difference table carries, as ``d<name>``.
#: ``PersistA`` is Sucrose Persistence, measured from training end (ADR-0014),
#: so its difference exists on Test rows only.
DIFF_METRICS: tuple[str, ...] = (
    "LicksA", "LicksB", "EventsA", "EventsB", "PI", "EventPI",
    "MedDurationA", "MedDurationB", "PersistA",
)
#: The differences the reports test, between treatments and against zero.
DIFF_REPORT_METRICS: tuple[str, ...] = (
    "dLicksA", "dLicksB", "dEventsA", "dPI", "dMedDurationA", "dPersistA",
)

#: Two wells clearing within this many minutes of each other count as together.
_FLAG_TOLERANCE_MIN = 0.1

#: The breaking-point columns every chamber's table carries.
BREAKING_POINT_COLUMNS: tuple[str, ...] = ("Minutes", "CumLicks", "DeltaMinutes",
                                           "DeltaLicks")
#: The Light Event Ledger columns a Paired chamber's table adds (light QC).
#: The last three are the firmware's own record of each Activation
#: (generation 3; NA otherwise): the ``Requirement`` it met, the
#: ``FirmwareEvents`` it took and its ``Phase`` (``A`` or ``M``).
LEDGER_COLUMNS: tuple[str, ...] = ("MinutesSincePrev", "LicksSincePrev", "Explained",
                                   "LickFree", "RestingLevel", "Requirement",
                                   "FirmwareEvents", "Phase")
#: One row per Chamber Group — ``analysis/pr_light_qc.csv``.
#: ``TrendRhoFirmware`` is the trend of Firmware Events per Light Event
#: (generation 3; NA otherwise).
LIGHT_QC_COLUMNS: tuple[str, ...] = (
    "DFM", "Group", "PairedChamber", "YokedChamber", "SucroseWell", "Treatment",
    "TrainingComplete", "TrainingMinutes", "TrainingLightEvents", "TrainingLicks",
    "TestLightEvents", "LickFreeEvents", "LongestLickFreeRun", "LickFreeRunStartMin",
    "TrendRho", "TrendSlope", "TrendRhoFirmware", "RestingStart", "RestingMax",
    "RestingEnd", "RestingRise", "RestingRatio", "Flags", "Verdict", "Excluded",
    "Notes",
)
#: What the light QC adds to every per-chamber summary row.
LIGHT_QC_ROW_COLUMNS: tuple[str, ...] = ("LightQC", "LickFreeLightEvents")
#: Sucrose Persistence on every per-chamber summary row (ADR-0014): minutes
#: since training end of the chamber's last Sucrose Well feeding event before
#: the first gap over ``pr_break_gap_min``, and whether that is censored.
PERSIST_COLUMNS: tuple[str, ...] = ("PersistA", "PersistACensored")
#: On a Paired chamber's breaking-point table and the Light Event Ledger:
#: whether the Light Event is one the group's Breaking Point counts.
COUNTED_COLUMN = "Counted"
#: ``analysis/pr_breaking_point.csv``, after ``Treatment`` and the factors.
#: ``BreakRequirement`` and ``FirmwareEvents`` are the firmware's record of
#: the counted Light Events (generation 3; NA otherwise; ADR-0014 addendum);
#: ``ParadigmType`` the Test Paradigm's type
#: (:meth:`ProgressiveRatioExperiment.test_paradigm_type`).
BREAKING_POINT_SUMMARY_COLUMNS: tuple[str, ...] = (
    "DFM", "Group", "PairedChamber", "BreakingPoint", "BreakMin", "Censored",
    "TestMinutes", "LargestRequirement", "BreakRequirement", "FirmwareEvents",
    "LickFreeLightEvents", "LightQC", "ParadigmType",
)


@dataclass(frozen=True, slots=True)
class ChamberRole:
    dfm_id: int
    chamber: int
    group: int
    role: str          # ROLE_PAIRED | ROLE_YOKED
    partner: int | None  # None: a solo group's chamber has no partner


@dataclass(frozen=True, slots=True)
class GroupTraining:
    """One Chamber Group's training verdict."""

    dfm_id: int
    group: int
    paired_chamber: int
    yoked_chamber: int | None      # None: a solo group
    sucrose_well: int
    training_end: float | None     # minutes; None = never completed
    notes: tuple[str, ...] = ()    # QC notes, never fatal
    #: False when the recording carries no Training information at all
    #: (generation 2): then nothing completed and nothing failed to.
    known: bool = True

    @property
    def complete(self) -> bool:
        """Training ended, so the group has a Test phase."""
        return self.training_end is not None

    @property
    def training_complete(self) -> bool | None:
        """The ``TrainingComplete`` cell: :attr:`complete`, or ``None`` (NA)
        when the recording has no Training information (decision 15)."""
        return self.complete if self.known else None


@dataclass(slots=True)
class ProgressiveRatioExperiment(TwoWellExperiment):
    """A Progressive Ratio specialisation of :class:`TwoWellExperiment`."""

    _pr_groups: dict | None = None      # {dfm_id: (ChamberGroup, ...)}
    _pr_training: dict | None = None
    _pr_light: dict | None = None       # light QC caches, per DFM object
    _pr_snapshot: dict | None = None    # design before auto-removal thinned it

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        experiment_dir: str | Path,
        *,
        range_minutes: Sequence[float] = (0, 0),
        parallel: bool = True,
        max_workers: int | None = None,
        executor: Literal["threads", "processes"] = "threads",
    ) -> ProgressiveRatioExperiment:
        from .yaml_config import load_experiment_yaml

        base = load_experiment_yaml(
            experiment_dir,
            range_minutes=range_minutes,
            parallel=parallel,
            max_workers=max_workers,
            executor=executor,
        )
        if isinstance(base, cls):
            return base
        return cls(**{f.name: getattr(base, f.name)
                      for f in dataclasses.fields(Experiment)})

    # ------------------------------------------------------------------
    # Roles: paired / yoked within Chamber Groups
    # ------------------------------------------------------------------

    def chamber_groups_by_dfm(self) -> dict[int, tuple[ChamberGroup, ...]]:
        """``{dfm_id: (ChamberGroup, ...)}``: each loaded DFM's Chamber
        Groups, in number order, from its resolved ``dfms:`` entry — the
        config's ``paired_chambers`` or ``chamber_groups``, or what the
        loader took from Program.txt
        (:func:`~pyflic.base.experiment_types.progressive_ratio.resolve_chamber_groups`).
        Validated at load time; a DFM whose groups cannot be resolved
        raises."""
        if self._pr_groups is not None:
            return self._pr_groups
        nodes = (self.config or {}).get("dfms", (self.config or {}).get("DFMs")) or []
        if isinstance(nodes, dict):
            nodes = [{"id": int(k), **dict(v)} for k, v in nodes.items()]
        out: dict[int, tuple[ChamberGroup, ...]] = {}
        for node in nodes:
            dfm_id = int(node.get("id", node.get("ID")))
            if dfm_id not in self.dfms:
                continue
            result = resolve_chamber_groups(node)
            if result.problems or result.groups is None:
                raise ValueError(
                    f"DFM {dfm_id}: " + "; ".join(result.problems))
            out[dfm_id] = result.groups
        missing = sorted(set(self.dfms) - set(out))
        if missing:
            raise ValueError(
                f"DFM(s) {missing} have no chamber groups: give '{PAIRED_KEY}', or "
                f"'chamber_groups: single' for a design without yoked controls, or a "
                f"Program.txt whose Linkage sets them.")
        self._pr_groups = out
        return out

    def chamber_groups(self, dfm_id: int) -> tuple[ChamberGroup, ...]:
        """*dfm_id*'s Chamber Groups, in number order."""
        return self.chamber_groups_by_dfm()[int(dfm_id)]

    def chamber_group(self, dfm_id: int, group: int) -> ChamberGroup:
        """Chamber Group *group* of *dfm_id*."""
        return next(g for g in self.chamber_groups(dfm_id) if g.number == int(group))

    def _group_pairs(self) -> list[tuple[int, ChamberGroup]]:
        """``(dfm_id, group)`` for every Chamber Group of every DFM, in
        order: the rows of every per-group table."""
        return [(d, g) for d in sorted(self.dfms) for g in self.chamber_groups(d)]

    def solo_groups(self) -> list[tuple[int, int]]:
        """``(dfm_id, group)`` of every solo Chamber Group: one chamber, no
        yoked control."""
        return [(d, g.number) for d, g in self._group_pairs() if g.solo]

    def has_yoked_controls(self) -> bool:
        """Whether any Chamber Group is a pair, so a Paired-Yoked Difference
        can exist."""
        return any(not g.solo for _d, g in self._group_pairs())

    def paired_chambers_by_dfm(self) -> dict[int, list[int]]:
        """``{dfm_id: [paired chamber per group]}``, in chamber order; every
        chamber of a solo design is Paired."""
        return {d: sorted(g.paired for g in groups)
                for d, groups in self.chamber_groups_by_dfm().items()}

    def role_of(self, dfm_id: int, chamber: int) -> ChamberRole:
        dfm_id, chamber = int(dfm_id), int(chamber)
        group = next(g for g in self.chamber_groups(dfm_id) if chamber in g.chambers)
        paired = chamber == group.paired
        partner = group.yoked if paired else group.paired
        return ChamberRole(dfm_id=dfm_id, chamber=chamber, group=group.number,
                           role=ROLE_PAIRED if paired else ROLE_YOKED, partner=partner)

    def roles_table(self) -> pd.DataFrame:
        """``DFM, Chamber, Group, Role, Partner`` for every chamber of every
        loaded DFM, assigned to a Treatment or not.  ``Partner`` is NA for a
        solo group's chamber."""
        rows = []
        for dfm_id in sorted(self.dfms):
            for chamber in range(1, 7):
                r = self.role_of(dfm_id, chamber)
                rows.append({"DFM": dfm_id, "Chamber": chamber, "Group": r.group,
                             "Role": r.role, "Partner": r.partner})
        out = pd.DataFrame(rows, columns=["DFM", "Chamber", "Group", "Role", "Partner"])
        out["Partner"] = _chamber_column(out["Partner"].tolist())
        return out

    def _sucrose_well(self, dfm: DFM, chamber: int) -> int:
        return int(dfm.chambers[int(chamber) - 1].well_a)

    def _chamber_wells(self, dfm: DFM, chamber: int) -> tuple[int, int]:
        ch = dfm.chambers[int(chamber) - 1]
        return int(ch.left_well), int(ch.right_well)

    # ------------------------------------------------------------------
    # Training end, per Chamber Group
    # ------------------------------------------------------------------

    def group_training(self, dfm_id: int, group: int) -> GroupTraining:
        """The group's training end, read from the paired chamber's sucrose
        well, plus notes: on generation 1 where the other wells' flags
        disagree, on generation 3 in Phase and Acclimation terms
        (:meth:`_phase_training_notes`), and on generation 2 that there is
        no Training information at all."""
        dfm_id, group = int(dfm_id), int(group)
        cache = self._pr_training if self._pr_training is not None else {}
        key = (dfm_id, group)
        if key in cache:
            return cache[key]
        dfm = self.dfms[dfm_id]
        layout = self.chamber_group(dfm_id, group)
        chambers = layout.chambers
        paired, yoked = layout.paired, layout.yoked
        sucrose = self._sucrose_well(dfm, paired)
        end = dfm.training_end_minutes(sucrose)

        if dfm.training_info == "none" or self._reads_phase(dfm):
            known = dfm.training_info != "none"
            notes = (self._phase_training_notes(dfm, sucrose, end) if known
                     else [NO_TRAINING_INFORMATION])
            result = GroupTraining(dfm_id=dfm_id, group=group, paired_chamber=paired,
                                   yoked_chamber=yoked, sucrose_well=sucrose,
                                   training_end=end if known else None,
                                   notes=tuple(notes), known=known)
            cache[key] = result
            self._pr_training = cache
            return result

        notes: list[str] = []
        any_flag = False
        ## Wells in the same state are reported together: with the firmware
        ## seen so far every non-sucrose well stays flagged, and three
        ## identical lines per group would bury the one that matters.
        stayed: list[str] = []
        never: list[str] = []
        for chamber in chambers:
            role = ROLE_PAIRED if chamber == paired else ROLE_YOKED
            for well in self._chamber_wells(dfm, chamber):
                state = dfm.training_flag_state(well)
                any_flag = any_flag or state != "never_flagged"
                if well == sucrose:
                    continue
                label = (f"W{well} ({role} ch{chamber} "
                         f"{'A' if well == self._sucrose_well(dfm, chamber) else 'B'})")
                if state == "never_cleared":
                    stayed.append(label)
                elif state == "never_flagged":
                    never.append(label)
                else:
                    t = dfm.training_end_minutes(well)
                    if end is None:
                        notes.append(f"{label} cleared at {t:.1f} min while the "
                                     f"paired sucrose well W{sucrose} never cleared")
                    elif abs(float(t) - float(end)) > _FLAG_TOLERANCE_MIN:
                        notes.append(f"{label} cleared at {t:.1f} min, paired "
                                     f"sucrose well W{sucrose} at {end:.1f} min")
        if stayed:
            notes.append(f"{', '.join(stayed)} stayed flagged in training to the "
                         f"end of the recording")
        if never and any_flag:
            notes.append(f"{', '.join(never)} never carried the training flag")
        if not any_flag:
            ## No flag on any well of the group: not a disagreement between
            ## wells but an absence of training data (a v2 file, or a run that
            ## never entered training).  One note, not four.
            notes = [f"no training flag on any well of group {group}"]
        elif end is None:
            state = dfm.training_flag_state(sucrose)
            notes.insert(0, f"paired sucrose well W{sucrose} "
                            f"{'stayed flagged to the end of the recording' if state == 'never_cleared' else 'never carried the training flag'}"
                            f" — training never completed")
        result = GroupTraining(dfm_id=dfm_id, group=group, paired_chamber=paired,
                               yoked_chamber=yoked, sucrose_well=sucrose,
                               training_end=end, notes=tuple(notes))
        cache[key] = result
        self._pr_training = cache
        return result

    @staticmethod
    def _reads_phase(dfm: DFM) -> bool:
        """Whether *dfm* records the firmware's Paradigm state (generation
        3), so Training, Light Events and light come from it."""
        return dfm.generation == GENERATION_PARADIGM and dfm.paradigm is not None

    def _phase_training_notes(self, dfm: DFM, sucrose: int, end: float | None) -> list[str]:
        """Generation 3's notes on one group's Training, in Phase and
        Acclimation terms.

        Only the Paired Sucrose Well is read: the group's other three wells
        are ``Off`` and read Phase ``M`` throughout, so they carry nothing to
        disagree about, unlike generation 1's per-well flags.  Noted: a
        training that never completed and why, an empty one (decision 11), a
        Paradigm that restarted (a later run in Acclimation) and a later ratio
        run, which falls outside the Test window (decision 12).
        """
        state = dfm.paradigm
        w = int(sucrose) - 1
        notes: list[str] = []
        if end is None:
            if not state.ratio_rows(sucrose).any():
                why = "never ran a ratio Paradigm"
            elif len(state) and state.phase_a[-1, w]:
                why = "was still in Acclimation (Phase A) when the recording ended"
            else:
                why = "never reached Phase M under a ratio Paradigm"
            notes.append(f"paired Sucrose Well W{sucrose} {why} — training never completed")
        elif (dfm.training_flag_state(sucrose) == "never_flagged"
              and (state.training_end(sucrose) or 0) > 0):
            ## Row 0 is a range filter starting inside Test: Training lies
            ## before the rows loaded, not nowhere.
            notes.append(EMPTY_TRAINING_NOTE)
        test_run = self._test_ratio_run(state, sucrose)
        later = ([run for run in _true_runs(state.ratio_rows(sucrose)) if run[0] >= test_run[1]]
                 if test_run is not None else [])
        ## A DFM restart (decision 15) starts the Paradigm state over too:
        ## said so, with its minute, where the state starts again after it.
        rebooted = {span.resume_row: span for span in state.restarts()}

        def after_restart(i0: int) -> str:
            span = rebooted.get(i0)
            return ("" if span is None else
                    f" after the DFM restarted at minute {span.minute:.1f}")

        restarted: set[int] = set()
        for i0, _i1 in state.acclimation_runs(sucrose)[1:]:
            note = (f"Paradigm restarted at Program Interval {_interval_at(state, i0)}"
                    f"{after_restart(i0)}")
            run = next((r for r in later if r[0] <= i0 < r[1]), None)
            if run is not None:
                restarted.add(run[0])
                note += (" (after the Test window, which ends at the restart)" if i0 in rebooted
                         else " (after the Test window, which is the first ratio run only)")
            notes.append(note)
        for i0, _i1 in later:
            if i0 not in restarted:
                notes.append(f"a ratio Paradigm ran again from Program Interval "
                             f"{_interval_at(state, i0)}{after_restart(i0)}, after the Test "
                             f"window (" + ("which ends at the restart)" if i0 in rebooted
                                            else "the first ratio run only)"))
        return notes

    def test_restart(self, dfm_id: int, group: int):
        """The first DFM restart (a
        :class:`~pyflic.base.paradigm_state.RestartSpan`) inside the group's
        Test window, which then ends at it (decision 15); ``None`` when the
        DFM did not restart there, off generation 3, and for a group whose
        training never ended.

        The Test window is the Paired Sucrose Well's ratio run that holds
        its training end; a restart's unknown-instruction rows end that run,
        so a restart is inside it when its first such row is the run's end
        or falls within it."""
        dfm = self.dfms[int(dfm_id)]
        if not self._reads_phase(dfm):
            return None
        gt = self.group_training(dfm_id, group)
        if not gt.complete:
            return None
        state = dfm.paradigm
        run = self._test_ratio_run(state, gt.sucrose_well)
        start = state.training_end(gt.sucrose_well)
        if run is None or start is None:
            return None
        return next((span for span in state.restarts() if start < span.row <= run[1]), None)

    @staticmethod
    def _test_ratio_run(state: ParadigmState, well: int) -> tuple[int, int] | None:
        """``[i0, i1)`` of the run of rows in which *well* is under a ratio
        Paradigm that holds its training end: the Test (decision 12).
        ``None`` when training never ended."""
        end = state.training_end(well)
        if end is None:
            return None
        return next((run for run in _true_runs(state.ratio_rows(well))
                     if run[0] <= end < run[1]), None)

    def test_paradigm(self, dfm_id: int, group: int) -> tuple[str | None, Any]:
        """``(name, paradigm)`` of the group's **Test Paradigm**, the Paired
        Sucrose Well's Paradigm on its training-end row (generation 3), with
        its :class:`~pyflic.base.opto_program.Paradigm` from ``Program.txt``
        — ``None`` when the program does not define it or there is none, so
        its type is unknown.  ``(None, None)`` on generations 1 and 2 and for
        a group whose training never ended."""
        dfm = self.dfms[int(dfm_id)]
        gt = self.group_training(dfm_id, group)
        if not self._reads_phase(dfm) or not gt.complete:
            return None, None
        row = dfm.paradigm.training_end(gt.sucrose_well)
        if row is None:
            return None, None
        name = dfm.paradigm.paradigm[row]
        if pd.isna(name):
            return None, None
        program = getattr(self, "opto_program", None)
        paradigms = getattr(program, "paradigms", None) or {}
        return str(name), paradigms.get(str(name))

    def test_paradigm_type(self, dfm_id: int, group: int) -> str | None:
        """The group's Test Paradigm type in PyMCU spelling
        (``ProgressiveRatio`` or ``FixedRatio``), ``pr_breaking_point.csv``'s
        ``ParadigmType``.  On generation 3 the type of :meth:`test_paradigm`;
        on generations 1 and 2 ``ProgressiveRatio`` when a legacy
        ``Program.txt`` has the DFM's section, the firmware that wrote one
        having no other ratio Paradigm.  ``None`` when nothing says: no
        program for the DFM, or a Test Paradigm the program does not define."""
        if self._reads_phase(self.dfms[int(dfm_id)]):
            _name, paradigm = self.test_paradigm(dfm_id, group)
            return paradigm.type_name if paradigm is not None else None
        program = getattr(self, "opto_program", None)
        if program is not None and program.grammar == GRAMMAR_LEGACY \
                and program.section(dfm_id) is not None:
            return TYPE_NAMES[PROGRESSIVE_RATIO]
        return None

    def _fixed_ratio(self, dfm_id: int, group: int) -> bool:
        """Whether the group's Test Paradigm is a FixedRatio (decision 14).
        Without its type (no program) it is taken as progressive."""
        _name, paradigm = self.test_paradigm(dfm_id, group)
        return paradigm is not None and paradigm.type == FIXED_RATIO

    def training_information(self) -> str | None:
        """:data:`NO_TRAINING_INFORMATION` when DFMs of this experiment
        recorded no Training (generation 2, decision 15), prefixed with
        those DFMs when only some did not; ``None`` when every DFM did.  For
        ``summary.txt``, the report, the Hub status and the Project Report."""
        missing = sorted(int(d) for d, dfm in self.dfms.items()
                         if dfm.training_info == "none")
        if not missing:
            return None
        if len(missing) == len(self.dfms):
            return NO_TRAINING_INFORMATION
        return f"DFM {', '.join(str(d) for d in missing)}: {NO_TRAINING_INFORMATION}"

    def training_table(self) -> pd.DataFrame:
        """One row per Chamber Group: ``DFM, Group, PairedChamber, YokedChamber,
        SucroseWell, TrainingEndMin, TrainingComplete, Notes``.
        ``TrainingComplete`` is NA on a DFM that records no Training, and
        ``YokedChamber`` for a solo group."""
        rows = []
        for dfm_id, layout in self._group_pairs():
            gt = self.group_training(dfm_id, layout.number)
            rows.append({
                "DFM": dfm_id, "Group": layout.number,
                "PairedChamber": gt.paired_chamber,
                "YokedChamber": gt.yoked_chamber,
                "SucroseWell": gt.sucrose_well,
                "TrainingEndMin": (np.nan if gt.training_end is None
                                   else float(gt.training_end)),
                "TrainingComplete": gt.training_complete,
                "Notes": "; ".join(gt.notes),
            })
        out = pd.DataFrame(rows)
        if not out.empty:
            out["YokedChamber"] = _chamber_column(out["YokedChamber"].tolist())
        return out

    def training_warnings(self) -> list[str]:
        """Human-readable Training notes, for the summary and QC: per-well
        flag disagreements on generation 1, Phase and Acclimation notes on
        generation 3, and one line per DFM that records no Training."""
        out: list[str] = []
        for dfm_id in sorted(self.dfms):
            if self.dfms[dfm_id].training_info == "none":
                out.append(f"DFM {dfm_id}: {NO_TRAINING_INFORMATION}")
                continue
            for layout in self.chamber_groups(dfm_id):
                group = layout.number
                gt = self.group_training(dfm_id, group)
                for note in gt.notes:
                    out.append(f"DFM {dfm_id} group {group}: {note}")
        return out

    def _recording_end(self, dfm: DFM) -> float:
        return float(pd.to_numeric(dfm.raw_df["Minutes"], errors="coerce").max())

    # ------------------------------------------------------------------
    # Light on, per chamber
    # ------------------------------------------------------------------

    def _chamber_light(self, dfm: DFM, chamber: int) -> pd.Series:
        """Boolean per-sample light state for *chamber*: either well's bit
        in ``OptoCol1``, or on generation 3 either well's LED active at some
        point in the record (``OnMs`` above 0)."""
        w1, w2 = self._chamber_wells(dfm, chamber)
        if self._reads_phase(dfm):
            lit = dfm.paradigm.on_ms[:, [w1 - 1, w2 - 1]].max(axis=1) > 0
            return pd.Series(lit, index=dfm.lick_df.index)
        lights = dfm.lights_df
        if lights is None or f"W{w1}" not in lights.columns:
            return pd.Series(False, index=lights.index if lights is not None else [])
        return lights[f"W{w1}"].astype(bool) | lights[f"W{w2}"].astype(bool)

    def _light_on_seconds(self, dfm: DFM, chamber: int,
                          range_minutes: Sequence[float]) -> float:
        """``LightOn_sec`` over the (a, b] window.  Generation 3: the
        record-by-record larger of the chamber's two wells' ``OnMs``, summed
        and in seconds — exact, and the two wells' light counted once
        (decision 31).  Otherwise the lit ``OptoCol1`` samples, one sample
        period each."""
        if self._reads_phase(dfm):
            w1, w2 = self._chamber_wells(dfm, chamber)
            on_ms = dfm.paradigm.on_ms[:, [w1 - 1, w2 - 1]].max(axis=1)
            if range_is_specified(range_minutes):
                a, b = range_bounds(range_minutes)
                mins = dfm.paradigm.minutes
                on_ms = on_ms[(mins > a) & (mins <= b)]
            return float(on_ms.sum()) / 1000.0
        on = self._chamber_light(dfm, chamber)
        if range_is_specified(range_minutes):
            a, b = range_bounds(range_minutes)
            mins = dfm.lights_df["Minutes"].to_numpy(dtype=float)
            on = on[(mins > a) & (mins <= b)]
        return float(on.sum()) / float(dfm.params.samples_per_second)

    # ------------------------------------------------------------------
    # Light QC: did the Paired fly earn its light?  (pr_light_qc)
    # ------------------------------------------------------------------
    #
    # The firmware lights a group from its own reading of the Paired
    # chamber's Sucrose Well during the run; pyflic counts licks afterwards,
    # from the baselined signal.  A Sucrose Well whose resting level creeps up
    # looks continuously touched to the firmware and flat to pyflic, so the
    # light runs on its own and every light-on number describes the sensor.
    # These methods measure that disagreement per Chamber Group, over the
    # whole recording whatever window a table uses: they describe the
    # hardware, not a window.

    def light_qc_settings(self) -> lqc.LightQCSettings:
        """The light QC's thresholds, from the design's ``constants:``."""
        return lqc.LightQCSettings.from_constants(self.global_constants)

    def _design_snapshot(self) -> dict[tuple[int, int], str]:
        """``{(dfm, chamber): treatment}`` as the design stood before
        auto-removal first thinned it.

        The light QC table and the QC figures keep an auto-removed group in
        view — seeing why it left is what they are for — while a chamber
        excluded by hand in ``remove_chambers.csv`` never enters the design and
        stays out of both.
        """
        if self._pr_snapshot is None:
            self._pr_snapshot = {
                (int(tc.dfm_id), int(tc.chamber_index)): name
                for name, treatment in self.design.treatments.items()
                for tc in treatment.chambers
            }
        return self._pr_snapshot

    def _remove_chambers_from_design(self, remove_set: set[tuple[int, int]]) -> None:
        self._design_snapshot()
        Experiment._remove_chambers_from_design(self, remove_set)

    def _group_treatment(self, dfm_id: int, group: int) -> str:
        snapshot = self._design_snapshot()
        return next((snapshot[(int(dfm_id), c)]
                     for c in self.chamber_group(dfm_id, group).chambers
                     if snapshot.get((int(dfm_id), c))), "")

    def _light_cache(self, dfm_id: int) -> dict:
        """The light QC's cache for one DFM, dropped whenever that DFM object
        is replaced — a QC Viewer recompute or a parameter sweep swaps in DFMs
        re-detected under other parameters, and their licks differ."""
        dfm = self.dfms[int(dfm_id)]
        if self._pr_light is None:
            self._pr_light = {}
        entry = self._pr_light.get(int(dfm_id))
        if entry is None or entry[0] is not dfm:
            entry = (dfm, {})
            self._pr_light[int(dfm_id)] = entry
        return entry[1]

    def resting_levels(self, dfm_id: int) -> pd.DataFrame:
        """Every well's Resting Level on one DFM: the per-minute median of its
        raw, un-baselined signal (:func:`pr_light_qc.resting_levels`)."""
        cache = self._light_cache(dfm_id)
        if "resting" not in cache:
            dfm = self.dfms[int(dfm_id)]
            minutes = pd.to_numeric(dfm.raw_df["Minutes"], errors="coerce")
            cache["resting"] = lqc.resting_levels(dfm.raw_df,
                                                  minutes.to_numpy(dtype=float))
        return cache["resting"]

    def resting_reference(self, dfm_id: int, well: int) -> pd.Series:
        """Per-minute median Resting Level of the DFM's *other* Sucrose Wells.

        Sucrose Wells only: the yeast wells of a long run drift by hundreds of
        counts, and a reference that climbs with them would hide exactly the
        slow rise of one Sucrose Well this is here to show.
        """
        dfm = self.dfms[int(dfm_id)]
        levels = self.resting_levels(dfm_id)
        others = [f"W{int(ch.well_a)}" for ch in dfm.chambers
                  if int(ch.well_a) != int(well) and f"W{int(ch.well_a)}" in levels.columns]
        if not others:
            return pd.Series(dtype=float)
        return levels[others].median(axis=1)

    def light_events_table(self, dfm_id: int, group: int) -> pd.DataFrame:
        """Every Light Event of one Chamber Group, Training and Test.

        Light and licks are the Paired chamber's: its Sucrose Well is the one
        the firmware watches.  A Light Event is an onset of the chamber's
        ``OptoCol1`` light, or on generation 3 an Activation of the Sucrose
        Well (:meth:`_activation_events`; an Acclimation Activation is a
        Training Light Event).  Columns:

        * ``Facet`` — Training or Test (onset after the group's training end,
          on generation 1 after the row that clears the flag:
          :meth:`_test_onsets_after`; ``Recording`` on a DFM that records no
          Training);
        * ``RecordingMinute`` — the onset, in recording minutes;
        * ``Minutes`` — the onset in minutes since training end (NaN when
          training never ended);
        * ``CumLicks`` — Sucrose Well licks since training end, at the onset;
        * ``MinutesSincePrev`` — onset to onset (NaN for the first event);
        * ``LicksSincePrev`` — Sucrose Well licks from the end of the previous
          event to the end of this one (:func:`pr_light_qc.licks_between_events`),
          so the first Test event counts from the last Training one;
        * ``Explained`` — Sucrose Well activity, a feeding lick or a tasting
          touch, within the light's decay plus the tolerance before the event
          to the tolerance after it (the Opto Program's decay; without one,
          ``opto_default_decay_ms``) — Explained Light, as the optogenetic
          light QC defines it;
        * ``LickFree`` — ``LicksSincePrev == 0`` and not ``Explained``;
        * ``RestingLevel`` — the Sucrose Well's Resting Level in the onset's
          minute;
        * ``Requirement``, ``FirmwareEvents``, ``Phase`` — the firmware's
          record of the Activation (generation 3; NA otherwise): the
          Requirement it met, the Firmware Events it took (from the previous
          onset, exclusive, to this one, inclusive) and the Sucrose Well's
          Phase, ``A`` or ``M``, on its onset row
          (:meth:`_firmware_columns`).
        """
        cache = self._light_cache(dfm_id)
        key = ("events", int(group))
        if key in cache:
            return cache[key]
        dfm = self.dfms[int(dfm_id)]
        gt = self.group_training(dfm_id, group)
        mins = pd.to_numeric(dfm.lick_df["Minutes"], errors="coerce").to_numpy(dtype=float)
        licks = dfm.lick_df[f"W{gt.sucrose_well}"].to_numpy(dtype=bool)
        if self._reads_phase(dfm):
            onsets, ends = self._activation_events(dfm.paradigm, gt.sucrose_well)
        else:
            light = self._chamber_light(dfm, gt.paired_chamber).to_numpy(dtype=bool)
            onsets, ends = lqc.light_events(light)
        on_min = mins[onsets]
        end = float(gt.training_end) if gt.complete else np.inf
        cum = np.cumsum(licks & (mins > end))
        counts = lqc.licks_between_events(licks, ends)
        ## Decay-aware: the firmware keeps the light on for the program's decay
        ## after the lick that earned it, and a brief touch it counts may sit
        ## under pyflic's feeding threshold.  Activity at the Sucrose Well
        ## within that window explains the event, as it explains light for the
        ## optogenetic light QC; only an event with neither is lick-free.
        column = f"W{gt.sucrose_well}"
        tasting = (dfm.tasting_df[column].to_numpy(dtype=bool)
                   if dfm.tasting_df is not None and column in dfm.tasting_df.columns
                   else np.zeros(licks.size, dtype=bool))
        explained = self._events_explained(dfm_id, licks | tasting, onsets, ends)
        levels = self.resting_levels(dfm_id)
        column = f"W{gt.sucrose_well}"
        if column in levels.columns and onsets.size:
            rest = levels[column].reindex(np.floor(on_min).astype(int)).to_numpy(dtype=float)
        else:
            rest = np.full(onsets.size, np.nan)
        if gt.known:
            facet = np.where(on_min > self._test_onsets_after(dfm, gt), TEST_LABEL,
                             TRAINING_LABEL)
        else:
            facet = np.full(onsets.size, RECORDING_LABEL)
        table = pd.DataFrame({
            "Facet": facet,
            "RecordingMinute": on_min,
            "Minutes": (on_min - end) if gt.complete else np.full(onsets.size, np.nan),
            "CumLicks": cum[onsets].astype(float),
            "MinutesSincePrev": np.diff(on_min, prepend=np.nan),
            "LicksSincePrev": counts,
            "Explained": explained,
            "LickFree": (counts == 0) & ~explained,
            "RestingLevel": rest,
            **self._firmware_columns(dfm, gt.sucrose_well, onsets),
        })
        cache[key] = table
        return table

    def _test_onsets_after(self, dfm: DFM, gt: GroupTraining) -> float:
        """The recording minute after which a Light Event's onset is a Test
        one: the group's training end, or on generation 1 the row after it.

        Generation 1's training end is the last row its flag marks, and the
        firmware lights the last Training pairing in the record that clears
        the flag: in MBON02, 46 of 48 groups' first unflagged row holds an
        onset credited with Training licks, which made every Breaking Point
        one too many.  So an onset within one sample period of the training
        end is Training's (review C10).  Generation 3 records the Phase, and
        its first Main row's Activation is the first Test one."""
        if not gt.complete:
            return np.inf
        end = float(gt.training_end)
        if self._reads_phase(dfm):
            return end
        period = 1.0 / (60.0 * float(dfm.params.samples_per_second))
        ## Half a period of slack for the minutes' floating point.
        return end + 1.5 * period

    @staticmethod
    def _activation_events(state: ParadigmState, well: int) -> tuple[np.ndarray, np.ndarray]:
        """``(onsets, ends)`` of *well*'s Activations, as row indices: each
        starts on the row where Activations rose and ends on the first later
        row where the well's ``OnMs`` is 0, or at the next onset, whichever
        comes first (``len(state)`` for one lit to the end)."""
        onsets = state.activation_onsets(well).astype(int)
        n = len(state)
        if not onsets.size:
            return onsets, onsets.copy()
        ## The first dark row after each onset; n where none follows.
        dark = np.append(np.flatnonzero(state.on_ms[:, int(well) - 1] == 0), n)
        first_dark = dark[np.searchsorted(dark[:-1], onsets, side="right")]
        following = np.append(onsets[1:], n)
        return onsets, np.minimum(first_dark, following).astype(int)

    def _firmware_columns(self, dfm: DFM, well: int, onsets: np.ndarray) -> dict[str, Any]:
        """``Requirement``, ``FirmwareEvents`` and ``Phase`` for Light Events
        starting at *onsets* — the firmware's record on generation 3
        (decision 13), all NA otherwise.

        ``Requirement`` is :meth:`ParadigmState.requirement_met`; for an
        Activation on the first row of a new Paradigm, whose Requirement no
        row shows, it is the program's ``requirement_for(k)`` when
        ``Program.txt`` defines that ratio Paradigm (k counts the Main
        Activations of the row), and NA otherwise.  An Acclimation
        Activation meets none.  ``FirmwareEvents`` counts rows from the
        previous onset, exclusive, to this one, inclusive: the Events that
        built the Progress and the one that completed it, so under a ratio
        Paradigm in Main it equals the Requirement met.
        """
        n = int(np.asarray(onsets).size)
        if not self._reads_phase(dfm):
            return {"Requirement": pd.array([pd.NA] * n, dtype="Int64"),
                    "FirmwareEvents": pd.array([pd.NA] * n, dtype="Int64"),
                    "Phase": pd.array([pd.NA] * n, dtype="string")}
        state = dfm.paradigm
        w = int(well) - 1
        names = state.paradigm
        program = getattr(self, "opto_program", None)
        paradigms = getattr(program, "paradigms", None) or {}
        requirement: list[int | None] = []
        events: list[int] = []
        previous, same_row = -1, 0
        for onset in np.asarray(onsets, dtype=int):
            same_row = same_row + 1 if onset == previous else 0
            met = state.requirement_met(well, int(onset))
            if (met is None and 0 < onset < len(state) and not state.phase_a[onset, w]
                    and _new_paradigm(names, int(onset))):
                paradigm = paradigms.get(str(names[onset])) if not pd.isna(names[onset]) else None
                if paradigm is not None and paradigm.type in RATIO_PARADIGMS:
                    met = paradigm.requirement_for(1 + same_row)
            requirement.append(met)
            events.append(state.firmware_events_between(well, previous + 1, int(onset) + 1))
            previous = int(onset)
        phase = np.where(state.phase_a[np.asarray(onsets, dtype=int), w], "A", "M") \
            if n else np.zeros(0, dtype=object)
        return {"Requirement": pd.array(requirement, dtype="Int64"),
                "FirmwareEvents": pd.array(events, dtype="Int64"),
                "Phase": pd.array(list(phase), dtype="string")}

    def _events_explained(self, dfm_id: int, activity: np.ndarray,
                          onsets: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Whether *activity* falls from each event's onset less the decay and
        tolerance to its end plus the tolerance."""
        onsets = np.asarray(onsets, dtype=int)
        if onsets.size == 0:
            return np.zeros(0, dtype=bool)
        n = int(np.asarray(activity).size)
        decay = self.opto.decay_samples(dfm_id)
        tol = int(self.opto.settings().tolerance_samples)
        cum = np.concatenate([[0], np.cumsum(np.asarray(activity, dtype=bool),
                                             dtype=np.int64)])
        lo = np.clip(onsets - decay[onsets] - tol, 0, n)
        hi = np.clip(np.asarray(ends, dtype=int) + tol, 0, n)
        return (cum[hi] - cum[lo]) > 0

    def _light_qc_row(self, dfm_id: int, group: int,
                      settings: lqc.LightQCSettings) -> dict:
        cache = self._light_cache(dfm_id)
        key = ("qc", int(group), settings)
        if key in cache:
            return cache[key]
        dfm = self.dfms[int(dfm_id)]
        gt = self.group_training(dfm_id, group)
        events = self.light_events_table(dfm_id, group)
        test = events[events["Facet"] == TEST_LABEL]
        training = events[events["Facet"] == TRAINING_LABEL]
        mins = pd.to_numeric(dfm.lick_df["Minutes"], errors="coerce").to_numpy(dtype=float)
        licks = dfm.lick_df[f"W{gt.sucrose_well}"].to_numpy(dtype=bool)
        end = float(gt.training_end) if gt.complete else np.inf
        training_licks = int(licks[mins <= end].sum())
        levels = self.resting_levels(dfm_id)
        column = f"W{gt.sucrose_well}"
        resting = (lqc.resting_summary(levels[column]) if column in levels.columns
                   else lqc.resting_summary(pd.Series(dtype=float)))
        reference = self.resting_reference(dfm_id, gt.sucrose_well)
        reference_level = float(reference.median()) if not reference.empty else np.nan
        phase = self._reads_phase(dfm)
        name, paradigm = self.test_paradigm(dfm_id, group)
        ## Requirement series wrong (decision 6), on the Test window's Main
        ## Activations of the Paired Sucrose Well against the Test Paradigm's
        ## series: lqc.requirement_series_mismatches calls the optogenetic
        ## light QC's own check (rule_checks) rather than repeating it.
        series = (lqc.requirement_series_mismatches(
                      self._test_ledger_rows(dfm_id, group), paradigm)
                  if phase and paradigm is not None and paradigm.type in RATIO_PARADIGMS
                  else None)
        windowed = test[self._in_test_window(dfm_id, group, test)]
        verdict = lqc.judge_group(
            settings,
            training_complete=gt.training_complete,
            training_light_events=len(training),
            training_licks=training_licks,
            test_counts=test["LicksSincePrev"].to_numpy(),
            resting=resting,
            reference_level=reference_level,
            any_light=len(events) > 0,
            lick_free=test["LickFree"].to_numpy(dtype=bool),
            firmware_events=(windowed["FirmwareEvents"].to_numpy(dtype=float, na_value=np.nan)
                             if phase else None),
            check_trend=not self._fixed_ratio(dfm_id, group),
            light_source=("Activations at the paired Sucrose Well" if phase
                          else "OptoCol1 light bits"),
            series_mismatches=series,
            trend_counts=windowed["LicksSincePrev"].to_numpy(),
        )
        notes = list(verdict.notes)
        if name is not None and paradigm is None:
            notes.append(f"the Test Paradigm {name}'s type is unknown without its "
                         f"Program.txt entry, so it is judged as a progressive ratio")
        restart = self.test_restart(dfm_id, group)
        if restart is not None:
            notes.append(restart_note(restart, float(gt.training_end)))
        run_start = (float(test["Minutes"].iloc[verdict.lick_free_run_start])
                     if verdict.lick_free_run_start is not None else np.nan)
        row = {
            "DFM": int(dfm_id), "Group": int(group),
            "PairedChamber": gt.paired_chamber, "YokedChamber": gt.yoked_chamber,
            "SucroseWell": gt.sucrose_well,
            "Treatment": self._group_treatment(dfm_id, group),
            "TrainingComplete": gt.training_complete,
            "TrainingMinutes": float(gt.training_end) if gt.complete else np.nan,
            ## A recording without Training has neither phase to count in.
            "TrainingLightEvents": int(len(training)) if gt.known else np.nan,
            "TrainingLicks": training_licks if gt.known else np.nan,
            "TestLightEvents": int(len(test)) if gt.known else np.nan,
            "LickFreeEvents": int(verdict.lick_free_events) if gt.complete else np.nan,
            "LongestLickFreeRun": int(verdict.longest_lick_free_run) if gt.complete else np.nan,
            "LickFreeRunStartMin": run_start,
            "TrendRho": verdict.trend_rho,
            "TrendSlope": verdict.trend_slope,
            "TrendRhoFirmware": verdict.trend_rho_firmware,
            "RestingStart": resting["start"],
            "RestingMax": resting["max"],
            "RestingEnd": resting["end"],
            "RestingRise": resting["max"] - resting["start"],
            "RestingRatio": lqc.resting_ratio(resting["level"], reference_level),
            "Flags": ", ".join(verdict.flags),
            "Verdict": verdict.verdict,
            "Excluded": bool(verdict.failed and settings.exclude),
            "Notes": "; ".join(notes),
        }
        cache[key] = row
        return row

    def light_qc_table(self) -> pd.DataFrame:
        """One row per Chamber Group: did the Paired fly earn its light?

        ``Flags`` lists what fired — *self-triggered light*, *implausible
        training* and, on generation 3, *requirement series wrong* fail the
        group; *no increasing trend*, *resting level rise* and *resting level
        elevated* are warnings.  ``YokedChamber`` is NA for a solo group, and
        ``Notes`` names a DFM restart that ends the Test window.  ``Verdict`` is ``ok``,
        ``warning`` or ``failed``; ``Excluded`` says a failed group leaves the
        analysis (``exclude_failed_pr_groups``, on by default).
        ``LickFreeRunStartMin`` is where the first failing run of Lick-free
        Light Events begins, in minutes since training end — the latest a
        hand-set cutoff could fall; the licks-per-event figure shows whether
        the group degraded earlier.  The definitions are in
        :mod:`pyflic.base.pr_light_qc`.
        """
        settings = self.light_qc_settings()
        rows = [self._light_qc_row(dfm_id, layout.number, settings)
                for dfm_id, layout in self._group_pairs()]
        out = pd.DataFrame(rows, columns=list(LIGHT_QC_COLUMNS))
        if not out.empty:
            out["YokedChamber"] = _chamber_column(out["YokedChamber"].tolist())
        return out

    def _light_qc_by_group(self) -> dict[tuple[int, int], dict]:
        settings = self.light_qc_settings()
        return {(int(d), int(g.number)): self._light_qc_row(d, g.number, settings)
                for d, g in self._group_pairs()}

    def _test_ledger_rows(self, dfm_id: int, group: int) -> pd.DataFrame:
        """The Activation ledger's rows
        (:func:`~pyflic.base.activation_ledger.ledger`) of the group's Paired
        Sucrose Well inside its Test window: Phase ``M``, from training end to
        the end of the Test ratio run (or the restart that ends it).  The
        ledger is computed once per DFM, with the Linkage the optogenetic
        light QC passes it.  Empty off generation 3 and for a group whose
        training never ended."""
        from . import activation_ledger as al

        dfm = self.dfms[int(dfm_id)]
        gt = self.group_training(dfm_id, group)
        empty = pd.DataFrame(columns=list(al.LEDGER_COLUMNS))
        if not self._reads_phase(dfm) or not gt.complete:
            return empty
        cache = self._light_cache(dfm_id)
        if "ledger" not in cache:
            state = dfm.paradigm
            program = getattr(self, "opto_program", None)
            section = program.section(int(dfm_id)) if program is not None else None
            if section is not None:
                groups = list(section.linkage_groups().values())
            else:
                from .opto_roles import linkage_view

                groups = list(linkage_view(None, state, None).groups.values())
            wells = sorted({self.group_training(dfm_id, layout.number).sucrose_well
                            for layout in self.chamber_groups(dfm_id)})
            cache["ledger"] = al.ledger(state, wells=wells,
                                        linked={w: tuple(g) for g in groups for w in g})
        book = cache["ledger"]
        state = dfm.paradigm
        start = state.training_end(gt.sucrose_well)
        run = self._test_ratio_run(state, gt.sucrose_well)
        if start is None or run is None:
            return empty
        stop = run[1]
        restart = self.test_restart(dfm_id, group)
        if restart is not None:
            stop = min(stop, restart.row)
        rows = book["Row"].to_numpy(dtype=np.int64)
        keep = ((book["Well"].to_numpy(dtype=np.int64) == int(gt.sucrose_well))
                & (rows >= int(start)) & (rows < int(stop))
                & (book["Phase"].astype(str).to_numpy() == "M"))
        return book[keep].reset_index(drop=True)

    def light_qc_failed_groups(self) -> dict[tuple[int, int], list[str]]:
        """``{(dfm, group): failing flags}`` for the groups the light QC
        excludes — empty when ``exclude_failed_pr_groups`` is off."""
        out: dict[tuple[int, int], list[str]] = {}
        for key, row in self._light_qc_by_group().items():
            if row["Excluded"]:
                out[key] = [f for f in str(row["Flags"]).split(", ")
                            if f in lqc.FAILING_FLAGS]
        return out

    def estimated_increment(self) -> tuple[float, float, int] | None:
        """``(slope, intercept, n_groups)`` of the requirement across this
        experiment's groups whose Test Paradigm is progressive
        (:func:`pr_light_qc.estimate_increment`), for the reference line and
        the summary; ``None`` when no group qualifies.  A fixed ratio's
        requirement does not rise, so its groups give none (decision 14)."""
        series = []
        for dfm_id, layout in self._group_pairs():
            group = layout.number
            if not self.group_training(dfm_id, group).complete or self._fixed_ratio(dfm_id, group):
                continue
            test = self._test_events(dfm_id, group)
            series.append(test.loc[self._in_test_window(dfm_id, group, test),
                                   "LicksSincePrev"].to_numpy())
        return lqc.estimate_increment(series)

    def _in_test_window(self, dfm_id: int, group: int, test: pd.DataFrame) -> np.ndarray:
        """``(n,)`` bool over the Test rows of :meth:`light_events_table`:
        the onsets inside the group's Test window (:meth:`_test_window`),
        the events the Breaking Point is read from.  The licks-per-event
        trend and the requirement estimate are read from the same events, not
        from a later ratio run, the Acclimation after a DFM restart, or past
        ``pr_test_window_min``."""
        window = self._test_window(dfm_id, group, self.break_settings())
        minutes = test["Minutes"].to_numpy(dtype=float)
        if window is None:
            return np.ones(minutes.size, dtype=bool)
        return minutes <= float(window)

    def progressive_groups(self, dfm_id: int | None = None) -> list[tuple[int, int]]:
        """``(dfm_id, group)`` of the groups whose Test Paradigm is, or is
        taken as, a progressive ratio: every group but a FixedRatio one
        (decision 14), on *dfm_id* alone when given.  The requirement
        estimate and the wording about a climbing requirement are theirs."""
        return [(d, g.number) for d, g in self._group_pairs()
                if (dfm_id is None or d == int(dfm_id)) and not self._fixed_ratio(d, g.number)]

    def light_events_ledger(self) -> pd.DataFrame:
        """Every group's Test-phase Light Event Ledger stacked — the Paired
        chamber's :meth:`breaking_point_table` with ``DFM, Group,
        PairedChamber, Event`` in front; ``Counted`` marks the events the
        group's Breaking Point holds.  ``analysis/pr_light_events.csv``."""
        cols = ["DFM", "Group", "PairedChamber", "Event",
                *BREAKING_POINT_COLUMNS, *LEDGER_COLUMNS, COUNTED_COLUMN]
        frames = []
        for dfm_id, layout in self._group_pairs():
            group = layout.number
            gt = self.group_training(dfm_id, group)
            bp = self.breaking_point_table(dfm_id, gt.paired_chamber)
            if bp.empty:
                continue
            bp = bp.copy()
            bp.insert(0, "Event", np.arange(1, len(bp) + 1))
            bp.insert(0, "PairedChamber", gt.paired_chamber)
            bp.insert(0, "Group", group)
            bp.insert(0, "DFM", dfm_id)
            frames.append(bp)
        if not frames:
            return pd.DataFrame(columns=cols)
        return pd.concat(frames, ignore_index=True)[cols]

    def write_light_qc(self) -> dict[str, Path]:
        """Write ``analysis/pr_light_qc.csv`` (one row per Chamber Group) and
        ``analysis/pr_light_events.csv`` (one row per Test Light Event)."""
        if self.analysis_dir is None:
            raise ValueError("experiment_dir must be set to write the light QC.")
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        qc_path = self.analysis_dir / "pr_light_qc.csv"
        self.light_qc_table().to_csv(qc_path, index=False, na_rep="NA")
        events_path = self.analysis_dir / "pr_light_events.csv"
        self.light_events_ledger().to_csv(events_path, index=False, na_rep="NA")
        return {"pr_light_qc": qc_path, "pr_light_events": events_path}

    def light_qc_lines(self) -> list[str]:
        """One plain-language line per flagged or noteworthy Chamber Group,
        for the summary, the Hub's log and the report."""
        settings = self.light_qc_settings()
        lines: list[str] = []
        for _, r in self.light_qc_table().iterrows():
            where = f"DFM {int(r['DFM'])} group {int(r['Group'])}"
            if r["Treatment"]:
                where += f" ({r['Treatment']})"
            reasons: list[str] = []
            flags = [f for f in str(r["Flags"]).split(", ") if f]
            for flag in flags:
                if flag == lqc.SELF_TRIGGERED:
                    reasons.append(
                        f"self-triggered light: {int(r['LongestLickFreeRun'])} "
                        f"consecutive lick-free light events, the first run of "
                        f"{settings.lick_free_run} beginning "
                        f"{r['LickFreeRunStartMin']:.1f} min after training end")
                elif flag == lqc.IMPLAUSIBLE_TRAINING:
                    reasons.append(
                        f"implausible training: {int(r['TrainingLightEvents'])} "
                        f"training light events and no sucrose licks before "
                        f"training ended at {r['TrainingMinutes']:.1f} min")
                elif flag == lqc.NO_TREND:
                    rho = r["TrendRho"]
                    reasons.append(
                        f"no increasing trend over {int(r['TestLightEvents'])} Test "
                        f"light events (" + (f"rho {rho:.2f} < {settings.trend_min_rho:g}"
                                             if pd.notna(rho) else
                                             "licks per event never changed") + ")")
                elif flag == lqc.RESTING_RISE:
                    reasons.append(
                        f"resting level rise: {r['RestingStart']:.0f} → peak "
                        f"{r['RestingMax']:.0f} counts (+{r['RestingRise']:.0f})")
                elif flag == lqc.RESTING_ELEVATED:
                    reasons.append(
                        f"resting level elevated: {r['RestingRatio']:.1f}× the "
                        f"DFM's other sucrose wells")
                elif flag == lqc.SERIES_WRONG:
                    reasons.append(
                        f"{lqc.SERIES_WRONG}: a Test light event met a Requirement the "
                        f"Test Paradigm's series does not give; Program.txt is probably "
                        f"not the program that ran")
            if r["Verdict"] == lqc.VERDICT_FAILED:
                outcome = ("EXCLUDED (exclude_failed_pr_groups)" if r["Excluded"]
                           else "FAILED but retained (exclude_failed_pr_groups is off)")
            elif r["Verdict"] == lqc.VERDICT_WARNING:
                outcome = "warning — kept"
            else:
                outcome = ""
            if reasons:
                lines.append(f"{where}: {outcome}")
                lines.extend(f"    - {reason}" for reason in reasons)
            if r["Notes"]:
                if not reasons:
                    lines.append(f"{where}:")
                lines.extend(f"    · {note}" for note in _split_notes(r["Notes"]))
        return lines

    def _group_chambers_phrase(self, of: str = "each") -> str:
        """What leaving takes from a group, in words: ``both chambers of
        <of>`` for pairs, ``the chamber of <of>`` for solo groups, ``every
        chamber of <of>`` with both."""
        kinds = {g.solo for _d, g in self._group_pairs()}
        if kinds == {True}:
            return f"the chamber of {of}"
        return f"every chamber of {of}" if kinds == {True, False} else f"both chambers of {of}"

    def _light_qc_summary_lines(self) -> list[str]:
        settings = self.light_qc_settings()
        table = self.light_qc_table()
        out = ["", "Progressive ratio light QC",
               "--------------------------",
               "Did the paired fly earn its light?  The firmware lights a group from",
               "its own reading of the paired Sucrose Well; a lick-free light event",
               "has no Sucrose Well licks since the previous light event ended, and",
               "no Sucrose Well lick or touch within the light's decay around it.",
               "Computed over the whole recording, whatever window a table uses.",
               f"Settings: {settings.describe()}", ""]
        if table.empty:
            return out + ["(no DFMs)", ""]
        show = table[["DFM", "Group", "PairedChamber", "Treatment",
                      "TrainingLightEvents", "TrainingLicks", "TestLightEvents",
                      "LickFreeEvents", "LongestLickFreeRun", "TrendRho",
                      "RestingRise", "RestingRatio", "Verdict"]].copy()
        show.columns = ["DFM", "Group", "Paired", "Treatment", "TrainEv",
                        "TrainLicks", "TestEv", "LickFree", "Run", "Rho",
                        "Rise", "Ratio", "Verdict"]
        formats = [("Rho", "{:.2f}"), ("Rise", "{:.0f}"), ("Ratio", "{:.1f}")]
        if table["TrendRhoFirmware"].notna().any():
            ## Generation 3: the trend of the firmware's own Events beside it.
            show.insert(list(show.columns).index("Rho") + 1, "RhoFW", table["TrendRhoFirmware"])
            formats.append(("RhoFW", "{:.2f}"))
        for col, fmt in formats:
            show[col] = show[col].map(lambda v, f=fmt: "—" if pd.isna(v) else f.format(v))
        for col in ("LickFree", "Run"):
            show[col] = show[col].map(lambda v: "—" if pd.isna(v) else f"{int(v)}")
        for col in ("TrainEv", "TrainLicks", "TestEv"):
            if show[col].isna().any():      # a DFM that records no Training
                show[col] = show[col].map(lambda v: "—" if pd.isna(v) else f"{int(v)}")
        out.append(show.to_string(index=False))
        out.append("")
        inc = self.estimated_increment()
        if not self.progressive_groups():
            pass                  # a fixed ratio's requirement does not rise (decision 14)
        elif inc is None:
            out.append("Estimated requirement increment: n/a (no chamber group has "
                       f"{lqc.INCREMENT_EVENTS} Test light events with a rising count).")
        else:
            out.append(f"Estimated requirement increment: {inc[0]:.1f} licks per light "
                       f"event (median over {inc[2]} chamber group(s)' first "
                       f"{lqc.INCREMENT_EVENTS} Test light events).")
        detail = self.light_qc_lines()
        if detail:
            out.append("")
            out.extend(f"  {line}" for line in detail)
        excluded = table[table["Excluded"]]
        failed_kept = table[(table["Verdict"] == lqc.VERDICT_FAILED) & ~table["Excluded"]]
        out.append("")
        if not excluded.empty:
            groups = ", ".join(f"DFM {int(r.DFM)} group {int(r.Group)}"
                               for r in excluded.itertuples())
            out.append(f"Excluded by the light QC ({self._group_chambers_phrase()}): "
                       f"{groups}.")
        if not failed_kept.empty:
            groups = ", ".join(f"DFM {int(r.DFM)} group {int(r.Group)}"
                               for r in failed_kept.itertuples())
            out.append(f"FAILED but retained (exclude_failed_pr_groups is off): {groups}.")
        if excluded.empty and failed_kept.empty:
            out.append("No chamber group failed the light QC.")
        out.append("")
        return out

    # ------------------------------------------------------------------
    # Feeding summary: standard two-well columns + Group/Role/Training/Light
    # ------------------------------------------------------------------

    def _augment_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append ``Group, Role, TrainingMinutes, TrainingComplete, LightOn_sec``,
        the light QC's ``LightQC, LickFreeLightEvents`` and Sucrose
        Persistence's ``PersistA, PersistACensored`` to a per-chamber frame
        carrying ``DFM, Chamber, StartMin, EndMin``.

        The light QC and persistence columns are redone on every call, even on
        a frame that already has the rest: a summary read back from the disk
        cache carries whatever they said when it was written, and their
        thresholds live in the design, which that cache's key does not cover.
        """
        if df is None or df.empty:
            return df
        if "Role" not in df.columns:
            df = self._augment_role_rows(df)
        df = self._with_persistence_columns(self._with_light_qc_columns(df))
        return self.opto.with_columns(df)

    def _with_persistence_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """``PersistA`` and ``PersistACensored`` on every row: the chamber's
        Sucrose Persistence over its group's Test phase
        (:meth:`chamber_persistence`), whatever window the row covers — like
        ``TrainingMinutes``, a fact about the group, not the window.
        :meth:`feeding_summary_facet` blanks it on Training rows; a group that
        never completed training has none."""
        settings = self.break_settings()
        df = df.drop(columns=[c for c in PERSIST_COLUMNS if c in df.columns])
        values: list[float] = []
        flags: list[bool | None] = []
        for dfm_id, chamber in zip(df["DFM"].astype(int), df["Chamber"].astype(int)):
            result = self.chamber_persistence(dfm_id, chamber, settings)
            values.append(np.nan if result is None else float(result[0]))
            flags.append(None if result is None else bool(result[1]))
        df["PersistA"] = values
        df["PersistACensored"] = pd.Series(flags, index=df.index, dtype=object)
        return df

    def _with_light_qc_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """``LightQC`` (the group's flags, comma-joined; empty when clean) and
        ``LickFreeLightEvents`` on every row, from :meth:`light_qc_table`."""
        by_group = self._light_qc_by_group()
        df = df.drop(columns=[c for c in LIGHT_QC_ROW_COLUMNS if c in df.columns])
        keys = list(zip(df["DFM"].astype(int), df["Group"].astype(int)))
        df["LightQC"] = [by_group[k]["Flags"] if k in by_group else "" for k in keys]
        df["LickFreeLightEvents"] = [by_group[k]["LickFreeEvents"] if k in by_group
                                     else np.nan for k in keys]
        return df

    def _augment_role_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        groups, roles, tmins, tdone, light = [], [], [], [], []
        for _, row in df.iterrows():
            dfm_id, chamber = int(row["DFM"]), int(row["Chamber"])
            r = self.role_of(dfm_id, chamber)
            gt = self.group_training(dfm_id, r.group)
            groups.append(r.group)
            roles.append(r.role)
            ## Training end is a property of the Chamber Group, not of the
            ## paired fly alone: it is the moment the group's light stopped
            ## being purely closed-loop, and the yoked fly lived through the
            ## same moment.  So both rows carry it — the yoked cell used to be
            ## NA, which read as missing data in every table and plot that
            ## groups by it.  A group that never finished has no such moment,
            ## and both its rows stay NA.
            tmins.append(float(gt.training_end) if gt.complete else np.nan)
            ## NA, not false, where the recording has no Training at all.
            tdone.append(gt.training_complete)
            rng = (float(row.get("StartMin", 0.0)), float(row.get("EndMin", 0.0)))
            light.append(self._light_on_seconds(self.dfms[dfm_id], chamber, rng))
        pos = list(df.columns).index("Chamber") + 1
        df.insert(pos, "Group", groups)
        df.insert(pos + 1, "Role", roles)
        df["TrainingMinutes"] = tmins
        df["TrainingComplete"] = tdone
        df["LightOn_sec"] = light
        return df

    def feeding_summary(
        self,
        *,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool | None = None,
    ) -> pd.DataFrame:
        if transform_licks is None:
            transform_licks = self.transform_licks
        base = Experiment.feeding_summary(
            self, range_minutes=range_minutes, transform_licks=transform_licks)
        if base is None or base.empty:
            return base
        out = self._augment_rows(base)
        ## The base class's key: range_minutes may be a list of windows.
        key = windowing.range_key(range_minutes), bool(transform_licks)
        self._feeding_summary_cache[key] = out
        return out

    def _chamber_rows(self, dfm_id: int, chambers: Sequence[int],
                      range_minutes: Sequence[float],
                      transform_licks: bool) -> pd.DataFrame:
        """Treatment-labelled summary rows for *chambers* of one DFM over one
        window — the per-group building block of the faceted summary.  On an
        optogenetic experiment they carry ``LitSec`` and ``Activations`` over
        that window, where :meth:`Experiment.feeding_summary` puts them; the
        ``Role`` stays this type's own."""
        from .light_summary import with_light_columns

        dfm = self.dfms[int(dfm_id)]
        wanted = {int(c) for c in chambers
                  if self.design.treatment_for(dfm_id, c) is not None}
        if not wanted:
            return pd.DataFrame()
        summ = dfm.feeding_summary(range_minutes=range_minutes,
                                   transform_licks=transform_licks)
        summ = summ[summ["Chamber"].astype(int).isin(wanted)].copy()
        if summ.empty:
            return summ
        summ.insert(0, "Treatment",
                    [self.design.treatment_for(dfm_id, int(c)) for c in summ["Chamber"]])
        summ = with_light_columns(self, self._append_factor_columns(summ), range_minutes)
        return self._augment_rows(summ)

    # ------------------------------------------------------------------
    # Facets: Training / Test, split at each group's own training end
    # ------------------------------------------------------------------

    def resolved_facet_cutoffs(self):
        return None

    def facet_windows(self):
        """Empty: this type's windows are per Chamber Group, see
        :meth:`group_windows`.  Consumers that need "is this faceted" should
        use :meth:`facet_labels`."""
        return []

    def facet_labels(self):
        """The ordered union of every DFM's Facets: Training and Test, and
        ``Recording`` for a DFM that records no Training (decision 15)."""
        present = {label for dfm in self.dfms.values()
                   for label in ((RECORDING_LABEL,) if dfm.training_info == "none"
                                 else FACET_LABELS)}
        return [label for label in _FACET_ORDER if label in present] or list(FACET_LABELS)

    def group_windows(self, dfm_id: int, group: int) -> list[tuple[str, tuple]]:
        """``[(label, (start, end)), ...]`` for one Chamber Group.  A group
        that never completed training has only a Training window, spanning
        the whole recording; a group on a DFM that records no Training has
        one ``Recording`` window."""
        gt = self.group_training(dfm_id, group)
        if not gt.known:
            return [(RECORDING_LABEL, (0.0, float("inf")))]
        if not gt.complete:
            return [(TRAINING_LABEL, (0.0, float("inf")))]
        ## Rounded *up* to the thousandth of a minute (0.06 s) so the
        ## FacetRange cell and the StartMin/EndMin columns read as a time, not
        ## a float dump, while the split stays between the last Training row
        ## and the first Main row, 0.2 s later, where the Light Events split.
        ## Rounding to the hundredth moved it up to three rows either way.
        last = float(gt.training_end)
        end = max(float(np.ceil(last * 1000.0)) / 1000.0, last)
        return [(TRAINING_LABEL, (0.0, end)), (TEST_LABEL, (end, float("inf")))]

    def feeding_summary_facet(
        self,
        *,
        transform_licks: bool | None = None,
    ) -> pd.DataFrame:
        if transform_licks is None:
            transform_licks = self.transform_licks
        frames: list[pd.DataFrame] = []
        for dfm_id, layout in self._group_pairs():
            for label, window in self.group_windows(dfm_id, layout.number):
                df = self._chamber_rows(
                    dfm_id, layout.chambers,
                    range_minutes=windowing.as_range_minutes(window),
                    transform_licks=bool(transform_licks))
                if df is None or df.empty:
                    continue
                df = df.copy()
                df.insert(0, "FacetRange", windowing.format_range(window))
                df.insert(0, "Facet", label)
                if label != TEST_LABEL:
                    ## Sucrose Persistence is measured from training end:
                    ## a Training row has none.
                    df["PersistA"] = np.nan
                    df["PersistACensored"] = None
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        order = {label: i for i, label in enumerate(_FACET_ORDER)}
        out["_o"] = out["Facet"].map(order)
        out = out.sort_values(["_o", "DFM", "Chamber"], kind="stable").drop(columns="_o")
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Paired-Yoked Difference
    # ------------------------------------------------------------------

    def paired_yoked_diff(
        self,
        *,
        transform_licks: bool | None = None,
    ) -> pd.DataFrame:
        """One row per Chamber Group per Facet: paired minus yoked for each of
        :data:`DIFF_METRICS` (as ``d<metric>``), and ``dPersistCensored`` —
        either fly's Sucrose Persistence censored, so ``dPersistA`` is a
        difference of lower bounds (kept and flagged, never dropped;
        ADR-0014).  A group missing either chamber contributes no row."""
        facet = self.feeding_summary_facet(transform_licks=transform_licks)
        cols = ["Treatment", *(self.design_factors or []), "DFM", "Group", "Facet",
                "FacetRange", "PairedChamber", "YokedChamber", "StartMin", "EndMin",
                "TrainingMinutes", "TrainingComplete", "LightOn_sec",
                *LIGHT_QC_ROW_COLUMNS, *(f"d{m}" for m in DIFF_METRICS),
                "dPersistCensored"]
        if facet is None or facet.empty:
            return pd.DataFrame(columns=cols)
        rows = []
        keys = ["DFM", "Group", "Facet"]
        for (dfm_id, group, label), sub in facet.groupby(keys, sort=False):
            paired = sub[sub["Role"] == ROLE_PAIRED]
            yoked = sub[sub["Role"] == ROLE_YOKED]
            if len(paired) != 1 or len(yoked) != 1:
                continue
            p, y = paired.iloc[0], yoked.iloc[0]
            row = {"Treatment": p["Treatment"]}
            for f in (self.design_factors or []):
                row[f] = p.get(f, "")
            row.update({
                "DFM": int(dfm_id), "Group": int(group), "Facet": label,
                "FacetRange": p["FacetRange"],
                "PairedChamber": int(p["Chamber"]), "YokedChamber": int(y["Chamber"]),
                "StartMin": float(p["StartMin"]), "EndMin": float(p["EndMin"]),
                "TrainingMinutes": p["TrainingMinutes"],
                "TrainingComplete": _flag(p["TrainingComplete"]),
                "LightOn_sec": float(p["LightOn_sec"]),
                "LightQC": p.get("LightQC", ""),
                "LickFreeLightEvents": p.get("LickFreeLightEvents", np.nan),
            })
            for m in DIFF_METRICS:
                pv, yv = p.get(m, np.nan), y.get(m, np.nan)
                row[f"d{m}"] = (float(pv) - float(yv)
                                if pd.notna(pv) and pd.notna(yv) else np.nan)
            pc = _flag(p.get("PersistACensored"))
            yc = _flag(y.get("PersistACensored"))
            row["dPersistCensored"] = None if pc is None or yc is None else (pc or yc)
            rows.append(row)
        return pd.DataFrame(rows, columns=cols)

    def write_paired_yoked_diff(
        self,
        path: str | Path | None = None,
        *,
        transform_licks: bool | None = None,
    ) -> Path:
        if path is None:
            if self.analysis_dir is None:
                raise ValueError("path must be provided when no experiment_dir is set.")
            path = self.analysis_dir / "paired_yoked_diff.csv"
        out = Path(path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        self.paired_yoked_diff(transform_licks=transform_licks).to_csv(
            out, index=False, na_rep="NA")
        return out

    # ------------------------------------------------------------------
    # The standard per-treatment plots, made role-aware
    # ------------------------------------------------------------------
    #
    # A Treatment in this type names *both* flies of a Chamber Group, so the
    # inherited plots — which group by Treatment alone — draw one cloud of
    # points per treatment holding paired and yoked flies together.  That
    # cloud has no referent: the yoked fly's PI is not a second measurement
    # of the paired fly's preference, it is the control the paired fly is
    # measured *against*.  Pooling them averages an effect with its own
    # control and lands halfway to nothing.
    #
    # Two fixes, one per plot shape.  The time courses and the multi-metric
    # feeding summary split their groups by Role, so a treatment becomes two
    # series — the honest version of the same picture.  The dot plot, where
    # one point is one observation, goes further and plots the within-group
    # difference itself: paired minus yoked, one point per Chamber Group,
    # which is the unit CONTEXT.md fixes for this type ("a difference is
    # always taken *within* a Chamber Group, never between group means").

    def _append_role_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ``Group`` and ``Role`` to any frame carrying ``DFM`` and
        ``Chamber``.

        The cheap half of :meth:`_augment_rows`: roles are a config fact, so
        this costs a dict lookup per row, while the training end and the
        light-on seconds mean reading the signal.  Binned and moving-window
        tables have a row per chamber *per bin* and need only the role.
        """
        if df is None or df.empty or "Role" in df.columns:
            return df
        if "DFM" not in df.columns or "Chamber" not in df.columns:
            return df
        df = df.copy()
        keys = list(zip(df["DFM"].astype(int), df["Chamber"].astype(int)))
        roles = {k: self.role_of(*k) for k in set(keys)}
        pos = list(df.columns).index("Chamber") + 1
        df.insert(pos, "Group", [roles[k].group for k in keys])
        df.insert(pos + 1, "Role", [roles[k].role for k in keys])
        return df

    def _assemble_treatment_table(
        self, summary_by_dfm: dict[int, pd.DataFrame]
    ) -> pd.DataFrame:
        """The inherited treatment table, plus each row's Chamber Group and
        Role — so every plot built on it can keep the two flies apart."""
        return self._append_role_columns(
            Experiment._assemble_treatment_table(self, summary_by_dfm))

    def _resolve_group_col(self, df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        """The inherited grouping, with the Role appended to every label.

        ``w1118`` becomes ``w1118 · paired`` and ``w1118 · yoked``: one
        series per fly role, never one series holding both.  A frame with no
        ``Role`` column (anything not per-chamber) is grouped as before.
        """
        df, group_col = Experiment._resolve_group_col(self, df)
        if "Role" not in df.columns:
            return df, group_col
        df = df.copy()
        df["_RoleGroup"] = (df[group_col].astype(str) + " · "
                            + df["Role"].astype(str))
        return df, "_RoleGroup"

    def paired_yoked_delta(
        self,
        *,
        metric: str = "PI",
        two_well_mode: Literal["total", "mean_ab", "A", "B"] = "total",
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool | None = None,
    ) -> pd.DataFrame:
        """One row per Chamber Group: ``Delta`` = paired *metric* − yoked.

        The general form of :meth:`paired_yoked_diff`, which carries only the
        fixed :data:`DIFF_METRICS` as stored columns.  Here any metric the
        summary can express — including the A/B combinations behind
        *two_well_mode* — is resolved per chamber first and differenced
        after, so the metric catalogue the UI offers works unchanged.

        With no *range_minutes*, rows come from the per-group Training / Test
        Facets and a ``Facet`` column says which.  Given an explicit window,
        that window is used for every group and ``Facet`` reads ``"Custom"`` —
        the Facets are per Chamber Group, so one shared window is a different
        question, not a filter on the same one.

        Groups missing either chamber (excluded, or never assigned) contribute
        no row, exactly as in :meth:`paired_yoked_diff`.
        """
        factors = list(self.design_factors or [])
        cols = ["Treatment", *factors, "DFM", "Group", "Facet",
                "PairedChamber", "YokedChamber", "Paired", "Yoked", "Delta"]
        if range_is_specified(range_minutes):
            per_chamber = self.feeding_summary(
                range_minutes=range_minutes, transform_licks=transform_licks)
            if per_chamber is not None and not per_chamber.empty:
                per_chamber = per_chamber.copy()
                per_chamber["Facet"] = "Custom"
        else:
            per_chamber = self.feeding_summary_facet(
                transform_licks=transform_licks)
        if per_chamber is None or per_chamber.empty:
            return pd.DataFrame(columns=cols)

        per_chamber = per_chamber.copy()
        transformed = bool(self.transform_licks if transform_licks is None else transform_licks)
        per_chamber["_Value"] = self._metric_series_from_binned_rows(
            per_chamber, metric=metric, two_well_mode=two_well_mode,
            transform_licks=transformed)

        rows: list[dict] = []
        for (dfm_id, group, label), sub in per_chamber.groupby(
                ["DFM", "Group", "Facet"], sort=False):
            paired = sub[sub["Role"] == ROLE_PAIRED]
            yoked = sub[sub["Role"] == ROLE_YOKED]
            if len(paired) != 1 or len(yoked) != 1:
                continue
            p, y = paired.iloc[0], yoked.iloc[0]
            pv, yv = p["_Value"], y["_Value"]
            row = {"Treatment": p["Treatment"]}
            for f in factors:
                row[f] = p.get(f, "")
            row.update({
                "DFM": int(dfm_id), "Group": int(group), "Facet": label,
                "PairedChamber": int(p["Chamber"]),
                "YokedChamber": int(y["Chamber"]),
                "Paired": float(pv) if pd.notna(pv) else np.nan,
                "Yoked": float(yv) if pd.notna(yv) else np.nan,
                "Delta": (float(pv) - float(yv)
                          if pd.notna(pv) and pd.notna(yv) else np.nan),
            })
            rows.append(row)
        out = pd.DataFrame(rows, columns=cols)
        return out.dropna(subset=["Delta"]).reset_index(drop=True)

    def plot_dot_metric_by_treatment(
        self,
        *,
        metric: str = "Licks",
        two_well_mode: Literal["total", "mean_ab", "A", "B"] = "total",
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool | None = None,
    ) -> Any:
        """Jitter + mean ± SE of the **within-group difference** in *metric*.

        One point is one Chamber Group's paired-minus-yoked value, not one
        fly: the comparison this design licenses.  Zero is drawn, because
        zero — not the axis floor — is the null this plot is read against.

        The x axis is the first design factor (or the Treatment when there
        are none) and the panels are the Facets, with any further factors
        folded into the panel label.
        """
        from plotnine import annotate, ggplot, theme_bw

        df = self.paired_yoked_delta(
            metric=metric, two_well_mode=two_well_mode,
            range_minutes=range_minutes, transform_licks=transform_licks)
        if df.empty:
            return (ggplot()
                    + annotate("text", x=0, y=0,
                               label="No complete chamber group to difference")
                    + theme_bw())

        factor_cols = [f for f in (self.design_factors or []) if f in df.columns]
        x_col = factor_cols[0] if factor_cols else "Treatment"
        ## Facet first, so the panels read Training | Test at a glance; any
        ## factor past the x axis joins the panel label rather than the x.
        panel_cols = ["Facet", *factor_cols[1:]]
        df = df.copy()
        panel = df[panel_cols].astype(str).agg(" / ".join, axis=1)
        ## Ordered by first appearance, which is the order the type defines
        ## its Facets in \u2014 Training then Test, not the alphabet's Test first.
        df["_Panel"] = pd.Categorical(
            panel, categories=list(dict.fromkeys(panel)), ordered=True)

        label = f"\u0394{metric} (paired \u2212 yoked)"
        transformed = bool(self.transform_licks if transform_licks is None else transform_licks)
        if transformed and str(metric).startswith("Licks"):
            from . import report_content as rc

            label = rc.on_root_scale(label)
        return self.plot_jitter_summary(
            df,
            x_col=x_col,
            y_col="Delta",
            facet_col="_Panel",
            title="One point per chamber group",
            y_label=label,
            hline_at=0.0,
        )

    # ------------------------------------------------------------------
    # Auto-removal: two-well thresholds + require_training_complete + light QC
    # ------------------------------------------------------------------

    def auto_remove_chambers(
        self,
        *,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool = True,
        min_untransformed_licks_cutoff: float | None = None,
    ) -> pd.DataFrame:
        ## The light QC reads the design as loaded, so it runs before anything
        ## below thins it; its verdicts are cached for the tables written later.
        light_settings = self.light_qc_settings()
        light_failed = self.light_qc_failed_groups()
        base = Experiment.auto_remove_chambers(
            self, range_minutes=range_minutes,
            min_untransformed_licks_cutoff=min_untransformed_licks_cutoff)
        constants = self.global_constants or {}
        require_training = _truthy(constants.get("require_training_complete", True))

        fs = self.feeding_summary(range_minutes=range_minutes,
                                  transform_licks=transform_licks)
        already: set[tuple[int, int]] = set()
        if base is not None and not base.empty:
            already = {(int(r["DFM"]), int(r["Chamber"])) for _, r in base.iterrows()}

        removed_rows: list[dict] = []
        to_remove: set[tuple[int, int]] = set()
        for _, row in fs.iterrows():
            key = (int(row["DFM"]), int(row["Chamber"]))
            if key in already or key in to_remove:
                continue
            ## The cutoffs every two-well type applies (ADR-0019).
            reasons: list[str] = self.two_well_removal_reasons(row)
            ## Only a group known not to have trained: on a DFM that records no
            ## Training, TrainingComplete is NA and removes nothing.
            if require_training and _flag(row.get("TrainingComplete", True)) is False:
                reasons.append(
                    f"training never completed in chamber group {int(row['Group'])} "
                    f"(require_training_complete)")
            failing = light_failed.get((key[0], int(row["Group"])))
            if failing:
                reasons.append(
                    f"light QC failed in chamber group {int(row['Group'])}: "
                    f"{', '.join(failing)} (exclude_failed_pr_groups)")
            if reasons:
                to_remove.add(key)
                removed_rows.append({"DFM": key[0], "Chamber": key[1],
                                     "Treatment": str(row.get("Treatment", "")),
                                     "Reason": "; ".join(reasons)})
                print(f"  [auto_remove_chambers] Removing DFM {key[0]} Chamber {key[1]}"
                      f" ({row.get('Treatment', '')}): {'; '.join(reasons)}", flush=True)
        if to_remove:
            self._remove_chambers_from_design(to_remove)

        extra = pd.DataFrame(removed_rows, columns=["DFM", "Chamber", "Treatment", "Reason"])
        if base is None or base.empty:
            combined = extra
        elif extra.empty:
            combined = base
        else:
            combined = pd.concat([base, extra], ignore_index=True)
        self.filtered_chambers = combined

        lines = ["", "Progressive Ratio filter (additional criteria)", "",
                 *self.two_well_criteria_lines()]
        whose = self._group_chambers_phrase("a chamber group")
        verb = "is" if whose.startswith("the chamber") else "are"
        lines.append(f"  • require_training_complete = true: {whose} whose paired fly "
                     f"never finished training {verb} excluded"
                     if require_training else
                     "  • require_training_complete = false: incomplete-training "
                     "groups are kept (TrainingComplete = false).")
        lines.append(f"  • exclude_failed_pr_groups = true: {whose} whose light QC fails "
                     f"(self-triggered light, implausible training, a requirement series "
                     f"the program does not give) {verb} excluded — see pr_light_qc.csv"
                     if light_settings.exclude else
                     "  • exclude_failed_pr_groups = false: groups that fail the light "
                     "QC are kept; pr_light_qc.csv still lists them.")
        self.filter_criteria_summary = self.filter_criteria_summary + "\n" + "\n".join(lines)
        self.write_removed_chambers()
        return combined

    # ------------------------------------------------------------------
    # Cumulative curves since training end
    # ------------------------------------------------------------------

    def cumulative_curve_data(
        self,
        *,
        binsize_min: float = 1.0,
        well: str = "A",
        qc: bool = False,
    ) -> pd.DataFrame:
        """Per-chamber cumulative raw licks on *well* since the group's
        training end, at *binsize_min* resolution.

        Columns: ``Treatment, [factors], DFM, Chamber, Group, Role, Minutes,
        CumLicks, LightOn`` where ``Minutes`` is the bin's right edge in
        minutes since training end and ``LightOn`` says whether the group's
        light was on at any sample in the bin.  Groups that never completed
        training contribute nothing.

        With *qc*, chambers auto-removal took out are kept (see
        :meth:`_design_snapshot`) — the QC traces exist to show why a group
        left, the result curves must not include it.
        """
        if binsize_min <= 0:
            raise ValueError("binsize_min must be positive.")
        frames: list[pd.DataFrame] = []
        for dfm_id in sorted(self.dfms):
            dfm = self.dfms[dfm_id]
            mins_all = dfm.lick_df["Minutes"].to_numpy(dtype=float)
            for layout in self.chamber_groups(dfm_id):
                group, chambers = layout.number, layout.chambers
                gt = self.group_training(dfm_id, group)
                if not gt.complete:
                    continue
                end = float(gt.training_end)
                mask = mins_all > end
                if not mask.any():
                    continue
                rel = mins_all[mask] - end
                bins = np.floor(rel / float(binsize_min)).astype(int)
                edges = (bins + 1) * float(binsize_min)
                for chamber in chambers:
                    treatment = (self._design_snapshot().get((dfm_id, chamber)) if qc
                                 else self.design.treatment_for(dfm_id, chamber))
                    if treatment is None:
                        continue
                    ch = dfm.chambers[chamber - 1]
                    w = int(ch.well_a if well.upper() == "A" else ch.well_b)
                    licks = dfm.lick_df[f"W{w}"].to_numpy(dtype=float)[mask]
                    light = self._chamber_light(dfm, chamber).to_numpy(dtype=bool)[mask]
                    tmp = pd.DataFrame({"bin": bins, "Minutes": edges,
                                        "licks": licks, "light": light})
                    agg = tmp.groupby("bin", sort=True).agg(
                        Minutes=("Minutes", "first"), licks=("licks", "sum"),
                        LightOn=("light", "any")).reset_index(drop=True)
                    agg["CumLicks"] = agg["licks"].cumsum()
                    r = self.role_of(dfm_id, chamber)
                    agg.insert(0, "Role", r.role)
                    agg.insert(0, "Group", group)
                    agg.insert(0, "Chamber", chamber)
                    agg.insert(0, "DFM", dfm_id)
                    agg.insert(0, "Treatment", treatment)
                    frames.append(agg.drop(columns="licks"))
        cols = ["Treatment", "DFM", "Chamber", "Group", "Role", "Minutes",
                "CumLicks", "LightOn"]
        if not frames:
            return pd.DataFrame(columns=cols)
        out = pd.concat(frames, ignore_index=True)[cols]
        return self._append_factor_columns(out)

    def cumulative_diff_data(self, *, binsize_min: float = 1.0) -> pd.DataFrame:
        """Per-group paired-minus-yoked cumulative sucrose-well licks since
        training end: ``Treatment, [factors], DFM, Group, Minutes,
        DiffCumLicks``.  The data behind the Cumulative Difference Curve and
        the ``timecourse_pr_diff`` Plot Spec."""
        curves = self.cumulative_curve_data(binsize_min=binsize_min, well="A")
        cols = ["Treatment", *(self.design_factors or []), "DFM", "Group",
                "Minutes", "DiffCumLicks"]
        if curves.empty:
            return pd.DataFrame(columns=cols)
        frames = []
        for (dfm_id, group), sub in curves.groupby(["DFM", "Group"], sort=True):
            p = sub[sub["Role"] == ROLE_PAIRED].set_index("Minutes")["CumLicks"]
            y = sub[sub["Role"] == ROLE_YOKED].set_index("Minutes")["CumLicks"]
            if p.empty or y.empty:
                continue
            idx = p.index.intersection(y.index)
            diff = pd.DataFrame({"Minutes": idx,
                                 "DiffCumLicks": p.loc[idx].to_numpy() - y.loc[idx].to_numpy()})
            first = sub.iloc[0]
            diff.insert(0, "Group", int(group))
            diff.insert(0, "DFM", int(dfm_id))
            for f in reversed(self.design_factors or []):
                diff.insert(0, f, first.get(f, ""))
            diff.insert(0, "Treatment", first["Treatment"])
            frames.append(diff)
        if not frames:
            return pd.DataFrame(columns=cols)
        return pd.concat(frames, ignore_index=True)[cols]

    def cumulative_diff_stat(self, *, binsize_min: float = 1.0) -> pd.DataFrame:
        """Mean ± SEM of the paired-minus-yoked curve per Treatment and bin —
        the line the Cumulative Difference Curve draws.  Runs only as far as
        the shortest Chamber Group, so every group is in every averaged point
        and the curve never jumps when one group's recording ends."""
        data = self.cumulative_diff_data(binsize_min=binsize_min)
        cols = ["Treatment", "Minutes", "mean", "sem", "count", "ymin", "ymax"]
        if data.empty:
            return pd.DataFrame(columns=cols)
        common_end = float(data.groupby(["DFM", "Group"])["Minutes"].max().min())
        stat = (data[data["Minutes"] <= common_end]
                .groupby(["Treatment", "Minutes"], sort=True)["DiffCumLicks"]
                .agg(["mean", "sem", "count"]).reset_index())
        stat["sem"] = stat["sem"].fillna(0.0)
        stat["ymin"] = stat["mean"] - stat["sem"]
        stat["ymax"] = stat["mean"] + stat["sem"]
        return stat[cols]

    def write_cumulative_diff(self, path: str | Path | None = None, *,
                              binsize_min: float = 1.0) -> Path:
        """Write ``analysis/pr_cumulative_diff.csv`` — the binned difference
        curve a Project stacks for the pooled ``timecourse_pr_diff`` figure."""
        if path is None:
            if self.analysis_dir is None:
                raise ValueError("path must be provided when no experiment_dir is set.")
            path = self.analysis_dir / "pr_cumulative_diff.csv"
        out = Path(path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        self.cumulative_diff_data(binsize_min=binsize_min).to_csv(
            out, index=False, na_rep="NA")
        return out

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    def plot_cumulative_diff(self, *, binsize_min: float = 1.0,
                             base_font_size: float = 10.0,
                             figsize: tuple[float, float] = (8.0, 5.0)):
        """The Cumulative Difference Curve: mean ± SEM per Treatment of the
        paired-minus-yoked cumulative sucrose-well licks against minutes since
        training end, the individual Chamber Group traces faint behind.  The
        mean is drawn only over the range every group covers."""
        import plotnine as p9

        from .experiment import _OKABE_ITO

        data = self.cumulative_diff_data(binsize_min=binsize_min)
        if data.empty:
            why = ("no chamber group completed training" if self.has_yoked_controls()
                   else NO_YOKED_CONTROLS)
            return p9.ggplot() + p9.labs(title=f"Paired − yoked cumulative licks — no data "
                                               f"({why})")
        data = data.copy()
        data["Trace"] = data["DFM"].astype(str) + ":" + data["Group"].astype(str)
        stat = self.cumulative_diff_stat(binsize_min=binsize_min)
        treatments = list(dict.fromkeys(data["Treatment"]))
        palette = {t: _OKABE_ITO[i % len(_OKABE_ITO)] for i, t in enumerate(treatments)}
        well_a = (self.well_names or {}).get("A", "well A")
        g = (p9.ggplot()
             + p9.geom_line(data, p9.aes("Minutes", "DiffCumLicks", group="Trace",
                                         color="Treatment"), alpha=0.25, size=0.4)
             + p9.geom_hline(yintercept=0.0, linetype="dashed", color="#888888", size=0.3)
             + p9.geom_ribbon(stat, p9.aes("Minutes", ymin="ymin", ymax="ymax",
                                           fill="Treatment"), alpha=0.2, color="none")
             + p9.geom_line(stat, p9.aes("Minutes", "mean", color="Treatment"), size=1.0)
             + p9.scale_color_manual(values=palette)
             + p9.scale_fill_manual(values=palette)
             + p9.labs(title="Paired − yoked cumulative licks since training end",
                       x="Time since training end (min)",
                       y=f"Paired − yoked cumulative licks ({well_a})")
             + p9.theme_bw(base_size=base_font_size)
             + p9.theme(figure_size=figsize, legend_position="right"))
        return g

    def _qc_panel_label(self, dfm_id: int, group: int, head: str) -> str:
        """``Group g (Treatment) — <head>`` over the group's light QC verdict,
        the strip of every per-DFM QC figure."""
        treatment = self._group_treatment(dfm_id, group)
        first = f"Group {group}" + (f" ({treatment})" if treatment else "") + f" — {head}"
        row = self._light_qc_row(dfm_id, group, self.light_qc_settings())
        if row["Verdict"] == lqc.VERDICT_FAILED:
            failing = [f for f in row["Flags"].split(", ") if f in lqc.FAILING_FLAGS]
            second = ("EXCLUDED: " if row["Excluded"] else "FAILED: ") + ", ".join(failing)
        elif row["Verdict"] == lqc.VERDICT_WARNING:
            second = "warning: " + row["Flags"]
        else:
            second = "light QC ok"
        return f"{first}\n{second}"

    def plot_cumulative_licks_dfm(self, dfm_id: int, *, binsize_min: float = 1.0,
                                  base_font_size: float = 10.0,
                                  figsize: tuple[float, float] | None = None):
        """QC figure for one DFM: one panel per Chamber Group, paired and
        yoked cumulative sucrose-well licks since the group's training end,
        light-on bins drawn as points and Lick-free Light Events as rings on
        the paired trace.  A group auto-removal took out stays in, its strip
        saying why."""
        import plotnine as p9

        dfm_id = int(dfm_id)
        curves = self.cumulative_curve_data(binsize_min=binsize_min, well="A", qc=True)
        curves = curves[curves["DFM"] == dfm_id] if not curves.empty else curves
        if curves.empty:
            why = (NO_TRAINING_INFORMATION if self.dfms[dfm_id].training_info == "none"
                   else "no chamber group completed training")
            return p9.ggplot() + p9.labs(title=f"DFM {dfm_id} — {why}")
        curves = curves.copy()
        panel = {}
        numbers = [g.number for g in self.chamber_groups(dfm_id)]
        for group in numbers:
            gt = self.group_training(dfm_id, group)
            end = "n/a" if not gt.complete else f"{gt.training_end:.1f} min"
            panel[group] = self._qc_panel_label(dfm_id, group, f"trained {end}")
        present = set(curves["Group"])
        order = [panel[g] for g in numbers if g in present]
        curves["Panel"] = pd.Categorical(curves["Group"].map(panel), categories=order,
                                         ordered=True)
        curves["Role"] = pd.Categorical(curves["Role"], categories=[ROLE_PAIRED, ROLE_YOKED])
        lit = curves[curves["LightOn"]]
        ## Lick-free Light Events, on the paired trace at the moment they fired.
        rings = []
        for group in numbers:
            if group not in present:
                continue
            events = self.light_events_table(dfm_id, group)
            free = events[(events["Facet"] == TEST_LABEL) & events["LickFree"]]
            if not free.empty:
                rings.append(pd.DataFrame({"Minutes": free["Minutes"].to_numpy(),
                                           "CumLicks": free["CumLicks"].to_numpy(),
                                           "Panel": panel[group]}))
        colors = {ROLE_PAIRED: "#D55E00", ROLE_YOKED: "#0072B2"}
        well_a = (self.well_names or {}).get("A", "well A")
        n = curves["Panel"].nunique()
        if figsize is None:
            figsize = _panel_figsize(n)
        g = (p9.ggplot(curves, p9.aes("Minutes", "CumLicks", color="Role"))
             + p9.geom_line(size=0.7)
             + p9.geom_point(data=lit, size=1.4, alpha=0.9))
        if rings:
            ring_data = pd.concat(rings, ignore_index=True)
            ring_data["Panel"] = pd.Categorical(ring_data["Panel"], categories=order,
                                                ordered=True)
            g += p9.geom_point(ring_data, p9.aes("Minutes", "CumLicks"),
                               inherit_aes=False, shape="o", fill="none",
                               color="#000000", size=2.6, stroke=0.6)
        g = (g + p9.facet_wrap("~ Panel", ncol=3, scales="free_x")
              + p9.scale_color_manual(values=colors)
              + p9.labs(title=f"DFM {dfm_id} — cumulative {well_a} licks since training end "
                              f"(points: light on; rings: lick-free light events)",
                        x="Time since training end (min)", y=f"Cumulative licks ({well_a})")
              + p9.theme_bw(base_size=base_font_size)
              + p9.theme(figure_size=figsize, legend_position="bottom"))
        return g

    def plot_light_events_dfm(self, dfm_id: int, *, base_font_size: float = 10.0,
                              figsize: tuple[float, float] | None = None):
        """Licks per light event (QC) for one DFM, one panel per Chamber Group.

        x is the Test-phase Light Event number, y the Sucrose Well licks
        credited to it (``LicksSincePrev``).  A working progressive ratio
        climbs; a Lick-free Light Event is a hollow red ring; the dashed line
        is the group's own linear trend and the faint grey one the
        requirement estimated across the experiment
        (:meth:`estimated_increment`), when there is one.  The grey line and
        its caption are drawn only for groups whose Test Paradigm is a
        progressive ratio: a fixed ratio's requirement does not climb
        (decision 14).
        """
        import plotnine as p9

        dfm_id = int(dfm_id)
        frames, fits = [], []
        panel = {}
        numbers = [g.number for g in self.chamber_groups(dfm_id)]
        progressive = {g for _d, g in self.progressive_groups(dfm_id)}
        for group in numbers:
            events = self.light_events_table(dfm_id, group)
            test = events[events["Facet"] == TEST_LABEL]
            if test.empty:
                continue
            panel[group] = self._qc_panel_label(
                dfm_id, group, f"{len(test)} Test light events")
            x = np.arange(1, len(test) + 1, dtype=float)
            y = test["LicksSincePrev"].to_numpy(dtype=float)
            frames.append(pd.DataFrame({"Event": x, "Licks": y,
                                        "LickFree": test["LickFree"].to_numpy(dtype=bool),
                                        "Group": group}))
            if len(test) >= 2:
                slope, intercept = np.polyfit(x, y, 1)
                fits.append(pd.DataFrame({"Event": [x[0], x[-1]],
                                          "Licks": [intercept + slope * x[0],
                                                    intercept + slope * x[-1]],
                                          "Group": group}))
        if not frames:
            return p9.ggplot() + p9.labs(
                title=f"DFM {dfm_id} — no Test light events in any chamber group")
        data = pd.concat(frames, ignore_index=True)
        order = [panel[g] for g in numbers if g in panel]

        def with_panel(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df["Panel"] = pd.Categorical(df["Group"].map(panel), categories=order,
                                         ordered=True)
            return df

        data = with_panel(data)
        earned, free = data[~data["LickFree"]], data[data["LickFree"]]
        well_a = (self.well_names or {}).get("A", "well A")
        if figsize is None:
            figsize = _panel_figsize(len(order))
        g = p9.ggplot(data, p9.aes("Event", "Licks"))
        inc = self.estimated_increment()
        climbing = data[data["Group"].isin(progressive)]
        if inc is not None and not climbing.empty:
            slope, intercept, _n = inc
            ref = []
            for group, sub in climbing.groupby("Group", sort=True):
                last = float(sub["Event"].max())
                ## Stop the line just above the panel's own data: drawn to the
                ## last of 400 self-triggered events it would set the y axis
                ## and flatten everything the panel is there to show.
                top = max(float(sub["Licks"].max()), 1.0) * 1.1
                last = min(last, max(1.0, (top - intercept) / slope))
                ref.append(pd.DataFrame({"Event": [1.0, last],
                                         "Licks": [intercept + slope, intercept + slope * last],
                                         "Group": group}))
            g += p9.geom_line(with_panel(pd.concat(ref, ignore_index=True)),
                              p9.aes("Event", "Licks"), color="#9E9E9E", alpha=0.7, size=0.8)
        if fits:
            g += p9.geom_line(with_panel(pd.concat(fits, ignore_index=True)),
                              p9.aes("Event", "Licks"), color="#0072B2",
                              linetype="dashed", size=0.6)
        if not earned.empty:
            g += p9.geom_point(earned, color="#333333", size=1.3, alpha=0.85)
        if not free.empty:
            g += p9.geom_point(free, shape="o", fill="none", color="#D62728",
                               size=2.2, stroke=0.6)
        caption = ("Filled: light events with licks; hollow red: lick-free. Dashed: "
                   "the group's trend. ")
        if not climbing.empty:
            caption += ("Grey: estimated requirement "
                        f"({inc[0]:.1f} licks per event)." if inc is not None
                        else "No requirement could be estimated.")
        g = (g + p9.facet_wrap("~ Panel", ncol=3, scales="free")
              + p9.labs(title=f"DFM {dfm_id} — {well_a} licks per light event, Test phase",
                        caption=caption.rstrip(), x="Light event (Test phase)",
                        y=f"{well_a} licks since the previous light event")
              + p9.theme_bw(base_size=base_font_size)
              + p9.theme(figure_size=figsize))
        return g

    def plot_resting_level_dfm(self, dfm_id: int, *, base_font_size: float = 10.0,
                               figsize: tuple[float, float] | None = None):
        """Sucrose Well resting level (QC) for one DFM, one panel per Chamber
        Group.

        The paired chamber's Sucrose Well — the one the firmware watches — as
        its per-minute median raw signal over the whole recording, against the
        median of the DFM's other Sucrose Wells.  Light onsets are the rug
        along the bottom and the dashed line is training end.  A well that
        creeps up while its light fires ever more often is a sensor, not a fly.
        """
        import plotnine as p9

        dfm_id = int(dfm_id)
        dfm = self.dfms[dfm_id]
        levels = self.resting_levels(dfm_id)
        series, rugs, ends = [], [], []
        panel = {}
        own, others = "Paired Sucrose Well", "Other Sucrose Wells (median)"
        numbers = [g.number for g in self.chamber_groups(dfm_id)]
        for group in numbers:
            gt = self.group_training(dfm_id, group)
            column = f"W{gt.sucrose_well}"
            if column not in levels.columns:
                continue
            panel[group] = self._qc_panel_label(
                dfm_id, group, f"paired ch{gt.paired_chamber}, W{gt.sucrose_well}")
            s = levels[column]
            series.append(pd.DataFrame({"Minute": s.index.to_numpy(dtype=float),
                                        "Level": s.to_numpy(dtype=float),
                                        "Series": own, "Group": group}))
            ref = self.resting_reference(dfm_id, gt.sucrose_well)
            if not ref.empty:
                series.append(pd.DataFrame({"Minute": ref.index.to_numpy(dtype=float),
                                            "Level": ref.to_numpy(dtype=float),
                                            "Series": others, "Group": group}))
            events = self.light_events_table(dfm_id, group)
            if not events.empty:
                rugs.append(pd.DataFrame({"Minute": events["RecordingMinute"].to_numpy(),
                                          "Group": group}))
            if gt.complete:
                ends.append(pd.DataFrame({"Minute": [float(gt.training_end)],
                                          "Group": group}))
        if not series:
            return p9.ggplot() + p9.labs(title=f"DFM {dfm_id} — no Sucrose Well signal")
        order = [panel[g] for g in numbers if g in panel]

        def with_panel(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df["Panel"] = pd.Categorical(df["Group"].map(panel), categories=order,
                                         ordered=True)
            return df

        data = with_panel(pd.concat(series, ignore_index=True))
        data["Series"] = pd.Categorical(data["Series"], categories=[own, others])
        if figsize is None:
            figsize = _panel_figsize(len(order))
        g = (p9.ggplot(data, p9.aes("Minute", "Level", color="Series"))
             + p9.geom_line(size=0.6))
        if rugs:
            g += p9.geom_rug(with_panel(pd.concat(rugs, ignore_index=True)),
                             p9.aes(x="Minute"), inherit_aes=False, sides="b",
                             color="#0072B2", alpha=0.35, length=0.05)
        if ends:
            g += p9.geom_vline(with_panel(pd.concat(ends, ignore_index=True)),
                               p9.aes(xintercept="Minute"), linetype="dashed",
                               color="#555555", size=0.5)
        g = (g + p9.facet_wrap("~ Panel", ncol=3, scales="free_y")
              + p9.scale_color_manual(values={own: "#D55E00", others: "#9E9E9E"})
              + p9.labs(title=f"DFM {dfm_id} — Sucrose Well resting level "
                              f"(per-minute median raw signal)",
                        caption="Rug: light onsets. Dashed: training end.",
                        x="Minutes", y="Raw signal (counts)", color="")
              + p9.theme_bw(base_size=base_font_size)
              + p9.theme(figure_size=figsize, legend_position="bottom"))
        return g

    def write_pr_figures(self, *, binsize_min: float = 1.0, dpi: int = 200) -> dict[str, Path]:
        """Write the headline curve, the still-responding curve and the per-DFM
        QC figures into ``analysis/``: training-aligned traces, licks per light
        event and the Sucrose Well resting level."""
        if self.analysis_dir is None:
            raise ValueError("experiment_dir must be set to write figures.")
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        path = self.analysis_dir / "pr_cumulative_diff.png"
        self.plot_cumulative_diff(binsize_min=binsize_min).save(str(path), dpi=dpi, verbose=False)
        written["pr_cumulative_diff"] = path
        path = self.analysis_dir / "pr_still_responding.png"
        self.plot_still_responding().save(str(path), dpi=dpi, verbose=False)
        written["pr_still_responding"] = path
        for dfm_id in sorted(self.dfms):
            for stem, build in (
                ("pr_cumulative_licks",
                 lambda d: self.plot_cumulative_licks_dfm(d, binsize_min=binsize_min)),
                ("pr_light_events", self.plot_light_events_dfm),
                ("pr_resting_level", self.plot_resting_level_dfm),
            ):
                path = self.analysis_dir / f"{stem}_dfm{dfm_id}.png"
                build(dfm_id).save(str(path), dpi=dpi, verbose=False)
                written[f"{stem}_dfm{dfm_id}"] = path
        return written

    # ------------------------------------------------------------------
    # Breaking point and Sucrose Persistence: the first-gap rule (ADR-0014)
    # ------------------------------------------------------------------
    #
    # When did the fly stop?  Responses are read in order from the group's
    # training end, and the first pause longer than pr_break_gap_min ends the
    # count (pyflic.base.pr_breaking_point).  The Breaking Point asks it of
    # the Paired fly's lick-backed Test Light Events; Sucrose Persistence asks
    # it of either fly's Sucrose Well feeding events.

    def break_settings(self) -> pbp.BreakSettings:
        """The first-gap rule's settings, from the design's ``constants:``
        (``pr_break_gap_min``, ``pr_test_window_min``)."""
        return pbp.BreakSettings.from_constants(self.global_constants)

    def _test_window(self, dfm_id: int, group: int,
                     settings: pbp.BreakSettings) -> float | None:
        """The group's Test window in minutes since training end — to the end
        of the recording, capped at ``pr_test_window_min`` — or ``None`` when
        training never ended.

        On generation 3 the window ends with the last row of the Paired
        Sucrose Well's ratio Paradigm run that holds its training end, not at
        the end of the recording (decision 12): a ClosedLoop Interval after it
        is no part of the Test, and a later ratio run is noted
        (:meth:`group_training`) and left to Program Facets.  A DFM restart
        inside that run ends it at the last row before the restart
        (:meth:`test_restart`; decision 15 of
        ``docs/paradigm-analyses-plan.md``): the ratio starts again after
        it, so the first-gap rule sees the window only up to there and a
        count still running is censored."""
        gt = self.group_training(dfm_id, group)
        if not gt.complete:
            return None
        dfm = self.dfms[int(dfm_id)]
        end = self._recording_end(dfm)
        if self._reads_phase(dfm):
            run = self._test_ratio_run(dfm.paradigm, gt.sucrose_well)
            if run is not None:
                last = run[1] - 1
                restart = self.test_restart(dfm_id, group)
                if restart is not None:
                    last = min(last, restart.row - 1)
                end = float(dfm.paradigm.minutes[last])
        return settings.test_end(end - float(gt.training_end))

    def _test_events(self, dfm_id: int, group: int) -> pd.DataFrame:
        """The Test-phase rows of :meth:`light_events_table`."""
        events = self.light_events_table(dfm_id, group)
        return events[events["Facet"] == TEST_LABEL]

    def group_break(self, dfm_id: int, group: int,
                    settings: pbp.BreakSettings | None = None) -> pbp.GapBreak | None:
        """The first-gap rule on one Chamber Group's Test Light Events.

        ``count`` is the group's **Breaking Point**, ``last_min`` the minute
        since training end of the last event counted (0 when none),
        ``censored`` whether the Test window ended before any gap longer than
        ``pr_break_gap_min``, and ``counted`` which Test events count —
        aligned with the Test rows of :meth:`light_events_table`.  Lick-free
        Light Events never count and never end a gap.  ``None`` for a group
        that never completed training.  A DFM restart inside the Test window
        ends the window (:meth:`_test_window`), so events after it never
        count, and a count still running at it is censored.
        """
        settings = settings or self.break_settings()
        window = self._test_window(dfm_id, group, settings)
        if window is None:
            return None
        test = self._test_events(dfm_id, group)
        return pbp.breaking_point(test["Minutes"].to_numpy(dtype=float),
                                  ~test["LickFree"].to_numpy(dtype=bool),
                                  window, settings.gap_min)

    def chamber_persistence(self, dfm_id: int, chamber: int,
                            settings: pbp.BreakSettings | None = None,
                            ) -> tuple[float, bool] | None:
        """``(minutes, censored)``: the chamber's **Sucrose Persistence** — the
        first-gap rule on the onsets of its Sucrose Well feeding events (the
        events ``EventsA`` counts) since its group's training end.  Paired and
        yoked alike; ``None`` for a group that never completed training."""
        settings = settings or self.break_settings()
        dfm_id, chamber = int(dfm_id), int(chamber)
        cache = self._light_cache(dfm_id)
        key = ("persist", chamber, settings)
        if key in cache:
            return cache[key]
        group = self.role_of(dfm_id, chamber).group
        window = self._test_window(dfm_id, group, settings)
        result = None
        if window is not None:
            dfm = self.dfms[dfm_id]
            end = float(self.group_training(dfm_id, group).training_end)
            column = f"W{self._sucrose_well(dfm, chamber)}"
            mins = pd.to_numeric(dfm.event_df["Minutes"], errors="coerce").to_numpy(dtype=float)
            onsets = mins[dfm.event_df[column].to_numpy(dtype=float) > 0]
            result = pbp.persistence(onsets[onsets > end] - end, window, settings.gap_min)
        cache[key] = result
        return result

    def breaking_point_table(self, dfm_id: int, chamber: int) -> pd.DataFrame:
        """Per-light-on-period table for one chamber after its group's training
        end: one row at each moment the group's light switches on, with
        ``Minutes`` (since training end), ``CumLicks`` (sucrose well, since
        training end), ``DeltaMinutes`` and ``DeltaLicks`` since the previous
        onset.  Empty for a group that never completed training.

        For the **Paired** chamber — the one whose Sucrose Well the light
        answers to — the rows are the group's Light Event Ledger and carry
        :data:`LEDGER_COLUMNS` as well: ``MinutesSincePrev`` and
        ``LicksSincePrev`` count from the previous light event even when it
        fell in training, ``LickFree`` marks an event with no licks since the
        previous one ended, and ``RestingLevel`` is the well's per-minute
        median raw level at the onset (see :meth:`light_events_table`).
        ``Counted`` says whether the event is one of the group's Breaking
        Point (:meth:`group_break`).  The Yoked chamber's rows are the same
        onsets with its own licks; it has no breaking point of its own.

        An onset is a light event beginning after training end (on generation
        1, after the row that clears the flag: :meth:`_test_onsets_after`); an
        event lit across the training end belongs to training.
        """
        dfm_id, chamber = int(dfm_id), int(chamber)
        role = self.role_of(dfm_id, chamber)
        paired = role.role == ROLE_PAIRED
        cols = [*BREAKING_POINT_COLUMNS,
                *((*LEDGER_COLUMNS, COUNTED_COLUMN) if paired else ())]
        dfm = self.dfms[dfm_id]
        gt = self.group_training(dfm_id, role.group)
        if not gt.complete:
            return pd.DataFrame(columns=cols)
        end = float(gt.training_end)
        if paired:
            test = self._test_events(dfm_id, role.group)
            out = pd.DataFrame({"Minutes": test["Minutes"].to_numpy(dtype=float),
                                "CumLicks": test["CumLicks"].to_numpy(dtype=float)})
            for col in LEDGER_COLUMNS:
                ## The Series, not its values: Requirement and FirmwareEvents
                ## are nullable integers and Phase nullable text.
                out[col] = test[col].reset_index(drop=True)
            result = self.group_break(dfm_id, role.group)
            out[COUNTED_COLUMN] = (result.counted if result is not None
                                   else np.zeros(len(out), dtype=bool))
        else:
            mins = pd.to_numeric(dfm.lick_df["Minutes"], errors="coerce").to_numpy(dtype=float)
            well_a = self._sucrose_well(dfm, chamber)
            licks = dfm.lick_df[f"W{well_a}"].to_numpy(dtype=float)
            cum = np.cumsum(np.where(mins > end, licks, 0.0))
            if self._reads_phase(dfm):
                ## The group's Light Events are its Paired Sucrose Well's
                ## Activations, which no light of the Yoked chamber's own
                ## records: the Yoked rows are the same onsets.
                onsets, _ends = self._activation_events(dfm.paradigm, gt.sucrose_well)
            else:
                onsets, _ends = lqc.light_events(
                    self._chamber_light(dfm, chamber).to_numpy(dtype=bool))
            ## The Paired chamber's Test onsets, by the same boundary.
            onsets = onsets[mins[onsets] > self._test_onsets_after(dfm, gt)]
            out = pd.DataFrame({"Minutes": mins[onsets] - end, "CumLicks": cum[onsets]})
        if out.empty:
            return pd.DataFrame(columns=cols)
        out["DeltaMinutes"] = np.concatenate([[0.0], np.diff(out["Minutes"].to_numpy())])
        out["DeltaLicks"] = np.concatenate([[0.0], np.diff(out["CumLicks"].to_numpy())])
        return out[cols].reset_index(drop=True)

    def breaking_point_dfm(self, dfm_id: int) -> dict[int, pd.DataFrame]:
        """:meth:`breaking_point_table` for every chamber of *dfm_id*."""
        return {c: self.breaking_point_table(dfm_id, c) for c in range(1, 7)}

    def plot_breaking_point_dfm(self, dfm_id: int, *, base_font_size: float = 10.0,
                                figsize: tuple[float, float] = (10.5, 6.0)):
        """ΔLicks per light-on period against minutes since training end, one
        panel per chamber, the role and the group's breaking point in the strip.

        The break is marked on both chambers of a group: a dashed line at the
        last light event the Breaking Point counts, and every onset past it —
        or past the Test window — in grey.  On the paired panel a Lick-free
        Light Event, which never counts, is a hollow red ring.  A censored
        group has no line: it was still responding when its window ended.
        """
        import plotnine as p9

        dfm_id = int(dfm_id)
        settings = self.break_settings()
        before, after, free_label = "before the break", "after the break", "lick-free"
        frames, breaks = [], []
        for chamber, df in self.breaking_point_dfm(dfm_id).items():
            if df.empty:
                continue
            r = self.role_of(dfm_id, chamber)
            result = self.group_break(dfm_id, r.group, settings)
            window = self._test_window(dfm_id, r.group, settings)
            minutes = df["Minutes"].to_numpy(dtype=float)
            ## A group auto-removal took out is drawn for reference only: its
            ## light followed the sensor, so no break is marked on it.
            removed = (self.design.treatment_for(dfm_id, chamber) is None
                       and (dfm_id, chamber) in self._design_snapshot())
            marked = result is not None and not removed
            if r.role == ROLE_PAIRED:
                lick_free = df["LickFree"].to_numpy(dtype=bool)
                counted = (df[COUNTED_COLUMN].to_numpy(dtype=bool) if marked
                           else ~lick_free)
                status = np.where(counted, before,
                                  np.where(lick_free, free_label, after))
            else:
                late = np.zeros(minutes.size, dtype=bool)
                if marked and not result.censored:
                    late |= minutes > result.last_min
                if marked and window is not None:
                    late |= minutes > window
                status = np.where(late, after, before)
            label = f"Chamber {chamber} ({r.role}, group {r.group})"
            if marked and r.role == ROLE_PAIRED:
                label += f" — BP {pbp.format_count(result.count, result.censored)}"
            if removed:
                label += " — excluded"
            frames.append(pd.DataFrame({
                "Minutes": minutes,
                "DeltaLicks": df["DeltaLicks"].to_numpy(dtype=float),
                "Status": status, "Chamber": label}))
            if marked and not result.censored:
                breaks.append({"Chamber": label, "BreakMin": float(result.last_min)})
        if not frames:
            return p9.ggplot() + p9.labs(title=f"DFM {dfm_id} — no breaking-point data")
        data = pd.concat(frames, ignore_index=True)
        order = list(dict.fromkeys(data["Chamber"]))
        data["Chamber"] = pd.Categorical(data["Chamber"], categories=order, ordered=True)
        dots = data[data["Status"] != free_label]
        free = data[data["Status"] == free_label]
        g = (p9.ggplot(data, p9.aes("Minutes", "DeltaLicks"))
             + p9.geom_line(size=0.4, color="#9E9E9E"))
        if breaks:
            marks = pd.DataFrame(breaks)
            marks["Chamber"] = pd.Categorical(marks["Chamber"], categories=order,
                                              ordered=True)
            g += p9.geom_vline(marks, p9.aes(xintercept="BreakMin"), linetype="dashed",
                               color="#555555", size=0.5)
        if not dots.empty:
            g += p9.geom_point(dots, p9.aes(color="Status"), size=1.6)
        if not free.empty:
            g += p9.geom_point(free, shape="o", fill="none", color="#D62728", size=2.2,
                               stroke=0.6)
        return (g
                + p9.scale_color_manual(values={before: "#2166ac", after: "#BDBDBD"})
                + p9.facet_wrap("~ Chamber", ncol=3, scales="free_y")
                + p9.labs(title=f"DFM {dfm_id} — ΔLicks per light-on period",
                          caption=(f"Dashed: the break, the last light event counted "
                                   f"(pause over {settings.gap_min:g} min).\n"
                                   f"Grey: past the break.  Hollow red: lick-free, "
                                   f"never counted.  BP n+: censored."),
                          x="Time since training end (min)", y="ΔLicks", color="")
                + p9.theme_bw(base_size=base_font_size)
                + p9.theme(figure_size=figsize, legend_position="bottom"))

    def plot_still_responding(self, *, base_font_size: float = 10.0,
                              figsize: tuple[float, float] = (6.5, 4.5)):
        """The still-responding curve: per Treatment, the Kaplan-Meier fraction
        of paired flies that reached each ratio, a censored fly as a tick
        (:func:`pyflic.base.analytics.still_responding`)."""
        from . import report_content as rc

        return rc.still_responding_plot(self.breaking_point_summary(),
                                        base_font_size=base_font_size, figsize=figsize)

    # ------------------------------------------------------------------
    # Summary text and the basic pipeline
    # ------------------------------------------------------------------

    def summary_text(self, *, include_qc: bool = True,
                     qc_data_breaks_multiplier: float = 4.0,
                     qc_bleeding_cutoff: float = 50.0) -> str:
        text = Experiment.summary_text(
            self, include_qc=include_qc,
            qc_data_breaks_multiplier=qc_data_breaks_multiplier,
            qc_bleeding_cutoff=qc_bleeding_cutoff)
        buf = ["", "Progressive Ratio — training by chamber group",
               "--------------------------------------------"]
        information = self.training_information()
        if information:
            buf.append(information)
        table = self.training_table()
        if table.empty:
            buf.append("(no DFMs)")
        else:
            show = table.drop(columns=["Notes"]).copy()
            unknown = show["TrainingComplete"].isna()
            show["TrainingEndMin"] = [
                "n/a" if missing else "never" if pd.isna(v) else f"{v:.1f}"
                for v, missing in zip(show["TrainingEndMin"], unknown)]
            if unknown.any():
                show["TrainingComplete"] = ["NA" if missing else str(bool(v))
                                            for v, missing in zip(show["TrainingComplete"], unknown)]
            buf.append(show.to_string(index=False))
        warnings = self.training_warnings()
        buf.append("")
        ## Generation 1 keeps its words; Phase and the missing information
        ## have no flags to speak of.
        if any(dfm.training_info == "flag" for dfm in self.dfms.values()):
            buf.append("Training-flag notes (QC, not errors)")
            buf.append("------------------------------------")
            quiet = "  (all four wells of every group cleared together)"
        else:
            buf.append("Training notes (QC, not errors)")
            buf.append("-------------------------------")
            quiet = "  (none)"
        if warnings:
            buf.extend(f"  {w}" for w in warnings)
        else:
            buf.append(quiet)
        buf.extend(self._light_qc_summary_lines())
        buf.extend(self._breaking_point_summary_lines())
        return text + "\n".join(buf) + "\n"

    # ------------------------------------------------------------------
    # Breaking point, per Chamber Group
    # ------------------------------------------------------------------

    def breaking_point_summary(self) -> pd.DataFrame:
        """One row per Chamber Group in the analysis — its Paired chamber in
        the design (a pair's Yoked chamber may be excluded: the Breaking Point
        is read from the Paired fly alone) and training complete — as
        ``analysis/pr_breaking_point.csv``.

        ``BreakingPoint`` is the paired fly's lick-backed Test Light Events
        before the first gap longer than ``pr_break_gap_min`` (ADR-0014);
        ``BreakMin`` the minute since training end of the last one counted (0
        when none); ``Censored`` true when no such gap came before the Test
        window ended, so the count is a lower bound; ``TestMinutes`` the window
        the rule saw (recording end minus training end, capped at
        ``pr_test_window_min``; on generation 3 the end of the ratio
        Paradigm's run, or a DFM restart inside it); ``LargestRequirement`` the most Sucrose Well licks
        credited to one counted event, descriptive only, and on generation 3
        the recorded ``BreakRequirement``; ``BreakRequirement`` the
        Requirement the last counted Light Event met and ``FirmwareEvents``
        the Firmware Events credited to the counted ones (generation 3, NA
        otherwise; :meth:`_break_requirement`); ``LickFreeLightEvents`` and
        ``LightQC`` the group's light QC count and flags.  A group the light QC
        excluded is not here; one it failed but kept
        (``exclude_failed_pr_groups: false``) is, whole, with its flags.  A
        group with no Test phase, which includes every group on a DFM that
        records no Training, has no row: its light QC notes say why.
        """
        factors = list(self.design_factors or [])
        cols = ["Treatment", *factors, *BREAKING_POINT_SUMMARY_COLUMNS]
        settings = self.break_settings()
        by_group = self._light_qc_by_group()
        rows = []
        for dfm_id, layout in self._group_pairs():
            group = layout.number
            ## The Breaking Point is the Paired fly's alone, so only its chamber
            ## must still be in the analysis: a pair whose Yoked chamber was
            ## excluded (lick cutoff, remove_chambers.csv) keeps its row.
            if self.design.treatment_for(dfm_id, layout.paired) is None:
                continue
            result = self.group_break(dfm_id, group, settings)
            if result is None:
                continue
            gt = self.group_training(dfm_id, group)
            test = self._test_events(dfm_id, group)
            credited = test["LicksSincePrev"].to_numpy(dtype=float)[result.counted]
            largest = int(credited.max()) if credited.size else 0
            recorded = self._reads_phase(self.dfms[dfm_id])
            requirement, firmware = self._break_requirement(
                dfm_id, group, test, result.counted)
            qc = by_group.get((int(dfm_id), int(group)), {})
            rows.append({
                "Treatment": self.design.treatment_for(dfm_id, gt.paired_chamber),
                "DFM": int(dfm_id), "Chamber": int(gt.paired_chamber),
                "Group": int(group), "PairedChamber": int(gt.paired_chamber),
                "BreakingPoint": int(result.count),
                "BreakMin": float(result.last_min),
                "Censored": bool(result.censored),
                "TestMinutes": float(self._test_window(dfm_id, group, settings)),
                "LargestRequirement": requirement if recorded else largest,
                "BreakRequirement": requirement,
                "FirmwareEvents": firmware,
                "LickFreeLightEvents": qc.get("LickFreeEvents", np.nan),
                "LightQC": qc.get("Flags", ""),
                "ParadigmType": self.test_paradigm_type(dfm_id, group),
            })
        if not rows:
            return pd.DataFrame(columns=cols)
        out = pd.DataFrame(rows)
        for col in ("BreakRequirement", "FirmwareEvents"):
            out[col] = pd.array(out[col].tolist(), dtype="Int64")
        if out["LargestRequirement"].isna().any():
            out["LargestRequirement"] = pd.array(out["LargestRequirement"].tolist(),
                                                 dtype="Int64")
        return self._append_factor_columns(out)[cols].reset_index(drop=True)

    def _break_requirement(self, dfm_id: int, group: int, test: pd.DataFrame,
                           counted: np.ndarray) -> tuple[int | None, int | None]:
        """``(BreakRequirement, FirmwareEvents)`` of a group's Breaking Point
        from its Test Light Events and the mask of those counted.

        ``BreakRequirement`` is the Requirement the last counted Light Event
        met, read from the row before the one where ``Activations`` rose
        (ADR-0014 addendum), or under a FixedRatio Test Paradigm its Ratio
        (decision 14); NA when nothing was counted or the rows do not show
        it.  ``FirmwareEvents`` sums the Firmware Events credited to the
        counted events.  ``(None, None)`` off generation 3.
        """
        if not self._reads_phase(self.dfms[int(dfm_id)]):
            return None, None
        counted = np.asarray(counted, dtype=bool)
        firmware = int(test["FirmwareEvents"][counted].sum())
        if not counted.any():
            return None, firmware
        _name, paradigm = self.test_paradigm(dfm_id, group)
        if paradigm is not None and paradigm.type == FIXED_RATIO and paradigm.ratio is not None:
            return int(paradigm.ratio), firmware
        last = test["Requirement"][counted].iloc[-1]
        return (None if pd.isna(last) else int(last)), firmware

    def breaking_point_sensitivity(self, gaps: Sequence[float] | None = None) -> pd.DataFrame:
        """Every group of :meth:`breaking_point_summary` under other values of
        ``pr_break_gap_min``: ``DFM, Group`` and, per gap, ``BP_<gap>`` (the
        Breaking Point) and ``Censored_<gap>``.  By default the gaps are
        :data:`pyflic.base.pr_breaking_point.SENSITIVITY_GAPS_MIN` and the
        configured one."""
        settings = self.break_settings()
        if gaps is None:
            gaps = sorted({*pbp.SENSITIVITY_GAPS_MIN, settings.gap_min})
        gaps = [float(g) for g in gaps]
        cols = ["DFM", "Group",
                *(c for g in gaps for c in (f"BP_{g:g}", f"Censored_{g:g}"))]
        rows = []
        for r in self.breaking_point_summary().itertuples(index=False):
            row: dict[str, Any] = {"DFM": int(r.DFM), "Group": int(r.Group)}
            for g in gaps:
                result = self.group_break(r.DFM, r.Group, settings.with_gap(g))
                row[f"BP_{g:g}"] = int(result.count)
                row[f"Censored_{g:g}"] = bool(result.censored)
            rows.append(row)
        return pd.DataFrame(rows, columns=cols)

    def write_breaking_point(self, path: str | Path | None = None) -> Path:
        """Write ``analysis/pr_breaking_point.csv`` — one row per Chamber Group
        (:meth:`breaking_point_summary`), which a Project stacks."""
        if path is None:
            if self.analysis_dir is None:
                raise ValueError("path must be provided when no experiment_dir is set.")
            path = self.analysis_dir / "pr_breaking_point.csv"
        out = Path(path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        self.breaking_point_summary().to_csv(out, index=False, na_rep="NA")
        return out

    def breaking_point_lines(self) -> list[str]:
        """The breaking point by chamber group and its sensitivity to
        ``pr_break_gap_min`` as plain text — for ``summary.txt`` and the
        Script Editor's log.  ``+`` marks a censored count."""
        settings = self.break_settings()
        summary = self.breaking_point_summary()
        lines = [f"Settings: {settings.describe()}", ""]
        if summary.empty:
            return lines + ["(no chamber group in the analysis has a Test phase)"]
        show = pd.DataFrame({
            "DFM": summary["DFM"], "Group": summary["Group"],
            "Paired": summary["PairedChamber"], "Treatment": summary["Treatment"],
            "BreakingPoint": [pbp.format_count(c, x) for c, x
                              in zip(summary["BreakingPoint"], summary["Censored"])],
            "BreakMin": [f"{v:.1f}" for v in summary["BreakMin"]],
            "TestMin": [f"{v:.1f}" for v in summary["TestMinutes"]],
            "LargestReq": summary["LargestRequirement"],
            "LightQC": [_light_qc_verdict_text(f) for f in summary["LightQC"]],
        })
        if summary["BreakRequirement"].notna().any():
            ## Generation 3: the Requirement the firmware recorded, beside the
            ## count it belongs to.
            show.insert(list(show.columns).index("BreakingPoint") + 1, "BreakReq",
                        [_count_text(v) for v in summary["BreakRequirement"]])
            show["LargestReq"] = [_count_text(v) for v in summary["LargestRequirement"]]
        lines.append(show.to_string(index=False))
        sens = self.breaking_point_sensitivity()
        table = pd.DataFrame({"DFM": sens["DFM"], "Group": sens["Group"]})
        for col in [c for c in sens.columns if c.startswith("BP_")]:
            gap = col[3:]
            head = f"gap {gap}" + ("*" if float(gap) == settings.gap_min else "")
            table[head] = [pbp.format_count(c, x) for c, x
                           in zip(sens[col], sens[f"Censored_{gap}"])]
        lines += ["", "Breaking point at other gaps (minutes; * = pr_break_gap_min):",
                  table.to_string(index=False)]
        censored = int(summary["Censored"].astype(bool).sum())
        if censored:
            lines += ["", f"{censored} of {len(summary)} chamber group(s) censored: still "
                          f"responding when the Test window ended, so the count is a "
                          f"lower bound."]
        ## Decision 15: a DFM restart inside a Test window ends it there.
        for r in summary.itertuples(index=False):
            restart = self.test_restart(r.DFM, r.Group)
            if restart is not None:
                end = float(self.group_training(r.DFM, r.Group).training_end)
                lines.append(f"DFM {int(r.DFM)} group {int(r.Group)}: "
                             f"{restart_note(restart, end)}")
        return lines

    def _breaking_point_summary_lines(self) -> list[str]:
        lines = ["", "Progressive ratio breaking point",
                 "--------------------------------",
                 "The paired fly's lick-backed Test light events before its first pause",
                 "longer than pr_break_gap_min; a lick-free light event neither counts",
                 "nor ends a pause.  '+' marks a censored count: no such pause came",
                 "before the Test window ended, so the count is a lower bound."]
        if any(self._reads_phase(dfm) for dfm in self.dfms.values()):
            lines += ["BreakReq is the Requirement the last counted light event met, as",
                      "the firmware recorded it; a fixed ratio's is its Ratio."]
        missing = sorted(int(d) for d, dfm in self.dfms.items() if dfm.training_info == "none")
        if missing:
            lines.append(f"DFM {', '.join(str(d) for d in missing)}: no Breaking Point or "
                         f"Sucrose Persistence — {NO_TRAINING_INFORMATION}, so there is "
                         f"no Test phase.")
        return [*lines, *self.breaking_point_lines(), ""]

    # ------------------------------------------------------------------
    # Experiment report hooks (pyflic.base.pdf_report)
    # ------------------------------------------------------------------

    def report_glance_blocks(self) -> list[Any]:
        from . import report_content as rc
        from . import report_layout as rl

        table = self.light_qc_table()
        excluded = [(int(r.DFM), int(r.Group)) for r in table.itertuples() if r.Excluded]
        kept = [(int(r.DFM), int(r.Group)) for r in table.itertuples()
                if r.Verdict == lqc.VERDICT_FAILED and not r.Excluded]
        warned = [(int(r.DFM), int(r.Group)) for r in table.itertuples()
                  if r.Verdict == lqc.VERDICT_WARNING]
        n = len(table)
        if excluded or kept:
            parts = []
            if excluded:
                parts.append(f"{len(excluded)} of {n} chamber groups failed and were "
                             f"excluded ({rc.join_groups(excluded)})")
            if kept:
                parts.append(f"{len(kept)} failed but were kept because "
                             f"exclude_failed_pr_groups is off ({rc.join_groups(kept)})")
            text = "; ".join(parts) + "."
            tone = "failed"
        else:
            text = f"Every chamber group's light followed its paired fly ({n} of {n})."
            tone = "ok"
        if warned:
            text += (f"  {len(warned)} warning(s): {rc.join_groups(warned)} — "
                     f"see Quality control.")
            tone = "warning" if tone == "ok" else tone
        ## Only groups known not to have trained: TrainingComplete is NA on a
        ## DFM that records no Training, which has its own callout.
        incomplete = [(int(r.DFM), int(r.Group)) for r in table.itertuples()
                      if _flag(r.TrainingComplete) is False]
        blocks = [rl.Callout(text, tone=tone,
                             title="Light QC — did the paired fly earn its light?")]
        information = self.training_information()
        if information:
            ## The line verbatim: the Hub and the Project Report say the same.
            blocks.append(rl.Callout(
                f"{information}.  Each chamber group is one Recording facet, kept "
                f"whole: no Test phase, Breaking Point or Sucrose Persistence.",
                tone="warning", title="Training"))
        if incomplete:
            required = _truthy((self.global_constants or {}).get(
                "require_training_complete", True))
            whose = self._group_chambers_phrase("each")
            blocks.append(rl.Callout(
                f"Training never completed in {rc.join_groups(incomplete)}; "
                + (f"{whose} {'was' if whose.startswith('the chamber') else 'were'} "
                   f"excluded." if required
                   else "they are kept (require_training_complete is off)."),
                tone="warning", title="Training"))
        return blocks

    def report_qc_blocks(self) -> list[Any]:
        from . import report_layout as rl

        training = self.training_table()
        training_view = pd.DataFrame({
            "DFM": training["DFM"], "Group": training["Group"],
            "Treatment": [self._group_treatment(d, g)
                          for d, g in zip(training["DFM"], training["Group"])],
            "Paired chamber": training["PairedChamber"],
            "Yoked chamber": training["YokedChamber"],
            "Sucrose Well": [f"W{int(w)}" for w in training["SucroseWell"]],
            "Training end (min)": training["TrainingEndMin"],
            "Training": ["no information" if _flag(c) is None else
                         "complete" if c else "never completed"
                         for c in training["TrainingComplete"]],
        })
        if not self.has_yoked_controls():
            ## Every group is one chamber: a column of NA says nothing.
            training_view = training_view.drop(columns="Yoked chamber")

        def training_tone(value: Any) -> str | None:
            return ("ok" if value == "complete" else
                    "warning" if value == "no information" else "failed")

        kinds = {dfm.training_info for dfm in self.dfms.values()}
        where = []
        if "flag" in kinds:
            where.append("at the last minute the firmware flags the paired chamber's "
                         "Sucrose Well as in training")
        if "phase" in kinds:
            where.append("at the paired chamber's Sucrose Well's first row in Phase M "
                         "under a ratio Paradigm (the end of its Acclimation, or of the "
                         "Interval before the ratio Paradigm)")
        lead = (f"Training ends, per chamber group, {' or '.join(where)}; the Test phase is "
                f"everything after it, and Progressive Ratio time axes run from there."
                if where else "")
        information = self.training_information()
        if information:
            lead += f"  ({information} — one Recording facet per chamber group.)"
        blocks: list[Any] = [
            rl.Heading("Progressive ratio: training", level=2),
            rl.Paragraph(lead.strip()),
            rl.Table(training_view, caption="Training by chamber group",
                     formats={"Training end (min)": "{:.1f}"},
                     status={"Training": training_tone}),
        ]
        notes = self.training_warnings()
        if notes:
            blocks.append(rl.Paragraph(("Training-flag notes (QC, not errors):"
                                        if "flag" in kinds else
                                        "Training notes (QC, not errors):"),
                                       size=rl.SIZE_SMALL, color=rl.MUTED))
            blocks.append(rl.Bullets(notes, size=rl.SIZE_SMALL, color=rl.MUTED))

        settings = self.light_qc_settings()
        table = self.light_qc_table()
        view = pd.DataFrame({
            "DFM": table["DFM"], "Group": table["Group"], "Treatment": table["Treatment"],
            "Training light events": table["TrainingLightEvents"],
            "Training licks": table["TrainingLicks"],
            "Test light events": table["TestLightEvents"],
            "Lick-free": table["LickFreeEvents"],
            "Longest run": table["LongestLickFreeRun"],
            "Trend rho": table["TrendRho"],
            "Resting rise": table["RestingRise"],
            "Resting ratio": table["RestingRatio"],
            "Verdict": [("excluded" if ex else v)
                        for v, ex in zip(table["Verdict"], table["Excluded"])],
        })
        firmware = ""
        if table["TrendRhoFirmware"].notna().any():
            view.insert(list(view.columns).index("Trend rho") + 1, "Firmware trend rho",
                        table["TrendRhoFirmware"])
            firmware = ("  The firmware records each light event's Activation, so the "
                        "trend is also read on its Firmware Events; one that rises while "
                        "the licks do not is a note on pyflic's detection, not a warning.")
        progressive = bool(self.progressive_groups())
        warnings = ("no increasing trend in licks per light event and a rising or "
                    "elevated resting level are warnings" if progressive else
                    "a rising or elevated resting level is a warning; under a fixed ratio "
                    "the requirement does not rise, so licks per light event are not "
                    "judged for a trend")
        blocks += [
            rl.Heading("Progressive ratio: light QC", level=2),
            rl.Paragraph(
                "The firmware lights a group from its own reading of the paired chamber's "
                "Sucrose Well during the run; pyflic counts licks afterwards, from the "
                "baselined signal.  A Sucrose Well whose resting level creeps up looks "
                "continuously touched to the firmware and flat to pyflic, so the light "
                "fires on its own schedule.  Self-triggered light (a run of "
                f"{settings.lick_free_run} or more lick-free Test light events) and "
                "implausible training (training light events with no sucrose lick) fail a "
                f"group; {warnings}.  Computed over the whole recording."
                + (firmware if progressive else "")),
            rl.Table(view, caption="Light QC by chamber group",
                     formats={"Trend rho": "{:.2f}", "Firmware trend rho": "{:.2f}",
                              "Resting rise": "{:.0f}", "Resting ratio": "{:.1f}"},
                     status={"Verdict": rl.tone_of}),
            rl.Paragraph(f"Thresholds: {settings.describe()}.", size=rl.SIZE_SMALL,
                         color=rl.MUTED),
        ]
        detail = self._light_qc_bullets()
        if detail:
            blocks.append(rl.Bullets(detail, size=rl.SIZE_SMALL + 0.5))
        inc = self.estimated_increment()
        if inc is not None:
            blocks.append(rl.Paragraph(
                f"Estimated requirement increment: {inc[0]:.1f} licks per light event "
                f"(median over {inc[2]} chamber group(s)' first {lqc.INCREMENT_EVENTS} Test "
                f"light events).", size=rl.SIZE_SMALL, color=rl.MUTED))
        for dfm_id in sorted(self.dfms):
            ## One row of panels per three groups: six solo groups take two.
            rows = -(-len(self.chamber_groups(dfm_id)) // 3)
            caption = ("Sucrose Well licks credited to each Test light event. "
                       "A working progressive ratio climbs; hollow red rings are "
                       "lick-free light events, the dashed line the group's trend, "
                       "the grey line the estimated requirement."
                       if self.progressive_groups(dfm_id) else
                       "Sucrose Well licks credited to each Test light event, under a "
                       "fixed ratio; hollow red rings are lick-free light events, the "
                       "dashed line the group's trend.")
            blocks += [
                rl.Heading(f"Light QC figures — DFM {dfm_id}", level=2),
                rl.Plot(lambda d=dfm_id: self.plot_light_events_dfm(d, base_font_size=8.5),
                        height=3.4 * rows, title=f"Licks per light event — DFM {dfm_id}",
                        caption=caption),
                rl.Plot(lambda d=dfm_id: self.plot_resting_level_dfm(d, base_font_size=8.5),
                        height=3.2 * rows, title=f"Sucrose Well resting level — DFM {dfm_id}",
                        caption="Per-minute median raw signal of the paired Sucrose Well "
                                "against the DFM's other Sucrose Wells; light onsets as a "
                                "rug, training end dashed."),
                rl.Plot(lambda d=dfm_id: self.plot_cumulative_licks_dfm(d, base_font_size=8.5),
                        height=3.4 * rows, title=f"Training-aligned traces — DFM {dfm_id}",
                        caption=("Paired and yoked cumulative Sucrose Well licks since "
                                 "training end; points: light on; rings: lick-free light "
                                 "events." if any(not g.solo for g in self.chamber_groups(dfm_id))
                                 else "Each paired fly's cumulative Sucrose Well licks since "
                                      "training end; points: light on; rings: lick-free "
                                      "light events.")),
            ]
        return blocks

    def _light_qc_bullets(self) -> list[str]:
        """:meth:`light_qc_lines` as one line per flagged or noted group."""
        items: list[str] = []
        current: str | None = None
        for line in self.light_qc_lines():
            stripped = line.strip()
            if not line.startswith(" "):
                if current:
                    items.append(current)
                current = stripped
            elif current is not None:
                sep = " " if current.endswith(":") else "; "
                current += sep + stripped.lstrip("-· ").strip()
        if current:
            items.append(current)
        return items

    def report_results_blocks(self, generic: list[Any], options: Any) -> list[Any]:
        """Replaces the layout's results: the per-treatment plots of a two-well
        experiment pool each paired fly with its own yoked control.  In their
        place — the Cumulative Difference Curve, the Paired-Yoked Difference in
        the Test phase (between treatments and against zero), and the breaking
        point (ADR-0014).  A design of solo groups has no yoked controls, so
        the first two give way to one sentence saying so (decision 13)."""
        from . import report_content as rc
        from . import report_layout as rl

        names = self.well_names or {}
        well_a = names.get("A") or "well A"
        settings = self.break_settings()
        half = (rl.CONTENT_W - 0.25) / 2

        def dot(frame, column, label, hline=None):
            return lambda: rc.dot_plot(frame, column, y_label=label,
                                       factors=self.design_factors, hline_at=hline)

        if self.has_yoked_controls():
            blocks = self._paired_yoked_blocks(options, well_a, settings, half, dot)
        else:
            blocks = [rl.Paragraph(f"{NO_YOKED_CONTROLS.capitalize()}: every chamber group "
                                   f"is one chamber, lit by its own fly alone, so there is "
                                   f"no paired − yoked difference or cumulative difference "
                                   f"curve.")]
        blocks += self._breaking_point_blocks(options, settings)
        return blocks

    def _paired_yoked_blocks(self, options: Any, well_a: str, settings, half: float,
                             dot) -> list[Any]:
        """The Cumulative Difference Curve and the Paired-Yoked Difference in
        the Test phase, over the groups that are pairs; a sentence names the
        solo groups left out."""
        from . import report_content as rc
        from . import report_layout as rl
        from .analytics import treatment_comparisons, zero_tests

        names = self.well_names or {}
        diff = self.paired_yoked_diff()
        test = diff[diff["Facet"] == TEST_LABEL] if not diff.empty else diff
        ## paired_yoked_diff() differences the summary as this experiment
        ## transforms it: a lick difference on the fourth-root scale says so.
        transformed = bool(self.transform_licks)
        scale = (f"  Lick differences are on the {rc.ROOT_SCALE} (transform_licks): "
                 f"the paired fly's licks^0.25 minus its yoked partner's."
                 if transformed else "")
        solo = self.solo_groups()
        left_out = ""
        if solo:
            where = rc.join_groups(solo)
            left_out = (f"  {where[0].upper()}{where[1:]} "
                        f"{'is one chamber' if len(solo) == 1 else 'are one chamber each'}, "
                        f"with no yoked control, and {'is' if len(solo) == 1 else 'are'} "
                        f"left out of the paired − yoked comparison.")
        blocks: list[Any] = [
            rl.Heading("Cumulative difference curve", level=2),
            rl.Paragraph(
                f"Paired minus yoked cumulative {well_a} licks since each chamber group's "
                f"training end: above zero, the paired fly worked for the light more than "
                f"its yoked control fed.  Mean ± SEM per treatment, drawn only over the "
                f"time every group covers; the faint lines are the individual groups.  "
                f"Groups the light QC excluded are not included.{left_out}"),
            rl.Plot(lambda: self.plot_cumulative_diff(base_font_size=9.5), height=3.5),
            rl.Heading("Paired − yoked difference, Test phase", level=2),
            rl.Paragraph(
                "One point per chamber group: the paired fly's value minus its yoked "
                "partner's over the Test phase.  Zero is the null.  The first table asks "
                "whether treatments differ, the second whether the difference is non-zero "
                f"within each treatment.  Sucrose persistence is the time from training end "
                f"to a fly's last {well_a} feeding event before a pause longer than "
                f"{settings.gap_min:g} minutes; where either fly was still feeding when "
                f"the recording ended, the difference is one of lower bounds.{scale}"),
        ]
        if test.empty:
            blocks.append(rl.Callout("No chamber group has both chambers and a Test phase "
                                     "in the analysis.", tone="warning"))
        else:
            blocks.append(rl.PlotRow([
                rl.Plot(dot(test, "dLicksA", rc.metric_label("dLicksA", names,
                                                             transformed=transformed), 0.0),
                        title=rc.metric_label("dLicksA", names, transformed=transformed)),
                rl.Plot(dot(test, "dPI", rc.metric_label("dPI", names), 0.0),
                        title=rc.metric_label("dPI", names)),
            ], height=3.0))
            if "dPersistA" in test.columns and test["dPersistA"].notna().any():
                blocks.append(rl.PlotRow([
                    rl.Plot(dot(test, "dPersistA", rc.metric_label("dPersistA", names), 0.0),
                            title=rc.metric_label("dPersistA", names), width=half),
                ], height=3.0))
            if options.include_comparison:
                metrics = list(DIFF_REPORT_METRICS)
                rows = treatment_comparisons([(TEST_LABEL, test)], metrics)
                blocks += rc.stats_table(rows, caption="Treatment comparisons: paired − "
                                                       "yoked difference, Test phase",
                                         well_names=names, show_phase=False,
                                         transformed=transformed)
                blocks += rc.zero_test_table(
                    zero_tests([(TEST_LABEL, test)], metrics),
                    caption="Paired − yoked difference against zero, per treatment, "
                            "Test phase", well_names=names, transformed=transformed)
        return blocks

    def _breaking_point_blocks(self, options: Any, settings) -> list[Any]:
        """The breaking point section of the results (ADR-0014): its rule, the
        still-responding curve and dot plot, the comparisons, the table by
        group and the per-DFM ΔLicks figures."""
        from . import report_content as rc
        from . import report_layout as rl
        from .analytics import breaking_point_comparisons

        names = self.well_names or {}
        blocks: list[Any] = []
        summary = self.breaking_point_summary()
        window = ("" if settings.test_window_min is None else
                  f"  Every Test window is capped at {settings.test_window_min:g} minutes "
                  f"(pr_test_window_min).")
        ## A FixedRatio Test Paradigm runs the same first-gap rule; what it
        ## measures is activations earned at a constant requirement (decision 14).
        tested = [(d, g.number) for d, g in self._group_pairs()
                  if self.group_training(d, g.number).complete]
        fixed = bool(tested) and all(self._fixed_ratio(d, g) for d, g in tested)
        recorded = summary["BreakRequirement"].notna().any() if not summary.empty else False
        extra = ""
        if fixed:
            extra = ("  The Test Paradigm is a fixed ratio: every activation needed the "
                     "same number of Firmware Events, its Ratio, so the count is the "
                     "activations earned before the first pause.")
        elif recorded:
            extra = ("  The firmware recorded the Requirement each light event met; the "
                     "break requirement is the last counted one's.")
        yoked = ("  The yoked fly has no breaking point: its light is its partner's."
                 if self.has_yoked_controls() else "")
        blocks += [
            rl.Heading("Fixed ratio: activations earned" if fixed else "Breaking point",
                       level=2),
            rl.Paragraph(
                f"The breaking point is the number of lick-backed Test light events the "
                f"paired fly completed before its first pause longer than "
                f"{settings.gap_min:g} minutes (pr_break_gap_min); a lick-free light event "
                f"neither counts nor ends a pause.  A group still responding when its Test "
                f"window ended is censored: its count is a lower bound, drawn open, and the "
                f"curve and the log-rank test treat it so.{yoked}{window}{extra}"),
        ]
        restarts = [f"DFM {d} group {g.number}: "
                    f"{restart_note(span, float(self.group_training(d, g.number).training_end))}"
                    for d, g in self._group_pairs()
                    for span in [self.test_restart(d, g.number)] if span is not None]
        if restarts:
            blocks.append(rl.Bullets(restarts, size=rl.SIZE_SMALL))
        if summary.empty:
            blocks.append(rl.Callout("No chamber group in the analysis has a Test phase.",
                                     tone="warning"))
        else:
            blocks.append(rl.PlotRow([
                rl.Plot(lambda: rc.still_responding_plot(summary, base_font_size=8.5),
                        title="Still responding"),
                rl.Plot(lambda: rc.censored_dot_plot(summary, "BreakingPoint",
                                                     y_label="Light events earned",
                                                     factors=self.design_factors),
                        title="Breaking point by treatment"),
            ], height=3.2))
            if options.include_comparison:
                blocks += rc.stats_table(breaking_point_comparisons(summary),
                                         caption="Treatment comparisons: breaking point",
                                         well_names=names, show_phase=False)
            view = pd.DataFrame({
                "DFM": summary["DFM"], "Group": summary["Group"],
                "Treatment": summary["Treatment"],
                "Paired chamber": summary["PairedChamber"],
                "Breaking point": [pbp.format_count(c, x) for c, x
                                   in zip(summary["BreakingPoint"], summary["Censored"])],
                "Break (min)": summary["BreakMin"],
                "Test minutes": summary["TestMinutes"],
                "Largest requirement": summary["LargestRequirement"],
                "Light QC": [_light_qc_verdict_text(f) for f in summary["LightQC"]],
            })
            if recorded:
                view.insert(list(view.columns).index("Breaking point") + 1,
                            "Break requirement", summary["BreakRequirement"])
            blocks.append(rl.Table(view, caption="Breaking point by chamber group "
                                                 "(+: censored, a lower bound)",
                                   formats={"Break (min)": "{:.0f}",
                                            "Test minutes": "{:.0f}"},
                                   status={"Light QC": rl.tone_of}))
        for dfm_id in sorted(self.dfms):
            roles = ("both roles; " if any(not g.solo for g in self.chamber_groups(dfm_id))
                     else "")
            blocks.append(rl.Plot(
                lambda d=dfm_id: self.plot_breaking_point_dfm(d, base_font_size=8.5),
                height=4.4, title=f"Licks per light-on period — DFM {dfm_id}",
                caption=f"ΔLicks between successive light onsets after training end, one "
                        f"panel per chamber ({roles}groups excluded by the light QC "
                        f"included for reference).  Dashed: the break; grey: past it; "
                        f"hollow red: lick-free light events, never counted."))
        return blocks

    def execute_basic_analysis(
        self,
        *,
        data_breaks_multiplier: float = 4.0,
        bleeding_cutoff: float = 50.0,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool | None = None,
        plot_format: str = "png",
        dpi: int = 200,
        skip_qc: bool = False,
    ) -> dict[str, Path]:
        result = TwoWellExperiment.execute_basic_analysis(
            self,
            data_breaks_multiplier=data_breaks_multiplier,
            bleeding_cutoff=bleeding_cutoff,
            range_minutes=range_minutes,
            transform_licks=transform_licks,
            plot_format=plot_format,
            dpi=dpi,
            skip_qc=skip_qc,
        )
        print("\n[PR] Paired-yoked difference table...", flush=True)
        result["paired_yoked_diff"] = self.write_paired_yoked_diff(
            transform_licks=transform_licks)
        print(f"  Done → {result['paired_yoked_diff']}", flush=True)
        print("[PR] Cumulative difference curve data...", flush=True)
        result["pr_cumulative_diff_csv"] = self.write_cumulative_diff()
        print(f"  Done → {result['pr_cumulative_diff_csv']}", flush=True)
        print("[PR] Light QC tables...", flush=True)
        light = self.write_light_qc()
        result.update(light)
        print(f"  Done → {light['pr_light_qc']}", flush=True)
        information = self.training_information()
        if information:
            print(f"  {information}", flush=True)
        for line in self.light_qc_lines():
            print(f"  {line}", flush=True)
        print("[PR] Breaking point...", flush=True)
        result["pr_breaking_point"] = self.write_breaking_point()
        print(f"  Done → {result['pr_breaking_point']}", flush=True)
        print("[PR] Figures...", flush=True)
        figs = self.write_pr_figures(dpi=dpi)
        result.update(figs)
        print(f"  Done → {len(figs)} figure(s)", flush=True)
        return result


def restart_note(span, training_end: float) -> str:
    """What a group's light QC notes and the breaking point lines say of a
    DFM restart (a :class:`~pyflic.base.paradigm_state.RestartSpan`) inside
    its Test window (decision 15 of ``docs/paradigm-analyses-plan.md``)."""
    return (f"the DFM restarted at minute {span.minute:.1f}, "
            f"{span.minute - float(training_end):.1f} min after training end: the Test "
            f"window ends at the restart, because the ratio starts again after it, so a "
            f"breaking point still running there is censored")


def _chamber_column(values: Sequence[Any]) -> Any:
    """Chamber numbers as a table column: plain integers, or nullable ones
    with NA for a solo group's missing Yoked chamber when any is missing."""
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in values):
        return pd.array([None if v is None or pd.isna(v) else int(v) for v in values],
                        dtype="Int64")
    return [int(v) for v in values]


def _panel_figsize(n: int) -> tuple[float, float]:
    """The size of a per-DFM figure with one panel per Chamber Group, three
    panels to a row: six solo groups take two rows, not one 24-inch one."""
    columns, rows = min(max(int(n), 1), 3), -(-max(int(n), 1) // 3)
    return 4.0 * columns, 5.0 * rows


def _truthy(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(value)


def _flag(value) -> bool | None:
    """A boolean cell that may be missing — ``PersistACensored`` on a Training
    row — as ``None`` when missing, else its truth."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return _truthy(value)


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[i0, i1)`` of each run of True in *mask*, in order."""
    m = np.asarray(mask, dtype=bool)
    if not m.size:
        return []
    edges = np.flatnonzero(np.diff(np.concatenate(([False], m, [False])).astype(np.int8)))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def _interval_at(state: ParadigmState, row: int) -> str:
    """The Program Interval recorded on *row*, as text (``unknown`` for an
    instruction outside the Program)."""
    value = state.program_interval[int(row)]
    return "unknown" if pd.isna(value) else str(int(value))


def _new_paradigm(names, row: int) -> bool:
    """Whether *row* records a different Paradigm name from the row before
    (``NA`` counts as a name of its own), so the firmware reset every Well."""
    here, before = names[row], names[row - 1]
    if pd.isna(here) or pd.isna(before):
        return bool(pd.isna(here) != pd.isna(before))
    return str(here) != str(before)


def _split_notes(text: Any) -> list[str]:
    """A ``Notes`` cell back into its notes.  They are joined with ``"; "``,
    which :data:`pyflic.base.pr_light_qc.FIRMWARE_TREND_NOTE` also holds, so
    that one note is kept whole."""
    hold = "\x00"
    joined = str(text).replace(lqc.FIRMWARE_TREND_NOTE,
                               lqc.FIRMWARE_TREND_NOTE.replace("; ", hold))
    return [note.replace(hold, "; ") for note in joined.split("; ")]


def _count_text(value: Any) -> str:
    """A nullable count as text, ``NA`` when missing."""
    return "NA" if value is None or pd.isna(value) else str(int(value))


def _light_qc_verdict_text(flags: Any) -> str:
    """A group's light QC flags as one table cell whose first word carries the
    tone: ``ok``, ``warning: …`` or ``failed: …``."""
    names = [f for f in str(flags if flags is not None else "").split(", ")
             if f and f.lower() != "nan"]
    if not names:
        return "ok"
    failing = any(f in lqc.FAILING_FLAGS for f in names)
    return ("failed: " if failing else "warning: ") + ", ".join(names)
