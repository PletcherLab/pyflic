"""Optogenetic light QC: was the light where the licks were?

The DFM firmware switches an Optolid LED from its own reading of a Trigger
Well, every millisecond, during the run.  pyflic counts licks afterwards, from
the *baselined* signal.  The two should agree, and where they do not the light
is evidence about the sensor rather than the fly: a well whose signal drifts
up is "touched" continuously for the firmware and flat for pyflic, so the light
runs with no lick behind it.  The firmware's decay keeps an LED lit for a set
time after contact ends, so a short tail of light after the last lick is
normal; long stretches of light with no lick are not.

This module holds the arithmetic on plain arrays, for any Experiment Type; the
experiment assembles it per DFM (:mod:`pyflic.base.opto_light`).  Terms, fixed
in ``CONTEXT.md``:

* **Explained Light** is a lit sample with activity — a feeding lick or a
  tasting sample — in one of its Linkage Group's Trigger Wells, from the
  interval's decay plus a tolerance before it to the tolerance after it;
* the **Emulated Trigger** is the firmware's own test re-run on the recorded
  signal: raw minus the mean of the run's first ten seconds, against the
  interval's threshold.  Contact the Emulated Trigger sees and pyflic does not
  is the drifting-baseline signature.

The unit of judgement is the Linkage Group: its light is one circuit, lit when
any member is triggered.  Verdicts are broadcast to every chamber the group
touches.

A **Generation**-3 recording (DFM firmware 4.0.0) records the firmware's
decisions, so the QC reads them instead of reconstructing them
(``docs/v4-data-plan.md``, stream 3; ADR-0015's addendum):

* the rows are the schedule: each run of rows under one Program Interval and
  Cycle is a window, joined to the program's printed line and judged under
  the Paradigm its rows name;
* lit time is ``OnMs``, exact to the millisecond, and each Activation of a
  Trigger Well owns the light from its onset to the next onset or the light
  going out.  A fixed-length Activation (FixedInterval, ProgressiveRatio,
  FixedRatio in Main) is explained when its well had a lick or touch within
  the tolerance of its onset; closed-loop and Acclimation light, and light
  no Activation accounts for, keep the window rule above with the Hold;
* the Emulated Trigger is not run: an Activation or Firmware Event with no
  pyflic activity at its well is the drifting-baseline signature directly;
* five more checks read the file's own bookkeeping: broken yoking, an
  Acclimation count that is not the Paradigm's, Firmware Events that
  disagree with pyflic's, and lost, late or unknown-instruction records;
* each Paradigm is checked against its own rule, one Activation at a time,
  with ``Program.txt``'s settings (:mod:`pyflic.base.rule_checks`,
  ``docs/paradigm-analyses-plan.md`` stream S1): the light period's length
  against the Hold, ``ActivationMs`` or ``MaxOnMs``, the lockout, the
  Requirement series, NonFeeding's light after contact, each open-loop
  window, each Program Interval's start, and restarts.

Generations 1 and 2 are judged as before, from ``OptoCol1`` and the
program's clock-time schedule.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Mapping

import numpy as np
import pandas as pd

from . import activation_ledger as al
from . import opto_program as op
from . import opto_roles as orl
from . import paradigm_families as pfam
from . import rule_checks as rc
from .constant_fields import ConstantField
from .constant_fields import constant_problems as _constant_problems
from .paradigm_state import GENERATION_FLAG, GENERATION_PARADIGM, IDLE, ParadigmState
from .pr_light_qc import VERDICT_FAILED, VERDICT_OK, VERDICT_WARNING, light_events

#: Design ``constants:`` read by the optogenetic light QC, with their defaults.
#: Shared by every Experiment Type and never written into a new config: the
#: editors show them as placeholders (ADR-0011) and the QC merges them under
#: whatever the design states.
DEFAULT_CONSTANTS: dict[str, Any] = {
    "exclude_failed_opto_chambers": False,
    "opto_unexplained_warn_fraction": 0.10,
    "opto_unexplained_fail_fraction": 0.30,
    "opto_unexplained_min_sec": 30,
    "opto_default_decay_ms": 1000,
    "opto_decay_tolerance_samples": 2,
    "opto_unlit_feeding_fraction": 0.50,
    "opto_open_loop_min_lit_fraction": 0.95,
    ## Generation 3 only (decision 21 of docs/v4-data-plan.md); none has data
    ## behind it yet, and the bench checkpoint revisits all five.
    "opto_event_disagreement_warn_fraction": 0.25,
    "opto_yoking_fail_fraction": 0.01,
    "opto_lost_records_warn": 25,
    "opto_unknown_rows_warn_fraction": 0.01,
    "opto_late_records_warn": 10,
    ## The rule checks of each Paradigm (docs/paradigm-analyses-plan.md,
    ## decision 7), generation 3 only, calibrated on the simulator's and
    ## PyMCU's simulated recordings; the bench recordings revisit them.
    "opto_activation_tolerance_ms": 25,
    "opto_activation_length_fail_fraction": 0.05,
    "opto_lockout_fail_fraction": 0.02,
    "opto_missed_contact_warn_fraction": 0.25,
    "opto_non_feeding_lit_warn_fraction": 0.10,
    "opto_non_feeding_min_sec": 30,
    "opto_non_feeding_dark_warn_fraction": 0.10,
    "opto_interval_lag_warn_sec": 5,
}

#: The editors' and the validation's description of those constants.
OPTO_CONSTANT_FIELDS: tuple[ConstantField, ...] = (
    ConstantField(
        "exclude_failed_opto_chambers", "Exclude chambers that fail the opto light QC",
        "exclude_failed_opto_chambers — every chamber of a linkage group whose light "
        "ran without the licks to explain it leaves the analysis through auto-removal.  "
        "Off (the default) keeps them and flags them: a faulty light does not "
        "invalidate the feeding record.  qc/opto/opto_light_qc.csv lists them either "
        "way.",
        "switch", short_label="Opto light QC failed",
        choices=("exclude the chambers", "keep and flag")),
    ConstantField(
        "opto_unexplained_warn_fraction", "Unexplained light, warning (fraction)",
        "opto_unexplained_warn_fraction — a linkage group whose lit time is at least "
        "this fraction unexplained (no trigger-well lick or touch within the decay "
        "window) gets a warning.  Default 0.10.",
        "opto", minimum=0, maximum=1),
    ConstantField(
        "opto_unexplained_fail_fraction", "Unexplained light, failure (fraction)",
        "opto_unexplained_fail_fraction — at least this fraction of a linkage group's "
        "lit time unexplained fails the group.  Default 0.30.",
        "opto", minimum=0, maximum=1),
    ConstantField(
        "opto_unexplained_min_sec", "Unexplained light at least (s)",
        "opto_unexplained_min_sec — neither fraction is judged until a group has at "
        "least this many seconds of unexplained light, so a few stray samples in a "
        "run lit for seconds are not called a fault.  Default 30; 0 judges any.",
        "opto", minimum=0),
    ConstantField(
        "opto_default_decay_ms", "Decay without Program.txt (ms)",
        "opto_default_decay_ms — the light decay assumed when data/ holds no "
        "Program.txt stating it.  Default 1000.",
        "opto", integer=True, minimum=0),
    ConstantField(
        "opto_decay_tolerance_samples", "Decay tolerance (samples)",
        "opto_decay_tolerance_samples — light within the decay plus this many "
        "samples of a lick still counts as explained; it absorbs the difference "
        "between the firmware's 1 ms decisions and the recorded samples.  Default 2.",
        "opto", integer=True, minimum=0),
    ConstantField(
        "opto_unlit_feeding_fraction", "Unlit feeding, warning (fraction)",
        "opto_unlit_feeding_fraction — in a closed-loop interval every feeding bout "
        "at a trigger well should light its group; when at least this fraction do "
        "not, the group gets a warning (a dead LED, a loose lid, or a threshold the "
        "fly never crossed).  Default 0.50.",
        "opto", minimum=0, maximum=1, minimum_exclusive=True),
    ConstantField(
        "opto_open_loop_min_lit_fraction", "Open loop lit at least (fraction)",
        "opto_open_loop_min_lit_fraction — an open-loop interval (threshold 0) should "
        "be lit throughout; lit for less than this fraction of it is a warning.  "
        "Default 0.95.",
        "opto", minimum=0, maximum=1),
    ConstantField(
        "opto_event_disagreement_warn_fraction", "Firmware events disagree, warning (fraction)",
        "opto_event_disagreement_warn_fraction — firmware 4.0.0 only: under a ratio "
        "Paradigm's Main phase the DFM counts its own Firmware Events; a trigger well "
        "whose count differs from pyflic's feeding events by more than this fraction "
        "of the Firmware Events gets a warning.  Like the firmware, pyflic's count "
        "leaves out bouts that start while the well's own Activation is lit.  "
        "Default 0.25.",
        "opto", minimum=0),
    ConstantField(
        "opto_yoking_fail_fraction", "Broken yoking, failure (fraction)",
        "opto_yoking_fail_fraction — firmware 4.0.0 only: linked wells share one "
        "light, so their OnMs match; a linkage group whose wells' OnMs differ by more "
        "than 1 ms in more than this fraction of its lit records fails (as does an "
        "Off well linked to a trigger that counts Activations).  Default 0.01.",
        "opto", minimum=0, maximum=1),
    ConstantField(
        "opto_lost_records_warn", "Lost records, warning (records)",
        "opto_lost_records_warn — firmware 4.0.0 only: a DFM missing more than this "
        "many records from its Index gets a warning (25 records are 5 s at 5 Hz).  "
        "Default 25.",
        "opto", integer=True, minimum=0),
    ConstantField(
        "opto_unknown_rows_warn_fraction", "Unknown-instruction rows, warning (fraction)",
        "opto_unknown_rows_warn_fraction — firmware 4.0.0 only: rows recorded under "
        "an instruction outside the program (Interval NA), after the lead-in, "
        "are not judged; more than this fraction of them warns.  Default 0.01.",
        "opto", minimum=0, maximum=1),
    ConstantField(
        "opto_late_records_warn", "Late records, warning (records)",
        "opto_late_records_warn — firmware 4.0.0 only: a record whose OnMs exceeds "
        "200 ms came late; more than this many late records on a DFM warns.  "
        "Default 10.",
        "opto", integer=True, minimum=0),
    ## The rule checks (firmware 4.0.0): each Paradigm against its own rule.
    ConstantField(
        "opto_activation_tolerance_ms", "Activation timing tolerance (ms)",
        "opto_activation_tolerance_ms — firmware 4.0.0 only: how far a light period may "
        "differ from its Paradigm's Hold, ActivationMs or MaxOnMs, or two onsets from its "
        "lockout, and still keep the rule (a period timed only to the record gets one "
        "record more).  Default 25.",
        "opto", integer=True, minimum=0),
    ConstantField(
        "opto_activation_length_fail_fraction", "Wrong or short Activations, failure (fraction)",
        "opto_activation_length_fail_fraction — firmware 4.0.0 only: a linkage group fails "
        "when more than this fraction of its judged Activations break their length rule: "
        "a fixed-length one not lasting ActivationMs (wrong activation length), a "
        "closed-loop one lit for less than its Hold (short activations).  Default 0.05.",
        "opto", minimum=0, maximum=1),
    ConstantField(
        "opto_lockout_fail_fraction", "Lockout broken, failure (fraction)",
        "opto_lockout_fail_fraction — firmware 4.0.0 only: a FixedInterval linkage group "
        "fails when more than this fraction of its judged Main Activations started sooner "
        "than the lockout (the longer of LockoutMs and ActivationMs) after the one "
        "before.  Default 0.02.",
        "opto", minimum=0, maximum=1),
    ConstantField(
        "opto_missed_contact_warn_fraction", "Contact without activation, warning (fraction)",
        "opto_missed_contact_warn_fraction — firmware 4.0.0 only: under FixedInterval, a "
        "run of pyflic licks or touches that begins with the well free should light it; "
        "when more than this fraction of at least 5 such runs do not, the group gets a "
        "warning.  Default 0.25.",
        "opto", minimum=0, maximum=1),
    ConstantField(
        "opto_non_feeding_lit_warn_fraction", "Non-feeding light during feeding, warning (fraction)",
        "opto_non_feeding_lit_warn_fraction — firmware 4.0.0 only: a NonFeeding light goes "
        "out while the fly feeds; lit for more than this fraction of the feeding at its "
        "trigger wells, the group gets a warning.  Default 0.10.",
        "opto", minimum=0, maximum=1),
    ConstantField(
        "opto_non_feeding_min_sec", "Non-feeding light during feeding at least (s)",
        "opto_non_feeding_min_sec — firmware 4.0.0 only: non-feeding light during feeding "
        "is not judged until a group has at least this many seconds of it.  Default 30; "
        "0 judges any.",
        "opto", minimum=0),
    ConstantField(
        "opto_non_feeding_dark_warn_fraction", "Non-feeding dark, warning (fraction)",
        "opto_non_feeding_dark_warn_fraction — firmware 4.0.0 only: a NonFeeding light is "
        "on OnDelayMs after the fly's last contact (for at most MaxOnMs); dark for more "
        "than this fraction of that time, the group gets a warning (contact the firmware "
        "read that pyflic did not, such as a drifting baseline, or a longer OnDelayMs "
        "than Program.txt states).  Default 0.10.",
        "opto", minimum=0, maximum=1),
    ConstantField(
        "opto_interval_lag_warn_sec", "Interval started late, warning (s)",
        "opto_interval_lag_warn_sec — firmware 4.0.0 only: a Program Interval whose first "
        "row comes more than this many seconds after its Program.txt time (on any Cycle) "
        "warns its DFM; the firmware's own lag is one record.  Default 5.",
        "opto", minimum=0),
)

# ---------------------------------------------------------------------------
# Flags and verdicts
# ---------------------------------------------------------------------------

UNEXPLAINED = "unexplained light"
PARTLY_UNEXPLAINED = "partly unexplained light"
UNEXPLAINED_NO_PROGRAM = "unexplained light (no Program.txt)"
LIGHT_WHILE_OFF = "light while off"
NO_LIGHT = "no light recorded"
NO_LIGHT_EVENTS = "no light events"
UNLIT_FEEDING = "unlit feeding"
OPEN_LOOP_DARK = "open loop not lit"
PROGRAM_MISMATCH = "program mismatch"
NO_PROGRAM_SECTION = "no program section"
## Generation 3 (decision 20 of docs/v4-data-plan.md).
YOKING_BROKEN = "broken yoking"
PARADIGM_NOT_IN_PROGRAM = "paradigm not in program"
ACCLIMATION_COUNT_MISMATCH = "acclimation count mismatch"
EVENT_DISAGREEMENT = "firmware events disagree"
LOST_RECORDS = "lost records"
UNKNOWN_INSTRUCTION_ROWS = "unknown-instruction rows"
LATE_RECORDS = "late records"
PROGRAM_MISSING = "no Program.txt for firmware 4.0.0"
NON_FEEDING_LIT_WHILE_FEEDING = "non-feeding light during feeding"
## The rule checks (docs/paradigm-analyses-plan.md, stream S1; rule_checks).
SHORT_ACTIVATIONS = rc.SHORT_ACTIVATIONS
MAX_ON_EXCEEDED = rc.MAX_ON_EXCEEDED
WRONG_ACTIVATION_LENGTH = rc.WRONG_ACTIVATION_LENGTH
LOCKOUT_BROKEN = rc.LOCKOUT_BROKEN
CONTACT_WITHOUT_ACTIVATION = rc.CONTACT_WITHOUT_ACTIVATION
REQUIREMENT_SERIES_WRONG = rc.REQUIREMENT_SERIES_WRONG
NON_FEEDING_DARK = rc.NON_FEEDING_DARK
INTERVAL_STARTED_LATE = rc.INTERVAL_STARTED_LATE
DFM_RESTARTED = rc.DFM_RESTARTED
#: The series check, for the Progressive Ratio light QC (stream S4).
requirement_series_mismatches = rc.requirement_series_mismatches

#: Flags that fail a Linkage Group.  The rest are warnings: they point at the
#: setup or the program rather than proving the light was wrong.  A broken
#: yoking does prove it (a Yoked fly did not get its partner's light), and a
#: row naming a Paradigm the program lacks proves the program is the wrong
#: file, so nothing judged against it can stand.  The rule checks that fail
#: do so because the firmware keeps each rule exactly (decision 6).
FAILING_FLAGS: tuple[str, ...] = (UNEXPLAINED, LIGHT_WHILE_OFF, NO_LIGHT, YOKING_BROKEN,
                                  PARADIGM_NOT_IN_PROGRAM, *rc.FAILING_RULE_FLAGS)
WARNING_FLAGS: tuple[str, ...] = (PARTLY_UNEXPLAINED, UNEXPLAINED_NO_PROGRAM,
                                  NO_LIGHT_EVENTS, UNLIT_FEEDING, OPEN_LOOP_DARK,
                                  PROGRAM_MISMATCH, NO_PROGRAM_SECTION,
                                  ACCLIMATION_COUNT_MISMATCH, EVENT_DISAGREEMENT,
                                  LOST_RECORDS, UNKNOWN_INSTRUCTION_ROWS, LATE_RECORDS,
                                  PROGRAM_MISSING, NON_FEEDING_LIT_WHILE_FEEDING,
                                  *rc.WARNING_RULE_FLAGS)

#: Where a DFM's program came from.
PROGRAM_OK = "Program.txt"
PROGRAM_NO_FILE = "no Program.txt"
PROGRAM_NO_SECTION = "no section"
PROGRAM_DROPPED = "section unreadable"
#: ``Program.txt`` is the ``***ALL DFM***`` export of a recording made
#: without a Program: every well ``Off`` throughout.
PROGRAM_ALL_OFF = "no Program: every well Off"

#: Generation-3 window modes the light does not get judged under: rows
#: recorded under an instruction outside the program, and a Program Interval
#: whose Paradigm or thresholds the QC cannot know.
UNKNOWN_INSTRUCTION = "unknown instruction"
NOT_JUDGED = "not judged"
#: The Paradigm types whose Main Activation lasts a fixed time once earned.
FIXED_LENGTH_PARADIGMS: tuple[str, ...] = (op.FIXED_INTERVAL, op.PROGRESSIVE_RATIO,
                                           op.FIXED_RATIO)

#: Where lit time came from (``LitSource``) and what a light event is
#: (``Source``): ``OnMs`` and Activations on generation 3, ``OptoCol1`` and
#: its lit runs before.
LIT_SOURCE_ON_MS = "OnMs"
LIT_SOURCE_OPTOCOL1 = "OptoCol1"
EVENT_SOURCE_ACTIVATIONS = "activations"
EVENT_SOURCE_OPTOCOL1 = "optocol1"

#: Minutes per window of the unexplained-light onset (as the PR Resting Level).
WINDOW_MIN = 30
#: Feeding bouts a closed-loop group needs before its unlit share is judged.
MIN_FEEDING_EVENTS = 5
#: The firmware's baseline: the mean of the run's first ten seconds.
BASELINE_SECONDS = 10.0
#: A data start this far from the program's Start Time is a mismatch.
START_TOLERANCE_SEC = 60.0
#: Data starting later than this after the program leaves the firmware's
#: baseline window unrecorded, so the Emulated Trigger is approximate.
BASELINE_ALIGN_SEC = 2.0

CAUSE_DRIFT = "signal above the firmware threshold with no pyflic activity: a " \
              "drifting baseline or sustained contact"
CAUSE_HARDWARE = "light without firmware-visible contact: hardware, linkage or " \
                 "program mismatch"
#: :data:`CAUSE_DRIFT` split in two where the data have PyMCU's Baseline
#: columns, so the Baseline in force is known: either the well rested at
#: that baseline and the signal really stayed up (a fly standing on the food,
#: which pyflic's running median absorbs), or the well's resting level had
#: drifted :data:`BASELINE_DRIFT_COUNTS` or more above it.
CAUSE_SUSTAINED_CONTACT = "signal above the firmware threshold with no pyflic activity, " \
                          "the well resting at the baseline in force: sustained contact"
CAUSE_DRIFTING_BASELINE = "signal above the firmware threshold with no pyflic activity, " \
                          "the well resting 10 or more above the baseline in force: a " \
                          "drifting baseline"
#: The causes that mean the firmware saw contact that pyflic did not.
CONTACT_CAUSES = (CAUSE_DRIFT, CAUSE_SUSTAINED_CONTACT, CAUSE_DRIFTING_BASELINE)
#: PyMCU's update step: a resting level this far above the Baseline in force
#: is drift PyMCU had not yet followed (``PyMCU/docs/adaptive-baseline-plan.md``).
BASELINE_DRIFT_COUNTS = 10
#: The resting level the Baseline in force is compared with: the 10th
#: percentile of a well's signal in each minute, as PyMCU's Floor.
FLOOR_QUANTILE = 0.1

#: ``qc/opto/opto_light_qc.csv`` — one row per Linkage Group per DFM.
#:
#: On generation 3 ``LitSec`` and the other light seconds are ``OnMs``
#: sums; ``LightEvents`` counts the Trigger Wells' Activations;
#: ``ContactSec`` is the lit time of those Activations (the firmware's
#: contact decisions) and ``ContactWithoutActivitySec`` the lit time of the
#: ones whose onset had no pyflic lick or touch at their well;
#: ``FirmwareEventsWithoutActivity`` counts rises in Progress with none
#: either.  ``Roles`` names each chamber's Role (``opto_roles``);
#: ``FirmwareEvents`` / ``PyflicEvents`` compare the two event counts under a
#: ratio Paradigm's Main phase, like with like: pyflic's leaves out feeding
#: events that start within the well's own Activation, after its onset row,
#: as the firmware starts no Event while its light is on;
#: ``AcclimationActivations`` / ``AcclimationExpected`` sum the Trigger Wells'
#: completed Acclimation runs.
#: The rule checks' columns (:data:`rule_checks.RULE_QC_COLUMNS`, declared
#: per family in :mod:`paradigm_families`) follow: each counts the judgeable
#: Activations that broke their Paradigm's rule (``CutOffs`` those its
#: ``MaxOnMs`` ended), ``NonFeedingDarkSec`` is seconds dark where the rule
#: says lit, ``RelightDelayMs`` a median, ``IntervalLagSec`` the DFM's worst
#: Program Interval start and ``Restarts`` its restarts; each is ``NA``
#: outside its family and without ``Program.txt`` (``Restarts`` needs none).
#: On generation 3 ``OpenLoopLitFraction`` is the lowest over the group's
#: open-loop windows.
#: These are ``NA`` on generations 1 and 2, as is ``EmulationApproximate`` on
#: generation 3, where nothing is emulated.
OPTO_QC_COLUMNS: tuple[str, ...] = (
    "DFM", "Generation", "Group", "Linkage", "Wells", "TriggerWells", "Chambers", "Roles",
    "Treatment", "Program", "Paradigm", "LitSec", "JudgedLitSec", "UnexplainedSec",
    "UnexplainedFraction", "UnexplainedOnsetMin", "LightEvents", "UnexplainedEvents",
    "FeedingEvents", "ClosedLoopFeedingEvents", "UnlitFeedingEvents",
    "FirmwareEvents", "PyflicEvents", "AcclimationActivations", "AcclimationExpected",
    "OpenLoopLitFraction", "LitWhileOffSec",
    "ContactSec", "ContactWithoutActivitySec", "FirmwareEventsWithoutActivity",
    "UnexplainedWithContactSec", *rc.RULE_QC_COLUMNS,
    "EmulationApproximate", "LikelyCause", "Flags", "Verdict", "Excluded", "Notes",
)
#: The QC table's whole-number rule-check columns, held as nullable integers.
OPTO_QC_INTEGER_COLUMNS: tuple[str, ...] = rc.RULE_QC_INTEGER_COLUMNS
#: ``qc/opto/opto_light_intervals.csv`` — one row per group per window: a
#: scheduled interval (generations 1 and 2) or a run of rows under one
#: Program Interval and Cycle (generation 3).  ``Interval`` is the program's
#: printed line, ``ProgramInterval`` / ``Cycle`` / ``Paradigm`` /
#: ``ParadigmType`` what the rows recorded (``NA`` before generation 3);
#: ``HoldMs`` is the Hold the window was judged with (the decay, before).
OPTO_INTERVAL_COLUMNS: tuple[str, ...] = (
    "DFM", "Group", "Window", "Interval", "ProgramInterval", "Cycle", "Occurrence",
    "StartMin", "EndMin", "Paradigm", "ParadigmType", "Mode", "HoldMs", "TriggerWells",
    "Samples", "LitSource", "LitSec", "LitFraction", "UnexplainedSec",
    "ContactSec", "ContactWithoutActivitySec", "FeedingEvents", "UnlitFeedingEvents",
)
#: ``qc/opto/opto_light_events.csv`` — one row per light event: a lit run of
#: ``OptoCol1`` (``Source`` ``optocol1``) or, on generation 3, one
#: Activation of a Trigger Well (``activations``), with its ``Well``,
#: ``Phase``, the Requirement it met and the Firmware Events behind it
#: (``NA`` for an Acclimation Activation, which no Firmware Event earns).
#: The generation-3 rows also carry the Activation ledger's columns
#: (:data:`LEDGER_EVENT_COLUMNS`, :mod:`pyflic.base.activation_ledger`), the
#: ``Family`` of their ``Mode`` and ``Phase``
#: (:func:`paradigm_families.family_of`) and ``EndedBy``, what ended the
#: Activation (:func:`rule_checks.ended_by`); ``OptoCol1`` rows have them
#: ``NA``.
LEDGER_EVENT_COLUMNS: tuple[str, ...] = (
    "OnsetMs", "EndMs", "LitMs", "Timing", "Overlap", "Boundary", "SinceLastOnsetMs", "K",
)
OPTO_EVENT_COLUMNS: tuple[str, ...] = (
    "DFM", "Group", "Event", "Well", "OnsetMin", "EndMin", "DurationSec", "Interval",
    "Mode", "Phase", "RequirementMet", "FirmwareEvents", "UnexplainedSec", "Explained",
    "OverrunSec", "Source", *LEDGER_EVENT_COLUMNS, "Family", "EndedBy",
)
#: The light-events table's whole-number ledger columns, held as nullable
#: integers (``NA`` on the ``OptoCol1`` rows, and where the ledger has none).
OPTO_EVENT_INTEGER_COLUMNS: tuple[str, ...] = (
    "OnsetMs", "EndMs", "LitMs", "SinceLastOnsetMs", "K",
)
#: The ledger's columns on a light event the ledger does not time.
_NO_LEDGER: dict[str, Any] = {c: np.nan for c in (*LEDGER_EVENT_COLUMNS, "Family", "EndedBy")}
#: ``qc/opto/opto_program.csv`` — one row per DFM interval of the program.
#: ``Paradigm`` is the Paradigm's name and ``ParadigmType`` PyMCU's spelling of
#: its type; the settings columns are the Paradigm's, ``NA`` where its type has
#: no such setting.  ``LegacyF`` and ``LegacyP`` are the raw frequency and
#: pulse width a pre-4.0.0 export echoed (``NA`` for the 4.0.0 grammar).
OPTO_PROGRAM_COLUMNS: tuple[str, ...] = (
    "DFM", "Interval", "Start", "DurationMin", "ProgramType", "Paradigm",
    "ParadigmType", "Generated", "Dark", "FrequencyHz", "PulseWidthMs", "HoldMs",
    "MaxOnMs", "ActivationMs", "LockoutMs", "OnDelayMs", "EventGapMs", "Acclimation",
    "Series", "FirstRequirement", "Step", "Scale", "Rate", "Cap", "Ratio", "Linkage",
    "TriggerWells", "OpenLoopWells", "LegacyF", "LegacyP",
)
#: The program table's whole-number columns, held as nullable integers.
OPTO_PROGRAM_INTEGER_COLUMNS: tuple[str, ...] = (
    "DFM", "Interval", "FrequencyHz", "PulseWidthMs", "HoldMs", "MaxOnMs",
    "ActivationMs", "LockoutMs", "OnDelayMs", "EventGapMs", "Acclimation",
    "FirstRequirement", "Step", "Cap", "Ratio", "LegacyF", "LegacyP",
)
#: The Paradigm fields behind the program table's settings columns.
_PROGRAM_SETTING_COLUMNS: tuple[tuple[str, str], ...] = (
    ("HoldMs", "hold_ms"), ("MaxOnMs", "max_on_ms"), ("ActivationMs", "activation_ms"),
    ("LockoutMs", "lockout_ms"), ("OnDelayMs", "on_delay_ms"),
    ("EventGapMs", "event_gap_ms"), ("Series", "series"),
    ("FirstRequirement", "first_requirement"), ("Step", "step"), ("Scale", "scale"),
    ("Rate", "rate"), ("Cap", "cap"), ("Ratio", "ratio"),
)


def _truthy(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(value)


@dataclass(frozen=True, slots=True)
class OptoQCSettings:
    """The optogenetic light QC's thresholds, from the design's ``constants:``."""

    exclude: bool = False
    warn_fraction: float = 0.10
    fail_fraction: float = 0.30
    min_unexplained_sec: float = 30.0
    default_decay_ms: int = 1000
    tolerance_samples: int = 2
    unlit_fraction: float = 0.50
    open_loop_min_fraction: float = 0.95
    event_disagreement_fraction: float = 0.25
    yoking_fail_fraction: float = 0.01
    lost_records_warn: int = 25
    unknown_rows_fraction: float = 0.01
    late_records_warn: int = 10
    activation_tolerance_ms: int = 25
    activation_length_fail_fraction: float = 0.05
    lockout_fail_fraction: float = 0.02
    missed_contact_fraction: float = 0.25
    non_feeding_lit_fraction: float = 0.10
    non_feeding_min_sec: float = 30.0
    non_feeding_dark_fraction: float = 0.10
    interval_lag_warn_sec: float = 5.0

    @classmethod
    def from_constants(cls, constants: Mapping[str, Any] | None) -> OptoQCSettings:
        merged = {**DEFAULT_CONSTANTS, **dict(constants or {})}

        def number(key: str) -> float:
            try:
                value = float(merged[key])
            except (TypeError, ValueError):
                return float(DEFAULT_CONSTANTS[key])
            return value if math.isfinite(value) else float(DEFAULT_CONSTANTS[key])

        def fraction(key: str) -> float:
            return min(max(number(key), 0.0), 1.0)

        return cls(
            exclude=_truthy(merged["exclude_failed_opto_chambers"]),
            warn_fraction=fraction("opto_unexplained_warn_fraction"),
            fail_fraction=fraction("opto_unexplained_fail_fraction"),
            min_unexplained_sec=max(0.0, number("opto_unexplained_min_sec")),
            default_decay_ms=max(0, int(number("opto_default_decay_ms"))),
            tolerance_samples=max(0, int(number("opto_decay_tolerance_samples"))),
            unlit_fraction=fraction("opto_unlit_feeding_fraction"),
            open_loop_min_fraction=fraction("opto_open_loop_min_lit_fraction"),
            event_disagreement_fraction=max(
                0.0, number("opto_event_disagreement_warn_fraction")),
            yoking_fail_fraction=fraction("opto_yoking_fail_fraction"),
            lost_records_warn=max(0, int(number("opto_lost_records_warn"))),
            unknown_rows_fraction=fraction("opto_unknown_rows_warn_fraction"),
            late_records_warn=max(0, int(number("opto_late_records_warn"))),
            activation_tolerance_ms=max(0, int(number("opto_activation_tolerance_ms"))),
            activation_length_fail_fraction=fraction("opto_activation_length_fail_fraction"),
            lockout_fail_fraction=fraction("opto_lockout_fail_fraction"),
            missed_contact_fraction=fraction("opto_missed_contact_warn_fraction"),
            non_feeding_lit_fraction=fraction("opto_non_feeding_lit_warn_fraction"),
            non_feeding_min_sec=max(0.0, number("opto_non_feeding_min_sec")),
            non_feeding_dark_fraction=fraction("opto_non_feeding_dark_warn_fraction"),
            interval_lag_warn_sec=max(0.0, number("opto_interval_lag_warn_sec")),
        )

    def describe(self) -> str:
        return (f"exclude_failed_opto_chambers={'true' if self.exclude else 'false'}, "
                f"opto_unexplained_warn_fraction={self.warn_fraction:g}, "
                f"opto_unexplained_fail_fraction={self.fail_fraction:g}, "
                f"opto_unexplained_min_sec={self.min_unexplained_sec:g}, "
                f"opto_default_decay_ms={self.default_decay_ms}, "
                f"opto_decay_tolerance_samples={self.tolerance_samples}, "
                f"opto_unlit_feeding_fraction={self.unlit_fraction:g}, "
                f"opto_open_loop_min_lit_fraction={self.open_loop_min_fraction:g}")

    def describe_firmware4(self) -> str:
        """The generation-3 checks' constants, for a summary that has a
        firmware 4.0.0 recording to judge with them."""
        return (f"opto_event_disagreement_warn_fraction={self.event_disagreement_fraction:g}, "
                f"opto_yoking_fail_fraction={self.yoking_fail_fraction:g}, "
                f"opto_lost_records_warn={self.lost_records_warn}, "
                f"opto_unknown_rows_warn_fraction={self.unknown_rows_fraction:g}, "
                f"opto_late_records_warn={self.late_records_warn}, "
                f"opto_activation_tolerance_ms={self.activation_tolerance_ms}, "
                f"opto_activation_length_fail_fraction="
                f"{self.activation_length_fail_fraction:g}, "
                f"opto_lockout_fail_fraction={self.lockout_fail_fraction:g}, "
                f"opto_missed_contact_warn_fraction={self.missed_contact_fraction:g}, "
                f"opto_non_feeding_lit_warn_fraction={self.non_feeding_lit_fraction:g}, "
                f"opto_non_feeding_min_sec={self.non_feeding_min_sec:g}, "
                f"opto_non_feeding_dark_warn_fraction={self.non_feeding_dark_fraction:g}, "
                f"opto_interval_lag_warn_sec={self.interval_lag_warn_sec:g}")


def opto_constant_problems(constants: Mapping[str, Any] | None) -> list[str]:
    """What is wrong with the optogenetic light QC constants a ``constants:``
    block states (:data:`OPTO_CONSTANT_FIELDS`), plus a warning fraction above
    the failure one.  Never raises."""
    problems = _constant_problems(OPTO_CONSTANT_FIELDS, constants)
    stated = dict(constants or {})
    warn = stated.get("opto_unexplained_warn_fraction")
    fail = stated.get("opto_unexplained_fail_fraction")
    numeric = (int, float)
    if isinstance(warn, numeric) and isinstance(fail, numeric) \
            and not isinstance(warn, bool) and not isinstance(fail, bool) \
            and float(warn) > float(fail):
        problems.append("'constants.opto_unexplained_warn_fraction' must not exceed "
                        "'constants.opto_unexplained_fail_fraction'")
    return problems


def verdict_of(flags) -> str:
    """``ok``, ``warning`` or ``failed`` for a list of flags."""
    flags = [f for f in flags if f]
    if any(f in FAILING_FLAGS for f in flags):
        return VERDICT_FAILED
    return VERDICT_WARNING if flags else VERDICT_OK


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------

def near(mask: np.ndarray, before: int, after: int) -> np.ndarray:
    """``out[t]``: whether *mask* is true anywhere in ``[t - before, t + after]``."""
    m = np.asarray(mask, dtype=bool)
    n = m.size
    if n == 0:
        return m.copy()
    cum = np.concatenate([[0], np.cumsum(m, dtype=np.int64)])
    idx = np.arange(n)
    lo = np.clip(idx - int(before), 0, n)
    hi = np.clip(idx + int(after) + 1, 0, n)
    return (cum[hi] - cum[lo]) > 0


def explained_light(lit: np.ndarray, activity: np.ndarray, decay_samples: int,
                    tolerance: int) -> np.ndarray:
    """Lit samples with *activity* from ``decay + tolerance`` samples before to
    *tolerance* samples after them — Explained Light."""
    lit = np.asarray(lit, dtype=bool)
    return lit & near(activity, int(decay_samples) + int(tolerance), int(tolerance))


def decay_samples_for(decay_ms: float, samples_per_second: float) -> int:
    """The decay in whole samples, rounded up: a 500 ms decay at 5 Hz can keep
    the LED lit into a third sample."""
    return int(math.ceil(max(0.0, float(decay_ms)) * float(samples_per_second) / 1000.0
                         - 1e-9))


def carried_over(lit: np.ndarray, limit: int) -> int:
    """How many rows of light were already on at row 0, up to *limit* (a
    Hold plus the tolerance): light that came on before the rows analysed —
    the start of a Slice, or of the recording — whose onset and cause they
    do not hold.  A light on for longer than that is judged from there on, so
    a stuck light is still seen."""
    lit = np.asarray(lit, dtype=bool)
    if not lit.size or not lit[0]:
        return 0
    head = lit[:max(0, int(limit))]
    dark = np.flatnonzero(~head)
    return int(dark[0]) if dark.size else int(head.size)


def event_spans(column: np.ndarray) -> list[tuple[int, int]]:
    """``[(start, end)]`` sample spans of pyflic's events in one ``event_df``
    column, which holds each event's length at its first sample."""
    col = np.asarray(column)
    starts = np.flatnonzero(col > 0)
    return [(int(s), int(s + col[s])) for s in starts]


def firmware_baseline(raw: np.ndarray, start: int, samples: int) -> float:
    """The mean of *samples* raw values from *start* — the firmware's baseline."""
    raw = np.asarray(raw, dtype=float)
    seg = raw[max(0, int(start)):max(0, int(start)) + max(1, int(samples))]
    seg = seg[np.isfinite(seg)]
    return float(seg.mean()) if seg.size else 0.0


def resting_floor(minutes: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Each row's well Floors, ``(n, 12)``: the 10th percentile of the well's
    signal over the recording minute the row is in."""
    raw = np.asarray(raw, dtype=float)
    key = np.floor(np.asarray(minutes, dtype=float))
    frame = pd.DataFrame(raw)
    floors = frame.groupby(key).quantile(FLOOR_QUANTILE)
    return floors.reindex(key).to_numpy(dtype=float)


def baseline_drift_rows(recorded: np.ndarray, floor: np.ndarray,
                        wells: tuple[int, ...] | set[int]) -> np.ndarray:
    """``(n,)`` bool: rows where any of *wells* (1-12) rested
    :data:`BASELINE_DRIFT_COUNTS` or more above its Baseline in force."""
    out = np.zeros(len(recorded), dtype=bool)
    for w in wells:
        gap = floor[:, w - 1] - recorded[:, w - 1]
        out |= np.nan_to_num(gap, nan=-np.inf) >= BASELINE_DRIFT_COUNTS
    return out


def contact_cause(weights: np.ndarray, evidence: np.ndarray,
                  drift_rows: np.ndarray | None) -> str:
    """Which of the contact causes the light with firmware-visible contact
    and no pyflic activity (*evidence*, weighted by *weights*) points to:
    :data:`CAUSE_DRIFT` without the Baseline columns, otherwise
    :data:`CAUSE_DRIFTING_BASELINE` when most of it fell where a trigger well
    rested well above the Baseline in force, and
    :data:`CAUSE_SUSTAINED_CONTACT` when it did not."""
    if drift_rows is None:
        return CAUSE_DRIFT
    weights = np.asarray(weights, dtype=float)
    total = float(weights[evidence].sum())
    drifted = float(weights[evidence & drift_rows].sum())
    return CAUSE_DRIFTING_BASELINE if total and drifted >= 0.5 * total \
        else CAUSE_SUSTAINED_CONTACT


#: ``qc/opto/opto_baselines.csv``: one row per well of every DFM whose data
#: have PyMCU's Baseline columns.
OPTO_BASELINE_COLUMNS: tuple[str, ...] = (
    "DFM", "Well", "BaselineStart", "BaselineEnd", "BaselineMin", "BaselineMax",
    "BaselineMaxMove", "Updates", "CapCounts", "ReachedCap")


def baseline_rows(dfm_id: int, recorded: np.ndarray | None,
                  program: op.OptoProgram | None) -> list[dict]:
    """How each well's Baseline in force moved over the rows that have one:
    its first, last, lowest and highest value, its largest move from the
    first, how many times it changed, and whether it reached the Program's
    ``BaselineCapCounts`` (``Baseline: Adaptive`` only)."""
    if recorded is None:
        return []
    cap = program.baseline_cap_counts if program is not None and program.adaptive_baseline else None
    rows = []
    for w in range(1, op.N_WELLS + 1):
        values = recorded[:, w - 1]
        values = values[np.isfinite(values)]
        if not values.size:
            continue
        move = float(np.max(np.abs(values - values[0])))
        rows.append({"DFM": int(dfm_id), "Well": w, "BaselineStart": float(values[0]),
                     "BaselineEnd": float(values[-1]), "BaselineMin": float(values.min()),
                     "BaselineMax": float(values.max()), "BaselineMaxMove": move,
                     "Updates": int(np.count_nonzero(np.diff(values))),
                     "CapCounts": cap if cap is not None else np.nan,
                     "ReachedCap": bool(cap is not None and move >= cap)})
    return rows


def baseline_note(rows: list[dict]) -> str | None:
    """A DFM note when any well's Baseline in force moved, for ``summary.txt``
    and the report."""
    moved = [r for r in rows if r["BaselineMaxMove"] > 0]
    if not moved:
        return None
    biggest = max(moved, key=lambda r: r["BaselineMaxMove"])
    capped = [r["Well"] for r in rows if r["ReachedCap"]]
    text = (f"the baseline in force moved on {len(moved)} well(s), at most "
            f"{biggest['BaselineMaxMove']:.0f} counts (well {biggest['Well']})")
    if capped:
        text += f"; well(s) {', '.join(str(w) for w in capped)} reached the cap"
    return text


def onset_minute(minutes: np.ndarray, lit: np.ndarray, unexplained: np.ndarray,
                 fraction: float, window_min: float = WINDOW_MIN) -> float:
    """Start minute of the first *window_min*-minute window of the recording in
    which at least *fraction* of the lit samples are unexplained; NaN when none."""
    lit = np.asarray(lit, dtype=bool)
    if not lit.any():
        return float("nan")
    mins = np.asarray(minutes, dtype=float)
    bins = np.floor(mins / float(window_min)).astype(int)
    lit_bins = bins[lit]
    unexplained_bins = bins[np.asarray(unexplained, dtype=bool) & lit]
    lit_n = pd.Series(1, index=lit_bins).groupby(level=0).sum()
    un_n = pd.Series(1, index=unexplained_bins).groupby(level=0).sum() \
        .reindex(lit_n.index, fill_value=0)
    share = un_n / lit_n
    hit = share[share >= float(fraction) - 1e-12]
    if hit.empty:
        return float("nan")
    return float(hit.index.min() * float(window_min))


def mode_value(values: np.ndarray):
    """The most common finite value of *values*, or ``None``."""
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if v.empty:
        return None
    return float(v.mode().iloc[0])


# ---------------------------------------------------------------------------
# One DFM
# ---------------------------------------------------------------------------

@dataclass
class DFMInputs:
    """One DFM's arrays, each ``n`` samples long (wells are columns 0..11)."""

    dfm_id: int
    minutes: np.ndarray
    samples_per_second: float
    lights: np.ndarray                   # bool (n, 12): OptoCol1, the LEDs lit
    activity: np.ndarray                 # bool (n, 12): feeding lick or tasting sample
    feeding: np.ndarray                  # int (n, 12): event_df, lengths at starts
    raw: np.ndarray | None = None        # float (n, 12): signal, training flag removed
    chamber_of_well: dict[int, int] = field(default_factory=dict)
    origin: datetime | None = None       # absolute time of recording minute 0
    data_frequency: np.ndarray | None = None
    data_pulse_width: np.ndarray | None = None
    data_dark: np.ndarray | None = None
    #: The recording's Generation, and on generation 3 its Paradigm state,
    #: row for row with these arrays; with both, the QC reads the firmware's
    #: decisions instead of ``lights`` and the clock-time schedule.
    generation: int = GENERATION_FLAG
    state: ParadigmState | None = None
    #: The recording's first rows when a range clipped them off (a Slice
    #: that starts later; :class:`pyflic.base.dfm.BeforeRange`): their
    #: minutes, and their raw signal ``(k, 12)`` with the training flag
    #: removed.  The firmware's baseline is read from them, as it was
    #: captured at the start of the run, not of the Slice.
    start_minutes: np.ndarray | None = None
    start_raw: np.ndarray | None = None
    #: PyMCU's Baseline columns, ``(n, 12)`` float, NaN where ``NA``: the
    #: baseline the firmware's thresholds were built on, row by row.  With
    #: them the Emulated Trigger needs no reconstruction, and the drift and
    #: sustained-contact causes are told apart.  ``None`` without them.
    baseline_in_force: np.ndarray | None = None


@dataclass
class DFMOptoResult:
    """The light QC of one DFM."""

    dfm_id: int
    setting: str
    program_status: str
    groups: list[dict]
    intervals: list[dict]
    events: list[dict]
    flags: list[str]
    notes: list[str]
    decay_samples: np.ndarray             # per sample, for the PR light QC
    lit_samples: int = 0
    #: Per Linkage Group, seconds per recording minute: ``Minute, LitSec,
    #: ExplainedSec, UnexplainedSec, LitWhileOffSec, NotJudgedSec,
    #: ContactWithoutActivitySec`` (the QC figure).  Lit time splits four
    #: ways: judged against licks and explained or not, lit while every
    #: threshold was -1, and lit where licks do not decide (open loop,
    #: non-feeding activation, outside the schedule).  On generation 3 the
    #: seconds are ``OnMs`` sums and ``ContactWithoutActivitySec`` is the lit
    #: time of Activations whose onset had no pyflic lick or touch.
    timelines: dict = field(default_factory=dict)
    generation: int = GENERATION_FLAG
    #: :func:`baseline_rows`: how each well's Baseline in force moved, when
    #: the data have PyMCU's Baseline columns (``opto_baselines.csv``).
    baselines: list[dict] = field(default_factory=list)


@dataclass
class _Window:
    index: int
    interval: op.ProgramInterval | None
    occurrence: int
    start_min: float
    end_min: float
    i0: int
    i1: int
    #: The simple export: every well Off, whatever the data's linkage.
    all_off: bool = False


#: What a DFM's notes say when ``Program.txt`` is the simple export.
ALL_OFF_NOTE = ("Program.txt is the export of a recording made without a Program "
                "(***ALL DFM***): every well Off throughout, so any light is a fault")


def _check_program_times(program: op.OptoProgram, inp: DFMInputs, origin: datetime,
                         dfm_flags: list[str], dfm_notes: list[str], *,
                         settings: OptoQCSettings | None = None) -> float:
    """A data start more than a minute from the program's, and data running
    past its End Time.

    On generation 3 (*inp* carries its Paradigm state, and *settings* are
    given) also *interval started late* (decision 18): each Program
    Interval's first row, on every Cycle, against its ``Program.txt`` time
    (:func:`rule_checks.interval_lags`), warning above
    ``opto_interval_lag_warn_sec``.  Returns the worst lag in seconds
    (``IntervalLagSec``), ``NaN`` when none is measured."""
    minutes = np.asarray(inp.minutes, dtype=float)
    if program.start is not None and inp.origin is not None:
        lag = (inp.origin - program.start).total_seconds()
        if abs(lag) > START_TOLERANCE_SEC:
            dfm_flags.append(PROGRAM_MISMATCH)
            dfm_notes.append(f"the data start {abs(lag) / 60.0:.1f} min "
                             f"{'after' if lag > 0 else 'before'} the program's "
                             f"Start Time")
    end = program.resolved_end()
    if end is not None and minutes.size:
        past = (origin + timedelta(minutes=float(minutes[-1])) - end).total_seconds()
        if past > 60.0:
            dfm_notes.append(f"the data run {past / 60.0:.1f} min past the program's "
                             f"End Time; light after it is not judged")
    section = program.section(inp.dfm_id)
    if settings is None or inp.state is None or section is None or program.start is None \
            or inp.origin is None:
        return float("nan")
    lags = rc.interval_lags(inp.state, section, program.start, inp.origin)
    late = rc.interval_started_late(lags, warn_sec=settings.interval_lag_warn_sec)
    if late.fired:
        dfm_flags.append(INTERVAL_STARTED_LATE)
        dfm_notes.append(late.reason)
    return float(late.value) if late.applies else float("nan")


def _check_light_settings(inp: DFMInputs, paradigm: op.Paradigm, dark: bool | None,
                          i0: int, i1: int, key: str, where: str, seen: set[str],
                          dfm_flags: list[str], dfm_notes: list[str]) -> None:
    """The data's ``OptoFreq``, ``OptoPW`` and ``Dark`` over rows ``[i0, i1)``
    against *paradigm* and the interval's *dark* (``None``: not known), each
    difference noted once per *key* and flagged as a program mismatch."""
    legacy = paradigm.legacy
    ## A legacy export echoed raw values the data may carry either decoded or
    ## as written; a 4.0.0 Paradigm states them plainly.
    if legacy is not None:
        frequencies = (legacy.frequency_hz, legacy.frequency)
        pulse_widths = (legacy.pulse_width_ms, legacy.pulse_width)
    else:
        frequencies = (paradigm.frequency_hz,)
        pulse_widths = (paradigm.pulse_width_ms,)
    checks = [
        ("OptoFreq", inp.data_frequency, frequencies, f"{paradigm.frequency_hz} Hz"),
        ("OptoPW", inp.data_pulse_width, pulse_widths, f"{paradigm.pulse_width_ms} ms"),
    ]
    if dark is not None:
        checks.append(("Dark", inp.data_dark, (1.0 if dark else 0.0,),
                       "on" if dark else "off"))
    for column, values, allowed, expected in checks:
        if values is None:
            continue
        observed = mode_value(np.asarray(values)[i0:i1])
        if observed is None:
            continue
        if column == "Dark":
            observed = 1.0 if observed != 0 else 0.0
        if any(abs(observed - float(a)) < 1e-9 for a in allowed):
            continue
        if f"{column}:{key}" in seen:
            continue
        seen.add(f"{column}:{key}")
        if PROGRAM_MISMATCH not in dfm_flags:
            dfm_flags.append(PROGRAM_MISMATCH)
        shown = (("on" if observed else "off") if column == "Dark" else f"{observed:g}")
        dfm_notes.append(f"{where}: the data's {column} is {shown}, the program says "
                         f"{expected}")


def _chambers_label(chambers) -> str:
    return ", ".join(str(c) for c in sorted(set(chambers))) or "—"


def _to_minutes(when: datetime, origin: datetime) -> float:
    return (when - origin).total_seconds() / 60.0


def _program_windows(section: op.DFMProgram, program: op.OptoProgram,
                     inp: DFMInputs, origin: datetime) -> list[_Window]:
    minutes = inp.minutes
    n = minutes.size
    anchor = program.start or origin
    data_end = origin + timedelta(minutes=float(minutes[-1]) if n else 0.0,
                                  seconds=1.0 / float(inp.samples_per_second))
    end = program.resolved_end() or data_end
    windows: list[_Window] = []
    for k, sched in enumerate(section.schedule(anchor, end)):
        a = _to_minutes(sched.start, origin)
        b = _to_minutes(sched.end, origin)
        i0 = int(np.searchsorted(minutes, a, side="left"))
        i1 = int(np.searchsorted(minutes, b, side="left"))
        windows.append(_Window(index=k + 1, interval=sched.interval,
                               occurrence=sched.occurrence, start_min=a, end_min=b,
                               i0=i0, i1=i1))
    return windows


def decay_profile(dfm_id: int, minutes: np.ndarray, samples_per_second: float,
                  origin: datetime | None, program: op.OptoProgram | None,
                  settings: OptoQCSettings, *, state: ParadigmState | None = None
                  ) -> np.ndarray:
    """Per sample, the light decay in samples: the program's interval Hold
    where one of *dfm_id*'s intervals applies, else ``opto_default_decay_ms``.
    With a generation-3 *state* the windows are its rows' Program Intervals,
    not the clock-time schedule."""
    minutes = np.asarray(minutes, dtype=float)
    sps = float(samples_per_second)
    out = np.full(minutes.size, decay_samples_for(settings.default_decay_ms, sps),
                  dtype=int)
    if state is not None and len(state) == minutes.size:
        for w in _paradigm_windows(state, program, int(dfm_id))[0]:
            hold = w.hold_ms(settings)
            if hold is not None and w.i1 > w.i0:
                out[w.i0:w.i1] = decay_samples_for(hold, sps)
        return out
    section = program.section(dfm_id) if program is not None else None
    if section is None or minutes.size == 0:
        return out
    anchor = origin or program.start or datetime(2000, 1, 1)
    probe = DFMInputs(dfm_id=int(dfm_id), minutes=minutes, samples_per_second=sps,
                      lights=np.zeros((0, op.N_WELLS), bool),
                      activity=np.zeros((0, op.N_WELLS), bool),
                      feeding=np.zeros((0, op.N_WELLS), int), origin=origin)
    for w in _program_windows(section, program, probe, anchor):
        if w.i1 > w.i0:
            out[w.i0:w.i1] = decay_samples_for(w.interval.hold_ms, sps)
    return out


def _inferred_groups(lights: np.ndarray) -> dict[int, tuple[int, ...]]:
    """Wells whose light is identical over the whole recording, grouped — the
    linkage the data shows when no program states it.  Unlit wells are left
    out: there is no light of theirs to explain."""
    groups: dict[bytes, list[int]] = {}
    for w in range(1, op.N_WELLS + 1):
        col = lights[:, w - 1]
        if not col.any():
            continue
        groups.setdefault(np.packbits(col).tobytes(), []).append(w)
    ordered = sorted(groups.values(), key=lambda ws: ws[0])
    return {k + 1: tuple(ws) for k, ws in enumerate(ordered)}


def analyze_dfm(inp: DFMInputs, *, program: op.OptoProgram | None, setting: str,
                settings: OptoQCSettings) -> DFMOptoResult:
    """The optogenetic light QC of one DFM.

    With a program section, each Linkage Group is judged interval by interval
    under that interval's mode: open loop (lit fraction), off (any light is a
    fault), a feeding paradigm (Explained Light, and for closed loop the
    reverse — feeding that never lit the group) or non-feeding activation (not
    judged).  Without one, groups are inferred from wells lit together, every
    member counts as a Trigger Well, the decay is the design's fallback and
    unexplained light can warn but not fail: an open-loop schedule is
    indistinguishable from a stuck light without the program.  The
    ``***ALL DFM***`` export of a recording made without a Program says every
    well was ``Off``, so the whole recording is judged off.

    A generation-3 recording (*inp* carries its Paradigm state) is judged
    from the firmware's own record instead (:func:`_analyze_paradigm_state`).
    """
    if inp.generation == GENERATION_PARADIGM and inp.state is not None \
            and len(inp.state) == np.asarray(inp.minutes).size:
        return _analyze_paradigm_state(inp, program=program, setting=setting,
                                       settings=settings)
    minutes = np.asarray(inp.minutes, dtype=float)
    n = minutes.size
    sps = float(inp.samples_per_second)
    tol = int(settings.tolerance_samples)
    lights = np.asarray(inp.lights, dtype=bool).reshape(n, op.N_WELLS)
    activity = np.asarray(inp.activity, dtype=bool).reshape(n, op.N_WELLS)
    feeding = np.asarray(inp.feeding).reshape(n, op.N_WELLS)
    raw = None if inp.raw is None else np.asarray(inp.raw, dtype=float).reshape(n, op.N_WELLS)
    recorded = _recorded_baseline(inp, n)
    drift_rows_for = _drift_rows_factory(recorded, minutes, raw)
    fallback_decay = decay_samples_for(settings.default_decay_ms, sps)
    decay_by_sample = np.full(n, fallback_decay, dtype=int)
    dfm_flags: list[str] = []
    dfm_notes: list[str] = []

    all_off = program is not None and program.grammar == op.GRAMMAR_SIMPLE
    section = program.section(inp.dfm_id) if program is not None else None
    if all_off:
        status = PROGRAM_ALL_OFF
        dfm_notes.append(ALL_OFF_NOTE)
    elif program is None:
        status = PROGRAM_NO_FILE
        dfm_notes.append("no Program.txt in data/, so this QC is limited: linkage is "
                         "inferred from wells lit together, every lit well counts as a "
                         f"trigger well, the decay is taken as {settings.default_decay_ms} "
                         "ms (opto_default_decay_ms) and unexplained light can warn "
                         "but not fail")
    elif section is None:
        if inp.dfm_id in program.rejected:
            status = PROGRAM_DROPPED
            dfm_notes.append(f"Program.txt's section for DFM {inp.dfm_id} could not be "
                             f"read ({program.rejected[inp.dfm_id]}), so this DFM is "
                             f"judged as if there were no Program.txt")
        else:
            status = PROGRAM_NO_SECTION
            dfm_notes.append(f"Program.txt has no section for DFM {inp.dfm_id}, so this "
                             f"DFM is judged as if there were no Program.txt")
        dfm_flags.append(NO_PROGRAM_SECTION)
    else:
        status = PROGRAM_OK

    origin = inp.origin
    if section is not None and origin is None:
        origin = program.start or datetime(2000, 1, 1)
        dfm_notes.append("the data carry no date and time, so the recording is taken "
                         "to start at the program's Start Time")

    ## ---- windows and groups ----
    windows: list[_Window]
    if section is not None:
        windows = _program_windows(section, program, inp, origin)
        groups = section.linkage_groups()
        linkage_source = "program"
    else:
        windows = [_Window(index=1, interval=None, occurrence=1,
                           start_min=float(minutes[0]) if n else 0.0,
                           end_min=float(minutes[-1]) if n else 0.0, i0=0, i1=n,
                           all_off=all_off)]
        groups = _inferred_groups(lights) if n else {}
        linkage_source = "inferred"
    for w in windows:
        if w.interval is not None and w.i1 > w.i0:
            decay_by_sample[w.i0:w.i1] = decay_samples_for(w.interval.hold_ms, sps)

    ## ---- the firmware's baseline, for the Emulated Trigger ----
    emulate = section is not None and raw is not None
    approximate = False
    base = np.zeros(op.N_WELLS)
    ## PyMCU's Baseline columns give it row by row, so nothing is reconstructed.
    use_recorded = emulate and recorded is not None and program.baseline
    if emulate and not use_recorded:
        n_base = max(1, int(round(BASELINE_SECONDS * sps)))
        ## When the firmware captured its baseline, in recording minutes: at
        ## the program's start, or at the recording's first row when the
        ## program started before it or the data carry no clock.
        base_min = 0.0
        if program.start is not None and inp.origin is not None:
            lag = (inp.origin - program.start).total_seconds()
            if lag > BASELINE_ALIGN_SEC:
                approximate = True
                dfm_notes.append(f"the data start {lag / 60.0:.1f} min after the "
                                 f"program, so the firmware's baseline is not in them; "
                                 f"the emulated trigger uses the first "
                                 f"{BASELINE_SECONDS:g} s of data instead (approximate)")
            else:
                base_min = _to_minutes(program.start, origin)
        elif inp.origin is None:
            approximate = True
        source_minutes, source_raw = minutes, raw
        ## A range that starts after that moment (a Slice) clipped the
        ## baseline off the rows analysed: read it from the recording's first
        ## rows, kept aside at load, or say the emulation is approximate.
        if n and minutes[0] > base_min + BASELINE_ALIGN_SEC / 60.0:
            kept = inp.start_minutes
            at = (int(np.searchsorted(kept, base_min, side="left"))
                  if kept is not None and inp.start_raw is not None and len(kept) else None)
            if at is not None and kept[0] <= base_min + BASELINE_ALIGN_SEC / 60.0 \
                    and at + n_base <= len(kept):
                source_minutes = np.asarray(kept, dtype=float)
                source_raw = np.asarray(inp.start_raw, dtype=float).reshape(-1, op.N_WELLS)
            else:
                approximate = True
                dfm_notes.append(f"the rows analysed begin at minute {minutes[0]:.1f}, after "
                                 f"the firmware captured its baseline, so the emulated "
                                 f"trigger uses their first {BASELINE_SECONDS:g} s instead "
                                 f"(approximate)")
        start_idx = int(np.searchsorted(source_minutes, base_min, side="left"))
        if program.baseline:
            base = np.array([firmware_baseline(source_raw[:, w], start_idx, n_base)
                             for w in range(op.N_WELLS)])
    base_rows = np.broadcast_to(base, (n, op.N_WELLS))
    if use_recorded:
        base_rows = _filled_baseline(recorded)

    ## ---- program versus data ----
    if section is not None:
        _check_program_times(program, inp, origin, dfm_flags, dfm_notes)
        seen: set[str] = set()
        for w in windows:
            if w.interval is None or w.i1 <= w.i0:
                continue
            _check_light_settings(inp, w.interval.paradigm, w.interval.dark, w.i0, w.i1,
                                  str(w.interval.index), f"interval {w.interval.index}",
                                  seen, dfm_flags, dfm_notes)

    ## ---- each Linkage Group ----
    group_rows: list[dict] = []
    interval_rows: list[dict] = []
    event_rows: list[dict] = []
    ## Which window each sample falls in (-1: outside the program's schedule).
    window_of = np.full(n, -1, dtype=int)
    for k, w in enumerate(windows):
        window_of[w.i0:w.i1] = k
    covered = window_of >= 0
    near_cache: dict[tuple, np.ndarray] = {}

    def any_activity(trig: tuple[int, ...]) -> np.ndarray:
        if not trig:
            return np.zeros(n, dtype=bool)
        return activity[:, [t - 1 for t in trig]].any(axis=1)

    def activity_near(trig: tuple[int, ...], before: int, after: int) -> np.ndarray:
        key = (trig, before, after)
        if key not in near_cache:
            near_cache[key] = near(any_activity(trig), before, after)
        return near_cache[key]

    expects_light = False
    timelines: dict = {}
    for label, wells in groups.items():
        idx = [w - 1 for w in wells]
        lit = lights[:, idx].any(axis=1)
        flags: list[str] = []
        notes: list[str] = []
        judged = np.zeros(n, dtype=bool)
        unexplained = np.zeros(n, dtype=bool)
        contact_all = np.zeros(n, dtype=bool)
        contact_near_all = np.zeros(n, dtype=bool)
        contact_alone = np.zeros(n, dtype=bool)
        off_mask = np.zeros(n, dtype=bool)
        feeding_events = unlit_events = closed_events = 0
        open_samples = open_lit = 0
        off_lit = 0
        nonfeeding_samples = 0
        contact_without = 0
        paradigms: list[str] = []
        window_modes: list[str] = []
        triggers_seen: set[int] = set()
        partial = lit & ~lights[:, idx].all(axis=1)
        if len(idx) > 1 and partial.sum() > max(tol, 0.01 * lit.sum()):
            notes.append(f"its linked wells were not lit together for "
                         f"{partial.sum() / sps:.0f} s")

        for k, w in enumerate(windows):
            if w.interval is not None:
                mode = w.interval.group_mode(wells)
                ## Under an OpenLoop Paradigm a level well is lit regardless:
                ## the firmware runs no rule for it (only a 4.0.0 export has
                ## the type, so generation 1 never takes this branch).
                trig = (() if w.interval.paradigm.type == op.OPEN_LOOP
                        else w.interval.trigger_wells(wells))
                decay_ms = w.interval.hold_ms
            elif w.all_off:
                mode = op.LIGHTS_OFF
                trig = ()
                decay_ms = 0
            else:
                mode = op.UNKNOWN_PARADIGM
                trig = tuple(wells)
                decay_ms = settings.default_decay_ms
            window_modes.append(mode)
            if w.i1 <= w.i0:
                continue
            decay = decay_samples_for(decay_ms, sps)
            ## Light left over from the interval before is not this one's to
            ## explain: skip its decay at every boundary.  At the first row
            ## the light already on came on before the rows (a Slice's start).
            grace = (decay + tol if k > 0
                     else carried_over(lit, decay + tol) if w.i0 == 0 else 0)
            s0 = min(w.i0 + grace, w.i1)
            seg = slice(s0, w.i1)
            full = slice(w.i0, w.i1)
            if mode not in paradigms:
                paradigms.append(mode)
            triggers_seen.update(trig)
            row = {"DFM": inp.dfm_id, "Group": label, "Window": w.index,
                   "Interval": w.interval.index if w.interval is not None else np.nan,
                   "ProgramInterval": np.nan, "Cycle": np.nan,
                   "Occurrence": w.occurrence, "StartMin": w.start_min,
                   "EndMin": w.end_min, "Paradigm": np.nan, "ParadigmType": np.nan,
                   "Mode": mode, "HoldMs": decay_ms,
                   "TriggerWells": op.wells_label(trig), "Samples": int(w.i1 - w.i0),
                   "LitSource": LIT_SOURCE_OPTOCOL1,
                   "LitSec": float(lit[full].sum()) / sps,
                   "LitFraction": float(lit[full].mean()),
                   "UnexplainedSec": np.nan, "ContactSec": np.nan,
                   "ContactWithoutActivitySec": np.nan, "FeedingEvents": np.nan,
                   "UnlitFeedingEvents": np.nan}
            if mode == op.OPEN_LOOP:
                open_samples += int(w.i1 - s0)
                open_lit += int(lit[seg].sum())
                expects_light = expects_light or (w.i1 - s0) > 0
            elif mode == op.LIGHTS_OFF:
                off_lit += int(lit[seg].sum())
                off_mask[seg] |= lit[seg]
            elif mode == op.NON_FEEDING:
                nonfeeding_samples += int(w.i1 - w.i0)
            else:
                explained = activity_near(tuple(trig), decay + tol, tol)
                un = lit[seg] & ~explained[seg]
                judged[seg] |= lit[seg]
                unexplained[seg] |= un
                row["UnexplainedSec"] = float(un.sum()) / sps
                closed = mode in op.CLOSED_LOOP_PARADIGMS
                n_feed = n_unlit = 0
                for t in trig:
                    for a, b in event_spans(feeding[full, t - 1]):
                        a, b = a + w.i0, b + w.i0
                        n_feed += 1
                        if closed and not lit[max(0, a - tol):min(n, b + decay + tol)].any():
                            n_unlit += 1
                feeding_events += n_feed
                row["FeedingEvents"] = n_feed
                if closed:
                    closed_events += n_feed
                    unlit_events += n_unlit
                    row["UnlitFeedingEvents"] = n_unlit
                if emulate and trig:
                    contact = np.zeros(w.i1 - w.i0, dtype=bool)
                    for t in trig:
                        level = w.interval.threshold(t)
                        if not op.is_trigger(level):
                            continue
                        contact |= (raw[full, t - 1] - base_rows[full, t - 1]) > float(level)
                    contact_all[full] |= contact
                    act_close = activity_near(tuple(trig), tol, tol)[full]
                    alone = contact & ~act_close
                    contact_alone[full] |= alone
                    without = int(alone.sum())
                    contact_without += without
                    row["ContactSec"] = float(contact.sum()) / sps
                    row["ContactWithoutActivitySec"] = without / sps
            interval_rows.append(row)

        if emulate and contact_all.any():
            ## The light persists for the decay after contact ends.
            contact_near_all = near(contact_all, int(decay_by_sample.max(initial=0)) + tol,
                                    tol)
        lit_total = int(lit.sum())
        judged_lit = int(judged.sum())
        unexplained_n = int(unexplained.sum())
        fraction = unexplained_n / judged_lit if judged_lit else float("nan")
        unexplained_sec = unexplained_n / sps
        onset = float("nan")
        if judged_lit and unexplained_n and unexplained_sec >= settings.min_unexplained_sec:
            if fraction >= settings.fail_fraction:
                flags.append(UNEXPLAINED if section is not None else UNEXPLAINED_NO_PROGRAM)
            elif fraction >= settings.warn_fraction:
                flags.append(PARTLY_UNEXPLAINED)
            if flags:
                onset = onset_minute(minutes, judged, unexplained, settings.warn_fraction)
        elif judged_lit and unexplained_n and fraction >= settings.warn_fraction:
            notes.append(f"{unexplained_sec:.1f} s of unexplained light "
                         f"({100 * fraction:.0f}% of lit time) is under "
                         f"opto_unexplained_min_sec, so not judged")
        unexplained_with_contact = int((unexplained & contact_near_all).sum())
        cause = ""
        if emulate and unexplained_n and any(f in (UNEXPLAINED, PARTLY_UNEXPLAINED)
                                             for f in flags):
            cause = (contact_cause(np.ones(n), unexplained & contact_near_all,
                                   drift_rows_for(triggers_seen))
                     if unexplained_with_contact >= 0.5 * unexplained_n else CAUSE_HARDWARE)
        ## A fired flag says its own numbers (the table has them as columns);
        ## notes are for what no flag says.
        if off_lit > 0:
            flags.append(LIGHT_WHILE_OFF)
        open_fraction = open_lit / open_samples if open_samples else float("nan")
        if open_samples and open_fraction < settings.open_loop_min_fraction:
            flags.append(OPEN_LOOP_DARK)
        if closed_events >= MIN_FEEDING_EVENTS and \
                unlit_events / closed_events >= settings.unlit_fraction:
            flags.append(UNLIT_FEEDING)
        elif 0 < closed_events < MIN_FEEDING_EVENTS and unlit_events:
            notes.append(f"{unlit_events} of {closed_events} closed-loop feeding bouts "
                         f"unlit; {MIN_FEEDING_EVENTS} are needed for a verdict")
        feeding_windows = any(m in op.FEEDING_PARADIGMS for m in paradigms)
        if section is not None and feeding_windows and judged_lit == 0 and feeding_events:
            flags.append(NO_LIGHT_EVENTS)
            expects_light = True
        if nonfeeding_samples:
            notes.append(f"{nonfeeding_samples / sps / 60.0:.0f} min of non-feeding "
                         f"activation, which this QC does not judge")
        ## A sample or two past the program's end is the sample clock, not
        ## news; ten seconds of light outside the schedule is.
        unscheduled = int((lit & ~covered).sum())
        if section is not None and unscheduled >= 10 * sps:
            notes.append(f"{unscheduled / sps:.1f} s of light outside the program's "
                         f"schedule, not judged")

        ## ---- this group's light events ----
        onsets, ends = light_events(lit)
        act = any_activity(tuple(sorted(triggers_seen)))
        ## The last activity sample at or before each index, for the overrun.
        last_act = (np.maximum.accumulate(np.where(act, np.arange(n), -1)) if n
                    else np.zeros(0, dtype=int))
        cum_judged = np.concatenate([[0], np.cumsum(judged, dtype=np.int64)])
        cum_explained = np.concatenate([[0], np.cumsum(judged & ~unexplained,
                                                       dtype=np.int64)])
        cum_unexplained = np.concatenate([[0], np.cumsum(unexplained, dtype=np.int64)])
        unexplained_events = 0
        for e, (a, b) in enumerate(zip(onsets.tolist(), ends.tolist()), start=1):
            k = int(window_of[a])
            mode = window_modes[k] if k >= 0 else "unscheduled"
            interval = (windows[k].interval.index
                        if k >= 0 and windows[k].interval is not None else np.nan)
            judged_event = bool(cum_judged[b] > cum_judged[a])
            un_sec = overrun = np.nan
            explained_flag = None
            if judged_event:
                un_sec = float(cum_unexplained[b] - cum_unexplained[a]) / sps
                explained_flag = bool(cum_explained[b] > cum_explained[a])
                d = int(decay_by_sample[a])
                lo, hi = max(0, a - d - tol), min(n, b + tol)
                last = int(last_act[hi - 1]) if hi > 0 else -1
                if last >= lo:
                    overrun = max(0.0, (b - (last + d + tol + 1)) / sps)
                else:
                    overrun = (b - a) / sps
                if not explained_flag:
                    unexplained_events += 1
            event_rows.append({
                "DFM": inp.dfm_id, "Group": label, "Event": e, "Well": np.nan,
                "OnsetMin": float(minutes[a]),
                "EndMin": float(minutes[b - 1]) + 1.0 / sps / 60.0,
                "DurationSec": (b - a) / sps, "Interval": interval, "Mode": mode,
                "Phase": np.nan, "RequirementMet": np.nan, "FirmwareEvents": np.nan,
                "UnexplainedSec": un_sec, "Explained": explained_flag,
                "OverrunSec": overrun, "Source": EVENT_SOURCE_OPTOCOL1, **_NO_LEDGER,
            })

        if n:
            ## From the first minute of the rows, so a Slice's timeline does
            ## not open with empty minutes before its start.
            first_minute = int(max(0.0, np.floor(minutes[0])))
            minute_bin = np.clip(np.floor(minutes), first_minute, None).astype(int) - first_minute
            size = int(minute_bin.max()) + 1

            def per_minute(mask: np.ndarray) -> np.ndarray:
                return np.bincount(minute_bin[mask], minlength=size) / sps

            timelines[label] = pd.DataFrame({
                "Minute": np.arange(size, dtype=float) + first_minute,
                "LitSec": per_minute(lit),
                "ExplainedSec": per_minute(judged & ~unexplained),
                "UnexplainedSec": per_minute(unexplained),
                "LitWhileOffSec": per_minute(off_mask),
                "NotJudgedSec": per_minute(lit & ~judged & ~off_mask),
                "ContactWithoutActivitySec": per_minute(contact_alone),
            })

        chambers = sorted({inp.chamber_of_well[w] for w in wells
                           if w in inp.chamber_of_well})
        group_rows.append({
            "DFM": inp.dfm_id, "Generation": int(inp.generation), "Group": label,
            "Linkage": label if linkage_source == "program" else np.nan,
            "Wells": op.wells_label(wells),
            "TriggerWells": op.wells_label(triggers_seen),
            "Chambers": _chambers_label(chambers), "Roles": np.nan,
            "_chambers": tuple(chambers),
            "_wells": tuple(wells), "Treatment": "",
            "Program": status, "Paradigm": ", ".join(paradigms),
            "LitSec": lit_total / sps, "JudgedLitSec": judged_lit / sps,
            "UnexplainedSec": unexplained_sec,
            "UnexplainedFraction": fraction, "UnexplainedOnsetMin": onset,
            "LightEvents": int(onsets.size), "UnexplainedEvents": unexplained_events,
            "FeedingEvents": feeding_events,
            "ClosedLoopFeedingEvents": closed_events if closed_events else np.nan,
            "UnlitFeedingEvents": unlit_events if closed_events else np.nan,
            **_FIRMWARE4_ONLY,
            "OpenLoopLitFraction": open_fraction,
            "LitWhileOffSec": (off_lit / sps if section is not None or all_off
                               else np.nan),
            "ContactSec": contact_all.sum() / sps if emulate else np.nan,
            "ContactWithoutActivitySec": contact_without / sps if emulate else np.nan,
            "UnexplainedWithContactSec": (unexplained_with_contact / sps if emulate
                                          else np.nan),
            "EmulationApproximate": bool(approximate) if emulate else np.nan,
            "LikelyCause": cause, "_flags": flags, "_notes": notes,
            "Excluded": False,
        })

    ## ---- the DFM as a whole ----
    lit_any = int(lights.any(axis=1).sum())
    _no_light_check(dfm_flags, dfm_notes, setting, lit_any,
                    (section is None and not all_off) or expects_light)
    if not group_rows:
        group_rows.append(_empty_group_row(inp, status))
    _finish_groups(group_rows, dfm_flags)
    baselines = _baseline_summary(inp, recorded, program, dfm_notes)
    return DFMOptoResult(dfm_id=inp.dfm_id, setting=setting, program_status=status,
                         groups=group_rows, intervals=interval_rows, events=event_rows,
                         flags=dfm_flags, notes=dfm_notes, decay_samples=decay_by_sample,
                         lit_samples=lit_any, timelines=timelines,
                         generation=int(inp.generation), baselines=baselines)


def _recorded_baseline(inp: DFMInputs, n: int) -> np.ndarray | None:
    """*inp*'s Baseline in force as ``(n, 12)``, or None without one."""
    if inp.baseline_in_force is None:
        return None
    recorded = np.asarray(inp.baseline_in_force, dtype=float)
    return recorded.reshape(n, op.N_WELLS) if recorded.size == n * op.N_WELLS else None


def _filled_baseline(recorded: np.ndarray) -> np.ndarray:
    """The Baseline in force with its ``NA`` rows (before the first Interval,
    outside the Program) given each well's first recorded value."""
    first = np.zeros(op.N_WELLS)
    for w in range(op.N_WELLS):
        finite = recorded[:, w][np.isfinite(recorded[:, w])]
        if finite.size:
            first[w] = finite[0]
    return np.where(np.isfinite(recorded), recorded, first)


def _drift_rows_factory(recorded: np.ndarray | None, minutes: np.ndarray,
                        raw: np.ndarray | None):
    """A function of trigger wells giving :func:`baseline_drift_rows`, or
    None when the data have no Baseline columns (the causes stay merged).
    The Floors are computed once, on first use."""
    cache: dict[str, np.ndarray] = {}

    def rows(wells) -> np.ndarray | None:
        if recorded is None or raw is None:
            return None
        if "floor" not in cache:
            cache["floor"] = resting_floor(minutes, raw)
        return baseline_drift_rows(recorded, cache["floor"], tuple(wells))
    return rows


def _baseline_summary(inp: DFMInputs, recorded: np.ndarray | None,
                      program: op.OptoProgram | None, notes: list[str]) -> list[dict]:
    """:func:`baseline_rows`, with :func:`baseline_note` added to *notes*."""
    rows = baseline_rows(inp.dfm_id, recorded, program)
    note = baseline_note(rows)
    if note is not None:
        notes.append(note)
    return rows


#: The QC table's generation-3 columns, ``NA`` on generations 1 and 2.
_FIRMWARE4_ONLY: dict[str, Any] = {
    "FirmwareEvents": np.nan, "PyflicEvents": np.nan, "AcclimationActivations": np.nan,
    "AcclimationExpected": np.nan, "FirmwareEventsWithoutActivity": np.nan,
    **{c: np.nan for c in rc.RULE_QC_COLUMNS},
}


def _no_light_check(dfm_flags: list[str], dfm_notes: list[str], setting: str,
                    lit_samples: int, expects_light: bool) -> None:
    """``optogenetics: yes`` and nothing ever lit, where light was expected."""
    if setting == op.SETTING_YES and lit_samples == 0 and expects_light:
        dfm_flags.append(NO_LIGHT)
        dfm_notes.append("optogenetics is 'yes' but no LED was lit on this DFM for the "
                         "whole recording: an unplugged lid, or a program that did not "
                         "load")


def _empty_group_row(inp: DFMInputs, status: str) -> dict:
    """Nothing lit and nothing programmed: one row speaks for the DFM, so a
    failure has somewhere to be recorded."""
    chambers = sorted(set(inp.chamber_of_well.values()))
    return {
        "DFM": inp.dfm_id, "Generation": int(inp.generation), "Group": 0,
        "Linkage": np.nan, "Wells": "W1-W12",
        "TriggerWells": "—", "Chambers": _chambers_label(chambers), "Roles": np.nan,
        "_chambers": tuple(chambers), "_wells": tuple(range(1, op.N_WELLS + 1)),
        "Treatment": "", "Program": status, "Paradigm": "",
        "LitSec": 0.0, "JudgedLitSec": 0.0, "UnexplainedSec": 0.0,
        "UnexplainedFraction": np.nan, "UnexplainedOnsetMin": np.nan,
        "LightEvents": 0, "UnexplainedEvents": 0, "FeedingEvents": 0,
        "ClosedLoopFeedingEvents": np.nan,
        "UnlitFeedingEvents": np.nan, **_FIRMWARE4_ONLY, "OpenLoopLitFraction": np.nan,
        "LitWhileOffSec": np.nan, "ContactSec": np.nan,
        "ContactWithoutActivitySec": np.nan, "UnexplainedWithContactSec": np.nan,
        "EmulationApproximate": np.nan, "LikelyCause": "", "_flags": [],
        "_notes": ["no LED was lit on this DFM"], "Excluded": False,
    }


def _finish_groups(group_rows: list[dict], dfm_flags: list[str]) -> None:
    """Each group's ``Flags`` (its own, then its DFM's), ``Verdict`` and
    ``Notes``, in place."""
    for row in group_rows:
        own = [f for f in row.pop("_flags")
               if not (f == NO_LIGHT_EVENTS and NO_LIGHT in dfm_flags)]
        flags = list(dict.fromkeys([*own, *dfm_flags]))
        row["Flags"] = ", ".join(flags)
        row["Verdict"] = verdict_of(flags)
        row["Notes"] = "; ".join(row.pop("_notes"))


# ---------------------------------------------------------------------------
# Generation 3: the firmware's own record
# ---------------------------------------------------------------------------

@dataclass
class _RunWindow:
    """One generation-3 window: a run of rows under one Program Interval and
    Cycle (``ParadigmState.interval_runs``), joined to the program.

    *kind* is ``"judged"`` (its Paradigm is known, or no program says
    otherwise), ``"off"`` (every well ``Off``: the simple export, or PyMCU's
    ``Idle``) or ``"unjudged"`` (*why* says what: an unknown instruction, a
    Paradigm name the program lacks).  *paradigm* is ``None`` when judged
    without a program; *interval* is the printed line, ``None`` beyond the
    lines the export prints.
    """

    index: int
    i0: int
    i1: int
    kind: str
    program_interval: int | None = None
    cycle: int | None = None
    name: str | None = None
    paradigm: op.Paradigm | None = None
    interval: op.ProgramInterval | None = None
    occurrence: int = 1
    why: str = ""

    def hold_ms(self, settings: OptoQCSettings) -> int | None:
        """The Hold this window's light is judged with: the printed line's
        (which keeps a legacy decay), else the Paradigm's, else
        ``opto_default_decay_ms`` without a program; ``None`` unjudged."""
        if self.kind == "off":
            return 0
        if self.kind != "judged":
            return None
        if self.paradigm is None:
            return int(settings.default_decay_ms)
        if self.interval is not None and self.interval.paradigm is self.paradigm:
            return int(self.interval.hold_ms)
        return int(self.paradigm.hold or 0)


def _paradigm_named(program: op.OptoProgram, name: str | None) -> op.Paradigm | None:
    """The program's Paradigm called *name*; PyMCU matches names ignoring case."""
    if not name:
        return None
    found = program.paradigms.get(name)
    if found is None:
        folded = name.lower()
        found = next((p for key, p in program.paradigms.items() if key.lower() == folded),
                     None)
    return found


def _paradigm_windows(state: ParadigmState, program: op.OptoProgram | None, dfm_id: int
                      ) -> tuple[list[_RunWindow], dict[str, list[int]], list[str]]:
    """``(windows, {name absent from the program: its Program Intervals},
    notes)`` for one DFM's generation-3 rows.

    Baseline runs (``Interval`` 0) are left out: they are outside every light
    judgement (decision 30).  A program run joins to the program's printed
    line by its Program Interval; one beyond the printed lines (the export
    omits Intervals past the duration and lists a Repeating program's first
    pass only) is judged by its Paradigm by name, thresholds unknown.
    """
    windows: list[_RunWindow] = []
    missing: dict[str, list[int]] = {}
    notes: list[str] = []
    n = len(state)
    simple = program is not None and program.grammar == op.GRAMMAR_SIMPLE
    if simple:
        lead = min(state.lead_in_rows(), n)
        if n > lead:
            windows.append(_RunWindow(index=1, i0=lead, i1=n, kind="off", name=None))
        return windows, missing, notes
    section = program.section(dfm_id) if program is not None else None
    beyond: list[int] = []
    renamed: list[str] = []
    occurrences: Counter = Counter()
    for run in state.interval_runs():
        if run.kind == "baseline":
            continue
        common = dict(index=len(windows) + 1, i0=run.i0, i1=run.i1,
                      program_interval=run.program_interval, cycle=run.cycle,
                      name=run.paradigm)
        if run.kind != "program":
            windows.append(_RunWindow(kind="unjudged", why=UNKNOWN_INSTRUCTION, **common))
            continue
        occurrences[run.program_interval] += 1
        common["occurrence"] = occurrences[run.program_interval]
        if section is None:
            ## No program to say what the Paradigm is: judged on the data
            ## alone (the caller's fallback rule).
            windows.append(_RunWindow(kind="judged", **common))
            continue
        k = int(run.program_interval)
        line = program.interval_for(dfm_id, k)
        paradigm = _paradigm_named(program, run.paradigm)
        if run.paradigm is None:
            paradigm = line.paradigm if line is not None else None
        if paradigm is None and run.paradigm is not None \
                and run.paradigm.lower() == IDLE.lower():
            ## PyMCU's own Paradigm: ClosedLoop with every Well Off.
            windows.append(_RunWindow(kind="off", **common))
            continue
        if paradigm is None:
            if run.paradigm is not None:
                missing.setdefault(run.paradigm, [])
                if k not in missing[run.paradigm]:
                    missing[run.paradigm].append(k)
                why = f"Paradigm {run.paradigm} is not in Program.txt"
            else:
                why = "no Paradigm recorded and no printed line"
            windows.append(_RunWindow(kind="unjudged", why=why, **common))
            continue
        if line is None:
            if k not in beyond:
                beyond.append(k)
        elif line.paradigm.name.lower() != paradigm.name.lower():
            text = f"{k} ({paradigm.name}, printed {line.paradigm.name})"
            if text not in renamed:
                renamed.append(text)
        windows.append(_RunWindow(kind="judged", paradigm=paradigm, interval=line,
                                  **common))
    if beyond:
        notes.append(f"Program Interval(s) {', '.join(str(k) for k in beyond)} are beyond "
                     f"the lines Program.txt prints (it omits Intervals past the duration "
                     f"and lists a Repeating program's first pass only), so they are "
                     f"judged by their Paradigm by name with thresholds unknown, their "
                     f"trigger wells read from the Activations")
    if renamed:
        notes.append(f"rows name another Paradigm than Program.txt's line for Program "
                     f"Interval(s) {'; '.join(renamed)}; judged by the rows' Paradigm")
    return windows, missing, notes


def _run_mode(w: _RunWindow, wells: tuple[int, ...], data_triggers: frozenset[int],
              settings: OptoQCSettings) -> tuple[str, tuple[int, ...], int | None]:
    """``(mode, Trigger Wells, Hold in ms)`` of the Linkage Group of *wells*
    during *w*: any member ``On`` is open loop, every member ``Off`` is off,
    an OpenLoop Paradigm is open loop for the members not ``Off``, else the
    Paradigm's mode with the level wells as triggers.  Thresholds unknown
    (beyond the printed lines, or no program) take the wells with
    Activations as triggers; a group without one is not judged."""
    if w.kind == "off":
        return op.LIGHTS_OFF, (), 0
    if w.kind != "judged":
        return (UNKNOWN_INSTRUCTION if w.why == UNKNOWN_INSTRUCTION else NOT_JUDGED), (), None
    hold = w.hold_ms(settings)
    p = w.paradigm
    if p is None:
        trig = tuple(x for x in wells if x in data_triggers)
        return (op.UNKNOWN_PARADIGM, trig, hold) if trig else (NOT_JUDGED, (), None)
    line = w.interval
    if line is not None:
        if any(line.threshold(x) == op.ON for x in wells):
            return op.OPEN_LOOP, (), hold
        if all(line.threshold(x) == op.OFF for x in wells):
            return op.LIGHTS_OFF, (), hold
        if p.type == op.OPEN_LOOP:
            return op.OPEN_LOOP, (), hold
        return p.mode, line.trigger_wells(wells), hold
    trig = tuple(x for x in wells if x in data_triggers)
    if p.type == op.OPEN_LOOP or not trig:
        return NOT_JUDGED, (), None
    return p.mode, trig, hold


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[start, end)`` of each run of True in *mask*."""
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def _paradigm_starts(state: ParadigmState) -> np.ndarray:
    """(n,) bool: rows whose Paradigm name differs from the row before, where
    the firmware reset every Well (row 0 starts one)."""
    n = len(state)
    codes, _ = pd.factorize(state.paradigm, use_na_sentinel=True)
    starts = np.ones(n, dtype=bool)
    if n > 1:
        starts[1:] = codes[1:] != codes[:-1]
    return starts


def _onset_minute_ms(minutes: np.ndarray, judged_ms: np.ndarray, unexplained_ms: np.ndarray,
                     fraction: float, window_min: float = WINDOW_MIN) -> float:
    """:func:`onset_minute` weighted by the milliseconds each row was lit."""
    judged = np.asarray(judged_ms, dtype=float) > 0
    if not judged.any():
        return float("nan")
    bins = np.floor(np.asarray(minutes, dtype=float) / float(window_min)).astype(int)[judged]
    lit = pd.Series(np.asarray(judged_ms, dtype=float)[judged]).groupby(bins).sum()
    bad = pd.Series(np.asarray(unexplained_ms, dtype=float)[judged]).groupby(bins).sum()
    share = bad.reindex(lit.index, fill_value=0.0) / lit
    hit = share[share >= float(fraction) - 1e-12]
    return float("nan") if hit.empty else float(hit.index.min() * float(window_min))


def _record_checks(state: ParadigmState, settings: OptoQCSettings, sps: float,
                   dfm_flags: list[str], dfm_notes: list[str], *,
                   origin: datetime | None = None) -> int:
    """The DFM's own bookkeeping: restarts (*DFM restarted*, decision 15),
    records lost from ``Index`` outside them, rows under an unknown
    instruction after the lead-in, and late records.  Returns the number of
    restarts (``Restarts``)."""
    spans = state.restarts()
    restarted = rc.dfm_restarted(spans, state.minutes, origin=origin,
                                 record_ms=int(round(1000.0 / sps)))
    if restarted.fired:
        dfm_flags.append(DFM_RESTARTED)
        dfm_notes.append(restarted.reason)
    ## A restart's gap is the restart's, which DFM restarted reports.
    gaps, to_restarts = rc.gaps_outside_restarts(state, spans)
    lost = int(sum(missing for _after, missing, _minute in gaps))
    if lost > settings.lost_records_warn:
        dfm_flags.append(LOST_RECORDS)
        after, longest, minute = max(gaps, key=lambda g: g[1])
        dfm_notes.append(f"{LOST_RECORDS}: {lost} record(s) ({lost / sps:.1f} s) missing "
                         f"from Index in {len(gaps)} gap(s), the longest {longest} after "
                         f"Index {after} (minute {minute:.1f}); opto_lost_records_warn is "
                         f"{settings.lost_records_warn}"
                         + (f"; the {to_restarts} record(s) the DFM's restart(s) lost are "
                            f"not counted, as {DFM_RESTARTED} reports them"
                            if to_restarts else ""))
    after_lead = len(state) - state.lead_in_rows()
    ## A restart's rows under no instruction, until PyMCU resent it, are the
    ## restart's too: DFM restarted reports them, as it reports its gap.
    unknown_rows = state.unknown_rows().copy()
    in_restarts = 0
    for span in spans:
        in_restarts += int(unknown_rows[span.row:span.resume_row].sum())
        unknown_rows[span.row:span.resume_row] = False
    unknown = int(unknown_rows.sum())
    if after_lead > 0 and unknown / after_lead > settings.unknown_rows_fraction:
        dfm_flags.append(UNKNOWN_INSTRUCTION_ROWS)
        dfm_notes.append(f"{UNKNOWN_INSTRUCTION_ROWS}: {unknown} of the {after_lead} rows "
                         f"after the lead-in ({100.0 * unknown / after_lead:.1f}%) were "
                         f"recorded under an instruction outside the program and are not "
                         f"judged; opto_unknown_rows_warn_fraction is "
                         f"{settings.unknown_rows_fraction:g}"
                         + (f"; the {in_restarts} row(s) after the DFM's restart(s), "
                            f"before PyMCU resent the instruction, are not counted, as "
                            f"{DFM_RESTARTED} reports them" if in_restarts else ""))
    late = int(state.late_rows().sum())
    if late > settings.late_records_warn:
        dfm_flags.append(LATE_RECORDS)
        dfm_notes.append(f"{LATE_RECORDS}: {late} record(s) with an OnMs above 200 ms, so "
                         f"the record came late; opto_late_records_warn is "
                         f"{settings.late_records_warn}")
    return len(spans)


#: How the light of one Activation is judged (generation 3).
_UNIT_NONE, _UNIT_FIXED, _UNIT_WINDOW, _UNIT_NON_FEEDING = 0, 1, 2, 3


def _analyze_paradigm_state(inp: DFMInputs, *, program: op.OptoProgram | None,
                            setting: str, settings: OptoQCSettings) -> DFMOptoResult:
    """The light QC of one generation-3 DFM, read from its Paradigm state.

    The windows are its rows' Program Intervals (:func:`_paradigm_windows`);
    a group's mode in each is :func:`_run_mode`'s, and its lit time is
    ``OnMs``.  Each Activation of a Trigger Well owns the group's light from
    its onset to the next onset, the light going out, or the Hold past the
    end of its window, whichever comes first.  A fixed-length Main
    Activation is explained, whole, when its well had a lick or touch within
    the tolerance of its onset; a closed-loop or Acclimation Activation's
    light, and light no Activation owns, is judged sample by sample with the
    window rule and the Hold.  NonFeeding light is checked for staying on
    through feeding, open loop for its lit fraction per window, off for any
    light.  With a program, each Paradigm's rule checks run per group
    (:func:`_group_rule_checks`) and the Program Intervals' starts are
    checked against its times; the DFM's restarts are found either way.
    Without a program the linkage and the Trigger Wells come from the data
    (identical ``OnMs``, Activations), every Activation is judged by the
    window rule with ``opto_default_decay_ms`` and unexplained light warns
    without failing, as on generation 1.
    """
    state = inp.state
    minutes = np.asarray(inp.minutes, dtype=float)
    n = minutes.size
    if len(state) != n:
        raise ValueError(f"DFM {inp.dfm_id}: {len(state)} Paradigm rows for {n} samples")
    sps = float(inp.samples_per_second)
    record_ms = 1000.0 / sps
    tol = int(settings.tolerance_samples)
    on_ms = np.asarray(state.on_ms, dtype=float).reshape(n, op.N_WELLS)
    phase_a = np.asarray(state.phase_a, dtype=bool).reshape(n, op.N_WELLS)
    activity = np.asarray(inp.activity, dtype=bool).reshape(n, op.N_WELLS)
    feeding = np.asarray(inp.feeding).reshape(n, op.N_WELLS)
    recorded = _recorded_baseline(inp, n)
    raw = None if inp.raw is None else np.asarray(inp.raw, dtype=float).reshape(n, op.N_WELLS)
    drift_rows_for = _drift_rows_factory(recorded, minutes, raw)
    dfm_flags: list[str] = []
    dfm_notes: list[str] = []

    ## ---- the program ----
    simple = program is not None and program.grammar == op.GRAMMAR_SIMPLE
    section = None if program is None or simple else program.section(inp.dfm_id)
    if simple:
        status = PROGRAM_ALL_OFF
        dfm_notes.append(ALL_OFF_NOTE)
    elif section is None:
        if program is None:
            status = PROGRAM_NO_FILE
            missing_what = "no Program.txt in data/"
        elif inp.dfm_id in program.rejected:
            status = PROGRAM_DROPPED
            missing_what = (f"Program.txt's section for DFM {inp.dfm_id} could not be read "
                            f"({program.rejected[inp.dfm_id]})")
        else:
            status = PROGRAM_NO_SECTION
            missing_what = f"Program.txt has no section for DFM {inp.dfm_id}"
        dfm_flags.append(PROGRAM_MISSING)
        dfm_notes.append(f"{PROGRAM_MISSING}: {missing_what}.  The rows name each "
                         f"Paradigm but not its type, settings or thresholds, so linkage "
                         f"and trigger wells come from the data (identical OnMs, "
                         f"Activations), every Activation is judged against its group's "
                         f"licks within {settings.default_decay_ms} ms "
                         f"(opto_default_decay_ms), the type-dependent checks (each "
                         f"Paradigm's rule checks among them) are not made and unexplained "
                         f"light can warn but not fail")
    else:
        status = PROGRAM_OK

    origin = inp.origin
    if section is not None and origin is None:
        origin = program.start or datetime(2000, 1, 1)

    windows, missing, window_notes = _paradigm_windows(state, program, inp.dfm_id)
    dfm_notes.extend(window_notes)
    if missing:
        dfm_flags.append(PARADIGM_NOT_IN_PROGRAM)
        named = "; ".join(f"{name} (Program Interval {', '.join(str(k) for k in ks)})"
                          for name, ks in missing.items())
        dfm_notes.append(f"{PARADIGM_NOT_IN_PROGRAM}: the rows name {named}, which "
                         f"Program.txt does not define, so Program.txt is not this "
                         f"recording's; those rows are not judged")
    if any(w.interval is not None and w.paradigm is not None
           and w.interval.paradigm.name.lower() != w.paradigm.name.lower() for w in windows):
        dfm_flags.append(PROGRAM_MISMATCH)

    ## ---- program versus data ----
    interval_lag = float("nan")
    if section is not None:
        interval_lag = _check_program_times(program, inp, origin, dfm_flags, dfm_notes,
                                            settings=settings)
        seen: set[str] = set()
        for w in windows:
            if w.kind == "judged" and w.paradigm is not None and w.i1 > w.i0:
                _check_light_settings(
                    inp, w.paradigm, w.interval.dark if w.interval is not None else None,
                    w.i0, w.i1, str(w.program_interval),
                    f"Program Interval {w.program_interval}", seen, dfm_flags, dfm_notes)
    restart_count = _record_checks(state, settings, sps, dfm_flags, dfm_notes,
                                   origin=inp.origin)

    decay_by_sample = np.full(n, decay_samples_for(settings.default_decay_ms, sps), dtype=int)
    window_of = np.full(n, -1, dtype=int)
    for k, w in enumerate(windows):
        window_of[w.i0:w.i1] = k
        hold = w.hold_ms(settings)
        if hold is not None:
            decay_by_sample[w.i0:w.i1] = decay_samples_for(hold, sps)
    judged_window = np.array([w.kind != "unjudged" for w in windows] + [False], dtype=bool)
    scheduled = judged_window[window_of]          # window_of -1 picks the False
    starts = _paradigm_starts(state)
    segment_start = np.maximum.accumulate(np.where(starts, np.arange(n), 0)) if n \
        else np.zeros(0, dtype=int)

    ## ---- linkage, trigger wells, roles ----
    data_triggers = state.wells_with_activations()
    if section is not None:
        groups = section.linkage_groups()
        linkage_source = "program"
    else:
        view = orl.linkage_view(None, state, None)
        groups = {int(label): tuple(wells) for label, wells in view.groups.items()}
        linkage_source = "data"
    roles = (orl.chamber_roles(orl.linkage_view(section, state, None), inp.chamber_of_well)
             if inp.chamber_of_well else {})

    ## ---- the Activation ledger, once for the DFM (contract C3, C4) ----
    ## Each Activation's light period to the millisecond; the light events
    ## join it on (Well, onset row, order within the row).
    book = al.ledger(state, linked={w: tuple(g) for g in groups.values() for w in g},
                     record_ms=int(round(record_ms)))
    book_at = {key: pos for pos, key in enumerate(zip(
        book["Well"].tolist(), book["Row"].tolist(),
        book.groupby(["Well", "Row"]).cumcount().tolist()))}
    book_values = {c: book[c].to_numpy(dtype=float, na_value=np.nan)
                   for c in OPTO_EVENT_INTEGER_COLUMNS}
    book_values.update({c: book[c].to_numpy() for c in ("Timing", "Overlap", "Boundary",
                                                        "Phase")})

    def ledger_fields(pos: int) -> dict[str, Any]:
        """The ledger's columns of its row *pos*, ``NA`` when the ledger has
        none (*pos* -1)."""
        if pos < 0:
            return dict(_NO_LEDGER)
        out: dict[str, Any] = {c: book_values[c][pos] for c in LEDGER_EVENT_COLUMNS}
        out["Overlap"] = bool(out["Overlap"])
        out["_phase_a"] = book_values["Phase"][pos] == "A"
        return out

    near_cache: dict[tuple, np.ndarray] = {}

    def any_activity(trig: tuple[int, ...]) -> np.ndarray:
        if not trig:
            return np.zeros(n, dtype=bool)
        return activity[:, [t - 1 for t in trig]].any(axis=1)

    def activity_near(trig: tuple[int, ...], before: int, after: int) -> np.ndarray:
        key = (tuple(trig), before, after)
        if key not in near_cache:
            near_cache[key] = near(any_activity(tuple(trig)), before, after)
        return near_cache[key]

    def onset_activity(well: int) -> np.ndarray:
        """Activity at *well* itself within the tolerance of each row."""
        return activity_near((well,), tol, tol)

    group_rows: list[dict] = []
    interval_rows: list[dict] = []
    event_rows: list[dict] = []
    timelines: dict = {}
    expects_light = False
    for label, wells in groups.items():
        wells = tuple(int(w) for w in wells)
        idx = [w - 1 for w in wells]
        g_ms = on_ms[:, idx].max(axis=1) if n else np.zeros(0)
        lit = g_ms > 0
        capped = np.minimum(g_ms, record_ms)
        flags: list[str] = []
        notes: list[str] = []
        reasons: dict[str, str] = {}
        modes = [_run_mode(w, wells, data_triggers, settings) for w in windows]
        decays = [decay_samples_for(h, sps) if h is not None else 0 for _m, _t, h in modes]

        ## ---- the Activations of its Trigger Wells, each owning its light ----
        u_rows, u_wells, u_wins = [], [], []
        for well in wells:
            onsets = state.activation_onsets(well)
            if not onsets.size:
                continue
            ks = window_of[onsets]
            is_trigger = np.array([well in m[1] for m in modes] + [False], dtype=bool)
            keep = is_trigger[ks]
            u_rows.append(onsets[keep])
            u_wells.append(np.full(int(keep.sum()), well, dtype=int))
            u_wins.append(ks[keep])
        empty = np.zeros(0, dtype=int)
        unit_row = np.concatenate(u_rows) if u_rows else empty
        unit_well = np.concatenate(u_wells) if u_wells else empty
        unit_window = np.concatenate(u_wins) if u_wins else empty
        order = np.lexsort((unit_well, unit_row))
        unit_row, unit_well, unit_window = (unit_row[order], unit_well[order],
                                            unit_window[order])
        m = unit_row.size
        unit_in_a = phase_a[unit_row, unit_well - 1]
        ## Each Activation's ledger row, joined on (Well, onset row, order
        ## within the row); -1 where the ledger has none.
        unit_pos = np.full(m, -1, dtype=np.int64)
        in_row: Counter = Counter()
        for i in range(m):
            key = (int(unit_well[i]), int(unit_row[i]))
            unit_pos[i] = book_at.get((*key, in_row[key]), -1)
            in_row[key] += 1
        ## How each Activation's light is judged: a fixed-length Main one
        ## whole, by activity at its onset; a closed-loop or Acclimation one
        ## sample by sample with the window rule; a NonFeeding one not at all.
        unit_cls = np.array([
            _UNIT_FIXED if mode in FIXED_LENGTH_PARADIGMS and not in_a
            else _UNIT_WINDOW if (mode in FIXED_LENGTH_PARADIGMS
                                  or mode in op.CLOSED_LOOP_PARADIGMS
                                  or mode == op.UNKNOWN_PARADIGM)
            else _UNIT_NON_FEEDING if mode == op.NON_FEEDING
            else _UNIT_NONE
            for mode, in_a in zip((modes[k][0] for k in unit_window.tolist()),
                                  unit_in_a.tolist())], dtype=int)
        unit_activity = np.array([bool(onset_activity(w)[r]) for r, w
                                  in zip(unit_row.tolist(), unit_well.tolist())], dtype=bool)
        ## The light an Activation owns: to the next onset, the light going
        ## out, or its window's end plus the Hold, whichever comes first (an
        ## open-loop window after it never goes out).
        next_unlit = np.minimum.accumulate(np.where(lit, n, np.arange(n))[::-1])[::-1]
        following = np.append(unit_row[1:], n)[:m]
        cap =np.array([windows[k].i1 + decays[k] + tol for k in unit_window.tolist()],
                       dtype=int)
        unit_end = np.maximum(np.minimum.reduce([following, next_unlit[unit_row], cap,
                                                 np.full(m, n)]), unit_row)
        owner = np.full(n, -1, dtype=int)
        for i in np.flatnonzero(unit_end > unit_row).tolist():
            owner[unit_row[i]:unit_end[i]] = i
        owned = owner >= 0
        owned_rows = np.flatnonzero(owned)
        owned_unit = owner[owned_rows]

        ## ---- Explained Light ----
        judged_ms = np.zeros(n)
        unexplained_ms = np.zeros(n)
        off_ms = np.zeros(n)
        cls_of_row = unit_cls[owned_unit]
        fixed_rows = owned_rows[cls_of_row == _UNIT_FIXED]
        judged_ms[fixed_rows] = g_ms[fixed_rows]
        bad_rows = owned_rows[(cls_of_row == _UNIT_FIXED) & ~unit_activity[owned_unit]]
        unexplained_ms[bad_rows] = g_ms[bad_rows]
        window_rows = cls_of_row == _UNIT_WINDOW
        for k in np.unique(unit_window[owned_unit[window_rows]]).tolist():
            sel = owned_rows[window_rows & (unit_window[owned_unit] == k)]
            _mode, trig, _hold = modes[k]
            ok = activity_near(trig, decays[k] + tol, tol)[sel]
            judged_ms[sel] = g_ms[sel]
            unexplained_ms[sel] = np.where(ok, 0.0, g_ms[sel])

        paradigms: list[str] = []
        triggers_seen: set[int] = set()
        open_windows: list[tuple[str, float]] = []
        nf_rows = 0
        nf_lit_ms = 0.0
        feeding_events = closed_events = unlit_events = 0
        feeding_windows = False
        unit_ms = np.bincount(owned_unit, weights=g_ms[owned_rows], minlength=m)
        feeding_unit = (unit_cls == _UNIT_FIXED) | (unit_cls == _UNIT_WINDOW)
        no_activity_unit = feeding_unit & ~unit_activity
        for k, w in enumerate(windows):
            mode, trig, hold = modes[k]
            if w.i1 <= w.i0:
                continue
            d = decays[k]
            ## Light carried over from the window before is not this one's to
            ## explain: skip its Hold at a boundary.  At the first row the
            ## light already on came on before the rows (a Slice's start): its
            ## Activation's onset, and the activity behind it, are not in them.
            grace = (decays[k - 1] + tol if k > 0 and windows[k - 1].i1 == w.i0
                     else carried_over(lit, d + tol) if w.i0 == 0 else 0)
            s0 = min(w.i0 + grace, w.i1)
            full = slice(w.i0, w.i1)
            free = np.flatnonzero(lit[s0:w.i1] & ~owned[s0:w.i1]) + s0
            if mode not in paradigms:
                paradigms.append(mode)
            triggers_seen.update(trig)
            in_window = unit_window == k
            row = {"DFM": inp.dfm_id, "Group": label, "Window": w.index,
                   "Interval": w.interval.index if w.interval is not None else np.nan,
                   "ProgramInterval": (w.program_interval if w.program_interval is not None
                                       else np.nan),
                   "Cycle": w.cycle if w.cycle is not None else np.nan,
                   "Occurrence": w.occurrence, "StartMin": float(minutes[w.i0]),
                   "EndMin": float(minutes[w.i1 - 1]) + 1.0 / sps / 60.0,
                   "Paradigm": w.name if w.name is not None else np.nan,
                   "ParadigmType": (w.paradigm.type_name if w.paradigm is not None
                                    else np.nan),
                   "Mode": mode, "HoldMs": hold if hold is not None else np.nan,
                   "TriggerWells": op.wells_label(trig), "Samples": int(w.i1 - w.i0),
                   "LitSource": LIT_SOURCE_ON_MS,
                   "LitSec": float(g_ms[full].sum()) / 1000.0,
                   "LitFraction": float(capped[full].sum()) / ((w.i1 - w.i0) * record_ms),
                   "UnexplainedSec": np.nan, "ContactSec": np.nan,
                   "ContactWithoutActivitySec": np.nan, "FeedingEvents": np.nan,
                   "UnlitFeedingEvents": np.nan}
            if mode == op.OPEN_LOOP:
                ## Each open-loop window judged on its own lit fraction, so
                ## one dark Program Interval or pass shows (stream S1).
                if w.i1 > s0:
                    where = (f"Program Interval {w.program_interval}"
                             + (f" (Cycle {w.cycle})" if w.cycle is not None else ""))
                    open_windows.append((where, float(capped[s0:w.i1].sum())
                                         / ((w.i1 - s0) * record_ms)))
                expects_light = expects_light or (w.i1 - s0) > 0
            elif mode == op.LIGHTS_OFF:
                off_ms[free] = g_ms[free]
            elif mode == op.NON_FEEDING:
                ## A NonFeeding light is on until the fly feeds, so a DFM
                ## that never lit is no light recorded.
                expects_light = True
                during = np.zeros(w.i1 - w.i0, dtype=bool)
                for t in trig:
                    for a, b in event_spans(feeding[full, t - 1]):
                        during[a:b] = True
                nf_rows += int(during.sum())
                nf_lit_ms += float(capped[full][during].sum())
            elif mode in op.FEEDING_PARADIGMS or mode == op.UNKNOWN_PARADIGM:
                feeding_windows = feeding_windows or mode in op.FEEDING_PARADIGMS
                ok = activity_near(trig, d + tol, tol)[free]
                judged_ms[free] = g_ms[free]
                unexplained_ms[free] = np.where(ok, 0.0, g_ms[free])
                row["UnexplainedSec"] = float(unexplained_ms[full].sum()) / 1000.0
                ## Feeding that never lit the group, where every bout should:
                ## closed loop, and a well's Acclimation (closed loop with the
                ## Activation as its Hold).  A FixedInterval lockout or a ratio
                ## Requirement not yet met leaves feeding unlit by design.
                closed = mode in op.CLOSED_LOOP_PARADIGMS
                n_feed = n_closed = n_unlit = 0
                for t in trig:
                    for a, b in event_spans(feeding[full, t - 1]):
                        a, b = a + w.i0, b + w.i0
                        n_feed += 1
                        if closed or phase_a[a, t - 1]:
                            n_closed += 1
                            if not lit[max(0, a - tol):min(n, b + d + tol)].any():
                                n_unlit += 1
                feeding_events += n_feed
                closed_events += n_closed
                unlit_events += n_unlit
                row["FeedingEvents"] = n_feed
                if n_closed:
                    row["UnlitFeedingEvents"] = n_unlit
                row["ContactSec"] = float(unit_ms[in_window & feeding_unit].sum()) / 1000.0
                row["ContactWithoutActivitySec"] = float(
                    unit_ms[in_window & no_activity_unit].sum()) / 1000.0
            interval_rows.append(row)

        ## ---- the drifting-baseline signature: the firmware decided, pyflic
        ## saw no lick or touch ----
        evidence = np.zeros(n, dtype=bool)
        evidence[owned_rows] = no_activity_unit[owned_unit]
        events_without = 0
        lone_rises = np.zeros(n, dtype=bool)
        for t in sorted(triggers_seen):
            rises = state.firmware_event_rises(t)
            if not rises.size:
                continue
            counted = np.array([t in mm[1] and mm[0] in FIXED_LENGTH_PARADIGMS
                                for mm in modes] + [False], dtype=bool)
            rises = rises[counted[window_of[rises]]]
            lone = rises[~onset_activity(t)[rises]]
            events_without += int(lone.size)
            lone_rises[lone] = True
        if lone_rises.any():
            evidence |= near(lone_rises, int(max(decays, default=0)) + tol, 0)
        contact_ms = float(unit_ms[feeding_unit].sum())
        without_ms = float(unit_ms[no_activity_unit].sum())

        ## ---- the verdict on Explained Light ----
        judged_total = float(judged_ms.sum())
        unexplained_total = float(unexplained_ms.sum())
        fraction = unexplained_total / judged_total if judged_total else float("nan")
        unexplained_sec = unexplained_total / 1000.0
        onset = float("nan")
        if judged_total and unexplained_total and unexplained_sec >= settings.min_unexplained_sec:
            if fraction >= settings.fail_fraction:
                flags.append(UNEXPLAINED if section is not None else UNEXPLAINED_NO_PROGRAM)
            elif fraction >= settings.warn_fraction:
                flags.append(PARTLY_UNEXPLAINED)
            if flags:
                onset = _onset_minute_ms(minutes, judged_ms, unexplained_ms,
                                         settings.warn_fraction)
        elif judged_total and unexplained_total and fraction >= settings.warn_fraction:
            notes.append(f"{unexplained_sec:.1f} s of unexplained light "
                         f"({100 * fraction:.0f}% of lit time) is under "
                         f"opto_unexplained_min_sec, so not judged")
        with_contact_ms = float(unexplained_ms[evidence].sum())
        cause = ""
        if unexplained_total and any(f in (UNEXPLAINED, PARTLY_UNEXPLAINED,
                                           UNEXPLAINED_NO_PROGRAM) for f in flags):
            cause = (contact_cause(unexplained_ms, evidence, drift_rows_for(triggers_seen))
                     if with_contact_ms >= 0.5 * unexplained_total else CAUSE_HARDWARE)
        off_total = float(off_ms.sum())
        if off_total > 0:
            flags.append(LIGHT_WHILE_OFF)
        open_check = rc.open_loop_not_lit(open_windows,
                                          min_fraction=settings.open_loop_min_fraction)
        open_fraction = open_check.value if open_check is not None else float("nan")
        if open_check is not None and open_check.fired:
            flags.append(OPEN_LOOP_DARK)
            reasons[OPEN_LOOP_DARK] = open_check.reason
        if closed_events >= MIN_FEEDING_EVENTS and \
                unlit_events / closed_events >= settings.unlit_fraction:
            flags.append(UNLIT_FEEDING)
        elif 0 < closed_events < MIN_FEEDING_EVENTS and unlit_events:
            notes.append(f"{unlit_events} of {closed_events} closed-loop feeding bouts "
                         f"unlit; {MIN_FEEDING_EVENTS} are needed for a verdict")
        if section is not None and feeding_windows and judged_total == 0 and feeding_events:
            flags.append(NO_LIGHT_EVENTS)
            expects_light = True

        ## ---- NonFeeding: the light should go out while the fly feeds, on
        ## constants of its own (their defaults are the ones it borrowed) ----
        if nf_rows:
            nf_share = nf_lit_ms / (nf_rows * record_ms)
            if nf_share > settings.non_feeding_lit_fraction and \
                    nf_lit_ms / 1000.0 >= settings.non_feeding_min_sec:
                flags.append(NON_FEEDING_LIT_WHILE_FEEDING)
                reasons[NON_FEEDING_LIT_WHILE_FEEDING] = (
                    f"{NON_FEEDING_LIT_WHILE_FEEDING}: lit {nf_lit_ms / 1000.0:.0f} s of "
                    f"{nf_rows / sps:.0f} s of feeding at its trigger wells "
                    f"({100 * nf_share:.0f}%) under a NonFeeding Paradigm, whose light "
                    f"goes out while the fly feeds; opto_non_feeding_lit_warn_fraction is "
                    f"{settings.non_feeding_lit_fraction:g} and opto_non_feeding_min_sec "
                    f"{settings.non_feeding_min_sec:g}")

        ## ---- yoking: linked wells share one light ----
        yoking = _yoking_problems(state, wells, windows, on_ms, lit, settings)
        if yoking:
            flags.append(YOKING_BROKEN)
            reasons[YOKING_BROKEN] = f"{YOKING_BROKEN}: " + "; ".join(yoking)

        ## ---- Acclimation, as counted against the Paradigm's ----
        acclimation_count = acclimation_expected = 0
        acclimation_runs = 0
        expected_known = True
        mismatches: list[str] = []
        for t in sorted(triggers_seen):
            for count, expected, s0, s1 in _acclimation_counts(
                    state, t, windows, window_of, starts, record_ms):
                acclimation_runs += 1
                acclimation_count += count
                if expected is None:
                    expected_known = False
                    continue
                acclimation_expected += expected
                if count != expected:
                    mismatches.append(f"W{t} counted {count} Activations in Acclimation "
                                      f"(minutes {minutes[s0]:.1f}-{minutes[s1 - 1]:.1f}), "
                                      f"its Paradigm's Acclimation is {expected}")
        if mismatches:
            flags.append(ACCLIMATION_COUNT_MISMATCH)
            reasons[ACCLIMATION_COUNT_MISMATCH] = (f"{ACCLIMATION_COUNT_MISMATCH}: "
                                                   + "; ".join(mismatches))

        ## ---- Firmware Events against pyflic's, under a ratio Paradigm's Main ----
        firmware_total = pyflic_total = 0
        ratio_seen = False
        disagreements: list[str] = []
        for t in sorted(triggers_seen):
            counted = np.array([t in mm[1] and mm[0] in op.RATIO_PARADIGMS for mm in modes]
                               + [False], dtype=bool)
            main = state.ratio_rows(t) & ~phase_a[:, t - 1] & counted[window_of]
            if not main.any():
                continue
            ratio_seen = True
            fe = sum(state.firmware_events_between(t, a, b) for a, b in _true_runs(main))
            ## Like with like (decision 16): no Firmware Event starts while
            ## the well's own Activation is lit, so a pyflic feeding event
            ## whose first sample lies in its light period's rows is left
            ## out.  Not in the onset row: the Event that met the Requirement
            ## starts there, in the millisecond the light comes on, and the
            ## firmware counts it.
            own = al.activation_rows(book, t, n, onset_row=False)
            pe = int(((feeding[:, t - 1] > 0) & main & ~own).sum())
            firmware_total += fe
            pyflic_total += pe
            if max(fe, pe) < MIN_FEEDING_EVENTS:
                continue
            share = abs(fe - pe) / fe if fe else float("inf")
            if share > settings.event_disagreement_fraction:
                disagreements.append(
                    f"W{t} {fe} Firmware Events, {pe} pyflic feeding events"
                    + (f" ({100 * share:.0f}% of the Firmware Events)" if fe else ""))
        if disagreements:
            flags.append(EVENT_DISAGREEMENT)
            reasons[EVENT_DISAGREEMENT] = (f"{EVENT_DISAGREEMENT} under the ratio "
                                           f"Paradigm's Main phase: "
                                           + "; ".join(disagreements))

        ## ---- each Paradigm against its own rule (stream S1) ----
        rule = _group_rule_checks(
            book=book, unit_pos=unit_pos, unit_window=unit_window, windows=windows,
            modes=modes, state=state, activity=activity, on_ms=on_ms, starts=starts,
            settings=settings, record_ms=int(round(record_ms)), judged=section is not None)
        flags.extend(rule.flags)
        reasons.update(rule.reasons)
        notes.extend(rule.notes)

        ## Light where nothing is judged is noted, not judged.
        unscheduled_ms = float(g_ms[~scheduled].sum())
        if unscheduled_ms >= 10_000.0:
            notes.append(f"{unscheduled_ms / 1000.0:.1f} s of light under an unknown "
                         f"instruction or a Paradigm this check cannot judge, not judged")
        if section is not None:
            unknown_thresholds_ms = float(sum(
                g_ms[w.i0:w.i1].sum() for w, (mode, _t, _h) in zip(windows, modes)
                if mode == NOT_JUDGED and w.kind == "judged"))
            if unknown_thresholds_ms >= 10_000.0:
                notes.append(f"{unknown_thresholds_ms / 1000.0:.1f} s of light in Program "
                             f"Intervals whose thresholds Program.txt does not print and "
                             f"whose group counted no Activation, not judged")

        ## ---- this group's Activations, one event each ----
        unit_judged = np.bincount(owned_unit, weights=judged_ms[owned_rows], minlength=m)
        unit_unexplained = np.bincount(owned_unit, weights=unexplained_ms[owned_rows],
                                       minlength=m)
        last_seen: dict[int, int] = {}
        last_activity: dict[tuple, np.ndarray] = {}
        unexplained_events = 0
        for i in range(m):
            r, well, k = int(unit_row[i]), int(unit_well[i]), int(unit_window[i])
            e = int(unit_end[i])
            mode, trig, _hold = modes[k]
            cls = int(unit_cls[i])
            w = windows[k]
            timed = ledger_fields(int(unit_pos[i]))
            timed["EndedBy"] = rule.ended_by[i]
            ## The ledger's Phase: the row where a well leaves Acclimation
            ## reads M though an Acclimation Activation can be counted on it.
            in_a = bool(timed.pop("_phase_a", unit_in_a[i]))
            requirement = np.nan
            events = np.nan
            if in_a:
                ## Acclimation counts Activations, not Firmware Events: a fly
                ## back within the Event gap after the light went off gets a
                ## new Activation without a new Event (DFMV3_V2 024b692), and
                ## Progress counts neither.  So no Event count is given.
                pass
            elif mode in op.RATIO_PARADIGMS:
                ## The Requirement met is on the row before; the first row of
                ## a new Paradigm shows none, so the program's series says it.
                met = state.requirement_met(well, r)
                if met is None and starts[r] and w.paradigm is not None:
                    met = w.paradigm.requirement_for(1)
                requirement = met if met is not None else np.nan
                ## The Firmware Events since this well's previous Activation
                ## under the same Paradigm, read from Progress.
                i0 = max(last_seen.get(well, -1) + 1, int(segment_start[r]))
                counted = state.firmware_events_between(well, i0, r + 1)
                events = counted if counted else requirement
            last_seen[well] = r
            explained: bool | None = None
            un_sec = overrun = np.nan
            if cls == _UNIT_FIXED:
                explained = bool(unit_activity[i])
            elif cls == _UNIT_WINDOW:
                explained = (bool(unit_judged[i] - unit_unexplained[i] > 0)
                             if unit_judged[i] > 0 else bool(unit_activity[i]))
            if explained is not None:
                un_sec = float(unit_unexplained[i]) / 1000.0
                if not explained:
                    unexplained_events += 1
                d = decays[k]
                if trig not in last_activity:
                    act = any_activity(trig)
                    last_activity[trig] = (np.maximum.accumulate(
                        np.where(act, np.arange(n), -1)) if n else np.zeros(0, dtype=int))
                end = max(e, r + 1)
                lo, hi = max(0, r - d - tol), min(n, end + tol)
                last = int(last_activity[trig][hi - 1]) if hi > 0 else -1
                overrun = (max(0.0, (end - (last + d + tol + 1)) / sps) if last >= lo
                           else (end - r) / sps)
            event_rows.append({
                "DFM": inp.dfm_id, "Group": label, "Event": i + 1, "Well": well,
                "OnsetMin": float(minutes[r]),
                "EndMin": float(minutes[max(e, r + 1) - 1]) + 1.0 / sps / 60.0,
                "DurationSec": float(unit_ms[i]) / 1000.0,
                "Interval": (w.program_interval if w.program_interval is not None
                             else np.nan),
                "Mode": mode, "Phase": "A" if in_a else "M",
                "RequirementMet": requirement, "FirmwareEvents": events,
                "UnexplainedSec": un_sec, "Explained": explained, "OverrunSec": overrun,
                "Source": EVENT_SOURCE_ACTIVATIONS, **timed,
                "Family": pfam.family_of(mode, acclimation=in_a) or np.nan,
            })

        if n:
            ## From the first minute of the rows, so a Slice's timeline does
            ## not open with empty minutes before its start.
            first_minute = int(max(0.0, np.floor(minutes[0])))
            minute_bin = np.clip(np.floor(minutes), first_minute, None).astype(int) - first_minute
            size = int(minute_bin.max()) + 1

            def per_minute(ms: np.ndarray) -> np.ndarray:
                return np.bincount(minute_bin, weights=ms, minlength=size) / 1000.0

            timelines[label] = pd.DataFrame({
                "Minute": np.arange(size, dtype=float) + first_minute,
                "LitSec": per_minute(g_ms),
                "ExplainedSec": per_minute(judged_ms - unexplained_ms),
                "UnexplainedSec": per_minute(unexplained_ms),
                "LitWhileOffSec": per_minute(off_ms),
                "NotJudgedSec": per_minute(np.clip(g_ms - judged_ms - off_ms, 0, None)),
                "ContactWithoutActivitySec": per_minute(np.where(evidence & owned, g_ms,
                                                                 0.0)),
            })

        chambers = sorted({inp.chamber_of_well[w] for w in wells
                           if w in inp.chamber_of_well})
        role_text = ", ".join(f"{c} {roles[c]}" for c in chambers if roles.get(c))
        group_rows.append({
            "DFM": inp.dfm_id, "Generation": int(inp.generation), "Group": label,
            "Linkage": label if linkage_source == "program" else np.nan,
            "Wells": op.wells_label(wells),
            "TriggerWells": op.wells_label(triggers_seen),
            "Chambers": _chambers_label(chambers), "Roles": role_text or np.nan,
            "_chambers": tuple(chambers), "_wells": tuple(wells), "Treatment": "",
            "Program": status, "Paradigm": ", ".join(paradigms),
            "LitSec": float(g_ms.sum()) / 1000.0, "JudgedLitSec": judged_total / 1000.0,
            "UnexplainedSec": unexplained_sec,
            "UnexplainedFraction": fraction, "UnexplainedOnsetMin": onset,
            "LightEvents": int(m), "UnexplainedEvents": unexplained_events,
            "FeedingEvents": feeding_events,
            "ClosedLoopFeedingEvents": closed_events if closed_events else np.nan,
            "UnlitFeedingEvents": unlit_events if closed_events else np.nan,
            "FirmwareEvents": firmware_total if ratio_seen else np.nan,
            "PyflicEvents": pyflic_total if ratio_seen else np.nan,
            "AcclimationActivations": acclimation_count if acclimation_runs else np.nan,
            "AcclimationExpected": (acclimation_expected
                                    if acclimation_runs and expected_known else np.nan),
            "OpenLoopLitFraction": open_fraction,
            "LitWhileOffSec": off_total / 1000.0,
            "ContactSec": contact_ms / 1000.0,
            "ContactWithoutActivitySec": without_ms / 1000.0,
            "FirmwareEventsWithoutActivity": events_without,
            "UnexplainedWithContactSec": with_contact_ms / 1000.0,
            **rule.columns, rc.INTERVAL_LAG_COLUMN: interval_lag,
            rc.RESTARTS_COLUMN: restart_count,
            "EmulationApproximate": np.nan,
            "LikelyCause": cause, "_flags": flags, "_notes": notes, "_reasons": reasons,
            "Excluded": False,
        })

    ## ---- the DFM as a whole ----
    lit_any = int((on_ms > 0).any(axis=1).sum()) if n else 0
    _no_light_check(dfm_flags, dfm_notes, setting, lit_any,
                    (section is None and not simple) or expects_light)
    if not group_rows:
        group_rows.append({**_empty_group_row(inp, status),
                           rc.INTERVAL_LAG_COLUMN: interval_lag,
                           rc.RESTARTS_COLUMN: restart_count})
    _finish_groups(group_rows, dfm_flags)
    baselines = _baseline_summary(inp, recorded, program, dfm_notes)
    return DFMOptoResult(dfm_id=inp.dfm_id, setting=setting, program_status=status,
                         groups=group_rows, intervals=interval_rows, events=event_rows,
                         flags=dfm_flags, notes=dfm_notes, decay_samples=decay_by_sample,
                         lit_samples=lit_any, timelines=timelines,
                         generation=int(inp.generation), baselines=baselines)


@dataclass
class _RuleOutcome:
    """One Linkage Group's rule checks (:func:`_group_rule_checks`): its
    columns, the flags that fired with their reasons, its notes, and each
    Activation's ``EndedBy``."""

    columns: dict[str, Any]
    flags: list[str]
    reasons: dict[str, str]
    notes: list[str]
    ended_by: np.ndarray


#: (check's flag, its QC column), in the order the flags are listed.
_RULE_COLUMNS: tuple[tuple[str, str], ...] = (
    (rc.SHORT_ACTIVATIONS, rc.SHORT_COLUMN),
    (rc.MAX_ON_EXCEEDED, rc.OVER_MAX_ON_COLUMN),
    (rc.CUT_OFFS, rc.CUT_OFFS_COLUMN),
    (rc.WRONG_ACTIVATION_LENGTH, rc.WRONG_LENGTH_COLUMN),
    (rc.LOCKOUT_BROKEN, rc.LOCKOUT_COLUMN),
    (rc.REQUIREMENT_SERIES_WRONG, rc.SERIES_COLUMN),
    (rc.CONTACT_WITHOUT_ACTIVATION, rc.MISSED_CONTACTS_COLUMN),
    (rc.NON_FEEDING_DARK, rc.NON_FEEDING_DARK_COLUMN),
)


def _group_rule_checks(*, book: pd.DataFrame, unit_pos: np.ndarray, unit_window: np.ndarray,
                       windows: list[_RunWindow], modes: list[tuple], state: ParadigmState,
                       activity: np.ndarray, on_ms: np.ndarray, starts: np.ndarray,
                       settings: OptoQCSettings, record_ms: int, judged: bool
                       ) -> _RuleOutcome:
    """Each Paradigm a Linkage Group's Trigger Wells ran, checked against its
    own rule (:mod:`pyflic.base.rule_checks`, stream S1).

    The group's Activations (*unit_pos*: each one's ledger row in *book*,
    *unit_window*: its window) are taken per Paradigm, over every window that
    ran it with the group's Trigger Wells, and each check is made on them.
    A column sums its check over the Paradigms; it is ``NA`` where no
    Paradigm of the group has the rule, and every column is ``NA`` when
    nothing is *judged* (no ``Program.txt`` section: the settings are not
    known).  ``EndedBy`` is given for every Activation the ledger has."""
    m = unit_pos.size
    tol = int(settings.activation_tolerance_ms)
    index = np.asarray(state.index)
    ended = np.full(m, np.nan, dtype=object)
    columns: dict[str, Any] = {c: np.nan for c in rc.RULE_QC_COLUMNS
                               if c not in (rc.INTERVAL_LAG_COLUMN, rc.RESTARTS_COLUMN)}
    flags: list[str] = []
    reasons: dict[str, str] = {}
    notes: list[str] = []

    def well_rows(units: np.ndarray, paradigm: op.Paradigm | None) -> None:
        """``EndedBy`` of *units*, which share *paradigm*."""
        units = units[unit_pos[units] >= 0]
        if not units.size:
            return
        order = np.argsort(unit_pos[units], kind="stable")
        units = units[order]
        rows = book.iloc[unit_pos[units]]
        wells = sorted({int(w) for w in rows["Well"]})
        ended[units] = rc.ended_by(rows, paradigm,
                                   activity={w: activity[:, w - 1] for w in wells},
                                   index=index, tol_ms=tol, record_ms=record_ms)

    if not judged:
        well_rows(np.arange(m), None)
        return _RuleOutcome(columns, flags, reasons, notes, ended)

    ## The group's Paradigms: those of its judged windows with Trigger Wells.
    by_name: dict[str, tuple[op.Paradigm, list[int]]] = {}
    for k, w in enumerate(windows):
        if w.kind == "judged" and w.paradigm is not None and modes[k][1]:
            by_name.setdefault(w.paradigm.name, (w.paradigm, []))[1].append(k)
    found: dict[str, list[rc.CheckResult]] = {flag: [] for flag, _c in _RULE_COLUMNS}
    delays: list[np.ndarray] = []
    relit = False
    in_paradigm = np.zeros(m, dtype=bool)
    n = len(state)
    for name, (p, ks) in by_name.items():
        units = np.flatnonzero(np.isin(unit_window, ks))
        in_paradigm[units] = True
        well_rows(units, p)
        rows = book.iloc[np.sort(unit_pos[units][unit_pos[units] >= 0])]
        triggers = sorted({t for k in ks for t in modes[k][1]})
        ## Per Trigger Well, the rows where it ran this Paradigm as one.
        ran = {t: np.zeros(n, dtype=bool) for t in triggers}
        for k in ks:
            for t in modes[k][1]:
                ran[t][windows[k].i0:windows[k].i1] = True
        acts = {t: activity[:, t - 1] for t in triggers}
        found[rc.SHORT_ACTIVATIONS].append(rc.short_activations(
            rows, p, tol_ms=tol, fail_fraction=settings.activation_length_fail_fraction,
            record_ms=record_ms))
        found[rc.MAX_ON_EXCEEDED].append(rc.max_on_exceeded(rows, p, tol_ms=tol,
                                                            record_ms=record_ms))
        found[rc.CUT_OFFS].append(rc.cut_offs(rows, p, tol_ms=tol, record_ms=record_ms))
        found[rc.WRONG_ACTIVATION_LENGTH].append(rc.wrong_activation_length(
            rows, p, tol_ms=tol, fail_fraction=settings.activation_length_fail_fraction,
            record_ms=record_ms))
        found[rc.LOCKOUT_BROKEN].append(rc.lockout_violations(
            rows, p, tol_ms=tol, fail_fraction=settings.lockout_fail_fraction,
            record_ms=record_ms))
        found[rc.REQUIREMENT_SERIES_WRONG].append(rc.requirement_series(rows, p))
        if p.type == op.FIXED_INTERVAL:
            ## Free means free of any of the well's Activations, so all of them.
            mine = book[book["Well"].isin(triggers)]
            main = {t: ran[t] & ~state.phase_a[:, t - 1] for t in triggers}
            found[rc.CONTACT_WITHOUT_ACTIVATION].append(rc.missed_contacts(
                mine, p, acts, index, main, tol_ms=tol,
                warn_fraction=settings.missed_contact_fraction, record_ms=record_ms))
        if p.type == op.NON_FEEDING:
            found[rc.NON_FEEDING_DARK].append(rc.non_feeding_dark(
                {t: on_ms[:, t - 1] for t in triggers}, acts, index, ran, starts, p,
                tol_ms=tol, warn_fraction=settings.non_feeding_dark_fraction,
                record_ms=record_ms))
            delays.append(rc.relight_delays(rows, acts, index, record_ms=record_ms))
            relit = True
    ## Activations under no known Paradigm (none, with a section) keep the
    ## ledger's words only.
    well_rows(np.flatnonzero(~in_paradigm), None)

    for flag, column in _RULE_COLUMNS:
        result = rc.combine(found[flag])
        if result is None:
            continue
        columns[column] = result.value
        if result.fired:
            flags.append(flag)
            reasons[flag] = result.reason
        if result.note:
            notes.append(result.note)
    if relit:
        relight = rc.relight_delay(np.concatenate(delays) if delays else np.zeros(0))
        columns[rc.RELIGHT_DELAY_COLUMN] = relight.value

    ## Not judged: Activations whose light is not their own, or that the
    ## program stopped, counted once over the group.
    rows = book.iloc[np.sort(unit_pos[in_paradigm & (unit_pos >= 0)])]
    if len(rows):
        overlap = rows["Overlap"].to_numpy(dtype=bool)
        cut = ~overlap & ~rc.judgeable(rows)
        if overlap.any() or cut.any():
            why = [f"{int(overlap.sum())} merged with other light" if overlap.any() else "",
                   f"{int(cut.sum())} cut short by an Interval change, a gap in Index or the "
                   f"end of the data" if cut.any() else ""]
            notes.append(f"rule checks: {int(overlap.sum() + cut.sum())} of {len(rows)} "
                         f"Activations not judged ({', '.join(w for w in why if w)})")
    return _RuleOutcome(columns, flags, reasons, notes, ended)


def _yoking_problems(state: ParadigmState, wells: tuple[int, ...],
                     windows: list[_RunWindow], on_ms: np.ndarray, lit: np.ndarray,
                     settings: OptoQCSettings) -> list[str]:
    """What breaks a Linkage Group's yoking: its wells' ``OnMs`` differing by
    more than 1 ms in more than ``opto_yoking_fail_fraction`` of its lit
    records, or an ``Off`` well linked to a trigger counting Activations."""
    if len(wells) < 2:
        return []
    out: list[str] = []
    idx = [w - 1 for w in wells]
    lit_records = int(lit.sum())
    if lit_records:
        spread = on_ms[:, idx].max(axis=1) - on_ms[:, idx].min(axis=1)
        differ = int((lit & (spread > 1)).sum())
        if differ / lit_records > settings.yoking_fail_fraction:
            out.append(f"its wells' OnMs differ by more than 1 ms in {differ} of "
                       f"{lit_records} lit records ({100.0 * differ / lit_records:.1f}%)")
    counted: dict[int, int] = {}
    for w in windows:
        line = w.interval
        if line is None or w.kind != "judged" or w.paradigm is None \
                or w.paradigm.type == op.OPEN_LOOP or not line.trigger_wells(wells):
            continue
        for well in line.off_wells(wells):
            got = state.activations_between(well, w.i0, w.i1)
            if got:
                counted[well] = counted.get(well, 0) + got
    for well, got in sorted(counted.items()):
        out.append(f"W{well} is Off and linked to a trigger well but counted {got} "
                   f"Activation(s)")
    return out


def _acclimation_counts(state: ParadigmState, well: int, windows: list[_RunWindow],
                        window_of: np.ndarray, starts: np.ndarray, record_ms: float
                        ) -> list[tuple[int, int | None, int, int]]:
    """``(Activations, the Paradigm's Acclimation or None, i0, i1)`` for each
    of *well*'s completed Acclimation runs: rows in Phase ``A`` that end in
    ``M`` under the same Paradigm, with the well still a level threshold.  A
    run cut by the end of the data, a new Paradigm, or an ``On`` / ``Off``
    Interval (which freezes the well, so the file reads ``M``) never
    completed, and is not counted: that is never completing Training."""
    n = len(state)
    out: list[tuple[int, int | None, int, int]] = []
    for a0, a1 in state.acclimation_runs(well):
        if a1 >= n or starts[a1]:
            continue
        cuts = np.flatnonzero(starts[a0 + 1:a1]) + a0 + 1
        s0 = int(cuts[-1]) if cuts.size else int(a0)
        k_end, k_start = int(window_of[a1]), int(window_of[s0])
        if k_end < 0 or k_start < 0:
            continue
        line = windows[k_end].interval
        if line is not None and not op.is_trigger(line.threshold(well)):
            continue
        paradigm = windows[k_start].paradigm
        expected = int(paradigm.acclimation) if paradigm is not None else None
        count = state.activations_between(well, s0, a1)
        ## An Acclimation shorter than a record can start and end in one: its
        ## Activation is then counted on the first row in M.
        hold = paradigm.activation_ms if paradigm is not None else None
        if expected is not None and count < expected and hold is not None \
                and hold < record_ms:
            count += min(state.activations_between(well, a1, a1 + 1), expected - count)
        out.append((int(count), expected, s0, int(a1)))
    return out


# ---------------------------------------------------------------------------
# The program as a table
# ---------------------------------------------------------------------------

def program_rows(program: op.OptoProgram, dfm_ids=None) -> list[dict]:
    """One row per interval of every DFM section (or of *dfm_ids*): the
    Paradigm by name and type with its settings, ``NaN`` where the type has no
    such setting (:data:`OPTO_PROGRAM_COLUMNS`)."""
    rows: list[dict] = []
    ids = sorted(program.dfms) if dfm_ids is None else [d for d in sorted(dfm_ids)
                                                        if d in program.dfms]
    for dfm_id in ids:
        section = program.dfms[dfm_id]
        for iv in section.intervals:
            p = iv.paradigm
            legacy = p.legacy
            row = {
                "DFM": dfm_id, "Interval": iv.index,
                "Start": iv.start.strftime("%Y-%m-%d %H:%M:%S") if iv.start else "",
                "DurationMin": float(iv.duration_min),
                "ProgramType": section.program_type, "Paradigm": p.name,
                "ParadigmType": p.type_name, "Generated": bool(p.generated),
                "Dark": bool(iv.dark), "FrequencyHz": int(p.frequency_hz),
                "PulseWidthMs": int(p.pulse_width_ms),
            }
            for column, attr in _PROGRAM_SETTING_COLUMNS:
                value = getattr(p, attr)
                row[column] = np.nan if value is None else value
            ## Acclimation is a count every Paradigm has (0 when unused), but
            ## only the types that take it print it.
            row["Acclimation"] = (int(p.acclimation) if p.type in op.ACCLIMATION_PARADIGMS
                                  or p.acclimation else np.nan)
            row.update({
                "Linkage": op.linkage_label(section),
                "TriggerWells": op.wells_label(iv.trigger_wells()),
                "OpenLoopWells": op.wells_label(iv.open_wells()),
                "LegacyF": int(legacy.frequency) if legacy is not None else np.nan,
                "LegacyP": int(legacy.pulse_width) if legacy is not None else np.nan,
            })
            rows.append(row)
    return rows


def program_frame(program: op.OptoProgram | None, dfm_ids=None) -> pd.DataFrame:
    """:func:`program_rows` as a table (``qc/opto/opto_program.csv``), its
    whole-number columns nullable integers so that a setting a type lacks
    reads ``NA`` beside whole numbers, not a float."""
    rows = [] if program is None else program_rows(program, dfm_ids)
    frame = pd.DataFrame(rows, columns=list(OPTO_PROGRAM_COLUMNS))
    for column in OPTO_PROGRAM_INTEGER_COLUMNS:
        frame[column] = pd.array(pd.to_numeric(frame[column]), dtype="Int64")
    return frame


def paradigms_in(program: op.OptoProgram) -> set[str]:
    """Every mode a Trigger Well runs under anywhere in *program*, and open
    loop for an interval whose only lit wells are ``On``."""
    out: set[str] = set()
    for section in program.dfms.values():
        for iv in section.intervals:
            if iv.trigger_wells():
                out.add(iv.paradigm.mode)
            elif iv.open_wells():
                out.add(op.OPEN_LOOP)
    return out
