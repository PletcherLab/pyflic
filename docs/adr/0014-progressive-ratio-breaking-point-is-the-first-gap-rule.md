---
status: accepted
---

# The Progressive Ratio breaking point is the first-gap rule, with censoring

A progressive ratio's breaking point is the last requirement the animal met
before it stopped working, and "stopped" needs a definition. The R port had
none: it tabulated licks per light period and left the reading to the eye. The
first rewrite counted every Test **Light Event** to the end of the recording.
We now use the classical **first-gap rule**. Responses are read in order from
the **Chamber Group**'s training end, and the fly is taken to have stopped at
the first pause longer than `pr_break_gap_min` (Δt, default 120 minutes).

- The **Breaking Point** is the number of lick-backed Test Light Events the
  **Paired** fly completed before that pause. `BreakMin` is the minute, since
  training end, of the last one.
- A group with no such pause before its Test window ended is **Censored**: its
  count is a lower bound, not a measurement.
- The same rule on either fly's Sucrose Well feeding events gives **Sucrose
  Persistence**, the persistence measure the **Yoked** fly has too.

## Decisions

- **Paired only.** The Yoked fly shares its partner's light circuit, so any
  rule built on light events returns its partner's number, and a paired-minus-
  yoked breaking point is zero by construction. Paired versus yoked is asked
  of feeding metrics instead, Sucrose Persistence among them.
- **The unit is the count of light events**, the ordinal of the last ratio
  completed. The firmware's lick schedule is not recorded, and pyflic
  under-counts licks against the firmware, so a requirement in licks would be
  an estimate of an estimate. `LargestRequirement` stays in the table as a
  descriptive column, with no statistics.
- **Lick-free Light Events are removed before the rule runs.** One neither
  counts nor ends a pause: a light the fly did not lick for is no evidence it
  was still responding.
- **The gaps run from training end to the Test end.** The first gap is
  training end to the first response, so a fly whose first Test response came
  more than Δt after training end has a breaking point of 0. The last gap is
  the last response to the end of the Test window, so a fly that stopped
  inside its window is observed rather than censored. A gap must exceed Δt;
  one of exactly Δt does not end the count.
- **Δt is a design constant, `pr_break_gap_min`, default 120 minutes.** Every
  Member of a Project is analysed under one rule. `summary.txt` tabulates
  each group's count at 60, 120 and 240 minutes beside the configured value,
  because the number moves with Δt.
- **`pr_test_window_min` caps every group's Test window**, off by default.
  Training ends at a different time in every group, so Test phases differ in
  length (540 to 1335 minutes in the first real dataset). The cap is applied
  before the rule, and censoring is judged against the capped window.
- **No clock.** The data carry date and time but no photoperiod, and the rule
  does not know the night. A long pause at night ends a count like any other.
- **Statistics, one observation per Chamber Group.** Treatments are compared
  as for any metric: Welch's t or Tukey HSD, and the mixed model when members
  are pooled, with a censored count entered as observed. Beside them, a
  pairwise **log-rank** test on the ratio reached treats a censored count as
  the lower bound it is, and a Kaplan-Meier **still-responding curve** shows
  the fraction of paired flies that reached each ratio, per treatment.
- **The Paired-Yoked Difference is also tested against zero**, per treatment:
  the paired t-test, which is a one-sample t-test on the differences, with the
  Wilcoxon signed-rank test beside it, and the mixed model's intercept when
  members are pooled. Every earlier test asked whether treatments differ in
  the difference; none asked whether paired and yoked flies differ at all.
- **Sucrose Persistence shares Δt and the Test window** with the breaking
  point. A difference in which either fly is censored is kept and flagged
  (`dPersistCensored`) rather than dropped: in a recording that ends during a
  feeding peak, dropping them would remove most groups.

## Considered options

- **The session-end rule with censoring.** Count every Test event, and let Δt
  decide only whether the fly had stopped by the end of the recording. In the
  first real dataset three of four groups resumed at a higher requirement
  after a pause longer than 100 minutes, and under this rule those later
  ratios would count. Not chosen: the first-gap rule is the field's
  definition, and its cost — a pause at night can end a count — is made
  visible by the sensitivity table and the per-DFM figure rather than
  corrected.
- **Clock-aware gaps** that skip the dark phase. No photoperiod is recorded,
  so this would need a new design key with QC of its own. Not chosen.
- **A yoked breaking point.** Counting the Yoked fly's licks against the
  requirement its partner met would score natural feeding against a schedule
  the Yoked fly never experienced. Rejected.
- **Counting lick-free Light Events, or letting them end a pause.** The first
  ties the number to the firmware alone, the second lets a sensor-driven light
  split a real pause in two. Rejected.
- **Capping every group at the member's shortest Test phase.** The cap would
  depend on which groups survived the light QC and would differ between
  Members. Rejected for the explicit, Design-owned `pr_test_window_min`.

## Consequences

- New outputs: `pr_breaking_point.csv`, one row per Chamber Group; the
  Combined Analysis's `<project>_BreakingPoint.csv`; `pr_still_responding.png`;
  a `Counted` column in the Light Event Ledger (`pr_light_events.csv`);
  `PersistA` and `PersistACensored` on every per-chamber summary row, blank on
  Training rows; `dPersistA` and `dPersistCensored` in `paired_yoked_diff.csv`.
- The breaking point figure marks the break on both chambers of a group, greys
  every onset past it, and shows lick-free events as hollow rings.
- The experiment report, the Project Report and the Stats text carry the
  log-rank column and the tests against zero; the AI narrative, which reads
  the Stats text, sees them too.
- A Member analysed before this change has no `pr_breaking_point.csv`. The
  Project Report says so and asks for a re-run instead of deriving a number
  from `pr_light_qc.csv`, as it used to.
