---
status: accepted
---

# pyflic reads the MCU's Program.txt, and every optogenetic experiment gets a light QC

The DFM firmware decides when an Optolid LED is on, every millisecond, from a
program the stand-alone MCU loads: a threshold per well, a linkage that ORs
wells together, and five optogenetic parameters (frequency, pulse width,
decay, delay, max-time-on) whose combination selects a paradigm. The DFM
CSVs record only the outcome, `OptoCol1`, the LEDs actually lit. pyflic
counts licks afterwards from the baselined signal. The Progressive Ratio
light QC (ADR-0013 addendum) showed that the two disagree exactly when an
assay fails: a drifting Sucrose Well is touched for the firmware and flat for
pyflic, so the light runs with no lick behind it. Nothing about that failure
is specific to progressive ratio, and without the program pyflic cannot tell
a light that should have been on from one that should not.

So pyflic now reads the program, as the MCU exports it, and judges the light
of every optogenetic experiment, whatever its type. Terms are in
`CONTEXT.md`: *Opto Program*, *Linkage Group*, *Trigger Well*, *Explained
Light*, *Emulated Trigger*.

## Decisions

- **The export, not the authored program.** The MCU writes the program it
  ran back out as `Program.txt`, echoing each interval as it parsed it
  (`F:`, `P:`, `D:`, `L:`, `M:` — frequency, pulse width, decay, delay,
  max-time-on). That is a record of what ran, including the resolved start
  time; the `[General]` / `[DFM]` file is only the MCU's input. The user
  copies the export into `data/`. Exactly one per recording: more than one
  stops the load, because which describes the run is not pyflic's guess.
- **Optional, and tolerant.** Without the file the check runs with less
  knowledge. An unknown line is skipped and listed; a DFM section that cannot
  be read is dropped and that DFM judged as if there were no program; a file
  with no readable section is set aside with the reason. MCU/DFM version 2.0
  semantics throughout; the frequency's acclimation-event bits and the pulse
  width's non-feeding bit are decoded.
- **One setting, `optogenetics: auto | yes | no`, default `auto`.** A Design
  key (ADR-0005), so a Project states it once. `auto` covers every DFM with a
  program section or any lit LED; `yes` covers every DFM and fails one that
  is never lit; `no` covers none. **The per-experiment override lives on the
  DFM entry**, not in a Member's `global:`: whether a lid was lit is a fact
  about the recording, like the rest of `dfms:`, which stays free inside a
  Project, whereas a Member's `global:` may not differ from the Design.
- **The Linkage Group is the unit of judgement.** Its light is one circuit,
  lit when any member triggers, so it is judged once and its verdict passed
  to every chamber it touches (the worst verdict, the union of flags).
- **Explained Light is a lit sample with a feeding lick or a tasting sample
  in a Trigger Well from the decay plus a tolerance before it to the
  tolerance after it.** Tasting counts because the firmware decides on
  contact at millisecond resolution and a brief touch reads below pyflic's
  feeding threshold: on the first real dataset a healthy group reads 56%
  unexplained on feeding licks alone and 0% with tasting. The tolerance,
  `opto_decay_tolerance_samples` (2), absorbs the firmware's 1 ms decisions
  against 5 Hz samples.
- **The verdict is the unexplained share of the lit time judged:** a warning
  at `opto_unexplained_warn_fraction` (10%), a failure at
  `opto_unexplained_fail_fraction` (30%), neither until the group has
  `opto_unexplained_min_sec` (30 s) of unexplained light. The floor was added
  in implementation, beyond the agreed fractions: without it a few stray
  samples in a group lit for seconds would fail it. `UnexplainedOnsetMin` is
  the first 30-minute window at or over the warning fraction.
- **Each interval is judged under its own mode.** The program's intervals are
  laid out by clock time (Linear, Repeating, Circadian, Constant) and a
  group's mode comes from its members' thresholds: any 0 is open loop (lit at
  least `opto_open_loop_min_lit_fraction`, 95%, or a warning), all -1 is off
  (any light fails), otherwise the interval's paradigm. Under closed loop the
  reverse is checked too: at least `opto_unlit_feeding_fraction` (50%) of the
  feeding bouts at its trigger wells never lighting the group is a warning.
  Non-feeding activation is noted and not judged: no rule for it has been
  validated. Light within the decay of an interval boundary belongs to the
  interval before.
- **The Emulated Trigger names the cause.** With a program, the firmware's
  test is re-run on the raw signal: raw minus the mean of the first ten
  seconds (the firmware's baseline, 50 samples at 5 Hz, captured once),
  against the interval's threshold, or raw alone under `Baseline: No`.
  Contact it sees and pyflic does not is the drifting-baseline signature;
  `LikelyCause` reads *drifting baseline or sustained contact* when most
  unexplained light fell during such contact, else *hardware, linkage or
  program mismatch*. Data beginning more than two seconds after the program
  leave the baseline window unrecorded, and the emulation is marked
  approximate.
- **Program against data.** `OptoFreq`, `OptoPW` and `Dark`, and a start more
  than a minute from the program's, are compared and warn when they differ;
  a DFM with no usable section warns and is judged without one; data running
  past the End Time are noted.
- **Without a program, linkage is inferred and nothing fails.** Wells whose
  light is identical over the whole recording are one group, every member
  counts as a trigger well, and the decay is `opto_default_decay_ms` (1000).
  The agreed fallback was per-well judgement; that was changed in
  implementation because it fails every yoked or linked well, whose light is
  its partner's. Unexplained light then warns but cannot fail, since an
  open-loop schedule is indistinguishable from a stuck light.
- **Failures are flagged, not excluded, by default.**
  `exclude_failed_opto_chambers` (false) sends a failed group's chambers
  through auto-removal. A faulty light does not invalidate the feeding
  record. Progressive Ratio keeps `exclude_failed_pr_groups` (true) for its
  own checks, so its exclusions do not change.
- **Progressive Ratio builds on it.** A DFM entry may omit `paired_chambers`
  when the program names the trigger wells; given both, the config wins and
  a disagreement is reported. A Light Event is lick-free only with no
  credited lick *and* no Sucrose Well lick or touch within its decay, so a
  light the fly touched for neither counts against it nor is dropped from the
  breaking point. On the first real dataset no PR verdict changes: two
  lick-free events of a healthy group become explained. The general check
  fails one group the PR check passes — four hours of training light with no
  lick — and flags it.
- **Outputs.** `qc/opto/` holds the verdicts per Linkage Group, the per-
  interval and per-event tables, the program as read and one figure per DFM;
  `summary.txt` and the experiment report gain a section for every type; the
  Project stacks each Member's verdicts into `<project>_LightQC.csv` beside
  the PR ones, marked by a `Source` column; the Hub's QC panel groups both
  checks under *Optogenetics*.

## Considered options

- **Require `Program.txt`.** Not chosen: every existing optogenetic recording
  lacks one, and the check without it still catches a light that ran for
  minutes with no lick.
- **Per-well judgement without a program.** The agreed fallback, rejected in
  implementation as above: it calls a yoked fly's light unexplained.
- **Feeding licks alone explain light.** Rejected on the first real dataset,
  where it fails healthy groups.
- **A member-level `global.optogenetics` that may differ from the Design.**
  Rejected: it would be the one `global:` key a Member may contradict, and
  the per-DFM key already says the same thing at the level the fact lives.
- **Read the authored program too.** Deferred until someone has only that
  file; it does not record what ran.
