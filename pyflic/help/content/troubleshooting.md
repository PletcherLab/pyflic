# Troubleshooting

## Loading

### "flic_config.yaml not found in experiment directory"

The folder you chose is not an Experiment Directory. It needs a `flic_config.yaml` at its
top level — not inside `data/`. The Hub says "is not a Batch, a Project, or an Experiment
Directory" for the same reason. See
[Setting up an experiment directory](getting-started-project.md).

### A DFM in my config has no data

The load stops with *DFM data file(s) not found for id=3*. CSV filenames must match the DFM
numbers in your configuration. A DFM keyed `3` needs `DFM3_0.csv` (v3) or `DFM_3.csv` (v2)
in `data/`. Check for a leading zero, a different
separator, or a file left in the folder's root instead of `data/` — inside a Project the Hub
shows that as a blocked member, **unfiled recording**, and offers to file it.

### "'params.chamber_size' is owned by experiment_type"

A typed config states `chamber_size` (or `chamber_layout`), which the Experiment Type now
owns. Delete the key; the value is derived. An old `experiment_type: two_well` or
`single_well` fails the same way and names its replacement, `chamber_layout:`. Run
`pyflic lint` on the folder to list every config that needs this. See
[Experiment types](concepts-experiment-types.md#migrating).

### "requires well_names for A, B"

Hedonic and Progressive Ratio experiments must name both wells under `global.well_names`.

### "Unknown experiment_type"

The config names a type pyflic cannot find: it is not shipped, there is no
`experiment_type.yaml` copy of it beside the data, and the lab's type library has none. The
message lists the types it knows. Check the spelling, point the Experiment Type Editor at
your lab's library, or copy the type's file in. See
[Where types live](concepts-experiment-types.md#where-types-live).

### "was read from the lab library: the data carries no copy of it"

A note, not an error: the type was found in your lab's library, but the data has no copy of
it, so another machine could not analyse it the same way. Save the Project design (or the
standalone config) again, and the copy is written.

### "the Opto Program runs … and experiment_type … expects …"

The recording's `Program.txt` runs a Paradigm type the experiment type does not take — for
example ClosedLoop light under `Hedonic`, which expects no light. The member does not load.
Choose a type that expects that light, or build one. See
[The light a type expects](concepts-experiment-types.md#the-light-a-type-expects).

### A member fails to load inside its Project

The Project's design owns every `global:` key. A member whose own `global:` block
disagrees with it — a different threshold, different `well_names` — is refused. Delete the
member's `global:` block so it inherits; the Project Design dialog offers to do it for
every member at once. See [Projects and members](concepts-project.md).

### Only part of my recording loaded

Check which [Slice](concepts-slices.md) you are looking at: a Slice's results hold only its
own minutes, and live in `slices/<name>/`. In the QC Viewer, check **Start time** and **End
time**: an end of `0` means through the end of the recording; any other value truncates. Multi-segment experiments stitch in filename order, so a
misnamed segment can land out of sequence.

### "DFM 99 is the Environmental Monitor, not a DFM"

The config has an entry for DFM 99. That id is the Environmental Monitor's, which records
the rig's temperature, humidity and light, and pyflic never reads it as a DFM. Delete the
entry. See [DFMs and chambers](config-dfms-chambers.md#dfm-99-is-the-environmental-monitor).

### "facet_source: program takes the Facets from data/Program.txt"

The design asks for [Program Facets](concepts-facets.md#program-facets), and a DFM has no
program to take them from: `data/` has no `Program.txt`, or its `Program.txt` has no section
for that DFM. Copy in the `Program.txt` the MCU exported for this recording, or go back to
`facet_source: cutoffs`.

## Results that look wrong

### My exclusions are not being applied

Three common causes, in the order worth checking:

1. **`excluded_chambers` in the YAML.** This key is ignored at load time. pyflic prints a
   warning telling you to migrate to `remove_chambers.csv` — look for it in the load
   output. See [exclusions](config-dfms-chambers.md#excluding-chambers).
2. **A group-name mismatch.** Inside a Project, the design's `exclusion_group:` names the
   group every member is read under. In a script, a bare `remove_chambers` step uses the
   **script's name** as the group. If no group of that name exists in the CSV, nothing is
   excluded and nothing complains.
3. **Results that predate the declaration.** Declaring a chamber changes nothing already on
   disk. The Hub's Analyzed column reads **re-run needed** until you re-run the member.

### Lick counts look far too small

They are probably fourth-root transformed — that is the default. A `Licks` value of `6.2`
means about 1478 licks. Set `global.transform_licks: false` to turn it off, and note that
the automatic removal's reasons, in `summary.txt` and `removed_chambers.csv`, quote
untransformed counts — `min_untransformed_licks_cutoff` is on that scale — while
`feeding_summary.csv` reports transformed ones. See [the lick transform](concepts-metrics.md#the-lick-transform).

### Far too many, or too few, events

Look at the baselined QC plot with thresholds drawn. Events where the trace is flat means
thresholds are too low; obvious feeding unmarked means too high. If instead single meals
are split into many events — or separate visits merged into one — the culprit is the
[link gap](reference-parameters.md#feeding_event_link_gap), not the thresholds.

Run a [parameter sensitivity sweep](scripts-actions.md#analyse-actions) rather than
guessing.

### Every preference index is near zero

Check that `correct_for_dual_feeding` is on — it defaults to `true` for two-well
experiments. Uncorrected crosstalk between the two wells of a chamber pushes every PI
toward zero, which reads as indifference. The **Sim. Feeding** tab in the
[QC Viewer](app-qc-viewer.md) shows how much simultaneous feeding was detected.

### The preference is backwards

`pi_direction` maps physical side to Well A. If you counterbalanced food position across
DFMs but left `pi_direction` at its default everywhere, half your plate has Well A meaning
the opposite substance. See
[Two-well choice and the preference index](concepts-two-well-pi.md).

### Durations changed and I did not change the thresholds

Check `samples_per_second`. Durations are samples ÷ sampling rate, so a wrong rate scales
every duration and interval by a constant while leaving counts untouched — which is why it
is easy to miss.

## Progressive Ratio

### Every chamber group was removed

Look at the training table in `summary.txt` or the report. A group whose paired fly never
finished training has no Test phase, and `require_training_complete` removes it. Its note
says why: the paired sucrose well *never ran a ratio Paradigm*, *never reached Phase M under
a ratio Paradigm*, or *was still in Acclimation (Phase A) when the recording ended*.

On an older recording (generation 1), "no training flag on any well" means the file carries
no training flag — a PyMCU 2.x file, or firmware that did not mark training. It needs
PyMCU 3.x files with light data (`OptoCol1`). See
[Progressive Ratio experiments](concepts-progressive-ratio.md#requirements).

### A group I expected to keep was excluded

Check its light QC verdict in `pr_light_qc.csv`. **Self-triggered light** (a run of
lick-free light events), **implausible training** and **requirement series wrong** fail a
group, and with `exclude_failed_pr_groups` on, every
chamber of it leaves the analysis. The licks-per-event and resting-level figures on the QC
panel show why. *Requirement series wrong* means a light event met a Requirement the Test
Paradigm's series does not give: check that `Program.txt` is this recording's. See
[Light QC](concepts-progressive-ratio.md#light-qc).

### "the design has no yoked controls"

Every chamber group is a single chamber, lit by its own fly alone: `chamber_groups: single`,
or a `Program.txt` whose linkage lights one chamber per circuit. There is then no paired −
yoked difference and no cumulative difference curve; the breaking point and the rest are
reported as usual. If the chambers were meant to be pairs, state `paired_chambers` or check
the program's `Linkage:` line. See
[Without yoked controls](concepts-progressive-ratio.md#without-yoked-controls).

### "the DFM restarted at minute …"

A DFM that restarts on its own inside a group's Test window starts its ratio again, so the
Test window ends at the restart: the breaking point is judged up to it, and censored when
the fly was still responding there. The note is in `pr_light_qc.csv`, `summary.txt` and the
report. Nothing is lost by it but the Test after the restart.

### "the firmware counted rising Events per light event; pyflic's licks did not follow"

A note under a ratio Paradigm with `Counter: Events`: the firmware counted more feeding
bouts for each light event as the requirement rose, as a working progressive ratio should,
while pyflic's licks per light event did not rise. The fly worked to the schedule and
pyflic's detection missed some of it. Look at the baselined trace of the paired sucrose
well, and the detection parameters, before trusting the *no increasing trend* warning beside
it. Under `Counter: Contact`, the default, the firmware counts milliseconds of contact, not
bouts, so this note never appears.

### Most breaking points are censored

`n+` means the group never paused longer than `pr_break_gap_min` before its Test window
ended — it was still responding. A long recording with a short gap censors few groups; a
short one, or a large gap, censors many. Check the sensitivity table in `summary.txt`
before changing the gap. See [Breaking point](concepts-progressive-ratio.md#breaking-point).

### The breaking point stopped overnight

The rule does not know the time of day: a pause at night longer than `pr_break_gap_min`
ends the count. Raise the gap in the design if your flies routinely pause that long and then
resume — the sensitivity table shows what each choice gives.

### A member is missing from the Project's breaking point

The Project pools each member's `analysis/pr_breaking_point.csv`, and a member without one
is left out rather than guessed at: it was analysed before the breaking point existed, or
with *Breaking point* unticked. Count the chamber groups pooled in the Project Report against
what you expect. Re-run that member's analysis — **Run analysis** with *Breaking point*
ticked, or the Project's **Analyze all** — then **Create report** again.

### "no training information: recorded with DFM firmware older than 4.0.0"

An older recording (generation 2): PyMCU 4 recorded a DFM whose firmware was not updated.
The file has the Paradigm columns, all empty, and no training flag either: there is no
record of Training at all. Nothing is removed. Each chamber group gets one `Recording`
Facet, with no Test phase, no breaking point and no sucrose persistence, and the member says
so on the Hub's Loaded line, in the report and in the Project Report. Update the DFM's
firmware for the next recording; this one cannot gain what it did not record. See
[No training information](concepts-progressive-ratio.md#no-training-information).

## Optogenetic recordings

### The optogenetic light QC failed a group I trust

Open `qc/opto/opto_light_qc.csv` and the **Light explained by licks** figure. Red light
beside orange in the panel to its right is a trigger well the firmware read as touched while
pyflic saw nothing: a drifting or leaking well, whose light is not evidence of feeding. Red
light with no orange is light the firmware had no reason to switch on — check the lid and
the linkage. A failed
group is flagged, not excluded, unless `exclude_failed_opto_chambers` is on. Without
`data/Program.txt` the check cannot tell an open-loop schedule from a stuck light, so it
only warns. See [Optogenetic experiments](concepts-optogenetics.md).

### "holds more than one Program.txt"

`data/` may hold one `Program.txt`, the one the MCU exported for this recording. Remove the
others; which one describes the run is not a guess pyflic will make.

### "firmware 4.0.0 recording without Program.txt"

The full warning reads *DFM n: firmware 4.0.0 recording without Program.txt: Paradigm types
and settings unknown*. The data rows name each Paradigm but not its type, its settings or
the thresholds, so everything that depends on the type is `NA` and the light QC cannot
fail a group. The light QC flags it as *no Program.txt for firmware 4.0.0*. Copy the
`Program.txt` PyMCU exported for this recording into `data/` and run again.

A `Program.txt` that starts `***ALL DFM***` is not missing: it is the export of a recording
started without a Program, and every well was `Off`. Any light then fails the group.

### "paradigm not in program"

A row of the data names a Paradigm that `Program.txt` does not define. The file belongs to
another recording: every group of that DFM fails, and those rows are not judged. Find the
`Program.txt` exported with this recording. Names are matched ignoring case, as PyMCU
matches them.

A row naming a *defined* Paradigm other than the one `Program.txt` prints for its interval
is a **program mismatch** warning instead: the rows are judged by their own Paradigm.

### "broken yoking"

Linked wells share one light, so their `OnMs` should match to the millisecond, and a well set
`Off` should count no Activations of its own. The group fails when its linked wells' light
differs in more than `opto_yoking_fail_fraction` (1%) of its lit records, or when an `Off`
well linked to a trigger well counted Activations. The yoked fly did not get its partner's
light, so the comparison between them does not hold. Check the lid's wiring and the
program's `Linkage:` line. The [Program timeline](plots-catalog.md#program-timeline) shows
which well differs.

### Lost, repeated, late and unknown-instruction records

The integrity report and the light QC read the DFM's own bookkeeping:

| Finding | What happened | What to do |
|---|---|---|
| **lost records**: gaps in `Index` | the DFM could not keep records while the MCU was not reading them, and they are gone; PyMCU logs *Lost N records* (older builds *Skipped index*) in `Messages.csv`, and the integrity report checks its count against the gaps | a short gap changes little; more than `opto_lost_records_warn` (25 records, 5 s) warns, not counting a restart's gap. Treat results across a long gap with care, as for a data break |
| **DFM restarted**: a gap in `Index` followed by rows under no instruction | the DFM restarted on its own, for example on a power dip; PyMCU resent its instruction, and every well's Paradigm state started over | the integrity report's *DFM restarts* lists each, and whether `Messages.csv` names it; the light QC warns on every group of that DFM, and the Program timeline draws a red band. Acclimation and the Requirement series start again after it. Check the DFM's power and cabling |
| **repeated records** | PyMCU wrote a block of records twice after a failed file write | nothing: the second copy is dropped at load, matched on `Index` and time, and `summary.txt` and the integrity report say how many |
| **late records**: `OnMs` above 200 ms | a record came later than its 200 ms | more than `opto_late_records_warn` (10) warns. Look at the `Error` column's interrupt flags in the integrity report |
| **unknown-instruction rows**: `Interval` `NA` after the start | the DFM ran an instruction outside the program, for example outputs switched by hand during the recording | those rows are not judged. More than `opto_unknown_rows_warn_fraction` (1%) of the rows warns |

### "acclimation count mismatch"

A trigger well's completed Acclimation counted a different number of Activations than its
Paradigm's `Acclimation` setting. The firmware gives exactly that many, so first check
that `Program.txt` is this recording's. If it is, the DFM did not do what its Paradigm says:
look at the well's lane in the [Program timeline](plots-catalog.md#program-timeline). An
Acclimation cut short by the end of the recording or by a new Paradigm is never counted.

### "firmware events disagree"

Under a ratio Paradigm with `Counter: Events` the firmware counts its own feeding bouts,
Firmware Events, and pyflic counts its feeding events from the same signal. They are
different detectors and differ at the edges. Like the firmware, pyflic's count leaves out
the bouts that start while the well's own Activation is lit. A trigger well whose two counts differ by more than
`opto_event_disagreement_warn_fraction` (25%) of the Firmware Events warns, once either count
reaches 5. Far more Firmware Events than pyflic events is a drifting well or thresholds that
miss real feeding; far fewer is pyflic splitting bouts the firmware joins. Compare the
firmware's `EventGapMs` with your `feeding_event_link_gap`, and look at the baselined trace.
Under `Counter: Contact`, the default, there are no Firmware Events, so this check does not
run.

### A Paradigm's rule broken: "wrong activation length", "lockout broken", "requirement series wrong", "max on exceeded", "short activations"

These fail a group. The firmware keeps each Paradigm's rule to
the millisecond, so light that breaks the rule of the Paradigm `Program.txt` names is not a
fly's doing. The likely cause, in order:

1. **`Program.txt` is not the program that ran.** A file from another recording, or a
   program edited and re-exported, gives a Paradigm the same name and other settings. The
   reason names the setting and the value measured: compare them with what you ran.
2. **The LED or its `OnMs` report is faulty.** Look at the group's rule figure in the
   report, and its lanes in the Program timeline.

`opto_activation_tolerance_ms` (25 ms) is how far a light period may miss its setting. Do
not widen it to clear a failure: the firmware's own timing is exact.

### "contact without activation", "non-feeding dark", "interval started late"

Warnings, because they rest on pyflic's lick detection or on PyMCU's timing rather than on
the firmware's own record:

- **contact without activation**: FixedInterval contact pyflic saw, with the well free,
  that earned no light. Contact below the firmware's threshold, a sensor that stopped
  responding, or a `feeding_threshold` that counts noise. Look at the baselined trace.
- **non-feeding dark**: a NonFeeding light that stayed off when the fly was not feeding. A
  resting signal that drifted up, which the firmware reads as contact, or a longer
  `OnDelayMs` than `Program.txt` states.
- **interval started late**: a Program Interval's first row came more than
  `opto_interval_lag_warn_sec` (5 s) after its `Program.txt` time. The DFM took its
  instruction late and PyMCU sent it again; the light before it ran the previous
  Interval's rule.

See [Each Paradigm's own rule](concepts-optogenetics.md#each-paradigms-own-rule).

### "open loop not lit"

A warning: an open-loop interval, whose wells are set to light whatever the fly does, was lit
for less than `opto_open_loop_min_lit_fraction` (95%) of its time. The reason names the
group's darkest open-loop window, so one dark Program Interval shows.
Check the lid and its LEDs, and that `Program.txt` is this recording's. See
[The checks](concepts-optogenetics.md#the-checks).

### No Role, or every Role blank

A chamber's Role comes from `Program.txt`, or without it from the data. An older recording
without `Program.txt` cannot tell a trigger well from a yoked one, so its Roles are blank.
See [Roles](concepts-optogenetics.md#roles).

## The QC, Analyze and Plots panels

### A file I expected was not written

Look at the log line the run printed, and for Run analysis at `summary.txt`: each item is
listed as run, *Left out* (its box is unticked) or *Not applicable* (the member cannot have
it, or its data cannot answer it). A box missing from the panel is one the member cannot
have; a greyed box is one its data cannot answer, with the reason in its tooltip. See
[The ticks are configuration](app-hub.md#the-ticks-are-configuration).

No run removes a file, so a file dated before the run is left from an earlier one, for an
item this run left out.

### Unticking a box on one member changed every member

Inside a Project the ticks belong to the Project's design, `design.global.omit` in
`project.yaml`, so every member is analysed alike; the log line says so. There is no
Selection of one member's own: a member that must be analysed differently belongs in another
Project.

### Analyze all did not follow my ticks

It is not meant to. The Project panel's **Analyze all**, and a script's
`run_all_analyses` step (which a Batch Run's default script uses), run each member's Basic
analysis, whatever the panels' ticks. Use **Run QC**, **Run
analysis** and **Generate plots** on a loaded member, or a script's `run_qc`,
`run_analysis` and `run_plots` steps, to run the Selection. See
[The Analysis card](app-hub.md#the-analysis-card).

## Batch runs

### A Project did not run

The scan reports every folder it skipped — a `project.yaml` with no member directory, an
unreadable folder, a symlink — in the Output tab at the start of the run, and in the Errors
tab too when you **Rescan**. A Project with nothing usable starts unchecked in the review
window. See [Where Projects are found](scripts-batch.md#where-projects-are-found).

### An old folder is not treated as a batch

Subdir-batch mode, keyed on a script named `batch`, is retired. A Batch is now any folder
with Projects beneath it; `pyflic lint` reports the old constructs. See
[Migrating from the old batch modes](scripts-batch.md#migrating-from-the-old-batch-modes).

## Performance

### Loading is slow

Keep **parallel** loading on (the Project panel's load options, or `parallel=True` in
`load_experiment_yaml`). Loading is the expensive operation; everything after it reuses the
result.

### Do I need to clear the cache?

Almost never. `.pyflic_cache/` is keyed by input, so a stale entry cannot be served for
changed inputs. Clear it to reclaim disk space, not to fix results.

## Getting more detail

- `pyflic lint <folder>` validates every configuration and reports line numbers.
- **Validate every YAML here** on the Hub's Tools panel parses every `project.yaml`,
  `batch.yaml` and `flic_config.yaml` under the selection. **Validate YAMLs** on the Project
  panel does the same for the open Project and all its members.
- The Hub's Output tab carries the full message from any failure, and the Errors tab
  collects the warnings and failures on their own.
