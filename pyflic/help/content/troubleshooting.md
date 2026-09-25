# Troubleshooting

## Loading

### "flic_config.yaml not found in experiment directory"

The folder you chose is not an Experiment Directory. It needs a `flic_config.yaml` at its
top level — not inside `data/`. The Hub says "is not a Batch, a Project, or an Experiment
Directory" for the same reason. See
[Setting up an experiment directory](getting-started-project.md).

### A DFM in my config has no data

CSV filenames must match the DFM numbers in your configuration. A DFM keyed `3` needs
`DFM3_0.csv` (v3) or `DFM_3.csv` (v2) in `data/`. Check for a leading zero, a different
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

### A member fails to load inside its Project

The Project's design owns every `global:` key. A member whose own `global:` block
disagrees with it — a different threshold, different `well_names` — is refused. Delete the
member's `global:` block so it inherits; the Project Design dialog offers to do it for
every member at once. See [Projects and members](concepts-project.md).

### Only part of my recording loaded

Check the start and end minute controls. `end: 0` means through the end of the recording;
any other value truncates. Multi-segment experiments stitch in filename order, so a
misnamed segment can land out of sequence.

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
`summary.txt` reports untransformed counts while `feeding_summary.csv` reports transformed
ones. See [the lick transform](concepts-metrics.md#the-lick-transform).

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

Look at the training table in `summary.txt` or the report. "no training flag on any well"
means the recording carries no training flag — a version-2 file, or firmware that did not
mark training — so no group ever finished training and `require_training_complete`
removed them all. The type needs version-3 files with light data (`OptoCol1`). See
[Progressive Ratio experiments](concepts-progressive-ratio.md#requirements).

### A group I expected to keep was excluded

Check its light QC verdict in `pr_light_qc.csv`. **Self-triggered light** (a run of
lick-free light events) and **implausible training** fail a group, and with
`exclude_failed_pr_groups` on, both its chambers leave the analysis. The licks-per-event and
resting-level figures on the QC panel show why. See
[Light QC](concepts-progressive-ratio.md#light-qc).

### The optogenetic light QC failed a group I trust

Open `qc/opto/opto_light_qc.csv` and the **Light explained by licks (QC)** figure. Red light
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

### Most breaking points are censored

`n+` means the group never paused longer than `pr_break_gap_min` before its Test window
ended — it was still responding. A long recording with a short gap censors few groups; a
short one, or a large gap, censors many. Check the sensitivity table in `summary.txt`
before changing the gap. See [Breaking point](concepts-progressive-ratio.md#breaking-point).

### The breaking point stopped overnight

The rule does not know the time of day: a pause at night longer than `pr_break_gap_min`
ends the count. Raise the gap in the design if your flies routinely pause that long and then
resume — the sensitivity table shows what each choice gives.

### The Project Report says a member has no breaking point table

That member was analysed before the breaking point existed. Re-run its basic analysis.

## Batch runs

### A Project did not run

The scan reports every folder it skipped — a `project.yaml` with no member directory, an
unreadable folder, a symlink — in the Output tab at the start of the run. A Project with
nothing usable starts unchecked in the review window. See
[Running many projects at once](scripts-batch.md).

### An old folder is not treated as a batch

Subdir-batch mode, keyed on a script named `batch`, is retired. A Batch is now any folder
with Projects beneath it; `pyflic lint` reports the old constructs.

## Performance

### Loading is slow

Keep **parallel** loading on (the Project panel's load options, or `parallel: true` on the
`load` step). Loading is the expensive operation; everything after it reuses the result.

### Do I need to clear the cache?

Almost never. `.pyflic_cache/` is keyed by input, so a stale entry cannot be served for
changed inputs. Clear it to reclaim disk space, not to fix results.

## Getting more detail

- `pyflic lint <folder>` validates every configuration and reports line numbers.
- **Validate every YAML here** on the Hub's Tools panel parses every `project.yaml`,
  `batch.yaml` and `flic_config.yaml` under the selection.
- The Hub's Output tab carries the full message from any failure, and the Errors tab
  collects the warnings and failures on their own.
