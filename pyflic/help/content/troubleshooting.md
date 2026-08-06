# Troubleshooting

## Loading

### "No YAML config files found"

The folder you chose is not a project directory. It needs a `flic_config.yaml` (or another
`.yaml`) at its top level — not inside `data/`. See
[Setting up a project directory](getting-started-project.md).

### A DFM in my config has no data

CSV filenames must match the DFM numbers in your configuration. A DFM keyed `3` needs
`DFM3_0.csv` (v3) or `DFM_3.csv` (v2) in `data/`. Check for a leading zero, a different
separator, or a file left in the project root instead of `data/`.

### `chamber_size` must be explicitly specified

pyflic refuses to guess this, because the wrong value silently regroups every well on the
plate. Add it under `global.params`. See
[`chamber_size`](reference-parameters.md#chamber_size).

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
2. **A group-name mismatch.** A bare `remove_chambers` step uses the **script's name** as
   the group. If no group of that name exists in the CSV, nothing is excluded and nothing
   complains.
3. **No `remove_chambers` step at all.** Exclusions are not automatic; the step has to be
   in the script.

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

## Batch runs

### A folder did not run

Look for it in the **near-miss** log lines. A directory with YAMLs but no script named
`batch` is logged as a near-miss precisely so a forgotten script is visible. A directory
with no YAMLs at all is skipped silently.

Also check that the folder is not under a pruned directory name: anything starting with
`.`, or named `analysis`, `plots`, `qc`, `__pycache__`, or `node_modules`. Symlinks are not
followed.

### The batch count is higher than expected

Every YAML defining `batch` contributes its own run, and parent directories run alongside
their descendants — there is no leaf-only filter. See
[Running many projects at once](scripts-batch.md).

## Performance

### Loading is slow

Enable `parallel: true` on the `load` step. Loading is the expensive operation; everything
after it reuses the result.

### Do I need to clear the cache?

Almost never. `.pyflic_cache/` is keyed by input, so a stale entry cannot be served for
changed inputs. Clear it to reclaim disk space, not to fix results.

### Many figures make the hub sluggish

Turn off **Interactive plots** in the top bar. Static rendering paints faster and holds no
live figure in memory.

## Getting more detail

- `pyflic lint <project>` validates a configuration and reports line numbers.
- The hub's output panel carries the full message from any failure.
- **YAML info** in the Project card summarises every configuration in a folder.
