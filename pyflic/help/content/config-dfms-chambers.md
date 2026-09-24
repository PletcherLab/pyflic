# DFMs, chambers and exclusions

The `dfms` block is where your configuration stops being generic and starts describing
*your* experiment: which device recorded what, and which treatment each fly received.

## The `dfms` block

One entry per physical device. The key is the device number, and it must match the number
in your CSV filenames — `DFM1_0.csv` needs an entry keyed `1`.

```yaml
dfms:
  1:
    params:
      pi_direction: left      # overrides the global value for this DFM only
    chambers:
      1: Paired,WCS
      2: Unpaired,WCS
  2:
    params: {}                # no overrides; inherits everything from global
    chambers:
      1: Unpaired,Chrim
      2: Paired,Chrim
```

**`params`** accepts the same keys as `global.params` and takes precedence for that device.
This is how counterbalancing works — see
[Two-well choice and the preference index](concepts-two-well-pi.md).

**`chambers`** maps chamber number to treatment name, or to comma-separated factor levels
when [factors](config-factors.md) are defined.

## Progressive Ratio: `paired_chambers`

A Progressive Ratio DFM also names the **paired** chamber of each chamber group —
chambers 1+2, 3+4 and 5+6 — and the other chamber of the group is yoked:

```yaml
- id: 1
  params: {pi_direction: left}   # side of well A, the sucrose well
  paired_chambers: [1, 4, 5]     # one from each group
  chambers: {1: Ctrl, 2: Ctrl, 3: Exp, 4: Exp, 5: Exp, 6: Exp}
```

Exactly one chamber per group, and both chambers of a group carry the same treatment;
the loader and `pyflic lint` refuse anything else. Well A is always the sucrose well, so
there is no separate side key: `pi_direction` places it, as in any two-well experiment.
The Config Editor shows the three pickers on each DFM tab. See
[Progressive Ratio experiments](concepts-progressive-ratio.md#chamber-groups-paired-and-yoked).

## How chambers map to wells

Chamber numbering depends on the chamber layout, which the Experiment Type fixes (or a
Custom experiment states):

| Chamber layout | Chambers per DFM | Chamber *n* covers |
|---|---|---|
| `single_well` | 12 | well *n* |
| `two_well` | 6 | wells 2*n*−1 and 2*n* — so chamber 1 is W1+W2, chamber 2 is W3+W4, … |

A two-well configuration listing a chamber `7` is therefore an error: there are only six.

## Chambers you leave out

Only the chambers you list are analysed. A chamber absent from the `chambers` block is not
part of the experiment — that is the right way to describe an empty position or a well you
never loaded.

Use it for positions that were *never* part of the design. For chambers that were part of
the design but produced unusable data, use exclusions instead, so the removal is recorded
rather than invisible.

## Excluding chambers

Exclusions live in **`remove_chambers.csv`** in the experiment directory, not in the YAML.

```csv
group,dfm_id,chamber,note
general,1,3,low lick count
general,2,5,
Standard Analysis,1,4,noisy signal
```

The `group` column lets one file hold several exclusion sets, so you can keep a
conservative set and a permissive set side by side and choose between them per analysis
without editing the file.

Which group applies depends on how you load:

| How you load | Group applied |
|---|---|
| `load_experiment_yaml(..., exclusion_group="general")` | the one you name |
| The Hub, loading a Project's member | the design's `exclusion_group:` (default `general`) |
| The Hub, loading a standalone experiment | `general` |
| The `remove_chambers` script step | the step's `group:`, defaulting to the **script's name** |
| QC Viewer, **Save Exclusions…** | saves the current selection to any group you name |

That script-step default is worth internalising: a script named `Standard Analysis` with a
bare `remove_chambers` step applies the group `Standard Analysis`. It is what lets one CSV
serve several scripts with different exclusion sets and no extra configuration — and it is
also why a typo in a script name silently applies *no* exclusions, since a group that does
not exist is simply empty.

> **`excluded_chambers` in YAML no longer works.** The key is ignored at load time and
> pyflic prints a warning telling you to migrate it to `remove_chambers.csv`. If you are
> carrying an old configuration, check the load output for that warning — your exclusions
> are not being applied.

## Automatic exclusion

`auto_remove_chambers()` removes chambers that fail the cutoffs in `global.constants`, and
always removes chambers whose lick value is `NaN`. See
[`global.constants`](config-structure.md#globalconstants). It updates the design in place
and clears the feeding-summary cache, so everything computed afterwards reflects the
filtered set.

It does not run at load. Basic analysis runs it once, before writing the summary, so the
Hub's *Basic analysis*, *Analyze all* and a Batch Run all apply it; the QC Viewer's **Auto
Filter** and the Python API run it on demand. For a Progressive Ratio experiment
it also takes out the chamber groups that fail the light QC — see
[Progressive Ratio experiments](concepts-progressive-ratio.md#light-qc).

---

Related: [Configuration file structure](config-structure.md) ·
[Factorial designs](config-factors.md) · [QC Viewer](app-qc-viewer.md)
