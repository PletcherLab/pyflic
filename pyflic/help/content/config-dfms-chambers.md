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

The same block may be written as a list, one entry per device with its number as `id:` —
the form the Config Editor writes, and the one the examples below use for a single entry:

```yaml
dfms:
  - id: 1
    params: {pi_direction: left}
    chambers:
      1: Paired,WCS
      2: Unpaired,WCS
```

**`params`** accepts the same keys as `global.params` and takes precedence for that device.
This is how counterbalancing works — see
[Two-well choice and the preference index](concepts-two-well-pi.md). Inside a Project only
the physical keys `pi_direction` and `chamber_sets` may be overridden here.

**`chambers`** maps chamber number to treatment name, or to comma-separated factor levels
when [factors](config-factors.md) are defined.

**`well_names`** on a DFM entry overrides `global.well_names`, well by well, for that
device.

## Progressive Ratio: `paired_chambers` and `chamber_groups`

Under a type with the Progressive Ratio analysis, a DFM's chambers form **chamber groups**,
each lit by one light circuit:
a pair of a **paired** and a **yoked** chamber, or a **solo** group of one paired chamber
with no yoked control. Three ways to say which, per DFM entry:

- **`paired_chambers`**: pairs 1+2, 3+4 and 5+6, naming the paired chamber of each; the
  other chamber of the group is yoked.

  ```yaml
  - id: 1
    params: {pi_direction: left}   # side of well A, the sucrose well
    paired_chambers: [1, 4, 5]     # one from each pair
    chambers: {1: Ctrl, 2: Ctrl, 3: Exp, 4: Exp, 5: Exp, 6: Exp}
  ```

  Exactly one chamber per pair, and both chambers of a pair carry the same treatment; the
  loader and `pyflic lint` refuse anything else.
- **`chamber_groups: single`**: every chamber its own solo group, with no yoked control.
  It cannot be stated with `paired_chambers`.
- **Neither**, with the MCU's `Program.txt` in `data/`: the groups its linkage makes. A
  linkage group lighting two chambers is a pair, its paired chamber the one holding the
  trigger well; one lighting a single chamber is a solo group. Pairs that are not
  neighbours (1+4, say) are read as the program pairs them.

What the config states wins: a `Program.txt` that disagrees is reported in `summary.txt`.
Without `Program.txt`, and with `chamber_groups: paired` (the default), `paired_chambers`
is required. `chamber_groups` is read from a DFM entry only, never from `global:`. Well A is
always the sucrose well, so there is no separate side key: `pi_direction` places it, as in
any two-well experiment. The Config Editor's DFM tab has a **Groups** choice and the
pickers for pairs. See
[Progressive Ratio experiments](concepts-progressive-ratio.md#chamber-groups-paired-and-yoked).

## Optogenetics per DFM

`optogenetics:` on a DFM entry overrides the experiment's setting for that DFM alone — a
DFM that ran without an Optolid, say. It takes `auto`, `true` or `false`; left out, the DFM
inherits. It is a fact about the recording, so it stays free inside a Project. See
[Optogenetic experiments](concepts-optogenetics.md#turning-it-on).

## Roles are read, not written

On an optogenetic DFM every chamber gets a **Role** — Paired, Yoked, Mutual, Self, Open or
Unlit — that says what it was to the light. pyflic reads it from `Program.txt`'s linkage and
thresholds, or without it from the data. You never write it, and it is
not a treatment: do not declare it as a design factor. It appears as a `Role` column beside
`Treatment`. A Progressive Ratio DFM is the one case you may state, with `paired_chambers`
or `chamber_groups`, and the program is checked against it. See
[Optogenetic experiments](concepts-optogenetics.md#roles).

## DFM 99 is the Environmental Monitor

ID 99 belongs to the Environmental Monitor, which records the rig's temperature, humidity
and light, not to a DFM. pyflic never reads it as one:

- a new member's `dfms:` block never gets an entry for it;
- a config that names `99` under `dfms:` is refused, and `pyflic lint` reports it, with
  *DFM 99 is the Environmental Monitor, not a DFM*. Remove the entry;
- its file is never read as a DFM. PyMCU names it `EnvMon_<n>.csv`, and a `DFM99_<n>.csv`
  is not read either.

Each DFM's own rows already carry the monitor's readings (`Temp`, `Humid`, `LUX`).

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
| `load_experiment_yaml(...)` naming no group | the config's `exclusion_group:` (default `general`) |
| The Hub, loading a Project's member | the design's `exclusion_group:` (default `general`) |
| The Hub, loading a standalone experiment | its `exclusion_group:` (default `general`) |
| The `remove_chambers` script step | the step's `group:`, defaulting to the **script's name** |
| QC Viewer, **Save Exclusions…** | saves the current selection to any group you name, asking first when it is another analysis's own group |

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

It does not run at load. An analysis runs it once, before writing the summary, so the
Hub's *Run analysis* and *Analyze all*, and a Batch Run, all apply it; the QC Viewer's **Auto
Filter** and the Python API run it on demand. For a Progressive Ratio experiment
it also takes out the chamber groups that fail the light QC — see
[Progressive Ratio experiments](concepts-progressive-ratio.md#light-qc). With
`exclude_failed_opto_chambers` on, it takes out every chamber of a linkage group that fails
the optogenetic light QC — see [Optogenetic experiments](concepts-optogenetics.md#thresholds).

---

Related: [Configuration file structure](config-structure.md) ·
[Factorial designs](config-factors.md) · [QC Viewer](app-qc-viewer.md)
