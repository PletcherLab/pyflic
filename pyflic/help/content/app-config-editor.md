# Config Editor

```bash
pyflic config my_experiment/
```

A form-based editor for `flic_config.yaml`. It knows which keys exist, which values are
allowed, and which are required, so it writes a valid configuration rather than leaving you
to discover a typo three steps later.

Name the folder, or the YAML file, to edit. Started without one, the editor opens the
`flic_config.yaml` of the folder you are in, if it has one; anywhere else it opens empty,
and **File → Save** asks where to save.

Use it for your first configuration. Hand-editing YAML is fine once you know the format —
see [Configuration file structure](config-structure.md) — but there is no advantage to
starting there.

**F1**, or the editor's **Help** menu, opens this topic; every card carries a `?` for its
own section, and every detection parameter a `?` that opens
[Parameter reference](reference-parameters.md) at that parameter, so you can read what a
value does without leaving the form. **File → Script Editor…** (Ctrl+E), or the scripts
button in the top bar, opens the [Script Editor](scripts-editor.md) on the saved file.

## The two tabs

**Experiment** — everything that applies to the recording as a whole: the Experiment
Type, the Chamber Layout it implies, well names, the detection parameters, the factors
of your design, and its exclusion groups and Slices.

**DFMs & Chambers** — one tab down the left side per device, each with its own chamber
assignments and parameter overrides. **Add DFM** and **Remove DFM** sit below the list;
Remove acts on the DFM you have selected and names it before it discards anything.

**Problems are counted on the tab they live on.** A ⚠ and a number on a tab label means
something there needs attention — a Hedonic experiment with no well names, a chamber
missing a factor level. Saving with problems outstanding is allowed; the editor lists them
first so you are choosing to. The one exception is a Slice that cannot be used, which stops
the save (see [Exclusion Groups and Slices](#exclusion-groups-and-slices)).

## The Experiment tab

### Experiment Settings

- **Data Folder** — where the recording is read from, when it is not the `data/` beside
  this config: a path relative to this folder, with forward slashes. Blank (the usual
  case) reads `data/`, and no key is written. pyflic only reads that folder; everything it
  writes stays in this one. A Project's design never locks it, because it belongs to this
  experiment alone. See [`data_folder`](config-structure.md#data-folder).
- **Experiment Type** — the assay; its `?` opens
  [Experiment types](concepts-experiment-types.md). **The Experiment Type owns the Chamber
  Layout.** Choose *Hedonic Feeding* or *Progressive Ratio* and the layout is fixed for you
  — shown, but not editable, because those assays are two-well by definition. A **Custom
  Experiment**, or a lab type that leaves the layout open, chooses its own. Your lab's own
  types are listed with *(lab)* after them, and **New type…** opens the
  [Experiment Type Editor](app-type-editor.md); saving a config whose type is one of your
  lab's writes the type beside it as `experiment_type.yaml`.
- **Chamber Layout** — single-well gives 12 chambers per DFM; two-well gives 6. Change the
  layout and the chamber fields on the other tab change with it — and if that would discard
  assignments you have already made, the editor names them and asks first. The editor
  writes `experiment_type:` for any type but Custom, `chamber_layout:` when the type leaves
  the layout open, and `params.chamber_size` for neither: it is derived from the layout,
  never stored.
- **Well A** and **Well B** — the names of the two wells, for a two-well layout.
- **Transform Licks** — the 0.25-power transform of lick counts; see
  [The lick transform](concepts-metrics.md#the-lick-transform).
- **Control Level** — shown for a type with a Treatment control: the level of its control
  factor that every other level is compared with. See
  [Control comparisons](concepts-experiment-types.md#control-comparisons).
- **Optogenetics** — `auto`, `yes` or `no`: whether the optogenetic light QC runs. Each
  DFM tab has its own picker, which inherits unless you pick. See
  [`global.optogenetics`](config-structure.md#globaloptogenetics).
- **Facet source** — where the Facets come from: minute cutoffs (the default), or the
  Program Intervals of `Program.txt`, with or without their Cycles pooled. A type that
  owns its Facets, such as Progressive Ratio, greys it out. See
  [Program Facets](concepts-facets.md#program-facets).

### Auto-filter thresholds

Each of **Min Untransformed Licks**, **Max Median Duration**, **Max Events** and **Max
Minutes at Ceiling** has a **Use** checkbox and a value field:

- Uncheck **Use** to turn that filter off. The editor writes YAML `null`, so a typed
  experiment does not fall back to its default.
- Check **Use** and leave the value blank to keep the Experiment Type's default, shown as
  grey placeholder text.
- Check **Use** and type a number to override it.

A Custom Experiment has no defaults for the first three, so there those filters run only
when **Use** is checked and a number is typed. **Max Minutes at Ceiling** defaults to 30
under every type, Custom included. What each cutoff removes is in
[Automatic exclusion](config-dfms-chambers.md#automatic-exclusion).

### Progressive Ratio settings

For a type with the Progressive Ratio module, a *Progressive Ratio* section below the
thresholds holds the module's own constants: whether a chamber group whose training never
completed, or whose light QC failed, leaves the analysis; the light QC's thresholds; and
the breaking point's **Break gap** (`pr_break_gap_min`) and **Test window cap**
(`pr_test_window_min`). They follow the thresholds' rule. A blank field shows the type's
default as grey text and keeps it, and a switch left on *default* does the same; only a
value you type or pick is written under `global.constants`. A value out of range, such as
a gap of 0, is counted on the tab and listed before saving, and pyflic refuses to load a
config that states one. What each setting does is in
[the light QC](concepts-progressive-ratio.md#light-qc) and
[the breaking point](concepts-progressive-ratio.md#breaking-point).

### Optogenetic light QC settings

Once the experiment is known to be optogenetic — **Optogenetics** is `yes`, or it is `auto`
and `data/` holds a `Program.txt` or a run has already written its light QC — an
*Optogenetic light QC* section shows the thresholds, shared by every type, under the same
rule as the rest: grey defaults, and only what you type is written. Thirteen of them use
the Paradigm columns, so they do not apply to older recordings; eight of those are for the
checks of each Paradigm's own rule. See
[Optogenetic experiments](concepts-optogenetics.md#thresholds).

### Global Parameters

The detection parameters, applied to every DFM unless a DFM tab overrides them. Each has a
`?` for its entry in [Parameter reference](reference-parameters.md). **The editor writes
canonical parameter names.** Older aliases such as `link_gap` and `tasting_low` are still
accepted when read, but what gets written is the canonical name.

### Experimental Design Factors

Optional. Declaring a factor here gives the chamber table on the other tab one column per
factor, each a dropdown of that factor's levels, which is the main reason to use the editor
rather than hand editing: adding a factor by hand means rewriting every chamber assignment
in the file. See [Factorial designs](config-factors.md).

### Exclusion Groups and Slices

Optional. Which `remove_chambers.csv` groups each analysis reads, and the
[Slices](concepts-slices.md): separate analyses of part of the recording.

- **Exclusion Group**: the whole recording's own group, the one its removals are read
  from and written to (`general` when blank). See
  [`global.exclusion_group`](config-structure.md#globalexclusion-group).
- **Also Exclude**: further groups, comma-separated, whose chambers the whole recording
  also leaves out without writing to them, such as `dead` for flies dead from the start.
  See [Exclusions](concepts-slices.md#exclusions).
- **The Slices table**, one row per Slice:
  - **Name**.
  - **Start (min)**. Blank is 0.
  - **End (min)**. `end`, or blank, runs to the end of the recording.
  - **Own group**: the group this Slice's exclusions are read from and written to. It is
    the Slice's name unless you change it; a blank cell becomes the name, and the group is
    always written to the file. Renaming a Slice renames its own group too while that was
    still the old name, and a Slice left out of the Hub's Selection stays left out under
    its new name.
  - **Also exclude**.
  - The cutoffs a Slice may override. **Min licks** takes a number, or `off`. While the
    optogenetic settings are shown, **Opto failed** takes `yes` or `no`, **Fail fraction**
    takes the unexplained-light failure fraction (0 to 1, not below the design's warning
    fraction), and **Unexplained (s)** takes a number of seconds.

  A blank cutoff keeps the experiment's own. **Add Slice** adds a row, and **Remove**
  removes the selected one.
- **Coverage tolerance (min)**: how far short of a Slice's end a recording may stop and
  still cover it. Blank is 5. See [Coverage](concepts-slices.md#coverage).

The rows are written under `global.slices:` in the order shown, each with only what you
filled in. **A Slice that cannot be used stops the save.** That includes an end before its
start, a reserved name (`whole`, `all`, `none`), or an own group that another analysis
already has. The editor counts these on the tab, lists them when you save, and writes nothing
until they are corrected or removed, because every run would skip such a Slice. For a type
with the Ratio analysis (Progressive Ratio, Fixed Ratio) the table is greyed out, with the
reason: [such a type has no Slices](concepts-slices.md#progressive-ratio-and-fixed-ratio).

## The DFMs & Chambers tab

### DFM Identity

The DFM's id — the number in its data file names — and its own **Optogenetics** picker:
*inherit* takes the Experiment tab's setting; `auto`, `yes` or `no` decide for this DFM
alone, because whether a lid was lit is a fact about the recording. See
[Optogenetics per DFM](config-dfms-chambers.md#optogenetics-per-dfm).

### Chamber assignments

The **Chamber → Treatment Assignments** table, one row per chamber. Leave a Treatment cell
blank to omit that chamber from the config. With factors declared, the table has a column
per factor instead, each a dropdown of its levels.

**Factor assignments are positional and must be complete.** With two factors declared,
every assigned chamber needs a level in both columns — the columns are read in declaration
order, so a half-filled row is an incomplete assignment, not a shorter one. The editor
marks the gap and leaves that chamber out of the file rather than writing something that
would read back wrong. See [How chambers map to wells](config-dfms-chambers.md#how-chambers-map-to-wells).

### Paired chamber per chamber group

With a type that has the Progressive Ratio module, each DFM tab shows **Paired chamber per
chamber group**, with a **Groups** choice:

- **Pairs**: one picker per pair (1+2, 3+4, 5+6) naming its paired chamber, written as
  `paired_chambers:`. The other chamber is yoked and never written. Give both chambers of
  a pair the same treatment.
- **Single chambers**: every chamber its own group, with no yoked control, written as
  `chamber_groups: single`. The pickers are greyed out.
- **From Program.txt**: neither key is written, and the loader takes the groups from the
  linkage of the `Program.txt` in `data/`. Without one, the DFM does not load.

A DFM entry that already names neither key opens on *From Program.txt*. Set **PI
Direction** on the DFM to the side the sucrose well (well A) is on. See
[Progressive Ratio experiments](concepts-progressive-ratio.md#chamber-groups-paired-and-yoked).

### Parameter Overrides

Tick a parameter's box to override the global value for this DFM alone; an unticked one
inherits. Per-DFM overrides are how counterbalancing is set up; see
[`pi_direction`](reference-parameters.md#pi_direction).

## A member's config

Opened on a Project's member, the editor fills the global settings from `project.yaml`,
the exclusion groups and Slices among them, marks them read-only with a banner, and writes
`dfms:`, `data_folder:` when the member names one, and under `global:` only the design's
description of the recording (type, layout, well names, factors, control level), so the
file reads on its own — see [Projects and members](concepts-project.md). A member whose
own description conflicts with the design keeps its `global:` as written. Any save keeps
the keys the editor does not show, `scripts:` and `about:` above all, as they were.

For a recording's title, purpose, flies and what each level means, use the
[Experiment Editor](app-experiment-editor.md).

**`about:` is written back as it is on disk when you save**, for a member and for a
standalone experiment alike. A catalogue that files the recording rewrites that section
afterwards, perhaps while the config is open here. The editor does not edit it, so it
reads it again at each save and writes that, never the copy it read when it opened the
file. **Save As** into another recording's folder leaves out the catalogue's id (`run:`),
which names the recording the config was read from.

## Things worth knowing

**Old configurations migrate on save.** Configurations written before pyflic split
Experiment Type from Chamber Layout still open: `experiment_type: two_well` opens as a
Custom Experiment with that layout, and saving migrates it.

**Excluded chambers are not listed here.** The editor names the exclusion groups; which
chambers a group holds is `remove_chambers.csv`'s to say. `excluded_chambers` in YAML is
ignored at load time. See
[exclusions](config-dfms-chambers.md#excluding-chambers).

## After saving

Validate:

```bash
pyflic lint my_experiment/
```

or use **Lint / migration check** on the Hub's Tools panel. The editor produces valid YAML,
but the linter also catches things it cannot know — a DFM with no matching CSV, a stale key
from an older version of pyflic.

If the Hub already has this member loaded, load it again — double-click its row in the
Project panel — to pick up your changes.

---

Related: [Creating the configuration file](getting-started-config.md) ·
[Configuration file structure](config-structure.md) · [Analysis Hub](app-hub.md)
