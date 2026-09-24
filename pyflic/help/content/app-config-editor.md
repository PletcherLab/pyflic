# Config Editor

```bash
pyflic config
```

A form-based editor for `flic_config.yaml`. It knows which keys exist, which values are
allowed, and which are required, so it writes a valid configuration rather than leaving you
to discover a typo three steps later.

Use it for your first configuration. Hand-editing YAML is fine once you know the format —
see [Configuration file structure](config-structure.md) — but there is no advantage to
starting there.

## The two tabs

**Experiment** — everything that applies to the recording as a whole: the
Experiment Type, the Chamber Layout it implies, well names, the detection
parameters, and the factors of your design. Each parameter field carries a `?`
that opens [Parameter reference](reference-parameters.md) at that parameter, so
you can read what a value does without leaving the form.

Declaring a factor here regenerates the chamber assignment fields on the other
tab to match, which is the main reason to use the editor rather than hand
editing: adding a factor by hand means rewriting every chamber assignment in
the file. See [Factorial designs](config-factors.md).

**DFMs & Chambers** — one tab down the left side per device, each with its own
chamber assignments and parameter overrides. **Add DFM** and **Remove DFM** sit
below the list; Remove acts on the DFM you have selected and names it before it
discards anything. Per-DFM overrides are how counterbalancing is set up; see
[`pi_direction`](reference-parameters.md#pi_direction).

## Things worth knowing

**The Experiment Type owns the Chamber Layout.** Choose *Hedonic Feeding* or
*Progressive Ratio* and the layout is fixed for you — shown, but not editable,
because those assays are two-well by definition. Only a **Custom Experiment**
chooses its own layout. The editor writes `chamber_layout:` for a Custom
Experiment and `experiment_type:` for a typed one, and `params.chamber_size` for
neither: it is derived from the layout, never stored.

**Chamber count follows the Chamber Layout.** Single-well gives 12 chambers per
DFM; two-well gives 6. Change the layout and the chamber fields change with it —
and if that would discard assignments you have already made, the editor names
them and asks first.

**Auto-filter thresholds show the type's defaults as grey placeholder text.**
Leaving a threshold blank does not mean "no filtering": it means the Experiment
Type's default applies. Type a value only to override it. A Custom Experiment
has no defaults, so there a blank field really does skip the filter.

**Factor assignments are positional and must be complete.** With two factors
declared, every assigned chamber needs a level in both columns — the columns are
read in declaration order, so a half-filled row is an incomplete assignment, not
a shorter one. The editor marks the gap and leaves that chamber out of the file
rather than writing something that would read back wrong.

**Problems are counted on the tab they live on.** A ⚠ and a number on a tab
label means something there needs attention — a Hedonic experiment with no well
names, a chamber missing a factor level. Saving with problems outstanding is
allowed; the editor lists them first so you are choosing to.

**The editor writes canonical parameter names.** Older aliases such as `link_gap`
and `tasting_low` are still accepted when read, but what gets written is the
canonical name. The same goes for configurations written before pyflic split
Experiment Type from Chamber Layout: `experiment_type: two_well` still opens — as
a Custom Experiment with that layout — and saving migrates it.

**Progressive Ratio adds a paired-chamber picker.** With *Progressive Ratio* chosen, each
DFM tab shows **Paired chamber per chamber group**: one picker per group (1+2, 3+4, 5+6)
naming its paired chamber, written as `paired_chambers:`. The other chamber is yoked and
never written. Give both chambers of a group the same treatment, and set **PI Direction**
on the DFM to the side the sucrose well (well A) is on. See
[Progressive Ratio experiments](concepts-progressive-ratio.md#chamber-groups-paired-and-yoked).

**Progressive Ratio has its own settings.** Below the auto-filter thresholds, a
*Progressive Ratio* section holds the type's own constants: whether a chamber group whose
training never completed, or whose light QC failed, leaves the analysis; the light QC's
thresholds; and the breaking point's **Break gap** (`pr_break_gap_min`) and **Test window
cap** (`pr_test_window_min`). They follow the thresholds' rule. A blank field shows the
type's default as grey text and keeps it, and a switch left on *default* does the same;
only a value you type or pick is written under `global.constants`. A value out of range,
such as a gap of 0, is counted on the tab and listed before saving, and pyflic refuses to
load a config that states one. What each setting does is in
[the light QC](concepts-progressive-ratio.md#light-qc) and
[the breaking point](concepts-progressive-ratio.md#breaking-point).

**Optogenetics has a setting and a section.** *Optogenetics* on the Experiment tab is
`auto`, `yes` or `no`, and each DFM tab has its own picker, which inherits unless you pick.
Once the experiment is known to be optogenetic — the setting is `yes`, `data/` holds a
`Program.txt`, or a run has already written its light QC — an *Optogenetic light QC*
section shows the thresholds, shared by every type, under the same rule as the rest: grey
defaults, and only what you type is written. See
[Optogenetic experiments](concepts-optogenetics.md).

**A member's config shows the design read-only.** Opened on a Project's member, the
editor fills the global settings from `project.yaml`, marks them read-only with a banner,
and writes only `dfms:` — see [Projects and members](concepts-project.md).

**Exclusions do not live here.** `excluded_chambers` in YAML is ignored at load
time. Chamber exclusions belong in `remove_chambers.csv` — see
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
Project panel — to pick up your changes. **F1**, or the editor's **Help** menu, opens this
topic.

---

Related: [Creating the configuration file](getting-started-config.md) ·
[Configuration file structure](config-structure.md) · [Analysis Hub](app-hub.md)
