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

**Exclusions do not live here.** `excluded_chambers` in YAML is ignored at load
time. Chamber exclusions belong in `remove_chambers.csv` — see
[exclusions](config-dfms-chambers.md#excluding-chambers).

## After saving

Validate:

```bash
pyflic lint my_experiment/
```

or use **Lint config** in the hub's Tools card. The editor produces valid YAML, but the
linter also catches things it cannot know — a DFM with no matching CSV, a stale key from an
older version of pyflic.

If the hub is already open on this project, click **Reload config** there to pick up your
changes.

---

Related: [Creating the configuration file](getting-started-config.md) ·
[Configuration file structure](config-structure.md) · [Analysis Hub](app-hub.md)
