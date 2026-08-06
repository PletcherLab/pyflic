# Script Editor

A visual editor for the `scripts:` section of your configuration file. It writes the same
YAML you could write by hand, but it knows which actions exist and which parameters each
one takes, so you cannot invent an action that does not exist or misspell a parameter.

Open it from the **Scripts** card in the hub.

## Layout

Three panes side by side, with a YAML preview beneath:

| Pane | Purpose |
|---|---|
| **Palette** (left) | Every available action, grouped by category. Add one to the script from here. |
| **Canvas** (centre) | The steps of the current script, in order. Reorder and remove them here. |
| **Inspector** (right) | The parameters of the selected step, with its defaults and notes. |
| **YAML preview** (bottom) | Live view of the YAML being written. |

Above them sits the **script switcher**: a dropdown of the scripts in this configuration,
with buttons to create, rename, and delete.

Watch the preview while you work. It is the fastest way to build an accurate mental model
of the file format, and it means the editor never becomes a black box — you can always see
exactly what will be saved.

## Working in it

1. Pick the script to edit, or create a new one, in the switcher.
2. Click an action in the palette to append it to the canvas.
3. Select a step on the canvas; set its parameters in the inspector.
4. Reorder steps until the sequence is right — order matters; see below.
5. **Save**.

A dirty indicator next to the file path shows unsaved changes. **Reload** discards them and
re-reads the file from disk, which is what you want if you have also edited the YAML in a
text editor.

## Order matters

The canvas is a sequence, not a set, and several steps depend on what came before:

- `load` must come first — everything else operates on the loaded experiment.
- `remove_chambers` must precede any analysis whose results should exclude those chambers.
- `write_summary` after `remove_chambers` records what was actually excluded.
- Plot and analysis steps can be in any order among themselves.

## Blank parameters inherit

A parameter left blank in the inspector is not an error and is not zero — it inherits, from
the script level and then from the hub. The inspector shows each parameter's default and a
note describing it, so you can tell an inherited blank from a value you meant to set.

This is why a script can be re-run over a different time window just by changing a spinbox
in the hub: leave `start` and `end` blank on the steps and they follow the hub. Pin them on
the step when the window is part of the analysis rather than a thing you vary.

## Naming a script `batch`

A script named exactly `batch` is what makes its directory a **batch target** — see
[Running many projects at once](scripts-batch.md). The name is also the default exclusion
group for a bare `remove_chambers` step, so pick names deliberately and keep them
consistent with the groups in your `remove_chambers.csv`.

---

Related: [What a script is](scripts-overview.md) · [Script actions](scripts-actions.md)
