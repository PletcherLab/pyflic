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

## Sections

**Parameters** — the detection parameters. Each field carries a `?` that opens
[Parameter reference](reference-parameters.md) at that parameter, so you can read what a
value does without leaving the form.

**Factors** — declare a factorial design. Adding a factor here regenerates the chamber
assignment fields to match, which is the main reason to use the editor rather than hand
editing: adding a factor by hand means rewriting every chamber assignment in the file. See
[Factorial designs](config-factors.md).

**DFMs** — one tab per device, each with its own parameter overrides and chamber
assignments. Per-DFM overrides are how counterbalancing is set up; see
[`pi_direction`](reference-parameters.md#pi_direction).

## Things worth knowing

**`chamber_size` is required and cannot be inferred.** It determines which physical wells
form a chamber, so the editor asks for it before it can lay out the chamber fields. Setting
it wrong silently reinterprets the whole plate.

**Chamber count follows chamber size.** `chamber_size: 1` gives 12 chambers per DFM;
`chamber_size: 2` gives 6. Change the size and the chamber fields change with it.

**The editor writes canonical parameter names.** Older aliases such as `link_gap` and
`tasting_low` are still accepted when read, but what gets written is the canonical name.

**Exclusions do not live here.** `excluded_chambers` in YAML is ignored at load time.
Chamber exclusions belong in `remove_chambers.csv` — see
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
