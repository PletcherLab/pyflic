# Creating the configuration file

`flic_config.yaml` tells pyflic what your experiment *means*: which DFMs to read, how
wells are grouped into chambers, which treatment each chamber received, and which
detection parameters to use. Everything else follows from it.

You do not have to write it by hand.

## The easy way

```bash
pyflic config
```

This opens the **Config Editor**, a form-based editor that writes valid YAML for you. It
knows which keys are required, offers the accepted values, and saves into your project
directory. If you have never written YAML before, use this and skip the rest of this page
until something goes wrong.

See [Config Editor](app-config-editor.md) for a tour of the form.

## The smallest config that works

If you would rather start from text, this is a complete, valid two-well configuration:

```yaml
global:
  experiment_type: two_well
  params:
    chamber_size: 2
    pi_direction: left

dfms:
  1:
    chambers:
      1: Sucrose
      2: Water
      3: Sucrose
      4: Water
      5: Sucrose
      6: Water
```

Three things are doing the work:

- **`experiment_type`** picks the analysis. See
  [Experiment types](concepts-experiment-types.md) for the four options.
- **`chamber_size`** is `1` or `2` — wells per chamber. This one is **required**; pyflic
  raises an error rather than guessing, because getting it wrong silently reinterprets
  your whole plate.
- **`chambers`** maps each chamber number to the treatment that chamber received. This is
  the part only you know.

Everything else has a default. See [Parameter reference](reference-parameters.md) for what
those defaults are and when to change them.

## Checking it before you run

```bash
pyflic lint my_experiment/
```

The linter validates your configuration against the schema and reports errors and
warnings with line numbers where it can. Run it whenever you edit the file by hand — a
mistyped key is far easier to find here than in a confusing plot three steps later.

## What to add next

Once the basic file works, these are the additions most experiments need:

| You want to… | Add | Described in |
|---|---|---|
| Change how bouts are detected | `global.params` entries | [Parameter reference](reference-parameters.md) |
| Label the two wells in plots | `global.well_names` | [Configuration file structure](config-structure.md) |
| Use a factorial design | `global.experimental_design_factors` | [Factorial designs](config-factors.md) |
| Drop bad chambers | `remove_chambers.csv` | [DFMs, chambers and exclusions](config-dfms-chambers.md) |
| Automate a whole pipeline | `scripts:` | [What a script is](scripts-overview.md) |

---

Next: **[Running your first analysis](getting-started-first-run.md)**
