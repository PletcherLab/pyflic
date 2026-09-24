# Creating the configuration file

`flic_config.yaml` tells pyflic what your experiment *means*: which DFMs to read, what kind
of assay it is, which treatment each chamber received, and which detection parameters to
use. Everything else follows from it.

You do not have to write it by hand.

## The easy way

```bash
pyflic config
```

This opens the **Config Editor**, a form-based editor that writes valid YAML for you. It
knows which keys are required, offers the accepted values, and saves into your experiment
directory. If you have never written YAML before, use this and skip the rest of this page
until something goes wrong.

Inside a Project you rarely start from nothing: the Hub's **Create member…** scaffolds a
member's config from the Project's design, and the member only needs its chamber
assignments. See [Config Editor](app-config-editor.md) for a tour of the form.

## The smallest config that works

If you would rather start from text, this is a complete, valid two-well configuration:

```yaml
global:
  chamber_layout: two_well
  params:
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

- **The assay.** Either name an `experiment_type` — `Hedonic` or `ProgressiveRatio` — which
  fixes the chamber layout for you, or leave it out (a **Custom** experiment) and say
  `chamber_layout: single_well` or `two_well`. See
  [Experiment types](concepts-experiment-types.md).
- **The layout decides what a chamber is**: 12 one-well chambers per DFM, or 6 two-well
  ones. pyflic derives `chamber_size` from it; you never write that yourself.
- **`chambers`** maps each chamber number to the treatment that chamber received. This is
  the part only you know.

A typed experiment adds what its type requires: `Hedonic` and `ProgressiveRatio` need
`well_names` for wells A and B, and `ProgressiveRatio` needs `paired_chambers` on every DFM
([Progressive Ratio experiments](concepts-progressive-ratio.md)).

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
| Split the recording into phases | `global.facet_cutoffs` | [Facets](concepts-facets.md) |
| Use a factorial design | `global.experimental_design_factors` | [Factorial designs](config-factors.md) |
| Drop bad chambers | `remove_chambers.csv` | [DFMs, chambers and exclusions](config-dfms-chambers.md) |
| Automate a whole pipeline | `scripts:` | [What a script is](scripts-overview.md) |

---

Next: **[Running your first analysis](getting-started-first-run.md)**
