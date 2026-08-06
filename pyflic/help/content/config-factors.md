# Factorial designs

Most experiments vary more than one thing at once. Rather than inventing composite
treatment names like `Paired_Chrim` and unpicking them later, declare the factors and let
pyflic keep them separate.

## Declaring factors

```yaml
global:
  experimental_design_factors:
    paired: [Paired, Unpaired]
    genotype: [Chrim, WCS]
```

Each key is a factor name; its value is the list of levels that factor can take.

## Assigning chambers

Once factors are declared, chamber assignments become **comma-separated levels, in the
same order the factors are listed**:

```yaml
dfms:
  1:
    chambers:
      1: Paired,Chrim
      2: Unpaired,Chrim
      3: Paired,WCS
      4: Unpaired,WCS
```

Chamber 1 is tagged `paired=Paired` and `genotype=Chrim`.

The order is positional, not by name. With the declaration above, `Chrim,Paired` is not a
clever reordering — it is an error, because `Chrim` is not a level of `paired`. If you
reorder the factor declarations later, every chamber assignment in the file must be
reordered to match. Run `pyflic lint` after any such edit.

## Why bother

Two things become possible that composite names cannot give you.

**Statistics on each factor separately.** With factors declared, pyflic can fit models with
main effects and interactions — the effect of pairing, the effect of genotype, and whether
pairing acts differently in the two genotypes. A single fused treatment label collapses all
of that into one categorical variable and the interaction becomes unaskable.

**Plots faceted by factor.** Figures can be split by one factor and coloured by another,
which is the normal way to present a two-factor design.

## Keeping it consistent

- Every chamber must specify a level for **every** declared factor. Partial assignments
  are not allowed — there is no "not applicable" level unless you declare one.
- Levels are matched exactly, so `WCS` and `wcs` are different levels and one of them is
  almost certainly a typo.
- Adding a factor means editing every chamber assignment in the file. Decide your design
  before you write the configuration, or use the [Config Editor](app-config-editor.md),
  which regenerates the assignments for you.

## Without factors

If you declare no factors, chamber assignments are plain treatment names:

```yaml
chambers:
  1: Sucrose
  2: Water
```

This is right for a one-factor experiment. Reach for factors when you genuinely cross two
or more variables — not to avoid typing a longer name.

---

Related: [DFMs, chambers and exclusions](config-dfms-chambers.md) ·
[Configuration file structure](config-structure.md)
