# Plot Editor

A **Project-level** tool for building journal-ready vector figures.

```bash
pyflic plots my_project/
```

Opening a member redirects up to its Project — a publication figure is a statement about
the pooled result, not about one recording.

The Editor is presentation only. It never alters a `flic_config.yaml`; everything it writes
goes to `plot_specs.yaml` and `figures/`.

## Spec and Style

Two things define a figure, and they are deliberately separate:

- a **Plot Spec** — this figure's *content*: axis labels, which facets and treatments to
  include, their order and display names, y limits, a reference line
- a **Plot Style** — a *look*, shared by every figure that references it: size, theme,
  fonts, point and mean styling, and the treatment → colour mapping

Both live in `<project>/plot_specs.yaml`. A Style is what makes a Project's figures look
like one set rather than a folder of unrelated pictures, so styles are named and reused
rather than set per figure.

Colours are keyed by the treatment's **original** name, so renaming a treatment for one
figure never changes its colour.

## Two figure families

| Family | Shape | Extra controls |
|---|---|---|
| `faceted_<metric>` | x = Treatment, one panel per facet, points + mean ± SEM | facet inclusion, free y |
| `timecourse_<metric>` | x = time bin, one line per treatment, SEM ribbon | bin size, ribbon |

The Content tab swaps its controls to match the selected plot's family. The Style tab is
identical for both.

A faceted figure needs the Combined Analysis (build it from the Hub's Project panel). A
time course needs saved binned summaries — run a binned CSV in each member first.

Preference-index plots are offered only for two-well projects; they have no meaning with a
single well.

## Marking members

**Mark members by point shape** gives each member its own marker in a pooled figure,
so batch structure is visible inside the pooled cloud. It is on by default for pooled report
figures, because seeing that structure is most of why one pools at all. It does not apply to
a time course, which plots treatment means.

## Preview and output

The preview is rendered by the same Spec + Style that saving uses — what you see is what
lands in `figures/`.

**Render figures** writes SVG (or PDF). SVG uses `svg.fonttype='none'`, so labels arrive in
Illustrator as live, editable text rather than outlined paths.
