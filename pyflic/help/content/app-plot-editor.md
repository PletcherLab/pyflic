# Plot Editor

A **Project-level** tool for building journal-ready vector figures.

```bash
pyflic plots my_project/
```

Opening a member redirects up to its Project — a publication figure is a statement about
the pooled result, not about one recording.

The Editor is presentation only. It never alters a `flic_config.yaml`; everything it writes
goes to `plot_specs.yaml` and `figures/`.

## One plot at a time

The **Plot** picker in the toolbar chooses which figure you are editing; the preview shows
that one, and the Content and Style tabs edit that one. Switching plots keeps whatever you
changed on the plot you left.

There is nothing to include or exclude. Every figure the Project's data supports is part of
its set, so `plot_specs.yaml` records *how* each figure is drawn, never *whether* it is; a
plot you have never opened is simply drawn from its defaults. **Render figures** writes them
all, skipping any whose data is missing.

**Restore defaults** discards the current plot's saved Spec and starts it over. It is
content only — Styles are shared, so resetting one figure never repaints the rest.

## One panel, four groups

Everything that shapes the current figure is on one scrolling panel, because a figure is
one thing and paging half its knobs behind a tab only asked which half was in force:

| Group | Holds |
|---|---|
| **Style (shared across plots)** | the named Style and every look decision in it |
| **This plot** | title, axis labels, y range, and the family-specific rows |
| **Facets** | which phases the figure panels over (faceted family only) |
| **Treatments** | one row each: shown or not, its printed Label, its Colour |

Sizing can be per **figure** or per **facet**. *Figure (mm)* sets the whole image;
*Facet width* and *Facet height* instead size one panel and let the figure grow with the
number of facets, so a three-facet plot is not three squeezed panels in the width of a
one-facet one. Both read `off` at zero, which means the figure size above applies.

In the Treatments table the first column is the treatment's **original** name — the one the
data and the Style's colours are keyed by, so it is shown rather than edited — and the
Label column is what the figure prints. Colours live in the Style, so changing one repaints
every figure sharing that Style.

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

The **Facets** group and the bin-size row swap places to match the selected plot's family;
everything else is the same for both. The check boxes on the panel are *within* one
figure — which facets it shows and which treatments it draws — not a choice between
figures.

A faceted figure needs the Combined Analysis (build it from the Hub's Project panel). A
time course needs saved binned summaries — run a binned CSV in each member first.

Preference-index plots are offered only for two-well projects; they have no meaning with a
single well.

## Progressive Ratio projects

A Progressive Ratio Project adds **Paired − yoked cumulative licks since training**
(`timecourse_pr_diff`), the type's headline figure, drawn from each member's
`pr_cumulative_diff.csv` rather than the binned summary — run basic analysis in each member
first. Its x axis is minutes since each chamber group's training end, zero is drawn as the
reference line, and the mean is shown only over the range every group covers.

The faceted figures (`faceted_licks`, `faceted_pi`, …) still pool the paired and yoked
flies of each treatment. For a paired-versus-yoked statement use the Project Report, whose
Progressive Ratio section plots and tests the within-group differences and the breaking
point. See [Progressive Ratio experiments](concepts-progressive-ratio.md) and
[Reports](reports.md).

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

Press **F1**, or the `?` at the end of the toolbar, to open this topic.
