# Reports

pyflic writes two PDF reports: one per member, and one per Project. Both are laid out the
same way — US Letter pages of one size, a running header, "Page n of N" in the footer, a
contents list on the first pages, tables that continue across pages with their header
repeated, and figures drawn to fit their box under the report's own heading.

## The experiment report

Written to the member's `analysis/experiment_report.pdf` by the Hub's **PDF report**
button, by **Analyze all**, by the `pdf_report` script action and by
`write_experiment_report(exp)`.

The report describes the analysis the pipeline runs, so it first applies the design's
auto-removal — once, exactly as basic analysis does — and every result after the Quality
control section stands on the chambers that remain.

| Section | What is in it |
|---|---|
| **Cover** | What was recorded (DFMs, dates, hours), the treatments with their chamber counts, design factors, wells, Facets, whether licks are transformed; **At a glance** — data integrity, exclusions and the type's own QC verdicts as coloured callouts; the contents |
| **1 Quality control** | Data integrity per DFM (samples, hours, data breaks, firmware error flags, a continuous sample index); for two-well layouts, simultaneous feeding per chamber and the largest bleeding response between wells; every excluded chamber and why; then the Experiment Type's own checks |
| **2 Results** | The figures the assay is read by, each with its treatment statistics |
| **Appendix A** | Every feeding-summary metric by treatment, for reference |
| **Appendix B** | The detection parameters and the `constants:` block the analysis used |

**Results by type.** A two-well experiment gets **Preference** — the preference index and
event PI by treatment, per Facet, and PI over time — then **Consumption** (licks, events and
median bout duration at each well). A single-well experiment gets Consumption only. A
Hedonic experiment adds its duration plot and the event-weighted duration table. A
Progressive Ratio experiment replaces the per-chamber figures, which would pool each paired
fly with its own yoked control, with the **cumulative difference curve**, the **paired −
yoked difference** in the Test phase, tested between treatments and against zero within each
treatment, and the **breaking point**: the lick-backed Test light events each paired fly
completed before its first pause longer than `pr_break_gap_min`, with the still-responding
curve, its per-group table and the per-DFM licks-per-light-period plots. Its Quality control
section adds the training table and the
[light QC](concepts-progressive-ratio.md#light-qc) with its three figures per DFM.

**Statistics.** Under each figure: two treatments, Welch's t-test; more, Tukey HSD on every
pair; p < 0.05 highlighted. One observation per chamber (per chamber group for a
Progressive Ratio difference or breaking point). *Mean* and *n* read as the first treatment
named / the second, and *Difference* is the first minus the second. The breaking point adds
**p (log-rank)**, which treats a censored count as the lower bound it is. The paired −
yoked difference adds a table testing it against zero in each treatment: the paired t-test,
which is a one-sample t-test on the differences, with the Wilcoxon signed-rank test beside
it.

## The Project Report

Written to `<project>/<name>_report.pdf` by **Create report** and the `project_report`
script action. It reads the members' saved results only — it never analyses a member — and
builds the Combined Analysis first when there is none.

| Section | What is in it |
|---|---|
| **Cover** | The Project, its members, pooled chambers and treatments, factors and Facets; **At a glance** — missing members, exclusions, the type's QC flags; the contents |
| **1 Members** | One row per member: DFMs, chambers analysed and excluded, whether its analysis is current, whether it has a report |
| **2 Quality control** | Every excluded chamber across members with its reason; for Progressive Ratio, every chamber group the light QC flagged and whether the pooled numbers include it |
| **3 Results** | The type's pooled figures (the Plot Editor's styles), the type's own pooled figures (Progressive Ratio: the breaking point, led by its still-responding curve, and the paired − yoked difference), and the statistics as tables — pooled test and linear mixed model (DFM nested within experiment) side by side, with the log-rank test for the breaking point and the paired − yoked difference tested against zero |
| **Appendix** | The Design's detection parameters and constants |

An optional AI summary follows the results when one has been written; it is marked as
written by a language model.

See [Plot catalogue](plots-catalog.md) for the figures themselves and
[Excluding chambers](concepts-exclusions.md) for what leaves the analysis and how.
