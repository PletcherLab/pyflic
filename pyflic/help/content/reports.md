# Reports

pyflic writes two PDF reports: one per member, and one per Project. Both are laid out the
same way — US Letter pages of one size, a running header, "Page n of N" in the footer, a
contents list on the first pages, tables that continue across pages with their header
repeated, and figures drawn to fit their box under the report's own heading.

## The experiment report

Written to the member's `analysis/experiment_report.pdf` by the Hub's **Run analysis**,
while its *Experiment report (PDF)* box is ticked (it is written last, so it reflects the
run), by **Analyze all**, by the `run_analysis` and `pdf_report` script actions and by
`write_experiment_report(exp)`. From Run analysis, its time courses use the Analyze panel's
**Bin size**.

The report describes the analysis the pipeline runs, so it first applies the design's
auto-removal — once, exactly as basic analysis does — and every result after the Quality
control section stands on the chambers that remain.

| Section | What is in it |
|---|---|
| **Cover** | The Experiment Type and layout, with the type's description (none for Custom); what was recorded (DFMs, dates, hours), the **Generation** of each DFM's data file, and for an optogenetic member where its **Lit time** came from; the treatments with their chamber counts, design factors, wells, Facets, whether licks are transformed; **At a glance** — data integrity, exclusions, a missing `Program.txt`, the optogenetic light QC and the type's own QC verdicts as coloured callouts; the contents |
| **1 Quality control** | Data integrity per DFM; for two-well layouts, simultaneous feeding per chamber and the largest bleeding response between wells; the Program timeline of each DFM that has one; every excluded chamber and why; for an optogenetic experiment, the optogenetic light QC — the program as read, the verdict per linkage group and the figure per DFM, then a **Rule check** subsection per Paradigm; then the Experiment Type's own checks |
| **2 Results** | First the type's **control comparisons**, when its definition names any; then the figures the assay is read by, each with its treatment statistics; for an optogenetic member, **Light behaviour** after them, one subsection per Paradigm Family, and **Paired and yoked** when a linkage group yokes chambers to a paired one |
| **Appendix A** | Every feeding-summary metric by treatment, for reference |
| **Appendix B** | The detection parameters and the `constants:` block the analysis used |

**The cover.** The **Generation** line says what shape of file each DFM wrote: 3 for
firmware 4.0.1 data, with the Paradigm columns filled (on an older recording, 2 for a
PyMCU 4 file with those columns empty, 1 for 27 columns from PyMCU 3.x; see
[Generations](concepts-optogenetics.md#generations)). DFMs that differ are listed
separately. **Lit time from** reads *OnMs, exact* (on an older recording, *OptoCol1,
sampled at 5 Hz*, the DFM's sample rate). *At a glance* warns about a recording whose
`Program.txt` is missing, and about a recording with no training information. Its
optogenetic light QC callout counts the linkage groups that failed or warned and, for each
flag, how many groups raised it (*wrong activation length in 2 groups, lockout broken in 1
group*); with nothing flagged it says every group's light was explained by its licks and
kept its Paradigm's rule.

**Quality control.**

- **Data integrity** gives each DFM its samples, start, hours, whether the sample index is
  continuous, the **Lost records** its index skipped, its data breaks and its firmware error
  flags. A DFM is *ok* with none of these, otherwise *warning*.
- **Simultaneous feeding** shows each chamber's samples licked on both wells at once
  *before* the dual-feeding correction — the crosstalk found — beside the count *after* it,
  which the correction drives to zero. With the correction off the two agree.
- **Bleeding between wells**, on a two-well layout: per DFM, the largest mean response of
  another well while one well's signal is above the cutoff. It is a *warning* when it is
  above the feeding threshold, since it would register as licks that never happened.
- **Program timeline**: in an optogenetic experiment, one figure per DFM (on an older
  recording, per DFM with a `Program.txt` section). Each lane is hatched where
  `Program.txt` sets the well `On`, grey where it sets it `Off`, and each restart of the
  DFM is a red band. See
  [Program timeline](plots-catalog.md#program-timeline).
- **Excluded chambers**: every chamber left out, by hand (`remove_chambers.csv`, the active
  exclusion group) or automatically (the design's `constants:` cutoffs), with its reason.
- **Optogenetic light QC**: the program as read, the verdict per linkage group and the
  *Light explained by licks* figure per DFM. With the recording's `Program.txt`, one
  **Rule check** subsection follows per Paradigm the program runs, in
  family order (open loop, closed loop, fixed length, non-feeding), then by name, each
  titled with the Paradigm, its family and its DFMs: the rule in words with its settings, a
  table of its checks per linkage group (the Activations judged and not judged, and the
  verdict), and its [rule figure](plots-catalog.md#rule-figures), the values that break the
  rule in red. A Paradigm with Acclimation has a second, closed-loop subsection for its
  Acclimation, and wells set `On` under another Paradigm get a *Wells set On* subsection
  of their own. If the rule subsections cannot be built, one red box says why and the rest
  of the report stands. On an older recording, a note names the rule checks that did not
  run, and why. See
  [Each Paradigm's own rule](concepts-optogenetics.md#each-paradigms-own-rule).

**Control comparisons.** A type whose definition names control comparisons opens the
Results with them (see [Control comparisons](concepts-experiment-types.md#control-comparisons)),
on licks, events and median bout duration (per well on a two-well layout):

- a **Role comparison** compares the chambers of two Roles within each treatment: a table,
  Welch's t-test, and a dot plot of the first metric. Paired against Yoked is the paired −
  yoked comparison the report already makes, moved to the front rather than repeated;
- a **Treatment control** compares each level of its factor with the Design's
  `control_level`, the other factors held at the same levels: a table, Dunnett's test with
  the control as the reference, and a dot plot. A design that names no control level gets a
  warning in its place.

**Results by layout and type.** A two-well experiment gets **Preference** — the preference
index and event PI by treatment, per Facet, and *Preference over time* — then
**Consumption** (licks, events and median bout duration at each well). A single-well
experiment gets Consumption only, with median and mean bout duration. Both end with **Over
time**: the time course of each report metric (licks, events and median duration by
default) in bins of the Bin size. A type with the Hedonic module adds **Hedonic feeding**:
its duration plot and the event-weighted duration table. A type with the Progressive Ratio
module replaces all of these, since the per-chamber figures would pool each paired
fly with its own yoked control, with the **cumulative difference curve**, the **paired −
yoked difference** in the Test phase (sucrose-well licks, PI and, where measured, sucrose
persistence), tested between treatments and against zero within each treatment, and the
**breaking point**: the lick-backed Test light events each paired fly completed before its
first pause longer than `pr_break_gap_min`, with the still-responding curve, a dot plot by
treatment, its per-group table and the per-DFM licks-per-light-period plots. The table
shows `BreakRequirement`, the Requirement the last counted event met, beside it. When
every group's Test Paradigm is a FixedRatio the section is titled **Fixed ratio:
activations earned**, and the progressive-only wording (the estimated requirement, "a
working progressive ratio climbs") appears only for groups whose Test Paradigm is
progressive. A design whose chamber groups are single chambers has no yoked controls: one
sentence says so where the paired − yoked difference and the cumulative difference curve
would be, and the breaking point follows. A DFM restart inside a group's Test window is
listed under the breaking point. Its Quality control section adds the training table and
the [light QC](concepts-progressive-ratio.md#light-qc) with its three figures per DFM.

**Light behaviour.** An optogenetic member of any type gets this section after the feeding
results, from its [light table](concepts-optogenetics.md#the-light-table), with one
subsection per [Paradigm Family](concepts-optogenetics.md#paradigm-families) its program
runs: open loop, closed loop, fixed length by its interval and ratio branches, non-feeding,
and the Acclimation. Each holds a dot plot per headline metric of its family, by treatment,
per Facet when there are Facets, from the chambers whose window ran it, with the treatment
comparisons. Under a type with the Light and dark module (Custom and Progressive Ratio
have it), open loop adds **Feeding in light and dark**: each open-loop chamber's licks per
minute in its own light and in the dark, joined by a line. A metric no chamber of the
subsection has is left out and named; a metric the same in every chamber, such as an
open-loop chamber's lit time, is named once as *the same in every chamber, so not compared
between treatments*. The Activation measures compare only the chambers that earn light
(Paired, Self, Mutual); a Yoked chamber keeps its lit time beside its partner's. A
Progressive Ratio member compares its paired chambers only. A member without
`Program.txt`, or with an older recording, keeps one section with four metrics:
Activations, lit time, Firmware Events and the end of Acclimation.

**Paired and yoked.** When a linkage group lights one Paired chamber by its fly's feeding
and Yoked chambers with it, under a type with the Paired and yoked module (Custom has it;
Progressive Ratio keeps its own paired − yoked difference), the Results end with the paired
fly's feeding minus the mean of its yoked partners' (licks, events and median duration, per
well on a two-well layout), by treatment: tested against zero, then compared between
treatments, with the paired-against-yoked figure, one per Facet. When the type's Role
comparison is Paired against Yoked, this section opens the Results instead. See
[Paired and yoked designs](concepts-optogenetics.md#paired-and-yoked-designs).

**Statistics.** Under each figure: two treatments, Welch's t-test; more, Tukey HSD on every
pair; p < 0.05 highlighted. One observation per chamber (per chamber group for a
Progressive Ratio difference or breaking point). *Mean* and *n* read as the first treatment
named / the second, and *Difference* is the first minus the second. The breaking point adds
**p (log-rank)**, which treats a censored count as the lower bound it is. The paired −
yoked difference adds a table testing it against zero in each treatment: the paired t-test,
which is a one-sample t-test on the differences, with the Wilcoxon signed-rank test beside
it.

**One treatment, or nothing to draw.** A design with a single treatment, such as one factor
with one level, is reported as one: Results open with a sentence saying nothing is compared
between treatments, and no comparison table follows. A figure whose metric no chamber has a
value for, such as a bout duration when no feeding event was detected, becomes a grey note
saying so and why, without its caption; Appendix A leaves such metrics out of its grid and
names them. A red note is kept for a figure that failed. Dot plots spread points only across
a treatment, never up or down, so chambers with equal values sit on that value.

## The Project Report

Written to `<project>/<name>_report.pdf` by **Create report** and the `project_report`
script action. It reads the members' saved results only — it never analyses a member — and
builds the Combined Analysis first when there is none.

| Section | What is in it |
|---|---|
| **Cover** | The Project, its type and layout, its members, pooled chambers and treatments, factors and Facets; **At a glance** — missing members, exclusions, the light QC's flagged groups, and differences between members; the contents |
| **1 Members** | One row per member: DFMs, the **Generation** of its recordings, chambers analysed and excluded, whether its analysis is current, whether it is faceted, whether it has a report, and a **Note** for a member with no training information |
| **2 Quality control** | Every excluded chamber across members with its reason; every group a light QC flagged — Progressive Ratio chamber groups and optogenetic linkage groups, marked by source — and whether the pooled numbers include it |
| **3 Results** | The type's control comparisons first, pooled across members, when its definition names any; then the type's pooled figures (the Plot Editor's styles), the type's own pooled figures (Progressive Ratio: the breaking point, led by its still-responding curve, and the paired − yoked difference), and the statistics as tables — pooled test and linear mixed model (DFM nested within experiment) side by side, with the log-rank test for the breaking point and the paired − yoked difference tested against zero; then **Light behaviour** for a Project with optogenetic members, by Paradigm Family, and **Paired and yoked** when a member has a yoked group, under a type with the Paired and yoked module |
| **Appendix** | The Design's detection parameters and constants |

The **Generation** column is read from each member's saved outputs, so it needs no load. The
**Note** column appears only when some member needs one: *no training information*, for a
member recorded with older DFM firmware (generation 2).

**Light behaviour** pools every optogenetic member's light table
(`<project>_LightSummary.csv`) the way the feeding metrics are pooled, one subsection per
Paradigm Family the members' programs run (each member's `data/Program.txt`): a faceted
figure per headline metric of the family, open loop's *Feeding in light and dark* (with the
Light and dark module), and the treatment comparisons, per chamber, pooled test beside the
mixed model. Members with older recordings contribute lit time only. A Project whose
members' programs give no family keeps one section with four metrics, as a member does.

**Paired and yoked** pools the members' yoked comparisons
(`<project>_YokedComparison.csv`): the differences by treatment against zero and between
treatments, with the mixed models, and the paired-against-yoked figure.

For a Progressive Ratio Project the breaking point section adds the median
`BreakRequirement` per treatment beside the median breaking point, when any member recorded
one. When every pooled group ran a FixedRatio Test, the section is titled **Fixed ratio:
activations earned**. When every pooled chamber group is a single chamber, the report says
once that the design has no yoked controls, and the Combined Analysis's
`<project>_Stats.txt` reads *Paired − yoked difference: none; the design has no yoked
controls*.

An [AI summary](concepts-ai-summary.md) follows the results when the `project_report` script
action's **Include AI narrative** (`ai_summary: true`) is on and one has been generated (the
AI panel's **Generate narrative**); it is marked as written by a language model. **Create
report** leaves it out.

See [Plot catalogue](plots-catalog.md) for the figures themselves and
[Excluding chambers](concepts-exclusions.md) for what leaves the analysis and how.
