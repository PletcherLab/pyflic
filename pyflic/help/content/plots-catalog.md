# Plot catalogue

What each figure shows, when it is produced, and what to look for in it.

## QC plots

Written to the member's `qc/` folder for every DFM whenever QC runs — the Hub's **Run QC**,
the `run_qc` script action, the Project's **Analyze all**, or `execute_basic_analysis()`
from Python. Each is an item of the QC panel's QC Set: Run QC and `run_qc` write the ones
ticked there (see [QC panel](app-hub.md#qc-panel)), while Analyze all and
`execute_basic_analysis()` write them all. The QC panel's **View QC plots** opens the saved
ones in the output area, one tab per DFM and kind. These are diagnostics, not figures for a
paper.

| Plot | Saved as | Shows |
|---|---|---|
| Raw signal | `qc/raw_signal/DFM{n}_raw.png` | Every well's unprocessed signal |
| Baselined signal | `qc/baselined/DFM{n}_baselined.png` | Every well after baseline subtraction, with thresholds drawn |
| Cumulative licks | `qc/cumulative_licks/DFM{n}_cumulative_licks.png` | Running lick total per well over time |
| Light explained by licks | `qc/opto/opto_light_dfm{n}.png` | Optogenetic experiments only: each linkage group's lit time, explained and unexplained by trigger-well licks |
| Program timeline | `qc/program/program_timeline_dfm{n}.png` | Optogenetic experiments only, a DFM with a `Program.txt` section or firmware 4.0.0 data: its Program Intervals, and each well's Acclimation, Activations and light |

The **baselined signal with thresholds** is the most useful one in the set. It puts your
`feeding_threshold` and `feeding_minimum` lines directly on the trace, so whether they sit
in a sensible place is a visual question rather than a guess.

**Cumulative licks** curves reveal timing at a glance: a steady slope is steady feeding, a
plateau is a fly that stopped, a step is a burst. A well whose curve never leaves zero is
inactive and should be excluded.

**Light explained by licks** (`plot_opto_light`) is drawn for an optogenetic experiment of
any type, by Run QC and by basic analysis. One row per linkage group. The left panel is its lit
time per bin, stacked: grey-blue where a trigger-well lick or touch within the light's
Hold explains it, red where nothing does, purple where it was lit while every threshold
was `Off` (-1), and light grey where the light is not judged by licks (open loop,
NonFeeding). The right panel is orange: on a recording older than firmware 4.0.0, when
`data/Program.txt` allows it, the time the emulated firmware trigger read the trigger wells
above threshold with no lick or touch; on a firmware 4.0.0 recording, titled *firmware
events, no lick*, the light of Activations the firmware counted with no lick or touch at
their onset. Lit time there is `OnMs`, in milliseconds. Red beside orange in the same bins
is a drifting sensor. The strip carries the group's verdict. See
[Optogenetic experiments](concepts-optogenetics.md).

### Program timeline

**Program timeline** (`plot_program_timeline`) is drawn, in an optogenetic experiment, for
every DFM that has something to draw: firmware 4.0.0 data, or a `Program.txt` section.
**Run QC** and **Analyze all** write it for each such DFM; the `basic_analysis` script
action, which skips the QC reports, still writes it for each firmware 4.0.0 DFM. A recording
whose light never came on (every well Off) gets none.

- **The band at the top** is the program, one block per Program Interval, coloured by
  Paradigm type and labelled with the Paradigm's name (and `C<n>`, the Cycle, when the
  program repeats). Light grey is the baseline before the first instruction reached the DFM,
  hatched white an instruction outside the program, the palest grey `Idle` (PyMCU's own
  Paradigm, every well `Off`, known without a `Program.txt`), and dark grey a Paradigm whose
  type `Program.txt` does not give.
- **Below it, one lane per well**, labelled with its chamber and Role. A trigger well's label
  is bold and gives its threshold (*trigger 20*); a well lit regardless says *On*. Pale
  orange shading is Acclimation (Phase `A`), each black tick an Activation, and the blue
  strip the share of each bin the LED was on. On a firmware 4.0.0 recording a lane is also
  hatched where `Program.txt` sets the well `On`, and grey where it sets it `Off`, Program
  Interval by Program Interval. On an older recording each tick is a trigger well's light
  coming on, which under a ratio or fixed-interval Paradigm is one Activation.
- **The strip's scale fits the program.** It runs from 0 to 100% of a bin when the program
  lights a well open loop, where a lane lit throughout is what to check. Otherwise it runs
  to the DFM's brightest bin and the colour bar says *scaled to this DFM*, because light
  that follows the fly fills only a few percent of a bin.
- **A red band labelled "restart"**, across the band and every lane, marks each restart of
  the DFM (firmware 4.0.0): from the last record before the gap in `Index` to the first row
  under the instruction again. A restart that only `Messages.csv` names has no rows to place
  it by, and is not drawn.

How to read it:

- **Each trigger well should leave Acclimation**, its shading ending, within the
  interval that started it. A well still shaded at the end never completed Acclimation.
- **Ticks should come where the fly fed.** Under a ratio Paradigm they thin out as the
  Requirement rises. A dense comb of ticks with a flat cumulative-licks curve is the light
  following the sensor.
- **A yoked well has no ticks** and the same blue strip as its paired partner. Ticks on a
  yoked well, or a strip that differs, is broken yoking.
- **The band should match the program you meant to run.** A hatched block after the start
  is the DFM running something outside the program. A dark-grey block is a Paradigm the
  `Program.txt` in `data/` does not know, which usually means it is the wrong file; every
  block but `Idle` is dark grey when there is no `Program.txt` at all.
- **A well set `On` should be lit through its hatching**, and a well set `Off` should have
  no light of its own in its grey: light there that its group's trigger well did not cause
  is a fault.
- **After a red restart band** every trigger well starts again: Acclimation shading and the
  ticks of a new Requirement series.

On a recording older than firmware 4.0.0 the band is laid out from the program's clock
times, approximate, and the lanes show the sampled `OptoCol1` light only: no Acclimation, no
ticks, no `On` or `Off` shading and no restarts. The bin width is the plot action's
`binsize` (10 minutes by default, and always 10 minutes from the Hub's Run QC). See
[Optogenetic experiments](concepts-optogenetics.md#the-program-timeline).

### Rule figures

In the experiment report's *Optogenetic light QC*, on a firmware 4.0.0 recording with its
`Program.txt`: each Paradigm's **Rule check** subsection ends with its rule figure, one
panel per DFM, that puts the measured light against the rule's setting. Light that keeps the
rule is grey-blue; light that breaks it, by more than `opto_activation_tolerance_ms` (one
record more for an Activation timed only to the record), is **red**, as bars and as ticks
under the axis, so that one failing Activation among hundreds still shows. Only the
Activations the checks judged are drawn.

| Family | The figure |
|---|---|
| Closed loop | lit time per Activation, dashed at the Hold and at `MaxOnMs`: red below the Hold (*short activations*) and above `MaxOnMs` (*max on exceeded*) |
| Fixed length, interval | the time from the previous Main onset, dashed at the longer of `LockoutMs` and `ActivationMs`: red where an Activation came sooner (*lockout broken*); beside it the lit time, dashed at `ActivationMs` (*wrong activation length*) |
| Fixed length, ratio | the Requirement each Main Activation met against its number K, over the dashed line of the series: red points it does not give (*requirement series wrong*); beside it the lit time, dashed at `ActivationMs` |
| Non-feeding | each Activation's relight delay after the fly's last activity sample, dashed at `OnDelayMs`, with no red (it is reported, not judged); with `MaxOnMs`, beside it the lit time, dashed there, red above it (*max on exceeded*) |
| Open loop | none: the Program timeline shows the schedule |

A Paradigm with Acclimation has a closed-loop figure for its Acclimation, dashed at
`ActivationMs`, the Hold there. A subsection whose Activations could all not be judged
(merged with other light, or cut short by an Interval change, a gap in `Index` or the end
of the data) shows a grey note saying so instead of a figure.

How to read them: on a working system with the right program every bar sits on its dashed
line or on the side the rule allows, and nothing is red. Red is first a question about
`Program.txt`: is it the file exported with this recording? If it is, the LED or its `OnMs`
report is faulty. See
[Each Paradigm's own rule](concepts-optogenetics.md#each-paradigms-own-rule).

## Summary plots

### Feeding summary

One panel per metric, with the treatments along the x axis — or the factor combinations
when [factors](config-factors.md) are declared. Each point is a chamber; the black diamond
and bars are the treatment mean ± SEM. Written to `analysis/feeding_summary.png` by the
Plots panel's **Feeding summary** and by basic analysis, and drawn again as the experiment
report's Appendix A. A metric no chamber has a value for, such as a bout duration when no
feeding event was detected, is left out.

Which metrics are panelled depends on the experiment type:

- **Single-well:** `Licks`, `Events`, `MeanDuration`, `MedDuration`, `MeanTimeBtw`,
  `MedTimeBtw`, `MeanInt`, `MedianInt`
- **Two-well** (Hedonic, Progressive Ratio and any two-well type): `PI`, `EventPI`, and
  A/B pairs of `Licks`, `Events`, `MeanDuration`, `MedDuration`, `MeanTimeBtw`,
  `MedTimeBtw`, `MeanInt`, `MedianInt`

Panel titles use your `well_names` when you have set them, so a panel reads
`Licks (Sucrose)` rather than `Licks (A)`.

Because each point is a chamber, this is the plot that shows you your *spread*. A treatment
difference resting on two outlying chambers is visible here and invisible in a bar chart.

### Binned metric by treatment

A time course: per-treatment mean with an SEM band, across time bins. Use it to ask when
feeding happened rather than how much.

Any feeding-summary column works as the metric, plus base names — `Licks`, `Events`,
`MeanDuration`, `MedDuration`, `MeanTimeBtw`, `MedTimeBtw`, `MeanInt`, `MedianInt` — which
resolve automatically for the chamber size.

For two-well experiments, the combining mode matters and the sensible choice differs by
metric: **`total`** (A+B) for counts like `Licks` and `Events`; **`mean_ab`** for durations
and intervals, since summing two durations is meaningless. `A` and `B` isolate one well.

Bin width is yours to choose. Narrow bins show structure but are noisy; wide bins are
smooth but can hide a short-lived effect entirely. Try more than one before concluding
there is nothing there.

On the Hub this is the Plots panel's **Binned time course**, in the *Chosen metric* group:
the group's **Metric** menu picks the metric, and the Analyze panel's **Bin size** the bin.
The group's **Dot plot** draws the same metric, one point per chamber.

### Sliding-window plots

Three actions share this shape, each summarising a window of data and advancing by a step —
both in minutes, defaulting to 60 and 30.

| Plot | Shows |
|---|---|
| `plot_moving_window` | Treatment mean ± SEM of any metric |
| `plot_moving_median_chambers` | Median bout duration, one line per chamber, faceted by treatment |
| `plot_moving_median_treatment` | Treatment mean ± SEM of median bout duration |

Where binned plots cut time into non-overlapping slices, these slide a window across it, so
a step smaller than the window gives overlapping data and a smoother curve. Only windows
that end by the end of the recording are drawn, so the last point never stands for half a
window. That smoothing
is cosmetic: adjacent points share data and are not independent observations, so do not
treat them as such in a test.

`..._chambers` is the diagnostic version — it shows whether an effect is consistent across
flies or driven by a few. `..._treatment` is the version for a figure.

They are script actions only; no Hub panel offers them. They save
`analysis/moving_window_<metric>.png`, `moving_median_duration_chambers.png` and
`moving_median_duration_treatments.png`.

## Choice-specific plots

**Well A vs B** — median bout duration at well A beside well B, one panel per treatment: a
direct visual comparison of the two options. The Plots panel's *Well A vs B* saves it as
`analysis/well_comparison_MedDuration.png`; the `plot_well_comparison` action draws another
metric.

**Cumulative preference index**, per DFM, from Python only
(`exp.get_dfm(1).plot_cumulative_pi()`) — how a preference developed over the session. A PI
that starts near zero and separates gradually is a different phenomenon from one that is
established from the first minutes, and only a cumulative view distinguishes them. The
experiment report's *Preference over time* is the binned version: the treatment mean PI per
bin.

## Hedonic plots

**Hedonic feeding plot** — Well A against Well B median durations, faceted by treatment.
Duration is the measure of interest in these designs; see
[Experiment types](concepts-experiment-types.md). It is the Plots panel's *Hedonic* group,
offered under a type with the Hedonic module, and saves
`analysis/hedonic_feeding_plot.png`; the experiment report shows it under *Hedonic feeding*.

## Light behaviour plots

Drawn for an optogenetic experiment of any type, from its
[light table](concepts-optogenetics.md#the-light-table): what the light did, one point per
chamber.

**Light behaviour dot plots** — in the experiment report's and the Project Report's *Light
behaviour* section, one subsection per [Paradigm Family](concepts-optogenetics.md#paradigm-families)
the program runs (fixed length by its branch, and the Acclimation apart). Each has one dot
plot per headline metric of its family, by Treatment, per Facet when the experiment has
them, two to a row, from the chambers whose window ran that family:

- **open loop**: Lit time (s), Light feeding index;
- **closed loop**: Activations, Lit time (s), Median Activation (s), and Cut-offs with a
  ClosedLoop `MaxOnMs`;
- **fixed length, interval**: Activations, Reward efficiency, Median return after the
  lockout (s);
- **fixed length, ratio**: Activations, Last Requirement met, Firmware Events;
- **non-feeding**: Fraction of the time lit, Suppressions;
- **Acclimation**: Acclimation end (min).

The diamond and bars are the treatment mean ± SEM, and the treatment comparisons sit
underneath. A metric no chamber of the subsection has is left out and named. A metric with
the same value in every chamber, such as the lit time of open-loop chambers lit for the whole
program, is named once as *the same in every chamber, so not compared between treatments*.
The Activation measures compare only the chambers that earn light (Paired, Self, Mutual): a
Yoked chamber keeps its lit time beside its partner's. A Progressive Ratio experiment shows
its paired chambers only. A member without a firmware 4.0.0 recording keeps one section
with the four metrics of before (Activations, Lit time, Firmware Events, Acclimation end),
everything but lit time empty.

**Feeding in light and dark** — in the open-loop subsection, under a type with the Light
and dark module (Custom and Progressive Ratio have it; see
[Analysis Modules](concepts-experiment-types.md#analysis-modules)): each open-loop chamber's
licks per minute in its own light and in the dark, as two points joined by a line, a panel per
treatment (and per Facet), with each phase's treatment mean as a black diamond. Lines that
fall from light to dark are a fly that fed more in the light. Only rows under a Program
Interval count, so the baseline lead-in is not taken as dark, and a phase shorter than a
minute has no rate. A chamber lit for the whole program therefore has no dark rate and no
light feeding index; when no chamber was both lit and dark, the figure is a grey note saying
so.

**Paired against yoked** — in the *Paired and yoked* subsection, for a design with a yoked
group under a type with the Paired and yoked module (Custom has it; Progressive Ratio keeps
its own paired − yoked figures): one point per yoked group, its paired chamber's value
against the mean of its yoked chambers', a panel per metric, over the dashed identity line;
above it, the paired fly's value is the larger. There is one figure per Facet. See
[Paired and yoked designs](concepts-optogenetics.md#paired-and-yoked-designs).

**Light Publication Figures** — in a Project whose Combined Analysis has a light table, the
[Plot Editor](app-plot-editor.md) offers every numeric column of it as a faceted figure, on
the shape of `faceted_licks`: x = Treatment, one panel per Facet, dots with mean ± SEM. Each
is listed as *… (light, faceted)* and drawn from `<project>_LightSummary_Facet.csv`, or
`<project>_LightSummary.csv` when the members have no Facets.

| Plot id | Metric |
|---|---|
| `faceted_activations` | Activations |
| `faceted_litsec` | Lit time (s) |
| `faceted_firmwareevents` | Firmware Events |
| `faceted_acclimationendmin` | Acclimation end (min) |
| `faceted_litfraction` | Fraction of the time lit |
| `faceted_activationsperhour` | Activations per hour |
| `faceted_lightonsets` | Light onsets |
| `faceted_acclimationactivations` | Acclimation Activations |
| `faceted_lastrequirementmet` | Last Requirement met |
| `faceted_progressatend` | Progress at the end |
| `faceted_eventsperactivation` | Firmware Events per Activation |
| `faceted_lockedoutcontactsec` | Feeding while locked out (s) |
| `faceted_unlitsec` | Unlit time (s) |
| `faceted_suppressions` | Suppressions |
| `faceted_mainactivations` | Main Activations |
| `faceted_medianactivationsec` | Median Activation (s) |
| `faceted_cutoffs` | Cut-offs |
| `faceted_maxactivations` | Most Activations possible |
| `faceted_rewardefficiency` | Reward efficiency (fraction) |
| `faceted_medianreturnsec` | Median return after the lockout (s) |
| `faceted_mediansuppressionsec` | Median suppression (s) |
| `faceted_licksperminlit` | Licks in the light (per min) |
| `faceted_lickspermindark` | Licks in the dark (per min) |
| `faceted_lightfeedingindex` | Light feeding index, on a fixed axis from −1 to 1 with a line at 0 |

The reports test the headline metrics of each family, above; the first four are the ones a
member without a firmware 4.0.0 recording keeps.

## Progressive-ratio plots

### Paired and yoked are never pooled

A Treatment in this type names *both* flies of a Chamber Group, so grouping by Treatment
alone would draw one cloud of points holding an effect together with its own control. The
yoked fly's PI is not a second measurement of the paired fly's preference — it is what the
paired fly is measured *against*, and averaging the two lands halfway to nothing.

So for a Progressive Ratio member the standard plots behave differently:

- The **dot plot** shows the **within-group difference**: one point is one Chamber Group's
  paired-minus-yoked value of the chosen metric, panelled by Facet (Training, then Test),
  with zero drawn as the null the points are read against. This is the unit the type is
  built on — a difference is always taken within a Chamber Group, never between group
  means. Any metric in the dropdown works, including the A/B combinations; the fixed set
  in `paired_yoked_diff.csv` is the same numbers for the metrics it stores.
- The **feeding summary**, the **binned time course** and the **sliding-window** plots
  split every group by role, so a treatment becomes two series — `w1118 · paired` and
  `w1118 · yoked` — rather than one.

The difference is taken over each group's own Training and Test Facets: the Facets are per
Chamber Group, which is also why a Progressive Ratio experiment has no
[Slices](concepts-slices.md#progressive-ratio-and-fixed-ratio).

Project-level faceted publication figures in the [Plot Editor](app-plot-editor.md) panel by
Role when the Combined Analysis carries it. Use the pooled `<project>_PairedYokedDiff.csv`
or the Project Report for the paired − yoked *difference* statement across members.

A design whose chamber groups are single chambers, with no yoked control, has no paired −
yoked difference to plot: its paired-minus-yoked figures are empty, titled with the reason,
and the reports say once that the design has no yoked controls. Its breaking point figures
are drawn as usual. See
[Without yoked controls](concepts-progressive-ratio.md#without-yoked-controls).

### The type's own figures

**Cumulative difference curve** (`plot_pr_cumulative_diff`, and `timecourse_pr_diff` in a
Project) — the headline figure: paired-minus-yoked cumulative sucrose-well licks against
minutes since the chamber group's training end, one mean ± SEM curve per treatment with the
individual group traces faint behind. The mean runs only as far as the shortest group, so
every group is in every averaged point and the curve never jumps when one group's recording
ends; the faint traces run to each group's own end.

**Breaking point vs Test time** (`scatter_pr_breaking_point` in a Project) — one point per
chamber group from the stacked `pr_breaking_point.csv`: Breaking Point against `BreakMin`
(minutes since training end of the last counted Test light event), coloured by treatment.
Open symbols are censored (still responding at the end of the Test window).

**Breaking point by treatment** (`dot_pr_breaking_point` in a Project) — the same rows with
Treatment on the x axis and Breaking Point on y (no Phase/Role facets). Open symbols are
censored; mean ± SEM overlays each treatment.

**Training-aligned traces** (`plot_pr_cumulative_licks`) — per DFM, one panel per chamber
group: paired and yoked cumulative sucrose-well licks since training end (the paired fly's
alone in a solo group), with the bins in which the group's light was on drawn as points and
lick-free light events as black rings on the paired trace. A QC figure, not a result. The
per-group figures put three panels to a row, so six solo groups take two rows.

**Still-responding curve** (`plot_pr_still_responding`, and the first figure of the Project
Report's breaking point section) — per treatment, the fraction of paired flies whose
[breaking point](concepts-progressive-ratio.md#breaking-point) reached each ratio. It is a
Kaplan-Meier curve: a fly still responding when its Test window ended is censored, a lower
bound, and is drawn as a tick where it leaves the curve rather than counted as having
stopped.

**Breaking-point plots** (`plot_breaking_point`) — per DFM, one panel per chamber:
`DeltaLicks` per light-on period against minutes since training end, the paired panel's
strip giving the group's breaking point (`n+` when censored). A dashed line marks the break,
the last light event the breaking point counts, on both chambers of the group; onsets past
it are grey, and lick-free light events, which never count, are hollow red rings.

All of these need the recording's light: `OptoCol1`, or on firmware 4.0.0 the LEDs'
`OnMs`.

### Did the paired fly earn its light? (QC)

The firmware lights a chamber group from its own reading of the paired chamber's Sucrose
Well *during* the run; pyflic counts licks *afterwards*, from the baselined signal. When a
Sucrose Well's resting level creeps up, the firmware reads it as continuous contact and
fires the light on its own schedule, while the baselined signal is flat and no licks are
recorded — so the light looks earned in every light-on number and nothing was earned. Two
figures show it, beside the training-aligned traces:

**Licks per light event** (`plot_pr_light_events`) — per DFM, one panel per chamber group:
the sucrose licks credited to each Test-phase light event, against the event's number. A
working progressive ratio climbs. Hollow red rings are **lick-free light events**, the
dashed line is the group's own trend, and the faint grey line is the requirement estimated
across the experiment (the median slope of the groups' first eight Test events). A long row
of red rings along zero is the light following the sensor, not the fly. Under a fixed ratio
the requirement does not climb, so the grey line and the "climbs" caption are left out for
those groups.

**Sucrose Well resting level** (`plot_pr_resting_level`) — per DFM, one panel per chamber
group: the paired chamber's Sucrose Well as its per-minute median raw signal over the whole
recording, against the median of the DFM's other Sucrose Wells (grey). Light onsets are the
rug along the bottom and the dashed line is training end. A well that creeps up while its
rug thickens is the cause behind the red rings.

Both are on the Hub's QC panel, in its *Progressive Ratio* group beside the **Light QC
table**, and save `analysis/pr_light_events_dfm<n>.png` and
`analysis/pr_resting_level_dfm<n>.png`.

Every panel's strip carries the group's light QC verdict. A group that failed — and so left
the analysis — stays in these three figures, its strip saying why; the result figures
(the cumulative difference curve, the dot plot) no longer contain it. See
[Light QC](concepts-progressive-ratio.md#light-qc) for the checks behind the verdict.

## Where plots are saved

**Generate plots** writes every figure ticked on the Hub's Plots panel to the member's
`analysis/` folder (the file names are in the [Plots panel](app-hub.md#plots-panel) table and
in [Script actions](scripts-actions.md#plot-actions)), and opens each as a tab in the output
area, replacing a tab of the same title. The Hub shows figures as static images, which paint fast and hold no live
canvas in memory. Publication figures are the [Plot Editor](app-plot-editor.md)'s, written
to the Project's `figures/`.

---

Related: [Summary metrics](concepts-metrics.md) · [Script actions](scripts-actions.md)
