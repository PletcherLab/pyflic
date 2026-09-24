# pyflic

A toolkit for analysing FLIC (Fly Liquid-food Interaction Counter)
feeding-behaviour experiments: detects feeding and tasting bouts from raw DFM
signal, and pools a Project's member recordings into publication-ready figures
and statistics. Structurally modelled on PyTrackingAnalysis — Batch → Project →
Experiment, tile-strip Hub, two-level scripting, Experiment Types, publication
figures — with pooling at the Project level. This glossary fixes the domain
language; it is not a spec.

**The one structural difference from PyTrackingAnalysis.** There a Project holds
**replicates**: the same experiment repeated, and pooling is a bigger n. Here a
Project holds **Members**: different experiments that address one question — a
dose series, a genotype panel, a pilot beside its follow-up. Pooling still
happens, and the Design still forces one detection rule across all of them, but
"member" never means "another run of the same thing", and the Combined Analysis
keeps `Experiment` as a column precisely because the members are not
interchangeable. Every mirrored surface from the sibling app renames
*replicate* to *member*; a Project inside a **Batch** is deliberately NOT called
a member, because "a member with four members" is a sentence this codebase must
not be able to write.

## Language

**Project**:
A directory with a `project.yaml` at its root whose immediate subdirectories
holding a `flic_config.yaml` are its **Members**. The Project is the level
at which results are pooled, and it owns the pooled outputs. Modelled on the
PyTrackingAnalysis Project.
_Avoid_: study, collection, batch parent, parent directory.

**Experiment Directory**:
One FLIC recording's directory — `flic_config.yaml` at its root, a `data/`
folder of raw DFM CSVs, and the outputs pyflic writes — either standalone or
as a Member inside a Project.
_Avoid_: project, project directory (the pre-overhaul name), folder,
dataset directory.

**Member**:
An Experiment Directory inside a Project — one of the several *different*
experiments that Project brings to bear on its question, not a repeat of its
neighbours. Its `flic_config.yaml` normally holds only the free parts — `dfms:` and its own `scripts:` — and inherits
`global:` from the Design; a `global:` block that *is* present (a standalone
experiment moved in) is validated key-by-key and any deviation is a load
error. A new Member is **scaffolded** by copying the first Member's
`dfms:` block and reconciling it against the DFM ids actually present in
`data/` (parsed from `DFM<id>_<n>.csv`): ids with no entry are added with
chambers unassigned, entries with no data are flagged rather than dropped.
The Project panel lists experiment-shaped folders — a `data/` holding at
least one `DFM*.csv` — that have no config yet as scaffolding candidates.
Scaffolding never overwrites an existing config.
_Avoid_: replicate (the sibling app's word; its members repeat one experiment
and pyflic's do not), run, dataset.

**Blocked Member**:
A folder inside a Project that a run cannot use as it stands: an **Unfiled
Recording** (DFM CSVs at its root rather than in `data/`), a recording with no
`flic_config.yaml`, or a configured directory with no DFM CSV at all
(ADR-0009). Blocked is decided by layout alone — no YAML parsed, no data read —
using the loader's own test, so a directory the classifier calls healthy is one
the loader can actually open. Blocked is a property of the **Member**, never of
the Project: a Project with four healthy members and one blocked one runs the
four, and a run is never refused because of one. Blocked Members are named when
the Project is selected, again in the Batch Run preflight, and again in the run
summary.
_Avoid_: invalid member, broken member (nothing is broken — the run just cannot
use it yet), missing member.

**Unfiled Recording**:
The one Blocked state a button clears: DFM CSVs sitting at a member's root
because that is where the rig wrote them. **Filing** moves `DFM*.csv` into
`data/` and every other loose file into `extra_files/`; every `.yaml`/`.yml`
and the Exclusion Sheet stay at the root, because there they are configuration
or declaration, never data. Filing never overwrites and never guesses: an
existing destination is skipped and reported, and the same DFM id present both
loose and filed refuses to file at all rather than deciding which copy the
analysis is of.
_Avoid_: import, ingest, tidy up.

**Design**:
The `design:` section of `project.yaml` — the **authority** for every key
under a Member's `global:` (experiment type, detection `params:`,
`well_names`, `transform_licks`, `constants:`, `experimental_design_factors`).
A Member that deviates on any of them fails to load, not merely warns.
Left free per Member: the whole `dfms:` block — DFM count and chamber →
treatment assignment — and per-DFM `params:` overrides
restricted to the *physical* keys (`pi_direction`, `chamber_sets`). A
per-DFM override of an analysis key is rejected inside a Project; a
standalone Experiment Directory keeps unrestricted overrides.
Authored in the **Project design** editor (Hub → Project panel), which is also
what "Create Project…" and "Initialize this folder…" open, so a Project states
its Design at creation
rather than acquiring one by accident; a Project with no `design:` falls back
to validating Members against each other, where the first Member silently
becomes the standard. The Design is **reinforced downwards**: a Member's config
opens in the config editor with the Design's values shown read-only and its
`global:` omitted on save, per-DFM overrides limited to the physical keys, and
saving a Design offers to delete any `global:` block a Member still carries.
_Avoid_: defaults, template (a Design is enforced, not inherited-and-
overridable).

**Experiment Type**:
A named bundle that selects one Chamber Layout and constrains the rest of an
experiment — the required `well_names`, the facet cutoffs and phase labels,
the default `constants:`, the set of analyses that run, the plot set, and the
report produced. It is the top-level thing a scientist chooses; everything
else is derived or constrained from it. A composed strategy object, not an
`Experiment` subclass.
_Avoid_: assay, assay type, protocol, template.

**Custom Experiment**:
The absence of a chosen Experiment Type — the permissive freeform mode, where
the config is driven directly by `chamber_layout:` with no constraints. A
config with no `experiment_type` key IS a Custom Experiment. Selectable
explicitly too.
_Avoid_: generic, freeform, none.

**Chamber Layout**:
How many wells a chamber has — `single_well` or `two_well` — and therefore
which `Experiment` subclass computes the metrics (a type carrying analysis of
its own may name a further subclass via `experiment_class`). pyflic's analogue of
PyTrackingAnalysis's Tracking Type: a lower-level, implementation-facing
concept that an Experiment Type selects exactly one of. A typed config writes
neither `chamber_layout:` nor `params.chamber_size` — the type owns both and
they are derived, never written to disk; only a Custom Experiment states
`chamber_layout:`.
_Avoid_: tracking type (nothing is tracked in FLIC), chamber size (that is
the derived numeric parameter), experiment type (the layer above).

**DFM**:
One Drosophila Feeding Monitor board — the instrument that samples twelve
wells (`W1`–`W12`) and writes them as `DFM<id>_<n>.csv`. A recording uses one
or more; each carries an integer **id** that is the identity used everywhere
(filenames, `dfms:` keys, per-DFM overrides), not its position in any list.
_Avoid_: device, board, monitor, plate.

**Chamber**:
The unit an animal occupies, and the unit every metric is computed for: one
well under `single_well`, a pair of wells under `two_well`. Chamber count is
derived from the Chamber Layout — 12 or 6 per DFM — never stated directly.
_Avoid_: well (a well is the electrode; a two-well chamber has two of them),
arena, position.

**Experimental Design Factor**:
A named independent variable declared once for the whole experiment under
`global.experimental_design_factors:` with its permitted **levels**. Factors
are **positional**: a chamber assignment is one level per factor in
declaration order, every factor must get a level, and reordering the
declaration invalidates every assignment in the file.
_Avoid_: variable, condition, group, category.

**Treatment**:
What a Chamber is assigned to, and the pooling key — chambers sharing one
across DFMs are analysed together. With no Factors declared it is a free
name; with Factors declared it is the ordered tuple of their levels, written
comma-separated. A blank assignment omits the Chamber from the config
entirely; a *partial* one is an error, never a shorter tuple.
_Avoid_: group, condition, label, cohort.

**Facet**:
A named time window within a recording, normally fixed by the Design's
`facet_cutoffs:` so every Member is windowed identically. The one exception
is the Progressive Ratio type, whose two Facets — **Training** and **Test** —
are split at each Chamber Group's own data-derived training end; the type
owns them, so a PR config never states `facet_cutoffs`. A Facet is still one
named window per row, so every consumer of the Facet column works unchanged. Facets are a
**column**, not a directory: analysis writes `feeding_summary.csv` plus
`feeding_summary_facet.csv` carrying a Facet column. The Hub's Start/End
controls select a Facet rather than filtering the load. Replaces the
retired `analysis_<start>_<end>/` output directories.
_Avoid_: window, range, time bin (a bin is the separate `binsize` concept
used by binned summaries).

**Excluded Chamber**:
A chamber removed from every result. Two live sources, both applied before
pooling: a row in that Member's `remove_chambers.csv` under the active
exclusion group, and `auto_remove_chambers()` driven by the Design's
`constants:` cutoffs. (A third, `excluded_chambers:` in a `dfms:` entry, was
already deprecated before this overhaul — the loader warns and ignores it.)
The file stays per-Member — which chambers are bad is a fact about that
recording — but the Design names the active `exclusion_group:`, so every
Member is filtered by the same rule. Authoring in bulk is the **Exclusion
Sheet**'s job, one level up. A Member whose declaration is newer than its saved
`feeding_summary.csv` reads **"re-run needed"** rather than "analyzed": those
results describe a chamber population nobody asked for, and a date beside them
would be true about a number that is not. The Combined Analysis stacks only
filtered rows and writes an aggregated exclusions table (Experiment, DFM,
Chamber, Source, Note) into the Project Report.
_Avoid_: dropped chamber, filtered chamber, QC filter (data quality is a
separate, always-reported concern).

**Combined Analysis**:
The Project-level results built by stacking each Member's *filtered*
feeding summaries (auto-removals already applied) with an added `Experiment`
column. Because DFM ids repeat across Members, DFM is only ever
interpreted **within** an Experiment. Statistics run twice: per-chamber
pooled tests matching the plots, beside a linear mixed model with treatment
fixed and **DFM nested within Experiment** as the random structure.
_Avoid_: merged analysis, meta-analysis.

**Project Report**:
The Project-level PDF: pooled figures rendered from the Combined Analysis,
the pooled + mixed statistics tables, and a per-Member summary table
(chamber counts, exclusions). Members never get their own figure sets in
it — that is what a Member's own report is for.

**Publication Figure**:
A hand-curated, journal-ready vector figure (SVG with editable text, or PDF)
rendered by plotnine from a Plot Spec + Plot Style and saved under
`<project>/figures/` — distinct from the matplotlib figures embedded in the
PDF report, and always regenerable from the spec.
_Avoid_: report figure, plot export.

**Plot Style**:
A named, reusable look shared by every Publication Figure that references it:
figure size, theme, fonts, point/mean styling, and the treatment→color
mapping. Stored in the Project root's `plot_specs.yaml` under `styles:`;
`default_style:` names the one the Plot Editor auto-loads. One Style covers
both figure families, so a Project has one look.
_Avoid_: theme (a plotnine theme is one field inside a style).

**Plot Spec**:
One Publication Figure's content decisions plus the name of its Plot Style,
stored in `plot_specs.yaml` under `plots:`, keyed by plot id. Two shapes:
a **faceted metric** spec (`faceted_licks`, `faceted_events`,
`faceted_medduration`, `faceted_pi` — x = Treatment, faceted by Facet, dots +
mean±SEM; carries axis labels, facet/treatment inclusion, order and display
names, y-limits, reference line) and a **time-course** spec
(`timecourse_<metric>` — x = time bin, one line per treatment with an SEM
ribbon; carries binsize and mode instead of facet ordering).
_Avoid_: plot config, settings.

**Plot Editor**:
The Project-level app that opens a Project, renders a live preview of the
pooled figures from the same Spec+Style that saving uses, and writes the
vector Publication Figures. Presentation only — it never alters a
`flic_config.yaml`.

**Analysis Hub**:
The main app: a horizontal **tile strip** across the top (Batch · Project ·
Analyze · Plots · Scripts · AI · Tools — each tile shows only live status,
with a **status readout** filling the strip to their right), and a full-width
output/plots area below. All controls live in a tile's **anchored panel**
(one open at a time). Tiles never move or hide — an inapplicable tile dims
and its panel holds the fix. The selection names the working container — a
Batch or a Project. The Hub is **Project-first**: an experiment is loaded
only by double-clicking its row in the Project panel's members table, so
there is no Load tile; the parallel/executor/max_workers options live in the
Project panel. Double-clicking a Batch row opens the Project panel and
double-clicking a member opens the Analyze panel — selecting is only ever a
step toward doing something. The Batch and Project tiles are **never dimmed**,
because their panels hold the controls that fix the empty state; a dimmed
tile's panel dims its cards too, and every card stays clickable. Every other
tile follows its subject: **Scripts** dims with Analyze and Plots, because it
is the *member* level — Project Scripts and the Design editor are in the
**Project** panel, with the Project they act on.
The Project panel offers **three ways in and no more** — the folder is a
Project / does not exist / exists without a `project.yaml` — plus the editor
for the one that is open, and the **Members** row repeats the trio one level
down (Create member / Initialize existing folder / Member configs). The cases
are disjoint by construction: each button refuses the others' case and names
the one that handles it, so "which button is mine" is never a guess. Mirrors
PyTrackingAnalysis's Create/Load and Experiments cards.
_Avoid_: card column (the pre-overhaul layout), Load card.

**Experiment Script**:
A saved, re-runnable step list of experiment-level actions. Lives in an
Experiment Directory's `flic_config.yaml` `scripts:` — or, for Members,
centrally in the Project's `experiment_scripts:`, where one recipe serves
every Member without being copied. Central scripts run only through the
`run_in_experiments` bridge.
_Avoid_: recipe, macro, pipeline, job.

**Project Script**:
A saved step list of project-level actions in `project.yaml` `scripts:`.
Same shape and visual editor as an Experiment Script, but a **separate action
registry** — levels cannot mix; the only bridge is `run_in_experiments`,
which runs a named Experiment Script in every Member. There is no third
(Batch) script level: what a Batch Run executes IS a Project Script, named by
`batch.yaml`'s `script:` key.

**AI Summary**:
An optional, AI-written narrative of an analysis, generated from the report's
own content by a user-chosen provider, opt-in per report and offered only
when a provider API key is configured in `.env`. It *summarizes* the
pipeline's analysis; it never performs its own. A derivative of a single run:
re-running the analysis deletes it.
_Avoid_: AI analysis, AI interpretation.

**Batch**:
A directory with at least one Project **anywhere beneath it** (ADR-0009).
Discovery is recursive and **prunes at each Project** — a Project's
subdirectories are its Members by definition, so the walk never looks inside
one — which makes grouping folders (`Sept2026/`, `Archive/2025/`) transparent
and means no Member can be analyzed twice in one run. Purely a processing
convenience for running many Projects unattended: it is not itself a Project,
holds no analysis of its own, and never pools across Projects. A **Batch Run**
executes one designated Project Script in every checked Project,
continue-on-error, after a **Preflight** states what will run. Nothing marks a
Batch — being one is structural; an optional `batch.yaml` appears only to name
the designated `script:` or hold central `project_scripts:`, and only the
selected Batch's file governs. A Project inside a Batch is keyed by its POSIX
path relative to the Batch root (`Sept2026/ProjA`), so a top-level Project keeps
its bare name and every designation and sheet row written before recursion still
resolves.
_Avoid_: batch target, subdir-batch mode, yaml-batch mode (all retired, see
ADR-0006), study, collection, batch root; **member** for a Project inside one.

**Preflight**:
The modal a Batch Run always opens first: the discovered Projects with their
relative-path keys, their usable-member counts, and every Blocked Member with
its reason and the action that clears it; a preview of the Exclusion Sheet with
one switch to decline it for this run; then Run or Cancel. Shown even when
nothing is wrong, because with recursive discovery the target list is the one
thing no other surface states. Not a gate: it repairs and confirms, it never
refuses.
_Avoid_: confirmation dialog, wizard.

**Exclusion Sheet**:
A `remove_chambers.csv` (or `.xlsx`) at a **Batch root or a Project root** —
one level above the per-Member files — whose rows name a project, member, DFM,
chamber, group and reason. Applying it writes those rows down into each
Member's own `remove_chambers.csv`. It is a **writer, never an overlay**:
nothing reads the sheet at analysis time, so a sheet that is deleted or never
applied changes no result. The standing declaration always wins — a chamber
already declared is never rewritten and a differing reason is reported as a
**conflict** — because a Batch Run re-applies the sheet every time. Selecting a
Batch *reports* its sheet; only a Batch Run or an explicit button applies one,
and then only to the Projects actually running.
_Avoid_: exclusion overlay, removal config, the CSV (ambiguous with the
per-Member file it writes into).

### Progressive Ratio

**Chamber Group**:
In a Progressive Ratio experiment, a fixed pair of adjacent two-well
Chambers on one DFM — 1+2, 3+4, 5+6 — that share one light circuit and one
Treatment. The unit of pairing and of paired-vs-yoked comparison: a difference
is always taken *within* a Chamber Group, never between group means. Three per
DFM, derived from the Chamber Layout; never stated in the config.
_Avoid_: pair (ambiguous with the paired fly), block, quad, well group (the
group is four wells, but it is counted in chambers).

**Paired Chamber** / **Yoked Chamber**:
The two roles inside a Chamber Group. The **Paired** fly receives light-driven
neuronal stimulation contingent on its own feeding at the sucrose well, and is
the only chamber that undergoes and completes **Training**. The **Yoked** fly
is lit at the same moments as its Paired partner, independent of its own
behaviour. The role is structural, not a Treatment level: the config names the
Paired chamber per DFM (`paired_chambers: [1, 4, 5]`, exactly one from each
Chamber Group) and Yoked is always the other member, never written. Both
chambers of a Chamber Group must carry the same Treatment.
_Avoid_: test fly / control fly, stimulated / unstimulated (the yoked fly is
stimulated too), master / slave, active / passive.

**Sucrose Well**:
In a Progressive Ratio experiment, the well whose feeding triggers the light
for the Paired fly — always **well A**. The type requires `well_names` to name
A (sucrose) and B (yeast), and the existing per-DFM physical key
`pi_direction` says which side of that DFM well A sits on, exactly as in any
two-well experiment. No separate side key exists: one physical fact, one key.
_Avoid_: reward well, stimulated well (the yoked fly's well is lit too),
training well (it stays the sucrose well after training ends), left/right
well (that is the physical position, which varies per DFM).

**Training**:
The opening phase of a Progressive Ratio recording in which the Paired fly's
feeding at the Sucrose Well always turns the light on (pure closed loop). The
firmware marks it **per well**: a raw sample above 40000 means that well is
still in training, and its true value is the raw value minus 65536. A well's
**training end** is the last minute its own column is flagged; every well
reads its own column, and the four wells of a Chamber Group are expected to
clear together, because the group is trained as a unit. The **group's**
training end is the Sucrose Well (well A) of its Paired Chamber; any other
well of the group that clears at a different minute, or never clears, is a
per-well QC warning, never a load error. Training end varies
between Chamber Groups (it is behaviour-contingent), so it is a data-derived
per-group time, never a fixed-minute Facet cutoff. The **Test** phase is
everything after it, and Progressive Ratio time axes run from training end,
not from the start of the recording.
_Avoid_: acclimation, baseline period, phase 1, conditioning (the light is
conditioned on feeding, but "conditioning" is not the protocol's word).

**Paired-Yoked Difference**:
The within-Chamber-Group contrast — Paired minus Yoked — for a metric within
one Facet, written to `paired_yoked_diff.csv` with one row per Chamber Group
per Facet. Never a difference of group means: a group missing either chamber
(excluded, or unassigned) contributes no row. Pooled by the Combined Analysis
like any other summary, so statistics can be run on the difference directly.
Two questions are asked of it: whether it is non-zero within a treatment
(are paired and yoked flies different?) and whether treatments differ in it.
_Avoid_: delta (used for per-period lick differences in the breaking-point
table), effect, contrast (the model term, not the table).

**Cumulative Difference Curve**:
The Progressive Ratio headline figure: Paired-Yoked Difference of cumulative
Sucrose-Well licks against minutes since the group's training end, binned
(1 min by default), one mean ± SEM curve per Treatment with the individual
Chamber Group traces faint behind it. The mean is drawn only over the range
every group covers — it stops at the shortest group rather than jumping when
one group's recording ends — while the faint traces run to each group's own
end. The companion per-DFM **training-aligned trace** (one
panel per Chamber Group, Paired and Yoked as two lines, light-on samples
drawn as points) is a QC figure, not a result.
_Avoid_: breaking-point plot (that is the per-light-period ΔLicks table's
figure), PR timecourse (the generic `timecourse_*` family uses recording
time, not training-aligned time).

**Light Event**:
One onset of a Chamber Group's light — the moment the firmware judged the
Paired fly to have met its requirement (or, in Training, to have fed). The
firmware decides from its own reading of the Paired chamber's Sucrose Well
*during* the recording; pyflic counts licks *afterwards*, from the baselined
signal, so the two can disagree, and the **light QC** exists to measure that.
A Light Event is credited with the Sucrose Well licks from the end of the
previous Light Event to the end of its own — the licks that earned it. The
per-event table is the Paired chamber's breaking-point table, the **Light
Event Ledger** (`pr_light_events.csv`), whose `Counted` column marks the
events the **Breaking Point** holds.
_Avoid_: light period (that is the lit interval, not its onset), trigger
(which names the cause the light QC is testing, not the event).

**Lick-free Light Event**:
A Light Event credited with no Sucrose Well licks at all, and with no Sucrose
Well lick or touch within its light's decay — the **Opto Program**'s decay, or
`opto_default_decay_ms` without one — so a light the fly touched for is never
lick-free (ADR-0015). A few occur in healthy data (pyflic's feeding threshold
misses brief touches the firmware counts); a run of them does not.
_Avoid_: unearned light, false trigger (both presume the cause).

**Self-triggered light**:
The light QC's failure verdict when at least `pr_lick_free_run` (default 5)
consecutive Test-phase Light Events are lick-free: the light was following the
sensor, not the fly — typically a Sucrose Well whose Resting Level crept up
until the firmware read it as continuous contact. It fails the Chamber Group,
as does **implausible training** (Training completed with light events and no
Sucrose Well licks at all); by default (`exclude_failed_pr_groups`) both
chambers of a failed group leave the analysis through auto-removal. *No
increasing trend* in licks per Light Event and a rising or elevated Resting
Level are warnings only.
_Avoid_: clock-driven run, stuck light, runaway light.

**Resting Level**:
The raw, un-baselined signal a well sits at between licks, measured as the
per-minute median — a median over a minute ignores licks, which are brief.
Baseline subtraction removes it, so no lick count can show it; the Sucrose
Well Resting Level figure does, against the median of the DFM's other Sucrose
Wells (never its yeast wells, which drift by hundreds of counts over a day).
_Avoid_: baseline (that is the running median the baseline subtraction
removes, over a 3-minute window), offset, DC level.

**Breaking Point**:
The number of lick-backed Test Light Events a Chamber Group's Paired fly
completed before its first pause longer than `pr_break_gap_min` (Δt, default
120 minutes) — the ordinal of the last ratio it met. The pauses run from the
group's training end to the first event, between events, and from the last
event to the end of the **Test window**; a Lick-free Light Event neither
counts nor ends a pause, and a pause of exactly Δt does not end the count.
`BreakMin` is the minute, since training end, of the last event counted. A
property of the group, read from the Paired chamber: the Yoked fly has none,
because its light is its partner's. One row per group in
`pr_breaking_point.csv` (ADR-0014).
_Avoid_: breakpoint, final ratio, Test light events (that raw count is
`TestLightEvents` in the light QC, which ignores pauses and lick-free
events), yoked breaking point.

**Censored**:
Said of a Breaking Point or a Sucrose Persistence when the Test window ended
before any pause longer than Δt: the fly was still responding when the
recording stopped, so the value is a lower bound, not a measurement. Written
`n+` in text tables, drawn as an open symbol or a tick. The still-responding
curve and the log-rank test treat it as a lower bound; the t-tests and the
mixed model enter it as observed.
_Avoid_: truncated (the Test window cap truncates; censoring is what that can
cause), incomplete, missing.

**Test window**:
The span a Chamber Group's Breaking Point and Sucrose Persistence are judged
over: from the group's training end to the end of the recording, capped at
`pr_test_window_min` when that design constant is set (it is off by
default). It differs between groups because training end does, which is what
the cap is for.
_Avoid_: Test phase (the Facet, which is never capped), session.

**Sucrose Persistence**:
For either fly of a Chamber Group, the minutes since training end of its last
Sucrose Well feeding event before its first pause longer than Δt — the
Breaking Point's rule applied to feeding events (the ones `EventsA` counts)
over the same Test window, and so the one persistence measure the Yoked fly
has too. `PersistA` on every per-chamber summary row (none on a Training
row); `dPersistA`, Paired minus Yoked, in the Paired-Yoked Difference, with
`dPersistCensored` when either fly is censored.
_Avoid_: persistence (alone), latency, time to stop.

### Optogenetics

**Opto Program**:
The program the stand-alone MCU ran, read from the `Program.txt` it exports
and the experimenter copies into `data/` (ADR-0015): per DFM, a threshold per
well, the linkage, and per interval the frequency, pulse width, decay, delay
and max-time-on, laid out over the run by its program type. The export is a
record of what ran; the authored `[General]` / `[DFM]` file is the MCU's
input and is not read. Optional — without it the light QC runs with less
knowledge — and one per recording.
_Avoid_: protocol, schedule (the schedule is one part of it), light program.

**Linkage Group**:
The wells of one DFM that share a linkage number in the Opto Program, lit
together whenever any member is triggered — one light circuit, and so the
unit the optogenetic light QC judges, its verdict passed to every chamber it
touches. Without a program, the wells whose light is identical over the whole
recording. In a Progressive Ratio experiment a Linkage Group is a Chamber
Group's four wells.
_Avoid_: light group, circuit (informal), chamber group (a PR structure,
which a Linkage Group need not match).

**Trigger Well**:
A well whose threshold is positive in some interval of the Opto Program, so
its own signal can switch its Linkage Group's light on. A threshold of 0 keeps
the group lit regardless (open loop); -1 never triggers. In a Progressive
Ratio experiment the Trigger Well is the Paired chamber's Sucrose Well, which
is how `paired_chambers` can be read from the program.
_Avoid_: active well, stimulated well (the whole group is lit).

**Explained Light**:
A lit sample of a Linkage Group with activity — a feeding lick or a tasting
sample — in one of its Trigger Wells from the interval's decay plus
`opto_decay_tolerance_samples` before it to that tolerance after it. What
the optogenetic light QC measures is the share of lit time that is *not*
explained: a warning at `opto_unexplained_warn_fraction`, a failure at
`opto_unexplained_fail_fraction`. Judged only under a paradigm the fly's
feeding drives; open loop and non-feeding activation are not.
_Avoid_: earned light (the PR question, asked of Light Events), justified
light.

**Emulated Trigger**:
The firmware's own test re-run on the recorded raw signal: raw minus the mean
of the run's first ten seconds (its baseline, captured once) against the
interval's threshold. Contact the Emulated Trigger sees and pyflic does not —
a well the firmware read as touched while the baselined signal was flat — is
the drifting-baseline signature, and it names the likely cause of
unexplained light. Approximate when the data begin after the baseline window.
_Avoid_: simulated firmware, firmware replay.

### Cross-app

**MIRRORED.md**:
The cross-app change ledger shared byte-identically across pyflic,
PyTrackingAnalysis, and pySurvAnalysis: newest-first entries naming a change,
the app it originated in, its ADR, and a per-app implementation status. The
three apps duplicate their shared machinery (Hub shell, Script Editor,
Plot Editor, `ui/`, `help/`) rather than depending on a common package, and
this file is what keeps that duplication honest. **Its format and content are
owned by the sibling app that proposed it** — pyflic follows the convention,
it does not define it.

### Help

**Help topic**:
One addressable, self-contained piece of user-facing explanation — a concept, a screen, or a task. The unit a help button points at and the unit a reader navigates to.
_Avoid_: doc, page, article, section, help file.

**Help window**:
The single in-app window that displays help topics, with navigation across the whole topic set. There is one, shared by every help button in every pyflic app.
_Avoid_: help dialog, docs viewer, manual, popup.

**Help button**:
The small `[?]` control placed beside a screen region or setting that opens the help window at one specific help topic.
_Avoid_: info button, question mark, tooltip (a tooltip is the separate hover-only one-liner).

**Guide**:
An ordered reading path assembled from many help topics — front-to-back prose for someone learning a whole subject rather than answering one question. Always derived from topics; never authored in its own right.
_Avoid_: manual, documentation, the docs, USAGE.

## Relationships

- A **Batch** holds **Projects**; a **Project** holds **Members**; a
  **Member** is an **Experiment Directory**. Membership is by marker file
  (`project.yaml`, `flic_config.yaml`). A Project's Members are its *immediate*
  children; a Batch's Projects may sit at any depth, found by a walk that prunes
  at each Project (ADR-0009).
- **Blocked** is a property of a **Member**, never of its **Project**, and never
  refuses a run. Discovery, blocked status, and filing are Batch-level concerns
  but Member-level facts, so the Project panel and the **Preflight** read the
  same classification.
- A **Project**'s **Design** is the authority for every `global:` key in its
  Members; the `dfms:` block stays free, as do per-DFM overrides of the
  physical keys. A Member normally omits `global:` entirely and inherits.
- An **Experiment Type** selects exactly one **Chamber Layout**, fixes the
  **Facet** cutoffs, and declares the report set. Script actions are gated on
  both: `plot_well_comparison` needs a Chamber Layout, `plot_breaking_point`
  needs an Experiment Type.
- An **Experiment Directory** holds one or more **DFMs**; a **DFM** holds
  **Chambers** whose count the **Chamber Layout** derives (12 single-well, 6
  two-well); each **Chamber** carries at most one **Treatment**. A DFM is
  identified by its id, never by its ordinal.
- **Experimental Design Factors** are declared once for the experiment and turn
  every **Treatment** into an ordered tuple of levels. Declaring, removing, or
  reordering a Factor rewrites every Chamber assignment in the file — which is
  why the Config Editor regenerates them rather than leaving it to hand-editing.
- The **Combined Analysis** stacks Member summaries; the **Project Report**
  and the **Publication Figures** are both rendered from it — the report by
  matplotlib, the figures by plotnine from a **Plot Spec** + **Plot Style**.
- **Experiment Scripts** and **Project Scripts** have separate action
  registries and cannot mix; `run_in_experiments` is the only bridge. A
  **Batch Run** runs a Project Script — there is no Batch script level.
- In a **Progressive Ratio** experiment a **DFM** holds three **Chamber
  Groups**, each holding one **Paired Chamber** and one **Yoked Chamber** that
  share one **Treatment**, one light circuit and one **Training** end. The
  config names the Paired chamber per DFM; Yoked is derived. The **Sucrose
  Well** is always well A, and `pi_direction` places it, exactly as in any
  two-well experiment.
- The two Progressive Ratio **Facets**, Training and Test, are split at each
  Chamber Group's own training end (ADR-0013). The **Paired-Yoked Difference**
  table has one row per Chamber Group per Facet and is the primary input to the
  **Combined Analysis** statistics for this type; the per-chamber summaries,
  carrying Group and Role columns, are secondary.
- Every Progressive Ratio **Chamber Group** gets a light QC verdict from its
  **Light Events** and its Sucrose Well's **Resting Level**, over the whole
  recording whatever window a table uses. A failed group (**Self-triggered
  light**, implausible training) leaves through auto-removal like one that
  never finished Training, so every result table and the **Combined Analysis**
  stand on groups whose light followed the fly; the QC figures and
  `pr_light_qc.csv` keep it in view.
- A **Breaking Point** belongs to a **Chamber Group** and is read from its
  **Paired** chamber's **Light Events**; **Sucrose Persistence** belongs to
  each chamber. One rule judges both over the group's **Test window**, and
  either can be **Censored** (ADR-0014). Paired and yoked are compared on
  Sucrose Persistence and the other feeding metrics, never on the Breaking
  Point.
- An optogenetic **Experiment Directory** may carry one **Opto Program**,
  whose sections are per **DFM**. A DFM's **Linkage Groups** partition its
  wells; each group's **Trigger Wells** decide its light, and its lit time is
  judged as **Explained Light** or not, whatever the **Experiment Type**
  (ADR-0015). `optogenetics:` is a **Design** key; its per-DFM override lives
  in `dfms:`, which stays free inside a **Project**.
- In a **Progressive Ratio** experiment a **Chamber Group** is a **Linkage
  Group**, its **Paired** chamber's **Sucrose Well** is the **Trigger Well**,
  and two light QCs run: the type's, on **Light Events**, and the general one,
  on lit time. A group can pass one and fail the other.
- Many **Help buttons** across the apps open the one **Help window**; each
  names a single **Help topic**. A tooltip may summarise a topic but never
  restates it — the topic is the only copy of the text.
- A **Guide** is assembled from **Help topics** in a chosen order. No prose
  appears in a guide that is not in a topic.

## Example dialogue

> **Dev:** "Two Members both have a DFM 1. Do those chambers share a random
> effect in the pooled model?"
> **Domain expert:** "No. DFM ids are per-recording — DFM 1 in one Member
> is a different physical device from DFM 1 in another. DFM is nested within
> Experiment, never grouped across it."
>
> **Dev:** "A Member needs `feeding_threshold: 22` because its rig is
> noisier. Can it override the Design?"
> **Domain expert:** "No. The Design owns every `global:` key — that Member
> fails to load. Either the whole Project uses 22, or that recording isn't a
> Member of this Project. Only `pi_direction` and `chamber_sets` vary, and
> only per-DFM, because those describe hardware rather than analysis."

>
> **Dev:** "Chamber 2 on DFM 3 is yoked. Its wells read above 40000 for the
> whole recording — is that fly still in training at the end?"
> **Domain expert:** "No. Only the Paired fly *does* training; the group's
> training ended when the Paired chamber's Sucrose Well cleared. The yoked
> wells should have cleared at the same minute, and if they didn't that is a
> QC note about the firmware, not a fact about the fly. But the yoked fly
> lived through that same minute, so it carries the same TrainingMinutes —
> the value is the *group's* training end, not a claim about who trained.
> Only a group that never finished has none.""

## Flagged ambiguities

- "replicate" was the sibling app's word for a Project's children and was
  borrowed wholesale. Resolved: a pyflic Project's children are **Members** —
  different experiments addressing one question, not repeats — and the word
  *replicate* is retired everywhere except the Exclusion Sheet's accepted header
  spellings, where it stays for sheets already written. `Project.member_names`
  and friends keep `experiment_*` aliases so notebooks written earlier still
  run.
- "member" would also have been the natural word for a Project inside a Batch.
  Resolved: it is not used there. A Project is a **Project** at every level, and
  the Batch layer calls its entries `BatchProject`.
- "project" meant *one experiment* before this overhaul and now means the
  *pooling parent*. Resolved: one recording is an **Experiment Directory**;
  `Experiment.project_dir` is renamed. See ADR-0005.
- "batch" was ambiguous between subdir-batch and yaml-batch mode. Resolved:
  both are retired; **Batch** is a structural level. See ADR-0006.
- "experiment type" conflated the assay with the hardware — `hedonic` and
  `two_well` were both values of one key. Resolved: **Experiment Type** is the
  assay, **Chamber Layout** is the hardware, and a typed config states neither
  the layout nor `chamber_size`.
- The word `batch` still names things at two levels: the default **Project
  Script** written into every new `project.yaml` (PyTrackingAnalysis's
  convention), and the legacy Experiment Scripts named `batch` that
  subdir-batch used to run. The latter have no meaning after ADR-0006 —
  `pyflic lint` reports them; they are renamed by hand.
- The Config Editor presented **Chamber Size** and **Experiment Type** as two
  free controls and offered `two_well`/`single_well` as types — the pre-ADR-0007
  language, still live in the GUI long after the loader moved on. Resolved: the
  editor names **Chamber Layout**, the Experiment Type owns it, and `(auto)` is
  spelled **Custom Experiment** because that is what it is.
- "the docs" was ambiguous between `doc/` (user prose) and `docs/` (ADRs).
  Resolved: user-facing prose is now **help topics**, which live with the
  shipped code; `docs/` holds decision records for developers and nothing
  else. Long-form prose is a **guide**, assembled from topics rather than
  written. Say **help topic**, **guide**, or **ADR**, never "the docs".

- The Progressive Ratio training flag was described as clearing on all four
  wells of a Chamber Group at once, but the first real dataset
  (`test_data/progressive_ratio`) clears it on exactly one well per group —
  the Paired chamber's Sucrose Well — and leaves the other three flagged to the
  end of the recording. Resolved as a rule plus a tolerance: each well reads
  its own column and the four *should* agree; the group's training end is the
  Paired Sucrose Well's; any other well disagreeing or never clearing is a
  per-well QC warning, never a load error. Revisit if new firmware confirms
  which behaviour is intended.
- The R port's "configuration 1-4" conflated *which chamber of the group is
  Paired* with *which side the Sucrose Well is on*, and forced one position for
  all three groups of a DFM. Resolved: `paired_chambers:` names the Paired
  chamber per group and `pi_direction` names the side; the configuration
  integer is retired.
- "the light was on" was read as "the fly earned it". The first real dataset
  says otherwise: on DFM 1 of `test_data/progressive_ratio` one group's
  Sucrose Well drifted from 11 to 38 counts over the day and another's rested
  at 60-240 from the start, and in both the firmware fired the light hundreds
  of times with no lick pyflic could count — the second group even "completed"
  Training in 13 s with none. Resolved: light-on time is evidence about the
  sensor until the **light QC** says it is evidence about the fly; the rule
  that fails a group is a run of **Lick-free Light Events**, never a
  lick-per-pairing ratio, because pyflic's feeding threshold misses brief
  touches the firmware counts (a healthy group trained on 5 lick samples over
  8 pairings).
- "the breaking point" was described as "where `DeltaLicks` falls away",
  read by eye, and then computed as every Test Light Event to the end of the
  recording. Resolved (ADR-0014): the first pause longer than
  `pr_break_gap_min` ends the count, lick-free events are ignored, and a group
  still responding at the end of its Test window is **Censored**. The Yoked
  fly has no breaking point.

- "Lights on" was read as "the light answered the fly", with the firmware's
  own settings nowhere in pyflic. The first real dataset showed a group lit for
  four hours of training while pyflic saw no lick, which the Progressive Ratio
  light QC passes (one long event is not a run of lick-free events).
  Resolved (ADR-0015): pyflic reads the MCU's exported **Opto Program** when
  it is in `data/`, and every optogenetic experiment's light is judged by lit
  time, per **Linkage Group**, as **Explained Light** — with the
  **Emulated Trigger** to tell a drifting sensor from a light the firmware had
  no reason to switch on.

## Migration

There is no migration tool. The overhaul is a hard break: `pyflic lint`
reports each offending construct and the new form, and configs are fixed by
hand. Affected: `experiment_type: two_well|single_well` (→ Custom +
`chamber_layout:`), `params.chamber_size` in a typed config (→ delete),
multi-YAML directories (→ keep one `flic_config.yaml`), `<stem>_results/` and
`analysis_<start>_<end>/` outputs (→ orphaned, never deleted by pyflic),
Experiment Scripts named `batch`, and `.pyflic_cache/` entries (keys change
once Facets exist).
