# Projects and members

pyflic has three structural levels, each recognised by a marker file:

| Level | Marker | Its children are | Found by |
|---|---|---|---|
| **Batch** | none (structural) | Projects | a recursive walk that stops at each Project |
| **Project** | `project.yaml` | Members | its *immediate* subdirectories |
| **Experiment Directory** | `flic_config.yaml` | — | — |

```
my_study/                     ← a Batch
  drug_trial/                 ← a Project
    project.yaml
    experiment_type.yaml      ← the copy of a lab type, when the design names one
    remove_chambers.csv       ← an exclusion sheet, optional
    analysis/                 ← the Combined Analysis
    figures/                  ← publication figures
    plot_specs.yaml
    drug_trial_report.pdf
    rep1/                     ← a Member
      flic_config.yaml
      data/                   ← DFM CSVs
      analysis/
      qc/
    rep2/
```

Projects may sit at any depth *below the batch folder*, because grouping folders like
`Sept2026/` are transparent to the scan — but a Project's own members are its immediate
children and nothing deeper. See
[Running many projects at once](scripts-batch.md).

An **Experiment Directory** is one FLIC recording. A **Member** is an Experiment
Directory inside a Project. If you are working on a single recording with no siblings, you
never need a Project at all — open the Experiment Directory directly.

## Members are not replicates

A Project's members are **different experiments that address one question** — a dose
series, a genotype panel, a pilot beside its follow-up — not repeats of one another. That
is the difference from the sister application PyTrackingAnalysis, whose projects hold
replicates.

It matters in two places. The Design still forces one detection rule across every member,
because comparing them otherwise would be meaningless. But the Combined Analysis keeps
`Experiment` as a column and the mixed model nests DFM *within* Experiment, because the
rows are not interchangeable: DFM 1 in one member is a different physical device from DFM 1
in another, and the members themselves are different experiments.

## The design is an authority

The `design:` section of `project.yaml` owns **every** key under a member's `global:` —
the experiment type and chamber layout, the detection parameters, `well_names`,
`transform_licks`, the `constants:` cutoffs, the design factors, the Facets
(`facet_cutoffs`, `facet_labels`, `facet_source`), `optogenetics`, the exclusion groups
(`exclusion_group`, `also_exclude`), the [Slices](concepts-slices.md) (`slices`,
`slice_tolerance_min`), the `control_level`, and the Selection (`omit:`) that the QC,
Analyze and Plots panels'
ticks write (see [The ticks are configuration](app-hub.md#the-ticks-are-configuration)).

A member that contradicts any of them, or states a `global:` key the design does not own,
**fails to load**. This is stricter than it might
seem necessary, and deliberately so: if `well_names` says well A is S5 in one member and
S5Y5 in another, a pooled licks figure is not noisy, it is *wrong*, and nothing in the
output would show it.

```yaml
# project.yaml
name: drug_trial
design:
  global:
    experiment_type: Hedonic
    transform_licks: false
    facet_cutoffs: [60]
    params:
      feeding_threshold: 20
      samples_per_second: 5
    well_names: {A: S5, B: S5Y5}
    experimental_design_factors:
      TreatmentNew: [Ctrl, Exp]
```

## Writing the design

**Project panel → Project design…** is the editor for `project.yaml` — the Project's name,
its notes, and the whole design. **Create project…** and **Initialize existing directory…** open
the same editor, so a Project states its design when it is created rather than acquiring
one later by accident.

Three ways in, because a folder can be in three states: it already has a `project.yaml`
(**Open Project**), it does not exist yet (**Create project…**), or it exists without one
(**Initialize existing directory…**). Each refuses the other two's case rather than guessing.

The button reads **Project design… (none set)** for a Project with no `design:` block. Such
a Project still loads, but it validates its members against *each other* instead of against
an authority: whichever member happens to be first becomes the standard, silently, and
editing that member moves the standard. Writing a design fixes the rule in one place.

Opening the editor on a folder that is not a Project yet reads the design off the first
member it finds, which is the migration path — a folder of standalone experiments becomes a
Project without retyping what they already agree on.

Saving writes `project.yaml`. When the design's type is one of your lab's, saving also
writes its copy, `experiment_type.yaml`, beside it, and a new Project starts its
`plot_specs.yaml` with the type's own Plot Specs; choosing a shipped type removes the copy
(see [Where types live](concepts-experiment-types.md#where-types-live)).

### The Project design dialog

One dialog, titled *New Project*, *Edit Project design* or *Initialize existing directory*
by what it is doing, with four cards:

- **Project** — the **Directory** (its **Browse…** can make a new folder; an existing
  Project's design fills the form once you finish typing the path or pick it), the **Project
  name** (the folder's name by default, and fixed when initializing in place) and
  **Notes**, shown near the top of the Project Report.
- **Design** — what every member inherits: the **Experiment type**, with **New type…** to
  build one in the [Experiment Type Editor](app-type-editor.md) and, when the Project's copy
  of a lab type differs from the library's, **Update the copy from the library**; the
  **Chamber layout** (fixed by a type that owns it); **Well A** and **Well B**; **Transform
  licks**; **Optogenetics** (auto, yes or no); the **Facet source**, **Facet cutoffs** and
  **Phase names** ([Facets](concepts-facets.md)); the **Exclusion group** every member's
  `remove_chambers.csv` is read under, and **Also exclude**, the groups the whole recording
  also leaves out ([Exclusions](concepts-slices.md#exclusions)); the **Control level**, for
  a type with a [Treatment control](concepts-experiment-types.md#control-comparisons); and
  the **Design factors**.
- **Slices**: the [Slices](concepts-slices.md#slices-in-the-design) every member runs,
  one row each. A row gives the name, the start and end minute (`end`, or blank, for the end
  of the recording), the Slice's own exclusion group (its name unless you change it), the groups it also
  excludes, and the auto-removal cutoffs it overrides (blank keeps the design's). Below the
  rows is the **Coverage tolerance** ([Coverage](concepts-slices.md#coverage)). A Slice that
  cannot be used stops the save, and the dialog lists why. For a type with the Ratio
  analysis the card is greyed out with the reason, since such a type has no Slices.
- **Detection parameters** — the [detection parameters](reference-parameters.md), one rule
  for every member; the **Auto-filter thresholds**, each with a **Use** box (unticked turns
  that filter off; ticked and blank takes the type's default); for a type with the
  Progressive Ratio module, the **Progressive ratio light QC** and **Progressive ratio
  breaking point** groups ([Light QC](concepts-progressive-ratio.md#light-qc),
  [Breaking point](concepts-progressive-ratio.md#breaking-point)); and, unless
  Optogenetics is *no*, the **Optogenetic light QC** thresholds
  ([Thresholds](concepts-optogenetics.md#thresholds)).

Pointed at an Experiment Directory, *Initialize existing directory* offers to initialize
its parent instead, so the experiment becomes one of its members.

Initializing a folder of experiments seeds the design from the first member, with each
factor's levels widened to every level any member declares — the members of a Project are
different experiments, so the first one's levels are rarely all of them.

When you save a design, the members whose config does not carry its **description** are
listed, and you are offered to **write it into them** (below). Any analysis setting a
member states is removed, since the design owns it; its `dfms:`, `about:`, `scripts:` and
`data_folder:` are left untouched. A member whose own description **conflicts** with the
design — other well names, another chamber layout, a level the design does not declare —
is never rewritten: it is named with the reason, and the Project will not load until you
correct the design or the member.

## Members carry the design's description, and inherit the rest

The design owns all of `global:`, but each member's `flic_config.yaml` carries a copy of
the part that says what the recording *is* — `experiment_type`, `chamber_layout`,
`well_names`, `experimental_design_factors` and `control_level` — so the config can be
read on its own. Every analysis setting is inherited:

```yaml
# rep1/flic_config.yaml
global:                         # the design's description, copied
  experiment_type: Hedonic
  well_names: {A: Sucrose, B: Yeast}
  experimental_design_factors: {Genotype: [Ctrl, MBON10, MBON11]}
dfms:
  - id: 1
    params: {pi_direction: left}
    chambers: {1: Ctrl, 2: Ctrl, 3: MBON10}
```

A `global:` block that is present is validated key by key, so a copy that differs from the
design is an error, as any deviation is. A member with no `global:` at all still loads and
inherits everything; the Project warns that its config does not read on its own.

Writing the copy never changes what a member says its recording was. It may fill in keys
the member does not state, give a Custom member the design's type when the layout agrees,
and widen the member's declared levels to the design's — its chambers, which say which
flies had which level, are never touched.

That is why the **config editor** reads the parent's design when you open a member's
config. The global settings are filled in from the design and shown **read-only**, with a
banner naming the `project.yaml` they came from, and the per-DFM override checkboxes are
limited to the physical keys. Saving writes `dfms:` (and leaves `scripts:` and `about:`
alone) with only the design's description under `global:` — a member's config cannot be
turned into a design violation by editing it. A standalone experiment, governed by
nothing, keeps every field editable.

## What stays free

- the whole `dfms:` block — DFM count, chamber → treatment assignment, and a DFM's own
  keys such as `paired_chambers`, `chamber_groups` and `optogenetics` (whether that DFM's
  lid was lit is a fact about the recording)
- per-DFM `params:` overrides, but **only** the physical keys `pi_direction` and
  `chamber_sets`

`pi_direction` genuinely varies between DFMs of one recording — it says which side of the
chamber the reference well sits on. An override of an *analysis* key is rejected inside a
Project, because it would reintroduce the divergence the design outlaws, one level lower
and much harder to see.

## Blocked members

A folder inside a Project can hold a perfectly good recording and still be unusable. The
loader reads `experiment_dir/data` — or the folder the config's
[`data_folder`](config-structure.md#data-folder) names — and nothing else, so DFM CSVs
sitting at the folder's root are invisible — not broken, not warned about, simply absent
from the Project.

The Project panel lists these as **blocked members**, in red, with the reason in the
tooltip, and the Batch panel counts them for each Project:

| Reason | Fix |
|---|---|
| `unfiled recording` — DFM CSVs at the root, not in `data/` | **File unfiled recordings** |
| `no config` — `data/` holds DFM CSVs but there is no `flic_config.yaml` | **Member configs…** or **Initialize existing directory…** |
| `no recording` — a config, but no DFM CSV the loader can find (a CSV whose name is spelled in the wrong case is named), or a `data_folder` that does not exist or is a file | by hand |
| `ambiguous` — the same DFM id both loose and in `data/`, or a `data` that is a file, or a `data_folder` beside a recording of the member's own | by hand: decide which copy is the experiment |
| `duplicate` — reads the same recording as a member before it by name, so only that one is analysed | by hand: decide which member reads that recording |
| `unreadable` — the folder cannot be listed (permissions?) | by hand |

Double-click a blocked row for its repair in place. In the Batch panel, right-click a
Project for **Fix blocked members in …**, which opens the
[review window](scripts-batch.md#the-review-window) on it.

Blocked belongs to the **member**, never to the Project: a Project with four healthy
members and one blocked one analyses the four.

**Filing** moves `DFM*.csv` into `data/`, with the `Program.txt` and `Messages.csv`
PyMCU exported beside them, since pyflic reads both only from `data/`. Every other loose
file goes into `extra_files/`. Every `.yaml` file and any `remove_chambers.csv` stays
exactly where it is — at a member root those are configuration and declaration, never
data. Nothing is ever overwritten. A member whose config names a `data_folder` is never
filed: its recording is read where the key says, and nothing is moved into or out of it.

## A member whose recording is elsewhere

A member does not have to hold its recording. When the recording must stay where it was
recorded — a run folder that is locked once registered — the member's `flic_config.yaml`
names it with [`data_folder`](config-structure.md#data-folder), relative to the member:

```
managed/
  runs/flic/2026/R-20261020-asmith-01/
    data/                          ← the recording, only ever read
  studies/mbon10-silencing/analyses/A-20261105-asmith-01/
    project.yaml                   ← the Project
    R-20261020-asmith-01/          ← a member
      flic_config.yaml             ← data_folder: ../../../../../runs/flic/2026/R-20261020-asmith-01/data
      analysis/  qc/               ← everything pyflic writes for it
```

Such a member's config usually holds `data_folder:`, `dfms:` and the design's description
under `global:`, and inherits every analysis setting like any other. Everything pyflic writes for it — the analysis,
the QC, its report, its Slices, the cache — stays in the member's folder, and the pooled
outputs in the Project's. Two Projects may read one recording with different designs;
neither changes the other. Within one Project a recording is read once: a second member
naming the same folder is a `duplicate`.

A Project whose members read their recordings this way no longer carries them. Copy the
analysis folder on its own and those members are blocked, each naming the folder it
looked for; move the whole tree and the relative paths still resolve.

## Adding a member

The same three cases, one level down — **Create member…** when it does not exist yet,
**Initialize existing directory…** when the folder is there but its config is not, and
**Member configs…** for doing the second in bulk. Double-clicking a blocked row offers the
same repair in place.

Right after **Create member…** the Hub asks how to finish the config: **Edit config…**
opens the scaffold, or **Copy config from…** replaces it with a config from a member that
already works — the common case for the second and later members of a run. A copy is
**checked against the design before it is written**, using the same test every member
passes at load. That ordering is the point: a non-conforming member makes the whole Project
refuse to load, and you would otherwise have to find and undo the copy by hand. A copy that
conforms is still not necessarily *ready* — it carries the other member's chamber
assignments, which are about that plate. Two things are not copied, because they belong to
the other member's recording: its `data_folder:`, and from its
[`about:`](config-structure.md#about) the title, the purpose, the notes and the
catalogue's id (`run:`). What members share of `about:` is copied, as scaffolding copies
it (below).

**Initialize existing directory…** files what is loose before it configures anything: the DFM
CSVs into `data/`, everything else into `extra_files/`. A config written first would
describe a member whose data the loader cannot see.

Scaffolding copies the `dfms:` block from the first existing member — members of one
design almost always reuse the plate layout, and retyping forty-eight chamber assignments is
the work worth avoiding — then reconciles it against the DFM ids actually in `data/`:

- an id in the data with no entry is **added**, chambers unassigned
- an entry with no data is **flagged**, never silently dropped, because a missing CSV is
  usually a copy that has not finished

Scaffolding also copies what members share of the first member's
[`about:`](config-structure.md#about): the experimenter, the series, the organism, the sex,
the age at the start, the protocol, and what the design's factors and levels mean, with a
level's registry term when a catalogue has given it one. Members of one design share
those, so nobody types them twice. It never copies the title, the purpose, the notes or
the catalogue's id (`run:`): give the new member its own in the
[Experiment Editor](app-experiment-editor.md). The copy is made once; after that each
member's `about:` is its own.

An existing config is never overwritten.

If the folder's DFM CSVs are still loose at its root, file them first — scaffolding
reconciles against what is in `data/`, and doing it beforehand would write a config
reconciled against nothing.

### The Member configs dialog

**Member configs…** lists every folder of the Project that holds a recording or a config,
with whether it has a `flic_config.yaml`; a blocked one is red, with its reason in the
tooltip.

- **Create config** scaffolds the selected folder's config as described above, filing a
  loose recording first.
- **Create n missing config(s)** does the same for every folder that has none.
- **Edit config…** opens a configured folder's config in the Config Editor.

Double-clicking a row does whichever applies. Every note the scaffolding makes — a DFM
added with chambers unassigned, one kept and flagged — goes to the Hub's log, because each
is something to finish by hand. The Batch Run's review window hands a member whose reason
is `no config` to the same dialog.
