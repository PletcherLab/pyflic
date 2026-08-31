---
status: accepted
---

# The Experiment group tile, the sub-strip, and the three-card Project panel

The Hub's flat seven-tile strip becomes a two-row ribbon mirroring
PyTrackingAnalysis (their ADR-0012): a top strip of wide container tiles —
**Batch · Project · Experiment** — plus a regular **Tools** chip and the
status readout, and a collapsible sub-strip of four compact, title-only
subtiles — **Analyze · Plots · Scripts · AI** — that the Experiment tile
expands.  The Project panel is split from one long card into three stacked
cards: **Create/Load**, **Experiments**, **Analysis**.

## Context

The flat strip put seven peers side by side, but four of them — Analyze,
Plots, Scripts, AI — are not peers of the containers.  They act on the loaded
member and are meaningless without one, and the strip's geometry said nothing
about that containment: a dimmed Analyze tile sat between two always-lit
container tiles, reading as broken rather than as "nothing loaded".

The Project panel had grown into one card holding everything: the three ways
in, the members table, member scaffolding, load options, project actions and
Project Scripts.  Nothing said which section answered which question.

PyTrackingAnalysis solved both, and pyflic mirrors its Hub deliberately
(MIRRORED.md): diverging here costs every future port.

## Decision

**The top strip is the containment hierarchy, left to right.**  A Batch holds
Projects, a Project holds members, and the loaded member holds the tools that
act on it.  The three container tiles are wide (`StatusTile(wide=True)`,
1.75×¾ of a regular tile); three levels get three colors — Batch tints
LOAD-blue, Project stays neutral, Experiment is QC-red, otherwise unused on
the ribbon.  Tools keeps a regular chip for now; upstream removed theirs, and
this one is expected to follow.

**The Experiment tile opens no panel: it expands the sub-strip.**  The four
subtiles are compact, title-only chips (38px), their status a hover away in
the tooltip.  Their panels are narrow button columns.  One thing is open at a
time: expanding the group closes a container panel and vice versa; click-away
closes the panel *and* folds the group; Esc closes only the panel.  With
nothing loaded the Experiment tile is the single exception to
"dimmed-but-clickable": it opens no panel, so a click could not show the fix —
it is dimmed *and* inert (`set_clickable(False)`), and its hint names where
the fix is.  Unloading folds the group on the lit→dimmed transition only, so
the refresh that follows every finished task cannot fold away a
programmatically opened subtile panel.  A subtile panel whose group cannot
expand is refused outright — a panel anchored to a hidden tile would float in
space.

**The Project panel reads top to bottom as project identity, the members,
then what to do with them.**  *Create/Load* holds the three disjoint ways in —
Open Project / Create project… / Initialize existing directory… — plus
Project design…, a full-width Validate YAMLs (a check over the open Project,
not a fifth way in), and the loaded-project summary, which lives in this
always-visible card so a load failure is readable while the sections below
stay down.  *Experiments* holds the members table and the same three cases
one level down (Create member… / Initialize existing directory… / Member
configs…), the filing repair, and the load options, beside the table that
triggers the load.  *Analysis* holds the pooled actions — including the
Project-level publication-figure buttons, moved here from the per-member
Plots panel because `plot_specs.yaml` and `figures/` live at the project
root — and the Project Script row.  It is built hidden and appears only when
a Project is open: the Experiments card stays visible-but-gated instead, so
an empty Project still shows where members will appear.  Two empty-state
strategies, both deliberate.

**Loading reveals Analyze *after* the load.**  Double-clicking a member used
to open the Analyze panel immediately; the group that panel hangs from now
does not exist until a member is loaded, so the reveal moved into the load's
completion callback.

## Consequences

* The ribbon costs one extra row (46px) only while the group is expanded.
* `AnalysisHubWindow.tiles` still holds every tile by key and
  `AnalysisHubWindow.panels` every panel, so tests addressing them by key
  survived the split; the `experiment` key exists in `tiles` but not
  `panels`.
* AI moved into the group and follows its gate — enabled while a member of an
  open Project is loaded (plus a provider key).  Because the narrative itself
  is about the Project's Combined Analysis, the Analysis card carries a
  project-level **AI narrative…** button, so the action stays reachable with
  just a Project open; and a standalone experiment (loaded with no Project)
  keeps the subtile dimmed rather than lighting a dead end.
* `Category.AI` joins the palette (`#0d9488`/`#2dd4bf`), and the surface
  colors adopt upstream's values, so the two Hubs read as one family.
