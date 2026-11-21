---
status: accepted
---

# A Member carries a copy of the Design's description of its recording

Amends ADR-0005, whose consequence "a Replicate's config is therefore no
longer readable in isolation — the accepted price of not duplicating the
design" no longer holds.  The Design stays the authority for every key under
`global:`; what changes is what a Member's file says.

## Context

ADR-0005 had a Member omit `global:` and inherit everything from the
`design:` in `project.yaml`, so a Member's `flic_config.yaml` said which
chamber had which levels but not what the levels were, what the wells held,
or what kind of experiment it was.  Read on its own — by `labdata`, by a
person given one folder, or after the folder is copied out of its Project —
the config could not be interpreted.  pyflic made it so: the Config Editor
dropped a Member's `global:` on save, scaffolding wrote none, and saving a
Design offered to delete any a Member still had, discarding its values when
they disagreed.

## Decision

- **The Design Copy.**  A Member's `global:` holds the Design's values of the
  keys that say what the recording *is* — `experiment_type`, `chamber_layout`,
  `well_names`, `experimental_design_factors`, `control_level`
  (`project.DESCRIPTION_KEYS`, `project.design_copy`) — and nothing else.
  Every analysis setting (detection `params:`, `constants:`,
  `transform_licks`, Facets, exclusion groups, Slices, the Selection,
  optogenetics) is still inherited.
- **Validation is unchanged.**  A `global:` that is present is compared key by
  key with the Design, so a copy that differs is a load error, as any stated
  key always was.  A Member without a copy still loads; the Project warns that
  its config does not read on its own.
- **Writing the copy never overwrites what a Member says its recording was.**
  `project.description_conflicts` names the differences that would: another
  Chamber Layout or well name, another control level, a type the Member names
  explicitly, other factors or another order, a level the Design does not
  declare, or chambers that are not levels of the Design's factors.  Such a
  Member is left exactly as written, named with the reason, and refuses to
  load until a person corrects the Design or the Member.  ADR-0005's repair —
  deleting the block — discarded those values; that is the change here.
- **What writing the copy may change.**  It removes a Member's own analysis
  settings (the Design owns them, as before), fills in keys the Member does
  not state, takes the Design's type for a Custom Member when the layout
  agrees (Custom means "unspecified"), and widens the Member's declared levels
  to the Design's.  The chamber map, which says which flies had which level,
  is never touched, so a Member of a genotype panel that declared only its own
  genotypes loses nothing by declaring the panel's.
- **Where it is written.**  The Config Editor's and the Experiment Editor's
  save of a Member; scaffolding a Member; and saving a Design, which offers to
  write it into every Member out of step (`project.member_design_states`,
  `project.adopt_design`) and names the conflicting ones.  `about:`,
  `dfms:`, `scripts:` and `data_folder:` are written back untouched.
- **A first Design declares every level.**  Making a Project of a folder of
  standalone experiments seeds the Design from the first Member's `global:`,
  with each factor's levels widened to every level any Member declares, when
  they declare the same factors in the same order
  (`project.design_from_members`).  Members of a pyflic Project are different
  experiments, so the first one's levels are rarely all of them.

## Consequences

- A Design edit to a description key leaves every Member's copy out of date,
  and the Project refuses to load until the copies are rewritten.  Saving the
  Design offers that at once, defaulting to Yes; declining leaves the Project
  unloadable, with the load error naming the key.
- Existing Projects load unchanged.  Their Members gain copies the next time
  the Design is saved and the offer is accepted, or a Member is saved in an
  editor.
- `labdata`'s run folders are not Members and are never touched; its analysis
  folders hold copies of a run's config, and only those are rewritten.
