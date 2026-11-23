---
status: accepted
---

# A Member may read its recording's own config

Extends ADR-0021 (a Member may read its recording from elsewhere) from the data
to the config, and amends ADR-0005 and ADR-0025 for such a Member.

## Context

The lab's catalogue keeps each recording in a run folder: the recording in
`data/` and its `flic_config.yaml` beside it.  An analysis of runs is a pyflic
Project, and until now each of its Members held a copy of the run's config,
with `data_folder:` pointing back at the data (ADR-0021).  One recording then
had a config in its run folder and another in every Project that analysed it,
and they drifted apart.  On 2026-11-22 a PI Direction corrected in a run's
config never reached the Project's copy, which went on reading those DFMs with
well A on the left, and an empty `params: {}` in a copy stopped the Project
loading (ADR-0030).

The lab head decided: "There should only be a single config yaml.  Software
parameters can exist in the project yaml and be specific for different
analyses."

## Decision

- **A Member may hold `flic_member.yaml` instead of a config.**  Its `config:`
  names the recording's `flic_config.yaml`, and its optional `data_folder:`
  names the data; both are resolved against the Member folder, as
  `data_folder:` is (ADR-0021).  Without `data_folder:` the data are the
  `data/` beside the named config, or the folder that config's own
  `data_folder:` names, resolved from there.
- **One resolver answers which config a Member reads**:
  `layout.config_location`, beside `layout.data_location`.  The loader, the
  Project, the classifier, the disk cache, the Exclusion Sheet, the QC Viewer,
  the Hub and the editors all ask it.
- **The named config is only read from the Member.**  Everything pyflic writes
  for the Member (`analysis/`, `qc/`, `remove_chambers.csv`, the cache) stays
  in the Member folder.  *Write the design into the members* never writes a
  Member that names its config, and a Selection of such a Member outside a
  Project has nowhere to go and is refused.
- **Inside a Project its analysis settings are the design's alone.**  Of the
  named config's `global:`, only the description of the recording
  (`DESCRIPTION_KEYS`: type, layout, well names, factors, control level) is
  read, and it must agree with the design, as any Member's must.  The settings
  it states (`params:`, `constants:`, `transform_licks:`…) are not used, and
  neither is a per-DFM setting other than the physical ones
  (`PHYSICAL_DFM_KEYS`); the Project names them in a warning, never an error,
  so one recording may be analysed by several Projects with different
  settings.  Loaded on its own, outside a Project, the config is read whole.
- **What blocks such a Member** is reported by the classifier and the loader,
  each naming the key, the value as written and the resolved path: a
  `config:` that is missing or not a path, a config that does not exist or is
  a folder, a Member that also holds a `flic_config.yaml` ("which is the
  experiment's config?") or a recording of its own, and a recording folder
  that holds the Member folder.
- **Two Members naming one recording are one Member**, as two naming one data
  folder are (ADR-0021).

## Consequences

- A chamber map or PI Direction corrected in the recording's config reaches
  every Project that reads it, at once, and so may change what re-running an
  analysis computes.  The lab's catalogue records a fingerprint of each named
  config when an analysis is marked done and warns when it changes.
- A Member holding its own config behaves exactly as before.  A Project may
  mix the two kinds.
- `flic_member.yaml` is pyflic's own file name.  PyTrackingAnalysis has no
  equivalent yet; its Members still hold copies.
