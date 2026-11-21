# Experiment Editor

A short form for one recording, for the moment you file it — usually before any Project
exists. It says what the recording is, what the design was, and which chamber had which
levels, and leaves every analysis setting to the [Config Editor](app-config-editor.md).

```bash
pyflic-experiment                     # the current folder
pyflic-experiment <folder>            # a recording's folder, or its flic_config.yaml
pyflic experiment <folder>            # the same
```

Pointed at a folder with no `flic_config.yaml` yet, it opens one DFM tab for each DFM its
data files name, and **Save** writes the config. **File → Open in the Config Editor**
opens the same file in the full editor.

## About

Everything on this tab is optional and is saved under [`about:`](config-structure.md#about).
pyflic never uses it to analyse; it is for whoever reads the config later.

- **Title**: one line. The counter turns amber past 60 characters, which is what fits on
  one line of a list of runs.
- **Purpose**: one or two sentences: what was varied, in which flies, and what was
  measured.
- **Experimenter** and **Series**: who ran it, and a name shared by replicate recordings
  of the same question.
- **Recorded**: the start, read from `Program.txt` in the data. It is shown, not saved.
- **The flies**: organism, sex and age at the start.
- **Protocol and notes**: the protocol's name, version and where it is written down, and
  anything else worth knowing.

## Design

- **Experiment Type** and **Chamber Layout**: as in the Config Editor. A type that fixes
  the layout says so beside it.
- **Well A** and **Well B**: what each well held, for a two-well layout.
- **Control Level**: shown for a type that compares every level of a factor with a
  control.
- **Design factors**: each factor, its **kind** (genotype, treatment, diet, condition, sex
  or other) and its levels, comma-separated. See [Factorial designs](config-factors.md).
- **What each level means**: a few words per level — the cross, the dose, the diet. The
  rows follow the factors; renaming a level drops what was typed for the old name.

## Chambers

One tab per DFM, with the chamber table of the Config Editor: pick a level of each factor
for each chamber, or leave a row blank to leave the chamber out. A Progressive Ratio type
adds the paired chamber of each group. Above the tabs, the editor says which DFMs the data
holds and which tabs have no data; **Add the DFMs found in the data** adds the missing
ones.

## What a save writes

`about:`, the design keys under `global:`, and `dfms:`. Every other key the file holds —
detection `params:`, `constants:`, Slices, exclusions, the Selection, optogenetics, a
DFM's parameter overrides, `scripts:`, `data_folder:` — is written back exactly as it was
read. A new standalone config gets an empty `params:`, which means every default.

Inside a Project, the design is the Project's: it is shown read-only, and a save writes
only a copy of the design's description under `global:`, as the Config Editor does — see
[Projects and members](concepts-project.md). The About tab, the factor kinds and meanings,
and the chambers stay this recording's own.
