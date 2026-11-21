# Experiment Editor

A short form for one recording, for the moment you file it — usually before any Project
exists. It says what the recording is, what the design was, and which chamber had which
levels, and leaves every analysis setting to the [Config Editor](app-config-editor.md).

```bash
pyflic-experiment                     # the current folder
pyflic-experiment <folder>            # a recording's folder, or its flic_config.yaml
pyflic experiment <folder>            # the same
```

In the Analysis Hub, **Experiment editor** in the [Tools panel](app-hub.md#tools-panel)
opens it on the loaded member's config.

Pointed at a folder with no `flic_config.yaml` yet, it opens one DFM tab for each DFM its
data files name, and **Save** writes the config. **File → Open in the Config Editor**
opens the same file in the full editor.

## About

Everything on this tab is optional and is saved under [`about:`](config-structure.md#about).
pyflic never uses it to analyse; it is for whoever reads the config later.

- **Filed as**: the recording's id in the catalogue that filed it (`about.run`). It appears
  only for a filed recording. The catalogue writes it; you cannot change it here, and a
  save writes it back as it was.
- **Start from another recording…**: see [below](#start-from-another-recording).
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

## Start from another recording

A replicate shares most of its description with the recording before it. **Start from
another recording…**, on the About tab and in the File menu, asks for that recording's
`flic_config.yaml` and fills in this form from it. Nothing is saved until you save.

It takes what replicates share:

- the experimenter, the series, the organism, the sex, the age at the start and the
  protocol;
- each design factor's kind and what each level means, for the factors and levels this
  recording declares. A level's registry term comes with its meaning.

It leaves the **title**, the **purpose** and the **notes** as they are, and it never takes
**Filed as**: those belong to one recording.

A recording with no design yet takes more. When this recording has no design factors and
no chamber assignments, and is not a member of a Project with a design, the action also
takes the other recording's:

- **design**: Experiment Type, Chamber Layout, well names, design factors and control
  level;
- **chamber map**, for the DFMs this recording's data holds: each chamber's levels and,
  for a Progressive Ratio type, the chamber groups. A DFM only the other recording has is
  not added, and a DFM only this one has stays unassigned.

It takes no analysis setting and none of a DFM's parameter overrides. If you have already
chosen a type or a single-well layout, or typed a well name or a control level that the
other recording does not share, it asks before replacing them. Answer **No** and only the
description is taken. What it says about factors this recording does not declare yet is
kept, and the editor names them when you save; declare the factor on the Design tab and
its kind and meanings appear.

## Design

- **Experiment Type** and **Chamber Layout**: as in the Config Editor. A type that fixes
  the layout says so beside it.
- **Well A** and **Well B**: what each well held, for a two-well layout.
- **Control Level**: shown for a type that compares every level of a factor with a
  control.
- **Design factors**: each factor, its **kind** (genotype, treatment, diet, condition, sex
  or other) and its levels, comma-separated. See [Factorial designs](config-factors.md).
- **What each level means**: a few words per level — the cross, the dose, the diet. The
  rows follow the factors; renaming a level drops what was typed for the old name. Typing
  here changes only the level's words: a registry term a catalogue gave the level is kept.

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

Inside `about:`, the editor writes the fields it shows and keeps the rest as read: the
catalogue's id, a level's registry term, any key pyflic does not know, and what the file
says about a factor or level the design does not declare.

Inside a Project, the design is the Project's: it is shown read-only, and a save writes
only a copy of the design's description under `global:`, as the Config Editor does — see
[Projects and members](concepts-project.md). The About tab, the factor kinds and meanings,
and the chambers stay this recording's own.

**Save As** into another recording's folder leaves the catalogue's id out: it names the
recording the config was read from. A config already in that folder keeps its own id.

## When the file changed on disk

Another program may write the file while it is open here: the Config Editor, the Hub, or
a catalogue that files the recording and then rewrites its `about:`. When that has
happened, **Save** tells you first and offers three answers:

- **Reload** reads the file again. What you changed here is lost.
- **Overwrite** saves this editor's About, design and chambers over the file as it is now.
  Everything else in the file is kept as it is now, the catalogue's id and registry terms
  included.
- **Cancel** does neither.

The editor never overwrites a changed file without asking.
