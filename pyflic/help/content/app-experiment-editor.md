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
- **Recorded**: the start, read from `Program.txt` in the data, or beside the config in
  an instrument folder. It is shown, not saved.
- **The flies**: organism, sex and age at the start. The organism starts as *Drosophila melanogaster*, the lab's usual one, and is saved with the rest; change it, or empty the box to name none.
- **Protocol and notes**: the protocol's name, version and where it is written down, and
  anything else worth knowing.
- **From Program.txt**: what each DFM ran, in order and for how long, and which PyMCU
  wrote the file (`PyMCU 3.x` or `PyMCU 4.x`). `Program.txt` names no protocol and no
  version, so this is shown, not saved. **Use for protocol** copies it into **Protocol**
  and **Version** where they are blank; a field you have typed in is left alone. Edit what
  it writes before you save: the catalogue files it as the protocol's name.

## Start from another recording

A replicate repeats nearly everything about the recording before it. **Start from
another recording…**, on the About tab and in the File menu, asks for that recording's
`flic_config.yaml` and fills in this form with everything it can take. Nothing is saved
until you save.

It takes:

- **the About**: the title, the purpose, the experimenter, the series, the flies, the
  protocol, the notes, and each design factor's kind and what each level means, with a
  level's registry term. A field the other recording leaves blank stays as it is here. It
  never takes **Filed as**: that is the catalogue's id of the other recording.
- **the design**: Experiment Type, Chamber Layout, well names, design factors and control
  level. Inside a Project with a design, the design stays the Project's.
- **the chamber map**, for the DFMs this recording's data holds: each chamber's levels and,
  for a Progressive Ratio type, the chamber groups. A DFM only the other recording has is
  not added. When the factors change, a DFM only this recording has is cleared, since its
  levels would name nothing. Inside a Project, the map is taken when the other
  recording's factors are the design's.

- **each DFM's PI Direction**, the side well A is on, where the other recording states one
  for a DFM this recording's data holds, on a two-well layout (ADR-0028): for that DFM, or
  for all its DFMs in its `global:` parameters. It is how the DFM was set up, so it is
  taken with or without the chamber map. A DFM the other recording states none for keeps
  its own. A side this editor filled in (see *Chambers*) is replaced without asking.

It takes no analysis setting and none of a DFM's other parameter overrides, optogenetics
or notes: the Hub's **Copy config from…** is for those.

**When this recording already says something.** If what it takes would replace anything
stated here — a title or purpose, an experimenter, a factor's meaning, the type, a well
name, the factors, a DFM's chambers — it lists each one and asks first. **Yes** replaces
them all; **No** leaves the recording exactly as it is. A recording with nothing stated
yet is filled in without a question.

If the other recording names an Experiment Type this one cannot use, it says so and takes
everything but the design.

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

On a two-well layout each DFM also has **PI Direction**, **left** or **right**: the side
well A is on in that DFM's chambers, whose preference index is 1. It is fixed when the
recording is set up, so every DFM states its own, written as that DFM's
`params: {pi_direction: ...}`; there is no inheriting (ADR-0029). A DFM whose config states
none shows the side it is read with, the experiment's (its `global:` parameters, or the
Project design's) or else left, and **Save** writes it. Check each DFM before saving. It is
the one parameter that may differ between DFMs inside a Project, since it says how the DFM
was set up, not how it is analysed. A DFM's other parameter overrides are kept as read; the
Config Editor shows them.

## What a save writes

`about:`, the design keys under `global:`, and `dfms:`. Every other key the file holds —
detection `params:`, `constants:`, Slices, exclusions, the Selection, optogenetics, a
DFM's other parameter overrides, `scripts:`, `data_folder:` — is written back exactly as it was
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
