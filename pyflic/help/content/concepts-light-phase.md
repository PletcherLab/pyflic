# Light state and phase analysis

Feeding in *Drosophila* is strongly rhythmic, so when a fly ate is often as interesting as
how much. If your rig recorded light state, pyflic can split every metric by light phase.

## Where light state comes from

If a DFM's CSV contains an **`OptoCol1`** column — and optionally **`OptoCol2`** — pyflic
decodes the per-well light on/off state from those bit-encoded values and stores it
alongside the signal.

No `OptoCol1` column means no light information. Nothing fails; the light-phase features
simply have nothing to report and the light columns read zero. If you expected light data
and see none, check that your rig was configured to record it — pyflic cannot infer light
state from the feeding signal.

## What it is used for

**Light-on markers.** Cumulative lick plots draw the light transitions, so you can see
feeding bouts in relation to lights-on and lights-off without cross-referencing a separate
file.

**Light-phase summary.** An analysis that splits each chamber's activity into light and
dark phases and reports metrics for each, exported with these columns:

```
DFM, Chamber, Treatment, Phase (light/dark), PhaseSeconds, Licks, Events, ...
```

`PhaseSeconds` is what makes the rest comparable. Light and dark phases are rarely equal
in length across a recording — a run that starts mid-afternoon has an unequal first
phase — so raw counts per phase are not comparable until you normalise by the time spent
in each. Divide by `PhaseSeconds` before comparing phases, or you will report a difference
that is really just a difference in exposure.

**Per-well light-on seconds.** The feeding summary carries `OptoOn_sec` for each chamber
(`OptoOn_sec_A` and `OptoOn_sec_B` for two-well chambers), giving the total time the light
was on for that well.

## Optogenetics

The same columns carry optogenetic stimulation state, which is why they are named `Opto`.
If you drive a channelrhodopsin with the FLIC's light output, the light-phase split
becomes a stimulation-on versus stimulation-off comparison, and everything above applies
unchanged.

An experiment with light in these columns also gets the **optogenetic light QC**, which
checks that the light was where the licks were and reads the MCU's `Program.txt` for the
thresholds, linkage and decay behind it. See
[Optogenetic experiments and the light QC](concepts-optogenetics.md).

---

Related: [Summary metrics](concepts-metrics.md) · [Plot catalogue](plots-catalog.md) ·
[Optogenetic experiments](concepts-optogenetics.md)
