# QC Viewer

```bash
pyflic qc my_experiment/
```

The QC viewer is where you decide whether to believe your results. It shows the signal
pyflic worked from and the events it found, so you can check that detection agrees with
what you would have called by eye.

Do this before running statistics, not after. Detection parameters that are wrong for your
data produce output that looks entirely plausible.

## Tabs

| Tab | Shows |
|---|---|
| **Load** | Load the experiment and choose the time range |
| **Feeding Summary** | The per-chamber summary table |
| **DFM *n*** | One tab per device, with per-well traces |
| **Params** | Detection parameters, with live recompute |

Each **DFM** tab has its own sub-tabs:

- **Integrity** — data continuity: gaps, breaks, and anything odd about the recording.
- **Sim. Feeding** — samples where both wells of a chamber registered feeding at once.
  This is your window onto electrical crosstalk; see
  [dual-feeding correction](concepts-two-well-pi.md#dual-feeding-correction).
- **Bleeding** — signal bleeding between wells.

## What to look for

**Does the baseline sit where it should?** It should track the background and ignore
feeding. A baseline that visibly rises during long bouts means the running median is being
pulled up by the feeding itself — see [the baseline](concepts-signal.md).

**Do detected events match the visible bouts?** Events marked where the trace is flat means
thresholds are too low. Obvious feeding left unmarked means they are too high.

**Are bouts being split or merged?** One visible meal broken into many events means the
link gap is too small. Long events spanning obvious gaps means it is too large. This is the
single most common cause of surprising duration and event-count results — see
[event linking](concepts-licks-events.md#event-linking-the-link-gap).

**Are some wells dead?** A well with no signal at all is an empty position or a hardware
fault. Exclude it rather than letting a zero into your averages.

## The Params tab

Change a detection parameter and recompute without editing your configuration or
restarting. This is the right way to choose parameter values: try one, look at the traces,
try another.

The workflow that works:

1. Open a DFM tab and find a chamber whose behaviour you can interpret by eye.
2. Change one parameter in **Params** and recompute.
3. Look at whether detection now matches your reading of the trace.
4. When it does, write that value into your configuration file — **the Params tab does not
   save it for you.**

Each parameter has a `?` button that opens
[Parameter reference](reference-parameters.md) at that parameter.

Changing parameters here affects only this session. It is a place to experiment, not a
place to configure.

## Saving exclusions

**Save removed chambers…** writes your current chamber selection to a named group in
`remove_chambers.csv`. That is the intended path from "this chamber looks wrong" to a
recorded, reusable exclusion set — see
[exclusions](config-dfms-chambers.md#excluding-chambers).

Name the group after the analysis it belongs to. A script's bare `remove_chambers` step
looks for a group named after the script.

## Auto Filter Criteria

Applies the cutoffs from `global.constants` to select chambers automatically. Review what
it selected before saving — automatic criteria are a starting point, not a verdict.

---

Related: [How feeding is detected](concepts-licks-events.md) ·
[Parameter reference](reference-parameters.md) · [Analysis Hub](app-hub.md)
