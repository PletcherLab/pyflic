# QC Viewer

```bash
pyflic qc my_experiment/
```

The QC viewer is where you decide whether to believe your results. It shows the signal
pyflic worked from and the events it found, so you can check that detection agrees with
what you would have called by eye.

Do this before running statistics, not after. Detection parameters that are wrong for your
data produce output that looks entirely plausible.

Opened from the Hub's QC panel, the viewer starts directly on the member the Hub already
loaded — no second load. The **Run QC** button in the top bar computes the QC bundle for
that experiment (integrity, data breaks, bleeding, and the raw / baselined /
cumulative-licks plots) and refreshes the DFM tabs in place, so the plots appear without a
round-trip back to the Hub; progress streams to the Load tab's log, and unsaved exclusion
checkboxes survive the refresh. Loading from the Load tab runs the same QC as part of the
load.

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
- **Raw Signal**, **Baselined** and **Cumulative Licks** — the saved QC plots from `qc/`,
  once QC has run.

A Progressive Ratio experiment's own QC — the light QC table and its figures — is on the
Hub's QC panel rather than here; see
[Light QC](concepts-progressive-ratio.md#light-qc). The Feeding Summary tab shows its
`Group`, `Role` and light QC columns like any other.

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

On the **Feeding Summary** tab, tick the chambers to exclude — **Mark All Excluded** and
**Clear All** set every box at once. **Save Exclusions…** writes your current selection to a
named group in
`remove_chambers.csv`. That is the intended path from "this chamber looks wrong" to a
recorded, reusable exclusion set — see
[exclusions](config-dfms-chambers.md#excluding-chambers).

Name the group after the analysis it belongs to. A script's bare `remove_chambers` step
looks for a group named after the script.

## Auto Filter

**Auto Filter** applies the cutoffs from `global.constants` to select chambers
automatically — for a Progressive Ratio experiment, that includes the chamber groups whose
training never completed or whose light QC failed. **View Criteria** shows what the last run
used and why each chamber was selected. Review the selection before saving — automatic
criteria are a starting point, not a verdict.

---

Related: [How feeding is detected](concepts-licks-events.md) ·
[Parameter reference](reference-parameters.md) · [Analysis Hub](app-hub.md)
