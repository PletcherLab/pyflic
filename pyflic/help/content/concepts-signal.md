# The raw signal and the baseline

Everything pyflic reports is derived from one thing: an electrical signal sampled from
each well. Before any bout can be detected, that signal has to be made comparable across
wells, devices, and time. That is what baselining does.

## What the rig records

Each DFM samples up to 12 wells continuously — by default **5 samples per second**. When
a fly's proboscis contacts the liquid food, it completes a circuit and the signal for that
well rises. Between contacts the signal sits near a background level.

That background level is not a constant. It differs between devices, between wells on the
same device, and it drifts over hours as food evaporates and the well depletes. A fixed
threshold applied to the raw signal would therefore mean something different in every
well, and something different at hour 1 than at hour 20.

## Baseline subtraction

pyflic removes the background with a **running median**: for each sample, the median of
the signal in a centred window around it. The **baselined signal** is the raw signal minus
that running median.

The window is set by `baseline_window_minutes`, default **3 minutes** — 900 samples at
5 Hz.

The assumption behind it is worth stating plainly, because it is the assumption you are
accepting when you use the default: **feeding interactions in any given 3-minute window
are rare enough that the median of that window represents the background rather than the
feeding.** A median is used rather than a mean precisely because it is insensitive to a
minority of extreme values — a few seconds of licking in a 3-minute window barely move it.

Where that assumption breaks, so does the baseline. A fly that feeds almost continuously
for minutes at a stretch will pull its own baseline upward, and the tail of a long bout
can be flattened away. If you work with an assay that produces sustained feeding, inspect
the baselined trace in the [QC Viewer](app-qc-viewer.md) before trusting the event counts.

## What baselining fixes

- **Inter-device variation** — different hardware and different food levels produce
  different absolute signal levels.
- **Slow drift** — evaporation and well depletion over the course of a long recording.

After subtraction, a threshold of 20 means the same thing in every well of every device,
which is what makes a single set of parameters valid across an experiment.

## Adjusting the window

Making `baseline_window_minutes` **longer** produces a more stable baseline that is less
disturbed by feeding, but it tracks genuine drift more slowly. Making it **shorter**
follows drift closely but is more easily pulled up by the feeding you are trying to
measure. The default of 3 minutes suits typical FLIC assays; change it only if you can see
in the QC traces that it is misbehaving.

---

Next: **[How feeding is detected](concepts-licks-events.md)** — what pyflic does with the
baselined signal.
