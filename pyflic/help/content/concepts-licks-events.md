# How feeding is detected

This is the core of pyflic. Every number it reports — lick counts, event counts, bout
durations, preference indices — comes out of the pipeline described here. Understanding it
is what lets you choose parameters deliberately rather than by trial and error.

Detection runs independently on each well, on the [baselined
signal](concepts-signal.md).

## Licks

A **lick** is a single *sample* whose baselined signal is high enough to count as contact
with the food. It is not a behavioural unit — at 5 samples per second, one second of
continuous contact is five licks.

There are two lick thresholds, and the distinction between them matters:

- **`feeding_minimum`** — the lower threshold. A sample above it is a **candidate lick**.
- **`feeding_threshold`** — the upper threshold. A sample above it is a **confirmed lick**.

## Events

An **event** is a contiguous run of licks — one bout of feeding, the actual behavioural
unit. Where "lick" counts samples, "event" counts bouts.

### Surviving events

A run of candidate licks is kept only if **at least one sample within it also crosses the
upper `feeding_threshold`**. Runs that never reach the upper threshold are discarded. A
run that survives this test is a **surviving event** — and the *whole* run is kept, not
just the part above the upper threshold.

This dual-threshold design exists to solve a specific problem. A single threshold forces
an impossible choice: set it high and you measure only the peaks of bouts, truncating
their duration; set it low and electrical noise becomes feeding. Using two lets the upper
threshold decide *whether* something is real while the lower one decides *how long* it
lasted. Real bouts commonly dip below the upper threshold mid-bout, and this keeps them
whole.

### Minimum event length

Surviving events shorter than `feeding_minevents` **samples** are then discarded. The
default is `1`, which keeps everything — at 5 Hz, even a single-sample contact of 0.2 s
counts. Raise it if you want to exclude the briefest contacts from your event counts.

### Event linking (the link gap)

After events are detected, short gaps *between* them are bridged. If two events are
separated by a run of non-lick samples of **`feeding_event_link_gap` samples or fewer**,
that gap is filled in and the two events merge into one.

The default is `5` samples — **1 second** at 5 Hz. The intent is that a fly briefly
breaking contact in the middle of a meal should not be recorded as two meals.

Two details are easy to miss and both matter:

- **Only interior gaps are bridged.** A gap must have a real event on *both* sides.
  Non-lick runs at the very start or end of a recording are never converted, because there
  is nothing to link them to.
- **The gap is filled with licks.** Bridging does not merely join two events in a list —
  the samples in the gap become licks. Merged bouts therefore have a duration spanning the
  gap, and the total lick count rises.

The link gap has a **substantial** effect on your results, larger than most users expect.
A larger gap merges more events, giving you **fewer but longer** bouts and more licks. A
smaller gap preserves brief interruptions as separate events, giving you **more but
shorter** bouts. Any comparison of event counts or bout durations between conditions is
only meaningful if both were analysed with the same link gap.

If you are unsure what value to use, run a
[sensitivity sweep](scripts-actions.md#analyse-actions) across several link gaps and see
where your effect is and is not stable.

### Final events

After linking, the merged lick vector is scanned once more to produce the final event
list. Each event is recorded as a start position and a duration in samples. Those events
are what everything downstream is built from.

## The order of operations

The sequence is fixed, and it explains results that otherwise look strange:

1. Baseline subtraction — [see the previous topic](concepts-signal.md)
2. Compute per-well thresholds
3. Candidate licks (above `feeding_minimum`)
4. Surviving events (runs containing at least one sample above `feeding_threshold`)
5. Discard events shorter than `feeding_minevents`
6. Bridge gaps of `feeding_event_link_gap` samples or fewer
7. Extract the final event list

Note that step 5 happens **before** step 6. Two brief contacts that would together exceed
the minimum length are each tested against it separately, and may both be discarded before
they ever have a chance to be linked.

## Fixed and adaptive thresholds

Thresholds are normally **fixed** — `feeding_threshold: 20` means a baselined signal of 20,
in every well.

If `feeding_threshold` is **negative**, pyflic switches to **adaptive** thresholding, and
all four detection thresholds are recomputed per well as a fraction of that well's maximum
baselined signal:

```
threshold = round(max(baselined signal in this well) × abs(parameter))
```

This applies to `feeding_threshold`, `feeding_minimum`, `tasting_minimum` and
`tasting_maximum` together — the sign of `feeding_threshold` alone selects the mode for all
of them. So `feeding_threshold: -0.5` with `feeding_minimum: -0.2` sets each well's upper
threshold to half of its own maximum and its lower threshold to a fifth.

Adaptive thresholding suits plates where absolute signal amplitude varies widely between
wells for reasons unrelated to feeding. It has a real cost: because each well is scaled by
its own maximum, a well containing a single large artefact will have all of its thresholds
raised, and genuine feeding in that well may fall below them.

## Related

- [Tasting](concepts-tasting.md) — the contacts that are not feeding
- [Parameter reference](reference-parameters.md) — every parameter, its default, and its effect
- [Summary metrics](concepts-metrics.md) — what is computed from these events
