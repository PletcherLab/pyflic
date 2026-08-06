# Parameter reference

Every detection parameter, what it does, and which way your results move when you change
it. These go under `global.params`, and any DFM may override any of them for itself.

```yaml
global:
  params:
    chamber_size: 2
    feeding_threshold: 20
    feeding_event_link_gap: 5
```

## Defaults at a glance

pyflic starts from a preset chosen by `chamber_size` — it never uses a bare set of
defaults — so these are the values you actually get when you leave a parameter out:

| Parameter | `chamber_size: 1` | `chamber_size: 2` | Units |
|---|---|---|---|
| `baseline_window_minutes` | `3` | `3` | minutes |
| `samples_per_second` | `5` | `5` | Hz |
| `feeding_threshold` | `20` | `20` | signal |
| `feeding_minimum` | `10` | `10` | signal |
| `feeding_minevents` | `1` | `1` | samples |
| `feeding_event_link_gap` | `5` | `5` | samples |
| `tasting_minimum` | `5` | `5` | signal |
| `tasting_maximum` | `20` | `20` | signal |
| `tasting_minevents` | `1` | `1` | samples |
| `pi_direction` | `left` | `left` | — |
| `correct_for_dual_feeding` | `false` | **`true`** | — |
| `chamber_size` | *required* | *required* | wells |

Only `chamber_size` and `correct_for_dual_feeding` differ between the two presets.

---

## `chamber_size`

**Required.** Wells per chamber: `1` or `2`.

pyflic raises an error if it is missing rather than guessing, because the value determines
which physical wells are grouped into a chamber — the wrong value silently reinterprets
your entire plate rather than failing visibly.

`1` gives 12 independent chambers per DFM. `2` pairs wells into 6 chambers, and enables the
preference index. Any other value is rejected.

## `baseline_window_minutes`

**Default `3`.** Width in minutes of the centred running-median window used for baseline
subtraction. At 5 Hz, 3 minutes is 900 samples.

**Longer** gives a more stable baseline that is less disturbed by feeding, but tracks
genuine drift more slowly. **Shorter** follows drift closely but is more easily dragged
upward by the feeding you are trying to measure.

Change it only if the QC traces show the baseline misbehaving. See
[The raw signal and the baseline](concepts-signal.md).

## `samples_per_second`

**Default `5`.** The hardware sampling rate in Hz. Set it to match your rig.

This is not a tuning knob — it is a statement of fact about your hardware, and getting it
wrong corrupts every derived quantity. Durations are computed as samples ÷
`samples_per_second`, so a wrong value scales every duration and interval in your output
by a constant factor while leaving lick and event *counts* untouched. That combination is
easy to miss, because the counts look perfectly reasonable.

It also silently reinterprets `feeding_minevents` and `feeding_event_link_gap`, which are
specified in samples.

## `feeding_threshold`

**Default `20`.** The **upper** lick threshold. A run of candidate licks is kept as a real
event only if at least one sample in it exceeds this value.

This decides *whether* a bout is real. It does not decide how long the bout is — that is
`feeding_minimum`'s job.

**Higher** is more conservative: fewer events, and weak feeding may disappear entirely.
**Lower** admits more events, and eventually noise.

A **negative** value switches all four thresholds into adaptive mode — see
[Fixed and adaptive thresholds](concepts-licks-events.md#fixed-and-adaptive-thresholds).

## `feeding_minimum`

**Default `10`.** The **lower** lick threshold. Samples above it are candidate licks.

This decides *how long* a bout is. Because the whole run above `feeding_minimum` is kept
once the run qualifies, this parameter sets the measured duration of every event.

**Higher** truncates bouts, shortening durations and reducing lick counts. **Lower**
extends bouts and eventually merges them with noise. It should sit below
`feeding_threshold`; setting the two equal collapses the dual-threshold design into a
single-threshold one and will systematically shorten your bouts.

## `feeding_minevents`

**Default `1`.** Minimum event length in **samples**. Shorter events are discarded.

At the default of 1 and 5 Hz, even a single 0.2 s contact counts as an event. Raise it to
exclude the briefest contacts — `3` requires 0.6 s at 5 Hz.

Note the ordering: this filter runs **before** event linking, so two brief contacts that
would together exceed the minimum are each tested separately and may both be discarded
before they have any chance to merge.

## `feeding_event_link_gap`

**Default `5`** — one second at 5 Hz. The maximum gap, in **samples**, between two events
that will be bridged into a single bout.

A fly briefly breaking contact mid-meal should not be recorded as two meals; this is the
parameter that decides how brief "briefly" is.

It has a **substantial** effect on your results, larger than most users expect:

- **Larger** merges more events → **fewer, longer** bouts, and **more** licks (bridged
  gaps are filled with licks).
- **Smaller** preserves interruptions → **more, shorter** bouts.

Only gaps with a real event on both sides are bridged; runs at the start or end of a
recording are never converted. Any comparison of event counts or durations between
conditions is meaningful only if both used the same link gap. If you are unsure what value
to use, run a sensitivity sweep and find where your effect is stable.

See [Event linking](concepts-licks-events.md#event-linking-the-link-gap).

## `tasting_minimum`

**Default `5`.** Lower bound of the tasting band. Contacts below it are treated as noise.

## `tasting_maximum`

**Default `20`.** Upper bound of the tasting band. Contacts above it are feeding, not
tasting.

It defaults to the same value as `feeding_threshold`, so the tasting band sits directly
below the level at which a contact is confirmed as feeding.

## `tasting_minevents`

**Default `1`.** Minimum tasting-event length in samples — the tasting counterpart of
`feeding_minevents`.

Remember that tasting is defined as *what feeding did not claim*, so changing your feeding
parameters changes tasting results even when the tasting parameters are untouched. See
[Tasting](concepts-tasting.md).

## `pi_direction`

**Default `left`.** Which physical side is Well A in a two-well chamber.

- `left` — the odd-numbered well (W1, W3, W5, …) is Well A
- `right` — the even-numbered well (W2, W4, W6, …) is Well A

Set it **per DFM** to counterbalance food position, so a positive preference index always
means the same substance regardless of which side it sat on. This is the main reason to
override a parameter at DFM level. See
[Two-well choice and the preference index](concepts-two-well-pi.md).

For backwards compatibility a numeric `pi_multiplier` is accepted: `1` maps to `left`,
anything else to `right`.

## `correct_for_dual_feeding`

**Default `true` for two-well experiments, `false` for single-well.**

Detects samples where both wells of a chamber register feeding at once, adjusts the
baseline of the non-preferred well to remove electrical crosstalk, and re-runs detection on
the corrected signal.

Because it is on by default for choice assays, your results depend on it whether or not you
chose it — state it in your methods. Uncorrected crosstalk biases every preference index
toward zero, which reads as indifference.

Meaningless for single-well experiments, where there is no neighbouring well to bleed in.

## `chamber_sets`

Which wells make up each chamber. You will rarely set this.

The defaults follow the physical layout: for `chamber_size: 2`, the pairs
`(1,2), (3,4), (5,6), (7,8), (9,10), (11,12)`; for `chamber_size: 1`, wells 1 through 12
individually. Values must be between 1 and 12, and the number of columns must equal
`chamber_size`.

Override it only if your rig is wired unusually.

---

## Accepted aliases

These older names are accepted and mapped to the canonical parameter:

| Alias | Canonical name |
|---|---|
| `baseline_window`, `baseline_window_min` | `baseline_window_minutes` |
| `link_gap` | `feeding_event_link_gap` |
| `tasting_low` | `tasting_minimum` |
| `tasting_high` | `tasting_maximum` |
| `samples_per_sec` | `samples_per_second` |
| `pi_multiplier` | `pi_direction` (`1` → `left`, otherwise `right`) |

The Config Editor writes canonical names. Prefer them in new configurations.

---

Related: [How feeding is detected](concepts-licks-events.md) ·
[Configuration file structure](config-structure.md) · [QC Viewer](app-qc-viewer.md)
