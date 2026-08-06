# Summary metrics

Once events are detected, pyflic computes one row of metrics per chamber. This is the
**feeding summary**, written to `analysis/feeding_summary.csv`, and it is what statistics
and plots are built from.

## The lick transform

Read this before interpreting any lick count.

By default, **lick counts in the feeding summary are fourth-root transformed** —
`Licks ** 0.25`. Lick counts are strongly right-skewed and variance grows with the mean,
which violates the assumptions of ANOVA and linear models; the fourth root pulls them
toward normality so those tests behave.

It is controlled per experiment:

```yaml
global:
  transform_licks: false     # default is true
```

Three consequences follow, and all three surprise people:

- A `Licks` value of `6.2` in `feeding_summary.csv` is **not 6.2 licks** — it is
  `6.2⁴ ≈ 1478` licks. Raise it to the fourth power to recover the count.
- `summary.txt` reports **untransformed** licks, so the two output files legitimately
  disagree. Neither is wrong; they are on different scales.
- **The preference index is unaffected.** PI and event PI are computed from raw counts
  before the transform is applied, so turning the transform on or off never changes a PI.

Only lick counts are transformed. Events, durations, intervals and intensities are always
raw.

## Metrics for every experiment

| Column | Meaning |
|---|---|
| `Licks` | Number of lick samples (transformed by default — see above) |
| `Events` | Number of feeding events (bouts) |
| `MeanDuration`, `MedDuration` | Mean and median event duration, in **seconds** |
| `MeanTimeBtw`, `MedTimeBtw` | Mean and median interval between consecutive events, in seconds |
| `MeanInt`, `MedianInt`, `MinInt`, `MaxInt` | Intensity of the baselined signal within events |
| `OptoOn_sec` | Seconds the light was on for that well |
| `StartMin`, `EndMin` | The time range this row was computed over |

**Duration** is event length in samples divided by `samples_per_second`, so it is real
seconds regardless of your sampling rate. **Intensity** summarises the baselined signal
inside events — roughly, how strong the contact was, as distinct from how long it lasted.

`MedDuration` is usually the more trustworthy duration measure: bout lengths are skewed,
and a single very long event moves the mean far more than the median.

## Additional metrics for two-well experiments

Two-well chambers report per-well values, suffixed `A` and `B` according to
[`pi_direction`](concepts-two-well-pi.md), plus the preference indices:

| Column | Meaning |
|---|---|
| `PI` | Preference index from lick counts, −1 to +1 |
| `EventPI` | Preference index from event counts |
| `LicksA`, `LicksB` | Licks per well |
| `EventsA`, `EventsB` | Events per well |
| `MeanDurationA/B`, `MedDurationA/B` | Durations per well |
| `MeanTimeBtwA/B`, `MedTimeBtwA/B` | Inter-event intervals per well |
| `MeanIntA/B`, `MedianIntA/B`, `MinIntA/B`, `MaxIntA/B` | Intensities per well |
| `OptoOn_sec_A`, `OptoOn_sec_B` | Light-on seconds per well |

## Metrics over time

`feeding_summary` collapses the whole recording into one number per chamber. When you care
about *when* feeding happened, use the **binned feeding summary**, which computes the same
metrics within fixed-width time bins:

```python
exp.binned_feeding_summary(binsize_min=30)
```

This is what the binned time-course plots are drawn from — see
[Plot catalogue](plots-catalog.md).

## Empty chambers

A chamber that recorded no events has no duration or interval to report, and those columns
are `NaN` rather than `0`. A chamber with no licks at all has an undefined PI, also `NaN`.
This distinction is deliberate: a fly that never fed is not a fly that fed with zero
duration, and treating the two as equal will bias any average you compute. Exclude
inactive chambers rather than letting them through — see
[exclusions](config-dfms-chambers.md#excluding-chambers).

---

Related: [How feeding is detected](concepts-licks-events.md) ·
[Two-well choice and the preference index](concepts-two-well-pi.md)
