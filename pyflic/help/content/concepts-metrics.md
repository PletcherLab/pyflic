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
- The auto-removal cutoff `min_untransformed_licks_cutoff` works on **untransformed**
  licks, and so do the lick counts `summary.txt` quotes when it says why a chamber was
  removed. They are on a different scale from the CSV's; neither is wrong.
- **The preference index is unaffected.** PI and event PI are computed from raw counts
  before the transform is applied, so turning the transform on or off never changes a PI.

Two more follow from it. A two-well **total** of licks, wherever a table, a plot or a
statistic adds the two wells (a two-well `Licks` figure, the binned plots' `total` mode),
is the root of the summed counts, (A + B)^¼, not A^¼ + B^¼. A **difference** of licks, such
as a paired − yoked `dLicksA`, is the difference of the transformed values, a contrast on
the fourth-root scale; its labels say "¼-root scale".

Only lick counts are transformed. Events, durations, intervals and intensities are always
raw.

## Metrics for every experiment

Each row names its `DFM`, `Chamber` and `Treatment`, and carries a column per design factor
when the experiment has any. Only chambers assigned to a treatment have a row.

| Column | Meaning |
|---|---|
| `Licks` | Number of lick samples (transformed by default — see above) |
| `Events` | Number of feeding events (bouts) |
| `MeanDuration`, `MedDuration` | Mean and median event duration, in **seconds** |
| `MeanTimeBtw`, `MedTimeBtw` | Mean and median length, in seconds, of the stretches without feeding: the gaps between events, and also the stretch before the first event and after the last; each counted in the window where it begins, and `NaN` in a window where none begins |
| `MeanInt`, `MedianInt`, `MinInt`, `MaxInt` | Intensity of the baselined signal within events |
| `OptoOn_sec` | Seconds the light was on for that well |
| `StartMin`, `EndMin` | The time range this row was computed over; `inf` for an open end (the last Facet), and a Slice's own minutes for its unfaceted rows |

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

## Additional columns for optogenetic experiments

Every row of an optogenetic experiment also says what the chamber's light did:

- `Role`, after `Chamber` — what the chamber was to the light: Paired, Yoked, Mutual, Self,
  Open or Unlit. See [Roles](concepts-optogenetics.md#roles).
- `LitSec`, after the `OptoOn_sec` columns — seconds the chamber's own light was on,
  counted once however many of its wells were lit.
- `Activations` — the on-periods its wells' own Paradigms started. Blank on an older
  recording, which does not record them.

They are the [light table](concepts-optogenetics.md#the-light-table)'s columns of the same
name. A DFM the light QC does not cover leaves them blank.

## Additional columns for Progressive Ratio experiments

Every per-chamber row also says where the chamber sits in its chamber group and how the
group fared. See [Progressive Ratio experiments](concepts-progressive-ratio.md).

| Column | Meaning |
|---|---|
| `Group` | The chamber group, numbered in chamber order: 1–3 for pairs (chambers 1+2, 3+4, 5+6, or as `Program.txt` links them), 1–6 for single chambers |
| `Role` | `paired` or `yoked`; every chamber of a design without yoked controls is `paired` |
| `TrainingMinutes` | The group's training end, in minutes — the same on both chambers of a pair; blank if training never completed |
| `TrainingComplete` | Whether the paired fly finished training; `NA` when the recording has no Training at all |
| `LightOn_sec` | Seconds the group's light was on |
| `LightQC` | The group's light QC flags; empty when clean |
| `LickFreeLightEvents` | Test light events the group's paired fly was credited no sucrose lick for |
| `PersistA` | Sucrose persistence: minutes from training end to the chamber's last sucrose feeding event before a pause longer than `pr_break_gap_min`; blank on Training rows |
| `PersistACensored` | `PersistA` is a lower bound: the fly was still feeding when its Test window ended |

The paired − yoked difference of each metric is in `paired_yoked_diff.csv` as `d<metric>`
(`dLicksA`, `dPI`, `dPersistA`, …), for the groups that are pairs. The breaking point is a
property of the group, not of a chamber, so it has its own table, `pr_breaking_point.csv`.

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

A chamber that recorded no events has no duration or intensity to report, and those
columns are `NaN` rather than `0`. A chamber with no licks at all has an undefined PI, also
`NaN`. This distinction is deliberate: a fly that never fed is not a fly that fed with zero
duration, and treating the two as equal will bias any average you compute. Its
`MeanTimeBtw` and `MedTimeBtw` mean nothing: the whole recording is one stretch without
feeding, counted in the window where it begins, so they read as its length there, and
`NaN` in every window where no stretch begins. Exclude inactive
chambers rather than letting them through — see
[exclusions](config-dfms-chambers.md#excluding-chambers).

---

Related: [How feeding is detected](concepts-licks-events.md) ·
[Two-well choice and the preference index](concepts-two-well-pi.md)
