# Tasting

Not every contact with the food is feeding. A fly may touch the liquid briefly without
ingesting — sampling it rather than consuming it. pyflic separates these as **tasting**
events.

## How tasting is identified

A sample is a **tasting lick** when both of these hold:

- its baselined signal falls **between `tasting_minimum` and `tasting_maximum`**, and
- it was **not already classified as a feeding lick**.

The second condition is what keeps the two categories disjoint. Feeding detection runs
first and takes precedence; tasting is identified from what remains. A sample can never be
counted as both.

Contiguous runs of tasting licks are grouped into **tasting events**, and runs shorter
than `tasting_minevents` samples are discarded — the same minimum-length rule that applies
to feeding, with its own parameter.

## The default thresholds

For both single-well and two-well experiments the defaults are:

| Parameter | Default |
|---|---|
| `tasting_minimum` | `5` |
| `tasting_maximum` | `20` |
| `tasting_minevents` | `1` |

Note that `tasting_maximum` (20) equals the default `feeding_threshold` (20). That is
deliberate: the tasting band sits *below* the level at which a contact is confirmed as
feeding, so the two categories meet without overlapping.

## Choosing the band

The tasting window is the region of signal that is clearly a contact but not clearly a
meal. Its lower edge, `tasting_minimum`, is a noise floor — set it too low and electrical
noise is recorded as tasting. Its upper edge, `tasting_maximum`, is where you consider a
contact substantial enough to be feeding instead.

Because tasting is defined as *what feeding did not claim*, changing your feeding
parameters changes your tasting counts even if you leave the tasting parameters alone. If
you raise `feeding_threshold`, contacts that used to be confirmed as feeding may fall into
the tasting band. Always report which detection parameters produced a tasting result.

## Where tasting appears

Tasting events are counted per chamber and appear in the feeding summary alongside feeding
metrics. In two-well experiments they are tracked per well, so you can ask whether a fly
sampled one option and fed on the other — a distinction that lick counts alone would hide.

---

Related: [How feeding is detected](concepts-licks-events.md) ·
[Parameter reference](reference-parameters.md)
