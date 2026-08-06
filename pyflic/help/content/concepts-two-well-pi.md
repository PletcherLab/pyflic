# Two-well choice and the preference index

In a two-well experiment each chamber offers the fly two options, and the question is not
how much it ate but **which one it chose**. That requires a stable way to say which well
is which.

## Well A and Well B

Wells within a DFM are numbered `W1` to `W12`. Chambers pair them: chamber 1 is wells 1
and 2, chamber 2 is wells 3 and 4, and so on up to chamber 6.

The `pi_direction` parameter maps physical position to the logical labels **Well A** and
**Well B**:

| `pi_direction` | Well A is | Well B is |
|---|---|---|
| `left` (default) | the odd-numbered well — W1, W3, W5, … | the even-numbered well |
| `right` | the even-numbered well — W2, W4, W6, … | the odd-numbered well |

## Why you would ever set it to `right`

Because side bias is real. A fly may prefer the left position for reasons that have
nothing to do with what is in the well, and if every DFM has the test substance on the
left, that bias is indistinguishable from a preference for the substance.

The standard remedy is **counterbalancing**: put the test substance on the left in half
your DFMs and on the right in the other half, then set `pi_direction` per DFM so that
Well A always means *the test substance* regardless of where it physically sits.

```yaml
dfms:
  1:
    params:
      pi_direction: left      # test substance in the odd wells
    chambers: { 1: Sucrose, 2: Sucrose }
  2:
    params:
      pi_direction: right     # test substance in the even wells
    chambers: { 1: Sucrose, 2: Sucrose }
```

Done this way, a positive preference index always means preference for the test substance,
and position bias averages out instead of masquerading as an effect.

## The preference index

The **preference index (PI)** is computed from lick counts:

```
PI = (LicksA − LicksB) / (LicksA + LicksB)
```

It ranges from **−1** (all licks on Well B) through **0** (equal) to **+1** (all licks on
Well A).

An **event PI** is computed the same way from event counts rather than lick counts. The
two can disagree, and the disagreement is informative: a fly that made many short visits to
one well and a few long meals at the other will have an event PI and a lick PI pointing in
opposite directions. Report which one you used.

Both are undefined when a chamber recorded no licks at all — a chamber with zero activity
has no preference, and you should exclude it rather than treat it as indifferent. See
[exclusions](config-dfms-chambers.md#excluding-chambers).

## Dual-feeding correction

Because the two wells of a chamber are electrically close, a strong signal in one can bleed
into the other, producing apparent simultaneous feeding at both. Left uncorrected, this
pushes every PI toward zero — it looks like indifference.

When `correct_for_dual_feeding` is enabled, pyflic finds samples where both wells register
feeding licks at once, adjusts the baseline of the non-preferred well to remove the
crosstalk contribution, and re-runs feeding detection on the corrected signal.

**This defaults to `true` for two-well experiments** and `false` for single-well ones,
where it has nothing to correct. It is on unless you turn it off, which is usually what
you want — but it does mean your results depend on it, so state it in your methods.

The QC viewer has a **Sim. Feeding** tab showing where simultaneous feeding was detected,
which is the place to look if you suspect crosstalk is distorting a plate.

## Naming the wells in output

```yaml
global:
  well_names:
    A: Sucrose
    B: Yeast
```

These labels are used in plots and reports, so figures read `Sucrose` and `Yeast` rather
than `A` and `B`. They are cosmetic — they do not affect detection or the PI.

---

Related: [Experiment types](concepts-experiment-types.md) ·
[Plot catalogue](plots-catalog.md) · [Parameter reference](reference-parameters.md)
