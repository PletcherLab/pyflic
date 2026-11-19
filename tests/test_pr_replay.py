"""The ratio replay behind the Simulated Yoked Breaking Point (ADR-0023).

:mod:`pyflic.base.pr_replay` on hand-built arrays: the contact credit a 5 Hz
record is worth, its fit to the firmware's recorded Progress, and the replay
of a progressive ratio counting Contact (``LEDControl.c``, ``StepRatio``).
The replay on whole recordings is in ``test_pr_simulated_yoked.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyflic.base import pr_replay as prp

SAMPLE = 200.0


def series(first: int = 1000, step: int = 1000):
    return lambda k: first + step * (k - 1)


def replay(credit, start=0, end=None, *, activation=500.0, requirement=None):
    credit = np.asarray(credit, dtype=float)
    return prp.replay_contact_ratio(credit, start, credit.size if end is None else end,
                                    requirement_for=requirement or series(),
                                    activation_ms=activation, sample_ms=SAMPLE)


# ---------------------------------------------------------------------------
# Contact credit
# ---------------------------------------------------------------------------

def test_the_flat_credit_is_the_whole_record_above_threshold_and_nothing_below():
    flat = prp.ContactCredit.flat(SAMPLE)
    above = np.array([False, True, True, False])
    assert flat.credit(above).tolist() == [0.0, 200.0, 200.0, 0.0]


def test_a_records_credit_depends_on_its_neighbours():
    credit = prp.ContactCredit(sample_ms=SAMPLE, ms={(False, True, False): 120.0,
                                                     (True, True, True): 199.0,
                                                     (True, False, False): 110.0})
    above = np.array([False, True, False, False, True, True, True, False])
    out = credit.credit(above)
    assert out[1] == 120.0                   # an isolated record: one touch
    assert out[2] == 110.0                   # just after a bout: its tail
    assert out[5] == 199.0                   # inside a bout
    ## A pattern the credit does not name takes the flat value.
    assert out[4] == 200.0 and out[0] == 0.0


def test_the_fit_recovers_the_contact_the_firmware_counted():
    rng = np.random.default_rng(3)
    above = rng.random(5000) < 0.3
    truth = {p: (190.0 if p[1] else 20.0) - (40.0 if not p[0] else 0.0) for p in prp.PATTERNS}
    truth = {p: max(v, 0.0) for p, v in truth.items()}
    rise = prp.ContactCredit(sample_ms=SAMPLE, ms=truth).credit(above)
    well = prp.WellRecord(above=above, progress=np.cumsum(rise),
                          requirement=np.full(above.size, 1e9),
                          activations=np.zeros(above.size),
                          acclimation=np.zeros(above.size, dtype=bool),
                          on_ms=np.zeros(above.size))
    fit = prp.fit_contact_credit([well], sample_ms=SAMPLE, activation_ms=500)
    for p in prp.PATTERNS:
        assert fit.value(p) == pytest.approx(truth[p])
    assert fit.fitted == above.size - 1      # the first record has no rise


def test_the_fit_leaves_out_acclimation_lights_and_activations():
    n = 40
    above = np.ones(n, dtype=bool)
    progress = np.arange(n) * 200.0
    acclimation = np.zeros(n, dtype=bool)
    acclimation[:10] = True                  # Training rows
    on_ms = np.zeros(n)
    on_ms[20] = 150                          # a light in record 20
    activations = np.zeros(n)
    activations[30:] = 1                     # an Activation counted in record 30
    progress[30:] = (np.arange(n - 30)) * 200.0   # Progress restarts there
    well = prp.WellRecord(above, progress, np.full(n, 5e4), activations, acclimation, on_ms)
    fit = prp.fit_contact_credit([well], sample_ms=SAMPLE, activation_ms=500)
    ## Records 1-10 (Acclimation, or the first after it), 20-23 (the light and
    ## the three an Activation of 500 ms reaches from) and 30 are out.
    assert fit.fitted == n - 11 - 4 - 1
    assert fit.value((True, True, True)) == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# The replay
# ---------------------------------------------------------------------------

def test_light_k_needs_k_seconds_of_contact_and_contact_in_a_light_does_not_count():
    r = replay(np.full(60, 200.0))
    ## 1 s of contact by the end of record 4.  The 500-ms light covers records
    ## 5, 6 and half of 7, so 2 s more are reached half-way through record 17.
    assert r.onsets.tolist()[:2] == [4, 17]
    assert r.requirements.tolist()[:2] == [1000.0, 2000.0]


def test_contact_carries_over_across_bouts_and_after_a_light():
    credit = np.zeros(40)
    credit[[0, 1, 2]] = 200.0                # 600 ms
    credit[[10, 11]] = 200.0                 # 400 more: 1 s in record 11
    credit[12:16] = 200.0                    # lit to half-way through 14, then 300 ms
    r = replay(credit)
    assert r.onsets.tolist() == [11]
    ## Record 14's second half and record 15 counted toward light 2.
    credit[25:34] = 200.0                    # 1 700 ms more: 2 000 ms in record 33
    assert replay(credit).onsets.tolist() == [11, 33]


def test_the_replay_starts_at_its_start_with_the_first_requirement():
    credit = np.full(30, 200.0)
    assert replay(credit, start=10).onsets.tolist()[0] == 14
    assert replay(credit, start=10, end=14).onsets.size == 0


def test_no_contact_no_light():
    r = replay(np.zeros(100))
    assert r.onsets.size == 0 and r.requirements.size == 0


def test_the_series_is_the_programs():
    r = replay(np.full(100, 200.0), requirement=series(first=400, step=0), activation=0)
    ## A fixed 400 ms with no light: every second record.
    assert r.onsets.tolist()[:4] == [1, 3, 5, 7]
    assert replay(np.full(100, 200.0), requirement=lambda k: None).onsets.size == 0
