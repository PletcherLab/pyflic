"""
Property-based tests for ``link_events``.

Lock down algorithmic invariants without needing an external reference:

  - bridging never decreases the number of TRUE samples
  - the count of derived events is monotone non-increasing in ``thresh``
  - thresh==0 is a no-op
  - thresh >= n produces a single TRUE block (or nothing if input was all False)
"""

from __future__ import annotations

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st

from pyflic.base.algorithms.events import get_events, link_events


_arrays = st.lists(st.booleans(), min_size=0, max_size=200).map(np.asarray)


@given(_arrays)
@settings(max_examples=200, deadline=None)
def test_thresh_zero_is_noop(z):
    out = link_events(z, thresh=0)
    np.testing.assert_array_equal(out, z.astype(bool))


@given(_arrays, st.integers(min_value=0, max_value=50))
@settings(max_examples=200, deadline=None)
def test_bridging_does_not_decrease_true_count(z, thresh):
    z = z.astype(bool)
    out = link_events(z, thresh=thresh)
    assert int(out.sum()) >= int(z.sum())


@given(_arrays, st.integers(min_value=0, max_value=20))
@settings(max_examples=150, deadline=None)
def test_event_count_monotone_in_thresh(z, thresh):
    """Linking can only merge events together, never split — so event count
    is non-increasing as thresh grows from 0 to thresh+1."""
    z = z.astype(bool)
    n0 = int(np.count_nonzero(get_events(link_events(z, thresh=thresh))))
    n1 = int(np.count_nonzero(get_events(link_events(z, thresh=thresh + 1))))
    assert n1 <= n0


@given(_arrays)
@settings(max_examples=100, deadline=None)
def test_huge_thresh_fills_interior_only(z):
    """With ``thresh`` larger than the array, every interior FALSE gap is
    bridged, but leading/trailing FALSE runs are preserved (since they
    have nothing to link to)."""
    z = z.astype(bool)
    if z.size == 0:
        return
    out = link_events(z, thresh=z.size + 1)
    trues = np.flatnonzero(z)
    if trues.size == 0:
        # No events → no bridging possible
        assert not bool(out.any())
        return
    first, last = int(trues[0]), int(trues[-1])
    # Interior of [first, last] must all be TRUE
    assert bool(out[first:last + 1].all())
    # Leading and trailing FALSE runs unchanged
    np.testing.assert_array_equal(out[:first], z[:first])
    np.testing.assert_array_equal(out[last + 1:], z[last + 1:])
