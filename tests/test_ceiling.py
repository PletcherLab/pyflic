"""The signal ceiling (pyflic.base.ceiling).

* A well is at the ceiling in a minute whose Resting Level (per-minute
  median of the raw signal) is 1000 or more; a fly's brief contacts reaching
  the top never count, and a board that saturates at 1016 counts like one at
  1023.
* The integrity report lists every well's minutes at the ceiling, and the
  minutes every well spent there together.
* Auto-removal takes out a chamber with a well at the ceiling longer than
  ``max_ceiling_minutes`` (default 30, every Experiment Type's) of the
  analysed range; ``null`` turns it off, and the minutes are counted inside
  the range.
* The constant is validated, and both editors show it.
"""

from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from pyflic.base import ceiling
from pyflic.base import experiment_types as et
from pyflic.base.pr_light_qc import resting_levels
from pyflic.base.yaml_config import load_experiment_yaml
from pyflic.base.yaml_lint import lint_flic_config

from opto_fixtures import SPS, make_opto_dir

#: The quiet well the fixture pushes to the ceiling, and when.
PINNED_WELL = 8
PINNED_FROM_MIN, PINNED_TO_MIN = 40, 85


def _levels(columns: dict[str, list[float]]) -> pd.DataFrame:
    frame = pd.DataFrame(columns, dtype=float)
    frame.index.name = "Minute"
    return frame


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------

def test_a_well_is_at_the_ceiling_in_a_minute_resting_at_1000_or_more():
    levels = _levels({"W1": [5, 999, 1000, 1016, 1023, 40],
                      "W2": [5, 5, 5, 5, 5, 5]})
    found = ceiling.well_ceilings(levels)
    assert set(found) == {1}
    c = found[1]
    assert (c.minutes, c.first, c.last) == (3, 2.0, 4.0)
    assert c.level == 1016.0
    assert c.describe() == "W1: 3 min at the ceiling (minutes 2-4), resting at 1016"


def test_brief_contacts_that_reach_the_top_do_not_count():
    """A median over a minute ignores licks: a fly touching hard enough to hit
    1023 for two seconds every half minute leaves the well resting at 8."""
    n = 10 * 60 * SPS
    signal = np.full(n, 8.0)
    for start in range(0, n, 30 * SPS):
        signal[start:start + 2 * SPS] = 1023.0
    raw = pd.DataFrame({"W1": signal})
    minutes = np.arange(n) / (SPS * 60.0)
    assert ceiling.well_ceilings(resting_levels(raw, minutes)) == {}


def test_minutes_are_counted_inside_the_range_only():
    levels = _levels({"W3": [5.0] * 10 + [1023.0] * 20})
    assert ceiling.well_ceilings(levels)[3].minutes == 20
    ## (0, 15]: minutes 0-14 by their middles, five of them at the ceiling.
    assert ceiling.well_ceilings(levels, (0, 15))[3].minutes == 5
    assert ceiling.well_ceilings(levels, (0, 10)) == {}
    ## A Program Facet's list of windows pools them.
    assert ceiling.well_ceilings(levels, [(0, 12), (25, 27)])[3].minutes == 4


def test_every_well_at_the_ceiling_together_is_the_board():
    levels = _levels({f"W{w}": [5, 1016, 1016, 1016, 5] for w in range(1, 13)})
    levels.loc[1, "W12"] = 900.0
    assert ceiling.board_minutes(levels) == 2
    assert ceiling.board_minutes(levels, (0, 2)) == 0


def test_removal_reasons_name_each_well_longer_than_the_cutoff():
    found = ceiling.well_ceilings(_levels({"W1": [1023.0] * 31, "W2": [5.0] + [1023.0] * 30}))
    assert ceiling.removal_reasons(found, (1, 2), 30) == [
        "W1 at the signal ceiling 31 min > max_ceiling_minutes=30"]
    assert ceiling.removal_reasons(found, (2,), 30) == []
    assert ceiling.removal_reasons(found, (1, 2), None) == []


def test_the_cutoff_resolves_to_its_default_off_or_a_number():
    assert ceiling.max_minutes({}) == 30.0
    assert ceiling.max_minutes(None) == 30.0
    assert ceiling.max_minutes({"max_ceiling_minutes": None}) is None
    assert ceiling.max_minutes({"max_ceiling_minutes": 90}) == 90.0


def test_the_cutoff_is_validated():
    assert ceiling.ceiling_problems({"max_ceiling_minutes": None}) == []
    assert ceiling.ceiling_problems({"max_ceiling_minutes": 0}) == []
    assert ceiling.ceiling_problems({"max_ceiling_minutes": -1})
    assert ceiling.ceiling_problems({"max_ceiling_minutes": "long"})


@pytest.mark.parametrize("name", ["Custom", "Hedonic", "ProgressiveRatio"])
def test_every_type_defaults_the_cutoff(name):
    t = et.get_experiment_type(name)
    assert t.default_constants["max_ceiling_minutes"] == ceiling.DEFAULT_MAX_MINUTES
    assert t.resolve_constants({"constants": {"max_ceiling_minutes": None}}) \
        ["max_ceiling_minutes"] is None


# ---------------------------------------------------------------------------
# A recording
# ---------------------------------------------------------------------------

def _pinned_dir(root: Path, constants: dict | None = None) -> Path:
    """A 90-minute single-well recording, optogenetics off, whose quiet well
    :data:`PINNED_WELL` sits at 1023 from minute 40 to minute 85."""
    make_opto_dir(root, program=None, optogenetics=False, constants=constants,
                  minutes=90.0)
    path = root / "data" / "DFM1_0.csv"
    df = pd.read_csv(path)
    rows = slice(PINNED_FROM_MIN * 60 * SPS, PINNED_TO_MIN * 60 * SPS - 1)
    df.loc[rows, f"W{PINNED_WELL}"] = 1023.0
    df.to_csv(path, index=False)
    return root


def _load(root: Path):
    with contextlib.redirect_stdout(io.StringIO()):
        return load_experiment_yaml(root, parallel=False, use_disk_cache=False)


def _removed(exp, **kw) -> pd.DataFrame:
    with contextlib.redirect_stdout(io.StringIO()):
        return exp.auto_remove_chambers(**kw)


@pytest.fixture(scope="module")
def pinned(tmp_path_factory):
    return _load(_pinned_dir(tmp_path_factory.mktemp("pinned") / "exp"))


def test_the_integrity_report_lists_the_well(pinned):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report = pinned.dfms[1].integrity_report()
    text = buf.getvalue()
    assert "## Signal ceiling" in text
    assert (f"W{PINNED_WELL}: 45 min at the ceiling (minutes 40-84), resting at 1023"
            in text)
    assert report["ceiling_wells"] == f"W{PINNED_WELL} 45 min"
    assert report["ceiling_board_minutes"] == 0


def test_a_chamber_at_the_ceiling_longer_than_the_cutoff_leaves(pinned):
    removed = _removed(pinned)
    reasons = removed.set_index("Chamber")["Reason"]
    assert reasons[PINNED_WELL] == (f"W{PINNED_WELL} at the signal ceiling 45 min "
                                    f"> max_ceiling_minutes=30")
    assert not reasons.drop(PINNED_WELL).str.contains("ceiling").any()
    assert "max_ceiling_minutes = 30" in pinned.filter_criteria_summary


def test_a_range_before_the_pad_saturated_keeps_the_chamber(tmp_path):
    exp = _load(_pinned_dir(tmp_path / "exp"))
    removed = _removed(exp, range_minutes=(0, 50))
    assert not removed["Reason"].str.contains("ceiling").any()


@pytest.mark.parametrize("value", [None, 60])
def test_null_or_a_longer_cutoff_keeps_the_chamber(tmp_path, value):
    exp = _load(_pinned_dir(tmp_path / "exp", {"max_ceiling_minutes": value}))
    removed = _removed(exp)
    assert not removed["Reason"].str.contains("ceiling").any()
    if value is None:
        assert "max_ceiling_minutes: off" in exp.filter_criteria_summary


def test_summary_text_names_the_well(pinned):
    text = pinned.summary_text(include_qc=True)
    assert (f"- signal ceiling: W{PINNED_WELL}: 45 min at the ceiling "
            f"(minutes 40-84), resting at 1023") in text


def test_lint_refuses_a_negative_cutoff(tmp_path):
    root = make_opto_dir(tmp_path / "exp", program=None, optogenetics=False,
                         constants={"max_ceiling_minutes": -5})
    issues = lint_flic_config(root / "flic_config.yaml")
    assert any("max_ceiling_minutes" in i.message and i.severity == "error"
               for i in issues)


# ---------------------------------------------------------------------------
# The editors
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_the_design_dialog_shows_the_cutoff_checked_and_writes_null_unchecked(app, tmp_path):
    from pyflic.base.design_editor import ProjectDesignDialog

    root = tmp_path / "proj"
    root.mkdir()
    (root / "project.yaml").write_text(yaml.safe_dump({"name": "P", "design": {"global": {
        "experiment_type": "Custom", "chamber_layout": "single_well"}}}), encoding="utf-8")
    dialog = ProjectDesignDialog(start_dir=str(root))
    check = dialog.constant_checks["max_ceiling_minutes"]
    assert check.isChecked()
    assert "30" in dialog.constant_edits["max_ceiling_minutes"].placeholderText()
    assert "max_ceiling_minutes" not in (dialog._build_design()["global"].get("constants") or {})
    check.setChecked(False)
    assert dialog._build_design()["global"]["constants"]["max_ceiling_minutes"] is None


def test_the_config_editor_shows_the_cutoff_for_a_single_well_layout(app, tmp_path):
    from pyflic.base.config_editor import FLICConfigEditor

    root = make_opto_dir(tmp_path / "exp", program=None, optogenetics=False)
    editor = FLICConfigEditor(initial_path=root)
    assert editor._threshold_row_visible("max_ceiling_minutes")
    assert editor._ceiling_check.isChecked()
    assert "30" in editor._ceiling_edit.placeholderText()
    editor._ceiling_check.setChecked(False)
    assert editor._collect_yaml()["global"]["constants"]["max_ceiling_minutes"] is None
