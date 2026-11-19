"""The Output log's line colouring and spacing (``ui/textformat.py``)."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyflic.base.ui.textformat import log_line_to_html  # noqa: E402

RED = "#d9534f"


@pytest.mark.parametrize("line", [
    ## The integrity report on a clean DFM: the firmware's column is named
    ## Error, and nothing in these lines is a failure.
    "## Error column details",
    "Found error-like column(s): Error",
    "Column: Error",
    "  Error-type counts (among flagged rows):",
])
def test_the_error_column_is_not_an_error(line):
    assert RED not in log_line_to_html(line)


@pytest.mark.parametrize("line", [
    "ERROR: config.yaml could not be read",
    "[load] ERROR loading member 'Rep1'",
    "FAILED to load: no data/ folder",
    "Error: not a directory: C:/data",
    "ValueError: Unknown experiment_type 'X'",
])
def test_a_failure_is_still_red(line):
    assert RED in log_line_to_html(line)


# ---------------------------------------------------------------------------
# Spacing: the log is fixed-width, so every line keeps its layout
# ---------------------------------------------------------------------------

def test_an_indented_line_keeps_its_indentation():
    assert log_line_to_html("  Done → analysis/feeding_summary.csv").startswith(
        "&nbsp;&nbsp;Done")


def test_a_prefixed_line_keeps_its_muted_prefix_and_its_spacing():
    out = log_line_to_html("[rep1] running analysis...")
    assert out == '<span style="color:#8a8a8a">[rep1]</span> running analysis...'
    assert "[rep1]&nbsp;&nbsp;&nbsp;step two" in log_line_to_html("[rep1]   step two")


def test_table_rows_keep_their_columns_and_shorten_decimals_in_place():
    header = log_line_to_html("Chamber  Licks      Events")
    row = log_line_to_html("1        12.3456789 4")
    assert header == "Chamber&nbsp;&nbsp;Licks&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Events"
    ## 12.3456789 is ten characters; 12.3457 is padded back to ten.
    assert row == ("1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
                   "12.3457&nbsp;&nbsp;&nbsp;&nbsp;4")


def test_tabs_expand_to_terminal_stops():
    assert log_line_to_html("DFM\tChamber") == "DFM" + "&nbsp;" * 5 + "Chamber"


def test_single_spaces_in_prose_stay_breakable():
    assert log_line_to_html("Analysis complete for every member") == (
        "Analysis complete for every member")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_the_log_font_is_fixed_width(qapp):
    from PyQt6.QtGui import QFontDatabase, QFontInfo

    from pyflic.base.ui.fonts import fixed_family
    from pyflic.base.ui.widgets import OutputLog

    assert QFontDatabase.isFixedPitch(fixed_family())
    log = OutputLog()
    assert QFontInfo(log.font()).fixedPitch()
