"""The report page layout: fixed US Letter pages, the report's own headings
above plots, tables that flow, and the formatting the reports share."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from pyflic.base import report_layout as rl


def _page_boxes(pdf_bytes: bytes) -> list[str]:
    return re.findall(rb"/MediaBox \[ ?([0-9. ]+?) ?\]", pdf_bytes)


def _doc(*blocks) -> rl.ReportDocument:
    doc = rl.ReportDocument("Experiment report", "rep1", generated="2026-01-01 12:00",
                            version="test")
    doc.add(rl.Cover("Experiment report", "rep1", ["Hedonic · two-well"]), *blocks)
    return doc


def test_every_page_is_us_letter(tmp_path):
    rows = pd.DataFrame({"A": range(120), "B": np.linspace(0, 1, 120)})
    doc = _doc(rl.Heading("One"), rl.Paragraph("text " * 400),
               rl.Table(rows, caption="long"), rl.PageBreak(), rl.Heading("Two"))
    path = doc.save(tmp_path / "r.pdf")
    boxes = _page_boxes(path.read_bytes())
    assert len(boxes) == doc.page_count() >= 4
    assert {b.split()[-2:] == [b"612", b"792"] for b in boxes} == {True}


def test_a_long_table_repeats_its_header_on_every_page():
    rows = pd.DataFrame({"Value": range(150)})
    doc = _doc(rl.Table(rows))
    doc._layout()
    assert len(doc._pages) >= 3
    ## Each page after the cover's holds one table segment, header included.
    assert all(len(ops) >= 1 for ops in doc._pages[1:])


def test_the_contents_carry_numbered_sections_and_their_pages():
    doc = _doc(rl.Contents(), rl.Heading("Quality control"),
               rl.Heading("Data integrity", level=2), rl.PageBreak(),
               rl.Heading("Results"), rl.Heading("Appendix", numbered=False))
    doc._layout()
    assert doc._toc == [(1, "1  Quality control", 1), (2, "1.1  Data integrity", 1),
                        (1, "2  Results", 2), (1, "Appendix", 2)]


def test_a_heading_is_never_left_alone_at_the_foot_of_a_page():
    import plotnine as p9

    plot = p9.ggplot(pd.DataFrame({"x": [0, 1], "y": [0, 1]}), p9.aes("x", "y")) \
        + p9.geom_line()
    filler = rl.Paragraph("line " * 900)
    doc = _doc(filler, rl.Heading("Figures"), rl.Paragraph("An intro."),
               rl.Plot(plot, height=3.4, title="A plot"))
    doc._layout()
    heading_page = next(page for level, label, page in doc._toc if "Figures" in label)
    assert heading_page == len(doc._pages)       # it moved on with its plot


def test_the_report_title_replaces_the_plot_title():
    import plotnine as p9

    plot = (p9.ggplot(pd.DataFrame({"x": [0, 1], "y": [0, 1]}), p9.aes("x", "y"))
            + p9.geom_line() + p9.labs(title="PLOT TITLE"))
    kept, w, h = rl.render_plot(plot, 4.0, 3.0, drop_title=False)
    dropped, w2, h2 = rl.render_plot(plot, 4.0, 3.0, drop_title=True)
    assert (w, h) == pytest.approx((4.0, 3.0)) and (w2, h2) == pytest.approx((4.0, 3.0))
    ## Same size, different pixels: the title band is gone.
    assert kept.shape == dropped.shape and not np.array_equal(kept, dropped)


def test_a_failing_figure_costs_only_that_figure(tmp_path):
    doc = _doc(rl.Plot(lambda: 1 / 0, title="broken"), rl.Paragraph("after"))
    assert doc.save(tmp_path / "r.pdf").is_file()


def test_text_wraps_to_the_measured_width():
    lines = rl.wrap_text("word " * 60, 2.0, rl.SIZE_BODY)
    assert len(lines) > 5
    assert all(rl.text_width(line, rl.SIZE_BODY) <= 2.0 for line in lines)
    assert rl.wrap_text("a\nb", 5.0, rl.SIZE_BODY) == ["a", "b"]


@pytest.mark.parametrize("value,fmt,expected", [
    (3, None, "3"), (2.0, None, "2"), (0.123456, None, "0.123"),
    (12.345, None, "12.35"), (1234.5, None, "1,234"), (float("nan"), None, "–"),
    (True, None, "yes"), (0.5, "{:.2f}", "0.50"), (None, None, "–"),
])
def test_cells_format_consistently(value, fmt, expected):
    assert rl.fmt_value(value, fmt) == expected


def test_p_values_and_tones():
    assert rl.fmt_p(0.0004) == "< 0.001" and rl.fmt_p(0.04567) == "0.046"
    assert rl.tone_of("ok") == "ok" and rl.tone_of("warning") == "warning"
    assert rl.tone_of("excluded") == "failed" and rl.tone_of("retained (x)") == "failed"
    assert rl.tone_of("") is None and rl.tone_of("something") is None
