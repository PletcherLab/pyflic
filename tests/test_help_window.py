"""Behavioural tests for the help window.

These cover three defects found in review that a reference-resolution test
cannot see, because each depends on Qt document state at runtime:

* a queued anchor scroll applying to a topic the reader has already left
* heading formatting bleeding onto adjacent paragraphs and giving ordinary
  blocks a non-zero ``headingLevel()``
* help buttons keeping the previous theme's colour after a live toggle
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from pyflic.help import topics as _topics  # noqa: E402
from pyflic.help.toc import all_topic_ids  # noqa: E402


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    from pyflic.base.ui import apply_theme

    apply_theme(application, mode="light")
    yield application


@pytest.fixture
def window(app):
    from pyflic.help.window import HelpWindow

    win = HelpWindow()
    win.resize(900, 700)
    yield win
    win.close()


# ---------------------------------------------------------------------------
# Heading formatting must not corrupt block metadata
# ---------------------------------------------------------------------------

def _heading_blocks(window):
    doc = window._view.document()
    out = []
    block = doc.begin()
    while block.isValid():
        level = block.blockFormat().headingLevel()
        if level:
            out.append((level, block.text().strip()))
        block = block.next()
    return out


@pytest.mark.parametrize("topic_id", all_topic_ids())
def test_styling_does_not_invent_headings(window, topic_id):
    """The set of heading blocks must match the markdown's own headings.

    ``BlockUnderCursor`` spans the preceding block separator, which propagated
    the heading block format onto the paragraph above and left ordinary text
    reporting a heading level — corrupting exactly the metadata anchors match
    against.
    """
    topic = _topics.load(topic_id)
    assert topic is not None

    window.show_topic(topic_id)
    rendered = _heading_blocks(window)

    expected = [h.text for h in topic.headings]
    got = [text for _level, text in rendered]

    assert len(got) == len(expected), (
        f"{topic_id}: markdown has {len(expected)} headings, "
        f"document reports {len(got)}"
    )
    # Qt strips inline markup, so compare on slugs rather than raw text.
    assert [_topics.slugify(t) for t in got] == [
        _topics.slugify(t) for t in expected
    ]


def test_heading_levels_survive_styling(window):
    window.show_topic("reference-parameters")
    levels = [lvl for lvl, _ in _heading_blocks(window)]
    assert levels[0] == 1, "the title should still be a level-1 heading"
    assert 2 in levels, "parameter sections should still be level-2"


# ---------------------------------------------------------------------------
# Anchor navigation
# ---------------------------------------------------------------------------

def test_anchor_scrolls_to_the_requested_heading(window, app):
    window.show()
    window.show_topic("reference-parameters", "feeding_event_link_gap")
    app.processEvents()
    assert window._scroll_to_heading("feeding_event_link_gap")
    assert window._view.verticalScrollBar().value() > 0


def test_queued_scroll_does_not_apply_to_a_later_topic(window, app):
    """Navigating away before the deferred scroll fires must cancel it.

    ``scripts-actions`` and ``plots-catalog`` both contain a "Sliding-window
    plots" heading, so a stale scroll lands somewhere plausible and silently
    wrong.
    """
    window.show()
    app.processEvents()

    window.show_topic("scripts-actions", "sliding-window-plots")
    # Navigate away *before* the queued scroll runs.
    window.show_topic("plots-catalog")
    window._view.verticalScrollBar().setValue(0)
    app.processEvents()
    app.processEvents()

    assert window.current_topic_id() == "plots-catalog"
    assert window._view.verticalScrollBar().value() == 0, (
        "a stale queued scroll moved the newly-opened topic"
    )


def test_unknown_anchor_leaves_the_view_at_the_top(window, app):
    window.show()
    window.show_topic("concepts-metrics", "no-such-heading")
    app.processEvents()
    assert window._view.verticalScrollBar().value() == 0


# ---------------------------------------------------------------------------
# Missing content degrades
# ---------------------------------------------------------------------------

def test_missing_topic_renders_a_message(window):
    window.show_topic("definitely-not-a-topic")
    assert "not available" in window._view.toPlainText().lower()


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

def test_help_button_retints_on_theme_change(app):
    from pyflic.base.ui import apply_theme
    from pyflic.base.ui.icons import help_color, help_hover_background
    from pyflic.help import HelpButton

    apply_theme(app, mode="light")
    btn = HelpButton("getting-started")
    ## The resting tint is muted; the amber accent survives as the hover wash.
    light = help_hover_background()
    assert light in btn.styleSheet()
    assert help_color(muted=True) != help_color(), "resting tint should be quieter"

    apply_theme(app, mode="dark")
    app.processEvents()
    dark = help_hover_background()
    assert dark != light, "the two themes should use different amber"
    assert dark in btn.styleSheet(), "help button kept the previous theme's colour"

    apply_theme(app, mode="light")
