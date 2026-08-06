"""Validation for the in-app help topic set.

Topic ids and anchors are plain strings at every call site, so nothing but a
test stops a renamed heading from silently sending a help button to the top of
the wrong page.  These checks resolve every reference — in the table of
contents, in the guides, in the topics' own cross-links, and in the GUI source
— against the content files on disk.

Deliberately Qt-free: ``pyflic.help.topics`` and ``pyflic.help.toc`` import no
Qt, so this runs headless and fast.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pyflic.help import topics as _topics
from pyflic.help.toc import GUIDES, TOC, all_topic_ids

# GUI modules that may reference help topics.
_GUI_ROOT = Path(__file__).resolve().parent.parent / "pyflic"

# HelpButton("ref"), install_help_shortcut(x, "ref"), open_help("ref")
_CALL_SITE_RE = re.compile(
    r"""(?:HelpButton|install_help_shortcut|open_help)\(
        [^)"']*                       # leading args (widget, self, …)
        ["'](?P<ref>[a-z0-9\-]+(?:\#[^"']+)?)["']
    """,
    re.VERBOSE,
)

#: Parameter keys that GUI controls turn into ``reference-parameters#<key>``
#: at runtime.  Sourced from the widgets themselves so the test tracks the
#: code rather than a copy of it.
def _gui_parameter_keys() -> dict[str, list[str]]:
    from pyflic.base.config_editor import _PARAM_ORDER
    from pyflic.base.qc_viewer import ParamsTab

    return {
        "config_editor._PARAM_ORDER": list(_PARAM_ORDER),
        "qc_viewer.ParamsTab._PARAM_FIELDS": [f[0] for f in ParamsTab._PARAM_FIELDS],
    }

# Markdown links to a sibling topic: [text](topic-id.md#anchor)
_MD_LINK_RE = re.compile(r"\]\(\s*(?P<target>[A-Za-z0-9\-_]+\.md)(?:\#(?P<anchor>[^)\s]+))?\s*\)")

# Same-page anchor links: [text](#anchor)
_MD_SELF_LINK_RE = re.compile(r"\]\(\s*\#(?P<anchor>[^)\s]+)\s*\)")


def _iter_gui_sources():
    for path in sorted(_GUI_ROOT.rglob("*.py")):
        if "help" in path.parts and path.parent.name == "help":
            continue          # the help package's own docstring examples
        yield path


# ---------------------------------------------------------------------------
# The topic set itself
# ---------------------------------------------------------------------------

def test_every_topic_file_is_in_the_toc():
    on_disk = set(_topics.available())
    in_toc = all_topic_ids()
    assert len(in_toc) == len(set(in_toc)), "a topic appears twice in the TOC"
    orphans = on_disk - set(in_toc)
    assert not orphans, f"topic files missing from the TOC: {sorted(orphans)}"


def test_every_toc_entry_exists_on_disk():
    missing = [tid for tid in all_topic_ids() if _topics.load(tid) is None]
    assert not missing, f"TOC lists topics with no content file: {missing}"


def test_every_topic_has_a_title():
    for topic_id in all_topic_ids():
        topic = _topics.load(topic_id)
        assert topic is not None
        levels = [h.level for h in topic.headings]
        assert levels.count(1) == 1, (
            f"{topic_id}: expected exactly one level-1 heading, found {levels.count(1)}"
        )
        assert levels[0] == 1, f"{topic_id}: first heading must be the level-1 title"


def test_guides_reference_real_topics():
    for guide in GUIDES:
        assert guide.topic_ids, f"guide {guide.id!r} is empty"
        for topic_id in guide.topic_ids:
            assert _topics.load(topic_id) is not None, (
                f"guide {guide.id!r} references missing topic {topic_id!r}"
            )


def test_guide_ids_are_unique():
    ids = [g.id for g in GUIDES]
    assert len(ids) == len(set(ids))


def test_section_titles_are_unique():
    titles = [s.title for s in TOC]
    assert len(titles) == len(set(titles))


# ---------------------------------------------------------------------------
# Cross-links inside the content
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("topic_id", all_topic_ids())
def test_internal_links_resolve(topic_id):
    """Every [text](other-topic.md#anchor) points at a real topic and heading."""
    topic = _topics.load(topic_id)
    assert topic is not None
    problems: list[str] = []

    for m in _MD_LINK_RE.finditer(topic.source):
        target_id = m.group("target")[:-3]      # strip .md
        target = _topics.load(target_id)
        if target is None:
            problems.append(f"link to unknown topic {target_id!r}")
            continue
        anchor = m.group("anchor")
        if anchor and not target.has_anchor(anchor):
            problems.append(f"link to {target_id}#{anchor} — no such heading")

    for m in _MD_SELF_LINK_RE.finditer(topic.source):
        anchor = m.group("anchor")
        if not topic.has_anchor(anchor):
            problems.append(f"same-page link #{anchor} — no such heading")

    assert not problems, f"{topic_id}:\n  " + "\n  ".join(problems)


def test_anchor_slugs_are_unique_within_a_topic():
    for topic_id in all_topic_ids():
        topic = _topics.load(topic_id)
        assert topic is not None
        slugs = [h.slug for h in topic.headings]
        dupes = {s for s in slugs if slugs.count(s) > 1}
        assert not dupes, f"{topic_id}: duplicate heading slugs {sorted(dupes)}"


# ---------------------------------------------------------------------------
# References from the GUI
# ---------------------------------------------------------------------------

def test_gui_help_references_resolve():
    """Every topic id and anchor named in GUI code exists."""
    problems: list[str] = []
    seen = 0

    for path in _iter_gui_sources():
        source = path.read_text(encoding="utf-8")
        for m in _CALL_SITE_RE.finditer(source):
            ref = m.group("ref")
            if "{" in ref:
                # An f-string builds this ref at runtime; the key lists behind
                # them are checked by the dedicated tests below instead.
                continue
            seen += 1
            topic_id, anchor = _topics.parse_ref(ref)
            topic = _topics.load(topic_id)
            rel = path.relative_to(_GUI_ROOT.parent)
            if topic is None:
                problems.append(f"{rel}: unknown topic {topic_id!r}")
                continue
            if anchor and not topic.has_anchor(anchor):
                problems.append(f"{rel}: {topic_id}#{anchor} — no such heading")

    assert not problems, "dangling help references:\n  " + "\n  ".join(problems)
    assert seen > 0, "found no help call sites — has the regex drifted from the code?"


def test_every_gui_parameter_has_a_reference_heading():
    """Each parameter control's ``?`` must land on a real heading.

    These refs are built at runtime from the widgets' key lists, so the static
    scan above cannot see them.  Checking the lists directly is stronger: it
    fails if a parameter is added to a form without being documented.
    """
    pytest.importorskip("PyQt6")
    reference = _topics.load("reference-parameters")
    assert reference is not None

    problems: list[str] = []
    for source, keys in _gui_parameter_keys().items():
        assert keys, f"{source} is empty — has it moved?"
        for key in keys:
            if not reference.has_anchor(key):
                problems.append(f"{source}: no '## {key}' in reference-parameters.md")

    assert not problems, "undocumented parameters:\n  " + "\n  ".join(problems)


# ---------------------------------------------------------------------------
# Behaviour of the loader
# ---------------------------------------------------------------------------

def test_missing_topic_returns_none_rather_than_raising():
    assert _topics.load("no-such-topic") is None
    assert _topics.load("") is None


@pytest.mark.parametrize("evil", ["../secrets", "..\\secrets", "sub/dir", ".hidden"])
def test_loader_refuses_paths_outside_the_content_dir(evil):
    assert _topics.load(evil) is None


def test_slugify_ignores_inline_code_and_case():
    assert _topics.slugify("`feeding_event_link_gap`") == _topics.slugify(
        "feeding_event_link_gap"
    )
    assert _topics.slugify("Event linking (the link gap)") == "event-linking-the-link-gap"


def test_search_finds_a_known_term():
    hits = _topics.search("link gap")
    assert hits, "expected search to find the link gap"
    assert any(h.topic_id in ("concepts-licks-events", "reference-parameters") for h in hits)
