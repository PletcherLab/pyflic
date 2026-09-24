"""Ordered table of contents and reading guides for the help topics.

Order here is editorial, not alphabetical — orientation material must come
before reference material — so it is declared explicitly rather than derived
from filenames.  ``tests/test_help_refs.py`` asserts that every topic on disk
appears here exactly once, so a new topic cannot be silently orphaned.

A *guide* is an ordered reading path across topics for someone learning a whole
subject rather than answering one question.  Guides carry no prose of their own:
a guide is a title and a sequence of topic ids, nothing more.  The same topic
may appear in several guides.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Section:
    title: str
    topic_ids: tuple[str, ...]


@dataclass(frozen=True)
class Guide:
    """An ordered reading path over existing topics.

    ``summary`` is a one-line label shown as a tooltip in the topic tree.  It is
    metadata, not content — no explanation belongs here that is not in a topic.
    """

    id: str
    title: str
    summary: str
    topic_ids: tuple[str, ...]


TOC: tuple[Section, ...] = (
    Section(
        "Getting started",
        (
            "getting-started",
            "getting-started-project",
            "getting-started-config",
            "getting-started-first-run",
        ),
    ),
    Section(
        "Concepts",
        (
            "concepts-signal",
            "concepts-licks-events",
            "concepts-tasting",
            "concepts-two-well-pi",
            "concepts-metrics",
            "concepts-light-phase",
            "concepts-experiment-types",
            "concepts-project",
            "concepts-facets",
            "concepts-exclusions",
            "concepts-ai-summary",
        ),
    ),
    Section(
        "Configuration",
        (
            "config-structure",
            "config-dfms-chambers",
            "config-factors",
            "reference-parameters",
        ),
    ),
    Section(
        "Scripts",
        (
            "scripts-overview",
            "scripts-actions",
            "scripts-editor",
            "scripts-batch",
        ),
    ),
    Section(
        "Applications",
        (
            "app-hub",
            "app-plot-editor",
            "app-qc-viewer",
            "app-config-editor",
        ),
    ),
    Section(
        "Reference",
        (
            "plots-catalog",
            "reports",
            "python-api",
            "performance",
            "troubleshooting",
            "install",
        ),
    ),
)


GUIDES: tuple[Guide, ...] = (
    Guide(
        "first-experiment",
        "Your first experiment",
        "From an empty folder to your first plot",
        (
            "getting-started",
            "getting-started-project",
            "getting-started-config",
            "getting-started-first-run",
        ),
    ),
    Guide(
        "understanding-detection",
        "How feeding is detected",
        "What licks and events are, and how the parameters change them",
        (
            "concepts-signal",
            "concepts-licks-events",
            "concepts-tasting",
            "concepts-metrics",
            "reference-parameters",
        ),
    ),
    Guide(
        "choice-assays",
        "Choice and hedonic assays",
        "Two-well designs, preference index, and the plots that go with them",
        (
            "concepts-experiment-types",
            "concepts-two-well-pi",
            "config-dfms-chambers",
            "plots-catalog",
        ),
    ),
    Guide(
        "automating",
        "Automating your analysis",
        "Scripts, the script editor, and running many projects at once",
        (
            "scripts-overview",
            "scripts-actions",
            "scripts-editor",
            "scripts-batch",
            "concepts-exclusions",
        ),
    ),
)


def guide(guide_id: str) -> Guide | None:
    return next((g for g in GUIDES if g.id == guide_id), None)


def guide_neighbours(guide_id: str, topic_id: str) -> tuple[str | None, str | None]:
    """Previous and next topic *within a guide's* reading order."""
    g = guide(guide_id)
    if g is None:
        return None, None
    try:
        i = g.topic_ids.index(topic_id)
    except ValueError:
        return None, None
    prev_id = g.topic_ids[i - 1] if i > 0 else None
    next_id = g.topic_ids[i + 1] if i + 1 < len(g.topic_ids) else None
    return prev_id, next_id


def all_topic_ids() -> list[str]:
    """Every topic id in the table of contents, in reading order."""
    return [tid for section in TOC for tid in section.topic_ids]


def section_for(topic_id: str) -> Section | None:
    for section in TOC:
        if topic_id in section.topic_ids:
            return section
    return None


def neighbours(topic_id: str) -> tuple[str | None, str | None]:
    """The previous and next topic in reading order, spanning sections."""
    ordered = all_topic_ids()
    try:
        i = ordered.index(topic_id)
    except ValueError:
        return None, None
    prev_id = ordered[i - 1] if i > 0 else None
    next_id = ordered[i + 1] if i + 1 < len(ordered) else None
    return prev_id, next_id
