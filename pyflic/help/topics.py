"""Help-topic content model — loading, headings, and search.

This module is deliberately **Qt-free** so the topic set can be validated in a
headless test without starting a ``QApplication``.  Rendering lives in
:mod:`pyflic.help.window`.

A *help topic* is one markdown file in ``content/``.  Its id is the filename
stem; its title is the first level-1 heading (falling back to the id).  Any
heading inside a topic is addressable as an *anchor* using a GitHub-style slug,
so ``reference-parameters#feeding_event_link_gap`` resolves to the
``## feeding_event_link_gap`` section.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

CONTENT_DIR = Path(__file__).parent / "content"

# A markdown ATX heading, ignoring those inside fenced code blocks (stripped
# before this is applied).
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$", re.MULTILINE)
_FENCE_RE = re.compile(r"^(```|~~~).*?^\1", re.MULTILINE | re.DOTALL)


def slugify(text: str) -> str:
    """GitHub-compatible anchor slug for a heading.

    Backticks and other inline markup are dropped, spaces become hyphens, and
    the result is lower-cased.  ``"`feeding_event_link_gap`"`` and
    ``"feeding_event_link_gap"`` therefore slug identically, so a call site may
    spell the anchor either way.
    """
    s = text.strip().lower()
    s = re.sub(r"`([^`]*)`", r"\1", s)          # inline code -> its contents
    s = re.sub(r"\*\*([^*]*)\*\*", r"\1", s)    # bold
    s = re.sub(r"\*([^*]*)\*", r"\1", s)        # italic
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)  # links -> label
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s.strip())
    return s.strip("-")


@dataclass(frozen=True)
class Heading:
    level: int
    text: str

    @property
    def slug(self) -> str:
        return slugify(self.text)


@dataclass(frozen=True)
class Topic:
    id: str
    source: str

    @property
    def title(self) -> str:
        for h in self.headings:
            if h.level == 1:
                return h.text
        return self.id.replace("-", " ").capitalize()

    @property
    def headings(self) -> list[Heading]:
        body = _FENCE_RE.sub("", self.source)
        return [
            Heading(len(m.group(1)), m.group(2).strip())
            for m in _HEADING_RE.finditer(body)
        ]

    def has_anchor(self, anchor: str) -> bool:
        want = slugify(anchor)
        return any(h.slug == want for h in self.headings)


def _content_path(topic_id: str) -> Path:
    return CONTENT_DIR / f"{topic_id}.md"


@lru_cache(maxsize=None)
def load(topic_id: str) -> Topic | None:
    """Return the topic, or ``None`` when it does not exist or is unreadable.

    Never raises: a missing or corrupt topic degrades to ``None`` so a help
    button can never take down the app that hosts it.
    """
    # Reject anything that could escape the content directory.
    if not topic_id or "/" in topic_id or "\\" in topic_id or topic_id.startswith("."):
        return None
    path = _content_path(topic_id)
    try:
        if not path.is_file():
            return None
        return Topic(id=topic_id, source=path.read_text(encoding="utf-8"))
    except OSError:
        return None


def available() -> list[str]:
    """Every topic id present on disk, sorted."""
    try:
        return sorted(
            p.stem for p in CONTENT_DIR.glob("*.md") if not p.name.startswith("_")
        )
    except OSError:
        return []


def parse_ref(ref: str) -> tuple[str, str | None]:
    """Split ``"topic-id#anchor"`` into its parts."""
    topic_id, _, anchor = ref.partition("#")
    return topic_id, (anchor or None)


@dataclass(frozen=True)
class SearchHit:
    topic_id: str
    title: str
    heading: str | None
    snippet: str


def search(query: str, *, limit: int = 60) -> list[SearchHit]:
    """Case-insensitive substring search across every topic.

    Hits are attributed to the nearest preceding heading so a result can be
    opened at the right place, and ranked so title matches surface first.
    """
    q = query.strip().lower()
    if len(q) < 2:
        return []

    hits: list[tuple[int, SearchHit]] = []
    for topic_id in available():
        topic = load(topic_id)
        if topic is None:
            continue
        current: str | None = None
        for line in topic.source.splitlines():
            m = _HEADING_RE.match(line)
            if m:
                current = m.group(2).strip()
            if q not in line.lower():
                continue
            text = line.strip().lstrip("#").strip()
            if not text:
                continue
            # Rank: title match beats heading match beats body match.
            if q in topic.title.lower():
                rank = 0
            elif m:
                rank = 1
            else:
                rank = 2
            idx = text.lower().find(q)
            start = max(0, idx - 40)
            snippet = ("…" if start else "") + text[start:start + 160]
            hits.append(
                (rank, SearchHit(topic_id, topic.title, current, snippet))
            )
            if len(hits) >= limit * 3:
                break

    hits.sort(key=lambda pair: pair[0])
    # Collapse duplicates from the same heading of the same topic.
    seen: set[tuple[str, str | None]] = set()
    out: list[SearchHit] = []
    for _, hit in hits:
        key = (hit.topic_id, hit.heading)
        if key in seen:
            continue
        seen.add(key)
        out.append(hit)
        if len(out) >= limit:
            break
    return out
