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
def _read(topic_id: str, _stamp: tuple[int, int]) -> Topic | None:
    """Read and parse one topic.  ``_stamp`` participates in the cache key."""
    try:
        return Topic(
            id=topic_id,
            source=_content_path(topic_id).read_text(encoding="utf-8"),
        )
    except OSError:
        return None


def load(topic_id: str) -> Topic | None:
    """Return the topic, or ``None`` when it does not exist or is unreadable.

    Never raises: a missing or corrupt topic degrades to ``None`` so a help
    button can never take down the app that hosts it.

    Results are cached against the file's size and modification time, so an
    edited topic is picked up without restarting — the shipped content never
    changes at runtime, but authoring one with the app open is routine.
    """
    # Reject anything that could escape the content directory.
    if not topic_id or "/" in topic_id or "\\" in topic_id or topic_id.startswith("."):
        return None
    try:
        stat = _content_path(topic_id).stat()
    except OSError:
        return None
    return _read(topic_id, (stat.st_size, stat.st_mtime_ns))


def clear_cache() -> None:
    """Drop cached topic content.  Intended for tests and authoring tools."""
    _read.cache_clear()


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


#: Ranks, lowest first.  A whole-word match always beats a mid-word one, so a
#: search for "pi" ranks the preference-index topics above "python-api".
_RANK_TITLE_WORD, _RANK_HEADING_WORD, _RANK_BODY_WORD = 0, 1, 2
_RANK_TITLE_SUB, _RANK_HEADING_SUB, _RANK_BODY_SUB = 3, 4, 5

#: Most hits to return from any single topic.  Without a cap, one long topic
#: that mentions a common word fills the whole result list.
_MAX_HITS_PER_TOPIC = 3


def _snippet(text: str, q: str) -> str:
    idx = text.lower().find(q)
    start = max(0, idx - 40)
    return ("…" if start else "") + text[start:start + 160]


def search(query: str, *, limit: int = 60) -> list[SearchHit]:
    """Case-insensitive search across every topic.

    Hits are attributed to the nearest preceding heading so a result opens at
    the right place.  Whole-word matches outrank mid-word ones, headings
    outrank body text, and no single topic may dominate the results.
    """
    q = query.strip().lower()
    if len(q) < 2:
        return []
    word_re = re.compile(rf"\b{re.escape(q)}", re.IGNORECASE)

    ranked: list[tuple[int, SearchHit]] = []

    for topic_id in available():
        topic = load(topic_id)
        if topic is None:
            continue

        per_topic: list[tuple[int, SearchHit]] = []

        # The title contributes at most one hit — not one per line, which
        # previously let a title match promote a whole topic's worth of lines
        # above everything else.
        if q in topic.title.lower():
            rank = _RANK_TITLE_WORD if word_re.search(topic.title) else _RANK_TITLE_SUB
            per_topic.append(
                (rank, SearchHit(topic_id, topic.title, None, topic.title))
            )

        current: str | None = None
        for line in topic.source.splitlines():
            heading = _HEADING_RE.match(line)
            if heading:
                current = heading.group(2).strip()
            if q not in line.lower():
                continue
            text = line.strip().lstrip("#").strip()
            if not text:
                continue
            is_word = bool(word_re.search(text))
            if heading:
                rank = _RANK_HEADING_WORD if is_word else _RANK_HEADING_SUB
            else:
                rank = _RANK_BODY_WORD if is_word else _RANK_BODY_SUB
            per_topic.append(
                (rank, SearchHit(topic_id, topic.title, current, _snippet(text, q)))
            )

        # Keep only this topic's strongest hits, one per heading.
        per_topic.sort(key=lambda pair: pair[0])
        seen_headings: set[str | None] = set()
        kept = 0
        for rank, hit in per_topic:
            if hit.heading in seen_headings:
                continue
            seen_headings.add(hit.heading)
            ranked.append((rank, hit))
            kept += 1
            if kept >= _MAX_HITS_PER_TOPIC:
                break

    ranked.sort(key=lambda pair: pair[0])
    return [hit for _, hit in ranked[:limit]]
