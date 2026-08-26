"""AI-assisted functions — today, the AI Summary.

A deliberately small provider layer: :class:`AISummarizer` subclasses talk to
the official Anthropic / OpenAI SDKs directly.  No agent frameworks, no tool
use, no streaming — one request in (report text + figure images), one page of
prose out.

An AI Summary is a **derivative of a single analysis run**: re-running the
analysis deletes it, and the report embeds it only while the saved file exists.
That rule is what keeps a stale narrative from sitting beside fresh numbers.
"""

from __future__ import annotations

import os

from .anthropic_provider import AnthropicSummarizer
from .base import AISummarizer, AISummaryError, SummaryPayload, load_api_env
from .openai_provider import OpenAISummarizer
from .payload import PROJECT_INSTRUCTIONS, build_project_payload

#: Registration order is presentation order in the UI.
PROVIDERS: tuple[type[AISummarizer], ...] = (
    AnthropicSummarizer,
    OpenAISummarizer,
)

#: Where a Project's narrative is saved, relative to its analysis directory.
NARRATIVE_FILENAME = "ai_summary.txt"

#: The Batch narrative, written at the Batch root.  Named distinctly from a
#: Project's own so a recursive glob can tell the levels apart.
BATCH_NARRATIVE_FILENAME = "batch_ai_narrative.md"

__all__ = [
    "AISummaryError", "AISummarizer", "AnthropicSummarizer",
    "OpenAISummarizer", "PROVIDERS", "SummaryPayload",
    "available_providers", "build_project_payload", "get_summarizer",
    "load_api_env", "generate_project_narrative", "read_project_narrative",
    "delete_project_narrative", "NARRATIVE_FILENAME",
    "BATCH_NARRATIVE_FILENAME", "generate_batch_narrative",
]


def available_providers() -> list[type[AISummarizer]]:
    """The providers whose API key is present — presence-gated, not validated."""
    return [provider for provider in PROVIDERS if provider.is_available()]


def get_summarizer(provider_name: str, model: str | None = None) -> AISummarizer:
    """A ready summarizer for *provider_name*, or raise :class:`AISummaryError`."""
    for provider in PROVIDERS:
        if provider.provider_name == provider_name:
            return provider(model=model)
    known = ", ".join(p.provider_name for p in PROVIDERS)
    raise AISummaryError(
        f"Unknown AI provider '{provider_name}'. Known providers: {known}.")


def narrative_path(project) -> str:
    return os.path.join(project.analysis_path, NARRATIVE_FILENAME)


def generate_project_narrative(project, provider: str = "anthropic",
                               model: str | None = None) -> str:
    """Write an AI narrative of *project*'s Combined Analysis and return its path."""
    summarizer = get_summarizer(provider, model)
    payload = build_project_payload(project)
    text = summarizer.summarize(payload, PROJECT_INSTRUCTIONS)
    os.makedirs(project.analysis_path, exist_ok=True)
    path = narrative_path(project)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n\n"
                     f"— written by {summarizer.display_name} "
                     f"({summarizer.model}) from this report's own content.\n")
    return path


def read_project_narrative(project) -> str | None:
    """The saved narrative, or ``None`` when there is none."""
    path = narrative_path(project)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return None


def delete_project_narrative(project) -> bool:
    """Remove the saved narrative.

    Called whenever the Combined Analysis is rebuilt: the narrative describes
    one run's numbers, so it must not outlive them.
    """
    path = narrative_path(project)
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def generate_batch_narrative(batch_dir, provider: str,
                             model: str | None = None,
                             project_keys: list[str] | None = None,
                             log=print) -> str | None:
    """Synthesize the Projects' own narratives into one at the Batch root.

    A *synthesis*, not a pooling: each Project keeps its own design and its own
    numbers, and the batch narrative says where they agree and where they do
    not.  Returns the written path, or ``None`` when there was nothing to
    synthesize.

    Never raises for a Project it cannot read.  This runs at the end of an
    unattended Batch Run, and a narrative that fails must not turn a successful
    overnight batch into a failed one.
    """
    from .. import batch as batch_mod
    from ..project import Project

    root = os.path.abspath(str(batch_dir))
    keys = (list(project_keys) if project_keys is not None
            else batch_mod.batch_project_names(root))
    sections: list[str] = []
    for key in keys:
        try:
            saved = read_project_narrative(
                Project(batch_mod.project_directory(root, key)))
        except Exception:  # noqa: BLE001
            saved = None
        if saved:
            sections.append(f"## {key}\n\n{saved.strip()}")
        else:
            log(f"[ai] {key} has no narrative to synthesize — skipped")
    if not sections:
        log("[ai] no Project narratives in this batch — nothing to synthesize")
        return None

    summarizer = get_summarizer(provider, model)
    payload = SummaryPayload(text="\n\n".join(sections))
    text = summarizer.summarize(payload, batch_mod.BATCH_NARRATIVE_PROMPT)
    path = os.path.join(root, BATCH_NARRATIVE_FILENAME)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n\n"
                     f"— written by {summarizer.display_name} "
                     f"({summarizer.model}) from {len(sections)} Project "
                     "narrative(s); no new analysis was performed.\n")
    log(f"[ai] wrote {path}")
    return path
