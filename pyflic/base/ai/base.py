"""Provider-agnostic core of the AI Summary feature.

An :class:`AISummarizer` turns a :class:`SummaryPayload` (the report's own text
and figures) into one page of prose via an external provider.  This module owns
what the concrete providers share — API-key discovery, availability checks, the
summarize contract — so the Anthropic and OpenAI subclasses stay thin request
builders.

Design rules:

* No provider SDK is imported at module level — the package must import, and
  the availability check must run, on a machine with neither SDK installed.
* Keys come from the environment, optionally topped up from ``.env`` files.
  Presence of a key makes a provider *available*; validity is only discovered
  at call time, because a failed generation must never block a report.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class AISummaryError(RuntimeError):
    """A summary could not be produced (missing key, provider error, refusal).

    Carries a user-facing message; callers surface it and move on.
    """


@dataclass
class SummaryPayload:
    """What a provider gets to see: serialized report text + figure PNGs."""

    text: str = ""
    #: ``(title, png_bytes)`` per figure, in report order.
    images: list[tuple[str, bytes]] = field(default_factory=list)


_env_loaded = False


def load_api_env() -> None:
    """Load API keys from ``.env`` files into the environment, once.

    Looks in the per-user config directory and then walks up from the current
    directory, so both an installed app and a repo checkout find their keys.
    Real environment variables always win — nothing is overridden.
    """
    global _env_loaded
    if _env_loaded:
        return
    _env_loaded = True
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return
    config_env = os.path.join(
        os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")),
        "pyflic", ".env")
    if os.path.isfile(config_env):
        load_dotenv(config_env, override=False)
    found = find_dotenv(usecwd=True)
    if found:
        load_dotenv(found, override=False)


class AISummarizer(ABC):
    """One provider's way of writing an AI Summary."""

    #: Stable key used in code, settings, and the registry (e.g. "anthropic").
    provider_name: str = ""
    #: Human label for dialogs and the summary's provenance line.
    display_name: str = ""
    #: Environment variable holding the API key.
    env_var: str = ""
    #: Curated, vision-capable model ids offered in the UI.  First = default.
    models: tuple[str, ...] = ()

    @classmethod
    def default_model(cls) -> str:
        return cls.models[0]

    @classmethod
    def fetch_models(cls) -> list[str]:
        """This provider's current vision-capable model ids.  The base default
        is the curated list, so a provider without a listing endpoint still
        refreshes cleanly."""
        return list(cls.models)

    @classmethod
    def is_available(cls) -> bool:
        """True when this provider's API key is present — not validated."""
        load_api_env()
        return bool(os.environ.get(cls.env_var, "").strip())

    def __init__(self, model: str | None = None) -> None:
        load_api_env()
        self.api_key = os.environ.get(self.env_var, "").strip()
        if not self.api_key:
            raise AISummaryError(
                f"{self.display_name}: no API key found. Set {self.env_var} "
                f"in the environment or a .env file.")
        self.model = model or self.default_model()

    def summarize(self, payload: SummaryPayload, instructions: str) -> str:
        """The summary prose for *payload*, or raise :class:`AISummaryError`."""
        text = (self._request(instructions, payload) or "").strip()
        if not text:
            raise AISummaryError(
                f"{self.display_name} ({self.model}) returned an empty summary.")
        return text

    @abstractmethod
    def _request(self, instructions: str, payload: SummaryPayload) -> str:
        """Perform one provider call and return the raw summary text.

        Implementations translate their SDK's failures into
        :class:`AISummaryError` with a message a scientist can act on.
        """
