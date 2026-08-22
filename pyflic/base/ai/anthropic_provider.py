"""The Anthropic (Claude) summarizer."""

from __future__ import annotations

import base64
import os

from .base import AISummarizer, AISummaryError, SummaryPayload

# One summary is a single, possibly image-heavy request; give it room beyond
# the SDK's defaults but never let a hung call sit for the 10-minute default.
_TIMEOUT_SECONDS = 300.0
_MAX_RETRIES = 3      # SDK-side backoff for 429/5xx/connection errors.
_MAX_TOKENS = 4096    # A one-page summary; far below this in practice.
_LIST_TIMEOUT_SECONDS = 10.0


def _filter_model_ids(ids: list[str]) -> list[str]:
    """Anthropic's list is Claude-only and every current Claude model accepts
    image input, so only guard against non-model ids appearing."""
    return [model_id for model_id in ids if model_id.startswith("claude")]


class AnthropicSummarizer(AISummarizer):
    provider_name = "anthropic"
    display_name = "Anthropic (Claude)"
    env_var = "ANTHROPIC_API_KEY"
    models = ("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5")

    @classmethod
    def fetch_models(cls) -> list[str]:
        import anthropic

        from .base import load_api_env

        load_api_env()
        api_key = os.environ.get(cls.env_var, "").strip()
        if not api_key:
            raise AISummaryError(
                f"{cls.display_name}: no API key found. Set {cls.env_var} "
                f"in the environment or a .env file.")
        client = anthropic.Anthropic(api_key=api_key,
                                     timeout=_LIST_TIMEOUT_SECONDS,
                                     max_retries=0)
        try:
            ids = [model.id for model in client.models.list()]
        except anthropic.APIError as err:
            raise AISummaryError(
                f"{cls.display_name}: could not fetch the model list "
                f"({err.__class__.__name__}).") from err
        return _filter_model_ids(ids)

    def _request(self, instructions: str, payload: SummaryPayload) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key,
                                     timeout=_TIMEOUT_SECONDS,
                                     max_retries=_MAX_RETRIES)
        content: list[dict] = []
        for title, png in payload.images:
            content.append({"type": "text", "text": f"Figure: {title}"})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.standard_b64encode(png).decode("ascii"),
                },
            })
        content.append({"type": "text", "text": payload.text})

        try:
            response = client.messages.create(
                model=self.model, max_tokens=_MAX_TOKENS, system=instructions,
                messages=[{"role": "user", "content": content}])
        except anthropic.AuthenticationError as err:
            raise AISummaryError(
                f"{self.display_name}: the API key was rejected. Check "
                f"{self.env_var} in your .env file.") from err
        except anthropic.NotFoundError as err:
            raise AISummaryError(
                f"{self.display_name}: model '{self.model}' was not "
                f"recognised by the API.") from err
        except anthropic.RateLimitError as err:
            raise AISummaryError(
                f"{self.display_name}: rate-limited even after retries — try "
                f"again in a minute.") from err
        except anthropic.APIStatusError as err:
            raise AISummaryError(
                f"{self.display_name}: API error {err.status_code}: "
                f"{err.message}") from err
        except anthropic.APIConnectionError as err:
            raise AISummaryError(
                f"{self.display_name}: could not reach the API — check the "
                f"network connection.") from err

        if getattr(response, "stop_reason", None) == "refusal":
            raise AISummaryError(
                f"{self.display_name} declined to summarize this content.")
        return "".join(block.text for block in response.content
                       if block.type == "text")
