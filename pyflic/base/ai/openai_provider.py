"""The OpenAI summarizer."""

from __future__ import annotations

import base64
import os

from .base import AISummarizer, AISummaryError, SummaryPayload

_TIMEOUT_SECONDS = 300.0
_MAX_RETRIES = 3
_MAX_COMPLETION_TOKENS = 4096
_LIST_TIMEOUT_SECONDS = 10.0

#: Model-id prefixes known to accept image input.  OpenAI's listing includes
#: embeddings, audio, and image-generation models, none of which can read a
#: figure — offering one would produce a confusing failure at call time.
_VISION_PREFIXES = ("gpt-4o", "gpt-4.1", "gpt-5", "o1", "o3", "o4")


def _filter_model_ids(ids: list[str]) -> list[str]:
    return sorted({m for m in ids if m.startswith(_VISION_PREFIXES)})


class OpenAISummarizer(AISummarizer):
    provider_name = "openai"
    display_name = "OpenAI"
    env_var = "OPENAI_API_KEY"
    models = ("gpt-5.2", "gpt-4o")

    @classmethod
    def fetch_models(cls) -> list[str]:
        import openai

        from .base import load_api_env

        load_api_env()
        api_key = os.environ.get(cls.env_var, "").strip()
        if not api_key:
            raise AISummaryError(
                f"{cls.display_name}: no API key found. Set {cls.env_var} "
                f"in the environment or a .env file.")
        client = openai.OpenAI(api_key=api_key,
                               timeout=_LIST_TIMEOUT_SECONDS, max_retries=0)
        try:
            ids = [model.id for model in client.models.list()]
        except openai.OpenAIError as err:
            raise AISummaryError(
                f"{cls.display_name}: could not fetch the model list "
                f"({err.__class__.__name__}).") from err
        return _filter_model_ids(ids)

    def _request(self, instructions: str, payload: SummaryPayload) -> str:
        import openai

        client = openai.OpenAI(api_key=self.api_key,
                               timeout=_TIMEOUT_SECONDS,
                               max_retries=_MAX_RETRIES)
        content: list[dict] = []
        for title, png in payload.images:
            content.append({"type": "text", "text": f"Figure: {title}"})
            data = base64.standard_b64encode(png).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{data}"},
            })
        content.append({"type": "text", "text": payload.text})

        try:
            response = client.chat.completions.create(
                model=self.model,
                max_completion_tokens=_MAX_COMPLETION_TOKENS,
                messages=[{"role": "system", "content": instructions},
                          {"role": "user", "content": content}])
        except openai.AuthenticationError as err:
            raise AISummaryError(
                f"{self.display_name}: the API key was rejected. Check "
                f"{self.env_var} in your .env file.") from err
        except openai.NotFoundError as err:
            raise AISummaryError(
                f"{self.display_name}: model '{self.model}' was not "
                f"recognised by the API.") from err
        except openai.RateLimitError as err:
            raise AISummaryError(
                f"{self.display_name}: rate-limited even after retries — try "
                f"again in a minute.") from err
        except openai.APIStatusError as err:
            raise AISummaryError(
                f"{self.display_name}: API error {err.status_code}.") from err
        except openai.APIConnectionError as err:
            raise AISummaryError(
                f"{self.display_name}: could not reach the API — check the "
                f"network connection.") from err

        return response.choices[0].message.content or ""
