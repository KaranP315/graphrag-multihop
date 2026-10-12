from __future__ import annotations

import json
import logging
import time

from openai import OpenAI

from config import LLM_MAX_TOKENS, LLM_MODEL, LLM_TEMPERATURE, OPENAI_API_KEY, OPENAI_BASE_URL

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # seconds


class LLMClient:
    """Wrapper around the OpenAI chat completions API with retry logic."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model or LLM_MODEL
        self.client = OpenAI(
            api_key=api_key or OPENAI_API_KEY,
            base_url=base_url or OPENAI_BASE_URL,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
    ) -> str:
        """Generate a text response for the given prompt."""
        return self._call(prompt, system_prompt, response_format=None)

    def generate_json(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant. Respond with valid JSON only.",
    ) -> dict:
        """Generate a response and parse it as JSON.

        Uses the API's json_object response format to guarantee valid JSON output.
        """
        raw = self._call(
            prompt,
            system_prompt,
            response_format={"type": "json_object"},
        )
        return json.loads(raw)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _call(
        self,
        prompt: str,
        system_prompt: str,
        response_format: dict | None,
    ) -> str:
        """Make an API call with simple exponential-backoff retry."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        last_exc: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                kwargs: dict = dict(
                    model=self.model,
                    messages=messages,
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_TOKENS,
                )
                if response_format is not None:
                    kwargs["response_format"] = response_format

                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content

            except Exception as exc:
                last_exc = exc
                wait = INITIAL_BACKOFF * (2 ** (attempt - 1))
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt, MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"LLM call failed after {MAX_RETRIES} attempts"
        ) from last_exc
