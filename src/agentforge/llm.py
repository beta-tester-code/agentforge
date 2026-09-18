"""Minimal OpenAI-compatible LLM client."""

from __future__ import annotations

import os
from typing import Any

import httpx


class LLMError(Exception):
    pass


class OpenAICompatibleClient:
    """Thin client for any OpenAI-compatible chat completions API.

    Works with OpenAI, Groq, Together, Fireworks, local vLLM, etc.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout: float = 60.0,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

        if not self.api_key:
            raise LLMError(
                "No API key found. Set OPENAI_API_KEY or LLM_API_KEY, "
                "or pass api_key=..."
            )

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Send chat messages and return the assistant content."""
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.3),
        }
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            )

        if resp.status_code >= 400:
            raise LLMError(f"LLM API error {resp.status_code}: {resp.text}")

        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Unexpected response shape: {data}") from e


def make_llm_call(
    api_key: str | None = None,
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
) -> Any:
    """Return a callable suitable for Runner(llm_call=...)."""
    client = OpenAICompatibleClient(api_key=api_key, base_url=base_url, model=model)

    def _call(messages: list[dict[str, str]]) -> str:
        return client.chat(messages)

    return _call
