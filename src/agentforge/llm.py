"""Minimal OpenAI-compatible LLM client."""

from __future__ import annotations

import os
from typing import Any

import httpx


class LLMError(Exception):
    pass


class OpenAICompatibleClient:
    """Thin client for any OpenAI-compatible chat completions API."""

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

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send chat messages; return full message dict (content + optional tool_calls)."""
        payload: dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.3),
        }
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

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
            return data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Unexpected response shape: {data}") from e

    def chat_text(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        msg = self.chat(messages, **kwargs)
        content = msg.get("content") or ""
        return content if isinstance(content, str) else str(content)


def make_llm_call(
    api_key: str | None = None,
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
) -> Any:
    """Return a callable (messages) -> str for Runner(llm_call=...)."""
    client = OpenAICompatibleClient(api_key=api_key, base_url=base_url, model=model)

    def _call(messages: list[dict[str, str]]) -> str:
        return client.chat_text(messages)

    return _call


def tools_to_openai_schema(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert simple (name, description) tools to OpenAI function schema stubs.

    Args are left open (empty object) unless the description embeds JSON hints.
    This is intentional: AgentForge prioritizes text TOOL_CALL format for
    provider portability; native schema is optional acceleration.
    """
    out: list[dict[str, Any]] = []
    for t in tools:
        name = getattr(t, "name", None)
        desc = getattr(t, "description", "") or ""
        if not name:
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": True,
                    },
                },
            }
        )
    return out
