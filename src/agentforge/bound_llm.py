"""Bind an OpenAI-compatible client to a tool list for every turn."""

from __future__ import annotations

from typing import Any, Callable

from agentforge.llm import OpenAICompatibleClient
from agentforge.tools import ToolRegistry


class BoundLLM:
    """Callable for Runner: messages -> str | dict (with tool_calls)."""

    def __init__(
        self,
        client: OpenAICompatibleClient,
        tools: ToolRegistry | None = None,
        *,
        tool_choice: str | dict | None = "auto",
        max_tools: int | None = None,
        prefer_native: bool = True,
    ):
        self.client = client
        self.tools = tools
        self.tool_choice = tool_choice
        self.max_tools = max_tools
        self.prefer_native = prefer_native

    def __call__(self, messages: list[dict[str, Any]]) -> dict[str, Any] | str:
        schema = None
        if self.tools is not None:
            schema = self.tools.openai_tools(max_tools=self.max_tools)
        if schema and self.prefer_native:
            return self.client.chat(
                messages,
                tools=schema,
                tool_choice=self.tool_choice,
            )
        return self.client.chat_text(messages)


def bind_llm(
    client: OpenAICompatibleClient,
    tools: ToolRegistry | None = None,
    **kwargs: Any,
) -> BoundLLM:
    return BoundLLM(client, tools, **kwargs)
