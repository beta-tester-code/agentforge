"""Tool registration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agentforge.schema import openai_tool_from_callable, schema_from_callable


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., Any]
    parameters: dict[str, Any] = field(default_factory=dict)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    def openai_schema(self) -> dict[str, Any]:
        params = self.parameters or schema_from_callable(self.func)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": params,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        parameters: dict[str, Any] | None = None,
    ) -> None:
        self._tools[name] = Tool(
            name=name,
            description=description,
            func=func,
            parameters=parameters or schema_from_callable(func),
        )

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def descriptions(self, max_chars: int | None = None) -> str:
        if not self._tools:
            return "(no tools registered)"
        lines: list[str] = []
        omitted = 0
        used = 0
        for t in self._tools.values():
            line = f"- {t.name}: {t.description}"
            if max_chars is not None and used + len(line) + 1 > max_chars and lines:
                omitted += 1
                continue
            lines.append(line)
            used += len(line) + 1
        text = "\n".join(lines)
        if omitted:
            text += f"\n- … +{omitted} tools omitted (schema budget); call by exact name if known"
        return text

    def openai_tools(self, max_tools: int | None = None) -> list[dict[str, Any]]:
        tools = self.list()
        if max_tools is not None:
            tools = tools[:max_tools]
        return [t.openai_schema() for t in tools]
