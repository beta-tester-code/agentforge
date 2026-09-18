"""Tool registration and simple execution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)


class ToolRegistry:
    """Simple registry of tools available to an agent."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, description: str, func: Callable[..., Any]) -> None:
        self._tools[name] = Tool(name=name, description=description, func=func)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def descriptions(self, max_chars: int | None = None) -> str:
        """Compact text description of tools (for prompts).

        If max_chars is set, truncate the block and note omitted tools.
        GitHub reported multi-KB schema tax from unused tools — we bound it.
        """
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
