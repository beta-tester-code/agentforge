"""Simple memory / context management primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    role: str
    content: str
    tokens: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Memory:
    """In-memory conversation + state store with room for compaction hooks."""

    messages: list[Message] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)

    def add(self, role: str, content: str, tokens: int | None = None, **meta: Any) -> None:
        self.messages.append(Message(role=role, content=content, tokens=tokens, meta=meta))

    def clear(self) -> None:
        self.messages.clear()

    def recent(self, n: int) -> list[Message]:
        return self.messages[-n:] if n > 0 else []
