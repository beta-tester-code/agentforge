"""Simple memory / context management primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Message:
    role: str
    content: str
    tokens: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "tokens": self.tokens,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        return cls(
            role=str(data.get("role", "user")),
            content=str(data.get("content", "")),
            tokens=data.get("tokens"),
            meta=dict(data.get("meta") or {}),
        )


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

    def token_total(self) -> int:
        return sum(m.tokens or 0 for m in self.messages)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for checkpoints / hosted traces later."""
        # Offload store can be large; keep it under state as-is.
        return {
            "messages": [m.to_dict() for m in self.messages],
            "state": dict(self.state),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Memory:
        mem = cls()
        mem.messages = [Message.from_dict(m) for m in data.get("messages") or []]
        mem.state = dict(data.get("state") or {})
        return mem
