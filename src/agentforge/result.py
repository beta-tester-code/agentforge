"""Structured result of a Runner.run call."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunResult:
    """Everything you need after a run without digging into Runner internals."""

    reply: str
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    tool_calls: int = 0
    compactions: int = 0
    stopped_reason: str = "completed"  # completed | max_steps | skeleton | error_streak
    trace: list[dict[str, Any]] = field(default_factory=list)
    by_action: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reply": self.reply,
            "steps": self.steps,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "tool_calls": self.tool_calls,
            "compactions": self.compactions,
            "stopped_reason": self.stopped_reason,
            "by_action": dict(self.by_action),
            "trace": list(self.trace),
        }
