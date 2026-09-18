"""Core Agent definition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Agent:
    """Minimal agent definition.

    Attributes:
        name: Human-readable name.
        goal: High-level objective the agent should pursue.
        system_prompt: Base instructions given to the model.
        tools: Mapping of tool name -> callable.
    """

    name: str
    goal: str
    system_prompt: str
    tools: dict[str, Callable[..., Any]] = field(default_factory=dict)

    def add_tool(self, name: str, func: Callable[..., Any]) -> None:
        """Register a tool the agent can call."""
        self.tools[name] = func
