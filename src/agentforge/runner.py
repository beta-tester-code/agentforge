"""Basic agent runner (skeleton)."""

from __future__ import annotations

from agentforge.agent import Agent
from agentforge.memory import Memory
from agentforge.observability import Tracer


class Runner:
    """Executes an agent step by step.

    This is intentionally minimal in the first commit.
    Compaction, tool calling, and LLM integration will be added next.
    """

    def __init__(self, agent: Agent):
        self.agent = agent
        self.memory = Memory()
        self.tracer = Tracer()
        self.step = 0

    def run(self, user_input: str, max_steps: int = 10) -> str:
        """Placeholder run loop.

        Real implementation will:
        - call the model
        - handle tool calls
        - apply compaction when needed
        - record token usage per step
        """
        self.memory.add("user", user_input)
        self.tracer.record(self.step, "user_input")
        self.step += 1

        reply = f"[AgentForge skeleton] Received: {user_input!r}. Full execution coming next."
        self.memory.add("assistant", reply)
        self.tracer.record(self.step, "assistant_reply")
        self.step += 1

        return reply
