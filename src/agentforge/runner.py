"""Agent runner with token tracking and compaction hooks."""

from __future__ import annotations

from typing import Any, Callable

from agentforge.agent import Agent
from agentforge.compaction import CompactionConfig, compact_memory, estimate_tokens
from agentforge.memory import Memory, Message
from agentforge.observability import Tracer


class Runner:
    """Executes an agent step by step with basic observability and compaction."""

    def __init__(
        self,
        agent: Agent,
        *,
        compaction_config: CompactionConfig | None = None,
        token_counter: Callable[[str], int] | None = None,
        llm_call: Callable[..., str] | None = None,
    ):
        self.agent = agent
        self.memory = Memory()
        self.tracer = Tracer()
        self.step = 0
        self.compaction_config = compaction_config or CompactionConfig()
        self.token_counter = token_counter or estimate_tokens
        self.llm_call = llm_call  # optional: (messages) -> response text

    def _count(self, text: str) -> int:
        return self.token_counter(text)

    def _maybe_compact(self) -> None:
        result = compact_memory(
            self.memory,
            self.compaction_config,
            token_counter=self.token_counter,
            summarizer=None,  # real summarizer comes later
        )
        if result.actions:
            self.tracer.record(
                self.step,
                "compaction",
                input_tokens=result.tokens_before,
                output_tokens=result.tokens_after,
                actions=result.actions,
                messages_before=result.messages_before,
                messages_after=result.messages_after,
            )

    def run(self, user_input: str, max_steps: int = 8) -> str:
        """Run the agent for up to max_steps.

        Current behaviour:
        - Records user input with token count
        - Optionally calls an injected llm_call
        - Applies compaction if over threshold
        - Returns the final assistant reply
        """
        user_tokens = self._count(user_input)
        self.memory.add("user", user_input, tokens=user_tokens)
        self.tracer.record(self.step, "user_input", input_tokens=user_tokens)
        self.step += 1

        # Build simple prompt context
        system = self.agent.system_prompt
        if self.agent.goal:
            system = f"{system}\n\nGoal: {self.agent.goal}"

        if self.llm_call is not None:
            # Real path: call the provided LLM function
            messages_for_llm = [{"role": "system", "content": system}]
            for m in self.memory.messages:
                messages_for_llm.append({"role": m.role, "content": m.content})

            reply = self.llm_call(messages_for_llm)
            reply_tokens = self._count(reply)
        else:
            # Skeleton path (no LLM wired yet)
            reply = (
                f"[AgentForge] Received: {user_input!r}. "
                f"LLM not wired yet — this is the skeleton response."
            )
            reply_tokens = self._count(reply)

        self.memory.add("assistant", reply, tokens=reply_tokens)
        self.tracer.record(
            self.step,
            "assistant_reply",
            output_tokens=reply_tokens,
        )
        self.step += 1

        self._maybe_compact()

        return reply

    def get_trace_summary(self) -> dict[str, Any]:
        return self.tracer.summary()
