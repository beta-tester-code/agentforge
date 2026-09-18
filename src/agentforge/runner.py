"""Agent runner with token tracking, compaction and optional LLM."""

from __future__ import annotations

from typing import Any, Callable

from agentforge.agent import Agent
from agentforge.compaction import CompactionConfig, compact_memory
from agentforge.memory import Memory
from agentforge.observability import Tracer
from agentforge.tokens import get_token_counter


class Runner:
    """Executes an agent step by step with observability and compaction."""

    def __init__(
        self,
        agent: Agent,
        *,
        compaction_config: CompactionConfig | None = None,
        token_counter: Callable[[str], int] | None = None,
        llm_call: Callable[[list[dict[str, str]]], str] | None = None,
        model_name: str = "gpt-4o-mini",
    ):
        self.agent = agent
        self.memory = Memory()
        self.tracer = Tracer()
        self.step = 0
        self.compaction_config = compaction_config or CompactionConfig()
        self.token_counter = token_counter or get_token_counter(model_name)
        self.llm_call = llm_call

    def _count(self, text: str) -> int:
        return self.token_counter(text)

    def _maybe_compact(self) -> None:
        result = compact_memory(
            self.memory,
            self.compaction_config,
            token_counter=self.token_counter,
            summarizer=None,
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
        """Run one turn (user -> assistant).

        Multi-step tool loops will be added next.
        """
        user_tokens = self._count(user_input)
        self.memory.add("user", user_input, tokens=user_tokens)
        self.tracer.record(self.step, "user_input", input_tokens=user_tokens)
        self.step += 1

        system = self.agent.system_prompt
        if self.agent.goal:
            system = f"{system}\n\nGoal: {self.agent.goal}"

        if self.llm_call is not None:
            messages_for_llm: list[dict[str, str]] = [
                {"role": "system", "content": system}
            ]
            for m in self.memory.messages:
                role = m.role if m.role in ("user", "assistant", "system") else "user"
                messages_for_llm.append({"role": role, "content": m.content})

            reply = self.llm_call(messages_for_llm)
            reply_tokens = self._count(reply)
        else:
            reply = (
                f"[AgentForge] Received: {user_input!r}. "
                f"No llm_call provided — skeleton response."
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
