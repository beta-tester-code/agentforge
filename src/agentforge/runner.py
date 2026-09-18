"""Agent runner with token tracking, compaction, LLM and tool loop."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from agentforge.agent import Agent
from agentforge.compaction import CompactionConfig, compact_memory
from agentforge.memory import Memory
from agentforge.observability import Tracer
from agentforge.tokens import get_token_counter
from agentforge.tools import ToolRegistry


# Very small, robust tool-call format we ask the model to use.
# Example:
# TOOL_CALL
# name: search
# args: {"query": "agent frameworks 2026"}
# END_TOOL_CALL

TOOL_CALL_RE = re.compile(
    r"TOOL_CALL\s*\n\s*name:\s*(?P<name>\S+)\s*\n\s*args:\s*(?P<args>\{.*?\})\s*\n\s*END_TOOL_CALL",
    re.DOTALL | re.IGNORECASE,
)


def _parse_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    m = TOOL_CALL_RE.search(text)
    if not m:
        return None
    name = m.group("name").strip()
    raw_args = m.group("args").strip()
    try:
        args = json.loads(raw_args)
        if not isinstance(args, dict):
            args = {}
    except json.JSONDecodeError:
        args = {}
    return name, args


class Runner:
    """Executes an agent with optional tools, observability and compaction."""

    def __init__(
        self,
        agent: Agent,
        *,
        tools: ToolRegistry | None = None,
        compaction_config: CompactionConfig | None = None,
        token_counter: Callable[[str], int] | None = None,
        llm_call: Callable[[list[dict[str, str]]], str] | None = None,
        model_name: str = "gpt-4o-mini",
    ):
        self.agent = agent
        self.tools = tools or ToolRegistry()
        self.memory = Memory()
        self.tracer = Tracer()
        self.step = 0
        self.compaction_config = compaction_config or CompactionConfig()
        self.token_counter = token_counter or get_token_counter(model_name)
        self.llm_call = llm_call

        # Also register tools that were added directly on the Agent
        for name, func in agent.tools.items():
            if self.tools.get(name) is None:
                self.tools.register(name, description=f"Tool '{name}'", func=func)

    def _count(self, text: str) -> int:
        return self.token_counter(text)

    def _system_prompt(self) -> str:
        parts = [self.agent.system_prompt]
        if self.agent.goal:
            parts.append(f"Goal: {self.agent.goal}")

        tool_desc = self.tools.descriptions()
        if tool_desc and tool_desc != "(no tools registered)":
            parts.append(
                "You can call tools using this exact format:\n"
                "TOOL_CALL\n"
                "name: <tool_name>\n"
                "args: {\"key\": \"value\"}\n"
                "END_TOOL_CALL\n\n"
                "Available tools:\n"
                f"{tool_desc}\n\n"
                "After receiving a tool result, continue reasoning. "
                "When you have the final answer, reply normally without TOOL_CALL."
            )
        return "\n\n".join(parts)

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

    def _build_messages(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt()}
        ]
        for m in self.memory.messages:
            role = m.role if m.role in ("user", "assistant", "system", "tool") else "user"
            # Some providers don't like role=tool; map to user with a prefix
            if role == "tool":
                content = f"[tool result]\n{m.content}"
                role = "user"
            else:
                content = m.content
            messages.append({"role": role, "content": content})
        return messages

    def _execute_tool(self, name: str, args: dict[str, Any]) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'"
        try:
            result = tool.func(**args)
            return str(result)
        except Exception as e:
            return f"Error executing tool '{name}': {e}"

    def run(self, user_input: str, max_steps: int = 8) -> str:
        """Run the agent until it produces a final answer or hits max_steps."""
        user_tokens = self._count(user_input)
        self.memory.add("user", user_input, tokens=user_tokens)
        self.tracer.record(self.step, "user_input", input_tokens=user_tokens)
        self.step += 1

        final_reply = ""

        for _ in range(max_steps):
            if self.llm_call is None:
                final_reply = (
                    f"[AgentForge] Received: {user_input!r}. "
                    f"No llm_call provided — skeleton response."
                )
                reply_tokens = self._count(final_reply)
                self.memory.add("assistant", final_reply, tokens=reply_tokens)
                self.tracer.record(self.step, "assistant_reply", output_tokens=reply_tokens)
                self.step += 1
                break

            messages = self._build_messages()
            reply = self.llm_call(messages)
            reply_tokens = self._count(reply)

            parsed = _parse_tool_call(reply)
            if parsed is None:
                # Final answer
                self.memory.add("assistant", reply, tokens=reply_tokens)
                self.tracer.record(
                    self.step,
                    "assistant_reply",
                    output_tokens=reply_tokens,
                )
                self.step += 1
                final_reply = reply
                break

            # Tool call path
            tool_name, tool_args = parsed
            self.memory.add("assistant", reply, tokens=reply_tokens)
            self.tracer.record(
                self.step,
                "tool_call",
                output_tokens=reply_tokens,
                tool=tool_name,
                args=tool_args,
            )
            self.step += 1

            tool_result = self._execute_tool(tool_name, tool_args)
            tool_tokens = self._count(tool_result)
            self.memory.add("tool", tool_result, tokens=tool_tokens, tool=tool_name)
            self.tracer.record(
                self.step,
                "tool_result",
                input_tokens=tool_tokens,
                tool=tool_name,
            )
            self.step += 1

            self._maybe_compact()
        else:
            final_reply = final_reply or "[AgentForge] max_steps reached without final answer."

        self._maybe_compact()
        return final_reply

    def get_trace_summary(self) -> dict[str, Any]:
        return self.tracer.summary()
