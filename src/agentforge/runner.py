"""Agent runner with token tracking, compaction, LLM and tool loop."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from agentforge.agent import Agent
from agentforge.compaction import (
    CompactionConfig,
    compact_memory,
    make_llm_summarizer,
)
from agentforge.memory import Memory
from agentforge.observability import Tracer
from agentforge.pressure import pressure_level
from agentforge.result import RunResult
from agentforge.tokens import get_token_counter
from agentforge.tools import ToolRegistry


TOOL_CALL_RE = re.compile(
    r"TOOL_CALL\s*\n\s*name:\s*(?P<name>\S+)\s*\n\s*args:\s*(?P<args>\{.*\})\s*\n\s*END_TOOL_CALL",
    re.DOTALL | re.IGNORECASE,
)

TOOL_JSON_RE = re.compile(
    r"```(?:json)?\s*(\{\s*\"name\"\s*:.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)


def _parse_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    m = TOOL_CALL_RE.search(text)
    if m:
        name = m.group("name").strip()
        raw_args = m.group("args").strip()
        return name, _loads_obj(raw_args)

    m2 = TOOL_JSON_RE.search(text)
    if m2:
        try:
            obj = json.loads(m2.group(1))
        except json.JSONDecodeError:
            return None
        if isinstance(obj, dict) and "name" in obj:
            args = obj.get("args") or obj.get("arguments") or {}
            if not isinstance(args, dict):
                args = {}
            return str(obj["name"]), args
    return None


def _loads_obj(raw: str) -> dict[str, Any]:
    try:
        args = json.loads(raw)
        return args if isinstance(args, dict) else {}
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                args = json.loads(raw[start : end + 1])
                return args if isinstance(args, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}


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
        use_llm_summarizer: bool = True,
        reanchor_goal: bool = True,
        enable_offload_recall: bool = True,
        tool_schema_max_chars: int = 2_500,
        max_tool_errors: int = 3,
    ):
        self.agent = agent
        self.tools = tools or ToolRegistry()
        self.memory = Memory()
        self.tracer = Tracer()
        self.step = 0
        self.compaction_config = compaction_config or CompactionConfig()
        self.token_counter = token_counter or get_token_counter(model_name)
        self.llm_call = llm_call
        self.reanchor_goal = reanchor_goal
        self.tool_schema_max_chars = tool_schema_max_chars
        self.max_tool_errors = max_tool_errors
        self._system_prompt_cache: str | None = None
        self.last_result: RunResult | None = None

        self._summarizer: Callable | None = None
        if use_llm_summarizer and llm_call is not None:
            self._summarizer = make_llm_summarizer(llm_call)

        for name, func in agent.tools.items():
            if self.tools.get(name) is None:
                self.tools.register(name, description=f"Tool '{name}'", func=func)

        if enable_offload_recall and self.tools.get("recall_offload") is None:

            def recall_offload(key: str) -> str:
                store = self.memory.state.get("_offload_store") or {}
                if key not in store:
                    return f"Error: unknown offload key {key!r}"
                return store[key]

            self.tools.register(
                "recall_offload",
                'Retrieve full offloaded tool content. args: {"key": "tool_result_..."}',
                recall_offload,
            )

    def reset(self) -> None:
        """Clear memory, tracer and step counter for a fresh run."""
        self.memory = Memory()
        self.tracer = Tracer()
        self.step = 0
        self.last_result = None
        # keep system prompt cache — agent/tools unchanged

    def _count(self, text: str) -> int:
        return self.token_counter(text)

    def _system_prompt(self) -> str:
        if self._system_prompt_cache is not None:
            return self._system_prompt_cache

        parts = [self.agent.system_prompt]
        if self.agent.goal:
            parts.append(f"Goal: {self.agent.goal}")

        tool_desc = self.tools.descriptions(max_chars=self.tool_schema_max_chars)
        if tool_desc and tool_desc != "(no tools registered)":
            parts.append(
                "You can call tools using this exact format:\n"
                "TOOL_CALL\n"
                "name: <tool_name>\n"
                "args: {\"key\": \"value\"}\n"
                "END_TOOL_CALL\n\n"
                "Alternatively, a single fenced JSON block:\n"
                '```json\n{"name": "<tool_name>", "args": {"key": "value"}}\n```\n\n'
                "Available tools:\n"
                f"{tool_desc}\n\n"
                "After a tool result, continue reasoning. "
                "Final answers must not include a tool call."
            )
        self._system_prompt_cache = "\n\n".join(parts)
        return self._system_prompt_cache

    def invalidate_prompt_cache(self) -> None:
        self._system_prompt_cache = None

    def _maybe_compact(self) -> None:
        before_len = len(self.memory.messages)
        tokens_now = self.memory.token_total()
        level = pressure_level(tokens_now, self.compaction_config.max_tokens)
        if level.name == "warn":
            self.tracer.record(
                self.step,
                "pressure_warn",
                input_tokens=tokens_now,
                level=level.name,
                ratio=round(level.ratio, 3),
            )

        result = compact_memory(
            self.memory,
            self.compaction_config,
            token_counter=self.token_counter,
            summarizer=self._summarizer,
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
                pressure=level.name,
            )
            if (
                self.reanchor_goal
                and self.agent.goal
                and result.messages_after < before_len
            ):
                anchor = f"[goal reminder] {self.agent.goal}"
                self.memory.add(
                    "system",
                    anchor,
                    tokens=self._count(anchor),
                    type="goal_reanchor",
                )

    def _build_messages(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt()}
        ]
        for m in self.memory.messages:
            role = m.role if m.role in ("user", "assistant", "system", "tool") else "user"
            if role == "tool":
                content = f"[tool result]\n{m.content}"
                role = "user"
            else:
                content = m.content
            messages.append({"role": role, "content": content})
        return messages

    def _estimate_prompt_tokens(self, messages: list[dict[str, str]]) -> int:
        return sum(self._count(m.get("content", "")) for m in messages)

    def _execute_tool(self, name: str, args: dict[str, Any]) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'"
        try:
            return str(tool.func(**args))
        except TypeError as e:
            return f"Error executing tool '{name}' (bad args): {e}"
        except Exception as e:
            return f"Error executing tool '{name}': {e}"

    def _build_result(self, reply: str, stopped_reason: str) -> RunResult:
        summary = self.tracer.summary()
        by = summary.get("by_action") or {}
        result = RunResult(
            reply=reply,
            steps=summary.get("steps", 0),
            input_tokens=summary.get("input_tokens", 0),
            output_tokens=summary.get("output_tokens", 0),
            total_tokens=summary.get("total_tokens", 0),
            tool_calls=int(by.get("tool_call", 0)),
            compactions=int(by.get("compaction", 0)),
            stopped_reason=stopped_reason,
            trace=self.tracer.steps(),
            by_action=dict(by),
        )
        self.last_result = result
        return result

    def run(self, user_input: str, max_steps: int = 8) -> str:
        """Run until final answer or stop condition. Returns reply text.

        Full metrics: ``runner.last_result`` or ``run_detailed(...)``.
        """
        return self.run_detailed(user_input, max_steps=max_steps).reply

    def run_detailed(self, user_input: str, max_steps: int = 8) -> RunResult:
        user_tokens = self._count(user_input)
        self.memory.add("user", user_input, tokens=user_tokens)
        self.tracer.record(self.step, "user_input", input_tokens=user_tokens)
        self.step += 1

        final_reply = ""
        stopped = "completed"
        tool_error_streak = 0

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
                stopped = "skeleton"
                break

            messages = self._build_messages()
            prompt_tokens = self._estimate_prompt_tokens(messages)
            reply = self.llm_call(messages)
            reply_tokens = self._count(reply)

            parsed = _parse_tool_call(reply)
            if parsed is None:
                self.memory.add("assistant", reply, tokens=reply_tokens)
                self.tracer.record(
                    self.step,
                    "assistant_reply",
                    input_tokens=prompt_tokens,
                    output_tokens=reply_tokens,
                )
                self.step += 1
                final_reply = reply
                stopped = "completed"
                break

            tool_name, tool_args = parsed
            self.memory.add("assistant", reply, tokens=reply_tokens)
            self.tracer.record(
                self.step,
                "tool_call",
                input_tokens=prompt_tokens,
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
                is_error=tool_result.startswith("Error"),
            )
            self.step += 1

            if tool_result.startswith("Error"):
                tool_error_streak += 1
                if tool_error_streak >= self.max_tool_errors:
                    final_reply = (
                        f"[AgentForge] stopped after {tool_error_streak} consecutive tool errors. "
                        f"Last: {tool_result[:200]}"
                    )
                    stopped = "error_streak"
                    break
            else:
                tool_error_streak = 0

            self._maybe_compact()
        else:
            final_reply = final_reply or "[AgentForge] max_steps reached without final answer."
            stopped = "max_steps"

        self._maybe_compact()
        return self._build_result(final_reply, stopped)

    def get_trace_summary(self) -> dict[str, Any]:
        return self.tracer.summary()

    def get_trace(self) -> list[dict[str, Any]]:
        return self.tracer.steps()
