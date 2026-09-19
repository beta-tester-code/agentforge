"""Agent runner mixins."""
from __future__ import annotations
from typing import Any, Callable
import json
from agentforge.compaction import compact_memory, make_llm_summarizer
from agentforge.memory import Memory
from agentforge.observability import Tracer
from agentforge.offload_store import DiskOffloadStore, MemoryOffloadStore
from agentforge.pressure import pressure_level
from agentforge.result import RunResult
from agentforge.tokens import get_token_counter
from agentforge.tool_parse import _native_tool_calls, _parse_all_tool_calls
from agentforge.tools import ToolRegistry
from agentforge.trace_store import TraceStore
from agentforge.agent import Agent
from agentforge.compaction import CompactionConfig

class _run_mixin:
    def run(self, user_input: str, max_steps: int = 8) -> str:
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
            if self._cancelled():
                final_reply = "[AgentForge] cancelled"
                stopped = "cancelled"
                break

            if self.hard_token_budget is not None:
                spent = self.tracer.summary().get("total_tokens", 0)
                if spent >= self.hard_token_budget:
                    final_reply = (
                        f"[AgentForge] hard token budget reached ({spent}/"
                        f"{self.hard_token_budget})."
                    )
                    stopped = "budget"
                    break

            if self.llm_call is None:
                final_reply = (
                    f"[AgentForge] Received: {user_input!r}. "
                    f"No llm_call provided — skeleton response."
                )
                reply_tokens = self._count(final_reply)
                self.memory.add("assistant", final_reply, tokens=reply_tokens)
                self.tracer.record(
                    self.step, "assistant_reply", output_tokens=reply_tokens
                )
                self.step += 1
                stopped = "skeleton"
                break

            messages = self._build_messages()
            prompt_tokens = self._estimate_prompt_tokens(messages)

            raw = self.llm_call(messages)
            native_with_ids: list[tuple[str, dict[str, Any], str | None]] = []
            if isinstance(raw, dict):
                content = str(raw.get("content") or "")
                native_with_ids = _native_tool_calls(raw)
                calls = [(n, a) for n, a, _ in native_with_ids]
                if not calls and content:
                    calls = _parse_all_tool_calls(content)
                    native_with_ids = [(n, a, None) for n, a in calls]
                reply_for_mem = content or json.dumps(
                    raw.get("tool_calls") or [], default=str
                )
            else:
                content = str(raw)
                calls = _parse_all_tool_calls(content)
                native_with_ids = [(n, a, None) for n, a in calls]
                reply_for_mem = content

            reply_tokens = self._count(reply_for_mem)

            if not native_with_ids:
                self.memory.add("assistant", reply_for_mem, tokens=reply_tokens)
                self.tracer.record(
                    self.step,
                    "assistant_reply",
                    input_tokens=prompt_tokens,
                    output_tokens=reply_tokens,
                )
                self.step += 1
                final_reply = content or reply_for_mem
                stopped = "completed"
                break

            self.memory.add("assistant", reply_for_mem, tokens=reply_tokens)
            self.tracer.record(
                self.step,
                "tool_call",
                input_tokens=prompt_tokens,
                output_tokens=reply_tokens,
                tools=[c[0] for c in native_with_ids],
            )
            self.step += 1

            any_error = False
            cancelled = False
            for tool_name, tool_args, tool_call_id in native_with_ids:
                if self._cancelled():
                    final_reply = "[AgentForge] cancelled"
                    stopped = "cancelled"
                    cancelled = True
                    break
                tool_result = self._execute_tool(tool_name, tool_args)
                tool_tokens = self._count(tool_result)
                meta: dict[str, Any] = {"tool": tool_name}
                if tool_call_id:
                    meta["tool_call_id"] = tool_call_id
                self.memory.add(
                    "tool", tool_result, tokens=tool_tokens, **meta
                )
                self.tracer.record(
                    self.step,
                    "tool_result",
                    input_tokens=tool_tokens,
                    tool=tool_name,
                    is_error=tool_result.startswith("Error"),
                )
                self.step += 1
                if tool_result.startswith("Error"):
                    any_error = True

            if cancelled:
                break

            if any_error:
                tool_error_streak += 1
                if tool_error_streak >= self.max_tool_errors:
                    final_reply = (
                        f"[AgentForge] stopped after {tool_error_streak} consecutive "
                        f"tool-error turns."
                    )
                    stopped = "error_streak"
                    break
            else:
                tool_error_streak = 0

            self._maybe_compact()
        else:
            final_reply = final_reply or (
                "[AgentForge] max_steps reached without final answer."
            )
            stopped = "max_steps"

        self._maybe_compact()
        return self._build_result(final_reply, stopped)

    def get_trace_summary(self) -> dict[str, Any]:
        return self.tracer.summary()

    def get_trace(self) -> list[dict[str, Any]]:
        return self.tracer.steps()
