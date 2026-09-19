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

class _compact_mixin:
    def _maybe_compact(self, force: bool = False) -> None:
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
            force_summarize=force or self._force_compact,
            put_fn=self._put_offload,
        )
        self._force_compact = False
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
            if self.reanchor_goal and self.agent.goal and result.messages_after < before_len:
                anchor = f"[goal reminder] {self.agent.goal}"
                self.memory.add(
                    "system", anchor, tokens=self._count(anchor), type="goal_reanchor"
                )

    def _build_messages(self) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt()}
        ]
        for m in self.memory.messages:
            role = m.role if m.role in ("user", "assistant", "system", "tool") else "user"
            if role == "tool":
                if self.use_native_tools:
                    item: dict[str, Any] = {
                        "role": "tool",
                        "content": m.content,
                    }
                    if m.meta.get("tool_call_id"):
                        item["tool_call_id"] = m.meta["tool_call_id"]
                    if m.meta.get("tool"):
                        item["name"] = m.meta["tool"]
                    messages.append(item)
                else:
                    messages.append(
                        {"role": "user", "content": f"[tool result]\n{m.content}"}
                    )
            else:
                messages.append({"role": role, "content": m.content})
        return messages

    def _estimate_prompt_tokens(self, messages: list[dict[str, Any]]) -> int:
        return sum(self._count(str(m.get("content", ""))) for m in messages)

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
        if self._trace_store is not None:
            self._trace_store.append(
                result,
                meta={"agent": self.agent.name, "goal": self.agent.goal},
            )
        return result
