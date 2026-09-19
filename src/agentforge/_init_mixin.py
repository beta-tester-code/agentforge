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

class _init_mixin:
    def __init__(
        self,
        agent: Agent,
        *,
        tools: ToolRegistry | None = None,
        compaction_config: CompactionConfig | None = None,
        token_counter: Callable[[str], int] | None = None,
        llm_call: Callable[..., Any] | None = None,
        model_name: str = "gpt-4o-mini",
        use_llm_summarizer: bool = True,
        reanchor_goal: bool = True,
        enable_offload_recall: bool = True,
        enable_compact_now: bool = True,
        tool_schema_max_chars: int = 2_500,
        max_tool_errors: int = 3,
        use_native_tools: bool = False,
        offload_dir: str | None = None,
        cancel_check: Callable[[], bool] | None = None,
        trace_path: str | None = None,
        hard_token_budget: int | None = None,
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
        self.use_native_tools = use_native_tools
        self.cancel_check = cancel_check
        self.hard_token_budget = hard_token_budget
        self._system_prompt_cache: str | None = None
        self.last_result: RunResult | None = None
        self._force_compact = False
        self._trace_store = TraceStore(trace_path) if trace_path else None

        if offload_dir:
            self._disk = DiskOffloadStore(offload_dir)
            self._mem_store = MemoryOffloadStore()
        else:
            self._disk = None
            self._mem_store = MemoryOffloadStore()

        self._summarizer: Callable | None = None
        if use_llm_summarizer and llm_call is not None:

            def _text_call(messages: list[dict[str, str]]) -> str:
                out = llm_call(messages)
                if isinstance(out, dict):
                    return str(out.get("content") or "")
                return str(out)

            self._summarizer = make_llm_summarizer(_text_call)

        for name, func in agent.tools.items():
            if self.tools.get(name) is None:
                self.tools.register(name, description=f"Tool '{name}'", func=func)

        if enable_offload_recall and self.tools.get("recall_offload") is None:

            def recall_offload(key: str) -> str:
                if self._disk is not None:
                    val = self._disk.get(key)
                    if val is not None:
                        return val
                store = self.memory.state.get("_offload_store") or {}
                if key in store:
                    return store[key]
                val = self._mem_store.get(key)
                if val is not None:
                    return val
                return f"Error: unknown offload key {key!r}"

            self.tools.register(
                "recall_offload",
                'Retrieve full offloaded tool content. args: {"key": "off_..."}',
                recall_offload,
            )

        if enable_compact_now and self.tools.get("compact_now") is None:

            def compact_now(reason: str = "model requested") -> str:
                self._force_compact = True
                return f"ok: will compact before next model call ({reason})"

            self.tools.register(
                "compact_now",
                "Request context compaction now (after a finished subtask). "
                'args: {"reason": "optional string"}',
                compact_now,
            )

    def reset(self) -> None:
        self.memory = Memory()
        self.tracer = Tracer()
        self.step = 0
        self.last_result = None
        self._force_compact = False

    def _count(self, text: str) -> int:
        return self.token_counter(text)

    def _cancelled(self) -> bool:
        return bool(self.cancel_check and self.cancel_check())

    def _put_offload(self, content: str) -> str:
        if self._disk is not None:
            key = self._disk.put(content)
            self._mem_store.put(content)
            return key
        return self._mem_store.put(content)

    def _system_prompt(self) -> str:
        if self._system_prompt_cache is not None:
            return self._system_prompt_cache
        parts = [self.agent.system_prompt]
        if self.agent.goal:
            parts.append(f"Goal: {self.agent.goal}")
        tool_desc = self.tools.descriptions(max_chars=self.tool_schema_max_chars)
        if tool_desc and tool_desc != "(no tools registered)":
            if self.use_native_tools:
                parts.append(
                    "You have tools available via the API. Call them when needed. "
                    "Prefer compact_now after finishing a subtask if context is large. "
                    "Final answer: no tool call."
                )
            else:
                parts.append(
                    "Tools — use this format:\n"
                    "TOOL_CALL\nname: <tool_name>\nargs: {\"key\": \"value\"}\nEND_TOOL_CALL\n\n"
                    "Or a fenced JSON block with name/args. "
                    "You may emit multiple TOOL_CALL blocks in one reply.\n"
                    "Call compact_now after a finished subtask if the context is heavy.\n\n"
                    f"Available tools:\n{tool_desc}\n\n"
                    "After tool results, continue. Final answer: no tool call."
                )
        self._system_prompt_cache = "\n\n".join(parts)
        return self._system_prompt_cache

    def invalidate_prompt_cache(self) -> None:
        self._system_prompt_cache = None
