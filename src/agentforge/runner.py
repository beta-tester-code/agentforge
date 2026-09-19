"""Agent runner: text + native tool loop, compaction, cancel, traces."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from agentforge.agent import Agent
from agentforge.compaction import CompactionConfig, compact_memory, make_llm_summarizer
from agentforge.memory import Memory
from agentforge.observability import Tracer
from agentforge.offload_store import DiskOffloadStore, MemoryOffloadStore
from agentforge.pressure import pressure_level
from agentforge.result import RunResult
from agentforge.tokens import get_token_counter
from agentforge.tools import ToolRegistry

TOOL_CALL_RE = re.compile(
    r"TOOL_CALL\s*\n\s*name:\s*(?P<name>\S+)\s*\n\s*args:\s*(?P<args>\{.*?\})\s*\n\s*END_TOOL_CALL",
    re.DOTALL | re.IGNORECASE,
)
TOOL_JSON_RE = re.compile(
    r"```(?:json)?\s*(\{\s*\"name\"\s*:.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)


def _loads_obj(raw: str) -> dict[str, Any]:
    try:
        args = json.loads(raw)
        return args if isinstance(args, dict) else {}
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            try:
                args = json.loads(raw[start : end + 1])
                return args if isinstance(args, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}


def _parse_all_tool_calls(text: str) -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    for m in TOOL_CALL_RE.finditer(text):
        found.append((m.group("name").strip(), _loads_obj(m.group("args").strip())))
    if found:
        return found
    for m in TOOL_JSON_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj:
            args = obj.get("args") or obj.get("arguments") or {}
            if not isinstance(args, dict):
                args = {}
            found.append((str(obj["name"]), args))
    return found


def _parse_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    calls = _parse_all_tool_calls(text)
    return calls[0] if calls else None


def _native_tool_calls(message: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Extract OpenAI-style tool_calls from a message dict."""
    raw = message.get("tool_calls") or []
    out: list[tuple[str, dict[str, Any]]] = []
    for tc in raw:
        try:
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            args_raw = fn.get("arguments") or "{}"
            if isinstance(args_raw, dict):
                args = args_raw
            else:
                args = json.loads(args_raw) if args_raw else {}
            if not isinstance(args, dict):
                args = {}
            if name:
                out.append((str(name), args))
        except (TypeError, json.JSONDecodeError, AttributeError):
            continue
    return out


class Runner:
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
        self._system_prompt_cache: str | None = None
        self.last_result: RunResult | None = None
        self._force_compact = False

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

    def _maybe_compact(self, force: bool = False) -> None:
        before_len = len(self.memory.messages)
        tokens_now = self.memory.token_total()
        level = pressure_level(tokens_now, self.compaction_config.max_tokens)
        if level.name == "warn":
            self.tracer.record(
                self.step, "pressure_warn", input_tokens=tokens_now,
                level=level.name, ratio=round(level.ratio, 3),
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
                self.step, "compaction",
                input_tokens=result.tokens_before,
                output_tokens=result.tokens_after,
                actions=result.actions,
                messages_before=result.messages_before,
                messages_after=result.messages_after,
                pressure=level.name,
            )
            if self.reanchor_goal and self.agent.goal and result.messages_after < before_len:
                anchor = f"[goal reminder] {self.agent.goal}"
                self.memory.add("system", anchor, tokens=self._count(anchor), type="goal_reanchor")

    def _build_messages(self) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": self._system_prompt()}]
        for m in self.memory.messages:
            role = m.role if m.role in ("user", "assistant", "system", "tool") else "user"
            if role == "tool":
                messages.append(
                    {
                        "role": "tool" if self.use_native_tools else "user",
                        "content": m.content if self.use_native_tools else f"[tool result]\n{m.content}",
                        **({"name": m.meta.get("tool", "tool")} if self.use_native_tools else {}),
                    }
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
        return result

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

            raw = self.llm_call(messages)
            if isinstance(raw, dict):
                content = str(raw.get("content") or "")
                calls = _native_tool_calls(raw)
                if not calls and content:
                    calls = _parse_all_tool_calls(content)
                reply_for_mem = content or json.dumps(raw.get("tool_calls") or [], default=str)
            else:
                content = str(raw)
                calls = _parse_all_tool_calls(content)
                reply_for_mem = content

            reply_tokens = self._count(reply_for_mem)

            if not calls:
                self.memory.add("assistant", reply_for_mem, tokens=reply_tokens)
                self.tracer.record(
                    self.step, "assistant_reply",
                    input_tokens=prompt_tokens, output_tokens=reply_tokens,
                )
                self.step += 1
                final_reply = content or reply_for_mem
                stopped = "completed"
                break

            self.memory.add("assistant", reply_for_mem, tokens=reply_tokens)
            self.tracer.record(
                self.step, "tool_call",
                input_tokens=prompt_tokens, output_tokens=reply_tokens,
                tools=[c[0] for c in calls],
            )
            self.step += 1

            any_error = False
            for tool_name, tool_args in calls:
                if self._cancelled():
                    final_reply = "[AgentForge] cancelled"
                    stopped = "cancelled"
                    break
                tool_result = self._execute_tool(tool_name, tool_args)
                tool_tokens = self._count(tool_result)
                self.memory.add("tool", tool_result, tokens=tool_tokens, tool=tool_name)
                self.tracer.record(
                    self.step, "tool_result",
                    input_tokens=tool_tokens, tool=tool_name,
                    is_error=tool_result.startswith("Error"),
                )
                self.step += 1
                if tool_result.startswith("Error"):
                    any_error = True
            else:
                # for-else: only if we didn't break on cancel
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
                continue
            break  # cancelled mid tools
        else:
            final_reply = final_reply or "[AgentForge] max_steps reached without final answer."
            stopped = "max_steps"

        self._maybe_compact()
        return self._build_result(final_reply, stopped)

    def get_trace_summary(self) -> dict[str, Any]:
        return self.tracer.summary()

    def get_trace(self) -> list[dict[str, Any]]:
        return self.tracer.steps()
