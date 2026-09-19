"""Agent runner: text + native tool loop, compaction, cancel, budget, traces."""

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
from agentforge.trace_store import TraceStore

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


def _native_tool_calls(
    message: dict[str, Any],
) -> list[tuple[str, dict[str, Any], str | None]]:
    raw = message.get("tool_calls") or []
    out: list[tuple[str, dict[str, Any], str | None]] = []
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
            tid = tc.get("id")
            if name:
                out.append((str(name), args, tid))
        except (TypeError, json.JSONDecodeError, AttributeError):
            continue
    return out
