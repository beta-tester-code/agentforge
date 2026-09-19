"""Context compaction: offload, iterative summarize, head+tail retention."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from agentforge.memory import Message, Memory


@dataclass
class CompactionConfig:
    max_tokens: int = 32_000
    keep_recent_messages: int = 6
    keep_recent_tokens: int = 4_000
    keep_task_message: bool = True
    summarize_threshold_ratio: float = 0.75
    offload_tool_results: bool = True
    max_tool_result_chars: int = 2_000
    max_summary_chars: int = 2_400
    summary_ratio: float = 0.2
    summary_min_tokens: int = 200
    summary_max_tokens: int = 1_200


@dataclass
class CompactionResult:
    messages_before: int
    messages_after: int
    tokens_before: int
    tokens_after: int
    actions: list[str] = field(default_factory=list)


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def total_message_tokens(messages: list[Message]) -> int:
    return sum(m.tokens or estimate_tokens(m.content) for m in messages)


def _stable_key(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:12]


def _json_preview(text: str, max_chars: int) -> str | None:
    s = text.strip()
    if not s or s[0] not in "{[":
        return None
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict):
        keys = list(obj.keys())[:20]
        parts = [f"json object keys={keys}"]
        for k in keys[:5]:
            v = obj[k]
            if isinstance(v, (dict, list)):
                parts.append(f"  {k}: {type(v).__name__} len={len(v)}")
            else:
                sv = repr(v)
                if len(sv) > 80:
                    sv = sv[:77] + "..."
                parts.append(f"  {k}: {sv}")
        out = "\n".join(parts)
    elif isinstance(obj, list):
        out = f"json array len={len(obj)} sample={json.dumps(obj[:3], default=str)[: max_chars // 2]}"
    else:
        return None
    if len(out) > max_chars:
        out = out[: max_chars - 3] + "..."
    return out


def offload_large_tool_results(
    memory: Memory,
    config: CompactionConfig,
    store: dict[str, str] | None = None,
) -> list[str]:
    actions: list[str] = []
    if store is None:
        store = {}

    for i, msg in enumerate(memory.messages):
        if msg.role != "tool":
            continue
        if msg.meta.get("offloaded"):
            continue
        if len(msg.content) <= config.max_tool_result_chars:
            continue

        key = f"off_{_stable_key(msg.content)}"
        store[key] = msg.content
        structured = _json_preview(msg.content, config.max_tool_result_chars)
        if structured:
            preview = structured
        else:
            preview = msg.content[: config.max_tool_result_chars]
        msg.content = (
            f"[offloaded:{key}] full={len(store[key])} chars. "
            f"Use recall_offload(key=\"{key}\") for full content.\n"
            f"{preview}"
        )
        if not structured and len(store[key]) > config.max_tool_result_chars:
            msg.content += "\n...[truncated]"
        msg.tokens = estimate_tokens(msg.content)
        msg.meta["offloaded"] = key
        actions.append(f"offloaded tool@{i} -> {key}")

    return actions


def _tail_by_messages(messages: list[Message], keep: int) -> tuple[list[Message], list[Message]]:
    keep = max(1, keep)
    if len(messages) <= keep:
        return [], list(messages)
    return messages[:-keep], messages[-keep:]


def _tail_by_tokens(messages: list[Message], budget: int) -> tuple[list[Message], list[Message]]:
    acc = 0
    tail: list[Message] = []
    for m in reversed(messages):
        t = m.tokens or estimate_tokens(m.content)
        if tail and acc + t > budget:
            break
        tail.append(m)
        acc += t
    tail.reverse()
    if not tail:
        tail = messages[-1:]
    older = messages[: len(messages) - len(tail)]
    return older, tail


def _select_tail(messages: list[Message], config: CompactionConfig) -> tuple[list[Message], list[Message]]:
    if not messages:
        return [], []
    if config.keep_recent_tokens > 0:
        older, tail = _tail_by_tokens(messages, config.keep_recent_tokens)
        if older:
            return older, tail
        return _tail_by_messages(messages, config.keep_recent_messages)
    return _tail_by_messages(messages, config.keep_recent_messages)


def _extract_task(messages: list[Message]) -> Message | None:
    for m in messages:
        if m.role == "user" and m.meta.get("type") != "goal_reanchor":
            return m
    return None


def default_summarizer(messages: list[Message]) -> str:
    decisions: list[str] = []
    artifacts: list[str] = []
    other: list[str] = []

    for m in messages:
        content = " ".join(m.content.split())
        if len(content) > 100:
            content = content[:97] + "..."
        line = f"{m.role}:{content}"
        low = content.lower()
        if m.role == "tool" or "offloaded:" in content:
            artifacts.append(line)
        elif any(k in low for k in ("decid", "chose", "will use", "plan:")):
            decisions.append(line)
        else:
            other.append(line)

    parts = [
        "## Progress",
        " | ".join(other[:12]) or "(none)",
        "## Decisions",
        " | ".join(decisions[:8]) or "(none)",
        "## Artifacts/tools",
        " | ".join(artifacts[:8]) or "(none)",
    ]
    text = "\n".join(parts)
    if len(text) > 1200:
        text = text[:1197] + "..."
    return text


def _extract_previous_summary(messages: list[Message]) -> str | None:
    for m in messages:
        if m.meta.get("type") == "compaction_summary":
            return m.content
        if m.role == "system" and m.content.startswith("[context summary]"):
            return m.content
    return None


def compact_memory(
    memory: Memory,
    config: CompactionConfig,
    token_counter: Callable[[str], int] | None = None,
    summarizer: Callable[[list[Message]], str] | None = None,
) -> CompactionResult:
    counter = token_counter or estimate_tokens
    summarize = summarizer or default_summarizer

    for m in memory.messages:
        if m.tokens is None:
            m.tokens = counter(m.content)

    tokens_before = total_message_tokens(memory.messages)
    messages_before = len(memory.messages)
    actions: list[str] = []

    if config.offload_tool_results:
        offload_store: dict[str, str] = memory.state.setdefault("_offload_store", {})
        actions.extend(offload_large_tool_results(memory, config, offload_store))

    tokens_now = total_message_tokens(memory.messages)
    threshold = int(config.max_tokens * config.summarize_threshold_ratio)

    if tokens_now > threshold and len(memory.messages) > 2:
        older, recent = _select_tail(memory.messages, config)
        if older:
            task = _extract_task(older) if config.keep_task_message else None
            prev = _extract_previous_summary(older)
            body = [
                m
                for m in older
                if m.meta.get("type") != "compaction_summary"
                and not (m.role == "system" and m.content.startswith("[context summary]"))
                and m is not task
            ]

            summary_text = summarize(body) if body else "(empty middle)"
            if prev:
                summary_text = (
                    "## Carried forward\n"
                    + prev.replace("[context summary]\n", "").strip()[:800]
                    + "\n## New since last compaction\n"
                    + summary_text
                )

            if len(summary_text) > config.max_summary_chars:
                summary_text = summary_text[: config.max_summary_chars - 3] + "..."

            summary_msg = Message(
                role="system",
                content=f"[context summary]\n{summary_text}",
                tokens=counter(summary_text) + 5,
                meta={
                    "type": "compaction_summary",
                    "original_count": len(older),
                    "iterative": bool(prev),
                },
            )
            head: list[Message] = []
            if task is not None and task not in recent:
                head.append(task)
            memory.messages = head + [summary_msg] + recent
            actions.append(
                f"summarized {len(older)} older -> summary"
                + (" (iterative)" if prev else "")
                + (" +task" if head else "")
            )

    tokens_after = total_message_tokens(memory.messages)
    return CompactionResult(
        messages_before=messages_before,
        messages_after=len(memory.messages),
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        actions=actions,
    )


def make_llm_summarizer(
    llm_call: Callable[[list[dict[str, str]]], str],
) -> Callable[[list[Message]], str]:
    def _summarize(messages: list[Message]) -> str:
        transcript = [f"{m.role.upper()}: {m.content}" for m in messages]
        body = "\n".join(transcript)
        if len(body) > 12_000:
            body = body[:12_000] + "\n...[truncated for summarizer]"
        prompt = [
            {
                "role": "system",
                "content": (
                    "Compress agent history into a dense brief.\n"
                    "Sections: ## Goal ## Decisions ## Artifacts ## Open ## Next\n"
                    "Keep paths, tool names, numbers, constraints. Drop chatter."
                ),
            },
            {"role": "user", "content": f"History:\n\n{body}"},
        ]
        return llm_call(prompt).strip()

    return _summarize
