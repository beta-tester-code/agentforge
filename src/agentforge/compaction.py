"""Context compaction primitives.

Design goals:
- Prefer reversible operations (offload) over destructive ones (summarize).
- Keep a recent lossless tail.
- Make every compaction decision measurable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agentforge.memory import Message, Memory


@dataclass
class CompactionConfig:
    """Controls when and how context is reduced."""

    max_tokens: int = 32_000
    keep_recent_messages: int = 6
    summarize_threshold_ratio: float = 0.75  # start considering compaction at 75%
    offload_tool_results: bool = True
    max_tool_result_chars: int = 2_000  # keep only a preview above this size


@dataclass
class CompactionResult:
    messages_before: int
    messages_after: int
    tokens_before: int
    tokens_after: int
    actions: list[str] = field(default_factory=list)


def estimate_tokens(text: str) -> int:
    """Very cheap token estimate (approx 4 chars/token).

    Real counting with tiktoken is done at the Runner level when available.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def total_message_tokens(messages: list[Message]) -> int:
    return sum(m.tokens or estimate_tokens(m.content) for m in messages)


def offload_large_tool_results(
    memory: Memory,
    config: CompactionConfig,
    store: dict[str, str] | None = None,
) -> list[str]:
    """Replace large tool outputs with a short preview + pointer.

    This is reversible: the full content stays in `store` if provided.
    """
    actions: list[str] = []
    if store is None:
        store = {}

    for i, msg in enumerate(memory.messages):
        if msg.role != "tool":
            continue
        if len(msg.content) <= config.max_tool_result_chars:
            continue

        key = f"tool_result_{i}"
        store[key] = msg.content
        preview = msg.content[: config.max_tool_result_chars]
        msg.content = (
            f"[offloaded:{key}] preview ({len(store[key])} chars):\n"
            f"{preview}\n...[truncated]"
        )
        msg.tokens = estimate_tokens(msg.content)
        msg.meta["offloaded"] = key
        actions.append(f"offloaded tool result at index {i} -> {key}")

    return actions


def compact_memory(
    memory: Memory,
    config: CompactionConfig,
    token_counter: Callable[[str], int] | None = None,
    summarizer: Callable[[list[Message]], str] | None = None,
) -> CompactionResult:
    """Apply compaction strategy to memory.

    Strategy order:
    1. Offload large tool results (reversible)
    2. If still over threshold and summarizer is available, summarize older messages
       while keeping the recent tail intact.
    """
    counter = token_counter or estimate_tokens

    # Ensure every message has a token count
    for m in memory.messages:
        if m.tokens is None:
            m.tokens = counter(m.content)

    tokens_before = total_message_tokens(memory.messages)
    messages_before = len(memory.messages)
    actions: list[str] = []

    # 1. Offload large tool results
    if config.offload_tool_results:
        offload_store: dict[str, str] = memory.state.setdefault("_offload_store", {})
        actions.extend(offload_large_tool_results(memory, config, offload_store))

    tokens_now = total_message_tokens(memory.messages)
    threshold = int(config.max_tokens * config.summarize_threshold_ratio)

    # 2. Summarize older messages if still over threshold
    if tokens_now > threshold and summarizer is not None and len(memory.messages) > config.keep_recent_messages:
        keep = config.keep_recent_messages
        older = memory.messages[:-keep]
        recent = memory.messages[-keep:]

        summary_text = summarizer(older)
        summary_msg = Message(
            role="system",
            content=f"[context summary]\n{summary_text}",
            tokens=counter(summary_text) + 5,
            meta={"type": "compaction_summary", "original_count": len(older)},
        )

        memory.messages = [summary_msg] + recent
        actions.append(f"summarized {len(older)} older messages into 1 summary")

    tokens_after = total_message_tokens(memory.messages)
    return CompactionResult(
        messages_before=messages_before,
        messages_after=len(memory.messages),
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        actions=actions,
    )
