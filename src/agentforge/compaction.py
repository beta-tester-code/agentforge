"""Context compaction primitives.

Engineering details that matter in production (2026 lessons):
- Iterative summaries: each compaction PRESERVES prior summary content and ADDS progress
  (without this, the 2nd compaction destroys the 1st).
- Token-budget tail: protect recent context by tokens, not fixed message count.
- Offload before summarize (reversible > destructive).
- Structured summary fields: goal, decisions, artifacts, open questions.
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
    # Prefer token budget for tail when set (>0). Falls back to keep_recent_messages.
    keep_recent_tokens: int = 4_000
    summarize_threshold_ratio: float = 0.75
    offload_tool_results: bool = True
    max_tool_result_chars: int = 2_000
    # Cap extractive/default summary size
    max_summary_chars: int = 2_400
    # Scaled summary budget as fraction of compressed content (LLM path)
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


def offload_large_tool_results(
    memory: Memory,
    config: CompactionConfig,
    store: dict[str, str] | None = None,
) -> list[str]:
    """Replace large tool outputs with preview + pointer (reversible)."""
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

        key = f"tool_result_{i}_{len(store)}"
        store[key] = msg.content
        preview = msg.content[: config.max_tool_result_chars]
        msg.content = (
            f"[offloaded:{key}] preview ({len(store[key])} chars). "
            f"Use recall_offload(key=\"{key}\") to read full content.\n"
            f"{preview}\n...[truncated]"
        )
        msg.tokens = estimate_tokens(msg.content)
        msg.meta["offloaded"] = key
        actions.append(f"offloaded tool result at index {i} -> {key}")

    return actions


def _select_tail(messages: list[Message], config: CompactionConfig) -> tuple[list[Message], list[Message]]:
    """Split into older + recent tail using token budget when possible."""
    if not messages:
        return [], []

    if config.keep_recent_tokens > 0:
        acc = 0
        tail: list[Message] = []
        for m in reversed(messages):
            t = m.tokens or estimate_tokens(m.content)
            if tail and acc + t > config.keep_recent_tokens:
                break
            tail.append(m)
            acc += t
        tail.reverse()
        if not tail:
            tail = messages[-1:]
        older = messages[: len(messages) - len(tail)]
        return older, tail

    keep = max(1, config.keep_recent_messages)
    if len(messages) <= keep:
        return [], list(messages)
    return messages[:-keep], messages[-keep:]


def default_summarizer(messages: list[Message]) -> str:
    """Aggressive extractive fallback with structured sections."""
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
    """Apply compaction strategy to memory."""
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
            prev = _extract_previous_summary(older)
            # Drop old summary messages from the body being re-summarized
            body = [
                m
                for m in older
                if m.meta.get("type") != "compaction_summary"
                and not (m.role == "system" and m.content.startswith("[context summary]"))
            ]

            summary_text = summarize(body)
            if prev:
                # Iterative update: preserve + add (Pi-mono / Hermes lesson)
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
            memory.messages = [summary_msg] + recent
            actions.append(
                f"summarized {len(older)} older messages into 1 summary"
                + (" (iterative)" if prev else "")
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
    """Build a structured LLM summarizer."""

    def _summarize(messages: list[Message]) -> str:
        transcript = []
        for m in messages:
            transcript.append(f"{m.role.upper()}: {m.content}")
        body = "\n".join(transcript)
        if len(body) > 12_000:
            body = body[:12_000] + "\n...[truncated for summarizer]"

        prompt = [
            {
                "role": "system",
                "content": (
                    "Compress agent history into a dense structured brief.\n"
                    "Sections (use exactly these headings):\n"
                    "## Goal\n## Decisions\n## Artifacts\n## Open questions\n## Next\n"
                    "Preserve file paths, tool names, numbers, and constraints. "
                    "Drop chatter and repeated failed attempts unless they constrain future work. "
                    "Return only the brief."
                ),
            },
            {"role": "user", "content": f"History to compress:\n\n{body}"},
        ]
        return llm_call(prompt).strip()

    return _summarize
