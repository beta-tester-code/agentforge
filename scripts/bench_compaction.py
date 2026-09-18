"""Measure compaction token reduction without an API key."""

from __future__ import annotations

from agentforge import Memory, CompactionConfig, compact_memory, get_token_counter


def main() -> None:
    counter = get_token_counter()
    mem = Memory()
    for i in range(40):
        mem.add("user", f"User turn {i}: " + ("context data " * 30))
        mem.add("assistant", f"Assistant turn {i}: " + ("response text " * 25))
        mem.add("tool", ("TOOL_OUTPUT_LINE\n" * 80))

    for m in mem.messages:
        m.tokens = counter(m.content)

    before = sum(m.tokens or 0 for m in mem.messages)
    cfg = CompactionConfig(
        max_tokens=800,
        keep_recent_messages=4,
        summarize_threshold_ratio=0.4,
        max_tool_result_chars=300,
    )
    result = compact_memory(mem, cfg, token_counter=counter)
    after = sum(m.tokens or 0 for m in mem.messages)
    saved = before - after
    pct = (100.0 * saved / before) if before else 0.0

    print("messages:", result.messages_before, "->", result.messages_after)
    print("tokens:", before, "->", after)
    print(f"saved: {saved} ({pct:.1f}%)")
    print("actions:")
    for a in result.actions:
        print(" -", a)


if __name__ == "__main__":
    main()
