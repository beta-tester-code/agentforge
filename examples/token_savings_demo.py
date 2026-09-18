"""Show compaction savings without needing an API key."""

from agentforge import Memory, CompactionConfig, compact_memory, get_token_counter


def main() -> None:
    counter = get_token_counter()
    mem = Memory()

    # Simulate a long tool-heavy agent session
    for i in range(25):
        mem.add("user", f"Step {i}: please analyze batch and continue.")
        mem.add(
            "assistant",
            f"Calling tools for step {i}. " + ("reasoning " * 20),
        )
        mem.add(
            "tool",
            ("ROW,value,score\n" * 100) + f"batch_id={i}\n",
        )

    for m in mem.messages:
        m.tokens = counter(m.content)

    before = sum(m.tokens or 0 for m in mem.messages)
    cfg = CompactionConfig(
        max_tokens=1_500,
        keep_recent_messages=4,
        summarize_threshold_ratio=0.5,
        max_tool_result_chars=400,
    )
    result = compact_memory(mem, cfg, token_counter=counter)
    after = sum(m.tokens or 0 for m in mem.messages)
    saved = before - after
    pct = (100.0 * saved / before) if before else 0.0

    print("AgentForge token savings demo (no API key)")
    print(f"messages: {result.messages_before} -> {result.messages_after}")
    print(f"tokens:   {before} -> {after}")
    print(f"saved:    {saved} ({pct:.1f}%)")
    print("actions:")
    for a in result.actions[:8]:
        print(" -", a)
    if len(result.actions) > 8:
        print(f" - … +{len(result.actions) - 8} more")


if __name__ == "__main__":
    main()
