"""Benchmark iterative compaction vs naive single-pass behaviour."""

from __future__ import annotations

from agentforge import Memory, CompactionConfig, compact_memory, get_token_counter


def fill(mem: Memory, n: int, prefix: str) -> None:
    for i in range(n):
        mem.add("user", f"{prefix} {i} " + ("context " * 25))
        mem.add("assistant", f"{prefix} reply {i} " + ("text " * 20))
        mem.add("tool", ("LINE\n" * 60) + f"id={prefix}-{i}")


def main() -> None:
    counter = get_token_counter()
    cfg = CompactionConfig(
        max_tokens=1_200,
        keep_recent_tokens=400,
        summarize_threshold_ratio=0.4,
        max_tool_result_chars=250,
    )

    mem = Memory()
    fill(mem, 20, "phase1")
    for m in mem.messages:
        m.tokens = counter(m.content)
    before = sum(m.tokens or 0 for m in mem.messages)
    r1 = compact_memory(mem, cfg, token_counter=counter)
    mid = sum(m.tokens or 0 for m in mem.messages)

    fill(mem, 20, "phase2")
    for m in mem.messages:
        if m.tokens is None:
            m.tokens = counter(m.content)
    r2 = compact_memory(mem, cfg, token_counter=counter)
    after = sum(m.tokens or 0 for m in mem.messages)

    print("Advanced compaction bench")
    print(f"phase1: {before} -> {mid} actions={r1.actions[-1:]}")
    print(f"phase2: -> {after} actions={r2.actions[-2:]}")
    print(f"iterative: {any('iterative' in a for a in r2.actions)}")
    print(f"summary head: {mem.messages[0].content[:180]!r}...")


if __name__ == "__main__":
    main()
