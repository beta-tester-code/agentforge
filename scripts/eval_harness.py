"""Offline eval: token reduction vs full-history baseline on synthetic tool-heavy runs."""

from __future__ import annotations

from agentforge import CompactionConfig, Memory, compact_memory, get_token_counter


def build_session(n_tools: int = 30) -> Memory:
    mem = Memory()
    mem.add("user", "Investigate failing auth and report root cause.")
    for i in range(n_tools):
        mem.add("assistant", f"calling tool search_{i}")
        payload = ("LINE\n" * 40) + f"id={i}" + (" data" * 50)
        mem.add("tool", payload)
        mem.add("assistant", f"noted result {i}")
    return mem


def main() -> None:
    counter = get_token_counter()
    mem = build_session()
    for m in mem.messages:
        m.tokens = counter(m.content)
    before = sum(m.tokens or 0 for m in mem.messages)

    cfg = CompactionConfig(
        max_tokens=2_000,
        keep_recent_tokens=600,
        summarize_threshold_ratio=0.4,
        max_tool_result_chars=200,
        keep_task_message=True,
    )
    result = compact_memory(mem, cfg, token_counter=counter)
    after = sum(m.tokens or 0 for m in mem.messages)
    saved = before - after
    pct = 100.0 * saved / before if before else 0.0

    task_ok = any("Investigate failing auth" in m.content for m in mem.messages)
    print("eval_harness")
    print(f"  messages: {result.messages_before} -> {result.messages_after}")
    print(f"  tokens:   {before} -> {after}  saved={saved} ({pct:.1f}%)")
    print(f"  task retained: {task_ok}")
    print(f"  actions: {result.actions[-3:]}")
    if pct < 50 or not task_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
