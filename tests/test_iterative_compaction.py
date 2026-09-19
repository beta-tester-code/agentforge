"""Iterative summary + token-budget tail + offload recall."""

from agentforge import (
    Agent,
    CompactionConfig,
    Memory,
    Runner,
    ToolRegistry,
    compact_memory,
)


def _summary_msgs(mem: Memory):
    return [m for m in mem.messages if m.meta.get("type") == "compaction_summary"]


def test_token_budget_tail_keeps_recent_by_tokens():
    mem = Memory()
    for i in range(20):
        mem.add("user", f"msg {i} " + ("x" * 40), tokens=20)
    cfg = CompactionConfig(
        max_tokens=80,
        keep_recent_tokens=50,
        keep_recent_messages=2,
        summarize_threshold_ratio=0.3,
        offload_tool_results=False,
    )
    result = compact_memory(mem, cfg)
    assert result.messages_after < result.messages_before
    assert _summary_msgs(mem), "expected a compaction summary message"


def test_second_compaction_is_iterative():
    mem = Memory()
    for i in range(15):
        mem.add("user", f"turn {i} data " + ("word " * 30), tokens=40)
    cfg = CompactionConfig(
        max_tokens=100,
        keep_recent_tokens=60,
        summarize_threshold_ratio=0.2,
        offload_tool_results=False,
    )
    r1 = compact_memory(mem, cfg)
    assert any("summarized" in a for a in r1.actions)
    for i in range(15):
        mem.add("user", f"later {i} " + ("more " * 30), tokens=40)
    r2 = compact_memory(mem, cfg)
    assert any("iterative" in a for a in r2.actions)
    joined = " ".join(m.content for m in _summary_msgs(mem))
    assert "Carried forward" in joined or "New since" in joined


def test_recall_offload_tool():
    tools = ToolRegistry()
    tools.register("big", "returns big string", lambda: "Z" * 5000)

    class Scripted:
        def __init__(self):
            self.n = 0

        def __call__(self, messages):
            self.n += 1
            if self.n == 1:
                return "TOOL_CALL\nname: big\nargs: {}\nEND_TOOL_CALL"
            return "done"

    agent = Agent(name="t", goal="g", system_prompt="s")
    runner = Runner(
        agent,
        tools=tools,
        llm_call=Scripted(),
        use_llm_summarizer=False,
        compaction_config=CompactionConfig(
            max_tokens=50_000,
            max_tool_result_chars=100,
            summarize_threshold_ratio=0.99,
        ),
    )
    runner.run("go", max_steps=4)
    store = runner.memory.state.get("_offload_store") or {}
    assert store, "expected offloaded content"
    key = next(iter(store))
    tool = runner.tools.get("recall_offload")
    assert tool is not None
    full = tool.func(key=key)
    assert full == store[key]
    assert len(full) == 5000
