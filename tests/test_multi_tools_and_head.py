"""Multi tool-calls in one turn + task head retention."""

from agentforge import Agent, CompactionConfig, Memory, Runner, ToolRegistry, compact_memory
from agentforge.runner import _parse_all_tool_calls


def test_parse_multiple_tool_calls():
    text = """
TOOL_CALL
name: a
args: {"x": 1}
END_TOOL_CALL
some text
TOOL_CALL
name: b
args: {"y": 2}
END_TOOL_CALL
"""
    calls = _parse_all_tool_calls(text)
    assert len(calls) == 2
    assert calls[0] == ("a", {"x": 1})
    assert calls[1] == ("b", {"y": 2})


def test_runner_executes_multiple_tools_one_turn():
    tools = ToolRegistry()
    tools.register("a", "a", lambda x: x + 1)
    tools.register("b", "b", lambda y: y * 2)

    class Once:
        def __init__(self):
            self.n = 0

        def __call__(self, messages):
            self.n += 1
            if self.n == 1:
                return (
                    "TOOL_CALL\nname: a\nargs: {\"x\": 3}\nEND_TOOL_CALL\n"
                    "TOOL_CALL\nname: b\nargs: {\"y\": 4}\nEND_TOOL_CALL"
                )
            return "done"

    runner = Runner(
        Agent("t", "g", "s"),
        tools=tools,
        llm_call=Once(),
        use_llm_summarizer=False,
        compaction_config=CompactionConfig(max_tokens=50_000),
    )
    result = runner.run_detailed("go", max_steps=4)
    assert result.reply == "done"
    roles = [m.role for m in runner.memory.messages]
    assert roles.count("tool") >= 2


def test_task_message_kept_after_compaction():
    mem = Memory()
    mem.add("user", "ORIGINAL TASK: find the bug in auth", tokens=10)
    for i in range(20):
        mem.add("assistant", f"step {i} " + ("word " * 30), tokens=40)
        mem.add("user", f"continue {i}", tokens=5)
    cfg = CompactionConfig(
        max_tokens=80,
        keep_recent_tokens=40,
        summarize_threshold_ratio=0.2,
        offload_tool_results=False,
        keep_task_message=True,
    )
    compact_memory(mem, cfg)
    contents = [m.content for m in mem.messages]
    assert any("ORIGINAL TASK" in c for c in contents)


def test_json_offload_preview():
    mem = Memory()
    big = '{"users": [1,2,3], "meta": {"ok": true}, "blob": "' + ("x" * 3000) + '"}'
    mem.add("tool", big, tokens=900)
    cfg = CompactionConfig(max_tokens=50_000, max_tool_result_chars=200)
    r = compact_memory(mem, cfg)
    assert any("offloaded" in a for a in r.actions)
    assert "keys=" in mem.messages[0].content or "json" in mem.messages[0].content.lower()
