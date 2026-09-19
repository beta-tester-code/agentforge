"""Schema inference, native tool_calls, disk offload, compact_now, cancel."""

from pathlib import Path

from agentforge import (
    Agent,
    CompactionConfig,
    DiskOffloadStore,
    Memory,
    Runner,
    ToolRegistry,
    compact_memory,
    schema_from_callable,
)


def test_schema_from_callable():
    def add(a: int, b: int = 0) -> int:
        return a + b

    schema = schema_from_callable(add)
    assert schema["type"] == "object"
    assert "a" in schema["properties"]
    assert schema["properties"]["a"].get("type") == "integer"
    assert "a" in schema["required"]
    assert "b" not in schema.get("required", [])


def test_native_tool_calls_from_dict_message():
    tools = ToolRegistry()
    tools.register("add", "add two ints", lambda a, b: a + b)

    class NativeLLM:
        def __init__(self):
            self.n = 0

        def __call__(self, messages):
            self.n += 1
            if self.n == 1:
                return {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "1",
                            "type": "function",
                            "function": {
                                "name": "add",
                                "arguments": '{"a": 2, "b": 3}',
                            },
                        }
                    ],
                }
            return {"content": "5", "tool_calls": []}

    runner = Runner(
        Agent("t", "g", "s"),
        tools=tools,
        llm_call=NativeLLM(),
        use_llm_summarizer=False,
        use_native_tools=True,
        compaction_config=CompactionConfig(max_tokens=50_000),
    )
    result = runner.run_detailed("sum", max_steps=4)
    assert result.reply == "5"
    assert any(m.role == "tool" for m in runner.memory.messages)


def test_disk_offload(tmp_path: Path):
    store = DiskOffloadStore(tmp_path)
    key = store.put("hello" * 100)
    assert store.get(key) == "hello" * 100
    assert (tmp_path / f"{key}.txt").exists()


def test_compact_now_forces_summary():
    tools = ToolRegistry()
    # register only via runner defaults

    class Script:
        def __init__(self):
            self.n = 0

        def __call__(self, messages):
            self.n += 1
            if self.n == 1:
                return (
                    "TOOL_CALL\nname: compact_now\nargs: {\"reason\": \"done\"}\nEND_TOOL_CALL"
                )
            return "ok"

    runner = Runner(
        Agent("t", "g", "s"),
        tools=tools,
        llm_call=Script(),
        use_llm_summarizer=False,
        compaction_config=CompactionConfig(
            max_tokens=50_000,
            summarize_threshold_ratio=0.99,
            respect_prefix_cache=True,
            offload_only_until_ratio=0.99,
        ),
    )
    # pad memory so summarize does something when forced
    for i in range(12):
        runner.memory.add("user", f"pad {i} " + ("x" * 80), tokens=30)
    result = runner.run_detailed("go", max_steps=4)
    assert result.reply == "ok"
    # force path exercised without crash
    assert result.stopped_reason == "completed"


def test_cancel_check():
    flag = {"c": False}

    class AlwaysTool:
        def __call__(self, messages):
            flag["c"] = True
            return "TOOL_CALL\nname: compact_now\nargs: {}\nEND_TOOL_CALL"

    runner = Runner(
        Agent("t", "g", "s"),
        llm_call=AlwaysTool(),
        use_llm_summarizer=False,
        cancel_check=lambda: flag["c"],
        compaction_config=CompactionConfig(max_tokens=50_000),
    )
    # First iteration will set flag during llm_call; cancel checked at start of next
    # Force cancel immediately:
    runner.cancel_check = lambda: True
    result = runner.run_detailed("x", max_steps=3)
    assert result.stopped_reason == "cancelled"


def test_prefix_cache_skips_summary_until_high():
    mem = Memory()
    for i in range(10):
        mem.add("user", f"m {i} " + ("word " * 20), tokens=25)
    cfg = CompactionConfig(
        max_tokens=400,
        summarize_threshold_ratio=0.3,
        offload_only_until_ratio=0.95,
        respect_prefix_cache=True,
        offload_tool_results=False,
        keep_recent_tokens=50,
    )
    r = compact_memory(mem, cfg)
    # under 95% of 400 with ~250 tokens -> offload-only path, no summarize
    assert not any("summarized" in a for a in r.actions)
