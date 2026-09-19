"""RunResult + Memory serialization + error streak."""

from agentforge import Agent, Memory, Message, Runner, ToolRegistry, CompactionConfig
from agentforge.pressure import pressure_level


def test_memory_roundtrip():
    mem = Memory()
    mem.add("user", "hi", tokens=1, foo="bar")
    mem.state["_offload_store"] = {"k": "v" * 10}
    data = mem.to_dict()
    mem2 = Memory.from_dict(data)
    assert len(mem2.messages) == 1
    assert mem2.messages[0].content == "hi"
    assert mem2.messages[0].meta["foo"] == "bar"
    assert mem2.state["_offload_store"]["k"].startswith("v")


def test_run_detailed_skeleton():
    agent = Agent(name="t", goal="g", system_prompt="s")
    runner = Runner(agent, llm_call=None)
    result = runner.run_detailed("hello")
    assert result.reply.startswith("[AgentForge]")
    assert result.stopped_reason == "skeleton"
    assert result.total_tokens >= 0
    assert runner.last_result is result


def test_error_streak_stops():
    tools = ToolRegistry()
    tools.register("boom", "always fails", lambda: (_ for _ in ()).throw(RuntimeError("x")))

    class AlwaysTool:
        def __call__(self, messages):
            return "TOOL_CALL\nname: boom\nargs: {}\nEND_TOOL_CALL"

    agent = Agent(name="t", goal="g", system_prompt="s")
    runner = Runner(
        agent,
        tools=tools,
        llm_call=AlwaysTool(),
        use_llm_summarizer=False,
        max_tool_errors=2,
        compaction_config=CompactionConfig(max_tokens=50_000),
    )
    result = runner.run_detailed("go", max_steps=10)
    assert result.stopped_reason == "error_streak"
    assert "tool errors" in result.reply.lower() or "Error" in result.reply


def test_pressure_levels():
    assert pressure_level(10, 100).name == "ok"
    assert pressure_level(75, 100).name == "warn"
    assert pressure_level(85, 100).name == "mask"
    assert pressure_level(95, 100).name == "aggressive"
    assert pressure_level(100, 100).name == "critical"


def test_message_from_dict_defaults():
    m = Message.from_dict({"content": "only"})
    assert m.role == "user"
    assert m.content == "only"
