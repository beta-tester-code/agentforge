"""Runner works without an LLM (skeleton mode)."""

from agentforge import Agent, Runner, CompactionConfig


def test_runner_skeleton_mode():
    agent = Agent(
        name="t",
        goal="test",
        system_prompt="You are a test agent.",
    )
    runner = Runner(
        agent,
        llm_call=None,
        compaction_config=CompactionConfig(max_tokens=4_000),
    )
    reply = runner.run("hello")
    assert "AgentForge" in reply or "skeleton" in reply.lower() or "Received" in reply
    summary = runner.get_trace_summary()
    assert summary["steps"] >= 1
