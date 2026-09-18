"""Trace detail API."""

from agentforge import Agent, Runner


def test_get_trace_returns_steps():
    agent = Agent(name="t", goal="g", system_prompt="s")
    runner = Runner(agent, llm_call=None)
    runner.run("hi")
    steps = runner.get_trace()
    assert isinstance(steps, list)
    assert len(steps) >= 1
    assert "action" in steps[0]
    summary = runner.get_trace_summary()
    assert "by_action" in summary
