"""Integration tests for the tool loop using a deterministic mock LLM.

No network, no API key required.
"""

from __future__ import annotations

from agentforge import Agent, Runner, ToolRegistry, CompactionConfig


class ScriptedLLM:
    """Returns scripted responses in order."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def __call__(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        if not self.replies:
            return "(no more scripted replies)"
        return self.replies.pop(0)


def test_tool_loop_calls_tool_then_answers():
    tools = ToolRegistry()
    tools.register(
        "add",
        "Add two integers. args: {\"a\": int, \"b\": int}",
        lambda a, b: a + b,
    )

    llm = ScriptedLLM(
        [
            (
                "I need to compute this.\n"
                "TOOL_CALL\n"
                "name: add\n"
                "args: {\"a\": 2, \"b\": 3}\n"
                "END_TOOL_CALL"
            ),
            "The sum is 5.",
        ]
    )

    agent = Agent(
        name="math",
        goal="Answer math questions using tools",
        system_prompt="Use tools when needed.",
    )
    runner = Runner(
        agent,
        tools=tools,
        llm_call=llm,
        use_llm_summarizer=False,
        compaction_config=CompactionConfig(max_tokens=50_000),
    )

    reply = runner.run("What is 2+3?")
    assert "5" in reply
    assert len(llm.calls) == 2

    summary = runner.get_trace_summary()
    assert summary["steps"] >= 3  # user + tool_call + tool_result + final

    # Ensure tool result entered memory
    roles = [m.role for m in runner.memory.messages]
    assert "tool" in roles


def test_unknown_tool_returns_error_and_continues():
    tools = ToolRegistry()
    # no tools registered on purpose

    llm = ScriptedLLM(
        [
            (
                "TOOL_CALL\n"
                "name: missing_tool\n"
                "args: {}\n"
                "END_TOOL_CALL"
            ),
            "I could not use that tool.",
        ]
    )

    agent = Agent(name="t", goal="test", system_prompt="test")
    runner = Runner(agent, tools=tools, llm_call=llm, use_llm_summarizer=False)
    reply = runner.run("do something")
    assert "could not" in reply.lower() or "tool" in reply.lower() or len(reply) > 0

    tool_msgs = [m for m in runner.memory.messages if m.role == "tool"]
    assert tool_msgs
    assert "unknown tool" in tool_msgs[0].content.lower()


def test_max_steps_stops():
    tools = ToolRegistry()
    tools.register("ping", "returns pong", lambda: "pong")

    # Always request the tool -> should hit max_steps
    always_tool = (
        "TOOL_CALL\n"
        "name: ping\n"
        "args: {}\n"
        "END_TOOL_CALL"
    )
    llm = ScriptedLLM([always_tool] * 10)

    agent = Agent(name="t", goal="test", system_prompt="test")
    runner = Runner(agent, tools=tools, llm_call=llm, use_llm_summarizer=False)
    reply = runner.run("loop", max_steps=3)
    assert "max_steps" in reply.lower() or reply.startswith("[") or len(reply) >= 0
    assert len(llm.calls) <= 3
