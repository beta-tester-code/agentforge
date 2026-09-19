"""Optional live API tests.

Skipped unless AGENTFORGE_LIVE=1 and OPENAI_API_KEY (or LLM_API_KEY) is set.
Never commit secrets. Export keys only in the local environment.
"""

from __future__ import annotations

import os

import pytest

from agentforge import Agent, Runner, ToolRegistry, make_llm_call, CompactionConfig

pytestmark = pytest.mark.skipif(
    os.getenv("AGENTFORGE_LIVE") != "1"
    or not (os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")),
    reason="Set AGENTFORGE_LIVE=1 and OPENAI_API_KEY/LLM_API_KEY to run live tests",
)


def _llm():
    base = os.getenv(
        "AGENTFORGE_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    model = os.getenv("AGENTFORGE_MODEL", "gemini-3.6-flash")
    return make_llm_call(base_url=base, model=model)


def test_live_simple_chat():
    agent = Agent(name="t", goal="Be brief", system_prompt="Reply in one short sentence.")
    runner = Runner(agent, llm_call=_llm(), use_llm_summarizer=False)
    reply = runner.run("Say hi.")
    assert isinstance(reply, str) and len(reply.strip()) > 0
    assert runner.get_trace_summary()["steps"] >= 2


def test_live_tool_loop():
    tools = ToolRegistry()
    tools.register(
        "add",
        'Add two ints. args: {"a": int, "b": int}',
        lambda a, b: a + b,
    )
    agent = Agent(
        name="math",
        goal="Use tools for arithmetic",
        system_prompt=(
            "When you need to add numbers, call the add tool using the TOOL_CALL format."
        ),
    )
    runner = Runner(
        agent,
        tools=tools,
        llm_call=_llm(),
        use_llm_summarizer=False,
        compaction_config=CompactionConfig(max_tokens=50_000),
    )
    reply = runner.run("What is 17+25? Use the add tool.", max_steps=6)
    assert "42" in reply
    roles = [m.role for m in runner.memory.messages]
    assert "tool" in roles
