"""Isolated sub-agent run with its own memory window."""

from __future__ import annotations

from typing import Any, Callable

from agentforge.agent import Agent
from agentforge.compaction import CompactionConfig
from agentforge.result import RunResult
from agentforge.runner import Runner
from agentforge.tools import ToolRegistry


def run_subagent(
    *,
    name: str,
    goal: str,
    system_prompt: str,
    user_input: str,
    llm_call: Callable[..., Any],
    tools: ToolRegistry | None = None,
    max_steps: int = 6,
    max_tokens: int = 8_000,
    parent_context: str | None = None,
) -> RunResult:
    """Run a child agent with a fresh memory.

    Only the final reply (and optional short summary) should be fed back to the
    parent — that is the main isolation win vs stuffing the whole child history
    into the parent window.
    """
    agent = Agent(name=name, goal=goal, system_prompt=system_prompt)
    cfg = CompactionConfig(max_tokens=max_tokens, keep_recent_tokens=max(1_500, max_tokens // 4))
    runner = Runner(
        agent,
        tools=tools or ToolRegistry(),
        llm_call=llm_call,
        compaction_config=cfg,
        use_llm_summarizer=False,
    )
    prompt = user_input
    if parent_context:
        prompt = (
            f"[parent context — do not repeat wholesale]\n{parent_context[:2_000]}\n\n"
            f"[your task]\n{user_input}"
        )
    return runner.run_detailed(prompt, max_steps=max_steps)


def subagent_tool(
    *,
    name: str,
    goal: str,
    system_prompt: str,
    llm_call: Callable[..., Any],
    tools: ToolRegistry | None = None,
    max_steps: int = 6,
) -> Callable[..., str]:
    """Build a parent-facing tool that runs an isolated subagent."""

    def _run(task: str, parent_context: str = "") -> str:
        result = run_subagent(
            name=name,
            goal=goal,
            system_prompt=system_prompt,
            user_input=task,
            llm_call=llm_call,
            tools=tools,
            max_steps=max_steps,
            parent_context=parent_context or None,
        )
        return (
            f"[subagent:{name}] stopped={result.stopped_reason} "
            f"tokens={result.total_tokens}\n{result.reply}"
        )

    return _run
