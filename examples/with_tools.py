"""Example: agent that can use a simple tool."""

import os
from datetime import datetime, timezone

from agentforge import Agent, Runner, CompactionConfig, ToolRegistry, make_llm_call


def get_current_utc_time() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def main() -> None:
    tools = ToolRegistry()
    tools.register(
        "get_current_utc_time",
        "Returns the current date and time in UTC.",
        get_current_utc_time,
    )

    agent = Agent(
        name="time-agent",
        goal="Help the user with time-related questions using tools when needed.",
        system_prompt="You are a precise assistant. Use tools when they help you answer accurately.",
    )

    llm_call = None
    if os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY"):
        llm_call = make_llm_call(model="gpt-4o-mini")
        print("LLM wired.")
    else:
        print("No API key — skeleton mode (tool loop will not run fully).")

    runner = Runner(
        agent,
        tools=tools,
        llm_call=llm_call,
        compaction_config=CompactionConfig(max_tokens=8_000),
    )

    result = runner.run("What time is it in UTC right now?")
    print("\nFinal reply:")
    print(result)
    print("\nTrace:", runner.get_trace_summary())


if __name__ == "__main__":
    main()
