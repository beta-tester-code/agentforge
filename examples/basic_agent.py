"""Minimal example using AgentForge.

Without an API key it runs in skeleton mode.
With OPENAI_API_KEY (or LLM_API_KEY) it calls a real model.
"""

import os

from agentforge import Agent, Runner, CompactionConfig, make_llm_call


def main() -> None:
    agent = Agent(
        name="demo",
        goal="Answer the user helpfully and concisely.",
        system_prompt="You are a careful, concise assistant focused on clarity.",
    )

    llm_call = None
    if os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY"):
        llm_call = make_llm_call(model="gpt-4o-mini")
        print("LLM wired (OpenAI-compatible).")
    else:
        print("No API key found — running in skeleton mode.")

    runner = Runner(
        agent,
        compaction_config=CompactionConfig(max_tokens=8_000, keep_recent_messages=4),
        llm_call=llm_call,
    )

    result = runner.run("Hello, AgentForge. What is your main focus?")
    print("\nReply:")
    print(result)
    print("\nTrace summary:", runner.get_trace_summary())


if __name__ == "__main__":
    main()
