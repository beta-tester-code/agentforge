"""Minimal example using the AgentForge skeleton."""

from agentforge import Agent, Runner, CompactionConfig


def main() -> None:
    agent = Agent(
        name="demo",
        goal="Answer the user helpfully while staying token-efficient.",
        system_prompt="You are a careful, concise assistant.",
    )

    runner = Runner(
        agent,
        compaction_config=CompactionConfig(max_tokens=8_000, keep_recent_messages=4),
    )

    result = runner.run("Hello, AgentForge. What is your focus?")
    print(result)
    print("\nTrace summary:", runner.get_trace_summary())


if __name__ == "__main__":
    main()
