"""Minimal example using the AgentForge skeleton."""

from agentforge.agent import Agent
from agentforge.runner import Runner


def main() -> None:
    agent = Agent(
        name="demo",
        goal="Answer the user helpfully while staying token-efficient.",
        system_prompt="You are a careful, concise assistant.",
    )

    runner = Runner(agent)
    result = runner.run("Hello, AgentForge")
    print(result)
    print("\nTracer summary:", runner.tracer.summary())


if __name__ == "__main__":
    main()
