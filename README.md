# agentforge

Small Python library for agent loops that spend fewer tokens and stay debuggable.

If you've shipped something on LangGraph or CrewAI and watched the context window fill with tool dumps and repeated system prompts, this is aimed at that problem. It is not another orchestration framework.

## What it does

- Counts tokens per step
- Offloads large tool results (with a `recall_offload` tool to pull them back)
- Compacts older history with an iterative summary so the second pass does not erase the first
- Keeps a recent tail by token budget, not a fixed message count
- Caps how much tool schema gets injected every turn
- Returns a `RunResult` with a full step trace

Requires Python 3.11+.

## Install

```bash
git clone https://github.com/beta-tester-code/agentforge.git
cd agentforge
pip install -e ".[dev]"
```

```bash
PYTHONPATH=src pytest -q
```

## Quick start

No API key (skeleton mode):

```bash
PYTHONPATH=src python examples/basic_agent.py
PYTHONPATH=src python -m agentforge.cli run -p "Hello"
```

With Gemini:

```bash
export OPENAI_API_KEY=your-key
PYTHONPATH=src python -m agentforge.cli run --provider gemini -p "Summarize context rot" --trace
```

In code:

```python
from agentforge import Agent, Runner, make_llm_call

llm = make_llm_call(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    model="gemini-3.5-flash",
)
agent = Agent(
    name="demo",
    goal="Answer briefly",
    system_prompt="Be direct. Prefer tools when the question needs them.",
)
runner = Runner(agent, llm_call=llm)
print(runner.run("Hello"))
print(runner.last_result)  # tokens, tool_calls, stopped_reason, ...
```

Offline compaction check:

```bash
PYTHONPATH=src python scripts/bench_compaction.py
PYTHONPATH=src python examples/token_savings_demo.py
```

More detail: [docs/TESTING.md](docs/TESTING.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Paid help

If you already have a multi-step agent in production and token cost or context drift is the bottleneck, I take a limited number of fixed-scope audits.

Details and pricing: [DESIGN_PARTNER.md](DESIGN_PARTNER.md).

## Name collision

Several unrelated projects use "AgentForge". This one is only the Python package in this repo (token/context tooling). Apache-2.0.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
