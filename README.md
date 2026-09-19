# agentforge

Small Python library for agent loops that spend fewer tokens and stay debuggable.

If you've shipped something on LangGraph or CrewAI and watched the context window fill with tool dumps and repeated system prompts, this is aimed at that problem. It is not another orchestration framework.

## What it does

- Counts tokens per step; returns a `RunResult` + step trace
- Offloads large tool results (`recall_offload`); optional disk store via `offload_dir=`
- Iterative summaries (second pass keeps the first) + keeps the original task message
- Recent tail by token budget; optional prefix-cache-friendly mode (offload before summarize)
- Text `TOOL_CALL` format and native OpenAI-style `tool_calls`
- JSON Schema for tools inferred from type hints
- `compact_now` tool so the model can request compaction after a subtask
- Cooperative cancel via `cancel_check`; streaming helper on the HTTP client

Requires Python 3.11+.

## Install

```bash
git clone https://github.com/beta-tester-code/agentforge.git
cd agentforge
pip install -e ".[dev]"
```

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python scripts/eval_harness.py
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
print(runner.last_result)
```

Offline checks:

```bash
PYTHONPATH=src python scripts/bench_compaction.py
PYTHONPATH=src python scripts/eval_harness.py
PYTHONPATH=src python examples/token_savings_demo.py
```

More: [docs/TESTING.md](docs/TESTING.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Paid help

Fixed-scope audits for production token/context issues: [DESIGN_PARTNER.md](DESIGN_PARTNER.md).

## Name collision

Several unrelated projects use "AgentForge". This repo is only the Python package here. Apache-2.0.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
