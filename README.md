# AgentForge (beta-tester-code)

**Python toolkit for token-efficient, observable agents — compaction first, framework last.**

This is *not* DataBassGit/AgentForge, agentforge.dev, Agentman, or the ICP protocol.
This repo: [github.com/beta-tester-code/agentforge](https://github.com/beta-tester-code/agentforge)

Focus:

- Measure every token that hits the model
- Offload fat tool results before they rot the window
- Summarize only after the recent tail is kept lossless
- Emit a compact run trace (local now; hosted later)

Not a second LangGraph. A small runtime you can drop next to an existing loop when **CrewAI/LangGraph token bills** and **context rot** become the problem.

See [docs/WHY.md](docs/WHY.md) for the market argument.

---

## Status

**v0.1.0 — usable MVP** · created 2026-09-17 · **0 stars / 0 paying customers** (honest).

| Feature | Status |
|---------|--------|
| Agent + Memory + Tracer | ✅ |
| Token counting (tiktoken) | ✅ |
| Tool registry + multi-step tool loop | ✅ |
| Tool-result offloading + summarization compaction | ✅ |
| OpenAI-compatible client (OpenAI / Groq / Gemini / …) | ✅ |
| CLI with provider presets | ✅ |
| Mock + optional live tests | ✅ |
| Hosted traces | planned (paid) |
| PyPI publish | not yet |

---

## Install

```bash
git clone https://github.com/beta-tester-code/agentforge.git
cd agentforge
pip install -e ".[dev]"
```

```bash
PYTHONPATH=src pytest -q
```

---

## Quick Start

### Skeleton (no key)

```bash
PYTHONPATH=src python examples/basic_agent.py
PYTHONPATH=src python -m agentforge.cli run -p "Hello"
```

### Gemini (Google AI Studio)

```bash
export OPENAI_API_KEY=your-key
PYTHONPATH=src python -m agentforge.cli run --provider gemini -p "What is context rot?" --trace
```

### Groq

```bash
export OPENAI_API_KEY=your-groq-key
PYTHONPATH=src python -m agentforge.cli run --provider groq -p "Hi" --trace
```

### Library

```python
from agentforge import Agent, Runner, make_llm_call, ToolRegistry

llm = make_llm_call(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    model="gemini-3.5-flash",
)
agent = Agent(name="demo", goal="Be helpful", system_prompt="Be careful and concise.")
runner = Runner(agent, llm_call=llm)
print(runner.run("Hello"))
print(runner.get_trace_summary())
```

### Compaction bench (offline)

```bash
PYTHONPATH=src python scripts/bench_compaction.py
```

See [docs/TESTING.md](docs/TESTING.md).

---

## Design partners (paid, this week)

If production agents are burning tokens and you want a focused audit / setup:

→ **[DESIGN_PARTNER.md](DESIGN_PARTNER.md)**  
→ [Open a Design Partner issue](https://github.com/beta-tester-code/agentforge/issues/new?template=design_partner.md) (label `design-partner`)

| Package | Price | After you apply |
|---------|-------|-----------------|
| Audit only | USD 150 | Reply in 1 business day; invoice / Stripe link; work after pay |
| Audit + setup | USD 400 | Same |

Opening an issue is the application, not a charge. Email fallback: `eron6237@gmail.com`.

**Later:** hosted traces / team observability on the open-source core.

---

## Design Principles

1. Measure everything that costs tokens
2. Prefer reversible operations (offload before destroy)
3. Keep the recent tail lossless
4. Minimal surface area
5. Open by default — do not claim traction we do not have

---

## License

MIT
