# AgentForge

**Open-source toolkit for building reliable, token-efficient AI agents.**

Focus:

- Extreme **token efficiency**
- Strong **context control** (less drift / context rot)
- Lightweight **observability**
- Minimal runtime overhead

Not another heavy framework. A small, measurable toolkit.

---

## Status

**v0.1.0 — usable MVP**

| Feature | Status |
|---------|--------|
| Agent + Memory + Tracer | ✅ |
| Token counting | ✅ |
| Tool registry + multi-step tool loop | ✅ |
| Tool-result offloading + summarization compaction | ✅ |
| OpenAI-compatible client (OpenAI / Groq / Gemini / …) | ✅ |
| CLI with provider presets | ✅ |
| Mock + optional live tests | ✅ |

---

## Install

```bash
git clone https://github.com/beta-tester-code/agentforge.git
cd agentforge
pip install -e ".[dev]"
```

Or without install:

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

### Tests

```bash
PYTHONPATH=src pytest -q                        # offline
AGENTFORGE_LIVE=1 PYTHONPATH=src pytest -q      # needs key in env
```

See [docs/TESTING.md](docs/TESTING.md).

---

## Design partners (paid)

If your agents are burning tokens and you want a focused audit / setup:

→ **[DESIGN_PARTNER.md](DESIGN_PARTNER.md)**  
→ Open an issue with title `[design-partner] ...`

| Package | Price |
|---------|-------|
| Audit only | USD 150 |
| Audit + setup | USD 400 |

**Later:** hosted traces / team observability on top of the open-source core. Design partners shape that roadmap.

---

## Design Principles

1. Measure everything that costs tokens  
2. Prefer reversible operations (offload before destroy)  
3. Keep the recent tail lossless  
4. Minimal surface area  
5. Open by default  

---

## License

MIT
