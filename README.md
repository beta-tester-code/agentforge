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
| Token counting (tiktoken / fallback) | ✅ |
| Tool registry + multi-step tool loop | ✅ |
| Tool-result offloading | ✅ |
| LLM summarization compaction | ✅ |
| OpenAI-compatible LLM client | ✅ |
| CLI (`agentforge run`) | ✅ |
| Basic tests | ✅ |

---

## Install

```bash
git clone https://github.com/beta-tester-code/agentforge.git
cd agentforge
pip install -e ".[dev]"
```

---

## Quick Start

### Python

```bash
python examples/basic_agent.py
python examples/with_tools.py

export OPENAI_API_KEY=sk-...
python examples/with_tools.py
```

### CLI

```bash
agentforge --version

# skeleton (no key)
agentforge run -p "Hello"

# real model
export OPENAI_API_KEY=sk-...
agentforge run -p "What is context rot?" --trace

# other OpenAI-compatible providers
agentforge run -p "Hi" --base-url https://api.groq.com/openai/v1 -m llama-3.3-70b-versatile
```

### Library

```python
from agentforge import Agent, Runner, make_llm_call, ToolRegistry

llm = make_llm_call(model="gpt-4o-mini")
agent = Agent(
    name="demo",
    goal="Be helpful and concise",
    system_prompt="You are a careful assistant.",
)
runner = Runner(agent, llm_call=llm)
print(runner.run("Hello"))
print(runner.get_trace_summary())
```

### Tests

```bash
pytest -q
```

---

## Design Principles

1. Measure everything that costs tokens  
2. Prefer reversible operations (offload before destroy)  
3. Keep the recent tail lossless  
4. Minimal surface area  
5. Open by default  

---

## Layout

```text
src/agentforge/
├── agent.py
├── memory.py
├── observability.py
├── runner.py
├── compaction.py
├── tokens.py
├── tools.py
├── llm.py
└── cli.py
```

---

## License

MIT
