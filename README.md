# AgentForge

**Open-source toolkit for building reliable, token-efficient AI agents.**

AgentForge focuses on the hard problems that still break most agent systems in production:

- Extreme **token efficiency**
- Strong **context control** (reduce drift and context rot)
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
| Basic tests | ✅ |

---

## Quick Start

```bash
git clone https://github.com/beta-tester-code/agentforge.git
cd agentforge
pip install -e ".[dev]"

# skeleton (no key)
python examples/basic_agent.py

# with tools
python examples/with_tools.py

# real model
export OPENAI_API_KEY=sk-...
python examples/with_tools.py

# tests
pytest -q
```

Any OpenAI-compatible endpoint works:

```python
from agentforge import Agent, Runner, make_llm_call, ToolRegistry

llm = make_llm_call(
    base_url="https://api.groq.com/openai/v1",
    model="llama-3.3-70b-versatile",
)

tools = ToolRegistry()
# tools.register("name", "description", func)

agent = Agent(
    name="demo",
    goal="Be helpful and concise",
    system_prompt="You are a careful assistant.",
)

runner = Runner(agent, tools=tools, llm_call=llm)
print(runner.run("Hello"))
print(runner.get_trace_summary())
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
└── llm.py
```

---

## License

MIT
