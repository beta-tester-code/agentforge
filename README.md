# AgentForge

**Open-source toolkit for building reliable, token-efficient AI agents.**

AgentForge focuses on the hard problems that still break most agent systems in production:

- Extreme **token efficiency** (less context, same or better results)
- Strong **context control** (reduce drift, rot, and goal loss)
- Lightweight, high-quality **observability**
- Minimal runtime overhead

This is not another general-purpose agent framework.  
It is an engineering-focused toolkit designed to make agents cheaper to run and harder to break.

---

## Status

**Early development (v0.1.0)**

Working right now:

- Agent definition
- Memory + per-message token tracking
- Compaction primitives (tool-result offloading)
- Tracer / step observability
- OpenAI-compatible LLM client
- Runner that can call a real model when an API key is present

Coming next:

- Multi-step tool calling loop
- Real summarization-based compaction
- Better examples and docs

---

## Design Principles

1. **Measure everything that costs tokens**
2. **Prefer reversible operations** (offload before destroy)
3. **Keep the recent tail lossless**
4. **Minimal surface area**
5. **Open by default**

---

## Quick Start

```bash
# clone
git clone https://github.com/beta-tester-code/agentforge.git
cd agentforge

# install (editable)
pip install -e .

# skeleton mode (no key needed)
python examples/basic_agent.py

# real model
export OPENAI_API_KEY=sk-...
python examples/basic_agent.py
```

You can also point to any OpenAI-compatible endpoint:

```python
from agentforge import Agent, Runner, make_llm_call

llm = make_llm_call(
    base_url="https://api.groq.com/openai/v1",
    model="llama-3.3-70b-versatile",
)

agent = Agent(
    name="demo",
    goal="Be helpful and concise",
    system_prompt="You are a careful assistant.",
)

runner = Runner(agent, llm_call=llm)
print(runner.run("Hello"))
print(runner.get_trace_summary())
```

---

## Project Structure

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

MIT (planned)
