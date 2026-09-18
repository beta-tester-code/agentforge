# Testing strategy for AgentForge

## Layer 1 — Always available (no keys, no network)

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python scripts/bench_compaction.py
PYTHONPATH=src python examples/basic_agent.py
```

Includes:

- unit tests (memory, compaction, parse)
- **mock LLM tool-loop integration tests** (`tests/test_tool_loop_mock.py`)
- compaction benchmark

This layer is the default CI path.

## Layer 2 — Free cloud APIs (optional, needs a key)

AgentForge already speaks OpenAI-compatible HTTP. Preferred free options:

| Provider | Env var | Base URL | Notes |
|----------|---------|----------|-------|
| **Groq** | `GROQ_API_KEY` or `LLM_API_KEY` | `https://api.groq.com/openai/v1` | Fast, good for agent loops |
| **Google AI Studio** | key via Gemini OpenAI-compat proxy or custom | check current docs | Generous free tier |
| **OpenRouter free models** | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` | Use model ids with `:free` |

Example:

```bash
export LLM_API_KEY=...          # or GROQ_API_KEY
export OPENAI_API_KEY=$LLM_API_KEY  # our client also reads this

python -c "
from agentforge import Agent, Runner, make_llm_call
llm = make_llm_call(
    base_url='https://api.groq.com/openai/v1',
    model='llama-3.3-70b-versatile',
)
r = Runner(Agent('t','goal','You are concise.'), llm_call=llm)
print(r.run('Say hi in 5 words'))
print(r.get_trace_summary())
"
```

**Human step only:** create the free key and connect/export it.  
Everything after that is automated.

## Layer 3 — Local (Ollama)

If Ollama is installed on a machine:

```bash
ollama pull llama3.2

python -c "
from agentforge import Agent, Runner, make_llm_call
llm = make_llm_call(base_url='http://localhost:11434/v1', api_key='ollama', model='llama3.2')
print(Runner(Agent('t','g','Be brief.'), llm_call=llm).run('Hello'))
"
```

Not available in the default remote sandbox (no Ollama binary).

## What we do *not* depend on

- Paid OpenAI credits for core correctness
- MCP for unit/integration tests
- Manual clicking beyond pasting a free API key into the environment
