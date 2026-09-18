# Testing strategy for AgentForge

## Layer 1 — Always available (no keys, no network)

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python scripts/bench_compaction.py
PYTHONPATH=src python examples/basic_agent.py
```

Includes unit tests, mock tool-loop tests, and compaction benchmark.

## Layer 2 — Live cloud APIs (optional)

Requires a key in the **environment** (never commit keys).

### Verified: Google Gemini (OpenAI-compatible)

```bash
export OPENAI_API_KEY="your-google-ai-studio-key"
export AGENTFORGE_LIVE=1
export AGENTFORGE_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
export AGENTFORGE_MODEL="gemini-3.5-flash"

PYTHONPATH=src pytest -q tests/test_live_optional.py
```

Notes (validated 2026-09):

- `gemini-2.5-flash` may return 404 for new users; prefer `gemini-3.5-flash` / `gemini-3.1-flash-lite`.
- Tool loop with text `TOOL_CALL` format works with Gemini.

CLI:

```bash
export OPENAI_API_KEY=...
PYTHONPATH=src python -m agentforge.cli run --provider gemini -p "Hello" --trace
```

### Other providers

| Provider | Base URL | Notes |
|----------|----------|-------|
| Groq | `https://api.groq.com/openai/v1` | Fast; `--provider groq` |
| OpenRouter | `https://openrouter.ai/api/v1` | Prefer `:free` model ids |
| Ollama local | `http://localhost:11434/v1` | `api_key=ollama` |

## Layer 3 — Local Ollama

Only if Ollama is installed on the machine. Not available in the default remote sandbox.

## Policy

- Default CI = Layer 1 only
- Live tests are opt-in via `AGENTFORGE_LIVE=1`
- Never paste API keys into git, issues, or chat if avoidable; rotate if exposed
