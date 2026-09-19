# Tests

## Offline (default)

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python scripts/bench_compaction.py
PYTHONPATH=src python examples/token_savings_demo.py
```

Covers memory, compaction, tool parse, mock tool loop, RunResult, pressure levels.

## Live (optional)

```bash
export OPENAI_API_KEY=...
export AGENTFORGE_LIVE=1
export AGENTFORGE_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
export AGENTFORGE_MODEL=gemini-3.5-flash
PYTHONPATH=src pytest -q tests/test_live_optional.py
```

Only runs when `AGENTFORGE_LIVE=1` is set. Do not commit keys.

## Providers

| Provider | Base URL |
|----------|----------|
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| Groq | `https://api.groq.com/openai/v1` |
| OpenRouter | `https://openrouter.ai/api/v1` |

CLI presets: `--provider gemini|groq|openai|openrouter`.
