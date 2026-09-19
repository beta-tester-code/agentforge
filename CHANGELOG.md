# Changelog

## 0.4.0

- `BoundLLM` / `bind_llm` — attach tool schemas to every HTTP turn
- `TraceStore` — append-only JSONL of `RunResult`
- `run_subagent` / `subagent_tool` — isolated child window
- `hard_token_budget` stop reason
- Native tool results carry `tool_call_id` when present

## 0.3.0

- Native tool_calls, schema from hints, compact_now, prefix-cache, disk offload, eval, stream/cancel

## 0.2.0

- RunResult, memory serialization, pressure, error streak, CI

## 0.1.0

- Initial loop
