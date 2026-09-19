# Changelog

## 0.3.0

- Native OpenAI-style `tool_calls` in the runner (dict messages)
- JSON Schema from type hints (`schema_from_callable`)
- `compact_now` tool for model-requested compaction
- Prefix-cache aware policy (`respect_prefix_cache` / `offload_only_until_ratio`)
- Disk offload store (`offload_dir=`)
- `cancel_check` for cooperative cancel
- Streaming helper on `OpenAICompatibleClient.chat_stream`
- Offline `scripts/eval_harness.py`

## 0.2.0

- RunResult, memory serialization, pressure levels, error streak, CI

## 0.1.1

- Tool schema budget, Apache-2.0

## 0.1.0

- Initial loop: tools, compaction, offload, CLI, tests
