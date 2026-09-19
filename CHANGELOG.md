# Changelog

## 0.2.0

- `RunResult` + `Runner.run_detailed()` / `last_result`
- Serializable `Memory` / `Message` (`to_dict` / `from_dict`)
- Context pressure levels (`ok` → `critical`) logged on warn+
- Stop on consecutive tool error streak (`max_tool_errors`)
- `Runner.reset()` for clean multi-run sessions
- GitHub Actions CI (3.11 / 3.12)
- `docs/ARCHITECTURE.md`

## 0.1.1

- Tool schema budget, selective tools demo
- LLM client `tools` payload helper
- Apache-2.0 license

## 0.1.0

- Initial MVP: agent loop, compaction, offload, CLI, tests
