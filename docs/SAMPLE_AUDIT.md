# Sample audit deliverable

This is the shape of a $150 written audit. Numbers below are placeholders so you can see the sections. A real audit uses your traces or a sanitized run, not invented savings.

## 1. Workflow under review

- Stack: e.g. LangGraph + tools that return raw HTML/JSON
- Loop length: N steps typical
- Pain: context fills with tool dumps; hard to see which step burned tokens

## 2. Where tokens go (by step class)

| Class | What repeats | Why it grows |
| --- | --- | --- |
| System + tool schemas | Every step | Prefix-cache friendly if stable; still billed if the rest of the prompt shifts |
| Tool results | Full payloads kept in history | Usually the leak |
| Prior reasoning | Uncompacted assistant text | Second leak |

Fill this from `RunResult` / JSONL traces or from a dump of messages you already log.

## 3. Safe offload vs must-keep

- Offload: large search/HTML/JSON tool bodies; keep a short preview + recall id
- Keep: original task message, last few turns within a token budget, decisions that later tools depend on
- Do not summarize away IDs, URLs, or error text the next tool needs

## 4. Integration plan (this library, if it fits)

1. Wrap the existing LLM call; keep your graph/crew.
2. Enable `recall_offload` + optional `offload_dir`.
3. Pin task head; compact tail by token budget.
4. Write JSONL traces; review one bad run before changing prompts.
5. Only then consider `compact_now` or a hard budget stop.

If the fit is poor (you need a full graph product, or JS-only), the writeup says so and still maps the token leak.

## 5. What you walk away with

- One-page token map for *your* workflow
- A short list of changes ranked by risk
- Optional: $400 add-on is one remote session to wire the runner and a follow-up note

See [DESIGN_PARTNER.md](../DESIGN_PARTNER.md).
