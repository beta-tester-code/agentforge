# Architecture

AgentForge is a **small agent runtime**, not an orchestration framework.

```
User input
   │
   ▼
┌──────────┐     ┌─────────────┐     ┌──────────┐
│  Runner  │────▶│  LLM call   │────▶│  parse   │
│  loop    │◀────│  (any API)  │     │ tool call│
└────┬─────┘     └─────────────┘     └────┬─────┘
     │                                    │
     │         ┌──────────────┐           │
     ├────────▶│ ToolRegistry │◀──────────┘
     │         └──────────────┘
     │
     ▼
┌──────────────┐    threshold / pressure
│   Memory     │──────────────────────────▶ Compaction
│ messages+    │   offload → summarize     (iterative)
│ offload store│   token-budget tail
└──────┬───────┘
       ▼
┌──────────────┐
│    Tracer    │──▶ RunResult (reply + metrics + steps)
└──────────────┘
```

## Design contracts

1. **Measure** — every user/tool/assistant turn records tokens when possible.
2. **Reversible first** — large tool results are offloaded with `recall_offload`, not deleted.
3. **Iterative summary** — second compaction carries forward the first summary.
4. **Tail policy** — protect recent context by **token budget**, fall back to message count.
5. **Schema budget** — tool descriptions are capped so unused tools cannot tax every turn.
6. **Observable** — `RunResult` + full step trace are the foundation for hosted traces later.

## What we intentionally skip (for now)

- Multi-agent graphs / supervisor topologies
- Built-in vector store
- Provider-specific SDKs as hard deps (OpenAI-compatible HTTP only)
