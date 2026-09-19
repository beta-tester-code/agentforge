# Where this sits

Not a LangGraph/CrewAI replacement. A small loop + compaction + local traces.

| Tool | Job | Compaction | Tokens measured | Drop-in next to an existing loop |
| --- | --- | --- | --- | --- |
| LangGraph | Graphs, checkpoints | You compose write/select/compress/isolate | If you add it | No — you adopt the graph |
| CrewAI | Role crews | Summarize on overflow (`respect_context_window`) | Weak | No |
| Headroom | Proxy that compresses prompts in front of a model | Lossy payload compression | Yes (proxy metrics) | Yes, as infra |
| TanStack AI `withCompaction` | Chat middleware | Evict / summarize / stub tool results | Estimate or custom | JS/TS chat |
| **This repo** | Python runner | Offload fat tool results, keep task head + token-budget tail, iterative summaries, optional `compact_now` | tiktoken per step + `RunResult` | Yes, next to a custom loop |

Demand signal (industry, not this repo): tool dumps are the token leak; naive summarization is lossy; O(N²) input cost on long sessions is the bill.

Use this if you already have a tool loop and want offload-before-summarize plus a step trace. Use LangGraph if the product *is* the graph. Use Headroom if you want a process-external compressor and will accept lossy JSON shrink.
