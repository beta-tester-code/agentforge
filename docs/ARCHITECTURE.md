# Architecture

```
input → Runner → LLM → tool parse?
                ↘ tools → memory
memory → (pressure) → offload / compact → next turn
                     → Tracer → RunResult
```

**Runner** owns the step loop, prompt assembly, and stop conditions (max steps, tool error streak, normal completion).

**Memory** holds messages plus an offload store for large tool bodies. Serializable via `to_dict` / `from_dict`.

**Compaction** runs offload first, then iterative summarization when over threshold. Recent tail is chosen by token budget, with a message-count fallback so small chats still compact when needed.

**Tracer** records per-step actions. **RunResult** is the public snapshot after `run_detailed`.

**ToolRegistry** can cap description length so a long tool list does not dominate every prompt.

Dependencies stay thin: httpx, pydantic, tiktoken. Any OpenAI-compatible chat endpoint works through `make_llm_call`.
