# AgentForge

**Open-source toolkit for building reliable, token-efficient AI agents.**

AgentForge focuses on the hard problems that still break most agent systems in production:

- Extreme **token efficiency** (less context, same or better results)
- Strong **context control** (reduce drift, rot, and goal loss)
- Lightweight, high-quality **observability**
- Minimal runtime overhead

This is not another general-purpose agent framework.  
It is an engineering-focused toolkit designed to make agents cheaper to run and harder to break.

---

## Status

**Early development / MVP phase**

Current goal: deliver a usable core by the end of the first development weekend that demonstrates:

1. Step-by-step agent execution
2. Basic context compaction + tool-result offloading
3. Clear per-step token tracking
4. Inspectable state at any point

---

## Design Principles

1. **Measure everything that costs tokens**  
   Every decision that touches the context window must be visible and countable.

2. **Prefer reversible operations**  
   Offload before you destroy. Summarize only when recovery is no longer needed.

3. **Keep the recent tail lossless**  
   The last N turns stay intact. Older content is compacted or moved out.

4. **Minimal surface area**  
   Small core. Clear extension points. No heavy framework tax.

5. **Open by default**  
   Core is fully open source. Hosted/paid layers only appear after proven value.

---

## Planned MVP Scope

- Define an agent (goal + system prompt + tools)
- Execute step-by-step with explicit state
- Simple memory with compaction triggers
- Tool-result offloading (store large outputs externally, keep pointer + preview)
- Per-step token accounting
- CLI entry point + one clear example
- Clean documentation

Out of scope for MVP:
- Complex multi-agent orchestration
- Fancy UI
- Authentication / multi-tenant
- Payment systems

---

## Project Structure (target)

```text
agentforge/
├── README.md
├── pyproject.toml
├── .gitignore
├── src/
│   └── agentforge/
│       ├── __init__.py
│       ├── agent.py
│       ├── memory.py
│       ├── tools.py
│       ├── runner.py
│       ├── observability.py
│       └── compaction.py
├── examples/
│   └── basic_agent.py
├── tests/
└── docs/
