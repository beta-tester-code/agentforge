# Why AgentForge exists

## The pain (2026)

Public write-ups keep repeating the same production facts:

- CrewAI injects role/goal/backstory on **every** LLM call. That is structural token tax, not a bug.
- LangGraph is cheaper *if* you control every node; most teams still dump tool JSON and full history back into the window.
- Multi-agent frameworks compact **per agent, independently** — no coordinated context policy.
- Headroom-style proxies report ~70% prompt-token cuts on tool-heavy workloads with no accuracy drop (Google Cloud / Gemini measurements, Sep 2026).
- Framework surveys still list **context-rot awareness** as something almost nobody ships as a first-class loop.

AgentForge is a small Python loop that treats those as the product:

1. Count tokens every turn.
2. Offload oversized tool results.
3. Compact with a lossless recent tail.
4. Keep a local trace you can later host.

## What this is not

Name collision is severe. Unrelated projects also called AgentForge:

- DataBassGit/AgentForge (YAML cogs / AGI framework)
- agentforge.dev and Agentman testing product
- @agentforge/cli npm packages
- ICP / FORGE token protocol
- SWE-bench research paper of the same name
- Mohit Goyal's PyPI coding-agent harness (different author)

If you landed here from those, this repo is a **token-efficiency + observability toolkit**, MIT, Python 3.11+.

## Path to money

- Open-source core stays free.
- Near-term cash: fixed-price design-partner audit ($150) or audit+setup ($400).
- Later: hosted traces (LangSmith-shaped, smaller).

No claimed customers until an issue or invoice exists.
