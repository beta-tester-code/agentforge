# Background

Most agent stacks optimize for graph features and multi-agent wiring. The bill often comes from somewhere else: tool payloads left in the window, system text repeated every call, and summaries that wipe state on the second compaction.

AgentForge is a small loop that treats that as the main problem.

- Count tokens on the way in and out
- Offload large tool results instead of truncating blindly
- Summarize older turns without discarding the previous summary wholesale
- Keep recent context by token budget
- Emit a step trace you can inspect locally (and host later if it earns its keep)

It will not replace LangGraph for complex graphs. It is meant to sit next to an existing loop when context cost and opacity are the issue.

Same name as a few other projects on the internet. This repo is only the Python toolkit here, Apache-2.0.
