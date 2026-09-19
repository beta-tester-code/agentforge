# Fixed-scope audits

For teams that already run tool-using agents and want a concrete look at token use and context handling.

Example of the written deliverable: [docs/SAMPLE_AUDIT.md](docs/SAMPLE_AUDIT.md).

## Scope

**Audit (USD 150)**

- One existing workflow (LangGraph, CrewAI, or custom loop)
- Written token map: where spend goes by step class
- What is safe to offload vs must-keep
- Short integration plan for this library *if it fits* (or an explicit no-fit)

**Audit + setup (USD 400)**

Same as above, plus one remote session to wire AgentForge into that workflow and a short follow-up writeup.

Turnaround is usually a few business days after materials arrive. The library stays Apache-2.0 either way.

Not included: rewriting your product, multi-week retainers, hosted observability (later, not this week).

## What I need from you

- A production or staging agent that does multi-step tool calls
- Rough monthly token spend or request volume (order of magnitude is enough)
- The failure mode you care about (cost, drift, hard-to-debug traces)
- Optional: a sanitized message dump or trace from one painful run

## How to start

Open an issue with the [audit request template](https://github.com/beta-tester-code/agentforge/issues/new?template=design_partner.md), or title it:

```text
[design-partner] one-line description
```

I reply on the issue. Payment is invoice or Stripe link after we confirm fit. Opening an issue does not charge anything.

Email if GitHub is awkward: eron6237@gmail.com
