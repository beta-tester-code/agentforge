# Notes for publishing

Primary surface is the GitHub repo.

**PyPI:** the name `agentforge` is already taken (DataBassGit / GPL low-code framework). Do not publish this package as `agentforge`. Import can stay `agentforge`; distribution name should be distinct, e.g. `agentforge-compact` or `tokencraft-agentforge`. Confirm availability on pypi.org before first upload. Human must create the PyPI account and token.

Suggested one-liner:

```text
agentforge (beta-tester-code) — Python helpers for cheaper, inspectable agent context
```

Avoid claiming stars, customers, or third-party benchmarks. Offline checks live under `scripts/` and `examples/`.

Near-term cash: [DESIGN_PARTNER.md](../DESIGN_PARTNER.md) ($150 audit / $400 audit+setup). Hosted traces later.
