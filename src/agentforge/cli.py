"""Minimal CLI for AgentForge."""

from __future__ import annotations

import argparse
import os
import sys

from agentforge import __version__
from agentforge.agent import Agent
from agentforge.compaction import CompactionConfig
from agentforge.llm import make_llm_call
from agentforge.runner import Runner

PRESETS: dict[str, dict[str, str]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-3.6-flash",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openrouter/auto",
    },
}


def cmd_version(_: argparse.Namespace) -> int:
    print(f"agentforge {__version__}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    prompt = args.prompt
    if not prompt:
        print("Error: provide a prompt with -p/--prompt", file=sys.stderr)
        return 2

    base_url = args.base_url
    model = args.model
    if args.provider:
        preset = PRESETS.get(args.provider.lower())
        if not preset:
            print(
                f"Unknown provider {args.provider!r}. "
                f"Choose from: {', '.join(PRESETS)}",
                file=sys.stderr,
            )
            return 2
        if args.base_url == PRESETS["openai"]["base_url"]:
            base_url = preset["base_url"]
        if args.model == PRESETS["openai"]["model"]:
            model = preset["model"]

    agent = Agent(
        name="cli",
        goal="Answer the user helpfully and concisely.",
        system_prompt="You are a careful, concise assistant.",
    )

    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    llm_call = None
    if api_key:
        llm_call = make_llm_call(api_key=api_key, base_url=base_url, model=model)

    runner = Runner(
        agent,
        llm_call=llm_call,
        compaction_config=CompactionConfig(max_tokens=args.max_tokens),
        use_llm_summarizer=bool(llm_call) and not args.no_summarizer,
    )
    result = runner.run_detailed(prompt, max_steps=args.max_steps)
    print(result.reply)
    if args.trace:
        import json

        print("--- trace ---", file=sys.stderr)
        print(json.dumps(result.to_dict(), indent=2, default=str), file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agentforge", description="AgentForge CLI")
    p.add_argument("--version", action="store_true")
    sub = p.add_subparsers(dest="cmd")

    run = sub.add_parser("run", help="Run a one-shot agent turn")
    run.add_argument("-p", "--prompt", required=True)
    run.add_argument("--provider", default=None, help="gemini | groq | openai | openrouter")
    run.add_argument("--base-url", default=PRESETS["openai"]["base_url"])
    run.add_argument("--model", default=PRESETS["openai"]["model"])
    run.add_argument("--api-key", default=None)
    run.add_argument("--max-steps", type=int, default=8)
    run.add_argument("--max-tokens", type=int, default=32_000)
    run.add_argument("--trace", action="store_true")
    run.add_argument("--no-summarizer", action="store_true")
    run.set_defaults(func=cmd_run)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "version", False) and not getattr(args, "cmd", None):
        return cmd_version(args)
    if not getattr(args, "cmd", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
