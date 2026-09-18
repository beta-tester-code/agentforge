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


def cmd_version(_: argparse.Namespace) -> int:
    print(f"agentforge {__version__}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    prompt = args.prompt
    if not prompt:
        print("Error: provide a prompt with -p/--prompt", file=sys.stderr)
        return 2

    agent = Agent(
        name="cli",
        goal="Answer the user helpfully and concisely.",
        system_prompt="You are a careful, concise assistant.",
    )

    llm_call = None
    if os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or args.api_key:
        llm_call = make_llm_call(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
        )

    runner = Runner(
        agent,
        llm_call=llm_call,
        compaction_config=CompactionConfig(max_tokens=args.max_tokens),
        model_name=args.model,
    )

    reply = runner.run(prompt, max_steps=args.max_steps)
    print(reply)

    if args.trace:
        print("\n--- trace ---", file=sys.stderr)
        print(runner.get_trace_summary(), file=sys.stderr)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentforge",
        description="Token-efficient, observable AI agents",
    )
    parser.add_argument("--version", action="store_true", help="Show version and exit")

    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run a one-shot agent prompt")
    run_p.add_argument("-p", "--prompt", required=True, help="User prompt")
    run_p.add_argument("-m", "--model", default="gpt-4o-mini", help="Model name")
    run_p.add_argument(
        "--base-url",
        default="https://api.openai.com/v1",
        help="OpenAI-compatible API base URL",
    )
    run_p.add_argument("--api-key", default=None, help="API key (or use env)")
    run_p.add_argument("--max-steps", type=int, default=8)
    run_p.add_argument("--max-tokens", type=int, default=32_000)
    run_p.add_argument("--trace", action="store_true", help="Print trace summary")
    run_p.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        return cmd_version(args)

    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
