"""Demonstrate tool-schema budget (avoid multi-KB unused tool tax)."""

from agentforge import ToolRegistry


def main() -> None:
    tools = ToolRegistry()
    for i in range(40):
        tools.register(
            f"tool_{i}",
            f"Long description for tool {i} that would otherwise bloat every prompt "
            + ("detail " * 15),
            lambda i=i: i,
        )

    full = tools.descriptions()
    capped = tools.descriptions(max_chars=800)
    print(f"full schema chars:  {len(full)}")
    print(f"capped schema chars:{len(capped)}")
    print(f"reduction:          {100 * (1 - len(capped) / len(full)):.1f}%")
    print("--- capped preview ---")
    print(capped[:400], "...")


if __name__ == "__main__":
    main()
