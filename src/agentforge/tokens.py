"""Token counting utilities."""

from __future__ import annotations

from typing import Callable


def _fallback_estimate(text: str) -> int:
    """Rough estimate: ~4 characters per token."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def get_token_counter(model: str = "gpt-4o") -> Callable[[str], int]:
    """Return a token counting function.

    Tries to use tiktoken when available. Falls back to a cheap estimate otherwise.
    """
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")

        def count(text: str) -> int:
            if not text:
                return 0
            return len(enc.encode(text))

        return count
    except ImportError:
        return _fallback_estimate
