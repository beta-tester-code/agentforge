"""Context pressure levels — decide *when* to act, not only *that* tokens exist."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PressureLevel:
    name: str
    ratio: float  # tokens / max_tokens


def pressure_level(tokens: int, max_tokens: int) -> PressureLevel:
    if max_tokens <= 0:
        return PressureLevel("unknown", 0.0)
    ratio = tokens / max_tokens
    if ratio >= 0.99:
        return PressureLevel("critical", ratio)
    if ratio >= 0.90:
        return PressureLevel("aggressive", ratio)
    if ratio >= 0.80:
        return PressureLevel("mask", ratio)
    if ratio >= 0.70:
        return PressureLevel("warn", ratio)
    return PressureLevel("ok", ratio)
