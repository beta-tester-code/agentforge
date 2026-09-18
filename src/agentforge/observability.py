"""Lightweight token and step tracking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class StepRecord:
    step: int
    action: str
    input_tokens: int = 0
    output_tokens: int = 0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Tracer:
    """Collects per-step metrics for later inspection."""

    records: list[StepRecord] = field(default_factory=list)

    def record(
        self,
        step: int,
        action: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        **details: Any,
    ) -> None:
        self.records.append(
            StepRecord(
                step=step,
                action=action,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                details=details,
            )
        )

    def total_tokens(self) -> tuple[int, int]:
        inp = sum(r.input_tokens for r in self.records)
        out = sum(r.output_tokens for r in self.records)
        return inp, out

    def summary(self) -> dict[str, Any]:
        inp, out = self.total_tokens()
        by_action: dict[str, int] = {}
        for r in self.records:
            by_action[r.action] = by_action.get(r.action, 0) + 1
        return {
            "steps": len(self.records),
            "input_tokens": inp,
            "output_tokens": out,
            "total_tokens": inp + out,
            "by_action": by_action,
        }

    def steps(self) -> list[dict[str, Any]]:
        """Full step list for debugging / future hosted traces."""
        out: list[dict[str, Any]] = []
        for r in self.records:
            item = asdict(r)
            out.append(item)
        return out
