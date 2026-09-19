"""Agent runner composed from mixins."""
from __future__ import annotations

from agentforge._init_mixin import _init_mixin
from agentforge._compact_mixin import _compact_mixin
from agentforge._run_mixin import _run_mixin
from agentforge.tool_parse import (  # re-export for tests
    _parse_tool_call,
    _parse_all_tool_calls,
    _native_tool_calls,
    _loads_obj,
)

class Runner(_init_mixin, _compact_mixin, _run_mixin):
    pass

__all__ = [
    "Runner",
    "_parse_tool_call",
    "_parse_all_tool_calls",
    "_native_tool_calls",
    "_loads_obj",
]
