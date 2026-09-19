"""AgentForge - Token-efficient, observable AI agents."""

from agentforge.agent import Agent
from agentforge.bound_llm import BoundLLM, bind_llm
from agentforge.compaction import CompactionConfig, compact_memory, make_llm_summarizer
from agentforge.llm import (
    LLMError,
    OpenAICompatibleClient,
    make_llm_call,
    make_llm_call_with_tools,
    tools_to_openai_schema,
)
from agentforge.memory import Memory, Message
from agentforge.observability import Tracer
from agentforge.offload_store import DiskOffloadStore, MemoryOffloadStore
from agentforge.pressure import PressureLevel, pressure_level
from agentforge.result import RunResult
from agentforge.runner import Runner
from agentforge.schema import schema_from_callable
from agentforge.subagent import run_subagent, subagent_tool
from agentforge.tools import Tool, ToolRegistry
from agentforge.tokens import get_token_counter
from agentforge.trace_store import TraceStore

__version__ = "0.4.0"

__all__ = [
    "Agent",
    "Memory",
    "Message",
    "Tracer",
    "Runner",
    "RunResult",
    "CompactionConfig",
    "compact_memory",
    "make_llm_summarizer",
    "Tool",
    "ToolRegistry",
    "get_token_counter",
    "OpenAICompatibleClient",
    "make_llm_call",
    "make_llm_call_with_tools",
    "tools_to_openai_schema",
    "schema_from_callable",
    "LLMError",
    "pressure_level",
    "PressureLevel",
    "DiskOffloadStore",
    "MemoryOffloadStore",
    "BoundLLM",
    "bind_llm",
    "TraceStore",
    "run_subagent",
    "subagent_tool",
    "__version__",
]
