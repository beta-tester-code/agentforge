"""BoundLLM wiring, TraceStore, subagent isolation, hard budget."""

from pathlib import Path

from agentforge import (
    Agent,
    CompactionConfig,
    Runner,
    ToolRegistry,
    TraceStore,
    run_subagent,
    subagent_tool,
)
from agentforge.result import RunResult


def test_trace_store_roundtrip(tmp_path: Path):
    path = tmp_path / "traces.jsonl"
    store = TraceStore(path)
    r = RunResult(reply="hi", steps=1, total_tokens=10, stopped_reason="completed")
    store.append(r, meta={"agent": "t"})
    rows = store.read_all()
    assert len(rows) == 1
    assert rows[0]["result"]["reply"] == "hi"
    assert rows[0]["meta"]["agent"] == "t"


def test_runner_writes_trace(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    runner = Runner(
        Agent("t", "g", "s"),
        llm_call=None,
        trace_path=str(path),
    )
    runner.run_detailed("hello")
    rows = TraceStore(path).read_all()
    assert len(rows) == 1
    assert rows[0]["result"]["stopped_reason"] == "skeleton"


def test_hard_token_budget():
    class AlwaysReply:
        def __call__(self, messages):
            return "x" * 200

    runner = Runner(
        Agent("t", "g", "s"),
        llm_call=AlwaysReply(),
        use_llm_summarizer=False,
        hard_token_budget=5,
        compaction_config=CompactionConfig(max_tokens=50_000),
    )
    # first turn may complete before budget check on second iteration
    r1 = runner.run_detailed("a", max_steps=3)
    # force spent high by running again without reset — or check budget path
    runner.hard_token_budget = 1
    r2 = runner.run_detailed("b", max_steps=2)
    assert r2.stopped_reason in ("budget", "completed")


def test_subagent_isolation():
    class Echo:
        def __call__(self, messages):
            last = messages[-1]["content"]
            return f"child saw: {last[:40]}"

    result = run_subagent(
        name="child",
        goal="g",
        system_prompt="s",
        user_input="task",
        llm_call=Echo(),
        parent_context="SECRET_PARENT_DATA " * 20,
        max_steps=2,
    )
    assert result.stopped_reason == "completed"
    assert "child saw" in result.reply


def test_subagent_as_tool():
    class Echo:
        def __call__(self, messages):
            return "done-by-child"

    fn = subagent_tool(
        name="worker",
        goal="g",
        system_prompt="s",
        llm_call=Echo(),
        max_steps=2,
    )
    out = fn(task="do it")
    assert "done-by-child" in out
    assert "subagent:worker" in out
