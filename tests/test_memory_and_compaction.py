"""Basic tests for memory and compaction."""

from agentforge import Memory, CompactionConfig, compact_memory


def test_memory_add_and_recent():
    mem = Memory()
    mem.add("user", "hello", tokens=2)
    mem.add("assistant", "hi there", tokens=3)
    assert len(mem.messages) == 2
    assert mem.recent(1)[0].content == "hi there"


def test_offload_large_tool_result():
    mem = Memory()
    big = "x" * 5000
    mem.add("tool", big, tokens=1200)
    mem.add("user", "ok", tokens=1)

    config = CompactionConfig(
        max_tokens=10_000,
        max_tool_result_chars=100,
        keep_recent_messages=2,
        summarize_threshold_ratio=0.99,
    )
    result = compact_memory(mem, config)

    assert any("offloaded" in a for a in result.actions)
    assert "offloaded:" in mem.messages[0].content
    assert len(mem.messages[0].content) < len(big)


def test_summarize_when_over_threshold():
    mem = Memory()
    for i in range(10):
        mem.add("user", f"message number {i} with some content", tokens=20)

    config = CompactionConfig(
        max_tokens=100,
        keep_recent_messages=3,
        summarize_threshold_ratio=0.5,
        offload_tool_results=False,
    )
    result = compact_memory(mem, config)

    assert result.messages_after < result.messages_before
    assert any("summarized" in a for a in result.actions)
    assert mem.messages[0].meta.get("type") == "compaction_summary"


def test_default_summarizer_saves_tokens():
    mem = Memory()
    for i in range(20):
        mem.add("user", f"Long user content {i} " + ("word " * 40), tokens=None)
        mem.add("assistant", f"Long assistant content {i} " + ("text " * 40), tokens=None)

    config = CompactionConfig(
        max_tokens=200,
        keep_recent_messages=2,
        summarize_threshold_ratio=0.3,
        offload_tool_results=False,
    )
    result = compact_memory(mem, config)
    assert result.tokens_after < result.tokens_before
    # Expect meaningful reduction with the aggressive default summarizer
    assert result.tokens_after < result.tokens_before * 0.5
