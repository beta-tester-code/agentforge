"""Tool call parsing."""

from agentforge.runner import _parse_tool_call


def test_parse_valid_tool_call():
    text = """
Some reasoning...

TOOL_CALL
name: get_current_utc_time
args: {}
END_TOOL_CALL
"""
    parsed = _parse_tool_call(text)
    assert parsed is not None
    name, args = parsed
    assert name == "get_current_utc_time"
    assert args == {}


def test_parse_tool_call_with_args():
    text = """
TOOL_CALL
name: search
args: {"query": "agent frameworks"}
END_TOOL_CALL
"""
    parsed = _parse_tool_call(text)
    assert parsed is not None
    name, args = parsed
    assert name == "search"
    assert args["query"] == "agent frameworks"


def test_parse_fenced_json_tool_call():
    text = """
I'll use a tool.

```json
{"name": "add", "args": {"a": 1, "b": 2}}
```
"""
    parsed = _parse_tool_call(text)
    assert parsed is not None
    name, args = parsed
    assert name == "add"
    assert args == {"a": 1, "b": 2}


def test_parse_no_tool_call():
    assert _parse_tool_call("Just a normal answer.") is None
