"""Build OpenAI-style JSON schemas from Python callables."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints


def _py_to_json_type(annotation: Any) -> dict[str, Any]:
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}
    origin = get_origin(annotation)
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return _py_to_json_type(args[0])
        return {}
    mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
    }
    if annotation in mapping:
        return dict(mapping[annotation])
    if origin is list:
        items = get_args(annotation)
        schema: dict[str, Any] = {"type": "array"}
        if items:
            schema["items"] = _py_to_json_type(items[0]) or {}
        return schema
    if origin is dict:
        return {"type": "object"}
    # string names from postponed annotations
    if isinstance(annotation, str):
        low = annotation.lower()
        if low in ("str", "string"):
            return {"type": "string"}
        if low in ("int", "integer"):
            return {"type": "integer"}
        if low in ("float", "number"):
            return {"type": "number"}
        if low in ("bool", "boolean"):
            return {"type": "boolean"}
    return {}


def schema_from_callable(func: Callable[..., Any]) -> dict[str, Any]:
    """Return a JSON Schema object for function parameters."""
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}
    sig = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        ann = hints.get(name, param.annotation)
        prop = _py_to_json_type(ann)
        if not prop:
            prop = {}
        properties[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    if not properties:
        schema["additionalProperties"] = True
    return schema


def openai_tool_from_callable(
    name: str,
    description: str,
    func: Callable[..., Any],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema_from_callable(func),
        },
    }
