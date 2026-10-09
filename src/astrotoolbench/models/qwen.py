"""Decode the tool-call tags emitted by the pinned Qwen chat template."""

import json
from uuid import uuid4

from ..benchmark.loader import _unique_object
from .base import BackendProtocolError, ToolCall


def parse_tool_calls(text: str) -> tuple[str | None, tuple[ToolCall, ...]]:
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object)
    calls, content, cursor = [], [], 0
    try:
        while (start := text.find("<tool_call>", cursor)) != -1:
            content.append(text[cursor:start])
            body_start = start + len("<tool_call>")
            while body_start < len(text) and text[body_start].isspace():
                body_start += 1
            payload, length = decoder.raw_decode(text[body_start:])
            end = body_start + length
            while end < len(text) and text[end].isspace():
                end += 1
            if not text.startswith("</tool_call>", end):
                raise ValueError("Missing tool-call closing tag.")
            if not isinstance(payload, dict) or set(payload) != {"name", "arguments"}:
                raise ValueError("Expected a function name and arguments.")
            arguments = payload["arguments"]
            if not isinstance(arguments, (dict, str)):
                raise ValueError("Expected an argument object or JSON string.")
            calls.append(ToolCall(
                id="local-" + uuid4().hex, name=payload["name"],
                arguments=arguments if isinstance(arguments, str) else json.dumps(arguments, allow_nan=False),
            ))
            cursor = end + len("</tool_call>")
        content.append(text[cursor:])
        remainder = "".join(content).strip()
        if "</tool_call>" in remainder:
            raise ValueError("Unexpected tool-call closing tag.")
        return (remainder or None, tuple(calls)) if calls else (text, ())
    except (ValueError, TypeError, RecursionError) as error:
        raise BackendProtocolError("Invalid local tool-call response.", raw_response={"content": text}) from error
