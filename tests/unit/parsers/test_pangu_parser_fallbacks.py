# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Fallback paths of the Pangu parser helpers.

Both modules swallow a malformed-JSON failure and degrade to plain text
instead of propagating the error to the streaming parser.
"""

import pytest

from vllm.parser.engine.events import EventType, SemanticEvent

import omni_npu.v1.parsers._pangu_parser_engine_config as cfg_mod
from omni_npu.v1.parsers._pangu_parser_engine_config import (
    PanguTerminals,
    pangu_tool_arg_converter,
)
from omni_npu.v1.parsers._pangu_tool_array_event_expander import (
    PanguToolArrayEventExpander,
)


TERMINALS = PanguTerminals(
    think_start="<|think_start|>",
    think_end="<|think_end|>",
    tool_start="<|tool_call_start|>",
    tool_end="<|tool_call_end|>",
)


def test_partial_parser_failure_returns_empty(monkeypatch):
    """partial_json_loads raising ValueError must degrade to ""."""

    def _boom(*_args, **_kwargs):
        raise ValueError("cannot recover")

    monkeypatch.setattr(cfg_mod, "partial_json_loads", _boom)
    assert pangu_tool_arg_converter('{"name": "f", "argum', partial=True) == ""


def test_partial_parser_type_error_returns_empty(monkeypatch):
    """TypeError from the partial parser is swallowed the same way."""

    def _boom(*_args, **_kwargs):
        raise TypeError("bad type")

    monkeypatch.setattr(cfg_mod, "partial_json_loads", _boom)
    assert pangu_tool_arg_converter("{oops", partial=True) == ""


def test_non_partial_broken_json_returns_empty():
    """Without partial mode the broken JSON short-circuits before the
    partial parser, so the fallback is not reached.
    """
    assert pangu_tool_arg_converter("{oops", partial=False) == ""


def test_valid_json_still_unwrapped():
    """Guard: the fallback must not shadow the normal path."""
    out = pangu_tool_arg_converter('{"name": "f", "arguments": {"a": 1}}',
                                   partial=False)
    assert "a" in out


@pytest.mark.parametrize(
    "body",
    [
        "not json at all",          # JSONDecodeError (a ValueError subclass)
        '{"name": "f"}',            # valid JSON, but not the required array
        "[1, 2]",                   # array whose items are not objects
    ],
)
def test_unparsable_wrapper_degrades_to_text(body):
    expander = PanguToolArrayEventExpander(TERMINALS)
    events = [
        SemanticEvent(EventType.TOOL_CALL_START),
        SemanticEvent(EventType.ARG_VALUE_CHUNK, value=body),
        SemanticEvent(EventType.TOOL_CALL_END, value=TERMINALS.tool_end),
    ]

    out = expander.expand(events)

    assert [e.type for e in out] == [EventType.TEXT_CHUNK]
    assert out[0].value == TERMINALS.tool_start + body + TERMINALS.tool_end
