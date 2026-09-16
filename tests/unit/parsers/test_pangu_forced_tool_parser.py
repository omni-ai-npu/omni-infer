# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Fallback paths of the Pangu forced (required / named) tool parser.

``required`` tool_choice does not go through ``pangu_tool_arg_converter``:
``vllm.tool_parsers.streaming.extract_required_tool_call_streaming`` calls
``partial_json_loads`` itself and only catches ``MalformedJSON`` /
``JSONDecodeError``.  The fast fixer's ``AssertionError`` / ``IndexError``
therefore escape and abort the response stream, which is why
``PanguForcedToolParser`` guards the call.
"""

import pytest

import omni_npu.v1.parsers._pangu_forced_tool_parser as fp_mod
import omni_npu.v1.parsers._pangu_parser_engine_config as cfg_mod
from omni_npu.v1.parsers._pangu_forced_tool_parser import PanguForcedToolParser

# malformed case for partial_json_parser
MALFORMED_BODY = '{"a": 1, 2'


@pytest.fixture(autouse=True)
def _reset_partial_failure_log_state():
    """The warning-dedup state is module-level; isolate every test in this file."""
    cfg_mod._LOGGED_PARTIAL_FAILURES.clear()
    yield
    cfg_mod._LOGGED_PARTIAL_FAILURES.clear()


def _parser() -> PanguForcedToolParser:
    # ``tokenizer=object()`` is safe: only the named + Mistral branch touches it,
    # and is_mistral_tokenizer(object()) short-circuits on the missing class attr.
    return PanguForcedToolParser(tokenizer=object())


def test_required_stream_swallows_assertion_error(monkeypatch):
    """An AssertionError leaking from the upstream helper must be caught."""

    def _boom(**_kwargs):
        raise AssertionError(MALFORMED_BODY)

    monkeypatch.setattr(
        fp_mod.tool_streaming, "extract_required_tool_call_streaming", _boom
    )
    assert _parser().parse_delta(MALFORMED_BODY, None, finished=False) is None


def test_required_stream_swallows_index_error(monkeypatch):
    """The fast fixer's empty-stack IndexError must be caught as well."""

    def _boom(**_kwargs):
        raise IndexError("pop from empty list")

    monkeypatch.setattr(
        fp_mod.tool_streaming, "extract_required_tool_call_streaming", _boom
    )
    assert _parser().parse_delta("[}", None, finished=False) is None


def test_required_stream_real_malformed_payload_returns_none():
    """No monkeypatch: real library, bad payload degrades to "no delta this tick".

    This input makes partial_json_parser's fast fixer raise AssertionError, which
    the except clause at vllm/tool_parsers/streaming.py:148 does not cover.  After
    the fix ``parse_delta`` catches it and returns None.  Even if upstream later
    fixes the library and returns a non-array result, the helper still takes its
    ``delta_message=None`` branch, so ``is None`` holds in either world.
    """
    parser = _parser()
    assert parser.parse_delta(MALFORMED_BODY, None, finished=False) is None
    # Non-vacuity guard: make sure the text really reached the parser instead of
    # being short-circuited by an early return such as ``if not delta_text``,
    # which would make this test pass forever.
    assert parser._state.current_text == MALFORMED_BODY


def test_named_stream_still_returns_delta():
    delta = _parser().parse_delta('{"city": "Beijing"}', "get_weather",
                                  finished=False)
    assert delta is not None
    assert delta.tool_calls[0].function.name == "get_weather"


def test_required_stream_warns_when_current_text_doubles(monkeypatch):
    """The forced path accumulates current_text, so the warning recurs on doubles.

    Unlike the converter (which is handed the same full text every tick), this
    path rebuilds ``current_text = previous_text + delta_text``.  Five identical
    deltas therefore grow it 10 -> 20 -> 30 -> 40 -> 50, and the shared dedup
    helper emits a line on the first failure and on every doubling.
    """
    warnings = []

    def _boom(**_kwargs):
        raise AssertionError(MALFORMED_BODY)

    monkeypatch.setattr(
        fp_mod.tool_streaming, "extract_required_tool_call_streaming", _boom
    )
    monkeypatch.setattr(fp_mod.logger, "warning",
                        lambda msg, *args: warnings.append(msg % args))

    parser = _parser()
    for _ in range(5):
        assert parser.parse_delta(MALFORMED_BODY, None, finished=False) is None

    assert len(warnings) == 3
    assert warnings[0] == (
        "Pangu forced tool streaming parse failed (len=10) '{\"a\": 1, 2'"
    )
    assert [int(w.split("(len=")[1].split(")")[0]) for w in warnings] == [10, 20, 40]
    # The stream never broke despite the repeated failures.
    assert parser._state.current_text == MALFORMED_BODY * 5
