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

# malformed case for partial_json_parser
MALFORMED_STREAM = (
    '{"name": "执行 SQL —", "arguments": 11111111111111111111.000000-04-04'
    " 01:01:01 * *  * 4-04-04 01:01:01\n0,0,0"
)


@pytest.fixture(autouse=True)
def _reset_partial_failure_log_state():
    """The warning-dedup state is module-level; isolate every test in this file."""
    cfg_mod._LOGGED_PARTIAL_FAILURES.clear()
    yield
    cfg_mod._LOGGED_PARTIAL_FAILURES.clear()


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


def test_partial_parser_assertion_error_returns_empty(monkeypatch):
    """The fast fixer's AssertionError must be swallowed, not abort the
    response stream.
    """

    def _boom(*_args, **_kwargs):
        raise AssertionError('{"a": 1, 2')

    monkeypatch.setattr(cfg_mod, "partial_json_loads", _boom)
    assert pangu_tool_arg_converter('{"a": 1, 2', partial=True) == ""


def test_partial_parser_index_error_returns_empty(monkeypatch):
    """The IndexError the fast fixer raises on an empty stack is swallowed
    the same way.
    """

    def _boom(*_args, **_kwargs):
        raise IndexError("pop from empty list")

    monkeypatch.setattr(cfg_mod, "partial_json_loads", _boom)
    assert pangu_tool_arg_converter("[}", partial=True) == ""


def test_malformed_stream_degrades_to_empty():
    """Real production payload through the real library: degrade to "",
    no exception.
    """
    assert pangu_tool_arg_converter(MALFORMED_STREAM, partial=True) == ""
    assert pangu_tool_arg_converter(MALFORMED_STREAM, partial=False) == ""


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


# ── Warning dedup ───────────────────────────────────────────────────────────
# A runaway generation keeps failing on every later chunk, so these cases pin
# down when a warning is actually emitted.  The dedup state is module-level.


def test_should_log_partial_failure_first_time():
    assert cfg_mod._should_log_partial_failure('{"a": 1, 2') is True


def test_should_log_partial_failure_suppresses_growth_below_double():
    assert cfg_mod._should_log_partial_failure("x" * 100) is True
    assert cfg_mod._should_log_partial_failure("x" * 150) is False
    assert cfg_mod._should_log_partial_failure("x" * 199) is False


def test_should_log_partial_failure_logs_again_when_text_doubles():
    assert cfg_mod._should_log_partial_failure("x" * 100) is True
    assert cfg_mod._should_log_partial_failure("x" * 200) is True
    assert cfg_mod._should_log_partial_failure("x" * 350) is False
    assert cfg_mod._should_log_partial_failure("x" * 400) is True


def test_should_log_partial_failure_keeps_distinct_tool_calls():
    """Bad text from another tool call is not a prefix of this one."""
    assert cfg_mod._should_log_partial_failure("x" * 100) is True
    assert cfg_mod._should_log_partial_failure('{"b": 2, 3') is True


def test_should_log_partial_failure_state_stays_bounded():
    for index in range(20):
        assert cfg_mod._should_log_partial_failure(f"payload-{index}") is True
    assert len(cfg_mod._LOGGED_PARTIAL_FAILURES) == 8


def test_format_partial_failure_dumps_short_text_whole():
    """Below the split threshold, head/tail would repeat the same characters."""
    assert cfg_mod.format_partial_failure('{"a": 1, 2') == '(len=10) \'{"a": 1, 2\''


def test_format_partial_failure_dumps_text_at_the_threshold_whole():
    text = "x" * 400                      # exactly 2 * limit
    assert cfg_mod.format_partial_failure(text) == f"(len=400) {text!r}"


def test_format_partial_failure_splits_only_past_the_threshold():
    text = "x" * 401
    assert cfg_mod.format_partial_failure(text) == (
        f"(len=401) head={'x' * 200!r} tail={'x' * 200!r}"
    )


def test_format_partial_failure_has_no_overlap_when_split():
    text = "".join(chr(ord("a") + index % 26) for index in range(1000))
    excerpt = cfg_mod.format_partial_failure(text)
    assert f"head={text[:200]!r}" in excerpt
    assert f"tail={text[-200:]!r}" in excerpt
    assert str(len(text)) in excerpt


def test_converter_warns_once_for_a_repeated_failure(monkeypatch):
    """50 identical ticks produce a single warning carrying the payload."""
    warnings = []

    def _boom(*_args, **_kwargs):
        raise AssertionError('{"a": 1, 2')

    monkeypatch.setattr(cfg_mod, "partial_json_loads", _boom)
    monkeypatch.setattr(cfg_mod.logger, "warning",
                        lambda msg, *args: warnings.append(msg % args))

    for _ in range(50):
        assert pangu_tool_arg_converter('{"a": 1, 2', partial=True) == ""

    assert len(warnings) == 1
    assert warnings[0] == (
        "Pangu partial JSON parse failed (len=10) '{\"a\": 1, 2'"
    )


def test_converter_warns_again_once_the_text_doubles(monkeypatch):
    """The runaway trajectory stays observable, not just its first failure."""
    warnings = []

    def _boom(*_args, **_kwargs):
        raise AssertionError("boom")

    monkeypatch.setattr(cfg_mod, "partial_json_loads", _boom)
    monkeypatch.setattr(cfg_mod.logger, "warning",
                        lambda msg, *args: warnings.append(msg % args))

    short = '{"a": 1, 2' + "0" * 20          # 30 chars
    grown = '{"a": 1, 2' + "0" * 120         # 130 chars, > 2x
    for text in (short, grown):
        assert pangu_tool_arg_converter(text, partial=True) == ""
    assert len(warnings) == 2
