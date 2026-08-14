# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Split one Pangu JSON-array wrapper into native per-call events.

Pangu wraps one or more tool calls in a single marker pair:

    <|tool_call_start|>[{"name":"...","arguments":{...}}, ...]<|tool_call_end|>

vLLM's ParserEngine already extracts ``name`` / ``arguments`` from each
JSON object (``_try_extract_name``, ``_extract_name_and_args``,
``arg_converter``). This expander only walks the outer array and emits
one TOOL_CALL_START / ARG_VALUE_CHUNK / TOOL_CALL_END group per object.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto

from vllm.parser.engine.events import EventType, SemanticEvent

from omni_npu.v1.parsers._pangu_parser_engine_config import PanguTerminals


class _Phase(Enum):
    SEEK_ARRAY = auto()
    SEEK_OBJECT = auto()
    IN_OBJECT = auto()
    DONE = auto()


@dataclass(slots=True)
class _WrapperState:
    """Mutable scanner state for one Pangu marker pair."""

    body: str = ""
    pos: int = 0
    phase: _Phase = _Phase.SEEK_ARRAY
    value_start: int = -1
    value_emitted: int = 0
    emitted_any: bool = False
    open_tool: bool = False
    tool_index: int = -1


def _consume_json_value(text: str, start: int) -> tuple[int, bool]:
    """Scan one JSON value from ``start``.

    Same brace/string rules as ``StreamingParserEngine._feed_args_char``.
    Returns ``(end_pos, complete)``. ``end_pos`` is exclusive for the value
    itself (does not consume a trailing ``,`` / parent ``}``).
    """
    n = len(text)
    if start >= n:
        return start, False

    if text[start] == '"':
        escaped = False
        i = start + 1
        while i < n:
            c = text[i]
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                return i + 1, True
            i += 1
        return i, False

    depth = 0
    in_string = False
    escaped = False
    i = start
    while i < n:
        c = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_string = False
            i += 1
            continue

        if c == '"':
            in_string = True
        elif c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
            i += 1
            if depth <= 0:
                return i, True
            continue
        elif depth == 0 and c in ",}":
            return i, True
        i += 1
    return i, False


class PanguToolArrayEventExpander:
    """Stream a Pangu tool-array wrapper as ordinary ParserEngine tool events."""

    def __init__(self, terminals: PanguTerminals) -> None:
        self.terminals = terminals
        self._phase_handlers = {
            _Phase.SEEK_ARRAY: self._scan_array,
            _Phase.SEEK_OBJECT: self._scan_object,
            _Phase.IN_OBJECT: self._scan_json_object,
        }
        self.reset()

    def reset(self) -> None:
        self._inside = False
        self._state = _WrapperState()
        self._next_tool_index = 0

    def _begin_wrapper(self) -> None:
        self._inside = True
        self._state = _WrapperState()

    def _content(self, *, closed: bool) -> SemanticEvent:
        text = self.terminals.tool_start + self._state.body
        if closed:
            text += self.terminals.tool_end
        return SemanticEvent(EventType.TEXT_CHUNK, value=text)

    def _start_object(self) -> list[SemanticEvent]:
        state = self._state
        idx = self._next_tool_index
        self._next_tool_index += 1
        state.tool_index = idx
        state.emitted_any = True
        state.open_tool = True
        return [SemanticEvent(EventType.TOOL_CALL_START, tool_index=idx)]

    def _end_tool(self) -> list[SemanticEvent]:
        state = self._state
        if not state.open_tool:
            return []
        idx = state.tool_index
        state.open_tool = False
        state.tool_index = -1
        return [SemanticEvent(EventType.TOOL_CALL_END, tool_index=idx)]

    def _skip_ws(self) -> None:
        state = self._state
        while state.pos < len(state.body) and state.body[state.pos].isspace():
            state.pos += 1

    def _scan_array(self, _: list[SemanticEvent]) -> bool:
        state = self._state
        self._skip_ws()
        if state.pos >= len(state.body):
            return False
        if state.body[state.pos] != "[":
            state.phase = _Phase.DONE
            return False
        state.pos += 1
        state.phase = _Phase.SEEK_OBJECT
        return True

    def _scan_object(self, out: list[SemanticEvent]) -> bool:
        state = self._state
        self._skip_ws()
        if state.pos >= len(state.body):
            return False

        char = state.body[state.pos]
        if char == ",":
            state.pos += 1
            return True
        if char == "]":
            state.pos += 1
            state.phase = _Phase.DONE
            return False
        if char != "{":
            state.phase = _Phase.DONE
            return False

        out.extend(self._start_object())
        state.value_start = state.pos
        state.value_emitted = 0
        state.phase = _Phase.IN_OBJECT
        return True

    def _scan_json_object(self, out: list[SemanticEvent]) -> bool:
        state = self._state
        end, complete = _consume_json_value(state.body, state.value_start)
        emitted_end = state.value_start + state.value_emitted
        if end > emitted_end:
            out.append(
                SemanticEvent(
                    EventType.ARG_VALUE_CHUNK,
                    value=state.body[emitted_end:end],
                    tool_index=state.tool_index,
                )
            )
            state.value_emitted = end - state.value_start
        state.pos = end
        if not complete:
            return False
        out.extend(self._end_tool())
        state.phase = _Phase.SEEK_OBJECT
        return True

    def _drain(self) -> list[SemanticEvent]:
        out: list[SemanticEvent] = []
        state = self._state
        while state.pos < len(state.body) and state.phase is not _Phase.DONE:
            if not self._phase_handlers[state.phase](out):
                break
        return out

    def _finalize(self, *, closed: bool) -> list[SemanticEvent]:
        events = self._drain()
        events.extend(self._end_tool())
        if self._state.emitted_any:
            return events
        try:
            parsed = json.loads(self._state.body)
            if not isinstance(parsed, list):
                raise ValueError("Pangu auto tool wrapper must be a JSON array")
            for item in parsed:
                if not isinstance(item, dict):
                    raise ValueError("Pangu auto tool call must be an object")
                events.extend(self._start_object())
                events.append(
                    SemanticEvent(
                        EventType.ARG_VALUE_CHUNK,
                        value=json.dumps(item, ensure_ascii=False),
                        tool_index=self._state.tool_index,
                    )
                )
                events.extend(self._end_tool())
            return events
        except (json.JSONDecodeError, TypeError, ValueError):
            return [self._content(closed=closed)]

    def expand(
        self,
        events: Sequence[SemanticEvent],
    ) -> list[SemanticEvent]:
        expanded: list[SemanticEvent] = []
        for event in events:
            if event.type is EventType.TOOL_CALL_START:
                self._begin_wrapper()
                continue

            if self._inside and event.type is EventType.ARG_VALUE_CHUNK:
                if event.value:
                    self._state.body += event.value
                    expanded.extend(self._drain())
                continue

            if self._inside and event.type is EventType.TOOL_CALL_END:
                # Native ``finish()`` synthesizes TOOL_CALL_END with the
                # SemanticEvent default value; a real marker carries its text.
                synthetic = not event.value
                if synthetic and not self._state.emitted_any:
                    expanded.append(self._content(closed=False))
                else:
                    expanded.extend(self._finalize(closed=not synthetic))
                self._inside = False
                self._state = _WrapperState()
                continue

            expanded.append(event)
        return expanded
