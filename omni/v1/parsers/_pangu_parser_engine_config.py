# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declarative vLLM ParserEngine configuration for Pangu output."""

from __future__ import annotations

import json
from dataclasses import dataclass

from partial_json_parser.core.options import Allow
from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.tool_parsers.utils import partial_json_loads


@dataclass(frozen=True, slots=True)
class PanguTerminals:
    think_start: str
    think_end: str
    tool_start: str
    tool_end: str

    @classmethod
    def from_tokenizer(cls, tokenizer) -> "PanguTerminals":
        vocab = tokenizer.get_vocab()

        def choose(primary: str, fallback: str) -> str:
            return primary if primary in vocab else fallback

        return cls(
            think_start=choose("<think>", "[unused16]"),
            think_end=choose("</think>", "[unused17]"),
            tool_start=choose("<|tool_call_start|>", "[unused11]"),
            tool_end=choose("<|tool_call_end|>", "[unused12]"),
        )


def pangu_tool_arg_converter(raw_args: str, partial: bool) -> str:
    """Unwrap ``{"name", "arguments"|"parameters"}`` using vLLM helpers.

    The array expander streams each tool-call object as ARG_VALUE_CHUNK.
    ParserEngine then extracts the name via ``_try_extract_name`` and the
    argument JSON via this converter / ``_extract_args_json``.
    """
    text = raw_args.strip()
    if not text:
        return ""

    parsed: object | None = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        if not partial:
            return ""
        try:
            parsed, _ = partial_json_loads(text, Allow.ALL)
        except (json.JSONDecodeError, ValueError, TypeError):
            return ""

    if not isinstance(parsed, dict):
        return ""

    args = ParserEngine._extract_args_value(parsed)
    if args is not None:
        return args
    if partial:
        return ""
    without_name = {key: value for key, value in parsed.items() if key != "name"}
    return json.dumps(without_name, ensure_ascii=False) if without_name else "{}"


def pangu_parser_engine_config(
    terminals: PanguTerminals,
    *,
    thinking_enabled: bool,
    implicit_tool_end: bool,
) -> ParserEngineConfig:
    terminal_map = {
        "TOOL_START": terminals.tool_start,
        "TOOL_END": terminals.tool_end,
    }
    transitions = {
        (ParserState.CONTENT, "TOOL_START"): Transition(
            ParserState.TOOL_ARGS, (EventType.TOOL_CALL_START,)
        ),
        (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
            ParserState.CONTENT, (EventType.TOOL_CALL_END,)
        ),
    }

    if thinking_enabled:
        terminal_map.update(
            {
                "THINK_START": terminals.think_start,
                "THINK_END": terminals.think_end,
            }
        )
        transitions.update(
            {
                (ParserState.REASONING, "THINK_START"): Transition(
                    ParserState.REASONING
                ),
                (ParserState.REASONING, "THINK_END"): Transition(
                    ParserState.CONTENT, (EventType.REASONING_END,)
                ),
                (ParserState.CONTENT, "THINK_START"): Transition(ParserState.CONTENT),
                (ParserState.CONTENT, "THINK_END"): Transition(ParserState.CONTENT),
            }
        )
        if implicit_tool_end:
            transitions[(ParserState.REASONING, "TOOL_START")] = Transition(
                ParserState.TOOL_ARGS,
                (EventType.REASONING_END, EventType.TOOL_CALL_START),
            )

    return ParserEngineConfig(
        name="pangu",
        initial_state=(
            ParserState.REASONING if thinking_enabled else ParserState.CONTENT
        ),
        terminals=terminal_map,
        token_id_terminals=dict(terminal_map),
        transitions=transitions,
        content_events={
            ParserState.CONTENT: EventType.TEXT_CHUNK,
            ParserState.REASONING: EventType.REASONING_CHUNK,
            ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,
        },
        arg_converter=pangu_tool_arg_converter,
        tool_args_json=False,
        preserve_tokens=(
            frozenset({terminals.think_start, terminals.think_end})
            if not thinking_enabled
            else frozenset()
        ),
        strip_trailing_reasoning_whitespace=False,
        drop_whitespace_only_content_before_tools=False,
        strip_content_whitespace_with_tools=False,
    )
