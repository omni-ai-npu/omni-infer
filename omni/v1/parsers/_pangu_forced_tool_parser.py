# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forced named/required parsing via vLLM's native helpers, unchanged."""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from typing import Any

from openai.types.responses import ToolChoiceFunction
from pydantic import TypeAdapter, ValidationError
from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    FunctionCall,
    FunctionDefinition,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.tool_parsers import streaming as tool_streaming

_REQUIRED_CALLS_ADAPTER = TypeAdapter(list[FunctionDefinition])


def _mark_field_unset(model: Any, name: str) -> None:
    """Drop ``name`` from Pydantic's fields_set so exclude_unset omits it."""
    fields_set = getattr(model, "__pydantic_fields_set__", None)
    if fields_set is not None:
        fields_set.discard(name)


def _sanitize_tool_delta(delta: DeltaMessage | None) -> DeltaMessage | None:
    """Drop empty arguments and explicit null names from the SSE payload."""
    if delta is None or not delta.tool_calls:
        return delta
    kept = []
    for tool_call in delta.tool_calls:
        fn = tool_call.function
        if isinstance(fn, DeltaFunctionCall):
            if fn.name is None:
                _mark_field_unset(fn, "name")
            if not fn.arguments:
                _mark_field_unset(fn, "arguments")
            useful = "name" in fn.model_fields_set or "arguments" in fn.model_fields_set
            if not useful and not tool_call.id:
                continue
        kept.append(tool_call)
    if not kept:
        _mark_field_unset(delta, "tool_calls")
        delta.tool_calls = []
        if not delta.reasoning and not delta.content:
            return None
        return delta
    delta.tool_calls = kept
    return delta


def forced_tool_name(
    request: ChatCompletionRequest | ResponsesRequest,
) -> str | None:
    """Return the name from vLLM's two native named-choice request types."""
    choice = request.tool_choice
    if isinstance(choice, ToolChoiceFunction):
        return choice.name
    if isinstance(choice, ChatCompletionNamedToolChoiceParam):
        return choice.function.name
    return None


@dataclass(slots=True)
class _ForcedStreamState:
    previous_text: str = ""
    current_text: str = ""
    function_name_returned: bool = False


class PanguForcedToolParser:
    """Call vLLM named/required helpers with the model text as-is."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self.reset()

    def reset(self) -> None:
        self._state = _ForcedStreamState()

    def parse_complete(
        self,
        content: str | None,
        function_name: str | None,
    ) -> tuple[list[FunctionCall] | None, str | None]:
        """Same as ``AbstractParser._extract_tool_calls`` named/required."""
        if function_name is not None:
            if content is None:
                return [], None
            return [
                FunctionCall(
                    id=make_tool_call_id(),
                    name=function_name,
                    arguments=content,
                )
            ], None

        parsed_calls: list[FunctionDefinition] = []
        with contextlib.suppress(ValidationError):
            parsed_calls = _REQUIRED_CALLS_ADAPTER.validate_json(content or "")
        return [
            FunctionCall(
                id=make_tool_call_id(),
                name=call.name,
                arguments=json.dumps(call.parameters, ensure_ascii=False),
            )
            for call in parsed_calls
        ], None

    def parse_delta(
        self,
        delta_text: str,
        function_name: str | None,
        *,
        finished: bool,
    ) -> DeltaMessage | None:
        """Same as ``AbstractParser._extract_tool_calls_streaming`` named/required."""
        del finished
        if not delta_text:
            return None

        state = self._state
        previous_text = state.current_text
        current_text = previous_text + delta_text
        state.previous_text = previous_text
        state.current_text = current_text

        if function_name is not None:
            result, state.function_name_returned = (
                tool_streaming.extract_named_tool_call_streaming(
                    delta_text=delta_text,
                    function_name=function_name,
                    function_name_returned=state.function_name_returned,
                    tool_call_idx=None,
                    tool_call_id_type="random",
                    tokenizer=self._tokenizer,
                )
            )
            return _sanitize_tool_delta(result)

        result, state.function_name_returned = (
            tool_streaming.extract_required_tool_call_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=delta_text,
                function_name_returned=state.function_name_returned,
                tool_call_idx=None,
                tool_call_id_type="random",
            )
        )
        return _sanitize_tool_delta(result)
