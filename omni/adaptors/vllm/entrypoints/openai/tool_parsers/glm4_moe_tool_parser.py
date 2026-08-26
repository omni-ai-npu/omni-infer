# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math
from collections.abc import Sequence
from typing import Any

import regex as re

from vllm.entrypoints.chat_utils import random_tool_call_id
from vllm.entrypoints.openai.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser,
    ToolParserManager,
)

logger = init_logger(__name__)


def _partial_tag_overlap(text: str, tag: str) -> int:
    max_len = min(len(text), len(tag) - 1)
    for size in range(max_len, 0, -1):
        if text.endswith(tag[:size]):
            return size
    return 0


def _extract_types_from_schema(schema: Any) -> list[str]:
    if not isinstance(schema, dict):
        return ["string"]

    types: set[str] = set()
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        types.add(schema_type)
    elif isinstance(schema_type, list):
        types.update(item for item in schema_type if isinstance(item, str))

    enum_values = schema.get("enum")
    if isinstance(enum_values, list):
        for value in enum_values:
            if value is None:
                types.add("null")
            elif isinstance(value, bool):
                types.add("boolean")
            elif isinstance(value, int):
                types.add("integer")
            elif isinstance(value, float):
                types.add("number")
            elif isinstance(value, str):
                types.add("string")
            elif isinstance(value, list):
                types.add("array")
            elif isinstance(value, dict):
                types.add("object")

    for key in ("anyOf", "oneOf", "allOf"):
        for sub_schema in schema.get(key, []) or []:
            types.update(_extract_types_from_schema(sub_schema))
    return list(types) if types else ["string"]


_TYPE_ALIASES: dict[str, str] = {
    "str": "string",
    "text": "string",
    "varchar": "string",
    "char": "string",
    "enum": "string",
    "int": "integer",
    "int32": "integer",
    "int64": "integer",
    "uint": "integer",
    "uint32": "integer",
    "uint64": "integer",
    "long": "integer",
    "short": "integer",
    "unsigned": "integer",
    "float": "number",
    "float32": "number",
    "float64": "number",
    "double": "number",
    "bool": "boolean",
    "dict": "object",
    "arr": "array",
    "list": "array",
    "sequence": "array",
}


def _normalize_types(types: list[str]) -> set[str]:
    return {
        _TYPE_ALIASES.get(item.strip().lower(), item.strip().lower())
        for item in types
        if isinstance(item, str)
    }


def _is_json_finite(value: Any) -> bool:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True


def _tool_function(tool: ChatCompletionToolsParam | dict) -> Any:
    if isinstance(tool, dict):
        return tool.get("function", {})
    return tool.function


def _function_name(function: Any) -> str | None:
    if isinstance(function, dict):
        return function.get("name")
    return getattr(function, "name", None)


def _function_parameters(function: Any) -> dict[str, Any] | None:
    if isinstance(function, dict):
        return function.get("parameters")
    return getattr(function, "parameters", None)


@ToolParserManager.register_module("glm52")
@ToolParserManager.register_module("glm51")
@ToolParserManager.register_module("glm5")
@ToolParserManager.register_module("glm47")
@ToolParserManager.register_module("glm45")
@ToolParserManager.register_module("GlmMoe")
class Glm4MoeModelToolParser(ToolParser):
    """GLM XML tool parser backported to the vLLM 0.9.0 parser API."""

    supports_required_and_named = False
    structural_tag_model = "glm_4_7"

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)
        self.current_tool_name_sent = False
        self.prev_tool_call_arr: list[dict[str, Any]] = []
        self.current_tool_id = -1
        self.streamed_args_for_tool: list[str] = []
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"
        self.arg_key_start = "<arg_key>"
        self.arg_key_end = "</arg_key>"
        self.arg_val_start = "<arg_value>"
        self.arg_val_end = "</arg_value>"

        self.tool_calls_start_token = self.tool_call_start_token

        self.func_call_regex = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
        self.func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
            re.DOTALL,
        )
        self._arg_key_pattern = re.compile(
            re.escape(self.arg_key_start) + r"(.*?)" + re.escape(self.arg_key_end),
            re.DOTALL,
        )
        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)
        self._sent_content_idx = 0
        self._deferred_content = ""
        self._last_current_text = ""
        self._tool_call_ids: list[str] = []

    @staticmethod
    def _tools_enabled(request: ChatCompletionRequest) -> bool:
        try:
            tools = getattr(request, "tools", None)
            tool_choice = getattr(request, "tool_choice", None)
            return bool(tools) and tool_choice != "none"
        except Exception:
            logger.exception("Failed to determine if tools are enabled.")
            return False

    @staticmethod
    def _json_escape_string_content(value: str) -> str:
        if not value:
            return ""
        return json.dumps(value, ensure_ascii=False)[1:-1]

    @staticmethod
    def _safe_arg_prefix(json_str: str, string_keys: set[str] | None = None) -> str:
        last_colon = -1
        last_key: str | None = None
        pending_key: str | None = None
        in_string = False
        escape = False
        string_start = -1
        depth = 0

        for index, char in enumerate(json_str):
            if escape:
                escape = False
                continue
            if in_string:
                if char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                    if depth == 1 and string_start >= 0:
                        pending_key = json_str[string_start + 1 : index]
                continue
            if char == '"':
                in_string = True
                string_start = index
            elif char in ("{", "["):
                depth += 1
            elif char in ("}", "]"):
                depth -= 1
            elif char == ":" and depth == 1:
                last_colon = index
                last_key = pending_key
                pending_key = None

        if last_colon < 0:
            return ""

        end = last_colon + 1
        while end < len(json_str) and json_str[end] in (" ", "\t", "\n", "\r"):
            end += 1
        if end >= len(json_str) or json_str[end] != '"':
            return json_str[:end]
        if string_keys is not None and last_key not in string_keys:
            return json_str[:end]

        escape = False
        for index in range(end + 1, len(json_str)):
            char = json_str[index]
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                return json_str[:index]
        return json_str

    def _is_string_type(
        self,
        tool_name: str,
        arg_name: str,
        tools: list[ChatCompletionToolsParam] | None,
    ) -> bool:
        if tools is None:
            return False
        for tool in tools:
            function = _tool_function(tool)
            if _function_name(function) != tool_name:
                continue
            parameters = _function_parameters(function)
            if parameters is None:
                return False
            arg_schema = parameters.get("properties", {}).get(arg_name, {})
            arg_types = _extract_types_from_schema(arg_schema)
            return _normalize_types(arg_types) - {"null"} == {"string"}
        logger.warning("No tool named %s.", tool_name)
        return False

    def _schema_for_arg(
        self,
        tool_name: str,
        arg_name: str,
        tools: list[ChatCompletionToolsParam] | None,
    ) -> dict[str, Any]:
        if tools is None:
            return {}
        for tool in tools:
            function = _tool_function(tool)
            if _function_name(function) != tool_name:
                continue
            parameters = _function_parameters(function) or {}
            schema = parameters.get("properties", {}).get(arg_name, {})
            return schema if isinstance(schema, dict) else {}
        return {}

    def _streamable_string_keys(
        self,
        tool_name: str,
        tools: list[ChatCompletionToolsParam] | None,
    ) -> set[str] | None:
        if not tools:
            return None
        for tool in tools:
            function = _tool_function(tool)
            if _function_name(function) != tool_name:
                continue
            parameters = _function_parameters(function) or {}
            properties = parameters.get("properties", {})
            if not isinstance(properties, dict):
                return None
            streamable: set[str] = set()
            for key, schema in properties.items():
                if _normalize_types(_extract_types_from_schema(schema)) == {"string"}:
                    streamable.add(key)
            return streamable
        return None

    @classmethod
    def _coerce_value(cls, value: Any, schema: dict[str, Any]) -> Any:
        arg_types = _normalize_types(_extract_types_from_schema(schema))
        nullable = "null" in arg_types
        concrete_types = arg_types - {"null"}

        if value is None:
            return None if nullable else value

        if isinstance(value, dict):
            nested_props = schema.get("properties", {})
            if isinstance(nested_props, dict):
                for key, child_value in list(value.items()):
                    child_schema = nested_props.get(key, {})
                    if isinstance(child_schema, dict):
                        value[key] = cls._coerce_value(child_value, child_schema)
            return value

        if isinstance(value, list):
            item_schema = schema.get("items", {})
            if isinstance(item_schema, dict):
                return [cls._coerce_value(item, item_schema) for item in value]
            return value

        if not concrete_types:
            return None if nullable and str(value).strip().lower() == "null" else value

        if isinstance(value, str):
            stripped = value.strip()
            for candidate_type in (
                "null",
                "integer",
                "number",
                "boolean",
                "object",
                "array",
                "string",
            ):
                if candidate_type not in arg_types:
                    continue
                if candidate_type == "null":
                    if stripped.lower() == "null":
                        return None
                    continue
                if candidate_type == "integer":
                    try:
                        return int(stripped)
                    except (TypeError, ValueError):
                        continue
                if candidate_type == "number":
                    try:
                        number = float(stripped)
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(number):
                        continue
                    return int(number) if number == int(number) else number
                if candidate_type == "boolean":
                    lower_value = stripped.lower()
                    if lower_value in ("true", "1"):
                        return True
                    if lower_value in ("false", "0"):
                        return False
                    continue
                if candidate_type in ("object", "array"):
                    try:
                        parsed = json.loads(stripped)
                    except (TypeError, ValueError):
                        continue
                    if not _is_json_finite(parsed):
                        continue
                    if candidate_type == "object" and isinstance(parsed, dict):
                        return cls._coerce_value(parsed, schema)
                    if candidate_type == "array" and isinstance(parsed, list):
                        return cls._coerce_value(parsed, schema)
                    continue
                if candidate_type == "string":
                    return value
            return value

        if "boolean" in concrete_types and isinstance(value, bool):
            return value
        if (
            "integer" in concrete_types
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            return value
        if (
            "number" in concrete_types
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        ):
            return value
        return value

    def _deserialize_arg_value(
        self,
        tool_name: str,
        arg_name: str,
        raw_value: str,
        request: ChatCompletionRequest,
        partial: bool,
    ) -> Any:
        schema = self._schema_for_arg(tool_name, arg_name, request.tools)
        if self._is_string_type(tool_name, arg_name, request.tools):
            return raw_value
        if partial:
            return raw_value
        return self._coerce_value(raw_value, schema)

    def _is_valid_tool_name(
        self,
        tool_name: str,
        tools: list[ChatCompletionToolsParam] | None,
    ) -> bool:
        if not tools:
            return True
        return any(
            _function_name(_tool_function(tool)) == tool_name for tool in tools
        )

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        if getattr(request, "tools", None) and request.tool_choice != "none":
            request.skip_special_tokens = False

        if isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam):
            return request
        if request.tool_choice == "required":
            return request
        return super().adjust_request(request)

    def _extract_tool_call_regions(self, text: str) -> list[tuple[str, bool]]:
        regions: list[tuple[str, bool]] = []
        pos = 0
        while True:
            start = text.find(self.tool_call_start_token, pos)
            if start == -1:
                break
            inner_start = start + len(self.tool_call_start_token)
            end = text.find(self.tool_call_end_token, inner_start)
            if end != -1:
                regions.append((text[inner_start:end], True))
                pos = end + len(self.tool_call_end_token)
                continue

            raw = text[inner_start:]
            overlap = _partial_tag_overlap(raw, self.tool_call_end_token)
            if overlap:
                raw = raw[:-overlap]
            regions.append((raw, False))
            break
        return regions

    def _extract_tool_name_from_region(
        self,
        inner_text: str,
        is_complete: bool,
    ) -> str | None:
        inner_text = inner_text.lstrip()
        newline_idx = inner_text.find("\n")
        arg_key_idx = inner_text.find(self.arg_key_start)
        candidates = [idx for idx in (newline_idx, arg_key_idx) if idx != -1]
        if candidates:
            name = inner_text[: min(candidates)].strip()
            return name or None
        if is_complete:
            name = inner_text.strip()
            return name or None
        return None

    def _build_args_json_so_far(
        self,
        tool_name: str,
        inner_text: str,
        is_complete: bool,
        request: ChatCompletionRequest,
    ) -> str:
        pairs = self.func_arg_regex.findall(inner_text)
        args: dict[str, Any] = {}

        for key, value in pairs:
            arg_key = key.strip()
            args[arg_key] = self._deserialize_arg_value(
                tool_name,
                arg_key,
                value,
                request,
                partial=False,
            )

        last_val_start = inner_text.rfind(self.arg_val_start)
        last_val_end = inner_text.rfind(self.arg_val_end)
        has_partial_value = last_val_start != -1 and (
            last_val_end == -1 or last_val_end < last_val_start
        )

        if has_partial_value:
            last_key_match = None
            for match in self._arg_key_pattern.finditer(inner_text[:last_val_start]):
                last_key_match = match

            if last_key_match:
                partial_key = last_key_match.group(1).strip()
                partial_content_start = last_val_start + len(self.arg_val_start)
                partial_content = inner_text[partial_content_start:]
                overlap = _partial_tag_overlap(partial_content, self.arg_val_end)
                if overlap:
                    partial_content = partial_content[:-overlap]

                if is_complete:
                    args[partial_key] = self._deserialize_arg_value(
                        tool_name,
                        partial_key,
                        partial_content,
                        request,
                        partial=False,
                    )
                elif self._is_string_type(tool_name, partial_key, request.tools):
                    parts = [
                        f"{json.dumps(key, ensure_ascii=False)}: "
                        f"{json.dumps(value, ensure_ascii=False)}"
                        for key, value in args.items()
                    ]
                    key_json = json.dumps(partial_key, ensure_ascii=False)
                    escaped = self._json_escape_string_content(partial_content)
                    parts.append(f"{key_json}: \"{escaped}")
                    return "{" + ", ".join(parts)
                else:
                    args[partial_key] = self._deserialize_arg_value(
                        tool_name,
                        partial_key,
                        partial_content,
                        request,
                        partial=True,
                    )

        if not args:
            return "{}" if is_complete else ""

        args_json = json.dumps(args, ensure_ascii=False)
        if not is_complete:
            return self._safe_arg_prefix(
                args_json,
                self._streamable_string_keys(tool_name, request.tools),
            )
        return args_json

    def _extract_content(
        self,
        current_text: str,
        finished: bool = False,
        current_token_ids: Sequence[int] | None = None,
    ) -> str | None:
        content_segments: list[str] = []
        pos = self._sent_content_idx

        while pos < len(current_text):
            start = current_text.find(self.tool_call_start_token, pos)
            if start == -1:
                tail = current_text[pos:]
                overlap = _partial_tag_overlap(tail, self.tool_call_start_token)
                # At stream finish there is no further text coming to resolve a
                # partial tag, so a trailing "<" must be emitted, not dropped.
                if overlap and finished:
                    overlap = 0
                # When the tool_call tag is a single special token (true for all
                # GLM models), a trailing "<" is only a partial tag if that
                # token id is actually present in current_token_ids. Otherwise
                # it is a literal "<" in the content (e.g. "<head>", "i < days")
                # and must be emitted, not held back. The old code sliced it off
                # via tail[:-overlap] and advanced _sent_content_idx past it,
                # permanently dropping the "<" from the content stream.
                if (
                    overlap
                    and not finished
                    and self.tool_call_start_token_id is not None
                    and current_token_ids is not None
                    and self.tool_call_start_token_id not in current_token_ids
                ):
                    overlap = 0
                sendable = tail[: len(tail) - overlap] if overlap else tail
                if sendable:
                    content_segments.append(sendable)
                pos = len(current_text) - overlap
                break

            if start > pos:
                content_segments.append(current_text[pos:start])

            end = current_text.find(self.tool_call_end_token, start)
            if end != -1:
                pos = end + len(self.tool_call_end_token)
            else:
                pos = start
                break

        if content_segments:
            self._sent_content_idx = pos
            content = "".join(content_segments)
            if not finished and not self.prev_tool_call_arr and not content.strip():
                self._deferred_content += content
                return None
            if self.prev_tool_call_arr and not self._deferred_content.strip():
                self._deferred_content = ""
            if self._deferred_content:
                content = self._deferred_content + content
                self._deferred_content = ""
            if self.prev_tool_call_arr and not content.strip():
                return None
            return content
        if pos > self._sent_content_idx:
            self._sent_content_idx = pos
        if finished and self._deferred_content:
            content = self._deferred_content
            self._deferred_content = ""
            return content if content.strip() else None
        return None

    def _ensure_tool_state_for(self, index: int) -> None:
        while len(self._tool_call_ids) <= index:
            self._tool_call_ids.append(random_tool_call_id())
        while len(self.streamed_args_for_tool) <= index:
            self.streamed_args_for_tool.append("")
        while len(self.prev_tool_call_arr) <= index:
            self.prev_tool_call_arr.append({})

    def _compute_args_diff(
        self,
        index: int,
        args_so_far: str,
        is_complete: bool,
    ) -> str | None:
        if not args_so_far or len(args_so_far) <= len(self.streamed_args_for_tool[index]):
            if is_complete and args_so_far:
                self._store_complete_args(index, args_so_far)
            return None

        diff = args_so_far[len(self.streamed_args_for_tool[index]) :]
        self.streamed_args_for_tool[index] = args_so_far
        self.prev_tool_call_arr[index]["arguments"] = args_so_far
        if is_complete:
            self._store_complete_args(index, args_so_far)
        return diff

    def _store_complete_args(self, index: int, args_json: str) -> None:
        try:
            self.prev_tool_call_arr[index]["arguments"] = json.loads(args_json)
        except Exception:
            self.prev_tool_call_arr[index]["arguments"] = args_json

    @staticmethod
    def _coalesce_tool_call_deltas(
        deltas: list[DeltaToolCall],
    ) -> list[DeltaToolCall]:
        merged: dict[int, DeltaToolCall] = {}
        for delta in deltas:
            existing = merged.get(delta.index)
            if existing is None:
                merged[delta.index] = delta
                continue
            if delta.id is not None and existing.id is None:
                existing.id = delta.id
            if delta.type is not None and existing.type is None:
                existing.type = delta.type
            if delta.function is None:
                continue
            if existing.function is None:
                existing.function = delta.function
                continue
            if delta.function.name is not None and existing.function.name is None:
                existing.function.name = delta.function.name
            if delta.function.arguments is not None:
                if existing.function.arguments is None:
                    existing.function.arguments = delta.function.arguments
                else:
                    existing.function.arguments += delta.function.arguments
        return list(merged.values())

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        try:
            tool_calls: list[ToolCall] = []
            for inner_text, is_complete in self._extract_tool_call_regions(model_output):
                if not is_complete:
                    continue
                tool_name = self._extract_tool_name_from_region(inner_text, True)
                if not tool_name:
                    logger.warning("Failed to parse tool call name from: %s", inner_text)
                    continue
                if not self._is_valid_tool_name(tool_name, request.tools):
                    logger.warning("Invalid tool call name: %s", tool_name)
                    continue
                args_json = self._build_args_json_so_far(
                    tool_name,
                    inner_text,
                    True,
                    request,
                )
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(name=tool_name, arguments=args_json),
                    )
                )
        except Exception:
            logger.exception("Failed to extract tool call spec")
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        if tool_calls:
            content = model_output[: model_output.find(self.tool_calls_start_token)]
            if not content or not content.strip():
                content = None
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content,
            )
        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=model_output,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        if not self._tools_enabled(request):
            return DeltaMessage(content=delta_text) if delta_text else None

        self._last_current_text = current_text
        content = self._extract_content(current_text, finished=False, current_token_ids=current_token_ids)
        regions = self._extract_tool_call_regions(current_text)
        tool_call_deltas: list[DeltaToolCall] = []

        for index, (inner_text, is_complete) in enumerate(regions):
            self._ensure_tool_state_for(index)
            tool_name = self._extract_tool_name_from_region(inner_text, is_complete)
            if not tool_name:
                break
            if not self._is_valid_tool_name(tool_name, request.tools):
                continue

            if "name" not in self.prev_tool_call_arr[index]:
                self.prev_tool_call_arr[index]["name"] = tool_name
                tool_call_deltas.append(
                    DeltaToolCall(
                        index=index,
                        id=self._tool_call_ids[index],
                        type="function",
                        function=DeltaFunctionCall(
                            name=tool_name,
                            arguments="",
                        ),
                    )
                )

            args_so_far = self._build_args_json_so_far(
                tool_name,
                inner_text,
                is_complete,
                request,
            )
            args_diff = self._compute_args_diff(index, args_so_far, is_complete)
            if args_diff:
                tool_call_deltas.append(
                    DeltaToolCall(
                        index=index,
                        function=DeltaFunctionCall(arguments=args_diff),
                    )
                )

        if regions:
            self.current_tool_id = len(regions) - 1

        if len(tool_call_deltas) > 1:
            tool_call_deltas = self._coalesce_tool_call_deltas(tool_call_deltas)

        if content or tool_call_deltas:
            return DeltaMessage(content=content, tool_calls=tool_call_deltas)
        return None

    def finish_streaming(self, request: ChatCompletionRequest) -> DeltaMessage | None:
        if not self._tools_enabled(request):
            content = self._extract_content(self._last_current_text, finished=True)
            return DeltaMessage(content=content) if content else None

        current_text = self._last_current_text
        content = self._extract_content(current_text, finished=True)
        regions = self._extract_tool_call_regions(current_text)
        tool_call_deltas: list[DeltaToolCall] = []

        for index, (inner_text, is_complete) in enumerate(regions):
            self._ensure_tool_state_for(index)
            tool_name = self._extract_tool_name_from_region(inner_text, True)
            if not tool_name or not self._is_valid_tool_name(tool_name, request.tools):
                continue
            if "name" not in self.prev_tool_call_arr[index]:
                self.prev_tool_call_arr[index]["name"] = tool_name
                tool_call_deltas.append(
                    DeltaToolCall(
                        index=index,
                        id=self._tool_call_ids[index],
                        type="function",
                        function=DeltaFunctionCall(name=tool_name, arguments=""),
                    )
                )
            args_so_far = self._build_args_json_so_far(
                tool_name,
                inner_text,
                True,
                request,
            )
            args_diff = self._compute_args_diff(index, args_so_far, is_complete=True)
            if args_diff:
                tool_call_deltas.append(
                    DeltaToolCall(
                        index=index,
                        function=DeltaFunctionCall(arguments=args_diff),
                    )
                )

        if len(tool_call_deltas) > 1:
            tool_call_deltas = self._coalesce_tool_call_deltas(tool_call_deltas)

        if content or tool_call_deltas:
            return DeltaMessage(content=content, tool_calls=tool_call_deltas)
        return None
