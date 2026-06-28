# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright contributors to the vLLM project.

from __future__ import annotations

import argparse
import enum
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from pydantic import Field


import vllm.sampling_params as sampling_params_module
from vllm import EngineArgs, SamplingParams
from vllm.config import VllmConfig
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, CompletionRequest
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.logger import logger
from vllm.v1.core.sched.utils import check_stop as _original_check_stop
from vllm.v1.engine import FinishReason
from vllm.v1.request import Request, RequestStatus
from vllm.v1.core.sched.scheduler import Scheduler as OriginScheduler
import vllm.v1.engine as engine_module
import vllm.v1.request as request_module

from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.vllm_patches.patches.common.patch_thinking_limit import (
    ChatCompletionRequestPatch as _UpstreamChatCompletionRequestPatch,
    EngineArgsPatch as _UpstreamEngineArgsPatch,
    SamplingParamsPatch as _UpstreamSamplingParamsPatch,
    VllmConfigPatch as _UpstreamVllmConfigPatch,
)

try:
    from omni_npu.vllm_patches.patches.common.patch_ti_epd_zmq import (
        CompletionSamplingParamsPatch as _UpstreamCompletionSamplingParamsPatch,
    )
except ImportError:
    _UpstreamCompletionSamplingParamsPatch = None

_REPETITION_DETECTION_KEY = "repetition_detection"


def _relay_patch_attrs(
        dst_cls: type,
        src_cls: type,
        *,
        exclude: tuple[str, ...] = (),
) -> None:
    for name in src_cls._attr_names_to_apply:
        if name in exclude:
            continue
        if name in src_cls.__dict__:
            setattr(dst_cls, name, src_cls.__dict__[name])


@dataclass
class RepetitionDetectionParams:
    """Parameters for detecting repetitive N-gram patterns in output tokens."""

    max_pattern_size: int = 0
    """Maximum size of N-gram pattern to detect for sequence repetition.
    Set to 0 to disable. Must be used together with min_count."""

    min_pattern_size: int = 0
    """Minimum N-gram pattern size to check for sequence repetition.
    If set to 0, it defaults to 1.
    Must be <= max_pattern_size."""

    min_count: int = 0
    """Minimum number of times an N-gram pattern must repeat to trigger
    detection. Must be >= 2. Example: 3 for detecting a phrase repeated
    3 times. Must be used together with max_pattern_size."""

    def __post_init__(self) -> None:
        if (
            self.max_pattern_size < 0
            or self.min_pattern_size < 0
            or self.min_pattern_size > self.max_pattern_size
        ):
            raise ValueError(
                "max_pattern_size, min_pattern_size must be >=0, "
                "with min_pattern_size <= max_pattern_size. "
                "Set both to 0 to disable repetitive pattern detection."
            )
        if self.max_pattern_size > 0 and self.min_count < 2:
            raise ValueError(
                "min_count must be >= 2 to detect repetitive patterns "
                "in engine output. If you do not wish to detect repetitive "
                "patterns, set max_pattern_size to 0."
            )

    @staticmethod
    def as_argparse_dict() -> dict[str, Any]:
        return {
            "type": _parse_repetition_detection_cli,
            "default": None,
            "help": (
                "JSON object configuring repetitive N-gram detection in output "
                "tokens, e.g. "
                '\'{"max_pattern_size":10,"min_pattern_size":2,"min_count":3}\'. '
                "Set max_pattern_size to 0 to disable."
            ),
        }


def _coerce_repetition_detection(
        value: Any,
) -> RepetitionDetectionParams | None:
    if value is None or isinstance(value, RepetitionDetectionParams):
        return value
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        return RepetitionDetectionParams(**value)
    return None


def _parse_repetition_detection_cli(
        value: str,
) -> RepetitionDetectionParams | None:
    """Parse ``--repetition-detection`` the same way vLLM handles optional JSON args."""
    if value == "" or value == "None":
        return None
    try:
        return _coerce_repetition_detection(value)
    except Exception as e:
        raise ValueError(
            f"Invalid repetition-detection JSON: {value!r}"
        ) from e


def _resolve_repetition_detection(
        request_value: RepetitionDetectionParams | dict[str, Any] | None,
        default_sampling_params: dict | None,
) -> RepetitionDetectionParams | None:
    coerced_request = _coerce_repetition_detection(request_value)
    if coerced_request is not None:
        return coerced_request
    if default_sampling_params is None:
        return None
    return _coerce_repetition_detection(
        default_sampling_params.get(_REPETITION_DETECTION_KEY)
    )


def _inject_default_repetition_detection(serving: OpenAIServingChat | OpenAIServingCompletion) -> None:
    vllm_config = getattr(serving.engine_client, "vllm_config", None)
    if vllm_config is None:
        return
    repetition_detection = _coerce_repetition_detection(
        getattr(vllm_config, _REPETITION_DETECTION_KEY, None)
    )
    if repetition_detection is None:
        return
    if serving.default_sampling_params is None:
        serving.default_sampling_params = {}
    serving.default_sampling_params[_REPETITION_DETECTION_KEY] = repetition_detection


def _get_repetition_detection(
        params: SamplingParams | None,
) -> RepetitionDetectionParams | None:
    if params is None or params.extra_args is None:
        return None
    params = _coerce_repetition_detection(
        params.extra_args.get(_REPETITION_DETECTION_KEY)
    )
    return params 


def _set_repetition_detection(
        params: SamplingParams,
        repetition_detection: RepetitionDetectionParams | None,
) -> SamplingParams:
    coerced = _coerce_repetition_detection(repetition_detection)
    if coerced is None:
        return params
    if params.extra_args is None:
        params.extra_args = {}
    params.extra_args[_REPETITION_DETECTION_KEY] = coerced
    return params


def _get_request_repetition_detection(
        request: ChatCompletionRequest | CompletionRequest,
) -> RepetitionDetectionParams | None:
    value = getattr(request, _REPETITION_DETECTION_KEY, None)
    if value is not None:
        return _coerce_repetition_detection(value)
    model_extra = getattr(request, "model_extra", None)
    if model_extra:
        return _coerce_repetition_detection(
            model_extra.get(_REPETITION_DETECTION_KEY)
        )
    return None


def _register_repetition_detection_field(model_cls: type) -> None:
    if _REPETITION_DETECTION_KEY in model_cls.model_fields:
        return
    model_cls.__annotations__[_REPETITION_DETECTION_KEY] = (
        RepetitionDetectionParams | None
    )
    model_cls.model_fields = {
        **model_cls.model_fields,
        _REPETITION_DETECTION_KEY: Field(
            default=None,
            description="Parameters for detecting repetitive N-gram patterns "
            "in output tokens. If such repetition is detected, generation will "
            "be ended early. LLMs can sometimes generate repetitive, unhelpful "
            "token patterns, stopping only when they hit the maximum output length "
            "(e.g. 'abcdabcdabcd...' or '\\emoji \\emoji \\emoji ...'). This "
            "feature can detect such behavior and terminate early, saving time "
            "and tokens.",
        ),
    }
    model_cls.field_names = None
    model_cls.model_rebuild(force=True)


_register_repetition_detection_field(ChatCompletionRequest)
_register_repetition_detection_field(CompletionRequest)


@register_patch("VllmConfigPatch", VllmConfig)
class VllmConfigPatch(VLLMPatch):
    """Relay thinking-limit config patch and add repetition detection defaults."""

    _attr_names_to_apply = list(_UpstreamVllmConfigPatch._attr_names_to_apply) + [
        _REPETITION_DETECTION_KEY,
    ]

    repetition_detection: RepetitionDetectionParams | None = None


_relay_patch_attrs(VllmConfigPatch, _UpstreamVllmConfigPatch)

_orig_ea_add_cli_args = _UpstreamEngineArgsPatch.add_cli_args
_orig_ea_from_cli_args = _UpstreamEngineArgsPatch.from_cli_args.__func__
_orig_ea_create_engine_config = _UpstreamEngineArgsPatch.create_engine_config


@register_patch("EngineArgsPatch", EngineArgs)
class EngineArgsPatch(VLLMPatch):
    """Relay reasoning CLI args and add ``--repetition-detection``."""

    _attr_names_to_apply = list(_UpstreamEngineArgsPatch._attr_names_to_apply) + [
        _REPETITION_DETECTION_KEY,
    ]

    repetition_detection: RepetitionDetectionParams | None = None

    @staticmethod
    def add_cli_args(parser):
        parser = _orig_ea_add_cli_args(parser)
        vllm_group = parser.add_argument_group(
            title="VllmConfig",
            description=VllmConfig.__doc__,
        )
        vllm_group.add_argument(
            "--repetition-detection",
            **RepetitionDetectionParams.as_argparse_dict(),
        )
        return parser


    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        instance = _orig_ea_from_cli_args(cls, args)
        raw_repetition = getattr(args, _REPETITION_DETECTION_KEY, None)
        try:
            instance.repetition_detection = _coerce_repetition_detection(
                raw_repetition
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Error parsing repetition_detection: %s", exc)
            instance.repetition_detection = None
        return instance

    def create_engine_config(
            self,
            usage_context=None,
            headless: bool = False,
    ) -> VllmConfig:
        vllm_config = _orig_ea_create_engine_config(self, usage_context, headless)
        vllm_config.repetition_detection = _coerce_repetition_detection(
            getattr(self, _REPETITION_DETECTION_KEY, None)
        )
        return vllm_config


_relay_patch_attrs(
    EngineArgsPatch,
    _UpstreamEngineArgsPatch,
    exclude=("add_cli_args", "from_cli_args", "create_engine_config"),
)


_orig_openai_serving_chat_init = OpenAIServingChat.__init__
_orig_openai_serving_completion_init = OpenAIServingCompletion.__init__


@register_patch("OpenAIServingChatRepetitionPatch", OpenAIServingChat)
class OpenAIServingChatRepetitionPatch(VLLMPatch):
    _attr_names_to_apply = ["__init__"]

    def __init__(self, *args, **kwargs) -> None:
        _orig_openai_serving_chat_init(self, *args, **kwargs)
        _inject_default_repetition_detection(self)


@register_patch("OpenAIServingCompletionRepetitionPatch", OpenAIServingCompletion)
class OpenAIServingCompletionRepetitionPatch(VLLMPatch):
    _attr_names_to_apply = ["__init__"]

    def __init__(self, *args, **kwargs) -> None:
        _orig_openai_serving_completion_init(self, *args, **kwargs)
        _inject_default_repetition_detection(self)


@register_patch("RepetitionDetectionParamsModulePatch", sampling_params_module)
class RepetitionDetectionParamsModulePatch(VLLMPatch):
    """Expose repetition detection params through ``vllm.sampling_params``."""

    _attr_names_to_apply = ["RepetitionDetectionParams"]

    RepetitionDetectionParams = RepetitionDetectionParams


_upstream_sampling_params_from_optional = (
    _UpstreamSamplingParamsPatch.from_optional
)


@register_patch("ReasoningSamplingParamsPatch", SamplingParams)
class SamplingParamsPatch(VLLMPatch):
    """Relay thinking-budget patch, then carry repetition detection params."""

    _attr_names_to_apply = ["from_optional"]

    @staticmethod
    def from_optional(
            repetition_detection: RepetitionDetectionParams | None = None,
            **kwargs,
    ) -> SamplingParams:
       
        repetition_detection = kwargs.pop(
            _REPETITION_DETECTION_KEY,
            repetition_detection,
        )
        params = _upstream_sampling_params_from_optional(**kwargs)
        params = _set_repetition_detection(params, repetition_detection)

        return params
    

_upstream_chat_to_sampling_params = (
    _UpstreamChatCompletionRequestPatch.to_sampling_params
)


@register_patch("ChatCompletionRequestPatch", ChatCompletionRequest)
class ChatCompletionRequestPatch(VLLMPatch):
    """Relay upstream chat patches, then add repetition detection support."""

    _attr_names_to_apply = ["to_sampling_params"]

    def to_sampling_params(
            self,
            max_tokens: int,
            logits_processor_pattern: str | None,
            default_sampling_params: dict,
    ) -> SamplingParams:
        sampling_params = _upstream_chat_to_sampling_params(
            self,
            max_tokens,
            logits_processor_pattern,
            default_sampling_params,
        )
        sampling_params = _set_repetition_detection(
            sampling_params,
            _resolve_repetition_detection(
                _get_request_repetition_detection(self),
                default_sampling_params,
            ),
        )
        return sampling_params


if _UpstreamCompletionSamplingParamsPatch is not None:
    _upstream_completion_to_sampling_params = (
        _UpstreamCompletionSamplingParamsPatch.to_sampling_params
    )
else:
    _upstream_completion_to_sampling_params = CompletionRequest.to_sampling_params


@register_patch("EPDZmqCompletionSamplingParamsPatch", CompletionRequest)
class CompletionSamplingParamsPatch(VLLMPatch):
    """Relay upstream completion patches, then add repetition detection."""

    _attr_names_to_apply = ["to_sampling_params"]

    def to_sampling_params(
            self,
            max_tokens: int,
            logits_processor_pattern: str | None,
            default_sampling_params: dict | None = None,
    ) -> SamplingParams:
        sampling_params = _upstream_completion_to_sampling_params(
            self,
            max_tokens,
            logits_processor_pattern,
            default_sampling_params,
        )
        return _set_repetition_detection(
            sampling_params,
            _resolve_repetition_detection(
                _get_request_repetition_detection(self),
                default_sampling_params,
            ),
        )


def _has_repeating_pattern(
        token_ids: Sequence[int],
        pattern_len: int,
        repetition_min_count: int,
) -> bool:
    for n in range(1, pattern_len + 1):
        target_token = token_ids[-n]
        for m in range(1, repetition_min_count):
            if token_ids[-(pattern_len * m + n)] != target_token:
                return False
    return True


def check_sequence_repetition(
        token_ids: Sequence[int],
        params: RepetitionDetectionParams,
) -> bool:
    max_pattern_size = params.max_pattern_size
    min_pattern_size = params.min_pattern_size
    min_count = params.min_count

    if min_pattern_size <= 0:
        min_pattern_size = 1

    if (
            max_pattern_size <= 0
            or min_count < 2
            or min_pattern_size > max_pattern_size
    ):
        return False

    for pattern_len in range(min_pattern_size, max_pattern_size + 1):
        if pattern_len * min_count > len(token_ids):
            return False

        if _has_repeating_pattern(token_ids, pattern_len, min_count):
            return True

    return False


def check_stop(request: Request, max_model_len: int) -> bool:
    
    if _original_check_stop(request, max_model_len):
        return True

    sampling_params = request.sampling_params
    repetition_detection = _get_repetition_detection(sampling_params)
    if repetition_detection is None:
        return False
    
    if check_sequence_repetition(
            request.output_token_ids,
            repetition_detection,
    ):
        request.status = RequestStatus.FINISHED_REPETITION
        request.stop_reason = "repetition_detected"
        return True

    return False


def _extend_int_enum(
        enum_cls: type[enum.IntEnum],
        name: str,
        value: int,
) -> enum.IntEnum:
    """Register a new IntEnum member so msgspec can deserialize it."""
    if name in enum_cls._member_names_:
        return enum_cls[name]
    # A prior ``EnumCls.NEW = value`` assignment leaves a plain int attribute
    # that is not a real member and breaks msgspec validation.
    if hasattr(enum_cls, name):
        delattr(enum_cls, name)
    member = int.__new__(enum_cls, value)
    member._name_ = name
    member._value_ = value
    member.__objclass__ = enum_cls
    member._sort_order_ = len(enum_cls._member_names_)
    enum_cls._member_map_[name] = member
    enum_cls._member_names_.append(name)
    enum_cls._value2member_map_[value] = member
    return member


def _extend_finish_reason() -> None:
    if "REPETITION" in FinishReason._member_names_:
        return
    
    _extend_int_enum(FinishReason, "REPETITION", 4)
    engine_module.FINISH_REASON_STRINGS = (
        *engine_module.FINISH_REASON_STRINGS,
        "repetition",
    )


def _extend_request_status() -> None:
    if "FINISHED_REPETITION" in RequestStatus._member_names_:
        return
    
    max_value = max(member.value for member in RequestStatus)
    member = _extend_int_enum(
        RequestStatus,
        "FINISHED_REPETITION",
        max_value + 1,
    )
    request_module._FINISHED_REASON_MAP[member] = FinishReason.REPETITION


# Extend enums at import time so every process (APIServer, EngineCore, workers)
# agrees on the wire format before msgspec decodes EngineCoreOutput.
_extend_finish_reason()
_extend_request_status()


@register_patch("RepetitionDetectionRequestStatusPatch", RequestStatus)
class RequestStatusPatch(VLLMPatch):
    _attr_names_to_apply = []

    @classmethod
    def apply(cls):
        _extend_finish_reason()
        _extend_request_status()
