# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# EPD ZMQ: merges ``ec_connector_metadata`` into ``EngineCoreOutput.kv_transfer_params``
# after ``Scheduler.update_from_output``, propagates ``ec_transfer_params`` on API responses,
# and threads params through the engine/output path.
#
# This file must load *after* ``patch_routed_experts.py`` (``patch_ti_epd_zmq`` sorts later)
# so relay registrations overwrite the same ``register_patch`` names and call upstream.
#

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import torch

from vllm import SamplingParams
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.outputs import RequestOutput
from vllm.tokenizers import TokenizerLike
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.engine.output_processor import OutputProcessor, RequestState
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request

from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.vllm_patches.patches.common.patch_routed_experts import (
    ExpertIdServingCompletionFinal as _UpstreamExpertIdServingCompletion,
    RequestOutputRoutedExpertsAggregationPatch as _UpstreamRequestOutputRoutedAgg,
    SchedulerRoutedExpertsPatch as _UpstreamSchedulerRoutedExpertsPatch,
)

_EC_PARAMS_KEY = "__epd_ec_transfer_params__"

_original_request_init = Request.__init__
_original_request_output_init = RequestOutput.__init__

_upstream_reqout_add = _UpstreamRequestOutputRoutedAgg.add


@register_patch("EPDZmqRequestPatch", Request)
class RequestPatch(VLLMPatch):
    _attr_names_to_apply = ["__init__"]

    def __init__(self, *args, **kwargs):
        _original_request_init(self, *args, **kwargs)
        self.ec_transfer_params = None
        sampling_params = getattr(self, "sampling_params", None)
        if sampling_params is not None and sampling_params.extra_args is not None:
            self.ec_transfer_params = sampling_params.extra_args.get(
                "ec_transfer_params"
            )


# Relay: same register name as patch_routed_experts ``RequestOutputRoutedExpertsAggregation``.
@register_patch("RequestOutputRoutedExpertsAggregation", RequestOutput)
class RequestOutputRoutedExpertsEPDRelayPatch(VLLMPatch):
    """Relay-patch"""
    _attr_names_to_apply = ["__init__", "add"]

    def __init__(self, *args, **kwargs):
        ec_transfer_params = kwargs.pop("ec_transfer_params", None)
        _original_request_output_init(self, *args, **kwargs)
        self.ec_transfer_params = ec_transfer_params

    def add(self, next_output: RequestOutput, aggregate: bool) -> None:
        _upstream_reqout_add(self, next_output, aggregate)
        self.ec_transfer_params = getattr(next_output, "ec_transfer_params", None)


_original_request_state_make_request_output = RequestState.make_request_output
_original_output_processor_update_stats = OutputProcessor._update_stats_from_output


@register_patch("EPDZmqRequestStatePatch", RequestState)
class RequestStatePatch(VLLMPatch):
    _attr_names_to_apply = ["make_request_output"]

    def make_request_output(
            self,
            new_token_ids: list[int],
            pooling_output: torch.Tensor | None,
            finish_reason: FinishReason | None,
            stop_reason: int | str | None,
            kv_transfer_params: dict[str, Any] | None = None,
            routed_experts: np.ndarray | None = None,
            ec_transfer_params: dict[str, Any] | None = None,
    ):
        output = _original_request_state_make_request_output(
            self,
            new_token_ids,
            pooling_output,
            finish_reason,
            stop_reason,
            kv_transfer_params,
            routed_experts,
        )
        if output is not None and isinstance(output, RequestOutput):
            ec_params = ec_transfer_params
            if ec_params is None:
                ec_params = getattr(self, "_epd_ec_transfer_params", None)
            output.ec_transfer_params = ec_params
        return output


@register_patch("EPDZmqOutputProcessorPatch", OutputProcessor)
class OutputProcessorPatch(VLLMPatch):
    _attr_names_to_apply = ["_update_stats_from_output"]

    def _update_stats_from_output(
            self,
            req_state: RequestState,
            engine_core_output,
            engine_core_timestamp: float | None,
            iteration_stats,
    ):
        kv_transfer_params = getattr(engine_core_output, "kv_transfer_params", None)
        ec_transfer_params = None
        if isinstance(kv_transfer_params, dict):
            ec_transfer_params = kv_transfer_params.pop(_EC_PARAMS_KEY, None)
        req_state._epd_ec_transfer_params = ec_transfer_params
        return _original_output_processor_update_stats(
            self,
            req_state,
            engine_core_output,
            engine_core_timestamp,
            iteration_stats,
        )


_original_completion_to_sampling_params = CompletionRequest.to_sampling_params


@register_patch("EPDZmqCompletionSamplingParamsPatch", CompletionRequest)
class CompletionSamplingParamsPatch(VLLMPatch):
    _attr_names_to_apply = ["to_sampling_params"]

    def to_sampling_params(
            self,
            max_tokens: int,
            logits_processor_pattern: str | None,
            default_sampling_params: dict | None = None,
    ):
        params = _original_completion_to_sampling_params(
            self, max_tokens, logits_processor_pattern, default_sampling_params
        )
        ec_transfer_params = getattr(self, "ec_transfer_params", None)
        if ec_transfer_params:
            extra_args = params.extra_args or {}
            extra_args["ec_transfer_params"] = ec_transfer_params
            params.extra_args = extra_args
        return params


_upstream_completion = _UpstreamExpertIdServingCompletion.request_output_to_completion_response


# Relay: same register name as ``ExpertIdServingCompletionFinal`` in patch_routed_experts.
@register_patch("ExpertIdServingCompletionFinal", OpenAIServingCompletion)
class ExpertIdServingCompletionEPDRelayPatch(VLLMPatch):
    """Relay-patch"""
    _attr_names_to_apply = ["request_output_to_completion_response"]

    def request_output_to_completion_response(
            self,
            final_res_batch: list[RequestOutput],
            request,
            request_id: str,
            created_time: int,
            model_name: str,
            tokenizer,
            request_metadata,
    ):
        response = _upstream_completion(
            self,
            final_res_batch,
            request,
            request_id,
            created_time,
            model_name,
            tokenizer,
            request_metadata,
        )
        if final_res_batch:
            setattr(
                response,
                "ec_transfer_params",
                getattr(final_res_batch[0], "ec_transfer_params", None),
            )
        return response


# --- Scheduler: fold ec_connector_metadata into kv_transfer_params (EPD ZMQ) ---


def _add_ec_metadata(
        engine_core_output: EngineCoreOutput,
        ec_connector_metadata,
) -> None:
    kv_transfer_params = engine_core_output.kv_transfer_params
    if kv_transfer_params is None:
        kv_transfer_params = {}
    else:
        kv_transfer_params = dict(kv_transfer_params)
    kv_transfer_params[_EC_PARAMS_KEY] = ec_connector_metadata
    engine_core_output.kv_transfer_params = kv_transfer_params


def _inject_ec_metadata(
        scheduler,
        scheduler_output,
        model_runner_output: ModelRunnerOutput,
        engine_core_outputs: dict[int, EngineCoreOutputs],
) -> dict[int, EngineCoreOutputs]:
    ec_connector_metadata = scheduler_output.ec_connector_metadata
    if not ec_connector_metadata:
        return engine_core_outputs

    output_req_ids: set[str] = set()
    for client_outputs in engine_core_outputs.values():
        for output in client_outputs.outputs:
            _add_ec_metadata(output, ec_connector_metadata)
            output_req_ids.add(output.request_id)

    prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict or {}
    for req_id in scheduler_output.num_scheduled_tokens:
        if req_id in output_req_ids:
            continue

        request = scheduler.requests.get(req_id)
        if request is None:
            continue

        output = EngineCoreOutput(
            request_id=req_id,
            new_token_ids=[],
            finish_reason=request.get_finished_reason(),
            new_prompt_logprobs_tensors=prompt_logprobs_dict.get(req_id),
            stop_reason=request.stop_reason,
            events=request.take_events(),
            kv_transfer_params={_EC_PARAMS_KEY: ec_connector_metadata},
            trace_headers=request.trace_headers,
            num_cached_tokens=request.num_cached_tokens,
            num_nans_in_logits=request.num_nans_in_logits,
        )

        client_outputs = engine_core_outputs.get(request.client_index)
        if client_outputs is None:
            engine_core_outputs[request.client_index] = EngineCoreOutputs(
                outputs=[output]
            )
        else:
            client_outputs.outputs.append(output)

    return engine_core_outputs


_original_omni_update_from_output = _UpstreamSchedulerRoutedExpertsPatch.update_from_output


# Relay: same register name as ``SchedulerRoutedExpertsPatch`` in patch_routed_experts.
@register_patch("SchedulerRoutedExpertsPatch", Scheduler)
class SchedulerRoutedExpertsEPDRelayPatch(VLLMPatch):
    """Relay-patch"""
    _attr_names_to_apply = ["update_from_output"]

    def update_from_output(
            self,
            scheduler_output,
            model_runner_output: ModelRunnerOutput,
    ):
        engine_core_outputs = _original_omni_update_from_output(
            self, scheduler_output, model_runner_output
        )
        return _inject_ec_metadata(
            self,
            scheduler_output,
            model_runner_output,
            engine_core_outputs,
        )


# --- OpenAIServingChat: relay-patch to propagate ec_transfer_params onto response ---

from omni_npu.vllm_patches.patches.common.patch_serving_apc import (
    OpenAIServingChatFullAPCPatch as _UpstreamOpenAIServingChatPatch,
)

_upstream_chat_completion_full_generator = _UpstreamOpenAIServingChatPatch.chat_completion_full_generator


class _TrackingIterator:
    """Wraps an async iterator and records the last yielded RequestOutput."""

    def __init__(self, inner: AsyncIterator[RequestOutput]) -> None:
        self._inner = inner
        self.last: RequestOutput | None = None

    def __aiter__(self):
        return self

    async def __anext__(self) -> RequestOutput:
        item = await self._inner.__anext__()
        self.last = item
        return item


@register_patch("PrefilledTokenSkipOpenAIServingChat", OpenAIServingChat)
class OpenAIServingChatPatch(VLLMPatch):
    """Relay-patch: calls the upstream prefilled-token patch, then appends
    ec_transfer_params from the EPD ZMQ pipeline onto the response."""

    _attr_names_to_apply = ["chat_completion_full_generator"]

    async def chat_completion_full_generator(
            self,
            request: ChatCompletionRequest,
            result_generator: AsyncIterator[RequestOutput],
            request_id: str,
            model_name: str,
            conversation: list,
            tokenizer: TokenizerLike | None,
            request_metadata: Any,
    ) -> ErrorResponse | ChatCompletionResponse:
        tracker = _TrackingIterator(result_generator)
        response = await _upstream_chat_completion_full_generator(
            self,
            request,
            tracker,
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
        )
        if isinstance(response, ErrorResponse):
            return response
        response.ec_transfer_params = getattr(
            tracker.last, "ec_transfer_params", None
        )
        return response


# --- ChatCompletionRequest: relay-patch to inject ec_transfer_params into extra_args ---

from omni_npu.vllm_patches.patches.common.patch_thinking_limit import (
    ChatCompletionRequestPatch as _UpstreamChatCompletionRequestPatch,
)

_upstream_to_sampling_params = _UpstreamChatCompletionRequestPatch.to_sampling_params


@register_patch("ChatCompletionRequestPatch", ChatCompletionRequest)
class ChatCompletionRequestPatch(VLLMPatch):
    """Relay-patch: calls the upstream thinking_token_budget patch, then
    injects ec_transfer_params into extra_args for EPD ZMQ pipeline."""

    _attr_names_to_apply = ["to_sampling_params"]

    def to_sampling_params(
            self,
            max_tokens: int,
            logits_processor_pattern: str | None,
            default_sampling_params: dict,
    ) -> SamplingParams:

        sampling_params = _upstream_to_sampling_params(
            self, max_tokens, logits_processor_pattern, default_sampling_params
        )
        ec_transfer_params = getattr(self, "ec_transfer_params", None)
        if ec_transfer_params:
            if sampling_params.extra_args is None:
                sampling_params.extra_args = {}
            sampling_params.extra_args["ec_transfer_params"] = ec_transfer_params
        return sampling_params
