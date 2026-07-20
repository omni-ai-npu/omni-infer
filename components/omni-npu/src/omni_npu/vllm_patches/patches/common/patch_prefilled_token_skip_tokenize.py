# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright contributors to the vLLM project.

# This patch is used for reuse prefilled tokens

import asyncio
import os
from collections.abc import AsyncIterator, Mapping
from typing import Any, AsyncGenerator, Optional, Callable

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam, ChatTemplateContentFormatOption
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse, ResponsesRequest,
)
from vllm.entrypoints.openai.serving_engine import OpenAIServing, ChatLikeRequest
from vllm.inputs.data import PromptType, TokensPrompt
from vllm.tool_parsers import ToolParser
from vllm.tokenizers import TokenizerLike
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput, CompletionOutput
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.request import Request, RequestStatus
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logger = init_logger(__name__)

_original_chat_completion_full_generator = OpenAIServingChat.chat_completion_full_generator
_original_preprocess_chat = OpenAIServingChat._preprocess_chat
_original_preprocess_chat_engine = OpenAIServing._preprocess_chat
_original_generate = AsyncLLM.generate
_original_update_from_kv_xfer_finished = Scheduler._update_from_kv_xfer_finished
_original_add_request = Scheduler.add_request
_ROUTED_EXPERT_KEYS = (
    "routed_experts_shape",
    "routed_experts_dtype",
    "routed_experts_str_len",
    "routed_experts_str",
)


def _get_request_max_tokens(request: Any) -> int | None:
    if isinstance(request, ChatCompletionRequest):
        return request.max_completion_tokens or request.max_tokens
    return getattr(request, "max_tokens", None)


def _speculative_margin(vllm_config: Any) -> int:
    """Tokens to reserve below max_model_len when spec decoding is on.
    max_model_len - num_speculative_tokens*3. 
    """
    spec = getattr(vllm_config, "speculative_config", None)
    if spec is None:
        return 0
    num_spec = getattr(spec, "num_speculative_tokens", 0) or 0
    return num_spec * 3


def _is_kv_producer(serving: OpenAIServing) -> bool:
    engine_client = getattr(serving, "engine_client", None)
    vllm_config = getattr(engine_client, "vllm_config", None)
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_config is not None and getattr(
        kv_transfer_config, "is_kv_transfer_instance", False
    ):
        kv_role = getattr(kv_transfer_config, "kv_role", None)
        return kv_role == "kv_producer" or getattr(
            kv_transfer_config, "is_kv_producer", False
        )
    return os.getenv("ROLE") == "prefill"


def _queue_finish_notification(
    scheduler: Scheduler, request: Request, finished_status: "RequestStatus"
) -> None:
    pending = getattr(scheduler, "_omni_pending_finish_outputs", None)
    if pending is None:
        pending = []
        scheduler._omni_pending_finish_outputs = pending
    pending.append(
        (
            request.client_index,
            EngineCoreOutput(
                request_id=request.request_id,
                new_token_ids=[],
                finish_reason=RequestStatus.get_finished_reason(finished_status),
                stop_reason=request.stop_reason,
                events=request.take_events(),
                trace_headers=request.trace_headers,
                num_cached_tokens=request.num_cached_tokens,
            ),
        )
    )


def _finish_request_and_notify_client(
    scheduler: Scheduler, request: Request, finished_status: "RequestStatus"
) -> None:
    scheduler.finish_requests(request.request_id, finished_status)
    _queue_finish_notification(scheduler, request, finished_status)


def drain_pending_finish_outputs(scheduler: Scheduler, outputs) -> None:
    """Drain queued finish notifications into update_from_output outputs."""
    pending = getattr(scheduler, "_omni_pending_finish_outputs", None)
    if not pending:
        return
    for client_index, engine_core_output in pending:
        outputs[client_index].append(engine_core_output)
    pending.clear()


def _reject_if_prompt_overflows_max_model_len(
    serving: OpenAIServing,
    request: Any,
    prompt_token_ids: Optional[list],
) -> None:
    if not prompt_token_ids:
        return
    max_model_len = getattr(serving, "max_model_len", None)
    if max_model_len is None:
        return

    engine_client = getattr(serving, "engine_client", None)
    vllm_config = getattr(engine_client, "vllm_config", None)
    spec_margin = _speculative_margin(vllm_config)
    effective_max_model_len = max_model_len - spec_margin
    limit_note = (
        ""
        if spec_margin == 0
        else f" (effective {effective_max_model_len} after reserving "
             f"{spec_margin} tokens for speculative decoding)"
    )

    token_num = len(prompt_token_ids)
    if token_num >= effective_max_model_len:
        raise VLLMValidationError(
            f"This model's maximum context length is {max_model_len} tokens"
            f"{limit_note}. However, the decode-side prompt has {token_num} "
            "input tokens, leaving no room to generate. Please reduce the "
            "length of the input messages.",
            parameter="input_tokens",
            value=token_num,
        )
    max_tokens = _get_request_max_tokens(request)
    if max_tokens is None:
        return
    reuse_prefilled_tokens = os.getenv("OMNI_REUSE_PREFILLED_TOKENS", "0") == "1"
    prefill_overhead = (
        1 if reuse_prefilled_tokens and _is_kv_producer(serving) else 0
    )
    if token_num + max_tokens + prefill_overhead > effective_max_model_len:
        raise VLLMValidationError(
            "'max_tokens' or 'max_completion_tokens' is too large: "
            f"{max_tokens}. This model's maximum context length is "
            f"{max_model_len} tokens{limit_note} and the decode-side prompt "
            f"has {token_num} input tokens "
            f"({max_tokens} > {effective_max_model_len} - {token_num}).",
            parameter="max_tokens",
            value=max_tokens,
        )


def _update_waiting_for_remote_kv_patched(self: Scheduler, request: Request) -> bool:
    assert self.connector is not None
    if request.request_id not in self.finished_recving_kv_req_ids:
        return False

    if request.request_id in self.failed_recving_kv_req_ids:
        # Request had KV load failures; num_computed_tokens was already
        # updated in _update_requests_with_invalid_blocks
        if request.num_computed_tokens:
            # Cache any valid computed tokens.
            self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
        else:
            # No valid computed tokens, release allocated blocks.
            # There may be a local cache hit on retry.
            self.kv_cache_manager.free(request)

        self.failed_recving_kv_req_ids.remove(request.request_id)
    else:
        num_computed_tokens = request.num_tokens - 1
        self.kv_cache_manager.cache_blocks(request, num_computed_tokens)
        request.num_computed_tokens = num_computed_tokens

    # Return that we are ready.
    self.finished_recving_kv_req_ids.remove(request.request_id)
    return True


async def to_async_iterator(input: RequestOutput) -> AsyncIterator[RequestOutput]:
    yield input


def _slice_first_logprobs(logprobs):
    return logprobs[:1] if logprobs else logprobs


def _first_token_cumulative_logprob(logprobs, token_id, fallback):
    first_step = logprobs[0] if logprobs else None
    if first_step:
        logprob_obj = first_step.get(token_id) or first_step.get(str(token_id))
        if isinstance(logprob_obj, dict):
            value = logprob_obj.get("logprob")
        else:
            value = getattr(logprob_obj, "logprob", None)
        if value is not None:
            return value
    return fallback


class PrefilledTextPrompt(TokensPrompt):
    """
    This is used when the model supports reusing prefilled tokens.
    """
    prefilled_token_ids: Optional[list[int]] = []
    prefilled_texts: Optional[str] = ""


@register_patch("PrefilledTokenSkipOpenAIServingChat", OpenAIServingChat)
class OpenAIServingChatPatch(VLLMPatch):
    _attr_names_to_apply = ['chat_completion_full_generator']

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list,
        tokenizer: TokenizerLike | None,
        request_metadata,
    ) -> ErrorResponse | ChatCompletionResponse:
        final_res: RequestOutput | None = None

        try:
            async for res in result_generator:
                final_res = res
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        assert final_res is not None

        reuse_prefilled_tokens = os.getenv("OMNI_REUSE_PREFILLED_TOKENS", "0") == "1"
        skip_decode_tokenize = os.getenv("OMNI_SKIP_DECODE_TOKENIZE", "0") == "1"

        if reuse_prefilled_tokens:
            prefilled_token_ids = getattr(request, "_omni_prefilled_token_ids", None)
            if prefilled_token_ids:
                output = final_res.outputs[0]
                prefilled_text = getattr(request, "_omni_prefilled_text", "")
                prefilled_logprobs = getattr(
                    request, "_omni_prefilled_logprobs", None)
                prefilled_cumulative_logprob = getattr(
                    request, "_omni_prefilled_cumulative_logprob", None)
                if prefilled_text and not output.text.startswith(prefilled_text):
                    output.text = prefilled_text + output.text
                if not output.token_ids or output.token_ids[0] != prefilled_token_ids[0]:
                    output.token_ids = prefilled_token_ids + list(output.token_ids)
                    if output.logprobs is not None and prefilled_logprobs:
                        output.logprobs = prefilled_logprobs + output.logprobs
                    if (prefilled_cumulative_logprob is not None
                            and output.cumulative_logprob is not None):
                        output.cumulative_logprob += prefilled_cumulative_logprob
        ## In Prefill node, the response will carry prompt_token_ids with kv_transfer_params

        if skip_decode_tokenize:
            if final_res.kv_transfer_params:
                final_res.kv_transfer_params["prompt_token_ids"] = final_res.prompt_token_ids
        if reuse_prefilled_tokens:
            if final_res.kv_transfer_params:
                prefilled_token_id = final_res.outputs[0].token_ids[0]
                prefilled_logprobs = _slice_first_logprobs(final_res.outputs[0].logprobs)
                final_res.kv_transfer_params["prefilled_token"] = [prefilled_token_id]
                final_res.kv_transfer_params["stop_reasons"] = [
                    output.stop_reason for output in final_res.outputs
                ]
                final_res.kv_transfer_params["prefilled_logprobs"] = prefilled_logprobs
                final_res.kv_transfer_params[
                    "prefilled_cumulative_logprob"
                ] = _first_token_cumulative_logprob(
                    prefilled_logprobs,
                    prefilled_token_id,
                    final_res.outputs[0].cumulative_logprob,
                )

        request_kv_payload = None
        if request.kv_transfer_params and all(
            key in request.kv_transfer_params for key in _ROUTED_EXPERT_KEYS
        ):
            request_kv_payload = {
                key: request.kv_transfer_params[key] for key in _ROUTED_EXPERT_KEYS
            }

        is_prefill_node = _is_kv_producer(self)

        if is_prefill_node:
            for output in final_res.outputs:
                routed_experts = getattr(output, "routed_experts", None)
                if routed_experts is None or getattr(routed_experts, "shape", None) is None:
                    continue
                if routed_experts.shape[0] == 0:
                    continue
                if final_res.kv_transfer_params is None:
                    final_res.kv_transfer_params = {}
                self.add_ndarray_info_to_dict(
                    routed_experts,
                    final_res.kv_transfer_params,
                )
                break

        response = await _original_chat_completion_full_generator(self, request, to_async_iterator(final_res), request_id, model_name, conversation, tokenizer, request_metadata)
        if isinstance(response, ErrorResponse):
            return response

        payloads = []
        for output in final_res.outputs:
            routed_experts = getattr(output, "routed_experts", None)
            if routed_experts is not None and getattr(routed_experts, "shape", None) is not None:
                if routed_experts.shape[0] == 0:
                    routed_experts = None

            if routed_experts is not None:
                if request_kv_payload is not None:
                    routed_experts = self.concatenate_dict_and_ndarray(
                        request_kv_payload,
                        routed_experts,
                    )
                payload = {}
                self.add_ndarray_info_to_dict(routed_experts, payload)
                payloads.append(payload)
            else:
                payloads.append(request_kv_payload)

        for choice, payload in zip(response.choices, payloads):
            if payload is not None:
                choice.routed_experts = payload
        return response


# Not registered as a patch — patch_input_ids_piggyback owns
# OpenAIServingChat._preprocess_chat and calls this class's method as the
# next link in the relay chain (input_ids_piggyback -> here -> upstream).
class OpenAIServingChatPreprocessPatch(VLLMPatch):
    _attr_names_to_apply = ['_preprocess_chat']

    async def _preprocess_chat(
            self,
            request: ChatLikeRequest | ResponsesRequest,
            tokenizer: TokenizerLike | None,
            messages: list[ChatCompletionMessageParam],
            chat_template: str | None,
            chat_template_content_format: ChatTemplateContentFormatOption,
            add_generation_prompt: bool = True,
            continue_final_message: bool = False,
            tool_dicts: list[dict[str, Any]] | None = None,
            documents: list[dict[str, str]] | None = None,
            chat_template_kwargs: dict[str, Any] | None = None,
            default_chat_template_kwargs: dict[str, Any] | None = None,
            tool_parser: Callable[[TokenizerLike], ToolParser] | None = None,
            add_special_tokens: bool = False,
    ):
        conversation, [engine_prompt] = await _original_preprocess_chat(
            self, request, tokenizer, messages, chat_template,
            chat_template_content_format, add_generation_prompt,
            continue_final_message, tool_dicts, documents,
            chat_template_kwargs, default_chat_template_kwargs, tool_parser,
            add_special_tokens)
        reuse_prefilled_tokens = os.getenv("OMNI_REUSE_PREFILLED_TOKENS",
                                           "0") == "1"
        for attr in (
                "_omni_prefilled_token_ids",
                "_omni_prefilled_text",
                "_omni_prefilled_logprobs",
                "_omni_prefilled_cumulative_logprob",
        ):
            if hasattr(request, attr):
                delattr(request, attr)
        if reuse_prefilled_tokens:
            if request.kv_transfer_params and "prefilled_token" in request.kv_transfer_params:
                new_tokens = tokenizer.convert_ids_to_tokens(
                    request.kv_transfer_params["prefilled_token"][0])
                delta_text = tokenizer.convert_tokens_to_string([new_tokens])
                # If the decoded text ends with '�', it indicates an incomplete UTF-8 sequence — abandon reuse.
                # If the prefilled side has already triggered a stop reason, we also fall back to normal generation.
                if delta_text.endswith("�") or any(s is not None for s in request.kv_transfer_params["stop_reasons"]) or \
                        request.kv_transfer_params["prefilled_token"][0] == tokenizer.eos_token_id:
                    request.kv_transfer_params.pop("prefilled_token", None)
                else:
                    engine_prompt[
                        "prefilled_token_ids"] = request.kv_transfer_params[
                            "prefilled_token"]
                    engine_prompt["prefilled_texts"] = delta_text
                    prefilled_logprobs = _slice_first_logprobs(
                        request.kv_transfer_params["prefilled_logprobs"])
                    prefilled_token_id = request.kv_transfer_params[
                        "prefilled_token"][0]
                    prefilled_cumulative_logprob = _first_token_cumulative_logprob(
                        prefilled_logprobs,
                        prefilled_token_id,
                        request.kv_transfer_params["prefilled_cumulative_logprob"],
                    )
                    engine_prompt["prefilled_logprobs"] = prefilled_logprobs
                    engine_prompt[
                        "prefilled_cumulative_logprob"] = prefilled_cumulative_logprob
                    request._omni_prefilled_token_ids = request.kv_transfer_params[
                        "prefilled_token"]
                    request._omni_prefilled_text = delta_text
                    request._omni_prefilled_logprobs = prefilled_logprobs
                    request._omni_prefilled_cumulative_logprob = (
                        prefilled_cumulative_logprob)
        _reject_if_prompt_overflows_max_model_len(
            self, request, engine_prompt.get("prompt_token_ids"))
        return conversation, [engine_prompt]


@register_patch("PrefilledTokenSkipOpenAIServing", OpenAIServing)
class OpenAIServingPatch(VLLMPatch):
    _attr_names_to_apply = ['_preprocess_chat']

    async def _preprocess_chat(
        self,
        request: Any,
        tokenizer: TokenizerLike | None,
        messages: list,
        chat_template: str | None,
        chat_template_content_format: str,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tool_dicts: list[dict[str, Any]] | None = None,
        documents: list[dict[str, str]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        default_chat_template_kwargs: dict[str, Any] | None = None,
        tool_parser=None,
        add_special_tokens: bool = False,
    ):
        conversation, [engine_prompt] = await _original_preprocess_chat(
            self, request, tokenizer, messages, chat_template,
            chat_template_content_format, add_generation_prompt,
            continue_final_message, tool_dicts, documents,
            chat_template_kwargs, default_chat_template_kwargs, tool_parser,
            add_special_tokens)
        if request.kv_transfer_params and "prompt_token_ids" in request.kv_transfer_params:
            engine_prompt = PrefilledTextPrompt(
                prompt_token_ids=request.kv_transfer_params["prompt_token_ids"])
        _reject_if_prompt_overflows_max_model_len(
            self, request, engine_prompt.get("prompt_token_ids"))
        return conversation, [engine_prompt]


@register_patch("PrefilledTokenSkipScheduler", Scheduler)
class SchedulerPatch(VLLMPatch):
    _attr_names_to_apply = ['_update_waiting_for_remote_kv']

    def _update_waiting_for_remote_kv(self, request: Request) -> bool:
        reuse_prefilled_tokens = os.getenv("OMNI_REUSE_PREFILLED_TOKENS", "0") == "1"
        if (reuse_prefilled_tokens and
                request.request_id in self.finished_recving_kv_req_ids and
                request.request_id not in self.failed_recving_kv_req_ids):
            extra_args = getattr(request.sampling_params, "extra_args", None) or {}
            kv_params = extra_args.get("kv_transfer_params")
            prefilled_token = kv_params.get("prefilled_token") if kv_params else None
            if prefilled_token:
                new_len = len(request.prompt_token_ids) + len(prefilled_token)
                if new_len < self.max_model_len:
                    request.prompt_token_ids.extend(prefilled_token)
                    request.append_output_token_ids(prefilled_token)
                else:
                    logger.warning(
                        "Skipping prefilled_token append for request %s at "
                        "the context boundary (prompt would reach %d tokens, "
                        "max_model_len=%d); falling back to normal "
                        "generation.",
                        request.request_id, new_len, self.max_model_len,
                    )
                kv_params.pop("prefilled_token", None)

        return _update_waiting_for_remote_kv_patched(self, request)


@register_patch("PrefilledTokenSkipKvXferFinished", Scheduler)
class SchedulerKvXferFinishedPatch(VLLMPatch):
    _attr_names_to_apply = ['_update_from_kv_xfer_finished']

    def _update_from_kv_xfer_finished(self, kv_connector_output) -> None:
        _original_update_from_kv_xfer_finished(self, kv_connector_output)

        reuse_prefilled_tokens = os.getenv("OMNI_REUSE_PREFILLED_TOKENS", "0") == "1"
        if not reuse_prefilled_tokens:
            return

        for req_id in kv_connector_output.finished_recving or ():
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                continue

            if req_id in self.failed_recving_kv_req_ids:
                continue

            extra_args = getattr(request.sampling_params, "extra_args", None) or {}
            kv_params = extra_args.get("kv_transfer_params")
            prefilled_token = kv_params.get("prefilled_token") if kv_params else None
            if not prefilled_token:
                continue

            request.prompt_token_ids.extend(prefilled_token)
            request.append_output_token_ids(prefilled_token)
            kv_params.pop("prefilled_token", None)

            if check_stop(request, self.max_model_len):
                finished_status = request.status
                if (finished_status == RequestStatus.FINISHED_LENGTH_CAPPED
                        and request.num_output_tokens < request.max_tokens):
                    logger.error(
                        "Request %s: prefilled token exhausts "
                        "max_model_len before max_tokens (prompt_len=%d, "
                        "max_tokens=%d, max_model_len=%d)",
                        req_id, request.num_tokens, request.max_tokens,
                        self.max_model_len,
                    )
                    finished_status = RequestStatus.FINISHED_ERROR
                request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                self.finished_recving_kv_req_ids.discard(req_id)
                _finish_request_and_notify_client(self, request, finished_status)


@register_patch("PrefilledTokenSkipSchedulerAddRequestGuard", Scheduler)
class SchedulerAddRequestGuardPatch(VLLMPatch):
    _attr_names_to_apply = ['add_request']

    def add_request(self, request: Request) -> None:
        pd_flags_enabled = (
            os.getenv("OMNI_REUSE_PREFILLED_TOKENS", "0") == "1"
            or os.getenv("OMNI_SKIP_DECODE_TOKENIZE", "0") == "1"
        )
        if not pd_flags_enabled or request.sampling_params is None:
            _original_add_request(self, request)
            return

        max_tokens = request.max_tokens or 1
        spec_margin = _speculative_margin(getattr(self, "vllm_config", None))
        effective_max_model_len = self.max_model_len - spec_margin
        if request.num_tokens + max_tokens > effective_max_model_len:
            logger.error(
                "Rejecting request %s: prompt_len(%d) + max_tokens(%d) > "
                "effective max_model_len(%d) (max_model_len=%d, spec_margin=%d); "
                "failing instead of scheduling it.",
                request.request_id, request.num_tokens, max_tokens,
                effective_max_model_len, self.max_model_len, spec_margin,
            )
            _original_add_request(self, request)
            _finish_request_and_notify_client(
                self, request, RequestStatus.FINISHED_ERROR)
            return
        _original_add_request(self, request)


@register_patch("PrefilledTokenSkipAsyncLLM", AsyncLLM)
class AsyncLLMPatch(VLLMPatch):
    _attr_names_to_apply = ['generate']

    async def generate(
        self,
        prompt: EngineCoreRequest | PromptType,
        sampling_params,
        request_id: str,
        *,
        prompt_text: str | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
    )-> AsyncGenerator[RequestOutput, None]:
        from collections import namedtuple
        from typing import List, Dict, Any
        from vllm.sampling_params import RequestOutputKind

        Logprob = namedtuple('Logprob', ['logprob', 'rank', 'decoded_token'])

        
        def convert_to_standard_logprobs(prefilled_logprobs: List[Dict[str, Any]]) -> List[Dict[int, Logprob]]:
            if prefilled_logprobs is None:
                return None
            result = []
            for item in prefilled_logprobs:
                new_item = {}
                for token_id_str, data in item.items():
                    token_id = int(token_id_str)
                    logprob_obj = Logprob(
                        logprob=data['logprob'],
                        rank=data['rank'],
                        decoded_token=data['decoded_token']
                    )
                    new_item[token_id] = logprob_obj
                result.append(new_item)
            return result

        reuse_prefilled_tokens = os.getenv("OMNI_REUSE_PREFILLED_TOKENS", "0") == "1"
        pending_prefilled_output = None
        if reuse_prefilled_tokens:
            prefilled_token_ids: list[int] = []
            prompt_token_ids: list[int] | None = None
            prefilled_text = ""
            prefilled_logprobs = None
            prefill_cumulative_logprob = None

            if isinstance(prompt, EngineCoreRequest):
                prompt_token_ids = prompt.prompt_token_ids
                extra_args = getattr(sampling_params, "extra_args", None) or {}
                kv_transfer_params = extra_args.get("kv_transfer_params", None)
                if kv_transfer_params and "prefilled_token" in kv_transfer_params:
                    prefilled_token_ids = kv_transfer_params["prefilled_token"] or []
                    if prefilled_token_ids:
                        tokenizer = getattr(self.input_processor, "tokenizer", None)
                        if tokenizer is not None:
                            token = tokenizer.convert_ids_to_tokens(prefilled_token_ids[0])
                            prefilled_text = tokenizer.convert_tokens_to_string([token])

                    prefilled_logprobs = convert_to_standard_logprobs(kv_transfer_params["prefilled_logprobs"])
                    prefill_cumulative_logprob = kv_transfer_params["prefilled_cumulative_logprob"]

            elif isinstance(prompt, Mapping):
                if "prefilled_token_ids" in prompt:
                    prefilled_token_ids = prompt["prefilled_token_ids"] or []
                    prompt_token_ids = prompt.get("prompt_token_ids")
                    prefilled_text = prompt.get("prefilled_texts", "")
                    prompt["prefilled_token_ids"] = []
                    prefilled_logprobs = convert_to_standard_logprobs(
                        prompt.get("prefilled_logprobs"))
                    prefill_cumulative_logprob = prompt.get(
                        "prefilled_cumulative_logprob")

            if prefilled_token_ids:
                synthetic_outputs = [
                    CompletionOutput(
                        index=idx,
                        cumulative_logprob=prefill_cumulative_logprob,
                        logprobs=prefilled_logprobs,
                        text=prefilled_text,
                        token_ids=prefilled_token_ids,
                    )
                    for idx in range(sampling_params.n)
                ]
                synthetic_res = RequestOutput(
                    request_id=request_id,
                    prompt=None,
                    finished=False,
                    prompt_logprobs=None,
                    prompt_token_ids=prompt_token_ids,
                    outputs=synthetic_outputs,
                )
                if getattr(sampling_params, "output_kind", None) == RequestOutputKind.DELTA:
                    yield synthetic_res
                else:
                    pending_prefilled_output = synthetic_res
        async for res in _original_generate(self, prompt, sampling_params, request_id, prompt_text=prompt_text, lora_request=lora_request,
                                 tokenization_kwargs=tokenization_kwargs, trace_headers=trace_headers, priority=priority, data_parallel_rank=data_parallel_rank):
            if pending_prefilled_output is not None:
                by_index = {
                    output.index: output
                    for output in pending_prefilled_output.outputs
                }
                for output in res.outputs:
                    prefilled_output = by_index.get(output.index)
                    if prefilled_output is None:
                        continue
                    prefilled_ids = list(prefilled_output.token_ids)
                    if output.token_ids and output.token_ids[0] == prefilled_ids[0]:
                        continue
                    output.token_ids = prefilled_ids + list(output.token_ids)
                    if (prefilled_output.text
                            and not output.text.startswith(prefilled_output.text)):
                        output.text = prefilled_output.text + output.text
                    if output.logprobs is not None and prefilled_output.logprobs:
                        output.logprobs = prefilled_output.logprobs + output.logprobs
                    if (output.cumulative_logprob is not None
                            and prefilled_output.cumulative_logprob is not None):
                        output.cumulative_logprob += (
                            prefilled_output.cumulative_logprob)
                pending_prefilled_output = None
            yield res
