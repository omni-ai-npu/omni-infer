# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Non-invasive patch for APC (automatic prefix cache) metrics when PD is split.
#
# Load order: this file must load after patch_prefilled_token_skip_tokenize.py and
# patch_routed_experts.py (filename sort makes patch_serving_apc_relay >
# patch_routed / patch_prefilled) so the register_patch names below override upstream
# and relay delegates to the saved upstream methods.
#
# Changes relative to upstream vLLM:
#   1. PromptTokenUsageInfo gains a cached_rate field.
#   2. Inject prompt_cache_hit_rate() helper into the protocol module.
#   3. Inject resolve_num_cached_tokens_for_usage() into the utils module.
#      PD: read prefill_cached_tokens only from kv_transfer_params; otherwise use engine counts.
#      Denominator: prefer forwarded prefill_prompt_tokens, else usage.prompt_tokens.
#   4. serving_chat streaming: wire resolve_num_cached_tokens_for_usage and the
#      stream_usage_info closure that includes cached_rate.
#   5. serving_chat non-streaming: add cached_rate to usage; if the response already has
#      kv_transfer_params (PD, etc.) merge APC fields. In mixed deployments when kv is None,
#      do not create one; it stays null.
#   6. serving_completion streaming/non-streaming: same as (4)/(5); streaming falls back to
#      engine num_cached.

import vllm.entrypoints.openai.protocol as _protocol_module
import vllm.entrypoints.openai.utils as _utils_module

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

import json
import inspect
from collections.abc import AsyncGenerator

from vllm.entrypoints.openai.protocol import (
    PromptTokenUsageInfo,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion

from omni_npu.vllm_patches.patches.common.patch_routed_experts import (
    ExpertIdServingChatStream,
    ExpertIdServingCompletionStream,
)
from omni_npu.vllm_patches.patches.common.patch_prefilled_token_skip_tokenize import OpenAIServingChatPatch


# ---------------------------------------------------------------------------
# 1. prompt_cache_hit_rate helper + PromptTokenUsageInfo.cached_rate field
# ---------------------------------------------------------------------------

def _prompt_cache_hit_rate(cached_tokens: int, prompt_tokens: int) -> float:
    """Return cached_tokens / prompt_tokens rounded to four decimals; 0.0 if prompt is invalid."""
    if prompt_tokens <= 0:
        return 0.0
    return round(cached_tokens / prompt_tokens, 4)


def _prefill_cached_from_request(request) -> int | None:
    kv_params = getattr(request, "kv_transfer_params", None)
    if isinstance(kv_params, dict) and "prefill_cached_tokens" in kv_params:
        return int(kv_params["prefill_cached_tokens"])
    return None


@register_patch("PromptTokenUsageInfoPatch", PromptTokenUsageInfo)
class PromptTokenUsageInfoPatch(VLLMPatch):
    """Add cached_rate field to PromptTokenUsageInfo."""

    _attr_names_to_apply = ["cached_rate"]
    cached_rate: float | None = None


# ---------------------------------------------------------------------------
# 2. resolve_num_cached_tokens_for_usage helper injected into the utils module
# ---------------------------------------------------------------------------


def _resolve_num_cached_tokens_for_usage(
        request,
        engine_num_cached: int | None,
) -> int:
    """Return the cached token count to report in usage.

    If kv_transfer_params forwards prefill_cached_tokens from prefill, use that value
    (including 0); otherwise use the local engine count.
    """
    prefill_cached = _prefill_cached_from_request(request)
    if prefill_cached is not None:
        return prefill_cached
    return int(engine_num_cached or 0)


def _prompt_tokens_denominator_pd(request, usage_prompt_tokens: int) -> int:
    """Denominator for cached_rate; when forwarded, align with prefill-side prompt count."""
    kv_params = getattr(request, "kv_transfer_params", None) or {}
    prefill_prompt_tokens = kv_params.get("prefill_prompt_tokens")
    if prefill_prompt_tokens is not None:
        iv = int(prefill_prompt_tokens)
        if iv > 0:
            return iv
    return int(usage_prompt_tokens or 0)


def _merge_apc_into_kv_transfer_params_if_present(
        result,
        num_cached: int,
        prefill_prompt_tokens_eff: int,
        usage_prompt_tokens: int,
) -> None:
    """Write APC fields only when result.kv_transfer_params already exists (for PD proxy relay).

    In mixed deployments this is often None; keep null and do not create a new dict.
    """
    kv = getattr(result, "kv_transfer_params", None)
    if not isinstance(kv, dict):
        return
    pt_stored = (
        prefill_prompt_tokens_eff if prefill_prompt_tokens_eff > 0
        else int(usage_prompt_tokens or 0)
    )
    kv["prefill_cached_tokens"] = num_cached
    kv["prefill_prompt_tokens"] = pt_stored


def _engine_cached_from_usage(result) -> int | None:
    """Read prompt_tokens_details.cached_tokens from RequestOutput-like usage."""
    if (
            not getattr(result, "usage", None)
            or result.usage.prompt_tokens_details is None
    ):
        return None
    return result.usage.prompt_tokens_details.cached_tokens


# ---------------------------------------------------------------------------
# Shared streaming SSE: normalize chunks that include usage
# ---------------------------------------------------------------------------


def _normalize_usage_chunk(
        chunk: str,
        usage_num_cached: int | None,
        enable_details: bool,
        request=None,
) -> str:
    """Attach prompt_tokens_details to SSE chunks that include usage:
    null when details are disabled; when enabled and usage_num_cached is not None (including 0),
    write cached_*.

    usage_num_cached is usually from _resolve_num_cached_tokens_for_usage; in mixed deployments
    it falls back to engine count, consistent with the PD forwarding path.
    """
    if not (chunk.startswith("data: {") and '"usage"' in chunk):
        return chunk
    _, _, rest = chunk.partition("data: ")
    try:
        obj = json.loads(rest)
        usage = obj.get("usage")
        if not isinstance(usage, dict):
            return chunk

        if not enable_details or usage_num_cached is None:
            # Unify: always include this key with value null.
            usage.setdefault("prompt_tokens_details", None)
        else:
            prompt_tokens = usage.get("prompt_tokens", 0)
            prompt_tokens_denominator_pd = (
                _prompt_tokens_denominator_pd(request, prompt_tokens)
                if request is not None
                else int(prompt_tokens or 0)
            )
            if prompt_tokens_denominator_pd <= 0:
                prompt_tokens_denominator_pd = int(prompt_tokens or 0)

            details = usage.get("prompt_tokens_details")
            effective = usage_num_cached
            capped = (
                effective
                if prompt_tokens_denominator_pd <= 0
                else min(effective, prompt_tokens_denominator_pd)
            )
            cached_rate_val = _prompt_cache_hit_rate(capped, prompt_tokens_denominator_pd)

            if isinstance(details, dict):
                details["cached_tokens"] = effective
                details["cached_rate"] = cached_rate_val
            else:
                usage["prompt_tokens_details"] = {
                    "cached_tokens": effective,
                    "cached_rate": cached_rate_val,
                }
        return f"data: {json.dumps(obj)}\n\n"
    except (json.JSONDecodeError, KeyError):
        return chunk


# ---------------------------------------------------------------------------
# 3 & 4. serving_chat: streaming and non-streaming generators
# ---------------------------------------------------------------------------

_orig_chat_stream = ExpertIdServingChatStream.chat_completion_stream_generator
_orig_chat_full = OpenAIServingChatPatch.chat_completion_full_generator


# Relay-patch: same name overrides ExpertIdServingChatStream; outer wrapper normalizes
# prompt_tokens_details in streaming usage chunks.
@register_patch("ExpertIdServingChatStream", OpenAIServingChat)
class OpenAIServingChatStreamAPCPatch(VLLMPatch):
    _attr_names_to_apply = ["chat_completion_stream_generator"]

    async def chat_completion_stream_generator(
            self,
            request,
            result_generator,
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
    ) -> AsyncGenerator[str, None]:
        latest_engine_cached: int | None = None

        async def track_engine_cached():
            nonlocal latest_engine_cached
            async for res in result_generator:
                latest_engine_cached = getattr(res, "num_cached_tokens", None)
                yield res

        async for chunk in _orig_chat_stream(
                self,
                request,
                track_engine_cached(),
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
        ):
            usage_cached = _resolve_num_cached_tokens_for_usage(
                request, latest_engine_cached)
            yield _normalize_usage_chunk(
                chunk,
                usage_cached,
                self.enable_prompt_tokens_details,
                request,
            )


# Relay-patch: same name overrides PrefilledTokenSkipOpenAIServingChat; add cached_rate;
# merge APC fields if the response already has kv_transfer_params (PD); do not create kv in mixed deployments.
@register_patch("PrefilledTokenSkipOpenAIServingChat", OpenAIServingChat)
class OpenAIServingChatFullAPCPatch(VLLMPatch):
    _attr_names_to_apply = ["chat_completion_full_generator"]

    async def chat_completion_full_generator(
            self,
            request,
            result_generator,
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
    ):
        last_num_cached = None

        async def capture_num_cached():
            nonlocal last_num_cached
            async for res in result_generator:
                last_num_cached = getattr(res, "num_cached_tokens", None)
                yield res

        result = await _orig_chat_full(
            self,
            request,
            capture_num_cached(),
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
        )
        if isinstance(result, ErrorResponse):
            return result

        num_cached = _resolve_num_cached_tokens_for_usage(
            request,
            _engine_cached_from_usage(result) if last_num_cached is None
            else last_num_cached,
        )
        prefill_prompt_tokens = _prompt_tokens_denominator_pd(
            request, result.usage.prompt_tokens or 0)
        rate_num = num_cached if prefill_prompt_tokens <= 0 else min(num_cached, prefill_prompt_tokens)

        _merge_apc_into_kv_transfer_params_if_present(
            result,
            num_cached,
            prefill_prompt_tokens,
            result.usage.prompt_tokens or 0,
        )

        if not self.enable_prompt_tokens_details:
            return result

        if result.usage and result.usage.prompt_tokens_details:
            details = result.usage.prompt_tokens_details
            cached = num_cached
            details.cached_tokens = cached
            details.cached_rate = _prompt_cache_hit_rate(rate_num, prefill_prompt_tokens)
        elif result.usage:
            cached = num_cached
            result.usage.prompt_tokens_details = PromptTokenUsageInfo(
                cached_tokens=cached,
                cached_rate=_prompt_cache_hit_rate(rate_num, prefill_prompt_tokens),
            )

        return result


# ---------------------------------------------------------------------------
# 5. serving_completion: streaming and non-streaming
# ---------------------------------------------------------------------------

_orig_compl_stream = ExpertIdServingCompletionStream.completion_stream_generator
_orig_compl_create = OpenAIServingCompletion.create_completion


# Relay-patch: same name overrides ExpertIdServingCompletionStream.
@register_patch("ExpertIdServingCompletionStream", OpenAIServingCompletion)
class OpenAIServingCompletionStreamAPCPatch(VLLMPatch):
    _attr_names_to_apply = ["completion_stream_generator"]

    async def completion_stream_generator(self, request, *args, **kwargs):
        latest_engine_cached: int | None = None
        args_list = list(args)
        if len(args_list) >= 2:
            _orig_rg = args_list[1]

            async def track_engine_cached():
                nonlocal latest_engine_cached
                async for item in _orig_rg:
                    res = (
                        item[1]
                        if isinstance(item, tuple) and len(item) >= 2
                        else item
                    )
                    latest_engine_cached = getattr(res, "num_cached_tokens", None)
                    yield item

            args_list[1] = track_engine_cached()
            args = tuple(args_list)

        async for chunk in _orig_compl_stream(self, request, *args, **kwargs):
            usage_cached = _resolve_num_cached_tokens_for_usage(
                request, latest_engine_cached)
            yield _normalize_usage_chunk(
                chunk,
                usage_cached,
                self.enable_prompt_tokens_details,
                request,
            )


@register_patch("OpenAIServingCompletionAPCPatch", OpenAIServingCompletion)
class OpenAIServingCompletionAPCPatch(VLLMPatch):
    _attr_names_to_apply = ["create_completion"]

    async def create_completion(self, request, raw_request=None):
        result = await _orig_compl_create(self, request, raw_request)
        if (
                inspect.isasyncgen(result)
                or isinstance(result, ErrorResponse)
        ):
            return result

        num_cached = _resolve_num_cached_tokens_for_usage(
            request,
            _engine_cached_from_usage(result),
        )
        pt = _prompt_tokens_denominator_pd(request,
                                           result.usage.prompt_tokens or 0)
        rate_num = num_cached if pt <= 0 else min(num_cached, pt)

        _merge_apc_into_kv_transfer_params_if_present(
            result,
            num_cached,
            pt,
            result.usage.prompt_tokens or 0,
        )

        if not self.enable_prompt_tokens_details:
            return result

        if result.usage and result.usage.prompt_tokens_details:
            details = result.usage.prompt_tokens_details
            cached = num_cached
            details.cached_tokens = cached
            details.cached_rate = _prompt_cache_hit_rate(rate_num, pt)
        elif result.usage:
            cached = num_cached
            result.usage.prompt_tokens_details = PromptTokenUsageInfo(
                cached_tokens=cached,
                cached_rate=_prompt_cache_hit_rate(rate_num, pt),
            )

        return result
