# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright contributors to the vLLM project.

"""APC (prefix cache) usage reporting -- adapted to vLLM 0.25.1.

What 0.25.1 already does natively, and this patch therefore no longer does:
  * write ``usage.prompt_tokens_details.cached_tokens`` from the local
    ``RequestOutput.num_cached_tokens`` on all four paths -- see
    ``chat_completion/serving.py::_make_prompt_tokens_details`` (used at :755 and
    :1020) and ``completion/serving.py:451,606``;
  * gate that on ``--enable-prompt-tokens-details``;
  * carry ``kv_transfer_params`` on both the request and the response models;
  * report ``cached_tokens == 0`` faithfully.  0.14 used a truthiness test
    (``and num_cached_tokens``) which silently dropped a genuine 0; 0.25.1 uses
    ``is not None``.

What is still missing upstream, i.e. what this patch is reduced to:
  1. ``cached_rate`` -- ``PromptTokenUsageInfo`` has ``cached_tokens`` and
     ``multimodal_tokens``, but no hit-rate field.
  2. PD relay -- the prefix-cache hit happens on the P node, but the client sees
     D's response.  On D, ``num_cached_tokens = local + external`` and the
     connector reports the whole prompt as "external" (it is about to fetch all of
     it from P), so the reported hit rate is pinned at **100%** -- for a stone
     cold first request too.  That is worse than reporting 0: it looks like the
     cache is working perfectly.  P writes its real numbers into
     ``kv_transfer_params``; the proxy relays that dict into the D request; D
     reads them back and overwrites the fake value.
     Measured, 4016-token prompt: without this patch D reports 4016 (1.0) on a
     cold request; with it, 0 on cold and 3968 (0.988) when warm, matching P's
     own ``vllm:prefix_cache_hits_total``.
  3. SSE key stability -- the final usage chunk is dumped with
     ``exclude_none=True`` (``chat_completion/serving.py:786``,
     ``completion/serving.py:483``), so ``prompt_tokens_details`` disappears
     entirely rather than becoming ``null`` when details are disabled.
"""

# Keeps `X | None` annotations from being evaluated at import time, so this
# module also imports under the Python 3.9 that lives outside the container
# (the offline tests import it directly).  Same convention as
# patch_routed_experts.py.
from __future__ import annotations

import importlib
import inspect
import json
from collections.abc import AsyncGenerator
from typing import Any

from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    PromptTokenUsageInfo,
)

from vllm.logger import init_logger

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# 1. cached_rate
# ---------------------------------------------------------------------------
#
# No patch is needed to put cached_rate on PromptTokenUsageInfo: OpenAIBaseModel
# declares ConfigDict(extra="allow"), so pydantic accepts and serialises the key
# on both paths we use -- construction and assignment onto an existing instance.
# Verified against the 0.25.1 container image:
#   PromptTokenUsageInfo(cached_tokens=800, cached_rate=0.8).model_dump_json()
#   d = PromptTokenUsageInfo(cached_tokens=800); d.cached_rate = 0.8
# both emit {"cached_tokens":800,"multimodal_tokens":null,"cached_rate":0.8}.


def _prompt_cache_hit_rate(cached_tokens: int, prompt_tokens: int) -> float:
    """cached_tokens / prompt_tokens rounded to four decimals; 0.0 if prompt is invalid."""
    if prompt_tokens <= 0:
        return 0.0
    return round(cached_tokens / prompt_tokens, 4)


# ---------------------------------------------------------------------------
# 2. PD relay: read from the request, write back to the response
# ---------------------------------------------------------------------------


def _resolve_num_cached_tokens_for_usage(
        request,
        engine_num_cached: int | None,
) -> int:
    """Cached token count to report in usage.

    If ``kv_transfer_params`` forwards ``prefill_cached_tokens`` from the P node, use
    that value -- including 0, hence the ``in`` test rather than a truthiness test.
    Otherwise fall back to the local engine count (mixed / non-PD deployment).
    """
    kv_params = getattr(request, "kv_transfer_params", None)
    if isinstance(kv_params, dict) and "prefill_cached_tokens" in kv_params:
        return int(kv_params["prefill_cached_tokens"])
    return int(engine_num_cached or 0)


def _prompt_tokens_denominator_pd(request, usage_prompt_tokens: int) -> int:
    """Denominator for cached_rate.

    When forwarded, align with the P-side prompt count: with prefilled-token skip the
    D node can see a shorter prompt than P did, and dividing P's numerator by D's
    denominator would yield a rate above 1.0.
    """
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
    """Write the APC fields only when ``result.kv_transfer_params`` already exists.

    In a mixed deployment it is None; leave it null instead of creating a dict that
    would make the response look like a PD one to the proxy.
    """
    kv = getattr(result, "kv_transfer_params", None)
    if not isinstance(kv, dict):
        return
    kv["prefill_cached_tokens"] = num_cached
    kv["prefill_prompt_tokens"] = (
        prefill_prompt_tokens_eff if prefill_prompt_tokens_eff > 0
        else int(usage_prompt_tokens or 0)
    )


def _engine_cached_from_usage(result) -> int | None:
    """Read back the cached_tokens upstream already wrote into the response usage.

    Returns None when ``--enable-prompt-tokens-details`` is off, since upstream then
    leaves ``prompt_tokens_details`` unset.
    """
    usage = getattr(result, "usage", None)
    if usage is None or usage.prompt_tokens_details is None:
        return None
    return usage.prompt_tokens_details.cached_tokens


def _apply_apc_to_response(self, request, result, engine_cached: int | None = None) -> None:
    """Non-streaming tail shared by chat and completion.

    ``engine_cached`` lets the caller supply a value captured off the result
    generator; it is the only source when details are disabled, because upstream then
    writes nothing into usage for us to read back.
    """
    if engine_cached is None:
        engine_cached = _engine_cached_from_usage(result)

    num_cached = _resolve_num_cached_tokens_for_usage(request, engine_cached)
    prompt_tokens = int(result.usage.prompt_tokens or 0) if result.usage else 0
    denominator = _prompt_tokens_denominator_pd(request, prompt_tokens)
    rate_num = num_cached if denominator <= 0 else min(num_cached, denominator)

    _merge_apc_into_kv_transfer_params_if_present(
        result, num_cached, denominator, prompt_tokens)

    if not self.enable_prompt_tokens_details or not result.usage:
        return

    cached_rate = _prompt_cache_hit_rate(rate_num, denominator)
    details = result.usage.prompt_tokens_details
    if details is not None:
        details.cached_tokens = num_cached
        details.cached_rate = cached_rate
    else:
        result.usage.prompt_tokens_details = PromptTokenUsageInfo(
            cached_tokens=num_cached,
            cached_rate=cached_rate,
        )


# ---------------------------------------------------------------------------
# 3. Shared streaming SSE rewrite
# ---------------------------------------------------------------------------


def _normalize_usage_chunk(
        chunk: str,
        usage_num_cached: int | None,
        enable_details: bool,
        request=None,
) -> str:
    """Attach ``prompt_tokens_details`` to SSE chunks that carry ``usage``.

    The streaming generators yield already-serialised SSE text, so the pydantic
    object is out of reach by the time we see it -- hence the json round-trip.
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
            # Keep the key present with an explicit null: upstream dumps the final
            # usage chunk with exclude_none=True, which would drop it entirely.
            usage.setdefault("prompt_tokens_details", None)
        else:
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            denominator = (
                _prompt_tokens_denominator_pd(request, prompt_tokens)
                if request is not None
                else prompt_tokens
            )
            if denominator <= 0:
                denominator = prompt_tokens

            capped = (
                usage_num_cached if denominator <= 0
                else min(usage_num_cached, denominator)
            )
            cached_rate = _prompt_cache_hit_rate(capped, denominator)

            details = usage.get("prompt_tokens_details")
            if isinstance(details, dict):
                details["cached_tokens"] = usage_num_cached
                details["cached_rate"] = cached_rate
            else:
                usage["prompt_tokens_details"] = {
                    "cached_tokens": usage_num_cached,
                    "cached_rate": cached_rate,
                }
        return f"data: {json.dumps(obj)}\n\n"
    except (json.JSONDecodeError, KeyError):
        return chunk


# ---------------------------------------------------------------------------
# 4. chat_completion: streaming + non-streaming
# ---------------------------------------------------------------------------
#
# Relay-patching: PatchManager keys the registry by name and patch modules are
# imported in filename order (prefilled_token_skip < routed_experts < serving_apc),
# so re-registering the *same* name replaces the earlier class -- which is what keeps
# VLLMPatch.apply() from raising "already patched by".  The module-level references
# below still hold the replaced class's method, so the chain
# ``vLLM -> routed_experts -> serving_apc`` is stitched together by hand.
#
# Extra arguments are forwarded blind (``*args, **kwargs``) so 0.25.1's new
# ``chat_template_kwargs`` / ``mm_token_counts`` / ``parser`` keywords reach the
# wrapped implementation without being mirrored here.

def _chain_to(module_path: str, class_name: str, method_name: str, fallback):
    """Return the method this patch should wrap.

    Normally that is the previous link in the relay chain.  If that module is
    unimportable -- during a vLLM version bump it may be listed in
    OMNI_NPU_SKIP_PATCH_FILES, or not yet ported -- fall back to the vLLM
    implementation so the APC line still works on its own, and say so loudly:
    the feature that link provided is then NOT active.
    """
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name).__dict__[method_name]
    except Exception as exc:  # noqa: BLE001 - the fallback is always safe
        logger.warning(
            "patch_serving_apc: %s.%s.%s unavailable (%s); chaining "
            "%s directly to vLLM. The patch that link provided is NOT active.",
            module_path, class_name, method_name, exc, method_name,
        )
        return fallback


_COMMON = "omni_npu.vllm_patches.patches.common"

_orig_chat_stream = _chain_to(
    f"{_COMMON}.patch_routed_experts",
    "ExpertIdServingChatStream",
    "chat_completion_stream_generator",
    OpenAIServingChat.chat_completion_stream_generator,
)
_orig_chat_full = _chain_to(
    f"{_COMMON}.patch_prefilled_token_skip_tokenize",
    "OpenAIServingChatPatch",
    "chat_completion_full_generator",
    OpenAIServingChat.chat_completion_full_generator,
)


@register_patch("ExpertIdServingChatStream", OpenAIServingChat)
class OpenAIServingChatStreamAPCPatch(VLLMPatch):
    _attr_names_to_apply = ["chat_completion_stream_generator"]

    async def chat_completion_stream_generator(
            self,
            request,
            result_generator,
            *args: Any,
            **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        latest_engine_cached: int | None = None

        async def track_engine_cached():
            nonlocal latest_engine_cached
            async for res in result_generator:
                latest_engine_cached = getattr(res, "num_cached_tokens", None)
                yield res

        async for chunk in _orig_chat_stream(
                self, request, track_engine_cached(), *args, **kwargs):
            yield _normalize_usage_chunk(
                chunk,
                _resolve_num_cached_tokens_for_usage(request, latest_engine_cached),
                self.enable_prompt_tokens_details,
                request,
            )


@register_patch("PrefilledTokenSkipOpenAIServingChat", OpenAIServingChat)
class OpenAIServingChatFullAPCPatch(VLLMPatch):
    _attr_names_to_apply = ["chat_completion_full_generator"]

    async def chat_completion_full_generator(
            self,
            request,
            result_generator,
            *args: Any,
            **kwargs: Any,
    ):
        last_num_cached: int | None = None

        async def capture_num_cached():
            nonlocal last_num_cached
            async for res in result_generator:
                last_num_cached = getattr(res, "num_cached_tokens", None)
                yield res

        result = await _orig_chat_full(
            self, request, capture_num_cached(), *args, **kwargs)
        if isinstance(result, ErrorResponse):
            return result

        _apply_apc_to_response(self, request, result, last_num_cached)
        return result


# ---------------------------------------------------------------------------
# 5. completion: streaming + non-streaming
# ---------------------------------------------------------------------------

_orig_compl_stream = _chain_to(
    f"{_COMMON}.patch_routed_experts",
    "ExpertIdServingCompletionStream",
    "completion_stream_generator",
    OpenAIServingCompletion.completion_stream_generator,
)
_orig_compl_create = OpenAIServingCompletion.create_completion


@register_patch("ExpertIdServingCompletionStream", OpenAIServingCompletion)
class OpenAIServingCompletionStreamAPCPatch(VLLMPatch):
    _attr_names_to_apply = ["completion_stream_generator"]

    async def completion_stream_generator(
            self,
            request,
            engine_inputs,
            result_generator,
            *args: Any,
            **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        latest_engine_cached: int | None = None

        async def track_engine_cached():
            nonlocal latest_engine_cached
            # completion yields (prompt_idx, RequestOutput) tuples.
            async for item in result_generator:
                res = item[1] if isinstance(item, tuple) and len(item) >= 2 else item
                latest_engine_cached = getattr(res, "num_cached_tokens", None)
                yield item

        async for chunk in _orig_compl_stream(
                self, request, engine_inputs, track_engine_cached(), *args, **kwargs):
            yield _normalize_usage_chunk(
                chunk,
                _resolve_num_cached_tokens_for_usage(request, latest_engine_cached),
                self.enable_prompt_tokens_details,
                request,
            )


@register_patch("OpenAIServingCompletionAPCPatch", OpenAIServingCompletion)
class OpenAIServingCompletionAPCPatch(VLLMPatch):
    _attr_names_to_apply = ["create_completion"]

    async def create_completion(self, request, raw_request=None):
        # 0.25.1's create_completion is a thin wrapper that adds
        # _with_kv_transfer_rejection_cleanup around _create_completion; call the
        # original so that cleanup is not lost.
        result = await _orig_compl_create(self, request, raw_request)
        if inspect.isasyncgen(result) or isinstance(result, ErrorResponse):
            return result

        _apply_apc_to_response(self, request, result)
        return result
