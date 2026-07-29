# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Reuse caller-supplied pre-tokenized `input_ids` on /v1/chat/completions.
#
# Omni Proxy already applies the chat template and tokenizes each request, then
# piggybacks the result as `"input_ids": [...]` on the chat request body. This patch
# makes the engine REUSE those ids and skip its own chat-template expansion +
# tokenization, which is id-identical to a normal encode (add_special_tokens=False)
# for text-only requests -- so it is safe for APC prefix matching.
#
# Scope / safety:
#   * Text-only ChatCompletionRequest only. Multimodal requests fall back to normal
#     tokenization (the token-only path is NOT equivalent to the text path for
#     several mm models, which mutate ids in _apply_hf_processor_tokens_only).
#   * Requests with truncate_prompt_tokens fall back (truncation-side handling).
#   * max_model_len validation still runs (via self._validate_input).
#   * Tool-call request adjustment still runs; tool-output parsing is unaffected.
#
# Gating: OMNI_PIGGYBACK_INPUT_IDS (default "0"); set "1" to enable. The patch is a
# no-op for any request that does not carry input_ids, so it is self-gating.
#
# Note on composition: patch_prefilled_token_skip_tokenize.py also wraps
# `_preprocess_chat` (PD prefilled-token reuse, gated by OMNI_REUSE_PREFILLED_TOKENS).
# That mechanism is orthogonal (kv_transfer_params), but both target the same method;
# if both must be active, compose them via a relay patch (see patch_serving_apc.py).

import os
import difflib
from typing import Any, Callable, List, Optional

from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateContentFormatOption,
)

# These helpers build the conversation + multimodal data on the fast path. Import
# defensively: the loader (import_patches_from_dir) has no error handling, so a
# missing symbol here would abort the whole common-patch sweep. If absent, the fast
# path is simply disabled and every request falls back to normal tokenization.
try:
    from vllm.entrypoints.chat_utils import (
        parse_chat_messages_futures,
        resolve_chat_template_content_format,
    )
except Exception:  # noqa: BLE001
    parse_chat_messages_futures = None
    resolve_chat_template_content_format = None

from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ResponsesRequest
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import ChatLikeRequest
from vllm.inputs.data import TokensPrompt
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser
from vllm.logger import init_logger

from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.vllm_patches.patches.common.patch_prefilled_token_skip_tokenize import (
    OpenAIServingChatPreprocessPatch as _PrefilledPreprocessPatch,
)

logger = init_logger(__name__)

# Captured at import (before any patch is applied) = the original / inherited method.
_prefilled_preprocess_chat = _PrefilledPreprocessPatch._preprocess_chat


def _register_input_ids_field() -> bool:
    """Declare `input_ids` on ChatCompletionRequest so it validates as a real field
    (not logged per-request as an ignored extra). Non-fatal: with extra="allow" the
    field is still readable via model_extra even if this fails. Returns True if the
    field is present afterwards."""
    try:
        from pydantic.fields import FieldInfo

        if "input_ids" in ChatCompletionRequest.model_fields:
            return True
        ChatCompletionRequest.model_fields["input_ids"] = FieldInfo(
            annotation=Optional[List[int]], default=None
        )
        ChatCompletionRequest.model_rebuild(force=True)
        # invalidate OpenAIBaseModel's cached allow-set so the "ignored extra" warning
        # recomputes and accepts input_ids
        ChatCompletionRequest.field_names = None
        return True
    except Exception:  # noqa: BLE001 - optional; reading falls back to model_extra
        return False


_FIELD_REGISTERED = _register_input_ids_field()

# Trace: confirms the plugin is loaded.
logger.info(
    "<<< InputIdsPiggyback: patch module loaded "
    "(enable=%s, validate=%s, input_ids_field_registered=%s, fast_path_available=%s)",
    os.getenv("OMNI_PIGGYBACK_INPUT_IDS", "0"),
    os.getenv("OMNI_VALIDATE_PIGGYBACK_INPUT_IDS", "0"),
    _FIELD_REGISTERED,
    parse_chat_messages_futures is not None,
)


def _caller_input_ids(request) -> Optional[List[int]]:
    """Read input_ids whether it is a declared field or an extra (extra='allow')."""
    ids = getattr(request, "input_ids", None)
    if ids is None:
        extra = getattr(request, "model_extra", None)
        if extra:
            ids = extra.get("input_ids")
    return ids


def _has_multimodal(messages) -> bool:
    """Cheap structural check for image/audio/video content (no fetching)."""
    for msg in messages or ():
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type", "text") != "text":
                    return True
    return False


@register_patch("InputIdsPiggyback", OpenAIServingChat)
class InputIdsPiggybackPatch(VLLMPatch):
    """Skip chat-template expansion + tokenization when the caller
    piggybacks pre-tokenized input_ids on a text-only
    ChatCompletionRequest.
    """

    _attr_names_to_apply = ["_preprocess_chat"]

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
        enabled = os.getenv("OMNI_PIGGYBACK_INPUT_IDS", "0") == "1"
        if enabled:
            assert os.getenv("OMNI_SKIP_DECODE_TOKENIZE", "0") == "0", (
                "OMNI_PIGGYBACK_INPUT_IDS=1 requires OMNI_SKIP_DECODE_TOKENIZE=0, "
                "but OMNI_SKIP_DECODE_TOKENIZE is currently set to "
                f"'{os.getenv('OMNI_SKIP_DECODE_TOKENIZE')}'"
            )
        validate_enabled = os.getenv("OMNI_VALIDATE_PIGGYBACK_INPUT_IDS", "0") == "1"

        caller_ids = _caller_input_ids(request) if enabled else None

        is_fast_path_candidate = (
            caller_ids
            and tokenizer is not None
            and parse_chat_messages_futures is not None
            and isinstance(request, ChatCompletionRequest)
            and getattr(request, "truncate_prompt_tokens", None) is None
            and not _has_multimodal(messages)
        )

        if is_fast_path_candidate:
            # Materialise any ValidatorIterator fields (e.g. tool_calls) so
            # they survive multiple iterations over the same messages list.
            if validate_enabled:
                for msg in messages:
                    tc = msg.get("tool_calls") if isinstance(msg, dict) else None
                    if tc is not None and not isinstance(tc, list):
                        msg["tool_calls"] = list(tc)

            resolved = resolve_chat_template_content_format(
                chat_template,
                tool_dicts,
                chat_template_content_format,
                tokenizer,
                model_config=self.model_config,
            )
            conversation, mm_future, _mm_uuids = parse_chat_messages_futures(
                messages, self.model_config, content_format=resolved
            )
            mm_data = await mm_future

            if mm_data is None:

                if validate_enabled:
                    _orig_conversation, orig_prompts = await _prefilled_preprocess_chat(
                        self, request, tokenizer, messages, chat_template,
                        chat_template_content_format, add_generation_prompt,
                        continue_final_message, tool_dicts, documents,
                        chat_template_kwargs, default_chat_template_kwargs,
                        tool_parser, add_special_tokens
                    )
                    vllm_ids = orig_prompts[0].get("prompt_token_ids", [])

                    if caller_ids != vllm_ids:
                        caller_tokens = (
                            [tokenizer.decode([tid]) for tid in caller_ids]
                            if tokenizer
                            else [str(tid) for tid in caller_ids]
                        )
                        vllm_tokens = (
                            [tokenizer.decode([tid]) for tid in vllm_ids]
                            if tokenizer
                            else [str(tid) for tid in vllm_ids]
                        )

                        diff = difflib.ndiff(vllm_tokens, caller_tokens)
                        diff_text = "\n".join(diff)

                        error_msg = (
                            f"=== InputIds Mismatch Detected ===\n"
                            f"Client provided input_ids length: "
                            f"{len(caller_ids)}, "
                            f"vLLM generated length: "
                            f"{len(vllm_ids)}\n"
                            f"Client IDs: "
                            f"{caller_ids[:20]}... (truncated)\n"
                            f"vLLM   IDs: "
                            f"{vllm_ids[:20]}... (truncated)\n"
                            f"Token string diff "
                            f"(- vLLM, + Client):\n"
                            f"{diff_text}\n"
                            f"================================="
                        )
                        logger.error(
                            "<<< InputIdsPiggyback: %s",
                            error_msg,
                        )
                        raise ValueError(
                            "Input IDs verification failed. "
                            "Tokenizer output mismatch. "
                            "See logs for detailed diff."
                        )

                if tool_parser is not None and (
                    getattr(request, "tool_choice", None) != "none"
                ):
                    request = tool_parser(tokenizer).adjust_request(request=request)

                prompt_inputs = self._validate_input(request, caller_ids, "")
                engine_prompt = TokensPrompt(
                    prompt_token_ids=prompt_inputs["prompt_token_ids"]
                )
                if getattr(request, "mm_processor_kwargs", None) is not None:
                    engine_prompt["mm_processor_kwargs"] = request.mm_processor_kwargs
                if getattr(request, "cache_salt", None) is not None:
                    engine_prompt["cache_salt"] = request.cache_salt

                logger.debug(
                    "<<< InputIdsPiggyback: ACTIVE (Validation=%s) — reusing piggybacked input_ids",
                    validate_enabled,
                )
                return conversation, [engine_prompt]

        if enabled and caller_ids:
            logger.debug(
                "<<< InputIdsPiggyback: input_ids present (%d) but NOT reused -> normal tokenization",
                len(caller_ids),
            )

        # Relay chain fallback: delegate to prefilled_token_skip_tokenize's
        # preprocess (which calls the true original then applies prefilled reuse).
        return await _prefilled_preprocess_chat(
            self, request, tokenizer, messages, chat_template,
            chat_template_content_format, add_generation_prompt,
            continue_final_message, tool_dicts, documents,
            chat_template_kwargs, default_chat_template_kwargs,
            tool_parser, add_special_tokens,
        )