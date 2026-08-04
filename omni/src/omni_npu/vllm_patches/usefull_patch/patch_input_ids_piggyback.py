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
#   * max_model_len validation runs afterwards in _create_chat_completion via
#     get_max_tokens.
#   * Tool-call request adjustment still runs; tool-output parsing is unaffected.
#
# Gating: OMNI_PIGGYBACK_INPUT_IDS (default "0"); set "1" to enable. The patch is a
# no-op for any request that does not carry input_ids, so it is self-gating.
#
# Note on composition: patch_prefilled_token_skip_tokenize.py also wraps
# `_preprocess_chat` (PD prefilled-token reuse, gated by OMNI_REUSE_PREFILLED_TOKENS).
# That mechanism is orthogonal (kv_transfer_params), but both target the same method;
# if both must be active, compose them via a relay patch (see patch_serving_apc.py).

import difflib
from typing import Any, List, Optional

from vllm.entrypoints.chat_utils import (
    ChatTemplateContentFormatOption,
    ConversationMessage,
)

# These helpers build the conversation + multimodal data on the fast path. Import
# defensively: the loader (import_patches_from_dir) has no error handling, so a
# missing symbol here would abort the whole common-patch sweep. If absent, the fast
# path is simply disabled and every request falls back to normal tokenization.
try:
    from vllm.entrypoints.chat_utils import parse_chat_messages
except ImportError:  # noqa: BLE001
    parse_chat_messages = None

try:
    from vllm.renderers.hf import resolve_chat_template_content_format
except ImportError:  # noqa: BLE001
    resolve_chat_template_content_format = None

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.inputs import EngineInput, tokens_input
from vllm.logger import init_logger
from vllm.parser import Parser
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.utils.mistral import is_mistral_tokenizer, is_mistral_tool_parser

from omni_npu import envs
from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logger = init_logger(__name__)

# Capture the original method before any patch is applied.  In v0.25.1 the
# relay chain (InputIdsPiggyback → PrefilledTokenSkip → upstream) lives on
# OnlineRenderer.preprocess_chat() rather than OpenAIServingChat._preprocess_chat().
_original_preprocess_chat = OnlineRenderer.preprocess_chat


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
    int(envs.OMNI_PIGGYBACK_INPUT_IDS),
    int(envs.OMNI_VALIDATE_PIGGYBACK_INPUT_IDS),
    _FIELD_REGISTERED,
    parse_chat_messages is not None,
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


@register_patch("InputIdsPiggyback", OnlineRenderer)
class InputIdsPiggybackPatch(VLLMPatch):
    """Skip chat-template expansion + tokenization when the caller
    piggybacks pre-tokenized input_ids on a text-only
    ChatCompletionRequest.
    """

    _attr_names_to_apply = ["preprocess_chat"]

    async def preprocess_chat(
        self,
        request: Any,
        messages: list[Any],
        default_template: str | None,
        default_template_content_format: ChatTemplateContentFormatOption,
        default_template_kwargs: dict[str, Any] | None,
        tool_dicts: list[dict[str, Any]] | None = None,
        parser: type[Parser] | None = None,
        *,
        skip_mm_cache: bool = False,
    ) -> tuple[list[ConversationMessage], list[EngineInput]]:
        enabled = envs.OMNI_PIGGYBACK_INPUT_IDS
        if enabled:
            assert not envs.OMNI_SKIP_DECODE_TOKENIZE, (
                "OMNI_PIGGYBACK_INPUT_IDS=1 requires OMNI_SKIP_DECODE_TOKENIZE=0, "
                "but OMNI_SKIP_DECODE_TOKENIZE is currently set to "
                f"'{int(envs.OMNI_SKIP_DECODE_TOKENIZE)}'"
            )
        validate_enabled = envs.OMNI_VALIDATE_PIGGYBACK_INPUT_IDS

        caller_ids = _caller_input_ids(request) if enabled else None

        is_fast_path_candidate = (
            caller_ids is not None
            and parse_chat_messages is not None
            and resolve_chat_template_content_format is not None
            and isinstance(request, ChatCompletionRequest)
            and getattr(request, "truncate_prompt_tokens", None) is None
            and not _has_multimodal(messages)
        )

        if is_fast_path_candidate:
            tokenizer = self.renderer.tokenizer

            # Materialise any ValidatorIterator fields (e.g. tool_calls) so
            # they survive multiple iterations over the same messages list.
            if validate_enabled:
                for msg in messages:
                    tc = msg.get("tool_calls") if isinstance(msg, dict) else None
                    if tc is not None and not isinstance(tc, list):
                        msg["tool_calls"] = list(tc)

            resolved = resolve_chat_template_content_format(
                default_template,
                tool_dicts,
                default_template_content_format,
                tokenizer,
                model_config=self.model_config,
            )
            
            conversation, mm_data, _mm_uuids = parse_chat_messages(
                messages, self.model_config, content_format=resolved
            )

            if mm_data is None:

                if validate_enabled:
                    # Run the full original path to get vLLM's token ids for comparison.
                    _orig_conversation, orig_engine_inputs = await _original_preprocess_chat(
                        self,
                        request,
                        messages,
                        default_template,
                        default_template_content_format,
                        default_template_kwargs,
                        tool_dicts=tool_dicts,
                        parser=parser,
                        skip_mm_cache=skip_mm_cache,
                    )
                    vllm_ids = orig_engine_inputs[0].get("prompt_token_ids", [])

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

                # Tool parsing — same logic as the original OnlineRenderer.preprocess_chat.
                if parser is not None:
                    tool_parser = parser.tool_parser_cls
                    tool_choice = getattr(request, "tool_choice", "none")
                    is_mistral_grammar_eligible = (
                        tool_parser is not None
                        and is_mistral_tool_parser(tool_parser)
                        and is_mistral_tokenizer(tokenizer)
                        and getattr(tokenizer, "supports_grammar", False)
                    )
                    should_adjust_request = (
                        parser.reasoning_parser_cls is not None
                        or tool_choice != "none"
                        or is_mistral_grammar_eligible
                    )
                    if should_adjust_request:
                        if not isinstance(request, ChatCompletionRequest | ResponsesRequest):
                            msg = (
                                "Tool usage is only supported "
                                "for Chat Completions API or Responses API requests, "
                                f"but got {type(request).__name__}"
                            )
                            raise NotImplementedError(msg)
                        chat_template_kwargs = request.build_chat_params(
                            default_template, default_template_content_format
                        ).with_defaults(default_template_kwargs or {}).chat_template_kwargs
                        request = parser(
                            tokenizer,
                            request.tools,
                            model_config=self.model_config,
                            chat_template_kwargs=chat_template_kwargs,
                        ).adjust_request(request=request)

                # Build engine_input directly from piggybacked ids.
                # tokens_input() produces TokensInput(type="token", prompt_token_ids=...)
                # which is accepted by _create_chat_completion downstream.
                # max_model_len validation runs later in get_max_tokens().
                cache_salt = getattr(request, "cache_salt", None)
                engine_input = tokens_input(caller_ids, cache_salt=cache_salt)

                return conversation, [engine_input]

        # Fallback: delegate to the original OnlineRenderer.preprocess_chat
        # (which itself may be wrapped by other patches in the relay chain).
        return await _original_preprocess_chat(
            self,
            request,
            messages,
            default_template,
            default_template_content_format,
            default_template_kwargs,
            tool_dicts=tool_dicts,
            parser=parser,
            skip_mm_cache=skip_mm_cache,
        )
