# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

from vllm import EngineArgs
from vllm.config import ModelConfig
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.logger import init_logger
import vllm.tokenizers as _vllm_tokenizers_module

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logger = init_logger(__name__)
# ────────────────────────────────────────────────────────────
# save originals before patching (Module-level fallbacks)
# NOTE: These are captured at import time, before any patches are
# applied.  For methods that must chain through *previously applied*
# patches (e.g. add_cli_args), the per-patch apply() override
# captures the upstream version at apply-time.
# ────────────────────────────────────────────────────────────
_original_ea_create_model_config = EngineArgs.create_model_config


# ────────────────────────────────────────────────────────────
# Patch 1: ModelConfig — add 3 LoPT fields
# ────────────────────────────────────────────────────────────
@register_patch("ModelConfigLoptPatch", ModelConfig)
class ModelConfigLoptPatch(VLLMPatch):
    _attr_names_to_apply = [
        "enable_lopt",
        "lopt_pool_size",
        "lopt_chunk_size",
    ]

    enable_lopt: bool = False
    lopt_pool_size: int = 16
    lopt_chunk_size: int = 4096


# ────────────────────────────────────────────────────────────
# Patch 2: EngineArgs — CLI args + wire to ModelConfig
# ────────────────────────────────────────────────────────────
@register_patch("EngineArgsLoptPatch", EngineArgs)
class EngineArgsLoptPatch(VLLMPatch):
    _attr_names_to_apply = [
        "enable_lopt",
        "lopt_pool_size",
        "lopt_chunk_size",
        "add_cli_args",
        "from_cli_args",
        "create_model_config",
    ]

    enable_lopt: bool = False
    lopt_pool_size: int = 16
    lopt_chunk_size: int = 4096

    @classmethod
    def apply(cls):
        """Override to avoid _omni_npu_applied_patches conflict with
        EngineArgsPatch (which already patches add_cli_args / from_cli_args).

        Capture the upstream (already-patched) versions at apply-time so the
        chain stays intact.
        """
        target = cls._target

        # Save the currently-active (possibly already patched) versions
        cls._upstream_add_cli_args = target.add_cli_args
        cls._upstream_from_cli_args = target.from_cli_args.__func__

        for name in cls._attr_names_to_apply:
            if name in cls.__dict__:
                setattr(target, name, cls.__dict__[name])

        logger.info("patch applied: %s => %s (bypass-conflict)", cls.__name__, target.__name__)

    @staticmethod
    def add_cli_args(parser):
        parser = EngineArgsLoptPatch._upstream_add_cli_args(parser)

        model_group = parser.add_argument_group(
            title="ModelConfig",
            description=ModelConfig.__doc__,
        )
        model_group.add_argument(
            "--enable-lopt",
            action="store_true",
            default=False,
            help="Enable Lossless Parallel Tokenizer (LoPT) for long-text tokenization.",
        )
        model_group.add_argument(
            "--lopt-pool-size",
            type=int,
            default=16,
            help="Number of parallel processes for LoPT tokenization.",
        )
        model_group.add_argument(
            "--lopt-chunk-size",
            type=int,
            default=4096,
            help="Chunk size in characters for LoPT text splitting.",
        )
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        instance = EngineArgsLoptPatch._upstream_from_cli_args(cls, args)
        instance.enable_lopt = getattr(args, "enable_lopt", False)
        instance.lopt_pool_size = getattr(args, "lopt_pool_size", 16)
        instance.lopt_chunk_size = getattr(args, "lopt_chunk_size", 4096)
        return instance

    def create_model_config(self):
        model_config = _original_ea_create_model_config(self)
        model_config.enable_lopt = self.enable_lopt
        model_config.lopt_pool_size = self.lopt_pool_size
        model_config.lopt_chunk_size = self.lopt_chunk_size
        return model_config


# ────────────────────────────────────────────────────────────
# Patch 3: OpenAIServing — init LoPT tokenizer + use in normalize
# ────────────────────────────────────────────────────────────
@register_patch("OpenAIServingLoptPatch", OpenAIServing)
class OpenAIServingLoptPatch(VLLMPatch):
    _attr_names_to_apply = [
        "__init__",
        "_normalize_prompt_text_to_input",
    ]

    @classmethod
    def apply(cls):
        """Override to avoid _omni_npu_applied_patches conflict with other
        patches that may also override __init__ or _normalize_prompt_text_to_input.
        """
        target = cls._target

        cls._upstream_os_init = target.__init__
        cls._upstream_os_normalize = target._normalize_prompt_text_to_input

        for name in cls._attr_names_to_apply:
            if name in cls.__dict__:
                setattr(target, name, cls.__dict__[name])

        logger.info("patch applied: %s => %s (bypass-conflict)", cls.__name__, target.__name__)

    def __init__(
        self,
        engine_client,
        models,
        *,
        request_logger=None,
        return_tokens_as_token_ids=False,
        log_error_stack=False,
    ):
        OpenAIServingLoptPatch._upstream_os_init(
            self,
            engine_client,
            models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
            log_error_stack=log_error_stack,
        )

        model_config = self.model_config
        self.enable_lopt = getattr(model_config, "enable_lopt", False)
        self.lopt_tokenizer = None

        if self.enable_lopt:
            logger.warning(
                "Lossless Parallel Tokenizer Enabled! "
                "pool size=%s, chunk length=%s.",
                getattr(model_config, "lopt_pool_size", 16),
                getattr(model_config, "lopt_chunk_size", 4096),
            )
            from omni_npu.lopt import maybe_get_lopt_tokenizer

            self.lopt_tokenizer = maybe_get_lopt_tokenizer(
                model_path=model_config.model,
                enable_lopt=True,
                lopt_pool_size=getattr(model_config, "lopt_pool_size", 16),
                lopt_chunk_size=getattr(model_config, "lopt_chunk_size", 4096),
            )
        else:
            logger.warning(
                "Lossless Parallel Tokenizer is Not Enabled! "
                "enable_lopt=%s",
                self.enable_lopt,
            )

    async def _normalize_prompt_text_to_input(
        self,
        request,
        prompt,
        tokenizer,
        add_special_tokens,
    ):
        if self.enable_lopt and self.lopt_tokenizer is not None:
            encoded = self.lopt_tokenizer(prompt, add_special_tokens)
            lopt_input_ids = encoded.input_ids

            truncate_prompt_tokens = getattr(request, "truncate_prompt_tokens", None)
            if truncate_prompt_tokens is not None:
                if truncate_prompt_tokens < 0:
                    lopt_input_ids = lopt_input_ids[: self.max_model_len]
                else:
                    lopt_input_ids = lopt_input_ids[
                        : min(truncate_prompt_tokens, self.max_model_len)
                    ]

            return self._validate_input(request, lopt_input_ids, prompt)

        return await OpenAIServingLoptPatch._upstream_os_normalize(
            self, request, prompt, tokenizer, add_special_tokens
        )


# ────────────────────────────────────────────────────────────
# Patch 4: vllm.tokenizers module — export maybe_get_lopt_tokenizer
# ────────────────────────────────────────────────────────────

@register_patch("TokenizerModuleLoptPatch", _vllm_tokenizers_module)
class TokenizerModuleLoptPatch(VLLMPatch):
    _attr_names_to_apply = ["maybe_get_lopt_tokenizer"]

    @staticmethod
    def maybe_get_lopt_tokenizer(*args, **kwargs):
        from omni_npu.lopt import maybe_get_lopt_tokenizer as _fn

        return _fn(*args, **kwargs)
