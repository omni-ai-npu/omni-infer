# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

from vllm import EngineArgs
from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.renderers.base import BaseRenderer
import vllm.tokenizers as _vllm_tokenizers_module

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logger = init_logger(__name__)
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

        logger.info(
            "patch applied: %s => %s (bypass-conflict)",
            cls.__name__,
            target.__name__,
        )

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
# Patch 3: BaseRenderer — use LoPT in the shared text tokenization path
# ────────────────────────────────────────────────────────────
@register_patch("BaseRendererLoptPatch", BaseRenderer)
class BaseRendererLoptPatch(VLLMPatch):
    _attr_names_to_apply = [
        "_tokenize_prompt",
    ]

    @classmethod
    def apply(cls):
        """Capture the current tokenizer hook so LoPT can safely fall back.

        Current vLLM routes OpenAI completion/chat prompt preprocessing through
        BaseRenderer. Patching the shared tokenizer hook covers both paths and
        avoids depending on the removed OpenAIServing class.
        """
        target = cls._target

        cls._upstream_tokenize_prompt = target._tokenize_prompt

        for name in cls._attr_names_to_apply:
            if name in cls.__dict__:
                setattr(target, name, cls.__dict__[name])

        logger.info(
            "patch applied: %s => %s (bypass-conflict)",
            cls.__name__,
            target.__name__,
        )

    @staticmethod
    def _get_lopt_tokenizer(self):
        model_config = self.model_config
        if not getattr(model_config, "enable_lopt", False):
            return None

        if hasattr(self, "_omni_lopt_tokenizer"):
            return self._omni_lopt_tokenizer

        from omni_npu.lopt import maybe_get_lopt_tokenizer

        tokenizer_path = (
            getattr(model_config, "tokenizer", None) or model_config.model
        )
        self._omni_lopt_tokenizer = maybe_get_lopt_tokenizer(
            model_path=tokenizer_path,
            enable_lopt=True,
            lopt_pool_size=getattr(model_config, "lopt_pool_size", 16),
            lopt_chunk_size=getattr(model_config, "lopt_chunk_size", 4096),
        )
        if self._omni_lopt_tokenizer is not None:
            logger.warning(
                "Lossless Parallel Tokenizer enabled. pool size=%s, "
                "chunk length=%s, tokenizer=%s.",
                getattr(model_config, "lopt_pool_size", 16),
                getattr(model_config, "lopt_chunk_size", 4096),
                tokenizer_path,
            )
        return self._omni_lopt_tokenizer

    def _tokenize_prompt(self, prompt, params):
        # return_token_offsets eventually maps to the native offsets path.
        want_offsets = self._wants_offsets(prompt, params)
        if want_offsets:
            return BaseRendererLoptPatch._upstream_tokenize_prompt(
                self, prompt, params
            )

        kwargs = params.get_encode_kwargs()
        if kwargs.get("return_offsets_mapping"):
            logger.debug_once(
                "Bypassing LoPT tokenizer because return_offsets_mapping is "
                "not supported.",
                scope="local",
            )
            return BaseRendererLoptPatch._upstream_tokenize_prompt(
                self, prompt, params
            )

        lopt_tokenizer = BaseRendererLoptPatch._get_lopt_tokenizer(self)
        if lopt_tokenizer is None:
            return BaseRendererLoptPatch._upstream_tokenize_prompt(
                self, prompt, params
            )

        try:
            encoding = lopt_tokenizer(
                prompt["prompt"],
                add_special_tokens=kwargs.get("add_special_tokens", False),
            )
        except Exception:
            logger.exception("LoPT tokenization failed; using native tokenizer.")
            return BaseRendererLoptPatch._upstream_tokenize_prompt(
                self, prompt, params
            )

        return self._build_tokens_prompt(encoding["input_ids"], prompt)


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
