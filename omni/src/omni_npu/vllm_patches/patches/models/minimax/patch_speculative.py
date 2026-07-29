# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Any, Callable

from pydantic.dataclasses import rebuild_dataclass

from vllm.config import SpeculativeConfig, VllmConfig

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


_ORIGINAL_VERIFY_ARGS = getattr(SpeculativeConfig, "_verify_args", None)


def _is_minimax_m2_eagle3_allowlist_error(
    speculative_config: SpeculativeConfig,
    exc: ValueError,
) -> bool:
    if speculative_config.method != "eagle3":
        return False

    target_model_config = getattr(speculative_config, "target_model_config", None)
    if target_model_config is None:
        return False

    hf_text_config = getattr(target_model_config, "hf_text_config", None)
    model_type = getattr(hf_text_config, "model_type", "") or ""
    if model_type != "minimax_m2":
        return False

    return "Eagle3 is only supported for" in str(exc)


def _verify_args_with_minimax_m2_allowlist(
    speculative_config: SpeculativeConfig,
    original_verify_args: Callable[[], Any],
) -> Any:
    try:
        return original_verify_args()
    except ValueError as exc:
        if not _is_minimax_m2_eagle3_allowlist_error(speculative_config, exc):
            raise
        verify_equal_vocab_size = getattr(
            speculative_config, "verify_equal_vocab_size_if_draft_model", None
        )
        if callable(verify_equal_vocab_size):
            verify_equal_vocab_size()
        return speculative_config


def _verify_args(self: SpeculativeConfig) -> Any:
    if _ORIGINAL_VERIFY_ARGS is None:
        return self
    return _verify_args_with_minimax_m2_allowlist(
        self,
        lambda: _ORIGINAL_VERIFY_ARGS(self),
    )


def _refresh_speculative_config_validator() -> None:
    decorators = getattr(SpeculativeConfig, "__pydantic_decorators__", None)
    model_validators = getattr(decorators, "model_validators", {})
    validator = model_validators.get("_verify_args")
    if validator is not None:
        object.__setattr__(validator, "func", _verify_args)

    SpeculativeConfig._verify_args = _verify_args
    rebuild_dataclass(SpeculativeConfig, force=True)
    rebuild_dataclass(VllmConfig, force=True)


_refresh_speculative_config_validator()


@register_patch("MiniMaxM2SpeculativeConfigValidatorPatch", SpeculativeConfig)
class MiniMaxM2SpeculativeConfigValidatorPatch(VLLMPatch):
    _attr_names_to_apply: list[str] = []
