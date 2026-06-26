# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
from typing import Callable

from vllm import EngineArgs
from vllm.config import VllmConfig
from vllm.utils.argparse_utils import FlexibleArgumentParser

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

ROUTED_EXPERTS_SERIALIZATION_MODES = ("zip_base64", "base64")
DEFAULT_ROUTED_EXPERTS_SERIALIZATION_MODE = "zip_base64"

_original_add_cli_args = EngineArgs.add_cli_args
_original_from_cli_args = EngineArgs.from_cli_args.__func__
_original_create_engine_config = EngineArgs.create_engine_config


def _apply_engine_config_extensions(
    engine_args: EngineArgs,
    vllm_config: VllmConfig,
) -> VllmConfig:
    vllm_config.routed_experts_serialization_mode = getattr(
        engine_args,
        "routed_experts_serialization_mode",
        DEFAULT_ROUTED_EXPERTS_SERIALIZATION_MODE,
    )
    return vllm_config


@register_patch("VllmConfigPatch", VllmConfig)
class VllmConfigPatch(VLLMPatch):
    """Patch to VllmConfig with extra engine-bridge fields."""

    _attr_names_to_apply = [
        "routed_experts_serialization_mode",
    ]

    routed_experts_serialization_mode: str = (
        DEFAULT_ROUTED_EXPERTS_SERIALIZATION_MODE
    )


@register_patch("EngineArgsPatch", EngineArgs)
class EngineArgsPatch(VLLMPatch):
    """Patch to EngineArgs for bridgeable CLI/config extensions."""

    _attr_names_to_apply = [
        "routed_experts_serialization_mode",
        "add_cli_args",
        "from_cli_args",
        "_omni_wrap_create_engine_config",
        "create_engine_config",
    ]

    routed_experts_serialization_mode: str = (
        DEFAULT_ROUTED_EXPERTS_SERIALIZATION_MODE
    )

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        """Shared CLI arguments for vLLM engine."""
        parser = _original_add_cli_args(parser)

        vllm_group = parser.add_argument_group(
            title="VllmConfig",
            description=VllmConfig.__doc__,
        )
        vllm_group.add_argument(
            "--routed-experts-serialization-mode",
            choices=ROUTED_EXPERTS_SERIALIZATION_MODES,
            default=DEFAULT_ROUTED_EXPERTS_SERIALIZATION_MODE,
            help="Serialization mode for routed experts payloads.",
        )
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        instance = _original_from_cli_args(cls, args)

        instance.routed_experts_serialization_mode = getattr(
            args,
            "routed_experts_serialization_mode",
            DEFAULT_ROUTED_EXPERTS_SERIALIZATION_MODE,
        )
        return instance

    def _omni_wrap_create_engine_config(
        self,
        builder: Callable[[], VllmConfig],
    ) -> VllmConfig:
        from vllm.platforms import current_platform

        if not getattr(self, "enable_eplb", False):
            return builder()

        original_is_cuda_alike = current_platform.is_cuda_alike

        def _npu_temp_cuda_alike_true() -> bool:
            if getattr(current_platform, "device_type", None) == "npu":
                return True
            return original_is_cuda_alike()

        current_platform.is_cuda_alike = _npu_temp_cuda_alike_true
        try:
            return builder()
        finally:
            current_platform.is_cuda_alike = original_is_cuda_alike

    def create_engine_config(
        self,
        usage_context=None,
        headless: bool = False,
    ) -> VllmConfig:
        def _builder() -> VllmConfig:
            return _original_create_engine_config(self, usage_context, headless)

        vllm_config = self._omni_wrap_create_engine_config(_builder)
        return _apply_engine_config_extensions(self, vllm_config)
