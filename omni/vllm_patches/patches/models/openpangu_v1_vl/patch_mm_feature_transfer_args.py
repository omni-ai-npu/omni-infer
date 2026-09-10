# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in CLI integration for multimodal feature transfer."""

import argparse

from vllm import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


@register_patch("MMFeatureTransferArgsPatch", EngineArgs)
class MMFeatureTransferArgsPatch(VLLMPatch):
    _attr_names_to_apply = ["add_cli_args", "from_cli_args"]

    @classmethod
    def apply(cls):
        # Common EngineArgs patches are applied first; retain their wrappers.
        cls.apply_bypass_conflict("add_cli_args", "from_cli_args")

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        parser = MMFeatureTransferArgsPatch._upstream_add_cli_args(parser)
        parser.add_argument(
            "--mm-feature-transfer-config",
            type=str,
            default=None,
            help="JSON configuration for multimodal feature transfer connectors.",
        )
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        instance = MMFeatureTransferArgsPatch._upstream_from_cli_args(cls, args)
        config = getattr(args, "mm_feature_transfer_config", None)
        if config:
            from omni_npu.connector.mm_feature_transfer import register

            register(config)
        return instance
