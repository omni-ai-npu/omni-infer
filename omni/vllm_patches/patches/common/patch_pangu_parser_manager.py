# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select the unified Pangu ParserEngine when both parser flags are Pangu."""

from vllm.parser.parser_manager import ParserManager

from omni_npu.v1.parsers.pangu_parser_engine import PanguParserEngine
from omni_npu.vllm_patches.core import VLLMPatch, register_patch


_original_get_parser = ParserManager.get_parser.__func__


@register_patch("PanguParserManagerPatch", ParserManager)
class PanguParserManagerPatch(VLLMPatch):
    _attr_names_to_apply = ["get_parser"]

    @classmethod
    def get_parser(
        cls,
        tool_parser_name: str | None = None,
        reasoning_parser_name: str | None = None,
        enable_auto_tools: bool = False,
        model_name: str | None = None,
        is_harmony: bool = False,
    ):
        both_pangu = (
            tool_parser_name == "pangu" and reasoning_parser_name == "pangu"
        )
        if not is_harmony and enable_auto_tools and both_pangu:
            return PanguParserEngine
        return _original_get_parser(
            cls,
            tool_parser_name=tool_parser_name,
            reasoning_parser_name=reasoning_parser_name,
            enable_auto_tools=enable_auto_tools,
            model_name=model_name,
            is_harmony=is_harmony,
        )
