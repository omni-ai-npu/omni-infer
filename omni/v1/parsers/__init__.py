# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from omni_npu.v1.parsers.pangu_adapters import (
    PanguParserEngineReasoningAdapter,
    PanguParserEngineToolAdapter,
)
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

PARSER_NAME = "pangu"


def register_lazy_parsers() -> None:
    ReasoningParserManager.register_lazy_module(
        PARSER_NAME,
        PanguParserEngineReasoningAdapter.__module__,
        PanguParserEngineReasoningAdapter.__name__,
    )
    ToolParserManager.register_lazy_module(
        PARSER_NAME,
        PanguParserEngineToolAdapter.__module__,
        PanguParserEngineToolAdapter.__name__,
    )



register_lazy_parsers()
