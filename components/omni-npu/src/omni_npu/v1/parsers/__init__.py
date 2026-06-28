# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright contributors to the vLLM project.

from omni_npu.v1.parsers.pangu_reasoning_parser import PanguReasoningParser
from omni_npu.v1.parsers.pangu_tool_parser import PanguToolParser
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager

"""
tool and reasoning parser
"""

parser_name = "pangu"

_TOOL_PARSERS_TO_REGISTER = {
    parser_name: PanguToolParser
}

_REASONING_PARSERS_TO_REGISTER = {
    parser_name: PanguReasoningParser
}


def register_lazy_parsers():
    for name, parser_cls in _REASONING_PARSERS_TO_REGISTER.items():
        module_path = parser_cls.__module__
        class_name = parser_cls.__name__
        ReasoningParserManager.register_lazy_module(name, module_path, class_name)
    for name, parser_cls in _TOOL_PARSERS_TO_REGISTER.items():
        module_path = parser_cls.__module__
        class_name = parser_cls.__name__
        ToolParserManager.register_lazy_module(name, module_path, class_name)


"""
tool and reasoning parsers register
"""
register_lazy_parsers()
