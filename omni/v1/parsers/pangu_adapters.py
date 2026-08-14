# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry adapters for the unified Pangu ParserEngine."""

from vllm.parser.engine.adapters import make_adapters

from omni_npu.v1.parsers.pangu_parser_engine import PanguParserEngine


(
    PanguParserEngineReasoningAdapter,
    PanguParserEngineToolAdapter,
) = make_adapters(PanguParserEngine)

# ``make_adapters`` creates the classes in vLLM's adapter module. Point their
# metadata back here so ParserManager's lazy import can resolve class names.
PanguParserEngineReasoningAdapter.__module__ = __name__
PanguParserEngineToolAdapter.__module__ = __name__
