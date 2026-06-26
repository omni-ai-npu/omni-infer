# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.


import copy
import logging

import vllm.engine.arg_utils as arg_utils
from vllm.engine.arg_utils import _compute_kwargs
from vllm.tasks import SupportedTask
from vllm.v1.engine.llm_engine import LLMEngine

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logger = logging.getLogger(__name__)
@register_patch("LLMEngineHelloWorld", LLMEngine)
class LLMEngineHelloWorldPatch(VLLMPatch):
    """
    Makes LLMEngines print 'Hello World' when get supported tasks.
    """

    _attr_names_to_apply = ['print_hello_world', 'get_supported_tasks']

    @staticmethod
    def print_hello_world():
        print("Hello World")

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        self.print_hello_world()
        return self.engine_core.get_supported_tasks()

@register_patch("GetKwargsHelloWorld", arg_utils)
class GetKwargsHelloWorldPatch(VLLMPatch):
    _attr_names_to_apply = ["get_kwargs"]

    def get_kwargs(cls):
        logger.info(">>> Hello World: get_kwargs is called for %s", cls)
        return copy.deepcopy(_compute_kwargs(cls))