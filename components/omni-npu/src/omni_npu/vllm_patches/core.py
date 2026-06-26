# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# vllm_patches Reference: https://blog.vllm.ai/2025/11/20/vllm-plugin-system.html


import logging
from packaging import version
from types import MethodType, ModuleType
from typing import Type, Union
import vllm
from omni_npu.vllm_patches import PatchManager

logger = logging.getLogger(__name__)

class VLLMPatch:
    """
    Usage:
        @register_patch(patch_name, TargetClass)
        class MyPatch(VLLMPatch):
            _attr_names_to_apply = [my_patch_method, ...]
            def my_patch_method(self):
                ...

        MyPatch.apply()
    """

    _attr_names_to_apply = []

    @classmethod
    def apply(cls):
        if cls is VLLMPatch:
            raise TypeError("base VLLMPatch class should not be applied directly")

        target = cls._target

        if not hasattr(target, '_omni_npu_applied_patches'):
            target._omni_npu_applied_patches = {}

        for name in cls._attr_names_to_apply:
            if name in ('apply', '_target', '_attr_names_to_apply'):
                logger.warning(f'{name} should not be applied as patch in PatchClass {cls.__name__}, bypassing')
                continue

            if name not in cls.__dict__:
                raise ValueError(f"cannot find {name} in PatchClass {cls.__name__}")

            attr = cls.__dict__[name]

            if name in target._omni_npu_applied_patches:
                existed_patch_class = target._omni_npu_applied_patches[name]
                raise ValueError(
                    f"{target.__name__}.{name} already patched by {existed_patch_class}"
                )

            target._omni_npu_applied_patches[name] = cls.__name__

            if isinstance(attr, MethodType):
                attr = MethodType(attr.__func__, target)

            setattr(target, name, attr)

            logger.info(f"patch applied: {cls.__name__} => {target.__name__}.{name}")

def register_patch(name: str, target: Union[Type, ModuleType]):
    if not isinstance(target, (type, ModuleType)):
        raise TypeError(f"can only patch classes or modules, but {target} type is {type(target)}")

    def decorator(cls):
        cls._target = target
        PatchManager.register(name, cls)
        return cls

    return decorator
