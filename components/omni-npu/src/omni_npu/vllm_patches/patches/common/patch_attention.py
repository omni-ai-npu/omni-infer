# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from vllm.v1.attention.backends.registry import AttentionBackendEnum

from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.attention.backends.utils import NPU_ATTENTION_BACKEND

_orig_getitem = AttentionBackendEnum.__class__.__getitem__


@register_patch("NPUAttentionBackendEnum", AttentionBackendEnum.__class__)
class NPUAttentionBackendEnum(VLLMPatch):
    """Patch for vLLM's AttentionBackendEnum to register omni-npu attentions.
    """

    _attr_names_to_apply = ['__getitem__']

    def __getitem__(cls, key):
        if key in NPU_ATTENTION_BACKEND:
            return NPU_ATTENTION_BACKEND[key]

        return _orig_getitem(cls, key)
