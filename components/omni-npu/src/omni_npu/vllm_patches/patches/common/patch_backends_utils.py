# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright contributors to the vLLM project.

from vllm.v1.attention.backends.utils import CommonAttentionMetadata

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


@register_patch("NPUCommonAttentionMetadata", CommonAttentionMetadata)
class CommonAttentionMetadataPatch(VLLMPatch):
    """
    Patch to modify the CommonAttentionMetadata behavior for compatibility with vLLM.
    """
    _attr_names_to_apply = ['unpadded']

    def unpadded(
        self, num_actual_tokens: int, num_actual_reqs: int
    ) -> "CommonAttentionMetadata":
        self.num_tokens_unpadded = num_actual_tokens
        self.num_reqs_unpadded = num_actual_reqs
        return self
