# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""KV-cache reshape for NPU backends. Implementation: omni/worker/npu/attn_utils.py.

NPU backends need their own KV layout and upstream _reshape_kv_cache has no
backend hook, so the replacement wraps it and lets any group exposing
backend.reshape_kv_cache override the result.

Only the defining module needs patching: _reshape_kv_cache is private and its
sole caller init_kv_cache resolves it as a module global. MRv1 imports
_reshape_attention_kv_cache from here, a different function.
"""

from vllm.logger import init_logger

from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.worker.npu.attn_utils import reshape_kv_cache

logger = init_logger(__name__)

try:
    import vllm.v1.worker.gpu.attn_utils as up_attn_utils
except ImportError:
    logger.error(
        "[omni-npu/mrv2] attn_utils patch target unavailable; not registered",
        exc_info=True,
    )
else:

    @register_patch("MRv2ReshapeKvCachePatch", up_attn_utils)
    class MRv2ReshapeKvCachePatch(VLLMPatch):
        _attr_names_to_apply = ["_reshape_kv_cache"]

        _reshape_kv_cache = reshape_kv_cache
