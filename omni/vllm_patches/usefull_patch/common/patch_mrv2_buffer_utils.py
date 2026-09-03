# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""V2 input buffers on NPU. Implementation: omni/worker/npu/buffer_utils.py.

Upstream UvaBuffer assumes CUDA UVA semantics. NPUUvaBuffer stages an H2D copy
by default and only aliases pinned host memory when OMNI_NPU_V2_UVA is set
(see omni/worker/npu/mode.py). UvaBufferPool looks the class up as a module
global and takes no factory argument, so the name has to be replaced.

Same target module as patch_uva.py but a different attribute; the framework
tracks ownership per attribute name.
"""

from vllm.logger import init_logger

from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.worker.npu.buffer_utils import NPUUvaBuffer

logger = init_logger(__name__)

try:
    import vllm.v1.worker.gpu.buffer_utils as up_buffer_utils
except ImportError:
    logger.error(
        "[omni-npu/mrv2] buffer_utils patch target unavailable; not registered",
        exc_info=True,
    )
else:

    @register_patch("MRv2UvaBufferPatch", up_buffer_utils)
    class MRv2UvaBufferPatch(VLLMPatch):
        _attr_names_to_apply = ["UvaBuffer"]

        UvaBuffer = NPUUvaBuffer
