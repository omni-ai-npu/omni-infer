# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture-time graph signals for MRv2. Implementation: omni/worker/npu/aclgraph_utils.py.

Two patches, both serving P5 (see 2-ModelRunnerV2支持 Graph分析/ModelRunnerV2图模式适配设计.md):

- ForwardContext.capturing: omni's attention backends read this attribute, which
  upstream does not define. MRv1 assigns it per instance; V2 never does, so the
  descriptor supplies a thread-local default while an explicit assignment still wins.
- CudaGraphManager.capture: upstream calls the recorded pass with NONE because no
  inner wrapper takes over once V2 records from the outside, and omni reads that
  field as "am I in a graph". Wrapping the create_forward_fn factory carries both
  the rewritten mode and the capturing flag on the one call that matters: the
  factory's own `warmup` argument separates the two passes, so neither signal
  reaches the warmup pass that runs outside the graph, and neither is attached to
  torch.cuda.graph, which unrelated code also enters.
"""

from vllm.logger import init_logger

from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.worker.npu.aclgraph_utils import (
    _CapturingDescriptor,
    build_capture_with_full_mode,
)

logger = init_logger(__name__)


try:
    from vllm.forward_context import ForwardContext
except ImportError:
    logger.error(
        "[omni-npu/mrv2] ForwardContext unavailable; capturing flag not registered",
        exc_info=True,
    )
else:

    @register_patch("MRv2CapturingFlagPatch", ForwardContext)
    class MRv2CapturingFlagPatch(VLLMPatch):
        _attr_names_to_apply = ["capturing"]

        capturing = _CapturingDescriptor()


try:
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
except ImportError:
    logger.error(
        "[omni-npu/mrv2] CudaGraphManager unavailable; capture mode not patched",
        exc_info=True,
    )
else:
    # The base method is what ModelCudaGraphManager and the speculator managers
    # all reach through super(), so one patch covers every capture path.
    @register_patch("MRv2CaptureModePatch", CudaGraphManager)
    class MRv2CaptureModePatch(VLLMPatch):
        _attr_names_to_apply = ["capture"]

        capture = build_capture_with_full_mode(CudaGraphManager.capture)
