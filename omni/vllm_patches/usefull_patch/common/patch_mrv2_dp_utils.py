# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""V2 DP coordination on NPU. Implementation: omni/worker/npu/dp_utils.py.

dispatch_cg_and_sync_dp publishes the LM head's all_gather pad target
(NPUParallelLMHead._dp_pad_n) every step; left stale the collective never
completes and surfaces as a stream timeout on the LM head GEMM.
sync_cudagraph_and_dp_padding levels the per-rank token counts in eager mode,
without which MoE EP deadlocks in npu_moe_distribute_dispatch_v2 (aclnn 561002).

sync_cudagraph_and_dp_padding is only reached through the module global, so the
defining module is enough. dispatch_cg_and_sync_dp has three from-import
consumers (model_runner plus the autoregressive and dflash speculators) that are
already imported when patches apply, so each is replaced explicitly.

MRv1's counterpart is patch_dp_utils.py on vllm/v1/worker/dp_utils.py -- change
one, consider the other.
"""

from vllm.logger import init_logger

from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.worker.npu.dp_utils import (
    dispatch_cg_and_sync_dp,
    sync_cudagraph_and_dp_padding,
)

logger = init_logger(__name__)

try:
    import vllm.v1.worker.gpu.dp_utils as up_dp_utils
    import vllm.v1.worker.gpu.model_runner as up_model_runner
except ImportError:
    logger.error(
        "[omni-npu/mrv2] dp_utils patch targets unavailable; not registered",
        exc_info=True,
    )
else:

    @register_patch("MRv2DpUtilsPatch", up_dp_utils)
    class MRv2DpUtilsPatch(VLLMPatch):
        """Defining module."""

        _attr_names_to_apply = [
            "dispatch_cg_and_sync_dp",
            "sync_cudagraph_and_dp_padding",
        ]

        dispatch_cg_and_sync_dp = dispatch_cg_and_sync_dp
        sync_cudagraph_and_dp_padding = sync_cudagraph_and_dp_padding

    @register_patch("MRv2ModelRunnerDispatchPatch", up_model_runner)
    class MRv2ModelRunnerDispatchPatch(VLLMPatch):
        """execute_model's from-import."""

        _attr_names_to_apply = ["dispatch_cg_and_sync_dp"]

        dispatch_cg_and_sync_dp = dispatch_cg_and_sync_dp


try:
    import vllm.v1.worker.gpu.spec_decode.autoregressive.speculator as up_ar_speculator
except ImportError:
    logger.warning(
        "[omni-npu/mrv2] autoregressive speculator unavailable; "
        "dispatch_cg_and_sync_dp not patched there"
    )
else:

    @register_patch("MRv2AutoregressiveSpeculatorDispatchPatch", up_ar_speculator)
    class MRv2AutoregressiveSpeculatorDispatchPatch(VLLMPatch):
        _attr_names_to_apply = ["dispatch_cg_and_sync_dp"]

        dispatch_cg_and_sync_dp = dispatch_cg_and_sync_dp


try:
    import vllm.v1.worker.gpu.spec_decode.dflash.speculator as up_dflash_speculator
except ImportError:
    logger.warning(
        "[omni-npu/mrv2] dflash speculator unavailable; "
        "dispatch_cg_and_sync_dp not patched there"
    )
else:

    @register_patch("MRv2DflashSpeculatorDispatchPatch", up_dflash_speculator)
    class MRv2DflashSpeculatorDispatchPatch(VLLMPatch):
        _attr_names_to_apply = ["dispatch_cg_and_sync_dp"]

        dispatch_cg_and_sync_dp = dispatch_cg_and_sync_dp
