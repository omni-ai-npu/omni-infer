# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Force DP token-count alignment when MoE EP spans DP ranks.

Upstream only enables DP padding when cudagraph is active (or ubatching).
With --enforce-eager, idle DP ranks keep their local token count (often 1)
while an active MTP verification step uses 1+num_spec tokens. When EP≈DP,
npu_moe_distribute_dispatch_v2 fullmesh then hangs.

Force should_dp_pad=True whenever expert parallel is on and DP>1.
"""

from vllm.config import ParallelConfig
from vllm.logger import init_logger
import vllm.v1.worker.dp_utils as dp_utils
from vllm.v1.worker.dp_utils import (
    _post_process_cudagraph_mode,
    _post_process_dp_padding,
    _post_process_ubatch,
    _run_ar,
)

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logger = init_logger(__name__)


def _synchronize_dp_ranks(
    num_tokens_unpadded: int,
    num_tokens_padded: int,
    should_attempt_ubatching: bool,
    cudagraph_mode: int,
    parallel_config: ParallelConfig,
) -> tuple[bool, object | None, int]:
    assert num_tokens_padded >= num_tokens_unpadded

    tensor = _run_ar(
        should_ubatch=should_attempt_ubatching,
        orig_num_tokens_per_ubatch=num_tokens_unpadded,
        padded_num_tokens_per_ubatch=num_tokens_padded,
        cudagraph_mode=cudagraph_mode,
        parallel_config=parallel_config,
    )

    synced_cudagraph_mode = _post_process_cudagraph_mode(tensor)
    should_ubatch = _post_process_ubatch(tensor, parallel_config.num_ubatches)

    # Upstream: pad only when cudagraph/ubatch is on.
    should_dp_pad = synced_cudagraph_mode != 0 or should_ubatch
    
    ## patch start
    # MoE EP over DP requires equal per-rank token counts even in eager,
    # otherwise EP collectives (dispatch/combine) deadlock across ranks.
    if (
        parallel_config.enable_expert_parallel
        and parallel_config.data_parallel_size > 1
    ):
        if not should_dp_pad:
            logger.debug_once(
                "Forcing DP padding for MoE EP in eager mode "
                "(enable_expert_parallel=True, data_parallel_size=%s).",
                parallel_config.data_parallel_size,
                scope="local",
            )
        should_dp_pad = True
    ## patch end

    
    num_tokens_after_padding = _post_process_dp_padding(tensor, should_dp_pad)
    return should_ubatch, num_tokens_after_padding, synced_cudagraph_mode


@register_patch("DpUtilsEagerEpPadPatch", dp_utils)
class DpUtilsEagerEpPadPatch(VLLMPatch):
    _attr_names_to_apply = ["_synchronize_dp_ranks"]

    # Assign the free function directly onto the module.
    _synchronize_dp_ranks = _synchronize_dp_ranks
