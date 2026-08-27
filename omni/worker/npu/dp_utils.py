# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""DP adaptations for the NPU V2 model runner.

Injected by omni/vllm_patches/usefull_patch/patch_mrv2_dp_utils.py.
"""

from __future__ import annotations

from dataclasses import replace

from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.worker.gpu import dp_utils as up_dp_utils

logger = init_logger(__name__)

# Captured at import, which the plugin performs before any patch is applied.
_DISPATCH_ORIGINAL = up_dp_utils.dispatch_cg_and_sync_dp
_SYNC_ORIGINAL = up_dp_utils.sync_cudagraph_and_dp_padding


def dispatch_cg_and_sync_dp(*args, **kwargs):
    """Publish the LM head's DP all_gather pad target on every step.

    ``NPULogitsProcessor._get_logits`` pads ``hidden_states`` up to
    ``NPUParallelLMHead._dp_pad_n`` before all_gathering, reading that class
    attribute instead of negotiating the size, so every rank must be handed
    the same value first; left at 0 the collective never completes and
    surfaces later as a stream timeout on the LM head GEMM. MRv1 does this in
    ``_capture_dp_pad_target``; V2 computes the same counts here, once per
    step before the forward context.
    """
    batch_desc, num_tokens_across_dp = _DISPATCH_ORIGINAL(*args, **kwargs)
    if num_tokens_across_dp is not None:
        _stash_lmhead_pad_target(num_tokens_across_dp)
    return batch_desc, num_tokens_across_dp


def _stash_lmhead_pad_target(num_tokens_across_dp) -> None:
    from omni_npu.model_config.config_loader.loader import model_extra_config

    parall = model_extra_config.parall_config
    dp_lmhead = getattr(parall, "ena_dp_lmhead_parallel", False)
    local_lmhead = getattr(parall, "ena_local_lmhead_parallel", False)
    if not (dp_lmhead or local_lmhead):
        return

    from omni_npu.v1.layers.vocab_parallel_embedding import NPUParallelLMHead

    if local_lmhead:
        # Only the ranks sharing this node take part in the collective.
        from omni_npu.v1.distributed.parallel_state_ext import (
            get_local_world_group,
        )

        ranks = get_local_world_group().ranks
        NPUParallelLMHead._dp_pad_n = max(
            int(num_tokens_across_dp[r]) for r in ranks
        )
    else:
        NPUParallelLMHead._dp_pad_n = int(num_tokens_across_dp.max())


def _ep_over_dp(cudagraph_manager) -> bool:
    vllm_config = getattr(cudagraph_manager, "vllm_config", None)
    if vllm_config is None:
        # The profiling run, where every rank runs one shape anyway.
        return False
    parallel_config = vllm_config.parallel_config
    return (
        bool(parallel_config.enable_expert_parallel)
        and parallel_config.data_parallel_size > 1
    )


def sync_cudagraph_and_dp_padding(cudagraph_manager, *args, **kwargs):
    """Force DP token padding for MoE EP, as MRv1's DpUtilsEagerEpPadPatch does.

    Upstream returns the rank's own unpadded token count once the synced
    cudagraph mode is NONE, so under --enforce-eager the ranks reach
    npu_moe_distribute_dispatch_v2 with different row counts and the fused MC2
    collectives deadlock; a matching global_bs is no substitute (the tiling
    rejects one that disagrees with x.size(0), aclnn 561002). MRv1 patches its
    own dp_utils._synchronize_dp_ranks. enable_expert_parallel comes off
    cudagraph_manager.vllm_config; get_current_vllm_config() is None at step.
    """
    batch_desc, num_tokens_across_dp = _SYNC_ORIGINAL(
        cudagraph_manager, *args, **kwargs
    )

    # None means every rank had zero tokens.
    if num_tokens_across_dp is None:
        return batch_desc, num_tokens_across_dp
    # The cudagraph branch already padded and rewrote the vector.
    if batch_desc.cg_mode != CUDAGraphMode.NONE:
        return batch_desc, num_tokens_across_dp
    if not _ep_over_dp(cudagraph_manager):
        return batch_desc, num_tokens_across_dp

    padded_num_tokens = int(num_tokens_across_dp.max().item())

    # Unconditional, like the cudagraph branch upstream: the vector reaches
    # every rank's forward context, so skipping it on the rank that already
    # held the maximum leaves that rank disagreeing with its peers.
    num_tokens_across_dp[:] = padded_num_tokens

    if padded_num_tokens == batch_desc.num_tokens:
        return batch_desc, num_tokens_across_dp

    logger.info_once(
        "[omni-npu/mrv2] forcing DP token padding for MoE EP in eager mode",
        scope="local",
    )
    return (
        replace(batch_desc, num_tokens=padded_num_tokens),
        num_tokens_across_dp,
    )
