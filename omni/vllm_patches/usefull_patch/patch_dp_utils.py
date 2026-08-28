# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Adapt DP coordination and padding policy for vLLM v0.25.1 on NPU.

This patch keeps the v0.25.1 four-row coordination tensor and signatures,
optionally moves DP synchronization to a dedicated AI_CPU HCCL group, and
aligns eager MoE token counts unless a validated A2 all2allv path can consume
uneven token counts.
"""

import torch
import torch.distributed as dist
import torch_npu

import vllm.v1.worker.dp_utils as dp_utils
from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import GroupCoordinator, get_dp_group
from vllm.logger import init_logger
from vllm.v1.worker.dp_utils import (
    _post_process_cudagraph_mode,
    _post_process_dp_padding,
    _post_process_ubatch,
)

from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.model_config.config_loader.loader import model_extra_config

logger = init_logger(__name__)

_original_run_ar = dp_utils._run_ar
_dp_sync_copy_stream: torch.npu.Stream | None = None
_dp_sync_event: torch.Event | None = None
_dp_sync_device_group: dist.ProcessGroup | None = None
_dp_sync_device: torch.device | None = None
_dp_sync_group_key: tuple[tuple[int, ...], str] | None = None
_aicpu_dp_sync_init_failed = False


def _get_dp_sync_primitives() -> (
    tuple[torch.npu.Stream, torch.Event, dist.ProcessGroup, torch.device] | None
):
    global _dp_sync_copy_stream, _dp_sync_event
    global _dp_sync_device_group, _dp_sync_device, _dp_sync_group_key
    global _aicpu_dp_sync_init_failed

    if _aicpu_dp_sync_init_failed:
        return None

    dp_group = get_dp_group()
    group_key = (tuple(dp_group.ranks), str(dp_group.device))
    if _dp_sync_group_key == group_key:
        assert _dp_sync_copy_stream is not None
        assert _dp_sync_event is not None
        assert _dp_sync_device_group is not None
        assert _dp_sync_device is not None
        return (
            _dp_sync_copy_stream,
            _dp_sync_event,
            _dp_sync_device_group,
            _dp_sync_device,
        )

    try:
        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        options.hccl_config = {
            "hccl_op_expansion_mode": 2,
            "hccl_buffer_size": 20,
            "group_name": f"{dp_group.unique_name}_aicpu",
        }
        device_group = dist.new_group(
            dp_group.ranks,
            backend=dist.get_backend(dp_group.device_group),
            pg_options=options,
            # Follow the GroupCoordinator-wide setting: in RL/ray deployments
            # the vllm ranks are a subset of the default world, and group
            # creation must rendezvous locally (group members only) instead of
            # doing a default-world-wide handshake.
            use_local_synchronization=getattr(
                GroupCoordinator, "use_local_synchronization", False
            ),
        )
        copy_stream = torch.npu.Stream()
        event = torch.Event()
    except Exception:
        _aicpu_dp_sync_init_failed = True
        logger.exception(
            "AICPU DP sync initialization failed; using native vLLM DP sync."
        )
        return None

    _dp_sync_copy_stream = copy_stream
    _dp_sync_event = event
    _dp_sync_device_group = device_group
    _dp_sync_device = dp_group.device
    _dp_sync_group_key = group_key
    logger.info_once(
        "Using AICPU HCCL all-reduce to synchronize DP ranks.",
        scope="local",
    )
    return copy_stream, event, device_group, dp_group.device


def _run_ar(
    should_ubatch: bool,
    orig_num_tokens_per_ubatch: int,
    padded_num_tokens_per_ubatch: int,
    cudagraph_mode: int,
    parallel_config: ParallelConfig,
) -> torch.Tensor:
    if not model_extra_config.parall_config.enable_aicpu_dp_sync:
        return _original_run_ar(
            should_ubatch,
            orig_num_tokens_per_ubatch,
            padded_num_tokens_per_ubatch,
            cudagraph_mode,
            parallel_config,
        )

    primitives = _get_dp_sync_primitives()
    if primitives is None:
        return _original_run_ar(
            should_ubatch,
            orig_num_tokens_per_ubatch,
            padded_num_tokens_per_ubatch,
            cudagraph_mode,
            parallel_config,
        )

    dp_size = parallel_config.data_parallel_size
    dp_rank = parallel_config.data_parallel_rank
    cpu_tensor = torch.zeros(
        4,
        dp_size,
        dtype=torch.int32,
        device="cpu",
        pin_memory=True,
    )
    cpu_tensor[0, dp_rank] = orig_num_tokens_per_ubatch
    cpu_tensor[1, dp_rank] = padded_num_tokens_per_ubatch
    cpu_tensor[2, dp_rank] = int(should_ubatch)
    cpu_tensor[3, dp_rank] = cudagraph_mode

    stream, event, device_group, device = primitives
    with torch.npu.stream(stream):
        device_tensor = cpu_tensor.to(device, non_blocking=True)
        dist.all_reduce(device_tensor, group=device_group)
        cpu_tensor.copy_(device_tensor, non_blocking=True)
        event.record()

    event.synchronize()
    return cpu_tensor


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
