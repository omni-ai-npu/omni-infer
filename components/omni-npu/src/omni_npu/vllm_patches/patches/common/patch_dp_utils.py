# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
import torch_npu
import torch.distributed as dist


from vllm.config import CUDAGraphMode, ParallelConfig
from vllm.distributed.parallel_state import get_dp_group
from vllm.logger import init_logger
import vllm.v1.worker.dp_utils as dp_utils
from vllm.v1.worker.dp_utils import _run_ar as _original_run_ar

from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.model_config.config_loader.loader import model_extra_config

logger = init_logger(__name__)

# adapt start: Lazy-initialized stream, event, device, and device group for async DP synchronization
_dp_sync_copy_stream: torch.npu.Stream | None = None
_dp_sync_event: torch.Event | None = None
_dp_sync_device_group: dist.ProcessGroup | None = None
_dp_sync_device: torch.device | None = None

def _get_dp_sync_primitives() -> tuple[torch.npu.Stream, torch.Event, dist.ProcessGroup, torch.device]:
    global _dp_sync_copy_stream, _dp_sync_event, _dp_sync_device_group, _dp_sync_device
    if _dp_sync_copy_stream is None:
        _dp_sync_copy_stream = torch.npu.Stream()
        _dp_sync_event = torch.Event()

        dp_group = get_dp_group()
        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        options.hccl_config = {
            # 0: default to HCCL_OP_EXPANSION_MODE
            # 1: Host
            # 2: AI_CPU
            # 3: AIV
            "hccl_op_expansion_mode": 2, # need HCCL_OP_EXPANSION_MODE=AI_CPU unless CANN >= 9.0
            "hccl_buffer_size": 20,
        }
        _dp_sync_device_group = dist.new_group(
            dp_group.ranks,
            backend=dist.get_backend(dp_group.device_group),
            pg_options=options,
        )
        _dp_sync_device = dp_group.device

    return _dp_sync_copy_stream, _dp_sync_event, _dp_sync_device_group, _dp_sync_device
# adapt end


def _run_ar(
    should_ubatch: bool,
    should_dp_pad: bool,
    orig_num_tokens_per_ubatch: int,
    padded_num_tokens_per_ubatch: int,
    cudagraph_mode: int,
    parallel_config: ParallelConfig,
) -> torch.Tensor:
    if not model_extra_config.parall_config.enable_aicpu_dp_sync:
        return _original_run_ar(
            should_ubatch,
            should_dp_pad,
            orig_num_tokens_per_ubatch,
            padded_num_tokens_per_ubatch,
            cudagraph_mode,
            parallel_config,
        )

    # adapt start: Use dedicated copy stream for async H2D/all_reduce/D2H
    dp_size = parallel_config.data_parallel_size
    dp_rank = parallel_config.data_parallel_rank

    cpu_tensor = torch.zeros(
        5, dp_size, dtype=torch.int32, device="cpu", pin_memory=True,
    )
    cpu_tensor[0][dp_rank] = orig_num_tokens_per_ubatch
    cpu_tensor[1][dp_rank] = padded_num_tokens_per_ubatch
    cpu_tensor[2][dp_rank] = 1 if should_ubatch else 0
    cpu_tensor[3][dp_rank] = 1 if should_dp_pad else 0
    cpu_tensor[4][dp_rank] = cudagraph_mode

    stream, event, device_group, device = _get_dp_sync_primitives()
    with torch.npu.stream(stream):
        device_tensor = cpu_tensor.to(device, non_blocking=True)
        dist.all_reduce(device_tensor, group=device_group)
        cpu_tensor.copy_(device_tensor, non_blocking=True)
        event.record()

    event.synchronize()
    return cpu_tensor
    # adapt end


def _synchronize_dp_ranks(
    num_tokens_unpadded: int,
    num_tokens_padded: int,
    should_attempt_ubatching: bool,
    should_attempt_dp_padding: bool,
    cudagraph_mode: int,
    parallel_config: ParallelConfig,
) -> tuple[bool, torch.Tensor | None, int]:
    assert num_tokens_padded >= num_tokens_unpadded

    tensor = _run_ar(
        should_ubatch=should_attempt_ubatching,
        should_dp_pad=should_attempt_dp_padding,
        orig_num_tokens_per_ubatch=num_tokens_unpadded,
        padded_num_tokens_per_ubatch=num_tokens_padded,
        cudagraph_mode=cudagraph_mode,
        parallel_config=parallel_config,
    )

    synced_cudagraph_mode = dp_utils._post_process_cudagraph_mode(tensor)

    should_dp_pad = bool(torch.all(tensor[3] == 1).item())

    should_ubatch = dp_utils._post_process_ubatch(
        tensor, parallel_config.num_ubatches
    )
    if should_ubatch and not should_dp_pad:
        logger.debug_once(
            "Microbatching has been triggered and requires DP padding. "
            "Enabling DP padding even though it has been explicitly "
            "disabled.",
            scope="global",
        )
        should_dp_pad = True

    # Based on vLLM issue #34102: eager DP padding can create dummy tokens
    # on idle DP ranks, which hot-spots MoE experts and may cause OOM.
    use_all2all_without_dp_padding = (
        torch_npu.npu.get_device_name(0).startswith("Ascend910B")
        and parallel_config.tensor_parallel_size != 1
        and parallel_config.data_parallel_size != 1
        and synced_cudagraph_mode != CUDAGraphMode.FULL.value
    )
    if use_all2all_without_dp_padding:
        should_dp_pad = False

    num_tokens_after_padding = dp_utils._post_process_dp_padding(
        tensor, should_dp_pad
    )

    return should_ubatch, num_tokens_after_padding, synced_cudagraph_mode


@register_patch("DpUtilsPatch", dp_utils)
class DpUtilsPatch(VLLMPatch):
    _attr_names_to_apply = [
        "_run_ar",
        "_synchronize_dp_ranks",
    ]

    _run_ar = _run_ar
    _synchronize_dp_ranks = _synchronize_dp_ranks