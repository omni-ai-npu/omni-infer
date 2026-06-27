# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Copy operations for KV cache memory pool."""

import os
import time
import ctypes
from typing import List, Optional

from vllm.logger import init_logger

logger = init_logger("vllm.v1.omni")


# Set OMNI_TRACE_MEMCPY=1 to sync after each individual memcpy call so a
# failing sync shows which entry caused it.
_TRACE_MEMCPY = os.getenv("OMNI_TRACE_MEMCPY", "0") == "1"


def batch_layer_copy_to_npu(
    get_block_fn,
    block_ids: list,
    npu_blocks,
    layer_indices: Optional[List[int]] = None,
):
    """Batch copy specific layers of multiple blocks from CPU to NPU.

    Merges consecutive blocks with continuous addresses for batch copy.

    Args:
        get_block_fn: Function to get block tensors.
        block_ids: List of block IDs to copy.
        npu_blocks: NPU block tensors.
        layer_indices: Optional layer indices to copy.

    Returns:
        Tuple of (batch_device_mem, batch_device_max, batch_host_mem, batch_host_sizes).
    """
    cpu_blocks = [get_block_fn(block_id) for block_id in block_ids]
    # Groups may have different sizes, so take the max here.
    layers = max([len(npu_blocks_i) for npu_blocks_i in npu_blocks])

    batch_device_mem: list = []
    batch_device_max: list = []
    batch_host_mem: list = []
    batch_host_sizes: list = []

    for kvi_idx in range(len(cpu_blocks[0])):
        for layer_idx in range(layers):
            cpu_layer_idx = layer_idx if layer_indices is None else layer_indices[layer_idx]
            cpu_addrs = []
            npu_addrs = []
            npu_sizes = []
            for i in range(len(block_ids)):
                if len(npu_blocks[i]) > len(cpu_blocks[i][kvi_idx]):
                    raise RuntimeError(
                        f"Device layers should not exceed host layers. "
                        f"But got {len(npu_blocks[i])=}, {len(cpu_blocks[i][kvi_idx])=}."
                    )
                if layer_idx >= len(npu_blocks[i]):
                    continue

                # NOTE: in h2d_copy_ops_hbm_buffer(), each npu_block is a tuple;
                # indexing by [layer_idx][kvi_idx] selects the per-kvi component.
                npu_block = npu_blocks[i][layer_idx][kvi_idx]
                tensor_size = npu_block.nbytes
                npu_addrs.append(npu_block.data_ptr())
                npu_sizes.append(tensor_size)

                cpu_block = cpu_blocks[i][kvi_idx][cpu_layer_idx]

                if cpu_block.nbytes > tensor_size:
                    # Host slot is wider than the NPU target — copy only the
                    # tail portion that fits (skip leading padding).
                    offset_bytes = cpu_block.nbytes - tensor_size
                    offset_elements = offset_bytes // cpu_block.element_size()
                    cpu_tail = cpu_block.flatten()[offset_elements:]
                    cpu_addrs.append(cpu_tail.data_ptr())
                else:
                    cpu_addrs.append(cpu_block.data_ptr())

            batch_start = 0
            while batch_start < len(cpu_addrs):
                batch_end = batch_start + 1
                prev_cpu_addr = cpu_addrs[batch_start]
                prev_npu_addr = npu_addrs[batch_start]
                cur_size = npu_sizes[batch_start]
                total_dev_bytes = cur_size
                total_host_bytes = cur_size
                while batch_end < len(cpu_addrs):
                    nxt_size = npu_sizes[batch_end]
                    if (cpu_addrs[batch_end] == prev_cpu_addr + nxt_size and
                        npu_addrs[batch_end] == prev_npu_addr + nxt_size):
                        prev_cpu_addr = cpu_addrs[batch_end]
                        prev_npu_addr = npu_addrs[batch_end]
                        total_dev_bytes += nxt_size
                        total_host_bytes += nxt_size
                        batch_end += 1
                    else:
                        break
                count_blocks = batch_end - batch_start
                batch_device_mem.append(npu_addrs[batch_start])
                batch_device_max.append(total_dev_bytes)
                batch_host_mem.append(cpu_addrs[batch_start])
                batch_host_sizes.append(total_host_bytes)
                batch_start = batch_end

    return batch_device_mem, batch_device_max, batch_host_mem, batch_host_sizes


def memcpy_async(
    ascend_cl_stream,
    batch_device_mem,
    batch_device_max,
    batch_host_mem,
    batch_host_sizes,
):
    """Execute async memory copy from host to device."""
    start_time = time.time()
    batch_count = len(batch_device_mem)

    for idx in range(batch_count):
        device_mem = ctypes.c_void_p(batch_device_mem[idx])
        device_max = batch_device_max[idx]
        host_ptr = ctypes.c_void_p(batch_host_mem[idx])
        host_size = batch_host_sizes[idx]

        logger.debug(
            "[MEMCPY-TRACE] idx=%d/%d dst=0x%x dst_max=%d src=0x%x src_size=%d",
            idx, batch_count,
            batch_device_mem[idx], device_max,
            batch_host_mem[idx], host_size,
        )

        try:
            ascend_cl_stream.memcpy_async(device_mem, device_max, host_ptr, host_size, 1)
        except RuntimeError as e:
            logger.error(
                f"memcpy_async failed at entry idx={idx}/{batch_count}: "
                f"dst=0x{batch_device_mem[idx]:x} dst_max={device_max} "
                f"src=0x{batch_host_mem[idx]:x} count={host_size}: {e}"
            )
            raise


        if _TRACE_MEMCPY:
            try:
                ascend_cl_stream.sync()
            except RuntimeError as e:
                logger.error(
                    f"post-memcpy sync failed at entry idx={idx}/{batch_count}: "
                    f"dst=0x{batch_device_mem[idx]:x} dst_max={device_max} "
                    f"src=0x{batch_host_mem[idx]:x} count={host_size}: {e}"
                )
                raise

    mb = sum(batch_host_sizes) >> 20
    try:
        ascend_cl_stream.sync()
    except RuntimeError as e:
        logger.error(f"final memcpy sync failed after {batch_count} entries: {e}")
        raise
    dur = time.time() - start_time
    logger.warning(
        f"Batch (merged) {batch_count} copy {mb} MB took {dur * 1000:.2f} ms"
    )
