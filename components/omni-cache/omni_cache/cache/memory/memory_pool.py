# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""KV Cache Memory Pool implementation."""

import os
from typing import List, Union

import torch
import torch_npu
from vllm.logger import init_logger
from vllm.distributed.parallel_state import get_tp_group
from vllm.config.model import ModelConfig

from omni_cache.cache.device_backend.ascend import AscendCLStream, NPUTensorRegister

from .constants import HUGE_PAGE_SIZE, ENABLE_C8_INDEXER, SHAPE_RANK_PREFILL
from .shape_utils import (
    compute_head_size_ratio,
    extract_dimensions,
    compute_tensor_shapes,
    calculate_offsets_and_sizes,
)
from .hugepage_ops import (
    open_hugepage_file,
    create_memory_mapping,
    create_shared_tensor,
    split_kvi_tensors,
    setup_device_mapping,
    create_layer_tensor_tuples,
)
from .copy_ops import batch_layer_copy_to_npu as _batch_layer_copy_to_npu, memcpy_async as _memcpy_async

logger = init_logger("vllm.v1.omni")


class KVCacheMemoryPool:
    """A CPU memory pool for KV cache blocks using hugepage shared memory.

    Simulates vLLM's KV cache block pool with BF16 data type.
    Uses mmap for shared memory access between processes.
    """

    def __init__(
        self,
        hugepage_path: str,
        mmap_size: int,
        shape,
        rank: int = 0,
        device: Union[str, torch.device] = None,
        is_hybrid_attn: bool = False,
        enable_dsa: bool = False,
        vllm_config=None,
    ):
        """Initialize KV cache memory pool with mmap'd hugepage shared memory.

        Args:
            hugepage_path: Path to hugepage shared memory file.
            mmap_size: Size of memory mapping in bytes.
            shape: Tensor shape tuple.
            rank: Process rank.
            device: Device to use.
            is_hybrid_attn: Whether using hybrid attention.
            enable_dsa: Whether DSA is enabled.
        """
        self.device = device
        self.shared_tensor_npu = None
        self.mmap_size = mmap_size
        self.dtype = torch.bfloat16
        self.element_size = 2
        self.rank = rank
        self.enable_dsa = enable_dsa
        self.vllm_config = vllm_config
        # self.is_pangu_v2 = _is_pangu_v2_model(vllm_config)
        self.is_hybrid_attn = is_hybrid_attn

        self._init_config(shape, is_hybrid_attn)
        self._init_dimensions(shape)
        self._init_tensor_shapes(shape)
        self._init_memory_layout()
        self._init_hugepage_memory(hugepage_path, mmap_size)
        self._init_streams()

        print(f"KV Cache Pool: {self.num_blocks} blocks, {self.block_size_bytes} bytes/block")

    def _init_config(self, shape: tuple, is_hybrid_attn: bool) -> None:
        """Initialize configuration parameters."""
        self.tp_world_size = get_tp_group().world_size
        self.is_prefill = (len(shape) == SHAPE_RANK_PREFILL)
        self.head_size_ratio = compute_head_size_ratio(
            shape, is_hybrid_attn, self.tp_world_size, self.enable_dsa,
            vllm_config=self.vllm_config,
        )
        self.head_total_slices = sum(self.head_size_ratio)

    def _init_dimensions(self, shape: tuple) -> None:
        """Extract and initialize dimensions from shape."""
        self.num_ranks, self.num_layers, self.num_blocks, self.block_size = \
            extract_dimensions(shape)

        # Store base shape (block_num, block_size, head_dim)
        if len(shape) == SHAPE_RANK_PREFILL:
            self.base_shape = shape[1:]
        else:
            self.base_shape = shape[2:]

    def _init_tensor_shapes(self, shape: tuple) -> None:
        """Initialize tensor shapes for each component."""
        self.shapes = compute_tensor_shapes(
            self.base_shape, self.head_size_ratio, self.head_total_slices
        )

        # Handle special cases for DSA and hybrid attention
        if self.enable_dsa and not self.is_hybrid_attn:
            self.shapes = self._get_dsa_shapes(self.base_shape)
        elif ModelConfig.is_deepseek_mla and not self.is_hybrid_attn:
            self.shapes = self._get_mla_shapes(self.base_shape)
        elif len(self.shapes) == 1 and not ENABLE_C8_INDEXER:
            self.shapes = self._get_hybrid_shapes(self.base_shape)

    def _get_dsa_shapes(self, base_shape: tuple) -> List[tuple]:
        """Get shapes for DSA mode."""
        ratios = self.head_size_ratio
        total = self.head_total_slices
        dim = base_shape[-1]
        return [
            (*base_shape[:-1], dim * ratios[0] // total),
            (*base_shape[:-1], dim * ratios[1] // total),
            (*base_shape[:-1], dim * ratios[2] // total),
        ]

    def _get_mla_shapes(self, base_shape: tuple) -> List[tuple]:
        """Get shapes for MLA mode."""
        ratios = self.head_size_ratio
        total = self.head_total_slices
        dim = base_shape[-1]
        return [
            (*base_shape[:-1], dim * ratios[0] // total),
            (*base_shape[:-1], dim * ratios[1] // total),
        ]

    def _get_hybrid_shapes(self, base_shape: tuple) -> List[tuple]:
        """Get shapes for hybrid attention mode."""
        ratios = self.head_size_ratio
        total = self.head_total_slices
        dim = base_shape[-1]
        return [
            (*base_shape[:-1], dim * ratios[0] // total),
        ]

    def _init_memory_layout(self) -> None:
        """Calculate sizes and offsets for memory layout."""
        self.sizes, self.offsets, self.layer_sizes, self.layer_size_bytes = \
            calculate_offsets_and_sizes(
                self.shapes, self.element_size, self.num_layers
            )

        self.block_size_bytes = self.layer_size_bytes // self.num_blocks * self.num_layers
        self.size_total = self.num_layers * self.layer_size_bytes
        # Round UP to the next 2MB boundary so every rank's tensor starts
        # at a 2MB-aligned address regardless of dp_local_rank.
        self.aligned_rank_size = (
            (self.size_total + HUGE_PAGE_SIZE - 1) // HUGE_PAGE_SIZE
        ) * HUGE_PAGE_SIZE

        if self.num_blocks == 0:
            raise ValueError(
                f"mmap_size {self.mmap_size} too small for block size {self.block_size_bytes}"
            )

    def _init_hugepage_memory(self, hugepage_path: str, mmap_size: int) -> None:
        """Initialize hugepage memory mapping."""
        self.npu_tensor_register = NPUTensorRegister()
        self._map_hugepage_memory(hugepage_path, mmap_size, self.rank)

    def _init_streams(self) -> None:
        """Initialize AscendCL streams."""
        self.ascend_cl_stream = AscendCLStream()

    def _map_hugepage_memory(self, hugepage_path: str, mmap_size: int, rank: int) -> None:
        """Map hugepage shared memory file using mmap."""
        self.fd = open_hugepage_file(hugepage_path)
        # Ensure the mmap covers all ranks with 2MB-aligned strides.
        required_mmap_size = self.num_ranks * self.aligned_rank_size
        actual_mmap_size = max(mmap_size, required_mmap_size)
        self.mmap_obj = create_memory_mapping(self.fd, actual_mmap_size)
        self.mmap_size = actual_mmap_size

        self.shared_tensor = create_shared_tensor(
            self.mmap_obj, self.dtype, self.size_total,
            self.element_size, self.num_ranks, rank,
            rank_stride=self.aligned_rank_size
        )

        self.kvi_tensors = split_kvi_tensors(
            self.shared_tensor, self.shapes,
            self.head_size_ratio, self.head_total_slices, self.num_layers
        )

        if not self.is_prefill:
            self._setup_decode_tensors()

    def _setup_decode_tensors(self) -> None:
        """Set up tensors for decode mode."""
        self.shared_tensor_swap = setup_device_mapping(self, self.shared_tensor)

        self.kvi_tensors_swap = split_kvi_tensors(
            self.shared_tensor_swap, self.shapes,
            self.head_size_ratio, self.head_total_slices, self.num_layers
        )

        self.shared_tensor_npu = create_layer_tensor_tuples(
            self.kvi_tensors_swap, self.num_layers
        )

    def host_swap_device(self, strided_tensor: torch.Tensor) -> torch.Tensor:
        """Map host tensor to device.

        Args:
            strided_tensor: Tensor to map.

        Returns:
            Device-mapped tensor.
        """
        device_id = self.device.index if self.device else 0

        logger.warning(
            f"<<< host_swap_device rank{self.rank}] before register: "
            f"ptr=0x{strided_tensor.data_ptr():x}, "
            f"size={strided_tensor.nbytes}, "
            f"2M-align={strided_tensor.data_ptr() % (2<<20) == 0}, "
            f"4K-align={strided_tensor.data_ptr() % (4096) == 0}, "
            f"is_pinned={strided_tensor.is_pinned()}"
        )

        _, tensor_swap = self.npu_tensor_register.host_tensor_register(
            strided_tensor, device_id=device_id
        )
        return tensor_swap

    def get_block(self, block_idx: int) -> List[torch.Tensor]:
        """Return all-layer tensors for the specified block index."""
        if block_idx >= self.num_blocks:
            raise IndexError(
                f"Block index {block_idx} out of range (max {self.num_blocks - 1})"
            )

        # On decode with DSA *and* ENABLE_HOST_MAPPING=1, the attention
        # kernel accesses k_nope / k_rope directly from host memory via
        # the NPU-MMU-mapped alias (`kvi_tensors_swap`, created by
        # aclrtHostRegister MAPPED). H2D memcpy is not needed for those
        # components — only the indexer in the classic 3-tensor layout
        # needs a separate HBM copy.
        #
        # When ENABLE_HOST_MAPPING=0 (the default in pangu_v2_pd.sh),
        # the decode worker allocates its own HBM kv_cache, so the bytes
        # we just pulled into the host pool must actually be copied into
        # HBM per block — fall through to the normal copy path.
        from .constants import ENABLE_HOST_MAPPING

        if not self.is_prefill and self.enable_dsa and ENABLE_HOST_MAPPING and not self.is_hybrid_attn:
            if len(self.kvi_tensors) >= 3:
                # Classic 3-component DSA: copy only the indexer (slot 2).
                return [self.kvi_tensors[2][:, block_idx, ...]]
            # Unified pool w/ MMU alias: nothing to memcpy.
            return []

        block_tensors = []
        for kvi_tensor in self.kvi_tensors:
            block_tensors.append(kvi_tensor[:, block_idx, ...])
        return block_tensors

    def batch_layer_copy_to_npu(
        self,
        block_ids: list,
        npu_blocks,
        layer_indices=None
    ):
        """Batch copy layers from CPU to NPU."""
        return _batch_layer_copy_to_npu(
            self.get_block, block_ids, npu_blocks, layer_indices
        )

    def memcpy_async(self, batch_device_mem, batch_device_max,
                     batch_host_mem, batch_host_sizes):
        """Execute async memory copy."""
        _memcpy_async(
            self.ascend_cl_stream, batch_device_mem, batch_device_max,
            batch_host_mem, batch_host_sizes
        )

    def set_block(self, block_idx: int, tensors: List[torch.Tensor]) -> None:
        """Set tensors for the specified block index."""
        expected_len = 3 if self.enable_dsa else 2
        if len(tensors) != expected_len:
            raise ValueError(f"Expected {expected_len} tensors, got {len(tensors)}")

        for i, (tensor, expected_shape) in enumerate(zip(tensors, self.shapes)):
            expected_full_shape = (self.num_layers, *expected_shape[1:])
            if tensor.shape != expected_full_shape:
                raise ValueError(
                    f"Tensor {i} shape mismatch: got {tensor.shape}, "
                    f"expected {expected_full_shape}"
                )
            if tensor.dtype != self.dtype:
                raise ValueError(f"Tensor {i} dtype mismatch: {tensor.dtype} vs {self.dtype}")

        for i, (kvi_tensor, tensor) in enumerate(zip(self.kvi_tensors, tensors)):
            if block_idx >= kvi_tensor.shape[1]:
                raise IndexError(f"Block {block_idx} out of range for tensor {i}")
            kvi_tensor[:, block_idx, ...].copy_(tensor, non_blocking=True)

    def __getitem__(self, block_idx: int) -> List[torch.Tensor]:
        """pool[block_idx] returns tensors for that block."""
        return self.get_block(block_idx)

    def __setitem__(self, block_idx: int, tensors: List[torch.Tensor]) -> None:
        """pool[block_idx] = tensors."""
        self.set_block(block_idx, tensors)

    def close(self) -> None:
        """Close mmap and file descriptor."""
        mmap_obj = getattr(self, 'mmap_obj', None)
        if mmap_obj is not None:
            try:
                mmap_obj.close()
            except (BufferError, ValueError, OSError):
                pass
            finally:
                self.mmap_obj = None

        fd = getattr(self, 'fd', None)
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
            finally:
                self.fd = None

    def __del__(self):
        """Ensure resources are cleaned up."""
        self.close()

    def info(self) -> None:
        """Print memory pool configuration."""
        print(f"\n=== KV Cache Memory Pool ===")
        print(f"Data type: {self.dtype}")
        print(f"Blocks: {self.num_blocks}")
        print(f"Block size: {self.block_size_bytes} bytes")
        print(f"Total memory: {self.mmap_size} bytes")
        print(f"Tensor shapes per block:")
        print(f"  Nope KV: {self.shapes[0]}")
        if len(self.shapes) > 1:
            print(f"  PE KV: {self.shapes[1]}")
        if self.enable_dsa and len(self.shapes) > 2:
            print(f"  Indexer: {self.shapes[2]}")
