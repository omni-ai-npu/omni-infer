import torch
import torch_npu
import mmap
import os
from typing import List, Optional, Union
import ctypes
import time
from omni.accelerators.cache.ascend_acl import *

from omni.models.config_loader.loader import model_extra_config
from vllm.distributed.parallel_state import get_tp_group, get_dp_group, get_world_group

if int(os.getenv("ENABLE_HOST_MAPPING", "0")):
    from omni.accelerators.cache.lib_tensor_register.acl_tensor_register import NPUTensorRegister

from vllm.logger import init_logger
logger = init_logger("vllm.v1.omni")

dump_data = False

HUGE_PAGE_SIZE = 2 * 1024 * 1024  # 2MB
REGIST_SIZE = 4096
MAP_HUGETLB = 0x40000

class KVCacheMemoryPool:
    """
    A CPU memory pool for KV cache blocks using hugepage shared memory.
    Simulates vLLM's KV cache block pool with BF16 data type.
    Uses mmap for shared memory access between processes.
    """

    def __init__(self,
                hugepage_path: str,
                mmap_size: int,
                shape, rank: int = 0,
                device: Union[str, torch.device] = None):
        """
        Initialize KV cache memory pool with mmap'd hugepage shared memory.

        Args:
            hugepage_path: Path to hugepage shared memory file (e.g., '/dev/hugepages/huge_shm')
            mmap_size: Size of memory mapping in bytes
        """
        self.device = device
        self.shared_tensor_npu = None
        self.mmap_size = mmap_size
        self.dtype = torch.bfloat16
        self.element_size = 2  # bfloat16 = 2 bytes
        self.tp_world_size = get_tp_group().world_size
        self.rank = rank

        self.is_prefill = (len(shape) == 4)

        if model_extra_config.operator_opt_config.enable_dsa:
            self.head_size_ratio = [512, 64, 128]
        else:
            self.head_size_ratio = [512, 64]
        self.head_size_ratio = [head_size_tmp // self.tp_world_size for head_size_tmp in self.head_size_ratio]
        self.head_total_slices = sum(self.head_size_ratio)

        self.num_ranks = 1
        # for prefill side
        if len(shape) == 4:
            # (layer_num, block_num, block_size, head_dim)
            self.num_layers = shape[0]
            self.num_blocks = shape[1]
            self.block_size = shape[2]
            shape = shape[1:] # (block_num, block_size, head_dim)
        # for decode side
        elif len(shape) == 5:
            # (die_num, layer_num, block_num, block_size, head_dim)
            self.num_ranks = shape[0]
            self.num_layers = shape[1]
            self.num_blocks = shape[2]
            self.block_size = shape[3]
            shape = shape[2:] # # (block_num, block_size, head_dim)
        else:
            raise ValueError("Unsupported shape for ram cache")

        if len(shape) != 3:
            raise ValueError("Unsupported shape for ram cache")

        # KV cache tensor shapes per layer: [block_num, block_size, dimension]
        if model_extra_config.operator_opt_config.enable_dsa:
            self.shapes = [
                (*shape[:-1], shape[-1] * self.head_size_ratio[0] // self.head_total_slices),  # Nope tensor: [block, block_size, nope_dim]
                (*shape[:-1], shape[-1] * self.head_size_ratio[1] // self.head_total_slices),  # PE tensor: [block, block_size, pe_dim]
                (*shape[:-1], shape[-1] * self.head_size_ratio[2] // self.head_total_slices)   # Indexer tensor: [block, block_size, indexer_dim]
            ]
        else:
            self.shapes = [
                (*shape[:-1], shape[-1] * self.head_size_ratio[0] // self.head_total_slices),  # Nope tensor: [block, block_size, nope_dim]
                (*shape[:-1], shape[-1] * self.head_size_ratio[1] // self.head_total_slices)  # PE tensor: [block, block_size, pe_dim]
            ]

        # Calculate sizes and offsets
        self.sizes = []  # Number of elements per tensor
        self.offsets = []  # Byte offsets within block
        self.layer_sizes = []
        self.block_size_byte_list = []
        current_offset = 0

        # (nope shape, pe shape, indexer shape)
        for shape in self.shapes:
            # (block, block_size, dim)
            # layer_elements = shape[1] * shape[2]
            block_elements = shape[1] * shape[2] # block_size * dim
            num_elements = shape[0] * block_elements # block * block_size * dim, one layer
            layer_elements = num_elements

            self.block_size_byte_list.append(num_elements)
            self.layer_sizes.append(layer_elements)
            self.sizes.append(num_elements) # Note: same as layer_sizes
            self.offsets.append(current_offset)
            current_offset += num_elements * self.element_size

        # self.block_size_bytes = current_offset
        self.layer_size_bytes = current_offset
        self.block_size_bytes = self.layer_size_bytes // self.num_blocks * self.num_layers
        self.size_total = self.num_layers * self.layer_size_bytes
        self.aligned_rank_size = (self.size_total // HUGE_PAGE_SIZE) * HUGE_PAGE_SIZE

        if self.num_blocks == 0:
            raise ValueError(f"mmap_size {mmap_size} too small for block size {self.block_size_bytes}")

        if int(os.getenv("ENABLE_HOST_MAPPING", "0")):
            self.npu_tensor_register = NPUTensorRegister()
        else:
            self.npu_tensor_register = None

        # Map hugepage shared memory using mmap
        self._map_hugepage_memory(hugepage_path, mmap_size, rank)

        self.ascend_cl_stream = AscendCLStream()

        print(f"KV Cache Pool: {self.num_blocks} blocks, {self.block_size_bytes} bytes/block")

    def _map_hugepage_memory(self, hugepage_path: str, mmap_size: int, rank: int = 0) -> None:
        """Map hugepage shared memory file using mmap for shared access"""
        if not os.path.exists(hugepage_path):
            raise FileNotFoundError(f"Hugepage file not found: {hugepage_path}")

        # Open hugepage file and create memory mapping
        self.fd = os.open(hugepage_path, os.O_RDWR)
        self.mmap_obj = mmap.mmap(self.fd, mmap_size, access=mmap.ACCESS_WRITE)

        # mlock the memory using ctypes
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            buf = (ctypes.c_char * mmap_size).from_buffer(self.mmap_obj)
            result = libc.mlock(ctypes.byref(buf), mmap_size)
            if result != 0:
                errno = ctypes.get_errno()
                logger.warning(f"Warning: OmniCache mlock failed with error: {errno}")
            else:
                logger.info(f"OmniCache mlock {hugepage_path} succeeded")
        except Exception as e:
            logger.warning(f"Warning: OmniCache mlock failed: {e}")

        offset_bytes = rank * self.size_total
        buf_view = memoryview(self.mmap_obj)[offset_bytes:]

        if self.num_ranks == 1:
            self.shared_tensor = torch.frombuffer(
                self.mmap_obj,
                dtype=self.dtype,
                count=self.size_total // self.element_size
            )
        else:
            self.shared_tensor = torch.frombuffer(
                buf_view,
                dtype=self.dtype,
                count=self.size_total // self.element_size
            )

        if self.is_prefill:
            cnt = self.size_total // self.element_size
            if cnt % self.head_total_slices != 0:
                raise ValueError(f"Expected total elements divisible by {self.head_total_slices}, got {cnt}.")
            split_sizes = [cnt * r // self.head_total_slices for r in self.head_size_ratio]
            self.kvi_tensors = list(torch.split(self.shared_tensor, split_sizes))
            for i, (tensor, shape) in enumerate(zip(self.kvi_tensors, self.shapes)):
                self.kvi_tensors[i] = tensor.view([self.num_layers] + list(shape))
        else:
            logger.warning(f"<<< {model_extra_config.operator_opt_config.enable_dsa=}")
            if int(os.getenv("ENABLE_HOST_MAPPING", "0")) and model_extra_config.operator_opt_config.enable_dsa:
                logger.warning(f"<<< {model_extra_config.operator_opt_config.enable_dsa=}, before host_swap_device")
                ptr = self.shared_tensor.data_ptr()
                assert ptr % REGIST_SIZE == 0, f"Address 0x{ptr:x} not 4K aligned!"
                self.shared_tensor_swap = self.host_swap_device(self.shared_tensor)
            else:
                self.shared_tensor_swap = self.shared_tensor
            cnt = self.size_total // self.element_size
            if cnt % self.head_total_slices != 0:
                raise ValueError(f"Expected total elements divisible by {self.head_total_slices}, got {cnt}.")
            split_sizes = [cnt * r // self.head_total_slices for r in self.head_size_ratio]
            self.kvi_tensors = list(torch.split(self.shared_tensor, split_sizes))
            for i, (tensor, shape) in enumerate(zip(self.kvi_tensors, self.shapes)):
                self.kvi_tensors[i] = tensor.view([self.num_layers] + list(shape))
            self.kvi_tensors_swap = list(torch.split(self.shared_tensor_swap, split_sizes))
            for i, (tensor, shape) in enumerate(zip(self.kvi_tensors_swap, self.shapes)):
                self.kvi_tensors_swap[i] = tensor.view([self.num_layers] + list(shape))
            shared_tensor_list = []
            for kvi_tensor in self.kvi_tensors_swap:
                tensor_list_swap = list(kvi_tensor.unsqueeze(-2).unbind(dim=0))
                shared_tensor_list.append(tensor_list_swap)
            self.shared_tensor_npu = [tuple(l[i] for l in shared_tensor_list) for i in range(self.num_layers)]

    def host_swap_device(self, strided_tensor):
        device_id = self.device.index

        logger.warning(f"<<< host_swap_device rank{self.rank}] before register: "
            f"ptr=0x{strided_tensor.data_ptr():x}, "
            f"size={strided_tensor.nbytes}, "
            f"2M-align={strided_tensor.data_ptr() % (2<<20) == 0}, "
            f"4K-align={strided_tensor.data_ptr() % (4096) == 0}, "
            f"is_pinned={strided_tensor.is_pinned()}")

        _, tensor_swap = self.npu_tensor_register.host_tensor_register(strided_tensor, device_id=device_id)
        return tensor_swap

    def get_block(self, block_idx: int) -> List[torch.Tensor]:
        """Return all-layer tensors for the specified block index."""
        if block_idx >= self.num_blocks:
            raise IndexError(f"Block index {block_idx} out of range (max {self.num_blocks - 1})")

        block_tensors = []
        for i, kvi_tensor in enumerate(self.kvi_tensors):
            if int(os.getenv("ENABLE_HOST_MAPPING", "0")) and i < 2 and not self.is_prefill and model_extra_config.operator_opt_config.enable_dsa:
                continue
            # Each tensor shape: [num_layer, num_blocks, block_size, dim]
            # block_view = kvi_tensor[:, block_idx, ...].contiguous()
            block_view = kvi_tensor[:, block_idx, ...]
            block_tensors.append(block_view)

        return block_tensors

    def batch_layer_copy_to_npu(self, block_ids: list, npu_blocks, layer_indices: Optional[List[int]] = None):
        """Batch copy specific layers of multiple blocks from CPU shared memory to NPU, merging consecutive blocks with continuous addresses (for batch copy)."""

        cpu_blocks = [self.get_block(block_id) for block_id in block_ids]
        layers = len(npu_blocks[0]) # npu_blocks layout: [block][layer][kvi_tensor]

        batch_device_mem = []
        batch_device_max = []
        batch_host_mem = []
        batch_host_sizes = []

        for kvi_idx in range(len(cpu_blocks[0])): # cpu_blocks layout: [block][kvi_tensor][layer]
            tensor_size = cpu_blocks[0][kvi_idx][0].nbytes
            for layer_idx in range(layers):
                cpu_layer_idx = layer_idx if layer_indices is None else layer_indices[layer_idx]
                cpu_addrs = [cpu_blocks[i][kvi_idx][cpu_layer_idx].unsqueeze(-2).data_ptr() for i in range(len(block_ids))]
                npu_addrs = [npu_blocks[i][layer_idx][kvi_idx].data_ptr() for i in range(len(block_ids))]
                batch_start = 0
                while batch_start < len(block_ids):
                    batch_end = batch_start + 1
                    prev_cpu_addr = cpu_addrs[batch_start]
                    prev_npu_addr = npu_addrs[batch_start]
                    while batch_end < len(block_ids):
                        curr_cpu_addr = cpu_addrs[batch_end]
                        curr_npu_addr = npu_addrs[batch_end]
                        if (curr_cpu_addr == prev_cpu_addr + tensor_size and
                            curr_npu_addr == prev_npu_addr + tensor_size):
                            prev_cpu_addr = curr_cpu_addr
                            prev_npu_addr = curr_npu_addr
                            batch_end += 1
                        else:
                            break
                    count_blocks = batch_end - batch_start
                    batch_device_mem.append(npu_addrs[batch_start])
                    batch_device_max.append(tensor_size * count_blocks)
                    batch_host_mem.append(cpu_addrs[batch_start])
                    batch_host_sizes.append(tensor_size * count_blocks)
                    batch_start = batch_end
        return batch_device_mem, batch_device_max, batch_host_mem, batch_host_sizes

    def memcpy_async(self, batch_device_mem, batch_device_max, batch_host_mem, batch_host_sizes):

        start_time = time.time()

        batch_count = len(batch_device_mem)

        for idx in range(len(batch_device_mem)):
            device_mem = ctypes.c_void_p(batch_device_mem[idx])
            device_max = batch_device_max[idx]
            host_mem = ctypes.c_void_p(batch_host_mem[idx])
            host_sizes = batch_host_sizes[idx]

            rc = self.ascend_cl_stream.memcpy_async(
                device_mem,
                device_max,
                host_mem,
                host_sizes,
                1
            )

        mb_copied = sum(batch_host_sizes) >> 20
        self.ascend_cl_stream.sync()
        duration = time.time() - start_time
        logger.warning(f"Batch (merged) {batch_count} copy {mb_copied} MB took {duration * 1000:.2f} ms")

    def set_block(self, block_idx: int, tensors: List[torch.Tensor]) -> None:
        """Set key, value, metadata tensors for the specified block index."""
        expected_len = 3 if model_extra_config.operator_opt_config.enable_dsa else 2
        if len(tensors) != expected_len:
            raise ValueError(f"Expected {expected_len} tensors, got {len(tensors)}")

        for i, (tensor, expected_shape) in enumerate(zip(tensors, self.shapes)):
            # Expect full-layer tensor: [num_layer, block_size, dim]
            if tensor.shape != (self.num_layers, *expected_shape[1:]):
                raise ValueError(
                    f"Tensor {i} shape mismatch: got {tensor.shape}, "
                    f"expected {(self.num_layers, *expected_shape[1:])}"
                )
            if tensor.dtype != self.dtype:
                raise ValueError(f"Tensor {i} dtype mismatch: {tensor.dtype} vs {self.dtype}")

        for i, (kvi_tensor, tensor) in enumerate(zip(self.kvi_tensors, tensors)):
            # Each kvi_tensor: [num_layer, num_blocks, block_size, dim]
            if block_idx >= kvi_tensor.shape[1]:
                raise IndexError(f"Block {block_idx} out of range for tensor {i}")
            kvi_tensor[:, block_idx, ...].copy_(tensor, non_blocking=True)

    def __getitem__(self, block_idx: int) -> List[torch.Tensor]:
        """pool[block_idx] returns [key, value, metadata] tensors"""
        return self.get_block(block_idx)

    def __setitem__(self, block_idx: int, tensors: List[torch.Tensor]) -> None:
        """pool[block_idx] = [key_tensor, value_tensor, metadata_tensor]"""
        self.set_block(block_idx, tensors)

    def close(self) -> None:
        """Close mmap and file descriptor"""
        if hasattr(self, 'mmap_obj'):
            self.mmap_obj.close()
        if hasattr(self, 'fd'):
            os.close(self.fd)

    def __del__(self):
        """Ensure resources are cleaned up"""
        self.close()

    def info(self) -> None:
        """Print memory pool configuration"""
        print(f"\n=== KV Cache Memory Pool ===")
        print(f"Data type: {self.dtype}")
        print(f"Blocks: {self.num_blocks}")
        print(f"Block size: {self.block_size_bytes} bytes")
        print(f"Total memory: {self.mmap_size} bytes")
        print(f"Tensor shapes per block:")
        print(f"  Nope KV: {self.shapes[0]} (layers, block_size, nope_dim=512)")
        print(f"  PE KV: {self.shapes[1]} (layers, block_size, pe_dim=64)")
        if model_extra_config.operator_opt_config.enable_dsa:
            print(f"  Indexer: {self.shapes[2]} (layers, block_size, indexer_dim=128)")