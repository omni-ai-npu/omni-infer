from dataclasses import dataclass
import math
import os
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional, List
import threading
import numpy as np
import torch
import torch_npu
from vllm.distributed.parallel_state import get_tp_group, get_dp_group, get_world_group
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import cdiv
from vllm.v1.kv_cache_interface import KVCacheConfig, AttentionSpec
from vllm.v1.utils import bind_kv_cache
from omni.adaptors.vllm.worker.npu_model_runner import NPUModelRunner
from vllm.model_executor.models.utils import extract_layer_index
from omni.layers.attention.backend.attention import AscendAttentionState
from omni.models.config_loader.loader import model_extra_config
import ctypes
from ctypes import pythonapi, py_object
from omni.accelerators.cache.kv_mem_pool import KVCacheMemoryPool
from concurrent.futures import ThreadPoolExecutor, Future


logger = init_logger("vllm.v1.omni")

hardware_platform = "A2" if torch_npu.npu.get_device_name(0) == "Ascend910B" else "A3"
NUM_DIE_PER_MACH = torch.npu.device_count() # 16 if hardware_platform == "A3" else 8 # get device count per using torch-npu API
NZ_DIM = 16                                     # nz dim

dump_data = False

SCALE_RATIO = int(os.getenv("DEVICE_CACHE_SCALE_RATIO", "1"))
D2H_CHUNK_SIZE = int(os.getenv("D2H_CHUNK_SIZE", "512")) # CHUNK_SIZE for D2H offloading when seq is long
SIZE_BYTES_PER_LAYER = int(os.getenv("SIZE_BYTES_PER_LAYER", "2147483648")) # 2GB per layer, which is calculated by 2 * 1024 * 1024 * 1024. Adjust this number if the actual per layer size is different, e.g., for models with larger head size.


# Load ACL library and define structures (should be done at module level)
try:
    libascendcl = ctypes.CDLL('/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64/libascendcl.so')

    # Define enums and structures
    ACL_MEM_LOCATION_TYPE_HOST = 0
    ACL_MEM_LOCATION_TYPE_DEVICE = 1

    class aclrtMemLocation(ctypes.Structure):
        _fields_ = [
            ('id', ctypes.c_uint32),
            ('type', ctypes.c_uint32)
        ]

    class aclrtMemcpyBatchAttr(ctypes.Structure):
        _fields_ = [
            ('dstLoc', aclrtMemLocation),
            ('srcLoc', aclrtMemLocation),
            ('rsv', ctypes.c_uint8 * 16)
        ]

    # Set function prototype
    aclrtMemcpyBatch = libascendcl.aclrtMemcpyBatch
    aclrtMemcpyBatch.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.POINTER(aclrtMemcpyBatchAttr),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t)
    ]
    aclrtMemcpyBatch.restype = ctypes.c_int

    ACL_BATCH_COPY_AVAILABLE = True
    logger.info("ACL batch copy library loaded successfully")
except Exception as e:
    logger.warning(f"ACL batch copy not available: {e}")
    ACL_BATCH_COPY_AVAILABLE = False

class BaseOmniCache(ABC):
    MEMMAP_PATH = os.getenv("OMNI_CACHE_MEMMAP_PATH", "/dev/hugepages/omni_cache")

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        runner: NPUModelRunner
    ):
        self.tp_rank = get_tp_group().rank
        self.tp_local_rank = get_tp_group().local_rank
        self.tp_world_size = get_tp_group().world_size
        self.dp_local_rank = get_dp_group().local_rank
        self.dp_world_size = get_dp_group().world_size
        self.dp_rank = get_dp_group().rank
        self.dp_world_size_local = NUM_DIE_PER_MACH
        self.device = runner.device

        attn_spec: AttentionSpec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
        self.num_layers = sum([len(kv_cache_group.layer_names) for kv_cache_group in kv_cache_config.kv_cache_groups])
        self.block_size = attn_spec.block_size
        if model_extra_config.operator_opt_config.enable_dsa:
            head_sizes = [512, 64, 128]
        elif attn_spec.use_mla:
            head_sizes = [512, 64]
        else:
            head_sizes = [attn_spec.head_size, attn_spec.head_size]
        self.head_sizes = [D * attn_spec.num_kv_heads for D in head_sizes]  # handle GQA and MLA with the same logic, i.e., no heads
        self.head_size = sum(self.head_sizes)
        self.dtype = attn_spec.dtype

        # Calculate shape and number of blocks
        self.shape, self.num_blocks = self.calc_cache_shape()

        logger.warning(f"**BaseOmniCache**: {self.shape=}, {self.tp_world_size=}")
        self.host_cache = self.initialize_shared_memory()

        # pass the ascend_cl_stream from KVMemoryPool to host_cache in OmniCache, so that other processes can call it
        self.ascend_cl_stream = self.host_cache.ascend_cl_stream

        self.host_swap_tensor = self.host_cache.shared_tensor_npu
        # block_len_dtype: how many elements of `dtype` in one block
        # dp_offset: how many blocks to start from for current rank.
        self.block_len_dtype, self.dp_offset = self.calculate_kv_xsfer_params()
        self.device_cache = self.initialize_device_cache(kv_cache_config, runner)

    @abstractmethod
    def calc_cache_shape(self) -> Tuple[Tuple[int, ...], int]:
        pass

    @abstractmethod
    def calculate_kv_xsfer_params(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def initialize_device_cache(
        self,
        kv_cache_config: KVCacheConfig,
        runner: NPUModelRunner
    ):
        pass

    def _log_and_validate_kv_blocks(
        self,
        kv_cache_config: KVCacheConfig,
        device_cache_blocks: Optional[int] = None
    ) -> None:
        """
        Log all KV block counts and validate that OmniCache allocated blocks
        are sufficient for vLLM KV manager requirements.

        Args:
            kv_cache_config: The KV cache configuration from vLLM, containing
                num_blocks which represents the number of GPU blocks vLLM needs.
            device_cache_blocks: Optional device cache block count for PrefillOmniCache.
                If provided, both host and device cache will be validated.

        Raises:
            ValueError: If host or device cache blocks are insufficient for vLLM
                requirements, with detailed guidance on how to fix the issue.

        Example:
            Configuration example with typical parameters:
            - SIZE_BYTES_PER_LAYER = 6 * 1024 * 1024 * 1024 (6GB per layer)
            - SCALE_RATIO = 2 (device to host ratio)
            - MAX_MODEL_LEN = 202752
            - max_num_seqs = 16
            - KV_DIM = 704, KV dtype = bf16 (2 bytes)
            - block_size = 128, tp = 1, node_block_size = 128

            Host KV blocks calculation:
                SIZE_BYTES_PER_LAYER // (KV_DIM * 2 * node_block_size)
                  = (6 * 1024 * 1024 * 1024) // (704 * 2 * 128) = 35756 blocks

            Device KV blocks calculation:
                (MAX_MODEL_LEN * max_num_seqs * 2) // block_size + 4
                  = (202752 * 2 * 16) // 128 + 4 = 50692 blocks
        """
        # Validate that total memory requirement fits within MAP_SIZE_BYTES
        self._validate_map_size()

        vllm_num_blocks = kv_cache_config.num_blocks
        host_cache_blocks = self.num_blocks

        # Log configuration parameters for debugging
        logger.debug(
            f"KV cache validation input: vllm_num_blocks={vllm_num_blocks}, "
            f"host_cache_blocks={host_cache_blocks}, "
            f"device_cache_blocks={device_cache_blocks}"
        )

        # Unified Info logging for all KV block counts
        block_info = [
            f"vllm required num_gpu_blocks={vllm_num_blocks}",
            f"host allocated kv_cache_blocks={host_cache_blocks}",
        ]
        if device_cache_blocks is not None:
            block_info.append(f"device allocated kv_cache_blocks={device_cache_blocks}")
        logger.info(f"OmniCache KV Blocks allocation: {', '.join(block_info)}")

        # Validate host cache block count
        host_deficit = vllm_num_blocks - host_cache_blocks
        if host_cache_blocks < vllm_num_blocks:
            error_msg = (
                f"Insufficient host KV cache blocks: "
                f"allocated {host_cache_blocks}, but vLLM requires {vllm_num_blocks} "
                f"(deficit: {host_deficit} blocks).\n"
                f"Suggested fixes:\n"
                f"  1. Increase SIZE_BYTES_PER_LAYER environment variable\n"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Validate device cache block count (if applicable)
        if device_cache_blocks is not None:
            device_deficit = vllm_num_blocks - device_cache_blocks
            if device_cache_blocks < vllm_num_blocks:
                error_msg = (
                    f"Insufficient device KV cache blocks: "
                    f"allocated {device_cache_blocks}, but vLLM requires {vllm_num_blocks} "
                    f"(deficit: {device_deficit} blocks).\n"
                    f"Suggested fixes:\n"
                    f"  1. Increase SCALE_RATIO environment variable for larger device cache\n"
                    f"  2. Increase max_num_reqs to allocate more device blocks\n"
                    f"  3. Reduce MAX_MODEL_LEN or max_num_seqs to decrease vLLM block requirement"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

        # Log success with margin information
        host_margin = host_cache_blocks - vllm_num_blocks
        margin_info = [f"host margin={host_margin} blocks ({host_margin / vllm_num_blocks * 100:.1f}%)"]
        if device_cache_blocks is not None:
            device_margin = device_cache_blocks - vllm_num_blocks
            margin_info.append(f"device margin={device_margin} blocks ({device_margin / vllm_num_blocks * 100:.1f}%)")
        logger.info(f"OmniCache KV Blocks validation passed: {', '.join(margin_info)}")

    def _validate_kv_blocks(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Validate that the number of kv_blocks allocated by OmniCache
        is greater than or equal to the number managed by vLLM KV manager.

        Args:
            kv_cache_config: The KV cache configuration from vLLM

        Raises:
            ValueError: If the validation fails
        """
        # For base class, only validate host cache (device cache handled in subclass)
        self._log_and_validate_kv_blocks(kv_cache_config, device_cache_blocks=None)

    def _validate_map_size(self) -> None:
        """
        Validate that SIZE_BYTES_PER_LAYER * num_layers does not exceed MAP_SIZE_BYTES.

        Raises:
            ValueError: If the total memory requirement exceeds the mapped memory size.
        """
        map_size_bytes = int(os.environ.get("MAP_SIZE_BYTES", "214748364800"))   # Default 200GB (1<<39)
        required_size = SIZE_BYTES_PER_LAYER * self.num_layers

        if required_size > map_size_bytes:
            error_msg = (
                f"Insufficient MAP_SIZE_BYTES for KV cache: "
                f"required {required_size} bytes ({required_size / 1024**3:.2f} GB), "
                f"but MAP_SIZE_BYTES is {map_size_bytes} bytes ({map_size_bytes / 1024**3:.2f} GB). "
                f"SIZE_BYTES_PER_LAYER={SIZE_BYTES_PER_LAYER} * num_layers={self.num_layers} = {required_size}.\n"
                f"Suggested fixes:\n"
                f"  1. Increase MAP_SIZE_BYTES environment variable\n"
                f"  2. Decrease SIZE_BYTES_PER_LAYER environment variable"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(
            f"MAP_SIZE validation passed: required {required_size / 1024**3:.2f} GB, "
            f"available {map_size_bytes / 1024**3:.2f} GB "
            f"(margin: {(map_size_bytes - required_size) / 1024**3:.2f} GB)"
        )

    def initialize_shared_memory(self) -> KVCacheMemoryPool:
        total_numel = math.prod(self.shape)
        itemsize = self.dtype.itemsize

        return KVCacheMemoryPool(BaseOmniCache.MEMMAP_PATH, total_numel * itemsize, self.shape, self.dp_local_rank, self.device)

    def __getitem__(self, index: int):
        return self.host_cache

    @abstractmethod
    def synchronize_h2d(self) -> None:
        pass

    @abstractmethod
    def synchronize_d2h(self, key_states: torch.Tensor, value_states: torch.Tensor, slot_mapping: torch.Tensor, layer_idx: int, kv_event: torch.npu.Event) -> None:
        pass


class PrefillOmniCache(BaseOmniCache):
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        runner: NPUModelRunner,
        max_num_batched_tokens: int,
        max_num_seqs: int,
        max_model_len: int,
    ):
        super().__init__(kv_cache_config, runner)
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.arange_cpu = torch.arange(max_model_len, device='cpu', dtype=torch.int64)
        self.arange = self.arange_cpu.to(device=self.device)

        self.batch_buffer_cpu = [
            torch.empty(
                (max_num_batched_tokens * 2) // self.block_size + 1,
                self.block_size // self.tp_world_size,
                D,
                dtype=self.dtype,
                device='cpu',
                pin_memory=True,
            ) for D in self.head_sizes
        ]
        shape_msg = ", ".join(map(str, [tensor.shape for tensor in self.batch_buffer_cpu]))
        logger.warning(f"**PrefillOmniCache**: CPU buffer shape is {shape_msg}.")

        self.batch_slots_cpu = torch.zeros(
            max_num_batched_tokens * 2 // self.tp_world_size,
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )
        self.batch_token_indices = None
        self.d2h_stream = torch.npu.Stream(device=self.device)
        self.d2h_thrp = ThreadPoolExecutor(max_workers=1, thread_name_prefix="D2H_Worker")
        self.copy_future: Future = None

        # buffer for prefix/chunk
        # layout: TND, where T is max possible total KV tokens for a batch
        self.prefix_buffer_npu = torch.empty(
            max_num_seqs * max_model_len,
            1, # num of heads is 1
            self.head_size,
            dtype=self.dtype,
            device=self.device,
        )
        self.h2d_stream = torch.npu.Stream(device=self.device)
        self.h2d_event = torch.npu.Event(blocking=False, enable_timing=False)

        self._nz_size = NZ_DIM

    # def calc_cache_shape(self) -> Tuple[Tuple[int, ...], int]:
    #     self.tp_node_id = self.tp_rank // NUM_DIE_PER_MACH
    #     self.tp_nnodes = divide_or_raise(self.tp_world_size, NUM_DIE_PER_MACH)

    #     # For prefill, each node only needs to store a segment of KV cache.
    #     # For example, if head_size = 576, and TP is across 2 nodes, then
    #     # node0 stores [0, 288) and node1 stores [288, 576).
    #     self.local_head_size = divide_or_raise(self.head_size, self.tp_nnodes)
    #     shape, num_blocks = PrefillOmniCache.calc_cache_shape_for_prefill(
    #         num_layers=self.num_layers,
    #         block_size=self.block_size,
    #         num_kv_heads=self.num_kv_heads,
    #         head_size=self.local_head_size,
    #         dtype=self.dtype,
    #     )
    #     total_num_nz_heads = shape[-1]                              # e.g., 36
    #     rank_num_nz_heads = total_num_nz_heads // NUM_DIE_PER_MACH  # 2
    #     remainder = total_num_nz_heads % NUM_DIE_PER_MACH           # 4

    #     # [3, 3, 3, 3, 2, 2, ...]
    #     get_rank_heads = lambda rank: rank_num_nz_heads + 1 if rank < remainder else rank_num_nz_heads
    #     starts = [0]
    #     for i in range(NUM_DIE_PER_MACH):
    #         starts.append(starts[-1] + get_rank_heads(i))
    #     assert starts[-1] == total_num_nz_heads, f"{total_num_nz_heads=}, while {starts=}."

    #     # how many nz heads the current rank is responsible to copy  // 512/64/128 --> // 32
    #     self.nz_heads_slc = slice(starts[self.tp_local_rank], starts[self.tp_local_rank+1])
    #     logger.warning(f"<<< {starts=}, {self.nz_heads_slc}")
    #     self.num_nz_heads = self.nz_heads_slc.stop - self.nz_heads_slc.start

    #     return shape, num_blocks

    def calc_cache_shape(self) -> Tuple[Tuple[int, ...], int]:
        self.tp_node_id = self.tp_rank // NUM_DIE_PER_MACH
        self.tp_nnodes = divide_or_raise(self.tp_world_size, NUM_DIE_PER_MACH)
        self.node_block_size = divide_or_raise(self.block_size, self.tp_nnodes)
        self.rank_block_size = divide_or_raise(self.block_size, self.tp_world_size)

        shape, num_blocks = PrefillOmniCache.calc_cache_shape_for_prefill(
            num_layers=self.num_layers,
            block_size=self.node_block_size,
            head_size=self.head_size,
            dtype=self.dtype,
        )

        return shape, num_blocks

    @staticmethod
    def calc_cache_shape_for_prefill(
        num_layers: int,
        block_size: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[Tuple[int, ...], int]:
        itemize = dtype.itemsize
        numel_per_layer = divide_or_raise(SIZE_BYTES_PER_LAYER, itemize)
        numel_per_block = block_size * head_size
        num_blocks_prefill = numel_per_layer // numel_per_block  # floor division

        p_shape = (
            num_layers,
            num_blocks_prefill,
            block_size,
            head_size
        )

        return p_shape, num_blocks_prefill

    def calculate_kv_xsfer_params(self) -> Tuple[int, int]:
        block_len_dtype = math.prod(self.shape[2:]) * self.shape[0]
        dp_offset = 0
        return block_len_dtype, dp_offset

    def initialize_device_cache(
        self,
        kv_cache_config: KVCacheConfig,
        runner: NPUModelRunner
    ) -> Optional[Tuple[torch.Tensor]]:

        # For DSV3.2, create a volatile KV cache and block_table on device
        max_num_blocks_per_req = cdiv(runner.max_model_len, self.block_size)
        num_blocks = runner.max_num_reqs * (SCALE_RATIO * max_num_blocks_per_req)
        device_cache_blocks = num_blocks + 4

        # Validate all KV block counts (host and device) with unified logging
        self._log_and_validate_kv_blocks(kv_cache_config, device_cache_blocks=device_cache_blocks)

        volatile_table = torch.arange(
            1, 1 + num_blocks,
            dtype=torch.int32,
            device=runner.device).view(runner.max_num_reqs, SCALE_RATIO * max_num_blocks_per_req)

        kv_cache_spec: AttentionSpec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
        kv_cache_shape = runner.attn_backends[0].get_kv_cache_shape(
            num_blocks + 4,  # avoid overflowing
            kv_cache_spec.block_size,
            kv_cache_spec.num_kv_heads,
            kv_cache_spec.head_size
        )
        volatile_cache = runner.attn_backends[0].init_kv_cache_each_layer(
            kv_cache_shape,
            runner.dtype,
            runner.device,
            runner.model_config,
            runner.enable_torchair_graph_mode
        )

        if not isinstance(volatile_cache, tuple):
            raise RuntimeError(f"The KV cache should be a tuple, but got {type(volatile_cache)}.")
        if not all(isinstance(t, torch.Tensor) for t in volatile_cache):
            raise RuntimeError("Error! All elements in volatile cache should be tensors.")
        if model_extra_config.operator_opt_config.enable_indexer_quant:
            expected_len = 4 if model_extra_config.operator_opt_config.enable_dsa else 2
        else:
            expected_len = 3 if model_extra_config.operator_opt_config.enable_dsa else 2
        if len(volatile_cache) != expected_len:
            raise RuntimeError(f"There should be {expected_len} tensors in KV cache, but got {len(volatile_cache)}.")

        bytes_mb = sum([t.nbytes for t in volatile_cache]) / 1024**2
        logger.warning(f"**PrefillOmniCache**: {num_blocks} blocks are allocated for volatile KV cache which consumes {bytes_mb:.2f} MB.")

        self.volatile_table = volatile_table

        logger.info(f"When using OmniCache to initialize device cache, the kv num_blocks is {num_blocks}")
        logger.info(f"When using OmniCache to initialize device cache, the shape of KV cache is [{num_blocks + 4}, {kv_cache_spec.block_size}, {kv_cache_spec.num_kv_heads}, {kv_cache_spec.head_size}].")

        return volatile_cache

    def get_prefill_prefix_copy_meta(
        self,
        block_size,
        kv_lens: np.ndarray,
        query_lens_list: list[int],
        block_tables: np.ndarray,
        attn_state: AscendAttentionState,
        slot_mapping: torch.Tensor
    ) -> Optional["PrefixCopyMeta"]:
        if attn_state != AscendAttentionState.ChunkedPrefill:
            return None

        bsz = block_tables.shape[0]
        all_segs, q_slots, q_slot_start = [], [], 0
        last_block_idx, remainder = (kv_lens-1) // block_size, kv_lens % block_size
        assert np.all(remainder == 0), f"For APC, remainder should be zeros, but {kv_lens=}."

        for i in range(bsz):
            m = last_block_idx[i]

            if m < 0:
                segs = []
            elif m == 0:
                single_block = block_tables[i, 0].item()
                segs = [[single_block, single_block+1]]
            else:
                bt = block_tables[i, :m+1]
                # bt[idx] - bt[idx-1] != 1
                split_idx = np.where(np.diff(bt, n=1, axis=0) != 1)[0] + 1
                start_idx = np.r_[0, split_idx]
                end_idx = np.r_[split_idx - 1, m]

                # consecutive blocks [start_blocks[j], start_blocks[j]+1, ..., end_blocks[j]-1] are occupied
                start_blocks = bt[start_idx]
                end_blocks = bt[end_idx] + 1  # exclusive
                segs = np.stack([start_blocks, end_blocks], axis=1).tolist()  # (N_seg, 2)

            all_segs.append(segs)
            q_slot_start += kv_lens[i]
            q_slots.append(self.arange_cpu[:query_lens_list[i]] + q_slot_start)
            q_slot_start += query_lens_list[i]

        q_slots = torch.cat(q_slots).to(device=self.device, non_blocking=True)
        prefix_meta = PrefixCopyMeta(consecutive_blocks=all_segs,
                                     query_lens=query_lens_list,
                                     query_slots=slot_mapping)
                                    #  query_slots=q_slots)
        return prefix_meta

    def get_volatile_metadata(
        self,
        query_lens_list: list[int],
        seq_lens_list: list[int],
        graph_pad_size: int,
        pad_slot_id: int,
        orig_slot_mapping: torch.Tensor,
    ):
        orig_shape = orig_slot_mapping.shape
        # NOTE: kill all items about updating slot mapping, as we do not use the volatile things
        # if model_extra_config.parall_config.attn_sp_size > 1:
        #     orig_slot_mapping = pad_inputs(
        #         orig_slot_mapping,
        #         query_lens_list,
        #         model_extra_config.parall_config.attn_sp_size * 2,
        #         pad_slot_id
        #     )
        # else:
        #     orig_slot_mapping = generate_full_block_slot(
        #         orig_slot_mapping,
        #         query_lens_list,
        #         self.block_size
        #     )
        query_lens_tensor = torch.tensor(query_lens_list)
        padding_lens_tensor = (self.block_size - query_lens_tensor % self.block_size) % self.block_size
        total_lens_list = (query_lens_tensor + padding_lens_tensor).tolist()
        self.sum_total_len = sum(total_lens_list)
        self.sum_query_len = sum(query_lens_list)
        self.query_mask = torch.zeros(self.sum_total_len, dtype = torch.bool)
        start = 0
        for ql, tl in zip(query_lens_list, total_lens_list):
            self.query_mask[start:start+ql] = 1
            start += tl

        blocks, slots = orig_slot_mapping // self.block_size, orig_slot_mapping % self.block_size
        ranks, rank_slots = slots // self.rank_block_size, slots % self.node_block_size
        token_idx = torch.nonzero((ranks == self.tp_rank) & (orig_slot_mapping > 0), as_tuple=True)[0]
        slot_mapping = blocks[token_idx] * self.node_block_size + rank_slots[token_idx]
        num_tokens = token_idx.shape[0]

        self.batch_slots_cpu[:num_tokens].copy_(slot_mapping, non_blocking=True)
        self.batch_token_indices = token_idx

        if not model_extra_config.operator_opt_config.enable_dsa:
            return None, None

        max_num_blocks_per_req = self.volatile_table.shape[1]
        slot_mapping = []
        for i, (q_len, seq_len) in enumerate(zip(query_lens_list, seq_lens_list)):
            start = (i * max_num_blocks_per_req + 1) * self.block_size
            slot_mapping.append(self.arange[seq_len-q_len:seq_len] + start)
        if graph_pad_size > 0:
            padding = torch.full((graph_pad_size,),
                                 fill_value=pad_slot_id,
                                 dtype=self.arange.dtype,
                                 device=self.arange.device)
            slot_mapping.append(padding)
        slot_mapping = torch.cat(slot_mapping, dim=0)

        if slot_mapping.shape != orig_shape:
            raise RuntimeError(f"Slot mapping shape mismatch! {slot_mapping.shape=}, {orig_shape=}.")
        return self.volatile_table, slot_mapping

    def get_current_rank_host_data(self, layer_idx, prefix_meta):
        # flatten
        block_ids = []
        for block_ranges in prefix_meta.consecutive_blocks:
            for start_block_id, end_block_id in block_ranges:
                block_ids.extend(range(start_block_id, end_block_id))

        num_heads_per_node = len(block_ids) * self.node_block_size
        num_heads_per_rank = num_heads_per_node // NUM_DIE_PER_MACH

        # Get block_ids of current rank
        current_rank_block_ids_start_idx = self.tp_local_rank * num_heads_per_rank // self.node_block_size
        current_rank_block_ids_end_idx = (self.tp_local_rank + 1) * num_heads_per_rank // self.node_block_size
        block_ids_of_this_rank = block_ids[current_rank_block_ids_start_idx:current_rank_block_ids_end_idx + 1]

        offset_in_block = self.tp_local_rank * num_heads_per_rank % self.node_block_size
        block_id = block_ids_of_this_rank[0]
        head_offset = block_id * self.node_block_size + offset_in_block

        ## Dram.contiguous()
        remain_heads = num_heads_per_rank
        num_copyed_heads = min(remain_heads, self.node_block_size - offset_in_block)
        remain_heads -= num_copyed_heads

        kvi_tensors = []
        host_data = []
        for i, tensor in enumerate(self.host_cache.kvi_tensors):
            host_data.append([])
            kvi_tensors.append(tensor.view(self.num_layers, -1, tensor.shape[-1]))

        for index, block_id in enumerate(block_ids_of_this_rank[:-1]):
            if block_ids_of_this_rank[index+1] == block_id + 1:
                num_copying_heads = min(remain_heads, self.node_block_size)
                num_copyed_heads += num_copying_heads
                remain_heads -= num_copying_heads
            else:
                for idx, tensor in enumerate(kvi_tensors):
                    host_data[idx].append(tensor[layer_idx, head_offset:head_offset+num_copyed_heads])

                block_id = block_ids_of_this_rank[index+1]
                head_offset = block_id * self.node_block_size
                num_copyed_heads = min(remain_heads, self.node_block_size)
                remain_heads -= num_copyed_heads

        for idx, tensor in enumerate(kvi_tensors):
            host_data[idx].append(tensor[layer_idx, head_offset:head_offset+num_copyed_heads])
            host_data[idx] = torch.concat(host_data[idx])

        host_data = torch.concat(host_data, dim=-1)

        return host_data

    def update_device_cache(self, prefix_meta, global_device_data):
        src_start = 0
        for block_ranges in prefix_meta.consecutive_blocks:
            num_blocks = sum([end_block_id - start_block_id for start_block_id, end_block_id in block_ranges])
            block_ids = []
            for start_block_id, end_block_id in block_ranges:
                block_ids.extend(range(start_block_id, end_block_id))
            for i in range(len(self.host_cache.kvi_tensors)):
                self.device_cache[i][block_ids] = global_device_data[i][src_start: src_start + num_blocks]
            src_start += num_blocks

    def synchronize_h2d(
        self,
        prefix_meta: "PrefixCopyMeta",
        layer_idx: int,
    ) -> None:
        """When prefix is hit, load the relevant KV from CPU to device buffer.
        key_states: (Tq, N, Dk)
        values_states: (Tq, N, Dv)
        """
        if prefix_meta is None or layer_idx >= self.num_layers:
            return

        with torch.npu.stream(self.h2d_stream):
            # Step0: get the contiguous host data
            host_data = self.get_current_rank_host_data(layer_idx, prefix_meta)

            # Step1: to_device -> [num_blocks*block_size//nn_nodes//NUM_DIE_PER_MACH * headsize]
            device_data = host_data.to(device=self.device)

            # Step2: All Gather -> [tp_world_size, num_blocks*block_size//nn_nodes//NUM_DIE_PER_MACH * headsize]
            global_device_data = tensor_model_parallel_all_gather(device_data, dim=0)

            # Step3: -> [nn_nodes, num_blocks, node_block_size, headsize]
            global_device_data = global_device_data.view(self.tp_nnodes, -1, self.node_block_size, self.head_size)

            # Step4: tranpose -> [num_blocks, nn_nodes, node_block_size, headsize]
            global_device_data = global_device_data.permute(1, 0, 2, 3) # is_contiguous: False

            # Step5: contiguous() --> [num_blocks*block_size, headsize]
            device_kvi_tensor = []
            start_index = 0
            for i, tensor in enumerate(self.host_cache.kvi_tensors):
                length = tensor.shape[-1]
                device_kvi_tensor.append(global_device_data[..., start_index : start_index + length].contiguous().view(-1, self.block_size, 1, length))
                start_index += length

            #  Step6: Device To Device Copy
            self.update_device_cache(prefix_meta, device_kvi_tensor)

            self.h2d_event.record(self.h2d_stream)

    def _copy_tensor_to_buffer(self, src_tensor, buffer_idx):
        for idx_block in range((src_tensor.shape[0] + self.block_size - 1) // self.block_size):
            offset = self.block_size // self.tp_world_size
            start = self.tp_rank * offset + idx_block * self.block_size
            end = min((self.tp_rank + 1) * offset + idx_block * self.block_size, src_tensor.shape[0])
            if start < src_tensor.shape[0]:
                tensor_slice = src_tensor[start:end, ...] # [2, 1,]
                logger.warning(f"<<<{self.batch_buffer_cpu[buffer_idx].shape=}, {self.batch_buffer_cpu[buffer_idx][0:offset, ...].shape=}, {src_tensor.shape=}, {tensor_slice.shape=}")
                self.batch_buffer_cpu[buffer_idx][idx_block*offset : (idx_block+1) * offset, ...].copy_(tensor_slice.squeeze(1), non_blocking=True)  # self.batch_buffer_cpu.shape[1] = 32

    # modified by gpt to improve efficiency
    def synchronize_d2h(
        self,
        kv_cache: Tuple[torch.Tensor,...],
        layer_idx: int,
        kv_event: torch.npu.Event,
        slot_mapping: torch.Tensor
    ) -> None:
        if self.copy_future is not None and not self.copy_future.done():
            self.copy_future.result()
        d2h_event = torch.npu.Event(blocking=False, enable_timing=False)

        # Check that block_ids are within the bounds of kv_cache  
        block_ids = torch.unique(slot_mapping[slot_mapping != -1] // self.block_size)
        max_valid_block = kv_cache[0].shape[0]
        if len(block_ids) > 0 and block_ids.max().item() >= max_valid_block:
            raise IndexError(
                f"block_ids {block_ids.tolist()} contain values >= kv_cache size {max_valid_block}. "
                f"This indicates a mismatch between slot_mapping and kv_cache dimensions."
                )

        with torch.npu.stream(self.d2h_stream):
            self.d2h_stream.wait_event(kv_event)

            # Choose a chunk size that balances memory usage and overhead.
            for i in range(len(kv_cache)):
                tp_stride = self.node_block_size // NUM_DIE_PER_MACH
                num_blocks = len(block_ids)
                
                for start in range(0, num_blocks, D2H_CHUNK_SIZE):
                    end = min(start + D2H_CHUNK_SIZE, num_blocks)
                    
                    # Copy chunk from device to the corresponding rows in the CPU buffer
                    self.batch_buffer_cpu[i][start:end].copy_(
                        kv_cache[i].squeeze(2)[
                            block_ids[start:end],
                            (self.tp_rank // NUM_DIE_PER_MACH)*self.node_block_size + self.tp_local_rank*tp_stride :
                            (self.tp_rank // NUM_DIE_PER_MACH)*self.node_block_size + (self.tp_local_rank+1)*tp_stride,
                            :
                        ],
                        non_blocking=True
                    )
            d2h_event.record(self.d2h_stream)
        
        self.copy_future = self.d2h_thrp.submit(
            self._update_host_cache_thread,
            block_ids, layer_idx, d2h_event
        )
    
    def _padding_kv_cache(self, tensor):
        result = torch.zeros((self.sum_total_len, *tensor.shape[1:]), dtype=tensor.dtype, device = tensor.device)
        result[self.query_mask] = tensor[:self.sum_query_len]
        return result
    
    def _nd_to_nz(self, tensor):
        # Padding KVCache per Req to full Block
        tensor = self._padding_kv_cache(tensor)

        # [global_num_tokens, hidden_size]
        golbal_num_tokens = tensor.size(0)

        # [num_blocks, block_size, num_nz, nz_dim]
        tensor = tensor.view(golbal_num_tokens//self.block_size, self.block_size, -1, self._nz_size)

        # [num_blocks, num_nz, block_size, nz_dim] by Permute & Contiguous
        tensor = tensor.permute(0, 2, 1, 3).contiguous()

        # view to fake global num_tokens
        tensor = tensor.view(golbal_num_tokens, -1)

        # Get local num tokens by Rank index
        tensor = tensor[self.batch_token_indices]

        return tensor

    def _update_host_cache_thread(self, block_ids, layer_idx, event):
        torch.npu.set_device(self.device)
        event.synchronize()

        for i, tensor in enumerate(self.host_cache.kvi_tensors):
            tp_stride = self.node_block_size // NUM_DIE_PER_MACH
            start_col = self.tp_local_rank * tp_stride
            end_col = (self.tp_local_rank + 1) * tp_stride
            tensor.view(self.num_layers, -1, self.block_size // self.tp_nnodes, tensor.shape[-1])[layer_idx, block_ids, start_col:end_col] = self.batch_buffer_cpu[i][:len(block_ids)]

class DecodeOmniCache(BaseOmniCache):
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        runner: NPUModelRunner
    ):
        super().__init__(kv_cache_config, runner)
        self.decode_h2d_stream = torch.npu.Stream(device=self.device)

        self.layer_indices = {
            layer_name: extract_layer_index(layer_name)
            for layer_name in self.device_cache.keys()
        }
        self.sorted_layer_names = list(sorted(
            self.device_cache.keys(),
            key=lambda k: self.layer_indices[k],
        ))
        if self.device_cache:
            first_key_name = next(iter(self.device_cache))
            self.device = self.device_cache[first_key_name][0].device

        self.enable_dsa = model_extra_config.operator_opt_config.enable_dsa

        dp_rank = self.dp_local_rank

        blocks_per_rank = self.host_cache.num_blocks

        # Validate kv_block count before allocating memory
        self._validate_kv_blocks(kv_cache_config)

        self.block_table = torch.arange(blocks_per_rank, dtype=torch.long, device=self.device)
        self._copy_stream = getattr(self, "_copy_stream", torch.npu.Stream())

        if len(kv_cache_config.kv_cache_groups) > 1:
            from omni.accelerators.cache import kv_cache_interface as itfc
            self.sink_blocks, self.recent_blocks = itfc.SINK, itfc.RECENT
            self.omni_attn_pattern: List[int] = itfc.PATTERN.copy()
            self.grouped_layer_indices = [[j for j in range(self.num_layers) if self.omni_attn_pattern[j] == i] for i in [0, 1]]

    def calc_cache_shape(self) -> Tuple[Tuple[int, ...], int]:
        return DecodeOmniCache.calc_cache_shape_for_decode(
            num_layers=self.num_layers,
            block_size=self.block_size,
            head_size=self.head_size,
            dtype=self.dtype,
        )

    @staticmethod
    def calc_cache_shape_for_decode(
        num_layers: int,
        block_size: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[Tuple[int, ...], int]:
        itemize = dtype.itemsize
        numel_per_layer = divide_or_raise(SIZE_BYTES_PER_LAYER, itemize)
        numel_per_block = block_size * head_size
        # For decode, each DP rank has an independent KV cache manager for block allocation.
        # Thus, we should divide num_blocks by the number of managers on each node.
        num_blocks_decode = (numel_per_layer // numel_per_block) // NUM_DIE_PER_MACH

        # Here we 'reshape' the cache to (num_dies, num_blocks_per_die, ...) for efficient addressing.
        d_shape = (
            NUM_DIE_PER_MACH,
            num_layers,
            num_blocks_decode,
            block_size,
            head_size
        )

        return d_shape, num_blocks_decode

    def calculate_kv_xsfer_params(self) -> Tuple[int, int]:
        # block_len_dtype = math.prod(self.shape[2:]) # previous: (die_num, block_num, layer_num, ...), now (die_num, layer_num, block_num)
        block_len_dtype = math.prod(self.shape[3:]) * self.shape[1]
        logger.warning(f"<<<<<< {block_len_dtype=}, {self.shape=}")
        dp_offset = self.dp_local_rank * self.num_blocks
        return block_len_dtype, dp_offset

    def initialize_device_cache(
        self,
        kv_cache_config: KVCacheConfig,
        runner: NPUModelRunner
    ) -> Optional[Dict[str, Tuple[torch.Tensor]]]:
        kv_caches = {}
        bsz_seq = runner.graph_block_tables.shape[0]  # batch_size * seq_len
        batch_size = runner.max_num_reqs
        num_tokens_per_reqs_decode = 1 if not runner.use_spec_decode else (1 + runner.speculative_config.num_speculative_tokens)
        seq_len = num_tokens_per_reqs_decode
        headnum = 1
        s_block_size = 128
        k_rope_sz = 64
        kvcache_sz = 512
        topk = 2048
        self.selection_topk_block_size = 1
        selection_max_seq_len = topk * self.selection_topk_block_size

        if int(os.getenv("ENABLE_HOST_MAPPING", "0")) and model_extra_config.operator_opt_config.enable_dsa:
            s_max_block_num = (selection_max_seq_len + s_block_size - 1) // s_block_size

            self.selection_k_rope = [
                torch.zeros(
                    [s_max_block_num * bsz_seq * headnum, s_block_size, k_rope_sz],
                    dtype=torch.bfloat16,
                    device=self.device,
                ).contiguous()
                for _ in range(self.num_layers)
            ]

            self.selection_kv_cache = [
                torch.zeros(
                    [s_max_block_num * bsz_seq * headnum, s_block_size, kvcache_sz],
                    dtype=torch.bfloat16,
                    device=self.device,
                ).contiguous()
                for _ in range(self.num_layers)
            ]

            # allocate full tensor for all layers to make it update faster
            self.selection_kv_block_table = torch.arange(
                bsz_seq * headnum * s_max_block_num,
                dtype=torch.int32,
                device=self.device,
            ).view(bsz_seq * headnum, s_max_block_num).contiguous()

            self.selection_kv_block_status = -torch.ones(
                            [self.num_layers, bsz_seq, 1, headnum, (topk + 1)],
                            dtype=torch.int32,
                            device=self.device,
                        ).contiguous()
            self.selection_kv_block_status_list = list(torch.unbind(self.selection_kv_block_status, dim=0))

            # work buffers initialization for reordering and updating in faster behavior
            self.selection_kv_block_table_buffer = torch.empty(
                            [batch_size, seq_len, s_max_block_num],
                            dtype=torch.int32,
                            device=self.device,
                        ).contiguous()
            self.selection_kv_block_status_buffer = torch.empty(
                            [self.num_layers, batch_size, seq_len, headnum, (topk + 1)],
                            dtype=torch.int32,
                            device=self.device,
                        ).contiguous()
            self.index_buffer = torch.empty(batch_size, dtype=torch.long, device=self.device)

            self.selection_topk_indices = torch.arange(
                                    topk,
                                    dtype=torch.int32,
                                    device=self.device,
                                ).expand(bsz_seq, headnum, topk).contiguous()

        for i, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            kv_cache_spec = kv_cache_group.kv_cache_spec
            for layer_name in kv_cache_group.layer_names:
                tensor_config = kv_cache_config.tensors[layer_name]
                if tensor_config.size % kv_cache_spec.page_size_bytes != 0:
                    raise RuntimeError("tensor_config.size must be divisible by kv_cache_spec.page_size_bytes")
                num_blocks = tensor_config.size // kv_cache_spec.page_size_bytes
                if isinstance(kv_cache_spec, AttentionSpec):
                    kv_cache_shape = runner.attn_backends[i].get_kv_cache_shape(
                        num_blocks,
                        kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size
                    )
                    kv_caches[layer_name] = runner.attn_backends[i].init_kv_cache_each_layer(
                        kv_cache_shape,
                        runner.dtype,
                        runner.device,
                        runner.model_config,
                        runner.enable_torchair_graph_mode,
                        is_prefill = False
                    )
                else:
                    raise ValueError("Unknown KV cache spec type.")
        return kv_caches

    def build_h2d_ops(self, local_block_ids: List[List[int]], tp_nnodes: int = 1) -> None:
        if len(local_block_ids) > 1:
            batch_device_mem, batch_device_max, batch_host_mem, batch_host_sizes = self.build_h2d_ops_omni_attn(local_block_ids)
        else:
            layer_indices = self.device_cache.keys()
            npu_blocks = []
            for block_id in local_block_ids[0]:
                layers = []
                for layer_name in layer_indices:
                    if model_extra_config.operator_opt_config.enable_dsa:
                        layers.append(
                            (
                                # only keep k_indexer cache on device
                                self.device_cache[layer_name][0][block_id],
                                self.device_cache[layer_name][1][block_id],
                                self.device_cache[layer_name][2][block_id]
                            )
                        )
                    else:
                        layers.append(
                            (
                                self.device_cache[layer_name][0][block_id],
                                self.device_cache[layer_name][1][block_id]
                            )
                        )
                npu_blocks.append(layers)
            batch_device_mem, batch_device_max, batch_host_mem, batch_host_sizes = self.host_cache.batch_layer_copy_to_npu(local_block_ids[0], npu_blocks)
        
        return batch_device_mem, batch_device_max, batch_host_mem, batch_host_sizes

    def synchronize_h2d(self, batch_device_mem, batch_device_max, batch_host_mem, batch_host_sizes):
        self.host_cache.memcpy_async(batch_device_mem, batch_device_max, batch_host_mem, batch_host_sizes)

    def build_h2d_ops_omni_attn(self, local_block_ids: List[List[int]]) -> None:
        if model_extra_config.operator_opt_config.enable_dsa:
            raise RuntimeError("Omni attention does not support DSV3.2 model yet.")
        if len(local_block_ids) != 2:
            raise RuntimeError(f"Omni attention needs two block tables, but got {len(local_block_ids)}: {local_block_ids}.")
        if len(local_block_ids[1]) != self.sink_blocks + self.recent_blocks:
            raise RuntimeError(f"Omni layer block table should have length {self.sink_blocks + self.recent_blocks},"
                               f" but got {len(local_block_ids[1])}. {local_block_ids=}.")

        host_omni_blocks = local_block_ids[0].copy()
        if len(host_omni_blocks) > self.sink_blocks + self.recent_blocks:
            host_omni_blocks = host_omni_blocks[:self.sink_blocks] + host_omni_blocks[-self.recent_blocks:]
        elif len(host_omni_blocks) < self.sink_blocks + self.recent_blocks:
            local_block_ids[1] = local_block_ids[1][:len(host_omni_blocks)]

        src_blocks = [local_block_ids[0], host_omni_blocks]
        tgt_blocks = local_block_ids
        
        batch_device_mem, batch_device_max, batch_host_mem, batch_host_sizes = [], [], [], []
        for i, (src, tgt) in enumerate(zip(src_blocks, tgt_blocks)):
            npu_blocks = []
            for block_id in tgt:
                layers = []
                for layer_name in self.sorted_layer_names:
                    layer_idx = self.layer_indices[layer_name]
                    if self.omni_attn_pattern[layer_idx] != i:
                        continue
                    layers.append(
                        (
                            self.device_cache[layer_name][0][block_id],
                            self.device_cache[layer_name][1][block_id]
                        )
                    )
                npu_blocks.append(layers)
            micro_batch_device_mem, micro_batch_device_max, micro_batch_host_mem, micro_batch_host_sizes = self.host_cache.batch_layer_copy_to_npu(src, npu_blocks, layer_indices=self.grouped_layer_indices[i])
            batch_device_mem.extend(micro_batch_device_mem)
            batch_device_max.extend(micro_batch_device_max)
            batch_host_mem.extend(micro_batch_host_mem)
            batch_host_sizes.extend(micro_batch_host_sizes)
        
        return batch_device_mem, batch_device_max, batch_host_mem, batch_host_sizes
    
    def synchronize_d2h(self, key_states: torch.Tensor, value_states: torch.Tensor, slot_mapping: torch.Tensor, layer_idx: int, kv_event: torch.npu.Event) -> None:
        raise NotImplementedError


def divide_or_raise(a: int, b: int):
    if a % b != 0:
        raise ValueError(f"Error! Number 'a' {a} is not divisible by number 'b' {b}.")
    return a // b


def create_omni_cache(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
    runner: NPUModelRunner
) -> BaseOmniCache:
    """
    Factory function to create the appropriate BaseOmniCache instance based on the is_prefill flag.

    Args:
        kv_cache_config: Configuration for the KV cache
        vllm_config: The VllmConfig object
        runner: NPU model runner instance

    Returns:
        PrefillOmniCache or DecodeOmniCache instance based on the is_prefill flag
    """
    is_prefill = vllm_config.kv_transfer_config.kv_role != "kv_consumer"
    if is_prefill:
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        max_model_len = vllm_config.scheduler_config.max_model_len
        omni_cache = PrefillOmniCache(kv_cache_config, runner, max_num_batched_tokens=max_num_batched_tokens, max_num_seqs=max_num_seqs, max_model_len=max_model_len)
        runner.kv_caches = None
        for layer_name in kv_cache_config.kv_cache_groups[0].layer_names:
            vllm_config.compilation_config.static_forward_context[layer_name].kv_cache = [None]
    else:
        omni_cache = DecodeOmniCache(kv_cache_config, runner)
        assert omni_cache.device_cache is not None
        bind_kv_cache(
            omni_cache.device_cache,
            vllm_config.compilation_config.static_forward_context,
            runner.kv_caches,
        )
        # replace kv_a and k_pe in self.kv_caches by host swap caches
        num_layers = len(runner.kv_caches)
        # if not is_prefill:
        if int(os.getenv("ENABLE_HOST_MAPPING", "0")) and model_extra_config.operator_opt_config.enable_dsa:
            for i in range(num_layers):
                rest = runner.kv_caches[i]
                t0, t1 = omni_cache.host_swap_tensor[i][0], omni_cache.host_swap_tensor[i][1]
                runner.kv_caches[i] = (t0, t1, *rest)
    return omni_cache


@dataclass
class PrefixCopyMeta:
    consecutive_blocks: list[list[Tuple[int, int]]]
    """The starts and ends of consecutive full blocks."""

    query_lens: list[int]
    """Number of tokens per sample in current batch."""

    query_slots: torch.Tensor
    """The positions to store the KV of current tokens."""

    num_actual_tokens: int = None
    """Total number of tokens in current batch."""

    num_copy_blocks: int = None
    """Total number of blocks in KV cache to copy."""

    last_block_id: Optional[int] = None
    """The last block which might be partially filled. In APC, it must be None."""

    last_block_len: Optional[int] = None
    """Number of tokens filled in the last block."""

    def __post_init__(self):
        if len(self.consecutive_blocks) != len(self.query_lens):
            raise RuntimeError(f"Lengths mismatch! {len(self.consecutive_blocks)=}, while {len(self.query_lens)=}.")
        self.num_actual_tokens = sum(self.query_lens)
        flatten = [pair[1] - pair[0] for segs in self.consecutive_blocks for pair in segs]
        total_copy_ops = len(flatten)
        total_copy_blocks = sum(flatten)

        if get_world_group() == 0:
            logger.warning(f"!!! Totally {total_copy_ops} copy operations with {total_copy_blocks} blocks will be executed. ***")

        self.num_copy_blocks = total_copy_blocks

def generate_full_block_slot(slot_mapping, query_lens, block_size):
    blocks = slot_mapping // block_size
    device = slot_mapping.device
    index_per_block = torch.arange(block_size, dtype=slot_mapping.dtype, device=device)
    result = []
    start = 0
    for query_len in query_lens:
        end = start + query_len
        num_block = math.ceil((end-start)/block_size)
        query_blocks = blocks[start:end]
        block_index = torch.arange(num_block, device= device) * block_size
        query_blocks = query_blocks[block_index]
        query_slot = index_per_block.repeat(num_block, 1)
        query_slot = query_slot + (query_blocks * block_size).unsqueeze(1)
        result.append(query_slot)
        start = end
    result = torch.concat(result, dim=0).view(-1)
    return result

def pad_inputs(input: torch.Tensor, query_lens: list[int], sp_size: int, pad_value: int):
    count = 0
    res = []
    for len in query_lens:
        pad_size = (sp_size - len % sp_size) % sp_size
        tmp_tensor = input[count:count + len]
        padded_tensor = pad_tensor(tmp_tensor, pad_size, pad_value)
        res.append(padded_tensor)
        count += len
    return torch.cat(res, dim=0)


def pad_tensor(tensor, pad_size, pad_value=0):
    """Pad tensor with specified value along first dimension."""
    padded_shape = (pad_size, tensor.shape[-1]) if tensor.dim() > 1 else (pad_size,)
    padding = torch.full(
        padded_shape,
        pad_value,
        dtype=tensor.dtype,
        device=tensor.device
    )
    return torch.cat([tensor, padding])
