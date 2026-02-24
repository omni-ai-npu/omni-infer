import fcntl
import pickle
import time
import gzip
import struct
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from abc import ABC
from pathlib import Path
from typing import Optional
from multiprocessing import shared_memory
from unittest.mock import patch
from vllm.logger import logger
from vllm.distributed import get_tp_group
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
)
from vllm.sampling_params import RequestOutputKind

_global_experts_capturer: Optional["RoutedExpertsCapturer"] = None
_global_experts_reader: Optional["RoutedExpertsReader"] = None

SAVE_DIR = os.getenv("EXPORT_MOE_EXPERTS_TEST_PATH", None)
ROLE = os.getenv("ROLE", None)
INVALID_INDICES = -1
FMT = "i"
SIZE = struct.calcsize(FMT)
BUFFER_PREFIX = "vllm_routed_experts_buffer"
EXPORT_MOE_EXPERTS_TMP_DIR = os.getenv("EXPORT_MOE_EXPERTS_TMP_DIR", "./tmp_lock_dir")
LOCK_FILE_PREFIX = os.path.join(
    EXPORT_MOE_EXPERTS_TMP_DIR, 
    "vllm_routed_experts"
)
Path(EXPORT_MOE_EXPERTS_TMP_DIR).mkdir(exist_ok=True)

def save_data(indices: np.ndarray, data: np.ndarray, save_dir: str, part_key: str) -> None:
    """Save indices and data to a compressed pickle file."""
    if save_dir is None:
        return

    data_dict = {part_key : {int(indices[i]): v for i, v in enumerate(data)}}
    save_name = os.path.join(save_dir, f"data_{time.time()}.pkl")

    with gzip.open(save_name, "wb") as file:
        pickle.dump(data_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

def save_indices(indices: np.ndarray, request_id: str, save_dir: str, part_key: str) -> None:
    """Save request indices to a compressed pickle file."""
    if save_dir is None:
        return
    
    save_name = os.path.join(save_dir, f"indices_{time.time()}.pkl")
    with gzip.open(save_name, "wb") as file:
        pickle.dump({request_id: {part_key: indices.tolist()}}, file, protocol=pickle.HIGHEST_PROTOCOL)

def pad_3d(data, target_shape):
    """padding 3d tensor"""
    if data.numel() == 0:
        return torch.zeros(target_shape, dtype=data.dtype, device=data.device)
    diff = target_shape[0] - data.shape[0]
    if diff < 0:
        raise ValueError(f"target_shape[0] = {target_shape[0]} must not be less than data.shape[0] = {data.shape[0]}")
    padded_data = F.pad(data, (0, 0, 0, 0, 0, diff), mode='constant', value=0)
    return padded_data

def pad_1d(data, target_len):
    """padding 1d tensor"""
    if data.numel() == 0:
        return torch.zeros(target_len, dtype=data.dtype, device=data.device)
    diff = target_len - data.shape[0]
    if diff < 0:
        raise ValueError(f"target_len = {target_len} must not be less than data.shape[0] = {data.shape[0]}")
    padded_data = F.pad(data, (0, diff), mode='constant', value=INVALID_INDICES)
    return padded_data

def lock_file(file_pointer) -> None:
    """Acquire an exclusive lock on a file."""
    fcntl.flock(file_pointer, fcntl.LOCK_EX)

def unlock_file(file_pointer) -> None:
    """Release a file lock."""
    fcntl.flock(file_pointer, fcntl.LOCK_UN)

class RoutedExpertsCapturer(ABC):
    """Abstract base class for routed experts capturer."""
    
    @staticmethod
    def create() -> "RoutedExpertsCapturer":
        """Create a singleton instance of the experts capturer."""
        global _global_experts_capturer
        if _global_experts_capturer is not None:
            raise RuntimeError("Experts capturer already created.")
        _global_experts_capturer = _RoutedExpertsCapturer()
        return _global_experts_capturer

    @staticmethod
    def get_instance() -> Optional["RoutedExpertsCapturer"]:
        """Get the singleton instance of the experts capturer."""
        return _global_experts_capturer

class _RoutedExpertsCapturer:
    """Concrete implementation of routed experts capturer."""

    def __init__(self) -> None:
        self._experts_capturer_device_buffer: Optional[torch.Tensor] = None
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._host_buffer_view: Optional[np.ndarray] = None
        self._shm_total_token_num: Optional[shared_memory.SharedMemory] = None
        self.instance_id: Optional[str] = None
        self.num_moe_layers: int = 0
        self.num_selected_experts: int = 0
        self.torch_dtype: torch.dtype = torch.uint8
        self.np_dtype: np.dtype = np.uint8
        self.max_num_kv_tokens: int = 0
        self.lock_file: Optional[str] = None

    def init_buffer(
        self,
        max_num_batched_tokens: int,
        num_blocks: int,
        num_cache_groups: int,
        block_size: int,
        hf_config,
        instance_id: str,
        enable_init_shared_memory: bool,
    ) -> None:
        """Initialize the experts capturer buffer."""
        self.is_pd_disaggregation = bool(ROLE)
        self.is_pd_prefill = ROLE == "prefill"
        num_hidden_layers = getattr(hf_config, "num_hidden_layers", 0)
        num_dense_layers = getattr(hf_config, "num_dense_layers", None)
        if num_dense_layers is None:
            mlp_only_layers = getattr(hf_config, "mlp_only_layers", None)
            num_dense_layers = len(mlp_only_layers) if mlp_only_layers else 0
        num_experts_per_tok = getattr(hf_config, "num_experts_per_tok", 0)
        self.instance_id = instance_id
        self.num_moe_layers = num_hidden_layers - num_dense_layers
        self.num_selected_experts = num_experts_per_tok
        self.torch_dtype = (
            torch.uint8
            if self.num_selected_experts - 1 <= torch.iinfo(torch.uint8).max
            else torch.uint16
        )
        self.np_dtype = np.uint8 if self.torch_dtype == torch.uint8 else np.uint16
        self.max_num_kv_tokens = (num_blocks // num_cache_groups + 1) * block_size

        if self._experts_capturer_device_buffer is None:
            self._experts_capturer_device_buffer = torch.zeros(
                (max_num_batched_tokens, self.num_moe_layers, self.num_selected_experts),
                dtype=self.torch_dtype,
                device="npu",
            )
            self.lock_file = f"{LOCK_FILE_PREFIX}_{instance_id}.lock"

            if enable_init_shared_memory:
                shape = (
                    self.max_num_kv_tokens,
                    self.num_moe_layers,
                    self.num_selected_experts,
                )
                nbytes = int(np.prod(shape)) * np.dtype(self.np_dtype).itemsize
                with open(self.lock_file, "wb") as fp:
                    lock_file(fp)
                    try:
                        self._shm = shared_memory.SharedMemory(
                            create=True,
                            size=nbytes,
                            name=f"{BUFFER_PREFIX}_{instance_id}",
                        )
                        self._host_buffer_view = np.ndarray(
                            shape, dtype=self.np_dtype, buffer=self._shm.buf
                        )
                        self._host_buffer_view.fill(0)
                        
                        self._shm_total_token_num = shared_memory.SharedMemory(
                            create=True,
                            size=SIZE,
                            name=f"{BUFFER_PREFIX}_{instance_id}_num",
                        )
                        struct.pack_into(FMT, self._shm_total_token_num.buf, 0, 0)
                    finally:
                        unlock_file(fp)

    def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        """Capture experts for a specific layer."""
        if layer_id is None or self._experts_capturer_device_buffer is None:
            return
        batch_size = topk_ids.shape[0]
        self._experts_capturer_device_buffer[:batch_size, layer_id, :] = topk_ids

    def clear_buffer(self) -> None:
        """Clear the device buffer."""
        if self._experts_capturer_device_buffer is not None:
            self._experts_capturer_device_buffer.zero_()

    def save_captured_experts(
        self,
        slot_mapping: np.ndarray,
        len_req_ids: int,
        max_num_seqs: int,
        enable_torchair_graph_mode: bool,
    ) -> None:
        """Save captured experts to shared memory and disk."""
        if self._experts_capturer_device_buffer is None:
            return None
        real_mapping = torch.from_numpy(slot_mapping).to(self._experts_capturer_device_buffer.device)
        indices = torch.where(real_mapping == -1)[0]
        num_valid_tokens = (
            indices[0].item() if indices.shape[0] > 0 else real_mapping.numel()
        )

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        
        if enable_torchair_graph_mode and num_valid_tokens == len_req_ids:
            split_size = (max_num_seqs - 1) // tp_size + 1
        else:
            split_size = (num_valid_tokens - 1) // tp_size + 1

        rank_lower_bound = split_size * tp_rank
        rank_upper_bound = min(split_size * tp_rank + split_size, num_valid_tokens)

        real_mapping = real_mapping[rank_lower_bound:rank_upper_bound]
        num_tokens = real_mapping.shape[0]
        data = self._experts_capturer_device_buffer[:num_tokens, :, :]

        if tp_size > 1:
            max_num_tokens = torch.tensor(num_tokens, device=self._experts_capturer_device_buffer.device)
            dist.all_reduce(max_num_tokens, op=dist.ReduceOp.MAX, group=get_tp_group().device_group)
            target_shape = list(data.shape)
            target_shape[0] = max_num_tokens.item()
            data = pad_3d(data, target_shape)
            real_mapping = pad_1d(real_mapping, max_num_tokens.item())
            if tp_rank == 0:
                all_data = [torch.empty_like(data) for _ in range(get_tp_group().world_size)]
                all_real_mapping = [torch.empty_like(real_mapping) for _ in range(get_tp_group().world_size)]
                dist.gather(data, gather_list=all_data, dst=get_tp_group().first_rank, group=get_tp_group().device_group)
                dist.gather(real_mapping, gather_list=all_real_mapping, dst=get_tp_group().first_rank, group=get_tp_group().device_group)
            else:
                dist.gather(data, dst=get_tp_group().first_rank, group=get_tp_group().device_group)
                dist.gather(real_mapping, dst=get_tp_group().first_rank, group=get_tp_group().device_group)
                return
            concatenated_data = torch.cat(all_data, dim=0)
            concatenated_real_mapping = torch.cat(all_real_mapping, dim=0)
            valid_mask = concatenated_real_mapping != INVALID_INDICES
            data = concatenated_data[valid_mask]
            real_mapping = concatenated_real_mapping[valid_mask]

        data = data.cpu().numpy()
        indices = real_mapping.cpu().numpy()

        if self._host_buffer_view is None:
            self._attach_shared_memory()

        with open(self.lock_file, "wb+") as fp:
            lock_file(fp)
            try:
                if (
                    self._host_buffer_view is not None
                    and self._experts_capturer_device_buffer is not None
                ):
                    self._host_buffer_view[indices, :, :] = data
                    if self.is_pd_disaggregation:
                        part_key = 'prefill' if self.is_pd_prefill else 'decode'
                    else:
                        part_key = 'total'
                    save_data(indices, data, SAVE_DIR, part_key)
                if self._shm_total_token_num is not None:
                    current_count = struct.unpack_from(
                        FMT, self._shm_total_token_num.buf, 0
                    )[0]
                    struct.pack_into(
                        FMT,
                        self._shm_total_token_num.buf,
                        0,
                        current_count + len(data),
                    )
            finally:
                unlock_file(fp)

    def _attach_shared_memory(self) -> None:
        """Attach to existing shared memory."""
        with open(self.lock_file, "rb+") as fp:
            lock_file(fp)
            try:
                with patch("multiprocessing.resource_tracker.register", lambda *args, **kwargs: None):
                    if self._shm is None:
                        self._shm = shared_memory.SharedMemory(
                            name=f"{BUFFER_PREFIX}_{self.instance_id}"
                        )
                    if self._shm_total_token_num is None:
                        self._shm_total_token_num = shared_memory.SharedMemory(
                            name=f"{BUFFER_PREFIX}_{self.instance_id}_num"
                        )
                    
                    buffer_size = (
                        self._shm.size
                        // self.num_moe_layers
                        // self.num_selected_experts
                        // np.dtype(self.np_dtype).itemsize
                    )
                    self._host_buffer_view = np.ndarray(
                        (buffer_size, self.num_moe_layers, self.num_selected_experts),
                        dtype=self.np_dtype,
                        buffer=self._shm.buf,
                    )
            finally:
                unlock_file(fp)

    def __del__(self) -> None:
        """Cleanup shared memory resources."""
        try:
            if self._shm is not None:
                self._shm.close()
                self._shm.unlink()
                if self._shm_total_token_num is not None:
                    self._shm_total_token_num.close()
                    self._shm_total_token_num.unlink()
        except Exception:
            logger.debug("Exception during __del__ cleanup for capturer", exc_info=True)

class RoutedExpertsReader(ABC):
    """Abstract base class for routed experts reader."""
    
    @staticmethod
    def create() -> "RoutedExpertsReader":
        """Create a singleton instance of the experts reader."""
        global _global_experts_reader
        if _global_experts_reader is None:
            _global_experts_reader = _RoutedExpertsReader()
        return _global_experts_reader

    @staticmethod
    def get_instance() -> Optional["RoutedExpertsReader"]:
        """Get the singleton instance of the experts reader."""
        if _global_experts_reader is None:
            logger.info("Experts reader not initialized.")
        return _global_experts_reader

class _RoutedExpertsReader:
    """Concrete implementation of routed experts reader."""

    def __init__(self) -> None:
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._host_buffer_view: Optional[np.ndarray] = None
        self._shm_total_token_num: Optional[shared_memory.SharedMemory] = None
        self.is_pd_disaggregation: bool = False
        self.block_size: int = 0
        self.lock_file: Optional[str] = None
        self.np_dtype: np.dtype = np.uint8

    def attach_buffer(
        self,
        hf_config,
        block_size: int,
        instance_id: str,
    ) -> None:
        """Attach to the shared memory buffer."""
        if self._shm is not None:
            return
        num_hidden_layers = getattr(hf_config, "num_hidden_layers", 0)
        num_dense_layers = getattr(hf_config, "num_dense_layers", None)
        if num_dense_layers is None:
            mlp_only_layers = getattr(hf_config, "mlp_only_layers", None)
            num_dense_layers = len(mlp_only_layers) if mlp_only_layers else 0
        num_experts_per_tok = getattr(hf_config, "num_experts_per_tok", 0)
        num_moe_layers = num_hidden_layers - num_dense_layers
        self.is_pd_disaggregation = bool(ROLE)
        self.is_pd_prefill = ROLE == "prefill"
        self.block_size = block_size
        self.lock_file = f"{LOCK_FILE_PREFIX}_{instance_id}.lock"
        self.np_dtype = (
            np.uint8
            if num_experts_per_tok - 1 <= np.iinfo(np.uint8).max
            else np.uint16
        )

        with open(self.lock_file, "rb+") as fp:
            lock_file(fp)
            try:
                with patch("multiprocessing.resource_tracker.register", lambda *args, **kwargs: None):
                    self._shm = shared_memory.SharedMemory(
                        name=f"{BUFFER_PREFIX}_{instance_id}"
                    )
                    buffer_size = (
                        self._shm.size
                        // num_moe_layers
                        // num_experts_per_tok
                        // np.dtype(self.np_dtype).itemsize
                    )
                    self._host_buffer_view = np.ndarray(
                        (buffer_size, num_moe_layers, num_experts_per_tok),
                        dtype=self.np_dtype,
                        buffer=self._shm.buf,
                    )
                    self._shm_total_token_num = shared_memory.SharedMemory(
                        name=f"{BUFFER_PREFIX}_{instance_id}_num"
                    )
            finally:
                unlock_file(fp)

    def get_routed_experts(
        self, request, kv_cache_manager, stopped: bool, num_new_token :int
    ) -> Optional[np.ndarray]:
        """Get routed experts for a request."""

        is_stream = request.sampling_params.output_kind != RequestOutputKind.FINAL_ONLY
        if not stopped and not is_stream:
            return None

        block_ids = kv_cache_manager.get_block_ids(request.request_id)[0]
        num_tokens = request.num_tokens - 1
        is_prefill = self.is_pd_prefill if self.is_pd_disaggregation else (num_tokens == request.num_prompt_tokens)
        start_tokens = (
            request.num_prompt_tokens
            if not is_prefill and self.is_pd_disaggregation
            else 0
        )
        block_ids_array = np.array(block_ids, dtype=np.uint32)
        indices = (
            block_ids_array.reshape((-1, 1)) * self.block_size
            + np.arange(self.block_size)
        ).flatten()[start_tokens:num_tokens]
        if indices is not None and stopped:
            if self.is_pd_disaggregation:
                part_key = 'prefill' if is_prefill else 'decode'
            else:
                part_key = 'total'
            save_indices(indices, request.request_id, SAVE_DIR, part_key)
        if not is_prefill and is_stream:
            indices = indices[-num_new_token:]
        expect_token_num = indices.size
        if is_prefill or not self.is_pd_disaggregation:
            expect_token_num = indices.size - request.num_cached_tokens

        while (
            struct.unpack_from(FMT, self._shm_total_token_num.buf, 0)[0] < expect_token_num
        ):
            pass

        with open(self.lock_file, "rb+") as fp:
            lock_file(fp)
            try:
                struct.pack_into(
                    FMT,
                    self._shm_total_token_num.buf,
                    0,
                    struct.unpack_from(FMT, self._shm_total_token_num.buf, 0)[0] - expect_token_num,
                )
                if self._host_buffer_view is None:
                    raise RuntimeError("Buffer not attached.")
                routed_experts = self._host_buffer_view[indices, :, :].copy()
            finally:
                unlock_file(fp)
        return routed_experts

    def __del__(self) -> None:
        """Cleanup shared memory resources."""

        try:
            if self._shm is not None:
                self._shm.close()
                self._shm_total_token_num.close()
                os.remove(self.lock_file)
        except Exception:
            logger.debug("Exception during __del__ cleanup for reader", exc_info=True)