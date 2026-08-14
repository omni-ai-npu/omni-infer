# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import base64
import time
import zlib
from collections import defaultdict
from contextvars import ContextVar
from multiprocessing import shared_memory
from unittest.mock import patch
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Iterator

import numpy as np
import torch
import torch.distributed as dist

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.distributed.kv_events import KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.generate.base.serving import GenerateBaseServing as OpenAIServing
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    _BUFFER_PREFIX,
    _LOCK_FILE_PREFIX,
    _create_or_attach_shared_memory,
    _file_lock,
    RoutedExpertsCapturer as VLLMRoutedExpertsCapturer,
    RoutedExpertsReader as VLLMRoutedExpertsReader,
)
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.utils import remove_all
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from omni_npu import envs
from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.plugin_decorators import update_from_output_decorator

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
    from vllm.v1.kv_cache_interface import AttentionSpec


logger = init_logger(__name__)

_ORIGINAL_SCHEDULER_INIT = Scheduler.__init__
_ORIGINAL_GPU_MODEL_RUNNER_INIT = GPUModelRunner.__init__
_ORIGINAL_INIT_ROUTED_EXPERTS_CAPTURER = GPUModelRunner.init_routed_experts_capturer
_ORIGINAL_SAVE_CAPTURED_EXPERTS = VLLMRoutedExpertsCapturer.save_captured_experts
_ORIGINAL_CHAT_STREAM = OpenAIServingChat.chat_completion_stream_generator
_ORIGINAL_COMPLETION_STREAM = OpenAIServingCompletion.completion_stream_generator
_ORIGINAL_COMPLETION_FINAL = (
    OpenAIServingCompletion.request_output_to_completion_response
)
_ORIGINAL_CHAT_CHOICE_INIT = ChatCompletionResponseChoice.__init__
_ORIGINAL_CHAT_STREAM_CHOICE_INIT = ChatCompletionResponseStreamChoice.__init__
_ORIGINAL_COMPLETION_CHOICE_INIT = CompletionResponseChoice.__init__
_ORIGINAL_COMPLETION_STREAM_CHOICE_INIT = CompletionResponseStreamChoice.__init__
_original_add = RequestOutput.add
_INVALID_INDEX = -1
_ROUTED_EXPERT_KEYS = (
    "routed_experts_shape",
    "routed_experts_dtype",
    "routed_experts_str_len",
    "routed_experts_str",
)
_CURRENT_ROUTED_EXPERTS_PAYLOAD: ContextVar[dict[str, Any] | None] = ContextVar(
    "current_routed_experts_payload",
    default=None,
)
_CURRENT_SCHEDULER_INIT_VLLM_CONFIG: ContextVar["VllmConfig | None"] = ContextVar(
    "current_scheduler_init_vllm_config",
    default=None,
)
_CHOICE_PATCHES_APPLIED = False


def _set_pd_flags(
    reader: VLLMRoutedExpertsReader,
    vllm_config: "VllmConfig | None" = None,
) -> None:
    kv_transfer_config = None
    if vllm_config is not None:
        kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)

    if kv_transfer_config is not None and kv_transfer_config.is_kv_transfer_instance:
        reader.is_pd_disaggregation = True
        reader.is_pd_prefill = kv_transfer_config.is_kv_producer
        return

    role = envs.OMNI_PD_ROLE
    if role == "prefill":
        reader.is_pd_disaggregation = True
        reader.is_pd_prefill = True
        return
    if role == "decode":
        reader.is_pd_disaggregation = True
        reader.is_pd_prefill = False
        return

    reader.is_pd_disaggregation = False
    reader.is_pd_prefill = False


def _pad_first_dim(
    tensor: torch.Tensor,
    target_size: int,
    fill_value: int,
) -> torch.Tensor:
    if tensor.shape[0] >= target_size:
        return tensor

    pad_shape = (target_size - tensor.shape[0], *tensor.shape[1:])
    pad_tensor = torch.full(
        pad_shape,
        fill_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, pad_tensor], dim=0)


def _write_to_shared_memory(
    capturer: VLLMRoutedExpertsCapturer,
    indices: np.ndarray,
    data: np.ndarray,
) -> None:
    import fcntl

    if capturer._lock_file is None:
        raise RuntimeError("Shared memory not initialized.")
    if capturer._host_buffer_view is None:
        return
    if capturer._device_buffer is None:
        raise RuntimeError("Device buffer not initialized.")

    with open(capturer._lock_file, "rb+") as fp:
        fcntl.flock(fp, fcntl.LOCK_EX)
        try:
            capturer._host_buffer_view[indices, :, :] = data
        finally:
            fcntl.flock(fp, fcntl.LOCK_UN)


@register_patch("RoutedExpertsReaderPDSupport", VLLMRoutedExpertsReader)
class RoutedExpertsReaderPDSupportPatch(VLLMPatch):
    _attr_names_to_apply = ["attach_buffer"]

    def attach_buffer(
        self,
        max_num_kv_tokens: int,
        model_config: "ModelConfig",
        instance_id: str,
        vllm_config: "VllmConfig | None" = None,
    ) -> None:
        if self._shm is not None:
            return

        if vllm_config is None:
            vllm_config = _CURRENT_SCHEDULER_INIT_VLLM_CONFIG.get()

        hf_config = model_config.hf_text_config
        total_experts = getattr(hf_config, "n_routed_experts", None)
        if total_experts is None:
            total_experts = getattr(hf_config, "num_experts", None)
        if total_experts is not None and total_experts <= 256:
            host_dtype = np.uint8
        else:
            host_dtype = np.int16
        shape = (
            max_num_kv_tokens,
            hf_config.num_hidden_layers,
            hf_config.num_experts_per_tok,
        )

        parallel_config = getattr(vllm_config, "parallel_config", None)
        if (
            parallel_config is not None
            and parallel_config.data_parallel_size > 1
        ):
            instance_id = (
                f"{instance_id}_dp{parallel_config.data_parallel_rank_local}"
            )

        self._lock_file = f"{_LOCK_FILE_PREFIX}_{instance_id}.lock"
        shm_name = f"{_BUFFER_PREFIX}_{instance_id}"

        with _file_lock(self._lock_file, mode="rb+"):
            # Avoid resource_tracker registering the shared memory.
            with patch(
                "multiprocessing.resource_tracker.register",
                lambda *args, **kwargs: None,
            ):
                self._shm = shared_memory.SharedMemory(name=shm_name)

            self._host_buffer_view = np.ndarray(
                shape,
                dtype=host_dtype,
                buffer=self._shm.buf,
            )

        _set_pd_flags(self, vllm_config)


@register_patch("RoutedExpertsCapturerTPAggregate", VLLMRoutedExpertsCapturer)
class RoutedExpertsCapturerTPAggregatePatch(VLLMPatch):
    _attr_names_to_apply = ["init_buffer", "save_captured_experts"]

    def init_buffer(
        self,
        max_num_batched_tokens: int,
        max_num_kv_tokens: int,
        model_config: "ModelConfig",
        instance_id: str,
        vllm_config: "VllmConfig | None" = None,
    ) -> None:
        if self._device_buffer is not None:
            raise RuntimeError("Device buffer has already been initialized")

        hf_config = model_config.hf_text_config
        num_layers = hf_config.num_hidden_layers
        num_experts_per_tok = hf_config.num_experts_per_tok
        total_experts = getattr(hf_config, "n_routed_experts", None)
        if total_experts is None:
            total_experts = getattr(hf_config, "num_experts", None)
        if total_experts is not None and total_experts <= 256:
            device_dtype = torch.uint8
            host_dtype = np.uint8
        else:
            device_dtype = torch.int16
            host_dtype = np.int16

        self._device_buffer = torch.zeros(
            (max_num_batched_tokens, num_layers, num_experts_per_tok),
            dtype=device_dtype,
            device="npu",
        )

        if get_tensor_model_parallel_rank() != 0:
            return

        shape = (max_num_kv_tokens, num_layers, num_experts_per_tok)
        buffer_size = int(np.prod(shape)) * np.dtype(host_dtype).itemsize

        parallel_config = getattr(vllm_config, "parallel_config", None)
        if (
            parallel_config is not None
            and parallel_config.data_parallel_size > 1
        ):
            instance_id = (
                f"{instance_id}_dp{parallel_config.data_parallel_rank_local}"
            )

        self._lock_file = f"{_LOCK_FILE_PREFIX}_{instance_id}.lock"
        self._shm_name = f"{_BUFFER_PREFIX}_{instance_id}"
        self._shm = _create_or_attach_shared_memory(
            self._shm_name,
            buffer_size,
            self._lock_file,
        )
        self._host_buffer_view = np.ndarray(
            shape,
            dtype=host_dtype,
            buffer=self._shm.buf,
        )
        self._host_buffer_view.fill(0)

    def save_captured_experts(self, indices: np.ndarray) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size <= 1:
            _ORIGINAL_SAVE_CAPTURED_EXPERTS(self, indices)
            return

        if self._device_buffer is None or indices.size == 0:
            return

        tp_rank = get_tensor_model_parallel_rank()
        tp_group = get_tp_group()
        real_mapping = torch.from_numpy(indices).to(self._device_buffer.device)

        invalid_positions = torch.where(real_mapping == _INVALID_INDEX)[0]
        num_valid_tokens = (
            invalid_positions[0].item()
            if invalid_positions.numel() > 0
            else real_mapping.numel()
        )
        if num_valid_tokens <= 0:
            return

        split_size = (num_valid_tokens - 1) // tp_size + 1
        rank_lower_bound = split_size * tp_rank
        rank_upper_bound = min(rank_lower_bound + split_size, num_valid_tokens)

        real_mapping = real_mapping[rank_lower_bound:rank_upper_bound]
        local_num_tokens = real_mapping.shape[0]
        local_data = self._device_buffer[:local_num_tokens, :, :]

        max_num_tokens = torch.tensor(
            local_num_tokens,
            dtype=torch.int64,
            device=self._device_buffer.device,
        )
        dist.all_reduce(
            max_num_tokens,
            op=dist.ReduceOp.MAX,
            group=tp_group.device_group,
        )
        max_num_tokens_value = int(max_num_tokens.item())
        if max_num_tokens_value <= 0:
            return

        local_data = _pad_first_dim(local_data, max_num_tokens_value, _INVALID_INDEX)
        real_mapping = _pad_first_dim(real_mapping, max_num_tokens_value, _INVALID_INDEX)

        all_data = [torch.empty_like(local_data) for _ in range(tp_group.world_size)]
        all_mapping = [
            torch.empty_like(real_mapping) for _ in range(tp_group.world_size)
        ]
        dist.all_gather(
            all_data,
            local_data,
            group=tp_group.device_group,
        )
        dist.all_gather(
            all_mapping,
            real_mapping,
            group=tp_group.device_group,
        )

        if tp_rank != 0:
            return

        # routed_experts are aggregated inside the current TP group only.
        gathered_data = torch.cat(all_data, dim=0)
        gathered_mapping = torch.cat(all_mapping, dim=0)
        valid_mask = gathered_mapping != _INVALID_INDEX
        final_indices = gathered_mapping[valid_mask].cpu().numpy()
        final_data = gathered_data[valid_mask].cpu().numpy()
        _write_to_shared_memory(self, final_indices, final_data)


def _concatenate_routed_experts(
    current: np.ndarray | None,
    incoming: np.ndarray | None,
) -> np.ndarray | None:
    if current is None:
        return incoming
    if incoming is None:
        return current
    if current.ndim != incoming.ndim:
        raise ValueError(
            "Cannot aggregate routed_experts with different ranks: "
            f"{current.ndim} != {incoming.ndim}"
        )
    if current.shape[1:] != incoming.shape[1:]:
        raise ValueError(
            "Cannot aggregate routed_experts with different non-token shapes: "
            f"{current.shape[1:]} != {incoming.shape[1:]}"
        )
    if current.dtype != incoming.dtype:
        raise ValueError(
            "Cannot aggregate routed_experts with different dtypes: "
            f"{current.dtype} != {incoming.dtype}"
        )
    return np.concatenate([current, incoming], axis=0)


@register_patch("RequestOutputRoutedExpertsAggregation", RequestOutput)
class RequestOutputRoutedExpertsAggregationPatch(VLLMPatch):
    _attr_names_to_apply = ["add"]

    def add(self, next_output: "RequestOutput", aggregate: bool) -> None:
        existing_routed_experts = {
            completion.index: completion.routed_experts
            for completion in self.outputs
        }
        incoming_routed_experts = {
            completion.index: completion.routed_experts
            for completion in next_output.outputs
        }

        _original_add(self, next_output, aggregate)

        if not aggregate:
            return

        for completion in self.outputs:
            if completion.index not in existing_routed_experts:
                continue
            if completion.index not in incoming_routed_experts:
                continue

            completion.routed_experts = _concatenate_routed_experts(
                existing_routed_experts[completion.index],
                incoming_routed_experts[completion.index],
            )


@register_patch("SchedulerInitPatch", Scheduler)
class SchedulerInitPatch(VLLMPatch):
    _attr_names_to_apply = ["__init__"]

    def __init__(self, *args: Any, **kwargs: Any):
        vllm_config = kwargs["vllm_config"]
        init_token = _CURRENT_SCHEDULER_INIT_VLLM_CONFIG.set(vllm_config)
        try:
            _ORIGINAL_SCHEDULER_INIT(self, *args, **kwargs)
        finally:
            _CURRENT_SCHEDULER_INIT_VLLM_CONFIG.reset(init_token)

        if (
            getattr(self, "routed_experts_reader", None) is not None
            and hasattr(self, "vllm_config")
            and hasattr(self, "max_num_kv_tokens")
        ):
            kv_cache_config = getattr(self, "kv_cache_config", None)
            self.routed_experts_attn_gid = (
                _get_attention_kv_cache_gid(kv_cache_config)
                if kv_cache_config is not None
                else 0
            )
            if kv_cache_config is not None:
                # Recalculate max_num_kv_tokens using attention group's block_size.
                # The original calculation used cache_config.block_size which may be
                # different from the attention group's block_size in hybrid models.
                attn_group = kv_cache_config.kv_cache_groups[self.routed_experts_attn_gid]
                attn_block_size = attn_group.kv_cache_spec.block_size
                self.max_num_kv_tokens = kv_cache_config.num_blocks * attn_block_size

            # Reset the reader's shm so that attach_buffer can re-initialize with the new max_num_kv_tokens.
            # The original Scheduler.__init__ already called attach_buffer with incorrect values.
            reader = self.routed_experts_reader
            if getattr(reader, "_shm", None) is not None:
                reader._shm = None
                reader._host_buffer_view = None

            reader.attach_buffer(
                max_num_kv_tokens=self.max_num_kv_tokens,
                model_config=self.vllm_config.model_config,
                instance_id=self.vllm_config.instance_id,
                vllm_config=self.vllm_config,
            )


@register_patch("GPUModelRunnerInitRoutedExpertsPatch", GPUModelRunner)
class GPUModelRunnerInitRoutedExpertsPatch(VLLMPatch):
    _attr_names_to_apply = ["__init__", "init_routed_experts_capturer"]

    def __init__(self, *args: Any, **kwargs: Any):
        _ORIGINAL_GPU_MODEL_RUNNER_INIT(self, *args, **kwargs)
        # Initialize routed_experts_attn_gid to 0 by default.
        # It will be properly set in init_routed_experts_capturer().
        self.routed_experts_attn_gid = 0

    def init_routed_experts_capturer(self) -> None:
        logger.info(
            "Initializing routed experts capturer, enable_return_routed_experts: %s",
            self.model_config.enable_return_routed_experts,
        )
        routed_experts_capturer = VLLMRoutedExpertsCapturer.create()

        self.routed_experts_attn_gid = _get_attention_kv_cache_gid(self.kv_cache_config)

        # Recalculate max_num_kv_tokens using attention group's block_size.
        attn_group = self.kv_cache_config.kv_cache_groups[self.routed_experts_attn_gid]
        attn_block_size = attn_group.kv_cache_spec.block_size
        self.max_num_kv_tokens = self.kv_cache_config.num_blocks * attn_block_size

        routed_experts_capturer.init_buffer(
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
            max_num_kv_tokens=self.max_num_kv_tokens,
            model_config=self.model_config,
            instance_id=self.vllm_config.instance_id,
            vllm_config=self.vllm_config,
        )


def _get_attention_kv_cache_gid(kv_cache_config) -> int:
    """Find the first attention group index for routed experts indexing.

    This fixes hybrid models (e.g., Mamba + Attention like Jamba, Qwen3.5)
    where group 0 might be Mamba (block_size=262144) instead of Attention.
    """
    from vllm.v1.kv_cache_interface import AttentionSpec

    for gid, group in enumerate(kv_cache_config.kv_cache_groups):
        if isinstance(group.kv_cache_spec, AttentionSpec):
            return gid
    return 0


def _concatenate_index_ranges(
    cached_indices: np.ndarray,
    fresh_indices: np.ndarray,
) -> np.ndarray:
    if cached_indices.size == 0:
        return fresh_indices
    if fresh_indices.size == 0:
        return cached_indices
    return np.concatenate([cached_indices, fresh_indices], axis=0)


def _get_request_slot_mapping(self: Scheduler, request: Request) -> np.ndarray | None:
    kv_blocks = self.kv_cache_manager.get_blocks(request.request_id)
    if kv_blocks is None:
        return None

    # Use the attention group index instead of hardcoded 0.
    # This fixes hybrid models (Mamba + Attention) where group 0 might be Mamba.
    attn_gid = getattr(self, "routed_experts_attn_gid", 0)
    block_ids = kv_blocks.get_block_ids()[attn_gid]
    if not block_ids:
        return None

    # Get block_size from the attention group's kv_cache_spec.
    kv_cache_config = getattr(self, "kv_cache_config", None)
    if kv_cache_config is not None and attn_gid < len(kv_cache_config.kv_cache_groups):
        block_size = kv_cache_config.kv_cache_groups[attn_gid].kv_cache_spec.block_size
    else:
        block_size = self.kv_cache_manager.coordinator.single_type_managers[0].block_size
    block_ids_array = np.array(block_ids, dtype=np.int32)
    num_blocks = len(block_ids)
    block_offsets = np.arange(block_size, dtype=np.int32)
    slot_mapping = (
        block_offsets.reshape((1, block_size))
        + block_ids_array.reshape((num_blocks, 1)) * block_size
    ).flatten()
    return slot_mapping


def _get_routed_experts_from_slot_mapping(
    reader: Any,
    request: Request,
    slot_mapping: np.ndarray | None,
    *,
    stopped: bool,
    num_new_tokens: int,
) -> np.ndarray | None:
    if reader is None or slot_mapping is None:
        return None

    output_kind = (
        RequestOutputKind.FINAL_ONLY
        if request.sampling_params is None
        else request.sampling_params.output_kind
    )
    if not stopped and output_kind == RequestOutputKind.FINAL_ONLY:
        return None

    num_tokens = request.num_tokens - 1
    if num_tokens <= 0:
        return None
    if request.num_cached_tokens < 0:
        raise RuntimeError("request.num_cached_tokens is not initialized")
    if request.num_cached_tokens > num_tokens:
        raise ValueError(
            "request.num_cached_tokens exceeds routed experts token range: "
            f"{request.num_cached_tokens} > {num_tokens}"
        )

    is_pd_disaggregation = getattr(reader, "is_pd_disaggregation", False)
    is_prefill = (
        getattr(reader, "is_pd_prefill", False)
        if is_pd_disaggregation
        else num_tokens == request.num_prompt_tokens
    )
    if is_pd_disaggregation and not is_prefill:
        cached_indices = slot_mapping[0:0]
        fresh_indices = slot_mapping[request.num_prompt_tokens:num_tokens]
    else:
        cached_indices = slot_mapping[:request.num_cached_tokens]
        fresh_indices = slot_mapping[request.num_cached_tokens:num_tokens]

    indices = _concatenate_index_ranges(cached_indices, fresh_indices)
    if indices.size == 0:
        return None

    if output_kind == RequestOutputKind.DELTA:
        if num_new_tokens <= 0:
            return None

        if is_pd_disaggregation:
            indices = indices[-num_new_tokens:]
        else:
            prompt_indices = slot_mapping[: min(request.num_prompt_tokens, num_tokens)]
            decode_indices = slot_mapping[request.num_prompt_tokens:num_tokens]
            delta_indices = decode_indices[-num_new_tokens:]
            prompt_sent = getattr(
                request,
                "_omni_stream_prompt_routed_experts_sent",
                False,
            )
            if prompt_sent:
                indices = delta_indices
            else:
                indices = _concatenate_index_ranges(prompt_indices, delta_indices)
                if indices.size > 0:
                    request._omni_stream_prompt_routed_experts_sent = True
        if indices.size == 0:
            return None
    elif output_kind not in (
        RequestOutputKind.FINAL_ONLY,
        RequestOutputKind.CUMULATIVE,
    ):
        raise ValueError(f"Unsupported RequestOutputKind: {output_kind}")

    return reader.get_routed_experts(indices=indices)


def _snapshot_scheduled_requests(
    self: Scheduler,
    scheduler_output: SchedulerOutput,
    model_runner_output: ModelRunnerOutput,
) -> dict[str, dict[str, Any]]:
    prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict or {}
    pooler_outputs = model_runner_output.pooler_output
    request_snapshots: dict[str, dict[str, Any]] = {}

    for req_id in scheduler_output.num_scheduled_tokens:
        request = self.requests.get(req_id)
        if request is None:
            continue

        slot_mapping = _get_request_slot_mapping(self, request)
        req_index = model_runner_output.req_id_to_index[req_id]
        pooler_output = pooler_outputs[req_index] if pooler_outputs else None
        request_snapshots[req_id] = {
            "request": request,
            "slot_mapping": slot_mapping,
            "client_index": request.client_index,
            "prompt_logprobs_tensors": prompt_logprobs_dict.get(req_id),
            "pooler_output": pooler_output,
        }

    return request_snapshots


@register_patch("SchedulerRoutedExpertsPatch", Scheduler)
class SchedulerRoutedExpertsPatch(VLLMPatch):
    _attr_names_to_apply = ["update_from_output"]

    @update_from_output_decorator
    def _omni_npu_update_from_output_backport(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output
        cudagraph_stats = model_runner_output.cudagraph_stats

        perf_stats = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(
                scheduler_output
            )

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if kv_connector_stats and self.connector:
            kv_stats = self.connector.get_kv_connector_stats()
            if kv_stats:
                kv_connector_stats = kv_connector_stats.aggregate(kv_stats)

        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            failed_kv_load_req_ids = self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids
            )

        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            if num_tokens_scheduled <= 0:
                raise ValueError(f"num_tokens_scheduled must be > 0, got {num_tokens_scheduled}")
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                continue
            request = self.requests.get(req_id)
            if request is None:
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            )
            if scheduled_spec_token_ids and generated_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
                if request.num_computed_tokens > 0:
                    request.num_computed_tokens -= num_rejected
                if request.num_output_placeholders > 0:
                    request.num_output_placeholders -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,
                )

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            kv_transfer_params = None
            status_before_stop = request.status

            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids
                )
            elif request.pooling_params and pooler_output is not None:
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True
                
            if new_token_ids and self.structured_output_manager.should_advance(
                request, new_token_ids=new_token_ids
                ):
                struct_output_request = request.structured_output_request
                if struct_output_request is None:
                    raise ValueError(f"structured_output_request is None")
                grammar = struct_output_request.grammar
                if grammar is None:
                    raise ValueError(f"grammar is None")
            
                # Reasoning content (up to and including the end marker) is
                # not grammar content; drop it before advancing (#44006).
                advance_token_ids = (
                    self.structured_output_manager.trim_reasoning_for_advance(
                        request, new_token_ids
                    )
                )
                if advance_token_ids and not grammar.accept_tokens(
                    req_id, advance_token_ids
                ):
                    logger.error(
                        "Unexpected: grammar rejected tokens %s for request %s. "
                        "Terminating request.",
                        advance_token_ids,
                        req_id,
                    )
                    request.status = RequestStatus.FINISHED_ERROR
                    request.resumable = False
                    stopped = True

            routed_experts = None
            if stopped:
                if getattr(self.vllm_config.model_config, "enable_return_routed_experts", False):
                    # Use the attention group index instead of hardcoded 0.
                    # This fixes hybrid models (Mamba + Attention) where group 0 might be Mamba.
                    attn_gid = getattr(self, "routed_experts_attn_gid", 0)
                    kv_blocks = self.kv_cache_manager.get_blocks(request.request_id)
                    block_ids = kv_blocks.get_block_ids()[attn_gid]
                    num_tokens = request.num_tokens - 1
                    block_ids_array = np.array(block_ids, dtype=np.int32)
                    num_blocks = len(block_ids)
                    # Get block_size from the attention group's kv_cache_spec.
                    kv_cache_config = getattr(self, "kv_cache_config", None)
                    if kv_cache_config is not None and attn_gid < len(kv_cache_config.kv_cache_groups):
                        block_size = kv_cache_config.kv_cache_groups[attn_gid].kv_cache_spec.block_size
                    else:
                        block_size = self.block_size
                    block_offsets = np.arange(0, block_size)
                    slot_mapping = (
                        block_offsets.reshape((1, block_size))
                        + block_ids_array.reshape((num_blocks, 1)) * block_size
                    ).flatten()[:num_tokens]
                    routed_experts = self.routed_experts_reader.get_routed_experts(
                        indices=slot_mapping
                    )
                kv_transfer_params = self._free_request(request)
                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            if (
                request.sampling_params is not None
                and request.sampling_params.logprobs is not None
                and logprobs
            ):
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or pooler_output is not None or kv_transfer_params:
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                        routed_experts=routed_experts,
                        num_nans_in_logits=request.num_nans_in_logits,
                    )
                )
            else:
                if prompt_logprobs_tensors:
                    raise RuntimeError("Expected no prompt_logprobs_tensors")

        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            self.waiting.remove_requests(stopped_preempted_reqs)

        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
            requests = [self.requests[req_id] for req_id in failed_kv_load_req_ids]
            self.finish_requests(failed_kv_load_req_ids, RequestStatus.FINISHED_ERROR)
            for request in requests:
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                    )
                )

        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        events = self.kv_cache_manager.take_events()
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            for client_index, finished_set in finished_req_ids.items():
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set
                    )
            finished_req_ids.clear()

        if (
            stats := self.make_stats(
                spec_decoding_stats, kv_connector_stats, cudagraph_stats, perf_stats
            )
        ) is not None:
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        base_update_from_output = (
            SchedulerRoutedExpertsPatch._omni_npu_update_from_output_backport
        )
        if not getattr(self.vllm_config.model_config, "enable_return_routed_experts", False):
            return base_update_from_output(
                self,
                scheduler_output,
                model_runner_output,
            )

        reader = getattr(self, "routed_experts_reader", None)
        if reader is None:
            return base_update_from_output(
                self,
                scheduler_output,
                model_runner_output,
            )

        request_snapshots = _snapshot_scheduled_requests(
            self,
            scheduler_output,
            model_runner_output,
        )
        engine_core_outputs = base_update_from_output(
            self,
            scheduler_output,
            model_runner_output,
        )

        outputs_by_request_id: dict[str, EngineCoreOutput] = {}
        for engine_core_output in engine_core_outputs.values():
            for request_output in engine_core_output.outputs:
                outputs_by_request_id[request_output.request_id] = request_output

        for req_id, snapshot in request_snapshots.items():
            request = snapshot["request"]
            slot_mapping = snapshot["slot_mapping"]
            client_index = snapshot["client_index"]

            existing_output = outputs_by_request_id.get(req_id)
            if existing_output is not None:
                existing_output.routed_experts = _get_routed_experts_from_slot_mapping(
                    reader,
                    request,
                    slot_mapping,
                    stopped=existing_output.finish_reason is not None,
                    num_new_tokens=len(existing_output.new_token_ids),
                )
                continue

            routed_experts = _get_routed_experts_from_slot_mapping(
                reader,
                request,
                slot_mapping,
                stopped=RequestStatus.is_finished(request.status),
                num_new_tokens=0,
            )
            if routed_experts is None:
                continue

            engine_core_output = engine_core_outputs.get(client_index)
            if engine_core_output is None:
                engine_core_output = EngineCoreOutputs(outputs=[])
                engine_core_outputs[client_index] = engine_core_output

            engine_core_output.outputs.append(
                EngineCoreOutput(
                    request_id=req_id,
                    new_token_ids=[],
                    finish_reason=request.get_finished_reason(),
                    new_prompt_logprobs_tensors=snapshot["prompt_logprobs_tensors"],
                    pooling_output=snapshot["pooler_output"],
                    stop_reason=request.stop_reason,
                    events=request.take_events(),
                    trace_headers=request.trace_headers,
                    num_cached_tokens=request.num_cached_tokens,
                    routed_experts=routed_experts,
                    num_nans_in_logits=request.num_nans_in_logits,
                )
            )

        return engine_core_outputs


@register_patch("RoutedExpertsSerializationPatch", OpenAIServing)
class RoutedExpertsSerializationPatch(VLLMPatch):
    _attr_names_to_apply = [
        "_get_routed_experts_serialization_mode",
        "convert_ndarray_to_str",
        "restore_str_to_ndarray",
        "concatenate_dict_and_ndarray",
        "add_ndarray_info_to_dict",
    ]

    def _get_routed_experts_serialization_mode(self) -> str:
        return self.engine_client.vllm_config.routed_experts_serialization_mode

    def convert_ndarray_to_str(
        self, data: np.ndarray
    ) -> tuple[str, tuple[int, ...], str]:
        shape = data.shape
        dtype_str = str(data.dtype)
        raw_bytes = data.tobytes()
        mode = self._get_routed_experts_serialization_mode()
        if mode == "zip_base64":
            serialized = zlib.compress(raw_bytes, level=1)
        elif mode == "base64":
            serialized = raw_bytes
        else:
            raise ValueError(f"Unsupported routed experts serialization mode: {mode}")
        b64_data = base64.b64encode(serialized).decode("utf-8")
        return b64_data, shape, dtype_str

    def restore_str_to_ndarray(
        self, b64_data: str, shape: tuple[int, ...], dtype_str: str
    ) -> np.ndarray:
        mode = self._get_routed_experts_serialization_mode()
        serialized = base64.b64decode(b64_data)
        if mode == "zip_base64":
            raw_bytes = zlib.decompress(serialized)
        elif mode == "base64":
            raw_bytes = serialized
        else:
            raise ValueError(f"Unsupported routed experts serialization mode: {mode}")
        return np.frombuffer(raw_bytes, dtype=np.dtype(dtype_str)).reshape(shape)

    def concatenate_dict_and_ndarray(
        self, routed_experts_dict: dict[str, Any], expert_ids_arr: np.ndarray
    ) -> np.ndarray:
        prefill_expert_ids = routed_experts_dict["routed_experts_str"]
        shape = tuple(routed_experts_dict["routed_experts_shape"])
        dtype = routed_experts_dict["routed_experts_dtype"]
        if len(shape) != expert_ids_arr.ndim:
            raise ValueError(
                "Cannot concatenate routed_experts with different ranks: "
                f"{shape} vs {expert_ids_arr.shape}"
            )
        if tuple(shape[1:]) != tuple(expert_ids_arr.shape[1:]):
            raise ValueError(
                "Cannot concatenate routed_experts with different non-token shapes: "
                f"{shape[1:]} vs {expert_ids_arr.shape[1:]}"
            )
        if dtype != str(expert_ids_arr.dtype):
            raise ValueError(
                "Cannot concatenate routed_experts with different dtypes: "
                f"{dtype} vs {expert_ids_arr.dtype}"
            )
        arr = self.restore_str_to_ndarray(prefill_expert_ids, shape, dtype)
        return np.concatenate([arr, expert_ids_arr], axis=0)

    def add_ndarray_info_to_dict(
        self, data: np.ndarray, routed_experts_dict: dict[str, Any]
    ) -> None:
        b64_data, shape, dtype = self.convert_ndarray_to_str(data)
        routed_experts_dict["routed_experts_shape"] = shape
        routed_experts_dict["routed_experts_dtype"] = dtype
        routed_experts_dict["routed_experts_str_len"] = len(b64_data)
        routed_experts_dict["routed_experts_str"] = b64_data


def _is_prefill_node(serving: Any) -> bool:
    engine_client = getattr(serving, "engine_client", None)
    vllm_config = getattr(engine_client, "vllm_config", None)
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_config is not None and getattr(
        kv_transfer_config, "is_kv_transfer_instance", False
    ):
        kv_role = getattr(kv_transfer_config, "kv_role", None)
        if kv_role == "kv_producer" or getattr(
            kv_transfer_config, "is_kv_producer", False
        ):
            return True
        if kv_role == "kv_consumer" or getattr(
            kv_transfer_config, "is_kv_consumer", False
        ):
            return False

    role = envs.OMNI_PD_ROLE
    if role == "prefill":
        return True
    if role == "decode":
        return False
    return False


def _extract_kv_routed_experts(
    kv_transfer_params: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not kv_transfer_params:
        return None
    if all(key in kv_transfer_params for key in _ROUTED_EXPERT_KEYS):
        return {key: kv_transfer_params[key] for key in _ROUTED_EXPERT_KEYS}
    return None


def _build_routed_experts_payload(
    serving: Any,
    output: Any,
    kv_payload: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    routed_experts = getattr(output, "routed_experts", None)
    if routed_experts is not None and getattr(routed_experts, "shape", None) is not None:
        if routed_experts.shape[0] == 0:
            routed_experts = None

    if routed_experts is not None:
        if kv_payload is not None:
            routed_experts = serving.concatenate_dict_and_ndarray(
                kv_payload, routed_experts
            )
        payload: dict[str, Any] = {}
        serving.add_ndarray_info_to_dict(routed_experts, payload)
        return payload
    return kv_payload


def _stash_prefill_routed_experts(serving: Any, request_output: Any) -> None:
    if not _is_prefill_node(serving):
        return

    outputs = getattr(request_output, "outputs", None) or []
    for output in outputs:
        routed_experts = getattr(output, "routed_experts", None)
        if routed_experts is None or getattr(routed_experts, "shape", None) is None:
            continue
        if routed_experts.shape[0] == 0:
            continue
        if getattr(request_output, "kv_transfer_params", None) is None:
            request_output.kv_transfer_params = {}
        serving.add_ndarray_info_to_dict(
            routed_experts,
            request_output.kv_transfer_params,
        )
        return


def _attach_context_routed_experts(choice: Any) -> None:
    payload = _CURRENT_ROUTED_EXPERTS_PAYLOAD.get()
    if payload is not None:
        choice.routed_experts = payload


def _ensure_choice_patches_applied() -> None:
    global _CHOICE_PATCHES_APPLIED
    if _CHOICE_PATCHES_APPLIED:
        return

    def _wrap_chat_stream_choice_init(self, *args: Any, **kwargs: Any):
        _ORIGINAL_CHAT_STREAM_CHOICE_INIT(self, *args, **kwargs)
        _attach_context_routed_experts(self)

    def _wrap_chat_choice_init(self, *args: Any, **kwargs: Any):
        _ORIGINAL_CHAT_CHOICE_INIT(self, *args, **kwargs)
        _attach_context_routed_experts(self)

    def _wrap_completion_stream_choice_init(self, *args: Any, **kwargs: Any):
        _ORIGINAL_COMPLETION_STREAM_CHOICE_INIT(self, *args, **kwargs)
        _attach_context_routed_experts(self)

    def _wrap_completion_choice_init(self, *args: Any, **kwargs: Any):
        _ORIGINAL_COMPLETION_CHOICE_INIT(self, *args, **kwargs)
        _attach_context_routed_experts(self)

    ChatCompletionResponseStreamChoice.__init__ = _wrap_chat_stream_choice_init
    ChatCompletionResponseChoice.__init__ = _wrap_chat_choice_init
    CompletionResponseStreamChoice.__init__ = _wrap_completion_stream_choice_init
    CompletionResponseChoice.__init__ = _wrap_completion_choice_init
    _CHOICE_PATCHES_APPLIED = True


class _OutputSequenceProxy:
    def __init__(
        self,
        outputs: list[Any],
        payload_builder: Callable[[Any], dict[str, Any] | None],
    ):
        self._outputs = outputs
        self._payload_builder = payload_builder

    def __iter__(self) -> Iterator[Any]:
        for output in self._outputs:
            token = _CURRENT_ROUTED_EXPERTS_PAYLOAD.set(
                self._payload_builder(output)
            )
            try:
                yield output
            finally:
                _CURRENT_ROUTED_EXPERTS_PAYLOAD.reset(token)

    def __len__(self) -> int:
        return len(self._outputs)

    def __getitem__(self, index: int) -> Any:
        return self._outputs[index]


class _RequestOutputProxy:
    def __init__(
        self,
        request_output: Any,
        payload_builder: Callable[[Any], dict[str, Any] | None],
    ):
        self._request_output = request_output
        self.outputs = _OutputSequenceProxy(request_output.outputs, payload_builder)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._request_output, name)


async def _wrap_request_output_stream(
    serving: Any,
    request: Any,
    result_generator: AsyncIterator[Any],
) -> AsyncIterator[Any]:
    kv_payload = _extract_kv_routed_experts(getattr(request, "kv_transfer_params", None))
    prefill_attached_choices: set[int] = set()

    def _build_streaming_payload(output: Any) -> dict[str, Any] | None:
        output_kv_payload = None
        if bool(getattr(output, "token_ids", None)) and output.index not in prefill_attached_choices:
            prefill_attached_choices.add(output.index)
            output_kv_payload = kv_payload
        return _build_routed_experts_payload(serving, output, output_kv_payload)

    async for result in result_generator:
        _stash_prefill_routed_experts(serving, result)
        yield _RequestOutputProxy(
            result,
            _build_streaming_payload,
        )


async def _wrap_indexed_request_output_stream(
    serving: Any,
    request: Any,
    result_generator: AsyncIterator[tuple[int, Any]],
) -> AsyncIterator[tuple[int, Any]]:
    kv_payload = _extract_kv_routed_experts(getattr(request, "kv_transfer_params", None))
    prefill_attached_outputs: set[tuple[int, int]] = set()

    async for prompt_idx, result in result_generator:
        def _build_streaming_payload(
            output: Any,
            current_prompt_idx: int = prompt_idx,
        ) -> dict[str, Any] | None:
            output_kv_payload = None
            key = (current_prompt_idx, output.index)
            if bool(getattr(output, "token_ids", None)) and key not in prefill_attached_outputs:
                prefill_attached_outputs.add(key)
                output_kv_payload = kv_payload
            return _build_routed_experts_payload(serving, output, output_kv_payload)

        _stash_prefill_routed_experts(serving, result)
        yield prompt_idx, _RequestOutputProxy(
            result,
            _build_streaming_payload,
        )


@register_patch("ExpertIdServingChatStream", OpenAIServingChat)
class ExpertIdServingChatStream(VLLMPatch):
    _attr_names_to_apply = ["chat_completion_stream_generator"]

    async def chat_completion_stream_generator(
        self,
        request: Any,
        result_generator: AsyncIterator[Any],
        request_id: str,
        model_name: str,
        conversation: list[Any],
        tokenizer: Any,
        request_metadata: Any,
    ):
        _ensure_choice_patches_applied()
        async for chunk in _ORIGINAL_CHAT_STREAM(
            self,
            request,
            _wrap_request_output_stream(self, request, result_generator),
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
        ):
            yield chunk


@register_patch("ExpertIdServingCompletionStream", OpenAIServingCompletion)
class ExpertIdServingCompletionStream(VLLMPatch):
    _attr_names_to_apply = ["completion_stream_generator"]

    async def completion_stream_generator(
        self,
        request: Any,
        engine_prompts: list[Any],
        result_generator: AsyncIterator[tuple[int, Any]],
        request_id: str,
        created_time: int,
        model_name: str,
        num_prompts: int,
        tokenizer: Any,
        request_metadata: Any,
    ):
        _ensure_choice_patches_applied()
        async for chunk in _ORIGINAL_COMPLETION_STREAM(
            self,
            request,
            engine_prompts,
            _wrap_indexed_request_output_stream(self, request, result_generator),
            request_id,
            created_time,
            model_name,
            num_prompts,
            tokenizer,
            request_metadata,
        ):
            yield chunk


@register_patch("ExpertIdServingCompletionFinal", OpenAIServingCompletion)
class ExpertIdServingCompletionFinal(VLLMPatch):
    _attr_names_to_apply = ["request_output_to_completion_response"]

    def request_output_to_completion_response(
        self,
        final_res_batch: list[Any],
        request: Any,
        request_id: str,
        created_time: int,
        model_name: str,
        tokenizer: Any,
        request_metadata: Any,
    ):
        # --- skip_tokenize P-side: set prompt_token_ids for D-side relay ---
        skip_decode = envs.OMNI_SKIP_DECODE_TOKENIZE
        if skip_decode and _is_prefill_node(self) and final_res_batch:
            first_res = final_res_batch[0]
            if first_res.kv_transfer_params is None:
                first_res.kv_transfer_params = {}
            raw_pt_ids = getattr(first_res, "prompt_token_ids", None)
            pt_ids = list(raw_pt_ids) if raw_pt_ids is not None else []
            first_res.kv_transfer_params["prompt_token_ids"] = pt_ids
            logger.info(
                "[SKIP_DECODE_CHECK][P-side-completion-nonstream] "
                "request_output_to_completion_response: set kv_transfer_params "
                "prompt_token_ids len=%s",
                len(pt_ids),
            )
        # --- existing ExpertId logic ---
        payloads: list[dict[str, Any] | None] = []
        kv_payload = _extract_kv_routed_experts(getattr(request, "kv_transfer_params", None))
        for result in final_res_batch:
            _stash_prefill_routed_experts(self, result)
            for output in result.outputs:
                payloads.append(
                    _build_routed_experts_payload(
                        self,
                        output,
                        kv_payload,
                    )
                )
        response = _ORIGINAL_COMPLETION_FINAL(
            self,
            final_res_batch,
            request,
            request_id,
            created_time,
            model_name,
            tokenizer,
            request_metadata,
        )
        for choice, payload in zip(response.choices, payloads):
            if payload is not None:
                choice.routed_experts = payload
        return response
