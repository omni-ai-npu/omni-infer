# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright contributors to the vLLM project.

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Union, Any, cast, TypeAlias
import os

import torch
import numpy as np
import torch.nn as nn
from dataclasses import replace
from functools import wraps

from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import (
    CompilationMode,
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
    set_current_vllm_config,
)
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
from vllm.distributed.parallel_state import (
    get_pp_group,
    prepare_communication_buffer_for_model,
)
from vllm.forward_context import BatchDescriptor, set_forward_context, get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.models.interfaces import supports_mm_encoder_only
from vllm.model_executor.models.utils import extract_layer_index
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype, get_dtype_size
from vllm.v1.attention.backend import (
    AttentionMetadata,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    KVCacheConfig,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    DraftTokenIds,
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput,
    SamplerOutput,
)
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.ubatch_utils import maybe_create_ubatch_slices
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.ubatch_utils import UBatchSlices
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

from omni_npu.compilation.acl_graph import (
    ACLGraphWrapper,
    consume_aclgraph_recapture,
    set_graph_params,
)
from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.sample.sampler import NPUSamplerV1, ENABLE_NPU_PENALTY_CACHE
from omni_npu.sample.rejection_sampler import NPURejectionSampler
from omni_npu.plugin_decorators import (
    init_config_decorator,
    prepare_inputs_decorator,
    reinitialize_input_batch_decorator,
    model_output_decorator,
)
from omni_npu.v1.layers.vocab_parallel_embedding import NPUParallelLMHead
from omni_npu.worker.npu_graph_dispatcher import NPUGraphDispatcher
from omni_npu.worker.npu_input_batch import NPUInputBatch
from omni_npu.worker.npu_mem_pool import NpuMemAllocator



AttnMetadataDict: TypeAlias = dict[str, AttentionMetadata]
# list when ubatching is enabled
PerLayerAttnMetadata: TypeAlias = list[AttnMetadataDict] | AttnMetadataDict


@contextmanager
def switch_torch_device():
    origin_cuda = torch.cuda
    torch.cuda = torch.npu
    try:
        yield
    finally:
        torch.cuda = origin_cuda


class NPUModelRunner(GPUModelRunner):

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        with switch_torch_device():
            super().__init__(vllm_config, device)

        self.dp_parallel_lmhead = model_extra_config.parall_config.ena_dp_lmhead_parallel
        self.local_parallel_lmhead = (
            model_extra_config.parall_config.ena_local_lmhead_parallel
        )

        # is_mm_prefix_lm is used in _build_attention_metadata
        self.is_mm_prefix_lm = self.model_config.is_mm_prefix_lm

        # enable mtp acl graph mode
        if self.speculative_config and get_pp_group().is_last_rank and isinstance(self.drafter, EagleProposer):
            if self.compilation_config.mode == CompilationMode.VLLM_COMPILE:
                self.drafter.use_cuda_graph = self.compilation_config.cudagraph_mode.has_mode(CUDAGraphMode.FULL)
                self.drafter.batch_desc = None
                self.drafter.target_model_cuda_graph_mode = None

        # Overwrite num_accepted_tokens from GPUModelRunner to make it int32
        self.num_accepted_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )
        self.num_prompt_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )

        # sampled_token_ids is int32 in npu, sampled_token_ids_pinned_cpu should
        # be same dtype to synchronize.
        self.sampled_token_ids_pinned_cpu = torch.empty(
            (self.max_model_len, 1),
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.pin_memory)

        # FIXME(runze): reusing VLLM's sampler fails, this sampler class is from omni_infer.
        # need to check why and try to remove it.
        self.sampler = NPUSamplerV1()

        if self.speculative_config and get_pp_group().is_last_rank:
            self.rejection_sampler = NPURejectionSampler(self.sampler)

        if vllm_config.additional_config is not None:
            from omni_npu.compilation.npugraph_ex_config import init_aclgraph_config
            init_aclgraph_config(vllm_config)
            self.use_rejection_sampler = vllm_config.additional_config.get("use_rejection_sampler", False)
            self.use_penalty = vllm_config.additional_config.get("use_penalty", False)
            self.total_step = vllm_config.additional_config.get("multi_step", 1)
            self.combine_block = vllm_config.additional_config.get("combine_block", 1)
            self.use_process_before_sample = vllm_config.additional_config.get("use_process_before_sample", False)
        else:
            self.use_rejection_sampler = False
            self.use_penalty = False
            self.total_step = 1
            self.combine_block = 1
            self.use_process_before_sample = False
        self.use_spec_decode = False
        num_tokens_per_reqs_decode = 1 if not self.use_spec_decode else (1 + self.speculative_config.num_speculative_tokens)
        self.block_size = vllm_config.cache_config.block_size
        self.max_num_blocks_per_req = cdiv(self.model_config.max_model_len,
                                           self.block_size * self.combine_block) * self.combine_block
        self.graph_block_tables = np.zeros(
            (self.max_num_reqs * num_tokens_per_reqs_decode,
             self.max_num_blocks_per_req),
            dtype=np.int32)
        val = getattr(self.model_config.hf_text_config, "router_sliding_window", 0)
        if isinstance(val, (int, float)):
            self.router_sliding_window = val
        else:
            self.router_sliding_window = 0
        if self.router_sliding_window > 0:
            self.req_cache_map = {self.max_num_reqs + 1: 0}
            self.cache_slot_id = torch.zeros(self.max_num_reqs,
                                    dtype=torch.long, device=self.device)

        hf_config = vllm_config.model_config.hf_config
        if getattr(hf_config, "use_mhc", False) and hasattr(hf_config, "vision_config"):
            self.inputs_embeds.gpu = self.inputs_embeds.gpu.repeat(1, hf_config.mhc_num_stream)
        
        self.batch_execution_and_padding_state: tuple[
            CUDAGraphMode,
            BatchDescriptor,
            torch.Tensor | None,
        ] | None = None

        # make buffer for speculative decode
        if self.speculative_config:
            self.cu_num_draft_tokens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
            self.cu_num_sampled_tokens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
            self.logits_indices = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
            self.target_logits_indices = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
            self.bonus_logits_indices = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
        
        self._is_mm_encoder_only = False

        # use npugraph dispatcher
        self.cudagraph_dispatcher = NPUGraphDispatcher(self.vllm_config)

    def _build_conv_context(self, dummy:bool = False):
        forward_context = get_forward_context()
        if not dummy:
            keys_to_remove = [k for k in self.req_cache_map if k not in self.input_batch.req_ids]
            for k in keys_to_remove:
                del self.req_cache_map[k]
            for idx, req_id in enumerate(self.input_batch.req_ids):
                if req_id in self.req_cache_map:
                    cache_id = self.req_cache_map[req_id]
                    self.cache_slot_id[idx] = cache_id
                else:
                    self.cache_slot_id[idx] = 0
                self.req_cache_map[req_id] = idx + 1
            self.cache_slot_id[self.input_batch.num_reqs:] = 0
        forward_context.cache_slot_id = self.cache_slot_id

    def _calc_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
    ) -> SpecDecodeMetadata:
        # Inputs:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]
        # Outputs:
        # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
        # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,
        #                            206, 207, 208]
        # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
        # bonus_logits_indices:     [  3,   4,   7,   8,  10]

        # Compute the logits indices.
        # [4, 1, 3, 1, 2]
        num_sampled_tokens = num_draft_tokens + 1

        # Step 1. cu_num_sampled_tokens: [4, 5, 8, 9, 11]
        # arange: [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
        cu_num_sampled_tokens, arange = self._get_cumsum_and_arange(
            num_sampled_tokens, cumsum_dtype=np.int32
        )
        # Step 2. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens
        )
        # Step 3. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
        logits_indices += arange

        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the draft logits indices.
        # cu_num_draft_tokens: [3, 3, 5, 5, 6]
        # arange: [0, 1, 2, 0, 1, 0]
        cu_num_draft_tokens, arange = self._get_cumsum_and_arange(
            num_draft_tokens, cumsum_dtype=np.int32
        )
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens
        )
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += arange

        # TODO: Optimize the CPU -> GPU copy.
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens)
        cu_num_sampled_tokens = torch.from_numpy(cu_num_sampled_tokens)
        logits_indices = torch.from_numpy(logits_indices)
        target_logits_indices = torch.from_numpy(target_logits_indices)
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices)

        self.cu_num_draft_tokens.cpu[:cu_num_draft_tokens.numel()] = cu_num_draft_tokens
        self.cu_num_sampled_tokens.cpu[:cu_num_sampled_tokens.numel()] = cu_num_sampled_tokens
        self.logits_indices.cpu[:logits_indices.numel()] = logits_indices
        self.target_logits_indices.cpu[:target_logits_indices.numel()] = target_logits_indices
        self.bonus_logits_indices.cpu[:bonus_logits_indices.numel()] = bonus_logits_indices

        self.cu_num_draft_tokens.copy_to_gpu(cu_num_draft_tokens.numel())
        self.cu_num_sampled_tokens.copy_to_gpu(cu_num_sampled_tokens.numel())
        self.logits_indices.copy_to_gpu(logits_indices.numel())
        self.target_logits_indices.copy_to_gpu(target_logits_indices.numel())
        self.bonus_logits_indices.copy_to_gpu(bonus_logits_indices.numel())

        cu_num_draft_tokens = self.cu_num_draft_tokens.gpu[:cu_num_draft_tokens.numel()]
        cu_num_sampled_tokens = self.cu_num_sampled_tokens.gpu[:cu_num_sampled_tokens.numel()]
        logits_indices = self.logits_indices.gpu[:logits_indices.numel()]
        target_logits_indices = self.target_logits_indices.gpu[:target_logits_indices.numel()]
        bonus_logits_indices = self.bonus_logits_indices.gpu[:bonus_logits_indices.numel()]

        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = self.input_ids.gpu[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]

        return SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            cu_num_sampled_tokens=cu_num_sampled_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )

    def _model_forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ) -> Any:
        """Helper method to call the model forward pass.

        This method can be overridden by subclasses for model execution.
        Motivation: We can inspect only this method versus
        the whole execute_model, which has additional logic.

        Args:
            input_ids: Input token IDs
            positions: Token positions
            intermediate_tensors: Tensors from previous pipeline stages
            inputs_embeds: Input embeddings (alternative to input_ids)
            **model_kwargs: Additional model arguments

        Returns:
            Model output tensor
        """
        if (
            self.router_sliding_window > 1 
            and not model_extra_config.operator_opt_config.use_noncontiguous_kv
        ):
            self._build_conv_context()
        forward_context = get_forward_context()
        forward_context.capturing = False
        self._capture_dp_pad_target(forward_context)
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )

    def _reshape_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
        kernel_block_sizes: list[int],
    ) -> dict[str, torch.Tensor]:
        kv_caches: dict[str, torch.Tensor] = {}
        has_tensor, has_tuple = False, False
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            attn_backend = group.backend
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                raw_tensor = kv_cache_raw_tensors[layer_name]
                assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0, \
                    f"{kv_cache_spec=}, {raw_tensor.numel()=}, {kv_cache_spec.page_size_bytes=}"
                num_blocks = (raw_tensor.numel() //
                              kv_cache_spec.page_size_bytes)
                kwargs = {}
                kv_cache_tensors = attn_backend.reshape_kv_cache(
                    raw_tensor,
                    num_blocks,
                    kv_cache_spec,
                    **kwargs,
                )
                if isinstance(kv_cache_tensors, torch.Tensor) and kv_cache_tensors.is_contiguous():
                    has_tensor = True
                elif isinstance(kv_cache_tensors, tuple) and len(kv_cache_tensors) > 1:
                    has_tuple = True
                else:
                    raise RuntimeError(
                        f"Invalid case! Cache shouldn't be non-contiguous Tensor or single-element tuple."
                    )
                kv_caches[layer_name] = kv_cache_tensors

        if has_tensor and has_tuple:
            self._update_hybrid_attention_mamba_layout(kv_caches)

        return kv_caches

    def _update_hybrid_attention_mamba_layout(
        self, kv_caches: dict[str, Union[torch.Tensor, tuple[torch.Tensor, ...]]]
    ) -> None:
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            for layer_name in group.layer_names:
                kv_cache = kv_caches[layer_name]
                if (
                    isinstance(kv_cache_spec, AttentionSpec)
                    and isinstance(kv_cache, torch.Tensor)
                    and kv_cache.shape[0] == 2
                ):
                    assert kv_cache.shape[1] != 2, (
                        "Fail to determine whether the layout is "
                        "(2, num_blocks, ...) or (num_blocks, 2, ...) for "
                        f"a tensor of shape {kv_cache.shape}"
                    )
                    hidden_size = kv_cache.shape[2:].numel()
                    kv_cache.as_strided_(
                        size=kv_cache.shape,
                        stride=(hidden_size, 2 * hidden_size, *kv_cache.stride()[2:]),
                    )

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        is_pangu_hybrid = any(name in os.getenv("OMNI_NPU_PATCHES_DIR", "") for name in ('pangu_v2_hybrid', 'pangu_v2_hybrid_vl'))
        if not is_pangu_hybrid and self.vllm_config.model_config.use_mla and hasattr(self.vllm_config.model_config.hf_config, "index_topk"):
            indexer_head_size = self.vllm_config.model_config.hf_config.index_head_dim
            kv_cache_spec: dict[str, KVCacheSpec] = {}
            layer_type = cast(type[Any], AttentionLayerBase)
            attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type)
            for layer_name, attn_module in attn_layers.items():
                config = self.vllm_config.model_config.hf_config
                is_dsa = not hasattr(config, "dsa_layers") or extract_layer_index(layer_name) in config.dsa_layers
                head_size = (attn_module.head_size if hasattr(attn_module, 'head_size') else attn_module.head_dim) + \
                    (indexer_head_size if is_dsa else 0)
                if is_dsa and self.vllm_config.cache_config.cache_dtype in ["hif8_ds_mla"]:
                    # In the "HiF8 with scale" format, each token's KV cache is 656 Bytes
                    # reference vllm/vllm/v1/attention/backends/mla/flashmla_sparse.py
                    kv_cache_spec[layer_name] = MLAAttentionSpec(
                        block_size=self.vllm_config.cache_config.block_size,
                        num_kv_heads=1,
                        head_size=656 + indexer_head_size + 4, # 4 bytes for one fp32 scale
                        dtype=kv_cache_dtype_str_to_dtype(self.vllm_config.cache_config.cache_dtype, self.vllm_config.model_config),
                        cache_dtype_str=self.vllm_config.cache_config.cache_dtype,
                    )
                elif not getattr(attn_module, 'sink_len', 0):
                    if int(os.getenv("ENABLE_OMNI_CACHE", "0")) and self.vllm_config.kv_transfer_config.kv_role == "kv_consumer":
                        head_size = indexer_head_size
                    # hif8_ds_mla kv quantization is only applied to DSA layers
                    if self.vllm_config.cache_config.cache_dtype in ["hif8_ds_mla"] and not is_dsa:
                        kv_dtype = kv_cache_dtype_str_to_dtype("auto", self.vllm_config.model_config)
                        kv_dtype_str = "auto"
                    else:  # keep original specified dtype
                        kv_dtype = kv_cache_dtype_str_to_dtype(self.vllm_config.cache_config.cache_dtype, self.vllm_config.model_config)
                        kv_dtype_str = self.vllm_config.cache_config.cache_dtype
                    kv_cache_spec[layer_name] = MLAAttentionSpec(
                        block_size=self.vllm_config.cache_config.block_size,
                        num_kv_heads=1,
                        head_size=head_size,
                        dtype=kv_dtype,
                        cache_dtype_str=kv_dtype_str,
                    )
                else:
                    from vllm.v1.kv_cache_interface import SinkMLAAttentionSpec
                    kv_cache_spec[layer_name] = SinkMLAAttentionSpec(
                        block_size=self.vllm_config.cache_config.block_size,
                        num_kv_heads=1,
                        head_size=head_size,
                        dtype=kv_cache_dtype_str_to_dtype(self.vllm_config.cache_config.cache_dtype, self.vllm_config.model_config),
                        cache_dtype_str=self.vllm_config.cache_config.cache_dtype,
                        sink_len=attn_module.sink_len,
                    )
            return kv_cache_spec
        else:
            return super().get_kv_cache_spec()

    # Note: used for model runner override.
    def _init_device_properties(self) -> None:
        """Initialize attributes from torch.npu.get_device_properties
        """
        self.device_properties = torch.npu.get_device_properties(self.device)
        self.num_sms = self.device_properties.multi_processor_count

    # Note: used for model runner override.
    def _sync_device(self) -> None:
        torch.npu.synchronize()

    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        num_reqs: int,
        num_scheduled_tokens_np: np.ndarray,
        max_num_scheduled_tokens: int,
        use_cascade_attn: bool,
        allow_microbatching: bool = True,
        force_eager: bool = False,
        # For cudagraph capture TODO(lucas): Refactor how we capture cudagraphs (will
        # be improved in model runner v2)
        force_uniform_decode: bool | None = None,
        force_has_lora: bool | None = None,
        num_encoder_reqs: int = 0,
    ) -> tuple[
        CUDAGraphMode,
        BatchDescriptor,
        bool,
        torch.Tensor | None,
        CUDAGraphStat | None,
    ]:
        uniform_decode = self._is_uniform_decode(
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            uniform_decode_query_len=self.uniform_decode_query_len,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            force_uniform_decode=force_uniform_decode,
        )
        # Encoder-decoder models only support CG for decoder_step > 0 (no enc_output
        # is present). Also, chunked-prefill is disabled, so batch are uniform.
        has_encoder_output = (
            self.model_config.is_encoder_decoder and num_encoder_reqs > 0
        )

        has_lora = (
            len(self.input_batch.lora_id_to_lora_request) > 0
            if force_has_lora is None
            else force_has_lora
        )

        num_tokens_padded = self._pad_for_sequence_parallelism(num_tokens)
        dispatch_cudagraph = (
            lambda num_tokens, disable_full: self.cudagraph_dispatcher.dispatch(
                num_tokens=num_tokens,
                has_lora=has_lora,
                uniform_decode=uniform_decode,
                disable_full=disable_full,
            )
            if not force_eager
            else (CUDAGraphMode.NONE, BatchDescriptor(num_tokens_padded))
        )

        cudagraph_mode, batch_descriptor = dispatch_cudagraph(
            num_tokens_padded, use_cascade_attn or has_encoder_output
        )
        num_tokens_padded = batch_descriptor.num_tokens
        if self.compilation_config.pass_config.enable_sp:
            assert (
                batch_descriptor.num_tokens
                % self.vllm_config.parallel_config.tensor_parallel_size
                == 0
            ), (
                "Sequence parallelism requires num_tokens to be "
                "a multiple of tensor parallel size"
            )

        # Extra coordination when running data-parallel since we need to coordinate
        # across ranks
        should_ubatch, num_tokens_across_dp = False, None
        if self.vllm_config.parallel_config.data_parallel_size > 1:
            # Disable DP padding when running eager to avoid excessive padding when
            # running prefills. This lets us set cudagraph_mode="NONE" on the prefiller
            # in a P/D setup and still use CUDA graphs (enabled by this padding) on the
            # decoder.

            # Adapt start: Add padding for EP
            allow_dp_padding = (
                self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
                or self.parallel_config.enable_expert_parallel
            )
            # Adapt end: Add padding for EP

            should_ubatch, num_tokens_across_dp, synced_cudagraph_mode = (
                coordinate_batch_across_dp(
                    num_tokens_unpadded=num_tokens,
                    parallel_config=self.parallel_config,
                    allow_microbatching=allow_microbatching,
                    allow_dp_padding=allow_dp_padding,
                    num_tokens_padded=num_tokens_padded,
                    uniform_decode=uniform_decode,
                    num_scheduled_tokens_per_request=num_scheduled_tokens_np,
                    cudagraph_mode=cudagraph_mode.value,
                )
            )

            # Extract DP-synced values
            if num_tokens_across_dp is not None:
                dp_rank = self.parallel_config.data_parallel_rank
                num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())
                # Re-dispatch with DP padding so we have the correct batch_descriptor
                cudagraph_mode, batch_descriptor = dispatch_cudagraph(
                    num_tokens_padded,
                    disable_full=synced_cudagraph_mode <= CUDAGraphMode.PIECEWISE.value,
                )
                # Assert to make sure the agreed upon token count is correct otherwise
                # num_tokens_across_dp will no-longer be valid
                assert batch_descriptor.num_tokens == num_tokens_padded

        cudagraph_stats = None
        if self.vllm_config.observability_config.cudagraph_metrics:
            cudagraph_stats = CUDAGraphStat(
                num_unpadded_tokens=num_tokens,
                num_padded_tokens=batch_descriptor.num_tokens,
                num_paddings=batch_descriptor.num_tokens - num_tokens,
                runtime_mode=str(cudagraph_mode),
            )

        # Adapt start: MTP extra property.
        # Add `batch_descriptor` and `cudagraph_mode` for latter use in mtp.
        if self.speculative_config and get_pp_group().is_last_rank and isinstance(self.drafter, EagleProposer):
            self.batch_execution_and_padding_state = (
                cudagraph_mode,
                batch_descriptor,
                num_tokens_across_dp,
            )
        return (
            cudagraph_mode,
            batch_descriptor,
            should_ubatch,
            num_tokens_across_dp,
            cudagraph_stats,
        )

    def _hook_model_load_weights(self, model: nn.Module | None) -> None:
        if model is None:
            return

        if getattr(model, "_omni_npu_load_weights_hooked", False):
            return

        original_load_weights = getattr(model, "load_weights", None)
        if not callable(original_load_weights):
            logger.error("model.load_weights is not callable.")
            return

        @wraps(original_load_weights)
        def wrapped_load_weights(*args, **kwargs):
            logger.info_once("Before calling self.model.load_weights")
            original_post_weight_load = getattr(model, "post_weight_load", None)
            suppress_post_weight_load = callable(original_post_weight_load)
            if suppress_post_weight_load:
                setattr(model, "post_weight_load", lambda *_, **__: None)
            try:
                if self.model_config.enable_sleep_mode:
                    allocator = NpuMemAllocator.get_instance()
                    context = allocator.use_memory_pool(tag="weights")
                else:
                    context = nullcontext()
                with context, set_current_vllm_config(self.vllm_config):
                    original_load_weights(*args, **kwargs)

            finally:
                if suppress_post_weight_load:
                    setattr(model, "post_weight_load", original_post_weight_load)
                logger.info_once("After calling self.model.load_weights")

        model.load_weights = wrapped_load_weights
        model._omni_npu_load_weights_hooked = True

    def _model_post_weight_load(self, model: nn.Module | None) -> None:
        if model is None:
            return
        post_weight_load = getattr(model, "post_weight_load", None)
        if callable(post_weight_load):
            post_weight_load()

    def model_post_weight_load(self) -> None:
        if self.model_config.enable_sleep_mode:
            allocator = NpuMemAllocator.get_instance()
            context = allocator.use_memory_pool(tag="weights")
        else:
            context = nullcontext()
        with context, set_current_vllm_config(self.vllm_config):
            self._model_post_weight_load(self.get_model())
            self._model_post_weight_load(self.get_drafter_model())

    def load_model(self, eep_scale_up: bool = False) -> None:
        """
        Args:
            eep_scale_up: the model loading is for elastic EP scale up.
        """
        logger.debug(f"<<< {self.vllm_config.npu_compilation_config.use_gegraph=}")
        if self.vllm_config.npu_compilation_config.use_gegraph:
            from vllm.model_executor.model_loader import get_model as original_get_model
            self.model = original_get_model(vllm_config=self.vllm_config)
            return
        super().load_model(eep_scale_up)

        if hasattr(self.model, "model"):
            prefetch_post_load_hook = getattr(self.model.model, "prefetch_post_load", None)
            if callable(prefetch_post_load_hook):
                prefetch_post_load_hook()

        if hasattr(self, "drafter") and isinstance(self.drafter, EagleProposer):
            prepare_communication_buffer_for_model(self.drafter.model)

        # wrap the model with full graph wrapper if needed.
        logger.debug(f"<<< {self.compilation_config.cudagraph_mode.has_full_cudagraphs()=}")
        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            set_graph_params(self.compilation_config.cudagraph_capture_sizes)
            self.update_stream: torch.npu.Stream = torch.npu.Stream()
            
            attn_layer_names = set(get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase).keys())
            if hasattr(self, "drafter") and isinstance(self.drafter, EagleProposer):
                attn_layer_names = attn_layer_names - set(self.drafter.attn_layer_names)
            attn_layer_names = list(attn_layer_names)

            self.model = ACLGraphWrapper(self.model.runnable,
                                         self.vllm_config,
                                         runtime_mode=CUDAGraphMode.FULL,
                                         update_stream=self.update_stream,
                                         attn_layer_names=attn_layer_names,
                                        )
            logger.debug("<<< Wrapped original model with ACLGraphWrapper")
            if hasattr(self, "drafter") and isinstance(self.drafter, EagleProposer):
                n_predict = getattr(self.drafter, "n_predict", 1)
                if n_predict == 1:
                    self.draft_update_stream: torch.npu.Stream = torch.npu.Stream()
                    self.drafter.model = ACLGraphWrapper(
                        self.drafter.model,
                        self.vllm_config,
                        runtime_mode=CUDAGraphMode.FULL,
                        update_stream=self.draft_update_stream,
                        attn_layer_names=self.drafter.attn_layer_names,
                    )
                    logger.debug("<<< Wrapped drafter model with ACLGraphWrapper")
                else:
                    mtp_start_layer_idx = self.drafter.model.config.num_hidden_layers
                    self.draft_update_stream: torch.npu.Stream = torch.npu.Stream()
                    
                    wrapped_layers = dict()
                    for i in range(n_predict):
                        mtp_layer_i = mtp_start_layer_idx + i
                        mtp_layer_i_attn_names = [
                            item for item in self.drafter.attn_layer_names
                            if self.drafter.model.get_spec_layer(item) == mtp_layer_i
                        ]
                        wrapped_layers[str(mtp_layer_i)] = ACLGraphWrapper(
                            self.drafter.model.model.layers[str(mtp_layer_i)],
                            self.vllm_config,
                            runtime_mode=CUDAGraphMode.FULL,
                            update_stream=self.draft_update_stream,
                            attn_layer_names=mtp_layer_i_attn_names,
                        )
                    self.drafter.model.model.wrapped_layers = wrapped_layers
                    logger.debug("<<< Wrapped multi mtp layers of drafter model with ACLGraphWrapper")
        self._is_mm_encoder_only = supports_mm_encoder_only(self.model)
        self._hook_model_load_weights(self.get_model())
        self._hook_model_load_weights(self.get_drafter_model())

    def _get_eagle3_aux_layers_from_config(self) -> tuple[int, ...] | None:
        if not (self.speculative_config and self.speculative_config.draft_model_config):
            return None

        hf_config = self.speculative_config.draft_model_config.hf_config

        layer_ids = getattr(hf_config, "eagle_aux_hidden_state_layer_ids", None)
        if layer_ids and isinstance(layer_ids, (list, tuple)):
            return tuple(layer_ids)

        eagle_config = getattr(hf_config, "eagle_config", None)
        if not isinstance(eagle_config, dict):
            return None

        layer_ids = eagle_config.get("eagle_aux_hidden_state_layer_ids")
        if layer_ids and isinstance(layer_ids, (list, tuple)):
            return tuple(layer_ids)

        return None

    def reset_input_batch(self) -> None:
        self.input_batch.block_table.clear()
        for block_table in self.input_batch.block_table.block_tables:
            block_table.slot_mapping.gpu.fill_(0)
            block_table.slot_mapping.cpu.fill_(0)

    def capture_model(self) -> int:
        logger.debug("<<< Capturing model in npu_model_runner")
        if self.vllm_config.npu_compilation_config.use_gegraph:
            logger.info(f"<<< capture_model use gegraph, dummy_run max_num_reqs={self.max_num_reqs}")
            self._dummy_run(self.max_num_reqs, force_attention=True, uniform_decode=True)
            return
        if consume_aclgraph_recapture():
            self.reset_input_batch()
            self._mark_aclgraph_wrappers_for_recapture()
        with switch_torch_device():
            super().capture_model()

    def _execute_mm_encoder(self, scheduler_output: "SchedulerOutput"):
        # Mitigation for the upstream VLLM bug where the MM encoder output
        # cache is never freed after _execute_mm_encoder, causing the cache to
        # grow indefinitely and eventually OOM. The encoder cache is redundant
        # for cross-attention models like Whisper and BART because the encoder
        # output is passed directly to the decoder. The better fix is to avoid
        # populating the cache in the first place, but that requires a larger
        # upstream change.
        encoder_outputs = super()._execute_mm_encoder(scheduler_output)
        if not self.model_config.is_encoder_decoder:
            return encoder_outputs

        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            req_state = self.requests[req_id]
            for mm_input_id in encoder_input_ids:
                mm_hash = req_state.mm_features[mm_input_id].identifier
                self.encoder_cache.pop(mm_hash, None)

        return encoder_outputs

    def _iter_aclgraph_wrappers(self):
        if isinstance(self.model, ACLGraphWrapper):
            yield self.model

        if not (hasattr(self, "drafter") and isinstance(self.drafter, EagleProposer)):
            return

        drafter_model = getattr(self.drafter, "model", None)
        if isinstance(drafter_model, ACLGraphWrapper):
            yield drafter_model

        wrapped_layers = getattr(getattr(drafter_model, "model", None),
                                 "wrapped_layers", None)
        if isinstance(wrapped_layers, dict):
            for layer in wrapped_layers.values():
                if isinstance(layer, ACLGraphWrapper):
                    yield layer

    def _mark_aclgraph_wrappers_for_recapture(self) -> None:
        for wrapper in self._iter_aclgraph_wrappers():
            wrapper.recapture = True

    @torch.inference_mode()
    @model_output_decorator
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, AsyncModelRunnerOutput, IntermediateTensors]:
        if ENABLE_NPU_PENALTY_CACHE and not isinstance(self.input_batch, NPUInputBatch):
            self.input_batch.__class__ = NPUInputBatch
            self.input_batch.init_npu_tensors(self.model_config.get_vocab_size())
        if ENABLE_NPU_PENALTY_CACHE:
            self.sampler.npu_input_batch = self.input_batch

        with (switch_torch_device()
              if self.use_async_scheduling else nullcontext()):
            # To avoid a bug of vllm which uses self.drafter in pp stages without a drafter
            if not hasattr(self, "drafter"):
                self.drafter = None
            res = super().execute_model(scheduler_output,
                                         intermediate_tensors)
            if self.drafter is None:
                del self.drafter
            return res

    @torch.inference_mode
    @model_output_decorator
    def sample_tokens(self, grammar_output):
        with switch_torch_device():
            return super().sample_tokens(grammar_output)

    # When async scheduling, valid_sampled_token_count_cpu already carries
    # the same information via its own async copy stream. num_accepted_tokens
    # is derived in _get_valid_sampled_token_count instead.
    def _update_states_after_model_execute(
        self, output_token_ids: torch.Tensor
    ) -> None:
        
        if self.use_async_scheduling:
            return

        return super()._update_states_after_model_execute(output_token_ids)

    # Override to also propagate num_accepted_tokens_cpu from the async-copied
    # valid_sampled_token_count. This runs during execute_model -> _update_requests,
    # before _build_attention_metadata, so self.runner.num_accepted_tokens.gpu will
    # see correct values.
    def _get_valid_sampled_token_count(self) -> list[int]:
        prev_sampled_token_ids = self.input_batch.prev_sampled_token_ids
        sampled_count_event = self.valid_sampled_token_count_event
        if sampled_count_event is None or prev_sampled_token_ids is None:
            return []

        counts_cpu = self.valid_sampled_token_count_cpu
        assert counts_cpu is not None
        sampled_count_event.synchronize()

        num_reqs = prev_sampled_token_ids.shape[0]
        self.input_batch.num_accepted_tokens_cpu[: num_reqs] = counts_cpu[: num_reqs]
        return counts_cpu[: num_reqs].tolist()

    def get_model(self) -> nn.Module:
        # get raw model out of the aclgraph wrapper.
        if isinstance(self.model, ACLGraphWrapper):
            return self.model.unwrap()
        return self.model

    def get_drafter_model(self) -> nn.Module | None:
        if hasattr(self, "drafter") and isinstance(self.drafter, EagleProposer):
            if isinstance(self.drafter.model, ACLGraphWrapper):
                return self.drafter.model.unwrap()
            return self.drafter.model
        return None

    def _capture_dp_pad_target(self, forward_context) -> None:
        """Stash the DP all_gather pad target on the NPUParallelLMHead
        class — same value across all instances (max sample count across
        DP). Read from forward_context.dp_metadata.max_tokens_across_dp_cpu
        (CPU tensor, int() is free). Called inside set_forward_context;
        compute_logits afterwards reads the class attribute and skips its
        own per-call DP size-allgather + host sync."""
        if not (self.dp_parallel_lmhead or self.local_parallel_lmhead):
            return
        dp_meta = getattr(forward_context, "dp_metadata", None)
        if dp_meta is None:
            return
        if self.local_parallel_lmhead:
            from omni_npu.v1.distributed.parallel_state_ext import (
                get_local_world_group,
            )
            local_group = get_local_world_group()
            num_tokens = dp_meta.num_tokens_across_dp_cpu
            NPUParallelLMHead._dp_pad_n = max(
                int(num_tokens[r]) for r in local_group.ranks
            )
        else:
            NPUParallelLMHead._dp_pad_n = int(dp_meta.max_tokens_across_dp_cpu)

    # --- DP lm_head sync helper -------------------------------------------
    # When dp_parallel_lmhead is on, NPULogitsProcessor._get_logits issues
    # DP collectives (allgather + all_to_all) inside compute_logits. Active
    # ranks hit these naturally; idle DP ranks go through _dummy_run, which
    # doesn't call compute_logits on its own, so we trigger it here to match
    # the active rank's order. Drafter-side sync happens inside the drafter
    # dummy_run loop (patch_eagle.py), pairing forward+compute_logits per
    # spec iteration with active rank's propose.

    def _dp_sync_main_compute_logits(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: np.ndarray,
        is_profile: bool,
    ) -> None:
        if not (self.dp_parallel_lmhead or self.local_parallel_lmhead) or is_profile:
            return
        logit_indices = torch.from_numpy(
            np.cumsum(num_scheduled_tokens) - 1
        ).to(self.device, non_blocking=True)
        sample_hs = hidden_states[logit_indices]
        self.model.compute_logits(sample_hs)

    @torch.inference_mode()
    def sync_and_slice_intermediate_tensors(
        self,
        num_tokens: int,
        intermediate_tensors: IntermediateTensors | None,
        sync_self: bool,
    ) -> IntermediateTensors:
        assert sync_self == True
        return intermediate_tensors

    @torch.inference_mode()
    def sync_and_slice_intermediate_tensors_dummy_run(
        self,
        num_tokens: int,
        intermediate_tensors: IntermediateTensors | None,
        sync_self: bool,
    ) -> IntermediateTensors:
        assert sync_self == False
        assert self.intermediate_tensors is not None

        tp = self.vllm_config.parallel_config.tensor_parallel_size
        is_rs = model_extra_config.parall_config.ena_seq_parallel

        return IntermediateTensors(
            {
                k: v[: num_tokens // tp]
                if is_rs
                else v[:num_tokens]
                for k, v in self.intermediate_tensors.items()
            }
        )
        return intermediate_tensors

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        remove_lora: bool = True,
        activate_lora: bool = False,
        is_graph_capturing: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a dummy forward pass to warm up/profile run or capture the
        CUDA graph for the model.

        Args:
            num_tokens: Number of tokens to run the dummy forward pass.
            cudagraph_runtime_mode: used to control the behavior.
                - if not set will determine the cudagraph mode based on using
                    the self.cudagraph_dispatcher.
                - CUDAGraphMode.NONE: No cudagraph, for warm up and profile run
                - CUDAGraphMode.PIECEWISE: Piecewise cudagraph.
                - CUDAGraphMode.FULL: Full cudagraph, attention metadata is
                    needed.
            force_attention: If True, always create attention metadata. Used to
                warm up attention backend when mode is NONE.
            uniform_decode: If True, the batch is a uniform decode batch.
            skip_eplb: If True, skip EPLB state update.
            is_profile: If True, this is a profile run.
            create_mixed_batch: If True, create a mixed batch with both decode
                (1 token) and prefill (multiple tokens) requests.
            remove_lora: If False, dummy LoRAs are not destroyed after the run
            activate_lora: If False, dummy_run is performed without LoRAs.
        """
        if self._is_mm_encoder_only:
            # The current dummy run only covers LM execution, so we can skip it.
            # mm encoder dummy run may need to add in the future.
            return torch.tensor([]), torch.tensor([])

        assert (
            cudagraph_runtime_mode is None
            or cudagraph_runtime_mode.valid_runtime_modes()
        )

        # If cudagraph_mode.decode_mode() == FULL and
        # cudagraph_mode.separate_routine(). This means that we are using
        # different graphs and/or modes for mixed prefill-decode batches vs.
        # uniform decode batches. A uniform decode batch means that all
        # requests have identical query length, except a potential virtual
        # request (shorter) in the batch account for padding.
        # Uniform decode batch could either be common pure decode, where
        # max_query_len == 1, or speculative decode, where
        # max_query_len == 1 + num_spec_decode_tokens.

        # When setting max_query_len = 1, we switch to and capture the optimized
        # routine of FA2 for pure decode, i.e., Flashdecode + an optimization
        # for GQA/MQA.
        max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        if create_mixed_batch:
            assert not uniform_decode
            # Create mixed batch:
            # first half decode tokens, second half one prefill
            num_decode_tokens = min(max_num_reqs - 1, num_tokens // 2)
            num_prefill_tokens = num_tokens - num_decode_tokens
            num_reqs = num_decode_tokens + 1

            # Create decode requests (1 token each) followed by prefill request
            num_scheduled_tokens_list = [1] * num_decode_tokens + [num_prefill_tokens]
            # Note: Overriding max_query_len to be the prefill tokens
            max_query_len = num_prefill_tokens
        elif uniform_decode:
            assert not create_mixed_batch
            num_reqs = min(max_num_reqs, cdiv(num_tokens, max_query_len))
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        else:
            num_reqs = min(num_tokens, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            for i in range(num_tokens % num_reqs):
                num_scheduled_tokens_list[i] += 1
            max_query_len = num_scheduled_tokens_list[0]

        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
        num_tokens_unpadded = int(num_scheduled_tokens.sum())

        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)

        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        logit_indices_device = torch.from_numpy(logit_indices).to(
            self.device, non_blocking=True
        )

        _cudagraph_mode, batch_desc, should_ubatch, num_tokens_across_dp, _ = (
            self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens,
                max_num_scheduled_tokens=max_query_len,
                use_cascade_attn=False,
                allow_microbatching=allow_microbatching,
                force_eager=is_profile
                or (cudagraph_runtime_mode == CUDAGraphMode.NONE),
                # `force_uniform_decode` is used for cudagraph capture; because for
                # capturing mixed prefill-decode batches, we sometimes use
                # num_tokens == num_reqs which looks like a uniform decode batch to the
                # dispatcher; but we actually want to capture a piecewise cudagraph
                force_uniform_decode=uniform_decode,
                # `force_has_lora` is used for cudagraph capture; because LoRA is
                # activated later in the context manager, but we need to know the
                # LoRA state when determining the batch descriptor for capture
                force_has_lora=activate_lora,
            )
        )

        if cudagraph_runtime_mode is None:
            cudagraph_runtime_mode = _cudagraph_mode
        else:
            assert cudagraph_runtime_mode == _cudagraph_mode, (
                f"Cudagraph runtime mode mismatch in dummy_run. "
                f"Expected {_cudagraph_mode}, but got {cudagraph_runtime_mode}."
            )

        num_tokens_padded = batch_desc.num_tokens
        num_reqs_padded = (
            batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
        )
        ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
            should_ubatch,
            num_scheduled_tokens,
            num_tokens_padded,
            num_reqs_padded,
            self.vllm_config.parallel_config.num_ubatches,
        )
        logger.debug(
            "ubatch_slices: %s, ubatch_slices_padded: %s",
            ubatch_slices,
            ubatch_slices_padded,
        )

        attn_metadata: PerLayerAttnMetadata | None = None

        # If force_attention is True, we always capture attention. Otherwise,
        # it only happens for cudagraph_runtime_mode=FULL.
        if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
            if create_mixed_batch:
                # In the mixed batch mode (used for FI warmup), we use
                # shorter sequence lengths to run faster.
                # TODO(luka) better system for describing dummy batches
                seq_lens = [1] * num_decode_tokens + [num_prefill_tokens + 1]
            else:
                seq_lens = max_query_len  # type: ignore[assignment]
            self.seq_lens.np[:num_reqs] = seq_lens
            self.seq_lens.np[num_reqs:] = 0
            self.seq_lens.copy_to_gpu()

            cum_num_tokens, _ = self._get_cumsum_and_arange(num_scheduled_tokens)
            self.query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens
            #TODO check deepseek model 
            self.query_start_loc.np[num_reqs + 1 :].fill(cum_num_tokens[-1])
            self.query_start_loc.copy_to_gpu()

            pad_attn = cudagraph_runtime_mode == CUDAGraphMode.FULL
            attn_metadata, _ = self._build_attention_metadata(
                num_tokens=num_tokens_unpadded,
                num_tokens_padded=num_tokens_padded,
                num_reqs=num_reqs_padded,
                max_query_len=max_query_len,
                ubatch_slices=ubatch_slices_padded if pad_attn else ubatch_slices,
                for_cudagraph_capture=is_graph_capturing,
            )

        with self.maybe_dummy_run_with_lora(
            self.lora_config,
            num_scheduled_tokens,
            num_sampled_tokens,
            activate_lora,
            remove_lora,
        ):
            # Make sure padding doesn't exceed max_num_tokens
            assert num_tokens_padded <= self.max_num_tokens
            model_kwargs = self._init_model_kwargs()
            if self.supports_mm_inputs and not self.model_config.is_encoder_decoder:
                input_ids, inputs_embeds = self._prepare_mm_inputs(num_tokens_padded)

                model_kwargs = {
                    **model_kwargs,
                    **self._dummy_mm_kwargs(num_reqs),
                }
            elif self.enable_prompt_embeds:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
                model_kwargs = self._init_model_kwargs()
            else:
                input_ids = self.input_ids.gpu[:num_tokens_padded]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions.gpu[:, :num_tokens_padded]
            elif self.uses_xdrope_dim > 0:
                positions = self.xdrope_positions.gpu[:, :num_tokens_padded]
            else:
                positions = self.positions.gpu[:num_tokens_padded]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=self.max_num_tokens,
                            dtype=self.model_config.dtype,
                            device=self.device,
                        )
                    )

                intermediate_tensors = self.sync_and_slice_intermediate_tensors_dummy_run(
                    num_tokens_padded, None, False
                )

            if ubatch_slices_padded is not None:
                # Adjust values to reflect a single ubatch.
                # TODO(sage,lucas): this is cruft that should be addressed in
                #  the padding refactor.
                num_tokens_padded = ubatch_slices_padded[0].num_tokens
                if num_tokens_across_dp is not None:
                    num_tokens_across_dp[:] = num_tokens_padded

            with (
                self.maybe_randomize_inputs(input_ids, inputs_embeds),
                set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens_padded,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_desc,
                    ubatch_slices=ubatch_slices_padded,
                ),
            ):
                if (
                    self.router_sliding_window > 1 
                    and not model_extra_config.operator_opt_config.use_noncontiguous_kv
                ):
                    self._build_conv_context(dummy=True)
                forward_context = get_forward_context()
                forward_context.capturing = False
                self._capture_dp_pad_target(forward_context)
                outputs = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )

            if self.use_aux_hidden_state_outputs:
                hidden_states, _ = outputs
            else:
                hidden_states = outputs

            self._dp_sync_main_compute_logits(
                hidden_states, num_scheduled_tokens, is_profile)

            if self.speculative_config and get_pp_group().is_last_rank and self.speculative_config.use_eagle():
                assert isinstance(self.drafter, EagleProposer)
                # Adapt start: enable mtp acl graph mode
                use_cudagraphs = (
                    (
                        is_graph_capturing
                        and cudagraph_runtime_mode == CUDAGraphMode.FULL
                    )
                    or (
                        not is_graph_capturing
                        and cudagraph_runtime_mode != CUDAGraphMode.NONE
                    )
                ) and not self.speculative_config.enforce_eager
                # Adapt end: enable mtp acl graph mode

                # Note(gnovack) - We need to disable cudagraphs for one of the two
                # lora cases when cudagraph_specialize_lora is enabled. This is a
                # short term mitigation for issue mentioned in
                # https://github.com/vllm-project/vllm/issues/28334
                if self.compilation_config.cudagraph_specialize_lora and activate_lora:
                    use_cudagraphs = False

                # Adapt start: to pass attn_metadata
                self.drafter.dummy_run(
                    attn_metadata,
                    num_tokens,
                    use_cudagraphs=use_cudagraphs,
                    is_graph_capturing=is_graph_capturing,
                    is_profile=is_profile,
                )
                # Adapt end: to pass attn_metadata

        # We register layerwise NVTX hooks here after the first dynamo tracing is
        # done to avoid nvtx operations in hook functions being traced by
        # torch dynamo and causing graph breaks.
        # Note that for DYNAMO_ONCE and VLLM_COMPILE mode,
        # compiled model's dynamo tracing is only done once and the compiled model's
        # __call__ function is replaced by calling the compiled function.
        # So it's safe to register hooks here. Hooks will be registered to
        # both compiled and uncompiled models but they will never
        # be called on the compiled model execution path.
        self._register_layerwise_nvtx_hooks()

        # This is necessary to avoid blocking DP.
        # For dummy runs, we typically skip EPLB since we don't have any real
        # requests to process.
        # However, in DP settings, there may be cases when some DP ranks do
        # not have any requests to process, so they're executing dummy batches.
        # In such cases, we still have to trigger EPLB to make sure
        # ranks execute the rearrangement in synchronization.
        if not skip_eplb:
            self.eplb_step(is_dummy=True, is_profile=is_profile)

        return hidden_states, hidden_states[logit_indices_device]

    @prepare_inputs_decorator
    def prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        num_tokens_after_padding: int,
    ) -> "InputBatch":
        input_batch = super().prepare_inputs(scheduler_output, num_tokens_after_padding)

        return input_batch

    @prepare_inputs_decorator
    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[
        torch.Tensor,
        SpecDecodeMetadata | None,
    ]:
        (logits_indices, spec_decode_metadata) = super()._prepare_inputs(scheduler_output, num_scheduled_tokens)

        return (logits_indices, spec_decode_metadata)

    @init_config_decorator
    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config
        self.may_add_encoder_only_layers_to_kv_cache_config()
        self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)
        self.initialize_attn_backend(kv_cache_config)
        # The kernel block size for all KV cache groups. For example, if
        # kv_cache_manager uses block_size 256 for a given group, but the attention
        # backends for that group only supports block_size 64, we will return
        # kernel_block_size 64 and split the 256-token-block to 4 blocks with 64
        # tokens each.
        kernel_block_sizes = self._prepare_kernel_block_sizes(kv_cache_config)

        # create metadata builders
        self.initialize_metadata_builders(kv_cache_config, kernel_block_sizes)

        # Reinitialize need to after initialize_attn_backend
        self.may_reinitialize_input_batch(kv_cache_config, kernel_block_sizes)
        kv_caches = self.initialize_kv_cache_tensors(
            kv_cache_config, kernel_block_sizes
        )

        if self.speculative_config and get_pp_group().is_last_rank and self.speculative_config.use_eagle():
            assert isinstance(self.drafter, EagleProposer)
            # validate all draft model layers belong to the same kv cache
            # group
            self.drafter.validate_same_kv_cache_group(kv_cache_config)

        if has_kv_transfer_group():
            kv_transfer_group = get_kv_transfer_group()
            if self.cross_layers_kv_cache is not None:
                assert self.cross_layers_attn_backend is not None
                kv_transfer_group.register_cross_layers_kv_cache(
                    self.cross_layers_kv_cache, self.cross_layers_attn_backend
                )
            else:
                kv_transfer_group.register_kv_caches(kv_caches)
            kv_transfer_group.set_host_xfer_buffer_ops(copy_kv_blocks)

        if self.model_config.enable_return_routed_experts:
            self.init_routed_experts_capturer()

    @reinitialize_input_batch_decorator
    def may_reinitialize_input_batch(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> None:
        super().may_reinitialize_input_batch(kv_cache_config, kernel_block_sizes)

    def initialize_kv_cache_tensors(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> dict[str, torch.Tensor]:
        kv_caches = super().initialize_kv_cache_tensors(kv_cache_config, kernel_block_sizes)
        if has_kv_transfer_group():
            self.kv_caches_dict = kv_caches
        return kv_caches
    
    def kv_cache_after_wake_up(self) -> None:
        attn_layers = self.compilation_config.static_forward_context
        if self.model_config.enable_sleep_mode:
            from vllm.model_executor.layers.attention.static_sink_attention import StaticSinkAttention
            sink_mla_available = False
            try:
                from vllm.model_executor.layers.attention.static_sink_attention import StaticSinkMLAAttention
                sink_mla_available = True
            except ImportError:
                logger.warning("StaticSinkMLAAttention has not been defined, skipping...")
            for name, module in attn_layers.items():
                if isinstance(module, StaticSinkAttention) or (sink_mla_available 
                                                               and isinstance(module, StaticSinkMLAAttention)):
                    self._kv_cache_sink_attn_after_wake_up(module)

    def _kv_cache_sink_attn_after_wake_up(self, module) -> None:
        sink_kv_cache = getattr(module, "kv_cache")

        # populate_sink_kv in SinkAttention retrieves the `virtual_engine` value from `ForwardContext`
        # but this value is unavailable here.
        # Since `virtual_engine` is defaulted to 0 and will be deprecated, we directly set it to 0 here.
        self_kv_cache = sink_kv_cache[0]
        if self_kv_cache is not None and len(self_kv_cache) > 0:
            if hasattr(module, "maybe_populate_sink_kv_after_wakeup"):
                populate_sink_kv_method = getattr(module, "maybe_populate_sink_kv_after_wakeup")
                populate_sink_kv_method(self_kv_cache[0], self_kv_cache[1])
            else:
                populate_sink_kv_method = getattr(module, "populate_sink_kv")
                populate_sink_kv_method(self_kv_cache[0], self_kv_cache[1])

        for kv_cache_group_id in range(len(self.kv_cache_config.kv_cache_groups)):
            for attn_group in self.attn_groups[kv_cache_group_id]:
                for attn_builder in attn_group.metadata_builders:
                    if hasattr(attn_builder, "reinit_block_table_with_sink"):
                        attn_builder.reinit_block_table_with_sink()

    def unregister_kv_caches(self):
        if self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.kv_connector == "LLMDataDistConnector":
            if has_kv_transfer_group():
                logger.info(f"unregister_kv_caches")
                get_kv_transfer_group().unregister_kv_caches()

    def reregister_kv_caches(self):
        if self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.kv_connector == "LLMDataDistConnector":
            if has_kv_transfer_group():
                logger.info(f"reregister_kv_caches")
                get_kv_transfer_group().register_kv_caches(self.kv_caches_dict)

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        if not self.num_spec_tokens or not self._draft_token_req_ids:
            return None
        draft_token_ids, req_ids = self._get_draft_token_ids_cpu()
        num_reqs = len(req_ids)
        mask_list = self.discard_request_mask.cpu[:num_reqs].tolist()
        filtered_req_ids = []
        filtered_draft_token_ids = []
        for req_id, tokens, discard in zip(req_ids, draft_token_ids, mask_list):
            if discard:
                continue
            if any(t < 0 for t in tokens):
                logger.warning(
                    "Dropping invalid draft token id(s) for request %s: %s",
                    req_id, tokens,
                )
                continue
            filtered_req_ids.append(req_id)
            filtered_draft_token_ids.append(tokens)
        if len(filtered_req_ids) == 0:
            return None
        return DraftTokenIds(filtered_req_ids, filtered_draft_token_ids)
