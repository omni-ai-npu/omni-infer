# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from contextlib import nullcontext
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Union, Any, TypeAlias

import torch
import numpy as np
import torch.nn as nn
from functools import wraps
from unittest.mock import patch

import vllm.envs as envs

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
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    initialize_mamba_ssu_backend,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv
from vllm.tracing import instrument
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.attention.backend import (
    AttentionMetadata,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    MambaSpec,
)
from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    DraftTokenIds,
    ModelRunnerOutput,
)
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.utils import PADDING_SLOT_ID
from vllm.v1.worker.ubatch_utils import maybe_create_ubatch_slices
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.utils import prepare_kernel_block_sizes
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.v1.worker.cp_utils import (
    get_total_cp_world_size,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

from omni_npu.attention.backends.mome import bind_num_prompt_tokens
from omni_npu.compilation.acl_graph import (
    ACLGraphWrapper,
    consume_aclgraph_recapture,
    reset_stale_aclgraph_resources,
    set_graph_params,
)
from omni_npu.configs import OmniAdditionalConfig
from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.sample.sampler import NPUSamplerV1, ENABLE_NPU_PENALTY_CACHE
from omni_npu.sample.rejection_sampler import NPURejectionSampler
from omni_npu.plugin_decorators import (
    init_config_decorator,
    prepare_inputs_decorator,
    model_output_decorator,
    reinitialize_input_batch_decorator,
)
from omni_npu.v1.layers.vocab_parallel_embedding import NPUParallelLMHead
from omni_npu.v1.utils import switch_torch_device
from omni_npu.worker.npu_input_batch import NPUInputBatch
from omni_npu.worker.npu_mem_pool import NpuMemAllocator
from omni_npu.connector.kv_dump import maybe_dump_kv

AttnMetadataDict: TypeAlias = dict[str, AttentionMetadata]
# list when ubatching is enabled
PerLayerAttnMetadata: TypeAlias = list[AttnMetadataDict] | AttnMetadataDict


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
            (self.max_num_reqs, 1),
            dtype=torch.int64,
            device="cpu",
            pin_memory=PIN_MEMORY,
        )

        # FIXME(runze): reusing VLLM's sampler fails, this sampler class is from omni_infer.
        # need to check why and try to remove it.
        self.sampler = NPUSamplerV1()

        if self.speculative_config and get_pp_group().is_last_rank:
            self.rejection_sampler = NPURejectionSampler(self.sampler)

        if vllm_config.additional_config is not None:
            from omni_npu.compilation.npugraph_ex_config import init_aclgraph_config
            init_aclgraph_config(vllm_config)
            omni_add = OmniAdditionalConfig.from_vllm_config(vllm_config)
            self.combine_block = omni_add.combine_block
        else:
            self.combine_block = 1
        self.use_spec_decode = False
        num_tokens_per_reqs_decode = (
            1 if not self.use_spec_decode
            else (1 + self.speculative_config.num_speculative_tokens)
        )
        self.block_size = vllm_config.cache_config.block_size
        self.max_num_blocks_per_req = (
            cdiv(self.model_config.max_model_len,
                 self.block_size * self.combine_block) * self.combine_block
        )
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
            self.inputs_embeds.gpu = self.inputs_embeds.gpu.repeat(
                1, hf_config.mhc_num_stream
            )

        self.batch_execution_and_padding_state: tuple[
            CUDAGraphMode,
            BatchDescriptor,
            torch.Tensor | None,
        ] | None = None

        self._is_mm_encoder_only = False

        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"
        self.exec_count = 0

        # TODO: penalty cache feature need to adapt vllm 0.25.1
        # self._init_npu_input_batch()

    def _init_npu_input_batch(
        self,
        block_sizes=None,
        kernel_block_sizes=None,
    ):
        self.input_batch = NPUInputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=max(self.max_model_len, self.max_encoder_len),
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.cache_config.block_size] if block_sizes is None else block_sizes,
            kernel_block_sizes=[self.cache_config.block_size] if kernel_block_sizes is None else kernel_block_sizes,
            num_spec_tokens=self.num_spec_tokens,
            logitsprocs=self.input_batch.logitsprocs,
            logitsprocs_need_output_token_ids=self.input_batch.logitsprocs_need_output_token_ids,
            is_pooling_model=self.is_pooling_model,
            reasoning_config=self.vllm_config.reasoning_config,
        )
        if ENABLE_NPU_PENALTY_CACHE:
            self.input_batch.init_penalty_cache(
                self.model_config.get_vocab_size(), self.device,
            )
            self.sampler.input_batch = self.input_batch
        if hasattr(self, "drafter") and getattr(self.drafter, "fix_multi_mtp_kvcache", False):
            self.input_batch.init_target_model_hidden_states_cache(
                getattr(self.drafter, "n_predict", 1),
                getattr(self.drafter, "hidden_size", 1),
                self.dtype,
                self.device,
            )
            self.drafter.input_batch = self.input_batch

    @reinitialize_input_batch_decorator
    def may_reinitialize_input_batch(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> None:
        """
        Re-initialize the input batch if the block sizes are different from
        `[self.cache_config.block_size]`. This usually happens when there
        are multiple KV cache groups.

        Args:
            kv_cache_config: The KV cache configuration.
            kernel_block_sizes: The kernel block sizes for each KV cache group.
        """
        block_sizes = []
        max_num_blocks = []
        max_model_len = max(self.max_model_len, self.max_encoder_len)
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            if isinstance(kv_cache_group.kv_cache_spec, EncoderOnlyAttentionSpec):
                continue
            block_size = kv_cache_group.kv_cache_spec.block_size
            block_sizes.append(block_size)
            max_num_blocks_per_req = cdiv(
                max_model_len, block_size * get_total_cp_world_size()
            )
            if isinstance(kv_cache_group.kv_cache_spec, MambaSpec):
                max_num_blocks_per_req = (
                    max_num_blocks_per_req
                    if self.cache_config.enable_prefix_caching
                    else 1
                ) + kv_cache_group.kv_cache_spec.num_speculative_blocks
            max_num_blocks.append(max_num_blocks_per_req)

        if (
            block_sizes != self._init_block_sizes
            or kernel_block_sizes != self._init_kernel_block_sizes
        ):
            self._init_block_sizes = block_sizes
            self._init_kernel_block_sizes = kernel_block_sizes
            self._init_npu_input_batch(block_sizes, kernel_block_sizes)

        assert self._init_block_sizes == block_sizes, (
            f"InputBatch block_sizes {self._init_block_sizes} != "
            f"kv_cache block_sizes {block_sizes}"
        )
        assert self._init_kernel_block_sizes == kernel_block_sizes, (
            f"InputBatch kernel_block_sizes {self._init_kernel_block_sizes} "
            f"!= kv_cache kernel_block_sizes {kernel_block_sizes}"
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
        forward_context = get_forward_context()
        forward_context.capturing = False
        self._capture_dp_pad_target(forward_context)
        if self.is_debugging_mode:
            self.exec_count += 1
            logger.debug(f"Executing model forward {self.exec_count=}")
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )

    def iter_kv_cache_attn_groups(self):
        """Public iterator over KV cache groups with attention backends."""
        yield from self._kv_cache_spec_attn_group_iterator()

    def _reshape_kv_cache_tensors(
        self,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
        kernel_block_sizes: list[int],
    ) -> dict[str, torch.Tensor]:
        kv_caches: dict[str, torch.Tensor] = {}
        has_tensor, has_tuple = False, False
        for group in self.iter_kv_cache_attn_groups():
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
        for group in self.iter_kv_cache_attn_groups():
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
        force_num_active_loras: int | None = None,
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

        # Compute LoRA state for cudagraph dispatch
        num_active_loras = (
            force_num_active_loras
            if force_num_active_loras is not None
            else len(self.input_batch.lora_id_to_lora_request)
        )
        has_lora = num_active_loras > 0 if force_has_lora is None else force_has_lora

        num_tokens_padded = self._pad_for_sequence_parallelism(num_tokens)

        def dispatch_cudagraph(
            num_tokens: int,
            disable_full: bool = False,
            valid_modes=None,
        ):
            if force_eager:
                return (CUDAGraphMode.NONE, BatchDescriptor(num_tokens_padded))
            return self.cudagraph_dispatcher.dispatch(
                num_tokens=num_tokens,
                has_lora=has_lora,
                uniform_decode=uniform_decode,
                num_active_loras=num_active_loras,
                valid_modes={CUDAGraphMode.NONE} if force_eager else valid_modes,
                invalid_modes={CUDAGraphMode.FULL} if disable_full else None,
            )

        cudagraph_mode, batch_descriptor = dispatch_cudagraph(
            num_tokens_padded, disable_full=use_cascade_attn or has_encoder_output
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

            should_ubatch, num_tokens_across_dp, synced_cudagraph_mode = (
                coordinate_batch_across_dp(
                    num_tokens_unpadded=num_tokens,
                    parallel_config=self.parallel_config,
                    allow_microbatching=allow_microbatching,
                    num_tokens_padded=num_tokens_padded,
                    uniform_decode=uniform_decode,
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
                    valid_modes={CUDAGraphMode(synced_cudagraph_mode)},
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

    @instrument(span_name="Loading (GPU)")
    def load_model(self, load_dummy_weights: bool = False) -> None:
        """
        Args:
            load_dummy_weights: load dummy weights instead of real weights.
        """
        super().load_model(load_dummy_weights)

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
        mm_config = self.vllm_config.model_config.multimodal_config
        self._is_mm_encoder_only = bool(mm_config and mm_config.mm_encoder_only)
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
        if consume_aclgraph_recapture():
            self.reset_input_batch()
            # Release old graphs and rotate the shared graph pool before
            # recapturing to avoid stale pool memory staying live.
            reset_stale_aclgraph_resources(self._iter_aclgraph_wrappers())
        with (
            switch_torch_device(),
            patch("torch.accelerator.empty_cache", torch.npu.empty_cache),
            patch(
                "torch.accelerator.get_memory_info",
                torch.npu.mem_get_info,
                create=True,
            ),
        ):
            return super().capture_model()

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
        # Keep this iterator in sync with every path that owns an
        # ACLGraphWrapper. Recapture uses it to release stale graphs and
        # repoint wrappers to the refreshed shared graph pool.
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

    @torch.inference_mode()
    @model_output_decorator
    @maybe_dump_kv
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, AsyncModelRunnerOutput, IntermediateTensors]:
        with (switch_torch_device()
              if self.use_async_scheduling else nullcontext()):
            current_device = torch.npu.current_device()
            if intermediate_tensors:
                tensors = intermediate_tensors.tensors
                for key in tensors:
                    tensors[key] = tensors[key].to(current_device)
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
        is_graph_capturing: bool = False,
        num_active_loras: int = 0,
        profile_seq_lens: int | None = None,
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
            num_active_loras: Number of distinct active LoRAs to capture for.
                LoRA is activated when num_active_loras > 0.
            profile_seq_lens: If provided, use this value for seq_lens instead
                of max_query_len. Used to profile attention workspace that
                scales with context length.
        """
        # Dummy runs skip _prepare_inputs, so the buffer would be stale here.
        bind_num_prompt_tokens(self.attn_groups, None)

        mm_config = self.vllm_config.model_config.multimodal_config
        if mm_config and mm_config.mm_encoder_only:
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
        assert num_tokens <= self.max_num_tokens
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
                force_has_lora=num_active_loras > 0,
                # `force_num_active_loras` is used for cudagraph capture; because we
                # need to capture graphs for specific num_active_loras counts
                force_num_active_loras=num_active_loras,
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

        # Match attn metadata sizing: only FULL cudagraph pads attn to the
        # DP-padded token count. Otherwise create unpadded slots so indexer
        # k/slot lengths stay aligned when MoE hs is DP-padded.
        pad_attn = cudagraph_runtime_mode == CUDAGraphMode.FULL
        slot_num_tokens = num_tokens_padded if pad_attn else num_tokens_unpadded
        slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
            num_tokens_padded=slot_num_tokens,
            num_reqs_padded=num_reqs_padded,
            num_tokens_unpadded=num_tokens_unpadded,
            ubatch_slices=(ubatch_slices_padded if pad_attn else ubatch_slices),
        )

        # Dummy runs have no real slot assignments — fill with PADDING_SLOT_ID
        # so concat_and_cache / indexer kernels skip the KV write.
        if slot_mappings_by_group is not None:
            for sm in slot_mappings_by_group.values():
                sm.fill_(PADDING_SLOT_ID)

        # _dummy_run shares pinned CPU buffers (seq_lens, query_start_loc,
        # etc.) with execute_model.  It must participate in the same event
        # protocol so that back-to-back dummy/real steps don't overwrite
        # pinned memory while a prior non_blocking H2D DMA is still reading.
        with self.synchronize_input_prep():
            # If force_attention is True, we always capture attention.
            # Otherwise, it only happens for cudagraph_runtime_mode=FULL.
            if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
                if profile_seq_lens is not None:
                    seq_lens = profile_seq_lens  # type: ignore[assignment]
                elif create_mixed_batch:
                    # In the mixed batch mode (used for FI warmup), we use
                    # shorter sequence lengths to run faster.
                    # TODO(luka) better system for describing dummy batches
                    seq_lens = torch.tensor(  # type: ignore[assignment]
                        [1] * num_decode_tokens + [num_prefill_tokens + 1],
                        dtype=torch.int,
                    )
                else:
                    seq_lens = max_query_len  # type: ignore[assignment]
                self.optimistic_seq_lens_cpu[:num_reqs] = seq_lens
                self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)
                self.seq_lens.copy_(self.optimistic_seq_lens_cpu, non_blocking=True)

                cum_num_tokens = self._get_cumsum_and_arange(
                    num_scheduled_tokens, self.query_pos.np
                )
                self.query_start_loc.np[1: num_reqs + 1] = cum_num_tokens
                self.query_start_loc.np[num_reqs + 1: num_reqs_padded + 1].fill(
                    cum_num_tokens[-1]
                )
                self.query_start_loc.copy_to_gpu()

                # Sync block table CPU->GPU so cleared rows from
                # remove_request() are visible to the attention metadata
                # builder. Without this, stale block IDs from finished
                # requests can corrupt Mamba state.
                self.input_batch.block_table.commit_block_table(num_reqs_padded)

                attn_metadata, _ = self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded,
                    num_tokens_padded=num_tokens_padded if pad_attn else None,
                    num_reqs=num_reqs_padded,
                    max_query_len=max_query_len,
                    ubatch_slices=(ubatch_slices_padded if pad_attn else ubatch_slices),
                    for_cudagraph_capture=is_graph_capturing,
                    slot_mappings=slot_mappings_by_group,
                    use_spec_decode=self.speculative_config is not None,
                )

        with self.maybe_dummy_run_with_lora(
            self.lora_config,
            num_scheduled_tokens,
            num_sampled_tokens,
            remove_lora,
            num_active_loras,
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
                positions = self.positions[:num_tokens_padded]

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
                    slot_mapping=slot_mappings,
                ),
            ):
                forward_context = get_forward_context()
                forward_context.capturing = False
                # Idle DP ranks must use the same LMHead all-gather pad target
                # as the active rank before entering dummy compute_logits.
                self._capture_dp_pad_target(forward_context)
                if self.is_debugging_mode:
                    self.exec_count += 1
                    logger.debug(f"Executing dummy forward {self.exec_count=}")
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
                if (
                    self.compilation_config.cudagraph_specialize_lora
                    and num_active_loras > 0
                ):
                    use_cudagraphs = False

                # Adapt start: to pass attn_metadata
                self.drafter.dummy_run(
                    attn_metadata,
                    num_tokens,
                    use_cudagraphs=use_cudagraphs,
                    is_graph_capturing=is_graph_capturing,
                    slot_mappings=slot_mappings,
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
    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[
        torch.Tensor,
        SpecDecodeMetadata | None,
    ]:
        (logits_indices, spec_decode_metadata) = super()._prepare_inputs(scheduler_output, num_scheduled_tokens)

        self._refresh_mome_num_prompt_tokens()

        return (logits_indices, spec_decode_metadata)

    def _refresh_mome_num_prompt_tokens(self) -> None:
        """Publish this step's prompt lengths to the MoME builders.

        Not gated on use_spec_decode: a prefill step has no drafts, but the
        step after it must still recognise that the previous one was a prefill.
        """
        num_reqs = self.input_batch.num_reqs
        self.num_prompt_tokens.np[:num_reqs] = self.input_batch.num_prompt_tokens[:num_reqs]
        self.num_prompt_tokens.np[num_reqs:].fill(0)
        self.num_prompt_tokens.copy_to_gpu()
        bind_num_prompt_tokens(self.attn_groups, self.num_prompt_tokens.gpu)

    @init_config_decorator
    def initialize_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
        is_profiling: bool = False,
    ) -> None:
        super().initialize_kv_cache(kv_cache_config, is_profiling)

        # patch start
        if self.model_config.enable_return_routed_experts:
            self.init_routed_experts_capturer()
        # patch end

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
            for _, module in attn_layers.items():
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

        for kv_cache_group_id, _ in enumerate(self.kv_cache_config.kv_cache_groups):
            for attn_group in self.attn_groups[kv_cache_group_id]:
                for attn_builder in attn_group.metadata_builders:
                    if hasattr(attn_builder, "reinit_block_table_with_sink"):
                        attn_builder.reinit_block_table_with_sink()

    def unregister_kv_caches(self):
        kv_transfer_config = self.vllm_config.kv_transfer_config
        is_llmdatadist = (
            kv_transfer_config is not None
            and kv_transfer_config.kv_connector == "LLMDataDistConnector"
        )
        if is_llmdatadist:
            if has_kv_transfer_group():
                logger.info("unregister_kv_caches")
                get_kv_transfer_group().unregister_kv_caches()

    def reregister_kv_caches(self):
        kv_transfer_config = self.vllm_config.kv_transfer_config
        is_llmdatadist = (
            kv_transfer_config is not None
            and kv_transfer_config.kv_connector == "LLMDataDistConnector"
        )
        if is_llmdatadist:
            if has_kv_transfer_group():
                logger.info(f"reregister_kv_caches")
                get_kv_transfer_group().register_kv_caches(self.kv_caches_dict)

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        if not self.num_spec_tokens or not self._draft_token_req_ids:
            return None

        draft_token_ids, req_ids = self._get_draft_token_ids_cpu()
        filtered_req_ids = []
        filtered_draft_token_ids = []
        for req_id, tokens in zip(req_ids, draft_token_ids):
            if any(token_id < 0 for token_id in tokens):
                # Can fire every decode step; keep alert once-per-process and
                # leave per-request detail at debug (warning_once is lru_cached
                # on message args).
                logger.warning_once(
                    "Dropping draft tokens containing negative ids; the "
                    "affected requests fall back to normal decode for that "
                    "step. Enable debug logging for per-request details. "
                    "This warning is logged only once."
                )
                logger.debug(
                    "Dropping invalid draft token id(s) for request %s: %s",
                    req_id,
                    tokens,
                )
                continue
            filtered_req_ids.append(req_id)
            filtered_draft_token_ids.append(tokens)

        if not filtered_req_ids:
            return None
        return DraftTokenIds(filtered_req_ids, filtered_draft_token_ids)
