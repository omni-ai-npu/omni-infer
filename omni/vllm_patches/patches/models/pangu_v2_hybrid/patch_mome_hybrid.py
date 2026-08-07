# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import sys
import types
from typing import Optional

import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.logger import init_logger
from vllm.model_executor import layers
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.config import get_current_vllm_config
from vllm.v1.attention.backend import AttentionBackend
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import CustomOp

from omni_npu.attention.backends.mome import (
    NPUMomeAttentionMetadata,
    NPUPanguMomeBackend,
)
from omni_npu.layers.mome.mome_rl import ColumnParallelMOMERL
from omni_npu.v1.utils import on_ascend950
from omni_npu.plugin_decorators import attn_decorator
from omni_npu.vllm_patches.core import VLLMPatch, register_patch


logger = init_logger(__name__)
dynamic_module = types.ModuleType("npumome")
sys.modules[layers.__name__ + ".npumome"] = dynamic_module
layers.npumome = dynamic_module


@register_patch("NPUMoMEPatch", layers)
class NPUMoMEPatch(VLLMPatch):
    _attr_names_to_apply = ['MomeAttention']

    @CustomOp.register("MomeAttention")
    class MomeAttention(MambaBase, CustomOp):
        """
        MOME attention layer with integrated vLLM KV cache management.

        This variant uses MOME which is a more efficient attention mechanism.
        It inherits from MambaBase and:
        1. Overrides get_kv_cache_spec() to return MomeSpec with three state
           components
        2. Integrates three convolutions using ColumnParallelMOMERL
        3. Uses NPUPanguMomeBackend for attention backend
        4. Supports vLLM KV cache management through MomeSpec

        KV Cache Flow:
        1. get_kv_cache_spec() → MomeSpec with (q_cache, kv_cache, o_cache) shapes
        2. Model runner allocates raw tensor (int8 buffer)
        3. NPUPanguMomeBackend.reshape_kv_cache() → strided tensors
        4. kv_cache bound to self.kv_cache
        5. forward() uses self.kv_cache for state management

        The three convolutions share the same kernel width (router_sliding_window)
        and are applied in:
        - q path: qa_conv before q projection
        - kv path: compresskv_conv on compressed KV
        - o path: o_conv on attention output

        Args:
            kernel_size: Kernel size for causal convolution (router_sliding_window)
            num_spec_tokens: Number of speculative tokens
            state_dtypes: Data types for three state components
            state_shapes: Shapes for three state components
            cache_config: Cache configuration
            quant_config: Quantization configuration
            prefix: Layer name prefix
            q_lora_rank: Q projection low-rank dimension (for qa_conv)
            kv_lora_rank: KV compression low-rank dimension (for compresskv_conv)
            num_heads: Number of attention heads (for o_conv)
            v_head_dim: Value head dimension (for o_conv)
        """

        def __init__(
            self,
            kernel_size: int,
            num_spec_tokens: int,
            state_dtypes: tuple[torch.dtype, ...],
            state_shapes: tuple[tuple[int, ...], ...],
            vllm_config: VllmConfig | None = None,
            quant_config: QuantizationConfig | None = None,
            prefix: str = "",
        ):
            super().__init__()

            self.kernel_size = kernel_size
            self.num_spec_tokens = num_spec_tokens
            self.vllm_config = vllm_config
            self.quant_config = quant_config
            self.prefix = prefix

            # MOME parameters
            self.num_total_tokens = self.kernel_size - 1 + self.num_spec_tokens

            # Store parameters for MOME convolutions
            self.q_lora_rank = state_shapes[0][0]
            self.kv_lora_rank = state_shapes[1][0]
            self.o_dim = state_shapes[2][0]
            self.state_shapes = state_shapes
            self.state_dtypes = state_dtypes

            self.on_ascend950 = on_ascend950()

            # KV cache will be set by model runner during initialization
            self.kv_cache = (torch.tensor([]), torch.tensor([]), torch.tensor([]))

            # Initialize MOME convolutions using ColumnParallelMOMERL
            self._init_mome_convs(quant_config, prefix)

            compilation_config = get_current_vllm_config().compilation_config
            if prefix in compilation_config.static_forward_context:
                raise ValueError(f"Duplicate layer name: {prefix}")
            compilation_config.static_forward_context[prefix] = self

        def _init_mome_convs(
            self,
            quant_config: QuantizationConfig | None,
            prefix: str,
        ) -> None:
            """
            Initialize three MOME convolutions using ColumnParallelMOMERL.

            These convolutions use the kernel_size from config (router_sliding_window)
            and are implemented as depthwise conv1d operations with TP support.
            """
            # qa_conv: applied to q_lora before q projection
            self.qa_conv = ColumnParallelMOMERL(
                dim=self.q_lora_rank,
                kernel_width=self.kernel_size,
                quant_config=quant_config,
                prefix=f"{prefix}.qa_conv",
                disable_tp=True,
            )

            # compresskv_conv: applied to compressed KV
            self.compresskv_conv = ColumnParallelMOMERL(
                dim=self.kv_lora_rank,
                kernel_width=self.kernel_size,
                quant_config=quant_config,
                prefix=f"{prefix}.compresskv_conv",
                disable_tp=True,
            )

            # o_conv: applied to attention output
            self.o_conv = ColumnParallelMOMERL(
                dim=self.o_dim,
                kernel_width=self.kernel_size,
                quant_config=quant_config,
                prefix=f"{prefix}.o_conv",
                disable_tp=True,
            )

        @property
        def mamba_type(self) -> str:
            return "mome"

        def get_attn_backend(self) -> type[AttentionBackend]:
            """Return NPUPanguMomeBackend for MOME attention computation."""
            return NPUPanguMomeBackend

        def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
            """
            The state shapes define the dimensions of each state tensor maintained
            by the MOME layer for efficient sequence modeling.

            Returns three component shapes for MomeSpec:
            - q_cache: (q_lora_rank,) - cached q_lora states for causal convolution
            - kv_cache: (kv_lora_rank,) - cached compressed KV states
            - o_cache: (num_heads * v_head_dim,) - cached output states
            """
            return self.state_shapes

        def get_state_dtype(self) -> tuple[torch.dtype, ...]:
            """
            The state dtypes specify the precision (e.g., float32, bfloat16) for
            each state tensor stored in the MOME layer.
            """
            return self.state_dtypes

        def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
            """
            Returns the KV cache specification for this MOME layer.

            For MOME, we use MomeSpec which includes state shapes/dtypes,
            block configuration, and MOME-specific parameters like kernel size
            and special token count.

            The KV cache is managed by vLLM:
            1. Model runner allocates raw tensor (int8 buffer)
            2. NPUPanguMomeBackend.reshape_kv_cache() converts to strided tensors
            3. Result is tuple of (q_cache, kv_cache, o_cache)
            4. Each cache has shape: (num_blocks, num_total_tokens, dim)
            5. num_total_tokens = kernel_size - 1 + num_spec_tokens
            """
            from vllm.v1.kv_cache_interface import MomeSpec
            mamba_block_size = vllm_config.cache_config.mamba_block_size
            page_size_padded = vllm_config.cache_config.mamba_page_size_padded

            return MomeSpec(
                shapes=self.get_state_shape(),
                dtypes=self.get_state_dtype(),
                block_size=mamba_block_size,
                page_size_padded=page_size_padded,
                mamba_type=self.mamba_type,
                kernel_size=self.kernel_size,
                num_spec_tokens=self.num_spec_tokens,
            )

        def forward(
            self,
            hidden_states: torch.Tensor,
            state_indice: int,
            is_prefill: bool = False,
            **kwargs,
        ) -> torch.Tensor:
            """
            Forward pass interface for MOME attention.

            This method provides access to KV cache and attention metadata.
            The actual MOME convolution computation is delegated to
            ColumnParallelMOMERL, which is called by the model layer.

            Args:
                hidden_states: Input hidden states
                **kwargs: Additional arguments from model layer

            Returns:
                dict containing:
                - kv_cache: Tuple of (q_cache, kv_cache, o_cache)
                - attn_metadata: Attention metadata with cache_indices
                - num_total_tokens: Total tokens per block

            Note:
                This method is designed to be called by model layers that implement
                the full attention computation. The model layer should:
                1. Call this method to get KV cache and metadata
                2. Apply qa_conv using ColumnParallelMOMERL forward methods
                3. Apply compresskv_conv using ColumnParallelMOMERL forward methods
                4. Compute MLA attention
                5. Apply o_conv using ColumnParallelMOMERL forward methods
            """
            forward_context = get_forward_context()

            # Get attention metadata
            metadata = forward_context.attn_metadata
            if metadata is None:
                return hidden_states
            kv_cache = self.kv_cache[forward_context.virtual_engine]

            mome_metadata = metadata[self.prefix]

            merge_conv = self.qa_conv
            if state_indice == 1:
                merge_conv = self.compresskv_conv
            elif state_indice == 2:
                merge_conv = self.o_conv
            conv_state = kv_cache[state_indice]

            # Apply MOME convolutions separately for decode and prefill
            return self.apply_mome_conv(
                x=hidden_states,
                conv_layer=merge_conv,
                cache=conv_state,
                mome_metadata=mome_metadata,
                is_prefill=is_prefill,
            )

        @attn_decorator(type='mome')
        def apply_mome_conv(
            self,
            x: torch.Tensor,
            conv_layer: ColumnParallelMOMERL,
            cache: torch.Tensor,
            is_prefill: bool,
            mome_metadata: Optional[NPUMomeAttentionMetadata] = None,
        ) -> torch.Tensor:
            """
            Apply MOME convolution using ColumnParallelMOMERL with KV cache update.

            This is a helper method for model layers to apply MOME convolution
            with proper cache management. The cache is updated in-place during
            the forward pass.
            Returns:
                Output tensor with same shape as input.
                The cache tensor is updated in-place as a side effect.
            """

            metadata = mome_metadata.prefill if is_prefill else mome_metadata.decode
            if metadata is None:
                return x

            x = conv_layer.forward(
                x=x,
                conv_states=cache,
                mome_metadata=metadata,
                inplace=False,
            )

            return x

    layers.npumome.MomeAttention = MomeAttention
