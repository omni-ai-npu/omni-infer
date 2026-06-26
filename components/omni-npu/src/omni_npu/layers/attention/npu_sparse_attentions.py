# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""
Pangu V2 Hybrid Attention Layers.

This module implements two attention variants for Pangu V2:
- PanguV2MLASWAAttention: MLA with sliding window attention (uses NPUPanguMLA backend)
- PanguV2DSAAttention: DeepSeek-style sparse attention (uses NPUPanguDSA backend)

Both inherit from MLAAttention and override get_kv_cache_spec() and get_attn_backend()
to support interleaved hybrid attention patterns.
"""

from typing import Optional

import torch

from vllm.config import VllmConfig, CacheConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.logger import init_logger
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.config import get_current_vllm_config
from vllm.v1.attention.backend import AttentionBackend
from vllm.attention.layer import MLAAttention
from vllm.model_executor.layers.mamba.abstract import MambaBase

from omni_npu.attention.backends.mome import NPUPanguMomeBackend
from omni_npu.attention.backends.mla import NPUMLABackend
from omni_npu.attention.backends.dsa import NPUDSABackend

logger = init_logger(__name__)


class MLASWAAttention(MLAAttention):
    """
    This variant uses MLA (Multi-Head Latent Attention) with a sliding window
    mechanism. It inherits from MLAAttention and overrides get_kv_cache_spec()
    to return ShareKVSlidingWindowSpec and get_attn_backend() to use NPUPanguMLA.

    Note: MLA-SWA does not use indexer or sparse attention.
    """

    def __init__(
        self,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        kv_b_proj: ColumnParallelLinear,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        cache_dtype_str: str | None = None,
        page_size_padded: Optional[int] = None,
        sliding_window: int | None = None,
        num_extra_reserved_blocks: int = 0,
        **extra_impl_args,
    ):
        self.cache_dtype_str = cache_dtype_str
        self.sliding_window = sliding_window
        self.page_size_padded = page_size_padded
        self.num_extra_reserved_blocks = num_extra_reserved_blocks

        super().__init__(
            num_heads=num_heads,
            scale=scale,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            kv_b_proj=kv_b_proj,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            use_sparse=False,
            indexer=None,
            **extra_impl_args,
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return NPUMLABackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:

        """
        For MLA-SWA:
        - If sliding_window is set: use ShareKVSlidingWindowSpec
        - If sliding_window is None: use standard MLAAttentionSpec (full attention)
        """

        from vllm.v1.kv_cache_interface import (
            MLAAttentionSpec,
            ShareKVSlidingWindowSpec,
        )

        # hif8_ds_mla/fp8_ds_mla/int8_ds_mla are DSA-only quantization formats;
        # SWA layers always use unquantized (auto) cache dtype.
        cache_dtype = self.kv_cache_dtype
        if cache_dtype in ("hif8_ds_mla", "fp8_ds_mla", "int8_ds_mla"):
            cache_dtype = "auto"
        kv_cache_dtype = kv_cache_dtype_str_to_dtype(
            cache_dtype, vllm_config.model_config
        )

        if self.sliding_window is None:
            return MLAAttentionSpec(
                block_size=vllm_config.cache_config.block_size,
                num_kv_heads=1,
                head_size=self.head_size,
                dtype=kv_cache_dtype,
                cache_dtype_str=self.cache_dtype_str,
                page_size_padded=self.page_size_padded,
            )

        return ShareKVSlidingWindowSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_size,
            dtype=kv_cache_dtype,
            sliding_window=self.sliding_window,
            page_size_padded=self.page_size_padded,
            num_extra_reserved_blocks=self.num_extra_reserved_blocks,
        )


class DSAAttention(MLAAttention):
    """
    This variant uses DSA (DeepSeek Sparse Attention) which combines
    NOPE (non-positional encoding), ROPE (rotary positional encoding),
    and an indexer for sparse attention patterns. It inherits from
    MLAAttention and overrides get_kv_cache_spec() to return DSAAttentionSpec
    and get_attn_backend() to use NPUPanguDSA.
    """

    def __init__(
        self,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        kv_b_proj: ColumnParallelLinear,
        indexer: torch.nn.Module,
        indexer_head_dim: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        cache_dtype_str: str | None = None,
        page_size_padded: Optional[int] = None,
        block_size: int | None = None,
        **extra_impl_args,
    ):
        self.cache_dtype_str = cache_dtype_str
        self.page_size_padded = page_size_padded
        self.indexer_head_dim = indexer_head_dim
        self.block_size = block_size

        super().__init__(
            num_heads=num_heads,
            scale=scale,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            kv_b_proj=kv_b_proj,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            use_sparse=True,
            indexer=indexer,
            **extra_impl_args,
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return NPUDSABackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        """
        For DSA, we use DSAAttentionSpec which handles the combined
        NOPE + ROPE + INDEXER format with optional quantization support.
        """
        from vllm.v1.kv_cache_interface import DSAAttentionSpec


        kv_cache_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, vllm_config.model_config
        )

        return DSAAttentionSpec(
            block_size=self.block_size or vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_size + self.indexer_head_dim,
            dtype=kv_cache_dtype,
            cache_dtype_str=self.cache_dtype_str,
            page_size_padded=self.page_size_padded,
        )


class MomeAttention(MambaBase):
    """
    This variant uses MOME which is a
    more efficient attention mechanism. It inherits from MambaBase and overrides
    get_kv_cache_spec() to return MambaSpec and get_attn_backend() to use NPUPanguMome.
    """

    def __init__(
        self,
        kernel_size: int,
        num_spec_tokens: int,
        state_dtypes: tuple[torch.dtype, ...],
        state_shapes: tuple[tuple[int, ...], ...],
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        page_size_padded: Optional[int] = None,
        num_extra_reserved_blocks: int = 0,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.num_spec_tokens = num_spec_tokens
        self.state_dtypes = state_dtypes
        self.state_shapes = state_shapes
        self.cache_config = cache_config
        self.quant_config = quant_config
        self.page_size_padded = page_size_padded
        self.num_extra_reserved_blocks = num_extra_reserved_blocks

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    @property
    def mamba_type(self) -> str:
        return "mome"
    
    def get_attn_backend(self) -> type[AttentionBackend]:
        return NPUPanguMomeBackend

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        """
        The state shapes define the dimensions of each state tensor maintained
        by the MOME layer for efficient sequence modeling.
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
        """
        from vllm.v1.kv_cache_interface import MomeSpec

        enable_prefix_caching = vllm_config.cache_config.enable_prefix_caching
        block_size = vllm_config.cache_config.block_size
        max_model_len = vllm_config.model_config.max_model_len
        mamba_block_size = block_size if enable_prefix_caching else max_model_len

        return MomeSpec(
            shapes=self.get_state_shape(),
            dtypes=self.get_state_dtype(),
            block_size=mamba_block_size,
            page_size_padded=self.page_size_padded,
            mamba_type=self.mamba_type,
            kernel_size=self.kernel_size,
            num_spec_tokens=self.num_spec_tokens,
            num_extra_reserved_blocks=self.num_extra_reserved_blocks,
        )
