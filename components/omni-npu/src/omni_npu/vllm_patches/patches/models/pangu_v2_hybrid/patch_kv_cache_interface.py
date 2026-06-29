# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright contributors to the vLLM project.

"""
Add new KV cache spec classes for Pangu V2 hybrid attention:
- DSAAttentionSpec: DeepseekV32 DSA with quantization support
- ShareKVSlidingWindowSpec: Sliding window for MLA/MQA with shared KV
- MomeSpec: Mamba-like state management
"""

from dataclasses import dataclass
from math import prod

import torch

from vllm.v1 import kv_cache_interface
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
    ChunkedLocalAttentionSpec,
    CrossAttentionSpec,
    MambaSpec,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_dtype_size

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


@dataclass(frozen=True, kw_only=True)
class DSAAttentionSpec(FullAttentionSpec):
    """
    A variant of DeepseekV32's DSA (Deepseek Sparse Attention). Different from
    original DeepseekV32 which combines two MLAAttentionSpec, this one considers
    NOPE, ROPE and INDEXER as a whole so that it can be hybrid with other attention.

    The page size calculation differs based on whether quantization is used.
    """
    # For quantization, indicates the cache dtype string (e.g., "fp8_ds_mla")
    cache_dtype_str: str | None = None

    @property
    def real_page_size_bytes(self) -> int:
        if self.cache_dtype_str == "fp8_ds_mla":
            # Quant case: 512 fp8 + 64 bf16 + 4 fp32 + 128 int8 + 1 fp32
            # See DeepseekV3 quantized DSA format
            return self.block_size * (656 + 128 + 4)
        elif self.cache_dtype_str == "int8_ds_mla":
            # Quant case: 512 int8 + 64 bf16 + 4 fp32 + 128 int8 + 1 bf16
            return self.block_size * (656 + 128 + 2)
        elif self.cache_dtype_str == "hif8_ds_mla":
            # HiF8 with scale: 656 bytes KV data + 128 bytes indexer + 4 bytes fp32 scale
            return self.block_size * (656 + 128 + 4)
        elif self.cache_dtype_str == "li_int8_ds_mla":
            # Li-Quant-Only case: 576 bf16 + 128 int8 + 1 bf16
            return self.block_size * (576 * 2 + 128 + 2)
        # Non-quant case: standard attention format
        return (
            self.block_size
            * self.num_kv_heads
            * self.head_size
            * get_dtype_size(self.dtype)
        )

    @classmethod
    def merge(cls, specs: list) -> "DSAAttentionSpec":
        assert all(isinstance(spec, DSAAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be DSAAttentionSpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        assert len(cache_dtype_str_set) == 1, (
            "All attention layers in the same KV cache group must use the same "
            "quantization method."
        )
        merged_spec = cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=specs[0].dtype,
            page_size_padded=specs[0].page_size_padded,
            cache_dtype_str=cache_dtype_str_set.pop(),
        )
        return merged_spec

    def __post_init__(self):
        assert self.sliding_window is None, (
            f"For DSAAttentionSpec, sliding window should not be enabled, but got {self.sliding_window}."
        )
        assert self.num_kv_heads == 1, (
            f"For DSAAttentionSpec, num_kv_heads should be 1, but got {self.num_kv_heads}."
        )


@dataclass(frozen=True, kw_only=True)
class ShareKVSlidingWindowSpec(SlidingWindowSpec):

    """
    A variant of SlidingWindowSpec that shares key and value in
    underlying storage, e.g., MLA and MQA.

    Adds padding to page size for hybrid attention alignment.
    """

    num_extra_reserved_blocks: int = 0

    @property
    def real_page_size_bytes(self) -> int:
        return (
            self.block_size
            * self.num_kv_heads
            * self.head_size
            * get_dtype_size(self.dtype)
        )

    def __post_init__(self):
        assert self.num_kv_heads == 1, f"Only support single KV head, but got {self.num_kv_heads}."
        assert self.head_size in [512, 576], f"Head size should be either 512 or 576, but got {self.head_size}."


@dataclass(frozen=True, kw_only=True)
class MomeSpec(MambaSpec):
    """
    A spec similar to Mamba that always stores representations of
    `kernel_size - 1 + num_spec_tokens` tokens per block.
    """
    kernel_size: int = 0
    num_spec_tokens: int = 0
    num_extra_reserved_blocks: int = 0

    @property
    def num_total_tokens(self) -> int:
        return self.kernel_size - 1 + self.num_spec_tokens

    @property
    def page_size_bytes(self) -> int:
        page_size = sum(
            prod(shape) * get_dtype_size(dtype)
            for (shape, dtype) in zip(self.shapes, self.dtypes)
        ) * self.num_total_tokens
        if self.page_size_padded is not None:
            assert self.page_size_padded >= page_size
            return self.page_size_padded
        return page_size

    def max_memory_usage_bytes(self, vllm_config) -> int:
        max_model_len = vllm_config.model_config.max_model_len
        return cdiv(max_model_len, self.block_size) * self.page_size_bytes

    def __post_init__(self):
        if len(self.shapes) != 3:
            raise ValueError(f"Mome has 3 components, but got {len(self.shapes)} shapes.")
        if len(self.dtypes) != 3:
            raise ValueError(f"Mome has 3 components, but got {len(self.dtypes)} dtypes.")
        if self.kernel_size <= 0:
            raise ValueError(f"Mome should have positive kernel_size, but got {self.kernel_size}.")


@register_patch("UniformTypeKVCacheSpecsPatch", UniformTypeKVCacheSpecs)
class UniformTypeKVCacheSpecsPatch(VLLMPatch):
    """Patch to add MomeSpec support to is_uniform_type method"""

    _attr_names_to_apply = ["is_uniform_type"]

    @classmethod
    def is_uniform_type(cls, kv_cache_specs: dict[str, KVCacheSpec]) -> bool:
        """
        Whether all layers have the same type of KV cache spec.
        """
        block_sizes = set(spec.block_size for spec in kv_cache_specs.values())
        if len(block_sizes) > 1:
            # Different block sizes, not uniform.
            return False
        one_spec = next(iter(kv_cache_specs.values()))
        if isinstance(one_spec, FullAttentionSpec):
            return all(
                isinstance(spec, FullAttentionSpec) for spec in kv_cache_specs.values()
            )
        elif isinstance(one_spec, CrossAttentionSpec):
            return all(
                isinstance(spec, CrossAttentionSpec) for spec in kv_cache_specs.values()
            )
        elif isinstance(one_spec, SlidingWindowSpec):
            return all(
                isinstance(spec, SlidingWindowSpec)
                and spec.sliding_window == one_spec.sliding_window
                for spec in kv_cache_specs.values()
            )
        elif isinstance(one_spec, ChunkedLocalAttentionSpec):
            return all(
                isinstance(spec, ChunkedLocalAttentionSpec)
                and spec.attention_chunk_size == one_spec.attention_chunk_size
                for spec in kv_cache_specs.values()
            )
        elif isinstance(one_spec, MambaSpec):
            return all(
                isinstance(spec, MambaSpec)
                and spec.num_speculative_blocks == one_spec.num_speculative_blocks
                for spec in kv_cache_specs.values()
            )
        elif isinstance(one_spec, MomeSpec):
            return all(
                isinstance(spec, MomeSpec)
                and spec.num_total_tokens == one_spec.num_total_tokens
                for spec in kv_cache_specs.values()
            )
        else:
            # NOTE(Chen): Please add new branches for new KV cache spec types.
            raise NotImplementedError(
                f"Unsupported KV cache spec type: {type(one_spec)}"
            )

from typing_extensions import Self
from vllm.v1.kv_cache_interface import MLAAttentionSpec


@dataclass(frozen=True)
class SinkMLAAttentionSpec(MLAAttentionSpec):
    sink_len: int = 0

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be MLAAttentionSpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        assert len(cache_dtype_str_set) == 1, (
            "All attention layers in the same KV cache group must use the same "
            "quantization method."
        )
        head_size_set = set(spec.head_size for spec in specs)
        assert len(head_size_set) == 1, (
            "All attention layers in the same KV cache group must use the same head_size."
        )
        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=specs[0].dtype,
            page_size_padded=specs[0].page_size_padded,
            cache_dtype_str=cache_dtype_str_set.pop(),
            sink_len=specs[0].sink_len,
        )


@register_patch("PanguNewKVCacheSpecsPatch", kv_cache_interface)
class PanguNewKVCacheSpecsPatch(VLLMPatch):
    """Patch to add new kv cache specs."""

    _attr_names_to_apply = ["DSAAttentionSpec", "ShareKVSlidingWindowSpec", "MomeSpec", "SinkMLAAttentionSpec"]
    DSAAttentionSpec = DSAAttentionSpec
    ShareKVSlidingWindowSpec = ShareKVSlidingWindowSpec
    MomeSpec = MomeSpec
    SinkMLAAttentionSpec = SinkMLAAttentionSpec
