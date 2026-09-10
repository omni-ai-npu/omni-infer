# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Add new KV cache spec classes for Pangu V2 hybrid attention:
- DSAAttentionSpec: DeepseekV32 DSA with quantization support
- ShareKVSlidingWindowSpec: Sliding window for MLA/MQA with shared KV
- MomeSpec: Mamba-like state management
"""

from dataclasses import dataclass, replace
from math import prod

from vllm.v1 import kv_cache_interface
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
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
            # HiF8 with scale: KV data + indexer + fp32 scale.
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
        cache_dtype_str = next(iter(cache_dtype_str_set))
        merged_spec = super().merge(specs)
        return replace(
            merged_spec,
            cache_dtype_str=cache_dtype_str,
        )

    def __post_init__(self):
        super().__post_init__()
        assert self.sliding_window is None, (
            "For DSAAttentionSpec, sliding window should not be enabled, "
            f"but got {self.sliding_window}."
        )
        assert self.num_kv_heads == 1, (
            "For DSAAttentionSpec, num_kv_heads should be 1, "
            f"but got {self.num_kv_heads}."
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
        super().__post_init__()
        assert self.num_kv_heads == 1, (
            f"Only support single KV head, but got {self.num_kv_heads}."
        )
        assert self.head_size in [512, 576], (
            f"Head size should be either 512 or 576, but got {self.head_size}."
        )


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
            assert self.page_size_padded >= page_size, (
                f"page_size_padded {self.page_size_padded} must be >= computed page_size {page_size}."
            )
            return self.page_size_padded
        return page_size

    def max_admission_blocks_per_request(
        self, max_num_batched_tokens: int, max_model_len: int
    ) -> int:
        """Per-request admission cap for recycling-aware Mome KV.

        Mirrors ``SlidingWindowSpec.max_admission_blocks_per_request`` and
        ``MomeManager.get_num_skipped_tokens``: during chunked prefill we retain
        at most ``kernel_size - 1 + num_extra_reserved_blocks * block_size``
        tokens plus the newly scheduled chunk.
        """
        num_retained = (
            self.kernel_size
            - 1
            + self.num_extra_reserved_blocks * self.block_size
        )
        num_tokens = min(
            num_retained + max_num_batched_tokens,
            max_model_len,
        )
        return (
            cdiv(num_tokens, self.block_size)
            + 1
            + self.num_speculative_blocks
        )

    def max_memory_usage_bytes(self, vllm_config) -> int:
        max_model_len = vllm_config.model_config.max_model_len
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        max_blocks = self.max_admission_blocks_per_request(
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
        )
        return max_blocks * self.page_size_bytes

    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        return all(
            isinstance(spec, MomeSpec)
            and spec.num_total_tokens == self.num_total_tokens
            for spec in kv_cache_specs.values()
        )

    def __post_init__(self):
        if len(self.shapes) != 3:
            raise ValueError(
                f"Mome has 3 components, but got {len(self.shapes)} shapes."
            )
        if len(self.dtypes) != 3:
            raise ValueError(
                f"Mome has 3 components, but got {len(self.dtypes)} dtypes."
            )
        if self.kernel_size <= 0:
            raise ValueError(
                "Mome should have positive kernel_size, "
                f"but got {self.kernel_size}."
            )


@register_patch("PanguNewKVCacheSpecsPatch", kv_cache_interface)
class PanguNewKVCacheSpecsPatch(VLLMPatch):
    """Patch to add new kv cache specs."""

    _attr_names_to_apply = [
        "DSAAttentionSpec",
        "ShareKVSlidingWindowSpec",
        "MomeSpec",
    ]
    DSAAttentionSpec = DSAAttentionSpec
    ShareKVSlidingWindowSpec = ShareKVSlidingWindowSpec
    MomeSpec = MomeSpec
