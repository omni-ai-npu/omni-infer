# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Register MomeManager and inject recycling-aware admission caps for MomeSpec.

Also registers ShareKVSlidingWindowManager for hybrid Pangu models loaded via
usefull_patch (low_latency path).

Note: ``HybridKVCacheCoordinator.find_longest_cache_hit`` is NOT patched here.
Current upstream vLLM already ships the iterative fixed-point hybrid APC
logic (with ``drop_eagle_block`` and ``num_uncached_common_prefix_tokens``).
The older pangu_v2_hybrid coordinator patch used the deprecated ``use_eagle``
kwarg and would regress APC if applied on top of modern vLLM.
"""

from vllm.v1.core import single_type_kv_cache_manager
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import (
    ChunkedLocalAttentionSpec,
    MambaManager,
    SingleTypeKVCacheManager,
    SlidingWindowManager,
    SlidingWindowSpec,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, SlidingWindowSpec as SWSpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.vllm_patches.usefull_patch.patch_kv_cache_interface import (
    MomeSpec,
    ShareKVSlidingWindowSpec,
)

_CUSTOM_SPECS_REGISTERED = False


def _ensure_custom_specs_registered() -> None:
    global _CUSTOM_SPECS_REGISTERED
    if _CUSTOM_SPECS_REGISTERED:
        return
    KVCacheSpecRegistry._ensure_registered()
    KVCacheSpecRegistry.register(
        MomeSpec,
        MomeManager,
        uniform_type_base_spec=MomeSpec,
    )
    KVCacheSpecRegistry.register(
        ShareKVSlidingWindowSpec,
        ShareKVSlidingWindowManager,
        uniform_type_base_spec=SWSpec,
    )
    _CUSTOM_SPECS_REGISTERED = True


class MomeManager(MambaManager):
    """Mome KV manager: Mamba APC deferral + Mome kernel-window recycling.

    Inherits ``find_longest_cache_hit`` from ``MambaManager`` (MomeSpec is a
    ``MambaSpec`` subclass). Only ``get_num_skipped_tokens`` differs from the
    default Mamba retention model.
    """

    def __init__(
        self,
        kv_cache_spec: MomeSpec,
        block_pool: BlockPool,
        **kwargs,
    ) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self.kernel_size = kv_cache_spec.kernel_size
        self.num_extra_reserved_blocks = kv_cache_spec.num_extra_reserved_blocks

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        return 0

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        num_retained = (
            self.kernel_size
            - 1
            + self.num_extra_reserved_blocks * self.block_size
        )
        return max(0, num_computed_tokens - num_retained)


class ShareKVSlidingWindowManager(SlidingWindowManager):
    def __init__(
        self,
        kv_cache_spec: ShareKVSlidingWindowSpec,
        **kwargs,
    ) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        self.num_extra_reserved_blocks = kv_cache_spec.num_extra_reserved_blocks

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        num_retained = (
            self.sliding_window
            - 1
            + self.num_extra_reserved_blocks * self.block_size
        )
        return max(0, num_computed_tokens - num_retained)


def get_manager_for_kv_cache_spec(
    kv_cache_spec: KVCacheSpec,
    max_num_batched_tokens: int,
    max_model_len: int,
    **kwargs,
) -> SingleTypeKVCacheManager:
    _ensure_custom_specs_registered()
    manager_class = KVCacheSpecRegistry.get_manager_class(kv_cache_spec)
    assert manager_class is not None, (
        f"No manager registered for KVCacheSpec {type(kv_cache_spec)}"
    )
    if isinstance(
        kv_cache_spec,
        (SlidingWindowSpec, ChunkedLocalAttentionSpec, MomeSpec),
    ):
        kwargs["max_admission_blocks_per_request"] = (
            kv_cache_spec.max_admission_blocks_per_request(
                max_num_batched_tokens=max_num_batched_tokens,
                max_model_len=max_model_len,
            )
        )
    return manager_class(kv_cache_spec, **kwargs)


@register_patch("SingleTypeKVCacheManagerPatch", single_type_kv_cache_manager)
class SingleTypeKVCacheManagerPatch(VLLMPatch):
    _attr_names_to_apply = [
        "get_manager_for_kv_cache_spec",
        "MomeManager",
        "ShareKVSlidingWindowManager",
    ]

    get_manager_for_kv_cache_spec = get_manager_for_kv_cache_spec
    MomeManager = MomeManager
    ShareKVSlidingWindowManager = ShareKVSlidingWindowManager

